#include "Families.h"
#include "llvm/ADT/bit.h"

using namespace mlir;
using namespace mlir::darwinn;
using namespace mlir::darwinn::codegen;

namespace {

constexpr int64_t kLanes = 8;
constexpr std::array<int64_t, 4> kThreadOrder = {1, 2, 3, 0};

unsigned destinationIndex(Operation *op) {
  return isa<UnaryTensorOpOp>(op) ? 1 : 2;
}

bool sameSlicing(Operation *redistribute) {
  Operation *inner = producer(redistribute, 0);
  return inner && slicingOf(inner) == slicingOf(redistribute);
}

bool rhsPerThread(Operation *op) {
  Operation *view = throughViews(producer(op, 1));
  auto slicing = slicingOf(view);
  if (!slicing || !slicing->hasThreads())
    return false;
  Index first = threadBox(*slicing, 0).lo;
  for (int64_t thread = 1; thread < kThreads; ++thread)
    if (threadBox(*slicing, thread).lo != first)
      return true;
  return false;
}

SmallVector<std::pair<uint8_t, Operation *>> operands(Operation *op) {
  if (isa<UnaryTensorOpOp>(op))
    return {{0, producer(op, 0)}, {2, producer(op, 1)}};
  SmallVector<std::pair<uint8_t, Operation *>> out{{0, producer(op, 0)},
                                                   {2, producer(op, 2)}};
  if (innerOperation(op) == InnerOperationKind::Elementwise &&
      rhsPerThread(op) && !tensorProduct(op))
    out.push_back({1, producer(op, 1)});
  return out;
}

struct Stream {
  int64_t count;
  int64_t lhs;
  int64_t out;
  int64_t rhs;
};

SmallVector<Stream> mergeStreams(ArrayRef<Stream> loops) {
  SmallVector<Stream> out;
  for (const Stream &loop : loops) {
    if (loop.count == 1)
      continue;
    if (!out.empty()) {
      Stream &last = out.back();
      if (loop.lhs == last.count * last.lhs &&
          loop.out == last.count * last.out &&
          loop.rhs == last.count * last.rhs) {
        last.count *= loop.count;
        continue;
      }
    }
    out.push_back(loop);
  }
  return out;
}

SmallVector<Counter> streamCounters(ArrayRef<Stream> loops,
                                    int64_t Stream::*stride) {
  SmallVector<Counter> out;
  for (const Stream &loop : loops) {
    int64_t step = loop.*stride;
    out.push_back(step ? counter((loop.count - 1) * step, step)
                       : counter(loop.count - 1, 1, false));
  }
  return out;
}

std::array<uint16_t, 8> mainCounters(ArrayRef<int64_t> ends) {
  std::array<uint16_t, 8> out{};
  for (auto [index, end] : llvm::enumerate(ends))
    out[index] = static_cast<uint16_t>(end);
  return out;
}

NonLinear nonLinear(Operation *op, ActivationFunction function,
                    int64_t outputBytes) {
  NonLinear out;
  out.operation = function;
  out.activationPipelineScale = 1.0f;
  if (auto bias = immediateBias(op))
    out.immediateBias = *bias;
  auto [low, high] = clips(op);
  out.highClipValue = high;
  out.lowClipValue = low;
  out.outputType = outputBytes == 4 ? OutputType::Single : OutputType::Bfloat;
  return out;
}

Linear baseLinear(LinearOperation operation, OperandType type) {
  Linear linear;
  linear.operation = operation;
  linear.activationType = type;
  linear.parameterType = type;
  linear.partialSumRead = PartialSumRead::Initialize;
  return linear;
}

void defaultProducer(Traversal &traversal) {
  if (traversal.syncProducer.empty())
    traversal.syncProducer = {producerSync(false)};
}

} // namespace

SmallVector<unsigned> codegen::windowDims(Operation *op) {
  SmallVector<unsigned> out;
  auto traversal = op->getAttrOfType<AffineMapAttr>("traversal");
  if (!isa<UnaryTensorOpOp>(op) || !traversal ||
      !op->getAttrOfType<ArrayAttr>("custom_constraints"))
    return out;
  for (unsigned dim = 0; dim < traversal.getValue().getNumDims(); ++dim)
    if (!traversal.getValue().isFunctionOfDim(dim))
      out.push_back(dim);
  return out;
}

bool codegen::macCopy(Operation *op) {
  if (!isa<UnaryTensorOpOp>(op) || !windowDims(op).empty() ||
      linearFunction(op) != LinearFunctionKind::Add ||
      nluFunction(op) != NluFunctionKind::Linear ||
      resultInfo(throughViews(producer(op, 0))).elementBytes != 2)
    return false;
  auto traversal = op->getAttrOfType<AffineMapAttr>("traversal");
  return llvm::none_of(traversal.getValue().getResults(), [](AffineExpr expr) {
    auto constant = dyn_cast<AffineConstantExpr>(expr);
    return constant && constant.getValue() == 0;
  });
}

namespace {

FailureOr<TensorOp> unaryTensorOp(Operation *op, Context &context,
                                  int64_t thread = 0) {
  AffineMap traversalMap =
      op->getAttrOfType<AffineMapAttr>("traversal").getValue();
  Operation *source = producer(op, 0);
  auto [inStrides, inElem] = operandLayout(source);
  auto [outStrides, outElem] = operandLayout(producer(op, 1), op);
  Index size = extent(viewThreadBox(source, thread));
  int64_t rank = traversalMap.getNumDims();
  SmallVector<int64_t> reduced;
  for (auto [index, expr] : llvm::enumerate(traversalMap.getResults())) {
    auto constant = dyn_cast<AffineConstantExpr>(expr);
    if (constant && constant.getValue() == 0 && size[index] > 1)
      reduced.push_back(index);
  }
  int64_t vector = rank - 1;
  LinearFunctionKind linearKind = *linearFunction(op);
  std::optional<NluFunctionKind> nlu = nluFunction(op);
  bool floatInput = inElem == 4;
  int64_t laneLimit = reduced.empty() ? std::min(kLanes, 16 / inElem) : kLanes;
  int64_t lanes = std::min(size[vector], laneLimit);
  int64_t blocks = ceilDiv(size[vector], laneLimit);
  int64_t readBytes = std::max<int64_t>(lanes * inElem, 4);
  int64_t writeBytes = lanes * outElem;
  int64_t lastLanes = lanes;

  TensorOp tensor;
  Linear linear;
  bool hasParameters = false;
  int64_t mainLoops = 1;
  if (SmallVector<unsigned> window = windowDims(op); !window.empty()) {
    AffineMap inputMap = op->getOperand(0)
                             .getDefiningOp()
                             ->getAttrOfType<AffineMapAttr>("traversal")
                             .getValue();
    auto lhsStride = [&](unsigned dim) {
      SmallVector<int64_t> origin(rank, 0), unit(rank, 0);
      unit[dim] = 1;
      SmallVector<int64_t> from = inputMap.compose(origin),
                           to = inputMap.compose(unit);
      int64_t stride = 0;
      for (size_t axis = 0; axis < from.size(); ++axis)
        stride += (to[axis] - from[axis]) * inStrides[axis];
      return stride;
    };
    auto constraints = op->getAttrOfType<ArrayAttr>("custom_constraints");
    Index outSize = extent(viewThreadBox(producer(op, 1), thread));
    int64_t channels = outSize.back();
    lanes = std::min(channels, kLanes);
    blocks = ceilDiv(channels, kLanes);
    readBytes = lanes * inElem;
    writeBytes = lanes * outElem;
    lastLanes = lanes;
    int64_t taps = 1;
    SmallVector<Counter> windowReads;
    for (unsigned dim : llvm::reverse(window)) {
      int64_t count = cast<IntegerAttr>(constraints[dim]).getInt();
      taps *= count;
      windowReads.push_back(
          counter((count - 1) * lhsStride(dim), lhsStride(dim)));
    }
    SmallVector<Stream> pixels;
    for (int64_t index = traversalMap.getNumResults() - 2; index >= 0;
         --index) {
      unsigned dim =
          cast<AffineDimExpr>(traversalMap.getResult(index)).getPosition();
      pixels.push_back({outSize[index], lhsStride(dim), outStrides[index], 0});
    }
    pixels = mergeStreams(pixels);
    int64_t count = 1;
    for (const Stream &loop : pixels)
      count *= loop.count;
    bool mac = linearKind == LinearFunctionKind::Add;
    SmallVector<Counter> read(windowReads), write{counter(0, writeBytes)};
    auto block = [&] {
      read.push_back(counter((blocks - 1) * readBytes, readBytes));
      write.push_back(counter((blocks - 1) * writeBytes, writeBytes));
    };
    auto pixelLoops = [&] {
      for (const Stream &loop : pixels) {
        read.push_back(counter((loop.count - 1) * loop.lhs, loop.lhs));
        write.push_back(counter((loop.count - 1) * loop.out, loop.out));
      }
    };
    SmallVector<int64_t> main{taps - 1};
    if (mac) {
      main.append({blocks - 1, count - 1});
      block();
      pixelLoops();
    } else {
      main.push_back(count - 1);
      pixelLoops();
      if (blocks > 1) {
        main.push_back(blocks - 1);
        block();
      }
    }
    mainLoops = mac ? 1 : 2;
    SmallVector<Counter> loops;
    for (int64_t end : main)
      loops.push_back(counter(end, 1, false));
    tensor.mainOperation.counter = mainCounters(main);
    tensor.narrowMemoryRead.counter = padded(read, 8);
    tensor.narrowMemoryRead.byteAddressMode =
        access(readBytes, 1u << mainLoops);
    tensor.narrowMemoryWriteFromNonLinear.counter = padded(write, 8);
    tensor.narrowMemoryWriteFromNonLinear.syncProducer = {
        producerSync(true, mac ? 1 : 1 + pixels.size())};
    tensor.narrowMemoryWriteFromNonLinear.byteAddressMode = access(writeBytes);
    tensor.wideMemoryReadForSums.counter = padded(loops, 8);
    tensor.wideMemoryReadForSums.doubleBufferLoop = mac && blocks == 1 ? 2 : 1;
    tensor.wideMemoryReadForSums.secondBufferOffset = 4;
    if (mac) {
      tensor.wideMemoryReadForParameters.counter = padded(loops, 8);
      tensor.wideMemoryReadForParameters.syncProducer = {producerSync(true, 3)};
      hasParameters = true;
      tensor.wideMemoryReadForSums.baseAddress = 4;
      tensor.wideMemoryReadForSums.syncProducer = {producerSync(true, 3)};
    }
    tensor.control.partialSumReuseMap = 1;
    linear = baseLinear(mac ? LinearOperation::HighBandwidthMac
                            : LinearOperation::HighBandwidthMaximum,
                        OperandType::Bfloat);
  } else if (!reduced.empty()) {
    int64_t axis = reduced.front();
    bool mac = linearKind == LinearFunctionKind::Add && !floatInput;
    SmallVector<Stream> pixels;
    for (int64_t dim = vector - 1; dim >= 0; --dim)
      if (!llvm::is_contained(reduced, dim))
        pixels.push_back({size[dim], inStrides[dim], outStrides[dim], 0});
    pixels = mergeStreams(pixels);
    int64_t count = 1;
    for (const Stream &loop : pixels)
      count *= loop.count;
    SmallVector<int64_t> main{size[axis] - 1};
    SmallVector<Counter> read{
        counter((size[axis] - 1) * inStrides[axis], inStrides[axis])};
    SmallVector<Counter> write{counter(0, writeBytes)};
    SmallVector<Counter> sums{counter(size[axis] - 1, 1, false)};
    if (mac) {
      main.push_back(blocks - 1);
      read.push_back(counter((blocks - 1) * lanes * inElem, lanes * inElem));
      write.push_back(counter((blocks - 1) * lanes * outElem, lanes * outElem));
      sums.push_back(counter(blocks - 1, 1, false));
    }
    int64_t writeDepth =
        mac ? 1 : write.size() + (count > 1 ? pixels.size() : 0);
    mainLoops = mac ? 1 : 1 + (count > 1);
    sums.push_back(counter(count - 1, 1, false));
    if (count > 1) {
      main.push_back(count - 1);
      for (const Stream &loop : pixels) {
        read.push_back(counter((loop.count - 1) * loop.lhs, loop.lhs));
        write.push_back(counter((loop.count - 1) * loop.out, loop.out));
      }
    }
    tensor.mainOperation.counter = mainCounters(main);
    tensor.narrowMemoryRead.counter = padded(read, 8);
    tensor.narrowMemoryRead.byteAddressMode =
        access(readBytes, 1u << mainLoops);
    tensor.narrowMemoryWriteFromNonLinear.counter = padded(write, 8);
    tensor.narrowMemoryWriteFromNonLinear.syncProducer = {
        producerSync(true, writeDepth)};
    tensor.narrowMemoryWriteFromNonLinear.byteAddressMode = access(writeBytes);
    if (mac) {
      SmallVector<Counter> parameters;
      for (int64_t end : main)
        parameters.push_back(counter(end, 1, false));
      tensor.wideMemoryReadForParameters.counter = padded(parameters, 8);
      tensor.wideMemoryReadForParameters.syncProducer = {
          producerSync(true, 2 + (count > 1))};
      hasParameters = true;
      tensor.wideMemoryReadForSums.baseAddress = 4;
      tensor.wideMemoryReadForSums.syncProducer = {producerSync(true, 3)};
    }
    if (count > 1 || (mac && blocks > 1)) {
      tensor.wideMemoryReadForSums.doubleBufferLoop =
          count > 1 ? main.size() - 1 : 1;
      tensor.wideMemoryReadForSums.secondBufferOffset = 4;
    }
    tensor.wideMemoryReadForSums.counter = padded(sums, 8);
    tensor.control.partialSumReuseMap = 1;
    if (mac)
      linear =
          baseLinear(LinearOperation::HighBandwidthMac, OperandType::Bfloat);
    else if (linearKind == LinearFunctionKind::Max)
      linear = baseLinear(LinearOperation::HighBandwidthMaximum,
                          OperandType::Bfloat);
    else
      linear = baseLinear(LinearOperation::PartialSumAdd, OperandType::Half);
  } else if (macCopy(op)) {
    SmallVector<Stream> pixels;
    for (int64_t dim = vector - 1; dim >= 0; --dim) {
      Stream loop{size[dim], inStrides[dim], outStrides[dim], 0};
      if (!pixels.empty() &&
          ((dim > 0 && loop.lhs == pixels.back().count * pixels.back().lhs &&
            loop.out == pixels.back().count * pixels.back().out) ||
           (loop.count == 1 && pixels.back().count == 1)))
        pixels.back().count *= loop.count;
      else
        pixels.push_back(loop);
    }
    SmallVector<int64_t> main;
    SmallVector<Counter> read, write{counter(0, writeBytes)};
    int64_t run = 1;
    for (const Stream &loop : pixels) {
      if (loop.count > 1) {
        run *= loop.count;
      } else {
        if (run > 1)
          main.push_back(run - 1);
        main.push_back(0);
        run = 1;
      }
      read.push_back(counter((loop.count - 1) * loop.lhs, loop.lhs));
      write.push_back(counter((loop.count - 1) * loop.out, loop.out));
    }
    if (run > 1)
      main.push_back(run - 1);
    main.push_back(blocks - 1);
    if (blocks > 1) {
      read.push_back(counter((blocks - 1) * readBytes, readBytes));
      write.push_back(counter((blocks - 1) * writeBytes, writeBytes));
    }
    SmallVector<Counter> loops;
    for (int64_t end : main)
      loops.push_back(counter(end, 1, false));
    tensor.mainOperation.counter = mainCounters(main);
    tensor.narrowMemoryRead.counter = padded(read, 8);
    tensor.narrowMemoryRead.byteAddressMode =
        access(readBytes, 1u << (main.size() - 1));
    tensor.narrowMemoryWriteFromNonLinear.counter = padded(write, 8);
    tensor.narrowMemoryWriteFromNonLinear.syncProducer = {
        producerSync(true, pixels.size() + 1)};
    tensor.narrowMemoryWriteFromNonLinear.byteAddressMode = access(writeBytes);
    tensor.wideMemoryReadForParameters.counter = padded(loops, 8);
    tensor.wideMemoryReadForParameters.syncProducer = {producerSync(true, 1)};
    hasParameters = true;
    tensor.wideMemoryReadForSums.counter = padded(loops, 8);
    tensor.wideMemoryReadForSums.baseAddress = 4;
    tensor.wideMemoryReadForSums.syncProducer = {producerSync(true, 1)};
    tensor.wideMemoryReadForSums.secondBufferOffset = 4;
    mainLoops = main.size() - 1;
    linear = baseLinear(LinearOperation::HighBandwidthMac, OperandType::Bfloat);
  } else {
    SmallVector<Stream> pixels;
    for (int64_t dim = vector - 1; dim >= 0; --dim)
      pixels.push_back({size[dim], inStrides[dim], outStrides[dim], 0});
    pixels = mergeStreams(pixels);
    if (pixels.empty())
      pixels.push_back({1, blocks * readBytes, blocks * writeBytes, 0});
    int64_t count = 1;
    SmallVector<Counter> read, write{counter(0, writeBytes)};
    for (const Stream &loop : pixels) {
      count *= loop.count;
      read.push_back(counter((loop.count - 1) * loop.lhs, loop.lhs));
      write.push_back(counter((loop.count - 1) * loop.out, loop.out));
    }
    SmallVector<int64_t> main{count - 1};
    SmallVector<Counter> sums{counter(count - 1, 1, false)};
    if (blocks > 1) {
      main.push_back(blocks - 1);
      read.push_back(counter((blocks - 1) * readBytes, readBytes));
      write.push_back(counter((blocks - 1) * writeBytes, writeBytes));
      sums.push_back(counter(blocks - 1, 1, false));
    }
    tensor.mainOperation.counter = mainCounters(main);
    tensor.narrowMemoryRead.counter = padded(read, 8);
    tensor.narrowMemoryRead.byteAddressMode = access(readBytes, 2);
    if (int64_t rest = size[vector] % laneLimit; blocks > 1 && rest) {
      tensor.narrowMemoryRead.byteAddressMode =
          access(rest * inElem, readBytes, 1u << pixels.size());
      lastLanes = rest;
    }
    tensor.narrowMemoryWriteFromNonLinear.counter = padded(write, 8);
    tensor.narrowMemoryWriteFromNonLinear.syncProducer = {
        producerSync(true, pixels.size() + 1)};
    tensor.narrowMemoryWriteFromNonLinear.byteAddressMode =
        lastLanes == lanes ? access(writeBytes)
                           : access(lastLanes * outElem, writeBytes,
                                    1u << (pixels.size() + 1));
    tensor.wideMemoryReadForSums.counter = padded(sums, 8);
    tensor.wideMemoryReadForSums.secondBufferOffset = 4;
    linear = baseLinear(LinearOperation::PartialSumAdd, OperandType::Half);
  }
  if (linear.operation == LinearOperation::HighBandwidthMac)
    linear.macDisable = MacDisable::SecondFloat;
  linear.partialSumWritebackDisable = true;
  defaultProducer(tensor.narrowMemoryRead);
  defaultProducer(tensor.narrowMemoryWriteFromNonLinear);
  if (hasParameters)
    defaultProducer(tensor.wideMemoryReadForParameters);
  defaultProducer(tensor.wideMemoryReadForSums);
  bool reciprocal = nlu == NluFunctionKind::Reciprocal;
  tensor.control.linear = linear;
  tensor.control.nonLinear = nonLinear(
      op,
      reciprocal ? ActivationFunction::Reciprocal : ActivationFunction::Relu,
      outElem);
  tensor.control.nonLinear.activationPipelineScale =
      computeOptions(op).getScaleImmediate().getValueAsDouble();
  if (reciprocal) {
    tensor.control.nonLinear.symmetricFunction = true;
    tensor.control.nonLinear.evenOddFunction = true;
  }
  tensor.control.threadMulticastBitmap = bits<4>("1111");
  tensor.control.zOutBlockLoopDepth = mainLoops;
  tensor.control.lastZOutBlockValidCount = lastLanes;
  tensor.control.defaultZOutBlockValidCount = lanes;
  return tensor;
}

FailureOr<TensorOp> elementwiseTensorOp(Operation *op, Context &context) {
  Operation *lhs = producer(op, 0), *rhs = producer(op, 1),
            *destination = producer(op, 2);
  auto [lhsStrides, elem] = operandLayout(lhs);
  auto [outStrides, outElem] = operandLayout(destination, op);
  Index rhsStrides = operandLayout(rhs).first;
  SmallVector<int64_t, 4> rhsShape = resultInfo(throughViews(rhs)).shape;
  Index size = extent(viewThreadBox(destination, 0));
  int64_t rank = size.size();
  int64_t rhsRank = rhsShape.size();
  auto rhsStride = [&](int64_t dim) -> int64_t {
    int64_t index = dim - (rank - rhsRank);
    if (index < 0 || rhsShape[index] == 1)
      return 0;
    return rhsStrides[index];
  };
  int64_t blocks = ceilDiv(size.back(), kLanes);
  int64_t lanes = std::min(size.back(), kLanes);
  SmallVector<Stream> loops{{blocks, kLanes * elem, kLanes * outElem,
                             rhsStride(rank - 1) ? kLanes * elem : 0}};
  for (int64_t dim = rank - 2; dim >= 0; --dim)
    loops.push_back(
        {size[dim], lhsStrides[dim], outStrides[dim], rhsStride(dim)});
  loops = mergeStreams(loops);
  if (blocks == 1)
    loops.insert(loops.begin(), {1, kLanes * elem, kLanes * outElem,
                                 rhsStride(rank - 1) ? kLanes * elem : 0});
  LinearFunctionKind linearKind = *linearFunction(op);
  bool reduce = linearKind == LinearFunctionKind::Add ||
                linearKind == LinearFunctionKind::Sub;
  int64_t readBytes = lanes * elem;
  int64_t writeBytes = lanes * outElem;

  TensorOp tensor;
  SmallVector<int64_t> counts{loops.front().count};
  if (loops.size() > 1) {
    int64_t pixels = 1;
    for (const Stream &loop : ArrayRef(loops).drop_front())
      pixels *= loop.count;
    counts.push_back(pixels);
  }
  SmallVector<int64_t> main{0};
  for (int64_t count : counts)
    main.push_back(count - 1);
  tensor.mainOperation.counter = mainCounters(main);
  tensor.narrowMemoryRead.counter =
      padded(streamCounters(loops, &Stream::lhs), 8);
  tensor.narrowMemoryRead.byteAddressMode = access(readBytes, 2);
  SmallVector<Counter> write{counter(0, writeBytes)};
  llvm::append_range(write, streamCounters(loops, &Stream::out));
  tensor.narrowMemoryWriteFromNonLinear.counter = padded(write, 8);
  tensor.narrowMemoryWriteFromNonLinear.syncProducer = {producerSync(true, 1)};
  tensor.narrowMemoryWriteFromNonLinear.byteAddressMode = access(writeBytes);
  tensor.wideMemoryReadForParameters.counter =
      padded(streamCounters(loops, &Stream::rhs), 8);
  if (!rhsPerThread(op)) {
    FailureOr<int64_t> address = operandAddress(rhs, 0, context);
    if (failed(address))
      return unsupported(op, "an unplaced elementwise operand");
    tensor.wideMemoryReadForParameters.baseAddress = *address;
  }
  SmallVector<Counter> sums;
  if (reduce)
    sums.push_back(counter(0, 1, false));
  for (int64_t count : counts)
    sums.push_back(counter(count - 1, 1, false));
  tensor.wideMemoryReadForSums.counter = padded(sums, 8);
  if (reduce)
    tensor.wideMemoryReadForSums.doubleBufferLoop =
        counts.front() > 1 || counts.size() == 1 ? 1 : 2;
  else if (counts.front() == 1 && counts.size() > 1)
    tensor.wideMemoryReadForSums.doubleBufferLoop = 1;
  tensor.wideMemoryReadForSums.secondBufferOffset = 4;
  defaultProducer(tensor.narrowMemoryRead);
  defaultProducer(tensor.narrowMemoryWriteFromNonLinear);
  defaultProducer(tensor.wideMemoryReadForParameters);
  defaultProducer(tensor.wideMemoryReadForSums);
  Linear linear = baseLinear(reduce ? LinearOperation::ParallelAccumulate
                                    : LinearOperation::ParallelDotProduct,
                             OperandType::Bfloat);
  linear.macDisable = MacDisable::SecondFloat;
  linear.partialSumWritebackDisable = true;
  linear.broadcastSecondOperand = rhsStride(rank - 1) == 0;
  tensor.control.linear = linear;
  tensor.control.nonLinear = nonLinear(op,
                                       nluFunction(op) == NluFunctionKind::Exp
                                           ? ActivationFunction::Table
                                           : ActivationFunction::Relu,
                                       outElem);
  tensor.control.threadMulticastBitmap = bits<4>("1111");
  tensor.control.zOutBlockLoopDepth = 1;
  tensor.control.lastZOutBlockValidCount = lanes;
  tensor.control.defaultZOutBlockValidCount = lanes;
  return tensor;
}

bool firstRowSplit(Operation *op) {
  if (!isa<TensorOpOp>(op) ||
      innerOperation(op) != InnerOperationKind::Elementwise)
    return false;
  auto slicing = slicingOf(throughViews(producer(op, 2)));
  if (!slicing || slicing->domain != SmallVector<int64_t, 3>{4, 4, 4} ||
      resultInfo(throughViews(producer(op, 1))).shape.back() == 1)
    return false;
  size_t rank = slicing->begin({0, 0, 0}).size();
  for (size_t dim = 0; dim < rank; ++dim) {
    bool unit = true;
    for (int64_t row = 0; row < 4 && unit; ++row)
      for (int64_t column = 0; column < 4 && unit; ++column)
        for (int64_t thread = 0; thread < 4 && unit; ++thread)
          unit = slicing->begin({row, column, thread})[dim] ==
                 slicing->end({row, column, thread})[dim];
    if (!unit)
      continue;
    for (int64_t thread = 1; thread < kThreads; ++thread)
      if (slicing->begin({0, 0, thread})[dim] != slicing->begin({0, 0, 0})[dim])
        return true;
  }
  return false;
}

} // namespace

Operation *codegen::throughViews(Operation *op) {
  while (isa<DistributedCreateViewOp, ReshapeOpOp>(op))
    op = producer(op, 0);
  return op;
}

Operation *codegen::storage(Operation *op) {
  Operation *node = op;
  while (true) {
    if (isa<GetTensorOp, DistributedCreateViewOp, ReshapeOpOp>(node)) {
      node = producer(node, 0);
      continue;
    }
    if (isa<RedistributeOp>(node) && slicingOf(node) && producer(node, 0) &&
        isa<GetTensorOp>(producer(node, 0)) && sameSlicing(node)) {
      node = producer(node, 0);
      continue;
    }
    return node;
  }
}

namespace {

Box clamped(Box box, Operation *sliced) {
  SmallVector<int64_t, 4> shape = resultInfo(sliced).shape;
  size_t skip = box.hi.size() - shape.size();
  for (auto [dim, size] : llvm::enumerate(shape))
    box.hi[skip + dim] = std::min(box.hi[skip + dim], size - 1);
  return box;
}

} // namespace

Box codegen::ownerTileBox(Operation *owner) {
  Operation *target = owner;
  if (isa<TensorOpOp, UnaryTensorOpOp>(owner))
    target = throughViews(producer(owner, destinationIndex(owner)));
  return clamped(unionBox(*slicingOf(target), 0, 0), target);
}

Box codegen::viewThreadBox(Operation *view, int64_t thread) {
  Operation *sliced = throughViews(view);
  return clamped(threadBox(*slicingOf(sliced), thread), sliced);
}

FailureOr<int64_t> codegen::operandAddress(Operation *view, int64_t thread,
                                           Context &context, Operation *owner) {
  Operation *sliced = throughViews(view);
  if (!owner)
    owner = storage(sliced);
  FailureOr<int64_t> base = context.storageAddress(owner);
  if (failed(base))
    return failure();
  Box box = ownerTileBox(owner);
  TensorInfo info = resultInfo(owner);
  size_t rank = info.shape.size();
  Index size = tail(extent(box), rank);
  Index point = threadBox(*slicingOf(sliced), thread).lo;
  for (Operation *node = sliced; node && node != owner;
       node = producer(node, 0))
    if (auto reshape = dyn_cast<ReshapeOpOp>(node)) {
      SmallVector<int64_t> mapped =
          reshape.getReverseIndexTransformationAttr().getValue().compose(point);
      point.assign(mapped.begin(), mapped.end());
    }
  Index start = tail(point, rank);
  Index lo = tail(box.lo, rank);
  Index offset;
  for (auto [a, b] : llvm::zip(start, lo))
    offset.push_back(a - b);
  return *base + dot(offset, strides(size, info.elementBytes));
}

std::pair<Index, int64_t> codegen::operandLayout(Operation *view,
                                                 Operation *owner) {
  Operation *sliced = throughViews(view);
  if (!owner)
    owner = storage(sliced);
  Box box = ownerTileBox(owner);
  TensorInfo info = resultInfo(owner);
  Index size = tail(extent(box), info.shape.size());
  Index layout = strides(size, info.elementBytes);
  SmallVector<AffineMap> reverse;
  for (Operation *node = view; node && node != owner; node = producer(node, 0))
    if (auto reshape = dyn_cast<ReshapeOpOp>(node))
      reverse.push_back(reshape.getReverseIndexTransformationAttr().getValue());
  if (reverse.empty())
    return {layout, info.elementBytes};
  auto offset = [&](Index point) {
    for (AffineMap map : reverse)
      point = llvm::to_vector(map.compose(point));
    return dot(tail(point, layout.size()), layout);
  };
  unsigned rank = reverse.front().getNumDims();
  Index origin(rank, 0), viewStrides;
  for (unsigned dim = 0; dim < rank; ++dim) {
    Index unit(rank, 0);
    unit[dim] = 1;
    viewStrides.push_back(offset(unit) - offset(origin));
  }
  return {viewStrides, info.elementBytes};
}

std::array<bool, 16> codegen::activeTiles(Operation *op) {
  Operation *destination = throughViews(producer(op, destinationIndex(op)));
  SmallVector<int64_t, 4> shape = resultInfo(destination).shape;
  std::array<bool, 16> out{};
  for (const TileBox &entry : tiles(*slicingOf(destination)))
    out[entry.tile] |=
        llvm::all_of_zip(tail(entry.box.lo, shape.size()), shape,
                         [](int64_t low, int64_t size) { return low < size; });
  return out;
}

FailureOr<SmallVector<Emitted, 0>>
codegen::registers(Operation *op, Context &context, ArrayRef<int64_t> threads) {
  std::array<bool, 16> tiles = activeTiles(op);
  SmallVector<Emitted, 0> out;
  for (int64_t thread : threads) {
    for (auto [reg, view] : operands(op)) {
      FailureOr<int64_t> address =
          operandAddress(view, thread, context, reg == 2 ? op : nullptr);
      if (failed(address))
        return unsupported(op, "an unplaced compute operand");
      out.push_back(load(*address, tiles, reg, threadBit(thread)));
    }
  }
  return out;
}

FailureOr<SmallVector<Emitted, 0>> codegen::registers(Operation *op,
                                                      Context &context) {
  return registers(op, context, kThreadOrder);
}

std::pair<float, float> codegen::clips(Operation *op) {
  ComputeOpOptionsAttr options = computeOptions(op);
  float low = -std::numeric_limits<float>::infinity();
  float high = std::numeric_limits<float>::infinity();
  if (auto lower = options.getClipLower())
    low = lower.getValueAsDouble();
  if (auto upper = options.getClipUpper())
    high = upper.getValueAsDouble();
  return {low, high};
}

std::optional<int32_t> codegen::immediateBias(Operation *op) {
  auto bias = computeOptions(op).getBiasImmediate();
  if (!bias)
    return std::nullopt;
  int32_t bits =
      llvm::bit_cast<int32_t>(static_cast<float>(bias.getValueAsDouble()));
  if (bits == 0)
    return std::nullopt;
  return bits;
}

FailureOr<Emitted> codegen::coefficientTables(Operation *op,
                                              NluFunctionKind function,
                                              const Context &context) {
  auto found = context.nluTables.find(function);
  if (found == context.nluTables.end())
    return unsupported(op, "an NLU function without a supplied spline");
  CoefficientTables tables = found->second;
  tables.threadMulticastBitmap = bits<4>("1111");
  ComputePacket packet;
  packet.header = tileHeader(activeTiles(op));
  packet.instruction = tables;
  return Emitted{packet, Role::Plain};
}

Body codegen::initialization(Operation *op, Context &context) {
  bool subtract = linearFunction(op) == LinearFunctionKind::Sub;
  std::array<uint8_t, 16> pattern{};
  pattern[0] = 0x80;
  pattern[1] = 0x3F;
  pattern[2] = 0x80;
  pattern[3] = subtract ? 0xBF : 0x3F;
  std::array<bool, 16> tiles = activeTiles(op);
  FailureOr<int64_t> scratch = context.narrowAddress(op, Suffix::Init);
  FailureOr<int64_t> wide = context.wideAddress(op, Suffix::Init);
  if (failed(scratch) || failed(wide))
    return unsupported(op, "an unplaced initialization scratch");
  int64_t bytes =
      4 * std::min(extent(viewThreadBox(producer(op, 0), 0)).back(), kLanes);
  Mesh mesh;
  mesh.direction = MeshDirection::OutboundEastInboundWest;
  mesh.write.baseAddress = *scratch;
  mesh.write.counter = padded({counter(bytes - 16, 16)}, 5);
  mesh.write.syncProducer = {producerSync(true, 1)};
  mesh.write.byteAddressMode = access(16);
  mesh.immediateValue = pattern;
  mesh.validBytes = 4;
  NarrowToWide copy;
  copy.read.baseAddress = *scratch;
  copy.read.counter = padded({counter(0, bytes)}, 6);
  copy.read.byteAddressMode = access(bytes);
  copy.write.baseAddress = *wide;
  copy.write.counter = padded({counter(0, 1)}, 4);
  copy.write.syncProducer = {producerSync(true)};
  copy.write.byteAddressMode = access(bytes);
  copy.readWatchers = {dmaWatcher(tileWatcher(TileSyncFlag::MeshInboundFromWest,
                                              1, 1, bytes == 16 ? 1 : 0))};
  copy.threadMulticastBitmap = bits<4>("1111");
  copy.byteAddress.strideUnitGranulesLoopMap = 1;
  copy.byteAddress.defaultStrideUnitGranules = 4;
  copy.byteAddress.lastStrideUnitGranules = 4;
  copy.byteAddress.cellStride = bytes / 4;
  copy.byteAddress.defaultCellStrideGroupCount = 1;
  copy.byteAddress.lastCellStrideGroupCount = 1;
  return SmallVector<Emitted, 0>{dma(mesh, tiles), dma(copy, tiles)};
}

Body codegen::unary(Operation *op, Context &context) {
  SmallVector<Emitted, 0> out;
  if (nluFunction(op) == NluFunctionKind::Reciprocal) {
    FailureOr<Emitted> tables =
        coefficientTables(op, NluFunctionKind::Reciprocal, context);
    if (failed(tables))
      return failure();
    out.push_back(*tables);
  }
  Index common = extent(viewThreadBox(producer(op, 1), 0));
  SmallVector<int64_t> shared, odd;
  for (int64_t thread : kThreadOrder) {
    Index size = extent(viewThreadBox(producer(op, 1), thread));
    bool empty = llvm::any_of(size, [](int64_t count) { return count <= 0; });
    (size == common || empty ? shared : odd).push_back(thread);
  }
  FailureOr<SmallVector<Emitted, 0>> loads = registers(op, context, shared);
  FailureOr<TensorOp> tensorOp = unaryTensorOp(op, context);
  if (failed(loads) || failed(tensorOp))
    return failure();
  std::array<bool, 4> threads{};
  for (int64_t thread : shared)
    threads[thread] = true;
  tensorOp->control.threadMulticastBitmap = threads;
  llvm::append_range(out, *loads);
  out.push_back(tensor(*tensorOp, activeTiles(op), bits<8>("10100000")));
  for (int64_t thread : odd) {
    FailureOr<TensorOp> single = unaryTensorOp(op, context, thread);
    FailureOr<int64_t> lhs = operandAddress(producer(op, 0), thread, context);
    FailureOr<int64_t> result =
        operandAddress(producer(op, 1), thread, context, op);
    if (failed(single) || failed(lhs) || failed(result))
      return unsupported(op, "an unplaced unary operand");
    single->narrowMemoryRead.baseAddress = *lhs;
    single->narrowMemoryWriteFromNonLinear.baseAddress = *result;
    single->control.threadMulticastBitmap = threadBit(thread);
    out.push_back(tensor(*single, activeTiles(op), {}));
  }
  return out;
}

namespace {

struct ProductPlan {
  int64_t units;
  int64_t rows;
  int64_t chunks;
  int64_t rowBytes;
  int64_t outRowBytes;
  int64_t outElem;
};

constexpr int64_t kProductUnit = 16;
constexpr int64_t kProductChunk = 256;
constexpr int64_t kProductFifo = 32;

FailureOr<ProductPlan> productPlan(Operation *op) {
  Operation *lhs = producer(op, 0), *destination = producer(op, 2);
  auto [lhsStrides, elem] = operandLayout(lhs);
  auto [outStrides, outElem] = operandLayout(destination, op);
  Index size = extent(viewThreadBox(destination, 0));
  if (size.size() != 4 || size[0] != 1 || elem != 2)
    return failure();
  int64_t segment = size[2] * size[3] * elem;
  if (lhsStrides[2] != size[3] * elem || segment % kProductUnit ||
      (segment > kProductChunk && segment % kProductChunk))
    return failure();
  int64_t run = size[1] == 1 ? segment : std::min(segment, kProductChunk);
  return ProductPlan{run / kProductUnit, size[1],       segment / run,
                     lhsStrides[1],      outStrides[1], outElem};
}

} // namespace

bool codegen::tensorProduct(Operation *op) {
  return isa<TensorOpOp>(op) &&
         innerOperation(op) == InnerOperationKind::Elementwise &&
         linearFunction(op) == LinearFunctionKind::Mac && rhsPerThread(op) &&
         resultInfo(throughViews(producer(op, 1))).shape ==
             resultInfo(throughViews(producer(op, 2))).shape;
}

int64_t codegen::tensorProductRows(Operation *op) {
  if (!tensorProduct(op))
    return 1;
  FailureOr<ProductPlan> plan = productPlan(op);
  if (failed(plan))
    return 1;
  return plan->rows == 1 ? std::min(plan->units, kProductFifo) + 1
                         : 2 * plan->units + 1;
}

static Body tensorProductOp(Operation *op, Context &context) {
  FailureOr<ProductPlan> planned = productPlan(op);
  FailureOr<int64_t> wide = context.wideAddress(op);
  if (failed(planned) || failed(wide))
    return unsupported(op, "an activation product without a row plan");
  const ProductPlan &p = *planned;
  bool fifo = p.rows == 1;
  int64_t buffer = fifo ? std::min(p.units, kProductFifo) : p.units;
  int64_t passes = p.units / buffer;
  auto loops = [&](int64_t inner, int64_t step) {
    SmallVector<Counter> items{counter((p.units - 1) * inner, inner)};
    if (p.rows > 1)
      items.push_back(counter((p.rows - 1) * step, step));
    if (p.chunks > 1)
      items.push_back(
          counter((p.chunks - 1) * kProductChunk * inner / kProductUnit,
                  kProductChunk * inner / kProductUnit));
    return items;
  };
  auto unmasked = [&](SmallVector<Counter> items) {
    if (passes > 1)
      items.push_back(counter(passes - 1, 1, false));
    if (p.rows > 1)
      items.push_back(counter(p.rows - 1, 1, false));
    if (p.chunks > 1)
      items.push_back(counter(p.chunks - 1, 1, false));
    return items;
  };

  NarrowToWide copy;
  copy.read.counter = padded(loops(kProductUnit, p.rowBytes), 6);
  copy.read.byteAddressMode = access(kProductUnit);
  copy.write.baseAddress = *wide;
  copy.write.counter = padded(unmasked({counter(buffer - 1, 1)}), 4);
  copy.write.syncProducer = {producerSync(true, !fifo)};
  copy.write.byteAddressMode = access(2);
  if (!fifo) {
    copy.write.doubleBufferLoop = 1;
    copy.write.secondBufferOffset = p.units;
  }
  copy.writeWatchers = {dmaWatcher(
      tileWatcher(TileSyncFlag::ParameterRead,
                  (int32_t(1) << 25) - (fifo ? buffer - 1 : 1), 1, !fifo, true),
      true)};
  copy.transpose = true;
  copy.byteAddress.strideUnitGranulesLoopMap = 1;
  copy.byteAddress.defaultStrideUnitGranules = 2;
  copy.byteAddress.lastStrideUnitGranules = 2;
  copy.byteAddress.cellStride = 1;
  copy.byteAddress.cellStrideGroupCountLoopMap = 1;
  copy.byteAddress.defaultCellStrideGroupCount = 8;
  copy.byteAddress.lastCellStrideGroupCount = 8;

  TensorOp product;
  SmallVector<int64_t> main{0, p.units - 1};
  if (p.rows > 1)
    main.push_back(p.rows - 1);
  if (p.chunks > 1)
    main.push_back(p.chunks - 1);
  product.mainOperation.counter = mainCounters(main);
  product.narrowMemoryRead.counter = padded(loops(kProductUnit, p.rowBytes), 8);
  product.narrowMemoryRead.byteAddressMode =
      access(kProductUnit, p.chunks > 1 ? 8 : 2);
  int64_t outUnit = kLanes * p.outElem;
  SmallVector<Counter> write{counter(0, outUnit)};
  llvm::append_range(write, loops(outUnit, p.outRowBytes));
  product.narrowMemoryWriteFromNonLinear.counter = padded(write, 8);
  product.narrowMemoryWriteFromNonLinear.syncProducer = {producerSync(true, 1)};
  product.narrowMemoryWriteFromNonLinear.byteAddressMode = access(outUnit);
  Traversal &parameters = product.wideMemoryReadForParameters;
  parameters.baseAddress = *wide * 4;
  parameters.counter = padded(unmasked({counter((buffer - 1) * 4, 4)}), 8);
  parameters.syncProducer = {producerSync(true, !fifo)};
  if (!fifo) {
    parameters.doubleBufferLoop = 1;
    parameters.secondBufferOffset = p.units * 4;
  }
  Traversal &sums = product.wideMemoryReadForSums;
  sums.baseAddress = (*wide + (fifo ? buffer : 2 * p.units)) * 4;
  sums.counter =
      fifo ? padded({counter(p.units - 1, 1, false)}, 8)
           : padded({counter(p.units - 1, 1, false), counter(0, 1, false),
                     counter(p.rows * p.chunks - 1, 1, false)},
                    8);
  sums.syncProducer = {producerSync(true, fifo ? 0 : 2)};
  sums.secondBufferOffset = 4;
  defaultProducer(product.narrowMemoryRead);
  product.syncWatchers.push_back(tileWatcher(TileSyncFlag::NarrowToWideWrite, 1,
                                             1, fifo || p.rows == 1 ? 1 : 2,
                                             true));
  Linear linear =
      baseLinear(LinearOperation::HighBandwidthMac, OperandType::Bfloat);
  linear.macDisable = MacDisable::SecondFloat;
  linear.partialSumWritebackDisable = true;
  product.control.linear = linear;
  product.control.nonLinear =
      nonLinear(op, ActivationFunction::Relu, p.outElem);
  product.control.zOutBlockLoopDepth = 1;
  product.control.lastZOutBlockValidCount = kLanes;
  product.control.defaultZOutBlockValidCount = kLanes;

  auto rhsLoads = [&](ArrayRef<int64_t> threads) -> Body {
    SmallVector<Emitted, 0> out;
    for (int64_t thread : threads) {
      FailureOr<int64_t> address =
          operandAddress(producer(op, 1), thread, context);
      if (failed(address))
        return unsupported(op, "an unplaced activation product operand");
      out.push_back(load(*address, activeTiles(op), 0, threadBit(thread)));
    }
    return out;
  };
  SmallVector<Emitted, 0> out;
  if (!firstRowSplit(op)) {
    Body loads = rhsLoads(kThreadOrder);
    FailureOr<SmallVector<Emitted, 0>> registersLoads = registers(op, context);
    if (failed(loads) || failed(registersLoads))
      return failure();
    copy.threadMulticastBitmap = bits<4>("1111");
    product.control.threadMulticastBitmap = bits<4>("1111");
    out = std::move(*loads);
    out.push_back(dma(copy, activeTiles(op), bits<8>("10000000")));
    llvm::append_range(out, *registersLoads);
    out.push_back(tensor(product, activeTiles(op), bits<8>("10100000")));
    return out;
  }

  NarrowToWide singleCopy = copy;
  TensorOp single = product;
  singleCopy.threadMulticastBitmap = bits<4>("1000");
  single.control.threadMulticastBitmap = bits<4>("1000");
  FailureOr<int64_t> rhs = operandAddress(producer(op, 1), 0, context);
  FailureOr<int64_t> lhs = operandAddress(producer(op, 0), 0, context);
  FailureOr<int64_t> result = operandAddress(producer(op, 2), 0, context, op);
  if (failed(rhs) || failed(lhs) || failed(result))
    return unsupported(op, "an unplaced activation product operand");
  singleCopy.read.baseAddress = *rhs;
  single.narrowMemoryRead.baseAddress = *lhs;
  single.narrowMemoryWriteFromNonLinear.baseAddress = *result;
  SmallVector<int64_t> threads{kThreadOrder[1], kThreadOrder[2],
                               kThreadOrder[0]};
  Body loads = rhsLoads(threads);
  FailureOr<SmallVector<Emitted, 0>> registersLoads =
      registers(op, context, threads);
  if (failed(loads) || failed(registersLoads))
    return failure();
  copy.threadMulticastBitmap = bits<4>("0111");
  product.control.threadMulticastBitmap = bits<4>("0111");
  std::array<bool, 16> rest = bits<16>("0111111111111111");
  out.push_back(dma(singleCopy, tileBit(0)));
  out.push_back(tensor(single, tileBit(0), {}));
  llvm::append_range(out, *loads);
  out.push_back(dma(copy, activeTiles(op), bits<8>("10000000")));
  llvm::append_range(out, *registersLoads);
  out.push_back(tensor(product, activeTiles(op), bits<8>("10100000")));
  out.push_back(dma(singleCopy, rest));
  out.push_back(tensor(single, rest, {}));
  return out;
}

Body codegen::elementwise(Operation *op, Context &context) {
  if (tensorProduct(op))
    return tensorProductOp(op, context);
  SmallVector<Emitted, 0> out;
  if (nluFunction(op) == NluFunctionKind::Exp) {
    FailureOr<Emitted> tables =
        coefficientTables(op, NluFunctionKind::Exp, context);
    if (failed(tables))
      return failure();
    out.push_back(*tables);
  }
  FailureOr<TensorOp> shared = elementwiseTensorOp(op, context);
  if (failed(shared))
    return failure();
  std::array<bool, 8> bitmap =
      rhsPerThread(op) ? bits<8>("11100000") : bits<8>("10100000");
  if (!firstRowSplit(op)) {
    FailureOr<SmallVector<Emitted, 0>> loads = registers(op, context);
    if (failed(loads))
      return failure();
    llvm::append_range(out, *loads);
    out.push_back(tensor(*shared, activeTiles(op), bitmap));
    return out;
  }
  TensorOp single = *shared;
  single.control.threadMulticastBitmap = bits<4>("1000");
  struct Field {
    Traversal TensorOp::*traversal;
    unsigned operand;
    bool owned;
  };
  for (Field field :
       {Field{&TensorOp::narrowMemoryRead, 0, false},
        Field{&TensorOp::wideMemoryReadForParameters, 1, false},
        Field{&TensorOp::narrowMemoryWriteFromNonLinear, 2, true}}) {
    FailureOr<int64_t> base = operandAddress(
        producer(op, field.operand), 0, context, field.owned ? op : nullptr);
    if (failed(base))
      return unsupported(op, "an unplaced elementwise operand");
    (single.*field.traversal).baseAddress = *base;
  }
  TensorOp threads = *shared;
  threads.control.threadMulticastBitmap = bits<4>("0111");
  FailureOr<SmallVector<Emitted, 0>> loads = registers(
      op, context, {kThreadOrder[1], kThreadOrder[2], kThreadOrder[0]});
  if (failed(loads))
    return failure();
  out.push_back(tensor(single, tileBit(0), {}));
  llvm::append_range(out, *loads);
  out.push_back(tensor(threads, activeTiles(op), bitmap));
  out.push_back(tensor(single, bits<16>("0111111111111111"), {}));
  return out;
}
