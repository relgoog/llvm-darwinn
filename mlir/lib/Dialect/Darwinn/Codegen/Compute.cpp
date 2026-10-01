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

FailureOr<TensorOp> unaryTensorOp(Operation *op, Context &context) {
  AffineMap traversalMap =
      op->getAttrOfType<AffineMapAttr>("traversal").getValue();
  Operation *source = producer(op, 0);
  auto [inStrides, inElem] = operandLayout(source);
  auto [outStrides, outElem] = operandLayout(producer(op, 1), op);
  Index size = extent(viewThreadBox(source, 0));
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

  TensorOp tensor;
  Linear linear;
  bool hasParameters = false;
  if (!reduced.empty()) {
    int64_t axis = reduced.front();
    bool mac = linearKind == LinearFunctionKind::Add && !floatInput;
    SmallVector<int64_t> main{size[axis] - 1};
    SmallVector<Counter> read{
        counter((size[axis] - 1) * inStrides[axis], inStrides[axis])};
    SmallVector<Counter> write{counter(0, writeBytes)};
    SmallVector<Counter> sums{counter(size[axis] - 1, 1, false)};
    if (mac) {
      main.push_back(blocks - 1);
      read.push_back(counter((blocks - 1) * kLanes * inElem, kLanes * inElem));
      write.push_back(
          counter((blocks - 1) * kLanes * outElem, kLanes * outElem));
      sums.push_back(counter(blocks - 1, 1, false));
    }
    tensor.mainOperation.counter = mainCounters(main);
    tensor.narrowMemoryRead.counter = padded(read, 8);
    tensor.narrowMemoryRead.byteAddressMode = access(readBytes, 2);
    tensor.narrowMemoryWriteFromNonLinear.counter = padded(write, 8);
    tensor.narrowMemoryWriteFromNonLinear.syncProducer = {
        producerSync(true, 1)};
    tensor.narrowMemoryWriteFromNonLinear.byteAddressMode = access(writeBytes);
    if (mac) {
      SmallVector<Counter> parameters;
      for (int64_t end : main)
        parameters.push_back(counter(end, 1, false));
      tensor.wideMemoryReadForParameters.counter = padded(parameters, 8);
      tensor.wideMemoryReadForParameters.syncProducer = {producerSync(true, 2)};
      hasParameters = true;
      tensor.wideMemoryReadForSums.baseAddress = 4;
      tensor.wideMemoryReadForSums.syncProducer = {producerSync(true, 3)};
      if (blocks > 1) {
        tensor.wideMemoryReadForSums.doubleBufferLoop = 1;
        tensor.wideMemoryReadForSums.secondBufferOffset = 4;
      }
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
    tensor.narrowMemoryWriteFromNonLinear.counter = padded(write, 8);
    tensor.narrowMemoryWriteFromNonLinear.syncProducer = {
        producerSync(true, pixels.size() + 1)};
    tensor.narrowMemoryWriteFromNonLinear.byteAddressMode = access(writeBytes);
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
  if (reciprocal) {
    tensor.control.nonLinear.symmetricFunction = true;
    tensor.control.nonLinear.evenOddFunction = true;
  }
  tensor.control.threadMulticastBitmap = bits<4>("1111");
  tensor.control.zOutBlockLoopDepth = 1;
  tensor.control.lastZOutBlockValidCount = lanes;
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
  LinearFunctionKind linearKind = *linearFunction(op);
  bool reduce = linearKind == LinearFunctionKind::Add ||
                linearKind == LinearFunctionKind::Sub;
  int64_t readBytes = lanes * elem;
  int64_t writeBytes = lanes * outElem;

  TensorOp tensor;
  SmallVector<int64_t> main{0};
  for (const Stream &loop : loops)
    main.push_back(loop.count - 1);
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
  for (const Stream &loop : loops)
    sums.push_back(counter(loop.count - 1, 1, false));
  tensor.wideMemoryReadForSums.counter = padded(sums, 8);
  if (reduce)
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
  if (!slicing || slicing->domain != SmallVector<int64_t, 3>{4, 4, 4})
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

Box codegen::ownerTileBox(Operation *owner) {
  Operation *target = owner;
  if (isa<TensorOpOp, UnaryTensorOpOp>(owner))
    target = throughViews(producer(owner, destinationIndex(owner)));
  return unionBox(*slicingOf(target), 0, 0);
}

Box codegen::viewThreadBox(Operation *view, int64_t thread) {
  return threadBox(*slicingOf(throughViews(view)), thread);
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
  Index start = tail(threadBox(*slicingOf(sliced), thread).lo, rank);
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
  return {strides(size, info.elementBytes), info.elementBytes};
}

std::array<bool, 16> codegen::activeTiles(Operation *op) {
  auto slicing = slicingOf(throughViews(producer(op, destinationIndex(op))));
  std::array<bool, 16> out{};
  for (int64_t tile = 0; tile < kTiles; ++tile)
    out[tile] =
        tile / kGrid < slicing->domain[0] && tile % kGrid < slicing->domain[1];
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
  Mesh mesh;
  mesh.direction = MeshDirection::OutboundEastInboundWest;
  mesh.write.baseAddress = *scratch;
  mesh.write.counter = padded({counter(16, 16)}, 5);
  mesh.write.syncProducer = {producerSync(true, 1)};
  mesh.write.byteAddressMode = access(16);
  mesh.immediateValue = pattern;
  mesh.validBytes = 4;
  NarrowToWide copy;
  copy.read.baseAddress = *scratch;
  copy.read.counter = padded({counter(0, 32)}, 6);
  copy.read.byteAddressMode = access(32);
  copy.write.baseAddress = *wide;
  copy.write.counter = padded({counter(0, 1)}, 4);
  copy.write.syncProducer = {producerSync(true)};
  copy.write.byteAddressMode = access(32);
  copy.readWatchers = {
      dmaWatcher(tileWatcher(TileSyncFlag::MeshInboundFromWest, 1, 1))};
  copy.threadMulticastBitmap = bits<4>("1111");
  copy.byteAddress.strideUnitGranulesLoopMap = 1;
  copy.byteAddress.defaultStrideUnitGranules = 4;
  copy.byteAddress.lastStrideUnitGranules = 4;
  copy.byteAddress.cellStride = 8;
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
  FailureOr<SmallVector<Emitted, 0>> loads = registers(op, context);
  FailureOr<TensorOp> tensorOp = unaryTensorOp(op, context);
  if (failed(loads) || failed(tensorOp))
    return failure();
  llvm::append_range(out, *loads);
  out.push_back(tensor(*tensorOp, activeTiles(op), bits<8>("10100000")));
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
  int64_t run = std::min(segment, kProductChunk);
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
  return succeeded(plan) ? 2 * plan->units + 1 : 1;
}

static Body tensorProductOp(Operation *op, Context &context) {
  if (firstRowSplit(op))
    return unsupported(op, "an activation product split across a first row");
  FailureOr<ProductPlan> planned = productPlan(op);
  FailureOr<int64_t> wide = context.wideAddress(op);
  if (failed(planned) || failed(wide))
    return unsupported(op, "an activation product without a row plan");
  const ProductPlan &p = *planned;
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
    if (p.rows > 1)
      items.push_back(counter(p.rows - 1, 1, false));
    if (p.chunks > 1)
      items.push_back(counter(p.chunks - 1, 1, false));
    return items;
  };

  SmallVector<Emitted, 0> out;
  for (int64_t thread : kThreadOrder) {
    FailureOr<int64_t> address =
        operandAddress(producer(op, 1), thread, context);
    if (failed(address))
      return unsupported(op, "an unplaced activation product operand");
    out.push_back(load(*address, activeTiles(op), 0, threadBit(thread)));
  }
  NarrowToWide copy;
  copy.read.counter = padded(loops(kProductUnit, p.rowBytes), 6);
  copy.read.byteAddressMode = access(kProductUnit);
  copy.write.baseAddress = *wide;
  copy.write.counter = padded(unmasked({counter(p.units - 1, 1)}), 4);
  copy.write.syncProducer = {producerSync(true, 1)};
  copy.write.byteAddressMode = access(2);
  if (p.rows > 1) {
    copy.write.doubleBufferLoop = 1;
    copy.write.secondBufferOffset = p.units;
  }
  copy.writeWatchers = {
      dmaWatcher(tileWatcher(TileSyncFlag::ParameterRead,
                             (int32_t(1) << 25) - 1, 1, 1, true),
                 true)};
  copy.transpose = true;
  copy.threadMulticastBitmap = bits<4>("1111");
  copy.byteAddress.strideUnitGranulesLoopMap = 1;
  copy.byteAddress.defaultStrideUnitGranules = 2;
  copy.byteAddress.lastStrideUnitGranules = 2;
  copy.byteAddress.cellStride = 1;
  copy.byteAddress.cellStrideGroupCountLoopMap = 1;
  copy.byteAddress.defaultCellStrideGroupCount = 8;
  copy.byteAddress.lastCellStrideGroupCount = 8;
  out.push_back(dma(copy, activeTiles(op), bits<8>("10000000")));

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
  parameters.counter = padded(unmasked({counter((p.units - 1) * 4, 4)}), 8);
  parameters.syncProducer = {producerSync(true, 1)};
  if (p.rows > 1) {
    parameters.doubleBufferLoop = 1;
    parameters.secondBufferOffset = p.units * 4;
  }
  Traversal &sums = product.wideMemoryReadForSums;
  sums.baseAddress = (*wide + 2 * p.units) * 4;
  sums.counter = padded({counter(p.units - 1, 1, false), counter(0, 1, false),
                         counter(p.rows * p.chunks - 1, 1, false)},
                        8);
  sums.syncProducer = {producerSync(true, 2)};
  sums.secondBufferOffset = 4;
  defaultProducer(product.narrowMemoryRead);
  product.syncWatchers.push_back(tileWatcher(TileSyncFlag::NarrowToWideWrite, 1,
                                             1, p.rows > 1 ? 2 : 1, true));
  Linear linear =
      baseLinear(LinearOperation::HighBandwidthMac, OperandType::Bfloat);
  linear.macDisable = MacDisable::SecondFloat;
  linear.partialSumWritebackDisable = true;
  product.control.linear = linear;
  product.control.nonLinear =
      nonLinear(op, ActivationFunction::Relu, p.outElem);
  product.control.threadMulticastBitmap = bits<4>("1111");
  product.control.zOutBlockLoopDepth = 1;
  product.control.lastZOutBlockValidCount = kLanes;
  product.control.defaultZOutBlockValidCount = kLanes;
  FailureOr<SmallVector<Emitted, 0>> loads = registers(op, context);
  if (failed(loads))
    return failure();
  llvm::append_range(out, *loads);
  out.push_back(tensor(product, activeTiles(op), bits<8>("10100000")));
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
