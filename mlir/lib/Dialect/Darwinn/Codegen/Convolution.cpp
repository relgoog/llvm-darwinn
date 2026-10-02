#include "Families.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/bit.h"

using namespace mlir;
using namespace mlir::darwinn;
using namespace mlir::darwinn::codegen;

namespace {

constexpr int64_t kWideRows = 64;
constexpr int64_t kOutLanes = 32;
constexpr int64_t kPartialSumPixels = 32;
constexpr int64_t kSingleRows = 32;
constexpr int64_t kGroupedPairs = 8;
constexpr int64_t kStencilLanes = 8;
constexpr int32_t kWatcherBase = (int32_t(1) << 25) - 1;

bool isTransposed(Operation *op) {
  return computeOptions(op).getComputeTypeHint() ==
         ComputeTypeHintKind::TransposedConv;
}

struct Geometry {
  int64_t row;
  int64_t col;
  int64_t elem;
  int64_t stride;
};

Geometry lhsGeometry(Operation *op) {
  Operation *lhs = producer(op, 0);
  auto [layout, elem] = operandLayout(lhs);
  int64_t stride = 1;
  AffineMap traversal =
      lhs->getAttrOfType<AffineMapAttr>("traversal").getValue();
  if (auto add = dyn_cast<AffineBinaryOpExpr>(traversal.getResult(1));
      add && add.getKind() == AffineExprKind::Add) {
    auto mul = dyn_cast<AffineBinaryOpExpr>(add.getLHS());
    if (mul && mul.getKind() == AffineExprKind::Mul &&
        isa<AffineDimExpr>(add.getRHS())) {
      auto dim = dyn_cast<AffineDimExpr>(mul.getLHS());
      auto factor = dyn_cast<AffineConstantExpr>(mul.getRHS());
      if (dim && dim.getPosition() == 1 && factor)
        stride = factor.getValue();
    }
  }
  return Geometry{layout[1], layout[2], elem, stride};
}

std::pair<int64_t, int64_t> threadPixels(Operation *op) {
  Index size = extent(viewThreadBox(producer(op, 2), 0));
  return {size[1], size[2]};
}

struct Strided {
  int64_t count;
  int64_t lhs;
  int64_t out;
};

SmallVector<std::pair<int64_t, int64_t>> mergeStream(ArrayRef<Strided> loops,
                                                     int64_t Strided::*stream) {
  SmallVector<std::pair<int64_t, int64_t>> out;
  for (const Strided &loop : loops) {
    if (loop.count == 1)
      continue;
    int64_t step = loop.*stream;
    if (!out.empty() && step && out.back().second &&
        step == out.back().first * out.back().second)
      out.back().first *= loop.count;
    else
      out.push_back({loop.count, step});
  }
  return out;
}

SmallVector<Counter> masked(ArrayRef<std::pair<int64_t, int64_t>> loops) {
  SmallVector<Counter> out;
  for (auto [count, step] : loops)
    out.push_back(step ? counter((count - 1) * step, step)
                       : counter(count - 1, 1, false));
  return out;
}

int64_t countOf(ArrayRef<Strided> loops) {
  int64_t out = 1;
  for (const Strided &loop : loops)
    out *= loop.count;
  return out;
}

std::array<uint16_t, 8> mainCounters(ArrayRef<int64_t> counts) {
  std::array<uint16_t, 8> out{};
  for (auto [index, count] : llvm::enumerate(counts))
    out[index] = static_cast<uint16_t>(count - 1);
  return out;
}

FailureOr<int32_t> scalarBiasBits(Operation *op) {
  Operation *node = producer(op, 3);
  while (node && !isa<FillOp>(node))
    node = producer(node, 0);
  if (!node)
    return failure();
  auto value = dyn_cast<DenseElementsAttr>(cast<FillOp>(node).getValue());
  if (!value || !value.getElementType().isF32())
    return failure();
  float bias = *value.getValues<float>().begin();
  return llvm::bit_cast<int32_t>(bias);
}

std::pair<int64_t, int64_t> blockBytes(const VmcPlan &plan) {
  int64_t weights = plan.taps * plan.chunks * plan.chunk * plan.lanesPadded * 2;
  int64_t bias = plan.biasRows ? plan.lanesPadded * 4 : 0;
  return {bias, weights};
}

int64_t parameterBytes(const VmcPlan &plan) {
  auto [bias, weights] = blockBytes(plan);
  return (bias + weights) * plan.outBlocks;
}

std::array<bool, 8> parameterChannels(Operation *op) {
  return activeTiles(op) == bits<16>("1111111111111111") ? bits<8>("01000000")
                                                         : bits<8>("00000001");
}

SmallVector<Emitted, 0> parameterInfeed(Operation *op, const VmcPlan &plan) {
  int64_t total = parameterBytes(plan);
  std::array<bool, 8> channels = parameterChannels(op);
  std::array<bool, 17> everyTile = targets(activeTiles(op), false);
  if (plan.outer == 1 || plan.single)
    return infeed(total, channels, everyTile, InputFifo::Parameter);
  auto [bias, weights] = blockBytes(plan);
  int64_t block = (bias + weights) / kAccess;
  int64_t blocks = plan.outBlocks;
  auto blockLoop = [&](SmallVector<Counter> &items, Traversal &traversal) {
    if (blocks == 2) {
      items.push_back(counter(1, 1, false));
      traversal.doubleBufferLoop = items.size() - 1;
      traversal.secondBufferOffset = block;
    } else if (blocks > 2) {
      items.push_back(counter((blocks - 1) * block, block));
    }
  };
  PopInput pop;
  pop.fifo = InputFifo::Parameter;
  SmallVector<Counter> popItems{counter(block - 1, 1)};
  blockLoop(popItems, pop.traversal);
  pop.traversal.counter = padded(popItems, 5);
  pop.traversal.syncProducer = {producerSync(true, 1)};
  if (blocks > 1)
    pop.watcher = ScalarSyncWatcher{
        ScalarSyncFlag::ParameterInfeed, 1,
        static_cast<uint32_t>((int64_t(1) << 25) - (blocks - 1)), 1, true};
  RingInfeed ring;
  ring.fifo = InputFifo::Parameter;
  SmallVector<Counter> ringItems{counter(weights / kAccess - 1, 1),
                                 counter(plan.outer - 1, 1, false)};
  blockLoop(ringItems, ring.traversal);
  ring.traversal.baseAddress = bias / kAccess;
  ring.traversal.counter = padded(ringItems, 5);
  ring.traversal.syncProducer = {producerSync(true, 2), producerSync(true, 2)};
  Prologue prologue;
  prologue.loopId = 1;
  prologue.innerLimit = bias / kAccess - 1;
  prologue.outerLimit =
      blocks > 1 ? (blocks - 1) * block + bias / kAccess - 1 : block - 1;
  prologue.outerStride = block;
  prologue.accessBytes = kAccess;
  ring.traversal.prologue = prologue;
  ring.traversal.byteAddressMode = access(kAccess);
  ring.watchers[0] = ScalarSyncWatcher{ScalarSyncFlag::ParameterPop, 2, 1,
                                       blocks > 1 ? 1u : 0u, true};
  ring.syncIncrement = 1;
  ring.virtualChannels = channels;
  ring.targets = everyTile;
  return {tagged(pop), tagged(ring)};
}

SmallVector<int64_t> weightLoops(const VmcPlan &plan) {
  SmallVector<int64_t> loops;
  if (plan.transposed)
    loops = {plan.chunks, plan.outer, plan.outBlocks};
  else if (plan.single)
    loops = {plan.outBlocks};
  else
    loops = {plan.chunks * plan.taps / plan.tapGroup, plan.outer,
             plan.outBlocks};
  SmallVector<int64_t> out;
  for (int64_t count : ArrayRef(loops).drop_back())
    if (count > 1)
      out.push_back(count);
  out.push_back(loops.back());
  return out;
}

int64_t loadsPerBlock(const VmcPlan &plan) {
  if (plan.transposed)
    return plan.chunks * plan.outer;
  if (plan.single)
    return plan.outBlocks > 1 ? 1 : 0;
  return plan.chunks * plan.taps / plan.tapGroup * plan.outer;
}

FailureOr<Emitted> weightsConsumer(Operation *op, const VmcPlan &plan,
                                   Context &context,
                                   const std::array<bool, 16> &multicast,
                                   bool broadcast) {
  FailureOr<int64_t> wide = context.wideAddress(op);
  if (failed(wide))
    return unsupported(op, "unplaced convolution weights");
  int64_t bufferRows =
      plan.chunk / 2 *
      (plan.single || plan.transposed ? plan.taps : plan.tapGroup);
  SmallVector<int64_t> loops = weightLoops(plan);
  bool rows = bufferRows > 1 || plan.single || plan.transposed;
  SmallVector<Counter> items;
  if (rows)
    items.push_back(counter(bufferRows - 1, 1));
  for (auto [index, count] : llvm::enumerate(loops))
    if (!(count == 1 && index == loops.size() - 1 && loops.size() > 1))
      items.push_back(counter(count - 1, 1, false));
  int64_t row = 4 * plan.lanesPadded;
  RingConsumer consumer;
  consumer.traversal.baseAddress = *wide;
  consumer.traversal.counter = padded(items, 4);
  consumer.traversal.syncProducer = {
      producerSync(true, rows && items.size() > 1 ? 1 : 0)};
  consumer.traversal.byteAddressMode = access(row, 1);
  bool doubled = plan.outBlocks > 1 || !plan.single;
  if (doubled) {
    consumer.traversal.doubleBufferLoop = rows;
    consumer.traversal.secondBufferOffset = bufferRows;
  }
  if (plan.biasRows) {
    FailureOr<int64_t> bias = context.wideAddress(op, Suffix::Bias);
    if (failed(bias))
      return unsupported(op, "an unplaced convolution bias");
    Prologue prologue;
    prologue.baseAddress = *bias;
    prologue.outerLimit = *bias + plan.biasRows - 1;
    prologue.outerStride = 1;
    prologue.accessBytes = row;
    bool blocksCounted = items.size() == loops.size() + 1;
    int64_t loopId = items.size() - (plan.transposed || !blocksCounted ? 1 : 2);
    if (loopId > 0)
      prologue.loopId = loopId;
    consumer.traversal.prologue = prologue;
  }
  if (doubled)
    for (uint8_t thread = 0; thread < kThreads; ++thread)
      consumer.watchers.push_back(
          dmaWatcher(tileWatcher(TileSyncFlag::ParameterRead, kWatcherBase, 1,
                                 rows, false, thread),
                     true));
  consumer.destination = RingDestination::WideMemory;
  consumer.threadMulticastBitmap = bits<4>("1111");
  consumer.virtualChannelSubscription = parameterChannels(op);
  consumer.filter.firstDiscardByteLoopMap =
      broadcast ? (1u << items.size()) - 1 : 0;
  return dma(consumer, multicast);
}

FailureOr<Emitted> biasMove(Operation *op, const VmcPlan &plan,
                            Context &context) {
  FailureOr<int64_t> bias = context.wideAddress(op, Suffix::Bias);
  if (failed(bias))
    return unsupported(op, "an unplaced convolution bias");
  int64_t lanesWords = ceilDiv(plan.lanes, 4);
  int64_t blocks = plan.outBlocks;
  WideToNarrow move;
  move.read.baseAddress = *bias;
  move.write.baseAddress = 16;
  move.write.syncProducer = {producerSync(true, 2)};
  if (blocks > 1) {
    move.read.counter = padded({counter(blocks - 1, 1, false)}, 4);
    move.read.secondBufferOffset = 1;
    move.write.counter = padded({counter(lanesWords - 1, 1), counter(0, 8),
                                 counter(blocks - 1, 1, false)},
                                6);
    move.write.doubleBufferLoop = 2;
    move.write.secondBufferOffset = 8;
    move.writeWatchers = {dmaWatcher(
        tileWatcher(TileSyncFlag::ActivationWrite, kWatcherBase, 1, 2, true),
        true)};
  } else {
    move.read.counter = padded({counter(0, 1)}, 4);
    move.write.counter = padded({counter(lanesWords - 1, 1), counter(0, 8)}, 6);
  }
  move.readWatchers = {dmaWatcher(
      tileWatcher(TileSyncFlag::RingBusReadA, 1, loadsPerBlock(plan)))};
  move.wideMemoryLoadStoreLoopId = 1;
  move.isScalingFactorBias = true;
  move.zInBundleValidCount = 1;
  move.threadMulticastBitmap = bits<4>("1111");
  return dma(move, activeTiles(op));
}

Linear macLinear(bool writebackDisable) {
  Linear linear;
  linear.operation = LinearOperation::MultiplyAccumulate;
  linear.activationType = OperandType::Bfloat;
  linear.parameterType = OperandType::Bfloat;
  linear.zeroActivationPowerSave = true;
  linear.partialSumRead = PartialSumRead::Initialize;
  linear.partialSumWritebackDisable = writebackDisable;
  return linear;
}

NonLinear reluNonLinear(Operation *op, int64_t outElem) {
  NonLinear nonLinear;
  nonLinear.operation = ActivationFunction::Relu;
  nonLinear.activationPipelineScale = 1.0f;
  auto [low, high] = clips(op);
  nonLinear.highClipValue = high;
  nonLinear.lowClipValue = low;
  nonLinear.outputType = outElem == 4 ? OutputType::Single : OutputType::Bfloat;
  return nonLinear;
}

FailureOr<TensorOp> vmcTensorOp(Operation *op, const VmcPlan &plan,
                                Context &context) {
  Geometry g = lhsGeometry(op);
  auto [rows, cols] = threadPixels(op);
  auto [outStrides, outElem] = operandLayout(producer(op, 2), op);
  SmallVector<int64_t, 4> shape = resultInfo(weightsView(op)).shape;
  int64_t kh = shape.size() == 4 ? shape[0] : 1;
  int64_t kw = shape.size() == 4 ? shape[1] : 1;
  int64_t taps = kh * kw;
  bool single = plan.single;
  int64_t cycles = plan.chunk / 2;
  int64_t laneBytes = plan.lanes * outElem;
  Strided colStep{cols, g.stride * g.col, outStrides[2]};
  Strided rowStep{rows, g.stride * g.row, outStrides[1]};
  SmallVector<Strided> pixelInner, pixelOuter;
  if (cols >= plan.inner && !single) {
    pixelInner = {{plan.inner, g.stride * g.col, outStrides[2]}};
    pixelOuter = {{cols / plan.inner, plan.inner * g.stride * g.col,
                   plan.inner * outStrides[2]},
                  rowStep};
  } else if (!single) {
    pixelInner = {colStep, rowStep};
  } else {
    pixelOuter = {colStep, rowStep};
  }
  int64_t group = plan.tapGroup;
  bool pairs = !single && plan.chunks == 1;
  SmallVector<Strided> groupLoops, tapLoops{{kw, g.col, 0}, {kh, g.row, 0}};
  if (group > 1) {
    groupLoops = {tapLoops.front()};
    tapLoops.erase(tapLoops.begin());
  }
  SmallVector<Strided> chunkLoops{{plan.chunks, plan.chunk * g.elem, 0}};
  int64_t inner = countOf(pixelInner), outer = countOf(pixelOuter);

  SmallVector<int64_t> main;
  SmallVector<bool> reduction;
  auto loop = [&](int64_t count, bool reduces) {
    main.push_back(count);
    reduction.push_back(reduces);
  };
  std::optional<int64_t> tapIndex;
  if (pairs) {
    loop(plan.cin <= 4 ? cycles : 1, true);
    loop(plan.cin <= 4 ? 1 : cycles, true);
    if (group > 1)
      loop(group, true);
    loop(inner, false);
    tapIndex = main.size();
    loop(taps / group, true);
  } else {
    loop(cycles, true);
    if (!single)
      loop(inner, false);
    loop(plan.chunks, true);
    loop(taps, true);
  }
  if (single || outer > 1)
    loop(outer, false);
  loop(plan.outBlocks, false);

  TensorOp tensor;
  tensor.mainOperation.counter = mainCounters(main);
  bool unitCycles = !pairs && cycles * group == 1;
  SmallVector<std::pair<int64_t, int64_t>> read;
  if (!unitCycles)
    read.push_back({cycles, 4});
  llvm::append_range(read, mergeStream(groupLoops, &Strided::lhs));
  llvm::append_range(read, mergeStream(pixelInner, &Strided::lhs));
  llvm::append_range(read, mergeStream(chunkLoops, &Strided::lhs));
  llvm::append_range(read, mergeStream(tapLoops, &Strided::lhs));
  llvm::append_range(read, mergeStream(pixelOuter, &Strided::lhs));
  if (plan.outBlocks > 1)
    read.push_back({plan.outBlocks, 0});
  int64_t last =
      single ? shape[shape.size() - 2] * g.elem - (cycles - 1) * 4 : 4;
  tensor.narrowMemoryRead.counter = padded(masked(read), 8);
  tensor.narrowMemoryRead.byteAddressMode =
      last != 4 ? access(last, 4, 1) : access(4);
  tensor.narrowMemoryRead.syncProducer = {producerSync(false)};

  SmallVector<Strided> pixels(pixelInner);
  llvm::append_range(pixels, pixelOuter);
  auto writePixels = mergeStream(pixels, &Strided::out);
  SmallVector<std::pair<int64_t, int64_t>> writeLoops(writePixels);
  if (plan.outBlocks > 1)
    writeLoops.push_back({plan.outBlocks, laneBytes});
  SmallVector<Counter> write{counter(0, laneBytes)};
  llvm::append_range(write, masked(writeLoops));
  tensor.narrowMemoryWriteFromNonLinear.counter = padded(write, 8);
  tensor.narrowMemoryWriteFromNonLinear.syncProducer = {
      producerSync(true, 1 + writePixels.size())};
  int64_t lastLanes = shape.back() - plan.lanes * (plan.outBlocks - 1);
  tensor.narrowMemoryWriteFromNonLinear.byteAddressMode =
      lastLanes != plan.lanes
          ? access(lastLanes * outElem, laneBytes, 1u << (write.size() - 1))
          : access(laneBytes);

  FailureOr<int64_t> wide = context.wideAddress(op);
  if (failed(wide))
    return unsupported(op, "unplaced convolution weights");
  int64_t bufferUnits = cycles * 4 * (single ? taps : group);
  SmallVector<std::pair<int64_t, int64_t>> parameters;
  bool merged = single && last == 4;
  if (single) {
    if (merged) {
      parameters.push_back({cycles * taps, 4});
    } else {
      parameters.push_back({cycles, 4});
      if (taps > 1)
        parameters.push_back({taps, cycles * 4});
    }
    parameters.push_back({outer, 0});
    if (plan.outBlocks > 1)
      parameters.push_back({plan.outBlocks, 0});
  } else {
    if (!unitCycles)
      parameters.push_back({cycles * group, 4});
    parameters.push_back({inner, 0});
    if (plan.lastChunk != plan.chunk)
      parameters.append({{plan.chunks, 0}, {taps / group, 0}});
    else
      parameters.push_back({plan.chunks * taps / group, 0});
    if (outer > 1)
      parameters.push_back({outer, 0});
    parameters.push_back({plan.outBlocks, 0});
  }
  Traversal &weights = tensor.wideMemoryReadForParameters;
  weights.counter = padded(masked(parameters), 8);
  weights.syncProducer = {producerSync(
      true, 2 + (single && !merged && taps > 1) - (unitCycles && !single))};
  weights.baseAddress = *wide * 4;
  if (!single || plan.outBlocks > 1) {
    weights.doubleBufferLoop = single ? parameters.size() - 1 : 2 - unitCycles;
    weights.secondBufferOffset = bufferUnits;
  }
  int64_t lastCycles = plan.lastChunk / 2;
  bool shortLast = !single && lastCycles != cycles;
  if (shortLast) {
    tensor.mainOperation.alternateInnerLimit = lastCycles - 1;
    tensor.mainOperation.alternateLimitLoopId = 2;
    tensor.narrowMemoryRead.alternateInnerLimit = (lastCycles - 1) * 4;
    tensor.narrowMemoryRead.alternateLimitLoopId =
        1 + mergeStream(pixelInner, &Strided::lhs).size();
    weights.alternateInnerLimit = (lastCycles - 1) * 4;
    weights.alternateLimitLoopId = 2;
  }

  Traversal &sums = tensor.wideMemoryReadForSums;
  sums.baseAddress = (*wide + plan.weightsRows) * 4;
  SmallVector<std::pair<int64_t, int64_t>> sumLoops;
  if (single) {
    bool split = !merged && taps > 1;
    sumLoops.push_back({split ? cycles : cycles * taps, 0});
    if (split)
      sumLoops.push_back({taps, 0});
    sumLoops.push_back({outer, 0});
    if (plan.outBlocks > 1)
      sumLoops.push_back({plan.outBlocks, 0});
    sums.syncProducer = {producerSync(true, 2 + split)};
    sums.doubleBufferLoop = 1 + split;
    sums.secondBufferOffset = 4;
  } else {
    sumLoops = {{cycles * group, 0}, {inner, 4}};
    if (plan.lastChunk != plan.chunk)
      sumLoops.append({{plan.chunks, 0}, {taps / group, 0}});
    else
      sumLoops.push_back({plan.chunks * taps / group, 0});
    if (outer > 1)
      sumLoops.push_back({outer, 0});
    sumLoops.push_back({plan.outBlocks, 0});
    sums.syncProducer = {producerSync(true, 2)};
  }
  sums.counter = padded(masked(sumLoops), 8);
  if (shortLast) {
    sums.alternateInnerLimit = lastCycles - 1;
    sums.alternateLimitLoopId = 2;
  }

  int64_t outIndex = main.size() - 1;
  int64_t blockDepth = plan.outBlocks > 1 ? outIndex : int64_t(main.size());
  int64_t reload = single ? blockDepth : tapIndex.value_or(2);
  uint32_t stride = single && plan.outBlocks == 1 ? 0 : 1;
  tensor.syncWatchers.push_back(
      tileWatcher(TileSyncFlag::RingBusReadA, 1, stride, reload));
  if (plan.biasRows)
    tensor.syncWatchers.push_back(tileWatcher(
        TileSyncFlag::WideToScaling, 1, plan.outBlocks > 1, blockDepth, true));
  tensor.control.linear = macLinear(single);
  tensor.control.nonLinear = reluNonLinear(op, outElem);
  if (plan.biasRows) {
    tensor.control.nonLinear.applyBias = true;
  } else if (hasBias(op)) {
    FailureOr<int32_t> bias = scalarBiasBits(op);
    if (failed(bias))
      return unsupported(op, "a scalar convolution bias that is not a fill");
    tensor.control.nonLinear.immediateBias = *bias;
  }
  tensor.control.threadMulticastBitmap = bits<4>("1111");
  tensor.control.zOutBlockLoopDepth = outIndex;
  tensor.control.lastZOutBlockValidCount = lastLanes;
  tensor.control.defaultZOutBlockValidCount = plan.lanes;
  uint8_t reuse = 0;
  for (auto [index, reduces] : llvm::enumerate(reduction))
    if (reduces)
      reuse |= 1u << index;
  tensor.control.partialSumReuseMap = reuse;
  return tensor;
}

FailureOr<TensorOp> transposedTensorOp(Operation *op, const VmcPlan &plan,
                                       Context &context) {
  Geometry g = lhsGeometry(op);
  auto [outStrides, outElem] = operandLayout(producer(op, 2), op);
  SmallVector<int64_t, 4> shape = resultInfo(weightsView(op)).shape;
  int64_t kh = shape[1], kw = shape[3];
  int64_t taps = plan.taps, rows = plan.rows, cols = plan.cols,
          chunks = plan.chunks;
  int64_t cycles = plan.chunk / 2, lastCycles = (plan.lastChunk + 1) / 2;
  int64_t laneBytes = plan.lanes * outElem;
  constexpr uint8_t tiled = 2;
  TensorOp tensor;
  tensor.mainOperation.counter =
      mainCounters({cycles, taps * rows, chunks, 1, cols, plan.outBlocks});
  tensor.mainOperation.alternateInnerLimit = lastCycles - 1;
  tensor.mainOperation.alternateLimitLoopId = tiled;
  Traversal &read = tensor.narrowMemoryRead;
  read.counter =
      padded({counter((cycles - 1) * 4, 4), counter(taps - 1, 1, false),
              counter((rows - 1) * g.row, g.row),
              counter((chunks - 1) * plan.chunk * g.elem, plan.chunk * g.elem),
              counter((cols - 1) * g.col, g.col)},
             8);
  read.syncProducer = {producerSync(false)};
  read.alternateInnerLimit = (lastCycles - 1) * 4;
  read.alternateLimitLoopId = 3;
  read.byteAddressMode = access(4);
  Traversal &write = tensor.narrowMemoryWriteFromNonLinear;
  SmallVector<Counter> writeItems{
      counter(0, laneBytes), counter((kw - 1) * outStrides[2], outStrides[2]),
      counter((kh * rows - 1) * outStrides[1], outStrides[1]),
      counter((cols - 1) * kw * outStrides[2], kw * outStrides[2])};
  write.counter = padded(writeItems, 8);
  write.syncProducer = {producerSync(true, writeItems.size())};
  write.byteAddressMode = access(laneBytes);
  FailureOr<int64_t> wide = context.wideAddress(op);
  if (failed(wide))
    return unsupported(op, "unplaced convolution weights");
  int64_t bufferUnits = cycles * taps * 4;
  Traversal &weights = tensor.wideMemoryReadForParameters;
  weights.counter =
      padded({counter((cycles - 1) * 4, 4),
              counter((taps - 1) * cycles * 4, cycles * 4),
              counter(rows - 1, 1, false), counter(chunks - 1, 1, false),
              counter(cols - 1, 1, false)},
             8);
  weights.syncProducer = {producerSync(true, 3)};
  weights.doubleBufferLoop = 3;
  weights.secondBufferOffset = bufferUnits;
  weights.alternateInnerLimit = (lastCycles - 1) * 4;
  weights.alternateLimitLoopId = 3;
  weights.baseAddress = *wide * 4;
  Traversal &sums = tensor.wideMemoryReadForSums;
  sums.baseAddress = (*wide + plan.weightsRows) * 4;
  sums.counter =
      padded({counter(cycles - 1, 1, false), counter((taps * rows - 1) * 4, 4),
              counter(chunks - 1, 1, false), counter(0, 1, false),
              counter(cols - 1, 1, false)},
             8);
  sums.syncProducer = {producerSync(true, tiled)};
  sums.alternateInnerLimit = lastCycles - 1;
  sums.alternateLimitLoopId = tiled;
  tensor.syncWatchers.push_back(
      tileWatcher(TileSyncFlag::RingBusReadA, 1, 1, tiled));
  if (plan.biasRows)
    tensor.syncWatchers.push_back(
        tileWatcher(TileSyncFlag::WideToScaling, 1, 0, 6, true));
  tensor.control.linear = macLinear(false);
  tensor.control.nonLinear = reluNonLinear(op, outElem);
  if (plan.biasRows)
    tensor.control.nonLinear.applyBias = true;
  tensor.control.threadMulticastBitmap = bits<4>("1111");
  tensor.control.zOutBlockLoopDepth = 5;
  tensor.control.lastZOutBlockValidCount = plan.lanes;
  tensor.control.defaultZOutBlockValidCount = plan.lanes;
  tensor.control.partialSumReuseMap = 13;
  return tensor;
}

struct StencilPlan {
  int64_t taps;
  int64_t tapsPadded;
  int64_t kh;
  int64_t kw;
  int64_t blocks;
  int64_t rows;
  bool bias;
  int64_t lanes;
};

StencilPlan stencilPlan(Operation *op) {
  SmallVector<int64_t, 4> shape = resultInfo(weightsView(op)).shape;
  int64_t taps = product(ArrayRef(shape).drop_back());
  int64_t tapsPadded = taps + taps % 2;
  int64_t channels = extent(viewThreadBox(producer(op, 2), 0)).back();
  return StencilPlan{taps,
                     tapsPadded,
                     shape[0],
                     shape[1],
                     ceilDiv(channels, kStencilLanes),
                     tapsPadded / 2,
                     hasBias(op),
                     std::min(channels, kStencilLanes)};
}

FailureOr<Emitted> stencilConsumer(Operation *op, const StencilPlan &plan,
                                   Context &context,
                                   const std::array<bool, 16> &multicast,
                                   bool broadcast) {
  FailureOr<int64_t> wide = context.wideAddress(op);
  if (failed(wide))
    return unsupported(op, "unplaced stencil weights");
  int64_t row = 2 * kStencilLanes * 2;
  RingConsumer consumer;
  SmallVector<Counter> items;
  consumer.traversal.baseAddress = *wide;
  if (plan.taps == 1) {
    items = {counter(plan.blocks - 1, 1, false)};
    consumer.traversal.syncProducer = {producerSync(true)};
    consumer.traversal.secondBufferOffset = 1;
  } else if (plan.blocks == 1) {
    items = {counter(plan.rows - 1, 1)};
    consumer.traversal.syncProducer = {producerSync(true, 1)};
  } else {
    items = {counter(plan.rows - 1, 1), counter(plan.blocks - 1, 1, false)};
    consumer.traversal.syncProducer = {producerSync(true, 1)};
    consumer.traversal.doubleBufferLoop = 1;
    consumer.traversal.secondBufferOffset = plan.rows;
  }
  consumer.traversal.counter = padded(items, 4);
  consumer.traversal.byteAddressMode = access(row, 1);
  if (plan.bias) {
    FailureOr<int64_t> bias = context.wideAddress(op, Suffix::Bias);
    if (failed(bias))
      return unsupported(op, "an unplaced stencil bias");
    Prologue prologue;
    prologue.baseAddress = *bias;
    prologue.outerLimit = *bias + (plan.blocks > 1);
    prologue.outerStride = 1;
    prologue.accessBytes = row;
    consumer.traversal.prologue = prologue;
  }
  for (uint8_t thread = 0; plan.blocks > 1 && thread < kThreads; ++thread)
    consumer.watchers.push_back(
        dmaWatcher(tileWatcher(TileSyncFlag::ParameterRead, kWatcherBase, 1,
                               plan.taps > 1 ? 1 : 0, false, thread),
                   true));
  consumer.destination = RingDestination::WideMemory;
  consumer.threadMulticastBitmap = bits<4>("1111");
  consumer.virtualChannelSubscription = parameterChannels(op);
  consumer.filter.firstDiscardByteLoopMap =
      broadcast ? (1u << items.size()) - 1 : 0;
  return dma(consumer, multicast);
}

FailureOr<TensorOp> stencilTensorOp(Operation *op, const StencilPlan &plan,
                                    Context &context) {
  Geometry g = lhsGeometry(op);
  auto [rows, cols] = threadPixels(op);
  auto [outStrides, outElem] = operandLayout(producer(op, 2), op);
  int64_t laneBytes = plan.lanes * outElem;
  int64_t blockIn = kStencilLanes * g.elem;
  SmallVector<Strided> pixels{{cols, g.stride * g.col, outStrides[2]},
                              {rows, g.stride * g.row, outStrides[1]}};
  SmallVector<Strided> tapLoops{{plan.kw, g.col, 0}, {plan.kh, g.row, 0}};
  TensorOp tensor;
  tensor.mainOperation.counter =
      mainCounters({plan.taps, cols * rows, plan.blocks});
  auto read = mergeStream(tapLoops, &Strided::lhs);
  llvm::append_range(read, mergeStream(pixels, &Strided::lhs));
  if (plan.blocks > 1)
    read.push_back({plan.blocks, blockIn});
  tensor.narrowMemoryRead.counter = padded(masked(read), 8);
  tensor.narrowMemoryRead.byteAddressMode = access(blockIn, 4);
  tensor.narrowMemoryRead.syncProducer = {producerSync(false)};
  auto writePixels = mergeStream(pixels, &Strided::out);
  SmallVector<std::pair<int64_t, int64_t>> writeLoops(writePixels);
  if (plan.blocks > 1)
    writeLoops.push_back({plan.blocks, laneBytes});
  SmallVector<Counter> write{counter(0, laneBytes)};
  llvm::append_range(write, masked(writeLoops));
  tensor.narrowMemoryWriteFromNonLinear.counter = padded(write, 8);
  tensor.narrowMemoryWriteFromNonLinear.syncProducer = {
      producerSync(true, 1 + writePixels.size())};
  tensor.narrowMemoryWriteFromNonLinear.byteAddressMode = access(laneBytes);
  FailureOr<int64_t> wide = context.wideAddress(op);
  if (failed(wide))
    return unsupported(op, "unplaced stencil weights");
  Traversal &weights = tensor.wideMemoryReadForParameters;
  int64_t sumsBase;
  if (plan.taps == 1) {
    weights.counter = padded({counter(cols * rows - 1, 1, false),
                              counter(plan.blocks - 1, 1, false)},
                             8);
    weights.syncProducer = {producerSync(true, 1)};
    weights.doubleBufferLoop = 1;
    weights.secondBufferOffset = 4;
    sumsBase = (*wide + 2) * 4;
  } else if (plan.blocks == 1) {
    weights.counter = padded(
        {counter((plan.taps - 1) * 2, 2), counter(cols * rows - 1, 1, false)},
        8);
    weights.syncProducer = {producerSync(true, 2)};
    sumsBase = (*wide + plan.rows) * 4;
  } else {
    weights.counter = padded({counter((plan.taps - 1) * 2, 2),
                              counter(cols * rows - 1, 1, false),
                              counter(plan.blocks - 1, 1, false)},
                             8);
    weights.syncProducer = {producerSync(true, 2)};
    weights.doubleBufferLoop = 2;
    weights.secondBufferOffset = plan.rows * 4;
    sumsBase = (*wide + 2 * plan.rows) * 4;
  }
  weights.baseAddress = *wide * 4;
  Traversal &sums = tensor.wideMemoryReadForSums;
  sums.baseAddress = sumsBase;
  SmallVector<Counter> sumItems{counter(plan.taps - 1, 1, false),
                                counter(cols * rows - 1, 1, false)};
  if (plan.blocks > 1)
    sumItems.push_back(counter(plan.blocks - 1, 1, false));
  sums.counter = padded(sumItems, 8);
  sums.syncProducer = {producerSync(true, 2)};
  sums.doubleBufferLoop = 1;
  sums.secondBufferOffset = 4;
  bool blocks = plan.blocks > 1;
  tensor.syncWatchers.push_back(
      tileWatcher(TileSyncFlag::RingBusReadA, 1, blocks, blocks ? 2 : 3));
  if (plan.bias)
    tensor.syncWatchers.push_back(tileWatcher(TileSyncFlag::WideToScaling, 1,
                                              blocks, blocks ? 2 : 3, true));
  Linear linear;
  linear.operation = LinearOperation::HighBandwidthMac;
  linear.activationType = OperandType::Bfloat;
  linear.parameterType = OperandType::Bfloat;
  linear.macDisable = MacDisable::SecondFloat;
  linear.partialSumRead = PartialSumRead::Initialize;
  linear.partialSumWritebackDisable = true;
  tensor.control.linear = linear;
  tensor.control.nonLinear = reluNonLinear(op, outElem);
  if (plan.bias)
    tensor.control.nonLinear.applyBias = true;
  tensor.control.threadMulticastBitmap = bits<4>("1111");
  tensor.control.zOutBlockLoopDepth = 2;
  tensor.control.lastZOutBlockValidCount = plan.lanes;
  tensor.control.defaultZOutBlockValidCount = plan.lanes;
  tensor.control.partialSumReuseMap = 1;
  return tensor;
}

SmallVector<int64_t> activeList(const std::array<bool, 16> &tiles) {
  SmallVector<int64_t> out;
  for (int64_t tile = 0; tile < kTiles; ++tile)
    if (tiles[tile])
      out.push_back(tile);
  return out;
}

// Fitted to SDK probes: the partial sum block is the largest divisor of a
// thread's columns that leaves wide rows for 7 channel pairs, or for one pair
// when there are at most 8.
FailureOr<VmcPlan> tiledPlan(Operation *op, VmcPlan plan, int64_t cols) {
  int64_t pairs = plan.cin / 2;
  int64_t inner = std::min(cols, kWideRows - plan.biasRows -
                                     (pairs <= kGroupedPairs ? 2 : 14));
  if (inner < 1)
    return unsupported(op, "a convolution whose partial sums exceed wide "
                           "memory");
  while (cols % inner)
    --inner;
  int64_t most = (kWideRows - inner - plan.biasRows) / 2;
  int64_t per = most;
  if (pairs <= 2 * most)
    per = ceilDiv(pairs, 2);
  else if (most >= kGroupedPairs)
    while (pairs % per)
      --per;
  plan.chunk = 2 * per;
  plan.chunks = ceilDiv(pairs, per);
  plan.lastChunk = plan.cin - plan.chunk * (plan.chunks - 1);
  plan.inner = inner;
  plan.outer = plan.pixels / inner;
  plan.weightsRows = plan.chunk;
  plan.sumsRows = inner;
  return plan;
}

} // namespace

Operation *codegen::weightsView(Operation *op) {
  Operation *weights = producer(op, 1);
  while (isa<NarrowToWideOp, DistributedCreateViewOp, GetTensorOp>(weights))
    weights = producer(weights, 0);
  return weights;
}

bool codegen::hasBias(Operation *op) {
  return hasAuxiliary(op, AuxTensorKind::Bias);
}

int64_t codegen::stencilBlocks(Operation *op) {
  return ceilDiv(extent(viewThreadBox(producer(op, 2), 0)).back(),
                 kStencilLanes);
}

int64_t codegen::stencilTaps(Operation *op) {
  return product(ArrayRef(resultInfo(weightsView(op)).shape).drop_back());
}

FailureOr<VmcPlan> codegen::vmcPlan(Operation *op) {
  SmallVector<int64_t, 4> shape = resultInfo(weightsView(op)).shape;
  VmcPlan plan;
  if (isTransposed(op)) {
    int64_t taps = product(ArrayRef(shape).drop_back(2));
    int64_t cin = shape[shape.size() - 2];
    Box input = viewThreadBox(producer(op, 0), 0);
    Box output = viewThreadBox(producer(op, 2), 0);
    int64_t rows = input.hi[1] - input.lo[1] + 1;
    int64_t cols = input.hi[2] - input.lo[2] + 1;
    int64_t cout = output.hi.back() - output.lo.back() + 1;
    if (cout > kOutLanes)
      return unsupported(op, "a transposed convolution with more than 32 "
                             "output channels");
    int64_t sums = taps * rows;
    int64_t biasRows = hasBias(op) ? 1 : 0;
    int64_t chunk = 0;
    for (int64_t c = 2; c < cin + 2; c += 2)
      if (2 * (c / 2 * taps) + sums + biasRows <= kWideRows)
        chunk = c;
    if (!chunk)
      return unsupported(op, "a transposed convolution whose partial sums "
                             "exceed wide memory");
    chunk = std::min(chunk, cin + cin % 2);
    int64_t chunks = ceilDiv(cin, chunk);
    plan.transposed = true;
    plan.lanesPadded = std::min(ceilDiv(cout, 8) * 8, kOutLanes);
    plan.cin = cin;
    plan.chunk = chunk;
    plan.chunks = chunks;
    plan.lastChunk = cin - chunk * (chunks - 1);
    plan.taps = taps;
    plan.rows = rows;
    plan.cols = cols;
    plan.pixels = rows * cols;
    plan.inner = 1;
    plan.outer = cols;
    plan.lanes = std::min(cout, kOutLanes);
    plan.outBlocks = 1;
    plan.weightsRows = 2 * (chunk / 2 * taps);
    plan.sumsRows = sums;
    plan.biasRows = biasRows;
    return plan;
  }
  int64_t taps = product(ArrayRef(shape).drop_back(2));
  int64_t cin = shape[shape.size() - 2];
  int64_t cinPadded = cin + cin % 2;
  Index size = extent(viewThreadBox(producer(op, 2), 0));
  int64_t cout = size.back();
  int64_t pixels = product(ArrayRef(size).drop_front().drop_back());
  plan.lanes = std::min(cout, kOutLanes);
  plan.lanesPadded = std::min(ceilDiv(cout, 8) * 8, kOutLanes);
  plan.outBlocks = ceilDiv(cout, kOutLanes);
  plan.biasRows = hasBias(op) && cout > 1 ? (plan.outBlocks == 1 ? 1 : 2) : 0;
  plan.taps = taps;
  plan.pixels = pixels;
  plan.cin = cinPadded;
  int64_t singleRows = cinPadded / 2 * taps * (plan.outBlocks > 1 ? 2 : 1) + 1;
  if (singleRows + plan.biasRows <= kWideRows &&
      (pixels > kPartialSumPixels ||
       cinPadded / 2 * taps + 1 + plan.biasRows <= kSingleRows)) {
    plan.single = true;
    plan.chunk = cinPadded;
    plan.chunks = 1;
    plan.inner = 1;
    plan.outer = pixels;
    plan.weightsRows = singleRows - 1;
    plan.sumsRows = 1;
    return plan;
  }
  if (pixels > kPartialSumPixels)
    return tiledPlan(op, plan, size[2]);
  int64_t chunk = 0;
  for (int64_t c = 2; c <= cinPadded; c += 2)
    if (cinPadded % c == 0 && c + pixels + plan.biasRows <= kWideRows)
      chunk = c;
  if (!chunk)
    return unsupported(op, "a convolution whose partial sums exceed wide "
                           "memory");
  plan.chunk = chunk;
  plan.chunks = cinPadded / chunk;
  plan.lastChunk = chunk;
  plan.inner = pixels;
  plan.outer = 1;
  if (plan.chunks == 1 && chunk / 2 < kGroupedPairs && shape.size() == 4 &&
      chunk * shape[1] + pixels + plan.biasRows <= kWideRows)
    plan.tapGroup = shape[1];
  plan.weightsRows = chunk * plan.tapGroup;
  plan.sumsRows = pixels;
  return plan;
}

Body codegen::vmc(Operation *op, Context &context) {
  FailureOr<VmcPlan> plan = vmcPlan(op);
  if (failed(plan))
    return failure();
  int64_t total = parameterBytes(*plan);
  SmallVector<Emitted, 0> out = groupFences();
  out.push_back(hibGather(
      {total}, {total}, 1, DmaQueue::Parameter,
      context.hib(DmaQueue::Parameter, HibRoot::Parameter, 0, total, op)));
  llvm::append_range(out, parameterInfeed(op, *plan));
  std::array<bool, 16> active = activeTiles(op);
  SmallVector<int64_t> tiles = activeList(active);
  FailureOr<SmallVector<Emitted, 0>> loads = registers(op, context);
  if (failed(loads))
    return failure();
  if (!plan->biasRows) {
    FailureOr<Emitted> consumer =
        weightsConsumer(op, *plan, context, active, true);
    FailureOr<TensorOp> tensorOp = vmcTensorOp(op, *plan, context);
    if (failed(consumer) || failed(tensorOp))
      return failure();
    out.push_back(*consumer);
    llvm::append_range(out, *loads);
    out.push_back(tensor(*tensorOp, active, bits<8>("10100000")));
    llvm::append_range(out, groupFences());
    return out;
  }
  FailureOr<Emitted> first =
      weightsConsumer(op, *plan, context, tileBit(tiles.front()), false);
  FailureOr<Emitted> bias = biasMove(op, *plan, context);
  FailureOr<TensorOp> tensorOp = plan->transposed
                                     ? transposedTensorOp(op, *plan, context)
                                     : vmcTensorOp(op, *plan, context);
  if (failed(first) || failed(bias) || failed(tensorOp))
    return failure();
  out.push_back(*first);
  out.push_back(*bias);
  llvm::append_range(out, *loads);
  out.push_back(tensor(*tensorOp, active, bits<8>("10100000")));
  for (int64_t tile : ArrayRef(tiles).drop_front()) {
    FailureOr<Emitted> consumer =
        weightsConsumer(op, *plan, context, tileBit(tile), false);
    if (failed(consumer))
      return failure();
    out.push_back(*consumer);
  }
  llvm::append_range(out, groupFences());
  return out;
}

Body codegen::stencil(Operation *op, Context &context) {
  StencilPlan plan = stencilPlan(op);
  int64_t perBlock =
      plan.tapsPadded * kStencilLanes * 2 + (plan.bias ? kStencilLanes * 4 : 0);
  int64_t total = perBlock * plan.blocks;
  SmallVector<Emitted, 0> out = groupFences();
  out.push_back(hibGather(
      {total}, {total}, 1, DmaQueue::Parameter,
      context.hib(DmaQueue::Parameter, HibRoot::Parameter, 0, total, op)));
  llvm::append_range(out, infeed(total, parameterChannels(op),
                                 targets(activeTiles(op), false),
                                 InputFifo::Parameter));
  std::array<bool, 16> active = activeTiles(op);
  SmallVector<int64_t> tiles = activeList(active);
  FailureOr<SmallVector<Emitted, 0>> loads = registers(op, context);
  FailureOr<TensorOp> tensorOp = stencilTensorOp(op, plan, context);
  if (failed(loads) || failed(tensorOp))
    return failure();
  if (!plan.bias) {
    FailureOr<Emitted> consumer =
        stencilConsumer(op, plan, context, active, true);
    if (failed(consumer))
      return failure();
    out.push_back(*consumer);
    llvm::append_range(out, *loads);
    out.push_back(tensor(*tensorOp, active, bits<8>("10100000")));
    llvm::append_range(out, groupFences());
    return out;
  }
  FailureOr<Emitted> first =
      stencilConsumer(op, plan, context, tileBit(tiles.front()), false);
  VmcPlan biasPlan;
  biasPlan.lanesPadded = kStencilLanes;
  biasPlan.lanes = kStencilLanes;
  biasPlan.outBlocks = plan.blocks;
  biasPlan.single = true;
  biasPlan.chunks = 1;
  biasPlan.taps = 1;
  biasPlan.outer = 1;
  FailureOr<Emitted> bias = biasMove(op, biasPlan, context);
  if (failed(first) || failed(bias))
    return failure();
  out.push_back(*first);
  out.push_back(*bias);
  llvm::append_range(out, *loads);
  out.push_back(tensor(*tensorOp, active, bits<8>("10100000")));
  for (int64_t tile : ArrayRef(tiles).drop_front()) {
    FailureOr<Emitted> consumer =
        stencilConsumer(op, plan, context, tileBit(tile), false);
    if (failed(consumer))
      return failure();
    out.push_back(*consumer);
  }
  llvm::append_range(out, groupFences());
  return out;
}
