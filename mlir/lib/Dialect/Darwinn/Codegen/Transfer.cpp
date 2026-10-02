#include "Families.h"

using namespace mlir;
using namespace mlir::darwinn;
using namespace mlir::darwinn::codegen;

namespace {

constexpr int64_t kFifo = 14;
constexpr int64_t kOutfeedLimit = int64_t(1) << 17;

std::optional<AffineMap> forwardMap(Operation *op) {
  if (auto redistribute = dyn_cast<RedistributeOp>(op)) {
    if (auto mapping = redistribute.getMappingAttr())
      return mapping.getForwardIndexTransformation().getValue();
    return std::nullopt;
  }
  if (auto attr =
          op->getAttrOfType<AffineMapAttr>("forward_index_transformation"))
    return attr.getValue();
  return std::nullopt;
}

std::optional<AffineMap> reverseMap(Operation *op) {
  if (auto redistribute = dyn_cast<RedistributeOp>(op)) {
    if (auto mapping = redistribute.getMappingAttr())
      return mapping.getReverseIndexTransformation().getValue();
    return std::nullopt;
  }
  if (auto attr =
          op->getAttrOfType<AffineMapAttr>("reverse_index_transformation"))
    return attr.getValue();
  return std::nullopt;
}

Traversal fifoTraversal(int64_t total) {
  int64_t accesses = ceilDiv(total, kAccess);
  Traversal traversal;
  if (accesses <= kFifo) {
    traversal.counter = padded({counter(accesses - 1, 1)}, 5);
    return traversal;
  }
  int64_t fifos = ceilDiv(accesses, kFifo);
  int64_t last = accesses - kFifo * (fifos - 1);
  traversal.counter =
      padded({counter(kFifo - 1, 1), counter(fifos - 1, 1, false)}, 5);
  if (last != kFifo) {
    traversal.alternateInnerLimit = last - 1;
    traversal.alternateLimitLoopId = 1;
  }
  return traversal;
}

int64_t fifoInner(int64_t total) {
  return std::min(ceilDiv(total, kAccess), kFifo);
}

ScalarSyncWatcher scalarWatcher(ScalarSyncFlag flag, uint32_t initial,
                                uint32_t stride, uint8_t depth = 0) {
  return ScalarSyncWatcher{flag, depth, initial, stride, true};
}

struct HostView {
  Index view;
  Index buffer;
  Index offset;
  int64_t elementBytes;
};

HostView hostView(Operation *redistribute) {
  TensorInfo info = operandInfo(redistribute, 0);
  HostView out{Index(info.shape), Index(info.shape),
               Index(info.shape.size(), 0), info.elementBytes};
  Operation *node = producer(redistribute, 0);
  Index zeros(info.shape.size(), 0);
  while (node && isa<DistributedCreateViewOp, CommunicatedJoinViewsOp>(node)) {
    if (isa<CommunicatedJoinViewsOp>(node)) {
      out.buffer = Index(resultInfo(node).shape);
      break;
    }
    Index shift = applyForward(node, zeros);
    for (size_t dim = 0; dim < out.offset.size(); ++dim)
      out.offset[dim] -= shift[dim];
    node = producer(node, 0);
    out.buffer = node ? Index(resultInfo(node).shape) : out.view;
  }
  return out;
}

HostView hostDestination(Operation *redistribute) {
  TensorInfo info = resultInfo(redistribute);
  HostView out{Index(info.shape), Index(info.shape),
               Index(info.shape.size(), 0), info.elementBytes};
  if (redistribute->getNumOperands() < 2)
    return out;
  Operation *view = producer(redistribute, 1);
  out.offset = applyReverse(view, Index(info.shape.size(), 0));
  Operation *root = view;
  while (isa<CommunicatedCreateWriteViewOp>(root))
    root = producer(root, 0);
  out.buffer = Index(resultInfo(root).shape);
  return out;
}

Operation *hostWriter(Operation *node) {
  while (
      isa<DistributedCreateViewOp, CommunicatedJoinViewsOp, ReshapeOpOp>(node))
    node = producer(node, 0);
  return node;
}

int64_t byteOffset(ArrayRef<int64_t> offset, ArrayRef<int64_t> buffer,
                   int64_t elementBytes) {
  return dot(offset, strides(buffer, elementBytes));
}

Emitted hibScatter(ArrayRef<int64_t> view, ArrayRef<int64_t> buffer,
                   int64_t elementBytes, DmaQueue queue, int64_t index) {
  if (view != buffer)
    return hibGather(view, buffer, elementBytes, queue, index);
  int64_t total = product(view) * elementBytes;
  HibDma hib;
  hib.queue = queue;
  hib.traversal.baseAddress = index;
  hib.traversal.counter = padded({counter(0, total)}, 4);
  hib.traversal.byteAddressMode = access(total);
  return tagged(hib);
}

Emitted ringProducer(int64_t base, ArrayRef<Loop> loops, int64_t tile,
                     int64_t order, int64_t cycle,
                     const std::array<bool, 8> &channels,
                     const std::array<bool, 17> &targets) {
  int64_t run = loops.back().count * loops.back().destination;
  SmallVector<Counter> items;
  for (const Loop &loop : llvm::reverse(loops.drop_back()))
    items.push_back(
        counter((loop.count - 1) * loop.destination, loop.destination));
  if (items.empty())
    items.push_back(counter(0, run));
  RingProducer producer;
  producer.traversal.baseAddress = base;
  producer.traversal.counter = padded(items, 4);
  producer.traversal.syncProducer = {producerSync(true), producerSync(true)};
  producer.traversal.byteAddressMode = access(run);
  producer.watchers = {
      dmaWatcher(tileWatcher(TileSyncFlag::RingBusProducerA, order, cycle))};
  producer.virtualChannelSubscription = channels;
  producer.targets = targets;
  return dma(producer, tileBit(tile));
}

SmallVector<int64_t> chunkOffsets(ArrayRef<Loop> loops, int64_t origin) {
  SmallVector<int64_t> offsets{origin};
  for (const Loop &loop : loops.drop_back()) {
    SmallVector<int64_t> next;
    for (int64_t offset : offsets)
      for (int64_t index = 0; index < loop.count; ++index)
        next.push_back(offset + index * loop.source);
    offsets = std::move(next);
  }
  llvm::sort(offsets);
  return offsets;
}

std::array<bool, 16> tilesOf(ArrayRef<TileBox> boxes) {
  std::array<bool, 16> out{};
  for (const TileBox &box : boxes)
    out[box.tile] = true;
  return out;
}

Emitted zeroFill(int64_t base, int64_t run,
                 ArrayRef<std::pair<int64_t, int64_t>> outer,
                 MeshDirection direction, const std::array<bool, 16> &tiles,
                 int64_t elementBytes, bool synced) {
  SmallVector<Counter> items;
  int64_t bytes;
  if (run >= 16) {
    items.push_back(counter(llvm::alignTo(run, 16) - 16, 16));
    bytes = 16;
  } else {
    items.push_back(counter(0, 16));
    bytes = run;
  }
  for (auto [count, stride] : outer)
    items.push_back(counter((count - 1) * stride, stride));
  Mesh mesh;
  mesh.direction = direction;
  mesh.write.baseAddress = base;
  mesh.write.counter = padded(items, 5);
  mesh.write.byteAddressMode =
      run > 16 && run % 16 ? access(run % 16, 16, 1) : access(bytes);
  mesh.write.syncProducer = {producerSync(synced)};
  mesh.validBytes = elementBytes;
  return dma(mesh, tiles);
}

MeshDirection padDirection(int dim, bool low) {
  if (dim == 2)
    return low ? MeshDirection::OutboundEastInboundWest
               : MeshDirection::OutboundWestInboundEast;
  return low ? MeshDirection::OutboundSouthInboundNorth
             : MeshDirection::OutboundNorthInboundSouth;
}

} // namespace

Index codegen::applyForward(Operation *op, ArrayRef<int64_t> point) {
  if (auto map = forwardMap(op))
    return evaluate(*map, point);
  return Index(point);
}

Index codegen::applyReverse(Operation *op, ArrayRef<int64_t> point) {
  if (auto map = reverseMap(op))
    return evaluate(*map, point);
  return Index(point);
}

SmallVector<Loop> codegen::mergeLoops(ArrayRef<Loop> loops) {
  SmallVector<Loop> kept;
  for (const Loop &loop : loops)
    if (loop.count > 1)
      kept.push_back(loop);
  if (kept.empty())
    kept.push_back(loops.back());
  SmallVector<Loop> merged{kept.back()};
  for (const Loop &loop : llvm::reverse(ArrayRef(kept).drop_back())) {
    Loop &inner = merged.front();
    if (loop.source == inner.count * inner.source &&
        loop.destination == inner.count * inner.destination)
      inner.count *= loop.count;
    else
      merged.insert(merged.begin(), loop);
  }
  return merged;
}

Emitted codegen::ringConsumer(int64_t base, ArrayRef<Loop> loops, int64_t first,
                              int64_t last, const std::array<bool, 8> &channels,
                              const std::array<bool, 16> &multicast,
                              RingDestination destination) {
  const Loop &innermost = loops.back();
  int64_t inner = innermost.count * innermost.destination;
  int64_t gap = loops.size() > 1 ? loops[loops.size() - 2].source -
                                       innermost.count * innermost.source
                                 : 0;
  SmallVector<Counter> items{
      counter(llvm::alignTo(inner, kAccess) - kAccess, kAccess)};
  for (const Loop &loop : llvm::reverse(loops.drop_back()))
    items.push_back(
        counter((loop.count - 1) * loop.destination, loop.destination));
  RingConsumer consumer;
  consumer.traversal.baseAddress = base;
  consumer.traversal.counter = padded(items, 4);
  consumer.traversal.syncProducer = {producerSync(true, items.size() - 1)};
  consumer.traversal.byteAddressMode = access(std::min(inner, kAccess));
  consumer.virtualChannelSubscription = channels;
  consumer.destination = destination;
  consumer.threadMulticastBitmap = bits<4>("1000");
  consumer.filter.firstDiscardByteLoopMap = (1u << items.size()) - 1;
  consumer.filter.discardByteLoopMap = gap ? 1 : 0;
  consumer.filter.firstDiscardByteCount = first;
  consumer.filter.discardByteCount = gap;
  consumer.filter.lastDiscardByteCount = last;
  return dma(consumer, multicast);
}

SmallVector<Emitted, 0> codegen::infeed(int64_t total,
                                        const std::array<bool, 8> &channels,
                                        const std::array<bool, 17> &targets,
                                        InputFifo fifo) {
  bool parameter = fifo == InputFifo::Parameter;
  PopInput pop;
  pop.fifo = fifo;
  pop.traversal = fifoTraversal(total);
  pop.traversal.syncProducer = {producerSync(true)};
  pop.watcher = scalarWatcher(parameter ? ScalarSyncFlag::ParameterInfeed
                                        : ScalarSyncFlag::ActivationInfeed,
                              (uint32_t(1) << 25) - fifoInner(total) + 1, 1);
  uint8_t depth = ceilDiv(total, kAccess) > kFifo ? 2 : 1;
  RingInfeed ring;
  ring.fifo = fifo;
  ring.traversal = fifoTraversal(total);
  ring.traversal.syncProducer = {producerSync(true, depth), producerSync(true)};
  ring.traversal.byteAddressMode = access(kAccess);
  ring.watchers[0] = scalarWatcher(parameter ? ScalarSyncFlag::ParameterPop
                                             : ScalarSyncFlag::ActivationPop,
                                   1, 1);
  ring.syncIncrement = 1;
  ring.virtualChannels = channels;
  ring.targets = targets;
  return {tagged(pop), tagged(ring)};
}

Emitted codegen::outfeed(int64_t total, const std::array<bool, 8> &channels) {
  int64_t accesses = ceilDiv(total, kAccess);
  SmallVector<Counter> items{counter(accesses - 1, 1, false)};
  if (accesses > kOutfeedLimit) {
    int64_t factor = 2;
    while (accesses % factor || accesses / factor > kOutfeedLimit)
      ++factor;
    items = {counter(accesses / factor - 1, 1, false),
             counter(factor - 1, 1, false)};
  }
  RingOutfeed out;
  out.traversal.counter = padded(items, 5);
  out.traversal.syncProducer = {producerSync(false)};
  out.traversal.byteAddressMode = access(std::min(total, kAccess));
  out.virtualChannels = channels;
  out.bytesToPop = total;
  return tagged(out);
}

Emitted codegen::hibGather(ArrayRef<int64_t> view, ArrayRef<int64_t> buffer,
                           int64_t elementBytes, DmaQueue queue,
                           int64_t index) {
  int64_t total = product(view) * elementBytes;
  HibDma hib;
  hib.queue = queue;
  hib.traversal.baseAddress = index;
  if (view == buffer) {
    hib.traversal.counter = padded({counter(0, 1)}, 4);
    hib.traversal.byteAddressMode = access(total, 1);
    return tagged(hib);
  }
  SmallVector<Loop> loops;
  Index stride = strides(buffer, elementBytes);
  for (auto [count, step] : llvm::zip_equal(view, stride))
    loops.push_back({count, step, step});
  loops = mergeLoops(loops);
  int64_t inner = loops.back().count * elementBytes;
  SmallVector<Counter> items{counter(0, inner)};
  for (const Loop &loop : llvm::reverse(ArrayRef(loops).drop_back()))
    if (loop.count > 1)
      items.push_back(counter((loop.count - 1) * loop.source, loop.source));
  hib.traversal.counter = padded(items, 4);
  hib.traversal.byteAddressMode = access(inner);
  return tagged(hib);
}

Body codegen::joinedCopies(Operation *op, Context &context) {
  auto join = cast<CommunicatedJoinViewsOp>(op);
  auto slicing = slicingOf(op);
  auto destinationBase = context.narrowAddress(op);
  if (!slicing || failed(destinationBase))
    return unsupported(op, "a join without placed tile storage");

  Index destinationShape = extent(ownerTileBox(op, 0));
  TensorInfo output = resultInfo(op);
  Index destinationStrides = strides(destinationShape, output.elementBytes);
  SmallVector<int64_t> active;
  for (const TileBox &tile : tiles(*slicing)) {
    if (extent(tile.box) != destinationShape)
      return unsupported(op, "a join with differing tile extents");
    active.push_back(tile.tile);
  }

  SmallVector<Emitted, 0> instructions;
  for (Value filled : join.getOperands()) {
    auto copy = filled.getDefiningOp<NarrowToNarrowOp>();
    if (!copy || copy.getShards().size() != 1)
      return unsupported(op, "a join without one narrow-copy shard per input");
    auto view =
        copy.getDestination().getDefiningOp<CommunicatedCreateWriteViewOp>();
    if (!copy.getInput().getDefiningOp())
      return unsupported(op, "a join copy without a placed input producer");
    auto sourceBase =
        context.storageAddress(storage(copy.getInput().getDefiningOp()));
    auto shard = cast<NarrowToNarrowShardAttr>(copy.getShards()[0]);
    if (!view || failed(sourceBase) ||
        shard.getSlices().size() != active.size())
      return unsupported(op, "a join copy without matching tile ownership");

    auto sample = cast<NarrowToNarrowSliceAttr>(shard.getSlices()[0]);
    if (!sample.getReadDomain() || !sample.getReadAffineMap() ||
        !sample.getWriteAffineMap() || !sample.getReadBytes() ||
        !sample.getWriteBytes())
      return unsupported(op, "a join copy without complete traversal fields");
    Index shape;
    for (Attribute dimension : sample.getReadDomain())
      shape.push_back(cast<IntegerAttr>(dimension).getInt());
    if (shape.empty() ||
        llvm::any_of(shape, [](int64_t size) { return size <= 0; }) ||
        shape.size() != destinationShape.size() ||
        sample.getReadBytes().getInt() != output.elementBytes ||
        sample.getWriteBytes().getInt() != output.elementBytes ||
        ArrayRef<int64_t>(shape).drop_back() !=
            ArrayRef<int64_t>(destinationShape).drop_back())
      return unsupported(op,
                         "a join copy that does not preserve spatial tiles");

    unsigned rank = shape.size();
    auto addressMap = [&](ArrayRef<int64_t> extents) {
      Index layout = strides(extents, 1);
      AffineExpr address = getAffineConstantExpr(0, op->getContext());
      for (auto [axis, stride] : llvm::enumerate(layout))
        address = address + getAffineDimExpr(axis, op->getContext()) * stride;
      return AffineMap::get(rank, 0, address);
    };
    if (sample.getReadAffineMap().getValue() != addressMap(shape) ||
        sample.getWriteAffineMap().getValue() != addressMap(destinationShape))
      return unsupported(op, "a join copy without dense channel traversals");
    for (auto [index, attribute] : llvm::enumerate(shard.getSlices())) {
      auto slice = cast<NarrowToNarrowSliceAttr>(attribute);
      auto id = slice.getId();
      if (!id || id.size() != 3 ||
          cast<IntegerAttr>(id[0]).getInt() != active[index] / kGrid ||
          cast<IntegerAttr>(id[1]).getInt() != active[index] % kGrid ||
          cast<IntegerAttr>(id[2]).getInt() != 0 ||
          slice.getReadDomain() != sample.getReadDomain() ||
          slice.getWriteDomain() != sample.getReadDomain() ||
          slice.getReadAffineMap() != sample.getReadAffineMap() ||
          slice.getWriteAffineMap() != sample.getWriteAffineMap() ||
          slice.getReadBytes() != sample.getReadBytes() ||
          slice.getWriteBytes() != sample.getWriteBytes())
        return unsupported(
            op, "a join copy with differing tile traversal geometry");
    }

    AffineMap mapping = view.getReverseIndexTransformation();
    Index origin(rank, 0);
    Index offset = evaluate(mapping, origin);
    for (unsigned axis = 0; axis < rank; ++axis) {
      if ((axis + 1 != rank && offset[axis] != 0) || offset[axis] < 0 ||
          offset[axis] + shape[axis] > destinationShape[axis] ||
          mapping.getResult(axis) !=
              getAffineDimExpr(axis, op->getContext()) + offset[axis])
        return unsupported(op,
                           "a join copy without a channel-offset write view");
    }

    int64_t channelBytes = shape.back() * output.elementBytes;
    int64_t pixels = product(ArrayRef<int64_t>(shape).drop_back());
    if (channelBytes % 16)
      return unsupported(
          op, "a join copy whose channels do not fill sixteen-byte accesses");
    NarrowToNarrow transfer;
    transfer.read.baseAddress = *sourceBase;
    transfer.read.counter =
        padded({counter(pixels * channelBytes - 16, 16)}, 4);
    transfer.read.syncProducer = {producerSync(false)};
    transfer.read.byteAddressMode = access(16);
    transfer.write.baseAddress =
        *destinationBase + dot(offset, destinationStrides);
    int64_t pixelStride = destinationShape.back() * output.elementBytes;
    transfer.write.counter =
        padded({counter(channelBytes - 16, 16),
                counter((pixels - 1) * pixelStride, pixelStride)},
               4);
    transfer.write.syncProducer = {producerSync(true, 1)};
    transfer.write.byteAddressMode = access(16);
    instructions.push_back(dma(transfer, tileSet(active)));
  }
  return instructions;
}

Body codegen::transferLoad(Operation *op, Context &context) {
  Operation *source = producer(op, 0);
  HostView host = hostView(op);
  SmallVector<int64_t, 4> destination = resultInfo(op).shape;
  int64_t elem = host.elementBytes;
  int64_t total = product(host.view) * elem;
  SmallVector<TileBox> boxes = tiles(*slicingOf(op));
  std::array<bool, 8> channels =
      boxes.size() == kTiles ? bits<8>("01000000") : bits<8>("00000001");
  SmallVector<Emitted, 0> out;
  if (source) {
    ScalarFence fence;
    fence.waitIdle = bits<5>("00001");
    fence.sendInterrupt = true;
    out.push_back(scalarFence(fence, Role::OutputActivationInterrupt));
  }
  if (host.view == host.buffer)
    llvm::append_range(out, groupFences());
  int64_t index;
  if (!source) {
    index = context.hib(DmaQueue::Activation, HibRoot::InputActivation, 0);
  } else {
    FailureOr<int64_t> offset = context.hostOffset(hostWriter(source));
    if (failed(offset))
      return unsupported(op, "a load from an unplaced host buffer");
    index = context.hib(DmaQueue::Activation, HibRoot::Scratch,
                        *offset + byteOffset(host.offset, host.buffer, elem));
  }
  out.push_back(
      hibGather(host.view, host.buffer, elem, DmaQueue::Activation, index));
  llvm::append_range(out,
                     infeed(total, channels, targets(tilesOf(boxes), false),
                            InputFifo::Activation));
  FailureOr<int64_t> base = context.narrowAddress(op);
  if (failed(base))
    return unsupported(op, "a load into an unplaced narrow block");
  Index stream = strides(host.view, elem);
  for (const auto &[tile, box] : boxes) {
    Index lo = tail(box.lo, destination.size());
    Index hi = tail(box.hi, destination.size());
    Index narrow = strides(extent(Box{lo, hi}), elem);
    Index sourceLo = applyReverse(op, lo), sourceHi = applyReverse(op, hi);
    for (size_t dim = 0; dim < sourceLo.size(); ++dim) {
      sourceLo[dim] = std::max<int64_t>(0, sourceLo[dim]);
      sourceHi[dim] = std::min(host.view[dim] - 1, sourceHi[dim]);
    }
    Index mapped = applyForward(op, sourceLo);
    Index destinationLo;
    for (auto [a, b] : llvm::zip(mapped, lo))
      destinationLo.push_back(a - b);
    int64_t start = dot(destinationLo, narrow);
    SmallVector<Loop> loops;
    for (size_t dim = 0; dim < sourceLo.size(); ++dim)
      loops.push_back(
          {sourceHi[dim] - sourceLo[dim] + 1, stream[dim], narrow[dim]});
    loops = mergeLoops(loops);
    int64_t first = dot(sourceLo, stream);
    int64_t last = total - (dot(sourceHi, stream) + elem);
    out.push_back(ringConsumer(*base + start, loops, first, last, channels,
                               tileBit(tile)));
  }
  llvm::append_range(out, groupFences());
  return out;
}

Body codegen::transferStore(Operation *op, Context &context) {
  Operation *source = producer(op, 0);
  HostView host = hostDestination(op);
  int64_t elem = host.elementBytes;
  int64_t total = product(host.view) * elem;
  std::array<bool, 8> channels = bits<8>("00000001");
  SmallVector<TileBox> boxes = tiles(*slicingOf(source));
  FailureOr<int64_t> base = context.storageAddress(source);
  if (failed(base))
    return unsupported(op, "a store from an unplaced narrow block");
  Index hostStrides = strides(host.view, elem);
  struct Plan {
    int64_t tile;
    Index lo;
    SmallVector<Loop> loops;
    SmallVector<int64_t> chunks;
  };
  SmallVector<Plan> plans;
  for (const auto &[tile, box] : boxes) {
    Index lo = tail(box.lo, host.view.size());
    Index hi = tail(box.hi, host.view.size());
    Index count = extent(Box{lo, hi});
    Index narrow = strides(count, elem);
    SmallVector<Loop> loops;
    for (size_t dim = 0; dim < count.size(); ++dim)
      loops.push_back({count[dim], hostStrides[dim], narrow[dim]});
    loops = mergeLoops(loops);
    plans.push_back(
        {tile, lo, loops, chunkOffsets(loops, dot(lo, hostStrides))});
  }
  SmallVector<std::pair<int64_t, int64_t>> order;
  for (const Plan &plan : plans)
    for (int64_t offset : plan.chunks)
      order.push_back({offset, plan.tile});
  llvm::sort(order);
  llvm::DenseMap<int64_t, int64_t> index;
  for (auto [position, entry] : llvm::enumerate(order))
    index.try_emplace(entry.second, position);
  std::optional<size_t> varying;
  for (size_t dim = 0; dim < host.view.size(); ++dim) {
    llvm::DenseSet<int64_t> starts;
    for (const Plan &plan : plans)
      starts.insert(plan.lo[dim]);
    if (starts.size() > 1)
      varying = dim;
  }
  int64_t cycle = 1;
  if (varying) {
    llvm::DenseSet<int64_t> starts;
    for (const Plan &plan : plans)
      starts.insert(plan.lo[*varying]);
    cycle = starts.size();
  }
  std::array<bool, 16> active = tilesOf(boxes);
  SmallVector<Emitted, 0> out;
  for (const Plan &plan : plans)
    out.push_back(ringProducer(*base, plan.loops, plan.tile, index[plan.tile],
                               cycle, channels, targets(active, true)));
  RingConsumer drain;
  drain.traversal.syncProducer = {producerSync(false)};
  drain.traversal.byteAddressMode = access(kAccess);
  drain.virtualChannelSubscription = channels;
  drain.threadMulticastBitmap = bits<4>("1000");
  drain.filter.firstDiscardByteLoopMap = 1;
  drain.filter.firstDiscardByteCount = total;
  out.push_back(dma(drain, active));
  FailureOr<int64_t> offset = context.hostOffset(op);
  if (failed(offset))
    return unsupported(op, "a store into an unplaced host buffer");
  int64_t hib =
      context.hib(DmaQueue::Output, HibRoot::Scratch,
                  *offset + byteOffset(host.offset, host.buffer, elem));
  out.push_back(
      hibScatter(host.view, host.buffer, elem, DmaQueue::Output, hib));
  out.push_back(outfeed(total, channels));
  llvm::append_range(out, groupFences());
  return out;
}

namespace {

SmallVector<TileBox> heldTiles(Operation *op) {
  SmallVector<TileBox> out = clampedTiles(op);
  llvm::erase_if(out, [](const TileBox &entry) {
    return llvm::any_of(extent(entry.box),
                        [](int64_t count) { return count <= 0; });
  });
  return out;
}

} // namespace

Body codegen::modelOutput(Operation *op, Context &context) {
  Operation *source = producer(op, 0);
  FailureOr<int64_t> base = context.storageAddress(source);
  if (failed(base))
    return unsupported(op, "a model output from an unplaced narrow block");
  Operation *view = op->getNumOperands() > 1 ? producer(op, 1) : nullptr;
  SmallVector<int64_t, 4> full =
      view ? resultInfo(producer(view, 0)).shape : resultInfo(op).shape;
  SmallVector<int64_t, 4> strides(full.size(), 1);
  for (int64_t dim = full.size() - 2; dim >= 0; --dim)
    strides[dim] = strides[dim + 1] * full[dim + 1];
  std::array<bool, 8> channels = bits<8>("10000000");
  int64_t elem = resultInfo(op).elementBytes;
  SmallVector<TileBox> boxes = heldTiles(source);
  SmallVector<Emitted, 0> out;
  for (auto [order, entry] : llvm::enumerate(boxes)) {
    Index size = extent(entry.box);
    Index low = tail(entry.box.lo, full.size());
    if (view)
      low = applyReverse(view, low);
    size = tail(size, full.size());
    int64_t total = product(size) * elem;
    int64_t split = full.size() - 1;
    while (split >= 0 && size[split] == full[split])
      --split;
    int64_t run = split < 0 ? total : size[split] * strides[split] * elem;
    SmallVector<Counter> loops{counter(0, run)};
    for (int64_t dim = split - 1; dim >= 0; --dim)
      if (size[dim] > 1)
        loops.push_back(counter((size[dim] - 1) * strides[dim] * elem,
                                strides[dim] * elem));
    int64_t offset = 0;
    for (auto [position, stride] : llvm::zip_equal(low, strides))
      offset += position * stride * elem;
    RingProducer producer;
    producer.traversal.baseAddress = *base;
    producer.traversal.counter =
        padded({counter(llvm::alignTo(total, kAccess) - kAccess, kAccess)}, 4);
    producer.traversal.syncProducer = {producerSync(true),
                                       producerSync(true, 1)};
    producer.traversal.byteAddressMode = access(std::min(total, kAccess));
    if (boxes.size() > 1)
      producer.watchers = {
          dmaWatcher(tileWatcher(TileSyncFlag::RingBusProducerA, order, 0))};
    producer.virtualChannelSubscription = channels;
    producer.targets = targets({}, true);
    out.push_back(dma(producer, tileBit(entry.tile)));
    HibDma hib;
    hib.queue = DmaQueue::Output;
    hib.traversal.baseAddress =
        context.hib(DmaQueue::Output, HibRoot::OutputActivation, offset);
    hib.traversal.counter = padded(loops, 4);
    hib.traversal.byteAddressMode = access(run);
    out.push_back(tagged(hib));
    out.push_back(outfeed(total, channels));
    if (boxes.size() == 1)
      continue;
    ScalarFence fence;
    fence.expectedScalarSyncFlag[5] = 2 * order + 1;
    fence.waitIdle = bits<5>("00010");
    fence.incrementGlobalTokenA = true;
    out.push_back(scalarFence(fence));
  }
  llvm::append_range(out, groupFences());
  return out;
}

Body codegen::fill(Operation *op, Context &context) {
  TensorInfo info = resultInfo(op);
  int64_t total = product(info.shape) * info.elementBytes;
  std::array<bool, 16> every = tilesOf(tiles(*slicingOf(op)));
  std::array<bool, 8> channels = bits<8>("01000000");
  SmallVector<Emitted, 0> out = groupFences();
  out.push_back(hibGather(
      info.shape, info.shape, info.elementBytes, DmaQueue::Parameter,
      context.hib(DmaQueue::Parameter, HibRoot::ParameterFill, 0, total, op)));
  llvm::append_range(out, infeed(total, channels, targets(every, false),
                                 InputFifo::Parameter));
  FailureOr<int64_t> base = context.narrowAddress(op);
  if (failed(base))
    return unsupported(op, "a fill into an unplaced narrow block");
  Loop loop{total / info.elementBytes, info.elementBytes, info.elementBytes};
  out.push_back(ringConsumer(*base, loop, 0, 0, channels, every));
  llvm::append_range(out, groupFences());
  return out;
}

Body codegen::padding(Operation *op, Context &context) {
  TensorInfo info = resultInfo(op);
  ArrayRef<int64_t> destination = info.shape;
  int64_t elem = info.elementBytes;
  SmallVector<int64_t, 4> sourceShape = operandInfo(op, 0).shape;
  Index low = applyForward(op, Index(sourceShape.size(), 0));
  Index lastPoint;
  for (int64_t size : sourceShape)
    lastPoint.push_back(size - 1);
  Index high = applyForward(op, lastPoint);
  FailureOr<int64_t> base =
      sourceSpace(op) == DistributedMemorySpace::TileMemory
          ? context.storageAddress(op)
          : context.narrowAddress(op);
  if (failed(base))
    return unsupported(op, "padding of an unplaced narrow block");
  SmallVector<TileBox> boxes = heldTiles(op);
  Index size = extent(boxes.front().box);
  Index stride = strides(size, elem);
  SmallVector<std::pair<Index, std::array<bool, 16>>> shapes;
  constexpr int rows = 1, cols = 2;
  auto padOnly = [&](const Box &box) {
    return box.hi[rows] < low[rows] || box.lo[rows] > high[rows];
  };
  for (const auto &[tile, box] : boxes) {
    if (padOnly(box))
      continue;
    Index held = extent(box);
    auto *same = llvm::find_if(
        shapes, [&](const auto &entry) { return entry.first == held; });
    if (same == shapes.end())
      same = &shapes.emplace_back(held, std::array<bool, 16>{});
    same->second[tile] = true;
  }
  SmallVector<Emitted, 0> out;
  for (const auto &[held, group] : shapes)
    for (bool low_side : {true, false}) {
      int64_t first = low_side ? 0 : high[cols] + 1;
      int64_t last = low_side ? low[cols] - 1 : destination[cols] - 1;
      if (first <= last)
        out.push_back(zeroFill(
            *base + first * stride[cols], (last - first + 1) * stride[cols],
            {{held[rows], stride[rows]}}, padDirection(cols, low_side), group,
            elem, true));
    }
  int64_t run = (high[cols] - low[cols] + 1) * stride[cols];
  for (bool low_side : {true, false}) {
    int64_t first = low_side ? 0 : high[rows] + 1;
    int64_t last = low_side ? low[rows] - 1 : destination[rows] - 1;
    for (const auto &[tile, box] : boxes) {
      Index lo = tail(box.lo, destination.size());
      Index hi = tail(box.hi, destination.size());
      int64_t from = std::max(first, lo[rows]), to = std::min(last, hi[rows]);
      if (from > to)
        continue;
      bool whole = padOnly(box);
      MeshDirection direction = low[rows] > 0 && !whole
                                    ? padDirection(rows, low_side)
                                    : padDirection(cols, true);
      SmallVector<std::pair<int64_t, int64_t>> outer;
      if (to > from)
        outer.push_back({to - from + 1, stride[rows]});
      out.push_back(zeroFill(*base + (from - lo[rows]) * stride[rows] +
                                 (whole ? 0 : low[cols] * stride[cols]),
                             whole ? stride[rows] : run, outer, direction,
                             tileBit(tile), elem, to > from));
    }
  }
  return out;
}
