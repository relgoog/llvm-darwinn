#include "Families.h"

using namespace mlir;
using namespace mlir::darwinn;
using namespace mlir::darwinn::codegen;

namespace {

bool sameCounters(ArrayRef<Counter> a, ArrayRef<Counter> b) {
  return llvm::equal(a, b, [](const Counter &x, const Counter &y) {
    return x.end == y.end && x.step == y.step && x.mask == y.mask;
  });
}

bool sameConsumer(const RingConsumer &a, const RingConsumer &b) {
  const Traversal &x = a.traversal, &y = b.traversal;
  return sameCounters(x.counter, y.counter) &&
         llvm::equal(x.syncProducer, y.syncProducer,
                     [](const SyncProducer &p, const SyncProducer &q) {
                       return p.increment == q.increment &&
                              p.syncFlagLoopDepth == q.syncFlagLoopDepth;
                     }) &&
         a.filter.firstDiscardByteLoopMap == b.filter.firstDiscardByteLoopMap &&
         a.filter.firstDiscardByteCount == b.filter.firstDiscardByteCount &&
         a.filter.lastDiscardByteCount == b.filter.lastDiscardByteCount;
}

FailureOr<std::pair<int64_t, int64_t>>
linearRange(const Box &box, ArrayRef<int64_t> shape, int64_t elementBytes) {
  Index size = extent(box);
  size_t split = 0;
  while (split < size.size() && size[split] == 1)
    ++split;
  for (size_t dim = split + 1; dim < size.size(); ++dim)
    if (size[dim] != shape[dim])
      return failure();
  return std::make_pair(dot(box.lo, strides(shape, elementBytes)),
                        product(size) * elementBytes);
}

} // namespace

SmallVector<TileBox> codegen::clampedTiles(Operation *op) {
  SmallVector<int64_t, 4> shape = resultInfo(op).shape;
  Operation *target = op;
  if (isa<TensorOpOp>(op))
    target = throughViews(producer(op, 2));
  else if (isa<UnaryTensorOpOp>(op))
    target = throughViews(producer(op, 1));
  SmallVector<TileBox> out = tiles(*slicingOf(target));
  for (TileBox &entry : out) {
    entry.box.lo = tail(entry.box.lo, shape.size());
    entry.box.hi = tail(entry.box.hi, shape.size());
    for (size_t dim = 0; dim < shape.size(); ++dim) {
      entry.box.lo[dim] = std::max<int64_t>(0, entry.box.lo[dim]);
      entry.box.hi[dim] = std::min(shape[dim] - 1, entry.box.hi[dim]);
    }
  }
  return out;
}

Body codegen::scatter(Operation *op, Context &context) {
  Operation *owner = storage(producer(op, 0));
  TensorInfo info = resultInfo(op);
  int64_t elem = info.elementBytes;
  size_t rank = info.shape.size();
  SmallVector<TileBox> sources = clampedTiles(owner);
  for (TileBox &source : sources) {
    source.box.lo = applyForward(op, source.box.lo);
    source.box.hi = applyForward(op, source.box.hi);
  }
  SmallVector<TileBox> destinations = clampedTiles(op);
  Index sourceExtent = tail(extent(ownerTileBox(owner)), rank);
  Index destinationExtent = tail(extent(unionBox(*slicingOf(op), 0, 0)), rank);
  Index sourceStrides = strides(sourceExtent, elem);
  Index destinationStrides = strides(destinationExtent, elem);
  FailureOr<int64_t> sourceBase = context.storageAddress(owner);
  FailureOr<int64_t> destinationBase = context.narrowAddress(op, Suffix::Dest);
  if (failed(sourceBase) || failed(destinationBase))
    return unsupported(op, "a scatter between unplaced narrow blocks");
  int64_t total = product(sourceExtent) * elem;
  llvm::sort(sources, [](const TileBox &a, const TileBox &b) {
    return a.tile < b.tile;
  });
  llvm::sort(destinations, [](const TileBox &a, const TileBox &b) {
    return a.tile < b.tile;
  });

  SmallVector<Emitted, 0> out;
  for (auto [order, source] : llvm::enumerate(sources)) {
    bool ringA = order % 2 == 0;
    std::array<bool, 8> channels =
        ringA ? bits<8>("01000000") : bits<8>("00000001");
    RingProducer producer;
    producer.traversal.baseAddress = *sourceBase;
    producer.traversal.counter = padded({counter(total - kAccess, kAccess)}, 4);
    producer.traversal.syncProducer = {producerSync(true),
                                       producerSync(true, 1)};
    producer.traversal.byteAddressMode = access(kAccess);
    producer.watchers = {dmaWatcher(tileWatcher(
        ringA ? TileSyncFlag::RingBusProducerA : TileSyncFlag::RingBusProducerB,
        order / 2, 0))};
    producer.virtualChannelSubscription = channels;
    producer.targets = targets(bits<16>("1111111111111111"), false);
    out.push_back(dma(producer, tileBit(source.tile)));

    struct Member {
      int64_t tile;
      int64_t base;
    };
    SmallVector<std::pair<RingConsumer, SmallVector<Member>>, 0> shared;
    SmallVector<int64_t> idle;
    auto consumerFor = [&]() {
      RingConsumer consumer;
      consumer.consumer =
          ringA ? RingConsumerId::ConsumerA : RingConsumerId::ConsumerB;
      consumer.virtualChannelSubscription = channels;
      consumer.threadMulticastBitmap = bits<4>("1000");
      return consumer;
    };
    for (const TileBox &destination : destinations) {
      Index lo, hi;
      bool empty = false;
      for (size_t dim = 0; dim < rank; ++dim) {
        lo.push_back(std::max(source.box.lo[dim], destination.box.lo[dim]));
        hi.push_back(std::min(source.box.hi[dim], destination.box.hi[dim]));
        empty |= hi.back() < lo.back();
      }
      if (empty) {
        idle.push_back(destination.tile);
        continue;
      }
      Index counts = extent(Box{lo, hi});
      Index sourceOffset, destinationOffset;
      for (size_t dim = 0; dim < rank; ++dim) {
        sourceOffset.push_back(lo[dim] - source.box.lo[dim]);
        destinationOffset.push_back(lo[dim] - destination.box.lo[dim]);
      }
      SmallVector<Loop> loops;
      for (size_t dim = 0; dim < rank; ++dim)
        loops.push_back(
            {counts[dim], sourceStrides[dim], destinationStrides[dim]});
      loops = mergeLoops(loops);
      int64_t first = dot(sourceOffset, sourceStrides);
      int64_t lastByte = elem;
      for (size_t dim = 0; dim < rank; ++dim)
        lastByte += (sourceOffset[dim] + counts[dim] - 1) * sourceStrides[dim];
      int64_t inner = loops.back().count * loops.back().destination;
      SmallVector<Counter> items{counter(inner - kAccess, kAccess)};
      for (const Loop &loop : llvm::reverse(ArrayRef(loops).drop_back()))
        items.push_back(
            counter((loop.count - 1) * loop.destination, loop.destination));
      RingConsumer consumer = consumerFor();
      consumer.traversal.counter = padded(items, 4);
      consumer.traversal.syncProducer = {producerSync(true, items.size())};
      consumer.traversal.byteAddressMode = access(kAccess);
      consumer.filter.firstDiscardByteLoopMap = (1u << items.size()) - 1;
      consumer.filter.firstDiscardByteCount = first;
      consumer.filter.lastDiscardByteCount = total - lastByte;
      Member member{destination.tile,
                    *destinationBase +
                        dot(destinationOffset, destinationStrides)};
      auto match = llvm::find_if(shared, [&](const auto &entry) {
        return sameConsumer(entry.first, consumer);
      });
      if (match == shared.end())
        shared.push_back({consumer, {member}});
      else
        match->second.push_back(member);
    }

    SmallVector<std::pair<int64_t, SmallVector<Emitted, 0>>> consumers;
    for (auto &[consumer, members] : shared) {
      if (members.size() == 1) {
        consumer.traversal.baseAddress = members.front().base;
        consumers.push_back({members.front().tile,
                             {dma(consumer, tileBit(members.front().tile))}});
        continue;
      }
      SmallVector<Emitted, 0> emitted;
      SmallVector<int64_t> memberTiles;
      for (const Member &member : llvm::reverse(members))
        emitted.push_back(load(member.base, tileBit(member.tile), 0));
      for (const Member &member : members)
        memberTiles.push_back(member.tile);
      consumer.traversal.baseAddress = 0;
      emitted.push_back(
          dma(consumer, tileSet(memberTiles), bits<8>("10000000")));
      consumers.push_back({members.front().tile, std::move(emitted)});
    }
    for (int64_t tile = 0; tile < kTiles; ++tile)
      if (llvm::none_of(destinations, [&](const TileBox &destination) {
            return destination.tile == tile;
          }))
        idle.push_back(tile);
    if (!idle.empty()) {
      RingConsumer discard = consumerFor();
      discard.traversal.syncProducer = {producerSync(false)};
      discard.traversal.byteAddressMode = access(kAccess);
      discard.filter.firstDiscardByteLoopMap = 1;
      discard.filter.firstDiscardByteCount = total;
      consumers.push_back(
          {*llvm::min_element(idle), {dma(discard, tileSet(idle))}});
    }
    llvm::stable_sort(consumers, [](const auto &a, const auto &b) {
      return a.first < b.first;
    });
    for (auto &[tile, emitted] : consumers)
      llvm::append_range(out, emitted);
  }
  llvm::append_range(out, groupFences());
  return out;
}

Body codegen::ringReshape(Operation *op, Context &context) {
  Operation *owner = storage(producer(op, 0));
  TensorInfo info = resultInfo(op);
  SmallVector<TileBox> sources = clampedTiles(owner);
  llvm::sort(sources, [](const TileBox &a, const TileBox &b) {
    return a.tile < b.tile;
  });
  int64_t expected = 0;
  SmallVector<int64_t> sizes;
  for (const TileBox &source : sources) {
    auto range = linearRange(source.box, info.shape, info.elementBytes);
    if (failed(range) || range->first != expected)
      return unsupported(op, "a ring reshape whose source tiles are not "
                             "contiguous in row major order");
    expected += range->second;
    sizes.push_back(range->second);
  }
  int64_t total = expected;
  FailureOr<int64_t> sourceBase = context.storageAddress(owner);
  FailureOr<int64_t> destinationBase = context.narrowAddress(op, Suffix::Dest);
  if (failed(sourceBase) || failed(destinationBase))
    return unsupported(op, "a ring reshape between unplaced narrow blocks");
  std::array<bool, 8> channels = bits<8>("01000000");
  SmallVector<Emitted, 0> out;
  for (auto [order, source] : llvm::enumerate(sources)) {
    int64_t size = sizes[order];
    RingProducer producer;
    producer.traversal.baseAddress = *sourceBase;
    producer.traversal.counter = padded({counter(0, size)}, 4);
    producer.traversal.syncProducer = {producerSync(true), producerSync(true)};
    producer.traversal.byteAddressMode = access(size);
    producer.watchers = {dmaWatcher(
        tileWatcher(TileSyncFlag::RingBusProducerA, order, sources.size()))};
    producer.virtualChannelSubscription = channels;
    producer.targets = targets(bits<16>("1111111111111111"), false);
    out.push_back(dma(producer, tileBit(source.tile)));
  }
  SmallVector<TileBox> destinations = clampedTiles(op);
  llvm::sort(destinations, [](const TileBox &a, const TileBox &b) {
    return a.tile < b.tile;
  });
  for (const TileBox &destination : destinations) {
    auto range = linearRange(destination.box, info.shape, info.elementBytes);
    if (failed(range))
      return unsupported(op, "a ring reshape whose destination tiles are not "
                             "contiguous in row major order");
    auto [start, size] = *range;
    RingConsumer consumer;
    consumer.traversal.baseAddress = *destinationBase;
    consumer.traversal.counter = padded({counter(size - kAccess, kAccess)}, 4);
    consumer.traversal.syncProducer = {producerSync(true)};
    consumer.traversal.byteAddressMode = access(kAccess);
    consumer.virtualChannelSubscription = channels;
    consumer.threadMulticastBitmap = bits<4>("1000");
    consumer.filter.firstDiscardByteLoopMap = 1;
    consumer.filter.firstDiscardByteCount = start;
    consumer.filter.lastDiscardByteCount = total - start - size;
    out.push_back(dma(consumer, tileBit(destination.tile)));
  }
  llvm::append_range(out, groupFences());
  return out;
}
