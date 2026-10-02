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

namespace {

struct ScatterGeometry {
  SmallVector<TileBox> sources;
  SmallVector<TileBox> destinations;
  Index sourceExtent;
  Index sourceStrides;
  Index destinationStrides;
  int64_t elem;
  int64_t total;
};

ScatterGeometry scatterGeometry(Operation *op) {
  Operation *owner = storage(producer(op, 0));
  TensorInfo info = resultInfo(op);
  size_t rank = info.shape.size();
  ScatterGeometry g;
  g.elem = info.elementBytes;
  g.sources = clampedTiles(owner);
  for (TileBox &source : g.sources) {
    source.box.lo = applyForward(op, source.box.lo);
    source.box.hi = applyForward(op, source.box.hi);
  }
  g.destinations = clampedTiles(op);
  g.sourceExtent = tail(extent(ownerTileBox(owner)), rank);
  g.sourceStrides = strides(g.sourceExtent, g.elem);
  g.destinationStrides =
      strides(tail(extent(unionBox(*slicingOf(op), 0, 0)), rank), g.elem);
  g.total = product(g.sourceExtent) * g.elem;
  auto byTile = [](const TileBox &a, const TileBox &b) {
    return a.tile < b.tile;
  };
  llvm::sort(g.sources, byTile);
  llvm::sort(g.destinations, byTile);
  return g;
}

constexpr int64_t kRelayBytes = 1024;

struct Relay {
  int64_t tile;
  int64_t source;
  int64_t destination;
  int64_t rows;
  int64_t rowWide;
  int64_t rowStride;
  int64_t readAccess;
  int64_t readStride;
  int64_t readCount;
};

FailureOr<SmallVector<Relay>> relays(const ScatterGeometry &g,
                                     const TileBox &source) {
  size_t rank = g.sourceExtent.size();
  if (rank < 3)
    return failure();
  size_t rowDim = rank - 3;
  int64_t rowBytes =
      product(ArrayRef(g.sourceExtent).drop_front(rowDim + 1)) * g.elem;
  SmallVector<Relay> out;
  for (const TileBox &destination : g.destinations) {
    Index lo, hi, sourceOffset, destinationOffset;
    bool empty = false;
    for (size_t dim = 0; dim < rank; ++dim) {
      lo.push_back(std::max(source.box.lo[dim], destination.box.lo[dim]));
      hi.push_back(std::min(source.box.hi[dim], destination.box.hi[dim]));
      empty |= hi.back() < lo.back();
      sourceOffset.push_back(lo[dim] - source.box.lo[dim]);
      destinationOffset.push_back(lo[dim] - destination.box.lo[dim]);
    }
    if (empty)
      continue;

    Index size = extent(Box{lo, hi});
    if (size.back() < g.sourceExtent.back()) {
      int64_t readAccess = size.back() * g.elem;
      int64_t readStride = g.sourceExtent.back() * g.elem;
      if (readAccess > 128 || destinationOffset.back() != 0 ||
          g.destinationStrides[rank - 2] != readAccess)
        return failure();

      int64_t readCount = 1;
      for (size_t dim = rank - 1; dim-- > 0;) {
        if (size[dim] > 1 &&
            (g.sourceStrides[dim] != readCount * readStride ||
             g.destinationStrides[dim] != readCount * readAccess))
          return failure();
        readCount *= size[dim];
      }

      int64_t packedBytes = readCount * readAccess;
      if (packedBytes % 128)
        return failure();

      out.push_back({destination.tile, dot(sourceOffset, g.sourceStrides),
                     dot(destinationOffset, g.destinationStrides), 1,
                     packedBytes / 128, packedBytes, readAccess, readStride,
                     readCount});
      continue;
    }

    if (rowBytes % 128)
      return failure();
    for (size_t dim = 0; dim < rank; ++dim)
      if (dim != rowDim && (hi[dim] - lo[dim] + 1 != g.sourceExtent[dim] ||
                            (dim > rowDim + 1 && g.destinationStrides[dim] !=
                                                     g.sourceStrides[dim])))
        return failure();
    int64_t rows = size[rowDim];
    out.push_back({destination.tile, dot(sourceOffset, g.sourceStrides),
                   dot(destinationOffset, g.destinationStrides), rows,
                   rowBytes / 128, g.destinationStrides[rowDim], 128, 128,
                   rows * rowBytes / 128});
  }
  return out;
}

int64_t bundleBytes(const ScatterGeometry &g) {
  return g.sourceExtent.back() * g.elem >= 32 ? 32 : 16;
}

std::pair<int64_t, int64_t> widePlan(int64_t wideRows, int64_t bundle) {
  int64_t perWideRow = 128 / bundle;
  if (wideRows < std::min<int64_t>(16, bundle * bundle / 32))
    return {wideRows, 1};
  int64_t inner = 1;
  while (inner * perWideRow < 12 || wideRows % inner)
    ++inner;
  return {inner, wideRows / inner};
}

Emitted ringProducer(size_t order, int64_t tile, int64_t base, int64_t total) {
  bool ringA = order % 2 == 0;
  RingProducer producer;
  producer.traversal.baseAddress = base;
  producer.traversal.counter = padded({counter(total - kAccess, kAccess)}, 4);
  producer.traversal.syncProducer = {producerSync(true), producerSync(true, 1)};
  producer.traversal.byteAddressMode = access(kAccess);
  producer.watchers = {dmaWatcher(tileWatcher(
      ringA ? TileSyncFlag::RingBusProducerA : TileSyncFlag::RingBusProducerB,
      order / 2, 0))};
  producer.virtualChannelSubscription =
      ringA ? bits<8>("01000000") : bits<8>("00000001");
  producer.targets = targets(bits<16>("1111111111111111"), false);
  return dma(producer, tileBit(tile));
}

Body relayScatter(Operation *op, Context &context, const ScatterGeometry &g) {
  FailureOr<int64_t> sourceBase =
      context.storageAddress(storage(producer(op, 0)));
  FailureOr<int64_t> destinationBase = context.narrowAddress(op, Suffix::Dest);
  FailureOr<int64_t> stageA = context.narrowAddress(op, Suffix::StageA);
  FailureOr<int64_t> stageB = context.narrowAddress(op, Suffix::StageB);
  FailureOr<int64_t> wide = context.wideAddress(op, Suffix::Relay);
  if (failed(sourceBase) || failed(destinationBase) || failed(stageA) ||
      failed(stageB) || failed(wide))
    return unsupported(op, "a relayed scatter between unplaced blocks");
  int64_t bundle = bundleBytes(g);
  int64_t perWideRow = 128 / bundle;
  std::map<std::pair<int64_t, bool>, int64_t> wideWrites, narrowWrites;
  SmallVector<Emitted, 0> out;
  for (auto [order, source] : llvm::enumerate(g.sources)) {
    bool ringA = order % 2 == 0;
    out.push_back(ringProducer(order, source.tile, *sourceBase, g.total));
    FailureOr<SmallVector<Relay>> copies = relays(g, source);
    if (failed(copies))
      return unsupported(op, "a relayed scatter that splits source rows");
    int64_t staging = ringA ? *stageA : *stageB;
    int64_t thread = ringA ? 0 : 1;

    auto consumerFor = [&](int64_t written) {
      RingConsumer consumer;
      consumer.consumer =
          ringA ? RingConsumerId::ConsumerA : RingConsumerId::ConsumerB;
      consumer.virtualChannelSubscription =
          ringA ? bits<8>("01000000") : bits<8>("00000001");
      consumer.threadMulticastBitmap = bits<4>("1000");
      consumer.traversal.baseAddress = staging;
      consumer.traversal.counter =
          padded({counter(g.total - kAccess, kAccess)}, 4);
      consumer.traversal.syncProducer = {producerSync(true, written ? 2 : 1)};
      consumer.traversal.byteAddressMode = access(kAccess);
      consumer.filter.firstDiscardByteLoopMap = 1;
      if (written)
        consumer.watchers = {
            dmaWatcher(tileWatcher(TileSyncFlag::WideToNarrowWrite, written, 0,
                                   1, false, thread),
                       true)};
      return consumer;
    };

    using Key = std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                           int64_t, int64_t>;
    struct Group {
      Key key;
      SmallVector<Relay> members;
      SmallVector<int64_t> attached;
    };
    SmallVector<Group> groups;
    std::map<int64_t, size_t> memberGroup;
    for (const Relay &copy : *copies) {
      Key key{narrowWrites[{copy.tile, ringA}],
              copy.rows,
              copy.rowWide,
              copy.rowStride,
              wideWrites[{copy.tile, ringA}],
              copy.readAccess,
              copy.readStride,
              copy.readCount};
      auto found = llvm::find_if(
          groups, [&](const Group &group) { return group.key == key; });
      if (found == groups.end())
        found = groups.insert(groups.end(), Group{key, {}, {}});
      found->members.push_back(copy);
      memberGroup[copy.tile] = found - groups.begin();
    }
    SmallVector<std::pair<int64_t, SmallVector<int64_t>>> lone;
    for (int64_t tile = 0; tile < kTiles; ++tile) {
      if (memberGroup.count(tile))
        continue;
      int64_t written = narrowWrites[{tile, ringA}];
      auto previous = memberGroup.lower_bound(tile);
      if (previous != memberGroup.begin()) {
        Group &group = groups[std::prev(previous)->second];
        if (std::get<0>(group.key) == written) {
          group.attached.push_back(tile);
          continue;
        }
      }
      auto found = llvm::find_if(
          lone, [&](const auto &entry) { return entry.first == written; });
      if (found == lone.end())
        lone.push_back({written, {tile}});
      else
        found->second.push_back(tile);
    }

    SmallVector<std::pair<int64_t, SmallVector<Emitted, 0>>> emitted;
    for (const auto &[written, tiles] : lone)
      emitted.push_back(
          {tiles.front(), {dma(consumerFor(written), tileSet(tiles))}});
    for (Group &group : groups) {
      auto [written, rows, rowWide, rowStride, wideBefore, readAccess,
            readStride, readCount] = group.key;
      SmallVector<int64_t> receivers;
      for (const Relay &copy : group.members)
        receivers.push_back(copy.tile);
      llvm::append_range(receivers, group.attached);
      SmallVector<Emitted, 0> body{
          dma(consumerFor(written), tileSet(receivers))};
      receivers.resize(group.members.size());

      int64_t wideRows = rows * rowWide;
      auto [inner, outer] = widePlan(wideRows, bundle);
      bool wideLoop = inner > 1;
      SmallVector<Counter> wideItems;
      if (wideLoop || outer == 1)
        wideItems.push_back(counter(inner - 1, 1));
      if (outer > 1)
        wideItems.push_back(counter(outer - 1, 1, false));
      int64_t step = bundle / 4;
      SmallVector<Counter> writeItems{counter((perWideRow - 1) * step, step)};
      SmallVector<int64_t> counts{perWideRow};
      auto push = [&](int64_t count, int64_t words) {
        if (count == 1)
          return;
        writeItems.push_back(counter((count - 1) * words, words));
        counts.push_back(count);
      };
      if (rowStride == rowWide * 128) {
        push(wideRows, 32);
      } else {
        push(rowWide, 32);
        push(rows, rowStride / 4);
      }
      int64_t depth = writeItems.size() - 1;
      int64_t perIncrement = counts.size() > 2 ? rowWide : 1;

      NarrowToWide gather;
      gather.read.counter =
          padded({counter((readCount - 1) * readStride, readStride)}, 6);
      gather.read.byteAddressMode = access(readAccess);
      gather.readWatchers = {dmaWatcher(tileWatcher(
          ringA ? TileSyncFlag::RingBusReadA : TileSyncFlag::RingBusReadB,
          order / 2 + 1, 1, 1))};
      gather.write.baseAddress = *wide;
      gather.write.counter = padded(wideItems, 4);
      gather.write.syncProducer = {producerSync(true, 1)};
      gather.write.byteAddressMode = access(128);
      if (outer > 1) {
        gather.write.doubleBufferLoop = wideLoop;
        gather.write.secondBufferOffset = inner;
        gather.writeWatchers = {
            dmaWatcher(tileWatcher(TileSyncFlag::WideToNarrowWrite,
                                   (int32_t(1) << 25) - inner / perIncrement,
                                   inner / perIncrement, 1, true),
                       true)};
      }
      gather.threadMulticastBitmap = threadBit(thread);
      gather.byteAddress.strideUnitGranulesLoopMap = 1;
      gather.byteAddress.defaultStrideUnitGranules = 2;
      gather.byteAddress.lastStrideUnitGranules = 2;
      gather.byteAddress.cellStride = 32;
      gather.byteAddress.defaultCellStrideGroupCount = 1;
      gather.byteAddress.lastCellStrideGroupCount = 1;

      WideToNarrow scatter;
      scatter.read.baseAddress = *wide;
      scatter.read.counter = padded(wideItems, 4);
      if (outer > 1) {
        scatter.read.doubleBufferLoop = wideLoop;
        scatter.read.secondBufferOffset = inner;
      }
      scatter.write.counter = padded(writeItems, 6);
      scatter.write.syncProducer = {producerSync(true, depth)};
      scatter.readWatchers = {dmaWatcher(tileWatcher(
          TileSyncFlag::NarrowToWideWrite, wideBefore + 1, 1, 1, true))};
      scatter.wideMemoryLoadStoreLoopId = 1;
      scatter.zInBundleValidCount = bundle / 4;
      scatter.threadMulticastBitmap = threadBit(thread);

      TileValues reads, writes;
      for (const Relay &copy : group.members) {
        reads.push_back({copy.tile, staging + copy.source});
        writes.push_back(
            {copy.tile, (*destinationBase + copy.destination) / 4});
      }
      auto sourced = [&](const TileValues &values, uint8_t reg,
                         uint64_t &base) -> SmallVector<Emitted, 0> {
        bool same = llvm::all_of(values, [&](const auto &value) {
          return value.second == values.front().second;
        });
        if (same) {
          base = values.front().second;
          return {};
        }
        Order order = latestFirst;
        if (std::get<5>(group.key) < std::get<6>(group.key))
          order = [](Groups groups) {
            std::rotate(groups.begin(), std::next(groups.begin()),
                        groups.end());
            return groups;
          };
        return registerLoads(values, reg, order, thread);
      };
      SmallVector<Emitted, 0> loads =
          sourced(reads, 0, gather.read.baseAddress);
      std::array<bool, 8> registers{};
      registers[0] = !loads.empty();
      llvm::append_range(body, loads);
      body.push_back(dma(gather, tileSet(receivers), registers));
      loads = sourced(writes, 1, scatter.write.baseAddress);
      registers = {};
      registers[1] = !loads.empty();
      llvm::append_range(body, loads);
      body.push_back(dma(scatter, tileSet(receivers), registers));
      emitted.push_back({group.members.front().tile, std::move(body)});

      for (const Relay &copy : group.members) {
        wideWrites[{copy.tile, ringA}] += outer;
        narrowWrites[{copy.tile, ringA}] += depth ? counts.back() : perWideRow;
      }
    }
    llvm::stable_sort(emitted, [](const auto &a, const auto &b) {
      return a.first < b.first;
    });
    for (auto &[tile, body] : emitted)
      llvm::append_range(out, body);
  }
  llvm::append_range(out, groupFences());
  return out;
}

} // namespace

int64_t codegen::scatterStagingWords(Operation *op) {
  int64_t total = scatterGeometry(op).total;
  return total <= kRelayBytes ? total / 4 : 0;
}

int64_t codegen::scatterRelayWords(Operation *op) {
  ScatterGeometry g = scatterGeometry(op);
  if (g.total > kRelayBytes)
    return 0;
  int64_t words = 0;
  for (const TileBox &source : g.sources) {
    FailureOr<SmallVector<Relay>> copies = relays(g, source);
    if (failed(copies))
      return 0;
    for (const Relay &copy : *copies) {
      int64_t wideRows = copy.readAccess * copy.readCount / 128;
      auto [inner, outer] = widePlan(wideRows, bundleBytes(g));
      words = std::max(words, outer > 1 ? 2 * inner : inner);
    }
  }
  return words;
}

Body codegen::scatter(Operation *op, Context &context) {
  ScatterGeometry g = scatterGeometry(op);
  if (g.total <= kRelayBytes)
    return relayScatter(op, context, g);
  size_t rank = g.sourceExtent.size();
  const Index &sourceStrides = g.sourceStrides;
  const Index &destinationStrides = g.destinationStrides;
  int64_t elem = g.elem, total = g.total;
  const SmallVector<TileBox> &sources = g.sources;
  const SmallVector<TileBox> &destinations = g.destinations;
  FailureOr<int64_t> sourceBase =
      context.storageAddress(storage(producer(op, 0)));
  FailureOr<int64_t> destinationBase = context.narrowAddress(op, Suffix::Dest);
  if (failed(sourceBase) || failed(destinationBase))
    return unsupported(op, "a scatter between unplaced narrow blocks");

  SmallVector<Emitted, 0> out;
  for (auto [order, source] : llvm::enumerate(sources)) {
    bool ringA = order % 2 == 0;
    std::array<bool, 8> channels =
        ringA ? bits<8>("01000000") : bits<8>("00000001");
    out.push_back(ringProducer(order, source.tile, *sourceBase, total));

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
  TensorInfo info = operandInfo(op, 0);
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
  if (cast<RedistributeOp>(op).getMappingAttr()) {
    size_t rank = info.shape.size();
    Index low = applyForward(op, Index(rank, 0));
    Index last;
    for (int64_t size : info.shape)
      last.push_back(size - 1);
    Index high = applyForward(op, last);
    Index blockStrides = strides(
        tail(extent(unionBox(*slicingOf(op), 0, 0)), rank), info.elementBytes);
    int64_t rowBytes =
        product(ArrayRef(info.shape).drop_front(2)) * info.elementBytes;
    for (const TileBox &destination : destinations) {
      Index lo, hi;
      for (size_t dim = 0; dim < rank; ++dim) {
        lo.push_back(std::max(destination.box.lo[dim], low[dim]));
        hi.push_back(std::min(destination.box.hi[dim], high[dim]));
        if (dim != 1 && (lo[dim] != low[dim] || hi[dim] != high[dim]))
          return unsupported(op, "a padded ring reshape that splits rows "
                                 "across tiles");
      }
      int64_t rows = hi[1] - lo[1] + 1;
      if (rows <= 0)
        return unsupported(op, "a padded ring reshape with an empty tile");
      int64_t start = (lo[1] - low[1]) * rowBytes;
      Index offset;
      for (size_t dim = 0; dim < rank; ++dim)
        offset.push_back(lo[dim] - destination.box.lo[dim]);
      RingConsumer consumer;
      consumer.traversal.baseAddress =
          *destinationBase + dot(offset, blockStrides);
      consumer.traversal.counter =
          padded({counter(rowBytes - kAccess, kAccess),
                  counter((rows - 1) * blockStrides[1], blockStrides[1])},
                 4);
      consumer.traversal.syncProducer = {producerSync(true, 1)};
      consumer.traversal.byteAddressMode = access(kAccess);
      consumer.virtualChannelSubscription = channels;
      consumer.threadMulticastBitmap = bits<4>("1000");
      consumer.filter.firstDiscardByteLoopMap = 3;
      consumer.filter.firstDiscardByteCount = start;
      consumer.filter.lastDiscardByteCount = total - start - rows * rowBytes;
      out.push_back(dma(consumer, tileBit(destination.tile)));
    }
    llvm::append_range(out, groupFences());
    return out;
  }
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
