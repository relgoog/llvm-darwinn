#include "Families.h"
#include <map>
#include <set>

using namespace mlir;
using namespace mlir::darwinn;
using namespace mlir::darwinn::codegen;

namespace {

constexpr int64_t kFlit = 16;
constexpr int64_t kSlice = 32;

using Rows = std::pair<int64_t, int64_t>;

struct Flow {
  int64_t from;
  int64_t to;
  int64_t first;
  int64_t last;
};

struct GatherGeometry {
  int64_t rowBytes = 0;
  int64_t sourceRowBytes = 0;
  int64_t sourcePixelBytes = 0;
  int64_t channels = 0;
  int64_t destinationRowBytes = 0;
  int64_t columnOffset = 0;
  bool columnFirst = false;
  bool vector = false;
  std::map<int64_t, int64_t> channelOffsets;
  std::map<int64_t, int64_t> origin;
  SmallVector<std::pair<int64_t, Rows>> sources;
  SmallVector<std::pair<int64_t, Rows>> destinations;
  std::map<int64_t, Rows> staging;
  SmallVector<Flow> rowsFlows;
  SmallVector<Flow> columnFlows;
  SmallVector<std::pair<std::pair<int64_t, int64_t>, SmallVector<int64_t>>>
      columns;
};

struct GatherPlan : GatherGeometry {
  int64_t sourceBase = 0;
  int64_t stagingBase = 0;
  int64_t destinationBase = 0;
  int64_t wide = 0;
};

const Rows &rowsOf(ArrayRef<std::pair<int64_t, Rows>> entries, int64_t tile) {
  return llvm::find_if(entries,
                       [&](const auto &entry) { return entry.first == tile; })
      ->second;
}

FailureOr<GatherGeometry> vectorGeometry(Operation *op) {
  TensorInfo source = operandInfo(op, 0), info = resultInfo(op);
  auto sourceType =
      dyn_cast<DistributedTensorType>(op->getOperand(0).getType());
  auto resultType = dyn_cast<DistributedTensorType>(op->getResult(0).getType());
  if (!sourceType || !resultType || !sourceType.getElementType().isBF16() ||
      !resultType.getElementType().isBF16() || source.shape != info.shape ||
      source.shape.size() != 2 || source.shape[0] != 1 ||
      source.shape[1] <= 0 || source.shape[1] % kTiles ||
      cast<RedistributeOp>(op).getMappingAttr())
    return unsupported(op, "a vector gather without identical unmapped "
                           "BF16 vectors divisible across sixteen tiles");

  SmallVector<TileBox> sources = clampedTiles(producer(op, 0));
  SmallVector<TileBox> destinations = clampedTiles(op);
  if (sources.size() != kTiles || destinations.size() != 1 ||
      destinations.front().tile != 0 ||
      destinations.front().box.lo != Index({0, 0}) ||
      destinations.front().box.hi != Index({0, source.shape[1] - 1}))
    return unsupported(op, "a vector gather without sixteen source tiles "
                           "and one complete destination on tile zero");

  GatherGeometry g;
  g.vector = true;
  g.channels = source.shape[1] / kTiles;
  g.rowBytes = g.sourceRowBytes = g.destinationRowBytes = g.channels * 2;
  if (g.rowBytes < 8 || g.rowBytes > kSlice || !llvm::isPowerOf2_64(g.rowBytes))
    return unsupported(op, "a vector gather whose source fragments are "
                           "not 8, 16 or 32 bytes");

  llvm::sort(sources, [](const TileBox &a, const TileBox &b) {
    return a.tile < b.tile;
  });
  for (auto [index, entry] : llvm::enumerate(sources)) {
    int64_t tile = index;
    if (entry.tile != tile || entry.box.lo != Index({0, tile * g.channels}) ||
        entry.box.hi != Index({0, (tile + 1) * g.channels - 1}))
      return unsupported(op, "a vector gather whose source tiles do not "
                             "hold consecutive equal channel intervals");
    g.sources.push_back({tile, {tile, tile}});
  }
  g.origin[0] = 0;
  g.destinations.push_back({0, {0, kTiles - 1}});
  return g;
}

FailureOr<GatherGeometry> geometry(Operation *op) {
  TensorInfo source = operandInfo(op, 0), info = resultInfo(op);
  Index shift = applyForward(op, Index(info.shape.size(), 0));
  GatherGeometry g;
  if (source.shape.size() == 2 || info.shape.size() == 2) {
    FailureOr<GatherGeometry> vector = vectorGeometry(op);
    if (failed(vector))
      return failure();
    g = std::move(*vector);
  } else {
    if (source.shape.size() != 4 || info.shape.size() != 4)
      return failure();
    g.rowBytes = source.shape[2] * source.shape[3] * source.elementBytes;
    g.sourceRowBytes = g.rowBytes;
    g.channels = source.shape[3];
    g.destinationRowBytes = info.shape[2] * info.shape[3] * info.elementBytes;
    g.columnOffset = shift[2] * info.shape[3] * info.elementBytes;
    SmallVector<TileBox> destinations = clampedTiles(op);
    int64_t heldChannels = extent(destinations.front().box).back();
    if (heldChannels < source.shape.back()) {
      if (info.shape != source.shape || info.elementBytes != 2 ||
          llvm::any_of(destinations, [&](const TileBox &entry) {
            return entry.tile / kGrid != destinations.front().tile / kGrid ||
                   entry.box.lo[1] != 0 || entry.box.lo[2] != 0 ||
                   entry.box.hi[1] + 1 != info.shape[1] ||
                   entry.box.hi[2] + 1 != info.shape[2] ||
                   extent(entry.box).back() != heldChannels;
          }))
        return failure();
      g.sourcePixelBytes = source.shape.back() * source.elementBytes;
      g.channels = heldChannels;
      g.rowBytes = g.destinationRowBytes = info.shape[2] * heldChannels * 2;
      for (const TileBox &entry : destinations)
        g.channelOffsets[entry.tile % kGrid] = entry.box.lo.back() * 2;
    }
    for (const TileBox &entry : clampedTiles(producer(op, 0)))
      g.sources.push_back({entry.tile, {entry.box.lo[1], entry.box.hi[1]}});
    g.columnFirst = llvm::all_of(g.sources, [&](const auto &entry) {
      return entry.first / kGrid == g.sources.front().first / kGrid;
    });
    for (const TileBox &entry : destinations) {
      int64_t first = entry.box.lo[1] - shift[1];
      g.origin[entry.tile] = first;
      g.destinations.push_back(
          {entry.tile,
           {std::max<int64_t>(first, 0),
            std::min(entry.box.hi[1] - shift[1], source.shape[1] - 1)}});
    }
  }
  auto line = [&](int64_t tile) {
    return g.columnFirst ? tile % kGrid : tile / kGrid;
  };
  std::map<int64_t, int64_t> owner;
  std::map<int64_t, std::set<int64_t>> held;
  for (auto [tile, rows] : g.sources)
    for (int64_t row = rows.first; row <= rows.second; ++row)
      owner[row] = tile;
  for (auto [d, rows] : g.destinations) {
    for (int64_t row = rows.first; row <= rows.second; ++row) {
      auto found = owner.find(row);
      if (found == owner.end())
        return failure();
      int64_t s = found->second;
      int64_t holder = g.columnFirst ? d / kGrid * kGrid + s % kGrid
                                     : s / kGrid * kGrid + d % kGrid;
      held[holder].insert(row);
      if (holder == d)
        continue;
      auto key = std::make_pair(holder, d);
      auto column = llvm::find_if(
          g.columns, [&](const auto &entry) { return entry.first == key; });
      if (column == g.columns.end())
        g.columns.push_back({key, {row}});
      else
        column->second.push_back(row);
    }
  }
  for (const auto &[tile, rows] : held)
    g.staging[tile] = {*rows.begin(), *rows.rbegin()};
  for (auto [h, rows] : g.staging)
    for (auto [s, sourceRows] : g.sources) {
      if (s == h || line(s) != line(h))
        continue;
      int64_t first = std::max(rows.first, sourceRows.first);
      int64_t last = std::min(rows.second, sourceRows.second);
      if (first <= last)
        g.rowsFlows.push_back({s, h, first, last});
    }
  for (const auto &[key, rows] : g.columns) {
    for (auto [index, row] : llvm::enumerate(rows))
      if (row != rows.front() + static_cast<int64_t>(index))
        return failure();
    g.columnFlows.push_back({key.first, key.second, rows.front(), rows.back()});
  }
  return g;
}

FailureOr<GatherPlan> plan(Operation *op, Context &context) {
  FailureOr<GatherGeometry> g = geometry(op);
  if (failed(g))
    return unsupported(op, "a gather whose rows are not contiguous");
  int64_t pixelBytes = g->channels * 2;
  if ((!g->vector && !g->sourcePixelBytes && pixelBytes < 16) || pixelBytes % 8)
    return unsupported(op, "a gather whose rows do not split into thread "
                           "granules");
  SmallVector<const StorageBlock *> blocks;
  for (const StorageBlock &block : context.narrow())
    if (block.value == op && block.suffix == Suffix::Dest)
      blocks.push_back(&block);
  llvm::stable_sort(blocks, [](const StorageBlock *a, const StorageBlock *b) {
    return a->start < b->start;
  });
  if (blocks.size() != 2)
    return unsupported(op, "a gather without a staging and a destination "
                           "block");
  FailureOr<int64_t> source = context.storageAddress(producer(op, 0));
  FailureOr<int64_t> wide = context.wideAddress(op, Suffix::Gather);
  if (failed(source) || failed(wide))
    return unsupported(op, "a gather between unplaced blocks");
  GatherPlan p;
  static_cast<GatherGeometry &>(p) = std::move(*g);
  p.sourceBase = *source;
  p.stagingBase = blocks[0]->offset * kThreads;
  p.destinationBase = blocks[1]->offset * kThreads;
  p.wide = *wide;
  return p;
}

enum class Part { Send, Receive, Forward };

struct Action;

struct MeshOp {
  int64_t tile;
  MeshDirection direction;
  int64_t hops;
  std::map<Part, Action *> parts;
};

struct Action {
  int64_t tile;
  Part role;
  int64_t size;
  MeshDirection direction;
  int64_t address = 0;
  MeshOp *op = nullptr;
  int64_t hops = 0;
};

using Address = std::function<int64_t(int64_t, int64_t, int64_t)>;

struct Lines {
  std::deque<Action> actions;
  std::deque<MeshOp> ops;
};

using Route = SmallVector<MeshOp *>;

SmallVector<Route> lineOps(ArrayRef<Flow> flows, bool horizontal,
                           const Address &read, const Address &write,
                           Lines &storage) {
  auto position = [&](int64_t tile) {
    return horizontal ? tile % kGrid : tile / kGrid;
  };
  int64_t step = horizontal ? 1 : kGrid;
  SmallVector<Flow> ordered(flows);
  llvm::stable_sort(ordered, [&](const Flow &a, const Flow &b) {
    auto key = [&](const Flow &f) {
      return std::make_pair(position(f.to),
                            std::abs(position(f.to) - position(f.from)));
    };
    return key(a) < key(b);
  });
  SmallVector<std::pair<int64_t, SmallVector<Action *>>> actions;
  SmallVector<SmallVector<Action *>> routes;
  for (const Flow &flow : ordered) {
    bool forward = position(flow.to) > position(flow.from);
    MeshDirection direction =
        horizontal ? (forward ? MeshDirection::OutboundEastInboundWest
                              : MeshDirection::OutboundWestInboundEast)
                   : (forward ? MeshDirection::OutboundSouthInboundNorth
                              : MeshDirection::OutboundNorthInboundSouth);
    int64_t delta = forward ? step : -step;
    int64_t hops = std::abs(position(flow.to) - position(flow.from));
    SmallVector<Action *> route;
    for (int64_t k = 0; k <= hops; ++k) {
      int64_t tile = flow.from + delta * k;
      Part role =
          k == 0 ? Part::Send : (k == hops ? Part::Receive : Part::Forward);
      Action &action = storage.actions.emplace_back(Action{
          tile, role, flow.last - flow.first + 1, direction, 0, nullptr, hops});
      if (role == Part::Send)
        action.address = read(flow.from, flow.first, flow.to);
      else if (role == Part::Receive)
        action.address = write(flow.to, flow.first, flow.from);
      auto entry = llvm::find_if(
          actions, [&](const auto &item) { return item.first == tile; });
      if (entry == actions.end())
        actions.push_back({tile, {&action}});
      else
        entry->second.push_back(&action);
      route.push_back(&action);
    }
    routes.push_back(route);
  }
  for (auto &[tile, items] : actions) {
    size_t k = 0;
    while (k < items.size()) {
      MeshOp &op = storage.ops.emplace_back(
          MeshOp{tile, items[k]->direction, items[k]->hops, {}});
      bool pair = k + 1 < items.size() &&
                  ((items[k]->role == Part::Send &&
                    items[k + 1]->role == Part::Receive) ||
                   (items[k]->role == Part::Receive &&
                    items[k + 1]->role == Part::Send)) &&
                  items[k]->direction == items[k + 1]->direction;
      size_t count = pair ? 2 : 1;
      for (size_t index = k; index < k + count; ++index) {
        op.parts[items[index]->role] = items[index];
        items[index]->op = &op;
      }
      k += count;
    }
  }
  SmallVector<Route> out;
  std::set<MeshOp *> seen;
  for (const auto &route : routes) {
    Route ops;
    for (Action *action : route)
      if (seen.insert(action->op).second)
        ops.push_back(action->op);
    if (!ops.empty())
      out.push_back(std::move(ops));
  }
  return out;
}

using Signature = std::tuple<MeshDirection, std::optional<int64_t>,
                             std::optional<int64_t>, std::optional<int64_t>>;

Signature signature(const MeshOp *op) {
  auto size = [&](Part part) -> std::optional<int64_t> {
    auto found = op->parts.find(part);
    if (found == op->parts.end())
      return std::nullopt;
    return found->second->size;
  };
  return {op->direction, size(Part::Send), size(Part::Receive),
          size(Part::Forward)};
}

SmallVector<SmallVector<MeshOp *>>
mergeLines(ArrayRef<SmallVector<Route>> lines) {
  SmallVector<SmallVector<MeshOp *>> merged;
  for (const auto &routes : lines) {
    int64_t cursor = -1;
    for (const Route &route : routes) {
      SmallVector<size_t> matches;
      int64_t at = cursor;
      for (MeshOp *op : route) {
        auto match = llvm::find_if(
            llvm::seq<size_t>(at + 1, merged.size()), [&](size_t k) {
              return signature(merged[k].front()) == signature(op) &&
                     merged[k].front()->hops == op->hops;
            });
        if (match == llvm::seq<size_t>(at + 1, merged.size()).end())
          break;
        matches.push_back(*match);
        at = *match;
      }
      if (matches.size() == route.size()) {
        for (auto [k, op] : llvm::zip(matches, route))
          merged[k].push_back(op);
        cursor = at;
        continue;
      }
      for (MeshOp *op : route)
        merged.push_back({op});
      cursor = merged.size() - 1;
    }
  }
  return merged;
}

} // namespace

SmallVector<Emitted, 0> codegen::registerLoads(const TileValues &values,
                                               uint8_t reg, const Order &order,
                                               std::optional<int64_t> thread) {
  Groups groups;
  for (const auto &item : values) {
    auto [tile, value] = item;
    auto entry = llvm::find_if(
        groups, [&](const auto &group) { return group.first == item.second; });
    if (entry == groups.end())
      groups.push_back({value, {tile}});
    else
      entry->second.push_back(tile);
  }
  SmallVector<Emitted, 0> out;
  for (const auto &[value, tiles] : order(groups))
    out.push_back(
        load(value, tileSet(tiles), reg,
             thread ? std::optional(threadBit(*thread)) : std::nullopt));
  return out;
}

Groups codegen::latestFirst(Groups groups) {
  llvm::stable_sort(groups, [](const auto &a, const auto &b) {
    return *llvm::max_element(a.second) > *llvm::max_element(b.second);
  });
  return groups;
}

namespace {

Groups byRotatedColumn(Groups groups) {
  SmallVector<int64_t> tiles = llvm::to_vector(llvm::seq<int64_t>(0, kTiles));
  llvm::stable_sort(tiles, [](int64_t a, int64_t b) {
    return (a % kGrid + kGrid - 1) % kGrid < (b % kGrid + kGrid - 1) % kGrid;
  });
  std::array<int64_t, kTiles> rank{};
  for (auto [index, tile] : llvm::enumerate(tiles))
    rank[tile] = index;
  auto key = [&](const auto &group) {
    int64_t out = 0;
    for (int64_t tile : group.second)
      out = std::max(out, rank[tile]);
    return out;
  };
  llvm::stable_sort(
      groups, [&](const auto &a, const auto &b) { return key(a) < key(b); });
  return groups;
}

Groups byBottomRow(Groups groups) {
  auto key = [](const auto &group) {
    std::pair<int64_t, int64_t> out{kGrid, kGrid};
    for (int64_t tile : group.second)
      out =
          std::min(out, std::make_pair(kGrid - 1 - tile / kGrid, tile % kGrid));
    return out;
  };
  llvm::stable_sort(
      groups, [&](const auto &a, const auto &b) { return key(a) < key(b); });
  return groups;
}

Groups byFirstTileDescending(Groups groups) {
  llvm::stable_sort(groups, [](const auto &a, const auto &b) {
    return *llvm::min_element(a.second) > *llvm::min_element(b.second);
  });
  return groups;
}

using Step = std::pair<int64_t, SmallVector<Emitted, 0>>;

SmallVector<Step> meshInstructions(ArrayRef<SmallVector<MeshOp *>> merged,
                                   const GatherPlan &p, int64_t receiveStride,
                                   int64_t sendPixelBytes) {
  SmallVector<Step> out;
  for (const auto &members : merged) {
    const MeshOp *sample = members.front();
    SmallVector<int64_t> tiles;
    for (const MeshOp *op : members)
      tiles.push_back(op->tile);
    Mesh mesh;
    mesh.direction = sample->direction;
    SmallVector<Emitted, 0> loads;
    std::array<bool, 8> sourced{};
    for (auto [role, traversal, reg] :
         {std::make_tuple(Part::Send, &Mesh::read, uint8_t(0)),
          std::make_tuple(Part::Receive, &Mesh::write, uint8_t(1))}) {
      if (!sample->parts.count(role))
        continue;
      int64_t rows = sample->parts.at(role)->size;
      Traversal &target = mesh.*traversal;
      bool strided =
          role == Part::Receive && rows > 1 && receiveStride != p.rowBytes;
      if (role == Part::Send && sendPixelBytes) {
        int64_t accessBytes = p.channels * 2;
        int64_t pixels = rows * p.rowBytes / accessBytes;
        target.counter =
            padded({counter((pixels - 1) * sendPixelBytes, sendPixelBytes)}, 5);
      } else if (strided)
        target.counter =
            padded({counter(p.rowBytes - kFlit, kFlit),
                    counter((rows - 1) * receiveStride, receiveStride)},
                   5);
      else {
        int64_t bytes = rows * p.rowBytes;
        int64_t end =
            p.vector ? (ceilDiv(bytes, kFlit) - 1) * kFlit : bytes - kFlit;
        target.counter = padded({counter(end, kFlit)}, 5);
      }
      target.syncProducer = {producerSync(true, strided)};
      int64_t accessBytes =
          p.vector ? std::min(kFlit, rows * p.rowBytes) : kFlit;
      target.byteAddressMode = access(
          role == Part::Send && sendPixelBytes ? p.channels * 2 : accessBytes);
      TileValues addresses;
      std::set<int64_t> distinct;
      for (const MeshOp *op : members) {
        addresses.push_back({op->tile, op->parts.at(role)->address});
        distinct.insert(op->parts.at(role)->address);
      }
      if (distinct.size() > 1) {
        llvm::append_range(loads,
                           registerLoads(addresses, reg, latestFirst, {}));
        sourced[reg] = true;
      } else {
        target.baseAddress = addresses.front().second;
      }
    }
    if (sample->parts.count(Part::Forward)) {
      int64_t size = sample->parts.at(Part::Forward)->size * p.rowBytes;
      int64_t flits = p.vector ? ceilDiv(size, kFlit) : size / kFlit;
      mesh.write.counter = padded({counter(flits - 1, 1, false)}, 5);
      mesh.write.syncProducer = {producerSync(false)};
      mesh.write.byteAddressMode =
          access(p.vector ? std::min(kFlit, size) : kFlit);
      mesh.forwardingMode = true;
    }
    loads.push_back(dma(mesh, tileSet(tiles), sourced));
    out.push_back({0, std::move(loads)});
  }
  return out;
}

struct Copy {
  int64_t source;
  int64_t destination;
  int64_t rows;
  int64_t stride;
};

FailureOr<SmallVector<Step>> vectorCopies(Operation *op,
                                          const std::map<int64_t, Copy> &copies,
                                          const GatherPlan &p) {
  using Key = std::tuple<int64_t, int64_t, int64_t>;
  std::map<Key, SmallVector<int64_t>> groups;
  for (const auto &[tile, copy] : copies) {
    if (copy.stride != p.rowBytes)
      return unsupported(op, "a vector gather with strided local copies");
    groups[{copy.source, copy.destination, copy.rows * p.rowBytes}].push_back(
        tile);
  }

  SmallVector<Step> out;
  for (const auto &[key, tiles] : groups) {
    auto [source, destination, bytes] = key;
    SmallVector<Emitted, 0> body;
    if (bytes <= kSlice) {
      int64_t accessBytes = std::min(kFlit, bytes);
      NarrowToNarrow copy;
      copy.read.baseAddress = source;
      copy.write.baseAddress = destination;
      copy.read.counter = copy.write.counter =
          padded({counter(bytes - accessBytes, accessBytes)}, 4);
      copy.read.byteAddressMode = copy.write.byteAddressMode =
          access(accessBytes);
      copy.read.syncProducer = {producerSync(false)};
      copy.write.syncProducer = {producerSync(true)};
      body.push_back(dma(copy, tileSet(tiles)));
    } else {
      if (bytes % kSlice || bytes > kThreads * kSlice || source % kSlice ||
          destination % kSlice)
        return unsupported(op, "a vector gather local copy that cannot "
                               "split into four aligned 32-byte slices");

      for (int64_t thread = 0; thread < bytes / kSlice; ++thread) {
        NarrowToWide gather;
        gather.read.baseAddress = source + thread * kSlice;
        gather.read.counter = padded({counter(0, kSlice)}, 6);
        gather.read.byteAddressMode = access(kSlice);
        gather.write.baseAddress = p.wide;
        gather.write.counter = padded({counter(0, 1)}, 4);
        gather.write.byteAddressMode = access(kSlice);
        gather.write.syncProducer = {producerSync(true, 1)};
        gather.threadMulticastBitmap = threadBit(thread);
        gather.byteAddress.strideUnitGranulesLoopMap = 1;
        gather.byteAddress.defaultStrideUnitGranules = 2;
        gather.byteAddress.lastStrideUnitGranules = 2;
        gather.byteAddress.cellStride = kSlice / 4;
        body.push_back(dma(gather, tileSet(tiles)));

        WideToNarrow scatter;
        scatter.read.baseAddress = p.wide;
        scatter.read.counter = padded({counter(0, 1)}, 4);
        scatter.write.baseAddress = (destination + thread * kSlice) / 4;
        scatter.write.counter = padded({counter(0, kSlice / 4)}, 6);
        scatter.write.syncProducer = {producerSync(true)};
        scatter.readWatchers = {dmaWatcher(
            tileWatcher(TileSyncFlag::NarrowToWideWrite, 1, 1, 1, true))};
        scatter.wideMemoryLoadStoreLoopId = 1;
        scatter.zInBundleValidCount = kSlice / 4;
        scatter.threadMulticastBitmap = threadBit(thread);
        body.push_back(dma(scatter, tileSet(tiles)));
      }
    }
    out.push_back({tiles.front() / kGrid, std::move(body)});
  }
  return out;
}

struct Dim {
  int64_t count;
  int64_t stride;
  bool offset;
};

SmallVector<Dim> mergeDims(ArrayRef<Dim> dims) {
  SmallVector<Dim> out;
  for (const Dim &dim : dims) {
    if (!out.empty() && dim.stride == out.back().count * out.back().stride) {
      out.back().count *= dim.count;
      out.back().offset |= dim.offset;
      continue;
    }
    out.push_back(dim);
  }
  return out;
}

void pushDim(SmallVector<Counter> &items, const Dim &dim) {
  if (dim.count > 1)
    items.push_back(counter((dim.count - 1) * dim.stride, dim.stride));
  else if (dim.offset)
    items.push_back(counter(0, dim.stride));
}

FailureOr<SmallVector<Step>>
localCopies(Operation *op, const std::map<int64_t, Copy> &copies,
            const GatherPlan &p, const Order &sourceOrder,
            const Order &destinationOrder, int64_t sourcePixelBytes = 0) {
  if (p.vector)
    return vectorCopies(op, copies, p);
  SmallVector<std::pair<int64_t, std::map<int64_t, Copy>>> groups;
  for (const auto &item : copies) {
    const auto &[tile, copy] = item;
    auto entry = llvm::find_if(groups, [&](const auto &group) {
      return group.first == item.second.rows &&
             (!sourcePixelBytes ||
              (group.second.begin()->second.source % sourcePixelBytes != 0) ==
                  (item.second.source % sourcePixelBytes != 0));
    });
    if (entry == groups.end())
      groups.push_back({copy.rows, {{tile, copy}}});
    else
      entry->second[tile] = copy;
  }
  llvm::stable_sort(groups, [](const auto &a, const auto &b) {
    return a.second.begin()->first < b.second.begin()->first;
  });
  int64_t width = p.rowBytes / (p.channels * 2);
  int64_t pixelBytes = p.channels * 2;
  int64_t readPixelBytes = sourcePixelBytes ? sourcePixelBytes : pixelBytes;
  int64_t readRowBytes = width * readPixelBytes;
  int64_t element = kSlice;
  int64_t granules = llvm::divideCeil(pixelBytes, element);
  while (pixelBytes <= 128 ? kThreads % granules : granules > kThreads)
    granules = llvm::divideCeil(pixelBytes, element *= 2);
  element = std::min(pixelBytes, element);
  int64_t free = pixelBytes <= 128 ? kThreads / granules : 1;
  int64_t segments = free;
  while (width % segments)
    segments /= 2;
  int64_t segmentPixels = width / segments;
  SmallVector<Step> blocks;
  for (const auto &[rows, members] : groups) {
    SmallVector<int64_t> tiles;
    for (const auto &entry : members)
      tiles.push_back(entry.first);
    SmallVector<Emitted, 0> &out =
        blocks.emplace_back(tiles.front() / kGrid, SmallVector<Emitted, 0>{})
            .second;
    std::array<bool, 16> multicast = tileSet(tiles);
    int64_t stride = members.begin()->second.stride;
    bool aligned =
        stride % kSlice == 0 && llvm::all_of(members, [](const auto &entry) {
          return entry.second.destination % kSlice == 0;
        });
    int64_t rowGroups = free / segments;
    while (rows % rowGroups)
      rowGroups /= 2;
    int64_t groupRows = rows / rowGroups;
    for (int64_t thread = 0; thread < granules * segments * rowGroups;
         ++thread) {
      std::array<bool, 4> bitmap = threadBit(thread);
      int64_t granule = thread % granules,
              segment = thread / granules % segments,
              rowGroup = thread / (granules * segments);
      int64_t offset = segment * segmentPixels * pixelBytes + granule * element;
      int64_t sourceOffset = segment * segmentPixels * readPixelBytes +
                             granule * element +
                             rowGroup * groupRows * readRowBytes;
      int64_t destinationOffset = offset + rowGroup * groupRows * stride;
      int64_t bytes = std::min(element, pixelBytes - granule * element);
      bool merged = bytes == pixelBytes;
      int64_t writeElement = aligned && (merged || pixelBytes % kSlice == 0)
                                 ? kSlice
                             : merged ? (bytes > 16 ? 16 : element)
                                      : std::min<int64_t>(16, bytes & -bytes);
      SmallVector<Dim> readDims =
          mergeDims({{1, bytes,
                      granule != 0 ||
                          (sourcePixelBytes &&
                           members.begin()->second.source % sourcePixelBytes)},
                     {segmentPixels, readPixelBytes, segment != 0},
                     {groupRows, readRowBytes, false}});
      int64_t run = readDims.front().count * bytes;
      int64_t accessBytes = std::min<int64_t>(
          run, llvm::isPowerOf2_64(run) || run > 128 ? 128 : run & -run);
      if (run > 128)
        while (accessBytes > 8 && run % accessBytes)
          accessBytes -= 8;
      if (!sourcePixelBytes)
        writeElement = std::min(writeElement, accessBytes & -accessBytes);
      SmallVector<Counter> readItems;
      pushDim(readItems,
              {run / accessBytes, accessBytes, readDims.front().offset});
      for (const Dim &dim : ArrayRef(readDims).drop_front())
        pushDim(readItems, dim);
      if (readItems.empty())
        readItems.push_back(counter(0, accessBytes));

      NarrowToWide gather;
      gather.read.counter = padded(readItems, 6);
      gather.read.byteAddressMode = access(accessBytes);
      TileValues sources;
      std::set<int64_t> distinctSources;
      for (const auto &[tile, copy] : members) {
        sources.push_back({tile, copy.source + sourceOffset});
        distinctSources.insert(copy.source + sourceOffset);
      }
      std::array<bool, 8> gatherRegisters{};
      SmallVector<Emitted, 0> loads;
      if (distinctSources.size() > 1) {
        loads = registerLoads(sources, 0, sourceOrder, thread);
        gatherRegisters = bits<8>("10000000");
      } else {
        gather.read.baseAddress = sources.front().second;
      }

      SmallVector<Dim> writeDims =
          mergeDims({{1, bytes, granule != 0},
                     {segmentPixels, pixelBytes, segment != 0},
                     {groupRows, stride, false}});
      int64_t perWideRow = 4;
      if (writeDims.front().stride == bytes) {
        Dim &run = writeDims.front();
        run.count = run.count * bytes / writeElement;
        run.stride = writeElement;
        perWideRow = 128 / writeElement;
      }
      for (Dim &dim : writeDims)
        dim.stride /= 4;
      SmallVector<Counter> scatterItems;
      SmallVector<int64_t> wideCounts;
      std::optional<int64_t> innerIndex;
      int64_t filled = 1;
      for (const Dim &dim : writeDims) {
        if (innerIndex || dim.count == 1) {
          if (innerIndex && (dim.count > 1 || dim.offset))
            wideCounts.push_back(dim.count);
          pushDim(scatterItems, dim);
          continue;
        }
        int64_t take = std::min(perWideRow / filled, dim.count);
        while (dim.count % take)
          --take;
        if (take > 1)
          scatterItems.push_back(counter((take - 1) * dim.stride, dim.stride));
        filled *= take;
        if (take == dim.count && perWideRow / filled >= 2)
          continue;
        if (scatterItems.empty())
          return unsupported(op, "a gather whose thread rows do not fill "
                                 "whole wide rows");
        innerIndex = scatterItems.size() - 1;
        if (dim.count > take) {
          int64_t wide = dim.count / take;
          wideCounts.push_back(wide);
          scatterItems.push_back(
              counter((wide - 1) * take * dim.stride, take * dim.stride));
        }
      }
      if (!innerIndex && !scatterItems.empty())
        innerIndex = scatterItems.size() - 1;
      if (!innerIndex)
        return unsupported(op, "a gather copy without a wide row");

      int64_t wideRows = 1;
      for (int64_t count : wideCounts)
        wideRows *= count;
      int64_t whole = std::min<int64_t>(16, writeElement * writeElement / 32);
      int64_t inner = wideRows;
      if (wideRows * filled >= whole * perWideRow)
        for (inner = 1; inner * perWideRow < 12 || wideRows % inner; ++inner)
          ;
      int64_t outer = wideRows / inner;
      int64_t wideLoop = inner > 1;
      int64_t depth = wideLoop || filled <= 8;
      SmallVector<Counter> wideItems;
      if (wideLoop || outer == 1)
        wideItems.push_back(counter(inner - 1, 1));
      if (outer > 1)
        wideItems.push_back(counter(outer - 1, 1, false));
      std::optional<int64_t> syncLoop;
      int64_t covered = 1, perIncrement = 1;
      for (auto [index, count] : llvm::enumerate(wideCounts)) {
        if (covered * count > inner || inner % (covered * count)) {
          syncLoop = *innerIndex + 1 + index;
          perIncrement = covered;
          break;
        }
        covered *= count;
      }

      gather.write.baseAddress = p.wide;
      gather.write.counter = padded(wideItems, 4);
      gather.write.syncProducer = {producerSync(true, depth)};
      gather.write.byteAddressMode = access(128 * filled / perWideRow);
      if (outer > 1) {
        gather.write.doubleBufferLoop = wideLoop;
        gather.write.secondBufferOffset = inner;
        gather.writeWatchers = {
            dmaWatcher(tileWatcher(TileSyncFlag::WideToNarrowWrite,
                                   (int32_t(1) << 25) - inner / perIncrement,
                                   inner / perIncrement, depth, true),
                       true)};
      }
      gather.threadMulticastBitmap = bitmap;
      gather.byteAddress.strideUnitGranulesLoopMap = 1;
      gather.byteAddress.defaultStrideUnitGranules = 2;
      gather.byteAddress.lastStrideUnitGranules = 2;
      gather.byteAddress.cellStride = 32 * filled / perWideRow;
      gather.byteAddress.defaultCellStrideGroupCount = 1;
      gather.byteAddress.lastCellStrideGroupCount = 1;
      llvm::append_range(out, loads);
      out.push_back(dma(gather, multicast, gatherRegisters));

      WideToNarrow scatter;
      scatter.read.baseAddress = p.wide;
      scatter.read.counter = padded(wideItems, 4);
      if (outer > 1) {
        scatter.read.doubleBufferLoop = wideLoop;
        scatter.read.secondBufferOffset = inner;
      }
      scatter.write.counter = padded(scatterItems, 6);
      scatter.write.syncProducer = {
          producerSync(true, syncLoop.value_or(scatterItems.size() - 1))};
      TileValues destinations;
      std::set<int64_t> distinctDestinations;
      for (const auto &[tile, copy] : members) {
        int64_t address = (copy.destination + destinationOffset) / 4;
        destinations.push_back({tile, address});
        distinctDestinations.insert(address);
      }
      std::array<bool, 8> scatterRegisters{};
      loads.clear();
      if (distinctDestinations.size() > 1) {
        loads = registerLoads(destinations, 1, destinationOrder, thread);
        scatterRegisters = bits<8>("01000000");
      } else {
        scatter.write.baseAddress = destinations.front().second;
      }
      scatter.readWatchers = {dmaWatcher(
          tileWatcher(TileSyncFlag::NarrowToWideWrite, 1, 1, depth, true))};
      scatter.wideMemoryLoadStoreLoopId = *innerIndex + 1;
      scatter.zInBundleValidCount = writeElement / 4;
      scatter.threadMulticastBitmap = bitmap;
      llvm::append_range(out, loads);
      out.push_back(dma(scatter, multicast, scatterRegisters));
    }
  }
  return blocks;
}

SmallVector<Emitted, 0> meshStage(ArrayRef<Flow> flows, bool horizontal,
                                  const Address &read, const Address &write,
                                  const GatherPlan &p, int64_t receiveStride,
                                  SmallVector<Step> copies,
                                  int64_t sendPixelBytes = 0) {
  std::map<int64_t, SmallVector<Flow>> lines;
  for (const Flow &flow : flows)
    lines[horizontal ? flow.from / kGrid : flow.from % kGrid].push_back(flow);
  auto position = [&](int64_t tile) {
    return horizontal ? tile % kGrid : tile / kGrid;
  };
  Lines storage;
  std::array<SmallVector<std::pair<int64_t, SmallVector<Route>>>, 2> sequences;
  for (bool forward : {true, false})
    for (const auto &[line, members] : lines) {
      SmallVector<Flow> selected;
      for (const Flow &flow : members)
        if ((position(flow.to) > position(flow.from)) == forward)
          selected.push_back(flow);
      sequences[!forward].push_back(
          {horizontal ? line : kGrid,
           lineOps(selected, horizontal, read, write, storage)});
    }

  SmallVector<Emitted, 0> out;
  auto emit = [&](ArrayRef<SmallVector<MeshOp *>> merged) {
    for (Step &step :
         meshInstructions(merged, p, receiveStride, sendPixelBytes))
      llvm::append_range(out, std::move(step.second));
  };
  std::map<MeshOp *, int64_t> lineOf;
  SmallVector<SmallVector<Route>> forward;
  for (auto &[key, sequence] : sequences[0]) {
    for (const Route &route : sequence)
      for (MeshOp *op : route)
        lineOf[op] = key;
    forward.push_back(std::move(sequence));
  }
  SmallVector<SmallVector<MeshOp *>> merged = mergeLines(forward);
  llvm::stable_sort(
      copies, [](const Step &a, const Step &b) { return a.first < b.first; });
  size_t emitted = 0;
  for (size_t index = 0; index <= copies.size(); ++index) {
    if (index < copies.size())
      llvm::append_range(out, std::move(copies[index].second));
    int64_t bound = index + 1 < copies.size()
                        ? copies[index + 1].first
                        : std::numeric_limits<int64_t>::max();
    if (index + 1 < copies.size() &&
        copies[index + 1].first == copies[index].first)
      continue;
    size_t first = emitted;
    while (emitted < merged.size() && lineOf[merged[emitted].front()] < bound)
      ++emitted;
    emit(ArrayRef(merged).slice(first, emitted - first));
  }
  SmallVector<SmallVector<Route>> backward;
  for (auto &[key, sequence] : sequences[1])
    backward.push_back(std::move(sequence));
  emit(mergeLines(backward));
  return out;
}

} // namespace

int64_t codegen::gatherStagingWords(Operation *op) {
  FailureOr<GatherGeometry> g = geometry(op);
  if (failed(g))
    return 0;
  int64_t rows = 0;
  for (const auto &[tile, range] : g->staging)
    rows = std::max(rows, range.second - range.first + 1);
  return rows * g->rowBytes / 4;
}

int64_t codegen::gatherWideRows(Operation *op) {
  FailureOr<GatherGeometry> g = geometry(op);
  if (failed(g))
    return 0;
  if (g->vector)
    return kGrid * g->rowBytes > kSlice ? 1 : 0;
  int64_t width = g->rowBytes / (g->channels * 2);
  SmallVector<int64_t> copies;
  for (auto [tile, rows] : g->sources)
    copies.push_back(rows.second - rows.first + 1);
  for (auto [d, rows] : g->destinations) {
    std::set<int64_t> moved;
    for (const auto &[key, columnRows] : g->columns)
      if (key.second == d)
        moved.insert(columnRows.begin(), columnRows.end());
    int64_t local = 0;
    for (int64_t row = rows.first; row <= rows.second; ++row)
      local += !moved.count(row);
    copies.push_back(local);
  }
  int64_t capacity = 0;
  for (int64_t rows : copies) {
    int64_t wideRows = rows * width * g->channels * 2 / 512;
    if (wideRows < 16)
      capacity = std::max(capacity, wideRows);
  }
  return capacity ? capacity : 8;
}

Body codegen::gatherRows(Operation *op, Context &context) {
  FailureOr<GatherPlan> planned = plan(op, context);
  if (failed(planned))
    return failure();
  const GatherPlan &p = *planned;
  std::map<int64_t, Copy> copies;
  for (auto [tile, rows] : p.sources) {
    auto staged = p.staging.find(tile);
    if (staged == p.staging.end())
      continue;
    int64_t first = std::max(rows.first, staged->second.first);
    int64_t last = std::min(rows.second, staged->second.second);
    if (first > last)
      continue;
    int64_t channel =
        p.sourcePixelBytes ? p.channelOffsets.at(tile % kGrid) : 0;
    copies[tile] =
        Copy{p.sourceBase + (first - rows.first) * p.sourceRowBytes + channel,
             p.stagingBase + (first - staged->second.first) * p.rowBytes,
             last - first + 1, p.rowBytes};
  }
  Address read = [&](int64_t s, int64_t first, int64_t destination) {
    int64_t channel =
        p.sourcePixelBytes ? p.channelOffsets.at(destination % kGrid) : 0;
    return p.sourceBase +
           (first - rowsOf(p.sources, s).first) * p.sourceRowBytes + channel;
  };
  Address write = [&](int64_t h, int64_t first, int64_t) {
    return p.stagingBase + (first - p.staging.at(h).first) * p.rowBytes;
  };
  Order order = byRotatedColumn;
  if (p.sourcePixelBytes)
    order = [](Groups groups) {
      std::rotate(groups.begin(), std::next(groups.begin()), groups.end());
      return groups;
    };
  FailureOr<SmallVector<Step>> blocks =
      localCopies(op, copies, p, order, order, p.sourcePixelBytes);
  if (failed(blocks))
    return failure();
  return meshStage(p.rowsFlows, !p.columnFirst, read, write, p, p.rowBytes,
                   std::move(*blocks), p.sourcePixelBytes);
}

Body codegen::gatherColumns(Operation *op, Context &context) {
  FailureOr<GatherPlan> planned = plan(op, context);
  if (failed(planned))
    return failure();
  const GatherPlan &p = *planned;
  std::map<int64_t, std::set<int64_t>> moved;
  for (const auto &[key, rows] : p.columns)
    moved[key.second].insert(rows.begin(), rows.end());
  std::map<int64_t, Copy> copies;
  for (auto [d, rows] : p.destinations) {
    SmallVector<int64_t> local;
    for (int64_t row = rows.first; row <= rows.second; ++row)
      if (!moved[d].count(row))
        local.push_back(row);
    for (auto [index, row] : llvm::enumerate(local))
      if (row != local.front() + static_cast<int64_t>(index))
        return unsupported(op, "a gather destination with non contiguous "
                               "local rows");
    if (local.empty())
      continue;
    copies[d] = Copy{
        p.stagingBase + (local.front() - p.staging.at(d).first) * p.rowBytes,
        p.destinationBase +
            (local.front() - p.origin.at(d)) * p.destinationRowBytes +
            p.columnOffset,
        static_cast<int64_t>(local.size()), p.destinationRowBytes};
  }
  Address read = [&](int64_t h, int64_t first, int64_t) {
    return p.stagingBase + (first - p.staging.at(h).first) * p.rowBytes;
  };
  Address write = [&](int64_t d, int64_t first, int64_t) {
    return p.destinationBase +
           (first - p.origin.at(d)) * p.destinationRowBytes + p.columnOffset;
  };
  FailureOr<SmallVector<Step>> blocks =
      localCopies(op, copies, p, byBottomRow, byFirstTileDescending);
  if (failed(blocks))
    return failure();
  return meshStage(p.columnFlows, p.columnFirst, read, write, p,
                   p.destinationRowBytes, std::move(*blocks));
}
