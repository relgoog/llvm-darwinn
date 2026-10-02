#include "Families.h"
#include <map>
#include <set>

using namespace mlir;
using namespace mlir::darwinn;
using namespace mlir::darwinn::codegen;

namespace {

constexpr int64_t kLanes = 8;
constexpr float kBilinearScale = 1.0f / 16384.0f;

enum class Edge { First, Inner, Last };

Edge classify(int64_t position, int64_t neighbours, int64_t size) {
  int64_t floor = position >> 16;
  if (floor < 0)
    return Edge::First;
  if (floor + neighbours - 1 > size - 1)
    return Edge::Last;
  return Edge::Inner;
}

using Range = std::pair<int64_t, int64_t>;

int64_t leading(ArrayRef<Edge> classes, Edge edge) {
  int64_t index = 0;
  while (index < static_cast<int64_t>(classes.size()) && classes[index] == edge)
    ++index;
  return index;
}

int64_t trailing(ArrayRef<Edge> classes, Edge edge) {
  int64_t index = 0;
  while (index < static_cast<int64_t>(classes.size()) &&
         classes[classes.size() - 1 - index] == edge)
    ++index;
  return index;
}

FailureOr<SmallVector<Range>> pieces(ArrayRef<Edge> classes) {
  int64_t size = classes.size();
  int64_t first = leading(classes, Edge::First);
  int64_t last = size - trailing(classes, Edge::Last);
  for (int64_t index = first; index < last; ++index)
    if (classes[index] != Edge::Inner)
      return failure();
  SmallVector<Range> out;
  for (Range range : {Range{first, last}, Range{last, size}, Range{0, first}})
    if (range.second > range.first)
      out.push_back(range);
  return out;
}

bool uniform(ArrayRef<Edge> classes) { return llvm::all_equal(classes); }

using Key = std::tuple<int64_t, int64_t, int64_t, bool, bool, int64_t, int64_t>;
using Part = std::pair<int64_t, int64_t>;

struct Item {
  Key key;
  int64_t tile;
  int64_t thread;
  int64_t read;
  int64_t write;
};

struct Planned {
  Key key;
  std::set<int64_t> tiles;
  std::set<int64_t> threads;
  SmallVector<std::pair<Part, std::pair<int64_t, int64_t>>> values;
};

std::pair<int64_t, int64_t> valueOf(const Planned &op, Part part) {
  return llvm::find_if(op.values,
                       [&](const auto &entry) { return entry.first == part; })
      ->second;
}

} // namespace

Body codegen::interpolate(Operation *op, Context &context) {
  auto hardware = cast<InterpolateHardwareOp>(op);
  bool nearest = hardware.getInterpolateMethod().getValue() ==
                 InterpolateMethodKind::NearestNeighbor;
  int64_t neighbours = nearest ? 1 : 2;
  TensorInfo info = resultInfo(op);
  int64_t width = info.shape[2], channels = info.shape[3];
  Operation *owner = storage(producer(op, 0));
  SmallVector<int64_t, 4> inShape = resultInfo(owner).shape;
  int64_t inHeight = inShape[1], inWidth = inShape[2];
  auto sampling = [&](StartOffsetAndStrideAttr attr)
      -> FailureOr<std::pair<int64_t, int64_t>> {
    if (attr.getNumFractionalBits() != 16)
      return failure();
    return std::make_pair(
        static_cast<int64_t>(static_cast<int32_t>(attr.getStartOffset())),
        static_cast<int64_t>(attr.getStride()));
  };
  auto x = sampling(hardware.getX()), y = sampling(hardware.getY());
  if (failed(x) || failed(y))
    return unsupported(op, "interpolation with other than 16 fractional bits");
  auto [xStart, xStride] = *x;
  auto [yStart, yStride] = *y;
  int64_t sizeZ = channels * info.elementBytes;
  int64_t sizeXz = inWidth * sizeZ;
  int64_t blocks = ceilDiv(channels, kLanes);
  FailureOr<int64_t> inBase = context.storageAddress(owner);
  FailureOr<int64_t> outBase = context.storageAddress(op);
  if (failed(inBase) || failed(outBase))
    return unsupported(op, "interpolation between unplaced narrow blocks");
  std::map<int64_t, Box> inBoxes;
  for (const TileBox &entry : clampedTiles(owner))
    inBoxes[entry.tile] = entry.box;
  Slicing slicing = *slicingOf(op);

  SmallVector<Edge> xClasses;
  for (int64_t column = 0; column < width; ++column)
    xClasses.push_back(
        classify(xStart + column * xStride, neighbours, inWidth));
  FailureOr<SmallVector<Range>> xPieces = pieces(xClasses);
  if (failed(xPieces))
    return unsupported(op, "interpolation columns that cannot be peeled");

  struct Thread {
    int64_t tile;
    int64_t thread;
    int64_t tileLo;
    int64_t firstRow;
    SmallVector<Edge> classes;
  };
  SmallVector<Thread> threads;
  for (int64_t ty = 0; ty < slicing.domain[0]; ++ty)
    for (int64_t tx = 0; tx < slicing.domain[1]; ++tx) {
      int64_t tileLo = std::numeric_limits<int64_t>::max();
      for (int64_t t = 0; t < slicing.domain[2]; ++t)
        tileLo = std::min(tileLo, slicing.begin({ty, tx, t})[1]);
      for (int64_t t = 0; t < slicing.domain[2]; ++t) {
        int64_t lo = slicing.begin({ty, tx, t})[1];
        int64_t hi = slicing.end({ty, tx, t})[1];
        Thread entry{ty * kGrid + tx, t, tileLo, lo, {}};
        for (int64_t row = lo; row <= hi; ++row)
          entry.classes.push_back(
              classify(yStart + row * yStride, neighbours, inHeight));
        threads.push_back(std::move(entry));
      }
    }
  int64_t firstPeel = 0, lastPeel = 0;
  for (const Thread &entry : threads) {
    if (uniform(entry.classes))
      continue;
    firstPeel = std::max(firstPeel, leading(entry.classes, Edge::First));
    lastPeel = std::max(lastPeel, trailing(entry.classes, Edge::Last));
  }

  SmallVector<Item> items;
  for (const Thread &entry : threads) {
    int64_t count = entry.classes.size();
    SmallVector<Range> yPieces;
    if (uniform(entry.classes)) {
      yPieces = {{0, count}};
    } else {
      for (Range range : {Range{firstPeel, count - lastPeel},
                          Range{count - lastPeel, count}, Range{0, firstPeel}})
        if (range.second > range.first)
          yPieces.push_back(range);
      for (Range range : yPieces)
        if (!uniform(ArrayRef(entry.classes)
                         .slice(range.first, range.second - range.first)))
          return unsupported(op, "interpolation rows that need more than one "
                                 "peel");
    }
    SmallVector<std::pair<Range, Range>> order;
    for (Range rows : yPieces)
      order.push_back({rows, xPieces->front()});
    for (Range rows : yPieces)
      for (Range columns : ArrayRef(*xPieces).drop_front())
        order.push_back({rows, columns});
    int64_t clock = 0;
    for (auto [rows, columns] : order) {
      int64_t row = entry.firstRow + rows.first;
      Edge xClass = xClasses[columns.first], yClass = entry.classes[rows.first];
      int64_t xPosition = xStart + columns.first * xStride;
      int64_t yPosition = yStart + row * yStride;
      int64_t inLo = inBoxes.at(entry.tile).lo[1];
      int64_t inRow = std::clamp<int64_t>(yPosition >> 16, 0, inHeight - 1);
      int64_t inCol = std::clamp<int64_t>(xPosition >> 16, 0, inWidth - 1);
      int64_t fractionX =
          nearest && xClass != Edge::Inner ? 0 : xPosition & 0xffff;
      int64_t fractionY =
          nearest && yClass != Edge::Inner ? 0 : yPosition & 0xffff;
      int64_t read = *inBase + (inRow - inLo) * sizeXz + inCol * sizeZ;
      int64_t write =
          *outBase + ((row - entry.tileLo) * width + columns.first) * sizeZ;
      int64_t xCount = columns.second - columns.first;
      int64_t yCount = rows.second - rows.first;
      items.push_back({Key{clock, xCount, yCount, xClass != Edge::Inner,
                           yClass != Edge::Inner, fractionX, fractionY},
                       entry.tile, entry.thread, read, write});
      clock += xCount * yCount;
    }
  }

  SmallVector<std::pair<std::pair<int64_t, Part>, Planned>> planned;
  std::set<Key> keys;
  for (const Item &item : items)
    keys.insert(item.key);
  for (const Key &key : keys) {
    using Exact = std::tuple<int64_t, int64_t, int64_t>;
    SmallVector<std::pair<Exact, std::set<int64_t>>> exact;
    for (const Item &item : items) {
      if (item.key != key)
        continue;
      Exact id{item.thread, item.read, item.write};
      auto entry = llvm::find_if(
          exact, [&](const auto &value) { return value.first == id; });
      if (entry == exact.end())
        exact.push_back({id, {item.tile}});
      else
        entry->second.insert(item.tile);
    }
    using PerThread = std::map<int64_t, std::pair<int64_t, int64_t>>;
    SmallVector<std::pair<std::set<int64_t>, PerThread>> byTiles;
    for (const auto &item : exact) {
      const auto &[id, tiles] = item;
      auto entry = llvm::find_if(byTiles, [&](const auto &value) {
        return value.first == item.second;
      });
      if (entry == byTiles.end())
        entry = byTiles.insert(byTiles.end(), {tiles, {}});
      entry->second[std::get<0>(id)] = {std::get<1>(id), std::get<2>(id)};
    }
    SmallVector<std::pair<std::set<int64_t>,
                          SmallVector<std::pair<std::set<int64_t>, PerThread>>>>
        byThreads;
    for (const auto &[tiles, perThread] : byTiles) {
      std::set<int64_t> threadSet;
      for (const auto &entry : perThread)
        threadSet.insert(entry.first);
      auto entry = llvm::find_if(byThreads, [&](const auto &value) {
        return value.first == threadSet;
      });
      if (entry == byThreads.end())
        entry = byThreads.insert(byThreads.end(), {threadSet, {}});
      entry->second.push_back({tiles, perThread});
    }
    for (const auto &[threadSet, parts] : byThreads) {
      Planned planOp{key, {}, threadSet, {}};
      for (const auto &[tiles, perThread] : parts) {
        planOp.tiles.insert(tiles.begin(), tiles.end());
        for (int64_t tile : tiles)
          for (const auto &[thread, value] : perThread)
            planOp.values.push_back({Part{tile, thread}, value});
      }
      Part least = planOp.values.front().first;
      for (const auto &entry : planOp.values)
        least = std::min(least, entry.first);
      planned.push_back({{std::get<0>(key), least}, std::move(planOp)});
    }
  }
  llvm::sort(planned,
             [](const auto &a, const auto &b) { return a.first < b.first; });

  SmallVector<Emitted, 0> out;
  for (const auto &[order, planOp] : planned) {
    auto [clock, xCount, yCount, xClamped, yClamped, fractionX, fractionY] =
        planOp.key;
    SmallVector<uint8_t> sourced;
    for (auto [reg, index] : {std::pair<uint8_t, int>{0, 0}, {2, 1}}) {
      std::set<int64_t> distinct;
      for (const auto &entry : planOp.values)
        distinct.insert(index == 0 ? entry.second.first : entry.second.second);
      if (distinct.size() > 1)
        sourced.push_back(reg);
    }
    SmallVector<int64_t> rotated(planOp.threads.begin(), planOp.threads.end());
    std::rotate(rotated.begin(), rotated.begin() + 1, rotated.end());
    SmallVector<std::pair<std::pair<uint8_t, int64_t>,
                          SmallVector<std::pair<int64_t, std::set<int64_t>>>>>
        rows;
    for (int64_t thread : rotated)
      for (uint8_t reg : {uint8_t(0), uint8_t(2)}) {
        if (!llvm::is_contained(sourced, reg))
          continue;
        for (int64_t tile : planOp.tiles) {
          auto value = valueOf(planOp, {tile, thread});
          std::pair<uint8_t, int64_t> id{reg, reg ? value.second : value.first};
          auto entry = llvm::find_if(
              rows, [&](const auto &row) { return row.first == id; });
          if (entry == rows.end())
            entry = rows.insert(rows.end(), {id, {}});
          auto perThread = llvm::find_if(entry->second, [&](const auto &item) {
            return item.first == thread;
          });
          if (perThread == entry->second.end())
            entry->second.push_back({thread, {tile}});
          else
            perThread->second.insert(tile);
        }
      }
    auto weight = [](const auto &row) {
      size_t total = 0;
      for (const auto &entry : row.second)
        total += entry.second.size();
      return total;
    };
    llvm::stable_sort(rows, [&](const auto &a, const auto &b) {
      return weight(a) > weight(b);
    });
    for (const auto &[id, perThread] : rows)
      for (const auto &[thread, tiles] : perThread)
        out.push_back(
            load(id.second,
                 tileSet(SmallVector<int64_t>(tiles.begin(), tiles.end())),
                 id.first, threadBit(thread)));

    auto [read, write] = planOp.values.front().second;
    bool hasX = xCount > 1;
    int64_t neighbourEnd = neighbours - 1;
    SmallVector<int64_t> main{neighbourEnd, neighbourEnd};
    if (hasX)
      main.push_back(xCount - 1);
    main.push_back(yCount - 1);
    main.push_back(blocks - 1);
    TensorOp tensorOp;
    for (auto [index, end] : llvm::enumerate(main))
      tensorOp.mainOperation.counter[index] = end;
    SmallVector<Counter> narrowRead{
        xClamped ? counter(neighbourEnd, 1, false)
                 : counter(neighbourEnd * sizeZ, sizeZ),
        yClamped ? counter(neighbourEnd, 1, false)
                 : counter(neighbourEnd * sizeXz, sizeXz)};
    if (hasX)
      narrowRead.push_back(counter(xCount - 1, 1, false));
    if (yCount > 1)
      narrowRead.push_back(counter(yCount - 1, 1, false));
    if (blocks > 1)
      narrowRead.push_back(counter((blocks - 1) * kLanes * 2, kLanes * 2));
    int64_t outRow = width * sizeZ;
    SmallVector<Counter> narrowWrite{counter(0, kLanes * 2)};
    if (hasX)
      narrowWrite.push_back(counter((xCount - 1) * sizeZ, sizeZ));
    if (yCount > 1)
      narrowWrite.push_back(counter((yCount - 1) * outRow, outRow));
    int64_t writeSync = narrowWrite.size();
    if (blocks > 1)
      narrowWrite.push_back(counter((blocks - 1) * kLanes * 2, kLanes * 2));
    int64_t depth = main.size() - 1;
    int64_t yIndex = depth - 1;
    int64_t loopX, loopY;
    if (hasX) {
      loopX = 2;
      loopY = yCount > 1 ? yIndex : yIndex + 1;
    } else if (yCount > 1) {
      loopX = main.size();
      loopY = yIndex;
    } else {
      loopX = loopY = yIndex + 1;
    }
    tensorOp.narrowMemoryRead.counter = padded(narrowRead, 8);
    tensorOp.narrowMemoryRead.syncProducer = {producerSync(false)};
    tensorOp.narrowMemoryRead.byteAddressMode =
        access(kLanes * 2, 1u << (3 + hasX));
    tensorOp.narrowMemoryWriteFromNonLinear.counter = padded(narrowWrite, 8);
    tensorOp.narrowMemoryWriteFromNonLinear.syncProducer = {
        producerSync(true, writeSync)};
    tensorOp.narrowMemoryWriteFromNonLinear.byteAddressMode =
        access(kLanes * 2);
    tensorOp.wideMemoryReadForSums.counter = padded(
        {counter(neighbourEnd, 1, false), counter(neighbourEnd, 1, false),
         counter(xCount - 1, 1, false), counter(yCount - 1, 1, false),
         counter(0, 1, false), counter(blocks - 1, 1, false)},
        8);
    tensorOp.wideMemoryReadForSums.syncProducer = {producerSync(false)};
    if (hasX || yCount > 1 || blocks > 1) {
      tensorOp.wideMemoryReadForSums.doubleBufferLoop =
          hasX ? 2 : (yCount > 1 ? 3 : 5);
      tensorOp.wideMemoryReadForSums.secondBufferOffset = 4;
    }
    for (int64_t thread : planOp.threads)
      tensorOp.control.threadMulticastBitmap[thread] = true;
    Linear &linear = tensorOp.control.linear;
    linear.operation = LinearOperation::Interpolation;
    linear.activationType = OperandType::Bfloat;
    linear.parameterType = OperandType::Half;
    linear.macDisable = MacDisable::SecondFloat;
    linear.partialSumRead = PartialSumRead::Initialize;
    linear.partialSumWritebackDisable = true;
    NonLinear &nonLinear = tensorOp.control.nonLinear;
    nonLinear.operation = ActivationFunction::Relu;
    nonLinear.activationPipelineScale = nearest ? 1.0f : kBilinearScale;
    nonLinear.outputType = OutputType::Bfloat;
    tensorOp.control.zOutBlockLoopDepth = depth;
    tensorOp.control.lastZOutBlockValidCount = kLanes;
    tensorOp.control.defaultZOutBlockValidCount = kLanes;
    tensorOp.control.partialSumReuseMap = 3;
    ComputedSamplePoint point;
    point.loopDepthX = loopX;
    point.loopDepthY = loopY;
    point.firstX = fractionX;
    point.firstY = fractionY;
    point.strideX = xStride;
    point.strideY = yStride;
    tensorOp.control.interpolation = Interpolation{
        point, static_cast<uint32_t>(sizeZ), static_cast<uint16_t>(sizeXz),
        nearest ? InterpolationMode::NearestNeighbor
                : InterpolationMode::Bilinear};
    if (!llvm::is_contained(sourced, uint8_t(0)))
      tensorOp.narrowMemoryRead.baseAddress = read;
    if (!llvm::is_contained(sourced, uint8_t(2)))
      tensorOp.narrowMemoryWriteFromNonLinear.baseAddress = write;
    std::array<bool, 8> registerBitmap{};
    for (uint8_t reg : sourced)
      registerBitmap[reg] = true;
    out.push_back(tensor(
        tensorOp,
        tileSet(SmallVector<int64_t>(planOp.tiles.begin(), planOp.tiles.end())),
        registerBitmap));
  }
  return out;
}
