#include "SlicingModel.h"
#include "mlir/IR/AffineExpr.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <cmath>
#include <set>

using namespace mlir;
using namespace mlir::darwinn;
using namespace mlir::darwinn::slicing;

namespace {

constexpr int64_t kTileBandwidth = 32;
constexpr int64_t kRingLatency = 30;
constexpr int64_t kTransferOverhead = 6;
constexpr int64_t kRingLength = 16;
constexpr double kHostStrideFactor = 1.1;

int64_t product(ArrayRef<int64_t> values) {
  int64_t result = 1;
  for (int64_t value : values)
    result *= value;
  return result;
}

FailureOr<int64_t> elementBits(Type type) {
  Type element = cast<ShapedType>(type).getElementType();
  if (element.isBF16() || element.isF16())
    return 16;
  if (element.isF32())
    return 32;
  if (auto integer = dyn_cast<IntegerType>(element))
    return integer.getWidth();
  return failure();
}

SmallVector<bool> usedDims(AffineMap map) {
  SmallVector<bool> used(map.getNumDims(), false);
  for (AffineExpr expression : map.getResults())
    expression.walk([&](AffineExpr sub) {
      if (auto dimension = dyn_cast<AffineDimExpr>(sub))
        used[dimension.getPosition()] = true;
    });
  return used;
}

std::set<unsigned> axesOf(AffineExpr expression) {
  std::set<unsigned> axes;
  expression.walk([&](AffineExpr sub) {
    if (auto dimension = dyn_cast<AffineDimExpr>(sub))
      axes.insert(dimension.getPosition());
  });
  return axes;
}

SmallVector<int64_t> iterationNest(AffineMap traversal, const Tile &box,
                                   ArrayRef<int64_t> extents) {
  unsigned dims = traversal.getNumDims();
  SmallVector<int64_t> low(dims, 0);
  SmallVector<int64_t> high;
  for (int64_t extent : extents)
    high.push_back(extent - 1);
  for (auto [index, expression] : llvm::enumerate(traversal.getResults())) {
    auto form = linearize(expression, dims);
    if (!form)
      continue;
    SmallVector<unsigned> active;
    for (unsigned dim = 0; dim < dims; ++dim)
      if (form->coefficients[dim] != 0)
        active.push_back(dim);
    if (active.empty())
      continue;
    if (active.size() == 1) {
      unsigned dim = active.front();
      int64_t coefficient = form->coefficients[dim];
      low[dim] = std::max(
          low[dim],
          llvm::divideCeilSigned(box.lo[index] - form->constant, coefficient));
      high[dim] = std::min(
          high[dim],
          llvm::divideFloorSigned(box.hi[index] - form->constant, coefficient));
      continue;
    }
    unsigned main =
        *llvm::max_element(active, [&](unsigned left, unsigned right) {
          return form->coefficients[left] < form->coefficients[right];
        });
    int64_t minimum = form->constant;
    int64_t maximum = form->constant;
    for (unsigned dim : active) {
      if (dim == main)
        continue;
      int64_t span = form->coefficients[dim] * (extents[dim] - 1);
      minimum += std::min<int64_t>(0, span);
      maximum += std::max<int64_t>(0, span);
    }
    int64_t coefficient = form->coefficients[main];
    low[main] =
        std::max(low[main],
                 llvm::divideCeilSigned(box.lo[index] - maximum, coefficient));
    high[main] =
        std::min(high[main],
                 llvm::divideFloorSigned(box.hi[index] - minimum, coefficient));
  }
  SmallVector<int64_t> nest;
  for (unsigned dim = 0; dim < dims; ++dim)
    nest.push_back(high[dim] - low[dim] + 1);
  return nest;
}

int64_t tensorOperation(ArrayRef<int64_t> nest, int64_t zo, int64_t zi) {
  int64_t outer = product(nest.drop_back(2));
  return outer * llvm::divideCeilSigned(nest.back(), zo) *
         llvm::divideCeilSigned(nest[nest.size() - 2], zi);
}

struct WideBlocking {
  int64_t inside;
  int64_t loads;
  bool unbuffered;
};

WideBlocking wideBlocking(ArrayRef<int64_t> nest, ArrayRef<bool> flags,
                          int64_t registers, int64_t zo, int64_t zi) {
  SmallVector<int64_t> loops{llvm::divideCeilSigned(nest.back(), zo)};
  SmallVector<bool> wide{flags.back()};
  for (unsigned index = 0; index + 2 < nest.size(); ++index) {
    loops.push_back(nest[index]);
    wide.push_back(flags[index]);
  }
  loops.push_back(llvm::divideCeilSigned(nest[nest.size() - 2], zi));
  wide.push_back(flags[flags.size() - 2]);

  int64_t footprint = 1;
  for (auto [loop, isWide] : llvm::zip(loops, wide))
    if (isWide)
      footprint *= loop;
  bool shift = footprint > registers || 2 * footprint <= registers;
  int64_t inside = 1;
  int64_t outside = 1;
  bool open = true;
  for (unsigned index = loops.size(); index-- > 0;) {
    int64_t loop = loops[index];
    if (wide[index]) {
      if (open && ((loop * inside) << (shift ? 1 : 0)) <= registers) {
        inside *= loop;
        continue;
      }
    } else if (open) {
      continue;
    }
    outside *= loop;
    open = false;
  }
  return {inside, outside, shift};
}

int64_t narrowToWide(ArrayRef<int64_t> nest, int64_t wideZo, int64_t zo,
                     int64_t zi, ArrayRef<bool> flags, int64_t registers,
                     int64_t bits, int64_t innerCycles) {
  constexpr int64_t kBandwidth = 32;
  constexpr int64_t kMinimumLatency = 6;
  WideBlocking blocking = wideBlocking(nest, flags, registers, zo, zi);
  int64_t lanes = zo == wideZo ? zi : 1;
  int64_t perCycle =
      std::min(llvm::divideCeilSigned(lanes * bits * zo, 8), kBandwidth);
  int64_t cyclesPerLoad = llvm::divideCeilSigned(
      zo * llvm::divideCeilSigned(zi * bits, 8), perCycle);
  int64_t perBlock =
      std::max(kMinimumLatency + 1, cyclesPerLoad * blocking.inside);
  return blocking.loads * perBlock + (blocking.unbuffered ? 0 : innerCycles);
}

std::optional<int64_t> nluLatency(ComputeOpOptionsAttr options) {
  std::optional<NluFunctionKind> function = options.getNluFunction();
  if (!function || *function == NluFunctionKind::Linear)
    return 1;
  if (llvm::is_contained({NluFunctionKind::Exp, NluFunctionKind::Reciprocal,
                          NluFunctionKind::LogisticSigmoid, NluFunctionKind::HardSwish},
                         *function))
    return 5;
  return std::nullopt;
}

bool hasBias(Operation *operation) {
  auto kinds = operation->getAttrOfType<ArrayAttr>("auxiliary_tensor_types");
  if (!kinds)
    return false;
  return llvm::any_of(kinds, [](Attribute kind) {
    auto typed = dyn_cast<AuxTensorTypeAttr>(kind);
    return typed && typed.getValue() == AuxTensorKind::Bias;
  });
}

Tile tileAt(const SliceMaps &maps, ArrayRef<int64_t> point) {
  return {maps.begins.compose(point), maps.ends.compose(point)};
}

struct DmaShape {
  int64_t minor;
  SmallVector<int64_t> levels;
};

DmaShape dmaLevels(ArrayRef<int64_t> shape, ArrayRef<int64_t> extent) {
  std::optional<unsigned> partial;
  for (unsigned index = 0; index < shape.size(); ++index)
    if (extent[index] < shape[index])
      partial = index;
  if (!partial)
    return {product(shape), {}};
  int64_t minor = extent[*partial] * product(shape.drop_front(*partial + 1));
  SmallVector<int64_t> levels;
  std::optional<int64_t> current;
  bool absorb = false;
  for (unsigned index = *partial; index-- > 0;) {
    if (!current)
      current = extent[index];
    else if (absorb)
      *current *= extent[index];
    else {
      levels.push_back(*current);
      current = extent[index];
    }
    absorb = extent[index] == shape[index];
  }
  if (current)
    levels.push_back(*current);
  while (!levels.empty() && levels.back() == 1)
    levels.pop_back();
  return {minor, levels};
}

struct Transfers {
  SmallVector<int64_t> keys;
  SmallVector<int64_t> repeats;
  SmallVector<int64_t> counts;
  SmallVector<int64_t> order;
  int64_t outer = 1;
};

FailureOr<Transfers> ringTransfers(ArrayRef<int64_t> shape,
                                   const SmallVector<Tile> *tiles,
                                   ArrayRef<int64_t> domain) {
  if (!tiles) {
    DmaShape whole = dmaLevels(shape, shape);
    return Transfers{{0},
                     {whole.levels.empty() ? 1 : whole.levels[0]},
                     {whole.minor},
                     {0},
                     whole.levels.size() > 1 ? whole.levels[1] : 1};
  }
  int64_t rows = domain[0];
  int64_t columns = domain[1];
  int64_t threads = domain.size() > 2 ? domain[2] : 1;
  auto at = [&](int64_t row, int64_t column, int64_t thread) -> const Tile & {
    return (*tiles)[(row * columns + column) * threads + thread];
  };
  auto moved = [&](bool alongRows) {
    std::set<unsigned> dims;
    const Tile &origin = at(0, 0, 0);
    const Tile &step = alongRows ? at(1, 0, 0) : at(0, 1, 0);
    for (unsigned dim = 0; dim < shape.size(); ++dim)
      if (origin.lo[dim] != step.lo[dim])
        dims.insert(dim);
    return dims;
  };
  std::optional<unsigned> keyAxis;
  if (rows > 1 && columns > 1) {
    std::set<unsigned> rowDims = moved(true);
    std::set<unsigned> columnDims = moved(false);
    if (rowDims != columnDims && !rowDims.empty() && !columnDims.empty())
      keyAxis = *rowDims.begin() < *columnDims.begin() ? 0 : 1;
  }
  Transfers result;
  std::optional<int64_t> outer;
  for (int64_t row = 0; row < rows; ++row)
    for (int64_t column = 0; column < columns; ++column) {
      SmallVector<int64_t> low = at(row, column, 0).lo;
      SmallVector<int64_t> high = at(row, column, 0).hi;
      for (int64_t thread = 1; thread < threads; ++thread) {
        const Tile &tile = at(row, column, thread);
        for (unsigned dim = 0; dim < shape.size(); ++dim) {
          low[dim] = std::min(low[dim], tile.lo[dim]);
          high[dim] = std::max(high[dim], tile.hi[dim]);
        }
      }
      SmallVector<int64_t> extent;
      for (unsigned dim = 0; dim < shape.size(); ++dim)
        extent.push_back(std::min(high[dim], shape[dim] - 1) -
                         std::max<int64_t>(low[dim], 0) + 1);
      if (llvm::any_of(extent, [](int64_t size) { return size <= 0; }))
        continue;
      DmaShape dma = dmaLevels(shape, extent);
      if (dma.levels.size() > 2)
        return failure();
      int64_t level = dma.levels.size() > 1 ? dma.levels[1] : 1;
      if (outer && *outer != level)
        return failure();
      outer = level;
      result.keys.push_back(!keyAxis ? 0 : (*keyAxis == 0 ? row : column));
      result.repeats.push_back(dma.levels.empty() ? 1 : dma.levels[0]);
      result.counts.push_back(dma.minor);
      result.order.push_back(row * 4 + column);
    }
  result.outer = outer.value_or(1);
  return result;
}

struct RingCost {
  int64_t transfer;
  int64_t order;
  int64_t overhead;
  int64_t totalBytes;
};

struct RingShape {
  int64_t overhead = kTransferOverhead;
  int64_t length = kRingLength;
  int64_t hopScale = 2;
  int64_t wrapScale = 2;
};

RingCost ringCost(const Transfers &transfers, int64_t bits,
                  RingShape shape = {}) {
  int64_t repeat = std::max<int64_t>(transfers.outer, 1);
  llvm::MapVector<int64_t, SmallVector<std::tuple<int64_t, int64_t, int64_t>>>
      groups;
  RingCost cost{0, 0, 0, 0};
  for (unsigned index = 0; index < transfers.keys.size(); ++index) {
    int64_t bytes = bits * transfers.counts[index] / 8;
    int64_t times = std::max<int64_t>(1, transfers.repeats[index]);
    int64_t weight = repeat * times;
    cost.totalBytes += weight * bytes;
    cost.transfer += weight * llvm::divideCeilSigned(bytes, kTileBandwidth);
    cost.overhead += shape.overhead * weight;
    groups[transfers.keys[index]].push_back(
        {times, transfers.counts[index], transfers.order[index]});
  }
  int64_t spans = 0;
  int64_t hops = 0;
  int64_t wraps = 0;
  for (auto &[key, group] : groups) {
    int64_t done = 0;
    while (true) {
      int64_t next = INT32_MAX;
      int64_t first = -1;
      int64_t wrap = 0;
      int64_t active = 0;
      int64_t previous = -1;
      int64_t carried = 0;
      for (auto [times, count, position] : group) {
        if (count == 0 || times <= done || position == -1)
          continue;
        next = std::min(next, times);
        if (first == -1)
          first = position;
        else if (position < previous) {
          wrap = position + carried + shape.length - previous;
          carried = wrap;
        }
        ++active;
        previous = position;
      }
      if (next == INT32_MAX || first == -1)
        break;
      wraps += (next - done) * wrap;
      spans += next - done;
      hops += (next - done) * active;
      done = next;
    }
  }
  hops *= shape.hopScale;
  wraps *= shape.wrapScale;
  cost.order = spans * repeat * kRingLatency + repeat * (hops + wraps);
  return cost;
}

std::map<std::pair<int64_t, int64_t>, Tile>
tileUnions(const SmallVector<Tile> &tiles, ArrayRef<int64_t> domain) {
  std::map<std::pair<int64_t, int64_t>, Tile> result;
  int64_t threads = domain.size() > 2 ? domain[2] : 1;
  for (auto [index, tile] : llvm::enumerate(tiles)) {
    int64_t flat = index / threads;
    std::pair<int64_t, int64_t> key{flat / domain[1], flat % domain[1]};
    auto [found, inserted] = result.try_emplace(key, tile);
    if (inserted)
      continue;
    for (unsigned dim = 0; dim < tile.lo.size(); ++dim) {
      found->second.lo[dim] = std::min(found->second.lo[dim], tile.lo[dim]);
      found->second.hi[dim] = std::max(found->second.hi[dim], tile.hi[dim]);
    }
  }
  return result;
}

bool samePlacement(ArrayRef<int64_t> from, ArrayRef<int64_t> to,
                   const SmallVector<Tile> &source,
                   ArrayRef<int64_t> sourceDomain,
                   const SmallVector<Tile> &destination,
                   ArrayRef<int64_t> destinationDomain) {
  if (sourceDomain.take_front(2) != destinationDomain.take_front(2))
    return false;
  auto sourceTiles = tileUnions(source, sourceDomain);
  auto destinationTiles = tileUnions(destination, destinationDomain);
  if (sourceTiles.size() != destinationTiles.size())
    return false;
  auto clamp = [](Tile tile, ArrayRef<int64_t> shape) {
    for (unsigned dim = 0; dim < shape.size(); ++dim) {
      tile.lo[dim] = std::max<int64_t>(tile.lo[dim], 0);
      tile.hi[dim] = std::min(tile.hi[dim], shape[dim] - 1);
    }
    return tile;
  };
  auto count = [](const Tile &tile) {
    int64_t total = 1;
    for (unsigned dim = 0; dim < tile.lo.size(); ++dim)
      total *= tile.hi[dim] - tile.lo[dim] + 1;
    return total;
  };
  for (auto &[key, tile] : sourceTiles) {
    auto found = destinationTiles.find(key);
    if (found == destinationTiles.end())
      return false;
    Tile mine = clamp(tile, from);
    Tile theirs = clamp(found->second, to);
    if (count(mine) != count(theirs))
      return false;
    if (from != to)
      mine = reshapeTile(from, to, mine);
    if (!(mine == theirs))
      return false;
  }
  return true;
}

}

FailureOr<Estimate> SlicingModel::estimate(Operation *operation,
                                           ArrayRef<unsigned> state) {
  if (isa<CreateEmptyTensorOp, DistributedCreateViewOp, ReshapeOpOp>(operation))
    return Estimate{};
  if (auto fill = dyn_cast<FillOp>(operation))
    return estimateFill(fill);
  std::optional<unsigned> blockId = blockOf(operation);
  if (!blockId)
    return failure();
  unsigned codeId = state[*blockId];
  if (isa<StaticComputeOpOp, StaticUnaryComputeOpOp>(operation)) {
    auto found = computeCache.find({operation, codeId});
    if (found != computeCache.end())
      return found->second;
    auto result = estimateCompute(operation, codes[codeId]);
    if (succeeded(result))
      computeCache[{operation, codeId}] = *result;
    return result;
  }
  if (auto copy = dyn_cast<CopyOpOp>(operation))
    return estimateCopy(copy, codes[codeId]);
  if (auto join = dyn_cast<MathJoinOp>(operation))
    return estimateJoin(join, codes[codeId]);
  if (auto interpolate = dyn_cast<InterpolateHardwareOp>(operation))
    return estimateInterpolate(interpolate, codes[codeId]);
  if (auto redistribute = dyn_cast<RedistributeOp>(operation))
    return estimateRedistribute(redistribute, state);
  return operation->emitOpError("has no slicing cost model");
}

FailureOr<Estimate> mlir::darwinn::slicing::estimateFill(FillOp fill) {
  auto bits = elementBits(fill.getOutput().getType());
  if (failed(bits))
    return failure();
  int64_t bytes = product(shapeOf(fill.getOutput())) * *bits / 8;
  return Estimate{0, {{11, llvm::divideCeilSigned(bytes, kTileBandwidth)}}};
}

FailureOr<Estimate>
mlir::darwinn::slicing::estimateCompute(Operation *operation, const Code &code,
                                        bool transposed) {
  AffineMap traversal = traversalOf(operation);
  SmallVector<Value> views(operation->getOperands());
  auto extents = iterationExtents(traversal.getNumDims(), operation);
  if (failed(extents))
    return operation->emitOpError("has views without linear traversals");
  SmallVector<int64_t> nest(*extents);
  if (!code.isUnsliced()) {
    SmallVector<int64_t> origin(code.domain.size(), 0);
    nest = iterationNest(traversal, tileAt(code.maps, origin), *extents);
  }
  auto options = operation->getAttrOfType<ComputeOpOptionsAttr>("compute");
  std::optional<InnerOperationKind> kind = options.getInnerOperation();
  auto bits = elementBits(views[0].getType());
  if (failed(bits))
    return failure();
  bool matrix = kind == InnerOperationKind::Vmc;
  auto resultBits = elementBits(operation->getResult(0).getType());
  if (failed(resultBits))
    return failure();
  int64_t zo = std::min<int64_t>(nest.back(), matrix ? (transposed ? 8 : 32)
                                                     : 8 * 16 / *bits);
  int64_t zi = matrix ? std::min<int64_t>(nest[nest.size() - 2],
                                          transposed ? 8 * 16 / *resultBits
                                                     : 8 * 4 / *bits)
                      : 1;
  int64_t elements = 1;
  for (auto [dim, used] : llvm::enumerate(usedDims(traversal)))
    if (used)
      elements *= nest[dim];
  std::optional<int64_t> latency = nluLatency(options);
  if (!latency)
    return operation->emitOpError("has an NLU function without a latency");

  Estimate estimate;
  estimate.base = 8;
  int64_t rows =
      llvm::divideCeilSigned(elements, zo) * llvm::divideCeilSigned(zo, 4);
  estimate.resources[6] = tensorOperation(nest, zo, zi);
  estimate.resources[7] = rows * *latency;
  estimate.resources[9] = rows;

  if (kind == InnerOperationKind::Elementwise) {
    int64_t factor = 1;
    if (!code.isUnsliced() && code.domain.size() == 3 && code.domain[2] >= 3) {
      int64_t threads = code.domain[2];
      Operation *rhsView = views[1].getDefiningOp();
      AffineMap rhsTraversal = traversalOf(rhsView);
      ArrayRef<int64_t> rhsShape = shapeOf(views[1]);
      llvm::SmallDenseSet<unsigned> rhsDims;
      for (auto [index, expression] :
           llvm::enumerate(rhsTraversal.getResults()))
        if (rhsShape[index] > 1)
          for (unsigned dim : axesOf(expression))
            rhsDims.insert(dim);
      bool shared = true;
      for (auto [index, expression] :
           llvm::enumerate(code.maps.begins.getResults()))
        if (axesOf(expression).count(2))
          for (unsigned dim : axesOf(traversal.getResult(index)))
            if (rhsDims.contains(dim))
              shared = false;
      if (shared)
        factor = threads - threads / 2;
    }
    estimate.resources[10] = 2 * estimate.resources[6] * factor;
  }

  if (matrix) {
    Operation *weightView = views[1].getDefiningOp();
    SmallVector<bool> wide = usedDims(traversalOf(weightView));
    int64_t registers = 64 - (hasBias(operation) ? 2 : 0);
    int64_t inner = cycles(Estimate{0, estimate.resources});
    estimate.resources[8] =
        narrowToWide(nest, nest.back(), zo, zi, wide, registers, *bits, inner);
    if (!code.isUnsliced()) {
      auto weightBits = elementBits(views[1].getType());
      if (failed(weightBits))
        return failure();
      int64_t parameters = llvm::divideCeilSigned(
          product(shapeOf(views[1])) * *weightBits, 8 * kTileBandwidth);
      std::set<unsigned> outputAxes =
          axesOf(code.maps.begins.getResult(traversal.getNumResults() - 1));
      std::set<unsigned> anyAxes;
      for (AffineExpr expression : code.maps.begins.getResults())
        for (unsigned axis : axesOf(expression))
          anyAxes.insert(axis);
      if (outputAxes.count(2))
        return operation->emitOpError(
            "slices output channels over threads, which has no cost model");
      if (!outputAxes.empty()) {
        int64_t factor = 1;
        for (unsigned axis : outputAxes)
          factor *= code.domain[axis];
        for (int key : {6, 7, 8, 9})
          estimate.resources[key] *= factor;
        estimate.resources[11] =
            std::max(cycles(Estimate{0, estimate.resources}), parameters);
      } else if (!anyAxes.empty()) {
        estimate.resources[11] = parameters;
      }
    }
  }
  return estimate;
}

FailureOr<Estimate> mlir::darwinn::slicing::estimateCopy(CopyOpOp copy,
                                                         const Code &code) {
  ArrayRef<int64_t> shape = shapeOf(copy.getOutput());
  SmallVector<int64_t> extent(shape);
  if (!code.isUnsliced()) {
    SmallVector<int64_t> origin(code.domain.size(), 0);
    Tile tile = tileAt(code.maps, origin);
    for (unsigned dim = 0; dim < shape.size(); ++dim)
      extent[dim] = tile.hi[dim] - tile.lo[dim] + 1;
  }
  auto bits = elementBits(copy.getOutput().getType());
  if (failed(bits))
    return failure();
  int64_t rows = product(ArrayRef<int64_t>(extent).drop_back());
  int64_t lines = llvm::divideCeilSigned(extent.back() * *bits, 128);
  return Estimate{0, {{0, rows * lines * 128 / *bits / 8}}};
}

FailureOr<Estimate> mlir::darwinn::slicing::estimateJoin(MathJoinOp join,
                                                         const Code &code) {
  SmallVector<int64_t> extent(shapeOf(join.getOutput()));
  if (!code.isUnsliced()) {
    SmallVector<int64_t> origin(code.domain.size(), 0);
    Tile tile = tileAt(code.maps, origin);
    for (unsigned dim = 0; dim < extent.size(); ++dim)
      extent[dim] = tile.hi[dim] - tile.lo[dim] + 1;
  }
  auto bits = elementBits(join.getOutput().getType());
  if (failed(bits))
    return failure();
  int64_t rows = product(ArrayRef<int64_t>(extent).drop_back());
  int64_t cycles = 0;
  for (Value input : join.getInputs())
    cycles += rows *
              llvm::divideCeilSigned(shapeOf(input).back() * *bits, 128) * 128 /
              *bits / 8;
  return Estimate{0, {{0, cycles}}};
}

FailureOr<Estimate>
mlir::darwinn::slicing::estimateInterpolate(InterpolateHardwareOp interpolate,
                                            const Code &code) {
  Operation *operation = interpolate.getOperation();
  auto results = operation->getAttrOfType<ArrayAttr>("result_traversals");
  auto domainAttr = operation->getAttrOfType<ArrayAttr>("traversal_domain");
  if (!results || !domainAttr)
    return operation->emitOpError("lacks interpolation traversals");
  AffineMap produced = cast<AffineMapAttr>(results[0]).getValue();
  SmallVector<int64_t> nest;
  for (Attribute extent : domainAttr)
    nest.push_back(cast<IntegerAttr>(extent).getInt());
  if (!code.isUnsliced()) {
    SmallVector<int64_t> origin(code.domain.size(), 0);
    nest = iterationNest(produced, tileAt(code.maps, origin), nest);
  }
  bool nearest = interpolate.getInterpolateMethod().getValue() ==
                 InterpolateMethodKind::NearestNeighbor;
  if (nearest) {
    nest[3] = 1;
    nest[4] = 1;
  }
  auto bits = elementBits(interpolate.getInput().getType());
  if (failed(bits))
    return failure();
  int64_t zo = std::min<int64_t>(nest.back(), 8 * 16 / *bits);
  int64_t elements = 1;
  for (auto [dim, used] : llvm::enumerate(usedDims(produced)))
    if (used && dim != 3 && dim != 4)
      elements *= nest[dim];
  return Estimate{0,
                  {{6, tensorOperation(nest, zo, 1)},
                   {9, llvm::divideCeilSigned(elements, zo) *
                           llvm::divideCeilSigned(zo, 4)}}};
}

FailureOr<Estimate>
SlicingModel::estimateRedistribute(RedistributeOp redistribute,
                                   ArrayRef<unsigned> state) {
  Operation *operation = redistribute.getOperation();
  Operation *producer = redistribute.getInput().getDefiningOp();
  std::optional<unsigned> sourceBlock =
      producer ? blockOf(producer) : std::nullopt;
  unsigned sourceCode = sourceBlock ? state[*sourceBlock] : ~0u;
  unsigned destinationCode = state[*blockOf(operation)];
  auto key = std::make_tuple(operation, sourceCode, destinationCode);
  auto cached = redistributeCache.find(key);
  if (cached != redistributeCache.end())
    return cached->second;

  SlicedValue source, destination{&codes[destinationCode], nullptr};
  if (sourceBlock && !codes[sourceCode].isUnsliced()) {
    auto found = tiles(redistribute.getInput(), sourceCode);
    if (failed(found))
      return operation->emitOpError("has no source tiles");
    source = {&codes[sourceCode], *found};
  }
  if (source.code && !destination.code->isUnsliced()) {
    auto found = tiles(redistribute.getOutput(), destinationCode);
    if (failed(found))
      return operation->emitOpError("has no destination tiles");
    destination.tiles = *found;
  }
  auto result =
      slicing::estimateRedistribute(redistribute, source, destination);
  if (succeeded(result))
    redistributeCache[key] = *result;
  return result;
}

FailureOr<Estimate> mlir::darwinn::slicing::estimateRedistribute(
    RedistributeOp redistribute, SlicedValue source, SlicedValue destination) {
  Operation *operation = redistribute.getOperation();
  Value input = redistribute.getInput();
  Value output = redistribute.getOutput();
  auto bits = elementBits(input.getType());
  if (failed(bits))
    return failure();
  ArrayRef<int64_t> from = shapeOf(input);
  ArrayRef<int64_t> to = shapeOf(output);
  Estimate estimate;

  if (memorySpaceOf(input) == DistributedMemorySpace::HostMemory) {
    int64_t blocks =
        llvm::divideCeilSigned(product(to) * *bits / 8, kTileBandwidth);
    int64_t reads = isa<DistributedViewType>(input.getType())
                        ? std::lround(kHostStrideFactor * blocks)
                        : blocks;
    estimate.resources = {{1, reads}, {11, blocks}, {2, kRingLatency}};
    return estimate;
  }

  bool sourceSliced = source.code && !source.code->isUnsliced();
  bool destinationSliced = !destination.code->isUnsliced();
  bool toHost = memorySpaceOf(output) == DistributedMemorySpace::HostMemory;

  if (!sourceSliced && !destinationSliced && !toHost) {
    if (product(from) == product(to))
      return estimate;
    int64_t lines = llvm::divideCeilSigned(from.back() * *bits, 128);
    int64_t pad = to[1] - from[1];
    if (to[2] - from[2] != pad)
      return operation->emitOpError("pads rows and columns unevenly");
    int64_t width = from[2];
    estimate.resources = {{0, (from[1] + pad) * width * lines},
                          {3, (4 * pad + 2) * width * lines},
                          {4, (pad + 1) * width * lines}};
    return estimate;
  }

  if (sourceSliced && destinationSliced &&
      samePlacement(from, to, *source.tiles, source.code->domain,
                    *destination.tiles, destination.code->domain))
    return estimate;

  auto transfers =
      ringTransfers(from, sourceSliced ? source.tiles : nullptr,
                    sourceSliced ? ArrayRef<int64_t>(source.code->domain)
                                 : ArrayRef<int64_t>());
  if (failed(transfers))
    return operation->emitOpError("has a transfer with more than two strides");
  RingCost ring = ringCost(*transfers, *bits);
  estimate.resources = {
      {1, ring.transfer}, {2, ring.order}, {5, ring.overhead}};
  if (from.size() == to.size() && from.back() != to.back()) {
    estimate.resources[0] = product(to) / to.back();
    estimate.resources[3] = 1;
  }
  if (toHost)
    estimate.resources[12] =
        llvm::divideCeilSigned(ring.totalBytes, kTileBandwidth);
  if (isa<FilledViewType>(output.getType())) {
    Transfers whole{{0}, {1}, {product(to)}, {0}, 1};
    RingCost write = ringCost(whole, *bits, {0, 1, 0, 2});
    int64_t written =
        std::lround(kHostStrideFactor * write.transfer) + write.order;
    if (ring.transfer + ring.order + ring.overhead < written)
      estimate.resources[1] = written;
  }
  return estimate;
}
