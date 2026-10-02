#include "SlicingModel.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Darwinn/IR/DarwinnOps.h"
#include "mlir/Dialect/Darwinn/Transforms/DistributedPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SetVector.h"
#include <map>

using namespace mlir;
using namespace mlir::darwinn;
using namespace mlir::darwinn::slicing;

namespace {

constexpr int64_t kTileBytes = 786432;
constexpr int64_t kTiles = 16;
constexpr int64_t kBandwidth = 32;
constexpr int64_t kComputeUnits = 4;

int64_t ceilDiv(int64_t value, int64_t divisor) {
  return llvm::divideCeilSigned(value, divisor);
}

bool onHost(Value value) {
  auto type = dyn_cast<DistributedTensorType>(value.getType());
  return type && type.getMemorySpace() == DistributedMemorySpace::HostMemory;
}

bool isHostReshape(Operation *operation) {
  return isa<ReshapeOpOp>(operation) && onHost(operation->getResult(0));
}

bool isConstantFill(Operation *operation) {
  auto fill = dyn_cast<FillOp>(operation);
  return fill && fill.getConstTypeAttr() &&
         fill.getConstTypeAttr().getValue() != ConstKind::None;
}

int64_t elementCount(ArrayRef<int64_t> shape) {
  int64_t count = 1;
  for (int64_t size : shape)
    count *= size;
  return count;
}

FailureOr<int64_t> elementBytes(Value value) {
  Type element = cast<ShapedType>(value.getType()).getElementType();
  if (!element.isIntOrFloat())
    return failure();
  return element.getIntOrFloatBitWidth() / 8;
}

SmallVector<Operation *> usersInBlock(Operation *operation) {
  llvm::SetVector<Operation *> users;
  for (Operation *user : operation->getUsers())
    if (!isa<func::ReturnOp>(user))
      users.insert(user);
  return users.takeVector();
}

bool isAnchor(Operation *operation) {
  if (isa<StaticComputeOpOp, StaticUnaryComputeOpOp, CopyOpOp,
          InterpolateHardwareOp>(operation))
    return true;
  return isa<RedistributeOp>(operation) && usersInBlock(operation).size() >= 2;
}

struct Shard {
  unsigned symbol;
  int64_t factor;
  int64_t extent;
};

struct Group {
  SmallVector<Operation *> operations;
  Operation *anchor = nullptr;
  int64_t id = 0;
  int64_t peak = 0;
  std::optional<Shard> shard;
  int64_t innerPeak = 0;
  DenseMap<Operation *, AffineMap> coordinates;
  SmallVector<int64_t> extents;
};

// Groups are numbered by seed, an anchor or a redistribute that only the
// forward pass places, in block order. A group holding several seeds is a
// merge, numbered after all seeds in the iteration order of the SDK's
// DenseSet<int> of first seeds, whose hash is 37 * key.
FailureOr<SmallVector<Group, 0>> partition(Block &block) {
  DenseMap<Operation *, unsigned> owner;
  SmallVector<Group, 0> groups;
  SmallVector<Operation *> operations;
  for (Operation &operation : block.without_terminator())
    operations.push_back(&operation);

  for (Operation *operation : operations) {
    if (!isAnchor(operation))
      continue;
    owner[operation] = groups.size();
    groups.emplace_back().anchor = operation;
  }

  for (Operation *operation : llvm::reverse(operations)) {
    if (owner.contains(operation) || isHostReshape(operation))
      continue;
    SmallVector<Operation *> users = usersInBlock(operation);
    if (users.empty() || llvm::any_of(users, llvm::IsaPred<RedistributeOp>))
      continue;
    auto first = owner.find(users.front());
    if (first == owner.end())
      continue;
    if (llvm::all_of(users, [&](Operation *user) {
          auto found = owner.find(user);
          return found != owner.end() && found->second == first->second;
        }))
      owner[operation] = first->second;
  }

  llvm::SmallDenseSet<Operation *> forward;
  for (Operation *operation : operations) {
    if (owner.contains(operation) || isHostReshape(operation))
      continue;
    forward.insert(operation);
    for (Value operand : operation->getOperands()) {
      auto found = owner.find(operand.getDefiningOp());
      if (found == owner.end())
        continue;
      owner[operation] = found->second;
      break;
    }
    if (!owner.contains(operation))
      return operation->emitOpError("belongs to no sharding group");
  }

  for (Operation *operation : operations)
    if (auto found = owner.find(operation); found != owner.end())
      groups[found->second].operations.push_back(operation);

  int64_t seeds = 0;
  DenseMap<unsigned, SmallVector<int64_t>> seedsOf;
  for (Operation *operation : operations)
    if (isAnchor(operation) ||
        (isa<RedistributeOp>(operation) && forward.contains(operation)))
      seedsOf[owner.lookup(operation)].push_back(seeds++);
  int64_t buckets = 64;
  while (seeds * 4 >= buckets * 3)
    buckets *= 2;
  SmallVector<unsigned> merges;
  for (auto [index, group] : llvm::enumerate(groups)) {
    ArrayRef<int64_t> own = seedsOf[index];
    if (own.size() == 1)
      group.id = own.front();
    else
      merges.push_back(index);
  }
  llvm::sort(merges, [&](unsigned left, unsigned right) {
    return seedsOf[left].front() * 37 % buckets <
           seedsOf[right].front() * 37 % buckets;
  });
  for (auto [rank, index] : llvm::enumerate(merges))
    groups[index].id = seeds + static_cast<int64_t>(rank);
  return groups;
}

// Peak tile memory of an unsharded group from the stage 256 slicing: per
// (y, x) tile the sum over its tile values of the union over threads of their
// slice box. This reproduces every unsharded group of the SDK, whose
// evaluator reaches the same number through trial slicing.
FailureOr<int64_t> unshardedPeak(const Group &group) {
  llvm::SmallDenseSet<Operation *> members(group.operations.begin(),
                                           group.operations.end());
  std::map<std::pair<int64_t, int64_t>, int64_t> tiles;
  for (Operation *operation : group.operations) {
    if (isa<DistributedCreateViewOp, ReshapeOpOp>(operation) ||
        isConstantFill(operation) || operation->getNumResults() != 1)
      continue;
    Value result = operation->getResult(0);
    auto begins = operation->getAttrOfType<AffineMapAttr>("slicing_begins");
    auto ends = operation->getAttrOfType<AffineMapAttr>("slicing_ends");
    auto domain = operation->getAttrOfType<ArrayAttr>("slicing_domain");
    if (onHost(result) || !begins || !ends || !domain)
      continue;
    SmallVector<int64_t> extents;
    for (Attribute extent : domain)
      extents.push_back(cast<IntegerAttr>(extent).getInt());
    if (extents.size() < 2)
      return operation->emitOpError("has a slicing domain without tiles");
    FailureOr<int64_t> bytes = elementBytes(result);
    if (failed(bytes))
      return operation->emitOpError("has a result without byte size");
    bool padded = isa<FillOp>(operation);
    if (auto redistribute = dyn_cast<RedistributeOp>(operation))
      padded = !members.contains(redistribute.getInput().getDefiningOp());
    int64_t threads = extents.size() > 2 ? extents[2] : 1;
    for (int64_t y = 0; y < extents[0]; ++y) {
      for (int64_t x = 0; x < extents[1]; ++x) {
        SmallVector<int64_t, 8> low, high;
        for (int64_t thread = 0; thread < threads; ++thread) {
          SmallVector<int64_t> point{y, x};
          if (extents.size() > 2)
            point.push_back(thread);
          SmallVector<int64_t, 8> lo = begins.getValue().compose(point);
          SmallVector<int64_t, 8> hi = ends.getValue().compose(point);
          if (low.empty()) {
            low = lo;
            high = hi;
            continue;
          }
          for (unsigned dim = 0; dim < lo.size(); ++dim) {
            low[dim] = std::min(low[dim], lo[dim]);
            high[dim] = std::max(high[dim], hi[dim]);
          }
        }
        int64_t size = *bytes;
        for (auto [lo, hi] : llvm::zip(low, high))
          size *= hi - lo + 1;
        tiles[{y, x}] += padded ? static_cast<int64_t>(size * 1.05) : size;
      }
    }
  }
  int64_t peak = 0;
  for (auto [tile, size] : tiles)
    peak = std::max(peak, size);
  return peak;
}

std::pair<int64_t, int64_t>
bounds(AffineExpr expression, ArrayRef<std::pair<int64_t, int64_t>> ranges) {
  if (auto dim = dyn_cast<AffineDimExpr>(expression))
    return ranges[dim.getPosition()];
  if (auto constant = dyn_cast<AffineConstantExpr>(expression))
    return {constant.getValue(), constant.getValue()};
  auto binary = cast<AffineBinaryOpExpr>(expression);
  auto [lhsLow, lhsHigh] = bounds(binary.getLHS(), ranges);
  auto [rhsLow, rhsHigh] = bounds(binary.getRHS(), ranges);
  switch (binary.getKind()) {
  case AffineExprKind::Add:
    return {lhsLow + rhsLow, lhsHigh + rhsHigh};
  case AffineExprKind::Mul: {
    int64_t products[] = {lhsLow * rhsLow, lhsLow * rhsHigh, lhsHigh * rhsLow,
                          lhsHigh * rhsHigh};
    return {*llvm::min_element(products), *llvm::max_element(products)};
  }
  case AffineExprKind::FloorDiv:
    return {llvm::divideFloorSigned(lhsLow, rhsLow),
            llvm::divideFloorSigned(lhsHigh, rhsLow)};
  case AffineExprKind::CeilDiv:
    return {llvm::divideCeilSigned(lhsLow, rhsLow),
            llvm::divideCeilSigned(lhsHigh, rhsLow)};
  default:
    llvm_unreachable("tile bounds of a mod expression");
  }
}

bool mentions(AffineMap map, unsigned dim) {
  return llvm::any_of(map.getResults(), [&](AffineExpr expression) {
    return expression.isFunctionOfDim(dim);
  });
}

// Fills `group.coordinates` with every member's result index as a function
// of the anchor's iteration space. The SDK builds the same relation while
// partitioning and uses it for trial slicing and for the emitted sharding
// maps. Groups anchored elsewhere keep whole tensor maps.
LogicalResult relate(Group &group, bool required) {
  Operation *anchor = group.anchor;
  group.coordinates.clear();
  if (isa<StaticComputeOpOp, StaticUnaryComputeOpOp>(anchor)) {
    AffineMap traversal = traversalOf(anchor);
    group.coordinates[anchor] = traversal;
    for (Value operand : anchor->getOperands()) {
      auto view = operand.getDefiningOp<DistributedCreateViewOp>();
      if (!view)
        continue;
      AffineMap own = traversalOf(view);
      group.coordinates[view] = own;
      if (Operation *source = view.getInput().getDefiningOp())
        group.coordinates[source] = own;
    }
  } else if (auto interpolate = dyn_cast<InterpolateHardwareOp>(anchor)) {
    auto results = anchor->getAttrOfType<ArrayAttr>("result_traversals");
    auto operands = anchor->getAttrOfType<ArrayAttr>("operand_traversals");
    auto domain = anchor->getAttrOfType<ArrayAttr>("traversal_domain");
    if (!results || !operands || !domain)
      return anchor->emitOpError("lacks interpolation traversals");
    group.coordinates[anchor] = cast<AffineMapAttr>(results[0]).getValue();
    for (auto [operand, map] : llvm::zip(anchor->getOperands(), operands))
      if (Operation *source = operand.getDefiningOp())
        group.coordinates[source] = cast<AffineMapAttr>(map).getValue();
    group.extents.clear();
    for (Attribute extent : domain)
      group.extents.push_back(cast<IntegerAttr>(extent).getInt());
  } else if (required) {
    return anchor->emitOpError("needs sharding outside compute groups");
  } else {
    return success();
  }

  llvm::SmallDenseSet<Operation *> members(group.operations.begin(),
                                           group.operations.end());
  bool changed = true;
  while (changed) {
    changed = false;
    for (Operation *operation : group.operations) {
      if (group.coordinates.contains(operation))
        continue;
      if (isa<RedistributeOp, DistributedCreateViewOp>(operation)) {
        auto found =
            group.coordinates.find(operation->getOperand(0).getDefiningOp());
        if (found != group.coordinates.end()) {
          group.coordinates[operation] = found->second;
          changed = true;
          continue;
        }
      }
      for (Operation *user : operation->getUsers()) {
        auto found = group.coordinates.find(user);
        if (!members.contains(user) || found == group.coordinates.end() ||
            !isa<RedistributeOp, DistributedCreateViewOp>(user))
          continue;
        group.coordinates[operation] = found->second;
        changed = true;
        break;
      }
    }
  }
  for (Operation *operation : group.operations) {
    if (group.coordinates.contains(operation))
      continue;
    if (required)
      return operation->emitOpError("has no index relation to its anchor");
    group.coordinates.clear();
    return success();
  }
  if (!group.extents.empty())
    return success();
  FailureOr<SmallVector<int64_t>> extents =
      iterationExtents(group.coordinates[anchor].getNumDims(), anchor);
  if (failed(extents))
    return anchor->emitOpError("has no iteration extents");
  group.extents = std::move(*extents);
  return success();
}

struct Evaluation {
  int64_t peak = 0;
  int64_t cost = 0;
  int64_t innerPeak = 0;
};

struct Evaluator {
  const Group &group;
  const DenseMap<Operation *, int64_t> &work;
  SmallVector<unsigned> outputs;
  bool matrix;
  unsigned channel;

  Evaluator(const Group &group, const DenseMap<Operation *, int64_t> &work)
      : group(group), work(work) {
    AffineMap traversal = traversalOf(group.anchor);
    ArrayRef<int64_t> output = shapeOf(group.anchor->getResult(0));
    for (auto [index, expression] : llvm::enumerate(traversal.getResults()))
      if (auto dim = dyn_cast<AffineDimExpr>(expression);
          dim && output[index] > 1)
        outputs.push_back(dim.getPosition());
    channel = cast<AffineDimExpr>(traversal.getResults().back()).getPosition();
    auto options = group.anchor->getAttrOfType<ComputeOpOptionsAttr>("compute");
    std::optional<InnerOperationKind> kind = options.getInnerOperation();
    matrix =
        kind == InnerOperationKind::Vmc || kind == InnerOperationKind::Stencil;
  }

  bool sharded(Operation *operation, std::optional<Shard> shard) const {
    return shard &&
           mentions(group.coordinates.lookup(operation), shard->symbol);
  }

  FailureOr<Evaluation> level(ArrayRef<Operation *> operations,
                              ArrayRef<std::pair<int64_t, int64_t>> ranges,
                              int64_t shards, unsigned trial,
                              bool channelTrial) const {
    Evaluation result;
    for (Operation *operation : operations) {
      Value value = operation->getResult(0);
      if (isa<RedistributeOp>(operation) && onHost(value))
        continue;
      ArrayRef<int64_t> shape = shapeOf(value);
      float ratio = 1.0f;
      int64_t last = 1;
      for (auto [index, expression] :
           llvm::enumerate(group.coordinates.lookup(operation).getResults())) {
        auto [low, high] = bounds(expression, ranges);
        last = std::min<int64_t>(high, shape[index] - 1) -
               std::max<int64_t>(low, 0) + 1;
        ratio *= static_cast<float>(last) / static_cast<float>(shape[index]);
      }
      FailureOr<int64_t> bytes = elementBytes(value);
      if (failed(bytes))
        return operation->emitOpError("has a result without byte size");
      bool constant = isConstantFill(operation);
      bool transfer = isa<RedistributeOp>(operation) ||
                      (isa<FillOp>(operation) && !constant);
      int64_t whole = elementCount(shape) * *bytes;
      float factor = 1.0f;
      if (transfer && onHost(operation->getOperand(0)))
        whole = static_cast<int64_t>(whole * 1.05);
      else if (transfer && mentions(group.coordinates.lookup(operation), trial))
        factor = 1.05f;
      int64_t tile = constant ? 0
                              : static_cast<int64_t>(static_cast<float>(whole) *
                                                     (ratio * factor));
      if (transfer || isa<CreateEmptyTensorOp>(operation))
        result.peak += tile;
      int64_t row = last == shape.back() ? kBandwidth
                                         : std::min(kBandwidth, last * *bytes);
      if (transfer || isa<StaticComputeOpOp, StaticUnaryComputeOpOp>(operation))
        result.cost += ceilDiv(tile, row) * shards;
      if (isa<StaticComputeOpOp, StaticUnaryComputeOpOp>(operation))
        result.cost +=
            ceilDiv(work.lookup(operation) * (channelTrial ? kTiles : 1),
                    kComputeUnits * shards) *
            shards;
    }
    return result;
  }

  FailureOr<Evaluation> evaluate(std::optional<Shard> shard,
                                 unsigned config) const {
    SmallVector<std::pair<int64_t, int64_t>> ranges;
    for (int64_t extent : group.extents)
      ranges.push_back({0, extent - 1});
    if (shard)
      ranges[shard->symbol] = {0, shard->extent - 1};
    unsigned trial = outputs[config];
    bool channelTrial = matrix && trial == channel;
    int64_t extent = ranges[trial].second + 1;
    int64_t slice = ceilDiv(extent, kTiles);
    // The SDK tiling solver aligns a matrix op's channel slice to 32
    // channels, and costs that trial at the whole op's work on every tile.
    if (channelTrial)
      slice = std::min(extent, ceilDiv(slice, 32) * 32);
    ranges[trial] = {0, slice - 1};

    SmallVector<Operation *> outer, inner;
    for (Operation *operation : group.operations)
      (sharded(operation, shard) ? inner : outer).push_back(operation);
    FailureOr<Evaluation> outside =
        level(outer, ranges, 1, trial, channelTrial);
    FailureOr<Evaluation> inside =
        level(inner, ranges, shard ? shard->factor : 1, trial, channelTrial);
    if (failed(outside) || failed(inside))
      return failure();
    return Evaluation{outside->peak + inside->peak,
                      outside->cost + inside->cost, inside->peak};
  }

  // The SDK keeps the first trial config, then takes a later one that fits
  // and costs strictly less, that fits where the kept one does not, or whose
  // peak is lower when neither fits.
  FailureOr<Evaluation> best(std::optional<Shard> shard, int64_t budget) const {
    std::optional<Evaluation> kept;
    for (unsigned config = 0; config < outputs.size(); ++config) {
      FailureOr<Evaluation> current = evaluate(shard, config);
      if (failed(current))
        return failure();
      bool take = !kept;
      if (kept && current->peak > budget && kept->peak > budget)
        take = current->peak < kept->peak;
      else if (kept && kept->peak >= budget)
        take = current->peak < budget;
      else if (kept)
        take = current->peak < budget && current->cost < kept->cost;
      if (take)
        kept = *current;
    }
    if (!kept)
      return group.anchor->emitOpError("has no trial slicing config");
    return *kept;
  }
};

// Each output symbol is tried at factor 2, growing by half while the shard
// still exceeds the budget, and the cheapest fitting candidate wins with the
// later one taken on ties. Reduction symbols are only tried by the SDK after
// these, and never win on the models this port has seen.
LogicalResult chooseShard(Group &group,
                          const DenseMap<Operation *, int64_t> &work,
                          int64_t budget) {
  if (failed(relate(group, true)))
    return failure();
  Evaluator evaluator(group, work);
  std::optional<std::pair<Shard, Evaluation>> chosen;
  for (unsigned symbol : evaluator.outputs) {
    int64_t size = group.extents[symbol];
    for (int64_t factor = 2;; factor = std::min(size, factor * 3 / 2)) {
      Shard shard{symbol, factor, ceilDiv(size, factor)};
      FailureOr<Evaluation> evaluation = evaluator.best(shard, budget);
      if (failed(evaluation))
        return failure();
      if (evaluation->peak < budget) {
        if (!chosen || evaluation->cost <= chosen->second.cost)
          chosen = {shard, *evaluation};
        break;
      }
      if (factor >= size)
        break;
    }
  }
  if (!chosen)
    return group.anchor->emitOpError(
        "fits the tile memory budget under no output sharding");
  group.shard = chosen->first;
  group.peak = chosen->second.peak;
  group.innerPeak = chosen->second.innerPeak;
  return success();
}

// Reverse post order over the group graph. Roots are taken by ascending id
// and each group's successors by ascending (peak, id), as sub_1692B50 does.
SmallVector<unsigned> scheduleGroups(ArrayRef<Group> groups) {
  DenseMap<Operation *, unsigned> owner;
  for (auto [index, group] : llvm::enumerate(groups))
    for (Operation *operation : group.operations)
      owner[operation] = index;
  auto producers = [&](Operation *operation, auto &&self,
                       llvm::SetVector<unsigned> &into) -> void {
    for (Value operand : operation->getOperands()) {
      Operation *producer = operand.getDefiningOp();
      if (!producer)
        continue;
      if (auto found = owner.find(producer); found != owner.end())
        into.insert(found->second);
      else
        self(producer, self, into);
    }
  };
  SmallVector<SmallVector<unsigned>> successors(groups.size());
  for (auto [index, group] : llvm::enumerate(groups)) {
    llvm::SetVector<unsigned> sources;
    for (Operation *operation : group.operations)
      producers(operation, producers, sources);
    for (unsigned source : sources)
      if (source != index)
        successors[source].push_back(index);
  }
  auto before = [&](unsigned left, unsigned right) {
    return std::make_pair(groups[left].peak, groups[left].id) <
           std::make_pair(groups[right].peak, groups[right].id);
  };
  for (SmallVector<unsigned> &list : successors)
    llvm::sort(list, before);
  auto roots = llvm::to_vector(llvm::seq<unsigned>(0, groups.size()));
  llvm::sort(roots, [&](unsigned left, unsigned right) {
    return groups[left].id < groups[right].id;
  });
  SmallVector<bool> visited(groups.size(), false);
  SmallVector<unsigned> post;
  auto visit = [&](unsigned node, auto &&self) -> void {
    visited[node] = true;
    for (unsigned next : successors[node])
      if (!visited[next])
        self(next, self);
    post.push_back(node);
  };
  for (unsigned root : roots)
    if (!visited[root])
      visit(root, visit);
  return SmallVector<unsigned>(llvm::reverse(post));
}

AffineMap wholeMap(MLIRContext *context, unsigned dims, ArrayRef<int64_t> shape,
                   bool upper) {
  SmallVector<AffineExpr> results;
  for (int64_t size : shape)
    results.push_back(getAffineConstantExpr(upper ? size - 1 : 0, context));
  return AffineMap::get(dims, 0, results, context);
}

void markUnsharded(Operation *operation, const Group &group) {
  MLIRContext *context = operation->getContext();
  Builder builder(context);
  AffineMap begins, ends;
  if (auto found = group.coordinates.find(operation);
      found != group.coordinates.end()) {
    SmallVector<AffineExpr> low, high;
    for (int64_t extent : group.extents) {
      low.push_back(getAffineConstantExpr(0, context));
      high.push_back(getAffineConstantExpr(extent - 1, context));
    }
    begins = found->second.replaceDimsAndSymbols(low, {}, 0, 0);
    ends = found->second.replaceDimsAndSymbols(high, {}, 0, 0);
  } else {
    ArrayRef<int64_t> shape = shapeOf(operation->getResult(0));
    begins = wholeMap(context, 0, shape, false);
    ends = wholeMap(context, 0, shape, true);
  }
  operation->setAttr("sharding_begins", AffineMapAttr::get(begins));
  operation->setAttr("sharding_domain", builder.getI32ArrayAttr({}));
  operation->setAttr("sharding_ends", AffineMapAttr::get(ends));
}

void markSharded(Operation *operation, const Group &group) {
  MLIRContext *context = operation->getContext();
  Builder builder(context);
  const Shard &shard = *group.shard;
  AffineExpr index = getAffineDimExpr(0, context);
  SmallVector<AffineExpr> low, high;
  for (auto [dim, extent] : llvm::enumerate(group.extents)) {
    if (dim == shard.symbol) {
      low.push_back(index * shard.extent);
      high.push_back(index * shard.extent + (shard.extent - 1));
      continue;
    }
    low.push_back(getAffineConstantExpr(0, context));
    high.push_back(getAffineConstantExpr(extent - 1, context));
  }
  AffineMap relation = group.coordinates.lookup(operation);
  AffineMap begins = relation.replaceDimsAndSymbols(low, {}, 1, 0);
  AffineMap ends = relation.replaceDimsAndSymbols(high, {}, 1, 0);
  operation->setAttr("sharding_begins", AffineMapAttr::get(begins));
  operation->setAttr(
      "sharding_domain",
      builder.getI32ArrayAttr({static_cast<int32_t>(shard.factor)}));
  operation->setAttr("sharding_ends", AffineMapAttr::get(ends));
}

void resetSlicing(Operation *operation) {
  if (!operation->hasAttr("slicing_domain"))
    return;
  MLIRContext *context = operation->getContext();
  Builder builder(context);
  ArrayRef<int64_t> shape = shapeOf(operation->getResult(0));
  operation->setAttr("slicing_begins",
                     AffineMapAttr::get(wholeMap(context, 2, shape, false)));
  operation->setAttr("slicing_domain", builder.getI32ArrayAttr({1, 1}));
  operation->setAttr("slicing_ends",
                     AffineMapAttr::get(wholeMap(context, 2, shape, true)));
}

affine::AffineParallelOp wrap(OpBuilder &builder,
                              ArrayRef<Operation *> operations,
                              ArrayRef<int64_t> ranges, int64_t peak,
                              int64_t id) {
  auto inside = [&](Operation *user) {
    return llvm::any_of(operations, [&](Operation *member) {
      return member->isAncestor(user);
    });
  };
  SmallVector<Value> outputs;
  for (Operation *operation : operations)
    for (Value result : operation->getResults())
      if (!llvm::all_of(result.getUsers(), inside))
        outputs.push_back(result);
  SmallVector<Type> types =
      llvm::map_to_vector(outputs, [](Value value) { return value.getType(); });
  SmallVector<arith::AtomicRMWKind> reductions(types.size(),
                                               arith::AtomicRMWKind::assign);
  Location location = operations.front()->getLoc();
  auto parallel = affine::AffineParallelOp::create(builder, location, types,
                                                   reductions, ranges);
  parallel->setAttr("IsReduction", builder.getBoolAttr(false));
  parallel->setAttr("PeakMemoryBytes", builder.getI64IntegerAttr(peak));
  parallel->setAttr("ShardingGroupId",
                    builder.getI32IntegerAttr(static_cast<int32_t>(id)));
  Block *body = parallel.getBody();
  for (Operation *operation : operations)
    operation->moveBefore(body, body->end());
  OpBuilder end = OpBuilder::atBlockEnd(body);
  affine::AffineYieldOp::create(end, location, outputs);
  for (auto [value, result] : llvm::zip(outputs, parallel.getResults()))
    value.replaceUsesWithIf(result, [&](OpOperand &use) {
      return !parallel->isAncestor(use.getOwner());
    });
  return parallel;
}

class ShardingPass
    : public PassWrapper<ShardingPass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ShardingPass)

  StringRef getArgument() const final { return "darwinn-sharding"; }

  StringRef getDescription() const final {
    return "Wrap operations into sharding groups and shard groups that "
           "exceed the tile memory budget";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<DarwinnDialect, affine::AffineDialect>();
  }

  void runOnOperation() final {
    func::FuncOp function = getOperation();
    if (!llvm::hasSingleElement(function.getBody())) {
      function.emitOpError("sharding needs a single block body");
      return signalPassFailure();
    }
    Block &block = function.getBody().front();
    FailureOr<SmallVector<Group, 0>> groups = partition(block);
    if (failed(groups))
      return signalPassFailure();

    MaterializedSlicing slicing;
    DenseMap<Operation *, int64_t> work;
    for (const Group &group : *groups) {
      for (Operation *operation : group.operations) {
        if (!isa<StaticComputeOpOp, StaticUnaryComputeOpOp>(operation))
          continue;
        FailureOr<Estimate> direct = slicing.estimate(operation);
        FailureOr<Estimate> transposed = slicing.estimate(operation, true);
        if (failed(direct) || failed(transposed))
          return signalPassFailure();
        work[operation] = std::min(cycles(*direct), cycles(*transposed));
      }
    }

    int64_t budget =
        static_cast<int64_t>(static_cast<float>(kTileBytes) * 0.9f);
    bool sharded = false;
    for (Group &group : *groups) {
      FailureOr<int64_t> peak = unshardedPeak(group);
      if (failed(peak))
        return signalPassFailure();
      group.peak = *peak;
      if (group.peak <= budget)
        continue;
      if (failed(chooseShard(group, work, budget)))
        return signalPassFailure();
      sharded = true;
    }
    if (!sharded) {
      function->setAttr("skip_sharding", BoolAttr::get(&getContext(), true));
      return;
    }

    SmallVector<unsigned> order = scheduleGroups(*groups);
    llvm::SmallDenseSet<Operation *> grouped;
    for (const Group &group : *groups)
      grouped.insert(group.operations.begin(), group.operations.end());
    SmallVector<Operation *> loose;
    for (Operation &operation : block.without_terminator())
      if (!grouped.contains(&operation))
        loose.push_back(&operation);

    OpBuilder builder(block.getTerminator());
    llvm::SmallDenseSet<Operation *> placed;
    for (unsigned index : order) {
      Group &group = (*groups)[index];
      if (!group.shard && failed(relate(group, false)))
        return signalPassFailure();
      for (Operation *operation : group.operations) {
        resetSlicing(operation);
        if (group.shard && group.coordinates.contains(operation) &&
            mentions(group.coordinates.lookup(operation), group.shard->symbol))
          markSharded(operation, group);
        else
          markUnsharded(operation, group);
      }
      Operation *wrapper;
      if (!group.shard) {
        wrapper = wrap(builder, group.operations, {}, group.peak, group.id);
      } else {
        SmallVector<Operation *> outer, inner;
        for (Operation *operation : group.operations)
          (mentions(group.coordinates.lookup(operation), group.shard->symbol)
               ? inner
               : outer)
              .push_back(operation);
        if (outer.empty()) {
          wrapper =
              wrap(builder, inner, {group.shard->factor}, group.peak, group.id);
        } else {
          auto sharded = wrap(builder, inner, {group.shard->factor},
                              group.innerPeak, group.id);
          SmallVector<Operation *> all(outer);
          all.push_back(sharded);
          wrapper = wrap(builder, all, {}, group.peak, group.id);
        }
      }
      placed.insert(wrapper);
      bool moved = true;
      while (moved) {
        moved = false;
        for (Operation *operation : loose) {
          if (placed.contains(operation))
            continue;
          bool ready = llvm::all_of(operation->getOperands(), [&](Value value) {
            Operation *producer = value.getDefiningOp();
            return !producer || placed.contains(producer);
          });
          if (!ready)
            continue;
          operation->moveBefore(block.getTerminator());
          placed.insert(operation);
          moved = true;
        }
      }
    }
    for (Operation *operation : loose)
      if (!placed.contains(operation)) {
        operation->emitOpError("has no producing sharding group");
        return signalPassFailure();
      }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::darwinn::createShardingPass() {
  return std::make_unique<ShardingPass>();
}

void mlir::darwinn::registerShardingPass() { PassRegistration<ShardingPass>(); }
