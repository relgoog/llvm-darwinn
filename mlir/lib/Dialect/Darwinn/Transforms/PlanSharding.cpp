#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Darwinn/IR/DarwinnOps.h"
#include "mlir/Dialect/Darwinn/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CheckedArithmetic.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>

namespace mlir::darwinn {
#define GEN_PASS_DEF_DARWINNPLANSHARDINGPASS
#include "mlir/Dialect/Darwinn/Transforms/Passes.h.inc"
}

using namespace mlir;
using namespace mlir::darwinn;

namespace {

struct TileBounds {
  SmallVector<int64_t> begins;
  SmallVector<int64_t> ends;
};

struct StorageLayout {
  Operation *owner;
  SmallVector<int64_t> shape;
  SmallVector<int64_t> domain;
  AffineMap begins;
  AffineMap ends;
  SmallVector<TileBounds> tiles;
  int64_t elementBytes;
};

struct ShardingGroup {
  Operation *compute;
  SmallVector<Operation *> operations;
  SmallVector<Value> outputs;
  SmallVector<unsigned> dependencies;
  SmallVector<StorageLayout, 2> inputs;
  StorageLayout destination;
  std::optional<unsigned> axis;
  int64_t count = 1;
  int64_t peakBytes = 0;
};

static FailureOr<int64_t> evaluate(AffineExpr expression,
                                   ArrayRef<int64_t> coordinates) {
  if (auto constant = dyn_cast<AffineConstantExpr>(expression))
    return constant.getValue();

  if (auto dimension = dyn_cast<AffineDimExpr>(expression))
    return coordinates[dimension.getPosition()];

  auto binary = dyn_cast<AffineBinaryOpExpr>(expression);
  if (!binary || (binary.getKind() != AffineExprKind::Add &&
                  binary.getKind() != AffineExprKind::Mul))
    return failure();

  FailureOr<int64_t> left = evaluate(binary.getLHS(), coordinates);
  FailureOr<int64_t> right = evaluate(binary.getRHS(), coordinates);
  if (failed(left) || failed(right))
    return failure();

  std::optional<int64_t> result = binary.getKind() == AffineExprKind::Add
                                      ? llvm::checkedAdd(*left, *right)
                                      : llvm::checkedMul(*left, *right);
  if (!result)
    return failure();

  return *result;
}

static bool isIdentity(AffineMapAttr map) {
  return !map || map.getValue().isIdentity();
}

static FailureOr<StorageLayout> readLayout(Operation *operation) {
  auto type =
      dyn_cast<DistributedTensorType>(operation->getResult(0).getType());
  auto begins = operation->getAttrOfType<AffineMapAttr>("slicing_begins");
  auto domain = operation->getAttrOfType<ArrayAttr>("slicing_domain");
  auto ends = operation->getAttrOfType<AffineMapAttr>("slicing_ends");

  if (!type || type.getMemorySpace() != DistributedMemorySpace::TileMemory ||
      !type.hasStaticShape() ||
      llvm::any_of(type.getShape(), [](int64_t size) { return size <= 0; })) {
    operation->emitOpError("sharding requires positive static tile storage");
    return failure();
  }

  if (!begins || !ends || !domain || domain.size() != 3 ||
      begins.getValue().getNumDims() != 3 ||
      ends.getValue().getNumDims() != 3 ||
      begins.getValue().getNumSymbols() != 0 ||
      ends.getValue().getNumSymbols() != 0 ||
      begins.getValue().getNumResults() != type.getRank() ||
      ends.getValue().getNumResults() != type.getRank()) {
    operation->emitOpError(
        "sharding requires preliminary slicing over two tile axes and one "
        "thread axis");
    return failure();
  }

  Type elementType = type.getElementType();
  if (!isa<IntegerType, FloatType>(elementType) ||
      elementType.getIntOrFloatBitWidth() % 8 != 0) {
    operation->emitOpError("sharding requires byte-addressable scalar storage");
    return failure();
  }

  StorageLayout layout{operation,
                       llvm::to_vector(type.getShape()),
                       {},
                       begins.getValue(),
                       ends.getValue(),
                       {},
                       elementType.getIntOrFloatBitWidth() / 8};

  for (Attribute attribute : domain) {
    auto extent = dyn_cast<IntegerAttr>(attribute);
    if (!extent || extent.getInt() < 1 || extent.getInt() > 4) {
      operation->emitOpError(
          "G5 slicing axes must have between one and four entries");
      return failure();
    }

    layout.domain.push_back(extent.getInt());
  }

  for (int64_t tileRow = 0; tileRow < layout.domain[0]; ++tileRow) {
    for (int64_t tileColumn = 0; tileColumn < layout.domain[1]; ++tileColumn) {
      TileBounds tile;

      for (int64_t thread = 0; thread < layout.domain[2]; ++thread) {
        SmallVector<int64_t> coordinates{tileRow, tileColumn, thread};
        TileBounds slice;

        for (unsigned dimension = 0; dimension < type.getRank(); ++dimension) {
          FailureOr<int64_t> lower =
              evaluate(begins.getValue().getResult(dimension), coordinates);
          FailureOr<int64_t> upper =
              evaluate(ends.getValue().getResult(dimension), coordinates);

          if (failed(lower) || failed(upper) || *lower < 0 || *upper < *lower ||
              *upper >= layout.shape[dimension]) {
            operation->emitOpError(
                "sharding requires bounded linear slices without padding or "
                "unresolved transfer staging");
            return failure();
          }

          slice.begins.push_back(*lower);
          slice.ends.push_back(*upper);
        }

        if (thread == 0) {
          tile = std::move(slice);
          continue;
        }

        unsigned changedDimensions = 0;

        for (unsigned dimension = 0; dimension < type.getRank(); ++dimension) {
          if (tile.begins[dimension] == slice.begins[dimension] &&
              tile.ends[dimension] == slice.ends[dimension])
            continue;

          ++changedDimensions;
          if (slice.begins[dimension] > tile.ends[dimension] + 1 ||
              tile.begins[dimension] > slice.ends[dimension] + 1) {
            operation->emitOpError(
                "sharding requires contiguous thread slices");
            return failure();
          }

          tile.begins[dimension] =
              std::min(tile.begins[dimension], slice.begins[dimension]);
          tile.ends[dimension] =
              std::max(tile.ends[dimension], slice.ends[dimension]);
        }

        if (changedDimensions > 1) {
          operation->emitOpError("sharding requires thread slices to differ "
                                 "along at most one axis");
          return failure();
        }
      }

      layout.tiles.push_back(std::move(tile));
    }
  }

  return layout;
}

static bool sameSlicing(const StorageLayout &left, const StorageLayout &right) {
  if (left.shape != right.shape || left.domain != right.domain ||
      left.begins != right.begins || left.ends != right.ends)
    return false;

  return llvm::all_of(llvm::zip(left.tiles, right.tiles), [](const auto &pair) {
    return std::get<0>(pair).begins == std::get<1>(pair).begins &&
           std::get<0>(pair).ends == std::get<1>(pair).ends;
  });
}

static FailureOr<int64_t> allocationBytes(const StorageLayout &layout,
                                          const TileBounds &tile,
                                          std::optional<unsigned> axis,
                                          int64_t count) {
  int64_t bytes = layout.elementBytes;

  for (unsigned dimension = 0; dimension < layout.shape.size(); ++dimension) {
    int64_t extent = tile.ends[dimension] - tile.begins[dimension] + 1;
    if (axis && *axis == dimension)
      extent = layout.shape[dimension] / count +
               (layout.shape[dimension] % count != 0);

    std::optional<int64_t> product = llvm::checkedMul(bytes, extent);
    if (!product)
      return failure();

    bytes = *product;
  }

  return bytes;
}

static FailureOr<int64_t> peakMemory(const ShardingGroup &group,
                                     std::optional<unsigned> axis,
                                     int64_t count) {
  int64_t peak = 0;

  for (auto [index, tile] : llvm::enumerate(group.destination.tiles)) {
    FailureOr<int64_t> output =
        allocationBytes(group.destination, tile, axis, count);
    if (failed(output))
      return failure();

    int64_t total = *output;

    for (const StorageLayout &input : group.inputs) {
      FailureOr<int64_t> bytes =
          allocationBytes(input, input.tiles[index], axis, count);
      if (failed(bytes))
        return failure();

      std::optional<int64_t> padded = llvm::checkedMul(*bytes, int64_t{105});
      if (!padded)
        return failure();

      std::optional<int64_t> sum = llvm::checkedAdd(total, *padded / 100);
      if (!sum)
        return failure();

      total = *sum;
    }

    peak = std::max(peak, total);
  }

  return peak;
}

static LogicalResult selectSharding(ShardingGroup &group, int64_t memoryLimit,
                                    double base, bool preferEvenDivisions) {
  FailureOr<int64_t> baseline = peakMemory(group, std::nullopt, 1);
  if (failed(baseline))
    return group.compute->emitOpError("sharding memory calculation overflowed");

  group.peakBytes = *baseline;
  if (*baseline <= memoryLimit)
    return success();

  SmallVector<unsigned> axes;

  for (unsigned axis = 0; axis + 1 < group.destination.shape.size(); ++axis) {
    int64_t extent = group.destination.shape[axis];
    if (extent <= 1 || extent > std::numeric_limits<int32_t>::max())
      continue;

    if (llvm::all_of(group.destination.tiles, [&](const TileBounds &tile) {
          return tile.begins[axis] == 0 && tile.ends[axis] == extent - 1;
        }))
      axes.push_back(axis);
  }

  if (axes.size() != 1)
    return group.compute->emitOpError(
        "sharding over budget requires exactly one fully replicated spatial "
        "axis until multidimensional and channel planning is available");

  unsigned axis = axes.front();
  int64_t maximum = group.destination.shape[axis];
  SmallVector<int64_t> divisors;

  if (preferEvenDivisions) {
    for (int64_t divisor = 1; divisor <= maximum / divisor; ++divisor) {
      if (maximum % divisor != 0)
        continue;

      divisors.push_back(divisor);
      if (divisor != maximum / divisor)
        divisors.push_back(maximum / divisor);
    }

    llvm::sort(divisors);
  }

  int64_t count = 1;

  while (count < maximum) {
    double product = base * static_cast<double>(count);
    int64_t next = product >= static_cast<double>(maximum)
                       ? maximum
                       : std::max(count + 1, static_cast<int64_t>(product));

    if (preferEvenDivisions) {
      auto divisor = llvm::upper_bound(divisors, count);
      if (divisor != divisors.end() && *divisor <= next)
        next = *divisor;
    }

    FailureOr<int64_t> peak = peakMemory(group, axis, next);
    if (failed(peak))
      return group.compute->emitOpError(
          "sharding memory calculation overflowed");

    if (*peak <= memoryLimit) {
      group.axis = axis;
      group.count = next;
      group.peakBytes = *peak;
      return success();
    }

    count = next;
  }

  return group.compute->emitOpError(
      "no supported spatial shard count fits the G5 search memory budget");
}

class ShardingPlanner {
public:
  explicit ShardingPlanner(func::FuncOp function) : function(function) {}

  LogicalResult plan(int64_t memoryLimit, double base,
                     bool preferEvenDivisions);
  void apply();

private:
  LogicalResult addCompute(Operation *operation);
  FailureOr<Operation *> addView(Value value, ShardingGroup &group,
                                 bool destination);
  LogicalResult own(Operation *operation, ShardingGroup &group);
  LogicalResult finishGroups();

  func::FuncOp function;
  SmallVector<ShardingGroup, 0> groups;
  DenseMap<Operation *, unsigned> owners;
  SmallVector<unsigned> schedule;
};

LogicalResult ShardingPlanner::own(Operation *operation, ShardingGroup &group) {
  auto [position, inserted] = owners.try_emplace(operation, groups.size());
  if (!inserted && position->second != groups.size())
    return operation->emitOpError(
        "sharding cannot merge shared setup across multiple compute groups");

  if (inserted)
    group.operations.push_back(operation);

  return success();
}

FailureOr<Operation *>
ShardingPlanner::addView(Value value, ShardingGroup &group, bool destination) {
  auto view = value.getDefiningOp<DistributedCreateViewOp>();
  if (!view) {
    group.compute->emitOpError(
        "sharding requires explicit local storage views");
    return failure();
  }

  auto inputType = cast<ShapedType>(view.getInput().getType());
  auto outputType = cast<ShapedType>(view.getOutput().getType());
  auto traversal = view->getAttrOfType<AffineMapAttr>("traversal");
  AffineMap computeTraversal =
      group.compute->getAttrOfType<AffineMapAttr>("traversal").getValue();

  if (inputType.getShape() != outputType.getShape() ||
      inputType.getElementType() != outputType.getElementType() ||
      !isIdentity(view.getForwardIndexTransformationAttr()) ||
      !isIdentity(view.getReverseIndexTransformationAttr()) || !traversal ||
      traversal.getValue() != computeTraversal) {
    view.emitOpError(
        "sharding supports shape-preserving views with the compute traversal "
        "and identity index transformations");
    return failure();
  }

  Operation *storage = view.getInput().getDefiningOp();
  if (!storage || storage->getBlock() != group.compute->getBlock()) {
    view.emitOpError("sharding requires storage in the compute block");
    return failure();
  }

  if (destination) {
    if (!isa<CreateEmptyTensorOp>(storage)) {
      view.emitOpError("sharding requires a distinct empty destination tensor");
      return failure();
    }
  } else if (auto transfer = dyn_cast<RedistributeOp>(storage)) {
    auto sourceType = cast<ShapedType>(transfer.getInput().getType());

    if (transfer.getDestination() ||
        sourceType.getShape() != inputType.getShape() ||
        sourceType.getElementType() != inputType.getElementType()) {
      transfer.emitOpError("sharding requires a transfer staging plan for "
                           "changed storage shapes");
      return failure();
    }

    if (MappingAttr mapping = transfer.getMappingAttr()) {
      if (!isIdentity(mapping.getForwardIndexTransformation()) ||
          !isIdentity(mapping.getReverseIndexTransformation())) {
        transfer.emitOpError("sharding requires a transfer staging plan for "
                             "nonidentity mappings");
        return failure();
      }
    }
  } else if (auto fill = dyn_cast<FillOp>(storage)) {
    if (fill.getConstTypeAttr() &&
        fill.getConstTypeAttr().getValue() != ConstKind::None) {
      fill.emitOpError("parameter streaming is outside elementwise sharding");
      return failure();
    }
  } else {
    view.emitOpError(
        "sharding requires an explicit redistribution or fill input");
    return failure();
  }

  if (failed(own(storage, group)) || failed(own(view, group)))
    return failure();

  return storage;
}

LogicalResult ShardingPlanner::addCompute(Operation *operation) {
  ShardingGroup group;
  group.compute = operation;
  Value destination;
  SmallVector<Value> inputs;
  ComputeOpOptionsAttr compute;
  AffineMap traversal;

  if (auto binary = dyn_cast<StaticComputeOpOp>(operation)) {
    destination = binary.getDestination();
    inputs = {binary.getLhs(), binary.getRhs()};
    compute = binary.getCompute();
    traversal = binary.getTraversal();

    if (!binary.getAuxiliaryTensors().empty() || binary.getCustomTilingAttr() ||
        binary.getDtcInfoAttr() || binary.getVexInfoAttr() ||
        compute.getInnerOperation() != InnerOperationKind::Elementwise)
      return binary.emitOpError(
          "sharding currently supports plain elementwise binary computation");
  } else {
    auto unary = cast<StaticUnaryComputeOpOp>(operation);
    destination = unary.getDestination();
    inputs = {unary.getInput()};
    compute = unary.getCompute();
    traversal = unary.getTraversal();

    if (!unary.getAuxiliaryTensors().empty() || unary.getCustomTilingAttr() ||
        unary.getCustomConstraintsAttr() || unary.getVexInfoAttr() ||
        compute.getInnerOperation() != InnerOperationKind::Unary)
      return unary.emitOpError(
          "sharding currently supports plain unary and cast computation");
  }

  if (!traversal.isPermutation() ||
      (compute.getReplicateReduce() &&
       compute.getReplicateReduce().getInt() != 0) ||
      compute.getLoweringHint().value_or(ComputeLoweringHintKind::None) !=
          ComputeLoweringHintKind::None ||
      compute.getComputeTypeHint().value_or(ComputeTypeHintKind::None) !=
          ComputeTypeHintKind::None)
    return operation->emitOpError("sharding requires a non-reduction "
                                  "permutation traversal without target "
                                  "lowering hints");

  FailureOr<Operation *> destinationStorage = addView(destination, group, true);
  if (failed(destinationStorage))
    return failure();

  FailureOr<StorageLayout> outputLayout = readLayout(*destinationStorage);
  if (failed(outputLayout))
    return failure();

  group.destination = std::move(*outputLayout);
  llvm::SmallPtrSet<Operation *, 4> distinctInputs;

  for (Value input : inputs) {
    FailureOr<Operation *> storage = addView(input, group, false);
    if (failed(storage))
      return failure();

    if (!distinctInputs.insert(*storage).second)
      continue;

    FailureOr<StorageLayout> inputLayout = readLayout(*storage);
    if (failed(inputLayout))
      return failure();

    if (!sameSlicing(*inputLayout, group.destination))
      return (*storage)->emitOpError(
          "sharding requires equal input and destination slice geometry");

    group.inputs.push_back(std::move(*inputLayout));
  }

  if (failed(own(operation, group)))
    return failure();

  groups.push_back(std::move(group));
  return success();
}

LogicalResult ShardingPlanner::finishGroups() {
  Block &block = function.getBody().front();

  for (Operation &operation : block.without_terminator()) {
    if (owners.contains(&operation))
      continue;

    auto transfer = dyn_cast<RedistributeOp>(&operation);
    auto source = transfer ? owners.find(transfer.getInput().getDefiningOp())
                           : owners.end();

    if (!transfer || source == owners.end() || transfer.getDestination())
      return operation.emitOpError(
          "operation is outside supported sharding groups");

    auto inputType =
        dyn_cast<DistributedTensorType>(transfer.getInput().getType());
    auto outputType =
        dyn_cast<DistributedTensorType>(transfer.getOutput().getType());
    MappingAttr mapping = transfer.getMappingAttr();

    if (!inputType || !outputType ||
        inputType.getMemorySpace() != DistributedMemorySpace::TileMemory ||
        outputType.getMemorySpace() != DistributedMemorySpace::HostMemory ||
        inputType.getShape() != outputType.getShape() ||
        inputType.getElementType() != outputType.getElementType() ||
        (mapping && (!isIdentity(mapping.getForwardIndexTransformation()) ||
                     !isIdentity(mapping.getReverseIndexTransformation()))))
      return transfer.emitOpError("sharding requires an explicit transfer "
                                  "staging plan for this boundary");

    unsigned owner = source->second;

    if (transfer.getInput().getDefiningOp() != groups[owner].compute)
      return transfer.emitOpError("sharding requires host output to consume "
                                  "its compute result directly");

    auto begins = transfer.getSlicingBegins();
    auto ends = transfer.getSlicingEnds();

    if (begins.getNumSymbols() != 0 || ends.getNumSymbols() != 0)
      return transfer.emitOpError(
          "host output slicing cannot have unbound symbols");

    for (unsigned dimension = 0; dimension < outputType.getRank();
         ++dimension) {
      auto lower = dyn_cast<AffineConstantExpr>(begins.getResult(dimension));
      auto upper = dyn_cast<AffineConstantExpr>(ends.getResult(dimension));

      if (!lower || !upper || lower.getValue() != 0 ||
          upper.getValue() != outputType.getShape()[dimension] - 1)
        return transfer.emitOpError(
            "sharding requires full-range host output slicing");
    }

    groups[owner].operations.push_back(&operation);
    owners.try_emplace(&operation, owner);
  }

  DenseMap<Operation *, unsigned> positions;
  unsigned position = 0;
  for (Operation &operation : block)
    positions.try_emplace(&operation, position++);

  for (auto [index, group] : llvm::enumerate(groups)) {
    unsigned groupIndex = index;
    llvm::sort(group.operations, [&](Operation *left, Operation *right) {
      return positions.lookup(left) < positions.lookup(right);
    });

    for (Operation *operation : group.operations) {
      for (Value operand : operation->getOperands()) {
        if (auto argument = dyn_cast<BlockArgument>(operand)) {
          if (argument.getOwner() != &block ||
              !isa<DistributedTensorType>(argument.getType()) ||
              cast<DistributedTensorType>(argument.getType())
                      .getMemorySpace() != DistributedMemorySpace::HostMemory)
            return operation->emitOpError(
                "sharding accepts only host tensor function inputs");

          continue;
        }

        auto producer = owners.find(operand.getDefiningOp());
        if (producer == owners.end())
          return operation->emitOpError(
              "sharding input has no planned producer");

        if (producer->second != index &&
            !llvm::is_contained(group.dependencies, producer->second))
          group.dependencies.push_back(producer->second);
      }

      for (Value result : operation->getResults()) {
        bool external = llvm::any_of(result.getUsers(), [&](Operation *user) {
          auto owner = owners.find(user);
          return owner == owners.end() || owner->second != groupIndex;
        });

        if (external)
          group.outputs.push_back(result);
      }
    }

    if (group.outputs.empty())
      return group.compute->emitOpError(
          "sharding requires a used group result");

    for (const StorageLayout &input : group.inputs) {
      auto transfer = dyn_cast<RedistributeOp>(input.owner);
      if (!transfer)
        continue;

      auto producer = owners.find(transfer.getInput().getDefiningOp());
      if (producer == owners.end())
        continue;

      const ShardingGroup &previous = groups[producer->second];
      if (transfer.getInput().getDefiningOp() != previous.compute ||
          !sameSlicing(input, previous.destination))
        return transfer.emitOpError("sharding requires a transfer staging plan "
                                    "when producer slicing changes");
    }
  }

  return success();
}

LogicalResult ShardingPlanner::plan(int64_t memoryLimit, double base,
                                    bool preferEvenDivisions) {
  if (!llvm::hasSingleElement(function.getBody()))
    return function.emitOpError(
        "sharding requires a single canonical graph block");

  for (Operation &operation : function.getBody().front().without_terminator()) {
    if (operation.getNumRegions() != 0)
      return operation.emitOpError(
          "sharding requires an ungrouped canonical graph");

    if (operation.hasAttr("sharding_begins") ||
        operation.hasAttr("sharding_domain") ||
        operation.hasAttr("sharding_ends"))
      return operation.emitOpError(
          "sharding requires an unplanned canonical graph");

    if (isa<StaticComputeOpOp, StaticUnaryComputeOpOp>(operation) &&
        failed(addCompute(&operation)))
      return failure();
  }

  if (groups.empty())
    return function.emitOpError(
        "sharding requires static distributed computation");

  if (failed(finishGroups()))
    return failure();

  bool requiresSharding = false;
  for (ShardingGroup &group : groups) {
    if (failed(selectSharding(group, memoryLimit, base, preferEvenDivisions)))
      return failure();

    requiresSharding |= group.count != 1;
  }

  if (!requiresSharding) {
    for (unsigned index = 0; index < groups.size(); ++index)
      schedule.push_back(index);

    return success();
  }

  SmallVector<int64_t> timestamps(groups.size(), -1);

  while (schedule.size() != groups.size()) {
    std::optional<unsigned> selected;
    int64_t selectedReady = -1;

    for (auto [index, group] : llvm::enumerate(groups)) {
      if (timestamps[index] >= 0)
        continue;

      int64_t ready = 0;
      bool available = true;

      for (unsigned dependency : group.dependencies) {
        if (timestamps[dependency] < 0) {
          available = false;
          break;
        }

        ready = std::max(ready, timestamps[dependency]);
      }

      if (!available)
        continue;

      if (!selected || ready > selectedReady ||
          (ready == selectedReady &&
           (group.peakBytes > groups[*selected].peakBytes ||
            (group.peakBytes == groups[*selected].peakBytes &&
             index > *selected)))) {
        selected = index;
        selectedReady = ready;
      }
    }

    if (!selected)
      return function.emitOpError(
          "sharding group dependencies contain a cycle");

    timestamps[*selected] = schedule.size() + 1;
    schedule.push_back(*selected);
  }

  return success();
}

void ShardingPlanner::apply() {
  Block &block = function.getBody().front();
  OpBuilder builder(function.getContext());
  builder.setInsertionPoint(&block.front());
  IRMapping mapping;

  for (unsigned index : schedule) {
    ShardingGroup &group = groups[index];
    SmallVector<Type> types = llvm::map_to_vector(
        group.outputs, [](Value value) { return value.getType(); });
    SmallVector<arith::AtomicRMWKind> reductions(types.size(),
                                                 arith::AtomicRMWKind::assign);
    SmallVector<int64_t> ranges;
    if (group.axis)
      ranges.push_back(group.count);

    auto parallel = affine::AffineParallelOp::create(
        builder, group.compute->getLoc(), types, reductions, ranges);
    parallel->setAttr("ShardingGroupId", builder.getI32IntegerAttr(index));
    parallel->setAttr("PeakMemoryBytes",
                      builder.getI64IntegerAttr(group.peakBytes));
    parallel->setAttr("IsReduction", builder.getBoolAttr(false));

    SmallVector<AffineExpr> begins;
    SmallVector<AffineExpr> ends;

    for (auto [dimension, extent] : llvm::enumerate(group.destination.shape)) {
      AffineExpr lower = builder.getAffineConstantExpr(0);
      int64_t width = extent;

      if (group.axis && *group.axis == dimension) {
        width = extent / group.count + (extent % group.count != 0);
        lower = builder.getAffineDimExpr(0) * width;
      }

      begins.push_back(lower);
      ends.push_back(lower + width - 1);
    }

    AffineMap beginMap =
        AffineMap::get(ranges.size(), 0, begins, builder.getContext());
    AffineMap endMap =
        AffineMap::get(ranges.size(), 0, ends, builder.getContext());
    SmallVector<int32_t> domain(ranges.begin(), ranges.end());
    builder.setInsertionPointToStart(parallel.getBody());

    for (Operation *operation : group.operations) {
      Operation *cloned = builder.clone(*operation, mapping);
      cloned->setAttr("sharding_begins", AffineMapAttr::get(beginMap));
      cloned->setAttr("sharding_domain", builder.getI32ArrayAttr(domain));
      cloned->setAttr("sharding_ends", AffineMapAttr::get(endMap));
    }

    SmallVector<Value> yielded;
    for (Value output : group.outputs)
      yielded.push_back(mapping.lookup(output));

    affine::AffineYieldOp::create(builder, group.compute->getLoc(), yielded);

    for (auto [output, result] :
         llvm::zip(group.outputs, parallel.getResults()))
      mapping.map(output, result);

    builder.setInsertionPointAfter(parallel);
  }

  for (OpOperand &operand : block.getTerminator()->getOpOperands())
    operand.set(mapping.lookupOrDefault(operand.get()));

  SmallVector<Operation *> originals;
  for (ShardingGroup &group : groups)
    llvm::append_range(originals, group.operations);

  llvm::sort(originals, [](Operation *left, Operation *right) {
    return left->isBeforeInBlock(right);
  });

  for (Operation *operation : llvm::reverse(originals))
    operation->erase();
}

struct DarwinnPlanShardingPass
    : darwinn::impl::DarwinnPlanShardingPassBase<DarwinnPlanShardingPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp function = getOperation();
    float ratio = static_cast<float>(memoryLimitRatio.getValue());
    float base = static_cast<float>(greedyBase.getValue());

    if (target != "g5" || !memoryLimitRatio.hasValue() ||
        !greedyBase.hasValue() || !preferEvenDivisions.hasValue() ||
        !std::isfinite(ratio) || ratio <= 0 || ratio > 1 ||
        !std::isfinite(base) || base <= 1) {
      function.emitOpError(
          "requires target g5 and explicit memory-limit-ratio in (0, 1], "
          "greedy-base greater than one and prefer-even-divisions");
      return signalPassFailure();
    }

    float budget = 786432.0f * ratio;
    ShardingPlanner planner(function);

    if (failed(planner.plan(static_cast<int64_t>(budget), base,
                           preferEvenDivisions)))
      return signalPassFailure();

    planner.apply();
  }
};

}

std::unique_ptr<Pass> mlir::darwinn::createDarwinnPlanShardingPass() {
  return std::make_unique<DarwinnPlanShardingPass>();
}
