#include "SlicingModel.h"
#include "mlir/Dialect/Darwinn/IR/DarwinnOps.h"
#include "mlir/Dialect/Darwinn/Transforms/DistributedPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/MapVector.h"

using namespace mlir;
using namespace mlir::darwinn;
using namespace mlir::darwinn::slicing;

namespace {

constexpr int64_t kTileMemoryBytes = 786416;
constexpr double kMaxMemoryRatio = 0.95;
constexpr double kStagingOverhead = 1.05;

bool isAlias(Operation *operation) {
  return isa<DistributedCreateViewOp, GetTensorOp, ReshapeOpOp>(operation);
}

Value aliasRoot(Value value) {
  while (Operation *producer = value.getDefiningOp()) {
    if (!isAlias(producer))
      break;
    value = producer->getOperand(0);
  }
  return value;
}

bool isParameter(Value value) {
  auto fill = aliasRoot(value).getDefiningOp<FillOp>();
  return fill && fill.getConstTypeAttr();
}

bool isHostLoad(Operation *operation) {
  auto redistribute = dyn_cast<RedistributeOp>(operation);
  return redistribute &&
         memorySpaceOf(redistribute.getInput()) ==
             DistributedMemorySpace::HostMemory &&
         memorySpaceOf(redistribute.getOutput()) ==
             DistributedMemorySpace::TileMemory;
}

bool isStaged(Operation *operation) {
  if (auto fill = dyn_cast_or_null<FillOp>(operation))
    return !fill.getConstTypeAttr();
  return operation && isHostLoad(operation);
}

int64_t staged(int64_t bytes) {
  return static_cast<int64_t>(bytes * kStagingOverhead);
}

Operation *slicingOwner(Value value) {
  Operation *operation = value.getDefiningOp();
  while (operation &&
         !operation->getAttrOfType<AffineMapAttr>("slicing_begins")) {
    if (auto compute = dyn_cast<StaticComputeOpOp>(operation))
      operation = compute.getDestination().getDefiningOp();
    else if (auto unary = dyn_cast<StaticUnaryComputeOpOp>(operation))
      operation = unary.getDestination().getDefiningOp();
    else if (isAlias(operation))
      operation = operation->getOperand(0).getDefiningOp();
    else
      return nullptr;
  }
  return operation;
}

// The SDK estimator clips operand reads to the tensor but sizes allocations
// with their out of bounds halo, traced on all 1579 selfie ops.
FailureOr<int64_t> tileBytes(Operation *owner, Type elementType, bool clip) {
  auto begins = owner->getAttrOfType<AffineMapAttr>("slicing_begins");
  auto ends = owner->getAttrOfType<AffineMapAttr>("slicing_ends");
  auto domain = owner->getAttrOfType<ArrayAttr>("slicing_domain");
  if (!begins || !ends || !domain || domain.size() < 2 || domain.size() > 3)
    return owner->emitOpError("has no tile slicing to size");
  int64_t threads =
      domain.size() == 3 ? cast<IntegerAttr>(domain[2]).getInt() : 1;
  ArrayRef<int64_t> shape = shapeOf(owner->getResult(0));
  SmallVector<int64_t> low, high;
  for (int64_t thread = 0; thread < threads; ++thread) {
    SmallVector<int64_t> point{0, 0};
    if (domain.size() == 3)
      point.push_back(thread);
    SmallVector<int64_t> first = begins.getValue().compose(point);
    SmallVector<int64_t> last = ends.getValue().compose(point);
    if (low.empty()) {
      low = first;
      high = last;
      continue;
    }
    for (unsigned dim = 0; dim < low.size(); ++dim) {
      low[dim] = std::min(low[dim], first[dim]);
      high[dim] = std::max(high[dim], last[dim]);
    }
  }
  int64_t elements = 1;
  for (unsigned dim = 0; dim < low.size(); ++dim) {
    int64_t lower = clip ? std::max<int64_t>(low[dim], 0) : low[dim];
    int64_t upper = clip ? std::min(high[dim], shape[dim] - 1) : high[dim];
    elements *= std::max<int64_t>(upper - lower + 1, 0);
  }
  return elements * elementType.getIntOrFloatBitWidth() / 8;
}

FailureOr<int64_t> valueBytes(Value value) {
  Operation *owner = slicingOwner(value);
  if (!owner) {
    if (Operation *producer = value.getDefiningOp())
      return producer->emitOpError("has no slicing owner to size");
    return failure();
  }
  return tileBytes(owner, getElementTypeOrSelf(value.getType()), true);
}

class UsageModel {
public:
  explicit UsageModel(func::FuncOp function) {
    function.walk([&](Operation *operation) {
      if (operation->getNumResults() == 1) {
        positions[operation] = order.size();
        order.push_back(operation);
      }
    });
  }

  LogicalResult build() {
    for (Operation *operation : order) {
      if (!isHostLoad(operation))
        continue;
      FailureOr<int64_t> bytes = tileBytes(
          operation, getElementTypeOrSelf(operation->getResult(0).getType()),
          false);
      if (failed(bytes))
        return failure();
      loads.push_back({operation, positions[operation],
                       lastUse(operation->getResult(0)), *bytes});
    }
    for (Operation *operation : order) {
      FailureOr<int64_t> bytes = usage(operation);
      if (failed(bytes))
        return failure();
      usages.push_back(*bytes);
    }
    return success();
  }

  std::optional<unsigned> position(Operation *operation) const {
    auto found = positions.find(operation);
    if (found == positions.end())
      return std::nullopt;
    return found->second;
  }

  SmallVector<int64_t> usages;

private:
  struct Load {
    Operation *operation;
    unsigned begin;
    unsigned end;
    int64_t bytes;
  };

  unsigned lastUse(Value value) const {
    unsigned last = positions.lookup(value.getDefiningOp());
    for (Operation *user : value.getUsers()) {
      if (isAlias(user)) {
        last = std::max(last, lastUse(user->getResult(0)));
        continue;
      }
      if (std::optional<unsigned> at = position(user))
        last = std::max(last, *at);
    }
    return last;
  }

  FailureOr<int64_t> usage(Operation *operation) {
    llvm::MapVector<Operation *, int64_t> parts;
    auto addOperand = [&](Value operand) -> LogicalResult {
      if (isParameter(operand))
        return success();
      FailureOr<int64_t> bytes = valueBytes(operand);
      if (failed(bytes))
        return failure();
      parts[aliasRoot(operand).getDefiningOp()] = *bytes;
      return success();
    };
    Type element = getElementTypeOrSelf(operation->getResult(0).getType());
    if (auto fill = dyn_cast<FillOp>(operation)) {
      if (!fill.getConstTypeAttr()) {
        FailureOr<int64_t> bytes = tileBytes(operation, element, false);
        if (failed(bytes))
          return failure();
        parts[operation] = staged(*bytes);
      }
    } else if (auto redistribute = dyn_cast<RedistributeOp>(operation)) {
      if (memorySpaceOf(redistribute.getInput()) ==
          DistributedMemorySpace::HostMemory) {
        FailureOr<int64_t> bytes = tileBytes(operation, element, false);
        if (failed(bytes))
          return failure();
        parts[operation] = staged(*bytes);
      } else if (failed(addOperand(redistribute.getInput()))) {
        return failure();
      }
    } else if (isa<StaticComputeOpOp, StaticUnaryComputeOpOp, CopyOpOp,
                   InterpolateHardwareOp>(operation)) {
      for (Value operand : operation->getOperands())
        if (failed(addOperand(operand)))
          return failure();
      if (isa<CopyOpOp, InterpolateHardwareOp>(operation)) {
        FailureOr<int64_t> bytes = tileBytes(operation, element, true);
        if (failed(bytes))
          return failure();
        parts[operation] = *bytes;
      }
    } else if (!isAlias(operation) &&
               !isa<CreateEmptyTensorOp, CommunicatedCreateEmptyTensorOp,
                    CommunicatedCreateWriteViewOp, CommunicatedJoinViewsOp>(
                   operation)) {
      return operation->emitOpError("has no recovered SDK memory usage");
    } else {
      return 0;
    }
    unsigned at = positions[operation];
    for (const Load &load : loads)
      if (load.begin < at && at <= load.end)
        parts.insert({load.operation, load.bytes});
    int64_t total = 0;
    for (auto &[part, bytes] : parts)
      total += bytes;
    return total;
  }

  DenseMap<Operation *, unsigned> positions;
  SmallVector<Operation *> order;
  SmallVector<Load> loads;
};

struct Candidate {
  RedistributeOp spill;
  Value root;
  int64_t bytes;
  SmallVector<std::pair<Operation *, RedistributeOp>> fills;
};

bool isIdentity(MappingAttr mapping) {
  return !mapping ||
         (mapping.getForwardIndexTransformation().getValue().isIdentity() &&
          mapping.getReverseIndexTransformation().getValue().isIdentity());
}

RedistributeOp fillThrough(Operation *user) {
  while (auto view = dyn_cast<DistributedCreateViewOp>(user)) {
    if (!view->hasOneUse())
      return {};
    user = *view->getUsers().begin();
  }
  auto fill = dyn_cast<RedistributeOp>(user);
  if (!fill ||
      memorySpaceOf(fill.getOutput()) != DistributedMemorySpace::TileMemory)
    return {};
  return fill;
}

class SpillFillOptimizationPass
    : public PassWrapper<SpillFillOptimizationPass,
                         OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SpillFillOptimizationPass)

  StringRef getArgument() const final {
    return "darwinn-spill-fill-optimization";
  }

  StringRef getDescription() const final {
    return "Keep spilled tile tensors resident where tile memory allows";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<DarwinnDialect>();
  }

  void runOnOperation() final {
    UsageModel model(getOperation());
    if (failed(model.build()))
      return signalPassFailure();

    SmallVector<Candidate> candidates;
    getOperation().walk([&](RedistributeOp spill) {
      if (memorySpaceOf(spill.getInput()) !=
              DistributedMemorySpace::TileMemory ||
          !isa<DistributedTensorType>(spill.getType()) ||
          memorySpaceOf(spill.getOutput()) !=
              DistributedMemorySpace::HostMemory)
        return;
      Candidate candidate{spill, aliasRoot(spill.getInput()), 0, {}};
      for (Operation *user : spill->getUsers())
        if (RedistributeOp fill = fillThrough(user))
          candidate.fills.push_back({user, fill});
      if (!candidate.fills.empty())
        candidates.push_back(std::move(candidate));
    });

    for (Candidate &candidate : candidates) {
      FailureOr<int64_t> bytes = valueBytes(candidate.spill.getInput());
      if (failed(bytes))
        return signalPassFailure();
      candidate.bytes =
          isStaged(candidate.root.getDefiningOp()) ? staged(*bytes) : *bytes;
      llvm::sort(candidate.fills, [&](const auto &left, const auto &right) {
        return *model.position(left.second) < *model.position(right.second);
      });
    }
    llvm::sort(candidates, [&](const Candidate &left, const Candidate &right) {
      return *model.position(left.spill) > *model.position(right.spill);
    });

    auto limit = static_cast<int64_t>(kTileMemoryBytes * kMaxMemoryRatio);
    SmallVector<int64_t> &usage = model.usages;
    for (Candidate &candidate : candidates) {
      unsigned begin = *model.position(candidate.spill);
      for (auto [head, fill] : candidate.fills) {
        unsigned end = *model.position(fill);
        int64_t peak =
            *std::max_element(usage.begin() + begin, usage.begin() + end + 1);
        if (candidate.bytes + peak > limit)
          continue;
        for (unsigned at = begin; at <= end; ++at)
          usage[at] += candidate.bytes;
        begin = end + 1;
        if (head != fill.getOperation() || !isIdentity(fill.getMappingAttr()) ||
            cast<ShapedType>(candidate.root.getType()).getNumElements() !=
                cast<ShapedType>(fill.getType()).getNumElements())
          continue;
        fill.getInputMutable().assign(candidate.root);
      }
      if (candidate.spill->use_empty())
        candidate.spill.erase();
    }
  }
};

}

std::unique_ptr<Pass> mlir::darwinn::createSpillFillOptimizationPass() {
  return std::make_unique<SpillFillOptimizationPass>();
}

void mlir::darwinn::registerSpillFillOptimizationPass() {
  PassRegistration<SpillFillOptimizationPass>();
}
