#include "mlir/Dialect/Darwinn/Transforms/PlanTensorTraversals.h"
#include "mlir/Dialect/Darwinn/IR/DarwinnOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <array>
#include <utility>

using namespace mlir;
using namespace mlir::darwinn;

namespace {

constexpr StringLiteral planAttributeName = "darwinn.traversal_plan";

struct ViewGeometry {
  SmallVector<int64_t> shape;
  SmallVector<int64_t> localShape;
  SmallVector<int64_t> slicingDomain;
  SmallVector<int64_t> byteStrides;
  int64_t elementBytes;
  AffineMap begins;
  AffineMap traversal;
};

struct IterationLoop {
  int64_t tripCount;
  std::array<int64_t, 3> byteStrides;
};

LogicalResult unsupported(Operation *operation, const Twine &reason) {
  return operation->emitOpError()
         << "cannot plan tensor traversal because " << reason;
}

FailureOr<std::pair<int64_t, int64_t>> linearBounds(AffineExpr expression,
                                                    ArrayRef<int64_t> domain) {
  if (auto constant = dyn_cast<AffineConstantExpr>(expression))
    return std::make_pair(constant.getValue(), constant.getValue());
  if (auto dimension = dyn_cast<AffineDimExpr>(expression)) {
    if (dimension.getPosition() >= domain.size())
      return failure();
    return std::make_pair(int64_t{0}, domain[dimension.getPosition()] - 1);
  }

  auto binary = dyn_cast<AffineBinaryOpExpr>(expression);
  if (!binary)
    return failure();

  auto left = linearBounds(binary.getLHS(), domain);
  auto right = linearBounds(binary.getRHS(), domain);
  if (failed(left) || failed(right))
    return failure();

  int64_t lower;
  int64_t upper;

  if (binary.getKind() == AffineExprKind::Add) {
    if (llvm::AddOverflow(left->first, right->first, lower) ||
        llvm::AddOverflow(left->second, right->second, upper))
      return failure();
    return std::make_pair(lower, upper);
  }

  if (binary.getKind() != AffineExprKind::Mul)
    return failure();
  if (left->first != left->second && right->first != right->second)
    return failure();
  if (left->first == left->second)
    std::swap(left, right);
  if (llvm::MulOverflow(left->first, right->first, lower) ||
      llvm::MulOverflow(left->second, right->first, upper))
    return failure();

  return std::make_pair(std::min(lower, upper), std::max(lower, upper));
}

FailureOr<ViewGeometry> readViewGeometry(Operation *operation, Value value) {
  auto view = value.getDefiningOp<DistributedCreateViewOp>();
  if (!view)
    return unsupported(operation,
                       "each operand must come from a distributed view");
  if (failed(verify(view, false)))
    return failure();

  auto type = dyn_cast<DistributedViewType>(value.getType());
  auto storageType = dyn_cast<DistributedTensorType>(view.getInput().getType());

  if (!type || !storageType || !type.hasStaticShape() || type.getRank() == 0 ||
      type.getMemorySpace() != DistributedMemorySpace::TileMemory ||
      storageType.getMemorySpace() != DistributedMemorySpace::TileMemory)
    return unsupported(operation,
                       "operands must be static ranked tile tensors");
  auto forward = view.getForwardIndexTransformationAttr();
  auto reverse = view.getReverseIndexTransformationAttr();

  if ((forward && !forward.getValue().isIdentity()) ||
      (reverse && !reverse.getValue().isIdentity()) ||
      type.getShape() != storageType.getShape() ||
      type.getElementType() != storageType.getElementType())
    return unsupported(operation, "only identity storage views are supported");

  Type element = type.getElementType();
  if (!element.isBF16() && !element.isF32())
    return unsupported(operation, "only BF16 and f32 storage is supported");

  Operation *slice = view.getInput().getDefiningOp();
  if (!slice || !isa<GetTensorOp, CreateEmptyTensorOp>(slice))
    return unsupported(operation,
                       "operand storage requires an explicit tensor slice");
  if (failed(verify(slice, false)))
    return failure();

  auto begins = slice->getAttrOfType<AffineMapAttr>("slicing_begins");
  auto ends = slice->getAttrOfType<AffineMapAttr>("slicing_ends");
  auto domain = slice->getAttrOfType<ArrayAttr>("slicing_domain");
  auto traversal = view->getAttrOfType<AffineMapAttr>("traversal");

  if (!begins || !ends || !domain || !traversal || domain.size() != 3 ||
      begins.getValue().getNumDims() != domain.size() ||
      ends.getValue().getNumDims() != domain.size() ||
      begins.getValue().getNumSymbols() || ends.getValue().getNumSymbols() ||
      begins.getValue().getNumResults() != type.getRank() ||
      ends.getValue().getNumResults() != type.getRank() ||
      traversal.getValue().getNumSymbols() ||
      traversal.getValue().getNumResults() != type.getRank())
    return unsupported(operation,
                       "operand slicing and traversal maps are incomplete");

  SmallVector<int64_t> domainShape;

  for (Attribute entry : domain) {
    auto integer = dyn_cast<IntegerAttr>(entry);
    if (!integer || integer.getInt() <= 0)
      return unsupported(operation, "slicing domain extents must be positive");
    domainShape.push_back(integer.getInt());
  }

  ViewGeometry geometry;
  geometry.shape.assign(type.getShape().begin(), type.getShape().end());
  geometry.elementBytes = element.isBF16() ? 2 : 4;
  geometry.slicingDomain = domainShape;
  geometry.begins = begins.getValue();
  geometry.traversal = traversal.getValue();

  for (unsigned axis = 0; axis < geometry.shape.size(); ++axis) {
    AffineExpr begin = begins.getValue().getResult(axis);
    AffineExpr end = ends.getValue().getResult(axis);
    auto beginBounds = linearBounds(begin, domainShape);
    auto endBounds = linearBounds(end, domainShape);
    auto extent = dyn_cast<AffineConstantExpr>(
        simplifyAffineExpr(end - begin + 1, domain.size(), 0));

    if (failed(beginBounds) || failed(endBounds) || !extent ||
        extent.getValue() <= 0 || beginBounds->first < 0 ||
        endBounds->second >= geometry.shape[axis])
      return unsupported(operation,
                         "slices must have uniform positive in-bounds extents");

    geometry.localShape.push_back(extent.getValue());
  }

  geometry.byteStrides.resize(geometry.shape.size());
  int64_t stride = geometry.elementBytes;

  for (size_t axis = geometry.shape.size(); axis > 0; --axis) {
    geometry.byteStrides[axis - 1] = stride;
    if (geometry.shape[axis - 1] <= 0 ||
        llvm::MulOverflow(stride, geometry.shape[axis - 1], stride))
      return unsupported(operation,
                         "stored tensor byte size exceeds signed 64-bit range");
  }

  return geometry;
}

FailureOr<SmallVector<int64_t>> projectStrides(Operation *operation,
                                               const ViewGeometry &source,
                                               const ViewGeometry &operand) {
  unsigned rank = source.shape.size();
  if (operand.traversal.getNumDims() != rank)
    return unsupported(operation,
                       "operand traversals require a common loop domain");
  if (operand.slicingDomain != source.slicingDomain)
    return unsupported(operation, "operand slicing domains must match");

  SmallVector<int64_t> strides(rank, 0);

  for (unsigned axis = 0; axis < operand.shape.size(); ++axis) {
    AffineExpr expression = operand.traversal.getResult(axis);

    if (auto dimension = dyn_cast<AffineDimExpr>(expression)) {
      unsigned position = dimension.getPosition();
      auto difference = dyn_cast<AffineConstantExpr>(simplifyAffineExpr(
          operand.begins.getResult(axis) - source.begins.getResult(position), 3,
          0));

      if (operand.shape[axis] != source.shape[position] ||
          operand.localShape[axis] != source.localShape[position] ||
          !difference || difference.getValue() != 0)
        return unsupported(
            operation, "projected operand slices must align with the input");
      if (llvm::AddOverflow(strides[position], operand.byteStrides[axis],
                            strides[position]))
        return unsupported(operation,
                           "projected byte strides exceed signed 64-bit range");
    } else if (auto constant = dyn_cast<AffineConstantExpr>(expression)) {
      if (constant.getValue() != 0 || operand.shape[axis] != 1 ||
          operand.localShape[axis] != 1)
        return unsupported(
            operation,
            "broadcast coordinates must select singleton dimensions");
    } else {
      return unsupported(
          operation,
          "operand traversals must use dimensions or zero broadcasts");
    }
  }

  return strides;
}

LogicalResult appendCoalesced(Operation *operation,
                              SmallVectorImpl<IterationLoop> &loops,
                              IterationLoop outer) {
  if (outer.tripCount == 1)
    return success();

  if (!loops.empty()) {
    IterationLoop &inner = loops.back();
    bool contiguous = true;

    for (unsigned operand = 0; operand < inner.byteStrides.size(); ++operand) {
      int64_t expected;
      if (llvm::MulOverflow(inner.byteStrides[operand], inner.tripCount,
                            expected))
        return unsupported(operation,
                           "coalesced byte strides exceed signed 64-bit range");
      contiguous &= outer.byteStrides[operand] == expected;
    }

    if (contiguous) {
      if (llvm::MulOverflow(inner.tripCount, outer.tripCount, inner.tripCount))
        return unsupported(operation,
                           "coalesced trip count exceeds signed 64-bit range");
      return success();
    }
  }

  loops.push_back(outer);
  return success();
}

LogicalResult verifyPlanningOptions(Operation *operation,
                                    ComputeOpOptionsAttr compute,
                                    ValueRange auxiliaries) {
  if (!auxiliaries.empty())
    return unsupported(operation, "auxiliary tensor geometry is not supported");
  if (compute.getReplicateReduce() &&
      compute.getReplicateReduce().getInt() != 0)
    return unsupported(operation, "replicated reduction is not supported");
  if (compute.getTryCellgroups().value_or(false))
    return unsupported(operation,
                       "cell group selection requires a separate kernel plan");
  if (compute.getLoweringHint().value_or(ComputeLoweringHintKind::None) !=
      ComputeLoweringHintKind::None)
    return unsupported(operation,
                       "explicit compute lowering hints are not supported");

  return success();
}

struct PlanGeometry {
  SmallVector<int64_t> main;
  SmallVector<std::pair<int64_t, int64_t>> read;
  SmallVector<std::pair<int64_t, int64_t>> write;
  SmallVector<std::pair<int64_t, int64_t>> parameters;
  SmallVector<std::pair<int64_t, int64_t>> sums;
  int64_t readBytes;
  int64_t writeBytes;
};

FailureOr<TensorTraversalPlanAttr>
materializePlan(Operation *operation, const PlanGeometry &geometry) {
  MLIRContext *context = operation->getContext();
  auto emitError = [&]() { return operation->emitOpError(); };
  auto path = [&](ArrayRef<std::pair<int64_t, int64_t>> loops,
                  ByteAccessPlanAttr access) -> MemoryTraversalPlanAttr {
    SmallVector<LoopCounterAttr> counters;

    for (auto [count, stride] : loops) {
      auto counter =
          LoopCounterAttr::getChecked(emitError, context, count, stride);
      if (!counter)
        return {};
      counters.push_back(counter);
    }

    return MemoryTraversalPlanAttr::getChecked(
        emitError, context, ArrayRef<LoopCounterAttr>(counters), access);
  };

  auto readAccess = ByteAccessPlanAttr::getChecked(
      emitError, context, uint32_t{2}, geometry.readBytes, geometry.readBytes);
  auto writeAccess =
      ByteAccessPlanAttr::getChecked(emitError, context, uint32_t{0},
                                     geometry.writeBytes, geometry.writeBytes);

  if (!readAccess || !writeAccess)
    return failure();

  MemoryTraversalPlanAttr read = path(geometry.read, readAccess);
  MemoryTraversalPlanAttr write = path(geometry.write, writeAccess);
  MemoryTraversalPlanAttr sums = path(geometry.sums, {});
  MemoryTraversalPlanAttr parameters;

  if (!geometry.parameters.empty()) {
    parameters = path(geometry.parameters, {});
    if (!parameters)
      return failure();
  }

  if (!read || !write || !sums)
    return failure();

  auto plan = TensorTraversalPlanAttr::getChecked(
      emitError, context, ArrayRef<int64_t>(geometry.main), read, write,
      parameters, sums);
  if (!plan)
    return failure();
  return plan;
}

FailureOr<TensorTraversalPlanAttr> planUnary(UnaryTensorOpOp operation) {
  if (operation.getCompute().getInnerOperation().value_or(
          InnerOperationKind::Unary) != InnerOperationKind::Unary)
    return unsupported(operation,
                       "unary planning requires UNARY inner operation");
  if (failed(verifyPlanningOptions(operation, operation.getCompute(),
                                   operation.getAuxiliaryTensors())))
    return failure();
  if (!operation.getSlice().empty() || operation.getResamplerOptionsAttr())
    return unsupported(
        operation,
        "explicit slices and resampler options require another plan");

  if (auto constraints = operation.getCustomConstraintsAttr()) {
    if (llvm::any_of(constraints, [](Attribute entry) {
          return cast<IntegerAttr>(entry).getInt() != 0;
        }))
      return unsupported(operation,
                         "custom traversal constraints are not supported");
  }

  auto input = readViewGeometry(operation, operation.getInput());
  auto output = readViewGeometry(operation, operation.getDestination());
  if (failed(input) || failed(output))
    return failure();
  if (!input->traversal.isIdentity() ||
      input->shape.size() != output->shape.size() ||
      operation.getTraversal() != output->traversal)
    return unsupported(operation, "unary planning requires identity input "
                                  "traversal and matching output rank");

  if (failed(projectStrides(operation, *input, *output)))
    return failure();

  SmallVector<unsigned> reductions;

  for (unsigned axis = 0; axis < input->shape.size(); ++axis) {
    AffineExpr expression = output->traversal.getResult(axis);
    if (auto dimension = dyn_cast<AffineDimExpr>(expression)) {
      if (dimension.getPosition() != axis)
        return unsupported(operation,
                           "unary output dimensions cannot be permuted");
    } else if (input->localShape[axis] > 1) {
      reductions.push_back(axis);
    }
  }

  int64_t features = input->localShape.back();
  bool additive =
      operation.getCompute().getLinearFunction() == LinearFunctionKind::Add;
  bool maximum =
      operation.getCompute().getLinearFunction() == LinearFunctionKind::Max;

  if (!additive && !maximum)
    return unsupported(operation, "unary planning supports ADD and MAX only");

  PlanGeometry geometry;
  geometry.readBytes = std::max(int64_t{4}, std::min(features, int64_t{8}) *
                                                input->elementBytes);
  geometry.writeBytes = std::min(features, int64_t{8}) * output->elementBytes;
  geometry.write.emplace_back(1, geometry.writeBytes);

  if (geometry.readBytes > 16)
    return unsupported(operation,
                       "the unary input lane group exceeds sixteen bytes");

  if (reductions.empty()) {
    if (!additive || input->elementBytes != 4 || output->elementBytes != 2 ||
        !output->traversal.isIdentity())
      return unsupported(
          operation,
          "nonreducing unary planning supports f32 to BF16 ADD only");

    SmallVector<IterationLoop> loops;

    for (size_t axis = input->shape.size() - 1; axis > 0; --axis) {
      unsigned dimension = axis - 1;
      IterationLoop loop{
          input->localShape[dimension],
          {input->byteStrides[dimension], output->byteStrides[dimension], 0}};

      if (failed(appendCoalesced(operation, loops, loop)))
        return failure();
    }

    if (loops.empty())
      return unsupported(operation,
                         "a scalar unary kernel requires a separate lane plan");

    for (const IterationLoop &loop : loops) {
      geometry.main.push_back(loop.tripCount);
      geometry.read.emplace_back(loop.tripCount, loop.byteStrides[0]);
      geometry.write.emplace_back(loop.tripCount, loop.byteStrides[1]);
    }
  } else {
    if (reductions.size() != 1 || reductions.front() + 1 == input->shape.size())
      return unsupported(
          operation,
          "unary planning requires one non-feature reduction dimension");

    unsigned reduction = reductions.front();

    for (unsigned axis = 0; axis + 1 < input->shape.size(); ++axis) {
      if (axis != reduction && input->localShape[axis] != 1)
        return unsupported(
            operation,
            "unary reduction outer dimensions must be singleton slices");
    }

    bool blocked = additive && input->elementBytes == 2;
    if ((!blocked && features > 8) ||
        (blocked && features > 8 && features % 8 != 0))
      return unsupported(operation,
                         "unary feature tails require a separate lane plan");
    if ((maximum && (input->elementBytes != 2 || output->elementBytes != 2)) ||
        (input->elementBytes == 4 && output->elementBytes != 2))
      return unsupported(operation,
                         "the unary reduction element types are not supported");

    int64_t count = input->localShape[reduction];
    geometry.main.push_back(count);
    geometry.read.emplace_back(count, input->byteStrides[reduction]);

    if (blocked) {
      int64_t blocks = features / 8 + (features % 8 != 0);
      geometry.main.push_back(blocks);
      geometry.read.emplace_back(blocks, 16);
      geometry.write.emplace_back(blocks, 8 * output->elementBytes);

      for (int64_t trips : geometry.main)
        geometry.parameters.emplace_back(trips, 0);
    }
  }

  for (int64_t count : geometry.main)
    geometry.sums.emplace_back(count, 0);

  return materializePlan(operation, geometry);
}

FailureOr<TensorTraversalPlanAttr> planElementwise(TensorOpOp operation) {
  if (failed(verifyPlanningOptions(operation, operation.getCompute(),
                                   operation.getAuxiliaryTensors())))
    return failure();

  for (Attribute attribute : operation.getShards()) {
    auto shard = cast<TensorOpShardAttr>(attribute);
    if ((shard.getShardId() && !shard.getShardId().empty()) ||
        (shard.getSlices() && !shard.getSlices().empty()))
      return unsupported(operation,
                         "explicit tensor shards require a separate loop plan");
  }

  if (operation.getShards().size() != 1)
    return unsupported(operation,
                       "elementwise planning requires one unsliced shard");

  LinearFunctionKind linear = operation.getCompute().getLinearFunction();
  if (linear != LinearFunctionKind::Add && linear != LinearFunctionKind::Sub &&
      linear != LinearFunctionKind::Mac)
    return unsupported(operation,
                       "elementwise planning supports ADD, SUB and MAC only");

  auto input = readViewGeometry(operation, operation.getLhs());
  auto parameter = readViewGeometry(operation, operation.getRhs());
  auto output = readViewGeometry(operation, operation.getDestination());
  if (failed(input) || failed(parameter) || failed(output))
    return failure();
  if (input->elementBytes != 2 || parameter->elementBytes != 2 ||
      output->elementBytes != 2)
    return unsupported(
        operation, "elementwise planning requires BF16 operands and output");
  if (!input->traversal.isIdentity() || !output->traversal.isIdentity() ||
      operation.getTraversal() != output->traversal)
    return unsupported(
        operation,
        "elementwise input and output traversals must be identity maps");

  auto parameterStrides = projectStrides(operation, *input, *parameter);
  auto outputStrides = projectStrides(operation, *input, *output);
  if (failed(parameterStrides) || failed(outputStrides))
    return failure();

  SmallVector<IterationLoop> loops;

  for (size_t axis = input->shape.size(); axis > 0; --axis) {
    unsigned dimension = axis - 1;
    IterationLoop loop{input->localShape[dimension],
                       {input->byteStrides[dimension],
                        (*parameterStrides)[dimension],
                        (*outputStrides)[dimension]}};

    if (failed(appendCoalesced(operation, loops, loop)))
      return failure();
  }

  if (loops.empty() || loops.front().tripCount % 8 != 0)
    return unsupported(
        operation,
        "elementwise innermost extent must divide into eight-element lanes");

  loops.front().tripCount /= 8;

  for (int64_t &stride : loops.front().byteStrides) {
    if (llvm::MulOverflow(stride, int64_t{8}, stride))
      return unsupported(operation,
                         "lane byte strides exceed signed 64-bit range");
  }

  PlanGeometry geometry;
  geometry.main.push_back(1);
  geometry.readBytes = 16;
  geometry.writeBytes = 16;
  geometry.write.emplace_back(1, 16);

  if (linear != LinearFunctionKind::Mac)
    geometry.sums.emplace_back(1, 0);

  for (const IterationLoop &loop : loops) {
    geometry.main.push_back(loop.tripCount);
    geometry.read.emplace_back(loop.tripCount, loop.byteStrides[0]);
    geometry.parameters.emplace_back(loop.tripCount, loop.byteStrides[1]);
    geometry.write.emplace_back(loop.tripCount, loop.byteStrides[2]);
    geometry.sums.emplace_back(loop.tripCount, 0);
  }

  return materializePlan(operation, geometry);
}

struct PlanTensorTraversalsPass
    : PassWrapper<PlanTensorTraversalsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PlanTensorTraversalsPass)

  StringRef getArgument() const final {
    return "darwinn-plan-tensor-traversals";
  }
  StringRef getDescription() const final {
    return "Plan traversal geometry for scheduled unary and elementwise "
           "tensors";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<DarwinnDialect>();
  }

  void runOnOperation() final {
    SmallVector<std::pair<Operation *, TensorTraversalPlanAttr>> plans;

    WalkResult result = getOperation().walk([&](Operation *operation) {
      if (!isTensorTraversalPlanningCandidate(operation))
        return WalkResult::advance();

      auto plan = deriveTensorTraversalPlan(operation);
      if (failed(plan))
        return WalkResult::interrupt();

      plans.emplace_back(operation, *plan);
      return WalkResult::advance();
    });

    if (result.wasInterrupted()) {
      signalPassFailure();
      return;
    }

    for (auto [operation, plan] : plans)
      operation->setAttr(planAttributeName, plan);
  }
};

}

bool mlir::darwinn::isTensorTraversalPlanningCandidate(Operation *operation) {
  if (isa<UnaryTensorOpOp>(operation))
    return true;
  auto tensor = dyn_cast<TensorOpOp>(operation);
  return tensor && tensor.getCompute() &&
         tensor.getCompute().getInnerOperation() ==
             InnerOperationKind::Elementwise;
}

FailureOr<TensorTraversalPlanAttr>
mlir::darwinn::deriveTensorTraversalPlan(Operation *operation) {
  if (failed(verify(operation, false)))
    return failure();
  if (auto unary = dyn_cast<UnaryTensorOpOp>(operation))
    return planUnary(unary);
  if (auto tensor = dyn_cast<TensorOpOp>(operation)) {
    if (tensor.getCompute().getInnerOperation() ==
        InnerOperationKind::Elementwise)
      return planElementwise(tensor);
  }

  return unsupported(
      operation,
      "this compute family is outside unary and elementwise planning");
}

LogicalResult mlir::darwinn::verifyTensorTraversalPlan(Operation *operation) {
  auto existing =
      operation->getAttrOfType<TensorTraversalPlanAttr>(planAttributeName);
  if (!existing)
    return operation->emitOpError(
        "requires a tensor traversal plan before allocation");

  auto current = deriveTensorTraversalPlan(operation);
  if (failed(current))
    return failure();
  if (existing != *current)
    return operation->emitOpError(
        "tensor traversal plan is stale after a geometry change");

  return success();
}

std::unique_ptr<Pass> mlir::darwinn::createPlanTensorTraversalsPass() {
  return std::make_unique<PlanTensorTraversalsPass>();
}

void mlir::darwinn::registerPlanTensorTraversalsPass() {
  PassRegistration<PlanTensorTraversalsPass>();
}
