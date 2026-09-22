#include "mlir/Dialect/Darwinn/Transforms/PropagateDistributedSlices.h"
#include "mlir/Dialect/Darwinn/IR/DarwinnOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>

using namespace mlir;
using namespace mlir::darwinn;

namespace {

struct LinearExpression {
  SmallVector<int64_t, 3> coefficients;
  int64_t constant = 0;
};

FailureOr<LinearExpression> linearize(AffineExpr expression, unsigned rank) {
  LinearExpression result{SmallVector<int64_t, 3>(rank, 0), 0};

  if (auto constant = dyn_cast<AffineConstantExpr>(expression)) {
    result.constant = constant.getValue();
    return result;
  }

  if (auto dimension = dyn_cast<AffineDimExpr>(expression)) {
    if (dimension.getPosition() >= rank)
      return failure();
    result.coefficients[dimension.getPosition()] = 1;
    return result;
  }

  auto binary = dyn_cast<AffineBinaryOpExpr>(expression);
  if (!binary)
    return failure();

  auto left = linearize(binary.getLHS(), rank);
  auto right = linearize(binary.getRHS(), rank);
  if (failed(left) || failed(right))
    return failure();

  if (binary.getKind() == AffineExprKind::Add) {
    if (llvm::AddOverflow(left->constant, right->constant, result.constant))
      return failure();

    for (unsigned axis = 0; axis < rank; ++axis) {
      if (llvm::AddOverflow(left->coefficients[axis], right->coefficients[axis],
                            result.coefficients[axis]))
        return failure();
    }

    return result;
  }

  if (binary.getKind() != AffineExprKind::Mul)
    return failure();

  auto isConstant = [](const LinearExpression &value) {
    return llvm::all_of(value.coefficients,
                        [](int64_t coefficient) { return coefficient == 0; });
  };

  if (isConstant(*left))
    std::swap(left, right);
  if (!isConstant(*right) ||
      llvm::MulOverflow(left->constant, right->constant, result.constant))
    return failure();

  for (unsigned axis = 0; axis < rank; ++axis) {
    if (llvm::MulOverflow(left->coefficients[axis], right->constant,
                          result.coefficients[axis]))
      return failure();
  }

  return result;
}

FailureOr<std::pair<int64_t, int64_t>>
bounds(const LinearExpression &expression, ArrayRef<int32_t> domain) {
  int64_t lower = expression.constant;
  int64_t upper = expression.constant;

  for (auto [coefficient, extent] :
       llvm::zip(expression.coefficients, domain)) {
    int64_t contribution;
    if (llvm::MulOverflow(coefficient, int64_t{extent} - 1, contribution) ||
        llvm::AddOverflow(lower, std::min(int64_t{0}, contribution), lower) ||
        llvm::AddOverflow(upper, std::max(int64_t{0}, contribution), upper))
      return failure();
  }

  return std::make_pair(lower, upper);
}

LogicalResult validateSlice(Operation *diagnostic, Value value,
                            const DistributedSlice &slice) {
  auto shaped = dyn_cast<ShapedType>(value.getType());
  DistributedMemorySpace memory;

  if (auto tensor = dyn_cast<DistributedTensorType>(value.getType()))
    memory = tensor.getMemorySpace();
  else if (auto view = dyn_cast<DistributedViewType>(value.getType()))
    memory = view.getMemorySpace();
  else
    return diagnostic->emitOpError(
        "requires distributed tensor or view slices");

  if (memory != DistributedMemorySpace::TileMemory ||
      !shaped.hasStaticShape() ||
      llvm::any_of(shaped.getShape(), [](int64_t size) { return size <= 0; }) ||
      (!shaped.getElementType().isBF16() && !shaped.getElementType().isF32()))
    return diagnostic->emitOpError(
        "requires positive static BF16 or f32 tile storage");

  if ((slice.domain.size() != 2 && slice.domain.size() != 3) ||
      llvm::any_of(slice.domain, [](int32_t extent) { return extent <= 0; }))
    return diagnostic->emitOpError(
        "requires two or three positive slicing domain extents");

  for (AffineMap map : {slice.begins, slice.ends}) {
    if (!map || map.getContext() != diagnostic->getContext() ||
        map.getNumDims() != slice.domain.size() || map.getNumSymbols() != 0 ||
        map.getNumResults() != shaped.getRank())
      return diagnostic->emitOpError(
          "requires slicing maps from the domain to the storage rank in the "
          "same context without symbols");
  }

  for (unsigned axis = 0; axis < shaped.getRank(); ++axis) {
    auto begin = linearize(slice.begins.getResult(axis), slice.domain.size());
    auto end = linearize(slice.ends.getResult(axis), slice.domain.size());
    if (failed(begin) || failed(end))
      return diagnostic->emitOpError(
          "requires slicing bounds linear in the hardware coordinates without "
          "signed 64-bit overflow");

    auto beginRange = bounds(*begin, slice.domain);
    auto endRange = bounds(*end, slice.domain);
    LinearExpression difference = *end;

    if (llvm::SubOverflow(end->constant, begin->constant, difference.constant))
      return diagnostic->emitOpError(
          "slicing extent exceeds signed 64-bit range");

    for (unsigned index = 0; index < slice.domain.size(); ++index) {
      if (llvm::SubOverflow(end->coefficients[index],
                            begin->coefficients[index],
                            difference.coefficients[index]))
        return diagnostic->emitOpError(
            "slicing extent exceeds signed 64-bit range");
    }

    auto differenceRange = bounds(difference, slice.domain);

    if (failed(beginRange) || failed(endRange) || failed(differenceRange) ||
        beginRange->first < 0 || endRange->second >= shaped.getDimSize(axis) ||
        differenceRange->first < 0)
      return diagnostic->emitOpError(
          "requires nonempty inclusive slices within the storage shape at "
          "every hardware coordinate");
  }

  return success();
}

bool equivalent(const DistributedSlice &left, const DistributedSlice &right) {
  if (left.domain != right.domain)
    return false;

  for (auto [first, second] : {std::pair{left.begins, right.begins},
                               std::pair{left.ends, right.ends}}) {
    if (first.getNumResults() != second.getNumResults())
      return false;

    for (auto [firstExpr, secondExpr] :
         llvm::zip(first.getResults(), second.getResults())) {
      auto firstLinear = linearize(firstExpr, left.domain.size());
      auto secondLinear = linearize(secondExpr, right.domain.size());

      if (failed(firstLinear) || failed(secondLinear) ||
          firstLinear->constant != secondLinear->constant ||
          firstLinear->coefficients != secondLinear->coefficients)
        return false;
    }
  }

  return true;
}

SmallVector<std::pair<Operation *, unsigned>> users(Value value) {
  SmallVector<std::pair<Operation *, unsigned>> result;
  for (OpOperand &use : value.getUses())
    result.emplace_back(use.getOwner(), use.getOperandNumber());
  return result;
}

}

FailureOr<DistributedSlicePropagationPlan>
mlir::darwinn::planDistributedSlicePropagation(
    func::FuncOp function,
    ArrayRef<DistributedSliceAssignment> selectedOutputs) {
  DistributedSlicePropagationPlan plan;
  DenseMap<Value, unsigned> assignmentIndices;
  llvm::SmallPtrSet<Operation *, 32> operations;
  operations.insert(function);

  auto constrain = [&](Value value,
                       const DistributedSlice &slice) -> LogicalResult {
    Operation *owner = value ? value.getDefiningOp() : nullptr;

    if (!owner || owner->getContext() != function.getContext() ||
        owner->getParentOfType<func::FuncOp>() != function)
      return function.emitOpError(
          "slice assignments must name results within this function");
    if (failed(validateSlice(owner, value, slice)))
      return failure();

    auto found = assignmentIndices.find(value);

    if (found != assignmentIndices.end()) {
      if (!equivalent(plan.assignments[found->second].slice, slice))
        return owner->emitOpError(
            "has conflicting distributed slice assignments");
      return success();
    }

    assignmentIndices.try_emplace(value, plan.assignments.size());
    plan.assignments.push_back({value, slice});
    operations.insert(owner);
    return success();
  };

  for (const DistributedSliceAssignment &assignment : selectedOutputs) {
    Operation *owner =
        assignment.output ? assignment.output.getDefiningOp() : nullptr;

    if (!owner || !isa<StaticComputeOpOp, StaticUnaryComputeOpOp>(owner)) {
      function.emitOpError(
          "selected slices must name static elementwise or unary outputs");
      return failure();
    }

    if (failed(constrain(assignment.output, assignment.slice)))
      return failure();
  }

  for (size_t index = 0; index < plan.assignments.size(); ++index) {
    DistributedSliceAssignment assignment = plan.assignments[index];
    Operation *owner = assignment.output.getDefiningOp();
    if (failed(verify(owner, false)))
      return failure();

    if (isa<CreateEmptyTensorOp, FillOp>(owner) ||
        (isa<RedistributeOp>(owner) && owner->getNumOperands() == 1)) {
      plan.storageAssignments.push_back(std::move(assignment));
      continue;
    }

    if (auto view = dyn_cast<DistributedCreateViewOp>(owner)) {
      auto viewType = view.getOutput().getType();
      auto storageType = view.getInput().getType();
      auto forward = view.getForwardIndexTransformationAttr();
      auto reverse = view.getReverseIndexTransformationAttr();

      if (viewType.getShape() != storageType.getShape() ||
          viewType.getElementType() != storageType.getElementType() ||
          (forward && !forward.getValue().isIdentity()) ||
          (reverse && !reverse.getValue().isIdentity())) {
        view.emitOpError(
            "slice propagation requires identity storage views with matching "
            "shape and element type");
        return failure();
      }

      if (failed(constrain(view.getInput(), assignment.slice)))
        return failure();
      continue;
    }

    ComputeOpOptionsAttr compute;
    AffineMap traversal;
    Value destination;
    ValueRange inputs;
    ValueRange auxiliary;
    bool unary = false;

    if (auto binary = dyn_cast<StaticComputeOpOp>(owner)) {
      compute = binary.getCompute();
      traversal = binary.getTraversal();
      destination = binary.getDestination();
      inputs = owner->getOperands().take_front(2);
      auxiliary = binary.getAuxiliaryTensors();
    } else if (auto operation = dyn_cast<StaticUnaryComputeOpOp>(owner)) {
      compute = operation.getCompute();
      traversal = operation.getTraversal();
      destination = operation.getDestination();
      inputs = owner->getOperands().take_front(1);
      auxiliary = operation.getAuxiliaryTensors();
      unary = true;
    } else {
      owner->emitOpError("is outside the supported distributed slicing graph");
      return failure();
    }

    auto expectedInner =
        unary ? InnerOperationKind::Unary : InnerOperationKind::Elementwise;
    auto linear = compute.getLinearFunction();

    if (compute.getInnerOperation() != expectedInner ||
        (linear != LinearFunctionKind::Add &&
         (unary || (linear != LinearFunctionKind::Sub &&
                    linear != LinearFunctionKind::Mac))) ||
        !traversal.isIdentity() || !auxiliary.empty() ||
        (compute.getReplicateReduce() &&
         compute.getReplicateReduce().getInt() != 0) ||
        compute.getTryCellgroups().value_or(false)) {
      owner->emitOpError(
          "slice propagation supports identity-traversal unary ADD and "
          "elementwise ADD, SUB or MAC without auxiliary tensors or "
          "reductions");
      return failure();
    }

    if (failed(constrain(destination, assignment.slice)))
      return failure();

    auto destinationView = destination.getDefiningOp<DistributedCreateViewOp>();

    if (!destinationView ||
        !destinationView.getInput().getDefiningOp<CreateEmptyTensorOp>()) {
      owner->emitOpError(
          "slice propagation requires a destination backed by empty storage");
      return failure();
    }

    auto outputType = cast<ShapedType>(assignment.output.getType());

    for (Value input : inputs) {
      auto view = input.getDefiningOp<DistributedCreateViewOp>();
      auto inputType = cast<DistributedViewType>(input.getType());
      auto projection = view ? view->getAttrOfType<AffineMapAttr>("traversal")
                             : AffineMapAttr{};

      if (!projection || projection.getValue().getNumSymbols() != 0 ||
          projection.getValue().getNumDims() != outputType.getRank() ||
          projection.getValue().getNumResults() != inputType.getRank() ||
          (unary && (inputType.getShape() != outputType.getShape() ||
                     !projection.getValue().isIdentity()))) {
        owner->emitOpError(
            "slice propagation requires operand views with projected "
            "elementwise traversals");
        return failure();
      }

      llvm::SmallBitVector used(outputType.getRank());
      SmallVector<AffineExpr> begins;
      SmallVector<AffineExpr> ends;

      for (auto [axis, expression] :
           llvm::enumerate(projection.getValue().getResults())) {
        if (auto dimension = dyn_cast<AffineDimExpr>(expression)) {
          unsigned position = dimension.getPosition();

          if (position >= outputType.getRank() || used.test(position) ||
              inputType.getDimSize(axis) != outputType.getDimSize(position)) {
            owner->emitOpError(
                "operand projections require distinct dimensions with "
                "matching extents");
            return failure();
          }

          used.set(position);
          begins.push_back(assignment.slice.begins.getResult(position));
          ends.push_back(assignment.slice.ends.getResult(position));
        } else if (auto constant = dyn_cast<AffineConstantExpr>(expression);
                   constant && constant.getValue() == 0 &&
                   inputType.getDimSize(axis) == 1) {
          begins.push_back(expression);
          ends.push_back(expression);
        } else {
          owner->emitOpError(
              "operand projections require dimensions or zero on singleton "
              "storage axes");
          return failure();
        }
      }

      unsigned domainRank = assignment.slice.domain.size();
      auto context = function.getContext();
      DistributedSlice projected{assignment.slice.domain,
                                 AffineMap::get(domainRank, 0, begins, context),
                                 AffineMap::get(domainRank, 0, ends, context)};

      if (failed(constrain(input, projected)))
        return failure();
    }
  }

  for (const DistributedSliceAssignment &assignment : plan.assignments) {
    for (OpOperand &use : assignment.output.getUses()) {
      if (operations.contains(use.getOwner()) ||
          (isa<RedistributeOp>(use.getOwner()) && use.getOperandNumber() == 0))
        continue;

      use.getOwner()->emitOpError(
          "uses storage whose new slicing is not covered by the propagation "
          "plan or a redistribution boundary");
      return failure();
    }
  }

  for (Operation *operation : operations) {
    DistributedSlicePropagationPlan::Snapshot snapshot;
    snapshot.operation = operation;
    snapshot.parent = operation->getParentOp();
    snapshot.block = operation->getBlock();
    snapshot.operands.assign(operation->operand_begin(),
                             operation->operand_end());
    llvm::append_range(snapshot.operandTypes, operation->getOperandTypes());
    llvm::append_range(snapshot.resultTypes, operation->getResultTypes());
    snapshot.attributes = operation->getAttrDictionary();

    for (Value result : operation->getResults())
      snapshot.users.push_back(users(result));

    plan.snapshots.push_back(std::move(snapshot));
  }

  return plan;
}

LogicalResult mlir::darwinn::applyDistributedSlicePropagation(
    const DistributedSlicePropagationPlan &plan) {
  for (const auto &snapshot : plan.snapshots) {
    Operation *operation = snapshot.operation;

    if (operation->getParentOp() != snapshot.parent ||
        operation->getBlock() != snapshot.block ||
        !llvm::equal(operation->getOperands(), snapshot.operands) ||
        !llvm::equal(operation->getOperandTypes(), snapshot.operandTypes) ||
        !llvm::equal(operation->getResultTypes(), snapshot.resultTypes) ||
        operation->getAttrDictionary() != snapshot.attributes)
      return operation->emitOpError(
          "distributed slice propagation plan is stale");

    for (auto [result, expected] :
         llvm::zip(operation->getResults(), snapshot.users)) {
      if (users(result) != expected)
        return operation->emitOpError(
            "distributed slice propagation plan has stale storage users");
    }
  }

  for (const DistributedSliceAssignment &assignment : plan.storageAssignments) {
    if (failed(validateSlice(assignment.output.getDefiningOp(),
                             assignment.output, assignment.slice)))
      return failure();
  }

  for (const DistributedSliceAssignment &assignment : plan.storageAssignments) {
    Operation *owner = assignment.output.getDefiningOp();
    Builder builder(owner->getContext());
    owner->setAttr("slicing_begins",
                   AffineMapAttr::get(assignment.slice.begins));
    owner->setAttr("slicing_domain",
                   builder.getI32ArrayAttr(assignment.slice.domain));
    owner->setAttr("slicing_ends", AffineMapAttr::get(assignment.slice.ends));
  }

  return success();
}
