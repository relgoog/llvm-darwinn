#include "mlir/Dialect/Darwinn/IR/DarwinnOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::darwinn;

namespace {

LogicalResult verifyAuxiliaryTensors(Operation *operation, ValueRange tensors,
                                     ArrayAttr kinds) {
  size_t count = kinds ? kinds.size() : 0;
  if (count != tensors.size())
    return operation->emitOpError(
        "requires one auxiliary_tensor_types entry per auxiliary tensor");

  return success();
}

LogicalResult verifyDestination(Operation *operation, Value destination,
                                Value output, AffineMap traversal) {
  auto destinationType = cast<DistributedViewType>(destination.getType());
  auto outputType = cast<ShapedType>(output.getType());
  DistributedMemorySpace outputMemory;

  if (auto tensorType = dyn_cast<DistributedTensorType>(output.getType()))
    outputMemory = tensorType.getMemorySpace();
  else
    outputMemory = cast<DistributedViewType>(output.getType()).getMemorySpace();

  if (destinationType.getShape() != outputType.getShape() ||
      destinationType.getElementType() != outputType.getElementType() ||
      destinationType.getMemorySpace() != outputMemory)
    return operation->emitOpError(
        "requires output shape, element type and memory space to match the "
        "destination view");

  if (traversal.getNumResults() != outputType.getRank())
    return operation->emitOpError(
        "requires traversal result count to match the output rank");

  if (Operation *producer = destination.getDefiningOp()) {
    if (auto destinationTraversal =
            producer->getAttrOfType<AffineMapAttr>("traversal")) {
      if (destinationTraversal.getValue() != traversal)
        return operation->emitOpError(
            "requires traversal to match the destination view traversal");
    }
  }

  return success();
}

LogicalResult verifyComputeOptions(Operation *operation,
                                   ComputeOpOptionsAttr compute,
                                   ArrayAttr auxiliaryKinds, Type outputType) {
  if (compute.getInnerOperation() == InnerOperationKind::InplaceElementwise &&
      compute.getLinearFunction() != LinearFunctionKind::Mac)
    return operation->emitOpError(
        "requires MAC for inplace elementwise computation");

  bool hasBias = false;
  bool hasScale = false;

  if (auxiliaryKinds) {
    for (Attribute attribute : auxiliaryKinds) {
      auto kind = cast<AuxTensorTypeAttr>(attribute).getValue();
      hasBias |=
          kind == AuxTensorKind::Bias || kind == AuxTensorKind::BiasScale;
      hasScale |=
          kind == AuxTensorKind::Scale || kind == AuxTensorKind::BiasScale;
    }
  }

  FloatAttr bias = compute.getBiasImmediate();
  FloatAttr scale = compute.getScaleImmediate();
  bool nonzeroBias = bias && !bias.getValue().isZero();
  bool nonunitScale = false;

  if (scale) {
    llvm::APFloat one = llvm::APFloat::getOne(scale.getValue().getSemantics());
    nonunitScale = scale.getValue().compare(one) != llvm::APFloat::cmpEqual;
  }

  if (hasBias && nonzeroBias &&
      compute.getBiasPredicate().value_or(NluPredicateKind::None) ==
          NluPredicateKind::None)
    return operation->emitOpError(
        "cannot combine an unconditional bias immediate with auxiliary bias");

  if (hasScale && nonunitScale &&
      compute.getScalePredicate().value_or(NluPredicateKind::None) ==
          NluPredicateKind::None)
    return operation->emitOpError(
        "cannot combine an unconditional scale immediate with auxiliary scale");

  auto linear = compute.getLinearFunction();
  bool hasFloatOperand =
      llvm::any_of(operation->getOperandTypes(), [](Type type) {
        Type element = cast<ShapedType>(type).getElementType();
        if (auto vector = dyn_cast<VectorType>(element))
          element = vector.getElementType();

        return isa<FloatType>(element);
      });

  if ((linear == LinearFunctionKind::Argmin ||
       linear == LinearFunctionKind::Argmax ||
       (hasFloatOperand && (linear == LinearFunctionKind::Min ||
                            linear == LinearFunctionKind::Max))) &&
      (hasBias || hasScale || nonzeroBias || nonunitScale))
    return operation->emitOpError(
        "cannot apply bias or scale to ARGMIN, ARGMAX or floating MIN/MAX");

  auto preprocess = compute.getNluPreprocessType();
  bool usesBitmask = preprocess == NluPreprocessKind::AndMask ||
                     preprocess == NluPreprocessKind::OrMask ||
                     preprocess == NluPreprocessKind::XorMask;

  if (usesBitmask != static_cast<bool>(compute.getBitmask()))
    return operation->emitOpError("requires bitmask exactly when AND_MASK, "
                                  "OR_MASK or XOR_MASK is selected");

  auto rounding = compute.getNluE8M0Rounding();
  Type elementType = cast<ShapedType>(outputType).getElementType();
  if (auto vectorType = dyn_cast<VectorType>(elementType))
    elementType = vectorType.getElementType();

  if (rounding && *rounding != NluE8M0RoundingKind::None &&
      !isa<Float8E8M0FNUType>(elementType))
    return operation->emitOpError(
        "requires float8E8M0FNU output for nlu_e8m0_rounding");

  return success();
}

template <typename OpType>
LogicalResult verifyComputeOperation(OpType operation) {
  if (failed(verifyAuxiliaryTensors(operation, operation.getAuxiliaryTensors(),
                                    operation.getAuxiliaryTensorTypesAttr())) ||
      failed(verifyDestination(operation, operation.getDestination(),
                               operation.getOutput(),
                               operation.getTraversal())) ||
      failed(verifyComputeOptions(operation, operation.getCompute(),
                                  operation.getAuxiliaryTensorTypesAttr(),
                                  operation.getOutput().getType())))
    return failure();

  if (auto constraints =
          operation->template getAttrOfType<ArrayAttr>("custom_constraints")) {
    if (constraints.size() != operation.getTraversal().getNumDims())
      return operation.emitOpError(
          "requires one custom constraint per traversal dimension");
  }

  return success();
}

LogicalResult verifyInterpolationTraversals(InterpolateHardwareOp operation) {
  Attribute operandAttribute = operation->getAttr("operand_traversals");
  Attribute resultAttribute = operation->getAttr("result_traversals");
  Attribute domainAttribute = operation->getAttr("traversal_domain");
  if (!operandAttribute && !resultAttribute && !domainAttribute)
    return success();

  auto operands = dyn_cast_or_null<ArrayAttr>(operandAttribute);
  auto results = dyn_cast_or_null<ArrayAttr>(resultAttribute);
  auto domain = dyn_cast_or_null<ArrayAttr>(domainAttribute);

  if (!operands || operands.size() != 1 || !results || results.size() != 1 ||
      !domain)
    return operation.emitOpError(
        "requires one operand traversal, one result traversal and an integer "
        "traversal domain");

  for (Attribute attribute : domain) {
    auto integer = dyn_cast<IntegerAttr>(attribute);
    if (!integer || !integer.getType().isSignlessInteger(32))
      return operation.emitOpError(
          "requires signless i32 traversal_domain entries");
  }

  auto operandMap = dyn_cast<AffineMapAttr>(operands[0]);
  auto resultMap = dyn_cast<AffineMapAttr>(results[0]);
  auto inputType = operation.getInput().getType();
  auto outputType = operation.getOutput().getType();

  if (!operandMap || !resultMap ||
      operandMap.getValue().getNumDims() != domain.size() ||
      resultMap.getValue().getNumDims() != domain.size() ||
      operandMap.getValue().getNumResults() != inputType.getRank() ||
      resultMap.getValue().getNumResults() != outputType.getRank())
    return operation.emitOpError(
        "requires traversal maps from traversal_domain to the operand and "
        "result ranks");

  if (auto constraints = operation.getCustomConstraintsAttr()) {
    if (constraints.size() != domain.size())
      return operation.emitOpError(
          "requires one custom constraint per traversal dimension");
  }

  return success();
}

}

LogicalResult StaticComputeOpOp::verify() {
  return verifyComputeOperation(*this);
}

LogicalResult StaticUnaryComputeOpOp::verify() {
  return verifyComputeOperation(*this);
}

LogicalResult StreamingComputeOpOp::verify() {
  return verifyComputeOperation(*this);
}

LogicalResult StreamingUnaryComputeOpOp::verify() {
  return verifyComputeOperation(*this);
}

LogicalResult SynchronizedComputeOpOp::verify() {
  return verifyComputeOperation(*this);
}

LogicalResult SynchronizedUnaryComputeOpOp::verify() {
  return verifyComputeOperation(*this);
}

LogicalResult TensorOpOp::verify() { return verifyComputeOperation(*this); }

LogicalResult UnaryTensorOpOp::verify() {
  return verifyComputeOperation(*this);
}

LogicalResult NarrowToWideOp::verify() {
  if (failed(verifyAuxiliaryTensors(getOperation(), getAuxiliaryTensors(),
                                    getAuxiliaryTensorTypesAttr())))
    return failure();

  auto inputType = cast<ShapedType>(getInput().getType());
  auto outputType = cast<ShapedType>(getOutput().getType());

  if (getTraversal().getNumResults() != outputType.getRank())
    return emitOpError(
        "requires traversal result count to match the output rank");

  if (auto forward = getForwardIndexTransformationAttr()) {
    if (forward.getValue().getNumDims() != inputType.getRank() ||
        forward.getValue().getNumResults() != outputType.getRank())
      return emitOpError(
          "requires forward_index_transformation from input rank to output "
          "rank");
  }

  if (auto reverse = getReverseIndexTransformationAttr()) {
    if (reverse.getValue().getNumDims() != outputType.getRank() ||
        reverse.getValue().getNumResults() != inputType.getRank())
      return emitOpError(
          "requires reverse_index_transformation from output rank to input "
          "rank");
  }

  return success();
}

LogicalResult InterpolateHardwareOp::verify() {
  auto outputType = getOutput().getType();
  AffineMapAttr begins = getSlicingBeginsAttr();
  AffineMapAttr ends = getSlicingEndsAttr();
  ArrayAttr domain = getSlicingDomainAttr();

  for (AffineMapAttr attribute : {begins, ends}) {
    if (!attribute)
      continue;

    AffineMap map = attribute.getValue();
    if (map.getNumResults() != outputType.getRank() ||
        (domain && map.getNumDims() != domain.size()))
      return emitOpError(
          "requires slicing maps from slicing_domain to the output rank");
  }

  if (begins && ends &&
      begins.getValue().getNumSymbols() != ends.getValue().getNumSymbols())
    return emitOpError("requires matching slicing map symbol counts");

  return verifyInterpolationTraversals(*this);
}
