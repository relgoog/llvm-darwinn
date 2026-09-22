#include "mlir/Dialect/Darwinn/Transforms/TensorComputeConfiguration.h"
#include "mlir/Dialect/Darwinn/IR/DarwinnOps.h"
#include "mlir/Dialect/Darwinn/Transforms/PlanTensorTraversals.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Verifier.h"
#include <utility>

using namespace mlir;
using namespace mlir::darwinn;

namespace {

LogicalResult unsupported(Operation *operation, const Twine &reason) {
  return operation->emitOpError()
         << "cannot configure tensor computation because " << reason;
}

LogicalResult verifyOptions(Operation *operation,
                            ComputeOpOptionsAttr compute) {
  if (!compute.getInnerOperation() || !compute.getNluFunction() ||
      !compute.getNluPreprocessType() || !compute.getBiasPredicate() ||
      !compute.getScalePredicate() || !compute.getReplicateReduce() ||
      !compute.getTryCellgroups())
    return unsupported(operation, "canonical compute options are required");

  for (auto [name, attribute] :
       {std::pair<StringRef, FloatAttr>{"scale_immediate",
                                        compute.getScaleImmediate()},
        {"bias_immediate", compute.getBiasImmediate()},
        {"clip_lower", compute.getClipLower()},
        {"clip_upper", compute.getClipUpper()}}) {
    if (!attribute || !attribute.getType().isF32())
      return unsupported(operation, name + " must be an explicit f32 value");
  }

  if (compute.getNluPreprocessType() != NluPreprocessKind::None ||
      compute.getBiasPredicate() != NluPredicateKind::None ||
      compute.getScalePredicate() != NluPredicateKind::None ||
      compute.getBitmask())
    return unsupported(operation,
                       "preprocessing and predication require another path");
  if (!compute.getReplicateReduce().getValue().isZero() ||
      *compute.getTryCellgroups())
    return unsupported(operation,
                       "replication and cell group selection are unsupported");

  for (IntegerAttr zeroPoint :
       {compute.getLhsZeroPoint(), compute.getRhsZeroPoint(),
        compute.getOutputZeroPoint()}) {
    if (zeroPoint && !zeroPoint.getValue().isZero())
      return unsupported(operation,
                         "nonzero zero points require a quantized path");
  }

  if (compute.getInputScaleImmediate() ||
      compute.getActivationSubChannelBlockSize() ||
      compute.getParameterSubChannelBlockSize() || compute.getAlpha() ||
      compute.getBeta() || compute.getNumGroups() || compute.getSplineSegment())
    return unsupported(operation,
                       "additional arithmetic options require another path");
  if (compute.getNluE8M0Rounding().value_or(NluE8M0RoundingKind::None) !=
          NluE8M0RoundingKind::None ||
      compute.getLoweringHint().value_or(ComputeLoweringHintKind::None) !=
          ComputeLoweringHintKind::None ||
      compute.getComputeTypeHint().value_or(ComputeTypeHintKind::None) !=
          ComputeTypeHintKind::None)
    return unsupported(operation,
                       "rounding and compute hints require another path");

  return success();
}

}

FailureOr<TensorComputeConfiguration>
mlir::darwinn::deriveTensorComputeConfiguration(
    Operation *operation, TensorTraversalPlanAttr traversal) {
  if (!traversal)
    return unsupported(operation, "a traversal plan is required");
  if (failed(verify(operation, false)))
    return failure();

  ComputeOpOptionsAttr compute;
  Type inputType;
  Type outputType;
  bool unary;

  if (auto unaryOperation = dyn_cast<UnaryTensorOpOp>(operation)) {
    compute = unaryOperation.getCompute();
    inputType = unaryOperation.getInput().getType().getElementType();
    outputType =
        cast<ShapedType>(unaryOperation.getOutput().getType()).getElementType();
    unary = true;
    if (compute.getInnerOperation() != InnerOperationKind::Unary)
      return unsupported(operation, "unary computation requires UNARY");
  } else if (auto binaryOperation = dyn_cast<TensorOpOp>(operation)) {
    compute = binaryOperation.getCompute();
    inputType = binaryOperation.getLhs().getType().getElementType();
    outputType = cast<ShapedType>(binaryOperation.getOutput().getType())
                     .getElementType();
    unary = false;
    if (compute.getInnerOperation() != InnerOperationKind::Elementwise)
      return unsupported(operation, "binary computation requires ELEMENTWISE");
  } else {
    return unsupported(operation,
                       "only scheduled unary and elementwise operations are "
                       "supported");
  }

  if (failed(verifyOptions(operation, compute)))
    return failure();

  auto current = deriveTensorTraversalPlan(operation);
  if (failed(current))
    return failure();
  if (traversal != *current)
    return unsupported(operation, "the traversal plan is stale");

  if ((!inputType.isBF16() && !inputType.isF32()) ||
      (!outputType.isBF16() && !outputType.isF32()))
    return unsupported(operation, "only BF16 and f32 types are supported");

  LinearOperation linear;
  OperandType operandType;
  MacDisable macDisable;
  bool broadcast = false;
  bool needsCoefficients = false;
  bool subtract = false;
  LinearFunctionKind function = compute.getLinearFunction();

  if (unary) {
    if (function == LinearFunctionKind::Max && inputType.isBF16()) {
      linear = LinearOperation::HighBandwidthMaximum;
      operandType = OperandType::Bfloat;
      macDisable = MacDisable::None;
    } else if (function == LinearFunctionKind::Add && inputType.isF32()) {
      linear = LinearOperation::PartialSumAdd;
      operandType = OperandType::Half;
      macDisable = MacDisable::None;
    } else if (function == LinearFunctionKind::Add && inputType.isBF16()) {
      linear = LinearOperation::HighBandwidthMac;
      operandType = OperandType::Bfloat;
      macDisable = MacDisable::SecondFloat;
      needsCoefficients = true;
    } else {
      return unsupported(operation, "the unary linear function is unsupported");
    }
  } else {
    if (!inputType.isBF16())
      return unsupported(operation, "elementwise computation requires BF16");
    if (function == LinearFunctionKind::Mac) {
      linear = LinearOperation::ParallelDotProduct;
    } else if (function == LinearFunctionKind::Add ||
               function == LinearFunctionKind::Sub) {
      linear = LinearOperation::ParallelAccumulate;
      needsCoefficients = true;
      subtract = function == LinearFunctionKind::Sub;
    } else {
      return unsupported(operation,
                         "the elementwise linear function is unsupported");
    }

    auto parameters = traversal.getParameters();
    if (!parameters || parameters.getCounters().empty())
      return unsupported(operation,
                         "elementwise computation requires parameter lanes");

    broadcast = parameters.getCounters().front().getByteStride() == 0;
    operandType = OperandType::Bfloat;
    macDisable = MacDisable::SecondFloat;
  }

  NluFunctionKind nonlinear = *compute.getNluFunction();
  ActivationFunction activation;
  bool symmetric = false;
  std::optional<NluFunctionKind> functionTable;

  if (nonlinear == NluFunctionKind::Linear) {
    activation = ActivationFunction::Relu;
  } else if (nonlinear == NluFunctionKind::Exp) {
    activation = ActivationFunction::Table;
    functionTable = nonlinear;
  } else if (nonlinear == NluFunctionKind::Reciprocal) {
    activation = ActivationFunction::Reciprocal;
    symmetric = true;
  } else {
    return unsupported(operation, "the nonlinear function is unsupported");
  }

  auto bits = [](FloatAttr attribute) {
    return static_cast<uint32_t>(
        attribute.getValue().bitcastToAPInt().getZExtValue());
  };

  TensorComputeConfiguration configuration{
      {linear, operandType, operandType, false, macDisable, broadcast},
      {activation, bits(compute.getScaleImmediate()),
       bits(compute.getBiasImmediate()), bits(compute.getClipUpper()),
       bits(compute.getClipLower()),
       outputType.isBF16() ? OutputType::Bfloat : OutputType::Single, false,
       false, symmetric, symmetric, Preprocess::None, NluPredicate::None,
       NluPredicate::None, 0},
      {},
      functionTable};

  if (needsCoefficients) {
    configuration.coefficients.push_back(
        llvm::APFloat::getOne(llvm::APFloat::BFloat()));
    configuration.coefficients.push_back(
        llvm::APFloat::getOne(llvm::APFloat::BFloat()));
    if (subtract)
      configuration.coefficients.back().changeSign();
  }

  return configuration;
}
