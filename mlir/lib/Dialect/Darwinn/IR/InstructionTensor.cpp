#include "mlir/Dialect/Darwinn/IR/InstructionOps.h"
#include <algorithm>

using namespace mlir;
using namespace mlir::darwinn::isa;

LogicalResult
LinearAttr::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                   LinearOperation operation, OperandType activationType,
                   OperandType parameterType, bool, MacDisable macDisable, bool,
                   bool, PartialSumRead partialSumRead, bool, bool,
                   unsigned activationZeroPoint, unsigned parameterZeroPoint,
                   unsigned zeroPointBaseAddress, unsigned zeroPointStride,
                   unsigned zeroPointLimit, bool, bool,
                   SparseParameters sparseParameters) {
  if (!symbolizeLinearOperation(static_cast<uint32_t>(operation)) ||
      !symbolizeOperandType(static_cast<uint32_t>(activationType)) ||
      !symbolizeOperandType(static_cast<uint32_t>(parameterType)) ||
      !symbolizeMacDisable(static_cast<uint32_t>(macDisable)) ||
      !symbolizePartialSumRead(static_cast<uint32_t>(partialSumRead)) ||
      !symbolizeSparseParameters(static_cast<uint32_t>(sparseParameters)))
    return emitError() << "invalid linear control enumeration";
  if (activationZeroPoint > UINT16_MAX || parameterZeroPoint > UINT16_MAX)
    return emitError()
           << "activation and parameter zero points must fit in 16 bits";
  if (zeroPointBaseAddress >= 64 || zeroPointStride >= 64 ||
      zeroPointLimit >= 64)
    return emitError()
           << "zero point address, stride, and limit must fit in 6 bits";
  return success();
}

LogicalResult NonLinearAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError, bool,
    ActivationFunction operation, FloatAttr activationPipelineScale, int32_t,
    FloatAttr highClipValue, FloatAttr lowClipValue, OutputType outputType,
    unsigned offset, bool, bool, bool, bool, Preprocess preprocess,
    NluPredicate biasPredicate, NluPredicate scalePredicate, uint32_t) {
  if (!symbolizeActivationFunction(static_cast<uint32_t>(operation)) ||
      !symbolizeOutputType(static_cast<uint32_t>(outputType)) ||
      !symbolizePreprocess(static_cast<uint32_t>(preprocess)) ||
      !symbolizeNluPredicate(static_cast<uint32_t>(biasPredicate)) ||
      !symbolizeNluPredicate(static_cast<uint32_t>(scalePredicate)))
    return emitError() << "invalid nonlinear control enumeration";
  for (FloatAttr value : {activationPipelineScale, highClipValue, lowClipValue})
    if (!value || !value.getType().isF32())
      return emitError()
             << "nonlinear scale and clip values must have type f32";
  if (offset > UINT16_MAX)
    return emitError() << "nonlinear offset must fit in 16 bits";
  return success();
}

LogicalResult ComputedSamplePointAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError, unsigned loopDepthX,
    unsigned loopDepthY, uint32_t firstX, uint32_t firstY, uint32_t strideX,
    uint32_t strideY) {
  if (loopDepthX >= 8 || loopDepthY >= 8)
    return emitError() << "sample point loop depths must be between 0 and 7";
  if (std::max({firstX, firstY, strideX, strideY}) >= (uint32_t{1} << 17))
    return emitError()
           << "sample point coordinates and strides must fit in 17 bits";
  return success();
}

LogicalResult
InterpolationAttr::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                          ComputedSamplePointAttr samplePoint, uint32_t,
                          unsigned sizeXz, InterpolationMode mode) {
  if (!symbolizeInterpolationMode(static_cast<uint32_t>(mode)))
    return emitError() << "invalid interpolation mode";
  if (sizeXz > UINT16_MAX)
    return emitError() << "interpolation xz size must fit in 16 bits";
  if (samplePoint)
    return ComputedSamplePointAttr::verify(
        emitError, samplePoint.getLoopDepthX(), samplePoint.getLoopDepthY(),
        samplePoint.getFirstX(), samplePoint.getFirstY(),
        samplePoint.getStrideX(), samplePoint.getStrideY());
  return success();
}

LogicalResult TensorControlAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError, LinearAttr linear,
    NonLinearAttr nonLinear, InterpolationAttr interpolation,
    DenseBoolArrayAttr threadMulticastBitmap, unsigned zOutBlockLoopDepth,
    unsigned lastZOutBlockValidCount, unsigned defaultZOutBlockValidCount,
    unsigned partialSumReuseMap,
    unsigned narrowReadAlternateLimitAppliedLoopDepth,
    unsigned parameterReadAlternateLimitAppliedLoopDepth,
    unsigned mainOperationAlternateLimitAppliedLoopDepth, bool,
    unsigned hiccupDuration, unsigned hiccupDutyCycle, bool) {
  if (!linear || !nonLinear || !threadMulticastBitmap)
    return emitError() << "tensor control requires linear and nonlinear "
                          "controls and a thread bitmap";
  if (failed(LinearAttr::verify(
          emitError, linear.getOperation(), linear.getActivationType(),
          linear.getParameterType(), linear.getSingleFeature(),
          linear.getMacDisable(), linear.getPartialSumWrapAround(),
          linear.getZeroActivationPowerSave(), linear.getPartialSumRead(),
          linear.getPartialSumWritebackDisable(),
          linear.getLoadParameterZeroPoint(), linear.getActivationZeroPoint(),
          linear.getParameterZeroPoint(), linear.getZeroPointBaseAddress(),
          linear.getZeroPointStride(), linear.getZeroPointLimit(),
          linear.getBroadcastSecondOperand(),
          linear.getUseAlternateBaseAddress(), linear.getSparseParameters())) ||
      failed(NonLinearAttr::verify(
          emitError, nonLinear.getDisable(), nonLinear.getOperation(),
          nonLinear.getActivationPipelineScale(), nonLinear.getImmediateBias(),
          nonLinear.getHighClipValue(), nonLinear.getLowClipValue(),
          nonLinear.getOutputType(), nonLinear.getOffset(),
          nonLinear.getUseScales(), nonLinear.getApplyBias(),
          nonLinear.getSymmetricFunction(), nonLinear.getEvenOddFunction(),
          nonLinear.getPreprocess(), nonLinear.getBiasPredicate(),
          nonLinear.getScalePredicate(), nonLinear.getBitmask())))
    return failure();
  if (interpolation &&
      failed(InterpolationAttr::verify(
          emitError, interpolation.getSamplePoint(), interpolation.getSizeZ(),
          interpolation.getSizeXz(), interpolation.getMode())))
    return failure();
  if (threadMulticastBitmap.size() != 4)
    return emitError() << "tensor thread multicast bitmap must contain 4 bits";
  if (zOutBlockLoopDepth >= 8 ||
      narrowReadAlternateLimitAppliedLoopDepth >= 8 ||
      parameterReadAlternateLimitAppliedLoopDepth >= 8 ||
      mainOperationAlternateLimitAppliedLoopDepth >= 8)
    return emitError() << "tensor loop depths must be between 0 and 7";
  if (lastZOutBlockValidCount == 0 || lastZOutBlockValidCount > 32 ||
      defaultZOutBlockValidCount == 0 || defaultZOutBlockValidCount > 32)
    return emitError() << "tensor output block counts must be between 1 and 32";
  if (partialSumReuseMap > UINT8_MAX)
    return emitError() << "partial sum reuse map must fit in 8 bits";
  if (hiccupDuration >= 128 || hiccupDutyCycle >= 1024)
    return emitError()
           << "hiccup duration must fit in 7 bits and duty cycle in 10 bits";
  if (static_cast<bool>(interpolation) !=
      (linear.getOperation() == LinearOperation::Interpolation))
    return emitError()
           << "interpolation operation requires interpolation parameters";
  if (interpolation &&
      (nonLinear.getImmediateBias() != 0 || nonLinear.getOffset() != 0 ||
       nonLinear.getApplyBias() ||
       !nonLinear.getHighClipValue().getValue().bitcastToAPInt().isZero() ||
       !nonLinear.getLowClipValue().getValue().bitcastToAPInt().isZero()))
    return emitError()
           << "interpolation replaces nonlinear clipping and bias fields";
  return success();
}

LogicalResult TensorOp::verify() {
  if (getRegisterSourcedOperands().size() != 8)
    return emitOpError("register_sourced_operands must contain 8 bits");
  if (getSyncWatchers().size() > 6)
    return emitOpError("supports at most 6 synchronization watchers");

  for (auto [name, traversal] :
       {std::pair<StringRef, TraversalAttr>{"narrow_memory_read",
                                            getNarrowMemoryRead()},
        {"narrow_memory_write_from_non_linear",
         getNarrowMemoryWriteFromNonLinear()},
        {"wide_memory_read_for_parameters", getWideMemoryReadForParameters()},
        {"wide_memory_read_for_sums", getWideMemoryReadForSums()}}) {
    if (!traversal.getCounter().empty() && traversal.getCounter().size() != 8)
      return emitOpError() << name
                           << " requires zero or eight traversal counters";
    if (traversal.getSyncProducer().size() > 1)
      return emitOpError() << name << " supports one synchronization producer";
    if (traversal.getPrologue() || !traversal.getAdditionalPrologues().empty())
      return emitOpError() << name << " does not support prologues";
  }
  TensorControlAttr control = getControl();
  return TensorControlAttr::verify(
      [&] { return emitOpError(); }, control.getLinear(),
      control.getNonLinear(), control.getInterpolation(),
      control.getThreadMulticastBitmap(), control.getZOutBlockLoopDepth(),
      control.getLastZOutBlockValidCount(),
      control.getDefaultZOutBlockValidCount(), control.getPartialSumReuseMap(),
      control.getNarrowReadAlternateLimitAppliedLoopDepth(),
      control.getParameterReadAlternateLimitAppliedLoopDepth(),
      control.getMainOperationAlternateLimitAppliedLoopDepth(),
      control.getHiccupEnable(), control.getHiccupDuration(),
      control.getHiccupDutyCycle(), control.getUseGatheredNarrowMemoryRead());
}
