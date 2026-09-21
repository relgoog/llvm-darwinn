#include "mlir/Target/Darwinn/TensorAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Target/Darwinn/EncodingAttributes.h"
#include "mlir/Target/Darwinn/TraversalAttributes.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::darwinn;

llvm::Expected<ComputePacket>
mlir::darwinn::convertTensor(isa::TensorOp operation) {
  if (!operation)
    return llvm::createStringError("missing tensor instruction");
  std::string diagnostic;
  {
    ScopedDiagnosticHandler handler(operation.getContext(),
                                    [&](Diagnostic &value) {
                                      diagnostic = value.str();
                                      return success();
                                    });
    if (failed(operation.verifyInvariants()))
      return llvm::createStringError(diagnostic);
  }
  ComputePacket packet;
  auto header = convertTileHeader(operation.getHeader());
  if (!header)
    return header.takeError();
  packet.header = *header;
  auto overwrite = convertOverwrite(operation.getOverwrite());
  if (!overwrite)
    return overwrite.takeError();
  packet.overwrite = *overwrite;

  llvm::copy(operation.getRegisterSourcedOperands(),
             packet.registerSourcedOperands.begin());

  darwinn::TensorOp tensor;
  auto mainOperation = convertMainOperation(operation.getMainOperation());
  if (!mainOperation)
    return mainOperation.takeError();
  tensor.mainOperation = *mainOperation;

  for (auto [attribute, destination] :
       {std::pair<isa::TraversalAttr, Traversal *>{
            operation.getNarrowMemoryRead(), &tensor.narrowMemoryRead},
        {operation.getNarrowMemoryWriteFromNonLinear(),
         &tensor.narrowMemoryWriteFromNonLinear},
        {operation.getWideMemoryReadForParameters(),
         &tensor.wideMemoryReadForParameters},
        {operation.getWideMemoryReadForSums(),
         &tensor.wideMemoryReadForSums}}) {
    auto traversal = convertTraversal(attribute);
    if (!traversal)
      return traversal.takeError();
    *destination = std::move(*traversal);
  }
  for (Attribute attribute : operation.getSyncWatchers()) {
    auto watcher = convertSyncWatcher(cast<isa::SyncWatcherAttr>(attribute));
    if (!watcher)
      return watcher.takeError();
    tensor.syncWatchers.push_back(*watcher);
  }

  isa::TensorControlAttr control = operation.getControl();
  isa::LinearAttr linear = control.getLinear();
  tensor.control.linear = {
      static_cast<LinearOperation>(linear.getOperation()),
      static_cast<OperandType>(linear.getActivationType()),
      static_cast<OperandType>(linear.getParameterType()),
      linear.getSingleFeature(),
      static_cast<MacDisable>(linear.getMacDisable()),
      linear.getPartialSumWrapAround(),
      linear.getZeroActivationPowerSave(),
      static_cast<PartialSumRead>(linear.getPartialSumRead()),
      linear.getPartialSumWritebackDisable(),
      linear.getLoadParameterZeroPoint(),
      static_cast<uint16_t>(linear.getActivationZeroPoint()),
      static_cast<uint16_t>(linear.getParameterZeroPoint()),
      static_cast<uint8_t>(linear.getZeroPointBaseAddress()),
      static_cast<uint8_t>(linear.getZeroPointStride()),
      static_cast<uint8_t>(linear.getZeroPointLimit()),
      linear.getBroadcastSecondOperand(),
      linear.getUseAlternateBaseAddress(),
      static_cast<SparseParameters>(linear.getSparseParameters())};

  isa::NonLinearAttr nonLinear = control.getNonLinear();
  tensor.control.nonLinear = {
      nonLinear.getDisable(),
      static_cast<ActivationFunction>(nonLinear.getOperation()),
      nonLinear.getActivationPipelineScale().getValue().convertToFloat(),
      nonLinear.getImmediateBias(),
      nonLinear.getHighClipValue().getValue().convertToFloat(),
      nonLinear.getLowClipValue().getValue().convertToFloat(),
      static_cast<OutputType>(nonLinear.getOutputType()),
      static_cast<uint16_t>(nonLinear.getOffset()),
      nonLinear.getUseScales(),
      nonLinear.getApplyBias(),
      nonLinear.getSymmetricFunction(),
      nonLinear.getEvenOddFunction(),
      static_cast<Preprocess>(nonLinear.getPreprocess()),
      static_cast<NluPredicate>(nonLinear.getBiasPredicate()),
      static_cast<NluPredicate>(nonLinear.getScalePredicate()),
      nonLinear.getBitmask()};

  if (isa::InterpolationAttr interpolation = control.getInterpolation()) {
    Interpolation converted;
    if (isa::ComputedSamplePointAttr samplePoint =
            interpolation.getSamplePoint())
      converted.samplePoint =
          ComputedSamplePoint{static_cast<uint8_t>(samplePoint.getLoopDepthX()),
                              static_cast<uint8_t>(samplePoint.getLoopDepthY()),
                              samplePoint.getFirstX(),
                              samplePoint.getFirstY(),
                              samplePoint.getStrideX(),
                              samplePoint.getStrideY()};
    else
      converted.samplePoint = NarrowMemorySamplePoint{};
    converted.sizeZ = interpolation.getSizeZ();
    converted.sizeXz = static_cast<uint16_t>(interpolation.getSizeXz());
    converted.mode = static_cast<InterpolationMode>(interpolation.getMode());
    tensor.control.interpolation = converted;
  }

  llvm::copy(control.getThreadMulticastBitmap().asArrayRef(),
             tensor.control.threadMulticastBitmap.begin());
  tensor.control.zOutBlockLoopDepth = control.getZOutBlockLoopDepth();
  tensor.control.lastZOutBlockValidCount = control.getLastZOutBlockValidCount();
  tensor.control.defaultZOutBlockValidCount =
      control.getDefaultZOutBlockValidCount();
  tensor.control.partialSumReuseMap = control.getPartialSumReuseMap();
  tensor.control.narrowReadAlternateLimitAppliedLoopDepth =
      control.getNarrowReadAlternateLimitAppliedLoopDepth();
  tensor.control.parameterReadAlternateLimitAppliedLoopDepth =
      control.getParameterReadAlternateLimitAppliedLoopDepth();
  tensor.control.mainOperationAlternateLimitAppliedLoopDepth =
      control.getMainOperationAlternateLimitAppliedLoopDepth();
  tensor.control.hiccupEnable = control.getHiccupEnable();
  tensor.control.hiccupDuration = control.getHiccupDuration();
  tensor.control.hiccupDutyCycle = control.getHiccupDutyCycle();
  tensor.control.useGatheredNarrowMemoryRead =
      control.getUseGatheredNarrowMemoryRead();
  packet.instruction = std::move(tensor);
  return packet;
}
