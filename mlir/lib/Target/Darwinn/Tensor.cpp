#include "mlir/Target/Darwinn/Tensor.h"
#include "mlir/Target/Darwinn/Compression.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/bit.h"
#include <utility>

using namespace mlir::darwinn;

namespace {

struct LinearOpcode {
  uint8_t instruction;
  uint8_t suboperation;
};

llvm::Expected<LinearOpcode> getLinearOpcode(LinearOperation operation) {
  switch (operation) {
  case LinearOperation::MultiplyAccumulate:
    return LinearOpcode{1, 0};
  case LinearOperation::SquaredDifference:
    return LinearOpcode{1, 1};
  case LinearOperation::AbsoluteDifference:
    return LinearOpcode{1, 2};
  case LinearOperation::Maximum:
    return LinearOpcode{2, 0};
  case LinearOperation::Add:
    return LinearOpcode{3, 0};
  case LinearOperation::ParallelDotProduct:
    return LinearOpcode{4, 0};
  case LinearOperation::Interpolation:
    return LinearOpcode{5, 0};
  case LinearOperation::HighBandwidthMaximum:
    return LinearOpcode{7, 0};
  case LinearOperation::HighBandwidthArgMaximum:
    return LinearOpcode{7, 1};
  case LinearOperation::HighBandwidthMinimum:
    return LinearOpcode{7, 2};
  case LinearOperation::HighBandwidthArgMinimum:
    return LinearOpcode{7, 3};
  case LinearOperation::HighBandwidthAbsoluteMaximum:
    return LinearOpcode{7, 4};
  case LinearOperation::HighBandwidthAbsoluteMinimum:
    return LinearOpcode{7, 5};
  case LinearOperation::HighBandwidthMac:
    return LinearOpcode{8, 0};
  case LinearOperation::PartialSumAdd:
    return LinearOpcode{9, 0};
  case LinearOperation::SumOfSquares:
    return LinearOpcode{10, 0};
  case LinearOperation::ParallelAccumulate:
    return LinearOpcode{12, 0};
  case LinearOperation::ParallelBitwiseOr:
    return LinearOpcode{12, 1};
  case LinearOperation::ParallelBitwiseAnd:
    return LinearOpcode{12, 2};
  case LinearOperation::ParallelBitwiseXor:
    return LinearOpcode{12, 3};
  case LinearOperation::ParallelLogicalShiftLeft:
    return LinearOpcode{12, 4};
  case LinearOperation::ParallelArithmeticShiftRight:
    return LinearOpcode{12, 5};
  case LinearOperation::ParallelLogicalShiftRight:
    return LinearOpcode{12, 6};
  }
  return llvm::createStringError("Invalid tensor linear operation");
}

llvm::Error writeFields(BitWriter &writer,
                        llvm::ArrayRef<std::pair<uint64_t, unsigned>> fields) {
  for (const auto &field : fields)
    if (llvm::Error error = writer.write(field.first, field.second))
      return error;
  return llvm::Error::success();
}

llvm::Error writeInterpolation(BitWriter &writer,
                               const Interpolation &interpolation) {
  if (const auto *computed =
          std::get_if<ComputedSamplePoint>(&interpolation.samplePoint)) {
    if (llvm::Error error = writeFields(writer, {{0, 1},
                                                 {computed->loopDepthX, 3},
                                                 {computed->firstX, 17},
                                                 {computed->strideX, 17},
                                                 {computed->loopDepthY, 3},
                                                 {computed->firstY, 17},
                                                 {computed->strideY, 17}}))
      return error;
  } else if (llvm::Error error =
                 writeFields(writer, {{1, 1}, {0, 37}, {0, 37}})) {
    return error;
  }
  return writeFields(writer, {{interpolation.sizeZ, 32},
                              {interpolation.sizeXz, 16},
                              {static_cast<uint8_t>(interpolation.mode), 1}});
}

llvm::Error writeControl(BitWriter &writer, const TensorControl &control,
                         uint8_t suboperation) {
  const Linear &linear = control.linear;
  const NonLinear &nonLinear = control.nonLinear;
  if (control.interpolation.has_value() !=
      (linear.operation == LinearOperation::Interpolation))
    return llvm::createStringError(
        "Tensor interpolation operation requires interpolation parameters");
  if (control.interpolation &&
      (nonLinear.immediateBias != 0 ||
       llvm::bit_cast<uint32_t>(nonLinear.highClipValue) != 0 ||
       llvm::bit_cast<uint32_t>(nonLinear.lowClipValue) != 0 ||
       nonLinear.offset != 0 || nonLinear.applyBias))
    return llvm::createStringError(
        "Tensor interpolation replaces the nonlinear clipping and bias fields");
  if (control.narrowReadAlternateLimitAppliedLoopDepth > 7 ||
      control.parameterReadAlternateLimitAppliedLoopDepth > 7 ||
      control.mainOperationAlternateLimitAppliedLoopDepth > 7)
    return llvm::createStringError(
        "Tensor alternate limits require an available loop");
  if (control.lastZOutBlockValidCount == 0 ||
      control.defaultZOutBlockValidCount == 0)
    return llvm::createStringError(
        "Tensor output block must contain at least one cell");
  if (linear.activationType > OperandType::Single ||
      linear.parameterType > OperandType::Single ||
      linear.partialSumRead > PartialSumRead::CumulativeSumInitialize ||
      linear.sparseParameters > SparseParameters::FourInEight ||
      nonLinear.operation > ActivationFunction::Reciprocal ||
      nonLinear.outputType > OutputType::Single ||
      nonLinear.preprocess > Preprocess::XorMask ||
      nonLinear.biasPredicate > NluPredicate::NotEqualZero ||
      nonLinear.scalePredicate > NluPredicate::NotEqualZero ||
      (control.interpolation &&
       control.interpolation->mode > InterpolationMode::NearestNeighbor))
    return llvm::createStringError("Invalid tensor control enumeration");
  if (linear.macDisable != MacDisable::None &&
      linear.macDisable != MacDisable::LastThreeFixed &&
      linear.macDisable != MacDisable::ThirdFourthFixed &&
      linear.macDisable != MacDisable::SecondFourthFixed &&
      linear.macDisable != MacDisable::FourthFixed &&
      linear.macDisable != MacDisable::SecondFloat)
    return llvm::createStringError("Invalid tensor MAC disable mode");

  if (llvm::Error error = writeFields(
          writer, {{control.hiccupEnable, 1},
                   {control.hiccupDuration, 7},
                   {control.hiccupDutyCycle, 10},
                   {linear.partialSumWrapAround, 1},
                   {linear.zeroActivationPowerSave, 1},
                   {static_cast<uint8_t>(linear.partialSumRead), 2},
                   {linear.partialSumWritebackDisable, 1},
                   {static_cast<uint8_t>(linear.sparseParameters), 3},
                   {control.useGatheredNarrowMemoryRead, 1}}))
    return error;
  if (llvm::Error error = writer.writeBitmap(control.threadMulticastBitmap))
    return error;
  if (llvm::Error error = writeFields(
          writer,
          {{suboperation, 3},
           {static_cast<uint8_t>(linear.activationType), 3},
           {static_cast<uint8_t>(linear.parameterType), 3},
           {linear.singleFeature, 1},
           {static_cast<uint8_t>(linear.macDisable), 4},
           {linear.loadParameterZeroPoint, 1},
           {linear.parameterZeroPoint, 16},
           {linear.activationZeroPoint, 16},
           {linear.zeroPointLimit, 6},
           {linear.zeroPointStride, 6},
           {linear.zeroPointBaseAddress, 6},
           {linear.useAlternateBaseAddress, 1},
           {control.zOutBlockLoopDepth, 3},
           {control.partialSumReuseMap, 8},
           {static_cast<uint8_t>(control.lastZOutBlockValidCount - 1), 5},
           {static_cast<uint8_t>(control.defaultZOutBlockValidCount - 1), 5},
           {linear.broadcastSecondOperand, 1},
           {control.narrowReadAlternateLimitAppliedLoopDepth != 0, 1},
           {control.parameterReadAlternateLimitAppliedLoopDepth != 0, 1},
           {control.mainOperationAlternateLimitAppliedLoopDepth != 0, 1},
           {nonLinear.bitmask, 32},
           {static_cast<uint8_t>(nonLinear.scalePredicate), 3},
           {static_cast<uint8_t>(nonLinear.biasPredicate), 3},
           {static_cast<uint8_t>(nonLinear.preprocess), 4},
           {nonLinear.disable, 1},
           {static_cast<uint8_t>(nonLinear.operation), 3},
           {static_cast<uint8_t>(nonLinear.outputType), 3},
           {llvm::bit_cast<uint32_t>(nonLinear.activationPipelineScale), 32}}))
    return error;

  if (control.interpolation) {
    if (llvm::Error error = writeInterpolation(writer, *control.interpolation))
      return error;
  } else if (llvm::Error error = writeFields(
                 writer,
                 {{0, 1},
                  {llvm::bit_cast<uint32_t>(nonLinear.highClipValue), 32},
                  {0, 5},
                  {llvm::bit_cast<uint32_t>(nonLinear.lowClipValue), 32},
                  {0, 5},
                  {static_cast<uint32_t>(nonLinear.immediateBias), 32},
                  {nonLinear.offset, 16},
                  {nonLinear.applyBias, 1}})) {
    return error;
  }
  return writeFields(writer, {{nonLinear.useScales, 1},
                              {nonLinear.symmetricFunction, 1},
                              {nonLinear.evenOddFunction, 1}});
}

llvm::Expected<BitWriter> encodeBody(const TensorOp &operation,
                                     uint8_t suboperation) {
  if (operation.syncWatchers.size() > 6)
    return llvm::createStringError(
        "Tensor operation supports six synchronization watchers");

  BitWriter writer;
  if (llvm::Error error = writeMainOperation(writer, operation.mainOperation))
    return error;

  const Traversal *traversals[] = {
      &operation.narrowMemoryRead, &operation.narrowMemoryWriteFromNonLinear,
      &operation.wideMemoryReadForParameters, &operation.wideMemoryReadForSums};
  const uint8_t widths[] = {20, 20, 20, 12};
  const std::optional<uint8_t> access[] = {4, 20, std::nullopt, std::nullopt};
  const uint8_t alternate[] = {
      operation.control.narrowReadAlternateLimitAppliedLoopDepth, 0,
      operation.control.parameterReadAlternateLimitAppliedLoopDepth, 0};

  for (size_t index = 0; index != 4; ++index) {
    const Traversal &traversal = *traversals[index];
    if (traversal.syncProducer.size() > 1)
      return llvm::createStringError(
          "Tensor traversal supports one synchronization producer");
    uint8_t width = widths[index];
    TraversalEncoding encoding{
        width, width,        width,           8, access[index], true, 0,
        0,     std::nullopt, alternate[index]};
    if (llvm::Error error = writeTraversal(writer, traversal, encoding))
      return error;
  }
  for (const Traversal *traversal :
       {traversals[1], traversals[2], traversals[3], traversals[0]}) {
    const SyncProducer *producer = traversal->syncProducer.empty()
                                       ? nullptr
                                       : &traversal->syncProducer.front();
    if (llvm::Error error = writeSyncProducer(writer, producer, 8))
      return error;
  }
  for (size_t index = 0; index != 6; ++index) {
    const SyncWatcher *watcher = index < operation.syncWatchers.size()
                                     ? &operation.syncWatchers[index]
                                     : nullptr;
    if (llvm::Error error = writeSyncWatcher(writer, watcher, 8))
      return error;
  }
  if (llvm::Error error = writeControl(writer, operation.control, suboperation))
    return error;
  return writer;
}

}

llvm::Expected<InstructionBytes>
TensorEncoder::encode(const TileHeader &header, OverwriteInfo overwrite,
                      llvm::ArrayRef<bool> registerSourcedOperandBitmap,
                      const TensorOp &operation) {
  llvm::Expected<LinearOpcode> opcode =
      getLinearOpcode(operation.control.linear.operation);
  if (!opcode)
    return opcode.takeError();
  llvm::Expected<BitWriter> body = encodeBody(operation, opcode->suboperation);
  if (!body)
    return body.takeError();

  BitWriter writer;
  if (llvm::Error error =
          writeTileComputeHeader(writer, header, opcode->instruction, overwrite,
                                 registerSourcedOperandBitmap))
    return error;
  llvm::Expected<CompressionLayout> layout =
      CompressionLayout::create(76, 2447, 20);
  if (!layout)
    return layout.takeError();
  llvm::Expected<bool> compressed =
      layout->writeBody(writer, previousBody ? &*previousBody : nullptr, *body);
  if (!compressed)
    return compressed.takeError();

  InstructionBytes bytes = writer.finish();
  previousBody = std::move(*body);
  return bytes;
}
