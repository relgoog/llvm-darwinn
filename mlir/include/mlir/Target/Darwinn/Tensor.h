#ifndef MLIR_TARGET_DARWINN_TENSOR_H
#define MLIR_TARGET_DARWINN_TENSOR_H

#include "mlir/Target/Darwinn/Encoding.h"
#include "mlir/Target/Darwinn/Traversal.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include <array>
#include <cstdint>
#include <optional>
#include <variant>

namespace mlir::darwinn {

enum class LinearOperation : uint8_t {
  MultiplyAccumulate,
  SquaredDifference,
  AbsoluteDifference,
  Maximum,
  Add,
  ParallelDotProduct,
  Interpolation,
  HighBandwidthMaximum,
  HighBandwidthArgMaximum,
  HighBandwidthMinimum,
  HighBandwidthArgMinimum,
  HighBandwidthAbsoluteMaximum,
  HighBandwidthAbsoluteMinimum,
  HighBandwidthMac,
  PartialSumAdd,
  SumOfSquares,
  ParallelAccumulate,
  ParallelBitwiseOr,
  ParallelBitwiseAnd,
  ParallelBitwiseXor,
  ParallelLogicalShiftLeft,
  ParallelArithmeticShiftRight,
  ParallelLogicalShiftRight
};

enum class OperandType : uint8_t {
  FixedPoint8,
  FixedPoint16,
  Bfloat,
  Half,
  Single
};

enum class MacDisable : uint8_t {
  None = 0,
  LastThreeFixed = 14,
  ThirdFourthFixed = 12,
  SecondFourthFixed = 10,
  FourthFixed = 8,
  SecondFloat = 4
};

enum class PartialSumRead : uint8_t {
  WideRead,
  Initialize,
  CumulativeSum,
  CumulativeSumInitialize
};

enum class SparseParameters : uint8_t {
  None,
  BlockOne,
  BlockTwo,
  BlockFour,
  TwoInFour,
  FourInEight
};

struct Linear {
  LinearOperation operation = LinearOperation::MultiplyAccumulate;
  OperandType activationType = OperandType::FixedPoint8;
  OperandType parameterType = OperandType::FixedPoint8;
  bool singleFeature = false;
  MacDisable macDisable = MacDisable::None;
  bool partialSumWrapAround = false;
  bool zeroActivationPowerSave = false;
  PartialSumRead partialSumRead = PartialSumRead::WideRead;
  bool partialSumWritebackDisable = false;
  bool loadParameterZeroPoint = false;
  uint16_t activationZeroPoint = 0;
  uint16_t parameterZeroPoint = 0;
  uint8_t zeroPointBaseAddress = 0;
  uint8_t zeroPointStride = 0;
  uint8_t zeroPointLimit = 0;
  bool broadcastSecondOperand = false;
  bool useAlternateBaseAddress = false;
  SparseParameters sparseParameters = SparseParameters::None;
};

enum class ActivationFunction : uint8_t {
  Relu,
  Table,
  SquareRoot,
  ReciprocalSquareRoot,
  Reciprocal
};

enum class OutputType : uint8_t {
  FixedPoint8,
  FixedPoint16,
  FixedPoint32,
  Bfloat,
  Half,
  Single
};

enum class Preprocess : uint8_t {
  None,
  ExtractExponent,
  LoadExponent,
  CopySignificand,
  IntegralPart,
  FractionalPart,
  AndMask,
  OrMask,
  PopulationCount,
  CountLeadingZeros,
  XorMask
};

enum class NluPredicate : uint8_t {
  None,
  GreaterThanZero,
  GreaterOrEqualZero,
  LessThanZero,
  LessOrEqualZero,
  EqualZero,
  NotEqualZero
};

struct NonLinear {
  bool disable = false;
  ActivationFunction operation = ActivationFunction::Relu;
  float activationPipelineScale = 0;
  int32_t immediateBias = 0;
  float highClipValue = 0;
  float lowClipValue = 0;
  OutputType outputType = OutputType::FixedPoint8;
  uint16_t offset = 0;
  bool useScales = false;
  bool applyBias = false;
  bool symmetricFunction = false;
  bool evenOddFunction = false;
  Preprocess preprocess = Preprocess::None;
  NluPredicate biasPredicate = NluPredicate::None;
  NluPredicate scalePredicate = NluPredicate::None;
  uint32_t bitmask = 0;
};

enum class InterpolationMode : uint8_t { Bilinear, NearestNeighbor };

struct ComputedSamplePoint {
  uint8_t loopDepthX = 0;
  uint8_t loopDepthY = 0;
  uint32_t firstX = 0;
  uint32_t firstY = 0;
  uint32_t strideX = 0;
  uint32_t strideY = 0;
};

struct NarrowMemorySamplePoint {};

struct Interpolation {
  std::variant<ComputedSamplePoint, NarrowMemorySamplePoint> samplePoint;
  uint32_t sizeZ = 0;
  uint16_t sizeXz = 0;
  InterpolationMode mode = InterpolationMode::Bilinear;
};

struct TensorControl {
  Linear linear;
  NonLinear nonLinear;
  std::optional<Interpolation> interpolation;
  std::array<bool, 4> threadMulticastBitmap{};
  uint8_t zOutBlockLoopDepth = 0;
  uint8_t lastZOutBlockValidCount = 1;
  uint8_t defaultZOutBlockValidCount = 1;
  uint8_t partialSumReuseMap = 0;
  uint8_t narrowReadAlternateLimitAppliedLoopDepth = 0;
  uint8_t parameterReadAlternateLimitAppliedLoopDepth = 0;
  uint8_t mainOperationAlternateLimitAppliedLoopDepth = 0;
  bool hiccupEnable = false;
  uint8_t hiccupDuration = 0;
  uint16_t hiccupDutyCycle = 0;
  bool useGatheredNarrowMemoryRead = false;
};

struct TensorOp {
  MainOperation mainOperation;
  Traversal narrowMemoryRead;
  Traversal narrowMemoryWriteFromNonLinear;
  Traversal wideMemoryReadForParameters;
  Traversal wideMemoryReadForSums;
  llvm::SmallVector<SyncWatcher, 6> syncWatchers;
  TensorControl control;
};

class TensorEncoder {
public:
  void reset() { previousBody.reset(); }

  llvm::Expected<InstructionBytes>
  encode(const TileHeader &header, OverwriteInfo overwrite,
         llvm::ArrayRef<bool> registerSourcedOperandBitmap,
         const TensorOp &operation);

private:
  std::optional<BitWriter> previousBody;
};

}

#endif
