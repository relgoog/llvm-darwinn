#ifndef MLIR_DIALECT_DARWINN_IR_INSTRUCTIONTENSOR_TD
#define MLIR_DIALECT_DARWINN_IR_INSTRUCTIONTENSOR_TD

include "mlir/Dialect/Darwinn/IR/InstructionDialect.td"
include "mlir/Dialect/Darwinn/IR/InstructionTraversal.td"

def DarwinnIsa_LinearOperation : DarwinnIsa_Enum<"LinearOperation", [
    I32EnumAttrCase<"MultiplyAccumulate", 0, "multiply_accumulate">,
    I32EnumAttrCase<"SquaredDifference", 1, "squared_difference">,
    I32EnumAttrCase<"AbsoluteDifference", 2, "absolute_difference">,
    I32EnumAttrCase<"Maximum", 3, "maximum">,
    I32EnumAttrCase<"Add", 4, "add">,
    I32EnumAttrCase<"ParallelDotProduct", 5, "parallel_dot_product">,
    I32EnumAttrCase<"Interpolation", 6, "interpolation">,
    I32EnumAttrCase<"HighBandwidthMaximum", 7, "high_bandwidth_maximum">,
    I32EnumAttrCase<"HighBandwidthArgMaximum", 8, "high_bandwidth_arg_maximum">,
    I32EnumAttrCase<"HighBandwidthMinimum", 9, "high_bandwidth_minimum">,
    I32EnumAttrCase<"HighBandwidthArgMinimum", 10, "high_bandwidth_arg_minimum">,
    I32EnumAttrCase<"HighBandwidthAbsoluteMaximum", 11, "high_bandwidth_absolute_maximum">,
    I32EnumAttrCase<"HighBandwidthAbsoluteMinimum", 12, "high_bandwidth_absolute_minimum">,
    I32EnumAttrCase<"HighBandwidthMac", 13, "high_bandwidth_mac">,
    I32EnumAttrCase<"PartialSumAdd", 14, "partial_sum_add">,
    I32EnumAttrCase<"SumOfSquares", 15, "sum_of_squares">,
    I32EnumAttrCase<"ParallelAccumulate", 16, "parallel_accumulate">,
    I32EnumAttrCase<"ParallelBitwiseOr", 17, "parallel_bitwise_or">,
    I32EnumAttrCase<"ParallelBitwiseAnd", 18, "parallel_bitwise_and">,
    I32EnumAttrCase<"ParallelBitwiseXor", 19, "parallel_bitwise_xor">,
    I32EnumAttrCase<"ParallelLogicalShiftLeft", 20, "parallel_logical_shift_left">,
    I32EnumAttrCase<"ParallelArithmeticShiftRight", 21, "parallel_arithmetic_shift_right">,
    I32EnumAttrCase<"ParallelLogicalShiftRight", 22, "parallel_logical_shift_right">]>;

def DarwinnIsa_OperandType : DarwinnIsa_Enum<"OperandType", [
    I32EnumAttrCase<"FixedPoint8", 0, "fixed_point8">,
    I32EnumAttrCase<"FixedPoint16", 1, "fixed_point16">,
    I32EnumAttrCase<"Bfloat", 2, "bfloat">,
    I32EnumAttrCase<"Half", 3, "half">,
    I32EnumAttrCase<"Single", 4, "single">]>;

def DarwinnIsa_MacDisable : DarwinnIsa_Enum<"MacDisable", [
    I32EnumAttrCase<"None", 0, "none">,
    I32EnumAttrCase<"LastThreeFixed", 14, "last_three_fixed">,
    I32EnumAttrCase<"ThirdFourthFixed", 12, "third_fourth_fixed">,
    I32EnumAttrCase<"SecondFourthFixed", 10, "second_fourth_fixed">,
    I32EnumAttrCase<"FourthFixed", 8, "fourth_fixed">,
    I32EnumAttrCase<"SecondFloat", 4, "second_float">]>;

def DarwinnIsa_PartialSumRead : DarwinnIsa_Enum<"PartialSumRead", [
    I32EnumAttrCase<"WideRead", 0, "wide_read">,
    I32EnumAttrCase<"Initialize", 1, "initialize">,
    I32EnumAttrCase<"CumulativeSum", 2, "cumulative_sum">,
    I32EnumAttrCase<"CumulativeSumInitialize", 3, "cumulative_sum_initialize">]>;

def DarwinnIsa_SparseParameters : DarwinnIsa_Enum<"SparseParameters", [
    I32EnumAttrCase<"None", 0, "none">,
    I32EnumAttrCase<"BlockOne", 1, "block_one">,
    I32EnumAttrCase<"BlockTwo", 2, "block_two">,
    I32EnumAttrCase<"BlockFour", 3, "block_four">,
    I32EnumAttrCase<"TwoInFour", 4, "two_in_four">,
    I32EnumAttrCase<"FourInEight", 5, "four_in_eight">]>;

def DarwinnIsa_ActivationFunction : DarwinnIsa_Enum<"ActivationFunction", [
    I32EnumAttrCase<"Relu", 0, "relu">,
    I32EnumAttrCase<"Table", 1, "table">,
    I32EnumAttrCase<"SquareRoot", 2, "square_root">,
    I32EnumAttrCase<"ReciprocalSquareRoot", 3, "reciprocal_square_root">,
    I32EnumAttrCase<"Reciprocal", 4, "reciprocal">]>;

def DarwinnIsa_OutputType : DarwinnIsa_Enum<"OutputType", [
    I32EnumAttrCase<"FixedPoint8", 0, "fixed_point8">,
    I32EnumAttrCase<"FixedPoint16", 1, "fixed_point16">,
    I32EnumAttrCase<"FixedPoint32", 2, "fixed_point32">,
    I32EnumAttrCase<"Bfloat", 3, "bfloat">,
    I32EnumAttrCase<"Half", 4, "half">,
    I32EnumAttrCase<"Single", 5, "single">]>;

def DarwinnIsa_Preprocess : DarwinnIsa_Enum<"Preprocess", [
    I32EnumAttrCase<"None", 0, "none">,
    I32EnumAttrCase<"ExtractExponent", 1, "extract_exponent">,
    I32EnumAttrCase<"LoadExponent", 2, "load_exponent">,
    I32EnumAttrCase<"CopySignificand", 3, "copy_significand">,
    I32EnumAttrCase<"IntegralPart", 4, "integral_part">,
    I32EnumAttrCase<"FractionalPart", 5, "fractional_part">,
    I32EnumAttrCase<"AndMask", 6, "and_mask">,
    I32EnumAttrCase<"OrMask", 7, "or_mask">,
    I32EnumAttrCase<"PopulationCount", 8, "population_count">,
    I32EnumAttrCase<"CountLeadingZeros", 9, "count_leading_zeros">,
    I32EnumAttrCase<"XorMask", 10, "xor_mask">]>;

def DarwinnIsa_NluPredicate : DarwinnIsa_Enum<"NluPredicate", [
    I32EnumAttrCase<"None", 0, "none">,
    I32EnumAttrCase<"GreaterThanZero", 1, "greater_than_zero">,
    I32EnumAttrCase<"GreaterOrEqualZero", 2, "greater_or_equal_zero">,
    I32EnumAttrCase<"LessThanZero", 3, "less_than_zero">,
    I32EnumAttrCase<"LessOrEqualZero", 4, "less_or_equal_zero">,
    I32EnumAttrCase<"EqualZero", 5, "equal_zero">,
    I32EnumAttrCase<"NotEqualZero", 6, "not_equal_zero">]>;

def DarwinnIsa_InterpolationMode : DarwinnIsa_Enum<"InterpolationMode", [
    I32EnumAttrCase<"Bilinear", 0, "bilinear">,
    I32EnumAttrCase<"NearestNeighbor", 1, "nearest_neighbor">]>;

def DarwinnIsa_LinearAttr : DarwinnIsa_Attr<"Linear", "linear"> {
  let parameters = (ins
      EnumParameter<DarwinnIsa_LinearOperation>:$operation,
      EnumParameter<DarwinnIsa_OperandType>:$activationType,
      EnumParameter<DarwinnIsa_OperandType>:$parameterType,
      "bool":$singleFeature,
      EnumParameter<DarwinnIsa_MacDisable>:$macDisable,
      "bool":$partialSumWrapAround,
      "bool":$zeroActivationPowerSave,
      EnumParameter<DarwinnIsa_PartialSumRead>:$partialSumRead,
      "bool":$partialSumWritebackDisable,
      "bool":$loadParameterZeroPoint,
      "unsigned":$activationZeroPoint,
      "unsigned":$parameterZeroPoint,
      "unsigned":$zeroPointBaseAddress,
      "unsigned":$zeroPointStride,
      "unsigned":$zeroPointLimit,
      "bool":$broadcastSecondOperand,
      "bool":$useAlternateBaseAddress,
      EnumParameter<DarwinnIsa_SparseParameters>:$sparseParameters);
  let genVerifyDecl = 1;
}

def DarwinnIsa_NonLinearAttr : DarwinnIsa_Attr<"NonLinear", "non_linear"> {
  let parameters = (ins
      "bool":$disable,
      EnumParameter<DarwinnIsa_ActivationFunction>:$operation,
      "::mlir::FloatAttr":$activationPipelineScale,
      "int32_t":$immediateBias,
      "::mlir::FloatAttr":$highClipValue,
      "::mlir::FloatAttr":$lowClipValue,
      EnumParameter<DarwinnIsa_OutputType>:$outputType,
      "unsigned":$offset,
      "bool":$useScales,
      "bool":$applyBias,
      "bool":$symmetricFunction,
      "bool":$evenOddFunction,
      EnumParameter<DarwinnIsa_Preprocess>:$preprocess,
      EnumParameter<DarwinnIsa_NluPredicate>:$biasPredicate,
      EnumParameter<DarwinnIsa_NluPredicate>:$scalePredicate,
      "uint32_t":$bitmask);
  let genVerifyDecl = 1;
}

def DarwinnIsa_ComputedSamplePointAttr
    : DarwinnIsa_Attr<"ComputedSamplePoint", "computed_sample_point"> {
  let parameters = (ins
      "unsigned":$loopDepthX, "unsigned":$loopDepthY,
      "uint32_t":$firstX, "uint32_t":$firstY,
      "uint32_t":$strideX, "uint32_t":$strideY);
  let genVerifyDecl = 1;
}

def DarwinnIsa_InterpolationAttr
    : DarwinnIsa_Attr<"Interpolation", "interpolation"> {
  let parameters = (ins
      OptionalParameter<"ComputedSamplePointAttr">:$samplePoint,
      "uint32_t":$sizeZ, "unsigned":$sizeXz,
      EnumParameter<DarwinnIsa_InterpolationMode>:$mode);
  let genVerifyDecl = 1;
}

def DarwinnIsa_TensorControlAttr
    : DarwinnIsa_Attr<"TensorControl", "tensor_control"> {
  let parameters = (ins
      "LinearAttr":$linear, "NonLinearAttr":$nonLinear,
      OptionalParameter<"InterpolationAttr">:$interpolation,
      "::mlir::DenseBoolArrayAttr":$threadMulticastBitmap,
      "unsigned":$zOutBlockLoopDepth,
      "unsigned":$lastZOutBlockValidCount,
      "unsigned":$defaultZOutBlockValidCount,
      "unsigned":$partialSumReuseMap,
      "unsigned":$narrowReadAlternateLimitAppliedLoopDepth,
      "unsigned":$parameterReadAlternateLimitAppliedLoopDepth,
      "unsigned":$mainOperationAlternateLimitAppliedLoopDepth,
      "bool":$hiccupEnable, "unsigned":$hiccupDuration,
      "unsigned":$hiccupDutyCycle, "bool":$useGatheredNarrowMemoryRead);
  let genVerifyDecl = 1;
}

def DarwinnIsa_TensorSyncWatchers : TypedArrayAttrBase<
    DarwinnIsa_SyncWatcherAttr, "tensor synchronization watchers">;

def DarwinnIsa_TensorOp : DarwinnIsa_Instruction<"tensor"> {
  let arguments = (ins
      DarwinnIsa_TileHeaderAttr:$header,
      DarwinnIsa_OverwriteInfoAttr:$overwrite,
      DenseBoolArrayAttr:$register_sourced_operands,
      DarwinnIsa_MainOperationAttr:$main_operation,
      DarwinnIsa_TraversalAttr:$narrow_memory_read,
      DarwinnIsa_TraversalAttr:$narrow_memory_write_from_non_linear,
      DarwinnIsa_TraversalAttr:$wide_memory_read_for_parameters,
      DarwinnIsa_TraversalAttr:$wide_memory_read_for_sums,
      DarwinnIsa_TensorSyncWatchers:$sync_watchers,
      DarwinnIsa_TensorControlAttr:$control);
}

#endif
