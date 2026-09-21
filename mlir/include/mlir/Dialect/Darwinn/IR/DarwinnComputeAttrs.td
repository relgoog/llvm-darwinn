#ifndef MLIR_DIALECT_DARWINN_IR_DARWINNCOMPUTEATTRS_TD
#define MLIR_DIALECT_DARWINN_IR_DARWINNCOMPUTEATTRS_TD

include "mlir/IR/EnumAttr.td"

def Darwinn_InnerOperationKind : I32EnumAttr<"InnerOperationKind", "InnerOperationKind", [
    I32EnumAttrCase<"Vmc", 0, "VMC">,
    I32EnumAttrCase<"Svc", 1, "SVC">,
    I32EnumAttrCase<"Stencil", 2, "STENCIL">,
    I32EnumAttrCase<"Elementwise", 3, "ELEMENTWISE">,
    I32EnumAttrCase<"Unary", 4, "UNARY">,
    I32EnumAttrCase<"WinToOutChannel", 5, "WIN_TO_OUT_CHANNEL">,
    I32EnumAttrCase<"InplaceElementwise", 6, "INPLACE_ELEMENTWISE">]> {
  let cppNamespace = "::mlir::darwinn";
  let genSpecializedAttr = 0;
}

def Darwinn_LinearFunctionKind : I32EnumAttr<"LinearFunctionKind", "LinearFunctionKind", [
    I32EnumAttrCase<"Mac", 0, "MAC">,
    I32EnumAttrCase<"MacSubChannel", 24, "MAC_SUB_CHANNEL">,
    I32EnumAttrCase<"Add", 1, "ADD">,
    I32EnumAttrCase<"Sub", 2, "SUB">,
    I32EnumAttrCase<"AxPlusBy", 3, "AX_PLUS_BY">,
    I32EnumAttrCase<"Sad", 4, "SAD">,
    I32EnumAttrCase<"Ssd", 5, "SSD">,
    I32EnumAttrCase<"CountGreaterThan", 10, "COUNT_GREATER_THAN">,
    I32EnumAttrCase<"CountGreaterThanEqual", 11, "COUNT_GREATER_THAN_EQUAL">,
    I32EnumAttrCase<"CountLessThan", 12, "COUNT_LESS_THAN">,
    I32EnumAttrCase<"CountLessThanEqual", 13, "COUNT_LESS_THAN_EQUAL">,
    I32EnumAttrCase<"CountEqual", 14, "COUNT_EQUAL">,
    I32EnumAttrCase<"CountNotEqual", 15, "COUNT_NOT_EQUAL">,
    I32EnumAttrCase<"CumulativeSum", 16, "CUMULATIVE_SUM">,
    I32EnumAttrCase<"Max", 6, "MAX">,
    I32EnumAttrCase<"Min", 7, "MIN">,
    I32EnumAttrCase<"Absmax", 25, "ABSMAX">,
    I32EnumAttrCase<"Absmin", 26, "ABSMIN">,
    I32EnumAttrCase<"Argmax", 8, "ARGMAX">,
    I32EnumAttrCase<"Argmin", 9, "ARGMIN">,
    I32EnumAttrCase<"BitwiseAnd", 17, "BITWISE_AND">,
    I32EnumAttrCase<"BitwiseOr", 18, "BITWISE_OR">,
    I32EnumAttrCase<"BitwiseXor", 19, "BITWISE_XOR">,
    I32EnumAttrCase<"LogicalShiftLeft", 20, "LOGICAL_SHIFT_LEFT">,
    I32EnumAttrCase<"LogicalShiftRight", 21, "LOGICAL_SHIFT_RIGHT">,
    I32EnumAttrCase<"ArithShiftRight", 22, "ARITH_SHIFT_RIGHT">,
    I32EnumAttrCase<"MmaRequantInterleaving", 27, "MMA_REQUANT_INTERLEAVING">,
    I32EnumAttrCase<"MmaRequant", 28, "MMA_REQUANT">,
    I32EnumAttrCase<"DtcNormalize", 29, "DTC_NORMALIZE">,
    I32EnumAttrCase<"Noop", 23, "NOOP">]> {
  let cppNamespace = "::mlir::darwinn";
  let genSpecializedAttr = 0;
}

def Darwinn_NluFunctionKind : I32EnumAttr<"NluFunctionKind", "NluFunctionKind", [
    I32EnumAttrCase<"Linear", 0, "LINEAR">,
    I32EnumAttrCase<"Prelu", 1, "PRELU">,
    I32EnumAttrCase<"Abs", 2, "ABS">,
    I32EnumAttrCase<"Atan", 3, "ATAN">,
    I32EnumAttrCase<"AtanRecip", 4, "ATAN_RECIP">,
    I32EnumAttrCase<"ClassifyNegativeZeroPositive", 5, "CLASSIFY_NEGATIVE_ZERO_POSITIVE">,
    I32EnumAttrCase<"Elu", 6, "ELU">,
    I32EnumAttrCase<"Erf", 7, "ERF">,
    I32EnumAttrCase<"Exp", 8, "EXP">,
    I32EnumAttrCase<"Gelu", 9, "GELU">,
    I32EnumAttrCase<"GeluApproximated", 10, "GELU_APPROXIMATED">,
    I32EnumAttrCase<"GeluApproximatedForPositive", 40, "GELU_APPROXIMATED_FOR_POSITIVE">,
    I32EnumAttrCase<"HardSwish", 11, "HARD_SWISH">,
    I32EnumAttrCase<"IsNegative", 12, "IS_NEGATIVE">,
    I32EnumAttrCase<"IsNotNegative", 13, "IS_NOT_NEGATIVE">,
    I32EnumAttrCase<"IsNotPositive", 14, "IS_NOT_POSITIVE">,
    I32EnumAttrCase<"IsPositive", 15, "IS_POSITIVE">,
    I32EnumAttrCase<"IsZero", 16, "IS_ZERO">,
    I32EnumAttrCase<"NotZero", 17, "NOT_ZERO">,
    I32EnumAttrCase<"Log2Exponent", 18, "LOG2_EXPONENT">,
    I32EnumAttrCase<"Log2Significand", 19, "LOG2_SIGNIFICAND">,
    I32EnumAttrCase<"OneTailedLogisticSigmoid", 20, "ONE_TAILED_LOGISTIC_SIGMOID">,
    I32EnumAttrCase<"PiWithSignOf", 21, "PI_WITH_SIGN_OF">,
    I32EnumAttrCase<"Rsqrt", 22, "RSQRT">,
    I32EnumAttrCase<"Reciprocal", 23, "RECIPROCAL">,
    I32EnumAttrCase<"SignBit", 24, "SIGN_BIT">,
    I32EnumAttrCase<"Sin", 25, "SIN">,
    I32EnumAttrCase<"Softplus", 26, "SOFTPLUS">,
    I32EnumAttrCase<"Sqrt", 27, "SQRT">,
    I32EnumAttrCase<"Tanh", 28, "TANH">,
    I32EnumAttrCase<"TwoToFractionalX", 29, "TWO_TO_FRACTIONAL_X">,
    I32EnumAttrCase<"LogisticSigmoid", 30, "LOGISTIC_SIGMOID">,
    I32EnumAttrCase<"TwoTailedLogisticSigmoid", 31, "TWO_TAILED_LOGISTIC_SIGMOID">,
    I32EnumAttrCase<"Minus16", 32, "MINUS_16">,
    I32EnumAttrCase<"Minus24", 33, "MINUS_24">,
    I32EnumAttrCase<"Cos", 34, "COS">,
    I32EnumAttrCase<"Log", 35, "LOG">,
    I32EnumAttrCase<"LogicalNot", 36, "LOGICAL_NOT">,
    I32EnumAttrCase<"Swish", 37, "SWISH">,
    I32EnumAttrCase<"Exp2", 38, "EXP2">,
    I32EnumAttrCase<"Cbrt", 39, "CBRT">]> {
  let cppNamespace = "::mlir::darwinn";
  let genSpecializedAttr = 0;
}

def Darwinn_NluPreprocessKind : I32EnumAttr<"NluPreprocessKind", "NluPreprocessKind", [
    I32EnumAttrCase<"None", 0, "NONE">,
    I32EnumAttrCase<"ExtractExponent", 1, "EXTRACT_EXPONENT">,
    I32EnumAttrCase<"LoadExponent", 2, "LOAD_EXPONENT">,
    I32EnumAttrCase<"CopySignificand", 3, "COPY_SIGNIFICAND">,
    I32EnumAttrCase<"IntegralPart", 4, "INTEGRAL_PART">,
    I32EnumAttrCase<"FractionalPart", 5, "FRACTIONAL_PART">,
    I32EnumAttrCase<"AndMask", 6, "AND_MASK">,
    I32EnumAttrCase<"OrMask", 7, "OR_MASK">,
    I32EnumAttrCase<"PopCount", 8, "POP_COUNT">,
    I32EnumAttrCase<"CountLeadingZeros", 9, "COUNT_LEADING_ZEROS">,
    I32EnumAttrCase<"XorMask", 10, "XOR_MASK">]> {
  let cppNamespace = "::mlir::darwinn";
  let genSpecializedAttr = 0;
}

def Darwinn_NluPredicateKind : I32EnumAttr<"NluPredicateKind", "NluPredicateKind", [
    I32EnumAttrCase<"None", 0, "NONE">,
    I32EnumAttrCase<"GtZero", 1, "GT_ZERO">,
    I32EnumAttrCase<"GeZero", 2, "GE_ZERO">,
    I32EnumAttrCase<"LtZero", 3, "LT_ZERO">,
    I32EnumAttrCase<"LeZero", 4, "LE_ZERO">,
    I32EnumAttrCase<"EqZero", 5, "EQ_ZERO">,
    I32EnumAttrCase<"NeqZero", 6, "NEQ_ZERO">]> {
  let cppNamespace = "::mlir::darwinn";
  let genSpecializedAttr = 0;
}

def Darwinn_ComputeTypeHintKind : I32EnumAttr<"ComputeTypeHintKind", "ComputeTypeHintKind", [
    I32EnumAttrCase<"None", 0, "NONE">,
    I32EnumAttrCase<"Fc", 1, "FC">,
    I32EnumAttrCase<"Conv", 2, "CONV">,
    I32EnumAttrCase<"Matmul", 3, "MATMUL">,
    I32EnumAttrCase<"FcZinIndexed", 4, "FC_ZIN_INDEXED">,
    I32EnumAttrCase<"FcZoutIndexed", 5, "FC_ZOUT_INDEXED">,
    I32EnumAttrCase<"TransposedConv", 6, "TRANSPOSED_CONV">,
    I32EnumAttrCase<"DynamicQuant", 7, "DYNAMIC_QUANT">]> {
  let cppNamespace = "::mlir::darwinn";
  let genSpecializedAttr = 0;
}

def Darwinn_ComputeLoweringHintKind : I32EnumAttr<"ComputeLoweringHintKind", "ComputeLoweringHintKind", [
    I32EnumAttrCase<"None", 0, "NONE">,
    I32EnumAttrCase<"Tile", 1, "TILE">,
    I32EnumAttrCase<"Pcore", 2, "PCORE">,
    I32EnumAttrCase<"Vcore", 3, "VCORE">,
    I32EnumAttrCase<"Pvcore", 4, "PVCORE">,
    I32EnumAttrCase<"Mimd", 5, "MIMD">,
    I32EnumAttrCase<"Mma", 6, "MMA">,
    I32EnumAttrCase<"Rkhy", 7, "RkHY">,
    I32EnumAttrCase<"Vex", 8, "VEX">]> {
  let cppNamespace = "::mlir::darwinn";
  let genSpecializedAttr = 0;
}

def Darwinn_NluE8M0RoundingKind : I32EnumAttr<"NluE8M0RoundingKind", "NluE8M0RoundingKind", [
    I32EnumAttrCase<"None", 0, "NONE">,
    I32EnumAttrCase<"Floor", 1, "FLOOR">,
    I32EnumAttrCase<"Ceil", 2, "CEIL">]> {
  let cppNamespace = "::mlir::darwinn";
  let genSpecializedAttr = 0;
}

def Darwinn_InterpolateMethodKind : I32EnumAttr<"InterpolateMethodKind", "InterpolateMethodKind", [
    I32EnumAttrCase<"Bilinear", 0, "BILINEAR">,
    I32EnumAttrCase<"NearestNeighbor", 1, "NEAREST_NEIGHBOR">]> {
  let cppNamespace = "::mlir::darwinn";
  let genSpecializedAttr = 0;
}

def Darwinn_FunctionSymmetryKind : I32EnumAttr<"FunctionSymmetryKind", "FunctionSymmetryKind", [
    I32EnumAttrCase<"None", 0, "NONE">,
    I32EnumAttrCase<"Odd", 1, "ODD">,
    I32EnumAttrCase<"Even", 2, "EVEN">]> {
  let cppNamespace = "::mlir::darwinn";
  let genSpecializedAttr = 0;
}

def Darwinn_FunctionSymmetryAttr
    : AttrDef<Darwinn_Dialect, "FunctionSymmetry"> {
  let mnemonic = "function_symmetry";
  let parameters = (ins EnumParameter<Darwinn_FunctionSymmetryKind>:$value);
  let assemblyFormat = "`<` $value `>`";
  let genVerifyDecl = 1;
}

def Darwinn_SplineSegmentAttr
    : AttrDef<Darwinn_Dialect, "SplineSegment"> {
  let mnemonic = "spline_segment";
  let parameters = (ins
      OptionalParameter<"::mlir::ArrayAttr">:$intervals,
      OptionalParameter<"::mlir::ArrayAttr">:$polynomials,
      EnumParameter<Darwinn_FunctionSymmetryKind>:$functionSymmetry);
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

def Darwinn_ComputeOpOptionsAttr
    : AttrDef<Darwinn_Dialect, "ComputeOpOptions"> {
  let mnemonic = "compute_op_options";
  let parameters = (ins
      OptionalParameter<"std::optional<::mlir::darwinn::InnerOperationKind>">:$innerOperation,
      OptionalParameter<"::mlir::IntegerAttr">:$lhsZeroPoint,
      OptionalParameter<"::mlir::IntegerAttr">:$rhsZeroPoint,
      OptionalParameter<"::mlir::FloatAttr">:$scaleImmediate,
      OptionalParameter<"::mlir::FloatAttr">:$inputScaleImmediate,
      OptionalParameter<"::mlir::FloatAttr">:$biasImmediate,
      OptionalParameter<"::mlir::FloatAttr">:$clipLower,
      OptionalParameter<"::mlir::FloatAttr">:$clipUpper,
      OptionalParameter<"::mlir::IntegerAttr">:$outputZeroPoint,
      EnumParameter<Darwinn_LinearFunctionKind>:$linearFunction,
      OptionalParameter<"::mlir::IntegerAttr">:$activationSubChannelBlockSize,
      OptionalParameter<"::mlir::IntegerAttr">:$parameterSubChannelBlockSize,
      OptionalParameter<"::mlir::IntegerAttr">:$replicateReduce,
      OptionalParameter<"std::optional<::mlir::darwinn::NluFunctionKind>">:$nluFunction,
      OptionalParameter<"std::optional<::mlir::darwinn::NluPreprocessKind>">:$nluPreprocessType,
      OptionalParameter<"std::optional<::mlir::darwinn::NluPredicateKind>">:$biasPredicate,
      OptionalParameter<"std::optional<::mlir::darwinn::NluPredicateKind>">:$scalePredicate,
      OptionalParameter<"::mlir::IntegerAttr">:$bitmask,
      OptionalParameter<"std::optional<bool>">:$tryCellgroups,
      OptionalParameter<"::mlir::IntegerAttr">:$alpha,
      OptionalParameter<"::mlir::IntegerAttr">:$beta,
      OptionalParameter<"::mlir::IntegerAttr">:$numGroups,
      OptionalParameter<"::mlir::darwinn::SplineSegmentAttr">:$splineSegment,
      OptionalParameter<"std::optional<::mlir::darwinn::NluE8M0RoundingKind>">:$nluE8M0Rounding,
      OptionalParameter<"std::optional<::mlir::darwinn::ComputeLoweringHintKind>">:$loweringHint,
      OptionalParameter<"std::optional<::mlir::darwinn::ComputeTypeHintKind>">:$computeTypeHint);
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

def Darwinn_InterpolateMethodAttr
    : AttrDef<Darwinn_Dialect, "InterpolateMethod"> {
  let mnemonic = "interpolate_method";
  let parameters = (ins EnumParameter<Darwinn_InterpolateMethodKind>:$value);
  let assemblyFormat = "`<` $value `>`";
  let genVerifyDecl = 1;
}

#endif
