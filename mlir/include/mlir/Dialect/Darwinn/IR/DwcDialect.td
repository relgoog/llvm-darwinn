//===-- DwcDialect.td - DWC dialect ----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the DWC dialect and its operation base class.
// Concrete operations live in DwcOps.td alongside this file.
//
//===----------------------------------------------------------------------===//

#ifndef DWC_DIALECT
#define DWC_DIALECT

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// The DWC Dialect.
//===----------------------------------------------------------------------===//

def Dwc_Dialect : Dialect {
  let name = "dwc";
  let cppNamespace = "::mlir::dwc";
  let description = [{
    The DWC dialect models the compute operations observed in dive binaries,
    including convolutions, elementwise ops, reductions, and reshapes.
  }];
  let useDefaultAttributePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// DWC base classes.
//===----------------------------------------------------------------------===//

class Dwc_Op<string mnemonic, list<Trait> traits = []>
    : Op<Dwc_Dialect, mnemonic, traits> {}

class Dwc_I32Enum<string name, string description, list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, description, cases> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::dwc";
}
class Dwc_I32EnumAttr<string mnemonic, Dwc_I32Enum enumInfo> :
    EnumAttr<Dwc_Dialect, enumInfo, mnemonic> {
  let assemblyFormat = "`<` $value `>`";
}

def Dwc_ComparisonTypeEqual : I32EnumAttrCase<"Equal", 0, "EQUAL">;
def Dwc_ComparisonTypeGreater : I32EnumAttrCase<"Greater", 1, "GREATER">;
def Dwc_ComparisonTypeGreaterEqual : I32EnumAttrCase<"GreaterEqual", 2, "GREATER_EQUAL">;
def Dwc_ComparisonTypeLess : I32EnumAttrCase<"Less", 3, "LESS">;
def Dwc_ComparisonTypeLessEqual : I32EnumAttrCase<"LessEqual", 4, "LESS_EQUAL">;
def Dwc_ComparisonTypeNotEqual : I32EnumAttrCase<"NotEqual", 5, "NOT_EQUAL">;
def Dwc_ComparisonTypeEnum : Dwc_I32Enum<
  "ComparisonType", "DWC comparison type", [
    Dwc_ComparisonTypeEqual,
    Dwc_ComparisonTypeGreater,
    Dwc_ComparisonTypeGreaterEqual,
    Dwc_ComparisonTypeLess,
    Dwc_ComparisonTypeLessEqual,
    Dwc_ComparisonTypeNotEqual
  ]>;
def Dwc_ComparisonTypeAttr : Dwc_I32EnumAttr<"comparison_type", Dwc_ComparisonTypeEnum>;

def Dwc_CumulativeOpTypeSum : I32EnumAttrCase<"Sum", 0, "SUM">;
def Dwc_CumulativeOpTypeEnum : Dwc_I32Enum<
  "CumulativeOpType", "DWC cumulative op type", [
    Dwc_CumulativeOpTypeSum
  ]>;
def Dwc_CumulativeOpTypeAttr : Dwc_I32EnumAttr<"cumulative_op_type", Dwc_CumulativeOpTypeEnum>;

def Dwc_ReductionTypeArgmax : I32EnumAttrCase<"Argmax", 0, "ARGMAX">;
def Dwc_ReductionTypeArgmin : I32EnumAttrCase<"Argmin", 1, "ARGMIN">;
def Dwc_ReductionTypeLogicalAnd : I32EnumAttrCase<"LogicalAnd", 2, "LOGICAL_AND">;
def Dwc_ReductionTypeLogicalOr : I32EnumAttrCase<"LogicalOr", 3, "LOGICAL_OR">;
def Dwc_ReductionTypeMax : I32EnumAttrCase<"Max", 4, "MAX">;
def Dwc_ReductionTypeMean : I32EnumAttrCase<"Mean", 5, "MEAN">;
def Dwc_ReductionTypeMin : I32EnumAttrCase<"Min", 6, "MIN">;
def Dwc_ReductionTypeSum : I32EnumAttrCase<"Sum", 7, "SUM">;
def Dwc_ReductionTypeEnum : Dwc_I32Enum<
  "ReductionType", "DWC reduction type", [
    Dwc_ReductionTypeArgmax,
    Dwc_ReductionTypeArgmin,
    Dwc_ReductionTypeLogicalAnd,
    Dwc_ReductionTypeLogicalOr,
    Dwc_ReductionTypeMax,
    Dwc_ReductionTypeMean,
    Dwc_ReductionTypeMin,
    Dwc_ReductionTypeSum
  ]>;
def Dwc_ReductionTypeAttr : Dwc_I32EnumAttr<"reduction_type", Dwc_ReductionTypeEnum>;

def Dwc_ScalarOpTypeIdentity : I32EnumAttrCase<"Identity", 0, "IDENTITY">;
def Dwc_ScalarOpTypeSubtract : I32EnumAttrCase<"Subtract", 1, "SUBTRACT">;
def Dwc_ScalarOpTypeEnum : Dwc_I32Enum<
  "ScalarOpType", "DWC scalar op type", [
    Dwc_ScalarOpTypeIdentity,
    Dwc_ScalarOpTypeSubtract
  ]>;
def Dwc_ScalarOpTypeAttr : Dwc_I32EnumAttr<"scalar_op_type", Dwc_ScalarOpTypeEnum>;

def Dwc_CwiseOpTypeAdd : I32EnumAttrCase<"Add", 0, "ADD">;
def Dwc_CwiseOpTypeArithmeticLeftShift : I32EnumAttrCase<"ArithmeticLeftShift", 1, "ARITHMETIC_LEFT_SHIFT">;
def Dwc_CwiseOpTypeArithmeticRightShift : I32EnumAttrCase<"ArithmeticRightShift", 2, "ARITHMETIC_RIGHT_SHIFT">;
def Dwc_CwiseOpTypeBitwiseAnd : I32EnumAttrCase<"BitwiseAnd", 3, "BITWISE_AND">;
def Dwc_CwiseOpTypeBitwiseOr : I32EnumAttrCase<"BitwiseOr", 4, "BITWISE_OR">;
def Dwc_CwiseOpTypeBitwiseXor : I32EnumAttrCase<"BitwiseXor", 5, "BITWISE_XOR">;
def Dwc_CwiseOpTypeDivide : I32EnumAttrCase<"Divide", 6, "DIVIDE">;
def Dwc_CwiseOpTypeEqual : I32EnumAttrCase<"Equal", 7, "EQUAL">;
def Dwc_CwiseOpTypeGreater : I32EnumAttrCase<"Greater", 8, "GREATER">;
def Dwc_CwiseOpTypeGreaterEqual : I32EnumAttrCase<"GreaterEqual", 9, "GREATER_EQUAL">;
def Dwc_CwiseOpTypeLess : I32EnumAttrCase<"Less", 10, "LESS">;
def Dwc_CwiseOpTypeLessEqual : I32EnumAttrCase<"LessEqual", 11, "LESS_EQUAL">;
def Dwc_CwiseOpTypeLogicalAnd : I32EnumAttrCase<"LogicalAnd", 12, "LOGICAL_AND">;
def Dwc_CwiseOpTypeLogicalRightShift : I32EnumAttrCase<"LogicalRightShift", 13, "LOGICAL_RIGHT_SHIFT">;
def Dwc_CwiseOpTypeMaximum : I32EnumAttrCase<"Maximum", 14, "MAXIMUM">;
def Dwc_CwiseOpTypeMinimum : I32EnumAttrCase<"Minimum", 15, "MINIMUM">;
def Dwc_CwiseOpTypeModulus : I32EnumAttrCase<"Modulus", 16, "MODULUS">;
def Dwc_CwiseOpTypeMultiply : I32EnumAttrCase<"Multiply", 17, "MULTIPLY">;
def Dwc_CwiseOpTypeNotEqual : I32EnumAttrCase<"NotEqual", 18, "NOT_EQUAL">;
def Dwc_CwiseOpTypePow : I32EnumAttrCase<"Pow", 19, "POW">;
def Dwc_CwiseOpTypeSubtract : I32EnumAttrCase<"Subtract", 20, "SUBTRACT">;
def Dwc_CwiseOpTypeEnum : Dwc_I32Enum<
  "CwiseOpType", "DWC cwise op type", [
    Dwc_CwiseOpTypeAdd,
    Dwc_CwiseOpTypeArithmeticLeftShift,
    Dwc_CwiseOpTypeArithmeticRightShift,
    Dwc_CwiseOpTypeBitwiseAnd,
    Dwc_CwiseOpTypeBitwiseOr,
    Dwc_CwiseOpTypeBitwiseXor,
    Dwc_CwiseOpTypeDivide,
    Dwc_CwiseOpTypeEqual,
    Dwc_CwiseOpTypeGreater,
    Dwc_CwiseOpTypeGreaterEqual,
    Dwc_CwiseOpTypeLess,
    Dwc_CwiseOpTypeLessEqual,
    Dwc_CwiseOpTypeLogicalAnd,
    Dwc_CwiseOpTypeLogicalRightShift,
    Dwc_CwiseOpTypeMaximum,
    Dwc_CwiseOpTypeMinimum,
    Dwc_CwiseOpTypeModulus,
    Dwc_CwiseOpTypeMultiply,
    Dwc_CwiseOpTypeNotEqual,
    Dwc_CwiseOpTypePow,
    Dwc_CwiseOpTypeSubtract
  ]>;
def Dwc_CwiseOpTypeAttr : Dwc_I32EnumAttr<"cwise_op_type", Dwc_CwiseOpTypeEnum>;

def Dwc_SimpleActivationFunctionNone : I32EnumAttrCase<"None", 0, "NONE">;
def Dwc_SimpleActivationFunctionEnum : Dwc_I32Enum<
  "SimpleActivationFunction", "DWC simple activation function", [
    Dwc_SimpleActivationFunctionNone
  ]>;
def Dwc_SimpleActivationFunctionAttr : Dwc_I32EnumAttr<"simple_activation_function", Dwc_SimpleActivationFunctionEnum>;

def Dwc_ActivationFunctionExp : I32EnumAttrCase<"Exp", 0, "EXP">;
def Dwc_ActivationFunctionGeluApproximated : I32EnumAttrCase<"GeluApproximated", 1, "GELU_APPROXIMATED">;
def Dwc_ActivationFunctionLogistic : I32EnumAttrCase<"Logistic", 2, "LOGISTIC">;
def Dwc_ActivationFunctionNone : I32EnumAttrCase<"None", 3, "NONE">;
def Dwc_ActivationFunctionReciprocalSqrt : I32EnumAttrCase<"ReciprocalSqrt", 4, "RECIPROCAL_SQRT">;
def Dwc_ActivationFunctionRelu : I32EnumAttrCase<"Relu", 5, "RELU">;
def Dwc_ActivationFunctionTanh : I32EnumAttrCase<"Tanh", 6, "TANH">;
def Dwc_ActivationFunctionEnum : Dwc_I32Enum<
  "ActivationFunction", "DWC activation function", [
    Dwc_ActivationFunctionExp,
    Dwc_ActivationFunctionGeluApproximated,
    Dwc_ActivationFunctionLogistic,
    Dwc_ActivationFunctionNone,
    Dwc_ActivationFunctionReciprocalSqrt,
    Dwc_ActivationFunctionRelu,
    Dwc_ActivationFunctionTanh
  ]>;
def Dwc_ActivationFunctionAttr : Dwc_I32EnumAttr<"activation_function", Dwc_ActivationFunctionEnum>;

def Dwc_CellOperationMac : I32EnumAttrCase<"Mac", 0, "MAC">;
def Dwc_CellOperationSsd : I32EnumAttrCase<"Ssd", 1, "SSD">;
def Dwc_CellOperationSad : I32EnumAttrCase<"Sad", 2, "SAD">;
def Dwc_CellOperationEnum : Dwc_I32Enum<
  "CellOperation", "DWC cell operation", [
    Dwc_CellOperationMac,
    Dwc_CellOperationSsd,
    Dwc_CellOperationSad
  ]>;
def Dwc_CellOperationAttr : Dwc_I32EnumAttr<"cell_operation", Dwc_CellOperationEnum>;

def Dwc_ClassificationTypeSoftmax : I32EnumAttrCase<"Softmax", 0, "SOFTMAX">;
def Dwc_ClassificationTypeEnum : Dwc_I32Enum<
  "ClassificationType", "DWC classification type", [
    Dwc_ClassificationTypeSoftmax
  ]>;
def Dwc_ClassificationTypeAttr : Dwc_I32EnumAttr<"classification_type", Dwc_ClassificationTypeEnum>;

def Dwc_PerZOutScalePaddingNone : I32EnumAttrCase<"None", 0, "NONE">;
def Dwc_PerZOutScalePaddingEnum : Dwc_I32Enum<
  "PerZOutScalePadding", "DWC per-z-out scale padding", [
    Dwc_PerZOutScalePaddingNone
  ]>;
def Dwc_PerZOutScalePaddingAttr : Dwc_I32EnumAttr<"per_z_out_scale_padding", Dwc_PerZOutScalePaddingEnum>;

def Dwc_PaddingNone : I32EnumAttrCase<"None", 0, "NONE">;
def Dwc_PaddingEnum : Dwc_I32Enum<
  "Padding", "DWC padding", [
    Dwc_PaddingNone
  ]>;
def Dwc_PaddingAttr : Dwc_I32EnumAttr<"padding", Dwc_PaddingEnum>;
def Dwc_MemoryLocationTileSram : I32EnumAttrCase<"TileSram", 0, "TILE_SRAM">;
def Dwc_MemoryLocationHostDram : I32EnumAttrCase<"HostDram", 1, "HOST_DRAM">;
def Dwc_MemoryLocationEnum : Dwc_I32Enum<
  "MemoryLocation", "DWC memory location", [
    Dwc_MemoryLocationTileSram,
    Dwc_MemoryLocationHostDram
  ]>;
def Dwc_MemoryLocationAttr : Dwc_I32EnumAttr<"memory_location", Dwc_MemoryLocationEnum>;

def Dwc_LinearFunctionTypeUnknown : I32EnumAttrCase<"Unknown", 0, "UNKNOWN">;
def Dwc_LinearFunctionTypeEnum : Dwc_I32Enum<
  "LinearFunctionType", "DWC linear function type", [
    Dwc_LinearFunctionTypeUnknown
  ]>;
def Dwc_LinearFunctionTypeAttr : Dwc_I32EnumAttr<"linear_function_type", Dwc_LinearFunctionTypeEnum>;

def Dwc_VicaCustomPaddingUnknown : I32EnumAttrCase<"Unknown", 0, "UNKNOWN">;
def Dwc_VicaCustomPaddingEnum : Dwc_I32Enum<
  "VicaCustomPadding", "DWC Vica custom padding", [
    Dwc_VicaCustomPaddingUnknown
  ]>;
def Dwc_VicaCustomPaddingAttr : Dwc_I32EnumAttr<"vica_custom_padding", Dwc_VicaCustomPaddingEnum>;

def Dwc_AFUnknown : I32EnumAttrCase<"Unknown", 0, "UNKNOWN">;
def Dwc_AFEnum : Dwc_I32Enum<
  "AF", "DWC AF", [
    Dwc_AFUnknown
  ]>;
def Dwc_AFAttr : Dwc_I32EnumAttr<"af", Dwc_AFEnum>;

def Dwc_AlgorithmUnknown : I32EnumAttrCase<"Unknown", 0, "UNKNOWN">;
def Dwc_AlgorithmEnum : Dwc_I32Enum<
  "Algorithm", "DWC algorithm", [
    Dwc_AlgorithmUnknown
  ]>;
def Dwc_AlgorithmAttr : Dwc_I32EnumAttr<"algorithm", Dwc_AlgorithmEnum>;

def Dwc_AttentionNonLinearFunctionUnknown : I32EnumAttrCase<"Unknown", 0, "UNKNOWN">;
def Dwc_AttentionNonLinearFunctionEnum : Dwc_I32Enum<
  "AttentionNonLinearFunction", "DWC attention non linear function", [
    Dwc_AttentionNonLinearFunctionUnknown
  ]>;
def Dwc_AttentionNonLinearFunctionAttr : Dwc_I32EnumAttr<"attention_non_linear_function", Dwc_AttentionNonLinearFunctionEnum>;

def Dwc_BitSelectTypeUnknown : I32EnumAttrCase<"Unknown", 0, "UNKNOWN">;
def Dwc_BitSelectTypeEnum : Dwc_I32Enum<
  "BitSelectType", "DWC bit select type", [
    Dwc_BitSelectTypeUnknown
  ]>;
def Dwc_BitSelectTypeAttr : Dwc_I32EnumAttr<"bit_select_type", Dwc_BitSelectTypeEnum>;

def Dwc_CompressionModeUnknown : I32EnumAttrCase<"Unknown", 0, "UNKNOWN">;
def Dwc_CompressionModeEnum : Dwc_I32Enum<
  "CompressionMode", "DWC compression mode", [
    Dwc_CompressionModeUnknown
  ]>;
def Dwc_CompressionModeAttr : Dwc_I32EnumAttr<"compression_mode", Dwc_CompressionModeEnum>;

def Dwc_CustomPadValueTypeUnknown : I32EnumAttrCase<"Unknown", 0, "UNKNOWN">;
def Dwc_CustomPadValueTypeEnum : Dwc_I32Enum<
  "CustomPadValueType", "DWC custom pad value type", [
    Dwc_CustomPadValueTypeUnknown
  ]>;
def Dwc_CustomPadValueTypeAttr : Dwc_I32EnumAttr<"custom_pad_value_type", Dwc_CustomPadValueTypeEnum>;

def Dwc_DeviceTypeUnknown : I32EnumAttrCase<"Unknown", 0, "UNKNOWN">;
def Dwc_DeviceTypeEnum : Dwc_I32Enum<
  "DeviceType", "DWC device type", [
    Dwc_DeviceTypeUnknown
  ]>;
def Dwc_DeviceTypeAttr : Dwc_I32EnumAttr<"device_type", Dwc_DeviceTypeEnum>;

def Dwc_EngineUnknown : I32EnumAttrCase<"Unknown", 0, "UNKNOWN">;
def Dwc_EngineEnum : Dwc_I32Enum<
  "Engine", "DWC engine", [
    Dwc_EngineUnknown
  ]>;
def Dwc_EngineAttr : Dwc_I32EnumAttr<"engine", Dwc_EngineEnum>;

def Dwc_HibGatherFilterUnknown : I32EnumAttrCase<"Unknown", 0, "UNKNOWN">;
def Dwc_HibGatherFilterEnum : Dwc_I32Enum<
  "HibGatherFilter", "DWC HIB gather filter", [
    Dwc_HibGatherFilterUnknown
  ]>;
def Dwc_HibGatherFilterAttr : Dwc_I32EnumAttr<"hib_gather_filter", Dwc_HibGatherFilterEnum>;

def Dwc_HostSpaceUnknown : I32EnumAttrCase<"Unknown", 0, "UNKNOWN">;
def Dwc_HostSpaceEnum : Dwc_I32Enum<
  "HostSpace", "DWC host space", [
    Dwc_HostSpaceUnknown
  ]>;
def Dwc_HostSpaceAttr : Dwc_I32EnumAttr<"host_space", Dwc_HostSpaceEnum>;

def Dwc_ImageFormatUnknown : I32EnumAttrCase<"Unknown", 0, "UNKNOWN">;
def Dwc_ImageFormatEnum : Dwc_I32Enum<
  "ImageFormat", "DWC image format", [
    Dwc_ImageFormatUnknown
  ]>;
def Dwc_ImageFormatAttr : Dwc_I32EnumAttr<"image_format", Dwc_ImageFormatEnum>;

def Dwc_KernelLevelUnknown : I32EnumAttrCase<"Unknown", 0, "UNKNOWN">;
def Dwc_KernelLevelEnum : Dwc_I32Enum<
  "KernelLevel", "DWC kernel level", [
    Dwc_KernelLevelUnknown
  ]>;
def Dwc_KernelLevelAttr : Dwc_I32EnumAttr<"kernel_level", Dwc_KernelLevelEnum>;

def Dwc_MeshDimUnknown : I32EnumAttrCase<"Unknown", 0, "UNKNOWN">;
def Dwc_MeshDimEnum : Dwc_I32Enum<
  "MeshDim", "DWC mesh dim", [
    Dwc_MeshDimUnknown
  ]>;
def Dwc_MeshDimAttr : Dwc_I32EnumAttr<"mesh_dim", Dwc_MeshDimEnum>;

def Dwc_NluE8M0RoundingUnknown : I32EnumAttrCase<"Unknown", 0, "UNKNOWN">;
def Dwc_NluE8M0RoundingEnum : Dwc_I32Enum<
  "NluE8M0Rounding", "DWC NLU E8M0 rounding", [
    Dwc_NluE8M0RoundingUnknown
  ]>;
def Dwc_NluE8M0RoundingAttr : Dwc_I32EnumAttr<"nlu_e8m0_rounding", Dwc_NluE8M0RoundingEnum>;

def Dwc_NluPreprocessingUnknown : I32EnumAttrCase<"Unknown", 0, "UNKNOWN">;
def Dwc_NluPreprocessingEnum : Dwc_I32Enum<
  "NluPreprocessing", "DWC NLU preprocessing", [
    Dwc_NluPreprocessingUnknown
  ]>;
def Dwc_NluPreprocessingAttr : Dwc_I32EnumAttr<"nlu_preprocessing", Dwc_NluPreprocessingEnum>;

def Dwc_NormalizationTypeUnknown : I32EnumAttrCase<"Unknown", 0, "UNKNOWN">;
def Dwc_NormalizationTypeEnum : Dwc_I32Enum<
  "NormalizationType", "DWC normalization type", [
    Dwc_NormalizationTypeUnknown
  ]>;
def Dwc_NormalizationTypeAttr : Dwc_I32EnumAttr<"normalization_type", Dwc_NormalizationTypeEnum>;

def Dwc_PaddingValueTypeUnknown : I32EnumAttrCase<"Unknown", 0, "UNKNOWN">;
def Dwc_PaddingValueTypeEnum : Dwc_I32Enum<
  "PaddingValueType", "DWC padding value type", [
    Dwc_PaddingValueTypeUnknown
  ]>;
def Dwc_PaddingValueTypeAttr : Dwc_I32EnumAttr<"padding_value_type", Dwc_PaddingValueTypeEnum>;

def Dwc_PoolUnknown : I32EnumAttrCase<"Unknown", 0, "UNKNOWN">;
def Dwc_PoolEnum : Dwc_I32Enum<
  "Pool", "DWC pool", [
    Dwc_PoolUnknown
  ]>;
def Dwc_PoolAttr : Dwc_I32EnumAttr<"pool", Dwc_PoolEnum>;

def Dwc_ReduceWindowTypeUnknown : I32EnumAttrCase<"Unknown", 0, "UNKNOWN">;
def Dwc_ReduceWindowTypeEnum : Dwc_I32Enum<
  "ReduceWindowType", "DWC reduce window type", [
    Dwc_ReduceWindowTypeUnknown
  ]>;
def Dwc_ReduceWindowTypeAttr : Dwc_I32EnumAttr<"reduce_window_type", Dwc_ReduceWindowTypeEnum>;

def Dwc_ScatterOperationUnknown : I32EnumAttrCase<"Unknown", 0, "UNKNOWN">;
def Dwc_ScatterOperationEnum : Dwc_I32Enum<
  "ScatterOperation", "DWC scatter operation", [
    Dwc_ScatterOperationUnknown
  ]>;
def Dwc_ScatterOperationAttr : Dwc_I32EnumAttr<"scatter_operation", Dwc_ScatterOperationEnum>;

def Dwc_ScatterTensorLsOperationUnknown : I32EnumAttrCase<"Unknown", 0, "UNKNOWN">;
def Dwc_ScatterTensorLsOperationEnum : Dwc_I32Enum<
  "ScatterTensorLsOperation", "DWC scatter tensor LS operation", [
    Dwc_ScatterTensorLsOperationUnknown
  ]>;
def Dwc_ScatterTensorLsOperationAttr : Dwc_I32EnumAttr<"scatter_tensor_ls_operation", Dwc_ScatterTensorLsOperationEnum>;

def Dwc_SpillLocationUnknown : I32EnumAttrCase<"Unknown", 0, "UNKNOWN">;
def Dwc_SpillLocationEnum : Dwc_I32Enum<
  "SpillLocation", "DWC spill location", [
    Dwc_SpillLocationUnknown
  ]>;
def Dwc_SpillLocationAttr : Dwc_I32EnumAttr<"spill_location", Dwc_SpillLocationEnum>;

def Dwc_StrideMethodUnknown : I32EnumAttrCase<"Unknown", 0, "UNKNOWN">;
def Dwc_StrideMethodEnum : Dwc_I32Enum<
  "StrideMethod", "DWC stride method", [
    Dwc_StrideMethodUnknown
  ]>;
def Dwc_StrideMethodAttr : Dwc_I32EnumAttr<"stride_method", Dwc_StrideMethodEnum>;

def Dwc_TensorOpGatherOperationUnknown : I32EnumAttrCase<"Unknown", 0, "UNKNOWN">;
def Dwc_TensorOpGatherOperationEnum : Dwc_I32Enum<
  "TensorOpGatherOperation", "DWC tensor op gather operation", [
    Dwc_TensorOpGatherOperationUnknown
  ]>;
def Dwc_TensorOpGatherOperationAttr : Dwc_I32EnumAttr<"tensor_op_gather_operation", Dwc_TensorOpGatherOperationEnum>;

def Dwc_TransformationTypeUnknown : I32EnumAttrCase<"Unknown", 0, "UNKNOWN">;
def Dwc_TransformationTypeEnum : Dwc_I32Enum<
  "TransformationType", "DWC transformation type", [
    Dwc_TransformationTypeUnknown
  ]>;
def Dwc_TransformationTypeAttr : Dwc_I32EnumAttr<"transformation_type", Dwc_TransformationTypeEnum>;

def Dwc_TruncateFloatsUnknown : I32EnumAttrCase<"Unknown", 0, "UNKNOWN">;
def Dwc_TruncateFloatsEnum : Dwc_I32Enum<
  "TruncateFloats", "DWC truncate floats", [
    Dwc_TruncateFloatsUnknown
  ]>;
def Dwc_TruncateFloatsAttr : Dwc_I32EnumAttr<"truncate_floats", Dwc_TruncateFloatsEnum>;

def Dwc_VRGKHOperationModeUnknown : I32EnumAttrCase<"Unknown", 0, "UNKNOWN">;
def Dwc_VRGKHOperationModeEnum : Dwc_I32Enum<
  "VRGKHOperationMode", "DWC VRGKH operation mode", [
    Dwc_VRGKHOperationModeUnknown
  ]>;
def Dwc_VRGKHOperationModeAttr : Dwc_I32EnumAttr<"vrgkh_operation_mode", Dwc_VRGKHOperationModeEnum>;

def Dwc_MemorySpaceHostMemory : I32EnumAttrCase<"HostMemory", 0, "HOST_MEMORY">;
def Dwc_MemorySpaceEnum : Dwc_I32Enum<
  "MemorySpace", "DWC memory space", [
    Dwc_MemorySpaceHostMemory
  ]>;
def Dwc_MemorySpaceAttr : Dwc_I32EnumAttr<"memory_space", Dwc_MemorySpaceEnum>;

def Dwc_ProbeSubTensorAttr : AttrDef<Dwc_Dialect, "ProbeSubTensor"> {
  let summary = "DWC probe subtensor attribute.";
  let mnemonic = "probe_subtensor";
  let parameters = (ins
    "StringAttr":$full_probe_id,
    "DenseElementsAttr":$padding_amount,
    "IntegerAttr":$padding_value,
    "DenseElementsAttr":$slice_begin,
    "DenseElementsAttr":$slice_size
  );
  let assemblyFormat = "`<` $full_probe_id `,` $padding_amount `,` $padding_value `,` $slice_begin `,` $slice_size `>`";
}

def Dwc_ParameterLookupTableAttr : AttrDef<Dwc_Dialect, "ParameterLookupTable"> {
  let summary = "DWC parameter lookup table attribute.";
  let mnemonic = "parameter_lookup_table";
  let parameters = (ins
    "DenseIntElementsAttr":$lookup_table,
    "IntegerAttr":$zero_point
  );
  let assemblyFormat = "`<` $lookup_table `,` $zero_point `>`";
}

def Dwc_ResamplerOptionsAttr : AttrDef<Dwc_Dialect, "ResamplerOptions"> {
  let summary = "DWC resampler options attribute.";
  let mnemonic = "resampler_options";
  let parameters = (ins
    "IntegerAttr":$fractional_index_bitwidth,
    "IntegerAttr":$integer_index_bitwidth
  );
  let assemblyFormat = "`<` $fractional_index_bitwidth `,` $integer_index_bitwidth `>`";
}

def Dwc_DimensionLayoutAttr : AttrDef<Dwc_Dialect, "DimensionLayout"> {
  let summary = "DWC dimension layout attribute.";
  let mnemonic = "dimension_layout";
  let parameters = (ins
    "ArrayAttr":$pattern,
    "IntegerAttr":$pattern_repeat
  );
  let assemblyFormat = "`<` `(` $pattern `)` `,` $pattern_repeat `>`";
}

def Dwc_MaterializePolicyAttr : AttrDef<Dwc_Dialect, "MaterializePolicy"> {
  let summary = "DWC materialize policy attribute.";
  let mnemonic = "materialize_policy";
  let parameters = (ins
    "IntegerAttr":$bundle_id,
    "DenseI32ArrayAttr":$sharding_factors,
    "DictionaryAttr":$tiling,
    "ArrayAttr":$transformation_types
  );
  let assemblyFormat = "`<` $bundle_id `,` $sharding_factors `,` $tiling `,` $transformation_types `>`";
}

#endif // DWC_DIALECT
