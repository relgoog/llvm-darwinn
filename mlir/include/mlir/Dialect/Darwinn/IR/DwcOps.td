//===-- DwcOps.td - DWC dialect operations -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Operation set for the DWC dialect, one op per blob-evidenced mnemonic.
// Attr requirements come from blob verifier strings, operand counts from
// operand constraint strings. Ops without operand evidence take any count.
//
//===----------------------------------------------------------------------===//

#ifndef DWC_OPS
#define DWC_OPS

include "mlir/IR/OpBase.td"

include "mlir/Dialect/Darwinn/IR/DwcDialect.td"

// Operator: add
def Dwc_AddOp : Dwc_Op<"add"> {
  let summary = "DWC add operation.";
  let description = [{
    Takes any number of operands. Requires `activation_function`.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: atan
def Dwc_AtanOp : Dwc_Op<"atan"> {
  let summary = "DWC atan operation.";
  let description = [{
    Exactly 1 i1, i8, i16, f16, bf16, or f32 operand per the blob constraint excluding quant types.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: batch_matrix_nms
def Dwc_BatchMatrixNmsOp : Dwc_Op<"batch_matrix_nms"> {
  let summary = "DWC batch_matrix_nms operation.";
  let description = [{
    Takes any number of operands. Requires `max_output_size`, `score_threshold`, `sigma`, `suppress_top_k`.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: bitcast
def Dwc_BitcastOp : Dwc_Op<"bitcast"> {
  let summary = "DWC bitcast operation.";
  let description = [{
    Takes any number of operands. Requires `output_element_type`.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: cast
def Dwc_CastOp : Dwc_Op<"cast"> {
  let summary = "DWC cast operation.";
  let description = [{
    Exactly 1 16-bit float or bfloat16 operand per the blob element constraints.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: ceil
def Dwc_CeilOp : Dwc_Op<"ceil"> {
  let summary = "DWC ceil operation.";
  let description = [{
    Exactly 1 i1, i8, i16, i32, f16, bf16, or f32 operand per the blob constraints excluding quant types.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: classifier
def Dwc_ClassifierOp : Dwc_Op<"classifier"> {
  let summary = "DWC classifier operation.";
  let description = [{
    Takes any number of operands. Requires `axis`, `beta`, `op_type`.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: compare
def Dwc_CompareOp : Dwc_Op<"compare"> {
  let summary = "DWC compare operation.";
  let description = [{
    Takes any number of operands. Requires `activation_function`, `compare_type`.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: concatenation
def Dwc_ConcatenationOp : Dwc_Op<"concatenation"> {
  let summary = "DWC concatenation operation.";
  let description = [{
    Takes any number of operands. Requires `mode`.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: const
def Dwc_ConstOp : Dwc_Op<"const"> {
  let summary = "DWC const operation.";
  let description = [{
    Takes any number of operands. Requires `value`.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: convolution
def Dwc_ConvolutionOp : Dwc_Op<"convolution"> {
  let summary = "DWC convolution operation.";
  let description = [{
    Takes any number of operands. Requires `activation_function`, `cell_operation`, `pad`, `x_dilation_rate`, `x_stride`, `y_dilation_rate`, `y_stride`.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: convolution_v2
def Dwc_ConvolutionV2Op : Dwc_Op<"convolution_v2"> {
  let summary = "DWC convolution_v2 operation.";
  let description = [{
    Takes any number of operands. Requires `activation_function`, `cell_operation`, `pad`, `x_dilation_rate`, `x_stride`, `y_dilation_rate`, `y_stride`.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: cos
def Dwc_CosOp : Dwc_Op<"cos"> {
  let summary = "DWC cos operation.";
  let description = [{
    Exactly 1 i1, i8, i16, f16, bf16, or f32 operand per the blob constraint excluding quant types.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: cumulative
def Dwc_CumulativeOp : Dwc_Op<"cumulative"> {
  let summary = "DWC cumulative operation.";
  let description = [{
    Exactly 1 i32, i64, u32, or u64 operand per the blob element constraint. Requires `axis`, `exclusive`, `op_type`.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: cwise
def Dwc_CwiseOp : Dwc_Op<"cwise"> {
  let summary = "DWC cwise operation.";
  let description = [{
    Exactly 2 non-0-ranked tensor operands per the blob constraints. Requires `activation_function`, `op_type`.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: depthwise_convolution_v2
def Dwc_DepthwiseConvolutionV2Op : Dwc_Op<"depthwise_convolution_v2"> {
  let summary = "DWC depthwise_convolution_v2 operation.";
  let description = [{
    Takes any number of operands. Requires `activation_function`, `cell_operation`, `depth_multiplier`, `pad`, `x_dilation_rate`, `x_stride`, `y_dilation_rate`, `y_stride`.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: divide
def Dwc_DivideOp : Dwc_Op<"divide"> {
  let summary = "DWC divide operation.";
  let description = [{
    Takes any number of operands. Requires `activation_function`.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: dynamic_slice
def Dwc_DynamicSliceOp : Dwc_Op<"dynamic_slice"> {
  let summary = "DWC dynamic_slice operation.";
  let description = [{
    Takes any number of operands. Requires `mode`, `read_location`, `slice_size`.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: dynamic_update_slice
def Dwc_DynamicUpdateSliceOp : Dwc_Op<"dynamic_update_slice"> {
  let summary = "DWC dynamic_update_slice operation.";
  let description = [{
    Takes any number of operands. Requires `mode`, `write_location`.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: erf
def Dwc_ErfOp : Dwc_Op<"erf"> {
  let summary = "DWC erf operation.";
  let description = [{
    Exactly 1 i1, i8, i16, f16, bf16, or f32 operand per the blob constraint excluding quant types.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: exp
def Dwc_ExpOp : Dwc_Op<"exp"> {
  let summary = "DWC exp operation.";
  let description = [{
    Exactly 1 32-bit float operand per the blob element constraint.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: expm1
def Dwc_Expm1Op : Dwc_Op<"expm1"> {
  let summary = "DWC expm1 operation.";
  let description = [{
    Exactly 1 i1, i8, i16, f16, bf16, or f32 operand per the blob constraint excluding quant types.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: floor
def Dwc_FloorOp : Dwc_Op<"floor"> {
  let summary = "DWC floor operation.";
  let description = [{
    Exactly 1 i1, i8, i16, i32, f16, bf16, or f32 operand per the blob constraints excluding quant types.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: floor_div
def Dwc_FloorDivOp : Dwc_Op<"floor_div"> {
  let summary = "DWC floor_div operation.";
  let description = [{
    Exactly 2 32-bit float, bfloat16, or 16-bit float operands per the blob element constraints.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: fully_connected
def Dwc_FullyConnectedOp : Dwc_Op<"fully_connected"> {
  let summary = "DWC fully_connected operation.";
  let description = [{
    Takes any number of operands. Requires `activation_function`, `cell_operation`.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: gather
def Dwc_GatherOp : Dwc_Op<"gather"> {
  let summary = "DWC gather operation.";
  let description = [{
    Takes any number of operands. Requires `axis`, `batch_dims`.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: generic_compute
def Dwc_GenericComputeOp : Dwc_Op<"generic_compute"> {
  let summary = "DWC generic_compute operation.";
  let description = [{
    Takes any number of operands. Requires `activation_function`, `indexing_maps`, `linear_function`.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: generic_constant
def Dwc_GenericConstantOp : Dwc_Op<"generic_constant"> {
  let summary = "DWC generic_constant operation.";
  let description = [{
    Takes any number of operands. Requires `value`.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: generic_conv
def Dwc_GenericConvOp : Dwc_Op<"generic_conv"> {
  let summary = "DWC generic_conv operation.";
  let description = [{
    Takes any number of operands. Requires `activation_function`, `batch_group_count`, `feature_group_count`, `input_dilation`, `padding_amount`, `param_dilation`, `param_reversal`, `stride`.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: generic_dot
def Dwc_GenericDotOp : Dwc_Op<"generic_dot"> {
  let summary = "DWC generic_dot operation.";
  let description = [{
    Takes any number of operands. Requires `activation_function`, `batch_dim_count`, `contracting_dim_count`.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: log1p
def Dwc_Log1pOp : Dwc_Op<"log1p"> {
  let summary = "DWC log1p operation.";
  let description = [{
    Exactly 1 i1, i8, i16, f16, bf16, or f32 operand per the blob constraint excluding quant types.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: logistic
def Dwc_LogisticOp : Dwc_Op<"logistic"> {
  let summary = "DWC logistic operation.";
  let description = [{
    Exactly 1 i1, i8, i16, f16, bf16, or f32 operand per the blob constraint excluding quant types.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: matrix_multiply
def Dwc_MatrixMultiplyOp : Dwc_Op<"matrix_multiply"> {
  let summary = "DWC matrix_multiply operation.";
  let description = [{
    Takes any number of operands. Requires `activation_function`, `transpose_rhs`.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: maximum
def Dwc_MaximumOp : Dwc_Op<"maximum"> {
  let summary = "DWC maximum operation.";
  let description = [{
    Takes any number of operands. Requires `activation_function`.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: minimum
def Dwc_MinimumOp : Dwc_Op<"minimum"> {
  let summary = "DWC minimum operation.";
  let description = [{
    Takes any number of operands. Requires `activation_function`.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: multiply
def Dwc_MultiplyOp : Dwc_Op<"multiply"> {
  let summary = "DWC multiply operation.";
  let description = [{
    Takes any number of operands. Requires `activation_function`.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: not
def Dwc_NotOp : Dwc_Op<"not"> {
  let summary = "DWC not operation.";
  let description = [{
    Exactly 1 1-bit signless integer operand per the blob element constraint.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: one_hot
def Dwc_OneHotOp : Dwc_Op<"one_hot"> {
  let summary = "DWC one_hot operation.";
  let description = [{
    Exactly 1 i16, i32, i64, or u16 operand per the blob element constraints. Requires `axis`.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: padding
def Dwc_PaddingOp : Dwc_Op<"padding"> {
  let summary = "DWC padding operation.";
  let description = [{
    Exactly 1 non-0-ranked tensor operand per the blob constraints. Requires `dimension`, `padding_value`, `post_padding`, `pre_padding`.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: pop_count
def Dwc_PopCountOp : Dwc_Op<"pop_count"> {
  let summary = "DWC pop_count operation.";
  let description = [{
    Exactly 1 i1, i8, i16, i32, u8, or u16 operand per the blob element constraints.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: pow
def Dwc_PowOp : Dwc_Op<"pow"> {
  let summary = "DWC pow operation.";
  let description = [{
    Exactly 2 bfloat16, f16, f32, i32, or u32 operands per the blob element constraints.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: pseudo_split
def Dwc_PseudoSplitOp : Dwc_Op<"pseudo_split"> {
  let summary = "DWC pseudo_split operation.";
  let description = [{
    Operand 0 is a 0D 32-bit signless integer tensor and operand 1 is f32, i16, i32, i64, QI8, or QUI8 per the blob constraints. Requires `num_splits`.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: reduction
def Dwc_ReductionOp : Dwc_Op<"reduction"> {
  let summary = "DWC reduction operation.";
  let description = [{
    Exactly 1 16-bit float, bfloat16, or 32-bit float operand per the blob element constraints. Requires `activation_function`, `dimensions`, `op_type`.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: remainder
def Dwc_RemainderOp : Dwc_Op<"remainder"> {
  let summary = "DWC remainder operation.";
  let description = [{
    Takes any number of operands. Requires `activation_function`.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: rescaling
def Dwc_RescalingOp : Dwc_Op<"rescaling"> {
  let summary = "DWC rescaling operation.";
  let description = [{
    Exactly 1 16-bit float or bfloat16 operand per the blob element constraint. Requires `activation_function`, `output_activation_per_z_out_scales`, `per_z_out_scales_padding`.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: reshape
def Dwc_ReshapeOp : Dwc_Op<"reshape"> {
  let summary = "DWC reshape operation.";
  let description = [{
    Exactly 1 Rank 3 or Rank 4 tensor operand per the blob constraints.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: round
def Dwc_RoundOp : Dwc_Op<"round"> {
  let summary = "DWC round operation.";
  let description = [{
    Exactly 1 i1, i8, i16, f16, bf16, or f32 operand per the blob constraint excluding quant types.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: round_nearest_afz
def Dwc_RoundNearestAfzOp : Dwc_Op<"round_nearest_afz"> {
  let summary = "DWC round_nearest_afz operation.";
  let description = [{
    Exactly 1 i1, i8, i16, f16, bf16, or f32 operand per the blob constraint excluding quant types.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: rsqrt
def Dwc_RsqrtOp : Dwc_Op<"rsqrt"> {
  let summary = "DWC rsqrt operation.";
  let description = [{
    Exactly 1 i1, i8, i16, f16, bf16, or f32 operand per the blob constraint excluding quant types.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: scalar
def Dwc_ScalarOp : Dwc_Op<"scalar"> {
  let summary = "DWC scalar operation.";
  let description = [{
    Takes any number of operands. Requires `op_type`.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: scatter_nd
def Dwc_ScatterNdOp : Dwc_Op<"scatter_nd"> {
  let summary = "DWC scatter_nd operation.";
  let description = [{
    Takes any number of operands. Requires `shape`.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: select
def Dwc_SelectOp : Dwc_Op<"select"> {
  let summary = "DWC select operation.";
  let description = [{
    Operand 0 is a statically shaped 1-bit signless integer tensor and operand 1 is 8/16/32/64-bit integer or 32/64-bit float per the blob constraints.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: sign
def Dwc_SignOp : Dwc_Op<"sign"> {
  let summary = "DWC sign operation.";
  let description = [{
    Exactly 1 i1, i8, i16, i32, i64, f16, bf16, or f32 operand per the blob constraints excluding quant types. Requires `preserve_negative_zero`.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: sin
def Dwc_SinOp : Dwc_Op<"sin"> {
  let summary = "DWC sin operation.";
  let description = [{
    Exactly 1 i1, i8, i16, f16, bf16, or f32 operand per the blob constraint excluding quant types.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: slice
def Dwc_SliceOp : Dwc_Op<"slice"> {
  let summary = "DWC slice operation.";
  let description = [{
    Takes any number of operands. Requires `in_begin`, `in_size`, `mode`.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: sort
def Dwc_SortOp : Dwc_Op<"sort"> {
  let summary = "DWC sort operation.";
  let description = [{
    Takes any number of operands. Requires `compare_tuple_projection`, `compare_type`, `dimension`, `is_stable`.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: sqrt
def Dwc_SqrtOp : Dwc_Op<"sqrt"> {
  let summary = "DWC sqrt operation.";
  let description = [{
    Exactly 1 i1, i8, i16, f16, bf16, or f32 operand per the blob constraint excluding quant types.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: subtract
def Dwc_SubtractOp : Dwc_Op<"subtract"> {
  let summary = "DWC subtract operation.";
  let description = [{
    Takes any number of operands. Requires `activation_function`.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: tan
def Dwc_TanOp : Dwc_Op<"tan"> {
  let summary = "DWC tan operation.";
  let description = [{
    Exactly 1 i1, i8, i16, f16, bf16, or f32 operand per the blob constraint excluding quant types.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: tanh
def Dwc_TanhOp : Dwc_Op<"tanh"> {
  let summary = "DWC tanh operation.";
  let description = [{
    Exactly 1 i1, i8, i16, f16, bf16, or f32 operand per the blob constraint excluding quant types.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: transpose
def Dwc_TransposeOp : Dwc_Op<"transpose"> {
  let summary = "DWC transpose operation.";
  let description = [{
    Exactly 1 Rank 3 tensor operand per the blob constraint. Requires `permutation`.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: transposed_convolution
def Dwc_TransposedConvolutionOp : Dwc_Op<"transposed_convolution"> {
  let summary = "DWC transposed_convolution operation.";
  let description = [{
    Takes any number of operands. Requires `activation_function`, `cell_operation`, `pad`, `x_dilation_rate`, `x_out_dim`, `x_stride`, `y_dilation_rate`, `y_out_dim`, `y_stride`.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: unsorted_segment_reduce
def Dwc_UnsortedSegmentReduceOp : Dwc_Op<"unsorted_segment_reduce"> {
  let summary = "DWC unsorted_segment_reduce operation.";
  let description = [{
    Takes any number of operands. Requires `num_segments`, `op_type`.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: abs
def Dwc_AbsOp : Dwc_Op<"abs"> {
  let summary = "DWC abs operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: arange
def Dwc_ArangeOp : Dwc_Op<"arange"> {
  let summary = "DWC arange operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: attention
def Dwc_AttentionOp : Dwc_Op<"attention"> {
  let summary = "DWC attention operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: attention_non_linear_function
def Dwc_AttentionNonLinearFunctionOp : Dwc_Op<"attention_non_linear_function"> {
  let summary = "DWC attention_non_linear_function operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: batch_to_space
def Dwc_BatchToSpaceOp : Dwc_Op<"batch_to_space"> {
  let summary = "DWC batch_to_space operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: bit_select
def Dwc_BitSelectOp : Dwc_Op<"bit_select"> {
  let summary = "DWC bit_select operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: broadcast
def Dwc_BroadcastOp : Dwc_Op<"broadcast"> {
  let summary = "DWC broadcast operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: cbrt
def Dwc_CbrtOp : Dwc_Op<"cbrt"> {
  let summary = "DWC cbrt operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: clamp
def Dwc_ClampOp : Dwc_Op<"clamp"> {
  let summary = "DWC clamp operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: collective_permute
def Dwc_CollectivePermuteOp : Dwc_Op<"collective_permute"> {
  let summary = "DWC collective_permute operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: composite
def Dwc_CompositeOp : Dwc_Op<"composite"> {
  let summary = "DWC composite operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: const_none
def Dwc_ConstNoneOp : Dwc_Op<"const_none"> {
  let summary = "DWC const_none operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: convolution_sub_channel
def Dwc_ConvolutionSubChannelOp : Dwc_Op<"convolution_sub_channel"> {
  let summary = "DWC convolution_sub_channel operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: cost_volume
def Dwc_CostVolumeOp : Dwc_Op<"cost_volume"> {
  let summary = "DWC cost_volume operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: count_leading_zeros
def Dwc_CountLeadingZerosOp : Dwc_Op<"count_leading_zeros"> {
  let summary = "DWC count_leading_zeros operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: custom_compute
def Dwc_CustomComputeOp : Dwc_Op<"custom_compute"> {
  let summary = "DWC custom_compute operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: declare_tensor
def Dwc_DeclareTensorOp : Dwc_Op<"declare_tensor"> {
  let summary = "DWC declare_tensor operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: depth_to_space
def Dwc_DepthToSpaceOp : Dwc_Op<"depth_to_space"> {
  let summary = "DWC depth_to_space operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: depthwise_convolution
def Dwc_DepthwiseConvolutionOp : Dwc_Op<"depthwise_convolution"> {
  let summary = "DWC depthwise_convolution operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: device_launch
def Dwc_DeviceLaunchOp : Dwc_Op<"device_launch"> {
  let summary = "DWC device_launch operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: dma_op_gather
def Dwc_DmaOpGatherOp : Dwc_Op<"dma_op_gather"> {
  let summary = "DWC dma_op_gather operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: dynamic_broadcast
def Dwc_DynamicBroadcastOp : Dwc_Op<"dynamic_broadcast"> {
  let summary = "DWC dynamic_broadcast operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: dynamic_quantize
def Dwc_DynamicQuantizeOp : Dwc_Op<"dynamic_quantize"> {
  let summary = "DWC dynamic_quantize operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: ensure_shape
def Dwc_EnsureShapeOp : Dwc_Op<"ensure_shape"> {
  let summary = "DWC ensure_shape operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: extern_call
def Dwc_ExternCallOp : Dwc_Op<"extern_call"> {
  let summary = "DWC extern_call operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: fast_walsh_hadamard_transform
def Dwc_FastWalshHadamardTransformOp : Dwc_Op<"fast_walsh_hadamard_transform"> {
  let summary = "DWC fast_walsh_hadamard_transform operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: fully_connected_sub_byte_param
def Dwc_FullyConnectedSubByteParamOp : Dwc_Op<"fully_connected_sub_byte_param"> {
  let summary = "DWC fully_connected_sub_byte_param operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: fully_connected_sub_channel
def Dwc_FullyConnectedSubChannelOp : Dwc_Op<"fully_connected_sub_channel"> {
  let summary = "DWC fully_connected_sub_channel operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: functional_if
def Dwc_FunctionalIfOp : Dwc_Op<"functional_if"> {
  let summary = "DWC functional_if operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: functional_while
def Dwc_FunctionalWhileOp : Dwc_Op<"functional_while"> {
  let summary = "DWC functional_while operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: gather_operation
def Dwc_GatherOperationOp : Dwc_Op<"gather_operation"> {
  let summary = "DWC gather_operation operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: generic_move
def Dwc_GenericMoveOp : Dwc_Op<"generic_move"> {
  let summary = "DWC generic_move operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: generic_scatter
def Dwc_GenericScatterOp : Dwc_Op<"generic_scatter"> {
  let summary = "DWC generic_scatter operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: hib_gather
def Dwc_HibGatherOp : Dwc_Op<"hib_gather"> {
  let summary = "DWC hib_gather operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: hib_gather_filter
def Dwc_HibGatherFilterOp : Dwc_Op<"hib_gather_filter"> {
  let summary = "DWC hib_gather_filter operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: hosted_tensor
def Dwc_HostedTensorOp : Dwc_Op<"hosted_tensor"> {
  let summary = "DWC hosted_tensor operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: image_interpolation
def Dwc_ImageInterpolationOp : Dwc_Op<"image_interpolation"> {
  let summary = "DWC image_interpolation operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: index_unpool
def Dwc_IndexUnpoolOp : Dwc_Op<"index_unpool"> {
  let summary = "DWC index_unpool operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: interleave
def Dwc_InterleaveOp : Dwc_Op<"interleave"> {
  let summary = "DWC interleave operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: is_finite
def Dwc_IsFiniteOp : Dwc_Op<"is_finite"> {
  let summary = "DWC is_finite operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: mask_indices
def Dwc_MaskIndicesOp : Dwc_Op<"mask_indices"> {
  let summary = "DWC mask_indices operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: matrix_multiply_sub_channel
def Dwc_MatrixMultiplySubChannelOp : Dwc_Op<"matrix_multiply_sub_channel"> {
  let summary = "DWC matrix_multiply_sub_channel operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: multinomial
def Dwc_MultinomialOp : Dwc_Op<"multinomial"> {
  let summary = "DWC multinomial operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: negate
def Dwc_NegateOp : Dwc_Op<"negate"> {
  let summary = "DWC negate operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: normalization
def Dwc_NormalizationOp : Dwc_Op<"normalization"> {
  let summary = "DWC normalization operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: one_hot_tpu
def Dwc_OneHotTpuOp : Dwc_Op<"one_hot_tpu"> {
  let summary = "DWC one_hot_tpu operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: pack_bits
def Dwc_PackBitsOp : Dwc_Op<"pack_bits"> {
  let summary = "DWC pack_bits operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: pooling
def Dwc_PoolingOp : Dwc_Op<"pooling"> {
  let summary = "DWC pooling operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: probe_subtensor
def Dwc_ProbeSubtensorOp : Dwc_Op<"probe_subtensor"> {
  let summary = "DWC probe_subtensor operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: reduce_precision
def Dwc_ReducePrecisionOp : Dwc_Op<"reduce_precision"> {
  let summary = "DWC reduce_precision operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: reduce_window
def Dwc_ReduceWindowOp : Dwc_Op<"reduce_window"> {
  let summary = "DWC reduce_window operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: resampler
def Dwc_ResamplerOp : Dwc_Op<"resampler"> {
  let summary = "DWC resampler operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: reverse
def Dwc_ReverseOp : Dwc_Op<"reverse"> {
  let summary = "DWC reverse operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: rkhy_add_pool
def Dwc_RkhyAddPoolOp : Dwc_Op<"rkhy_add_pool"> {
  let summary = "DWC rkhy_add_pool operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: rkhy_conv_d2s
def Dwc_RkhyConvD2sOp : Dwc_Op<"rkhy_conv_d2s"> {
  let summary = "DWC rkhy_conv_d2s operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: rkhy_custom_padding
def Dwc_RkhyCustomPaddingOp : Dwc_Op<"rkhy_custom_padding"> {
  let summary = "DWC rkhy_custom_padding operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: rkhy_fused_conv
def Dwc_RkhyFusedConvOp : Dwc_Op<"rkhy_fused_conv"> {
  let summary = "DWC rkhy_fused_conv operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: rkhy_fused_norm
def Dwc_RkhyFusedNormOp : Dwc_Op<"rkhy_fused_norm"> {
  let summary = "DWC rkhy_fused_norm operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: roll
def Dwc_RollOp : Dwc_Op<"roll"> {
  let summary = "DWC roll operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: scalar_core_constant
def Dwc_ScalarCoreConstantOp : Dwc_Op<"scalar_core_constant"> {
  let summary = "DWC scalar_core_constant operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: scatter_operation
def Dwc_ScatterOperationOp : Dwc_Op<"scatter_operation"> {
  let summary = "DWC scatter_operation operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: shift_left
def Dwc_ShiftLeftOp : Dwc_Op<"shift_left"> {
  let summary = "DWC shift_left operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: shift_right_logical
def Dwc_ShiftRightLogicalOp : Dwc_Op<"shift_right_logical"> {
  let summary = "DWC shift_right_logical operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: space_to_batch
def Dwc_SpaceToBatchOp : Dwc_Op<"space_to_batch"> {
  let summary = "DWC space_to_batch operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: space_to_depth
def Dwc_SpaceToDepthOp : Dwc_Op<"space_to_depth"> {
  let summary = "DWC space_to_depth operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: sparse_fully_connected_sub_byte_param
def Dwc_SparseFullyConnectedSubByteParamOp : Dwc_Op<"sparse_fully_connected_sub_byte_param"> {
  let summary = "DWC sparse_fully_connected_sub_byte_param operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: sparse_parameter
def Dwc_SparseParameterOp : Dwc_Op<"sparse_parameter"> {
  let summary = "DWC sparse_parameter operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: spill
def Dwc_SpillOp : Dwc_Op<"spill"> {
  let summary = "DWC spill operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: statistical_top_k
def Dwc_StatisticalTopKOp : Dwc_Op<"statistical_top_k"> {
  let summary = "DWC statistical_top_k operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: tensor_ls_gather
def Dwc_TensorLsGatherOp : Dwc_Op<"tensor_ls_gather"> {
  let summary = "DWC tensor_ls_gather operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: tensor_ls_scatter
def Dwc_TensorLsScatterOp : Dwc_Op<"tensor_ls_scatter"> {
  let summary = "DWC tensor_ls_scatter operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: tensor_ls_scatter_operation
def Dwc_TensorLsScatterOperationOp : Dwc_Op<"tensor_ls_scatter_operation"> {
  let summary = "DWC tensor_ls_scatter_operation operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: tensor_op_gather
def Dwc_TensorOpGatherOp : Dwc_Op<"tensor_op_gather"> {
  let summary = "DWC tensor_op_gather operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: top_k
def Dwc_TopKOp : Dwc_Op<"top_k"> {
  let summary = "DWC top_k operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: transposed_convolution_sub_channel
def Dwc_TransposedConvolutionSubChannelOp : Dwc_Op<"transposed_convolution_sub_channel"> {
  let summary = "DWC transposed_convolution_sub_channel operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: truncate_floats
def Dwc_TruncateFloatsOp : Dwc_Op<"truncate_floats"> {
  let summary = "DWC truncate_floats operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: uniform_random_number_generation
def Dwc_UniformRandomNumberGenerationOp : Dwc_Op<"uniform_random_number_generation"> {
  let summary = "DWC uniform_random_number_generation operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: atan2
def Dwc_Atan2Op : Dwc_Op<"atan2"> {
  let summary = "DWC atan2 operation.";
  let description = [{
    Correct lowering for atan2(+/-0, +/-0) is flag-gated per the blob pass description. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: gather_nd
def Dwc_GatherNdOp : Dwc_Op<"gather_nd"> {
  let summary = "DWC gather_nd operation.";
  let description = [{
    TPU tile optimizer handles dwc.gather and dwc.gather_nd per the blob pass description. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: annotate_materialize_policy
def Dwc_AnnotateMaterializePolicyOp : Dwc_Op<"annotate_materialize_policy"> {
  let summary = "DWC annotate_materialize_policy operation.";
  let description = [{
    Blocks are optimized, inlined into the parent region, and adjusted by cloning neighbor rescaling ops per the blob pass descriptions. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: activation_function
def Dwc_ActivationFunctionOp : Dwc_Op<"activation_function"> {
  let summary = "DWC activation_function operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: additional_input_output
def Dwc_AdditionalInputOutputOp : Dwc_Op<"additional_input_output"> {
  let summary = "DWC additional_input_output operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: af
def Dwc_AfOp : Dwc_Op<"af"> {
  let summary = "DWC af operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: algorithm
def Dwc_AlgorithmOp : Dwc_Op<"algorithm"> {
  let summary = "DWC algorithm operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: and
def Dwc_AndOp : Dwc_Op<"and"> {
  let summary = "DWC and operation.";
  let description = [{
    Takes any number of operands. Op token with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: argument_copy
def Dwc_ArgumentCopyOp : Dwc_Op<"argument_copy"> {
  let summary = "DWC argument_copy operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: bias_parameter
def Dwc_BiasParameterOp : Dwc_Op<"bias_parameter"> {
  let summary = "DWC bias_parameter operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: bit_select_type
def Dwc_BitSelectTypeOp : Dwc_Op<"bit_select_type"> {
  let summary = "DWC bit_select_type operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: branch
def Dwc_BranchOp : Dwc_Op<"branch"> {
  let summary = "DWC branch operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: cell
def Dwc_CellOp : Dwc_Op<"cell"> {
  let summary = "DWC cell operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: classification_type
def Dwc_ClassificationTypeOp : Dwc_Op<"classification_type"> {
  let summary = "DWC classification_type operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: codegen
def Dwc_CodegenOp : Dwc_Op<"codegen"> {
  let summary = "DWC codegen operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: comparison_type
def Dwc_ComparisonTypeOp : Dwc_Op<"comparison_type"> {
  let summary = "DWC comparison_type operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: compilation_unit
def Dwc_CompilationUnitOp : Dwc_Op<"compilation_unit"> {
  let summary = "DWC compilation_unit operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: compilation_unit_wrapper
def Dwc_CompilationUnitWrapperOp : Dwc_Op<"compilation_unit_wrapper"> {
  let summary = "DWC compilation_unit_wrapper operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: compression_mode
def Dwc_CompressionModeOp : Dwc_Op<"compression_mode"> {
  let summary = "DWC compression_mode operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: convolution_3d
def Dwc_Convolution3dOp : Dwc_Op<"convolution_3d"> {
  let summary = "DWC convolution_3d operation.";
  let description = [{
    Takes any number of operands. Op token with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: cumulative_type
def Dwc_CumulativeTypeOp : Dwc_Op<"cumulative_type"> {
  let summary = "DWC cumulative_type operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: custom_padding_value_type
def Dwc_CustomPaddingValueTypeOp : Dwc_Op<"custom_padding_value_type"> {
  let summary = "DWC custom_padding_value_type operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: cwise_type
def Dwc_CwiseTypeOp : Dwc_Op<"cwise_type"> {
  let summary = "DWC cwise_type operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: declare_tensor_static
def Dwc_DeclareTensorStaticOp : Dwc_Op<"declare_tensor_static"> {
  let summary = "DWC declare_tensor_static operation.";
  let description = [{
    Takes any number of operands. Op token with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: device_launch_func
def Dwc_DeviceLaunchFuncOp : Dwc_Op<"device_launch_func"> {
  let summary = "DWC device_launch_func operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: device_type
def Dwc_DeviceTypeOp : Dwc_Op<"device_type"> {
  let summary = "DWC device_type operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: dim_mapping
def Dwc_DimMappingOp : Dwc_Op<"dim_mapping"> {
  let summary = "DWC dim_mapping operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: dimension_layout
def Dwc_DimensionLayoutOp : Dwc_Op<"dimension_layout"> {
  let summary = "DWC dimension_layout operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: dive_unroll_factor
def Dwc_DiveUnrollFactorOp : Dwc_Op<"dive_unroll_factor"> {
  let summary = "DWC dive_unroll_factor operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: dynamic_intermediate_input_shard
def Dwc_DynamicIntermediateInputShardOp : Dwc_Op<"dynamic_intermediate_input_shard"> {
  let summary = "DWC dynamic_intermediate_input_shard operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: dynamic_intermediate_output_shard
def Dwc_DynamicIntermediateOutputShardOp : Dwc_Op<"dynamic_intermediate_output_shard"> {
  let summary = "DWC dynamic_intermediate_output_shard operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: dynamic_slice_nd
def Dwc_DynamicSliceNdOp : Dwc_Op<"dynamic_slice_nd"> {
  let summary = "DWC dynamic_slice_nd operation.";
  let description = [{
    Takes any number of operands. Op token with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: dynamic_slice_nd_v2
def Dwc_DynamicSliceNdV2Op : Dwc_Op<"dynamic_slice_nd_v2"> {
  let summary = "DWC dynamic_slice_nd_v2 operation.";
  let description = [{
    Takes any number of operands. Op token with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: dynamic_update_slice_nd
def Dwc_DynamicUpdateSliceNdOp : Dwc_Op<"dynamic_update_slice_nd"> {
  let summary = "DWC dynamic_update_slice_nd operation.";
  let description = [{
    Takes any number of operands. Op token with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: dynamic_update_slice_nd_v2
def Dwc_DynamicUpdateSliceNdV2Op : Dwc_Op<"dynamic_update_slice_nd_v2"> {
  let summary = "DWC dynamic_update_slice_nd_v2 operation.";
  let description = [{
    Takes any number of operands. Op token with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: encoding
def Dwc_EncodingOp : Dwc_Op<"encoding"> {
  let summary = "DWC encoding operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: engine
def Dwc_EngineOp : Dwc_Op<"engine"> {
  let summary = "DWC engine operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: fetch
def Dwc_FetchOp : Dwc_Op<"fetch"> {
  let summary = "DWC fetch operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: fully_connected_sub_channel_v2
def Dwc_FullyConnectedSubChannelV2Op : Dwc_Op<"fully_connected_sub_channel_v2"> {
  let summary = "DWC fully_connected_sub_channel_v2 operation.";
  let description = [{
    Takes any number of operands. Op token with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: fully_connected_zin_indexed
def Dwc_FullyConnectedZinIndexedOp : Dwc_Op<"fully_connected_zin_indexed"> {
  let summary = "DWC fully_connected_zin_indexed operation.";
  let description = [{
    Takes any number of operands. Op token with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: fully_connected_zout_indexed
def Dwc_FullyConnectedZoutIndexedOp : Dwc_Op<"fully_connected_zout_indexed"> {
  let summary = "DWC fully_connected_zout_indexed operation.";
  let description = [{
    Takes any number of operands. Op token with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: generic_pad
def Dwc_GenericPadOp : Dwc_Op<"generic_pad"> {
  let summary = "DWC generic_pad operation.";
  let description = [{
    Takes any number of operands. Op token with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: hardware_cluster_id_per_signature
def Dwc_HardwareClusterIdPerSignatureOp : Dwc_Op<"hardware_cluster_id_per_signature"> {
  let summary = "DWC hardware_cluster_id_per_signature operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: host_space
def Dwc_HostSpaceOp : Dwc_Op<"host_space"> {
  let summary = "DWC host_space operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: identity
def Dwc_IdentityOp : Dwc_Op<"identity"> {
  let summary = "DWC identity operation.";
  let description = [{
    Takes any number of operands. Op token with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: if
def Dwc_IfOp : Dwc_Op<"if"> {
  let summary = "DWC if operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: image_format
def Dwc_ImageFormatOp : Dwc_Op<"image_format"> {
  let summary = "DWC image_format operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: input
def Dwc_InputOp : Dwc_Op<"input"> {
  let summary = "DWC input operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: input_shard
def Dwc_InputShardOp : Dwc_Op<"input_shard"> {
  let summary = "DWC input_shard operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: inter_die_input
def Dwc_InterDieInputOp : Dwc_Op<"inter_die_input"> {
  let summary = "DWC inter_die_input operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: inter_die_output
def Dwc_InterDieOutputOp : Dwc_Op<"inter_die_output"> {
  let summary = "DWC inter_die_output operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: intermediate_input
def Dwc_IntermediateInputOp : Dwc_Op<"intermediate_input"> {
  let summary = "DWC intermediate_input operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: intermediate_input_shard
def Dwc_IntermediateInputShardOp : Dwc_Op<"intermediate_input_shard"> {
  let summary = "DWC intermediate_input_shard operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: intermediate_output
def Dwc_IntermediateOutputOp : Dwc_Op<"intermediate_output"> {
  let summary = "DWC intermediate_output operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: intermediate_output_shard
def Dwc_IntermediateOutputShardOp : Dwc_Op<"intermediate_output_shard"> {
  let summary = "DWC intermediate_output_shard operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: internal_padding
def Dwc_InternalPaddingOp : Dwc_Op<"internal_padding"> {
  let summary = "DWC internal_padding operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: io_colocation_pairs
def Dwc_IoColocationPairsOp : Dwc_Op<"io_colocation_pairs"> {
  let summary = "DWC io_colocation_pairs operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: is_external_parameter
def Dwc_IsExternalParameterOp : Dwc_Op<"is_external_parameter"> {
  let summary = "DWC is_external_parameter operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: jump
def Dwc_JumpOp : Dwc_Op<"jump"> {
  let summary = "DWC jump operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: kernel_level
def Dwc_KernelLevelOp : Dwc_Op<"kernel_level"> {
  let summary = "DWC kernel_level operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: known_trip_count
def Dwc_KnownTripCountOp : Dwc_Op<"known_trip_count"> {
  let summary = "DWC known_trip_count operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: launch_custom_kernel
def Dwc_LaunchCustomKernelOp : Dwc_Op<"launch_custom_kernel"> {
  let summary = "DWC launch_custom_kernel operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: linear_function
def Dwc_LinearFunctionOp : Dwc_Op<"linear_function"> {
  let summary = "DWC linear_function operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: log
def Dwc_LogOp : Dwc_Op<"log"> {
  let summary = "DWC log operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: loop_sharding
def Dwc_LoopShardingOp : Dwc_Op<"loop_sharding"> {
  let summary = "DWC loop_sharding operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: lower_bound
def Dwc_LowerBoundOp : Dwc_Op<"lower_bound"> {
  let summary = "DWC lower_bound operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: materialize_policy
def Dwc_MaterializePolicyOp : Dwc_Op<"materialize_policy"> {
  let summary = "DWC materialize_policy operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: memory_location
def Dwc_MemoryLocationOp : Dwc_Op<"memory_location"> {
  let summary = "DWC memory_location operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: memory_space
def Dwc_MemorySpaceOp : Dwc_Op<"memory_space"> {
  let summary = "DWC memory_space operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: mesh_dim
def Dwc_MeshDimOp : Dwc_Op<"mesh_dim"> {
  let summary = "DWC mesh_dim operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: mlir
def Dwc_MlirOp : Dwc_Op<"mlir"> {
  let summary = "DWC mlir operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: multimedia
def Dwc_MultimediaOp : Dwc_Op<"multimedia"> {
  let summary = "DWC multimedia operation.";
  let description = [{
    Takes any number of operands. Op token with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: nlu_e8m0_rounding
def Dwc_NluE8m0RoundingOp : Dwc_Op<"nlu_e8m0_rounding"> {
  let summary = "DWC nlu_e8m0_rounding operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: nlu_preprocess
def Dwc_NluPreprocessOp : Dwc_Op<"nlu_preprocess"> {
  let summary = "DWC nlu_preprocess operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: normalization_type
def Dwc_NormalizationTypeOp : Dwc_Op<"normalization_type"> {
  let summary = "DWC normalization_type operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: or
def Dwc_OrOp : Dwc_Op<"or"> {
  let summary = "DWC or operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: output
def Dwc_OutputOp : Dwc_Op<"output"> {
  let summary = "DWC output operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: output_shard
def Dwc_OutputShardOp : Dwc_Op<"output_shard"> {
  let summary = "DWC output_shard operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: pad
def Dwc_PadOp : Dwc_Op<"pad"> {
  let summary = "DWC pad operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: padding_value_type
def Dwc_PaddingValueTypeOp : Dwc_Op<"padding_value_type"> {
  let summary = "DWC padding_value_type operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: parameter
def Dwc_ParameterOp : Dwc_Op<"parameter"> {
  let summary = "DWC parameter operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: parameter_lookup_table
def Dwc_ParameterLookupTableOp : Dwc_Op<"parameter_lookup_table"> {
  let summary = "DWC parameter_lookup_table operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: per_z_out_scale_padding
def Dwc_PerZOutScalePaddingOp : Dwc_Op<"per_z_out_scale_padding"> {
  let summary = "DWC per_z_out_scale_padding operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: pool
def Dwc_PoolOp : Dwc_Op<"pool"> {
  let summary = "DWC pool operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: pooling_3d
def Dwc_Pooling3dOp : Dwc_Op<"pooling_3d"> {
  let summary = "DWC pooling_3d operation.";
  let description = [{
    Takes any number of operands. Op token with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: probe
def Dwc_ProbeOp : Dwc_Op<"probe"> {
  let summary = "DWC probe operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: pseudo_dynamic_image_interpolation
def Dwc_PseudoDynamicImageInterpolationOp : Dwc_Op<"pseudo_dynamic_image_interpolation"> {
  let summary = "DWC pseudo_dynamic_image_interpolation operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: pseudo_dynamic_pad
def Dwc_PseudoDynamicPadOp : Dwc_Op<"pseudo_dynamic_pad"> {
  let summary = "DWC pseudo_dynamic_pad operation.";
  let description = [{
    Takes any number of operands. Op token with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: pseudo_dynamic_reshape
def Dwc_PseudoDynamicReshapeOp : Dwc_Op<"pseudo_dynamic_reshape"> {
  let summary = "DWC pseudo_dynamic_reshape operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: pseudo_dynamic_slice
def Dwc_PseudoDynamicSliceOp : Dwc_Op<"pseudo_dynamic_slice"> {
  let summary = "DWC pseudo_dynamic_slice operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: pseudo_expand_dims
def Dwc_PseudoExpandDimsOp : Dwc_Op<"pseudo_expand_dims"> {
  let summary = "DWC pseudo_expand_dims operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: pseudo_fill
def Dwc_PseudoFillOp : Dwc_Op<"pseudo_fill"> {
  let summary = "DWC pseudo_fill operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: pseudo_generic_norm
def Dwc_PseudoGenericNormOp : Dwc_Op<"pseudo_generic_norm"> {
  let summary = "DWC pseudo_generic_norm operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: pseudo_group_norm
def Dwc_PseudoGroupNormOp : Dwc_Op<"pseudo_group_norm"> {
  let summary = "DWC pseudo_group_norm operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: pseudo_mirror_pad
def Dwc_PseudoMirrorPadOp : Dwc_Op<"pseudo_mirror_pad"> {
  let summary = "DWC pseudo_mirror_pad operation.";
  let description = [{
    Takes any number of operands. Op token with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: pseudo_range
def Dwc_PseudoRangeOp : Dwc_Op<"pseudo_range"> {
  let summary = "DWC pseudo_range operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: pseudo_shape
def Dwc_PseudoShapeOp : Dwc_Op<"pseudo_shape"> {
  let summary = "DWC pseudo_shape operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: pseudo_squeeze
def Dwc_PseudoSqueezeOp : Dwc_Op<"pseudo_squeeze"> {
  let summary = "DWC pseudo_squeeze operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: reduce_window_type
def Dwc_ReduceWindowTypeOp : Dwc_Op<"reduce_window_type"> {
  let summary = "DWC reduce_window_type operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: reduction_type
def Dwc_ReductionTypeOp : Dwc_Op<"reduction_type"> {
  let summary = "DWC reduction_type operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: resampler_options
def Dwc_ResamplerOptionsOp : Dwc_Op<"resampler_options"> {
  let summary = "DWC resampler_options operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: rkhy_add
def Dwc_RkhyAddOp : Dwc_Op<"rkhy_add"> {
  let summary = "DWC rkhy_add operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: scalar_type
def Dwc_ScalarTypeOp : Dwc_Op<"scalar_type"> {
  let summary = "DWC scalar_type operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: shard_barrier
def Dwc_ShardBarrierOp : Dwc_Op<"shard_barrier"> {
  let summary = "DWC shard_barrier operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: shard_body
def Dwc_ShardBodyOp : Dwc_Op<"shard_body"> {
  let summary = "DWC shard_body operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: shard_group
def Dwc_ShardGroupOp : Dwc_Op<"shard_group"> {
  let summary = "DWC shard_group operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: shard_sink
def Dwc_ShardSinkOp : Dwc_Op<"shard_sink"> {
  let summary = "DWC shard_sink operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: shard_source
def Dwc_ShardSourceOp : Dwc_Op<"shard_source"> {
  let summary = "DWC shard_source operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: shift_right_arithmetic
def Dwc_ShiftRightArithmeticOp : Dwc_Op<"shift_right_arithmetic"> {
  let summary = "DWC shift_right_arithmetic operation.";
  let description = [{
    Takes any number of operands. Op token with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: signature_na
def Dwc_SignatureNaOp : Dwc_Op<"signature_na"> {
  let summary = "DWC signature_na operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: signature_name
def Dwc_SignatureNameOp : Dwc_Op<"signature_name"> {
  let summary = "DWC signature_name operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: sparse_fully_connected
def Dwc_SparseFullyConnectedOp : Dwc_Op<"sparse_fully_connected"> {
  let summary = "DWC sparse_fully_connected operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: spill_location
def Dwc_SpillLocationOp : Dwc_Op<"spill_location"> {
  let summary = "DWC spill_location operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: stabletg_kernel
def Dwc_StabletgKernelOp : Dwc_Op<"stabletg_kernel"> {
  let summary = "DWC stabletg_kernel operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: stride_method
def Dwc_StrideMethodOp : Dwc_Op<"stride_method"> {
  let summary = "DWC stride_method operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: tile_mesh
def Dwc_TileMeshOp : Dwc_Op<"tile_mesh"> {
  let summary = "DWC tile_mesh operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: tile_uid
def Dwc_TileUidOp : Dwc_Op<"tile_uid"> {
  let summary = "DWC tile_uid operation.";
  let description = [{
    Takes any number of operands. Op token with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: tpu_group_id
def Dwc_TpuGroupIdOp : Dwc_Op<"tpu_group_id"> {
  let summary = "DWC tpu_group_id operation.";
  let description = [{
    Takes any number of operands. Op token with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: transformation_type
def Dwc_TransformationTypeOp : Dwc_Op<"transformation_type"> {
  let summary = "DWC transformation_type operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: visible_tiles_per_signature
def Dwc_VisibleTilesPerSignatureOp : Dwc_Op<"visible_tiles_per_signature"> {
  let summary = "DWC visible_tiles_per_signature operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: vrgkh_operation_mode
def Dwc_VrgkhOperationModeOp : Dwc_Op<"vrgkh_operation_mode"> {
  let summary = "DWC vrgkh_operation_mode operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: vtid
def Dwc_VtidOp : Dwc_Op<"vtid"> {
  let summary = "DWC vtid operation.";
  let description = [{
    Takes any number of operands. Op token with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: while
def Dwc_WhileOp : Dwc_Op<"while"> {
  let summary = "DWC while operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: xor
def Dwc_XorOp : Dwc_Op<"xor"> {
  let summary = "DWC xor operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: yield
def Dwc_YieldOp : Dwc_Op<"yield"> {
  let summary = "DWC yield operation.";
  let description = [{
    Takes any number of operands. Op token with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

#endif // DWC_OPS