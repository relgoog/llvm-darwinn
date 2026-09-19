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
    Exactly 1 operand per blob operand constraints.
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
    Exactly 1 operand per blob operand constraints.
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
    Exactly 1 operand per blob operand constraints.
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
    Exactly 1 operand per blob operand constraints.
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
    Exactly 1 operand per blob operand constraints. Requires `axis`, `exclusive`, `op_type`.
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
    Exactly 2 operands per blob operand constraints. Requires `activation_function`, `op_type`.
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
    Exactly 1 operand per blob operand constraints.
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
    Exactly 1 operand per blob operand constraints.
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
    Exactly 1 operand per blob operand constraints.
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
    Exactly 1 operand per blob operand constraints.
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
    Exactly 2 operands per blob operand constraints.
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
    Exactly 1 operand per blob operand constraints.
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
    Exactly 1 operand per blob operand constraints.
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
    Exactly 1 operand per blob operand constraints.
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
    Exactly 1 operand per blob operand constraints. Requires `axis`.
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
    Exactly 1 operand per blob operand constraints. Requires `dimension`, `padding_value`, `post_padding`, `pre_padding`.
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
    Exactly 1 operand per blob operand constraints.
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
    Exactly 2 operands per blob operand constraints.
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
    Exactly 2 operands per blob operand constraints. Requires `num_splits`.
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
    Exactly 1 operand per blob operand constraints. Requires `activation_function`, `dimensions`, `op_type`.
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
    Exactly 1 operand per blob operand constraints. Requires `activation_function`, `output_activation_per_z_out_scales`, `per_z_out_scales_padding`.
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
    Exactly 1 operand per blob operand constraints.
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
    Exactly 1 operand per blob operand constraints.
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
    Exactly 1 operand per blob operand constraints.
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
    Exactly 1 operand per blob operand constraints.
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
    Exactly 2 operands per blob operand constraints.
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
    Exactly 1 operand per blob operand constraints. Requires `preserve_negative_zero`.
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
    Exactly 1 operand per blob operand constraints.
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
    Exactly 1 operand per blob operand constraints.
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
    Exactly 1 operand per blob operand constraints.
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
    Exactly 1 operand per blob operand constraints.
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
    Exactly 1 operand per blob operand constraints. Requires `permutation`.
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

#endif // DWC_OPS