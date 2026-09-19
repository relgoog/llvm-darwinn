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

#endif // DWC_OPS