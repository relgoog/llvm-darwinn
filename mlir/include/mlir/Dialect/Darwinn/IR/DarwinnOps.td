//===-- DarwinnOps.td - Darwinn operations ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Boilerplate operation set for the Darwinn dialect, one op per mnemonic.
//
//===----------------------------------------------------------------------===//

#ifndef DARWINN_OPS
#define DARWINN_OPS

include "mlir/IR/OpBase.td"

include "mlir/Dialect/Darwinn/IR/DarwinnDialect.td"

// Operator: aux_tensor_type
def Darwinn_AuxTensorTypeOp : Darwinn_Op<"aux_tensor_type"> {
  let summary = "Darwinn aux_tensor_type operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.aux_tensor_type operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: binary_map
def Darwinn_BinaryMapOp : Darwinn_Op<"binary_map"> {
  let summary = "Darwinn binary_map operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.binary_map operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: bitcast
def Darwinn_BitcastOp : Darwinn_Op<"bitcast"> {
  let summary = "Darwinn bitcast operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.bitcast operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: broadcast_shard
def Darwinn_BroadcastShardOp : Darwinn_Op<"broadcast_shard"> {
  let summary = "Darwinn broadcast_shard operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.broadcast_shard operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: broadcast_slice
def Darwinn_BroadcastSliceOp : Darwinn_Op<"broadcast_slice"> {
  let summary = "Darwinn broadcast_slice operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.broadcast_slice operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: cast_in
def Darwinn_CastInOp : Darwinn_Op<"cast_in"> {
  let summary = "Darwinn cast_in operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.cast_in operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: cast_out
def Darwinn_CastOutOp : Darwinn_Op<"cast_out"> {
  let summary = "Darwinn cast_out operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.cast_out operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: ceppsyt_decompress
def Darwinn_CeppsytDecompressOp : Darwinn_Op<"ceppsyt_decompress"> {
  let summary = "Darwinn ceppsyt_decompress operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.ceppsyt_decompress operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: ceppsyt_hwsvgmllkwt
def Darwinn_CeppsytHwsvgmllkwtOp : Darwinn_Op<"ceppsyt_hwsvgmllkwt"> {
  let summary = "Darwinn ceppsyt_hwsvgmllkwt operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.ceppsyt_hwsvgmllkwt operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: chunking_reshape
def Darwinn_ChunkingReshapeOp : Darwinn_Op<"chunking_reshape"> {
  let summary = "Darwinn chunking_reshape operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.chunking_reshape operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: compression_mode
def Darwinn_CompressionModeOp : Darwinn_Op<"compression_mode"> {
  let summary = "Darwinn compression_mode operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.compression_mode operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: compute_lowering_hint
def Darwinn_ComputeLoweringHintOp : Darwinn_Op<"compute_lowering_hint"> {
  let summary = "Darwinn compute_lowering_hint operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.compute_lowering_hint operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: compute_op_options
def Darwinn_ComputeOpOptionsOp : Darwinn_Op<"compute_op_options"> {
  let summary = "Darwinn compute_op_options operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.compute_op_options operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: compute_type_hint
def Darwinn_ComputeTypeHintOp : Darwinn_Op<"compute_type_hint"> {
  let summary = "Darwinn compute_type_hint operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.compute_type_hint operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: condition_scope
def Darwinn_ConditionScopeOp : Darwinn_Op<"condition_scope"> {
  let summary = "Darwinn condition_scope operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.condition_scope operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: const_bias_scale
def Darwinn_ConstBiasScaleOp : Darwinn_Op<"const_bias_scale"> {
  let summary = "Darwinn const_bias_scale operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.const_bias_scale operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: const_type
def Darwinn_ConstTypeOp : Darwinn_Op<"const_type"> {
  let summary = "Darwinn const_type operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.const_type operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: constant_generator
def Darwinn_ConstantGeneratorOp : Darwinn_Op<"constant_generator"> {
  let summary = "Darwinn constant_generator operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.constant_generator operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: convert
def Darwinn_ConvertOp : Darwinn_Op<"convert"> {
  let summary = "Darwinn convert operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.convert operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: convolution
def Darwinn_ConvolutionOp : Darwinn_Op<"convolution"> {
  let summary = "Darwinn convolution operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.convolution operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: copy_from_host
def Darwinn_CopyFromHostOp : Darwinn_Op<"copy_from_host"> {
  let summary = "Darwinn copy_from_host operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.copy_from_host operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: copy_op
def Darwinn_CopyOpOp : Darwinn_Op<"copy_op"> {
  let summary = "Darwinn copy_op operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.copy_op operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: copy_using_wide
def Darwinn_CopyUsingWideOp : Darwinn_Op<"copy_using_wide"> {
  let summary = "Darwinn copy_using_wide operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.copy_using_wide operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: cost_hint
def Darwinn_CostHintOp : Darwinn_Op<"cost_hint"> {
  let summary = "Darwinn cost_hint operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.cost_hint operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: create_empty_tensor
def Darwinn_CreateEmptyTensorOp : Darwinn_Op<"create_empty_tensor"> {
  let summary = "Darwinn create_empty_tensor operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.create_empty_tensor operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: custom_tiling_options
def Darwinn_CustomTilingOptionsOp : Darwinn_Op<"custom_tiling_options"> {
  let summary = "Darwinn custom_tiling_options operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.custom_tiling_options operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: device_type
def Darwinn_DeviceTypeOp : Darwinn_Op<"device_type"> {
  let summary = "Darwinn device_type operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.device_type operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: dive_op
def Darwinn_DiveOpOp : Darwinn_Op<"dive_op"> {
  let summary = "Darwinn dive_op operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.dive_op operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: dive_ref_cwise
def Darwinn_DiveRefCwiseOp : Darwinn_Op<"dive_ref_cwise"> {
  let summary = "Darwinn dive_ref_cwise operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.dive_ref_cwise operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: dive_ref_reduction
def Darwinn_DiveRefReductionOp : Darwinn_Op<"dive_ref_reduction"> {
  let summary = "Darwinn dive_ref_reduction operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.dive_ref_reduction operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: dtc_info
def Darwinn_DtcInfoOp : Darwinn_Op<"dtc_info"> {
  let summary = "Darwinn dtc_info operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.dtc_info operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: dynamic_slice
def Darwinn_DynamicSliceOp : Darwinn_Op<"dynamic_slice"> {
  let summary = "Darwinn dynamic_slice operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.dynamic_slice operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: dynamic_slice_with_copy
def Darwinn_DynamicSliceWithCopyOp : Darwinn_Op<"dynamic_slice_with_copy"> {
  let summary = "Darwinn dynamic_slice_with_copy operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.dynamic_slice_with_copy operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: dynamic_update_slice
def Darwinn_DynamicUpdateSliceOp : Darwinn_Op<"dynamic_update_slice"> {
  let summary = "Darwinn dynamic_update_slice operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.dynamic_update_slice operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: dynamic_update_slice_with_offsets
def Darwinn_DynamicUpdateSliceWithOffsetsOp : Darwinn_Op<"dynamic_update_slice_with_offsets"> {
  let summary = "Darwinn dynamic_update_slice_with_offsets operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.dynamic_update_slice_with_offsets operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: fast_walsh_hadamard_transform
def Darwinn_FastWalshHadamardTransformOp : Darwinn_Op<"fast_walsh_hadamard_transform"> {
  let summary = "Darwinn fast_walsh_hadamard_transform operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.fast_walsh_hadamard_transform operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: fence
def Darwinn_FenceOp : Darwinn_Op<"fence"> {
  let summary = "Darwinn fence operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.fence operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: fill
def Darwinn_FillOp : Darwinn_Op<"fill"> {
  let summary = "Darwinn fill operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.fill operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: filter
def Darwinn_FilterOp : Darwinn_Op<"filter"> {
  let summary = "Darwinn filter operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.filter operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: filter_cmp
def Darwinn_FilterCmpOp : Darwinn_Op<"filter_cmp"> {
  let summary = "Darwinn filter_cmp operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.filter_cmp operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: filter_output
def Darwinn_FilterOutputOp : Darwinn_Op<"filter_output"> {
  let summary = "Darwinn filter_output operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.filter_output operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: filtered_indices_dispatch_mode
def Darwinn_FilteredIndicesDispatchModeOp : Darwinn_Op<"filtered_indices_dispatch_mode"> {
  let summary = "Darwinn filtered_indices_dispatch_mode operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.filtered_indices_dispatch_mode operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: fully_connected_zout_indexed
def Darwinn_FullyConnectedZoutIndexedOp : Darwinn_Op<"fully_connected_zout_indexed"> {
  let summary = "Darwinn fully_connected_zout_indexed operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.fully_connected_zout_indexed operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: function_symmetry
def Darwinn_FunctionSymmetryOp : Darwinn_Op<"function_symmetry"> {
  let summary = "Darwinn function_symmetry operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.function_symmetry operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: gather
def Darwinn_GatherOp : Darwinn_Op<"gather"> {
  let summary = "Darwinn gather operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.gather operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: gather_copy
def Darwinn_GatherCopyOp : Darwinn_Op<"gather_copy"> {
  let summary = "Darwinn gather_copy operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.gather_copy operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: get_full_tensor_of_dynamic_view
def Darwinn_GetFullTensorOfDynamicViewOp : Darwinn_Op<"get_full_tensor_of_dynamic_view"> {
  let summary = "Darwinn get_full_tensor_of_dynamic_view operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.get_full_tensor_of_dynamic_view operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: get_indexed_slice
def Darwinn_GetIndexedSliceOp : Darwinn_Op<"get_indexed_slice"> {
  let summary = "Darwinn get_indexed_slice operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.get_indexed_slice operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: get_tensor
def Darwinn_GetTensorOp : Darwinn_Op<"get_tensor"> {
  let summary = "Darwinn get_tensor operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.get_tensor operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: hib_gather
def Darwinn_HibGatherOp : Darwinn_Op<"hib_gather"> {
  let summary = "Darwinn hib_gather operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.hib_gather operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: hl_bitcast
def Darwinn_HlBitcastOp : Darwinn_Op<"hl_bitcast"> {
  let summary = "Darwinn hl_bitcast operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.hl_bitcast operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: host_to_ssram
def Darwinn_HostToSsramOp : Darwinn_Op<"host_to_ssram"> {
  let summary = "Darwinn host_to_ssram operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.host_to_ssram operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: host_to_ssram_shard
def Darwinn_HostToSsramShardOp : Darwinn_Op<"host_to_ssram_shard"> {
  let summary = "Darwinn host_to_ssram_shard operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.host_to_ssram_shard operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: host_to_tile
def Darwinn_HostToTileOp : Darwinn_Op<"host_to_tile"> {
  let summary = "Darwinn host_to_tile operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.host_to_tile operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: host_to_tile_shard
def Darwinn_HostToTileShardOp : Darwinn_Op<"host_to_tile_shard"> {
  let summary = "Darwinn host_to_tile_shard operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.host_to_tile_shard operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: image_format
def Darwinn_ImageFormatOp : Darwinn_Op<"image_format"> {
  let summary = "Darwinn image_format operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.image_format operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: index_filter
def Darwinn_IndexFilterOp : Darwinn_Op<"index_filter"> {
  let summary = "Darwinn index_filter operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.index_filter operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: infeed
def Darwinn_InfeedOp : Darwinn_Op<"infeed"> {
  let summary = "Darwinn infeed operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.infeed operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: inner_op
def Darwinn_InnerOpOp : Darwinn_Op<"inner_op"> {
  let summary = "Darwinn inner_op operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.inner_op operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: interpolate
def Darwinn_InterpolateOp : Darwinn_Op<"interpolate"> {
  let summary = "Darwinn interpolate operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.interpolate operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: interpolate_hardware
def Darwinn_InterpolateHardwareOp : Darwinn_Op<"interpolate_hardware"> {
  let summary = "Darwinn interpolate_hardware operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.interpolate_hardware operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: interpolate_method
def Darwinn_InterpolateMethodOp : Darwinn_Op<"interpolate_method"> {
  let summary = "Darwinn interpolate_method operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.interpolate_method operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: iota
def Darwinn_IotaOp : Darwinn_Op<"iota"> {
  let summary = "Darwinn iota operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.iota operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: is_bool
def Darwinn_IsBoolOp : Darwinn_Op<"is_bool"> {
  let summary = "Darwinn is_bool operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.is_bool operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: is_parameter
def Darwinn_IsParameterOp : Darwinn_Op<"is_parameter"> {
  let summary = "Darwinn is_parameter operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.is_parameter operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: is_sparsely_packed
def Darwinn_IsSparselyPackedOp : Darwinn_Op<"is_sparsely_packed"> {
  let summary = "Darwinn is_sparsely_packed operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.is_sparsely_packed operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: join_attributes
def Darwinn_JoinAttributesOp : Darwinn_Op<"join_attributes"> {
  let summary = "Darwinn join_attributes operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.join_attributes operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: kernel_level
def Darwinn_KernelLevelOp : Darwinn_Op<"kernel_level"> {
  let summary = "Darwinn kernel_level operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.kernel_level operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: launch_custom_kernel
def Darwinn_LaunchCustomKernelOp : Darwinn_Op<"launch_custom_kernel"> {
  let summary = "Darwinn launch_custom_kernel operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.launch_custom_kernel operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: launch_function
def Darwinn_LaunchFunctionOp : Darwinn_Op<"launch_function"> {
  let summary = "Darwinn launch_function operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.launch_function operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: legacy_interpolate
def Darwinn_LegacyInterpolateOp : Darwinn_Op<"legacy_interpolate"> {
  let summary = "Darwinn legacy_interpolate operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.legacy_interpolate operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: linear_func
def Darwinn_LinearFuncOp : Darwinn_Op<"linear_func"> {
  let summary = "Darwinn linear_func operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.linear_func operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: local_copy_attributes
def Darwinn_LocalCopyAttributesOp : Darwinn_Op<"local_copy_attributes"> {
  let summary = "Darwinn local_copy_attributes operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.local_copy_attributes operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: mapping
def Darwinn_MappingOp : Darwinn_Op<"mapping"> {
  let summary = "Darwinn mapping operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.mapping operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: mask_indices
def Darwinn_MaskIndicesOp : Darwinn_Op<"mask_indices"> {
  let summary = "Darwinn mask_indices operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.mask_indices operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: material
def Darwinn_MaterialOp : Darwinn_Op<"material"> {
  let summary = "Darwinn material operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.material operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: materialize_cast
def Darwinn_MaterializeCastOp : Darwinn_Op<"materialize_cast"> {
  let summary = "Darwinn materialize_cast operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.materialize_cast operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: materialize_policy
def Darwinn_MaterializePolicyOp : Darwinn_Op<"materialize_policy"> {
  let summary = "Darwinn materialize_policy operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.materialize_policy operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: mem_space
def Darwinn_MemSpaceOp : Darwinn_Op<"mem_space"> {
  let summary = "Darwinn mem_space operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.mem_space operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: mesh_pad_slice
def Darwinn_MeshPadSliceOp : Darwinn_Op<"mesh_pad_slice"> {
  let summary = "Darwinn mesh_pad_slice operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.mesh_pad_slice operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: mma_compute_op
def Darwinn_MmaComputeOpOp : Darwinn_Op<"mma_compute_op"> {
  let summary = "Darwinn mma_compute_op operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.mma_compute_op operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: multimedia
def Darwinn_MultimediaOp : Darwinn_Op<"multimedia"> {
  let summary = "Darwinn multimedia operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.multimedia operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: narrow_to_narrow
def Darwinn_NarrowToNarrowOp : Darwinn_Op<"narrow_to_narrow"> {
  let summary = "Darwinn narrow_to_narrow operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.narrow_to_narrow operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: narrow_to_narrow_shard
def Darwinn_NarrowToNarrowShardOp : Darwinn_Op<"narrow_to_narrow_shard"> {
  let summary = "Darwinn narrow_to_narrow_shard operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.narrow_to_narrow_shard operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: narrow_to_narrow_slice
def Darwinn_NarrowToNarrowSliceOp : Darwinn_Op<"narrow_to_narrow_slice"> {
  let summary = "Darwinn narrow_to_narrow_slice operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.narrow_to_narrow_slice operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: narrow_to_wide
def Darwinn_NarrowToWideOp : Darwinn_Op<"narrow_to_wide"> {
  let summary = "Darwinn narrow_to_wide operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.narrow_to_wide operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: narrow_to_wide_shard
def Darwinn_NarrowToWideShardOp : Darwinn_Op<"narrow_to_wide_shard"> {
  let summary = "Darwinn narrow_to_wide_shard operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.narrow_to_wide_shard operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: narrow_to_wide_slice
def Darwinn_NarrowToWideSliceOp : Darwinn_Op<"narrow_to_wide_slice"> {
  let summary = "Darwinn narrow_to_wide_slice operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.narrow_to_wide_slice operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: nlu_e8m0_rounding
def Darwinn_NluE8m0RoundingOp : Darwinn_Op<"nlu_e8m0_rounding"> {
  let summary = "Darwinn nlu_e8m0_rounding operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.nlu_e8m0_rounding operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: nlu_func
def Darwinn_NluFuncOp : Darwinn_Op<"nlu_func"> {
  let summary = "Darwinn nlu_func operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.nlu_func operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: nlu_predicate
def Darwinn_NluPredicateOp : Darwinn_Op<"nlu_predicate"> {
  let summary = "Darwinn nlu_predicate operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.nlu_predicate operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: nlu_preprocess
def Darwinn_NluPreprocessOp : Darwinn_Op<"nlu_preprocess"> {
  let summary = "Darwinn nlu_preprocess operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.nlu_preprocess operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: outfeed
def Darwinn_OutfeedOp : Darwinn_Op<"outfeed"> {
  let summary = "Darwinn outfeed operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.outfeed operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: packed_index_options
def Darwinn_PackedIndexOptionsOp : Darwinn_Op<"packed_index_options"> {
  let summary = "Darwinn packed_index_options operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.packed_index_options operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: parallel_mesh_copy
def Darwinn_ParallelMeshCopyOp : Darwinn_Op<"parallel_mesh_copy"> {
  let summary = "Darwinn parallel_mesh_copy operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.parallel_mesh_copy operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: preemption_point
def Darwinn_PreemptionPointOp : Darwinn_Op<"preemption_point"> {
  let summary = "Darwinn preemption_point operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.preemption_point operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: probe
def Darwinn_ProbeOp : Darwinn_Op<"probe"> {
  let summary = "Darwinn probe operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.probe operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: reinterpret_cast
def Darwinn_ReinterpretCastOp : Darwinn_Op<"reinterpret_cast"> {
  let summary = "Darwinn reinterpret_cast operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.reinterpret_cast operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: relu
def Darwinn_ReluOp : Darwinn_Op<"relu"> {
  let summary = "Darwinn relu operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.relu operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: resampler
def Darwinn_ResamplerOp : Darwinn_Op<"resampler"> {
  let summary = "Darwinn resampler operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.resampler operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: resampler_options
def Darwinn_ResamplerOptionsOp : Darwinn_Op<"resampler_options"> {
  let summary = "Darwinn resampler_options operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.resampler_options operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: reshape_op
def Darwinn_ReshapeOpOp : Darwinn_Op<"reshape_op"> {
  let summary = "Darwinn reshape_op operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.reshape_op operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: ring_to_tile_options
def Darwinn_RingToTileOptionsOp : Darwinn_Op<"ring_to_tile_options"> {
  let summary = "Darwinn ring_to_tile_options operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.ring_to_tile_options operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: ring_to_tile_slice
def Darwinn_RingToTileSliceOp : Darwinn_Op<"ring_to_tile_slice"> {
  let summary = "Darwinn ring_to_tile_slice operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.ring_to_tile_slice operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: rkhy_compute_op
def Darwinn_RkhyComputeOpOp : Darwinn_Op<"rkhy_compute_op"> {
  let summary = "Darwinn rkhy_compute_op operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.rkhy_compute_op operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: rkhy_custom_padding
def Darwinn_RkhyCustomPaddingOp : Darwinn_Op<"rkhy_custom_padding"> {
  let summary = "Darwinn rkhy_custom_padding operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.rkhy_custom_padding operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: rkhy_depth_to_space_op
def Darwinn_RkhyDepthToSpaceOpOp : Darwinn_Op<"rkhy_depth_to_space_op"> {
  let summary = "Darwinn rkhy_depth_to_space_op operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.rkhy_depth_to_space_op operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: rkhy_residual_add_op
def Darwinn_RkhyResidualAddOpOp : Darwinn_Op<"rkhy_residual_add_op"> {
  let summary = "Darwinn rkhy_residual_add_op operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.rkhy_residual_add_op operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: rkhy_unary_compute_op
def Darwinn_RkhyUnaryComputeOpOp : Darwinn_Op<"rkhy_unary_compute_op"> {
  let summary = "Darwinn rkhy_unary_compute_op operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.rkhy_unary_compute_op operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: rng_bit_generator
def Darwinn_RngBitGeneratorOp : Darwinn_Op<"rng_bit_generator"> {
  let summary = "Darwinn rng_bit_generator operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.rng_bit_generator operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: scalar_register_to_host_transfer
def Darwinn_ScalarRegisterToHostTransferOp : Darwinn_Op<"scalar_register_to_host_transfer"> {
  let summary = "Darwinn scalar_register_to_host_transfer operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.scalar_register_to_host_transfer operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: scalar_register_to_tile_transfer
def Darwinn_ScalarRegisterToTileTransferOp : Darwinn_Op<"scalar_register_to_tile_transfer"> {
  let summary = "Darwinn scalar_register_to_tile_transfer operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.scalar_register_to_tile_transfer operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: scale
def Darwinn_ScaleOp : Darwinn_Op<"scale"> {
  let summary = "Darwinn scale operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.scale operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: scatter
def Darwinn_ScatterOp : Darwinn_Op<"scatter"> {
  let summary = "Darwinn scatter operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.scatter operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: select
def Darwinn_SelectOp : Darwinn_Op<"select"> {
  let summary = "Darwinn select operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.select operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: slice_1d_extent
def Darwinn_Slice1dExtentOp : Darwinn_Op<"slice_1d_extent"> {
  let summary = "Darwinn slice_1d_extent operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.slice_1d_extent operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: slice_1d_extent_with_padding_info
def Darwinn_Slice1dExtentWithPaddingInfoOp : Darwinn_Op<"slice_1d_extent_with_padding_info"> {
  let summary = "Darwinn slice_1d_extent_with_padding_info operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.slice_1d_extent_with_padding_info operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: sparse_narrow_to_wide
def Darwinn_SparseNarrowToWideOp : Darwinn_Op<"sparse_narrow_to_wide"> {
  let summary = "Darwinn sparse_narrow_to_wide operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.sparse_narrow_to_wide operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: sparse_narrow_to_wide_slice
def Darwinn_SparseNarrowToWideSliceOp : Darwinn_Op<"sparse_narrow_to_wide_slice"> {
  let summary = "Darwinn sparse_narrow_to_wide_slice operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.sparse_narrow_to_wide_slice operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: sparse_tensor_op
def Darwinn_SparseTensorOpOp : Darwinn_Op<"sparse_tensor_op"> {
  let summary = "Darwinn sparse_tensor_op operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.sparse_tensor_op operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: sparsity
def Darwinn_SparsityOp : Darwinn_Op<"sparsity"> {
  let summary = "Darwinn sparsity operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.sparsity operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: sparsity_type
def Darwinn_SparsityTypeOp : Darwinn_Op<"sparsity_type"> {
  let summary = "Darwinn sparsity_type operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.sparsity_type operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: spline_segment
def Darwinn_SplineSegmentOp : Darwinn_Op<"spline_segment"> {
  let summary = "Darwinn spline_segment operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.spline_segment operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: split
def Darwinn_SplitOp : Darwinn_Op<"split"> {
  let summary = "Darwinn split operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.split operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: start_offset_and_stride
def Darwinn_StartOffsetAndStrideOp : Darwinn_Op<"start_offset_and_stride"> {
  let summary = "Darwinn start_offset_and_stride operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.start_offset_and_stride operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: static_compute_op
def Darwinn_StaticComputeOpOp : Darwinn_Op<"static_compute_op"> {
  let summary = "Darwinn static_compute_op operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.static_compute_op operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: static_sparse_compute_op
def Darwinn_StaticSparseComputeOpOp : Darwinn_Op<"static_sparse_compute_op"> {
  let summary = "Darwinn static_sparse_compute_op operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.static_sparse_compute_op operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: static_unary_compute_op
def Darwinn_StaticUnaryComputeOpOp : Darwinn_Op<"static_unary_compute_op"> {
  let summary = "Darwinn static_unary_compute_op operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.static_unary_compute_op operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: streaming_compute_op
def Darwinn_StreamingComputeOpOp : Darwinn_Op<"streaming_compute_op"> {
  let summary = "Darwinn streaming_compute_op operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.streaming_compute_op operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: streaming_copy_op
def Darwinn_StreamingCopyOpOp : Darwinn_Op<"streaming_copy_op"> {
  let summary = "Darwinn streaming_copy_op operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.streaming_copy_op operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: streaming_sparse_compute_op
def Darwinn_StreamingSparseComputeOpOp : Darwinn_Op<"streaming_sparse_compute_op"> {
  let summary = "Darwinn streaming_sparse_compute_op operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.streaming_sparse_compute_op operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: streaming_sparse_copy_op
def Darwinn_StreamingSparseCopyOpOp : Darwinn_Op<"streaming_sparse_copy_op"> {
  let summary = "Darwinn streaming_sparse_copy_op operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.streaming_sparse_copy_op operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: streaming_unary_compute_op
def Darwinn_StreamingUnaryComputeOpOp : Darwinn_Op<"streaming_unary_compute_op"> {
  let summary = "Darwinn streaming_unary_compute_op operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.streaming_unary_compute_op operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: swizzling
def Darwinn_SwizzlingOp : Darwinn_Op<"swizzling"> {
  let summary = "Darwinn swizzling operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.swizzling operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: synchronized_compute_op
def Darwinn_SynchronizedComputeOpOp : Darwinn_Op<"synchronized_compute_op"> {
  let summary = "Darwinn synchronized_compute_op operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.synchronized_compute_op operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: synchronized_copy_op
def Darwinn_SynchronizedCopyOpOp : Darwinn_Op<"synchronized_copy_op"> {
  let summary = "Darwinn synchronized_copy_op operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.synchronized_copy_op operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: synchronized_sparse_compute_op
def Darwinn_SynchronizedSparseComputeOpOp : Darwinn_Op<"synchronized_sparse_compute_op"> {
  let summary = "Darwinn synchronized_sparse_compute_op operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.synchronized_sparse_compute_op operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: synchronized_sparse_copy_op
def Darwinn_SynchronizedSparseCopyOpOp : Darwinn_Op<"synchronized_sparse_copy_op"> {
  let summary = "Darwinn synchronized_sparse_copy_op operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.synchronized_sparse_copy_op operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: synchronized_unary_compute_op
def Darwinn_SynchronizedUnaryComputeOpOp : Darwinn_Op<"synchronized_unary_compute_op"> {
  let summary = "Darwinn synchronized_unary_compute_op operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.synchronized_unary_compute_op operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: tensor_op
def Darwinn_TensorOpOp : Darwinn_Op<"tensor_op"> {
  let summary = "Darwinn tensor_op operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.tensor_op operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: tensor_op_shard
def Darwinn_TensorOpShardOp : Darwinn_Op<"tensor_op_shard"> {
  let summary = "Darwinn tensor_op_shard operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.tensor_op_shard operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: tensor_op_slice
def Darwinn_TensorOpSliceOp : Darwinn_Op<"tensor_op_slice"> {
  let summary = "Darwinn tensor_op_slice operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.tensor_op_slice operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: terminate
def Darwinn_TerminateOp : Darwinn_Op<"terminate"> {
  let summary = "Darwinn terminate operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.terminate operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: tgc_elementwise_add
def Darwinn_TgcElementwiseAddOp : Darwinn_Op<"tgc_elementwise_add"> {
  let summary = "Darwinn tgc_elementwise_add operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.tgc_elementwise_add operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: tgc_elementwise_mul
def Darwinn_TgcElementwiseMulOp : Darwinn_Op<"tgc_elementwise_mul"> {
  let summary = "Darwinn tgc_elementwise_mul operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.tgc_elementwise_mul operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: tgc_elementwise_sub
def Darwinn_TgcElementwiseSubOp : Darwinn_Op<"tgc_elementwise_sub"> {
  let summary = "Darwinn tgc_elementwise_sub operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.tgc_elementwise_sub operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: tile_to_host
def Darwinn_TileToHostOp : Darwinn_Op<"tile_to_host"> {
  let summary = "Darwinn tile_to_host operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.tile_to_host operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: tile_to_host_shard
def Darwinn_TileToHostShardOp : Darwinn_Op<"tile_to_host_shard"> {
  let summary = "Darwinn tile_to_host_shard operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.tile_to_host_shard operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: tile_to_ring_slice
def Darwinn_TileToRingSliceOp : Darwinn_Op<"tile_to_ring_slice"> {
  let summary = "Darwinn tile_to_ring_slice operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.tile_to_ring_slice operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: tile_to_scalar_register_transfer
def Darwinn_TileToScalarRegisterTransferOp : Darwinn_Op<"tile_to_scalar_register_transfer"> {
  let summary = "Darwinn tile_to_scalar_register_transfer operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.tile_to_scalar_register_transfer operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: tile_to_tile
def Darwinn_TileToTileOp : Darwinn_Op<"tile_to_tile"> {
  let summary = "Darwinn tile_to_tile operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.tile_to_tile operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: tile_to_tile_shard
def Darwinn_TileToTileShardOp : Darwinn_Op<"tile_to_tile_shard"> {
  let summary = "Darwinn tile_to_tile_shard operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.tile_to_tile_shard operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: unary_map
def Darwinn_UnaryMapOp : Darwinn_Op<"unary_map"> {
  let summary = "Darwinn unary_map operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.unary_map operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: unary_tensor_op
def Darwinn_UnaryTensorOpOp : Darwinn_Op<"unary_tensor_op"> {
  let summary = "Darwinn unary_tensor_op operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.unary_tensor_op operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: vex_info
def Darwinn_VexInfoOp : Darwinn_Op<"vex_info"> {
  let summary = "Darwinn vex_info operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.vex_info operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: vrgkh_operation_mode
def Darwinn_VrgkhOperationModeOp : Darwinn_Op<"vrgkh_operation_mode"> {
  let summary = "Darwinn vrgkh_operation_mode operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.vrgkh_operation_mode operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: while
def Darwinn_WhileOp : Darwinn_Op<"while"> {
  let summary = "Darwinn while operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.while operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: wide_to_narrow
def Darwinn_WideToNarrowOp : Darwinn_Op<"wide_to_narrow"> {
  let summary = "Darwinn wide_to_narrow operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.wide_to_narrow operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: wide_to_narrow_slice
def Darwinn_WideToNarrowSliceOp : Darwinn_Op<"wide_to_narrow_slice"> {
  let summary = "Darwinn wide_to_narrow_slice operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.wide_to_narrow_slice operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: yield
def Darwinn_YieldOp : Darwinn_Op<"yield"> {
  let summary = "Darwinn yield operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.yield operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: zero_point
def Darwinn_ZeroPointOp : Darwinn_Op<"zero_point"> {
  let summary = "Darwinn zero_point operation.";
  let description = [{
    Boilerplate skeleton for the darwinn.zero_point operation.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

#endif // DARWINN_OPS
