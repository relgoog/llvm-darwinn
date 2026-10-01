#ifndef MLIR_DIALECT_DARWINN_IR_DARWINNCOMPUTEOPS_TD
#define MLIR_DIALECT_DARWINN_IR_DARWINNCOMPUTEOPS_TD

include "mlir/Dialect/Darwinn/IR/DarwinnDistributed.td"
include "mlir/Dialect/Darwinn/IR/DarwinnComputeAttrs.td"
include "mlir/Dialect/Darwinn/IR/DarwinnScheduledAttrs.td"

def Darwinn_ComputeTensorOrView
    : AnyTypeOf<[Darwinn_DistributedTensor, Darwinn_DistributedView],
                "distributed tensor or view">;

def Darwinn_ComputeAuxiliaryKinds
    : TypedArrayAttrBase<Darwinn_AuxTensorTypeAttr,
                         "array of auxiliary tensor kinds">;

def Darwinn_ComputeTensorShards
    : TypedArrayAttrBase<Darwinn_TensorOpShardAttr,
                         "array of tensor operation shards">;

def Darwinn_ComputeTensorSlices
    : TypedArrayAttrBase<Darwinn_TensorOpSliceAttr,
                         "array of tensor operation slices">;

def Darwinn_ComputeNarrowToWideShards
    : TypedArrayAttrBase<Darwinn_NarrowToWideShardAttr,
                         "array of narrow to wide shards">;

def Darwinn_StaticComputeOpOp : Darwinn_Op<"static_compute_op"> {
  let arguments = (ins Darwinn_DistributedView:$lhs,
      Darwinn_DistributedView:$rhs, Darwinn_DistributedView:$destination,
      Variadic<Darwinn_ComputeTensorOrView>:$auxiliary_tensors,
      Darwinn_ComputeOpOptionsAttr:$compute, AffineMapAttr:$traversal,
      OptionalAttr<Darwinn_ComputeAuxiliaryKinds>:$auxiliary_tensor_types,
      OptionalAttr<Darwinn_CustomTilingOptionsAttr>:$custom_tiling,
      OptionalAttr<Darwinn_DtcInfoAttr>:$dtc_info,
      OptionalAttr<Darwinn_VexInfoAttr>:$vex_info);
  let results = (outs Darwinn_DistributedTensor:$output);
  let hasVerifier = 1;
  let assemblyFormat =
      "operands attr-dict `:` functional-type(operands, results)";
}

def Darwinn_StaticUnaryComputeOpOp : Darwinn_Op<"static_unary_compute_op"> {
  let arguments = (ins Darwinn_DistributedView:$input,
      Darwinn_DistributedView:$destination,
      Variadic<Darwinn_ComputeTensorOrView>:$auxiliary_tensors,
      Darwinn_ComputeOpOptionsAttr:$compute, AffineMapAttr:$traversal,
      OptionalAttr<Darwinn_ComputeAuxiliaryKinds>:$auxiliary_tensor_types,
      OptionalAttr<Darwinn_CustomTilingOptionsAttr>:$custom_tiling,
      OptionalAttr<I32ArrayAttr>:$custom_constraints,
      OptionalAttr<Darwinn_VexInfoAttr>:$vex_info);
  let results = (outs Darwinn_ComputeTensorOrView:$output);
  let hasVerifier = 1;
  let assemblyFormat =
      "operands attr-dict `:` functional-type(operands, results)";
}

def Darwinn_StreamingComputeOpOp : Darwinn_Op<"streaming_compute_op"> {
  let arguments = (ins Darwinn_DistributedView:$lhs,
      Darwinn_DistributedView:$rhs, Darwinn_DistributedView:$destination,
      Variadic<Darwinn_ComputeTensorOrView>:$auxiliary_tensors,
      Darwinn_ComputeOpOptionsAttr:$compute, AffineMapAttr:$traversal,
      AffineMapAttr:$lhs_traversal,
      OptionalAttr<Darwinn_ComputeAuxiliaryKinds>:$auxiliary_tensor_types,
      OptionalAttr<BoolAttr>:$streaming_input0,
      OptionalAttr<BoolAttr>:$streaming_input1,
      OptionalAttr<BoolAttr>:$streaming_output,
      OptionalAttr<Darwinn_CustomTilingOptionsAttr>:$custom_tiling,
      OptionalAttr<Darwinn_DtcInfoAttr>:$dtc_info,
      OptionalAttr<Darwinn_VexInfoAttr>:$vex_info);
  let results = (outs Darwinn_DistributedTensor:$output);
  let hasVerifier = 1;
  let assemblyFormat =
      "operands attr-dict `:` functional-type(operands, results)";
}

def Darwinn_StreamingUnaryComputeOpOp
    : Darwinn_Op<"streaming_unary_compute_op"> {
  let arguments = (ins Darwinn_DistributedView:$input,
      Darwinn_DistributedView:$destination,
      Variadic<Darwinn_ComputeTensorOrView>:$auxiliary_tensors,
      Darwinn_ComputeOpOptionsAttr:$compute, AffineMapAttr:$traversal,
      OptionalAttr<Darwinn_ComputeAuxiliaryKinds>:$auxiliary_tensor_types,
      OptionalAttr<Darwinn_CustomTilingOptionsAttr>:$custom_tiling,
      OptionalAttr<I32ArrayAttr>:$custom_constraints,
      OptionalAttr<Darwinn_VexInfoAttr>:$vex_info);
  let results = (outs Darwinn_ComputeTensorOrView:$output);
  let hasVerifier = 1;
  let assemblyFormat =
      "operands attr-dict `:` functional-type(operands, results)";
}

def Darwinn_StreamingCopyOpOp : Darwinn_Op<"streaming_copy_op"> {
  let arguments = (ins Darwinn_ComputeTensorOrView:$input,
      AffineMapAttr:$traversal,
      OptionalAttr<AffineMapAttr>:$forward_index_transformation,
      OptionalAttr<Darwinn_MemSpaceAttr>:$memory_space,
      OptionalAttr<BoolAttr>:$streaming_input,
      OptionalAttr<BoolAttr>:$streaming_output);
  let results = (outs Darwinn_ComputeTensorOrView:$output);
  let assemblyFormat =
      "operands attr-dict `:` functional-type(operands, results)";
}

def Darwinn_SynchronizedComputeOpOp
    : Darwinn_Op<"synchronized_compute_op"> {
  let arguments = (ins Darwinn_DistributedView:$lhs,
      Darwinn_DistributedView:$rhs, Darwinn_DistributedView:$destination,
      Variadic<Darwinn_ComputeTensorOrView>:$auxiliary_tensors,
      Darwinn_ComputeOpOptionsAttr:$compute, AffineMapAttr:$traversal,
      OptionalAttr<Darwinn_ComputeAuxiliaryKinds>:$auxiliary_tensor_types);
  let results = (outs Darwinn_DistributedTensor:$output);
  let hasVerifier = 1;
  let assemblyFormat =
      "operands attr-dict `:` functional-type(operands, results)";
}

def Darwinn_SynchronizedUnaryComputeOpOp
    : Darwinn_Op<"synchronized_unary_compute_op"> {
  let arguments = (ins Darwinn_DistributedView:$input,
      Darwinn_DistributedView:$destination,
      Variadic<Darwinn_ComputeTensorOrView>:$auxiliary_tensors,
      Darwinn_ComputeOpOptionsAttr:$compute, AffineMapAttr:$traversal,
      OptionalAttr<Darwinn_ComputeAuxiliaryKinds>:$auxiliary_tensor_types,
      OptionalAttr<I32ArrayAttr>:$custom_constraints);
  let results = (outs Darwinn_ComputeTensorOrView:$output);
  let hasVerifier = 1;
  let assemblyFormat =
      "operands attr-dict `:` functional-type(operands, results)";
}

def Darwinn_SynchronizedCopyOpOp : Darwinn_Op<"synchronized_copy_op"> {
  let arguments = (ins Darwinn_ComputeTensorOrView:$input,
      AffineMapAttr:$traversal,
      OptionalAttr<AffineMapAttr>:$forward_index_transformation);
  let results = (outs Darwinn_ComputeTensorOrView:$output);
  let assemblyFormat =
      "operands attr-dict `:` functional-type(operands, results)";
}

def Darwinn_TensorOpOp : Darwinn_Op<"tensor_op"> {
  let arguments = (ins Darwinn_DistributedView:$lhs,
      Darwinn_DistributedView:$rhs, Darwinn_DistributedView:$destination,
      Variadic<Darwinn_ComputeTensorOrView>:$auxiliary_tensors,
      Darwinn_ComputeOpOptionsAttr:$compute,
      Darwinn_ComputeTensorShards:$shards, AffineMapAttr:$traversal,
      OptionalAttr<Darwinn_ComputeAuxiliaryKinds>:$auxiliary_tensor_types,
      OptionalAttr<ArrayAttr>:$fences);
  let results = (outs Darwinn_ComputeTensorOrView:$output);
  let hasVerifier = 1;
  let assemblyFormat =
      "operands attr-dict `:` functional-type(operands, results)";
}

def Darwinn_UnaryTensorOpOp : Darwinn_Op<"unary_tensor_op"> {
  let arguments = (ins Darwinn_DistributedView:$input,
      Darwinn_DistributedView:$destination,
      Variadic<Darwinn_ComputeTensorOrView>:$auxiliary_tensors,
      Darwinn_ComputeOpOptionsAttr:$compute, Darwinn_ComputeTensorSlices:$slice,
      AffineMapAttr:$traversal,
      OptionalAttr<Darwinn_ComputeAuxiliaryKinds>:$auxiliary_tensor_types,
      OptionalAttr<ArrayAttr>:$fences,
      OptionalAttr<I32ArrayAttr>:$custom_constraints,
      OptionalAttr<Darwinn_ResamplerOptionsAttr>:$resampler_options);
  let results = (outs Darwinn_ComputeTensorOrView:$output);
  let hasVerifier = 1;
  let assemblyFormat =
      "operands attr-dict `:` functional-type(operands, results)";
}

def Darwinn_NarrowToWideOp : Darwinn_Op<"narrow_to_wide"> {
  let arguments = (ins Darwinn_ComputeTensorOrView:$input,
      Variadic<Darwinn_ComputeTensorOrView>:$auxiliary_tensors,
      Darwinn_ComputeNarrowToWideShards:$shards, AffineMapAttr:$traversal,
      OptionalAttr<AffineMapAttr>:$forward_index_transformation,
      OptionalAttr<AffineMapAttr>:$reverse_index_transformation,
      OptionalAttr<I32Attr>:$zero_point,
      OptionalAttr<I32Attr>:$input_channel_tile,
      OptionalAttr<Darwinn_ComputeAuxiliaryKinds>:$auxiliary_tensor_types,
      OptionalAttr<ArrayAttr>:$fences);
  let results = (outs Darwinn_ComputeTensorOrView:$output,
      Variadic<Darwinn_ComputeTensorOrView>:$auxiliary_outputs);
  let hasVerifier = 1;
  let assemblyFormat =
      "operands attr-dict `:` functional-type(operands, results)";
}

def Darwinn_InterpolateHardwareOp : Darwinn_Op<"interpolate_hardware"> {
  let arguments = (ins Darwinn_DistributedTensor:$input,
      Darwinn_InterpolateMethodAttr:$interpolate_method,
      Darwinn_StartOffsetAndStrideAttr:$x, Darwinn_StartOffsetAndStrideAttr:$y,
      OptionalAttr<AffineMapAttr>:$slicing_begins,
      OptionalAttr<I32ArrayAttr>:$slicing_domain,
      OptionalAttr<AffineMapAttr>:$slicing_ends,
      OptionalAttr<I32ArrayAttr>:$custom_constraints);
  let results = (outs Darwinn_DistributedTensor:$output);
  let hasVerifier = 1;
  let assemblyFormat =
      "operands attr-dict `:` functional-type(operands, results)";
}

#endif
