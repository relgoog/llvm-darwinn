//===-- DwgTensorOps.td - dwg_tensor operations -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tensor-level dwg operations with pass-description evidence in the blob.
//
//===----------------------------------------------------------------------===//

#ifndef DWG_TENSOR_OPS
#define DWG_TENSOR_OPS

include "mlir/IR/OpBase.td"

include "mlir/Dialect/DiveVm/IR/DiveVmDialect.td"

// Operator: codegen
def DwgTensor_CodegenOp : DwgTensor_Op<"codegen"> {
  let summary = "dwg_tensor codegen operation.";
  let description = [{
    Takes any number of operands. Passes report invalid dwg_tensor.codegen ops.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: const
def DwgTensor_ConstOp : DwgTensor_Op<"const"> {
  let summary = "dwg_tensor const operation.";
  let description = [{
    Takes any number of operands. Passes convert fill ops to dwg_tensor.const
    where feasible.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: dynamic_shape_scope
def DwgTensor_DynamicShapeScopeOp : DwgTensor_Op<"dynamic_shape_scope"> {
  let summary = "dwg_tensor dynamic_shape_scope operation.";
  let description = [{
    Takes any number of operands. Dynamic-shape operands must be defined by a
    dwg_tensor dynamic shape scope op.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: eval_with_shape
def DwgTensor_EvalWithShapeOp : DwgTensor_Op<"eval_with_shape"> {
  let summary = "dwg_tensor eval_with_shape operation.";
  let description = [{
    Takes any number of operands. Dynamic-shape operands must be defined by a
    dwg_tensor eval with shape op.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DwgTensor_AbsOp : DwgTensor_Op<"abs"> {
  let summary = "dwg_tensor abs operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DwgTensor_AddOp : DwgTensor_Op<"add"> {
  let summary = "dwg_tensor add operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DwgTensor_ArithmeticLeftShiftOp : DwgTensor_Op<"arithmetic_left_shift"> {
  let summary = "dwg_tensor arithmetic_left_shift operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DwgTensor_ArithmeticRightShiftOp : DwgTensor_Op<"arithmetic_right_shift"> {
  let summary = "dwg_tensor arithmetic_right_shift operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DwgTensor_BarrierOp : DwgTensor_Op<"barrier"> {
  let summary = "dwg_tensor barrier operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DwgTensor_BatchMatrixNmsOp : DwgTensor_Op<"batch_matrix_nms"> {
  let summary = "dwg_tensor batch_matrix_nms operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DwgTensor_BitwiseAndOp : DwgTensor_Op<"bitwise_and"> {
  let summary = "dwg_tensor bitwise_and operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DwgTensor_BitwiseOrOp : DwgTensor_Op<"bitwise_or"> {
  let summary = "dwg_tensor bitwise_or operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DwgTensor_BitwiseXorOp : DwgTensor_Op<"bitwise_xor"> {
  let summary = "dwg_tensor bitwise_xor operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DwgTensor_CastOp : DwgTensor_Op<"cast"> {
  let summary = "dwg_tensor cast operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DwgTensor_CastFromIndexOp : DwgTensor_Op<"cast_from_index"> {
  let summary = "dwg_tensor cast_from_index operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DwgTensor_CastToIndexOp : DwgTensor_Op<"cast_to_index"> {
  let summary = "dwg_tensor cast_to_index operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DwgTensor_ConcatenationOp : DwgTensor_Op<"concatenation"> {
  let summary = "dwg_tensor concatenation operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DwgTensor_ConstBytesOp : DwgTensor_Op<"const_bytes"> {
  let summary = "dwg_tensor const_bytes operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DwgTensor_ConstNoneOp : DwgTensor_Op<"const_none"> {
  let summary = "dwg_tensor const_none operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DwgTensor_CopyOp : DwgTensor_Op<"copy"> {
  let summary = "dwg_tensor copy operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DwgTensor_CopyOuterSliceOp : DwgTensor_Op<"copy_outer_slice"> {
  let summary = "dwg_tensor copy_outer_slice operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DwgTensor_CountLeadingZerosOp : DwgTensor_Op<"count_leading_zeros"> {
  let summary = "dwg_tensor count_leading_zeros operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DwgTensor_CumsumOp : DwgTensor_Op<"cumsum"> {
  let summary = "dwg_tensor cumsum operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DwgTensor_DeclareTensorOp : DwgTensor_Op<"declare_tensor"> {
  let summary = "dwg_tensor declare_tensor operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DwgTensor_DeclareTensorStaticOp : DwgTensor_Op<"declare_tensor_static"> {
  let summary = "dwg_tensor declare_tensor_static operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DwgTensor_DimensionLengthOp : DwgTensor_Op<"dimension_length"> {
  let summary = "dwg_tensor dimension_length operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DwgTensor_DivOp : DwgTensor_Op<"div"> {
  let summary = "dwg_tensor div operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DwgTensor_EqOp : DwgTensor_Op<"eq"> {
  let summary = "dwg_tensor eq operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DwgTensor_ForOp : DwgTensor_Op<"for"> {
  let summary = "dwg_tensor for operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DwgTensor_GatherOp : DwgTensor_Op<"gather"> {
  let summary = "dwg_tensor gather operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DwgTensor_GatherNdOp : DwgTensor_Op<"gather_nd"> {
  let summary = "dwg_tensor gather_nd operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DwgTensor_GeOp : DwgTensor_Op<"ge"> {
  let summary = "dwg_tensor ge operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DwgTensor_GtOp : DwgTensor_Op<"gt"> {
  let summary = "dwg_tensor gt operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DwgTensor_GumbelRngOp : DwgTensor_Op<"gumbel_rng"> {
  let summary = "dwg_tensor gumbel_rng operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DwgTensor_IfOp : DwgTensor_Op<"if"> {
  let summary = "dwg_tensor if operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DwgTensor_LandmarksToTransformMatrixOp : DwgTensor_Op<"landmarks_to_transform_matrix"> {
  let summary = "dwg_tensor landmarks_to_transform_matrix operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DwgTensor_LeOp : DwgTensor_Op<"le"> {
  let summary = "dwg_tensor le operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DwgTensor_LogicalAndOp : DwgTensor_Op<"logical_and"> {
  let summary = "dwg_tensor logical_and operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DwgTensor_LtOp : DwgTensor_Op<"lt"> {
  let summary = "dwg_tensor lt operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DwgTensor_MaximumOp : DwgTensor_Op<"maximum"> {
  let summary = "dwg_tensor maximum operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DwgTensor_MemoryCopyOp : DwgTensor_Op<"memory_copy"> {
  let summary = "dwg_tensor memory_copy operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DwgTensor_MinimumOp : DwgTensor_Op<"minimum"> {
  let summary = "dwg_tensor minimum operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DwgTensor_ModOp : DwgTensor_Op<"mod"> {
  let summary = "dwg_tensor mod operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DwgTensor_MulOp : DwgTensor_Op<"mul"> {
  let summary = "dwg_tensor mul operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DwgTensor_NeOp : DwgTensor_Op<"ne"> {
  let summary = "dwg_tensor ne operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DwgTensor_NewQosClassOp : DwgTensor_Op<"new_qos_class"> {
  let summary = "dwg_tensor new_qos_class operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DwgTensor_OneHotOp : DwgTensor_Op<"one_hot"> {
  let summary = "dwg_tensor one_hot operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DwgTensor_OneHotV2Op : DwgTensor_Op<"one_hot_v2"> {
  let summary = "dwg_tensor one_hot_v2 operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DwgTensor_OuterSliceOp : DwgTensor_Op<"outer_slice"> {
  let summary = "dwg_tensor outer_slice operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DwgTensor_PadOp : DwgTensor_Op<"pad"> {
  let summary = "dwg_tensor pad operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DwgTensor_PopCountOp : DwgTensor_Op<"pop_count"> {
  let summary = "dwg_tensor pop_count operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DwgTensor_PowOp : DwgTensor_Op<"pow"> {
  let summary = "dwg_tensor pow operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DwgTensor_RaiseErrorOp : DwgTensor_Op<"raise_error"> {
  let summary = "dwg_tensor raise_error operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DwgTensor_ReductionOp : DwgTensor_Op<"reduction"> {
  let summary = "dwg_tensor reduction operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DwgTensor_ReductionV2Op : DwgTensor_Op<"reduction_v2"> {
  let summary = "dwg_tensor reduction_v2 operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DwgTensor_RelayoutOp : DwgTensor_Op<"relayout"> {
  let summary = "dwg_tensor relayout operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DwgTensor_RoiToTransformMatrixOp : DwgTensor_Op<"roi_to_transform_matrix"> {
  let summary = "dwg_tensor roi_to_transform_matrix operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DwgTensor_RollOp : DwgTensor_Op<"roll"> {
  let summary = "dwg_tensor roll operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DwgTensor_ScatterNdOp : DwgTensor_Op<"scatter_nd"> {
  let summary = "dwg_tensor scatter_nd operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DwgTensor_SelectOp : DwgTensor_Op<"select"> {
  let summary = "dwg_tensor select operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DwgTensor_SelectV2Op : DwgTensor_Op<"select_v2"> {
  let summary = "dwg_tensor select_v2 operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DwgTensor_SetDimensionLengthOp : DwgTensor_Op<"set_dimension_length"> {
  let summary = "dwg_tensor set_dimension_length operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DwgTensor_SignOp : DwgTensor_Op<"sign"> {
  let summary = "dwg_tensor sign operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DwgTensor_SingleChipOffloadOp : DwgTensor_Op<"single_chip_offload"> {
  let summary = "dwg_tensor single_chip_offload operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DwgTensor_SliceV2Op : DwgTensor_Op<"slice_v2"> {
  let summary = "dwg_tensor slice_v2 operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DwgTensor_SortOp : DwgTensor_Op<"sort"> {
  let summary = "dwg_tensor sort operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DwgTensor_StaticErrorOp : DwgTensor_Op<"static_error"> {
  let summary = "dwg_tensor static_error operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DwgTensor_SubOp : DwgTensor_Op<"sub"> {
  let summary = "dwg_tensor sub operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DwgTensor_SwitchOp : DwgTensor_Op<"switch"> {
  let summary = "dwg_tensor switch operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DwgTensor_TestOnlyFingerprintCoreOp : DwgTensor_Op<"test_only_fingerprint_core_"> {
  let summary = "dwg_tensor test_only_fingerprint_core_ operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DwgTensor_TopKOp : DwgTensor_Op<"top_k"> {
  let summary = "dwg_tensor top_k operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DwgTensor_TpuOffloadOp : DwgTensor_Op<"tpu_offload"> {
  let summary = "dwg_tensor tpu_offload operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DwgTensor_TransformLandmarksOp : DwgTensor_Op<"transform_landmarks"> {
  let summary = "dwg_tensor transform_landmarks operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DwgTensor_TransformTensorBilinearOp : DwgTensor_Op<"transform_tensor_bilinear"> {
  let summary = "dwg_tensor transform_tensor_bilinear operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DwgTensor_TypeCastOp : DwgTensor_Op<"type_cast"> {
  let summary = "dwg_tensor type_cast operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DwgTensor_TypeCastDynamicOp : DwgTensor_Op<"type_cast_dynamic"> {
  let summary = "dwg_tensor type_cast_dynamic operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DwgTensor_UniformRandomNumberGenerationOp : DwgTensor_Op<"uniform_random_number_generation"> {
  let summary = "dwg_tensor uniform_random_number_generation operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DwgTensor_UnsortedSegmentReduceOp : DwgTensor_Op<"unsorted_segment_reduce"> {
  let summary = "dwg_tensor unsorted_segment_reduce operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DwgTensor_UpdateOuterSliceOp : DwgTensor_Op<"update_outer_slice"> {
  let summary = "dwg_tensor update_outer_slice operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DwgTensor_UseMaxConcurrencyModeOp : DwgTensor_Op<"use_max_concurrency_mode"> {
  let summary = "dwg_tensor use_max_concurrency_mode operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DwgTensor_WhileOp : DwgTensor_Op<"while"> {
  let summary = "dwg_tensor while operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DwgTensor_YieldOp : DwgTensor_Op<"yield"> {
  let summary = "dwg_tensor yield operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

#endif // DWG_TENSOR_OPS
