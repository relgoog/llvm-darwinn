//===-- DiveVmTensorOps.td - dive_vm_tensor operations -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tensor-level dive_vm operations with pass-description evidence in the blob.
//
//===----------------------------------------------------------------------===//

#ifndef DIVEVM_TENSOR_OPS
#define DIVEVM_TENSOR_OPS

include "mlir/IR/OpBase.td"

include "mlir/Dialect/DiveVm/IR/DiveVmDialect.td"

// Operator: allocate
def DiveVmTensor_AllocateOp : DiveVmTensor_Op<"allocate"> {
  let summary = "dive_vm_tensor allocate operation.";
  let description = [{
    Takes any number of operands. Passes move dive_vm_tensor.allocate ops
    right before their first user.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: const
def DiveVmTensor_ConstOp : DiveVmTensor_Op<"const"> {
  let summary = "dive_vm_tensor const operation.";
  let description = [{
    Takes any number of operands. Requires a `value` attribute per the
    missing-value diagnostic.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: extract_slice
def DiveVmTensor_ExtractSliceOp : DiveVmTensor_Op<"extract_slice"> {
  let summary = "dive_vm_tensor extract_slice operation.";
  let description = [{
    Takes any number of operands. Passes optimize the offset of
    dive_vm_tensor insert and extract slice ops.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: insert_slice
def DiveVmTensor_InsertSliceOp : DiveVmTensor_Op<"insert_slice"> {
  let summary = "dive_vm_tensor insert_slice operation.";
  let description = [{
    Takes any number of operands. Passes optimize the offset of
    dive_vm_tensor insert and extract slice ops.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Operator: type_cast
def DiveVmTensor_TypeCastOp : DiveVmTensor_Op<"type_cast"> {
  let summary = "dive_vm_tensor type_cast operation.";
  let description = [{
    Takes any number of operands. Passes convert dive_vm_tensor.type_cast
    to tensor.cast.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_AbsOp : DiveVmTensor_Op<"abs"> {
  let summary = "dive_vm_tensor abs operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_AddOp : DiveVmTensor_Op<"add"> {
  let summary = "dive_vm_tensor add operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_AddImmOp : DiveVmTensor_Op<"add_imm"> {
  let summary = "dive_vm_tensor add_imm operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_AddressOfOp : DiveVmTensor_Op<"address_of"> {
  let summary = "dive_vm_tensor address_of operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_AddressOfActivationOp : DiveVmTensor_Op<"address_of_activation"> {
  let summary = "dive_vm_tensor address_of_activation operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_AddressOfInputActivationOp : DiveVmTensor_Op<"address_of_input_activation"> {
  let summary = "dive_vm_tensor address_of_input_activation operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_AddressOfOutputActivationOp : DiveVmTensor_Op<"address_of_output_activation"> {
  let summary = "dive_vm_tensor address_of_output_activation operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_AddressOfParameterOp : DiveVmTensor_Op<"address_of_parameter"> {
  let summary = "dive_vm_tensor address_of_parameter operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_AddressOfParameterRegionOp : DiveVmTensor_Op<"address_of_parameter_region"> {
  let summary = "dive_vm_tensor address_of_parameter_region operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_AddressOfScratchOp : DiveVmTensor_Op<"address_of_scratch"> {
  let summary = "dive_vm_tensor address_of_scratch operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_ArithmeticLeftShiftOp : DiveVmTensor_Op<"arithmetic_left_shift"> {
  let summary = "dive_vm_tensor arithmetic_left_shift operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_ArithmeticLeftShiftImmOp : DiveVmTensor_Op<"arithmetic_left_shift_imm"> {
  let summary = "dive_vm_tensor arithmetic_left_shift_imm operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_ArithmeticRightShiftOp : DiveVmTensor_Op<"arithmetic_right_shift"> {
  let summary = "dive_vm_tensor arithmetic_right_shift operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_ArithmeticRightShiftImmOp : DiveVmTensor_Op<"arithmetic_right_shift_imm"> {
  let summary = "dive_vm_tensor arithmetic_right_shift_imm operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_BitcastOp : DiveVmTensor_Op<"bitcast"> {
  let summary = "dive_vm_tensor bitcast operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_BitwiseAndOp : DiveVmTensor_Op<"bitwise_and"> {
  let summary = "dive_vm_tensor bitwise_and operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_BitwiseAndImmOp : DiveVmTensor_Op<"bitwise_and_imm"> {
  let summary = "dive_vm_tensor bitwise_and_imm operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_BitwiseOrOp : DiveVmTensor_Op<"bitwise_or"> {
  let summary = "dive_vm_tensor bitwise_or operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_BitwiseOrImmOp : DiveVmTensor_Op<"bitwise_or_imm"> {
  let summary = "dive_vm_tensor bitwise_or_imm operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_BitwiseXorOp : DiveVmTensor_Op<"bitwise_xor"> {
  let summary = "dive_vm_tensor bitwise_xor operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_BitwiseXorImmOp : DiveVmTensor_Op<"bitwise_xor_imm"> {
  let summary = "dive_vm_tensor bitwise_xor_imm operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_BrOp : DiveVmTensor_Op<"br"> {
  let summary = "dive_vm_tensor br operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_CacheCleanInvalidateOp : DiveVmTensor_Op<"cache_clean_invalidate"> {
  let summary = "dive_vm_tensor cache_clean_invalidate operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_CastOp : DiveVmTensor_Op<"cast"> {
  let summary = "dive_vm_tensor cast operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_ConcatenateOp : DiveVmTensor_Op<"concatenate"> {
  let summary = "dive_vm_tensor concatenate operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_CondBrOp : DiveVmTensor_Op<"cond_br"> {
  let summary = "dive_vm_tensor cond_br operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_CopyOp : DiveVmTensor_Op<"copy"> {
  let summary = "dive_vm_tensor copy operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_CopyImmOp : DiveVmTensor_Op<"copy_imm"> {
  let summary = "dive_vm_tensor copy_imm operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_CountLeadingZerosOp : DiveVmTensor_Op<"count_leading_zeros"> {
  let summary = "dive_vm_tensor count_leading_zeros operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_CumsumOp : DiveVmTensor_Op<"cumsum"> {
  let summary = "dive_vm_tensor cumsum operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_CustomOnBuffersOp : DiveVmTensor_Op<"custom_on_buffers"> {
  let summary = "dive_vm_tensor custom_on_buffers operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_DeclareTensorOp : DiveVmTensor_Op<"declare_tensor"> {
  let summary = "dive_vm_tensor declare_tensor operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_DivOp : DiveVmTensor_Op<"div"> {
  let summary = "dive_vm_tensor div operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_DivImmOp : DiveVmTensor_Op<"div_imm"> {
  let summary = "dive_vm_tensor div_imm operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_DmaHintOp : DiveVmTensor_Op<"dma_hint"> {
  let summary = "dive_vm_tensor dma_hint operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_DynamicSliceYOp : DiveVmTensor_Op<"dynamic_slice_y"> {
  let summary = "dive_vm_tensor dynamic_slice_y operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_EqualOp : DiveVmTensor_Op<"equal"> {
  let summary = "dive_vm_tensor equal operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_EqualImmOp : DiveVmTensor_Op<"equal_imm"> {
  let summary = "dive_vm_tensor equal_imm operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_EvalWithShapeOp : DiveVmTensor_Op<"eval_with_shape"> {
  let summary = "dive_vm_tensor eval_with_shape operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_ExtractOp : DiveVmTensor_Op<"extract"> {
  let summary = "dive_vm_tensor extract operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_FillOp : DiveVmTensor_Op<"fill"> {
  let summary = "dive_vm_tensor fill operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_FloorDivOp : DiveVmTensor_Op<"floor_div"> {
  let summary = "dive_vm_tensor floor_div operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_GatherOp : DiveVmTensor_Op<"gather"> {
  let summary = "dive_vm_tensor gather operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_GatherNdOp : DiveVmTensor_Op<"gather_nd"> {
  let summary = "dive_vm_tensor gather_nd operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_GreaterOp : DiveVmTensor_Op<"greater"> {
  let summary = "dive_vm_tensor greater operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_GreaterEqualOp : DiveVmTensor_Op<"greater_equal"> {
  let summary = "dive_vm_tensor greater_equal operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_GreaterEqualImmOp : DiveVmTensor_Op<"greater_equal_imm"> {
  let summary = "dive_vm_tensor greater_equal_imm operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_GreaterImmOp : DiveVmTensor_Op<"greater_imm"> {
  let summary = "dive_vm_tensor greater_imm operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_IfOp : DiveVmTensor_Op<"if"> {
  let summary = "dive_vm_tensor if operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_InsertOp : DiveVmTensor_Op<"insert"> {
  let summary = "dive_vm_tensor insert operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_LessOp : DiveVmTensor_Op<"less"> {
  let summary = "dive_vm_tensor less operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_LessEqualOp : DiveVmTensor_Op<"less_equal"> {
  let summary = "dive_vm_tensor less_equal operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_LessEqualImmOp : DiveVmTensor_Op<"less_equal_imm"> {
  let summary = "dive_vm_tensor less_equal_imm operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_LessImmOp : DiveVmTensor_Op<"less_imm"> {
  let summary = "dive_vm_tensor less_imm operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_LoadOp : DiveVmTensor_Op<"load"> {
  let summary = "dive_vm_tensor load operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_LoadIndirectOp : DiveVmTensor_Op<"load_indirect"> {
  let summary = "dive_vm_tensor load_indirect operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_LogicalAndOp : DiveVmTensor_Op<"logical_and"> {
  let summary = "dive_vm_tensor logical_and operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_LogicalAndImmOp : DiveVmTensor_Op<"logical_and_imm"> {
  let summary = "dive_vm_tensor logical_and_imm operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_LogicalRightShiftOp : DiveVmTensor_Op<"logical_right_shift"> {
  let summary = "dive_vm_tensor logical_right_shift operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_LogicalRightShiftImmOp : DiveVmTensor_Op<"logical_right_shift_imm"> {
  let summary = "dive_vm_tensor logical_right_shift_imm operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_MarkValueWithShapeOp : DiveVmTensor_Op<"mark_value_with_shape"> {
  let summary = "dive_vm_tensor mark_value_with_shape operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_MaxOp : DiveVmTensor_Op<"max"> {
  let summary = "dive_vm_tensor max operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_MaxImmOp : DiveVmTensor_Op<"max_imm"> {
  let summary = "dive_vm_tensor max_imm operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_MinOp : DiveVmTensor_Op<"min"> {
  let summary = "dive_vm_tensor min operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_MinImmOp : DiveVmTensor_Op<"min_imm"> {
  let summary = "dive_vm_tensor min_imm operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_MulOp : DiveVmTensor_Op<"mul"> {
  let summary = "dive_vm_tensor mul operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_MulImmOp : DiveVmTensor_Op<"mul_imm"> {
  let summary = "dive_vm_tensor mul_imm operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_NotEqualOp : DiveVmTensor_Op<"not_equal"> {
  let summary = "dive_vm_tensor not_equal operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_NotEqualImmOp : DiveVmTensor_Op<"not_equal_imm"> {
  let summary = "dive_vm_tensor not_equal_imm operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_OneHotOp : DiveVmTensor_Op<"one_hot"> {
  let summary = "dive_vm_tensor one_hot operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_PadOp : DiveVmTensor_Op<"pad"> {
  let summary = "dive_vm_tensor pad operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_PerformSoftwarePreemptionIfRequestedOp : DiveVmTensor_Op<"perform_software_preemption_if_requested"> {
  let summary = "dive_vm_tensor perform_software_preemption_if_requested operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_PopCountOp : DiveVmTensor_Op<"pop_count"> {
  let summary = "dive_vm_tensor pop_count operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_PowOp : DiveVmTensor_Op<"pow"> {
  let summary = "dive_vm_tensor pow operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_PowImmOp : DiveVmTensor_Op<"pow_imm"> {
  let summary = "dive_vm_tensor pow_imm operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_PrintOp : DiveVmTensor_Op<"print"> {
  let summary = "dive_vm_tensor print operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_PutBitsOp : DiveVmTensor_Op<"put_bits"> {
  let summary = "dive_vm_tensor put_bits operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_ReductionOp : DiveVmTensor_Op<"reduction"> {
  let summary = "dive_vm_tensor reduction operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_RemOp : DiveVmTensor_Op<"rem"> {
  let summary = "dive_vm_tensor rem operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_RemImmOp : DiveVmTensor_Op<"rem_imm"> {
  let summary = "dive_vm_tensor rem_imm operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_RollOp : DiveVmTensor_Op<"roll"> {
  let summary = "dive_vm_tensor roll operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_RunSerializedModelOp : DiveVmTensor_Op<"run_serialized_model"> {
  let summary = "dive_vm_tensor run_serialized_model operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_ScatterNdOp : DiveVmTensor_Op<"scatter_nd"> {
  let summary = "dive_vm_tensor scatter_nd operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_SelectOp : DiveVmTensor_Op<"select"> {
  let summary = "dive_vm_tensor select operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_ShapeOfActivationOp : DiveVmTensor_Op<"shape_of_activation"> {
  let summary = "dive_vm_tensor shape_of_activation operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_SignOp : DiveVmTensor_Op<"sign"> {
  let summary = "dive_vm_tensor sign operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_StoreOp : DiveVmTensor_Op<"store"> {
  let summary = "dive_vm_tensor store operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_StoreIndirectOp : DiveVmTensor_Op<"store_indirect"> {
  let summary = "dive_vm_tensor store_indirect operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_SubOp : DiveVmTensor_Op<"sub"> {
  let summary = "dive_vm_tensor sub operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_SubImmOp : DiveVmTensor_Op<"sub_imm"> {
  let summary = "dive_vm_tensor sub_imm operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_SwitchOp : DiveVmTensor_Op<"switch"> {
  let summary = "dive_vm_tensor switch operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_TopKOp : DiveVmTensor_Op<"top_k"> {
  let summary = "dive_vm_tensor top_k operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_TpuOffloadOp : DiveVmTensor_Op<"tpu_offload"> {
  let summary = "dive_vm_tensor tpu_offload operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_WhileOp : DiveVmTensor_Op<"while"> {
  let summary = "dive_vm_tensor while operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_WriteDmaDescriptorOp : DiveVmTensor_Op<"write_dma_descriptor"> {
  let summary = "dive_vm_tensor write_dma_descriptor operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def DiveVmTensor_YieldOp : DiveVmTensor_Op<"yield"> {
  let summary = "dive_vm_tensor yield operation.";
  let description = [{
    Takes any number of operands. Registry-evidenced op with no verifier constraints in the blob.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

#endif // DIVEVM_TENSOR_OPS
