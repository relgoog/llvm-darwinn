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

#endif // DIVEVM_TENSOR_OPS
