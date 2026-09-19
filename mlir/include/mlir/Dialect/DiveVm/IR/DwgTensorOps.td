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

#endif // DWG_TENSOR_OPS
