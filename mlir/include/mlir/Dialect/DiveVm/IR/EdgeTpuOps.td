//===- EdgeTpuOps.td - EdgeTPU dialect operation definitions -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the operation set for the edgetpu dialect observed
// alongside dive_vm. It lives in the DiveVm directory the way TosaShapeOps.td
// and TosaUtilOps.td extend the Tosa directory.
//
//===----------------------------------------------------------------------===//

#ifndef EDGETPU_OPS
#define EDGETPU_OPS

include "mlir/IR/OpBase.td"
include "mlir/Dialect/DiveVm/IR/DiveVmDialect.td"

//===----------------------------------------------------------------------===//
// Operator: annotate_materialize_policy
//===----------------------------------------------------------------------===//
def EdgeTpu_AnnotateMaterializePolicyOp : EdgeTpu_Op<"annotate_materialize_policy"> {
  let summary = "EdgeTPU annotate_materialize_policy operation.";
  let description = [{
    Placeholder for the edgetpu.annotate_materialize_policy operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: attention_v1
//===----------------------------------------------------------------------===//
def EdgeTpu_AttentionV1Op : EdgeTpu_Op<"attention_v1"> {
  let summary = "EdgeTPU attention_v1 operation.";
  let description = [{
    Placeholder for the edgetpu.attention_v1 operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: convert_yuv_to_rgb
//===----------------------------------------------------------------------===//
def EdgeTpu_ConvertYuvToRgbOp : EdgeTpu_Op<"convert_yuv_to_rgb"> {
  let summary = "EdgeTPU convert_yuv_to_rgb operation.";
  let description = [{
    Placeholder for the edgetpu.convert_yuv_to_rgb operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: convolution_sub_channel
//===----------------------------------------------------------------------===//
def EdgeTpu_ConvolutionSubChannelOp : EdgeTpu_Op<"convolution_sub_channel"> {
  let summary = "EdgeTPU convolution_sub_channel operation.";
  let description = [{
    Placeholder for the edgetpu.convolution_sub_channel operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: convolution_sub_channel_drq
//===----------------------------------------------------------------------===//
def EdgeTpu_ConvolutionSubChannelDrqOp : EdgeTpu_Op<"convolution_sub_channel_drq"> {
  let summary = "EdgeTPU convolution_sub_channel_drq operation.";
  let description = [{
    Placeholder for the edgetpu.convolution_sub_channel_drq operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: depthwise_convolution_fp8
//===----------------------------------------------------------------------===//
def EdgeTpu_DepthwiseConvolutionFp8Op : EdgeTpu_Op<"depthwise_convolution_fp8"> {
  let summary = "EdgeTPU depthwise_convolution_fp8 operation.";
  let description = [{
    Placeholder for the edgetpu.depthwise_convolution_fp8 operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: fast_walsh_hadamard_transform
//===----------------------------------------------------------------------===//
def EdgeTpu_FastWalshHadamardTransformOp : EdgeTpu_Op<"fast_walsh_hadamard_transform"> {
  let summary = "EdgeTPU fast_walsh_hadamard_transform operation.";
  let description = [{
    Placeholder for the edgetpu.fast_walsh_hadamard_transform operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: fully_connected_fp8
//===----------------------------------------------------------------------===//
def EdgeTpu_FullyConnectedFp8Op : EdgeTpu_Op<"fully_connected_fp8"> {
  let summary = "EdgeTPU fully_connected_fp8 operation.";
  let description = [{
    Placeholder for the edgetpu.fully_connected_fp8 operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: fully_connected_sub_channel
//===----------------------------------------------------------------------===//
def EdgeTpu_FullyConnectedSubChannelOp : EdgeTpu_Op<"fully_connected_sub_channel"> {
  let summary = "EdgeTPU fully_connected_sub_channel operation.";
  let description = [{
    Placeholder for the edgetpu.fully_connected_sub_channel operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: fully_connected_sub_channel_drq
//===----------------------------------------------------------------------===//
def EdgeTpu_FullyConnectedSubChannelDrqOp : EdgeTpu_Op<"fully_connected_sub_channel_drq"> {
  let summary = "EdgeTPU fully_connected_sub_channel_drq operation.";
  let description = [{
    Placeholder for the edgetpu.fully_connected_sub_channel_drq operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: matrix_multiply_sub_channel
//===----------------------------------------------------------------------===//
def EdgeTpu_MatrixMultiplySubChannelOp : EdgeTpu_Op<"matrix_multiply_sub_channel"> {
  let summary = "EdgeTPU matrix_multiply_sub_channel operation.";
  let description = [{
    Placeholder for the edgetpu.matrix_multiply_sub_channel operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: matrix_multiply_sub_channel_drq
//===----------------------------------------------------------------------===//
def EdgeTpu_MatrixMultiplySubChannelDrqOp : EdgeTpu_Op<"matrix_multiply_sub_channel_drq"> {
  let summary = "EdgeTPU matrix_multiply_sub_channel_drq operation.";
  let description = [{
    Placeholder for the edgetpu.matrix_multiply_sub_channel_drq operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: transpose_convolution_fp8
//===----------------------------------------------------------------------===//
def EdgeTpu_TransposeConvolutionFp8Op : EdgeTpu_Op<"transpose_convolution_fp8"> {
  let summary = "EdgeTPU transpose_convolution_fp8 operation.";
  let description = [{
    Placeholder for the edgetpu.transpose_convolution_fp8 operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: transposed_convolution_sub_channel
//===----------------------------------------------------------------------===//
def EdgeTpu_TransposedConvolutionSubChannelOp : EdgeTpu_Op<"transposed_convolution_sub_channel"> {
  let summary = "EdgeTPU transposed_convolution_sub_channel operation.";
  let description = [{
    Placeholder for the edgetpu.transposed_convolution_sub_channel operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: transposed_convolution_sub_channel_drq
//===----------------------------------------------------------------------===//
def EdgeTpu_TransposedConvolutionSubChannelDrqOp : EdgeTpu_Op<"transposed_convolution_sub_channel_drq"> {
  let summary = "EdgeTPU transposed_convolution_sub_channel_drq operation.";
  let description = [{
    Placeholder for the edgetpu.transposed_convolution_sub_channel_drq operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

#endif // EDGETPU_OPS
