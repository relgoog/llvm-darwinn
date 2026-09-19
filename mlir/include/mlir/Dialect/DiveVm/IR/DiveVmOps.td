//===- DiveVmOps.td - DiveVM dialect operation definitions -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the operation set for the dive_vm dialect observed in
// dive binaries. Each operation is a boilerplate placeholder carrying generic
// operands, results, and attributes until finer semantics are specified.
//
//===----------------------------------------------------------------------===//

#ifndef DIVEVM_OPS
#define DIVEVM_OPS

include "mlir/IR/OpBase.td"
include "mlir/Dialect/DiveVm/IR/DiveVmDialect.td"

//===----------------------------------------------------------------------===//
// Operator: add
//===----------------------------------------------------------------------===//
def DiveVm_AddOp : DiveVm_Op<"add"> {
  let summary = "DiveVM add operation.";
  let description = [{
    Verifier requires exactly 2 operands. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.add to DiveRuntime_Log. No darwinn lowering forwarding to dive_vm.add evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: add_imm
//===----------------------------------------------------------------------===//
def DiveVm_AddImmOp : DiveVm_Op<"add_imm"> {
  let summary = "DiveVM add_imm operation.";
  let description = [{
    Verifier requires exactly 1 operand. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.add_imm to _ZN7silicon4dive7kernels21DiveVm_LegacyScalarOpENS1_24DiveVmLegacyScalarOpTypeEiPKPhPKlPKNS1_19DiveVmPrimitiveTypeEPKiPKbPKfSC_bi. No darwinn lowering forwarding to dive_vm.add_imm evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: address_of_activation
//===----------------------------------------------------------------------===//
def DiveVm_AddressOfActivationOp : DiveVm_Op<"address_of_activation"> {
  let summary = "DiveVM address_of_activation operation.";
  let description = [{
    Verifier accepts any operand count. LowerCopySlice DynamicSliceLowering creates dive_vm.address_of_activation from darwinn.dynamic_slice, darwinn.dynamic_slice_with_copy, darwinn.dynamic_update_slice, and darwinn.dynamic_update_slice_with_offsets to feed dive_vm.extract_slice or dive_vm.insert_slice. No callee symbol for dive_vm.address_of_activation evidenced in convert-dive-vm-to-llvm or convert-dive-vm-to-memref.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: address_of_input_activation
//===----------------------------------------------------------------------===//
def DiveVm_AddressOfInputActivationOp : DiveVm_Op<"address_of_input_activation"> {
  let summary = "DiveVM address_of_input_activation operation.";
  let description = [{
    Verifier accepts any operand count. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.address_of_input_activation to DiveVm_GetAddressOfInputActivation. No darwinn lowering forwarding to dive_vm.address_of_input_activation evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: address_of_output_activation
//===----------------------------------------------------------------------===//
def DiveVm_AddressOfOutputActivationOp : DiveVm_Op<"address_of_output_activation"> {
  let summary = "DiveVM address_of_output_activation operation.";
  let description = [{
    Verifier accepts any operand count. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.address_of_output_activation to DiveVm_GetAddressOfOutputActivation. No darwinn lowering forwarding to dive_vm.address_of_output_activation evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: address_of_parameter
//===----------------------------------------------------------------------===//
def DiveVm_AddressOfParameterOp : DiveVm_Op<"address_of_parameter"> {
  let summary = "DiveVM address_of_parameter operation.";
  let description = [{
    Verifier accepts any operand count. No callee symbol for dive_vm.address_of_parameter evidenced in convert-dive-vm-to-llvm or convert-dive-vm-to-memref, and no darwinn lowering forwarding to it evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: address_of_parameter_region
//===----------------------------------------------------------------------===//
def DiveVm_AddressOfParameterRegionOp : DiveVm_Op<"address_of_parameter_region"> {
  let summary = "DiveVM address_of_parameter_region operation.";
  let description = [{
    Verifier accepts any operand count. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.address_of_parameter_region to DiveVm_GetAddressOfParameterRegion. No darwinn lowering forwarding to dive_vm.address_of_parameter_region evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: address_of_scratch
//===----------------------------------------------------------------------===//
def DiveVm_AddressOfScratchOp : DiveVm_Op<"address_of_scratch"> {
  let summary = "DiveVM address_of_scratch operation.";
  let description = [{
    Verifier accepts any operand count. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.address_of_scratch to DiveVm_GetAddressOfScratch. No darwinn lowering forwarding to dive_vm.address_of_scratch evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: address_space
//===----------------------------------------------------------------------===//
def DiveVm_AddressSpaceOp : DiveVm_Op<"address_space"> {
  let summary = "DiveVM address_space operation.";
  let description = [{
    Verifier accepts any operand count. Descriptor op carrying indices or metadata with no kernel shape evidenced. No callee symbol for dive_vm.address_space evidenced in convert-dive-vm-to-llvm or convert-dive-vm-to-memref.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: allocate
//===----------------------------------------------------------------------===//
def DiveVm_AllocateOp : DiveVm_Op<"allocate"> {
  let summary = "DiveVM allocate operation.";
  let description = [{
    Verifier accepts any operand count. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.allocate to DiveRuntime_Allocate. No darwinn lowering forwarding to dive_vm.allocate evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: arithmetic_left_shift
//===----------------------------------------------------------------------===//
def DiveVm_ArithmeticLeftShiftOp : DiveVm_Op<"arithmetic_left_shift"> {
  let summary = "DiveVM arithmetic_left_shift operation.";
  let description = [{
    Verifier requires exactly 2 operands. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.arithmetic_left_shift to _ZN7silicon4dive7kernels21DiveVm_LegacyScalarOpENS1_24DiveVmLegacyScalarOpTypeEiPKPhPKlPKNS1_19DiveVmPrimitiveTypeEPKiPKbPKfSC_bi. No darwinn lowering forwarding to dive_vm.arithmetic_left_shift evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: arithmetic_left_shift_imm
//===----------------------------------------------------------------------===//
def DiveVm_ArithmeticLeftShiftImmOp : DiveVm_Op<"arithmetic_left_shift_imm"> {
  let summary = "DiveVM arithmetic_left_shift_imm operation.";
  let description = [{
    Verifier requires exactly 1 operand. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.arithmetic_left_shift_imm to _ZN7silicon4dive7kernels21DiveVm_LegacyScalarOpENS1_24DiveVmLegacyScalarOpTypeEiPKPhPKlPKNS1_19DiveVmPrimitiveTypeEPKiPKbPKfSC_bi. No darwinn lowering forwarding to dive_vm.arithmetic_left_shift_imm evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: arithmetic_right_shift
//===----------------------------------------------------------------------===//
def DiveVm_ArithmeticRightShiftOp : DiveVm_Op<"arithmetic_right_shift"> {
  let summary = "DiveVM arithmetic_right_shift operation.";
  let description = [{
    Verifier requires exactly 2 operands. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.arithmetic_right_shift to _ZN7silicon4dive7kernels21DiveVm_LegacyScalarOpENS1_24DiveVmLegacyScalarOpTypeEiPKPhPKlPKNS1_19DiveVmPrimitiveTypeEPKiPKbPKfSC_bi. No darwinn lowering forwarding to dive_vm.arithmetic_right_shift evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: arithmetic_right_shift_imm
//===----------------------------------------------------------------------===//
def DiveVm_ArithmeticRightShiftImmOp : DiveVm_Op<"arithmetic_right_shift_imm"> {
  let summary = "DiveVM arithmetic_right_shift_imm operation.";
  let description = [{
    Verifier requires exactly 1 operand. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.arithmetic_right_shift_imm to _ZN7silicon4dive7kernels21DiveVm_LegacyScalarOpENS1_24DiveVmLegacyScalarOpTypeEiPKPhPKlPKNS1_19DiveVmPrimitiveTypeEPKiPKbPKfSC_bi. No darwinn lowering forwarding to dive_vm.arithmetic_right_shift_imm evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: benchmark
//===----------------------------------------------------------------------===//
def DiveVm_BenchmarkOp : DiveVm_Op<"benchmark"> {
  let summary = "DiveVM benchmark operation.";
  let description = [{
    Verifier accepts any operand count. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.benchmark to DiveVm_Benchmark. No darwinn lowering forwarding to dive_vm.benchmark evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: bitcast
//===----------------------------------------------------------------------===//
def DiveVm_BitcastOp : DiveVm_Op<"bitcast"> {
  let summary = "DiveVM bitcast operation.";
  let description = [{
    Verifier requires exactly 1 operand. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.bitcast to DiveVm_Cast. DwcBitcastConvertPass forwards darwinn.bitcast, darwinn.reinterpret_cast, and darwinn.hl_bitcast to dive_vm.bitcast.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: bitwise_and
//===----------------------------------------------------------------------===//
def DiveVm_BitwiseAndOp : DiveVm_Op<"bitwise_and"> {
  let summary = "DiveVM bitwise_and operation.";
  let description = [{
    Verifier requires exactly 2 operands. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.bitwise_and to _ZN7silicon4dive7kernels21DiveVm_LegacyScalarOpENS1_24DiveVmLegacyScalarOpTypeEiPKPhPKlPKNS1_19DiveVmPrimitiveTypeEPKiPKbPKfSC_bi. No darwinn lowering forwarding to dive_vm.bitwise_and evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: bitwise_and_imm
//===----------------------------------------------------------------------===//
def DiveVm_BitwiseAndImmOp : DiveVm_Op<"bitwise_and_imm"> {
  let summary = "DiveVM bitwise_and_imm operation.";
  let description = [{
    Verifier requires exactly 1 operand. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.bitwise_and_imm to _ZN7silicon4dive7kernels21DiveVm_LegacyScalarOpENS1_24DiveVmLegacyScalarOpTypeEiPKPhPKlPKNS1_19DiveVmPrimitiveTypeEPKiPKbPKfSC_bi. No darwinn lowering forwarding to dive_vm.bitwise_and_imm evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: bitwise_or
//===----------------------------------------------------------------------===//
def DiveVm_BitwiseOrOp : DiveVm_Op<"bitwise_or"> {
  let summary = "DiveVM bitwise_or operation.";
  let description = [{
    Verifier requires exactly 2 operands. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.bitwise_or to _ZN7silicon4dive7kernels21DiveVm_LegacyScalarOpENS1_24DiveVmLegacyScalarOpTypeEiPKPhPKlPKNS1_19DiveVmPrimitiveTypeEPKiPKbPKfSC_bi. No darwinn lowering forwarding to dive_vm.bitwise_or evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: bitwise_or_imm
//===----------------------------------------------------------------------===//
def DiveVm_BitwiseOrImmOp : DiveVm_Op<"bitwise_or_imm"> {
  let summary = "DiveVM bitwise_or_imm operation.";
  let description = [{
    Verifier requires exactly 1 operand. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.bitwise_or_imm to _ZN7silicon4dive7kernels21DiveVm_LegacyScalarOpENS1_24DiveVmLegacyScalarOpTypeEiPKPhPKlPKNS1_19DiveVmPrimitiveTypeEPKiPKbPKfSC_bi. No darwinn lowering forwarding to dive_vm.bitwise_or_imm evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: bitwise_xor
//===----------------------------------------------------------------------===//
def DiveVm_BitwiseXorOp : DiveVm_Op<"bitwise_xor"> {
  let summary = "DiveVM bitwise_xor operation.";
  let description = [{
    Verifier requires exactly 2 operands. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.bitwise_xor to _ZN7silicon4dive7kernels21DiveVm_LegacyScalarOpENS1_24DiveVmLegacyScalarOpTypeEiPKPhPKlPKNS1_19DiveVmPrimitiveTypeEPKiPKbPKfSC_bi. No darwinn lowering forwarding to dive_vm.bitwise_xor evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: bitwise_xor_imm
//===----------------------------------------------------------------------===//
def DiveVm_BitwiseXorImmOp : DiveVm_Op<"bitwise_xor_imm"> {
  let summary = "DiveVM bitwise_xor_imm operation.";
  let description = [{
    Verifier requires exactly 1 operand. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.bitwise_xor_imm to _ZN7silicon4dive7kernels21DiveVm_LegacyScalarOpENS1_24DiveVmLegacyScalarOpTypeEiPKPhPKlPKNS1_19DiveVmPrimitiveTypeEPKiPKbPKfSC_bi. No darwinn lowering forwarding to dive_vm.bitwise_xor_imm evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: br
//===----------------------------------------------------------------------===//
def DiveVm_BrOp : DiveVm_Op<"br"> {
  let summary = "DiveVM br operation.";
  let description = [{
    Verifier accepts any operand count. convert-dive-vm-to-llvm and convert-dive-vm-to-memref handle dive_vm.br as control flow with a block split and no callee symbol. No darwinn lowering forwarding to dive_vm.br evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: buffer_dir
//===----------------------------------------------------------------------===//
def DiveVm_BufferDirOp : DiveVm_Op<"buffer_dir"> {
  let summary = "DiveVM buffer_dir operation.";
  let description = [{
    Verifier accepts any operand count. Descriptor op carrying indices or metadata with no kernel shape evidenced. No callee symbol for dive_vm.buffer_dir evidenced in convert-dive-vm-to-llvm or convert-dive-vm-to-memref.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: cache_clean_invalidate
//===----------------------------------------------------------------------===//
def DiveVm_CacheCleanInvalidateOp : DiveVm_Op<"cache_clean_invalidate"> {
  let summary = "DiveVM cache_clean_invalidate operation.";
  let description = [{
    Verifier accepts any operand count. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.cache_clean_invalidate to DiveSystem_CacheCleanInvalidate. No darwinn lowering forwarding to dive_vm.cache_clean_invalidate evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: cast
//===----------------------------------------------------------------------===//
def DiveVm_CastOp : DiveVm_Op<"cast"> {
  let summary = "DiveVM cast operation.";
  let description = [{
    Verifier requires exactly 1 operand. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.cast to DiveVm_Cast. applyDwcLowerConvertTrunc forwards darwinn.convert, darwinn.cast_in, darwinn.cast_out, and darwinn.materialize_cast to dive_vm.cast, and the LowerConvert patterns lower darwinn.convert, darwinn.cast_in, and darwinn.cast_out to it.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: chunk_offsets
//===----------------------------------------------------------------------===//
def DiveVm_ChunkOffsetsOp : DiveVm_Op<"chunk_offsets"> {
  let summary = "DiveVM chunk_offsets operation.";
  let description = [{
    Verifier accepts any operand count. Descriptor op carrying indices or metadata with no kernel shape evidenced. No callee symbol for dive_vm.chunk_offsets evidenced in convert-dive-vm-to-llvm or convert-dive-vm-to-memref.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: chunk_sizes
//===----------------------------------------------------------------------===//
def DiveVm_ChunkSizesOp : DiveVm_Op<"chunk_sizes"> {
  let summary = "DiveVM chunk_sizes operation.";
  let description = [{
    Verifier accepts any operand count. Descriptor op carrying indices or metadata with no kernel shape evidenced. No callee symbol for dive_vm.chunk_sizes evidenced in convert-dive-vm-to-llvm or convert-dive-vm-to-memref.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: compute_norm_stats_for_rkhy
//===----------------------------------------------------------------------===//
def DiveVm_ComputeNormStatsForRkhyOp : DiveVm_Op<"compute_norm_stats_for_rkhy"> {
  let summary = "DiveVM compute_norm_stats_for_rkhy operation.";
  let description = [{
    Verifier accepts any operand count. No callee symbol for dive_vm.compute_norm_stats_for_rkhy evidenced in convert-dive-vm-to-llvm or convert-dive-vm-to-memref, and no darwinn lowering forwarding to it evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: cond_br
//===----------------------------------------------------------------------===//
def DiveVm_CondBrOp : DiveVm_Op<"cond_br"> {
  let summary = "DiveVM cond_br operation.";
  let description = [{
    Verifier accepts any operand count. convert-dive-vm-to-llvm and convert-dive-vm-to-memref handle dive_vm.cond_br as control flow with a block split and no callee symbol. No darwinn lowering forwarding to dive_vm.cond_br evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: const
//===----------------------------------------------------------------------===//
def DiveVm_ConstOp : DiveVm_Op<"const"> {
  let summary = "DiveVM const operation.";
  let description = [{
    Verifier accepts any operand count. applyDwcLowerConstInline forwards arith.constant and darwinn.constant_generator to dive_vm.const. No callee symbol for dive_vm.const evidenced in convert-dive-vm-to-llvm or convert-dive-vm-to-memref.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: const_bytes
//===----------------------------------------------------------------------===//
def DiveVm_ConstBytesOp : DiveVm_Op<"const_bytes"> {
  let summary = "DiveVM const_bytes operation.";
  let description = [{
    Verifier accepts any operand count. No callee symbol for dive_vm.const_bytes evidenced in convert-dive-vm-to-llvm or convert-dive-vm-to-memref, and no darwinn lowering forwarding to it evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: copy
//===----------------------------------------------------------------------===//
def DiveVm_CopyOp : DiveVm_Op<"copy"> {
  let summary = "DiveVM copy operation.";
  let description = [{
    Verifier requires exactly 1 operand. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.copy to _ZN9platforms7darwinn4dive11runtime_lib10MemCpyPerfEPhPKhi. applyDwcLowerCopyLike forwards darwinn.copy_op, darwinn.copy_from_host, and darwinn.copy_using_wide to dive_vm.copy, and LowerCopySlice CopyOpLowering lowers darwinn.copy_op to it.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: copy_imm
//===----------------------------------------------------------------------===//
def DiveVm_CopyImmOp : DiveVm_Op<"copy_imm"> {
  let summary = "DiveVM copy_imm operation.";
  let description = [{
    Verifier requires exactly 1 operand. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.copy_imm to _ZN9platforms7darwinn4dive11runtime_lib10MemCpyPerfEPhPKhi. No darwinn lowering forwarding to dive_vm.copy_imm evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: core_id
//===----------------------------------------------------------------------===//
def DiveVm_CoreIdOp : DiveVm_Op<"core_id"> {
  let summary = "DiveVM core_id operation.";
  let description = [{
    Verifier accepts any operand count. No callee symbol for dive_vm.core_id evidenced in convert-dive-vm-to-llvm or convert-dive-vm-to-memref, and no darwinn lowering forwarding to it evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: cuda_emu_custom_op
//===----------------------------------------------------------------------===//
def DiveVm_CudaEmuCustomOpOp : DiveVm_Op<"cuda_emu_custom_op"> {
  let summary = "DiveVM cuda_emu_custom_op operation.";
  let description = [{
    Verifier accepts any operand count. No callee symbol for dive_vm.cuda_emu_custom_op evidenced in convert-dive-vm-to-llvm or convert-dive-vm-to-memref, and no darwinn lowering forwarding to it evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: cumsum
//===----------------------------------------------------------------------===//
def DiveVm_CumsumOp : DiveVm_Op<"cumsum"> {
  let summary = "DiveVM cumsum operation.";
  let description = [{
    Verifier accepts any operand count. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.cumsum to _ZN7silicon4dive7kernels6CumsumERKNS0_11interpreter14ResolvedTensorEibbRS3_. No darwinn lowering forwarding to dive_vm.cumsum evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: custom_on_addresses
//===----------------------------------------------------------------------===//
def DiveVm_CustomOnAddressesOp : DiveVm_Op<"custom_on_addresses"> {
  let summary = "DiveVM custom_on_addresses operation.";
  let description = [{
    Verifier accepts any operand count. No callee symbol for dive_vm.custom_on_addresses evidenced in convert-dive-vm-to-llvm or convert-dive-vm-to-memref, and no darwinn lowering forwarding to it evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: custom_on_buffers
//===----------------------------------------------------------------------===//
def DiveVm_CustomOnBuffersOp : DiveVm_Op<"custom_on_buffers"> {
  let summary = "DiveVM custom_on_buffers operation.";
  let description = [{
    Verifier accepts any operand count. No callee symbol for dive_vm.custom_on_buffers evidenced in convert-dive-vm-to-llvm or convert-dive-vm-to-memref, and no darwinn lowering forwarding to it evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: disable_itc_tracing
//===----------------------------------------------------------------------===//
def DiveVm_DisableItcTracingOp : DiveVm_Op<"disable_itc_tracing"> {
  let summary = "DiveVM disable_itc_tracing operation.";
  let description = [{
    Verifier accepts any operand count. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.disable_itc_tracing to DiveItcTracing_Disable. No darwinn lowering forwarding to dive_vm.disable_itc_tracing evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: dispatch_hardware_instruction
//===----------------------------------------------------------------------===//
def DiveVm_DispatchHardwareInstructionOp : DiveVm_Op<"dispatch_hardware_instruction"> {
  let summary = "DiveVM dispatch_hardware_instruction operation.";
  let description = [{
    Verifier accepts any operand count. No callee symbol for dive_vm.dispatch_hardware_instruction evidenced in convert-dive-vm-to-llvm or convert-dive-vm-to-memref, and no darwinn lowering forwarding to it evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: div
//===----------------------------------------------------------------------===//
def DiveVm_DivOp : DiveVm_Op<"div"> {
  let summary = "DiveVM div operation.";
  let description = [{
    Verifier requires exactly 2 operands. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.div to _ZN7silicon4dive7kernels21DiveVm_LegacyScalarOpENS1_24DiveVmLegacyScalarOpTypeEiPKPhPKlPKNS1_19DiveVmPrimitiveTypeEPKiPKbPKfSC_bi. No darwinn lowering forwarding to dive_vm.div evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: div_imm
//===----------------------------------------------------------------------===//
def DiveVm_DivImmOp : DiveVm_Op<"div_imm"> {
  let summary = "DiveVM div_imm operation.";
  let description = [{
    Verifier requires exactly 1 operand. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.div_imm to _ZN7silicon4dive7kernels21DiveVm_LegacyScalarOpENS1_24DiveVmLegacyScalarOpTypeEiPKPhPKlPKNS1_19DiveVmPrimitiveTypeEPKiPKbPKfSC_bi. No darwinn lowering forwarding to dive_vm.div_imm evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: dive_vm_module
//===----------------------------------------------------------------------===//
def DiveVm_DiveVmModuleOp : DiveVm_Op<"dive_vm_module"> {
  let summary = "DiveVM dive_vm_module operation.";
  let description = [{
    Verifier accepts any operand count. Module container op with no kernel shape evidenced. No callee symbol for dive_vm.dive_vm_module evidenced in convert-dive-vm-to-llvm or convert-dive-vm-to-memref, and no darwinn lowering forwarding to it evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: dma_hint
//===----------------------------------------------------------------------===//
def DiveVm_DmaHintOp : DiveVm_Op<"dma_hint"> {
  let summary = "DiveVM dma_hint operation.";
  let description = [{
    Verifier requires at least 2 operands. No callee symbol for dive_vm.dma_hint evidenced in convert-dive-vm-to-llvm or convert-dive-vm-to-memref, and no darwinn lowering forwarding to it evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: dma_queue
//===----------------------------------------------------------------------===//
def DiveVm_DmaQueueOp : DiveVm_Op<"dma_queue"> {
  let summary = "DiveVM dma_queue operation.";
  let description = [{
    Verifier requires at least 2 operands. No callee symbol for dive_vm.dma_queue evidenced in convert-dive-vm-to-llvm or convert-dive-vm-to-memref, and no darwinn lowering forwarding to it evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: dtc_mode
//===----------------------------------------------------------------------===//
def DiveVm_DtcModeOp : DiveVm_Op<"dtc_mode"> {
  let summary = "DiveVM dtc_mode operation.";
  let description = [{
    Verifier accepts any operand count. Descriptor op carrying indices or metadata with no kernel shape evidenced. No callee symbol for dive_vm.dtc_mode evidenced in convert-dive-vm-to-llvm or convert-dive-vm-to-memref.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: dvfs
//===----------------------------------------------------------------------===//
def DiveVm_DvfsOp : DiveVm_Op<"dvfs"> {
  let summary = "DiveVM dvfs operation.";
  let description = [{
    Verifier accepts any operand count. Descriptor op carrying indices or metadata with no kernel shape evidenced. No callee symbol for dive_vm.dvfs evidenced in convert-dive-vm-to-llvm or convert-dive-vm-to-memref.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: dynamic_slice_y
//===----------------------------------------------------------------------===//
def DiveVm_DynamicSliceYOp : DiveVm_Op<"dynamic_slice_y"> {
  let summary = "DiveVM dynamic_slice_y operation.";
  let description = [{
    Verifier accepts any operand count. No callee symbol for dive_vm.dynamic_slice_y evidenced in convert-dive-vm-to-llvm or convert-dive-vm-to-memref, and no darwinn lowering forwarding to it evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: enable_itc_tracing
//===----------------------------------------------------------------------===//
def DiveVm_EnableItcTracingOp : DiveVm_Op<"enable_itc_tracing"> {
  let summary = "DiveVM enable_itc_tracing operation.";
  let description = [{
    Verifier accepts any operand count. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.enable_itc_tracing to DiveItcTracing_Enable. No darwinn lowering forwarding to dive_vm.enable_itc_tracing evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: epilogue
//===----------------------------------------------------------------------===//
def DiveVm_EpilogueOp : DiveVm_Op<"epilogue"> {
  let summary = "DiveVM epilogue operation.";
  let description = [{
    Verifier accepts any operand count. Descriptor op carrying indices or metadata with no kernel shape evidenced. No callee symbol for dive_vm.epilogue evidenced in convert-dive-vm-to-llvm or convert-dive-vm-to-memref.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: equal
//===----------------------------------------------------------------------===//
def DiveVm_EqualOp : DiveVm_Op<"equal"> {
  let summary = "DiveVM equal operation.";
  let description = [{
    Verifier requires exactly 2 operands. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.equal to _ZN7silicon4dive7kernels21DiveVm_LegacyScalarOpENS1_24DiveVmLegacyScalarOpTypeEiPKPhPKlPKNS1_19DiveVmPrimitiveTypeEPKiPKbPKfSC_bi. No darwinn lowering forwarding to dive_vm.equal evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: equal_imm
//===----------------------------------------------------------------------===//
def DiveVm_EqualImmOp : DiveVm_Op<"equal_imm"> {
  let summary = "DiveVM equal_imm operation.";
  let description = [{
    Verifier requires exactly 1 operand. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.equal_imm to _ZN7silicon4dive7kernels21DiveVm_LegacyScalarOpENS1_24DiveVmLegacyScalarOpTypeEiPKPhPKlPKNS1_19DiveVmPrimitiveTypeEPKiPKbPKfSC_bi. No darwinn lowering forwarding to dive_vm.equal_imm evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: executable
//===----------------------------------------------------------------------===//
def DiveVm_ExecutableOp : DiveVm_Op<"executable"> {
  let summary = "DiveVM executable operation.";
  let description = [{
    Verifier accepts any operand count. Descriptor op carrying indices or metadata with no kernel shape evidenced. No callee symbol for dive_vm.executable evidenced in convert-dive-vm-to-llvm or convert-dive-vm-to-memref.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: extract_slice
//===----------------------------------------------------------------------===//
def DiveVm_ExtractSliceOp : DiveVm_Op<"extract_slice"> {
  let summary = "DiveVM extract_slice operation.";
  let description = [{
    Verifier accepts any operand count. LowerCopySlice DynamicSliceLowering creates dive_vm.extract_slice from darwinn.dynamic_slice and darwinn.dynamic_slice_with_copy with slice-offset generator attributes. No callee symbol for dive_vm.extract_slice evidenced in convert-dive-vm-to-llvm or convert-dive-vm-to-memref.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: fill
//===----------------------------------------------------------------------===//
def DiveVm_FillOp : DiveVm_Op<"fill"> {
  let summary = "DiveVM fill operation.";
  let description = [{
    Verifier requires exactly 1 operand. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.fill to _ZN7silicon4dive7kernels4FillERKNS0_11interpreter14ResolvedTensorERS3_. A darwinn lowering forwards darwinn.fill to dive_vm.fill.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: gather
//===----------------------------------------------------------------------===//
def DiveVm_GatherOp : DiveVm_Op<"gather"> {
  let summary = "DiveVM gather operation.";
  let description = [{
    Verifier requires at least 2 operands. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.gather to DiveVm_HIBGatherEditE32. applyDwcLowerGatherOob forwards darwinn.gather, darwinn.gather_copy, and darwinn.hib_gather to dive_vm.gather with oob_zero_fill, and LowerCopySlice GatherLowering lowers darwinn.gather, darwinn.gather_copy, and darwinn.hib_gather to it.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: gather_nd
//===----------------------------------------------------------------------===//
def DiveVm_GatherNdOp : DiveVm_Op<"gather_nd"> {
  let summary = "DiveVM gather_nd operation.";
  let description = [{
    Verifier requires at least 2 operands. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.gather_nd to _ZN7silicon4dive7kernels8GatherNdERKNS0_11interpreter14ResolvedTensorES5_iRNS2_16UnresolvedTensorE. No darwinn lowering forwarding to dive_vm.gather_nd evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: get_const
//===----------------------------------------------------------------------===//
def DiveVm_GetConstOp : DiveVm_Op<"get_const"> {
  let summary = "DiveVM get_const operation.";
  let description = [{
    Verifier accepts any operand count. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.get_const to DiveVm_GetConst. No darwinn lowering forwarding to dive_vm.get_const evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: greater
//===----------------------------------------------------------------------===//
def DiveVm_GreaterOp : DiveVm_Op<"greater"> {
  let summary = "DiveVM greater operation.";
  let description = [{
    Verifier requires exactly 2 operands. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.greater to _ZN7silicon4dive7kernels21DiveVm_LegacyScalarOpENS1_24DiveVmLegacyScalarOpTypeEiPKPhPKlPKNS1_19DiveVmPrimitiveTypeEPKiPKbPKfSC_bi. No darwinn lowering forwarding to dive_vm.greater evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: greater_equal
//===----------------------------------------------------------------------===//
def DiveVm_GreaterEqualOp : DiveVm_Op<"greater_equal"> {
  let summary = "DiveVM greater_equal operation.";
  let description = [{
    Verifier requires exactly 2 operands. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.greater_equal to _ZN7silicon4dive7kernels21DiveVm_LegacyScalarOpENS1_24DiveVmLegacyScalarOpTypeEiPKPhPKlPKNS1_19DiveVmPrimitiveTypeEPKiPKbPKfSC_bi. No darwinn lowering forwarding to dive_vm.greater_equal evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: greater_equal_imm
//===----------------------------------------------------------------------===//
def DiveVm_GreaterEqualImmOp : DiveVm_Op<"greater_equal_imm"> {
  let summary = "DiveVM greater_equal_imm operation.";
  let description = [{
    Verifier requires exactly 1 operand. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.greater_equal_imm to _ZN7silicon4dive7kernels21DiveVm_LegacyScalarOpENS1_24DiveVmLegacyScalarOpTypeEiPKPhPKlPKNS1_19DiveVmPrimitiveTypeEPKiPKbPKfSC_bi. No darwinn lowering forwarding to dive_vm.greater_equal_imm evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: greater_imm
//===----------------------------------------------------------------------===//
def DiveVm_GreaterImmOp : DiveVm_Op<"greater_imm"> {
  let summary = "DiveVM greater_imm operation.";
  let description = [{
    Verifier requires exactly 1 operand. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.greater_imm to _ZN7silicon4dive7kernels21DiveVm_LegacyScalarOpENS1_24DiveVmLegacyScalarOpTypeEiPKPhPKlPKNS1_19DiveVmPrimitiveTypeEPKiPKbPKfSC_bi. No darwinn lowering forwarding to dive_vm.greater_imm evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: has_idp
//===----------------------------------------------------------------------===//
def DiveVm_HasIdpOp : DiveVm_Op<"has_idp"> {
  let summary = "DiveVM has_idp operation.";
  let description = [{
    Verifier accepts any operand count. Descriptor op carrying indices or metadata with no kernel shape evidenced. No callee symbol for dive_vm.has_idp evidenced in convert-dive-vm-to-llvm or convert-dive-vm-to-memref.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: heap_size_bytes
//===----------------------------------------------------------------------===//
def DiveVm_HeapSizeBytesOp : DiveVm_Op<"heap_size_bytes"> {
  let summary = "DiveVM heap_size_bytes operation.";
  let description = [{
    Verifier accepts any operand count. Descriptor op carrying indices or metadata with no kernel shape evidenced. No callee symbol for dive_vm.heap_size_bytes evidenced in convert-dive-vm-to-llvm or convert-dive-vm-to-memref.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: hib_gather_edit
//===----------------------------------------------------------------------===//
def DiveVm_HibGatherEditOp : DiveVm_Op<"hib_gather_edit"> {
  let summary = "DiveVM hib_gather_edit operation.";
  let description = [{
    Verifier requires at least 2 operands. No callee symbol for dive_vm.hib_gather_edit evidenced in convert-dive-vm-to-llvm or convert-dive-vm-to-memref, and no darwinn lowering forwarding to it evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: idp
//===----------------------------------------------------------------------===//
def DiveVm_IdpOp : DiveVm_Op<"idp"> {
  let summary = "DiveVM idp operation.";
  let description = [{
    Verifier accepts any operand count. Descriptor op carrying indices or metadata with no kernel shape evidenced. No callee symbol for dive_vm.idp evidenced in convert-dive-vm-to-llvm or convert-dive-vm-to-memref.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: idp_instance
//===----------------------------------------------------------------------===//
def DiveVm_IdpInstanceOp : DiveVm_Op<"idp_instance"> {
  let summary = "DiveVM idp_instance operation.";
  let description = [{
    Verifier accepts any operand count. Descriptor op carrying indices or metadata with no kernel shape evidenced. No callee symbol for dive_vm.idp_instance evidenced in convert-dive-vm-to-llvm or convert-dive-vm-to-memref.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: insert_slice
//===----------------------------------------------------------------------===//
def DiveVm_InsertSliceOp : DiveVm_Op<"insert_slice"> {
  let summary = "DiveVM insert_slice operation.";
  let description = [{
    Verifier accepts any operand count. LowerCopySlice DynamicSliceLowering creates dive_vm.insert_slice from darwinn.dynamic_update_slice and darwinn.dynamic_update_slice_with_offsets with slice-offset generator attributes. No callee symbol for dive_vm.insert_slice evidenced in convert-dive-vm-to-llvm or convert-dive-vm-to-memref.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: instruction_id
//===----------------------------------------------------------------------===//
def DiveVm_InstructionIdOp : DiveVm_Op<"instruction_id"> {
  let summary = "DiveVM instruction_id operation.";
  let description = [{
    Verifier accepts any operand count. No callee symbol for dive_vm.instruction_id evidenced in convert-dive-vm-to-llvm or convert-dive-vm-to-memref, and no darwinn lowering forwarding to it evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: instruction_id_live_tensors
//===----------------------------------------------------------------------===//
def DiveVm_InstructionIdLiveTensorsOp : DiveVm_Op<"instruction_id_live_tensors"> {
  let summary = "DiveVM instruction_id_live_tensors operation.";
  let description = [{
    Verifier accepts any operand count. No callee symbol for dive_vm.instruction_id_live_tensors evidenced in convert-dive-vm-to-llvm or convert-dive-vm-to-memref, and no darwinn lowering forwarding to it evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: legacy_scalar
//===----------------------------------------------------------------------===//
def DiveVm_LegacyScalarOp : DiveVm_Op<"legacy_scalar"> {
  let summary = "DiveVM legacy_scalar operation.";
  let description = [{
    Verifier accepts any operand count. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.legacy_scalar to _ZN7silicon4dive7kernels21DiveVm_LegacyScalarOpENS1_24DiveVmLegacyScalarOpTypeEiPKPhPKlPKNS1_19DiveVmPrimitiveTypeEPKiPKbPKfSC_bi. No darwinn lowering forwarding to dive_vm.legacy_scalar evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: less
//===----------------------------------------------------------------------===//
def DiveVm_LessOp : DiveVm_Op<"less"> {
  let summary = "DiveVM less operation.";
  let description = [{
    Verifier requires exactly 2 operands. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.less to _ZN7silicon4dive7kernels21DiveVm_LegacyScalarOpENS1_24DiveVmLegacyScalarOpTypeEiPKPhPKlPKNS1_19DiveVmPrimitiveTypeEPKiPKbPKfSC_bi. No darwinn lowering forwarding to dive_vm.less evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: less_equal
//===----------------------------------------------------------------------===//
def DiveVm_LessEqualOp : DiveVm_Op<"less_equal"> {
  let summary = "DiveVM less_equal operation.";
  let description = [{
    Verifier requires exactly 2 operands. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.less_equal to _ZN7silicon4dive7kernels21DiveVm_LegacyScalarOpENS1_24DiveVmLegacyScalarOpTypeEiPKPhPKlPKNS1_19DiveVmPrimitiveTypeEPKiPKbPKfSC_bi. No darwinn lowering forwarding to dive_vm.less_equal evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: less_equal_imm
//===----------------------------------------------------------------------===//
def DiveVm_LessEqualImmOp : DiveVm_Op<"less_equal_imm"> {
  let summary = "DiveVM less_equal_imm operation.";
  let description = [{
    Verifier requires exactly 1 operand. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.less_equal_imm to _ZN7silicon4dive7kernels21DiveVm_LegacyScalarOpENS1_24DiveVmLegacyScalarOpTypeEiPKPhPKlPKNS1_19DiveVmPrimitiveTypeEPKiPKbPKfSC_bi. No darwinn lowering forwarding to dive_vm.less_equal_imm evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: less_imm
//===----------------------------------------------------------------------===//
def DiveVm_LessImmOp : DiveVm_Op<"less_imm"> {
  let summary = "DiveVM less_imm operation.";
  let description = [{
    Verifier requires exactly 1 operand. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.less_imm to _ZN7silicon4dive7kernels21DiveVm_LegacyScalarOpENS1_24DiveVmLegacyScalarOpTypeEiPKPhPKlPKNS1_19DiveVmPrimitiveTypeEPKiPKbPKfSC_bi. No darwinn lowering forwarding to dive_vm.less_imm evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: live_tensor
//===----------------------------------------------------------------------===//
def DiveVm_LiveTensorOp : DiveVm_Op<"live_tensor"> {
  let summary = "DiveVM live_tensor operation.";
  let description = [{
    Verifier accepts any operand count. Descriptor op carrying indices or metadata with no kernel shape evidenced. No callee symbol for dive_vm.live_tensor evidenced in convert-dive-vm-to-llvm or convert-dive-vm-to-memref.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: load
//===----------------------------------------------------------------------===//
def DiveVm_LoadOp : DiveVm_Op<"load"> {
  let summary = "DiveVM load operation.";
  let description = [{
    Verifier requires exactly 1 operand. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.load to _ZN9platforms7darwinn4dive11runtime_lib10MemCpyPerfEPhPKhi. No darwinn lowering forwarding to dive_vm.load evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: load_indirect
//===----------------------------------------------------------------------===//
def DiveVm_LoadIndirectOp : DiveVm_Op<"load_indirect"> {
  let summary = "DiveVM load_indirect operation.";
  let description = [{
    Verifier requires exactly 1 operand. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.load_indirect to _ZN9platforms7darwinn4dive11runtime_lib10MemCpyPerfEPhPKhi. No darwinn lowering forwarding to dive_vm.load_indirect evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: logical_and
//===----------------------------------------------------------------------===//
def DiveVm_LogicalAndOp : DiveVm_Op<"logical_and"> {
  let summary = "DiveVM logical_and operation.";
  let description = [{
    Verifier requires exactly 2 operands. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.logical_and to _ZN7silicon4dive7kernels21DiveVm_LegacyScalarOpENS1_24DiveVmLegacyScalarOpTypeEiPKPhPKlPKNS1_19DiveVmPrimitiveTypeEPKiPKbPKfSC_bi. No darwinn lowering forwarding to dive_vm.logical_and evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: logical_and_imm
//===----------------------------------------------------------------------===//
def DiveVm_LogicalAndImmOp : DiveVm_Op<"logical_and_imm"> {
  let summary = "DiveVM logical_and_imm operation.";
  let description = [{
    Verifier requires exactly 1 operand. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.logical_and_imm to _ZN7silicon4dive7kernels21DiveVm_LegacyScalarOpENS1_24DiveVmLegacyScalarOpTypeEiPKPhPKlPKNS1_19DiveVmPrimitiveTypeEPKiPKbPKfSC_bi. No darwinn lowering forwarding to dive_vm.logical_and_imm evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: logical_right_shift
//===----------------------------------------------------------------------===//
def DiveVm_LogicalRightShiftOp : DiveVm_Op<"logical_right_shift"> {
  let summary = "DiveVM logical_right_shift operation.";
  let description = [{
    Verifier requires exactly 2 operands. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.logical_right_shift to _ZN7silicon4dive7kernels21DiveVm_LegacyScalarOpENS1_24DiveVmLegacyScalarOpTypeEiPKPhPKlPKNS1_19DiveVmPrimitiveTypeEPKiPKbPKfSC_bi. No darwinn lowering forwarding to dive_vm.logical_right_shift evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: logical_right_shift_imm
//===----------------------------------------------------------------------===//
def DiveVm_LogicalRightShiftImmOp : DiveVm_Op<"logical_right_shift_imm"> {
  let summary = "DiveVM logical_right_shift_imm operation.";
  let description = [{
    Verifier requires exactly 1 operand. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.logical_right_shift_imm to _ZN7silicon4dive7kernels21DiveVm_LegacyScalarOpENS1_24DiveVmLegacyScalarOpTypeEiPKPhPKlPKNS1_19DiveVmPrimitiveTypeEPKiPKbPKfSC_bi. No darwinn lowering forwarding to dive_vm.logical_right_shift_imm evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: mark_dive_for_host_resident
//===----------------------------------------------------------------------===//
def DiveVm_MarkDiveForHostResidentOp : DiveVm_Op<"mark_dive_for_host_resident"> {
  let summary = "DiveVM mark_dive_for_host_resident operation.";
  let description = [{
    Verifier accepts any operand count. Descriptor op carrying indices or metadata with no kernel shape evidenced. No callee symbol for dive_vm.mark_dive_for_host_resident evidenced in convert-dive-vm-to-llvm or convert-dive-vm-to-memref.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: mark_value_with_shape
//===----------------------------------------------------------------------===//
def DiveVm_MarkValueWithShapeOp : DiveVm_Op<"mark_value_with_shape"> {
  let summary = "DiveVM mark_value_with_shape operation.";
  let description = [{
    Verifier accepts any operand count. Descriptor op carrying indices or metadata with no kernel shape evidenced. No callee symbol for dive_vm.mark_value_with_shape evidenced in convert-dive-vm-to-llvm or convert-dive-vm-to-memref.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: mask_indices
//===----------------------------------------------------------------------===//
def DiveVm_MaskIndicesOp : DiveVm_Op<"mask_indices"> {
  let summary = "DiveVM mask_indices operation.";
  let description = [{
    Verifier accepts any operand count. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.mask_indices to DiveVm_MaskIndices. A darwinn lowering forwards darwinn.mask_indices to dive_vm.mask_indices.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: max
//===----------------------------------------------------------------------===//
def DiveVm_MaxOp : DiveVm_Op<"max"> {
  let summary = "DiveVM max operation.";
  let description = [{
    Verifier requires exactly 2 operands. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.max to _ZN7silicon4dive7kernels21DiveVm_LegacyScalarOpENS1_24DiveVmLegacyScalarOpTypeEiPKPhPKlPKNS1_19DiveVmPrimitiveTypeEPKiPKbPKfSC_bi. No darwinn lowering forwarding to dive_vm.max evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: max_imm
//===----------------------------------------------------------------------===//
def DiveVm_MaxImmOp : DiveVm_Op<"max_imm"> {
  let summary = "DiveVM max_imm operation.";
  let description = [{
    Verifier requires exactly 1 operand. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.max_imm to _ZN7silicon4dive7kernels21DiveVm_LegacyScalarOpENS1_24DiveVmLegacyScalarOpTypeEiPKPhPKlPKNS1_19DiveVmPrimitiveTypeEPKiPKbPKfSC_bi. No darwinn lowering forwarding to dive_vm.max_imm evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: min
//===----------------------------------------------------------------------===//
def DiveVm_MinOp : DiveVm_Op<"min"> {
  let summary = "DiveVM min operation.";
  let description = [{
    Verifier requires exactly 2 operands. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.min to _ZN7silicon4dive7kernels21DiveVm_LegacyScalarOpENS1_24DiveVmLegacyScalarOpTypeEiPKPhPKlPKNS1_19DiveVmPrimitiveTypeEPKiPKbPKfSC_bi. No darwinn lowering forwarding to dive_vm.min evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: min_imm
//===----------------------------------------------------------------------===//
def DiveVm_MinImmOp : DiveVm_Op<"min_imm"> {
  let summary = "DiveVM min_imm operation.";
  let description = [{
    Verifier requires exactly 1 operand. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.min_imm to _ZN7silicon4dive7kernels21DiveVm_LegacyScalarOpENS1_24DiveVmLegacyScalarOpTypeEiPKPhPKlPKNS1_19DiveVmPrimitiveTypeEPKiPKbPKfSC_bi. No darwinn lowering forwarding to dive_vm.min_imm evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: module_group
//===----------------------------------------------------------------------===//
def DiveVm_ModuleGroupOp : DiveVm_Op<"module_group"> {
  let summary = "DiveVM module_group operation.";
  let description = [{
    Verifier accepts any operand count. Descriptor op carrying indices or metadata with no kernel shape evidenced. No callee symbol for dive_vm.module_group evidenced in convert-dive-vm-to-llvm or convert-dive-vm-to-memref.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: mul
//===----------------------------------------------------------------------===//
def DiveVm_MulOp : DiveVm_Op<"mul"> {
  let summary = "DiveVM mul operation.";
  let description = [{
    Verifier requires exactly 2 operands. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.mul to _ZN7silicon4dive7kernels21DiveVm_LegacyScalarOpENS1_24DiveVmLegacyScalarOpTypeEiPKPhPKlPKNS1_19DiveVmPrimitiveTypeEPKiPKbPKfSC_bi. No darwinn lowering forwarding to dive_vm.mul evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: mul_imm
//===----------------------------------------------------------------------===//
def DiveVm_MulImmOp : DiveVm_Op<"mul_imm"> {
  let summary = "DiveVM mul_imm operation.";
  let description = [{
    Verifier requires exactly 1 operand. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.mul_imm to _ZN7silicon4dive7kernels21DiveVm_LegacyScalarOpENS1_24DiveVmLegacyScalarOpTypeEiPKPhPKlPKNS1_19DiveVmPrimitiveTypeEPKiPKbPKfSC_bi. No darwinn lowering forwarding to dive_vm.mul_imm evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: multinomial
//===----------------------------------------------------------------------===//
def DiveVm_MultinomialOp : DiveVm_Op<"multinomial"> {
  let summary = "DiveVM multinomial operation.";
  let description = [{
    Verifier accepts any operand count. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.multinomial to _ZN9platforms7darwinn4dive11runtime_lib18ComputeMultinomialEPfS3_iiibfiiiPv. No darwinn lowering forwarding to dive_vm.multinomial evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: not_equal
//===----------------------------------------------------------------------===//
def DiveVm_NotEqualOp : DiveVm_Op<"not_equal"> {
  let summary = "DiveVM not_equal operation.";
  let description = [{
    Verifier requires exactly 2 operands. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.not_equal to _ZN7silicon4dive7kernels21DiveVm_LegacyScalarOpENS1_24DiveVmLegacyScalarOpTypeEiPKPhPKlPKNS1_19DiveVmPrimitiveTypeEPKiPKbPKfSC_bi. No darwinn lowering forwarding to dive_vm.not_equal evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: not_equal_imm
//===----------------------------------------------------------------------===//
def DiveVm_NotEqualImmOp : DiveVm_Op<"not_equal_imm"> {
  let summary = "DiveVM not_equal_imm operation.";
  let description = [{
    Verifier requires exactly 1 operand. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.not_equal_imm to _ZN7silicon4dive7kernels21DiveVm_LegacyScalarOpENS1_24DiveVmLegacyScalarOpTypeEiPKPhPKlPKNS1_19DiveVmPrimitiveTypeEPKiPKbPKfSC_bi. No darwinn lowering forwarding to dive_vm.not_equal_imm evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: num_input_tensors
//===----------------------------------------------------------------------===//
def DiveVm_NumInputTensorsOp : DiveVm_Op<"num_input_tensors"> {
  let summary = "DiveVM num_input_tensors operation.";
  let description = [{
    Verifier accepts any operand count. Descriptor op carrying indices or metadata with no kernel shape evidenced. No callee symbol for dive_vm.num_input_tensors evidenced in convert-dive-vm-to-llvm or convert-dive-vm-to-memref.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: num_output_tensors
//===----------------------------------------------------------------------===//
def DiveVm_NumOutputTensorsOp : DiveVm_Op<"num_output_tensors"> {
  let summary = "DiveVM num_output_tensors operation.";
  let description = [{
    Verifier accepts any operand count. Descriptor op carrying indices or metadata with no kernel shape evidenced. No callee symbol for dive_vm.num_output_tensors evidenced in convert-dive-vm-to-llvm or convert-dive-vm-to-memref.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: one_hot
//===----------------------------------------------------------------------===//
def DiveVm_OneHotOp : DiveVm_Op<"one_hot"> {
  let summary = "DiveVM one_hot operation.";
  let description = [{
    Verifier accepts any operand count. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.one_hot to _ZN7silicon4dive7kernels6OneHotERKNS0_11interpreter14ResolvedTensorES5_S5_jiS5_. No darwinn lowering forwarding to dive_vm.one_hot evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: pad
//===----------------------------------------------------------------------===//
def DiveVm_PadOp : DiveVm_Op<"pad"> {
  let summary = "DiveVM pad operation.";
  let description = [{
    Verifier accepts any operand count. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.pad to _ZN7silicon4dive7kernels3PadERKNS0_11interpreter14ResolvedTensorES5_S5_RS3_. A darwinn lowering forwards darwinn.rkhy_custom_padding and darwinn.mesh_pad_slice to dive_vm.pad.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: parameter_array
//===----------------------------------------------------------------------===//
def DiveVm_ParameterArrayOp : DiveVm_Op<"parameter_array"> {
  let summary = "DiveVM parameter_array operation.";
  let description = [{
    Verifier accepts any operand count. Descriptor op carrying indices or metadata with no kernel shape evidenced. No callee symbol for dive_vm.parameter_array evidenced in convert-dive-vm-to-llvm or convert-dive-vm-to-memref.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: patch_instruction_for_strided_io
//===----------------------------------------------------------------------===//
def DiveVm_PatchInstructionForStridedIoOp : DiveVm_Op<"patch_instruction_for_strided_io"> {
  let summary = "DiveVM patch_instruction_for_strided_io operation.";
  let description = [{
    Verifier accepts any operand count. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.patch_instruction_for_strided_io to DiveVm_PatchInstructionForStridedIo. No darwinn lowering forwarding to dive_vm.patch_instruction_for_strided_io evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: perform_software_preemption_if_requested
//===----------------------------------------------------------------------===//
def DiveVm_PerformSoftwarePreemptionIfRequestedOp : DiveVm_Op<"perform_software_preemption_if_requested"> {
  let summary = "DiveVM perform_software_preemption_if_requested operation.";
  let description = [{
    Verifier accepts any operand count. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.perform_software_preemption_if_requested to DiveTpu_PerformSoftwarePreemptionIfRequested. No darwinn lowering forwarding to dive_vm.perform_software_preemption_if_requested evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: pow
//===----------------------------------------------------------------------===//
def DiveVm_PowOp : DiveVm_Op<"pow"> {
  let summary = "DiveVM pow operation.";
  let description = [{
    Verifier requires exactly 2 operands. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.pow to _ZN7silicon4dive7kernels21DiveVm_LegacyScalarOpENS1_24DiveVmLegacyScalarOpTypeEiPKPhPKlPKNS1_19DiveVmPrimitiveTypeEPKiPKbPKfSC_bi. No darwinn lowering forwarding to dive_vm.pow evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: pow_imm
//===----------------------------------------------------------------------===//
def DiveVm_PowImmOp : DiveVm_Op<"pow_imm"> {
  let summary = "DiveVM pow_imm operation.";
  let description = [{
    Verifier requires exactly 1 operand. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.pow_imm to _ZN7silicon4dive7kernels21DiveVm_LegacyScalarOpENS1_24DiveVmLegacyScalarOpTypeEiPKPhPKlPKNS1_19DiveVmPrimitiveTypeEPKiPKbPKfSC_bi. No darwinn lowering forwarding to dive_vm.pow_imm evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: print
//===----------------------------------------------------------------------===//
def DiveVm_PrintOp : DiveVm_Op<"print"> {
  let summary = "DiveVM print operation.";
  let description = [{
    Verifier accepts any operand count. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.print to DiveRuntime_Log. No darwinn lowering forwarding to dive_vm.print evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: program_tensor_mapping_table
//===----------------------------------------------------------------------===//
def DiveVm_ProgramTensorMappingTableOp : DiveVm_Op<"program_tensor_mapping_table"> {
  let summary = "DiveVM program_tensor_mapping_table operation.";
  let description = [{
    Verifier accepts any operand count. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.program_tensor_mapping_table to _ZN9platforms7darwinn4dive11runtime_lib25ProgramTensorMappingTableERKNS2_18TensorMappingTableEii. No darwinn lowering forwarding to dive_vm.program_tensor_mapping_table evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: prologue
//===----------------------------------------------------------------------===//
def DiveVm_PrologueOp : DiveVm_Op<"prologue"> {
  let summary = "DiveVM prologue operation.";
  let description = [{
    Verifier accepts any operand count. Descriptor op carrying indices or metadata with no kernel shape evidenced. No callee symbol for dive_vm.prologue evidenced in convert-dive-vm-to-llvm or convert-dive-vm-to-memref.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: put_bits
//===----------------------------------------------------------------------===//
def DiveVm_PutBitsOp : DiveVm_Op<"put_bits"> {
  let summary = "DiveVM put_bits operation.";
  let description = [{
    Verifier accepts any operand count. No callee symbol for dive_vm.put_bits evidenced in convert-dive-vm-to-llvm or convert-dive-vm-to-memref, and no darwinn lowering forwarding to it evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: read_gn_stats
//===----------------------------------------------------------------------===//
def DiveVm_ReadGnStatsOp : DiveVm_Op<"read_gn_stats"> {
  let summary = "DiveVM read_gn_stats operation.";
  let description = [{
    Verifier accepts any operand count. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.read_gn_stats to DiveDtc_ReadGnStatsSum. No darwinn lowering forwarding to dive_vm.read_gn_stats evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: reduction
//===----------------------------------------------------------------------===//
def DiveVm_ReductionOp : DiveVm_Op<"reduction"> {
  let summary = "DiveVM reduction operation.";
  let description = [{
    Verifier accepts any operand count. No callee symbol for dive_vm.reduction evidenced in convert-dive-vm-to-llvm or convert-dive-vm-to-memref, and no darwinn lowering forwarding to it evidenced. The TopKVector symbol takes (float*, float*, int*, int, int, int), not op operands, so the switches reject with a named signature error.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: reduction_type
//===----------------------------------------------------------------------===//
def DiveVm_ReductionTypeOp : DiveVm_Op<"reduction_type"> {
  let summary = "DiveVM reduction_type operation.";
  let description = [{
    Verifier accepts any operand count. Descriptor op carrying indices or metadata with no kernel shape evidenced. No callee symbol for dive_vm.reduction_type evidenced in convert-dive-vm-to-llvm or convert-dive-vm-to-memref.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: rem
//===----------------------------------------------------------------------===//
def DiveVm_RemOp : DiveVm_Op<"rem"> {
  let summary = "DiveVM rem operation.";
  let description = [{
    Verifier requires exactly 2 operands. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.rem to _ZN7silicon4dive7kernels21DiveVm_LegacyScalarOpENS1_24DiveVmLegacyScalarOpTypeEiPKPhPKlPKNS1_19DiveVmPrimitiveTypeEPKiPKbPKfSC_bi. No darwinn lowering forwarding to dive_vm.rem evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: rem_imm
//===----------------------------------------------------------------------===//
def DiveVm_RemImmOp : DiveVm_Op<"rem_imm"> {
  let summary = "DiveVM rem_imm operation.";
  let description = [{
    Verifier requires exactly 1 operand. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.rem_imm to _ZN7silicon4dive7kernels21DiveVm_LegacyScalarOpENS1_24DiveVmLegacyScalarOpTypeEiPKPhPKlPKNS1_19DiveVmPrimitiveTypeEPKiPKbPKfSC_bi. No darwinn lowering forwarding to dive_vm.rem_imm evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: roll
//===----------------------------------------------------------------------===//
def DiveVm_RollOp : DiveVm_Op<"roll"> {
  let summary = "DiveVM roll operation.";
  let description = [{
    Verifier accepts any operand count. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.roll to _ZN7silicon4dive7kernels4RollERKNS0_11interpreter14ResolvedTensorEijS5_. No darwinn lowering forwarding to dive_vm.roll evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: scalar_type
//===----------------------------------------------------------------------===//
def DiveVm_ScalarTypeOp : DiveVm_Op<"scalar_type"> {
  let summary = "DiveVM scalar_type operation.";
  let description = [{
    Verifier accepts any operand count. Descriptor op carrying indices or metadata with no kernel shape evidenced. No callee symbol for dive_vm.scalar_type evidenced in convert-dive-vm-to-llvm or convert-dive-vm-to-memref.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: scatter_nd
//===----------------------------------------------------------------------===//
def DiveVm_ScatterNdOp : DiveVm_Op<"scatter_nd"> {
  let summary = "DiveVM scatter_nd operation.";
  let description = [{
    Verifier requires at least 2 operands. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.scatter_nd to _ZN7silicon4dive7kernels9ScatterNdERKNS0_11interpreter14ResolvedTensorES5_RS3_. A darwinn lowering forwards darwinn.scatter to dive_vm.scatter_nd with oob_zero_fill.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: select
//===----------------------------------------------------------------------===//
def DiveVm_SelectOp : DiveVm_Op<"select"> {
  let summary = "DiveVM select operation.";
  let description = [{
    Verifier requires exactly 2 operands. A darwinn lowering forwards darwinn.select to dive_vm.select. No callee symbol for dive_vm.select evidenced in convert-dive-vm-to-llvm or convert-dive-vm-to-memref.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: serialized_model_format_type
//===----------------------------------------------------------------------===//
def DiveVm_SerializedModelFormatTypeOp : DiveVm_Op<"serialized_model_format_type"> {
  let summary = "DiveVM serialized_model_format_type operation.";
  let description = [{
    Verifier accepts any operand count. Descriptor op carrying indices or metadata with no kernel shape evidenced. No callee symbol for dive_vm.serialized_model_format_type evidenced in convert-dive-vm-to-llvm or convert-dive-vm-to-memref.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: set_dtc_mode
//===----------------------------------------------------------------------===//
def DiveVm_SetDtcModeOp : DiveVm_Op<"set_dtc_mode"> {
  let summary = "DiveVM set_dtc_mode operation.";
  let description = [{
    Verifier accepts any operand count. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.set_dtc_mode to DiveVm_SetDtcMode. No darwinn lowering forwarding to dive_vm.set_dtc_mode evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: shape_of_activation
//===----------------------------------------------------------------------===//
def DiveVm_ShapeOfActivationOp : DiveVm_Op<"shape_of_activation"> {
  let summary = "DiveVM shape_of_activation operation.";
  let description = [{
    Verifier accepts any operand count. Descriptor op carrying indices or metadata with no kernel shape evidenced. No callee symbol for dive_vm.shape_of_activation evidenced in convert-dive-vm-to-llvm or convert-dive-vm-to-memref.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: shared_tensor
//===----------------------------------------------------------------------===//
def DiveVm_SharedTensorOp : DiveVm_Op<"shared_tensor"> {
  let summary = "DiveVM shared_tensor operation.";
  let description = [{
    Verifier accepts any operand count. No callee symbol for dive_vm.shared_tensor evidenced in convert-dive-vm-to-llvm or convert-dive-vm-to-memref, and no darwinn lowering forwarding to it evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: sign
//===----------------------------------------------------------------------===//
def DiveVm_SignOp : DiveVm_Op<"sign"> {
  let summary = "DiveVM sign operation.";
  let description = [{
    Verifier requires exactly 1 operand. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.sign to DiveVm_Sign. No darwinn lowering forwarding to dive_vm.sign evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: store
//===----------------------------------------------------------------------===//
def DiveVm_StoreOp : DiveVm_Op<"store"> {
  let summary = "DiveVM store operation.";
  let description = [{
    Verifier requires exactly 1 operand. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.store to _ZN9platforms7darwinn4dive11runtime_lib10MemCpyPerfEPhPKhi. No darwinn lowering forwarding to dive_vm.store evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: store_indirect
//===----------------------------------------------------------------------===//
def DiveVm_StoreIndirectOp : DiveVm_Op<"store_indirect"> {
  let summary = "DiveVM store_indirect operation.";
  let description = [{
    Verifier requires exactly 1 operand. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.store_indirect to _ZN9platforms7darwinn4dive11runtime_lib10MemCpyPerfEPhPKhi. No darwinn lowering forwarding to dive_vm.store_indirect evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: sub
//===----------------------------------------------------------------------===//
def DiveVm_SubOp : DiveVm_Op<"sub"> {
  let summary = "DiveVM sub operation.";
  let description = [{
    Verifier requires exactly 2 operands. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.sub to _ZN7silicon4dive7kernels21DiveVm_LegacyScalarOpENS1_24DiveVmLegacyScalarOpTypeEiPKPhPKlPKNS1_19DiveVmPrimitiveTypeEPKiPKbPKfSC_bi. No darwinn lowering forwarding to dive_vm.sub evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: sub_imm
//===----------------------------------------------------------------------===//
def DiveVm_SubImmOp : DiveVm_Op<"sub_imm"> {
  let summary = "DiveVM sub_imm operation.";
  let description = [{
    Verifier requires exactly 1 operand. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.sub_imm to _ZN7silicon4dive7kernels21DiveVm_LegacyScalarOpENS1_24DiveVmLegacyScalarOpTypeEiPKPhPKlPKNS1_19DiveVmPrimitiveTypeEPKiPKbPKfSC_bi. No darwinn lowering forwarding to dive_vm.sub_imm evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: top_k
//===----------------------------------------------------------------------===//
def DiveVm_TopKOp : DiveVm_Op<"top_k"> {
  let summary = "DiveVM top_k operation.";
  let description = [{
    Verifier accepts any operand count. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.top_k to _ZN7silicon4dive7kernels4TopKERNS0_11interpreter14ResolvedTensorES4_S4_. A darwinn lowering forwards darwinn.index_filter to dive_vm.top_k.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: tpu_offload
//===----------------------------------------------------------------------===//
def DiveVm_TpuOffloadOp : DiveVm_Op<"tpu_offload"> {
  let summary = "DiveVM tpu_offload operation.";
  let description = [{
    Verifier accepts any operand count. No callee symbol for dive_vm.tpu_offload evidenced in convert-dive-vm-to-llvm or convert-dive-vm-to-memref, and no darwinn lowering forwarding to it evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: transition_dtc_power_island
//===----------------------------------------------------------------------===//
def DiveVm_TransitionDtcPowerIslandOp : DiveVm_Op<"transition_dtc_power_island"> {
  let summary = "DiveVM transition_dtc_power_island operation.";
  let description = [{
    Verifier accepts any operand count. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.transition_dtc_power_island to DiveDtc_TransitionDtcPowmgKllandOn. No darwinn lowering forwarding to dive_vm.transition_dtc_power_island evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: translate_sram_address
//===----------------------------------------------------------------------===//
def DiveVm_TranslateSramAddressOp : DiveVm_Op<"translate_sram_address"> {
  let summary = "DiveVM translate_sram_address operation.";
  let description = [{
    Verifier accepts any operand count. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.translate_sram_address to DiveTpu_CastSharedMemoryAddressToPointer. No darwinn lowering forwarding to dive_vm.translate_sram_address evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: view_on_address
//===----------------------------------------------------------------------===//
def DiveVm_ViewOnAddressOp : DiveVm_Op<"view_on_address"> {
  let summary = "DiveVM view_on_address operation.";
  let description = [{
    Verifier accepts any operand count. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.view_on_address to DiveTpu_CastSharedMemoryAddressToPointer. No darwinn lowering forwarding to dive_vm.view_on_address evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: wait_for_fence_completion
//===----------------------------------------------------------------------===//
def DiveVm_WaitForFenceCompletionOp : DiveVm_Op<"wait_for_fence_completion"> {
  let summary = "DiveVM wait_for_fence_completion operation.";
  let description = [{
    Verifier accepts any operand count. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.wait_for_fence_completion to DiveTpu_WaitForFenceCompletion. No darwinn lowering forwarding to dive_vm.wait_for_fence_completion evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: wait_for_power_island_transition_complete
//===----------------------------------------------------------------------===//
def DiveVm_WaitForPowerIslandTransitionCompleteOp : DiveVm_Op<"wait_for_power_island_transition_complete"> {
  let summary = "DiveVM wait_for_power_island_transition_complete operation.";
  let description = [{
    Verifier accepts any operand count. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.wait_for_power_island_transition_complete to DiveVm_WaitForPowmgKllandTransitionComplete. No darwinn lowering forwarding to dive_vm.wait_for_power_island_transition_complete evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: wait_for_rkhy_completion
//===----------------------------------------------------------------------===//
def DiveVm_WaitForRkhyCompletionOp : DiveVm_Op<"wait_for_rkhy_completion"> {
  let summary = "DiveVM wait_for_rkhy_completion operation.";
  let description = [{
    Verifier accepts any operand count. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.wait_for_rkhy_completion to DiveTpu_WaitForRkhyCompletion. No darwinn lowering forwarding to dive_vm.wait_for_rkhy_completion evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: write_dma_descriptor
//===----------------------------------------------------------------------===//
def DiveVm_WriteDmaDescriptorOp : DiveVm_Op<"write_dma_descriptor"> {
  let summary = "DiveVM write_dma_descriptor operation.";
  let description = [{
    Verifier requires at least 2 operands. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.write_dma_descriptor to DiveTpu_EnqueueDmaDescriptor. No darwinn lowering forwarding to dive_vm.write_dma_descriptor evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: write_hib_data
//===----------------------------------------------------------------------===//
def DiveVm_WriteHibDataOp : DiveVm_Op<"write_hib_data"> {
  let summary = "DiveVM write_hib_data operation.";
  let description = [{
    Verifier accepts any operand count. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.write_hib_data to DiveTpu_WriteHibData. No darwinn lowering forwarding to dive_vm.write_hib_data evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: write_scalar_arch_register
//===----------------------------------------------------------------------===//
def DiveVm_WriteScalarArchRegisterOp : DiveVm_Op<"write_scalar_arch_register"> {
  let summary = "DiveVM write_scalar_arch_register operation.";
  let description = [{
    Verifier accepts any operand count. convert-dive-vm-to-llvm and convert-dive-vm-to-memref map dive_vm.write_scalar_arch_register to DiveTpu_WriteScalarArchRegister. No darwinn lowering forwarding to dive_vm.write_scalar_arch_register evidenced.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

#endif // DIVEVM_OPS
