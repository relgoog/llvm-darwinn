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
    Placeholder for the dive_vm.add operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: add_imm
//===----------------------------------------------------------------------===//
def DiveVm_AddImmOp : DiveVm_Op<"add_imm"> {
  let summary = "DiveVM add_imm operation.";
  let description = [{
    Placeholder for the dive_vm.add_imm operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: address_of_activation
//===----------------------------------------------------------------------===//
def DiveVm_AddressOfActivationOp : DiveVm_Op<"address_of_activation"> {
  let summary = "DiveVM address_of_activation operation.";
  let description = [{
    Placeholder for the dive_vm.address_of_activation operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: address_of_input_activation
//===----------------------------------------------------------------------===//
def DiveVm_AddressOfInputActivationOp : DiveVm_Op<"address_of_input_activation"> {
  let summary = "DiveVM address_of_input_activation operation.";
  let description = [{
    Placeholder for the dive_vm.address_of_input_activation operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: address_of_output_activation
//===----------------------------------------------------------------------===//
def DiveVm_AddressOfOutputActivationOp : DiveVm_Op<"address_of_output_activation"> {
  let summary = "DiveVM address_of_output_activation operation.";
  let description = [{
    Placeholder for the dive_vm.address_of_output_activation operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: address_of_parameter
//===----------------------------------------------------------------------===//
def DiveVm_AddressOfParameterOp : DiveVm_Op<"address_of_parameter"> {
  let summary = "DiveVM address_of_parameter operation.";
  let description = [{
    Placeholder for the dive_vm.address_of_parameter operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: address_of_parameter_region
//===----------------------------------------------------------------------===//
def DiveVm_AddressOfParameterRegionOp : DiveVm_Op<"address_of_parameter_region"> {
  let summary = "DiveVM address_of_parameter_region operation.";
  let description = [{
    Placeholder for the dive_vm.address_of_parameter_region operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: address_of_scratch
//===----------------------------------------------------------------------===//
def DiveVm_AddressOfScratchOp : DiveVm_Op<"address_of_scratch"> {
  let summary = "DiveVM address_of_scratch operation.";
  let description = [{
    Placeholder for the dive_vm.address_of_scratch operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: address_space
//===----------------------------------------------------------------------===//
def DiveVm_AddressSpaceOp : DiveVm_Op<"address_space"> {
  let summary = "DiveVM address_space operation.";
  let description = [{
    Placeholder for the dive_vm.address_space operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: allocate
//===----------------------------------------------------------------------===//
def DiveVm_AllocateOp : DiveVm_Op<"allocate"> {
  let summary = "DiveVM allocate operation.";
  let description = [{
    Placeholder for the dive_vm.allocate operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: arithmetic_left_shift
//===----------------------------------------------------------------------===//
def DiveVm_ArithmeticLeftShiftOp : DiveVm_Op<"arithmetic_left_shift"> {
  let summary = "DiveVM arithmetic_left_shift operation.";
  let description = [{
    Placeholder for the dive_vm.arithmetic_left_shift operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: arithmetic_left_shift_imm
//===----------------------------------------------------------------------===//
def DiveVm_ArithmeticLeftShiftImmOp : DiveVm_Op<"arithmetic_left_shift_imm"> {
  let summary = "DiveVM arithmetic_left_shift_imm operation.";
  let description = [{
    Placeholder for the dive_vm.arithmetic_left_shift_imm operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: arithmetic_right_shift
//===----------------------------------------------------------------------===//
def DiveVm_ArithmeticRightShiftOp : DiveVm_Op<"arithmetic_right_shift"> {
  let summary = "DiveVM arithmetic_right_shift operation.";
  let description = [{
    Placeholder for the dive_vm.arithmetic_right_shift operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: arithmetic_right_shift_imm
//===----------------------------------------------------------------------===//
def DiveVm_ArithmeticRightShiftImmOp : DiveVm_Op<"arithmetic_right_shift_imm"> {
  let summary = "DiveVM arithmetic_right_shift_imm operation.";
  let description = [{
    Placeholder for the dive_vm.arithmetic_right_shift_imm operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: benchmark
//===----------------------------------------------------------------------===//
def DiveVm_BenchmarkOp : DiveVm_Op<"benchmark"> {
  let summary = "DiveVM benchmark operation.";
  let description = [{
    Placeholder for the dive_vm.benchmark operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: bitcast
//===----------------------------------------------------------------------===//
def DiveVm_BitcastOp : DiveVm_Op<"bitcast"> {
  let summary = "DiveVM bitcast operation.";
  let description = [{
    Placeholder for the dive_vm.bitcast operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: bitwise_and
//===----------------------------------------------------------------------===//
def DiveVm_BitwiseAndOp : DiveVm_Op<"bitwise_and"> {
  let summary = "DiveVM bitwise_and operation.";
  let description = [{
    Placeholder for the dive_vm.bitwise_and operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: bitwise_and_imm
//===----------------------------------------------------------------------===//
def DiveVm_BitwiseAndImmOp : DiveVm_Op<"bitwise_and_imm"> {
  let summary = "DiveVM bitwise_and_imm operation.";
  let description = [{
    Placeholder for the dive_vm.bitwise_and_imm operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: bitwise_or
//===----------------------------------------------------------------------===//
def DiveVm_BitwiseOrOp : DiveVm_Op<"bitwise_or"> {
  let summary = "DiveVM bitwise_or operation.";
  let description = [{
    Placeholder for the dive_vm.bitwise_or operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: bitwise_or_imm
//===----------------------------------------------------------------------===//
def DiveVm_BitwiseOrImmOp : DiveVm_Op<"bitwise_or_imm"> {
  let summary = "DiveVM bitwise_or_imm operation.";
  let description = [{
    Placeholder for the dive_vm.bitwise_or_imm operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: bitwise_xor
//===----------------------------------------------------------------------===//
def DiveVm_BitwiseXorOp : DiveVm_Op<"bitwise_xor"> {
  let summary = "DiveVM bitwise_xor operation.";
  let description = [{
    Placeholder for the dive_vm.bitwise_xor operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: bitwise_xor_imm
//===----------------------------------------------------------------------===//
def DiveVm_BitwiseXorImmOp : DiveVm_Op<"bitwise_xor_imm"> {
  let summary = "DiveVM bitwise_xor_imm operation.";
  let description = [{
    Placeholder for the dive_vm.bitwise_xor_imm operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: br
//===----------------------------------------------------------------------===//
def DiveVm_BrOp : DiveVm_Op<"br"> {
  let summary = "DiveVM br operation.";
  let description = [{
    Placeholder for the dive_vm.br operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: buffer_dir
//===----------------------------------------------------------------------===//
def DiveVm_BufferDirOp : DiveVm_Op<"buffer_dir"> {
  let summary = "DiveVM buffer_dir operation.";
  let description = [{
    Placeholder for the dive_vm.buffer_dir operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: cache_clean_invalidate
//===----------------------------------------------------------------------===//
def DiveVm_CacheCleanInvalidateOp : DiveVm_Op<"cache_clean_invalidate"> {
  let summary = "DiveVM cache_clean_invalidate operation.";
  let description = [{
    Placeholder for the dive_vm.cache_clean_invalidate operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: cast
//===----------------------------------------------------------------------===//
def DiveVm_CastOp : DiveVm_Op<"cast"> {
  let summary = "DiveVM cast operation.";
  let description = [{
    Placeholder for the dive_vm.cast operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: chunk_offsets
//===----------------------------------------------------------------------===//
def DiveVm_ChunkOffsetsOp : DiveVm_Op<"chunk_offsets"> {
  let summary = "DiveVM chunk_offsets operation.";
  let description = [{
    Placeholder for the dive_vm.chunk_offsets operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: chunk_sizes
//===----------------------------------------------------------------------===//
def DiveVm_ChunkSizesOp : DiveVm_Op<"chunk_sizes"> {
  let summary = "DiveVM chunk_sizes operation.";
  let description = [{
    Placeholder for the dive_vm.chunk_sizes operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: compute_norm_stats_for_rkhy
//===----------------------------------------------------------------------===//
def DiveVm_ComputeNormStatsForRkhyOp : DiveVm_Op<"compute_norm_stats_for_rkhy"> {
  let summary = "DiveVM compute_norm_stats_for_rkhy operation.";
  let description = [{
    Placeholder for the dive_vm.compute_norm_stats_for_rkhy operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: cond_br
//===----------------------------------------------------------------------===//
def DiveVm_CondBrOp : DiveVm_Op<"cond_br"> {
  let summary = "DiveVM cond_br operation.";
  let description = [{
    Placeholder for the dive_vm.cond_br operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: const
//===----------------------------------------------------------------------===//
def DiveVm_ConstOp : DiveVm_Op<"const"> {
  let summary = "DiveVM const operation.";
  let description = [{
    Placeholder for the dive_vm.const operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: const_bytes
//===----------------------------------------------------------------------===//
def DiveVm_ConstBytesOp : DiveVm_Op<"const_bytes"> {
  let summary = "DiveVM const_bytes operation.";
  let description = [{
    Placeholder for the dive_vm.const_bytes operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: copy
//===----------------------------------------------------------------------===//
def DiveVm_CopyOp : DiveVm_Op<"copy"> {
  let summary = "DiveVM copy operation.";
  let description = [{
    Placeholder for the dive_vm.copy operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: copy_imm
//===----------------------------------------------------------------------===//
def DiveVm_CopyImmOp : DiveVm_Op<"copy_imm"> {
  let summary = "DiveVM copy_imm operation.";
  let description = [{
    Placeholder for the dive_vm.copy_imm operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: core_id
//===----------------------------------------------------------------------===//
def DiveVm_CoreIdOp : DiveVm_Op<"core_id"> {
  let summary = "DiveVM core_id operation.";
  let description = [{
    Placeholder for the dive_vm.core_id operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: cuda_emu_custom_op
//===----------------------------------------------------------------------===//
def DiveVm_CudaEmuCustomOpOp : DiveVm_Op<"cuda_emu_custom_op"> {
  let summary = "DiveVM cuda_emu_custom_op operation.";
  let description = [{
    Placeholder for the dive_vm.cuda_emu_custom_op operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: cumsum
//===----------------------------------------------------------------------===//
def DiveVm_CumsumOp : DiveVm_Op<"cumsum"> {
  let summary = "DiveVM cumsum operation.";
  let description = [{
    Placeholder for the dive_vm.cumsum operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: custom_on_addresses
//===----------------------------------------------------------------------===//
def DiveVm_CustomOnAddressesOp : DiveVm_Op<"custom_on_addresses"> {
  let summary = "DiveVM custom_on_addresses operation.";
  let description = [{
    Placeholder for the dive_vm.custom_on_addresses operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: custom_on_buffers
//===----------------------------------------------------------------------===//
def DiveVm_CustomOnBuffersOp : DiveVm_Op<"custom_on_buffers"> {
  let summary = "DiveVM custom_on_buffers operation.";
  let description = [{
    Placeholder for the dive_vm.custom_on_buffers operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: disable_itc_tracing
//===----------------------------------------------------------------------===//
def DiveVm_DisableItcTracingOp : DiveVm_Op<"disable_itc_tracing"> {
  let summary = "DiveVM disable_itc_tracing operation.";
  let description = [{
    Placeholder for the dive_vm.disable_itc_tracing operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: dispatch_hardware_instruction
//===----------------------------------------------------------------------===//
def DiveVm_DispatchHardwareInstructionOp : DiveVm_Op<"dispatch_hardware_instruction"> {
  let summary = "DiveVM dispatch_hardware_instruction operation.";
  let description = [{
    Placeholder for the dive_vm.dispatch_hardware_instruction operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: div
//===----------------------------------------------------------------------===//
def DiveVm_DivOp : DiveVm_Op<"div"> {
  let summary = "DiveVM div operation.";
  let description = [{
    Placeholder for the dive_vm.div operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: div_imm
//===----------------------------------------------------------------------===//
def DiveVm_DivImmOp : DiveVm_Op<"div_imm"> {
  let summary = "DiveVM div_imm operation.";
  let description = [{
    Placeholder for the dive_vm.div_imm operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: dive_vm_module
//===----------------------------------------------------------------------===//
def DiveVm_DiveVmModuleOp : DiveVm_Op<"dive_vm_module"> {
  let summary = "DiveVM dive_vm_module operation.";
  let description = [{
    Placeholder for the dive_vm.dive_vm_module operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: dma_hint
//===----------------------------------------------------------------------===//
def DiveVm_DmaHintOp : DiveVm_Op<"dma_hint"> {
  let summary = "DiveVM dma_hint operation.";
  let description = [{
    Placeholder for the dive_vm.dma_hint operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: dma_queue
//===----------------------------------------------------------------------===//
def DiveVm_DmaQueueOp : DiveVm_Op<"dma_queue"> {
  let summary = "DiveVM dma_queue operation.";
  let description = [{
    Placeholder for the dive_vm.dma_queue operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: dtc_mode
//===----------------------------------------------------------------------===//
def DiveVm_DtcModeOp : DiveVm_Op<"dtc_mode"> {
  let summary = "DiveVM dtc_mode operation.";
  let description = [{
    Placeholder for the dive_vm.dtc_mode operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: dvfs
//===----------------------------------------------------------------------===//
def DiveVm_DvfsOp : DiveVm_Op<"dvfs"> {
  let summary = "DiveVM dvfs operation.";
  let description = [{
    Placeholder for the dive_vm.dvfs operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: dynamic_slice_y
//===----------------------------------------------------------------------===//
def DiveVm_DynamicSliceYOp : DiveVm_Op<"dynamic_slice_y"> {
  let summary = "DiveVM dynamic_slice_y operation.";
  let description = [{
    Placeholder for the dive_vm.dynamic_slice_y operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: enable_itc_tracing
//===----------------------------------------------------------------------===//
def DiveVm_EnableItcTracingOp : DiveVm_Op<"enable_itc_tracing"> {
  let summary = "DiveVM enable_itc_tracing operation.";
  let description = [{
    Placeholder for the dive_vm.enable_itc_tracing operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: epilogue
//===----------------------------------------------------------------------===//
def DiveVm_EpilogueOp : DiveVm_Op<"epilogue"> {
  let summary = "DiveVM epilogue operation.";
  let description = [{
    Placeholder for the dive_vm.epilogue operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: equal
//===----------------------------------------------------------------------===//
def DiveVm_EqualOp : DiveVm_Op<"equal"> {
  let summary = "DiveVM equal operation.";
  let description = [{
    Placeholder for the dive_vm.equal operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: equal_imm
//===----------------------------------------------------------------------===//
def DiveVm_EqualImmOp : DiveVm_Op<"equal_imm"> {
  let summary = "DiveVM equal_imm operation.";
  let description = [{
    Placeholder for the dive_vm.equal_imm operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: executable
//===----------------------------------------------------------------------===//
def DiveVm_ExecutableOp : DiveVm_Op<"executable"> {
  let summary = "DiveVM executable operation.";
  let description = [{
    Placeholder for the dive_vm.executable operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: extract_slice
//===----------------------------------------------------------------------===//
def DiveVm_ExtractSliceOp : DiveVm_Op<"extract_slice"> {
  let summary = "DiveVM extract_slice operation.";
  let description = [{
    Placeholder for the dive_vm.extract_slice operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: fill
//===----------------------------------------------------------------------===//
def DiveVm_FillOp : DiveVm_Op<"fill"> {
  let summary = "DiveVM fill operation.";
  let description = [{
    Placeholder for the dive_vm.fill operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: gather
//===----------------------------------------------------------------------===//
def DiveVm_GatherOp : DiveVm_Op<"gather"> {
  let summary = "DiveVM gather operation.";
  let description = [{
    Placeholder for the dive_vm.gather operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: gather_nd
//===----------------------------------------------------------------------===//
def DiveVm_GatherNdOp : DiveVm_Op<"gather_nd"> {
  let summary = "DiveVM gather_nd operation.";
  let description = [{
    Placeholder for the dive_vm.gather_nd operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: get_const
//===----------------------------------------------------------------------===//
def DiveVm_GetConstOp : DiveVm_Op<"get_const"> {
  let summary = "DiveVM get_const operation.";
  let description = [{
    Placeholder for the dive_vm.get_const operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: greater
//===----------------------------------------------------------------------===//
def DiveVm_GreaterOp : DiveVm_Op<"greater"> {
  let summary = "DiveVM greater operation.";
  let description = [{
    Placeholder for the dive_vm.greater operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: greater_equal
//===----------------------------------------------------------------------===//
def DiveVm_GreaterEqualOp : DiveVm_Op<"greater_equal"> {
  let summary = "DiveVM greater_equal operation.";
  let description = [{
    Placeholder for the dive_vm.greater_equal operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: greater_equal_imm
//===----------------------------------------------------------------------===//
def DiveVm_GreaterEqualImmOp : DiveVm_Op<"greater_equal_imm"> {
  let summary = "DiveVM greater_equal_imm operation.";
  let description = [{
    Placeholder for the dive_vm.greater_equal_imm operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: greater_imm
//===----------------------------------------------------------------------===//
def DiveVm_GreaterImmOp : DiveVm_Op<"greater_imm"> {
  let summary = "DiveVM greater_imm operation.";
  let description = [{
    Placeholder for the dive_vm.greater_imm operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: has_idp
//===----------------------------------------------------------------------===//
def DiveVm_HasIdpOp : DiveVm_Op<"has_idp"> {
  let summary = "DiveVM has_idp operation.";
  let description = [{
    Placeholder for the dive_vm.has_idp operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: heap_size_bytes
//===----------------------------------------------------------------------===//
def DiveVm_HeapSizeBytesOp : DiveVm_Op<"heap_size_bytes"> {
  let summary = "DiveVM heap_size_bytes operation.";
  let description = [{
    Placeholder for the dive_vm.heap_size_bytes operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: hib_gather_edit
//===----------------------------------------------------------------------===//
def DiveVm_HibGatherEditOp : DiveVm_Op<"hib_gather_edit"> {
  let summary = "DiveVM hib_gather_edit operation.";
  let description = [{
    Placeholder for the dive_vm.hib_gather_edit operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: idp
//===----------------------------------------------------------------------===//
def DiveVm_IdpOp : DiveVm_Op<"idp"> {
  let summary = "DiveVM idp operation.";
  let description = [{
    Placeholder for the dive_vm.idp operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: idp_instance
//===----------------------------------------------------------------------===//
def DiveVm_IdpInstanceOp : DiveVm_Op<"idp_instance"> {
  let summary = "DiveVM idp_instance operation.";
  let description = [{
    Placeholder for the dive_vm.idp_instance operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: insert_slice
//===----------------------------------------------------------------------===//
def DiveVm_InsertSliceOp : DiveVm_Op<"insert_slice"> {
  let summary = "DiveVM insert_slice operation.";
  let description = [{
    Placeholder for the dive_vm.insert_slice operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: instruction_id
//===----------------------------------------------------------------------===//
def DiveVm_InstructionIdOp : DiveVm_Op<"instruction_id"> {
  let summary = "DiveVM instruction_id operation.";
  let description = [{
    Placeholder for the dive_vm.instruction_id operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: instruction_id_live_tensors
//===----------------------------------------------------------------------===//
def DiveVm_InstructionIdLiveTensorsOp : DiveVm_Op<"instruction_id_live_tensors"> {
  let summary = "DiveVM instruction_id_live_tensors operation.";
  let description = [{
    Placeholder for the dive_vm.instruction_id_live_tensors operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: legacy_scalar
//===----------------------------------------------------------------------===//
def DiveVm_LegacyScalarOp : DiveVm_Op<"legacy_scalar"> {
  let summary = "DiveVM legacy_scalar operation.";
  let description = [{
    Placeholder for the dive_vm.legacy_scalar operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: less
//===----------------------------------------------------------------------===//
def DiveVm_LessOp : DiveVm_Op<"less"> {
  let summary = "DiveVM less operation.";
  let description = [{
    Placeholder for the dive_vm.less operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: less_equal
//===----------------------------------------------------------------------===//
def DiveVm_LessEqualOp : DiveVm_Op<"less_equal"> {
  let summary = "DiveVM less_equal operation.";
  let description = [{
    Placeholder for the dive_vm.less_equal operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: less_equal_imm
//===----------------------------------------------------------------------===//
def DiveVm_LessEqualImmOp : DiveVm_Op<"less_equal_imm"> {
  let summary = "DiveVM less_equal_imm operation.";
  let description = [{
    Placeholder for the dive_vm.less_equal_imm operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: less_imm
//===----------------------------------------------------------------------===//
def DiveVm_LessImmOp : DiveVm_Op<"less_imm"> {
  let summary = "DiveVM less_imm operation.";
  let description = [{
    Placeholder for the dive_vm.less_imm operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: live_tensor
//===----------------------------------------------------------------------===//
def DiveVm_LiveTensorOp : DiveVm_Op<"live_tensor"> {
  let summary = "DiveVM live_tensor operation.";
  let description = [{
    Placeholder for the dive_vm.live_tensor operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: load
//===----------------------------------------------------------------------===//
def DiveVm_LoadOp : DiveVm_Op<"load"> {
  let summary = "DiveVM load operation.";
  let description = [{
    Placeholder for the dive_vm.load operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: load_indirect
//===----------------------------------------------------------------------===//
def DiveVm_LoadIndirectOp : DiveVm_Op<"load_indirect"> {
  let summary = "DiveVM load_indirect operation.";
  let description = [{
    Placeholder for the dive_vm.load_indirect operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: logical_and
//===----------------------------------------------------------------------===//
def DiveVm_LogicalAndOp : DiveVm_Op<"logical_and"> {
  let summary = "DiveVM logical_and operation.";
  let description = [{
    Placeholder for the dive_vm.logical_and operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: logical_and_imm
//===----------------------------------------------------------------------===//
def DiveVm_LogicalAndImmOp : DiveVm_Op<"logical_and_imm"> {
  let summary = "DiveVM logical_and_imm operation.";
  let description = [{
    Placeholder for the dive_vm.logical_and_imm operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: logical_right_shift
//===----------------------------------------------------------------------===//
def DiveVm_LogicalRightShiftOp : DiveVm_Op<"logical_right_shift"> {
  let summary = "DiveVM logical_right_shift operation.";
  let description = [{
    Placeholder for the dive_vm.logical_right_shift operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: logical_right_shift_imm
//===----------------------------------------------------------------------===//
def DiveVm_LogicalRightShiftImmOp : DiveVm_Op<"logical_right_shift_imm"> {
  let summary = "DiveVM logical_right_shift_imm operation.";
  let description = [{
    Placeholder for the dive_vm.logical_right_shift_imm operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: mark_dive_for_host_resident
//===----------------------------------------------------------------------===//
def DiveVm_MarkDiveForHostResidentOp : DiveVm_Op<"mark_dive_for_host_resident"> {
  let summary = "DiveVM mark_dive_for_host_resident operation.";
  let description = [{
    Placeholder for the dive_vm.mark_dive_for_host_resident operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: mark_value_with_shape
//===----------------------------------------------------------------------===//
def DiveVm_MarkValueWithShapeOp : DiveVm_Op<"mark_value_with_shape"> {
  let summary = "DiveVM mark_value_with_shape operation.";
  let description = [{
    Placeholder for the dive_vm.mark_value_with_shape operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: mask_indices
//===----------------------------------------------------------------------===//
def DiveVm_MaskIndicesOp : DiveVm_Op<"mask_indices"> {
  let summary = "DiveVM mask_indices operation.";
  let description = [{
    Placeholder for the dive_vm.mask_indices operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: max
//===----------------------------------------------------------------------===//
def DiveVm_MaxOp : DiveVm_Op<"max"> {
  let summary = "DiveVM max operation.";
  let description = [{
    Placeholder for the dive_vm.max operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: max_imm
//===----------------------------------------------------------------------===//
def DiveVm_MaxImmOp : DiveVm_Op<"max_imm"> {
  let summary = "DiveVM max_imm operation.";
  let description = [{
    Placeholder for the dive_vm.max_imm operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: min
//===----------------------------------------------------------------------===//
def DiveVm_MinOp : DiveVm_Op<"min"> {
  let summary = "DiveVM min operation.";
  let description = [{
    Placeholder for the dive_vm.min operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: min_imm
//===----------------------------------------------------------------------===//
def DiveVm_MinImmOp : DiveVm_Op<"min_imm"> {
  let summary = "DiveVM min_imm operation.";
  let description = [{
    Placeholder for the dive_vm.min_imm operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: module_group
//===----------------------------------------------------------------------===//
def DiveVm_ModuleGroupOp : DiveVm_Op<"module_group"> {
  let summary = "DiveVM module_group operation.";
  let description = [{
    Placeholder for the dive_vm.module_group operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: mul
//===----------------------------------------------------------------------===//
def DiveVm_MulOp : DiveVm_Op<"mul"> {
  let summary = "DiveVM mul operation.";
  let description = [{
    Placeholder for the dive_vm.mul operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: mul_imm
//===----------------------------------------------------------------------===//
def DiveVm_MulImmOp : DiveVm_Op<"mul_imm"> {
  let summary = "DiveVM mul_imm operation.";
  let description = [{
    Placeholder for the dive_vm.mul_imm operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: multinomial
//===----------------------------------------------------------------------===//
def DiveVm_MultinomialOp : DiveVm_Op<"multinomial"> {
  let summary = "DiveVM multinomial operation.";
  let description = [{
    Placeholder for the dive_vm.multinomial operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: not_equal
//===----------------------------------------------------------------------===//
def DiveVm_NotEqualOp : DiveVm_Op<"not_equal"> {
  let summary = "DiveVM not_equal operation.";
  let description = [{
    Placeholder for the dive_vm.not_equal operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: not_equal_imm
//===----------------------------------------------------------------------===//
def DiveVm_NotEqualImmOp : DiveVm_Op<"not_equal_imm"> {
  let summary = "DiveVM not_equal_imm operation.";
  let description = [{
    Placeholder for the dive_vm.not_equal_imm operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: num_input_tensors
//===----------------------------------------------------------------------===//
def DiveVm_NumInputTensorsOp : DiveVm_Op<"num_input_tensors"> {
  let summary = "DiveVM num_input_tensors operation.";
  let description = [{
    Placeholder for the dive_vm.num_input_tensors operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: num_output_tensors
//===----------------------------------------------------------------------===//
def DiveVm_NumOutputTensorsOp : DiveVm_Op<"num_output_tensors"> {
  let summary = "DiveVM num_output_tensors operation.";
  let description = [{
    Placeholder for the dive_vm.num_output_tensors operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: one_hot
//===----------------------------------------------------------------------===//
def DiveVm_OneHotOp : DiveVm_Op<"one_hot"> {
  let summary = "DiveVM one_hot operation.";
  let description = [{
    Placeholder for the dive_vm.one_hot operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: pad
//===----------------------------------------------------------------------===//
def DiveVm_PadOp : DiveVm_Op<"pad"> {
  let summary = "DiveVM pad operation.";
  let description = [{
    Placeholder for the dive_vm.pad operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: parameter_array
//===----------------------------------------------------------------------===//
def DiveVm_ParameterArrayOp : DiveVm_Op<"parameter_array"> {
  let summary = "DiveVM parameter_array operation.";
  let description = [{
    Placeholder for the dive_vm.parameter_array operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: patch_instruction_for_strided_io
//===----------------------------------------------------------------------===//
def DiveVm_PatchInstructionForStridedIoOp : DiveVm_Op<"patch_instruction_for_strided_io"> {
  let summary = "DiveVM patch_instruction_for_strided_io operation.";
  let description = [{
    Placeholder for the dive_vm.patch_instruction_for_strided_io operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: perform_software_preemption_if_requested
//===----------------------------------------------------------------------===//
def DiveVm_PerformSoftwarePreemptionIfRequestedOp : DiveVm_Op<"perform_software_preemption_if_requested"> {
  let summary = "DiveVM perform_software_preemption_if_requested operation.";
  let description = [{
    Placeholder for the dive_vm.perform_software_preemption_if_requested operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: pow
//===----------------------------------------------------------------------===//
def DiveVm_PowOp : DiveVm_Op<"pow"> {
  let summary = "DiveVM pow operation.";
  let description = [{
    Placeholder for the dive_vm.pow operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: pow_imm
//===----------------------------------------------------------------------===//
def DiveVm_PowImmOp : DiveVm_Op<"pow_imm"> {
  let summary = "DiveVM pow_imm operation.";
  let description = [{
    Placeholder for the dive_vm.pow_imm operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: print
//===----------------------------------------------------------------------===//
def DiveVm_PrintOp : DiveVm_Op<"print"> {
  let summary = "DiveVM print operation.";
  let description = [{
    Placeholder for the dive_vm.print operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: program_tensor_mapping_table
//===----------------------------------------------------------------------===//
def DiveVm_ProgramTensorMappingTableOp : DiveVm_Op<"program_tensor_mapping_table"> {
  let summary = "DiveVM program_tensor_mapping_table operation.";
  let description = [{
    Placeholder for the dive_vm.program_tensor_mapping_table operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: prologue
//===----------------------------------------------------------------------===//
def DiveVm_PrologueOp : DiveVm_Op<"prologue"> {
  let summary = "DiveVM prologue operation.";
  let description = [{
    Placeholder for the dive_vm.prologue operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: put_bits
//===----------------------------------------------------------------------===//
def DiveVm_PutBitsOp : DiveVm_Op<"put_bits"> {
  let summary = "DiveVM put_bits operation.";
  let description = [{
    Placeholder for the dive_vm.put_bits operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: read_gn_stats
//===----------------------------------------------------------------------===//
def DiveVm_ReadGnStatsOp : DiveVm_Op<"read_gn_stats"> {
  let summary = "DiveVM read_gn_stats operation.";
  let description = [{
    Placeholder for the dive_vm.read_gn_stats operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: reduction
//===----------------------------------------------------------------------===//
def DiveVm_ReductionOp : DiveVm_Op<"reduction"> {
  let summary = "DiveVM reduction operation.";
  let description = [{
    Placeholder for the dive_vm.reduction operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: reduction_type
//===----------------------------------------------------------------------===//
def DiveVm_ReductionTypeOp : DiveVm_Op<"reduction_type"> {
  let summary = "DiveVM reduction_type operation.";
  let description = [{
    Placeholder for the dive_vm.reduction_type operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: rem
//===----------------------------------------------------------------------===//
def DiveVm_RemOp : DiveVm_Op<"rem"> {
  let summary = "DiveVM rem operation.";
  let description = [{
    Placeholder for the dive_vm.rem operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: rem_imm
//===----------------------------------------------------------------------===//
def DiveVm_RemImmOp : DiveVm_Op<"rem_imm"> {
  let summary = "DiveVM rem_imm operation.";
  let description = [{
    Placeholder for the dive_vm.rem_imm operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: roll
//===----------------------------------------------------------------------===//
def DiveVm_RollOp : DiveVm_Op<"roll"> {
  let summary = "DiveVM roll operation.";
  let description = [{
    Placeholder for the dive_vm.roll operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: scalar_type
//===----------------------------------------------------------------------===//
def DiveVm_ScalarTypeOp : DiveVm_Op<"scalar_type"> {
  let summary = "DiveVM scalar_type operation.";
  let description = [{
    Placeholder for the dive_vm.scalar_type operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: scatter_nd
//===----------------------------------------------------------------------===//
def DiveVm_ScatterNdOp : DiveVm_Op<"scatter_nd"> {
  let summary = "DiveVM scatter_nd operation.";
  let description = [{
    Placeholder for the dive_vm.scatter_nd operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: select
//===----------------------------------------------------------------------===//
def DiveVm_SelectOp : DiveVm_Op<"select"> {
  let summary = "DiveVM select operation.";
  let description = [{
    Placeholder for the dive_vm.select operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: serialized_model_format_type
//===----------------------------------------------------------------------===//
def DiveVm_SerializedModelFormatTypeOp : DiveVm_Op<"serialized_model_format_type"> {
  let summary = "DiveVM serialized_model_format_type operation.";
  let description = [{
    Placeholder for the dive_vm.serialized_model_format_type operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: set_dtc_mode
//===----------------------------------------------------------------------===//
def DiveVm_SetDtcModeOp : DiveVm_Op<"set_dtc_mode"> {
  let summary = "DiveVM set_dtc_mode operation.";
  let description = [{
    Placeholder for the dive_vm.set_dtc_mode operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: shape_of_activation
//===----------------------------------------------------------------------===//
def DiveVm_ShapeOfActivationOp : DiveVm_Op<"shape_of_activation"> {
  let summary = "DiveVM shape_of_activation operation.";
  let description = [{
    Placeholder for the dive_vm.shape_of_activation operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: shared_tensor
//===----------------------------------------------------------------------===//
def DiveVm_SharedTensorOp : DiveVm_Op<"shared_tensor"> {
  let summary = "DiveVM shared_tensor operation.";
  let description = [{
    Placeholder for the dive_vm.shared_tensor operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: sign
//===----------------------------------------------------------------------===//
def DiveVm_SignOp : DiveVm_Op<"sign"> {
  let summary = "DiveVM sign operation.";
  let description = [{
    Placeholder for the dive_vm.sign operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: store
//===----------------------------------------------------------------------===//
def DiveVm_StoreOp : DiveVm_Op<"store"> {
  let summary = "DiveVM store operation.";
  let description = [{
    Placeholder for the dive_vm.store operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: store_indirect
//===----------------------------------------------------------------------===//
def DiveVm_StoreIndirectOp : DiveVm_Op<"store_indirect"> {
  let summary = "DiveVM store_indirect operation.";
  let description = [{
    Placeholder for the dive_vm.store_indirect operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: sub
//===----------------------------------------------------------------------===//
def DiveVm_SubOp : DiveVm_Op<"sub"> {
  let summary = "DiveVM sub operation.";
  let description = [{
    Placeholder for the dive_vm.sub operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: sub_imm
//===----------------------------------------------------------------------===//
def DiveVm_SubImmOp : DiveVm_Op<"sub_imm"> {
  let summary = "DiveVM sub_imm operation.";
  let description = [{
    Placeholder for the dive_vm.sub_imm operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: top_k
//===----------------------------------------------------------------------===//
def DiveVm_TopKOp : DiveVm_Op<"top_k"> {
  let summary = "DiveVM top_k operation.";
  let description = [{
    Placeholder for the dive_vm.top_k operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: tpu_offload
//===----------------------------------------------------------------------===//
def DiveVm_TpuOffloadOp : DiveVm_Op<"tpu_offload"> {
  let summary = "DiveVM tpu_offload operation.";
  let description = [{
    Placeholder for the dive_vm.tpu_offload operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: transition_dtc_power_island
//===----------------------------------------------------------------------===//
def DiveVm_TransitionDtcPowerIslandOp : DiveVm_Op<"transition_dtc_power_island"> {
  let summary = "DiveVM transition_dtc_power_island operation.";
  let description = [{
    Placeholder for the dive_vm.transition_dtc_power_island operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: translate_sram_address
//===----------------------------------------------------------------------===//
def DiveVm_TranslateSramAddressOp : DiveVm_Op<"translate_sram_address"> {
  let summary = "DiveVM translate_sram_address operation.";
  let description = [{
    Placeholder for the dive_vm.translate_sram_address operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: view_on_address
//===----------------------------------------------------------------------===//
def DiveVm_ViewOnAddressOp : DiveVm_Op<"view_on_address"> {
  let summary = "DiveVM view_on_address operation.";
  let description = [{
    Placeholder for the dive_vm.view_on_address operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: wait_for_fence_completion
//===----------------------------------------------------------------------===//
def DiveVm_WaitForFenceCompletionOp : DiveVm_Op<"wait_for_fence_completion"> {
  let summary = "DiveVM wait_for_fence_completion operation.";
  let description = [{
    Placeholder for the dive_vm.wait_for_fence_completion operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: wait_for_power_island_transition_complete
//===----------------------------------------------------------------------===//
def DiveVm_WaitForPowerIslandTransitionCompleteOp : DiveVm_Op<"wait_for_power_island_transition_complete"> {
  let summary = "DiveVM wait_for_power_island_transition_complete operation.";
  let description = [{
    Placeholder for the dive_vm.wait_for_power_island_transition_complete operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: wait_for_rkhy_completion
//===----------------------------------------------------------------------===//
def DiveVm_WaitForRkhyCompletionOp : DiveVm_Op<"wait_for_rkhy_completion"> {
  let summary = "DiveVM wait_for_rkhy_completion operation.";
  let description = [{
    Placeholder for the dive_vm.wait_for_rkhy_completion operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: write_dma_descriptor
//===----------------------------------------------------------------------===//
def DiveVm_WriteDmaDescriptorOp : DiveVm_Op<"write_dma_descriptor"> {
  let summary = "DiveVM write_dma_descriptor operation.";
  let description = [{
    Placeholder for the dive_vm.write_dma_descriptor operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: write_hib_data
//===----------------------------------------------------------------------===//
def DiveVm_WriteHibDataOp : DiveVm_Op<"write_hib_data"> {
  let summary = "DiveVM write_hib_data operation.";
  let description = [{
    Placeholder for the dive_vm.write_hib_data operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: write_scalar_arch_register
//===----------------------------------------------------------------------===//
def DiveVm_WriteScalarArchRegisterOp : DiveVm_Op<"write_scalar_arch_register"> {
  let summary = "DiveVM write_scalar_arch_register operation.";
  let description = [{
    Placeholder for the dive_vm.write_scalar_arch_register operation observed in dive binaries.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

#endif // DIVEVM_OPS
