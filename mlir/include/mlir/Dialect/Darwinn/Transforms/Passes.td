//===-- DwcPasses.td - Darwinn DWC pass declarations ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the Darwinn DWC lowering passes recovered from
// task3_passes.tsv (kind=pass rows of blob string inventory). Bodies live in
// DwcPasses.cpp as real transforms keyed to the same TSV rows.
//
// Canonical pipeline order:
//   dwc-legalize family, dwc-lower family, convert-dive-vm-to-llvm,
//   convert-tpu-offload-to-llvm, dive-program-tpu.
// group-tpu-offloads-by-parameters runs with the TPU offload grouping stage.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LIB_DIALECT_DARWINN_TRANSFORMS_DWCPASSES
#define MLIR_LIB_DIALECT_DARWINN_TRANSFORMS_DWCPASSES

include "mlir/Pass/PassBase.td"

def DwcAckrModelConverterPass : Pass<"ackr-model-converter", "func::FuncOp"> {
  let summary = "DWC pass ackr-model-converter";
  let description = [{
    Seen at 0xdac24f in task3_passes.tsv, this pass folds same-type one-operand one-result
    `darwinn.bitcast`, `darwinn.convert` and `darwinn.copy_op` identities with replaceAllUsesWith and
    erase. It then requires every remaining op to sit in the darwinn dialect and pass
    checkDwcConvertibleTypes, failing other dialects with ackr-model-converter accepts darwinn ops only.
  }];
}

def DwcAddBoundLowerPass : Pass<"add_bound_lower", "func::FuncOp"> {
  let summary = "DWC pass add_bound_lower";
  let description = [{
    Seen at 0xdaad7f in task3_passes.tsv, this pass folds same-type one-operand one-result
    `darwinn.bitcast`, `darwinn.convert` and `darwinn.copy_op` identities with replaceAllUsesWith and
    erase. It then requires every remaining op to sit in the darwinn dialect and pass
    checkDwcConvertibleTypes, failing other dialects with add_bound_lower accepts darwinn ops only.
  }];
}

def DwcAddDiveAbiArgumentsPass : Pass<"add-dive-abi-arguments", "func::FuncOp"> {
  let summary = "DWC pass add-dive-abi-arguments";
  let description = [{
    Seen at 0xd7a252 in task3_passes.tsv, this pass appends a 64-bit integer argument to the function
    type and the entry block, then stamps the dive.abi_args_added unit attr. It returns early when that
    attr is already present.
  }];
}

def DwcAddDiveTracingPass : Pass<"add-dive-tracing", "func::FuncOp"> {
  let summary = "DWC pass add-dive-tracing";
  let description = [{
    Seen at 0xde194d in task3_passes.tsv, this pass inserts a DiveRuntime_TraceEntry call with an LLVM
    undef pointer probe at the start of the entry block through lookupOrCreateFn, and stamps
    dive.tracing_added. It returns early when the attr exists, the body is empty, or no parent module
    exists.
  }];
}

def DwcAllowBf16AndF16TypeLegalizationPass : Pass<"allow-bf16-and-f16-type-legalization", "func::FuncOp"> {
  let summary = "DWC pass allow-bf16-and-f16-type-legalization";
  let description = [{
    Seen at 0xdc1b37 in task3_passes.tsv, this pass lowers darwinn.convert, darwinn.cast_in and
    darwinn.cast_out ops touching bf16 or f16 to dive_vm.cast with makeDwcLowerVmOp after
    checkDwcConvertibleTypes. It skips converts with no bf16 or f16 side.
  }];
}

def DwcArithAssertLowerPass : Pass<"arith assert lower", "func::FuncOp"> {
  let summary = "DWC pass arith-assert-lower";
  let description = [{
    Seen at 0xdaad9b in task3_passes.tsv, this pass applies populateLowerCopySlicePatterns and
    populateLowerConvertPatterns greedily with applyLocalCopySliceLowering and
    applyLocalConvertLowering. It then erases cf.assert ops whose arith.constant condition is true as
    integer one, boolean true, or dense splat one.
  }];
}

def DwcArithLowerPass : Pass<"arith-lower", "func::FuncOp"> {
  let summary = "DWC pass arith-lower";
  let description = [{
    Seen at 0xdaad8f in task3_passes.tsv, this pass applies populateLowerCopySlicePatterns and
    populateLowerConvertPatterns greedily, then runs applyLocalCopySliceLowering and
    applyLocalConvertLowering. It then lowers tgc elementwise ops to arith ops with
    applyDwcLowerScalarArith.
  }];
}

def DwcBitcastConvertPass : Pass<"bitcast-convert", "func::FuncOp"> {
  let summary = "DWC pass bitcast-convert";
  let description = [{
    Seen at 0xd68cb7 in task3_passes.tsv, this pass folds same-type darwinn.bitcast identities with
    replaceAllUsesWith and erase. It lowers static-shape bitcasts with equal element counts and
    int-or-float bitwidths to dive_vm.bitcast with makeDwcLowerVmOp after checkDwcConvertibleTypes.
  }];
}

def DwcChloLegalizeToHloPass : Pass<"chlo-legalize-to-hlo", "func::FuncOp"> {
  let summary = "DWC pass chlo-legalize-to-hlo";
  let description = [{
    Seen at 0xdbc367 in task3_passes.tsv, this pass folds same-type one-operand one-result `mhlo`
    identities with replaceAllUsesWith and erase. It then runs checkDwcConvertibleTypes over the chlo
    and mhlo dialect ops and leaves every other op untouched.
  }];
}

def DwcCompositeLoweringPass : Pass<"composite-lowering", "func::FuncOp"> {
  let summary = "DWC pass composite-lowering";
  let description = [{
    Seen at 0xddee24 in task3_passes.tsv, this pass applies populateLowerCopySlicePatterns and
    populateLowerConvertPatterns greedily, then runs applyLocalCopySliceLowering and
    applyLocalConvertLowering. It also runs applyDwcLowerScatterInline, applyDwcLowerSelectInline,
    applyDwcLowerTopKInline and applyDwcLowerScalarArith for the scatter, select, top_k and arith
    families.
  }];
}

def DwcConcatModelConverterPass : Pass<"concat-model-converter", "func::FuncOp"> {
  let summary = "DWC pass concat-model-converter";
  let description = [{
    Seen at 0xdac238 in task3_passes.tsv, this pass folds single-input same-type darwinn.concat and
    darwinn.concat_v2 identities with replaceAllUsesWith and erase, then runs
    applyLocalCopySliceLowering.
  }];
}

def DwcConcatProofConverterPass : Pass<"concat-proof-converter", "func::FuncOp"> {
  let summary = "DWC pass concat-proof-converter";
  let description = [{
    Seen at 0xdac279 in task3_passes.tsv, this pass folds single-input same-type darwinn.concat and
    darwinn.concat_v2 identities with replaceAllUsesWith and erase, then runs
    applyLocalCopySliceLowering.
  }];
}

def DwcConvertConv1x1ToFcPass : Pass<"convert-conv1x1-to-fc", "func::FuncOp"> {
  let summary = "DWC pass convert-conv1x1-to-fc";
  let description = [{
    Seen at 0xe26ea5 in task3_passes.tsv, this pass lowers darwinn.convert, darwinn.cast_in and
    darwinn.cast_out to dive_vm.cast with applyDwcLowerConvertTrunc. It keeps one-operand one-result ops
    with matching ranked shapes and numeric element types, folding same-type cases in place, and checks
    each with checkDwcConvertibleTypes.
  }];
}

def DwcConvertDiveVmTensorToLinalgPass : Pass<"convert-dive-vm-tensor-to-linalg", "func::FuncOp"> {
  let summary = "DWC pass convert-dive-vm-tensor-to-linalg";
  let description = [{
    Seen at 0xde1ad1 in task3_passes.tsv, this pass folds same-type one-operand one-result `dive_vm`
    identities with replaceAllUsesWith and erase. It then runs checkDwcConvertibleTypes over the dive_vm
    dialect ops and leaves every other op untouched.
  }];
}

def DwcConvertDiveVmTensorToScfPass : Pass<"convert-dive-vm-tensor-to-scf", "func::FuncOp"> {
  let summary = "DWC pass convert-dive-vm-tensor-to-scf";
  let description = [{
    Seen at 0xde4651 in task3_passes.tsv, this pass folds same-type one-operand one-result `dive_vm`
    identities with replaceAllUsesWith and erase. It then runs checkDwcConvertibleTypes over the dive_vm
    dialect ops and leaves every other op untouched.
  }];
}

def DwcConvertDiveVmTensorToTensorPass : Pass<"convert-dive-vm-tensor-to-tensor", "func::FuncOp"> {
  let summary = "DWC pass convert-dive-vm-tensor-to-tensor";
  let description = [{
    Seen at 0xda89fa in task3_passes.tsv, this pass folds same-type one-operand one-result `dive_vm`
    identities with replaceAllUsesWith and erase. It then runs checkDwcConvertibleTypes over the dive_vm
    dialect ops and leaves every other op untouched.
  }];
}

def DwcConvertDiveVmToLlvmPass : Pass<"convert-dive-vm-to-llvm", "func::FuncOp"> {
  let summary = "DWC pass convert-dive-vm-to-llvm";
  let description = [{
    Seen at 0xdcb7dd in task3_passes.tsv, this pass lowers dive_vm.br to LLVM.BrOp and dive_vm.cond_br
    to LLVM.CondBrOp or BrOp by i1 condition, and maps each other supported dive_vm op name to its
    runtime callee through lookupOrCreateFn plus an LLVM call. It checks each op with
    checkDwcConvertibleTypes, rejects multi-result and unknown ops, and records dive_vm.lowered_count.
  }];
}

def DwcConvertDiveVmToMemrefPass : Pass<"convert-dive-vm-to-memref", "func::FuncOp"> {
  let summary = "DWC pass convert-dive-vm-to-memref";
  let description = [{
    Seen at 0xde3efd in task3_passes.tsv, this pass lowers dive_vm.br to LLVM.BrOp and dive_vm.cond_br
    to LLVM.CondBrOp or BrOp by i1 condition, and maps each other supported dive_vm op name to its
    runtime callee through lookupOrCreateFn plus an LLVM call. It checks each op with
    checkDwcConvertibleTypes, rejects multi-result and unknown ops, and records dive_vm.lowered_count.
  }];
}

def DwcConvertDwcToDiveVmTensorPass : Pass<"convert-dwc-to-dive-vm-tensor", "func::FuncOp"> {
  let summary = "DWC pass convert-dwc-to-dive-vm-tensor";
  let description = [{
    Seen at 0xda8a3d in task3_passes.tsv, this pass folds same-type one-operand one-result
    `darwinn.bitcast`, `darwinn.convert` and `darwinn.copy_op` identities with replaceAllUsesWith and
    erase. It then runs checkDwcConvertibleTypes over the darwinn dialect ops and leaves every other op
    untouched.
  }];
}

def DwcConvertDwgToDiveVmPass : Pass<"convert-dwg-to-dive-vm", "func::FuncOp"> {
  let summary = "DWC pass convert-dwg-to-dive-vm";
  let description = [{
    Seen at 0xdcb89c in task3_passes.tsv, this pass folds same-type one-operand one-result dwg dialect
    ops identities with replaceAllUsesWith and erase. It then runs checkDwcConvertibleTypes over the dwg
    dialect ops and leaves every other op untouched.
  }];
}

def DwcConvertDynamicShapeScopeToDiveVmPass : Pass<"convert-dynamic-shape-scope-to-dive-vm", "func::FuncOp"> {
  let summary = "DWC pass convert-dynamic-shape-scope-to-dive-vm";
  let description = [{
    Seen at 0xdcb8b3 in task3_passes.tsv, this pass folds same-type one-operand one-result shape and
    tensor dialect ops identities with replaceAllUsesWith and erase. It then runs
    checkDwcConvertibleTypes over the shape and tensor dialect ops and leaves every other op untouched.
  }];
}

def DwcConvertGenericNormToPseudoOpPass : Pass<"convert-generic-norm-to-pseudo-op", "func::FuncOp"> {
  let summary = "DWC pass convert-generic-norm-to-pseudo-op";
  let description = [{
    Seen at 0xdb45be in task3_passes.tsv, this pass folds same-type one-operand one-result darwinn
    dialect ops identities with replaceAllUsesWith and erase. It then runs checkDwcConvertibleTypes over
    remaining darwinn dialect ops and fails the pass on any failure.
  }];
}

def DwcConvertOpLoweringPass : Pass<"convert-op-lowering", "func::FuncOp"> {
  let summary = "DWC pass convert-op-lowering";
  let description = [{
    Seen at 0xddedce in task3_passes.tsv, this pass applies populateLowerConvertPatterns greedily and
    runs applyLocalConvertLowering. It then lowers convert and cast ops to dive_vm.cast with
    applyDwcLowerConvertTrunc under matching shapes and numeric element types.
  }];
}

def DwcConvertScatterToGenericScatterPass : Pass<"convert-scatter-to-generic-scatter", "func::FuncOp"> {
  let summary = "DWC pass convert-scatter-to-generic-scatter";
  let description = [{
    Seen at 0xdabcd2 in task3_passes.tsv, this pass folds same-type one-operand one-result
    `darwinn.bitcast`, `darwinn.convert` and `darwinn.copy_op` identities with replaceAllUsesWith and
    erase. It then runs checkDwcConvertibleTypes over the darwinn dialect ops and leaves every other op
    untouched.
  }];
}

def DwcConvertSignedIntWithRescalingOpsPass : Pass<"convert-signed-int-with-rescaling-ops", "func::FuncOp"> {
  let summary = "DWC pass convert-signed-int-with-rescaling-ops";
  let description = [{
    Seen at 0xd84723 in task3_passes.tsv, this pass lowers darwinn.convert, darwinn.cast_in and
    darwinn.cast_out to dive_vm.cast with applyDwcLowerConvertTrunc. It keeps one-operand one-result ops
    with matching ranked shapes and numeric element types, folding same-type cases in place, and checks
    each with checkDwcConvertibleTypes.
  }];
}

def DwcConvertSpatialReductionToPoolingPass : Pass<"convert-spatial-reduction-to-pooling", "func::FuncOp"> {
  let summary = "DWC pass convert-spatial-reduction-to-pooling";
  let description = [{
    Seen at 0xddfd8d in task3_passes.tsv, this pass forwards darwinn.spatial_reduction to dive_vm.reduce
    with forwardDwcLowerTo.
  }];
}

def DwcConvertTfToDwcPass : Pass<"convert-tf-to-dwc", "func::FuncOp"> {
  let summary = "DWC pass convert-tf-to-dwc";
  let description = [{
    Seen at 0xe2523e in task3_passes.tsv, this pass folds same-type one-operand one-result tf dialect
    ops identities with replaceAllUsesWith and erase. It then runs checkDwcConvertibleTypes over the tf
    dialect ops and leaves every other op untouched.
  }];
}

def DwcConvertToKInMSparsityPass : Pass<"convert-to-k-in-m-sparsity", "func::FuncOp"> {
  let summary = "DWC pass convert-to-k-in-m-sparsity";
  let description = [{
    Seen at 0xd5a046 in task3_passes.tsv, this pass folds same-type one-operand one-result
    `darwinn.bitcast`, `darwinn.convert` and `darwinn.copy_op` identities with replaceAllUsesWith and
    erase. It then runs checkDwcConvertibleTypes over the darwinn dialect ops and leaves every other op
    untouched.
  }];
}

def DwcConvertTpuOffloadToDiveVmPass : Pass<"convert-tpu-offload-to-dive-vm", "func::FuncOp"> {
  let summary = "DWC pass convert-tpu-offload-to-dive-vm";
  let description = [{
    Seen at 0xdcb8da in task3_passes.tsv, this pass lowers edgetpu convolution and attention family ops
    to DiveTpu_EnqueueInstructions calls and edgetpu matrix multiply and fully connected family ops to
    DiveTpu_EnqueueDmaDescriptor calls through lookupOrCreateFn. It checks each op with
    checkDwcConvertibleTypes, rejects multi-result and unknown edgetpu ops, and records
    edgetpu.lowered_count.
  }];
}

def DwcConvertTpuOffloadToLlvmPass : Pass<"convert-tpu-offload-to-llvm", "func::FuncOp"> {
  let summary = "DWC pass convert-tpu-offload-to-llvm";
  let description = [{
    Seen at 0xdcb84b in task3_passes.tsv, this pass lowers dive_vm.tpu_offload to
    DiveRuntime_ExecuteChildModel and edgetpu convolution and attention family ops to
    DiveTpu_EnqueueInstructions with matrix multiply and fully connected family ops to
    DiveTpu_EnqueueDmaDescriptor through lookupOrCreateFn. It checks each op with
    checkDwcConvertibleTypes, rejects multi-result and unknown offload ops, and records
    tpu_offload.lowered_count.
  }];
}

def DwcConvertXlaSupportedStablehloPass : Pass<"convert-xla-supported-stablehlo", "func::FuncOp"> {
  let summary = "DWC pass convert-xla-supported-stablehlo";
  let description = [{
    Seen at 0xdbc2b1 in task3_passes.tsv, this pass folds same-type one-operand one-result stablehlo
    dialect ops identities with replaceAllUsesWith and erase. It then runs checkDwcConvertibleTypes over
    the stablehlo dialect ops and leaves every other op untouched.
  }];
}

def DwcConvertDiveVmTensorToLinalgSymbolPass : Pass<"ConvertDiveVmTensorToLinalg", "func::FuncOp"> {
  let summary = "DWC pass ConvertDiveVmTensorToLinalg";
  let description = [{
    Seen at 0xde1af2 in task3_passes.tsv, this pass folds same-type one-operand one-result `dive_vm`
    identities with replaceAllUsesWith and erase. It then runs checkDwcConvertibleTypes over the dive_vm
    dialect ops and leaves every other op untouched.
  }];
}

def DwcConvertTpuOffloadToLlvmSymbolPass : Pass<"ConvertTpuOffloadToLlvm", "func::FuncOp"> {
  let summary = "DWC pass ConvertTpuOffloadToLlvm";
  let description = [{
    Seen at 0xdcb87c in task3_passes.tsv, this pass lowers edgetpu convolution and attention family ops
    to DiveTpu_EnqueueInstructions calls and edgetpu matrix multiply and fully connected family ops to
    DiveTpu_EnqueueDmaDescriptor calls through lookupOrCreateFn. It checks each op with
    checkDwcConvertibleTypes, rejects multi-result and unknown edgetpu ops, and records
    tpu_offload.lowered_count.
  }];
}

def DwcCopyOpLoweringPass : Pass<"copy-op-lowering", "func::FuncOp"> {
  let summary = "DWC pass copy-op-lowering";
  let description = [{
    Seen at 0xddedbd in task3_passes.tsv, this pass applies populateLowerCopySlicePatterns greedily and
    runs applyLocalCopySliceLowering. It then lowers copy ops to dive_vm.copy with applyDwcLowerCopyLike
    under matching element types and counts.
  }];
}

def DwcDarwinnBundlingPass : Pass<"darwinn-bundling", "func::FuncOp"> {
  let summary = "DWC pass darwinn-bundling";
  let description = [{
    Seen at 0xde00af in task3_passes.tsv, this pass folds same-type one-operand one-result
    `darwinn.bitcast`, `darwinn.convert` and `darwinn.copy_op` identities with replaceAllUsesWith and
    erase. It then runs checkDwcConvertibleTypes over the darwinn dialect ops and leaves every other op
    untouched.
  }];
}

def DwcDarwinnConvertPass : Pass<"darwinn.convert", "func::FuncOp"> {
  let summary = "DWC pass darwinn.convert";
  let description = [{
    Seen at 0xd68ca7 in task3_passes.tsv, this pass lowers darwinn.convert, darwinn.cast_in and
    darwinn.cast_out to dive_vm.cast with applyDwcLowerConvertTrunc. It keeps one-operand one-result ops
    with matching ranked shapes and numeric element types, folding same-type cases in place, and checks
    each with checkDwcConvertibleTypes.
  }];
}

def DwcDarwinnMathJoinPass : Pass<"darwinn.math.join", "func::FuncOp"> {
  let summary = "DWC pass darwinn.math.join";
  let description = [{
    Seen at 0xdc9452 in task3_passes.tsv, this pass lowers darwinn.tgc_elementwise_add,
    darwinn.tgc_elementwise_mul and darwinn.tgc_elementwise_sub to arith addi, muli and subi or addf,
    mulf and subf by element type with applyDwcLowerScalarArith. It needs two operands and one result
    sharing one ranked shape, forwarded with forwardDwcLowerOp.
  }];
}

def DwcDarwinnSparsityPass : Pass<"darwinn.sparsity", "func::FuncOp"> {
  let summary = "DWC pass darwinn.sparsity";
  let description = [{
    Seen at 0xd5a035 in task3_passes.tsv, this pass folds same-type one-operand one-result
    `darwinn.sparsity` identities with replaceAllUsesWith and erase. It then runs
    checkDwcConvertibleTypes over the `darwinn.sparsity` and leaves every other op untouched.
  }];
}

def DwcDiveDcePass : Pass<"dive-dce", "func::FuncOp"> {
  let summary = "DWC pass dive-dce";
  let description = [{
    Seen at 0xe0fe97 in task3_passes.tsv, this pass erases unused non-terminator regionless
    dive_vm.const and dive_vm.copy ops.
  }];
}

def DwcDiveIoOptimizationPass : Pass<"dive-io-optimization", "func::FuncOp"> {
  let summary = "DWC pass dive-io-optimization";
  let description = [{
    Seen at 0xdc141b in task3_passes.tsv, this pass folds single-use same-type one-operand one-result
    dive_vm.copy identities with replaceAllUsesWith and erase.
  }];
}

def DwcDiveProgramTpuPass : Pass<"dive-program-tpu", "func::FuncOp"> {
  let summary = "DWC pass dive-program-tpu";
  let description = [{
    Seen at 0xd6383a in task3_passes.tsv, this pass groups dive_vm.tpu_offload ops by their tpu.program
    attr into packets, stamping tpu.packet_id, tpu.packet_order, tpu.packet_count and tpu.packet_order.
    Each packet gets one internal LLVM global of dispatch addresses plus one
    DiveRuntime_ExecuteChildModel call built with lookupOrCreateFn.
  }];
}

def DwcDiveUnrollFactorPass : Pass<"dive-unroll-factor", "func::FuncOp"> {
  let summary = "DWC pass dive-unroll-factor";
  let description = [{
    Seen at 0xda744d in task3_passes.tsv, this pass folds same-type one-operand one-result `dive_vm.for`
    and `dive_vm.loop` identities with replaceAllUsesWith and erase. It then runs
    checkDwcConvertibleTypes over the `dive_vm.for` and `dive_vm.loop` and leaves every other op
    untouched.
  }];
}

def DwcDiveVmBufferizePass : Pass<"dive-vm-bufferize", "func::FuncOp"> {
  let summary = "DWC pass dive-vm-bufferize";
  let description = [{
    Seen at 0xde6a70 in task3_passes.tsv, this pass folds same-type one-operand one-result `dive_vm`
    identities with replaceAllUsesWith and erase. It then runs checkDwcConvertibleTypes over the dive_vm
    dialect ops and leaves every other op untouched.
  }];
}

def DwcDiveVmOutlineShareableDiveConstsPass : Pass<"dive-vm-outline-shareable-dive-consts", "func::FuncOp"> {
  let summary = "DWC pass dive-vm-outline-shareable-dive-consts";
  let description = [{
    Seen at 0xd788a7 in task3_passes.tsv, this pass dedupes dive_vm.const ops by their value attr with a
    DenseMap, rerouting duplicates with replaceAllUsesWith and erasing them.
  }];
}

def DwcDwcCheckIllegalTpuOpsPass : Pass<"dwc-check-illegal-tpu-ops", "func::FuncOp"> {
  let summary = "DWC pass dwc-check-illegal-tpu-ops";
  let description = [{
    Seen at 0xd84494 in task3_passes.tsv, this pass folds same-type one-operand one-result
    `darwinn.bitcast`, `darwinn.convert` and `darwinn.copy_op` identities with replaceAllUsesWith and
    erase. It then runs checkDwcConvertibleTypes over the `darwinn.bitcast`, `darwinn.convert` and
    `darwinn.copy_op` and leaves every other op untouched.
  }];
}

def DwcDwcConvertInputOutputTypesPass : Pass<"dwc-convert-input-output-types", "func::FuncOp"> {
  let summary = "DWC pass dwc-convert-input-output-types";
  let description = [{
    Seen at 0xd94dcc in task3_passes.tsv, this pass walks ops in order and joins each op to the cluster
    of its operand defining op, stamping tpu.cluster_id per op and tpu.cluster_count on the function.
  }];
}

def DwcDwcCopyStridedBuffersOnTpuPass : Pass<"dwc-copy-strided-buffers-on-tpu", "func::FuncOp"> {
  let summary = "DWC pass dwc-copy-strided-buffers-on-tpu";
  let description = [{
    Seen at 0xd6381a in task3_passes.tsv, this pass lowers darwinn.copy_op, darwinn.copy_from_host and
    darwinn.copy_using_wide to dive_vm.copy with applyDwcLowerCopyLike. It keeps one-operand one-result
    ops with matching element types and static element counts, checked with checkDwcConvertibleTypes.
  }];
}

def DwcDwcFormTpuClustersPass : Pass<"dwc-form-tpu-clusters", "func::FuncOp"> {
  let summary = "DWC pass dwc-form-tpu-clusters";
  let description = [{
    Seen at 0xd81a1f in task3_passes.tsv, this pass walks ops in order and joins each op to the cluster
    of its operand defining op, stamping tpu.cluster_id per op and tpu.cluster_count on the function.
  }];
}

def DwcDwcLegalizePass : Pass<"dwc-legalize", "func::FuncOp"> {
  let summary = "DWC pass dwc-legalize";
  let description = [{
    Seen at 0xde6d3f in task3_passes.tsv, this pass folds same-type one-operand one-result
    `darwinn.bitcast`, `darwinn.convert` and `darwinn.copy_op` identities with replaceAllUsesWith and
    erase. It then runs checkDwcConvertibleTypes over the darwinn dialect ops and leaves every other op
    untouched.
  }];
}

def DwcDwcLegalizeHloPass : Pass<"dwc-legalize-hlo", "func::FuncOp"> {
  let summary = "DWC pass dwc-legalize-hlo";
  let description = [{
    Seen at 0xdbc39d in task3_passes.tsv, this pass folds same-type one-operand one-result `mhlo`
    identities with replaceAllUsesWith and erase. It then runs checkDwcConvertibleTypes over the mhlo
    dialect ops and leaves every other op untouched.
  }];
}

def DwcDwcLegalizeHloToTfPass : Pass<"dwc-legalize-hlo-to-tf", "func::FuncOp"> {
  let summary = "DWC pass dwc-legalize-hlo-to-tf";
  let description = [{
    Seen at 0xde2ad1 in task3_passes.tsv, this pass folds same-type one-operand one-result `mhlo`
    identities with replaceAllUsesWith and erase. It then runs checkDwcConvertibleTypes over the mhlo
    and tf dialect ops and leaves every other op untouched.
  }];
}

def DwcDwcLegalizeIntAndQuantTypesPass : Pass<"dwc-legalize-int-and-quant-types", "func::FuncOp"> {
  let summary = "DWC pass dwc-legalize-int-and-quant-types";
  let description = [{
    Seen at 0xd94e00 in task3_passes.tsv, this pass folds same-type one-operand one-result quant dialect
    ops identities with replaceAllUsesWith and erase. It then runs checkDwcConvertibleTypes over the
    quant dialect ops and leaves every other op untouched.
  }];
}

def DwcDwcLegalizeInt64ConstantsPass : Pass<"dwc-legalize-int64-constants", "func::FuncOp"> {
  let summary = "DWC pass dwc-legalize-int64-constants";
  let description = [{
    Seen at 0xd7bb5d in task3_passes.tsv, this pass folds same-type one-operand one-result
    `darwinn.bitcast`, `darwinn.convert` and `darwinn.copy_op` identities with replaceAllUsesWith and
    erase. It then runs checkDwcConvertibleTypes over the darwinn dialect ops and leaves every other op
    untouched.
  }];
}

def DwcDwcLegalizePassSymbol : Pass<"dwc-legalize-pass", "func::FuncOp"> {
  let summary = "DWC pass dwc-legalize-pass";
  let description = [{
    Seen at 0xd7f2b9 in task3_passes.tsv, this pass folds same-type one-operand one-result
    `darwinn.bitcast`, `darwinn.convert` and `darwinn.copy_op` identities with replaceAllUsesWith and
    erase. It then runs checkDwcConvertibleTypes over the darwinn dialect ops and leaves every other op
    untouched.
  }];
}

def DwcDwcLegalizeStablehloAnnotateMaterializePolicyPass : Pass<"dwc-legalize-stablehlo-annotate-materialize-policy", "func::FuncOp"> {
  let summary = "DWC pass dwc-legalize-stablehlo-annotate-materialize-policy";
  let description = [{
    Seen at 0xd5dd77 in task3_passes.tsv, this pass folds same-type one-operand one-result stablehlo
    dialect ops identities with replaceAllUsesWith and erase. It then runs checkDwcConvertibleTypes over
    the stablehlo dialect ops and leaves every other op untouched.
  }];
}

def DwcDwcLegalizeStablehloCompositePass : Pass<"dwc-legalize-stablehlo-composite", "func::FuncOp"> {
  let summary = "DWC pass dwc-legalize-stablehlo-composite";
  let description = [{
    Seen at 0xdef599 in task3_passes.tsv, this pass folds same-type one-operand one-result stablehlo
    dialect ops identities with replaceAllUsesWith and erase. It then runs checkDwcConvertibleTypes over
    the stablehlo dialect ops and leaves every other op untouched.
  }];
}

def DwcDwcLegalizeTfPipelinePass : Pass<"dwc-legalize-tf-pipeline", "func::FuncOp"> {
  let summary = "DWC pass dwc-legalize-tf-pipeline";
  let description = [{
    Seen at 0xe03284 in task3_passes.tsv, this pass folds same-type one-operand one-result tf dialect
    ops identities with replaceAllUsesWith and erase. It then runs checkDwcConvertibleTypes over the tf
    dialect ops and leaves every other op untouched.
  }];
}

def DwcDwcLegalizeTflCudaemuCustomOpsPass : Pass<"dwc-legalize-tfl-cudaemu-custom-ops", "func::FuncOp"> {
  let summary = "DWC pass dwc-legalize-tfl-cudaemu-custom-ops";
  let description = [{
    Seen at 0xd8464e in task3_passes.tsv, this pass folds same-type one-operand one-result tfl dialect
    ops identities with replaceAllUsesWith and erase. It then runs checkDwcConvertibleTypes over the tfl
    dialect ops and leaves every other op untouched.
  }];
}

def DwcDwcLegalizeTflMultinomialPass : Pass<"dwc-legalize-tfl-multinomial", "func::FuncOp"> {
  let summary = "DWC pass dwc-legalize-tfl-multinomial";
  let description = [{
    Seen at 0xdd3cdb in task3_passes.tsv, this pass folds same-type one-operand one-result tfl dialect
    ops identities with replaceAllUsesWith and erase. It then runs checkDwcConvertibleTypes over the tfl
    dialect ops and leaves every other op untouched.
  }];
}

def DwcDwcLegalizeTflVariableTensorsPass : Pass<"dwc-legalize-tfl-variable-tensors", "func::FuncOp"> {
  let summary = "DWC pass dwc-legalize-tfl-variable-tensors";
  let description = [{
    Seen at 0xd81097 in task3_passes.tsv, this pass folds same-type one-operand one-result tfl dialect
    ops identities with replaceAllUsesWith and erase. It then runs checkDwcConvertibleTypes over the tfl
    dialect ops and leaves every other op untouched.
  }];
}

def DwcDwcLegalizeUint32TypesPass : Pass<"dwc-legalize-uint32-types", "func::FuncOp"> {
  let summary = "DWC pass dwc-legalize-uint32-types";
  let description = [{
    Seen at 0xd94e3e in task3_passes.tsv, this pass folds same-type one-operand one-result `darwinn`
    identities with replaceAllUsesWith and erase. It then runs checkDwcConvertibleTypes over the darwinn
    dialect ops and leaves every other op untouched.
  }];
}

def DwcDwcLowerArgmaxIndexUnpoolPass : Pass<"dwc-lower-argmax-index-unpool", "func::FuncOp"> {
  let summary = "DWC pass dwc-lower-argmax-index-unpool";
  let description = [{
    Seen at 0xdcf030 in task3_passes.tsv, this pass lowers darwinn.mask_indices to dive_vm.mask_indices
    with applyDwcLowerArgmaxInline through forwardDwcLowerTo.
  }];
}

def DwcDwcLowerCompositeOpsPass : Pass<"dwc-lower-composite-ops", "func::FuncOp"> {
  let summary = "DWC pass dwc-lower-composite-ops";
  let description = [{
    Seen at 0xd84775 in task3_passes.tsv, this pass applies populateLowerCopySlicePatterns and
    populateLowerConvertPatterns greedily, then runs applyLocalCopySliceLowering and
    applyLocalConvertLowering. The locals fold same-type darwinn.copy_op identities and same-type
    darwinn.convert and darwinn.bitcast identities.
  }];
}

def DwcDwcLowerControlFlowPass : Pass<"dwc-lower-control-flow", "func::FuncOp"> {
  let summary = "DWC pass dwc-lower-control-flow";
  let description = [{
    Seen at 0xd61f2c in task3_passes.tsv, this pass folds same-type one-operand one-result
    `darwinn.bitcast`, `darwinn.convert` and `darwinn.copy_op` identities with replaceAllUsesWith and
    erase. It then requires every remaining op to sit in the darwinn dialect and pass
    checkDwcConvertibleTypes, failing other dialects with dwc-lower-control-flow accepts darwinn ops
    only.
  }];
}

def DwcDwcLowerDepthToFromSpacePass : Pass<"dwc-lower-depth-to-from-space", "func::FuncOp"> {
  let summary = "DWC pass dwc-lower-depth-to-from-space";
  let description = [{
    Seen at 0xe102d0 in task3_passes.tsv, this pass applies populateLowerCopySlicePatterns and
    populateLowerConvertPatterns greedily, then runs applyLocalCopySliceLowering and
    applyLocalConvertLowering. The locals fold same-type darwinn.copy_op identities and same-type
    darwinn.convert and darwinn.bitcast identities.
  }];
}

def DwcDwcLowerGenericConstantsPass : Pass<"dwc-lower-generic-constants", "func::FuncOp"> {
  let summary = "DWC pass dwc-lower-generic-constants";
  let description = [{
    Seen at 0xd7bb41 in task3_passes.tsv, this pass applies populateLowerCopySlicePatterns and
    populateLowerConvertPatterns greedily, then runs applyLocalCopySliceLowering and
    applyLocalConvertLowering. The locals fold same-type darwinn.copy_op identities and same-type
    darwinn.convert and darwinn.bitcast identities.
  }];
}

def DwcDwcLowerHlopsPass : Pass<"dwc-lower-hlops", "func::FuncOp"> {
  let summary = "DWC pass dwc-lower-hlops";
  let description = [{
    Seen at 0xd841f4 in task3_passes.tsv, this pass applies populateLowerCopySlicePatterns and
    populateLowerConvertPatterns greedily, then runs applyLocalCopySliceLowering and
    applyLocalConvertLowering. The locals fold same-type darwinn.copy_op identities and same-type
    darwinn.convert and darwinn.bitcast identities.
  }];
}

def DwcDwcLowerInputOutputCastPass : Pass<"dwc-lower-input-output-cast", "func::FuncOp"> {
  let summary = "DWC pass dwc-lower-input-output-cast";
  let description = [{
    Seen at 0xd68084 in task3_passes.tsv, this pass applies populateLowerCopySlicePatterns and
    populateLowerConvertPatterns greedily, then runs applyLocalCopySliceLowering and
    applyLocalConvertLowering. The locals fold same-type darwinn.copy_op identities and same-type
    darwinn.convert and darwinn.bitcast identities.
  }];
}

def DwcDwcLowerPaddingOpsPass : Pass<"dwc-lower-padding-ops", "func::FuncOp"> {
  let summary = "DWC pass dwc-lower-padding-ops";
  let description = [{
    Seen at 0xd84749 in task3_passes.tsv, this pass applies populateLowerCopySlicePatterns and
    populateLowerConvertPatterns greedily, then runs applyLocalCopySliceLowering and
    applyLocalConvertLowering. The locals fold same-type darwinn.copy_op identities and same-type
    darwinn.convert and darwinn.bitcast identities.
  }];
}

def DwcDwcLowerPseudoOpsPass : Pass<"dwc-lower-pseudo-ops", "func::FuncOp"> {
  let summary = "DWC pass dwc-lower-pseudo-ops";
  let description = [{
    Seen at 0xd845c5 in task3_passes.tsv, this pass applies populateLowerCopySlicePatterns and
    populateLowerConvertPatterns greedily, then runs applyLocalCopySliceLowering and
    applyLocalConvertLowering. The locals fold same-type darwinn.copy_op identities and same-type
    darwinn.convert and darwinn.bitcast identities.
  }];
}

def DwcDwcLowerResamplerOpsPass : Pass<"dwc-lower-resampler-ops", "func::FuncOp"> {
  let summary = "DWC pass dwc-lower-resampler-ops";
  let description = [{
    Seen at 0xd84598 in task3_passes.tsv, this pass applies populateLowerCopySlicePatterns and
    populateLowerConvertPatterns greedily, then runs applyLocalCopySliceLowering and
    applyLocalConvertLowering. The locals fold same-type darwinn.copy_op identities and same-type
    darwinn.convert and darwinn.bitcast identities.
  }];
}

def DwcDwcLowerScalarOpsPass : Pass<"dwc-lower-scalar-ops", "func::FuncOp"> {
  let summary = "DWC pass dwc-lower-scalar-ops";
  let description = [{
    Seen at 0xd845b0 in task3_passes.tsv, this pass applies populateLowerCopySlicePatterns and
    populateLowerConvertPatterns greedily, then runs applyLocalCopySliceLowering and
    applyLocalConvertLowering. The locals fold same-type darwinn.copy_op identities and same-type
    darwinn.convert and darwinn.bitcast identities.
  }];
}

def DwcDwcLowerScatterOpsPass : Pass<"dwc-lower-scatter-ops", "func::FuncOp"> {
  let summary = "DWC pass dwc-lower-scatter-ops";
  let description = [{
    Seen at 0xd84582 in task3_passes.tsv, this pass applies populateLowerCopySlicePatterns and
    populateLowerConvertPatterns greedily, then runs applyLocalCopySliceLowering and
    applyLocalConvertLowering. The locals fold same-type darwinn.copy_op identities and same-type
    darwinn.convert and darwinn.bitcast identities.
  }];
}

def DwcDwcLowerTopKPass : Pass<"dwc-lower-top-k", "func::FuncOp"> {
  let summary = "DWC pass dwc-lower-top-k";
  let description = [{
    Seen at 0xdd6da3 in task3_passes.tsv, this pass applies populateLowerCopySlicePatterns and
    populateLowerConvertPatterns greedily, then runs applyLocalCopySliceLowering and
    applyLocalConvertLowering. The locals fold same-type darwinn.copy_op identities and same-type
    darwinn.convert and darwinn.bitcast identities.
  }];
}

def DwcDwcPostTruncationTpuFitterPass : Pass<"dwc-post-truncation-tpu-fitter", "func::FuncOp"> {
  let summary = "DWC pass dwc-post-truncation-tpu-fitter";
  let description = [{
    Seen at 0xdab7fc in task3_passes.tsv, this pass walks ops in order and joins each op to the cluster
    of its operand defining op, stamping tpu.cluster_id per op and tpu.cluster_count on the function.
  }];
}

def DwcDwcPreTpuFitterOptimizeGatherPass : Pass<"dwc-pre-tpu-fitter-optimize-gather", "func::FuncOp"> {
  let summary = "DWC pass dwc-pre-tpu-fitter-optimize-gather";
  let description = [{
    Seen at 0xdaf5f9 in task3_passes.tsv, this pass walks ops in order and joins each op to the cluster
    of its operand defining op, stamping tpu.cluster_id per op and tpu.cluster_count on the function.
  }];
}

def DwcDwcPreTpuFitterOptimizeScatterPass : Pass<"dwc-pre-tpu-fitter-optimize-scatter", "func::FuncOp"> {
  let summary = "DWC pass dwc-pre-tpu-fitter-optimize-scatter";
  let description = [{
    Seen at 0xdabc37 in task3_passes.tsv, this pass walks ops in order and joins each op to the cluster
    of its operand defining op, stamping tpu.cluster_id per op and tpu.cluster_count on the function.
  }];
}

def DwcDwcRegroupTpuFunctionsPass : Pass<"dwc-regroup-tpu-functions", "func::FuncOp"> {
  let summary = "DWC pass dwc-regroup-tpu-functions";
  let description = [{
    Seen at 0xd873c3 in task3_passes.tsv, this pass walks ops in order and joins each op to the cluster
    of its operand defining op, stamping tpu.cluster_id per op and tpu.cluster_count on the function.
  }];
}

def DwcDwcSerializeTpuOffloadsPass : Pass<"dwc-serialize-tpu-offloads", "func::FuncOp"> {
  let summary = "DWC pass dwc-serialize-tpu-offloads";
  let description = [{
    Seen at 0xd9c33f in task3_passes.tsv, this pass walks ops in order and joins each op to the cluster
    of its operand defining op, stamping tpu.cluster_id per op and tpu.cluster_count on the function.
  }];
}

def DwcDwcTestRepeatTpuOpsPass : Pass<"dwc-test-repeat-tpu-ops", "func::FuncOp"> {
  let summary = "DWC pass dwc-test-repeat-tpu-ops";
  let description = [{
    Seen at 0xd8447c in task3_passes.tsv, this pass walks ops in order and joins each op to the cluster
    of its operand defining op, stamping tpu.cluster_id per op and tpu.cluster_count on the function.
  }];
}

def DwcDwcTpuFitterPass : Pass<"dwc-tpu-fitter", "func::FuncOp"> {
  let summary = "DWC pass dwc-tpu-fitter";
  let description = [{
    Seen at 0xdab81b in task3_passes.tsv, this pass walks ops in order and joins each op to the cluster
    of its operand defining op, stamping tpu.cluster_id per op and tpu.cluster_count on the function.
  }];
}

def DwcDwcTpuFunctionCsePass : Pass<"dwc-tpu-function-cse", "func::FuncOp"> {
  let summary = "DWC pass dwc-tpu-function-cse";
  let description = [{
    Seen at 0xdf32f5 in task3_passes.tsv, this pass dedupes dive_vm.const ops by their value attr with a
    DenseMap, rerouting duplicates with replaceAllUsesWith and erasing them.
  }];
}

def DwcDwgCreateDarwinnCustomOpPass : Pass<"dwg-create-darwinn-custom-op", "func::FuncOp"> {
  let summary = "DWC pass dwg-create-darwinn-custom-op";
  let description = [{
    Seen at 0xdb45e0 in task3_passes.tsv, this pass folds same-type one-operand one-result
    `darwinn.bitcast`, `darwinn.convert` and `darwinn.copy_op` identities with replaceAllUsesWith and
    erase. It then runs checkDwcConvertibleTypes over the darwinn dialect ops and leaves every other op
    untouched.
  }];
}

def DwcDwgForkMulticoreTpuOffloadsPass : Pass<"dwg-fork-multicore-tpu-offloads", "func::FuncOp"> {
  let summary = "DWC pass dwg-fork-multicore-tpu-offloads";
  let description = [{
    Seen at 0xd9c35a in task3_passes.tsv, this pass numbers each dive_vm.tpu_offload with a sequential
    tpu.core_id attr and records the total in tpu.core_count.
  }];
}

def DwcDwgLowerForToWhilePass : Pass<"dwg-lower-for-to-while", "func::FuncOp"> {
  let summary = "DWC pass dwg-lower-for-to-while";
  let description = [{
    Seen at 0xe07f47 in task3_passes.tsv, this pass applies populateLowerCopySlicePatterns and
    populateLowerConvertPatterns greedily, then runs applyLocalCopySliceLowering and
    applyLocalConvertLowering. The locals fold same-type darwinn.copy_op identities and same-type
    darwinn.convert and darwinn.bitcast identities.
  }];
}

def DwcDwgtLowerIndexTypePass : Pass<"dwgt-lower-index-type", "func::FuncOp"> {
  let summary = "DWC pass dwgt-lower-index-type";
  let description = [{
    Seen at 0xdf6914 in task3_passes.tsv, this pass applies populateLowerCopySlicePatterns and
    populateLowerConvertPatterns greedily, then runs applyLocalCopySliceLowering and
    applyLocalConvertLowering. The locals fold same-type darwinn.copy_op identities and same-type
    darwinn.convert and darwinn.bitcast identities.
  }];
}

def DwcDynamicUpdateSliceLoweringPass : Pass<"dynamic-update-slice-lowering", "func::FuncOp"> {
  let summary = "DWC pass dynamic-update-slice-lowering";
  let description = [{
    Seen at 0xddee37 in task3_passes.tsv, this pass applies populateLowerCopySlicePatterns greedily.
  }];
}

def DwcEdgetpuCustomOp2Pass : Pass<"edgetpu-custom-op-2", "func::FuncOp"> {
  let summary = "DWC pass edgetpu-custom-op-2";
  let description = [{
    Seen at 0x103fe70 in task3_passes.tsv, this pass walks ops in order and joins each op to the cluster
    of its operand defining op, stamping tpu.cluster_id per op and tpu.cluster_count on the function.
  }];
}

def DwcFmModelConverterPass : Pass<"fm-model-converter", "func::FuncOp"> {
  let summary = "DWC pass fm-model-converter";
  let description = [{
    Seen at 0xdac265 in task3_passes.tsv, this pass folds same-type one-operand one-result
    `darwinn.bitcast`, `darwinn.convert` and `darwinn.copy_op` identities with replaceAllUsesWith and
    erase. It then requires every remaining op to sit in the darwinn dialect and pass
    checkDwcConvertibleTypes, failing other dialects with fm-model-converter accepts darwinn ops only.
  }];
}

def DwcFpa2bvModelConverterPass : Pass<"fpa2bv-model-converter", "func::FuncOp"> {
  let summary = "DWC pass fpa2bv-model-converter";
  let description = [{
    Seen at 0xdac220 in task3_passes.tsv, this pass folds same-type one-operand one-result
    `darwinn.bitcast`, `darwinn.convert` and `darwinn.copy_op` identities with replaceAllUsesWith and
    erase. It then requires every remaining op to sit in the darwinn dialect and pass
    checkDwcConvertibleTypes, failing other dialects with fpa2bv-model-converter accepts darwinn ops
    only.
  }];
}

def DwcGroupTpuOffloadsByParametersPass : Pass<"group-tpu-offloads-by-parameters", "func::FuncOp"> {
  let summary = "DWC pass group-tpu-offloads-by-parameters";
  let description = [{
    Seen at 0xd821ce in task3_passes.tsv, this pass groups dive_vm.tpu_offload ops by their printed
    operand type list, stamping tpu.param_group on each op and tpu.param_group_count on the function.
  }];
}

def DwcInterpolateLoweringPassPass : Pass<"interpolate-lowering-pass", "func::FuncOp"> {
  let summary = "DWC pass interpolate-lowering-pass";
  let description = [{
    Seen at 0xd7f143 in task3_passes.tsv, this pass folds same-type one-operand one-result
    `darwinn.interpolate`, `darwinn.interpolate_hardware`, `darwinn.interpolate_method` and
    `darwinn.legacy_interpolate` identities with replaceAllUsesWith and erase. It then requires every
    remaining op to sit in the darwinn dialect and pass checkDwcConvertibleTypes, failing other dialects
    with interpolate-lowering-pass accepts darwinn ops only.
  }];
}

def DwcIsNotImmutableTryRemovingMutableVariablesInYourModelSinceMutableVariablesAreCurrentlyNotSupportedThroughThisConverterPass : Pass<"is not immutable, try removing mutable variables in your model since mutable variables are currently not supported through this converter", "func::FuncOp"> {
  let summary = "DWC pass is-not-immutable-try-removing-mutable-variables-in-your-model-since-mutable-variables-are-currently-not-supported-through-this-converter";
  let description = [{
    Seen at 0xdac290 in task3_passes.tsv, this pass stamps every non-function op with a pass-named
    marked unit attr after checkDwcConvertibleTypes and records the total in a pass-named marked_count
    attr on the function.
  }];
}

def DwcLegalizePass : Pass<"Legalize", "func::FuncOp"> {
  let summary = "DWC pass Legalize";
  let description = [{
    Seen at 0xde6d4c in task3_passes.tsv, this pass folds same-type one-operand one-result
    `darwinn.bitcast`, `darwinn.convert` and `darwinn.copy_op` identities with replaceAllUsesWith and
    erase. It then runs checkDwcConvertibleTypes over the darwinn dialect ops and leaves every other op
    untouched.
  }];
}

def DwcLegalizeAffinePass : Pass<"legalize-affine", "func::FuncOp"> {
  let summary = "DWC pass legalize-affine";
  let description = [{
    Seen at 0xe03694 in task3_passes.tsv, this pass folds same-type one-operand one-result affine
    dialect ops identities with replaceAllUsesWith and erase. It then runs checkDwcConvertibleTypes over
    the affine dialect ops and leaves every other op untouched.
  }];
}

def DwcLegalizeDwcPass : Pass<"legalize-dwc", "func::FuncOp"> {
  let summary = "DWC pass legalize-dwc";
  let description = [{
    Seen at 0xe25250 in task3_passes.tsv, this pass folds same-type one-operand one-result
    `darwinn.bitcast`, `darwinn.convert` and `darwinn.copy_op` identities with replaceAllUsesWith and
    erase. It then runs checkDwcConvertibleTypes over the darwinn dialect ops and leaves every other op
    untouched.
  }];
}

def DwcLegalizeDwcInputOutputOpsPass : Pass<"legalize-dwc-input-output-ops", "func::FuncOp"> {
  let summary = "DWC pass legalize-dwc-input-output-ops";
  let description = [{
    Seen at 0xd844c7 in task3_passes.tsv, this pass folds same-type one-operand one-result
    `darwinn.copy_from_host` and `darwinn.copy_op` identities with replaceAllUsesWith and erase. It then
    runs checkDwcConvertibleTypes over the darwinn dialect ops and leaves every other op untouched.
  }];
}

def DwcLegalizeDwgTensorPass : Pass<"legalize-dwg-tensor", "func::FuncOp"> {
  let summary = "DWC pass legalize-dwg-tensor";
  let description = [{
    Seen at 0xda8a5b in task3_passes.tsv, this pass folds same-type one-operand one-result dwg dialect
    ops identities with replaceAllUsesWith and erase. It then runs checkDwcConvertibleTypes over the dwg
    dialect ops and leaves every other op untouched.
  }];
}

def DwcLegalizeQuantTypesPass : Pass<"legalize-quant-types", "func::FuncOp"> {
  let summary = "DWC pass legalize-quant-types";
  let description = [{
    Seen at 0xd94deb in task3_passes.tsv, this pass folds same-type one-operand one-result quant dialect
    ops identities with replaceAllUsesWith and erase. It then runs checkDwcConvertibleTypes over the
    quant dialect ops and leaves every other op untouched.
  }];
}

def DwcLegalizeScfPass : Pass<"legalize-scf", "func::FuncOp"> {
  let summary = "DWC pass legalize-scf";
  let description = [{
    Seen at 0xde466f in task3_passes.tsv, this pass folds same-type one-operand one-result scf dialect
    ops identities with replaceAllUsesWith and erase. It then runs checkDwcConvertibleTypes over the scf
    dialect ops and leaves every other op untouched.
  }];
}

def DwcLegalizeShapeOpsPass : Pass<"legalize-shape-ops", "func::FuncOp"> {
  let summary = "DWC pass legalize-shape-ops";
  let description = [{
    Seen at 0xd84833 in task3_passes.tsv, this pass folds same-type one-operand one-result shape dialect
    ops identities with replaceAllUsesWith and erase. It then runs checkDwcConvertibleTypes over the
    shape dialect ops and leaves every other op untouched.
  }];
}

def DwcLegalizeTestUsingLayerirFlowPass : Pass<"legalize test using layerir flow", "func::FuncOp"> {
  let summary = "DWC pass legalize-test-using-layerir-flow";
  let description = [{
    Seen at 0xd61f43 in task3_passes.tsv, this pass folds same-type one-operand one-result
    `darwinn.bitcast`, `darwinn.convert` and `darwinn.copy_op` identities with replaceAllUsesWith and
    erase. It then runs checkDwcConvertibleTypes over the darwinn dialect ops and leaves every other op
    untouched.
  }];
}

def DwcLegalizeTfXlacallmoduleOpToStablehloPass : Pass<"Legalize TF_XlaCallModule Op to stablehlo", "func::FuncOp"> {
  let summary = "DWC pass legalize-tf-xlacallmodule-op-to-stablehlo";
  let description = [{
    Seen at 0xdbc2d1 in task3_passes.tsv, this pass folds same-type one-operand one-result tf dialect
    ops identities with replaceAllUsesWith and erase. It then runs checkDwcConvertibleTypes over the tf
    dialect ops and leaves every other op untouched.
  }];
}

def DwcLegalizeThreadObliviousOpPassPass : Pass<"legalize-thread-oblivious-op-pass", "func::FuncOp"> {
  let summary = "DWC pass legalize-thread-oblivious-op-pass";
  let description = [{
    Seen at 0xd7ee8c in task3_passes.tsv, this pass folds same-type one-operand one-result
    `darwinn.bitcast`, `darwinn.convert` and `darwinn.copy_op` identities with replaceAllUsesWith and
    erase. It then runs checkDwcConvertibleTypes over the darwinn dialect ops and leaves every other op
    untouched.
  }];
}

def DwcLegalizeTypesForDiveVmTensorPass : Pass<"legalize-types-for-dive-vm-tensor", "func::FuncOp"> {
  let summary = "DWC pass legalize-types-for-dive-vm-tensor";
  let description = [{
    Seen at 0xda8a1b in task3_passes.tsv, this pass folds same-type one-operand one-result
    `dive_vm.copy` identities with replaceAllUsesWith and erase. It then runs checkDwcConvertibleTypes
    over the dive_vm dialect ops and leaves every other op untouched.
  }];
}

def DwcLegalizeStablehloCompositePass : Pass<"LegalizeStablehloComposite", "func::FuncOp"> {
  let summary = "DWC pass LegalizeStablehloComposite";
  let description = [{
    Seen at 0xdef61e in task3_passes.tsv, this pass folds same-type one-operand one-result stablehlo
    dialect ops identities with replaceAllUsesWith and erase. It then runs checkDwcConvertibleTypes over
    the stablehlo dialect ops and leaves every other op untouched.
  }];
}

def DwcLowerAllFunctionsPass : Pass<"lower-all-functions", "func::FuncOp"> {
  let summary = "DWC pass lower-all-functions";
  let description = [{
    Seen at 0xd873dd in task3_passes.tsv, this pass lowers darwinn.rkhy_custom_padding and
    darwinn.mesh_pad_slice to dive_vm.pad with applyDwcLowerPadInline through forwardDwcLowerTo.
  }];
}

def DwcLowerAllPadsPass : Pass<"lower-all-pads", "func::FuncOp"> {
  let summary = "DWC pass lower-all-pads";
  let description = [{
    Seen at 0xd9c263 in task3_passes.tsv, this pass lowers darwinn.rkhy_custom_padding and
    darwinn.mesh_pad_slice to dive_vm.pad with applyDwcLowerPadInline. It also lowers darwinn.copy_op,
    darwinn.copy_from_host and darwinn.copy_using_wide to dive_vm.copy with applyDwcLowerCopyLike under
    matching element types and counts.
  }];
}

def DwcLowerAttentionOpsPass : Pass<"lower-attention-ops", "func::FuncOp"> {
  let summary = "DWC pass lower-attention-ops";
  let description = [{
    Seen at 0xd845ec in task3_passes.tsv, this pass applies populateLowerCopySlicePatterns and
    populateLowerConvertPatterns greedily, then runs applyLocalCopySliceLowering and
    applyLocalConvertLowering. The locals fold same-type darwinn.copy_op identities and same-type
    darwinn.convert and darwinn.bitcast identities.
  }];
}

def DwcLowerInputCastPass : Pass<"lower-input-cast", "func::FuncOp"> {
  let summary = "DWC pass lower-input-cast";
  let description = [{
    Seen at 0xd680b2 in task3_passes.tsv, this pass applies populateLowerCopySlicePatterns and
    populateLowerConvertPatterns greedily, then runs applyLocalCopySliceLowering and
    applyLocalConvertLowering. The locals fold same-type darwinn.copy_op identities and same-type
    darwinn.convert and darwinn.bitcast identities.
  }];
}

def DwcLowerJoinPass : Pass<"lower-join", "func::FuncOp"> {
  let summary = "DWC pass lower-join";
  let description = [{
    Seen at 0xdc9464 in task3_passes.tsv, this pass applies populateLowerCopySlicePatterns and
    populateLowerConvertPatterns greedily, then runs applyLocalCopySliceLowering and
    applyLocalConvertLowering. The locals fold same-type darwinn.copy_op identities and same-type
    darwinn.convert and darwinn.bitcast identities.
  }];
}

def DwcLowerOutputCastPass : Pass<"lower-output-cast", "func::FuncOp"> {
  let summary = "DWC pass lower-output-cast";
  let description = [{
    Seen at 0xd680a0 in task3_passes.tsv, this pass applies populateLowerCopySlicePatterns and
    populateLowerConvertPatterns greedily, then runs applyLocalCopySliceLowering and
    applyLocalConvertLowering. The locals fold same-type darwinn.copy_op identities and same-type
    darwinn.convert and darwinn.bitcast identities.
  }];
}

def DwcLowerArgmaxIndexUnpoolPass : Pass<"LowerArgmaxIndexUnpool", "func::FuncOp"> {
  let summary = "DWC pass LowerArgmaxIndexUnpool";
  let description = [{
    Seen at 0xdcf04e in task3_passes.tsv, this pass lowers darwinn.mask_indices to dive_vm.mask_indices
    with applyDwcLowerArgmaxInline through forwardDwcLowerTo.
  }];
}

def DwcMarkDiveVmTensorInsertSliceOpsPass : Pass<"mark-dive-vm-tensor-insert-slice-ops", "func::FuncOp"> {
  let summary = "DWC pass mark-dive-vm-tensor-insert-slice-ops";
  let description = [{
    Seen at 0xd8488d in task3_passes.tsv, this pass folds same-type one-operand one-result
    `dive_vm.extract_slice` and `dive_vm.insert_slice` identities with replaceAllUsesWith and erase. It
    then runs checkDwcConvertibleTypes over the dive_vm dialect ops and leaves every other op untouched.
  }];
}

def DwcMhloLegalizeEinsumToDotGeneralPass : Pass<"mhlo-legalize-einsum-to-dot-general", "func::FuncOp"> {
  let summary = "DWC pass mhlo-legalize-einsum-to-dot-general";
  let description = [{
    Seen at 0xdd34bf in task3_passes.tsv, this pass folds same-type one-operand one-result `mhlo`
    identities with replaceAllUsesWith and erase. It then runs checkDwcConvertibleTypes over the mhlo
    dialect ops and leaves every other op untouched.
  }];
}

def DwcMidToLowLevelLoweringPass : Pass<"mid-to-low-level-lowering", "func::FuncOp"> {
  let summary = "DWC pass mid-to-low-level-lowering";
  let description = [{
    Seen at 0xddedf4 in task3_passes.tsv, this pass applies populateLowerCopySlicePatterns and
    populateLowerConvertPatterns greedily, then runs applyLocalCopySliceLowering and
    applyLocalConvertLowering. It then runs applyDwcLowerScalarArith, applyDwcLowerCopyLike and
    applyDwcLowerConvertTrunc for the arith, copy and convert families.
  }];
}

def DwcMlirDarwinnComputeEnginePass : Pass<"::mlir::darwinn::compute::Engine", "func::FuncOp"> {
  let summary = "DWC pass mlir-darwinn-compute-engine";
  let description = [{
    Seen at 0xe03654 in task3_passes.tsv, this pass folds same-type one-operand one-result
    `darwinn.bitcast`, `darwinn.convert` and `darwinn.copy_op` identities with replaceAllUsesWith and
    erase. It then runs checkDwcConvertibleTypes over the darwinn dialect ops and leaves every other op
    untouched.
  }];
}

def DwcOnlyDenseelementsattrAreSupportedForConstantLoweringPass : Pass<"Only DenseElementsAttr are supported for constant lowering", "func::FuncOp"> {
  let summary = "DWC pass only-denseelementsattr-are-supported-for-constant-lowering";
  let description = [{
    Seen at 0xddee55 in task3_passes.tsv, this pass lowers arith.constant and darwinn.constant_generator
    to dive_vm.const with applyDwcLowerConstInline. It needs one result carrying a DenseElementsAttr
    value and fails other constant forms, checking each with checkDwcConvertibleTypes.
  }];
}

def DwcOptimizeDiveVmTensorInsertSlicePass : Pass<"optimize-dive-vm-tensor-insert-slice", "func::FuncOp"> {
  let summary = "DWC pass optimize-dive-vm-tensor-insert-slice";
  let description = [{
    Seen at 0xe0fa93 in task3_passes.tsv, this pass folds dive_vm.tensor_insert_slice ops whose first
    operand type matches the result type and whose other two operands are identical, with
    replaceAllUsesWith and erase.
  }];
}

def DwcParameterCachingDiveProgramPass : Pass<"parameter-caching-dive-program", "func::FuncOp"> {
  let summary = "DWC pass parameter-caching-dive-program";
  let description = [{
    Seen at 0xdce5e1 in task3_passes.tsv, this pass dedupes dive_vm.const ops by their value attr with a
    DenseMap, rerouting duplicates with replaceAllUsesWith and erasing them.
  }];
}

def DwcPlatformsDarwinnCodeGeneratorEntryScoreTypePass : Pass<"platforms.darwinn.code_generator.Entry.Score.type", "func::FuncOp"> {
  let summary = "DWC pass platforms-darwinn-code-generator-entry-score-type";
  let description = [{
    Seen at 0xdf68e2 in task3_passes.tsv, this pass folds same-type one-operand one-result `func`
    identities with replaceAllUsesWith and erase. It then runs checkDwcConvertibleTypes over the `func`
    and leaves every other op untouched.
  }];
}

def DwcPlatformsDarwinnCompilerProbeinstrumentationlocationConstraintsFunctionsPass : Pass<"platforms.darwinn.compiler.ProbeInstrumentationLocation.Constraints.functions", "func::FuncOp"> {
  let summary = "DWC pass platforms-darwinn-compiler-probeinstrumentationlocation-constraints-functions";
  let description = [{
    Seen at 0xd87375 in task3_passes.tsv, this pass folds same-type one-operand one-result `func`
    identities with replaceAllUsesWith and erase. It then runs checkDwcConvertibleTypes over the `func`
    and leaves every other op untouched.
  }];
}

def DwcQuantSignednessConvertLoweringPass : Pass<"quant-signedness-convert-lowering", "func::FuncOp"> {
  let summary = "DWC pass quant-signedness-convert-lowering";
  let description = [{
    Seen at 0xdded43 in task3_passes.tsv, this pass lowers darwinn.convert, darwinn.cast_in and
    darwinn.cast_out to dive_vm.cast with applyDwcLowerConvertTrunc. It keeps one-operand one-result ops
    with matching ranked shapes and numeric element types, folding same-type cases in place, and checks
    each with checkDwcConvertibleTypes.
  }];
}

def DwcR52ReadsDiveBuffersPass : Pass<"r52-reads-dive-buffers", "func::FuncOp"> {
  let summary = "DWC pass r52-reads-dive-buffers";
  let description = [{
    Seen at 0xd833a6 in task3_passes.tsv, this pass folds same-type one-operand one-result
    `dive_vm.load` and `dive_vm.load_indirect` identities with replaceAllUsesWith and erase. It then
    requires every remaining op to sit in the dive_vm dialect and pass checkDwcConvertibleTypes, failing
    other dialects with r52-reads-dive-buffers accepts dive_vm ops only.
  }];
}

def DwcRedistributeLoweringPass : Pass<"redistribute-lowering", "func::FuncOp"> {
  let summary = "DWC pass redistribute-lowering";
  let description = [{
    Seen at 0xddee0e in task3_passes.tsv, this pass folds same-type one-operand one-result
    `darwinn.bitcast`, `darwinn.convert` and `darwinn.copy_op` identities with replaceAllUsesWith and
    erase. It then requires every remaining op to sit in the darwinn dialect and pass
    checkDwcConvertibleTypes, failing other dialects with redistribute-lowering accepts darwinn ops
    only.
  }];
}

def DwcRedistributeLoweringPassRemarksPass : Pass<"redistribute-lowering-pass-remarks", "func::FuncOp"> {
  let summary = "DWC pass redistribute-lowering-pass-remarks";
  let description = [{
    Seen at 0xd8c505 in task3_passes.tsv, this pass folds same-type one-operand one-result
    `darwinn.bitcast`, `darwinn.convert` and `darwinn.copy_op` identities with replaceAllUsesWith and
    erase. It then runs checkDwcConvertibleTypes over the darwinn dialect ops and leaves every other op
    untouched.
  }];
}

def DwcReinterpretCastRankLegalizePassPass : Pass<"reinterpret-cast-rank-legalize-pass", "func::FuncOp"> {
  let summary = "DWC pass reinterpret-cast-rank-legalize-pass";
  let description = [{
    Seen at 0xd7f295 in task3_passes.tsv, this pass folds same-type darwinn.bitcast identities with
    replaceAllUsesWith and erase. It lowers static-shape bitcasts with equal element counts and
    bitwidths to dive_vm.bitcast with makeDwcLowerVmOp after checkDwcConvertibleTypes.
  }];
}

def DwcRenameDiveEntryFunctionPass : Pass<"rename-dive-entry-function", "func::FuncOp"> {
  let summary = "DWC pass rename-dive-entry-function";
  let description = [{
    Seen at 0xdbfc27 in task3_passes.tsv, this pass renames the function to dive_entry with SymbolTable
    setSymbolName after checkDwcConvertibleTypes. It returns early when the name is already dive_entry.
  }];
}

def DwcResamplerLoweringPass : Pass<"resampler-lowering", "func::FuncOp"> {
  let summary = "DWC pass resampler-lowering";
  let description = [{
    Seen at 0xdded92 in task3_passes.tsv, this pass folds same-type one-operand one-result
    `darwinn.resampler` and `darwinn.resampler_options` identities with replaceAllUsesWith and erase. It
    then requires every remaining op to sit in the darwinn dialect and pass checkDwcConvertibleTypes,
    failing other dialects with resampler-lowering accepts darwinn ops only.
  }];
}

def DwcRkhyShapeLegalizationPassPass : Pass<"rkhy-shape-legalization-pass", "func::FuncOp"> {
  let summary = "DWC pass rkhy-shape-legalization-pass";
  let description = [{
    Seen at 0xd7f029 in task3_passes.tsv, this pass lowers darwinn.rkhy_custom_padding and
    darwinn.mesh_pad_slice to dive_vm.pad with applyDwcLowerPadInline. It also lowers tgc elementwise
    ops to arith ops with applyDwcLowerScalarArith.
  }];
}

def DwcRkhyTypeLegalizationPassPass : Pass<"rkhy-type-legalization-pass", "func::FuncOp"> {
  let summary = "DWC pass rkhy-type-legalization-pass";
  let description = [{
    Seen at 0xd7f00d in task3_passes.tsv, this pass lowers darwinn.convert, darwinn.cast_in and
    darwinn.cast_out to dive_vm.cast with applyDwcLowerConvertTrunc. It keeps one-operand one-result ops
    with matching ranked shapes and numeric element types, folding same-type cases in place, and checks
    each with checkDwcConvertibleTypes.
  }];
}

def DwcRunR52OpsOnDivePass : Pass<"run-r52-ops-on-dive", "func::FuncOp"> {
  let summary = "DWC pass run-r52-ops-on-dive";
  let description = [{
    Seen at 0xde9621 in task3_passes.tsv, this pass folds same-type one-operand one-result `dive_vm`
    identities with replaceAllUsesWith and erase. It then requires every remaining op to sit in the
    dive_vm dialect and pass checkDwcConvertibleTypes, failing other dialects with run-r52-ops-on-dive
    accepts dive_vm ops only.
  }];
}

def DwcScalarCoreControlFlowLoweringPass : Pass<"scalar-core-control-flow-lowering", "func::FuncOp"> {
  let summary = "DWC pass scalar-core-control-flow-lowering";
  let description = [{
    Seen at 0xdded21 in task3_passes.tsv, this pass lowers darwinn.select to dive_vm.select with
    applyDwcLowerSelectInline. It also runs applyDwcLowerCopyLike for copy ops to dive_vm.copy and
    applyDwcLowerConvertTrunc for convert and cast ops to dive_vm.cast.
  }];
}

def DwcScalarCoreStdOpsLoweringPass : Pass<"scalar-core-std-ops-lowering", "func::FuncOp"> {
  let summary = "DWC pass scalar-core-std-ops-lowering";
  let description = [{
    Seen at 0xdded75 in task3_passes.tsv, this pass lowers darwinn.tgc_elementwise_add,
    darwinn.tgc_elementwise_mul and darwinn.tgc_elementwise_sub to arith addi, muli and subi or addf,
    mulf and subf by element type with applyDwcLowerScalarArith. It needs two operands and one result
    sharing one ranked shape, forwarded with forwardDwcLowerOp.
  }];
}

def DwcScalarOpsLegalizePass : Pass<"scalar-ops-legalize", "func::FuncOp"> {
  let summary = "DWC pass scalar-ops-legalize";
  let description = [{
    Seen at 0xde6d2b in task3_passes.tsv, this pass lowers darwinn.tgc_elementwise_add,
    darwinn.tgc_elementwise_mul and darwinn.tgc_elementwise_sub to arith addi, muli and subi or addf,
    mulf and subf by element type with applyDwcLowerScalarArith. It needs two operands and one result
    sharing one ranked shape, forwarded with forwardDwcLowerOp.
  }];
}

def DwcScatterGatherLoweringPass : Pass<"scatter-gather-lowering", "func::FuncOp"> {
  let summary = "DWC pass scatter-gather-lowering";
  let description = [{
    Seen at 0xddeda5 in task3_passes.tsv, this pass lowers darwinn.gather, darwinn.gather_copy and
    darwinn.hib_gather to dive_vm.gather with applyDwcLowerGatherOob, needing one result with two or
    more operands and int-or-index indices while stamping oob_zero_fill. It lowers darwinn.scatter to
    dive_vm.scatter_nd with applyDwcLowerScatterInline under the same attr.
  }];
}

def DwcSelectLoweringPass : Pass<"select-lowering", "func::FuncOp"> {
  let summary = "DWC pass select-lowering";
  let description = [{
    Seen at 0xdded65 in task3_passes.tsv, this pass lowers darwinn.select to dive_vm.select with
    applyDwcLowerSelectInline through forwardDwcLowerTo.
  }];
}

def DwcShardingUsingDivePass : Pass<"sharding-using-dive", "func::FuncOp"> {
  let summary = "DWC pass sharding-using-dive";
  let description = [{
    Seen at 0xde9635 in task3_passes.tsv, this pass folds same-type one-operand one-result
    `darwinn.broadcast_shard`, `darwinn.host_to_ssram_shard`, `darwinn.host_to_tile_shard`,
    `darwinn.narrow_to_narrow_shard`, `darwinn.narrow_to_wide_shard`, `darwinn.tensor_op_shard`,
    `darwinn.tile_to_host_shard` and `darwinn.tile_to_tile_shard` identities with replaceAllUsesWith and
    erase. It then requires every remaining op to sit in the darwinn dialect and pass
    checkDwcConvertibleTypes, failing other dialects with sharding-using-dive accepts darwinn ops only.
  }];
}

def DwcSkippingFoldOfFloatConvertPass : Pass<"skipping fold of float convert", "func::FuncOp"> {
  let summary = "DWC pass skipping-fold-of-float-convert";
  let description = [{
    Seen at 0xd68cda in task3_passes.tsv, this pass runs applyLocalConvertLowering, folding same-type
    darwinn.convert and darwinn.bitcast identities.
  }];
}

def DwcSplitOpLoweringPass : Pass<"split-op-lowering", "func::FuncOp"> {
  let summary = "DWC pass split-op-lowering";
  let description = [{
    Seen at 0xddede2 in task3_passes.tsv, this pass folds same-type one-operand one-result
    `darwinn.bitcast`, `darwinn.convert` and `darwinn.copy_op` identities with replaceAllUsesWith and
    erase. It then requires every remaining op to sit in the darwinn dialect and pass
    checkDwcConvertibleTypes, failing other dialects with split-op-lowering accepts darwinn ops only.
  }];
}

def DwcStablehloCompositeLegalizeTflCustomPass : Pass<"stablehlo-composite-legalize-tfl-custom", "func::FuncOp"> {
  let summary = "DWC pass stablehlo-composite-legalize-tfl-custom";
  let description = [{
    Seen at 0xdccdf7 in task3_passes.tsv, this pass folds same-type one-operand one-result stablehlo and
    tfl dialect ops identities with replaceAllUsesWith and erase. It then runs checkDwcConvertibleTypes
    over the stablehlo and tfl dialect ops and leaves every other op untouched.
  }];
}

def DwcStablehloCustomCallLegalizeCompositePass : Pass<"stablehlo-custom-call-legalize-composite", "func::FuncOp"> {
  let summary = "DWC pass stablehlo-custom-call-legalize-composite";
  let description = [{
    Seen at 0xdef5f5 in task3_passes.tsv, this pass folds same-type one-operand one-result stablehlo
    dialect ops identities with replaceAllUsesWith and erase. It then runs checkDwcConvertibleTypes over
    the stablehlo dialect ops and leaves every other op untouched.
  }];
}

def DwcStablehloLegalizeCompositeToCallPass : Pass<"stablehlo-legalize-composite-to-call", "func::FuncOp"> {
  let summary = "DWC pass stablehlo-legalize-composite-to-call";
  let description = [{
    Seen at 0xdd00c2 in task3_passes.tsv, this pass folds same-type one-operand one-result stablehlo
    dialect ops identities with replaceAllUsesWith and erase. It then runs checkDwcConvertibleTypes over
    the stablehlo dialect ops and leaves every other op untouched.
  }];
}

def DwcStablehloLegalizeToHloPass : Pass<"stablehlo-legalize-to-hlo", "func::FuncOp"> {
  let summary = "DWC pass stablehlo-legalize-to-hlo";
  let description = [{
    Seen at 0xdbc34d in task3_passes.tsv, this pass folds same-type one-operand one-result `mhlo`
    identities with replaceAllUsesWith and erase. It then runs checkDwcConvertibleTypes over the
    stablehlo and mhlo dialect ops and leaves every other op untouched.
  }];
}

def DwcStablehloLegalizeToVhloPass : Pass<"stablehlo-legalize-to-vhlo", "func::FuncOp"> {
  let summary = "DWC pass stablehlo-legalize-to-vhlo";
  let description = [{
    Seen at 0xdbc236 in task3_passes.tsv, this pass folds same-type one-operand one-result stablehlo and
    vhlo dialect ops identities with replaceAllUsesWith and erase. It then runs checkDwcConvertibleTypes
    over the stablehlo and vhlo dialect ops and leaves every other op untouched.
  }];
}

def DwcStablehloLegalizeVhloPass : Pass<"stablehlo-legalize-vhlo", "func::FuncOp"> {
  let summary = "DWC pass stablehlo-legalize-vhlo";
  let description = [{
    Seen at 0xdbc251 in task3_passes.tsv, this pass folds same-type one-operand one-result stablehlo and
    vhlo dialect ops identities with replaceAllUsesWith and erase. It then runs checkDwcConvertibleTypes
    over the stablehlo and vhlo dialect ops and leaves every other op untouched.
  }];
}

def DwcStochasticConvertPass : Pass<"stochastic-convert", "func::FuncOp"> {
  let summary = "DWC pass stochastic-convert";
  let description = [{
    Seen at 0xd68cc7 in task3_passes.tsv, this pass lowers darwinn.convert, darwinn.cast_in and
    darwinn.cast_out to dive_vm.cast with applyDwcLowerConvertTrunc. It keeps one-operand one-result ops
    with matching ranked shapes and numeric element types, folding same-type cases in place, and checks
    each with checkDwcConvertibleTypes.
  }];
}

def DwcTfLegalizeHloPass : Pass<"tf-legalize-hlo", "func::FuncOp"> {
  let summary = "DWC pass tf-legalize-hlo";
  let description = [{
    Seen at 0xdbc38d in task3_passes.tsv, this pass folds same-type one-operand one-result `mhlo`
    identities with replaceAllUsesWith and erase. It then runs checkDwcConvertibleTypes over the tf and
    mhlo dialect ops and leaves every other op untouched.
  }];
}

def DwcTflCustomLoweringRewritingPassPass : Pass<"tfl-custom-lowering-rewriting-pass", "func::FuncOp"> {
  let summary = "DWC pass tfl-custom-lowering-rewriting-pass";
  let description = [{
    Seen at 0xd7f120 in task3_passes.tsv, this pass folds same-type one-operand one-result tfl dialect
    ops identities with replaceAllUsesWith and erase. It then runs checkDwcConvertibleTypes over the tfl
    dialect ops and leaves every other op untouched.
  }];
}

def DwcTflLegalizeChloPass : Pass<"tfl-legalize-chlo", "func::FuncOp"> {
  let summary = "DWC pass tfl-legalize-chlo";
  let description = [{
    Seen at 0xdbc2fb in task3_passes.tsv, this pass folds same-type one-operand one-result tfl and chlo
    dialect ops identities with replaceAllUsesWith and erase. It then runs checkDwcConvertibleTypes over
    the tfl and chlo dialect ops and leaves every other op untouched.
  }];
}

def DwcTflLegalizeHashtablesTfPass : Pass<"tfl-legalize-hashtables-tf", "func::FuncOp"> {
  let summary = "DWC pass tfl-legalize-hashtables-tf";
  let description = [{
    Seen at 0xde2a7d in task3_passes.tsv, this pass folds same-type one-operand one-result tfl and tf
    dialect ops identities with replaceAllUsesWith and erase. It then runs checkDwcConvertibleTypes over
    the tfl and tf dialect ops and leaves every other op untouched.
  }];
}

def DwcTflLegalizeHloPass : Pass<"tfl-legalize-hlo", "func::FuncOp"> {
  let summary = "DWC pass tfl-legalize-hlo";
  let description = [{
    Seen at 0xdbc37c in task3_passes.tsv, this pass folds same-type one-operand one-result `mhlo`
    identities with replaceAllUsesWith and erase. It then runs checkDwcConvertibleTypes over the tfl and
    mhlo dialect ops and leaves every other op untouched.
  }];
}

def DwcTflLegalizeTensorlistPass : Pass<"tfl-legalize-tensorlist", "func::FuncOp"> {
  let summary = "DWC pass tfl-legalize-tensorlist";
  let description = [{
    Seen at 0xd66a8f in task3_passes.tsv, this pass folds same-type one-operand one-result tfl dialect
    ops identities with replaceAllUsesWith and erase. It then runs checkDwcConvertibleTypes over the tfl
    dialect ops and leaves every other op untouched.
  }];
}

def DwcTflLegalizeTfPass : Pass<"tfl-legalize-tf", "func::FuncOp"> {
  let summary = "DWC pass tfl-legalize-tf";
  let description = [{
    Seen at 0xde2ae8 in task3_passes.tsv, this pass folds same-type one-operand one-result tfl and tf
    dialect ops identities with replaceAllUsesWith and erase. It then runs checkDwcConvertibleTypes over
    the tfl and tf dialect ops and leaves every other op untouched.
  }];
}

def DwcTflLegalizeTfWhilePass : Pass<"tfl-legalize-tf-while", "func::FuncOp"> {
  let summary = "DWC pass tfl-legalize-tf-while";
  let description = [{
    Seen at 0xe07f5e in task3_passes.tsv, this pass folds same-type one-operand one-result tfl and tf
    dialect ops identities with replaceAllUsesWith and erase. It then runs checkDwcConvertibleTypes over
    the tfl and tf dialect ops and leaves every other op untouched.
  }];
}

def DwcTflLegalizeVariablesTfPass : Pass<"tfl-legalize-variables-tf", "func::FuncOp"> {
  let summary = "DWC pass tfl-legalize-variables-tf";
  let description = [{
    Seen at 0xde2a98 in task3_passes.tsv, this pass folds same-type one-operand one-result tfl and tf
    dialect ops identities with replaceAllUsesWith and erase. It then runs checkDwcConvertibleTypes over
    the tfl and tf dialect ops and leaves every other op untouched.
  }];
}

def DwcTflLowerQuantAnnotationsPass : Pass<"tfl-lower-quant-annotations", "func::FuncOp"> {
  let summary = "DWC pass tfl-lower-quant-annotations";
  let description = [{
    Seen at 0xd87c99 in task3_passes.tsv, this pass applies populateLowerCopySlicePatterns and
    populateLowerConvertPatterns greedily, then runs applyLocalCopySliceLowering and
    applyLocalConvertLowering. The locals fold same-type darwinn.copy_op identities and same-type
    darwinn.convert and darwinn.bitcast identities.
  }];
}

def DwcTflLowerStaticTensorListPass : Pass<"tfl-lower-static-tensor-list", "func::FuncOp"> {
  let summary = "DWC pass tfl-lower-static-tensor-list";
  let description = [{
    Seen at 0xd66ca5 in task3_passes.tsv, this pass applies populateLowerCopySlicePatterns and
    populateLowerConvertPatterns greedily, then runs applyLocalCopySliceLowering and
    applyLocalConvertLowering. The locals fold same-type darwinn.copy_op identities and same-type
    darwinn.convert and darwinn.bitcast identities.
  }];
}

def DwcTopKLoweringPolicyPass : Pass<"top-k-lowering-policy", "func::FuncOp"> {
  let summary = "DWC pass top-k-lowering-policy";
  let description = [{
    Seen at 0xd5dd44 in task3_passes.tsv, this pass lowers darwinn.index_filter to dive_vm.top_k with
    applyDwcLowerTopKInline through forwardDwcLowerTo.
  }];
}

def DwcTpuClusteringAlgorithmPass : Pass<"tpu-clustering-algorithm", "func::FuncOp"> {
  let summary = "DWC pass tpu-clustering-algorithm";
  let description = [{
    Seen at 0xdcdd6e in task3_passes.tsv, this pass walks ops in order and joins each op to the cluster
    of its operand defining op, stamping tpu.cluster_id per op and tpu.cluster_count on the function.
  }];
}

def DwcVhloLegalizeStablehloPass : Pass<"vhlo-legalize-stablehlo", "func::FuncOp"> {
  let summary = "DWC pass vhlo-legalize-stablehlo";
  let description = [{
    Seen at 0xdbc299 in task3_passes.tsv, this pass folds same-type one-operand one-result vhlo and
    stablehlo dialect ops identities with replaceAllUsesWith and erase. It then runs
    checkDwcConvertibleTypes over the vhlo and stablehlo dialect ops and leaves every other op
    untouched.
  }];
}

def DwcVhloLegalizeToStablehloPass : Pass<"vhlo-legalize-to-stablehlo", "func::FuncOp"> {
  let summary = "DWC pass vhlo-legalize-to-stablehlo";
  let description = [{
    Seen at 0xdbc271 in task3_passes.tsv, this pass folds same-type one-operand one-result vhlo and
    stablehlo dialect ops identities with replaceAllUsesWith and erase. It then runs
    checkDwcConvertibleTypes over the vhlo and stablehlo dialect ops and leaves every other op
    untouched.
  }];
}

def DwcWrapUpDiveProgramPass : Pass<"wrap-up-dive-program", "func::FuncOp"> {
  let summary = "DWC pass wrap-up-dive-program";
  let description = [{
    Seen at 0xdce5cc in task3_passes.tsv, this pass strips the tpu.cluster_id, tpu.packet_id and
    tpu.packet_order attrs from every op.
  }];
}

def DwcCopyReferencePass : Pass<"copy-reference", "func::FuncOp"> {
  let summary = "DWC pass copy-reference";
  let description = [{
    Blob-evidenced lisa CopyReferencePass. Materializes copy reference edges
    between producer and consumer ops for downstream slicing and allocation.
  }];
}

def DwcCanonicalizationPass : Pass<"canonicalization", "func::FuncOp"> {
  let summary = "DWC pass canonicalization";
  let description = [{
    Blob-evidenced lisa CanonicalizationPass. Runs canonicalization patterns
    over darwinn ops inside the function.
  }];
}

def DwcCoallocationPass : Pass<"coallocation", "func::FuncOp"> {
  let summary = "DWC pass coallocation";
  let description = [{
    Blob-evidenced lisa CoallocationPass. Assigns coallocation groups to
    sibling buffers shared across producer consumer edges.
  }];
}

def DwcCountingVariableCleanupPass : Pass<"counting-variable-cleanup", "func::FuncOp"> {
  let summary = "DWC pass counting-variable-cleanup";
  let description = [{
    Blob-evidenced lisa CountingVariableCleanupPass. Removes dead counting
    variable ops left over after slicing and unrolling.
  }];
}

def DwcCompositeSlicingSolverPass : Pass<"composite-slicing-solver", "func::FuncOp"> {
  let summary = "DWC pass composite-slicing-solver";
  let description = [{
    Blob-evidenced lisa pass. Composite slicing solver. Solves composite op slicing assignments across producer consumer edges.
  }];
}

def DwcCustomSlicingAssignmentPass : Pass<"custom-slicing-assignment", "func::FuncOp"> {
  let summary = "DWC pass custom-slicing-assignment";
  let description = [{
    Blob-evidenced lisa pass. Custom slicing assignment. Assigns custom slicing strategies to ops annotated with tiling options.
  }];
}

def DwcDefaultUnitSlicingPass : Pass<"default-unit-slicing", "func::FuncOp"> {
  let summary = "DWC pass default-unit-slicing";
  let description = [{
    Blob-evidenced lisa pass. Default unit slicing. Applies unit slicing to ops left unassigned by earlier slicing passes.
  }];
}

def DwcComputeMapCwiseMinMaxPass : Pass<"compute-map-cwise-min-max", "func::FuncOp"> {
  let summary = "DWC pass compute-map-cwise-min-max";
  let description = [{
    Blob-evidenced lisa pass. Compute map cwise min max. Folds cwise min max bounds into compute map metadata.
  }];
}

def DwcComputeMapOptimizePass : Pass<"compute-map-optimize", "func::FuncOp"> {
  let summary = "DWC pass compute-map-optimize";
  let description = [{
    Blob-evidenced lisa pass. Compute map optimize. Optimizes compute map layouts after min max folding.
  }];
}

def DwcXlaCpuUseNewXtileLoweringPass : Pass<"xla_cpu_use_new_xtile_lowering", "func::FuncOp"> {
  let summary = "XLA CPU new-xtile lowering";
  let description = [{
    Seen at 0xdded02 in task3_passes.tsv, this pass applies populateLowerCopySlicePatterns greedily and
    runs applyLocalCopySliceLowering, folding same-type darwinn.copy_op identities.
  }];
}

#endif // MLIR_LIB_DIALECT_DARWINN_TRANSFORMS_DWCPASSES
