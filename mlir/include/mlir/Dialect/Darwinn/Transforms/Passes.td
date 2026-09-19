//===-- DwcPasses.td - Darwinn DWC pass declarations ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the Darwinn DWC lowering passes recovered from
// task3_passes.tsv (kind=pass rows of blob string inventory). Each pass is
// boilerplate only. Bodies live in DwcPasses.cpp as empty runOnOperation
// stubs keyed to the same TSV rows.
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
    Boilerplate for the `(ackr-model-converter` pass name seen at 0xdac24f in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcAddBoundLowerPass : Pass<"add_bound_lower", "func::FuncOp"> {
  let summary = "DWC pass add_bound_lower";
  let description = [{
    Boilerplate for the `add_bound_lower` pass name seen at 0xdaad7f in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcAddDiveAbiArgumentsPass : Pass<"add-dive-abi-arguments", "func::FuncOp"> {
  let summary = "DWC pass add-dive-abi-arguments";
  let description = [{
    Boilerplate for the `add-dive-abi-arguments` pass name seen at 0xd7a252 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcAddDiveTracingPass : Pass<"add-dive-tracing", "func::FuncOp"> {
  let summary = "DWC pass add-dive-tracing";
  let description = [{
    Boilerplate for the `add-dive-tracing` pass name seen at 0xde194d in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcAllowBf16AndF16TypeLegalizationPass : Pass<"allow-bf16-and-f16-type-legalization", "func::FuncOp"> {
  let summary = "DWC pass allow-bf16-and-f16-type-legalization";
  let description = [{
    Boilerplate for the `allow-bf16-and-f16-type-legalization` pass name seen at 0xdc1b37 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcArithAssertLowerPass : Pass<"arith-assert-lower", "func::FuncOp"> {
  let summary = "DWC pass arith-assert-lower";
  let description = [{
    Boilerplate for the `arith assert lower` pass name seen at 0xdaad9b in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcArithLowerPass : Pass<"arith-lower", "func::FuncOp"> {
  let summary = "DWC pass arith-lower";
  let description = [{
    Boilerplate for the `arith-lower` pass name seen at 0xdaad8f in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcBitcastConvertPass : Pass<"bitcast-convert", "func::FuncOp"> {
  let summary = "DWC pass bitcast-convert";
  let description = [{
    Boilerplate for the `bitcast-convert` pass name seen at 0xd68cb7 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcChloLegalizeToHloPass : Pass<"chlo-legalize-to-hlo", "func::FuncOp"> {
  let summary = "DWC pass chlo-legalize-to-hlo";
  let description = [{
    Boilerplate for the `chlo-legalize-to-hlo` pass name seen at 0xdbc367 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcCompositeLoweringPass : Pass<"composite-lowering", "func::FuncOp"> {
  let summary = "DWC pass composite-lowering";
  let description = [{
    Boilerplate for the `composite-lowering` pass name seen at 0xddee24 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcConcatModelConverterPass : Pass<"concat-model-converter", "func::FuncOp"> {
  let summary = "DWC pass concat-model-converter";
  let description = [{
    Boilerplate for the `concat-model-converter` pass name seen at 0xdac238 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcConcatProofConverterPass : Pass<"concat-proof-converter", "func::FuncOp"> {
  let summary = "DWC pass concat-proof-converter";
  let description = [{
    Boilerplate for the `concat-proof-converter` pass name seen at 0xdac279 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcConvertArithToLlvmPass : Pass<"dwc-convert-arith-to-llvm", "func::FuncOp"> {
  let summary = "DWC pass convert-arith-to-llvm";
  let description = [{
    Boilerplate for the `convert-arith-to-llvm` pass name seen at 0xdcb7f5 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcConvertCfToLlvmPass : Pass<"dwc-convert-cf-to-llvm", "func::FuncOp"> {
  let summary = "DWC pass convert-cf-to-llvm";
  let description = [{
    Boilerplate for the `convert-cf-to-llvm` pass name seen at 0xdcb838 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcConvertConv1x1ToFcPass : Pass<"convert-conv1x1-to-fc", "func::FuncOp"> {
  let summary = "DWC pass convert-conv1x1-to-fc";
  let description = [{
    Boilerplate for the `convert-conv1x1-to-fc` pass name seen at 0xe26ea5 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcConvertDiveVmTensorToLinalgPass : Pass<"convert-dive-vm-tensor-to-linalg", "func::FuncOp"> {
  let summary = "DWC pass convert-dive-vm-tensor-to-linalg";
  let description = [{
    Boilerplate for the `convert-dive-vm-tensor-to-linalg` pass name seen at 0xde1ad1 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcConvertDiveVmTensorToScfPass : Pass<"convert-dive-vm-tensor-to-scf", "func::FuncOp"> {
  let summary = "DWC pass convert-dive-vm-tensor-to-scf";
  let description = [{
    Boilerplate for the `convert-dive-vm-tensor-to-scf` pass name seen at 0xde4651 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcConvertDiveVmTensorToTensorPass : Pass<"convert-dive-vm-tensor-to-tensor", "func::FuncOp"> {
  let summary = "DWC pass convert-dive-vm-tensor-to-tensor";
  let description = [{
    Boilerplate for the `convert-dive-vm-tensor-to-tensor` pass name seen at 0xda89fa in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcConvertDiveVmToLlvmPass : Pass<"convert-dive-vm-to-llvm", "func::FuncOp"> {
  let summary = "DWC pass convert-dive-vm-to-llvm";
  let description = [{
    Boilerplate for the `convert-dive-vm-to-llvm` pass name seen at 0xdcb7dd in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcConvertDiveVmToMemrefPass : Pass<"convert-dive-vm-to-memref", "func::FuncOp"> {
  let summary = "DWC pass convert-dive-vm-to-memref";
  let description = [{
    Boilerplate for the `convert-dive-vm-to-memref` pass name seen at 0xde3efd in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcConvertDwcToDiveVmTensorPass : Pass<"convert-dwc-to-dive-vm-tensor", "func::FuncOp"> {
  let summary = "DWC pass convert-dwc-to-dive-vm-tensor";
  let description = [{
    Boilerplate for the `convert-dwc-to-dive-vm-tensor` pass name seen at 0xda8a3d in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcConvertDwgToDiveVmPass : Pass<"convert-dwg-to-dive-vm", "func::FuncOp"> {
  let summary = "DWC pass convert-dwg-to-dive-vm";
  let description = [{
    Boilerplate for the `convert-dwg-to-dive-vm` pass name seen at 0xdcb89c in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcConvertDynamicShapeScopeToDiveVmPass : Pass<"convert-dynamic-shape-scope-to-dive-vm", "func::FuncOp"> {
  let summary = "DWC pass convert-dynamic-shape-scope-to-dive-vm";
  let description = [{
    Boilerplate for the `convert-dynamic-shape-scope-to-dive-vm` pass name seen at 0xdcb8b3 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcConvertFuncToLlvmPass : Pass<"dwc-convert-func-to-llvm", "func::FuncOp"> {
  let summary = "DWC pass convert-func-to-llvm";
  let description = [{
    Boilerplate for the `convert-func-to-llvm` pass name seen at 0xdcb867 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcConvertGenericNormToPseudoOpPass : Pass<"convert-generic-norm-to-pseudo-op", "func::FuncOp"> {
  let summary = "DWC pass convert-generic-norm-to-pseudo-op";
  let description = [{
    Boilerplate for the `convert-generic-norm-to-pseudo-op` pass name seen at 0xdb45be in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcConvertLinalgToLoopsPass : Pass<"dwc-convert-linalg-to-loops", "func::FuncOp"> {
  let summary = "DWC pass convert-linalg-to-loops";
  let description = [{
    Boilerplate for the `convert-linalg-to-loops` pass name seen at 0xd83df7 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcConvertMathToLibmPass : Pass<"dwc-convert-math-to-libm", "func::FuncOp"> {
  let summary = "DWC pass convert-math-to-libm";
  let description = [{
    Boilerplate for the `convert-math-to-libm` pass name seen at 0xdce410 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcConvertMathToLlvmPass : Pass<"dwc-convert-math-to-llvm", "func::FuncOp"> {
  let summary = "DWC pass convert-math-to-llvm";
  let description = [{
    Boilerplate for the `convert-math-to-llvm` pass name seen at 0xdcb80b in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcConvertOpLoweringPass : Pass<"convert-op-lowering", "func::FuncOp"> {
  let summary = "DWC pass convert-op-lowering";
  let description = [{
    Boilerplate for the `convert-op-lowering` pass name seen at 0xddedce in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcConvertPdlToPdlInterpPass : Pass<"dwc-convert-pdl-to-pdl-interp", "func::FuncOp"> {
  let summary = "DWC pass convert-pdl-to-pdl-interp";
  let description = [{
    Boilerplate for the `convert-pdl-to-pdl-interp` pass name seen at 0xdb3800 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcConvertScatterToGenericScatterPass : Pass<"convert-scatter-to-generic-scatter", "func::FuncOp"> {
  let summary = "DWC pass convert-scatter-to-generic-scatter";
  let description = [{
    Boilerplate for the `convert-scatter-to-generic-scatter` pass name seen at 0xdabcd2 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcConvertScfToCfPass : Pass<"dwc-convert-scf-to-cf", "func::FuncOp"> {
  let summary = "DWC pass convert-scf-to-cf";
  let description = [{
    Boilerplate for the `convert-scf-to-cf` pass name seen at 0xde4728 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcConvertSignedIntWithRescalingOpsPass : Pass<"convert-signed-int-with-rescaling-ops", "func::FuncOp"> {
  let summary = "DWC pass convert-signed-int-with-rescaling-ops";
  let description = [{
    Boilerplate for the `convert-signed-int-with-rescaling-ops` pass name seen at 0xd84723 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcConvertSpatialReductionToPoolingPass : Pass<"convert-spatial-reduction-to-pooling", "func::FuncOp"> {
  let summary = "DWC pass convert-spatial-reduction-to-pooling";
  let description = [{
    Boilerplate for the `convert-spatial-reduction-to-pooling` pass name seen at 0xddfd8d in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcConvertTfToDwcPass : Pass<"convert-tf-to-dwc", "func::FuncOp"> {
  let summary = "DWC pass convert-tf-to-dwc";
  let description = [{
    Boilerplate for the `convert-tf-to-dwc` pass name seen at 0xe2523e in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcConvertToKInMSparsityPass : Pass<"convert-to-k-in-m-sparsity", "func::FuncOp"> {
  let summary = "DWC pass convert-to-k-in-m-sparsity";
  let description = [{
    Boilerplate for the `convert-to-k-in-m-sparsity` pass name seen at 0xd5a046 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcConvertTpuOffloadToDiveVmPass : Pass<"convert-tpu-offload-to-dive-vm", "func::FuncOp"> {
  let summary = "DWC pass convert-tpu-offload-to-dive-vm";
  let description = [{
    Boilerplate for the `convert-tpu-offload-to-dive-vm` pass name seen at 0xdcb8da in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcConvertTpuOffloadToLlvmPass : Pass<"convert-tpu-offload-to-llvm", "func::FuncOp"> {
  let summary = "DWC pass convert-tpu-offload-to-llvm";
  let description = [{
    Boilerplate for the `convert-tpu-offload-to-llvm` pass name seen at 0xdcb84b in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcConvertXlaSupportedStablehloPass : Pass<"convert-xla-supported-stablehlo", "func::FuncOp"> {
  let summary = "DWC pass convert-xla-supported-stablehlo";
  let description = [{
    Boilerplate for the `convert-xla-supported-stablehlo` pass name seen at 0xdbc2b1 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcConvertDiveVmTensorToLinalgSymbolPass : Pass<"ConvertDiveVmTensorToLinalg", "func::FuncOp"> {
  let summary = "DWC pass ConvertDiveVmTensorToLinalg";
  let description = [{
    Boilerplate for the `ConvertDiveVmTensorToLinalg` pass name seen at 0xde1af2 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcConvertTpuOffloadToLlvmSymbolPass : Pass<"ConvertTpuOffloadToLlvm", "func::FuncOp"> {
  let summary = "DWC pass ConvertTpuOffloadToLlvm";
  let description = [{
    Boilerplate for the `ConvertTpuOffloadToLlvm` pass name seen at 0xdcb87c in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcCopyOpLoweringPass : Pass<"copy-op-lowering", "func::FuncOp"> {
  let summary = "DWC pass copy-op-lowering";
  let description = [{
    Boilerplate for the `copy-op-lowering` pass name seen at 0xddedbd in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcDarwinnBundlingPass : Pass<"darwinn-bundling", "func::FuncOp"> {
  let summary = "DWC pass darwinn-bundling";
  let description = [{
    Boilerplate for the `darwinn-bundling` pass name seen at 0xde00af in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcDarwinnConvertPass : Pass<"darwinn-convert", "func::FuncOp"> {
  let summary = "DWC pass darwinn-convert";
  let description = [{
    Boilerplate for the `darwinn.convert` pass name seen at 0xd68ca7 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcDarwinnMathJoinPass : Pass<"darwinn-math-join", "func::FuncOp"> {
  let summary = "DWC pass darwinn-math-join";
  let description = [{
    Boilerplate for the `darwinn.math.join` pass name seen at 0xdc9452 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcDarwinnSparsityPass : Pass<"darwinn-sparsity", "func::FuncOp"> {
  let summary = "DWC pass darwinn-sparsity";
  let description = [{
    Boilerplate for the `darwinn.sparsity` pass name seen at 0xd5a035 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcDiveDcePass : Pass<"dive-dce", "func::FuncOp"> {
  let summary = "DWC pass dive-dce";
  let description = [{
    Boilerplate for the `dive-dce` pass name seen at 0xe0fe97 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcDiveIoOptimizationPass : Pass<"dive-io-optimization", "func::FuncOp"> {
  let summary = "DWC pass dive-io-optimization";
  let description = [{
    Boilerplate for the `dive-io-optimization` pass name seen at 0xdc141b in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcDiveProgramTpuPass : Pass<"dive-program-tpu", "func::FuncOp"> {
  let summary = "DWC pass dive-program-tpu";
  let description = [{
    Boilerplate for the `dive-program-tpu` pass name seen at 0xd6383a in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcDiveUnrollFactorPass : Pass<"dive-unroll-factor", "func::FuncOp"> {
  let summary = "DWC pass dive-unroll-factor";
  let description = [{
    Boilerplate for the `dive-unroll-factor` pass name seen at 0xda744d in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcDiveVmBufferizePass : Pass<"dive-vm-bufferize", "func::FuncOp"> {
  let summary = "DWC pass dive-vm-bufferize";
  let description = [{
    Boilerplate for the `dive-vm-bufferize` pass name seen at 0xde6a70 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcDiveVmOutlineShareableDiveConstsPass : Pass<"dive-vm-outline-shareable-dive-consts", "func::FuncOp"> {
  let summary = "DWC pass dive-vm-outline-shareable-dive-consts";
  let description = [{
    Boilerplate for the `dive-vm-outline-shareable-dive-consts` pass name seen at 0xd788a7 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcDwcCheckIllegalTpuOpsPass : Pass<"dwc-check-illegal-tpu-ops", "func::FuncOp"> {
  let summary = "DWC pass dwc-check-illegal-tpu-ops";
  let description = [{
    Boilerplate for the `dwc-check-illegal-tpu-ops` pass name seen at 0xd84494 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcDwcConvertInputOutputTypesPass : Pass<"dwc-convert-input-output-types", "func::FuncOp"> {
  let summary = "DWC pass dwc-convert-input-output-types";
  let description = [{
    Boilerplate for the `dwc-convert-input-output-types` pass name seen at 0xd94dcc in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcDwcCopyStridedBuffersOnTpuPass : Pass<"dwc-copy-strided-buffers-on-tpu", "func::FuncOp"> {
  let summary = "DWC pass dwc-copy-strided-buffers-on-tpu";
  let description = [{
    Boilerplate for the `dwc-copy-strided-buffers-on-tpu` pass name seen at 0xd6381a in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcDwcFormTpuClustersPass : Pass<"dwc-form-tpu-clusters", "func::FuncOp"> {
  let summary = "DWC pass dwc-form-tpu-clusters";
  let description = [{
    Boilerplate for the `dwc-form-tpu-clusters` pass name seen at 0xd81a1f in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcDwcLegalizePass : Pass<"dwc-legalize", "func::FuncOp"> {
  let summary = "DWC pass dwc-legalize";
  let description = [{
    Boilerplate for the `dwc-legalize` pass name seen at 0xde6d3f in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcDwcLegalizeHloPass : Pass<"dwc-legalize-hlo", "func::FuncOp"> {
  let summary = "DWC pass dwc-legalize-hlo";
  let description = [{
    Boilerplate for the `dwc-legalize-hlo` pass name seen at 0xdbc39d in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcDwcLegalizeHloToTfPass : Pass<"dwc-legalize-hlo-to-tf", "func::FuncOp"> {
  let summary = "DWC pass dwc-legalize-hlo-to-tf";
  let description = [{
    Boilerplate for the `dwc-legalize-hlo-to-tf` pass name seen at 0xde2ad1 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcDwcLegalizeIntAndQuantTypesPass : Pass<"dwc-legalize-int-and-quant-types", "func::FuncOp"> {
  let summary = "DWC pass dwc-legalize-int-and-quant-types";
  let description = [{
    Boilerplate for the `dwc-legalize-int-and-quant-types` pass name seen at 0xd94e00 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcDwcLegalizeInt64ConstantsPass : Pass<"dwc-legalize-int64-constants", "func::FuncOp"> {
  let summary = "DWC pass dwc-legalize-int64-constants";
  let description = [{
    Boilerplate for the `dwc-legalize-int64-constants` pass name seen at 0xd7bb5d in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcDwcLegalizePassSymbol : Pass<"dwc-legalize-pass", "func::FuncOp"> {
  let summary = "DWC pass dwc-legalize-pass";
  let description = [{
    Boilerplate for the `dwc-legalize-pass` pass name seen at 0xd7f2b9 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcDwcLegalizeStablehloAnnotateMaterializePolicyPass : Pass<"dwc-legalize-stablehlo-annotate-materialize-policy", "func::FuncOp"> {
  let summary = "DWC pass dwc-legalize-stablehlo-annotate-materialize-policy";
  let description = [{
    Boilerplate for the `dwc-legalize-stablehlo-annotate-materialize-policy` pass name seen at 0xd5dd77 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcDwcLegalizeStablehloCompositePass : Pass<"dwc-legalize-stablehlo-composite", "func::FuncOp"> {
  let summary = "DWC pass dwc-legalize-stablehlo-composite";
  let description = [{
    Boilerplate for the `dwc-legalize-stablehlo-composite` pass name seen at 0xdef599 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcDwcLegalizeTfPipelinePass : Pass<"dwc-legalize-tf-pipeline", "func::FuncOp"> {
  let summary = "DWC pass dwc-legalize-tf-pipeline";
  let description = [{
    Boilerplate for the `dwc-legalize-tf-pipeline` pass name seen at 0xe03284 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcDwcLegalizeTflCudaemuCustomOpsPass : Pass<"dwc-legalize-tfl-cudaemu-custom-ops", "func::FuncOp"> {
  let summary = "DWC pass dwc-legalize-tfl-cudaemu-custom-ops";
  let description = [{
    Boilerplate for the `dwc-legalize-tfl-cudaemu-custom-ops` pass name seen at 0xd8464e in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcDwcLegalizeTflMultinomialPass : Pass<"dwc-legalize-tfl-multinomial", "func::FuncOp"> {
  let summary = "DWC pass dwc-legalize-tfl-multinomial";
  let description = [{
    Boilerplate for the `dwc-legalize-tfl-multinomial` pass name seen at 0xdd3cdb in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcDwcLegalizeTflVariableTensorsPass : Pass<"dwc-legalize-tfl-variable-tensors", "func::FuncOp"> {
  let summary = "DWC pass dwc-legalize-tfl-variable-tensors";
  let description = [{
    Boilerplate for the `dwc-legalize-tfl-variable-tensors` pass name seen at 0xd81097 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcDwcLegalizeUint32TypesPass : Pass<"dwc-legalize-uint32-types", "func::FuncOp"> {
  let summary = "DWC pass dwc-legalize-uint32-types";
  let description = [{
    Boilerplate for the `dwc-legalize-uint32-types` pass name seen at 0xd94e3e in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcDwcLowerArgmaxIndexUnpoolPass : Pass<"dwc-lower-argmax-index-unpool", "func::FuncOp"> {
  let summary = "DWC pass dwc-lower-argmax-index-unpool";
  let description = [{
    Boilerplate for the `dwc-lower-argmax-index-unpool` pass name seen at 0xdcf030 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcDwcLowerCompositeOpsPass : Pass<"dwc-lower-composite-ops", "func::FuncOp"> {
  let summary = "DWC pass dwc-lower-composite-ops";
  let description = [{
    Boilerplate for the `dwc-lower-composite-ops` pass name seen at 0xd84775 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcDwcLowerControlFlowPass : Pass<"dwc-lower-control-flow", "func::FuncOp"> {
  let summary = "DWC pass dwc-lower-control-flow";
  let description = [{
    Boilerplate for the `dwc-lower-control-flow` pass name seen at 0xd61f2c in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcDwcLowerDepthToFromSpacePass : Pass<"dwc-lower-depth-to-from-space", "func::FuncOp"> {
  let summary = "DWC pass dwc-lower-depth-to-from-space";
  let description = [{
    Boilerplate for the `dwc-lower-depth-to-from-space` pass name seen at 0xe102d0 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcDwcLowerGenericConstantsPass : Pass<"dwc-lower-generic-constants", "func::FuncOp"> {
  let summary = "DWC pass dwc-lower-generic-constants";
  let description = [{
    Boilerplate for the `dwc-lower-generic-constants` pass name seen at 0xd7bb41 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcDwcLowerHlopsPass : Pass<"dwc-lower-hlops", "func::FuncOp"> {
  let summary = "DWC pass dwc-lower-hlops";
  let description = [{
    Boilerplate for the `dwc-lower-hlops` pass name seen at 0xd841f4 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcDwcLowerInputOutputCastPass : Pass<"dwc-lower-input-output-cast", "func::FuncOp"> {
  let summary = "DWC pass dwc-lower-input-output-cast";
  let description = [{
    Boilerplate for the `dwc-lower-input-output-cast` pass name seen at 0xd68084 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcDwcLowerPaddingOpsPass : Pass<"dwc-lower-padding-ops", "func::FuncOp"> {
  let summary = "DWC pass dwc-lower-padding-ops";
  let description = [{
    Boilerplate for the `dwc-lower-padding-ops` pass name seen at 0xd84749 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcDwcLowerPseudoOpsPass : Pass<"dwc-lower-pseudo-ops", "func::FuncOp"> {
  let summary = "DWC pass dwc-lower-pseudo-ops";
  let description = [{
    Boilerplate for the `dwc-lower-pseudo-ops` pass name seen at 0xd845c5 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcDwcLowerResamplerOpsPass : Pass<"dwc-lower-resampler-ops", "func::FuncOp"> {
  let summary = "DWC pass dwc-lower-resampler-ops";
  let description = [{
    Boilerplate for the `dwc-lower-resampler-ops` pass name seen at 0xd84598 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcDwcLowerScalarOpsPass : Pass<"dwc-lower-scalar-ops", "func::FuncOp"> {
  let summary = "DWC pass dwc-lower-scalar-ops";
  let description = [{
    Boilerplate for the `dwc-lower-scalar-ops` pass name seen at 0xd845b0 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcDwcLowerScatterOpsPass : Pass<"dwc-lower-scatter-ops", "func::FuncOp"> {
  let summary = "DWC pass dwc-lower-scatter-ops";
  let description = [{
    Boilerplate for the `dwc-lower-scatter-ops` pass name seen at 0xd84582 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcDwcLowerTopKPass : Pass<"dwc-lower-top-k", "func::FuncOp"> {
  let summary = "DWC pass dwc-lower-top-k";
  let description = [{
    Boilerplate for the `dwc-lower-top-k` pass name seen at 0xdd6da3 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcDwcPostTruncationTpuFitterPass : Pass<"dwc-post-truncation-tpu-fitter", "func::FuncOp"> {
  let summary = "DWC pass dwc-post-truncation-tpu-fitter";
  let description = [{
    Boilerplate for the `dwc-post-truncation-tpu-fitter` pass name seen at 0xdab7fc in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcDwcPreTpuFitterOptimizeGatherPass : Pass<"dwc-pre-tpu-fitter-optimize-gather", "func::FuncOp"> {
  let summary = "DWC pass dwc-pre-tpu-fitter-optimize-gather";
  let description = [{
    Boilerplate for the `dwc-pre-tpu-fitter-optimize-gather` pass name seen at 0xdaf5f9 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcDwcPreTpuFitterOptimizeScatterPass : Pass<"dwc-pre-tpu-fitter-optimize-scatter", "func::FuncOp"> {
  let summary = "DWC pass dwc-pre-tpu-fitter-optimize-scatter";
  let description = [{
    Boilerplate for the `dwc-pre-tpu-fitter-optimize-scatter` pass name seen at 0xdabc37 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcDwcRegroupTpuFunctionsPass : Pass<"dwc-regroup-tpu-functions", "func::FuncOp"> {
  let summary = "DWC pass dwc-regroup-tpu-functions";
  let description = [{
    Boilerplate for the `dwc-regroup-tpu-functions` pass name seen at 0xd873c3 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcDwcSerializeTpuOffloadsPass : Pass<"dwc-serialize-tpu-offloads", "func::FuncOp"> {
  let summary = "DWC pass dwc-serialize-tpu-offloads";
  let description = [{
    Boilerplate for the `dwc-serialize-tpu-offloads` pass name seen at 0xd9c33f in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcDwcTestRepeatTpuOpsPass : Pass<"dwc-test-repeat-tpu-ops", "func::FuncOp"> {
  let summary = "DWC pass dwc-test-repeat-tpu-ops";
  let description = [{
    Boilerplate for the `dwc-test-repeat-tpu-ops` pass name seen at 0xd8447c in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcDwcTpuFitterPass : Pass<"dwc-tpu-fitter", "func::FuncOp"> {
  let summary = "DWC pass dwc-tpu-fitter";
  let description = [{
    Boilerplate for the `dwc-tpu-fitter` pass name seen at 0xdab81b in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcDwcTpuFunctionCsePass : Pass<"dwc-tpu-function-cse", "func::FuncOp"> {
  let summary = "DWC pass dwc-tpu-function-cse";
  let description = [{
    Boilerplate for the `dwc-tpu-function-cse` pass name seen at 0xdf32f5 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcDwgCreateDarwinnCustomOpPass : Pass<"dwg-create-darwinn-custom-op", "func::FuncOp"> {
  let summary = "DWC pass dwg-create-darwinn-custom-op";
  let description = [{
    Boilerplate for the `dwg-create-darwinn-custom-op` pass name seen at 0xdb45e0 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcDwgForkMulticoreTpuOffloadsPass : Pass<"dwg-fork-multicore-tpu-offloads", "func::FuncOp"> {
  let summary = "DWC pass dwg-fork-multicore-tpu-offloads";
  let description = [{
    Boilerplate for the `dwg-fork-multicore-tpu-offloads` pass name seen at 0xd9c35a in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcDwgLowerForToWhilePass : Pass<"dwg-lower-for-to-while", "func::FuncOp"> {
  let summary = "DWC pass dwg-lower-for-to-while";
  let description = [{
    Boilerplate for the `dwg-lower-for-to-while` pass name seen at 0xe07f47 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcDwgtLowerIndexTypePass : Pass<"dwgt-lower-index-type", "func::FuncOp"> {
  let summary = "DWC pass dwgt-lower-index-type";
  let description = [{
    Boilerplate for the `dwgt-lower-index-type` pass name seen at 0xdf6914 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcDynamicUpdateSliceLoweringPass : Pass<"dynamic-update-slice-lowering", "func::FuncOp"> {
  let summary = "DWC pass dynamic-update-slice-lowering";
  let description = [{
    Boilerplate for the `dynamic-update-slice-lowering` pass name seen at 0xddee37 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcEdgetpuCustomOp2Pass : Pass<"edgetpu-custom-op-2", "func::FuncOp"> {
  let summary = "DWC pass edgetpu-custom-op-2";
  let description = [{
    Boilerplate for the `edgetpu-custom-op-2` pass name seen at 0x103fe70 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcFmModelConverterPass : Pass<"fm-model-converter", "func::FuncOp"> {
  let summary = "DWC pass fm-model-converter";
  let description = [{
    Boilerplate for the `(fm-model-converter` pass name seen at 0xdac265 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcFpa2bvModelConverterPass : Pass<"fpa2bv-model-converter", "func::FuncOp"> {
  let summary = "DWC pass fpa2bv-model-converter";
  let description = [{
    Boilerplate for the `(fpa2bv-model-converter` pass name seen at 0xdac220 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcGroupTpuOffloadsByParametersPass : Pass<"group-tpu-offloads-by-parameters", "func::FuncOp"> {
  let summary = "DWC pass group-tpu-offloads-by-parameters";
  let description = [{
    Boilerplate for the `group-tpu-offloads-by-parameters` pass name seen at 0xd821ce in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcInterpolateLoweringPassPass : Pass<"interpolate-lowering-pass", "func::FuncOp"> {
  let summary = "DWC pass interpolate-lowering-pass";
  let description = [{
    Boilerplate for the `interpolate-lowering-pass` pass name seen at 0xd7f143 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcIsNotImmutableTryRemovingMutableVariablesInYourModelSinceMutableVariablesAreCurrentlyNotSupportedThroughThisConverterPass : Pass<"is-not-immutable-try-removing-mutable-variables-in-your-model-since-mutable-variables-are-currently-not-supported-through-this-converter", "func::FuncOp"> {
  let summary = "DWC pass is-not-immutable-try-removing-mutable-variables-in-your-model-since-mutable-variables-are-currently-not-supported-through-this-converter";
  let description = [{
    Boilerplate for the `is not immutable, try removing mutable variables in your model since mutable variables are currently not supported through this converter` pass name seen at 0xdac290 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcLegalizePass : Pass<"Legalize", "func::FuncOp"> {
  let summary = "DWC pass Legalize";
  let description = [{
    Boilerplate for the `Legalize` pass name seen at 0xde6d4c in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcLegalizeAffinePass : Pass<"legalize-affine", "func::FuncOp"> {
  let summary = "DWC pass legalize-affine";
  let description = [{
    Boilerplate for the `legalize-affine` pass name seen at 0xe03694 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcLegalizeDwcPass : Pass<"legalize-dwc", "func::FuncOp"> {
  let summary = "DWC pass legalize-dwc";
  let description = [{
    Boilerplate for the `legalize-dwc` pass name seen at 0xe25250 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcLegalizeDwcInputOutputOpsPass : Pass<"legalize-dwc-input-output-ops", "func::FuncOp"> {
  let summary = "DWC pass legalize-dwc-input-output-ops";
  let description = [{
    Boilerplate for the `legalize-dwc-input-output-ops` pass name seen at 0xd844c7 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcLegalizeDwgTensorPass : Pass<"legalize-dwg-tensor", "func::FuncOp"> {
  let summary = "DWC pass legalize-dwg-tensor";
  let description = [{
    Boilerplate for the `legalize-dwg-tensor` pass name seen at 0xda8a5b in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcLegalizeQuantTypesPass : Pass<"legalize-quant-types", "func::FuncOp"> {
  let summary = "DWC pass legalize-quant-types";
  let description = [{
    Boilerplate for the `legalize-quant-types` pass name seen at 0xd94deb in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcLegalizeScfPass : Pass<"legalize-scf", "func::FuncOp"> {
  let summary = "DWC pass legalize-scf";
  let description = [{
    Boilerplate for the `legalize-scf` pass name seen at 0xde466f in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcLegalizeShapeOpsPass : Pass<"legalize-shape-ops", "func::FuncOp"> {
  let summary = "DWC pass legalize-shape-ops";
  let description = [{
    Boilerplate for the `legalize-shape-ops` pass name seen at 0xd84833 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcLegalizeTestUsingLayerirFlowPass : Pass<"legalize-test-using-layerir-flow", "func::FuncOp"> {
  let summary = "DWC pass legalize-test-using-layerir-flow";
  let description = [{
    Boilerplate for the `legalize test using layerir flow` pass name seen at 0xd61f43 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcLegalizeTfXlacallmoduleOpToStablehloPass : Pass<"legalize-tf-xlacallmodule-op-to-stablehlo", "func::FuncOp"> {
  let summary = "DWC pass legalize-tf-xlacallmodule-op-to-stablehlo";
  let description = [{
    Boilerplate for the `Legalize TF_XlaCallModule Op to stablehlo` pass name seen at 0xdbc2d1 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcLegalizeThreadObliviousOpPassPass : Pass<"legalize-thread-oblivious-op-pass", "func::FuncOp"> {
  let summary = "DWC pass legalize-thread-oblivious-op-pass";
  let description = [{
    Boilerplate for the `legalize-thread-oblivious-op-pass` pass name seen at 0xd7ee8c in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcLegalizeTypesForDiveVmTensorPass : Pass<"legalize-types-for-dive-vm-tensor", "func::FuncOp"> {
  let summary = "DWC pass legalize-types-for-dive-vm-tensor";
  let description = [{
    Boilerplate for the `legalize-types-for-dive-vm-tensor` pass name seen at 0xda8a1b in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcLegalizeStablehloCompositePass : Pass<"LegalizeStablehloComposite", "func::FuncOp"> {
  let summary = "DWC pass LegalizeStablehloComposite";
  let description = [{
    Boilerplate for the `LegalizeStablehloComposite` pass name seen at 0xdef61e in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcLowerAffinePass : Pass<"dwc-lower-affine-only", "func::FuncOp"> {
  let summary = "DWC pass lower-affine";
  let description = [{
    Boilerplate for the `lower-affine` pass name seen at 0xe03675 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcLowerAllFunctionsPass : Pass<"lower-all-functions", "func::FuncOp"> {
  let summary = "DWC pass lower-all-functions";
  let description = [{
    Boilerplate for the `lower-all-functions` pass name seen at 0xd873dd in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcLowerAllPadsPass : Pass<"lower-all-pads", "func::FuncOp"> {
  let summary = "DWC pass lower-all-pads";
  let description = [{
    Boilerplate for the `lower-all-pads` pass name seen at 0xd9c263 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcLowerAttentionOpsPass : Pass<"lower-attention-ops", "func::FuncOp"> {
  let summary = "DWC pass lower-attention-ops";
  let description = [{
    Boilerplate for the `lower-attention-ops` pass name seen at 0xd845ec in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcLowerInputCastPass : Pass<"lower-input-cast", "func::FuncOp"> {
  let summary = "DWC pass lower-input-cast";
  let description = [{
    Boilerplate for the `lower-input-cast` pass name seen at 0xd680b2 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcLowerJoinPass : Pass<"lower-join", "func::FuncOp"> {
  let summary = "DWC pass lower-join";
  let description = [{
    Boilerplate for the `lower-join` pass name seen at 0xdc9464 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcLowerOutputCastPass : Pass<"lower-output-cast", "func::FuncOp"> {
  let summary = "DWC pass lower-output-cast";
  let description = [{
    Boilerplate for the `lower-output-cast` pass name seen at 0xd680a0 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcLowerArgmaxIndexUnpoolPass : Pass<"LowerArgmaxIndexUnpool", "func::FuncOp"> {
  let summary = "DWC pass LowerArgmaxIndexUnpool";
  let description = [{
    Boilerplate for the `LowerArgmaxIndexUnpool` pass name seen at 0xdcf04e in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcMarkDiveVmTensorInsertSliceOpsPass : Pass<"mark-dive-vm-tensor-insert-slice-ops", "func::FuncOp"> {
  let summary = "DWC pass mark-dive-vm-tensor-insert-slice-ops";
  let description = [{
    Boilerplate for the `mark-dive-vm-tensor-insert-slice-ops` pass name seen at 0xd8488d in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcMhloLegalizeEinsumToDotGeneralPass : Pass<"mhlo-legalize-einsum-to-dot-general", "func::FuncOp"> {
  let summary = "DWC pass mhlo-legalize-einsum-to-dot-general";
  let description = [{
    Boilerplate for the `mhlo-legalize-einsum-to-dot-general` pass name seen at 0xdd34bf in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcMidToLowLevelLoweringPass : Pass<"mid-to-low-level-lowering", "func::FuncOp"> {
  let summary = "DWC pass mid-to-low-level-lowering";
  let description = [{
    Boilerplate for the `mid-to-low-level-lowering` pass name seen at 0xddedf4 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcMlirDarwinnComputeEnginePass : Pass<"mlir-darwinn-compute-engine", "func::FuncOp"> {
  let summary = "DWC pass mlir-darwinn-compute-engine";
  let description = [{
    Boilerplate for the `::mlir::darwinn::compute::Engine` pass name seen at 0xe03654 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcOnlyDenseelementsattrAreSupportedForConstantLoweringPass : Pass<"only-denseelementsattr-are-supported-for-constant-lowering", "func::FuncOp"> {
  let summary = "DWC pass only-denseelementsattr-are-supported-for-constant-lowering";
  let description = [{
    Boilerplate for the `Only DenseElementsAttr are supported for constant lowering` pass name seen at 0xddee55 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcOptimizeDiveVmTensorInsertSlicePass : Pass<"optimize-dive-vm-tensor-insert-slice", "func::FuncOp"> {
  let summary = "DWC pass optimize-dive-vm-tensor-insert-slice";
  let description = [{
    Boilerplate for the `optimize-dive-vm-tensor-insert-slice` pass name seen at 0xe0fa93 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcParameterCachingDiveProgramPass : Pass<"parameter-caching-dive-program", "func::FuncOp"> {
  let summary = "DWC pass parameter-caching-dive-program";
  let description = [{
    Boilerplate for the `-parameter-caching-dive-program` pass name seen at 0xdce5e1 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcPlatformsDarwinnCodeGeneratorEntryScoreTypePass : Pass<"platforms-darwinn-code-generator-entry-score-type", "func::FuncOp"> {
  let summary = "DWC pass platforms-darwinn-code-generator-entry-score-type";
  let description = [{
    Boilerplate for the `platforms.darwinn.code_generator.Entry.Score.type` pass name seen at 0xdf68e2 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcPlatformsDarwinnCompilerProbeinstrumentationlocationConstraintsFunctionsPass : Pass<"platforms-darwinn-compiler-probeinstrumentationlocation-constraints-functions", "func::FuncOp"> {
  let summary = "DWC pass platforms-darwinn-compiler-probeinstrumentationlocation-constraints-functions";
  let description = [{
    Boilerplate for the `platforms.darwinn.compiler.ProbeInstrumentationLocation.Constraints.functions` pass name seen at 0xd87375 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcQuantSignednessConvertLoweringPass : Pass<"quant-signedness-convert-lowering", "func::FuncOp"> {
  let summary = "DWC pass quant-signedness-convert-lowering";
  let description = [{
    Boilerplate for the `quant-signedness-convert-lowering` pass name seen at 0xdded43 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcR52ReadsDiveBuffersPass : Pass<"r52-reads-dive-buffers", "func::FuncOp"> {
  let summary = "DWC pass r52-reads-dive-buffers";
  let description = [{
    Boilerplate for the `r52-reads-dive-buffers` pass name seen at 0xd833a6 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcRedistributeLoweringPass : Pass<"redistribute-lowering", "func::FuncOp"> {
  let summary = "DWC pass redistribute-lowering";
  let description = [{
    Boilerplate for the `redistribute-lowering` pass name seen at 0xddee0e in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcRedistributeLoweringPassRemarksPass : Pass<"redistribute-lowering-pass-remarks", "func::FuncOp"> {
  let summary = "DWC pass redistribute-lowering-pass-remarks";
  let description = [{
    Boilerplate for the `redistribute-lowering-pass-remarks` pass name seen at 0xd8c505 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcReinterpretCastRankLegalizePassPass : Pass<"reinterpret-cast-rank-legalize-pass", "func::FuncOp"> {
  let summary = "DWC pass reinterpret-cast-rank-legalize-pass";
  let description = [{
    Boilerplate for the `reinterpret-cast-rank-legalize-pass` pass name seen at 0xd7f295 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcRenameDiveEntryFunctionPass : Pass<"rename-dive-entry-function", "func::FuncOp"> {
  let summary = "DWC pass rename-dive-entry-function";
  let description = [{
    Boilerplate for the `rename-dive-entry-function` pass name seen at 0xdbfc27 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcResamplerLoweringPass : Pass<"resampler-lowering", "func::FuncOp"> {
  let summary = "DWC pass resampler-lowering";
  let description = [{
    Boilerplate for the `resampler-lowering` pass name seen at 0xdded92 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcRkhyShapeLegalizationPassPass : Pass<"rkhy-shape-legalization-pass", "func::FuncOp"> {
  let summary = "DWC pass rkhy-shape-legalization-pass";
  let description = [{
    Boilerplate for the `rkhy-shape-legalization-pass` pass name seen at 0xd7f029 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcRkhyTypeLegalizationPassPass : Pass<"rkhy-type-legalization-pass", "func::FuncOp"> {
  let summary = "DWC pass rkhy-type-legalization-pass";
  let description = [{
    Boilerplate for the `rkhy-type-legalization-pass` pass name seen at 0xd7f00d in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcRunR52OpsOnDivePass : Pass<"run-r52-ops-on-dive", "func::FuncOp"> {
  let summary = "DWC pass run-r52-ops-on-dive";
  let description = [{
    Boilerplate for the `run-r52-ops-on-dive` pass name seen at 0xde9621 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcScalarCoreControlFlowLoweringPass : Pass<"scalar-core-control-flow-lowering", "func::FuncOp"> {
  let summary = "DWC pass scalar-core-control-flow-lowering";
  let description = [{
    Boilerplate for the `scalar-core-control-flow-lowering` pass name seen at 0xdded21 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcScalarCoreStdOpsLoweringPass : Pass<"scalar-core-std-ops-lowering", "func::FuncOp"> {
  let summary = "DWC pass scalar-core-std-ops-lowering";
  let description = [{
    Boilerplate for the `scalar-core-std-ops-lowering` pass name seen at 0xdded75 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcScalarOpsLegalizePass : Pass<"scalar-ops-legalize", "func::FuncOp"> {
  let summary = "DWC pass scalar-ops-legalize";
  let description = [{
    Boilerplate for the `scalar-ops-legalize` pass name seen at 0xde6d2b in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcScatterGatherLoweringPass : Pass<"scatter-gather-lowering", "func::FuncOp"> {
  let summary = "DWC pass scatter-gather-lowering";
  let description = [{
    Boilerplate for the `scatter-gather-lowering` pass name seen at 0xddeda5 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcSelectLoweringPass : Pass<"select-lowering", "func::FuncOp"> {
  let summary = "DWC pass select-lowering";
  let description = [{
    Boilerplate for the `select-lowering` pass name seen at 0xdded65 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcShardingUsingDivePass : Pass<"sharding-using-dive", "func::FuncOp"> {
  let summary = "DWC pass sharding-using-dive";
  let description = [{
    Boilerplate for the `sharding-using-dive` pass name seen at 0xde9635 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcSkippingFoldOfFloatConvertPass : Pass<"skipping-fold-of-float-convert", "func::FuncOp"> {
  let summary = "DWC pass skipping-fold-of-float-convert";
  let description = [{
    Boilerplate for the `skipping fold of float convert` pass name seen at 0xd68cda in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcSplitOpLoweringPass : Pass<"split-op-lowering", "func::FuncOp"> {
  let summary = "DWC pass split-op-lowering";
  let description = [{
    Boilerplate for the `split-op-lowering` pass name seen at 0xddede2 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcStablehloCompositeLegalizeTflCustomPass : Pass<"stablehlo-composite-legalize-tfl-custom", "func::FuncOp"> {
  let summary = "DWC pass stablehlo-composite-legalize-tfl-custom";
  let description = [{
    Boilerplate for the `stablehlo-composite-legalize-tfl-custom` pass name seen at 0xdccdf7 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcStablehloCustomCallLegalizeCompositePass : Pass<"stablehlo-custom-call-legalize-composite", "func::FuncOp"> {
  let summary = "DWC pass stablehlo-custom-call-legalize-composite";
  let description = [{
    Boilerplate for the `stablehlo-custom-call-legalize-composite` pass name seen at 0xdef5f5 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcStablehloLegalizeCompositeToCallPass : Pass<"stablehlo-legalize-composite-to-call", "func::FuncOp"> {
  let summary = "DWC pass stablehlo-legalize-composite-to-call";
  let description = [{
    Boilerplate for the `stablehlo-legalize-composite-to-call` pass name seen at 0xdd00c2 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcStablehloLegalizeToHloPass : Pass<"stablehlo-legalize-to-hlo", "func::FuncOp"> {
  let summary = "DWC pass stablehlo-legalize-to-hlo";
  let description = [{
    Boilerplate for the `stablehlo-legalize-to-hlo` pass name seen at 0xdbc34d in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcStablehloLegalizeToVhloPass : Pass<"stablehlo-legalize-to-vhlo", "func::FuncOp"> {
  let summary = "DWC pass stablehlo-legalize-to-vhlo";
  let description = [{
    Boilerplate for the `stablehlo-legalize-to-vhlo` pass name seen at 0xdbc236 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcStablehloLegalizeVhloPass : Pass<"stablehlo-legalize-vhlo", "func::FuncOp"> {
  let summary = "DWC pass stablehlo-legalize-vhlo";
  let description = [{
    Boilerplate for the `stablehlo-legalize-vhlo` pass name seen at 0xdbc251 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcStochasticConvertPass : Pass<"stochastic-convert", "func::FuncOp"> {
  let summary = "DWC pass stochastic-convert";
  let description = [{
    Boilerplate for the `stochastic-convert` pass name seen at 0xd68cc7 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcTfLegalizeHloPass : Pass<"tf-legalize-hlo", "func::FuncOp"> {
  let summary = "DWC pass tf-legalize-hlo";
  let description = [{
    Boilerplate for the `tf-legalize-hlo` pass name seen at 0xdbc38d in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcTflCustomLoweringRewritingPassPass : Pass<"tfl-custom-lowering-rewriting-pass", "func::FuncOp"> {
  let summary = "DWC pass tfl-custom-lowering-rewriting-pass";
  let description = [{
    Boilerplate for the `tfl-custom-lowering-rewriting-pass` pass name seen at 0xd7f120 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcTflLegalizeChloPass : Pass<"tfl-legalize-chlo", "func::FuncOp"> {
  let summary = "DWC pass tfl-legalize-chlo";
  let description = [{
    Boilerplate for the `tfl-legalize-chlo` pass name seen at 0xdbc2fb in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcTflLegalizeHashtablesTfPass : Pass<"tfl-legalize-hashtables-tf", "func::FuncOp"> {
  let summary = "DWC pass tfl-legalize-hashtables-tf";
  let description = [{
    Boilerplate for the `tfl-legalize-hashtables-tf` pass name seen at 0xde2a7d in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcTflLegalizeHloPass : Pass<"tfl-legalize-hlo", "func::FuncOp"> {
  let summary = "DWC pass tfl-legalize-hlo";
  let description = [{
    Boilerplate for the `tfl-legalize-hlo` pass name seen at 0xdbc37c in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcTflLegalizeTensorlistPass : Pass<"tfl-legalize-tensorlist", "func::FuncOp"> {
  let summary = "DWC pass tfl-legalize-tensorlist";
  let description = [{
    Boilerplate for the `tfl-legalize-tensorlist` pass name seen at 0xd66a8f in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcTflLegalizeTfPass : Pass<"tfl-legalize-tf", "func::FuncOp"> {
  let summary = "DWC pass tfl-legalize-tf";
  let description = [{
    Boilerplate for the `tfl-legalize-tf` pass name seen at 0xde2ae8 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcTflLegalizeTfWhilePass : Pass<"tfl-legalize-tf-while", "func::FuncOp"> {
  let summary = "DWC pass tfl-legalize-tf-while";
  let description = [{
    Boilerplate for the `tfl-legalize-tf-while` pass name seen at 0xe07f5e in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcTflLegalizeVariablesTfPass : Pass<"tfl-legalize-variables-tf", "func::FuncOp"> {
  let summary = "DWC pass tfl-legalize-variables-tf";
  let description = [{
    Boilerplate for the `tfl-legalize-variables-tf` pass name seen at 0xde2a98 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcTflLowerQuantAnnotationsPass : Pass<"tfl-lower-quant-annotations", "func::FuncOp"> {
  let summary = "DWC pass tfl-lower-quant-annotations";
  let description = [{
    Boilerplate for the `tfl-lower-quant-annotations` pass name seen at 0xd87c99 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcTflLowerStaticTensorListPass : Pass<"tfl-lower-static-tensor-list", "func::FuncOp"> {
  let summary = "DWC pass tfl-lower-static-tensor-list";
  let description = [{
    Boilerplate for the `tfl-lower-static-tensor-list` pass name seen at 0xd66ca5 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcTopKLoweringPolicyPass : Pass<"top-k-lowering-policy", "func::FuncOp"> {
  let summary = "DWC pass top-k-lowering-policy";
  let description = [{
    Boilerplate for the `top-k-lowering-policy` pass name seen at 0xd5dd44 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcTpuClusteringAlgorithmPass : Pass<"tpu-clustering-algorithm", "func::FuncOp"> {
  let summary = "DWC pass tpu-clustering-algorithm";
  let description = [{
    Boilerplate for the `tpu-clustering-algorithm` pass name seen at 0xdcdd6e in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcVhloLegalizeStablehloPass : Pass<"vhlo-legalize-stablehlo", "func::FuncOp"> {
  let summary = "DWC pass vhlo-legalize-stablehlo";
  let description = [{
    Boilerplate for the `vhlo-legalize-stablehlo` pass name seen at 0xdbc299 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcVhloLegalizeToStablehloPass : Pass<"vhlo-legalize-to-stablehlo", "func::FuncOp"> {
  let summary = "DWC pass vhlo-legalize-to-stablehlo";
  let description = [{
    Boilerplate for the `vhlo-legalize-to-stablehlo` pass name seen at 0xdbc271 in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcWrapUpDiveProgramPass : Pass<"wrap-up-dive-program", "func::FuncOp"> {
  let summary = "DWC pass wrap-up-dive-program";
  let description = [{
    Boilerplate for the `wrap-up-dive-program` pass name seen at 0xdce5cc in
    task3_passes.tsv. Empty body pending real lowering logic.
  }];
}

def DwcLowerCopySlicePass : Pass<"lower-copy-slice", "func::FuncOp"> {
  let summary = "Lower darwinn copy, slice and gather ops to dive_vm";
  let description = [{
    Lowers `darwinn.copy_op` to `dive_vm.copy` after checking the source
    and destination element types agree and any static element counts match,
    lowers the `darwinn.dynamic_slice` family to `dive_vm.address_of_activation`
    feeding `dive_vm.extract_slice` or `dive_vm.insert_slice` with slice-offset
    generator attributes (mirroring the SliceOffsetGenerator plus
    GetContinuousSlice fast path in the InsertSlice and ExtractSlice kernels),
    and lowers the `darwinn.gather` family to `dive_vm.gather` with
    out-of-bounds zero-fill (mirroring the Gather kernel).
  }];
}

def DwcXlaCpuUseNewXtileLoweringPass : Pass<"xla-cpu-use-new-xtile-lowering", "func::FuncOp"> {
  let summary = "XLA CPU new-xtile lowering";
  let description = [{
    Boilerplate for the `xla-cpu-use-new-xtile-lowering` pass name seen at
    0xdded02 in task3_passes.tsv.
  }];
}

def DwcLowerConvertPass : Pass<"lower-convert", "func::FuncOp"> {
  let summary = "Lower darwinn convert and cast ops to dive_vm";
  let description = [{
    Lowers `darwinn.convert`, `darwinn.cast_in`, and `darwinn.cast_out` to
    the `dive_vm` elementwise convert. float-to-int truncates toward zero,
    int-to-float rounds to nearest even, and narrow stores keep the low
    bits. Same-type converts fold away. Patterns live in LowerConvert.cpp.
  }];
}

#endif // MLIR_LIB_DIALECT_DARWINN_TRANSFORMS_DWCPASSES
