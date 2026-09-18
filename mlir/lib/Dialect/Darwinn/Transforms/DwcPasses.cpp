//===- DwcPasses.cpp - Darwinn DWC pass boilerplate -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Boilerplate stubs for every kept pass name in task3_passes.tsv
// (kind=pass rows). Each runOnOperation body is empty pending real logic.
//
// Canonical pipeline order:
//   dwc-legalize family, dwc-lower family, convert-dive-vm-to-llvm,
//   convert-tpu-offload-to-llvm, dive-program-tpu.
// group-tpu-offloads-by-parameters runs with the TPU offload grouping stage.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace darwinn {
#define GEN_PASS_DECL
#include "DwcPasses.h.inc"
#define GEN_PASS_DEF_DWCACKRMODELCONVERTERPASS
#define GEN_PASS_DEF_DWCADDBOUNDLOWERPASS
#define GEN_PASS_DEF_DWCADDDIVEABIARGUMENTSPASS
#define GEN_PASS_DEF_DWCADDDIVETRACINGPASS
#define GEN_PASS_DEF_DWCALLOWBF16ANDF16TYPELEGALIZATIONPASS
#define GEN_PASS_DEF_DWCARITHASSERTLOWERPASS
#define GEN_PASS_DEF_DWCARITHLOWERPASS
#define GEN_PASS_DEF_DWCBITCASTCONVERTPASS
#define GEN_PASS_DEF_DWCCHLOLEGALIZETOHLOPASS
#define GEN_PASS_DEF_DWCCOMPOSITELOWERINGPASS
#define GEN_PASS_DEF_DWCCONCATMODELCONVERTERPASS
#define GEN_PASS_DEF_DWCCONCATPROOFCONVERTERPASS
#define GEN_PASS_DEF_DWCCONVERTARITHTOLLVMPASS
#define GEN_PASS_DEF_DWCCONVERTCFTOLLVMPASS
#define GEN_PASS_DEF_DWCCONVERTCONV1X1TOFCPASS
#define GEN_PASS_DEF_DWCCONVERTDIVEVMTENSORTOLINALGPASS
#define GEN_PASS_DEF_DWCCONVERTDIVEVMTENSORTOSCFPASS
#define GEN_PASS_DEF_DWCCONVERTDIVEVMTENSORTOTENSORPASS
#define GEN_PASS_DEF_DWCCONVERTDIVEVMTOLLVMPASS
#define GEN_PASS_DEF_DWCCONVERTDIVEVMTOMEMREFPASS
#define GEN_PASS_DEF_DWCCONVERTDWCTODIVEVMTENSORPASS
#define GEN_PASS_DEF_DWCCONVERTDWGTODIVEVMPASS
#define GEN_PASS_DEF_DWCCONVERTDYNAMICSHAPESCOPETODIVEVMPASS
#define GEN_PASS_DEF_DWCCONVERTFUNCTOLLVMPASS
#define GEN_PASS_DEF_DWCCONVERTGENERICNORMTOPSEUDOOPPASS
#define GEN_PASS_DEF_DWCCONVERTLINALGTOLOOPSPASS
#define GEN_PASS_DEF_DWCCONVERTMATHTOLIBMPASS
#define GEN_PASS_DEF_DWCCONVERTMATHTOLLVMPASS
#define GEN_PASS_DEF_DWCCONVERTOPLOWERINGPASS
#define GEN_PASS_DEF_DWCCONVERTPDLTOPDLINTERPPASS
#define GEN_PASS_DEF_DWCCONVERTSCATTERTOGENERICSCATTERPASS
#define GEN_PASS_DEF_DWCCONVERTSCFTOCFPASS
#define GEN_PASS_DEF_DWCCONVERTSIGNEDINTWITHRESCALINGOPSPASS
#define GEN_PASS_DEF_DWCCONVERTSPATIALREDUCTIONTOPOOLINGPASS
#define GEN_PASS_DEF_DWCCONVERTTFTODWCPASS
#define GEN_PASS_DEF_DWCCONVERTTOKINMSPARSITYPASS
#define GEN_PASS_DEF_DWCCONVERTTPUOFFLOADTODIVEVMPASS
#define GEN_PASS_DEF_DWCCONVERTTPUOFFLOADTOLLVMPASS
#define GEN_PASS_DEF_DWCCONVERTXLASUPPORTEDSTABLEHLOPASS
#define GEN_PASS_DEF_DWCCONVERTDIVEVMTENSORTOLINALGSYMBOLPASS
#define GEN_PASS_DEF_DWCCONVERTTPUOFFLOADTOLLVMSYMBOLPASS
#define GEN_PASS_DEF_DWCCOPYOPLOWERINGPASS
#define GEN_PASS_DEF_DWCDARWINNBUNDLINGPASS
#define GEN_PASS_DEF_DWCDARWINNCONVERTPASS
#define GEN_PASS_DEF_DWCDARWINNMATHJOINPASS
#define GEN_PASS_DEF_DWCDARWINNSPARSITYPASS
#define GEN_PASS_DEF_DWCDIVEDCEPASS
#define GEN_PASS_DEF_DWCDIVEIOOPTIMIZATIONPASS
#define GEN_PASS_DEF_DWCDIVEPROGRAMTPUPASS
#define GEN_PASS_DEF_DWCDIVEUNROLLFACTORPASS
#define GEN_PASS_DEF_DWCDIVEVMBUFFERIZEPASS
#define GEN_PASS_DEF_DWCDIVEVMOUTLINESHAREABLEDIVECONSTSPASS
#define GEN_PASS_DEF_DWCDWCCHECKILLEGALTPUOPSPASS
#define GEN_PASS_DEF_DWCDWCCONVERTINPUTOUTPUTTYPESPASS
#define GEN_PASS_DEF_DWCDWCCOPYSTRIDEDBUFFERSONTPUPASS
#define GEN_PASS_DEF_DWCDWCFORMTPUCLUSTERSPASS
#define GEN_PASS_DEF_DWCDWCLEGALIZEPASS
#define GEN_PASS_DEF_DWCDWCLEGALIZEHLOPASS
#define GEN_PASS_DEF_DWCDWCLEGALIZEHLOTOTFPASS
#define GEN_PASS_DEF_DWCDWCLEGALIZEINTANDQUANTTYPESPASS
#define GEN_PASS_DEF_DWCDWCLEGALIZEINT64CONSTANTSPASS
#define GEN_PASS_DEF_DWCDWCLEGALIZEPASSPASS
#define GEN_PASS_DEF_DWCDWCLEGALIZESTABLEHLOANNOTATEMATERIALIZEPOLICYPASS
#define GEN_PASS_DEF_DWCDWCLEGALIZESTABLEHLOCOMPOSITEPASS
#define GEN_PASS_DEF_DWCDWCLEGALIZETFPIPELINEPASS
#define GEN_PASS_DEF_DWCDWCLEGALIZETFLCUDAEMUCUSTOMOPSPASS
#define GEN_PASS_DEF_DWCDWCLEGALIZETFLMULTINOMIALPASS
#define GEN_PASS_DEF_DWCDWCLEGALIZETFLVARIABLETENSORSPASS
#define GEN_PASS_DEF_DWCDWCLEGALIZEUINT32TYPESPASS
#define GEN_PASS_DEF_DWCDWCLOWERARGMAXINDEXUNPOOLPASS
#define GEN_PASS_DEF_DWCDWCLOWERCOMPOSITEOPSPASS
#define GEN_PASS_DEF_DWCDWCLOWERCONTROLFLOWPASS
#define GEN_PASS_DEF_DWCDWCLOWERDEPTHTOFROMSPACEPASS
#define GEN_PASS_DEF_DWCDWCLOWERGENERICCONSTANTSPASS
#define GEN_PASS_DEF_DWCDWCLOWERHLOPSPASS
#define GEN_PASS_DEF_DWCDWCLOWERINPUTOUTPUTCASTPASS
#define GEN_PASS_DEF_DWCDWCLOWERPADDINGOPSPASS
#define GEN_PASS_DEF_DWCDWCLOWERPSEUDOOPSPASS
#define GEN_PASS_DEF_DWCDWCLOWERRESAMPLEROPSPASS
#define GEN_PASS_DEF_DWCDWCLOWERSCALAROPSPASS
#define GEN_PASS_DEF_DWCDWCLOWERSCATTEROPSPASS
#define GEN_PASS_DEF_DWCDWCLOWERTOPKPASS
#define GEN_PASS_DEF_DWCDWCPOSTTRUNCATIONTPUFITTERPASS
#define GEN_PASS_DEF_DWCDWCPRETPUFITTEROPTIMIZEGATHERPASS
#define GEN_PASS_DEF_DWCDWCPRETPUFITTEROPTIMIZESCATTERPASS
#define GEN_PASS_DEF_DWCDWCREGROUPTPUFUNCTIONSPASS
#define GEN_PASS_DEF_DWCDWCSERIALIZETPUOFFLOADSPASS
#define GEN_PASS_DEF_DWCDWCTESTREPEATTPUOPSPASS
#define GEN_PASS_DEF_DWCDWCTPUFITTERPASS
#define GEN_PASS_DEF_DWCDWCTPUFUNCTIONCSEPASS
#define GEN_PASS_DEF_DWCDWGCREATEDARWINNCUSTOMOPPASS
#define GEN_PASS_DEF_DWCDWGFORKMULTICORETPUOFFLOADSPASS
#define GEN_PASS_DEF_DWCDWGLOWERFORTOWHILEPASS
#define GEN_PASS_DEF_DWCDWGTLOWERINDEXTYPEPASS
#define GEN_PASS_DEF_DWCDYNAMICUPDATESLICELOWERINGPASS
#define GEN_PASS_DEF_DWCEDGETPUCUSTOMOP2PASS
#define GEN_PASS_DEF_DWCFMMODELCONVERTERPASS
#define GEN_PASS_DEF_DWCFPA2BVMODELCONVERTERPASS
#define GEN_PASS_DEF_DWCGROUPTPUOFFLOADSBYPARAMETERSPASS
#define GEN_PASS_DEF_DWCINTERPOLATELOWERINGPASSPASS
#define GEN_PASS_DEF_DWCISNOTIMMUTABLETRYREMOVINGMUTABLEVARIABLESINYOURMODELSINCEMUTABLEVARIABLESARECURRENTLYNOTSUPPORTEDTHROUGHTHISCONVERTERPASS
#define GEN_PASS_DEF_DWCLEGALIZEPASS
#define GEN_PASS_DEF_DWCLEGALIZEAFFINEPASS
#define GEN_PASS_DEF_DWCLEGALIZEDWCPASS
#define GEN_PASS_DEF_DWCLEGALIZEDWCINPUTOUTPUTOPSPASS
#define GEN_PASS_DEF_DWCLEGALIZEDWGTENSORPASS
#define GEN_PASS_DEF_DWCLEGALIZEQUANTTYPESPASS
#define GEN_PASS_DEF_DWCLEGALIZESCFPASS
#define GEN_PASS_DEF_DWCLEGALIZESHAPEOPSPASS
#define GEN_PASS_DEF_DWCLEGALIZETESTUSINGLAYERIRFLOWPASS
#define GEN_PASS_DEF_DWCLEGALIZETFXLACALLMODULEOPTOSTABLEHLOPASS
#define GEN_PASS_DEF_DWCLEGALIZETHREADOBLIVIOUSOPPASSPASS
#define GEN_PASS_DEF_DWCLEGALIZETYPESFORDIVEVMTENSORPASS
#define GEN_PASS_DEF_DWCLEGALIZESTABLEHLOCOMPOSITEPASS
#define GEN_PASS_DEF_DWCLOWERAFFINEPASS
#define GEN_PASS_DEF_DWCLOWERALLFUNCTIONSPASS
#define GEN_PASS_DEF_DWCLOWERALLPADSPASS
#define GEN_PASS_DEF_DWCLOWERATTENTIONOPSPASS
#define GEN_PASS_DEF_DWCLOWERINPUTCASTPASS
#define GEN_PASS_DEF_DWCLOWERJOINPASS
#define GEN_PASS_DEF_DWCLOWEROUTPUTCASTPASS
#define GEN_PASS_DEF_DWCLOWERARGMAXINDEXUNPOOLPASS
#define GEN_PASS_DEF_DWCMARKDIVEVMTENSORINSERTSLICEOPSPASS
#define GEN_PASS_DEF_DWCMHLOLEGALIZEEINSUMTODOTGENERALPASS
#define GEN_PASS_DEF_DWCMIDTOLOWLEVELLOWERINGPASS
#define GEN_PASS_DEF_DWCMLIRDARWINNCOMPUTEENGINEPASS
#define GEN_PASS_DEF_DWCONLYDENSEELEMENTSATTRARESUPPORTEDFORCONSTANTLOWERINGPASS
#define GEN_PASS_DEF_DWCOPTIMIZEDIVEVMTENSORINSERTSLICEPASS
#define GEN_PASS_DEF_DWCPARAMETERCACHINGDIVEPROGRAMPASS
#define GEN_PASS_DEF_DWCPLATFORMSDARWINNCODEGENERATORENTRYSCORETYPEPASS
#define GEN_PASS_DEF_DWCPLATFORMSDARWINNCOMPILERPROBEINSTRUMENTATIONLOCATIONCONSTRAINTSFUNCTIONSPASS
#define GEN_PASS_DEF_DWCQUANTSIGNEDNESSCONVERTLOWERINGPASS
#define GEN_PASS_DEF_DWCR52READSDIVEBUFFERSPASS
#define GEN_PASS_DEF_DWCREDISTRIBUTELOWERINGPASS
#define GEN_PASS_DEF_DWCREDISTRIBUTELOWERINGPASSREMARKSPASS
#define GEN_PASS_DEF_DWCREINTERPRETCASTRANKLEGALIZEPASSPASS
#define GEN_PASS_DEF_DWCRENAMEDIVEENTRYFUNCTIONPASS
#define GEN_PASS_DEF_DWCRESAMPLERLOWERINGPASS
#define GEN_PASS_DEF_DWCRKHYSHAPELEGALIZATIONPASSPASS
#define GEN_PASS_DEF_DWCRKHYTYPELEGALIZATIONPASSPASS
#define GEN_PASS_DEF_DWCRUNR52OPSONDIVEPASS
#define GEN_PASS_DEF_DWCSCALARCORECONTROLFLOWLOWERINGPASS
#define GEN_PASS_DEF_DWCSCALARCORESTDOPSLOWERINGPASS
#define GEN_PASS_DEF_DWCSCALAROPSLEGALIZEPASS
#define GEN_PASS_DEF_DWCSCATTERGATHERLOWERINGPASS
#define GEN_PASS_DEF_DWCSELECTLOWERINGPASS
#define GEN_PASS_DEF_DWCSHARDINGUSINGDIVEPASS
#define GEN_PASS_DEF_DWCSKIPPINGFOLDOFFLOATCONVERTPASS
#define GEN_PASS_DEF_DWCSPLITOPLOWERINGPASS
#define GEN_PASS_DEF_DWCSTABLEHLOCOMPOSITELEGALIZETFLCUSTOMPASS
#define GEN_PASS_DEF_DWCSTABLEHLOCUSTOMCALLLEGALIZECOMPOSITEPASS
#define GEN_PASS_DEF_DWCSTABLEHLOLEGALIZECOMPOSITETOCALLPASS
#define GEN_PASS_DEF_DWCSTABLEHLOLEGALIZETOHLOPASS
#define GEN_PASS_DEF_DWCSTABLEHLOLEGALIZETOVHLOPASS
#define GEN_PASS_DEF_DWCSTABLEHLOLEGALIZEVHLOPASS
#define GEN_PASS_DEF_DWCSTOCHASTICCONVERTPASS
#define GEN_PASS_DEF_DWCTFLEGALIZEHLOPASS
#define GEN_PASS_DEF_DWCTFLCUSTOMLOWERINGREWRITINGPASSPASS
#define GEN_PASS_DEF_DWCTFLLEGALIZECHLOPASS
#define GEN_PASS_DEF_DWCTFLLEGALIZEHASHTABLESTFPASS
#define GEN_PASS_DEF_DWCTFLLEGALIZEHLOPASS
#define GEN_PASS_DEF_DWCTFLLEGALIZETENSORLISTPASS
#define GEN_PASS_DEF_DWCTFLLEGALIZETFPASS
#define GEN_PASS_DEF_DWCTFLLEGALIZETFWHILEPASS
#define GEN_PASS_DEF_DWCTFLLEGALIZEVARIABLESTFPASS
#define GEN_PASS_DEF_DWCTFLLOWERQUANTANNOTATIONSPASS
#define GEN_PASS_DEF_DWCTFLLOWERSTATICTENSORLISTPASS
#define GEN_PASS_DEF_DWCTOPKLOWERINGPOLICYPASS
#define GEN_PASS_DEF_DWCTPUCLUSTERINGALGORITHMPASS
#define GEN_PASS_DEF_DWCVHLOLEGALIZESTABLEHLOPASS
#define GEN_PASS_DEF_DWCVHLOLEGALIZETOSTABLEHLOPASS
#define GEN_PASS_DEF_DWCWRAPUPDIVEPROGRAMPASS
#define GEN_PASS_DEF_DWCXLACPUUSENEWXTILELOWERINGPASS
#include "DwcPasses.h.inc"
} // namespace darwinn
} // namespace mlir

using namespace mlir;
using namespace mlir::darwinn;

namespace {

// TSV row: "(ackr-model-converter" at 0xdac24f.
struct DwcAckrModelConverterPass : public darwinn::impl::DwcAckrModelConverterPassBase<DwcAckrModelConverterPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "add_bound_lower" at 0xdaad7f.
struct DwcAddBoundLowerPass : public darwinn::impl::DwcAddBoundLowerPassBase<DwcAddBoundLowerPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "add-dive-abi-arguments" at 0xd7a252.
struct DwcAddDiveAbiArgumentsPass : public darwinn::impl::DwcAddDiveAbiArgumentsPassBase<DwcAddDiveAbiArgumentsPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "add-dive-tracing" at 0xde194d.
struct DwcAddDiveTracingPass : public darwinn::impl::DwcAddDiveTracingPassBase<DwcAddDiveTracingPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "allow-bf16-and-f16-type-legalization" at 0xdc1b37.
struct DwcAllowBf16AndF16TypeLegalizationPass : public darwinn::impl::DwcAllowBf16AndF16TypeLegalizationPassBase<DwcAllowBf16AndF16TypeLegalizationPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "arith assert lower" at 0xdaad9b.
struct DwcArithAssertLowerPass : public darwinn::impl::DwcArithAssertLowerPassBase<DwcArithAssertLowerPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "arith-lower" at 0xdaad8f.
struct DwcArithLowerPass : public darwinn::impl::DwcArithLowerPassBase<DwcArithLowerPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "bitcast-convert" at 0xd68cb7.
struct DwcBitcastConvertPass : public darwinn::impl::DwcBitcastConvertPassBase<DwcBitcastConvertPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "chlo-legalize-to-hlo" at 0xdbc367.
struct DwcChloLegalizeToHloPass : public darwinn::impl::DwcChloLegalizeToHloPassBase<DwcChloLegalizeToHloPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "composite-lowering" at 0xddee24.
struct DwcCompositeLoweringPass : public darwinn::impl::DwcCompositeLoweringPassBase<DwcCompositeLoweringPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "concat-model-converter" at 0xdac238.
struct DwcConcatModelConverterPass : public darwinn::impl::DwcConcatModelConverterPassBase<DwcConcatModelConverterPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "concat-proof-converter" at 0xdac279.
struct DwcConcatProofConverterPass : public darwinn::impl::DwcConcatProofConverterPassBase<DwcConcatProofConverterPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "convert-arith-to-llvm" at 0xdcb7f5.
struct DwcConvertArithToLlvmPass : public darwinn::impl::DwcConvertArithToLlvmPassBase<DwcConvertArithToLlvmPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "convert-cf-to-llvm" at 0xdcb838.
struct DwcConvertCfToLlvmPass : public darwinn::impl::DwcConvertCfToLlvmPassBase<DwcConvertCfToLlvmPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "convert-conv1x1-to-fc" at 0xe26ea5.
struct DwcConvertConv1x1ToFcPass : public darwinn::impl::DwcConvertConv1x1ToFcPassBase<DwcConvertConv1x1ToFcPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "convert-dive-vm-tensor-to-linalg" at 0xde1ad1.
struct DwcConvertDiveVmTensorToLinalgPass : public darwinn::impl::DwcConvertDiveVmTensorToLinalgPassBase<DwcConvertDiveVmTensorToLinalgPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "convert-dive-vm-tensor-to-scf" at 0xde4651.
struct DwcConvertDiveVmTensorToScfPass : public darwinn::impl::DwcConvertDiveVmTensorToScfPassBase<DwcConvertDiveVmTensorToScfPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "convert-dive-vm-tensor-to-tensor" at 0xda89fa.
struct DwcConvertDiveVmTensorToTensorPass : public darwinn::impl::DwcConvertDiveVmTensorToTensorPassBase<DwcConvertDiveVmTensorToTensorPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "convert-dive-vm-to-llvm" at 0xdcb7dd.
struct DwcConvertDiveVmToLlvmPass : public darwinn::impl::DwcConvertDiveVmToLlvmPassBase<DwcConvertDiveVmToLlvmPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "convert-dive-vm-to-memref" at 0xde3efd.
struct DwcConvertDiveVmToMemrefPass : public darwinn::impl::DwcConvertDiveVmToMemrefPassBase<DwcConvertDiveVmToMemrefPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "convert-dwc-to-dive-vm-tensor" at 0xda8a3d.
struct DwcConvertDwcToDiveVmTensorPass : public darwinn::impl::DwcConvertDwcToDiveVmTensorPassBase<DwcConvertDwcToDiveVmTensorPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "convert-dwg-to-dive-vm" at 0xdcb89c.
struct DwcConvertDwgToDiveVmPass : public darwinn::impl::DwcConvertDwgToDiveVmPassBase<DwcConvertDwgToDiveVmPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "convert-dynamic-shape-scope-to-dive-vm" at 0xdcb8b3.
struct DwcConvertDynamicShapeScopeToDiveVmPass : public darwinn::impl::DwcConvertDynamicShapeScopeToDiveVmPassBase<DwcConvertDynamicShapeScopeToDiveVmPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "convert-func-to-llvm" at 0xdcb867.
struct DwcConvertFuncToLlvmPass : public darwinn::impl::DwcConvertFuncToLlvmPassBase<DwcConvertFuncToLlvmPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "convert-generic-norm-to-pseudo-op" at 0xdb45be.
struct DwcConvertGenericNormToPseudoOpPass : public darwinn::impl::DwcConvertGenericNormToPseudoOpPassBase<DwcConvertGenericNormToPseudoOpPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "convert-linalg-to-loops" at 0xd83df7.
struct DwcConvertLinalgToLoopsPass : public darwinn::impl::DwcConvertLinalgToLoopsPassBase<DwcConvertLinalgToLoopsPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "convert-math-to-libm" at 0xdce410.
struct DwcConvertMathToLibmPass : public darwinn::impl::DwcConvertMathToLibmPassBase<DwcConvertMathToLibmPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "convert-math-to-llvm" at 0xdcb80b.
struct DwcConvertMathToLlvmPass : public darwinn::impl::DwcConvertMathToLlvmPassBase<DwcConvertMathToLlvmPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "convert-op-lowering" at 0xddedce.
struct DwcConvertOpLoweringPass : public darwinn::impl::DwcConvertOpLoweringPassBase<DwcConvertOpLoweringPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "convert-pdl-to-pdl-interp" at 0xdb3800.
struct DwcConvertPdlToPdlInterpPass : public darwinn::impl::DwcConvertPdlToPdlInterpPassBase<DwcConvertPdlToPdlInterpPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "convert-scatter-to-generic-scatter" at 0xdabcd2.
struct DwcConvertScatterToGenericScatterPass : public darwinn::impl::DwcConvertScatterToGenericScatterPassBase<DwcConvertScatterToGenericScatterPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "convert-scf-to-cf" at 0xde4728.
struct DwcConvertScfToCfPass : public darwinn::impl::DwcConvertScfToCfPassBase<DwcConvertScfToCfPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "convert-signed-int-with-rescaling-ops" at 0xd84723.
struct DwcConvertSignedIntWithRescalingOpsPass : public darwinn::impl::DwcConvertSignedIntWithRescalingOpsPassBase<DwcConvertSignedIntWithRescalingOpsPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "convert-spatial-reduction-to-pooling" at 0xddfd8d.
struct DwcConvertSpatialReductionToPoolingPass : public darwinn::impl::DwcConvertSpatialReductionToPoolingPassBase<DwcConvertSpatialReductionToPoolingPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "convert-tf-to-dwc" at 0xe2523e.
struct DwcConvertTfToDwcPass : public darwinn::impl::DwcConvertTfToDwcPassBase<DwcConvertTfToDwcPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "convert-to-k-in-m-sparsity" at 0xd5a046.
struct DwcConvertToKInMSparsityPass : public darwinn::impl::DwcConvertToKInMSparsityPassBase<DwcConvertToKInMSparsityPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "convert-tpu-offload-to-dive-vm" at 0xdcb8da.
struct DwcConvertTpuOffloadToDiveVmPass : public darwinn::impl::DwcConvertTpuOffloadToDiveVmPassBase<DwcConvertTpuOffloadToDiveVmPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "convert-tpu-offload-to-llvm" at 0xdcb84b.
struct DwcConvertTpuOffloadToLlvmPass : public darwinn::impl::DwcConvertTpuOffloadToLlvmPassBase<DwcConvertTpuOffloadToLlvmPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "convert-xla-supported-stablehlo" at 0xdbc2b1.
struct DwcConvertXlaSupportedStablehloPass : public darwinn::impl::DwcConvertXlaSupportedStablehloPassBase<DwcConvertXlaSupportedStablehloPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "ConvertDiveVmTensorToLinalg" at 0xde1af2.
struct DwcConvertDiveVmTensorToLinalgSymbolPass : public darwinn::impl::DwcConvertDiveVmTensorToLinalgSymbolPassBase<DwcConvertDiveVmTensorToLinalgSymbolPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "ConvertTpuOffloadToLlvm" at 0xdcb87c.
struct DwcConvertTpuOffloadToLlvmSymbolPass : public darwinn::impl::DwcConvertTpuOffloadToLlvmSymbolPassBase<DwcConvertTpuOffloadToLlvmSymbolPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "copy-op-lowering" at 0xddedbd.
struct DwcCopyOpLoweringPass : public darwinn::impl::DwcCopyOpLoweringPassBase<DwcCopyOpLoweringPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "darwinn-bundling" at 0xde00af.
struct DwcDarwinnBundlingPass : public darwinn::impl::DwcDarwinnBundlingPassBase<DwcDarwinnBundlingPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "darwinn.convert" at 0xd68ca7.
struct DwcDarwinnConvertPass : public darwinn::impl::DwcDarwinnConvertPassBase<DwcDarwinnConvertPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "darwinn.math.join" at 0xdc9452.
struct DwcDarwinnMathJoinPass : public darwinn::impl::DwcDarwinnMathJoinPassBase<DwcDarwinnMathJoinPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "darwinn.sparsity" at 0xd5a035.
struct DwcDarwinnSparsityPass : public darwinn::impl::DwcDarwinnSparsityPassBase<DwcDarwinnSparsityPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "dive-dce" at 0xe0fe97.
struct DwcDiveDcePass : public darwinn::impl::DwcDiveDcePassBase<DwcDiveDcePass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "dive-io-optimization" at 0xdc141b.
struct DwcDiveIoOptimizationPass : public darwinn::impl::DwcDiveIoOptimizationPassBase<DwcDiveIoOptimizationPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "dive-program-tpu" at 0xd6383a.
struct DwcDiveProgramTpuPass : public darwinn::impl::DwcDiveProgramTpuPassBase<DwcDiveProgramTpuPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "dive-unroll-factor" at 0xda744d.
struct DwcDiveUnrollFactorPass : public darwinn::impl::DwcDiveUnrollFactorPassBase<DwcDiveUnrollFactorPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "dive-vm-bufferize" at 0xde6a70.
struct DwcDiveVmBufferizePass : public darwinn::impl::DwcDiveVmBufferizePassBase<DwcDiveVmBufferizePass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "dive-vm-outline-shareable-dive-consts" at 0xd788a7.
struct DwcDiveVmOutlineShareableDiveConstsPass : public darwinn::impl::DwcDiveVmOutlineShareableDiveConstsPassBase<DwcDiveVmOutlineShareableDiveConstsPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "dwc-check-illegal-tpu-ops" at 0xd84494.
struct DwcDwcCheckIllegalTpuOpsPass : public darwinn::impl::DwcDwcCheckIllegalTpuOpsPassBase<DwcDwcCheckIllegalTpuOpsPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "dwc-convert-input-output-types" at 0xd94dcc.
struct DwcDwcConvertInputOutputTypesPass : public darwinn::impl::DwcDwcConvertInputOutputTypesPassBase<DwcDwcConvertInputOutputTypesPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "dwc-copy-strided-buffers-on-tpu" at 0xd6381a.
struct DwcDwcCopyStridedBuffersOnTpuPass : public darwinn::impl::DwcDwcCopyStridedBuffersOnTpuPassBase<DwcDwcCopyStridedBuffersOnTpuPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "dwc-form-tpu-clusters" at 0xd81a1f.
struct DwcDwcFormTpuClustersPass : public darwinn::impl::DwcDwcFormTpuClustersPassBase<DwcDwcFormTpuClustersPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "dwc-legalize" at 0xde6d3f.
struct DwcDwcLegalizePass : public darwinn::impl::DwcDwcLegalizePassBase<DwcDwcLegalizePass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "dwc-legalize-hlo" at 0xdbc39d.
struct DwcDwcLegalizeHloPass : public darwinn::impl::DwcDwcLegalizeHloPassBase<DwcDwcLegalizeHloPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "dwc-legalize-hlo-to-tf" at 0xde2ad1.
struct DwcDwcLegalizeHloToTfPass : public darwinn::impl::DwcDwcLegalizeHloToTfPassBase<DwcDwcLegalizeHloToTfPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "dwc-legalize-int-and-quant-types" at 0xd94e00.
struct DwcDwcLegalizeIntAndQuantTypesPass : public darwinn::impl::DwcDwcLegalizeIntAndQuantTypesPassBase<DwcDwcLegalizeIntAndQuantTypesPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "dwc-legalize-int64-constants" at 0xd7bb5d.
struct DwcDwcLegalizeInt64ConstantsPass : public darwinn::impl::DwcDwcLegalizeInt64ConstantsPassBase<DwcDwcLegalizeInt64ConstantsPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "dwc-legalize-pass" at 0xd7f2b9.
struct DwcDwcLegalizePassPass : public darwinn::impl::DwcDwcLegalizePassPassBase<DwcDwcLegalizePassPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "dwc-legalize-stablehlo-annotate-materialize-policy" at 0xd5dd77.
struct DwcDwcLegalizeStablehloAnnotateMaterializePolicyPass : public darwinn::impl::DwcDwcLegalizeStablehloAnnotateMaterializePolicyPassBase<DwcDwcLegalizeStablehloAnnotateMaterializePolicyPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "dwc-legalize-stablehlo-composite" at 0xdef599.
struct DwcDwcLegalizeStablehloCompositePass : public darwinn::impl::DwcDwcLegalizeStablehloCompositePassBase<DwcDwcLegalizeStablehloCompositePass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "dwc-legalize-tf-pipeline" at 0xe03284.
struct DwcDwcLegalizeTfPipelinePass : public darwinn::impl::DwcDwcLegalizeTfPipelinePassBase<DwcDwcLegalizeTfPipelinePass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "dwc-legalize-tfl-cudaemu-custom-ops" at 0xd8464e.
struct DwcDwcLegalizeTflCudaemuCustomOpsPass : public darwinn::impl::DwcDwcLegalizeTflCudaemuCustomOpsPassBase<DwcDwcLegalizeTflCudaemuCustomOpsPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "dwc-legalize-tfl-multinomial" at 0xdd3cdb.
struct DwcDwcLegalizeTflMultinomialPass : public darwinn::impl::DwcDwcLegalizeTflMultinomialPassBase<DwcDwcLegalizeTflMultinomialPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "dwc-legalize-tfl-variable-tensors" at 0xd81097.
struct DwcDwcLegalizeTflVariableTensorsPass : public darwinn::impl::DwcDwcLegalizeTflVariableTensorsPassBase<DwcDwcLegalizeTflVariableTensorsPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "dwc-legalize-uint32-types" at 0xd94e3e.
struct DwcDwcLegalizeUint32TypesPass : public darwinn::impl::DwcDwcLegalizeUint32TypesPassBase<DwcDwcLegalizeUint32TypesPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "dwc-lower-argmax-index-unpool" at 0xdcf030.
struct DwcDwcLowerArgmaxIndexUnpoolPass : public darwinn::impl::DwcDwcLowerArgmaxIndexUnpoolPassBase<DwcDwcLowerArgmaxIndexUnpoolPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "dwc-lower-composite-ops" at 0xd84775.
struct DwcDwcLowerCompositeOpsPass : public darwinn::impl::DwcDwcLowerCompositeOpsPassBase<DwcDwcLowerCompositeOpsPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "dwc-lower-control-flow" at 0xd61f2c.
struct DwcDwcLowerControlFlowPass : public darwinn::impl::DwcDwcLowerControlFlowPassBase<DwcDwcLowerControlFlowPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "dwc-lower-depth-to-from-space" at 0xe102d0.
struct DwcDwcLowerDepthToFromSpacePass : public darwinn::impl::DwcDwcLowerDepthToFromSpacePassBase<DwcDwcLowerDepthToFromSpacePass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "dwc-lower-generic-constants" at 0xd7bb41.
struct DwcDwcLowerGenericConstantsPass : public darwinn::impl::DwcDwcLowerGenericConstantsPassBase<DwcDwcLowerGenericConstantsPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "dwc-lower-hlops" at 0xd841f4.
struct DwcDwcLowerHlopsPass : public darwinn::impl::DwcDwcLowerHlopsPassBase<DwcDwcLowerHlopsPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "dwc-lower-input-output-cast" at 0xd68084.
struct DwcDwcLowerInputOutputCastPass : public darwinn::impl::DwcDwcLowerInputOutputCastPassBase<DwcDwcLowerInputOutputCastPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "dwc-lower-padding-ops" at 0xd84749.
struct DwcDwcLowerPaddingOpsPass : public darwinn::impl::DwcDwcLowerPaddingOpsPassBase<DwcDwcLowerPaddingOpsPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "dwc-lower-pseudo-ops" at 0xd845c5.
struct DwcDwcLowerPseudoOpsPass : public darwinn::impl::DwcDwcLowerPseudoOpsPassBase<DwcDwcLowerPseudoOpsPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "dwc-lower-resampler-ops" at 0xd84598.
struct DwcDwcLowerResamplerOpsPass : public darwinn::impl::DwcDwcLowerResamplerOpsPassBase<DwcDwcLowerResamplerOpsPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "dwc-lower-scalar-ops" at 0xd845b0.
struct DwcDwcLowerScalarOpsPass : public darwinn::impl::DwcDwcLowerScalarOpsPassBase<DwcDwcLowerScalarOpsPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "dwc-lower-scatter-ops" at 0xd84582.
struct DwcDwcLowerScatterOpsPass : public darwinn::impl::DwcDwcLowerScatterOpsPassBase<DwcDwcLowerScatterOpsPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "dwc-lower-top-k" at 0xdd6da3.
struct DwcDwcLowerTopKPass : public darwinn::impl::DwcDwcLowerTopKPassBase<DwcDwcLowerTopKPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "dwc-post-truncation-tpu-fitter" at 0xdab7fc.
struct DwcDwcPostTruncationTpuFitterPass : public darwinn::impl::DwcDwcPostTruncationTpuFitterPassBase<DwcDwcPostTruncationTpuFitterPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "dwc-pre-tpu-fitter-optimize-gather" at 0xdaf5f9.
struct DwcDwcPreTpuFitterOptimizeGatherPass : public darwinn::impl::DwcDwcPreTpuFitterOptimizeGatherPassBase<DwcDwcPreTpuFitterOptimizeGatherPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "dwc-pre-tpu-fitter-optimize-scatter" at 0xdabc37.
struct DwcDwcPreTpuFitterOptimizeScatterPass : public darwinn::impl::DwcDwcPreTpuFitterOptimizeScatterPassBase<DwcDwcPreTpuFitterOptimizeScatterPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "dwc-regroup-tpu-functions" at 0xd873c3.
struct DwcDwcRegroupTpuFunctionsPass : public darwinn::impl::DwcDwcRegroupTpuFunctionsPassBase<DwcDwcRegroupTpuFunctionsPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "dwc-serialize-tpu-offloads" at 0xd9c33f.
struct DwcDwcSerializeTpuOffloadsPass : public darwinn::impl::DwcDwcSerializeTpuOffloadsPassBase<DwcDwcSerializeTpuOffloadsPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "dwc-test-repeat-tpu-ops" at 0xd8447c.
struct DwcDwcTestRepeatTpuOpsPass : public darwinn::impl::DwcDwcTestRepeatTpuOpsPassBase<DwcDwcTestRepeatTpuOpsPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "dwc-tpu-fitter" at 0xdab81b.
struct DwcDwcTpuFitterPass : public darwinn::impl::DwcDwcTpuFitterPassBase<DwcDwcTpuFitterPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "dwc-tpu-function-cse" at 0xdf32f5.
struct DwcDwcTpuFunctionCsePass : public darwinn::impl::DwcDwcTpuFunctionCsePassBase<DwcDwcTpuFunctionCsePass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "dwg-create-darwinn-custom-op" at 0xdb45e0.
struct DwcDwgCreateDarwinnCustomOpPass : public darwinn::impl::DwcDwgCreateDarwinnCustomOpPassBase<DwcDwgCreateDarwinnCustomOpPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "dwg-fork-multicore-tpu-offloads" at 0xd9c35a.
struct DwcDwgForkMulticoreTpuOffloadsPass : public darwinn::impl::DwcDwgForkMulticoreTpuOffloadsPassBase<DwcDwgForkMulticoreTpuOffloadsPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "dwg-lower-for-to-while" at 0xe07f47.
struct DwcDwgLowerForToWhilePass : public darwinn::impl::DwcDwgLowerForToWhilePassBase<DwcDwgLowerForToWhilePass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "dwgt-lower-index-type" at 0xdf6914.
struct DwcDwgtLowerIndexTypePass : public darwinn::impl::DwcDwgtLowerIndexTypePassBase<DwcDwgtLowerIndexTypePass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "dynamic-update-slice-lowering" at 0xddee37.
struct DwcDynamicUpdateSliceLoweringPass : public darwinn::impl::DwcDynamicUpdateSliceLoweringPassBase<DwcDynamicUpdateSliceLoweringPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "edgetpu-custom-op-2" at 0x103fe70.
struct DwcEdgetpuCustomOp2Pass : public darwinn::impl::DwcEdgetpuCustomOp2PassBase<DwcEdgetpuCustomOp2Pass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "(fm-model-converter" at 0xdac265.
struct DwcFmModelConverterPass : public darwinn::impl::DwcFmModelConverterPassBase<DwcFmModelConverterPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "(fpa2bv-model-converter" at 0xdac220.
struct DwcFpa2bvModelConverterPass : public darwinn::impl::DwcFpa2bvModelConverterPassBase<DwcFpa2bvModelConverterPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "group-tpu-offloads-by-parameters" at 0xd821ce.
struct DwcGroupTpuOffloadsByParametersPass : public darwinn::impl::DwcGroupTpuOffloadsByParametersPassBase<DwcGroupTpuOffloadsByParametersPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "interpolate-lowering-pass" at 0xd7f143.
struct DwcInterpolateLoweringPassPass : public darwinn::impl::DwcInterpolateLoweringPassPassBase<DwcInterpolateLoweringPassPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "is not immutable, try removing mutable variables in your model since mutable variables are currently not supported through this converter" at 0xdac290.
struct DwcIsNotImmutableTryRemovingMutableVariablesInYourModelSinceMutableVariablesAreCurrentlyNotSupportedThroughThisConverterPass : public darwinn::impl::DwcIsNotImmutableTryRemovingMutableVariablesInYourModelSinceMutableVariablesAreCurrentlyNotSupportedThroughThisConverterPassBase<DwcIsNotImmutableTryRemovingMutableVariablesInYourModelSinceMutableVariablesAreCurrentlyNotSupportedThroughThisConverterPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "Legalize" at 0xde6d4c.
struct DwcLegalizePass : public darwinn::impl::DwcLegalizePassBase<DwcLegalizePass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "legalize-affine" at 0xe03694.
struct DwcLegalizeAffinePass : public darwinn::impl::DwcLegalizeAffinePassBase<DwcLegalizeAffinePass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "legalize-dwc" at 0xe25250.
struct DwcLegalizeDwcPass : public darwinn::impl::DwcLegalizeDwcPassBase<DwcLegalizeDwcPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "legalize-dwc-input-output-ops" at 0xd844c7.
struct DwcLegalizeDwcInputOutputOpsPass : public darwinn::impl::DwcLegalizeDwcInputOutputOpsPassBase<DwcLegalizeDwcInputOutputOpsPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "legalize-dwg-tensor" at 0xda8a5b.
struct DwcLegalizeDwgTensorPass : public darwinn::impl::DwcLegalizeDwgTensorPassBase<DwcLegalizeDwgTensorPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "legalize-quant-types" at 0xd94deb.
struct DwcLegalizeQuantTypesPass : public darwinn::impl::DwcLegalizeQuantTypesPassBase<DwcLegalizeQuantTypesPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "legalize-scf" at 0xde466f.
struct DwcLegalizeScfPass : public darwinn::impl::DwcLegalizeScfPassBase<DwcLegalizeScfPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "legalize-shape-ops" at 0xd84833.
struct DwcLegalizeShapeOpsPass : public darwinn::impl::DwcLegalizeShapeOpsPassBase<DwcLegalizeShapeOpsPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "legalize test using layerir flow" at 0xd61f43.
struct DwcLegalizeTestUsingLayerirFlowPass : public darwinn::impl::DwcLegalizeTestUsingLayerirFlowPassBase<DwcLegalizeTestUsingLayerirFlowPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "Legalize TF_XlaCallModule Op to stablehlo" at 0xdbc2d1.
struct DwcLegalizeTfXlacallmoduleOpToStablehloPass : public darwinn::impl::DwcLegalizeTfXlacallmoduleOpToStablehloPassBase<DwcLegalizeTfXlacallmoduleOpToStablehloPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "legalize-thread-oblivious-op-pass" at 0xd7ee8c.
struct DwcLegalizeThreadObliviousOpPassPass : public darwinn::impl::DwcLegalizeThreadObliviousOpPassPassBase<DwcLegalizeThreadObliviousOpPassPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "legalize-types-for-dive-vm-tensor" at 0xda8a1b.
struct DwcLegalizeTypesForDiveVmTensorPass : public darwinn::impl::DwcLegalizeTypesForDiveVmTensorPassBase<DwcLegalizeTypesForDiveVmTensorPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "LegalizeStablehloComposite" at 0xdef61e.
struct DwcLegalizeStablehloCompositePass : public darwinn::impl::DwcLegalizeStablehloCompositePassBase<DwcLegalizeStablehloCompositePass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "lower-affine" at 0xe03675.
struct DwcLowerAffinePass : public darwinn::impl::DwcLowerAffinePassBase<DwcLowerAffinePass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "lower-all-functions" at 0xd873dd.
struct DwcLowerAllFunctionsPass : public darwinn::impl::DwcLowerAllFunctionsPassBase<DwcLowerAllFunctionsPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "lower-all-pads" at 0xd9c263.
struct DwcLowerAllPadsPass : public darwinn::impl::DwcLowerAllPadsPassBase<DwcLowerAllPadsPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "lower-attention-ops" at 0xd845ec.
struct DwcLowerAttentionOpsPass : public darwinn::impl::DwcLowerAttentionOpsPassBase<DwcLowerAttentionOpsPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "lower-input-cast" at 0xd680b2.
struct DwcLowerInputCastPass : public darwinn::impl::DwcLowerInputCastPassBase<DwcLowerInputCastPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "lower-join" at 0xdc9464.
struct DwcLowerJoinPass : public darwinn::impl::DwcLowerJoinPassBase<DwcLowerJoinPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "lower-output-cast" at 0xd680a0.
struct DwcLowerOutputCastPass : public darwinn::impl::DwcLowerOutputCastPassBase<DwcLowerOutputCastPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "LowerArgmaxIndexUnpool" at 0xdcf04e.
struct DwcLowerArgmaxIndexUnpoolPass : public darwinn::impl::DwcLowerArgmaxIndexUnpoolPassBase<DwcLowerArgmaxIndexUnpoolPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "mark-dive-vm-tensor-insert-slice-ops" at 0xd8488d.
struct DwcMarkDiveVmTensorInsertSliceOpsPass : public darwinn::impl::DwcMarkDiveVmTensorInsertSliceOpsPassBase<DwcMarkDiveVmTensorInsertSliceOpsPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "mhlo-legalize-einsum-to-dot-general" at 0xdd34bf.
struct DwcMhloLegalizeEinsumToDotGeneralPass : public darwinn::impl::DwcMhloLegalizeEinsumToDotGeneralPassBase<DwcMhloLegalizeEinsumToDotGeneralPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "mid-to-low-level-lowering" at 0xddedf4.
struct DwcMidToLowLevelLoweringPass : public darwinn::impl::DwcMidToLowLevelLoweringPassBase<DwcMidToLowLevelLoweringPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "::mlir::darwinn::compute::Engine" at 0xe03654.
struct DwcMlirDarwinnComputeEnginePass : public darwinn::impl::DwcMlirDarwinnComputeEnginePassBase<DwcMlirDarwinnComputeEnginePass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "Only DenseElementsAttr are supported for constant lowering" at 0xddee55.
struct DwcOnlyDenseelementsattrAreSupportedForConstantLoweringPass : public darwinn::impl::DwcOnlyDenseelementsattrAreSupportedForConstantLoweringPassBase<DwcOnlyDenseelementsattrAreSupportedForConstantLoweringPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "optimize-dive-vm-tensor-insert-slice" at 0xe0fa93.
struct DwcOptimizeDiveVmTensorInsertSlicePass : public darwinn::impl::DwcOptimizeDiveVmTensorInsertSlicePassBase<DwcOptimizeDiveVmTensorInsertSlicePass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "-parameter-caching-dive-program" at 0xdce5e1.
struct DwcParameterCachingDiveProgramPass : public darwinn::impl::DwcParameterCachingDiveProgramPassBase<DwcParameterCachingDiveProgramPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "platforms.darwinn.code_generator.Entry.Score.type" at 0xdf68e2.
struct DwcPlatformsDarwinnCodeGeneratorEntryScoreTypePass : public darwinn::impl::DwcPlatformsDarwinnCodeGeneratorEntryScoreTypePassBase<DwcPlatformsDarwinnCodeGeneratorEntryScoreTypePass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "platforms.darwinn.compiler.ProbeInstrumentationLocation.Constraints.functions" at 0xd87375.
struct DwcPlatformsDarwinnCompilerProbeinstrumentationlocationConstraintsFunctionsPass : public darwinn::impl::DwcPlatformsDarwinnCompilerProbeinstrumentationlocationConstraintsFunctionsPassBase<DwcPlatformsDarwinnCompilerProbeinstrumentationlocationConstraintsFunctionsPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "quant-signedness-convert-lowering" at 0xdded43.
struct DwcQuantSignednessConvertLoweringPass : public darwinn::impl::DwcQuantSignednessConvertLoweringPassBase<DwcQuantSignednessConvertLoweringPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "r52-reads-dive-buffers" at 0xd833a6.
struct DwcR52ReadsDiveBuffersPass : public darwinn::impl::DwcR52ReadsDiveBuffersPassBase<DwcR52ReadsDiveBuffersPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "redistribute-lowering" at 0xddee0e.
struct DwcRedistributeLoweringPass : public darwinn::impl::DwcRedistributeLoweringPassBase<DwcRedistributeLoweringPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "redistribute-lowering-pass-remarks" at 0xd8c505.
struct DwcRedistributeLoweringPassRemarksPass : public darwinn::impl::DwcRedistributeLoweringPassRemarksPassBase<DwcRedistributeLoweringPassRemarksPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "reinterpret-cast-rank-legalize-pass" at 0xd7f295.
struct DwcReinterpretCastRankLegalizePassPass : public darwinn::impl::DwcReinterpretCastRankLegalizePassPassBase<DwcReinterpretCastRankLegalizePassPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "rename-dive-entry-function" at 0xdbfc27.
struct DwcRenameDiveEntryFunctionPass : public darwinn::impl::DwcRenameDiveEntryFunctionPassBase<DwcRenameDiveEntryFunctionPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "resampler-lowering" at 0xdded92.
struct DwcResamplerLoweringPass : public darwinn::impl::DwcResamplerLoweringPassBase<DwcResamplerLoweringPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "rkhy-shape-legalization-pass" at 0xd7f029.
struct DwcRkhyShapeLegalizationPassPass : public darwinn::impl::DwcRkhyShapeLegalizationPassPassBase<DwcRkhyShapeLegalizationPassPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "rkhy-type-legalization-pass" at 0xd7f00d.
struct DwcRkhyTypeLegalizationPassPass : public darwinn::impl::DwcRkhyTypeLegalizationPassPassBase<DwcRkhyTypeLegalizationPassPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "run-r52-ops-on-dive" at 0xde9621.
struct DwcRunR52OpsOnDivePass : public darwinn::impl::DwcRunR52OpsOnDivePassBase<DwcRunR52OpsOnDivePass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "scalar-core-control-flow-lowering" at 0xdded21.
struct DwcScalarCoreControlFlowLoweringPass : public darwinn::impl::DwcScalarCoreControlFlowLoweringPassBase<DwcScalarCoreControlFlowLoweringPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "scalar-core-std-ops-lowering" at 0xdded75.
struct DwcScalarCoreStdOpsLoweringPass : public darwinn::impl::DwcScalarCoreStdOpsLoweringPassBase<DwcScalarCoreStdOpsLoweringPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "scalar-ops-legalize" at 0xde6d2b.
struct DwcScalarOpsLegalizePass : public darwinn::impl::DwcScalarOpsLegalizePassBase<DwcScalarOpsLegalizePass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "scatter-gather-lowering" at 0xddeda5.
struct DwcScatterGatherLoweringPass : public darwinn::impl::DwcScatterGatherLoweringPassBase<DwcScatterGatherLoweringPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "select-lowering" at 0xdded65.
struct DwcSelectLoweringPass : public darwinn::impl::DwcSelectLoweringPassBase<DwcSelectLoweringPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "sharding-using-dive" at 0xde9635.
struct DwcShardingUsingDivePass : public darwinn::impl::DwcShardingUsingDivePassBase<DwcShardingUsingDivePass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "skipping fold of float convert" at 0xd68cda.
struct DwcSkippingFoldOfFloatConvertPass : public darwinn::impl::DwcSkippingFoldOfFloatConvertPassBase<DwcSkippingFoldOfFloatConvertPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "split-op-lowering" at 0xddede2.
struct DwcSplitOpLoweringPass : public darwinn::impl::DwcSplitOpLoweringPassBase<DwcSplitOpLoweringPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "stablehlo-composite-legalize-tfl-custom" at 0xdccdf7.
struct DwcStablehloCompositeLegalizeTflCustomPass : public darwinn::impl::DwcStablehloCompositeLegalizeTflCustomPassBase<DwcStablehloCompositeLegalizeTflCustomPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "stablehlo-custom-call-legalize-composite" at 0xdef5f5.
struct DwcStablehloCustomCallLegalizeCompositePass : public darwinn::impl::DwcStablehloCustomCallLegalizeCompositePassBase<DwcStablehloCustomCallLegalizeCompositePass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "stablehlo-legalize-composite-to-call" at 0xdd00c2.
struct DwcStablehloLegalizeCompositeToCallPass : public darwinn::impl::DwcStablehloLegalizeCompositeToCallPassBase<DwcStablehloLegalizeCompositeToCallPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "stablehlo-legalize-to-hlo" at 0xdbc34d.
struct DwcStablehloLegalizeToHloPass : public darwinn::impl::DwcStablehloLegalizeToHloPassBase<DwcStablehloLegalizeToHloPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "stablehlo-legalize-to-vhlo" at 0xdbc236.
struct DwcStablehloLegalizeToVhloPass : public darwinn::impl::DwcStablehloLegalizeToVhloPassBase<DwcStablehloLegalizeToVhloPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "stablehlo-legalize-vhlo" at 0xdbc251.
struct DwcStablehloLegalizeVhloPass : public darwinn::impl::DwcStablehloLegalizeVhloPassBase<DwcStablehloLegalizeVhloPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "stochastic-convert" at 0xd68cc7.
struct DwcStochasticConvertPass : public darwinn::impl::DwcStochasticConvertPassBase<DwcStochasticConvertPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "tf-legalize-hlo" at 0xdbc38d.
struct DwcTfLegalizeHloPass : public darwinn::impl::DwcTfLegalizeHloPassBase<DwcTfLegalizeHloPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "tfl-custom-lowering-rewriting-pass" at 0xd7f120.
struct DwcTflCustomLoweringRewritingPassPass : public darwinn::impl::DwcTflCustomLoweringRewritingPassPassBase<DwcTflCustomLoweringRewritingPassPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "tfl-legalize-chlo" at 0xdbc2fb.
struct DwcTflLegalizeChloPass : public darwinn::impl::DwcTflLegalizeChloPassBase<DwcTflLegalizeChloPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "tfl-legalize-hashtables-tf" at 0xde2a7d.
struct DwcTflLegalizeHashtablesTfPass : public darwinn::impl::DwcTflLegalizeHashtablesTfPassBase<DwcTflLegalizeHashtablesTfPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "tfl-legalize-hlo" at 0xdbc37c.
struct DwcTflLegalizeHloPass : public darwinn::impl::DwcTflLegalizeHloPassBase<DwcTflLegalizeHloPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "tfl-legalize-tensorlist" at 0xd66a8f.
struct DwcTflLegalizeTensorlistPass : public darwinn::impl::DwcTflLegalizeTensorlistPassBase<DwcTflLegalizeTensorlistPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "tfl-legalize-tf" at 0xde2ae8.
struct DwcTflLegalizeTfPass : public darwinn::impl::DwcTflLegalizeTfPassBase<DwcTflLegalizeTfPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "tfl-legalize-tf-while" at 0xe07f5e.
struct DwcTflLegalizeTfWhilePass : public darwinn::impl::DwcTflLegalizeTfWhilePassBase<DwcTflLegalizeTfWhilePass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "tfl-legalize-variables-tf" at 0xde2a98.
struct DwcTflLegalizeVariablesTfPass : public darwinn::impl::DwcTflLegalizeVariablesTfPassBase<DwcTflLegalizeVariablesTfPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "tfl-lower-quant-annotations" at 0xd87c99.
struct DwcTflLowerQuantAnnotationsPass : public darwinn::impl::DwcTflLowerQuantAnnotationsPassBase<DwcTflLowerQuantAnnotationsPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "tfl-lower-static-tensor-list" at 0xd66ca5.
struct DwcTflLowerStaticTensorListPass : public darwinn::impl::DwcTflLowerStaticTensorListPassBase<DwcTflLowerStaticTensorListPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "top-k-lowering-policy" at 0xd5dd44.
struct DwcTopKLoweringPolicyPass : public darwinn::impl::DwcTopKLoweringPolicyPassBase<DwcTopKLoweringPolicyPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "tpu-clustering-algorithm" at 0xdcdd6e.
struct DwcTpuClusteringAlgorithmPass : public darwinn::impl::DwcTpuClusteringAlgorithmPassBase<DwcTpuClusteringAlgorithmPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "vhlo-legalize-stablehlo" at 0xdbc299.
struct DwcVhloLegalizeStablehloPass : public darwinn::impl::DwcVhloLegalizeStablehloPassBase<DwcVhloLegalizeStablehloPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "vhlo-legalize-to-stablehlo" at 0xdbc271.
struct DwcVhloLegalizeToStablehloPass : public darwinn::impl::DwcVhloLegalizeToStablehloPassBase<DwcVhloLegalizeToStablehloPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "wrap-up-dive-program" at 0xdce5cc.
struct DwcWrapUpDiveProgramPass : public darwinn::impl::DwcWrapUpDiveProgramPassBase<DwcWrapUpDiveProgramPass> {
  using Base::Base;

  void runOnOperation() override {}
};

// TSV row: "xla_cpu_use_new_xtile_lowering" at 0xdded02.
struct DwcXlaCpuUseNewXtileLoweringPass : public darwinn::impl::DwcXlaCpuUseNewXtileLoweringPassBase<DwcXlaCpuUseNewXtileLoweringPass> {
  using Base::Base;

  void runOnOperation() override {}
};

} // namespace

namespace mlir {
namespace darwinn {
#define GEN_PASS_REGISTRATION
#include "DwcPasses.h.inc"
} // namespace darwinn
} // namespace mlir
