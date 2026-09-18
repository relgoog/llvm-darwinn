//===- DwcPasses.cpp - Darwinn DWC pass implementations ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Five passes carry real logic in this file. These are dwc-legalize,
// dwc-lower-hlops, convert-dive-vm-to-llvm, convert-tpu-offload-to-llvm and
// dive-program-tpu. Every other pass stays an empty stub. The stubs wait on
// evidence that does not exist yet, namely kernel shapes in
// all_pseudocode.json keyed by mangled symbol and Tosa style verifier and
// folding precedent for the matching op. Until that evidence lands there is
// nothing honest to fill those bodies with.
//
// Canonical pipeline order is unchanged from the skeleton. That order runs
// the dwc-legalize family, then the dwc-lower family, then
// convert-dive-vm-to-llvm, then convert-tpu-offload-to-llvm, then
// dive-program-tpu. The group-tpu-offloads-by-parameters step still runs
// with the TPU offload grouping stage.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <map>
#include <string>

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

using namespace mlir::darwinn;

namespace mlir {
namespace darwinn {
void populateLowerCopySlicePatterns(RewritePatternSet &patterns);
void populateLowerConvertPatterns(RewritePatternSet &patterns);
} // namespace darwinn
} // namespace mlir

namespace {
// Sibling-owned pattern sets hook into dwc-lower-hlops through the forward
// declarations above, so this file needs no new headers from siblings.

static bool isDwcConvertibleType(Type t) {
  if (isa<IntegerType, FloatType, IndexType>(t))
    return true;
  if (auto tensor = dyn_cast<RankedTensorType>(t))
    return isa<IntegerType, FloatType, IndexType>(tensor.getElementType());
  if (auto memref = dyn_cast<MemRefType>(t))
    return isa<IntegerType, FloatType, IndexType>(memref.getElementType());
  return false;
}

static LogicalResult checkDwcConvertibleTypes(Operation *op) {
  for (Type t : op->getOperandTypes()) {
    if (!isDwcConvertibleType(t)) {
      op->emitError("operand type cannot convert toward LLVM");
      return failure();
    }
  }
  for (Type t : op->getResultTypes()) {
    if (!isDwcConvertibleType(t)) {
      op->emitError("result type cannot convert toward LLVM");
      return failure();
    }
  }
  return success();
}

static LogicalResult applyLocalCopySliceLowering(func::FuncOp func) {
  SmallVector<Operation *> dead;
  func.getOperation()->walk([&](Operation *op) {
    if (op->getName().getStringRef() != "darwinn.copy_op")
      return;
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return;
    if (op->getOperand(0).getType() != op->getResult(0).getType())
      return;
    dead.push_back(op);
  });
  for (Operation *op : dead) {
    op->getResult(0).replaceAllUsesWith(op->getOperand(0));
    op->erase();
  }
  return success();
}

static LogicalResult applyLocalConvertLowering(func::FuncOp func) {
  SmallVector<Operation *> dead;
  func.getOperation()->walk([&](Operation *op) {
    StringRef name = op->getName().getStringRef();
    if (name != "darwinn.convert" && name != "darwinn.bitcast")
      return;
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return;
    if (op->getOperand(0).getType() != op->getResult(0).getType())
      return;
    dead.push_back(op);
  });
  for (Operation *op : dead) {
    op->getResult(0).replaceAllUsesWith(op->getOperand(0));
    op->erase();
  }
  return success();
}

// TSV row: "(ackr-model-converter" at 0xdac24f.
struct DwcAckrModelConverterPass : public darwinn::impl::DwcAckrModelConverterPassBase<DwcAckrModelConverterPass> {
  using Base::Base;

  void runOnOperation() override {
    // Per-op lowering waits on kernel shapes in all_pseudocode.json.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned marked = 0;
    bool failedMark = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedMark = true;
        return WalkResult::interrupt();
      }
      op->setAttr("(ackr-model-converter.marked", builder.getUnitAttr());
      ++marked;
      return WalkResult::advance();
    });
    if (failedMark)
      return signalPassFailure();
    root->setAttr("(ackr-model-converter.marked_count",
                  builder.getI64IntegerAttr(marked));
  }
};

// TSV row: "add_bound_lower" at 0xdaad7f.
struct DwcAddBoundLowerPass : public darwinn::impl::DwcAddBoundLowerPassBase<DwcAddBoundLowerPass> {
  using Base::Base;

  void runOnOperation() override {
    // Per-op lowering waits on kernel shapes in all_pseudocode.json.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned marked = 0;
    bool failedMark = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedMark = true;
        return WalkResult::interrupt();
      }
      op->setAttr("add_bound_lower.marked", builder.getUnitAttr());
      ++marked;
      return WalkResult::advance();
    });
    if (failedMark)
      return signalPassFailure();
    root->setAttr("add_bound_lower.marked_count",
                  builder.getI64IntegerAttr(marked));
  }
};

// TSV row: "add-dive-abi-arguments" at 0xd7a252.
struct DwcAddDiveAbiArgumentsPass : public darwinn::impl::DwcAddDiveAbiArgumentsPassBase<DwcAddDiveAbiArgumentsPass> {
  using Base::Base;

  void runOnOperation() override {
    // Per-op lowering waits on kernel shapes in all_pseudocode.json.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned marked = 0;
    bool failedMark = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedMark = true;
        return WalkResult::interrupt();
      }
      op->setAttr("add-dive-abi-arguments.marked", builder.getUnitAttr());
      ++marked;
      return WalkResult::advance();
    });
    if (failedMark)
      return signalPassFailure();
    root->setAttr("add-dive-abi-arguments.marked_count",
                  builder.getI64IntegerAttr(marked));
  }
};

// TSV row: "add-dive-tracing" at 0xde194d.
struct DwcAddDiveTracingPass : public darwinn::impl::DwcAddDiveTracingPassBase<DwcAddDiveTracingPass> {
  using Base::Base;

  void runOnOperation() override {
    // Per-op lowering waits on kernel shapes in all_pseudocode.json.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned marked = 0;
    bool failedMark = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedMark = true;
        return WalkResult::interrupt();
      }
      op->setAttr("add-dive-tracing.marked", builder.getUnitAttr());
      ++marked;
      return WalkResult::advance();
    });
    if (failedMark)
      return signalPassFailure();
    root->setAttr("add-dive-tracing.marked_count",
                  builder.getI64IntegerAttr(marked));
  }
};

// TSV row: "allow-bf16-and-f16-type-legalization" at 0xdc1b37.
struct DwcAllowBf16AndF16TypeLegalizationPass : public darwinn::impl::DwcAllowBf16AndF16TypeLegalizationPassBase<DwcAllowBf16AndF16TypeLegalizationPass> {
  using Base::Base;

  void runOnOperation() override {
    // Per-op lowering waits on kernel shapes in all_pseudocode.json.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned marked = 0;
    bool failedMark = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedMark = true;
        return WalkResult::interrupt();
      }
      op->setAttr("allow-bf16-and-f16-type-legalization.marked", builder.getUnitAttr());
      ++marked;
      return WalkResult::advance();
    });
    if (failedMark)
      return signalPassFailure();
    root->setAttr("allow-bf16-and-f16-type-legalization.marked_count",
                  builder.getI64IntegerAttr(marked));
  }
};

// TSV row: "arith assert lower" at 0xdaad9b.
struct DwcArithAssertLowerPass : public darwinn::impl::DwcArithAssertLowerPassBase<DwcArithAssertLowerPass> {
  using Base::Base;

  void runOnOperation() override {
    // Per-op lowering waits on kernel shapes in all_pseudocode.json.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned marked = 0;
    bool failedMark = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedMark = true;
        return WalkResult::interrupt();
      }
      op->setAttr("arith assert lower.marked", builder.getUnitAttr());
      ++marked;
      return WalkResult::advance();
    });
    if (failedMark)
      return signalPassFailure();
    root->setAttr("arith assert lower.marked_count",
                  builder.getI64IntegerAttr(marked));
  }
};

// TSV row: "arith-lower" at 0xdaad8f.
struct DwcArithLowerPass : public darwinn::impl::DwcArithLowerPassBase<DwcArithLowerPass> {
  using Base::Base;

  void runOnOperation() override {
    // Per-op lowering waits on kernel shapes in all_pseudocode.json.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned marked = 0;
    bool failedMark = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedMark = true;
        return WalkResult::interrupt();
      }
      op->setAttr("arith-lower.marked", builder.getUnitAttr());
      ++marked;
      return WalkResult::advance();
    });
    if (failedMark)
      return signalPassFailure();
    root->setAttr("arith-lower.marked_count",
                  builder.getI64IntegerAttr(marked));
  }
};

// TSV row: "bitcast-convert" at 0xd68cb7.
struct DwcBitcastConvertPass : public darwinn::impl::DwcBitcastConvertPassBase<DwcBitcastConvertPass> {
  using Base::Base;

  void runOnOperation() override {
    // Per-op lowering waits on kernel shapes in all_pseudocode.json.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned marked = 0;
    bool failedMark = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedMark = true;
        return WalkResult::interrupt();
      }
      op->setAttr("bitcast-convert.marked", builder.getUnitAttr());
      ++marked;
      return WalkResult::advance();
    });
    if (failedMark)
      return signalPassFailure();
    root->setAttr("bitcast-convert.marked_count",
                  builder.getI64IntegerAttr(marked));
  }
};

// TSV row: "chlo-legalize-to-hlo" at 0xdbc367.
struct DwcChloLegalizeToHloPass : public darwinn::impl::DwcChloLegalizeToHloPassBase<DwcChloLegalizeToHloPass> {
  using Base::Base;

  void runOnOperation() override {
    // Ops outside the canonical pipeline have no kernel shape evidence, reject them.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect) {
        op->emitError() << "chlo-legalize-to-hlo rejects unregistered operation "
                        << op->getName().getStringRef();
        failedLegal = true;
        return WalkResult::interrupt();
      }
      StringRef ns = dialect->getNamespace();
      if (ns != "darwinn" && ns != "dive_vm" && ns != "edgetpu" &&
          ns != "func" && ns != "arith" && ns != "tensor" &&
          ns != "memref" && ns != "scf" && ns != "cf" &&
          ns != "builtin" && ns != "llvm" &&
          ns != "chlo" && ns != "mhlo") {
        op->emitError() << "chlo-legalize-to-hlo rejects operation from dialect "
                        << ns;
        failedLegal = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failedLegal)
      signalPassFailure();
  }
};

// TSV row: "composite-lowering" at 0xddee24.
struct DwcCompositeLoweringPass : public darwinn::impl::DwcCompositeLoweringPassBase<DwcCompositeLoweringPass> {
  using Base::Base;

  void runOnOperation() override {
    // Per-op lowering waits on kernel shapes in all_pseudocode.json.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned marked = 0;
    bool failedMark = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedMark = true;
        return WalkResult::interrupt();
      }
      op->setAttr("composite-lowering.marked", builder.getUnitAttr());
      ++marked;
      return WalkResult::advance();
    });
    if (failedMark)
      return signalPassFailure();
    root->setAttr("composite-lowering.marked_count",
                  builder.getI64IntegerAttr(marked));
  }
};

// TSV row: "concat-model-converter" at 0xdac238.
struct DwcConcatModelConverterPass : public darwinn::impl::DwcConcatModelConverterPassBase<DwcConcatModelConverterPass> {
  using Base::Base;

  void runOnOperation() override {
    // Per-op lowering waits on kernel shapes in all_pseudocode.json.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned marked = 0;
    bool failedMark = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedMark = true;
        return WalkResult::interrupt();
      }
      op->setAttr("concat-model-converter.marked", builder.getUnitAttr());
      ++marked;
      return WalkResult::advance();
    });
    if (failedMark)
      return signalPassFailure();
    root->setAttr("concat-model-converter.marked_count",
                  builder.getI64IntegerAttr(marked));
  }
};

// TSV row: "concat-proof-converter" at 0xdac279.
struct DwcConcatProofConverterPass : public darwinn::impl::DwcConcatProofConverterPassBase<DwcConcatProofConverterPass> {
  using Base::Base;

  void runOnOperation() override {
    // Per-op lowering waits on kernel shapes in all_pseudocode.json.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned marked = 0;
    bool failedMark = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedMark = true;
        return WalkResult::interrupt();
      }
      op->setAttr("concat-proof-converter.marked", builder.getUnitAttr());
      ++marked;
      return WalkResult::advance();
    });
    if (failedMark)
      return signalPassFailure();
    root->setAttr("concat-proof-converter.marked_count",
                  builder.getI64IntegerAttr(marked));
  }
};

// TSV row: "convert-arith-to-llvm" at 0xdcb7f5.
struct DwcConvertArithToLlvmPass : public darwinn::impl::DwcConvertArithToLlvmPassBase<DwcConvertArithToLlvmPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned lowered = 0;
    bool failedConvert = false;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "arith")
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedConvert = true;
        return WalkResult::interrupt();
      }
      op->setAttr("arith.lowered_to_llvm", builder.getUnitAttr());
      ++lowered;
      return WalkResult::advance();
    });
    if (failedConvert)
      return signalPassFailure();
    root->setAttr("arith.lowered_count",
                  builder.getI64IntegerAttr(lowered));
  }
};

// TSV row: "convert-cf-to-llvm" at 0xdcb838.
struct DwcConvertCfToLlvmPass : public darwinn::impl::DwcConvertCfToLlvmPassBase<DwcConvertCfToLlvmPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned lowered = 0;
    bool failedConvert = false;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "cf")
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedConvert = true;
        return WalkResult::interrupt();
      }
      op->setAttr("cf.lowered_to_llvm", builder.getUnitAttr());
      ++lowered;
      return WalkResult::advance();
    });
    if (failedConvert)
      return signalPassFailure();
    root->setAttr("cf.lowered_count",
                  builder.getI64IntegerAttr(lowered));
  }
};

// TSV row: "convert-conv1x1-to-fc" at 0xe26ea5.
struct DwcConvertConv1x1ToFcPass : public darwinn::impl::DwcConvertConv1x1ToFcPassBase<DwcConvertConv1x1ToFcPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned lowered = 0;
    bool failedConvert = false;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "darwinn")
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedConvert = true;
        return WalkResult::interrupt();
      }
      op->setAttr("darwinn.lowered", builder.getUnitAttr());
      ++lowered;
      return WalkResult::advance();
    });
    if (failedConvert)
      return signalPassFailure();
    root->setAttr("darwinn.lowered_count",
                  builder.getI64IntegerAttr(lowered));
  }
};

// TSV row: "convert-dive-vm-tensor-to-linalg" at 0xde1ad1.
struct DwcConvertDiveVmTensorToLinalgPass : public darwinn::impl::DwcConvertDiveVmTensorToLinalgPassBase<DwcConvertDiveVmTensorToLinalgPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned lowered = 0;
    bool failedConvert = false;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "dive_vm")
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedConvert = true;
        return WalkResult::interrupt();
      }
      op->setAttr("dive_vm.lowered", builder.getUnitAttr());
      ++lowered;
      return WalkResult::advance();
    });
    if (failedConvert)
      return signalPassFailure();
    root->setAttr("dive_vm.lowered_count",
                  builder.getI64IntegerAttr(lowered));
  }
};

// TSV row: "convert-dive-vm-tensor-to-scf" at 0xde4651.
struct DwcConvertDiveVmTensorToScfPass : public darwinn::impl::DwcConvertDiveVmTensorToScfPassBase<DwcConvertDiveVmTensorToScfPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned lowered = 0;
    bool failedConvert = false;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "dive_vm")
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedConvert = true;
        return WalkResult::interrupt();
      }
      op->setAttr("dive_vm.lowered", builder.getUnitAttr());
      ++lowered;
      return WalkResult::advance();
    });
    if (failedConvert)
      return signalPassFailure();
    root->setAttr("dive_vm.lowered_count",
                  builder.getI64IntegerAttr(lowered));
  }
};

// TSV row: "convert-dive-vm-tensor-to-tensor" at 0xda89fa.
struct DwcConvertDiveVmTensorToTensorPass : public darwinn::impl::DwcConvertDiveVmTensorToTensorPassBase<DwcConvertDiveVmTensorToTensorPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned lowered = 0;
    bool failedConvert = false;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "dive_vm")
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedConvert = true;
        return WalkResult::interrupt();
      }
      op->setAttr("dive_vm.lowered", builder.getUnitAttr());
      ++lowered;
      return WalkResult::advance();
    });
    if (failedConvert)
      return signalPassFailure();
    root->setAttr("dive_vm.lowered_count",
                  builder.getI64IntegerAttr(lowered));
  }
};

// TSV row: "convert-dive-vm-to-llvm" at 0xdcb7dd.
struct DwcConvertDiveVmToLlvmPass : public darwinn::impl::DwcConvertDiveVmToLlvmPassBase<DwcConvertDiveVmToLlvmPass> {
  using Base::Base;

  void runOnOperation() override {
    // Per-op LLVM emission waits on kernel shapes in all_pseudocode.json.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned lowered = 0;
    bool failedConvert = false;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "dive_vm")
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedConvert = true;
        return WalkResult::interrupt();
      }
      op->setAttr("dive_vm.lowered_to_llvm", builder.getUnitAttr());
      ++lowered;
      return WalkResult::advance();
    });
    if (failedConvert)
      return signalPassFailure();
    root->setAttr("dive_vm.lowered_count",
                  builder.getI64IntegerAttr(lowered));
  }
};

// TSV row: "convert-dive-vm-to-memref" at 0xde3efd.
struct DwcConvertDiveVmToMemrefPass : public darwinn::impl::DwcConvertDiveVmToMemrefPassBase<DwcConvertDiveVmToMemrefPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned lowered = 0;
    bool failedConvert = false;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "dive_vm")
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedConvert = true;
        return WalkResult::interrupt();
      }
      op->setAttr("dive_vm.lowered_to_memref", builder.getUnitAttr());
      ++lowered;
      return WalkResult::advance();
    });
    if (failedConvert)
      return signalPassFailure();
    root->setAttr("dive_vm.lowered_count",
                  builder.getI64IntegerAttr(lowered));
  }
};

// TSV row: "convert-dwc-to-dive-vm-tensor" at 0xda8a3d.
struct DwcConvertDwcToDiveVmTensorPass : public darwinn::impl::DwcConvertDwcToDiveVmTensorPassBase<DwcConvertDwcToDiveVmTensorPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned lowered = 0;
    bool failedConvert = false;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "darwinn")
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedConvert = true;
        return WalkResult::interrupt();
      }
      op->setAttr("darwinn.lowered_to_dive_vm", builder.getUnitAttr());
      ++lowered;
      return WalkResult::advance();
    });
    if (failedConvert)
      return signalPassFailure();
    root->setAttr("darwinn.lowered_count",
                  builder.getI64IntegerAttr(lowered));
  }
};

// TSV row: "convert-dwg-to-dive-vm" at 0xdcb89c.
struct DwcConvertDwgToDiveVmPass : public darwinn::impl::DwcConvertDwgToDiveVmPassBase<DwcConvertDwgToDiveVmPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned lowered = 0;
    bool failedConvert = false;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "darwinn")
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedConvert = true;
        return WalkResult::interrupt();
      }
      op->setAttr("darwinn.lowered_to_dive_vm", builder.getUnitAttr());
      ++lowered;
      return WalkResult::advance();
    });
    if (failedConvert)
      return signalPassFailure();
    root->setAttr("darwinn.lowered_count",
                  builder.getI64IntegerAttr(lowered));
  }
};

// TSV row: "convert-dynamic-shape-scope-to-dive-vm" at 0xdcb8b3.
struct DwcConvertDynamicShapeScopeToDiveVmPass : public darwinn::impl::DwcConvertDynamicShapeScopeToDiveVmPassBase<DwcConvertDynamicShapeScopeToDiveVmPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned lowered = 0;
    bool failedConvert = false;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "darwinn")
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedConvert = true;
        return WalkResult::interrupt();
      }
      op->setAttr("darwinn.lowered_to_dive_vm", builder.getUnitAttr());
      ++lowered;
      return WalkResult::advance();
    });
    if (failedConvert)
      return signalPassFailure();
    root->setAttr("darwinn.lowered_count",
                  builder.getI64IntegerAttr(lowered));
  }
};

// TSV row: "convert-func-to-llvm" at 0xdcb867.
struct DwcConvertFuncToLlvmPass : public darwinn::impl::DwcConvertFuncToLlvmPassBase<DwcConvertFuncToLlvmPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned lowered = 0;
    bool failedConvert = false;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "func")
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedConvert = true;
        return WalkResult::interrupt();
      }
      op->setAttr("func.lowered_to_llvm", builder.getUnitAttr());
      ++lowered;
      return WalkResult::advance();
    });
    if (failedConvert)
      return signalPassFailure();
    root->setAttr("func.lowered_count",
                  builder.getI64IntegerAttr(lowered));
  }
};

// TSV row: "convert-generic-norm-to-pseudo-op" at 0xdb45be.
struct DwcConvertGenericNormToPseudoOpPass : public darwinn::impl::DwcConvertGenericNormToPseudoOpPassBase<DwcConvertGenericNormToPseudoOpPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned lowered = 0;
    bool failedConvert = false;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "darwinn")
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedConvert = true;
        return WalkResult::interrupt();
      }
      op->setAttr("darwinn.lowered", builder.getUnitAttr());
      ++lowered;
      return WalkResult::advance();
    });
    if (failedConvert)
      return signalPassFailure();
    root->setAttr("darwinn.lowered_count",
                  builder.getI64IntegerAttr(lowered));
  }
};

// TSV row: "convert-linalg-to-loops" at 0xd83df7.
struct DwcConvertLinalgToLoopsPass : public darwinn::impl::DwcConvertLinalgToLoopsPassBase<DwcConvertLinalgToLoopsPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned lowered = 0;
    bool failedConvert = false;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "linalg")
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedConvert = true;
        return WalkResult::interrupt();
      }
      op->setAttr("linalg.lowered", builder.getUnitAttr());
      ++lowered;
      return WalkResult::advance();
    });
    if (failedConvert)
      return signalPassFailure();
    root->setAttr("linalg.lowered_count",
                  builder.getI64IntegerAttr(lowered));
  }
};

// TSV row: "convert-math-to-libm" at 0xdce410.
struct DwcConvertMathToLibmPass : public darwinn::impl::DwcConvertMathToLibmPassBase<DwcConvertMathToLibmPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned lowered = 0;
    bool failedConvert = false;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "math")
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedConvert = true;
        return WalkResult::interrupt();
      }
      op->setAttr("math.lowered", builder.getUnitAttr());
      ++lowered;
      return WalkResult::advance();
    });
    if (failedConvert)
      return signalPassFailure();
    root->setAttr("math.lowered_count",
                  builder.getI64IntegerAttr(lowered));
  }
};

// TSV row: "convert-math-to-llvm" at 0xdcb80b.
struct DwcConvertMathToLlvmPass : public darwinn::impl::DwcConvertMathToLlvmPassBase<DwcConvertMathToLlvmPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned lowered = 0;
    bool failedConvert = false;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "math")
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedConvert = true;
        return WalkResult::interrupt();
      }
      op->setAttr("math.lowered_to_llvm", builder.getUnitAttr());
      ++lowered;
      return WalkResult::advance();
    });
    if (failedConvert)
      return signalPassFailure();
    root->setAttr("math.lowered_count",
                  builder.getI64IntegerAttr(lowered));
  }
};

// TSV row: "convert-op-lowering" at 0xddedce.
struct DwcConvertOpLoweringPass : public darwinn::impl::DwcConvertOpLoweringPassBase<DwcConvertOpLoweringPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned lowered = 0;
    bool failedConvert = false;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "darwinn")
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedConvert = true;
        return WalkResult::interrupt();
      }
      op->setAttr("darwinn.lowered", builder.getUnitAttr());
      ++lowered;
      return WalkResult::advance();
    });
    if (failedConvert)
      return signalPassFailure();
    root->setAttr("darwinn.lowered_count",
                  builder.getI64IntegerAttr(lowered));
  }
};

// TSV row: "convert-pdl-to-pdl-interp" at 0xdb3800.
struct DwcConvertPdlToPdlInterpPass : public darwinn::impl::DwcConvertPdlToPdlInterpPassBase<DwcConvertPdlToPdlInterpPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned lowered = 0;
    bool failedConvert = false;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "pdl")
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedConvert = true;
        return WalkResult::interrupt();
      }
      op->setAttr("pdl.lowered", builder.getUnitAttr());
      ++lowered;
      return WalkResult::advance();
    });
    if (failedConvert)
      return signalPassFailure();
    root->setAttr("pdl.lowered_count",
                  builder.getI64IntegerAttr(lowered));
  }
};

// TSV row: "convert-scatter-to-generic-scatter" at 0xdabcd2.
struct DwcConvertScatterToGenericScatterPass : public darwinn::impl::DwcConvertScatterToGenericScatterPassBase<DwcConvertScatterToGenericScatterPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned lowered = 0;
    bool failedConvert = false;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "darwinn")
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedConvert = true;
        return WalkResult::interrupt();
      }
      op->setAttr("darwinn.lowered", builder.getUnitAttr());
      ++lowered;
      return WalkResult::advance();
    });
    if (failedConvert)
      return signalPassFailure();
    root->setAttr("darwinn.lowered_count",
                  builder.getI64IntegerAttr(lowered));
  }
};

// TSV row: "convert-scf-to-cf" at 0xde4728.
struct DwcConvertScfToCfPass : public darwinn::impl::DwcConvertScfToCfPassBase<DwcConvertScfToCfPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned lowered = 0;
    bool failedConvert = false;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "scf")
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedConvert = true;
        return WalkResult::interrupt();
      }
      op->setAttr("scf.lowered", builder.getUnitAttr());
      ++lowered;
      return WalkResult::advance();
    });
    if (failedConvert)
      return signalPassFailure();
    root->setAttr("scf.lowered_count",
                  builder.getI64IntegerAttr(lowered));
  }
};

// TSV row: "convert-signed-int-with-rescaling-ops" at 0xd84723.
struct DwcConvertSignedIntWithRescalingOpsPass : public darwinn::impl::DwcConvertSignedIntWithRescalingOpsPassBase<DwcConvertSignedIntWithRescalingOpsPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned lowered = 0;
    bool failedConvert = false;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "arith")
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedConvert = true;
        return WalkResult::interrupt();
      }
      op->setAttr("arith.lowered", builder.getUnitAttr());
      ++lowered;
      return WalkResult::advance();
    });
    if (failedConvert)
      return signalPassFailure();
    root->setAttr("arith.lowered_count",
                  builder.getI64IntegerAttr(lowered));
  }
};

// TSV row: "convert-spatial-reduction-to-pooling" at 0xddfd8d.
struct DwcConvertSpatialReductionToPoolingPass : public darwinn::impl::DwcConvertSpatialReductionToPoolingPassBase<DwcConvertSpatialReductionToPoolingPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned lowered = 0;
    bool failedConvert = false;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "linalg")
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedConvert = true;
        return WalkResult::interrupt();
      }
      op->setAttr("linalg.lowered", builder.getUnitAttr());
      ++lowered;
      return WalkResult::advance();
    });
    if (failedConvert)
      return signalPassFailure();
    root->setAttr("linalg.lowered_count",
                  builder.getI64IntegerAttr(lowered));
  }
};

// TSV row: "convert-tf-to-dwc" at 0xe2523e.
struct DwcConvertTfToDwcPass : public darwinn::impl::DwcConvertTfToDwcPassBase<DwcConvertTfToDwcPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned lowered = 0;
    bool failedConvert = false;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "tf")
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedConvert = true;
        return WalkResult::interrupt();
      }
      op->setAttr("tf.lowered", builder.getUnitAttr());
      ++lowered;
      return WalkResult::advance();
    });
    if (failedConvert)
      return signalPassFailure();
    root->setAttr("tf.lowered_count",
                  builder.getI64IntegerAttr(lowered));
  }
};

// TSV row: "convert-to-k-in-m-sparsity" at 0xd5a046.
struct DwcConvertToKInMSparsityPass : public darwinn::impl::DwcConvertToKInMSparsityPassBase<DwcConvertToKInMSparsityPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned lowered = 0;
    bool failedConvert = false;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "darwinn")
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedConvert = true;
        return WalkResult::interrupt();
      }
      op->setAttr("darwinn.lowered", builder.getUnitAttr());
      ++lowered;
      return WalkResult::advance();
    });
    if (failedConvert)
      return signalPassFailure();
    root->setAttr("darwinn.lowered_count",
                  builder.getI64IntegerAttr(lowered));
  }
};

// TSV row: "convert-tpu-offload-to-dive-vm" at 0xdcb8da.
struct DwcConvertTpuOffloadToDiveVmPass : public darwinn::impl::DwcConvertTpuOffloadToDiveVmPassBase<DwcConvertTpuOffloadToDiveVmPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned lowered = 0;
    bool failedConvert = false;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "edgetpu")
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedConvert = true;
        return WalkResult::interrupt();
      }
      op->setAttr("edgetpu.lowered_to_dive_vm", builder.getUnitAttr());
      ++lowered;
      return WalkResult::advance();
    });
    if (failedConvert)
      return signalPassFailure();
    root->setAttr("edgetpu.lowered_count",
                  builder.getI64IntegerAttr(lowered));
  }
};

// TSV row: "convert-tpu-offload-to-llvm" at 0xdcb84b.
struct DwcConvertTpuOffloadToLlvmPass : public darwinn::impl::DwcConvertTpuOffloadToLlvmPassBase<DwcConvertTpuOffloadToLlvmPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned lowered = 0;
    bool failedConvert = false;
    root->walk([&](Operation *op) {
      bool isOffload = op->getName().getStringRef() == "dive_vm.tpu_offload";
      Dialect *dialect = op->getDialect();
      if (!isOffload && dialect &&
          dialect->getNamespace() == "edgetpu")
        isOffload = true;
      if (!isOffload)
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedConvert = true;
        return WalkResult::interrupt();
      }
      op->setAttr("tpu_offload.lowered_to_llvm", builder.getUnitAttr());
      ++lowered;
      return WalkResult::advance();
    });
    if (failedConvert)
      return signalPassFailure();
    root->setAttr("tpu_offload.lowered_count",
                  builder.getI64IntegerAttr(lowered));
  }
};

// TSV row: "convert-xla-supported-stablehlo" at 0xdbc2b1.
struct DwcConvertXlaSupportedStablehloPass : public darwinn::impl::DwcConvertXlaSupportedStablehloPassBase<DwcConvertXlaSupportedStablehloPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned lowered = 0;
    bool failedConvert = false;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "stablehlo")
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedConvert = true;
        return WalkResult::interrupt();
      }
      op->setAttr("stablehlo.lowered", builder.getUnitAttr());
      ++lowered;
      return WalkResult::advance();
    });
    if (failedConvert)
      return signalPassFailure();
    root->setAttr("stablehlo.lowered_count",
                  builder.getI64IntegerAttr(lowered));
  }
};

// TSV row: "ConvertDiveVmTensorToLinalg" at 0xde1af2.
struct DwcConvertDiveVmTensorToLinalgSymbolPass : public darwinn::impl::DwcConvertDiveVmTensorToLinalgSymbolPassBase<DwcConvertDiveVmTensorToLinalgSymbolPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned lowered = 0;
    bool failedConvert = false;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "dive_vm")
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedConvert = true;
        return WalkResult::interrupt();
      }
      op->setAttr("dive_vm.lowered", builder.getUnitAttr());
      ++lowered;
      return WalkResult::advance();
    });
    if (failedConvert)
      return signalPassFailure();
    root->setAttr("dive_vm.lowered_count",
                  builder.getI64IntegerAttr(lowered));
  }
};

// TSV row: "ConvertTpuOffloadToLlvm" at 0xdcb87c.
struct DwcConvertTpuOffloadToLlvmSymbolPass : public darwinn::impl::DwcConvertTpuOffloadToLlvmSymbolPassBase<DwcConvertTpuOffloadToLlvmSymbolPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned lowered = 0;
    bool failedConvert = false;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "edgetpu")
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedConvert = true;
        return WalkResult::interrupt();
      }
      op->setAttr("tpu_offload.lowered_to_llvm", builder.getUnitAttr());
      ++lowered;
      return WalkResult::advance();
    });
    if (failedConvert)
      return signalPassFailure();
    root->setAttr("tpu_offload.lowered_count",
                  builder.getI64IntegerAttr(lowered));
  }
};

// TSV row: "copy-op-lowering" at 0xddedbd.
struct DwcCopyOpLoweringPass : public darwinn::impl::DwcCopyOpLoweringPassBase<DwcCopyOpLoweringPass> {
  using Base::Base;

  void runOnOperation() override {
    // Per-op lowering waits on kernel shapes in all_pseudocode.json.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned marked = 0;
    bool failedMark = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedMark = true;
        return WalkResult::interrupt();
      }
      op->setAttr("copy-op-lowering.marked", builder.getUnitAttr());
      ++marked;
      return WalkResult::advance();
    });
    if (failedMark)
      return signalPassFailure();
    root->setAttr("copy-op-lowering.marked_count",
                  builder.getI64IntegerAttr(marked));
  }
};

// TSV row: "darwinn-bundling" at 0xde00af.
struct DwcDarwinnBundlingPass : public darwinn::impl::DwcDarwinnBundlingPassBase<DwcDarwinnBundlingPass> {
  using Base::Base;

  void runOnOperation() override {
    // Per-op lowering waits on kernel shapes in all_pseudocode.json.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned marked = 0;
    bool failedMark = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedMark = true;
        return WalkResult::interrupt();
      }
      op->setAttr("darwinn-bundling.marked", builder.getUnitAttr());
      ++marked;
      return WalkResult::advance();
    });
    if (failedMark)
      return signalPassFailure();
    root->setAttr("darwinn-bundling.marked_count",
                  builder.getI64IntegerAttr(marked));
  }
};

// TSV row: "darwinn.convert" at 0xd68ca7.
struct DwcDarwinnConvertPass : public darwinn::impl::DwcDarwinnConvertPassBase<DwcDarwinnConvertPass> {
  using Base::Base;

  void runOnOperation() override {
    // Per-op lowering waits on kernel shapes in all_pseudocode.json.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned marked = 0;
    bool failedMark = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedMark = true;
        return WalkResult::interrupt();
      }
      op->setAttr("darwinn.convert.marked", builder.getUnitAttr());
      ++marked;
      return WalkResult::advance();
    });
    if (failedMark)
      return signalPassFailure();
    root->setAttr("darwinn.convert.marked_count",
                  builder.getI64IntegerAttr(marked));
  }
};

// TSV row: "darwinn.math.join" at 0xdc9452.
struct DwcDarwinnMathJoinPass : public darwinn::impl::DwcDarwinnMathJoinPassBase<DwcDarwinnMathJoinPass> {
  using Base::Base;

  void runOnOperation() override {
    // Per-op lowering waits on kernel shapes in all_pseudocode.json.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned marked = 0;
    bool failedMark = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedMark = true;
        return WalkResult::interrupt();
      }
      op->setAttr("darwinn.math.join.marked", builder.getUnitAttr());
      ++marked;
      return WalkResult::advance();
    });
    if (failedMark)
      return signalPassFailure();
    root->setAttr("darwinn.math.join.marked_count",
                  builder.getI64IntegerAttr(marked));
  }
};

// TSV row: "darwinn.sparsity" at 0xd5a035.
struct DwcDarwinnSparsityPass : public darwinn::impl::DwcDarwinnSparsityPassBase<DwcDarwinnSparsityPass> {
  using Base::Base;

  void runOnOperation() override {
    // Per-op lowering waits on kernel shapes in all_pseudocode.json.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned marked = 0;
    bool failedMark = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedMark = true;
        return WalkResult::interrupt();
      }
      op->setAttr("darwinn.sparsity.marked", builder.getUnitAttr());
      ++marked;
      return WalkResult::advance();
    });
    if (failedMark)
      return signalPassFailure();
    root->setAttr("darwinn.sparsity.marked_count",
                  builder.getI64IntegerAttr(marked));
  }
};

// TSV row: "dive-dce" at 0xe0fe97.
struct DwcDiveDcePass : public darwinn::impl::DwcDiveDcePassBase<DwcDiveDcePass> {
  using Base::Base;

  void runOnOperation() override {
    // Per-op lowering waits on kernel shapes in all_pseudocode.json.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned marked = 0;
    bool failedMark = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedMark = true;
        return WalkResult::interrupt();
      }
      op->setAttr("dive-dce.marked", builder.getUnitAttr());
      ++marked;
      return WalkResult::advance();
    });
    if (failedMark)
      return signalPassFailure();
    root->setAttr("dive-dce.marked_count",
                  builder.getI64IntegerAttr(marked));
  }
};

// TSV row: "dive-io-optimization" at 0xdc141b.
struct DwcDiveIoOptimizationPass : public darwinn::impl::DwcDiveIoOptimizationPassBase<DwcDiveIoOptimizationPass> {
  using Base::Base;

  void runOnOperation() override {
    // Per-op lowering waits on kernel shapes in all_pseudocode.json.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned marked = 0;
    bool failedMark = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedMark = true;
        return WalkResult::interrupt();
      }
      op->setAttr("dive-io-optimization.marked", builder.getUnitAttr());
      ++marked;
      return WalkResult::advance();
    });
    if (failedMark)
      return signalPassFailure();
    root->setAttr("dive-io-optimization.marked_count",
                  builder.getI64IntegerAttr(marked));
  }
};

// TSV row: "dive-program-tpu" at 0xd6383a.
struct DwcDiveProgramTpuPass : public darwinn::impl::DwcDiveProgramTpuPassBase<DwcDiveProgramTpuPass> {
  using Base::Base;

  void runOnOperation() override {
    // Binary packet layout is absent from all_pseudocode.json, group and order only.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    std::map<std::string, SmallVector<Operation *>> groups;
    root->walk([&](Operation *op) {
      if (op->getName().getStringRef() != "dive_vm.tpu_offload")
        return;
      std::string key = "default";
      if (auto program = dyn_cast<StringAttr>(op->getAttr("tpu.program")))
        key = program.getValue().str();
      groups[key].push_back(op);
    });
    SmallVector<Attribute> order;
    unsigned packet = 0;
    for (auto &entry : groups) {
      order.push_back(builder.getStringAttr(entry.first));
      unsigned pos = 0;
      for (Operation *op : entry.second) {
        op->setAttr("tpu.packet_id", builder.getI64IntegerAttr(packet));
        op->setAttr("tpu.packet_order", builder.getI64IntegerAttr(pos++));
      }
      ++packet;
    }
    root->setAttr("tpu.packet_count", builder.getI64IntegerAttr(packet));
    root->setAttr("tpu.packet_order", builder.getArrayAttr(order));
  }
};

// TSV row: "dive-unroll-factor" at 0xda744d.
struct DwcDiveUnrollFactorPass : public darwinn::impl::DwcDiveUnrollFactorPassBase<DwcDiveUnrollFactorPass> {
  using Base::Base;

  void runOnOperation() override {
    // Per-op lowering waits on kernel shapes in all_pseudocode.json.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned marked = 0;
    bool failedMark = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedMark = true;
        return WalkResult::interrupt();
      }
      op->setAttr("dive-unroll-factor.marked", builder.getUnitAttr());
      ++marked;
      return WalkResult::advance();
    });
    if (failedMark)
      return signalPassFailure();
    root->setAttr("dive-unroll-factor.marked_count",
                  builder.getI64IntegerAttr(marked));
  }
};

// TSV row: "dive-vm-bufferize" at 0xde6a70.
struct DwcDiveVmBufferizePass : public darwinn::impl::DwcDiveVmBufferizePassBase<DwcDiveVmBufferizePass> {
  using Base::Base;

  void runOnOperation() override {
    // Per-op lowering waits on kernel shapes in all_pseudocode.json.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned marked = 0;
    bool failedMark = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedMark = true;
        return WalkResult::interrupt();
      }
      op->setAttr("dive-vm-bufferize.marked", builder.getUnitAttr());
      ++marked;
      return WalkResult::advance();
    });
    if (failedMark)
      return signalPassFailure();
    root->setAttr("dive-vm-bufferize.marked_count",
                  builder.getI64IntegerAttr(marked));
  }
};

// TSV row: "dive-vm-outline-shareable-dive-consts" at 0xd788a7.
struct DwcDiveVmOutlineShareableDiveConstsPass : public darwinn::impl::DwcDiveVmOutlineShareableDiveConstsPassBase<DwcDiveVmOutlineShareableDiveConstsPass> {
  using Base::Base;

  void runOnOperation() override {
    // Per-op lowering waits on kernel shapes in all_pseudocode.json.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned marked = 0;
    bool failedMark = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedMark = true;
        return WalkResult::interrupt();
      }
      op->setAttr("dive-vm-outline-shareable-dive-consts.marked", builder.getUnitAttr());
      ++marked;
      return WalkResult::advance();
    });
    if (failedMark)
      return signalPassFailure();
    root->setAttr("dive-vm-outline-shareable-dive-consts.marked_count",
                  builder.getI64IntegerAttr(marked));
  }
};

// TSV row: "dwc-check-illegal-tpu-ops" at 0xd84494.
struct DwcDwcCheckIllegalTpuOpsPass : public darwinn::impl::DwcDwcCheckIllegalTpuOpsPassBase<DwcDwcCheckIllegalTpuOpsPass> {
  using Base::Base;

  void runOnOperation() override {
    // Binary packet layout is absent from all_pseudocode.json, group and order only.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    std::map<Operation *, unsigned> clusterOf;
    unsigned nextCluster = 0;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      unsigned cluster = nextCluster;
      bool joined = false;
      for (Value operand : op->getOperands()) {
        Operation *def = operand.getDefiningOp();
        if (!def)
          continue;
        auto it = clusterOf.find(def);
        if (it != clusterOf.end()) {
          cluster = it->second;
          joined = true;
          break;
        }
      }
      if (!joined)
        ++nextCluster;
      clusterOf[op] = cluster;
      op->setAttr("tpu.cluster_id", builder.getI64IntegerAttr(cluster));
      return WalkResult::advance();
    });
    root->setAttr("tpu.cluster_count",
                  builder.getI64IntegerAttr(nextCluster));
  }
};

// TSV row: "dwc-convert-input-output-types" at 0xd94dcc.
struct DwcDwcConvertInputOutputTypesPass : public darwinn::impl::DwcDwcConvertInputOutputTypesPassBase<DwcDwcConvertInputOutputTypesPass> {
  using Base::Base;

  void runOnOperation() override {
    // Binary packet layout is absent from all_pseudocode.json, group and order only.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    std::map<Operation *, unsigned> clusterOf;
    unsigned nextCluster = 0;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      unsigned cluster = nextCluster;
      bool joined = false;
      for (Value operand : op->getOperands()) {
        Operation *def = operand.getDefiningOp();
        if (!def)
          continue;
        auto it = clusterOf.find(def);
        if (it != clusterOf.end()) {
          cluster = it->second;
          joined = true;
          break;
        }
      }
      if (!joined)
        ++nextCluster;
      clusterOf[op] = cluster;
      op->setAttr("tpu.cluster_id", builder.getI64IntegerAttr(cluster));
      return WalkResult::advance();
    });
    root->setAttr("tpu.cluster_count",
                  builder.getI64IntegerAttr(nextCluster));
  }
};

// TSV row: "dwc-copy-strided-buffers-on-tpu" at 0xd6381a.
struct DwcDwcCopyStridedBuffersOnTpuPass : public darwinn::impl::DwcDwcCopyStridedBuffersOnTpuPassBase<DwcDwcCopyStridedBuffersOnTpuPass> {
  using Base::Base;

  void runOnOperation() override {
    // Binary packet layout is absent from all_pseudocode.json, group and order only.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    std::map<Operation *, unsigned> clusterOf;
    unsigned nextCluster = 0;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      unsigned cluster = nextCluster;
      bool joined = false;
      for (Value operand : op->getOperands()) {
        Operation *def = operand.getDefiningOp();
        if (!def)
          continue;
        auto it = clusterOf.find(def);
        if (it != clusterOf.end()) {
          cluster = it->second;
          joined = true;
          break;
        }
      }
      if (!joined)
        ++nextCluster;
      clusterOf[op] = cluster;
      op->setAttr("tpu.cluster_id", builder.getI64IntegerAttr(cluster));
      return WalkResult::advance();
    });
    root->setAttr("tpu.cluster_count",
                  builder.getI64IntegerAttr(nextCluster));
  }
};

// TSV row: "dwc-form-tpu-clusters" at 0xd81a1f.
struct DwcDwcFormTpuClustersPass : public darwinn::impl::DwcDwcFormTpuClustersPassBase<DwcDwcFormTpuClustersPass> {
  using Base::Base;

  void runOnOperation() override {
    // Binary packet layout is absent from all_pseudocode.json, group and order only.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    std::map<Operation *, unsigned> clusterOf;
    unsigned nextCluster = 0;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      unsigned cluster = nextCluster;
      bool joined = false;
      for (Value operand : op->getOperands()) {
        Operation *def = operand.getDefiningOp();
        if (!def)
          continue;
        auto it = clusterOf.find(def);
        if (it != clusterOf.end()) {
          cluster = it->second;
          joined = true;
          break;
        }
      }
      if (!joined)
        ++nextCluster;
      clusterOf[op] = cluster;
      op->setAttr("tpu.cluster_id", builder.getI64IntegerAttr(cluster));
      return WalkResult::advance();
    });
    root->setAttr("tpu.cluster_count",
                  builder.getI64IntegerAttr(nextCluster));
  }
};

// TSV row: "dwc-legalize" at 0xde6d3f.
struct DwcDwcLegalizePass : public darwinn::impl::DwcDwcLegalizePassBase<DwcDwcLegalizePass> {
  using Base::Base;

  void runOnOperation() override {
    // Ops outside the canonical pipeline have no kernel shape evidence, reject them.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect) {
        op->emitError() << "dwc-legalize rejects unregistered operation "
                        << op->getName().getStringRef();
        failedLegal = true;
        return WalkResult::interrupt();
      }
      StringRef ns = dialect->getNamespace();
      if (ns != "darwinn" && ns != "dive_vm" && ns != "edgetpu" &&
          ns != "func" && ns != "arith" && ns != "tensor" &&
          ns != "memref" && ns != "scf" && ns != "cf" &&
          ns != "builtin" && ns != "llvm") {
        op->emitError() << "dwc-legalize rejects operation from dialect "
                        << ns;
        failedLegal = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failedLegal)
      signalPassFailure();
  }
};

// TSV row: "dwc-legalize-hlo" at 0xdbc39d.
struct DwcDwcLegalizeHloPass : public darwinn::impl::DwcDwcLegalizeHloPassBase<DwcDwcLegalizeHloPass> {
  using Base::Base;

  void runOnOperation() override {
    // Ops outside the canonical pipeline have no kernel shape evidence, reject them.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect) {
        op->emitError() << "dwc-legalize-hlo rejects unregistered operation "
                        << op->getName().getStringRef();
        failedLegal = true;
        return WalkResult::interrupt();
      }
      StringRef ns = dialect->getNamespace();
      if (ns != "darwinn" && ns != "dive_vm" && ns != "edgetpu" &&
          ns != "func" && ns != "arith" && ns != "tensor" &&
          ns != "memref" && ns != "scf" && ns != "cf" &&
          ns != "builtin" && ns != "llvm" && ns != "mhlo") {
        op->emitError() << "dwc-legalize-hlo rejects operation from dialect "
                        << ns;
        failedLegal = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failedLegal)
      signalPassFailure();
  }
};

// TSV row: "dwc-legalize-hlo-to-tf" at 0xde2ad1.
struct DwcDwcLegalizeHloToTfPass : public darwinn::impl::DwcDwcLegalizeHloToTfPassBase<DwcDwcLegalizeHloToTfPass> {
  using Base::Base;

  void runOnOperation() override {
    // Ops outside the canonical pipeline have no kernel shape evidence, reject them.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect) {
        op->emitError() << "dwc-legalize-hlo-to-tf rejects unregistered operation "
                        << op->getName().getStringRef();
        failedLegal = true;
        return WalkResult::interrupt();
      }
      StringRef ns = dialect->getNamespace();
      if (ns != "darwinn" && ns != "dive_vm" && ns != "edgetpu" &&
          ns != "func" && ns != "arith" && ns != "tensor" &&
          ns != "memref" && ns != "scf" && ns != "cf" &&
          ns != "builtin" && ns != "llvm" &&
          ns != "mhlo" && ns != "tf") {
        op->emitError() << "dwc-legalize-hlo-to-tf rejects operation from dialect "
                        << ns;
        failedLegal = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failedLegal)
      signalPassFailure();
  }
};

// TSV row: "dwc-legalize-int-and-quant-types" at 0xd94e00.
struct DwcDwcLegalizeIntAndQuantTypesPass : public darwinn::impl::DwcDwcLegalizeIntAndQuantTypesPassBase<DwcDwcLegalizeIntAndQuantTypesPass> {
  using Base::Base;

  void runOnOperation() override {
    // Ops outside the canonical pipeline have no kernel shape evidence, reject them.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect) {
        op->emitError() << "dwc-legalize-int-and-quant-types rejects unregistered operation "
                        << op->getName().getStringRef();
        failedLegal = true;
        return WalkResult::interrupt();
      }
      StringRef ns = dialect->getNamespace();
      if (ns != "darwinn" && ns != "dive_vm" && ns != "edgetpu" &&
          ns != "func" && ns != "arith" && ns != "tensor" &&
          ns != "memref" && ns != "scf" && ns != "cf" &&
          ns != "builtin" && ns != "llvm" && ns != "quant") {
        op->emitError() << "dwc-legalize-int-and-quant-types rejects operation from dialect "
                        << ns;
        failedLegal = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failedLegal)
      signalPassFailure();
  }
};

// TSV row: "dwc-legalize-int64-constants" at 0xd7bb5d.
struct DwcDwcLegalizeInt64ConstantsPass : public darwinn::impl::DwcDwcLegalizeInt64ConstantsPassBase<DwcDwcLegalizeInt64ConstantsPass> {
  using Base::Base;

  void runOnOperation() override {
    // Ops outside the canonical pipeline have no kernel shape evidence, reject them.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect) {
        op->emitError() << "dwc-legalize-int64-constants rejects unregistered operation "
                        << op->getName().getStringRef();
        failedLegal = true;
        return WalkResult::interrupt();
      }
      StringRef ns = dialect->getNamespace();
      if (ns != "darwinn" && ns != "dive_vm" && ns != "edgetpu" &&
          ns != "func" && ns != "arith" && ns != "tensor" &&
          ns != "memref" && ns != "scf" && ns != "cf" &&
          ns != "builtin" && ns != "llvm") {
        op->emitError() << "dwc-legalize-int64-constants rejects operation from dialect "
                        << ns;
        failedLegal = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failedLegal)
      signalPassFailure();
  }
};

// TSV row: "dwc-legalize-pass" at 0xd7f2b9.
struct DwcDwcLegalizePassPass : public darwinn::impl::DwcDwcLegalizePassPassBase<DwcDwcLegalizePassPass> {
  using Base::Base;

  void runOnOperation() override {
    // Ops outside the canonical pipeline have no kernel shape evidence, reject them.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect) {
        op->emitError() << "dwc-legalize-pass rejects unregistered operation "
                        << op->getName().getStringRef();
        failedLegal = true;
        return WalkResult::interrupt();
      }
      StringRef ns = dialect->getNamespace();
      if (ns != "darwinn" && ns != "dive_vm" && ns != "edgetpu" &&
          ns != "func" && ns != "arith" && ns != "tensor" &&
          ns != "memref" && ns != "scf" && ns != "cf" &&
          ns != "builtin" && ns != "llvm") {
        op->emitError() << "dwc-legalize-pass rejects operation from dialect "
                        << ns;
        failedLegal = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failedLegal)
      signalPassFailure();
  }
};

// TSV row: "dwc-legalize-stablehlo-annotate-materialize-policy" at 0xd5dd77.
struct DwcDwcLegalizeStablehloAnnotateMaterializePolicyPass : public darwinn::impl::DwcDwcLegalizeStablehloAnnotateMaterializePolicyPassBase<DwcDwcLegalizeStablehloAnnotateMaterializePolicyPass> {
  using Base::Base;

  void runOnOperation() override {
    // Ops outside the canonical pipeline have no kernel shape evidence, reject them.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect) {
        op->emitError() << "dwc-legalize-stablehlo-annotate-materialize-policy rejects unregistered operation "
                        << op->getName().getStringRef();
        failedLegal = true;
        return WalkResult::interrupt();
      }
      StringRef ns = dialect->getNamespace();
      if (ns != "darwinn" && ns != "dive_vm" && ns != "edgetpu" &&
          ns != "func" && ns != "arith" && ns != "tensor" &&
          ns != "memref" && ns != "scf" && ns != "cf" &&
          ns != "builtin" && ns != "llvm" && ns != "stablehlo") {
        op->emitError() << "dwc-legalize-stablehlo-annotate-materialize-policy rejects operation from dialect "
                        << ns;
        failedLegal = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failedLegal)
      signalPassFailure();
  }
};

// TSV row: "dwc-legalize-stablehlo-composite" at 0xdef599.
struct DwcDwcLegalizeStablehloCompositePass : public darwinn::impl::DwcDwcLegalizeStablehloCompositePassBase<DwcDwcLegalizeStablehloCompositePass> {
  using Base::Base;

  void runOnOperation() override {
    // Ops outside the canonical pipeline have no kernel shape evidence, reject them.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect) {
        op->emitError() << "dwc-legalize-stablehlo-composite rejects unregistered operation "
                        << op->getName().getStringRef();
        failedLegal = true;
        return WalkResult::interrupt();
      }
      StringRef ns = dialect->getNamespace();
      if (ns != "darwinn" && ns != "dive_vm" && ns != "edgetpu" &&
          ns != "func" && ns != "arith" && ns != "tensor" &&
          ns != "memref" && ns != "scf" && ns != "cf" &&
          ns != "builtin" && ns != "llvm" && ns != "stablehlo") {
        op->emitError() << "dwc-legalize-stablehlo-composite rejects operation from dialect "
                        << ns;
        failedLegal = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failedLegal)
      signalPassFailure();
  }
};

// TSV row: "dwc-legalize-tf-pipeline" at 0xe03284.
struct DwcDwcLegalizeTfPipelinePass : public darwinn::impl::DwcDwcLegalizeTfPipelinePassBase<DwcDwcLegalizeTfPipelinePass> {
  using Base::Base;

  void runOnOperation() override {
    // Ops outside the canonical pipeline have no kernel shape evidence, reject them.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect) {
        op->emitError() << "dwc-legalize-tf-pipeline rejects unregistered operation "
                        << op->getName().getStringRef();
        failedLegal = true;
        return WalkResult::interrupt();
      }
      StringRef ns = dialect->getNamespace();
      if (ns != "darwinn" && ns != "dive_vm" && ns != "edgetpu" &&
          ns != "func" && ns != "arith" && ns != "tensor" &&
          ns != "memref" && ns != "scf" && ns != "cf" &&
          ns != "builtin" && ns != "llvm" && ns != "tf") {
        op->emitError() << "dwc-legalize-tf-pipeline rejects operation from dialect "
                        << ns;
        failedLegal = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failedLegal)
      signalPassFailure();
  }
};

// TSV row: "dwc-legalize-tfl-cudaemu-custom-ops" at 0xd8464e.
struct DwcDwcLegalizeTflCudaemuCustomOpsPass : public darwinn::impl::DwcDwcLegalizeTflCudaemuCustomOpsPassBase<DwcDwcLegalizeTflCudaemuCustomOpsPass> {
  using Base::Base;

  void runOnOperation() override {
    // Ops outside the canonical pipeline have no kernel shape evidence, reject them.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect) {
        op->emitError() << "dwc-legalize-tfl-cudaemu-custom-ops rejects unregistered operation "
                        << op->getName().getStringRef();
        failedLegal = true;
        return WalkResult::interrupt();
      }
      StringRef ns = dialect->getNamespace();
      if (ns != "darwinn" && ns != "dive_vm" && ns != "edgetpu" &&
          ns != "func" && ns != "arith" && ns != "tensor" &&
          ns != "memref" && ns != "scf" && ns != "cf" &&
          ns != "builtin" && ns != "llvm" && ns != "tfl") {
        op->emitError() << "dwc-legalize-tfl-cudaemu-custom-ops rejects operation from dialect "
                        << ns;
        failedLegal = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failedLegal)
      signalPassFailure();
  }
};

// TSV row: "dwc-legalize-tfl-multinomial" at 0xdd3cdb.
struct DwcDwcLegalizeTflMultinomialPass : public darwinn::impl::DwcDwcLegalizeTflMultinomialPassBase<DwcDwcLegalizeTflMultinomialPass> {
  using Base::Base;

  void runOnOperation() override {
    // Ops outside the canonical pipeline have no kernel shape evidence, reject them.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect) {
        op->emitError() << "dwc-legalize-tfl-multinomial rejects unregistered operation "
                        << op->getName().getStringRef();
        failedLegal = true;
        return WalkResult::interrupt();
      }
      StringRef ns = dialect->getNamespace();
      if (ns != "darwinn" && ns != "dive_vm" && ns != "edgetpu" &&
          ns != "func" && ns != "arith" && ns != "tensor" &&
          ns != "memref" && ns != "scf" && ns != "cf" &&
          ns != "builtin" && ns != "llvm" && ns != "tfl") {
        op->emitError() << "dwc-legalize-tfl-multinomial rejects operation from dialect "
                        << ns;
        failedLegal = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failedLegal)
      signalPassFailure();
  }
};

// TSV row: "dwc-legalize-tfl-variable-tensors" at 0xd81097.
struct DwcDwcLegalizeTflVariableTensorsPass : public darwinn::impl::DwcDwcLegalizeTflVariableTensorsPassBase<DwcDwcLegalizeTflVariableTensorsPass> {
  using Base::Base;

  void runOnOperation() override {
    // Ops outside the canonical pipeline have no kernel shape evidence, reject them.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect) {
        op->emitError() << "dwc-legalize-tfl-variable-tensors rejects unregistered operation "
                        << op->getName().getStringRef();
        failedLegal = true;
        return WalkResult::interrupt();
      }
      StringRef ns = dialect->getNamespace();
      if (ns != "darwinn" && ns != "dive_vm" && ns != "edgetpu" &&
          ns != "func" && ns != "arith" && ns != "tensor" &&
          ns != "memref" && ns != "scf" && ns != "cf" &&
          ns != "builtin" && ns != "llvm" && ns != "tfl") {
        op->emitError() << "dwc-legalize-tfl-variable-tensors rejects operation from dialect "
                        << ns;
        failedLegal = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failedLegal)
      signalPassFailure();
  }
};

// TSV row: "dwc-legalize-uint32-types" at 0xd94e3e.
struct DwcDwcLegalizeUint32TypesPass : public darwinn::impl::DwcDwcLegalizeUint32TypesPassBase<DwcDwcLegalizeUint32TypesPass> {
  using Base::Base;

  void runOnOperation() override {
    // Ops outside the canonical pipeline have no kernel shape evidence, reject them.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect) {
        op->emitError() << "dwc-legalize-uint32-types rejects unregistered operation "
                        << op->getName().getStringRef();
        failedLegal = true;
        return WalkResult::interrupt();
      }
      StringRef ns = dialect->getNamespace();
      if (ns != "darwinn" && ns != "dive_vm" && ns != "edgetpu" &&
          ns != "func" && ns != "arith" && ns != "tensor" &&
          ns != "memref" && ns != "scf" && ns != "cf" &&
          ns != "builtin" && ns != "llvm") {
        op->emitError() << "dwc-legalize-uint32-types rejects operation from dialect "
                        << ns;
        failedLegal = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failedLegal)
      signalPassFailure();
  }
};

// TSV row: "dwc-lower-argmax-index-unpool" at 0xdcf030.
struct DwcDwcLowerArgmaxIndexUnpoolPass : public darwinn::impl::DwcDwcLowerArgmaxIndexUnpoolPassBase<DwcDwcLowerArgmaxIndexUnpoolPass> {
  using Base::Base;

  void runOnOperation() override {
    // The sibling-owned LowerCopySlice and LowerConvert sets do the real
    // lowering, identity folds below only clean up what patterns leave behind.
    RewritePatternSet patterns(&getContext());
    darwinn::populateLowerCopySlicePatterns(patterns);
    darwinn::populateLowerConvertPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns))))
      return signalPassFailure();
    func::FuncOp func = getOperation();
    if (failed(applyLocalCopySliceLowering(func)))
      return signalPassFailure();
    if (failed(applyLocalConvertLowering(func)))
      return signalPassFailure();
  }
};

// TSV row: "dwc-lower-composite-ops" at 0xd84775.
struct DwcDwcLowerCompositeOpsPass : public darwinn::impl::DwcDwcLowerCompositeOpsPassBase<DwcDwcLowerCompositeOpsPass> {
  using Base::Base;

  void runOnOperation() override {
    // The sibling-owned LowerCopySlice and LowerConvert sets do the real
    // lowering, identity folds below only clean up what patterns leave behind.
    RewritePatternSet patterns(&getContext());
    darwinn::populateLowerCopySlicePatterns(patterns);
    darwinn::populateLowerConvertPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns))))
      return signalPassFailure();
    func::FuncOp func = getOperation();
    if (failed(applyLocalCopySliceLowering(func)))
      return signalPassFailure();
    if (failed(applyLocalConvertLowering(func)))
      return signalPassFailure();
  }
};

// TSV row: "dwc-lower-control-flow" at 0xd61f2c.
struct DwcDwcLowerControlFlowPass : public darwinn::impl::DwcDwcLowerControlFlowPassBase<DwcDwcLowerControlFlowPass> {
  using Base::Base;

  void runOnOperation() override {
    // The sibling-owned LowerCopySlice and LowerConvert sets do the real
    // lowering, identity folds below only clean up what patterns leave behind.
    RewritePatternSet patterns(&getContext());
    darwinn::populateLowerCopySlicePatterns(patterns);
    darwinn::populateLowerConvertPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns))))
      return signalPassFailure();
    func::FuncOp func = getOperation();
    if (failed(applyLocalCopySliceLowering(func)))
      return signalPassFailure();
    if (failed(applyLocalConvertLowering(func)))
      return signalPassFailure();
  }
};

// TSV row: "dwc-lower-depth-to-from-space" at 0xe102d0.
struct DwcDwcLowerDepthToFromSpacePass : public darwinn::impl::DwcDwcLowerDepthToFromSpacePassBase<DwcDwcLowerDepthToFromSpacePass> {
  using Base::Base;

  void runOnOperation() override {
    // The sibling-owned LowerCopySlice and LowerConvert sets do the real
    // lowering, identity folds below only clean up what patterns leave behind.
    RewritePatternSet patterns(&getContext());
    darwinn::populateLowerCopySlicePatterns(patterns);
    darwinn::populateLowerConvertPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns))))
      return signalPassFailure();
    func::FuncOp func = getOperation();
    if (failed(applyLocalCopySliceLowering(func)))
      return signalPassFailure();
    if (failed(applyLocalConvertLowering(func)))
      return signalPassFailure();
  }
};

// TSV row: "dwc-lower-generic-constants" at 0xd7bb41.
struct DwcDwcLowerGenericConstantsPass : public darwinn::impl::DwcDwcLowerGenericConstantsPassBase<DwcDwcLowerGenericConstantsPass> {
  using Base::Base;

  void runOnOperation() override {
    // The sibling-owned LowerCopySlice and LowerConvert sets do the real
    // lowering, identity folds below only clean up what patterns leave behind.
    RewritePatternSet patterns(&getContext());
    darwinn::populateLowerCopySlicePatterns(patterns);
    darwinn::populateLowerConvertPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns))))
      return signalPassFailure();
    func::FuncOp func = getOperation();
    if (failed(applyLocalCopySliceLowering(func)))
      return signalPassFailure();
    if (failed(applyLocalConvertLowering(func)))
      return signalPassFailure();
  }
};

// TSV row: "dwc-lower-hlops" at 0xd841f4.
struct DwcDwcLowerHlopsPass : public darwinn::impl::DwcDwcLowerHlopsPassBase<DwcDwcLowerHlopsPass> {
  using Base::Base;

  void runOnOperation() override {
    // The sibling-owned LowerCopySlice and LowerConvert sets do the real
    // lowering, identity folds below only clean up what patterns leave behind.
    RewritePatternSet patterns(&getContext());
    darwinn::populateLowerCopySlicePatterns(patterns);
    darwinn::populateLowerConvertPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns))))
      return signalPassFailure();
    func::FuncOp func = getOperation();
    if (failed(applyLocalCopySliceLowering(func)))
      return signalPassFailure();
    if (failed(applyLocalConvertLowering(func)))
      return signalPassFailure();
  }
};

// TSV row: "dwc-lower-input-output-cast" at 0xd68084.
struct DwcDwcLowerInputOutputCastPass : public darwinn::impl::DwcDwcLowerInputOutputCastPassBase<DwcDwcLowerInputOutputCastPass> {
  using Base::Base;

  void runOnOperation() override {
    // The sibling-owned LowerCopySlice and LowerConvert sets do the real
    // lowering, identity folds below only clean up what patterns leave behind.
    RewritePatternSet patterns(&getContext());
    darwinn::populateLowerCopySlicePatterns(patterns);
    darwinn::populateLowerConvertPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns))))
      return signalPassFailure();
    func::FuncOp func = getOperation();
    if (failed(applyLocalCopySliceLowering(func)))
      return signalPassFailure();
    if (failed(applyLocalConvertLowering(func)))
      return signalPassFailure();
  }
};

// TSV row: "dwc-lower-padding-ops" at 0xd84749.
struct DwcDwcLowerPaddingOpsPass : public darwinn::impl::DwcDwcLowerPaddingOpsPassBase<DwcDwcLowerPaddingOpsPass> {
  using Base::Base;

  void runOnOperation() override {
    // The sibling-owned LowerCopySlice and LowerConvert sets do the real
    // lowering, identity folds below only clean up what patterns leave behind.
    RewritePatternSet patterns(&getContext());
    darwinn::populateLowerCopySlicePatterns(patterns);
    darwinn::populateLowerConvertPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns))))
      return signalPassFailure();
    func::FuncOp func = getOperation();
    if (failed(applyLocalCopySliceLowering(func)))
      return signalPassFailure();
    if (failed(applyLocalConvertLowering(func)))
      return signalPassFailure();
  }
};

// TSV row: "dwc-lower-pseudo-ops" at 0xd845c5.
struct DwcDwcLowerPseudoOpsPass : public darwinn::impl::DwcDwcLowerPseudoOpsPassBase<DwcDwcLowerPseudoOpsPass> {
  using Base::Base;

  void runOnOperation() override {
    // The sibling-owned LowerCopySlice and LowerConvert sets do the real
    // lowering, identity folds below only clean up what patterns leave behind.
    RewritePatternSet patterns(&getContext());
    darwinn::populateLowerCopySlicePatterns(patterns);
    darwinn::populateLowerConvertPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns))))
      return signalPassFailure();
    func::FuncOp func = getOperation();
    if (failed(applyLocalCopySliceLowering(func)))
      return signalPassFailure();
    if (failed(applyLocalConvertLowering(func)))
      return signalPassFailure();
  }
};

// TSV row: "dwc-lower-resampler-ops" at 0xd84598.
struct DwcDwcLowerResamplerOpsPass : public darwinn::impl::DwcDwcLowerResamplerOpsPassBase<DwcDwcLowerResamplerOpsPass> {
  using Base::Base;

  void runOnOperation() override {
    // The sibling-owned LowerCopySlice and LowerConvert sets do the real
    // lowering, identity folds below only clean up what patterns leave behind.
    RewritePatternSet patterns(&getContext());
    darwinn::populateLowerCopySlicePatterns(patterns);
    darwinn::populateLowerConvertPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns))))
      return signalPassFailure();
    func::FuncOp func = getOperation();
    if (failed(applyLocalCopySliceLowering(func)))
      return signalPassFailure();
    if (failed(applyLocalConvertLowering(func)))
      return signalPassFailure();
  }
};

// TSV row: "dwc-lower-scalar-ops" at 0xd845b0.
struct DwcDwcLowerScalarOpsPass : public darwinn::impl::DwcDwcLowerScalarOpsPassBase<DwcDwcLowerScalarOpsPass> {
  using Base::Base;

  void runOnOperation() override {
    // The sibling-owned LowerCopySlice and LowerConvert sets do the real
    // lowering, identity folds below only clean up what patterns leave behind.
    RewritePatternSet patterns(&getContext());
    darwinn::populateLowerCopySlicePatterns(patterns);
    darwinn::populateLowerConvertPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns))))
      return signalPassFailure();
    func::FuncOp func = getOperation();
    if (failed(applyLocalCopySliceLowering(func)))
      return signalPassFailure();
    if (failed(applyLocalConvertLowering(func)))
      return signalPassFailure();
  }
};

// TSV row: "dwc-lower-scatter-ops" at 0xd84582.
struct DwcDwcLowerScatterOpsPass : public darwinn::impl::DwcDwcLowerScatterOpsPassBase<DwcDwcLowerScatterOpsPass> {
  using Base::Base;

  void runOnOperation() override {
    // The sibling-owned LowerCopySlice and LowerConvert sets do the real
    // lowering, identity folds below only clean up what patterns leave behind.
    RewritePatternSet patterns(&getContext());
    darwinn::populateLowerCopySlicePatterns(patterns);
    darwinn::populateLowerConvertPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns))))
      return signalPassFailure();
    func::FuncOp func = getOperation();
    if (failed(applyLocalCopySliceLowering(func)))
      return signalPassFailure();
    if (failed(applyLocalConvertLowering(func)))
      return signalPassFailure();
  }
};

// TSV row: "dwc-lower-top-k" at 0xdd6da3.
struct DwcDwcLowerTopKPass : public darwinn::impl::DwcDwcLowerTopKPassBase<DwcDwcLowerTopKPass> {
  using Base::Base;

  void runOnOperation() override {
    // The sibling-owned LowerCopySlice and LowerConvert sets do the real
    // lowering, identity folds below only clean up what patterns leave behind.
    RewritePatternSet patterns(&getContext());
    darwinn::populateLowerCopySlicePatterns(patterns);
    darwinn::populateLowerConvertPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns))))
      return signalPassFailure();
    func::FuncOp func = getOperation();
    if (failed(applyLocalCopySliceLowering(func)))
      return signalPassFailure();
    if (failed(applyLocalConvertLowering(func)))
      return signalPassFailure();
  }
};

// TSV row: "dwc-post-truncation-tpu-fitter" at 0xdab7fc.
struct DwcDwcPostTruncationTpuFitterPass : public darwinn::impl::DwcDwcPostTruncationTpuFitterPassBase<DwcDwcPostTruncationTpuFitterPass> {
  using Base::Base;

  void runOnOperation() override {
    // Binary packet layout is absent from all_pseudocode.json, group and order only.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    std::map<Operation *, unsigned> clusterOf;
    unsigned nextCluster = 0;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      unsigned cluster = nextCluster;
      bool joined = false;
      for (Value operand : op->getOperands()) {
        Operation *def = operand.getDefiningOp();
        if (!def)
          continue;
        auto it = clusterOf.find(def);
        if (it != clusterOf.end()) {
          cluster = it->second;
          joined = true;
          break;
        }
      }
      if (!joined)
        ++nextCluster;
      clusterOf[op] = cluster;
      op->setAttr("tpu.cluster_id", builder.getI64IntegerAttr(cluster));
      return WalkResult::advance();
    });
    root->setAttr("tpu.cluster_count",
                  builder.getI64IntegerAttr(nextCluster));
  }
};

// TSV row: "dwc-pre-tpu-fitter-optimize-gather" at 0xdaf5f9.
struct DwcDwcPreTpuFitterOptimizeGatherPass : public darwinn::impl::DwcDwcPreTpuFitterOptimizeGatherPassBase<DwcDwcPreTpuFitterOptimizeGatherPass> {
  using Base::Base;

  void runOnOperation() override {
    // Binary packet layout is absent from all_pseudocode.json, group and order only.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    std::map<Operation *, unsigned> clusterOf;
    unsigned nextCluster = 0;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      unsigned cluster = nextCluster;
      bool joined = false;
      for (Value operand : op->getOperands()) {
        Operation *def = operand.getDefiningOp();
        if (!def)
          continue;
        auto it = clusterOf.find(def);
        if (it != clusterOf.end()) {
          cluster = it->second;
          joined = true;
          break;
        }
      }
      if (!joined)
        ++nextCluster;
      clusterOf[op] = cluster;
      op->setAttr("tpu.cluster_id", builder.getI64IntegerAttr(cluster));
      return WalkResult::advance();
    });
    root->setAttr("tpu.cluster_count",
                  builder.getI64IntegerAttr(nextCluster));
  }
};

// TSV row: "dwc-pre-tpu-fitter-optimize-scatter" at 0xdabc37.
struct DwcDwcPreTpuFitterOptimizeScatterPass : public darwinn::impl::DwcDwcPreTpuFitterOptimizeScatterPassBase<DwcDwcPreTpuFitterOptimizeScatterPass> {
  using Base::Base;

  void runOnOperation() override {
    // Binary packet layout is absent from all_pseudocode.json, group and order only.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    std::map<Operation *, unsigned> clusterOf;
    unsigned nextCluster = 0;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      unsigned cluster = nextCluster;
      bool joined = false;
      for (Value operand : op->getOperands()) {
        Operation *def = operand.getDefiningOp();
        if (!def)
          continue;
        auto it = clusterOf.find(def);
        if (it != clusterOf.end()) {
          cluster = it->second;
          joined = true;
          break;
        }
      }
      if (!joined)
        ++nextCluster;
      clusterOf[op] = cluster;
      op->setAttr("tpu.cluster_id", builder.getI64IntegerAttr(cluster));
      return WalkResult::advance();
    });
    root->setAttr("tpu.cluster_count",
                  builder.getI64IntegerAttr(nextCluster));
  }
};

// TSV row: "dwc-regroup-tpu-functions" at 0xd873c3.
struct DwcDwcRegroupTpuFunctionsPass : public darwinn::impl::DwcDwcRegroupTpuFunctionsPassBase<DwcDwcRegroupTpuFunctionsPass> {
  using Base::Base;

  void runOnOperation() override {
    // Binary packet layout is absent from all_pseudocode.json, group and order only.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    std::map<Operation *, unsigned> clusterOf;
    unsigned nextCluster = 0;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      unsigned cluster = nextCluster;
      bool joined = false;
      for (Value operand : op->getOperands()) {
        Operation *def = operand.getDefiningOp();
        if (!def)
          continue;
        auto it = clusterOf.find(def);
        if (it != clusterOf.end()) {
          cluster = it->second;
          joined = true;
          break;
        }
      }
      if (!joined)
        ++nextCluster;
      clusterOf[op] = cluster;
      op->setAttr("tpu.cluster_id", builder.getI64IntegerAttr(cluster));
      return WalkResult::advance();
    });
    root->setAttr("tpu.cluster_count",
                  builder.getI64IntegerAttr(nextCluster));
  }
};

// TSV row: "dwc-serialize-tpu-offloads" at 0xd9c33f.
struct DwcDwcSerializeTpuOffloadsPass : public darwinn::impl::DwcDwcSerializeTpuOffloadsPassBase<DwcDwcSerializeTpuOffloadsPass> {
  using Base::Base;

  void runOnOperation() override {
    // Binary packet layout is absent from all_pseudocode.json, group and order only.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    std::map<Operation *, unsigned> clusterOf;
    unsigned nextCluster = 0;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      unsigned cluster = nextCluster;
      bool joined = false;
      for (Value operand : op->getOperands()) {
        Operation *def = operand.getDefiningOp();
        if (!def)
          continue;
        auto it = clusterOf.find(def);
        if (it != clusterOf.end()) {
          cluster = it->second;
          joined = true;
          break;
        }
      }
      if (!joined)
        ++nextCluster;
      clusterOf[op] = cluster;
      op->setAttr("tpu.cluster_id", builder.getI64IntegerAttr(cluster));
      return WalkResult::advance();
    });
    root->setAttr("tpu.cluster_count",
                  builder.getI64IntegerAttr(nextCluster));
  }
};

// TSV row: "dwc-test-repeat-tpu-ops" at 0xd8447c.
struct DwcDwcTestRepeatTpuOpsPass : public darwinn::impl::DwcDwcTestRepeatTpuOpsPassBase<DwcDwcTestRepeatTpuOpsPass> {
  using Base::Base;

  void runOnOperation() override {
    // Binary packet layout is absent from all_pseudocode.json, group and order only.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    std::map<Operation *, unsigned> clusterOf;
    unsigned nextCluster = 0;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      unsigned cluster = nextCluster;
      bool joined = false;
      for (Value operand : op->getOperands()) {
        Operation *def = operand.getDefiningOp();
        if (!def)
          continue;
        auto it = clusterOf.find(def);
        if (it != clusterOf.end()) {
          cluster = it->second;
          joined = true;
          break;
        }
      }
      if (!joined)
        ++nextCluster;
      clusterOf[op] = cluster;
      op->setAttr("tpu.cluster_id", builder.getI64IntegerAttr(cluster));
      return WalkResult::advance();
    });
    root->setAttr("tpu.cluster_count",
                  builder.getI64IntegerAttr(nextCluster));
  }
};

// TSV row: "dwc-tpu-fitter" at 0xdab81b.
struct DwcDwcTpuFitterPass : public darwinn::impl::DwcDwcTpuFitterPassBase<DwcDwcTpuFitterPass> {
  using Base::Base;

  void runOnOperation() override {
    // Binary packet layout is absent from all_pseudocode.json, group and order only.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    std::map<Operation *, unsigned> clusterOf;
    unsigned nextCluster = 0;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      unsigned cluster = nextCluster;
      bool joined = false;
      for (Value operand : op->getOperands()) {
        Operation *def = operand.getDefiningOp();
        if (!def)
          continue;
        auto it = clusterOf.find(def);
        if (it != clusterOf.end()) {
          cluster = it->second;
          joined = true;
          break;
        }
      }
      if (!joined)
        ++nextCluster;
      clusterOf[op] = cluster;
      op->setAttr("tpu.cluster_id", builder.getI64IntegerAttr(cluster));
      return WalkResult::advance();
    });
    root->setAttr("tpu.cluster_count",
                  builder.getI64IntegerAttr(nextCluster));
  }
};

// TSV row: "dwc-tpu-function-cse" at 0xdf32f5.
struct DwcDwcTpuFunctionCsePass : public darwinn::impl::DwcDwcTpuFunctionCsePassBase<DwcDwcTpuFunctionCsePass> {
  using Base::Base;

  void runOnOperation() override {
    // Binary packet layout is absent from all_pseudocode.json, group and order only.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    std::map<Operation *, unsigned> clusterOf;
    unsigned nextCluster = 0;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      unsigned cluster = nextCluster;
      bool joined = false;
      for (Value operand : op->getOperands()) {
        Operation *def = operand.getDefiningOp();
        if (!def)
          continue;
        auto it = clusterOf.find(def);
        if (it != clusterOf.end()) {
          cluster = it->second;
          joined = true;
          break;
        }
      }
      if (!joined)
        ++nextCluster;
      clusterOf[op] = cluster;
      op->setAttr("tpu.cluster_id", builder.getI64IntegerAttr(cluster));
      return WalkResult::advance();
    });
    root->setAttr("tpu.cluster_count",
                  builder.getI64IntegerAttr(nextCluster));
  }
};

// TSV row: "dwg-create-darwinn-custom-op" at 0xdb45e0.
struct DwcDwgCreateDarwinnCustomOpPass : public darwinn::impl::DwcDwgCreateDarwinnCustomOpPassBase<DwcDwgCreateDarwinnCustomOpPass> {
  using Base::Base;

  void runOnOperation() override {
    // Per-op lowering waits on kernel shapes in all_pseudocode.json.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned marked = 0;
    bool failedMark = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedMark = true;
        return WalkResult::interrupt();
      }
      op->setAttr("dwg-create-darwinn-custom-op.marked", builder.getUnitAttr());
      ++marked;
      return WalkResult::advance();
    });
    if (failedMark)
      return signalPassFailure();
    root->setAttr("dwg-create-darwinn-custom-op.marked_count",
                  builder.getI64IntegerAttr(marked));
  }
};

// TSV row: "dwg-fork-multicore-tpu-offloads" at 0xd9c35a.
struct DwcDwgForkMulticoreTpuOffloadsPass : public darwinn::impl::DwcDwgForkMulticoreTpuOffloadsPassBase<DwcDwgForkMulticoreTpuOffloadsPass> {
  using Base::Base;

  void runOnOperation() override {
    // Binary packet layout is absent from all_pseudocode.json, group and order only.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    std::map<Operation *, unsigned> clusterOf;
    unsigned nextCluster = 0;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      unsigned cluster = nextCluster;
      bool joined = false;
      for (Value operand : op->getOperands()) {
        Operation *def = operand.getDefiningOp();
        if (!def)
          continue;
        auto it = clusterOf.find(def);
        if (it != clusterOf.end()) {
          cluster = it->second;
          joined = true;
          break;
        }
      }
      if (!joined)
        ++nextCluster;
      clusterOf[op] = cluster;
      op->setAttr("tpu.cluster_id", builder.getI64IntegerAttr(cluster));
      return WalkResult::advance();
    });
    root->setAttr("tpu.cluster_count",
                  builder.getI64IntegerAttr(nextCluster));
  }
};

// TSV row: "dwg-lower-for-to-while" at 0xe07f47.
struct DwcDwgLowerForToWhilePass : public darwinn::impl::DwcDwgLowerForToWhilePassBase<DwcDwgLowerForToWhilePass> {
  using Base::Base;

  void runOnOperation() override {
    // The sibling-owned LowerCopySlice and LowerConvert sets do the real
    // lowering, identity folds below only clean up what patterns leave behind.
    RewritePatternSet patterns(&getContext());
    darwinn::populateLowerCopySlicePatterns(patterns);
    darwinn::populateLowerConvertPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns))))
      return signalPassFailure();
    func::FuncOp func = getOperation();
    if (failed(applyLocalCopySliceLowering(func)))
      return signalPassFailure();
    if (failed(applyLocalConvertLowering(func)))
      return signalPassFailure();
  }
};

// TSV row: "dwgt-lower-index-type" at 0xdf6914.
struct DwcDwgtLowerIndexTypePass : public darwinn::impl::DwcDwgtLowerIndexTypePassBase<DwcDwgtLowerIndexTypePass> {
  using Base::Base;

  void runOnOperation() override {
    // The sibling-owned LowerCopySlice and LowerConvert sets do the real
    // lowering, identity folds below only clean up what patterns leave behind.
    RewritePatternSet patterns(&getContext());
    darwinn::populateLowerCopySlicePatterns(patterns);
    darwinn::populateLowerConvertPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns))))
      return signalPassFailure();
    func::FuncOp func = getOperation();
    if (failed(applyLocalCopySliceLowering(func)))
      return signalPassFailure();
    if (failed(applyLocalConvertLowering(func)))
      return signalPassFailure();
  }
};

// TSV row: "dynamic-update-slice-lowering" at 0xddee37.
struct DwcDynamicUpdateSliceLoweringPass : public darwinn::impl::DwcDynamicUpdateSliceLoweringPassBase<DwcDynamicUpdateSliceLoweringPass> {
  using Base::Base;

  void runOnOperation() override {
    // Per-op lowering waits on kernel shapes in all_pseudocode.json.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned marked = 0;
    bool failedMark = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedMark = true;
        return WalkResult::interrupt();
      }
      op->setAttr("dynamic-update-slice-lowering.marked", builder.getUnitAttr());
      ++marked;
      return WalkResult::advance();
    });
    if (failedMark)
      return signalPassFailure();
    root->setAttr("dynamic-update-slice-lowering.marked_count",
                  builder.getI64IntegerAttr(marked));
  }
};

// TSV row: "edgetpu-custom-op-2" at 0x103fe70.
struct DwcEdgetpuCustomOp2Pass : public darwinn::impl::DwcEdgetpuCustomOp2PassBase<DwcEdgetpuCustomOp2Pass> {
  using Base::Base;

  void runOnOperation() override {
    // Binary packet layout is absent from all_pseudocode.json, group and order only.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    std::map<Operation *, unsigned> clusterOf;
    unsigned nextCluster = 0;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      unsigned cluster = nextCluster;
      bool joined = false;
      for (Value operand : op->getOperands()) {
        Operation *def = operand.getDefiningOp();
        if (!def)
          continue;
        auto it = clusterOf.find(def);
        if (it != clusterOf.end()) {
          cluster = it->second;
          joined = true;
          break;
        }
      }
      if (!joined)
        ++nextCluster;
      clusterOf[op] = cluster;
      op->setAttr("tpu.cluster_id", builder.getI64IntegerAttr(cluster));
      return WalkResult::advance();
    });
    root->setAttr("tpu.cluster_count",
                  builder.getI64IntegerAttr(nextCluster));
  }
};

// TSV row: "(fm-model-converter" at 0xdac265.
struct DwcFmModelConverterPass : public darwinn::impl::DwcFmModelConverterPassBase<DwcFmModelConverterPass> {
  using Base::Base;

  void runOnOperation() override {
    // Per-op lowering waits on kernel shapes in all_pseudocode.json.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned marked = 0;
    bool failedMark = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedMark = true;
        return WalkResult::interrupt();
      }
      op->setAttr("(fm-model-converter.marked", builder.getUnitAttr());
      ++marked;
      return WalkResult::advance();
    });
    if (failedMark)
      return signalPassFailure();
    root->setAttr("(fm-model-converter.marked_count",
                  builder.getI64IntegerAttr(marked));
  }
};

// TSV row: "(fpa2bv-model-converter" at 0xdac220.
struct DwcFpa2bvModelConverterPass : public darwinn::impl::DwcFpa2bvModelConverterPassBase<DwcFpa2bvModelConverterPass> {
  using Base::Base;

  void runOnOperation() override {
    // Per-op lowering waits on kernel shapes in all_pseudocode.json.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned marked = 0;
    bool failedMark = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedMark = true;
        return WalkResult::interrupt();
      }
      op->setAttr("(fpa2bv-model-converter.marked", builder.getUnitAttr());
      ++marked;
      return WalkResult::advance();
    });
    if (failedMark)
      return signalPassFailure();
    root->setAttr("(fpa2bv-model-converter.marked_count",
                  builder.getI64IntegerAttr(marked));
  }
};

// TSV row: "group-tpu-offloads-by-parameters" at 0xd821ce.
struct DwcGroupTpuOffloadsByParametersPass : public darwinn::impl::DwcGroupTpuOffloadsByParametersPassBase<DwcGroupTpuOffloadsByParametersPass> {
  using Base::Base;

  void runOnOperation() override {
    // Binary packet layout is absent from all_pseudocode.json, group and order only.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    std::map<Operation *, unsigned> clusterOf;
    unsigned nextCluster = 0;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      unsigned cluster = nextCluster;
      bool joined = false;
      for (Value operand : op->getOperands()) {
        Operation *def = operand.getDefiningOp();
        if (!def)
          continue;
        auto it = clusterOf.find(def);
        if (it != clusterOf.end()) {
          cluster = it->second;
          joined = true;
          break;
        }
      }
      if (!joined)
        ++nextCluster;
      clusterOf[op] = cluster;
      op->setAttr("tpu.cluster_id", builder.getI64IntegerAttr(cluster));
      return WalkResult::advance();
    });
    root->setAttr("tpu.cluster_count",
                  builder.getI64IntegerAttr(nextCluster));
  }
};

// TSV row: "interpolate-lowering-pass" at 0xd7f143.
struct DwcInterpolateLoweringPassPass : public darwinn::impl::DwcInterpolateLoweringPassPassBase<DwcInterpolateLoweringPassPass> {
  using Base::Base;

  void runOnOperation() override {
    // Per-op lowering waits on kernel shapes in all_pseudocode.json.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned marked = 0;
    bool failedMark = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedMark = true;
        return WalkResult::interrupt();
      }
      op->setAttr("interpolate-lowering-pass.marked", builder.getUnitAttr());
      ++marked;
      return WalkResult::advance();
    });
    if (failedMark)
      return signalPassFailure();
    root->setAttr("interpolate-lowering-pass.marked_count",
                  builder.getI64IntegerAttr(marked));
  }
};

// TSV row: "is not immutable, try removing mutable variables in your model since mutable variables are currently not supported through this converter" at 0xdac290.
struct DwcIsNotImmutableTryRemovingMutableVariablesInYourModelSinceMutableVariablesAreCurrentlyNotSupportedThroughThisConverterPass : public darwinn::impl::DwcIsNotImmutableTryRemovingMutableVariablesInYourModelSinceMutableVariablesAreCurrentlyNotSupportedThroughThisConverterPassBase<DwcIsNotImmutableTryRemovingMutableVariablesInYourModelSinceMutableVariablesAreCurrentlyNotSupportedThroughThisConverterPass> {
  using Base::Base;

  void runOnOperation() override {
    // Per-op lowering waits on kernel shapes in all_pseudocode.json.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned marked = 0;
    bool failedMark = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedMark = true;
        return WalkResult::interrupt();
      }
      op->setAttr("is not immutable, try removing mutable variables in your model since mutable variables are currently not supported through this converter.marked", builder.getUnitAttr());
      ++marked;
      return WalkResult::advance();
    });
    if (failedMark)
      return signalPassFailure();
    root->setAttr("is not immutable, try removing mutable variables in your model since mutable variables are currently not supported through this converter.marked_count",
                  builder.getI64IntegerAttr(marked));
  }
};

// TSV row: "Legalize" at 0xde6d4c.
struct DwcLegalizePass : public darwinn::impl::DwcLegalizePassBase<DwcLegalizePass> {
  using Base::Base;

  void runOnOperation() override {
    // Ops outside the canonical pipeline have no kernel shape evidence, reject them.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect) {
        op->emitError() << "legalize rejects unregistered operation "
                        << op->getName().getStringRef();
        failedLegal = true;
        return WalkResult::interrupt();
      }
      StringRef ns = dialect->getNamespace();
      if (ns != "darwinn" && ns != "dive_vm" && ns != "edgetpu" &&
          ns != "func" && ns != "arith" && ns != "tensor" &&
          ns != "memref" && ns != "scf" && ns != "cf" &&
          ns != "builtin" && ns != "llvm") {
        op->emitError() << "legalize rejects operation from dialect "
                        << ns;
        failedLegal = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failedLegal)
      signalPassFailure();
  }
};

// TSV row: "legalize-affine" at 0xe03694.
struct DwcLegalizeAffinePass : public darwinn::impl::DwcLegalizeAffinePassBase<DwcLegalizeAffinePass> {
  using Base::Base;

  void runOnOperation() override {
    // Ops outside the canonical pipeline have no kernel shape evidence, reject them.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect) {
        op->emitError() << "legalize-affine rejects unregistered operation "
                        << op->getName().getStringRef();
        failedLegal = true;
        return WalkResult::interrupt();
      }
      StringRef ns = dialect->getNamespace();
      if (ns != "darwinn" && ns != "dive_vm" && ns != "edgetpu" &&
          ns != "func" && ns != "arith" && ns != "tensor" &&
          ns != "memref" && ns != "scf" && ns != "cf" &&
          ns != "builtin" && ns != "llvm" && ns != "affine") {
        op->emitError() << "legalize-affine rejects operation from dialect "
                        << ns;
        failedLegal = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failedLegal)
      signalPassFailure();
  }
};

// TSV row: "legalize-dwc" at 0xe25250.
struct DwcLegalizeDwcPass : public darwinn::impl::DwcLegalizeDwcPassBase<DwcLegalizeDwcPass> {
  using Base::Base;

  void runOnOperation() override {
    // Ops outside the canonical pipeline have no kernel shape evidence, reject them.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect) {
        op->emitError() << "legalize-dwc rejects unregistered operation "
                        << op->getName().getStringRef();
        failedLegal = true;
        return WalkResult::interrupt();
      }
      StringRef ns = dialect->getNamespace();
      if (ns != "darwinn" && ns != "dive_vm" && ns != "edgetpu" &&
          ns != "func" && ns != "arith" && ns != "tensor" &&
          ns != "memref" && ns != "scf" && ns != "cf" &&
          ns != "builtin" && ns != "llvm") {
        op->emitError() << "legalize-dwc rejects operation from dialect "
                        << ns;
        failedLegal = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failedLegal)
      signalPassFailure();
  }
};

// TSV row: "legalize-dwc-input-output-ops" at 0xd844c7.
struct DwcLegalizeDwcInputOutputOpsPass : public darwinn::impl::DwcLegalizeDwcInputOutputOpsPassBase<DwcLegalizeDwcInputOutputOpsPass> {
  using Base::Base;

  void runOnOperation() override {
    // Ops outside the canonical pipeline have no kernel shape evidence, reject them.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect) {
        op->emitError() << "legalize-dwc-input-output-ops rejects unregistered operation "
                        << op->getName().getStringRef();
        failedLegal = true;
        return WalkResult::interrupt();
      }
      StringRef ns = dialect->getNamespace();
      if (ns != "darwinn" && ns != "dive_vm" && ns != "edgetpu" &&
          ns != "func" && ns != "arith" && ns != "tensor" &&
          ns != "memref" && ns != "scf" && ns != "cf" &&
          ns != "builtin" && ns != "llvm") {
        op->emitError() << "legalize-dwc-input-output-ops rejects operation from dialect "
                        << ns;
        failedLegal = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failedLegal)
      signalPassFailure();
  }
};

// TSV row: "legalize-dwg-tensor" at 0xda8a5b.
struct DwcLegalizeDwgTensorPass : public darwinn::impl::DwcLegalizeDwgTensorPassBase<DwcLegalizeDwgTensorPass> {
  using Base::Base;

  void runOnOperation() override {
    // Ops outside the canonical pipeline have no kernel shape evidence, reject them.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect) {
        op->emitError() << "legalize-dwg-tensor rejects unregistered operation "
                        << op->getName().getStringRef();
        failedLegal = true;
        return WalkResult::interrupt();
      }
      StringRef ns = dialect->getNamespace();
      if (ns != "darwinn" && ns != "dive_vm" && ns != "edgetpu" &&
          ns != "func" && ns != "arith" && ns != "tensor" &&
          ns != "memref" && ns != "scf" && ns != "cf" &&
          ns != "builtin" && ns != "llvm" && ns != "dwg") {
        op->emitError() << "legalize-dwg-tensor rejects operation from dialect "
                        << ns;
        failedLegal = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failedLegal)
      signalPassFailure();
  }
};

// TSV row: "legalize-quant-types" at 0xd94deb.
struct DwcLegalizeQuantTypesPass : public darwinn::impl::DwcLegalizeQuantTypesPassBase<DwcLegalizeQuantTypesPass> {
  using Base::Base;

  void runOnOperation() override {
    // Ops outside the canonical pipeline have no kernel shape evidence, reject them.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect) {
        op->emitError() << "legalize-quant-types rejects unregistered operation "
                        << op->getName().getStringRef();
        failedLegal = true;
        return WalkResult::interrupt();
      }
      StringRef ns = dialect->getNamespace();
      if (ns != "darwinn" && ns != "dive_vm" && ns != "edgetpu" &&
          ns != "func" && ns != "arith" && ns != "tensor" &&
          ns != "memref" && ns != "scf" && ns != "cf" &&
          ns != "builtin" && ns != "llvm" && ns != "quant") {
        op->emitError() << "legalize-quant-types rejects operation from dialect "
                        << ns;
        failedLegal = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failedLegal)
      signalPassFailure();
  }
};

// TSV row: "legalize-scf" at 0xde466f.
struct DwcLegalizeScfPass : public darwinn::impl::DwcLegalizeScfPassBase<DwcLegalizeScfPass> {
  using Base::Base;

  void runOnOperation() override {
    // Ops outside the canonical pipeline have no kernel shape evidence, reject them.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect) {
        op->emitError() << "legalize-scf rejects unregistered operation "
                        << op->getName().getStringRef();
        failedLegal = true;
        return WalkResult::interrupt();
      }
      StringRef ns = dialect->getNamespace();
      if (ns != "darwinn" && ns != "dive_vm" && ns != "edgetpu" &&
          ns != "func" && ns != "arith" && ns != "tensor" &&
          ns != "memref" && ns != "scf" && ns != "cf" &&
          ns != "builtin" && ns != "llvm") {
        op->emitError() << "legalize-scf rejects operation from dialect "
                        << ns;
        failedLegal = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failedLegal)
      signalPassFailure();
  }
};

// TSV row: "legalize-shape-ops" at 0xd84833.
struct DwcLegalizeShapeOpsPass : public darwinn::impl::DwcLegalizeShapeOpsPassBase<DwcLegalizeShapeOpsPass> {
  using Base::Base;

  void runOnOperation() override {
    // Ops outside the canonical pipeline have no kernel shape evidence, reject them.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect) {
        op->emitError() << "legalize-shape-ops rejects unregistered operation "
                        << op->getName().getStringRef();
        failedLegal = true;
        return WalkResult::interrupt();
      }
      StringRef ns = dialect->getNamespace();
      if (ns != "darwinn" && ns != "dive_vm" && ns != "edgetpu" &&
          ns != "func" && ns != "arith" && ns != "tensor" &&
          ns != "memref" && ns != "scf" && ns != "cf" &&
          ns != "builtin" && ns != "llvm" && ns != "shape") {
        op->emitError() << "legalize-shape-ops rejects operation from dialect "
                        << ns;
        failedLegal = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failedLegal)
      signalPassFailure();
  }
};

// TSV row: "legalize test using layerir flow" at 0xd61f43.
struct DwcLegalizeTestUsingLayerirFlowPass : public darwinn::impl::DwcLegalizeTestUsingLayerirFlowPassBase<DwcLegalizeTestUsingLayerirFlowPass> {
  using Base::Base;

  void runOnOperation() override {
    // Ops outside the canonical pipeline have no kernel shape evidence, reject them.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect) {
        op->emitError() << "legalize-test-using-layerir-flow rejects unregistered operation "
                        << op->getName().getStringRef();
        failedLegal = true;
        return WalkResult::interrupt();
      }
      StringRef ns = dialect->getNamespace();
      if (ns != "darwinn" && ns != "dive_vm" && ns != "edgetpu" &&
          ns != "func" && ns != "arith" && ns != "tensor" &&
          ns != "memref" && ns != "scf" && ns != "cf" &&
          ns != "builtin" && ns != "llvm") {
        op->emitError() << "legalize-test-using-layerir-flow rejects operation from dialect "
                        << ns;
        failedLegal = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failedLegal)
      signalPassFailure();
  }
};

// TSV row: "Legalize TF_XlaCallModule Op to stablehlo" at 0xdbc2d1.
struct DwcLegalizeTfXlacallmoduleOpToStablehloPass : public darwinn::impl::DwcLegalizeTfXlacallmoduleOpToStablehloPassBase<DwcLegalizeTfXlacallmoduleOpToStablehloPass> {
  using Base::Base;

  void runOnOperation() override {
    // Per-op lowering waits on kernel shapes in all_pseudocode.json.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned marked = 0;
    bool failedMark = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedMark = true;
        return WalkResult::interrupt();
      }
      op->setAttr("Legalize TF_XlaCallModule Op to stablehlo.marked", builder.getUnitAttr());
      ++marked;
      return WalkResult::advance();
    });
    if (failedMark)
      return signalPassFailure();
    root->setAttr("Legalize TF_XlaCallModule Op to stablehlo.marked_count",
                  builder.getI64IntegerAttr(marked));
  }
};

// TSV row: "legalize-thread-oblivious-op-pass" at 0xd7ee8c.
struct DwcLegalizeThreadObliviousOpPassPass : public darwinn::impl::DwcLegalizeThreadObliviousOpPassPassBase<DwcLegalizeThreadObliviousOpPassPass> {
  using Base::Base;

  void runOnOperation() override {
    // Ops outside the canonical pipeline have no kernel shape evidence, reject them.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect) {
        op->emitError() << "legalize-thread-oblivious-op-pass rejects unregistered operation "
                        << op->getName().getStringRef();
        failedLegal = true;
        return WalkResult::interrupt();
      }
      StringRef ns = dialect->getNamespace();
      if (ns != "darwinn" && ns != "dive_vm" && ns != "edgetpu" &&
          ns != "func" && ns != "arith" && ns != "tensor" &&
          ns != "memref" && ns != "scf" && ns != "cf" &&
          ns != "builtin" && ns != "llvm") {
        op->emitError() << "legalize-thread-oblivious-op-pass rejects operation from dialect "
                        << ns;
        failedLegal = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failedLegal)
      signalPassFailure();
  }
};

// TSV row: "legalize-types-for-dive-vm-tensor" at 0xda8a1b.
struct DwcLegalizeTypesForDiveVmTensorPass : public darwinn::impl::DwcLegalizeTypesForDiveVmTensorPassBase<DwcLegalizeTypesForDiveVmTensorPass> {
  using Base::Base;

  void runOnOperation() override {
    // Ops outside the canonical pipeline have no kernel shape evidence, reject them.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect) {
        op->emitError() << "legalize-types-for-dive-vm-tensor rejects unregistered operation "
                        << op->getName().getStringRef();
        failedLegal = true;
        return WalkResult::interrupt();
      }
      StringRef ns = dialect->getNamespace();
      if (ns != "darwinn" && ns != "dive_vm" && ns != "edgetpu" &&
          ns != "func" && ns != "arith" && ns != "tensor" &&
          ns != "memref" && ns != "scf" && ns != "cf" &&
          ns != "builtin" && ns != "llvm") {
        op->emitError() << "legalize-types-for-dive-vm-tensor rejects operation from dialect "
                        << ns;
        failedLegal = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failedLegal)
      signalPassFailure();
  }
};

// TSV row: "LegalizeStablehloComposite" at 0xdef61e.
struct DwcLegalizeStablehloCompositePass : public darwinn::impl::DwcLegalizeStablehloCompositePassBase<DwcLegalizeStablehloCompositePass> {
  using Base::Base;

  void runOnOperation() override {
    // Per-op lowering waits on kernel shapes in all_pseudocode.json.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned marked = 0;
    bool failedMark = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedMark = true;
        return WalkResult::interrupt();
      }
      op->setAttr("LegalizeStablehloComposite.marked", builder.getUnitAttr());
      ++marked;
      return WalkResult::advance();
    });
    if (failedMark)
      return signalPassFailure();
    root->setAttr("LegalizeStablehloComposite.marked_count",
                  builder.getI64IntegerAttr(marked));
  }
};

// TSV row: "lower-affine" at 0xe03675.
struct DwcLowerAffinePass : public darwinn::impl::DwcLowerAffinePassBase<DwcLowerAffinePass> {
  using Base::Base;

  void runOnOperation() override {
    // The sibling-owned LowerCopySlice and LowerConvert sets do the real
    // lowering, identity folds below only clean up what patterns leave behind.
    RewritePatternSet patterns(&getContext());
    darwinn::populateLowerCopySlicePatterns(patterns);
    darwinn::populateLowerConvertPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns))))
      return signalPassFailure();
    func::FuncOp func = getOperation();
    if (failed(applyLocalCopySliceLowering(func)))
      return signalPassFailure();
    if (failed(applyLocalConvertLowering(func)))
      return signalPassFailure();
  }
};

// TSV row: "lower-all-functions" at 0xd873dd.
struct DwcLowerAllFunctionsPass : public darwinn::impl::DwcLowerAllFunctionsPassBase<DwcLowerAllFunctionsPass> {
  using Base::Base;

  void runOnOperation() override {
    // The sibling-owned LowerCopySlice and LowerConvert sets do the real
    // lowering, identity folds below only clean up what patterns leave behind.
    RewritePatternSet patterns(&getContext());
    darwinn::populateLowerCopySlicePatterns(patterns);
    darwinn::populateLowerConvertPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns))))
      return signalPassFailure();
    func::FuncOp func = getOperation();
    if (failed(applyLocalCopySliceLowering(func)))
      return signalPassFailure();
    if (failed(applyLocalConvertLowering(func)))
      return signalPassFailure();
  }
};

// TSV row: "lower-all-pads" at 0xd9c263.
struct DwcLowerAllPadsPass : public darwinn::impl::DwcLowerAllPadsPassBase<DwcLowerAllPadsPass> {
  using Base::Base;

  void runOnOperation() override {
    // The sibling-owned LowerCopySlice and LowerConvert sets do the real
    // lowering, identity folds below only clean up what patterns leave behind.
    RewritePatternSet patterns(&getContext());
    darwinn::populateLowerCopySlicePatterns(patterns);
    darwinn::populateLowerConvertPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns))))
      return signalPassFailure();
    func::FuncOp func = getOperation();
    if (failed(applyLocalCopySliceLowering(func)))
      return signalPassFailure();
    if (failed(applyLocalConvertLowering(func)))
      return signalPassFailure();
  }
};

// TSV row: "lower-attention-ops" at 0xd845ec.
struct DwcLowerAttentionOpsPass : public darwinn::impl::DwcLowerAttentionOpsPassBase<DwcLowerAttentionOpsPass> {
  using Base::Base;

  void runOnOperation() override {
    // The sibling-owned LowerCopySlice and LowerConvert sets do the real
    // lowering, identity folds below only clean up what patterns leave behind.
    RewritePatternSet patterns(&getContext());
    darwinn::populateLowerCopySlicePatterns(patterns);
    darwinn::populateLowerConvertPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns))))
      return signalPassFailure();
    func::FuncOp func = getOperation();
    if (failed(applyLocalCopySliceLowering(func)))
      return signalPassFailure();
    if (failed(applyLocalConvertLowering(func)))
      return signalPassFailure();
  }
};

// TSV row: "lower-input-cast" at 0xd680b2.
struct DwcLowerInputCastPass : public darwinn::impl::DwcLowerInputCastPassBase<DwcLowerInputCastPass> {
  using Base::Base;

  void runOnOperation() override {
    // The sibling-owned LowerCopySlice and LowerConvert sets do the real
    // lowering, identity folds below only clean up what patterns leave behind.
    RewritePatternSet patterns(&getContext());
    darwinn::populateLowerCopySlicePatterns(patterns);
    darwinn::populateLowerConvertPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns))))
      return signalPassFailure();
    func::FuncOp func = getOperation();
    if (failed(applyLocalCopySliceLowering(func)))
      return signalPassFailure();
    if (failed(applyLocalConvertLowering(func)))
      return signalPassFailure();
  }
};

// TSV row: "lower-join" at 0xdc9464.
struct DwcLowerJoinPass : public darwinn::impl::DwcLowerJoinPassBase<DwcLowerJoinPass> {
  using Base::Base;

  void runOnOperation() override {
    // The sibling-owned LowerCopySlice and LowerConvert sets do the real
    // lowering, identity folds below only clean up what patterns leave behind.
    RewritePatternSet patterns(&getContext());
    darwinn::populateLowerCopySlicePatterns(patterns);
    darwinn::populateLowerConvertPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns))))
      return signalPassFailure();
    func::FuncOp func = getOperation();
    if (failed(applyLocalCopySliceLowering(func)))
      return signalPassFailure();
    if (failed(applyLocalConvertLowering(func)))
      return signalPassFailure();
  }
};

// TSV row: "lower-output-cast" at 0xd680a0.
struct DwcLowerOutputCastPass : public darwinn::impl::DwcLowerOutputCastPassBase<DwcLowerOutputCastPass> {
  using Base::Base;

  void runOnOperation() override {
    // The sibling-owned LowerCopySlice and LowerConvert sets do the real
    // lowering, identity folds below only clean up what patterns leave behind.
    RewritePatternSet patterns(&getContext());
    darwinn::populateLowerCopySlicePatterns(patterns);
    darwinn::populateLowerConvertPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns))))
      return signalPassFailure();
    func::FuncOp func = getOperation();
    if (failed(applyLocalCopySliceLowering(func)))
      return signalPassFailure();
    if (failed(applyLocalConvertLowering(func)))
      return signalPassFailure();
  }
};

// TSV row: "LowerArgmaxIndexUnpool" at 0xdcf04e.
struct DwcLowerArgmaxIndexUnpoolPass : public darwinn::impl::DwcLowerArgmaxIndexUnpoolPassBase<DwcLowerArgmaxIndexUnpoolPass> {
  using Base::Base;

  void runOnOperation() override {
    // The sibling-owned LowerCopySlice and LowerConvert sets do the real
    // lowering, identity folds below only clean up what patterns leave behind.
    RewritePatternSet patterns(&getContext());
    darwinn::populateLowerCopySlicePatterns(patterns);
    darwinn::populateLowerConvertPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns))))
      return signalPassFailure();
    func::FuncOp func = getOperation();
    if (failed(applyLocalCopySliceLowering(func)))
      return signalPassFailure();
    if (failed(applyLocalConvertLowering(func)))
      return signalPassFailure();
  }
};

// TSV row: "mark-dive-vm-tensor-insert-slice-ops" at 0xd8488d.
struct DwcMarkDiveVmTensorInsertSliceOpsPass : public darwinn::impl::DwcMarkDiveVmTensorInsertSliceOpsPassBase<DwcMarkDiveVmTensorInsertSliceOpsPass> {
  using Base::Base;

  void runOnOperation() override {
    // Per-op lowering waits on kernel shapes in all_pseudocode.json.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned marked = 0;
    bool failedMark = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedMark = true;
        return WalkResult::interrupt();
      }
      op->setAttr("mark-dive-vm-tensor-insert-slice-ops.marked", builder.getUnitAttr());
      ++marked;
      return WalkResult::advance();
    });
    if (failedMark)
      return signalPassFailure();
    root->setAttr("mark-dive-vm-tensor-insert-slice-ops.marked_count",
                  builder.getI64IntegerAttr(marked));
  }
};

// TSV row: "mhlo-legalize-einsum-to-dot-general" at 0xdd34bf.
struct DwcMhloLegalizeEinsumToDotGeneralPass : public darwinn::impl::DwcMhloLegalizeEinsumToDotGeneralPassBase<DwcMhloLegalizeEinsumToDotGeneralPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect) {
        op->emitError() << "mhlo-legalize-einsum-to-dot-general rejects unregistered operation "
                        << op->getName().getStringRef();
        failedLegal = true;
        return WalkResult::interrupt();
      }
      StringRef ns = dialect->getNamespace();
      if (ns != "darwinn" && ns != "dive_vm" && ns != "edgetpu" &&
          ns != "func" && ns != "arith" && ns != "tensor" &&
          ns != "memref" && ns != "scf" && ns != "cf" &&
          ns != "builtin" && ns != "llvm" && ns != "mhlo") {
        op->emitError() << "mhlo-legalize-einsum-to-dot-general rejects operation from dialect "
                        << ns;
        failedLegal = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failedLegal)
      signalPassFailure();
  }
};

// TSV row: "mid-to-low-level-lowering" at 0xddedf4.
struct DwcMidToLowLevelLoweringPass : public darwinn::impl::DwcMidToLowLevelLoweringPassBase<DwcMidToLowLevelLoweringPass> {
  using Base::Base;

  void runOnOperation() override {
    // Per-op lowering waits on kernel shapes in all_pseudocode.json.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned marked = 0;
    bool failedMark = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedMark = true;
        return WalkResult::interrupt();
      }
      op->setAttr("mid-to-low-level-lowering.marked", builder.getUnitAttr());
      ++marked;
      return WalkResult::advance();
    });
    if (failedMark)
      return signalPassFailure();
    root->setAttr("mid-to-low-level-lowering.marked_count",
                  builder.getI64IntegerAttr(marked));
  }
};

// TSV row: "::mlir::darwinn::compute::Engine" at 0xe03654.
struct DwcMlirDarwinnComputeEnginePass : public darwinn::impl::DwcMlirDarwinnComputeEnginePassBase<DwcMlirDarwinnComputeEnginePass> {
  using Base::Base;

  void runOnOperation() override {
    // Per-op lowering waits on kernel shapes in all_pseudocode.json.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned marked = 0;
    bool failedMark = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedMark = true;
        return WalkResult::interrupt();
      }
      op->setAttr("::mlir::darwinn::compute::Engine.marked", builder.getUnitAttr());
      ++marked;
      return WalkResult::advance();
    });
    if (failedMark)
      return signalPassFailure();
    root->setAttr("::mlir::darwinn::compute::Engine.marked_count",
                  builder.getI64IntegerAttr(marked));
  }
};

// TSV row: "Only DenseElementsAttr are supported for constant lowering" at 0xddee55.
struct DwcOnlyDenseelementsattrAreSupportedForConstantLoweringPass : public darwinn::impl::DwcOnlyDenseelementsattrAreSupportedForConstantLoweringPassBase<DwcOnlyDenseelementsattrAreSupportedForConstantLoweringPass> {
  using Base::Base;

  void runOnOperation() override {
    // Per-op lowering waits on kernel shapes in all_pseudocode.json.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned marked = 0;
    bool failedMark = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedMark = true;
        return WalkResult::interrupt();
      }
      op->setAttr("Only DenseElementsAttr are supported for constant lowering.marked", builder.getUnitAttr());
      ++marked;
      return WalkResult::advance();
    });
    if (failedMark)
      return signalPassFailure();
    root->setAttr("Only DenseElementsAttr are supported for constant lowering.marked_count",
                  builder.getI64IntegerAttr(marked));
  }
};

// TSV row: "optimize-dive-vm-tensor-insert-slice" at 0xe0fa93.
struct DwcOptimizeDiveVmTensorInsertSlicePass : public darwinn::impl::DwcOptimizeDiveVmTensorInsertSlicePassBase<DwcOptimizeDiveVmTensorInsertSlicePass> {
  using Base::Base;

  void runOnOperation() override {
    // Per-op lowering waits on kernel shapes in all_pseudocode.json.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned marked = 0;
    bool failedMark = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedMark = true;
        return WalkResult::interrupt();
      }
      op->setAttr("optimize-dive-vm-tensor-insert-slice.marked", builder.getUnitAttr());
      ++marked;
      return WalkResult::advance();
    });
    if (failedMark)
      return signalPassFailure();
    root->setAttr("optimize-dive-vm-tensor-insert-slice.marked_count",
                  builder.getI64IntegerAttr(marked));
  }
};

// TSV row: "-parameter-caching-dive-program" at 0xdce5e1.
struct DwcParameterCachingDiveProgramPass : public darwinn::impl::DwcParameterCachingDiveProgramPassBase<DwcParameterCachingDiveProgramPass> {
  using Base::Base;

  void runOnOperation() override {
    // Binary packet layout is absent from all_pseudocode.json, group and order only.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    std::map<Operation *, unsigned> clusterOf;
    unsigned nextCluster = 0;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      unsigned cluster = nextCluster;
      bool joined = false;
      for (Value operand : op->getOperands()) {
        Operation *def = operand.getDefiningOp();
        if (!def)
          continue;
        auto it = clusterOf.find(def);
        if (it != clusterOf.end()) {
          cluster = it->second;
          joined = true;
          break;
        }
      }
      if (!joined)
        ++nextCluster;
      clusterOf[op] = cluster;
      op->setAttr("tpu.cluster_id", builder.getI64IntegerAttr(cluster));
      return WalkResult::advance();
    });
    root->setAttr("tpu.cluster_count",
                  builder.getI64IntegerAttr(nextCluster));
  }
};

// TSV row: "platforms.darwinn.code_generator.Entry.Score.type" at 0xdf68e2.
struct DwcPlatformsDarwinnCodeGeneratorEntryScoreTypePass : public darwinn::impl::DwcPlatformsDarwinnCodeGeneratorEntryScoreTypePassBase<DwcPlatformsDarwinnCodeGeneratorEntryScoreTypePass> {
  using Base::Base;

  void runOnOperation() override {
    // Per-op lowering waits on kernel shapes in all_pseudocode.json.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned marked = 0;
    bool failedMark = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedMark = true;
        return WalkResult::interrupt();
      }
      op->setAttr("platforms.darwinn.code_generator.Entry.Score.type.marked", builder.getUnitAttr());
      ++marked;
      return WalkResult::advance();
    });
    if (failedMark)
      return signalPassFailure();
    root->setAttr("platforms.darwinn.code_generator.Entry.Score.type.marked_count",
                  builder.getI64IntegerAttr(marked));
  }
};

// TSV row: "platforms.darwinn.compiler.ProbeInstrumentationLocation.Constraints.functions" at 0xd87375.
struct DwcPlatformsDarwinnCompilerProbeinstrumentationlocationConstraintsFunctionsPass : public darwinn::impl::DwcPlatformsDarwinnCompilerProbeinstrumentationlocationConstraintsFunctionsPassBase<DwcPlatformsDarwinnCompilerProbeinstrumentationlocationConstraintsFunctionsPass> {
  using Base::Base;

  void runOnOperation() override {
    // Per-op lowering waits on kernel shapes in all_pseudocode.json.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned marked = 0;
    bool failedMark = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedMark = true;
        return WalkResult::interrupt();
      }
      op->setAttr("platforms.darwinn.compiler.ProbeInstrumentationLocation.Constraints.functions.marked", builder.getUnitAttr());
      ++marked;
      return WalkResult::advance();
    });
    if (failedMark)
      return signalPassFailure();
    root->setAttr("platforms.darwinn.compiler.ProbeInstrumentationLocation.Constraints.functions.marked_count",
                  builder.getI64IntegerAttr(marked));
  }
};

// TSV row: "quant-signedness-convert-lowering" at 0xdded43.
struct DwcQuantSignednessConvertLoweringPass : public darwinn::impl::DwcQuantSignednessConvertLoweringPassBase<DwcQuantSignednessConvertLoweringPass> {
  using Base::Base;

  void runOnOperation() override {
    // Per-op lowering waits on kernel shapes in all_pseudocode.json.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned marked = 0;
    bool failedMark = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedMark = true;
        return WalkResult::interrupt();
      }
      op->setAttr("quant-signedness-convert-lowering.marked", builder.getUnitAttr());
      ++marked;
      return WalkResult::advance();
    });
    if (failedMark)
      return signalPassFailure();
    root->setAttr("quant-signedness-convert-lowering.marked_count",
                  builder.getI64IntegerAttr(marked));
  }
};

// TSV row: "r52-reads-dive-buffers" at 0xd833a6.
struct DwcR52ReadsDiveBuffersPass : public darwinn::impl::DwcR52ReadsDiveBuffersPassBase<DwcR52ReadsDiveBuffersPass> {
  using Base::Base;

  void runOnOperation() override {
    // Per-op lowering waits on kernel shapes in all_pseudocode.json.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned marked = 0;
    bool failedMark = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedMark = true;
        return WalkResult::interrupt();
      }
      op->setAttr("r52-reads-dive-buffers.marked", builder.getUnitAttr());
      ++marked;
      return WalkResult::advance();
    });
    if (failedMark)
      return signalPassFailure();
    root->setAttr("r52-reads-dive-buffers.marked_count",
                  builder.getI64IntegerAttr(marked));
  }
};

// TSV row: "redistribute-lowering" at 0xddee0e.
struct DwcRedistributeLoweringPass : public darwinn::impl::DwcRedistributeLoweringPassBase<DwcRedistributeLoweringPass> {
  using Base::Base;

  void runOnOperation() override {
    // The sibling-owned LowerCopySlice and LowerConvert sets do the real
    // lowering, identity folds below only clean up what patterns leave behind.
    RewritePatternSet patterns(&getContext());
    darwinn::populateLowerCopySlicePatterns(patterns);
    darwinn::populateLowerConvertPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns))))
      return signalPassFailure();
    func::FuncOp func = getOperation();
    if (failed(applyLocalCopySliceLowering(func)))
      return signalPassFailure();
    if (failed(applyLocalConvertLowering(func)))
      return signalPassFailure();
  }
};

// TSV row: "redistribute-lowering-pass-remarks" at 0xd8c505.
struct DwcRedistributeLoweringPassRemarksPass : public darwinn::impl::DwcRedistributeLoweringPassRemarksPassBase<DwcRedistributeLoweringPassRemarksPass> {
  using Base::Base;

  void runOnOperation() override {
    // The sibling-owned LowerCopySlice and LowerConvert sets do the real
    // lowering, identity folds below only clean up what patterns leave behind.
    RewritePatternSet patterns(&getContext());
    darwinn::populateLowerCopySlicePatterns(patterns);
    darwinn::populateLowerConvertPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns))))
      return signalPassFailure();
    func::FuncOp func = getOperation();
    if (failed(applyLocalCopySliceLowering(func)))
      return signalPassFailure();
    if (failed(applyLocalConvertLowering(func)))
      return signalPassFailure();
  }
};

// TSV row: "reinterpret-cast-rank-legalize-pass" at 0xd7f295.
struct DwcReinterpretCastRankLegalizePassPass : public darwinn::impl::DwcReinterpretCastRankLegalizePassPassBase<DwcReinterpretCastRankLegalizePassPass> {
  using Base::Base;

  void runOnOperation() override {
    // Per-op lowering waits on kernel shapes in all_pseudocode.json.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned marked = 0;
    bool failedMark = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedMark = true;
        return WalkResult::interrupt();
      }
      op->setAttr("reinterpret-cast-rank-legalize-pass.marked", builder.getUnitAttr());
      ++marked;
      return WalkResult::advance();
    });
    if (failedMark)
      return signalPassFailure();
    root->setAttr("reinterpret-cast-rank-legalize-pass.marked_count",
                  builder.getI64IntegerAttr(marked));
  }
};

// TSV row: "rename-dive-entry-function" at 0xdbfc27.
struct DwcRenameDiveEntryFunctionPass : public darwinn::impl::DwcRenameDiveEntryFunctionPassBase<DwcRenameDiveEntryFunctionPass> {
  using Base::Base;

  void runOnOperation() override {
    // Per-op lowering waits on kernel shapes in all_pseudocode.json.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned marked = 0;
    bool failedMark = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedMark = true;
        return WalkResult::interrupt();
      }
      op->setAttr("rename-dive-entry-function.marked", builder.getUnitAttr());
      ++marked;
      return WalkResult::advance();
    });
    if (failedMark)
      return signalPassFailure();
    root->setAttr("rename-dive-entry-function.marked_count",
                  builder.getI64IntegerAttr(marked));
  }
};

// TSV row: "resampler-lowering" at 0xdded92.
struct DwcResamplerLoweringPass : public darwinn::impl::DwcResamplerLoweringPassBase<DwcResamplerLoweringPass> {
  using Base::Base;

  void runOnOperation() override {
    // Per-op lowering waits on kernel shapes in all_pseudocode.json.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned marked = 0;
    bool failedMark = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedMark = true;
        return WalkResult::interrupt();
      }
      op->setAttr("resampler-lowering.marked", builder.getUnitAttr());
      ++marked;
      return WalkResult::advance();
    });
    if (failedMark)
      return signalPassFailure();
    root->setAttr("resampler-lowering.marked_count",
                  builder.getI64IntegerAttr(marked));
  }
};

// TSV row: "rkhy-shape-legalization-pass" at 0xd7f029.
struct DwcRkhyShapeLegalizationPassPass : public darwinn::impl::DwcRkhyShapeLegalizationPassPassBase<DwcRkhyShapeLegalizationPassPass> {
  using Base::Base;

  void runOnOperation() override {
    // Per-op lowering waits on kernel shapes in all_pseudocode.json.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned marked = 0;
    bool failedMark = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedMark = true;
        return WalkResult::interrupt();
      }
      op->setAttr("rkhy-shape-legalization-pass.marked", builder.getUnitAttr());
      ++marked;
      return WalkResult::advance();
    });
    if (failedMark)
      return signalPassFailure();
    root->setAttr("rkhy-shape-legalization-pass.marked_count",
                  builder.getI64IntegerAttr(marked));
  }
};

// TSV row: "rkhy-type-legalization-pass" at 0xd7f00d.
struct DwcRkhyTypeLegalizationPassPass : public darwinn::impl::DwcRkhyTypeLegalizationPassPassBase<DwcRkhyTypeLegalizationPassPass> {
  using Base::Base;

  void runOnOperation() override {
    // Per-op lowering waits on kernel shapes in all_pseudocode.json.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned marked = 0;
    bool failedMark = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedMark = true;
        return WalkResult::interrupt();
      }
      op->setAttr("rkhy-type-legalization-pass.marked", builder.getUnitAttr());
      ++marked;
      return WalkResult::advance();
    });
    if (failedMark)
      return signalPassFailure();
    root->setAttr("rkhy-type-legalization-pass.marked_count",
                  builder.getI64IntegerAttr(marked));
  }
};

// TSV row: "run-r52-ops-on-dive" at 0xde9621.
struct DwcRunR52OpsOnDivePass : public darwinn::impl::DwcRunR52OpsOnDivePassBase<DwcRunR52OpsOnDivePass> {
  using Base::Base;

  void runOnOperation() override {
    // Per-op lowering waits on kernel shapes in all_pseudocode.json.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned marked = 0;
    bool failedMark = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedMark = true;
        return WalkResult::interrupt();
      }
      op->setAttr("run-r52-ops-on-dive.marked", builder.getUnitAttr());
      ++marked;
      return WalkResult::advance();
    });
    if (failedMark)
      return signalPassFailure();
    root->setAttr("run-r52-ops-on-dive.marked_count",
                  builder.getI64IntegerAttr(marked));
  }
};

// TSV row: "scalar-core-control-flow-lowering" at 0xdded21.
struct DwcScalarCoreControlFlowLoweringPass : public darwinn::impl::DwcScalarCoreControlFlowLoweringPassBase<DwcScalarCoreControlFlowLoweringPass> {
  using Base::Base;

  void runOnOperation() override {
    // The sibling-owned LowerCopySlice and LowerConvert sets do the real
    // lowering, identity folds below only clean up what patterns leave behind.
    RewritePatternSet patterns(&getContext());
    darwinn::populateLowerCopySlicePatterns(patterns);
    darwinn::populateLowerConvertPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns))))
      return signalPassFailure();
    func::FuncOp func = getOperation();
    if (failed(applyLocalCopySliceLowering(func)))
      return signalPassFailure();
    if (failed(applyLocalConvertLowering(func)))
      return signalPassFailure();
  }
};

// TSV row: "scalar-core-std-ops-lowering" at 0xdded75.
struct DwcScalarCoreStdOpsLoweringPass : public darwinn::impl::DwcScalarCoreStdOpsLoweringPassBase<DwcScalarCoreStdOpsLoweringPass> {
  using Base::Base;

  void runOnOperation() override {
    // The sibling-owned LowerCopySlice and LowerConvert sets do the real
    // lowering, identity folds below only clean up what patterns leave behind.
    RewritePatternSet patterns(&getContext());
    darwinn::populateLowerCopySlicePatterns(patterns);
    darwinn::populateLowerConvertPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns))))
      return signalPassFailure();
    func::FuncOp func = getOperation();
    if (failed(applyLocalCopySliceLowering(func)))
      return signalPassFailure();
    if (failed(applyLocalConvertLowering(func)))
      return signalPassFailure();
  }
};

// TSV row: "scalar-ops-legalize" at 0xde6d2b.
struct DwcScalarOpsLegalizePass : public darwinn::impl::DwcScalarOpsLegalizePassBase<DwcScalarOpsLegalizePass> {
  using Base::Base;

  void runOnOperation() override {
    // Per-op lowering waits on kernel shapes in all_pseudocode.json.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned marked = 0;
    bool failedMark = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedMark = true;
        return WalkResult::interrupt();
      }
      op->setAttr("scalar-ops-legalize.marked", builder.getUnitAttr());
      ++marked;
      return WalkResult::advance();
    });
    if (failedMark)
      return signalPassFailure();
    root->setAttr("scalar-ops-legalize.marked_count",
                  builder.getI64IntegerAttr(marked));
  }
};

// TSV row: "scatter-gather-lowering" at 0xddeda5.
struct DwcScatterGatherLoweringPass : public darwinn::impl::DwcScatterGatherLoweringPassBase<DwcScatterGatherLoweringPass> {
  using Base::Base;

  void runOnOperation() override {
    // Per-op lowering waits on kernel shapes in all_pseudocode.json.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned marked = 0;
    bool failedMark = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedMark = true;
        return WalkResult::interrupt();
      }
      op->setAttr("scatter-gather-lowering.marked", builder.getUnitAttr());
      ++marked;
      return WalkResult::advance();
    });
    if (failedMark)
      return signalPassFailure();
    root->setAttr("scatter-gather-lowering.marked_count",
                  builder.getI64IntegerAttr(marked));
  }
};

// TSV row: "select-lowering" at 0xdded65.
struct DwcSelectLoweringPass : public darwinn::impl::DwcSelectLoweringPassBase<DwcSelectLoweringPass> {
  using Base::Base;

  void runOnOperation() override {
    // Per-op lowering waits on kernel shapes in all_pseudocode.json.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned marked = 0;
    bool failedMark = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedMark = true;
        return WalkResult::interrupt();
      }
      op->setAttr("select-lowering.marked", builder.getUnitAttr());
      ++marked;
      return WalkResult::advance();
    });
    if (failedMark)
      return signalPassFailure();
    root->setAttr("select-lowering.marked_count",
                  builder.getI64IntegerAttr(marked));
  }
};

// TSV row: "sharding-using-dive" at 0xde9635.
struct DwcShardingUsingDivePass : public darwinn::impl::DwcShardingUsingDivePassBase<DwcShardingUsingDivePass> {
  using Base::Base;

  void runOnOperation() override {
    // Per-op lowering waits on kernel shapes in all_pseudocode.json.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned marked = 0;
    bool failedMark = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedMark = true;
        return WalkResult::interrupt();
      }
      op->setAttr("sharding-using-dive.marked", builder.getUnitAttr());
      ++marked;
      return WalkResult::advance();
    });
    if (failedMark)
      return signalPassFailure();
    root->setAttr("sharding-using-dive.marked_count",
                  builder.getI64IntegerAttr(marked));
  }
};

// TSV row: "skipping fold of float convert" at 0xd68cda.
struct DwcSkippingFoldOfFloatConvertPass : public darwinn::impl::DwcSkippingFoldOfFloatConvertPassBase<DwcSkippingFoldOfFloatConvertPass> {
  using Base::Base;

  void runOnOperation() override {
    // Per-op lowering waits on kernel shapes in all_pseudocode.json.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned marked = 0;
    bool failedMark = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedMark = true;
        return WalkResult::interrupt();
      }
      op->setAttr("skipping fold of float convert.marked", builder.getUnitAttr());
      ++marked;
      return WalkResult::advance();
    });
    if (failedMark)
      return signalPassFailure();
    root->setAttr("skipping fold of float convert.marked_count",
                  builder.getI64IntegerAttr(marked));
  }
};

// TSV row: "split-op-lowering" at 0xddede2.
struct DwcSplitOpLoweringPass : public darwinn::impl::DwcSplitOpLoweringPassBase<DwcSplitOpLoweringPass> {
  using Base::Base;

  void runOnOperation() override {
    // Per-op lowering waits on kernel shapes in all_pseudocode.json.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned marked = 0;
    bool failedMark = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedMark = true;
        return WalkResult::interrupt();
      }
      op->setAttr("split-op-lowering.marked", builder.getUnitAttr());
      ++marked;
      return WalkResult::advance();
    });
    if (failedMark)
      return signalPassFailure();
    root->setAttr("split-op-lowering.marked_count",
                  builder.getI64IntegerAttr(marked));
  }
};

// TSV row: "stablehlo-composite-legalize-tfl-custom" at 0xdccdf7.
struct DwcStablehloCompositeLegalizeTflCustomPass : public darwinn::impl::DwcStablehloCompositeLegalizeTflCustomPassBase<DwcStablehloCompositeLegalizeTflCustomPass> {
  using Base::Base;

  void runOnOperation() override {
    // Per-op lowering waits on kernel shapes in all_pseudocode.json.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned marked = 0;
    bool failedMark = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedMark = true;
        return WalkResult::interrupt();
      }
      op->setAttr("stablehlo-composite-legalize-tfl-custom.marked", builder.getUnitAttr());
      ++marked;
      return WalkResult::advance();
    });
    if (failedMark)
      return signalPassFailure();
    root->setAttr("stablehlo-composite-legalize-tfl-custom.marked_count",
                  builder.getI64IntegerAttr(marked));
  }
};

// TSV row: "stablehlo-custom-call-legalize-composite" at 0xdef5f5.
struct DwcStablehloCustomCallLegalizeCompositePass : public darwinn::impl::DwcStablehloCustomCallLegalizeCompositePassBase<DwcStablehloCustomCallLegalizeCompositePass> {
  using Base::Base;

  void runOnOperation() override {
    // Per-op lowering waits on kernel shapes in all_pseudocode.json.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned marked = 0;
    bool failedMark = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedMark = true;
        return WalkResult::interrupt();
      }
      op->setAttr("stablehlo-custom-call-legalize-composite.marked", builder.getUnitAttr());
      ++marked;
      return WalkResult::advance();
    });
    if (failedMark)
      return signalPassFailure();
    root->setAttr("stablehlo-custom-call-legalize-composite.marked_count",
                  builder.getI64IntegerAttr(marked));
  }
};

// TSV row: "stablehlo-legalize-composite-to-call" at 0xdd00c2.
struct DwcStablehloLegalizeCompositeToCallPass : public darwinn::impl::DwcStablehloLegalizeCompositeToCallPassBase<DwcStablehloLegalizeCompositeToCallPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect) {
        op->emitError() << "stablehlo-legalize-composite-to-call rejects unregistered operation "
                        << op->getName().getStringRef();
        failedLegal = true;
        return WalkResult::interrupt();
      }
      StringRef ns = dialect->getNamespace();
      if (ns != "darwinn" && ns != "dive_vm" && ns != "edgetpu" &&
          ns != "func" && ns != "arith" && ns != "tensor" &&
          ns != "memref" && ns != "scf" && ns != "cf" &&
          ns != "builtin" && ns != "llvm" && ns != "stablehlo") {
        op->emitError() << "stablehlo-legalize-composite-to-call rejects operation from dialect "
                        << ns;
        failedLegal = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failedLegal)
      signalPassFailure();
  }
};

// TSV row: "stablehlo-legalize-to-hlo" at 0xdbc34d.
struct DwcStablehloLegalizeToHloPass : public darwinn::impl::DwcStablehloLegalizeToHloPassBase<DwcStablehloLegalizeToHloPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect) {
        op->emitError() << "stablehlo-legalize-to-hlo rejects unregistered operation "
                        << op->getName().getStringRef();
        failedLegal = true;
        return WalkResult::interrupt();
      }
      StringRef ns = dialect->getNamespace();
      if (ns != "darwinn" && ns != "dive_vm" && ns != "edgetpu" &&
          ns != "func" && ns != "arith" && ns != "tensor" &&
          ns != "memref" && ns != "scf" && ns != "cf" &&
          ns != "builtin" && ns != "llvm" &&
          ns != "stablehlo" && ns != "mhlo") {
        op->emitError() << "stablehlo-legalize-to-hlo rejects operation from dialect "
                        << ns;
        failedLegal = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failedLegal)
      signalPassFailure();
  }
};

// TSV row: "stablehlo-legalize-to-vhlo" at 0xdbc236.
struct DwcStablehloLegalizeToVhloPass : public darwinn::impl::DwcStablehloLegalizeToVhloPassBase<DwcStablehloLegalizeToVhloPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect) {
        op->emitError() << "stablehlo-legalize-to-vhlo rejects unregistered operation "
                        << op->getName().getStringRef();
        failedLegal = true;
        return WalkResult::interrupt();
      }
      StringRef ns = dialect->getNamespace();
      if (ns != "darwinn" && ns != "dive_vm" && ns != "edgetpu" &&
          ns != "func" && ns != "arith" && ns != "tensor" &&
          ns != "memref" && ns != "scf" && ns != "cf" &&
          ns != "builtin" && ns != "llvm" &&
          ns != "stablehlo" && ns != "vhlo") {
        op->emitError() << "stablehlo-legalize-to-vhlo rejects operation from dialect "
                        << ns;
        failedLegal = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failedLegal)
      signalPassFailure();
  }
};

// TSV row: "stablehlo-legalize-vhlo" at 0xdbc251.
struct DwcStablehloLegalizeVhloPass : public darwinn::impl::DwcStablehloLegalizeVhloPassBase<DwcStablehloLegalizeVhloPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect) {
        op->emitError() << "stablehlo-legalize-vhlo rejects unregistered operation "
                        << op->getName().getStringRef();
        failedLegal = true;
        return WalkResult::interrupt();
      }
      StringRef ns = dialect->getNamespace();
      if (ns != "darwinn" && ns != "dive_vm" && ns != "edgetpu" &&
          ns != "func" && ns != "arith" && ns != "tensor" &&
          ns != "memref" && ns != "scf" && ns != "cf" &&
          ns != "builtin" && ns != "llvm" &&
          ns != "stablehlo" && ns != "vhlo") {
        op->emitError() << "stablehlo-legalize-vhlo rejects operation from dialect "
                        << ns;
        failedLegal = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failedLegal)
      signalPassFailure();
  }
};

// TSV row: "stochastic-convert" at 0xd68cc7.
struct DwcStochasticConvertPass : public darwinn::impl::DwcStochasticConvertPassBase<DwcStochasticConvertPass> {
  using Base::Base;

  void runOnOperation() override {
    // Per-op lowering waits on kernel shapes in all_pseudocode.json.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned marked = 0;
    bool failedMark = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedMark = true;
        return WalkResult::interrupt();
      }
      op->setAttr("stochastic-convert.marked", builder.getUnitAttr());
      ++marked;
      return WalkResult::advance();
    });
    if (failedMark)
      return signalPassFailure();
    root->setAttr("stochastic-convert.marked_count",
                  builder.getI64IntegerAttr(marked));
  }
};

// TSV row: "tf-legalize-hlo" at 0xdbc38d.
struct DwcTfLegalizeHloPass : public darwinn::impl::DwcTfLegalizeHloPassBase<DwcTfLegalizeHloPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect) {
        op->emitError() << "tf-legalize-hlo rejects unregistered operation "
                        << op->getName().getStringRef();
        failedLegal = true;
        return WalkResult::interrupt();
      }
      StringRef ns = dialect->getNamespace();
      if (ns != "darwinn" && ns != "dive_vm" && ns != "edgetpu" &&
          ns != "func" && ns != "arith" && ns != "tensor" &&
          ns != "memref" && ns != "scf" && ns != "cf" &&
          ns != "builtin" && ns != "llvm" &&
          ns != "tf" && ns != "mhlo") {
        op->emitError() << "tf-legalize-hlo rejects operation from dialect "
                        << ns;
        failedLegal = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failedLegal)
      signalPassFailure();
  }
};

// TSV row: "tfl-custom-lowering-rewriting-pass" at 0xd7f120.
struct DwcTflCustomLoweringRewritingPassPass : public darwinn::impl::DwcTflCustomLoweringRewritingPassPassBase<DwcTflCustomLoweringRewritingPassPass> {
  using Base::Base;

  void runOnOperation() override {
    // Per-op lowering waits on kernel shapes in all_pseudocode.json.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned marked = 0;
    bool failedMark = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedMark = true;
        return WalkResult::interrupt();
      }
      op->setAttr("tfl-custom-lowering-rewriting-pass.marked", builder.getUnitAttr());
      ++marked;
      return WalkResult::advance();
    });
    if (failedMark)
      return signalPassFailure();
    root->setAttr("tfl-custom-lowering-rewriting-pass.marked_count",
                  builder.getI64IntegerAttr(marked));
  }
};

// TSV row: "tfl-legalize-chlo" at 0xdbc2fb.
struct DwcTflLegalizeChloPass : public darwinn::impl::DwcTflLegalizeChloPassBase<DwcTflLegalizeChloPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect) {
        op->emitError() << "tfl-legalize-chlo rejects unregistered operation "
                        << op->getName().getStringRef();
        failedLegal = true;
        return WalkResult::interrupt();
      }
      StringRef ns = dialect->getNamespace();
      if (ns != "darwinn" && ns != "dive_vm" && ns != "edgetpu" &&
          ns != "func" && ns != "arith" && ns != "tensor" &&
          ns != "memref" && ns != "scf" && ns != "cf" &&
          ns != "builtin" && ns != "llvm" &&
          ns != "tfl" && ns != "chlo") {
        op->emitError() << "tfl-legalize-chlo rejects operation from dialect "
                        << ns;
        failedLegal = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failedLegal)
      signalPassFailure();
  }
};

// TSV row: "tfl-legalize-hashtables-tf" at 0xde2a7d.
struct DwcTflLegalizeHashtablesTfPass : public darwinn::impl::DwcTflLegalizeHashtablesTfPassBase<DwcTflLegalizeHashtablesTfPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect) {
        op->emitError() << "tfl-legalize-hashtables-tf rejects unregistered operation "
                        << op->getName().getStringRef();
        failedLegal = true;
        return WalkResult::interrupt();
      }
      StringRef ns = dialect->getNamespace();
      if (ns != "darwinn" && ns != "dive_vm" && ns != "edgetpu" &&
          ns != "func" && ns != "arith" && ns != "tensor" &&
          ns != "memref" && ns != "scf" && ns != "cf" &&
          ns != "builtin" && ns != "llvm" &&
          ns != "tfl" && ns != "tf") {
        op->emitError() << "tfl-legalize-hashtables-tf rejects operation from dialect "
                        << ns;
        failedLegal = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failedLegal)
      signalPassFailure();
  }
};

// TSV row: "tfl-legalize-hlo" at 0xdbc37c.
struct DwcTflLegalizeHloPass : public darwinn::impl::DwcTflLegalizeHloPassBase<DwcTflLegalizeHloPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect) {
        op->emitError() << "tfl-legalize-hlo rejects unregistered operation "
                        << op->getName().getStringRef();
        failedLegal = true;
        return WalkResult::interrupt();
      }
      StringRef ns = dialect->getNamespace();
      if (ns != "darwinn" && ns != "dive_vm" && ns != "edgetpu" &&
          ns != "func" && ns != "arith" && ns != "tensor" &&
          ns != "memref" && ns != "scf" && ns != "cf" &&
          ns != "builtin" && ns != "llvm" &&
          ns != "tfl" && ns != "mhlo") {
        op->emitError() << "tfl-legalize-hlo rejects operation from dialect "
                        << ns;
        failedLegal = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failedLegal)
      signalPassFailure();
  }
};

// TSV row: "tfl-legalize-tensorlist" at 0xd66a8f.
struct DwcTflLegalizeTensorlistPass : public darwinn::impl::DwcTflLegalizeTensorlistPassBase<DwcTflLegalizeTensorlistPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect) {
        op->emitError() << "tfl-legalize-tensorlist rejects unregistered operation "
                        << op->getName().getStringRef();
        failedLegal = true;
        return WalkResult::interrupt();
      }
      StringRef ns = dialect->getNamespace();
      if (ns != "darwinn" && ns != "dive_vm" && ns != "edgetpu" &&
          ns != "func" && ns != "arith" && ns != "tensor" &&
          ns != "memref" && ns != "scf" && ns != "cf" &&
          ns != "builtin" && ns != "llvm" && ns != "tfl") {
        op->emitError() << "tfl-legalize-tensorlist rejects operation from dialect "
                        << ns;
        failedLegal = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failedLegal)
      signalPassFailure();
  }
};

// TSV row: "tfl-legalize-tf" at 0xde2ae8.
struct DwcTflLegalizeTfPass : public darwinn::impl::DwcTflLegalizeTfPassBase<DwcTflLegalizeTfPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect) {
        op->emitError() << "tfl-legalize-tf rejects unregistered operation "
                        << op->getName().getStringRef();
        failedLegal = true;
        return WalkResult::interrupt();
      }
      StringRef ns = dialect->getNamespace();
      if (ns != "darwinn" && ns != "dive_vm" && ns != "edgetpu" &&
          ns != "func" && ns != "arith" && ns != "tensor" &&
          ns != "memref" && ns != "scf" && ns != "cf" &&
          ns != "builtin" && ns != "llvm" &&
          ns != "tfl" && ns != "tf") {
        op->emitError() << "tfl-legalize-tf rejects operation from dialect "
                        << ns;
        failedLegal = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failedLegal)
      signalPassFailure();
  }
};

// TSV row: "tfl-legalize-tf-while" at 0xe07f5e.
struct DwcTflLegalizeTfWhilePass : public darwinn::impl::DwcTflLegalizeTfWhilePassBase<DwcTflLegalizeTfWhilePass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect) {
        op->emitError() << "tfl-legalize-tf-while rejects unregistered operation "
                        << op->getName().getStringRef();
        failedLegal = true;
        return WalkResult::interrupt();
      }
      StringRef ns = dialect->getNamespace();
      if (ns != "darwinn" && ns != "dive_vm" && ns != "edgetpu" &&
          ns != "func" && ns != "arith" && ns != "tensor" &&
          ns != "memref" && ns != "scf" && ns != "cf" &&
          ns != "builtin" && ns != "llvm" &&
          ns != "tfl" && ns != "tf") {
        op->emitError() << "tfl-legalize-tf-while rejects operation from dialect "
                        << ns;
        failedLegal = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failedLegal)
      signalPassFailure();
  }
};

// TSV row: "tfl-legalize-variables-tf" at 0xde2a98.
struct DwcTflLegalizeVariablesTfPass : public darwinn::impl::DwcTflLegalizeVariablesTfPassBase<DwcTflLegalizeVariablesTfPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect) {
        op->emitError() << "tfl-legalize-variables-tf rejects unregistered operation "
                        << op->getName().getStringRef();
        failedLegal = true;
        return WalkResult::interrupt();
      }
      StringRef ns = dialect->getNamespace();
      if (ns != "darwinn" && ns != "dive_vm" && ns != "edgetpu" &&
          ns != "func" && ns != "arith" && ns != "tensor" &&
          ns != "memref" && ns != "scf" && ns != "cf" &&
          ns != "builtin" && ns != "llvm" &&
          ns != "tfl" && ns != "tf") {
        op->emitError() << "tfl-legalize-variables-tf rejects operation from dialect "
                        << ns;
        failedLegal = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failedLegal)
      signalPassFailure();
  }
};

// TSV row: "tfl-lower-quant-annotations" at 0xd87c99.
struct DwcTflLowerQuantAnnotationsPass : public darwinn::impl::DwcTflLowerQuantAnnotationsPassBase<DwcTflLowerQuantAnnotationsPass> {
  using Base::Base;

  void runOnOperation() override {
    // The sibling-owned LowerCopySlice and LowerConvert sets do the real
    // lowering, identity folds below only clean up what patterns leave behind.
    RewritePatternSet patterns(&getContext());
    darwinn::populateLowerCopySlicePatterns(patterns);
    darwinn::populateLowerConvertPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns))))
      return signalPassFailure();
    func::FuncOp func = getOperation();
    if (failed(applyLocalCopySliceLowering(func)))
      return signalPassFailure();
    if (failed(applyLocalConvertLowering(func)))
      return signalPassFailure();
  }
};

// TSV row: "tfl-lower-static-tensor-list" at 0xd66ca5.
struct DwcTflLowerStaticTensorListPass : public darwinn::impl::DwcTflLowerStaticTensorListPassBase<DwcTflLowerStaticTensorListPass> {
  using Base::Base;

  void runOnOperation() override {
    // The sibling-owned LowerCopySlice and LowerConvert sets do the real
    // lowering, identity folds below only clean up what patterns leave behind.
    RewritePatternSet patterns(&getContext());
    darwinn::populateLowerCopySlicePatterns(patterns);
    darwinn::populateLowerConvertPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns))))
      return signalPassFailure();
    func::FuncOp func = getOperation();
    if (failed(applyLocalCopySliceLowering(func)))
      return signalPassFailure();
    if (failed(applyLocalConvertLowering(func)))
      return signalPassFailure();
  }
};

// TSV row: "top-k-lowering-policy" at 0xd5dd44.
struct DwcTopKLoweringPolicyPass : public darwinn::impl::DwcTopKLoweringPolicyPassBase<DwcTopKLoweringPolicyPass> {
  using Base::Base;

  void runOnOperation() override {
    // Per-op lowering waits on kernel shapes in all_pseudocode.json.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned marked = 0;
    bool failedMark = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedMark = true;
        return WalkResult::interrupt();
      }
      op->setAttr("top-k-lowering-policy.marked", builder.getUnitAttr());
      ++marked;
      return WalkResult::advance();
    });
    if (failedMark)
      return signalPassFailure();
    root->setAttr("top-k-lowering-policy.marked_count",
                  builder.getI64IntegerAttr(marked));
  }
};

// TSV row: "tpu-clustering-algorithm" at 0xdcdd6e.
struct DwcTpuClusteringAlgorithmPass : public darwinn::impl::DwcTpuClusteringAlgorithmPassBase<DwcTpuClusteringAlgorithmPass> {
  using Base::Base;

  void runOnOperation() override {
    // Binary packet layout is absent from all_pseudocode.json, group and order only.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    std::map<Operation *, unsigned> clusterOf;
    unsigned nextCluster = 0;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      unsigned cluster = nextCluster;
      bool joined = false;
      for (Value operand : op->getOperands()) {
        Operation *def = operand.getDefiningOp();
        if (!def)
          continue;
        auto it = clusterOf.find(def);
        if (it != clusterOf.end()) {
          cluster = it->second;
          joined = true;
          break;
        }
      }
      if (!joined)
        ++nextCluster;
      clusterOf[op] = cluster;
      op->setAttr("tpu.cluster_id", builder.getI64IntegerAttr(cluster));
      return WalkResult::advance();
    });
    root->setAttr("tpu.cluster_count",
                  builder.getI64IntegerAttr(nextCluster));
  }
};

// TSV row: "vhlo-legalize-stablehlo" at 0xdbc299.
struct DwcVhloLegalizeStablehloPass : public darwinn::impl::DwcVhloLegalizeStablehloPassBase<DwcVhloLegalizeStablehloPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect) {
        op->emitError() << "vhlo-legalize-stablehlo rejects unregistered operation "
                        << op->getName().getStringRef();
        failedLegal = true;
        return WalkResult::interrupt();
      }
      StringRef ns = dialect->getNamespace();
      if (ns != "darwinn" && ns != "dive_vm" && ns != "edgetpu" &&
          ns != "func" && ns != "arith" && ns != "tensor" &&
          ns != "memref" && ns != "scf" && ns != "cf" &&
          ns != "builtin" && ns != "llvm" &&
          ns != "vhlo" && ns != "stablehlo") {
        op->emitError() << "vhlo-legalize-stablehlo rejects operation from dialect "
                        << ns;
        failedLegal = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failedLegal)
      signalPassFailure();
  }
};

// TSV row: "vhlo-legalize-to-stablehlo" at 0xdbc271.
struct DwcVhloLegalizeToStablehloPass : public darwinn::impl::DwcVhloLegalizeToStablehloPassBase<DwcVhloLegalizeToStablehloPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect) {
        op->emitError() << "vhlo-legalize-to-stablehlo rejects unregistered operation "
                        << op->getName().getStringRef();
        failedLegal = true;
        return WalkResult::interrupt();
      }
      StringRef ns = dialect->getNamespace();
      if (ns != "darwinn" && ns != "dive_vm" && ns != "edgetpu" &&
          ns != "func" && ns != "arith" && ns != "tensor" &&
          ns != "memref" && ns != "scf" && ns != "cf" &&
          ns != "builtin" && ns != "llvm" &&
          ns != "vhlo" && ns != "stablehlo") {
        op->emitError() << "vhlo-legalize-to-stablehlo rejects operation from dialect "
                        << ns;
        failedLegal = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failedLegal)
      signalPassFailure();
  }
};

// TSV row: "wrap-up-dive-program" at 0xdce5cc.
struct DwcWrapUpDiveProgramPass : public darwinn::impl::DwcWrapUpDiveProgramPassBase<DwcWrapUpDiveProgramPass> {
  using Base::Base;

  void runOnOperation() override {
    // Binary packet layout is absent from all_pseudocode.json, group and order only.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    std::map<Operation *, unsigned> clusterOf;
    unsigned nextCluster = 0;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      unsigned cluster = nextCluster;
      bool joined = false;
      for (Value operand : op->getOperands()) {
        Operation *def = operand.getDefiningOp();
        if (!def)
          continue;
        auto it = clusterOf.find(def);
        if (it != clusterOf.end()) {
          cluster = it->second;
          joined = true;
          break;
        }
      }
      if (!joined)
        ++nextCluster;
      clusterOf[op] = cluster;
      op->setAttr("tpu.cluster_id", builder.getI64IntegerAttr(cluster));
      return WalkResult::advance();
    });
    root->setAttr("tpu.cluster_count",
                  builder.getI64IntegerAttr(nextCluster));
  }
};

// TSV row: "xla_cpu_use_new_xtile_lowering" at 0xdded02.
struct DwcXlaCpuUseNewXtileLoweringPass : public darwinn::impl::DwcXlaCpuUseNewXtileLoweringPassBase<DwcXlaCpuUseNewXtileLoweringPass> {
  using Base::Base;

  void runOnOperation() override {
    // Per-op lowering waits on kernel shapes in all_pseudocode.json.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    unsigned marked = 0;
    bool failedMark = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedMark = true;
        return WalkResult::interrupt();
      }
      op->setAttr("xla_cpu_use_new_xtile_lowering.marked", builder.getUnitAttr());
      ++marked;
      return WalkResult::advance();
    });
    if (failedMark)
      return signalPassFailure();
    root->setAttr("xla_cpu_use_new_xtile_lowering.marked_count",
                  builder.getI64IntegerAttr(marked));
  }
};

} // namespace

namespace mlir {
namespace darwinn {
#define GEN_PASS_REGISTRATION
#include "DwcPasses.h.inc"
} // namespace darwinn
} // namespace mlir
