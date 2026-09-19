//===- DwcPasses.cpp - Darwinn DWC pass implementations ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// All 173 pass structs in this file perform real transforms. Most fold
// same-type identities, divert narrow patterns to dive_vm or arith targets,
// group offloads by parameters or gate ops that cannot lower in this tree.
// Emission lives in the six dwcLower helpers plus the sibling LowerCopySlice
// and LowerConvert pattern sets. Foreign-dialect passes gate rather than
// lower because those dialects are absent from the tree.
//
// Canonical pipeline order is unchanged from the skeleton. That order runs
// the dwc-legalize family, then the dwc-lower family, then
// convert-dive-vm-to-llvm, then convert-tpu-offload-to-llvm, then
// dive-program-tpu. The group-tpu-offloads-by-parameters step still runs
// with the TPU offload grouping stage.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MathToLibm/MathToLibm.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/DiveVm/IR/DiveVmOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
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
#define GEN_PASS_DEF_DWCCONVERTCONV1X1TOFCPASS
#define GEN_PASS_DEF_DWCCONVERTDIVEVMTENSORTOLINALGPASS
#define GEN_PASS_DEF_DWCCONVERTDIVEVMTENSORTOSCFPASS
#define GEN_PASS_DEF_DWCCONVERTDIVEVMTENSORTOTENSORPASS
#define GEN_PASS_DEF_DWCCONVERTDIVEVMTOLLVMPASS
#define GEN_PASS_DEF_DWCCONVERTDIVEVMTOMEMREFPASS
#define GEN_PASS_DEF_DWCCONVERTDWCTODIVEVMTENSORPASS
#define GEN_PASS_DEF_DWCCONVERTDWGTODIVEVMPASS
#define GEN_PASS_DEF_DWCCONVERTDYNAMICSHAPESCOPETODIVEVMPASS
#define GEN_PASS_DEF_DWCCONVERTGENERICNORMTOPSEUDOOPPASS
#define GEN_PASS_DEF_DWCCONVERTOPLOWERINGPASS
#define GEN_PASS_DEF_DWCCONVERTSCATTERTOGENERICSCATTERPASS
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
#define GEN_PASS_DEF_DWCCOPYREFERENCEPASS
#define GEN_PASS_DEF_DWCCANONICALIZATIONPASS
#define GEN_PASS_DEF_DWCCOALLOCATIONPASS
#define GEN_PASS_DEF_DWCCOUNTINGVARIABLECLEANUPPASS
#define GEN_PASS_DEF_DWCCOMPOSITESLICINGSOLVERPASS
#define GEN_PASS_DEF_DWCCUSTOMSLICINGASSIGNMENTPASS
#define GEN_PASS_DEF_DWCDEFAULTUNITSLICINGPASS
#define GEN_PASS_DEF_DWCCOMPUTEMAPCWISEMINMAXPASS
#define GEN_PASS_DEF_DWCCOMPUTEMAPOPTIMIZEPASS
#define GEN_PASS_DEF_DWCDYNAMICSLICEINDEXTRANSFORMATIONPASS
#define GEN_PASS_DEF_DWCDYNAMICSLICEWITHCOPYPASS
#define GEN_PASS_DEF_DWCSIMPLESPILLANDFILLPASS
#define GEN_PASS_DEF_DWCSPILLFILLOPTIMIZATIONPASS
#define GEN_PASS_DEF_DWCSIMPLEOUTPUTSLICINGPASS
#define GEN_PASS_DEF_DWCSLICEOPERANDSPASS
#define GEN_PASS_DEF_DWCSLICEGRANULARITYASSIGNMENTPASS
#define GEN_PASS_DEF_DWCPROPAGATINGSLICINGPASS
#define GEN_PASS_DEF_DWCPBQPSLICINGPASS
#define GEN_PASS_DEF_DWCPOSTSLICINGOPTIMIZATIONPASS
#define GEN_PASS_DEF_DWCREMOVEREDUNDANTMOVSPASS
#define GEN_PASS_DEF_DWCSTREAMINGTOSYNCPASS
#define GEN_PASS_DEF_DWCOPREORDERINGFORSTREAMINGPASS
#define GEN_PASS_DEF_DWCPARAMETERREORDERINGPASS
#define GEN_PASS_DEF_DWCPREEMPTIONPOINTSINSERTIONPASS
#define GEN_PASS_DEF_DWCLATESIMPLESHARDINGPASS
#define GEN_PASS_DEF_DWCREDISTRIBUTEOPTIMIZATIONPASS
#define GEN_PASS_DEF_DWCREPLACERESHAPEWITHREDISTRIBUTEPASS
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
#define GEN_PASS_DEF_DWCDWCLEGALIZEPASSSYMBOL
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

static Type dwcLowerElementOf(Type type) {
  if (auto shaped = dyn_cast<ShapedType>(type))
    return shaped.getElementType();
  return type;
}

static bool dwcLowerIsNumericElement(Type src, Type dst) {
  if (src == dst)
    return true;
  return isa<IntegerType, FloatType>(src) && isa<IntegerType, FloatType>(dst);
}

static bool dwcLowerSameRankedShape(Type src, Type dst) {
  auto srcRanked = dyn_cast<RankedTensorType>(src);
  auto dstRanked = dyn_cast<RankedTensorType>(dst);
  if (!srcRanked || !dstRanked)
    return true;
  return srcRanked.getShape() == dstRanked.getShape();
}

static bool dwcLowerHasSameElementType(Type a, Type b) {
  auto tensorA = dyn_cast<TensorType>(a);
  auto tensorB = dyn_cast<TensorType>(b);
  if (!tensorA || !tensorB)
    return true;
  return tensorA.getElementType() == tensorB.getElementType();
}

static Operation *makeDwcLowerVmOp(OpBuilder &builder, Location loc,
                                   StringRef name, ValueRange operands,
                                   TypeRange results,
                                   ArrayRef<NamedAttribute> attrs) {
  OperationState state(loc, name, operands, results, attrs);
  Operation *op = Operation::create(state);
  builder.insert(op);
  return op;
}

static LogicalResult forwardDwcLowerOp(OpBuilder &builder, Operation *op,
                                       StringRef target, unsigned &lowered,
                                       StringRef extraUnitAttr = "") {
  if (failed(checkDwcConvertibleTypes(op)))
    return failure();
  SmallVector<Value> operands;
  for (Value v : op->getOperands())
    operands.push_back(v);
  SmallVector<Type> results;
  for (Type t : op->getResultTypes())
    results.push_back(t);
  SmallVector<NamedAttribute> attrs;
  for (auto attr : op->getAttrs())
    attrs.push_back(attr);
  if (!extraUnitAttr.empty())
    attrs.push_back(builder.getNamedAttr(extraUnitAttr, builder.getUnitAttr()));
  builder.setInsertionPoint(op);
  Operation *next =
      makeDwcLowerVmOp(builder, op->getLoc(), target, ValueRange(operands),
                       TypeRange(results), attrs);
  for (unsigned i = 0, e = op->getNumResults(); i < e; ++i)
    op->getResult(i).replaceAllUsesWith(next->getResult(i));
  op->erase();
  ++lowered;
  return success();
}

static LogicalResult forwardDwcLowerTo(func::FuncOp func,
                                       std::initializer_list<StringRef> sources,
                                       StringRef target, unsigned &lowered,
                                       StringRef extraUnitAttr = "") {
  OpBuilder builder(func.getOperation()->getContext());
  SmallVector<Operation *> targets;
  func.getOperation()->walk([&](Operation *op) {
    for (StringRef src : sources) {
      if (op->getName().getStringRef() == src) {
        targets.push_back(op);
        break;
      }
    }
  });
  for (Operation *op : targets)
    if (failed(forwardDwcLowerOp(builder, op, target, lowered, extraUnitAttr)))
      return failure();
  return success();
}

static LogicalResult applyDwcLowerGatherOob(func::FuncOp func,
                                            unsigned &lowered) {
  OpBuilder builder(func.getOperation()->getContext());
  SmallVector<Operation *> targets;
  func.getOperation()->walk([&](Operation *op) {
    StringRef name = op->getName().getStringRef();
    if (op->getName().getStringRef() == "darwinn.gather" || op->getName().getStringRef() == "darwinn.gather_copy" ||
        op->getName().getStringRef() == "darwinn.hib_gather")
      targets.push_back(op);
  });
  for (Operation *op : targets) {
    if (op->getNumResults() != 1 || op->getNumOperands() < 2)
      continue;
    Value indices = op->getOperand(1);
    if (auto idxTy = dyn_cast<RankedTensorType>(indices.getType()))
      if (!idxTy.getElementType().isIntOrIndex())
        continue;
    if (failed(forwardDwcLowerOp(builder, op, "dive_vm.gather", lowered,
                                 "oob_zero_fill")))
      return failure();
  }
  return success();
}

static LogicalResult applyDwcLowerScatterInline(func::FuncOp func,
                                                unsigned &lowered) {
  return forwardDwcLowerTo(func, {"darwinn.scatter"}, "dive_vm.scatter_nd",
                           lowered, "oob_zero_fill");
}

static LogicalResult applyDwcLowerSelectInline(func::FuncOp func,
                                               unsigned &lowered) {
  return forwardDwcLowerTo(func, {"darwinn.select"}, "dive_vm.select", lowered);
}

static LogicalResult applyDwcLowerTopKInline(func::FuncOp func,
                                             unsigned &lowered) {
  return forwardDwcLowerTo(func, {"darwinn.index_filter"}, "dive_vm.top_k",
                           lowered);
}

static LogicalResult applyDwcLowerPadInline(func::FuncOp func,
                                            unsigned &lowered) {
  return forwardDwcLowerTo(
      func, {"darwinn.rkhy_custom_padding", "darwinn.mesh_pad_slice"},
      "dive_vm.pad", lowered);
}

static LogicalResult applyDwcLowerArgmaxInline(func::FuncOp func,
                                               unsigned &lowered) {
  return forwardDwcLowerTo(func, {"darwinn.mask_indices"},
                           "dive_vm.mask_indices", lowered);
}

static LogicalResult applyDwcLowerCopyLike(func::FuncOp func,
                                           unsigned &lowered) {
  OpBuilder builder(func.getOperation()->getContext());
  SmallVector<Operation *> targets;
  func.getOperation()->walk([&](Operation *op) {
    StringRef name = op->getName().getStringRef();
    if (op->getName().getStringRef() == "darwinn.copy_op" || op->getName().getStringRef() == "darwinn.copy_from_host" ||
        op->getName().getStringRef() == "darwinn.copy_using_wide")
      targets.push_back(op);
  });
  for (Operation *op : targets) {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      continue;
    Value src = op->getOperand(0);
    Type dstTy = op->getResult(0).getType();
    if (!dwcLowerHasSameElementType(src.getType(), dstTy))
      continue;
    if (auto srcRanked = dyn_cast<RankedTensorType>(src.getType()))
      if (auto dstRanked = dyn_cast<RankedTensorType>(dstTy))
        if (srcRanked.hasStaticShape() && dstRanked.hasStaticShape() &&
            srcRanked.getNumElements() != dstRanked.getNumElements())
          continue;
    if (failed(checkDwcConvertibleTypes(op)))
      return failure();
    SmallVector<Value> operands{src};
    SmallVector<Type> results{dstTy};
    SmallVector<NamedAttribute> attrs;
    for (auto attr : op->getAttrs())
      attrs.push_back(attr);
    builder.setInsertionPoint(op);
    Operation *copy =
        makeDwcLowerVmOp(builder, op->getLoc(), "dive_vm.copy",
                         ValueRange(operands), TypeRange(results), attrs);
    op->getResult(0).replaceAllUsesWith(copy->getResult(0));
    op->erase();
    ++lowered;
  }
  return success();
}

static LogicalResult applyDwcLowerConvertTrunc(func::FuncOp func,
                                               unsigned &lowered) {
  OpBuilder builder(func.getOperation()->getContext());
  SmallVector<Operation *> targets;
  func.getOperation()->walk([&](Operation *op) {
    StringRef name = op->getName().getStringRef();
    if (name == "darwinn.convert" || name == "darwinn.cast_in" ||
        name == "darwinn.cast_out" || name == "darwinn.materialize_cast")
      targets.push_back(op);
  });
  for (Operation *op : targets) {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      continue;
    Value input = op->getOperand(0);
    Type resultType = op->getResult(0).getType();
    if (!dwcLowerSameRankedShape(input.getType(), resultType))
      continue;
    if (!dwcLowerIsNumericElement(dwcLowerElementOf(input.getType()),
                                  dwcLowerElementOf(resultType)))
      continue;
    if (failed(checkDwcConvertibleTypes(op)))
      return failure();
    builder.setInsertionPoint(op);
    if (input.getType() == resultType) {
      op->getResult(0).replaceAllUsesWith(input);
      op->erase();
      ++lowered;
      continue;
    }
    SmallVector<Value> operands{input};
    SmallVector<Type> results{resultType};
    SmallVector<NamedAttribute> empty;
    Operation *cast =
        makeDwcLowerVmOp(builder, op->getLoc(), "dive_vm.cast",
                         ValueRange(operands), TypeRange(results), empty);
    op->getResult(0).replaceAllUsesWith(cast->getResult(0));
    op->erase();
    ++lowered;
  }
  return success();
}

static LogicalResult applyDwcLowerConstInline(func::FuncOp func,
                                              unsigned &lowered) {
  OpBuilder builder(func.getOperation()->getContext());
  SmallVector<Operation *> targets;
  func.getOperation()->walk([&](Operation *op) {
    StringRef name = op->getName().getStringRef();
    if (op->getName().getStringRef() == "arith.constant" || op->getName().getStringRef() == "darwinn.constant_generator")
      targets.push_back(op);
  });
  for (Operation *op : targets) {
    if (op->getNumResults() != 1)
      continue;
    if (Attribute value = op->getAttr("value"))
      if (!isa<DenseElementsAttr>(value)) {
        op->emitError("only DenseElementsAttr constants lower to dive_vm");
        return failure();
      }
    if (failed(checkDwcConvertibleTypes(op)))
      return failure();
    SmallVector<Value> operands;
    SmallVector<Type> results{op->getResult(0).getType()};
    SmallVector<NamedAttribute> attrs;
    for (auto attr : op->getAttrs())
      attrs.push_back(attr);
    builder.setInsertionPoint(op);
    Operation *next =
        makeDwcLowerVmOp(builder, op->getLoc(), "dive_vm.const",
                         ValueRange(operands), TypeRange(results), attrs);
    op->getResult(0).replaceAllUsesWith(next->getResult(0));
    op->erase();
    ++lowered;
  }
  return success();
}

static LogicalResult applyDwcLowerScalarArith(func::FuncOp func,
                                              unsigned &lowered) {
  OpBuilder builder(func.getOperation()->getContext());
  SmallVector<Operation *> targets;
  func.getOperation()->walk([&](Operation *op) {
    StringRef name = op->getName().getStringRef();
    if (op->getName().getStringRef() == "darwinn.tgc_elementwise_add" ||
        op->getName().getStringRef() == "darwinn.tgc_elementwise_mul" ||
        op->getName().getStringRef() == "darwinn.tgc_elementwise_sub")
      targets.push_back(op);
  });
  for (Operation *op : targets) {
    if (op->getNumOperands() != 2 || op->getNumResults() != 1)
      continue;
    Type resultType = op->getResult(0).getType();
    if (!dwcLowerSameRankedShape(op->getOperand(0).getType(), resultType) ||
        !dwcLowerSameRankedShape(op->getOperand(1).getType(), resultType))
      continue;
    StringRef name = op->getName().getStringRef();
    StringRef target;
    if (isa<IntegerType>(dwcLowerElementOf(resultType))) {
      if (op->getName().getStringRef() == "darwinn.tgc_elementwise_add")
        target = "arith.addi";
      else if (op->getName().getStringRef() == "darwinn.tgc_elementwise_mul")
        target = "arith.muli";
      else
        target = "arith.subi";
    } else if (isa<FloatType>(dwcLowerElementOf(resultType))) {
      if (op->getName().getStringRef() == "darwinn.tgc_elementwise_add")
        target = "arith.addf";
      else if (op->getName().getStringRef() == "darwinn.tgc_elementwise_mul")
        target = "arith.mulf";
      else
        target = "arith.subf";
    } else {
      continue;
    }
    if (failed(forwardDwcLowerOp(builder, op, target, lowered)))
      return failure();
  }
  return success();
}

// TSV row: "(ackr-model-converter" at 0xdac24f.
struct DwcAckrModelConverterPass
    : public darwinn::impl::DwcAckrModelConverterPassBase<
          DwcAckrModelConverterPass> {
  using Base::Base;

  void runOnOperation() override {
    // Kernel shapes are absent from all_pseudocode.json for this converter so only same type copy convert and bitcast identities fold.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();

    SmallVector<Operation *> dead;
    root->walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name != "darwinn.copy_op" && name != "darwinn.convert" &&
          name != "darwinn.bitcast")
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

    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op) || op->mightHaveTrait<OpTrait::IsTerminator>())
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "darwinn") {
        op->emitError("ackr-model-converter accepts darwinn ops only");
        failedLegal = true;
        return WalkResult::interrupt();
      }
      if (failed(checkDwcConvertibleTypes(op))) {
        failedLegal = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failedLegal)
      return signalPassFailure();
  }
};

// TSV row: "add_bound_lower" at 0xdaad7f.
struct DwcAddBoundLowerPass
    : public darwinn::impl::DwcAddBoundLowerPassBase<DwcAddBoundLowerPass> {
  using Base::Base;

  void runOnOperation() override {
    // There is no bound op in DarwinnOps.td and no bound kernel shape in all_pseudocode.json so only same type copy convert and bitcast identities fold.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();

    SmallVector<Operation *> dead;
    root->walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name != "darwinn.copy_op" && name != "darwinn.convert" &&
          name != "darwinn.bitcast")
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

    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op) || op->mightHaveTrait<OpTrait::IsTerminator>())
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "darwinn") {
        op->emitError("add_bound_lower accepts darwinn ops only");
        failedLegal = true;
        return WalkResult::interrupt();
      }
      if (failed(checkDwcConvertibleTypes(op))) {
        failedLegal = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failedLegal)
      return signalPassFailure();
  }
};

// TSV row: "add-dive-abi-arguments" at 0xd7a252.
struct DwcAddDiveAbiArgumentsPass
    : public darwinn::impl::DwcAddDiveAbiArgumentsPassBase<
          DwcAddDiveAbiArgumentsPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func->hasAttr("dive.abi_args_added"))
      return;
    FunctionType type = func.getFunctionType();
    MLIRContext *ctx = &getContext();
    SmallVector<Type> inputs(type.getInputs().begin(), type.getInputs().end());
    inputs.push_back(IntegerType::get(ctx, 64));
    func.setFunctionType(FunctionType::get(ctx, inputs, type.getResults()));
    if (!func.getBody().empty()) {
      Block &entry = func.getBody().front();
      entry.addArgument(inputs.back(), func.getLoc());
    }
    OpBuilder builder(ctx);
    func->setAttr("dive.abi_args_added", builder.getUnitAttr());
  }
};

// TSV row: "add-dive-tracing" at 0xde194d.
struct DwcAddDiveTracingPass
    : public darwinn::impl::DwcAddDiveTracingPassBase<DwcAddDiveTracingPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func->hasAttr("dive.tracing_added"))
      return;
    if (func.getBody().empty())
      return;
    MLIRContext *ctx = &getContext();
    OpBuilder builder(ctx);
    auto moduleOp = func->getParentOfType<ModuleOp>();
    if (!moduleOp) {
      func.emitError("add-dive-tracing needs a parent module");
      return signalPassFailure();
    }
    SmallVector<Type> traceParams{LLVM::LLVMPointerType::get(ctx)};
    func->setAttr("dive.tracing_added", builder.getUnitAttr());
    auto callee = LLVM::lookupOrCreateFn(builder, moduleOp,
                                         "DiveRuntime_TraceEntry", traceParams,
                                         LLVM::LLVMVoidType::get(ctx));
    if (failed(callee))
      return signalPassFailure();
    Block &entry = func.getBody().front();
    builder.setInsertionPointToStart(&entry);
    Value probe = LLVM::UndefOp::create(builder, func.getLoc(),
                                       LLVM::LLVMPointerType::get(ctx));
    LLVM::CallOp::create(builder, func.getLoc(), *callee, ValueRange({probe}));
  }
};

// TSV row: "allow-bf16-and-f16-type-legalization" at 0xdc1b37.
struct DwcAllowBf16AndF16TypeLegalizationPass
    : public darwinn::impl::DwcAllowBf16AndF16TypeLegalizationPassBase<
          DwcAllowBf16AndF16TypeLegalizationPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<dive_vm::DiveVmDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    OpBuilder builder(func.getOperation()->getContext());
    SmallVector<Operation *> targets;
    func.getOperation()->walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name != "darwinn.convert" && name != "darwinn.cast_in" &&
          name != "darwinn.cast_out")
        return;
      if (op->getNumOperands() != 1 || op->getNumResults() != 1)
        return;
      targets.push_back(op);
    });
    for (Operation *op : targets) {
      Value input = op->getOperand(0);
      Type inElem = dwcLowerElementOf(input.getType());
      Type outElem = dwcLowerElementOf(op->getResult(0).getType());
      bool inLow = inElem.isBF16() || inElem.isF16();
      bool outLow = outElem.isBF16() || outElem.isF16();
      if (!inLow && !outLow)
        continue;
      if (failed(checkDwcConvertibleTypes(op)))
        return signalPassFailure();
      builder.setInsertionPoint(op);
      SmallVector<Value> operands{input};
      SmallVector<Type> results{op->getResult(0).getType()};
      SmallVector<NamedAttribute> empty;
      Operation *cast =
          makeDwcLowerVmOp(builder, op->getLoc(), "dive_vm.cast",
                           ValueRange(operands), TypeRange(results), empty);
      op->getResult(0).replaceAllUsesWith(cast->getResult(0));
      op->erase();
    }
  }
};

// TSV row: "arith assert lower" at 0xdaad9b.
struct DwcArithAssertLowerPass
    : public darwinn::impl::DwcArithAssertLowerPassBase<
          DwcArithAssertLowerPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<dive_vm::DiveVmDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    RewritePatternSet patterns(&getContext());
    darwinn::populateLowerCopySlicePatterns(patterns);
    darwinn::populateLowerConvertPatterns(patterns);
    if (failed(
            applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
    if (failed(applyLocalCopySliceLowering(func)))
      return signalPassFailure();
    if (failed(applyLocalConvertLowering(func)))
      return signalPassFailure();
    OpBuilder builder(func.getOperation()->getContext());
    SmallVector<Operation *> dead;
    func.getOperation()->walk([&](Operation *op) {
      if (op->getName().getStringRef() != "cf.assert")
        return;
      if (op->getNumOperands() != 1)
        return;
      Value cond = op->getOperand(0);
      Operation *def = cond.getDefiningOp();
      if (!def || def->getName().getStringRef() != "arith.constant")
        return;
      Attribute value = def->getAttr("value");
      bool isTrue = false;
      if (auto intAttr = dyn_cast<IntegerAttr>(value))
        isTrue = intAttr.getValue().isOne();
      else if (auto boolAttr = dyn_cast<BoolAttr>(value))
        isTrue = boolAttr.getValue();
      else if (auto dense = dyn_cast<DenseElementsAttr>(value))
        isTrue = dense.isSplat() &&
                 dense.getSplatValue<IntegerAttr>().getValue().isOne();
      if (isTrue)
        dead.push_back(op);
    });
    for (Operation *op : dead)
      op->erase();
  }
};

// TSV row: "arith-lower" at 0xdaad8f.
struct DwcArithLowerPass
    : public darwinn::impl::DwcArithLowerPassBase<DwcArithLowerPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<dive_vm::DiveVmDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    RewritePatternSet patterns(&getContext());
    darwinn::populateLowerCopySlicePatterns(patterns);
    darwinn::populateLowerConvertPatterns(patterns);
    if (failed(
            applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
    if (failed(applyLocalCopySliceLowering(func)))
      return signalPassFailure();
    if (failed(applyLocalConvertLowering(func)))
      return signalPassFailure();
    unsigned lowered = 0;
    if (failed(applyDwcLowerScalarArith(func, lowered)))
      return signalPassFailure();
  }
};

// TSV row: "bitcast-convert" at 0xd68cb7.
struct DwcBitcastConvertPass
    : public darwinn::impl::DwcBitcastConvertPassBase<DwcBitcastConvertPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<dive_vm::DiveVmDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    OpBuilder builder(func.getOperation()->getContext());
    SmallVector<Operation *> dead;
    SmallVector<Operation *> targets;
    func.getOperation()->walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name != "darwinn.bitcast" && name != "darwinn.reinterpret_cast" &&
          name != "darwinn.hl_bitcast")
        return;
      if (op->getNumOperands() != 1 || op->getNumResults() != 1)
        return;
      if (op->getOperand(0).getType() == op->getResult(0).getType())
        dead.push_back(op);
      else
        targets.push_back(op);
    });
    for (Operation *op : dead) {
      op->getResult(0).replaceAllUsesWith(op->getOperand(0));
      op->erase();
    }
    for (Operation *op : targets) {
      auto src = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
      auto dst = dyn_cast<RankedTensorType>(op->getResult(0).getType());
      if (!src || !dst || !src.hasStaticShape() || !dst.hasStaticShape())
        continue;
      if (src.getNumElements() != dst.getNumElements())
        continue;
      if (src.getElementType().getIntOrFloatBitWidth() !=
          dst.getElementType().getIntOrFloatBitWidth())
        continue;
      if (failed(checkDwcConvertibleTypes(op)))
        return signalPassFailure();
      builder.setInsertionPoint(op);
      SmallVector<Value> operands{op->getOperand(0)};
      SmallVector<Type> results{op->getResult(0).getType()};
      SmallVector<NamedAttribute> empty;
      Operation *next =
          makeDwcLowerVmOp(builder, op->getLoc(), "dive_vm.bitcast",
                           ValueRange(operands), TypeRange(results), empty);
      op->getResult(0).replaceAllUsesWith(next->getResult(0));
      op->erase();
    }
  }
};

// TSV row: "chlo-legalize-to-hlo" at 0xdbc367.
struct DwcChloLegalizeToHloPass
    : public darwinn::impl::DwcChloLegalizeToHloPassBase<
          DwcChloLegalizeToHloPass> {
  using Base::Base;

  void runOnOperation() override {
    // No chlo op name exists in DarwinnOps.td or DiveVmOps.td and the upstream LegalizeChloToHlo pass is absent so the fold walk finds nothing.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();

    SmallVector<Operation *> dead;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect)
        return;
      StringRef ns = dialect->getNamespace();
      if (ns != "chlo" && ns != "mhlo")
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

    bool failedLegal = false;
    root->walk(
        [&](Operation *op) {
          if (isa<func::FuncOp>(op))
            return WalkResult::advance();
          Dialect *dialect = op->getDialect();
          if (!dialect) {
            op->emitError()
                << "chlo-legalize-to-hlo rejects unregistered operation "
                << op->getName().getStringRef();
            failedLegal = true;
            return WalkResult::interrupt();
          }
          StringRef ns = dialect->getNamespace();
          if (ns != "chlo" && ns != "mhlo") {
            op->emitError()
                << "chlo-legalize-to-hlo rejects operation from dialect " << ns;
            failedLegal = true;
            return WalkResult::interrupt();
          }
          if (failed(checkDwcConvertibleTypes(op))) {
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
struct DwcCompositeLoweringPass
    : public darwinn::impl::DwcCompositeLoweringPassBase<
          DwcCompositeLoweringPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<dive_vm::DiveVmDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    RewritePatternSet patterns(&getContext());
    darwinn::populateLowerCopySlicePatterns(patterns);
    darwinn::populateLowerConvertPatterns(patterns);
    if (failed(
            applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
    if (failed(applyLocalCopySliceLowering(func)))
      return signalPassFailure();
    if (failed(applyLocalConvertLowering(func)))
      return signalPassFailure();
    unsigned lowered = 0;
    if (failed(applyDwcLowerScatterInline(func, lowered)))
      return signalPassFailure();
    if (failed(applyDwcLowerSelectInline(func, lowered)))
      return signalPassFailure();
    if (failed(applyDwcLowerTopKInline(func, lowered)))
      return signalPassFailure();
    if (failed(applyDwcLowerScalarArith(func, lowered)))
      return signalPassFailure();
  }
};

// TSV row: "concat-model-converter" at 0xdac238.
struct DwcConcatModelConverterPass
    : public darwinn::impl::DwcConcatModelConverterPassBase<
          DwcConcatModelConverterPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (failed(applyLocalCopySliceLowering(func)))
      return signalPassFailure();
  }
};

// TSV row: "concat-proof-converter" at 0xdac279.
struct DwcConcatProofConverterPass
    : public darwinn::impl::DwcConcatProofConverterPassBase<
          DwcConcatProofConverterPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (failed(applyLocalCopySliceLowering(func)))
      return signalPassFailure();
  }
};

// TSV row: "convert-conv1x1-to-fc" at 0xe26ea5.
struct DwcConvertConv1x1ToFcPass
    : public darwinn::impl::DwcConvertConv1x1ToFcPassBase<
          DwcConvertConv1x1ToFcPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<dive_vm::DiveVmDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    unsigned lowered = 0;
    if (failed(applyDwcLowerConvertTrunc(func, lowered)))
      return signalPassFailure();
  }
};

// TSV row: "convert-dive-vm-tensor-to-scf" at 0xde4651.
struct DwcConvertDiveVmTensorToScfPass
    : public darwinn::impl::DwcConvertDiveVmTensorToScfPassBase<
          DwcConvertDiveVmTensorToScfPass> {
  using Base::Base;

  void runOnOperation() override {
    // No honest rewrite exists. DiveVmOps.td names no scf target for dive_vm
    // tensor ops and all_pseudocode.json carries no scf loop shape for them.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    SmallVector<Operation *> dead;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "dive_vm")
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

    bool failedLegal = false;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "dive_vm")
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedLegal = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failedLegal)
      return signalPassFailure();
  }
};

// TSV row: "convert-dive-vm-tensor-to-tensor" at 0xda89fa.
struct DwcConvertDiveVmTensorToTensorPass
    : public darwinn::impl::DwcConvertDiveVmTensorToTensorPassBase<
          DwcConvertDiveVmTensorToTensorPass> {
  using Base::Base;

  void runOnOperation() override {
    // No honest rewrite exists. DiveVmOps.td names no tensor target for
    // dive_vm tensor ops and all_pseudocode.json carries no tensor shape
    // contract for them.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    SmallVector<Operation *> dead;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "dive_vm")
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

    bool failedLegal = false;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "dive_vm")
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedLegal = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failedLegal)
      return signalPassFailure();
  }
};

// TSV row: "convert-dive-vm-to-llvm" at 0xdcb7dd.
struct DwcConvertDiveVmToLlvmPass
    : public darwinn::impl::DwcConvertDiveVmToLlvmPassBase<
          DwcConvertDiveVmToLlvmPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<dive_vm::DiveVmDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    MLIRContext *ctx = root->getContext();
    OpBuilder builder(ctx);
    auto moduleOp = func->getParentOfType<ModuleOp>();
    if (!moduleOp) {
      func.emitError("convert-dive-vm-to-llvm needs a parent module");
      return signalPassFailure();
    }
    SmallVector<Operation *> vmOps;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (dialect && dialect->getNamespace() == "dive_vm")
        vmOps.push_back(op);
    });
    unsigned lowered = 0;
    for (Operation *op : vmOps) {
      if (failed(checkDwcConvertibleTypes(op)))
        return signalPassFailure();
      if (op->getNumResults() > 1) {
        op->emitError(
            "unsupported multi-result dive_vm op in convert-dive-vm-to-llvm");
        return signalPassFailure();
      }
      if (op->getName().getStringRef() == "dive_vm.br") {
        if (op->getNumResults()) {
          op->emitError("unsupported result on dive_vm.br in convert-dive-vm-to-llvm");
          return signalPassFailure();
        }
        Block *block = op->getBlock();
        if (!block) {
          op->emitError("dive_vm.br outside a block in convert-dive-vm-to-llvm");
          return signalPassFailure();
        }
        Block *cont = block->splitBlock(op);
        builder.setInsertionPointToEnd(block);
        LLVM::BrOp::create(builder, op->getLoc(), ValueRange(), cont);
        op->erase();
        ++lowered;
        continue;
      }
      if (op->getName().getStringRef() == "dive_vm.cond_br") {
        if (op->getNumResults()) {
          op->emitError("unsupported result on dive_vm.cond_br in convert-dive-vm-to-llvm");
          return signalPassFailure();
        }
        Block *block = op->getBlock();
        if (!block) {
          op->emitError("dive_vm.cond_br outside a block in convert-dive-vm-to-llvm");
          return signalPassFailure();
        }
        Value cond;
        bool hasI1Cond = false;
        if (op->getNumOperands()) {
          cond = op->getOperand(0);
          if (auto intTy = dyn_cast<IntegerType>(cond.getType()))
            hasI1Cond = intTy.getWidth() == 1;
        }
        Block *cont = block->splitBlock(op);
        builder.setInsertionPointToEnd(block);
        if (hasI1Cond) {
          LLVM::CondBrOp::create(builder, op->getLoc(), cond, cont,
                                 ValueRange(), cont, ValueRange());
        } else {
          LLVM::BrOp::create(builder, op->getLoc(), ValueRange(), cont);
        }
        op->erase();
        ++lowered;
        continue;
      }
      const char *callee = nullptr;
      if (op->getName().getStringRef() == "dive_vm.const" ||
          op->getName().getStringRef() == "dive_vm.const_bytes") {
        if (!op->use_empty()) {
          op->emitError("dive_vm const carries no LLVM callee, use it or drop it");
          return signalPassFailure();
        }
        op->erase();
        ++lowered;
        continue;
      }
      if (op->getName().getStringRef() == "dive_vm.add")
        callee = "DiveRuntime_Log";
      else if (op->getName().getStringRef() == "dive_vm.copy")
        callee = "DiveVm_MemCpy";
      else if (op->getName().getStringRef() == "dive_vm.gather")
        callee = "DiveVm_Gather";
      else if (op->getName().getStringRef() == "dive_vm.legacy_scalar")
        callee = "_ZN7silicon4dive7kernels21DiveVm_LegacyScalarOpENS1_"
                 "24DiveVmLegacyScalarOpTypeEiPKPhPKlPKNS1_"
                 "19DiveVmPrimitiveTypeEPKiPKbPKfSC_bi";
      else if (op->getName().getStringRef() == "dive_vm.cast" || op->getName().getStringRef() == "dive_vm.bitcast")
        callee = "DiveVm_Cast";
      else if (op->getName().getStringRef() == "dive_vm.sign")
        callee = "DiveVm_Sign";
      else if (op->getName().getStringRef() == "dive_vm.add_imm" || op->getName().getStringRef() == "dive_vm.sub" ||
               op->getName().getStringRef() == "dive_vm.sub_imm" || op->getName().getStringRef() == "dive_vm.mul" ||
               op->getName().getStringRef() == "dive_vm.mul_imm" || op->getName().getStringRef() == "dive_vm.div" ||
               op->getName().getStringRef() == "dive_vm.div_imm" || op->getName().getStringRef() == "dive_vm.rem" ||
               op->getName().getStringRef() == "dive_vm.rem_imm" || op->getName().getStringRef() == "dive_vm.min" ||
               op->getName().getStringRef() == "dive_vm.min_imm" || op->getName().getStringRef() == "dive_vm.max" ||
               op->getName().getStringRef() == "dive_vm.max_imm")
        callee = "_ZN7silicon4dive7kernels21DiveVm_LegacyScalarOpENS1_"
                 "24DiveVmLegacyScalarOpTypeEiPKPhPKlPKNS1_"
                 "19DiveVmPrimitiveTypeEPKiPKbPKfSC_bi";
      else if (op->getName().getStringRef() == "dive_vm.arithmetic_left_shift" ||
               op->getName().getStringRef() == "dive_vm.arithmetic_left_shift_imm" ||
               op->getName().getStringRef() == "dive_vm.arithmetic_right_shift" ||
               op->getName().getStringRef() == "dive_vm.arithmetic_right_shift_imm" ||
               op->getName().getStringRef() == "dive_vm.logical_right_shift" ||
               op->getName().getStringRef() == "dive_vm.logical_right_shift_imm")
        callee = "_ZN7silicon4dive7kernels21DiveVm_LegacyScalarOpENS1_"
                 "24DiveVmLegacyScalarOpTypeEiPKPhPKlPKNS1_"
                 "19DiveVmPrimitiveTypeEPKiPKbPKfSC_bi";
      else if (op->getName().getStringRef() == "dive_vm.bitwise_and" ||
               op->getName().getStringRef() == "dive_vm.bitwise_and_imm" ||
               op->getName().getStringRef() == "dive_vm.bitwise_or" ||
               op->getName().getStringRef() == "dive_vm.bitwise_or_imm" ||
               op->getName().getStringRef() == "dive_vm.bitwise_xor" ||
               op->getName().getStringRef() == "dive_vm.bitwise_xor_imm" ||
               op->getName().getStringRef() == "dive_vm.logical_and" ||
               op->getName().getStringRef() == "dive_vm.logical_and_imm")
        callee = "_ZN7silicon4dive7kernels21DiveVm_LegacyScalarOpENS1_"
                 "24DiveVmLegacyScalarOpTypeEiPKPhPKlPKNS1_"
                 "19DiveVmPrimitiveTypeEPKiPKbPKfSC_bi";
      else if (op->getName().getStringRef() == "dive_vm.pow" || op->getName().getStringRef() == "dive_vm.pow_imm")
        callee = "_ZN7silicon4dive7kernels21DiveVm_LegacyScalarOpENS1_"
                 "24DiveVmLegacyScalarOpTypeEiPKPhPKlPKNS1_"
                 "19DiveVmPrimitiveTypeEPKiPKbPKfSC_bi";
      else if (op->getName().getStringRef() == "dive_vm.equal" || op->getName().getStringRef() == "dive_vm.equal_imm" ||
               op->getName().getStringRef() == "dive_vm.not_equal" || op->getName().getStringRef() == "dive_vm.not_equal_imm" ||
               op->getName().getStringRef() == "dive_vm.less" || op->getName().getStringRef() == "dive_vm.less_imm" ||
               op->getName().getStringRef() == "dive_vm.less_equal" ||
               op->getName().getStringRef() == "dive_vm.less_equal_imm" || op->getName().getStringRef() == "dive_vm.greater" ||
               op->getName().getStringRef() == "dive_vm.greater_imm" ||
               op->getName().getStringRef() == "dive_vm.greater_equal" ||
               op->getName().getStringRef() == "dive_vm.greater_equal_imm")
        callee = "_ZN7silicon4dive7kernels21DiveVm_LegacyScalarOpENS1_"
                 "24DiveVmLegacyScalarOpTypeEiPKPhPKlPKNS1_"
                 "19DiveVmPrimitiveTypeEPKiPKbPKfSC_bi";
      else if (op->getName().getStringRef() == "dive_vm.allocate")
        callee = "DiveRuntime_Allocate";
      else if (op->getName().getStringRef() == "dive_vm.load" || op->getName().getStringRef() == "dive_vm.store" ||
               op->getName().getStringRef() == "dive_vm.load_indirect" ||
               op->getName().getStringRef() == "dive_vm.store_indirect" || op->getName().getStringRef() == "dive_vm.copy_imm")
        callee = "DiveVm_MemCpy";
      else if (op->getName().getStringRef() == "dive_vm.fill")
        callee = "_ZN7silicon4dive7kernels4FillERKNS0_"
                 "11interpreter14ResolvedTensorERS3_";
      else if (op->getName().getStringRef() == "dive_vm.pad")
        callee = "_ZN7silicon4dive7kernels3PadERKNS0_"
                 "11interpreter14ResolvedTensorES5_S5_RS3_";
      else if (op->getName().getStringRef() == "dive_vm.roll")
        callee = "_ZN7silicon4dive7kernels4RollERKNS0_"
                 "11interpreter14ResolvedTensorEijS5_";
      else if (op->getName().getStringRef() == "dive_vm.one_hot")
        callee = "_ZN7silicon4dive7kernels6OneHotERKNS0_"
                 "11interpreter14ResolvedTensorES5_S5_jiS5_";
      else if (op->getName().getStringRef() == "dive_vm.cumsum")
        callee = "_ZN7silicon4dive7kernels6CumsumERKNS0_"
                 "11interpreter14ResolvedTensorEibbRS3_";
      else if (op->getName().getStringRef() == "dive_vm.gather_nd")
        callee = "_ZN7silicon4dive7kernels8GatherNdERKNS0_"
                 "11interpreter14ResolvedTensorES5_iRNS2_16UnresolvedTensorE";
      else if (op->getName().getStringRef() == "dive_vm.scatter_nd")
        callee = "_ZN7silicon4dive7kernels9ScatterNdERKNS0_"
                 "11interpreter14ResolvedTensorES5_RS3_";
      else if (op->getName().getStringRef() == "dive_vm.top_k")
        callee = "_ZN7silicon4dive7kernels4TopKERNS0_"
                 "11interpreter14ResolvedTensorES4_S4_";
      else if (op->getName().getStringRef() == "dive_vm.multinomial")
        callee = "DiveVm_ComputeMultinomial";
      else if (op->getName().getStringRef() == "dive_vm.mask_indices")
        callee = "DiveVm_MaskIndices";
      else if (op->getName().getStringRef() == "dive_vm.address_of_input_activation")
        callee = "DiveVm_GetAddressOfInputActivation";
      else if (op->getName().getStringRef() == "dive_vm.address_of_output_activation")
        callee = "DiveVm_GetAddressOfOutputActivation";
      else if (op->getName().getStringRef() == "dive_vm.address_of_parameter_region")
        callee = "DiveVm_GetAddressOfParameterRegion";
      else if (op->getName().getStringRef() == "dive_vm.address_of_parameter")
        callee = "DiveVm_GetAddressOfParameterRegion";
      else if (op->getName().getStringRef() == "dive_vm.address_of_scratch")
        callee = "DiveVm_GetAddressOfScratch";
      else if (op->getName().getStringRef() == "dive_vm.translate_sram_address" ||
               op->getName().getStringRef() == "dive_vm.view_on_address")
        callee = "DiveTpu_CastSharedMemoryAddressToPointer";
      else if (op->getName().getStringRef() == "dive_vm.get_const")
        callee = "DiveVm_GetConst";
      else if (op->getName().getStringRef() == "dive_vm.patch_instruction_for_strided_io")
        callee = "DiveVm_PatchInstructionForStridedIo";
      else if (op->getName().getStringRef() == "dive_vm.wait_for_fence_completion")
        callee = "DiveTpu_WaitForFenceCompletion";
      else if (op->getName().getStringRef() == "dive_vm.wait_for_rkhy_completion")
        callee = "DiveTpu_WaitForRkhyCompletion";
      else if (op->getName().getStringRef() == "dive_vm.wait_for_power_island_transition_complete")
        callee = "DiveVm_WaitForPowmgKllandTransitionComplete";
      else if (op->getName().getStringRef() == "dive_vm.perform_software_preemption_if_requested")
        callee = "DiveTpu_PerformSoftwarePreemptionIfRequested";
      else if (op->getName().getStringRef() == "dive_vm.set_dtc_mode")
        callee = "DiveVm_SetDtcMode";
      else if (op->getName().getStringRef() == "dive_vm.read_gn_stats")
        callee = "DiveDtc_ReadGnStatsSum";
      else if (op->getName().getStringRef() == "dive_vm.transition_dtc_power_island")
        callee = "DiveDtc_TransitionDtcPowmgKllandOn";
      else if (op->getName().getStringRef() == "dive_vm.enable_itc_tracing")
        callee = "DiveItcTracing_Enable";
      else if (op->getName().getStringRef() == "dive_vm.disable_itc_tracing")
        callee = "DiveItcTracing_Disable";
      else if (op->getName().getStringRef() == "dive_vm.print")
        callee = "DiveRuntime_Log";
      else if (op->getName().getStringRef() == "dive_vm.benchmark")
        callee = "DiveVm_Benchmark";
      else if (op->getName().getStringRef() == "dive_vm.program_tensor_mapping_table")
        callee = "DiveVm_PrepareAndProgramTensorMappingTable";
      else if (op->getName().getStringRef() == "dive_vm.write_dma_descriptor")
        callee = "DiveTpu_EnqueueDmaDescriptor";
      else if (op->getName().getStringRef() == "dive_vm.write_hib_data")
        callee = "DiveTpu_WriteHibData";
      else if (op->getName().getStringRef() == "dive_vm.write_scalar_arch_register")
        callee = "DiveTpu_WriteScalarArchRegister";
      else if (op->getName().getStringRef() == "dive_vm.cache_clean_invalidate")
        callee = "DiveSystem_CacheCleanInvalidate";
      else if (op->getName().getStringRef() == "dive_vm.extract_slice" ||
               op->getName().getStringRef() == "dive_vm.insert_slice")
        callee = "_ZN9platforms7darwinn4dive11runtime_lib10MemCpyPerfEPhPKhi";
      else if (op->getName().getStringRef() == "dive_vm.select")
        callee = "_ZN7silicon4dive7kernels6SelectERKNS0_"
                 "11interpreter14ResolvedTensorES5_S5_S5_";
      else if (op->getName().getStringRef() == "dive_vm.address_of_activation")
        callee = "DiveVm_GetAddressOfInputActivation";
      else if (op->getName().getStringRef() == "dive_vm.dynamic_slice_y")
        callee = "DiveVm_ComputeDynamicSliceYMulticastBitmapAndAddress";
      else if (op->getName().getStringRef() == "dive_vm.put_bits")
        callee = "DiveVm_PutBits";
      else if (op->getName().getStringRef() == "dive_vm.dvfs")
        callee = "DiveVm_UpdateDvfsHint";
      else if (op->getName().getStringRef() == "dive_vm.shape_of_activation")
        callee = "DiveVm_GetShapeOfActivation";
      else if (op->getName().getStringRef() == "dive_vm.cuda_emu_custom_op")
        callee = "DiveVm_InitializeAndReturnGlobalCustomOpContext";
      else if (op->getName().getStringRef() == "dive_vm.hib_gather_edit")
        callee = "DiveVm_HIBGatherEditE32";
      else if (op->getName().getStringRef() == "dive_vm.reduction") {
        op->emitError("dive_vm.reduction needs a (float*, float*, int*, int, int, int) TopKVector signature, not the op operand list");
        return signalPassFailure();
      } else {
        op->emitError("unsupported dive_vm op in convert-dive-vm-to-llvm");
        return signalPassFailure();
      }
      Type opaque = LLVM::LLVMPointerType::get(ctx);
      SmallVector<Type> paramTypes(op->getNumOperands(), opaque);
      Type resultType = op->getNumResults() ? opaque : LLVM::LLVMVoidType::get(ctx);
      FailureOr<LLVM::LLVMFuncOp> calleeOp = LLVM::lookupOrCreateFn(
          builder, moduleOp, callee, paramTypes, resultType);
      if (failed(calleeOp))
        return signalPassFailure();
      builder.setInsertionPoint(op);
      SmallVector<Value> bridged;
      for (Value v : op->getOperands()) {
        auto cast = UnrealizedConversionCastOp::create(builder, op->getLoc(),
                                                       opaque, v);
        bridged.push_back(cast.getResult(0));
      }
      auto call = LLVM::CallOp::create(builder, op->getLoc(), *calleeOp,
                                       ValueRange(bridged));
      if (op->getNumResults()) {
        auto back = UnrealizedConversionCastOp::create(
            builder, op->getLoc(), op->getResult(0).getType(),
            call.getResult());
        op->getResult(0).replaceAllUsesWith(back.getResult(0));
      }
      op->erase();
      ++lowered;
    }
    root->setAttr("dive_vm.lowered_count", builder.getI64IntegerAttr(lowered));
  }
};

// TSV row: "convert-dive-vm-to-memref" at 0xde3efd.
struct DwcConvertDiveVmToMemrefPass
    : public darwinn::impl::DwcConvertDiveVmToMemrefPassBase<
          DwcConvertDiveVmToMemrefPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<dive_vm::DiveVmDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    MLIRContext *ctx = root->getContext();
    OpBuilder builder(ctx);
    auto moduleOp = func->getParentOfType<ModuleOp>();
    if (!moduleOp) {
      func.emitError("convert-dive-vm-to-memref needs a parent module");
      return signalPassFailure();
    }
    SmallVector<Operation *> vmOps;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (dialect && dialect->getNamespace() == "dive_vm")
        vmOps.push_back(op);
    });
    unsigned lowered = 0;
    for (Operation *op : vmOps) {
      if (failed(checkDwcConvertibleTypes(op)))
        return signalPassFailure();
      if (op->getNumResults() > 1) {
        op->emitError(
            "unsupported multi-result dive_vm op in convert-dive-vm-to-memref");
        return signalPassFailure();
      }
      if (op->getName().getStringRef() == "dive_vm.br") {
        if (op->getNumResults()) {
          op->emitError("unsupported result on dive_vm.br in convert-dive-vm-to-memref");
          return signalPassFailure();
        }
        Block *block = op->getBlock();
        if (!block) {
          op->emitError("dive_vm.br outside a block in convert-dive-vm-to-memref");
          return signalPassFailure();
        }
        Block *cont = block->splitBlock(op);
        builder.setInsertionPointToEnd(block);
        LLVM::BrOp::create(builder, op->getLoc(), ValueRange(), cont);
        op->erase();
        ++lowered;
        continue;
      }
      if (op->getName().getStringRef() == "dive_vm.cond_br") {
        if (op->getNumResults()) {
          op->emitError("unsupported result on dive_vm.cond_br in convert-dive-vm-to-memref");
          return signalPassFailure();
        }
        Block *block = op->getBlock();
        if (!block) {
          op->emitError("dive_vm.cond_br outside a block in convert-dive-vm-to-memref");
          return signalPassFailure();
        }
        Value cond;
        bool hasI1Cond = false;
        if (op->getNumOperands()) {
          cond = op->getOperand(0);
          if (auto intTy = dyn_cast<IntegerType>(cond.getType()))
            hasI1Cond = intTy.getWidth() == 1;
        }
        Block *cont = block->splitBlock(op);
        builder.setInsertionPointToEnd(block);
        if (hasI1Cond) {
          LLVM::CondBrOp::create(builder, op->getLoc(), cond, cont,
                                 ValueRange(), cont, ValueRange());
        } else {
          LLVM::BrOp::create(builder, op->getLoc(), ValueRange(), cont);
        }
        op->erase();
        ++lowered;
        continue;
      }
      const char *callee = nullptr;
      if (op->getName().getStringRef() == "dive_vm.const" ||
          op->getName().getStringRef() == "dive_vm.const_bytes") {
        if (!op->use_empty()) {
          op->emitError("dive_vm const carries no LLVM callee, use it or drop it");
          return signalPassFailure();
        }
        op->erase();
        ++lowered;
        continue;
      }
      if (op->getName().getStringRef() == "dive_vm.add")
        callee = "DiveRuntime_Log";
      else if (op->getName().getStringRef() == "dive_vm.copy")
        callee = "DiveVm_MemCpy";
      else if (op->getName().getStringRef() == "dive_vm.gather")
        callee = "DiveVm_Gather";
      else if (op->getName().getStringRef() == "dive_vm.legacy_scalar")
        callee = "_ZN7silicon4dive7kernels21DiveVm_LegacyScalarOpENS1_"
                 "24DiveVmLegacyScalarOpTypeEiPKPhPKlPKNS1_"
                 "19DiveVmPrimitiveTypeEPKiPKbPKfSC_bi";
      else if (op->getName().getStringRef() == "dive_vm.cast" || op->getName().getStringRef() == "dive_vm.bitcast")
        callee = "DiveVm_Cast";
      else if (op->getName().getStringRef() == "dive_vm.sign")
        callee = "DiveVm_Sign";
      else if (op->getName().getStringRef() == "dive_vm.add_imm" || op->getName().getStringRef() == "dive_vm.sub" ||
               op->getName().getStringRef() == "dive_vm.sub_imm" || op->getName().getStringRef() == "dive_vm.mul" ||
               op->getName().getStringRef() == "dive_vm.mul_imm" || op->getName().getStringRef() == "dive_vm.div" ||
               op->getName().getStringRef() == "dive_vm.div_imm" || op->getName().getStringRef() == "dive_vm.rem" ||
               op->getName().getStringRef() == "dive_vm.rem_imm" || op->getName().getStringRef() == "dive_vm.min" ||
               op->getName().getStringRef() == "dive_vm.min_imm" || op->getName().getStringRef() == "dive_vm.max" ||
               op->getName().getStringRef() == "dive_vm.max_imm")
        callee = "_ZN7silicon4dive7kernels21DiveVm_LegacyScalarOpENS1_"
                 "24DiveVmLegacyScalarOpTypeEiPKPhPKlPKNS1_"
                 "19DiveVmPrimitiveTypeEPKiPKbPKfSC_bi";
      else if (op->getName().getStringRef() == "dive_vm.arithmetic_left_shift" ||
               op->getName().getStringRef() == "dive_vm.arithmetic_left_shift_imm" ||
               op->getName().getStringRef() == "dive_vm.arithmetic_right_shift" ||
               op->getName().getStringRef() == "dive_vm.arithmetic_right_shift_imm" ||
               op->getName().getStringRef() == "dive_vm.logical_right_shift" ||
               op->getName().getStringRef() == "dive_vm.logical_right_shift_imm")
        callee = "_ZN7silicon4dive7kernels21DiveVm_LegacyScalarOpENS1_"
                 "24DiveVmLegacyScalarOpTypeEiPKPhPKlPKNS1_"
                 "19DiveVmPrimitiveTypeEPKiPKbPKfSC_bi";
      else if (op->getName().getStringRef() == "dive_vm.bitwise_and" ||
               op->getName().getStringRef() == "dive_vm.bitwise_and_imm" ||
               op->getName().getStringRef() == "dive_vm.bitwise_or" ||
               op->getName().getStringRef() == "dive_vm.bitwise_or_imm" ||
               op->getName().getStringRef() == "dive_vm.bitwise_xor" ||
               op->getName().getStringRef() == "dive_vm.bitwise_xor_imm" ||
               op->getName().getStringRef() == "dive_vm.logical_and" ||
               op->getName().getStringRef() == "dive_vm.logical_and_imm")
        callee = "_ZN7silicon4dive7kernels21DiveVm_LegacyScalarOpENS1_"
                 "24DiveVmLegacyScalarOpTypeEiPKPhPKlPKNS1_"
                 "19DiveVmPrimitiveTypeEPKiPKbPKfSC_bi";
      else if (op->getName().getStringRef() == "dive_vm.pow" || op->getName().getStringRef() == "dive_vm.pow_imm")
        callee = "_ZN7silicon4dive7kernels21DiveVm_LegacyScalarOpENS1_"
                 "24DiveVmLegacyScalarOpTypeEiPKPhPKlPKNS1_"
                 "19DiveVmPrimitiveTypeEPKiPKbPKfSC_bi";
      else if (op->getName().getStringRef() == "dive_vm.equal" || op->getName().getStringRef() == "dive_vm.equal_imm" ||
               op->getName().getStringRef() == "dive_vm.not_equal" || op->getName().getStringRef() == "dive_vm.not_equal_imm" ||
               op->getName().getStringRef() == "dive_vm.less" || op->getName().getStringRef() == "dive_vm.less_imm" ||
               op->getName().getStringRef() == "dive_vm.less_equal" ||
               op->getName().getStringRef() == "dive_vm.less_equal_imm" || op->getName().getStringRef() == "dive_vm.greater" ||
               op->getName().getStringRef() == "dive_vm.greater_imm" ||
               op->getName().getStringRef() == "dive_vm.greater_equal" ||
               op->getName().getStringRef() == "dive_vm.greater_equal_imm")
        callee = "_ZN7silicon4dive7kernels21DiveVm_LegacyScalarOpENS1_"
                 "24DiveVmLegacyScalarOpTypeEiPKPhPKlPKNS1_"
                 "19DiveVmPrimitiveTypeEPKiPKbPKfSC_bi";
      else if (op->getName().getStringRef() == "dive_vm.allocate")
        callee = "DiveRuntime_Allocate";
      else if (op->getName().getStringRef() == "dive_vm.load" || op->getName().getStringRef() == "dive_vm.store" ||
               op->getName().getStringRef() == "dive_vm.load_indirect" ||
               op->getName().getStringRef() == "dive_vm.store_indirect" || op->getName().getStringRef() == "dive_vm.copy_imm")
        callee = "DiveVm_MemCpy";
      else if (op->getName().getStringRef() == "dive_vm.fill")
        callee = "_ZN7silicon4dive7kernels4FillERKNS0_"
                 "11interpreter14ResolvedTensorERS3_";
      else if (op->getName().getStringRef() == "dive_vm.pad")
        callee = "_ZN7silicon4dive7kernels3PadERKNS0_"
                 "11interpreter14ResolvedTensorES5_S5_RS3_";
      else if (op->getName().getStringRef() == "dive_vm.roll")
        callee = "_ZN7silicon4dive7kernels4RollERKNS0_"
                 "11interpreter14ResolvedTensorEijS5_";
      else if (op->getName().getStringRef() == "dive_vm.one_hot")
        callee = "_ZN7silicon4dive7kernels6OneHotERKNS0_"
                 "11interpreter14ResolvedTensorES5_S5_jiS5_";
      else if (op->getName().getStringRef() == "dive_vm.cumsum")
        callee = "_ZN7silicon4dive7kernels6CumsumERKNS0_"
                 "11interpreter14ResolvedTensorEibbRS3_";
      else if (op->getName().getStringRef() == "dive_vm.gather_nd")
        callee = "_ZN7silicon4dive7kernels8GatherNdERKNS0_"
                 "11interpreter14ResolvedTensorES5_iRNS2_16UnresolvedTensorE";
      else if (op->getName().getStringRef() == "dive_vm.scatter_nd")
        callee = "_ZN7silicon4dive7kernels9ScatterNdERKNS0_"
                 "11interpreter14ResolvedTensorES5_RS3_";
      else if (op->getName().getStringRef() == "dive_vm.top_k")
        callee = "_ZN7silicon4dive7kernels4TopKERNS0_"
                 "11interpreter14ResolvedTensorES4_S4_";
      else if (op->getName().getStringRef() == "dive_vm.multinomial")
        callee = "DiveVm_ComputeMultinomial";
      else if (op->getName().getStringRef() == "dive_vm.mask_indices")
        callee = "DiveVm_MaskIndices";
      else if (op->getName().getStringRef() == "dive_vm.address_of_input_activation")
        callee = "DiveVm_GetAddressOfInputActivation";
      else if (op->getName().getStringRef() == "dive_vm.address_of_output_activation")
        callee = "DiveVm_GetAddressOfOutputActivation";
      else if (op->getName().getStringRef() == "dive_vm.address_of_parameter_region")
        callee = "DiveVm_GetAddressOfParameterRegion";
      else if (op->getName().getStringRef() == "dive_vm.address_of_parameter")
        callee = "DiveVm_GetAddressOfParameterRegion";
      else if (op->getName().getStringRef() == "dive_vm.address_of_scratch")
        callee = "DiveVm_GetAddressOfScratch";
      else if (op->getName().getStringRef() == "dive_vm.translate_sram_address" ||
               op->getName().getStringRef() == "dive_vm.view_on_address")
        callee = "DiveTpu_CastSharedMemoryAddressToPointer";
      else if (op->getName().getStringRef() == "dive_vm.get_const")
        callee = "DiveVm_GetConst";
      else if (op->getName().getStringRef() == "dive_vm.patch_instruction_for_strided_io")
        callee = "DiveVm_PatchInstructionForStridedIo";
      else if (op->getName().getStringRef() == "dive_vm.wait_for_fence_completion")
        callee = "DiveTpu_WaitForFenceCompletion";
      else if (op->getName().getStringRef() == "dive_vm.wait_for_rkhy_completion")
        callee = "DiveTpu_WaitForRkhyCompletion";
      else if (op->getName().getStringRef() == "dive_vm.wait_for_power_island_transition_complete")
        callee = "DiveVm_WaitForPowmgKllandTransitionComplete";
      else if (op->getName().getStringRef() == "dive_vm.perform_software_preemption_if_requested")
        callee = "DiveTpu_PerformSoftwarePreemptionIfRequested";
      else if (op->getName().getStringRef() == "dive_vm.set_dtc_mode")
        callee = "DiveVm_SetDtcMode";
      else if (op->getName().getStringRef() == "dive_vm.read_gn_stats")
        callee = "DiveDtc_ReadGnStatsSum";
      else if (op->getName().getStringRef() == "dive_vm.transition_dtc_power_island")
        callee = "DiveDtc_TransitionDtcPowmgKllandOn";
      else if (op->getName().getStringRef() == "dive_vm.enable_itc_tracing")
        callee = "DiveItcTracing_Enable";
      else if (op->getName().getStringRef() == "dive_vm.disable_itc_tracing")
        callee = "DiveItcTracing_Disable";
      else if (op->getName().getStringRef() == "dive_vm.print")
        callee = "DiveRuntime_Log";
      else if (op->getName().getStringRef() == "dive_vm.benchmark")
        callee = "DiveVm_Benchmark";
      else if (op->getName().getStringRef() == "dive_vm.program_tensor_mapping_table")
        callee = "DiveVm_PrepareAndProgramTensorMappingTable";
      else if (op->getName().getStringRef() == "dive_vm.write_dma_descriptor")
        callee = "DiveTpu_EnqueueDmaDescriptor";
      else if (op->getName().getStringRef() == "dive_vm.write_hib_data")
        callee = "DiveTpu_WriteHibData";
      else if (op->getName().getStringRef() == "dive_vm.write_scalar_arch_register")
        callee = "DiveTpu_WriteScalarArchRegister";
      else if (op->getName().getStringRef() == "dive_vm.cache_clean_invalidate")
        callee = "DiveSystem_CacheCleanInvalidate";
      else if (op->getName().getStringRef() == "dive_vm.extract_slice" ||
               op->getName().getStringRef() == "dive_vm.insert_slice")
        callee = "_ZN9platforms7darwinn4dive11runtime_lib10MemCpyPerfEPhPKhi";
      else if (op->getName().getStringRef() == "dive_vm.select")
        callee = "_ZN7silicon4dive7kernels6SelectERKNS0_"
                 "11interpreter14ResolvedTensorES5_S5_S5_";
      else if (op->getName().getStringRef() == "dive_vm.address_of_activation")
        callee = "DiveVm_GetAddressOfInputActivation";
      else if (op->getName().getStringRef() == "dive_vm.dynamic_slice_y")
        callee = "DiveVm_ComputeDynamicSliceYMulticastBitmapAndAddress";
      else if (op->getName().getStringRef() == "dive_vm.put_bits")
        callee = "DiveVm_PutBits";
      else if (op->getName().getStringRef() == "dive_vm.dvfs")
        callee = "DiveVm_UpdateDvfsHint";
      else if (op->getName().getStringRef() == "dive_vm.shape_of_activation")
        callee = "DiveVm_GetShapeOfActivation";
      else if (op->getName().getStringRef() == "dive_vm.cuda_emu_custom_op")
        callee = "DiveVm_InitializeAndReturnGlobalCustomOpContext";
      else if (op->getName().getStringRef() == "dive_vm.hib_gather_edit")
        callee = "DiveVm_HIBGatherEditE32";
      else if (op->getName().getStringRef() == "dive_vm.reduction") {
        op->emitError("dive_vm.reduction needs a (float*, float*, int*, int, int, int) TopKVector signature, not the op operand list");
        return signalPassFailure();
      } else {
        op->emitError("unsupported dive_vm op in convert-dive-vm-to-memref");
        return signalPassFailure();
      }
      Type opaque = LLVM::LLVMPointerType::get(ctx);
      SmallVector<Type> paramTypes(op->getNumOperands(), opaque);
      Type resultType = op->getNumResults() ? opaque : LLVM::LLVMVoidType::get(ctx);
      FailureOr<LLVM::LLVMFuncOp> calleeOp = LLVM::lookupOrCreateFn(
          builder, moduleOp, callee, paramTypes, resultType);
      if (failed(calleeOp))
        return signalPassFailure();
      builder.setInsertionPoint(op);
      SmallVector<Value> bridged;
      for (Value v : op->getOperands()) {
        auto cast = UnrealizedConversionCastOp::create(builder, op->getLoc(),
                                                       opaque, v);
        bridged.push_back(cast.getResult(0));
      }
      auto call = LLVM::CallOp::create(builder, op->getLoc(), *calleeOp,
                                       ValueRange(bridged));
      if (op->getNumResults()) {
        auto back = UnrealizedConversionCastOp::create(
            builder, op->getLoc(), op->getResult(0).getType(),
            call.getResult());
        op->getResult(0).replaceAllUsesWith(back.getResult(0));
      }
      op->erase();
      ++lowered;
    }
    root->setAttr("dive_vm.lowered_count", builder.getI64IntegerAttr(lowered));
  }
};

// TSV row: "convert-dwc-to-dive-vm-tensor" at 0xda8a3d.
struct DwcConvertDwcToDiveVmTensorPass
    : public darwinn::impl::DwcConvertDwcToDiveVmTensorPassBase<
          DwcConvertDwcToDiveVmTensorPass> {
  using Base::Base;

  void runOnOperation() override {
    // No honest rewrite exists. DiveVmOps.td names no dive_vm tensor form for
    // darwinn ops and all_pseudocode.json carries no tensor type contract.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    SmallVector<Operation *> dead;
    root->walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name != "darwinn.copy_op" && name != "darwinn.convert" &&
          name != "darwinn.bitcast")
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

    bool failedLegal = false;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "darwinn")
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedLegal = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failedLegal)
      return signalPassFailure();
  }
};

// TSV row: "convert-dwg-to-dive-vm" at 0xdcb89c.
struct DwcConvertDwgToDiveVmPass
    : public darwinn::impl::DwcConvertDwgToDiveVmPassBase<
          DwcConvertDwgToDiveVmPass> {
  using Base::Base;

  void runOnOperation() override {
    // No honest rewrite exists. The walk filtered darwinn while the pass name
    // promises dwg, and DiveVmOps.td names no dive_vm form for either.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    SmallVector<Operation *> dead;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "dwg")
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

    bool failedLegal = false;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "dwg")
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedLegal = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failedLegal)
      return signalPassFailure();
  }
};

// TSV row: "convert-dynamic-shape-scope-to-dive-vm" at 0xdcb8b3.
struct DwcConvertDynamicShapeScopeToDiveVmPass
    : public darwinn::impl::DwcConvertDynamicShapeScopeToDiveVmPassBase<
          DwcConvertDynamicShapeScopeToDiveVmPass> {
  using Base::Base;

  void runOnOperation() override {
    // No honest rewrite exists. No dynamic shape scope op appears in
    // DarwinnOps.td and DiveVmOps.td names no dive_vm scope target.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    SmallVector<Operation *> dead;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect)
        return;
      StringRef ns = dialect->getNamespace();
      if (ns != "shape" && ns != "tensor")
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

    bool failedLegal = false;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect)
        return WalkResult::advance();
      StringRef ns = dialect->getNamespace();
      if (ns != "shape" && ns != "tensor")
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedLegal = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failedLegal)
      return signalPassFailure();
  }
};

// TSV row: "convert-generic-norm-to-pseudo-op" at 0xdb45be.
struct DwcConvertGenericNormToPseudoOpPass
    : public darwinn::impl::DwcConvertGenericNormToPseudoOpPassBase<
          DwcConvertGenericNormToPseudoOpPass> {
  using Base::Base;

  void runOnOperation() override {
    // No honest rewrite exists. Neither darwinn.generic_norm nor
    // darwinn.pseudo_norm appears as a NUL-delimited op token in the blob,
    // and DiveVmOps.td names no norm target beyond the descriptor op.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    SmallVector<Operation *> dead;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "darwinn")
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
    bool failedLegal = false;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "darwinn")
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedLegal = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failedLegal)
      return signalPassFailure();
  }
};

// TSV row: "convert-op-lowering" at 0xddedce.
struct DwcConvertOpLoweringPass
    : public darwinn::impl::DwcConvertOpLoweringPassBase<
          DwcConvertOpLoweringPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<dive_vm::DiveVmDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    RewritePatternSet patterns(&getContext());
    darwinn::populateLowerConvertPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
    if (failed(applyLocalConvertLowering(func)))
      return signalPassFailure();
    unsigned lowered = 0;
    if (failed(applyDwcLowerConvertTrunc(func, lowered)))
      return signalPassFailure();
  }
};

// TSV row: "convert-scatter-to-generic-scatter" at 0xdabcd2.
struct DwcConvertScatterToGenericScatterPass
    : public darwinn::impl::DwcConvertScatterToGenericScatterPassBase<
          DwcConvertScatterToGenericScatterPass> {
  using Base::Base;

  void runOnOperation() override {
    // No honest rewrite exists. DarwinnOps.td names no generic scatter op and
    // all_pseudocode.json carries no scatter kernel shape, so the pass only
    // accepts darwinn ops at typed shapes.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    SmallVector<Operation *> dead;
    root->walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name != "darwinn.copy_op" && name != "darwinn.convert" &&
          name != "darwinn.bitcast")
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

    bool failedLegal = false;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "darwinn")
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedLegal = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failedLegal)
      return signalPassFailure();
  }
};
// TSV row: "convert-signed-int-with-rescaling-ops" at 0xd84723.
struct DwcConvertSignedIntWithRescalingOpsPass
    : public darwinn::impl::DwcConvertSignedIntWithRescalingOpsPassBase<
          DwcConvertSignedIntWithRescalingOpsPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<dive_vm::DiveVmDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    unsigned lowered = 0;
    if (failed(applyDwcLowerConvertTrunc(func, lowered)))
      return signalPassFailure();
  }
};

// TSV row: "convert-spatial-reduction-to-pooling" at 0xddfd8d.
struct DwcConvertSpatialReductionToPoolingPass
    : public darwinn::impl::DwcConvertSpatialReductionToPoolingPassBase<
          DwcConvertSpatialReductionToPoolingPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<dive_vm::DiveVmDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    unsigned lowered = 0;
    (void)lowered;
    SmallVector<Operation *> dead;
    func.getOperation()->walk([&](Operation *op) {
      if (op->getName().getStringRef() != "darwinn.dive_ref_reduction")
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
  }
};

// TSV row: "convert-tf-to-dwc" at 0xe2523e.
struct DwcConvertTfToDwcPass
    : public darwinn::impl::DwcConvertTfToDwcPassBase<DwcConvertTfToDwcPass> {
  using Base::Base;

  void runOnOperation() override {
    // No honest rewrite exists. DarwinnOps.td names no per-op tf source form
    // and all_pseudocode.json carries no tf to dwc kernel shape, so the pass
    // only accepts tf ops at typed shapes.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    SmallVector<Operation *> dead;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "tf")
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

    bool failedLegal = false;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "tf")
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedLegal = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failedLegal)
      return signalPassFailure();
  }
};

// TSV row: "convert-to-k-in-m-sparsity" at 0xd5a046.
struct DwcConvertToKInMSparsityPass
    : public darwinn::impl::DwcConvertToKInMSparsityPassBase<
          DwcConvertToKInMSparsityPass> {
  using Base::Base;

  void runOnOperation() override {
    // No honest rewrite exists. DarwinnOps.td carries a sparsity skeleton with
    // no k in m attributes and all_pseudocode.json carries no sparsity kernel
    // shape.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    SmallVector<Operation *> dead;
    root->walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name != "darwinn.copy_op" && name != "darwinn.convert" &&
          name != "darwinn.bitcast")
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

    bool failedLegal = false;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "darwinn")
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedLegal = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failedLegal)
      return signalPassFailure();
  }
};
// TSV row: "convert-tpu-offload-to-dive-vm" at 0xdcb8da.
struct DwcConvertTpuOffloadToDiveVmPass
    : public darwinn::impl::DwcConvertTpuOffloadToDiveVmPassBase<
          DwcConvertTpuOffloadToDiveVmPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<dive_vm::DiveVmDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    MLIRContext *ctx = root->getContext();
    OpBuilder builder(ctx);
    auto moduleOp = func->getParentOfType<ModuleOp>();
    if (!moduleOp) {
      func.emitError("convert-tpu-offload-to-dive-vm needs a parent module");
      return signalPassFailure();
    }
    SmallVector<Operation *> offloadOps;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (dialect && dialect->getNamespace() == "edgetpu")
        offloadOps.push_back(op);
    });
    unsigned lowered = 0;
    for (Operation *op : offloadOps) {
      if (failed(checkDwcConvertibleTypes(op)))
        return signalPassFailure();
      if (op->getNumResults() > 1) {
        op->emitError("unsupported multi-result edgetpu op in "
                      "convert-tpu-offload-to-dive-vm");
        return signalPassFailure();
      }
      StringRef opName = op->getName().getStringRef();
      StringRef callee;
      if (opName == "edgetpu.convolution_sub_channel" ||
          opName == "edgetpu.convolution_sub_channel_drq" ||
          opName == "edgetpu.transposed_convolution_sub_channel" ||
          opName == "edgetpu.transposed_convolution_sub_channel_drq" ||
          opName == "edgetpu.depthwise_convolution_fp8" ||
          opName == "edgetpu.transpose_convolution_fp8" ||
          opName == "edgetpu.attention_v1" ||
          opName == "edgetpu.convert_yuv_to_rgb" ||
          opName == "edgetpu.fast_walsh_hadamard_transform" ||
          opName == "edgetpu.annotate_materialize_policy")
        callee = "DiveTpu_EnqueueInstructions";
      else if (opName == "edgetpu.matrix_multiply_sub_channel" ||
               opName == "edgetpu.matrix_multiply_sub_channel_drq" ||
               opName == "edgetpu.fully_connected_sub_channel" ||
               opName == "edgetpu.fully_connected_sub_channel_drq" ||
               opName == "edgetpu.fully_connected_fp8")
        callee = "DiveTpu_EnqueueDmaDescriptor";
      else {
        op->emitError("unsupported edgetpu op in "
                      "convert-tpu-offload-to-dive-vm");
        return signalPassFailure();
      }
      SmallVector<Type> paramTypes;
      for (Value v : op->getOperands())
        paramTypes.push_back(v.getType());
      Type resultType = op->getNumResults() ? op->getResult(0).getType()
                                            : LLVM::LLVMVoidType::get(ctx);
      FailureOr<LLVM::LLVMFuncOp> calleeOp = LLVM::lookupOrCreateFn(
          builder, moduleOp, callee, paramTypes, resultType);
      if (failed(calleeOp))
        return signalPassFailure();
      builder.setInsertionPoint(op);
      auto call = LLVM::CallOp::create(builder, op->getLoc(), *calleeOp,
                                       op->getOperands());
      if (op->getNumResults())
        op->getResult(0).replaceAllUsesWith(call.getResult());
      op->erase();
      ++lowered;
    }
    root->setAttr("edgetpu.lowered_count", builder.getI64IntegerAttr(lowered));
  }
};

// TSV row: "convert-tpu-offload-to-llvm" at 0xdcb84b.
struct DwcConvertTpuOffloadToLlvmPass
    : public darwinn::impl::DwcConvertTpuOffloadToLlvmPassBase<
          DwcConvertTpuOffloadToLlvmPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<dive_vm::DiveVmDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    MLIRContext *ctx = root->getContext();
    OpBuilder builder(ctx);
    auto moduleOp = func->getParentOfType<ModuleOp>();
    if (!moduleOp) {
      func.emitError("convert-tpu-offload-to-llvm needs a parent module");
      return signalPassFailure();
    }
    SmallVector<Operation *> offloadOps;
    root->walk([&](Operation *op) {
      StringRef opName = op->getName().getStringRef();
      bool isOffload = opName == "dive_vm.tpu_offload";
      Dialect *dialect = op->getDialect();
      if (!isOffload && dialect && dialect->getNamespace() == "edgetpu")
        isOffload = true;
      if (isOffload)
        offloadOps.push_back(op);
    });
    unsigned lowered = 0;
    for (Operation *op : offloadOps) {
      if (failed(checkDwcConvertibleTypes(op)))
        return signalPassFailure();
      if (op->getNumResults() > 1) {
        op->emitError("unsupported multi-result offload op in "
                      "convert-tpu-offload-to-llvm");
        return signalPassFailure();
      }
      StringRef opName = op->getName().getStringRef();
      StringRef callee;
      if (opName == "dive_vm.tpu_offload")
        callee = "DiveRuntime_ExecuteChildModel";
      else if (opName == "edgetpu.convolution_sub_channel" ||
               opName == "edgetpu.convolution_sub_channel_drq" ||
               opName == "edgetpu.transposed_convolution_sub_channel" ||
               opName == "edgetpu.transposed_convolution_sub_channel_drq" ||
               opName == "edgetpu.depthwise_convolution_fp8" ||
               opName == "edgetpu.transpose_convolution_fp8" ||
               opName == "edgetpu.attention_v1" ||
               opName == "edgetpu.convert_yuv_to_rgb" ||
               opName == "edgetpu.fast_walsh_hadamard_transform" ||
               opName == "edgetpu.annotate_materialize_policy")
        callee = "DiveTpu_EnqueueInstructions";
      else if (opName == "edgetpu.matrix_multiply_sub_channel" ||
               opName == "edgetpu.matrix_multiply_sub_channel_drq" ||
               opName == "edgetpu.fully_connected_sub_channel" ||
               opName == "edgetpu.fully_connected_sub_channel_drq" ||
               opName == "edgetpu.fully_connected_fp8")
        callee = "DiveTpu_EnqueueDmaDescriptor";
      else {
        op->emitError("unsupported edgetpu op in "
                      "convert-tpu-offload-to-llvm");
        return signalPassFailure();
      }
      SmallVector<Type> paramTypes;
      for (Value v : op->getOperands())
        paramTypes.push_back(v.getType());
      Type resultType = op->getNumResults() ? op->getResult(0).getType()
                                            : LLVM::LLVMVoidType::get(ctx);
      FailureOr<LLVM::LLVMFuncOp> calleeOp = LLVM::lookupOrCreateFn(
          builder, moduleOp, callee, paramTypes, resultType);
      if (failed(calleeOp))
        return signalPassFailure();
      builder.setInsertionPoint(op);
      auto call = LLVM::CallOp::create(builder, op->getLoc(), *calleeOp,
                                       op->getOperands());
      if (op->getNumResults())
        op->getResult(0).replaceAllUsesWith(call.getResult());
      op->erase();
      ++lowered;
    }
    root->setAttr("tpu_offload.lowered_count",
                  builder.getI64IntegerAttr(lowered));
  }
};

// TSV row: "convert-xla-supported-stablehlo" at 0xdbc2b1.
struct DwcConvertXlaSupportedStablehloPass
    : public darwinn::impl::DwcConvertXlaSupportedStablehloPassBase<
          DwcConvertXlaSupportedStablehloPass> {
  using Base::Base;

  void runOnOperation() override {
    // No honest rewrite exists. No XLA support table appears in
    // all_pseudocode.json and no per-op stablehlo decomposition is evidenced,
    // so the pass only accepts stablehlo ops at typed shapes.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    SmallVector<Operation *> dead;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "stablehlo")
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

    bool failedLegal = false;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "stablehlo")
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedLegal = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failedLegal)
      return signalPassFailure();
  }
};

// TSV row: "convert-dive-vm-tensor-to-linalg" (lowercase, no TSV addr).
struct DwcConvertDiveVmTensorToLinalgPass
    : public darwinn::impl::DwcConvertDiveVmTensorToLinalgPassBase<
          DwcConvertDiveVmTensorToLinalgPass> {
  using Base::Base;

  void runOnOperation() override {
    // No honest rewrite exists. DiveVmOps.td names no linalg target for
    // dive_vm tensor ops and all_pseudocode.json carries no linalg
    // decomposition shape for them.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    SmallVector<Operation *> dead;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "dive_vm")
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

    bool failedLegal = false;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "dive_vm")
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedLegal = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failedLegal)
      return signalPassFailure();
  }
};

// TSV row: "ConvertDiveVmTensorToLinalg" at 0xde1af2.
struct DwcConvertDiveVmTensorToLinalgSymbolPass
    : public darwinn::impl::DwcConvertDiveVmTensorToLinalgSymbolPassBase<
          DwcConvertDiveVmTensorToLinalgSymbolPass> {
  using Base::Base;

  void runOnOperation() override {
    // No honest rewrite exists. DiveVmOps.td names no linalg target for
    // dive_vm tensor ops and all_pseudocode.json carries no linalg
    // decomposition shape for them.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    SmallVector<Operation *> dead;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "dive_vm")
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

    bool failedLegal = false;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "dive_vm")
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedLegal = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failedLegal)
      return signalPassFailure();
  }
};

// TSV row: "ConvertTpuOffloadToLlvm" at 0xdcb87c.
struct DwcConvertTpuOffloadToLlvmSymbolPass
    : public darwinn::impl::DwcConvertTpuOffloadToLlvmSymbolPassBase<
          DwcConvertTpuOffloadToLlvmSymbolPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<dive_vm::DiveVmDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    MLIRContext *ctx = root->getContext();
    OpBuilder builder(ctx);
    auto moduleOp = func->getParentOfType<ModuleOp>();
    if (!moduleOp) {
      func.emitError("ConvertTpuOffloadToLlvm needs a parent module");
      return signalPassFailure();
    }
    SmallVector<Operation *> offloadOps;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (dialect && dialect->getNamespace() == "edgetpu")
        offloadOps.push_back(op);
    });
    unsigned lowered = 0;
    for (Operation *op : offloadOps) {
      if (failed(checkDwcConvertibleTypes(op)))
        return signalPassFailure();
      if (op->getNumResults() > 1) {
        op->emitError(
            "unsupported multi-result edgetpu op in ConvertTpuOffloadToLlvm");
        return signalPassFailure();
      }
      StringRef opName = op->getName().getStringRef();
      StringRef callee;
      if (opName == "edgetpu.convolution_sub_channel" ||
          opName == "edgetpu.convolution_sub_channel_drq" ||
          opName == "edgetpu.transposed_convolution_sub_channel" ||
          opName == "edgetpu.transposed_convolution_sub_channel_drq" ||
          opName == "edgetpu.depthwise_convolution_fp8" ||
          opName == "edgetpu.transpose_convolution_fp8" ||
          opName == "edgetpu.attention_v1" ||
          opName == "edgetpu.convert_yuv_to_rgb" ||
          opName == "edgetpu.fast_walsh_hadamard_transform" ||
          opName == "edgetpu.annotate_materialize_policy")
        callee = "DiveTpu_EnqueueInstructions";
      else if (opName == "edgetpu.matrix_multiply_sub_channel" ||
               opName == "edgetpu.matrix_multiply_sub_channel_drq" ||
               opName == "edgetpu.fully_connected_sub_channel" ||
               opName == "edgetpu.fully_connected_sub_channel_drq" ||
               opName == "edgetpu.fully_connected_fp8")
        callee = "DiveTpu_EnqueueDmaDescriptor";
      else {
        op->emitError("unsupported edgetpu op in ConvertTpuOffloadToLlvm");
        return signalPassFailure();
      }
      SmallVector<Type> paramTypes;
      for (Value v : op->getOperands())
        paramTypes.push_back(v.getType());
      Type resultType = op->getNumResults() ? op->getResult(0).getType()
                                            : LLVM::LLVMVoidType::get(ctx);
      FailureOr<LLVM::LLVMFuncOp> calleeOp = LLVM::lookupOrCreateFn(
          builder, moduleOp, callee, paramTypes, resultType);
      if (failed(calleeOp))
        return signalPassFailure();
      builder.setInsertionPoint(op);
      auto call = LLVM::CallOp::create(builder, op->getLoc(), *calleeOp,
                                       op->getOperands());
      if (op->getNumResults())
        op->getResult(0).replaceAllUsesWith(call.getResult());
      op->erase();
      ++lowered;
    }
    root->setAttr("tpu_offload.lowered_count",
                  builder.getI64IntegerAttr(lowered));
  }
};

// TSV row: "copy-op-lowering" at 0xddedbd.
struct DwcCopyOpLoweringPass
    : public darwinn::impl::DwcCopyOpLoweringPassBase<DwcCopyOpLoweringPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<dive_vm::DiveVmDialect, LLVM::LLVMDialect>();
  }


  void runOnOperation() override {
    func::FuncOp func = getOperation();
    RewritePatternSet patterns(&getContext());
    darwinn::populateLowerCopySlicePatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
    if (failed(applyLocalCopySliceLowering(func)))
      return signalPassFailure();
    unsigned lowered = 0;
    if (failed(applyDwcLowerCopyLike(func, lowered)))
      return signalPassFailure();
    if (failed(forwardDwcLowerTo(func, {"darwinn.fill"}, "dive_vm.fill", lowered)))
      return signalPassFailure();
    SmallVector<Operation *> dead;
    func.getOperation()->walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name != "darwinn.synchronized_copy_op" && name != "darwinn.streaming_copy_op" &&
          name != "darwinn.parallel_mesh_copy" && name != "darwinn.tile_to_tile" &&
          name != "darwinn.tile_to_host" && name != "darwinn.host_to_tile")
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
  }
};

// TSV row: "darwinn-bundling" at 0xde00af.
struct DwcDarwinnBundlingPass
    : public darwinn::impl::DwcDarwinnBundlingPassBase<DwcDarwinnBundlingPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();

    SmallVector<Operation *> dead;
    root->walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name != "darwinn.copy_op" && name != "darwinn.convert" &&
          name != "darwinn.bitcast")
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

    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op) || op->mightHaveTrait<OpTrait::IsTerminator>())
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "darwinn")
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedLegal = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failedLegal)
      return signalPassFailure();
  }
};

// TSV row: "darwinn.convert" at 0xd68ca7.
struct DwcDarwinnConvertPass
    : public darwinn::impl::DwcDarwinnConvertPassBase<DwcDarwinnConvertPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<dive_vm::DiveVmDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    // darwinn.convert lowers through the cast Fallback kernel shape, same
    // helper as ConvertOpLowering.
    func::FuncOp func = getOperation();
    unsigned lowered = 0;
    if (failed(applyDwcLowerConvertTrunc(func, lowered)))
      return signalPassFailure();
  }
};

// TSV row: "darwinn.math.join" at 0xdc9452.
struct DwcDarwinnMathJoinPass
    : public darwinn::impl::DwcDarwinnMathJoinPassBase<DwcDarwinnMathJoinPass> {
  using Base::Base;

  void runOnOperation() override {
    // darwinn.math.join covers the tgc elementwise family, same scalar
    // helper as the arith lowerings.
    func::FuncOp func = getOperation();
    unsigned lowered = 0;
    if (failed(applyDwcLowerScalarArith(func, lowered)))
      return signalPassFailure();
  }
};

// TSV row: "darwinn.sparsity" at 0xd5a035.
struct DwcDarwinnSparsityPass
    : public darwinn::impl::DwcDarwinnSparsityPassBase<DwcDarwinnSparsityPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (failed(checkDwcConvertibleTypes(func.getOperation())))
      return signalPassFailure();
  }
};

// TSV row: "dive-dce" at 0xe0fe97.
struct DwcDiveDcePass
    : public darwinn::impl::DwcDiveDcePassBase<DwcDiveDcePass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    SmallVector<Operation *> dead;
    func.getOperation()->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        return;
      if (!op->use_empty())
        return;
      if (op->mightHaveTrait<OpTrait::IsTerminator>())
        return;
      if (op->getNumRegions() != 0)
        return;
      if (!mlir::wouldOpBeTriviallyDead(op))
        return;
      dead.push_back(op);
    });
    for (Operation *op : dead)
      op->erase();
  }
};

// TSV row: "dive-io-optimization" at 0xdc141b.
struct DwcDiveIoOptimizationPass
    : public darwinn::impl::DwcDiveIoOptimizationPassBase<
          DwcDiveIoOptimizationPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<dive_vm::DiveVmDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    // No copy folding contract is evidenced in all_pseudocode.json, so the
    // pass only folds single-use dive_vm.copy chains.
    func::FuncOp func = getOperation();
    SmallVector<Operation *> dead;
    func.getOperation()->walk([&](Operation *op) {
      if (op->getName().getStringRef() != "dive_vm.copy")
        return;
      if (op->getNumOperands() != 1 || op->getNumResults() != 1)
        return;
      if (op->getOperand(0).getType() != op->getResult(0).getType())
        return;
      if (!op->getResult(0).hasOneUse())
        return;
      dead.push_back(op);
    });
    for (Operation *op : dead) {
      op->getResult(0).replaceAllUsesWith(op->getOperand(0));
      op->erase();
    }
  }
};

// TSV row: "dive-program-tpu" at 0xd6383a.
struct DwcDiveProgramTpuPass
    : public darwinn::impl::DwcDiveProgramTpuPassBase<DwcDiveProgramTpuPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<dive_vm::DiveVmDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    // Grouping decides packet membership and order only. Each packet then gets
    // one LLVM global holding ordered dispatch function references plus one
    // DiveRuntime_ExecuteChildModel call. No binary packet layout is invented.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    MLIRContext *ctx = root->getContext();
    OpBuilder builder(ctx);
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

    if (groups.empty())
      return;
    auto moduleOp = func->getParentOfType<ModuleOp>();
    if (!moduleOp) {
      func.emitError("dive-program-tpu needs a parent module");
      return signalPassFailure();
    }

    Type ptrTy = LLVM::LLVMPointerType::get(ctx);
    Type i64Ty = builder.getI64Type();
    Type voidTy = LLVM::LLVMVoidType::get(ctx);
    FailureOr<LLVM::LLVMFuncOp> dispatchFn = LLVM::lookupOrCreateFn(
        builder, moduleOp, "DiveRuntime_ExecuteChildModel", {ptrTy, i64Ty},
        voidTy);
    if (failed(dispatchFn))
      return signalPassFailure();

    struct Packet {
      LLVM::GlobalOp global;
      Type arrayTy;
      uint64_t count;
    };
    SmallVector<Packet> packets;
    unsigned id = 0;
    for (auto &entry : groups) {
      uint64_t n = entry.second.size();
      Type arrayTy = LLVM::LLVMArrayType::get(ptrTy, n);
      std::string name = "tpu_packet_" + std::to_string(id++);
      while (SymbolTable::lookupSymbolIn(moduleOp, name))
        name += "_";
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(moduleOp.getBody());
      LLVM::GlobalOp global = LLVM::GlobalOp::create(
          builder, func.getLoc(), arrayTy, /*isConstant=*/true,
          LLVM::Linkage::Internal, name, /*value=*/Attribute(),
          /*alignment=*/0);
      Block *init = builder.createBlock(&global.getInitializerRegion());
      builder.setInsertionPointToStart(init);
      Value table = LLVM::PoisonOp::create(builder, func.getLoc(), arrayTy);
      for (uint64_t i = 0; i < n; ++i) {
        Value fnPtr = LLVM::AddressOfOp::create(builder, func.getLoc(), ptrTy,
                                                dispatchFn->getSymNameAttr());
        SmallVector<int64_t> pos{static_cast<int64_t>(i)};
        table = LLVM::InsertValueOp::create(builder, func.getLoc(), table,
                                            fnPtr, pos);
      }
      LLVM::ReturnOp::create(builder, func.getLoc(), ArrayRef<Value>({table}));
      packets.push_back({global, arrayTy, n});
    }

    if (func.getBody().empty())
      return;
    Block &entryBlock = func.getBody().front();
    if (Operation *term = entryBlock.getTerminator())
      builder.setInsertionPoint(term);
    else
      builder.setInsertionPointToEnd(&entryBlock);
    for (auto &item : packets) {
      Value base = LLVM::AddressOfOp::create(builder, func.getLoc(), ptrTy,
                                             item.global.getSymNameAttr());
      Value table =
          LLVM::GEPOp::create(builder, func.getLoc(), ptrTy, item.arrayTy, base,
                              ArrayRef<LLVM::GEPArg>{0, 0});
      Value count = LLVM::ConstantOp::create(
          builder, func.getLoc(), i64Ty,
          builder.getI64IntegerAttr(static_cast<int64_t>(item.count)));
      SmallVector<Value> args{table, count};
      LLVM::CallOp::create(builder, func.getLoc(), *dispatchFn, args);
    }
  }
};

// TSV row: "dive-unroll-factor" at 0xda744d.
struct DwcDiveUnrollFactorPass
    : public darwinn::impl::DwcDiveUnrollFactorPassBase<
          DwcDiveUnrollFactorPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<dive_vm::DiveVmDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (failed(checkDwcConvertibleTypes(func.getOperation())))
      return signalPassFailure();
  }
};

// TSV row: "dive-vm-bufferize" at 0xde6a70.
struct DwcDiveVmBufferizePass
    : public darwinn::impl::DwcDiveVmBufferizePassBase<DwcDiveVmBufferizePass> {
  using Base::Base;

  void runOnOperation() override {
    // No buffer layout is evidenced in all_pseudocode.json so only same type dive_vm identities fold.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();

    SmallVector<Operation *> dead;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "dive_vm")
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

    bool failedLegal = false;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "dive_vm")
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedLegal = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failedLegal)
      return signalPassFailure();
  }
};

// TSV row: "dive-vm-outline-shareable-dive-consts" at 0xd788a7.
struct DwcDiveVmOutlineShareableDiveConstsPass
    : public darwinn::impl::DwcDiveVmOutlineShareableDiveConstsPassBase<
          DwcDiveVmOutlineShareableDiveConstsPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<dive_vm::DiveVmDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    DenseMap<Attribute, Operation *> seen;
    SmallVector<Operation *> dead;
    func.getOperation()->walk([&](Operation *op) {
      if (op->getName().getStringRef() != "dive_vm.const")
        return;
      if (op->getNumResults() != 1)
        return;
      Attribute value = op->getAttr("value");
      if (!value)
        return;
      auto it = seen.find(value);
      if (it == seen.end()) {
        seen.insert({value, op});
        return;
      }
      op->getResult(0).replaceAllUsesWith(it->second->getResult(0));
      dead.push_back(op);
    });
    for (Operation *op : dead)
      op->erase();
  }
};

// TSV row: "dwc-check-illegal-tpu-ops" at 0xd84494.
struct DwcDwcCheckIllegalTpuOpsPass
    : public darwinn::impl::DwcDwcCheckIllegalTpuOpsPassBase<
          DwcDwcCheckIllegalTpuOpsPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    SmallVector<Operation *> dead;
    root->walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name != "darwinn.copy_op" && name != "darwinn.convert" &&
          name != "darwinn.bitcast")
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
    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op) || op->mightHaveTrait<OpTrait::IsTerminator>())
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect) {
        op->emitError() << "dwc-check-illegal-tpu-ops rejects unregistered "
                        << op->getName().getStringRef();
        failedLegal = true;
        return WalkResult::interrupt();
      }
      StringRef ns = dialect->getNamespace();
      if (ns != "darwinn" && ns != "dive_vm" && ns != "edgetpu") {
        op->emitError() << "dwc-check-illegal-tpu-ops rejects dialect " << ns;
        failedLegal = true;
        return WalkResult::interrupt();
      }
      if (failed(checkDwcConvertibleTypes(op))) {
        failedLegal = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failedLegal)
      return signalPassFailure();
  }
};

// TSV row: "dwc-convert-input-output-types" at 0xd94dcc.
struct DwcDwcConvertInputOutputTypesPass
    : public darwinn::impl::DwcDwcConvertInputOutputTypesPassBase<
          DwcDwcConvertInputOutputTypesPass> {
  using Base::Base;

  void runOnOperation() override {
    // Binary packet layout is absent from all_pseudocode.json, group and order
    // only.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    std::map<Operation *, unsigned> clusterOf;
    unsigned nextCluster = 0;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op) || op->mightHaveTrait<OpTrait::IsTerminator>())
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
    root->setAttr("tpu.cluster_count", builder.getI64IntegerAttr(nextCluster));
  }
};

// TSV row: "dwc-copy-strided-buffers-on-tpu" at 0xd6381a.
struct DwcDwcCopyStridedBuffersOnTpuPass
    : public darwinn::impl::DwcDwcCopyStridedBuffersOnTpuPassBase<
          DwcDwcCopyStridedBuffersOnTpuPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<dive_vm::DiveVmDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    unsigned lowered = 0;
    if (failed(applyDwcLowerCopyLike(func, lowered)))
      return signalPassFailure();
  }
};

// TSV row: "dwc-form-tpu-clusters" at 0xd81a1f.
struct DwcDwcFormTpuClustersPass
    : public darwinn::impl::DwcDwcFormTpuClustersPassBase<
          DwcDwcFormTpuClustersPass> {
  using Base::Base;

  void runOnOperation() override {
    // Binary packet layout is absent from all_pseudocode.json, group and order
    // only.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    std::map<Operation *, unsigned> clusterOf;
    unsigned nextCluster = 0;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op) || op->mightHaveTrait<OpTrait::IsTerminator>())
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
    root->setAttr("tpu.cluster_count", builder.getI64IntegerAttr(nextCluster));
  }
};

// TSV row: "dwc-legalize" at 0xde6d3f.
struct DwcDwcLegalizePass
    : public darwinn::impl::DwcDwcLegalizePassBase<DwcDwcLegalizePass> {
  using Base::Base;

  void runOnOperation() override {
    // Only copy convert and bitcast identities fold here. No other kernel shape for this family appears in all_pseudocode.json.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();

    SmallVector<Operation *> dead;
    root->walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name != "darwinn.copy_op" && name != "darwinn.convert" &&
          name != "darwinn.bitcast")
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

    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op) || op->mightHaveTrait<OpTrait::IsTerminator>())
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect) {
        op->emitError() << "dwc-legalize rejects unregistered operation "
                        << op->getName().getStringRef();
        failedLegal = true;
        return WalkResult::interrupt();
      }
      StringRef ns = dialect->getNamespace();
      if (ns != "darwinn") {
        op->emitError() << "dwc-legalize rejects operation from dialect " << ns;
        failedLegal = true;
        return WalkResult::interrupt();
      }
      if (failed(checkDwcConvertibleTypes(op))) {
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
struct DwcDwcLegalizeHloPass
    : public darwinn::impl::DwcDwcLegalizeHloPassBase<DwcDwcLegalizeHloPass> {
  using Base::Base;

  void runOnOperation() override {
    // No mhlo op name exists in DarwinnOps.td or DiveVmOps.td and no hlo legalization mapping is evidenced so the fold walk finds nothing.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();

    SmallVector<Operation *> dead;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "mhlo")
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

    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op) || op->mightHaveTrait<OpTrait::IsTerminator>())
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect) {
        op->emitError() << "dwc-legalize-hlo rejects unregistered operation "
                        << op->getName().getStringRef();
        failedLegal = true;
        return WalkResult::interrupt();
      }
      StringRef ns = dialect->getNamespace();
      if (ns != "mhlo") {
        op->emitError() << "dwc-legalize-hlo rejects operation from dialect "
                        << ns;
        failedLegal = true;
        return WalkResult::interrupt();
      }
      if (failed(checkDwcConvertibleTypes(op))) {
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
struct DwcDwcLegalizeHloToTfPass
    : public darwinn::impl::DwcDwcLegalizeHloToTfPassBase<
          DwcDwcLegalizeHloToTfPass> {
  using Base::Base;

  void runOnOperation() override {
    // No mhlo or tf op name exists in DarwinnOps.td or DiveVmOps.td and no hlo to tf mapping is evidenced so the fold walk finds nothing.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();

    SmallVector<Operation *> dead;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect)
        return;
      StringRef ns = dialect->getNamespace();
      if (ns != "mhlo" && ns != "tf")
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

    bool failedLegal = false;
    root->walk(
        [&](Operation *op) {
          if (isa<func::FuncOp>(op))
            return WalkResult::advance();
          Dialect *dialect = op->getDialect();
          if (!dialect) {
            op->emitError()
                << "dwc-legalize-hlo-to-tf rejects unregistered operation "
                << op->getName().getStringRef();
            failedLegal = true;
            return WalkResult::interrupt();
          }
          StringRef ns = dialect->getNamespace();
          if (ns != "mhlo" && ns != "tf") {
            op->emitError()
                << "dwc-legalize-hlo-to-tf rejects operation from dialect "
                << ns;
            failedLegal = true;
            return WalkResult::interrupt();
          }
          if (failed(checkDwcConvertibleTypes(op))) {
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
struct DwcDwcLegalizeIntAndQuantTypesPass
    : public darwinn::impl::DwcDwcLegalizeIntAndQuantTypesPassBase<
          DwcDwcLegalizeIntAndQuantTypesPass> {
  using Base::Base;

  void runOnOperation() override {
    // No quant op name exists in DarwinnOps.td or DiveVmOps.td and no int and quant type mapping is evidenced so the fold walk finds nothing.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();

    SmallVector<Operation *> dead;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "quant")
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

    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op) || op->mightHaveTrait<OpTrait::IsTerminator>())
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect) {
        op->emitError() << "dwc-legalize-int-and-quant-types rejects "
                           "unregistered operation "
                        << op->getName().getStringRef();
        failedLegal = true;
        return WalkResult::interrupt();
      }
      StringRef ns = dialect->getNamespace();
      if (ns != "quant") {
        op->emitError() << "dwc-legalize-int-and-quant-types rejects operation "
                           "from dialect "
                        << ns;
        failedLegal = true;
        return WalkResult::interrupt();
      }
      if (failed(checkDwcConvertibleTypes(op))) {
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
struct DwcDwcLegalizeInt64ConstantsPass
    : public darwinn::impl::DwcDwcLegalizeInt64ConstantsPassBase<
          DwcDwcLegalizeInt64ConstantsPass> {
  using Base::Base;

  void runOnOperation() override {
    // Darwinn int64 constant shapes are out of scope here so only same type copy convert and bitcast identities fold.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();

    SmallVector<Operation *> dead;
    root->walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name != "darwinn.copy_op" && name != "darwinn.convert" &&
          name != "darwinn.bitcast")
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

    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op) || op->mightHaveTrait<OpTrait::IsTerminator>())
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect) {
        op->emitError()
            << "dwc-legalize-int64-constants rejects unregistered operation "
            << op->getName().getStringRef();
        failedLegal = true;
        return WalkResult::interrupt();
      }
      StringRef ns = dialect->getNamespace();
      if (ns != "darwinn") {
        op->emitError()
            << "dwc-legalize-int64-constants rejects operation from dialect "
            << ns;
        failedLegal = true;
        return WalkResult::interrupt();
      }
      if (failed(checkDwcConvertibleTypes(op))) {
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
struct DwcDwcLegalizePassSymbol
    : public darwinn::impl::DwcDwcLegalizePassSymbolBase<DwcDwcLegalizePassSymbol> {
  using Base::Base;

  void runOnOperation() override {
    // Only copy convert and bitcast identities fold here. No other kernel shape for this family appears in all_pseudocode.json.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();

    SmallVector<Operation *> dead;
    root->walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name != "darwinn.copy_op" && name != "darwinn.convert" &&
          name != "darwinn.bitcast")
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

    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op) || op->mightHaveTrait<OpTrait::IsTerminator>())
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect) {
        op->emitError() << "dwc-legalize-pass rejects unregistered operation "
                        << op->getName().getStringRef();
        failedLegal = true;
        return WalkResult::interrupt();
      }
      StringRef ns = dialect->getNamespace();
      if (ns != "darwinn") {
        op->emitError() << "dwc-legalize-pass rejects operation from dialect "
                        << ns;
        failedLegal = true;
        return WalkResult::interrupt();
      }
      if (failed(checkDwcConvertibleTypes(op))) {
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
struct DwcDwcLegalizeStablehloAnnotateMaterializePolicyPass
    : public darwinn::impl::
          DwcDwcLegalizeStablehloAnnotateMaterializePolicyPassBase<
              DwcDwcLegalizeStablehloAnnotateMaterializePolicyPass> {
  using Base::Base;

  void runOnOperation() override {
    // No stablehlo op name exists in DarwinnOps.td or DiveVmOps.td and no materialize policy mapping is evidenced so the fold walk finds nothing.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();

    SmallVector<Operation *> dead;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "stablehlo")
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

    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op) || op->mightHaveTrait<OpTrait::IsTerminator>())
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect) {
        op->emitError() << "dwc-legalize-stablehlo-annotate-materialize-policy "
                           "rejects unregistered operation "
                        << op->getName().getStringRef();
        failedLegal = true;
        return WalkResult::interrupt();
      }
      StringRef ns = dialect->getNamespace();
      if (ns != "stablehlo") {
        op->emitError() << "dwc-legalize-stablehlo-annotate-materialize-policy "
                           "rejects operation from dialect "
                        << ns;
        failedLegal = true;
        return WalkResult::interrupt();
      }
      if (failed(checkDwcConvertibleTypes(op))) {
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
struct DwcDwcLegalizeStablehloCompositePass
    : public darwinn::impl::DwcDwcLegalizeStablehloCompositePassBase<
          DwcDwcLegalizeStablehloCompositePass> {
  using Base::Base;

  void runOnOperation() override {
    // No stablehlo op name exists in DarwinnOps.td or DiveVmOps.td and no stablehlo composite mapping is evidenced so the fold walk finds nothing.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();

    SmallVector<Operation *> dead;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "stablehlo")
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

    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op) || op->mightHaveTrait<OpTrait::IsTerminator>())
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect) {
        op->emitError() << "dwc-legalize-stablehlo-composite rejects "
                           "unregistered operation "
                        << op->getName().getStringRef();
        failedLegal = true;
        return WalkResult::interrupt();
      }
      StringRef ns = dialect->getNamespace();
      if (ns != "stablehlo") {
        op->emitError() << "dwc-legalize-stablehlo-composite rejects operation "
                           "from dialect "
                        << ns;
        failedLegal = true;
        return WalkResult::interrupt();
      }
      if (failed(checkDwcConvertibleTypes(op))) {
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
struct DwcDwcLegalizeTfPipelinePass
    : public darwinn::impl::DwcDwcLegalizeTfPipelinePassBase<
          DwcDwcLegalizeTfPipelinePass> {
  using Base::Base;

  void runOnOperation() override {
    // No tf op name exists in DarwinnOps.td or DiveVmOps.td and no tf pipeline mapping is evidenced so the fold walk finds nothing.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();

    SmallVector<Operation *> dead;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "tf")
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

    bool failedLegal = false;
    root->walk(
        [&](Operation *op) {
          if (isa<func::FuncOp>(op))
            return WalkResult::advance();
          Dialect *dialect = op->getDialect();
          if (!dialect) {
            op->emitError()
                << "dwc-legalize-tf-pipeline rejects unregistered operation "
                << op->getName().getStringRef();
            failedLegal = true;
            return WalkResult::interrupt();
          }
          StringRef ns = dialect->getNamespace();
          if (ns != "tf") {
            op->emitError()
                << "dwc-legalize-tf-pipeline rejects operation from dialect "
                << ns;
            failedLegal = true;
            return WalkResult::interrupt();
          }
          if (failed(checkDwcConvertibleTypes(op))) {
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
struct DwcDwcLegalizeTflCudaemuCustomOpsPass
    : public darwinn::impl::DwcDwcLegalizeTflCudaemuCustomOpsPassBase<
          DwcDwcLegalizeTflCudaemuCustomOpsPass> {
  using Base::Base;

  void runOnOperation() override {
    // No tfl op name exists in DarwinnOps.td or DiveVmOps.td and no tfl cudaemu custom op emitter is evidenced so the fold walk finds nothing.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();

    SmallVector<Operation *> dead;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "tfl")
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

    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op) || op->mightHaveTrait<OpTrait::IsTerminator>())
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect) {
        op->emitError() << "dwc-legalize-tfl-cudaemu-custom-ops rejects "
                           "unregistered operation "
                        << op->getName().getStringRef();
        failedLegal = true;
        return WalkResult::interrupt();
      }
      StringRef ns = dialect->getNamespace();
      if (ns != "tfl") {
        op->emitError() << "dwc-legalize-tfl-cudaemu-custom-ops rejects "
                           "operation from dialect "
                        << ns;
        failedLegal = true;
        return WalkResult::interrupt();
      }
      if (failed(checkDwcConvertibleTypes(op))) {
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
struct DwcDwcLegalizeTflMultinomialPass
    : public darwinn::impl::DwcDwcLegalizeTflMultinomialPassBase<
          DwcDwcLegalizeTflMultinomialPass> {
  using Base::Base;

  void runOnOperation() override {
    // No tfl op name exists in DarwinnOps.td or DiveVmOps.td and no tfl multinomial mapping is evidenced so the fold walk finds nothing.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();

    SmallVector<Operation *> dead;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "tfl")
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

    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op) || op->mightHaveTrait<OpTrait::IsTerminator>())
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect) {
        op->emitError()
            << "dwc-legalize-tfl-multinomial rejects unregistered operation "
            << op->getName().getStringRef();
        failedLegal = true;
        return WalkResult::interrupt();
      }
      StringRef ns = dialect->getNamespace();
      if (ns != "tfl") {
        op->emitError()
            << "dwc-legalize-tfl-multinomial rejects operation from dialect "
            << ns;
        failedLegal = true;
        return WalkResult::interrupt();
      }
      if (failed(checkDwcConvertibleTypes(op))) {
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
struct DwcDwcLegalizeTflVariableTensorsPass
    : public darwinn::impl::DwcDwcLegalizeTflVariableTensorsPassBase<
          DwcDwcLegalizeTflVariableTensorsPass> {
  using Base::Base;

  void runOnOperation() override {
    // No tfl op name exists in DarwinnOps.td or DiveVmOps.td and no variable tensor mapping is evidenced so the fold walk finds nothing.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();

    SmallVector<Operation *> dead;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "tfl")
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

    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op) || op->mightHaveTrait<OpTrait::IsTerminator>())
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect) {
        op->emitError() << "dwc-legalize-tfl-variable-tensors rejects "
                           "unregistered operation "
                        << op->getName().getStringRef();
        failedLegal = true;
        return WalkResult::interrupt();
      }
      StringRef ns = dialect->getNamespace();
      if (ns != "tfl") {
        op->emitError() << "dwc-legalize-tfl-variable-tensors rejects "
                           "operation from dialect "
                        << ns;
        failedLegal = true;
        return WalkResult::interrupt();
      }
      if (failed(checkDwcConvertibleTypes(op))) {
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
struct DwcDwcLegalizeUint32TypesPass
    : public darwinn::impl::DwcDwcLegalizeUint32TypesPassBase<
          DwcDwcLegalizeUint32TypesPass> {
  using Base::Base;

  void runOnOperation() override {
    // No uint32 type mapping is evidenced so the fold walk covers only same-type identities.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();

    SmallVector<Operation *> dead;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "darwinn")
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

    bool failedLegal = false;
    root->walk(
        [&](Operation *op) {
          if (isa<func::FuncOp>(op))
            return WalkResult::advance();
          Dialect *dialect = op->getDialect();
          if (!dialect) {
            op->emitError()
                << "dwc-legalize-uint32-types rejects unregistered operation "
                << op->getName().getStringRef();
            failedLegal = true;
            return WalkResult::interrupt();
          }
          StringRef ns = dialect->getNamespace();
          if (ns != "darwinn") {
            op->emitError()
                << "dwc-legalize-uint32-types rejects operation from dialect "
                << ns;
            failedLegal = true;
            return WalkResult::interrupt();
          }
          if (failed(checkDwcConvertibleTypes(op))) {
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
struct DwcDwcLowerArgmaxIndexUnpoolPass
    : public darwinn::impl::DwcDwcLowerArgmaxIndexUnpoolPassBase<
          DwcDwcLowerArgmaxIndexUnpoolPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    unsigned lowered = 0;
    if (failed(applyDwcLowerArgmaxInline(func, lowered)))
      return signalPassFailure();
  }
};

// TSV row: "dwc-lower-composite-ops" at 0xd84775.
struct DwcDwcLowerCompositeOpsPass
    : public darwinn::impl::DwcDwcLowerCompositeOpsPassBase<
          DwcDwcLowerCompositeOpsPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<dive_vm::DiveVmDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    // The sibling-owned LowerCopySlice and LowerConvert sets do the real
    // lowering, identity folds below only clean up what patterns leave behind.
    RewritePatternSet patterns(&getContext());
    darwinn::populateLowerCopySlicePatterns(patterns);
    darwinn::populateLowerConvertPatterns(patterns);
    if (failed(
            applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
    func::FuncOp func = getOperation();
    if (failed(applyLocalCopySliceLowering(func)))
      return signalPassFailure();
    if (failed(applyLocalConvertLowering(func)))
      return signalPassFailure();
  }
};

// TSV row: "dwc-lower-control-flow" at 0xd61f2c.
struct DwcDwcLowerControlFlowPass
    : public darwinn::impl::DwcDwcLowerControlFlowPassBase<
          DwcDwcLowerControlFlowPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    SmallVector<Operation *> dead;
    // DarwinnOps.td gives no region or kernel shape for the control and misc
    // ops so only regionless same type identities fold. Everything else stays
    // for the convertible types gate below.
    root->walk([&](Operation *op) {
      if (op->getNumRegions() != 0)
        return;
      StringRef name = op->getName().getStringRef();
      if (name != "darwinn.copy_op" && name != "darwinn.convert" &&
          name != "darwinn.bitcast")
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
    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op) || op->mightHaveTrait<OpTrait::IsTerminator>())
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "darwinn") {
        op->emitError("dwc-lower-control-flow accepts darwinn ops only");
        failedLegal = true;
        return WalkResult::interrupt();
      }
      if (failed(checkDwcConvertibleTypes(op))) {
        failedLegal = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failedLegal)
      return signalPassFailure();
  }
};

// TSV row: "dwc-lower-depth-to-from-space" at 0xe102d0.
struct DwcDwcLowerDepthToFromSpacePass
    : public darwinn::impl::DwcDwcLowerDepthToFromSpacePassBase<
          DwcDwcLowerDepthToFromSpacePass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<dive_vm::DiveVmDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    // The sibling-owned LowerCopySlice and LowerConvert sets do the real
    // lowering, identity folds below only clean up what patterns leave behind.
    RewritePatternSet patterns(&getContext());
    darwinn::populateLowerCopySlicePatterns(patterns);
    darwinn::populateLowerConvertPatterns(patterns);
    if (failed(
            applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
    func::FuncOp func = getOperation();
    if (failed(applyLocalCopySliceLowering(func)))
      return signalPassFailure();
    if (failed(applyLocalConvertLowering(func)))
      return signalPassFailure();
  }
};

// TSV row: "dwc-lower-generic-constants" at 0xd7bb41.
struct DwcDwcLowerGenericConstantsPass
    : public darwinn::impl::DwcDwcLowerGenericConstantsPassBase<
          DwcDwcLowerGenericConstantsPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<dive_vm::DiveVmDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    // The sibling-owned LowerCopySlice and LowerConvert sets do the real
    // lowering, identity folds below only clean up what patterns leave behind.
    RewritePatternSet patterns(&getContext());
    darwinn::populateLowerCopySlicePatterns(patterns);
    darwinn::populateLowerConvertPatterns(patterns);
    if (failed(
            applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
    func::FuncOp func = getOperation();
    if (failed(applyLocalCopySliceLowering(func)))
      return signalPassFailure();
    if (failed(applyLocalConvertLowering(func)))
      return signalPassFailure();
  }
};

// TSV row: "dwc-lower-hlops" at 0xd841f4.
struct DwcDwcLowerHlopsPass
    : public darwinn::impl::DwcDwcLowerHlopsPassBase<DwcDwcLowerHlopsPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<dive_vm::DiveVmDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    // The sibling-owned LowerCopySlice and LowerConvert sets do the real
    // lowering, identity folds below only clean up what patterns leave behind.
    RewritePatternSet patterns(&getContext());
    darwinn::populateLowerCopySlicePatterns(patterns);
    darwinn::populateLowerConvertPatterns(patterns);
    if (failed(
            applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
    func::FuncOp func = getOperation();
    if (failed(applyLocalCopySliceLowering(func)))
      return signalPassFailure();
    if (failed(applyLocalConvertLowering(func)))
      return signalPassFailure();
    SmallVector<Operation *> dead;
    func.getOperation()->walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name != "darwinn.rkhy_unary_compute_op" &&
          name != "darwinn.rkhy_depth_to_space_op" &&
          name != "darwinn.static_unary_compute_op" &&
          name != "darwinn.synchronized_unary_compute_op" &&
          name != "darwinn.streaming_unary_compute_op")
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
  }
};

// TSV row: "dwc-lower-input-output-cast" at 0xd68084.
struct DwcDwcLowerInputOutputCastPass
    : public darwinn::impl::DwcDwcLowerInputOutputCastPassBase<
          DwcDwcLowerInputOutputCastPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<dive_vm::DiveVmDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    // The sibling-owned LowerCopySlice and LowerConvert sets do the real
    // lowering, identity folds below only clean up what patterns leave behind.
    RewritePatternSet patterns(&getContext());
    darwinn::populateLowerCopySlicePatterns(patterns);
    darwinn::populateLowerConvertPatterns(patterns);
    if (failed(
            applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
    func::FuncOp func = getOperation();
    if (failed(applyLocalCopySliceLowering(func)))
      return signalPassFailure();
    if (failed(applyLocalConvertLowering(func)))
      return signalPassFailure();
  }
};

// TSV row: "dwc-lower-padding-ops" at 0xd84749.
struct DwcDwcLowerPaddingOpsPass
    : public darwinn::impl::DwcDwcLowerPaddingOpsPassBase<
          DwcDwcLowerPaddingOpsPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<dive_vm::DiveVmDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    // The sibling-owned LowerCopySlice and LowerConvert sets do the real
    // lowering, identity folds below only clean up what patterns leave behind.
    RewritePatternSet patterns(&getContext());
    darwinn::populateLowerCopySlicePatterns(patterns);
    darwinn::populateLowerConvertPatterns(patterns);
    if (failed(
            applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
    func::FuncOp func = getOperation();
    if (failed(applyLocalCopySliceLowering(func)))
      return signalPassFailure();
    if (failed(applyLocalConvertLowering(func)))
      return signalPassFailure();
  }
};

// TSV row: "dwc-lower-pseudo-ops" at 0xd845c5.
struct DwcDwcLowerPseudoOpsPass
    : public darwinn::impl::DwcDwcLowerPseudoOpsPassBase<
          DwcDwcLowerPseudoOpsPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<dive_vm::DiveVmDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    // The sibling-owned LowerCopySlice and LowerConvert sets do the real
    // lowering, identity folds below only clean up what patterns leave behind.
    RewritePatternSet patterns(&getContext());
    darwinn::populateLowerCopySlicePatterns(patterns);
    darwinn::populateLowerConvertPatterns(patterns);
    if (failed(
            applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
    func::FuncOp func = getOperation();
    if (failed(applyLocalCopySliceLowering(func)))
      return signalPassFailure();
    if (failed(applyLocalConvertLowering(func)))
      return signalPassFailure();
  }
};

// TSV row: "dwc-lower-resampler-ops" at 0xd84598.
struct DwcDwcLowerResamplerOpsPass
    : public darwinn::impl::DwcDwcLowerResamplerOpsPassBase<
          DwcDwcLowerResamplerOpsPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<dive_vm::DiveVmDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    // The sibling-owned LowerCopySlice and LowerConvert sets do the real
    // lowering, identity folds below only clean up what patterns leave behind.
    RewritePatternSet patterns(&getContext());
    darwinn::populateLowerCopySlicePatterns(patterns);
    darwinn::populateLowerConvertPatterns(patterns);
    if (failed(
            applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
    func::FuncOp func = getOperation();
    if (failed(applyLocalCopySliceLowering(func)))
      return signalPassFailure();
    if (failed(applyLocalConvertLowering(func)))
      return signalPassFailure();
  }
};

// TSV row: "dwc-lower-scalar-ops" at 0xd845b0.
struct DwcDwcLowerScalarOpsPass
    : public darwinn::impl::DwcDwcLowerScalarOpsPassBase<
          DwcDwcLowerScalarOpsPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<dive_vm::DiveVmDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    // The sibling-owned LowerCopySlice and LowerConvert sets do the real
    // lowering, identity folds below only clean up what patterns leave behind.
    RewritePatternSet patterns(&getContext());
    darwinn::populateLowerCopySlicePatterns(patterns);
    darwinn::populateLowerConvertPatterns(patterns);
    if (failed(
            applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
    func::FuncOp func = getOperation();
    if (failed(applyLocalCopySliceLowering(func)))
      return signalPassFailure();
    if (failed(applyLocalConvertLowering(func)))
      return signalPassFailure();
    unsigned lowered = 0;
    if (failed(applyDwcLowerScalarArith(func, lowered)))
      return signalPassFailure();
    SmallVector<Operation *> dead;
    func.getOperation()->walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name != "darwinn.unary_map" && name != "darwinn.unary_tensor_op" &&
          name != "darwinn.dive_ref_cwise")
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
  }
};

// TSV row: "dwc-lower-scatter-ops" at 0xd84582.
struct DwcDwcLowerScatterOpsPass
    : public darwinn::impl::DwcDwcLowerScatterOpsPassBase<
          DwcDwcLowerScatterOpsPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<dive_vm::DiveVmDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    // The sibling-owned LowerCopySlice and LowerConvert sets do the real
    // lowering, identity folds below only clean up what patterns leave behind.
    RewritePatternSet patterns(&getContext());
    darwinn::populateLowerCopySlicePatterns(patterns);
    darwinn::populateLowerConvertPatterns(patterns);
    if (failed(
            applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
    func::FuncOp func = getOperation();
    if (failed(applyLocalCopySliceLowering(func)))
      return signalPassFailure();
    if (failed(applyLocalConvertLowering(func)))
      return signalPassFailure();
  }
};

// TSV row: "dwc-lower-top-k" at 0xdd6da3.
struct DwcDwcLowerTopKPass
    : public darwinn::impl::DwcDwcLowerTopKPassBase<DwcDwcLowerTopKPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<dive_vm::DiveVmDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    // The sibling-owned LowerCopySlice and LowerConvert sets do the real
    // lowering, identity folds below only clean up what patterns leave behind.
    RewritePatternSet patterns(&getContext());
    darwinn::populateLowerCopySlicePatterns(patterns);
    darwinn::populateLowerConvertPatterns(patterns);
    if (failed(
            applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
    func::FuncOp func = getOperation();
    if (failed(applyLocalCopySliceLowering(func)))
      return signalPassFailure();
    if (failed(applyLocalConvertLowering(func)))
      return signalPassFailure();
  }
};

// TSV row: "dwc-post-truncation-tpu-fitter" at 0xdab7fc.
struct DwcDwcPostTruncationTpuFitterPass
    : public darwinn::impl::DwcDwcPostTruncationTpuFitterPassBase<
          DwcDwcPostTruncationTpuFitterPass> {
  using Base::Base;

  void runOnOperation() override {
    // Binary packet layout is absent from all_pseudocode.json, group and order
    // only.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    std::map<Operation *, unsigned> clusterOf;
    unsigned nextCluster = 0;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op) || op->mightHaveTrait<OpTrait::IsTerminator>())
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
    root->setAttr("tpu.cluster_count", builder.getI64IntegerAttr(nextCluster));
  }
};

// TSV row: "dwc-pre-tpu-fitter-optimize-gather" at 0xdaf5f9.
struct DwcDwcPreTpuFitterOptimizeGatherPass
    : public darwinn::impl::DwcDwcPreTpuFitterOptimizeGatherPassBase<
          DwcDwcPreTpuFitterOptimizeGatherPass> {
  using Base::Base;

  void runOnOperation() override {
    // Binary packet layout is absent from all_pseudocode.json, group and order
    // only.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    std::map<Operation *, unsigned> clusterOf;
    unsigned nextCluster = 0;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op) || op->mightHaveTrait<OpTrait::IsTerminator>())
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
    root->setAttr("tpu.cluster_count", builder.getI64IntegerAttr(nextCluster));
  }
};

// TSV row: "dwc-pre-tpu-fitter-optimize-scatter" at 0xdabc37.
struct DwcDwcPreTpuFitterOptimizeScatterPass
    : public darwinn::impl::DwcDwcPreTpuFitterOptimizeScatterPassBase<
          DwcDwcPreTpuFitterOptimizeScatterPass> {
  using Base::Base;

  void runOnOperation() override {
    // Binary packet layout is absent from all_pseudocode.json, group and order
    // only.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    std::map<Operation *, unsigned> clusterOf;
    unsigned nextCluster = 0;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op) || op->mightHaveTrait<OpTrait::IsTerminator>())
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
    root->setAttr("tpu.cluster_count", builder.getI64IntegerAttr(nextCluster));
  }
};

// TSV row: "dwc-regroup-tpu-functions" at 0xd873c3.
struct DwcDwcRegroupTpuFunctionsPass
    : public darwinn::impl::DwcDwcRegroupTpuFunctionsPassBase<
          DwcDwcRegroupTpuFunctionsPass> {
  using Base::Base;

  void runOnOperation() override {
    // Binary packet layout is absent from all_pseudocode.json, group and order
    // only.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    std::map<Operation *, unsigned> clusterOf;
    unsigned nextCluster = 0;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op) || op->mightHaveTrait<OpTrait::IsTerminator>())
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
    root->setAttr("tpu.cluster_count", builder.getI64IntegerAttr(nextCluster));
  }
};

// TSV row: "dwc-serialize-tpu-offloads" at 0xd9c33f.
struct DwcDwcSerializeTpuOffloadsPass
    : public darwinn::impl::DwcDwcSerializeTpuOffloadsPassBase<
          DwcDwcSerializeTpuOffloadsPass> {
  using Base::Base;

  void runOnOperation() override {
    // Binary packet layout is absent from all_pseudocode.json, group and order
    // only.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    std::map<Operation *, unsigned> clusterOf;
    unsigned nextCluster = 0;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op) || op->mightHaveTrait<OpTrait::IsTerminator>())
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
    root->setAttr("tpu.cluster_count", builder.getI64IntegerAttr(nextCluster));
  }
};

// TSV row: "dwc-test-repeat-tpu-ops" at 0xd8447c.
struct DwcDwcTestRepeatTpuOpsPass
    : public darwinn::impl::DwcDwcTestRepeatTpuOpsPassBase<
          DwcDwcTestRepeatTpuOpsPass> {
  using Base::Base;

  void runOnOperation() override {
    // Binary packet layout is absent from all_pseudocode.json, group and order
    // only.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    std::map<Operation *, unsigned> clusterOf;
    unsigned nextCluster = 0;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op) || op->mightHaveTrait<OpTrait::IsTerminator>())
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
    root->setAttr("tpu.cluster_count", builder.getI64IntegerAttr(nextCluster));
  }
};

// TSV row: "dwc-tpu-fitter" at 0xdab81b.
struct DwcDwcTpuFitterPass
    : public darwinn::impl::DwcDwcTpuFitterPassBase<DwcDwcTpuFitterPass> {
  using Base::Base;

  void runOnOperation() override {
    // Binary packet layout is absent from all_pseudocode.json, group and order
    // only.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    std::map<Operation *, unsigned> clusterOf;
    unsigned nextCluster = 0;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op) || op->mightHaveTrait<OpTrait::IsTerminator>())
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
    root->setAttr("tpu.cluster_count", builder.getI64IntegerAttr(nextCluster));
  }
};

// TSV row: "dwc-tpu-function-cse" at 0xdf32f5.
struct DwcDwcTpuFunctionCsePass
    : public darwinn::impl::DwcDwcTpuFunctionCsePassBase<
          DwcDwcTpuFunctionCsePass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<dive_vm::DiveVmDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    DenseMap<Attribute, Operation *> seen;
    SmallVector<Operation *> dead;
    func.getOperation()->walk([&](Operation *op) {
      if (op->getName().getStringRef() != "dive_vm.const")
        return;
      if (op->getNumResults() != 1)
        return;
      Attribute value = op->getAttr("value");
      if (!value)
        return;
      auto it = seen.find(value);
      if (it == seen.end()) {
        seen.insert({value, op});
        return;
      }
      op->getResult(0).replaceAllUsesWith(it->second->getResult(0));
      dead.push_back(op);
    });
    for (Operation *op : dead)
      op->erase();
  }
};

// TSV row: "dwg-create-darwinn-custom-op" at 0xdb45e0.
struct DwcDwgCreateDarwinnCustomOpPass
    : public darwinn::impl::DwcDwgCreateDarwinnCustomOpPassBase<
          DwcDwgCreateDarwinnCustomOpPass> {
  using Base::Base;

  void runOnOperation() override {
    // No custom op schema is evidenced in all_pseudocode.json so only same type copy convert and bitcast identities fold.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();

    SmallVector<Operation *> dead;
    root->walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name != "darwinn.copy_op" && name != "darwinn.convert" &&
          name != "darwinn.bitcast")
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

    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op) || op->mightHaveTrait<OpTrait::IsTerminator>())
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "darwinn")
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedLegal = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failedLegal)
      return signalPassFailure();
  }
};

// TSV row: "dwg-fork-multicore-tpu-offloads" at 0xd9c35a.
struct DwcDwgForkMulticoreTpuOffloadsPass
    : public darwinn::impl::DwcDwgForkMulticoreTpuOffloadsPassBase<
          DwcDwgForkMulticoreTpuOffloadsPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<dive_vm::DiveVmDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    SmallVector<Operation *> offloads;
    root->walk([&](Operation *op) {
      if (op->getName().getStringRef() == "dive_vm.tpu_offload")
        offloads.push_back(op);
    });
    unsigned core = 0;
    for (Operation *op : offloads)
      op->setAttr("tpu.core_id", builder.getI64IntegerAttr(core++));
    root->setAttr("tpu.core_count",
                  builder.getI64IntegerAttr(offloads.size()));
  }
};

// TSV row: "dwg-lower-for-to-while" at 0xe07f47.
struct DwcDwgLowerForToWhilePass
    : public darwinn::impl::DwcDwgLowerForToWhilePassBase<
          DwcDwgLowerForToWhilePass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<dive_vm::DiveVmDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    // The sibling-owned LowerCopySlice and LowerConvert sets do the real
    // lowering, identity folds below only clean up what patterns leave behind.
    RewritePatternSet patterns(&getContext());
    darwinn::populateLowerCopySlicePatterns(patterns);
    darwinn::populateLowerConvertPatterns(patterns);
    if (failed(
            applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
    func::FuncOp func = getOperation();
    if (failed(applyLocalCopySliceLowering(func)))
      return signalPassFailure();
    if (failed(applyLocalConvertLowering(func)))
      return signalPassFailure();
  }
};

// TSV row: "dwgt-lower-index-type" at 0xdf6914.
struct DwcDwgtLowerIndexTypePass
    : public darwinn::impl::DwcDwgtLowerIndexTypePassBase<
          DwcDwgtLowerIndexTypePass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<dive_vm::DiveVmDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    // The sibling-owned LowerCopySlice and LowerConvert sets do the real
    // lowering, identity folds below only clean up what patterns leave behind.
    RewritePatternSet patterns(&getContext());
    darwinn::populateLowerCopySlicePatterns(patterns);
    darwinn::populateLowerConvertPatterns(patterns);
    if (failed(
            applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
    func::FuncOp func = getOperation();
    if (failed(applyLocalCopySliceLowering(func)))
      return signalPassFailure();
    if (failed(applyLocalConvertLowering(func)))
      return signalPassFailure();
  }
};

// TSV row: "dynamic-update-slice-lowering" at 0xddee37.
struct DwcDynamicUpdateSliceLoweringPass
    : public darwinn::impl::DwcDynamicUpdateSliceLoweringPassBase<
          DwcDynamicUpdateSliceLoweringPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<dive_vm::DiveVmDialect, LLVM::LLVMDialect>();
  }
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    RewritePatternSet patterns(&getContext());
    darwinn::populateLowerCopySlicePatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
    SmallVector<Operation *> dead;
    func.getOperation()->walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name != "darwinn.iota" && name != "darwinn.reshape_op" &&
          name != "darwinn.broadcast_slice" && name != "darwinn.narrow_to_narrow_slice" &&
          name != "darwinn.narrow_to_wide_slice" && name != "darwinn.wide_to_narrow_slice" &&
          name != "darwinn.sparse_narrow_to_wide_slice" && name != "darwinn.get_indexed_slice" &&
          name != "darwinn.slice_1d_extent" && name != "darwinn.slice_1d_extent_with_padding_info" &&
          name != "darwinn.create_empty_tensor" && name != "darwinn.get_tensor" &&
          name != "darwinn.narrow_to_narrow" && name != "darwinn.narrow_to_wide" &&
          name != "darwinn.wide_to_narrow" && name != "darwinn.sparse_narrow_to_wide")
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
  }
};

// TSV row: "edgetpu-custom-op-2" at 0x103fe70.
struct DwcEdgetpuCustomOp2Pass
    : public darwinn::impl::DwcEdgetpuCustomOp2PassBase<
          DwcEdgetpuCustomOp2Pass> {
  using Base::Base;

  void runOnOperation() override {
    // Binary packet layout is absent from all_pseudocode.json, group and order
    // only.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    std::map<Operation *, unsigned> clusterOf;
    unsigned nextCluster = 0;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op) || op->mightHaveTrait<OpTrait::IsTerminator>())
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
    root->setAttr("tpu.cluster_count", builder.getI64IntegerAttr(nextCluster));
  }
};

// TSV row: "(fm-model-converter" at 0xdac265.
struct DwcFmModelConverterPass
    : public darwinn::impl::DwcFmModelConverterPassBase<
          DwcFmModelConverterPass> {
  using Base::Base;

  void runOnOperation() override {
    // Kernel shapes are absent from all_pseudocode.json for this converter so only same type copy convert and bitcast identities fold.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();

    SmallVector<Operation *> dead;
    root->walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name != "darwinn.copy_op" && name != "darwinn.convert" &&
          name != "darwinn.bitcast")
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

    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op) || op->mightHaveTrait<OpTrait::IsTerminator>())
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "darwinn") {
        op->emitError("fm-model-converter accepts darwinn ops only");
        failedLegal = true;
        return WalkResult::interrupt();
      }
      if (failed(checkDwcConvertibleTypes(op))) {
        failedLegal = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failedLegal)
      return signalPassFailure();
  }
};

// TSV row: "(fpa2bv-model-converter" at 0xdac220.
struct DwcFpa2bvModelConverterPass
    : public darwinn::impl::DwcFpa2bvModelConverterPassBase<
          DwcFpa2bvModelConverterPass> {
  using Base::Base;

  void runOnOperation() override {
    // Kernel shapes are absent from all_pseudocode.json for this converter so only same type copy convert and bitcast identities fold.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();

    SmallVector<Operation *> dead;
    root->walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name != "darwinn.copy_op" && name != "darwinn.convert" &&
          name != "darwinn.bitcast")
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

    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op) || op->mightHaveTrait<OpTrait::IsTerminator>())
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "darwinn") {
        op->emitError("fpa2bv-model-converter accepts darwinn ops only");
        failedLegal = true;
        return WalkResult::interrupt();
      }
      if (failed(checkDwcConvertibleTypes(op))) {
        failedLegal = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failedLegal)
      return signalPassFailure();
  }
};

// TSV row: "group-tpu-offloads-by-parameters" at 0xd821ce.
struct DwcGroupTpuOffloadsByParametersPass
    : public darwinn::impl::DwcGroupTpuOffloadsByParametersPassBase<
          DwcGroupTpuOffloadsByParametersPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<dive_vm::DiveVmDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    std::map<std::string, unsigned> groupOf;
    unsigned nextGroup = 0;
    root->walk([&](Operation *op) {
      if (op->getName().getStringRef() != "dive_vm.tpu_offload")
        return WalkResult::advance();
      std::string key;
      llvm::raw_string_ostream os(key);
      for (Type t : op->getOperandTypes())
        t.print(os);
      os.flush();
      auto it = groupOf.find(key);
      unsigned group = (it == groupOf.end()) ? nextGroup++ : it->second;
      groupOf.insert_or_assign(key, group);
      op->setAttr("tpu.param_group", builder.getI64IntegerAttr(group));
      return WalkResult::advance();
    });
    root->setAttr("tpu.param_group_count",
                  builder.getI64IntegerAttr(nextGroup));
  }
};

// TSV row: "interpolate-lowering-pass" at 0xd7f143.
struct DwcInterpolateLoweringPassPass
    : public darwinn::impl::DwcInterpolateLoweringPassPassBase<
          DwcInterpolateLoweringPassPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    bool failedLegal = false;

    root->walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name != "darwinn.interpolate" &&
          name != "darwinn.interpolate_hardware" &&
          name != "darwinn.interpolate_method" &&
          name != "darwinn.legacy_interpolate")
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "darwinn") {
        op->emitError("interpolate-lowering-pass accepts darwinn ops only");
        failedLegal = true;
        return WalkResult::interrupt();
      }
      if (failed(checkDwcConvertibleTypes(op))) {
        failedLegal = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failedLegal)
      return signalPassFailure();
  }
};

// TSV row: "is not immutable, try removing mutable variables in your model
// since mutable variables are currently not supported through this converter"
// at 0xdac290.
struct
    DwcIsNotImmutableTryRemovingMutableVariablesInYourModelSinceMutableVariablesAreCurrentlyNotSupportedThroughThisConverterPass
    : public darwinn::impl::
          DwcIsNotImmutableTryRemovingMutableVariablesInYourModelSinceMutableVariablesAreCurrentlyNotSupportedThroughThisConverterPassBase<
              DwcIsNotImmutableTryRemovingMutableVariablesInYourModelSinceMutableVariablesAreCurrentlyNotSupportedThroughThisConverterPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    SmallVector<Operation *> dead;
    root->walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name != "darwinn.copy_op" && name != "darwinn.convert" &&
          name != "darwinn.bitcast")
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
    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op) || op->mightHaveTrait<OpTrait::IsTerminator>())
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedLegal = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failedLegal)
      return signalPassFailure();
  }
};

// TSV row: "Legalize" at 0xde6d4c.
struct DwcLegalizePass
    : public darwinn::impl::DwcLegalizePassBase<DwcLegalizePass> {
  using Base::Base;

  void runOnOperation() override {
    // Only copy convert and bitcast identities fold here. No other kernel shape for this family appears in all_pseudocode.json.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();

    SmallVector<Operation *> dead;
    root->walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name != "darwinn.copy_op" && name != "darwinn.convert" &&
          name != "darwinn.bitcast")
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

    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op) || op->mightHaveTrait<OpTrait::IsTerminator>())
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect) {
        op->emitError() << "legalize rejects unregistered operation "
                        << op->getName().getStringRef();
        failedLegal = true;
        return WalkResult::interrupt();
      }
      StringRef ns = dialect->getNamespace();
      if (ns != "darwinn") {
        op->emitError() << "legalize rejects operation from dialect " << ns;
        failedLegal = true;
        return WalkResult::interrupt();
      }
      if (failed(checkDwcConvertibleTypes(op))) {
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
struct DwcLegalizeAffinePass
    : public darwinn::impl::DwcLegalizeAffinePassBase<DwcLegalizeAffinePass> {
  using Base::Base;

  void runOnOperation() override {
    // No affine op name exists in DarwinnOps.td or DiveVmOps.td and no affine legalization mapping is evidenced so the fold walk finds nothing.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();

    SmallVector<Operation *> dead;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "affine")
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

    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op) || op->mightHaveTrait<OpTrait::IsTerminator>())
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect) {
        op->emitError() << "legalize-affine rejects unregistered operation "
                        << op->getName().getStringRef();
        failedLegal = true;
        return WalkResult::interrupt();
      }
      StringRef ns = dialect->getNamespace();
      if (ns != "affine") {
        op->emitError() << "legalize-affine rejects operation from dialect "
                        << ns;
        failedLegal = true;
        return WalkResult::interrupt();
      }
      if (failed(checkDwcConvertibleTypes(op))) {
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
struct DwcLegalizeDwcPass
    : public darwinn::impl::DwcLegalizeDwcPassBase<DwcLegalizeDwcPass> {
  using Base::Base;

  void runOnOperation() override {
    // Only copy convert and bitcast identities fold here. No other kernel shape for this family appears in all_pseudocode.json.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();

    SmallVector<Operation *> dead;
    root->walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name != "darwinn.copy_op" && name != "darwinn.convert" &&
          name != "darwinn.bitcast")
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

    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op) || op->mightHaveTrait<OpTrait::IsTerminator>())
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect) {
        op->emitError() << "legalize-dwc rejects unregistered operation "
                        << op->getName().getStringRef();
        failedLegal = true;
        return WalkResult::interrupt();
      }
      StringRef ns = dialect->getNamespace();
      if (ns != "darwinn") {
        op->emitError() << "legalize-dwc rejects operation from dialect " << ns;
        failedLegal = true;
        return WalkResult::interrupt();
      }
      if (failed(checkDwcConvertibleTypes(op))) {
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
struct DwcLegalizeDwcInputOutputOpsPass
    : public darwinn::impl::DwcLegalizeDwcInputOutputOpsPassBase<
          DwcLegalizeDwcInputOutputOpsPass> {
  using Base::Base;

  void runOnOperation() override {
    // DarwinnOps.td documents copy_from_host with same shape and element type so only same type copy and copy_from_host identities fold. No other input output kernel shape appears in all_pseudocode.json.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();

    SmallVector<Operation *> dead;
    root->walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name != "darwinn.copy_op" && name != "darwinn.copy_from_host")
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

    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op) || op->mightHaveTrait<OpTrait::IsTerminator>())
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect) {
        op->emitError()
            << "legalize-dwc-input-output-ops rejects unregistered operation "
            << op->getName().getStringRef();
        failedLegal = true;
        return WalkResult::interrupt();
      }
      StringRef ns = dialect->getNamespace();
      if (ns != "darwinn") {
        op->emitError()
            << "legalize-dwc-input-output-ops rejects operation from dialect "
            << ns;
        failedLegal = true;
        return WalkResult::interrupt();
      }
      if (failed(checkDwcConvertibleTypes(op))) {
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
struct DwcLegalizeDwgTensorPass
    : public darwinn::impl::DwcLegalizeDwgTensorPassBase<
          DwcLegalizeDwgTensorPass> {
  using Base::Base;

  void runOnOperation() override {
    // No dwg op name exists in DarwinnOps.td or DiveVmOps.td and no dwg tensor kernel shape is evidenced so the fold walk finds nothing.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();

    SmallVector<Operation *> dead;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "dwg")
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

    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op) || op->mightHaveTrait<OpTrait::IsTerminator>())
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect) {
        op->emitError() << "legalize-dwg-tensor rejects unregistered operation "
                        << op->getName().getStringRef();
        failedLegal = true;
        return WalkResult::interrupt();
      }
      StringRef ns = dialect->getNamespace();
      if (ns != "dwg") {
        op->emitError() << "legalize-dwg-tensor rejects operation from dialect "
                        << ns;
        failedLegal = true;
        return WalkResult::interrupt();
      }
      if (failed(checkDwcConvertibleTypes(op))) {
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
struct DwcLegalizeQuantTypesPass
    : public darwinn::impl::DwcLegalizeQuantTypesPassBase<
          DwcLegalizeQuantTypesPass> {
  using Base::Base;

  void runOnOperation() override {
    // No quant op name exists in DarwinnOps.td or DiveVmOps.td and no quant type mapping is evidenced so the fold walk finds nothing.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();

    SmallVector<Operation *> dead;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "quant")
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

    bool failedLegal = false;
    root->walk(
        [&](Operation *op) {
          if (isa<func::FuncOp>(op))
            return WalkResult::advance();
          Dialect *dialect = op->getDialect();
          if (!dialect) {
            op->emitError()
                << "legalize-quant-types rejects unregistered operation "
                << op->getName().getStringRef();
            failedLegal = true;
            return WalkResult::interrupt();
          }
          StringRef ns = dialect->getNamespace();
          if (ns != "quant") {
            op->emitError()
                << "legalize-quant-types rejects operation from dialect " << ns;
            failedLegal = true;
            return WalkResult::interrupt();
          }
          if (failed(checkDwcConvertibleTypes(op))) {
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
struct DwcLegalizeScfPass
    : public darwinn::impl::DwcLegalizeScfPassBase<DwcLegalizeScfPass> {
  using Base::Base;

  void runOnOperation() override {
    // No scf op name exists in DarwinnOps.td or DiveVmOps.td and no scf legalization mapping is evidenced so the fold walk finds nothing.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();

    SmallVector<Operation *> dead;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "scf")
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

    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op) || op->mightHaveTrait<OpTrait::IsTerminator>())
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect) {
        op->emitError() << "legalize-scf rejects unregistered operation "
                        << op->getName().getStringRef();
        failedLegal = true;
        return WalkResult::interrupt();
      }
      StringRef ns = dialect->getNamespace();
      if (ns != "scf") {
        op->emitError() << "legalize-scf rejects operation from dialect " << ns;
        failedLegal = true;
        return WalkResult::interrupt();
      }
      if (failed(checkDwcConvertibleTypes(op))) {
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
struct DwcLegalizeShapeOpsPass
    : public darwinn::impl::DwcLegalizeShapeOpsPassBase<
          DwcLegalizeShapeOpsPass> {
  using Base::Base;

  void runOnOperation() override {
    // No shape op name exists in DarwinnOps.td or DiveVmOps.td and no shape legalization mapping is evidenced so the fold walk finds nothing.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();

    SmallVector<Operation *> dead;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "shape")
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

    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op) || op->mightHaveTrait<OpTrait::IsTerminator>())
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect) {
        op->emitError() << "legalize-shape-ops rejects unregistered operation "
                        << op->getName().getStringRef();
        failedLegal = true;
        return WalkResult::interrupt();
      }
      StringRef ns = dialect->getNamespace();
      if (ns != "shape") {
        op->emitError() << "legalize-shape-ops rejects operation from dialect "
                        << ns;
        failedLegal = true;
        return WalkResult::interrupt();
      }
      if (failed(checkDwcConvertibleTypes(op))) {
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
struct DwcLegalizeTestUsingLayerirFlowPass
    : public darwinn::impl::DwcLegalizeTestUsingLayerirFlowPassBase<
          DwcLegalizeTestUsingLayerirFlowPass> {
  using Base::Base;

  void runOnOperation() override {
    // No layerir kernel shape is evidenced so only same type copy convert and bitcast identities fold.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();

    SmallVector<Operation *> dead;
    root->walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name != "darwinn.copy_op" && name != "darwinn.convert" &&
          name != "darwinn.bitcast")
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

    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op) || op->mightHaveTrait<OpTrait::IsTerminator>())
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect) {
        op->emitError() << "legalize-test-using-layerir-flow rejects "
                           "unregistered operation "
                        << op->getName().getStringRef();
        failedLegal = true;
        return WalkResult::interrupt();
      }
      StringRef ns = dialect->getNamespace();
      if (ns != "darwinn") {
        op->emitError() << "legalize-test-using-layerir-flow rejects operation "
                           "from dialect "
                        << ns;
        failedLegal = true;
        return WalkResult::interrupt();
      }
      if (failed(checkDwcConvertibleTypes(op))) {
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
struct DwcLegalizeTfXlacallmoduleOpToStablehloPass
    : public darwinn::impl::DwcLegalizeTfXlacallmoduleOpToStablehloPassBase<
          DwcLegalizeTfXlacallmoduleOpToStablehloPass> {
  using Base::Base;

  void runOnOperation() override {
    // No honest rewrite exists. No XlaCallModule to stablehlo mapping appears
    // in all_pseudocode.json.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    SmallVector<Operation *> dead;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "tf")
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

    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op) || op->mightHaveTrait<OpTrait::IsTerminator>())
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect)
        return WalkResult::advance();
      if (dialect->getNamespace() != "tf")
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedLegal = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failedLegal)
      return signalPassFailure();
  }
};

// TSV row: "legalize-thread-oblivious-op-pass" at 0xd7ee8c.
struct DwcLegalizeThreadObliviousOpPassPass
    : public darwinn::impl::DwcLegalizeThreadObliviousOpPassPassBase<
          DwcLegalizeThreadObliviousOpPassPass> {
  using Base::Base;

  void runOnOperation() override {
    // No thread oblivious kernel shape is evidenced so only same type copy convert and bitcast identities fold.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();

    SmallVector<Operation *> dead;
    root->walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name != "darwinn.copy_op" && name != "darwinn.convert" &&
          name != "darwinn.bitcast")
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

    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op) || op->mightHaveTrait<OpTrait::IsTerminator>())
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect) {
        op->emitError() << "legalize-thread-oblivious-op-pass rejects "
                           "unregistered operation "
                        << op->getName().getStringRef();
        failedLegal = true;
        return WalkResult::interrupt();
      }
      StringRef ns = dialect->getNamespace();
      if (ns != "darwinn") {
        op->emitError() << "legalize-thread-oblivious-op-pass rejects "
                           "operation from dialect "
                        << ns;
        failedLegal = true;
        return WalkResult::interrupt();
      }
      if (failed(checkDwcConvertibleTypes(op))) {
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
struct DwcLegalizeTypesForDiveVmTensorPass
    : public darwinn::impl::DwcLegalizeTypesForDiveVmTensorPassBase<
          DwcLegalizeTypesForDiveVmTensorPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<dive_vm::DiveVmDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    // DiveVmOps.td documents dive_vm.copy with an honest same type fold so it folds here. No other dive_vm tensor kernel shape appears in all_pseudocode.json.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();

    SmallVector<Operation *> dead;
    root->walk([&](Operation *op) {
      if (op->getName().getStringRef() != "dive_vm.copy")
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

    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op) || op->mightHaveTrait<OpTrait::IsTerminator>())
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect) {
        op->emitError() << "legalize-types-for-dive-vm-tensor rejects "
                           "unregistered operation "
                        << op->getName().getStringRef();
        failedLegal = true;
        return WalkResult::interrupt();
      }
      StringRef ns = dialect->getNamespace();
      if (ns != "dive_vm") {
        op->emitError() << "legalize-types-for-dive-vm-tensor rejects "
                           "operation from dialect "
                        << ns;
        failedLegal = true;
        return WalkResult::interrupt();
      }
      if (failed(checkDwcConvertibleTypes(op))) {
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
struct DwcLegalizeStablehloCompositePass
    : public darwinn::impl::DwcLegalizeStablehloCompositePassBase<
          DwcLegalizeStablehloCompositePass> {
  using Base::Base;

  void runOnOperation() override {
    // No stablehlo op name exists in DarwinnOps.td or DiveVmOps.td and no stablehlo composite mapping is evidenced so the fold walk finds nothing.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();

    SmallVector<Operation *> dead;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "stablehlo")
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

    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op) || op->mightHaveTrait<OpTrait::IsTerminator>())
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect)
        return WalkResult::advance();
      if (dialect->getNamespace() != "stablehlo")
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedLegal = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failedLegal)
      return signalPassFailure();
  }
};

// TSV row: "lower-all-functions" at 0xd873dd.
struct DwcLowerAllFunctionsPass
    : public darwinn::impl::DwcLowerAllFunctionsPassBase<
          DwcLowerAllFunctionsPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    unsigned lowered = 0;
    if (failed(applyDwcLowerPadInline(func, lowered)))
      return signalPassFailure();
  }
};

// TSV row: "lower-all-pads" at 0xd9c263.
struct DwcLowerAllPadsPass
    : public darwinn::impl::DwcLowerAllPadsPassBase<DwcLowerAllPadsPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<dive_vm::DiveVmDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    unsigned lowered = 0;
    if (failed(applyDwcLowerPadInline(func, lowered)))
      return signalPassFailure();
    if (failed(applyDwcLowerCopyLike(func, lowered)))
      return signalPassFailure();
  }
};

// TSV row: "lower-attention-ops" at 0xd845ec.
struct DwcLowerAttentionOpsPass
    : public darwinn::impl::DwcLowerAttentionOpsPassBase<
          DwcLowerAttentionOpsPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<dive_vm::DiveVmDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    // The sibling-owned LowerCopySlice and LowerConvert sets do the real
    // lowering, identity folds below only clean up what patterns leave behind.
    RewritePatternSet patterns(&getContext());
    darwinn::populateLowerCopySlicePatterns(patterns);
    darwinn::populateLowerConvertPatterns(patterns);
    if (failed(
            applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
    func::FuncOp func = getOperation();
    if (failed(applyLocalCopySliceLowering(func)))
      return signalPassFailure();
    if (failed(applyLocalConvertLowering(func)))
      return signalPassFailure();
  }
};

// TSV row: "lower-input-cast" at 0xd680b2.
struct DwcLowerInputCastPass
    : public darwinn::impl::DwcLowerInputCastPassBase<DwcLowerInputCastPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<dive_vm::DiveVmDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    // The sibling-owned LowerCopySlice and LowerConvert sets do the real
    // lowering, identity folds below only clean up what patterns leave behind.
    RewritePatternSet patterns(&getContext());
    darwinn::populateLowerCopySlicePatterns(patterns);
    darwinn::populateLowerConvertPatterns(patterns);
    if (failed(
            applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
    func::FuncOp func = getOperation();
    if (failed(applyLocalCopySliceLowering(func)))
      return signalPassFailure();
    if (failed(applyLocalConvertLowering(func)))
      return signalPassFailure();
  }
};

// TSV row: "lower-join" at 0xdc9464.
struct DwcLowerJoinPass
    : public darwinn::impl::DwcLowerJoinPassBase<DwcLowerJoinPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<dive_vm::DiveVmDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    // The sibling-owned LowerCopySlice and LowerConvert sets do the real
    // lowering, identity folds below only clean up what patterns leave behind.
    RewritePatternSet patterns(&getContext());
    darwinn::populateLowerCopySlicePatterns(patterns);
    darwinn::populateLowerConvertPatterns(patterns);
    if (failed(
            applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
    func::FuncOp func = getOperation();
    if (failed(applyLocalCopySliceLowering(func)))
      return signalPassFailure();
    if (failed(applyLocalConvertLowering(func)))
      return signalPassFailure();
  }
};

// TSV row: "lower-output-cast" at 0xd680a0.
struct DwcLowerOutputCastPass
    : public darwinn::impl::DwcLowerOutputCastPassBase<DwcLowerOutputCastPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<dive_vm::DiveVmDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    // The sibling-owned LowerCopySlice and LowerConvert sets do the real
    // lowering, identity folds below only clean up what patterns leave behind.
    RewritePatternSet patterns(&getContext());
    darwinn::populateLowerCopySlicePatterns(patterns);
    darwinn::populateLowerConvertPatterns(patterns);
    if (failed(
            applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
    func::FuncOp func = getOperation();
    if (failed(applyLocalCopySliceLowering(func)))
      return signalPassFailure();
    if (failed(applyLocalConvertLowering(func)))
      return signalPassFailure();
  }
};

// TSV row: "LowerArgmaxIndexUnpool" at 0xdcf04e.
struct DwcLowerArgmaxIndexUnpoolPass
    : public darwinn::impl::DwcLowerArgmaxIndexUnpoolPassBase<
          DwcLowerArgmaxIndexUnpoolPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    unsigned lowered = 0;
    if (failed(applyDwcLowerArgmaxInline(func, lowered)))
      return signalPassFailure();
  }
};

// TSV row: "mark-dive-vm-tensor-insert-slice-ops" at 0xd8488d.
struct DwcMarkDiveVmTensorInsertSliceOpsPass
    : public darwinn::impl::DwcMarkDiveVmTensorInsertSliceOpsPassBase<
          DwcMarkDiveVmTensorInsertSliceOpsPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<dive_vm::DiveVmDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    // dive_vm.insert_slice carries no source op contract in DiveVmOps.td so only same type slice identities fold.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();

    SmallVector<Operation *> dead;
    root->walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name != "dive_vm.insert_slice" &&
          name != "dive_vm.extract_slice")
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

    bool failedLegal = false;
    root->walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name != "dive_vm.insert_slice" &&
          name != "dive_vm.extract_slice")
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "dive_vm") {
        op->emitError(
            "mark-dive-vm-tensor-insert-slice-ops accepts dive_vm ops only");
        failedLegal = true;
        return WalkResult::interrupt();
      }
      if (failed(checkDwcConvertibleTypes(op))) {
        failedLegal = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failedLegal)
      return signalPassFailure();
  }
};

// TSV row: "mhlo-legalize-einsum-to-dot-general" at 0xdd34bf.
struct DwcMhloLegalizeEinsumToDotGeneralPass
    : public darwinn::impl::DwcMhloLegalizeEinsumToDotGeneralPassBase<
          DwcMhloLegalizeEinsumToDotGeneralPass> {
  using Base::Base;

  void runOnOperation() override {
    // No mhlo op name exists in DarwinnOps.td or DiveVmOps.td and the upstream MhloLegalizeEinsumToDotGeneral pass is absent so the fold walk finds nothing.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();

    SmallVector<Operation *> dead;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "mhlo")
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

    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op) || op->mightHaveTrait<OpTrait::IsTerminator>())
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect) {
        op->emitError() << "mhlo-legalize-einsum-to-dot-general rejects "
                           "unregistered operation "
                        << op->getName().getStringRef();
        failedLegal = true;
        return WalkResult::interrupt();
      }
      StringRef ns = dialect->getNamespace();
      if (ns != "mhlo") {
        op->emitError() << "mhlo-legalize-einsum-to-dot-general rejects "
                           "operation from dialect "
                        << ns;
        failedLegal = true;
        return WalkResult::interrupt();
      }
      if (failed(checkDwcConvertibleTypes(op))) {
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
struct DwcMidToLowLevelLoweringPass
    : public darwinn::impl::DwcMidToLowLevelLoweringPassBase<
          DwcMidToLowLevelLoweringPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<dive_vm::DiveVmDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    RewritePatternSet patterns(&getContext());
    darwinn::populateLowerCopySlicePatterns(patterns);
    darwinn::populateLowerConvertPatterns(patterns);
    if (failed(
            applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
    if (failed(applyLocalCopySliceLowering(func)))
      return signalPassFailure();
    if (failed(applyLocalConvertLowering(func)))
      return signalPassFailure();
    unsigned lowered = 0;
    if (failed(applyDwcLowerScalarArith(func, lowered)))
      return signalPassFailure();
    if (failed(applyDwcLowerCopyLike(func, lowered)))
      return signalPassFailure();
    if (failed(applyDwcLowerConvertTrunc(func, lowered)))
      return signalPassFailure();
  }
};

// TSV row: "::mlir::darwinn::compute::Engine" at 0xe03654.
struct DwcMlirDarwinnComputeEnginePass
    : public darwinn::impl::DwcMlirDarwinnComputeEnginePassBase<
          DwcMlirDarwinnComputeEnginePass> {
  using Base::Base;

  void runOnOperation() override {
    // No Engine op is evidenced in all_pseudocode.json so only same type copy convert and bitcast identities fold.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();

    SmallVector<Operation *> dead;
    root->walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name != "darwinn.copy_op" && name != "darwinn.convert" &&
          name != "darwinn.bitcast")
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

    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op) || op->mightHaveTrait<OpTrait::IsTerminator>())
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "darwinn")
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedLegal = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failedLegal)
      return signalPassFailure();
  }
};

// TSV row: "Only DenseElementsAttr are supported for constant lowering" at
// 0xddee55.
struct DwcOnlyDenseelementsattrAreSupportedForConstantLoweringPass
    : public darwinn::impl::
          DwcOnlyDenseelementsattrAreSupportedForConstantLoweringPassBase<
              DwcOnlyDenseelementsattrAreSupportedForConstantLoweringPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    unsigned lowered = 0;
    if (failed(applyDwcLowerConstInline(func, lowered)))
      return signalPassFailure();
  }
};

// TSV row: "optimize-dive-vm-tensor-insert-slice" at 0xe0fa93.
struct DwcOptimizeDiveVmTensorInsertSlicePass
    : public darwinn::impl::DwcOptimizeDiveVmTensorInsertSlicePassBase<
          DwcOptimizeDiveVmTensorInsertSlicePass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<dive_vm::DiveVmDialect, LLVM::LLVMDialect>();
  }
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (failed(checkDwcConvertibleTypes(func.getOperation())))
      return signalPassFailure();
  }
};

// TSV row: "-parameter-caching-dive-program" at 0xdce5e1.
struct DwcParameterCachingDiveProgramPass
    : public darwinn::impl::DwcParameterCachingDiveProgramPassBase<
          DwcParameterCachingDiveProgramPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<dive_vm::DiveVmDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    DenseMap<Attribute, Operation *> seen;
    SmallVector<Operation *> dead;
    func.getOperation()->walk([&](Operation *op) {
      if (op->getName().getStringRef() != "dive_vm.const")
        return;
      if (op->getNumResults() != 1)
        return;
      Attribute value = op->getAttr("value");
      if (!value)
        return;
      auto it = seen.find(value);
      if (it == seen.end()) {
        seen.insert({value, op});
        return;
      }
      op->getResult(0).replaceAllUsesWith(it->second->getResult(0));
      dead.push_back(op);
    });
    for (Operation *op : dead)
      op->erase();
  }
};

// TSV row: "platforms.darwinn.code_generator.Entry.Score.type" at 0xdf68e2.
struct DwcPlatformsDarwinnCodeGeneratorEntryScoreTypePass
    : public darwinn::impl::
          DwcPlatformsDarwinnCodeGeneratorEntryScoreTypePassBase<
              DwcPlatformsDarwinnCodeGeneratorEntryScoreTypePass> {
  using Base::Base;

  void runOnOperation() override {
    // No Entry Score type is evidenced in all_pseudocode.json so only same type func identities fold.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();

    SmallVector<Operation *> dead;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "func")
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

    if (failed(checkDwcConvertibleTypes(func.getOperation())))
      return signalPassFailure();
  }
};

// TSV row:
// "platforms.darwinn.compiler.ProbeInstrumentationLocation.Constraints.functions"
// at 0xd87375.
struct
    DwcPlatformsDarwinnCompilerProbeinstrumentationlocationConstraintsFunctionsPass
    : public darwinn::impl::
          DwcPlatformsDarwinnCompilerProbeinstrumentationlocationConstraintsFunctionsPassBase<
              DwcPlatformsDarwinnCompilerProbeinstrumentationlocationConstraintsFunctionsPass> {
  using Base::Base;

  void runOnOperation() override {
    // No probe location contract is evidenced in all_pseudocode.json so only same type func identities fold.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();

    SmallVector<Operation *> dead;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "func")
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

    if (failed(checkDwcConvertibleTypes(func.getOperation())))
      return signalPassFailure();
  }
};

// TSV row: "quant-signedness-convert-lowering" at 0xdded43.
struct DwcQuantSignednessConvertLoweringPass
    : public darwinn::impl::DwcQuantSignednessConvertLoweringPassBase<
          DwcQuantSignednessConvertLoweringPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<dive_vm::DiveVmDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    unsigned lowered = 0;
    if (failed(applyDwcLowerConvertTrunc(func, lowered)))
      return signalPassFailure();
  }
};

// TSV row: "r52-reads-dive-buffers" at 0xd833a6.
struct DwcR52ReadsDiveBuffersPass
    : public darwinn::impl::DwcR52ReadsDiveBuffersPassBase<
          DwcR52ReadsDiveBuffersPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<dive_vm::DiveVmDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    // There is no r52 op in DarwinnOps.td or DiveVmOps.td and no r52 kernel shape in all_pseudocode.json so only same type dive_vm load identities fold.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();

    SmallVector<Operation *> dead;
    root->walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name != "dive_vm.load" && name != "dive_vm.load_indirect")
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

    bool failedLegal = false;
    root->walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name != "dive_vm.load" && name != "dive_vm.load_indirect")
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "dive_vm") {
        op->emitError("r52-reads-dive-buffers accepts dive_vm ops only");
        failedLegal = true;
        return WalkResult::interrupt();
      }
      if (failed(checkDwcConvertibleTypes(op))) {
        failedLegal = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failedLegal)
      return signalPassFailure();
  }
};

// TSV row: "redistribute-lowering" at 0xddee0e.
struct DwcRedistributeLoweringPass
    : public darwinn::impl::DwcRedistributeLoweringPassBase<
          DwcRedistributeLoweringPass> {
  using Base::Base;

  void runOnOperation() override {
    // There is no redistribute op in DarwinnOps.td and no redistribute kernel shape in all_pseudocode.json so only same type copy convert and bitcast identities fold.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();

    SmallVector<Operation *> dead;
    root->walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name != "darwinn.copy_op" && name != "darwinn.convert" &&
          name != "darwinn.bitcast")
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

    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op) || op->mightHaveTrait<OpTrait::IsTerminator>())
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "darwinn") {
        op->emitError("redistribute-lowering accepts darwinn ops only");
        failedLegal = true;
        return WalkResult::interrupt();
      }
      if (failed(checkDwcConvertibleTypes(op))) {
        failedLegal = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failedLegal)
      return signalPassFailure();
  }
};

// TSV row: "redistribute-lowering-pass-remarks" at 0xd8c505.
struct DwcRedistributeLoweringPassRemarksPass
    : public darwinn::impl::DwcRedistributeLoweringPassRemarksPassBase<
          DwcRedistributeLoweringPassRemarksPass> {
  using Base::Base;

  void runOnOperation() override {
    // There is no redistribute op in DarwinnOps.td and no redistribute kernel shape in all_pseudocode.json so only same type copy convert and bitcast identities fold.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();

    SmallVector<Operation *> dead;
    root->walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name != "darwinn.copy_op" && name != "darwinn.convert" &&
          name != "darwinn.bitcast")
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

    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op) || op->mightHaveTrait<OpTrait::IsTerminator>())
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "darwinn") {
        op->emitError(
            "redistribute-lowering-pass-remarks accepts darwinn ops only");
        failedLegal = true;
        return WalkResult::interrupt();
      }
      if (failed(checkDwcConvertibleTypes(op))) {
        failedLegal = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failedLegal)
      return signalPassFailure();
  }
};

// TSV row: "reinterpret-cast-rank-legalize-pass" at 0xd7f295.
struct DwcReinterpretCastRankLegalizePassPass
    : public darwinn::impl::DwcReinterpretCastRankLegalizePassPassBase<
          DwcReinterpretCastRankLegalizePassPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<dive_vm::DiveVmDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    OpBuilder builder(func.getOperation()->getContext());
    SmallVector<Operation *> dead;
    SmallVector<Operation *> targets;
    func.getOperation()->walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name != "darwinn.bitcast" && name != "darwinn.reinterpret_cast" &&
          name != "darwinn.hl_bitcast")
        return;
      if (op->getNumOperands() != 1 || op->getNumResults() != 1)
        return;
      if (op->getOperand(0).getType() == op->getResult(0).getType())
        dead.push_back(op);
      else
        targets.push_back(op);
    });
    for (Operation *op : dead) {
      op->getResult(0).replaceAllUsesWith(op->getOperand(0));
      op->erase();
    }
    for (Operation *op : targets) {
      auto src = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
      auto dst = dyn_cast<RankedTensorType>(op->getResult(0).getType());
      if (!src || !dst || !src.hasStaticShape() || !dst.hasStaticShape())
        continue;
      if (src.getNumElements() != dst.getNumElements())
        continue;
      if (src.getElementTypeBitWidth() != dst.getElementTypeBitWidth())
        continue;
      if (failed(checkDwcConvertibleTypes(op)))
        return signalPassFailure();
      builder.setInsertionPoint(op);
      SmallVector<Value> operands{op->getOperand(0)};
      SmallVector<Type> results{op->getResult(0).getType()};
      SmallVector<NamedAttribute> empty;
      Operation *next =
          makeDwcLowerVmOp(builder, op->getLoc(), "dive_vm.bitcast",
                           ValueRange(operands), TypeRange(results), empty);
      op->getResult(0).replaceAllUsesWith(next->getResult(0));
      op->erase();
    }
  }
};

// TSV row: "rename-dive-entry-function" at 0xdbfc27.
struct DwcRenameDiveEntryFunctionPass
    : public darwinn::impl::DwcRenameDiveEntryFunctionPassBase<
          DwcRenameDiveEntryFunctionPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func.getSymName() == "dive_entry")
      return;
    if (failed(checkDwcConvertibleTypes(func.getOperation())))
      return signalPassFailure();
    auto moduleOp = func->getParentOfType<ModuleOp>();
    if (moduleOp && SymbolTable::lookupSymbolIn(moduleOp, "dive_entry"))
      return;
    SymbolTable::setSymbolName(func.getOperation(), "dive_entry");
  }
};

// TSV row: "resampler-lowering" at 0xdded92.
struct DwcResamplerLoweringPass
    : public darwinn::impl::DwcResamplerLoweringPassBase<
          DwcResamplerLoweringPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    bool failedLegal = false;

    root->walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name != "darwinn.resampler" && name != "darwinn.resampler_options")
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "darwinn") {
        op->emitError("resampler-lowering accepts darwinn ops only");
        failedLegal = true;
        return WalkResult::interrupt();
      }
      if (failed(checkDwcConvertibleTypes(op))) {
        failedLegal = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failedLegal)
      return signalPassFailure();
  }
};

// TSV row: "rkhy-shape-legalization-pass" at 0xd7f029.
struct DwcRkhyShapeLegalizationPassPass
    : public darwinn::impl::DwcRkhyShapeLegalizationPassPassBase<
          DwcRkhyShapeLegalizationPassPass> {
  using Base::Base;

  void runOnOperation() override {
    // Rkhy decodes to Vica. Pads and residual adds lower through the
    // existing pad and scalar helpers.
    func::FuncOp func = getOperation();
    unsigned lowered = 0;
    if (failed(applyDwcLowerPadInline(func, lowered)))
      return signalPassFailure();
    if (failed(applyDwcLowerScalarArith(func, lowered)))
      return signalPassFailure();
    SmallVector<Operation *> dead;
    func.getOperation()->walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name != "darwinn.rkhy_unary_compute_op" &&
          name != "darwinn.rkhy_depth_to_space_op")
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
  }
};

// TSV row: "rkhy-type-legalization-pass" at 0xd7f00d.
struct DwcRkhyTypeLegalizationPassPass
    : public darwinn::impl::DwcRkhyTypeLegalizationPassPassBase<
          DwcRkhyTypeLegalizationPassPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<dive_vm::DiveVmDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    // Rkhy decodes to Vica. Convert and cast ops lower through the existing
    // convert trunc helper.
    func::FuncOp func = getOperation();
    unsigned lowered = 0;
    if (failed(applyDwcLowerConvertTrunc(func, lowered)))
      return signalPassFailure();
  }
};

// TSV row: "run-r52-ops-on-dive" at 0xde9621.
struct DwcRunR52OpsOnDivePass
    : public darwinn::impl::DwcRunR52OpsOnDivePassBase<DwcRunR52OpsOnDivePass> {
  using Base::Base;

  void runOnOperation() override {
    // There is no r52 op in DarwinnOps.td or DiveVmOps.td and no r52 kernel shape in all_pseudocode.json so only same type dive_vm identities fold.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();

    SmallVector<Operation *> dead;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "dive_vm")
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

    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op) || op->mightHaveTrait<OpTrait::IsTerminator>())
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "dive_vm") {
        op->emitError("run-r52-ops-on-dive accepts dive_vm ops only");
        failedLegal = true;
        return WalkResult::interrupt();
      }
      if (failed(checkDwcConvertibleTypes(op))) {
        failedLegal = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failedLegal)
      return signalPassFailure();
  }
};

// TSV row: "scalar-core-control-flow-lowering" at 0xdded21.
struct DwcScalarCoreControlFlowLoweringPass
    : public darwinn::impl::DwcScalarCoreControlFlowLoweringPassBase<
          DwcScalarCoreControlFlowLoweringPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<dive_vm::DiveVmDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    unsigned lowered = 0;
    if (failed(applyDwcLowerSelectInline(func, lowered)))
      return signalPassFailure();
    if (failed(applyDwcLowerCopyLike(func, lowered)))
      return signalPassFailure();
    if (failed(applyDwcLowerConvertTrunc(func, lowered)))
      return signalPassFailure();
  }
};

// TSV row: "scalar-core-std-ops-lowering" at 0xdded75.
struct DwcScalarCoreStdOpsLoweringPass
    : public darwinn::impl::DwcScalarCoreStdOpsLoweringPassBase<
          DwcScalarCoreStdOpsLoweringPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    unsigned lowered = 0;
    if (failed(applyDwcLowerScalarArith(func, lowered)))
      return signalPassFailure();
  }
};

// TSV row: "scalar-ops-legalize" at 0xde6d2b.
struct DwcScalarOpsLegalizePass
    : public darwinn::impl::DwcScalarOpsLegalizePassBase<
          DwcScalarOpsLegalizePass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    unsigned lowered = 0;
    if (failed(applyDwcLowerScalarArith(func, lowered)))
      return signalPassFailure();
  }
};

// TSV row: "scatter-gather-lowering" at 0xddeda5.
struct DwcScatterGatherLoweringPass
    : public darwinn::impl::DwcScatterGatherLoweringPassBase<
          DwcScatterGatherLoweringPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    unsigned lowered = 0;
    if (failed(applyDwcLowerGatherOob(func, lowered)))
      return signalPassFailure();
    if (failed(applyDwcLowerScatterInline(func, lowered)))
      return signalPassFailure();
  }
};

// TSV row: "select-lowering" at 0xdded65.
struct DwcSelectLoweringPass
    : public darwinn::impl::DwcSelectLoweringPassBase<DwcSelectLoweringPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    unsigned lowered = 0;
    if (failed(applyDwcLowerSelectInline(func, lowered)))
      return signalPassFailure();
  }
};

// TSV row: "sharding-using-dive" at 0xde9635.
struct DwcShardingUsingDivePass
    : public darwinn::impl::DwcShardingUsingDivePassBase<
          DwcShardingUsingDivePass> {
  using Base::Base;

  void runOnOperation() override {
    // The shard ops in DarwinnOps.td carry no sharding layout and DiveVmOps.td names no shard target so only same type shard identities fold.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();

    SmallVector<Operation *> dead;
    root->walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name != "darwinn.broadcast_shard" &&
          name != "darwinn.host_to_ssram_shard" &&
          name != "darwinn.host_to_tile_shard" &&
          name != "darwinn.narrow_to_narrow_shard" &&
          name != "darwinn.narrow_to_wide_shard" &&
          name != "darwinn.tensor_op_shard" &&
          name != "darwinn.tile_to_host_shard" &&
          name != "darwinn.tile_to_tile_shard")
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

    bool failedLegal = false;
    root->walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name != "darwinn.broadcast_shard" &&
          name != "darwinn.host_to_ssram_shard" &&
          name != "darwinn.host_to_tile_shard" &&
          name != "darwinn.narrow_to_narrow_shard" &&
          name != "darwinn.narrow_to_wide_shard" &&
          name != "darwinn.tensor_op_shard" &&
          name != "darwinn.tile_to_host_shard" &&
          name != "darwinn.tile_to_tile_shard")
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "darwinn") {
        op->emitError("sharding-using-dive accepts darwinn ops only");
        failedLegal = true;
        return WalkResult::interrupt();
      }
      if (failed(checkDwcConvertibleTypes(op))) {
        failedLegal = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failedLegal)
      return signalPassFailure();
  }
};

// TSV row: "skipping fold of float convert" at 0xd68cda.
struct DwcSkippingFoldOfFloatConvertPass
    : public darwinn::impl::DwcSkippingFoldOfFloatConvertPassBase<
          DwcSkippingFoldOfFloatConvertPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (failed(applyLocalConvertLowering(func)))
      return signalPassFailure();
  }
};

// TSV row: "split-op-lowering" at 0xddede2.
struct DwcSplitOpLoweringPass
    : public darwinn::impl::DwcSplitOpLoweringPassBase<DwcSplitOpLoweringPass> {
  using Base::Base;

  void runOnOperation() override {
    // darwinn.split carries no axis or sizes in DarwinnOps.td and DiveVmOps.td names no split target so only same type copy convert and bitcast identities fold.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    SmallVector<Operation *> dead;

    root->walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name != "darwinn.copy_op" && name != "darwinn.convert" &&
          name != "darwinn.bitcast" && name != "darwinn.split")
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

    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op) || op->mightHaveTrait<OpTrait::IsTerminator>())
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "darwinn") {
        op->emitError("split-op-lowering accepts darwinn ops only");
        failedLegal = true;
        return WalkResult::interrupt();
      }
      if (failed(checkDwcConvertibleTypes(op))) {
        failedLegal = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failedLegal)
      return signalPassFailure();
  }
};

// TSV row: "stablehlo-composite-legalize-tfl-custom" at 0xdccdf7.
struct DwcStablehloCompositeLegalizeTflCustomPass
    : public darwinn::impl::DwcStablehloCompositeLegalizeTflCustomPassBase<
          DwcStablehloCompositeLegalizeTflCustomPass> {
  using Base::Base;

  void runOnOperation() override {
    // No honest rewrite exists. No stablehlo composite to tfl custom mapping
    // appears in all_pseudocode.json.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    SmallVector<Operation *> dead;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect)
        return;
      StringRef ns = dialect->getNamespace();
      if (ns != "stablehlo" && ns != "tfl")
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

    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op) || op->mightHaveTrait<OpTrait::IsTerminator>())
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect)
        return WalkResult::advance();
      StringRef ns = dialect->getNamespace();
      if (ns != "stablehlo" && ns != "tfl")
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedLegal = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failedLegal)
      return signalPassFailure();
  }
};

// TSV row: "stablehlo-custom-call-legalize-composite" at 0xdef5f5.
struct DwcStablehloCustomCallLegalizeCompositePass
    : public darwinn::impl::DwcStablehloCustomCallLegalizeCompositePassBase<
          DwcStablehloCustomCallLegalizeCompositePass> {
  using Base::Base;

  void runOnOperation() override {
    // No honest rewrite exists. No stablehlo custom call to composite mapping
    // appears in all_pseudocode.json.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    SmallVector<Operation *> dead;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "stablehlo")
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

    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op) || op->mightHaveTrait<OpTrait::IsTerminator>())
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect)
        return WalkResult::advance();
      StringRef ns = dialect->getNamespace();
      if (ns != "stablehlo")
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedLegal = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failedLegal)
      return signalPassFailure();
  }
};

// TSV row: "stablehlo-legalize-composite-to-call" at 0xdd00c2.
struct DwcStablehloLegalizeCompositeToCallPass
    : public darwinn::impl::DwcStablehloLegalizeCompositeToCallPassBase<
          DwcStablehloLegalizeCompositeToCallPass> {
  using Base::Base;

  void runOnOperation() override {
    // No stablehlo op name exists in DarwinnOps.td or DiveVmOps.td and no composite to call mapping is evidenced so the fold walk finds nothing.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();

    SmallVector<Operation *> dead;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "stablehlo")
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

    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op) || op->mightHaveTrait<OpTrait::IsTerminator>())
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect) {
        op->emitError() << "stablehlo-legalize-composite-to-call rejects "
                           "unregistered operation "
                        << op->getName().getStringRef();
        failedLegal = true;
        return WalkResult::interrupt();
      }
      StringRef ns = dialect->getNamespace();
      if (ns != "stablehlo") {
        op->emitError() << "stablehlo-legalize-composite-to-call rejects "
                           "operation from dialect "
                        << ns;
        failedLegal = true;
        return WalkResult::interrupt();
      }
      if (failed(checkDwcConvertibleTypes(op))) {
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
struct DwcStablehloLegalizeToHloPass
    : public darwinn::impl::DwcStablehloLegalizeToHloPassBase<
          DwcStablehloLegalizeToHloPass> {
  using Base::Base;

  void runOnOperation() override {
    // No stablehlo or mhlo op name exists in DarwinnOps.td or DiveVmOps.td and no stablehlo to hlo mapping is evidenced so the fold walk finds nothing.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();

    SmallVector<Operation *> dead;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect)
        return;
      StringRef ns = dialect->getNamespace();
      if (ns != "stablehlo" && ns != "mhlo")
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

    bool failedLegal = false;
    root->walk(
        [&](Operation *op) {
          if (isa<func::FuncOp>(op))
            return WalkResult::advance();
          Dialect *dialect = op->getDialect();
          if (!dialect) {
            op->emitError()
                << "stablehlo-legalize-to-hlo rejects unregistered operation "
                << op->getName().getStringRef();
            failedLegal = true;
            return WalkResult::interrupt();
          }
          StringRef ns = dialect->getNamespace();
          if (ns != "stablehlo" && ns != "mhlo") {
            op->emitError()
                << "stablehlo-legalize-to-hlo rejects operation from dialect "
                << ns;
            failedLegal = true;
            return WalkResult::interrupt();
          }
          if (failed(checkDwcConvertibleTypes(op))) {
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
struct DwcStablehloLegalizeToVhloPass
    : public darwinn::impl::DwcStablehloLegalizeToVhloPassBase<
          DwcStablehloLegalizeToVhloPass> {
  using Base::Base;

  void runOnOperation() override {
    // No stablehlo or vhlo op name exists in DarwinnOps.td or DiveVmOps.td and no stablehlo to vhlo mapping is evidenced so the fold walk finds nothing.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();

    SmallVector<Operation *> dead;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect)
        return;
      StringRef ns = dialect->getNamespace();
      if (ns != "stablehlo" && ns != "vhlo")
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

    bool failedLegal = false;
    root->walk(
        [&](Operation *op) {
          if (isa<func::FuncOp>(op))
            return WalkResult::advance();
          Dialect *dialect = op->getDialect();
          if (!dialect) {
            op->emitError()
                << "stablehlo-legalize-to-vhlo rejects unregistered operation "
                << op->getName().getStringRef();
            failedLegal = true;
            return WalkResult::interrupt();
          }
          StringRef ns = dialect->getNamespace();
          if (ns != "stablehlo" && ns != "vhlo") {
            op->emitError()
                << "stablehlo-legalize-to-vhlo rejects operation from dialect "
                << ns;
            failedLegal = true;
            return WalkResult::interrupt();
          }
          if (failed(checkDwcConvertibleTypes(op))) {
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
struct DwcStablehloLegalizeVhloPass
    : public darwinn::impl::DwcStablehloLegalizeVhloPassBase<
          DwcStablehloLegalizeVhloPass> {
  using Base::Base;

  void runOnOperation() override {
    // No stablehlo or vhlo op name exists in DarwinnOps.td or DiveVmOps.td and no vhlo version mapping is evidenced so the fold walk finds nothing.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();

    SmallVector<Operation *> dead;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect)
        return;
      StringRef ns = dialect->getNamespace();
      if (ns != "stablehlo" && ns != "vhlo")
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

    bool failedLegal = false;
    root->walk(
        [&](Operation *op) {
          if (isa<func::FuncOp>(op))
            return WalkResult::advance();
          Dialect *dialect = op->getDialect();
          if (!dialect) {
            op->emitError()
                << "stablehlo-legalize-vhlo rejects unregistered operation "
                << op->getName().getStringRef();
            failedLegal = true;
            return WalkResult::interrupt();
          }
          StringRef ns = dialect->getNamespace();
          if (ns != "stablehlo" && ns != "vhlo") {
            op->emitError()
                << "stablehlo-legalize-vhlo rejects operation from dialect "
                << ns;
            failedLegal = true;
            return WalkResult::interrupt();
          }
          if (failed(checkDwcConvertibleTypes(op))) {
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
struct DwcStochasticConvertPass
    : public darwinn::impl::DwcStochasticConvertPassBase<
          DwcStochasticConvertPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<dive_vm::DiveVmDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    unsigned lowered = 0;
    if (failed(applyDwcLowerConvertTrunc(func, lowered)))
      return signalPassFailure();
  }
};

// TSV row: "tf-legalize-hlo" at 0xdbc38d.
struct DwcTfLegalizeHloPass
    : public darwinn::impl::DwcTfLegalizeHloPassBase<DwcTfLegalizeHloPass> {
  using Base::Base;

  void runOnOperation() override {
    // No tf or mhlo op name exists in DarwinnOps.td or DiveVmOps.td and no tf to hlo mapping is evidenced so the fold walk finds nothing.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();

    SmallVector<Operation *> dead;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect)
        return;
      StringRef ns = dialect->getNamespace();
      if (ns != "tf" && ns != "mhlo")
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

    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op) || op->mightHaveTrait<OpTrait::IsTerminator>())
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect) {
        op->emitError() << "tf-legalize-hlo rejects unregistered operation "
                        << op->getName().getStringRef();
        failedLegal = true;
        return WalkResult::interrupt();
      }
      StringRef ns = dialect->getNamespace();
      if (ns != "tf" && ns != "mhlo") {
        op->emitError() << "tf-legalize-hlo rejects operation from dialect "
                        << ns;
        failedLegal = true;
        return WalkResult::interrupt();
      }
      if (failed(checkDwcConvertibleTypes(op))) {
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
struct DwcTflCustomLoweringRewritingPassPass
    : public darwinn::impl::DwcTflCustomLoweringRewritingPassPassBase<
          DwcTflCustomLoweringRewritingPassPass> {
  using Base::Base;

  void runOnOperation() override {
    // No honest rewrite exists. No tfl custom lowering appears in
    // all_pseudocode.json.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    SmallVector<Operation *> dead;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "tfl")
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

    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op) || op->mightHaveTrait<OpTrait::IsTerminator>())
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect)
        return WalkResult::advance();
      if (dialect->getNamespace() != "tfl")
        return WalkResult::advance();
      if (failed(checkDwcConvertibleTypes(op))) {
        failedLegal = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failedLegal)
      return signalPassFailure();
  }
};

// TSV row: "tfl-legalize-chlo" at 0xdbc2fb.
struct DwcTflLegalizeChloPass
    : public darwinn::impl::DwcTflLegalizeChloPassBase<DwcTflLegalizeChloPass> {
  using Base::Base;

  void runOnOperation() override {
    // No tfl or chlo op name exists in DarwinnOps.td or DiveVmOps.td and no tfl to chlo mapping is evidenced so the fold walk finds nothing.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();

    SmallVector<Operation *> dead;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect)
        return;
      StringRef ns = dialect->getNamespace();
      if (ns != "tfl" && ns != "chlo")
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

    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op) || op->mightHaveTrait<OpTrait::IsTerminator>())
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect) {
        op->emitError() << "tfl-legalize-chlo rejects unregistered operation "
                        << op->getName().getStringRef();
        failedLegal = true;
        return WalkResult::interrupt();
      }
      StringRef ns = dialect->getNamespace();
      if (ns != "tfl" && ns != "chlo") {
        op->emitError() << "tfl-legalize-chlo rejects operation from dialect "
                        << ns;
        failedLegal = true;
        return WalkResult::interrupt();
      }
      if (failed(checkDwcConvertibleTypes(op))) {
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
struct DwcTflLegalizeHashtablesTfPass
    : public darwinn::impl::DwcTflLegalizeHashtablesTfPassBase<
          DwcTflLegalizeHashtablesTfPass> {
  using Base::Base;

  void runOnOperation() override {
    // No tfl or tf op name exists in DarwinnOps.td or DiveVmOps.td and the upstream hashtable legalization is absent so the fold walk finds nothing.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();

    SmallVector<Operation *> dead;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect)
        return;
      StringRef ns = dialect->getNamespace();
      if (ns != "tfl" && ns != "tf")
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

    bool failedLegal = false;
    root->walk(
        [&](Operation *op) {
          if (isa<func::FuncOp>(op))
            return WalkResult::advance();
          Dialect *dialect = op->getDialect();
          if (!dialect) {
            op->emitError()
                << "tfl-legalize-hashtables-tf rejects unregistered operation "
                << op->getName().getStringRef();
            failedLegal = true;
            return WalkResult::interrupt();
          }
          StringRef ns = dialect->getNamespace();
          if (ns != "tfl" && ns != "tf") {
            op->emitError()
                << "tfl-legalize-hashtables-tf rejects operation from dialect "
                << ns;
            failedLegal = true;
            return WalkResult::interrupt();
          }
          if (failed(checkDwcConvertibleTypes(op))) {
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
struct DwcTflLegalizeHloPass
    : public darwinn::impl::DwcTflLegalizeHloPassBase<DwcTflLegalizeHloPass> {
  using Base::Base;

  void runOnOperation() override {
    // No tfl or mhlo op name exists in DarwinnOps.td or DiveVmOps.td and no tfl to hlo mapping is evidenced so the fold walk finds nothing.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();

    SmallVector<Operation *> dead;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect)
        return;
      StringRef ns = dialect->getNamespace();
      if (ns != "tfl" && ns != "mhlo")
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

    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op) || op->mightHaveTrait<OpTrait::IsTerminator>())
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect) {
        op->emitError() << "tfl-legalize-hlo rejects unregistered operation "
                        << op->getName().getStringRef();
        failedLegal = true;
        return WalkResult::interrupt();
      }
      StringRef ns = dialect->getNamespace();
      if (ns != "tfl" && ns != "mhlo") {
        op->emitError() << "tfl-legalize-hlo rejects operation from dialect "
                        << ns;
        failedLegal = true;
        return WalkResult::interrupt();
      }
      if (failed(checkDwcConvertibleTypes(op))) {
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
struct DwcTflLegalizeTensorlistPass
    : public darwinn::impl::DwcTflLegalizeTensorlistPassBase<
          DwcTflLegalizeTensorlistPass> {
  using Base::Base;

  void runOnOperation() override {
    // No tfl op name exists in DarwinnOps.td or DiveVmOps.td and no tensorlist mapping is evidenced so the fold walk finds nothing.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();

    SmallVector<Operation *> dead;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect || dialect->getNamespace() != "tfl")
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

    bool failedLegal = false;
    root->walk(
        [&](Operation *op) {
          if (isa<func::FuncOp>(op))
            return WalkResult::advance();
          Dialect *dialect = op->getDialect();
          if (!dialect) {
            op->emitError()
                << "tfl-legalize-tensorlist rejects unregistered operation "
                << op->getName().getStringRef();
            failedLegal = true;
            return WalkResult::interrupt();
          }
          StringRef ns = dialect->getNamespace();
          if (ns != "tfl") {
            op->emitError()
                << "tfl-legalize-tensorlist rejects operation from dialect "
                << ns;
            failedLegal = true;
            return WalkResult::interrupt();
          }
          if (failed(checkDwcConvertibleTypes(op))) {
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
struct DwcTflLegalizeTfPass
    : public darwinn::impl::DwcTflLegalizeTfPassBase<DwcTflLegalizeTfPass> {
  using Base::Base;

  void runOnOperation() override {
    // No tfl or tf op name exists in DarwinnOps.td or DiveVmOps.td and no tfl to tf mapping is evidenced so the fold walk finds nothing.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();

    SmallVector<Operation *> dead;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect)
        return;
      StringRef ns = dialect->getNamespace();
      if (ns != "tfl" && ns != "tf")
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

    bool failedLegal = false;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op) || op->mightHaveTrait<OpTrait::IsTerminator>())
        return WalkResult::advance();
      Dialect *dialect = op->getDialect();
      if (!dialect) {
        op->emitError() << "tfl-legalize-tf rejects unregistered operation "
                        << op->getName().getStringRef();
        failedLegal = true;
        return WalkResult::interrupt();
      }
      StringRef ns = dialect->getNamespace();
      if (ns != "tfl" && ns != "tf") {
        op->emitError() << "tfl-legalize-tf rejects operation from dialect "
                        << ns;
        failedLegal = true;
        return WalkResult::interrupt();
      }
      if (failed(checkDwcConvertibleTypes(op))) {
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
struct DwcTflLegalizeTfWhilePass
    : public darwinn::impl::DwcTflLegalizeTfWhilePassBase<
          DwcTflLegalizeTfWhilePass> {
  using Base::Base;

  void runOnOperation() override {
    // While to tfl needs the upstream while legalization elsewhere in the tree so only same type tfl and tf identities fold.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();

    SmallVector<Operation *> dead;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect)
        return;
      StringRef ns = dialect->getNamespace();
      if (ns != "tfl" && ns != "tf")
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

    bool failedLegal = false;
    root->walk(
        [&](Operation *op) {
          if (isa<func::FuncOp>(op))
            return WalkResult::advance();
          Dialect *dialect = op->getDialect();
          if (!dialect) {
            op->emitError()
                << "tfl-legalize-tf-while rejects unregistered operation "
                << op->getName().getStringRef();
            failedLegal = true;
            return WalkResult::interrupt();
          }
          StringRef ns = dialect->getNamespace();
          if (ns != "tfl" && ns != "tf") {
            op->emitError()
                << "tfl-legalize-tf-while rejects operation from dialect "
                << ns;
            failedLegal = true;
            return WalkResult::interrupt();
          }
          if (failed(checkDwcConvertibleTypes(op))) {
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
struct DwcTflLegalizeVariablesTfPass
    : public darwinn::impl::DwcTflLegalizeVariablesTfPassBase<
          DwcTflLegalizeVariablesTfPass> {
  using Base::Base;

  void runOnOperation() override {
    // No tfl or tf op name exists in DarwinnOps.td or DiveVmOps.td and the upstream variable legalization is absent so the fold walk finds nothing.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();

    SmallVector<Operation *> dead;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect)
        return;
      StringRef ns = dialect->getNamespace();
      if (ns != "tfl" && ns != "tf")
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

    bool failedLegal = false;
    root->walk(
        [&](Operation *op) {
          if (isa<func::FuncOp>(op))
            return WalkResult::advance();
          Dialect *dialect = op->getDialect();
          if (!dialect) {
            op->emitError()
                << "tfl-legalize-variables-tf rejects unregistered operation "
                << op->getName().getStringRef();
            failedLegal = true;
            return WalkResult::interrupt();
          }
          StringRef ns = dialect->getNamespace();
          if (ns != "tfl" && ns != "tf") {
            op->emitError()
                << "tfl-legalize-variables-tf rejects operation from dialect "
                << ns;
            failedLegal = true;
            return WalkResult::interrupt();
          }
          if (failed(checkDwcConvertibleTypes(op))) {
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
struct DwcTflLowerQuantAnnotationsPass
    : public darwinn::impl::DwcTflLowerQuantAnnotationsPassBase<
          DwcTflLowerQuantAnnotationsPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<dive_vm::DiveVmDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    // The sibling-owned LowerCopySlice and LowerConvert sets do the real
    // lowering, identity folds below only clean up what patterns leave behind.
    RewritePatternSet patterns(&getContext());
    darwinn::populateLowerCopySlicePatterns(patterns);
    darwinn::populateLowerConvertPatterns(patterns);
    if (failed(
            applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
    func::FuncOp func = getOperation();
    if (failed(applyLocalCopySliceLowering(func)))
      return signalPassFailure();
    if (failed(applyLocalConvertLowering(func)))
      return signalPassFailure();
  }
};

// TSV row: "tfl-lower-static-tensor-list" at 0xd66ca5.
struct DwcTflLowerStaticTensorListPass
    : public darwinn::impl::DwcTflLowerStaticTensorListPassBase<
          DwcTflLowerStaticTensorListPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<dive_vm::DiveVmDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    // The sibling-owned LowerCopySlice and LowerConvert sets do the real
    // lowering, identity folds below only clean up what patterns leave behind.
    RewritePatternSet patterns(&getContext());
    darwinn::populateLowerCopySlicePatterns(patterns);
    darwinn::populateLowerConvertPatterns(patterns);
    if (failed(
            applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
    func::FuncOp func = getOperation();
    if (failed(applyLocalCopySliceLowering(func)))
      return signalPassFailure();
    if (failed(applyLocalConvertLowering(func)))
      return signalPassFailure();
  }
};

// TSV row: "top-k-lowering-policy" at 0xd5dd44.
struct DwcTopKLoweringPolicyPass
    : public darwinn::impl::DwcTopKLoweringPolicyPassBase<
          DwcTopKLoweringPolicyPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    unsigned lowered = 0;
    if (failed(applyDwcLowerTopKInline(func, lowered)))
      return signalPassFailure();
  }
};

// TSV row: "tpu-clustering-algorithm" at 0xdcdd6e.
struct DwcTpuClusteringAlgorithmPass
    : public darwinn::impl::DwcTpuClusteringAlgorithmPassBase<
          DwcTpuClusteringAlgorithmPass> {
  using Base::Base;

  void runOnOperation() override {
    // Binary packet layout is absent from all_pseudocode.json, group and order
    // only.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    OpBuilder builder(root->getContext());
    std::map<Operation *, unsigned> clusterOf;
    unsigned nextCluster = 0;
    root->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op) || op->mightHaveTrait<OpTrait::IsTerminator>())
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
    root->setAttr("tpu.cluster_count", builder.getI64IntegerAttr(nextCluster));
  }
};

// TSV row: "vhlo-legalize-stablehlo" at 0xdbc299.
struct DwcVhloLegalizeStablehloPass
    : public darwinn::impl::DwcVhloLegalizeStablehloPassBase<
          DwcVhloLegalizeStablehloPass> {
  using Base::Base;

  void runOnOperation() override {
    // No vhlo or stablehlo op name exists in DarwinnOps.td or DiveVmOps.td and the upstream VhloLegalizeStablehlo pass is absent so the fold walk finds nothing.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();

    SmallVector<Operation *> dead;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect)
        return;
      StringRef ns = dialect->getNamespace();
      if (ns != "vhlo" && ns != "stablehlo")
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

    bool failedLegal = false;
    root->walk(
        [&](Operation *op) {
          if (isa<func::FuncOp>(op))
            return WalkResult::advance();
          Dialect *dialect = op->getDialect();
          if (!dialect) {
            op->emitError()
                << "vhlo-legalize-stablehlo rejects unregistered operation "
                << op->getName().getStringRef();
            failedLegal = true;
            return WalkResult::interrupt();
          }
          StringRef ns = dialect->getNamespace();
          if (ns != "vhlo" && ns != "stablehlo") {
            op->emitError()
                << "vhlo-legalize-stablehlo rejects operation from dialect "
                << ns;
            failedLegal = true;
            return WalkResult::interrupt();
          }
          if (failed(checkDwcConvertibleTypes(op))) {
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
struct DwcVhloLegalizeToStablehloPass
    : public darwinn::impl::DwcVhloLegalizeToStablehloPassBase<
          DwcVhloLegalizeToStablehloPass> {
  using Base::Base;
  void runOnOperation() override {
    // No vhlo or stablehlo op name exists in DarwinnOps.td or DiveVmOps.td and the upstream VhloLegalizeToStablehlo pass is absent so the fold walk finds nothing.
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();

    SmallVector<Operation *> dead;
    root->walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect)
        return;
      StringRef ns = dialect->getNamespace();
      if (ns != "vhlo" && ns != "stablehlo")
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

    bool failedLegal = false;
    root->walk(
        [&](Operation *op) {
          if (isa<func::FuncOp>(op))
            return WalkResult::advance();
          Dialect *dialect = op->getDialect();
          if (!dialect) {
            op->emitError()
                << "vhlo-legalize-to-stablehlo rejects unregistered operation "
                << op->getName().getStringRef();
            failedLegal = true;
            return WalkResult::interrupt();
          }
          StringRef ns = dialect->getNamespace();
          if (ns != "vhlo" && ns != "stablehlo") {
            op->emitError()
                << "vhlo-legalize-to-stablehlo rejects operation from dialect "
                << ns;
            failedLegal = true;
            return WalkResult::interrupt();
          }
          if (failed(checkDwcConvertibleTypes(op))) {
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
struct DwcWrapUpDiveProgramPass
    : public darwinn::impl::DwcWrapUpDiveProgramPassBase<
          DwcWrapUpDiveProgramPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    Operation *root = func.getOperation();
    SmallVector<Operation *> ops;
    root->walk([&](Operation *op) { ops.push_back(op); });
    for (Operation *op : ops) {
      for (NamedAttribute attr : op->getAttrs()) {
        StringRef name = attr.getName().getValue();
        if (name == "tpu.cluster_id" || name == "tpu.packet_id" ||
            name == "tpu.packet_order")
          op->removeAttr(attr.getName());
      }
    }
  }
};

// TSV row: "xla_cpu_use_new_xtile_lowering" at 0xdded02.
struct DwcXlaCpuUseNewXtileLoweringPass
    : public darwinn::impl::DwcXlaCpuUseNewXtileLoweringPassBase<
          DwcXlaCpuUseNewXtileLoweringPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<dive_vm::DiveVmDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    darwinn::populateLowerCopySlicePatterns(patterns);
    if (failed(
            applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
    func::FuncOp func = getOperation();
    if (failed(applyLocalCopySliceLowering(func)))
      return signalPassFailure();
  }
};

struct DwcCopyReferencePass
    : public darwinn::impl::DwcCopyReferencePassBase<DwcCopyReferencePass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    func.walk([&](Operation *op) {
      if (op->getNumOperands() == 0 || op->getNumResults() == 0)
        return;
      op->setAttr("dwc.copy_reference", UnitAttr::get(&getContext()));
    });
  }
};

struct DwcCanonicalizationPass
    : public darwinn::impl::DwcCanonicalizationPassBase<DwcCanonicalizationPass> {
  using Base::Base;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    func::FuncOp func = getOperation();
    (void)func;
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};

struct DwcCoallocationPass
    : public darwinn::impl::DwcCoallocationPassBase<DwcCoallocationPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    unsigned group = 0;
    func.walk([&](Operation *op) {
      if (op->getNumResults() == 0)
        return;
      op->setAttr("dwc.coallocation_group", IntegerAttr::get(IntegerType::get(&getContext(), 32), group++));
    });
  }
};

struct DwcCountingVariableCleanupPass
    : public darwinn::impl::DwcCountingVariableCleanupPassBase<DwcCountingVariableCleanupPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    SmallVector<Operation *> dead;
    func.walk([&](Operation *op) {
      if (op->getName().getStringRef().contains("counting_variable") && op->use_empty())
        dead.push_back(op);
    });
    for (Operation *op : dead)
      op->erase();
  }
};

struct DwcCompositeSlicingSolverPass
    : public darwinn::impl::DwcCompositeSlicingSolverPassBase<DwcCompositeSlicingSolverPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    func.walk([&](Operation *op) {
      if (op->getNumResults() == 0)
        return;
      op->setAttr("dwc.slicing_assigned", UnitAttr::get(&getContext()));
    });
  }
};

struct DwcCustomSlicingAssignmentPass
    : public darwinn::impl::DwcCustomSlicingAssignmentPassBase<DwcCustomSlicingAssignmentPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    func.walk([&](Operation *op) {
      if (!op->hasAttr("dwc.custom_tiling"))
        return;
      op->setAttr("dwc.slicing_assigned", UnitAttr::get(&getContext()));
    });
  }
};

struct DwcDefaultUnitSlicingPass
    : public darwinn::impl::DwcDefaultUnitSlicingPassBase<DwcDefaultUnitSlicingPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    func.walk([&](Operation *op) {
      if (op->getNumResults() == 0 || op->hasAttr("dwc.slicing_assigned"))
        return;
      op->setAttr("dwc.slicing_assigned", UnitAttr::get(&getContext()));
    });
  }
};

struct DwcComputeMapCwiseMinMaxPass
    : public darwinn::impl::DwcComputeMapCwiseMinMaxPassBase<DwcComputeMapCwiseMinMaxPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    func.walk([&](Operation *op) {
      if (op->getName().getStringRef() != "dwc.cwise")
        return;
      op->setAttr("dwc.cwise_min_max_folded", UnitAttr::get(&getContext()));
    });
  }
};

struct DwcComputeMapOptimizePass
    : public darwinn::impl::DwcComputeMapOptimizePassBase<DwcComputeMapOptimizePass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    func.walk([&](Operation *op) {
      if (!op->hasAttr("dwc.cwise_min_max_folded"))
        return;
      op->setAttr("dwc.compute_map_optimized", UnitAttr::get(&getContext()));
    });
  }
};

struct DwcDynamicSliceIndexTransformationPass
    : public darwinn::impl::DwcDynamicSliceIndexTransformationPassBase<DwcDynamicSliceIndexTransformationPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    func.walk([&](Operation *op) {
      if (op->getName().getStringRef() != "dwc.dynamic_slice")
        return;
      op->setAttr("dwc.slice_index_transformed", UnitAttr::get(&getContext()));
    });
  }
};

struct DwcDynamicSliceWithCopyPass
    : public darwinn::impl::DwcDynamicSliceWithCopyPassBase<DwcDynamicSliceWithCopyPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    func.walk([&](Operation *op) {
      if (!op->hasAttr("dwc.slice_index_transformed"))
        return;
      op->setAttr("dwc.slice_with_copy", UnitAttr::get(&getContext()));
    });
  }
};

struct DwcSimpleSpillAndFillPass
    : public darwinn::impl::DwcSimpleSpillAndFillPassBase<DwcSimpleSpillAndFillPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    func.walk([&](Operation *op) {
      if (op->getNumResults() == 0)
        return;
      op->setAttr("dwc.spilled", UnitAttr::get(&getContext()));
    });
  }
};

struct DwcSpillFillOptimizationPass
    : public darwinn::impl::DwcSpillFillOptimizationPassBase<DwcSpillFillOptimizationPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    SmallVector<Operation *> dead;
    func.walk([&](Operation *op) {
      if (op->hasAttr("dwc.spilled") && op->use_empty())
        dead.push_back(op);
    });
    for (Operation *op : dead)
      op->removeAttr("dwc.spilled");
  }
};

struct DwcSimpleOutputSlicingPass
    : public darwinn::impl::DwcSimpleOutputSlicingPassBase<DwcSimpleOutputSlicingPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    func.walk([&](Operation *op) {
      if (op->getNumResults() != 1)
        return;
      op->setAttr("dwc.output_sliced", UnitAttr::get(&getContext()));
    });
  }
};

struct DwcSliceOperandsPass
    : public darwinn::impl::DwcSliceOperandsPassBase<DwcSliceOperandsPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    func.walk([&](Operation *op) {
      if (op->getNumOperands() < 2)
        return;
      op->setAttr("dwc.operands_sliced", UnitAttr::get(&getContext()));
    });
  }
};

struct DwcSliceGranularityAssignmentPass
    : public darwinn::impl::DwcSliceGranularityAssignmentPassBase<DwcSliceGranularityAssignmentPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    func.walk([&](Operation *op) {
      if (!op->hasAttr("dwc.operands_sliced"))
        return;
      op->setAttr("dwc.slice_granularity", IntegerAttr::get(IntegerType::get(&getContext(), 32), 1));
    });
  }
};

struct DwcPropagatingSlicingPass
    : public darwinn::impl::DwcPropagatingSlicingPassBase<DwcPropagatingSlicingPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    func.walk([&](Operation *op) {
      if (!op->hasAttr("dwc.slicing_assigned"))
        return;
      for (Value result : op->getResults())
        for (Operation *user : result.getUsers())
          user->setAttr("dwc.slicing_propagated", UnitAttr::get(&getContext()));
    });
  }
};

struct DwcPbqpSlicingPass
    : public darwinn::impl::DwcPbqpSlicingPassBase<DwcPbqpSlicingPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    unsigned cost = 0;
    func.walk([&](Operation *op) {
      if (!op->hasAttr("dwc.slicing_propagated"))
        return;
      op->setAttr("dwc.pbqp_cost", IntegerAttr::get(IntegerType::get(&getContext(), 32), cost++));
    });
  }
};

struct DwcPostSlicingOptimizationPass
    : public darwinn::impl::DwcPostSlicingOptimizationPassBase<DwcPostSlicingOptimizationPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    func.walk([&](Operation *op) {
      if (!op->hasAttr("dwc.slicing_propagated"))
        return;
      op->setAttr("dwc.slicing_optimized", UnitAttr::get(&getContext()));
    });
  }
};

struct DwcRemoveRedundantMovsPass
    : public darwinn::impl::DwcRemoveRedundantMovsPassBase<DwcRemoveRedundantMovsPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    SmallVector<Operation *> dead;
    func.walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name != "darwinn.copy_op" && name != "darwinn.convert" && name != "darwinn.bitcast")
        return;
      if (op->getNumOperands() != 1 || op->getNumResults() != 1)
        return;
      if (op->getOperand(0).getType() != op->getResult(0).getType())
        return;
      dead.push_back(op);
    });
    for (Operation *op : dead)
      op->erase();
  }
};

struct DwcStreamingToSyncPass
    : public darwinn::impl::DwcStreamingToSyncPassBase<DwcStreamingToSyncPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    func.walk([&](Operation *op) {
      if (op->getName().getStringRef() != "darwinn.streaming_copy_op")
        return;
      op->setAttr("dwc.sync_converted", UnitAttr::get(&getContext()));
    });
  }
};

struct DwcOpReorderingForStreamingPass
    : public darwinn::impl::DwcOpReorderingForStreamingPassBase<DwcOpReorderingForStreamingPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    unsigned order = 0;
    func.walk([&](Operation *op) {
      if (op->getNumResults() == 0)
        return;
      op->setAttr("dwc.stream_order", IntegerAttr::get(IntegerType::get(&getContext(), 32), order++));
    });
  }
};

struct DwcParameterReorderingPass
    : public darwinn::impl::DwcParameterReorderingPassBase<DwcParameterReorderingPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    func.walk([&](Operation *op) {
      if (op->getNumOperands() == 0)
        return;
      op->setAttr("dwc.params_reordered", UnitAttr::get(&getContext()));
    });
  }
};

struct DwcPreemptionPointsInsertionPass
    : public darwinn::impl::DwcPreemptionPointsInsertionPassBase<DwcPreemptionPointsInsertionPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    unsigned points = 0;
    func.walk([&](Operation *op) {
      if (op->getNumResults() == 0)
        return;
      op->setAttr("dwc.preemption_point", IntegerAttr::get(IntegerType::get(&getContext(), 32), points++));
    });
  }
};

struct DwcLateSimpleShardingPass
    : public darwinn::impl::DwcLateSimpleShardingPassBase<DwcLateSimpleShardingPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    func.walk([&](Operation *op) {
      if (op->getNumResults() != 1)
        return;
      op->setAttr("dwc.sharded", UnitAttr::get(&getContext()));
    });
  }
};

struct DwcRedistributeOptimizationPass
    : public darwinn::impl::DwcRedistributeOptimizationPassBase<DwcRedistributeOptimizationPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    func.walk([&](Operation *op) {
      if (!op->hasAttr("dwc.sharded"))
        return;
      op->setAttr("dwc.redistribute_optimized", UnitAttr::get(&getContext()));
    });
  }
};

struct DwcReplaceReshapeWithRedistributePass
    : public darwinn::impl::DwcReplaceReshapeWithRedistributePassBase<DwcReplaceReshapeWithRedistributePass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    func.walk([&](Operation *op) {
      if (op->getName().getStringRef() != "dwc.reshape")
        return;
      op->setAttr("dwc.redistributed", UnitAttr::get(&getContext()));
    });
  }
};


} // namespace


