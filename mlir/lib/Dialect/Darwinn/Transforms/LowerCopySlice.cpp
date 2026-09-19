//===- LowerCopySlice.cpp - Lower copy, slice and gather to dive_vm ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace darwinn {
void populateLowerCopySlicePatterns(RewritePatternSet &patterns);
} // namespace darwinn
} // namespace mlir

namespace mlir {
namespace darwinn {
#define GEN_PASS_DEF_DWCLOWERCOPYSLICEPASS
#include "DwcPasses.h.inc"
} // namespace darwinn
} // namespace mlir

using namespace mlir;

namespace {

Operation *makeVmOp(PatternRewriter &rewriter, Location loc, StringRef name,
                    ValueRange operands, TypeRange results,
                    ArrayRef<NamedAttribute> attrs = {}) {
  OperationState state(loc, name, operands, results, attrs);
  return rewriter.create(state);
}

bool hasSameElementType(Type a, Type b) {
  auto tensorA = dyn_cast<TensorType>(a);
  auto tensorB = dyn_cast<TensorType>(b);
  if (!tensorA || !tensorB)
    return true;
  return tensorA.getElementType() == tensorB.getElementType();
}

bool isSingleCopy(Type src, Type dst) {
  auto srcRanked = dyn_cast<RankedTensorType>(src);
  auto dstRanked = dyn_cast<RankedTensorType>(dst);
  if (!srcRanked || !dstRanked || !srcRanked.hasStaticShape() ||
      !dstRanked.hasStaticShape())
    return false;
  return srcRanked.getNumElements() == dstRanked.getNumElements();
}

struct CopyOpLowering : public RewritePattern {
  CopyOpLowering(MLIRContext *ctx)
      : RewritePattern("darwinn.copy_op", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1 || op->getNumOperands() < 1)
      return failure();
    Value src = op->getOperand(0);
    Type dstTy = op->getResult(0).getType();
    if (!hasSameElementType(src.getType(), dstTy))
      return failure();
    if (auto srcRanked = dyn_cast<RankedTensorType>(src.getType()))
      if (auto dstRanked = dyn_cast<RankedTensorType>(dstTy))
        if (srcRanked.hasStaticShape() && dstRanked.hasStaticShape() &&
            srcRanked.getNumElements() != dstRanked.getNumElements())
          return failure();
    SmallVector<NamedAttribute> attrs;
    for (auto attr : op->getAttrs())
      attrs.push_back(attr);
    Operation *copy =
        makeVmOp(rewriter, op->getLoc(), "dive_vm.copy", src, dstTy, attrs);
    rewriter.replaceOp(op, copy->getResults());
    return success();
  }
};

struct DynamicSliceLowering : public RewritePattern {
  DynamicSliceLowering(StringRef rootName, bool isUpdate, MLIRContext *ctx)
      : RewritePattern(rootName, 1, ctx), isUpdate(isUpdate) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1 || op->getNumOperands() < 2)
      return failure();
    Location loc = op->getLoc();
    Value src = op->getOperand(0);
    Type dstTy = op->getResult(0).getType();
    if (!hasSameElementType(src.getType(), dstTy))
      return failure();

    Operation *base = makeVmOp(rewriter, loc, "dive_vm.address_of_activation",
                               src, src.getType());

    SmallVector<Value> sliceOperands;
    sliceOperands.push_back(base->getResult(0));
    sliceOperands.append(op->operand_begin() + 1, op->operand_end());

    SmallVector<NamedAttribute> attrs;
    for (auto attr : op->getAttrs())
      attrs.push_back(attr);
    attrs.push_back(
        rewriter.getNamedAttr("offset_generator",
                              rewriter.getStringAttr("slice_offset_generator")));
    attrs.push_back(rewriter.getNamedAttr(
        "single_copy", rewriter.getBoolAttr(isSingleCopy(src.getType(), dstTy))));

    StringRef vmName =
        isUpdate ? "dive_vm.insert_slice" : "dive_vm.extract_slice";
    Operation *slice = makeVmOp(rewriter, loc, vmName, sliceOperands, dstTy,
                                attrs);
    rewriter.replaceOp(op, slice->getResults());
    return success();
  }

private:
  bool isUpdate;
};

struct GatherLowering : public RewritePattern {
  GatherLowering(StringRef rootName, MLIRContext *ctx)
      : RewritePattern(rootName, 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1 || op->getNumOperands() < 2)
      return failure();
    Value indices = op->getOperand(1);
    if (auto idxTy = dyn_cast<RankedTensorType>(indices.getType()))
      if (!idxTy.getElementType().isIntOrIndex())
        return failure();

    SmallVector<NamedAttribute> attrs;
    for (auto attr : op->getAttrs())
      attrs.push_back(attr);
    attrs.push_back(rewriter.getNamedAttr("oob_zero_fill",
                                          rewriter.getUnitAttr()));

    Operation *gather =
        makeVmOp(rewriter, op->getLoc(), "dive_vm.gather", op->getOperands(),
                 op->getResultTypes(), attrs);
    rewriter.replaceOp(op, gather->getResults());
    return success();
  }
};

struct DwcLowerCopySlicePass
    : public darwinn::impl::DwcLowerCopySlicePassBase<DwcLowerCopySlicePass> {
  using Base::Base;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    darwinn::populateLowerCopySlicePatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(),
                                            std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

void mlir::darwinn::populateLowerCopySlicePatterns(
    RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<CopyOpLowering>(ctx);
  patterns.add<DynamicSliceLowering>("darwinn.dynamic_slice",
                                     /*isUpdate=*/false, ctx);
  patterns.add<DynamicSliceLowering>("darwinn.dynamic_slice_with_copy",
                                     /*isUpdate=*/false, ctx);
  patterns.add<DynamicSliceLowering>("darwinn.dynamic_update_slice",
                                     /*isUpdate=*/true, ctx);
  patterns.add<DynamicSliceLowering>(
      "darwinn.dynamic_update_slice_with_offsets", /*isUpdate=*/true, ctx);
  patterns.add<GatherLowering>("darwinn.gather", ctx);
  patterns.add<GatherLowering>("darwinn.gather_copy", ctx);
  patterns.add<GatherLowering>("darwinn.hib_gather", ctx);
}
