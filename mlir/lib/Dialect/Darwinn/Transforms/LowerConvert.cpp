//===- LowerConvert.cpp - Lower darwinn.convert/cast_in/cast_out --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lowers darwinn.convert, darwinn.cast_in, and darwinn.cast_out to the
// dive_vm elementwise convert operation.
//
// Kernel semantics come from the cast::Fallback disassembly recorded in
// rederivation_cast_topk.md. float-to-int truncates toward zero (fcvt rtz
// on every scalar and vector path, compiler-rt __fix* for f64), int-to-float
// rounds to nearest even under the default dynamic rounding mode, and narrow
// stores keep the low bits (sb/sh after a 64-bit convert). dive_vm.cast
// implements exactly this elementwise behavior, so the patterns only check
// shape and element-type legality and forward the operand.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Darwinn/IR/DarwinnOps.h"
#include "mlir/Dialect/DiveVm/IR/DiveVmOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {
Type elementOf(Type type) {
  if (auto shaped = dyn_cast<ShapedType>(type))
    return shaped.getElementType();
  return type;
}

bool isConvertibleElement(Type src, Type dst) {
  if (src == dst)
    return true;
  bool srcNumeric = isa<IntegerType, FloatType>(src);
  bool dstNumeric = isa<IntegerType, FloatType>(dst);
  return srcNumeric && dstNumeric;
}

bool sameElementwiseShape(Type src, Type dst) {
  auto srcRanked = dyn_cast<RankedTensorType>(src);
  auto dstRanked = dyn_cast<RankedTensorType>(dst);
  if (!srcRanked || !dstRanked)
    return true;
  return srcRanked.getShape() == dstRanked.getShape();
}

template <typename SrcOp>
LogicalResult lowerConvertLike(SrcOp op, PatternRewriter &rewriter) {
  if (op->getNumOperands() != 1 || op->getNumResults() != 1)
    return failure();
  Value input = op->getOperand(0);
  Type resultType = op->getResult(0).getType();
  if (!sameElementwiseShape(input.getType(), resultType))
    return failure();
  if (!isConvertibleElement(elementOf(input.getType()), elementOf(resultType)))
    return failure();
  if (input.getType() == resultType) {
    rewriter.replaceOp(op, input);
    return success();
  }
  rewriter.replaceOpWithNewOp<dive_vm::CastOp>(op, resultType, input);
  return success();
}

struct ConvertLowering : public OpRewritePattern<darwinn::ConvertOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(darwinn::ConvertOp op,
                                PatternRewriter &rewriter) const override {
    return lowerConvertLike(op, rewriter);
  }
};

struct CastInLowering : public OpRewritePattern<darwinn::CastInOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(darwinn::CastInOp op,
                                PatternRewriter &rewriter) const override {
    return lowerConvertLike(op, rewriter);
  }
};

struct CastOutLowering : public OpRewritePattern<darwinn::CastOutOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(darwinn::CastOutOp op,
                                PatternRewriter &rewriter) const override {
    return lowerConvertLike(op, rewriter);
  }
};
} // namespace

namespace mlir {
namespace darwinn {
void populateLowerConvertPatterns(RewritePatternSet &patterns) {
  patterns.add<ConvertLowering, CastInLowering, CastOutLowering>(
      patterns.getContext());
}
} // namespace darwinn
} // namespace mlir

namespace mlir {
namespace darwinn {
#define GEN_PASS_DECL
#define GEN_PASS_DEF_DWCLOWERCONVERTPASS
#include "DwcPasses.h.inc"
} // namespace darwinn
} // namespace mlir

namespace {
struct DwcLowerConvertPass
    : public mlir::darwinn::impl::DwcLowerConvertPassBase<DwcLowerConvertPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<dive_vm::DiveVmDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    mlir::darwinn::populateLowerConvertPatterns(patterns);
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};
} // namespace
