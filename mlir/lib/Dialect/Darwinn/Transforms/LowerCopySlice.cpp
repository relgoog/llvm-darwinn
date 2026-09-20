//===- LowerCopySlice.cpp - Lower copy, slice and gather to dive_vm ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Darwinn/IR/DwcOps.h"
#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/Darwinn/IR/DwcAttributes.h.inc"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace darwinn {
void populateLowerCopySlicePatterns(RewritePatternSet &patterns);
} // namespace darwinn
} // namespace mlir

using namespace mlir;
using namespace mlir::dwc;

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

struct CwiseLowering : public RewritePattern {
  CwiseLowering(MLIRContext *ctx)
      : RewritePattern("dwc.cwise", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1 || op->getNumOperands() != 2)
      return failure();
    auto opType = dyn_cast<CwiseOpTypeAttr>(op->getAttr("op_type"));
    if (!opType)
      return failure();
    bool isRelu = false;
    if (auto activation = dyn_cast<ActivationFunctionAttr>(op->getAttr("activation_function"))) {
      if (activation.getValue() == ActivationFunction::Relu)
        isRelu = true;
      else if (activation.getValue() != ActivationFunction::None)
        return failure();
    }
    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    Type dstTy = op->getResult(0).getType();
    bool isCompare = false;
    switch (opType.getValue()) {
    case CwiseOpType::Equal:
    case CwiseOpType::NotEqual:
    case CwiseOpType::Greater:
    case CwiseOpType::GreaterEqual:
    case CwiseOpType::Less:
    case CwiseOpType::LessEqual:
      isCompare = true;
      break;
    default:
      break;
    }
    if (!isCompare) {
      if (!hasSameElementType(lhs.getType(), dstTy) ||
          !hasSameElementType(rhs.getType(), dstTy))
        return failure();
    } else {
      if (!hasSameElementType(lhs.getType(), rhs.getType()))
        return failure();
      auto dstRanked = dyn_cast<RankedTensorType>(dstTy);
      if (!dstRanked || !dstRanked.hasStaticShape())
        return failure();
      auto dstElem = dstRanked.getElementType();
      auto i1 = IntegerType::get(op->getContext(), 1, IntegerType::Signless);
      if (dstElem != i1)
        return failure();
    }
    auto ranked = dyn_cast<RankedTensorType>(dstTy);
    if (!ranked || !ranked.hasStaticShape())
      return failure();
    Location loc = op->getLoc();
    Value empty = rewriter.create<tensor::EmptyOp>(loc, ranked.getShape(), ranked.getElementType());
    Operation *elem = nullptr;
    switch (opType.getValue()) {
    case CwiseOpType::Add:
      elem = rewriter.create<linalg::AddOp>(loc, dstTy, ValueRange{lhs, rhs}, ValueRange{empty});
      break;
    case CwiseOpType::Subtract:
      elem = rewriter.create<linalg::SubOp>(loc, dstTy, ValueRange{lhs, rhs}, ValueRange{empty});
      break;
    case CwiseOpType::Multiply:
      elem = rewriter.create<linalg::MulOp>(loc, dstTy, ValueRange{lhs, rhs}, ValueRange{empty});
      break;
    case CwiseOpType::Divide:
      elem = rewriter.create<linalg::DivOp>(loc, dstTy, ValueRange{lhs, rhs}, ValueRange{empty});
      break;
    case CwiseOpType::Maximum:
      elem = rewriter.create<linalg::MaxOp>(loc, dstTy, ValueRange{lhs, rhs}, ValueRange{empty});
      break;
    case CwiseOpType::Minimum:
      elem = rewriter.create<linalg::MinOp>(loc, dstTy, ValueRange{lhs, rhs}, ValueRange{empty});
      break;
    default:
      break;
    }
    if (!elem) {
      auto lhsTy = dyn_cast<RankedTensorType>(lhs.getType());
      auto dstRanked = dyn_cast<RankedTensorType>(dstTy);
      if (!lhsTy || !dstRanked || !lhsTy.hasStaticShape() || !dstRanked.hasStaticShape())
        return failure();
      if (lhsTy.getShape() != dstRanked.getShape())
        return failure();
      arith::CmpIPredicate intPred;
      arith::CmpFPredicate floatPred;
      bool isFloat = isa<FloatType>(lhsTy.getElementType());
      switch (opType.getValue()) {
      case CwiseOpType::Equal:
        intPred = arith::CmpIPredicate::eq;
        floatPred = arith::CmpFPredicate::OEQ;
        break;
      case CwiseOpType::NotEqual:
        intPred = arith::CmpIPredicate::ne;
        floatPred = arith::CmpFPredicate::UNE;
        break;
      case CwiseOpType::Greater:
        intPred = arith::CmpIPredicate::sgt;
        floatPred = arith::CmpFPredicate::OGT;
        break;
      case CwiseOpType::GreaterEqual:
        intPred = arith::CmpIPredicate::sge;
        floatPred = arith::CmpFPredicate::OGE;
        break;
      case CwiseOpType::Less:
        intPred = arith::CmpIPredicate::slt;
        floatPred = arith::CmpFPredicate::OLT;
        break;
      case CwiseOpType::LessEqual:
        intPred = arith::CmpIPredicate::sle;
        floatPred = arith::CmpFPredicate::OLE;
        break;
      default:
        return failure();
      }
      int64_t rank = lhsTy.getRank();
      SmallVector<AffineMap> maps(3, rewriter.getMultiDimIdentityMap(rank));
      SmallVector<utils::IteratorType> iters(rank, utils::IteratorType::parallel);
      auto generic = rewriter.create<linalg::GenericOp>(loc, TypeRange{dstTy}, ValueRange{lhs, rhs}, ValueRange{empty},
          maps, iters,
          [&](OpBuilder &nested, Location nloc, ValueRange args) {
            Value cmp = isFloat ? static_cast<Value>(nested.create<arith::CmpFOp>(nloc, floatPred, args[0], args[1]))
                                : static_cast<Value>(nested.create<arith::CmpIOp>(nloc, intPred, args[0], args[1]));
            nested.create<linalg::YieldOp>(nloc, cmp);
          });
      elem = generic.getOperation();
    }
    Value result = elem->getResult(0);
    if (isRelu) {
      Value zeroBuf = rewriter.create<tensor::EmptyOp>(loc, ranked.getShape(), ranked.getElementType());
      Value zeroScalar = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(ranked.getElementType()));
      rewriter.create<linalg::FillOp>(loc, ValueRange{zeroScalar}, ValueRange{zeroBuf});
      Value reluEmpty = rewriter.create<tensor::EmptyOp>(loc, ranked.getShape(), ranked.getElementType());
      result = rewriter.create<linalg::MaxOp>(loc, dstTy, ValueRange{result, zeroBuf}, ValueRange{reluEmpty})->getResult(0);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

template <typename LinalgOp>
static LogicalResult lowerDwcBinaryToLinalg(Operation *op, PatternRewriter &rewriter) {
  if (op->getNumResults() != 1 || op->getNumOperands() != 2)
    return failure();
  bool isRelu = false;
  if (auto activation = dyn_cast<ActivationFunctionAttr>(op->getAttr("activation_function"))) {
    if (activation.getValue() == ActivationFunction::Relu)
      isRelu = true;
    else if (activation.getValue() != ActivationFunction::None)
      return failure();
  }
  Value lhs = op->getOperand(0);
  Value rhs = op->getOperand(1);
  Type dstTy = op->getResult(0).getType();
  if (!hasSameElementType(lhs.getType(), dstTy) ||
      !hasSameElementType(rhs.getType(), dstTy))
    return failure();
  auto ranked = dyn_cast<RankedTensorType>(dstTy);
  if (!ranked || !ranked.hasStaticShape())
    return failure();
  Location loc = op->getLoc();
  Value empty = rewriter.create<tensor::EmptyOp>(loc, ranked.getShape(), ranked.getElementType());
  Value result = rewriter.create<LinalgOp>(loc, dstTy, ValueRange{lhs, rhs}, ValueRange{empty})->getResult(0);
  if (isRelu) {
    Value zeroBuf = rewriter.create<tensor::EmptyOp>(loc, ranked.getShape(), ranked.getElementType());
    Value zeroScalar = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(ranked.getElementType()));
    rewriter.create<linalg::FillOp>(loc, ValueRange{zeroScalar}, ValueRange{zeroBuf});
    Value reluEmpty = rewriter.create<tensor::EmptyOp>(loc, ranked.getShape(), ranked.getElementType());
    result = rewriter.create<linalg::MaxOp>(loc, dstTy, ValueRange{result, zeroBuf}, ValueRange{reluEmpty})->getResult(0);
  }
  rewriter.replaceOp(op, result);
  return success();
}

struct DwcAddLowering : public RewritePattern {
  DwcAddLowering(MLIRContext *ctx)
      : RewritePattern("dwc.add", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    return lowerDwcBinaryToLinalg<linalg::AddOp>(op, rewriter);
  }
};

struct DwcBinaryLowering : public RewritePattern {
  StringRef root;
  using LowerFn = LogicalResult (*)(Operation *, PatternRewriter &);
  LowerFn lower;
  DwcBinaryLowering(StringRef rootName, LowerFn fn, MLIRContext *ctx)
      : RewritePattern(rootName, 1, ctx), root(rootName), lower(fn) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getName().getStringRef() != root)
      return failure();
    return lower(op, rewriter);
  }
};

template <typename LinalgOp>
static LogicalResult lowerDwcBinaryOp(Operation *op, PatternRewriter &rewriter) {
  return lowerDwcBinaryToLinalg<LinalgOp>(op, rewriter);
}

} // namespace

void mlir::darwinn::populateLowerCopySlicePatterns(RewritePatternSet &patterns) {
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
  patterns.add<CwiseLowering>(ctx);
  patterns.add<DwcAddLowering>(ctx);
  patterns.add<DwcBinaryLowering>("dwc.multiply", lowerDwcBinaryOp<linalg::MulOp>, ctx);
  patterns.add<DwcBinaryLowering>("dwc.divide", lowerDwcBinaryOp<linalg::DivOp>, ctx);
  patterns.add<DwcBinaryLowering>("dwc.maximum", lowerDwcBinaryOp<linalg::MaxOp>, ctx);
  patterns.add<DwcBinaryLowering>("dwc.minimum", lowerDwcBinaryOp<linalg::MinOp>, ctx);
  patterns.add<DwcBinaryLowering>("dwc.subtract", lowerDwcBinaryOp<linalg::SubOp>, ctx);
}
