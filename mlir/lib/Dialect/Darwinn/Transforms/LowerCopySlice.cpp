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
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <optional>

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

struct Slice1dLowering : public RewritePattern {
  StringRef root;
  Slice1dLowering(StringRef rootName, MLIRContext *ctx)
      : RewritePattern(rootName, 1, ctx), root(rootName) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getName().getStringRef() != root)
      return failure();
    if (op->getNumResults() != 1 || op->getNumOperands() != 1)
      return failure();
    Value input = op->getOperand(0);
    Type dstTy = op->getResult(0).getType();
    auto srcRanked = dyn_cast<RankedTensorType>(input.getType());
    auto dstRanked = dyn_cast<RankedTensorType>(dstTy);
    if (!srcRanked || !dstRanked || !srcRanked.hasStaticShape() || !dstRanked.hasStaticShape())
      return failure();
    if (srcRanked.getRank() != 1 || dstRanked.getRank() != 1)
      return failure();
    if (srcRanked.getElementType() != dstRanked.getElementType())
      return failure();
    if (dstRanked.getDimSize(0) > srcRanked.getDimSize(0))
      return failure();
    OpFoldResult offset = rewriter.getIndexAttr(0);
    OpFoldResult extent = rewriter.getIndexAttr(dstRanked.getDimSize(0));
    OpFoldResult stride = rewriter.getIndexAttr(1);
    rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(op, dstRanked, input, ArrayRef<OpFoldResult>{offset}, ArrayRef<OpFoldResult>{extent}, ArrayRef<OpFoldResult>{stride});
    return success();
  }
};

struct DiveRefReductionLowering : public RewritePattern {
  DiveRefReductionLowering(MLIRContext *ctx)
      : RewritePattern("darwinn.dive_ref_reduction", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1 || op->getNumOperands() != 1)
      return failure();
    auto axes = op->getAttrOfType<DenseI64ArrayAttr>("axes");
    if (!axes || axes.size() != 1 || !isa<RankedTensorType>(op->getOperand(0).getType()))
      return failure();
    Value input = op->getOperand(0);
    Type dstTy = op->getResult(0).getType();
    auto srcRanked = dyn_cast<RankedTensorType>(input.getType());
    auto dstRanked = dyn_cast<RankedTensorType>(dstTy);
    if (!srcRanked || !dstRanked || !srcRanked.hasStaticShape() || !dstRanked.hasStaticShape())
      return failure();
    if (!isa<FloatType>(srcRanked.getElementType()) || srcRanked.getElementType() != dstRanked.getElementType())
      return failure();
    int64_t axis = axes.asArrayRef()[0];
    if (axis < 0 || axis >= srcRanked.getRank())
      return failure();
    if (srcRanked.getRank() != dstRanked.getRank())
      return failure();
    for (int64_t d = 0; d < srcRanked.getRank(); ++d) {
      if (d == axis) {
        if (dstRanked.getDimSize(d) != 1)
          return failure();
      } else if (srcRanked.getDimSize(d) != dstRanked.getDimSize(d))
        return failure();
    }
    Location loc = op->getLoc();
    Value empty = rewriter.create<tensor::EmptyOp>(loc, dstRanked.getShape(), dstRanked.getElementType());
    Value zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(dstRanked.getElementType()));
    empty = rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{empty}).getResult(0);
    SmallVector<AffineExpr> outExprs;
    for (int64_t d = 0; d < srcRanked.getRank(); ++d)
      outExprs.push_back(d == axis ? rewriter.getAffineConstantExpr(0) : rewriter.getAffineDimExpr(d));
    SmallVector<AffineMap> maps{rewriter.getMultiDimIdentityMap(srcRanked.getRank()),
                                AffineMap::get(srcRanked.getRank(), 0, outExprs, op->getContext())};
    SmallVector<utils::IteratorType> iters(srcRanked.getRank(), utils::IteratorType::parallel);
    iters[axis] = utils::IteratorType::reduction;
    auto generic = rewriter.create<linalg::GenericOp>(loc, TypeRange{dstTy}, ValueRange{input}, ValueRange{empty},
        maps, iters,
        [&](OpBuilder &nested, Location nloc, ValueRange args) {
          Value out = nested.create<arith::AddFOp>(nloc, args[0], args[1]);
          nested.create<linalg::YieldOp>(nloc, out);
        });
    rewriter.replaceOp(op, generic->getResult(0));
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
struct InterleaveLowering : public RewritePattern {
  InterleaveLowering(MLIRContext *ctx) : RewritePattern("dwc.interleave", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1 || op->getNumOperands() != 2)
      return failure();
    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    Type dstTy = op->getResult(0).getType();
    auto lhsRanked = dyn_cast<RankedTensorType>(lhs.getType());
    auto rhsRanked = dyn_cast<RankedTensorType>(rhs.getType());
    auto dstRanked = dyn_cast<RankedTensorType>(dstTy);
    if (!lhsRanked || !rhsRanked || !dstRanked)
      return failure();
    if (!lhsRanked.hasStaticShape() || !rhsRanked.hasStaticShape() || !dstRanked.hasStaticShape())
      return failure();
    if (lhsRanked.getRank() != 1 || rhsRanked.getRank() != 1 || dstRanked.getRank() != 1)
      return failure();
    if (lhsRanked.getShape() != rhsRanked.getShape())
      return failure();
    if (lhsRanked.getDimSize(0) * 2 != dstRanked.getDimSize(0))
      return failure();
    if (lhsRanked.getElementType() != dstRanked.getElementType() ||
        rhsRanked.getElementType() != dstRanked.getElementType())
      return failure();
    Location loc = op->getLoc();
    Value empty = rewriter.create<tensor::EmptyOp>(loc, dstRanked.getShape(), dstRanked.getElementType());
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value c2 = rewriter.create<arith::ConstantIndexOp>(loc, 2);
    Value n = rewriter.create<arith::ConstantIndexOp>(loc, lhsRanked.getDimSize(0));
    auto loop = rewriter.create<scf::ForOp>(loc, c0, n, c1, ValueRange{empty});
    rewriter.setInsertionPointToStart(loop.getBody());
    Value iv = loop.getInductionVar();
    Value cur = loop.getRegionIterArg(0);
    Value a = rewriter.create<tensor::ExtractOp>(loc, lhs, ValueRange{iv});
    Value b = rewriter.create<tensor::ExtractOp>(loc, rhs, ValueRange{iv});
    Value even = rewriter.create<arith::MulIOp>(loc, iv, c2);
    Value odd = rewriter.create<arith::AddIOp>(loc, even, c1);
    Value ins0 = rewriter.create<tensor::InsertOp>(loc, a, cur, ValueRange{even});
    Value ins1 = rewriter.create<tensor::InsertOp>(loc, b, ins0, ValueRange{odd});
    rewriter.create<scf::YieldOp>(loc, ValueRange{ins1});
    rewriter.setInsertionPointAfter(loop);
    rewriter.replaceOp(op, loop->getResult(0));
    return success();
  }
};


struct VicaAddLowering : public RewritePattern {
  VicaAddLowering(MLIRContext *ctx) : RewritePattern("dwc.vica_add", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1 || op->getNumOperands() != 2)
      return failure();
    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    Type dstTy = op->getResult(0).getType();
    if (!hasSameElementType(lhs.getType(), dstTy) || !hasSameElementType(rhs.getType(), dstTy))
      return failure();
    auto dstRanked = dyn_cast<RankedTensorType>(dstTy);
    if (!dstRanked || !dstRanked.hasStaticShape())
      return failure();
    Location loc = op->getLoc();
    Value empty = rewriter.create<tensor::EmptyOp>(loc, dstRanked.getShape(), dstRanked.getElementType());
    rewriter.replaceOpWithNewOp<linalg::AddOp>(op, TypeRange{dstTy}, ValueRange{lhs, rhs}, ValueRange{empty});
    return success();
  }
};

struct ForwardAnyLowering : public RewritePattern {
  StringRef root;
  StringRef target;
  ForwardAnyLowering(StringRef rootName, StringRef targetName, MLIRContext *ctx)
      : RewritePattern(rootName, 1, ctx), root(rootName), target(targetName) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    if (op->getName().getStringRef() != root)
      return failure();
    if (op->getNumResults() != 1 || op->getNumOperands() < 1)
      return failure();
    SmallVector<NamedAttribute> attrs;
    for (auto attr : op->getAttrs())
      attrs.push_back(attr);
    Operation *next = makeVmOp(rewriter, op->getLoc(), target, op->getOperands(),
                               op->getResultTypes(), attrs);
    rewriter.replaceOp(op, next->getResults());
    return success();
  }
};
struct ForwardResultsLowering : public RewritePattern {
  StringRef root;
  StringRef target;
  ForwardResultsLowering(StringRef rootName, StringRef targetName, MLIRContext *ctx)
      : RewritePattern(rootName, 1, ctx), root(rootName), target(targetName) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    if (op->getName().getStringRef() != root)
      return failure();
    if (op->getNumResults() != 2 || op->getNumOperands() != 1)
      return failure();
    SmallVector<NamedAttribute> attrs;
    for (auto attr : op->getAttrs())
      attrs.push_back(attr);
    Operation *values = makeVmOp(rewriter, op->getLoc(), target, op->getOperands(),
                                 TypeRange{op->getResult(0).getType()}, attrs);
    Operation *indices = makeVmOp(rewriter, op->getLoc(), target, op->getOperands(),
                                  TypeRange{op->getResult(1).getType()}, attrs);
    rewriter.replaceOp(op, ValueRange{values->getResult(0), indices->getResult(0)});
    return success();
  }
};
struct ForwardLowering : public RewritePattern {
  StringRef root;
  StringRef target;
  ForwardLowering(StringRef rootName, StringRef targetName, MLIRContext *ctx)
      : RewritePattern(rootName, 1, ctx), root(rootName), target(targetName) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    if (op->getName().getStringRef() != root)
      return failure();
    if (op->getNumResults() != 1 || op->getNumOperands() < 1)
      return failure();
    SmallVector<NamedAttribute> attrs;
    for (auto attr : op->getAttrs())
      attrs.push_back(attr);
    Operation *next = makeVmOp(rewriter, op->getLoc(), target, op->getOperands(),
                               op->getResultTypes(), attrs);
    rewriter.replaceOp(op, next->getResults());
    return success();
  }
};

struct UnsortedSegmentReduceLowering : public RewritePattern {
  UnsortedSegmentReduceLowering(MLIRContext *ctx)
      : RewritePattern("dwc.unsorted_segment_reduce", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1 || op->getNumOperands() != 2)
      return failure();
    auto numSeg = op->getAttrOfType<IntegerAttr>("num_segments");
    auto opType = op->getAttrOfType<ReductionTypeAttr>("op_type");
    if (!numSeg || !opType)
      return failure();
    if (opType.getValue() != ReductionType::Sum && opType.getValue() != ReductionType::Max)
      return failure();
    int64_t n = numSeg.getInt();
    if (n <= 0)
      return failure();
    Value data = op->getOperand(0);
    Value ids = op->getOperand(1);
    Type dstTy = op->getResult(0).getType();
    auto dataRanked = dyn_cast<RankedTensorType>(data.getType());
    auto idsRanked = dyn_cast<RankedTensorType>(ids.getType());
    auto dstRanked = dyn_cast<RankedTensorType>(dstTy);
    if (!dataRanked || !idsRanked || !dstRanked)
      return failure();
    if (!dataRanked.hasStaticShape() || !idsRanked.hasStaticShape() || !dstRanked.hasStaticShape())
      return failure();
    if (dataRanked.getRank() != 1 || idsRanked.getRank() != 1 || dstRanked.getRank() != 1)
      return failure();
    if (dataRanked.getDimSize(0) != idsRanked.getDimSize(0) || dstRanked.getDimSize(0) != n)
      return failure();
    if (dataRanked.getElementType() != dstRanked.getElementType())
      return failure();
    if (!isa<FloatType, IntegerType>(dataRanked.getElementType()))
      return failure();
    Location loc = op->getLoc();
    Type element = dstRanked.getElementType();
    TypedAttr identity = rewriter.getZeroAttr(element);
    if (opType.getValue() == ReductionType::Max) {
      if (auto floating = dyn_cast<FloatType>(element))
        identity = rewriter.getFloatAttr(element, APFloat::getInf(floating.getFloatSemantics(), true));
      else
        identity = rewriter.getIntegerAttr(element, APInt::getSignedMinValue(cast<IntegerType>(element).getWidth()));
    }

    Value initial = rewriter.create<arith::ConstantOp>(loc, identity);
    Value acc = rewriter.create<tensor::EmptyOp>(loc, dstRanked.getShape(), dstRanked.getElementType());
    acc = rewriter.create<linalg::FillOp>(loc, ValueRange{initial}, ValueRange{acc}).getResult(0);
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value dim = rewriter.create<arith::ConstantIndexOp>(loc, dataRanked.getDimSize(0));
    auto loop = rewriter.create<scf::ForOp>(loc, c0, dim, c1, ValueRange{acc});
    rewriter.setInsertionPointToStart(loop.getBody());
    Value iv = loop.getInductionVar();
    Value cur = loop.getRegionIterArg(0);
    Value d = rewriter.create<tensor::ExtractOp>(loc, data, ValueRange{iv});
    Value id = rewriter.create<tensor::ExtractOp>(loc, ids, ValueRange{iv});
    Value idx = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), id);
    Value old = rewriter.create<tensor::ExtractOp>(loc, cur, ValueRange{idx});
    Value upd;
    if (opType.getValue() == ReductionType::Max) {
      if (isa<FloatType>(dataRanked.getElementType()))
        upd = rewriter.create<arith::MaximumFOp>(loc, old, d);
      else
        upd = rewriter.create<arith::MaxSIOp>(loc, old, d);
    } else if (isa<FloatType>(dataRanked.getElementType())) {
      upd = rewriter.create<arith::AddFOp>(loc, old, d);
    } else {
      upd = rewriter.create<arith::AddIOp>(loc, old, d);
    }
    Value next = rewriter.create<tensor::InsertOp>(loc, upd, cur, ValueRange{idx});
    rewriter.create<scf::YieldOp>(loc, ValueRange{next});
    rewriter.setInsertionPointAfter(loop);
    rewriter.replaceOp(op, loop->getResult(0));
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
    auto opType = op->getAttrOfType<CwiseOpTypeAttr>("op_type");
    if (!opType)
      return failure();
    bool isRelu = false;
    if (auto activation = op->getAttrOfType<ActivationFunctionAttr>("activation_function")) {
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
    if (!elem && opType.getValue() == CwiseOpType::Pow) {
      auto lhsTy = dyn_cast<RankedTensorType>(lhs.getType());
      auto dstRanked = dyn_cast<RankedTensorType>(dstTy);
      if (!lhsTy || !dstRanked || !lhsTy.hasStaticShape() || !dstRanked.hasStaticShape())
        return failure();
      if (lhsTy.getShape() != dstRanked.getShape())
        return failure();
      if (!isa<FloatType>(lhsTy.getElementType()))
        return failure();
      int64_t rank = lhsTy.getRank();
      SmallVector<AffineMap> maps(3, rewriter.getMultiDimIdentityMap(rank));
      SmallVector<utils::IteratorType> iters(rank, utils::IteratorType::parallel);
      auto generic = rewriter.create<linalg::GenericOp>(loc, TypeRange{dstTy}, ValueRange{lhs, rhs}, ValueRange{empty},
          maps, iters,
          [&](OpBuilder &nested, Location nloc, ValueRange args) {
            Value p = nested.create<math::PowFOp>(nloc, args[0], args[1]);
            nested.create<linalg::YieldOp>(nloc, p);
          });
      elem = generic.getOperation();
    }
    if (!elem) {
      auto lhsTy = dyn_cast<RankedTensorType>(lhs.getType());
      auto dstRanked = dyn_cast<RankedTensorType>(dstTy);
      if (!lhsTy || !dstRanked || !lhsTy.hasStaticShape() || !dstRanked.hasStaticShape())
        return failure();
      if (lhsTy.getShape() != dstRanked.getShape())
        return failure();
      std::optional<arith::CmpIPredicate> intPred;
      std::optional<arith::CmpFPredicate> floatPred;
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
        break;
      }
      if (!intPred) {
        auto bitKind = opType.getValue();
        if (bitKind != CwiseOpType::BitwiseAnd && bitKind != CwiseOpType::BitwiseOr &&
            bitKind != CwiseOpType::BitwiseXor && bitKind != CwiseOpType::LogicalAnd &&
            bitKind != CwiseOpType::ArithmeticLeftShift && bitKind != CwiseOpType::ArithmeticRightShift &&
            bitKind != CwiseOpType::LogicalRightShift && bitKind != CwiseOpType::Modulus)
          return failure();
        int64_t brank = lhsTy.getRank();
        SmallVector<AffineMap> bmaps(3, rewriter.getMultiDimIdentityMap(brank));
        SmallVector<utils::IteratorType> biters(brank, utils::IteratorType::parallel);
        auto generic = rewriter.create<linalg::GenericOp>(loc, TypeRange{dstTy}, ValueRange{lhs, rhs}, ValueRange{empty},
            bmaps, biters,
            [&](OpBuilder &nested, Location nloc, ValueRange args) {
              Type signless = IntegerType::get(nloc->getContext(), dstRanked.getElementType().getIntOrFloatBitWidth(), IntegerType::Signless);
              Value a = args[0].getType() == signless ? args[0] : nested.create<UnrealizedConversionCastOp>(nloc, signless, args[0]).getResult(0);
              Value b = args[1].getType() == signless ? args[1] : nested.create<UnrealizedConversionCastOp>(nloc, signless, args[1]).getResult(0);
              Value out;
              switch (bitKind) {
              case CwiseOpType::BitwiseAnd:
                out = nested.create<arith::AndIOp>(nloc, a, b);
                break;
              case CwiseOpType::BitwiseOr:
                out = nested.create<arith::OrIOp>(nloc, a, b);
                break;
              case CwiseOpType::BitwiseXor:
                out = nested.create<arith::XOrIOp>(nloc, a, b);
                break;
              case CwiseOpType::LogicalAnd:
                out = nested.create<arith::AndIOp>(nloc, a, b);
                break;
              case CwiseOpType::ArithmeticLeftShift:
                out = nested.create<arith::ShLIOp>(nloc, a, b);
                break;
              case CwiseOpType::ArithmeticRightShift:
                out = nested.create<arith::ShRSIOp>(nloc, a, b);
                break;
              case CwiseOpType::LogicalRightShift:
                out = nested.create<arith::ShRUIOp>(nloc, a, b);
                break;
              default:
                out = nested.create<arith::RemUIOp>(nloc, a, b);
                break;
              }
              if (out.getType() != dstRanked.getElementType())
                out = nested.create<UnrealizedConversionCastOp>(nloc, dstRanked.getElementType(), out).getResult(0);
              nested.create<linalg::YieldOp>(nloc, out);
            });
        elem = generic.getOperation();
        Value result = elem->getResult(0);
        if (isRelu) {
          Value zeroBuf = rewriter.create<tensor::EmptyOp>(loc, ranked.getShape(), ranked.getElementType());
          Value zeroScalar = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(ranked.getElementType()));
          zeroBuf = rewriter.create<linalg::FillOp>(loc, ValueRange{zeroScalar}, ValueRange{zeroBuf}).getResult(0);
          Value reluEmpty = rewriter.create<tensor::EmptyOp>(loc, ranked.getShape(), ranked.getElementType());
          result = rewriter.create<linalg::MaxOp>(loc, dstTy, ValueRange{result, zeroBuf}, ValueRange{reluEmpty})->getResult(0);
        }
        rewriter.replaceOp(op, result);
        return success();
      }
      int64_t rank = lhsTy.getRank();
      SmallVector<AffineMap> maps(3, rewriter.getMultiDimIdentityMap(rank));
      SmallVector<utils::IteratorType> iters(rank, utils::IteratorType::parallel);
      auto generic = rewriter.create<linalg::GenericOp>(loc, TypeRange{dstTy}, ValueRange{lhs, rhs}, ValueRange{empty},
          maps, iters,
          [&](OpBuilder &nested, Location nloc, ValueRange args) {
            Value cmp = isFloat ? static_cast<Value>(nested.create<arith::CmpFOp>(nloc, *floatPred, args[0], args[1]))
                                : static_cast<Value>(nested.create<arith::CmpIOp>(nloc, *intPred, args[0], args[1]));
            nested.create<linalg::YieldOp>(nloc, cmp);
          });
      elem = generic.getOperation();
    }
    Value result = elem->getResult(0);
    if (isRelu) {
      Value zeroBuf = rewriter.create<tensor::EmptyOp>(loc, ranked.getShape(), ranked.getElementType());
      Value zeroScalar = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(ranked.getElementType()));
      zeroBuf = rewriter.create<linalg::FillOp>(loc, ValueRange{zeroScalar}, ValueRange{zeroBuf}).getResult(0);
      Value reluEmpty = rewriter.create<tensor::EmptyOp>(loc, ranked.getShape(), ranked.getElementType());
      result = rewriter.create<linalg::MaxOp>(loc, dstTy, ValueRange{result, zeroBuf}, ValueRange{reluEmpty})->getResult(0);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct SelectLowering : public RewritePattern {
  SelectLowering(MLIRContext *ctx)
      : RewritePattern("dwc.select", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1 || op->getNumOperands() != 2)
      return failure();
    Value cond = op->getOperand(0);
    Value data = op->getOperand(1);
    Type dstTy = op->getResult(0).getType();
    auto condRanked = dyn_cast<RankedTensorType>(cond.getType());
    auto dataRanked = dyn_cast<RankedTensorType>(data.getType());
    auto dstRanked = dyn_cast<RankedTensorType>(dstTy);
    if (!condRanked || !dataRanked || !dstRanked)
      return failure();
    if (!condRanked.hasStaticShape() || !dataRanked.hasStaticShape() || !dstRanked.hasStaticShape())
      return failure();
    if (condRanked.getShape() != dataRanked.getShape() || dataRanked.getShape() != dstRanked.getShape())
      return failure();
    auto i1 = IntegerType::get(op->getContext(), 1, IntegerType::Signless);
    if (condRanked.getElementType() != i1)
      return failure();
    if (dataRanked.getElementType() != dstRanked.getElementType())
      return failure();
    Location loc = op->getLoc();
    Value empty = rewriter.create<tensor::EmptyOp>(loc, dstRanked.getShape(), dstRanked.getElementType());
    int64_t rank = dstRanked.getRank();
    SmallVector<AffineMap> maps(4, rewriter.getMultiDimIdentityMap(rank));
    SmallVector<utils::IteratorType> iters(rank, utils::IteratorType::parallel);
    auto generic = rewriter.create<linalg::GenericOp>(loc, TypeRange{dstTy}, ValueRange{cond, data, data}, ValueRange{empty},
        maps, iters,
        [&](OpBuilder &nested, Location nloc, ValueRange args) {
          Value picked = nested.create<arith::SelectOp>(nloc, args[0], args[1], args[2]);
          nested.create<linalg::YieldOp>(nloc, picked);
        });
    rewriter.replaceOp(op, generic->getResult(0));
    return success();
  }
};
struct DarwinnReluLowering : public RewritePattern {
  DarwinnReluLowering(MLIRContext *ctx) : RewritePattern("darwinn.relu", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1 || op->getNumOperands() != 1)
      return failure();
    Value input = op->getOperand(0);
    Type dstTy = op->getResult(0).getType();
    auto srcRanked = dyn_cast<RankedTensorType>(input.getType());
    auto dstRanked = dyn_cast<RankedTensorType>(dstTy);
    if (!srcRanked || !dstRanked || !srcRanked.hasStaticShape() || !dstRanked.hasStaticShape())
      return failure();
    if (srcRanked.getShape() != dstRanked.getShape())
      return failure();
    if (!isa<FloatType>(srcRanked.getElementType()) || srcRanked.getElementType() != dstRanked.getElementType())
      return failure();
    Location loc = op->getLoc();
    Value zeroBuf = rewriter.create<tensor::EmptyOp>(loc, dstRanked.getShape(), dstRanked.getElementType());
    Value zeroScalar = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(dstRanked.getElementType()));
    zeroBuf = rewriter.create<linalg::FillOp>(loc, ValueRange{zeroScalar}, ValueRange{zeroBuf}).getResult(0);
    Value empty = rewriter.create<tensor::EmptyOp>(loc, dstRanked.getShape(), dstRanked.getElementType());
    rewriter.replaceOpWithNewOp<linalg::MaxOp>(op, TypeRange{dstTy}, ValueRange{input, zeroBuf}, ValueRange{empty});
    return success();
  }
};
struct DarwinnScalarArithLowering : public RewritePattern {
  StringRef root;
  DarwinnScalarArithLowering(StringRef rootName, MLIRContext *ctx)
      : RewritePattern(rootName, 1, ctx), root(rootName) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    if (op->getName().getStringRef() != root)
      return failure();
    if (op->getNumResults() != 1 || op->getNumOperands() != 2)
      return failure();
    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    Type dstTy = op->getResult(0).getType();
    if (!isa<Float16Type, BFloat16Type>(lhs.getType()) || lhs.getType() != rhs.getType() || rhs.getType() != dstTy)
      return failure();
    if (root == "darwinn.scalar.fadd")
      rewriter.replaceOpWithNewOp<arith::AddFOp>(op, dstTy, lhs, rhs);
    else
      rewriter.replaceOpWithNewOp<arith::SubFOp>(op, dstTy, lhs, rhs);
    return success();
  }
};

struct DarwinnScalarCmpLowering : public RewritePattern {
  StringRef root;
  arith::CmpFPredicate pred;
  DarwinnScalarCmpLowering(StringRef rootName, arith::CmpFPredicate p, MLIRContext *ctx)
      : RewritePattern(rootName, 1, ctx), root(rootName), pred(p) {}


  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    if (op->getName().getStringRef() != root)
      return failure();
    if (op->getNumResults() != 1 || op->getNumOperands() != 2)
      return failure();
    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    Type dstTy = op->getResult(0).getType();
    if (!isa<Float16Type, BFloat16Type>(lhs.getType()) || lhs.getType() != rhs.getType())
      return failure();
    if (dstTy != lhs.getType())
      return failure();
    Value cmp = rewriter.create<arith::CmpFOp>(op->getLoc(), pred, lhs, rhs);
    Value ext = rewriter.create<arith::UIToFPOp>(op->getLoc(), dstTy, cmp);
    rewriter.replaceOp(op, ext);
    return success();
  }
};

struct DarwinnBinaryMapLowering : public RewritePattern {
  DarwinnBinaryMapLowering(MLIRContext *ctx) : RewritePattern("darwinn.binary_map", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1 || op->getNumOperands() != 2)
      return failure();
    auto fn = op->getAttrOfType<StringAttr>("function");
    if (!fn || fn.getValue() != "add")
      return failure();
    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    Type dstTy = op->getResult(0).getType();
    auto dstRanked = dyn_cast<RankedTensorType>(dstTy);
    if (!dstRanked || !dstRanked.hasStaticShape())
      return failure();
    if (!hasSameElementType(lhs.getType(), dstTy) || !hasSameElementType(rhs.getType(), dstTy))
      return failure();
    Location loc = op->getLoc();
    Value empty = rewriter.create<tensor::EmptyOp>(loc, dstRanked.getShape(), dstRanked.getElementType());
    rewriter.replaceOpWithNewOp<linalg::AddOp>(op, TypeRange{dstTy}, ValueRange{lhs, rhs}, ValueRange{empty});
    return success();
  }
};

struct DarwinnResidualAddLowering : public RewritePattern {
  DarwinnResidualAddLowering(MLIRContext *ctx) : RewritePattern("darwinn.vica_residual_add_op", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1 || op->getNumOperands() != 2)
      return failure();
    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    Type dstTy = op->getResult(0).getType();
    auto dstRanked = dyn_cast<RankedTensorType>(dstTy);
    if (!dstRanked || !dstRanked.hasStaticShape())
      return failure();
    if (!hasSameElementType(lhs.getType(), dstTy) || !hasSameElementType(rhs.getType(), dstTy))
      return failure();
    Location loc = op->getLoc();
    Value empty = rewriter.create<tensor::EmptyOp>(loc, dstRanked.getShape(), dstRanked.getElementType());
    rewriter.replaceOpWithNewOp<linalg::AddOp>(op, TypeRange{dstTy}, ValueRange{lhs, rhs}, ValueRange{empty});
    return success();
  }
};


struct DarwinnSelectLowering : public RewritePattern {
  DarwinnSelectLowering(MLIRContext *ctx) : RewritePattern("darwinn.select", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1 || op->getNumOperands() != 3)
      return failure();
    Value cond = op->getOperand(0);
    Value lhs = op->getOperand(1);
    Value rhs = op->getOperand(2);
    Type dstTy = op->getResult(0).getType();
    auto condRanked = dyn_cast<RankedTensorType>(cond.getType());
    auto dstRanked = dyn_cast<RankedTensorType>(dstTy);
    if (!condRanked || !dstRanked || !condRanked.hasStaticShape() || !dstRanked.hasStaticShape())
      return failure();
    if (condRanked.getShape() != dstRanked.getShape())
      return failure();
    auto i1 = IntegerType::get(op->getContext(), 1, IntegerType::Signless);
    if (condRanked.getElementType() != i1)
      return failure();
    if (!hasSameElementType(lhs.getType(), dstTy) || !hasSameElementType(rhs.getType(), dstTy))
      return failure();
    Location loc = op->getLoc();
    Value empty = rewriter.create<tensor::EmptyOp>(loc, dstRanked.getShape(), dstRanked.getElementType());
    int64_t rank = dstRanked.getRank();
    SmallVector<AffineMap> maps(4, rewriter.getMultiDimIdentityMap(rank));
    SmallVector<utils::IteratorType> iters(rank, utils::IteratorType::parallel);
    auto generic = rewriter.create<linalg::GenericOp>(loc, TypeRange{dstTy}, ValueRange{cond, lhs, rhs}, ValueRange{empty},
        maps, iters,
        [&](OpBuilder &nested, Location nloc, ValueRange args) {
          Value picked = nested.create<arith::SelectOp>(nloc, args[0], args[1], args[2]);
          nested.create<linalg::YieldOp>(nloc, picked);
        });
    rewriter.replaceOp(op, generic->getResult(0));
    return success();
  }
};

struct DarwinnScatterLowering : public RewritePattern {
  DarwinnScatterLowering(MLIRContext *ctx) : RewritePattern("darwinn.scatter", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1 || op->getNumOperands() < 2)
      return failure();
    SmallVector<NamedAttribute> attrs;
    for (auto attr : op->getAttrs())
      attrs.push_back(attr);
    Operation *next = makeVmOp(rewriter, op->getLoc(), "dive_vm.scatter_nd", op->getOperands(),
                               op->getResultTypes(), attrs);
    rewriter.replaceOp(op, next->getResults());
    return success();
  }
};

struct DarwinnSplitLowering : public RewritePattern {
  DarwinnSplitLowering(MLIRContext *ctx) : RewritePattern("darwinn.split", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1 || op->getNumOperands() != 1)
      return failure();
    Value input = op->getOperand(0);
    if (input.getType() != op->getResult(0).getType())
      return failure();
    rewriter.replaceOp(op, input);
    return success();
  }
};


struct ReductionLowering : public RewritePattern {
  ReductionLowering(MLIRContext *ctx)
      : RewritePattern("dwc.reduction", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1 || op->getNumOperands() != 1)
      return failure();
    auto opType = op->getAttrOfType<ReductionTypeAttr>("op_type");
    if (!opType)
      return failure();
    if (opType.getValue() != ReductionType::Sum && opType.getValue() != ReductionType::Max)
      return failure();
    if (auto activation = op->getAttrOfType<SimpleActivationFunctionAttr>("activation_function"))
      if (activation.getValue() != SimpleActivationFunction::None)
        return failure();
    auto dims = op->getAttrOfType<DenseIntElementsAttr>("dimensions");
    if (!dims)
      return failure();
    SmallVector<int64_t> reduceDims;
    for (auto v : dims.getValues<APInt>())
      reduceDims.push_back(v.getSExtValue());
    if (reduceDims.size() != 1)
      return failure();
    Value input = op->getOperand(0);
    Type dstTy = op->getResult(0).getType();
    auto srcRanked = dyn_cast<RankedTensorType>(input.getType());
    auto dstRanked = dyn_cast<RankedTensorType>(dstTy);
    if (!srcRanked || !dstRanked || !srcRanked.hasStaticShape() || !dstRanked.hasStaticShape())
      return failure();
    if (!isa<FloatType>(srcRanked.getElementType()) || srcRanked.getElementType() != dstRanked.getElementType())
      return failure();
    if (srcRanked.getRank() != dstRanked.getRank() + static_cast<int64_t>(reduceDims.size()))
      return failure();
    Location loc = op->getLoc();
    Value initBuf = rewriter.create<tensor::EmptyOp>(loc, dstRanked.getShape(), dstRanked.getElementType());
    Value initScalar;
    if (opType.getValue() == ReductionType::Sum)
      initScalar = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(dstRanked.getElementType()));
    else
      initScalar = rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(dstRanked.getElementType(), -std::numeric_limits<float>::infinity()));
    initBuf = rewriter.create<linalg::FillOp>(loc, ValueRange{initScalar}, ValueRange{initBuf}).getResult(0);
    bool isMax = opType.getValue() == ReductionType::Max;
    auto reduce = rewriter.create<linalg::ReduceOp>(loc, ValueRange{input}, ValueRange{initBuf}, reduceDims,
        [&](OpBuilder &nested, Location nloc, ValueRange args) {
          Value out = isMax ? static_cast<Value>(nested.create<arith::MaximumFOp>(nloc, args[0], args[1]))
                            : static_cast<Value>(nested.create<arith::AddFOp>(nloc, args[0], args[1]));
          nested.create<linalg::YieldOp>(nloc, out);
        });
    rewriter.replaceOp(op, reduce->getResult(0));
    return success();
  }
};

template <typename LinalgOp>
static LogicalResult lowerDwcBinaryToLinalg(Operation *op, PatternRewriter &rewriter) {
  if (op->getNumResults() != 1 || op->getNumOperands() != 2)
    return failure();
  bool isRelu = false;
  if (auto activation = op->getAttrOfType<ActivationFunctionAttr>("activation_function")) {
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
    zeroBuf = rewriter.create<linalg::FillOp>(loc, ValueRange{zeroScalar}, ValueRange{zeroBuf}).getResult(0);
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

struct ReshapeLowering : public RewritePattern {
  ReshapeLowering(MLIRContext *ctx)
      : RewritePattern("dwc.reshape", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1 || op->getNumOperands() != 1)
      return failure();
    Value input = op->getOperand(0);
    Type dstTy = op->getResult(0).getType();
    auto srcRanked = dyn_cast<RankedTensorType>(input.getType());
    auto dstRanked = dyn_cast<RankedTensorType>(dstTy);
    if (!srcRanked || !dstRanked || !srcRanked.hasStaticShape() || !dstRanked.hasStaticShape())
      return failure();
    if (srcRanked.getNumElements() != dstRanked.getNumElements() ||
        srcRanked.getElementType() != dstRanked.getElementType())
      return failure();
    if (srcRanked == dstRanked) {
      rewriter.replaceOp(op, input);
      return success();
    }

    Location loc = op->getLoc();
    if (srcRanked.getRank() == 0) {
      rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
          op, dstTy, input, ArrayRef<ReassociationIndices>{});
      return success();
    }

    if (dstRanked.getRank() == 0) {
      rewriter.replaceOpWithNewOp<tensor::CollapseShapeOp>(
          op, dstTy, input, ArrayRef<ReassociationIndices>{});
      return success();
    }

    Value flat = input;
    if (srcRanked.getRank() > 1) {
      ReassociationIndices dimensions;
      for (int64_t dim = 0; dim < srcRanked.getRank(); ++dim)
        dimensions.push_back(dim);

      auto flatType = RankedTensorType::get({srcRanked.getNumElements()},
                                            srcRanked.getElementType());
      flat = rewriter.create<tensor::CollapseShapeOp>(
          loc, flatType, input, ArrayRef<ReassociationIndices>{dimensions});
    }

    if (dstRanked.getRank() == 1) {
      rewriter.replaceOp(op, flat);
      return success();
    }

    ReassociationIndices dimensions;
    for (int64_t dim = 0; dim < dstRanked.getRank(); ++dim)
      dimensions.push_back(dim);

    rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
        op, dstTy, flat, ArrayRef<ReassociationIndices>{dimensions});
    return success();
  }
};

struct TransposeLowering : public RewritePattern {
  TransposeLowering(MLIRContext *ctx)
      : RewritePattern("dwc.transpose", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1 || op->getNumOperands() != 1)
      return failure();
    auto perm = op->getAttrOfType<DenseIntElementsAttr>("permutation");
    if (!perm)
      return failure();
    SmallVector<int64_t> permutation;
    for (auto v : perm.getValues<APInt>())
      permutation.push_back(v.getSExtValue());
    Value input = op->getOperand(0);
    Type dstTy = op->getResult(0).getType();
    auto srcRanked = dyn_cast<RankedTensorType>(input.getType());
    auto dstRanked = dyn_cast<RankedTensorType>(dstTy);
    if (!srcRanked || !dstRanked || !srcRanked.hasStaticShape() || !dstRanked.hasStaticShape())
      return failure();
    if (static_cast<int64_t>(permutation.size()) != srcRanked.getRank())
      return failure();
    Location loc = op->getLoc();
    Value empty = rewriter.create<tensor::EmptyOp>(loc, dstRanked.getShape(), dstRanked.getElementType());
    rewriter.replaceOpWithNewOp<linalg::TransposeOp>(op, input, empty, permutation);
    return success();
  }
};

struct MatmulLowering : public RewritePattern {
  StringRef root;
  MatmulLowering(StringRef rootName, MLIRContext *ctx)
      : RewritePattern(rootName, 1, ctx), root(rootName) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getName().getStringRef() != root)
      return failure();
    if (op->getNumResults() != 1 || op->getNumOperands() != 2)
      return failure();
    auto transpose = op->getAttrOfType<BoolAttr>("transpose_rhs");
    if (op->hasAttr("transpose_rhs") && !transpose)
      return failure();
    bool transposeRhs = transpose && transpose.getValue();
    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    Type dstTy = op->getResult(0).getType();
    auto lhsRanked = dyn_cast<RankedTensorType>(lhs.getType());
    auto rhsRanked = dyn_cast<RankedTensorType>(rhs.getType());
    auto dstRanked = dyn_cast<RankedTensorType>(dstTy);
    if (!lhsRanked || !rhsRanked || !dstRanked)
      return failure();
    if (!lhsRanked.hasStaticShape() || !rhsRanked.hasStaticShape() || !dstRanked.hasStaticShape())
      return failure();
    if (lhsRanked.getRank() != 2 || rhsRanked.getRank() != 2 || dstRanked.getRank() != 2)
      return failure();
    int64_t contractingDimension = transposeRhs ? 1 : 0;
    int64_t outputDimension = transposeRhs ? 0 : 1;
    if (lhsRanked.getDimSize(1) != rhsRanked.getDimSize(contractingDimension))
      return failure();
    if (lhsRanked.getDimSize(0) != dstRanked.getDimSize(0) ||
        rhsRanked.getDimSize(outputDimension) != dstRanked.getDimSize(1))
      return failure();
    if (!isa<FloatType>(lhsRanked.getElementType()) || lhsRanked.getElementType() != rhsRanked.getElementType() ||
        rhsRanked.getElementType() != dstRanked.getElementType())
      return failure();
    Location loc = op->getLoc();
    Value empty = rewriter.create<tensor::EmptyOp>(loc, dstRanked.getShape(), dstRanked.getElementType());
    Value zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(dstRanked.getElementType()));
    empty = rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{empty}).getResult(0);
    Value product = transposeRhs
        ? linalg::MatmulTransposeBOp::create(
              rewriter, loc, TypeRange{dstTy}, ValueRange{lhs, rhs}, ValueRange{empty}).getResult(0)
        : linalg::MatmulOp::create(
              rewriter, loc, TypeRange{dstTy}, ValueRange{lhs, rhs}, ValueRange{empty}).getResult(0);
    rewriter.replaceOp(op, product);
    return success();
  }
};
struct SubByteMatmulLowering : public RewritePattern {
  SubByteMatmulLowering(StringRef rootName, MLIRContext *ctx)
      : RewritePattern(rootName, 1, ctx), root(rootName) {}
  StringRef root;

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    if (op->getName().getStringRef() != root)
      return failure();
    if (op->getNumResults() != 1 || op->getNumOperands() != 2)
      return failure();
    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    Type dstTy = op->getResult(0).getType();
    auto lhsRanked = dyn_cast<RankedTensorType>(lhs.getType());
    auto rhsRanked = dyn_cast<RankedTensorType>(rhs.getType());
    auto dstRanked = dyn_cast<RankedTensorType>(dstTy);
    if (!lhsRanked || !rhsRanked || !dstRanked)
      return failure();
    if (!lhsRanked.hasStaticShape() || !rhsRanked.hasStaticShape() || !dstRanked.hasStaticShape())
      return failure();
    if (lhsRanked.getRank() != 2 || rhsRanked.getRank() != 2 || dstRanked.getRank() != 2)
      return failure();
    if (lhsRanked.getDimSize(1) != rhsRanked.getDimSize(0))
      return failure();
    if (lhsRanked.getDimSize(0) != dstRanked.getDimSize(0) || rhsRanked.getDimSize(1) != dstRanked.getDimSize(1))
      return failure();
    if (!isa<FloatType>(lhsRanked.getElementType()) || lhsRanked.getElementType() != dstRanked.getElementType())
      return failure();
    if (!isa<IntegerType>(rhsRanked.getElementType()) || rhsRanked.getElementTypeBitWidth() > 8)
      return failure();
    Location loc = op->getLoc();
    Value rhsEmpty = rewriter.create<tensor::EmptyOp>(loc, rhsRanked.getShape(), lhsRanked.getElementType());
    int64_t rank = rhsRanked.getRank();
    SmallVector<AffineMap> extMaps(2, rewriter.getMultiDimIdentityMap(rank));
    SmallVector<utils::IteratorType> extIters(rank, utils::IteratorType::parallel);
    bool isSigned = rhsRanked.getElementType().isSignedInteger();
    auto rhsF = rewriter.create<linalg::GenericOp>(loc, TypeRange{RankedTensorType::get(rhsRanked.getShape(), lhsRanked.getElementType())}, ValueRange{rhs}, ValueRange{rhsEmpty},
        extMaps, extIters,
        [&](OpBuilder &nested, Location nloc, ValueRange args) {
          Value e = isSigned ? static_cast<Value>(nested.create<arith::SIToFPOp>(nloc, lhsRanked.getElementType(), args[0]))
                             : static_cast<Value>(nested.create<arith::UIToFPOp>(nloc, lhsRanked.getElementType(), args[0]));
          nested.create<linalg::YieldOp>(nloc, e);
        });
    Value empty = rewriter.create<tensor::EmptyOp>(loc, dstRanked.getShape(), dstRanked.getElementType());
    Value zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(dstRanked.getElementType()));
    empty = rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{empty}).getResult(0);
    rewriter.replaceOpWithNewOp<linalg::MatmulOp>(op, TypeRange{dstTy}, ValueRange{lhs, rhsF->getResult(0)}, ValueRange{empty});
    return success();
  }
};


struct BroadcastLowering : public RewritePattern {
  BroadcastLowering(MLIRContext *ctx)
      : RewritePattern("dwc.broadcast", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1 || op->getNumOperands() != 1)
      return failure();
    Value input = op->getOperand(0);
    Type dstTy = op->getResult(0).getType();
    auto srcRanked = dyn_cast<RankedTensorType>(input.getType());
    auto dstRanked = dyn_cast<RankedTensorType>(dstTy);
    if (!srcRanked || !dstRanked || !srcRanked.hasStaticShape() || !dstRanked.hasStaticShape())
      return failure();
    if (srcRanked.getElementType() != dstRanked.getElementType())
      return failure();
    if (srcRanked.getRank() + 1 != dstRanked.getRank())
      return failure();
    for (int64_t d = 0; d < srcRanked.getRank(); ++d)
      if (srcRanked.getDimSize(d) != dstRanked.getDimSize(d + 1))
        return failure();
    Location loc = op->getLoc();
    Value empty = rewriter.create<tensor::EmptyOp>(loc, dstRanked.getShape(), dstRanked.getElementType());
    SmallVector<int64_t> dims{0};
    rewriter.replaceOpWithNewOp<linalg::BroadcastOp>(op, input, empty, dims);
    return success();
  }
};

struct ConcatenationLowering : public RewritePattern {
  ConcatenationLowering(MLIRContext *ctx)
      : RewritePattern("dwc.concatenation", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1 || op->getNumOperands() != 2)
      return failure();
    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    Type dstTy = op->getResult(0).getType();
    auto lhsRanked = dyn_cast<RankedTensorType>(lhs.getType());
    auto rhsRanked = dyn_cast<RankedTensorType>(rhs.getType());
    auto dstRanked = dyn_cast<RankedTensorType>(dstTy);
    if (!lhsRanked || !rhsRanked || !dstRanked)
      return failure();
    if (!lhsRanked.hasStaticShape() || !rhsRanked.hasStaticShape() || !dstRanked.hasStaticShape())
      return failure();
    if (lhsRanked.getRank() != 1 || rhsRanked.getRank() != 1 || dstRanked.getRank() != 1)
      return failure();
    if (lhsRanked.getElementType() != rhsRanked.getElementType() ||
        rhsRanked.getElementType() != dstRanked.getElementType())
      return failure();
    if (lhsRanked.getDimSize(0) + rhsRanked.getDimSize(0) != dstRanked.getDimSize(0))
      return failure();
    rewriter.replaceOpWithNewOp<tensor::ConcatOp>(op, dstTy, rewriter.getI64IntegerAttr(0), ValueRange{lhs, rhs});
    return success();
  }
};

struct SliceLowering : public RewritePattern {
  SliceLowering(MLIRContext *ctx)
      : RewritePattern("dwc.slice", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1 || op->getNumOperands() != 1)
      return failure();
    auto getI32 = [&](StringRef name, int64_t &out) -> bool {
      auto attr = op->getAttrOfType<IntegerAttr>(name);
      if (!attr)
        return false;
      out = attr.getInt();
      return true;
    };
    int64_t begin, size;
    if (!getI32("in_begin", begin) || !getI32("in_size", size))
      return failure();
    if (begin < 0 || size <= 0)
      return failure();
    Value input = op->getOperand(0);
    Type dstTy = op->getResult(0).getType();
    auto srcRanked = dyn_cast<RankedTensorType>(input.getType());
    auto dstRanked = dyn_cast<RankedTensorType>(dstTy);
    if (!srcRanked || !dstRanked || !srcRanked.hasStaticShape() || !dstRanked.hasStaticShape())
      return failure();
    if (srcRanked.getRank() != 1 || dstRanked.getRank() != 1)
      return failure();
    if (srcRanked.getElementType() != dstRanked.getElementType())
      return failure();
    if (begin + size > srcRanked.getDimSize(0) || size != dstRanked.getDimSize(0))
      return failure();
    OpFoldResult offset = rewriter.getIndexAttr(begin);
    OpFoldResult extent = rewriter.getIndexAttr(size);
    OpFoldResult stride = rewriter.getIndexAttr(1);
    rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(op, dstRanked, input, ArrayRef<OpFoldResult>{offset}, ArrayRef<OpFoldResult>{extent}, ArrayRef<OpFoldResult>{stride});
    return success();
  }
};
struct DynamicSliceNdLowering : public RewritePattern {
  DynamicSliceNdLowering(StringRef rootName, MLIRContext *ctx)
      : RewritePattern(rootName, 1, ctx), root(rootName) {}
  StringRef root;

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    if (op->getName().getStringRef() != root)
      return failure();
    if (op->getNumResults() != 1 || op->getNumOperands() != 1)
      return failure();
    auto mode = op->getAttrOfType<IntegerAttr>("mode");
    auto size = op->getAttrOfType<IntegerAttr>("slice_size");
    if (!mode || !size)
      return failure();
    if (size.getInt() <= 0)
      return failure();
    Value input = op->getOperand(0);
    Type dstTy = op->getResult(0).getType();
    auto srcRanked = dyn_cast<RankedTensorType>(input.getType());
    auto dstRanked = dyn_cast<RankedTensorType>(dstTy);
    if (!srcRanked || !dstRanked || !srcRanked.hasStaticShape() || !dstRanked.hasStaticShape())
      return failure();
    if (srcRanked.getRank() != 1 || dstRanked.getRank() != 1)
      return failure();
    if (srcRanked.getElementType() != dstRanked.getElementType())
      return failure();
    if (size.getInt() != dstRanked.getDimSize(0) || size.getInt() > srcRanked.getDimSize(0))
      return failure();
    OpFoldResult offset = rewriter.getIndexAttr(0);
    OpFoldResult extent = rewriter.getIndexAttr(size.getInt());
    OpFoldResult stride = rewriter.getIndexAttr(1);
    rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(op, dstRanked, input, ArrayRef<OpFoldResult>{offset}, ArrayRef<OpFoldResult>{extent}, ArrayRef<OpFoldResult>{stride});
    return success();
  }
};

struct DynamicUpdateSliceNdLowering : public RewritePattern {
  DynamicUpdateSliceNdLowering(MLIRContext *ctx)
      : RewritePattern("dwc.dynamic_update_slice", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1 || op->getNumOperands() != 2)
      return failure();
    auto mode = op->getAttrOfType<IntegerAttr>("mode");
    if (!mode)
      return failure();
    Value update = op->getOperand(0);
    Value base = op->getOperand(1);
    Type dstTy = op->getResult(0).getType();
    auto updRanked = dyn_cast<RankedTensorType>(update.getType());
    auto baseRanked = dyn_cast<RankedTensorType>(base.getType());
    auto dstRanked = dyn_cast<RankedTensorType>(dstTy);
    if (!updRanked || !baseRanked || !dstRanked)
      return failure();
    if (!updRanked.hasStaticShape() || !baseRanked.hasStaticShape() || !dstRanked.hasStaticShape())
      return failure();
    if (updRanked.getRank() != 1 || baseRanked.getRank() != 1 || dstRanked.getRank() != 1)
      return failure();
    if (baseRanked.getShape() != dstRanked.getShape())
      return failure();
    if (updRanked.getDimSize(0) > baseRanked.getDimSize(0))
      return failure();
    if (updRanked.getElementType() != dstRanked.getElementType() ||
        baseRanked.getElementType() != dstRanked.getElementType())
      return failure();
    OpFoldResult offset = rewriter.getIndexAttr(0);
    OpFoldResult extent = rewriter.getIndexAttr(updRanked.getDimSize(0));
    OpFoldResult stride = rewriter.getIndexAttr(1);
    rewriter.replaceOpWithNewOp<tensor::InsertSliceOp>(op, update, base, ArrayRef<OpFoldResult>{offset}, ArrayRef<OpFoldResult>{extent}, ArrayRef<OpFoldResult>{stride});
    return success();
  }
};


static Value emitLogistic(OpBuilder &builder, Location loc, Value input);

static Value applyConvolutionActivation(PatternRewriter &rewriter, Location loc,
                                       Value input, ActivationFunction activation,
                                       Value bias = {}, FloatAttr clipMin = {},
                                       FloatAttr clipMax = {}) {
  if (activation == ActivationFunction::None && !bias && !clipMin)
    return input;

  auto type = cast<RankedTensorType>(input.getType());
  Value empty = rewriter.create<tensor::EmptyOp>(loc, type.getShape(), type.getElementType());
  SmallVector<Value> inputs{input};
  SmallVector<AffineMap> maps{rewriter.getMultiDimIdentityMap(type.getRank())};

  if (bias) {
    inputs.push_back(bias);
    maps.push_back(AffineMap::get(type.getRank(), 0,
                                rewriter.getAffineDimExpr(type.getRank() - 1),
                                rewriter.getContext()));
  }

  maps.push_back(rewriter.getMultiDimIdentityMap(type.getRank()));
  SmallVector<utils::IteratorType> iterators(type.getRank(), utils::IteratorType::parallel);
  return rewriter.create<linalg::GenericOp>(
      loc, TypeRange{type}, inputs, ValueRange{empty}, maps, iterators,
      [&](OpBuilder &nested, Location nestedLoc, ValueRange args) {
        Value element = args[0];

        if (bias)
          element = nested.create<arith::AddFOp>(nestedLoc, element, args[1]);

        Value result;
        auto constant = [&](double value) {
          return nested.create<arith::ConstantOp>(
              nestedLoc, nested.getFloatAttr(type.getElementType(), value)).getResult();
        };

        switch (activation) {
        case ActivationFunction::None:
          result = element;
          break;
        case ActivationFunction::Exp:
          result = nested.create<math::ExpOp>(nestedLoc, element);
          break;
        case ActivationFunction::Logistic:
          result = emitLogistic(nested, nestedLoc, element);
          break;
        case ActivationFunction::ReciprocalSqrt:
          result = nested.create<math::RsqrtOp>(nestedLoc, element);
          break;
        case ActivationFunction::Relu:
          result = nested.create<arith::MaximumFOp>(nestedLoc, element, constant(0.0));
          break;
        case ActivationFunction::Tanh:
          result = nested.create<math::TanhOp>(nestedLoc, element);
          break;
        case ActivationFunction::GeluApproximated: {
          Value square = nested.create<arith::MulFOp>(nestedLoc, element, element);
          Value cube = nested.create<arith::MulFOp>(nestedLoc, square, element);
          Value cubicTerm = nested.create<arith::MulFOp>(nestedLoc, constant(0.044715), cube);
          Value sum = nested.create<arith::AddFOp>(nestedLoc, element, cubicTerm);
          Value scaled = nested.create<arith::MulFOp>(nestedLoc, constant(0.7978845608028654), sum);
          Value hyperbolic = nested.create<math::TanhOp>(nestedLoc, scaled);
          Value factor = nested.create<arith::AddFOp>(nestedLoc, constant(1.0), hyperbolic);
          Value halfInput = nested.create<arith::MulFOp>(nestedLoc, constant(0.5), element);
          result = nested.create<arith::MulFOp>(nestedLoc, halfInput, factor);
          break;
        }
        }

        if (clipMin) {
          Value lower = nested.create<arith::ConstantOp>(nestedLoc, clipMin);
          Value upper = nested.create<arith::ConstantOp>(nestedLoc, clipMax);
          result = nested.create<arith::MaximumFOp>(nestedLoc, result, lower);
          result = nested.create<arith::MinimumFOp>(nestedLoc, result, upper);
        }

        nested.create<linalg::YieldOp>(nestedLoc, result);
      }).getResult(0);
}

struct ConvolutionLowering : public RewritePattern {
  bool depthwise;
  ConvolutionLowering(StringRef root, bool isDepthwise, MLIRContext *ctx)
      : RewritePattern(root, 1, ctx), depthwise(isDepthwise) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1 || (op->getNumOperands() != 2 && op->getNumOperands() != 3))
      return failure();
    auto activation = op->getAttrOfType<ActivationFunctionAttr>("activation_function");
    auto cell = op->getAttrOfType<CellOperationAttr>("cell_operation");
    auto padding = op->getAttrOfType<PaddingAttr>("pad");
    if (!activation || !cell || !padding || cell.getValue() != CellOperation::Mac ||
        padding.getValue() != Padding::None)
      return failure();

    if (auto multiplier = op->getAttrOfType<IntegerAttr>("depth_multiplier")) {
      if (multiplier.getInt() != 1)
        return failure();
    }

    auto getI64 = [&](StringRef name, int64_t &out) -> bool {
      auto attr = op->getAttrOfType<IntegerAttr>(name);
      if (!attr)
        return false;
      out = attr.getInt();
      return true;
    };
    int64_t xStride, yStride, xDilation, yDilation;
    if (!getI64("x_stride", xStride) || !getI64("y_stride", yStride) ||
        !getI64("x_dilation_rate", xDilation) || !getI64("y_dilation_rate", yDilation))
      return failure();
    if (xStride < 1 || xStride > 8 || yStride < 1 || yStride > 8 || xDilation < 1 || xDilation > 8 || yDilation < 1 || yDilation > 8)
      return failure();
    Value input = op->getOperand(0);
    Value filter = op->getOperand(1);
    Type dstTy = op->getResult(0).getType();
    auto inRanked = dyn_cast<RankedTensorType>(input.getType());
    auto filtRanked = dyn_cast<RankedTensorType>(filter.getType());
    auto dstRanked = dyn_cast<RankedTensorType>(dstTy);
    if (!inRanked || !filtRanked || !dstRanked)
      return failure();
    if (!inRanked.hasStaticShape() || !filtRanked.hasStaticShape() || !dstRanked.hasStaticShape())
      return failure();
    if (inRanked.getRank() != 4 || filtRanked.getRank() != (depthwise ? 3 : 4) || dstRanked.getRank() != 4)
      return failure();
    if (!isa<FloatType>(inRanked.getElementType()) || inRanked.getElementType() != filtRanked.getElementType() ||
        filtRanked.getElementType() != dstRanked.getElementType())
      return failure();
    int64_t outputChannels = depthwise ? inRanked.getDimSize(3) : filtRanked.getDimSize(3);
    if (inRanked.getDimSize(0) != dstRanked.getDimSize(0) || inRanked.getDimSize(3) != filtRanked.getDimSize(2) ||
        outputChannels != dstRanked.getDimSize(3))
      return failure();
    if (filtRanked.getDimSize(0) <= 0 || filtRanked.getDimSize(1) <= 0)
      return failure();

    int64_t effKh = (filtRanked.getDimSize(0) - 1) * yDilation + 1;
    int64_t effKw = (filtRanked.getDimSize(1) - 1) * xDilation + 1;
    if (inRanked.getDimSize(1) < effKh || inRanked.getDimSize(2) < effKw ||
        dstRanked.getDimSize(1) != (inRanked.getDimSize(1) - effKh) / yStride + 1 ||
        dstRanked.getDimSize(2) != (inRanked.getDimSize(2) - effKw) / xStride + 1)
      return failure();
    Value bias;

    if (op->getNumOperands() == 3) {
      bias = op->getOperand(2);
      auto biasType = dyn_cast<RankedTensorType>(bias.getType());

      if (!biasType || biasType.getRank() != 1 ||
          biasType.getDimSize(0) != outputChannels ||
          biasType.getElementType() != dstRanked.getElementType())
        return failure();
    }

    auto clipMin = op->getAttrOfType<FloatAttr>("activation_clip_min");
    auto clipMax = op->getAttrOfType<FloatAttr>("activation_clip_max");

    if (static_cast<bool>(clipMin) != static_cast<bool>(clipMax) ||
        (clipMin && (clipMin.getType() != dstRanked.getElementType() ||
                     clipMax.getType() != dstRanked.getElementType() ||
                     clipMin.getValue().isNaN() || clipMax.getValue().isNaN() ||
                     clipMin.getValue().compare(clipMax.getValue()) == APFloat::cmpGreaterThan)))
      return failure();

    Location loc = op->getLoc();
    Value empty = rewriter.create<tensor::EmptyOp>(loc, dstRanked.getShape(), dstRanked.getElementType());
    Value zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(dstRanked.getElementType()));
    Value initial = rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{empty}).getResult(0);
    auto strides = rewriter.getI64TensorAttr({yStride, xStride});
    auto dilations = rewriter.getI64TensorAttr({yDilation, xDilation});
    Value result = depthwise
        ? rewriter.create<linalg::DepthwiseConv2DNhwcHwcOp>(
              loc, TypeRange{dstTy}, ValueRange{input, filter}, ValueRange{initial},
              strides, dilations).getResult(0)
        : rewriter.create<linalg::Conv2DNhwcHwcfOp>(
              loc, TypeRange{dstTy}, ValueRange{input, filter}, ValueRange{initial},
              strides, dilations).getResult(0);
    rewriter.replaceOp(op, applyConvolutionActivation(rewriter, loc, result, activation.getValue(),
                                                        bias, clipMin, clipMax));
    return success();
  }
};

struct GenericDotLowering : public RewritePattern {
  GenericDotLowering(MLIRContext *ctx) : RewritePattern("dwc.generic_dot", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1 || op->getNumOperands() != 2)
      return failure();
    auto act = op->getAttrOfType<ActivationFunctionAttr>("activation_function");
    auto batch = op->getAttrOfType<IntegerAttr>("batch_dim_count");
    auto contracting = op->getAttrOfType<IntegerAttr>("contracting_dim_count");
    if (!act || !batch || !contracting)
      return failure();
    if (act.getValue() != ActivationFunction::None || batch.getInt() != 1 || contracting.getInt() != 1)
      return failure();
    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    Type dstTy = op->getResult(0).getType();
    auto lhsRanked = dyn_cast<RankedTensorType>(lhs.getType());
    auto rhsRanked = dyn_cast<RankedTensorType>(rhs.getType());
    auto dstRanked = dyn_cast<RankedTensorType>(dstTy);
    if (!lhsRanked || !rhsRanked || !dstRanked)
      return failure();
    if (!lhsRanked.hasStaticShape() || !rhsRanked.hasStaticShape() || !dstRanked.hasStaticShape())
      return failure();
    if (lhsRanked.getRank() != 3 || rhsRanked.getRank() != 3 || dstRanked.getRank() != 3)
      return failure();
    if (lhsRanked.getDimSize(0) != rhsRanked.getDimSize(0) || lhsRanked.getDimSize(0) != dstRanked.getDimSize(0))
      return failure();
    if (lhsRanked.getDimSize(2) != rhsRanked.getDimSize(1))
      return failure();
    if (lhsRanked.getDimSize(1) != dstRanked.getDimSize(1) || rhsRanked.getDimSize(2) != dstRanked.getDimSize(2))
      return failure();
    if (!isa<FloatType>(lhsRanked.getElementType()) || lhsRanked.getElementType() != rhsRanked.getElementType() ||
        rhsRanked.getElementType() != dstRanked.getElementType())
      return failure();
    Location loc = op->getLoc();
    Value empty = rewriter.create<tensor::EmptyOp>(loc, dstRanked.getShape(), dstRanked.getElementType());
    Value zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(dstRanked.getElementType()));
    empty = rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{empty}).getResult(0);
    rewriter.replaceOpWithNewOp<linalg::BatchMatmulOp>(op, TypeRange{dstTy}, ValueRange{lhs, rhs}, ValueRange{empty});
    return success();
  }
};


struct ClassifierLowering : public RewritePattern {
  ClassifierLowering(MLIRContext *ctx) : RewritePattern("dwc.classifier", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    auto axis = op->getAttrOfType<IntegerAttr>("axis");
    auto beta = op->getAttrOfType<FloatAttr>("beta");
    auto opType = op->getAttrOfType<ClassificationTypeAttr>("op_type");
    if (!axis || !beta || !opType)
      return failure();
    if (axis.getInt() != -1 || beta.getValueAsDouble() != 1.0 ||
        opType.getValue() != ClassificationType::Softmax)
      return failure();
    if (op->getNumResults() != 1 || op->getNumOperands() != 1)
      return failure();
    Value input = op->getOperand(0);
    Type dstTy = op->getResult(0).getType();
    auto srcRanked = dyn_cast<RankedTensorType>(input.getType());
    auto dstRanked = dyn_cast<RankedTensorType>(dstTy);
    if (!srcRanked || !dstRanked || !srcRanked.hasStaticShape() || !dstRanked.hasStaticShape())
      return failure();
    if (srcRanked.getShape() != dstRanked.getShape() ||
        srcRanked.getElementType() != dstRanked.getElementType() ||
        !isa<FloatType>(srcRanked.getElementType()))
      return failure();
    int64_t rank = srcRanked.getRank();
    if (rank < 1 || srcRanked.getDimSize(rank - 1) == 0)
      return failure();
    int64_t last = rank - 1;
    Location loc = op->getLoc();
    Type element = srcRanked.getElementType();
    SmallVector<int64_t> outerShape(srcRanked.getShape().drop_back());
    bool rankOne = outerShape.empty();
    if (rankOne)
      outerShape.push_back(1);
    Value emptyMax = rewriter.create<tensor::EmptyOp>(loc, outerShape, element);
    Value negativeInfinity = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(element,
            APFloat::getInf(cast<FloatType>(element).getFloatSemantics(), true)));
    Value initialMax = rewriter.create<linalg::FillOp>(
        loc, ValueRange{negativeInfinity}, ValueRange{emptyMax}).getResult(0);
    SmallVector<AffineExpr> outerExprs;
    for (int64_t d = 0; d < last; ++d)
      outerExprs.push_back(rewriter.getAffineDimExpr(d));
    AffineMap outMap = rankOne ? AffineMap::get(rank, 0, rewriter.getAffineConstantExpr(0), op->getContext()) : AffineMap::get(rank, 0, outerExprs, op->getContext());
    SmallVector<AffineMap> maxMaps{rewriter.getMultiDimIdentityMap(rank), outMap};
    SmallVector<utils::IteratorType> maxIters(rank, utils::IteratorType::parallel);
    maxIters[last] = utils::IteratorType::reduction;
    auto maxOut = rewriter.create<linalg::GenericOp>(loc, RankedTensorType::get(outerShape, element), ValueRange{input}, ValueRange{initialMax}, maxMaps, maxIters, [&](OpBuilder &nested, Location nloc, ValueRange args) {
         Value best = nested.create<arith::MaximumFOp>(nloc, args[0], args[1]);
         nested.create<linalg::YieldOp>(nloc, best);
      }).getResult(0);
    Value emptyExp = rewriter.create<tensor::EmptyOp>(loc, dstRanked.getShape(), element);
    auto idMap = rewriter.getMultiDimIdentityMap(rank);
    SmallVector<utils::IteratorType> expIters(rank, utils::IteratorType::parallel);
    Value zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto shifted = rewriter.create<linalg::GenericOp>(loc, TypeRange{dstTy}, ValueRange{input}, ValueRange{emptyExp}, SmallVector<AffineMap>{idMap, idMap}, expIters, [&](OpBuilder &nested, Location nloc, ValueRange args) {
         SmallVector<Value> idx;
         if (rankOne)
            idx.push_back(zeroIdx);
         else for (int64_t d = 0; d < last; ++d)
            idx.push_back(nested.create<linalg::IndexOp>(nloc, d));
         Value best = nested.create<tensor::ExtractOp>(nloc, maxOut, ValueRange{idx});
         Value diff = nested.create<arith::SubFOp>(nloc, args[0], best);
         Value exp = nested.create<math::ExpOp>(nloc, diff);
         nested.create<linalg::YieldOp>(nloc, exp);
      }).getResult(0);
    Value emptySum = rewriter.create<tensor::EmptyOp>(loc, outerShape, element);
    Value zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(element));
    Value initialSum = rewriter.create<linalg::FillOp>(
        loc, ValueRange{zero}, ValueRange{emptySum}).getResult(0);
    auto total = rewriter.create<linalg::GenericOp>(loc, RankedTensorType::get(outerShape, element), ValueRange{shifted}, ValueRange{initialSum}, maxMaps, maxIters, [&](OpBuilder &nested, Location nloc, ValueRange args) {
         Value acc = nested.create<arith::AddFOp>(nloc, args[0], args[1]);
         nested.create<linalg::YieldOp>(nloc, acc);
      }).getResult(0);
    Value empty = rewriter.create<tensor::EmptyOp>(loc, dstRanked.getShape(), element);
    auto out = rewriter.create<linalg::GenericOp>(loc, TypeRange{dstTy}, ValueRange{shifted}, ValueRange{empty}, SmallVector<AffineMap>{idMap, idMap}, expIters, [&](OpBuilder &nested, Location nloc, ValueRange args) {
         SmallVector<Value> idx;
         if (rankOne)
            idx.push_back(zeroIdx);
         else for (int64_t d = 0; d < last; ++d)
            idx.push_back(nested.create<linalg::IndexOp>(nloc, d));
         Value sum = nested.create<tensor::ExtractOp>(nloc, total, ValueRange{idx});
         Value norm = nested.create<arith::DivFOp>(nloc, args[0], sum);
         nested.create<linalg::YieldOp>(nloc, norm);
      }).getResult(0);
    rewriter.replaceOp(op, out);
    return success();
  }
};

struct GenericConvLowering : public RewritePattern {
  GenericConvLowering(MLIRContext *ctx) : RewritePattern("dwc.generic_conv", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1 || op->getNumOperands() != 2)
      return failure();
    auto act = op->getAttrOfType<ActivationFunctionAttr>("activation_function");
    auto batchGroup = op->getAttrOfType<IntegerAttr>("batch_group_count");
    auto featGroup = op->getAttrOfType<IntegerAttr>("feature_group_count");
    auto stride = op->getAttrOfType<DenseIntElementsAttr>("stride");
    auto inputDil = op->getAttrOfType<DenseIntElementsAttr>("input_dilation");
    auto paramDil = op->getAttrOfType<DenseIntElementsAttr>("param_dilation");
    auto padding = op->getAttrOfType<DenseIntElementsAttr>("padding_amount");
    auto reversal = op->getAttrOfType<DenseIntElementsAttr>("param_reversal");
    if (!act || !batchGroup || !featGroup || !stride || !inputDil || !paramDil || !padding || !reversal)
      return failure();
    if (act.getValue() != ActivationFunction::None || batchGroup.getInt() != 1 || featGroup.getInt() != 1)
      return failure();
    auto isOnes = [](DenseIntElementsAttr a) {
      for (APInt v : a.getValues<APInt>())
        if (v.getSExtValue() != 1)
          return false;
      return true;
    };
    auto isZeros = [](DenseIntElementsAttr a) {
      for (APInt v : a.getValues<APInt>())
        if (!v.isZero())
          return false;
      return true;
    };
    if (!isOnes(stride) || !isOnes(inputDil) || !isOnes(paramDil) ||
        !isZeros(padding) || !isZeros(reversal))
      return failure();
    Value input = op->getOperand(0);
    Value filter = op->getOperand(1);
    Type dstTy = op->getResult(0).getType();
    auto inRanked = dyn_cast<RankedTensorType>(input.getType());
    auto filtRanked = dyn_cast<RankedTensorType>(filter.getType());
    auto dstRanked = dyn_cast<RankedTensorType>(dstTy);
    if (!inRanked || !filtRanked || !dstRanked)
      return failure();
    if (!inRanked.hasStaticShape() || !filtRanked.hasStaticShape() || !dstRanked.hasStaticShape())
      return failure();
    if (inRanked.getRank() != 4 || filtRanked.getRank() != 4 || dstRanked.getRank() != 4)
      return failure();
    if (!isa<FloatType>(inRanked.getElementType()) || inRanked.getElementType() != filtRanked.getElementType() ||
        filtRanked.getElementType() != dstRanked.getElementType())
      return failure();
    if (inRanked.getDimSize(0) != dstRanked.getDimSize(0) || inRanked.getDimSize(3) != filtRanked.getDimSize(2) ||
        filtRanked.getDimSize(3) != dstRanked.getDimSize(3))
      return failure();
    if (dstRanked.getDimSize(1) != inRanked.getDimSize(1) - filtRanked.getDimSize(0) + 1 ||
        dstRanked.getDimSize(2) != inRanked.getDimSize(2) - filtRanked.getDimSize(1) + 1)
      return failure();
    Location loc = op->getLoc();
    Value empty = rewriter.create<tensor::EmptyOp>(loc, dstRanked.getShape(), dstRanked.getElementType());
    Value zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(dstRanked.getElementType()));
    Value initial = rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{empty}).getResult(0);
    auto strides = rewriter.getI64TensorAttr({1, 1});
    auto dilations = rewriter.getI64TensorAttr({1, 1});
    rewriter.replaceOpWithNewOp<linalg::Conv2DNhwcHwcfOp>(op, TypeRange{dstTy}, ValueRange{input, filter}, ValueRange{initial}, strides, dilations);
    return success();
  }
};

struct TransposedConvLowering : public RewritePattern {
  TransposedConvLowering(MLIRContext *ctx) : RewritePattern("dwc.transposed_convolution", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1 || op->getNumOperands() != 2)
      return failure();
    auto act = op->getAttrOfType<ActivationFunctionAttr>("activation_function");
    auto cell = op->getAttrOfType<CellOperationAttr>("cell_operation");
    auto padding = op->getAttrOfType<PaddingAttr>("pad");
    auto xs = op->getAttrOfType<IntegerAttr>("x_stride");
    auto ys = op->getAttrOfType<IntegerAttr>("y_stride");
    auto xd = op->getAttrOfType<IntegerAttr>("x_dilation_rate");
    auto yd = op->getAttrOfType<IntegerAttr>("y_dilation_rate");
    auto xOut = op->getAttrOfType<IntegerAttr>("x_out_dim");
    auto yOut = op->getAttrOfType<IntegerAttr>("y_out_dim");
    if (!act || !cell || !padding || !xs || !ys || !xd || !yd || !xOut || !yOut)
      return failure();
    if (cell.getValue() != CellOperation::Mac || padding.getValue() != Padding::None)
      return failure();
    if (xs.getInt() <= 0 || ys.getInt() <= 0 || xd.getInt() <= 0 || yd.getInt() <= 0)
      return failure();
    Value input = op->getOperand(0);
    Value filter = op->getOperand(1);
    Type dstTy = op->getResult(0).getType();
    auto inRanked = dyn_cast<RankedTensorType>(input.getType());
    auto filtRanked = dyn_cast<RankedTensorType>(filter.getType());
    auto dstRanked = dyn_cast<RankedTensorType>(dstTy);
    if (!inRanked || !filtRanked || !dstRanked)
      return failure();
    if (!inRanked.hasStaticShape() || !filtRanked.hasStaticShape() || !dstRanked.hasStaticShape())
      return failure();
    if (inRanked.getRank() != 4 || filtRanked.getRank() != 4 || dstRanked.getRank() != 4)
      return failure();
    if (!isa<FloatType>(inRanked.getElementType()) || inRanked.getElementType() != filtRanked.getElementType() ||
        filtRanked.getElementType() != dstRanked.getElementType())
      return failure();
    if (inRanked.getDimSize(0) != dstRanked.getDimSize(0) || inRanked.getDimSize(3) != filtRanked.getDimSize(2) ||
        filtRanked.getDimSize(3) != dstRanked.getDimSize(3))
      return failure();
    if (inRanked.getDimSize(1) <= 0 || inRanked.getDimSize(2) <= 0 ||
        filtRanked.getDimSize(0) <= 0 || filtRanked.getDimSize(1) <= 0)
      return failure();

    if (dstRanked.getDimSize(1) != (inRanked.getDimSize(1) - 1) * ys.getInt() +
            (filtRanked.getDimSize(0) - 1) * yd.getInt() + 1 ||
        dstRanked.getDimSize(2) != (inRanked.getDimSize(2) - 1) * xs.getInt() +
            (filtRanked.getDimSize(1) - 1) * xd.getInt() + 1)
      return failure();
    if (yOut.getInt() != dstRanked.getDimSize(1) || xOut.getInt() != dstRanked.getDimSize(2))
      return failure();
    Location loc = op->getLoc();
    int64_t hIn = inRanked.getDimSize(1);
    int64_t wIn = inRanked.getDimSize(2);
    int64_t kh = filtRanked.getDimSize(0);
    int64_t kw = filtRanked.getDimSize(1);
    Value empty = rewriter.create<tensor::EmptyOp>(loc, dstRanked.getShape(), dstRanked.getElementType());
    Value zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(dstRanked.getElementType()));
    Value initial = rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{empty}).getResult(0);
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value nV = rewriter.create<arith::ConstantIndexOp>(loc, inRanked.getDimSize(0));
    Value hV = rewriter.create<arith::ConstantIndexOp>(loc, hIn);
    Value wV = rewriter.create<arith::ConstantIndexOp>(loc, wIn);
    Value khV = rewriter.create<arith::ConstantIndexOp>(loc, kh);
    Value kwV = rewriter.create<arith::ConstantIndexOp>(loc, kw);
    Value ciV = rewriter.create<arith::ConstantIndexOp>(loc, inRanked.getDimSize(3));
    Value coV = rewriter.create<arith::ConstantIndexOp>(loc, dstRanked.getDimSize(3));
    Value yStride = rewriter.create<arith::ConstantIndexOp>(loc, ys.getInt());
    Value xStride = rewriter.create<arith::ConstantIndexOp>(loc, xs.getInt());
    Value yDilation = rewriter.create<arith::ConstantIndexOp>(loc, yd.getInt());
    Value xDilation = rewriter.create<arith::ConstantIndexOp>(loc, xd.getInt());
    auto ln = rewriter.create<scf::ForOp>(loc, c0, nV, c1, ValueRange{initial});
    rewriter.setInsertionPointToStart(ln.getBody());
    Value bn = ln.getInductionVar();
    auto lh = rewriter.create<scf::ForOp>(loc, c0, hV, c1, ValueRange{ln.getRegionIterArg(0)});
    rewriter.setInsertionPointToStart(lh.getBody());
    Value hi = lh.getInductionVar();
    auto lw = rewriter.create<scf::ForOp>(loc, c0, wV, c1, ValueRange{lh.getRegionIterArg(0)});
    rewriter.setInsertionPointToStart(lw.getBody());
    Value wi = lw.getInductionVar();
    auto lkh = rewriter.create<scf::ForOp>(loc, c0, khV, c1, ValueRange{lw.getRegionIterArg(0)});
    rewriter.setInsertionPointToStart(lkh.getBody());
    Value khi = lkh.getInductionVar();
    auto lkw = rewriter.create<scf::ForOp>(loc, c0, kwV, c1, ValueRange{lkh.getRegionIterArg(0)});
    rewriter.setInsertionPointToStart(lkw.getBody());
    Value kwi = lkw.getInductionVar();
    auto lci = rewriter.create<scf::ForOp>(loc, c0, ciV, c1, ValueRange{lkw.getRegionIterArg(0)});
    rewriter.setInsertionPointToStart(lci.getBody());
    Value cii = lci.getInductionVar();
    auto lco = rewriter.create<scf::ForOp>(loc, c0, coV, c1, ValueRange{lci.getRegionIterArg(0)});
    rewriter.setInsertionPointToStart(lco.getBody());
    Value coi = lco.getInductionVar();
    Value cur = lco.getRegionIterArg(0);
    Value inputY = rewriter.create<arith::MulIOp>(loc, hi, yStride);
    Value inputX = rewriter.create<arith::MulIOp>(loc, wi, xStride);
    Value kernelY = rewriter.create<arith::MulIOp>(loc, khi, yDilation);
    Value kernelX = rewriter.create<arith::MulIOp>(loc, kwi, xDilation);
    Value ho = rewriter.create<arith::AddIOp>(loc, inputY, kernelY);
    Value wo = rewriter.create<arith::AddIOp>(loc, inputX, kernelX);
    Value a = rewriter.create<tensor::ExtractOp>(loc, input, ValueRange{bn, hi, wi, cii});
    Value f = rewriter.create<tensor::ExtractOp>(loc, filter, ValueRange{khi, kwi, cii, coi});
    Value old = rewriter.create<tensor::ExtractOp>(loc, cur, ValueRange{bn, ho, wo, coi});
    Value prod = rewriter.create<arith::MulFOp>(loc, a, f);
    Value sum = rewriter.create<arith::AddFOp>(loc, old, prod);
    Value upd = rewriter.create<tensor::InsertOp>(loc, sum, cur, ValueRange{bn, ho, wo, coi});
    rewriter.create<scf::YieldOp>(loc, ValueRange{upd});
    rewriter.setInsertionPointAfter(lco);
    rewriter.create<scf::YieldOp>(loc, ValueRange{lco->getResult(0)});
    rewriter.setInsertionPointAfter(lci);
    rewriter.create<scf::YieldOp>(loc, ValueRange{lci->getResult(0)});
    rewriter.setInsertionPointAfter(lkw);
    rewriter.create<scf::YieldOp>(loc, ValueRange{lkw->getResult(0)});
    rewriter.setInsertionPointAfter(lkh);
    rewriter.create<scf::YieldOp>(loc, ValueRange{lkh->getResult(0)});
    rewriter.setInsertionPointAfter(lw);
    rewriter.create<scf::YieldOp>(loc, ValueRange{lw->getResult(0)});
    rewriter.setInsertionPointAfter(lh);
    rewriter.create<scf::YieldOp>(loc, ValueRange{lh->getResult(0)});
    rewriter.setInsertionPointAfter(ln);
    rewriter.replaceOp(op, applyConvolutionActivation(
        rewriter, loc, ln->getResult(0), act.getValue()));
    return success();
  }
};

struct ScalarLowering : public RewritePattern {
  ScalarLowering(MLIRContext *ctx) : RewritePattern("dwc.scalar", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    auto opType = op->getAttrOfType<ScalarOpTypeAttr>("op_type");
    if (!opType)
      return failure();
    if (op->getNumResults() != 1 || op->getNumOperands() != 1)
      return failure();
    Value input = op->getOperand(0);
    Type dstTy = op->getResult(0).getType();
    auto srcRanked = dyn_cast<RankedTensorType>(input.getType());
    auto dstRanked = dyn_cast<RankedTensorType>(dstTy);
    if (!srcRanked || !dstRanked || !srcRanked.hasStaticShape() || !dstRanked.hasStaticShape())
      return failure();
    if (srcRanked.getShape() != dstRanked.getShape())
      return failure();
    if (srcRanked.getElementType() != dstRanked.getElementType())
      return failure();
    if (opType.getValue() == ScalarOpType::Identity) {
      rewriter.replaceOp(op, input);
      return success();
    }
    auto imm = op->getAttrOfType<IntegerAttr>("immediate");
    if (!imm)
      return failure();
    Location loc = op->getLoc();
    Value empty = rewriter.create<tensor::EmptyOp>(loc, dstRanked.getShape(), dstRanked.getElementType());
    int64_t rank = dstRanked.getRank();
    SmallVector<AffineMap> maps(2, rewriter.getMultiDimIdentityMap(rank));
    SmallVector<utils::IteratorType> iters(rank, utils::IteratorType::parallel);
    int64_t scalar = imm.getInt();
    auto generic = rewriter.create<linalg::GenericOp>(loc, TypeRange{dstTy}, ValueRange{input}, ValueRange{empty},
        maps, iters,
        [&](OpBuilder &nested, Location nloc, ValueRange args) {
          Value rhs = isa<FloatType>(dstRanked.getElementType())
              ? nested.create<arith::ConstantOp>(nloc, nested.getFloatAttr(dstRanked.getElementType(), static_cast<double>(scalar)))
              : nested.create<arith::ConstantOp>(nloc, nested.getIntegerAttr(dstRanked.getElementType(), scalar));
          Value out = isa<FloatType>(dstRanked.getElementType())
              ? static_cast<Value>(nested.create<arith::SubFOp>(nloc, args[0], rhs))
              : static_cast<Value>(nested.create<arith::SubIOp>(nloc, args[0], rhs));
          nested.create<linalg::YieldOp>(nloc, out);
        });
    rewriter.replaceOp(op, generic->getResult(0));
    return success();
  }
};

struct TruncateFloatsLowering : public RewritePattern {
  TruncateFloatsLowering(MLIRContext *ctx) : RewritePattern("dwc.truncate_floats", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1 || op->getNumOperands() != 1)
      return failure();
    Value input = op->getOperand(0);
    Type dstTy = op->getResult(0).getType();
    auto srcRanked = dyn_cast<RankedTensorType>(input.getType());
    auto dstRanked = dyn_cast<RankedTensorType>(dstTy);
    if (!srcRanked || !dstRanked || !srcRanked.hasStaticShape() || !dstRanked.hasStaticShape())
      return failure();
    if (srcRanked.getShape() != dstRanked.getShape())
      return failure();
    if (!isa<FloatType>(srcRanked.getElementType()) || !isa<FloatType>(dstRanked.getElementType()))
      return failure();
    if (srcRanked.getElementTypeBitWidth() <= dstRanked.getElementTypeBitWidth())
      return failure();
    Location loc = op->getLoc();
    Value empty = rewriter.create<tensor::EmptyOp>(loc, dstRanked.getShape(), dstRanked.getElementType());
    int64_t rank = dstRanked.getRank();
    SmallVector<AffineMap> maps(2, rewriter.getMultiDimIdentityMap(rank));
    SmallVector<utils::IteratorType> iters(rank, utils::IteratorType::parallel);
    auto generic = rewriter.create<linalg::GenericOp>(loc, TypeRange{dstTy}, ValueRange{input}, ValueRange{empty},
        maps, iters,
        [&](OpBuilder &nested, Location nloc, ValueRange args) {
          Value out = nested.create<arith::TruncFOp>(nloc, dstRanked.getElementType(), args[0]);
          nested.create<linalg::YieldOp>(nloc, out);
        });
    rewriter.replaceOp(op, generic->getResult(0));
    return success();
  }
};


struct PaddingLowering : public RewritePattern {
  PaddingLowering(MLIRContext *ctx)
      : RewritePattern("dwc.padding", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1 || op->getNumOperands() != 1)
      return failure();
    auto dim = op->getAttrOfType<IntegerAttr>("dimension");
    auto pre = op->getAttrOfType<IntegerAttr>("pre_padding");
    auto post = op->getAttrOfType<IntegerAttr>("post_padding");
    if (!dim || !pre || !post)
      return failure();
    int64_t axis = dim.getInt();
    int64_t lo = pre.getInt();
    int64_t hi = post.getInt();
    if (lo < 0 || hi < 0)
      return failure();
    Value padValue;
    if (auto floatAttr = op->getAttrOfType<FloatAttr>("padding_value"))
      padValue = rewriter.create<arith::ConstantOp>(op->getLoc(), floatAttr);
    else
      return failure();
    Value input = op->getOperand(0);
    Type dstTy = op->getResult(0).getType();
    auto srcRanked = dyn_cast<RankedTensorType>(input.getType());
    auto dstRanked = dyn_cast<RankedTensorType>(dstTy);
    if (!srcRanked || !dstRanked || !srcRanked.hasStaticShape() || !dstRanked.hasStaticShape())
      return failure();
    if (srcRanked.getRank() != dstRanked.getRank())
      return failure();
    int64_t rank = srcRanked.getRank();
    if (axis < 0 || axis >= rank)
      return failure();
    for (int64_t d = 0; d < rank; ++d) {
      int64_t expect = srcRanked.getDimSize(d) + (d == axis ? lo + hi : 0);
      if (dstRanked.getDimSize(d) != expect)
        return failure();
    }
    if (srcRanked.getElementType() != dstRanked.getElementType())
      return failure();
    SmallVector<OpFoldResult> lows(rank, rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> highs(rank, rewriter.getIndexAttr(0));
    lows[axis] = rewriter.getIndexAttr(lo);
    highs[axis] = rewriter.getIndexAttr(hi);
    rewriter.replaceOpWithNewOp<tensor::PadOp>(op, dstTy, input, lows, highs, padValue);
    return success();
  }
};
struct ConstValueLowering : public RewritePattern {
  ConstValueLowering(MLIRContext *ctx) : RewritePattern("dwc.const", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1)
      return failure();
    auto value = op->getAttrOfType<DenseElementsAttr>("value");
    if (!value)
      return failure();
    Type dstTy = op->getResult(0).getType();
    auto dstRanked = dyn_cast<RankedTensorType>(dstTy);
    if (!dstRanked || !dstRanked.hasStaticShape())
      return failure();
    if (value.getType() != dstTy)
      return failure();
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, dstTy, value);
    return success();
  }
};

struct GenericConstantLowering : public RewritePattern {
  GenericConstantLowering(MLIRContext *ctx) : RewritePattern("dwc.generic_constant", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1)
      return failure();
    auto value = op->getAttrOfType<DenseElementsAttr>("value");
    if (!value)
      return failure();
    Type dstTy = op->getResult(0).getType();
    auto dstRanked = dyn_cast<RankedTensorType>(dstTy);
    if (!dstRanked || !dstRanked.hasStaticShape())
      return failure();
    if (value.getType() != dstTy)
      return failure();
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, dstTy, value);
    return success();
  }
};




struct Conv3DLowering : public RewritePattern {
  Conv3DLowering(MLIRContext *ctx) : RewritePattern("dwc.convolution_3d", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1 || op->getNumOperands() != 2)
      return failure();
    Value input = op->getOperand(0);
    Value filter = op->getOperand(1);
    Type dstTy = op->getResult(0).getType();
    auto inRanked = dyn_cast<RankedTensorType>(input.getType());
    auto filtRanked = dyn_cast<RankedTensorType>(filter.getType());
    auto dstRanked = dyn_cast<RankedTensorType>(dstTy);
    if (!inRanked || !filtRanked || !dstRanked)
      return failure();
    if (!inRanked.hasStaticShape() || !filtRanked.hasStaticShape() || !dstRanked.hasStaticShape())
      return failure();
    if (inRanked.getRank() != 5 || filtRanked.getRank() != 5 || dstRanked.getRank() != 5)
      return failure();
    if (!isa<FloatType>(inRanked.getElementType()) || inRanked.getElementType() != filtRanked.getElementType() ||
        filtRanked.getElementType() != dstRanked.getElementType())
      return failure();
    if (inRanked.getDimSize(0) != dstRanked.getDimSize(0) || inRanked.getDimSize(4) != filtRanked.getDimSize(3) ||
        filtRanked.getDimSize(4) != dstRanked.getDimSize(4))
      return failure();
    if (dstRanked.getDimSize(1) != inRanked.getDimSize(1) - filtRanked.getDimSize(0) + 1 ||
        dstRanked.getDimSize(2) != inRanked.getDimSize(2) - filtRanked.getDimSize(1) + 1 ||
        dstRanked.getDimSize(3) != inRanked.getDimSize(3) - filtRanked.getDimSize(2) + 1)
      return failure();
    Location loc = op->getLoc();
    Value empty = rewriter.create<tensor::EmptyOp>(loc, dstRanked.getShape(), dstRanked.getElementType());
    Value zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(dstRanked.getElementType()));
    Value initial = rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{empty}).getResult(0);
    auto strides = rewriter.getI64TensorAttr({1, 1, 1});
    auto dilations = rewriter.getI64TensorAttr({1, 1, 1});
    rewriter.replaceOpWithNewOp<linalg::Conv3DNdhwcDhwcfOp>(op, TypeRange{dstTy}, ValueRange{input, filter}, ValueRange{initial}, strides, dilations);
    return success();
  }
};

struct Pool2DLowering : public RewritePattern {
  StringRef root;
  Pool2DLowering(StringRef rootName, MLIRContext *ctx)
      : RewritePattern(rootName, 1, ctx), root(rootName) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    if (op->getName().getStringRef() != root)
      return failure();
    if (op->getNumResults() != 1 || op->getNumOperands() != 1)
      return failure();
    Value input = op->getOperand(0);
    Type dstTy = op->getResult(0).getType();
    auto srcRanked = dyn_cast<RankedTensorType>(input.getType());
    auto dstRanked = dyn_cast<RankedTensorType>(dstTy);
    if (!srcRanked || !dstRanked || !srcRanked.hasStaticShape() || !dstRanked.hasStaticShape())
      return failure();
    if (srcRanked.getRank() != 4 || dstRanked.getRank() != 4)
      return failure();
    if (!isa<FloatType>(srcRanked.getElementType()) || srcRanked.getElementType() != dstRanked.getElementType())
      return failure();
    if (srcRanked.getDimSize(0) != dstRanked.getDimSize(0) || srcRanked.getDimSize(3) != dstRanked.getDimSize(3))
      return failure();
    if (srcRanked.getDimSize(1) != 2 * dstRanked.getDimSize(1) || srcRanked.getDimSize(2) != 2 * dstRanked.getDimSize(2))
      return failure();
    Location loc = op->getLoc();
    Value window = rewriter.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>{2, 2}, srcRanked.getElementType());
    Value empty = rewriter.create<tensor::EmptyOp>(loc, dstRanked.getShape(), dstRanked.getElementType());
    Value minimum = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(dstRanked.getElementType(),
            APFloat::getInf(cast<FloatType>(dstRanked.getElementType()).getFloatSemantics(), true)));
    empty = rewriter.create<linalg::FillOp>(loc, ValueRange{minimum}, ValueRange{empty}).getResult(0);
    auto strides = rewriter.getI64TensorAttr({2, 2});
    auto dilations = rewriter.getI64TensorAttr({1, 1});
    rewriter.replaceOpWithNewOp<linalg::PoolingNhwcMaxOp>(op, TypeRange{dstTy}, ValueRange{input, window}, ValueRange{empty}, strides, dilations);
    return success();
  }
};
struct WalshHadamardLowering : public RewritePattern {
  WalshHadamardLowering(MLIRContext *ctx) : RewritePattern("dwc.fast_walsh_hadamard_transform", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1 || op->getNumOperands() != 1)
      return failure();
    Value input = op->getOperand(0);
    Type dstTy = op->getResult(0).getType();
    auto srcRanked = dyn_cast<RankedTensorType>(input.getType());
    auto dstRanked = dyn_cast<RankedTensorType>(dstTy);
    if (!srcRanked || !dstRanked || !srcRanked.hasStaticShape() || !dstRanked.hasStaticShape())
      return failure();
    if (srcRanked.getRank() != 1 || dstRanked.getRank() != 1)
      return failure();
    if (srcRanked.getShape() != dstRanked.getShape())
      return failure();
    if (!isa<FloatType>(srcRanked.getElementType()) || srcRanked.getElementType() != dstRanked.getElementType())
      return failure();
    int64_t n = srcRanked.getDimSize(0);
    if (n <= 0 || (n & (n - 1)) != 0)
      return failure();
    Location loc = op->getLoc();
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value nV = rewriter.create<arith::ConstantIndexOp>(loc, n);
    Value buf = rewriter.create<tensor::EmptyOp>(loc, dstRanked.getShape(), dstRanked.getElementType());
    SmallVector<AffineMap> copyMaps(2, rewriter.getMultiDimIdentityMap(1));
    SmallVector<utils::IteratorType> copyIters(1, utils::IteratorType::parallel);
    rewriter.create<linalg::GenericOp>(loc, TypeRange{}, ValueRange{input}, ValueRange{buf},
        copyMaps, copyIters,
        [&](OpBuilder &nested, Location nloc, ValueRange args) {
          nested.create<linalg::YieldOp>(nloc, args[0]);
        });
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    int64_t stages = 0;
    for (int64_t t = n; t > 1; t >>= 1)
      ++stages;
    Value stagesV = rewriter.create<arith::ConstantIndexOp>(loc, stages);
    Value step0 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    auto stageLoop = rewriter.create<scf::ForOp>(loc, c0, stagesV, c1, ValueRange{buf, step0});
    rewriter.setInsertionPointToStart(stageLoop.getBody());
    Value sbuf = stageLoop.getRegionIterArg(0);
    Value sstep = stageLoop.getRegionIterArg(1);
    auto outer = rewriter.create<scf::ForOp>(loc, c0, nV, c1, ValueRange{sbuf});
    rewriter.setInsertionPointToStart(outer.getBody());
    Value oi = outer.getInductionVar();
    Value ocur = outer.getRegionIterArg(0);
    Value pair = rewriter.create<arith::DivUIOp>(loc, oi, sstep);
    Value even = rewriter.create<arith::RemUIOp>(loc, pair, one);
    Value isEven = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, even, c0);
    Value j = rewriter.create<arith::AddIOp>(loc, oi, sstep);
    Value inRange = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, j, nV);
    Value guard = rewriter.create<arith::AndIOp>(loc, isEven, inRange);
    auto doSwap = rewriter.create<scf::IfOp>(loc, TypeRange{dstTy}, guard, true);
    rewriter.setInsertionPointToStart(&doSwap.getThenRegion().front());
    Value a = rewriter.create<tensor::ExtractOp>(loc, ocur, ValueRange{oi});
    Value b = rewriter.create<tensor::ExtractOp>(loc, ocur, ValueRange{j});
    Value s1 = rewriter.create<arith::AddFOp>(loc, a, b);
    Value d1 = rewriter.create<arith::SubFOp>(loc, a, b);
    Value u1 = rewriter.create<tensor::InsertOp>(loc, s1, ocur, ValueRange{oi});
    Value u2 = rewriter.create<tensor::InsertOp>(loc, d1, u1, ValueRange{j});
    rewriter.create<scf::YieldOp>(loc, ValueRange{u2});
    rewriter.setInsertionPointToStart(&doSwap.getElseRegion().front());
    rewriter.create<scf::YieldOp>(loc, ValueRange{ocur});
    rewriter.setInsertionPointAfter(doSwap);
    rewriter.create<scf::YieldOp>(loc, ValueRange{doSwap->getResult(0)});
    rewriter.setInsertionPointAfter(outer);
    Value nextStep = rewriter.create<arith::MulIOp>(loc, sstep, c1);
    (void)nextStep;
    Value dbl = rewriter.create<arith::AddIOp>(loc, sstep, sstep);
    rewriter.create<scf::YieldOp>(loc, ValueRange{outer->getResult(0), dbl});
    rewriter.setInsertionPointAfter(stageLoop);
    rewriter.replaceOp(op, stageLoop->getResult(0));
    return success();
  }
};


struct ImageInterpolationLowering : public RewritePattern {
  ImageInterpolationLowering(MLIRContext *ctx)
      : RewritePattern("dwc.image_interpolation", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1 || op->getNumOperands() != 1)
      return failure();
    auto algorithm = op->getAttrOfType<StringAttr>("algorithm");
    auto strideMethod = op->getAttrOfType<StringAttr>("stride_method");
    if (!algorithm || !strideMethod)
      return failure();

    bool bilinear = algorithm.getValue() == "BILINEAR";
    if (!bilinear && algorithm.getValue() != "NEAREST_NEIGHBOR")
      return failure();

    bool alignCorners = strideMethod.getValue() == "TENSORFLOW_ALIGN_CORNERS";
    bool halfPixelCenters = strideMethod.getValue() == "HALF_PIXEL_CENTERS";
    if (!alignCorners && !halfPixelCenters &&
        strideMethod.getValue() != "TENSORFLOW_DEFAULT")
      return failure();

    Value input = op->getOperand(0);
    Type dstTy = op->getResult(0).getType();
    auto srcRanked = dyn_cast<RankedTensorType>(input.getType());
    auto dstRanked = dyn_cast<RankedTensorType>(dstTy);
    if (!srcRanked || !dstRanked || !srcRanked.hasStaticShape() || !dstRanked.hasStaticShape())
      return failure();
    if (srcRanked.getRank() != 4 || dstRanked.getRank() != 4)
      return failure();
    if (srcRanked.getElementType() != dstRanked.getElementType() ||
        (bilinear && !srcRanked.getElementType().isF32()))
      return failure();
    if (srcRanked.getDimSize(0) != dstRanked.getDimSize(0) || srcRanked.getDimSize(3) != dstRanked.getDimSize(3))
      return failure();
    if (srcRanked.getDimSize(1) <= 0 || srcRanked.getDimSize(2) <= 0 ||
        dstRanked.getDimSize(1) <= 0 || dstRanked.getDimSize(2) <= 0)
      return failure();

    Location loc = op->getLoc();
    Value empty = rewriter.create<tensor::EmptyOp>(loc, dstRanked.getShape(), dstRanked.getElementType());
    SmallVector<AffineMap> maps{rewriter.getMultiDimIdentityMap(4)};
    SmallVector<utils::IteratorType> iters(4, utils::IteratorType::parallel);
    auto generic = rewriter.create<linalg::GenericOp>(loc, TypeRange{dstTy}, ValueRange{}, ValueRange{empty},
        maps, iters,
        [&](OpBuilder &nested, Location nloc, ValueRange) {
          Value batch = nested.create<linalg::IndexOp>(nloc, 0);
          Value channel = nested.create<linalg::IndexOp>(nloc, 3);
          Value zero = nested.create<arith::ConstantOp>(nloc, nested.getF32FloatAttr(0.0));
          Value one = nested.create<arith::ConstantOp>(nloc, nested.getF32FloatAttr(1.0));
          Value half = nested.create<arith::ConstantOp>(nloc, nested.getF32FloatAttr(0.5));
          SmallVector<Value> lowerIndices;
          SmallVector<Value> upperIndices;
          SmallVector<Value> fractions;

          for (int64_t axis : {1, 2}) {
            int64_t inputSize = srcRanked.getDimSize(axis);
            int64_t outputSize = dstRanked.getDimSize(axis);
            float scale = alignCorners && outputSize > 1
                ? static_cast<float>(inputSize - 1) / static_cast<float>(outputSize - 1)
                : static_cast<float>(inputSize) / static_cast<float>(outputSize);
            Value scaleValue = nested.create<arith::ConstantOp>(
                nloc, nested.getF32FloatAttr(scale));
            Value outputIndex = nested.create<linalg::IndexOp>(nloc, axis);
            Value integerIndex = nested.create<arith::IndexCastOp>(
                nloc, nested.getI64Type(), outputIndex);
            Value coordinate = nested.create<arith::SIToFPOp>(
                nloc, nested.getF32Type(), integerIndex);
            if (halfPixelCenters)
              coordinate = nested.create<arith::AddFOp>(nloc, coordinate, half);

            coordinate = nested.create<arith::MulFOp>(nloc, coordinate, scaleValue);
            if (bilinear && halfPixelCenters)
              coordinate = nested.create<arith::SubFOp>(nloc, coordinate, half);

            Value last = nested.create<arith::ConstantOp>(
                nloc, nested.getF32FloatAttr(static_cast<float>(inputSize - 1)));
            Value rounded = !bilinear && alignCorners
                ? static_cast<Value>(nested.create<math::RoundOp>(nloc, coordinate))
                : static_cast<Value>(nested.create<math::FloorOp>(nloc, coordinate));
            Value lower = nested.create<arith::MaximumFOp>(nloc, rounded, zero);
            lower = nested.create<arith::MinimumFOp>(nloc, lower, last);
            Value lowerInteger = nested.create<arith::FPToSIOp>(
                nloc, nested.getI64Type(), lower);
            lowerIndices.push_back(nested.create<arith::IndexCastOp>(
                nloc, nested.getIndexType(), lowerInteger));
            if (!bilinear)
              continue;

            Value upper = nested.create<math::CeilOp>(nloc, coordinate);
            upper = nested.create<arith::MinimumFOp>(nloc, upper, last);
            upper = nested.create<arith::MaximumFOp>(nloc, upper, zero);
            Value upperInteger = nested.create<arith::FPToSIOp>(
                nloc, nested.getI64Type(), upper);
            upperIndices.push_back(nested.create<arith::IndexCastOp>(
                nloc, nested.getIndexType(), upperInteger));
            fractions.push_back(nested.create<arith::SubFOp>(nloc, coordinate, lower));
          }

          Value topLeft = nested.create<tensor::ExtractOp>(
              nloc, input, ValueRange{batch, lowerIndices[0], lowerIndices[1], channel});
          if (!bilinear) {
            nested.create<linalg::YieldOp>(nloc, topLeft);
            return;
          }

          Value bottomLeft = nested.create<tensor::ExtractOp>(
              nloc, input, ValueRange{batch, upperIndices[0], lowerIndices[1], channel});
          Value topRight = nested.create<tensor::ExtractOp>(
              nloc, input, ValueRange{batch, lowerIndices[0], upperIndices[1], channel});
          Value bottomRight = nested.create<tensor::ExtractOp>(
              nloc, input, ValueRange{batch, upperIndices[0], upperIndices[1], channel});
          Value inverseY = nested.create<arith::SubFOp>(nloc, one, fractions[0]);
          Value inverseX = nested.create<arith::SubFOp>(nloc, one, fractions[1]);
          auto weighted = [&](Value sample, Value yWeight, Value xWeight) {
            Value product = nested.create<arith::MulFOp>(nloc, sample, yWeight);
            return nested.create<arith::MulFOp>(nloc, product, xWeight).getResult();
          };
          Value result = nested.create<arith::AddFOp>(
              nloc, weighted(topLeft, inverseY, inverseX),
              weighted(bottomLeft, fractions[0], inverseX));
          result = nested.create<arith::AddFOp>(
              nloc, result, weighted(topRight, inverseY, fractions[1]));
          result = nested.create<arith::AddFOp>(
              nloc, result, weighted(bottomRight, fractions[0], fractions[1]));
          nested.create<linalg::YieldOp>(nloc, result);
        });
    rewriter.replaceOp(op, generic->getResult(0));
    return success();
  }
};

struct Pool3DLowering : public RewritePattern {
  Pool3DLowering(MLIRContext *ctx) : RewritePattern("dwc.pooling_3d", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1 || op->getNumOperands() != 1)
      return failure();
    Value input = op->getOperand(0);
    Type dstTy = op->getResult(0).getType();
    auto srcRanked = dyn_cast<RankedTensorType>(input.getType());
    auto dstRanked = dyn_cast<RankedTensorType>(dstTy);
    if (!srcRanked || !dstRanked || !srcRanked.hasStaticShape() || !dstRanked.hasStaticShape())
      return failure();
    if (srcRanked.getRank() != 5 || dstRanked.getRank() != 5)
      return failure();
    if (!isa<FloatType>(srcRanked.getElementType()) || srcRanked.getElementType() != dstRanked.getElementType())
      return failure();
    if (srcRanked.getDimSize(0) != dstRanked.getDimSize(0) || srcRanked.getDimSize(4) != dstRanked.getDimSize(4))
      return failure();
    if (srcRanked.getDimSize(1) != 2 * dstRanked.getDimSize(1) ||
        srcRanked.getDimSize(2) != 2 * dstRanked.getDimSize(2) ||
        srcRanked.getDimSize(3) != 2 * dstRanked.getDimSize(3))
      return failure();
    Location loc = op->getLoc();
    Value window = rewriter.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>{2, 2, 2}, srcRanked.getElementType());
    Value empty = rewriter.create<tensor::EmptyOp>(loc, dstRanked.getShape(), dstRanked.getElementType());
    Value minimum = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(dstRanked.getElementType(),
            APFloat::getInf(cast<FloatType>(dstRanked.getElementType()).getFloatSemantics(), true)));
    empty = rewriter.create<linalg::FillOp>(loc, ValueRange{minimum}, ValueRange{empty}).getResult(0);
    auto strides = rewriter.getI64TensorAttr({2, 2, 2});
    auto dilations = rewriter.getI64TensorAttr({1, 1, 1});
    rewriter.replaceOpWithNewOp<linalg::PoolingNdhwcMaxOp>(op, TypeRange{dstTy}, ValueRange{input, window}, ValueRange{empty}, strides, dilations);
    return success();
  }
};

struct VicaAddPoolLowering : public RewritePattern {
  VicaAddPoolLowering(MLIRContext *ctx) : RewritePattern("dwc.vica_add_pool", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1 || (op->getNumOperands() != 1 && op->getNumOperands() != 2))
      return failure();
    Value input = op->getNumOperands() == 2 ? op->getOperand(0) : op->getOperand(0);
    Value sum = input;
    Type dstTy = op->getResult(0).getType();
    auto dstRanked = dyn_cast<RankedTensorType>(dstTy);
    auto inRanked = dyn_cast<RankedTensorType>(input.getType());
    if (!inRanked || !dstRanked || !inRanked.hasStaticShape() || !dstRanked.hasStaticShape())
      return failure();
    if (inRanked.getRank() != 4 || dstRanked.getRank() != 4)
      return failure();
    if (!isa<FloatType>(inRanked.getElementType()) ||
        inRanked.getElementType() != dstRanked.getElementType())
      return failure();
    Location loc = op->getLoc();
    if (op->getNumOperands() == 2) {
      Value lhs = op->getOperand(0);
      Value rhs = op->getOperand(1);
      auto lhsRanked = dyn_cast<RankedTensorType>(lhs.getType());
      auto rhsRanked = dyn_cast<RankedTensorType>(rhs.getType());
      if (!lhsRanked || !rhsRanked || !lhsRanked.hasStaticShape() || !rhsRanked.hasStaticShape())
        return failure();
      if (lhsRanked.getShape() != dstRanked.getShape())
        return failure();
      if (!isa<FloatType>(lhsRanked.getElementType()) || lhsRanked.getElementType() != dstRanked.getElementType())
        return failure();
      Value addEmpty = rewriter.create<tensor::EmptyOp>(loc, dstRanked.getShape(), dstRanked.getElementType());
      sum = rewriter.create<linalg::AddOp>(loc, dstTy, ValueRange{lhs, rhs}, ValueRange{addEmpty})->getResult(0);
    } else if (inRanked.getDimSize(0) != dstRanked.getDimSize(0) || inRanked.getDimSize(3) != dstRanked.getDimSize(3) ||
               inRanked.getDimSize(1) != 2 * dstRanked.getDimSize(1) || inRanked.getDimSize(2) != 2 * dstRanked.getDimSize(2)) {
      return failure();
    }
    Value window = rewriter.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>{2, 2}, dstRanked.getElementType());
    Value poolEmpty = rewriter.create<tensor::EmptyOp>(loc, dstRanked.getShape(), dstRanked.getElementType());
    Value minimum = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(dstRanked.getElementType(),
            APFloat::getInf(cast<FloatType>(dstRanked.getElementType()).getFloatSemantics(), true)));
    poolEmpty = rewriter.create<linalg::FillOp>(loc, ValueRange{minimum}, ValueRange{poolEmpty}).getResult(0);
    if (op->getNumOperands() == 1) {
      auto strides = rewriter.getI64TensorAttr({2, 2});
      auto dilations = rewriter.getI64TensorAttr({1, 1});
      rewriter.replaceOpWithNewOp<linalg::PoolingNhwcMaxOp>(op, TypeRange{dstTy}, ValueRange{sum, window}, ValueRange{poolEmpty}, strides, dilations);
      return success();
    }
    if (inRanked.getShape() == dstRanked.getShape()) {
      rewriter.replaceOp(op, sum);
      return success();
    }
    if (inRanked.getDimSize(0) != dstRanked.getDimSize(0) || inRanked.getDimSize(3) != dstRanked.getDimSize(3) ||
        inRanked.getDimSize(1) != 2 * dstRanked.getDimSize(1) || inRanked.getDimSize(2) != 2 * dstRanked.getDimSize(2))
      return failure();
    auto strides = rewriter.getI64TensorAttr({1, 1});
    auto dilations = rewriter.getI64TensorAttr({1, 1});
    rewriter.replaceOpWithNewOp<linalg::PoolingNhwcMaxOp>(op, TypeRange{dstTy}, ValueRange{sum, window}, ValueRange{poolEmpty}, strides, dilations);
    return success();
  }
};
struct VicaFusedConvLowering : public RewritePattern {
  VicaFusedConvLowering(MLIRContext *ctx) : RewritePattern("dwc.vica_fused_conv", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1 || op->getNumOperands() != 2)
      return failure();
    Value input = op->getOperand(0);
    Value filter = op->getOperand(1);
    Type dstTy = op->getResult(0).getType();
    auto inRanked = dyn_cast<RankedTensorType>(input.getType());
    auto filtRanked = dyn_cast<RankedTensorType>(filter.getType());
    auto dstRanked = dyn_cast<RankedTensorType>(dstTy);
    if (!inRanked || !filtRanked || !dstRanked)
      return failure();
    if (!inRanked.hasStaticShape() || !filtRanked.hasStaticShape() || !dstRanked.hasStaticShape())
      return failure();
    if (inRanked.getRank() != 4 || filtRanked.getRank() != 4 || dstRanked.getRank() != 4)
      return failure();
    if (!isa<FloatType>(inRanked.getElementType()) || inRanked.getElementType() != filtRanked.getElementType() ||
        filtRanked.getElementType() != dstRanked.getElementType())
      return failure();
    if (inRanked.getDimSize(0) != dstRanked.getDimSize(0) || inRanked.getDimSize(3) != filtRanked.getDimSize(2) ||
        filtRanked.getDimSize(3) != dstRanked.getDimSize(3))
      return failure();
    if (dstRanked.getDimSize(1) != inRanked.getDimSize(1) - filtRanked.getDimSize(0) + 1 ||
        dstRanked.getDimSize(2) != inRanked.getDimSize(2) - filtRanked.getDimSize(1) + 1)
      return failure();
    Location loc = op->getLoc();
    Value empty = rewriter.create<tensor::EmptyOp>(loc, dstRanked.getShape(), dstRanked.getElementType());
    Value zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(dstRanked.getElementType()));
    empty = rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{empty}).getResult(0);
    auto strides = rewriter.getI64TensorAttr({1, 1});
    auto dilations = rewriter.getI64TensorAttr({1, 1});
    rewriter.replaceOpWithNewOp<linalg::Conv2DNhwcHwcfOp>(op, TypeRange{dstTy}, ValueRange{input, filter}, ValueRange{empty}, strides, dilations);
    return success();
  }
};


struct IndexUnpoolLowering : public RewritePattern {
  IndexUnpoolLowering(MLIRContext *ctx) : RewritePattern("dwc.index_unpool", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1 || op->getNumOperands() != 2)
      return failure();
    Value input = op->getOperand(0);
    Value indices = op->getOperand(1);
    Type dstTy = op->getResult(0).getType();
    auto srcRanked = dyn_cast<RankedTensorType>(input.getType());
    auto idxRanked = dyn_cast<RankedTensorType>(indices.getType());
    auto dstRanked = dyn_cast<RankedTensorType>(dstTy);
    if (!srcRanked || !idxRanked || !dstRanked)
      return failure();
    if (!srcRanked.hasStaticShape() || !idxRanked.hasStaticShape() || !dstRanked.hasStaticShape())
      return failure();
    if (srcRanked.getRank() != 2 || dstRanked.getRank() != 2)
      return failure();
    if (idxRanked.getRank() != 1 && idxRanked.getRank() != 2)
      return failure();
    if (srcRanked.getDimSize(0) != dstRanked.getDimSize(0))
      return failure();
    if (idxRanked.getRank() == 1 && srcRanked.getDimSize(0) != idxRanked.getDimSize(0))
      return failure();
    if (srcRanked.getElementType() != dstRanked.getElementType() || !isa<FloatType>(srcRanked.getElementType()))
      return failure();
    if (!isa<IntegerType>(idxRanked.getElementType()))
      return failure();
    Location loc = op->getLoc();
    Value empty = rewriter.create<tensor::EmptyOp>(loc, dstRanked.getShape(), dstRanked.getElementType());
    Value zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(dstRanked.getElementType()));
    empty = rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{empty}).getResult(0);
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value n = rewriter.create<arith::ConstantIndexOp>(loc, srcRanked.getDimSize(0));
    Value w = rewriter.create<arith::ConstantIndexOp>(loc, srcRanked.getDimSize(1));
    auto outer = rewriter.create<scf::ForOp>(loc, c0, n, c1, ValueRange{empty});
    rewriter.setInsertionPointToStart(outer.getBody());
    Value oi = outer.getInductionVar();
    Value ocur = outer.getRegionIterArg(0);
    auto inner = rewriter.create<scf::ForOp>(loc, c0, w, c1, ValueRange{ocur});
    rewriter.setInsertionPointToStart(inner.getBody());
    Value ii = inner.getInductionVar();
    Value icur = inner.getRegionIterArg(0);
    Value v = rewriter.create<tensor::ExtractOp>(loc, input, ValueRange{oi, ii});
    Value idx = idxRanked.getRank() == 1
        ? rewriter.create<tensor::ExtractOp>(loc, indices, ValueRange{oi})
        : rewriter.create<tensor::ExtractOp>(loc, indices, ValueRange{oi, ii});
    Value col = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), idx);
    Value cur = rewriter.create<tensor::ExtractOp>(loc, icur, ValueRange{oi, col});
    Value sum = rewriter.create<arith::AddFOp>(loc, cur, v);
    Value next = rewriter.create<tensor::InsertOp>(loc, sum, icur, ValueRange{oi, col});
    rewriter.create<scf::YieldOp>(loc, ValueRange{next});
    rewriter.setInsertionPointAfter(inner);
    rewriter.create<scf::YieldOp>(loc, ValueRange{inner->getResult(0)});
    rewriter.setInsertionPointAfter(outer);
    rewriter.replaceOp(op, outer->getResult(0));
    return success();
  }
};


struct ArangeLowering : public RewritePattern {
  ArangeLowering(MLIRContext *ctx) : RewritePattern("dwc.arange", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1)
      return failure();
    Type dstTy = op->getResult(0).getType();
    auto dstRanked = dyn_cast<RankedTensorType>(dstTy);
    if (!dstRanked || !dstRanked.hasStaticShape() || dstRanked.getRank() != 1)
      return failure();
    if (!isa<IntegerType>(dstRanked.getElementType()))
      return failure();
    int64_t n = dstRanked.getDimSize(0);
    if (n <= 0)
      return failure();
    Location loc = op->getLoc();
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value nV = rewriter.create<arith::ConstantIndexOp>(loc, n);
    Value empty = rewriter.create<tensor::EmptyOp>(loc, dstRanked.getShape(), dstRanked.getElementType());
    auto loop = rewriter.create<scf::ForOp>(loc, c0, nV, c1, ValueRange{empty});
    rewriter.setInsertionPointToStart(loop.getBody());
    Value iv = loop.getInductionVar();
    Value cur = loop.getRegionIterArg(0);
    Value v = rewriter.create<arith::IndexCastOp>(loc, dstRanked.getElementType(), iv);
    Value next = rewriter.create<tensor::InsertOp>(loc, v, cur, ValueRange{iv});
    rewriter.create<scf::YieldOp>(loc, ValueRange{next});
    rewriter.setInsertionPointAfter(loop);
    rewriter.replaceOp(op, loop->getResult(0));
    return success();
  }
};

struct PackBitsLowering : public RewritePattern {
  PackBitsLowering(MLIRContext *ctx) : RewritePattern("dwc.pack_bits", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1 || op->getNumOperands() != 1)
      return failure();
    Value input = op->getOperand(0);
    Type dstTy = op->getResult(0).getType();
    auto srcRanked = dyn_cast<RankedTensorType>(input.getType());
    auto dstRanked = dyn_cast<RankedTensorType>(dstTy);
    if (!srcRanked || !dstRanked || !srcRanked.hasStaticShape() || !dstRanked.hasStaticShape())
      return failure();
    if (srcRanked.getRank() != 1 || dstRanked.getRank() != 1)
      return failure();
    auto i1 = IntegerType::get(op->getContext(), 1, IntegerType::Signless);
    if (srcRanked.getElementType() != i1 || !isa<IntegerType>(dstRanked.getElementType()) ||
        dstRanked.getElementTypeBitWidth() != 8)
      return failure();
    if (srcRanked.getDimSize(0) != 8 * dstRanked.getDimSize(0))
      return failure();
    Location loc = op->getLoc();
    Value empty = rewriter.create<tensor::EmptyOp>(loc, dstRanked.getShape(), dstRanked.getElementType());
    Value zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(dstRanked.getElementType()));
    empty = rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{empty}).getResult(0);
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value c8 = rewriter.create<arith::ConstantIndexOp>(loc, 8);
    Value n = rewriter.create<arith::ConstantIndexOp>(loc, dstRanked.getDimSize(0));
    auto outer = rewriter.create<scf::ForOp>(loc, c0, n, c1, ValueRange{empty});
    rewriter.setInsertionPointToStart(outer.getBody());
    Value oi = outer.getInductionVar();
    Value ocur = outer.getRegionIterArg(0);
    auto inner = rewriter.create<scf::ForOp>(loc, c0, c8, c1, ValueRange{ocur});
    rewriter.setInsertionPointToStart(inner.getBody());
    Value ii = inner.getInductionVar();
    Value icur = inner.getRegionIterArg(0);
    Value bitBase = rewriter.create<arith::MulIOp>(loc, oi, c8);
    Value bitIdx = rewriter.create<arith::AddIOp>(loc, bitBase, ii);
    Value bit = rewriter.create<tensor::ExtractOp>(loc, input, ValueRange{bitIdx});
    Value bitExt = rewriter.create<arith::ExtUIOp>(loc, dstRanked.getElementType(), bit);
    Value shift = rewriter.create<arith::IndexCastOp>(loc, dstRanked.getElementType(), ii);
    Value shifted = rewriter.create<arith::ShLIOp>(loc, bitExt, shift);
    Value old = rewriter.create<tensor::ExtractOp>(loc, icur, ValueRange{oi});
    Value acc = rewriter.create<arith::OrIOp>(loc, old, shifted);
    Value next = rewriter.create<tensor::InsertOp>(loc, acc, icur, ValueRange{oi});
    rewriter.create<scf::YieldOp>(loc, ValueRange{next});
    rewriter.setInsertionPointAfter(inner);
    rewriter.create<scf::YieldOp>(loc, ValueRange{inner->getResult(0)});
    rewriter.setInsertionPointAfter(outer);
    rewriter.replaceOp(op, outer->getResult(0));
    return success();
  }
};


struct PseudoSplitLowering : public RewritePattern {
  PseudoSplitLowering(MLIRContext *ctx)
      : RewritePattern("dwc.pseudo_split", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1 || op->getNumOperands() != 2)
      return failure();
    auto splits = op->getAttrOfType<IntegerAttr>("num_splits");
    if (!splits || splits.getInt() != 1)
      return failure();
    Value data = op->getOperand(1);
    if (data.getType() != op->getResult(0).getType())
      return failure();
    rewriter.replaceOp(op, data);
    return success();
  }
};


struct UnaryLowering : public RewritePattern {
  StringRef root;
  using EmitFn = Value (*)(OpBuilder &, Location, Value);
  EmitFn emit;
  UnaryLowering(StringRef rootName, EmitFn fn, MLIRContext *ctx)
      : RewritePattern(rootName, 1, ctx), root(rootName), emit(fn) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getName().getStringRef() != root)
      return failure();
    if (op->getNumResults() != 1 || op->getNumOperands() != 1)
      return failure();
    Value input = op->getOperand(0);
    Type dstTy = op->getResult(0).getType();
    auto srcRanked = dyn_cast<RankedTensorType>(input.getType());
    auto dstRanked = dyn_cast<RankedTensorType>(dstTy);
    if (!srcRanked || !dstRanked || !srcRanked.hasStaticShape() || !dstRanked.hasStaticShape())
      return failure();
    if (srcRanked.getShape() != dstRanked.getShape())
      return failure();
    if (!isa<FloatType>(srcRanked.getElementType()) || srcRanked.getElementType() != dstRanked.getElementType())
      return failure();
    Location loc = op->getLoc();
    Value empty = rewriter.create<tensor::EmptyOp>(loc, dstRanked.getShape(), dstRanked.getElementType());
    int64_t rank = dstRanked.getRank();
    SmallVector<AffineMap> maps(2, rewriter.getMultiDimIdentityMap(rank));
    SmallVector<utils::IteratorType> iters(rank, utils::IteratorType::parallel);
    auto generic = rewriter.create<linalg::GenericOp>(loc, TypeRange{dstTy}, ValueRange{input}, ValueRange{empty},
        maps, iters,
        [&](OpBuilder &nested, Location nloc, ValueRange args) {
          Value out = emit(nested, nloc, args[0]);
          nested.create<linalg::YieldOp>(nloc, out);
        });
    rewriter.replaceOp(op, generic->getResult(0));
    return success();
  }
};

struct NotLowering : public RewritePattern {
  NotLowering(MLIRContext *ctx)
      : RewritePattern("dwc.not", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1 || op->getNumOperands() != 1)
      return failure();
    Value input = op->getOperand(0);
    Type dstTy = op->getResult(0).getType();
    auto srcRanked = dyn_cast<RankedTensorType>(input.getType());
    auto dstRanked = dyn_cast<RankedTensorType>(dstTy);
    if (!srcRanked || !dstRanked || !srcRanked.hasStaticShape() || !dstRanked.hasStaticShape())
      return failure();
    if (srcRanked.getShape() != dstRanked.getShape())
      return failure();
    auto i1 = IntegerType::get(op->getContext(), 1, IntegerType::Signless);
    if (srcRanked.getElementType() != i1 || dstRanked.getElementType() != i1)
      return failure();
    Location loc = op->getLoc();
    Value empty = rewriter.create<tensor::EmptyOp>(loc, dstRanked.getShape(), dstRanked.getElementType());
    Value scalar = rewriter.create<arith::ConstantOp>(loc, rewriter.getBoolAttr(true));
    Value filled = empty;
    filled = rewriter.create<linalg::FillOp>(loc, ValueRange{scalar}, ValueRange{filled}).getResult(0);
    int64_t rank = dstRanked.getRank();
    SmallVector<AffineMap> maps(3, rewriter.getMultiDimIdentityMap(rank));
    SmallVector<utils::IteratorType> iters(rank, utils::IteratorType::parallel);
    auto generic = rewriter.create<linalg::GenericOp>(loc, TypeRange{dstTy}, ValueRange{input, filled}, ValueRange{empty},
        maps, iters,
        [&](OpBuilder &nested, Location nloc, ValueRange args) {
          Value out = nested.create<arith::XOrIOp>(nloc, args[0], args[1]);
          nested.create<linalg::YieldOp>(nloc, out);
        });
    rewriter.replaceOp(op, generic->getResult(0));
    return success();
  }
};

struct CastLowering : public RewritePattern {
  CastLowering(MLIRContext *ctx)
      : RewritePattern("dwc.cast", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1 || op->getNumOperands() != 1)
      return failure();
    Value input = op->getOperand(0);
    Type dstTy = op->getResult(0).getType();
    auto srcRanked = dyn_cast<RankedTensorType>(input.getType());
    auto dstRanked = dyn_cast<RankedTensorType>(dstTy);
    if (!srcRanked || !dstRanked || !srcRanked.hasStaticShape() || !dstRanked.hasStaticShape())
      return failure();
    if (srcRanked.getShape() != dstRanked.getShape())
      return failure();
    Type srcElem = srcRanked.getElementType();
    Type dstElem = dstRanked.getElementType();
    if (!isa<FloatType>(srcElem) || !isa<FloatType>(dstElem))
      return failure();
    if (srcElem == dstElem)
      return failure();
    Location loc = op->getLoc();
    Value empty = rewriter.create<tensor::EmptyOp>(loc, dstRanked.getShape(), dstElem);
    int64_t rank = dstRanked.getRank();
    SmallVector<AffineMap> maps(2, rewriter.getMultiDimIdentityMap(rank));
    SmallVector<utils::IteratorType> iters(rank, utils::IteratorType::parallel);
    bool srcWider = srcElem.getIntOrFloatBitWidth() > dstElem.getIntOrFloatBitWidth();
    auto generic = rewriter.create<linalg::GenericOp>(loc, TypeRange{dstTy}, ValueRange{input}, ValueRange{empty},
        maps, iters,
        [&](OpBuilder &nested, Location nloc, ValueRange args) {
          Value out = srcWider ? static_cast<Value>(nested.create<arith::TruncFOp>(nloc, dstElem, args[0]))
                               : static_cast<Value>(nested.create<arith::ExtFOp>(nloc, dstElem, args[0]));
          nested.create<linalg::YieldOp>(nloc, out);
        });
    rewriter.replaceOp(op, generic->getResult(0));
    return success();
  }
};

struct OneHotLowering : public RewritePattern {
  OneHotLowering(MLIRContext *ctx)
      : RewritePattern("dwc.one_hot", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1 || op->getNumOperands() != 1)
      return failure();
    auto axis = op->getAttrOfType<IntegerAttr>("axis");
    if (!axis)
      return failure();
    Value input = op->getOperand(0);
    Type dstTy = op->getResult(0).getType();
    auto srcRanked = dyn_cast<RankedTensorType>(input.getType());
    auto dstRanked = dyn_cast<RankedTensorType>(dstTy);
    if (!srcRanked || !dstRanked || !srcRanked.hasStaticShape() || !dstRanked.hasStaticShape())
      return failure();
    if (srcRanked.getRank() != 1 || dstRanked.getRank() != 2)
      return failure();
    if (srcRanked.getDimSize(0) != dstRanked.getDimSize(0))
      return failure();
    if (!isa<IntegerType>(srcRanked.getElementType()) || !isa<FloatType>(dstRanked.getElementType()))
      return failure();
    Location loc = op->getLoc();
    Value empty = rewriter.create<tensor::EmptyOp>(loc, dstRanked.getShape(), dstRanked.getElementType());
    Value zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(dstRanked.getElementType()));
    empty = rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{empty}).getResult(0);
    Value one = rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(dstRanked.getElementType(), 1.0));
    int64_t depth = dstRanked.getDimSize(1);
    auto generic = rewriter.create<linalg::GenericOp>(loc, TypeRange{dstTy}, ValueRange{input}, ValueRange{empty},
        SmallVector<AffineMap>{AffineMap::get(2, 0, rewriter.getAffineDimExpr(0), op->getContext()), rewriter.getMultiDimIdentityMap(2)},
        SmallVector<utils::IteratorType>{utils::IteratorType::parallel, utils::IteratorType::parallel},
        [&](OpBuilder &nested, Location nloc, ValueRange args) {
          Value idx = args[0].getType().isIndex() ? args[0] : nested.create<arith::IndexCastOp>(nloc, rewriter.getIndexType(), args[0]);
          Value col = nested.create<linalg::IndexOp>(nloc, 1);
          Value eq = nested.create<arith::CmpIOp>(nloc, arith::CmpIPredicate::eq, idx, col);
          Value pick = nested.create<arith::SelectOp>(nloc, eq, one, args[1]);
          nested.create<linalg::YieldOp>(nloc, pick);
        });
    (void)depth;
    rewriter.replaceOp(op, generic->getResult(0));
    return success();
  }
};

struct CumulativeLowering : public RewritePattern {
  CumulativeLowering(MLIRContext *ctx)
      : RewritePattern("dwc.cumulative", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1 || op->getNumOperands() != 1)
      return failure();
    auto axis = op->getAttrOfType<IntegerAttr>("axis");
    auto exclusive = op->getAttrOfType<BoolAttr>("exclusive");
    if (!axis || !exclusive || exclusive.getValue())
      return failure();
    if (axis.getInt() != 0)
      return failure();
    Value input = op->getOperand(0);
    Type dstTy = op->getResult(0).getType();
    auto srcRanked = dyn_cast<RankedTensorType>(input.getType());
    auto dstRanked = dyn_cast<RankedTensorType>(dstTy);
    if (!srcRanked || !dstRanked || !srcRanked.hasStaticShape() || !dstRanked.hasStaticShape())
      return failure();
    if (srcRanked.getRank() != 1 || dstRanked.getRank() != 1)
      return failure();
    if (srcRanked.getShape() != dstRanked.getShape())
      return failure();
    if (!isa<IntegerType>(srcRanked.getElementType()) || srcRanked.getElementType() != dstRanked.getElementType())
      return failure();
    Location loc = op->getLoc();
    Value empty = rewriter.create<tensor::EmptyOp>(loc, dstRanked.getShape(), dstRanked.getElementType());
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value n = rewriter.create<arith::ConstantIndexOp>(loc, dstRanked.getDimSize(0));
    Value acc0 = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(dstRanked.getElementType()));
    auto loop = rewriter.create<scf::ForOp>(loc, c0, n, c1, ValueRange{empty, acc0},
        [&](OpBuilder &nested, Location nloc, Value iv, ValueRange iter) {
          Value elem = nested.create<tensor::ExtractOp>(nloc, input, ValueRange{iv});
          Value acc = nested.create<arith::AddIOp>(nloc, iter[1], elem);
          Value out = nested.create<tensor::InsertOp>(nloc, acc, iter[0], ValueRange{iv});
          nested.create<scf::YieldOp>(nloc, ValueRange{out, acc});
        });
    rewriter.replaceOp(op, loop->getResult(0));
    return success();
  }
};

struct IntUnaryLowering : public RewritePattern {
  StringRef root;
  using EmitFn = Value (*)(OpBuilder &, Location, Value);
  EmitFn emit;
  IntUnaryLowering(StringRef rootName, EmitFn fn, MLIRContext *ctx)
      : RewritePattern(rootName, 1, ctx), root(rootName), emit(fn) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getName().getStringRef() != root)
      return failure();
    if (op->getNumResults() != 1 || op->getNumOperands() != 1)
      return failure();
    Value input = op->getOperand(0);
    Type dstTy = op->getResult(0).getType();
    auto srcRanked = dyn_cast<RankedTensorType>(input.getType());
    auto dstRanked = dyn_cast<RankedTensorType>(dstTy);
    if (!srcRanked || !dstRanked || !srcRanked.hasStaticShape() || !dstRanked.hasStaticShape())
      return failure();
    if (srcRanked.getShape() != dstRanked.getShape())
      return failure();
    if (!isa<IntegerType>(srcRanked.getElementType()) || srcRanked.getElementType() != dstRanked.getElementType())
      return failure();
    Location loc = op->getLoc();
    Value empty = rewriter.create<tensor::EmptyOp>(loc, dstRanked.getShape(), dstRanked.getElementType());
    int64_t rank = dstRanked.getRank();
    SmallVector<AffineMap> maps(2, rewriter.getMultiDimIdentityMap(rank));
    SmallVector<utils::IteratorType> iters(rank, utils::IteratorType::parallel);
    auto generic = rewriter.create<linalg::GenericOp>(loc, TypeRange{dstTy}, ValueRange{input}, ValueRange{empty},
        maps, iters,
        [&](OpBuilder &nested, Location nloc, ValueRange args) {
          Value out = emit(nested, nloc, args[0]);
          nested.create<linalg::YieldOp>(nloc, out);
        });
    rewriter.replaceOp(op, generic->getResult(0));
    return success();
  }
};
static Value emitPopCount(OpBuilder &b, Location loc, Value x) { return b.create<math::CtPopOp>(loc, x); }
static Value emitShl(OpBuilder &b, Location loc, Value x, Value y) { return b.create<arith::ShLIOp>(loc, x, y); }
static Value emitShrS(OpBuilder &b, Location loc, Value x, Value y) { return b.create<arith::ShRSIOp>(loc, x, y); }
static Value emitShrU(OpBuilder &b, Location loc, Value x, Value y) { return b.create<arith::ShRUIOp>(loc, x, y); }
static Value emitCtlz(OpBuilder &b, Location loc, Value x) { return b.create<math::CountLeadingZerosOp>(loc, x); }
static Value emitFloorDiv(OpBuilder &b, Location loc, Value x, Value y) {
  Value d = b.create<arith::DivFOp>(loc, x, y);
  return b.create<math::FloorOp>(loc, d);
}
static Value emitPowF(OpBuilder &b, Location loc, Value x, Value y) {
  return b.create<math::PowFOp>(loc, x, y);
}

struct BinaryGenericLowering : public RewritePattern {
  StringRef root;
  using Emit2Fn = Value (*)(OpBuilder &, Location, Value, Value);
  Emit2Fn emit;
  BinaryGenericLowering(StringRef rootName, Emit2Fn fn, MLIRContext *ctx)
      : RewritePattern(rootName, 1, ctx), root(rootName), emit(fn) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getName().getStringRef() != root)
      return failure();
    if (op->getNumResults() != 1 || op->getNumOperands() != 2)
      return failure();
    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    Type dstTy = op->getResult(0).getType();
    auto lhsRanked = dyn_cast<RankedTensorType>(lhs.getType());
    auto rhsRanked = dyn_cast<RankedTensorType>(rhs.getType());
    auto dstRanked = dyn_cast<RankedTensorType>(dstTy);
    if (!lhsRanked || !rhsRanked || !dstRanked)
      return failure();
    if (!lhsRanked.hasStaticShape() || !rhsRanked.hasStaticShape() || !dstRanked.hasStaticShape())
      return failure();
    if (lhsRanked.getShape() != rhsRanked.getShape() || rhsRanked.getShape() != dstRanked.getShape())
      return failure();
    if (!isa<FloatType>(lhsRanked.getElementType()) || lhsRanked.getElementType() != rhsRanked.getElementType() ||
        rhsRanked.getElementType() != dstRanked.getElementType())
      return failure();
    Location loc = op->getLoc();
    Value empty = rewriter.create<tensor::EmptyOp>(loc, dstRanked.getShape(), dstRanked.getElementType());
    int64_t rank = dstRanked.getRank();
    SmallVector<AffineMap> maps(3, rewriter.getMultiDimIdentityMap(rank));
    SmallVector<utils::IteratorType> iters(rank, utils::IteratorType::parallel);
    auto generic = rewriter.create<linalg::GenericOp>(loc, TypeRange{dstTy}, ValueRange{lhs, rhs}, ValueRange{empty},
        maps, iters,
        [&](OpBuilder &nested, Location nloc, ValueRange args) {
          Value out = emit(nested, nloc, args[0], args[1]);
          nested.create<linalg::YieldOp>(nloc, out);
        });
    rewriter.replaceOp(op, generic->getResult(0));
    return success();
  }
};
struct CompareLowering : public RewritePattern {
  CompareLowering(MLIRContext *ctx) : RewritePattern("dwc.compare", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1 || op->getNumOperands() != 2)
      return failure();
    auto cmpType = op->getAttrOfType<ComparisonTypeAttr>("compare_type");
    if (!cmpType)
      return failure();
    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    Type dstTy = op->getResult(0).getType();
    auto lhsRanked = dyn_cast<RankedTensorType>(lhs.getType());
    auto dstRanked = dyn_cast<RankedTensorType>(dstTy);
    if (!lhsRanked || !dstRanked || !lhsRanked.hasStaticShape() || !dstRanked.hasStaticShape())
      return failure();
    if (lhsRanked.getShape() != dstRanked.getShape())
      return failure();
    arith::CmpIPredicate intPred;
    arith::CmpFPredicate floatPred;
    switch (cmpType.getValue()) {
    case ComparisonType::Equal:
      intPred = arith::CmpIPredicate::eq;
      floatPred = arith::CmpFPredicate::OEQ;
      break;
    case ComparisonType::NotEqual:
      intPred = arith::CmpIPredicate::ne;
      floatPred = arith::CmpFPredicate::UNE;
      break;
    case ComparisonType::Greater:
      intPred = arith::CmpIPredicate::sgt;
      floatPred = arith::CmpFPredicate::OGT;
      break;
    case ComparisonType::GreaterEqual:
      intPred = arith::CmpIPredicate::sge;
      floatPred = arith::CmpFPredicate::OGE;
      break;
    case ComparisonType::Less:
      intPred = arith::CmpIPredicate::slt;
      floatPred = arith::CmpFPredicate::OLT;
      break;
    case ComparisonType::LessEqual:
      intPred = arith::CmpIPredicate::sle;
      floatPred = arith::CmpFPredicate::OLE;
      break;
    }
    bool isFloat = isa<FloatType>(lhsRanked.getElementType());
    Location loc = op->getLoc();
    Value empty = rewriter.create<tensor::EmptyOp>(loc, dstRanked.getShape(), dstRanked.getElementType());
    int64_t rank = dstRanked.getRank();
    SmallVector<AffineMap> maps(3, rewriter.getMultiDimIdentityMap(rank));
    SmallVector<utils::IteratorType> iters(rank, utils::IteratorType::parallel);
    auto generic = rewriter.create<linalg::GenericOp>(loc, TypeRange{dstTy}, ValueRange{lhs, rhs}, ValueRange{empty},
        maps, iters,
        [&](OpBuilder &nested, Location nloc, ValueRange args) {
          Value cmp = isFloat ? static_cast<Value>(nested.create<arith::CmpFOp>(nloc, floatPred, args[0], args[1]))
                              : static_cast<Value>(nested.create<arith::CmpIOp>(nloc, intPred, args[0], args[1]));
          nested.create<linalg::YieldOp>(nloc, cmp);
        });
    rewriter.replaceOp(op, generic->getResult(0));
    return success();
  }
};

struct BitwiseLowering : public RewritePattern {
  StringRef root;
  using Emit2Fn = Value (*)(OpBuilder &, Location, Value, Value);
  Emit2Fn emit;
  BitwiseLowering(StringRef rootName, Emit2Fn fn, MLIRContext *ctx)
      : RewritePattern(rootName, 1, ctx), root(rootName), emit(fn) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    if (op->getName().getStringRef() != root)
      return failure();
    if (op->getNumResults() != 1 || op->getNumOperands() != 2)
      return failure();
    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    Type dstTy = op->getResult(0).getType();
    auto lhsRanked = dyn_cast<RankedTensorType>(lhs.getType());
    auto dstRanked = dyn_cast<RankedTensorType>(dstTy);
    if (!lhsRanked || !dstRanked || !lhsRanked.hasStaticShape() || !dstRanked.hasStaticShape())
      return failure();
    if (lhsRanked.getShape() != dstRanked.getShape())
      return failure();
    if (!isa<IntegerType>(lhsRanked.getElementType()) || lhsRanked.getElementType() != dstRanked.getElementType())
      return failure();
    Location loc = op->getLoc();
    Value empty = rewriter.create<tensor::EmptyOp>(loc, dstRanked.getShape(), dstRanked.getElementType());
    int64_t rank = dstRanked.getRank();
    SmallVector<AffineMap> maps(3, rewriter.getMultiDimIdentityMap(rank));
    SmallVector<utils::IteratorType> iters(rank, utils::IteratorType::parallel);
    auto generic = rewriter.create<linalg::GenericOp>(loc, TypeRange{dstTy}, ValueRange{lhs, rhs}, ValueRange{empty},
        maps, iters,
        [&](OpBuilder &nested, Location nloc, ValueRange args) {
          Value out = emit(nested, nloc, args[0], args[1]);
          nested.create<linalg::YieldOp>(nloc, out);
        });
    rewriter.replaceOp(op, generic->getResult(0));
    return success();
  }
};
struct BatchMatrixNmsLowering : public RewritePattern {
  BatchMatrixNmsLowering(MLIRContext *ctx) : RewritePattern("dwc.batch_matrix_nms", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1 || op->getNumOperands() != 2)
      return failure();
    auto maxOut = op->getAttrOfType<IntegerAttr>("max_output_size");
    auto thresh = op->getAttrOfType<FloatAttr>("score_threshold");
    auto topk = op->getAttrOfType<IntegerAttr>("suppress_top_k");
    if (!maxOut || !thresh || !topk)
      return failure();
    if (maxOut.getInt() <= 0 || topk.getInt() <= 0)
      return failure();
    Value boxes = op->getOperand(0);
    Value scores = op->getOperand(1);
    Type dstTy = op->getResult(0).getType();
    auto boxesRanked = dyn_cast<RankedTensorType>(boxes.getType());
    auto scoresRanked = dyn_cast<RankedTensorType>(scores.getType());
    auto dstRanked = dyn_cast<RankedTensorType>(dstTy);
    if (!boxesRanked || !scoresRanked || !dstRanked)
      return failure();
    if (!boxesRanked.hasStaticShape() || !scoresRanked.hasStaticShape() || !dstRanked.hasStaticShape())
      return failure();
    if (boxesRanked.getRank() != 3 || scoresRanked.getRank() != 2 || dstRanked.getRank() != 2)
      return failure();
    if (boxesRanked.getDimSize(0) != scoresRanked.getDimSize(0) || boxesRanked.getDimSize(0) != dstRanked.getDimSize(0))
      return failure();
    if (boxesRanked.getDimSize(1) != scoresRanked.getDimSize(1))
      return failure();
    if (boxesRanked.getDimSize(2) != 4)
      return failure();
    if (dstRanked.getDimSize(1) != maxOut.getInt())
      return failure();
    if (!isa<FloatType>(boxesRanked.getElementType()) || !isa<FloatType>(scoresRanked.getElementType()) ||
        !isa<IntegerType>(dstRanked.getElementType()))
      return failure();
    Location loc = op->getLoc();
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value nB = rewriter.create<arith::ConstantIndexOp>(loc, boxesRanked.getDimSize(0));
    Value nBoxes = rewriter.create<arith::ConstantIndexOp>(loc, boxesRanked.getDimSize(1));
    Value nOut = rewriter.create<arith::ConstantIndexOp>(loc, maxOut.getInt());
    Value empty = rewriter.create<tensor::EmptyOp>(loc, dstRanked.getShape(), dstRanked.getElementType());
    Value minusOne = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(dstRanked.getElementType(), -1));
    empty = rewriter.create<linalg::FillOp>(loc, ValueRange{minusOne}, ValueRange{empty}).getResult(0);
    Value threshF = rewriter.create<arith::ConstantOp>(loc, thresh);
    auto lb = rewriter.create<scf::ForOp>(loc, c0, nB, c1, ValueRange{empty});
    rewriter.setInsertionPointToStart(lb.getBody());
    Value bi = lb.getInductionVar();
    Value bcur = lb.getRegionIterArg(0);
    auto lm = rewriter.create<scf::ForOp>(loc, c0, nOut, c1, ValueRange{bcur});
    rewriter.setInsertionPointToStart(lm.getBody());
    Value mi = lm.getInductionVar();
    Value mcur = lm.getRegionIterArg(0);
    Value best = rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(scoresRanked.getElementType(), -std::numeric_limits<float>::infinity()));
    Value bestIdx = rewriter.create<arith::ConstantIndexOp>(loc, -1);
    auto ls = rewriter.create<scf::ForOp>(loc, c0, nBoxes, c1, ValueRange{best, bestIdx, mcur});
    rewriter.setInsertionPointToStart(ls.getBody());
    Value si = ls.getInductionVar();
    Value curBest = ls.getRegionIterArg(0);
    Value curIdx = ls.getRegionIterArg(1);
    Value curOut = ls.getRegionIterArg(2);
    Value score = rewriter.create<tensor::ExtractOp>(loc, scores, ValueRange{bi, si});
    Value over = rewriter.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGE, score, threshF);
    Value taken = rewriter.create<tensor::ExtractOp>(loc, curOut, ValueRange{bi, mi});
    Value takenIdx = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), taken);
    Value already = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, si, takenIdx);
    Value cand = rewriter.create<arith::AndIOp>(loc, over, rewriter.create<arith::XOrIOp>(loc, already, rewriter.create<arith::ConstantOp>(loc, rewriter.getBoolAttr(true))));
    (void)cand;
    Value better = rewriter.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT, score, curBest);
    Value pick = rewriter.create<arith::AndIOp>(loc, over, better);
    Value nBest = rewriter.create<arith::SelectOp>(loc, pick, score, curBest);
    Value nIdx = rewriter.create<arith::SelectOp>(loc, pick, si, curIdx);
    rewriter.create<scf::YieldOp>(loc, ValueRange{nBest, nIdx, curOut});
    rewriter.setInsertionPointAfter(ls);
    Value selIdx = ls->getResult(1);
    Value has = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, selIdx, c0);
    auto put = rewriter.create<scf::IfOp>(loc, TypeRange{dstTy}, has, true);
    rewriter.setInsertionPointToStart(&put.getThenRegion().front());
    Value sel32 = rewriter.create<arith::IndexCastOp>(loc, dstRanked.getElementType(), selIdx);
    Value upd = rewriter.create<tensor::InsertOp>(loc, sel32, mcur, ValueRange{bi, mi});
    rewriter.create<scf::YieldOp>(loc, ValueRange{upd});
    rewriter.setInsertionPointToStart(&put.getElseRegion().front());
    rewriter.create<scf::YieldOp>(loc, ValueRange{mcur});
    rewriter.setInsertionPointAfter(put);
    rewriter.create<scf::YieldOp>(loc, ValueRange{put->getResult(0)});
    rewriter.setInsertionPointAfter(lm);
    rewriter.create<scf::YieldOp>(loc, ValueRange{lm->getResult(0)});
    rewriter.setInsertionPointAfter(lb);
    rewriter.replaceOp(op, lb->getResult(0));
    return success();
  }
};

struct CostVolumeLowering : public RewritePattern {
  CostVolumeLowering(MLIRContext *ctx) : RewritePattern("dwc.cost_volume", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1 || op->getNumOperands() != 2)
      return failure();
    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    Type dstTy = op->getResult(0).getType();
    auto lhsRanked = dyn_cast<RankedTensorType>(lhs.getType());
    auto rhsRanked = dyn_cast<RankedTensorType>(rhs.getType());
    auto dstRanked = dyn_cast<RankedTensorType>(dstTy);
    if (!lhsRanked || !rhsRanked || !dstRanked)
      return failure();
    if (!lhsRanked.hasStaticShape() || !rhsRanked.hasStaticShape() || !dstRanked.hasStaticShape())
      return failure();
    if (lhsRanked.getShape() != rhsRanked.getShape() || lhsRanked.getRank() != 4)
      return failure();
    if (dstRanked.getRank() != 5)
      return failure();
    for (int64_t d = 0; d < 4; ++d)
      if (dstRanked.getDimSize(d) != lhsRanked.getDimSize(d))
        return failure();
    if (!isa<FloatType>(lhsRanked.getElementType()) || lhsRanked.getElementType() != rhsRanked.getElementType() ||
        rhsRanked.getElementType() != dstRanked.getElementType())
      return failure();
    int64_t dMax = dstRanked.getDimSize(4);
    int64_t w = lhsRanked.getDimSize(2);
    if (dMax <= 0 || dMax > w)
      return failure();
    Location loc = op->getLoc();
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value nB = rewriter.create<arith::ConstantIndexOp>(loc, dstRanked.getDimSize(0));
    Value nH = rewriter.create<arith::ConstantIndexOp>(loc, dstRanked.getDimSize(1));
    Value nW = rewriter.create<arith::ConstantIndexOp>(loc, dstRanked.getDimSize(2));
    Value nD = rewriter.create<arith::ConstantIndexOp>(loc, dMax);
    Value nC = rewriter.create<arith::ConstantIndexOp>(loc, lhsRanked.getDimSize(3));
    Value empty = rewriter.create<tensor::EmptyOp>(loc, dstRanked.getShape(), dstRanked.getElementType());
    auto lb = rewriter.create<scf::ForOp>(loc, c0, nB, c1, ValueRange{empty});
    rewriter.setInsertionPointToStart(lb.getBody());
    auto lh = rewriter.create<scf::ForOp>(loc, c0, nH, c1, ValueRange{lb.getRegionIterArg(0)});
    rewriter.setInsertionPointToStart(lh.getBody());
    auto lw = rewriter.create<scf::ForOp>(loc, c0, nW, c1, ValueRange{lh.getRegionIterArg(0)});
    rewriter.setInsertionPointToStart(lw.getBody());
    auto ld = rewriter.create<scf::ForOp>(loc, c0, nD, c1, ValueRange{lw.getRegionIterArg(0)});
    rewriter.setInsertionPointToStart(ld.getBody());
    Value di = ld.getInductionVar();
    Value dcur = ld.getRegionIterArg(0);
    Value zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(dstRanked.getElementType()));
    auto lc = rewriter.create<scf::ForOp>(loc, c0, nC, c1, ValueRange{zero});
    rewriter.setInsertionPointToStart(lc.getBody());
    Value ci = lc.getInductionVar();
    Value csum = lc.getRegionIterArg(0);
    Value bi = lb.getInductionVar();
    Value hi = lh.getInductionVar();
    Value wi = lw.getInductionVar();
    Value rshift = rewriter.create<arith::AddIOp>(loc, wi, di);
    Value inR = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, rshift, rewriter.create<arith::ConstantIndexOp>(loc, w));
    Value a = rewriter.create<tensor::ExtractOp>(loc, lhs, ValueRange{bi, hi, wi, ci});
    Value b = rewriter.create<tensor::ExtractOp>(loc, rhs, ValueRange{bi, hi, rshift, ci});
    Value diff = rewriter.create<arith::SubFOp>(loc, a, b);
    Value sq = rewriter.create<arith::MulFOp>(loc, diff, diff);
    Value gated = rewriter.create<arith::SelectOp>(loc, inR, sq, zero);
    Value acc = rewriter.create<arith::AddFOp>(loc, csum, gated);
    rewriter.create<scf::YieldOp>(loc, ValueRange{acc});
    rewriter.setInsertionPointAfter(lc);
    Value upd = rewriter.create<tensor::InsertOp>(loc, lc->getResult(0), dcur, ValueRange{bi, hi, wi, di, c0});
    rewriter.create<scf::YieldOp>(loc, ValueRange{upd});
    rewriter.setInsertionPointAfter(ld);
    rewriter.create<scf::YieldOp>(loc, ValueRange{ld->getResult(0)});
    rewriter.setInsertionPointAfter(lw);
    rewriter.create<scf::YieldOp>(loc, ValueRange{lw->getResult(0)});
    rewriter.setInsertionPointAfter(lh);
    rewriter.create<scf::YieldOp>(loc, ValueRange{lh->getResult(0)});
    rewriter.setInsertionPointAfter(lb);
    rewriter.replaceOp(op, lb->getResult(0));
    return success();
  }
};

static Value emitAnd(OpBuilder &b, Location loc, Value x, Value y) { return b.create<arith::AndIOp>(loc, x, y); }
static Value emitOr(OpBuilder &b, Location loc, Value x, Value y) { return b.create<arith::OrIOp>(loc, x, y); }
static Value emitXor(OpBuilder &b, Location loc, Value x, Value y) { return b.create<arith::XOrIOp>(loc, x, y); }
struct BitSelectLowering : public RewritePattern {
  BitSelectLowering(MLIRContext *ctx) : RewritePattern("dwc.bit_select", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1 || op->getNumOperands() != 3)
      return failure();
    Value cond = op->getOperand(0);
    Value lhs = op->getOperand(1);
    Value rhs = op->getOperand(2);
    Type dstTy = op->getResult(0).getType();
    auto condRanked = dyn_cast<RankedTensorType>(cond.getType());
    auto dstRanked = dyn_cast<RankedTensorType>(dstTy);
    if (condRanked.getShape() != dstRanked.getShape())
      return failure();
    auto i1 = IntegerType::get(op->getContext(), 1, IntegerType::Signless);
    if (condRanked.getElementType() != i1 && condRanked.getElementType() != dstRanked.getElementType())
      return failure();
    if (!isa<IntegerType>(dstRanked.getElementType()))
      return failure();
    Location loc = op->getLoc();
    Value empty = rewriter.create<tensor::EmptyOp>(loc, dstRanked.getShape(), dstRanked.getElementType());
    int64_t rank = dstRanked.getRank();
    SmallVector<AffineMap> maps(4, rewriter.getMultiDimIdentityMap(rank));
    SmallVector<utils::IteratorType> iters(rank, utils::IteratorType::parallel);
    auto generic = rewriter.create<linalg::GenericOp>(loc, TypeRange{dstTy}, ValueRange{cond, lhs, rhs}, ValueRange{empty},
        maps, iters,
        [&](OpBuilder &nested, Location nloc, ValueRange args) {
          Value out = nested.create<arith::SelectOp>(nloc, args[0], args[1], args[2]);
          nested.create<linalg::YieldOp>(nloc, out);
        });
    rewriter.replaceOp(op, generic->getResult(0));
    return success();
  }
};

struct IsFiniteLowering : public RewritePattern {
  IsFiniteLowering(MLIRContext *ctx) : RewritePattern("dwc.is_finite", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1 || op->getNumOperands() != 1)
      return failure();
    Value input = op->getOperand(0);
    Type dstTy = op->getResult(0).getType();
    auto srcRanked = dyn_cast<RankedTensorType>(input.getType());
    auto dstRanked = dyn_cast<RankedTensorType>(dstTy);
    if (!srcRanked || !dstRanked || !srcRanked.hasStaticShape() || !dstRanked.hasStaticShape())
      return failure();
    if (srcRanked.getShape() != dstRanked.getShape())
      return failure();
    if (!isa<FloatType>(srcRanked.getElementType()))
      return failure();
    auto i1 = IntegerType::get(op->getContext(), 1, IntegerType::Signless);
    if (dstRanked.getElementType() != i1)
      return failure();
    Location loc = op->getLoc();
    Value empty = rewriter.create<tensor::EmptyOp>(loc, dstRanked.getShape(), dstRanked.getElementType());
    int64_t rank = dstRanked.getRank();
    SmallVector<AffineMap> maps(2, rewriter.getMultiDimIdentityMap(rank));
    SmallVector<utils::IteratorType> iters(rank, utils::IteratorType::parallel);
    auto generic = rewriter.create<linalg::GenericOp>(loc, TypeRange{dstTy}, ValueRange{input}, ValueRange{empty},
        maps, iters,
        [&](OpBuilder &nested, Location nloc, ValueRange args) {
          Value out = nested.create<math::IsFiniteOp>(nloc, args[0]);
          nested.create<linalg::YieldOp>(nloc, out);
        });
    rewriter.replaceOp(op, generic->getResult(0));
    return success();
  }
};


struct IdentityLowering : public RewritePattern {
  IdentityLowering(StringRef rootName, MLIRContext *ctx) : RewritePattern(rootName, 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1 || op->getNumOperands() != 1 ||
        op->getNumRegions() != 0 || op->getNumSuccessors() != 0)
      return failure();
    Value input = op->getOperand(0);
    if (input.getType() != op->getResult(0).getType())
      return failure();
    rewriter.replaceOp(op, input);
    return success();
  }
};

struct FenceEraser : public RewritePattern {
  FenceEraser(MLIRContext *ctx) : RewritePattern("darwinn.fence", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 0 || op->getNumOperands() != 0)
      return failure();
    rewriter.eraseOp(op);
    return success();
  }
};


struct ClampLowering : public RewritePattern {
  ClampLowering(MLIRContext *ctx) : RewritePattern("dwc.clamp", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1 || op->getNumOperands() != 3)
      return failure();
    Value input = op->getOperand(0);
    Value lo = op->getOperand(1);
    Value hi = op->getOperand(2);
    Type dstTy = op->getResult(0).getType();
    auto dstRanked = dyn_cast<RankedTensorType>(dstTy);
    if (!dstRanked || !dstRanked.hasStaticShape())
      return failure();
    if (!hasSameElementType(input.getType(), dstTy) ||
        !hasSameElementType(lo.getType(), dstTy) || !hasSameElementType(hi.getType(), dstTy))
      return failure();
    Location loc = op->getLoc();
    Value loEmpty = rewriter.create<tensor::EmptyOp>(loc, dstRanked.getShape(), dstRanked.getElementType());
    Value geLo = rewriter.create<linalg::MaxOp>(loc, dstTy, ValueRange{lo, input}, ValueRange{loEmpty})->getResult(0);
    Value hiEmpty = rewriter.create<tensor::EmptyOp>(loc, dstRanked.getShape(), dstRanked.getElementType());
    Value clamped = rewriter.create<linalg::MinOp>(loc, dstTy, ValueRange{geLo, hi}, ValueRange{hiEmpty})->getResult(0);
    rewriter.replaceOp(op, clamped);
    return success();
  }
};
struct ReverseLowering : public RewritePattern {
  ReverseLowering(MLIRContext *ctx) : RewritePattern("dwc.reverse", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1 || op->getNumOperands() != 1)
      return failure();
    auto dim = op->getAttrOfType<IntegerAttr>("dimension");
    if (!dim)
      return failure();
    int64_t axis = dim.getInt();
    Value input = op->getOperand(0);
    Type dstTy = op->getResult(0).getType();
    auto srcRanked = dyn_cast<RankedTensorType>(input.getType());
    auto dstRanked = dyn_cast<RankedTensorType>(dstTy);
    if (!srcRanked || !dstRanked || !srcRanked.hasStaticShape() || !dstRanked.hasStaticShape())
      return failure();
    if (srcRanked.getShape() != dstRanked.getShape())
      return failure();
    int64_t rank = srcRanked.getRank();
    if (axis < 0 || axis >= rank)
      return failure();
    if (srcRanked.getElementType() != dstRanked.getElementType())
      return failure();
    Location loc = op->getLoc();
    Value empty = rewriter.create<tensor::EmptyOp>(loc, dstRanked.getShape(), dstRanked.getElementType());
    SmallVector<AffineMap> maps;
    SmallVector<AffineExpr> flipDims;
    for (int64_t d = 0; d < rank; ++d) {
      if (d == axis)
        flipDims.push_back(rewriter.getAffineConstantExpr(srcRanked.getDimSize(d) - 1) - rewriter.getAffineDimExpr(d));
      else
        flipDims.push_back(rewriter.getAffineDimExpr(d));
    }
    maps.push_back(AffineMap::get(rank, 0, flipDims, op->getContext()));
    maps.push_back(rewriter.getMultiDimIdentityMap(rank));
    SmallVector<utils::IteratorType> iters(rank, utils::IteratorType::parallel);
    auto generic = rewriter.create<linalg::GenericOp>(loc, TypeRange{dstTy}, ValueRange{input}, ValueRange{empty},
        maps, iters,
        [&](OpBuilder &nested, Location nloc, ValueRange args) {
          nested.create<linalg::YieldOp>(nloc, args[0]);
        });
    rewriter.replaceOp(op, generic->getResult(0));
    return success();
  }
};
struct BitcastLowering : public RewritePattern {
  BitcastLowering(MLIRContext *ctx) : RewritePattern("dwc.bitcast", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1 || op->getNumOperands() != 1)
      return failure();
    auto outTy = op->getAttrOfType<TypeAttr>("output_element_type");
    if (!outTy)
      return failure();
    Value input = op->getOperand(0);
    Type dstTy = op->getResult(0).getType();
    auto srcRanked = dyn_cast<RankedTensorType>(input.getType());
    auto dstRanked = dyn_cast<RankedTensorType>(dstTy);
    if (!srcRanked || !dstRanked || !srcRanked.hasStaticShape() || !dstRanked.hasStaticShape())
      return failure();
    if (srcRanked.getShape() != dstRanked.getShape())
      return failure();
    if (dstRanked.getElementType() != outTy.getValue())
      return failure();
    if (srcRanked.getElementTypeBitWidth() != dstRanked.getElementTypeBitWidth())
      return failure();
    rewriter.replaceOpWithNewOp<tensor::BitcastOp>(op, dstTy, input);
    return success();
  }
};


static Value emitSin(OpBuilder &b, Location loc, Value x) { return b.create<math::SinOp>(loc, x); }
static Value emitCos(OpBuilder &b, Location loc, Value x) { return b.create<math::CosOp>(loc, x); }
static Value emitExp(OpBuilder &b, Location loc, Value x) { return b.create<math::ExpOp>(loc, x); }
static Value emitLog(OpBuilder &b, Location loc, Value x) { return b.create<math::LogOp>(loc, x); }
static Value emitLogistic(OpBuilder &builder, Location loc, Value input) {
  Value one = builder.create<arith::ConstantOp>(
      loc, builder.getFloatAttr(input.getType(), 1.0));
  Value negative = builder.create<arith::NegFOp>(loc, input);
  Value exponential = builder.create<math::ExpOp>(loc, negative);
  Value denominator = builder.create<arith::AddFOp>(loc, one, exponential);
  return builder.create<arith::DivFOp>(loc, one, denominator);
}
static Value emitSqrt(OpBuilder &b, Location loc, Value x) { return b.create<math::SqrtOp>(loc, x); }
static Value emitRsqrt(OpBuilder &b, Location loc, Value x) { return b.create<math::RsqrtOp>(loc, x); }
static Value emitTanh(OpBuilder &b, Location loc, Value x) { return b.create<math::TanhOp>(loc, x); }
static Value emitNeg(OpBuilder &b, Location loc, Value x) { return b.create<arith::NegFOp>(loc, x); }
static Value emitAbs(OpBuilder &b, Location loc, Value x) { return b.create<math::AbsFOp>(loc, x); }
static Value emitCeil(OpBuilder &b, Location loc, Value x) { return b.create<math::CeilOp>(loc, x); }
static Value emitFloor(OpBuilder &b, Location loc, Value x) { return b.create<math::FloorOp>(loc, x); }
static Value emitRound(OpBuilder &b, Location loc, Value x) { return b.create<math::RoundOp>(loc, x); }
static Value emitAtan(OpBuilder &b, Location loc, Value x) { return b.create<math::AtanOp>(loc, x); }
static Value emitErf(OpBuilder &b, Location loc, Value x) { return b.create<math::ErfOp>(loc, x); }
static Value emitTan(OpBuilder &b, Location loc, Value x) { return b.create<math::TanOp>(loc, x); }
static Value emitExpm1(OpBuilder &b, Location loc, Value x) { return b.create<math::ExpM1Op>(loc, x); }
static Value emitLog1p(OpBuilder &b, Location loc, Value x) { return b.create<math::Log1pOp>(loc, x); }
static Value emitSign(OpBuilder &b, Location loc, Value x) { return b.create<math::CopySignOp>(loc, b.create<arith::ConstantOp>(loc, b.getF32FloatAttr(1.0)), x); }
static Value emitRoundAfz(OpBuilder &b, Location loc, Value x) { return b.create<math::RoundEvenOp>(loc, x); }
static Value emitCbrt(OpBuilder &b, Location loc, Value x) { return b.create<math::CbrtOp>(loc, x); }
static Value emitRemF(OpBuilder &b, Location loc, Value x, Value y) { return b.create<arith::RemFOp>(loc, x, y); }
static Value emitAtan2(OpBuilder &b, Location loc, Value x, Value y) { return b.create<math::Atan2Op>(loc, x, y); }
} // namespace

void mlir::darwinn::populateLowerCopySlicePatterns(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
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
  patterns.add<SelectLowering>(ctx);
  patterns.add<DarwinnReluLowering>(ctx);
  patterns.add<DarwinnSelectLowering>(ctx);
  patterns.add<DarwinnScatterLowering>(ctx);
  patterns.add<DarwinnSplitLowering>(ctx);
  patterns.add<DarwinnScalarArithLowering>("darwinn.scalar.fadd", ctx);
  patterns.add<DarwinnScalarCmpLowering>("darwinn.scalar.feq", arith::CmpFPredicate::OEQ, ctx);
  patterns.add<DarwinnScalarCmpLowering>("darwinn.scalar.fgt", arith::CmpFPredicate::OGT, ctx);
  patterns.add<DarwinnScalarCmpLowering>("darwinn.scalar.fgte", arith::CmpFPredicate::OGE, ctx);
  patterns.add<DarwinnScalarCmpLowering>("darwinn.scalar.flt", arith::CmpFPredicate::OLT, ctx);
  patterns.add<DarwinnScalarCmpLowering>("darwinn.scalar.flte", arith::CmpFPredicate::OLE, ctx);
  patterns.add<DarwinnBinaryMapLowering>(ctx);
  patterns.add<DarwinnResidualAddLowering>(ctx);
  patterns.add<FenceEraser>(ctx);
  patterns.add<ReductionLowering>(ctx);
  patterns.add<CompareLowering>(ctx);
  patterns.add<BroadcastLowering>(ctx);
  patterns.add<NotLowering>(ctx);
  patterns.add<DwcAddLowering>(ctx);
  patterns.add<DwcBinaryLowering>("dwc.multiply", lowerDwcBinaryOp<linalg::MulOp>, ctx);
  patterns.add<DwcBinaryLowering>("dwc.divide", lowerDwcBinaryOp<linalg::DivOp>, ctx);
  patterns.add<DwcBinaryLowering>("dwc.maximum", lowerDwcBinaryOp<linalg::MaxOp>, ctx);
  patterns.add<DwcBinaryLowering>("dwc.minimum", lowerDwcBinaryOp<linalg::MinOp>, ctx);
  patterns.add<DwcBinaryLowering>("dwc.subtract", lowerDwcBinaryOp<linalg::SubOp>, ctx);
  patterns.add<CastLowering>(ctx);
  patterns.add<ReshapeLowering>(ctx);
  patterns.add<TransposeLowering>(ctx);
  patterns.add<MatmulLowering>("dwc.matrix_multiply", ctx);
  patterns.add<MatmulLowering>("dwc.fully_connected", ctx);
  patterns.add<MatmulLowering>("dwc.fully_connected_sub_channel", ctx);
  patterns.add<MatmulLowering>("dwc.fully_connected_sub_channel_v2", ctx);
  patterns.add<MatmulLowering>("dwc.matrix_multiply_sub_channel", ctx);
  patterns.add<MatmulLowering>("dwc.sparse_fully_connected", ctx);
  patterns.add<MatmulLowering>("dwc.sparse_fully_connected_sub_byte_param", ctx);
  patterns.add<InterleaveLowering>(ctx);
  patterns.add<VicaAddLowering>(ctx);
  patterns.add<ForwardLowering>("dwc.gather_operation", "dive_vm.gather", ctx);
  patterns.add<ForwardLowering>("dwc.tensor_op_gather", "dive_vm.gather", ctx);
  patterns.add<ForwardLowering>("dwc.tensor_ls_gather", "dive_vm.gather", ctx);
  patterns.add<ForwardLowering>("dwc.dma_op_gather", "dive_vm.gather", ctx);
  patterns.add<ForwardLowering>("dwc.scatter_operation", "dive_vm.scatter_nd", ctx);
  patterns.add<ForwardLowering>("dwc.tensor_ls_scatter", "dive_vm.scatter_nd", ctx);
  patterns.add<ForwardLowering>("dwc.tensor_ls_scatter_operation", "dive_vm.scatter_nd", ctx);
  patterns.add<ConvolutionLowering>("dwc.convolution", false, ctx);
  patterns.add<ConvolutionLowering>("dwc.depthwise_convolution", true, ctx);
  patterns.add<Conv3DLowering>(ctx);
  patterns.add<Pool2DLowering>("dwc.pool", ctx);
  patterns.add<Pool2DLowering>("dwc.pooling", ctx);
  patterns.add<Pool2DLowering>("dwc.reduce_window", ctx);
  patterns.add<Pool3DLowering>(ctx);
  patterns.add<WalshHadamardLowering>(ctx);
  patterns.add<ImageInterpolationLowering>(ctx);
  patterns.add<ForwardLowering>("dwc.multinormal", "dive_vm.multinomial", ctx);
  patterns.add<ForwardLowering>("dwc.uniform_random_number_generation", "dive_vm.multinomial", ctx);
  patterns.add<VicaAddPoolLowering>(ctx);
  patterns.add<VicaFusedConvLowering>(ctx);
  patterns.add<IndexUnpoolLowering>(ctx);
  patterns.add<ForwardLowering>("dwc.vica_custom_padding", "dive_vm.pad", ctx);
  patterns.add<SubByteMatmulLowering>("dwc.fully_connected_sub_byte_param", ctx);
  patterns.add<MatmulLowering>("dwc.fully_connected_zin_indexed", ctx);
  patterns.add<MatmulLowering>("dwc.fully_connected_zout_indexed", ctx);
  patterns.add<ForwardLowering>("dwc.mask_indices", "dive_vm.mask_indices", ctx);
  patterns.add<ForwardLowering>("dwc.hib_gather", "dive_vm.hib_gather_edit", ctx);
  patterns.add<ForwardLowering>("dwc.one_hot_tpu", "dive_vm.one_hot", ctx);
  patterns.add<ForwardLowering>("dwc.generic_scatter", "dive_vm.scatter_nd", ctx);
  patterns.add<ConvolutionLowering>("dwc.convolution_v2", false, ctx);
  patterns.add<ConvolutionLowering>("dwc.depthwise_convolution_v2", true, ctx);
  patterns.add<ScalarLowering>(ctx);
  patterns.add<ClassifierLowering>(ctx);
  patterns.add<GenericConvLowering>(ctx);
  patterns.add<TransposedConvLowering>(ctx);
  patterns.add<GenericDotLowering>(ctx);
  patterns.add<TruncateFloatsLowering>(ctx);
  patterns.add<ConcatenationLowering>(ctx);
  patterns.add<SliceLowering>(ctx);
  patterns.add<DynamicSliceNdLowering>("dwc.dynamic_slice", ctx);
  patterns.add<DynamicUpdateSliceNdLowering>(ctx);
  patterns.add<ForwardLowering>("dwc.sort", "dive_vm.top_k", ctx);
  patterns.add<PaddingLowering>(ctx);
  patterns.add<UnaryLowering>("dwc.sin", emitSin, ctx);
  patterns.add<UnaryLowering>("dwc.cos", emitCos, ctx);
  patterns.add<UnaryLowering>("dwc.exp", emitExp, ctx);
  patterns.add<UnaryLowering>("dwc.logistic", emitLogistic, ctx);
  patterns.add<UnaryLowering>("dwc.sqrt", emitSqrt, ctx);
  patterns.add<UnaryLowering>("dwc.rsqrt", emitRsqrt, ctx);
  patterns.add<UnaryLowering>("dwc.tanh", emitTanh, ctx);
  patterns.add<UnaryLowering>("dwc.negate", emitNeg, ctx);
  patterns.add<UnaryLowering>("dwc.abs", emitAbs, ctx);
  patterns.add<UnaryLowering>("dwc.ceil", emitCeil, ctx);
  patterns.add<UnaryLowering>("dwc.floor", emitFloor, ctx);
  patterns.add<UnaryLowering>("dwc.round", emitRound, ctx);
  patterns.add<UnaryLowering>("dwc.atan", emitAtan, ctx);
  patterns.add<UnaryLowering>("dwc.erf", emitErf, ctx);
  patterns.add<UnaryLowering>("dwc.tan", emitTan, ctx);
  patterns.add<UnaryLowering>("dwc.expm1", emitExpm1, ctx);
  patterns.add<UnaryLowering>("dwc.log1p", emitLog1p, ctx);
  patterns.add<UnaryLowering>("dwc.sign", emitSign, ctx);
  patterns.add<UnaryLowering>("dwc.round_nearest_afz", emitRoundAfz, ctx);
  patterns.add<UnaryLowering>("dwc.log", emitLog, ctx);
  patterns.add<UnaryLowering>("dwc.cbrt", emitCbrt, ctx);
  patterns.add<BinaryGenericLowering>("dwc.remainder", emitRemF, ctx);
  patterns.add<BinaryGenericLowering>("dwc.atan2", emitAtan2, ctx);
  patterns.add<BinaryGenericLowering>("dwc.floor_div", emitFloorDiv, ctx);
  patterns.add<BinaryGenericLowering>("dwc.pow", emitPowF, ctx);
  patterns.add<IntUnaryLowering>("dwc.pop_count", emitPopCount, ctx);
  patterns.add<OneHotLowering>(ctx);
  patterns.add<CumulativeLowering>(ctx);
  patterns.add<PseudoSplitLowering>(ctx);
  patterns.add<ArangeLowering>(ctx);
  patterns.add<PackBitsLowering>(ctx);
  patterns.add<ForwardLowering>("dwc.statistical_top_k", "dive_vm.top_k", ctx);
  patterns.add<ForwardLowering>("dwc.multinomial", "dive_vm.multinomial", ctx);
  patterns.add<BitwiseLowering>("dwc.and", emitAnd, ctx);
  patterns.add<BitwiseLowering>("dwc.or", emitOr, ctx);
  patterns.add<BitwiseLowering>("dwc.xor", emitXor, ctx);
  patterns.add<BitSelectLowering>(ctx);
  patterns.add<BatchMatrixNmsLowering>(ctx);
  patterns.add<CostVolumeLowering>(ctx);
  patterns.add<ConstValueLowering>(ctx);
  patterns.add<GenericConstantLowering>(ctx);
  patterns.add<BitwiseLowering>("dwc.shift_left", emitShl, ctx);
  patterns.add<BitwiseLowering>("dwc.shift_right_arithmetic", emitShrS, ctx);
  patterns.add<BitwiseLowering>("dwc.shift_right_logical", emitShrU, ctx);
  patterns.add<IsFiniteLowering>(ctx);
  patterns.add<IntUnaryLowering>("dwc.count_leading_zeros", emitCtlz, ctx);
  patterns.add<ClampLowering>(ctx);
  patterns.add<IdentityLowering>("dwc.identity", ctx);
  patterns.add<ReverseLowering>(ctx);
  patterns.add<BitcastLowering>(ctx);
  patterns.add<ForwardLowering>("dwc.gather", "dive_vm.gather", ctx);
  patterns.add<ForwardLowering>("dwc.gather_nd", "dive_vm.gather_nd", ctx);
  patterns.add<ForwardLowering>("dwc.scatter_nd", "dive_vm.scatter_nd", ctx);
  patterns.add<ForwardLowering>("dwc.pad", "dive_vm.pad", ctx);
  patterns.add<ForwardLowering>("dwc.roll", "dive_vm.roll", ctx);
  patterns.add<ForwardLowering>("dwc.top_k", "dive_vm.top_k", ctx);
  patterns.add<ForwardAnyLowering>("darwinn.vica_custom_padding", "dive_vm.pad", ctx);
  patterns.add<ForwardAnyLowering>("darwinn.mesh_pad_slice", "dive_vm.pad", ctx);
  patterns.add<ForwardAnyLowering>("darwinn.mask_indices", "dive_vm.mask_indices", ctx);
  patterns.add<Slice1dLowering>("darwinn.slice_1d_extent", ctx);
  patterns.add<Slice1dLowering>("darwinn.slice_1d_extent_with_padding_info", ctx);
  patterns.add<DiveRefReductionLowering>(ctx);
  patterns.add<ForwardLowering>("darwinn.select", "dive_vm.select", ctx);
  patterns.add<ForwardResultsLowering>("darwinn.index_filter", "dive_vm.top_k", ctx);
  patterns.add<UnsortedSegmentReduceLowering>(ctx);
}
