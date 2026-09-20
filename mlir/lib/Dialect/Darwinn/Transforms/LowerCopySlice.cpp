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

struct DarwinnFillLowering : public RewritePattern {
  DarwinnFillLowering(MLIRContext *ctx)
      : RewritePattern("darwinn.fill", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1 || op->getNumOperands() != 1)
      return failure();
    Value scalar = op->getOperand(0);
    Type dstTy = op->getResult(0).getType();
    auto dstRanked = dyn_cast<RankedTensorType>(dstTy);
    if (!dstRanked || !dstRanked.hasStaticShape())
      return failure();
    Type elem = dstRanked.getElementType();
    Type scalarTy = scalar.getType();
    if (auto shaped = dyn_cast<ShapedType>(scalarTy)) {
      if (shaped.getElementType() != elem)
        return failure();
    } else if (scalarTy != elem) {
      return failure();
    }
    Location loc = op->getLoc();
    Value empty = rewriter.create<tensor::EmptyOp>(loc, dstRanked.getShape(), elem);
    rewriter.replaceOpWithNewOp<linalg::FillOp>(op, TypeRange{dstTy}, ValueRange{scalar}, ValueRange{empty});
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

struct ForwardLowering : public RewritePattern {
  StringRef root;
  StringRef target;
  ForwardLowering(StringRef rootName, StringRef targetName, MLIRContext *ctx)
      : RewritePattern(rootName, 1, ctx), root(rootName), target(targetName) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    if (op->getName().getStringRef() != root)
      return failure();
    if (op->getNumResults() != 1 || op->getNumOperands() < 2)
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
    Value zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(dstRanked.getElementType()));
    Value acc = rewriter.create<tensor::EmptyOp>(loc, dstRanked.getShape(), dstRanked.getElementType());
    rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{acc});
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
        break;
      }
      bool isCompare = intPred == arith::CmpIPredicate::eq || intPred == arith::CmpIPredicate::ne ||
          intPred == arith::CmpIPredicate::sgt || intPred == arith::CmpIPredicate::sge ||
          intPred == arith::CmpIPredicate::slt || intPred == arith::CmpIPredicate::sle;
      if (!isCompare) {
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
          rewriter.create<linalg::FillOp>(loc, ValueRange{zeroScalar}, ValueRange{zeroBuf});
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
    if (srcRanked.getNumElements() != dstRanked.getNumElements())
      return failure();
    if (srcRanked.getRank() > dstRanked.getRank()) {
      SmallVector<ReassociationIndices> reassoc(dstRanked.getRank());
      for (int64_t d = 0; d < srcRanked.getRank(); ++d)
        reassoc[d < dstRanked.getRank() ? d : dstRanked.getRank() - 1].push_back(d);
      rewriter.replaceOpWithNewOp<tensor::CollapseShapeOp>(op, dstTy, input, reassoc);
      return success();
    }
    if (srcRanked.getRank() < dstRanked.getRank())
      return failure();
    SmallVector<ReassociationIndices> reassoc;
    for (int64_t d = 0; d < srcRanked.getRank(); ++d)
      reassoc.push_back(ReassociationIndices{d});
    rewriter.replaceOpWithNewOp<tensor::CollapseShapeOp>(op, dstTy, input, reassoc);
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
    if (!isa<FloatType>(lhsRanked.getElementType()) || lhsRanked.getElementType() != rhsRanked.getElementType() ||
        rhsRanked.getElementType() != dstRanked.getElementType())
      return failure();
    Location loc = op->getLoc();
    Value empty = rewriter.create<tensor::EmptyOp>(loc, dstRanked.getShape(), dstRanked.getElementType());
    Value zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(dstRanked.getElementType()));
    rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{empty});
    rewriter.replaceOpWithNewOp<linalg::MatmulOp>(op, TypeRange{dstTy}, ValueRange{lhs, rhs}, ValueRange{empty});
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
    rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{empty});
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
    Location loc = op->getLoc();
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
    Location loc = op->getLoc();
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
    Location loc = op->getLoc();
    OpFoldResult offset = rewriter.getIndexAttr(0);
    OpFoldResult extent = rewriter.getIndexAttr(updRanked.getDimSize(0));
    OpFoldResult stride = rewriter.getIndexAttr(1);
    rewriter.replaceOpWithNewOp<tensor::InsertSliceOp>(op, update, base, ArrayRef<OpFoldResult>{offset}, ArrayRef<OpFoldResult>{extent}, ArrayRef<OpFoldResult>{stride});
    return success();
  }
};


struct ConvolutionLowering : public RewritePattern {
  ConvolutionLowering(MLIRContext *ctx)
      : RewritePattern("dwc.convolution", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1 || op->getNumOperands() != 2)
      return failure();
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
    if (xStride != 1 || yStride != 1 || xDilation != 1 || yDilation != 1)
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
    rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{empty});
    auto strides = rewriter.getDenseI64ArrayAttr({1, 1});
    auto dilations = rewriter.getDenseI64ArrayAttr({1, 1});
    rewriter.replaceOpWithNewOp<linalg::Conv2DNhwcHwcfOp>(op, TypeRange{dstTy}, ValueRange{input, filter}, ValueRange{empty}, strides, dilations);
    return success();
  }
};
struct ConvolutionV2Lowering : public RewritePattern {
  StringRef root;
  ConvolutionV2Lowering(StringRef rootName, MLIRContext *ctx)
      : RewritePattern(rootName, 1, ctx), root(rootName) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    if (op->getName().getStringRef() != root)
      return failure();
    if (op->getNumResults() != 1 || op->getNumOperands() != 2)
      return failure();
    auto xs = op->getAttrOfType<IntegerAttr>("x_stride");
    auto ys = op->getAttrOfType<IntegerAttr>("y_stride");
    auto xd = op->getAttrOfType<IntegerAttr>("x_dilation_rate");
    auto yd = op->getAttrOfType<IntegerAttr>("y_dilation_rate");
    if (xs || ys || xd || yd) {
      if (!xs || !ys || !xd || !yd)
        return failure();
      if (xs.getInt() != 1 || ys.getInt() != 1 || xd.getInt() != 1 || yd.getInt() != 1)
        return failure();
    }
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
    if (inRanked.getRank() != 4 || dstRanked.getRank() != 4)
      return failure();
    if (!isa<FloatType>(inRanked.getElementType()) || inRanked.getElementType() != filtRanked.getElementType() ||
        filtRanked.getElementType() != dstRanked.getElementType())
      return failure();
    Location loc = op->getLoc();
    Value empty = rewriter.create<tensor::EmptyOp>(loc, dstRanked.getShape(), dstRanked.getElementType());
    Value zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(dstRanked.getElementType()));
    rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{empty});
    auto strides = rewriter.getDenseI64ArrayAttr({1, 1});
    auto dilations = rewriter.getDenseI64ArrayAttr({1, 1});
    if (root == "dwc.depthwise_convolution_v2") {
      auto mult = op->getAttrOfType<IntegerAttr>("depth_multiplier");
      if (!mult || mult.getInt() != 1)
        return failure();
      if (filtRanked.getRank() != 3 || filtRanked.getDimSize(2) != inRanked.getDimSize(3) ||
          dstRanked.getDimSize(3) != inRanked.getDimSize(3))
        return failure();
      if (dstRanked.getDimSize(1) != inRanked.getDimSize(1) - filtRanked.getDimSize(0) + 1 ||
          dstRanked.getDimSize(2) != inRanked.getDimSize(2) - filtRanked.getDimSize(1) + 1)
        return failure();
      rewriter.replaceOpWithNewOp<linalg::DepthwiseConv2DNhwcHwcOp>(op, TypeRange{dstTy}, ValueRange{input, filter}, ValueRange{empty}, strides, dilations);
      return success();
    }
    if (filtRanked.getRank() != 4)
      return failure();
    if (inRanked.getDimSize(0) != dstRanked.getDimSize(0) || inRanked.getDimSize(3) != filtRanked.getDimSize(2) ||
        filtRanked.getDimSize(3) != dstRanked.getDimSize(3))
      return failure();
    if (dstRanked.getDimSize(1) != inRanked.getDimSize(1) - filtRanked.getDimSize(0) + 1 ||
        dstRanked.getDimSize(2) != inRanked.getDimSize(2) - filtRanked.getDimSize(1) + 1)
      return failure();
    rewriter.replaceOpWithNewOp<linalg::Conv2DNhwcHwcfOp>(op, TypeRange{dstTy}, ValueRange{input, filter}, ValueRange{empty}, strides, dilations);
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
    rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{empty});
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
    if (srcRanked.getShape() != dstRanked.getShape() || srcRanked.getRank() != 1)
      return failure();
    Location loc = op->getLoc();
    Value empty = rewriter.create<tensor::EmptyOp>(loc, dstRanked.getShape(), dstRanked.getElementType());
    auto dimAttr = rewriter.getI64IntegerAttr(0);
    rewriter.replaceOpWithNewOp<linalg::SoftmaxOp>(op, TypeRange{dstTy}, input, empty, dimAttr);
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
    if (!act || !batchGroup || !featGroup || !stride || !inputDil || !paramDil || !padding)
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
    if (!isOnes(stride) || !isOnes(inputDil) || !isOnes(paramDil) || !isZeros(padding))
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
    rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{empty});
    auto strides = rewriter.getDenseI64ArrayAttr({1, 1});
    auto dilations = rewriter.getDenseI64ArrayAttr({1, 1});
    rewriter.replaceOpWithNewOp<linalg::Conv2DNhwcHwcfOp>(op, TypeRange{dstTy}, ValueRange{input, filter}, ValueRange{empty}, strides, dilations);
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
    auto xs = op->getAttrOfType<IntegerAttr>("x_stride");
    auto ys = op->getAttrOfType<IntegerAttr>("y_stride");
    auto xd = op->getAttrOfType<IntegerAttr>("x_dilation_rate");
    auto yd = op->getAttrOfType<IntegerAttr>("y_dilation_rate");
    auto xOut = op->getAttrOfType<IntegerAttr>("x_out_dim");
    auto yOut = op->getAttrOfType<IntegerAttr>("y_out_dim");
    if (!act || !cell || !xs || !ys || !xd || !yd || !xOut || !yOut)
      return failure();
    if (act.getValue() != ActivationFunction::None || cell.getValue() != CellOperation::Mac)
      return failure();
    if (xs.getInt() != 1 || ys.getInt() != 1 || xd.getInt() != 1 || yd.getInt() != 1)
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
    if (dstRanked.getDimSize(1) != (inRanked.getDimSize(1) - 1) + filtRanked.getDimSize(0) ||
        dstRanked.getDimSize(2) != (inRanked.getDimSize(2) - 1) + filtRanked.getDimSize(1))
      return failure();
    if (xOut.getInt() != dstRanked.getDimSize(1) || yOut.getInt() != dstRanked.getDimSize(2))
      return failure();
    Location loc = op->getLoc();
    int64_t hIn = inRanked.getDimSize(1);
    int64_t wIn = inRanked.getDimSize(2);
    int64_t kh = filtRanked.getDimSize(0);
    int64_t kw = filtRanked.getDimSize(1);
    Value empty = rewriter.create<tensor::EmptyOp>(loc, dstRanked.getShape(), dstRanked.getElementType());
    Value zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(dstRanked.getElementType()));
    rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{empty});
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value nV = rewriter.create<arith::ConstantIndexOp>(loc, inRanked.getDimSize(0));
    Value hV = rewriter.create<arith::ConstantIndexOp>(loc, hIn);
    Value wV = rewriter.create<arith::ConstantIndexOp>(loc, wIn);
    Value khV = rewriter.create<arith::ConstantIndexOp>(loc, kh);
    Value kwV = rewriter.create<arith::ConstantIndexOp>(loc, kw);
    Value ciV = rewriter.create<arith::ConstantIndexOp>(loc, inRanked.getDimSize(3));
    Value coV = rewriter.create<arith::ConstantIndexOp>(loc, dstRanked.getDimSize(3));
    auto ln = rewriter.create<scf::ForOp>(loc, c0, nV, c1, ValueRange{empty});
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
    Value ho = rewriter.create<arith::AddIOp>(loc, hi, khi);
    Value wo = rewriter.create<arith::AddIOp>(loc, wi, kwi);
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
    rewriter.replaceOp(op, ln->getResult(0));
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
struct PassThroughLowering : public RewritePattern {
  StringRef root;
  PassThroughLowering(StringRef rootName, MLIRContext *ctx)
      : RewritePattern(rootName, 1, ctx), root(rootName) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    if (op->getName().getStringRef() != root)
      return failure();
    if (op->getNumResults() != 1 || op->getNumOperands() != 1)
      return failure();
    Value input = op->getOperand(0);
    if (input.getType() != op->getResult(0).getType())
      return failure();
    rewriter.replaceOp(op, input);
    return success();
  }
};
struct DepthwiseConvLowering : public RewritePattern {
  DepthwiseConvLowering(MLIRContext *ctx) : RewritePattern("dwc.depthwise_convolution", 1, ctx) {}

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
    if (inRanked.getRank() != 4 || dstRanked.getRank() != 4 || filtRanked.getRank() != 3)
      return failure();
    if (!isa<FloatType>(inRanked.getElementType()) || inRanked.getElementType() != filtRanked.getElementType() ||
        filtRanked.getElementType() != dstRanked.getElementType())
      return failure();
    if (filtRanked.getDimSize(2) != inRanked.getDimSize(3) || dstRanked.getDimSize(3) != inRanked.getDimSize(3))
      return failure();
    if (inRanked.getDimSize(0) != dstRanked.getDimSize(0))
      return failure();
    if (dstRanked.getDimSize(1) != inRanked.getDimSize(1) - filtRanked.getDimSize(0) + 1 ||
        dstRanked.getDimSize(2) != inRanked.getDimSize(2) - filtRanked.getDimSize(1) + 1)
      return failure();
    Location loc = op->getLoc();
    Value empty = rewriter.create<tensor::EmptyOp>(loc, dstRanked.getShape(), dstRanked.getElementType());
    Value zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(dstRanked.getElementType()));
    rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{empty});
    auto strides = rewriter.getDenseI64ArrayAttr({1, 1});
    auto dilations = rewriter.getDenseI64ArrayAttr({1, 1});
    rewriter.replaceOpWithNewOp<linalg::DepthwiseConv2DNhwcHwcOp>(op, TypeRange{dstTy}, ValueRange{input, filter}, ValueRange{empty}, strides, dilations);
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
    rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{empty});
    auto strides = rewriter.getDenseI64ArrayAttr({1, 1, 1});
    auto dilations = rewriter.getDenseI64ArrayAttr({1, 1, 1});
    rewriter.replaceOpWithNewOp<linalg::Conv3DNdhwcDhwcfOp>(op, TypeRange{dstTy}, ValueRange{input, filter}, ValueRange{empty}, strides, dilations);
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
    auto strides = rewriter.getDenseI64ArrayAttr({2, 2});
    auto dilations = rewriter.getDenseI64ArrayAttr({1, 1});
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

struct LayerNormLowering : public RewritePattern {
  StringRef root;
  LayerNormLowering(StringRef rootName, MLIRContext *ctx)
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
    if (srcRanked.getRank() != 2 || dstRanked.getRank() != 2)
      return failure();
    if (srcRanked.getShape() != dstRanked.getShape())
      return failure();
    if (!isa<FloatType>(srcRanked.getElementType()) || srcRanked.getElementType() != dstRanked.getElementType())
      return failure();
    int64_t n = srcRanked.getDimSize(0);
    int64_t c = srcRanked.getDimSize(1);
    Location loc = op->getLoc();
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value nV = rewriter.create<arith::ConstantIndexOp>(loc, n);
    Value cV = rewriter.create<arith::ConstantIndexOp>(loc, c);
    Value countF = rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(srcRanked.getElementType(), static_cast<double>(c)));
    Value empty = rewriter.create<tensor::EmptyOp>(loc, dstRanked.getShape(), dstRanked.getElementType());
    auto outer = rewriter.create<scf::ForOp>(loc, c0, nV, c1, ValueRange{empty});
    rewriter.setInsertionPointToStart(outer.getBody());
    Value row = outer.getInductionVar();
    Value ocur = outer.getRegionIterArg(0);
    Value zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(srcRanked.getElementType()));
    auto acc = rewriter.create<scf::ForOp>(loc, c0, cV, c1, ValueRange{zero});
    rewriter.setInsertionPointToStart(acc.getBody());
    Value col = acc.getInductionVar();
    Value asum = acc.getRegionIterArg(0);
    Value x = rewriter.create<tensor::ExtractOp>(loc, input, ValueRange{row, col});
    Value s = rewriter.create<arith::AddFOp>(loc, asum, x);
    rewriter.create<scf::YieldOp>(loc, ValueRange{s});
    rewriter.setInsertionPointAfter(acc);
    Value mean = rewriter.create<arith::DivFOp>(loc, acc->getResult(0), countF);
    auto norm = rewriter.create<scf::ForOp>(loc, c0, cV, c1, ValueRange{ocur});
    rewriter.setInsertionPointToStart(norm.getBody());
    Value ncol = norm.getInductionVar();
    Value ncur = norm.getRegionIterArg(0);
    Value nx = rewriter.create<tensor::ExtractOp>(loc, input, ValueRange{row, ncol});
    Value d = rewriter.create<arith::SubFOp>(loc, nx, mean);
    Value nnext = rewriter.create<tensor::InsertOp>(loc, d, ncur, ValueRange{row, ncol});
    rewriter.create<scf::YieldOp>(loc, ValueRange{nnext});
    rewriter.setInsertionPointAfter(norm);
    rewriter.create<scf::YieldOp>(loc, ValueRange{norm->getResult(0)});
    rewriter.setInsertionPointAfter(outer);
    rewriter.replaceOp(op, outer->getResult(0));
    return success();
  }
};

struct BilinearUpsampleLowering : public RewritePattern {
  StringRef root;
  BilinearUpsampleLowering(StringRef rootName, MLIRContext *ctx)
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
    if (dstRanked.getDimSize(1) != 2 * srcRanked.getDimSize(1) || dstRanked.getDimSize(2) != 2 * srcRanked.getDimSize(2))
      return failure();
    Location loc = op->getLoc();
    Value empty = rewriter.create<tensor::EmptyOp>(loc, dstRanked.getShape(), dstRanked.getElementType());
    SmallVector<AffineMap> maps = {
      AffineMap::get(4, 0, {rewriter.getAffineDimExpr(0), rewriter.getAffineDimExpr(1).floorDiv(2), rewriter.getAffineDimExpr(2).floorDiv(2), rewriter.getAffineDimExpr(3)}, op->getContext()),
      rewriter.getMultiDimIdentityMap(4),
    };
    SmallVector<utils::IteratorType> iters(4, utils::IteratorType::parallel);
    auto generic = rewriter.create<linalg::GenericOp>(loc, TypeRange{dstTy}, ValueRange{input}, ValueRange{empty},
        maps, iters,
        [&](OpBuilder &nested, Location nloc, ValueRange args) {
          nested.create<linalg::YieldOp>(nloc, args[0]);
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
    auto strides = rewriter.getDenseI64ArrayAttr({2, 2, 2});
    auto dilations = rewriter.getDenseI64ArrayAttr({1, 1, 1});
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
    if (op->getNumOperands() == 1) {
      auto strides = rewriter.getDenseI64ArrayAttr({2, 2});
      auto dilations = rewriter.getDenseI64ArrayAttr({1, 1});
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
    auto strides = rewriter.getDenseI64ArrayAttr({1, 1});
    auto dilations = rewriter.getDenseI64ArrayAttr({1, 1});
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
    rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{empty});
    auto strides = rewriter.getDenseI64ArrayAttr({1, 1});
    auto dilations = rewriter.getDenseI64ArrayAttr({1, 1});
    rewriter.replaceOpWithNewOp<linalg::Conv2DNhwcHwcfOp>(op, TypeRange{dstTy}, ValueRange{input, filter}, ValueRange{empty}, strides, dilations);
    return success();
  }
};

struct VicaFusedNormLowering : public RewritePattern {
  VicaFusedNormLowering(MLIRContext *ctx) : RewritePattern("dwc.vica_fused_norm", 1, ctx) {}

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
    if (srcRanked.getRank() != 2)
      return failure();
    int64_t n = srcRanked.getDimSize(0);
    int64_t c = srcRanked.getDimSize(1);
    Location loc = op->getLoc();
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value nV = rewriter.create<arith::ConstantIndexOp>(loc, n);
    Value cV = rewriter.create<arith::ConstantIndexOp>(loc, c);
    Value countF = rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(srcRanked.getElementType(), static_cast<double>(c)));
    Value empty = rewriter.create<tensor::EmptyOp>(loc, dstRanked.getShape(), dstRanked.getElementType());
    auto outer = rewriter.create<scf::ForOp>(loc, c0, nV, c1, ValueRange{empty});
    rewriter.setInsertionPointToStart(outer.getBody());
    Value row = outer.getInductionVar();
    Value ocur = outer.getRegionIterArg(0);
    Value zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(srcRanked.getElementType()));
    auto acc = rewriter.create<scf::ForOp>(loc, c0, cV, c1, ValueRange{zero});
    rewriter.setInsertionPointToStart(acc.getBody());
    Value col = acc.getInductionVar();
    Value asum = acc.getRegionIterArg(0);
    Value x = rewriter.create<tensor::ExtractOp>(loc, input, ValueRange{row, col});
    Value s = rewriter.create<arith::AddFOp>(loc, asum, x);
    rewriter.create<scf::YieldOp>(loc, ValueRange{s});
    rewriter.setInsertionPointAfter(acc);
    Value sum = acc->getResult(0);
    Value mean = rewriter.create<arith::DivFOp>(loc, sum, countF);
    auto norm = rewriter.create<scf::ForOp>(loc, c0, cV, c1, ValueRange{ocur});
    rewriter.setInsertionPointToStart(norm.getBody());
    Value ncol = norm.getInductionVar();
    Value ncur = norm.getRegionIterArg(0);
    Value nx = rewriter.create<tensor::ExtractOp>(loc, input, ValueRange{row, ncol});
    Value d = rewriter.create<arith::SubFOp>(loc, nx, mean);
    Value nnext = rewriter.create<tensor::InsertOp>(loc, d, ncur, ValueRange{row, ncol});
    rewriter.create<scf::YieldOp>(loc, ValueRange{nnext});
    rewriter.setInsertionPointAfter(norm);
    rewriter.create<scf::YieldOp>(loc, ValueRange{norm->getResult(0)});
    rewriter.setInsertionPointAfter(outer);
    rewriter.replaceOp(op, outer->getResult(0));
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
    rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{empty});
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
    rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{empty});
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


struct ElementCountReshapeLowering : public RewritePattern {
  StringRef root;
  ElementCountReshapeLowering(StringRef rootName, MLIRContext *ctx)
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
    if (srcRanked.getNumElements() != dstRanked.getNumElements())
      return failure();
    if (srcRanked.getElementType() != dstRanked.getElementType())
      return failure();
    if (srcRanked.getRank() != dstRanked.getRank())
      return failure();
    SmallVector<ReassociationIndices> reassoc;
    for (int64_t d = 0; d < srcRanked.getRank(); ++d)
      reassoc.push_back(ReassociationIndices{d});
    rewriter.replaceOpWithNewOp<tensor::CollapseShapeOp>(op, dstTy, input, reassoc);
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

struct RescalingNoneLowering : public RewritePattern {
  RescalingNoneLowering(MLIRContext *ctx)
      : RewritePattern("dwc.rescaling", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1 || op->getNumOperands() != 1)
      return failure();
    auto activation = op->getAttrOfType<ActivationFunctionAttr>("activation_function");
    if (!activation || activation.getValue() != ActivationFunction::None)
      return failure();
    Value input = op->getOperand(0);
    if (input.getType() != op->getResult(0).getType())
      return failure();
    rewriter.replaceOp(op, input);
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
    rewriter.create<linalg::FillOp>(loc, ValueRange{scalar}, ValueRange{filled});
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
    rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{empty});
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
    if (op->getNumResults() != 1 || op->getNumOperands() != 1)
      return failure();
    Value input = op->getOperand(0);
    if (input.getType() != op->getResult(0).getType())
      return failure();
    rewriter.replaceOp(op, input);
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
  patterns.add<CopyOpLowering>(ctx);
  patterns.add<DarwinnFillLowering>(ctx);
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
  patterns.add<ReductionLowering>(ctx);
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
  patterns.add<ConvolutionLowering>(ctx);
  patterns.add<DepthwiseConvLowering>(ctx);
  patterns.add<Conv3DLowering>(ctx);
  patterns.add<Pool2DLowering>("dwc.pool", ctx);
  patterns.add<Pool2DLowering>("dwc.pooling", ctx);
  patterns.add<Pool2DLowering>("dwc.reduce_window", ctx);
  patterns.add<Pool3DLowering>(ctx);
  patterns.add<WalshHadamardLowering>(ctx);
  patterns.add<LayerNormLowering>("dwc.normalization", ctx);
  patterns.add<BilinearUpsampleLowering>("dwc.resampler", ctx);
  patterns.add<BilinearUpsampleLowering>("dwc.image_interpolation", ctx);
  patterns.add<ForwardLowering>("dwc.multinormal", "dive_vm.multinomial", ctx);
  patterns.add<ForwardLowering>("dwc.uniform_random_number_generation", "dive_vm.multinomial", ctx);
  patterns.add<VicaAddPoolLowering>(ctx);
  patterns.add<VicaFusedConvLowering>(ctx);
  patterns.add<VicaFusedNormLowering>(ctx);
  patterns.add<IndexUnpoolLowering>(ctx);
  patterns.add<ForwardLowering>("dwc.vica_conv_d2s", "dive_vm.pad", ctx);
  patterns.add<ForwardLowering>("dwc.vica_custom_padding", "dive_vm.pad", ctx);
  patterns.add<SubByteMatmulLowering>("dwc.fully_connected_sub_byte_param", ctx);
  patterns.add<MatmulLowering>("dwc.fully_connected_zin_indexed", ctx);
  patterns.add<MatmulLowering>("dwc.fully_connected_zout_indexed", ctx);
  patterns.add<ConvolutionV2Lowering>("dwc.convolution_sub_channel", ctx);
  patterns.add<ForwardLowering>("dwc.transposed_convolution_sub_channel", "dive_vm.scatter_nd", ctx);
  patterns.add<ForwardLowering>("dwc.mask_indices", "dive_vm.mask_indices", ctx);
  patterns.add<ForwardLowering>("dwc.hib_gather", "dive_vm.hib_gather_edit", ctx);
  patterns.add<ForwardLowering>("dwc.one_hot_tpu", "dive_vm.one_hot", ctx);
  patterns.add<ForwardLowering>("dwc.generic_scatter", "dive_vm.scatter_nd", ctx);
  patterns.add<ConvolutionV2Lowering>("dwc.convolution_v2", ctx);
  patterns.add<ConvolutionV2Lowering>("dwc.depthwise_convolution_v2", ctx);
  patterns.add<ScalarLowering>(ctx);
  patterns.add<ClassifierLowering>(ctx);
  patterns.add<GenericConvLowering>(ctx);
  patterns.add<TransposedConvLowering>(ctx);
  patterns.add<GenericDotLowering>(ctx);
  patterns.add<IdentityLowering>("dwc.reduce_precision", ctx);
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
  patterns.add<UnaryLowering>("dwc.logistic", emitLog, ctx);
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
  patterns.add<PassThroughLowering>("dwc.pseudo_fill", ctx);
  patterns.add<PassThroughLowering>("dwc.pseudo_dynamic_slice", ctx);
  patterns.add<PassThroughLowering>("dwc.pseudo_dynamic_pad", ctx);
  patterns.add<PassThroughLowering>("dwc.pseudo_dynamic_reshape", ctx);
  patterns.add<PassThroughLowering>("dwc.pseudo_expand_dims", ctx);
  patterns.add<PassThroughLowering>("dwc.pseudo_squeeze", ctx);
  patterns.add<PassThroughLowering>("dwc.pseudo_shape", ctx);
  patterns.add<PassThroughLowering>("dwc.pseudo_range", ctx);
  patterns.add<PassThroughLowering>("dwc.pseudo_mirror_pad", ctx);
  patterns.add<PassThroughLowering>("dwc.pseudo_generic_norm", ctx);
  patterns.add<PassThroughLowering>("dwc.pseudo_group_norm", ctx);
  patterns.add<PassThroughLowering>("dwc.pseudo_dynamic_image_interpolation", ctx);
  patterns.add<PassThroughLowering>("dwc.ensure_shape", ctx);
  patterns.add<PassThroughLowering>("dwc.generic_move", ctx);
  patterns.add<PassThroughLowering>("dwc.generic_pad", ctx);
  patterns.add<PassThroughLowering>("dwc.dynamic_broadcast", ctx);
  patterns.add<PassThroughLowering>("dwc.dynamic_quantize", ctx);
  patterns.add<ArangeLowering>(ctx);
  patterns.add<PackBitsLowering>(ctx);
  patterns.add<ForwardLowering>("dwc.statistical_top_k", "dive_vm.top_k", ctx);
  patterns.add<ForwardLowering>("dwc.multinomial", "dive_vm.multinomial", ctx);
  patterns.add<ElementCountReshapeLowering>("dwc.space_to_batch", ctx);
  patterns.add<ElementCountReshapeLowering>("dwc.space_to_depth", ctx);
  patterns.add<ElementCountReshapeLowering>("dwc.batch_to_space", ctx);
  patterns.add<ElementCountReshapeLowering>("dwc.depth_to_space", ctx);
  patterns.add<ElementCountReshapeLowering>("dwc.dynamic_slice_nd", ctx);
  patterns.add<ElementCountReshapeLowering>("dwc.dynamic_slice_nd_v2", ctx);
  patterns.add<ElementCountReshapeLowering>("dwc.dynamic_update_slice_nd", ctx);
  patterns.add<ElementCountReshapeLowering>("dwc.dynamic_update_slice_nd_v2", ctx);
  patterns.add<RescalingNoneLowering>(ctx);
  patterns.add<BitwiseLowering>("dwc.and", emitAnd, ctx);
  patterns.add<BitwiseLowering>("dwc.or", emitOr, ctx);
  patterns.add<BitwiseLowering>("dwc.xor", emitXor, ctx);
  patterns.add<BitSelectLowering>(ctx);
  patterns.add<PassThroughLowering>("dwc.attention", ctx);
  patterns.add<PassThroughLowering>("dwc.custom_compute", ctx);
  patterns.add<PassThroughLowering>("dwc.extern_call", ctx);
  patterns.add<PassThroughLowering>("dwc.launch_custom_kernel", ctx);
  patterns.add<PassThroughLowering>("dwc.collective_permute", ctx);
  patterns.add<BitwiseLowering>("dwc.shift_left", emitShl, ctx);
  patterns.add<BitwiseLowering>("dwc.shift_right_arithmetic", emitShrS, ctx);
  patterns.add<BitwiseLowering>("dwc.shift_right_logical", emitShrU, ctx);
  patterns.add<IsFiniteLowering>(ctx);
  patterns.add<IntUnaryLowering>("dwc.count_leading_zeros", emitCtlz, ctx);
  patterns.add<ClampLowering>(ctx);
  patterns.add<IdentityLowering>("dwc.identity", ctx);
  patterns.add<IdentityLowering>("dwc.const_none", ctx);
  patterns.add<ReverseLowering>(ctx);
  patterns.add<BitcastLowering>(ctx);
  patterns.add<ForwardLowering>("dwc.gather_nd", "dive_vm.gather_nd", ctx);
  patterns.add<ForwardLowering>("dwc.scatter_nd", "dive_vm.scatter_nd", ctx);
  patterns.add<ForwardLowering>("dwc.pad", "dive_vm.pad", ctx);
  patterns.add<ForwardLowering>("dwc.roll", "dive_vm.roll", ctx);
  patterns.add<ForwardLowering>("dwc.top_k", "dive_vm.top_k", ctx);
  patterns.add<UnsortedSegmentReduceLowering>(ctx);
}
