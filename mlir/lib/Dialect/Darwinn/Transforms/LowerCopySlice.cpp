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
    auto opType = dyn_cast<ReductionTypeAttr>(op->getAttr("op_type"));
    if (!opType)
      return failure();
    if (opType.getValue() != ReductionType::Sum && opType.getValue() != ReductionType::Max)
      return failure();
    if (auto activation = dyn_cast<SimpleActivationFunctionAttr>(op->getAttr("activation_function")))
      if (activation.getValue() != SimpleActivationFunction::None)
        return failure();
    auto dims = dyn_cast<DenseIntElementsAttr>(op->getAttr("dimensions"));
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
    Location loc = op->getLoc();
    Value initBuf = rewriter.create<tensor::EmptyOp>(loc, dstRanked.getShape(), dstRanked.getElementType());
    Value initScalar;
    if (opType.getValue() == ReductionType::Sum)
      initScalar = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(dstRanked.getElementType()));
    else
      initScalar = rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(dstRanked.getElementType(), -std::numeric_limits<float>::infinity()));
    rewriter.create<linalg::FillOp>(loc, ValueRange{initScalar}, ValueRange{initBuf});
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
    auto perm = dyn_cast<DenseIntElementsAttr>(op->getAttr("permutation"));
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
      auto attr = dyn_cast<IntegerAttr>(op->getAttr(name));
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

struct ConvolutionLowering : public RewritePattern {
  ConvolutionLowering(MLIRContext *ctx)
      : RewritePattern("dwc.convolution", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1 || op->getNumOperands() != 2)
      return failure();
    auto getI64 = [&](StringRef name, int64_t &out) -> bool {
      auto attr = dyn_cast<IntegerAttr>(op->getAttr(name));
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

struct PaddingLowering : public RewritePattern {
  PaddingLowering(MLIRContext *ctx)
      : RewritePattern("dwc.padding", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1 || op->getNumOperands() != 1)
      return failure();
    auto dim = dyn_cast<IntegerAttr>(op->getAttr("dimension"));
    auto pre = dyn_cast<IntegerAttr>(op->getAttr("pre_padding"));
    auto post = dyn_cast<IntegerAttr>(op->getAttr("post_padding"));
    if (!dim || !pre || !post)
      return failure();
    int64_t axis = dim.getInt();
    int64_t lo = pre.getInt();
    int64_t hi = post.getInt();
    if (lo < 0 || hi < 0)
      return failure();
    Value padValue;
    if (auto floatAttr = dyn_cast<FloatAttr>(op->getAttr("padding_value")))
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

struct PseudoSplitLowering : public RewritePattern {
  PseudoSplitLowering(MLIRContext *ctx)
      : RewritePattern("dwc.pseudo_split", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1 || op->getNumOperands() != 2)
      return failure();
    auto splits = dyn_cast<IntegerAttr>(op->getAttr("num_splits"));
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
    auto activation = dyn_cast<ActivationFunctionAttr>(op->getAttr("activation_function"));
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
    auto axis = dyn_cast<IntegerAttr>(op->getAttr("axis"));
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
    auto axis = dyn_cast<IntegerAttr>(op->getAttr("axis"));
    auto exclusive = dyn_cast<BoolAttr>(op->getAttr("exclusive"));
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
  patterns.add<ConvolutionLowering>(ctx);
  patterns.add<BroadcastLowering>(ctx);
  patterns.add<ConcatenationLowering>(ctx);
  patterns.add<SliceLowering>(ctx);
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
  patterns.add<BinaryGenericLowering>("dwc.floor_div", emitFloorDiv, ctx);
  patterns.add<BinaryGenericLowering>("dwc.pow", emitPowF, ctx);
  patterns.add<IntUnaryLowering>("dwc.pop_count", emitPopCount, ctx);
  patterns.add<OneHotLowering>(ctx);
  patterns.add<CumulativeLowering>(ctx);
  patterns.add<PseudoSplitLowering>(ctx);
  patterns.add<RescalingNoneLowering>(ctx);
}
