#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DiveVm/IR/DiveVmOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <algorithm>
#include <cstdint>

using namespace mlir;

namespace {

struct PutBitsLowering : OpRewritePattern<dive_vm::PutBitsOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(dive_vm::PutBitsOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    int64_t offset = op.getBitOffsetAttr().getInt();
    int64_t width = op.getNbitsAttr().getInt();
    int64_t end = offset + width;

    for (int64_t byte = offset / 8; byte <= (end - 1) / 8; ++byte) {
      int64_t firstBit = std::max(offset, byte * 8);
      int64_t sourceShift = firstBit - offset;
      int64_t targetShift = firstBit - byte * 8;
      int64_t bitCount = std::min(end - firstBit, 8 - targetShift);
      int64_t mask = ((int64_t{1} << bitCount) - 1) << targetShift;
      Value part = op.getValue();

      if (sourceShift != 0) {
        Value amount =
            arith::ConstantIntOp::create(rewriter, loc, sourceShift, 64);
        part = arith::ShRUIOp::create(rewriter, loc, part, amount);
      }

      part = arith::TruncIOp::create(rewriter, loc, rewriter.getI8Type(), part);

      if (targetShift != 0) {
        Value amount =
            arith::ConstantIntOp::create(rewriter, loc, targetShift, 8);
        part = arith::ShLIOp::create(rewriter, loc, part, amount);
      }

      Value index = arith::ConstantIndexOp::create(rewriter, loc, byte);

      if (mask != 255) {
        Value fieldMask = arith::ConstantIntOp::create(rewriter, loc, mask, 8);
        Value keepMask =
            arith::ConstantIntOp::create(rewriter, loc, 255 ^ mask, 8);
        Value previous =
            memref::LoadOp::create(rewriter, loc, op.getTarget(), index);
        Value kept = arith::AndIOp::create(rewriter, loc, previous, keepMask);
        Value selected = arith::AndIOp::create(rewriter, loc, part, fieldMask);
        part = arith::OrIOp::create(rewriter, loc, kept, selected);
      }

      memref::StoreOp::create(rewriter, loc, part, op.getTarget(), index);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

}

namespace mlir::darwinn {

LogicalResult lowerDiveVmPutBits(Operation *operation) {
  RewritePatternSet patterns(operation->getContext());
  patterns.add<PutBitsLowering>(operation->getContext());
  return applyPatternsGreedily(operation, std::move(patterns));
}

}
