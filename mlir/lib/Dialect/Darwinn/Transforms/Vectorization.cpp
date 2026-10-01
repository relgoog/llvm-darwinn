#include "mlir/Dialect/Darwinn/IR/DarwinnOps.h"
#include "mlir/Dialect/Darwinn/Transforms/DistributedPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::darwinn;

namespace {

bool feedsMatrixWeights(StreamingCopyOpOp copy) {
  if (!copy->hasOneUse())
    return false;
  auto compute = dyn_cast<StreamingComputeOpOp>(*copy->getUsers().begin());
  return compute && compute.getRhs() == copy.getOutput() &&
         compute.getCompute().getInnerOperation() == InnerOperationKind::Vmc;
}

LogicalResult vectorize(StreamingCopyOpOp copy) {
  auto view = dyn_cast<DistributedViewType>(copy.getOutput().getType());
  if (!view || view.getRank() < 2 || copy.getForwardIndexTransformation() ||
      isa<VectorType>(view.getElementType()))
    return copy.emitOpError("streams matrix weights of an unexpected form");
  ArrayRef<int64_t> shape = view.getShape();
  auto lanes = VectorType::get({shape.back()}, view.getElementType());
  auto vectorized =
      DistributedViewType::get(copy.getContext(), shape.drop_back(), lanes,
                               view.getMemorySpace());
  AffineMap traversal = copy.getTraversal();
  copy.setTraversal(traversal.dropResult(traversal.getNumResults() - 1));
  copy.setForwardIndexTransformation(
      AffineMap::getMultiDimIdentityMap(shape.size(), copy.getContext())
          .dropResult(shape.size() - 1));
  copy.getOutput().setType(vectorized);
  return success();
}

class VectorizationPass
    : public PassWrapper<VectorizationPass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VectorizationPass)

  StringRef getArgument() const final { return "darwinn-vectorization"; }

  StringRef getDescription() const final {
    return "Pack the output channels of streamed matrix weights into vectors";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<DarwinnDialect>();
  }

  void runOnOperation() final {
    SmallVector<StreamingCopyOpOp> copies;
    getOperation().walk([&](StreamingCopyOpOp copy) {
      if (feedsMatrixWeights(copy))
        copies.push_back(copy);
    });
    for (StreamingCopyOpOp copy : copies)
      if (failed(vectorize(copy)))
        return signalPassFailure();
  }
};

}

std::unique_ptr<Pass> mlir::darwinn::createVectorizationPass() {
  return std::make_unique<VectorizationPass>();
}

void mlir::darwinn::registerVectorizationPass() {
  PassRegistration<VectorizationPass>();
}
