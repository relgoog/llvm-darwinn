#include "mlir/Dialect/Darwinn/IR/DarwinnOps.h"
#include "mlir/Dialect/Darwinn/Transforms/DistributedPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::darwinn;

namespace {

AffineMap wideRegisterTraversal(SynchronizedCopyOpOp weights) {
  unsigned first = ~0u;
  weights.getTraversal().walkExprs([&](AffineExpr expression) {
    if (auto dim = dyn_cast<AffineDimExpr>(expression))
      first = std::min(first, dim.getPosition());
  });
  unsigned rank = cast<ShapedType>(weights.getType()).getRank();
  SmallVector<AffineExpr> results;
  for (unsigned dim = first; dim < first + rank; ++dim)
    results.push_back(getAffineDimExpr(dim, weights.getContext()));
  return AffineMap::get(first + rank, 0, results, weights.getContext());
}

void convert(StreamingCopyOpOp copy) {
  OpBuilder builder(copy);
  auto synchronized = SynchronizedCopyOpOp::create(
      builder, copy.getLoc(), copy.getType(), copy.getInput(),
      copy.getTraversalAttr(), copy.getForwardIndexTransformationAttr());
  for (StringRef name : {"memory_space", "streaming_input", "streaming_output"})
    if (Attribute attribute = copy->getAttr(name))
      synchronized->setDiscardableAttr(name, attribute);
  unsigned rank = cast<ShapedType>(copy.getType()).getRank();
  synchronized->setDiscardableAttr(
      "wide_register_traversal",
      AffineMapAttr::get(
          AffineMap::getMultiDimIdentityMap(rank, copy.getContext())));
  copy.replaceAllUsesWith(synchronized.getOutput());
  copy.erase();
}

void convert(StreamingComputeOpOp compute) {
  OpBuilder builder(compute);
  auto synchronized = SynchronizedComputeOpOp::create(
      builder, compute.getLoc(), compute.getType(), compute.getLhs(),
      compute.getRhs(), compute.getDestination(), compute.getAuxiliaryTensors(),
      compute.getComputeAttr(), compute.getTraversalAttr(),
      compute.getAuxiliaryTensorTypesAttr());
  for (StringRef name : {"lhs_traversal", "streaming_input0",
                         "streaming_input1", "streaming_output"})
    if (Attribute attribute = compute->getAttr(name))
      synchronized->setDiscardableAttr(name, attribute);
  if (auto weights = compute.getRhs().getDefiningOp<SynchronizedCopyOpOp>())
    synchronized->setDiscardableAttr(
        "wide_register_traversal",
        AffineMapAttr::get(wideRegisterTraversal(weights)));
  compute.replaceAllUsesWith(synchronized.getOutput());
  compute.erase();
}

void convert(StreamingUnaryComputeOpOp compute) {
  OpBuilder builder(compute);
  auto synchronized = SynchronizedUnaryComputeOpOp::create(
      builder, compute.getLoc(), compute.getType(), compute.getInput(),
      compute.getDestination(), compute.getAuxiliaryTensors(),
      compute.getComputeAttr(), compute.getTraversalAttr(),
      compute.getAuxiliaryTensorTypesAttr(),
      compute.getCustomConstraintsAttr());
  compute.replaceAllUsesWith(synchronized.getOutput());
  compute.erase();
}

class StreamingToSynchronizationPass
    : public PassWrapper<StreamingToSynchronizationPass,
                         OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StreamingToSynchronizationPass)

  StringRef getArgument() const final {
    return "darwinn-streaming-to-synchronization";
  }

  StringRef getDescription() const final {
    return "Turn streaming compute and copies into synchronized operations";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<DarwinnDialect>();
  }

  void runOnOperation() final {
    SmallVector<Operation *> streaming;
    getOperation().walk([&](Operation *operation) {
      if (isa<StreamingCopyOpOp, StreamingComputeOpOp,
              StreamingUnaryComputeOpOp>(operation))
        streaming.push_back(operation);
    });
    for (Operation *operation : streaming) {
      if (auto copy = dyn_cast<StreamingCopyOpOp>(operation))
        convert(copy);
      else if (auto compute = dyn_cast<StreamingComputeOpOp>(operation))
        convert(compute);
      else
        convert(cast<StreamingUnaryComputeOpOp>(operation));
    }
  }
};

}

std::unique_ptr<Pass> mlir::darwinn::createStreamingToSynchronizationPass() {
  return std::make_unique<StreamingToSynchronizationPass>();
}

void mlir::darwinn::registerStreamingToSynchronizationPass() {
  PassRegistration<StreamingToSynchronizationPass>();
}
