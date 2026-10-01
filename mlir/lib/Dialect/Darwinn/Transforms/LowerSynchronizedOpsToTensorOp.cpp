#include "mlir/Dialect/Darwinn/IR/DarwinnOps.h"
#include "mlir/Dialect/Darwinn/Transforms/DistributedPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::darwinn;

namespace {

void lower(SynchronizedCopyOpOp copy) {
  OpBuilder builder(copy);
  MLIRContext *context = builder.getContext();
  auto shard = NarrowToWideShardAttr::get(context, builder.getArrayAttr({}),
                                          builder.getArrayAttr({}));
  auto wide = NarrowToWideOp::create(
      builder, copy.getLoc(), copy.getType(), TypeRange{}, copy.getInput(),
      ValueRange{}, builder.getArrayAttr({shard}), copy.getTraversalAttr(),
      copy.getForwardIndexTransformationAttr(), AffineMapAttr{}, IntegerAttr{},
      IntegerAttr{}, ArrayAttr{}, ArrayAttr{});
  copy.replaceAllUsesWith(wide.getOutput());
  copy.erase();
}

void lower(SynchronizedComputeOpOp compute) {
  OpBuilder builder(compute);
  MLIRContext *context = builder.getContext();
  auto shard = TensorOpShardAttr::get(context, builder.getArrayAttr({}),
                                      builder.getArrayAttr({}));
  auto tensor = TensorOpOp::create(
      builder, compute.getLoc(), compute.getType(), compute.getLhs(),
      compute.getRhs(), compute.getDestination(), compute.getAuxiliaryTensors(),
      compute.getComputeAttr(), builder.getArrayAttr({shard}),
      compute.getTraversalAttr(), compute.getAuxiliaryTensorTypesAttr(),
      ArrayAttr{});
  compute.replaceAllUsesWith(tensor.getOutput());
  compute.erase();
}

void lower(SynchronizedUnaryComputeOpOp compute) {
  OpBuilder builder(compute);
  auto tensor = UnaryTensorOpOp::create(
      builder, compute.getLoc(), compute.getType(), compute.getInput(),
      compute.getDestination(), compute.getAuxiliaryTensors(),
      compute.getComputeAttr(), builder.getArrayAttr({}),
      compute.getTraversalAttr(), compute.getAuxiliaryTensorTypesAttr(),
      ArrayAttr{}, compute.getCustomConstraintsAttr(), ResamplerOptionsAttr{});
  compute.replaceAllUsesWith(tensor.getOutput());
  compute.erase();
}

class LowerSynchronizedOpsToTensorOpPass
    : public PassWrapper<LowerSynchronizedOpsToTensorOpPass,
                         OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      LowerSynchronizedOpsToTensorOpPass)

  StringRef getArgument() const final {
    return "darwinn-lower-synchronized-ops-to-tensor-op";
  }

  StringRef getDescription() const final {
    return "Lower synchronized compute and copies to tensor operations";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<DarwinnDialect>();
  }

  void runOnOperation() final {
    SmallVector<Operation *> synchronized;
    getOperation().walk([&](Operation *operation) {
      if (isa<SynchronizedCopyOpOp, SynchronizedComputeOpOp,
              SynchronizedUnaryComputeOpOp>(operation))
        synchronized.push_back(operation);
    });
    for (Operation *operation : synchronized) {
      if (auto copy = dyn_cast<SynchronizedCopyOpOp>(operation))
        lower(copy);
      else if (auto compute = dyn_cast<SynchronizedComputeOpOp>(operation))
        lower(compute);
      else
        lower(cast<SynchronizedUnaryComputeOpOp>(operation));
    }
  }
};

}

std::unique_ptr<Pass>
mlir::darwinn::createLowerSynchronizedOpsToTensorOpPass() {
  return std::make_unique<LowerSynchronizedOpsToTensorOpPass>();
}

void mlir::darwinn::registerLowerSynchronizedOpsToTensorOpPass() {
  PassRegistration<LowerSynchronizedOpsToTensorOpPass>();
}
