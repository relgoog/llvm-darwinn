#include "SlicingModel.h"
#include "mlir/Dialect/Darwinn/IR/DarwinnOps.h"
#include "mlir/Dialect/Darwinn/Transforms/DistributedPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;
using namespace mlir::darwinn;
using namespace mlir::darwinn::slicing;

namespace {

bool isMovable(Operation *operation) {
  if (isa<DistributedCreateViewOp, FillOp, StreamingCopyOpOp>(operation))
    return true;
  auto redistribute = dyn_cast<RedistributeOp>(operation);
  return redistribute && memorySpaceOf(redistribute.getInput()) ==
                             DistributedMemorySpace::HostMemory;
}

void sinkStreamedOperand(StreamingComputeOpOp compute) {
  llvm::SetVector<Operation *> chain;
  SmallVector<Value> frontier{compute.getRhs()};
  while (!frontier.empty()) {
    Operation *producer = frontier.pop_back_val().getDefiningOp();
    if (!producer || !isMovable(producer) || !producer->hasOneUse() ||
        !chain.insert(producer))
      continue;
    llvm::append_range(frontier, producer->getOperands());
  }
  SmallVector<Operation *> ordered = chain.takeVector();
  llvm::sort(ordered, [](Operation *left, Operation *right) {
    return left->isBeforeInBlock(right);
  });
  for (Operation *operation : ordered)
    operation->moveBefore(compute);
}

class OpReorderingForStreamingPass
    : public PassWrapper<OpReorderingForStreamingPass,
                         OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OpReorderingForStreamingPass)

  StringRef getArgument() const final {
    return "darwinn-op-reordering-for-streaming";
  }

  StringRef getDescription() const final {
    return "Sink each streamed operand chain to its compute";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<DarwinnDialect>();
  }

  void runOnOperation() final {
    SmallVector<StreamingComputeOpOp> computes;
    getOperation().walk(
        [&](StreamingComputeOpOp compute) { computes.push_back(compute); });
    for (StreamingComputeOpOp compute : computes)
      sinkStreamedOperand(compute);
  }
};

}

std::unique_ptr<Pass> mlir::darwinn::createOpReorderingForStreamingPass() {
  return std::make_unique<OpReorderingForStreamingPass>();
}

void mlir::darwinn::registerOpReorderingForStreamingPass() {
  PassRegistration<OpReorderingForStreamingPass>();
}
