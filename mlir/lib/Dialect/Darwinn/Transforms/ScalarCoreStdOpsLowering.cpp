#include "mlir/Dialect/Darwinn/IR/DarwinnOps.h"
#include "mlir/Dialect/Darwinn/Transforms/DistributedPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::darwinn;

namespace {

class ScalarCoreStdOpsLoweringPass
    : public PassWrapper<ScalarCoreStdOpsLoweringPass,
                         OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ScalarCoreStdOpsLoweringPass)

  StringRef getArgument() const final {
    return "darwinn-scalar-core-std-ops-lowering";
  }

  StringRef getDescription() const final {
    return "Drop reshapes left without users before scalar core lowering";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<DarwinnDialect>();
  }

  void runOnOperation() final {
    SmallVector<ReshapeOpOp> dead;
    WalkResult walk = getOperation().walk([&](Operation *operation) {
      StringRef dialect = operation->getDialect()
                              ? operation->getDialect()->getNamespace()
                              : StringRef();
      if (dialect == "arith" || dialect == "math" || dialect == "scf") {
        operation->emitOpError("needs scalar core lowering, which is not "
                               "recovered");
        return WalkResult::interrupt();
      }
      auto reshape = dyn_cast<ReshapeOpOp>(operation);
      if (reshape && reshape->use_empty())
        dead.push_back(reshape);
      return WalkResult::advance();
    });
    if (walk.wasInterrupted())
      return signalPassFailure();
    for (ReshapeOpOp reshape : dead)
      reshape.erase();
  }
};

}

std::unique_ptr<Pass> mlir::darwinn::createScalarCoreStdOpsLoweringPass() {
  return std::make_unique<ScalarCoreStdOpsLoweringPass>();
}

void mlir::darwinn::registerScalarCoreStdOpsLoweringPass() {
  PassRegistration<ScalarCoreStdOpsLoweringPass>();
}
