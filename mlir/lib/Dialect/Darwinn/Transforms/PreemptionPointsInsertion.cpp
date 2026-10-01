#include "SlicingModel.h"
#include "mlir/Dialect/Darwinn/IR/DarwinnOps.h"
#include "mlir/Dialect/Darwinn/Transforms/DistributedPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::darwinn;
using namespace mlir::darwinn::slicing;

namespace {

// The SDK converts estimated cycles to microseconds at 1120 MHz, traced as
// 49310 cycles reported as 44.0268 us on the selfie input transfer.
constexpr double kCyclesPerMicrosecond = 1120.0;
constexpr double kInsertionIntervalMicroseconds = 200.0;

class PreemptionPointsInsertionPass
    : public PassWrapper<PreemptionPointsInsertionPass,
                         OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PreemptionPointsInsertionPass)

  StringRef getArgument() const final {
    return "darwinn-preemption-points-insertion";
  }

  StringRef getDescription() const final {
    return "Insert software preemption points at a fixed estimated interval";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<DarwinnDialect>();
  }

  void runOnOperation() final {
    MaterializedSlicing slicing;
    SmallVector<std::pair<Operation *, double>> timed;
    WalkResult walk = getOperation().walk([&](Operation *operation) {
      if (operation->getNumResults() == 0)
        return WalkResult::advance();
      FailureOr<Estimate> estimate = slicing.estimate(operation);
      if (failed(estimate))
        return WalkResult::interrupt();
      double microseconds = cycles(*estimate) / kCyclesPerMicrosecond;
      if (microseconds != 0)
        timed.push_back({operation, microseconds});
      return WalkResult::advance();
    });
    if (walk.wasInterrupted())
      return signalPassFailure();

    double elapsed = 0;
    for (auto [operation, microseconds] : timed) {
      elapsed += microseconds;
      if (elapsed < kInsertionIntervalMicroseconds)
        continue;
      OpBuilder builder(operation);
      PreemptionPointOp::create(builder, operation->getLoc(),
                                builder.getBoolAttr(false));
      elapsed = microseconds;
    }
  }
};

}

std::unique_ptr<Pass> mlir::darwinn::createPreemptionPointsInsertionPass() {
  return std::make_unique<PreemptionPointsInsertionPass>();
}

void mlir::darwinn::registerPreemptionPointsInsertionPass() {
  PassRegistration<PreemptionPointsInsertionPass>();
}
