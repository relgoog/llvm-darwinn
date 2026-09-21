#ifndef MLIR_DIALECT_DARWINN_TRANSFORMS_PLANTENSORTRAVERSALS_H
#define MLIR_DIALECT_DARWINN_TRANSFORMS_PLANTENSORTRAVERSALS_H

#include "mlir/Dialect/Darwinn/IR/DarwinnScheduledAttrs.h"
#include "mlir/Support/LogicalResult.h"
#include <memory>

namespace mlir {
class Operation;
class Pass;

namespace darwinn {

bool isTensorTraversalPlanningCandidate(Operation *operation);
FailureOr<TensorTraversalPlanAttr>
deriveTensorTraversalPlan(Operation *operation);
LogicalResult verifyTensorTraversalPlan(Operation *operation);
std::unique_ptr<Pass> createPlanTensorTraversalsPass();
void registerPlanTensorTraversalsPass();

}
}

#endif
