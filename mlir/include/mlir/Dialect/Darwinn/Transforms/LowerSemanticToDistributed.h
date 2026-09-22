#ifndef MLIR_DIALECT_DARWINN_TRANSFORMS_LOWERSEMANTICTODISTRIBUTED_H
#define MLIR_DIALECT_DARWINN_TRANSFORMS_LOWERSEMANTICTODISTRIBUTED_H

#include <memory>

namespace mlir {
class Pass;

namespace darwinn {

std::unique_ptr<Pass> createLowerSemanticToDistributedPass();
void registerLowerSemanticToDistributedPass();

}
}

#endif
