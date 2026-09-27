#ifndef MLIR_DIALECT_DARWINN_TRANSFORMS_DISTRIBUTEDPASSES_H
#define MLIR_DIALECT_DARWINN_TRANSFORMS_DISTRIBUTEDPASSES_H

#include <memory>

namespace mlir {
class Pass;

namespace darwinn {

std::unique_ptr<Pass> createPropagatingSlicingPass();
void registerPropagatingSlicingPass();
std::unique_ptr<Pass> createMoveRedistributeToHostPass();
void registerMoveRedistributeToHostPass();
std::unique_ptr<Pass> createNodePartitionPass();
void registerNodePartitionPass();
std::unique_ptr<Pass> createTransferCanonicalizationPass();
void registerTransferCanonicalizationPass();

}
}

#endif
