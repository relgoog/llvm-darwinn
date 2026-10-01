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
std::unique_ptr<Pass> createShardingPass();
void registerShardingPass();
std::unique_ptr<Pass> createNodePartitionPass();
void registerNodePartitionPass();
std::unique_ptr<Pass> createTransferCanonicalizationPass();
void registerTransferCanonicalizationPass();
std::unique_ptr<Pass> createRedistributeOptimizationPass();
void registerRedistributeOptimizationPass();
std::unique_ptr<Pass> createRemoveParameterRedistributesPass();
void registerRemoveParameterRedistributesPass();
std::unique_ptr<Pass> createSpillFillOptimizationPass();
void registerSpillFillOptimizationPass();
std::unique_ptr<Pass> createStaticToStreamingPass();
void registerStaticToStreamingPass();
std::unique_ptr<Pass> createOpReorderingForStreamingPass();
void registerOpReorderingForStreamingPass();
std::unique_ptr<Pass> createPreemptionPointsInsertionPass();
void registerPreemptionPointsInsertionPass();
std::unique_ptr<Pass> createScalarCoreStdOpsLoweringPass();
void registerScalarCoreStdOpsLoweringPass();
std::unique_ptr<Pass> createLegalizeThreadObliviousOpPass();
void registerLegalizeThreadObliviousOpPass();
std::unique_ptr<Pass> createVectorizationPass();
void registerVectorizationPass();
std::unique_ptr<Pass> createStreamingToSynchronizationPass();
void registerStreamingToSynchronizationPass();
std::unique_ptr<Pass> createLowerSynchronizedOpsToTensorOpPass();
void registerLowerSynchronizedOpsToTensorOpPass();

}
}

#endif
