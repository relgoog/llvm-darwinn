#ifndef MLIR_DIALECT_DARWINN_TRANSFORMS_OPTIMIZEPOINTWISE_H
#define MLIR_DIALECT_DARWINN_TRANSFORMS_OPTIMIZEPOINTWISE_H

#include <memory>

namespace mlir {
class Pass;

namespace darwinn {
std::unique_ptr<Pass> createOptimizePointwisePass();
void registerOptimizePointwisePass();
}
}

#endif
