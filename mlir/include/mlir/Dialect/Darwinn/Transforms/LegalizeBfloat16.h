#ifndef MLIR_DIALECT_DARWINN_TRANSFORMS_LEGALIZEBFLOAT16_H
#define MLIR_DIALECT_DARWINN_TRANSFORMS_LEGALIZEBFLOAT16_H

#include <memory>

namespace mlir {
class Pass;

namespace darwinn {
std::unique_ptr<Pass> createLegalizeBfloat16Pass();
void registerLegalizeBfloat16Pass();
}
}

#endif
