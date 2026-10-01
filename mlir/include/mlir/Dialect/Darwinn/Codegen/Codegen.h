#ifndef MLIR_DIALECT_DARWINN_CODEGEN_CODEGEN_H
#define MLIR_DIALECT_DARWINN_CODEGEN_CODEGEN_H

#include <memory>

namespace mlir {
class Pass;

namespace darwinn {

std::unique_ptr<Pass> createLegalizeDwcPass();
void registerLegalizeDwcPass();

} // namespace darwinn
} // namespace mlir

#endif
