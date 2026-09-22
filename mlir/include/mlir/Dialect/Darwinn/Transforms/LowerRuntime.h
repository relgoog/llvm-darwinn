#ifndef MLIR_DIALECT_DARWINN_TRANSFORMS_LOWERRUNTIME_H
#define MLIR_DIALECT_DARWINN_TRANSFORMS_LOWERRUNTIME_H

#include "mlir/Support/LogicalResult.h"

namespace mlir {
class ModuleOp;

namespace darwinn {
LogicalResult lowerDiveVmRuntime(ModuleOp module);
}
}

#endif
