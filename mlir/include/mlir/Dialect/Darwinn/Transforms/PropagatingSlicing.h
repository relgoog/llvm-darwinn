#ifndef MLIR_DIALECT_DARWINN_TRANSFORMS_PROPAGATINGSLICING_H
#define MLIR_DIALECT_DARWINN_TRANSFORMS_PROPAGATINGSLICING_H

#include <memory>

namespace mlir {
class Pass;

namespace darwinn {

std::unique_ptr<Pass> createPropagatingSlicingPass();
void registerPropagatingSlicingPass();

}
}

#endif
