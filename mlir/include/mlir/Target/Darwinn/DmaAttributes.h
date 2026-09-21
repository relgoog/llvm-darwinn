#ifndef MLIR_TARGET_DARWINN_DMAATTRIBUTES_H
#define MLIR_TARGET_DARWINN_DMAATTRIBUTES_H

#include "mlir/Target/Darwinn/Dma.h"

namespace mlir {
class Operation;
}

namespace mlir::darwinn {

llvm::Expected<DmaInstruction> convertDmaInstruction(Operation *operation);

}

#endif
