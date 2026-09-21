#ifndef MLIR_TARGET_DARWINN_TENSORATTRIBUTES_H
#define MLIR_TARGET_DARWINN_TENSORATTRIBUTES_H

#include "mlir/Dialect/Darwinn/IR/InstructionOps.h"
#include "mlir/Target/Darwinn/Serialization.h"
#include "llvm/Support/Error.h"

namespace mlir::darwinn {

llvm::Expected<ComputePacket> convertTensor(isa::TensorOp operation);

}

#endif
