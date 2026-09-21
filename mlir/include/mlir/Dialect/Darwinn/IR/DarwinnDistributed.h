#ifndef MLIR_DIALECT_DARWINN_IR_DARWINNDISTRIBUTED_H
#define MLIR_DIALECT_DARWINN_IR_DARWINNDISTRIBUTED_H

#include "mlir/IR/BuiltinTypes.h"
#include <cstdint>
#include <optional>

namespace mlir::darwinn {

enum class DistributedMemorySpace : uint8_t {
  HostMemory,
  TileMemory,
  TileRegisters
};

}

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/Darwinn/IR/DarwinnTypes.h.inc"

#endif
