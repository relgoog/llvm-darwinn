#ifndef MLIR_TARGET_DARWINN_PARAMETERTRANSLATION_H
#define MLIR_TARGET_DARWINN_PARAMETERTRANSLATION_H

#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>

namespace mlir {

namespace func {
class FuncOp;
}

namespace darwinn {

enum class ParameterLayout { BfloatVector, Convolution, Stencil };

struct BorrowedParameterPlacement {
  Value constant;
  Value bias;
  Value conversion;
  llvm::SmallVector<Value> consumers;
  ParameterLayout layout;
  uint64_t byteOffset;
  uint64_t byteLength;
  std::optional<uint32_t> inputChannelTile;
  bool hasPackedBias;
};

struct BorrowedBiasImmediate {
  Value consumer;
  uint32_t bits;
};

struct PackedParameters {
  llvm::SmallVector<uint8_t, 0> bytes;
  llvm::SmallVector<BorrowedParameterPlacement> placements;
  llvm::SmallVector<BorrowedBiasImmediate> biasImmediates;
};

FailureOr<PackedParameters> translateParameters(func::FuncOp function);

}

void registerToDarwinnParametersTranslation();

}

#endif
