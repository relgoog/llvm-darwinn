#ifndef MLIR_TARGET_DARWINN_ENCODINGATTRIBUTES_H
#define MLIR_TARGET_DARWINN_ENCODINGATTRIBUTES_H

#include "mlir/Dialect/Darwinn/IR/InstructionAttrs.h"
#include "mlir/Target/Darwinn/Serialization.h"

namespace mlir::darwinn {

llvm::Expected<Header> convertHeader(isa::HeaderAttr attribute);
llvm::Expected<TileHeader> convertTileHeader(isa::TileHeaderAttr attribute);
llvm::Expected<OverwriteInfo>
convertOverwrite(isa::OverwriteInfoAttr attribute);
llvm::Expected<Instruction> convertCoreInstruction(Operation *operation);

}

#endif
