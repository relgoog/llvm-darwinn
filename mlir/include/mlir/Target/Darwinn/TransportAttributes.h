#ifndef MLIR_TARGET_DARWINN_TRANSPORTATTRIBUTES_H
#define MLIR_TARGET_DARWINN_TRANSPORTATTRIBUTES_H

#include "mlir/Dialect/Darwinn/IR/InstructionOps.h"
#include "mlir/Target/Darwinn/Serialization.h"

namespace mlir::darwinn {

llvm::Expected<TaggedPacket> convertTransport(isa::HibDmaOp operation);
llvm::Expected<TaggedPacket> convertTransport(isa::PopInputOp operation);
llvm::Expected<TaggedPacket> convertTransport(isa::RingInfeedOp operation);
llvm::Expected<TaggedPacket> convertTransport(isa::RingOutfeedOp operation);

}

#endif
