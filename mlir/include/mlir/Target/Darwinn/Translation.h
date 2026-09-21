#ifndef MLIR_TARGET_DARWINN_TRANSLATION_H
#define MLIR_TARGET_DARWINN_TRANSLATION_H

#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/Darwinn/Encoding.h"

namespace mlir {

namespace darwinn {
namespace isa {
class ProgramOp;
}

FailureOr<llvm::SmallVector<InstructionBytes, 0>>
translateProgram(isa::ProgramOp program);
}

void registerToDarwinnTranslation();

}

#endif
