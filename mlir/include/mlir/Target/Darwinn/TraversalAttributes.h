#ifndef MLIR_TARGET_DARWINN_TRAVERSALATTRIBUTES_H
#define MLIR_TARGET_DARWINN_TRAVERSALATTRIBUTES_H

#include "mlir/Dialect/Darwinn/IR/InstructionAttrs.h"
#include "mlir/Target/Darwinn/Traversal.h"

namespace mlir::darwinn {

llvm::Expected<Counter> convertCounter(isa::CounterAttr attribute);
llvm::Expected<Prologue> convertPrologue(isa::PrologueAttr attribute);
llvm::Expected<ByteAddressMode>
convertByteAddressMode(isa::ByteAddressModeAttr attribute);
llvm::Expected<SyncProducer>
convertSyncProducer(isa::SyncProducerAttr attribute);
llvm::Expected<SyncWatcher> convertSyncWatcher(isa::SyncWatcherAttr attribute);
llvm::Expected<MainOperation>
convertMainOperation(isa::MainOperationAttr attribute);
llvm::Expected<Traversal> convertTraversal(isa::TraversalAttr attribute);

}

#endif
