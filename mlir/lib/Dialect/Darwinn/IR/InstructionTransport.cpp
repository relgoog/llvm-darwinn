#include "mlir/Dialect/Darwinn/IR/InstructionOps.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::darwinn::isa;

LogicalResult ScalarSyncWatcherAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError, ScalarSyncFlag flag,
    unsigned loopDepth, uint32_t initialValue, uint32_t stride, bool) {
  if (static_cast<uint32_t>(flag) > 16)
    return emitError() << "scalar sync flag is unavailable on G5";
  if (loopDepth > 5)
    return emitError() << "scalar sync watcher loop depth exceeds 5";
  if (initialValue >= (1u << 25) || stride >= (1u << 25))
    return emitError() << "scalar sync values must fit in 25 bits";
  return success();
}

LogicalResult
FieldSourcesAttr::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                         ArrayRef<FieldSourceAttr> values) {
  if (values.size() != 10)
    return emitError() << "scalar traversal requires 10 field sources";
  if (llvm::any_of(values, [](FieldSourceAttr value) { return !value; }))
    return emitError() << "field sources must not contain null attributes";
  return success();
}

LogicalResult
InfeedConversionAttr::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                             InfeedConversion mode, unsigned zeroPoint) {
  switch (mode) {
  case InfeedConversion::None:
    if (zeroPoint != 0)
      return emitError()
             << "disabled infeed conversion requires a zero point of 0";
    return success();
  case InfeedConversion::Unsigned8ToBfloat:
    if (zeroPoint > 255)
      return emitError() << "infeed conversion zero point must fit in 8 bits";
    return success();
  }
  return emitError() << "infeed conversion is unavailable on G5";
}

namespace {

LogicalResult verifyTransport(Operation *operation, HeaderAttr header,
                              uint64_t tag, TraversalAttr traversal,
                              unsigned loopDepth, unsigned producers,
                              bool doubleBuffer, bool prologue,
                              bool byteAddress) {
  if (header.getEnableSequencerOverwrite())
    return operation->emitOpError("does not support sequencer overwrite on G5");
  if (tag >= (1u << 20))
    return operation->emitOpError("tag must fit in 20 unsigned bits");
  if (!traversal.getCounter().empty() &&
      traversal.getCounter().size() != loopDepth)
    return operation->emitOpError()
           << "requires " << loopDepth << " traversal counters";
  if (traversal.getSyncProducer().size() > producers)
    return operation->emitOpError()
           << "supports at most " << producers << " sync producers";
  if (!doubleBuffer && (traversal.getDoubleBufferLoop() != 0 ||
                        traversal.getSecondBufferOffset() != 0 ||
                        traversal.getInitializeInPongState()))
    return operation->emitOpError("does not support double buffering");
  if ((!prologue && traversal.getPrologue()) ||
      !traversal.getAdditionalPrologues().empty())
    return operation->emitOpError("contains an unsupported traversal prologue");
  if (!byteAddress && traversal.getByteAddressMode())
    return operation->emitOpError("does not support byte address mode");
  return success();
}

}

LogicalResult HibDmaOp::verify() {
  return verifyTransport(getOperation(), getHeader(), getTag(), getTraversal(),
                         4, 0, false, false, true);
}

LogicalResult PopInputOp::verify() {
  return verifyTransport(getOperation(), getHeader(), getTag(), getTraversal(),
                         5, 1, true, false, false);
}

LogicalResult RingInfeedOp::verify() {
  if (failed(verifyTransport(getOperation(), getHeader(), getTag(),
                             getTraversal(), 5, 2, true, true, true)))
    return failure();
  if (getSyncIncrement() >= (1u << 25))
    return emitOpError("sync increment must fit in 25 unsigned bits");
  if (getVirtualChannels().size() != 8)
    return emitOpError("requires 8 virtual channel bits");
  if (getTargets().size() != 17)
    return emitOpError("requires 17 target bits");
  return success();
}

LogicalResult RingOutfeedOp::verify() {
  if (failed(verifyTransport(getOperation(), getHeader(), getTag(),
                             getTraversal(), 5, 1, true, false, true)))
    return failure();
  if (getVirtualChannels().size() != 8)
    return emitOpError("requires 8 virtual channel bits");
  if (getBytesToPop() == 0 || getBytesToPop() > UINT32_MAX)
    return emitOpError("bytes to pop must be between 1 and 4294967295");
  return success();
}
