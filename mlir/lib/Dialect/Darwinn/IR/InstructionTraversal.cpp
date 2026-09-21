#include "mlir/Dialect/Darwinn/IR/InstructionAttrs.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
#include <limits>

using namespace mlir;
using namespace mlir::darwinn::isa;

LogicalResult
CounterAttr::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                    int64_t end, int64_t step, bool mask) {
  if (step == 0) {
    if (end != 0 || mask)
      return emitError()
             << "a zero counter step requires a zero limit and mask";
    return success();
  }
  if (end == std::numeric_limits<int64_t>::min() && step == -1)
    return emitError() << "counter iteration count overflows";
  if (end % step != 0 || end / step < 0)
    return emitError()
           << "counter limit must be a nonnegative multiple of its step";
  return success();
}

LogicalResult
PrologueAttr::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                     unsigned loopId, int64_t, uint64_t, uint64_t, uint64_t,
                     uint64_t, PrologueTarget prologueTarget) {
  if (loopId >= 8)
    return emitError() << "prologue loop must be below eight";
  if (!symbolizePrologueTarget(static_cast<uint32_t>(prologueTarget)))
    return emitError() << "invalid prologue target";
  return success();
}

LogicalResult
ByteAddressModeAttr::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                            uint32_t accessBytesLoopMap,
                            uint64_t lastAccessBytes,
                            uint64_t defaultAccessBytes) {
  if (accessBytesLoopMap >= 256)
    return emitError() << "access byte loop map exceeds eight loops";
  if (lastAccessBytes == 0 || defaultAccessBytes == 0)
    return emitError() << "traversal accesses must contain at least one byte";
  return success();
}

LogicalResult
SyncProducerAttr::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                         bool, unsigned syncFlagLoopDepth) {
  if (syncFlagLoopDepth > 8)
    return emitError() << "sync producer loop depth exceeds eight";
  return success();
}

LogicalResult SyncWatcherAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError, TileSyncFlag syncFlag,
    unsigned threadId, bool, unsigned syncFlagLoopDepth,
    int32_t initialExpectedSyncFlagValue, uint32_t waitSyncFlagStride, bool) {
  if (!symbolizeTileSyncFlag(static_cast<uint32_t>(syncFlag)))
    return emitError() << "invalid tile sync flag";
  if (threadId >= 4)
    return emitError() << "sync watcher thread must be below four";
  if (syncFlagLoopDepth > 8)
    return emitError() << "sync watcher loop depth exceeds eight";
  if (initialExpectedSyncFlagValue < -(1 << 24) ||
      initialExpectedSyncFlagValue >= (1 << 25))
    return emitError() << "sync watcher initial value exceeds its 25 bit field";
  if (waitSyncFlagStride >= (1U << 25))
    return emitError() << "sync watcher stride exceeds its 25 bit field";
  return success();
}

LogicalResult
MainOperationAttr::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                          llvm::ArrayRef<uint16_t> counter,
                          unsigned alternateInnerLimit,
                          unsigned alternateLimitLoopId) {
  if (counter.size() != 8)
    return emitError() << "main operation requires eight counters";
  if (llvm::any_of(counter,
                   [](uint16_t limit) { return limit >= (1U << 14); }) ||
      alternateInnerLimit >= (1U << 14))
    return emitError() << "main operation limit exceeds its 14 bit field";
  if (alternateLimitLoopId >= 8)
    return emitError() << "main operation alternate loop must be below eight";
  if (alternateLimitLoopId == 0 && alternateInnerLimit != 0)
    return emitError()
           << "main operation alternate limit requires a loop selector";
  return success();
}

LogicalResult
TraversalAttr::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                      uint64_t, llvm::ArrayRef<CounterAttr> counter,
                      llvm::ArrayRef<SyncProducerAttr> syncProducer,
                      unsigned doubleBufferLoop, int64_t,
                      bool initializeInPongState, int64_t alternateInnerLimit,
                      unsigned alternateLimitLoopId, PrologueAttr prologue,
                      llvm::ArrayRef<PrologueAttr> additionalPrologues,
                      ByteAddressModeAttr byteAddressMode) {
  unsigned depth = counter.size();
  if (depth != 0 && depth != 4 && depth != 5 && depth != 6 && depth != 8)
    return emitError()
           << "traversal requires zero, four, five, six or eight counters";
  if (syncProducer.size() > 2)
    return emitError() << "traversal supports at most two sync producers";
  if (additionalPrologues.size() > 7)
    return emitError() << "traversal supports at most eight prologues";
  for (CounterAttr value : counter) {
    if (!value)
      return emitError() << "traversal counter cannot be null";
    if (failed(CounterAttr::verify(emitError, value.getEnd(), value.getStep(),
                                   value.getMask())))
      return failure();
  }
  if (!counter.empty() && counter.front().getStep() == 0 &&
      llvm::any_of(counter,
                   [](CounterAttr value) { return value.getStep() != 0; }))
    return emitError()
           << "disabled traversal counters must all have zero steps";
  if (doubleBufferLoop >= (depth ? depth : 1) ||
      (depth == 0 && initializeInPongState))
    return emitError() << "double buffer loop selects an unavailable counter";
  if (alternateLimitLoopId >= (depth ? depth : 1))
    return emitError() << "alternate limit selects an unavailable counter";
  if (alternateLimitLoopId == 0 && alternateInnerLimit != 0)
    return emitError() << "alternate limit requires a loop selector";

  for (SyncProducerAttr value : syncProducer) {
    if (!value)
      return emitError() << "traversal sync producer cannot be null";
    if (failed(SyncProducerAttr::verify(emitError, value.getIncrement(),
                                        value.getSyncFlagLoopDepth())))
      return failure();
    if (value.getSyncFlagLoopDepth() > depth)
      return emitError() << "sync producer selects an unavailable loop depth";
  }

  auto verifyPrologue = [&](PrologueAttr value) -> LogicalResult {
    if (!value)
      return emitError() << "traversal prologue cannot be null";
    if (failed(PrologueAttr::verify(
            emitError, value.getLoopId(), value.getBaseAddress(),
            value.getInnerLimit(), value.getOuterLimit(),
            value.getOuterStride(), value.getAccessBytes(),
            value.getPrologueTarget())))
      return failure();
    if (value.getLoopId() >= depth)
      return emitError() << "prologue selects an unavailable loop";
    if (value.getAccessBytes() != 0 && !byteAddressMode)
      return emitError() << "prologue access bytes require byte address mode";
    return success();
  };
  if (prologue && failed(verifyPrologue(prologue)))
    return failure();
  for (PrologueAttr value : additionalPrologues)
    if (failed(verifyPrologue(value)))
      return failure();
  if (byteAddressMode) {
    if (failed(ByteAddressModeAttr::verify(
            emitError, byteAddressMode.getAccessBytesLoopMap(),
            byteAddressMode.getLastAccessBytes(),
            byteAddressMode.getDefaultAccessBytes())))
      return failure();
    if (!llvm::isUIntN(depth, byteAddressMode.getAccessBytesLoopMap()))
      return emitError() << "access byte loop map selects an unavailable loop";
  }
  return success();
}
