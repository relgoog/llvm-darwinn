#include "mlir/Target/Darwinn/TraversalAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::darwinn;

template <typename AttributeT, typename Verify>
static llvm::Error validateAttribute(AttributeT attribute, Verify verify) {
  if (!attribute)
    return llvm::createStringError("Missing Darwinn traversal attribute");
  std::string diagnostic;
  ScopedDiagnosticHandler handler(attribute.getContext(),
                                  [&](Diagnostic &value) {
                                    diagnostic = value.str();
                                    return success();
                                  });
  auto emit = [&] {
    return emitError(UnknownLoc::get(attribute.getContext()));
  };
  if (failed(verify(emit)))
    return llvm::createStringError(diagnostic);
  return llvm::Error::success();
}

llvm::Expected<Counter>
mlir::darwinn::convertCounter(isa::CounterAttr attribute) {
  if (llvm::Error error = validateAttribute(attribute, [&](auto emit) {
        return isa::CounterAttr::verify(
            emit, attribute.getEnd(), attribute.getStep(), attribute.getMask());
      }))
    return std::move(error);
  return Counter{attribute.getEnd(), attribute.getStep(), attribute.getMask()};
}

llvm::Expected<Prologue>
mlir::darwinn::convertPrologue(isa::PrologueAttr attribute) {
  if (llvm::Error error = validateAttribute(attribute, [&](auto emit) {
        return isa::PrologueAttr::verify(
            emit, attribute.getLoopId(), attribute.getBaseAddress(),
            attribute.getInnerLimit(), attribute.getOuterLimit(),
            attribute.getOuterStride(), attribute.getAccessBytes(),
            attribute.getPrologueTarget());
      }))
    return std::move(error);
  return Prologue{static_cast<uint8_t>(attribute.getLoopId()),
                  attribute.getBaseAddress(),
                  attribute.getInnerLimit(),
                  attribute.getOuterLimit(),
                  attribute.getOuterStride(),
                  attribute.getAccessBytes(),
                  static_cast<uint8_t>(attribute.getPrologueTarget())};
}

llvm::Expected<ByteAddressMode>
mlir::darwinn::convertByteAddressMode(isa::ByteAddressModeAttr attribute) {
  if (llvm::Error error = validateAttribute(attribute, [&](auto emit) {
        return isa::ByteAddressModeAttr::verify(
            emit, attribute.getAccessBytesLoopMap(),
            attribute.getLastAccessBytes(), attribute.getDefaultAccessBytes());
      }))
    return std::move(error);
  return ByteAddressMode{attribute.getAccessBytesLoopMap(),
                         attribute.getLastAccessBytes(),
                         attribute.getDefaultAccessBytes()};
}

llvm::Expected<SyncProducer>
mlir::darwinn::convertSyncProducer(isa::SyncProducerAttr attribute) {
  if (llvm::Error error = validateAttribute(attribute, [&](auto emit) {
        return isa::SyncProducerAttr::verify(emit, attribute.getIncrement(),
                                             attribute.getSyncFlagLoopDepth());
      }))
    return std::move(error);
  return SyncProducer{attribute.getIncrement(),
                      static_cast<uint8_t>(attribute.getSyncFlagLoopDepth())};
}

llvm::Expected<SyncWatcher>
mlir::darwinn::convertSyncWatcher(isa::SyncWatcherAttr attribute) {
  if (llvm::Error error = validateAttribute(attribute, [&](auto emit) {
        return isa::SyncWatcherAttr::verify(
            emit, attribute.getSyncFlag(), attribute.getThreadId(),
            attribute.getBykjWtLmxpJcgmyqKq(), attribute.getSyncFlagLoopDepth(),
            attribute.getInitialExpectedSyncFlagValue(),
            attribute.getWaitSyncFlagStride(), attribute.getSyncWaitValid());
      }))
    return std::move(error);
  return SyncWatcher{static_cast<TileSyncFlag>(attribute.getSyncFlag()),
                     static_cast<uint8_t>(attribute.getThreadId()),
                     attribute.getBykjWtLmxpJcgmyqKq(),
                     static_cast<uint8_t>(attribute.getSyncFlagLoopDepth()),
                     attribute.getInitialExpectedSyncFlagValue(),
                     attribute.getWaitSyncFlagStride(),
                     attribute.getSyncWaitValid()};
}

llvm::Expected<MainOperation>
mlir::darwinn::convertMainOperation(isa::MainOperationAttr attribute) {
  if (llvm::Error error = validateAttribute(attribute, [&](auto emit) {
        return isa::MainOperationAttr::verify(
            emit, attribute.getCounter(), attribute.getAlternateInnerLimit(),
            attribute.getAlternateLimitLoopId());
      }))
    return std::move(error);
  MainOperation result;
  llvm::copy(attribute.getCounter(), result.counter.begin());
  result.alternateInnerLimit = attribute.getAlternateInnerLimit();
  result.alternateLimitLoopId = attribute.getAlternateLimitLoopId();
  return result;
}

llvm::Expected<Traversal>
mlir::darwinn::convertTraversal(isa::TraversalAttr attribute) {
  if (llvm::Error error = validateAttribute(attribute, [&](auto emit) {
        return isa::TraversalAttr::verify(
            emit, attribute.getBaseAddress(), attribute.getCounter(),
            attribute.getSyncProducer(), attribute.getDoubleBufferLoop(),
            attribute.getSecondBufferOffset(),
            attribute.getInitializeInPongState(),
            attribute.getAlternateInnerLimit(),
            attribute.getAlternateLimitLoopId(), attribute.getPrologue(),
            attribute.getAdditionalPrologues(), attribute.getByteAddressMode());
      }))
    return std::move(error);
  Traversal result;
  result.baseAddress = attribute.getBaseAddress();
  for (isa::CounterAttr value : attribute.getCounter()) {
    auto converted = convertCounter(value);
    if (!converted)
      return converted.takeError();
    result.counter.push_back(*converted);
  }
  for (isa::SyncProducerAttr value : attribute.getSyncProducer()) {
    auto converted = convertSyncProducer(value);
    if (!converted)
      return converted.takeError();
    result.syncProducer.push_back(*converted);
  }
  result.doubleBufferLoop = attribute.getDoubleBufferLoop();
  result.secondBufferOffset = attribute.getSecondBufferOffset();
  result.initializeInPongState = attribute.getInitializeInPongState();
  result.alternateInnerLimit = attribute.getAlternateInnerLimit();
  result.alternateLimitLoopId = attribute.getAlternateLimitLoopId();
  if (isa::PrologueAttr value = attribute.getPrologue()) {
    auto converted = convertPrologue(value);
    if (!converted)
      return converted.takeError();
    result.prologue = *converted;
  }
  for (isa::PrologueAttr value : attribute.getAdditionalPrologues()) {
    auto converted = convertPrologue(value);
    if (!converted)
      return converted.takeError();
    result.additionalPrologues.push_back(*converted);
  }
  if (isa::ByteAddressModeAttr value = attribute.getByteAddressMode()) {
    auto converted = convertByteAddressMode(value);
    if (!converted)
      return converted.takeError();
    result.byteAddressMode = *converted;
  }
  return result;
}
