#ifndef MLIR_TARGET_DARWINN_TRAVERSAL_H
#define MLIR_TARGET_DARWINN_TRAVERSAL_H

#include "mlir/Target/Darwinn/Encoding.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include <array>
#include <cstdint>
#include <optional>

namespace mlir::darwinn {

struct Counter {
  int64_t end = 0;
  int64_t step = 0;
  bool mask = false;
};

struct MainOperation {
  std::array<uint16_t, 8> counter{};
  uint16_t alternateInnerLimit = 0;
  uint8_t alternateLimitLoopId = 0;
};

struct SyncProducer {
  bool increment = false;
  uint8_t syncFlagLoopDepth = 0;
};

struct ByteAddressMode {
  uint32_t accessBytesLoopMap = 0;
  uint64_t lastAccessBytes = 0;
  uint64_t defaultAccessBytes = 0;
};

struct Prologue {
  uint8_t loopId = 0;
  int64_t baseAddress = 0;
  uint64_t innerLimit = 0;
  uint64_t outerLimit = 0;
  uint64_t outerStride = 0;
  uint64_t accessBytes = 0;
  uint8_t prologueTarget = 0;
};

struct Traversal {
  uint64_t baseAddress = 0;
  llvm::SmallVector<Counter, 8> counter;
  llvm::SmallVector<SyncProducer, 2> syncProducer;
  uint8_t doubleBufferLoop = 0;
  int64_t secondBufferOffset = 0;
  bool initializeInPongState = false;
  int64_t alternateInnerLimit = 0;
  uint8_t alternateLimitLoopId = 0;
  std::optional<Prologue> prologue;
  llvm::SmallVector<Prologue, 1> additionalPrologues;
  std::optional<ByteAddressMode> byteAddressMode;
};

struct TraversalEncoding {
  uint8_t baseBits;
  uint8_t counterBits;
  uint8_t limitBits;
  uint8_t loopDepth;
  std::optional<uint8_t> accessBits;
  bool doubleBuffer;
  uint8_t paddingCount;
  uint8_t paddingBaseBits;
  std::optional<uint8_t> paddingTargetBits;
  uint8_t alternateCounter;
};

enum class TileSyncFlag : uint8_t {
  ActivationWrite = 0,
  ParameterRead = 1,
  PartialSumWrite = 2,
  MeshInboundFromNorth = 3,
  MeshInboundFromEast = 4,
  MeshInboundFromSouth = 5,
  MeshInboundFromWest = 6,
  MeshOutboundForNorth = 7,
  MeshOutboundForEast = 8,
  MeshOutboundForSouth = 9,
  MeshOutboundForWest = 10,
  WideToNarrowWrite = 11,
  WideToScaling = 12,
  NarrowToWideWrite = 13,
  RingBusReadA = 14,
  RingBusReadB = 15,
  RingBusWrite = 16,
  RingBusProducerA = 17,
  RingBusProducerB = 18,
  NarrowToNarrowRead = 19,
  NarrowToNarrowWrite = 20,
  ActivationRead = 21
};

struct SyncWatcher {
  TileSyncFlag syncFlag = TileSyncFlag::ActivationWrite;
  uint8_t threadId = 0;
  bool bykjWtLmxpJcgmyqKq = false;
  uint8_t syncFlagLoopDepth = 0;
  int32_t initialExpectedSyncFlagValue = 0;
  uint32_t waitSyncFlagStride = 1;
  bool syncWaitValid = false;
};

llvm::Error writeTraversal(BitWriter &writer, const Traversal &traversal,
                           const TraversalEncoding &encoding);
llvm::Error writeMainOperation(BitWriter &writer,
                               const MainOperation &operation);
llvm::Error writeSyncProducer(BitWriter &writer, const SyncProducer *producer,
                              uint8_t loopDepth);
llvm::Error writeSyncWatcher(BitWriter &writer, const SyncWatcher *watcher,
                             uint8_t loopDepth);

}

#endif
