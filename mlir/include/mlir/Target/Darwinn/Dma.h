#ifndef MLIR_TARGET_DARWINN_DMA_H
#define MLIR_TARGET_DARWINN_DMA_H

#include "mlir/Target/Darwinn/Compression.h"
#include "mlir/Target/Darwinn/Encoding.h"
#include "mlir/Target/Darwinn/Traversal.h"
#include "llvm/ADT/SmallVector.h"
#include <array>
#include <cstdint>
#include <optional>
#include <variant>

namespace mlir::darwinn {

enum class RingConsumerId : uint8_t { ConsumerA, ConsumerB };
enum class RingDestination : uint8_t { NarrowMemory, WideMemory };

struct ByteFilter {
  uint8_t firstDiscardByteLoopMap = 0;
  uint8_t discardByteLoopMap = 0;
  uint32_t firstDiscardByteCount = 0;
  uint32_t discardByteCount = 0;
  uint32_t lastDiscardByteCount = 0;
};

struct DmaWatcher {
  SyncWatcher watcher;
  bool stallTtuId = false;
};

struct RingConsumer {
  RingConsumerId consumer = RingConsumerId::ConsumerA;
  Traversal traversal;
  bool baseAddressOverride = false;
  llvm::SmallVector<DmaWatcher, 6> watchers;
  std::array<bool, 8> virtualChannelSubscription{};
  RingDestination destination = RingDestination::NarrowMemory;
  ByteFilter filter;
  std::array<bool, 4> threadMulticastBitmap{};
};

struct RingProducer {
  Traversal traversal;
  bool baseAddressOverride = false;
  llvm::SmallVector<DmaWatcher, 6> watchers;
  std::array<bool, 8> virtualChannelSubscription{};
  std::array<bool, 17> targets{};
  RingConsumerId consumer = RingConsumerId::ConsumerA;
};

enum class MeshDirection : uint8_t {
  OutboundNorthInboundSouth,
  OutboundEastInboundWest,
  OutboundWestInboundEast,
  OutboundSouthInboundNorth
};

enum class MeshReduction : uint8_t { None = 0, Forward = 1, Consume = 2 };

struct Mesh {
  MeshDirection direction = MeshDirection::OutboundNorthInboundSouth;
  Traversal read;
  Traversal write;
  bool baseAddressOverride = false;
  llvm::SmallVector<DmaWatcher, 6> readWatchers;
  llvm::SmallVector<DmaWatcher, 6> writeWatchers;
  std::array<uint8_t, 16> immediateValue{};
  uint8_t validBytes = 0;
  bool forwardingMode = false;
  MeshReduction reduction = MeshReduction::None;
  bool dataTypeFloat = false;
};

struct WideToNarrow {
  Traversal read;
  Traversal write;
  bool baseAddressOverride = false;
  llvm::SmallVector<DmaWatcher, 6> readWatchers;
  llvm::SmallVector<DmaWatcher, 6> writeWatchers;
  bool transpose = false;
  bool doubleOperandMode = false;
  uint8_t wideMemoryLoadStoreLoopId = 0;
  bool isScalingFactorBias = false;
  uint8_t zInBundleValidCount = 1;
  std::array<bool, 4> threadMulticastBitmap{};
};

struct WideByteAddressMode {
  uint8_t strideUnitGranulesLoopMap = 0;
  uint8_t defaultStrideUnitGranules = 1;
  uint8_t lastStrideUnitGranules = 1;
  uint8_t cellStride = 1;
  uint8_t cellStrideGroupCountLoopMap = 0;
  uint8_t defaultCellStrideGroupCount = 1;
  uint8_t lastCellStrideGroupCount = 1;
  uint32_t immediateValue = 0;
};

struct NarrowToWide {
  Traversal read;
  Traversal write;
  bool baseAddressOverride = false;
  llvm::SmallVector<DmaWatcher, 6> readWatchers;
  llvm::SmallVector<DmaWatcher, 6> writeWatchers;
  WideByteAddressMode byteAddress;
  bool transpose = false;
  std::array<bool, 4> threadMulticastBitmap{};
};

struct NarrowToNarrow {
  Traversal read;
  Traversal write;
};

using DmaOperation = std::variant<RingConsumer, RingProducer, Mesh,
                                  WideToNarrow, NarrowToWide, NarrowToNarrow>;

struct DmaInstruction {
  TileHeader header;
  OverwriteInfo overwrite;
  std::array<bool, 8> registerSourcedOperands{};
  DmaOperation operation;

  llvm::Expected<InstructionBytes> encodeUncompressed() const;
};

class DmaEncoder {
public:
  explicit DmaEncoder(
      std::optional<CompressionLayout> compression = std::nullopt)
      : compression(compression) {}

  void reset() { previous.reset(); }
  llvm::Expected<InstructionBytes> encode(const DmaInstruction &instruction);

private:
  std::optional<CompressionLayout> compression;
  std::optional<BitWriter> previous;
};

}

#endif
