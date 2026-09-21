#include "mlir/Target/Darwinn/Dma.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <utility>

using namespace mlir::darwinn;

static llvm::Expected<uint8_t> getOpcode(const RingConsumer &consumer) {
  switch (consumer.consumer) {
  case RingConsumerId::ConsumerA:
    return 17;
  case RingConsumerId::ConsumerB:
    return 18;
  }

  return llvm::createStringError("Invalid ring DMA consumer");
}

static llvm::Expected<uint8_t> getOpcode(const RingProducer &) { return 16; }

static llvm::Expected<uint8_t> getOpcode(const Mesh &mesh) {
  switch (mesh.direction) {
  case MeshDirection::OutboundNorthInboundSouth:
    return 23;
  case MeshDirection::OutboundEastInboundWest:
    return 24;
  case MeshDirection::OutboundWestInboundEast:
    return 22;
  case MeshDirection::OutboundSouthInboundNorth:
    return 21;
  }

  return llvm::createStringError("Invalid mesh DMA direction");
}

static llvm::Expected<uint8_t> getOpcode(const WideToNarrow &) { return 20; }
static llvm::Expected<uint8_t> getOpcode(const NarrowToWide &) { return 19; }

static llvm::Error writeRingTransfer(BitWriter &writer,
                                     const Traversal &traversal,
                                     bool baseAddressOverride,
                                     llvm::ArrayRef<DmaWatcher> watchers,
                                     size_t producerCount) {
  if (traversal.syncProducer.size() > producerCount)
    return llvm::createStringError("Too many ring DMA sync producers");
  if (watchers.size() > 6)
    return llvm::createStringError(
        "Ring DMA supports at most six sync watchers");

  TraversalEncoding encoding{20, 20, 20, 4, 20, true, 1, 20, std::nullopt, 0};
  if (auto error = writeTraversal(writer, traversal, encoding))
    return error;
  if (auto error = writer.write(baseAddressOverride, 1))
    return error;

  for (size_t index = 0; index < producerCount; ++index) {
    const SyncProducer *producer = index < traversal.syncProducer.size()
                                       ? &traversal.syncProducer[index]
                                       : nullptr;
    if (auto error = writeSyncProducer(writer, producer, 4))
      return error;
  }

  for (size_t index = 0; index < 6; ++index) {
    const DmaWatcher *watcher =
        index < watchers.size() ? &watchers[index] : nullptr;
    if (auto error =
            writeSyncWatcher(writer, watcher ? &watcher->watcher : nullptr, 4))
      return error;
    if (auto error = writer.write(watcher && watcher->stallTtuId, 1))
      return error;
  }

  return llvm::Error::success();
}

static llvm::Error writeTransferPair(BitWriter &writer, const Traversal &read,
                                     const Traversal &write,
                                     const TraversalEncoding &readEncoding,
                                     const TraversalEncoding &writeEncoding,
                                     bool baseAddressOverride,
                                     llvm::ArrayRef<DmaWatcher> readWatchers,
                                     llvm::ArrayRef<DmaWatcher> writeWatchers,
                                     size_t watcherCount, bool readSync) {
  if (read.syncProducer.size() > 1 || write.syncProducer.size() > 1)
    return llvm::createStringError(
        "DMA traversal supports at most one sync producer");
  if (!readSync && !read.syncProducer.empty())
    return llvm::createStringError(
        "Wide DMA does not support a read sync producer");
  if (readWatchers.size() > watcherCount ||
      writeWatchers.size() > watcherCount - readWatchers.size())
    return llvm::createStringError("Too many DMA sync watchers");

  if (auto error = writeTraversal(writer, read, readEncoding))
    return error;
  if (auto error = writeTraversal(writer, write, writeEncoding))
    return error;
  if (auto error = writer.write(baseAddressOverride, 1))
    return error;

  if (readSync) {
    const SyncProducer *producer =
        read.syncProducer.empty() ? nullptr : &read.syncProducer.front();
    if (auto error =
            writeSyncProducer(writer, producer, readEncoding.loopDepth))
      return error;
  }

  const SyncProducer *producer =
      write.syncProducer.empty() ? nullptr : &write.syncProducer.front();
  if (auto error = writeSyncProducer(writer, producer, writeEncoding.loopDepth))
    return error;

  uint8_t loopDepth = std::max(readEncoding.loopDepth, writeEncoding.loopDepth);
  size_t suppliedWatchers = readWatchers.size() + writeWatchers.size();

  for (size_t index = 0; index < watcherCount; ++index) {
    const DmaWatcher *watcher = nullptr;
    if (index < readWatchers.size())
      watcher = &readWatchers[index];
    else if (index < suppliedWatchers)
      watcher = &writeWatchers[index - readWatchers.size()];

    if (auto error = writeSyncWatcher(
            writer, watcher ? &watcher->watcher : nullptr, loopDepth))
      return error;
    if (auto error = writer.write(watcher && watcher->stallTtuId, 1))
      return error;
  }

  return llvm::Error::success();
}

static llvm::Error writeBody(BitWriter &writer, const RingConsumer &consumer) {
  if (consumer.destination != RingDestination::NarrowMemory &&
      consumer.destination != RingDestination::WideMemory)
    return llvm::createStringError("Invalid ring DMA destination");

  if (auto error =
          writeRingTransfer(writer, consumer.traversal,
                            consumer.baseAddressOverride, consumer.watchers, 1))
    return error;

  if (auto error = writer.writeBitmap(consumer.virtualChannelSubscription))
    return error;
  if (auto error =
          writer.write(consumer.destination == RingDestination::WideMemory, 1))
    return error;
  if (auto error = writer.write(consumer.filter.firstDiscardByteLoopMap, 4))
    return error;
  if (auto error = writer.write(consumer.filter.discardByteLoopMap, 4))
    return error;
  if (auto error = writer.write(consumer.filter.firstDiscardByteCount, 24))
    return error;
  if (auto error = writer.write(consumer.filter.discardByteCount, 24))
    return error;
  if (auto error = writer.write(consumer.filter.lastDiscardByteCount, 24))
    return error;
  return writer.writeBitmap(consumer.threadMulticastBitmap);
}

static llvm::Error writeBody(BitWriter &writer, const RingProducer &producer) {
  if (producer.consumer != RingConsumerId::ConsumerA &&
      producer.consumer != RingConsumerId::ConsumerB)
    return llvm::createStringError("Invalid ring DMA consumer");

  if (auto error =
          writeRingTransfer(writer, producer.traversal,
                            producer.baseAddressOverride, producer.watchers, 2))
    return error;

  if (auto error = writer.writeBitmap(producer.virtualChannelSubscription))
    return error;
  if (auto error = writer.writeBitmap(producer.targets))
    return error;
  return writer.write(producer.consumer == RingConsumerId::ConsumerB, 1);
}

static llvm::Error writeBody(BitWriter &writer, const Mesh &mesh) {
  if (mesh.validBytes > 16)
    return llvm::createStringError(
        "Mesh immediate value cannot exceed sixteen bytes");

  uint8_t reduction = static_cast<uint8_t>(mesh.reduction);
  if (reduction > static_cast<uint8_t>(MeshReduction::Consume))
    return llvm::createStringError("Invalid mesh DMA reduction");

  TraversalEncoding encoding{20, 20, 20, 5, 20, true, 0, 20, std::nullopt, 0};
  if (auto error =
          writeTransferPair(writer, mesh.read, mesh.write, encoding, encoding,
                            mesh.baseAddressOverride, mesh.readWatchers,
                            mesh.writeWatchers, 12, true))
    return error;

  if (auto error = writer.write(mesh.validBytes, 5))
    return error;

  for (uint8_t byte : mesh.immediateValue) {
    if (auto error = writer.write(byte, 8))
      return error;
  }

  if (auto error = writer.write(mesh.forwardingMode, 1))
    return error;
  if (auto error = writer.write(reduction, 2))
    return error;
  return writer.write(mesh.dataTypeFloat, 1);
}

static llvm::Error writeBody(BitWriter &writer, const WideToNarrow &transfer) {
  if (!llvm::isPowerOf2_32(transfer.zInBundleValidCount) ||
      transfer.zInBundleValidCount > 8)
    return llvm::createStringError(
        "Wide DMA valid bundle count must be one, two, four or eight");

  TraversalEncoding readEncoding{18, 18, 18,           4, std::nullopt, true,
                                 0,  18, std::nullopt, 0};
  TraversalEncoding writeEncoding = readEncoding;
  writeEncoding.loopDepth = 6;
  writeEncoding.paddingCount = 1;
  writeEncoding.alternateCounter = transfer.transpose;

  if (auto error = writeTransferPair(
          writer, transfer.read, transfer.write, readEncoding, writeEncoding,
          transfer.baseAddressOverride, transfer.readWatchers,
          transfer.writeWatchers, 6, false))
    return error;

  if (auto error = writer.write(transfer.isScalingFactorBias, 1))
    return error;
  if (auto error = writer.write(transfer.transpose, 1))
    return error;
  if (auto error = writer.write(transfer.doubleOperandMode, 1))
    return error;
  if (auto error = writer.write(transfer.wideMemoryLoadStoreLoopId, 3))
    return error;
  if (auto error = writer.write(llvm::Log2_32(transfer.zInBundleValidCount), 2))
    return error;
  return writer.writeBitmap(transfer.threadMulticastBitmap);
}

static llvm::Error writeNonzero(BitWriter &writer, uint8_t value,
                                unsigned width) {
  if (!value)
    return llvm::createStringError(
        "Wide DMA stride and group counts must be positive");
  return writer.write(value - 1, width);
}

static llvm::Error writeBody(BitWriter &writer, const NarrowToWide &transfer) {
  TraversalEncoding readEncoding{20, 20, 20,           6, 20, true,
                                 1,  20, std::nullopt, 0};
  TraversalEncoding writeEncoding{12, 12, 12,           4, 7, true,
                                  1,  12, std::nullopt, 0};

  if (auto error = writeTransferPair(
          writer, transfer.read, transfer.write, readEncoding, writeEncoding,
          transfer.baseAddressOverride, transfer.readWatchers,
          transfer.writeWatchers, 6, false))
    return error;

  if (auto error =
          writer.write(transfer.byteAddress.strideUnitGranulesLoopMap, 4))
    return error;
  if (auto error = writeNonzero(
          writer, transfer.byteAddress.defaultStrideUnitGranules, 7))
    return error;
  if (auto error =
          writeNonzero(writer, transfer.byteAddress.lastStrideUnitGranules, 7))
    return error;
  if (auto error = writeNonzero(writer, transfer.byteAddress.cellStride, 5))
    return error;
  if (auto error =
          writer.write(transfer.byteAddress.cellStrideGroupCountLoopMap, 4))
    return error;
  if (auto error = writeNonzero(
          writer, transfer.byteAddress.defaultCellStrideGroupCount, 5))
    return error;
  if (auto error = writeNonzero(
          writer, transfer.byteAddress.lastCellStrideGroupCount, 5))
    return error;
  if (auto error = writer.write(transfer.byteAddress.immediateValue, 32))
    return error;
  if (auto error = writer.write(0, 8))
    return error;
  if (auto error = writer.write(0, 2))
    return error;
  if (auto error = writer.write(3, 2))
    return error;
  return writer.writeBitmap(transfer.threadMulticastBitmap);
}

llvm::Expected<InstructionBytes>
DmaEncoder::encode(const DmaInstruction &instruction) {
  auto opcode =
      std::visit([](const auto &operation) { return getOpcode(operation); },
                 instruction.operation);
  if (!opcode)
    return opcode.takeError();

  BitWriter writer;
  if (auto error = writeTileComputeHeader(writer, instruction.header, *opcode,
                                          instruction.overwrite,
                                          instruction.registerSourcedOperands))
    return error;

  std::optional<BitWriter> body;

  if (compression) {
    body.emplace();
    if (auto error = std::visit(
            [&](const auto &operation) { return writeBody(*body, operation); },
            instruction.operation))
      return error;

    auto compressed =
        compression->writeBody(writer, previous ? &*previous : nullptr, *body);
    if (!compressed)
      return compressed.takeError();
  } else {
    if (auto error = writer.write(0, 1))
      return error;
    if (auto error = std::visit(
            [&](const auto &operation) { return writeBody(writer, operation); },
            instruction.operation))
      return error;
  }

  auto encoded = writer.finish();
  if (body)
    previous = std::move(body);
  return encoded;
}

llvm::Expected<InstructionBytes> DmaInstruction::encodeUncompressed() const {
  return DmaEncoder().encode(*this);
}
