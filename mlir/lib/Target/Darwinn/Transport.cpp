#include "mlir/Target/Darwinn/Transport.h"
#include "llvm/Support/Error.h"

using namespace mlir::darwinn;

namespace {

const TraversalEncoding hibTraversal{64, 32, 32,           4, 32, false,
                                     0,  32, std::nullopt, 0};
const TraversalEncoding popTraversal{
    15, 15, 15, 5, std::nullopt, true, 0, 15, std::nullopt, 0};
const TraversalEncoding infeedTraversal{15, 15, 15,           5, 5, true,
                                        1,  15, std::nullopt, 0};
const TraversalEncoding outfeedTraversal{18, 18, 18,           5, 5, true,
                                         0,  18, std::nullopt, 0};

llvm::Error
writeScalarWatcher(BitWriter &writer,
                   const std::optional<ScalarSyncWatcher> &watcher) {
  if (!watcher)
    return writer.write(0, 59);
  if (watcher->loopDepth > 5)
    return llvm::createStringError(
        "G5 scalar sync watcher selects an unavailable loop");
  if (static_cast<uint8_t>(watcher->flag) > 16)
    return llvm::createStringError("G5 scalar sync flag is unavailable");

  if (auto error = writer.write(static_cast<uint8_t>(watcher->flag), 5))
    return error;
  if (auto error = writer.write(watcher->loopDepth, 3))
    return error;
  if (auto error = writer.write(watcher->initialValue, 25))
    return error;
  if (auto error = writer.write(watcher->stride, 25))
    return error;
  return writer.write(watcher->valid, 1);
}

}

llvm::Expected<InstructionBytes>
mlir::darwinn::encodeHibDma(Header header, uint32_t tag, const HibDma &dma) {
  if (!dma.traversal.syncProducer.empty())
    return llvm::createStringError(
        "G5 host DMA does not encode sync producers");
  if (static_cast<uint8_t>(dma.queue) > 3)
    return llvm::createStringError("G5 host DMA queue is unavailable");

  BitWriter writer;
  if (auto error = writeCommonHeader(writer, header, 41, false))
    return std::move(error);
  if (auto error = writer.write(tag, 20))
    return std::move(error);
  if (auto error = writer.write(static_cast<uint8_t>(dma.queue), 3))
    return std::move(error);
  if (auto error = writer.write(0, 2))
    return std::move(error);
  if (auto error = writeTraversal(writer, dma.traversal, hibTraversal))
    return std::move(error);
  return writer.finish();
}

llvm::Expected<InstructionBytes>
mlir::darwinn::encodePopInput(Header header, uint32_t tag,
                              const PopInput &input) {
  if (input.traversal.syncProducer.size() > 1)
    return llvm::createStringError("G5 pop input supports one sync producer");

  BitWriter writer;
  if (auto error = writeCommonHeader(writer, header, 37, false))
    return std::move(error);
  if (auto error = writer.write(tag, 20))
    return std::move(error);
  if (auto error = writer.write(static_cast<uint8_t>(input.fifo), 1))
    return std::move(error);
  if (auto error = writeTraversal(writer, input.traversal, popTraversal))
    return std::move(error);

  const auto &producers = input.traversal.syncProducer;
  if (auto error = writeSyncProducer(
          writer, producers.empty() ? nullptr : &producers.front(), 5))
    return std::move(error);
  if (auto error = writeScalarWatcher(writer, input.watcher))
    return std::move(error);
  return writer.finish();
}

llvm::Expected<InstructionBytes>
mlir::darwinn::encodeRingInfeed(Header header, uint32_t tag,
                                const RingInfeed &infeed) {
  if (infeed.traversal.syncProducer.size() > 2)
    return llvm::createStringError(
        "G5 ring infeed supports two sync producers");
  if (static_cast<uint8_t>(infeed.fifo) > 1)
    return llvm::createStringError("G5 ring infeed FIFO is unavailable");

  BitWriter writer;
  if (auto error = writeCommonHeader(writer, header, 38, false))
    return std::move(error);
  if (auto error = writer.write(tag, 20))
    return std::move(error);
  if (auto error = writer.write(static_cast<uint8_t>(infeed.fifo), 2))
    return std::move(error);
  if (auto error = writeTraversal(writer, infeed.traversal, infeedTraversal))
    return std::move(error);

  const auto &producers = infeed.traversal.syncProducer;
  for (size_t index = 0; index != 2; ++index) {
    if (auto error = writeSyncProducer(
            writer, index < producers.size() ? &producers[index] : nullptr, 5))
      return std::move(error);
  }

  for (const auto &watcher : infeed.watchers) {
    if (auto error = writeScalarWatcher(writer, watcher))
      return std::move(error);
  }

  if (auto error = writer.write(infeed.syncIncrement, 25))
    return std::move(error);
  if (auto error = writer.writeBitmap(infeed.virtualChannels))
    return std::move(error);
  if (auto error = writer.writeBitmap(infeed.targets))
    return std::move(error);
  if (auto error = writer.write(static_cast<uint8_t>(infeed.bitmapSource), 1))
    return std::move(error);
  if (auto error =
          writer.write(static_cast<uint8_t>(infeed.baseAddressSource), 1))
    return std::move(error);

  for (FieldSource source : infeed.counterSources) {
    if (auto error = writer.write(static_cast<uint8_t>(source), 1))
      return std::move(error);
  }

  if (auto error =
          writer.write(infeed.unsigned8ToBfloatZeroPoint.has_value(), 1))
    return std::move(error);
  if (auto error =
          writer.write(infeed.unsigned8ToBfloatZeroPoint.value_or(0), 8))
    return std::move(error);
  return writer.finish();
}

llvm::Expected<InstructionBytes>
mlir::darwinn::encodeRingOutfeed(Header header, uint32_t tag,
                                 const RingOutfeed &outfeed) {
  if (outfeed.traversal.syncProducer.size() > 1)
    return llvm::createStringError(
        "G5 ring outfeed supports one sync producer");
  if (outfeed.bytesToPop == 0)
    return llvm::createStringError(
        "G5 ring outfeed must pop at least one byte");

  BitWriter writer;
  if (auto error = writeCommonHeader(writer, header, 39, false))
    return std::move(error);
  if (auto error = writer.write(tag, 20))
    return std::move(error);
  if (auto error = writeTraversal(writer, outfeed.traversal, outfeedTraversal))
    return std::move(error);

  const auto &producers = outfeed.traversal.syncProducer;
  if (auto error = writeSyncProducer(
          writer, producers.empty() ? nullptr : &producers.front(), 5))
    return std::move(error);
  if (auto error = writeScalarWatcher(writer, outfeed.watcher))
    return std::move(error);
  if (auto error =
          writer.write(static_cast<uint8_t>(outfeed.baseAddressSource), 1))
    return std::move(error);

  for (FieldSource source : outfeed.counterSources) {
    if (auto error = writer.write(static_cast<uint8_t>(source), 1))
      return std::move(error);
  }

  if (auto error = writer.writeBitmap(outfeed.virtualChannels))
    return std::move(error);
  if (auto error = writer.write(1, 1))
    return std::move(error);
  if (auto error = writer.write(outfeed.bytesToPop, 32))
    return std::move(error);
  return writer.finish();
}
