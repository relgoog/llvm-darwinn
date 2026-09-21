#include "mlir/Dialect/Darwinn/IR/InstructionOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace mlir::darwinn::isa;

static LogicalResult
verifyBitmap(llvm::function_ref<InFlightDiagnostic()> emitError,
             DenseBoolArrayAttr bitmap, size_t count, StringRef name) {
  if (!bitmap || static_cast<size_t>(bitmap.size()) != count)
    return emitError() << name << " requires " << count << " bits";
  return success();
}

static LogicalResult
verifyTraversalShape(llvm::function_ref<InFlightDiagnostic()> emitError,
                     TraversalAttr traversal, size_t loopDepth,
                     size_t producerCount) {
  if (!traversal)
    return emitError() << "DMA traversal cannot be null";
  if (!traversal.getCounter().empty() &&
      traversal.getCounter().size() != loopDepth)
    return emitError() << "DMA traversal requires " << loopDepth << " counters";
  if (traversal.getSyncProducer().size() > producerCount)
    return emitError() << "DMA traversal supports at most " << producerCount
                       << " sync producers";
  return success();
}

static LogicalResult
verifyWatchers(llvm::function_ref<InFlightDiagnostic()> emitError,
               ArrayRef<DmaWatcherAttr> read, ArrayRef<DmaWatcherAttr> write,
               size_t limit) {
  if (read.size() > limit || write.size() > limit - read.size())
    return emitError() << "DMA supports at most " << limit << " sync watchers";

  for (DmaWatcherAttr value : llvm::concat<const DmaWatcherAttr>(read, write)) {
    if (!value)
      return emitError() << "DMA watcher cannot be null";
    if (failed(DmaWatcherAttr::verify(emitError, value.getWatcher(),
                                      value.getStallTtuId())))
      return failure();
  }

  return success();
}

LogicalResult ByteFilterAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError,
    unsigned firstDiscardByteLoopMap, unsigned discardByteLoopMap,
    uint32_t firstDiscardByteCount, uint32_t discardByteCount,
    uint32_t lastDiscardByteCount) {
  if (firstDiscardByteLoopMap >= 16 || discardByteLoopMap >= 16)
    return emitError() << "byte filter loop maps require four bits";
  if (firstDiscardByteCount >= (1U << 24) || discardByteCount >= (1U << 24) ||
      lastDiscardByteCount >= (1U << 24))
    return emitError() << "byte filter counts require 24 bits";
  return success();
}

LogicalResult
DmaWatcherAttr::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                       SyncWatcherAttr watcher, bool) {
  if (!watcher)
    return emitError() << "DMA sync watcher cannot be null";
  return SyncWatcherAttr::verify(
      emitError, watcher.getSyncFlag(), watcher.getThreadId(),
      watcher.getBykjWtLmxpJcgmyqKq(), watcher.getSyncFlagLoopDepth(),
      watcher.getInitialExpectedSyncFlagValue(),
      watcher.getWaitSyncFlagStride(), watcher.getSyncWaitValid());
}

LogicalResult RingConsumerAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError, RingConsumerId consumer,
    TraversalAttr traversal, bool, ArrayRef<DmaWatcherAttr> watchers,
    DenseBoolArrayAttr virtualChannelSubscription, RingDestination destination,
    ByteFilterAttr filter, DenseBoolArrayAttr threadMulticastBitmap) {
  if (!symbolizeRingConsumerId(static_cast<uint32_t>(consumer)))
    return emitError() << "invalid ring DMA consumer";
  if (!symbolizeRingDestination(static_cast<uint32_t>(destination)))
    return emitError() << "invalid ring DMA destination";
  if (!filter)
    return emitError() << "ring DMA byte filter cannot be null";
  if (failed(ByteFilterAttr::verify(
          emitError, filter.getFirstDiscardByteLoopMap(),
          filter.getDiscardByteLoopMap(), filter.getFirstDiscardByteCount(),
          filter.getDiscardByteCount(), filter.getLastDiscardByteCount())) ||
      failed(verifyTraversalShape(emitError, traversal, 4, 1)) ||
      failed(verifyWatchers(emitError, watchers, {}, 6)) ||
      failed(verifyBitmap(emitError, virtualChannelSubscription, 8,
                          "virtual channel subscription")) ||
      failed(verifyBitmap(emitError, threadMulticastBitmap, 4,
                          "thread multicast bitmap")))
    return failure();
  return success();
}

LogicalResult
RingProducerAttr::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                         TraversalAttr traversal, bool,
                         ArrayRef<DmaWatcherAttr> watchers,
                         DenseBoolArrayAttr virtualChannelSubscription,
                         DenseBoolArrayAttr targets, RingConsumerId consumer) {
  if (!symbolizeRingConsumerId(static_cast<uint32_t>(consumer)))
    return emitError() << "invalid ring DMA consumer";
  if (failed(verifyTraversalShape(emitError, traversal, 4, 2)) ||
      failed(verifyWatchers(emitError, watchers, {}, 6)) ||
      failed(verifyBitmap(emitError, virtualChannelSubscription, 8,
                          "virtual channel subscription")) ||
      failed(verifyBitmap(emitError, targets, 17, "ring target bitmap")))
    return failure();
  return success();
}

LogicalResult MeshAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError, MeshDirection direction,
    TraversalAttr read, TraversalAttr write, bool,
    ArrayRef<DmaWatcherAttr> readWatchers,
    ArrayRef<DmaWatcherAttr> writeWatchers, DenseI8ArrayAttr immediateValue,
    unsigned validBytes, bool, MeshReduction reduction, bool) {
  if (!symbolizeMeshDirection(static_cast<uint32_t>(direction)))
    return emitError() << "invalid mesh DMA direction";
  if (!symbolizeMeshReduction(static_cast<uint32_t>(reduction)))
    return emitError() << "invalid mesh DMA reduction";
  if (!immediateValue || immediateValue.size() != 16)
    return emitError() << "mesh immediate value requires sixteen bytes";
  if (validBytes > 16)
    return emitError() << "mesh valid byte count exceeds sixteen bytes";
  if (failed(verifyTraversalShape(emitError, read, 5, 1)) ||
      failed(verifyTraversalShape(emitError, write, 5, 1)) ||
      failed(verifyWatchers(emitError, readWatchers, writeWatchers, 12)))
    return failure();
  return success();
}

LogicalResult WideToNarrowAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError, TraversalAttr read,
    TraversalAttr write, bool, ArrayRef<DmaWatcherAttr> readWatchers,
    ArrayRef<DmaWatcherAttr> writeWatchers, bool, bool,
    unsigned wideMemoryLoadStoreLoopId, bool, unsigned zInBundleValidCount,
    DenseBoolArrayAttr threadMulticastBitmap) {
  if (wideMemoryLoadStoreLoopId >= 8)
    return emitError() << "wide memory load store loop requires three bits";
  if (!llvm::isPowerOf2_32(zInBundleValidCount) || zInBundleValidCount > 8)
    return emitError() << "valid bundle count must be one, two, four or eight";
  if (failed(verifyTraversalShape(emitError, read, 4, 0)) ||
      failed(verifyTraversalShape(emitError, write, 6, 1)) ||
      failed(verifyWatchers(emitError, readWatchers, writeWatchers, 6)) ||
      failed(verifyBitmap(emitError, threadMulticastBitmap, 4,
                          "thread multicast bitmap")))
    return failure();
  return success();
}

LogicalResult WideByteAddressModeAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError,
    unsigned strideUnitGranulesLoopMap, unsigned defaultStrideUnitGranules,
    unsigned lastStrideUnitGranules, unsigned cellStride,
    unsigned cellStrideGroupCountLoopMap, unsigned defaultCellStrideGroupCount,
    unsigned lastCellStrideGroupCount, uint32_t) {
  if (strideUnitGranulesLoopMap >= 16 || cellStrideGroupCountLoopMap >= 16)
    return emitError() << "wide DMA loop maps require four bits";
  if (defaultStrideUnitGranules == 0 || defaultStrideUnitGranules > 128 ||
      lastStrideUnitGranules == 0 || lastStrideUnitGranules > 128)
    return emitError()
           << "wide DMA stride granules must be between one and 128";
  if (cellStride == 0 || cellStride > 32 || defaultCellStrideGroupCount == 0 ||
      defaultCellStrideGroupCount > 32 || lastCellStrideGroupCount == 0 ||
      lastCellStrideGroupCount > 32)
    return emitError() << "wide DMA cell stride and group counts must be "
                          "between one and 32";
  return success();
}

LogicalResult NarrowToWideAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError, TraversalAttr read,
    TraversalAttr write, bool, ArrayRef<DmaWatcherAttr> readWatchers,
    ArrayRef<DmaWatcherAttr> writeWatchers, WideByteAddressModeAttr byteAddress,
    DenseBoolArrayAttr threadMulticastBitmap) {
  if (!byteAddress)
    return emitError() << "wide DMA byte address mode cannot be null";
  if (failed(WideByteAddressModeAttr::verify(
          emitError, byteAddress.getStrideUnitGranulesLoopMap(),
          byteAddress.getDefaultStrideUnitGranules(),
          byteAddress.getLastStrideUnitGranules(), byteAddress.getCellStride(),
          byteAddress.getCellStrideGroupCountLoopMap(),
          byteAddress.getDefaultCellStrideGroupCount(),
          byteAddress.getLastCellStrideGroupCount(),
          byteAddress.getImmediateValue())) ||
      failed(verifyTraversalShape(emitError, read, 6, 0)) ||
      failed(verifyTraversalShape(emitError, write, 4, 1)) ||
      failed(verifyWatchers(emitError, readWatchers, writeWatchers, 6)) ||
      failed(verifyBitmap(emitError, threadMulticastBitmap, 4,
                          "thread multicast bitmap")))
    return failure();
  return success();
}

static LogicalResult
verifyDmaHeader(Operation *operation, TileHeaderAttr header,
                OverwriteInfoAttr overwrite,
                DenseBoolArrayAttr registerSourcedOperands) {
  auto emitError = [&]() { return operation->emitOpError(); };
  if (!header || !overwrite)
    return emitError() << "DMA header and overwrite configuration are required";
  if (failed(TileHeaderAttr::verify(emitError, header.getHeader(),
                                    header.getMulticastBitmap(),
                                    header.getTag())) ||
      failed(
          OverwriteInfoAttr::verify(emitError, overwrite.getBaseAddressSource(),
                                    overwrite.getMulticastBitmapSource(),
                                    overwrite.getOverwriteMulticastBitmap())))
    return failure();
  return verifyBitmap(emitError, registerSourcedOperands, 8,
                      "register sourced operand bitmap");
}

LogicalResult RingConsumerOp::verify() {
  if (failed(verifyDmaHeader(*this, getHeader(), getOverwrite(),
                             getRegisterSourcedOperandsAttr())))
    return failure();
  RingConsumerAttr config = getConfig();
  return RingConsumerAttr::verify(
      [&]() { return emitOpError(); }, config.getConsumer(),
      config.getTraversal(), config.getBaseAddressOverride(),
      config.getWatchers(), config.getVirtualChannelSubscription(),
      config.getDestination(), config.getFilter(),
      config.getThreadMulticastBitmap());
}

LogicalResult RingProducerOp::verify() {
  if (failed(verifyDmaHeader(*this, getHeader(), getOverwrite(),
                             getRegisterSourcedOperandsAttr())))
    return failure();
  RingProducerAttr config = getConfig();
  return RingProducerAttr::verify(
      [&]() { return emitOpError(); }, config.getTraversal(),
      config.getBaseAddressOverride(), config.getWatchers(),
      config.getVirtualChannelSubscription(), config.getTargets(),
      config.getConsumer());
}

LogicalResult MeshOp::verify() {
  if (failed(verifyDmaHeader(*this, getHeader(), getOverwrite(),
                             getRegisterSourcedOperandsAttr())))
    return failure();
  MeshAttr config = getConfig();
  return MeshAttr::verify([&]() { return emitOpError(); },
                          config.getDirection(), config.getRead(),
                          config.getWrite(), config.getBaseAddressOverride(),
                          config.getReadWatchers(), config.getWriteWatchers(),
                          config.getImmediateValue(), config.getValidBytes(),
                          config.getForwardingMode(), config.getReduction(),
                          config.getDataTypeFloat());
}

LogicalResult WideToNarrowOp::verify() {
  if (failed(verifyDmaHeader(*this, getHeader(), getOverwrite(),
                             getRegisterSourcedOperandsAttr())))
    return failure();
  WideToNarrowAttr config = getConfig();
  return WideToNarrowAttr::verify(
      [&]() { return emitOpError(); }, config.getRead(), config.getWrite(),
      config.getBaseAddressOverride(), config.getReadWatchers(),
      config.getWriteWatchers(), config.getTranspose(),
      config.getDoubleOperandMode(), config.getWideMemoryLoadStoreLoopId(),
      config.getIsScalingFactorBias(), config.getZInBundleValidCount(),
      config.getThreadMulticastBitmap());
}

LogicalResult NarrowToWideOp::verify() {
  if (failed(verifyDmaHeader(*this, getHeader(), getOverwrite(),
                             getRegisterSourcedOperandsAttr())))
    return failure();
  NarrowToWideAttr config = getConfig();
  return NarrowToWideAttr::verify(
      [&]() { return emitOpError(); }, config.getRead(), config.getWrite(),
      config.getBaseAddressOverride(), config.getReadWatchers(),
      config.getWriteWatchers(), config.getByteAddress(),
      config.getThreadMulticastBitmap());
}
