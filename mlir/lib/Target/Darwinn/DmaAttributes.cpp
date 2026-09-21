#include "mlir/Target/Darwinn/DmaAttributes.h"
#include "mlir/Dialect/Darwinn/IR/InstructionOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Target/Darwinn/EncodingAttributes.h"
#include "mlir/Target/Darwinn/TraversalAttributes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include <utility>

using namespace mlir;
using namespace mlir::darwinn;

template <size_t Size>
static llvm::Expected<std::array<bool, Size>>
convertBitmap(DenseBoolArrayAttr attribute) {
  if (!attribute || static_cast<size_t>(attribute.size()) != Size)
    return llvm::createStringError("DMA bitmap has an invalid size");
  std::array<bool, Size> result;
  llvm::copy(attribute.asArrayRef(), result.begin());
  return result;
}

static llvm::Expected<llvm::SmallVector<DmaWatcher, 6>>
convertWatchers(ArrayRef<isa::DmaWatcherAttr> attributes) {
  llvm::SmallVector<DmaWatcher, 6> result;
  result.reserve(attributes.size());

  for (isa::DmaWatcherAttr attribute : attributes) {
    if (!attribute)
      return llvm::createStringError("Missing DMA watcher attribute");
    auto watcher = convertSyncWatcher(attribute.getWatcher());
    if (!watcher)
      return watcher.takeError();
    result.push_back({*watcher, attribute.getStallTtuId()});
  }

  return result;
}

template <typename AttributeT, typename TransferT>
static llvm::Error convertTransfer(AttributeT attribute, TransferT &result) {
  auto read = convertTraversal(attribute.getRead());
  if (!read)
    return read.takeError();
  auto write = convertTraversal(attribute.getWrite());
  if (!write)
    return write.takeError();
  auto readWatchers = convertWatchers(attribute.getReadWatchers());
  if (!readWatchers)
    return readWatchers.takeError();
  auto writeWatchers = convertWatchers(attribute.getWriteWatchers());
  if (!writeWatchers)
    return writeWatchers.takeError();

  result.read = std::move(*read);
  result.write = std::move(*write);
  result.baseAddressOverride = attribute.getBaseAddressOverride();
  result.readWatchers = std::move(*readWatchers);
  result.writeWatchers = std::move(*writeWatchers);
  return llvm::Error::success();
}

static llvm::Expected<DmaOperation>
convertConfig(isa::RingConsumerAttr attribute) {
  auto traversal = convertTraversal(attribute.getTraversal());
  if (!traversal)
    return traversal.takeError();
  auto watchers = convertWatchers(attribute.getWatchers());
  if (!watchers)
    return watchers.takeError();
  auto subscription =
      convertBitmap<8>(attribute.getVirtualChannelSubscription());
  if (!subscription)
    return subscription.takeError();
  auto threads = convertBitmap<4>(attribute.getThreadMulticastBitmap());
  if (!threads)
    return threads.takeError();

  isa::ByteFilterAttr filter = attribute.getFilter();
  RingConsumer result;
  result.consumer = static_cast<RingConsumerId>(attribute.getConsumer());
  result.traversal = std::move(*traversal);
  result.baseAddressOverride = attribute.getBaseAddressOverride();
  result.watchers = std::move(*watchers);
  result.virtualChannelSubscription = *subscription;
  result.destination = static_cast<RingDestination>(attribute.getDestination());
  result.filter = {static_cast<uint8_t>(filter.getFirstDiscardByteLoopMap()),
                   static_cast<uint8_t>(filter.getDiscardByteLoopMap()),
                   filter.getFirstDiscardByteCount(),
                   filter.getDiscardByteCount(),
                   filter.getLastDiscardByteCount()};
  result.threadMulticastBitmap = *threads;
  return DmaOperation(std::move(result));
}

static llvm::Expected<DmaOperation>
convertConfig(isa::RingProducerAttr attribute) {
  auto traversal = convertTraversal(attribute.getTraversal());
  if (!traversal)
    return traversal.takeError();
  auto watchers = convertWatchers(attribute.getWatchers());
  if (!watchers)
    return watchers.takeError();
  auto subscription =
      convertBitmap<8>(attribute.getVirtualChannelSubscription());
  if (!subscription)
    return subscription.takeError();
  auto targets = convertBitmap<17>(attribute.getTargets());
  if (!targets)
    return targets.takeError();

  RingProducer result;
  result.traversal = std::move(*traversal);
  result.baseAddressOverride = attribute.getBaseAddressOverride();
  result.watchers = std::move(*watchers);
  result.virtualChannelSubscription = *subscription;
  result.targets = *targets;
  result.consumer = static_cast<RingConsumerId>(attribute.getConsumer());
  return DmaOperation(std::move(result));
}

static llvm::Expected<DmaOperation> convertConfig(isa::MeshAttr attribute) {
  Mesh result;
  if (auto error = convertTransfer(attribute, result))
    return error;

  result.direction = static_cast<MeshDirection>(attribute.getDirection());
  llvm::copy(attribute.getImmediateValue().asArrayRef(),
             result.immediateValue.begin());
  result.validBytes = attribute.getValidBytes();
  result.forwardingMode = attribute.getForwardingMode();
  result.reduction = static_cast<MeshReduction>(attribute.getReduction());
  result.dataTypeFloat = attribute.getDataTypeFloat();
  return DmaOperation(std::move(result));
}

static llvm::Expected<DmaOperation>
convertConfig(isa::WideToNarrowAttr attribute) {
  WideToNarrow result;
  if (auto error = convertTransfer(attribute, result))
    return error;
  auto threads = convertBitmap<4>(attribute.getThreadMulticastBitmap());
  if (!threads)
    return threads.takeError();

  result.transpose = attribute.getTranspose();
  result.doubleOperandMode = attribute.getDoubleOperandMode();
  result.wideMemoryLoadStoreLoopId = attribute.getWideMemoryLoadStoreLoopId();
  result.isScalingFactorBias = attribute.getIsScalingFactorBias();
  result.zInBundleValidCount = attribute.getZInBundleValidCount();
  result.threadMulticastBitmap = *threads;
  return DmaOperation(std::move(result));
}

static llvm::Expected<DmaOperation>
convertConfig(isa::NarrowToWideAttr attribute) {
  NarrowToWide result;
  if (auto error = convertTransfer(attribute, result))
    return error;
  auto threads = convertBitmap<4>(attribute.getThreadMulticastBitmap());
  if (!threads)
    return threads.takeError();

  isa::WideByteAddressModeAttr address = attribute.getByteAddress();
  result.byteAddress = {
      static_cast<uint8_t>(address.getStrideUnitGranulesLoopMap()),
      static_cast<uint8_t>(address.getDefaultStrideUnitGranules()),
      static_cast<uint8_t>(address.getLastStrideUnitGranules()),
      static_cast<uint8_t>(address.getCellStride()),
      static_cast<uint8_t>(address.getCellStrideGroupCountLoopMap()),
      static_cast<uint8_t>(address.getDefaultCellStrideGroupCount()),
      static_cast<uint8_t>(address.getLastCellStrideGroupCount()),
      address.getImmediateValue()};
  result.threadMulticastBitmap = *threads;
  return DmaOperation(std::move(result));
}

llvm::Expected<DmaInstruction>
mlir::darwinn::convertDmaInstruction(Operation *operation) {
  if (!operation ||
      !llvm::isa<isa::RingConsumerOp, isa::RingProducerOp, isa::MeshOp,
                 isa::WideToNarrowOp, isa::NarrowToWideOp>(operation))
    return llvm::createStringError("Expected a DMA instruction operation");

  std::string diagnostic;
  ScopedDiagnosticHandler handler(operation->getContext(),
                                  [&](Diagnostic &value) {
                                    diagnostic = value.str();
                                    return success();
                                  });

  if (failed(verify(operation, false)))
    return llvm::createStringError(diagnostic);

  return llvm::TypeSwitch<Operation *, llvm::Expected<DmaInstruction>>(
             operation)
      .Case<isa::RingConsumerOp, isa::RingProducerOp, isa::MeshOp,
            isa::WideToNarrowOp, isa::NarrowToWideOp>(
          [](auto value) -> llvm::Expected<DmaInstruction> {
            auto header = convertTileHeader(value.getHeader());
            if (!header)
              return header.takeError();
            auto overwrite = convertOverwrite(value.getOverwrite());
            if (!overwrite)
              return overwrite.takeError();
            auto registers =
                convertBitmap<8>(value.getRegisterSourcedOperandsAttr());
            if (!registers)
              return registers.takeError();
            auto config = convertConfig(value.getConfig());
            if (!config)
              return config.takeError();
            return DmaInstruction{*header, *overwrite, *registers,
                                  std::move(*config)};
          })
      .Default([](Operation *) -> llvm::Expected<DmaInstruction> {
        return llvm::createStringError("Expected a DMA instruction operation");
      });
}
