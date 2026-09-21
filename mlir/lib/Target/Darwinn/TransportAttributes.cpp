#include "mlir/Target/Darwinn/TransportAttributes.h"
#include "mlir/Target/Darwinn/EncodingAttributes.h"
#include "mlir/Target/Darwinn/TraversalAttributes.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::darwinn;

namespace {

llvm::Expected<std::optional<ScalarSyncWatcher>>
convertWatcher(isa::ScalarSyncWatcherAttr attribute) {
  if (!attribute)
    return std::nullopt;
  if (attribute.getLoopDepth() > 5 ||
      attribute.getInitialValue() >= (1u << 25) ||
      attribute.getStride() >= (1u << 25) ||
      static_cast<uint32_t>(attribute.getFlag()) > 16)
    return llvm::createStringError("invalid G5 scalar sync watcher");
  return ScalarSyncWatcher{static_cast<ScalarSyncFlag>(attribute.getFlag()),
                           static_cast<uint8_t>(attribute.getLoopDepth()),
                           attribute.getInitialValue(), attribute.getStride(),
                           attribute.getValid()};
}

llvm::Expected<std::array<FieldSource, 10>>
convertSources(isa::FieldSourcesAttr attribute) {
  auto values = attribute.getValues();
  if (values.size() != 10)
    return llvm::createStringError(
        "scalar traversal requires 10 field sources");

  std::array<FieldSource, 10> sources;
  for (auto [index, value] : llvm::enumerate(values)) {
    if (!value || static_cast<uint32_t>(value.getValue()) > 1)
      return llvm::createStringError("invalid scalar traversal field source");
    sources[index] = static_cast<FieldSource>(value.getValue());
  }
  return sources;
}

template <typename Op, typename Payload>
llvm::Expected<TaggedPacket>
convertPacket(Op operation, Payload payload,
              llvm::Expected<InstructionBytes> (*encode)(Header, uint32_t,
                                                         const Payload &)) {
  if (operation.getTag() >= (1u << 20))
    return llvm::createStringError(
        "transport tag must fit in 20 unsigned bits");

  auto header = convertHeader(operation.getHeader());
  if (!header)
    return header.takeError();
  auto traversal = convertTraversal(operation.getTraversal());
  if (!traversal)
    return traversal.takeError();

  payload.traversal = std::move(*traversal);
  uint32_t tag = static_cast<uint32_t>(operation.getTag());
  auto encoded = encode(*header, tag, payload);
  if (!encoded)
    return encoded.takeError();
  return TaggedPacket{*header, tag, std::move(payload)};
}

}

llvm::Expected<TaggedPacket>
mlir::darwinn::convertTransport(isa::HibDmaOp operation) {
  if (static_cast<uint32_t>(operation.getQueue()) > 3)
    return llvm::createStringError("G5 host DMA queue is unavailable");

  HibDma payload;
  payload.queue = static_cast<DmaQueue>(operation.getQueue());
  return convertPacket(operation, std::move(payload), encodeHibDma);
}

llvm::Expected<TaggedPacket>
mlir::darwinn::convertTransport(isa::PopInputOp operation) {
  if (static_cast<uint32_t>(operation.getFifo()) > 1)
    return llvm::createStringError("G5 pop input FIFO is unavailable");

  PopInput payload;
  payload.fifo = static_cast<InputFifo>(operation.getFifo());
  auto watcher = convertWatcher(operation.getWatcherAttr());
  if (!watcher)
    return watcher.takeError();
  payload.watcher = *watcher;
  return convertPacket(operation, std::move(payload), encodePopInput);
}

llvm::Expected<TaggedPacket>
mlir::darwinn::convertTransport(isa::RingInfeedOp operation) {
  if (static_cast<uint32_t>(operation.getFifo()) > 1)
    return llvm::createStringError("G5 ring infeed FIFO is unavailable");
  if (static_cast<uint32_t>(operation.getBitmapSource()) > 1 ||
      static_cast<uint32_t>(operation.getBaseAddressSource()) > 1)
    return llvm::createStringError("invalid ring infeed field source");
  if (operation.getSyncIncrement() >= (1u << 25))
    return llvm::createStringError(
        "sync increment must fit in 25 unsigned bits");
  if (operation.getVirtualChannels().size() != 8 ||
      operation.getTargets().size() != 17)
    return llvm::createStringError(
        "ring infeed requires 8 channel and 17 target bits");

  RingInfeed payload;
  payload.fifo = static_cast<InputFifo>(operation.getFifo());
  auto first = convertWatcher(operation.getWatcher0Attr());
  if (!first)
    return first.takeError();
  auto second = convertWatcher(operation.getWatcher1Attr());
  if (!second)
    return second.takeError();
  payload.watchers = {*first, *second};
  payload.syncIncrement = static_cast<uint32_t>(operation.getSyncIncrement());
  llvm::copy(operation.getVirtualChannels(), payload.virtualChannels.begin());
  llvm::copy(operation.getTargets(), payload.targets.begin());
  payload.bitmapSource = static_cast<FieldSource>(operation.getBitmapSource());
  payload.baseAddressSource =
      static_cast<FieldSource>(operation.getBaseAddressSource());

  auto sources = convertSources(operation.getCounterSources());
  if (!sources)
    return sources.takeError();
  payload.counterSources = *sources;

  auto conversion = operation.getConversion();
  if (static_cast<uint32_t>(conversion.getMode()) > 1)
    return llvm::createStringError("infeed conversion is unavailable on G5");

  switch (conversion.getMode()) {
  case isa::InfeedConversion::None:
    if (conversion.getZeroPoint() != 0)
      return llvm::createStringError(
          "disabled infeed conversion has a nonzero zero point");
    break;
  case isa::InfeedConversion::Unsigned8ToBfloat:
    if (conversion.getZeroPoint() > 255)
      return llvm::createStringError(
          "infeed conversion zero point must fit in 8 bits");
    payload.unsigned8ToBfloatZeroPoint =
        static_cast<uint8_t>(conversion.getZeroPoint());
    break;
  }
  return convertPacket(operation, std::move(payload), encodeRingInfeed);
}

llvm::Expected<TaggedPacket>
mlir::darwinn::convertTransport(isa::RingOutfeedOp operation) {
  if (static_cast<uint32_t>(operation.getBaseAddressSource()) > 1)
    return llvm::createStringError("invalid ring outfeed field source");
  if (operation.getVirtualChannels().size() != 8)
    return llvm::createStringError(
        "ring outfeed requires 8 virtual channel bits");
  if (operation.getBytesToPop() == 0 || operation.getBytesToPop() > UINT32_MAX)
    return llvm::createStringError(
        "bytes to pop must be between 1 and 4294967295");

  RingOutfeed payload;
  auto watcher = convertWatcher(operation.getWatcherAttr());
  if (!watcher)
    return watcher.takeError();
  payload.watcher = *watcher;
  payload.baseAddressSource =
      static_cast<FieldSource>(operation.getBaseAddressSource());
  auto sources = convertSources(operation.getCounterSources());
  if (!sources)
    return sources.takeError();
  payload.counterSources = *sources;
  llvm::copy(operation.getVirtualChannels(), payload.virtualChannels.begin());
  payload.bytesToPop = static_cast<uint32_t>(operation.getBytesToPop());
  return convertPacket(operation, std::move(payload), encodeRingOutfeed);
}
