#include "Codegen.h"

using namespace mlir;
using namespace mlir::darwinn;
using namespace mlir::darwinn::codegen;

std::array<bool, 16> codegen::tileBit(int64_t tile) {
  std::array<bool, 16> out{};
  out[tile] = true;
  return out;
}

std::array<bool, 16> codegen::tileSet(ArrayRef<int64_t> tiles) {
  std::array<bool, 16> out{};
  for (int64_t tile : tiles)
    out[tile] = true;
  return out;
}

std::array<bool, 4> codegen::threadBit(int64_t thread) {
  std::array<bool, 4> out{};
  out[thread] = true;
  return out;
}

std::array<bool, 17> codegen::targets(const std::array<bool, 16> &tiles,
                                      bool host) {
  std::array<bool, 17> out{};
  llvm::copy(tiles, out.begin());
  out[16] = host;
  return out;
}

Counter codegen::counter(int64_t end, int64_t step, bool mask) {
  return Counter{end, step, mask};
}

SmallVector<Counter, 8> codegen::padded(ArrayRef<Counter> counters,
                                        size_t length) {
  SmallVector<Counter, 8> out(counters);
  while (out.size() < length)
    out.push_back(Counter{0, 1, false});
  return out;
}

ByteAddressMode codegen::access(int64_t bytes, uint32_t loopMap) {
  return ByteAddressMode{loopMap, static_cast<uint64_t>(bytes),
                         static_cast<uint64_t>(bytes)};
}

ByteAddressMode codegen::access(int64_t last, int64_t defaultBytes,
                                uint32_t loopMap) {
  return ByteAddressMode{loopMap, static_cast<uint64_t>(last),
                         static_cast<uint64_t>(defaultBytes)};
}

SyncProducer codegen::producerSync(bool increment, uint8_t depth) {
  return SyncProducer{increment, depth};
}

SyncWatcher codegen::tileWatcher(TileSyncFlag flag, int32_t initial,
                                 uint32_t stride, uint8_t depth, bool bykj,
                                 uint8_t thread) {
  SyncWatcher watcher;
  watcher.syncFlag = flag;
  watcher.threadId = thread;
  watcher.bykjWtLmxpJcgmyqKq = bykj;
  watcher.syncFlagLoopDepth = depth;
  watcher.initialExpectedSyncFlagValue = initial;
  watcher.waitSyncFlagStride = stride;
  watcher.syncWaitValid = true;
  return watcher;
}

DmaWatcher codegen::dmaWatcher(SyncWatcher watcher, bool stall) {
  return DmaWatcher{watcher, stall};
}

TileHeader codegen::tileHeader(const std::array<bool, 16> &multicast) {
  TileHeader header;
  header.multicastBitmap = multicast;
  return header;
}

Emitted codegen::tileFence(TileFence fence,
                           const std::array<bool, 16> &multicast) {
  return Emitted{TilePacket{tileHeader(multicast), fence}, Role::Plain};
}

Emitted codegen::scalarFence(ScalarFence fence, Role role) {
  return Emitted{TaggedPacket{Header{}, 0, fence}, role};
}

Emitted codegen::tagged(
    std::variant<ScalarFence, HibDma, PopInput, RingInfeed, RingOutfeed>
        instruction) {
  TaggedPacket packet;
  packet.instruction = std::move(instruction);
  return Emitted{std::move(packet), Role::Plain};
}

Emitted codegen::dma(DmaOperation operation,
                     const std::array<bool, 16> &multicast,
                     const std::array<bool, 8> &registers) {
  DmaInstruction instruction;
  instruction.header = tileHeader(multicast);
  instruction.registerSourcedOperands = registers;
  instruction.operation = std::move(operation);
  return Emitted{std::move(instruction), Role::Plain};
}

Emitted codegen::tensor(TensorOp operation,
                        const std::array<bool, 16> &multicast,
                        const std::array<bool, 8> &registers) {
  ComputePacket packet;
  packet.header = tileHeader(multicast);
  packet.registerSourcedOperands = registers;
  packet.instruction = std::move(operation);
  return Emitted{std::move(packet), Role::Plain};
}

Emitted codegen::load(int64_t value, const std::array<bool, 16> &multicast,
                      uint8_t baseRegister,
                      std::optional<std::array<bool, 4>> threads) {
  TileLoadStore load;
  load.operation = TileMemoryOperation::Load;
  load.registerBurstLength = 1;
  load.baseRegister = baseRegister;
  load.useImmediate = true;
  load.immediateValue = static_cast<uint32_t>(value);
  if (threads)
    load.threadBitmap = *threads;
  return Emitted{TilePacket{tileHeader(multicast), load}, Role::Plain};
}

SmallVector<Emitted, 0> codegen::resetAllSyncFlags() {
  TileFence tile;
  tile.resetSyncFlag = bits<22>("1111111111111111111111");
  tile.resetSyncFlagThreadIds = bits<4>("1111");
  tile.waitIdle = bits<13>("1111111111111");
  tile.waitIdleThreadIds = bits<4>("1111");
  tile.increment = true;
  ScalarFence scalar;
  scalar.resetTileFence = kTileFenceValid;
  scalar.resetSyncFlag = bits<17>("11111111111111111");
  scalar.expectedTileFenceCount = 1;
  scalar.tileFenceValid = kTileFenceValid;
  scalar.waitIdle = bits<5>("11111");
  return {tileFence(tile), scalarFence(scalar)};
}

Emitted codegen::localReset() {
  TileFence tile;
  tile.resetSyncFlag = bits<22>("1111111111111111100111");
  tile.resetSyncFlagThreadIds = bits<4>("1111");
  tile.waitIdle = bits<13>("1111111111111");
  tile.waitIdleThreadIds = bits<4>("1111");
  return tileFence(tile);
}

SmallVector<Emitted, 0>
codegen::subscribeVc(const std::array<bool, 8> &tileChannels,
                     const std::array<bool, 8> &scalarChannels) {
  TileFence tile;
  tile.resetSyncFlagThreadIds = bits<4>("1000");
  tile.waitIdle = bits<13>("0110000000000");
  tile.waitIdleThreadIds = bits<4>("1000");
  tile.increment = true;
  tile.virtualChannelSubscription = tileChannels;
  tile.virtualChannelSubscriptionValid = true;
  ScalarFence scalar;
  scalar.resetTileFence = kTileFenceValid;
  scalar.expectedTileFenceCount = 1;
  scalar.tileFenceValid = kTileFenceValid;
  scalar.waitIdle = bits<5>("00010");
  scalar.virtualChannelSubscription = scalarChannels;
  scalar.virtualChannelSubscriptionValid = true;
  return {tileFence(tile), scalarFence(scalar)};
}

SmallVector<Emitted, 0> codegen::groupFences() {
  TileFence tile;
  tile.resetSyncFlagThreadIds = bits<4>("1111");
  tile.waitIdle = bits<13>("0110001111000");
  tile.waitIdleThreadIds = bits<4>("1111");
  tile.increment = true;
  ScalarFence scalar;
  scalar.resetTileFence = kTileFenceValid;
  scalar.expectedTileFenceCount = 1;
  scalar.tileFenceValid = kTileFenceValid;
  return {tileFence(tile), scalarFence(scalar)};
}
