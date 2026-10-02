#include "Families.h"

using namespace mlir;
using namespace mlir::darwinn;
using namespace mlir::darwinn::codegen;

namespace {

constexpr std::array<bool, 16> kAllTiles = {true, true, true, true, true, true,
                                            true, true, true, true, true, true,
                                            true, true, true, true};

SmallVector<Emitted, 0> subscribeTiles(const std::array<bool, 16> &tiles,
                                       const std::array<bool, 8> &channels,
                                       bool tileOnly = false,
                                       bool scalarChannel7 = false) {
  SmallVector<Emitted, 0> pair = subscribeVc(
      channels, scalarChannel7 ? bits<8>("00000001") : bits<8>("00000000"));
  std::get<TilePacket>(pair[0].instruction).header.multicastBitmap = tiles;
  auto &scalar = std::get<ScalarFence>(
      std::get<TaggedPacket>(pair[1].instruction).instruction);
  std::array<bool, 20> valid{};
  llvm::copy(tiles, valid.begin());
  scalar.resetTileFence = valid;
  scalar.tileFenceValid = valid;
  if (tileOnly)
    pair.pop_back();
  return pair;
}

} // namespace

namespace mlir::darwinn::codegen {

class Channels {
public:
  SmallVector<Emitted, 0> use(const std::array<bool, 16> &wanted, bool scatter,
                              bool outfeed) {
    SmallVector<Emitted, 0> out;
    if (!current) {
      out = subscribeTiles(wanted, bits<8>("01000001"), false, outfeed);
    } else if (*current == wanted && wanted == kAllTiles && !dirty &&
               !scatter) {
    } else {
      out = subscribeTiles(*current, bits<8>("01000000"));
      llvm::append_range(
          out, subscribeTiles(wanted, bits<8>("01000001"), false, outfeed));
    }
    if (!out.empty())
      scalar = outfeed;
    current = wanted;
    dirty = wanted != kAllTiles || scatter;
    return out;
  }

  SmallVector<Emitted, 0> preempt() const {
    std::array<bool, 16> tiles = current.value_or(kAllTiles);
    SmallVector<Emitted, 0> out = subscribeTiles(
        tiles, current ? bits<8>("01000001") : bits<8>("01000000"), true);
    if (tiles != kAllTiles) {
      std::array<bool, 16> complement;
      for (size_t tile = 0; tile < tiles.size(); ++tile)
        complement[tile] = !tiles[tile];
      llvm::append_range(out,
                         subscribeTiles(complement, bits<8>("01000000"), true));
    }
    out.push_back(
        subscribeVc(bits<8>("01000001"),
                    scalar ? bits<8>("00000001") : bits<8>("00000000"))[1]);
    return out;
  }

private:
  std::optional<std::array<bool, 16>> current;
  bool dirty = false;
  bool scalar = false;
};

} // namespace mlir::darwinn::codegen

namespace {

SmallVector<Emitted, 0> preemption(const Channels &channels) {
  ScalarFence interrupt;
  interrupt.resetTileFence = kTileFenceValid;
  interrupt.expectedTileFenceCount = 1;
  interrupt.tileFenceValid = kTileFenceValid;
  interrupt.waitIdle = bits<5>("11111");
  interrupt.sendInterrupt = true;
  ScalarFence resetCounters;
  resetCounters.resetTileFence = kTileFenceValid;
  resetCounters.waitIdle = bits<5>("11111");
  ScalarFence globalReset;
  globalReset.resetTileFence = kTileFenceValid;
  globalReset.resetSyncFlag = bits<17>("11111111111111111");
  globalReset.waitIdle = bits<5>("11111");
  TileFence tileGlobalReset;
  tileGlobalReset.resetSyncFlag = bits<22>("0000000000000000011000");
  tileGlobalReset.resetSyncFlagThreadIds = bits<4>("1111");
  tileGlobalReset.waitIdle = bits<13>("1111111111111");
  tileGlobalReset.waitIdleThreadIds = bits<4>("1111");
  tileGlobalReset.increment = true;
  ScalarFence waits;
  waits.resetTileFence = kTileFenceValid;
  waits.expectedTileFenceCount = 1;
  waits.tileFenceValid = kTileFenceValid;
  waits.waitIdle = bits<5>("11111");
  TileFence fence;
  fence.resetSyncFlag = bits<22>("1111111111111111100111");
  fence.resetSyncFlagThreadIds = bits<4>("1111");
  fence.waitIdle = bits<13>("1111111111111");
  fence.waitIdleThreadIds = bits<4>("1111");
  fence.increment = true;
  SmallVector<Emitted, 0> out{localReset(), tileFence(fence),
                              scalarFence(interrupt, Role::PreemptionInterrupt),
                              scalarFence(resetCounters)};
  llvm::append_range(out, channels.preempt());
  out.push_back(localReset());
  out.push_back(scalarFence(globalReset));
  out.push_back(tileFence(tileGlobalReset));
  out.push_back(scalarFence(waits));
  return out;
}

SmallVector<Emitted, 0> epilogue() {
  SmallVector<Emitted, 0> out =
      subscribeVc(bits<8>("00000000"), bits<8>("00000000"));
  TileFence sync;
  sync.resetSyncFlagThreadIds = bits<4>("1111");
  sync.waitIdle = bits<13>("1111111111111");
  sync.waitIdleThreadIds = bits<4>("1111");
  sync.increment = true;
  ScalarFence wait;
  wait.resetTileFence = kTileFenceValid;
  wait.expectedTileFenceCount = 1;
  wait.tileFenceValid = kTileFenceValid;
  wait.waitIdle = bits<5>("00110");
  ScalarFence finish;
  finish.resetTileFence = bits<20>("11111111111111111111");
  finish.tileFenceValid = bits<20>("11111111111111111111");
  finish.waitIdle = bits<5>("11111");
  finish.sendInterrupt = true;
  out.push_back(tileFence(sync));
  out.push_back(scalarFence(wait));
  out.push_back(scalarFence(finish, Role::FinalInterrupt));
  return out;
}

struct ChannelUse {
  std::array<bool, 16> tiles;
  bool scatter;
  bool outfeed;
};

std::array<bool, 16> tileSetOf(ArrayRef<TileBox> boxes) {
  std::array<bool, 16> out{};
  for (const TileBox &box : boxes)
    out[box.tile] = true;
  return out;
}

std::optional<ChannelUse> channel7Tiles(const Group &group) {
  Operation *op = group.op;
  if (group.kind == GroupKind::Op && isa<TensorOpOp>(op)) {
    std::optional<InnerOperationKind> inner = innerOperation(op);
    std::array<bool, 16> active = activeTiles(op);
    if ((inner == InnerOperationKind::Vmc ||
         inner == InnerOperationKind::Stencil) &&
        active != kAllTiles)
      return ChannelUse{active, false, false};
    return std::nullopt;
  }
  if (!isa<RedistributeOp>(op))
    return std::nullopt;
  if (group.kind == GroupKind::Scatter)
    return ChannelUse{kAllTiles, true, false};
  if (group.kind != GroupKind::Op)
    return std::nullopt;
  DistributedMemorySpace from = sourceSpace(op), to = resultSpace(op);
  if (from == DistributedMemorySpace::TileMemory &&
      to == DistributedMemorySpace::HostMemory && !unused(op))
    return ChannelUse{tileSetOf(tiles(*slicingOf(producer(op, 0)))), false,
                      true};
  if (from == DistributedMemorySpace::HostMemory) {
    SmallVector<TileBox> boxes = tiles(*slicingOf(op));
    if (boxes.size() < kTiles)
      return ChannelUse{tileSetOf(boxes), false, false};
  }
  return std::nullopt;
}

Body groupBody(const Group &group, Context &context) {
  Operation *op = group.op;
  switch (group.kind) {
  case GroupKind::Preempt:
    return preemption(*context.channels);
  case GroupKind::Relayout:
    return SmallVector<Emitted, 0>{};
  case GroupKind::Scatter:
    return scatter(op, context);
  case GroupKind::Gather:
    context.gathered.insert(op);
    return gatherRows(op, context);
  case GroupKind::RingReshapeIdentity:
    return ringReshape(op, context);
  case GroupKind::Permute:
    return permuteCopy(op, context);
  case GroupKind::Padding:
    return padding(op, context);
  case GroupKind::Init:
    return initialization(op, context);
  case GroupKind::Empty:
    return SmallVector<Emitted, 0>{};
  case GroupKind::Op:
    break;
  }
  if (auto redistribute = dyn_cast<RedistributeOp>(op)) {
    DistributedMemorySpace from = sourceSpace(op), to = resultSpace(op);
    if (from == DistributedMemorySpace::HostMemory &&
        to == DistributedMemorySpace::TileMemory)
      return transferLoad(op, context);
    if (from == DistributedMemorySpace::TileMemory &&
        to == DistributedMemorySpace::HostMemory)
      return isModelOutput(op) ? modelOutput(op, context)
                               : transferStore(op, context);
    if (context.gathered.contains(op))
      return gatherColumns(op, context);
  }
  if (isa<FillOp>(op))
    return fill(op, context);
  if (isa<InterpolateHardwareOp>(op))
    return interpolate(op, context);
  if (isa<UnaryTensorOpOp>(op))
    return unary(op, context);
  if (isa<TensorOpOp>(op)) {
    switch (*innerOperation(op)) {
    case InnerOperationKind::Elementwise:
      return elementwise(op, context);
    case InnerOperationKind::Vmc:
      return vmc(op, context);
    case InnerOperationKind::Stencil:
      return stencil(op, context);
    default:
      break;
    }
  }
  return unsupported(op, "this tensor group");
}

uint32_t &tagOf(Instruction &instruction) {
  return std::visit(
      llvm::makeVisitor(
          [](ScalarPacket &) -> uint32_t & {
            llvm_unreachable("codegen emits no untagged scalar packets");
          },
          [](TaggedPacket &packet) -> uint32_t & { return packet.tag; },
          [](TilePacket &packet) -> uint32_t & { return packet.header.tag; },
          [](ComputePacket &packet) -> uint32_t & { return packet.header.tag; },
          [](DmaInstruction &dma) -> uint32_t & { return dma.header.tag; }),
      instruction);
}

bool sendsInterrupt(const Instruction &instruction) {
  auto *tagged = std::get_if<TaggedPacket>(&instruction);
  if (!tagged)
    return false;
  auto *fence = std::get_if<ScalarFence>(&tagged->instruction);
  return fence && fence->sendInterrupt;
}

void assignTags(MutableArrayRef<Segment> segments) {
  uint32_t ordinary = 0, interrupts = 0;
  for (Segment &segment : segments)
    for (Emitted &emitted : segment.instructions)
      tagOf(emitted.instruction) = sendsInterrupt(emitted.instruction)
                                       ? (uint32_t(1) << 18) + interrupts++
                                       : ordinary++;
}

} // namespace

FailureOr<SmallVector<Segment>> codegen::generate(ArrayRef<Group> groups,
                                                  Context &context) {
  Channels channels;
  context.channels = &channels;
  context.gathered.clear();
  SmallVector<Segment> segments;
  segments.push_back(Segment{
      nullptr, 0, subscribeVc(bits<8>("01000000"), bits<8>("00000000"))});
  for (const Group &group : groups) {
    if (group.kind == GroupKind::Empty)
      continue;
    if (group.kind == GroupKind::Relayout)
      continue;
    SmallVector<Emitted, 0> header = resetAllSyncFlags();
    if (segments.size() == 1)
      llvm::append_range(header, resetAllSyncFlags());
    header.push_back(localReset());
    if (auto use = channel7Tiles(group))
      llvm::append_range(header,
                         channels.use(use->tiles, use->scatter, use->outfeed));
    Body body = groupBody(group, context);
    if (failed(body))
      return failure();
    Segment segment{&group, header.size(), std::move(header)};
    llvm::append_range(segment.instructions, *body);
    segments.push_back(std::move(segment));
  }
  llvm::append_range(segments.back().instructions, epilogue());
  assignTags(segments);
  context.channels = nullptr;
  return segments;
}
