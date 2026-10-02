#ifndef MLIR_LIB_DIALECT_DARWINN_CODEGEN_CODEGEN_H
#define MLIR_LIB_DIALECT_DARWINN_CODEGEN_CODEGEN_H

#include "mlir/Dialect/Darwinn/IR/DarwinnOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Target/Darwinn/Serialization.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include <array>
#include <map>
#include <optional>
#include <string>

namespace mlir::darwinn::codegen {

using Index = SmallVector<int64_t, 4>;

constexpr int64_t kThreads = 4;
constexpr int64_t kGrid = 4;
constexpr int64_t kTiles = 16;

Index evaluate(AffineMap map, ArrayRef<int64_t> point);

struct Slicing {
  AffineMap begins;
  AffineMap ends;
  SmallVector<int64_t, 3> domain;

  bool hasThreads() const { return domain.size() > 2; }
  Index begin(ArrayRef<int64_t> point) const { return evaluate(begins, point); }
  Index end(ArrayRef<int64_t> point) const { return evaluate(ends, point); }
  bool operator==(const Slicing &other) const {
    return begins == other.begins && ends == other.ends &&
           domain == other.domain;
  }
};

std::optional<Slicing> slicingOf(Operation *op);

struct Box {
  Index lo;
  Index hi;
};

Box unionBox(const Slicing &slicing, int64_t row, int64_t column);
Box threadBox(const Slicing &slicing, int64_t thread, int64_t tile = 0);
Index extent(const Box &box);

struct TileBox {
  int64_t tile;
  Box box;
};

SmallVector<TileBox> tiles(const Slicing &slicing);

struct TensorInfo {
  SmallVector<int64_t, 4> shape;
  int64_t elementBytes = 0;
  DistributedMemorySpace space = DistributedMemorySpace::TileMemory;
};

TensorInfo describe(Type type);
TensorInfo resultInfo(Operation *op);
TensorInfo operandInfo(Operation *op, unsigned index);
Index strides(ArrayRef<int64_t> shape, int64_t elementBytes);
Index tail(ArrayRef<int64_t> values, size_t count);
int64_t product(ArrayRef<int64_t> values);
int64_t dot(ArrayRef<int64_t> left, ArrayRef<int64_t> right);
int64_t ceilDiv(int64_t numerator, int64_t denominator);

Operation *producer(Operation *op, unsigned index);
Operation *fillBehind(Value value);
SmallVector<Operation *> usersOf(Operation *op);
bool unused(Operation *op);
bool isModelOutput(Operation *op);
DistributedMemorySpace sourceSpace(Operation *op);
DistributedMemorySpace resultSpace(Operation *op);

std::optional<InnerOperationKind> innerOperation(Operation *op);
std::optional<LinearFunctionKind> linearFunction(Operation *op);
std::optional<NluFunctionKind> nluFunction(Operation *op);
ComputeOpOptionsAttr computeOptions(Operation *op);
bool hasAuxiliary(Operation *op, AuxTensorKind kind);

enum class GroupKind {
  Preempt,
  Init,
  Op,
  Permute,
  Empty,
  Gather,
  Scatter,
  RingReshapeIdentity,
  Relayout,
  Padding
};

struct Group {
  int64_t step = 0;
  GroupKind kind = GroupKind::Op;
  Operation *op = nullptr;
};

SmallVector<Group> deriveSchedule(func::FuncOp function);

enum class Suffix {
  None,
  Init,
  Dest,
  Gather,
  Permute,
  Bias,
  StageA,
  StageB,
  Relay
};

struct StorageBlock {
  Operation *value = nullptr;
  Suffix suffix = Suffix::None;
  int64_t start = 0;
  int64_t end = 0;
  int64_t size = 0;
  SmallVector<Operation *, 2> writers;
  int64_t offset = 0;
};

struct StorageProblems {
  SmallVector<StorageBlock> narrow;
  SmallVector<StorageBlock> host;
};

FailureOr<StorageProblems> storageProblems(ArrayRef<Group> groups);
FailureOr<SmallVector<StorageBlock>> wideProblem(ArrayRef<Group> groups);
FailureOr<SmallVector<int64_t>> tinyMalloc(ArrayRef<StorageBlock> blocks,
                                           int64_t alignment, int64_t capacity);

enum class HibRoot {
  InputActivation,
  OutputActivation,
  ParameterRegion,
  Scratch,
  Parameter,
  ParameterFill
};

struct Hib {
  DmaQueue queue = DmaQueue::Activation;
  HibRoot root = HibRoot::Scratch;
  int64_t offset = 0;
  int64_t size = 0;
  Operation *source = nullptr;
};

enum class Role {
  Plain,
  OutputActivationInterrupt,
  PreemptionInterrupt,
  FinalInterrupt
};

struct Emitted {
  Instruction instruction;
  Role role = Role::Plain;
};

class Channels;

class Context {
public:
  static FailureOr<Context> create(func::FuncOp function,
                                   ArrayRef<Group> groups);

  FailureOr<int64_t> narrowAddress(Operation *value,
                                   Suffix suffix = Suffix::None) const;
  FailureOr<int64_t> storageAddress(Operation *op) const;
  FailureOr<int64_t> wideAddress(Operation *value,
                                 Suffix suffix = Suffix::None) const;
  FailureOr<int64_t> hostOffset(Operation *writer) const;
  int64_t hib(DmaQueue queue, HibRoot root, int64_t offset, int64_t size = 0,
              Operation *source = nullptr);
  SmallVector<Hib> resolvedHibs() const;
  int64_t tileMemoryBytes() const;
  int64_t scratchBytes() const;

  ArrayRef<StorageBlock> narrow() const { return narrowBlocks; }
  ArrayRef<StorageBlock> host() const { return hostBlocks; }
  ArrayRef<StorageBlock> wide() const { return wideBlocks; }

  Channels *channels = nullptr;
  llvm::DenseSet<Operation *> gathered;
  std::map<NluFunctionKind, CoefficientTables> nluTables;

private:
  SmallVector<StorageBlock> narrowBlocks;
  SmallVector<StorageBlock> hostBlocks;
  SmallVector<StorageBlock> wideBlocks;
  SmallVector<Hib> hibs;
};

struct Segment {
  const Group *group = nullptr;
  size_t header = 0;
  SmallVector<Emitted, 0> instructions;
};

FailureOr<SmallVector<Segment>> generate(ArrayRef<Group> groups,
                                         Context &context);

struct ChunkMeta {
  bool startsWithInputHib = false;
  std::optional<Role> interrupt;
  SmallVector<std::pair<int64_t, InstructionRef>> hibs;
};

struct GeneratedProgram {
  SmallVector<ProgramChunk, 0> chunks;
  SmallVector<ChunkMeta, 0> meta;
  SmallVector<Hib> hibs;
};

FailureOr<GeneratedProgram> buildProgram(ArrayRef<Segment> segments,
                                         const Context &context);
FailureOr<SmallVector<uint8_t, 0>> packParameters(ArrayRef<Hib> hibs);

struct HostEvent {
  enum class Kind { Dispatch, Root, Copy, Add, Put, Wait };
  Kind kind;
  int64_t first = 0;
  int64_t second = 0;
  int64_t third = 0;
};

FailureOr<SmallVector<HostEvent, 0>> hostEvents(const GeneratedProgram &program,
                                                const EncodedProgram &encoded);
OwningOpRef<ModuleOp> hostModule(MLIRContext *context,
                                 ArrayRef<HostEvent> events, int64_t slotCount,
                                 StringRef programSymbol);

template <size_t N>
std::array<bool, N> bits(StringRef pattern) {
  std::array<bool, N> out{};
  for (auto [index, character] : llvm::enumerate(pattern.take_front(N)))
    out[index] = character == '1';
  return out;
}

std::array<bool, 16> tileBit(int64_t tile);
std::array<bool, 16> tileSet(ArrayRef<int64_t> tiles);
std::array<bool, 4> threadBit(int64_t thread);
std::array<bool, 17> targets(const std::array<bool, 16> &tiles, bool host);

Counter counter(int64_t end, int64_t step, bool mask = true);
SmallVector<Counter, 8> padded(ArrayRef<Counter> counters, size_t length);
ByteAddressMode access(int64_t bytes, uint32_t loopMap = 0);
ByteAddressMode access(int64_t last, int64_t defaultBytes, uint32_t loopMap);
SyncProducer producerSync(bool increment, uint8_t depth = 0);
SyncWatcher tileWatcher(TileSyncFlag flag, int32_t initial, uint32_t stride,
                        uint8_t depth = 0, bool bykj = false,
                        uint8_t thread = 0);
DmaWatcher dmaWatcher(SyncWatcher watcher, bool stall = false);

TileHeader tileHeader(const std::array<bool, 16> &multicast);
Emitted
tileFence(TileFence fence,
          const std::array<bool, 16> &multicast = bits<16>("1111111111111111"));
Emitted scalarFence(ScalarFence fence, Role role = Role::Plain);
Emitted dma(DmaOperation operation, const std::array<bool, 16> &multicast,
            const std::array<bool, 8> &registers = {});
Emitted tensor(TensorOp operation, const std::array<bool, 16> &multicast,
               const std::array<bool, 8> &registers);
Emitted load(int64_t value, const std::array<bool, 16> &multicast,
             uint8_t baseRegister,
             std::optional<std::array<bool, 4>> threads = std::nullopt);
Emitted
tagged(std::variant<ScalarFence, HibDma, PopInput, RingInfeed, RingOutfeed>
           instruction);

inline constexpr std::array<bool, 20> kTileFenceValid = {
    true, true, true, true, true, true, true,  true,  true,  true,
    true, true, true, true, true, true, false, false, false, false};
SmallVector<Emitted, 0> resetAllSyncFlags();
Emitted localReset();
SmallVector<Emitted, 0> subscribeVc(const std::array<bool, 8> &tileChannels,
                                    const std::array<bool, 8> &scalarChannels);
SmallVector<Emitted, 0> groupFences();

LogicalResult unsupported(Operation *op, const Twine &reason);

} // namespace mlir::darwinn::codegen

#endif
