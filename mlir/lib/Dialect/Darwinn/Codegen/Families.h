#ifndef MLIR_LIB_DIALECT_DARWINN_CODEGEN_FAMILIES_H
#define MLIR_LIB_DIALECT_DARWINN_CODEGEN_FAMILIES_H

#include "Codegen.h"

namespace mlir::darwinn::codegen {

using Body = FailureOr<SmallVector<Emitted, 0>>;
using TileValues = SmallVector<std::pair<int64_t, int64_t>>;
using Groups = SmallVector<std::pair<int64_t, SmallVector<int64_t>>>;
using Order = std::function<Groups(Groups)>;

constexpr int64_t kAccess = 32;

struct Loop {
  int64_t count;
  int64_t source;
  int64_t destination;
};

SmallVector<Loop> mergeLoops(ArrayRef<Loop> loops);
Index applyForward(Operation *op, ArrayRef<int64_t> point);
Index applyReverse(Operation *op, ArrayRef<int64_t> point);
Emitted
ringConsumer(int64_t base, ArrayRef<Loop> loops, int64_t first, int64_t last,
             const std::array<bool, 8> &channels,
             const std::array<bool, 16> &multicast,
             RingDestination destination = RingDestination::NarrowMemory);
SmallVector<Emitted, 0> infeed(int64_t total,
                               const std::array<bool, 8> &channels,
                               const std::array<bool, 17> &targets,
                               InputFifo fifo);
Emitted outfeed(int64_t total, const std::array<bool, 8> &channels);
Emitted hibGather(ArrayRef<int64_t> view, ArrayRef<int64_t> buffer,
                  int64_t elementBytes, DmaQueue queue, int64_t index);

Body transferLoad(Operation *op, Context &context);
Body transferStore(Operation *op, Context &context);
Body modelOutput(Operation *op, Context &context);
Body fill(Operation *op, Context &context);
Body padding(Operation *op, Context &context);

SmallVector<TileBox> clampedTiles(Operation *op);
Body scatter(Operation *op, Context &context);
int64_t scatterStagingWords(Operation *op);
int64_t scatterRelayWords(Operation *op);
SmallVector<Emitted, 0> registerLoads(const TileValues &values, uint8_t reg,
                                      const Order &order,
                                      std::optional<int64_t> thread);
Groups latestFirst(Groups groups);
Body ringReshape(Operation *op, Context &context);

Operation *throughViews(Operation *op);
Operation *storage(Operation *op);
Box ownerTileBox(Operation *owner, int64_t tile = 0);
Box viewThreadBox(Operation *view, int64_t thread, int64_t tile = 0);
FailureOr<int64_t> operandAddress(Operation *view, int64_t thread,
                                  Context &context, Operation *owner = nullptr);
std::pair<Index, int64_t>
operandLayout(Operation *view, Operation *owner = nullptr, int64_t tile = 0);
std::array<bool, 16> activeTiles(Operation *op);
FailureOr<SmallVector<Emitted, 0>> registers(Operation *op, Context &context,
                                             ArrayRef<int64_t> threads);
FailureOr<SmallVector<Emitted, 0>> registers(Operation *op, Context &context,
                                             ArrayRef<int64_t> threads,
                                             const std::array<bool, 16> &tiles);
FailureOr<SmallVector<Emitted, 0>> registers(Operation *op, Context &context);
std::pair<float, float> clips(Operation *op);
std::optional<int32_t> immediateBias(Operation *op);
Body initialization(Operation *op, Context &context);
Body unary(Operation *op, Context &context);
Body elementwise(Operation *op, Context &context);
bool tensorProduct(Operation *op);
int64_t tensorProductRows(Operation *op);
FailureOr<Emitted> coefficientTables(Operation *op, NluFunctionKind function,
                                     const Context &context);

struct VmcPlan {
  bool transposed = false;
  bool single = false;
  int64_t lanesPadded = 0;
  int64_t cin = 0;
  int64_t chunk = 0;
  int64_t chunks = 0;
  int64_t lastChunk = 0;
  int64_t taps = 0;
  int64_t rows = 0;
  int64_t cols = 0;
  int64_t pixels = 0;
  int64_t inner = 0;
  int64_t outer = 0;
  int64_t lanes = 0;
  int64_t outBlocks = 0;
  int64_t weightsRows = 0;
  int64_t sumsRows = 0;
  int64_t biasRows = 0;
  int64_t tapGroup = 1;
};

FailureOr<VmcPlan> vmcPlan(Operation *op);
Operation *weightsView(Operation *op);
int64_t stencilTaps(Operation *op);
int64_t stencilBlocks(Operation *op);
bool macCopy(Operation *op);
SmallVector<unsigned> windowDims(Operation *op);
bool hasBias(Operation *op);
Body vmc(Operation *op, Context &context);
Body stencil(Operation *op, Context &context);

int64_t permuteWideRows(Operation *op);
Body permuteCopy(Operation *op, Context &context);

int64_t gatherWideRows(Operation *op);
int64_t gatherStagingWords(Operation *op);
Body gatherRows(Operation *op, Context &context);
Body gatherColumns(Operation *op, Context &context);

Body interpolate(Operation *op, Context &context);

} // namespace mlir::darwinn::codegen

#endif
