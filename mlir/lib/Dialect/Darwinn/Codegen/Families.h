#ifndef MLIR_LIB_DIALECT_DARWINN_CODEGEN_FAMILIES_H
#define MLIR_LIB_DIALECT_DARWINN_CODEGEN_FAMILIES_H

#include "Codegen.h"

namespace mlir::darwinn::codegen {

using Body = FailureOr<SmallVector<Emitted, 0>>;

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
Body ringReshape(Operation *op, Context &context);

Operation *throughViews(Operation *op);
Operation *storage(Operation *op);
Box ownerTileBox(Operation *owner);
Box viewThreadBox(Operation *view, int64_t thread);
FailureOr<int64_t> operandAddress(Operation *view, int64_t thread,
                                  Context &context, Operation *owner = nullptr);
std::pair<Index, int64_t> operandLayout(Operation *view,
                                        Operation *owner = nullptr);
std::array<bool, 16> activeTiles(Operation *op);
FailureOr<SmallVector<Emitted, 0>> registers(Operation *op, Context &context,
                                             ArrayRef<int64_t> threads);
FailureOr<SmallVector<Emitted, 0>> registers(Operation *op, Context &context);
std::pair<float, float> clips(Operation *op);
std::optional<int32_t> immediateBias(Operation *op);
Body initialization(Operation *op, Context &context);
Body unary(Operation *op, Context &context);
Body elementwise(Operation *op, Context &context);
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
};

FailureOr<VmcPlan> vmcPlan(Operation *op);
int64_t stencilTaps(Operation *op);
bool hasBias(Operation *op);
Body vmc(Operation *op, Context &context);
Body stencil(Operation *op, Context &context);

int64_t permuteWideRows(Operation *op);
Body permuteCopy(Operation *op, Context &context);

int64_t gatherWideRows(Operation *op);
Body gatherRows(Operation *op, Context &context);
Body gatherColumns(Operation *op, Context &context);

Body interpolate(Operation *op, Context &context);

} // namespace mlir::darwinn::codegen

#endif
