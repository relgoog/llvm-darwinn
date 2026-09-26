#include "mlir/Dialect/Darwinn/Transforms/PropagatingSlicing.h"
#include "SlicingModel.h"
#include "mlir/Dialect/Darwinn/IR/DarwinnOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/BitVector.h"
#include <map>

using namespace mlir;
using namespace mlir::darwinn;
using namespace mlir::darwinn::slicing;

namespace {

SmallVector<unsigned> slotOrder(ArrayRef<unsigned> keys) {
  constexpr unsigned kSlots = 64;
  std::map<unsigned, unsigned> slots;
  for (unsigned key : keys) {
    unsigned slot = (37 * key) % kSlots;
    while (slots.count(slot))
      slot = (slot + 1) % kSlots;
    slots[slot] = key;
  }
  SmallVector<unsigned> result;
  for (auto &[slot, key] : slots)
    result.push_back(key);
  return result;
}

class Solver {
public:
  explicit Solver(SlicingModel &model) : model(model) {}

  LogicalResult run();
  ArrayRef<unsigned> assignment() const { return state; }

private:
  FailureOr<int64_t> operationCycles(Operation *operation,
                                     ArrayRef<unsigned> codes);
  FailureOr<int64_t> cost(unsigned block, unsigned code,
                          const llvm::BitVector &slicedBlocks);
  LogicalResult visit(unsigned block, std::optional<unsigned> parent);
  std::vector<const void *> signature(unsigned block) const;

  SlicingModel &model;
  SmallVector<unsigned> state;
  llvm::BitVector sliced;
  std::map<std::vector<const void *>, unsigned> memo;
};

FailureOr<int64_t> Solver::operationCycles(Operation *operation,
                                           ArrayRef<unsigned> codes) {
  auto estimate = model.estimate(operation, codes);
  if (failed(estimate))
    return failure();
  return cycles(*estimate);
}

FailureOr<int64_t> Solver::cost(unsigned block, unsigned code,
                                const llvm::BitVector &slicedBlocks) {
  SmallVector<unsigned> codes(state);
  codes[block] = code;
  int64_t total = 0;
  for (Operation *member : model.block(block).members) {
    auto value = operationCycles(member, codes);
    if (failed(value))
      return failure();
    total += *value;
  }
  for (unsigned successor : model.successors(block))
    for (Operation *edge : model.edgeOperations(block, successor)) {
      auto value = operationCycles(edge, codes);
      if (failed(value))
        return failure();
      total += slicedBlocks.test(successor) ? *value : *value / 2;
    }
  return total;
}

std::vector<const void *> Solver::signature(unsigned block) const {
  std::vector<const void *> result;
  auto describe = [&](unsigned index) {
    for (Operation *member : model.block(index).members) {
      result.push_back(member->getName().getAsOpaquePointer());
      result.push_back(member->getAttrDictionary().getAsOpaquePointer());
      for (Type type : member->getOperandTypes())
        result.push_back(type.getAsOpaquePointer());
      for (Type type : member->getResultTypes())
        result.push_back(type.getAsOpaquePointer());
      result.push_back(nullptr);
    }
  };
  describe(block);
  SmallVector<unsigned> neighbours(model.predecessors(block));
  llvm::append_range(neighbours, model.successors(block));
  llvm::sort(neighbours);
  neighbours.erase(llvm::unique(neighbours), neighbours.end());
  for (unsigned neighbour : neighbours) {
    describe(neighbour);
    result.push_back(reinterpret_cast<const void *>(
        static_cast<uintptr_t>(state[neighbour]) + 1));
  }
  return result;
}

LogicalResult Solver::visit(unsigned block, std::optional<unsigned> parent) {
  if (sliced.test(block))
    return success();
  std::vector<const void *> key = signature(block);
  SmallVector<unsigned> choices;
  auto remembered = memo.find(key);
  if (remembered != memo.end())
    choices.push_back(remembered->second);
  else
    llvm::append_range(choices, model.candidates(block));

  std::optional<std::pair<unsigned, int64_t>> best;
  for (unsigned code : choices) {
    auto value = cost(block, code, sliced);
    if (failed(value))
      return failure();
    if (!best || *value < best->second)
      best = {code, *value};
  }
  unsigned previous = state[block];
  state[block] = best->first;
  memo.try_emplace(key, best->first);

  if (parent) {
    ArrayRef<Operation *> edge =
        llvm::is_contained(model.predecessors(*parent), block)
            ? model.edgeOperations(block, *parent)
            : model.edgeOperations(*parent, block);
    int64_t conversion = 0;
    for (Operation *operation : edge) {
      auto value = operationCycles(operation, state);
      if (failed(value))
        return failure();
      conversion += *value;
    }
    if (conversion >= static_cast<int64_t>(0.1 * best->second)) {
      state[block] = previous;
      return success();
    }
  }

  sliced.set(block);
  for (unsigned neighbour : slotOrder(model.predecessors(block)))
    if (failed(visit(neighbour, block)))
      return failure();
  for (unsigned neighbour : slotOrder(model.successors(block)))
    if (failed(visit(neighbour, block)))
      return failure();
  return success();
}

LogicalResult Solver::run() {
  unsigned blocks = model.numBlocks();
  for (unsigned block = 0; block < blocks; ++block)
    state.push_back(model.unslicedCode(block));

  llvm::BitVector everything(blocks, true);
  SmallVector<std::pair<int64_t, unsigned>> ranking;
  for (unsigned block = 0; block < blocks; ++block) {
    auto value = cost(block, state[block], everything);
    if (failed(value))
      return failure();
    ranking.push_back({*value, block});
  }
  llvm::sort(ranking, [](auto left, auto right) {
    return left.first != right.first ? left.first > right.first
                                     : left.second > right.second;
  });

  for (unsigned round = 0; round < 2; ++round) {
    sliced = llvm::BitVector(blocks, false);
    for (auto [unused, seed] : ranking)
      if (!sliced.test(seed) && failed(visit(seed, std::nullopt)))
        return failure();
  }
  return success();
}

class PropagatingSlicingPass
    : public PassWrapper<PropagatingSlicingPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PropagatingSlicingPass)

  StringRef getArgument() const final { return "darwinn-propagating-slicing"; }

  StringRef getDescription() const final {
    return "Choose a tile slicing for every distributed block by cost driven "
           "propagation";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<DarwinnDialect, func::FuncDialect>();
  }

  void runOnOperation() final {
    for (auto function : getOperation().getOps<func::FuncOp>()) {
      SlicingModel model(function);
      if (failed(model.build()))
        return signalPassFailure();
      Solver solver(model);
      if (failed(solver.run()) || failed(model.emit(solver.assignment())))
        return signalPassFailure();
    }
  }
};

}

std::unique_ptr<Pass> mlir::darwinn::createPropagatingSlicingPass() {
  return std::make_unique<PropagatingSlicingPass>();
}

void mlir::darwinn::registerPropagatingSlicingPass() {
  PassRegistration<PropagatingSlicingPass>();
}
