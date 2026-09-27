#ifndef MLIR_LIB_DIALECT_DARWINN_TRANSFORMS_SLICINGMODEL_H
#define MLIR_LIB_DIALECT_DARWINN_TRANSFORMS_SLICINGMODEL_H

#include "mlir/Dialect/Darwinn/IR/DarwinnOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineMap.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include <map>
#include <optional>
#include <tuple>
#include <vector>

namespace mlir::darwinn::slicing {

struct Tile {
  SmallVector<int64_t> lo;
  SmallVector<int64_t> hi;
  bool operator==(const Tile &other) const {
    return lo == other.lo && hi == other.hi;
  }
};

struct SliceMaps {
  AffineMap begins;
  AffineMap ends;
};

struct Code {
  SmallVector<int64_t, 3> domain;
  SliceMaps maps;
  bool isUnsliced() const {
    return domain.size() == 2 && domain[0] == 1 && domain[1] == 1;
  }
};

using Resources = std::map<int, int64_t>;

struct Estimate {
  int64_t base = 0;
  Resources resources;
};

int64_t cycles(const Estimate &estimate);

struct LinearForm {
  SmallVector<int64_t> coefficients;
  int64_t constant = 0;
};

std::optional<LinearForm> linearize(AffineExpr expression, unsigned dims);
ArrayRef<int64_t> shapeOf(Value value);
DistributedMemorySpace memorySpaceOf(Value value);
AffineMap traversalOf(Operation *operation);
FailureOr<SmallVector<int64_t>> iterationExtents(unsigned dims,
                                                 ArrayRef<Value> views);
Tile reshapeTile(ArrayRef<int64_t> from, ArrayRef<int64_t> to,
                 const Tile &tile);
AffineMap unitReshapeMap(ArrayRef<int64_t> from, ArrayRef<int64_t> to,
                         MLIRContext *context);

struct SlicingBlock {
  SmallVector<Operation *> members;
  Operation *anchor = nullptr;
  Value key;
};

class SlicingModel {
public:
  explicit SlicingModel(func::FuncOp function);

  LogicalResult build();

  unsigned numBlocks() const { return blocks.size(); }
  const SlicingBlock &block(unsigned index) const { return blocks[index]; }
  std::optional<unsigned> blockOf(Operation *operation) const;

  ArrayRef<unsigned> predecessors(unsigned block) const;
  ArrayRef<unsigned> successors(unsigned block) const;
  ArrayRef<Operation *> edgeOperations(unsigned from, unsigned to) const;

  const Code &code(unsigned id) const { return codes[id]; }
  unsigned unslicedCode(unsigned block) const { return unsliced[block]; }
  ArrayRef<unsigned> candidates(unsigned block) const {
    return candidateLists[block];
  }
  bool isShared(unsigned block) const;
  FailureOr<unsigned> sharedCode(unsigned block, ArrayRef<unsigned> state);

  FailureOr<const DenseMap<Value, SliceMaps> *> derive(unsigned block,
                                                       unsigned code);
  FailureOr<const SmallVector<Tile> *> tiles(Value value, unsigned code);

  FailureOr<Estimate> estimate(Operation *operation, ArrayRef<unsigned> state);

  LogicalResult emit(ArrayRef<unsigned> state);

  func::FuncOp function;

private:
  LogicalResult partition();
  void buildGraph();
  LogicalResult generateCodes();
  unsigned intern(Code code);

  FailureOr<Estimate> estimateCompute(Operation *operation, unsigned code);
  FailureOr<Estimate> estimateCopy(CopyOpOp copy, unsigned code);
  FailureOr<Estimate> estimateInterpolate(InterpolateHardwareOp interpolate,
                                          unsigned code);
  FailureOr<Estimate> estimateRedistribute(RedistributeOp redistribute,
                                           ArrayRef<unsigned> state);

  SmallVector<SlicingBlock> blocks;
  DenseMap<Operation *, unsigned> blockIndex;
  SmallVector<SmallVector<unsigned>> preds;
  SmallVector<SmallVector<unsigned>> succs;
  DenseMap<std::pair<unsigned, unsigned>, SmallVector<Operation *>> edges;

  SmallVector<Code> codes;
  std::map<std::tuple<std::vector<int64_t>, const void *, const void *>,
           unsigned>
      codeIndex;
  SmallVector<unsigned> unsliced;
  SmallVector<SmallVector<unsigned>> candidateLists;

  std::map<std::pair<unsigned, unsigned>, DenseMap<Value, SliceMaps>>
      derivations;
  std::map<std::pair<const void *, unsigned>, SmallVector<Tile>> tileCache;
  std::map<std::tuple<Operation *, unsigned, unsigned>, Estimate>
      redistributeCache;
  DenseMap<std::pair<Operation *, unsigned>, Estimate> computeCache;
};

} // namespace mlir::darwinn::slicing

#endif
