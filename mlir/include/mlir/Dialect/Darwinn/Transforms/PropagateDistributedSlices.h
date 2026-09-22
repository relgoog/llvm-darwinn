#ifndef MLIR_DIALECT_DARWINN_TRANSFORMS_PROPAGATEDISTRIBUTEDSLICES_H
#define MLIR_DIALECT_DARWINN_TRANSFORMS_PROPAGATEDISTRIBUTEDSLICES_H

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <utility>

namespace mlir {
namespace func {
class FuncOp;
}

namespace darwinn {

struct DistributedSlice {
  SmallVector<int32_t, 3> domain;
  AffineMap begins;
  AffineMap ends;
};

struct DistributedSliceAssignment {
  Value output;
  DistributedSlice slice;
};

class DistributedSlicePropagationPlan {
public:
  ArrayRef<DistributedSliceAssignment> getAssignments() const {
    return assignments;
  }

  ArrayRef<DistributedSliceAssignment> getStorageAssignments() const {
    return storageAssignments;
  }

private:
  struct Snapshot {
    Operation *operation;
    Operation *parent;
    Block *block;
    SmallVector<Value> operands;
    SmallVector<Type> operandTypes;
    SmallVector<Type> resultTypes;
    DictionaryAttr attributes;
    SmallVector<SmallVector<std::pair<Operation *, unsigned>>> users;
  };

  SmallVector<DistributedSliceAssignment> assignments;
  SmallVector<DistributedSliceAssignment> storageAssignments;
  SmallVector<Snapshot, 0> snapshots;

  friend FailureOr<DistributedSlicePropagationPlan>
  planDistributedSlicePropagation(
      func::FuncOp function,
      ArrayRef<DistributedSliceAssignment> selectedOutputs);
  friend LogicalResult
  applyDistributedSlicePropagation(const DistributedSlicePropagationPlan &plan);
};

FailureOr<DistributedSlicePropagationPlan> planDistributedSlicePropagation(
    func::FuncOp function,
    ArrayRef<DistributedSliceAssignment> selectedOutputs);
LogicalResult
applyDistributedSlicePropagation(const DistributedSlicePropagationPlan &plan);

}
}

#endif
