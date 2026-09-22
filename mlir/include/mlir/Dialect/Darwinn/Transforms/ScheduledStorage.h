#ifndef MLIR_DIALECT_DARWINN_TRANSFORMS_SCHEDULEDSTORAGE_H
#define MLIR_DIALECT_DARWINN_TRANSFORMS_SCHEDULEDSTORAGE_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Target/Darwinn/Allocation.h"
#include "llvm/Support/Error.h"

namespace mlir::darwinn {

enum class ScheduledStoragePhase { Compute, HostTransfer, HostRelayout };
enum class ScheduledStoragePool { Narrow, Host };

struct ScheduledStorageEvent {
  Operation *operation;
  uint64_t position;
  ScheduledStoragePhase phase;
};

struct ScheduledStorageLimits {
  uint64_t narrowCapacityUnits;
  uint64_t hostCapacityBytes;
};

struct ScheduledStorageOwner {
  Value owner;
  ScheduledStoragePool pool;
  AllocationRecord request;
  bool borrowed;
};

struct ScheduledStorageAccess {
  uint64_t eventIndex;
  uint32_t operandIndex;
  uint64_t ownerIdentity;
  uint32_t tileRow;
  uint32_t tileColumn;
  uint32_t thread;
  uint64_t byteOffset;
  uint64_t byteSpan;
};

class ScheduledStorage {
public:
  func::FuncOp getScope() const { return scope; }
  ArrayRef<ScheduledStorageOwner> getOwners() const { return owners; }
  ArrayRef<ScheduledStorageAccess> getAccesses() const { return accesses; }

private:
  friend FailureOr<ScheduledStorage>
      deriveScheduledStorage(func::FuncOp, ArrayRef<ScheduledStorageEvent>,
                             ValueRange);

  func::FuncOp scope;
  SmallVector<ScheduledStorageOwner> owners;
  SmallVector<ScheduledStorageAccess> accesses;
};

struct ScheduledStoragePlacement {
  uint64_t ownerIdentity;
  uint64_t byteBase;
  uint64_t byteSize;
};

struct ScheduledStorageAddress {
  uint64_t accessIndex;
  uint64_t byteAddress;
};

FailureOr<ScheduledStorage>
deriveScheduledStorage(func::FuncOp scope,
                       ArrayRef<ScheduledStorageEvent> events,
                       ValueRange externalOwners);

llvm::Expected<SmallVector<ScheduledStorageAddress>>
bindStoragePlacements(const ScheduledStorage &storage,
                      ArrayRef<ScheduledStoragePlacement> placements,
                      ScheduledStorageLimits limits);

}

#endif
