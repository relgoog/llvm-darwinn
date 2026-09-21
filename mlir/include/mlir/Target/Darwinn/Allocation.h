#ifndef MLIR_TARGET_DARWINN_ALLOCATION_H
#define MLIR_TARGET_DARWINN_ALLOCATION_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include <cstdint>

namespace mlir::darwinn {

struct AllocationRecord {
  uint64_t identity;
  uint64_t first;
  uint64_t last;
  uint64_t size;
  uint64_t alignment;
};

struct AllocationOffset {
  uint64_t identity;
  uint64_t offset;
};

struct StorageAllocation {
  llvm::SmallVector<AllocationOffset> offsets;
  uint64_t peakUsage = 0;
};

llvm::Expected<StorageAllocation>
allocateStorage(llvm::ArrayRef<AllocationRecord> records, uint64_t capacity);

}

#endif
