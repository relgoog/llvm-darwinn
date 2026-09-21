#include "mlir/Dialect/Darwinn/IR/DarwinnScheduledAttrs.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace mlir::darwinn;

LogicalResult
LoopCounterAttr::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                        int64_t tripCount, int64_t byteStride) {
  if (tripCount <= 0)
    return emitError() << "tripCount must be positive";

  int64_t displacement;

  if (llvm::MulOverflow(tripCount - 1, byteStride, displacement))
    return emitError() << "loop displacement exceeds signed 64-bit range";

  return success();
}

LogicalResult
ByteAccessPlanAttr::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                           uint32_t loopMap, int64_t lastBytes,
                           int64_t defaultBytes) {
  if (loopMap > 255)
    return emitError() << "loopMap must fit eight loop slots";
  if (lastBytes <= 0 || defaultBytes <= 0)
    return emitError() << "byte access sizes must be positive";

  return success();
}

LogicalResult MemoryTraversalPlanAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError,
    ArrayRef<LoopCounterAttr> counters, ByteAccessPlanAttr access) {
  if (counters.empty() || counters.size() > 8)
    return emitError()
           << "memory traversal requires one through eight counters";

  for (LoopCounterAttr counter : counters) {
    if (!counter)
      return emitError() << "memory traversal requires typed loop counters";
    if (failed(LoopCounterAttr::verify(emitError, counter.getTripCount(),
                                       counter.getByteStride())))
      return failure();
  }

  if (access)
    return ByteAccessPlanAttr::verify(emitError, access.getLoopMap(),
                                      access.getLastBytes(),
                                      access.getDefaultBytes());

  return success();
}

LogicalResult TensorTraversalPlanAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError,
    ArrayRef<int64_t> mainTripCounts, MemoryTraversalPlanAttr narrowRead,
    MemoryTraversalPlanAttr narrowWrite, MemoryTraversalPlanAttr parameters,
    MemoryTraversalPlanAttr sums) {
  if (mainTripCounts.empty() || mainTripCounts.size() > 8 ||
      llvm::any_of(mainTripCounts, [](int64_t count) { return count <= 0; }))
    return emitError()
           << "main operation requires one through eight positive trip counts";

  for (MemoryTraversalPlanAttr path : {narrowRead, narrowWrite, sums}) {
    if (!path)
      return emitError()
             << "narrow read, narrow write and sums plans are required";
    if (failed(MemoryTraversalPlanAttr::verify(emitError, path.getCounters(),
                                               path.getAccess())))
      return failure();
  }

  if (parameters &&
      failed(MemoryTraversalPlanAttr::verify(
          emitError, parameters.getCounters(), parameters.getAccess())))
    return failure();

  if (!narrowRead.getAccess() || !narrowWrite.getAccess())
    return emitError() << "narrow paths require byte access plans";
  if (sums.getAccess() || (parameters && parameters.getAccess()))
    return emitError()
           << "parameter and sums paths cannot use byte access plans";

  return success();
}
