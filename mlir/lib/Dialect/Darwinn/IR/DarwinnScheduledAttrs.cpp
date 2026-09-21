#include "mlir/Dialect/Darwinn/IR/DarwinnScheduledAttrs.h"
#include "mlir/IR/Diagnostics.h"

using namespace mlir;
using namespace mlir::darwinn;

static LogicalResult
verifyIntegerArray(llvm::function_ref<InFlightDiagnostic()> emitError,
                   StringRef name, ArrayAttr array) {
  if (!array)
    return success();

  for (Attribute value : array) {
    if (!isa<IntegerAttr>(value))
      return emitError() << name << " must contain only integer attributes";
  }

  return success();
}

LogicalResult
MappingAttr::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                    AffineMapAttr forwardIndexTransformation,
                    AffineMapAttr reverseIndexTransformation) {
  if (!forwardIndexTransformation || !reverseIndexTransformation)
    return emitError() << "mapping requires forward and reverse affine maps";
  return success();
}

LogicalResult TensorOpSliceAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError, ArrayAttr id,
    ArrayAttr read_domain, IntegerAttr, IntegerAttr, AffineMapAttr, IntegerAttr,
    IntegerAttr, IntegerAttr, IntegerAttr, AffineMapAttr, AffineMapAttr,
    AffineMapAttr, AffineMapAttr, IntegerAttr, AffineMapAttr, IntegerAttr,
    IntegerAttr, IntegerAttr, IntegerAttr, IntegerAttr, IntegerAttr,
    IntegerAttr, IntegerAttr, AffineMapAttr, IntegerAttr, IntegerAttr,
    AffineMapAttr, ArrayAttr write_domain, AffineMapAttr, IntegerAttr,
    IntegerAttr, IntegerAttr, IntegerAttr, IntegerAttr, IntegerAttr,
    IntegerAttr) {
  if (failed(verifyIntegerArray(emitError, "id", id)) ||
      failed(verifyIntegerArray(emitError, "read_domain", read_domain)))
    return failure();
  return verifyIntegerArray(emitError, "write_domain", write_domain);
}

LogicalResult NarrowToWideSliceAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError, ArrayAttr id,
    IntegerAttr, ArrayAttr read_domain, AffineMapAttr, IntegerAttr, IntegerAttr,
    IntegerAttr, IntegerAttr, IntegerAttr, IntegerAttr, IntegerAttr,
    IntegerAttr, IntegerAttr, IntegerAttr, IntegerAttr, IntegerAttr,
    IntegerAttr, IntegerAttr, IntegerAttr, ArrayAttr write_domain,
    AffineMapAttr, IntegerAttr, IntegerAttr, IntegerAttr, IntegerAttr,
    IntegerAttr, IntegerAttr, IntegerAttr, IntegerAttr, IntegerAttr,
    IntegerAttr, IntegerAttr, IntegerAttr, IntegerAttr, BoolAttr, IntegerAttr,
    IntegerAttr, IntegerAttr, IntegerAttr, IntegerAttr, IntegerAttr) {
  if (failed(verifyIntegerArray(emitError, "id", id)) ||
      failed(verifyIntegerArray(emitError, "read_domain", read_domain)))
    return failure();
  return verifyIntegerArray(emitError, "write_domain", write_domain);
}

LogicalResult
TensorOpShardAttr::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                          ArrayAttr shardId, ArrayAttr slices) {
  if (failed(verifyIntegerArray(emitError, "shard_id", shardId)))
    return failure();
  if (!slices)
    return success();

  for (Attribute slice : slices) {
    if (!isa<TensorOpSliceAttr>(slice))
      return emitError()
             << "slices must contain only darwinn.tensor_op_slice attributes";
  }

  return success();
}

LogicalResult NarrowToWideShardAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError, ArrayAttr shardId,
    ArrayAttr slices) {
  if (failed(verifyIntegerArray(emitError, "shard_id", shardId)))
    return failure();
  if (!slices)
    return success();

  for (Attribute slice : slices) {
    if (!isa<NarrowToWideSliceAttr>(slice))
      return emitError() << "slices must contain only "
                            "darwinn.narrow_to_wide_slice attributes";
  }

  return success();
}
