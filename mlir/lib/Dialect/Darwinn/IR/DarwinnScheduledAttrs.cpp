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

static LogicalResult
verifyNonnegativeI32Array(llvm::function_ref<InFlightDiagnostic()> emitError,
                          StringRef name, ArrayAttr array) {
  if (!array)
    return success();

  for (Attribute value : array) {
    auto integer = dyn_cast<IntegerAttr>(value);

    if (!integer || !integer.getType().isSignlessInteger(32) ||
        integer.getValue().isNegative())
      return emitError() << name << " must contain nonnegative i32 values";
  }

  return success();
}

LogicalResult
JoinAttributesAttr::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                           ArrayAttr descriptors) {
  if (!descriptors)
    return emitError() << "join_attributes requires descriptors";

  for (Attribute descriptor : descriptors) {
    if (!isa<LocalCopyAttributesAttr>(descriptor))
      return emitError() << "descriptors must contain only "
                            "darwinn.local_copy_attributes attributes";
  }

  return success();
}

LogicalResult LocalCopyAttributesAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError, ArrayAttr limits,
    AffineMapAttr readTraversal, AffineMapAttr writeTraversal) {
  if (!limits || !readTraversal || !writeTraversal)
    return emitError() << "local_copy_attributes requires limits and read and "
                          "write traversals";

  if (failed(verifyNonnegativeI32Array(emitError, "limits", limits)))
    return failure();

  for (AffineMapAttr attribute : {readTraversal, writeTraversal}) {
    AffineMap map = attribute.getValue();

    if (map.getNumSymbols() || map.getNumDims() != limits.size())
      return emitError() << "local copy traversals require one dimension per "
                            "limit and no symbols";
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

LogicalResult NarrowToNarrowSliceAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError, ArrayAttr id,
    ArrayAttr readDomain, AffineMapAttr readAffineMap, IntegerAttr readBytes,
    ArrayAttr writeDomain, AffineMapAttr writeAffineMap,
    IntegerAttr writeBytes) {
  if (failed(verifyNonnegativeI32Array(emitError, "id", id)) ||
      failed(verifyNonnegativeI32Array(emitError, "read_domain", readDomain)) ||
      failed(verifyNonnegativeI32Array(emitError, "write_domain", writeDomain)))
    return failure();

  auto verifyTransfer = [&](ArrayAttr domain, AffineMapAttr affineMap,
                            IntegerAttr bytes) -> LogicalResult {
    if (static_cast<bool>(domain) != static_cast<bool>(affineMap))
      return emitError()
             << "narrow copy requires domain and affine map together";

    if (affineMap) {
      AffineMap map = affineMap.getValue();

      if (map.getNumSymbols() || map.getNumDims() != domain.size() ||
          map.getNumResults() != 1)
        return emitError() << "narrow copy affine maps require one dimension "
                              "per domain extent and one address result";
    }

    if (bytes && (!bytes.getType().isSignlessInteger(32) ||
                  !bytes.getValue().isStrictlyPositive()))
      return emitError()
             << "narrow copy byte counts must be positive i32 values";

    return success();
  };

  if (failed(verifyTransfer(readDomain, readAffineMap, readBytes)) ||
      failed(verifyTransfer(writeDomain, writeAffineMap, writeBytes)))
    return failure();

  if (readBytes && writeBytes && readBytes != writeBytes)
    return emitError()
           << "narrow copy requires matching read and write byte counts";

  return success();
}

LogicalResult NarrowToNarrowShardAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError, ArrayAttr shardId,
    ArrayAttr slices) {
  if (!shardId || !slices)
    return emitError() << "narrow copy shard requires shard_id and slices";

  for (Attribute attribute : shardId) {
    auto indices = dyn_cast<ArrayAttr>(attribute);

    if (!indices)
      return emitError() << "shard_id must contain arrays of i32 values";

    if (failed(verifyNonnegativeI32Array(emitError, "shard_id", indices)))
      return failure();
  }

  for (Attribute slice : slices) {
    if (!isa<NarrowToNarrowSliceAttr>(slice))
      return emitError() << "slices must contain only "
                            "darwinn.narrow_to_narrow_slice attributes";
  }

  return success();
}
