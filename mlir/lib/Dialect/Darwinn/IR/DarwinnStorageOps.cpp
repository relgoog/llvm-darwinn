#include "mlir/Dialect/Darwinn/IR/DarwinnOps.h"
#include "mlir/IR/AffineMap.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::darwinn;

static DistributedMemorySpace getStorageMemorySpace(Type type) {
  return llvm::TypeSwitch<Type, DistributedMemorySpace>(type)
      .Case<DistributedTensorType, DistributedViewType, EmptyTensorType,
            WriteViewType, FilledViewType>(
          [](auto storageType) { return storageType.getMemorySpace(); })
      .Default([](Type) -> DistributedMemorySpace {
        llvm_unreachable("expected a distributed storage or view type");
      });
}

static LogicalResult verifyMap(Operation *operation, StringRef name,
                               AffineMap map, unsigned dimensions,
                               unsigned results) {
  if (map.getNumSymbols() != 0)
    return operation->emitOpError() << name << " cannot have unbound symbols";

  if (map.getNumDims() != dimensions || map.getNumResults() != results)
    return operation->emitOpError()
           << name << " requires " << dimensions << " dimensions and "
           << results << " results";

  return success();
}

static LogicalResult verifyIndexTransformations(Operation *operation,
                                                ShapedType input,
                                                ShapedType output,
                                                AffineMapAttr forward,
                                                AffineMapAttr reverse) {
  if (!forward && !reverse)
    return success();

  if (!forward || !reverse)
    return operation->emitOpError(
        "requires both forward and reverse index transformations");

  if (failed(verifyMap(operation, "forward_index_transformation",
                       forward.getValue(), input.getRank(), output.getRank())))
    return failure();

  return verifyMap(operation, "reverse_index_transformation",
                   reverse.getValue(), output.getRank(), input.getRank());
}

static LogicalResult verifySlicing(Operation *operation, StringRef name,
                                   AffineMapAttr begins, ArrayAttr domain,
                                   AffineMapAttr ends, ShapedType output) {
  if (!begins || !domain || !ends)
    return operation->emitOpError()
           << name
           << " requires affine begins and ends and an i32 domain array";

  for (Attribute extent : domain) {
    auto integer = dyn_cast<IntegerAttr>(extent);

    if (!integer || !integer.getType().isSignlessInteger(32) ||
        integer.getValue().isNegative())
      return operation->emitOpError()
             << name << " domain requires nonnegative i32 extents";
  }

  if (failed(verifyMap(operation, (name + "_begins").str(), begins.getValue(),
                       domain.size(), output.getRank())))
    return failure();

  return verifyMap(operation, (name + "_ends").str(), ends.getValue(),
                   domain.size(), output.getRank());
}

static LogicalResult verifyDiscardableSlicing(Operation *operation,
                                              StringRef name,
                                              ShapedType output) {
  Attribute begins = operation->getDiscardableAttr((name + "_begins").str());
  Attribute domain = operation->getDiscardableAttr((name + "_domain").str());
  Attribute ends = operation->getDiscardableAttr((name + "_ends").str());

  if (!begins && !domain && !ends)
    return success();

  return verifySlicing(operation, name,
                       dyn_cast_if_present<AffineMapAttr>(begins),
                       dyn_cast_if_present<ArrayAttr>(domain),
                       dyn_cast_if_present<AffineMapAttr>(ends), output);
}

static LogicalResult verifyTraversal(Operation *operation, AffineMap traversal,
                                     ShapedType output) {
  return verifyMap(operation, "traversal", traversal, traversal.getNumDims(),
                   output.getRank());
}

static LogicalResult verifyViewStorage(Operation *operation, ShapedType input,
                                       ShapedType output) {
  if (input.getElementType() != output.getElementType())
    return operation->emitOpError(
        "requires matching storage and view element types");

  if (getStorageMemorySpace(input) != getStorageMemorySpace(output))
    return operation->emitOpError(
        "requires matching storage and view memory spaces");

  return success();
}

LogicalResult CreateEmptyTensorOp::verify() {
  return verifySlicing(*this, "slicing", getSlicingBeginsAttr(),
                       getSlicingDomainAttr(), getSlicingEndsAttr(),
                       cast<ShapedType>(getOutput().getType()));
}

LogicalResult FillOp::verify() {
  auto output = cast<ShapedType>(getOutput().getType());
  auto constant = dyn_cast<ShapedType>(getValue().getType());

  if (!constant || !constant.hasRank() ||
      constant.getShape() != output.getShape() ||
      constant.getElementType() != output.getElementType())
    return emitOpError(
        "requires constant shape and element type to match output");

  return verifySlicing(*this, "slicing", getSlicingBeginsAttr(),
                       getSlicingDomainAttr(), getSlicingEndsAttr(), output);
}

LogicalResult GetTensorOp::verify() {
  if (getInput().getType() != getOutput().getType())
    return emitOpError("requires matching input and output types");

  return verifyDiscardableSlicing(*this, "slicing",
                                  cast<ShapedType>(getOutput().getType()));
}

LogicalResult ReshapeOpOp::verify() {
  auto input = cast<ShapedType>(getInput().getType());
  auto output = cast<ShapedType>(getOutput().getType());

  if (failed(verifyViewStorage(*this, input, output)))
    return failure();

  if (input.hasStaticShape() && output.hasStaticShape()) {
    auto elementCount = [](ArrayRef<int64_t> shape) {
      llvm::APInt count(64, 1);

      for (int64_t dimension : shape)
        count = count.zext(count.getBitWidth() + 64) * dimension;

      return count;
    };

    if (!llvm::APInt::isSameValue(elementCount(input.getShape()),
                                  elementCount(output.getShape())))
      return emitOpError("requires matching input and output element counts");
  }

  if (failed(verifyIndexTransformations(*this, input, output,
                                        getForwardIndexTransformationAttr(),
                                        getReverseIndexTransformationAttr())))
    return failure();

  return verifyDiscardableSlicing(*this, "sharding", output);
}

LogicalResult CopyOpOp::verify() {
  auto input = cast<ShapedType>(getInput().getType());
  auto output = cast<ShapedType>(getOutput().getType());

  if (input.getElementType() != output.getElementType())
    return emitOpError("requires matching input and output element types");

  if (failed(verifyIndexTransformations(*this, input, output,
                                        getForwardIndexTransformationAttr(),
                                        getReverseIndexTransformationAttr())) ||
      failed(verifySlicing(*this, "slicing", getSlicingBeginsAttr(),
                           getSlicingDomainAttr(), getSlicingEndsAttr(),
                           output)))
    return failure();

  return verifyTraversal(*this, getTraversal(), output);
}

LogicalResult DistributedCreateViewOp::verify() {
  auto input = cast<ShapedType>(getInput().getType());
  auto output = cast<ShapedType>(getOutput().getType());

  if (getStorageMemorySpace(input) != getStorageMemorySpace(output))
    return emitOpError("requires matching storage and view memory spaces");

  if (AffineMapAttr forward = getForwardIndexTransformationAttr()) {
    if (failed(verifyMap(*this, "forward_index_transformation",
                         forward.getValue(), input.getRank(),
                         output.getRank())))
      return failure();
  }

  if (AffineMapAttr reverse = getReverseIndexTransformationAttr()) {
    if (failed(verifyMap(*this, "reverse_index_transformation",
                         reverse.getValue(), output.getRank(),
                         input.getRank())))
      return failure();
  }

  if (failed(verifyDiscardableSlicing(*this, "slicing", output)))
    return failure();

  if (Attribute traversal = getOperation()->getDiscardableAttr("traversal")) {
    auto map = dyn_cast<AffineMapAttr>(traversal);

    if (!map)
      return emitOpError("requires an affine traversal map");

    return verifyTraversal(*this, map.getValue(), output);
  }

  return success();
}

LogicalResult RedistributeOp::verify() {
  auto input = cast<ShapedType>(getInput().getType());
  auto output = cast<ShapedType>(getOutput().getType());

  if (input.getElementType() != output.getElementType())
    return emitOpError("requires matching input and output element types");

  if (Value destination = getDestination()) {
    if (!isa<FilledViewType>(output))
      return emitOpError("requires a filled view result when writing a view");

    auto writeView = cast<ShapedType>(destination.getType());

    if (writeView.getShape() != output.getShape())
      return emitOpError("requires the filled view to match its destination");

    if (failed(verifyViewStorage(*this, writeView, output)))
      return failure();
  } else if (!isa<DistributedTensorType>(output)) {
    return emitOpError("requires a tensor result without a destination view");
  }

  if (MappingAttr mapping = getMappingAttr()) {
    if (failed(verifyIndexTransformations(
            *this, input, output, mapping.getForwardIndexTransformation(),
            mapping.getReverseIndexTransformation())))
      return failure();
  }

  return verifySlicing(*this, "slicing", getSlicingBeginsAttr(),
                       getSlicingDomainAttr(), getSlicingEndsAttr(), output);
}

LogicalResult CommunicatedCreateEmptyTensorOp::verify() {
  return verifySlicing(*this, "slicing", getSlicingBeginsAttr(),
                       getSlicingDomainAttr(), getSlicingEndsAttr(),
                       cast<ShapedType>(getOutput().getType()));
}

LogicalResult CommunicatedCreateWriteViewOp::verify() {
  auto input = cast<ShapedType>(getInput().getType());
  auto output = cast<ShapedType>(getOutput().getType());

  if (failed(verifyViewStorage(*this, input, output)))
    return failure();

  return verifyMap(*this, "reverse_index_transformation",
                   getReverseIndexTransformation(), output.getRank(),
                   input.getRank());
}

LogicalResult CommunicatedJoinViewsOp::verify() {
  if (getInputs().empty())
    return emitOpError("requires at least one filled view");

  auto output = cast<ShapedType>(getOutput().getType());
  Value storage;

  for (Value input : getInputs()) {
    if (failed(verifyViewStorage(*this, cast<ShapedType>(input.getType()),
                                 output)))
      return failure();

    auto transfer = input.getDefiningOp<RedistributeOp>();

    if (!transfer || !transfer.getDestination())
      continue;

    auto writeView = transfer.getDestination()
                         .getDefiningOp<CommunicatedCreateWriteViewOp>();

    if (!writeView)
      continue;

    if (storage && storage != writeView.getInput())
      return emitOpError("cannot join views of different storage values");

    storage = writeView.getInput();
  }

  if (storage) {
    auto storageType = cast<ShapedType>(storage.getType());

    if (storageType.getShape() != output.getShape())
      return emitOpError("requires output to match the underlying storage");

    if (failed(verifyViewStorage(*this, storageType, output)))
      return failure();
  }

  return success();
}
