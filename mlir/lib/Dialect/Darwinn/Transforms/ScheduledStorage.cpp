#include "mlir/Dialect/Darwinn/Transforms/ScheduledStorage.h"
#include "mlir/Dialect/Darwinn/IR/DarwinnOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <limits>
#include <map>

using namespace mlir;
using namespace mlir::darwinn;

namespace {

struct TensorShape {
  SmallVector<int64_t> dimensions;
  ScheduledStoragePool pool;
  uint64_t elementBytes;
};

struct Slice {
  uint32_t row;
  uint32_t column;
  uint32_t thread;
  SmallVector<int64_t> lower;
  SmallVector<int64_t> upper;
};

struct OwnerGeometry {
  TensorShape shape;
  std::map<std::pair<uint32_t, uint32_t>, Slice> tiles;
};

struct PendingAccess {
  ScheduledStorageAccess access;
  Slice slice;
};

LogicalResult unsupported(Operation *operation, const Twine &reason) {
  return operation->emitOpError()
         << "cannot derive scheduled storage because " << reason;
}

bool add(uint64_t left, uint64_t right, uint64_t &result) {
  if (right > std::numeric_limits<uint64_t>::max() - left)
    return false;
  result = left + right;
  return true;
}

bool multiply(uint64_t left, uint64_t right, uint64_t &result) {
  if (right && left > std::numeric_limits<uint64_t>::max() / right)
    return false;
  result = left * right;
  return true;
}

FailureOr<TensorShape> readShape(Value value) {
  TensorShape shape;
  Type element;
  DistributedMemorySpace memory;

  if (auto tensor = dyn_cast<DistributedTensorType>(value.getType())) {
    shape.dimensions.assign(tensor.getShape().begin(), tensor.getShape().end());
    element = tensor.getElementType();
    memory = tensor.getMemorySpace();
  } else if (auto view = dyn_cast<DistributedViewType>(value.getType())) {
    shape.dimensions.assign(view.getShape().begin(), view.getShape().end());
    element = view.getElementType();
    memory = view.getMemorySpace();
  } else {
    return failure();
  }

  if (shape.dimensions.empty() ||
      llvm::any_of(shape.dimensions,
                   [](int64_t extent) { return extent <= 0; }) ||
      (!element.isBF16() && !element.isF32()))
    return failure();

  if (memory == DistributedMemorySpace::TileMemory)
    shape.pool = ScheduledStoragePool::Narrow;
  else if (memory == DistributedMemorySpace::HostMemory)
    shape.pool = ScheduledStoragePool::Host;
  else
    return failure();

  shape.elementBytes = element.isBF16() ? 2 : 4;
  return shape;
}

bool sameShape(const TensorShape &left, const TensorShape &right) {
  return left.dimensions == right.dimensions &&
         left.elementBytes == right.elementBytes;
}

bool belongsTo(func::FuncOp scope, Value value) {
  if (!value || value.getContext() != scope.getContext())
    return false;
  Operation *parent = value.getParentBlock()->getParentOp();
  return parent == scope.getOperation() ||
         parent->getParentOfType<func::FuncOp>() == scope;
}

FailureOr<Value> resolveOwner(func::FuncOp scope, Value value) {
  while (true) {
    if (!belongsTo(scope, value))
      return failure();

    Operation *producer = value.getDefiningOp();
    if (!producer)
      return value;
    if (failed(verify(producer, false)))
      return failure();

    if (isa<CreateEmptyTensorOp, RedistributeOp>(producer))
      return value;

    if (auto compute = dyn_cast<TensorOpOp>(producer)) {
      if (!compute.getAuxiliaryTensors().empty())
        return failure();
      value = compute.getDestination();
      continue;
    }

    if (auto compute = dyn_cast<UnaryTensorOpOp>(producer)) {
      if (!compute.getAuxiliaryTensors().empty())
        return failure();
      value = compute.getDestination();
      continue;
    }

    if (!isa<GetTensorOp, DistributedCreateViewOp>(producer))
      return failure();

    auto output = readShape(value);
    auto input = readShape(producer->getOperand(0));
    if (failed(input) || failed(output) || !sameShape(*input, *output) ||
        input->pool != output->pool)
      return failure();

    for (StringRef name :
         {"forward_index_transformation", "reverse_index_transformation"}) {
      auto mapping = producer->getAttrOfType<AffineMapAttr>(name);
      if (mapping && !mapping.getValue().isIdentity())
        return failure();
    }

    value = producer->getOperand(0);
  }
}

FailureOr<int64_t> evaluate(AffineExpr expression,
                            ArrayRef<int64_t> coordinates) {
  if (auto constant = dyn_cast<AffineConstantExpr>(expression))
    return constant.getValue();

  if (auto dimension = dyn_cast<AffineDimExpr>(expression)) {
    if (dimension.getPosition() >= coordinates.size())
      return failure();
    return coordinates[dimension.getPosition()];
  }

  auto binary = dyn_cast<AffineBinaryOpExpr>(expression);
  if (!binary)
    return failure();

  auto left = evaluate(binary.getLHS(), coordinates);
  auto right = evaluate(binary.getRHS(), coordinates);
  if (failed(left) || failed(right))
    return failure();

  int64_t result;
  if (binary.getKind() == AffineExprKind::Add &&
      !llvm::AddOverflow(*left, *right, result))
    return result;
  if (binary.getKind() == AffineExprKind::Mul &&
      (isa<AffineConstantExpr>(binary.getLHS()) ||
       isa<AffineConstantExpr>(binary.getRHS())) &&
      !llvm::MulOverflow(*left, *right, result))
    return result;
  return failure();
}

FailureOr<SmallVector<Slice>> readSlices(Operation *operation,
                                         const TensorShape &shape) {
  auto begins = operation->getAttrOfType<AffineMapAttr>("slicing_begins");
  auto ends = operation->getAttrOfType<AffineMapAttr>("slicing_ends");
  auto domain = operation->getAttrOfType<ArrayAttr>("slicing_domain");
  if (!begins || !ends || !domain ||
      (domain.size() != 2 && domain.size() != 3) ||
      begins.getValue().getNumDims() != domain.size() ||
      ends.getValue().getNumDims() != domain.size() ||
      begins.getValue().getNumSymbols() || ends.getValue().getNumSymbols() ||
      begins.getValue().getNumResults() != shape.dimensions.size() ||
      ends.getValue().getNumResults() != shape.dimensions.size())
    return unsupported(operation, "storage slicing maps are incomplete");

  SmallVector<int64_t> extents;
  uint64_t count = 1;
  for (Attribute entry : domain) {
    auto integer = dyn_cast<IntegerAttr>(entry);
    if (!integer || integer.getInt() <= 0 ||
        integer.getInt() > std::numeric_limits<uint32_t>::max() ||
        !multiply(count, integer.getInt(), count))
      return unsupported(operation, "storage slicing domain is invalid");
    extents.push_back(integer.getInt());
  }

  SmallVector<Slice> slices;
  if (count > slices.max_size())
    return unsupported(operation,
                       "storage slicing count exceeds container size");

  if (extents.size() == 2)
    extents.push_back(1);

  for (int64_t row = 0; row < extents[0]; ++row) {
    for (int64_t column = 0; column < extents[1]; ++column) {
      for (int64_t thread = 0; thread < extents[2]; ++thread) {
        SmallVector<int64_t> coordinates{row, column, thread};
        Slice slice{static_cast<uint32_t>(row),
                    static_cast<uint32_t>(column),
                    static_cast<uint32_t>(thread),
                    {},
                    {}};

        for (unsigned axis = 0; axis < shape.dimensions.size(); ++axis) {
          auto lower = evaluate(begins.getValue().getResult(axis), coordinates);
          auto upper = evaluate(ends.getValue().getResult(axis), coordinates);
          if (failed(lower) || failed(upper) || *lower < 0 || *upper < *lower ||
              *upper >= shape.dimensions[axis])
            return unsupported(operation,
                               "storage slice is not an in-bounds box");
          slice.lower.push_back(*lower);
          slice.upper.push_back(*upper);
        }

        slices.push_back(std::move(slice));
      }
    }
  }

  return slices;
}

void includeSlice(OwnerGeometry &geometry, const Slice &slice) {
  auto [entry, inserted] = geometry.tiles.try_emplace(
      std::make_pair(slice.row, slice.column), slice);
  if (inserted)
    return;

  for (unsigned axis = 0; axis < slice.lower.size(); ++axis) {
    entry->second.lower[axis] =
        std::min(entry->second.lower[axis], slice.lower[axis]);
    entry->second.upper[axis] =
        std::max(entry->second.upper[axis], slice.upper[axis]);
  }
}

FailureOr<uint64_t> byteSize(const Slice &slice, uint64_t elementBytes) {
  uint64_t size = elementBytes;
  for (unsigned axis = 0; axis < slice.lower.size(); ++axis) {
    if (!multiply(size, slice.upper[axis] - slice.lower[axis] + 1, size))
      return failure();
  }
  return size;
}

FailureOr<std::pair<uint64_t, uint64_t>>
byteRange(const Slice &storage, const Slice &slice, uint64_t elementBytes) {
  uint64_t lower = 0;
  uint64_t upper = 0;
  for (unsigned axis = 0; axis < storage.lower.size(); ++axis) {
    if (slice.lower[axis] < storage.lower[axis] ||
        slice.upper[axis] > storage.upper[axis])
      return failure();
    uint64_t extent = storage.upper[axis] - storage.lower[axis] + 1;
    if (!multiply(lower, extent, lower) ||
        !add(lower, slice.lower[axis] - storage.lower[axis], lower) ||
        !multiply(upper, extent, upper) ||
        !add(upper, slice.upper[axis] - storage.lower[axis], upper))
      return failure();
  }

  uint64_t span;
  if (!add(upper - lower, 1, span) || !multiply(span, elementBytes, span) ||
      !multiply(lower, elementBytes, lower))
    return failure();
  return std::make_pair(lower, span);
}

Slice hostSlice(const TensorShape &shape) {
  Slice slice{0, 0, 0, SmallVector<int64_t>(shape.dimensions.size(), 0), {}};
  for (int64_t extent : shape.dimensions)
    slice.upper.push_back(extent - 1);
  return slice;
}

}

FailureOr<ScheduledStorage>
mlir::darwinn::deriveScheduledStorage(func::FuncOp scope,
                                      ArrayRef<ScheduledStorageEvent> events,
                                      ValueRange externalOwners) {
  if (!scope)
    return failure();

  ScheduledStorage storage;
  storage.scope = scope;
  DenseSet<Value> external;
  DenseMap<Value, uint64_t> identities;
  SmallVector<OwnerGeometry> geometry;
  SmallVector<PendingAccess> pending;

  for (Value owner : externalOwners) {
    auto canonical = resolveOwner(scope, owner);
    if (failed(canonical) || *canonical != owner ||
        !external.insert(owner).second)
      return unsupported(
          scope,
          "external owners must be unique canonical values in this function");
  }

  auto touch = [&](Value value, uint64_t eventIndex,
                   uint32_t operandIndex) -> LogicalResult {
    const ScheduledStorageEvent &event = events[eventIndex];
    auto canonical = resolveOwner(scope, value);
    if (failed(canonical))
      return unsupported(event.operation,
                         "value has an unsupported storage producer or alias");
    auto shape = readShape(*canonical);
    if (failed(shape))
      return unsupported(
          event.operation,
          "storage requires static BF16 or f32 host or tile tensors");

    auto [identity, inserted] =
        identities.try_emplace(*canonical, storage.owners.size());
    uint64_t ownerIndex = identity->second;
    if (inserted) {
      bool borrowed = external.contains(*canonical);
      Operation *producer = canonical->getDefiningOp();
      if (!producer && !borrowed)
        return unsupported(event.operation,
                           "block argument storage requires an external owner");

      if (auto transfer = dyn_cast_or_null<RedistributeOp>(producer)) {
        auto input = readShape(transfer.getInput());
        if (failed(input) || !sameShape(*input, *shape) ||
            input->pool == shape->pool || transfer.getDestination() ||
            transfer.getMappingAttr())
          return unsupported(producer, "redistribution requires identity "
                                       "host-to-tile or tile-to-host storage");
      }

      if (!borrowed) {
        bool createsStorage =
            producer == event.operation &&
            event.phase == ScheduledStoragePhase::HostTransfer;
        if (event.phase == ScheduledStoragePhase::Compute) {
          Value destination =
              isa<TensorOpOp>(event.operation)
                  ? cast<TensorOpOp>(event.operation).getDestination()
                  : cast<UnaryTensorOpOp>(event.operation).getDestination();
          auto destinationOwner = resolveOwner(scope, destination);
          createsStorage =
              succeeded(destinationOwner) && *destinationOwner == *canonical;
        }
        if (!createsStorage)
          return unsupported(
              event.operation,
              "storage used before its scheduled producer must be external");
      }

      storage.owners.push_back(
          {*canonical,
           shape->pool,
           {ownerIndex, event.position, event.position, 0,
            shape->pool == ScheduledStoragePool::Narrow ? 8u : 1u},
           borrowed});
      geometry.push_back({*shape, {}});
      if (shape->pool == ScheduledStoragePool::Narrow) {
        if (!producer)
          return unsupported(
              event.operation,
              "external tile arguments need explicit storage geometry");
        auto slices = readSlices(producer, *shape);
        if (failed(slices))
          return failure();
        for (const Slice &slice : *slices)
          includeSlice(geometry.back(), slice);
      } else {
        includeSlice(geometry.back(), hostSlice(*shape));
      }
    }

    auto &owner = storage.owners[ownerIndex];
    owner.request.first = std::min(owner.request.first, event.position);
    owner.request.last = std::max(owner.request.last, event.position);

    SmallVector<Slice> slices;
    if (shape->pool == ScheduledStoragePool::Host) {
      slices.push_back(hostSlice(*shape));
    } else {
      Value sliced = value;
      if (auto view = sliced.getDefiningOp<DistributedCreateViewOp>())
        sliced = view.getInput();
      Operation *sliceProducer = sliced.getDefiningOp();
      if (!sliceProducer ||
          !isa<GetTensorOp, CreateEmptyTensorOp, RedistributeOp>(sliceProducer))
        return unsupported(event.operation,
                           "tile access requires explicit slicing geometry");
      auto read = readSlices(sliceProducer, *shape);
      if (failed(read))
        return failure();
      slices = std::move(*read);
    }

    for (Slice &slice : slices) {
      includeSlice(geometry[ownerIndex], slice);
      pending.push_back({{eventIndex, operandIndex, ownerIndex, slice.row,
                          slice.column, slice.thread, 0, 0},
                         std::move(slice)});
    }
    return success();
  };

  DenseSet<std::pair<Operation *, unsigned>> phases;
  uint64_t previous = 0;
  for (auto [eventIndex, event] : llvm::enumerate(events)) {
    Operation *operation = event.operation;
    if (!operation || operation->getContext() != scope.getContext() ||
        operation->getParentOfType<func::FuncOp>() != scope ||
        event.position < previous ||
        !phases.insert({operation, static_cast<unsigned>(event.phase)}).second)
      return unsupported(
          scope,
          "events must be ordered unique operation phases from this function");
    previous = event.position;
    if (failed(verify(operation, false)))
      return failure();

    switch (event.phase) {
    case ScheduledStoragePhase::Compute:
      if (!isa<TensorOpOp, UnaryTensorOpOp>(operation))
        return unsupported(operation,
                           "compute event requires a tensor compute operation");
      if ((isa<TensorOpOp>(operation) && operation->getNumOperands() != 3) ||
          (isa<UnaryTensorOpOp>(operation) &&
           operation->getNumOperands() != 2) ||
          operation->hasAttr("resampler_options"))
        return unsupported(operation,
                           "auxiliary and resampler storage is unsupported");
      if (auto tensor = dyn_cast<TensorOpOp>(operation)) {
        if (tensor.getCompute().getInnerOperation() !=
                InnerOperationKind::Elementwise ||
            tensor.getShards().size() != 1)
          return unsupported(operation,
                             "tensor storage requires one elementwise shard");
        auto shard = dyn_cast<TensorOpShardAttr>(tensor.getShards()[0]);
        if (!shard || (shard.getSlices() && !shard.getSlices().empty()))
          return unsupported(operation, "tensor shard slicing is unsupported");
      } else {
        auto unary = cast<UnaryTensorOpOp>(operation);
        if (!unary.getSlice().empty() ||
            unary.getCompute().getInnerOperation().value_or(
                InnerOperationKind::Unary) != InnerOperationKind::Unary)
          return unsupported(
              operation, "unary storage requires an unsliced unary operation");
      }
      for (auto [operandIndex, operand] :
           llvm::enumerate(operation->getOperands())) {
        if (!operand.getDefiningOp<DistributedCreateViewOp>())
          return unsupported(operation,
                             "compute operands require identity storage views");
        if (failed(touch(operand, eventIndex, operandIndex)))
          return failure();
      }
      break;
    case ScheduledStoragePhase::HostTransfer:
      if (!isa<RedistributeOp>(operation) || operation->getNumOperands() != 1 ||
          operation->getNumResults() != 1)
        return unsupported(operation,
                           "transfer event requires a direct redistribution");
      if (failed(touch(operation->getOperand(0), eventIndex, 0)) ||
          failed(touch(operation->getResult(0), eventIndex, 1)))
        return failure();
      break;
    case ScheduledStoragePhase::HostRelayout: {
      auto transfer = dyn_cast<RedistributeOp>(operation);
      if (!transfer)
        return unsupported(operation,
                           "host relayout requires a redistribution");
      auto shape = readShape(transfer.getOutput());
      if (failed(shape) || shape->pool != ScheduledStoragePool::Host)
        return unsupported(operation, "host relayout requires a host result");
      if (failed(touch(transfer.getOutput(), eventIndex, 1)))
        return failure();
      break;
    }
    }
  }

  for (Value owner : external) {
    if (!identities.contains(owner))
      return unsupported(scope, "external owner is not used by the schedule");
  }

  for (auto [ownerIndex, owner] : llvm::enumerate(storage.owners)) {
    uint64_t bytes = 0;
    for (const auto &[tile, slice] : geometry[ownerIndex].tiles) {
      auto size = byteSize(slice, geometry[ownerIndex].shape.elementBytes);
      if (failed(size))
        return unsupported(scope, "storage byte size overflowed");
      bytes = std::max(bytes, *size);
    }
    if (owner.pool == ScheduledStoragePool::Narrow) {
      if (!add(bytes, 3, bytes))
        return unsupported(scope, "narrow allocation size overflowed");
      bytes /= 4;
    }
    storage.owners[ownerIndex].request.size = bytes;
  }

  for (PendingAccess &entry : pending) {
    auto &owner = geometry[entry.access.ownerIdentity];
    const Slice &tile = owner.tiles.at({entry.slice.row, entry.slice.column});
    auto range = byteRange(tile, entry.slice, owner.shape.elementBytes);
    if (failed(range))
      return unsupported(scope, "storage access byte range overflowed");
    entry.access.byteOffset = range->first;
    entry.access.byteSpan = range->second;
    storage.accesses.push_back(entry.access);
  }

  return storage;
}

llvm::Expected<SmallVector<ScheduledStorageAddress>>
mlir::darwinn::bindStoragePlacements(
    const ScheduledStorage &storage,
    ArrayRef<ScheduledStoragePlacement> placements,
    ScheduledStorageLimits limits) {
  auto invalid = [](const Twine &reason) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(), reason);
  };
  DenseMap<uint64_t, ScheduledStoragePlacement> byIdentity;
  for (const ScheduledStoragePlacement &placement : placements) {
    if (placement.ownerIdentity >= storage.getOwners().size() ||
        !byIdentity.try_emplace(placement.ownerIdentity, placement).second)
      return invalid(
          "storage placements contain an unknown or duplicate identity");
  }
  if (byIdentity.size() != storage.getOwners().size())
    return invalid("storage placements omit a required owner");

  for (const ScheduledStorageOwner &owner : storage.getOwners()) {
    const auto &placement = byIdentity.at(owner.request.identity);
    uint64_t unitBytes = owner.pool == ScheduledStoragePool::Narrow ? 4 : 1;
    uint64_t required;
    uint64_t alignment;
    uint64_t capacity = limits.hostCapacityBytes;
    uint64_t end;
    if (!multiply(owner.request.size, unitBytes, required) ||
        !multiply(owner.request.alignment, unitBytes, alignment) ||
        (owner.pool == ScheduledStoragePool::Narrow &&
         !multiply(limits.narrowCapacityUnits, unitBytes, capacity)) ||
        placement.byteSize < required || placement.byteBase % alignment ||
        !add(placement.byteBase, placement.byteSize, end) || end > capacity)
      return invalid("storage placement violates size, alignment, capacity, or "
                     "address bounds");

    for (const ScheduledStorageOwner &other : storage.getOwners()) {
      if (other.request.identity >= owner.request.identity ||
          other.pool != owner.pool ||
          owner.request.last < other.request.first ||
          other.request.last < owner.request.first)
        continue;
      const auto &otherPlacement = byIdentity.at(other.request.identity);
      uint64_t otherEnd;
      if (!add(otherPlacement.byteBase, otherPlacement.byteSize, otherEnd))
        return invalid("storage placement address overflowed");
      if (placement.byteBase < otherEnd && otherPlacement.byteBase < end)
        return invalid("storage placements overlap during inclusive lifetimes");
    }
  }

  SmallVector<ScheduledStorageAddress> addresses;
  for (auto [index, access] : llvm::enumerate(storage.getAccesses())) {
    const auto &placement = byIdentity.at(access.ownerIdentity);
    uint64_t end;
    uint64_t address;
    if (!add(access.byteOffset, access.byteSpan, end) ||
        end > placement.byteSize ||
        !add(placement.byteBase, access.byteOffset, address))
      return invalid("storage access exceeds its placement");
    addresses.push_back({index, address});
  }
  return addresses;
}
