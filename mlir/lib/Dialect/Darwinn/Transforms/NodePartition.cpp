#include "SlicingModel.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Darwinn/IR/DarwinnOps.h"
#include "mlir/Dialect/Darwinn/Transforms/DistributedPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::darwinn;
using namespace mlir::darwinn::slicing;

namespace {

constexpr StringLiteral kShardingAttributes[] = {
    "sharding_begins", "sharding_domain", "sharding_ends"};

DistributedTensorType inMemory(Type type, DistributedMemorySpace space) {
  auto shaped = cast<ShapedType>(type);
  return DistributedTensorType::get(type.getContext(), shaped.getShape(),
                                    shaped.getElementType(), space);
}

RedistributeOp unslicedRedistribute(OpBuilder &builder, Location location,
                                    Value input, DistributedTensorType type) {
  MLIRContext *context = builder.getContext();
  SmallVector<AffineExpr> begins, ends;
  for (int64_t size : type.getShape()) {
    begins.push_back(builder.getAffineConstantExpr(0));
    ends.push_back(builder.getAffineConstantExpr(size - 1));
  }
  return RedistributeOp::create(
      builder, location, type, input, Value{}, MappingAttr{},
      AffineMap::get(2, 0, begins, context), builder.getI32ArrayAttr({1, 1}),
      AffineMap::get(2, 0, ends, context));
}

void addIdentityTransformations(DistributedCreateViewOp view) {
  if (view.getForwardIndexTransformation())
    return;
  unsigned rank = cast<ShapedType>(view.getInput().getType()).getRank();
  auto identity = AffineMap::getMultiDimIdentityMap(rank, view.getContext());
  view.setForwardIndexTransformation(identity);
  view.setReverseIndexTransformation(identity);
}

SmallVector<int64_t> evaluate(AffineMap map, int64_t shard) {
  SmallVector<int64_t> point(map.getNumDims(), shard);
  return map.compose(point);
}

Tile shardBox(Operation *operation, int64_t shard) {
  auto begins = operation->getAttrOfType<AffineMapAttr>("sharding_begins");
  auto ends = operation->getAttrOfType<AffineMapAttr>("sharding_ends");
  return {evaluate(begins.getValue(), shard), evaluate(ends.getValue(), shard)};
}

SmallVector<int64_t> extentOf(const Tile &box) {
  SmallVector<int64_t> shape;
  for (auto [low, high] : llvm::zip(box.lo, box.hi))
    shape.push_back(high - low + 1);
  return shape;
}

void setUnsliced(Operation *operation, ArrayRef<int64_t> shape,
                 bool discardable) {
  MLIRContext *context = operation->getContext();
  Builder builder(context);
  SmallVector<AffineExpr> begins, ends;
  for (int64_t size : shape) {
    begins.push_back(builder.getAffineConstantExpr(0));
    ends.push_back(builder.getAffineConstantExpr(size - 1));
  }
  auto set = [&](StringRef name, Attribute value) {
    if (discardable)
      operation->setDiscardableAttr(name, value);
    else
      operation->setAttr(name, value);
  };
  set("slicing_begins",
      AffineMapAttr::get(AffineMap::get(2, 0, begins, context)));
  set("slicing_domain", builder.getI32ArrayAttr({1, 1}));
  set("slicing_ends", AffineMapAttr::get(AffineMap::get(2, 0, ends, context)));
}

FailureOr<DenseElementsAttr> sliceElements(Attribute value, const Tile &box,
                                           ShapedType type) {
  auto dense = dyn_cast<DenseElementsAttr>(value);
  if (!dense)
    return failure();
  if (dense.isSplat())
    return DenseElementsAttr::get(type, dense.getSplatValue<Attribute>());
  ArrayRef<int64_t> shape = dense.getType().getShape();
  int64_t bytes = dense.getElementType().getIntOrFloatBitWidth() / 8;
  if (bytes == 0)
    return failure();
  ArrayRef<char> raw = dense.getRawData();
  SmallVector<char> sliced;
  SmallVector<int64_t> strides(shape.size(), 1);
  for (unsigned dim = shape.size() - 1; dim-- > 0;)
    strides[dim] = strides[dim + 1] * shape[dim + 1];
  SmallVector<int64_t> index(box.lo);
  int64_t row = box.hi.back() - box.lo.back() + 1;
  while (true) {
    int64_t offset = 0;
    for (unsigned dim = 0; dim < shape.size(); ++dim)
      offset += index[dim] * strides[dim];
    llvm::append_range(sliced, raw.slice(offset * bytes, row * bytes));
    int dim = static_cast<int>(shape.size()) - 2;
    while (dim >= 0 && ++index[dim] > box.hi[dim]) {
      index[dim] = box.lo[dim];
      --dim;
    }
    if (dim < 0)
      break;
  }
  return DenseElementsAttr::getFromRawBuffer(type, sliced);
}

Value hostShardView(OpBuilder &builder, Location location, Value source,
                    RedistributeOp redistribute, const Tile &outputBox) {
  MLIRContext *context = builder.getContext();
  ArrayRef<int64_t> shape = shapeOf(source);
  Tile box = outputBox;
  if (MappingAttr mapping = redistribute.getMappingAttr()) {
    AffineMap reverse = mapping.getReverseIndexTransformation().getValue();
    box = {reverse.compose(outputBox.lo), reverse.compose(outputBox.hi)};
  }
  SmallVector<AffineExpr> forward, backward;
  for (unsigned dim = 0; dim < shape.size(); ++dim) {
    box.lo[dim] = std::max<int64_t>(box.lo[dim], 0);
    box.hi[dim] = std::min(box.hi[dim], shape[dim] - 1);
    AffineExpr position = builder.getAffineDimExpr(dim);
    forward.push_back(position - box.lo[dim]);
    backward.push_back(position + box.lo[dim]);
  }
  auto type = DistributedViewType::get(
      context, extentOf(box),
      cast<ShapedType>(source.getType()).getElementType(),
      DistributedMemorySpace::HostMemory);
  auto view = DistributedCreateViewOp::create(
      builder, location, type, source,
      AffineMapAttr::get(AffineMap::get(shape.size(), 0, forward, context)),
      AffineMapAttr::get(AffineMap::get(shape.size(), 0, backward, context)));
  setUnsliced(view, type.getShape(), /*discardable=*/true);
  return view;
}

void readInputsFromHost(Operation *operation) {
  if (isa<RedistributeOp>(operation))
    return;
  for (OpOperand &operand : operation->getOpOperands()) {
    if (memorySpaceOf(operand.get()) != DistributedMemorySpace::HostMemory)
      continue;
    OpBuilder builder(operation);
    auto copy = unslicedRedistribute(
        builder, operation->getLoc(), operand.get(),
        inMemory(operand.get().getType(), DistributedMemorySpace::TileMemory));
    operand.set(copy);
  }
}

LogicalResult inlineUnitGroup(affine::AffineParallelOp group) {
  Block *body = group.getBody();
  auto yield = cast<affine::AffineYieldOp>(body->getTerminator());
  for (Operation &operation :
       llvm::make_early_inc_range(body->without_terminator())) {
    if (isa<affine::AffineParallelOp>(operation))
      return operation.emitOpError("sharded groups are not partitioned yet");
    operation.moveBefore(group);
    for (StringRef name : kShardingAttributes)
      operation.removeAttr(name);
    if (auto view = dyn_cast<DistributedCreateViewOp>(operation))
      addIdentityTransformations(view);
    readInputsFromHost(&operation);
  }

  OpBuilder builder(group);
  SmallVector<Operation *> bypassed;
  for (auto [result, produced] :
       llvm::zip(group.getResults(), yield.getOperands())) {
    if (result.use_empty())
      continue;
    Value exported = produced;
    auto view = produced.getDefiningOp<DistributedCreateViewOp>();
    if (view && !view->hasAttr("traversal")) {
      exported = view.getInput();
      bypassed.push_back(view);
    }
    if (memorySpaceOf(exported) == DistributedMemorySpace::HostMemory)
      exported.setType(
          inMemory(exported.getType(), DistributedMemorySpace::TileMemory));
    builder.setInsertionPointAfterValue(exported);
    auto host = unslicedRedistribute(
        builder, group.getLoc(), exported,
        inMemory(exported.getType(), DistributedMemorySpace::HostMemory));
    result.replaceAllUsesWith(host.getResult());
  }
  group.erase();
  for (Operation *view : bypassed)
    if (view->use_empty())
      view->erase();
  return success();
}

LogicalResult partitionShardedGroup(affine::AffineParallelOp group) {
  affine::AffineParallelOp inner = group;
  if (group.getNumDims() == 0) {
    Block *body = group.getBody();
    for (Operation &operation :
         llvm::make_early_inc_range(body->without_terminator())) {
      if (auto nested = dyn_cast<affine::AffineParallelOp>(operation)) {
        inner = nested;
        continue;
      }
      operation.moveBefore(group);
      for (StringRef name : kShardingAttributes)
        operation.removeAttr(name);
      readInputsFromHost(&operation);
    }
  }
  if (inner.getNumDims() != 1)
    return inner.emitOpError("shards over more than one dimension");
  auto bounds = inner.getConstantRanges();
  if (!bounds)
    return inner.emitOpError("has non constant shard bounds");
  int64_t shards = (*bounds)[0];

  Block *body = inner.getBody();
  auto yield = cast<affine::AffineYieldOp>(body->getTerminator());
  Operation *producer = yield.getOperand(0).getDefiningOp();
  auto fullType = cast<ShapedType>(yield.getOperand(0).getType());
  OpBuilder builder(group);
  Location location = group.getLoc();
  Value empty;
  SmallVector<Value> filled;
  SmallVector<Operation *> outerViews;

  for (int64_t shard = 0; shard < shards; ++shard) {
    IRMapping mapping;
    for (Operation &operation : body->without_terminator()) {
      Tile box = shardBox(&operation, shard);
      SmallVector<int64_t> shape = extentOf(box);
      for (OpOperand &operand : operation.getOpOperands()) {
        Value value = operand.get();
        if (mapping.contains(value))
          continue;
        auto view = value.getDefiningOp<DistributedCreateViewOp>();
        if (view && !inner->isAncestor(view.getOperation())) {
          Operation *shared = view.getInput().getDefiningOp();
          shared->setDiscardableAttr("tensor_shared_by_users",
                                     builder.getBoolAttr(true));
          auto copy = unslicedRedistribute(
              builder, location, view.getInput(),
              cast<DistributedTensorType>(view.getInput().getType()));
          IRMapping local;
          local.map(view.getInput(), copy.getResult());
          auto shardView =
              cast<DistributedCreateViewOp>(builder.clone(*view, local));
          addIdentityTransformations(shardView);
          mapping.map(value, shardView.getResult());
          if (!llvm::is_contained(outerViews, view.getOperation()))
            outerViews.push_back(view);
          continue;
        }
        auto redistribute = dyn_cast<RedistributeOp>(operation);
        if (redistribute && operand.getOperandNumber() == 0 &&
            memorySpaceOf(value) == DistributedMemorySpace::HostMemory)
          mapping.map(value, hostShardView(builder, location, value,
                                           redistribute, box));
      }
      if (&operation == producer && shard == 0) {
        auto emptyType = EmptyTensorType::get(
            builder.getContext(), fullType.getShape(),
            fullType.getElementType(), DistributedMemorySpace::HostMemory);
        auto created = CommunicatedCreateEmptyTensorOp::create(
            builder, location, emptyType, AffineMapAttr{}, ArrayAttr{},
            AffineMapAttr{});
        setUnsliced(created, fullType.getShape(), /*discardable=*/false);
        empty = created;
      }
      Operation *clone = builder.clone(operation, mapping);
      for (StringRef name : kShardingAttributes)
        clone->removeAttr(name);
      for (OpResult result : clone->getResults()) {
        auto type = cast<ShapedType>(result.getType());
        if (isa<DistributedViewType>(type))
          result.setType(DistributedViewType::get(builder.getContext(), shape,
                                                  type.getElementType(),
                                                  memorySpaceOf(result)));
        else
          result.setType(inMemory(type.clone(shape), memorySpaceOf(result)));
      }
      if (clone->hasAttr("slicing_domain"))
        setUnsliced(clone, shape, /*discardable=*/false);
      if (auto view = dyn_cast<DistributedCreateViewOp>(clone))
        addIdentityTransformations(view);
      if (auto fill = dyn_cast<FillOp>(clone)) {
        auto value = sliceElements(
            fill.getValue(), box,
            RankedTensorType::get(shape, fill.getValue().getElementType()));
        if (failed(value))
          return operation.emitOpError("has a value that cannot be sharded");
        fill.setValueAttr(*value);
      }
      if (&operation == producer) {
        SmallVector<AffineExpr> offsets;
        for (unsigned dim = 0; dim < shape.size(); ++dim)
          offsets.push_back(builder.getAffineDimExpr(dim) + box.lo[dim]);
        auto writeType = WriteViewType::get(builder.getContext(), shape,
                                            fullType.getElementType(),
                                            DistributedMemorySpace::HostMemory);
        auto write = CommunicatedCreateWriteViewOp::create(
            builder, location, writeType, empty,
            AffineMap::get(shape.size(), 0, offsets, builder.getContext()));
        auto filledType = FilledViewType::get(
            builder.getContext(), shape, fullType.getElementType(),
            DistributedMemorySpace::HostMemory);
        auto store = RedistributeOp::create(
            builder, location, filledType, clone->getResult(0), write,
            MappingAttr{}, AffineMapAttr{}, ArrayAttr{}, AffineMapAttr{});
        setUnsliced(store, shape, /*discardable=*/false);
        filled.push_back(store);
      }
      mapping.map(operation.getResults(), clone->getResults());
    }
  }
  auto joined = CommunicatedJoinViewsOp::create(
      builder, location, inMemory(fullType, DistributedMemorySpace::HostMemory),
      filled);
  group.getResult(0).replaceAllUsesWith(joined.getResult());
  group.erase();
  for (Operation *view : outerViews)
    if (view->use_empty())
      view->erase();
  return success();
}

class NodePartitionPass
    : public PassWrapper<NodePartitionPass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NodePartitionPass)

  StringRef getArgument() const final { return "darwinn-node-partition"; }

  StringRef getDescription() const final {
    return "Materialize sharding groups as nodes that exchange tensors "
           "through host memory";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<DarwinnDialect, affine::AffineDialect>();
  }

  void runOnOperation() final {
    Builder builder(&getContext());
    for (auto reshape : getOperation().getOps<ReshapeOpOp>()) {
      if (reshape->hasAttr("sharding_domain"))
        continue;
      SmallVector<AffineExpr> begins, ends;
      for (int64_t size : shapeOf(reshape.getOutput())) {
        begins.push_back(builder.getAffineConstantExpr(0));
        ends.push_back(builder.getAffineConstantExpr(size - 1));
      }
      reshape->setAttr("sharding_begins", AffineMapAttr::get(AffineMap::get(
                                              0, 0, begins, &getContext())));
      reshape->setAttr("sharding_domain", builder.getI32ArrayAttr({}));
      reshape->setAttr("sharding_ends", AffineMapAttr::get(AffineMap::get(
                                            0, 0, ends, &getContext())));
    }
    SmallVector<affine::AffineParallelOp> groups;
    for (Operation &operation : getOperation().getBody().front())
      if (auto group = dyn_cast<affine::AffineParallelOp>(operation))
        groups.push_back(group);
    for (affine::AffineParallelOp group : groups) {
      bool sharded = group.getNumDims() > 0;
      group.getBody()->walk([&](affine::AffineParallelOp) { sharded = true; });
      if (failed(sharded ? partitionShardedGroup(group)
                         : inlineUnitGroup(group)))
        return signalPassFailure();
    }
  }
};

}

std::unique_ptr<Pass> mlir::darwinn::createNodePartitionPass() {
  return std::make_unique<NodePartitionPass>();
}

void mlir::darwinn::registerNodePartitionPass() {
  PassRegistration<NodePartitionPass>();
}
