#include "SlicingModel.h"
#include "mlir/Dialect/Darwinn/IR/DarwinnOps.h"
#include "mlir/Dialect/Darwinn/Transforms/DistributedPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace mlir::darwinn;
using namespace mlir::darwinn::slicing;

namespace {

void lower(SynchronizedCopyOpOp copy) {
  OpBuilder builder(copy);
  MLIRContext *context = builder.getContext();
  auto shard = NarrowToWideShardAttr::get(context, builder.getArrayAttr({}),
                                          builder.getArrayAttr({}));
  auto wide = NarrowToWideOp::create(
      builder, copy.getLoc(), copy.getType(), TypeRange{}, copy.getInput(),
      ValueRange{}, builder.getArrayAttr({shard}), copy.getTraversalAttr(),
      copy.getForwardIndexTransformationAttr(), AffineMapAttr{}, IntegerAttr{},
      IntegerAttr{}, ArrayAttr{}, ArrayAttr{});
  copy.replaceAllUsesWith(wide.getOutput());
  copy.erase();
}

void lower(SynchronizedComputeOpOp compute) {
  OpBuilder builder(compute);
  MLIRContext *context = builder.getContext();
  auto shard = TensorOpShardAttr::get(context, builder.getArrayAttr({}),
                                      builder.getArrayAttr({}));
  auto tensor = TensorOpOp::create(
      builder, compute.getLoc(), compute.getType(), compute.getLhs(),
      compute.getRhs(), compute.getDestination(), compute.getAuxiliaryTensors(),
      compute.getComputeAttr(), builder.getArrayAttr({shard}),
      compute.getTraversalAttr(), compute.getAuxiliaryTensorTypesAttr(),
      ArrayAttr{});
  compute.replaceAllUsesWith(tensor.getOutput());
  compute.erase();
}

void lower(SynchronizedUnaryComputeOpOp compute) {
  OpBuilder builder(compute);
  auto tensor = UnaryTensorOpOp::create(
      builder, compute.getLoc(), compute.getType(), compute.getInput(),
      compute.getDestination(), compute.getAuxiliaryTensors(),
      compute.getComputeAttr(), builder.getArrayAttr({}),
      compute.getTraversalAttr(), compute.getAuxiliaryTensorTypesAttr(),
      ArrayAttr{}, compute.getCustomConstraintsAttr(), ResamplerOptionsAttr{});
  compute.replaceAllUsesWith(tensor.getOutput());
  compute.erase();
}

LogicalResult lower(CommunicatedLocalCopyOp copy) {
  OpBuilder builder(copy);
  auto inputType = copy.getInput().getType();
  auto writeView = copy.getDestination().getDefiningOp<CommunicatedCreateWriteViewOp>();
  auto attributes = copy.getLocalCopyAttributes();

  if (!writeView || !inputType.hasStaticShape() || inputType.getRank() == 0 ||
      inputType.getMemorySpace() != DistributedMemorySpace::TileMemory ||
      !attributes.getReadTraversal().getValue().isIdentity() ||
      !attributes.getWriteTraversal().getValue().isIdentity() ||
      attributes.getLimits().size() != inputType.getShape().size())
    return copy.emitOpError("requires a dense identity copy into a communicated write view");

  auto storageType = writeView.getInput().getType();
  unsigned rank = inputType.getRank();
  Type element = inputType.getElementType();

  if (!storageType.hasStaticShape() || storageType.getRank() != rank ||
      storageType.getShape().drop_back() != inputType.getShape().drop_back() ||
      storageType.getMemorySpace() != DistributedMemorySpace::TileMemory ||
      storageType.getElementType() != element || !isa<IntegerType, FloatType>(element) ||
      (element.getIntOrFloatBitWidth() != 8 && element.getIntOrFloatBitWidth() != 16 &&
       element.getIntOrFloatBitWidth() != 32))
    return copy.emitOpError("requires matching static tile storage with 8, 16 or 32 bit elements");

  AffineMap reverse = writeView.getReverseIndexTransformation();
  int64_t channelOffset = 0;

  for (unsigned axis = 0; axis < rank; ++axis) {
    auto limit = dyn_cast<IntegerAttr>(attributes.getLimits()[axis]);
    auto offset = dyn_cast<AffineConstantExpr>(
        simplifyAffineExpr(reverse.getResult(axis) - builder.getAffineDimExpr(axis), rank, 0));

    if (!limit || limit.getInt() != inputType.getDimSize(axis) || !offset ||
        offset.getValue() < 0 || (axis + 1 != rank && offset.getValue() != 0))
      return copy.emitOpError("supports only complete inputs written at a channel offset");

    channelOffset = offset.getValue();
  }

  if (inputType.getShape().back() <= 0 ||
      channelOffset > storageType.getShape().back() - inputType.getShape().back())
    return copy.emitOpError("requires the input channels to fit the destination storage");

  auto readSlicing = [&](Value value) -> FailureOr<Code> {
    Operation *owner = value.getDefiningOp();

    if (!owner)
      return copy.emitOpError("requires explicit input and output tile slicing");

    auto begins = owner->getAttrOfType<AffineMapAttr>("slicing_begins");
    auto ends = owner->getAttrOfType<AffineMapAttr>("slicing_ends");
    auto domain = owner->getAttrOfType<ArrayAttr>("slicing_domain");

    if (!begins || !ends || !domain || domain.size() != 2 || begins.getValue().getNumDims() != 2 ||
        ends.getValue().getNumDims() != 2 || begins.getValue().getNumSymbols() ||
        ends.getValue().getNumSymbols() || begins.getValue().getNumResults() != rank ||
        ends.getValue().getNumResults() != rank)
      return copy.emitOpError("requires slicing with mesh rows and columns and no thread axis");

    Code code{{}, {begins.getValue(), ends.getValue()}};

    for (Attribute attribute : domain) {
      auto extent = dyn_cast<IntegerAttr>(attribute);

      if (!extent || extent.getInt() <= 0 || extent.getInt() > 4)
        return copy.emitOpError("requires tile slicing within the 4 by 4 mesh");

      code.domain.push_back(extent.getInt());
    }

    return code;
  };

  auto inputSlicing = readSlicing(copy.getInput());
  auto outputSlicing = readSlicing(writeView.getInput());

  if (failed(inputSlicing) || failed(outputSlicing))
    return failure();

  if (inputSlicing->domain != outputSlicing->domain)
    return copy.emitOpError("requires matching input and output tile ownership");

  auto tileBox = [&](const Code &code, ShapedType type, int64_t row,
                     int64_t column) -> FailureOr<Tile> {
    SmallVector<Attribute> point{builder.getIndexAttr(row), builder.getIndexAttr(column)};
    SmallVector<Attribute> lower;
    SmallVector<Attribute> upper;
    bool poison = false;

    if (failed(code.maps.begins.constantFold(point, lower, &poison)) || poison ||
        failed(code.maps.ends.constantFold(point, upper, &poison)) || poison)
      return copy.emitOpError("requires evaluable tile slicing maps");

    Tile box;

    for (unsigned axis = 0; axis < rank; ++axis) {
      int64_t begin = cast<IntegerAttr>(lower[axis]).getInt();
      int64_t end = cast<IntegerAttr>(upper[axis]).getInt();

      if (begin < 0 || end < begin || end >= type.getDimSize(axis))
        return copy.emitOpError("requires positive tile boxes inside their storage shapes");

      box.lo.push_back(begin);
      box.hi.push_back(end);
    }

    return box;
  };

  auto addressMap = [&](ArrayRef<int64_t> shape) -> FailureOr<AffineMapAttr> {
    SmallVector<int64_t> strides(rank);
    int64_t stride = 1;

    for (unsigned axis = rank; axis > 0; --axis) {
      strides[axis - 1] = stride;

      if (llvm::MulOverflow(stride, shape[axis - 1], stride))
        return copy.emitOpError("tile storage element count exceeds signed 64 bit range");
    }

    AffineExpr address = builder.getAffineConstantExpr(0);

    for (unsigned axis = 0; axis < rank; ++axis)
      address = address + builder.getAffineDimExpr(axis) * strides[axis];

    return AffineMapAttr::get(AffineMap::get(rank, 0, address));
  };

  SmallVector<Attribute> slices;
  auto bytes = builder.getI32IntegerAttr(element.getIntOrFloatBitWidth() / 8);

  for (int64_t row = 0; row < inputSlicing->domain[0]; ++row) {
    for (int64_t column = 0; column < inputSlicing->domain[1]; ++column) {
      auto inputBox = tileBox(*inputSlicing, inputType, row, column);
      auto outputBox = tileBox(*outputSlicing, storageType, row, column);

      if (failed(inputBox) || failed(outputBox))
        return failure();

      SmallVector<int64_t> inputShape;
      SmallVector<int64_t> outputShape;
      SmallVector<Attribute> domain;

      for (unsigned axis = 0; axis < rank; ++axis) {
        if (axis + 1 != rank && (inputBox->lo[axis] != outputBox->lo[axis] ||
                                 inputBox->hi[axis] != outputBox->hi[axis]))
          return copy.emitOpError("requires aligned spatial input and output tile boxes");

        int64_t extent = inputBox->hi[axis] - inputBox->lo[axis] + 1;

        if (!llvm::isInt<32>(extent))
          return copy.emitOpError("requires local copy domains representable as i32 extents");

        inputShape.push_back(extent);
        outputShape.push_back(outputBox->hi[axis] - outputBox->lo[axis] + 1);
        domain.push_back(builder.getI32IntegerAttr(extent));
      }

      if (inputBox->lo.back() != 0 || outputBox->lo.back() != 0 ||
          inputShape.back() != inputType.getShape().back() ||
          outputShape.back() != storageType.getShape().back())
        return copy.emitOpError("requires each tile to own every channel");

      auto readMap = addressMap(inputShape);
      auto writeMap = addressMap(outputShape);

      if (failed(readMap) || failed(writeMap))
        return failure();

      auto id =
          builder.getI32ArrayAttr({static_cast<int32_t>(row), static_cast<int32_t>(column), 0});
      auto localDomain = builder.getArrayAttr(domain);
      slices.push_back(NarrowToNarrowSliceAttr::get(builder.getContext(), id, localDomain, *readMap,
                                                    bytes, localDomain, *writeMap, bytes));
    }
  }

  auto shardId = builder.getArrayAttr({builder.getArrayAttr({})});
  auto shard =
      NarrowToNarrowShardAttr::get(builder.getContext(), shardId, builder.getArrayAttr(slices));
  auto narrow = NarrowToNarrowOp::create(builder, copy.getLoc(), copy.getType(), copy.getInput(),
                                         copy.getDestination(), builder.getArrayAttr({shard}));
  copy.replaceAllUsesWith(narrow.getOutput());
  copy.erase();
  return success();
}

class LowerSynchronizedOpsToTensorOpPass
    : public PassWrapper<LowerSynchronizedOpsToTensorOpPass,
                         OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      LowerSynchronizedOpsToTensorOpPass)

  StringRef getArgument() const final {
    return "darwinn-lower-synchronized-ops-to-tensor-op";
  }

  StringRef getDescription() const final {
    return "Lower synchronized compute and copies to tensor operations";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<DarwinnDialect>();
  }

  void runOnOperation() final {
    SmallVector<Operation *> synchronized;
    getOperation().walk([&](Operation *operation) {
      if (isa<SynchronizedCopyOpOp, SynchronizedComputeOpOp, SynchronizedUnaryComputeOpOp,
              CommunicatedLocalCopyOp>(operation))
        synchronized.push_back(operation);
    });
    for (Operation *operation : synchronized) {
      if (auto copy = dyn_cast<SynchronizedCopyOpOp>(operation))
        lower(copy);
      else if (auto compute = dyn_cast<SynchronizedComputeOpOp>(operation))
        lower(compute);
      else if (auto copy = dyn_cast<CommunicatedLocalCopyOp>(operation)) {
        if (failed(lower(copy)))
          return signalPassFailure();
      } else
        lower(cast<SynchronizedUnaryComputeOpOp>(operation));
    }
  }
};

}

std::unique_ptr<Pass>
mlir::darwinn::createLowerSynchronizedOpsToTensorOpPass() {
  return std::make_unique<LowerSynchronizedOpsToTensorOpPass>();
}

void mlir::darwinn::registerLowerSynchronizedOpsToTensorOpPass() {
  PassRegistration<LowerSynchronizedOpsToTensorOpPass>();
}
