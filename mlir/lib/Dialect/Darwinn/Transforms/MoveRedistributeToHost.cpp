#include "mlir/Dialect/Darwinn/IR/DarwinnOps.h"
#include "mlir/Dialect/Darwinn/Transforms/DistributedPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::darwinn;

namespace {

SmallVector<int64_t> nonUnitDims(ArrayRef<int64_t> shape) {
  SmallVector<int64_t> result;
  for (int64_t size : shape)
    if (size != 1)
      result.push_back(size);
  return result;
}

// The SDK moves a reshape to the host when either row major index map
// between the shapes needs a mod, which happens exactly when a non unit
// dim is split or merged.
bool needsHostReshape(DistributedTensorType from, DistributedTensorType to) {
  return from.getNumElements() == to.getNumElements() &&
         nonUnitDims(from.getShape()) != nonUnitDims(to.getShape());
}

Operation *slicedSource(Value value) {
  Operation *op = value.getDefiningOp();
  while (op && !op->hasAttr("slicing_begins")) {
    Value next;
    if (auto compute = dyn_cast<StaticComputeOpOp>(op))
      next = compute.getDestination();
    else if (auto unary = dyn_cast<StaticUnaryComputeOpOp>(op))
      next = unary.getDestination();
    else if (op->getNumOperands() == 1)
      next = op->getOperand(0);
    op = next ? next.getDefiningOp() : nullptr;
  }
  return op;
}

// The SDK keeps the reshape on the tiles when it is local to each tile: a
// computed source and the result share the slicing of the leading unchanged
// dims, and every dim the reshape splits or merges is held whole.
bool tileLocal(RedistributeOp redistribute, ArrayRef<int64_t> from,
               ArrayRef<int64_t> to) {
  Operation *source = slicedSource(redistribute.getInput());
  if (!source || isa<RedistributeOp>(source) ||
      source->getAttr("slicing_domain") != redistribute.getSlicingDomainAttr())
    return false;
  size_t prefix = 0;
  while (prefix < from.size() && prefix < to.size() &&
         from[prefix] == to[prefix])
    ++prefix;
  for (StringRef name : {"slicing_begins", "slicing_ends"}) {
    AffineMap mine =
        cast<AffineMapAttr>(redistribute->getAttr(name)).getValue();
    AffineMap theirs = cast<AffineMapAttr>(source->getAttr(name)).getValue();
    for (size_t dim = 0; dim < prefix; ++dim)
      if (mine.getResult(dim) != theirs.getResult(dim))
        return false;
    for (size_t dim = prefix; dim < to.size(); ++dim) {
      auto bound = dyn_cast<AffineConstantExpr>(mine.getResult(dim));
      if (!bound ||
          bound.getValue() != (name == "slicing_begins" ? 0 : to[dim] - 1))
        return false;
    }
  }
  return true;
}

MappingAttr identityMapping(MLIRContext *context, unsigned rank) {
  auto identity =
      AffineMapAttr::get(AffineMap::getMultiDimIdentityMap(rank, context));
  return MappingAttr::get(context, identity, identity);
}

DistributedTensorType onHost(DistributedTensorType type) {
  return DistributedTensorType::get(type.getContext(), type.getShape(),
                                    type.getElementType(),
                                    DistributedMemorySpace::HostMemory);
}

void moveToHost(RedistributeOp redistribute) {
  MLIRContext *context = redistribute.getContext();
  OpBuilder builder(redistribute);
  Location location = redistribute.getLoc();
  auto sourceType =
      cast<DistributedTensorType>(redistribute.getInput().getType());
  auto resultType = cast<DistributedTensorType>(redistribute.getType());
  unsigned domainRank = redistribute.getSlicingDomain().size();

  Value hosted = redistribute.getInput();
  if (sourceType.getMemorySpace() != DistributedMemorySpace::HostMemory) {
    SmallVector<AffineExpr> begins, ends;
    for (int64_t size : sourceType.getShape()) {
      begins.push_back(builder.getAffineConstantExpr(0));
      ends.push_back(builder.getAffineConstantExpr(size - 1));
    }
    hosted = RedistributeOp::create(
        builder, location, onHost(sourceType), hosted, Value{},
        identityMapping(context, sourceType.getRank()),
        AffineMap::get(domainRank, 0, begins, context),
        builder.getI32ArrayAttr(SmallVector<int32_t>(domainRank, 1)),
        AffineMap::get(domainRank, 0, ends, context));
  }

  Value reshaped =
      ReshapeOpOp::create(builder, location, onHost(resultType), hosted,
                          AffineMapAttr{}, AffineMapAttr{});
  if (resultType.getMemorySpace() != DistributedMemorySpace::HostMemory) {
    MappingAttr mapping = redistribute.getMappingAttr();
    if (!mapping)
      mapping = identityMapping(context, resultType.getRank());
    reshaped = RedistributeOp::create(
        builder, location, resultType, reshaped, Value{}, mapping,
        redistribute.getSlicingBeginsAttr(),
        redistribute.getSlicingDomainAttr(), redistribute.getSlicingEndsAttr());
  }
  redistribute.replaceAllUsesWith(reshaped);
  redistribute.erase();
}

class MoveRedistributeToHostPass
    : public PassWrapper<MoveRedistributeToHostPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MoveRedistributeToHostPass)

  StringRef getArgument() const final {
    return "darwinn-move-redistribute-to-host";
  }

  StringRef getDescription() const final {
    return "Perform element reordering reshapes on the host";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<DarwinnDialect>();
  }

  void runOnOperation() final {
    SmallVector<RedistributeOp> moves;
    getOperation().walk([&](RedistributeOp redistribute) {
      auto from =
          dyn_cast<DistributedTensorType>(redistribute.getInput().getType());
      auto to = dyn_cast<DistributedTensorType>(redistribute.getType());
      if (from && to && !redistribute.getDestination() &&
          needsHostReshape(from, to) &&
          (from.getMemorySpace() == DistributedMemorySpace::HostMemory ||
           to.getMemorySpace() == DistributedMemorySpace::HostMemory ||
           !tileLocal(redistribute, from.getShape(), to.getShape())))
        moves.push_back(redistribute);
    });
    for (RedistributeOp redistribute : moves)
      moveToHost(redistribute);
  }
};

}

std::unique_ptr<Pass> mlir::darwinn::createMoveRedistributeToHostPass() {
  return std::make_unique<MoveRedistributeToHostPass>();
}

void mlir::darwinn::registerMoveRedistributeToHostPass() {
  PassRegistration<MoveRedistributeToHostPass>();
}
