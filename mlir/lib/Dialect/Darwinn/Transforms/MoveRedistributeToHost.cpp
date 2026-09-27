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
          needsHostReshape(from, to))
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
