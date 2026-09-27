#include "SlicingModel.h"
#include "mlir/Dialect/Darwinn/IR/DarwinnOps.h"
#include "mlir/Dialect/Darwinn/Transforms/DistributedPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::darwinn;
using namespace mlir::darwinn::slicing;

namespace {

bool isIdentity(MappingAttr mapping) {
  return !mapping ||
         (mapping.getForwardIndexTransformation().getValue().isIdentity() &&
          mapping.getReverseIndexTransformation().getValue().isIdentity());
}

bool movesOnlyUnitDims(ArrayRef<int64_t> from, ArrayRef<int64_t> to) {
  auto nonUnit = [](ArrayRef<int64_t> shape) {
    SmallVector<int64_t> result;
    for (int64_t size : shape)
      if (size != 1)
        result.push_back(size);
    return result;
  };
  return from != to && nonUnit(from) == nonUnit(to);
}

void bypassTileCopyBeforeExport(RedistributeOp copy) {
  Value input = copy.getInput();
  if (copy.getDestination() || !copy->hasOneUse() ||
      !isIdentity(copy.getMappingAttr()) || input.getType() != copy.getType() ||
      memorySpaceOf(input) != DistributedMemorySpace::TileMemory)
    return;
  auto exported = dyn_cast<RedistributeOp>(*copy->getUsers().begin());
  if (!exported ||
      memorySpaceOf(exported.getOutput()) != DistributedMemorySpace::HostMemory)
    return;
  exported.getInputMutable().assign(input);
  copy.erase();
}

void reshapeOnHost(RedistributeOp read) {
  Value input = read.getInput();
  auto from = dyn_cast<DistributedTensorType>(input.getType());
  auto to = dyn_cast<DistributedTensorType>(read.getType());
  if (!from || !to || read.getMappingAttr() || read.getDestination() ||
      from.getMemorySpace() != DistributedMemorySpace::HostMemory ||
      !movesOnlyUnitDims(from.getShape(), to.getShape()))
    return;
  OpBuilder builder(read);
  MLIRContext *context = builder.getContext();
  auto hostType =
      DistributedTensorType::get(context, to.getShape(), to.getElementType(),
                                 DistributedMemorySpace::HostMemory);
  auto reshape = ReshapeOpOp::create(
      builder, read.getLoc(), hostType, input,
      AffineMapAttr::get(
          unitReshapeMap(from.getShape(), to.getShape(), context)),
      AffineMapAttr::get(
          unitReshapeMap(to.getShape(), from.getShape(), context)));
  read.getInputMutable().assign(reshape);
}

class TransferCanonicalizationPass
    : public PassWrapper<TransferCanonicalizationPass,
                         OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TransferCanonicalizationPass)

  StringRef getArgument() const final {
    return "darwinn-canonicalize-transfers";
  }

  StringRef getDescription() const final {
    return "Fold tile copies into host exports and move unit dimension "
           "reshapes of host reads onto the host";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<DarwinnDialect>();
  }

  void runOnOperation() final {
    SmallVector<RedistributeOp> redistributes;
    getOperation().walk([&](RedistributeOp redistribute) {
      redistributes.push_back(redistribute);
    });
    for (RedistributeOp redistribute : redistributes)
      bypassTileCopyBeforeExport(redistribute);
    getOperation().walk(
        [&](RedistributeOp redistribute) { reshapeOnHost(redistribute); });
  }
};

}

std::unique_ptr<Pass> mlir::darwinn::createTransferCanonicalizationPass() {
  return std::make_unique<TransferCanonicalizationPass>();
}

void mlir::darwinn::registerTransferCanonicalizationPass() {
  PassRegistration<TransferCanonicalizationPass>();
}
