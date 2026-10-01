#include "mlir/Dialect/Darwinn/IR/DarwinnOps.h"
#include "mlir/Dialect/Darwinn/Transforms/DistributedPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::darwinn;

namespace {

constexpr StringLiteral kSlicingAttributes[] = {
    "slicing_begins", "slicing_domain", "slicing_ends"};

bool copiesInPlace(RedistributeOp redistribute) {
  auto fill = redistribute.getInput().getDefiningOp<FillOp>();
  if (!fill || !fill.getConstType() || redistribute.getMappingAttr() ||
      redistribute.getDestination() || fill.getType() != redistribute.getType())
    return false;
  return llvm::all_of(kSlicingAttributes, [&](StringRef name) {
    Attribute attribute = fill->getAttr(name);
    return attribute && attribute == redistribute->getAttr(name);
  });
}

class RemoveParameterRedistributesPass
    : public PassWrapper<RemoveParameterRedistributesPass,
                         OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RemoveParameterRedistributesPass)

  StringRef getArgument() const final {
    return "darwinn-remove-parameter-redistributes";
  }

  StringRef getDescription() const final {
    return "Read parameter fills directly where a redistribute would copy "
           "every tile slice in place";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<DarwinnDialect>();
  }

  void runOnOperation() final {
    SmallVector<RedistributeOp> copies;
    getOperation().walk([&](RedistributeOp redistribute) {
      if (copiesInPlace(redistribute))
        copies.push_back(redistribute);
    });
    for (RedistributeOp copy : copies) {
      copy.replaceAllUsesWith(copy.getInput());
      copy.erase();
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::darwinn::createRemoveParameterRedistributesPass() {
  return std::make_unique<RemoveParameterRedistributesPass>();
}

void mlir::darwinn::registerRemoveParameterRedistributesPass() {
  PassRegistration<RemoveParameterRedistributesPass>();
}
