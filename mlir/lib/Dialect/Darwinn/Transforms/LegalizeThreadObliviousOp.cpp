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

constexpr StringLiteral kBegins = "slicing_begins";
constexpr StringLiteral kDomain = "slicing_domain";
constexpr StringLiteral kEnds = "slicing_ends";

bool isThreadSliced(Operation *operation) {
  auto domain = operation->getAttrOfType<ArrayAttr>(kDomain);
  return domain && domain.size() == 3;
}

void copySlicing(Operation *from, Operation *to) {
  for (StringRef name : {kBegins, kDomain, kEnds})
    to->setDiscardableAttr(name, from->getAttr(name));
}

void mergeSlicing(Operation *operation) {
  auto domain = operation->getAttrOfType<ArrayAttr>(kDomain);
  int64_t threads = cast<IntegerAttr>(domain[2]).getInt();
  Builder builder(operation->getContext());
  operation->setAttr(
      kBegins,
      AffineMapAttr::get(mergeThreads(
          operation->getAttrOfType<AffineMapAttr>(kBegins).getValue(), 0)));
  operation->setAttr(
      kEnds, AffineMapAttr::get(mergeThreads(
                 operation->getAttrOfType<AffineMapAttr>(kEnds).getValue(),
                 threads - 1)));
  operation->setAttr(kDomain, builder.getArrayAttr({domain[0], domain[1]}));
}

FailureOr<Operation *> threadSlicingOwner(Value value) {
  Operation *producer = value.getDefiningOp();
  if (!producer)
    return failure();
  if (auto compute = dyn_cast<StreamingComputeOpOp>(producer))
    return threadSlicingOwner(compute.getDestination());
  if (auto unary = dyn_cast<StreamingUnaryComputeOpOp>(producer))
    return threadSlicingOwner(unary.getDestination());
  if (auto view = dyn_cast<DistributedCreateViewOp>(producer))
    return threadSlicingOwner(view.getInput());
  if (auto reshape = dyn_cast<ReshapeOpOp>(producer))
    return threadSlicingOwner(reshape.getInput());
  if (producer->getAttrOfType<ArrayAttr>(kDomain))
    return producer;
  return failure();
}

void restoreThreads(Operation *operation) {
  OpBuilder builder(operation->getContext());
  builder.setInsertionPointAfter(operation);
  Value merged = operation->getResult(0);
  auto read = GetTensorOp::create(builder, operation->getLoc(),
                                  merged.getType(), merged);
  copySlicing(operation, read);
  merged.replaceAllUsesExcept(read.getResult(), read);
  mergeSlicing(operation);
}

LogicalResult readMerged(RedistributeOp redistribute) {
  FailureOr<Operation *> owner = threadSlicingOwner(redistribute.getInput());
  if (failed(owner))
    return redistribute.emitOpError("reads a value without a tile slicing");
  if (!isThreadSliced(*owner))
    return success();
  OpBuilder builder(redistribute);
  Value input = redistribute.getInput();
  auto read = GetTensorOp::create(builder, redistribute.getLoc(),
                                  input.getType(), input);
  copySlicing(*owner, read);
  mergeSlicing(read);
  redistribute.getInputMutable().assign(read.getResult());
  return success();
}

class LegalizeThreadObliviousOpPass
    : public PassWrapper<LegalizeThreadObliviousOpPass,
                         OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LegalizeThreadObliviousOpPass)

  StringRef getArgument() const final {
    return "darwinn-legalize-thread-oblivious-op";
  }

  StringRef getDescription() const final {
    return "Merge the thread axis out of transfer and fill slicing and read "
           "thread slices through get_tensor";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<DarwinnDialect>();
  }

  void runOnOperation() final {
    SmallVector<Operation *> oblivious;
    SmallVector<RedistributeOp> transfers;
    getOperation().walk([&](Operation *operation) {
      if (auto redistribute = dyn_cast<RedistributeOp>(operation))
        transfers.push_back(redistribute);
      if (isa<RedistributeOp, FillOp>(operation) && isThreadSliced(operation))
        oblivious.push_back(operation);
    });
    for (Operation *operation : oblivious)
      restoreThreads(operation);
    for (RedistributeOp redistribute : transfers)
      if (memorySpaceOf(redistribute.getInput()) !=
              DistributedMemorySpace::HostMemory &&
          failed(readMerged(redistribute)))
        return signalPassFailure();
  }
};

}

std::unique_ptr<Pass> mlir::darwinn::createLegalizeThreadObliviousOpPass() {
  return std::make_unique<LegalizeThreadObliviousOpPass>();
}

void mlir::darwinn::registerLegalizeThreadObliviousOpPass() {
  PassRegistration<LegalizeThreadObliviousOpPass>();
}
