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
  AffineMap ends = mergeThreads(
      operation->getAttrOfType<AffineMapAttr>(kEnds).getValue(), threads - 1);
  if (auto shaped = dyn_cast<ShapedType>(operation->getResult(0).getType())) {
    SmallVector<AffineExpr> clamped(ends.getResults());
    for (auto [dim, end] : llvm::enumerate(clamped))
      if (auto constant = dyn_cast<AffineConstantExpr>(end);
          constant && constant.getValue() >= shaped.getDimSize(dim))
        end = builder.getAffineConstantExpr(shaped.getDimSize(dim) - 1);
    ends = AffineMap::get(ends.getNumDims(), 0, clamped, builder.getContext());
  }
  operation->setAttr(kEnds, AffineMapAttr::get(ends));
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

LogicalResult readMerged(OpOperand &operand) {
  Operation *operation = operand.getOwner();
  FailureOr<Operation *> owner = threadSlicingOwner(operand.get());
  if (failed(owner))
    return operation->emitOpError("reads a value without a tile slicing");
  if (!isThreadSliced(*owner))
    return success();

  OpBuilder builder(operation);
  Value input = operand.get();

  if (auto read = input.getDefiningOp<GetTensorOp>();
      read && isa<MathJoinOp>(operation)) {
    if (read->hasOneUse()) {
      mergeSlicing(read);
      return success();
    }

    input = read.getInput();
  }

  auto read =
      GetTensorOp::create(builder, operation->getLoc(), input.getType(), input);
  copySlicing(*owner, read);
  mergeSlicing(read);
  operand.set(read.getResult());
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
    SmallVector<MathJoinOp> joins;
    getOperation().walk([&](Operation *operation) {
      if (auto redistribute = dyn_cast<RedistributeOp>(operation))
        transfers.push_back(redistribute);
      if (auto join = dyn_cast<MathJoinOp>(operation))
        joins.push_back(join);
      if (isa<RedistributeOp, FillOp, MathJoinOp>(operation) &&
          isThreadSliced(operation))
        oblivious.push_back(operation);
    });
    for (Operation *operation : oblivious) {
      if (isa<MathJoinOp>(operation) &&
          llvm::all_of(operation->getUsers(),
                       [](Operation *user) { return isa<GetTensorOp>(user); }))
        mergeSlicing(operation);
      else
        restoreThreads(operation);
    }

    for (RedistributeOp redistribute : transfers)
      if (memorySpaceOf(redistribute.getInput()) !=
              DistributedMemorySpace::HostMemory &&
          failed(readMerged(redistribute->getOpOperand(0))))
        return signalPassFailure();

    for (MathJoinOp join : joins)
      for (OpOperand &operand : join->getOpOperands())
        if (failed(readMerged(operand)))
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
