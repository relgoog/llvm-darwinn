#include "mlir/Dialect/Darwinn/IR/DarwinnOps.h"
#include "mlir/Dialect/Darwinn/Transforms/DistributedPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::darwinn;

namespace {

bool streamsParameters(std::optional<InnerOperationKind> inner) {
  return inner == InnerOperationKind::Vmc ||
         inner == InnerOperationKind::Stencil;
}

LogicalResult convert(StaticComputeOpOp compute) {
  OpBuilder builder(compute);
  MLIRContext *context = builder.getContext();
  std::optional<InnerOperationKind> inner =
      compute.getCompute().getInnerOperation();
  Value rhs = compute.getRhs();
  BoolAttr streamingInput0, streamingInput1, streamingOutput;

  if (streamsParameters(inner)) {
    auto traversal =
        rhs.getDefiningOp()
            ? rhs.getDefiningOp()->getAttrOfType<AffineMapAttr>("traversal")
            : AffineMapAttr{};
    if (!traversal)
      return compute.emitOpError("streams a parameter without a traversal");
    auto view = cast<DistributedViewType>(rhs.getType());
    auto registers = DistributedViewType::get(
        context, view.getShape(), view.getElementType(),
        DistributedMemorySpace::TileRegisters);
    rhs = StreamingCopyOpOp::create(
        builder, compute.getLoc(), registers, rhs, traversal, AffineMapAttr{},
        MemSpaceAttr::get(context, MemSpaceKind::TileRegisters),
        builder.getBoolAttr(false), builder.getBoolAttr(true));
    streamingInput0 = builder.getBoolAttr(false);
    streamingInput1 = builder.getBoolAttr(true);
    streamingOutput = builder.getBoolAttr(false);
  } else if (inner != InnerOperationKind::Elementwise) {
    return compute.emitOpError("has no recovered streaming form");
  }

  auto streaming = StreamingComputeOpOp::create(
      builder, compute.getLoc(), compute.getType(), compute.getLhs(), rhs,
      compute.getDestination(), compute.getAuxiliaryTensors(),
      compute.getComputeAttr(), compute.getTraversalAttr(),
      AffineMapAttr::get(AffineMap::get(context)),
      compute.getAuxiliaryTensorTypesAttr(), streamingInput0, streamingInput1,
      streamingOutput, compute.getCustomTilingAttr(), compute.getDtcInfoAttr(),
      compute.getVexInfoAttr());
  compute.replaceAllUsesWith(streaming.getOutput());
  compute.erase();
  return success();
}

void convert(StaticUnaryComputeOpOp compute) {
  OpBuilder builder(compute);
  auto streaming = StreamingUnaryComputeOpOp::create(
      builder, compute.getLoc(), compute.getType(), compute.getInput(),
      compute.getDestination(), compute.getAuxiliaryTensors(),
      compute.getComputeAttr(), compute.getTraversalAttr(),
      compute.getAuxiliaryTensorTypesAttr(), compute.getCustomTilingAttr(),
      compute.getCustomConstraintsAttr(), compute.getVexInfoAttr());
  compute.replaceAllUsesWith(streaming.getOutput());
  compute.erase();
}

class StaticToStreamingPass
    : public PassWrapper<StaticToStreamingPass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StaticToStreamingPass)

  StringRef getArgument() const final { return "darwinn-static-to-streaming"; }

  StringRef getDescription() const final {
    return "Stream compute parameters from tile memory into tile registers";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<DarwinnDialect>();
  }

  void runOnOperation() final {
    SmallVector<Operation *> computes;
    getOperation().walk([&](Operation *operation) {
      if (isa<StaticComputeOpOp, StaticUnaryComputeOpOp>(operation))
        computes.push_back(operation);
    });
    for (Operation *operation : computes) {
      if (auto unary = dyn_cast<StaticUnaryComputeOpOp>(operation)) {
        convert(unary);
        continue;
      }
      if (failed(convert(cast<StaticComputeOpOp>(operation))))
        return signalPassFailure();
    }
  }
};

}

std::unique_ptr<Pass> mlir::darwinn::createStaticToStreamingPass() {
  return std::make_unique<StaticToStreamingPass>();
}

void mlir::darwinn::registerStaticToStreamingPass() {
  PassRegistration<StaticToStreamingPass>();
}
