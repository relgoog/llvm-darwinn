#include "SlicingModel.h"
#include "mlir/Dialect/Darwinn/IR/DarwinnOps.h"
#include "mlir/Dialect/Darwinn/Transforms/DistributedPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/MapVector.h"

using namespace mlir;
using namespace mlir::darwinn;
using namespace mlir::darwinn::slicing;

namespace {

constexpr StringLiteral kSlicingAttributes[] = {
    "slicing_begins", "slicing_domain", "slicing_ends"};

struct Slicing {
  AffineMapAttr begins;
  ArrayAttr domain;
  AffineMapAttr ends;

  bool operator==(const Slicing &other) const {
    return begins == other.begins && domain == other.domain &&
           ends == other.ends;
  }
};

std::optional<Slicing> slicingOf(Operation *operation) {
  auto begins = operation->getAttrOfType<AffineMapAttr>("slicing_begins");
  auto domain = operation->getAttrOfType<ArrayAttr>("slicing_domain");
  auto ends = operation->getAttrOfType<AffineMapAttr>("slicing_ends");
  if (!begins || !domain || !ends)
    return std::nullopt;
  return Slicing{begins, domain, ends};
}

std::optional<Slicing> slicingOf(Value value) {
  Operation *producer = value.getDefiningOp();
  if (!producer)
    return std::nullopt;
  if (isa<StaticComputeOpOp, StaticUnaryComputeOpOp>(producer)) {
    auto destination = producer->getOperands().back();
    if (auto view = destination.getDefiningOp<DistributedCreateViewOp>())
      producer = view.getInput().getDefiningOp();
    if (!producer)
      return std::nullopt;
  }
  return slicingOf(producer);
}

bool isNoOp(RedistributeOp redistribute) {
  Value input = redistribute.getInput();
  if (redistribute.getMappingAttr() || redistribute.getDestination() ||
      input.getType() != redistribute.getType() ||
      memorySpaceOf(input) != DistributedMemorySpace::TileMemory)
    return false;
  std::optional<Slicing> source = slicingOf(input);
  return source && source == slicingOf(redistribute.getOperation());
}

LogicalResult mergeSharedSlicing(Operation *source) {
  std::optional<Slicing> slicing = slicingOf(source);
  if (!slicing || slicing->domain.size() != 3)
    return source->emitOpError("shares a tile slicing without a thread axis");
  auto domain =
      llvm::to_vector(llvm::map_range(slicing->domain, [](Attribute extent) {
        return static_cast<int32_t>(cast<IntegerAttr>(extent).getInt());
      }));
  Builder builder(source->getContext());
  source->setAttr("slicing_begins", AffineMapAttr::get(mergeThreads(
                                        slicing->begins.getValue(), 0)));
  source->setAttr("slicing_domain",
                  builder.getI32ArrayAttr({domain[0], domain[1]}));
  source->setAttr("slicing_ends",
                  AffineMapAttr::get(
                      mergeThreads(slicing->ends.getValue(), domain[2] - 1)));
  return success();
}

class RedistributeOptimizationPass
    : public PassWrapper<RedistributeOptimizationPass,
                         OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RedistributeOptimizationPass)

  StringRef getArgument() const final {
    return "darwinn-redistribute-optimization";
  }

  StringRef getDescription() const final {
    return "Replace redistributes that leave every tile slice in place with "
           "tile local tensor reads";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<DarwinnDialect>();
  }

  void runOnOperation() final {
    llvm::MapVector<Value, SmallVector<RedistributeOp>> readers;
    getOperation().walk([&](RedistributeOp redistribute) {
      if (isNoOp(redistribute))
        readers[redistribute.getInput()].push_back(redistribute);
    });
    for (auto &[source, copies] : readers) {
      for (RedistributeOp copy : copies) {
        OpBuilder builder(copy);
        auto read =
            GetTensorOp::create(builder, copy.getLoc(), copy.getType(), source);
        for (StringRef name : kSlicingAttributes)
          read->setDiscardableAttr(name, copy->getAttr(name));
        copy.replaceAllUsesWith(read.getResult());
        copy.erase();
      }
      if (copies.size() < 2)
        continue;
      Operation *producer = source.getDefiningOp();
      if (!isa<RedistributeOp>(producer)) {
        producer->emitOpError(
            "is read in place by several tensors, which has no known slicing");
        return signalPassFailure();
      }
      if (failed(mergeSharedSlicing(producer)))
        return signalPassFailure();
    }
  }
};

}

std::unique_ptr<Pass> mlir::darwinn::createRedistributeOptimizationPass() {
  return std::make_unique<RedistributeOptimizationPass>();
}

void mlir::darwinn::registerRedistributeOptimizationPass() {
  PassRegistration<RedistributeOptimizationPass>();
}
