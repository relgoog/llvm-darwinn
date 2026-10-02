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
    if (domain != other.domain)
      return false;
    if (begins == other.begins && ends == other.ends)
      return true;
    SmallVector<int64_t> point(domain.size(), 0);
    while (true) {
      if (begins.getValue().compose(point) !=
              other.begins.getValue().compose(point) ||
          ends.getValue().compose(point) !=
              other.ends.getValue().compose(point))
        return false;
      size_t axis = 0;
      for (; axis < point.size(); ++axis) {
        if (++point[axis] < cast<IntegerAttr>(domain[axis]).getInt())
          break;
        point[axis] = 0;
      }
      if (axis == point.size())
        return true;
    }
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

std::optional<Slicing> slicingOf(Value value);

// A tile reshape keeps the leading dims its source shares and holds the rest
// whole, so its slicing follows from the source's when those dims were whole.
std::optional<Slicing> reshapedSlicing(ReshapeOpOp reshape) {
  std::optional<Slicing> source = slicingOf(reshape.getInput());
  if (!source)
    return std::nullopt;
  ArrayRef<int64_t> from =
      cast<DistributedTensorType>(reshape.getInput().getType()).getShape();
  ArrayRef<int64_t> to =
      cast<DistributedTensorType>(reshape.getOutput().getType()).getShape();
  size_t prefix = 0;
  while (prefix < from.size() && prefix < to.size() &&
         from[prefix] == to[prefix])
    ++prefix;
  AffineMap begins = source->begins.getValue(), ends = source->ends.getValue();
  for (size_t dim = prefix; dim < from.size(); ++dim) {
    auto begin = dyn_cast<AffineConstantExpr>(begins.getResult(dim));
    auto end = dyn_cast<AffineConstantExpr>(ends.getResult(dim));
    if (!begin || !end || begin.getValue() != 0 ||
        end.getValue() != from[dim] - 1)
      return std::nullopt;
  }
  MLIRContext *context = reshape.getContext();
  SmallVector<AffineExpr> newBegins(begins.getResults().take_front(prefix));
  SmallVector<AffineExpr> newEnds(ends.getResults().take_front(prefix));
  for (size_t dim = prefix; dim < to.size(); ++dim) {
    newBegins.push_back(getAffineConstantExpr(0, context));
    newEnds.push_back(getAffineConstantExpr(to[dim] - 1, context));
  }
  return Slicing{AffineMapAttr::get(AffineMap::get(begins.getNumDims(), 0,
                                                   newBegins, context)),
                 source->domain,
                 AffineMapAttr::get(
                     AffineMap::get(ends.getNumDims(), 0, newEnds, context))};
}

std::optional<Slicing> slicingOf(Value value) {
  Operation *producer = value.getDefiningOp();
  if (!producer)
    return std::nullopt;
  if (auto reshape = dyn_cast<ReshapeOpOp>(producer))
    return reshapedSlicing(reshape);
  Value destination;
  if (auto compute = dyn_cast<StaticComputeOpOp>(producer))
    destination = compute.getDestination();
  else if (auto unary = dyn_cast<StaticUnaryComputeOpOp>(producer))
    destination = unary.getDestination();
  if (destination) {
    if (auto view = destination.getDefiningOp<DistributedCreateViewOp>())
      producer = view.getInput().getDefiningOp();
    if (!producer)
      return std::nullopt;
  }
  return slicingOf(producer);
}

bool sameTiles(const Slicing &left, const Slicing &right,
               ArrayRef<int64_t> shape) {
  if (left.domain.size() != 3 || right.domain.size() != 3 ||
      left.domain[0] != right.domain[0] || left.domain[1] != right.domain[1])
    return false;
  int64_t extents[3];
  for (auto [axis, extent] : llvm::enumerate(left.domain))
    extents[axis] = cast<IntegerAttr>(extent).getInt();
  auto box = [&](const Slicing &slicing, int64_t row, int64_t column) {
    SmallVector<int64_t> low(shape.size(), INT64_MAX);
    SmallVector<int64_t> high(shape.size(), INT64_MIN);
    int64_t threads = cast<IntegerAttr>(slicing.domain[2]).getInt();
    for (int64_t thread = 0; thread < threads; ++thread) {
      SmallVector<int64_t> begins =
          slicing.begins.getValue().compose({row, column, thread});
      SmallVector<int64_t> ends =
          slicing.ends.getValue().compose({row, column, thread});
      for (size_t dim = 0; dim < shape.size(); ++dim) {
        low[dim] = std::min(low[dim], std::max<int64_t>(begins[dim], 0));
        high[dim] = std::max(high[dim], std::min(ends[dim], shape[dim] - 1));
      }
    }
    return std::make_pair(low, high);
  };
  for (int64_t row = 0; row < extents[0]; ++row)
    for (int64_t column = 0; column < extents[1]; ++column)
      if (box(left, row, column) != box(right, row, column))
        return false;
  return true;
}

bool isNoOp(RedistributeOp redistribute) {
  Value input = redistribute.getInput();
  MappingAttr mapping = redistribute.getMappingAttr();
  if ((mapping &&
       (!mapping.getForwardIndexTransformation().getValue().isIdentity() ||
        !mapping.getReverseIndexTransformation().getValue().isIdentity())) ||
      redistribute.getDestination() ||
      input.getType() != redistribute.getType() ||
      memorySpaceOf(input) != DistributedMemorySpace::TileMemory)
    return false;
  std::optional<Slicing> source = slicingOf(input);
  std::optional<Slicing> result = slicingOf(redistribute.getOperation());
  return source && result &&
         (source == result ||
          sameTiles(*source, *result,
                    cast<DistributedTensorType>(input.getType()).getShape()));
}

// A reshape that splits or merges only dims every tile holds whole, with the
// leading dims sliced as in its source, needs no data movement.
bool isLocalReshape(RedistributeOp redistribute) {
  Value input = redistribute.getInput();
  auto from = dyn_cast<DistributedTensorType>(input.getType());
  auto to = dyn_cast<DistributedTensorType>(redistribute.getType());
  if (!from || !to || redistribute.getMappingAttr() ||
      redistribute.getDestination() || from.getShape() == to.getShape() ||
      from.getNumElements() != to.getNumElements() ||
      memorySpaceOf(input) != DistributedMemorySpace::TileMemory ||
      to.getMemorySpace() != DistributedMemorySpace::TileMemory)
    return false;
  std::optional<Slicing> source = slicingOf(input);
  std::optional<Slicing> result = slicingOf(redistribute.getOperation());
  if (!source || !result || source->domain != result->domain ||
      source->domain.size() != 3)
    return false;
  ArrayRef<int64_t> fromShape = from.getShape(), toShape = to.getShape();
  size_t prefix = 0;
  while (prefix < fromShape.size() && prefix < toShape.size() &&
         fromShape[prefix] == toShape[prefix])
    ++prefix;
  auto whole = [](AffineMap begins, AffineMap ends, ArrayRef<int64_t> shape,
                  size_t first) {
    for (size_t dim = first; dim < shape.size(); ++dim) {
      auto begin = dyn_cast<AffineConstantExpr>(begins.getResult(dim));
      auto end = dyn_cast<AffineConstantExpr>(ends.getResult(dim));
      if (!begin || !end || begin.getValue() != 0 ||
          end.getValue() != shape[dim] - 1)
        return false;
    }
    return true;
  };
  for (size_t dim = 0; dim < prefix; ++dim)
    if (source->begins.getValue().getResult(dim) !=
            result->begins.getValue().getResult(dim) ||
        source->ends.getValue().getResult(dim) !=
            result->ends.getValue().getResult(dim))
      return false;
  return whole(source->begins.getValue(), source->ends.getValue(), fromShape,
               prefix) &&
         whole(result->begins.getValue(), result->ends.getValue(), toShape,
               prefix);
}

AffineMap reshapeMap(ArrayRef<int64_t> from, ArrayRef<int64_t> to,
                     MLIRContext *context) {
  size_t prefix = 0;
  while (prefix < from.size() && prefix < to.size() &&
         from[prefix] == to[prefix])
    ++prefix;
  SmallVector<AffineExpr> results;
  for (size_t dim = 0; dim < prefix; ++dim)
    results.push_back(from[dim] == 1 ? getAffineConstantExpr(0, context)
                                     : getAffineDimExpr(dim, context));
  AffineExpr linear = getAffineConstantExpr(0, context);
  for (size_t dim = prefix; dim < from.size(); ++dim)
    linear = linear * from[dim] + getAffineDimExpr(dim, context);
  int64_t stride = 1;
  SmallVector<AffineExpr> suffix;
  for (size_t dim = to.size(); dim-- > prefix;) {
    AffineExpr value = stride == 1 ? linear : linear.floorDiv(stride);
    suffix.push_back(dim == prefix ? value : value % to[dim]);
    stride *= to[dim];
  }
  results.append(suffix.rbegin(), suffix.rend());
  return AffineMap::get(from.size(), 0, results, context);
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
    SmallVector<RedistributeOp> reshapes;
    getOperation().walk([&](RedistributeOp redistribute) {
      if (isLocalReshape(redistribute))
        reshapes.push_back(redistribute);
    });
    for (RedistributeOp redistribute : reshapes) {
      OpBuilder builder(redistribute);
      MLIRContext *context = redistribute.getContext();
      Value input = redistribute.getInput();
      std::optional<Slicing> source = slicingOf(input);
      auto domain =
          llvm::to_vector(llvm::map_range(source->domain, [](Attribute extent) {
            return static_cast<int32_t>(cast<IntegerAttr>(extent).getInt());
          }));
      auto tile = GetTensorOp::create(builder, redistribute.getLoc(),
                                      input.getType(), input);
      tile->setDiscardableAttr(
          "slicing_begins",
          AffineMapAttr::get(mergeThreads(source->begins.getValue(), 0)));
      tile->setDiscardableAttr("slicing_domain",
                               builder.getI32ArrayAttr({domain[0], domain[1]}));
      tile->setDiscardableAttr("slicing_ends",
                               AffineMapAttr::get(mergeThreads(
                                   source->ends.getValue(), domain[2] - 1)));
      ArrayRef<int64_t> from =
          cast<DistributedTensorType>(input.getType()).getShape();
      ArrayRef<int64_t> to =
          cast<DistributedTensorType>(redistribute.getType()).getShape();
      auto reshaped = ReshapeOpOp::create(
          builder, redistribute.getLoc(), redistribute.getType(),
          tile.getResult(), AffineMapAttr::get(reshapeMap(from, to, context)),
          AffineMapAttr::get(reshapeMap(to, from, context)));
      auto read = GetTensorOp::create(builder, redistribute.getLoc(),
                                      redistribute.getType(), reshaped);
      for (StringRef name : kSlicingAttributes)
        read->setDiscardableAttr(name, redistribute->getAttr(name));
      redistribute.replaceAllUsesWith(read.getResult());
      redistribute.erase();
    }

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
      if (!isa_and_nonnull<RedistributeOp>(producer))
        continue;
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
