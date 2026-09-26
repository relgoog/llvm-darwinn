#include "SlicingModel.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>

using namespace mlir;
using namespace mlir::darwinn;
using namespace mlir::darwinn::slicing;

namespace {

unsigned nonReturnUses(Operation *operation) {
  unsigned count = 0;
  for (OpOperand &use : operation->getUses())
    if (!isa<func::ReturnOp>(use.getOwner()))
      ++count;
  return count;
}

bool isAnchor(Operation *operation) {
  if (isa<StaticComputeOpOp, StaticUnaryComputeOpOp, CopyOpOp,
          InterpolateHardwareOp>(operation))
    return true;
  if (auto view = dyn_cast<DistributedCreateViewOp>(operation))
    return view.getForwardIndexTransformation().has_value();
  if (isa<RedistributeOp>(operation))
    return nonReturnUses(operation) != 1;
  return false;
}

AffineMap wholeMap(MLIRContext *context, unsigned dims, ArrayRef<int64_t> shape,
                   bool ends) {
  SmallVector<AffineExpr> results;
  for (int64_t size : shape)
    results.push_back(getAffineConstantExpr(ends ? size - 1 : 0, context));
  return AffineMap::get(dims, 0, results, context);
}

}

int64_t mlir::darwinn::slicing::cycles(const Estimate &estimate) {
  auto get = [&](int key) {
    auto found = estimate.resources.find(key);
    return found == estimate.resources.end() ? int64_t{0} : found->second;
  };
  int64_t peak = std::max({get(1) + get(2), get(0), get(4)});
  for (int key = 6; key < 20; ++key)
    peak = std::max(peak, get(key));
  return estimate.base + peak + get(3) + get(20) + get(5);
}

SlicingModel::SlicingModel(func::FuncOp function) : function(function) {}

std::optional<unsigned> SlicingModel::blockOf(Operation *operation) const {
  auto found = blockIndex.find(operation);
  if (found == blockIndex.end())
    return std::nullopt;
  return found->second;
}

ArrayRef<unsigned> SlicingModel::predecessors(unsigned block) const {
  return preds[block];
}

ArrayRef<unsigned> SlicingModel::successors(unsigned block) const {
  return succs[block];
}

ArrayRef<Operation *> SlicingModel::edgeOperations(unsigned from,
                                                   unsigned to) const {
  auto found = edges.find({from, to});
  if (found == edges.end())
    return {};
  return found->second;
}

LogicalResult SlicingModel::build() {
  if (failed(partition()))
    return failure();
  buildGraph();
  return generateCodes();
}

LogicalResult SlicingModel::partition() {
  SmallVector<Operation *> operations;
  for (Operation &operation : function.getBody().front())
    if (!isa<func::ReturnOp>(operation))
      operations.push_back(&operation);

  for (Operation *operation : operations) {
    if (!isAnchor(operation))
      continue;
    blockIndex[operation] = blocks.size();
    blocks.push_back({{}, operation, operation->getResult(0)});
  }

  bool changed = true;
  while (changed) {
    changed = false;
    for (Operation *operation : llvm::reverse(operations)) {
      if (blockIndex.contains(operation))
        continue;
      if (auto reshape = dyn_cast<ReshapeOpOp>(operation)) {
        Operation *producer = reshape.getInput().getDefiningOp();
        if (producer && blockIndex.contains(producer)) {
          blockIndex[operation] = blockIndex[producer];
          changed = true;
        }
        continue;
      }
      std::optional<unsigned> target;
      bool uniform = false;
      for (OpOperand &use : operation->getUses()) {
        if (isa<func::ReturnOp>(use.getOwner())) {
          uniform = false;
          target.reset();
          break;
        }
        auto found = blockIndex.find(use.getOwner());
        if (found == blockIndex.end() || (target && *target != found->second)) {
          uniform = false;
          target.reset();
          break;
        }
        target = found->second;
        uniform = true;
      }
      if (!uniform || !target)
        continue;
      blockIndex[operation] = *target;
      changed = true;
    }
  }

  for (Operation *operation : operations) {
    auto found = blockIndex.find(operation);
    if (found == blockIndex.end())
      return operation->emitOpError("is not covered by a slicing block");
    blocks[found->second].members.push_back(operation);
  }
  return success();
}

void SlicingModel::buildGraph() {
  preds.assign(blocks.size(), {});
  succs.assign(blocks.size(), {});
  for (Operation &operation : function.getBody().front()) {
    std::optional<unsigned> to = blockOf(&operation);
    if (!to)
      continue;
    for (Value operand : operation.getOperands()) {
      Operation *producer = operand.getDefiningOp();
      std::optional<unsigned> from =
          producer ? blockOf(producer) : std::nullopt;
      if (!from || *from == *to)
        continue;
      if (!llvm::is_contained(preds[*to], *from))
        preds[*to].push_back(*from);
      if (!llvm::is_contained(succs[*from], *to))
        succs[*from].push_back(*to);
      SmallVector<Operation *> &list = edges[{*from, *to}];
      if (!llvm::is_contained(list, &operation))
        list.push_back(&operation);
    }
  }
}

unsigned SlicingModel::intern(Code code) {
  std::vector<int64_t> domain(code.domain.begin(), code.domain.end());
  auto key = std::make_tuple(domain, code.maps.begins.getAsOpaquePointer(),
                             code.maps.ends.getAsOpaquePointer());
  auto [found, inserted] = codeIndex.try_emplace(key, codes.size());
  if (inserted)
    codes.push_back(std::move(code));
  return found->second;
}

namespace {

struct Split {
  SmallVector<unsigned, 3> axes;
  SmallVector<int64_t, 3> extents;
  int64_t unit;
};

SmallVector<Split> splitDim(int64_t size, ArrayRef<unsigned> axes,
                            ArrayRef<int64_t> domain, int64_t granule) {
  SmallVector<int64_t, 3> extents;
  int64_t product = 1;
  for (unsigned axis : axes) {
    extents.push_back(domain[axis]);
    product *= domain[axis];
  }
  int64_t unit = llvm::divideCeilSigned(size, product);
  if (granule > 1) {
    unit = llvm::divideCeilSigned(unit, granule) * granule;
    if (unit > granule)
      unit = llvm::divideCeilSigned(unit, 2 * granule) * 2 * granule;
  }
  int64_t tiles = llvm::divideCeilSigned(size, unit);
  SmallVector<unsigned, 3> axisList(axes.begin(), axes.end());
  if (tiles >= product)
    return {{axisList, extents, unit}};

  auto fill = [&](ArrayRef<unsigned> order) {
    SmallVector<int64_t, 3> result(extents.size(), 0);
    int64_t remaining = tiles;
    for (unsigned index : order) {
      result[index] = std::min(extents[index], remaining);
      remaining = llvm::divideCeilSigned(remaining, result[index]);
    }
    return result;
  };
  SmallVector<unsigned, 3> outerFirst;
  for (unsigned index = extents.size(); index-- > 0;)
    outerFirst.push_back(index);
  SmallVector<Split> result{{axisList, fill(outerFirst), unit}};
  if (llvm::is_contained(axes, 2u)) {
    SmallVector<unsigned, 3> threadFirst;
    for (unsigned index = extents.size(); index-- > 0;)
      if (axes[index] != 2)
        threadFirst.push_back(index);
    for (unsigned index = 0; index < extents.size(); ++index)
      if (axes[index] == 2)
        threadFirst.push_back(index);
    SmallVector<int64_t, 3> variant = fill(threadFirst);
    if (variant != result.front().extents)
      result.push_back({axisList, variant, unit});
  }
  return result;
}

struct Assignment {
  unsigned outer;
  unsigned inner;
  std::optional<unsigned> thread;
};

bool hasRankOneOperand(Operation *operation) {
  return llvm::any_of(operation->getOperands(), [](Value operand) {
    return cast<ShapedType>(operand.getType()).getRank() == 1;
  });
}

}

LogicalResult SlicingModel::generateCodes() {
  MLIRContext *context = function.getContext();
  unsliced.clear();
  candidateLists.assign(blocks.size(), {});

  for (auto [index, block] : llvm::enumerate(blocks)) {
    ArrayRef<int64_t> shape = shapeOf(block.key);
    unsliced.push_back(intern({{1, 1},
                               {wholeMap(context, 2, shape, false),
                                wholeMap(context, 2, shape, true)}}));
  }

  for (auto [index, block] : llvm::enumerate(blocks)) {
    SmallVector<unsigned> &list = candidateLists[index];
    auto memory = [&](Value value) {
      if (auto tensor = dyn_cast<DistributedTensorType>(value.getType()))
        return tensor.getMemorySpace();
      return DistributedMemorySpace::TileMemory;
    };
    if (memory(block.key) == DistributedMemorySpace::HostMemory) {
      list.push_back(unsliced[index]);
      continue;
    }

    Operation *anchor = block.anchor;
    ArrayRef<int64_t> shape = shapeOf(block.key);
    unsigned rank = shape.size();
    SmallVector<int64_t, 3> domain = isa<CopyOpOp>(anchor)
                                         ? SmallVector<int64_t, 3>{4, 4, 1}
                                         : SmallVector<int64_t, 3>{4, 4, 4};
    llvm::SmallDenseSet<unsigned> threadForbidden;
    if (auto compute = dyn_cast<StaticComputeOpOp>(anchor)) {
      std::optional<InnerOperationKind> kind =
          compute.getCompute().getInnerOperation();
      if (kind == InnerOperationKind::Vmc ||
          kind == InnerOperationKind::Stencil ||
          (kind == InnerOperationKind::Elementwise &&
           hasRankOneOperand(anchor)))
        threadForbidden.insert(rank - 1);
    }
    auto granule = [&](unsigned dim) -> int64_t {
      return dim == rank - 1 && threadForbidden.contains(dim) ? 4 : 1;
    };

    SmallVector<unsigned> allowed;
    for (unsigned dim = 0; dim < rank; ++dim)
      if (shape[dim] > 1)
        allowed.push_back(dim);
    SmallVector<Assignment> assignments;
    bool threaded = domain[2] > 1;
    for (unsigned outer : allowed)
      for (unsigned inner : allowed) {
        if (!threaded) {
          assignments.push_back({outer, inner, std::nullopt});
          continue;
        }
        for (unsigned thread : allowed)
          assignments.push_back({outer, inner,
                                 threadForbidden.contains(thread)
                                     ? std::nullopt
                                     : std::optional(thread)});
      }

    for (const Assignment &assignment : assignments) {
      llvm::MapVector<unsigned, SmallVector<unsigned, 3>> axesOf;
      axesOf[assignment.outer].push_back(0);
      axesOf[assignment.inner].push_back(1);
      if (assignment.thread)
        axesOf[*assignment.thread].push_back(2);

      SmallVector<std::pair<unsigned, SmallVector<Split>>> options;
      for (auto &[dim, axes] : axesOf)
        options.push_back(
            {dim, splitDim(shape[dim], axes, domain, granule(dim))});

      SmallVector<SmallVector<std::pair<unsigned, Split>>> combinations{{}};
      for (auto &[dim, splits] : options) {
        SmallVector<SmallVector<std::pair<unsigned, Split>>> next;
        for (auto &prefix : combinations)
          for (const Split &split : splits) {
            auto extended = prefix;
            extended.push_back({dim, split});
            next.push_back(std::move(extended));
          }
        combinations = std::move(next);
      }

      for (auto &combination : combinations) {
        SmallVector<int64_t, 3> codeDomain{1, 1, 1};
        DenseMap<unsigned, const Split *> splitOf;
        for (auto &[dim, split] : combination) {
          for (auto [axis, extent] : llvm::zip(split.axes, split.extents))
            codeDomain[axis] = extent;
          splitOf[dim] = &split;
        }
        SmallVector<AffineExpr> begins;
        SmallVector<AffineExpr> ends;
        for (unsigned dim = 0; dim < rank; ++dim) {
          auto found = splitOf.find(dim);
          if (found == splitOf.end()) {
            begins.push_back(getAffineConstantExpr(0, context));
            ends.push_back(getAffineConstantExpr(shape[dim] - 1, context));
            continue;
          }
          const Split &split = *found->second;
          SmallVector<int64_t, 3> strides(split.axes.size());
          int64_t stride = split.unit;
          for (unsigned position = split.axes.size(); position-- > 0;) {
            strides[position] = stride;
            stride *= split.extents[position];
          }
          AffineExpr begin;
          for (unsigned position = 0; position < split.axes.size();
               ++position) {
            if (split.extents[position] <= 1)
              continue;
            AffineExpr term = getAffineDimExpr(split.axes[position], context) *
                              strides[position];
            begin = begin ? begin + term : term;
          }
          if (!begin)
            begin = getAffineConstantExpr(0, context);
          begins.push_back(begin);
          ends.push_back(split.unit > 1 ? begin + (split.unit - 1) : begin);
        }
        unsigned id = intern({codeDomain,
                              {AffineMap::get(3, 0, begins, context),
                               AffineMap::get(3, 0, ends, context)}});
        if (isa<InterpolateHardwareOp>(anchor) && assignment.thread != 1u)
          continue;
        if (assignment.inner < assignment.outer &&
            codeDomain[0] == codeDomain[1])
          continue;
        if (codeDomain[0] * codeDomain[1] == 1)
          continue;
        if (!llvm::is_contained(list, id))
          list.push_back(id);
      }
    }
    llvm::sort(list);
    if (list.empty())
      list.push_back(unsliced[index]);
  }
  return success();
}

namespace {

AffineExpr substitute(AffineExpr expression, ArrayRef<AffineExpr> values) {
  return expression.replaceDims(values);
}

}

FailureOr<const DenseMap<Value, SliceMaps> *>
SlicingModel::derive(unsigned blockId, unsigned codeId) {
  auto key = std::make_pair(blockId, codeId);
  auto found = derivations.find(key);
  if (found != derivations.end())
    return &found->second;

  MLIRContext *context = function.getContext();
  const SlicingBlock &block = blocks[blockId];
  const Code &code = codes[codeId];
  unsigned dims = code.domain.size();
  DenseMap<Value, SliceMaps> result;
  result[block.key] = code.maps;

  auto whole = [&](Value value) {
    return SliceMaps{wholeMap(context, dims, shapeOf(value), false),
                     wholeMap(context, dims, shapeOf(value), true)};
  };
  auto image = [&](AffineMap map, ArrayRef<AffineExpr> low,
                   ArrayRef<AffineExpr> high) {
    SmallVector<AffineExpr> begins;
    SmallVector<AffineExpr> ends;
    for (AffineExpr expression : map.getResults()) {
      begins.push_back(substitute(expression, low));
      ends.push_back(substitute(expression, high));
    }
    return SliceMaps{AffineMap::get(dims, 0, begins, context),
                     AffineMap::get(dims, 0, ends, context)};
  };

  Operation *anchor = block.anchor;
  if (isa<StaticComputeOpOp, StaticUnaryComputeOpOp>(anchor)) {
    AffineMap traversal = traversalOf(anchor);
    SmallVector<Value> views(anchor->getOperands());
    auto extents = iterationExtents(traversal.getNumDims(), views);
    if (failed(extents))
      return anchor->emitOpError("has views without linear traversals");
    unsigned iterationDims = traversal.getNumDims();
    SmallVector<AffineExpr> low;
    SmallVector<AffineExpr> high;
    for (int64_t extent : *extents) {
      low.push_back(getAffineConstantExpr(0, context));
      high.push_back(getAffineConstantExpr(extent - 1, context));
    }
    for (auto [index, expression] : llvm::enumerate(traversal.getResults())) {
      auto form = linearize(expression, iterationDims);
      if (!form)
        continue;
      SmallVector<unsigned> active;
      for (unsigned dim = 0; dim < iterationDims; ++dim)
        if (form->coefficients[dim] != 0)
          active.push_back(dim);
      if (active.empty())
        continue;
      AffineExpr begin = code.maps.begins.getResult(index);
      AffineExpr end = code.maps.ends.getResult(index);
      if (active.size() == 1) {
        unsigned dim = active.front();
        int64_t coefficient = form->coefficients[dim];
        low[dim] = (begin - form->constant).ceilDiv(coefficient);
        high[dim] = (end - form->constant).floorDiv(coefficient);
        continue;
      }
      unsigned main =
          *llvm::max_element(active, [&](unsigned left, unsigned right) {
            return form->coefficients[left] < form->coefficients[right];
          });
      int64_t minimum = form->constant;
      for (unsigned dim : active)
        if (dim != main)
          minimum += std::min<int64_t>(0, form->coefficients[dim] *
                                              ((*extents)[dim] - 1));
      int64_t coefficient = form->coefficients[main];
      low[main] = (begin - minimum).floorDiv(coefficient);
      high[main] = (end - minimum).floorDiv(coefficient);
    }
    Value destination =
        isa<StaticComputeOpOp>(anchor)
            ? cast<StaticComputeOpOp>(anchor).getDestination()
            : cast<StaticUnaryComputeOpOp>(anchor).getDestination();
    for (Value view : views) {
      auto createView = view.getDefiningOp<DistributedCreateViewOp>();
      if (!createView)
        continue;
      Value source = createView.getInput();
      if (view == destination) {
        result[source] = code.maps;
        continue;
      }
      if (source.getDefiningOp<FillOp>()) {
        result[source] = whole(source);
        continue;
      }
      result[source] = image(traversalOf(createView), low, high);
    }
  } else if (auto copy = dyn_cast<CopyOpOp>(anchor)) {
    result[copy.getInput()] =
        image(copy.getReverseIndexTransformation(),
              code.maps.begins.getResults(), code.maps.ends.getResults());
  } else if (auto interpolate = dyn_cast<InterpolateHardwareOp>(anchor)) {
    auto operands = anchor->getAttrOfType<ArrayAttr>("operand_traversals");
    auto resultTraversals =
        anchor->getAttrOfType<ArrayAttr>("result_traversals");
    auto domainAttr = anchor->getAttrOfType<ArrayAttr>("traversal_domain");
    if (!operands || !resultTraversals || !domainAttr)
      return anchor->emitOpError("lacks interpolation traversals");
    AffineMap operand = cast<AffineMapAttr>(operands[0]).getValue();
    AffineMap produced = cast<AffineMapAttr>(resultTraversals[0]).getValue();
    SmallVector<int64_t> extents;
    for (Attribute extent : domainAttr)
      extents.push_back(cast<IntegerAttr>(extent).getInt());
    unsigned iterationDims = produced.getNumDims();
    SmallVector<AffineExpr> low;
    SmallVector<AffineExpr> high;
    SmallVector<bool> full(iterationDims, true);
    for (int64_t extent : extents) {
      low.push_back(getAffineConstantExpr(0, context));
      high.push_back(getAffineConstantExpr(extent - 1, context));
    }
    for (auto [index, expression] : llvm::enumerate(produced.getResults())) {
      auto dimension = dyn_cast<AffineDimExpr>(expression);
      if (!dimension)
        continue;
      unsigned dim = dimension.getPosition();
      low[dim] = code.maps.begins.getResult(index);
      high[dim] = code.maps.ends.getResult(index);
      auto begin = dyn_cast<AffineConstantExpr>(low[dim]);
      auto end = dyn_cast<AffineConstantExpr>(high[dim]);
      full[dim] = begin && end && begin.getValue() == 0 &&
                  end.getValue() == extents[dim] - 1;
    }
    Value input = interpolate.getInput();
    SliceMaps maps = image(operand, low, high);
    SmallVector<AffineExpr> begins(maps.begins.getResults());
    SmallVector<AffineExpr> ends(maps.ends.getResults());
    ArrayRef<int64_t> inputShape = shapeOf(input);
    for (auto [index, expression] : llvm::enumerate(operand.getResults())) {
      bool allFull = true;
      expression.walk([&](AffineExpr sub) {
        if (auto dimension = dyn_cast<AffineDimExpr>(sub))
          allFull &= full[dimension.getPosition()];
      });
      if (!allFull)
        continue;
      begins[index] = getAffineConstantExpr(0, context);
      ends[index] = getAffineConstantExpr(inputShape[index] - 1, context);
    }
    result[input] = {AffineMap::get(dims, 0, begins, context),
                     AffineMap::get(dims, 0, ends, context)};
  } else {
    for (Operation *member : block.members)
      if (isa<RedistributeOp>(member))
        result[member->getResult(0)] = code.maps;
  }

  for (Operation *member : block.members)
    if (auto fill = dyn_cast<FillOp>(member))
      result[fill.getOutput()] = whole(fill.getOutput());

  auto [inserted, unused] = derivations.try_emplace(key, std::move(result));
  return &inserted->second;
}

namespace {

SmallVector<Tile> evaluate(const SliceMaps &maps, ArrayRef<int64_t> domain) {
  SmallVector<Tile> result;
  SmallVector<int64_t> point(domain.size(), 0);
  while (true) {
    result.push_back({maps.begins.compose(point), maps.ends.compose(point)});
    unsigned axis = domain.size();
    while (axis > 0) {
      --axis;
      if (++point[axis] < domain[axis])
        break;
      point[axis] = 0;
      if (axis == 0)
        return result;
    }
  }
}

}

FailureOr<const SmallVector<Tile> *> SlicingModel::tiles(Value value,
                                                         unsigned codeId) {
  auto key = std::make_pair(value.getAsOpaquePointer(), codeId);
  auto found = tileCache.find(key);
  if (found != tileCache.end())
    return &found->second;

  Operation *producer = value.getDefiningOp();
  std::optional<unsigned> blockId = producer ? blockOf(producer) : std::nullopt;
  if (!blockId)
    return failure();
  SmallVector<Tile> result;
  if (auto reshape = dyn_cast<ReshapeOpOp>(producer)) {
    auto inner = tiles(reshape.getInput(), codeId);
    if (failed(inner))
      return failure();
    for (const Tile &tile : **inner)
      result.push_back(
          reshapeTile(shapeOf(reshape.getInput()), shapeOf(value), tile));
  } else {
    auto derived = derive(*blockId, codeId);
    if (failed(derived))
      return failure();
    auto maps = (*derived)->find(value);
    if (maps == (*derived)->end())
      maps = (*derived)->find(blocks[*blockId].key);
    result = evaluate(maps->second, codes[codeId].domain);
  }
  auto [inserted, unused] = tileCache.try_emplace(key, std::move(result));
  return &inserted->second;
}

LogicalResult SlicingModel::emit(ArrayRef<unsigned> state) {
  MLIRContext *context = function.getContext();
  Builder builder(context);
  for (auto [blockId, block] : llvm::enumerate(blocks)) {
    const Code &code = codes[state[blockId]];
    auto derived = derive(blockId, state[blockId]);
    if (failed(derived))
      return failure();
    SmallVector<int32_t> domain(code.domain.begin(), code.domain.end());
    for (Operation *member : block.members) {
      if (!isa<RedistributeOp, CreateEmptyTensorOp, FillOp, CopyOpOp,
               InterpolateHardwareOp>(member))
        continue;
      Value value = member->getResult(0);
      auto maps = (*derived)->find(value);
      if (maps == (*derived)->end())
        return member->emitOpError("has no derived slicing");
      member->setAttr("slicing_begins",
                      AffineMapAttr::get(maps->second.begins));
      member->setAttr("slicing_domain", builder.getI32ArrayAttr(domain));
      member->setAttr("slicing_ends", AffineMapAttr::get(maps->second.ends));
    }
  }
  return success();
}

ArrayRef<int64_t> mlir::darwinn::slicing::shapeOf(Value value) {
  return cast<ShapedType>(value.getType()).getShape();
}

std::optional<LinearForm>
mlir::darwinn::slicing::linearize(AffineExpr expression, unsigned dims) {
  LinearForm result{SmallVector<int64_t>(dims, 0), 0};
  if (auto constant = dyn_cast<AffineConstantExpr>(expression)) {
    result.constant = constant.getValue();
    return result;
  }
  if (auto dimension = dyn_cast<AffineDimExpr>(expression)) {
    result.coefficients[dimension.getPosition()] = 1;
    return result;
  }
  auto binary = dyn_cast<AffineBinaryOpExpr>(expression);
  if (!binary)
    return std::nullopt;
  auto left = linearize(binary.getLHS(), dims);
  auto right = linearize(binary.getRHS(), dims);
  if (!left || !right)
    return std::nullopt;
  if (binary.getKind() == AffineExprKind::Add) {
    for (unsigned index = 0; index < dims; ++index)
      result.coefficients[index] =
          left->coefficients[index] + right->coefficients[index];
    result.constant = left->constant + right->constant;
    return result;
  }
  if (binary.getKind() != AffineExprKind::Mul)
    return std::nullopt;
  auto isConstant = [](const LinearForm &form) {
    return llvm::all_of(form.coefficients,
                        [](int64_t coefficient) { return coefficient == 0; });
  };
  if (isConstant(*left))
    std::swap(left, right);
  if (!isConstant(*right))
    return std::nullopt;
  for (unsigned index = 0; index < dims; ++index)
    result.coefficients[index] = left->coefficients[index] * right->constant;
  result.constant = left->constant * right->constant;
  return result;
}

AffineMap mlir::darwinn::slicing::traversalOf(Operation *operation) {
  if (auto attribute = operation->getAttrOfType<AffineMapAttr>("traversal"))
    return attribute.getValue();
  return {};
}

FailureOr<SmallVector<int64_t>>
mlir::darwinn::slicing::iterationExtents(unsigned dims, ArrayRef<Value> views) {
  struct Equation {
    LinearForm form;
    int64_t size;
  };
  SmallVector<Equation> equations;
  for (Value view : views) {
    Operation *producer = view.getDefiningOp();
    AffineMap traversal = producer ? traversalOf(producer) : AffineMap();
    if (!traversal)
      return failure();
    for (auto [index, result] : llvm::enumerate(traversal.getResults()))
      if (auto form = linearize(result, dims))
        equations.push_back({*form, shapeOf(view)[index]});
  }
  SmallVector<std::optional<int64_t>> extents(dims);
  bool changed = true;
  while (changed) {
    changed = false;
    for (const Equation &equation : equations) {
      SmallVector<unsigned> unknown;
      for (unsigned index = 0; index < dims; ++index)
        if (equation.form.coefficients[index] != 0 && !extents[index])
          unknown.push_back(index);
      if (unknown.size() != 1)
        continue;
      unsigned target = unknown.front();
      int64_t rest = 0;
      for (unsigned index = 0; index < dims; ++index)
        if (index != target && equation.form.coefficients[index] != 0)
          rest += equation.form.coefficients[index] * (*extents[index] - 1);
      extents[target] = (equation.size - 1 - equation.form.constant - rest) /
                            equation.form.coefficients[target] +
                        1;
      changed = true;
    }
  }
  SmallVector<int64_t> result;
  for (std::optional<int64_t> extent : extents)
    result.push_back(extent.value_or(1));
  return result;
}

Tile mlir::darwinn::slicing::reshapeTile(ArrayRef<int64_t> from,
                                         ArrayRef<int64_t> to,
                                         const Tile &tile) {
  Tile result{SmallVector<int64_t>(to.size(), 0),
              SmallVector<int64_t>(to.size(), 0)};
  unsigned left = 0;
  unsigned right = 0;
  while (left < from.size() || right < to.size()) {
    SmallVector<unsigned> groupFrom;
    SmallVector<unsigned> groupTo;
    int64_t productFrom = 1;
    int64_t productTo = 1;
    if (left < from.size()) {
      groupFrom.push_back(left);
      productFrom = from[left];
    }
    if (right < to.size()) {
      groupTo.push_back(right);
      productTo = to[right];
    }
    while (productFrom != productTo) {
      if (productFrom < productTo) {
        ++left;
        groupFrom.push_back(left);
        productFrom *= from[left];
      } else {
        ++right;
        groupTo.push_back(right);
        productTo *= to[right];
      }
    }
    int64_t low = 0;
    int64_t high = 0;
    for (unsigned dim : groupFrom) {
      low = low * from[dim] + tile.lo[dim];
      high = high * from[dim] + tile.hi[dim];
    }
    for (unsigned dim : llvm::reverse(groupTo)) {
      result.lo[dim] = low % to[dim];
      result.hi[dim] = high % to[dim];
      low /= to[dim];
      high /= to[dim];
    }
    left = groupFrom.empty() ? left + 1 : groupFrom.back() + 1;
    right = groupTo.empty() ? right + 1 : groupTo.back() + 1;
  }
  return result;
}
