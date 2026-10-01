#include "Families.h"
#include "mlir/IR/AffineExpr.h"

using namespace mlir;
using namespace mlir::darwinn;
using namespace mlir::darwinn::codegen;

namespace {

bool initNeeded(Operation *op) {
  auto inner = innerOperation(op);
  auto linear = linearFunction(op);
  if (inner == InnerOperationKind::Elementwise)
    return linear == LinearFunctionKind::Add ||
           linear == LinearFunctionKind::Sub;
  if (inner != InnerOperationKind::Unary)
    return false;
  if (linear != LinearFunctionKind::Add ||
      nluFunction(op) != NluFunctionKind::Linear)
    return false;
  auto traversal = op->getAttrOfType<AffineMapAttr>("traversal");
  return llvm::any_of(traversal.getValue().getResults(), [](AffineExpr expr) {
    auto constant = dyn_cast<AffineConstantExpr>(expr);
    return constant && constant.getValue() == 0;
  });
}

using TileBoxes = std::map<int64_t, Box>;

TileBoxes clampedBoxes(Operation *op) {
  TileBoxes out;
  auto slicing = slicingOf(op);
  if (!slicing)
    return out;
  SmallVector<int64_t, 4> shape = resultInfo(op).shape;
  for (auto [tile, box] : tiles(*slicing)) {
    Index lo = tail(box.lo, shape.size()), hi = tail(box.hi, shape.size());
    for (size_t dim = 0; dim < shape.size(); ++dim) {
      lo[dim] = std::max<int64_t>(0, lo[dim]);
      hi[dim] = std::min(shape[dim] - 1, hi[dim]);
    }
    out[tile] = Box{lo, hi};
  }
  return out;
}

GroupKind movement(Operation *input, Operation *op) {
  TileBoxes from = clampedBoxes(input), to = clampedBoxes(op);
  for (auto &[tile, box] : from)
    box = Box{applyForward(op, box.lo), applyForward(op, box.hi)};
  auto squeeze = [](Operation *owner, const TileBoxes &boxes) {
    SmallVector<int64_t, 4> shape = resultInfo(owner).shape;
    std::map<int64_t, SmallVector<std::pair<int64_t, int64_t>>> out;
    for (const auto &[tile, box] : boxes)
      for (size_t dim = 0; dim < shape.size(); ++dim)
        if (shape[dim] > 1)
          out[tile].push_back({box.lo[dim], box.hi[dim]});
    return out;
  };
  if (from.size() == to.size() && squeeze(input, from) == squeeze(op, to))
    return GroupKind::Empty;
  bool covers = llvm::all_of(to, [&](const auto &entry) {
    auto found = from.find(entry.first);
    if (found == from.end())
      return false;
    const Box &inner = found->second, &outer = entry.second;
    for (size_t dim = 0; dim < inner.lo.size(); ++dim)
      if (outer.lo[dim] > inner.lo[dim] || inner.hi[dim] > outer.hi[dim])
        return false;
    return true;
  });
  if (from.size() != to.size() || !covers)
    return GroupKind::Scatter;
  const Box &first = from.at(0);
  int64_t rows = first.hi[1] - first.lo[1] + 1;
  return rows >= 4 ? GroupKind::RingReshapeIdentity : GroupKind::Gather;
}

bool isPadded(Operation *op) {
  if (sourceSpace(op) != DistributedMemorySpace::HostMemory)
    return false;
  SmallVector<SmallVector<int64_t, 4>> shapes;
  for (Type type : op->getOperandTypes())
    shapes.push_back(describe(type).shape);
  for (Type type : op->getResultTypes())
    shapes.push_back(describe(type).shape);
  return llvm::any_of(
      shapes, [&](const auto &shape) { return shape != shapes.front(); });
}

void hostReaders(Operation *op, SmallVectorImpl<Operation *> &out) {
  for (Operation *user : usersOf(op)) {
    if (isa<RedistributeOp>(user) && producer(user, 0) == op)
      out.push_back(user);
    else if (isa<RedistributeOp, CommunicatedCreateWriteViewOp,
                 CommunicatedJoinViewsOp, DistributedCreateViewOp>(user))
      hostReaders(user, out);
  }
}

bool isView(Operation *op) {
  return isa<GetTensorOp, DistributedCreateViewOp, ReshapeOpOp>(op);
}

void consumerPositions(Operation *op,
                       SmallVectorImpl<std::pair<Operation *, unsigned>> &out) {
  for (Operation *user : usersOf(op)) {
    if (isView(user)) {
      consumerPositions(user, out);
      continue;
    }
    for (OpOperand &operand : user->getOpOperands())
      if (operand.get().getDefiningOp() == op)
        out.push_back({user, operand.getOperandNumber()});
  }
}

bool onlyRhsOfTensorOp(Operation *op) {
  SmallVector<std::pair<Operation *, unsigned>> positions;
  consumerPositions(op, positions);
  return positions.size() == 1 && isa<TensorOpOp>(positions[0].first) &&
         positions[0].second == 1;
}

} // namespace

SmallVector<Group> codegen::deriveSchedule(func::FuncOp function) {
  SmallVector<Group> groups;
  auto add = [&](GroupKind kind, Operation *op) {
    groups.push_back(Group{static_cast<int64_t>(groups.size()), kind, op});
  };
  for (Operation &operation : function.getBody().front()) {
    Operation *op = &operation;
    if (isa<PreemptionPointOp>(op)) {
      add(GroupKind::Preempt, op);
    } else if (isa<TensorOpOp, UnaryTensorOpOp>(op)) {
      if (initNeeded(op))
        add(GroupKind::Init, op);
      add(GroupKind::Op, op);
    } else if (auto fill = dyn_cast<FillOp>(op)) {
      if (!fill.getConstType())
        add(GroupKind::Op, op);
    } else if (isa<CopyOpOp>(op)) {
      add(GroupKind::Permute, op);
    } else if (isa<CommunicatedCreateEmptyTensorOp>(op)) {
      SmallVector<Operation *> readers;
      hostReaders(op, readers);
      if (llvm::none_of(readers, isPadded))
        add(GroupKind::Empty, op);
    } else if (isa<InterpolateHardwareOp>(op)) {
      add(GroupKind::Op, op);
    } else if (isa<RedistributeOp>(op)) {
      DistributedMemorySpace source = sourceSpace(op);
      DistributedMemorySpace target = resultSpace(op);
      Operation *input = producer(op, 0);
      if (source == DistributedMemorySpace::TileMemory &&
          target == DistributedMemorySpace::TileMemory &&
          op->getNumOperands() == 1) {
        if (input && slicingOf(input) == slicingOf(op))
          continue;
        GroupKind kind = movement(input, op);
        if (kind == GroupKind::Empty) {
          if (onlyRhsOfTensorOp(op))
            add(GroupKind::Empty, op);
        } else if (kind == GroupKind::Gather) {
          add(GroupKind::Gather, op);
          add(GroupKind::Op, op);
        } else {
          add(kind, op);
        }
        if (cast<RedistributeOp>(op).getMappingAttr() &&
            operandInfo(op, 0).shape != resultInfo(op).shape)
          add(GroupKind::Padding, op);
        continue;
      }
      add(GroupKind::Op, op);
      if (target == DistributedMemorySpace::HostMemory && unused(op))
        add(GroupKind::Relayout, op);
      if (isPadded(op))
        add(GroupKind::Padding, op);
    }
  }
  return groups;
}
