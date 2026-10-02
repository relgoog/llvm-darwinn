#include "Codegen.h"
#include "mlir/IR/AffineExpr.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace mlir::darwinn;
using namespace mlir::darwinn::codegen;

static int64_t evaluateExpr(AffineExpr expr, ArrayRef<int64_t> point) {
  if (auto constant = dyn_cast<AffineConstantExpr>(expr))
    return constant.getValue();
  if (auto dim = dyn_cast<AffineDimExpr>(expr))
    return point[dim.getPosition()];
  auto binary = cast<AffineBinaryOpExpr>(expr);
  int64_t lhs = evaluateExpr(binary.getLHS(), point);
  int64_t rhs = evaluateExpr(binary.getRHS(), point);
  switch (binary.getKind()) {
  case AffineExprKind::Add:
    return lhs + rhs;
  case AffineExprKind::Mul:
    return lhs * rhs;
  case AffineExprKind::FloorDiv:
    return llvm::divideFloorSigned(lhs, rhs);
  case AffineExprKind::CeilDiv:
    return llvm::divideCeilSigned(lhs, rhs);
  case AffineExprKind::Mod:
    return llvm::mod(lhs, rhs);
  default:
    llvm_unreachable("slicing maps carry no symbols");
  }
}

Index codegen::evaluate(AffineMap map, ArrayRef<int64_t> point) {
  Index out;
  for (AffineExpr expr : map.getResults())
    out.push_back(evaluateExpr(expr, point));
  return out;
}

std::optional<Slicing> codegen::slicingOf(Operation *op) {
  auto begins = op->getAttrOfType<AffineMapAttr>("slicing_begins");
  auto ends = op->getAttrOfType<AffineMapAttr>("slicing_ends");
  auto domain = op->getAttrOfType<ArrayAttr>("slicing_domain");
  if (!begins || !ends || !domain)
    return std::nullopt;
  Slicing slicing{begins.getValue(), ends.getValue(), {}};
  for (Attribute value : domain)
    slicing.domain.push_back(cast<IntegerAttr>(value).getInt());
  return slicing;
}

Box codegen::unionBox(const Slicing &slicing, int64_t row, int64_t column) {
  SmallVector<Index> points;
  if (slicing.hasThreads()) {
    for (int64_t thread = 0; thread < slicing.domain[2]; ++thread)
      points.push_back({row, column, thread});
  } else {
    points.push_back({row, column});
  }
  Box box{slicing.begin(points.front()), slicing.end(points.front())};
  for (const Index &point : ArrayRef(points).drop_front()) {
    Index begin = slicing.begin(point), end = slicing.end(point);
    for (size_t dim = 0; dim < box.lo.size(); ++dim) {
      box.lo[dim] = std::min(box.lo[dim], begin[dim]);
      box.hi[dim] = std::max(box.hi[dim], end[dim]);
    }
  }
  return box;
}

Box codegen::threadBox(const Slicing &slicing, int64_t thread, int64_t tile) {
  Index point = slicing.hasThreads() ? Index{tile / kGrid, tile % kGrid, thread}
                                     : Index{tile / kGrid, tile % kGrid};
  return Box{slicing.begin(point), slicing.end(point)};
}

Index codegen::extent(const Box &box) {
  Index out;
  for (auto [lo, hi] : llvm::zip_equal(box.lo, box.hi))
    out.push_back(hi - lo + 1);
  return out;
}

SmallVector<TileBox> codegen::tiles(const Slicing &slicing) {
  SmallVector<TileBox> out;
  for (int64_t row = 0; row < slicing.domain[0]; ++row)
    for (int64_t column = 0; column < slicing.domain[1]; ++column)
      out.push_back({row * kGrid + column, unionBox(slicing, row, column)});
  return out;
}

static int64_t elementBytes(Type type) {
  if (type.isF32() || type.isInteger(32))
    return 4;
  if (type.isBF16() || type.isF16() || type.isInteger(16))
    return 2;
  if (type.isInteger(8))
    return 1;
  llvm_unreachable("unsupported distributed element type");
}

TensorInfo codegen::describe(Type type) {
  return llvm::TypeSwitch<Type, TensorInfo>(type)
      .Case<DistributedTensorType, DistributedViewType, EmptyTensorType,
            WriteViewType, FilledViewType>([](auto typed) {
        TensorInfo info;
        info.shape.assign(typed.getShape().begin(), typed.getShape().end());
        info.elementBytes = elementBytes(typed.getElementType());
        info.space = typed.getMemorySpace();
        return info;
      })
      .Default([](Type) -> TensorInfo {
        llvm_unreachable("codegen expects distributed types");
      });
}

TensorInfo codegen::resultInfo(Operation *op) {
  return describe(op->getResult(0).getType());
}

TensorInfo codegen::operandInfo(Operation *op, unsigned index) {
  return describe(op->getOperand(index).getType());
}

Index codegen::strides(ArrayRef<int64_t> shape, int64_t elementBytes) {
  Index out(shape.size(), elementBytes);
  for (int64_t dim = static_cast<int64_t>(shape.size()) - 2; dim >= 0; --dim)
    out[dim] = out[dim + 1] * shape[dim + 1];
  return out;
}

Index codegen::tail(ArrayRef<int64_t> values, size_t count) {
  return Index(values.take_back(count));
}

int64_t codegen::product(ArrayRef<int64_t> values) {
  int64_t out = 1;
  for (int64_t value : values)
    out *= value;
  return out;
}

int64_t codegen::dot(ArrayRef<int64_t> left, ArrayRef<int64_t> right) {
  int64_t out = 0;
  for (auto [a, b] : llvm::zip(left, right))
    out += a * b;
  return out;
}

int64_t codegen::ceilDiv(int64_t numerator, int64_t denominator) {
  return llvm::divideCeilSigned(numerator, denominator);
}

Operation *codegen::fillBehind(Value value) {
  Operation *node = value.getDefiningOp();
  while (node && !isa<FillOp>(node))
    node = producer(node, 0);
  return node;
}

Operation *codegen::producer(Operation *op, unsigned index) {
  if (index >= op->getNumOperands())
    return nullptr;
  return op->getOperand(index).getDefiningOp();
}

SmallVector<Operation *> codegen::usersOf(Operation *op) {
  SmallVector<Operation *> out;
  for (Value result : op->getResults())
    for (Operation *user : result.getUsers())
      if (!isa<func::ReturnOp>(user) && !llvm::is_contained(out, user))
        out.push_back(user);
  llvm::sort(out,
             [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });
  return out;
}

bool codegen::unused(Operation *op) { return usersOf(op).empty(); }

bool codegen::isModelOutput(Operation *op) {
  return llvm::any_of(op->getUsers(), [](Operation *user) {
    return isa<func::ReturnOp>(user) ||
           (isa<CommunicatedJoinViewsOp>(user) && isModelOutput(user));
  });
}

DistributedMemorySpace codegen::sourceSpace(Operation *op) {
  if (op->getNumOperands() == 0)
    return resultSpace(op);
  return operandInfo(op, 0).space;
}

DistributedMemorySpace codegen::resultSpace(Operation *op) {
  return resultInfo(op).space;
}

ComputeOpOptionsAttr codegen::computeOptions(Operation *op) {
  return op->getAttrOfType<ComputeOpOptionsAttr>("compute");
}

std::optional<InnerOperationKind> codegen::innerOperation(Operation *op) {
  if (auto options = computeOptions(op))
    return options.getInnerOperation();
  return std::nullopt;
}

std::optional<LinearFunctionKind> codegen::linearFunction(Operation *op) {
  if (auto options = computeOptions(op))
    return options.getLinearFunction();
  return std::nullopt;
}

std::optional<NluFunctionKind> codegen::nluFunction(Operation *op) {
  if (auto options = computeOptions(op))
    return options.getNluFunction();
  return std::nullopt;
}

bool codegen::hasAuxiliary(Operation *op, AuxTensorKind kind) {
  auto kinds = op->getAttrOfType<ArrayAttr>("auxiliary_tensor_types");
  if (!kinds)
    return false;
  return llvm::any_of(kinds, [&](Attribute attr) {
    return cast<AuxTensorTypeAttr>(attr).getValue() == kind;
  });
}

LogicalResult codegen::unsupported(Operation *op, const Twine &reason) {
  return op->emitOpError() << "codegen does not support " << reason;
}
