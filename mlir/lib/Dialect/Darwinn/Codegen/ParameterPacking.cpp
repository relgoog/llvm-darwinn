#include "Families.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Target/Darwinn/Parameters.h"
#include "llvm/Support/Endian.h"

using namespace mlir;
using namespace mlir::darwinn;
using namespace mlir::darwinn::codegen;

namespace {

FailureOr<SmallVector<float>> fillValues(Operation *fill) {
  auto values = dyn_cast<DenseElementsAttr>(cast<FillOp>(fill).getValue());
  if (!values || !isa<FloatType>(values.getElementType()))
    return failure();
  int64_t count = product(resultInfo(fill).shape);
  SmallVector<float> out;
  for (const APFloat &value : values.getValues<APFloat>())
    out.push_back(value.convertToFloat());
  if (values.isSplat())
    out.assign(count, out.front());
  if (static_cast<int64_t>(out.size()) != count)
    return failure();
  return out;
}

Operation *fillBehind(Value value) {
  Operation *node = value.getDefiningOp();
  while (node && !isa<FillOp>(node))
    node = producer(node, 0);
  return node;
}

FailureOr<std::optional<SmallVector<float>>> biasValues(Operation *op,
                                                        bool packed) {
  if (!packed)
    return std::optional<SmallVector<float>>();
  Operation *fill =
      op->getNumOperands() > 3 ? fillBehind(op->getOperand(3)) : nullptr;
  if (!fill)
    return failure();
  FailureOr<SmallVector<float>> values = fillValues(fill);
  if (failed(values))
    return failure();
  return std::optional<SmallVector<float>>(std::move(*values));
}

llvm::Expected<SmallVector<uint8_t>> packVmc(Operation *op) {
  FailureOr<VmcPlan> plan = vmcPlan(op);
  Operation *weights = fillBehind(op->getOperand(1));
  if (failed(plan) || !weights)
    return llvm::createStringError("unplanned convolution weights");
  SmallVector<int64_t, 4> shape = resultInfo(weightsView(op)).shape;
  int64_t cout = shape.back(), cin = shape[shape.size() - 2];
  int64_t kh = plan->transposed ? shape[1] : (shape.size() == 4 ? shape[0] : 1);
  int64_t kw = plan->transposed ? shape[3] : (shape.size() == 4 ? shape[1] : 1);
  FailureOr<SmallVector<float>> hwio = fillValues(weights);
  FailureOr<std::optional<SmallVector<float>>> bias =
      biasValues(op, plan->biasRows != 0);
  if (failed(hwio) || failed(bias))
    return llvm::createStringError("unreadable convolution parameters");
  SmallVector<float> ohwi(hwio->size());
  for (int64_t o = 0; o < cout; ++o)
    for (int64_t y = 0; y < kh; ++y)
      for (int64_t x = 0; x < kw; ++x)
        for (int64_t i = 0; i < cin; ++i)
          ohwi[((o * kh + y) * kw + x) * cin + i] =
              (*hwio)[((y * kw + x) * cin + i) * cout + o];
  llvm::Expected<ConvolutionShape> convolution = ConvolutionShape::create(
      {static_cast<size_t>(cout), static_cast<size_t>(kh),
       static_cast<size_t>(kw), static_cast<size_t>(cin)});
  if (!convolution)
    return convolution.takeError();
  std::optional<ArrayRef<float>> biasRef;
  if (*bias)
    biasRef = ArrayRef<float>(**bias);
  return packConvolution(*convolution, ohwi, biasRef,
                         plan->transposed ? plan->chunk : plan->cin);
}

llvm::Expected<SmallVector<uint8_t>> packStencil(Operation *op) {
  Operation *weights = fillBehind(op->getOperand(1));
  if (!weights)
    return llvm::createStringError("unplanned stencil weights");
  SmallVector<int64_t, 4> shape = resultInfo(weightsView(op)).shape;
  FailureOr<SmallVector<float>> filter = fillValues(weights);
  FailureOr<std::optional<SmallVector<float>>> bias =
      biasValues(op, hasBias(op));
  if (failed(filter) || failed(bias))
    return llvm::createStringError("unreadable stencil parameters");
  llvm::Expected<DepthwiseShape> depthwise = DepthwiseShape::create(
      {static_cast<size_t>(shape[0]), static_cast<size_t>(shape[1]),
       static_cast<size_t>(shape.back())});
  if (!depthwise)
    return depthwise.takeError();
  std::optional<ArrayRef<float>> biasRef;
  if (*bias)
    biasRef = ArrayRef<float>(**bias);
  return packDepthwise(*depthwise, *filter, biasRef);
}

llvm::Expected<SmallVector<uint8_t>> packFill(Operation *op) {
  FailureOr<SmallVector<float>> values = fillValues(op);
  if (failed(values))
    return llvm::createStringError("unreadable fill");
  if (resultInfo(op).elementBytes == 2)
    return packBfloat(*values);
  SmallVector<uint8_t> out;
  for (float value : *values) {
    uint8_t encoded[4];
    llvm::support::endian::write32le(encoded, llvm::bit_cast<uint32_t>(value));
    out.append(encoded, encoded + 4);
  }
  return out;
}

} // namespace

FailureOr<SmallVector<uint8_t, 0>> codegen::packParameters(ArrayRef<Hib> hibs) {
  SmallVector<uint8_t, 0> region;
  for (const Hib &hib : hibs) {
    if (hib.root != HibRoot::ParameterRegion || !hib.source)
      continue;
    Operation *op = hib.source;
    llvm::Expected<SmallVector<uint8_t>> bytes =
        isa<FillOp>(op)                                     ? packFill(op)
        : innerOperation(op) == InnerOperationKind::Stencil ? packStencil(op)
                                                            : packVmc(op);
    if (!bytes)
      return unsupported(op, llvm::toString(bytes.takeError()));
    if (static_cast<int64_t>(bytes->size()) != hib.size)
      return unsupported(op, "packed parameters whose size differs from the "
                             "streamed size");
    if (region.size() < static_cast<size_t>(hib.offset + hib.size))
      region.resize(hib.offset + hib.size, 0);
    llvm::copy(*bytes, region.begin() + hib.offset);
  }
  return region;
}
