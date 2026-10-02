#include "mlir/Dialect/Darwinn/Transforms/OptimizePointwise.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Darwinn/IR/DwcOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Target/Darwinn/Parameters.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/CheckedArithmetic.h"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/Darwinn/IR/DwcAttributes.h.inc"

using namespace mlir;
using namespace mlir::darwinn;

namespace {

bool isLinearPointwise(dwc::ConvolutionOp operation) {
  if (operation->getNumOperands() != 3 || operation->getNumResults() != 1)
    return false;

  for (NamedAttribute attribute : operation->getAttrs()) {
    StringRef name = attribute.getName().getValue();

    if (name != "activation_function" && name != "cell_operation" &&
        name != "pad" && name != "x_stride" && name != "y_stride" &&
        name != "x_dilation_rate" && name != "y_dilation_rate")
      return false;
  }

  auto activation = operation->getAttrOfType<dwc::ActivationFunctionAttr>("activation_function");
  auto cell = operation->getAttrOfType<dwc::CellOperationAttr>("cell_operation");
  auto padding = operation->getAttrOfType<dwc::PaddingAttr>("pad");

  if (!activation || activation.getValue() != dwc::ActivationFunction::None ||
      !cell || cell.getValue() != dwc::CellOperation::Mac ||
      !padding || padding.getValue() != dwc::Padding::None)
    return false;

  for (StringRef name : {"x_stride", "y_stride", "x_dilation_rate", "y_dilation_rate"}) {
    auto value = operation->getAttrOfType<IntegerAttr>(name);

    if (!value || value.getValue() != 1)
      return false;
  }

  auto input = dyn_cast<RankedTensorType>(operation->getOperand(0).getType());
  auto filter = dyn_cast<RankedTensorType>(operation->getOperand(1).getType());
  auto output = dyn_cast<RankedTensorType>(operation->getResult(0).getType());
  auto bias = dyn_cast<RankedTensorType>(operation->getOperand(2).getType());

  for (RankedTensorType type : {input, filter, output, bias}) {
    if (!type || !type.hasStaticShape() || !type.getElementType().isF32() ||
        type.getEncoding() ||
        llvm::any_of(type.getShape(), [](int64_t extent) { return extent <= 0; }))
      return false;
  }

  if (input.getRank() != 4 || filter.getRank() != 4 || output.getRank() != 4 ||
      bias.getRank() != 1 || filter.getDimSize(0) != 1 || filter.getDimSize(1) != 1 ||
      input.getDimSize(3) != filter.getDimSize(2) ||
      output.getDimSize(3) != filter.getDimSize(3) ||
      bias.getDimSize(0) != output.getDimSize(3) ||
      input.getShape().take_front(3) != output.getShape().take_front(3))
    return false;

  for (Value value : operation->getOperands().drop_front()) {
    auto constant = value.getDefiningOp<arith::ConstantOp>();

    if (!constant || constant->getAttrs().size() != 1 ||
        !isa<DenseFPElementsAttr>(constant.getValue()))
      return false;
  }

  return true;
}

LogicalResult compose(dwc::ConvolutionOp second) {
  if (!isLinearPointwise(second))
    return success();

  // Zero SAME padding reaches this pass as a nofold pad, which a pointwise chain looks through.
  Value middle = second->getOperand(0);
  auto padding = middle.getDefiningOp<tensor::PadOp>();
  if (padding && padding->hasOneUse() &&
      llvm::all_of(llvm::concat<const int64_t>(padding.getStaticLow(), padding.getStaticHigh()),
                   [](int64_t amount) { return amount == 0; }) &&
      padding.getLow().empty() && padding.getHigh().empty())
    middle = padding.getSource();

  auto first = middle.getDefiningOp<dwc::ConvolutionOp>();

  if (!first || !first->hasOneUse() || first->getBlock() != second->getBlock() ||
      !isLinearPointwise(first))
    return success();

  auto firstType = cast<RankedTensorType>(first->getOperand(1).getType());
  auto secondType = cast<RankedTensorType>(second->getOperand(1).getType());
  size_t inputChannels = firstType.getDimSize(2);
  size_t middleChannels = firstType.getDimSize(3);
  size_t outputChannels = secondType.getDimSize(3);
  auto fusedElements = llvm::checkedMulUnsigned(inputChannels, outputChannels);

  if (!fusedElements)
    return success();

  auto fusedCost = llvm::checkedAddUnsigned(*fusedElements, outputChannels);
  std::optional<size_t> sourceCost = 0;

  for (Operation *operation : {first.getOperation(), second.getOperation()}) {
    for (Value value : operation->getOperands().drop_front()) {
      size_t count = cast<ShapedType>(value.getType()).getNumElements();
      sourceCost = llvm::checkedAddUnsigned(*sourceCost, count);

      if (!sourceCost)
        return success();
    }
  }

  if (!fusedCost || *fusedCost >= *sourceCost)
    return success();

  auto readValues = [](Value value) {
    auto constant = value.getDefiningOp<arith::ConstantOp>();
    auto dense = cast<DenseFPElementsAttr>(constant.getValue());
    return llvm::to_vector(dense.getValues<float>());
  };
  auto toOutputMajor = [&](Value value) {
    auto type = cast<RankedTensorType>(value.getType());
    SmallVector<float> input = readValues(value);
    SmallVector<float> output(input.size());
    size_t inputs = type.getDimSize(2);
    size_t outputs = type.getDimSize(3);

    for (size_t channel = 0; channel < outputs; ++channel) {
      for (size_t source = 0; source < inputs; ++source)
        output[channel * inputs + source] = input[source * outputs + channel];
    }

    return output;
  };
  auto firstShape = ConvolutionShape::create({middleChannels, 1, 1, inputChannels});
  auto secondShape = ConvolutionShape::create({outputChannels, 1, 1, middleChannels});

  if (!firstShape)
    return first.emitOpError() << llvm::toString(firstShape.takeError());
  if (!secondShape)
    return second.emitOpError() << llvm::toString(secondShape.takeError());

  SmallVector<float> firstFilter = toOutputMajor(first->getOperand(1));
  SmallVector<float> secondFilter = toOutputMajor(second->getOperand(1));
  SmallVector<float> firstBias = readValues(first->getOperand(2));
  SmallVector<float> secondBias = readValues(second->getOperand(2));
  auto composed = composePointwise(*firstShape, firstFilter, firstBias,
                                   *secondShape, secondFilter, secondBias);

  if (!composed)
    return second.emitOpError() << llvm::toString(composed.takeError());

  SmallVector<float> filter(composed->filter.size());

  for (size_t channel = 0; channel < outputChannels; ++channel) {
    for (size_t source = 0; source < inputChannels; ++source)
      filter[source * outputChannels + channel] =
          composed->filter[channel * inputChannels + source];
  }

  OpBuilder builder(second);
  Location location = FusedLoc::get(second.getContext(), {first.getLoc(), second.getLoc()});
  auto filterType = RankedTensorType::get(
      {1, 1, static_cast<int64_t>(inputChannels), static_cast<int64_t>(outputChannels)},
      builder.getF32Type());
  auto biasType = cast<RankedTensorType>(second->getOperand(2).getType());
  auto filterConstant = arith::ConstantOp::create(
      builder, location, filterType, DenseFPElementsAttr::get(filterType, ArrayRef<float>(filter)));
  auto biasConstant = arith::ConstantOp::create(
      builder, location, biasType, DenseFPElementsAttr::get(biasType, ArrayRef<float>(composed->bias)));
  llvm::SmallPtrSet<Operation *, 4> constants;

  for (Operation *operation : {first.getOperation(), second.getOperation()}) {
    for (Value value : operation->getOperands().drop_front())
      constants.insert(value.getDefiningOp());
  }

  second->setOperands({first->getOperand(0), filterConstant.getResult(), biasConstant.getResult()});
  second->setLoc(location);
  if (padding)
    padding.erase();
  first.erase();

  for (Operation *constant : constants) {
    if (constant->use_empty())
      constant->erase();
  }

  return success();
}

void fuseActivation(dwc::RescalingOp rescaling) {
  auto activation = rescaling->getAttrOfType<dwc::ActivationFunctionAttr>("activation_function");
  auto scales = rescaling->getAttrOfType<ArrayAttr>("output_activation_per_z_out_scales");
  auto padding = rescaling->getAttrOfType<dwc::PerZOutScalePaddingAttr>("per_z_out_scales_padding");
  if (rescaling->getNumOperands() != 2 || rescaling->getNumResults() != 1 || !activation ||
      (activation.getValue() != dwc::ActivationFunction::Logistic &&
       activation.getValue() != dwc::ActivationFunction::HardSwish) ||
      !scales || !scales.empty() || !padding ||
      padding.getValue() != dwc::PerZOutScalePadding::None ||
      !rescaling->getOperand(1).getDefiningOp<dwc::ConstNoneOp>() ||
      rescaling->getOperand(0).getType() != rescaling->getResult(0).getType())
    return;

  Operation *convolution = rescaling->getOperand(0).getDefiningOp();
  if (!isa_and_present<dwc::ConvolutionOp, dwc::DepthwiseConvolutionOp>(convolution) ||
      !convolution->hasOneUse() || convolution->hasAttr("activation_clip_min") ||
      convolution->getAttrOfType<dwc::ActivationFunctionAttr>("activation_function").getValue() !=
          dwc::ActivationFunction::None)
    return;

  Operation *none = rescaling->getOperand(1).getDefiningOp();
  convolution->setAttr("activation_function", activation);
  rescaling->getResult(0).replaceAllUsesWith(convolution->getResult(0));
  rescaling->erase();
  if (none->use_empty())
    none->erase();
}

class OptimizePointwisePass
    : public PassWrapper<OptimizePointwisePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OptimizePointwisePass)

  StringRef getArgument() const final { return "dwc-optimize-pointwise"; }
  StringRef getDescription() const final {
    return "Fuse activations into convolutions and compose shrinking linear "
           "pointwise "
           "convolutions before precision assignment";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<dwc::DwcDialect, arith::ArithDialect>();
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();
    OwningOpRef<ModuleOp> optimized(cast<ModuleOp>(module->clone()));
    SmallVector<dwc::RescalingOp> rescalings;
    optimized->walk([&](dwc::RescalingOp operation) { rescalings.push_back(operation); });
    for (dwc::RescalingOp operation : rescalings)
      fuseActivation(operation);

    SmallVector<dwc::ConvolutionOp> convolutions;
    optimized->walk([&](dwc::ConvolutionOp operation) { convolutions.push_back(operation); });

    for (dwc::ConvolutionOp operation : convolutions) {
      if (failed(compose(operation)))
        return signalPassFailure();
    }

    if (failed(verify(*optimized)))
      return signalPassFailure();

    module.getBodyRegion().takeBody(optimized->getBodyRegion());
  }
};

}

std::unique_ptr<Pass> mlir::darwinn::createOptimizePointwisePass() {
  return std::make_unique<OptimizePointwisePass>();
}

void mlir::darwinn::registerOptimizePointwisePass() {
  PassRegistration<OptimizePointwisePass>();
}
