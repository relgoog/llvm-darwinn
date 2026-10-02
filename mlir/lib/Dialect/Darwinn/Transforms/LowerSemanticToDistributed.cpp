#include "mlir/Dialect/Darwinn/Transforms/LowerSemanticToDistributed.h"
#include "SlicingModel.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Darwinn/IR/DarwinnOps.h"
#include "mlir/Dialect/Darwinn/IR/DwcOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include <array>
#include <limits>

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/Darwinn/IR/DwcAttributes.h.inc"

using namespace mlir;
using namespace mlir::darwinn;

namespace {

LogicalResult unsupported(Operation *operation, const Twine &reason) {
  return operation->emitOpError() << "cannot lower semantic graph because " << reason;
}

bool isSupportedTensor(Type type, bool rankFour = false) {
  auto tensor = dyn_cast<RankedTensorType>(type);
  if (!tensor || tensor.getEncoding() || !tensor.hasStaticShape() || tensor.getRank() < 1 ||
      tensor.getRank() > 4 || (rankFour && tensor.getRank() != 4) ||
      (!tensor.getElementType().isBF16() && !tensor.getElementType().isF32()))
    return false;

  return llvm::all_of(tensor.getShape(), [](int64_t extent) {
    return extent > 0 && extent <= std::numeric_limits<int32_t>::max();
  });
}

bool isZeroConstant(Value value) {
  Operation *constant = value.getDefiningOp();
  if (!constant || !isa<dwc::GenericConstantOp, arith::ConstantOp>(constant))
    return false;

  auto elements = constant->getAttrOfType<DenseFPElementsAttr>("value");
  return elements && llvm::all_of(elements.getValues<APFloat>(),
                                  [](const APFloat &element) { return element.isZero(); });
}

bool isConvolution(Operation *operation) {
  return isa<dwc::ConvolutionOp, dwc::DepthwiseConvolutionOp, dwc::TransposedConvolutionOp>(
      operation);
}

LogicalResult validateStructure(Operation *operation) {
  if (operation->getNumOperands() != 1 || operation->getNumResults() != 1 ||
      !isSupportedTensor(operation->getOperand(0).getType()) ||
      !isSupportedTensor(operation->getResult(0).getType()))
    return unsupported(operation, "reshape and transpose require one positive static tensor "
                                  "input and output of rank one through four");

  auto input = cast<RankedTensorType>(operation->getOperand(0).getType());
  auto output = cast<RankedTensorType>(operation->getResult(0).getType());
  if (input.getElementType() != output.getElementType())
    return unsupported(operation, "reshape and transpose must preserve element type");

  SmallVector<int64_t, 2> counts;
  for (auto type : {input, output}) {
    int64_t count = 1;
    for (int64_t extent : type.getShape()) {
      if (count > std::numeric_limits<int64_t>::max() / extent)
        return unsupported(operation, "tensor element count exceeds signed 64-bit storage");
      count *= extent;
    }
    counts.push_back(count);
  }

  if (counts[0] != counts[1])
    return unsupported(operation, "reshape and transpose must preserve element count");
  if (isa<dwc::ReshapeOp>(operation))
    return success();

  auto permutation = operation->getAttrOfType<DenseIntElementsAttr>("permutation");
  if (!permutation || permutation.getType().getRank() != 1 ||
      permutation.getElementType().getIntOrFloatBitWidth() > 64 ||
      permutation.getNumElements() != input.getRank() || input.getRank() != output.getRank())
    return unsupported(operation, "transpose requires a dense integer permutation matching "
                                  "the input and output rank");

  SmallVector<bool> seen(input.getRank(), false);
  for (auto [axis, attribute] : llvm::enumerate(permutation.getValues<APInt>())) {
    int64_t sourceAxis = attribute.getSExtValue();
    if (sourceAxis < 0 || sourceAxis >= input.getRank() || seen[sourceAxis] ||
        output.getDimSize(axis) != input.getDimSize(sourceAxis))
      return unsupported(operation, "transpose permutation must bijectively map input axes "
                                    "to the declared output shape");
    seen[sourceAxis] = true;
  }

  return success();
}

LogicalResult validateConvolution(Operation *operation) {
  bool depthwise = isa<dwc::DepthwiseConvolutionOp>(operation);
  auto input = cast<RankedTensorType>(operation->getOperand(0).getType());
  auto filter = cast<RankedTensorType>(operation->getOperand(1).getType());
  auto output = cast<RankedTensorType>(operation->getResult(0).getType());

  if (!isSupportedTensor(input, true) || !isSupportedTensor(filter) ||
      !isSupportedTensor(output, true) || !input.getElementType().isBF16() ||
      !filter.getElementType().isBF16())
    return unsupported(operation, "convolution requires positive static bf16 NHWC input and "
                                  "HWCF or depthwise HWC filter with bf16 or f32 output");

  auto activation = operation->getAttrOfType<dwc::ActivationFunctionAttr>("activation_function");
  auto cell = operation->getAttrOfType<dwc::CellOperationAttr>("cell_operation");
  auto padding = operation->getAttrOfType<dwc::PaddingAttr>("pad");

  if (cell.getValue() != dwc::CellOperation::Mac || padding.getValue() != dwc::Padding::None ||
      !llvm::is_contained({dwc::ActivationFunction::None, dwc::ActivationFunction::Relu,
                           dwc::ActivationFunction::Logistic, dwc::ActivationFunction::HardSwish},
                          activation.getValue()) ||
      (activation.getValue() != dwc::ActivationFunction::None &&
       activation.getValue() != dwc::ActivationFunction::Relu &&
       operation->hasAttr("activation_clip_min")))
    return unsupported(operation, "convolution requires MAC, explicit padding and NONE or RELU "
                                  "activation with optional typed clip bounds, or LOGISTIC or "
                                  "HARD_SWISH without them");

  if (operation->getAttrOfType<IntegerAttr>("x_dilation_rate").getInt() != 1 ||
      operation->getAttrOfType<IntegerAttr>("y_dilation_rate").getInt() != 1)
    return unsupported(operation, "convolution dilation must be one");

  bool transposed = isa<dwc::TransposedConvolutionOp>(operation);
  for (unsigned axis = 1; axis <= 2; ++axis) {
    int64_t stride = operation->getAttrOfType<IntegerAttr>(axis == 1 ? "y_stride" : "x_stride")
                         .getInt();
    int64_t extent = input.getDimSize(axis);
    int64_t kernel = filter.getDimSize(axis - 1);
    if (transposed && (kernel != stride || output.getDimSize(axis) != extent * stride))
      return unsupported(operation, "transposed convolution requires kernel extents equal to "
                                    "their strides without overlap");
    if (!transposed && (extent < kernel || output.getDimSize(axis) != (extent - kernel) / stride + 1))
      return unsupported(operation, "convolution output shape does not match filter and stride");
  }

  if (operation->getNumOperands() == 3 &&
      (!isSupportedTensor(operation->getOperand(2).getType()) ||
       !cast<ShapedType>(operation->getOperand(2).getType()).getElementType().isF32()))
    return unsupported(operation, "convolution bias requires a positive static f32 tensor");

  if (depthwise && filter.getRank() != 3)
    return unsupported(operation, "depthwise convolution requires multiplier-one HWC filters");

  return success();
}

LogicalResult validatePadding(tensor::PadOp operation) {
  auto input = operation.getSourceType();
  auto output = operation.getResultType();
  if (!isSupportedTensor(input, true) || !isSupportedTensor(output, true) ||
      !input.getElementType().isBF16() || !operation.getLow().empty() ||
      !operation.getHigh().empty())
    return unsupported(operation, "padding requires static rank-four bf16 storage and static "
                                  "spatial bounds");

  Value paddingValue = operation.getConstantPaddingValue();
  auto constant = paddingValue ? paddingValue.getDefiningOp<arith::ConstantOp>() : arith::ConstantOp{};
  auto value = constant ? dyn_cast<FloatAttr>(constant.getValue()) : FloatAttr{};
  if (!value || !value.getValue().isZero() || value.getValue().isNegative())
    return unsupported(operation, "padding requires positive floating-point zero");

  for (auto [axis, low, high] :
       llvm::enumerate(operation.getStaticLow(), operation.getStaticHigh())) {
    if (low < 0 || high < 0 || low > std::numeric_limits<int32_t>::max() ||
        high > std::numeric_limits<int32_t>::max() ||
        ((axis == 0 || axis == 3) && (low != 0 || high != 0)) ||
        output.getDimSize(axis) != input.getDimSize(axis) + low + high)
      return unsupported(operation, "padding bounds must be nonnegative spatial extents matching "
                                    "the result shape");
  }

  for (Operation &nested : operation.getRegion().front()) {
    if (&nested == constant.getOperation() || isa<tensor::YieldOp>(nested))
      continue;
    return unsupported(operation, "padding region must contain only its constant and yield");
  }

  return success();
}

LogicalResult validateCwise(dwc::CwiseOp operation) {
  if (operation->getNumOperands() != 2 || operation->getNumResults() != 1 ||
      !isSupportedTensor(operation->getResult(0).getType(), true))
    return unsupported(operation, "binary operations require two inputs and "
                                  "one positive static rank-four result");

  auto activation = operation->getAttrOfType<dwc::ActivationFunctionAttr>("activation_function");
  auto kind = operation->getAttrOfType<dwc::CwiseOpTypeAttr>("op_type");

  if (!activation || !kind ||
      (kind.getValue() != dwc::CwiseOpType::Add && kind.getValue() != dwc::CwiseOpType::Subtract &&
       kind.getValue() != dwc::CwiseOpType::Multiply) ||
      (activation.getValue() != dwc::ActivationFunction::None &&
       (activation.getValue() != dwc::ActivationFunction::Exp ||
        kind.getValue() != dwc::CwiseOpType::Subtract)))
    return unsupported(operation, "only ADD, SUBTRACT and MULTIPLY with NONE activation, or "
                                  "SUBTRACT with EXP, have a supported binary lowering");

  SmallVector<int64_t> joined(4, 1);
  unsigned maximumRank = 0;

  for (Value operand : operation->getOperands()) {
    if (!isSupportedTensor(operand.getType()))
      return unsupported(operation, "binary inputs require positive static "
                                    "tensors of rank one through four");

    auto input = cast<RankedTensorType>(operand.getType());
    if (!input.getElementType().isBF16())
      return unsupported(operation, "binary inputs require explicitly selected bf16 precision");

    maximumRank = std::max(maximumRank, unsigned(input.getRank()));
    unsigned leading = 4 - input.getRank();

    for (auto [axis, extent] : llvm::enumerate(input.getShape())) {
      int64_t &joinedExtent = joined[leading + axis];
      if (joinedExtent != 1 && extent != 1 && joinedExtent != extent)
        return unsupported(operation, "binary input shapes cannot broadcast");
      joinedExtent = std::max(joinedExtent, extent);
    }
  }

  auto output = cast<RankedTensorType>(operation->getResult(0).getType());
  if (maximumRank != 4 || ArrayRef<int64_t>(joined) != output.getShape())
    return unsupported(operation, "binary result shape must equal the broadcast input shape");

  return success();
}

FailureOr<SmallVector<unsigned>> reducedAxes(Operation *operation) {
  auto input = cast<RankedTensorType>(operation->getOperand(0).getType());
  auto dimensions = operation->getAttrOfType<DenseIntElementsAttr>("dimensions");
  if (!dimensions || dimensions.getNumElements() == 0)
    return failure();

  SmallVector<unsigned> axes;
  for (const APInt &value : dimensions.getValues<APInt>()) {
    int64_t axis = value.getSExtValue();
    if (axis < 0)
      axis += input.getRank();
    if (axis < 0 || axis >= input.getRank() || llvm::is_contained(axes, axis))
      return failure();
    axes.push_back(axis);
  }
  llvm::sort(axes);
  return axes;
}

LogicalResult validateReduction(dwc::ReductionOp operation) {
  if (operation->getNumOperands() != 1 || operation->getNumResults() != 1 ||
      !isSupportedTensor(operation->getOperand(0).getType(), true) ||
      !isSupportedTensor(operation->getResult(0).getType(), true))
    return unsupported(operation, "reduction requires one positive static rank-four input "
                                  "and result");

  auto input = cast<RankedTensorType>(operation->getOperand(0).getType());
  auto output = cast<RankedTensorType>(operation->getResult(0).getType());
  auto kind = operation->getAttrOfType<dwc::ReductionTypeAttr>("op_type");
  auto activation =
      operation->getAttrOfType<dwc::SimpleActivationFunctionAttr>("activation_function");
  FailureOr<SmallVector<unsigned>> axes = reducedAxes(operation);
  unsigned minor = input.getRank() - 1;

  if (failed(axes) || (axes->size() > 1 && llvm::is_contained(*axes, minor)) || !kind ||
      !activation || !input.getElementType().isBF16() ||
      (!output.getElementType().isBF16() && !output.getElementType().isF32()))
    return unsupported(operation, "reduction requires in-range axes, several only off the minor "
                                  "axis, bf16 input and a bf16 or f32 result");

  SmallVector<int64_t> reduced(input.getShape());
  int64_t count = 1;
  for (unsigned axis : *axes) {
    if (count > std::numeric_limits<int64_t>::max() / input.getDimSize(axis))
      return unsupported(operation, "reduced element count exceeds signed 64-bit storage");
    count *= input.getDimSize(axis);
    reduced[axis] = 1;
  }
  if (output.getShape() != ArrayRef<int64_t>(reduced))
    return unsupported(operation, "reduction result must keep the reduced axes with extent one");

  if ((kind.getValue() != dwc::ReductionType::Max && kind.getValue() != dwc::ReductionType::Sum &&
       kind.getValue() != dwc::ReductionType::Mean) ||
      (activation.getValue() == dwc::SimpleActivationFunction::Reciprocal &&
       kind.getValue() != dwc::ReductionType::Sum))
    return unsupported(operation, "only MAX, SUM, MEAN and reciprocal SUM reductions have a "
                                  "supported lowering");

  return success();
}

struct InterpolationAxis {
  int64_t startOffset;
  int64_t stride;
};

int64_t floorDivide(int64_t numerator, int64_t denominator) {
  int64_t quotient = numerator / denominator;
  return quotient - (numerator % denominator != 0 && (numerator < 0) != (denominator < 0));
}

int64_t roundDivide(int64_t numerator, int64_t denominator) {
  return floorDivide(2 * numerator + denominator, 2 * denominator);
}

// Reproduces all 34 SDK stage 240 resize probes whose spatial inputs exceed one, across both
// algorithms and every stride method. Extent-one inputs follow rules the probes leave open.
FailureOr<InterpolationAxis> interpolationAxis(int64_t input, int64_t output, bool nearest,
                                               StringRef method) {
  constexpr int64_t one = 1 << 16;
  bool align = method == "TENSORFLOW_ALIGN_CORNERS";
  bool half = method == "HALF_PIXEL_CENTERS";
  if (input < 2 || output < 1 || (align && output < 2) ||
      input > std::numeric_limits<int32_t>::max() / 4 ||
      output > std::numeric_limits<int32_t>::max() / 4)
    return failure();

  int64_t numerator = align ? input - 1 : input;
  int64_t denominator = align ? output - 1 : output;
  int64_t stride = roundDivide(numerator * one, denominator);
  int64_t start = half ? roundDivide((numerator - denominator) * (one / 2), denominator) : 0;

  if (nearest) {
    if (!half && !align)
      start = -one / 2;
    start += (std::max(input, output) + std::min(input, output) - 1) / std::min(input, output);
  }

  if (stride > std::numeric_limits<uint32_t>::max() ||
      start < std::numeric_limits<int32_t>::min() || start > std::numeric_limits<int32_t>::max())
    return failure();
  return InterpolationAxis{start, stride};
}

LogicalResult validateInterpolation(dwc::ImageInterpolationOp operation) {
  if (operation->getNumOperands() != 1 || operation->getNumResults() != 1 ||
      !isSupportedTensor(operation->getOperand(0).getType(), true) ||
      !isSupportedTensor(operation->getResult(0).getType(), true))
    return unsupported(operation, "interpolation requires one positive static rank-four input "
                                  "and result");

  auto input = cast<RankedTensorType>(operation->getOperand(0).getType());
  auto output = cast<RankedTensorType>(operation->getResult(0).getType());
  auto algorithm = operation->getAttrOfType<StringAttr>("algorithm");
  auto method = operation->getAttrOfType<StringAttr>("stride_method");
  if (!input.getElementType().isBF16() || output.getElementType() != input.getElementType() ||
      input.getDimSize(0) != output.getDimSize(0) || input.getDimSize(3) != output.getDimSize(3))
    return unsupported(operation, "interpolation requires bf16 NHWC tensors with matching batch "
                                  "and channels");

  for (unsigned axis = 1; axis <= 2; ++axis) {
    if (failed(interpolationAxis(input.getDimSize(axis), output.getDimSize(axis),
                                 algorithm.getValue() == "NEAREST_NEIGHBOR", method.getValue())))
      return unsupported(operation, "interpolation requires spatial input extents above one "
                                    "and a 16-bit fixed-point stride and start offset");
  }

  return success();
}

LogicalResult validateRescaling(dwc::RescalingOp operation) {
  if (operation->getNumOperands() != 2 || operation->getNumResults() != 1 ||
      !isSupportedTensor(operation->getOperand(0).getType(), true) ||
      !isSupportedTensor(operation->getResult(0).getType(), true))
    return unsupported(operation, "rescaling requires an input tensor, an absent "
                                  "bias and a positive static rank-four result");

  auto input = cast<RankedTensorType>(operation->getOperand(0).getType());
  auto output = cast<RankedTensorType>(operation->getResult(0).getType());
  auto activation = operation->getAttrOfType<dwc::ActivationFunctionAttr>("activation_function");
  auto scales = operation->getAttrOfType<ArrayAttr>("output_activation_per_z_out_scales");
  auto padding = operation->getAttrOfType<dwc::PerZOutScalePaddingAttr>("per_z_out_scales_padding");

  bool narrowing = input.getElementType().isF32() && output.getElementType().isBF16();
  bool widening = input.getElementType().isBF16() && output.getElementType().isF32();
  if (input.getShape() != output.getShape() || !(narrowing || widening) ||
      !operation->getOperand(1).getDefiningOp<dwc::ConstNoneOp>() || !activation ||
      activation.getValue() != dwc::ActivationFunction::None || !scales || !scales.empty() ||
      !padding || padding.getValue() != dwc::PerZOutScalePadding::None)
    return unsupported(operation, "rescaling currently supports only shape-preserving "
                                  "f32 and bf16 conversion without bias, scale or activation");

  return success();
}

FailureOr<bool> validateFunction(func::FuncOp function) {
  if (function.isExternal() || !function.getBody().hasOneBlock())
    return unsupported(function, "one defined block is required");

  for (Type type : function.getArgumentTypes()) {
    if (!isSupportedTensor(type, true))
      return unsupported(function, "arguments require positive static rank-four "
                                   "bf16 or f32 tensors");
  }

  if (function.getNumResults() == 0)
    return unsupported(function, "at least one tensor result is required");

  for (Type type : function.getResultTypes()) {
    if (!isSupportedTensor(type, true))
      return unsupported(function, "results require positive static rank-four "
                                   "bf16 or f32 tensors");
  }

  auto returned = dyn_cast<func::ReturnOp>(function.getBody().front().getTerminator());
  if (!returned)
    return unsupported(function, "the block must terminate with func.return");

  bool explicitBoundary = llvm::any_of(function.getBody().front(), [](Operation &operation) {
    return isa<InfeedOp, OutfeedOp>(operation);
  });

  if (explicitBoundary) {
    for (BlockArgument argument : function.getArguments()) {
      if (!argument.hasOneUse() || !isa<InfeedOp>(*argument.getUsers().begin()))
        return unsupported(function, "explicit host boundaries require exactly "
                                     "one infeed use of each function argument");
    }

    for (Value value : returned.getOperands()) {
      if (!value.getDefiningOp<OutfeedOp>())
        return unsupported(returned, "explicit host boundaries require an "
                                     "outfeed for each function result");
    }
  }

  for (Operation &operation : function.getBody().front()) {
    for (NamedAttribute attribute : operation.getAttrs()) {
      StringRef name = attribute.getName().getValue();
      bool known =
          (isa<dwc::CwiseOp>(operation) && (name == "activation_function" || name == "op_type")) ||
          (isa<dwc::RescalingOp>(operation) &&
           (name == "activation_function" || name == "output_activation_per_z_out_scales" ||
            name == "per_z_out_scales_padding")) ||
          (isConvolution(&operation) &&
           (name == "activation_function" || name == "cell_operation" || name == "pad" ||
            name == "x_dilation_rate" || name == "x_stride" || name == "y_dilation_rate" ||
            name == "y_stride" || name == "activation_clip_min" || name == "activation_clip_max" ||
            (isa<dwc::TransposedConvolutionOp>(operation) &&
             (name == "x_out_dim" || name == "y_out_dim")))) ||
          (isa<dwc::ImageInterpolationOp>(operation) &&
           (name == "algorithm" || name == "stride_method")) ||
          (isa<dwc::ReductionOp>(operation) &&
           (name == "activation_function" || name == "dimensions" || name == "op_type")) ||
          (isa<dwc::PoolingOp>(operation) &&
           (name == "pool" || name == "pad" || name == "kernel_x_dim" || name == "kernel_y_dim" ||
            name == "x_stride" || name == "y_stride")) ||
          (isa<tensor::PadOp>(operation) &&
           (name == "static_low" || name == "static_high" || name == "operandSegmentSizes" ||
            name == "nofold" || name == "pad_value")) ||
          (isa<dwc::TransposeOp>(operation) && name == "permutation") ||
          (isa<dwc::GenericConstantOp, arith::ConstantOp>(operation) &&
           (name == "value" || name == "darwinn.is_parameter"));
      if (!known)
        return unsupported(&operation, "unsupported semantic attribute " + name);
    }

    if (isa<dwc::ReshapeOp, dwc::TransposeOp>(operation)) {
      if (failed(validateStructure(&operation)))
        return failure();
      continue;
    }

    if (isConvolution(&operation)) {
      if (failed(validateConvolution(&operation)))
        return failure();
      continue;
    }

    if (auto padding = dyn_cast<tensor::PadOp>(operation)) {
      if (failed(validatePadding(padding)))
        return failure();
      continue;
    }

    if (auto cwise = dyn_cast<dwc::CwiseOp>(operation)) {
      if (failed(validateCwise(cwise)))
        return failure();
      continue;
    }

    if (auto interpolation = dyn_cast<dwc::ImageInterpolationOp>(operation)) {
      if (failed(validateInterpolation(interpolation)))
        return failure();
      continue;
    }

    if (auto reduction = dyn_cast<dwc::ReductionOp>(operation)) {
      if (failed(validateReduction(reduction)))
        return failure();
      continue;
    }

    if (auto rescaling = dyn_cast<dwc::RescalingOp>(operation)) {
      if (failed(validateRescaling(rescaling)))
        return failure();
      continue;
    }

    if (isa<dwc::PoolingOp>(operation)) {
      if (!isSupportedTensor(operation.getOperand(0).getType()) ||
          !isSupportedTensor(operation.getResult(0).getType(), true) ||
          !cast<ShapedType>(operation.getOperand(0).getType()).getElementType().isBF16())
        return unsupported(&operation, "pooling requires a bf16 input and a bf16 or f32 result");
      continue;
    }

    if (isa<dwc::GenericConstantOp, arith::ConstantOp>(operation)) {
      if (auto scalar = operation.getAttrOfType<FloatAttr>("value")) {
        if (!isa<arith::ConstantOp>(operation) || !scalar.getType().isBF16() ||
            !scalar.getValue().isZero() || scalar.getValue().isNegative())
          return unsupported(&operation, "scalar constants are only supported for positive-zero "
                                         "bf16 padding");

        for (Operation *user : operation.getResult(0).getUsers()) {
          auto yielded = dyn_cast<tensor::YieldOp>(user);
          auto padding = yielded ? dyn_cast<tensor::PadOp>(yielded->getParentOp()) : tensor::PadOp{};
          if (!padding || failed(validatePadding(padding)))
            return unsupported(&operation, "scalar constants may only supply tensor padding");
        }
        continue;
      }

      auto value = operation.getAttrOfType<ElementsAttr>("value");
      if (operation.getNumOperands() != 0 || operation.getNumResults() != 1 || !value ||
          !isSupportedTensor(operation.getResult(0).getType()) ||
          value.getType() != operation.getResult(0).getType() ||
          !isa<DenseElementsAttr, DenseResourceElementsAttr>(value))
        return unsupported(&operation,
                           "constants require dense positive static bf16 or f32 tensors");

      if (Attribute parameter = operation.getAttr("darwinn.is_parameter")) {
        if (!isa<BoolAttr>(parameter))
          return unsupported(&operation, "darwinn.is_parameter must be a boolean");
      }
      continue;
    }

    if (isa<dwc::ConstNoneOp>(operation)) {
      if (operation.getNumOperands() != 0 || operation.getNumResults() != 1 ||
          !isa<NoneType>(operation.getResult(0).getType()))
        return unsupported(&operation, "const_none requires one none result");

      for (OpOperand &use : operation.getResult(0).getUses()) {
        if (!isa<dwc::RescalingOp>(use.getOwner()) || use.getOperandNumber() != 1)
          return unsupported(&operation,
                             "const_none is only supported as an absent rescaling bias");
      }
      continue;
    }

    if (isa<InfeedOp, OutfeedOp>(operation)) {
      if (operation.getNumOperands() != 1 || operation.getNumResults() != 1 ||
          operation.getOperand(0).getType() != operation.getResult(0).getType() ||
          !isSupportedTensor(operation.getResult(0).getType(), true))
        return unsupported(&operation, "host transfers must preserve one ranked tensor");

      if (isa<InfeedOp>(operation)) {
        auto argument = dyn_cast<BlockArgument>(operation.getOperand(0));
        if (!argument || argument.getOwner() != &function.getBody().front())
          return unsupported(&operation, "infeed requires a function argument");
      } else {
        if (!operation.getResult(0).hasOneUse() ||
            !isa<func::ReturnOp>(*operation.getResult(0).getUsers().begin()))
          return unsupported(&operation, "outfeed must feed only the function return");
      }
      continue;
    }

    if (!isa<func::ReturnOp>(operation))
      return unsupported(&operation, "this operation has no semantic lowering");
  }

  return explicitBoundary;
}

class SemanticLowering {
public:
  explicit SemanticLowering(func::FuncOp output)
      : builder(output.getContext()), context(output.getContext()) {
    builder.setInsertionPointToStart(output.addEntryBlock());
  }

  void lower(func::FuncOp input, func::FuncOp output, bool explicitBoundary) {
    for (auto [source, destination] : llvm::zip(input.getArguments(), output.getArguments())) {
      Value mapped = destination;
      if (!explicitBoundary)
        mapped = redistribute(destination, DistributedMemorySpace::TileMemory);
      values.map(source, mapped);
    }

    for (Operation &operation : input.getBody().front()) {
      Location location = operation.getLoc();

      if (isa<dwc::ConstNoneOp>(operation))
        continue;

      if (isa<dwc::ReshapeOp, dwc::TransposeOp>(operation)) {
        auto type = tileType(operation.getResult(0).getType());
        Value input = values.lookup(operation.getOperand(0));
        Value result;
        if (isa<dwc::ReshapeOp>(operation)) {
          result = RedistributeOp::create(
              builder, location, type, input, Value{}, MappingAttr{}, sliceBegins(type.getRank()),
              builder.getI32ArrayAttr({1, 1}), sliceEnds(type.getShape()), IntegerAttr{});
        } else {
          auto permutation = operation.getAttrOfType<DenseIntElementsAttr>("permutation");
          SmallVector<AffineExpr> forward(type.getRank());
          SmallVector<AffineExpr> reverse(type.getRank());
          for (auto [axis, attribute] : llvm::enumerate(permutation.getValues<APInt>())) {
            unsigned sourceAxis = attribute.getZExtValue();
            forward[axis] = builder.getAffineDimExpr(sourceAxis);
            reverse[sourceAxis] = builder.getAffineDimExpr(axis);
          }

          result = CopyOpOp::create(
              builder, location, type, tileOperand(input), AffineMap::get(type.getRank(), 0, forward, context),
              AffineMap::get(type.getRank(), 0, reverse, context), AffineMapAttr{}, ArrayAttr{},
              AffineMapAttr{}, AffineMap::getMultiDimIdentityMap(type.getRank(), context));
        }

        values.map(operation.getResult(0), result);
        continue;
      }

      if (isa<dwc::GenericConstantOp, arith::ConstantOp>(operation)) {
        if (operation.getAttrOfType<FloatAttr>("value"))
          continue;
        auto value = operation.getAttrOfType<ElementsAttr>("value");
        auto type = tileType(operation.getResult(0).getType());
        auto parameter = operation.getAttrOfType<BoolAttr>("darwinn.is_parameter");
        unsigned ordinaryRole = parameter && parameter.getValue() ? 1 : 0;
        std::array<bool, 3> roles{};
        Value source = operation.getResult(0);

        for (OpOperand &use : source.getUses()) {
          unsigned role = ordinaryRole;
          if (isConvolution(use.getOwner()) && use.getOperandNumber() != 0)
            role = use.getOperandNumber();
          if (role != 2 || !isZeroConstant(source))
            roles[role] = true;
        }

        if (source.use_empty())
          roles[ordinaryRole] = true;
        auto &fills = constantValues[source];

        for (auto [role, needed] : llvm::enumerate(roles)) {
          if (!needed)
            continue;
          ConstTypeAttr kind;
          if (role != 0)
            kind = ConstTypeAttr::get(context,
                                      role == 1 ? ConstKind::Parameter : ConstKind::BiasOrScale);
          ElementsAttr contents = value;
          auto fillType = type;
          if (role == 1) {
            contents = parameterContents(source, value);
            fillType = tileType(contents.getType());
          }
          fills[role] = FillOp::create(builder, location, fillType, contents, kind,
                                       sliceBegins(fillType.getRank()),
                                       builder.getI32ArrayAttr({1, 1}),
                                       sliceEnds(fillType.getShape()));
        }

        if (fills[ordinaryRole])
          values.map(source, fills[ordinaryRole]);
        continue;
      }

      if (auto padding = dyn_cast<tensor::PadOp>(operation)) {
        SmallVector<AffineExpr> forward;
        SmallVector<AffineExpr> reverse;
        for (auto [axis, low] : llvm::enumerate(padding.getStaticLow())) {
          forward.push_back(builder.getAffineDimExpr(axis) + low);
          reverse.push_back(builder.getAffineDimExpr(axis) - low);
        }
        auto mapping = MappingAttr::get(
            context, AffineMapAttr::get(AffineMap::get(4, 0, forward, context)),
            AffineMapAttr::get(AffineMap::get(4, 0, reverse, context)));
        auto type = tileType(padding.getResultType());
        Value padded = RedistributeOp::create(
            builder, location, type, values.lookup(padding.getSource()), Value{}, mapping,
            sliceBegins(4), builder.getI32ArrayAttr({1, 1}), sliceEnds(type.getShape()),
            padding->getAttrOfType<IntegerAttr>("pad_value"));
        values.map(padding.getResult(), padded);
        continue;
      }

      if (isConvolution(&operation)) {
        values.map(operation.getResult(0), lowerConvolution(&operation));
        continue;
      }

      if (isa<InfeedOp, OutfeedOp>(operation)) {
        auto memory = isa<InfeedOp>(operation) ? DistributedMemorySpace::TileMemory
                                               : DistributedMemorySpace::HostMemory;
        values.map(operation.getResult(0),
                   redistribute(values.lookup(operation.getOperand(0)), memory));
        continue;
      }

      if (auto returned = dyn_cast<func::ReturnOp>(operation)) {
        SmallVector<Value> operands;
        for (Value value : returned.getOperands()) {
          Value mapped = values.lookup(value);
          if (!explicitBoundary)
            mapped = redistribute(mapped, DistributedMemorySpace::HostMemory);
          operands.push_back(mapped);
        }
        func::ReturnOp::create(builder, location, operands);
        continue;
      }

      if (isa<dwc::ReductionOp>(operation)) {
        values.map(operation.getResult(0), lowerReduction(&operation));
        continue;
      }

      if (isa<dwc::ImageInterpolationOp>(operation)) {
        values.map(operation.getResult(0), lowerInterpolation(&operation));
        continue;
      }

      if (isa<dwc::PoolingOp>(operation)) {
        values.map(operation.getResult(0), lowerPooling(&operation));
        continue;
      }

      auto outputType = tileType(operation.getResult(0).getType());
      auto traversal = AffineMap::getMultiDimIdentityMap(outputType.getRank(), context);
      SmallVector<Value> operands;
      unsigned count = isa<dwc::CwiseOp>(operation) ? 2 : 1;

      for (Value operand : operation.getOperands().take_front(count)) {
        operands.push_back(view(tileOperand(values.lookup(operand)), outputType.getShape()));
      }

      Value empty = CreateEmptyTensorOp::create(
          builder, location, outputType, sliceBegins(outputType.getRank()),
          builder.getI32ArrayAttr({1, 1}), sliceEnds(outputType.getShape()));
      Value destination = view(empty, outputType.getShape());
      Value result;

      if (auto cwise = dyn_cast<dwc::CwiseOp>(operation)) {
        auto kind = cwise->getAttrOfType<dwc::CwiseOpTypeAttr>("op_type").getValue();
        auto linear = kind == dwc::CwiseOpType::Multiply   ? LinearFunctionKind::Mac
                      : kind == dwc::CwiseOpType::Subtract ? LinearFunctionKind::Sub
                                                           : LinearFunctionKind::Add;
        auto activation =
            cwise->getAttrOfType<dwc::ActivationFunctionAttr>("activation_function").getValue();
        ComputeOpOptionsAttr options = computeOptions(InnerOperationKind::Elementwise, linear);

        // SDK stage 238 clips every softmax exponent input to this range, across four probes
        // with differing beta and shape. Its derivation in the SDK is not recovered.
        if (activation == dwc::ActivationFunction::Exp)
          options = computeOptions(InnerOperationKind::Elementwise, linear,
                                   builder.getF32FloatAttr(-12.9571848f),
                                   builder.getF32FloatAttr(0.0f), NluFunctionKind::Exp);

        Value computed = StaticComputeOpOp::create(
            builder, location, outputType, operands[0], operands[1], destination, ValueRange{},
            options, traversal, ArrayAttr{}, CustomTilingOptionsAttr{}, DtcInfoAttr{},
            VexInfoAttr{});
        result = computed;
      } else {
        result = StaticUnaryComputeOpOp::create(
            builder, location, outputType, operands[0], destination, ValueRange{},
            computeOptions(InnerOperationKind::Unary, LinearFunctionKind::Add), traversal,
            builder.getArrayAttr({}), CustomTilingOptionsAttr{}, ArrayAttr{}, VexInfoAttr{});
      }

      values.map(operation.getResult(0), result);
    }

    // Chained tile redistributes collapse into the later one when the earlier carries no mapping,
    // so a reshape followed by padding becomes one mapped redistribute as in SDK stage 248.
    SmallVector<RedistributeOp> chained;
    output.walk([&](RedistributeOp later) {
      auto earlier = later.getInput().getDefiningOp<RedistributeOp>();
      if (earlier && earlier->hasOneUse() && !earlier.getMappingAttr() &&
          cast<DistributedTensorType>(earlier.getInput().getType()).getMemorySpace() ==
              DistributedMemorySpace::TileMemory)
        chained.push_back(later);
    });
    for (RedistributeOp later : chained) {
      Operation *earlier = later.getInput().getDefiningOp();
      later->setOperand(0, earlier->getOperand(0));
      earlier->erase();
    }

    // A redistributed tensor read by more than one consumer gets a private tile copy per view,
    // as SDK stage 255 shows for multiply-used reshapes.
    SmallVector<DistributedCreateViewOp> shared;
    output.walk([&](DistributedCreateViewOp created) {
      Value storage = created.getInput();
      if (storage.getDefiningOp<RedistributeOp>() && !storage.hasOneUse())
        shared.push_back(created);
    });
    for (DistributedCreateViewOp created : shared) {
      builder.setInsertionPoint(created);
      created->setOperand(0, redistribute(created.getInput(), DistributedMemorySpace::TileMemory));
    }
  }

private:
  Value lowerTransposedConvolution(Operation *operation, Value input, Value filter, Value bias) {
    SmallVector<AffineExpr> dimensions;
    for (unsigned axis = 0; axis < 9; ++axis)
      dimensions.push_back(builder.getAffineDimExpr(axis));

    int64_t yStride = operation->getAttrOfType<IntegerAttr>("y_stride").getInt();
    int64_t xStride = operation->getAttrOfType<IntegerAttr>("x_stride").getInt();
    auto map = [&](ArrayRef<AffineExpr> results) {
      return AffineMap::get(9, 0, results, context);
    };
    auto traversal = map({dimensions[0], dimensions[1] * yStride + dimensions[5],
                          dimensions[2] * xStride + dimensions[6], dimensions[8]});
    Value inputView = view(input, map({dimensions[0], dimensions[1] + dimensions[3],
                                       dimensions[2] + dimensions[4], dimensions[7]}));
    Value filterView = view(filter, map({dimensions[3], dimensions[5], dimensions[4],
                                         dimensions[6], dimensions[7], dimensions[8]}));
    Value computed =
        lowerMultiplyAccumulate(operation, inputView, filterView, bias, traversal,
                                InnerOperationKind::Vmc, ComputeTypeHintKind::TransposedConv);

    // SDK stage 248 reads a transposed convolution result through an identity output view.
    Value moved = redistribute(computed, DistributedMemorySpace::TileMemory);
    auto type = cast<DistributedTensorType>(moved.getType());
    auto identity = AffineMapAttr::get(AffineMap::getMultiDimIdentityMap(type.getRank(), context));
    return DistributedCreateViewOp::create(
        builder, moved.getLoc(),
        DistributedViewType::get(context, type.getShape(), type.getElementType(),
                                 type.getMemorySpace()),
        moved, identity, identity);
  }

  Value lowerConvolution(Operation *operation) {
    bool depthwise = isa<dwc::DepthwiseConvolutionOp>(operation);
    unsigned rank = depthwise ? 8 : 7;
    SmallVector<AffineExpr> dimensions;
    for (unsigned axis = 0; axis < rank; ++axis)
      dimensions.push_back(builder.getAffineDimExpr(axis));

    auto operandStorage = [&](unsigned index) {
      Value source = operation->getOperand(index);
      auto found = constantValues.find(source);
      Value storage = index != 0 && found != constantValues.end()
                          ? found->second[index]
                          : values.lookup(source);
      return tileOperand(storage);
    };

    Value input = operandStorage(0);
    Value filter = operandStorage(1);
    // SDK stage 113 drops biases whose elements are all positive or negative zero.
    Value bias = operation->getNumOperands() == 3 && !isZeroConstant(operation->getOperand(2))
                     ? operandStorage(2)
                     : Value{};
    if (isa<dwc::TransposedConvolutionOp>(operation))
      return lowerTransposedConvolution(operation, input, filter, bias);

    int64_t xStride = operation->getAttrOfType<IntegerAttr>("x_stride").getInt();
    int64_t yStride = operation->getAttrOfType<IntegerAttr>("y_stride").getInt();
    AffineExpr channel = dimensions[depthwise ? 7 : 5];
    Value inputView = view(input, AffineMap::get(
        rank, 0, {dimensions[0], dimensions[1] * yStride + dimensions[3],
                  dimensions[2] * xStride + dimensions[4], channel}, context));
    SmallVector<AffineExpr> filterCoordinates(dimensions.begin() + 3, dimensions.end());
    Value filterView = view(filter, AffineMap::get(rank, 0, filterCoordinates, context));
    auto traversal = AffineMap::get(
        rank, 0, {dimensions[0], dimensions[1], dimensions[2], dimensions.back()}, context);
    return lowerMultiplyAccumulate(operation, inputView, filterView, bias, traversal,
                                   depthwise ? InnerOperationKind::Stencil : InnerOperationKind::Vmc,
                                   depthwise ? std::nullopt
                                             : std::optional(ComputeTypeHintKind::Conv));
  }

  Value lowerMultiplyAccumulate(Operation *operation, Value inputView, Value filterView,
                                Value bias, AffineMap traversal, InnerOperationKind inner,
                                std::optional<ComputeTypeHintKind> hint) {
    Location location = operation->getLoc();
    auto outputType = tileType(operation->getResult(0).getType());
    Value empty = CreateEmptyTensorOp::create(
        builder, location, outputType, sliceBegins(4), builder.getI32ArrayAttr({1, 1}),
        sliceEnds(outputType.getShape()));
    Value destination = view(empty, traversal);
    SmallVector<Value> auxiliary;
    ArrayAttr auxiliaryKinds;

    if (bias) {
      auto extent = cast<DistributedTensorType>(bias.getType()).getShape().front();
      AffineExpr channel = extent == 1 ? builder.getAffineConstantExpr(0)
                                       : traversal.getResults().back();
      auxiliary.push_back(view(bias, AffineMap::get(traversal.getNumDims(), 0, channel)));
      auxiliaryKinds = builder.getArrayAttr({AuxTensorTypeAttr::get(context, AuxTensorKind::Bias)});
    }

    auto lower = operation->getAttrOfType<FloatAttr>("activation_clip_min");
    auto upper = operation->getAttrOfType<FloatAttr>("activation_clip_max");
    if (!lower) {
      lower = builder.getF32FloatAttr(-std::numeric_limits<float>::infinity());
      upper = builder.getF32FloatAttr(std::numeric_limits<float>::infinity());
    }

    auto activation = operation->getAttrOfType<dwc::ActivationFunctionAttr>("activation_function");
    NluFunctionKind nlu = NluFunctionKind::Linear;
    if (activation.getValue() == dwc::ActivationFunction::Relu) {
      float bound = std::min(std::max(0.0f, lower.getValue().convertToFloat()),
                             upper.getValue().convertToFloat());
      lower = builder.getF32FloatAttr(bound);
    } else if (activation.getValue() == dwc::ActivationFunction::Logistic) {
      nlu = NluFunctionKind::LogisticSigmoid;
      lower = builder.getF32FloatAttr(-11.1f);
      upper = builder.getF32FloatAttr(11.1f);
    } else if (activation.getValue() == dwc::ActivationFunction::HardSwish) {
      nlu = NluFunctionKind::HardSwish;
      lower = builder.getF32FloatAttr(-4.0f);
    }

    Value computed = StaticComputeOpOp::create(
        builder, location, outputType, inputView, filterView, destination, auxiliary,
        computeOptions(inner, LinearFunctionKind::Mac, lower, upper, nlu, {}, hint), traversal,
        auxiliaryKinds, CustomTilingOptionsAttr{}, DtcInfoAttr{}, VexInfoAttr{});
    return computed;
  }

  // Mirrors SDK stage 255. A reduction over the minor axis becomes a two-stage tree over lanes, the
  // largest divisor of the extent that is at most eight and below it. Other axes are traversed
  // just before the minor axis.
  Value lowerReduction(Operation *operation) {
    SmallVector<unsigned> axes = *reducedAxes(operation);
    auto kind = operation->getAttrOfType<dwc::ReductionTypeAttr>("op_type").getValue();
    bool maximum = kind == dwc::ReductionType::Max;
    bool reciprocal =
        operation->getAttrOfType<dwc::SimpleActivationFunctionAttr>("activation_function")
            .getValue() == dwc::SimpleActivationFunction::Reciprocal;
    auto inputType = cast<ShapedType>(operation->getOperand(0).getType());
    auto outputType = tileType(operation->getResult(0).getType());
    unsigned rank = inputType.getRank();
    Value source = values.lookup(operation->getOperand(0));
    int64_t count = 1;
    for (unsigned axis : axes)
      count *= inputType.getDimSize(axis);
    float scale = kind == dwc::ReductionType::Mean ? 1.0f / count : 1.0f;
    auto finalOptions = [&] {
      return reductionOptions(maximum, reciprocal,
                              builder.getF32FloatAttr(maximum ? -0.0f : 0.0f), scale);
    };

    if (axes.back() != rank - 1) {
      SmallVector<unsigned> order;
      for (unsigned dimension = 0; dimension + 1 < rank; ++dimension) {
        if (!llvm::is_contained(axes, dimension))
          order.push_back(dimension);
      }
      llvm::append_range(order, axes);
      order.push_back(rank - 1);

      SmallVector<AffineExpr> inputCoordinates(rank);
      for (auto [position, dimension] : llvm::enumerate(order))
        inputCoordinates[dimension] = builder.getAffineDimExpr(position);
      SmallVector<AffineExpr> outputCoordinates(inputCoordinates);
      for (unsigned axis : axes)
        outputCoordinates[axis] = builder.getAffineConstantExpr(0);

      auto traversal = AffineMap::get(rank, 0, outputCoordinates, context);
      Value inputView =
          view(tileOperand(source), AffineMap::get(rank, 0, inputCoordinates, context));
      return reduce(inputView, outputType, traversal, finalOptions());
    }

    int64_t extent = inputType.getDimSize(rank - 1);
    int64_t lanes = 1;
    for (int64_t candidate = std::min<int64_t>(8, extent - 1); candidate > 1; --candidate) {
      if (extent % candidate == 0) {
        lanes = candidate;
        break;
      }
    }

    auto shapeWith = [&](ArrayRef<int64_t> minor, Type element) {
      SmallVector<int64_t> shape(inputType.getShape().drop_back());
      shape.append(minor.begin(), minor.end());
      return DistributedTensorType::get(context, shape, element,
                                        DistributedMemorySpace::TileMemory);
    };
    SmallVector<AffineExpr> minorZero;
    for (unsigned dimension = 0; dimension <= rank; ++dimension) {
      minorZero.push_back(dimension == rank - 1 ? builder.getAffineConstantExpr(0)
                                                : builder.getAffineDimExpr(dimension));
    }

    auto traversal = AffineMap::get(rank + 1, 0, minorZero, context);
    auto identity = AffineMap::getMultiDimIdentityMap(rank + 1, context);
    Type element = outputType.getElementType();
    Value split;

    if (lanes == 1) {
      split = reshape(tileOperand(source), shapeWith({extent, 1}, inputType.getElementType()));
    } else {
      auto splitType = shapeWith({extent / lanes, lanes}, inputType.getElementType());
      split = RedistributeOp::create(builder, operation->getLoc(), splitType, source, Value{},
                                     MappingAttr{}, sliceBegins(rank + 1),
                                     builder.getI32ArrayAttr({1, 1}),
                                     sliceEnds(splitType.getShape()), IntegerAttr{});
      Type partial = maximum ? inputType.getElementType() : builder.getF32Type();
      Value lanesFirst = reduce(view(split, identity), shapeWith({1, lanes}, partial), traversal,
                                reductionOptions(maximum, false, builder.getF32FloatAttr(-0.0f)));
      split = redistribute(reshape(lanesFirst, shapeWith({lanes, 1}, partial)),
                           DistributedMemorySpace::TileMemory);
    }

    Value reduced =
        reduce(view(split, identity), shapeWith({1, 1}, element), traversal, finalOptions());
    return reshape(reduced, outputType);
  }

  ComputeOpOptionsAttr reductionOptions(bool maximum, bool reciprocal, FloatAttr bias,
                                        float scale = 1.0f) {
    return computeOptions(InnerOperationKind::Unary,
                          maximum ? LinearFunctionKind::Max : LinearFunctionKind::Add, {}, {},
                          reciprocal ? NluFunctionKind::Reciprocal : NluFunctionKind::Linear, bias,
                          std::nullopt, true, scale);
  }

  Value reduce(Value inputView, DistributedTensorType outputType, AffineMap traversal,
               ComputeOpOptionsAttr options) {
    unsigned rank = outputType.getRank();
    Value empty = CreateEmptyTensorOp::create(builder, inputView.getLoc(), outputType,
                                              sliceBegins(rank), builder.getI32ArrayAttr({1, 1}),
                                              sliceEnds(outputType.getShape()));
    return StaticUnaryComputeOpOp::create(
        builder, inputView.getLoc(), outputType, inputView, view(empty, traversal), ValueRange{},
        options, traversal, builder.getArrayAttr({}), CustomTilingOptionsAttr{}, ArrayAttr{},
        VexInfoAttr{});
  }

  Value reshape(Value input, DistributedTensorType outputType) {
    auto inputShape = cast<DistributedTensorType>(input.getType()).getShape();
    auto outputShape = outputType.getShape();
    return ReshapeOpOp::create(
        builder, input.getLoc(), outputType, input,
        AffineMapAttr::get(slicing::unitReshapeMap(inputShape, outputShape, context)),
        AffineMapAttr::get(slicing::unitReshapeMap(outputShape, inputShape, context)));
  }

  Value lowerInterpolation(Operation *operation) {
    auto inputType = cast<ShapedType>(operation->getOperand(0).getType());
    auto outputType = tileType(operation->getResult(0).getType());
    bool nearest = operation->getAttrOfType<StringAttr>("algorithm").getValue() == "NEAREST_NEIGHBOR";
    StringRef method = operation->getAttrOfType<StringAttr>("stride_method").getValue();
    SmallVector<AffineExpr> dimensions;
    for (unsigned axis = 0; axis < 6; ++axis)
      dimensions.push_back(builder.getAffineDimExpr(axis));

    SmallVector<StartOffsetAndStrideAttr, 2> coordinates;
    SmallVector<AffineExpr> source{dimensions[0]};
    for (unsigned axis = 1; axis <= 2; ++axis) {
      InterpolationAxis parameters = *interpolationAxis(
          inputType.getDimSize(axis), outputType.getShape()[axis], nearest, method);
      coordinates.push_back(StartOffsetAndStrideAttr::get(
          context, uint32_t(parameters.startOffset), uint32_t(parameters.stride), 16));
      source.push_back((dimensions[axis] * parameters.stride + parameters.startOffset)
                           .floorDiv(1 << 16) +
                       dimensions[axis + 2]);
    }
    source.push_back(dimensions[5]);

    auto domain = builder.getI32ArrayAttr({int32_t(outputType.getShape()[0]),
                                           int32_t(outputType.getShape()[1]),
                                           int32_t(outputType.getShape()[2]), 2, 2,
                                           int32_t(outputType.getShape()[3])});
    auto interpolation = InterpolateHardwareOp::create(
        builder, operation->getLoc(), outputType,
        redistribute(values.lookup(operation->getOperand(0)), DistributedMemorySpace::TileMemory),
        InterpolateMethodAttr::get(context, nearest ? InterpolateMethodKind::NearestNeighbor
                                                    : InterpolateMethodKind::Bilinear),
        coordinates[1], coordinates[0], AffineMapAttr{}, ArrayAttr{}, AffineMapAttr{},
        builder.getI32ArrayAttr({-1, -1, -1, 2, 2, -1}));
    interpolation->setAttr("operand_traversals",
                           builder.getArrayAttr(AffineMapAttr::get(
                               AffineMap::get(6, 0, source, context))));
    interpolation->setAttr(
        "result_traversals",
        builder.getArrayAttr(AffineMapAttr::get(AffineMap::get(
            6, 0, {dimensions[0], dimensions[1], dimensions[2], dimensions[5]}, context))));
    interpolation->setAttr("traversal_domain", domain);
    return interpolation;
  }

  Value lowerPooling(Operation *operation) {
    bool maximum = operation->getAttrOfType<dwc::PoolAttr>("pool").getValue() == dwc::Pool::Max;
    auto integer = [&](StringRef name) {
      return operation->getAttrOfType<IntegerAttr>(name).getInt();
    };
    int64_t kernelY = integer("kernel_y_dim"), kernelX = integer("kernel_x_dim");
    auto outputType = tileType(operation->getResult(0).getType());
    SmallVector<AffineExpr> dimensions;
    for (unsigned axis = 0; axis < 6; ++axis)
      dimensions.push_back(builder.getAffineDimExpr(axis));
    auto input = AffineMap::get(6, 0,
                                {dimensions[0], dimensions[1] * integer("y_stride") + dimensions[3],
                                 dimensions[2] * integer("x_stride") + dimensions[4],
                                 dimensions[5]},
                                context);
    auto traversal = AffineMap::get(
        6, 0, {dimensions[0], dimensions[1], dimensions[2], dimensions[5]}, context);
    ComputeOpOptionsAttr options = computeOptions(
        InnerOperationKind::Unary, maximum ? LinearFunctionKind::Max : LinearFunctionKind::Add, {},
        {}, NluFunctionKind::Linear, builder.getF32FloatAttr(maximum ? -0.0f : 0.0f), std::nullopt,
        true, maximum ? 1.0f : 1.0f / float(kernelY * kernelX));
    Value empty = CreateEmptyTensorOp::create(builder, operation->getLoc(), outputType,
                                              sliceBegins(4), builder.getI32ArrayAttr({1, 1}),
                                              sliceEnds(outputType.getShape()));
    return StaticUnaryComputeOpOp::create(
        builder, operation->getLoc(), outputType,
        view(tileOperand(values.lookup(operation->getOperand(0))), input), view(empty, traversal),
        ValueRange{}, options, traversal, builder.getArrayAttr({}), CustomTilingOptionsAttr{},
        builder.getI32ArrayAttr({-1, -1, -1, int32_t(kernelY), int32_t(kernelX), -1}),
        VexInfoAttr{});
  }

  DistributedTensorType tileType(Type type) {
    auto tensor = cast<ShapedType>(type);
    return DistributedTensorType::get(context, tensor.getShape(), tensor.getElementType(),
                                      DistributedMemorySpace::TileMemory);
  }

  AffineMap sliceBegins(unsigned rank) {
    return AffineMap::get(2, 0, SmallVector<AffineExpr>(rank, builder.getAffineConstantExpr(0)),
                          context);
  }

  AffineMap sliceEnds(ArrayRef<int64_t> shape) {
    SmallVector<AffineExpr> ends;
    for (int64_t extent : shape)
      ends.push_back(builder.getAffineConstantExpr(extent - 1));
    return AffineMap::get(2, 0, ends, context);
  }

  // Depthwise and transposed filters are stored in the rank their compute traversal reads, a
  // pure reshape of the semantic HWC and HWCF layouts.
  ElementsAttr parameterContents(Value source, ElementsAttr value) {
    auto shape = cast<ShapedType>(value.getType()).getShape();
    SmallVector<int64_t> reshaped;
    for (Operation *user : source.getUsers()) {
      if (isa<dwc::DepthwiseConvolutionOp>(user))
        reshaped = {shape[0], shape[1], 1, 1, shape[2]};
      if (isa<dwc::TransposedConvolutionOp>(user))
        reshaped = {1, shape[0], 1, shape[1], shape[2], shape[3]};
    }

    if (reshaped.empty())
      return value;
    auto type = RankedTensorType::get(reshaped, cast<ShapedType>(value.getType()).getElementType());
    if (auto dense = dyn_cast<DenseElementsAttr>(value))
      return dense.reshape(type);
    auto resource = cast<DenseResourceElementsAttr>(value);
    return DenseResourceElementsAttr::get(type, resource.getRawHandle());
  }

  // SDK stage 255 reads parameters and redistributed tensors in place. Every other tile tensor
  // is redistributed again before a compute reads it.
  Value tileOperand(Value storage) {
    if (isa_and_present<FillOp, RedistributeOp>(storage.getDefiningOp()))
      return storage;
    return redistribute(storage, DistributedMemorySpace::TileMemory);
  }

  Value redistribute(Value input, DistributedMemorySpace memory) {
    auto inputType = cast<ShapedType>(input.getType());
    auto outputType = DistributedTensorType::get(context, inputType.getShape(),
                                                 inputType.getElementType(), memory);
    return RedistributeOp::create(builder, input.getLoc(), outputType, input, Value{},
                                  MappingAttr{}, sliceBegins(inputType.getRank()),
                                  builder.getI32ArrayAttr({1, 1}), sliceEnds(inputType.getShape()),
                                  IntegerAttr{});
  }

  Value view(Value storage, AffineMap traversal) {
    auto type = cast<DistributedTensorType>(storage.getType());
    auto viewType = DistributedViewType::get(context, type.getShape(), type.getElementType(),
                                             type.getMemorySpace());
    auto created = DistributedCreateViewOp::create(builder, storage.getLoc(), viewType, storage,
                                                   AffineMapAttr{}, AffineMapAttr{});
    created->setAttr("traversal", AffineMapAttr::get(traversal));
    return created;
  }

  Value view(Value storage, ArrayRef<int64_t> iterationShape) {
    auto type = cast<DistributedTensorType>(storage.getType());
    SmallVector<AffineExpr> coordinates;
    unsigned leading = iterationShape.size() - type.getRank();

    for (auto [axis, extent] : llvm::enumerate(type.getShape())) {
      bool broadcast = type.getShape() != iterationShape && extent == 1;
      coordinates.push_back(broadcast ? builder.getAffineConstantExpr(0)
                                      : builder.getAffineDimExpr(leading + axis));
    }

    return view(storage, AffineMap::get(iterationShape.size(), 0, coordinates, context));
  }

  ComputeOpOptionsAttr computeOptions(InnerOperationKind inner, LinearFunctionKind linear,
                                      FloatAttr lower = {}, FloatAttr upper = {},
                                      NluFunctionKind nlu = NluFunctionKind::Linear,
                                      FloatAttr bias = {},
                                      std::optional<ComputeTypeHintKind> type = std::nullopt,
                                      bool reductionHints = false, float scale = 1.0f) {
    if (!lower) {
      lower = builder.getF32FloatAttr(-std::numeric_limits<float>::infinity());
      upper = builder.getF32FloatAttr(std::numeric_limits<float>::infinity());
    }

    std::optional<NluE8M0RoundingKind> rounding;
    std::optional<ComputeLoweringHintKind> lowering;
    if (reductionHints) {
      rounding = NluE8M0RoundingKind::None;
      lowering = ComputeLoweringHintKind::None;
      type = ComputeTypeHintKind::None;
    }

    return ComputeOpOptionsAttr::get(
        context, inner, {}, {}, builder.getF32FloatAttr(scale), {},
        bias ? bias : builder.getF32FloatAttr(0), lower, upper, {}, linear, {}, {},
        builder.getI32IntegerAttr(0), nlu, NluPreprocessKind::None, NluPredicateKind::None,
        NluPredicateKind::None, {}, false, {}, {}, {}, {}, rounding, lowering, type);
  }

  OpBuilder builder;
  MLIRContext *context;
  IRMapping values;
  llvm::DenseMap<Value, std::array<Value, 3>> constantValues;
};

// SDK stage 255 orders each compute anchor's operand producers just before
// it: parameter fills first, then the operand chains as lhs, rhs, auxiliary
// operands and destination, each depth first. Tile reshapes of a result, and
// the identity output view of a transposed convolution, follow their
// producer directly.
void orderLikeSdk(Block &block) {
  auto anchor = [](Operation *operation) {
    return isa<StaticComputeOpOp, StaticUnaryComputeOpOp, CopyOpOp, InterpolateHardwareOp>(
        operation);
  };
  auto parameter = [](Operation *operation) {
    auto fill = dyn_cast_if_present<FillOp>(operation);
    return fill && fill.getConstTypeAttr() &&
           fill.getConstTypeAttr().getValue() == ConstKind::Parameter;
  };
  auto usersInOrder = [](Operation *operation) {
    SmallVector<Operation *> users(operation->getUsers());
    llvm::sort(users, [](Operation *left, Operation *right) { return left->isBeforeInBlock(right); });
    users.erase(std::unique(users.begin(), users.end()), users.end());
    return users;
  };

  SmallVector<Operation *> original;
  for (Operation &operation : block.without_terminator())
    original.push_back(&operation);
  llvm::SmallPtrSet<Operation *, 32> done;
  SmallVector<Operation *> order;
  auto emit = [&](Operation *operation, auto &&self) -> void {
    if (!operation || operation->getBlock() != &block || !done.insert(operation).second)
      return;
    for (Value operand : operation->getOperands())
      self(operand.getDefiningOp(), self);
    order.push_back(operation);
  };

  for (Operation *operation : original) {
    if (!anchor(operation) || done.contains(operation))
      continue;
    SmallVector<Operation *> producers;
    for (Value operand : operation->getOperands())
      producers.push_back(operand.getDefiningOp());
    for (Operation *producer : producers) {
      if (!producer)
        continue;
      if (parameter(producer))
        emit(producer, emit);
      for (Value operand : producer->getOperands())
        if (parameter(operand.getDefiningOp()))
          emit(operand.getDefiningOp(), emit);
    }
    auto roles = llvm::to_vector(llvm::seq<unsigned>(0, producers.size()));
    if (isa<StaticComputeOpOp>(operation) && producers.size() >= 3) {
      roles = {0, 1};
      llvm::append_range(roles, llvm::seq<unsigned>(3, producers.size()));
      roles.push_back(2);
    } else if (isa<StaticUnaryComputeOpOp>(operation) && producers.size() >= 2) {
      roles = {0};
      llvm::append_range(roles, llvm::seq<unsigned>(2, producers.size()));
      roles.push_back(1);
    }
    for (unsigned role : roles)
      emit(producers[role], emit);
    emit(operation, emit);

    auto options = operation->getAttrOfType<ComputeOpOptionsAttr>("compute");
    bool transposed = options && options.getComputeTypeHint() == ComputeTypeHintKind::TransposedConv;
    for (Operation *user : usersInOrder(operation)) {
      auto reshape = dyn_cast<ReshapeOpOp>(user);
      if (reshape && cast<DistributedTensorType>(reshape.getType()).getMemorySpace() ==
                         DistributedMemorySpace::TileMemory)
        emit(user, emit);
      if (!transposed || !isa<RedistributeOp>(user))
        continue;
      emit(user, emit);
      for (Operation *view : usersInOrder(user))
        if (isa<DistributedCreateViewOp>(view))
          emit(view, emit);
    }
  }
  for (Operation *operation : original)
    emit(operation, emit);

  Operation *terminator = block.getTerminator();
  for (Operation *operation : order)
    operation->moveBefore(terminator);
}

class LowerSemanticToDistributedPass
    : public PassWrapper<LowerSemanticToDistributedPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerSemanticToDistributedPass)

  StringRef getArgument() const final { return "darwinn-lower-semantic-to-distributed"; }

  StringRef getDescription() const final {
    return "Lower precision-explicit semantic arithmetic to distributed "
           "compute";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<DarwinnDialect, dwc::DwcDialect, func::FuncDialect, arith::ArithDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();
    func::FuncOp function;

    for (Operation &operation : *module.getBody()) {
      auto candidate = dyn_cast<func::FuncOp>(operation);
      if (!candidate || function) {
        (void)unsupported(&operation, "the module must contain exactly one function");
        return signalPassFailure();
      }
      function = candidate;
    }

    if (!function) {
      (void)unsupported(module, "the module must contain exactly one function");
      return signalPassFailure();
    }

    WalkResult structure = function.walk([&](Operation *operation) {
      if (isa<dwc::ReshapeOp, dwc::TransposeOp>(operation) && failed(validateStructure(operation)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (structure.wasInterrupted() || failed(verify(function)))
      return signalPassFailure();

    auto boundary = validateFunction(function);
    if (failed(boundary))
      return signalPassFailure();

    OwningOpRef<func::FuncOp> output(cast<func::FuncOp>(function->clone()));
    output->getBody().getBlocks().clear();
    SmallVector<Type> inputs;
    SmallVector<Type> outputs;
    auto hostType = [&](Type type) -> Type {
      auto tensor = cast<RankedTensorType>(type);
      return DistributedTensorType::get(module.getContext(), tensor.getShape(),
                                        tensor.getElementType(),
                                        DistributedMemorySpace::HostMemory);
    };

    llvm::transform(function.getArgumentTypes(), std::back_inserter(inputs), hostType);
    llvm::transform(function.getResultTypes(), std::back_inserter(outputs), hostType);
    output->setType(FunctionType::get(module.getContext(), inputs, outputs));
    output->getOperation()->removeAttr("tf.entry_function");
    SemanticLowering lowering(*output);
    lowering.lower(function, *output, *boundary);
    orderLikeSdk(output->getBody().front());

    if (failed(verify(*output)))
      return signalPassFailure();

    function->setAttrs(output->getOperation()->getAttrs());
    function.getBody().takeBody(output->getBody());
  }
};

}

std::unique_ptr<Pass> mlir::darwinn::createLowerSemanticToDistributedPass() {
  return std::make_unique<LowerSemanticToDistributedPass>();
}

void mlir::darwinn::registerLowerSemanticToDistributedPass() {
  PassRegistration<LowerSemanticToDistributedPass>();
}
