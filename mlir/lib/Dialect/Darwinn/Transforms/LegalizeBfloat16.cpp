#include "mlir/Dialect/Darwinn/Transforms/LegalizeBfloat16.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Darwinn/IR/DwcOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <optional>

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/Darwinn/IR/DwcAttributes.h.inc"

using namespace mlir;
using namespace mlir::darwinn;

namespace {

LogicalResult unsupported(Operation *operation, const Twine &reason) {
  return operation->emitOpError() << "cannot legalize bfloat16 because " << reason;
}

bool staticF32(Type type) {
  auto tensor = dyn_cast<RankedTensorType>(type);
  return tensor && tensor.hasStaticShape() && tensor.getRank() >= 1 &&
         tensor.getRank() <= 4 && tensor.getElementType().isF32() &&
         llvm::all_of(tensor.getShape(), [](int64_t extent) { return extent > 0; });
}

std::optional<dwc::CwiseOpType> binaryKind(linalg::GenericOp operation) {
  if (operation.getNumDpsInputs() != 2 || operation.getNumDpsInits() != 1 ||
      operation->getNumResults() != 1 || !operation.hasPureTensorSemantics() ||
      !staticF32(operation.getResult(0).getType()) ||
      operation.getNumParallelLoops() != operation.getNumLoops() ||
      !llvm::hasSingleElement(operation.getRegion()))
    return std::nullopt;

  auto output = cast<RankedTensorType>(operation.getResult(0).getType());
  auto maps = operation.getIndexingMapsArray();

  if (maps.size() != 3 || !maps[2].isIdentity() ||
      maps[2].getNumDims() != output.getRank() ||
      operation.getDpsInits()[0].getType() != output ||
      !operation.getDpsInits()[0].getDefiningOp<tensor::EmptyOp>())
    return std::nullopt;

  for (unsigned index = 0; index < 2; ++index) {
    auto input = dyn_cast<RankedTensorType>(operation.getDpsInputs()[index].getType());

    if (!input || !staticF32(input) || input.getRank() > output.getRank() ||
        maps[index].getNumSymbols() != 0 || maps[index].getNumDims() != output.getRank() ||
        maps[index].getNumResults() != input.getRank())
      return std::nullopt;

    unsigned offset = output.getRank() - input.getRank();

    for (unsigned axis = 0; axis < input.getRank(); ++axis) {
      int64_t extent = input.getDimSize(axis);
      AffineExpr expected = extent == 1
          ? getAffineConstantExpr(0, operation.getContext())
          : getAffineDimExpr(offset + axis, operation.getContext());

      if ((extent != 1 && extent != output.getDimSize(offset + axis)) ||
          maps[index].getResult(axis) != expected)
        return std::nullopt;
    }
  }

  for (int64_t axis = 0; axis < output.getRank(); ++axis) {
    int64_t extent = 1;
    for (Value input : operation.getDpsInputs()) {
      auto type = cast<RankedTensorType>(input.getType());
      int64_t inputAxis = axis - (output.getRank() - type.getRank());
      if (inputAxis >= 0)
        extent = std::max(extent, type.getDimSize(inputAxis));
    }
    if (extent != output.getDimSize(axis))
      return std::nullopt;
  }

  Block &body = operation.getRegion().front();

  if (body.getNumArguments() != 3 || body.getOperations().size() != 2 ||
      !body.getArgument(2).use_empty())
    return std::nullopt;

  Operation &arithmetic = body.front();
  auto yield = dyn_cast<linalg::YieldOp>(body.back());

  if (!yield || yield.getNumOperands() != 1 || arithmetic.getNumOperands() != 2 ||
      arithmetic.getNumResults() != 1 || arithmetic.getOperand(0) != body.getArgument(0) ||
      arithmetic.getOperand(1) != body.getArgument(1) ||
      yield.getOperand(0) != arithmetic.getResult(0) ||
      arithmetic.getResult(0).getType() != Float32Type::get(operation.getContext()))
    return std::nullopt;

  if (auto flags = arithmetic.getAttrOfType<arith::FastMathFlagsAttr>("fastmath")) {
    if (flags.getValue() != arith::FastMathFlags::none)
      return std::nullopt;
  }

  if (isa<arith::AddFOp>(arithmetic))
    return dwc::CwiseOpType::Add;
  if (isa<arith::SubFOp>(arithmetic))
    return dwc::CwiseOpType::Subtract;
  if (isa<arith::MulFOp>(arithmetic))
    return dwc::CwiseOpType::Multiply;
  return std::nullopt;
}

bool isCompute(Operation *operation) {
  return isa<dwc::ConvolutionOp, dwc::DepthwiseConvolutionOp,
             dwc::TransposedConvolutionOp, dwc::CwiseOp, dwc::ReductionOp,
             linalg::GenericOp>(operation);
}

std::optional<int32_t> sumAxis(linalg::GenericOp operation) {
  if (operation.getNumDpsInputs() != 1 || operation.getNumDpsInits() != 1 ||
      operation->getNumResults() != 1 || !operation.hasPureTensorSemantics() ||
      !staticF32(operation.getResult(0).getType()) ||
      !staticF32(operation.getDpsInputs()[0].getType()) ||
      !llvm::hasSingleElement(operation.getRegion()))
    return std::nullopt;

  auto input = cast<RankedTensorType>(operation.getDpsInputs()[0].getType());
  auto output = cast<RankedTensorType>(operation.getResult(0).getType());
  auto maps = operation.getIndexingMapsArray();
  auto iterators = operation.getIteratorTypesArray();

  if (input.getRank() != output.getRank() || maps.size() != 2 ||
      !maps[0].isIdentity() || maps[0].getNumDims() != input.getRank() ||
      maps[1].getNumDims() != input.getRank() || maps[1].getNumSymbols() != 0 ||
      maps[1].getNumResults() != output.getRank() ||
      iterators.size() != static_cast<size_t>(input.getRank()))
    return std::nullopt;

  std::optional<int32_t> reduced;

  for (int32_t axis = 0; axis < input.getRank(); ++axis) {
    if (iterators[axis] == utils::IteratorType::reduction) {
      if (reduced || output.getDimSize(axis) != 1 ||
          maps[1].getResult(axis) != getAffineConstantExpr(0, operation.getContext()))
        return std::nullopt;
      reduced = axis;
    } else if (iterators[axis] != utils::IteratorType::parallel ||
               output.getDimSize(axis) != input.getDimSize(axis) ||
               maps[1].getResult(axis) != getAffineDimExpr(axis, operation.getContext())) {
      return std::nullopt;
    }
  }

  auto fill = operation.getDpsInits()[0].getDefiningOp<linalg::FillOp>();
  if (!reduced || !fill || fill.getNumDpsInputs() != 1 || fill.getNumDpsInits() != 1 ||
      fill->getNumResults() != 1 || fill.getResult(0).getType() != output ||
      !fill.getDpsInits()[0].getDefiningOp<tensor::EmptyOp>())
    return std::nullopt;

  auto zero = fill.getDpsInputs()[0].getDefiningOp<arith::ConstantOp>();
  auto scalar = zero ? dyn_cast<FloatAttr>(zero.getValue()) : FloatAttr();
  if (!scalar || !scalar.getType().isF32() || !scalar.getValue().isZero() ||
      scalar.getValue().isNegative())
    return std::nullopt;

  Block &body = operation.getRegion().front();
  if (body.getNumArguments() != 2 || body.getOperations().size() != 2)
    return std::nullopt;

  auto add = dyn_cast<arith::AddFOp>(body.front());
  auto yield = dyn_cast<linalg::YieldOp>(body.back());
  if (!add || !yield || yield.getNumOperands() != 1 ||
      add.getLhs() != body.getArgument(0) || add.getRhs() != body.getArgument(1) ||
      yield.getOperand(0) != add.getResult() ||
      add.getFastmath() != arith::FastMathFlags::none)
    return std::nullopt;

  return reduced;
}

LogicalResult recoverSemanticOperations(func::FuncOp function) {
  if (function.isExternal() || !llvm::hasSingleElement(function.getBody()))
    return unsupported(function, "a single block is required");

  for (Operation &operation : llvm::make_early_inc_range(function.getBody().front())) {
    OpBuilder builder(&operation);
    if (isa<tensor::CollapseShapeOp, tensor::ExpandShapeOp, tensor::CastOp>(operation)) {
      if (operation.getNumOperands() != 1 || operation.getNumResults() != 1 ||
          !staticF32(operation.getOperand(0).getType()) ||
          !staticF32(operation.getResult(0).getType()))
        return unsupported(&operation, "only static floating reshape operations are supported");

      Value input = operation.getOperand(0);
      auto inputType = cast<RankedTensorType>(input.getType());
      auto resultType = cast<RankedTensorType>(operation.getResult(0).getType());
      if (inputType.getNumElements() != resultType.getNumElements())
        return unsupported(&operation, "reshape element counts must agree");

      auto preceding = input.getDefiningOp<dwc::ReshapeOp>();
      bool collapsePreceding = preceding && input.hasOneUse();
      if (collapsePreceding)
        input = preceding.getOperand(0);

      OperationState state(operation.getLoc(), dwc::ReshapeOp::getOperationName());
      state.addOperands(input);
      state.addTypes(resultType);
      Operation *result = builder.create(state);
      operation.replaceAllUsesWith(result->getResults());
      operation.erase();
      if (collapsePreceding)
        preceding.erase();
      continue;
    }

    if (auto transpose = dyn_cast<linalg::TransposeOp>(operation)) {
      auto input = transpose.getInput();
      auto output = transpose.getInit();
      if (!staticF32(input.getType()) || !staticF32(output.getType()) ||
          !output.getDefiningOp<tensor::EmptyOp>())
        return unsupported(transpose, "transpose requires static tensors and an empty initializer");

      OperationState state(operation.getLoc(), dwc::TransposeOp::getOperationName());
      state.addOperands(input);
      state.addTypes(output.getType());
      SmallVector<int32_t> permutation;
      for (int64_t axis : transpose.getPermutation())
        permutation.push_back(static_cast<int32_t>(axis));
      state.addAttribute("permutation", builder.getI32TensorAttr(permutation));
      Operation *result = builder.create(state);
      operation.replaceAllUsesWith(result->getResults());
      operation.erase();
      continue;
    }

    auto generic = dyn_cast<linalg::GenericOp>(operation);
    if (!generic)
      continue;

    if (auto axis = sumAxis(generic)) {
      OperationState state(operation.getLoc(), dwc::ReductionOp::getOperationName());
      state.addOperands(generic.getDpsInputs()[0]);
      state.addTypes(operation.getResult(0).getType());
      state.addAttribute("dimensions", builder.getI32TensorAttr({*axis}));
      state.addAttribute("activation_function", dwc::SimpleActivationFunctionAttr::get(
          builder.getContext(), dwc::SimpleActivationFunction::None));
      state.addAttribute("op_type", dwc::ReductionTypeAttr::get(
          builder.getContext(), dwc::ReductionType::Sum));
      Operation *result = builder.create(state);
      operation.replaceAllUsesWith(result->getResults());
      operation.erase();
      continue;
    }

    if (binaryKind(generic) != dwc::CwiseOpType::Add)
      continue;

    Value input = generic.getDpsInputs()[0];
    auto convolution = input.getDefiningOp<dwc::TransposedConvolutionOp>();
    Value bias = generic.getDpsInputs()[1];
    auto biasConstant = bias.getDefiningOp<arith::ConstantOp>();
    auto biasType = cast<RankedTensorType>(bias.getType());
    auto outputType = cast<RankedTensorType>(generic.getResult(0).getType());
    if (!convolution || !input.hasOneUse() || convolution->getNumOperands() != 2 ||
        !biasConstant || !isa<DenseFPElementsAttr>(biasConstant.getValue()) ||
        outputType.getRank() != 4 || biasType.getRank() != 1 ||
        biasType.getDimSize(0) != outputType.getDimSize(3) || input.getType() != outputType)
      continue;

    auto activation = convolution->getAttrOfType<dwc::ActivationFunctionAttr>("activation_function");
    if (!activation || activation.getValue() != dwc::ActivationFunction::None ||
        convolution->hasAttr("activation_clip_min") || convolution->hasAttr("activation_clip_max"))
      continue;

    OperationState state(operation.getLoc(), dwc::TransposedConvolutionOp::getOperationName());
    state.addOperands(convolution->getOperands());
    state.addOperands(bias);
    state.addTypes(outputType);
    state.addAttributes(convolution->getAttrs());
    Operation *result = builder.create(state);
    operation.replaceAllUsesWith(result->getResults());
    operation.erase();
    convolution.erase();
  }

  for (Operation &operation : llvm::make_early_inc_range(
           llvm::reverse(function.getBody().front()))) {
    if (operation.use_empty() && isa<arith::ConstantOp, tensor::EmptyOp, linalg::FillOp>(operation))
      operation.erase();
  }

  return success();
}

LogicalResult validateFunction(func::FuncOp function) {
  if (function.isExternal() || !llvm::hasSingleElement(function.getBody()) ||
      function.getNumArguments() == 0 || function.getNumResults() == 0 ||
      !llvm::all_of(function.getArgumentTypes(), staticF32) ||
      !llvm::all_of(function.getResultTypes(), staticF32))
    return unsupported(function, "a single block with positive static f32 tensor "
                                 "arguments and results is required");

  for (Operation &operation : function.getBody().front()) {
    if (auto constant = dyn_cast<arith::ConstantOp>(operation)) {
      if (auto dense = dyn_cast<DenseFPElementsAttr>(constant.getValue())) {
        if (staticF32(dense.getType()))
          continue;
      }

      auto scalar = dyn_cast<FloatAttr>(constant.getValue());
      if (scalar && scalar.getType().isF32() && scalar.getValue().isZero() &&
          !scalar.getValue().isNegative())
        continue;
      return unsupported(&operation, "only dense f32 tensors and positive scalar zero are supported");
    }

    if (auto empty = dyn_cast<tensor::EmptyOp>(operation)) {
      if (!staticF32(empty.getType()) || llvm::any_of(empty->getUsers(), [](Operation *user) {
            return !isa<linalg::GenericOp>(user);
          }))
        return unsupported(&operation, "empty tensors must initialize supported binary generics");
      continue;
    }

    if (auto binary = dyn_cast<linalg::GenericOp>(operation)) {
      if (!binaryKind(binary))
        return unsupported(&operation, "only exact broadcast ADD, SUBTRACT and MULTIPLY regions are supported");
      continue;
    }

    if (auto pad = dyn_cast<tensor::PadOp>(operation)) {
      auto value = pad.getConstantPaddingValue();
      auto constant = value ? value.getDefiningOp<arith::ConstantOp>() : arith::ConstantOp();
      auto scalar = constant ? dyn_cast<FloatAttr>(constant.getValue()) : FloatAttr();

      if (!staticF32(pad.getSourceType()) || !staticF32(pad.getResultType()) ||
          !pad.getLow().empty() || !pad.getHigh().empty() || !scalar ||
          !scalar.getValue().isZero() || scalar.getValue().isNegative() ||
          pad.getRegion().front().getOperations().size() != 1)
        return unsupported(&operation, "only static zero padding is supported");
      continue;
    }

    if (isa<dwc::ConvolutionOp, dwc::DepthwiseConvolutionOp,
            dwc::TransposedConvolutionOp>(operation)) {
      if (!staticF32(operation.getOperand(0).getType()) ||
          !staticF32(operation.getOperand(1).getType()) ||
          !staticF32(operation.getResult(0).getType()))
        return unsupported(&operation, "convolutions must have f32 input, weights and output");

      auto activation = operation.getAttrOfType<dwc::ActivationFunctionAttr>("activation_function");
      auto cell = operation.getAttrOfType<dwc::CellOperationAttr>("cell_operation");
      auto padding = operation.getAttrOfType<dwc::PaddingAttr>("pad");
      if (!activation || !cell || !padding || cell.getValue() != dwc::CellOperation::Mac ||
          padding.getValue() != dwc::Padding::None ||
          !llvm::is_contained({dwc::ActivationFunction::None, dwc::ActivationFunction::Relu,
                               dwc::ActivationFunction::Logistic,
                               dwc::ActivationFunction::HardSwish},
                              activation.getValue()))
        return unsupported(&operation, "convolutions require MAC, explicit padding and NONE, RELU, "
                                       "LOGISTIC or HARD_SWISH activation");

      if (operation.getNumOperands() == 3) {
        auto bias = operation.getOperand(2).getDefiningOp<arith::ConstantOp>();
        if (!bias || !isa<DenseFPElementsAttr>(bias.getValue()))
          return unsupported(&operation, "convolution bias must be a dense f32 constant");
      }
      continue;
    }

    if (isa<dwc::ReshapeOp, dwc::TransposeOp>(operation)) {
      if (operation.getNumOperands() != 1 || operation.getNumResults() != 1 ||
          !staticF32(operation.getOperand(0).getType()) ||
          !staticF32(operation.getResult(0).getType()))
        return unsupported(&operation, "structural operations require static f32 input and output");

      auto input = cast<RankedTensorType>(operation.getOperand(0).getType());
      auto output = cast<RankedTensorType>(operation.getResult(0).getType());
      if (input.getNumElements() != output.getNumElements())
        return unsupported(&operation, "structural operations must preserve the element count");

      if (isa<dwc::TransposeOp>(operation)) {
        auto permutation = operation.getAttrOfType<DenseIntElementsAttr>("permutation");
        if (!permutation || input.getRank() != output.getRank() ||
            permutation.getNumElements() != input.getRank())
          return unsupported(&operation, "transpose permutation must match input and output rank");

        for (auto [axis, value] : llvm::enumerate(permutation.getValues<APInt>())) {
          int64_t sourceAxis = value.getSExtValue();
          if (sourceAxis < 0 || sourceAxis >= input.getRank() ||
              output.getDimSize(axis) != input.getDimSize(sourceAxis))
            return unsupported(&operation, "transpose output shape must follow its permutation");
        }
      }
      continue;
    }

    if (isa<dwc::ReductionOp>(operation)) {
      auto input = operation.getNumOperands() == 1
          ? dyn_cast<RankedTensorType>(operation.getOperand(0).getType()) : RankedTensorType();
      auto output = operation.getNumResults() == 1
          ? dyn_cast<RankedTensorType>(operation.getResult(0).getType()) : RankedTensorType();
      auto dimensions = operation.getAttrOfType<DenseIntElementsAttr>("dimensions");
      auto kind = operation.getAttrOfType<dwc::ReductionTypeAttr>("op_type");
      auto activation = operation.getAttrOfType<dwc::SimpleActivationFunctionAttr>("activation_function");
      if (!input || !output || !staticF32(input) || !staticF32(output) ||
          input.getRank() != output.getRank() || !dimensions || dimensions.getNumElements() != 1 ||
          !kind || kind.getValue() != dwc::ReductionType::Sum || !activation ||
          activation.getValue() != dwc::SimpleActivationFunction::None)
        return unsupported(&operation, "only static keep-dimensions SUM with NONE activation is supported");

      int64_t axis = (*dimensions.getValues<APInt>().begin()).getSExtValue();
      if (axis < 0)
        axis += input.getRank();
      if (axis < 0 || axis >= input.getRank())
        return unsupported(&operation, "reduction axis is outside the input rank");
      for (int64_t index = 0; index < input.getRank(); ++index) {
        if (output.getDimSize(index) != (index == axis ? 1 : input.getDimSize(index)))
          return unsupported(&operation, "reduction output shape does not preserve unreduced dimensions");
      }
      continue;
    }

    if (isa<dwc::ClassifierOp>(operation)) {
      auto axis = operation.getAttrOfType<IntegerAttr>("axis");
      auto beta = operation.getAttrOfType<FloatAttr>("beta");
      auto kind = operation.getAttrOfType<dwc::ClassificationTypeAttr>("op_type");
      if (operation.getNumOperands() != 1 || operation.getNumResults() != 1 ||
          !staticF32(operation.getOperand(0).getType()) ||
          operation.getOperand(0).getType() != operation.getResult(0).getType() ||
          !axis || axis.getInt() != -1 || !beta || beta.getValueAsDouble() != 1.0 ||
          !kind || kind.getValue() != dwc::ClassificationType::Softmax)
        return unsupported(&operation, "softmax requires matching static f32 tensors, axis -1 and beta 1");
      continue;
    }

    if (isa<dwc::ImageInterpolationOp>(operation)) {
      auto algorithm = operation.getAttrOfType<StringAttr>("algorithm");
      auto stride = operation.getAttrOfType<StringAttr>("stride_method");
      if (operation.getNumOperands() != 1 || operation.getNumResults() != 1 ||
          !staticF32(operation.getOperand(0).getType()) ||
          !staticF32(operation.getResult(0).getType()) || !algorithm || !stride ||
          (algorithm.getValue() != "BILINEAR" && algorithm.getValue() != "NEAREST_NEIGHBOR") ||
          (stride.getValue() != "TENSORFLOW_DEFAULT" &&
           stride.getValue() != "TENSORFLOW_ALIGN_CORNERS" &&
           stride.getValue() != "HALF_PIXEL_CENTERS"))
        return unsupported(&operation, "resize requires static f32 tensors and supported coordinate attributes");
      continue;
    }

    if (auto cwise = dyn_cast<dwc::CwiseOp>(operation)) {
      auto kind = operation.getAttrOfType<dwc::CwiseOpTypeAttr>("op_type");
      auto activation = operation.getAttrOfType<dwc::ActivationFunctionAttr>("activation_function");
      if (operation.getNumResults() != 1 || !staticF32(operation.getResult(0).getType()) ||
          !llvm::all_of(operation.getOperandTypes(), staticF32) || !kind || !activation ||
          activation.getValue() != dwc::ActivationFunction::None ||
          (kind.getValue() != dwc::CwiseOpType::Add &&
           kind.getValue() != dwc::CwiseOpType::Subtract &&
           kind.getValue() != dwc::CwiseOpType::Multiply))
        return unsupported(cwise, "only f32 ADD, SUBTRACT and MULTIPLY with NONE activation are supported");

      auto output = cast<RankedTensorType>(operation.getResult(0).getType());
      for (Type operand : operation.getOperandTypes()) {
        if (cast<RankedTensorType>(operand).getRank() > output.getRank())
          return unsupported(cwise, "operand rank exceeds result rank");
      }

      for (int64_t axis = 0; axis < output.getRank(); ++axis) {
        int64_t extent = 1;
        for (Type operand : operation.getOperandTypes()) {
          auto input = cast<RankedTensorType>(operand);
          int64_t inputAxis = axis - (output.getRank() - input.getRank());
          if (inputAxis < 0)
            continue;
          int64_t dimension = input.getDimSize(inputAxis);
          if (dimension != 1 && dimension != output.getDimSize(axis))
            return unsupported(cwise, "operand shape cannot broadcast to the result");
          extent = std::max(extent, dimension);
        }
        if (extent != output.getDimSize(axis))
          return unsupported(cwise, "result shape does not match binary broadcasting");
      }
      continue;
    }

    if (auto returned = dyn_cast<func::ReturnOp>(operation)) {
      for (Value value : returned.getOperands()) {
        if (!value.getDefiningOp() ||
            (!isCompute(value.getDefiningOp()) &&
             !isa<dwc::ReshapeOp, dwc::TransposeOp, dwc::ClassifierOp,
                  dwc::ImageInterpolationOp>(value.getDefiningOp())))
          return unsupported(returned, "each return must have a supported tensor producer");
      }
      continue;
    }

    return unsupported(&operation, "the operation has no established precision policy");
  }

  if (llvm::none_of(function.getBody().front(), [](Operation &operation) {
        return isCompute(&operation) ||
            isa<dwc::ClassifierOp, dwc::ImageInterpolationOp, dwc::TransposeOp>(operation);
      }))
    return unsupported(function, "a reshape-only function has no established NPU truncation policy");

  return success();
}

class PrecisionLowering {
public:
  explicit PrecisionLowering(func::FuncOp function) : builder(function.getContext()) {}

  void lower(func::FuncOp source, func::FuncOp target) {
    Block *entry = target.addEntryBlock();
    builder.setInsertionPointToStart(entry);
    OperationState absent(source.getLoc(), dwc::ConstNoneOp::getOperationName());
    absent.addTypes(builder.getNoneType());
    Value noBias = builder.create(absent)->getResult(0);

    for (auto [original, argument] : llvm::zip(source.getArguments(), entry->getArguments())) {
      OperationState cast(source.getLoc(), dwc::RescalingOp::getOperationName());
      cast.addOperands({argument, noBias});
      cast.addTypes(bfloatType(argument.getType()));
      cast.addAttribute("activation_function", noneActivation());
      cast.addAttribute("output_activation_per_z_out_scales", builder.getArrayAttr({}));
      cast.addAttribute("per_z_out_scales_padding", dwc::PerZOutScalePaddingAttr::get(
          builder.getContext(), dwc::PerZOutScalePadding::None));
      values.map(original, builder.create(cast)->getResult(0));
    }

    for (Operation &operation : source.getBody().front()) {
      if (isa<arith::ConstantOp, tensor::EmptyOp>(operation))
        continue;

      if (auto returned = dyn_cast<func::ReturnOp>(operation)) {
        SmallVector<Value> results;
        for (Value value : returned.getOperands())
          results.push_back(values.lookup(value));
        func::ReturnOp::create(builder, operation.getLoc(), results);
        continue;
      }

      if (auto pad = dyn_cast<tensor::PadOp>(operation)) {
        auto type = cast<RankedTensorType>(bfloatType(pad.getType()));
        Value zero = arith::ConstantOp::create(builder, operation.getLoc(),
            builder.getFloatAttr(builder.getBF16Type(), 0.0));
        Value input = converted(pad.getSource(), false);
        auto output = tensor::PadOp::create(builder, operation.getLoc(), type, input,
            pad.getMixedLowPad(), pad.getMixedHighPad(), zero, pad.getNofold());
        values.map(pad.getResult(), output.getResult());
        continue;
      }

      bool terminal = operation.getResult(0).hasOneUse() &&
          isa<func::ReturnOp>(*operation.getResult(0).getUsers().begin());
      bool boundaryCast = terminal && isa<dwc::ReshapeOp, dwc::TransposeOp,
                                          dwc::ImageInterpolationOp>(operation);
      Type resultType = terminal && !boundaryCast ? operation.getResult(0).getType()
                                                : bfloatType(operation.getResult(0).getType());

      if (auto binary = dyn_cast<linalg::GenericOp>(operation)) {
        OperationState state(operation.getLoc(), dwc::CwiseOp::getOperationName());
        state.addOperands({converted(binary.getDpsInputs()[0], false),
                           converted(binary.getDpsInputs()[1], false)});
        state.addTypes(resultType);
        state.addAttribute("op_type", dwc::CwiseOpTypeAttr::get(builder.getContext(),
                                                               *binaryKind(binary)));
        state.addAttribute("activation_function", noneActivation());
        values.map(operation.getResult(0), builder.create(state)->getResult(0));
        continue;
      }

      OperationState state(operation.getLoc(), operation.getName());
      for (auto [index, value] : llvm::enumerate(operation.getOperands()))
        state.addOperands(converted(value, index == 2));
      state.addTypes(resultType);
      state.addAttributes(operation.getAttrs());
      Value result = builder.create(state)->getResult(0);
      if (boundaryCast) {
        OperationState cast(operation.getLoc(), dwc::RescalingOp::getOperationName());
        cast.addOperands({result, noBias});
        cast.addTypes(operation.getResult(0).getType());
        cast.addAttribute("activation_function", noneActivation());
        cast.addAttribute("output_activation_per_z_out_scales", builder.getArrayAttr({}));
        cast.addAttribute("per_z_out_scales_padding", dwc::PerZOutScalePaddingAttr::get(
            builder.getContext(), dwc::PerZOutScalePadding::None));
        result = builder.create(cast)->getResult(0);
      }
      values.map(operation.getResult(0), result);
    }

  }

private:
  Type bfloatType(Type type) {
    auto tensor = cast<RankedTensorType>(type);
    return RankedTensorType::get(tensor.getShape(), builder.getBF16Type(), tensor.getEncoding());
  }

  Attribute noneActivation() {
    return dwc::ActivationFunctionAttr::get(builder.getContext(), dwc::ActivationFunction::None);
  }

  Value converted(Value value, bool bias) {
    IRMapping &mapping = bias ? biases : values;
    if (mapping.contains(value))
      return mapping.lookup(value);

    auto constant = value.getDefiningOp<arith::ConstantOp>();
    auto dense = cast<DenseFPElementsAttr>(constant.getValue());
    SmallVector<APFloat> elements;

    for (APFloat element : dense.getValues<APFloat>()) {
      if (element.isInfinity())
        element = APFloat::getLargest(APFloat::IEEEsingle(), element.isNegative());

      if (!bias) {
        if (element.isNaN()) {
          element = APFloat::getQNaN(APFloat::BFloat(), element.isNegative());
        } else {
          bool losesInformation;
          element.convert(APFloat::BFloat(), APFloat::rmNearestTiesToEven, &losesInformation);
          if (element.isInfinity())
            element = APFloat::getLargest(APFloat::BFloat(), element.isNegative());
        }
      }

      elements.push_back(element);
    }

    Type resultType = bias ? value.getType() : bfloatType(value.getType());
    auto result = DenseFPElementsAttr::get(cast<RankedTensorType>(resultType), elements);
    Value output = arith::ConstantOp::create(builder, constant.getLoc(), result);
    mapping.map(value, output);
    return output;
  }

  OpBuilder builder;
  IRMapping values;
  IRMapping biases;
};

class LegalizeBfloat16Pass
    : public PassWrapper<LegalizeBfloat16Pass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LegalizeBfloat16Pass)

  StringRef getArgument() const final { return "dwc-legalize-bfloat16"; }
  StringRef getDescription() const final {
    return "Assign explicit bfloat16 tensor precision while preserving f32 bias and boundaries";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<dwc::DwcDialect, func::FuncDialect, arith::ArithDialect,
                    linalg::LinalgDialect, tensor::TensorDialect>();
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();
    OwningOpRef<ModuleOp> recovered(cast<ModuleOp>(module->clone()));
    SmallVector<func::FuncOp> functions;

    for (Operation &operation : *recovered->getBody()) {
      auto function = dyn_cast<func::FuncOp>(operation);
      if (!function || failed(recoverSemanticOperations(function)) ||
          failed(validateFunction(function))) {
        if (!function)
          (void)unsupported(&operation, "only functions are accepted in the module");
        return signalPassFailure();
      }
      functions.push_back(function);
    }

    if (failed(verify(*recovered)))
      return signalPassFailure();

    if (functions.empty()) {
      (void)unsupported(module, "the module must contain a function");
      return signalPassFailure();
    }

    SmallVector<OwningOpRef<func::FuncOp>> outputs;
    for (func::FuncOp function : functions) {
      OwningOpRef<func::FuncOp> output(cast<func::FuncOp>(function->clone()));
      output->getBody().getBlocks().clear();
      PrecisionLowering lowering(*output);
      lowering.lower(function, *output);
      if (failed(verify(*output)))
        return signalPassFailure();
      outputs.push_back(std::move(output));
    }

    for (auto [function, output] : llvm::zip(functions, outputs))
      function.getBody().takeBody(output->getBody());

    if (failed(verify(*recovered)))
      return signalPassFailure();
    module.getBodyRegion().takeBody(recovered->getBodyRegion());
  }
};

}

std::unique_ptr<Pass> mlir::darwinn::createLegalizeBfloat16Pass() {
  return std::make_unique<LegalizeBfloat16Pass>();
}

void mlir::darwinn::registerLegalizeBfloat16Pass() {
  PassRegistration<LegalizeBfloat16Pass>();
}
