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
  return isa<dwc::ConvolutionOp, dwc::DepthwiseConvolutionOp, dwc::CwiseOp,
             linalg::GenericOp>(operation);
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

    if (isa<dwc::ConvolutionOp, dwc::DepthwiseConvolutionOp>(operation)) {
      if (!staticF32(operation.getOperand(0).getType()) ||
          !staticF32(operation.getOperand(1).getType()) ||
          !staticF32(operation.getResult(0).getType()))
        return unsupported(&operation, "convolutions must have f32 input, weights and output");

      auto activation = operation.getAttrOfType<dwc::ActivationFunctionAttr>("activation_function");
      auto cell = operation.getAttrOfType<dwc::CellOperationAttr>("cell_operation");
      auto padding = operation.getAttrOfType<dwc::PaddingAttr>("pad");
      if (!activation || !cell || !padding || cell.getValue() != dwc::CellOperation::Mac ||
          padding.getValue() != dwc::Padding::None ||
          (activation.getValue() != dwc::ActivationFunction::None &&
           activation.getValue() != dwc::ActivationFunction::Relu))
        return unsupported(&operation, "convolutions require MAC, explicit padding and NONE or RELU activation");

      if (operation.getNumOperands() == 3) {
        auto bias = operation.getOperand(2).getDefiningOp<arith::ConstantOp>();
        if (!bias || !isa<DenseFPElementsAttr>(bias.getValue()))
          return unsupported(&operation, "convolution bias must be a dense f32 constant");
      }
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
        if (!value.hasOneUse() || !value.getDefiningOp() || !isCompute(value.getDefiningOp()))
          return unsupported(returned, "each return must have a sole-use supported compute producer");
      }
      continue;
    }

    return unsupported(&operation, "the operation has no established precision policy");
  }

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
            pad.getMixedLowPad(), pad.getMixedHighPad(), zero);
        values.map(pad.getResult(), output.getResult());
        continue;
      }

      bool terminal = operation.getResult(0).hasOneUse() &&
          isa<func::ReturnOp>(*operation.getResult(0).getUsers().begin());
      Type resultType = terminal ? operation.getResult(0).getType()
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
      values.map(operation.getResult(0), builder.create(state)->getResult(0));
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
    SmallVector<func::FuncOp> functions;

    for (Operation &operation : *module.getBody()) {
      auto function = dyn_cast<func::FuncOp>(operation);
      if (!function || failed(validateFunction(function))) {
        if (!function)
          (void)unsupported(&operation, "only functions are accepted in the module");
        return signalPassFailure();
      }
      functions.push_back(function);
    }

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
  }
};

}

std::unique_ptr<Pass> mlir::darwinn::createLegalizeBfloat16Pass() {
  return std::make_unique<LegalizeBfloat16Pass>();
}

void mlir::darwinn::registerLegalizeBfloat16Pass() {
  PassRegistration<LegalizeBfloat16Pass>();
}
