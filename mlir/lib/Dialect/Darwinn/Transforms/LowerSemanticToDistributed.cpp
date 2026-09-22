#include "mlir/Dialect/Darwinn/Transforms/LowerSemanticToDistributed.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Darwinn/IR/DarwinnOps.h"
#include "mlir/Dialect/Darwinn/IR/DwcOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/STLExtras.h"
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

LogicalResult validateCwise(dwc::CwiseOp operation) {
  if (operation->getNumOperands() != 2 || operation->getNumResults() != 1 ||
      !isSupportedTensor(operation->getResult(0).getType(), true))
    return unsupported(operation, "binary operations require two inputs and "
                                  "one positive static rank-four result");

  auto activation = operation->getAttrOfType<dwc::ActivationFunctionAttr>("activation_function");
  auto kind = operation->getAttrOfType<dwc::CwiseOpTypeAttr>("op_type");

  if (!activation || activation.getValue() != dwc::ActivationFunction::None || !kind ||
      (kind.getValue() != dwc::CwiseOpType::Add && kind.getValue() != dwc::CwiseOpType::Subtract &&
       kind.getValue() != dwc::CwiseOpType::Multiply))
    return unsupported(operation, "only ADD, SUBTRACT and MULTIPLY with NONE activation "
                                  "have a supported binary lowering");

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

  if (input.getShape() != output.getShape() || !input.getElementType().isF32() ||
      !output.getElementType().isBF16() ||
      !operation->getOperand(1).getDefiningOp<dwc::ConstNoneOp>() || !activation ||
      activation.getValue() != dwc::ActivationFunction::None || !scales || !scales.empty() ||
      !padding || padding.getValue() != dwc::PerZOutScalePadding::None)
    return unsupported(operation, "rescaling currently supports only shape-preserving "
                                  "f32 to bf16 conversion without bias, scale or activation");

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
          (isa<dwc::GenericConstantOp, arith::ConstantOp>(operation) &&
           (name == "value" || name == "darwinn.is_parameter"));
      if (!known)
        return unsupported(&operation, "unsupported semantic attribute " + name);
    }

    if (auto cwise = dyn_cast<dwc::CwiseOp>(operation)) {
      if (failed(validateCwise(cwise)))
        return failure();
      continue;
    }

    if (auto rescaling = dyn_cast<dwc::RescalingOp>(operation)) {
      if (failed(validateRescaling(rescaling)))
        return failure();
      continue;
    }

    if (isa<dwc::GenericConstantOp, arith::ConstantOp>(operation)) {
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

      if (isa<dwc::GenericConstantOp, arith::ConstantOp>(operation)) {
        auto value = operation.getAttrOfType<ElementsAttr>("value");
        auto type = tileType(operation.getResult(0).getType());
        auto parameter = operation.getAttrOfType<BoolAttr>("darwinn.is_parameter");
        ConstTypeAttr kind;
        if (parameter && parameter.getValue())
          kind = ConstTypeAttr::get(context, ConstKind::Parameter);
        Value filled =
            FillOp::create(builder, location, type, value, kind, sliceBegins(type.getRank()),
                           builder.getI32ArrayAttr({1, 1}), sliceEnds(type.getShape()));
        values.map(operation.getResult(0), filled);
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

      auto outputType = tileType(operation.getResult(0).getType());
      auto traversal = AffineMap::getMultiDimIdentityMap(outputType.getRank(), context);
      SmallVector<Value> operands;
      unsigned count = isa<dwc::CwiseOp>(operation) ? 2 : 1;

      for (Value operand : operation.getOperands().take_front(count)) {
        Value storage = redistribute(values.lookup(operand), DistributedMemorySpace::TileMemory);
        operands.push_back(view(storage, outputType.getShape()));
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
        Value computed = StaticComputeOpOp::create(
            builder, location, outputType, operands[0], operands[1], destination, ValueRange{},
            computeOptions(InnerOperationKind::Elementwise, linear), traversal, ArrayAttr{},
            CustomTilingOptionsAttr{}, DtcInfoAttr{}, VexInfoAttr{});
        result = GetTensorOp::create(builder, location, outputType, computed);
      } else {
        result = StaticUnaryComputeOpOp::create(
            builder, location, outputType, operands[0], destination, ValueRange{},
            computeOptions(InnerOperationKind::Unary, LinearFunctionKind::Add), traversal,
            builder.getArrayAttr({}), CustomTilingOptionsAttr{}, ArrayAttr{}, VexInfoAttr{});
      }

      values.map(operation.getResult(0), result);
    }
  }

private:
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

  Value redistribute(Value input, DistributedMemorySpace memory) {
    auto inputType = cast<DistributedTensorType>(input.getType());
    auto outputType = DistributedTensorType::get(context, inputType.getShape(),
                                                 inputType.getElementType(), memory);
    return RedistributeOp::create(builder, input.getLoc(), outputType, input, Value{},
                                  MappingAttr{}, sliceBegins(inputType.getRank()),
                                  builder.getI32ArrayAttr({1, 1}), sliceEnds(inputType.getShape()));
  }

  Value view(Value storage, ArrayRef<int64_t> iterationShape) {
    auto type = cast<DistributedTensorType>(storage.getType());
    auto viewType = DistributedViewType::get(context, type.getShape(), type.getElementType(),
                                             type.getMemorySpace());
    auto created = DistributedCreateViewOp::create(builder, storage.getLoc(), viewType, storage,
                                                   AffineMapAttr{}, AffineMapAttr{});
    SmallVector<AffineExpr> coordinates;
    unsigned leading = iterationShape.size() - type.getRank();

    for (auto [axis, extent] : llvm::enumerate(type.getShape())) {
      bool broadcast = type.getShape() != iterationShape && extent == 1;
      coordinates.push_back(broadcast ? builder.getAffineConstantExpr(0)
                                      : builder.getAffineDimExpr(leading + axis));
    }

    created->setAttr("traversal", AffineMapAttr::get(AffineMap::get(iterationShape.size(), 0,
                                                                    coordinates, context)));
    return created;
  }

  ComputeOpOptionsAttr computeOptions(InnerOperationKind inner, LinearFunctionKind linear) {
    return ComputeOpOptionsAttr::get(
        context, inner, {}, {}, builder.getF32FloatAttr(1), {}, builder.getF32FloatAttr(0),
        builder.getF32FloatAttr(-std::numeric_limits<float>::infinity()),
        builder.getF32FloatAttr(std::numeric_limits<float>::infinity()), {}, linear, {}, {},
        builder.getI32IntegerAttr(0), NluFunctionKind::Linear, NluPreprocessKind::None,
        NluPredicateKind::None, NluPredicateKind::None, {}, false, {}, {}, {}, {}, {}, {}, {});
  }

  OpBuilder builder;
  MLIRContext *context;
  IRMapping values;
};

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
    registry.insert<DarwinnDialect, dwc::DwcDialect, func::FuncDialect, arith::ArithDialect>();
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

    if (failed(verify(function)))
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
