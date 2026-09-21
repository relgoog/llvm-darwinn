#include "mlir/Target/Darwinn/ParameterTranslation.h"

#include "mlir/Dialect/Darwinn/IR/DarwinnOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Target/Darwinn/Parameters.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <limits>

using namespace mlir;
using namespace mlir::darwinn;

namespace {

struct WeightedUse {
  TensorOpOp consumer;
  NarrowToWideOp conversion;
  FillOp filter;
  FillOp bias;
};

LogicalResult verifyFullSlice(Operation *operation, ArrayRef<int64_t> shape) {
  auto begins = operation->getAttrOfType<AffineMapAttr>("slicing_begins");
  auto ends = operation->getAttrOfType<AffineMapAttr>("slicing_ends");
  if (!begins && !ends)
    return success();
  if (!begins || !ends || begins.getValue().getNumResults() != shape.size() ||
      ends.getValue().getNumResults() != shape.size())
    return operation->emitError(
        "parameter translation requires a full constant slice");

  for (auto [index, size] : llvm::enumerate(shape)) {
    auto begin =
        dyn_cast<AffineConstantExpr>(begins.getValue().getResult(index));
    auto end = dyn_cast<AffineConstantExpr>(ends.getValue().getResult(index));
    if (!begin || !end || begin.getValue() != 0 || end.getValue() != size - 1)
      return operation->emitError(
          "parameter translation requires a full constant slice");
  }

  return success();
}

FailureOr<FillOp> traceConstant(Value value) {
  SmallVector<Operation *> path;
  while (Operation *operation = value.getDefiningOp()) {
    if (auto fill = dyn_cast<FillOp>(operation)) {
      auto dense = dyn_cast<DenseFPElementsAttr>(fill.getValue());
      if (!dense || !dense.getType().hasStaticShape())
        return fill.emitError("parameter translation requires concrete "
                              "floating point DenseElementsAttr constants");

      auto shape = dense.getType().getShape();
      if (llvm::any_of(shape, [](int64_t size) { return size <= 0; }))
        return fill.emitError(
            "parameter translation requires positive constant dimensions");
      uint64_t elementCount = 1;
      uint64_t maximum = std::min<uint64_t>(
          SmallVector<float>().max_size(), std::numeric_limits<int64_t>::max());
      for (int64_t dimension : shape) {
        if (uint64_t(dimension) > maximum / elementCount)
          return fill.emitError(
              "parameter constant element count is too large");
        elementCount *= uint64_t(dimension);
      }
      if (failed(verifyFullSlice(fill, shape)))
        return failure();

      for (Operation *step : path) {
        if (auto view = dyn_cast<DistributedCreateViewOp>(step)) {
          auto forward = view.getForwardIndexTransformationAttr();
          auto reverse = view.getReverseIndexTransformationAttr();
          if (!forward || !reverse || !forward.getValue().isIdentity() ||
              !reverse.getValue().isIdentity())
            return view.emitError(
                "parameter translation requires identity constant views");
        } else if (failed(verifyFullSlice(step, shape))) {
          return failure();
        }
      }

      return fill;
    }
    if (auto tensor = dyn_cast<GetTensorOp>(operation)) {
      path.push_back(operation);
      value = tensor.getInput();
      continue;
    }
    if (auto view = dyn_cast<DistributedCreateViewOp>(operation)) {
      path.push_back(operation);
      value = view.getInput();
      continue;
    }
    break;
  }

  return FillOp();
}

ConstKind constantKind(FillOp fill) {
  auto kind = fill.getConstTypeAttr();
  return kind ? kind.getValue() : ConstKind::None;
}

LogicalResult verifyConstantUses(FillOp fill,
                                 const llvm::DenseSet<OpOperand *> &consumed) {
  SmallVector<Value> pending{fill.getOutput()};
  llvm::DenseSet<Value> visited;
  while (!pending.empty()) {
    Value value = pending.pop_back_val();
    if (!visited.insert(value).second)
      continue;

    for (OpOperand &use : value.getUses()) {
      if (consumed.contains(&use))
        continue;
      Operation *owner = use.getOwner();
      if (isa<GetTensorOp, DistributedCreateViewOp>(owner) &&
          use.getOperandNumber() == 0) {
        pending.push_back(owner->getResult(0));
        continue;
      }
      if (auto conversion = dyn_cast<NarrowToWideOp>(owner)) {
        if (use.getOperandNumber() == 0 &&
            conversion.getAuxiliaryOutputs().empty()) {
          pending.push_back(conversion.getOutput());
          continue;
        }
      }
      return owner->emitError(
          "constant use has no supported parameter binding");
    }
  }

  return success();
}

SmallVector<float> readFloats(DenseFPElementsAttr values) {
  SmallVector<float> result;
  result.reserve(values.getNumElements());
  for (const APFloat &value : values.getValues<APFloat>())
    result.push_back(value.convertToFloat());
  return result;
}

LogicalResult verifyConversion(WeightedUse use, ArrayRef<int64_t> shape,
                               bool vectorPacked) {
  auto conversion = use.conversion;
  if (!conversion.getAuxiliaryTensors().empty() ||
      !conversion.getAuxiliaryOutputs().empty() ||
      conversion.getReverseIndexTransformationAttr() ||
      (conversion.getZeroPointAttr() && conversion.getZeroPoint() != 0))
    return conversion.emitError("parameter translation does not support "
                                "auxiliary or quantized parameter conversions");
  if (!conversion.getOutput().hasOneUse())
    return conversion.emitError("parameter translation requires one scheduled "
                                "consumer per parameter conversion");

  auto input = dyn_cast<DistributedViewType>(conversion.getInput().getType());
  auto output = dyn_cast<DistributedViewType>(conversion.getOutput().getType());
  if (!input || !output || input.getShape() != shape ||
      !input.getElementType().isBF16() ||
      input.getMemorySpace() != DistributedMemorySpace::TileMemory ||
      output.getMemorySpace() != DistributedMemorySpace::TileRegisters)
    return conversion.emitError("parameter translation requires BF16 parameter "
                                "views from tile memory to tile registers");

  auto mapping = conversion.getForwardIndexTransformationAttr();
  if (!vectorPacked) {
    if (mapping || output.getShape() != shape ||
        !output.getElementType().isBF16())
      return conversion.emitError(
          "stencil parameter conversion must preserve its scalar layout");
    return success();
  }

  auto vector = dyn_cast<VectorType>(output.getElementType());
  if (!vector || vector.isScalable() || vector.getRank() != 1 ||
      !vector.getElementType().isBF16() ||
      vector.getDimSize(0) != shape.back() ||
      output.getShape() != shape.drop_back() || !mapping ||
      mapping.getValue().getNumDims() != shape.size() ||
      mapping.getValue().getNumSymbols() != 0 ||
      mapping.getValue().getNumResults() != shape.size() - 1)
    return conversion.emitError("convolution parameter conversion must "
                                "vectorize the output channel dimension");

  for (auto [index, expression] :
       llvm::enumerate(mapping.getValue().getResults())) {
    auto dimension = dyn_cast<AffineDimExpr>(expression);
    if (!dimension || dimension.getPosition() != index)
      return conversion.emitError("convolution parameter conversion must "
                                  "preserve the remaining dimensions");
  }

  return success();
}

LogicalResult appendBytes(PackedParameters &result, ArrayRef<uint8_t> bytes,
                          Operation *source) {
  if (bytes.size() > result.bytes.max_size() - result.bytes.size())
    return source->emitError("packed parameter payload is too large");
  result.bytes.append(bytes.begin(), bytes.end());
  return success();
}

LogicalResult appendWeighted(WeightedUse use, PackedParameters &result) {
  auto filter = cast<DenseFPElementsAttr>(use.filter.getValue());
  auto shape = filter.getType().getShape();
  auto kind = use.consumer.getCompute().getInnerOperation();
  bool stencil = kind == InnerOperationKind::Stencil;
  if (shape.empty() || !filter.getElementType().isBF16())
    return use.filter.emitError(
        "scheduled filter constants must be ranked BF16 tensors");
  if (failed(verifyConversion(use, shape, !stencil)))
    return failure();

  SmallVector<float> bias;
  std::optional<uint32_t> biasImmediate;
  if (use.bias) {
    auto values = cast<DenseFPElementsAttr>(use.bias.getValue());
    if (!values.getElementType().isF32() || values.getType().getRank() != 1 ||
        values.getType().getDimSize(0) != shape.back())
      return use.bias.emitError(
          "convolution bias must be a full f32 output channel vector");

    auto compute = use.consumer.getCompute();
    if ((compute.getBiasImmediate() &&
         !compute.getBiasImmediate().getValue().isZero()) ||
        (compute.getBiasPredicate() &&
         compute.getBiasPredicate() != NluPredicateKind::None))
      return use.consumer.emitError(
          "parameter translation requires an unpredicated auxiliary bias "
          "without an existing immediate");

    if (values.isSplat())
      biasImmediate =
          values.getSplatValue<APFloat>().bitcastToAPInt().getZExtValue();
    else
      bias = readFloats(values);
  }

  auto filterValues = readFloats(filter);
  std::optional<ArrayRef<float>> packedBias;
  if (!bias.empty())
    packedBias = bias;
  std::optional<uint32_t> inputChannelTile;
  auto packed = [&]() -> llvm::Expected<SmallVector<uint8_t>> {
    if (stencil) {
      if (shape.size() != 5 || shape[2] != 1 || shape[3] != 1)
        return llvm::createStringError("stencil parameters require [height, "
                                       "width, 1, 1, channels] constants");
      auto dimensions = DepthwiseShape::create(
          {size_t(shape[0]), size_t(shape[1]), size_t(shape[4])});
      if (!dimensions)
        return dimensions.takeError();
      return packDepthwise(*dimensions, filterValues, packedBias);
    }

    auto tile = use.conversion.getInputChannelTileAttr();
    if (!tile)
      return llvm::createStringError(
          "convolution parameter conversion requires a verified "
          "input_channel_tile plan");
    int64_t tileSize = tile.getValue().getSExtValue();
    if (tileSize <= 0 || tileSize % 2 != 0)
      return llvm::createStringError(
          "convolution input_channel_tile must be positive and even");
    inputChannelTile = uint32_t(tileSize);

    std::array<size_t, 4> dimensions;
    if (shape.size() == 4) {
      dimensions = {size_t(shape[3]), size_t(shape[0]), size_t(shape[1]),
                    size_t(shape[2])};
    } else if (shape.size() == 6 && shape[0] == 1 && shape[2] == 1 &&
               use.consumer.getCompute().getComputeTypeHint() ==
                   ComputeTypeHintKind::TransposedConv) {
      dimensions = {size_t(shape[5]), size_t(shape[1]), size_t(shape[3]),
                    size_t(shape[4])};
    } else {
      return llvm::createStringError(
          "convolution parameters require HWIO or unit-stride-expanded "
          "transposed HWIO constants");
    }

    auto convolution = ConvolutionShape::create(dimensions);
    if (!convolution)
      return convolution.takeError();
    SmallVector<float> outputMajor(filterValues.size());
    size_t outputChannels = dimensions[0];
    size_t remainingElements = filterValues.size() / outputChannels;
    for (size_t element = 0; element < remainingElements; ++element)
      for (size_t output = 0; output < outputChannels; ++output)
        outputMajor[output * remainingElements + element] =
            filterValues[element * outputChannels + output];

    return packConvolution(*convolution, outputMajor, packedBias,
                           *inputChannelTile);
  }();

  if (!packed)
    return use.consumer.emitError(llvm::toString(packed.takeError()));

  uint64_t offset = result.bytes.size();
  if (failed(appendBytes(result, *packed, use.consumer)))
    return failure();
  bool hasPackedBias = !bias.empty();
  result.placements.push_back(
      {use.filter.getOutput(),
       use.bias ? use.bias.getOutput() : Value(),
       use.conversion.getOutput(),
       {use.consumer.getOutput()},
       stencil ? ParameterLayout::Stencil : ParameterLayout::Convolution,
       offset,
       packed->size(),
       inputChannelTile,
       hasPackedBias});
  if (biasImmediate)
    result.biasImmediates.push_back({use.consumer.getOutput(), *biasImmediate});
  return success();
}

LogicalResult translateModule(ModuleOp module, llvm::raw_ostream &output) {
  func::FuncOp function;
  for (Operation &operation : module.getBody()->getOperations()) {
    if (function || !isa<func::FuncOp>(operation))
      return module.emitError(
          "parameter translation requires exactly one scheduled function");
    function = cast<func::FuncOp>(operation);
  }
  if (!function || function.isExternal())
    return module.emitError(
        "parameter translation requires one non-external scheduled function");

  auto parameters = translateParameters(function);
  if (failed(parameters))
    return failure();
  output.write(reinterpret_cast<const char *>(parameters->bytes.data()),
               parameters->bytes.size());
  return success();
}

}

FailureOr<PackedParameters>
mlir::darwinn::translateParameters(func::FuncOp function) {
  if (failed(verify(function)))
    return failure();
  if (function.isExternal() || !function.getBody().hasOneBlock())
    return function.emitError(
        "parameter translation requires one scheduled block");

  SmallVector<WeightedUse> weighted;
  llvm::DenseMap<Operation *, SmallVector<Value>> broadcasts;
  llvm::DenseSet<Operation *> accounted;
  llvm::DenseSet<OpOperand *> consumed;
  for (Operation &operation : function.getBody().front())
    if (operation.getNumRegions() != 0)
      return operation.emitError(
          "parameter translation requires flat scheduled operations");

  for (auto tensor : function.getBody().front().getOps<TensorOpOp>()) {
    auto conversion = tensor.getRhs().getDefiningOp<NarrowToWideOp>();
    auto source =
        traceConstant(conversion ? conversion.getInput() : tensor.getRhs());
    if (failed(source))
      return failure();
    if (!*source)
      continue;

    auto kind = tensor.getCompute().getInnerOperation();
    if (!conversion) {
      auto values = cast<DenseFPElementsAttr>((*source).getValue());
      if (constantKind(*source) != ConstKind::None ||
          kind != InnerOperationKind::Elementwise ||
          values.getType().getRank() != 1 || !values.getElementType().isBF16())
        return tensor.emitError(
            "unsupported direct constant operand in parameter translation");
      broadcasts[*source].push_back(tensor.getOutput());
      accounted.insert(*source);
      consumed.insert(&tensor->getOpOperand(1));
      continue;
    }

    if (constantKind(*source) != ConstKind::Parameter ||
        (kind != InnerOperationKind::Vmc &&
         kind != InnerOperationKind::Stencil))
      return tensor.emitError(
          "unsupported weighted operation in parameter translation");

    FillOp bias;
    auto auxiliary = tensor.getAuxiliaryTensors();
    if (!auxiliary.empty()) {
      auto types = tensor.getAuxiliaryTensorTypesAttr();
      if (auxiliary.size() != 1 || !types || types.size() != 1 ||
          cast<AuxTensorTypeAttr>(types[0]).getValue() != AuxTensorKind::Bias)
        return tensor.emitError(
            "parameter translation only supports a single BIAS auxiliary");
      auto biasSource = traceConstant(auxiliary.front());
      if (failed(biasSource))
        return failure();
      if (!*biasSource || constantKind(*biasSource) != ConstKind::BiasOrScale)
        return tensor.emitError("parameter translation requires a constant "
                                "BIAS_OR_SCALE auxiliary");
      bias = *biasSource;
      accounted.insert(bias);
      consumed.insert(&tensor->getOpOperand(3));
    }

    weighted.push_back({tensor, conversion, *source, bias});
    accounted.insert(*source);
    consumed.insert(&tensor->getOpOperand(1));
  }

  PackedParameters result;
  for (auto fill : function.getBody().front().getOps<FillOp>()) {
    if (!fill.getOutput().use_empty() && !accounted.contains(fill))
      return fill.emitError(
          "live constant has no supported parameter placement");
    if (failed(verifyConstantUses(fill, consumed)))
      return failure();
    auto found = broadcasts.find(fill);
    if (found == broadcasts.end())
      continue;

    auto values = readFloats(cast<DenseFPElementsAttr>(fill.getValue()));
    auto packed = packBfloat(values);
    if (!packed)
      return fill.emitError(llvm::toString(packed.takeError()));
    uint64_t offset = result.bytes.size();
    if (failed(appendBytes(result, *packed, fill)))
      return failure();
    result.placements.push_back({fill.getOutput(), Value(), Value(),
                                 std::move(found->second),
                                 ParameterLayout::BfloatVector, offset,
                                 packed->size(), std::nullopt, false});
  }

  for (WeightedUse use : weighted)
    if (failed(appendWeighted(use, result)))
      return failure();
  return result;
}

void mlir::registerToDarwinnParametersTranslation() {
  TranslateFromMLIRRegistration registration(
      "darwinn-to-parameters", "Pack scheduled Darwinn parameter constants",
      translateModule, [](DialectRegistry &registry) {
        registry.insert<DarwinnDialect, func::FuncDialect>();
      });
}
