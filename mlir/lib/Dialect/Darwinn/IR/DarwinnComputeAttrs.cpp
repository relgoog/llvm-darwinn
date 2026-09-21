#include "mlir/Dialect/Darwinn/IR/DarwinnOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/StringSet.h"

using namespace mlir;
using namespace mlir::darwinn;

namespace {

template <typename Enum>
ParseResult parseComputeEnum(AsmParser &parser, std::optional<Enum> &result,
                             std::optional<Enum> (*symbolize)(StringRef)) {
  SMLoc location = parser.getCurrentLocation();
  StringRef keyword;
  if (parser.parseLess() || parser.parseKeyword(&keyword) ||
      parser.parseGreater())
    return failure();

  result = symbolize(keyword);
  if (!result)
    return parser.emitError(location)
           << "invalid compute option enum value '" << keyword << "'";
  return success();
}

ParseResult
parseNamedFields(AsmParser &parser,
                 llvm::function_ref<ParseResult(StringRef)> parseField) {
  if (parser.parseLess())
    return failure();

  llvm::StringSet<> seen;
  do {
    SMLoc location = parser.getCurrentLocation();
    StringRef name;
    if (parser.parseKeyword(&name) || parser.parseEqual())
      return failure();
    if (!seen.insert(name).second)
      return parser.emitError(location)
             << "duplicate parameter '" << name << "'";
    if (parseField(name))
      return failure();
  } while (succeeded(parser.parseOptionalComma()));

  return parser.parseGreater();
}

template <typename Enum>
bool isValidEnum(std::optional<Enum> value,
                 std::optional<Enum> (*symbolize)(uint32_t)) {
  return !value || symbolize(static_cast<uint32_t>(*value)).has_value();
}

class NamedFieldPrinter {
public:
  explicit NamedFieldPrinter(AsmPrinter &printer) : printer(printer) {}

  template <typename AttributeT>
  void attribute(StringRef name, AttributeT value) {
    if (!value)
      return;
    field(name);
    printer << value;
  }

  template <typename Enum>
  void enumeration(StringRef name, std::optional<Enum> value,
                   StringRef (*stringify)(Enum)) {
    if (!value)
      return;
    field(name);
    printer << '<' << stringify(*value) << '>';
  }

  void field(StringRef name) {
    if (needsComma)
      printer << ", ";
    printer << name << " = ";
    needsComma = true;
  }

private:
  AsmPrinter &printer;
  bool needsComma = false;
};

}

Attribute ComputeOpOptionsAttr::parse(AsmParser &parser, Type) {
  SMLoc location = parser.getCurrentLocation();
  std::optional<InnerOperationKind> innerOperation;
  IntegerAttr lhsZeroPoint, rhsZeroPoint, outputZeroPoint;
  FloatAttr scaleImmediate, inputScaleImmediate, biasImmediate, clipLower,
      clipUpper;
  std::optional<LinearFunctionKind> linearFunction;
  IntegerAttr activationSubChannelBlockSize, parameterSubChannelBlockSize,
      replicateReduce;
  std::optional<NluFunctionKind> nluFunction;
  std::optional<NluPreprocessKind> nluPreprocessType;
  std::optional<NluPredicateKind> biasPredicate, scalePredicate;
  IntegerAttr bitmask, alpha, beta, numGroups;
  std::optional<bool> tryCellgroups;
  SplineSegmentAttr splineSegment;
  std::optional<NluE8M0RoundingKind> nluE8M0Rounding;
  std::optional<ComputeLoweringHintKind> loweringHint;
  std::optional<ComputeTypeHintKind> computeTypeHint;

  if (parseNamedFields(parser, [&](StringRef name) -> ParseResult {
        if (name == "inner_operation")
          return parseComputeEnum(parser, innerOperation,
                                  symbolizeInnerOperationKind);
        if (name == "lhs_zero_point")
          return parser.parseAttribute(lhsZeroPoint);
        if (name == "rhs_zero_point")
          return parser.parseAttribute(rhsZeroPoint);
        if (name == "scale_immediate")
          return parser.parseAttribute(scaleImmediate);
        if (name == "input_scale_immediate")
          return parser.parseAttribute(inputScaleImmediate);
        if (name == "bias_immediate")
          return parser.parseAttribute(biasImmediate);
        if (name == "clips")
          return failure(
              parser.parseLSquare() || parser.parseAttribute(clipLower) ||
              parser.parseComma() || parser.parseAttribute(clipUpper) ||
              parser.parseRSquare());
        if (name == "output_zero_point")
          return parser.parseAttribute(outputZeroPoint);
        if (name == "linear_function")
          return parseComputeEnum(parser, linearFunction,
                                  symbolizeLinearFunctionKind);
        if (name == "activation_sub_channel_block_size")
          return parser.parseAttribute(activationSubChannelBlockSize);
        if (name == "parameter_sub_channel_block_size")
          return parser.parseAttribute(parameterSubChannelBlockSize);
        if (name == "replicate_reduce")
          return parser.parseAttribute(replicateReduce);
        if (name == "nlu_function")
          return parseComputeEnum(parser, nluFunction,
                                  symbolizeNluFunctionKind);
        if (name == "nlu_preprocess_type")
          return parseComputeEnum(parser, nluPreprocessType,
                                  symbolizeNluPreprocessKind);
        if (name == "bias_predicate")
          return parseComputeEnum(parser, biasPredicate,
                                  symbolizeNluPredicateKind);
        if (name == "scale_predicate")
          return parseComputeEnum(parser, scalePredicate,
                                  symbolizeNluPredicateKind);
        if (name == "bitmask")
          return parser.parseAttribute(bitmask);
        if (name == "try_cellgroups") {
          BoolAttr value;
          if (parser.parseAttribute(value))
            return failure();
          tryCellgroups = value.getValue();
          return success();
        }
        if (name == "alpha")
          return parser.parseAttribute(alpha);
        if (name == "beta")
          return parser.parseAttribute(beta);
        if (name == "num_groups")
          return parser.parseAttribute(numGroups);
        if (name == "spline_segment") {
          splineSegment = dyn_cast_or_null<SplineSegmentAttr>(
              SplineSegmentAttr::parse(parser, {}));
          return success(static_cast<bool>(splineSegment));
        }
        if (name == "nlu_e8m0_rounding")
          return parseComputeEnum(parser, nluE8M0Rounding,
                                  symbolizeNluE8M0RoundingKind);
        if (name == "lowering_hint")
          return parseComputeEnum(parser, loweringHint,
                                  symbolizeComputeLoweringHintKind);
        if (name == "compute_type_hint")
          return parseComputeEnum(parser, computeTypeHint,
                                  symbolizeComputeTypeHintKind);
        return parser.emitError(parser.getCurrentLocation())
               << "unknown compute option '" << name << "'";
      }))
    return {};

  if (!linearFunction) {
    parser.emitError(location) << "missing compute option 'linear_function'";
    return {};
  }

  return parser.getChecked<ComputeOpOptionsAttr>(
      location, parser.getContext(), innerOperation, lhsZeroPoint, rhsZeroPoint,
      scaleImmediate, inputScaleImmediate, biasImmediate, clipLower, clipUpper,
      outputZeroPoint, *linearFunction, activationSubChannelBlockSize,
      parameterSubChannelBlockSize, replicateReduce, nluFunction,
      nluPreprocessType, biasPredicate, scalePredicate, bitmask, tryCellgroups,
      alpha, beta, numGroups, splineSegment, nluE8M0Rounding, loweringHint,
      computeTypeHint);
}

void ComputeOpOptionsAttr::print(AsmPrinter &printer) const {
  printer << '<';
  NamedFieldPrinter fields(printer);
  fields.enumeration("inner_operation", getInnerOperation(),
                     stringifyInnerOperationKind);
  fields.attribute("lhs_zero_point", getLhsZeroPoint());
  fields.attribute("rhs_zero_point", getRhsZeroPoint());
  fields.attribute("scale_immediate", getScaleImmediate());
  fields.attribute("input_scale_immediate", getInputScaleImmediate());
  fields.attribute("bias_immediate", getBiasImmediate());
  if (getClipLower()) {
    fields.field("clips");
    printer << '[' << getClipLower() << ", " << getClipUpper() << ']';
  }

  fields.attribute("output_zero_point", getOutputZeroPoint());
  fields.enumeration("linear_function", std::optional{getLinearFunction()},
                     stringifyLinearFunctionKind);
  fields.attribute("activation_sub_channel_block_size",
                   getActivationSubChannelBlockSize());
  fields.attribute("parameter_sub_channel_block_size",
                   getParameterSubChannelBlockSize());
  fields.attribute("replicate_reduce", getReplicateReduce());
  fields.enumeration("nlu_function", getNluFunction(),
                     stringifyNluFunctionKind);
  fields.enumeration("nlu_preprocess_type", getNluPreprocessType(),
                     stringifyNluPreprocessKind);
  fields.enumeration("bias_predicate", getBiasPredicate(),
                     stringifyNluPredicateKind);
  fields.enumeration("scale_predicate", getScalePredicate(),
                     stringifyNluPredicateKind);
  fields.attribute("bitmask", getBitmask());
  if (auto value = getTryCellgroups()) {
    fields.field("try_cellgroups");
    printer << (*value ? "true" : "false");
  }

  fields.attribute("alpha", getAlpha());
  fields.attribute("beta", getBeta());
  fields.attribute("num_groups", getNumGroups());
  if (auto value = getSplineSegment()) {
    fields.field("spline_segment");
    value.print(printer);
  }

  fields.enumeration("nlu_e8m0_rounding", getNluE8M0Rounding(),
                     stringifyNluE8M0RoundingKind);
  fields.enumeration("lowering_hint", getLoweringHint(),
                     stringifyComputeLoweringHintKind);
  fields.enumeration("compute_type_hint", getComputeTypeHint(),
                     stringifyComputeTypeHintKind);
  printer << '>';
}

LogicalResult ComputeOpOptionsAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError,
    std::optional<InnerOperationKind> innerOperation, IntegerAttr, IntegerAttr,
    FloatAttr, FloatAttr, FloatAttr, FloatAttr clipLower, FloatAttr clipUpper,
    IntegerAttr, LinearFunctionKind linearFunction, IntegerAttr, IntegerAttr,
    IntegerAttr, std::optional<NluFunctionKind> nluFunction,
    std::optional<NluPreprocessKind> nluPreprocessType,
    std::optional<NluPredicateKind> biasPredicate,
    std::optional<NluPredicateKind> scalePredicate, IntegerAttr,
    std::optional<bool>, IntegerAttr, IntegerAttr, IntegerAttr,
    SplineSegmentAttr splineSegment,
    std::optional<NluE8M0RoundingKind> nluE8M0Rounding,
    std::optional<ComputeLoweringHintKind> loweringHint,
    std::optional<ComputeTypeHintKind> computeTypeHint) {
  if (!isValidEnum(innerOperation, symbolizeInnerOperationKind) ||
      !symbolizeLinearFunctionKind(static_cast<uint32_t>(linearFunction)) ||
      !isValidEnum(nluFunction, symbolizeNluFunctionKind) ||
      !isValidEnum(nluPreprocessType, symbolizeNluPreprocessKind) ||
      !isValidEnum(biasPredicate, symbolizeNluPredicateKind) ||
      !isValidEnum(scalePredicate, symbolizeNluPredicateKind) ||
      !isValidEnum(nluE8M0Rounding, symbolizeNluE8M0RoundingKind) ||
      !isValidEnum(loweringHint, symbolizeComputeLoweringHintKind) ||
      !isValidEnum(computeTypeHint, symbolizeComputeTypeHintKind))
    return emitError() << "invalid compute option enumeration";
  if (static_cast<bool>(clipLower) != static_cast<bool>(clipUpper))
    return emitError() << "clipping bounds must both be present or absent";
  if (splineSegment &&
      failed(SplineSegmentAttr::verify(emitError, splineSegment.getIntervals(),
                                       splineSegment.getPolynomials(),
                                       splineSegment.getFunctionSymmetry())))
    return failure();
  return success();
}

Attribute SplineSegmentAttr::parse(AsmParser &parser, Type) {
  SMLoc location = parser.getCurrentLocation();
  ArrayAttr intervals, polynomials;
  std::optional<FunctionSymmetryKind> functionSymmetry;
  if (parseNamedFields(parser, [&](StringRef name) -> ParseResult {
        if (name == "intervals")
          return parser.parseAttribute(intervals);
        if (name == "polynomials")
          return parser.parseAttribute(polynomials);
        if (name == "function_symmetry")
          return parseComputeEnum(parser, functionSymmetry,
                                  symbolizeFunctionSymmetryKind);
        return parser.emitError(parser.getCurrentLocation())
               << "unknown spline parameter '" << name << "'";
      }))
    return {};

  if (!functionSymmetry) {
    parser.emitError(location)
        << "missing spline parameter 'function_symmetry'";
    return {};
  }

  return parser.getChecked<SplineSegmentAttr>(
      location, parser.getContext(), intervals, polynomials, *functionSymmetry);
}

void SplineSegmentAttr::print(AsmPrinter &printer) const {
  printer << '<';
  NamedFieldPrinter fields(printer);
  fields.attribute("intervals", getIntervals());
  fields.attribute("polynomials", getPolynomials());
  fields.enumeration("function_symmetry", std::optional{getFunctionSymmetry()},
                     stringifyFunctionSymmetryKind);
  printer << '>';
}

LogicalResult
SplineSegmentAttr::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                          ArrayAttr, ArrayAttr,
                          FunctionSymmetryKind functionSymmetry) {
  if (!symbolizeFunctionSymmetryKind(static_cast<uint32_t>(functionSymmetry)))
    return emitError() << "invalid spline function symmetry";
  return success();
}

LogicalResult
FunctionSymmetryAttr::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                             FunctionSymmetryKind value) {
  if (!symbolizeFunctionSymmetryKind(static_cast<uint32_t>(value)))
    return emitError() << "invalid function symmetry";
  return success();
}

LogicalResult InterpolateMethodAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError,
    InterpolateMethodKind value) {
  if (!symbolizeInterpolateMethodKind(static_cast<uint32_t>(value)))
    return emitError() << "invalid interpolation method";
  return success();
}
