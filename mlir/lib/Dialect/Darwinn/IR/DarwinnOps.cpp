//===- DarwinnOps.cpp - MLIR Dialect for Darwinn ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Darwinn/IR/DarwinnOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"

using namespace mlir;
using namespace mlir::darwinn;
using namespace mlir::dwc;

#include "mlir/Dialect/Darwinn/IR/DarwinnOpsDialect.cpp.inc"
#include "mlir/Dialect/Darwinn/IR/DwcOpsDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/Darwinn/IR/DarwinnAttributes.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/Darwinn/IR/DarwinnTypes.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/Darwinn/IR/DarwinnOps.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/Darwinn/IR/DwcOps.cpp.inc"

//===----------------------------------------------------------------------===//
// Darwinn dialect initialization.
//===----------------------------------------------------------------------===//

void DwcDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Darwinn/IR/DwcOps.cpp.inc"
      >();
}

void DarwinnDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Darwinn/IR/DarwinnOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/Darwinn/IR/DarwinnAttributes.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/Darwinn/IR/DarwinnTypes.cpp.inc"
      >();
}

static LogicalResult verifyGatherLike(Operation *op, Value params, Value indices,
                                      int64_t axis, Value output) {
  auto paramsType = llvm::dyn_cast<RankedTensorType>(params.getType());
  auto indicesType = llvm::dyn_cast<RankedTensorType>(indices.getType());
  auto outputType = llvm::dyn_cast<RankedTensorType>(output.getType());
  if (!paramsType || !indicesType || !outputType)
    return success();
  if (!llvm::isa<IntegerType, IndexType>(indicesType.getElementType()))
    return op->emitOpError("expects indices element type to be integer");
  if (paramsType.getElementType() != outputType.getElementType())
    return op->emitOpError(
               "expects params and output to have the same element type");
  int64_t rank = paramsType.getRank();
  int64_t normAxis = axis < 0 ? axis + rank : axis;
  if (normAxis < 0 || normAxis >= rank)
    return op->emitOpError("expects axis in range [-rank, rank), got axis ")
           << axis << " with params rank " << rank;
  int64_t indicesRank = indicesType.getRank();
  if (outputType.getRank() != rank + indicesRank - 1)
    return op->emitOpError("expects output rank to be params rank plus "
                           "indices rank minus one");
  ArrayRef<int64_t> paramsShape = paramsType.getShape();
  ArrayRef<int64_t> indicesShape = indicesType.getShape();
  ArrayRef<int64_t> outputShape = outputType.getShape();
  auto sameOrDynamic = [](int64_t a, int64_t b) {
    return a == ShapedType::kDynamic || b == ShapedType::kDynamic || a == b;
  };
  for (int64_t i = 0; i < normAxis; ++i)
    if (!sameOrDynamic(outputShape[i], paramsShape[i]))
      return op->emitOpError("expects output dim ")
             << i << " to match params dim " << i;
  for (int64_t i = 0; i < indicesRank; ++i)
    if (!sameOrDynamic(outputShape[normAxis + i], indicesShape[i]))
      return op->emitOpError("expects output dim ")
             << normAxis + i << " to match indices dim " << i;
  for (int64_t i = normAxis + 1; i < rank; ++i)
    if (!sameOrDynamic(outputShape[i + indicesRank - 1], paramsShape[i]))
      return op->emitOpError("expects output dim ")
             << i + indicesRank - 1 << " to match params dim " << i;
  return success();
}

LogicalResult darwinn::GatherOp::verify() {
  return verifyGatherLike(*this, getParams(), getIndices(), getAxis(),
                          getOutput());
}

LogicalResult darwinn::GatherCopyOp::verify() {
  return verifyGatherLike(*this, getParams(), getIndices(), getAxis(),
                          getOutput());
}

LogicalResult darwinn::HibGatherOp::verify() {
  return verifyGatherLike(*this, getParams(), getIndices(), getAxis(),
                          getOutput());
}

static LogicalResult verifyDynamicSliceLike(Operation *op, Value input,
                                            Value startIndices,
                                            ArrayRef<int64_t> sliceSizes,
                                            Value output) {
  auto inputType = llvm::dyn_cast<RankedTensorType>(input.getType());
  auto startType = llvm::dyn_cast<RankedTensorType>(startIndices.getType());
  auto outputType = llvm::dyn_cast<RankedTensorType>(output.getType());
  if (!inputType || !startType || !outputType)
    return success();
  int64_t rank = inputType.getRank();
  if (startType.getRank() != 1)
    return op->emitOpError("expects start_indices to be 1-d");
  if (!startType.isDynamicDim(0) && startType.getDimSize(0) != rank)
    return op->emitOpError("expects start_indices size to match input rank");
  if (static_cast<int64_t>(sliceSizes.size()) != rank)
    return op->emitOpError("expects slice_sizes length to match input rank");
  if (outputType.getRank() != rank)
    return op->emitOpError("expects input and output to have the same rank");
  ArrayRef<int64_t> inputShape = inputType.getShape();
  ArrayRef<int64_t> outputShape = outputType.getShape();
  for (int64_t i = 0; i < rank; ++i) {
    int64_t size = sliceSizes[i];
    if (size != -1 && size <= 0)
      return op->emitOpError("expects slice_sizes entries to be -1 or > 0");
    int64_t outputDim = outputShape[i];
    if (outputDim == ShapedType::kDynamic)
      continue;
    int64_t expected = size == -1 ? inputShape[i] : size;
    if (expected != ShapedType::kDynamic && outputDim != expected)
      return op->emitOpError("expects output dim ") << i << " to match slice size";
  }
  return success();
}

LogicalResult darwinn::DynamicSliceOp::verify() {
  return verifyDynamicSliceLike(*this, getInput(), getStartIndices(),
                                getSliceSizes(), getOutput());
}

LogicalResult darwinn::DynamicSliceWithCopyOp::verify() {
  return verifyDynamicSliceLike(*this, getInput(), getStartIndices(),
                                getSliceSizes(), getOutput());
}

LogicalResult darwinn::DynamicUpdateSliceOp::verify() {
  auto inputType = llvm::dyn_cast<RankedTensorType>(getInput().getType());
  auto updateType = llvm::dyn_cast<RankedTensorType>(getUpdate().getType());
  auto startType =
      llvm::dyn_cast<RankedTensorType>(getStartIndices().getType());
  auto outputType = llvm::dyn_cast<RankedTensorType>(getOutput().getType());
  if (!inputType || !updateType || !startType || !outputType)
    return success();
  int64_t rank = inputType.getRank();
  if (updateType.getRank() != rank || outputType.getRank() != rank)
    return emitOpError(
               "expects input, update and output to have the same rank");
  if (startType.getRank() != 1)
    return emitOpError("expects start_indices to be 1-d");
  if (!startType.isDynamicDim(0) && startType.getDimSize(0) != rank)
    return emitOpError("expects start_indices size to match input rank");
  ArrayRef<int64_t> inputShape = inputType.getShape();
  ArrayRef<int64_t> updateShape = updateType.getShape();
  ArrayRef<int64_t> outputShape = outputType.getShape();
  for (int64_t i = 0; i < rank; ++i) {
    if (inputShape[i] != ShapedType::kDynamic &&
        updateShape[i] != ShapedType::kDynamic &&
        updateShape[i] > inputShape[i])
      return emitOpError("expects update dim ")
             << i << " to fit inside input dim " << i;
    if (inputShape[i] != ShapedType::kDynamic &&
        outputShape[i] != ShapedType::kDynamic &&
        outputShape[i] != inputShape[i])
      return emitOpError("expects output dim ")
             << i << " to match input dim " << i;
  }
  return success();
}

static bool isSupportedCastPair(Type inTy, Type outTy) {
  bool inFloat = llvm::isa<FloatType>(inTy);
  bool outFloat = llvm::isa<FloatType>(outTy);
  bool inInt = llvm::isa<IntegerType>(inTy);
  bool outInt = llvm::isa<IntegerType>(outTy);
  return (inFloat || inInt) && (outFloat || outInt);
}

static LogicalResult verifyCastLike(Operation *op, Value input, Value output) {
  auto inputType = llvm::dyn_cast<RankedTensorType>(input.getType());
  auto outputType = llvm::dyn_cast<RankedTensorType>(output.getType());
  if (!inputType || !outputType)
    return success();
  Type inElem = inputType.getElementType();
  Type outElem = outputType.getElementType();
  if (!isSupportedCastPair(inElem, outElem))
    return op->emitOpError("expects float or integer element types, got ")
           << inElem << " to " << outElem;
  if (inputType.getShape() != outputType.getShape())
    return op->emitOpError("expects input and output to have the same shape");
  return success();
}

LogicalResult darwinn::ConvertOp::verify() {
  return verifyCastLike(*this, getInput(), getOutput());
}

LogicalResult darwinn::CastInOp::verify() {
  return verifyCastLike(*this, getInput(), getOutput());
}

LogicalResult darwinn::CastOutOp::verify() {
  return verifyCastLike(*this, getInput(), getOutput());
}

static LogicalResult verifyCopyLike(Operation *op, Value input, Value output) {
  auto inputType = llvm::dyn_cast<RankedTensorType>(input.getType());
  auto outputType = llvm::dyn_cast<RankedTensorType>(output.getType());
  if (!inputType || !outputType)
    return success();
  if (inputType.getElementType() != outputType.getElementType())
    return op->emitOpError("expects input and output to have the same element type, got ")
           << inputType.getElementType() << " and " << outputType.getElementType();
  if (inputType.getShape() != outputType.getShape())
    return op->emitOpError("expects input and output to have the same shape");
  return success();
}

LogicalResult darwinn::CopyOpOp::verify() {
  return verifyCopyLike(*this, getInput(), getOutput());
}

LogicalResult darwinn::CopyFromHostOp::verify() {
  return verifyCopyLike(*this, getInput(), getOutput());
}

LogicalResult darwinn::CopyUsingWideOp::verify() {
  return verifyCopyLike(*this, getInput(), getOutput());
}

static bool isSupportedReduceElement(Type elem) {
  if (llvm::isa<IntegerType>(elem))
    return !elem.isInteger(1);
  return elem.isF32() || elem.isF64();
}

static LogicalResult verifyAccType(Type inputElem, Type accType, Operation *op) {
  if (llvm::isa<IntegerType>(inputElem)) {
    if (!accType.isInteger(64))
      return op->emitOpError("expects i64 accumulator for integer input, got ")
             << accType;
    return success();
  }
  if (inputElem.isF32()) {
    if (!accType.isF32())
      return op->emitOpError("expects f32 accumulator for f32 input, got ")
             << accType;
    return success();
  }
  if (inputElem.isF64()) {
    if (!accType.isF64())
      return op->emitOpError("expects f64 accumulator for f64 input, got ")
             << accType;
    return success();
  }
  return op->emitOpError("expects integer, f32 or f64 input element type, got ")
         << inputElem;
}

LogicalResult darwinn::DiveRefReductionOp::verify() {
  auto inputType = llvm::dyn_cast<RankedTensorType>(getInput().getType());
  auto outputType = llvm::dyn_cast<RankedTensorType>(getOutput().getType());
  if (!inputType || !outputType)
    return success();
  int64_t rank = inputType.getRank();
  if (rank > 10)
    return emitOpError("expects input rank of at most 10, got ") << rank;
  if (outputType.getRank() != rank)
    return emitOpError("expects input and output to have the same rank");
  Type inputElem = inputType.getElementType();
  if (!isSupportedReduceElement(inputElem))
    return emitOpError("expects integer, f32 or f64 input element type, got ")
           << inputElem;
  if (failed(verifyAccType(inputElem, getAccType(), *this)))
    return failure();
  llvm::SmallVector<int64_t> normed;
  for (int64_t axis : getAxes()) {
    int64_t norm = axis < 0 ? axis + rank : axis;
    if (norm < 0 || norm >= rank)
      return emitOpError("expects axes in range [-rank, rank), got axis ")
             << axis << " with input rank " << rank;
    bool seen = false;
    for (int64_t prior : normed)
      seen = seen || prior == norm;
    if (seen)
      return emitOpError("expects no duplicate axes, got duplicate axis ")
             << axis;
    normed.push_back(norm);
  }
  ArrayRef<int64_t> inputShape = inputType.getShape();
  ArrayRef<int64_t> outputShape = outputType.getShape();
  for (int64_t i = 0; i < rank; ++i) {
    int64_t outputDim = outputShape[i];
    if (outputDim == ShapedType::kDynamic)
      continue;
    bool reduced = false;
    for (int64_t prior : normed)
      reduced = reduced || prior == i;
    if (reduced) {
      if (outputDim != 1)
        return emitOpError("expects reduced dim ") << i << " to have extent 1";
    } else if (inputShape[i] != ShapedType::kDynamic && outputDim != inputShape[i]) {
      return emitOpError("expects output dim ") << i << " to match input dim " << i;
    }
  }
  return success();
}

LogicalResult darwinn::IndexFilterOp::verify() {
  auto inputType = llvm::dyn_cast<RankedTensorType>(getInput().getType());
  auto valuesType = llvm::dyn_cast<RankedTensorType>(getValues().getType());
  auto indicesType = llvm::dyn_cast<RankedTensorType>(getIndices().getType());
  if (!inputType || !valuesType || !indicesType)
    return success();
  int64_t rank = inputType.getRank();
  if (rank < 1)
    return emitOpError("expects ranked input of rank at least 1");
  int64_t depth = inputType.getDimSize(rank - 1);
  int64_t k = getK();
  if (k <= 0)
    return emitOpError("expects k to be positive, got ") << k;
  if (depth != ShapedType::kDynamic && k > depth)
    return emitOpError("expects k to be no larger than the depth, got k ")
           << k << " with depth " << depth;
  if (valuesType.getRank() != rank || indicesType.getRank() != rank)
    return emitOpError("expects input, values and indices to have the same rank");
  if (valuesType.getElementType() != inputType.getElementType())
    return emitOpError("expects values and input to have the same element type");
  if (!llvm::isa<IntegerType, IndexType>(indicesType.getElementType()))
    return emitOpError("expects indices element type to be integer");
  ArrayRef<int64_t> inputShape = inputType.getShape();
  ArrayRef<int64_t> valuesShape = valuesType.getShape();
  ArrayRef<int64_t> indicesShape = indicesType.getShape();
  for (int64_t i = 0; i < rank - 1; ++i) {
    if (inputShape[i] != ShapedType::kDynamic &&
        valuesShape[i] != ShapedType::kDynamic && valuesShape[i] != inputShape[i])
      return emitOpError("expects values dim ") << i << " to match input dim " << i;
    if (inputShape[i] != ShapedType::kDynamic &&
        indicesShape[i] != ShapedType::kDynamic && indicesShape[i] != inputShape[i])
      return emitOpError("expects indices dim ") << i << " to match input dim " << i;
  }
  int64_t valuesLast = valuesShape[rank - 1];
  if (valuesLast != ShapedType::kDynamic && valuesLast != k)
    return emitOpError("expects values last dim to equal k");
  int64_t indicesLast = indicesShape[rank - 1];
  if (indicesLast != ShapedType::kDynamic && indicesLast != k)
    return emitOpError("expects indices last dim to equal k");
  if (valuesShape != indicesShape)
    return emitOpError("expects values and indices to have the same shape");
  return success();
}

LogicalResult darwinn::MaskIndicesOp::verify() {
  auto inputType = llvm::dyn_cast<RankedTensorType>(getInput().getType());
  auto outputType = llvm::dyn_cast<RankedTensorType>(getOutput().getType());
  if (!inputType || !outputType)
    return success();
  int64_t rank = inputType.getRank();
  if (rank < 1)
    return emitOpError("expects ranked input of rank at least 1");
  int64_t axis = getAxis();
  int64_t normAxis = axis < 0 ? axis + rank : axis;
  if (normAxis < 0 || normAxis >= rank)
    return emitOpError("expects axis in range [-rank, rank), got axis ")
           << axis << " with input rank " << rank;
  if (!outputType.getElementType().isInteger(32))
    return emitOpError("expects output element type to be i32");
  if (outputType.getRank() != rank - 1)
    return emitOpError("expects output rank to be input rank minus one");
  ArrayRef<int64_t> inputShape = inputType.getShape();
  ArrayRef<int64_t> outputShape = outputType.getShape();
  for (int64_t i = 0, j = 0; i < rank; ++i) {
    if (i == normAxis)
      continue;
    if (inputShape[i] != ShapedType::kDynamic &&
        outputShape[j] != ShapedType::kDynamic && outputShape[j] != inputShape[i])
      return emitOpError("expects output dim ") << j << " to match input dim " << i;
    ++j;
  }
  return success();
}

static LogicalResult verifyPoolingLike(Operation *op, Value input,
                                       ArrayRef<int64_t> kernel,
                                       ArrayRef<int64_t> stride,
                                       ArrayRef<int64_t> pad, Value output) {
  for (int64_t s : kernel)
    if (s < 1)
      return op->emitOpError("expects all kernel values to be >= 1, got ") << kernel;
  for (int64_t s : stride)
    if (s < 1)
      return op->emitOpError("expects all stride values to be >= 1, got ") << stride;
  for (int64_t p : pad)
    if (p < 0)
      return op->emitOpError("expects all padding values to be >= 0, got ") << pad;
  int64_t spatial = static_cast<int64_t>(kernel.size());
  if (static_cast<int64_t>(stride.size()) != spatial)
    return op->emitOpError("expects kernel and stride to have the same length");
  if (static_cast<int64_t>(pad.size()) != 2 * spatial)
    return op->emitOpError("expects pad length to be twice the kernel length");
  for (int64_t i = 0; i < spatial; ++i) {
    if (pad[2 * i] >= kernel[i] || pad[2 * i + 1] >= kernel[i])
      return op->emitOpError("expects padding below the kernel extent on spatial dim ")
             << i;
  }
  auto inputType = llvm::dyn_cast<RankedTensorType>(input.getType());
  auto outputType = llvm::dyn_cast<RankedTensorType>(output.getType());
  if (!inputType || !outputType)
    return success();
  int64_t rank = inputType.getRank();
  if (rank < spatial + 1)
    return op->emitOpError("expects input rank to cover all spatial dims");
  if (outputType.getRank() != rank)
    return op->emitOpError("expects input and output to have the same rank");
  ArrayRef<int64_t> inputShape = inputType.getShape();
  ArrayRef<int64_t> outputShape = outputType.getShape();
  for (int64_t i = 0; i < rank - spatial; ++i) {
    if (inputShape[i] != ShapedType::kDynamic &&
        outputShape[i] != ShapedType::kDynamic && outputShape[i] != inputShape[i])
      return op->emitOpError("expects output dim ") << i << " to match input dim " << i;
  }
  for (int64_t i = 0; i < spatial; ++i) {
    int64_t dim = rank - spatial + i;
    int64_t in = inputShape[dim];
    int64_t out = outputShape[dim];
    if (in == ShapedType::kDynamic || out == ShapedType::kDynamic)
      continue;
    int64_t span = in + pad[2 * i] + pad[2 * i + 1] - kernel[i];
    if (span < 0 || span % stride[i] != 0)
      return op->emitOpError("expects input plus pad minus kernel to be wholly "
                             "divisible by stride on spatial dim ")
             << i;
    if (out != span / stride[i] + 1)
      return op->emitOpError("expects output dim ") << dim << " to match the pooled size";
  }
  return success();
}

LogicalResult darwinn::InterpolateOp::verify() {
  return verifyPoolingLike(*this, getInput(), getKernel(),
                           getStride(), getPad(),
                           getOutput());
}

LogicalResult darwinn::InterpolateHardwareOp::verify() {
  return verifyPoolingLike(*this, getInput(), getKernel(),
                           getStride(), getPad(),
                           getOutput());
}

LogicalResult darwinn::ResamplerOp::verify() {
  return verifyPoolingLike(*this, getInput(), getKernel(),
                           getStride(), getPad(),
                           getOutput());
}

static LogicalResult verifyDwcArityN(Operation *op, size_t numInputs,
                                    size_t expected) {
  if (numInputs != expected)
    return op->emitOpError("expects ")
           << expected << " operands, got " << numInputs;
  return success();
}

static LogicalResult verifyDwcArityAtLeast(Operation *op, size_t numInputs,
                                          size_t min) {
  if (numInputs < min)
    return op->emitOpError("expects at least ")
           << min << " operands, got " << numInputs;
  return success();
}

static LogicalResult verifyDwcArityAtMost(Operation *op, size_t numInputs,
                                          size_t max) {
  if (numInputs > max)
    return op->emitOpError("expects at most ")
           << max << " operands, got " << numInputs;
  return success();
}

LogicalResult darwinn::AuxTensorTypeOp::verify() {
  // No shape contract: type descriptor carries no operand shape to check.
  return success();
}

LogicalResult darwinn::BinaryMapOp::verify() {
  if (failed(verifyDwcArityN(*this, getInputs().size(), 2)))
    return failure();
  if (!(*this)->hasAttr("function"))
    return (*this)->emitOpError(
        "expected op 'darwinn.binary_map' to have attribute 'function'");
  return success();
}

LogicalResult darwinn::BitcastOp::verify() {
  return verifyDwcArityN(*this, getInputs().size(), 1);
}

LogicalResult darwinn::BroadcastShardOp::verify() {
  return verifyDwcArityN(*this, getInputs().size(), 1);
}

LogicalResult darwinn::BroadcastSliceOp::verify() {
  return verifyDwcArityN(*this, getInputs().size(), 1);
}

LogicalResult darwinn::CeppsytDecompressOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult darwinn::CeppsytHwsvgmllkwtOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult darwinn::ChunkingReshapeOp::verify() {
  return verifyDwcArityN(*this, getInputs().size(), 1);
}

LogicalResult darwinn::CompressionModeOp::verify() {
  // No shape contract: mode selector carries no operand shape to check.
  return success();
}

LogicalResult darwinn::ComputeLoweringHintOp::verify() {
  // No shape contract: hint payload carries no operand shape to check.
  return success();
}

LogicalResult darwinn::ComputeOpOptionsOp::verify() {
  // No shape contract: option payload carries no operand shape to check.
  return success();
}

LogicalResult darwinn::ComputeTypeHintOp::verify() {
  // No shape contract: hint payload carries no operand shape to check.
  return success();
}

LogicalResult darwinn::ConditionScopeOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult darwinn::ConstBiasScaleOp::verify() {
  // No shape contract: const descriptor carries no operand shape to check.
  return success();
}

LogicalResult darwinn::ConstTypeOp::verify() {
  // No shape contract: type descriptor carries no operand shape to check.
  return success();
}

LogicalResult darwinn::ConstantGeneratorOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult darwinn::ConvolutionOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult darwinn::CostHintOp::verify() {
  // No shape contract: hint payload carries no operand shape to check.
  return success();
}

LogicalResult darwinn::CreateEmptyTensorOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult darwinn::CustomTilingOptionsOp::verify() {
  // No shape contract: option payload carries no operand shape to check.
  return success();
}

LogicalResult darwinn::DeviceTypeOp::verify() {
  // No shape contract: type descriptor carries no operand shape to check.
  return success();
}

LogicalResult darwinn::DiveOpOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult darwinn::DiveRefCwiseOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult darwinn::DtcInfoOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult darwinn::DynamicUpdateSliceWithOffsetsOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult darwinn::FastWalshHadamardTransformOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult darwinn::FenceOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult darwinn::FillOp::verify() {
  return verifyDwcArityAtMost(*this, getInputs().size(), 1);
}

LogicalResult darwinn::FilterOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult darwinn::FilterCmpOp::verify() {
  return verifyDwcArityN(*this, getInputs().size(), 2);
}

LogicalResult darwinn::FilterOutputOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult darwinn::FilteredIndicesDispatchModeOp::verify() {
  // No shape contract: mode selector carries no operand shape to check.
  return success();
}

LogicalResult darwinn::FullyConnectedZoutIndexedOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult darwinn::FunctionSymmetryOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult darwinn::GetFullTensorOfDynamicViewOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult darwinn::GetIndexedSliceOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult darwinn::GetTensorOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult darwinn::HlBitcastOp::verify() {
  return verifyDwcArityN(*this, getInputs().size(), 1);
}

LogicalResult darwinn::HostToSsramOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult darwinn::HostToSsramShardOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult darwinn::HostToTileOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult darwinn::HostToTileShardOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult darwinn::ImageFormatOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult darwinn::InfeedOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult darwinn::InnerOpOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult darwinn::InterpolateMethodOp::verify() {
  // No shape contract: method selector carries no operand shape to check.
  return success();
}

LogicalResult darwinn::IotaOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult darwinn::IsBoolOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult darwinn::IsParameterOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult darwinn::IsSparselyPackedOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult darwinn::JoinAttributesOp::verify() {
  // No shape contract: attribute payload carries no operand shape to check.
  return success();
}

LogicalResult darwinn::KernelLevelOp::verify() {
  // No shape contract: level descriptor carries no operand shape to check.
  return success();
}

LogicalResult darwinn::LaunchCustomKernelOp::verify() {
  // No shape contract: region contract absent, kernel body carries no operand shape to check.
  return success();
}

LogicalResult darwinn::LaunchFunctionOp::verify() {
  // No shape contract: region contract absent, callee body carries no operand shape to check.
  return success();
}

LogicalResult darwinn::LegacyInterpolateOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult darwinn::LinearFuncOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult darwinn::LocalCopyAttributesOp::verify() {
  // No shape contract: attribute payload carries no operand shape to check.
  return success();
}

LogicalResult darwinn::MappingOp::verify() {
  // No shape contract: mapping descriptor carries no operand shape to check.
  return success();
}


LogicalResult darwinn::MaterializeCastOp::verify() {
  return verifyDwcArityN(*this, getInputs().size(), 1);
}

LogicalResult darwinn::MaterializePolicyOp::verify() {
  // No shape contract: policy descriptor carries no operand shape to check.
  return success();
}

LogicalResult darwinn::MemSpaceOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult darwinn::MeshPadSliceOp::verify() {
  return verifyDwcArityAtLeast(*this, getInputs().size(), 1);
}

LogicalResult darwinn::MmaComputeOpOp::verify() {
  return verifyDwcArityAtLeast(*this, getInputs().size(), 1);
}

LogicalResult darwinn::MultimediaOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult darwinn::NarrowToNarrowOp::verify() {
  return verifyDwcArityN(*this, getInputs().size(), 1);
}

LogicalResult darwinn::NarrowToNarrowShardOp::verify() {
  return verifyDwcArityN(*this, getInputs().size(), 1);
}

LogicalResult darwinn::NarrowToNarrowSliceOp::verify() {
  return verifyDwcArityAtLeast(*this, getInputs().size(), 1);
}

LogicalResult darwinn::NarrowToWideOp::verify() {
  return verifyDwcArityN(*this, getInputs().size(), 1);
}

LogicalResult darwinn::NarrowToWideShardOp::verify() {
  return verifyDwcArityN(*this, getInputs().size(), 1);
}

LogicalResult darwinn::NarrowToWideSliceOp::verify() {
  return verifyDwcArityAtLeast(*this, getInputs().size(), 1);
}

LogicalResult darwinn::NluE8m0RoundingOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult darwinn::NluFuncOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult darwinn::NluPredicateOp::verify() {
  // No shape contract: predicate descriptor carries no operand shape to check.
  return success();
}

LogicalResult darwinn::NluPreprocessOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult darwinn::OutfeedOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult darwinn::PackedIndexOptionsOp::verify() {
  // No shape contract: option payload carries no operand shape to check.
  return success();
}

LogicalResult darwinn::ParallelMeshCopyOp::verify() {
  return verifyDwcArityN(*this, getInputs().size(), 2);
}

LogicalResult darwinn::PreemptionPointOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult darwinn::ProbeOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult darwinn::ReinterpretCastOp::verify() {
  return verifyDwcArityN(*this, getInputs().size(), 1);
}

LogicalResult darwinn::ReluOp::verify() {
  return verifyDwcArityN(*this, getInputs().size(), 1);
}

LogicalResult darwinn::ResamplerOptionsOp::verify() {
  // No shape contract: option payload carries no operand shape to check.
  return success();
}

LogicalResult darwinn::ReshapeOpOp::verify() {
  return verifyDwcArityN(*this, getInputs().size(), 1);
}

LogicalResult darwinn::RingToTileOptionsOp::verify() {
  // No shape contract: option payload carries no operand shape to check.
  return success();
}

LogicalResult darwinn::RingToTileSliceOp::verify() {
  return verifyDwcArityAtLeast(*this, getInputs().size(), 1);
}

LogicalResult darwinn::RkhyComputeOpOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult darwinn::RkhyCustomPaddingOp::verify() {
  return verifyDwcArityAtLeast(*this, getInputs().size(), 1);
}

LogicalResult darwinn::RkhyDepthToSpaceOpOp::verify() {
  return verifyDwcArityN(*this, getInputs().size(), 1);
}

LogicalResult darwinn::RkhyResidualAddOpOp::verify() {
  return verifyDwcArityN(*this, getInputs().size(), 2);
}

LogicalResult darwinn::RkhyUnaryComputeOpOp::verify() {
  return verifyDwcArityN(*this, getInputs().size(), 1);
}

LogicalResult darwinn::RngBitGeneratorOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult darwinn::ScalarRegisterToHostTransferOp::verify() {
  return verifyDwcArityAtLeast(*this, getInputs().size(), 1);
}

LogicalResult darwinn::ScalarRegisterToTileTransferOp::verify() {
  return verifyDwcArityAtLeast(*this, getInputs().size(), 1);
}

LogicalResult darwinn::ScaleOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult darwinn::ScatterOp::verify() {
  return verifyDwcArityAtLeast(*this, getInputs().size(), 2);
}

LogicalResult darwinn::SelectOp::verify() {
  return verifyDwcArityN(*this, getInputs().size(), 3);
}

LogicalResult darwinn::Slice1dExtentOp::verify() {
  return verifyDwcArityAtLeast(*this, getInputs().size(), 1);
}

LogicalResult darwinn::Slice1dExtentWithPaddingInfoOp::verify() {
  return verifyDwcArityAtLeast(*this, getInputs().size(), 1);
}

LogicalResult darwinn::SparseNarrowToWideOp::verify() {
  return verifyDwcArityN(*this, getInputs().size(), 1);
}

LogicalResult darwinn::SparseNarrowToWideSliceOp::verify() {
  return verifyDwcArityAtLeast(*this, getInputs().size(), 1);
}

LogicalResult darwinn::SparseTensorOpOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult darwinn::SparsityOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult darwinn::SparsityTypeOp::verify() {
  // No shape contract: type descriptor carries no operand shape to check.
  return success();
}

LogicalResult darwinn::SplineSegmentOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult darwinn::SplitOp::verify() {
  return verifyDwcArityAtLeast(*this, getInputs().size(), 1);
}

LogicalResult darwinn::StartOffsetAndStrideOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult darwinn::StaticComputeOpOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult darwinn::StaticSparseComputeOpOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult darwinn::StaticUnaryComputeOpOp::verify() {
  return verifyDwcArityN(*this, getInputs().size(), 1);
}

LogicalResult darwinn::StreamingComputeOpOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult darwinn::StreamingCopyOpOp::verify() {
  return verifyDwcArityN(*this, getInputs().size(), 2);
}

LogicalResult darwinn::StreamingSparseComputeOpOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult darwinn::StreamingSparseCopyOpOp::verify() {
  return verifyDwcArityN(*this, getInputs().size(), 2);
}

LogicalResult darwinn::StreamingUnaryComputeOpOp::verify() {
  return verifyDwcArityN(*this, getInputs().size(), 1);
}

LogicalResult darwinn::SwizzlingOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult darwinn::SynchronizedComputeOpOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult darwinn::SynchronizedCopyOpOp::verify() {
  return verifyDwcArityN(*this, getInputs().size(), 2);
}

LogicalResult darwinn::SynchronizedSparseComputeOpOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult darwinn::SynchronizedSparseCopyOpOp::verify() {
  return verifyDwcArityN(*this, getInputs().size(), 2);
}

LogicalResult darwinn::SynchronizedUnaryComputeOpOp::verify() {
  return verifyDwcArityN(*this, getInputs().size(), 1);
}

LogicalResult darwinn::TensorOpOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult darwinn::TensorOpShardOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult darwinn::TensorOpSliceOp::verify() {
  return verifyDwcArityAtLeast(*this, getInputs().size(), 1);
}

LogicalResult darwinn::TerminateOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult darwinn::TgcElementwiseAddOp::verify() {
  return verifyDwcArityN(*this, getInputs().size(), 2);
}

LogicalResult darwinn::TgcElementwiseMulOp::verify() {
  return verifyDwcArityN(*this, getInputs().size(), 2);
}

LogicalResult darwinn::TgcElementwiseSubOp::verify() {
  return verifyDwcArityN(*this, getInputs().size(), 2);
}

LogicalResult darwinn::TileToHostOp::verify() {
  return verifyDwcArityAtLeast(*this, getInputs().size(), 1);
}

LogicalResult darwinn::TileToHostShardOp::verify() {
  return verifyDwcArityAtLeast(*this, getInputs().size(), 1);
}

LogicalResult darwinn::TileToRingSliceOp::verify() {
  return verifyDwcArityAtLeast(*this, getInputs().size(), 1);
}

LogicalResult darwinn::TileToScalarRegisterTransferOp::verify() {
  return verifyDwcArityAtLeast(*this, getInputs().size(), 1);
}

LogicalResult darwinn::TileToTileOp::verify() {
  return verifyDwcArityAtLeast(*this, getInputs().size(), 1);
}

LogicalResult darwinn::TileToTileShardOp::verify() {
  return verifyDwcArityAtLeast(*this, getInputs().size(), 1);
}

LogicalResult darwinn::UnaryMapOp::verify() {
  return verifyDwcArityN(*this, getInputs().size(), 1);
}

LogicalResult darwinn::UnaryTensorOpOp::verify() {
  return verifyDwcArityN(*this, getInputs().size(), 1);
}

LogicalResult darwinn::VexInfoOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult darwinn::VrgkhOperationModeOp::verify() {
  // No shape contract: mode selector carries no operand shape to check.
  return success();
}

LogicalResult darwinn::WhileOp::verify() {
  // No shape contract: region contract absent, loop body carries no operand shape to check.
  return success();
}

LogicalResult darwinn::WideToNarrowOp::verify() {
  return verifyDwcArityN(*this, getInputs().size(), 1);
}

LogicalResult darwinn::WideToNarrowSliceOp::verify() {
  return verifyDwcArityAtLeast(*this, getInputs().size(), 1);
}

LogicalResult darwinn::YieldOp::verify() {
  // No shape contract: region contract absent, terminator carries no operand shape to check.
  return success();
}

LogicalResult darwinn::ZeroPointOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult dwc::AddOp::verify() {
  if (!(*this)->hasAttr("activation_function"))
    return (*this)->emitOpError("expected op 'dwc.add' to have attribute 'activation_function'");
  return success();
}

LogicalResult dwc::AtanOp::verify() {
  if (getInputs().size() != 1)
    return emitOpError("expects 1 operands, got ") << getInputs().size();
  return success();
}

LogicalResult dwc::BatchMatrixNmsOp::verify() {
  if (!(*this)->hasAttr("max_output_size"))
    return (*this)->emitOpError("expected op 'dwc.batch_matrix_nms' to have attribute 'max_output_size'");
  if (!(*this)->hasAttr("score_threshold"))
    return (*this)->emitOpError("expected op 'dwc.batch_matrix_nms' to have attribute 'score_threshold'");
  if (!(*this)->hasAttr("sigma"))
    return (*this)->emitOpError("expected op 'dwc.batch_matrix_nms' to have attribute 'sigma'");
  if (!(*this)->hasAttr("suppress_top_k"))
    return (*this)->emitOpError("expected op 'dwc.batch_matrix_nms' to have attribute 'suppress_top_k'");
  return success();
}

LogicalResult dwc::BitcastOp::verify() {
  if (!(*this)->hasAttr("output_element_type"))
    return (*this)->emitOpError("expected op 'dwc.bitcast' to have attribute 'output_element_type'");
  return success();
}

LogicalResult dwc::CastOp::verify() {
  if (getInputs().size() != 1)
    return emitOpError("expects 1 operands, got ") << getInputs().size();
  return success();
}

LogicalResult dwc::CeilOp::verify() {
  if (getInputs().size() != 1)
    return emitOpError("expects 1 operands, got ") << getInputs().size();
  return success();
}

LogicalResult dwc::ClassifierOp::verify() {
  if (!(*this)->hasAttr("axis"))
    return (*this)->emitOpError("expected op 'dwc.classifier' to have attribute 'axis'");
  if (!(*this)->hasAttr("beta"))
    return (*this)->emitOpError("expected op 'dwc.classifier' to have attribute 'beta'");
  if (!(*this)->hasAttr("op_type"))
    return (*this)->emitOpError("expected op 'dwc.classifier' to have attribute 'op_type'");
  return success();
}

LogicalResult dwc::CompareOp::verify() {
  if (!(*this)->hasAttr("activation_function"))
    return (*this)->emitOpError("expected op 'dwc.compare' to have attribute 'activation_function'");
  if (!(*this)->hasAttr("compare_type"))
    return (*this)->emitOpError("expected op 'dwc.compare' to have attribute 'compare_type'");
  return success();
}

LogicalResult dwc::ConcatenationOp::verify() {
  if (!(*this)->hasAttr("mode"))
    return (*this)->emitOpError("expected op 'dwc.concatenation' to have attribute 'mode'");
  return success();
}

LogicalResult dwc::ConstOp::verify() {
  if (!(*this)->hasAttr("value"))
    return (*this)->emitOpError("expected op 'dwc.const' to have attribute 'value'");
  return success();
}

LogicalResult dwc::ConvolutionOp::verify() {
  if (!(*this)->hasAttr("activation_function"))
    return (*this)->emitOpError("expected op 'dwc.convolution' to have attribute 'activation_function'");
  if (!(*this)->hasAttr("cell_operation"))
    return (*this)->emitOpError("expected op 'dwc.convolution' to have attribute 'cell_operation'");
  if (!(*this)->hasAttr("pad"))
    return (*this)->emitOpError("expected op 'dwc.convolution' to have attribute 'pad'");
  if (!(*this)->hasAttr("x_dilation_rate"))
    return (*this)->emitOpError("expected op 'dwc.convolution' to have attribute 'x_dilation_rate'");
  if (!(*this)->hasAttr("x_stride"))
    return (*this)->emitOpError("expected op 'dwc.convolution' to have attribute 'x_stride'");
  if (!(*this)->hasAttr("y_dilation_rate"))
    return (*this)->emitOpError("expected op 'dwc.convolution' to have attribute 'y_dilation_rate'");
  if (!(*this)->hasAttr("y_stride"))
    return (*this)->emitOpError("expected op 'dwc.convolution' to have attribute 'y_stride'");
  return success();
}

LogicalResult dwc::ConvolutionV2Op::verify() {
  if (!(*this)->hasAttr("activation_function"))
    return (*this)->emitOpError("expected op 'dwc.convolution_v2' to have attribute 'activation_function'");
  if (!(*this)->hasAttr("cell_operation"))
    return (*this)->emitOpError("expected op 'dwc.convolution_v2' to have attribute 'cell_operation'");
  if (!(*this)->hasAttr("pad"))
    return (*this)->emitOpError("expected op 'dwc.convolution_v2' to have attribute 'pad'");
  if (!(*this)->hasAttr("x_dilation_rate"))
    return (*this)->emitOpError("expected op 'dwc.convolution_v2' to have attribute 'x_dilation_rate'");
  if (!(*this)->hasAttr("x_stride"))
    return (*this)->emitOpError("expected op 'dwc.convolution_v2' to have attribute 'x_stride'");
  if (!(*this)->hasAttr("y_dilation_rate"))
    return (*this)->emitOpError("expected op 'dwc.convolution_v2' to have attribute 'y_dilation_rate'");
  if (!(*this)->hasAttr("y_stride"))
    return (*this)->emitOpError("expected op 'dwc.convolution_v2' to have attribute 'y_stride'");
  return success();
}

LogicalResult dwc::CosOp::verify() {
  if (getInputs().size() != 1)
    return emitOpError("expects 1 operands, got ") << getInputs().size();
  return success();
}

LogicalResult dwc::CumulativeOp::verify() {
  if (getInputs().size() != 1)
    return emitOpError("expects 1 operands, got ") << getInputs().size();
  if (!(*this)->hasAttr("axis"))
    return (*this)->emitOpError("expected op 'dwc.cumulative' to have attribute 'axis'");
  if (!(*this)->hasAttr("exclusive"))
    return (*this)->emitOpError("expected op 'dwc.cumulative' to have attribute 'exclusive'");
  if (!(*this)->hasAttr("op_type"))
    return (*this)->emitOpError("expected op 'dwc.cumulative' to have attribute 'op_type'");
  return success();
}

LogicalResult dwc::CwiseOp::verify() {
  if (getInputs().size() != 2)
    return emitOpError("expects 2 operands, got ") << getInputs().size();
  if (!(*this)->hasAttr("activation_function"))
    return (*this)->emitOpError("expected op 'dwc.cwise' to have attribute 'activation_function'");
  if (!(*this)->hasAttr("op_type"))
    return (*this)->emitOpError("expected op 'dwc.cwise' to have attribute 'op_type'");
  return success();
}

LogicalResult dwc::DepthwiseConvolutionV2Op::verify() {
  if (!(*this)->hasAttr("activation_function"))
    return (*this)->emitOpError("expected op 'dwc.depthwise_convolution_v2' to have attribute 'activation_function'");
  if (!(*this)->hasAttr("cell_operation"))
    return (*this)->emitOpError("expected op 'dwc.depthwise_convolution_v2' to have attribute 'cell_operation'");
  if (!(*this)->hasAttr("depth_multiplier"))
    return (*this)->emitOpError("expected op 'dwc.depthwise_convolution_v2' to have attribute 'depth_multiplier'");
  if (!(*this)->hasAttr("pad"))
    return (*this)->emitOpError("expected op 'dwc.depthwise_convolution_v2' to have attribute 'pad'");
  if (!(*this)->hasAttr("x_dilation_rate"))
    return (*this)->emitOpError("expected op 'dwc.depthwise_convolution_v2' to have attribute 'x_dilation_rate'");
  if (!(*this)->hasAttr("x_stride"))
    return (*this)->emitOpError("expected op 'dwc.depthwise_convolution_v2' to have attribute 'x_stride'");
  if (!(*this)->hasAttr("y_dilation_rate"))
    return (*this)->emitOpError("expected op 'dwc.depthwise_convolution_v2' to have attribute 'y_dilation_rate'");
  if (!(*this)->hasAttr("y_stride"))
    return (*this)->emitOpError("expected op 'dwc.depthwise_convolution_v2' to have attribute 'y_stride'");
  return success();
}

LogicalResult dwc::DivideOp::verify() {
  if (!(*this)->hasAttr("activation_function"))
    return (*this)->emitOpError("expected op 'dwc.divide' to have attribute 'activation_function'");
  return success();
}

LogicalResult dwc::DynamicSliceOp::verify() {
  if (!(*this)->hasAttr("mode"))
    return (*this)->emitOpError("expected op 'dwc.dynamic_slice' to have attribute 'mode'");
  if (!(*this)->hasAttr("read_location"))
    return (*this)->emitOpError("expected op 'dwc.dynamic_slice' to have attribute 'read_location'");
  if (!(*this)->hasAttr("slice_size"))
    return (*this)->emitOpError("expected op 'dwc.dynamic_slice' to have attribute 'slice_size'");
  return success();
}

LogicalResult dwc::DynamicUpdateSliceOp::verify() {
  if (!(*this)->hasAttr("mode"))
    return (*this)->emitOpError("expected op 'dwc.dynamic_update_slice' to have attribute 'mode'");
  if (!(*this)->hasAttr("write_location"))
    return (*this)->emitOpError("expected op 'dwc.dynamic_update_slice' to have attribute 'write_location'");
  return success();
}

LogicalResult dwc::ErfOp::verify() {
  if (getInputs().size() != 1)
    return emitOpError("expects 1 operands, got ") << getInputs().size();
  return success();
}

LogicalResult dwc::ExpOp::verify() {
  if (getInputs().size() != 1)
    return emitOpError("expects 1 operands, got ") << getInputs().size();
  return success();
}

LogicalResult dwc::Expm1Op::verify() {
  if (getInputs().size() != 1)
    return emitOpError("expects 1 operands, got ") << getInputs().size();
  return success();
}

LogicalResult dwc::FloorOp::verify() {
  if (getInputs().size() != 1)
    return emitOpError("expects 1 operands, got ") << getInputs().size();
  return success();
}

LogicalResult dwc::FloorDivOp::verify() {
  if (getInputs().size() != 2)
    return emitOpError("expects 2 operands, got ") << getInputs().size();
  return success();
}

LogicalResult dwc::FullyConnectedOp::verify() {
  if (!(*this)->hasAttr("activation_function"))
    return (*this)->emitOpError("expected op 'dwc.fully_connected' to have attribute 'activation_function'");
  if (!(*this)->hasAttr("cell_operation"))
    return (*this)->emitOpError("expected op 'dwc.fully_connected' to have attribute 'cell_operation'");
  return success();
}

LogicalResult dwc::GatherOp::verify() {
  if (!(*this)->hasAttr("axis"))
    return (*this)->emitOpError("expected op 'dwc.gather' to have attribute 'axis'");
  if (!(*this)->hasAttr("batch_dims"))
    return (*this)->emitOpError("expected op 'dwc.gather' to have attribute 'batch_dims'");
  return success();
}

LogicalResult dwc::GenericComputeOp::verify() {
  if (!(*this)->hasAttr("activation_function"))
    return (*this)->emitOpError("expected op 'dwc.generic_compute' to have attribute 'activation_function'");
  if (!(*this)->hasAttr("indexing_maps"))
    return (*this)->emitOpError("expected op 'dwc.generic_compute' to have attribute 'indexing_maps'");
  if (!(*this)->hasAttr("linear_function"))
    return (*this)->emitOpError("expected op 'dwc.generic_compute' to have attribute 'linear_function'");
  return success();
}

LogicalResult dwc::GenericConstantOp::verify() {
  if (!(*this)->hasAttr("value"))
    return (*this)->emitOpError("expected op 'dwc.generic_constant' to have attribute 'value'");
  return success();
}

LogicalResult dwc::GenericConvOp::verify() {
  if (!(*this)->hasAttr("activation_function"))
    return (*this)->emitOpError("expected op 'dwc.generic_conv' to have attribute 'activation_function'");
  if (!(*this)->hasAttr("batch_group_count"))
    return (*this)->emitOpError("expected op 'dwc.generic_conv' to have attribute 'batch_group_count'");
  if (!(*this)->hasAttr("feature_group_count"))
    return (*this)->emitOpError("expected op 'dwc.generic_conv' to have attribute 'feature_group_count'");
  if (!(*this)->hasAttr("input_dilation"))
    return (*this)->emitOpError("expected op 'dwc.generic_conv' to have attribute 'input_dilation'");
  if (!(*this)->hasAttr("padding_amount"))
    return (*this)->emitOpError("expected op 'dwc.generic_conv' to have attribute 'padding_amount'");
  if (!(*this)->hasAttr("param_dilation"))
    return (*this)->emitOpError("expected op 'dwc.generic_conv' to have attribute 'param_dilation'");
  if (!(*this)->hasAttr("param_reversal"))
    return (*this)->emitOpError("expected op 'dwc.generic_conv' to have attribute 'param_reversal'");
  if (!(*this)->hasAttr("stride"))
    return (*this)->emitOpError("expected op 'dwc.generic_conv' to have attribute 'stride'");
  return success();
}

LogicalResult dwc::GenericDotOp::verify() {
  if (!(*this)->hasAttr("activation_function"))
    return (*this)->emitOpError("expected op 'dwc.generic_dot' to have attribute 'activation_function'");
  if (!(*this)->hasAttr("batch_dim_count"))
    return (*this)->emitOpError("expected op 'dwc.generic_dot' to have attribute 'batch_dim_count'");
  if (!(*this)->hasAttr("contracting_dim_count"))
    return (*this)->emitOpError("expected op 'dwc.generic_dot' to have attribute 'contracting_dim_count'");
  return success();
}

LogicalResult dwc::Log1pOp::verify() {
  if (getInputs().size() != 1)
    return emitOpError("expects 1 operands, got ") << getInputs().size();
  return success();
}

LogicalResult dwc::LogisticOp::verify() {
  if (getInputs().size() != 1)
    return emitOpError("expects 1 operands, got ") << getInputs().size();
  return success();
}

LogicalResult dwc::MatrixMultiplyOp::verify() {
  if (!(*this)->hasAttr("activation_function"))
    return (*this)->emitOpError("expected op 'dwc.matrix_multiply' to have attribute 'activation_function'");
  if (!(*this)->hasAttr("transpose_rhs"))
    return (*this)->emitOpError("expected op 'dwc.matrix_multiply' to have attribute 'transpose_rhs'");
  return success();
}

LogicalResult dwc::MaximumOp::verify() {
  if (!(*this)->hasAttr("activation_function"))
    return (*this)->emitOpError("expected op 'dwc.maximum' to have attribute 'activation_function'");
  return success();
}

LogicalResult dwc::MinimumOp::verify() {
  if (!(*this)->hasAttr("activation_function"))
    return (*this)->emitOpError("expected op 'dwc.minimum' to have attribute 'activation_function'");
  return success();
}

LogicalResult dwc::MultiplyOp::verify() {
  if (!(*this)->hasAttr("activation_function"))
    return (*this)->emitOpError("expected op 'dwc.multiply' to have attribute 'activation_function'");
  return success();
}

LogicalResult dwc::NotOp::verify() {
  if (getInputs().size() != 1)
    return emitOpError("expects 1 operands, got ") << getInputs().size();
  return success();
}

LogicalResult dwc::OneHotOp::verify() {
  if (getInputs().size() != 1)
    return emitOpError("expects 1 operands, got ") << getInputs().size();
  if (!(*this)->hasAttr("axis"))
    return (*this)->emitOpError("expected op 'dwc.one_hot' to have attribute 'axis'");
  return success();
}

LogicalResult dwc::PaddingOp::verify() {
  if (getInputs().size() != 1)
    return emitOpError("expects 1 operands, got ") << getInputs().size();
  if (!(*this)->hasAttr("dimension"))
    return (*this)->emitOpError("expected op 'dwc.padding' to have attribute 'dimension'");
  if (!(*this)->hasAttr("padding_value"))
    return (*this)->emitOpError("expected op 'dwc.padding' to have attribute 'padding_value'");
  if (!(*this)->hasAttr("post_padding"))
    return (*this)->emitOpError("expected op 'dwc.padding' to have attribute 'post_padding'");
  if (!(*this)->hasAttr("pre_padding"))
    return (*this)->emitOpError("expected op 'dwc.padding' to have attribute 'pre_padding'");
  return success();
}

LogicalResult dwc::PopCountOp::verify() {
  if (getInputs().size() != 1)
    return emitOpError("expects 1 operands, got ") << getInputs().size();
  return success();
}

LogicalResult dwc::PowOp::verify() {
  if (getInputs().size() != 2)
    return emitOpError("expects 2 operands, got ") << getInputs().size();
  return success();
}

LogicalResult dwc::PseudoSplitOp::verify() {
  if (getInputs().size() != 2)
    return emitOpError("expects 2 operands, got ") << getInputs().size();
  if (!(*this)->hasAttr("num_splits"))
    return (*this)->emitOpError("expected op 'dwc.pseudo_split' to have attribute 'num_splits'");
  return success();
}

LogicalResult dwc::ReductionOp::verify() {
  if (getInputs().size() != 1)
    return emitOpError("expects 1 operands, got ") << getInputs().size();
  if (!(*this)->hasAttr("activation_function"))
    return (*this)->emitOpError("expected op 'dwc.reduction' to have attribute 'activation_function'");
  if (!(*this)->hasAttr("dimensions"))
    return (*this)->emitOpError("expected op 'dwc.reduction' to have attribute 'dimensions'");
  if (!(*this)->hasAttr("op_type"))
    return (*this)->emitOpError("expected op 'dwc.reduction' to have attribute 'op_type'");
  return success();
}

LogicalResult dwc::RemainderOp::verify() {
  if (!(*this)->hasAttr("activation_function"))
    return (*this)->emitOpError("expected op 'dwc.remainder' to have attribute 'activation_function'");
  return success();
}

LogicalResult dwc::RescalingOp::verify() {
  if (getInputs().size() != 1)
    return emitOpError("expects 1 operands, got ") << getInputs().size();
  if (!(*this)->hasAttr("activation_function"))
    return (*this)->emitOpError("expected op 'dwc.rescaling' to have attribute 'activation_function'");
  if (!(*this)->hasAttr("output_activation_per_z_out_scales"))
    return (*this)->emitOpError("expected op 'dwc.rescaling' to have attribute 'output_activation_per_z_out_scales'");
  if (!(*this)->hasAttr("per_z_out_scales_padding"))
    return (*this)->emitOpError("expected op 'dwc.rescaling' to have attribute 'per_z_out_scales_padding'");
  return success();
}

LogicalResult dwc::ReshapeOp::verify() {
  if (getInputs().size() != 1)
    return emitOpError("expects 1 operands, got ") << getInputs().size();
  return success();
}

LogicalResult dwc::RoundOp::verify() {
  if (getInputs().size() != 1)
    return emitOpError("expects 1 operands, got ") << getInputs().size();
  return success();
}

LogicalResult dwc::RoundNearestAfzOp::verify() {
  if (getInputs().size() != 1)
    return emitOpError("expects 1 operands, got ") << getInputs().size();
  return success();
}

LogicalResult dwc::RsqrtOp::verify() {
  if (getInputs().size() != 1)
    return emitOpError("expects 1 operands, got ") << getInputs().size();
  return success();
}

LogicalResult dwc::ScalarOp::verify() {
  if (!(*this)->hasAttr("op_type"))
    return (*this)->emitOpError("expected op 'dwc.scalar' to have attribute 'op_type'");
  return success();
}

LogicalResult dwc::ScatterNdOp::verify() {
  if (!(*this)->hasAttr("shape"))
    return (*this)->emitOpError("expected op 'dwc.scatter_nd' to have attribute 'shape'");
  return success();
}

LogicalResult dwc::SelectOp::verify() {
  if (getInputs().size() != 2)
    return emitOpError("expects 2 operands, got ") << getInputs().size();
  return success();
}

LogicalResult dwc::SignOp::verify() {
  if (getInputs().size() != 1)
    return emitOpError("expects 1 operands, got ") << getInputs().size();
  if (!(*this)->hasAttr("preserve_negative_zero"))
    return (*this)->emitOpError("expected op 'dwc.sign' to have attribute 'preserve_negative_zero'");
  return success();
}

LogicalResult dwc::SinOp::verify() {
  if (getInputs().size() != 1)
    return emitOpError("expects 1 operands, got ") << getInputs().size();
  return success();
}

LogicalResult dwc::SliceOp::verify() {
  if (!(*this)->hasAttr("in_begin"))
    return (*this)->emitOpError("expected op 'dwc.slice' to have attribute 'in_begin'");
  if (!(*this)->hasAttr("in_size"))
    return (*this)->emitOpError("expected op 'dwc.slice' to have attribute 'in_size'");
  if (!(*this)->hasAttr("mode"))
    return (*this)->emitOpError("expected op 'dwc.slice' to have attribute 'mode'");
  return success();
}

LogicalResult dwc::SortOp::verify() {
  if (!(*this)->hasAttr("compare_tuple_projection"))
    return (*this)->emitOpError("expected op 'dwc.sort' to have attribute 'compare_tuple_projection'");
  if (!(*this)->hasAttr("compare_type"))
    return (*this)->emitOpError("expected op 'dwc.sort' to have attribute 'compare_type'");
  if (!(*this)->hasAttr("dimension"))
    return (*this)->emitOpError("expected op 'dwc.sort' to have attribute 'dimension'");
  if (!(*this)->hasAttr("is_stable"))
    return (*this)->emitOpError("expected op 'dwc.sort' to have attribute 'is_stable'");
  return success();
}

LogicalResult dwc::SqrtOp::verify() {
  if (getInputs().size() != 1)
    return emitOpError("expects 1 operands, got ") << getInputs().size();
  return success();
}

LogicalResult dwc::SubtractOp::verify() {
  if (!(*this)->hasAttr("activation_function"))
    return (*this)->emitOpError("expected op 'dwc.subtract' to have attribute 'activation_function'");
  return success();
}

LogicalResult dwc::TanOp::verify() {
  if (getInputs().size() != 1)
    return emitOpError("expects 1 operands, got ") << getInputs().size();
  return success();
}

LogicalResult dwc::TanhOp::verify() {
  if (getInputs().size() != 1)
    return emitOpError("expects 1 operands, got ") << getInputs().size();
  return success();
}

LogicalResult dwc::TransposeOp::verify() {
  if (getInputs().size() != 1)
    return emitOpError("expects 1 operands, got ") << getInputs().size();
  if (!(*this)->hasAttr("permutation"))
    return (*this)->emitOpError("expected op 'dwc.transpose' to have attribute 'permutation'");
  return success();
}

LogicalResult dwc::TransposedConvolutionOp::verify() {
  if (!(*this)->hasAttr("activation_function"))
    return (*this)->emitOpError("expected op 'dwc.transposed_convolution' to have attribute 'activation_function'");
  if (!(*this)->hasAttr("cell_operation"))
    return (*this)->emitOpError("expected op 'dwc.transposed_convolution' to have attribute 'cell_operation'");
  if (!(*this)->hasAttr("pad"))
    return (*this)->emitOpError("expected op 'dwc.transposed_convolution' to have attribute 'pad'");
  if (!(*this)->hasAttr("x_dilation_rate"))
    return (*this)->emitOpError("expected op 'dwc.transposed_convolution' to have attribute 'x_dilation_rate'");
  if (!(*this)->hasAttr("x_out_dim"))
    return (*this)->emitOpError("expected op 'dwc.transposed_convolution' to have attribute 'x_out_dim'");
  if (!(*this)->hasAttr("x_stride"))
    return (*this)->emitOpError("expected op 'dwc.transposed_convolution' to have attribute 'x_stride'");
  if (!(*this)->hasAttr("y_dilation_rate"))
    return (*this)->emitOpError("expected op 'dwc.transposed_convolution' to have attribute 'y_dilation_rate'");
  if (!(*this)->hasAttr("y_out_dim"))
    return (*this)->emitOpError("expected op 'dwc.transposed_convolution' to have attribute 'y_out_dim'");
  if (!(*this)->hasAttr("y_stride"))
    return (*this)->emitOpError("expected op 'dwc.transposed_convolution' to have attribute 'y_stride'");
  return success();
}

LogicalResult dwc::UnsortedSegmentReduceOp::verify() {
  if (!(*this)->hasAttr("num_segments"))
    return (*this)->emitOpError("expected op 'dwc.unsorted_segment_reduce' to have attribute 'num_segments'");
  if (!(*this)->hasAttr("op_type"))
    return (*this)->emitOpError("expected op 'dwc.unsorted_segment_reduce' to have attribute 'op_type'");
  return success();
}
