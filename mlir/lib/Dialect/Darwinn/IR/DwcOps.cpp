//===- DwcOps.cpp - MLIR Dialect for DWC ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Darwinn/IR/DwcOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"

using namespace mlir;
using namespace mlir::dwc;

#include "mlir/Dialect/Darwinn/IR/DwcOpsDialect.cpp.inc"

#include "mlir/Dialect/Darwinn/IR/DwcOpsEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/Darwinn/IR/DwcAttributes.h.inc"
#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/Darwinn/IR/DwcAttributes.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/Darwinn/IR/DwcOps.cpp.inc"

//===----------------------------------------------------------------------===//
// DWC dialect initialization.
//===----------------------------------------------------------------------===//

void DwcDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/Darwinn/IR/DwcAttributes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Darwinn/IR/DwcOps.cpp.inc"
      >();
}

LogicalResult dwc::AddOp::verify() {
  if (!(*this)->hasAttr("activation_function"))
    return (*this)->emitOpError("expected op 'dwc.add' to have attribute 'activation_function'");
  if (!llvm::isa<ActivationFunctionAttr>((*this)->getAttr("activation_function")))
    return (*this)->emitOpError("attribute 'activation_function' expects ActivationFunctionAttr");
  auto activation = llvm::cast<ActivationFunctionAttr>((*this)->getAttr("activation_function")).getValue();
  if (activation != ActivationFunction::None && activation != ActivationFunction::Relu)
    return (*this)->emitOpError("attribute 'activation_function' expects NONE or RELU");
  return success();
}

LogicalResult dwc::AtanOp::verify() {
  if (getInputs().size() != 1)
    return emitOpError("expects 1 operands, got ") << getInputs().size();
  auto tensor = llvm::dyn_cast<TensorType>(getInputs()[0].getType());
  Type element = tensor ? tensor.getElementType() : getInputs()[0].getType();
  if (llvm::isa<Float16Type, BFloat16Type, Float32Type>(element))
    return success();
  if (auto integer = llvm::dyn_cast<IntegerType>(element)) {
    if (integer.isSignless() && (integer.getWidth() == 1 || integer.getWidth() == 8 || integer.getWidth() == 16))
      return success();
  }
  return (*this)->emitOpError("operand 0 expects i1, i8, i16, f16, bf16, or f32");
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
  if (!llvm::isa<IntegerAttr>((*this)->getAttr("max_output_size")))
    return (*this)->emitOpError("attribute 'max_output_size' expects IntegerAttr");
  if (!llvm::isa<FloatAttr>((*this)->getAttr("score_threshold")))
    return (*this)->emitOpError("attribute 'score_threshold' expects FloatAttr");
  if (!llvm::isa<FloatAttr>((*this)->getAttr("sigma")))
    return (*this)->emitOpError("attribute 'sigma' expects FloatAttr");
  if (!llvm::isa<IntegerAttr>((*this)->getAttr("suppress_top_k")))
    return (*this)->emitOpError("attribute 'suppress_top_k' expects IntegerAttr");
  return success();
}

LogicalResult dwc::BitcastOp::verify() {
  if (!(*this)->hasAttr("output_element_type"))
    return (*this)->emitOpError("expected op 'dwc.bitcast' to have attribute 'output_element_type'");
  if (!llvm::isa<TypeAttr>((*this)->getAttr("output_element_type")))
    return (*this)->emitOpError("attribute 'output_element_type' expects TypeAttr");
  return success();
}

LogicalResult dwc::CastOp::verify() {
  if (getInputs().size() != 1)
    return emitOpError("expects 1 operands, got ") << getInputs().size();
  auto tensor = llvm::dyn_cast<TensorType>(getInputs()[0].getType());
  Type element = tensor ? tensor.getElementType() : getInputs()[0].getType();
  if (llvm::isa<Float16Type, BFloat16Type>(element))
    return success();
  return (*this)->emitOpError("operand 0 expects 16-bit float or bfloat16");
}

LogicalResult dwc::CeilOp::verify() {
  if (getInputs().size() != 1)
    return emitOpError("expects 1 operands, got ") << getInputs().size();
  auto tensor = llvm::dyn_cast<TensorType>(getInputs()[0].getType());
  Type element = tensor ? tensor.getElementType() : getInputs()[0].getType();
  if (llvm::isa<Float16Type, BFloat16Type, Float32Type>(element))
    return success();
  if (auto integer = llvm::dyn_cast<IntegerType>(element)) {
    if (integer.isSignless() && (integer.getWidth() == 1 || integer.getWidth() == 8 || integer.getWidth() == 16 || integer.getWidth() == 32))
      return success();
  }
  return (*this)->emitOpError("operand 0 expects i1, i8, i16, i32, f16, bf16, or f32");
}


LogicalResult dwc::ClassifierOp::verify() {
  if (!(*this)->hasAttr("axis"))
    return (*this)->emitOpError("expected op 'dwc.classifier' to have attribute 'axis'");
  if (!(*this)->hasAttr("beta"))
    return (*this)->emitOpError("expected op 'dwc.classifier' to have attribute 'beta'");
  if (!(*this)->hasAttr("op_type"))
    return (*this)->emitOpError("expected op 'dwc.classifier' to have attribute 'op_type'");
  if (!llvm::isa<IntegerAttr>((*this)->getAttr("axis")))
    return (*this)->emitOpError("attribute 'axis' expects IntegerAttr");
  if (!llvm::isa<FloatAttr>((*this)->getAttr("beta")))
    return (*this)->emitOpError("attribute 'beta' expects FloatAttr");
  if (!llvm::isa<ClassificationTypeAttr>((*this)->getAttr("op_type")))
    return (*this)->emitOpError("attribute 'op_type' expects ClassificationTypeAttr");
  if (llvm::cast<IntegerAttr>((*this)->getAttr("axis")).getInt() != -1)
    return (*this)->emitOpError("attribute 'axis' expects -1");
  if (llvm::cast<FloatAttr>((*this)->getAttr("beta")).getValueAsDouble() != 1.0)
    return (*this)->emitOpError("attribute 'beta' expects 1.0");
  if (llvm::cast<ClassificationTypeAttr>((*this)->getAttr("op_type")).getValue() != ClassificationType::Softmax)
    return (*this)->emitOpError("attribute 'op_type' expects SOFTMAX");
  return success();
}

LogicalResult dwc::CompareOp::verify() {
  if (!(*this)->hasAttr("activation_function"))
    return (*this)->emitOpError("expected op 'dwc.compare' to have attribute 'activation_function'");
  if (!(*this)->hasAttr("compare_type"))
    return (*this)->emitOpError("expected op 'dwc.compare' to have attribute 'compare_type'");
  if (!llvm::isa<ActivationFunctionAttr>((*this)->getAttr("activation_function")))
    return (*this)->emitOpError("attribute 'activation_function' expects ActivationFunctionAttr");
  if (!llvm::isa<ComparisonTypeAttr>((*this)->getAttr("compare_type")))
    return (*this)->emitOpError("attribute 'compare_type' expects ComparisonTypeAttr");
  return success();
}

LogicalResult dwc::ConcatenationOp::verify() {
  if (!(*this)->hasAttr("mode"))
    return (*this)->emitOpError("expected op 'dwc.concatenation' to have attribute 'mode'");
  if (!llvm::isa<IntegerAttr>((*this)->getAttr("mode")))
    return (*this)->emitOpError("attribute 'mode' expects IntegerAttr");
  auto mode = llvm::cast<IntegerAttr>((*this)->getAttr("mode"));
  if (!mode.getType().isSignlessInteger(32))
    return (*this)->emitOpError("attribute 'mode' expects 32-bit signless integer");
  return success();
}

LogicalResult dwc::ConstOp::verify() {
  if (!(*this)->hasAttr("value"))
    return (*this)->emitOpError("expected op 'dwc.const' to have attribute 'value'");
  if (!llvm::isa<ElementsAttr>((*this)->getAttr("value")))
    return (*this)->emitOpError("attribute 'value' expects ElementsAttr");
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
  if (!llvm::isa<ActivationFunctionAttr>((*this)->getAttr("activation_function")))
    return (*this)->emitOpError("attribute 'activation_function' expects ActivationFunctionAttr");
  if (!llvm::isa<CellOperationAttr>((*this)->getAttr("cell_operation")))
    return (*this)->emitOpError("attribute 'cell_operation' expects CellOperationAttr");
  if (!llvm::isa<PaddingAttr>((*this)->getAttr("pad")))
    return (*this)->emitOpError("attribute 'pad' expects PaddingAttr");
  if (!llvm::isa<IntegerAttr>((*this)->getAttr("x_dilation_rate")))
    return (*this)->emitOpError("attribute 'x_dilation_rate' expects IntegerAttr");
  if (!llvm::isa<IntegerAttr>((*this)->getAttr("x_stride")))
    return (*this)->emitOpError("attribute 'x_stride' expects IntegerAttr");
  if (!llvm::isa<IntegerAttr>((*this)->getAttr("y_dilation_rate")))
    return (*this)->emitOpError("attribute 'y_dilation_rate' expects IntegerAttr");
  if (!llvm::isa<IntegerAttr>((*this)->getAttr("y_stride")))
    return (*this)->emitOpError("attribute 'y_stride' expects IntegerAttr");
  if (llvm::cast<ActivationFunctionAttr>((*this)->getAttr("activation_function")).getValue() != ActivationFunction::None)
    return (*this)->emitOpError("attribute 'activation_function' expects NONE");
  for (const char *name : {"x_dilation_rate", "x_stride", "y_dilation_rate", "y_stride"}) {
    if (llvm::cast<IntegerAttr>((*this)->getAttr(name)).getInt() != 1)
      return (*this)->emitOpError("attribute '") << name << "' expects 1";
  }
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
  if (!llvm::isa<ActivationFunctionAttr>((*this)->getAttr("activation_function")))
    return (*this)->emitOpError("attribute 'activation_function' expects ActivationFunctionAttr");
  if (!llvm::isa<CellOperationAttr>((*this)->getAttr("cell_operation")))
    return (*this)->emitOpError("attribute 'cell_operation' expects CellOperationAttr");
  if (!llvm::isa<PaddingAttr>((*this)->getAttr("pad")))
    return (*this)->emitOpError("attribute 'pad' expects PaddingAttr");
  if (!llvm::isa<IntegerAttr>((*this)->getAttr("x_dilation_rate")))
    return (*this)->emitOpError("attribute 'x_dilation_rate' expects IntegerAttr");
  if (!llvm::isa<IntegerAttr>((*this)->getAttr("x_stride")))
    return (*this)->emitOpError("attribute 'x_stride' expects IntegerAttr");
  if (!llvm::isa<IntegerAttr>((*this)->getAttr("y_dilation_rate")))
    return (*this)->emitOpError("attribute 'y_dilation_rate' expects IntegerAttr");
  if (!llvm::isa<IntegerAttr>((*this)->getAttr("y_stride")))
    return (*this)->emitOpError("attribute 'y_stride' expects IntegerAttr");
  if (llvm::cast<CellOperationAttr>((*this)->getAttr("cell_operation")).getValue() != CellOperation::Mac)
    return (*this)->emitOpError("attribute 'cell_operation' expects MAC");
  for (const char *name : {"x_dilation_rate", "x_stride", "y_dilation_rate", "y_stride"}) {
    if (llvm::cast<IntegerAttr>((*this)->getAttr(name)).getInt() != 1)
      return (*this)->emitOpError("attribute '") << name << "' expects 1";
  }
  return success();
}

LogicalResult dwc::CosOp::verify() {
  if (getInputs().size() != 1)
    return emitOpError("expects 1 operands, got ") << getInputs().size();
  auto tensor = llvm::dyn_cast<TensorType>(getInputs()[0].getType());
  Type element = tensor ? tensor.getElementType() : getInputs()[0].getType();
  if (llvm::isa<Float16Type, BFloat16Type, Float32Type>(element))
    return success();
  if (auto integer = llvm::dyn_cast<IntegerType>(element)) {
    if (integer.isSignless() && (integer.getWidth() == 1 || integer.getWidth() == 8 || integer.getWidth() == 16))
      return success();
  }
  return (*this)->emitOpError("operand 0 expects i1, i8, i16, f16, bf16, or f32");
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
  if (!llvm::isa<IntegerAttr>((*this)->getAttr("axis")))
    return (*this)->emitOpError("attribute 'axis' expects IntegerAttr");
  if (!llvm::isa<BoolAttr>((*this)->getAttr("exclusive")))
    return (*this)->emitOpError("attribute 'exclusive' expects BoolAttr");
  if (!llvm::isa<CumulativeOpTypeAttr>((*this)->getAttr("op_type")))
    return (*this)->emitOpError("attribute 'op_type' expects CumulativeOpTypeAttr");
  if (llvm::cast<CumulativeOpTypeAttr>((*this)->getAttr("op_type")).getValue() != CumulativeOpType::Sum)
    return (*this)->emitOpError("attribute 'op_type' expects SUM");
  auto tensor = llvm::dyn_cast<TensorType>(getInputs()[0].getType());
  Type element = tensor ? tensor.getElementType() : getInputs()[0].getType();
  if (auto integer = llvm::dyn_cast<IntegerType>(element)) {
    if ((integer.isSignless() || integer.isUnsigned()) && (integer.getWidth() == 32 || integer.getWidth() == 64))
      return success();
  }
  return (*this)->emitOpError("operand 0 expects i32, i64, u32, or u64");
}

LogicalResult dwc::CwiseOp::verify() {
  if (getInputs().size() != 2)
    return emitOpError("expects 2 operands, got ") << getInputs().size();
  if (!(*this)->hasAttr("activation_function"))
    return (*this)->emitOpError("expected op 'dwc.cwise' to have attribute 'activation_function'");
  if (!(*this)->hasAttr("op_type"))
    return (*this)->emitOpError("expected op 'dwc.cwise' to have attribute 'op_type'");
  if (!llvm::isa<ActivationFunctionAttr>((*this)->getAttr("activation_function")))
    return (*this)->emitOpError("attribute 'activation_function' expects ActivationFunctionAttr");
  if (!llvm::isa<CwiseOpTypeAttr>((*this)->getAttr("op_type")))
    return (*this)->emitOpError("attribute 'op_type' expects CwiseOpTypeAttr");
  switch (llvm::cast<ActivationFunctionAttr>((*this)->getAttr("activation_function")).getValue()) {
  case ActivationFunction::None:
  case ActivationFunction::Relu:
  case ActivationFunction::Tanh:
  case ActivationFunction::ReciprocalSqrt:
    break;
  default:
    return (*this)->emitOpError("attribute 'activation_function' expects NONE, RELU, TANH, or RECIPROCAL_SQRT");
  }
  for (unsigned index = 0; index < 2; ++index) {
    if (auto ranked = llvm::dyn_cast<RankedTensorType>(getInputs()[index].getType())) {
      if (ranked.getRank() == 0)
        return (*this)->emitOpError("operand ") << index << " expects non-0-ranked tensor";
      continue;
    }
    return (*this)->emitOpError("operand ") << index << " expects ranked tensor";
  }
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
  if (!llvm::isa<ActivationFunctionAttr>((*this)->getAttr("activation_function")))
    return (*this)->emitOpError("attribute 'activation_function' expects ActivationFunctionAttr");
  if (!llvm::isa<CellOperationAttr>((*this)->getAttr("cell_operation")))
    return (*this)->emitOpError("attribute 'cell_operation' expects CellOperationAttr");
  if (!llvm::isa<PaddingAttr>((*this)->getAttr("pad")))
    return (*this)->emitOpError("attribute 'pad' expects PaddingAttr");
  if (!llvm::isa<IntegerAttr>((*this)->getAttr("depth_multiplier")))
    return (*this)->emitOpError("attribute 'depth_multiplier' expects IntegerAttr");
  if (!llvm::isa<IntegerAttr>((*this)->getAttr("x_dilation_rate")))
    return (*this)->emitOpError("attribute 'x_dilation_rate' expects IntegerAttr");
  if (!llvm::isa<IntegerAttr>((*this)->getAttr("x_stride")))
    return (*this)->emitOpError("attribute 'x_stride' expects IntegerAttr");
  if (!llvm::isa<IntegerAttr>((*this)->getAttr("y_dilation_rate")))
    return (*this)->emitOpError("attribute 'y_dilation_rate' expects IntegerAttr");
  if (!llvm::isa<IntegerAttr>((*this)->getAttr("y_stride")))
    return (*this)->emitOpError("attribute 'y_stride' expects IntegerAttr");
  return success();
}

LogicalResult dwc::DivideOp::verify() {
  if (!(*this)->hasAttr("activation_function"))
    return (*this)->emitOpError("expected op 'dwc.divide' to have attribute 'activation_function'");
  if (!llvm::isa<ActivationFunctionAttr>((*this)->getAttr("activation_function")))
    return (*this)->emitOpError("attribute 'activation_function' expects ActivationFunctionAttr");
  if (llvm::cast<ActivationFunctionAttr>((*this)->getAttr("activation_function")).getValue() != ActivationFunction::Relu)
    return (*this)->emitOpError("attribute 'activation_function' expects RELU");
  return success();
}

LogicalResult dwc::DynamicSliceOp::verify() {
  if (!(*this)->hasAttr("mode"))
    return (*this)->emitOpError("expected op 'dwc.dynamic_slice' to have attribute 'mode'");
  if (!(*this)->hasAttr("read_location"))
    return (*this)->emitOpError("expected op 'dwc.dynamic_slice' to have attribute 'read_location'");
  if (!(*this)->hasAttr("slice_size"))
    return (*this)->emitOpError("expected op 'dwc.dynamic_slice' to have attribute 'slice_size'");
  if (!llvm::isa<IntegerAttr>((*this)->getAttr("mode")))
    return (*this)->emitOpError("attribute 'mode' expects IntegerAttr");
  if (!llvm::isa<IntegerAttr>((*this)->getAttr("slice_size")))
    return (*this)->emitOpError("attribute 'slice_size' expects IntegerAttr");
  if (!llvm::isa<MemoryLocationAttr>((*this)->getAttr("read_location")))
    return (*this)->emitOpError("attribute 'read_location' expects MemoryLocationAttr");
  return success();
}

LogicalResult dwc::DynamicUpdateSliceOp::verify() {
  if (!(*this)->hasAttr("mode"))
    return (*this)->emitOpError("expected op 'dwc.dynamic_update_slice' to have attribute 'mode'");
  if (!(*this)->hasAttr("write_location"))
    return (*this)->emitOpError("expected op 'dwc.dynamic_update_slice' to have attribute 'write_location'");
  if (!llvm::isa<IntegerAttr>((*this)->getAttr("mode")))
    return (*this)->emitOpError("attribute 'mode' expects IntegerAttr");
  if (!llvm::isa<MemoryLocationAttr>((*this)->getAttr("write_location")))
    return (*this)->emitOpError("attribute 'write_location' expects MemoryLocationAttr");
  return success();
}

LogicalResult dwc::ErfOp::verify() {
  if (getInputs().size() != 1)
    return emitOpError("expects 1 operands, got ") << getInputs().size();
  auto tensor = llvm::dyn_cast<TensorType>(getInputs()[0].getType());
  Type element = tensor ? tensor.getElementType() : getInputs()[0].getType();
  if (llvm::isa<Float16Type, BFloat16Type, Float32Type>(element))
    return success();
  if (auto integer = llvm::dyn_cast<IntegerType>(element)) {
    if (integer.isSignless() && (integer.getWidth() == 1 || integer.getWidth() == 8 || integer.getWidth() == 16))
      return success();
  }
  return (*this)->emitOpError("operand 0 expects i1, i8, i16, f16, bf16, or f32");
}


LogicalResult dwc::ExpOp::verify() {
  if (getInputs().size() != 1)
    return emitOpError("expects 1 operands, got ") << getInputs().size();
  auto tensor = llvm::dyn_cast<TensorType>(getInputs()[0].getType());
  Type element = tensor ? tensor.getElementType() : getInputs()[0].getType();
  auto floatElement = llvm::dyn_cast<Float32Type>(element);
  if (!floatElement)
    return (*this)->emitOpError("operand 0 expects 32-bit float");
  return success();
}

LogicalResult dwc::Expm1Op::verify() {
  if (getInputs().size() != 1)
    return emitOpError("expects 1 operands, got ") << getInputs().size();
  auto tensor = llvm::dyn_cast<TensorType>(getInputs()[0].getType());
  Type element = tensor ? tensor.getElementType() : getInputs()[0].getType();
  if (llvm::isa<Float16Type, BFloat16Type, Float32Type>(element))
    return success();
  if (auto integer = llvm::dyn_cast<IntegerType>(element)) {
    if (integer.isSignless() && (integer.getWidth() == 1 || integer.getWidth() == 8 || integer.getWidth() == 16))
      return success();
  }
  return (*this)->emitOpError("operand 0 expects i1, i8, i16, f16, bf16, or f32");
}


LogicalResult dwc::FloorOp::verify() {
  if (getInputs().size() != 1)
    return emitOpError("expects 1 operands, got ") << getInputs().size();
  auto tensor = llvm::dyn_cast<TensorType>(getInputs()[0].getType());
  Type element = tensor ? tensor.getElementType() : getInputs()[0].getType();
  if (llvm::isa<Float16Type, BFloat16Type, Float32Type>(element))
    return success();
  if (auto integer = llvm::dyn_cast<IntegerType>(element)) {
    if (integer.isSignless() && (integer.getWidth() == 1 || integer.getWidth() == 8 || integer.getWidth() == 16 || integer.getWidth() == 32))
      return success();
  }
  return (*this)->emitOpError("operand 0 expects i1, i8, i16, i32, f16, bf16, or f32");
}

LogicalResult dwc::FloorDivOp::verify() {
  if (getInputs().size() != 2)
    return emitOpError("expects 2 operands, got ") << getInputs().size();
  for (unsigned index = 0; index < 2; ++index) {
    auto tensor = llvm::dyn_cast<TensorType>(getInputs()[index].getType());
    Type element = tensor ? tensor.getElementType() : getInputs()[index].getType();
    if (llvm::isa<Float32Type, BFloat16Type, Float16Type>(element))
      continue;
    return (*this)->emitOpError("operand ") << index << " expects 32-bit float, bfloat16, or 16-bit float";
  }
  return success();
}

LogicalResult dwc::FullyConnectedOp::verify() {
  if (!(*this)->hasAttr("activation_function"))
    return (*this)->emitOpError("expected op 'dwc.fully_connected' to have attribute 'activation_function'");
  if (!(*this)->hasAttr("cell_operation"))
    return (*this)->emitOpError("expected op 'dwc.fully_connected' to have attribute 'cell_operation'");
  if (!llvm::isa<ActivationFunctionAttr>((*this)->getAttr("activation_function")))
    return (*this)->emitOpError("attribute 'activation_function' expects ActivationFunctionAttr");
  if (!llvm::isa<CellOperationAttr>((*this)->getAttr("cell_operation")))
    return (*this)->emitOpError("attribute 'cell_operation' expects CellOperationAttr");
  if (llvm::cast<ActivationFunctionAttr>((*this)->getAttr("activation_function")).getValue() != ActivationFunction::None)
    return (*this)->emitOpError("attribute 'activation_function' expects NONE");
  if (llvm::cast<CellOperationAttr>((*this)->getAttr("cell_operation")).getValue() != CellOperation::Mac)
    return (*this)->emitOpError("attribute 'cell_operation' expects MAC");
  return success();
}

LogicalResult dwc::GatherOp::verify() {
  if (!(*this)->hasAttr("axis"))
    return (*this)->emitOpError("expected op 'dwc.gather' to have attribute 'axis'");
  if (!(*this)->hasAttr("batch_dims"))
    return (*this)->emitOpError("expected op 'dwc.gather' to have attribute 'batch_dims'");
  if (!llvm::isa<IntegerAttr>((*this)->getAttr("axis")))
    return (*this)->emitOpError("attribute 'axis' expects IntegerAttr");
  if (!llvm::isa<IntegerAttr>((*this)->getAttr("batch_dims")))
    return (*this)->emitOpError("attribute 'batch_dims' expects IntegerAttr");
  return success();
}

LogicalResult dwc::GenericComputeOp::verify() {
  if (!(*this)->hasAttr("activation_function"))
    return (*this)->emitOpError("expected op 'dwc.generic_compute' to have attribute 'activation_function'");
  if (!(*this)->hasAttr("indexing_maps"))
    return (*this)->emitOpError("expected op 'dwc.generic_compute' to have attribute 'indexing_maps'");
  if (!(*this)->hasAttr("linear_function"))
    return (*this)->emitOpError("expected op 'dwc.generic_compute' to have attribute 'linear_function'");
  if (!llvm::isa<ArrayAttr>((*this)->getAttr("indexing_maps")))
    return (*this)->emitOpError("attribute 'indexing_maps' expects ArrayAttr");
  if (!llvm::isa<ActivationFunctionAttr>((*this)->getAttr("activation_function")))
    return (*this)->emitOpError("attribute 'activation_function' expects ActivationFunctionAttr");
  if (!llvm::isa<LinearFunctionTypeAttr>((*this)->getAttr("linear_function")))
    return (*this)->emitOpError("attribute 'linear_function' expects LinearFunctionTypeAttr");
  if (llvm::cast<ActivationFunctionAttr>((*this)->getAttr("activation_function")).getValue() != ActivationFunction::None)
    return (*this)->emitOpError("attribute 'activation_function' expects NONE");
  return success();
}

LogicalResult dwc::GenericConstantOp::verify() {
  if (!(*this)->hasAttr("value"))
    return (*this)->emitOpError("expected op 'dwc.generic_constant' to have attribute 'value'");
  if (!llvm::isa<ElementsAttr>((*this)->getAttr("value")))
    return (*this)->emitOpError("attribute 'value' expects ElementsAttr");
  auto shaped = llvm::dyn_cast<ShapedType>(llvm::cast<ElementsAttr>((*this)->getAttr("value")).getType());
  if (!shaped || !shaped.hasRank() || shaped.getRank() < 1)
    return (*this)->emitOpError("attribute 'value' expects constant vector/tensor");
  return success();
}

LogicalResult dwc::GenericConvOp::verify() {
  if (!(*this)->hasAttr("activation_function"))
    return (*this)->emitOpError("expected op 'dwc.generic_conv' to have attribute 'activation_function'");
  if (!llvm::isa<ActivationFunctionAttr>((*this)->getAttr("activation_function")))
    return (*this)->emitOpError("attribute 'activation_function' expects ActivationFunctionAttr");
  if (llvm::cast<ActivationFunctionAttr>((*this)->getAttr("activation_function")).getValue() != ActivationFunction::None)
    return (*this)->emitOpError("attribute 'activation_function' expects NONE");
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
  if (!llvm::isa<IntegerAttr>((*this)->getAttr("batch_group_count")))
    return (*this)->emitOpError("attribute 'batch_group_count' expects IntegerAttr");
  if (!llvm::isa<IntegerAttr>((*this)->getAttr("feature_group_count")))
    return (*this)->emitOpError("attribute 'feature_group_count' expects IntegerAttr");
  if (!llvm::isa<DenseIntElementsAttr>((*this)->getAttr("input_dilation")))
    return (*this)->emitOpError("attribute 'input_dilation' expects DenseIntElementsAttr");
  if (!llvm::isa<DenseIntElementsAttr>((*this)->getAttr("padding_amount")))
    return (*this)->emitOpError("attribute 'padding_amount' expects DenseIntElementsAttr");
  if (!llvm::isa<DenseIntElementsAttr>((*this)->getAttr("param_dilation")))
    return (*this)->emitOpError("attribute 'param_dilation' expects DenseIntElementsAttr");
  if (!llvm::isa<ElementsAttr>((*this)->getAttr("param_reversal")))
    return (*this)->emitOpError("attribute 'param_reversal' expects ElementsAttr");
  if (!llvm::isa<DenseIntElementsAttr>((*this)->getAttr("stride")))
    return (*this)->emitOpError("attribute 'stride' expects DenseIntElementsAttr");
  return success();
}

LogicalResult dwc::GenericDotOp::verify() {
  if (!(*this)->hasAttr("activation_function"))
    return (*this)->emitOpError("expected op 'dwc.generic_dot' to have attribute 'activation_function'");
  if (!(*this)->hasAttr("batch_dim_count"))
    return (*this)->emitOpError("expected op 'dwc.generic_dot' to have attribute 'batch_dim_count'");
  if (!(*this)->hasAttr("contracting_dim_count"))
    return (*this)->emitOpError("expected op 'dwc.generic_dot' to have attribute 'contracting_dim_count'");
  if (!llvm::isa<IntegerAttr>((*this)->getAttr("batch_dim_count")))
    return (*this)->emitOpError("attribute 'batch_dim_count' expects IntegerAttr");
  if (!llvm::isa<IntegerAttr>((*this)->getAttr("contracting_dim_count")))
    return (*this)->emitOpError("attribute 'contracting_dim_count' expects IntegerAttr");
  if (!llvm::isa<ActivationFunctionAttr>((*this)->getAttr("activation_function")))
    return (*this)->emitOpError("attribute 'activation_function' expects ActivationFunctionAttr");
  if (llvm::cast<ActivationFunctionAttr>((*this)->getAttr("activation_function")).getValue() != ActivationFunction::None)
    return (*this)->emitOpError("attribute 'activation_function' expects NONE");
  if (llvm::cast<IntegerAttr>((*this)->getAttr("batch_dim_count")).getInt() != 1)
    return (*this)->emitOpError("attribute 'batch_dim_count' expects 1");
  return success();
}

LogicalResult dwc::Log1pOp::verify() {
  if (getInputs().size() != 1)
    return emitOpError("expects 1 operands, got ") << getInputs().size();
  auto tensor = llvm::dyn_cast<TensorType>(getInputs()[0].getType());
  Type element = tensor ? tensor.getElementType() : getInputs()[0].getType();
  if (llvm::isa<Float16Type, BFloat16Type, Float32Type>(element))
    return success();
  if (auto integer = llvm::dyn_cast<IntegerType>(element)) {
    if (integer.isSignless() && (integer.getWidth() == 1 || integer.getWidth() == 8 || integer.getWidth() == 16))
      return success();
  }
  return (*this)->emitOpError("operand 0 expects i1, i8, i16, f16, bf16, or f32");
}


LogicalResult dwc::LogisticOp::verify() {
  if (getInputs().size() != 1)
    return emitOpError("expects 1 operands, got ") << getInputs().size();
  auto tensor = llvm::dyn_cast<TensorType>(getInputs()[0].getType());
  Type element = tensor ? tensor.getElementType() : getInputs()[0].getType();
  if (llvm::isa<Float16Type, BFloat16Type, Float32Type>(element))
    return success();
  if (auto integer = llvm::dyn_cast<IntegerType>(element)) {
    if (integer.isSignless() && (integer.getWidth() == 1 || integer.getWidth() == 8 || integer.getWidth() == 16))
      return success();
  }
  return (*this)->emitOpError("operand 0 expects i1, i8, i16, f16, bf16, or f32");
}


LogicalResult dwc::MatrixMultiplyOp::verify() {
  if (!(*this)->hasAttr("activation_function"))
    return (*this)->emitOpError("expected op 'dwc.matrix_multiply' to have attribute 'activation_function'");
  if (!(*this)->hasAttr("transpose_rhs"))
    return (*this)->emitOpError("expected op 'dwc.matrix_multiply' to have attribute 'transpose_rhs'");
  if (!llvm::isa<BoolAttr>((*this)->getAttr("transpose_rhs")))
    return (*this)->emitOpError("attribute 'transpose_rhs' expects BoolAttr");
  if (!llvm::isa<ActivationFunctionAttr>((*this)->getAttr("activation_function")))
    return (*this)->emitOpError("attribute 'activation_function' expects ActivationFunctionAttr");
  if (llvm::cast<ActivationFunctionAttr>((*this)->getAttr("activation_function")).getValue() != ActivationFunction::None)
    return (*this)->emitOpError("attribute 'activation_function' expects NONE");
  return success();
}

LogicalResult dwc::MaximumOp::verify() {
  if (!(*this)->hasAttr("activation_function"))
    return (*this)->emitOpError("expected op 'dwc.maximum' to have attribute 'activation_function'");
  if (!llvm::isa<ActivationFunctionAttr>((*this)->getAttr("activation_function")))
    return (*this)->emitOpError("attribute 'activation_function' expects ActivationFunctionAttr");
  return success();
}

LogicalResult dwc::MinimumOp::verify() {
  if (!(*this)->hasAttr("activation_function"))
    return (*this)->emitOpError("expected op 'dwc.minimum' to have attribute 'activation_function'");
  if (!llvm::isa<ActivationFunctionAttr>((*this)->getAttr("activation_function")))
    return (*this)->emitOpError("attribute 'activation_function' expects ActivationFunctionAttr");
  return success();
}

LogicalResult dwc::MultiplyOp::verify() {
  if (!(*this)->hasAttr("activation_function"))
    return (*this)->emitOpError("expected op 'dwc.multiply' to have attribute 'activation_function'");
  if (!llvm::isa<ActivationFunctionAttr>((*this)->getAttr("activation_function")))
    return (*this)->emitOpError("attribute 'activation_function' expects ActivationFunctionAttr");
  auto activation = llvm::cast<ActivationFunctionAttr>((*this)->getAttr("activation_function")).getValue();
  if (activation != ActivationFunction::None && activation != ActivationFunction::Relu)
    return (*this)->emitOpError("attribute 'activation_function' expects NONE or RELU");
  return success();
}

LogicalResult dwc::NotOp::verify() {
  if (getInputs().size() != 1)
    return emitOpError("expects 1 operands, got ") << getInputs().size();
  auto tensor = llvm::dyn_cast<TensorType>(getInputs()[0].getType());
  Type element = tensor ? tensor.getElementType() : getInputs()[0].getType();
  auto integer = llvm::dyn_cast<IntegerType>(element);
  if (!integer || !integer.isSignless() || integer.getWidth() != 1)
    return (*this)->emitOpError("operand 0 expects 1-bit signless integer");
  return success();
}

LogicalResult dwc::OneHotOp::verify() {
  if (getInputs().size() != 1)
    return emitOpError("expects 1 operands, got ") << getInputs().size();
  if (!(*this)->hasAttr("axis"))
    return (*this)->emitOpError("expected op 'dwc.one_hot' to have attribute 'axis'");
  if (!llvm::isa<IntegerAttr>((*this)->getAttr("axis")))
    return (*this)->emitOpError("attribute 'axis' expects IntegerAttr");
  auto tensor = llvm::dyn_cast<TensorType>(getInputs()[0].getType());
  Type element = tensor ? tensor.getElementType() : getInputs()[0].getType();
  if (auto integer = llvm::dyn_cast<IntegerType>(element)) {
    if (integer.isSignless() && (integer.getWidth() == 16 || integer.getWidth() == 32))
      return success();
    if (integer.isSignless() && integer.getWidth() == 64)
      return success();
    if (integer.isUnsigned() && integer.getWidth() == 16)
      return success();
  }
  return (*this)->emitOpError("operand 0 expects i16, i32, i64, or u16");
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
  if (!llvm::isa<IntegerAttr>((*this)->getAttr("dimension")))
    return (*this)->emitOpError("attribute 'dimension' expects IntegerAttr");
  if (!llvm::cast<IntegerAttr>((*this)->getAttr("dimension")).getType().isSignlessInteger(32))
    return (*this)->emitOpError("attribute 'dimension' expects I32");
  if (!llvm::isa<IntegerAttr>((*this)->getAttr("post_padding")))
    return (*this)->emitOpError("attribute 'post_padding' expects IntegerAttr");
  if (!llvm::isa<IntegerAttr>((*this)->getAttr("pre_padding")))
    return (*this)->emitOpError("attribute 'pre_padding' expects IntegerAttr");
  if (auto floatValue = llvm::dyn_cast<FloatAttr>((*this)->getAttr("padding_value"))) {
    if (!floatValue.getType().isF32())
      return (*this)->emitOpError("attribute 'padding_value' expects 32-bit float");
  } else if (auto dense = llvm::dyn_cast<DenseElementsAttr>((*this)->getAttr("padding_value"))) {
    auto shaped = llvm::dyn_cast<ShapedType>(dense.getType());
    if (!shaped || !shaped.hasStaticShape() || shaped.getNumElements() != 2)
      return (*this)->emitOpError("attribute 'padding_value' expects 32-bit float or 2-element on-off pair");
  }
  for (const char *name : {"post_padding", "pre_padding"}) {
    if (!llvm::cast<IntegerAttr>((*this)->getAttr(name)).getType().isSignlessInteger(32))
      return (*this)->emitOpError("attribute '") << name << "' expects 32-bit signless integer";
  }
  if (llvm::cast<IntegerAttr>((*this)->getAttr("post_padding")).getInt() < 0)
    return (*this)->emitOpError("attribute 'post_padding' expects non-negative");
  if (llvm::cast<IntegerAttr>((*this)->getAttr("pre_padding")).getInt() < 0)
    return (*this)->emitOpError("attribute 'pre_padding' expects non-negative");
  if (auto ranked = llvm::dyn_cast<RankedTensorType>(getInputs()[0].getType())) {
    if (ranked.getRank() == 0)
      return (*this)->emitOpError("operand 0 expects non-0-ranked tensor");
    return success();
  }
  return (*this)->emitOpError("operand 0 expects ranked tensor");
}

LogicalResult dwc::PopCountOp::verify() {
  if (getInputs().size() != 1)
    return emitOpError("expects 1 operands, got ") << getInputs().size();
  auto tensor = llvm::dyn_cast<TensorType>(getInputs()[0].getType());
  Type element = tensor ? tensor.getElementType() : getInputs()[0].getType();
  if (auto integer = llvm::dyn_cast<IntegerType>(element)) {
    if (integer.isSignless() && integer.getWidth() == 1)
      return success();
    if (integer.isSignless() && (integer.getWidth() == 8 || integer.getWidth() == 16))
      return success();
    if (integer.isSignless() && integer.getWidth() == 32)
      return success();
    if (integer.isUnsigned() && (integer.getWidth() == 8 || integer.getWidth() == 16))
      return success();
  }
  return (*this)->emitOpError("operand 0 expects i1, i8, i16, i32, u8, or u16");
}

LogicalResult dwc::PowOp::verify() {
  if (getInputs().size() != 2)
    return emitOpError("expects 2 operands, got ") << getInputs().size();
  for (unsigned index = 0; index < 2; ++index) {
    auto tensor = llvm::dyn_cast<TensorType>(getInputs()[index].getType());
    Type element = tensor ? tensor.getElementType() : getInputs()[index].getType();
    if (llvm::isa<Float16Type, BFloat16Type, Float32Type>(element))
      continue;
    if (auto integer = llvm::dyn_cast<IntegerType>(element)) {
      if (integer.isSignless() && integer.getWidth() == 32)
        continue;
      if (integer.isUnsigned() && integer.getWidth() == 32)
        continue;
    }
    return (*this)->emitOpError("operand ") << index << " expects bfloat16, f16, f32, i32, or u32";
  }
  return success();
}

LogicalResult dwc::PseudoSplitOp::verify() {
  if (getInputs().size() != 2)
    return emitOpError("expects 2 operands, got ") << getInputs().size();
  if (!(*this)->hasAttr("num_splits"))
    return (*this)->emitOpError("expected op 'dwc.pseudo_split' to have attribute 'num_splits'");
  if (!llvm::isa<IntegerAttr>((*this)->getAttr("num_splits")))
    return (*this)->emitOpError("attribute 'num_splits' expects IntegerAttr");
  if (llvm::cast<IntegerAttr>((*this)->getAttr("num_splits")).getInt() != 1)
    return (*this)->emitOpError("attribute 'num_splits' expects 1");
  if (auto ranked = llvm::dyn_cast<RankedTensorType>(getInputs()[0].getType())) {
    if (ranked.getRank() != 0)
      return (*this)->emitOpError("operand 0 expects 0D tensor");
    if (auto integer = llvm::dyn_cast<IntegerType>(ranked.getElementType())) {
      if (integer.isSignless() && integer.getWidth() == 32)
        return success();
    }
    return (*this)->emitOpError("operand 0 expects 0D tensor of 32-bit signless integer");
  }
  auto tensor1 = llvm::dyn_cast<TensorType>(getInputs()[1].getType());
  Type element1 = tensor1 ? tensor1.getElementType() : getInputs()[1].getType();
  if (llvm::isa<Float32Type>(element1))
    return success();
  if (auto integer1 = llvm::dyn_cast<IntegerType>(element1)) {
    if ((integer1.isSignless() && (integer1.getWidth() == 16 || integer1.getWidth() == 32 || integer1.getWidth() == 64)))
      return success();
  }
  return (*this)->emitOpError("operand 1 expects f32, i16, i32, i64, QI8, or QUI8");
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
  if (!llvm::isa<ElementsAttr>((*this)->getAttr("dimensions")))
    return (*this)->emitOpError("attribute 'dimensions' expects ElementsAttr");
  if (!llvm::isa<SimpleActivationFunctionAttr>((*this)->getAttr("activation_function")))
    return (*this)->emitOpError("attribute 'activation_function' expects SimpleActivationFunctionAttr");
  if (!llvm::isa<ReductionTypeAttr>((*this)->getAttr("op_type")))
    return (*this)->emitOpError("attribute 'op_type' expects ReductionTypeAttr");
  if (llvm::cast<SimpleActivationFunctionAttr>((*this)->getAttr("activation_function")).getValue() != SimpleActivationFunction::None)
    return (*this)->emitOpError("attribute 'activation_function' expects NONE");
  {
    auto dims = llvm::cast<ElementsAttr>((*this)->getAttr("dimensions"));
    auto shaped = llvm::dyn_cast<ShapedType>(dims.getType());
    if (!shaped || !shaped.hasRank() || shaped.getRank() != 1 || !shaped.getElementType().isSignlessInteger(32))
      return (*this)->emitOpError("attribute 'dimensions' expects 1D tensor of I32 elements");
    if (!shaped.hasStaticShape() || shaped.getNumElements() != 1)
      return (*this)->emitOpError("attribute 'dimensions' expects singleton tensor");
  }
  auto tensor = llvm::dyn_cast<TensorType>(getInputs()[0].getType());
  Type element = tensor ? tensor.getElementType() : getInputs()[0].getType();
  if (llvm::isa<Float16Type, BFloat16Type, Float32Type>(element))
    return success();
  return (*this)->emitOpError("operand 0 expects 16-bit float, bfloat16, or 32-bit float");
}

LogicalResult dwc::RemainderOp::verify() {
  if (!(*this)->hasAttr("activation_function"))
    return (*this)->emitOpError("expected op 'dwc.remainder' to have attribute 'activation_function'");
  if (!llvm::isa<ActivationFunctionAttr>((*this)->getAttr("activation_function")))
    return (*this)->emitOpError("attribute 'activation_function' expects ActivationFunctionAttr");
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
  if (!llvm::isa<ArrayAttr>((*this)->getAttr("output_activation_per_z_out_scales")))
    return (*this)->emitOpError("attribute 'output_activation_per_z_out_scales' expects ArrayAttr");
  if (!llvm::isa<ActivationFunctionAttr>((*this)->getAttr("activation_function")))
    return (*this)->emitOpError("attribute 'activation_function' expects ActivationFunctionAttr");
  if (!llvm::isa<PerZOutScalePaddingAttr>((*this)->getAttr("per_z_out_scales_padding")))
    return (*this)->emitOpError("attribute 'per_z_out_scales_padding' expects PerZOutScalePaddingAttr");
  if (llvm::cast<PerZOutScalePaddingAttr>((*this)->getAttr("per_z_out_scales_padding")).getValue() != PerZOutScalePadding::None)
    return (*this)->emitOpError("attribute 'per_z_out_scales_padding' expects NONE");
  if (!llvm::cast<ArrayAttr>((*this)->getAttr("output_activation_per_z_out_scales")).empty())
    return (*this)->emitOpError("attribute 'output_activation_per_z_out_scales' expects empty array");
  switch (llvm::cast<ActivationFunctionAttr>((*this)->getAttr("activation_function")).getValue()) {
  case ActivationFunction::None:
  case ActivationFunction::Exp:
  case ActivationFunction::Logistic:
  case ActivationFunction::Tanh:
  case ActivationFunction::ReciprocalSqrt:
  case ActivationFunction::GeluApproximated:
    break;
  default:
    return (*this)->emitOpError("attribute 'activation_function' expects NONE, EXP, LOGISTIC, TANH, RECIPROCAL_SQRT, or GELU_APPROXIMATED");
  }
  auto tensor = llvm::dyn_cast<TensorType>(getInputs()[0].getType());
  Type element = tensor ? tensor.getElementType() : getInputs()[0].getType();
  if (llvm::isa<Float16Type, BFloat16Type>(element))
    return success();
  return (*this)->emitOpError("operand 0 expects 16-bit float or bfloat16");
}

LogicalResult dwc::ReshapeOp::verify() {
  if (getInputs().size() != 1)
    return emitOpError("expects 1 operands, got ") << getInputs().size();
  if (auto ranked = llvm::dyn_cast<RankedTensorType>(getInputs()[0].getType())) {
    if (ranked.getRank() != 3 && ranked.getRank() != 4)
      return (*this)->emitOpError("operand 0 expects Rank 3 or Rank 4 tensor");
  }
  return success();
}

LogicalResult dwc::RoundOp::verify() {
  if (getInputs().size() != 1)
    return emitOpError("expects 1 operands, got ") << getInputs().size();
  auto tensor = llvm::dyn_cast<TensorType>(getInputs()[0].getType());
  Type element = tensor ? tensor.getElementType() : getInputs()[0].getType();
  if (llvm::isa<Float16Type, BFloat16Type, Float32Type>(element))
    return success();
  if (auto integer = llvm::dyn_cast<IntegerType>(element)) {
    if (integer.isSignless() && (integer.getWidth() == 1 || integer.getWidth() == 8 || integer.getWidth() == 16))
      return success();
  }
  return (*this)->emitOpError("operand 0 expects i1, i8, i16, f16, bf16, or f32");
}


LogicalResult dwc::RoundNearestAfzOp::verify() {
  if (getInputs().size() != 1)
    return emitOpError("expects 1 operands, got ") << getInputs().size();
  auto tensor = llvm::dyn_cast<TensorType>(getInputs()[0].getType());
  Type element = tensor ? tensor.getElementType() : getInputs()[0].getType();
  if (llvm::isa<Float16Type, BFloat16Type, Float32Type>(element))
    return success();
  if (auto integer = llvm::dyn_cast<IntegerType>(element)) {
    if (integer.isSignless() && (integer.getWidth() == 1 || integer.getWidth() == 8 || integer.getWidth() == 16))
      return success();
  }
  return (*this)->emitOpError("operand 0 expects i1, i8, i16, f16, bf16, or f32");
}


LogicalResult dwc::RsqrtOp::verify() {
  if (getInputs().size() != 1)
    return emitOpError("expects 1 operands, got ") << getInputs().size();
  auto tensor = llvm::dyn_cast<TensorType>(getInputs()[0].getType());
  Type element = tensor ? tensor.getElementType() : getInputs()[0].getType();
  if (llvm::isa<Float16Type, BFloat16Type, Float32Type>(element))
    return success();
  if (auto integer = llvm::dyn_cast<IntegerType>(element)) {
    if (integer.isSignless() && (integer.getWidth() == 1 || integer.getWidth() == 8 || integer.getWidth() == 16))
      return success();
  }
  return (*this)->emitOpError("operand 0 expects i1, i8, i16, f16, bf16, or f32");
}


LogicalResult dwc::ScalarOp::verify() {
  if (!(*this)->hasAttr("op_type"))
    return (*this)->emitOpError("expected op 'dwc.scalar' to have attribute 'op_type'");
  if (!llvm::isa<ScalarOpTypeAttr>((*this)->getAttr("op_type")))
    return (*this)->emitOpError("attribute 'op_type' expects ScalarOpTypeAttr");
  if ((*this)->hasAttr("immediate")) {
    auto immediate = llvm::dyn_cast<IntegerAttr>((*this)->getAttr("immediate"));
    if (!immediate || !immediate.getType().isSignlessInteger(32))
      return (*this)->emitOpError("attribute 'immediate' expects 32-bit signless integer");
  }
  return success();
}

LogicalResult dwc::ScatterNdOp::verify() {
  if (!(*this)->hasAttr("shape"))
    return (*this)->emitOpError("expected op 'dwc.scatter_nd' to have attribute 'shape'");
  if (!llvm::isa<ElementsAttr>((*this)->getAttr("shape")))
    return (*this)->emitOpError("attribute 'shape' expects ElementsAttr");
  return success();
}

LogicalResult dwc::SelectOp::verify() {
  if (getInputs().size() != 2)
    return emitOpError("expects 2 operands, got ") << getInputs().size();
  auto tensor0 = llvm::dyn_cast<TensorType>(getInputs()[0].getType());
  Type element0 = tensor0 ? tensor0.getElementType() : getInputs()[0].getType();
  auto integer0 = llvm::dyn_cast<IntegerType>(element0);
  if (!integer0 || !integer0.isSignless() || integer0.getWidth() != 1)
    return (*this)->emitOpError("operand 0 expects 1-bit signless integer");
  if (!llvm::isa<RankedTensorType>(getInputs()[0].getType()))
    return (*this)->emitOpError("operand 0 expects statically shaped tensor");
  if (!llvm::cast<RankedTensorType>(getInputs()[0].getType()).hasStaticShape())
    return (*this)->emitOpError("operand 0 expects statically shaped tensor");
  auto tensor1 = llvm::dyn_cast<TensorType>(getInputs()[1].getType());
  Type element1 = tensor1 ? tensor1.getElementType() : getInputs()[1].getType();
  if (llvm::isa<Float32Type, Float64Type>(element1))
    return success();
  if (auto integer1 = llvm::dyn_cast<IntegerType>(element1)) {
    if ((integer1.isSignless() || integer1.isUnsigned()) && (integer1.getWidth() == 8 || integer1.getWidth() == 16 || integer1.getWidth() == 32 || integer1.getWidth() == 64))
      return success();
    if (integer1.isSignless() && integer1.getWidth() == 1)
      return success();
  }
  return (*this)->emitOpError("operand 1 expects 8/16/32/64-bit int or 32/64-bit float");
}

LogicalResult dwc::SignOp::verify() {
  if (getInputs().size() != 1)
    return emitOpError("expects 1 operands, got ") << getInputs().size();
  if (!(*this)->hasAttr("preserve_negative_zero"))
    return (*this)->emitOpError("expected op 'dwc.sign' to have attribute 'preserve_negative_zero'");
  if (!llvm::isa<BoolAttr>((*this)->getAttr("preserve_negative_zero")))
    return (*this)->emitOpError("attribute 'preserve_negative_zero' expects BoolAttr");
  auto tensor = llvm::dyn_cast<TensorType>(getInputs()[0].getType());
  Type element = tensor ? tensor.getElementType() : getInputs()[0].getType();
  if (llvm::isa<Float16Type, BFloat16Type, Float32Type>(element))
    return success();
  if (auto integer = llvm::dyn_cast<IntegerType>(element)) {
    if (integer.isSignless() && (integer.getWidth() == 1 || integer.getWidth() == 8 || integer.getWidth() == 16 || integer.getWidth() == 32 || integer.getWidth() == 64))
      return success();
  }
  return (*this)->emitOpError("operand 0 expects i1, i8, i16, i32, i64, f16, bf16, or f32");
}

LogicalResult dwc::SinOp::verify() {
  if (getInputs().size() != 1)
    return emitOpError("expects 1 operands, got ") << getInputs().size();
  auto tensor = llvm::dyn_cast<TensorType>(getInputs()[0].getType());
  Type element = tensor ? tensor.getElementType() : getInputs()[0].getType();
  if (llvm::isa<Float16Type, BFloat16Type, Float32Type>(element))
    return success();
  if (auto integer = llvm::dyn_cast<IntegerType>(element)) {
    if (integer.isSignless() && (integer.getWidth() == 1 || integer.getWidth() == 8 || integer.getWidth() == 16))
      return success();
  }
  return (*this)->emitOpError("operand 0 expects i1, i8, i16, f16, bf16, or f32");
}


LogicalResult dwc::SliceOp::verify() {
  if (!(*this)->hasAttr("in_begin"))
    return (*this)->emitOpError("expected op 'dwc.slice' to have attribute 'in_begin'");
  if (!(*this)->hasAttr("in_size"))
    return (*this)->emitOpError("expected op 'dwc.slice' to have attribute 'in_size'");
  if (!(*this)->hasAttr("mode"))
    return (*this)->emitOpError("expected op 'dwc.slice' to have attribute 'mode'");
  if (!llvm::isa<IntegerAttr>((*this)->getAttr("in_begin")))
    return (*this)->emitOpError("attribute 'in_begin' expects IntegerAttr");
  if (!llvm::isa<IntegerAttr>((*this)->getAttr("in_size")))
    return (*this)->emitOpError("attribute 'in_size' expects IntegerAttr");
  if (!llvm::isa<IntegerAttr>((*this)->getAttr("mode")))
    return (*this)->emitOpError("attribute 'mode' expects IntegerAttr");
  for (const char *name : {"in_begin", "in_size", "mode"}) {
    if (!llvm::cast<IntegerAttr>((*this)->getAttr(name)).getType().isSignlessInteger(32))
      return (*this)->emitOpError("attribute '") << name << "' expects 32-bit signless integer";
  }
  if (llvm::cast<IntegerAttr>((*this)->getAttr("in_begin")).getInt() < 0)
    return (*this)->emitOpError("attribute 'in_begin' expects non-negative");
  if (llvm::cast<IntegerAttr>((*this)->getAttr("mode")).getInt() < 0)
    return (*this)->emitOpError("attribute 'mode' expects non-negative");
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
  if (!llvm::isa<ArrayAttr>((*this)->getAttr("compare_tuple_projection")))
    return (*this)->emitOpError("attribute 'compare_tuple_projection' expects ArrayAttr");
  if (!llvm::isa<ArrayAttr>((*this)->getAttr("compare_type")))
    return (*this)->emitOpError("attribute 'compare_type' expects ArrayAttr");
  if (!llvm::isa<IntegerAttr>((*this)->getAttr("dimension")))
    return (*this)->emitOpError("attribute 'dimension' expects IntegerAttr");
  if (!llvm::isa<BoolAttr>((*this)->getAttr("is_stable")))
    return (*this)->emitOpError("attribute 'is_stable' expects BoolAttr");
  return success();
}

LogicalResult dwc::SqrtOp::verify() {
  if (getInputs().size() != 1)
    return emitOpError("expects 1 operands, got ") << getInputs().size();
  auto tensor = llvm::dyn_cast<TensorType>(getInputs()[0].getType());
  Type element = tensor ? tensor.getElementType() : getInputs()[0].getType();
  if (llvm::isa<Float16Type, BFloat16Type, Float32Type>(element))
    return success();
  if (auto integer = llvm::dyn_cast<IntegerType>(element)) {
    if (integer.isSignless() && (integer.getWidth() == 1 || integer.getWidth() == 8 || integer.getWidth() == 16))
      return success();
  }
  return (*this)->emitOpError("operand 0 expects i1, i8, i16, f16, bf16, or f32");
}

LogicalResult dwc::SubtractOp::verify() {
  if (!(*this)->hasAttr("activation_function"))
    return (*this)->emitOpError("expected op 'dwc.subtract' to have attribute 'activation_function'");
  if (!llvm::isa<ActivationFunctionAttr>((*this)->getAttr("activation_function")))
    return (*this)->emitOpError("attribute 'activation_function' expects ActivationFunctionAttr");
  auto activation = llvm::cast<ActivationFunctionAttr>((*this)->getAttr("activation_function")).getValue();
  if (activation != ActivationFunction::None && activation != ActivationFunction::Relu)
    return (*this)->emitOpError("attribute 'activation_function' expects NONE or RELU");
  return success();
}

LogicalResult dwc::TanOp::verify() {
  if (getInputs().size() != 1)
    return emitOpError("expects 1 operands, got ") << getInputs().size();
  auto tensor = llvm::dyn_cast<TensorType>(getInputs()[0].getType());
  Type element = tensor ? tensor.getElementType() : getInputs()[0].getType();
  if (llvm::isa<Float16Type, BFloat16Type, Float32Type>(element))
    return success();
  if (auto integer = llvm::dyn_cast<IntegerType>(element)) {
    if (integer.isSignless() && (integer.getWidth() == 1 || integer.getWidth() == 8 || integer.getWidth() == 16))
      return success();
  }
  return (*this)->emitOpError("operand 0 expects i1, i8, i16, f16, bf16, or f32");
}


LogicalResult dwc::TanhOp::verify() {
  if (getInputs().size() != 1)
    return emitOpError("expects 1 operands, got ") << getInputs().size();
  auto tensor = llvm::dyn_cast<TensorType>(getInputs()[0].getType());
  Type element = tensor ? tensor.getElementType() : getInputs()[0].getType();
  if (llvm::isa<Float16Type, BFloat16Type, Float32Type>(element))
    return success();
  if (auto integer = llvm::dyn_cast<IntegerType>(element)) {
    if (integer.isSignless() && (integer.getWidth() == 1 || integer.getWidth() == 8 || integer.getWidth() == 16))
      return success();
  }
  return (*this)->emitOpError("operand 0 expects i1, i8, i16, f16, bf16, or f32");
}


LogicalResult dwc::TransposeOp::verify() {
  if (getInputs().size() != 1)
    return emitOpError("expects 1 operands, got ") << getInputs().size();
  if (!(*this)->hasAttr("permutation"))
    return (*this)->emitOpError("expected op 'dwc.transpose' to have attribute 'permutation'");
  if (!llvm::isa<ElementsAttr>((*this)->getAttr("permutation")))
    return (*this)->emitOpError("attribute 'permutation' expects ElementsAttr");
  if (auto ranked = llvm::dyn_cast<RankedTensorType>(getInputs()[0].getType())) {
    if (ranked.getRank() != 3)
      return (*this)->emitOpError("operand 0 expects Rank 3 tensor");
    auto perm = llvm::cast<ElementsAttr>((*this)->getAttr("permutation"));
    auto shaped = llvm::dyn_cast<ShapedType>(perm.getType());
    if (!shaped || !shaped.hasStaticShape() || shaped.getNumElements() != 3)
      return (*this)->emitOpError("attribute 'permutation' expects 3 elements for Rank 3 tensor");
    SmallVector<int64_t> values;
    for (auto element : perm.getValues<IntegerAttr>())
      values.push_back(element.getInt());
    if (values.size() != 3)
      return (*this)->emitOpError("attribute 'permutation' expects integer elements");
    llvm::SmallDenseSet<int64_t, 4> seen;
    for (int64_t value : values) {
      if (value < 0 || value >= 3)
        return (*this)->emitOpError("attribute 'permutation' expects values in [0, 3)");
      if (!seen.insert(value).second)
        return (*this)->emitOpError("attribute 'permutation' expects distinct values");
    }
  }
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
  if (!llvm::isa<ActivationFunctionAttr>((*this)->getAttr("activation_function")))
    return (*this)->emitOpError("attribute 'activation_function' expects ActivationFunctionAttr");
  if (!llvm::isa<CellOperationAttr>((*this)->getAttr("cell_operation")))
    return (*this)->emitOpError("attribute 'cell_operation' expects CellOperationAttr");
  if (!llvm::isa<PaddingAttr>((*this)->getAttr("pad")))
    return (*this)->emitOpError("attribute 'pad' expects PaddingAttr");
  if (!llvm::isa<IntegerAttr>((*this)->getAttr("x_dilation_rate")))
    return (*this)->emitOpError("attribute 'x_dilation_rate' expects IntegerAttr");
  if (!llvm::isa<IntegerAttr>((*this)->getAttr("x_out_dim")))
    return (*this)->emitOpError("attribute 'x_out_dim' expects IntegerAttr");
  if (!llvm::isa<IntegerAttr>((*this)->getAttr("x_stride")))
    return (*this)->emitOpError("attribute 'x_stride' expects IntegerAttr");
  if (!llvm::isa<IntegerAttr>((*this)->getAttr("y_dilation_rate")))
    return (*this)->emitOpError("attribute 'y_dilation_rate' expects IntegerAttr");
  if (!llvm::isa<IntegerAttr>((*this)->getAttr("y_out_dim")))
    return (*this)->emitOpError("attribute 'y_out_dim' expects IntegerAttr");
  if (!llvm::isa<IntegerAttr>((*this)->getAttr("y_stride")))
    return (*this)->emitOpError("attribute 'y_stride' expects IntegerAttr");
  if (llvm::cast<ActivationFunctionAttr>((*this)->getAttr("activation_function")).getValue() != ActivationFunction::None)
    return (*this)->emitOpError("attribute 'activation_function' expects NONE");
  return success();
}

LogicalResult dwc::UnsortedSegmentReduceOp::verify() {
  if (!(*this)->hasAttr("num_segments"))
    return (*this)->emitOpError("expected op 'dwc.unsorted_segment_reduce' to have attribute 'num_segments'");
  if (!(*this)->hasAttr("op_type"))
    return (*this)->emitOpError("expected op 'dwc.unsorted_segment_reduce' to have attribute 'op_type'");
  if (!llvm::isa<IntegerAttr>((*this)->getAttr("num_segments")))
    return (*this)->emitOpError("attribute 'num_segments' expects IntegerAttr");
  if (!llvm::isa<ReductionTypeAttr>((*this)->getAttr("op_type")))
    return (*this)->emitOpError("attribute 'op_type' expects ReductionTypeAttr");
  auto opType = llvm::cast<ReductionTypeAttr>((*this)->getAttr("op_type")).getValue();
  if (opType != ReductionType::Sum && opType != ReductionType::Max)
    return (*this)->emitOpError("attribute 'op_type' expects SUM or MAX");
  return success();
}

LogicalResult dwc::AbsOp::verify() {
  return success();
}

LogicalResult dwc::ArangeOp::verify() {
  return success();
}

LogicalResult dwc::AttentionOp::verify() {
  return success();
}

LogicalResult dwc::AttentionNonLinearFunctionOp::verify() {
  return success();
}

LogicalResult dwc::BatchToSpaceOp::verify() {
  return success();
}

LogicalResult dwc::BitSelectOp::verify() {
  return success();
}

LogicalResult dwc::BroadcastOp::verify() {
  return success();
}

LogicalResult dwc::CbrtOp::verify() {
  return success();
}

LogicalResult dwc::ClampOp::verify() {
  return success();
}

LogicalResult dwc::CollectivePermuteOp::verify() {
  return success();
}

LogicalResult dwc::CompositeOp::verify() {
  return success();
}

LogicalResult dwc::ConstNoneOp::verify() {
  return success();
}

LogicalResult dwc::ConvolutionSubChannelOp::verify() {
  return success();
}

LogicalResult dwc::CostVolumeOp::verify() {
  return success();
}

LogicalResult dwc::CountLeadingZerosOp::verify() {
  return success();
}

LogicalResult dwc::CustomComputeOp::verify() {
  return success();
}

LogicalResult dwc::DeclareTensorOp::verify() {
  return success();
}

LogicalResult dwc::DepthToSpaceOp::verify() {
  return success();
}

LogicalResult dwc::DepthwiseConvolutionOp::verify() {
  return success();
}

LogicalResult dwc::DeviceLaunchOp::verify() {
  return success();
}

LogicalResult dwc::DmaOpGatherOp::verify() {
  return success();
}

LogicalResult dwc::DynamicBroadcastOp::verify() {
  return success();
}

LogicalResult dwc::DynamicQuantizeOp::verify() {
  return success();
}

LogicalResult dwc::EnsureShapeOp::verify() {
  return success();
}

LogicalResult dwc::ExternCallOp::verify() {
  return success();
}

LogicalResult dwc::FastWalshHadamardTransformOp::verify() {
  return success();
}

LogicalResult dwc::FullyConnectedSubByteParamOp::verify() {
  return success();
}

LogicalResult dwc::FullyConnectedSubChannelOp::verify() {
  return success();
}

LogicalResult dwc::FunctionalIfOp::verify() {
  return success();
}

LogicalResult dwc::FunctionalWhileOp::verify() {
  return success();
}

LogicalResult dwc::GatherOperationOp::verify() {
  return success();
}

LogicalResult dwc::GenericMoveOp::verify() {
  return success();
}

LogicalResult dwc::GenericScatterOp::verify() {
  return success();
}

LogicalResult dwc::HibGatherOp::verify() {
  return success();
}

LogicalResult dwc::HibGatherFilterOp::verify() {
  return success();
}

LogicalResult dwc::HostedTensorOp::verify() {
  return success();
}

LogicalResult dwc::ImageInterpolationOp::verify() {
  if (getInputs().size() != 1 && getInputs().size() != 2)
    return emitOpError("expects 1 or 2 operands, got ") << getInputs().size();
  return success();
}

LogicalResult dwc::IndexUnpoolOp::verify() {
  return success();
}

LogicalResult dwc::InterleaveOp::verify() {
  return success();
}

LogicalResult dwc::IsFiniteOp::verify() {
  return success();
}

LogicalResult dwc::MaskIndicesOp::verify() {
  return success();
}

LogicalResult dwc::MatrixMultiplySubChannelOp::verify() {
  return success();
}

LogicalResult dwc::MultinomialOp::verify() {
  return success();
}

LogicalResult dwc::MultinormalOp::verify() {
  return success();
}

LogicalResult dwc::NegateOp::verify() {
  return success();
}

LogicalResult dwc::NormalizationOp::verify() {
  return success();
}

LogicalResult dwc::OneHotTpuOp::verify() {
  return success();
}

LogicalResult dwc::PackBitsOp::verify() {
  return success();
}

LogicalResult dwc::PoolingOp::verify() {
  return success();
}

LogicalResult dwc::ProbeSubtensorOp::verify() {
  return success();
}

LogicalResult dwc::ReducePrecisionOp::verify() {
  return success();
}

LogicalResult dwc::ReduceWindowOp::verify() {
  return success();
}

LogicalResult dwc::ResamplerOp::verify() {
  return success();
}

LogicalResult dwc::ReverseOp::verify() {
  return success();
}

LogicalResult dwc::VicaAddPoolOp::verify() {
  return success();
}

LogicalResult dwc::VicaConvD2sOp::verify() {
  return success();
}

LogicalResult dwc::VicaCustomPaddingOp::verify() {
  return success();
}

LogicalResult dwc::VicaFusedConvOp::verify() {
  return success();
}

LogicalResult dwc::VicaFusedNormOp::verify() {
  return success();
}

LogicalResult dwc::RollOp::verify() {
  return success();
}

LogicalResult dwc::ScalarCoreConstantOp::verify() {
  return success();
}

LogicalResult dwc::ScatterOperationOp::verify() {
  return success();
}

LogicalResult dwc::ShiftLeftOp::verify() {
  return success();
}

LogicalResult dwc::ShiftRightLogicalOp::verify() {
  return success();
}

LogicalResult dwc::SpaceToBatchOp::verify() {
  return success();
}

LogicalResult dwc::SpaceToDepthOp::verify() {
  return success();
}

LogicalResult dwc::SparseFullyConnectedSubByteParamOp::verify() {
  return success();
}

LogicalResult dwc::SparseParameterOp::verify() {
  return success();
}

LogicalResult dwc::SpillOp::verify() {
  return success();
}

LogicalResult dwc::StatisticalTopKOp::verify() {
  return success();
}

LogicalResult dwc::TensorLsGatherOp::verify() {
  return success();
}

LogicalResult dwc::TensorLsScatterOp::verify() {
  return success();
}

LogicalResult dwc::TensorLsScatterOperationOp::verify() {
  return success();
}

LogicalResult dwc::TensorOpGatherOp::verify() {
  return success();
}

LogicalResult dwc::TopKOp::verify() {
  return success();
}

LogicalResult dwc::TransposedConvolutionSubChannelOp::verify() {
  return success();
}

LogicalResult dwc::TruncateFloatsOp::verify() {
  return success();
}

LogicalResult dwc::UniformRandomNumberGenerationOp::verify() {
  return success();
}

LogicalResult dwc::Atan2Op::verify() {
  return success();
}

LogicalResult dwc::GatherNdOp::verify() {
  return success();
}

LogicalResult dwc::AnnotateMaterializePolicyOp::verify() {
  return success();
}

LogicalResult dwc::ActivationFunctionOp::verify() {
  return success();
}

LogicalResult dwc::AdditionalInputOutputOp::verify() {
  return success();
}

LogicalResult dwc::AfOp::verify() {
  return success();
}

LogicalResult dwc::AlgorithmOp::verify() {
  return success();
}

LogicalResult dwc::AndOp::verify() {
  return success();
}

LogicalResult dwc::ArgumentCopyOp::verify() {
  return success();
}

LogicalResult dwc::BiasParameterOp::verify() {
  return success();
}

LogicalResult dwc::BitSelectTypeOp::verify() {
  return success();
}

LogicalResult dwc::BranchOp::verify() {
  return success();
}

LogicalResult dwc::CellOp::verify() {
  return success();
}

LogicalResult dwc::ClassificationTypeOp::verify() {
  return success();
}

LogicalResult dwc::CodegenOp::verify() {
  return success();
}

LogicalResult dwc::ComparisonTypeOp::verify() {
  return success();
}

LogicalResult dwc::CompilationUnitOp::verify() {
  return success();
}

LogicalResult dwc::CompilationUnitWrapperOp::verify() {
  return success();
}

LogicalResult dwc::CompressionModeOp::verify() {
  return success();
}

LogicalResult dwc::Convolution3dOp::verify() {
  return success();
}

LogicalResult dwc::CumulativeTypeOp::verify() {
  return success();
}

LogicalResult dwc::CustomPaddingValueTypeOp::verify() {
  return success();
}

LogicalResult dwc::CwiseTypeOp::verify() {
  return success();
}

LogicalResult dwc::DeclareTensorStaticOp::verify() {
  return success();
}

LogicalResult dwc::DeviceLaunchFuncOp::verify() {
  return success();
}

LogicalResult dwc::DeviceTypeOp::verify() {
  return success();
}

LogicalResult dwc::DimMappingOp::verify() {
  return success();
}

LogicalResult dwc::DimensionLayoutOp::verify() {
  return success();
}

LogicalResult dwc::DiveUnrollFactorOp::verify() {
  return success();
}

LogicalResult dwc::DynamicIntermediateInputShardOp::verify() {
  return success();
}

LogicalResult dwc::DynamicIntermediateOutputShardOp::verify() {
  return success();
}

LogicalResult dwc::DynamicSliceNdOp::verify() {
  return success();
}

LogicalResult dwc::DynamicSliceNdV2Op::verify() {
  return success();
}

LogicalResult dwc::DynamicUpdateSliceNdOp::verify() {
  return success();
}

LogicalResult dwc::DynamicUpdateSliceNdV2Op::verify() {
  return success();
}

LogicalResult dwc::EncodingOp::verify() {
  return success();
}

LogicalResult dwc::EngineOp::verify() {
  return success();
}

LogicalResult dwc::FetchOp::verify() {
  return success();
}

LogicalResult dwc::FullyConnectedSubChannelV2Op::verify() {
  return success();
}

LogicalResult dwc::FullyConnectedZinIndexedOp::verify() {
  return success();
}

LogicalResult dwc::FullyConnectedZoutIndexedOp::verify() {
  return success();
}

LogicalResult dwc::GenericPadOp::verify() {
  return success();
}

LogicalResult dwc::HardwareClusterIdPerSignatureOp::verify() {
  return success();
}

LogicalResult dwc::HostSpaceOp::verify() {
  return success();
}

LogicalResult dwc::IdentityOp::verify() {
  return success();
}

LogicalResult dwc::IfOp::verify() {
  return success();
}

LogicalResult dwc::ImageFormatOp::verify() {
  return success();
}

LogicalResult dwc::InputOp::verify() {
  return success();
}

LogicalResult dwc::InputShardOp::verify() {
  return success();
}

LogicalResult dwc::InterDieInputOp::verify() {
  return success();
}

LogicalResult dwc::InterDieOutputOp::verify() {
  return success();
}

LogicalResult dwc::IntermediateInputOp::verify() {
  return success();
}

LogicalResult dwc::IntermediateInputShardOp::verify() {
  return success();
}

LogicalResult dwc::IntermediateOutputOp::verify() {
  return success();
}

LogicalResult dwc::IntermediateOutputShardOp::verify() {
  return success();
}

LogicalResult dwc::InternalPaddingOp::verify() {
  return success();
}

LogicalResult dwc::IoColocationPairsOp::verify() {
  return success();
}

LogicalResult dwc::IsExternalParameterOp::verify() {
  return success();
}

LogicalResult dwc::JumpOp::verify() {
  return success();
}

LogicalResult dwc::KernelLevelOp::verify() {
  return success();
}

LogicalResult dwc::KnownTripCountOp::verify() {
  return success();
}

LogicalResult dwc::LaunchCustomKernelOp::verify() {
  return success();
}

LogicalResult dwc::LinearFunctionOp::verify() {
  return success();
}

LogicalResult dwc::LogOp::verify() {
  return success();
}

LogicalResult dwc::LoopShardingOp::verify() {
  return success();
}

LogicalResult dwc::LowerBoundOp::verify() {
  return success();
}

LogicalResult dwc::MaterializePolicyOp::verify() {
  return success();
}

LogicalResult dwc::MemoryLocationOp::verify() {
  return success();
}

LogicalResult dwc::MemorySpaceOp::verify() {
  return success();
}

LogicalResult dwc::MeshDimOp::verify() {
  return success();
}

LogicalResult dwc::MlirOp::verify() {
  return success();
}

LogicalResult dwc::MultimediaOp::verify() {
  return success();
}

LogicalResult dwc::NluE8m0RoundingOp::verify() {
  return success();
}

LogicalResult dwc::NluPreprocessOp::verify() {
  return success();
}

LogicalResult dwc::NormalizationTypeOp::verify() {
  return success();
}

LogicalResult dwc::OrOp::verify() {
  return success();
}

LogicalResult dwc::OutputOp::verify() {
  return success();
}

LogicalResult dwc::OutputShardOp::verify() {
  return success();
}

LogicalResult dwc::PadOp::verify() {
  return success();
}

LogicalResult dwc::PaddingValueTypeOp::verify() {
  return success();
}

LogicalResult dwc::ParameterOp::verify() {
  return success();
}

LogicalResult dwc::ParameterLookupTableOp::verify() {
  return success();
}

LogicalResult dwc::PerZOutScalePaddingOp::verify() {
  return success();
}

LogicalResult dwc::PoolOp::verify() {
  return success();
}

LogicalResult dwc::Pooling3dOp::verify() {
  return success();
}

LogicalResult dwc::ProbeOp::verify() {
  return success();
}

LogicalResult dwc::PseudoDynamicImageInterpolationOp::verify() {
  return success();
}

LogicalResult dwc::PseudoDynamicPadOp::verify() {
  return success();
}

LogicalResult dwc::PseudoDynamicReshapeOp::verify() {
  return success();
}

LogicalResult dwc::PseudoDynamicSliceOp::verify() {
  return success();
}

LogicalResult dwc::PseudoExpandDimsOp::verify() {
  return success();
}

LogicalResult dwc::PseudoFillOp::verify() {
  return success();
}

LogicalResult dwc::PseudoGenericNormOp::verify() {
  return success();
}

LogicalResult dwc::PseudoGroupNormOp::verify() {
  return success();
}

LogicalResult dwc::PseudoMirrorPadOp::verify() {
  return success();
}

LogicalResult dwc::PseudoRangeOp::verify() {
  return success();
}

LogicalResult dwc::PseudoShapeOp::verify() {
  return success();
}

LogicalResult dwc::PseudoSqueezeOp::verify() {
  return success();
}

LogicalResult dwc::ReduceWindowTypeOp::verify() {
  return success();
}

LogicalResult dwc::ReductionTypeOp::verify() {
  return success();
}

LogicalResult dwc::ResamplerOptionsOp::verify() {
  return success();
}

LogicalResult dwc::VicaAddOp::verify() {
  return success();
}

LogicalResult dwc::ScalarTypeOp::verify() {
  return success();
}

LogicalResult dwc::ShardBarrierOp::verify() {
  return success();
}

LogicalResult dwc::ShardBodyOp::verify() {
  return success();
}

LogicalResult dwc::ShardGroupOp::verify() {
  return success();
}

LogicalResult dwc::ShardSinkOp::verify() {
  return success();
}

LogicalResult dwc::ShardSourceOp::verify() {
  return success();
}

LogicalResult dwc::ShiftRightArithmeticOp::verify() {
  return success();
}

LogicalResult dwc::SignatureNaOp::verify() {
  return success();
}

LogicalResult dwc::SignatureNameOp::verify() {
  return success();
}

LogicalResult dwc::SparseFullyConnectedOp::verify() {
  return success();
}

LogicalResult dwc::SpillLocationOp::verify() {
  return success();
}

LogicalResult dwc::StabletgKernelOp::verify() {
  return success();
}

LogicalResult dwc::StrideMethodOp::verify() {
  return success();
}

LogicalResult dwc::TileMeshOp::verify() {
  return success();
}

LogicalResult dwc::TileUidOp::verify() {
  return success();
}

LogicalResult dwc::TpuGroupIdOp::verify() {
  return success();
}

LogicalResult dwc::TransformationTypeOp::verify() {
  return success();
}

LogicalResult dwc::VisibleTilesPerSignatureOp::verify() {
  return success();
}

LogicalResult dwc::VrgkhOperationModeOp::verify() {
  return success();
}

LogicalResult dwc::VtidOp::verify() {
  return success();
}

LogicalResult dwc::WhileOp::verify() {
  return success();
}

LogicalResult dwc::XorOp::verify() {
  return success();
}

LogicalResult dwc::YieldOp::verify() {
  return success();
}
