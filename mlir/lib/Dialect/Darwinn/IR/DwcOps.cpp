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
  if (auto ranked = llvm::dyn_cast<RankedTensorType>(getInputs()[0].getType())) {
    if (ranked.getRank() != 3)
      return (*this)->emitOpError("operand 0 expects Rank 3 tensor");
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
  return success();
}

LogicalResult dwc::UnsortedSegmentReduceOp::verify() {
  if (!(*this)->hasAttr("num_segments"))
    return (*this)->emitOpError("expected op 'dwc.unsorted_segment_reduce' to have attribute 'num_segments'");
  if (!(*this)->hasAttr("op_type"))
    return (*this)->emitOpError("expected op 'dwc.unsorted_segment_reduce' to have attribute 'op_type'");
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

LogicalResult dwc::RkhyAddPoolOp::verify() {
  return success();
}

LogicalResult dwc::RkhyConvD2sOp::verify() {
  return success();
}

LogicalResult dwc::RkhyCustomPaddingOp::verify() {
  return success();
}

LogicalResult dwc::RkhyFusedConvOp::verify() {
  return success();
}

LogicalResult dwc::RkhyFusedNormOp::verify() {
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

LogicalResult dwc::RkhyAddOp::verify() {
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
