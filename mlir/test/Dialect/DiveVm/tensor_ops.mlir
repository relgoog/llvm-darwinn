// RUN: mlir-opt %s --verify-each | FileCheck %s

// -----
// CHECK-LABEL: allocate
func.func @test_allocate() -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.allocate
  %0 = "dive_vm_tensor.allocate"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: const
func.func @test_const() -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.const
  %0 = "dive_vm_tensor.const"() {value = dense<0.0> : tensor<4xf32>} : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: extract_slice
func.func @test_extract_slice(%arg0: tensor<8xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.extract_slice
  %0 = "dive_vm_tensor.extract_slice"(%arg0) : (tensor<8xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: insert_slice
func.func @test_insert_slice(%arg0: tensor<4xf32>, %arg1: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK: dive_vm_tensor.insert_slice
  %0 = "dive_vm_tensor.insert_slice"(%arg0, %arg1) : (tensor<4xf32>, tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----
// CHECK-LABEL: type_cast
func.func @test_type_cast(%arg0: tensor<4xf32>) -> tensor<4xbf16> {
  // CHECK: dive_vm_tensor.type_cast
  %0 = "dive_vm_tensor.type_cast"(%arg0) : (tensor<4xf32>) -> tensor<4xbf16>
  return %0 : tensor<4xbf16>
}

// -----
// CHECK-LABEL: codegen
func.func @test_codegen(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.codegen
  %0 = "dwg_tensor.codegen"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_const
func.func @test_dwg_const(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.const
  %0 = "dwg_tensor.const"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dynamic_shape_scope
func.func @test_dynamic_shape_scope(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK: dwg_tensor.dynamic_shape_scope
  %0 = "dwg_tensor.dynamic_shape_scope"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----
// CHECK-LABEL: eval_with_shape
func.func @test_eval_with_shape(%arg0: tensor<?xf32>, %arg1: tensor<1xi64>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.eval_with_shape
  %0 = "dwg_tensor.eval_with_shape"(%arg0, %arg1) : (tensor<?xf32>, tensor<1xi64>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_abs
func.func @test_dive_vm_tensor_abs(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.abs
  %0 = "dive_vm_tensor.abs"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_add
func.func @test_dive_vm_tensor_add(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.add
  %0 = "dive_vm_tensor.add"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_add_imm
func.func @test_dive_vm_tensor_add_imm(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.add_imm
  %0 = "dive_vm_tensor.add_imm"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_address_of
func.func @test_dive_vm_tensor_address_of(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.address_of
  %0 = "dive_vm_tensor.address_of"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_address_of_activation
func.func @test_dive_vm_tensor_address_of_activation(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.address_of_activation
  %0 = "dive_vm_tensor.address_of_activation"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_address_of_input_activation
func.func @test_dive_vm_tensor_address_of_input_activation(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.address_of_input_activation
  %0 = "dive_vm_tensor.address_of_input_activation"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_address_of_output_activation
func.func @test_dive_vm_tensor_address_of_output_activation(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.address_of_output_activation
  %0 = "dive_vm_tensor.address_of_output_activation"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_address_of_parameter
func.func @test_dive_vm_tensor_address_of_parameter(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.address_of_parameter
  %0 = "dive_vm_tensor.address_of_parameter"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_address_of_parameter_region
func.func @test_dive_vm_tensor_address_of_parameter_region(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.address_of_parameter_region
  %0 = "dive_vm_tensor.address_of_parameter_region"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_address_of_scratch
func.func @test_dive_vm_tensor_address_of_scratch(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.address_of_scratch
  %0 = "dive_vm_tensor.address_of_scratch"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_arithmetic_left_shift
func.func @test_dive_vm_tensor_arithmetic_left_shift(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.arithmetic_left_shift
  %0 = "dive_vm_tensor.arithmetic_left_shift"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_arithmetic_left_shift_imm
func.func @test_dive_vm_tensor_arithmetic_left_shift_imm(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.arithmetic_left_shift_imm
  %0 = "dive_vm_tensor.arithmetic_left_shift_imm"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_arithmetic_right_shift
func.func @test_dive_vm_tensor_arithmetic_right_shift(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.arithmetic_right_shift
  %0 = "dive_vm_tensor.arithmetic_right_shift"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_arithmetic_right_shift_imm
func.func @test_dive_vm_tensor_arithmetic_right_shift_imm(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.arithmetic_right_shift_imm
  %0 = "dive_vm_tensor.arithmetic_right_shift_imm"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_bitcast
func.func @test_dive_vm_tensor_bitcast(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.bitcast
  %0 = "dive_vm_tensor.bitcast"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_bitwise_and
func.func @test_dive_vm_tensor_bitwise_and(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.bitwise_and
  %0 = "dive_vm_tensor.bitwise_and"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_bitwise_and_imm
func.func @test_dive_vm_tensor_bitwise_and_imm(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.bitwise_and_imm
  %0 = "dive_vm_tensor.bitwise_and_imm"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_bitwise_or
func.func @test_dive_vm_tensor_bitwise_or(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.bitwise_or
  %0 = "dive_vm_tensor.bitwise_or"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_bitwise_or_imm
func.func @test_dive_vm_tensor_bitwise_or_imm(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.bitwise_or_imm
  %0 = "dive_vm_tensor.bitwise_or_imm"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_bitwise_xor
func.func @test_dive_vm_tensor_bitwise_xor(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.bitwise_xor
  %0 = "dive_vm_tensor.bitwise_xor"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_bitwise_xor_imm
func.func @test_dive_vm_tensor_bitwise_xor_imm(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.bitwise_xor_imm
  %0 = "dive_vm_tensor.bitwise_xor_imm"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_br
func.func @test_dive_vm_tensor_br(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.br
  %0 = "dive_vm_tensor.br"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_cache_clean_invalidate
func.func @test_dive_vm_tensor_cache_clean_invalidate(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.cache_clean_invalidate
  %0 = "dive_vm_tensor.cache_clean_invalidate"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_cast
func.func @test_dive_vm_tensor_cast(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.cast
  %0 = "dive_vm_tensor.cast"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_concatenate
func.func @test_dive_vm_tensor_concatenate(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.concatenate
  %0 = "dive_vm_tensor.concatenate"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_cond_br
func.func @test_dive_vm_tensor_cond_br(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.cond_br
  %0 = "dive_vm_tensor.cond_br"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_copy
func.func @test_dive_vm_tensor_copy(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.copy
  %0 = "dive_vm_tensor.copy"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_copy_imm
func.func @test_dive_vm_tensor_copy_imm(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.copy_imm
  %0 = "dive_vm_tensor.copy_imm"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_count_leading_zeros
func.func @test_dive_vm_tensor_count_leading_zeros(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.count_leading_zeros
  %0 = "dive_vm_tensor.count_leading_zeros"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_cumsum
func.func @test_dive_vm_tensor_cumsum(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.cumsum
  %0 = "dive_vm_tensor.cumsum"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_custom_on_buffers
func.func @test_dive_vm_tensor_custom_on_buffers(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.custom_on_buffers
  %0 = "dive_vm_tensor.custom_on_buffers"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_declare_tensor
func.func @test_dive_vm_tensor_declare_tensor(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.declare_tensor
  %0 = "dive_vm_tensor.declare_tensor"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_div
func.func @test_dive_vm_tensor_div(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.div
  %0 = "dive_vm_tensor.div"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_div_imm
func.func @test_dive_vm_tensor_div_imm(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.div_imm
  %0 = "dive_vm_tensor.div_imm"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_dma_hint
func.func @test_dive_vm_tensor_dma_hint(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.dma_hint
  %0 = "dive_vm_tensor.dma_hint"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_dynamic_slice_y
func.func @test_dive_vm_tensor_dynamic_slice_y(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.dynamic_slice_y
  %0 = "dive_vm_tensor.dynamic_slice_y"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_equal
func.func @test_dive_vm_tensor_equal(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.equal
  %0 = "dive_vm_tensor.equal"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_equal_imm
func.func @test_dive_vm_tensor_equal_imm(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.equal_imm
  %0 = "dive_vm_tensor.equal_imm"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_eval_with_shape
func.func @test_dive_vm_tensor_eval_with_shape(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.eval_with_shape
  %0 = "dive_vm_tensor.eval_with_shape"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_extract
func.func @test_dive_vm_tensor_extract(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.extract
  %0 = "dive_vm_tensor.extract"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_fill
func.func @test_dive_vm_tensor_fill(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.fill
  %0 = "dive_vm_tensor.fill"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_floor_div
func.func @test_dive_vm_tensor_floor_div(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.floor_div
  %0 = "dive_vm_tensor.floor_div"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_gather
func.func @test_dive_vm_tensor_gather(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.gather
  %0 = "dive_vm_tensor.gather"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_gather_nd
func.func @test_dive_vm_tensor_gather_nd(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.gather_nd
  %0 = "dive_vm_tensor.gather_nd"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_greater
func.func @test_dive_vm_tensor_greater(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.greater
  %0 = "dive_vm_tensor.greater"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_greater_equal
func.func @test_dive_vm_tensor_greater_equal(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.greater_equal
  %0 = "dive_vm_tensor.greater_equal"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_greater_equal_imm
func.func @test_dive_vm_tensor_greater_equal_imm(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.greater_equal_imm
  %0 = "dive_vm_tensor.greater_equal_imm"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_greater_imm
func.func @test_dive_vm_tensor_greater_imm(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.greater_imm
  %0 = "dive_vm_tensor.greater_imm"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_if
func.func @test_dive_vm_tensor_if(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.if
  %0 = "dive_vm_tensor.if"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_insert
func.func @test_dive_vm_tensor_insert(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.insert
  %0 = "dive_vm_tensor.insert"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_less
func.func @test_dive_vm_tensor_less(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.less
  %0 = "dive_vm_tensor.less"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_less_equal
func.func @test_dive_vm_tensor_less_equal(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.less_equal
  %0 = "dive_vm_tensor.less_equal"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_less_equal_imm
func.func @test_dive_vm_tensor_less_equal_imm(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.less_equal_imm
  %0 = "dive_vm_tensor.less_equal_imm"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_less_imm
func.func @test_dive_vm_tensor_less_imm(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.less_imm
  %0 = "dive_vm_tensor.less_imm"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_load
func.func @test_dive_vm_tensor_load(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.load
  %0 = "dive_vm_tensor.load"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_load_indirect
func.func @test_dive_vm_tensor_load_indirect(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.load_indirect
  %0 = "dive_vm_tensor.load_indirect"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_logical_and
func.func @test_dive_vm_tensor_logical_and(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.logical_and
  %0 = "dive_vm_tensor.logical_and"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_logical_and_imm
func.func @test_dive_vm_tensor_logical_and_imm(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.logical_and_imm
  %0 = "dive_vm_tensor.logical_and_imm"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_logical_right_shift
func.func @test_dive_vm_tensor_logical_right_shift(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.logical_right_shift
  %0 = "dive_vm_tensor.logical_right_shift"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_logical_right_shift_imm
func.func @test_dive_vm_tensor_logical_right_shift_imm(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.logical_right_shift_imm
  %0 = "dive_vm_tensor.logical_right_shift_imm"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_mark_value_with_shape
func.func @test_dive_vm_tensor_mark_value_with_shape(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.mark_value_with_shape
  %0 = "dive_vm_tensor.mark_value_with_shape"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_max
func.func @test_dive_vm_tensor_max(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.max
  %0 = "dive_vm_tensor.max"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_max_imm
func.func @test_dive_vm_tensor_max_imm(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.max_imm
  %0 = "dive_vm_tensor.max_imm"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_min
func.func @test_dive_vm_tensor_min(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.min
  %0 = "dive_vm_tensor.min"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_min_imm
func.func @test_dive_vm_tensor_min_imm(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.min_imm
  %0 = "dive_vm_tensor.min_imm"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_mul
func.func @test_dive_vm_tensor_mul(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.mul
  %0 = "dive_vm_tensor.mul"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_mul_imm
func.func @test_dive_vm_tensor_mul_imm(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.mul_imm
  %0 = "dive_vm_tensor.mul_imm"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_not_equal
func.func @test_dive_vm_tensor_not_equal(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.not_equal
  %0 = "dive_vm_tensor.not_equal"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_not_equal_imm
func.func @test_dive_vm_tensor_not_equal_imm(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.not_equal_imm
  %0 = "dive_vm_tensor.not_equal_imm"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_one_hot
func.func @test_dive_vm_tensor_one_hot(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.one_hot
  %0 = "dive_vm_tensor.one_hot"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_pad
func.func @test_dive_vm_tensor_pad(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.pad
  %0 = "dive_vm_tensor.pad"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_perform_software_preemption_if_requested
func.func @test_dive_vm_tensor_perform_software_preemption_if_requested(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.perform_software_preemption_if_requested
  %0 = "dive_vm_tensor.perform_software_preemption_if_requested"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_pop_count
func.func @test_dive_vm_tensor_pop_count(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.pop_count
  %0 = "dive_vm_tensor.pop_count"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_pow
func.func @test_dive_vm_tensor_pow(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.pow
  %0 = "dive_vm_tensor.pow"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_pow_imm
func.func @test_dive_vm_tensor_pow_imm(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.pow_imm
  %0 = "dive_vm_tensor.pow_imm"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_print
func.func @test_dive_vm_tensor_print(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.print
  %0 = "dive_vm_tensor.print"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_put_bits
func.func @test_dive_vm_tensor_put_bits(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.put_bits
  %0 = "dive_vm_tensor.put_bits"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_reduction
func.func @test_dive_vm_tensor_reduction(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.reduction
  %0 = "dive_vm_tensor.reduction"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_rem
func.func @test_dive_vm_tensor_rem(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.rem
  %0 = "dive_vm_tensor.rem"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_rem_imm
func.func @test_dive_vm_tensor_rem_imm(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.rem_imm
  %0 = "dive_vm_tensor.rem_imm"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_roll
func.func @test_dive_vm_tensor_roll(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.roll
  %0 = "dive_vm_tensor.roll"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_run_serialized_model
func.func @test_dive_vm_tensor_run_serialized_model(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.run_serialized_model
  %0 = "dive_vm_tensor.run_serialized_model"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_scatter_nd
func.func @test_dive_vm_tensor_scatter_nd(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.scatter_nd
  %0 = "dive_vm_tensor.scatter_nd"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_select
func.func @test_dive_vm_tensor_select(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.select
  %0 = "dive_vm_tensor.select"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_shape_of_activation
func.func @test_dive_vm_tensor_shape_of_activation(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.shape_of_activation
  %0 = "dive_vm_tensor.shape_of_activation"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_sign
func.func @test_dive_vm_tensor_sign(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.sign
  %0 = "dive_vm_tensor.sign"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_store
func.func @test_dive_vm_tensor_store(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.store
  %0 = "dive_vm_tensor.store"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_store_indirect
func.func @test_dive_vm_tensor_store_indirect(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.store_indirect
  %0 = "dive_vm_tensor.store_indirect"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_sub
func.func @test_dive_vm_tensor_sub(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.sub
  %0 = "dive_vm_tensor.sub"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_sub_imm
func.func @test_dive_vm_tensor_sub_imm(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.sub_imm
  %0 = "dive_vm_tensor.sub_imm"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_switch
func.func @test_dive_vm_tensor_switch(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.switch
  %0 = "dive_vm_tensor.switch"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_top_k
func.func @test_dive_vm_tensor_top_k(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.top_k
  %0 = "dive_vm_tensor.top_k"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_tpu_offload
func.func @test_dive_vm_tensor_tpu_offload(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.tpu_offload
  %0 = "dive_vm_tensor.tpu_offload"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_while
func.func @test_dive_vm_tensor_while(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.while
  %0 = "dive_vm_tensor.while"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_write_dma_descriptor
func.func @test_dive_vm_tensor_write_dma_descriptor(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.write_dma_descriptor
  %0 = "dive_vm_tensor.write_dma_descriptor"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_tensor_yield
func.func @test_dive_vm_tensor_yield(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm_tensor.yield
  %0 = "dive_vm_tensor.yield"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_tensor_abs
func.func @test_dwg_tensor_abs(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.abs
  %0 = "dwg_tensor.abs"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_tensor_add
func.func @test_dwg_tensor_add(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.add
  %0 = "dwg_tensor.add"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_tensor_arithmetic_left_shift
func.func @test_dwg_tensor_arithmetic_left_shift(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.arithmetic_left_shift
  %0 = "dwg_tensor.arithmetic_left_shift"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_tensor_arithmetic_right_shift
func.func @test_dwg_tensor_arithmetic_right_shift(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.arithmetic_right_shift
  %0 = "dwg_tensor.arithmetic_right_shift"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_tensor_barrier
func.func @test_dwg_tensor_barrier(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.barrier
  %0 = "dwg_tensor.barrier"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_tensor_batch_matrix_nms
func.func @test_dwg_tensor_batch_matrix_nms(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.batch_matrix_nms
  %0 = "dwg_tensor.batch_matrix_nms"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_tensor_bitwise_and
func.func @test_dwg_tensor_bitwise_and(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.bitwise_and
  %0 = "dwg_tensor.bitwise_and"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_tensor_bitwise_or
func.func @test_dwg_tensor_bitwise_or(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.bitwise_or
  %0 = "dwg_tensor.bitwise_or"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_tensor_bitwise_xor
func.func @test_dwg_tensor_bitwise_xor(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.bitwise_xor
  %0 = "dwg_tensor.bitwise_xor"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_tensor_cast
func.func @test_dwg_tensor_cast(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.cast
  %0 = "dwg_tensor.cast"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_tensor_cast_from_index
func.func @test_dwg_tensor_cast_from_index(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.cast_from_index
  %0 = "dwg_tensor.cast_from_index"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_tensor_cast_to_index
func.func @test_dwg_tensor_cast_to_index(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.cast_to_index
  %0 = "dwg_tensor.cast_to_index"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_tensor_concatenation
func.func @test_dwg_tensor_concatenation(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.concatenation
  %0 = "dwg_tensor.concatenation"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_tensor_const_bytes
func.func @test_dwg_tensor_const_bytes(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.const_bytes
  %0 = "dwg_tensor.const_bytes"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_tensor_const_none
func.func @test_dwg_tensor_const_none(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.const_none
  %0 = "dwg_tensor.const_none"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_tensor_copy
func.func @test_dwg_tensor_copy(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.copy
  %0 = "dwg_tensor.copy"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_tensor_copy_outer_slice
func.func @test_dwg_tensor_copy_outer_slice(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.copy_outer_slice
  %0 = "dwg_tensor.copy_outer_slice"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_tensor_count_leading_zeros
func.func @test_dwg_tensor_count_leading_zeros(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.count_leading_zeros
  %0 = "dwg_tensor.count_leading_zeros"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_tensor_cumsum
func.func @test_dwg_tensor_cumsum(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.cumsum
  %0 = "dwg_tensor.cumsum"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_tensor_declare_tensor
func.func @test_dwg_tensor_declare_tensor(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.declare_tensor
  %0 = "dwg_tensor.declare_tensor"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_tensor_declare_tensor_static
func.func @test_dwg_tensor_declare_tensor_static(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.declare_tensor_static
  %0 = "dwg_tensor.declare_tensor_static"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_tensor_dimension_length
func.func @test_dwg_tensor_dimension_length(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.dimension_length
  %0 = "dwg_tensor.dimension_length"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_tensor_div
func.func @test_dwg_tensor_div(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.div
  %0 = "dwg_tensor.div"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_tensor_eq
func.func @test_dwg_tensor_eq(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.eq
  %0 = "dwg_tensor.eq"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_tensor_for
func.func @test_dwg_tensor_for(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.for
  %0 = "dwg_tensor.for"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_tensor_gather
func.func @test_dwg_tensor_gather(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.gather
  %0 = "dwg_tensor.gather"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_tensor_gather_nd
func.func @test_dwg_tensor_gather_nd(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.gather_nd
  %0 = "dwg_tensor.gather_nd"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_tensor_ge
func.func @test_dwg_tensor_ge(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.ge
  %0 = "dwg_tensor.ge"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_tensor_gt
func.func @test_dwg_tensor_gt(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.gt
  %0 = "dwg_tensor.gt"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_tensor_gumbel_rng
func.func @test_dwg_tensor_gumbel_rng(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.gumbel_rng
  %0 = "dwg_tensor.gumbel_rng"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_tensor_if
func.func @test_dwg_tensor_if(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.if
  %0 = "dwg_tensor.if"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_tensor_landmarks_to_transform_matrix
func.func @test_dwg_tensor_landmarks_to_transform_matrix(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.landmarks_to_transform_matrix
  %0 = "dwg_tensor.landmarks_to_transform_matrix"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_tensor_le
func.func @test_dwg_tensor_le(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.le
  %0 = "dwg_tensor.le"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_tensor_logical_and
func.func @test_dwg_tensor_logical_and(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.logical_and
  %0 = "dwg_tensor.logical_and"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_tensor_lt
func.func @test_dwg_tensor_lt(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.lt
  %0 = "dwg_tensor.lt"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_tensor_maximum
func.func @test_dwg_tensor_maximum(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.maximum
  %0 = "dwg_tensor.maximum"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_tensor_memory_copy
func.func @test_dwg_tensor_memory_copy(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.memory_copy
  %0 = "dwg_tensor.memory_copy"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_tensor_minimum
func.func @test_dwg_tensor_minimum(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.minimum
  %0 = "dwg_tensor.minimum"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_tensor_mod
func.func @test_dwg_tensor_mod(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.mod
  %0 = "dwg_tensor.mod"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_tensor_mul
func.func @test_dwg_tensor_mul(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.mul
  %0 = "dwg_tensor.mul"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_tensor_ne
func.func @test_dwg_tensor_ne(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.ne
  %0 = "dwg_tensor.ne"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_tensor_new_qos_class
func.func @test_dwg_tensor_new_qos_class(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.new_qos_class
  %0 = "dwg_tensor.new_qos_class"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_tensor_one_hot
func.func @test_dwg_tensor_one_hot(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.one_hot
  %0 = "dwg_tensor.one_hot"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_tensor_one_hot_v2
func.func @test_dwg_tensor_one_hot_v2(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.one_hot_v2
  %0 = "dwg_tensor.one_hot_v2"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_tensor_outer_slice
func.func @test_dwg_tensor_outer_slice(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.outer_slice
  %0 = "dwg_tensor.outer_slice"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_tensor_pad
func.func @test_dwg_tensor_pad(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.pad
  %0 = "dwg_tensor.pad"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_tensor_pop_count
func.func @test_dwg_tensor_pop_count(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.pop_count
  %0 = "dwg_tensor.pop_count"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_tensor_pow
func.func @test_dwg_tensor_pow(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.pow
  %0 = "dwg_tensor.pow"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_tensor_raise_error
func.func @test_dwg_tensor_raise_error(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.raise_error
  %0 = "dwg_tensor.raise_error"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_tensor_reduction
func.func @test_dwg_tensor_reduction(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.reduction
  %0 = "dwg_tensor.reduction"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_tensor_reduction_v2
func.func @test_dwg_tensor_reduction_v2(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.reduction_v2
  %0 = "dwg_tensor.reduction_v2"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_tensor_relayout
func.func @test_dwg_tensor_relayout(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.relayout
  %0 = "dwg_tensor.relayout"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_tensor_roi_to_transform_matrix
func.func @test_dwg_tensor_roi_to_transform_matrix(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.roi_to_transform_matrix
  %0 = "dwg_tensor.roi_to_transform_matrix"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_tensor_roll
func.func @test_dwg_tensor_roll(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.roll
  %0 = "dwg_tensor.roll"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_tensor_scatter_nd
func.func @test_dwg_tensor_scatter_nd(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.scatter_nd
  %0 = "dwg_tensor.scatter_nd"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_tensor_select
func.func @test_dwg_tensor_select(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.select
  %0 = "dwg_tensor.select"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_tensor_select_v2
func.func @test_dwg_tensor_select_v2(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.select_v2
  %0 = "dwg_tensor.select_v2"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_tensor_set_dimension_length
func.func @test_dwg_tensor_set_dimension_length(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.set_dimension_length
  %0 = "dwg_tensor.set_dimension_length"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_tensor_sign
func.func @test_dwg_tensor_sign(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.sign
  %0 = "dwg_tensor.sign"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_tensor_single_chip_offload
func.func @test_dwg_tensor_single_chip_offload(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.single_chip_offload
  %0 = "dwg_tensor.single_chip_offload"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_tensor_slice_v2
func.func @test_dwg_tensor_slice_v2(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.slice_v2
  %0 = "dwg_tensor.slice_v2"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_tensor_sort
func.func @test_dwg_tensor_sort(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.sort
  %0 = "dwg_tensor.sort"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_tensor_static_error
func.func @test_dwg_tensor_static_error(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.static_error
  %0 = "dwg_tensor.static_error"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_tensor_sub
func.func @test_dwg_tensor_sub(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.sub
  %0 = "dwg_tensor.sub"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_tensor_switch
func.func @test_dwg_tensor_switch(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.switch
  %0 = "dwg_tensor.switch"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_tensor_test_only_fingerprint_core_
func.func @test_dwg_tensor_test_only_fingerprint_core_(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.test_only_fingerprint_core_
  %0 = "dwg_tensor.test_only_fingerprint_core_"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_tensor_top_k
func.func @test_dwg_tensor_top_k(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.top_k
  %0 = "dwg_tensor.top_k"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_tensor_tpu_offload
func.func @test_dwg_tensor_tpu_offload(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.tpu_offload
  %0 = "dwg_tensor.tpu_offload"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_tensor_transform_landmarks
func.func @test_dwg_tensor_transform_landmarks(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.transform_landmarks
  %0 = "dwg_tensor.transform_landmarks"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_tensor_transform_tensor_bilinear
func.func @test_dwg_tensor_transform_tensor_bilinear(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.transform_tensor_bilinear
  %0 = "dwg_tensor.transform_tensor_bilinear"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_tensor_type_cast
func.func @test_dwg_tensor_type_cast(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.type_cast
  %0 = "dwg_tensor.type_cast"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_tensor_type_cast_dynamic
func.func @test_dwg_tensor_type_cast_dynamic(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.type_cast_dynamic
  %0 = "dwg_tensor.type_cast_dynamic"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_tensor_uniform_random_number_generation
func.func @test_dwg_tensor_uniform_random_number_generation(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.uniform_random_number_generation
  %0 = "dwg_tensor.uniform_random_number_generation"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_tensor_unsorted_segment_reduce
func.func @test_dwg_tensor_unsorted_segment_reduce(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.unsorted_segment_reduce
  %0 = "dwg_tensor.unsorted_segment_reduce"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_tensor_update_outer_slice
func.func @test_dwg_tensor_update_outer_slice(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.update_outer_slice
  %0 = "dwg_tensor.update_outer_slice"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_tensor_use_max_concurrency_mode
func.func @test_dwg_tensor_use_max_concurrency_mode(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.use_max_concurrency_mode
  %0 = "dwg_tensor.use_max_concurrency_mode"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_tensor_while
func.func @test_dwg_tensor_while(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.while
  %0 = "dwg_tensor.while"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dwg_tensor_yield
func.func @test_dwg_tensor_yield(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwg_tensor.yield
  %0 = "dwg_tensor.yield"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
