// RUN: mlir-opt %s --verify-each | mlir-opt | FileCheck %s
// RUN: mlir-opt %s --mlir-print-op-generic | mlir-opt | FileCheck %s

// -----
// CHECK-LABEL: function_symmetry
func.func @test_function_symmetry(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: darwinn.function_symmetry
  %0 = darwinn.copy_op %arg0 {darwinn.function_symmetry = #darwinn.function_symmetry<symmetric>} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: materialize_policy
func.func @test_materialize_policy(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: darwinn.materialize_policy
  %0 = darwinn.copy_op %arg0 {darwinn.materialize_policy = #darwinn.materialize_policy<on_chip>} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: filter_output
func.func @test_cost_hint_alias(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: darwinn.filter_output
  %0 = darwinn.copy_op %arg0 {darwinn.cost_hint = #darwinn.cost_hint<low>} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: cost_hint
func.func @test_cost_hint(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: darwinn.cost_hint
  %0 = darwinn.copy_op %arg0 {darwinn.cost_hint = #darwinn.cost_hint<low>} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: compute_lowering_hint
func.func @test_compute_lowering_hint(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: darwinn.compute_lowering_hint
  %0 = darwinn.copy_op %arg0 {darwinn.compute_lowering_hint = #darwinn.compute_lowering_hint<direct>} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: compute_type_hint
func.func @test_compute_type_hint(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: darwinn.compute_type_hint
  %0 = darwinn.copy_op %arg0 {darwinn.compute_type_hint = #darwinn.compute_type_hint<dense>} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: image_format
func.func @test_image_format(%arg0: tensor<1x8x8x4xf32>) -> tensor<1x8x8x4xf32> {
  // CHECK: darwinn.image_format
  %0 = darwinn.copy_op %arg0 {darwinn.image_format = #darwinn.image_format<nhwc>} : (tensor<1x8x8x4xf32>) -> tensor<1x8x8x4xf32>
  return %0 : tensor<1x8x8x4xf32>
}

// -----
// CHECK-LABEL: mem_space
func.func @test_mem_space(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: darwinn.mem_space
  %0 = darwinn.copy_op %arg0 {darwinn.mem_space = #darwinn.mem_space<low>} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: mem_space_ssram
func.func @test_mem_space_ssram(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: darwinn.mem_space
  %0 = darwinn.copy_op %arg0 {darwinn.mem_space = #darwinn.mem_space<ssram>} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: kernel_level
func.func @test_kernel_level(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: darwinn.kernel_level
  %0 = darwinn.copy_op %arg0 {darwinn.kernel_level = #darwinn.kernel_level<low>} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: compression_mode
func.func @test_compression_mode(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: darwinn.compression_mode
  %0 = darwinn.copy_op %arg0 {darwinn.compression_mode = #darwinn.compression_mode<*>} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: compute_op_options
func.func @test_compute_op_options(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: darwinn.compute_op_options
  %0 = darwinn.copy_op %arg0 {darwinn.compute_op_options = #darwinn.compute_op_options<*>} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: custom_tiling_options
func.func @test_custom_tiling_options(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: darwinn.custom_tiling_options
  %0 = darwinn.copy_op %arg0 {darwinn.custom_tiling_options = #darwinn.custom_tiling_options<*>} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dtc_info
func.func @test_dtc_info(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: darwinn.dtc_info
  %0 = darwinn.copy_op %arg0 {darwinn.dtc_info = #darwinn.dtc_info<*>} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: filtered_indices_dispatch_mode
func.func @test_filtered_indices_dispatch_mode(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: darwinn.filtered_indices_dispatch_mode
  %0 = darwinn.copy_op %arg0 {darwinn.filtered_indices_dispatch_mode = #darwinn.filtered_indices_dispatch_mode<*>} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: join_attributes
func.func @test_join_attributes(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: darwinn.join_attributes
  %0 = darwinn.copy_op %arg0 {darwinn.join_attributes = #darwinn.join_attributes<*>} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: local_copy_attributes
func.func @test_local_copy_attributes(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: darwinn.local_copy_attributes
  %0 = darwinn.copy_op %arg0 {darwinn.local_copy_attributes = #darwinn.local_copy_attributes<*>} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: packed_index_options
func.func @test_packed_index_options(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: darwinn.packed_index_options
  %0 = darwinn.copy_op %arg0 {darwinn.packed_index_options = #darwinn.packed_index_options<*>} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: resampler_options
func.func @test_resampler_options(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: darwinn.resampler_options
  %0 = darwinn.copy_op %arg0 {darwinn.resampler_options = #darwinn.resampler_options<*>} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: ring_to_tile_options
func.func @test_ring_to_tile_options(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: darwinn.ring_to_tile_options
  %0 = darwinn.copy_op %arg0 {darwinn.ring_to_tile_options = #darwinn.ring_to_tile_options<*>} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: vex_info
func.func @test_vex_info(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: darwinn.vex_info
  %0 = darwinn.copy_op %arg0 {darwinn.vex_info = #darwinn.vex_info<*>} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: vrgkh_operation_mode
func.func @test_vrgkh_operation_mode(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: darwinn.vrgkh_operation_mode
  %0 = darwinn.copy_op %arg0 {darwinn.vrgkh_operation_mode = #darwinn.vrgkh_operation_mode<*>} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
