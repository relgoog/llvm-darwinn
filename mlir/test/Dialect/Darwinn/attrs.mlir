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
