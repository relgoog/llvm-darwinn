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
