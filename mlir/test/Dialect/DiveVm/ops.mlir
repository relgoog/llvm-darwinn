// RUN: mlir-opt %s --verify-each | mlir-opt | FileCheck %s
// RUN: mlir-opt %s --mlir-print-op-generic | mlir-opt | FileCheck %s


// -----
// CHECK-LABEL: copy
func.func @test_copy(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  // CHECK: dive_vm.copy
  %0 = dive_vm.copy %arg0 : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
// CHECK-LABEL: add
func.func @test_add(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  // CHECK: dive_vm.add
  %0 = dive_vm.add %arg0, %arg1 : (tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
// CHECK-LABEL: min
func.func @test_min(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  // CHECK: dive_vm.min
  %0 = dive_vm.min %arg0, %arg1 : (tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
// CHECK-LABEL: div
func.func @test_div(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  // CHECK: dive_vm.div
  %0 = dive_vm.div %arg0, %arg1 : (tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
// CHECK-LABEL: address_of_parameter
func.func @test_address_of_parameter() -> tensor<4xi32> {
  // CHECK: dive_vm.address_of_parameter
  %0 = dive_vm.address_of_parameter : () -> tensor<4xi32>
  return %0 : tensor<4xi32>
}

// -----
// CHECK-LABEL: address_of_activation
func.func @test_address_of_activation() -> tensor<4xi32> {
  // CHECK: dive_vm.address_of_activation
  %0 = dive_vm.address_of_activation : () -> tensor<4xi32>
  return %0 : tensor<4xi32>
}

// -----
// CHECK-LABEL: address_of_input_activation
func.func @test_address_of_input_activation() -> tensor<4xi32> {
  // CHECK: dive_vm.address_of_input_activation
  %0 = dive_vm.address_of_input_activation : () -> tensor<4xi32>
  return %0 : tensor<4xi32>
}

// -----
// CHECK-LABEL: address_of_output_activation
func.func @test_address_of_output_activation() -> tensor<4xi32> {
  // CHECK: dive_vm.address_of_output_activation
  %0 = dive_vm.address_of_output_activation : () -> tensor<4xi32>
  return %0 : tensor<4xi32>
}

// -----
// CHECK-LABEL: address_of_parameter_region
func.func @test_address_of_parameter_region() -> tensor<4xi32> {
  // CHECK: dive_vm.address_of_parameter_region
  %0 = dive_vm.address_of_parameter_region : () -> tensor<4xi32>
  return %0 : tensor<4xi32>
}

// -----
// CHECK-LABEL: address_of_scratch
func.func @test_address_of_scratch() -> tensor<4xi32> {
  // CHECK: dive_vm.address_of_scratch
  %0 = dive_vm.address_of_scratch : () -> tensor<4xi32>
  return %0 : tensor<4xi32>
}

// -----
// CHECK-LABEL: annotate_materialize_policy
func.func @test_annotate_materialize_policy(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  // CHECK: edgetpu.annotate_materialize_policy
  %0 = edgetpu.annotate_materialize_policy %arg0 : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
// CHECK-LABEL: convolution_sub_channel
func.func @test_convolution_sub_channel(%arg0: tensor<1x4x4x4xf32>, %arg1: tensor<8x1x1x4xf32>) -> tensor<1x4x4x8xf32> {
  // CHECK: edgetpu.convolution_sub_channel
  %0 = edgetpu.convolution_sub_channel %arg0, %arg1 : (tensor<1x4x4x4xf32>, tensor<8x1x1x4xf32>) -> tensor<1x4x4x8xf32>
  return %0 : tensor<1x4x4x8xf32>
}

// -----
// CHECK-LABEL: matrix_multiply_sub_channel
func.func @test_matrix_multiply_sub_channel(%arg0: tensor<1x14x19xf32>, %arg1: tensor<1x19x28xf32>) -> tensor<1x14x28xf32> {
  // CHECK: edgetpu.matrix_multiply_sub_channel
  %0 = edgetpu.matrix_multiply_sub_channel %arg0, %arg1 : (tensor<1x14x19xf32>, tensor<1x19x28xf32>) -> tensor<1x14x28xf32>
  return %0 : tensor<1x14x28xf32>
}

// -----
// CHECK-LABEL: fully_connected_sub_channel
func.func @test_fully_connected_sub_channel(%arg0: tensor<1x19xf32>, %arg1: tensor<28x19xf32>) -> tensor<1x28xf32> {
  // CHECK: edgetpu.fully_connected_sub_channel
  %0 = edgetpu.fully_connected_sub_channel %arg0, %arg1 : (tensor<1x19xf32>, tensor<28x19xf32>) -> tensor<1x28xf32>
  return %0 : tensor<1x28xf32>
}

// -----
// CHECK-LABEL: transposed_convolution_sub_channel
func.func @test_transposed_convolution_sub_channel(%arg0: tensor<1x4x4x4xf32>, %arg1: tensor<8x1x1x4xf32>) -> tensor<1x4x4x8xf32> {
  // CHECK: edgetpu.transposed_convolution_sub_channel
  %0 = edgetpu.transposed_convolution_sub_channel %arg0, %arg1 : (tensor<1x4x4x4xf32>, tensor<8x1x1x4xf32>) -> tensor<1x4x4x8xf32>
  return %0 : tensor<1x4x4x8xf32>
}

// -----
// CHECK-LABEL: attention_v1
func.func @test_attention_v1(%arg0: tensor<1x14x19xf32>) -> tensor<1x14x19xf32> {
  // CHECK: edgetpu.attention_v1
  %0 = edgetpu.attention_v1 %arg0 : (tensor<1x14x19xf32>) -> tensor<1x14x19xf32>
  return %0 : tensor<1x14x19xf32>
}
