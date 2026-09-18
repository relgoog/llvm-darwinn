// RUN: mlir-opt %s --verify-each | mlir-opt | FileCheck %s
// RUN: mlir-opt %s --mlir-print-op-generic | mlir-opt | FileCheck %s

// -----
// CHECK-LABEL: aux_tensor
func.func @test_aux_tensor(%arg0: tensor<4x!darwinn.aux_tensor_type>) -> tensor<4x!darwinn.aux_tensor_type> {
  // CHECK: darwinn.aux_tensor_type
  %0 = darwinn.copy_op %arg0 : (tensor<4x!darwinn.aux_tensor_type>) -> tensor<4x!darwinn.aux_tensor_type>
  return %0 : tensor<4x!darwinn.aux_tensor_type>
}

// -----
// CHECK-LABEL: const_type
func.func @test_const_type(%arg0: tensor<4x!darwinn.const_type>) -> tensor<4x!darwinn.const_type> {
  // CHECK: darwinn.const_type
  %0 = darwinn.copy_op %arg0 : (tensor<4x!darwinn.const_type>) -> tensor<4x!darwinn.const_type>
  return %0 : tensor<4x!darwinn.const_type>
}

// -----
// CHECK-LABEL: device_type
func.func @test_device_type(%arg0: tensor<4x!darwinn.device_type>) -> tensor<4x!darwinn.device_type> {
  // CHECK: darwinn.device_type
  %0 = darwinn.copy_op %arg0 : (tensor<4x!darwinn.device_type>) -> tensor<4x!darwinn.device_type>
  return %0 : tensor<4x!darwinn.device_type>
}

// -----
// CHECK-LABEL: sparsity_type
func.func @test_sparsity_type(%arg0: tensor<4x!darwinn.sparsity_type>) -> tensor<4x!darwinn.sparsity_type> {
  // CHECK: darwinn.sparsity_type
  %0 = darwinn.copy_op %arg0 : (tensor<4x!darwinn.sparsity_type>) -> tensor<4x!darwinn.sparsity_type>
  return %0 : tensor<4x!darwinn.sparsity_type>
}
