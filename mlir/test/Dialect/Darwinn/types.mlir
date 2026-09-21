// RUN: mlir-opt %s --verify-each | mlir-opt | FileCheck %s
// RUN: mlir-opt %s --mlir-print-op-generic | mlir-opt | FileCheck %s

// -----
// CHECK-LABEL: device_type
func.func @test_device_type(%arg0: tensor<4x!darwinn.device_type>) -> tensor<4x!darwinn.device_type> {
  // CHECK: darwinn.device_type
  return %arg0 : tensor<4x!darwinn.device_type>
}

// -----
// CHECK-LABEL: sparsity_type
func.func @test_sparsity_type(%arg0: tensor<4x!darwinn.sparsity_type>) -> tensor<4x!darwinn.sparsity_type> {
  // CHECK: darwinn.sparsity_type
  return %arg0 : tensor<4x!darwinn.sparsity_type>
}

// -----
// CHECK-LABEL: packed
func.func @test_packed(%arg0: tensor<4x!darwinn.packed<48>>) -> tensor<4x!darwinn.packed<48>> {
  // CHECK: darwinn.packed
  return %arg0 : tensor<4x!darwinn.packed<48>>
}
