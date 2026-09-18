// RUN: mlir-opt %s --verify-each | mlir-opt | FileCheck %s
// RUN: mlir-opt %s --mlir-print-op-generic | mlir-opt | FileCheck %s

// -----
// CHECK-LABEL: convert
func.func @test_convert(%arg0: tensor<4xf32>) -> tensor<4xbf16> {
  // CHECK: darwinn.convert
  %0 = darwinn.convert %arg0 : (tensor<4xf32>) -> tensor<4xbf16>
  return %0 : tensor<4xbf16>
}

// -----
// CHECK-LABEL: convolution
func.func @test_convolution(%arg0: tensor<1x8x8x4xf32>, %arg1: tensor<3x3x4x8xf32>) -> tensor<1x8x8x8xf32> {
  // CHECK: darwinn.convolution
  %0 = darwinn.convolution %arg0, %arg1 : (tensor<1x8x8x4xf32>, tensor<3x3x4x8xf32>) -> tensor<1x8x8x8xf32>
  return %0 : tensor<1x8x8x8xf32>
}

// -----
// CHECK-LABEL: copy_op
func.func @test_copy_op(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK: darwinn.copy_op
  %0 = darwinn.copy_op %arg0 : (tensor<8x8xf32>) -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----
// CHECK-LABEL: fence
func.func @test_fence() {
  // CHECK: darwinn.fence
  darwinn.fence : () -> ()
  return
}

// -----
// CHECK-LABEL: fill
func.func @test_fill(%arg0: f32) -> tensor<4x4xf32> {
  // CHECK: darwinn.fill
  %0 = darwinn.fill %arg0 : (f32) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// -----
// CHECK-LABEL: gather
func.func @test_gather(%arg0: tensor<16x8xf32>, %arg1: tensor<4xi32>) -> tensor<4x8xf32> {
  // CHECK: darwinn.gather
  %0 = darwinn.gather %arg0, %arg1 : (tensor<16x8xf32>, tensor<4xi32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// -----
// CHECK-LABEL: hib_gather
func.func @test_hib_gather(%arg0: tensor<16x8xf32>, %arg1: tensor<4xi32>) -> tensor<4x8xf32> {
  // CHECK: darwinn.hib_gather
  %0 = darwinn.hib_gather %arg0, %arg1 : (tensor<16x8xf32>, tensor<4xi32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// -----
// CHECK-LABEL: gather_copy
func.func @test_gather_copy(%arg0: tensor<16x8xf32>, %arg1: tensor<4xi32>) -> tensor<4x8xf32> {
  // CHECK: darwinn.gather_copy
  %0 = darwinn.gather_copy %arg0, %arg1 : (tensor<16x8xf32>, tensor<4xi32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// -----
// CHECK-LABEL: top_k_lowering_target
func.func @test_top_k_lowering_target(%arg0: tensor<16x8xf32>, %arg1: tensor<4xi32>) -> tensor<4x8xf32> {
  // CHECK: darwinn.gather
  %0 = darwinn.gather %arg0, %arg1 : (tensor<16x8xf32>, tensor<4xi32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// -----
// CHECK-LABEL: scatter
func.func @test_scatter(%arg0: tensor<16x8xf32>, %arg1: tensor<4xi32>, %arg2: tensor<4x8xf32>) -> tensor<16x8xf32> {
  // CHECK: darwinn.scatter
  %0 = darwinn.scatter %arg0, %arg1, %arg2 : (tensor<16x8xf32>, tensor<4xi32>, tensor<4x8xf32>) -> tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}
