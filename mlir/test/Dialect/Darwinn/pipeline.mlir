// RUN: mlir-opt --split-input-file --dwc-legalize --dwc-lower-hlops --convert-dive-vm-to-llvm --convert-tpu-offload-to-llvm --dive-program-tpu %s | FileCheck %s

// -----
// CHECK-LABEL: @test_pipeline_convert
func.func @test_pipeline_convert(%arg0: tensor<4xf32>) -> tensor<4xbf16> {
  // CHECK: darwinn.convert
  %0 = darwinn.convert %arg0 : (tensor<4xf32>) -> tensor<4xbf16>
  return %0 : tensor<4xbf16>
}

// -----
// CHECK-LABEL: @test_pipeline_convolution
func.func @test_pipeline_convolution(%arg0: tensor<1x8x8x4xf32>, %arg1: tensor<3x3x4x8xf32>) -> tensor<1x8x8x8xf32> {
  // CHECK: darwinn.convolution
  %0 = darwinn.convolution %arg0, %arg1 : (tensor<1x8x8x4xf32>, tensor<3x3x4x8xf32>) -> tensor<1x8x8x8xf32>
  return %0 : tensor<1x8x8x8xf32>
}

// -----
// CHECK-LABEL: @test_pipeline_copy
func.func @test_pipeline_copy(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK: darwinn.copy_op
  %0 = darwinn.copy_op %arg0 : (tensor<8x8xf32>) -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----
// CHECK-LABEL: @test_pipeline_fill
func.func @test_pipeline_fill(%arg0: f32) -> tensor<4x4xf32> {
  // CHECK: darwinn.fill
  %0 = darwinn.fill %arg0 : (f32) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}
