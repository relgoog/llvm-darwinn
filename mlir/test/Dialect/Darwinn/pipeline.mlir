// RUN: mlir-opt --split-input-file --dwc-lower-hlops --convert-dive-vm-to-llvm --convert-tpu-offload-to-llvm --dive-program-tpu %s | FileCheck %s

// -----
// CHECK-LABEL: func.func @test_pipeline_convert(
func.func @test_pipeline_convert(%arg0: tensor<4xf16>) -> tensor<4xf32> {
  // CHECK: arith.extf {{.*}} : f16 to f32
  %0 = dwc.cast %arg0 : (tensor<4xf16>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: func.func @test_pipeline_convolution(
func.func @test_pipeline_convolution(%arg0: tensor<1x8x8x4xf32>, %arg1: tensor<3x3x4x8xf32>) -> tensor<1x6x6x8xf32> {
  // CHECK: linalg.conv_2d_nhwc_hwcf
  %0 = dwc.convolution %arg0, %arg1 {activation_function = #dwc.activation_function<NONE>, cell_operation = #dwc.cell_operation<MAC>, pad = #dwc.padding<NONE>, x_dilation_rate = 1 : i64, x_stride = 1 : i64, y_dilation_rate = 1 : i64, y_stride = 1 : i64} : (tensor<1x8x8x4xf32>, tensor<3x3x4x8xf32>) -> tensor<1x6x6x8xf32>
  return %0 : tensor<1x6x6x8xf32>
}

// -----
// CHECK-LABEL: func.func @test_pipeline_copy(
func.func @test_pipeline_copy(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK: return %arg0 : tensor<8x8xf32>
  %0 = darwinn.copy_op %arg0 : (tensor<8x8xf32>) -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----
// CHECK-LABEL: func.func @test_pipeline_fill(
func.func @test_pipeline_fill(%arg0: f32) -> tensor<4x4xf32> {
  // CHECK: linalg.fill ins(%arg0 : f32)
  %0 = darwinn.fill %arg0 : (f32) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}
