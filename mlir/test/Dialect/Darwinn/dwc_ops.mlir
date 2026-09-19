// RUN: mlir-opt %s --verify-each | FileCheck %s

// -----
// CHECK-LABEL: add
func.func @test_add(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.add
  %0 = "dwc.add"(%arg0) {activation_function = "x"} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: atan
func.func @test_atan(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.atan
  %0 = "dwc.atan"(%arg0) {} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: batch_matrix_nms
func.func @test_batch_matrix_nms(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.batch_matrix_nms
  %0 = "dwc.batch_matrix_nms"(%arg0) {max_output_size = "x", score_threshold = "x", sigma = "x", suppress_top_k = "x"} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: bitcast
func.func @test_bitcast(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.bitcast
  %0 = "dwc.bitcast"(%arg0) {output_element_type = "x"} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: cast
func.func @test_cast(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.cast
  %0 = "dwc.cast"(%arg0) {} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: ceil
func.func @test_ceil(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.ceil
  %0 = "dwc.ceil"(%arg0) {} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: classifier
func.func @test_classifier(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.classifier
  %0 = "dwc.classifier"(%arg0) {axis = "x", beta = "x", op_type = "x"} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: compare
func.func @test_compare(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.compare
  %0 = "dwc.compare"(%arg0) {activation_function = "x", compare_type = "x"} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: concatenation
func.func @test_concatenation(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.concatenation
  %0 = "dwc.concatenation"(%arg0) {mode = "x"} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: const
func.func @test_const(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.const
  %0 = "dwc.const"(%arg0) {value = "x"} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: convolution
func.func @test_convolution(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.convolution
  %0 = "dwc.convolution"(%arg0) {activation_function = "x", cell_operation = "x", pad = "x", x_dilation_rate = "x", x_stride = "x", y_dilation_rate = "x", y_stride = "x"} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: convolution_v2
func.func @test_convolution_v2(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.convolution_v2
  %0 = "dwc.convolution_v2"(%arg0) {activation_function = "x", cell_operation = "x", pad = "x", x_dilation_rate = "x", x_stride = "x", y_dilation_rate = "x", y_stride = "x"} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: cos
func.func @test_cos(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.cos
  %0 = "dwc.cos"(%arg0) {} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: cumulative
func.func @test_cumulative(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.cumulative
  %0 = "dwc.cumulative"(%arg0) {axis = "x", exclusive = "x", op_type = "x"} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: cwise
func.func @test_cwise(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.cwise
  %0 = "dwc.cwise"(%arg0, %arg1) {activation_function = "x", op_type = "x"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: depthwise_convolution_v2
func.func @test_depthwise_convolution_v2(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.depthwise_convolution_v2
  %0 = "dwc.depthwise_convolution_v2"(%arg0) {activation_function = "x", cell_operation = "x", depth_multiplier = "x", pad = "x", x_dilation_rate = "x", x_stride = "x", y_dilation_rate = "x", y_stride = "x"} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: divide
func.func @test_divide(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.divide
  %0 = "dwc.divide"(%arg0) {activation_function = "x"} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dynamic_slice
func.func @test_dynamic_slice(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.dynamic_slice
  %0 = "dwc.dynamic_slice"(%arg0) {mode = "x", read_location = "x", slice_size = "x"} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dynamic_update_slice
func.func @test_dynamic_update_slice(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.dynamic_update_slice
  %0 = "dwc.dynamic_update_slice"(%arg0) {mode = "x", write_location = "x"} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: erf
func.func @test_erf(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.erf
  %0 = "dwc.erf"(%arg0) {} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: exp
func.func @test_exp(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.exp
  %0 = "dwc.exp"(%arg0) {} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: expm1
func.func @test_expm1(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.expm1
  %0 = "dwc.expm1"(%arg0) {} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: floor
func.func @test_floor(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.floor
  %0 = "dwc.floor"(%arg0) {} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: floor_div
func.func @test_floor_div(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.floor_div
  %0 = "dwc.floor_div"(%arg0, %arg1) {} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: fully_connected
func.func @test_fully_connected(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.fully_connected
  %0 = "dwc.fully_connected"(%arg0) {activation_function = "x", cell_operation = "x"} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: gather
func.func @test_gather(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.gather
  %0 = "dwc.gather"(%arg0) {axis = "x", batch_dims = "x"} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: generic_compute
func.func @test_generic_compute(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.generic_compute
  %0 = "dwc.generic_compute"(%arg0) {activation_function = "x", indexing_maps = "x", linear_function = "x"} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: generic_constant
func.func @test_generic_constant(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.generic_constant
  %0 = "dwc.generic_constant"(%arg0) {value = "x"} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: generic_conv
func.func @test_generic_conv(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.generic_conv
  %0 = "dwc.generic_conv"(%arg0) {activation_function = "x", batch_group_count = "x", feature_group_count = "x", input_dilation = "x", padding_amount = "x", param_dilation = "x", param_reversal = "x", stride = "x"} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: generic_dot
func.func @test_generic_dot(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.generic_dot
  %0 = "dwc.generic_dot"(%arg0) {activation_function = "x", batch_dim_count = "x", contracting_dim_count = "x"} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: log1p
func.func @test_log1p(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.log1p
  %0 = "dwc.log1p"(%arg0) {} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: logistic
func.func @test_logistic(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.logistic
  %0 = "dwc.logistic"(%arg0) {} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: matrix_multiply
func.func @test_matrix_multiply(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.matrix_multiply
  %0 = "dwc.matrix_multiply"(%arg0) {activation_function = "x", transpose_rhs = "x"} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: maximum
func.func @test_maximum(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.maximum
  %0 = "dwc.maximum"(%arg0) {activation_function = "x"} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: minimum
func.func @test_minimum(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.minimum
  %0 = "dwc.minimum"(%arg0) {activation_function = "x"} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: multiply
func.func @test_multiply(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.multiply
  %0 = "dwc.multiply"(%arg0) {activation_function = "x"} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: not
func.func @test_not(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.not
  %0 = "dwc.not"(%arg0) {} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: one_hot
func.func @test_one_hot(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.one_hot
  %0 = "dwc.one_hot"(%arg0) {axis = "x"} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: padding
func.func @test_padding(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.padding
  %0 = "dwc.padding"(%arg0) {dimension = "x", padding_value = "x", post_padding = "x", pre_padding = "x"} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: pop_count
func.func @test_pop_count(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.pop_count
  %0 = "dwc.pop_count"(%arg0) {} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: pow
func.func @test_pow(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.pow
  %0 = "dwc.pow"(%arg0, %arg1) {} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: pseudo_split
func.func @test_pseudo_split(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.pseudo_split
  %0 = "dwc.pseudo_split"(%arg0, %arg1) {num_splits = "x"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: reduction
func.func @test_reduction(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.reduction
  %0 = "dwc.reduction"(%arg0) {activation_function = "x", dimensions = "x", op_type = "x"} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: remainder
func.func @test_remainder(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.remainder
  %0 = "dwc.remainder"(%arg0) {activation_function = "x"} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: rescaling
func.func @test_rescaling(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.rescaling
  %0 = "dwc.rescaling"(%arg0) {activation_function = "x", output_activation_per_z_out_scales = "x", per_z_out_scales_padding = "x"} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: reshape
func.func @test_reshape(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.reshape
  %0 = "dwc.reshape"(%arg0) {} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: round
func.func @test_round(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.round
  %0 = "dwc.round"(%arg0) {} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: round_nearest_afz
func.func @test_round_nearest_afz(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.round_nearest_afz
  %0 = "dwc.round_nearest_afz"(%arg0) {} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: rsqrt
func.func @test_rsqrt(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.rsqrt
  %0 = "dwc.rsqrt"(%arg0) {} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: scalar
func.func @test_scalar(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.scalar
  %0 = "dwc.scalar"(%arg0) {op_type = "x"} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: scatter_nd
func.func @test_scatter_nd(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.scatter_nd
  %0 = "dwc.scatter_nd"(%arg0) {shape = "x"} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: select
func.func @test_select(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.select
  %0 = "dwc.select"(%arg0, %arg1) {} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: sign
func.func @test_sign(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.sign
  %0 = "dwc.sign"(%arg0) {preserve_negative_zero = "x"} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: sin
func.func @test_sin(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.sin
  %0 = "dwc.sin"(%arg0) {} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: slice
func.func @test_slice(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.slice
  %0 = "dwc.slice"(%arg0) {in_begin = "x", in_size = "x", mode = "x"} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: sort
func.func @test_sort(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.sort
  %0 = "dwc.sort"(%arg0) {compare_tuple_projection = "x", compare_type = "x", dimension = "x", is_stable = "x"} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: sqrt
func.func @test_sqrt(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.sqrt
  %0 = "dwc.sqrt"(%arg0) {} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: subtract
func.func @test_subtract(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.subtract
  %0 = "dwc.subtract"(%arg0) {activation_function = "x"} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: tan
func.func @test_tan(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.tan
  %0 = "dwc.tan"(%arg0) {} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: tanh
func.func @test_tanh(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.tanh
  %0 = "dwc.tanh"(%arg0) {} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: transpose
func.func @test_transpose(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.transpose
  %0 = "dwc.transpose"(%arg0) {permutation = "x"} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: transposed_convolution
func.func @test_transposed_convolution(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.transposed_convolution
  %0 = "dwc.transposed_convolution"(%arg0) {activation_function = "x", cell_operation = "x", pad = "x", x_dilation_rate = "x", x_out_dim = "x", x_stride = "x", y_dilation_rate = "x", y_out_dim = "x", y_stride = "x"} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: unsorted_segment_reduce
func.func @test_unsorted_segment_reduce(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.unsorted_segment_reduce
  %0 = "dwc.unsorted_segment_reduce"(%arg0) {num_segments = "x", op_type = "x"} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
