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

// RUN: mlir-opt %s --verify-each | FileCheck %s

// -----
// CHECK-LABEL: abs
func.func @test_abs(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.abs
  %0 = "dwc.abs"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: arange
func.func @test_arange(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.arange
  %0 = "dwc.arange"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: attention
func.func @test_attention(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.attention
  %0 = "dwc.attention"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: attention_non_linear_function
func.func @test_attention_non_linear_function(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.attention_non_linear_function
  %0 = "dwc.attention_non_linear_function"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: batch_to_space
func.func @test_batch_to_space(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.batch_to_space
  %0 = "dwc.batch_to_space"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: bit_select
func.func @test_bit_select(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.bit_select
  %0 = "dwc.bit_select"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: broadcast
func.func @test_broadcast(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.broadcast
  %0 = "dwc.broadcast"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: cbrt
func.func @test_cbrt(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.cbrt
  %0 = "dwc.cbrt"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: clamp
func.func @test_clamp(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.clamp
  %0 = "dwc.clamp"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: collective_permute
func.func @test_collective_permute(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.collective_permute
  %0 = "dwc.collective_permute"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: composite
func.func @test_composite(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.composite
  %0 = "dwc.composite"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: const_none
func.func @test_const_none(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.const_none
  %0 = "dwc.const_none"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: convolution_sub_channel
func.func @test_convolution_sub_channel(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.convolution_sub_channel
  %0 = "dwc.convolution_sub_channel"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: cost_volume
func.func @test_cost_volume(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.cost_volume
  %0 = "dwc.cost_volume"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: count_leading_zeros
func.func @test_count_leading_zeros(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.count_leading_zeros
  %0 = "dwc.count_leading_zeros"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: custom_compute
func.func @test_custom_compute(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.custom_compute
  %0 = "dwc.custom_compute"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: declare_tensor
func.func @test_declare_tensor(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.declare_tensor
  %0 = "dwc.declare_tensor"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: depth_to_space
func.func @test_depth_to_space(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.depth_to_space
  %0 = "dwc.depth_to_space"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: depthwise_convolution
func.func @test_depthwise_convolution(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.depthwise_convolution
  %0 = "dwc.depthwise_convolution"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: device_launch
func.func @test_device_launch(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.device_launch
  %0 = "dwc.device_launch"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dma_op_gather
func.func @test_dma_op_gather(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.dma_op_gather
  %0 = "dwc.dma_op_gather"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dynamic_broadcast
func.func @test_dynamic_broadcast(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.dynamic_broadcast
  %0 = "dwc.dynamic_broadcast"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dynamic_quantize
func.func @test_dynamic_quantize(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.dynamic_quantize
  %0 = "dwc.dynamic_quantize"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: ensure_shape
func.func @test_ensure_shape(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.ensure_shape
  %0 = "dwc.ensure_shape"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: extern_call
func.func @test_extern_call(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.extern_call
  %0 = "dwc.extern_call"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: fast_walsh_hadamard_transform
func.func @test_fast_walsh_hadamard_transform(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.fast_walsh_hadamard_transform
  %0 = "dwc.fast_walsh_hadamard_transform"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: fully_connected_sub_byte_param
func.func @test_fully_connected_sub_byte_param(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.fully_connected_sub_byte_param
  %0 = "dwc.fully_connected_sub_byte_param"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: fully_connected_sub_channel
func.func @test_fully_connected_sub_channel(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.fully_connected_sub_channel
  %0 = "dwc.fully_connected_sub_channel"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: functional_if
func.func @test_functional_if(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.functional_if
  %0 = "dwc.functional_if"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: functional_while
func.func @test_functional_while(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.functional_while
  %0 = "dwc.functional_while"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: gather_operation
func.func @test_gather_operation(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.gather_operation
  %0 = "dwc.gather_operation"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: generic_move
func.func @test_generic_move(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.generic_move
  %0 = "dwc.generic_move"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: generic_scatter
func.func @test_generic_scatter(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.generic_scatter
  %0 = "dwc.generic_scatter"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: hib_gather
func.func @test_hib_gather(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.hib_gather
  %0 = "dwc.hib_gather"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: hib_gather_filter
func.func @test_hib_gather_filter(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.hib_gather_filter
  %0 = "dwc.hib_gather_filter"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: hosted_tensor
func.func @test_hosted_tensor(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.hosted_tensor
  %0 = "dwc.hosted_tensor"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: image_interpolation
func.func @test_image_interpolation(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.image_interpolation
  %0 = "dwc.image_interpolation"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: index_unpool
func.func @test_index_unpool(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.index_unpool
  %0 = "dwc.index_unpool"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: interleave
func.func @test_interleave(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.interleave
  %0 = "dwc.interleave"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: is_finite
func.func @test_is_finite(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.is_finite
  %0 = "dwc.is_finite"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: mask_indices
func.func @test_mask_indices(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.mask_indices
  %0 = "dwc.mask_indices"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: matrix_multiply_sub_channel
func.func @test_matrix_multiply_sub_channel(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.matrix_multiply_sub_channel
  %0 = "dwc.matrix_multiply_sub_channel"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: multinomial
func.func @test_multinomial(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.multinomial
  %0 = "dwc.multinomial"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: negate
func.func @test_negate(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.negate
  %0 = "dwc.negate"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: normalization
func.func @test_normalization(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.normalization
  %0 = "dwc.normalization"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: one_hot_tpu
func.func @test_one_hot_tpu(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.one_hot_tpu
  %0 = "dwc.one_hot_tpu"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: pack_bits
func.func @test_pack_bits(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.pack_bits
  %0 = "dwc.pack_bits"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: pooling
func.func @test_pooling(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.pooling
  %0 = "dwc.pooling"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: probe_subtensor
func.func @test_probe_subtensor(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.probe_subtensor
  %0 = "dwc.probe_subtensor"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: reduce_precision
func.func @test_reduce_precision(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.reduce_precision
  %0 = "dwc.reduce_precision"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: reduce_window
func.func @test_reduce_window(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.reduce_window
  %0 = "dwc.reduce_window"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: resampler
func.func @test_resampler(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.resampler
  %0 = "dwc.resampler"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: reverse
func.func @test_reverse(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.reverse
  %0 = "dwc.reverse"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: rkhy_add_pool
func.func @test_rkhy_add_pool(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.rkhy_add_pool
  %0 = "dwc.rkhy_add_pool"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: rkhy_conv_d2s
func.func @test_rkhy_conv_d2s(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.rkhy_conv_d2s
  %0 = "dwc.rkhy_conv_d2s"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: rkhy_custom_padding
func.func @test_rkhy_custom_padding(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.rkhy_custom_padding
  %0 = "dwc.rkhy_custom_padding"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: rkhy_fused_conv
func.func @test_rkhy_fused_conv(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.rkhy_fused_conv
  %0 = "dwc.rkhy_fused_conv"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: rkhy_fused_norm
func.func @test_rkhy_fused_norm(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.rkhy_fused_norm
  %0 = "dwc.rkhy_fused_norm"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: roll
func.func @test_roll(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.roll
  %0 = "dwc.roll"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: scalar_core_constant
func.func @test_scalar_core_constant(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.scalar_core_constant
  %0 = "dwc.scalar_core_constant"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: scatter_operation
func.func @test_scatter_operation(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.scatter_operation
  %0 = "dwc.scatter_operation"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: shift_left
func.func @test_shift_left(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.shift_left
  %0 = "dwc.shift_left"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: shift_right_logical
func.func @test_shift_right_logical(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.shift_right_logical
  %0 = "dwc.shift_right_logical"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: space_to_batch
func.func @test_space_to_batch(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.space_to_batch
  %0 = "dwc.space_to_batch"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: space_to_depth
func.func @test_space_to_depth(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.space_to_depth
  %0 = "dwc.space_to_depth"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: sparse_fully_connected_sub_byte_param
func.func @test_sparse_fully_connected_sub_byte_param(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.sparse_fully_connected_sub_byte_param
  %0 = "dwc.sparse_fully_connected_sub_byte_param"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: sparse_parameter
func.func @test_sparse_parameter(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.sparse_parameter
  %0 = "dwc.sparse_parameter"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: spill
func.func @test_spill(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.spill
  %0 = "dwc.spill"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: statistical_top_k
func.func @test_statistical_top_k(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.statistical_top_k
  %0 = "dwc.statistical_top_k"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: tensor_ls_gather
func.func @test_tensor_ls_gather(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.tensor_ls_gather
  %0 = "dwc.tensor_ls_gather"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: tensor_ls_scatter
func.func @test_tensor_ls_scatter(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.tensor_ls_scatter
  %0 = "dwc.tensor_ls_scatter"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: tensor_ls_scatter_operation
func.func @test_tensor_ls_scatter_operation(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.tensor_ls_scatter_operation
  %0 = "dwc.tensor_ls_scatter_operation"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: tensor_op_gather
func.func @test_tensor_op_gather(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.tensor_op_gather
  %0 = "dwc.tensor_op_gather"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: top_k
func.func @test_top_k(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.top_k
  %0 = "dwc.top_k"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: transposed_convolution_sub_channel
func.func @test_transposed_convolution_sub_channel(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.transposed_convolution_sub_channel
  %0 = "dwc.transposed_convolution_sub_channel"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: truncate_floats
func.func @test_truncate_floats(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.truncate_floats
  %0 = "dwc.truncate_floats"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: uniform_random_number_generation
func.func @test_uniform_random_number_generation(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.uniform_random_number_generation
  %0 = "dwc.uniform_random_number_generation"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: atan2
func.func @test_atan2(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.atan2
  %0 = "dwc.atan2"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: gather_nd
func.func @test_gather_nd(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.gather_nd
  %0 = "dwc.gather_nd"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: annotate_materialize_policy
func.func @test_annotate_materialize_policy(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.annotate_materialize_policy
  %0 = "dwc.annotate_materialize_policy"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
