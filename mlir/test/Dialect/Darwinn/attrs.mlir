// RUN: mlir-opt %s --verify-each | mlir-opt | FileCheck %s
// RUN: mlir-opt %s --mlir-print-op-generic | mlir-opt | FileCheck %s

// -----
// CHECK-LABEL: func.func @test_function_symmetry(
func.func @test_function_symmetry(%arg0: tensor<4xf32>) -> tensor<4xf32> attributes {darwinn.function_symmetry = #darwinn.function_symmetry<EVEN>} {
  // CHECK: #darwinn.function_symmetry<EVEN>
  return %arg0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: func.func @test_materialize_policy(
func.func @test_materialize_policy(%arg0: tensor<4xf32>) -> tensor<4xf32> attributes {darwinn.materialize_policy = #darwinn.materialize_policy<on_chip>} {
  // CHECK: #darwinn.materialize_policy<on_chip>
  return %arg0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: func.func @test_cost_hint_alias(
func.func @test_cost_hint_alias(%arg0: tensor<4xf32>) -> tensor<4xf32> attributes {darwinn.cost_hint = #darwinn.cost_hint<low>} {
  // CHECK: #darwinn.cost_hint<low>
  return %arg0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: func.func @test_cost_hint(
func.func @test_cost_hint(%arg0: tensor<4xf32>) -> tensor<4xf32> attributes {darwinn.cost_hint = #darwinn.cost_hint<low>} {
  // CHECK: #darwinn.cost_hint<low>
  return %arg0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: func.func @test_compute_lowering_hint(
func.func @test_compute_lowering_hint(%arg0: tensor<4xf32>) -> tensor<4xf32> attributes {darwinn.compute_lowering_hint = #darwinn.compute_lowering_hint<direct>} {
  // CHECK: #darwinn.compute_lowering_hint<direct>
  return %arg0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: func.func @test_compute_type_hint(
func.func @test_compute_type_hint(%arg0: tensor<4xf32>) -> tensor<4xf32> attributes {darwinn.compute_type_hint = #darwinn.compute_type_hint<dense>} {
  // CHECK: #darwinn.compute_type_hint<dense>
  return %arg0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: func.func @test_image_format(
func.func @test_image_format(%arg0: tensor<1x8x8x4xf32>) -> tensor<1x8x8x4xf32> attributes {darwinn.image_format = #darwinn.image_format<nhwc>} {
  // CHECK: #darwinn.image_format<nhwc>
  return %arg0 : tensor<1x8x8x4xf32>
}

// -----
// CHECK-LABEL: func.func @test_mem_space(
func.func @test_mem_space(%arg0: tensor<4xf32>) -> tensor<4xf32> attributes {darwinn.mem_space = #darwinn.mem_space<low>} {
  // CHECK: #darwinn.mem_space<low>
  return %arg0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: func.func @test_mem_space_ssram(
func.func @test_mem_space_ssram(%arg0: tensor<4xf32>) -> tensor<4xf32> attributes {darwinn.mem_space = #darwinn.mem_space<ssram>} {
  // CHECK: #darwinn.mem_space<ssram>
  return %arg0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: func.func @test_kernel_level(
func.func @test_kernel_level(%arg0: tensor<4xf32>) -> tensor<4xf32> attributes {darwinn.kernel_level = #darwinn.kernel_level<low>} {
  // CHECK: #darwinn.kernel_level<low>
  return %arg0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: func.func @test_compression_mode(
func.func @test_compression_mode(%arg0: tensor<4xf32>) -> tensor<4xf32> attributes {darwinn.compression_mode = #darwinn.compression_mode<*>} {
  // CHECK: #darwinn.compression_mode<*>
  return %arg0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: func.func @test_compute_op_options(
func.func @test_compute_op_options(%arg0: tensor<4xf32>) -> tensor<4xf32> attributes {darwinn.compute_op_options = #darwinn.compute_op_options<inner_operation = <UNARY>, scale_immediate = 1.000000e+00 : f32, bias_immediate = -0.000000e+00 : f32, clips = [0xFF800000 : f32, 0x7F800000 : f32], linear_function = <MAX>, replicate_reduce = 0 : i32, nlu_function = <LINEAR>, nlu_preprocess_type = <NONE>, bias_predicate = <NONE>, scale_predicate = <NONE>, try_cellgroups = false, nlu_e8m0_rounding = <NONE>, lowering_hint = <NONE>, compute_type_hint = <NONE>>} {
  // CHECK: #darwinn.compute_op_options<inner_operation = <UNARY>, scale_immediate = 1.000000e+00 : f32, bias_immediate = -0.000000e+00 : f32, clips = [0xFF800000 : f32, 0x7F800000 : f32], linear_function = <MAX>, replicate_reduce = 0 : i32, nlu_function = <LINEAR>, nlu_preprocess_type = <NONE>, bias_predicate = <NONE>, scale_predicate = <NONE>, try_cellgroups = false, nlu_e8m0_rounding = <NONE>, lowering_hint = <NONE>, compute_type_hint = <NONE>>
  return %arg0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: func.func @test_custom_tiling_options(
func.func @test_custom_tiling_options(%arg0: tensor<4xf32>) -> tensor<4xf32> attributes {darwinn.custom_tiling_options = #darwinn.custom_tiling_options<*>} {
  // CHECK: #darwinn.custom_tiling_options<*>
  return %arg0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: func.func @test_dtc_info(
func.func @test_dtc_info(%arg0: tensor<4xf32>) -> tensor<4xf32> attributes {darwinn.dtc_info = #darwinn.dtc_info<*>} {
  // CHECK: #darwinn.dtc_info<*>
  return %arg0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: func.func @test_filtered_indices_dispatch_mode(
func.func @test_filtered_indices_dispatch_mode(%arg0: tensor<4xf32>) -> tensor<4xf32> attributes {darwinn.filtered_indices_dispatch_mode = #darwinn.filtered_indices_dispatch_mode<*>} {
  // CHECK: #darwinn.filtered_indices_dispatch_mode<*>
  return %arg0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: func.func @test_join_attributes(
func.func @test_join_attributes(%arg0: tensor<4xf32>) -> tensor<4xf32> attributes {darwinn.join_attributes = #darwinn.join_attributes<*>} {
  // CHECK: #darwinn.join_attributes<*>
  return %arg0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: func.func @test_local_copy_attributes(
func.func @test_local_copy_attributes(%arg0: tensor<4xf32>) -> tensor<4xf32> attributes {darwinn.local_copy_attributes = #darwinn.local_copy_attributes<*>} {
  // CHECK: #darwinn.local_copy_attributes<*>
  return %arg0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: func.func @test_packed_index_options(
func.func @test_packed_index_options(%arg0: tensor<4xf32>) -> tensor<4xf32> attributes {darwinn.packed_index_options = #darwinn.packed_index_options<*>} {
  // CHECK: #darwinn.packed_index_options<*>
  return %arg0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: func.func @test_resampler_options(
func.func @test_resampler_options(%arg0: tensor<4xf32>) -> tensor<4xf32> attributes {darwinn.resampler_options = #darwinn.resampler_options<*>} {
  // CHECK: #darwinn.resampler_options<*>
  return %arg0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: func.func @test_ring_to_tile_options(
func.func @test_ring_to_tile_options(%arg0: tensor<4xf32>) -> tensor<4xf32> attributes {darwinn.ring_to_tile_options = #darwinn.ring_to_tile_options<*>} {
  // CHECK: #darwinn.ring_to_tile_options<*>
  return %arg0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: func.func @test_vex_info(
func.func @test_vex_info(%arg0: tensor<4xf32>) -> tensor<4xf32> attributes {darwinn.vex_info = #darwinn.vex_info<*>} {
  // CHECK: #darwinn.vex_info<*>
  return %arg0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: func.func @test_vrgkh_operation_mode(
func.func @test_vrgkh_operation_mode(%arg0: tensor<4xf32>) -> tensor<4xf32> attributes {darwinn.vrgkh_operation_mode = #darwinn.vrgkh_operation_mode<*>} {
  // CHECK: #darwinn.vrgkh_operation_mode<*>
  return %arg0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: func.func @test_fence(
func.func @test_fence(%arg0: tensor<4xf32>) -> tensor<4xf32> attributes {darwinn.fence = #darwinn.fence<*>} {
  // CHECK: #darwinn.fence<*>
  return %arg0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: func.func @test_filter_cmp(
func.func @test_filter_cmp(%arg0: tensor<4xf32>) -> tensor<4xf32> attributes {darwinn.filter_cmp = #darwinn.filter_cmp<*>} {
  // CHECK: #darwinn.filter_cmp<*>
  return %arg0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: func.func @test_filter_for_hib_gather(
func.func @test_filter_for_hib_gather(%arg0: tensor<4xf32>) -> tensor<4xf32> attributes {darwinn.filter_for_hib_gather = #darwinn.filter_for_hib_gather<*>} {
  // CHECK: #darwinn.filter_for_hib_gather<*>
  return %arg0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: func.func @test_inner_op(
func.func @test_inner_op(%arg0: tensor<4xf32>) -> tensor<4xf32> attributes {darwinn.inner_op = #darwinn.inner_op<*>} {
  // CHECK: #darwinn.inner_op<*>
  return %arg0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: func.func @test_interpolate_method(
func.func @test_interpolate_method(%arg0: tensor<4xf32>) -> tensor<4xf32> attributes {darwinn.interpolate_method = #darwinn.interpolate_method<BILINEAR>} {
  // CHECK: #darwinn.interpolate_method<BILINEAR>
  return %arg0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: func.func @test_linear_func(
func.func @test_linear_func(%arg0: tensor<4xf32>) -> tensor<4xf32> attributes {darwinn.linear_func = #darwinn.linear_func<*>} {
  // CHECK: #darwinn.linear_func<*>
  return %arg0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: func.func @test_memory_space(
func.func @test_memory_space(%arg0: tensor<4xf32>) -> tensor<4xf32> attributes {darwinn.memory_space = #darwinn.memory_space<*>} {
  // CHECK: #darwinn.memory_space<*>
  return %arg0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: func.func @test_nlu_e8m0_rounding(
func.func @test_nlu_e8m0_rounding(%arg0: tensor<4xf32>) -> tensor<4xf32> attributes {darwinn.nlu_e8m0_rounding = #darwinn.nlu_e8m0_rounding<*>} {
  // CHECK: #darwinn.nlu_e8m0_rounding<*>
  return %arg0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: func.func @test_nlu_func(
func.func @test_nlu_func(%arg0: tensor<4xf32>) -> tensor<4xf32> attributes {darwinn.nlu_func = #darwinn.nlu_func<*>} {
  // CHECK: #darwinn.nlu_func<*>
  return %arg0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: func.func @test_nlu_preprocess(
func.func @test_nlu_preprocess(%arg0: tensor<4xf32>) -> tensor<4xf32> attributes {darwinn.nlu_preprocess = #darwinn.nlu_preprocess<*>} {
  // CHECK: #darwinn.nlu_preprocess<*>
  return %arg0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: func.func @test_nlu_predicate(
func.func @test_nlu_predicate(%arg0: tensor<4xf32>) -> tensor<4xf32> attributes {darwinn.nlu_predicate = #darwinn.nlu_predicate<*>} {
  // CHECK: #darwinn.nlu_predicate<*>
  return %arg0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: func.func @test_vica_custom_padding_attr(
func.func @test_vica_custom_padding_attr(%arg0: tensor<4xf32>) -> tensor<4xf32> attributes {darwinn.vica_custom_padding = #darwinn.vica_custom_padding<*>} {
  // CHECK: #darwinn.vica_custom_padding<*>
  return %arg0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: func.func @test_dwc_custom_pad_value_type(
func.func @test_dwc_custom_pad_value_type(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.custom_pad_value_type
  %0 = "dwc.pad"(%arg0) {custom_pad_value_type = #dwc.custom_pad_value_type<UNKNOWN>} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: func.func @test_dwc_nlu_preprocessing(
func.func @test_dwc_nlu_preprocessing(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.nlu_preprocessing
  %0 = "dwc.pad"(%arg0) {nlu_preprocessing = #dwc.nlu_preprocessing<UNKNOWN>} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: func.func @test_dwc_scatter_tensor_ls_operation(
func.func @test_dwc_scatter_tensor_ls_operation(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.scatter_tensor_ls_operation
  %0 = "dwc.pad"(%arg0) {scatter_tensor_ls_operation = #dwc.scatter_tensor_ls_operation<UNKNOWN>} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: func.func @test_dwc_tensor_op_gather_operation(
func.func @test_dwc_tensor_op_gather_operation(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.tensor_op_gather_operation
  %0 = "dwc.pad"(%arg0) {tensor_op_gather_operation = #dwc.tensor_op_gather_operation<UNKNOWN>} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: func.func @test_dwc_probe_subtensor(
func.func @test_dwc_probe_subtensor(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.probe_subtensor
  %0 = "dwc.pad"(%arg0) {probe_subtensor = #dwc.probe_subtensor<"id", dense<0> : tensor<1xi32>, 0, dense<0> : tensor<1xi32>, dense<1> : tensor<1xi32>>} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: func.func @test_dwc_parameter_lookup_table(
func.func @test_dwc_parameter_lookup_table(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.parameter_lookup_table
  %0 = "dwc.pad"(%arg0) {parameter_lookup_table = #dwc.parameter_lookup_table<dense<0> : tensor<4xi32>, 0>} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: func.func @test_dwc_resampler_options(
func.func @test_dwc_resampler_options(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.resampler_options
  %0 = "dwc.pad"(%arg0) {resampler_options = #dwc.resampler_options<8, 4>} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: func.func @test_dwc_dimension_layout(
func.func @test_dwc_dimension_layout(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.dimension_layout
  %0 = "dwc.dimension_layout"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: func.func @test_dwc_materialize_policy(
func.func @test_dwc_materialize_policy(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dwc.materialize_policy
  %0 = "dwc.pad"(%arg0) {materialize_policy = #dwc.materialize_policy<0, [1], {}, []>} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
