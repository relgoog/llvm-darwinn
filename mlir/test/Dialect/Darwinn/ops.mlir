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
  %0 = darwinn.gather %arg0, %arg1 {axis = 0 : si32} : (tensor<16x8xf32>, tensor<4xi32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// -----
// CHECK-LABEL: hib_gather
func.func @test_hib_gather(%arg0: tensor<16x8xf32>, %arg1: tensor<4xi32>) -> tensor<4x8xf32> {
  // CHECK: darwinn.hib_gather
  %0 = darwinn.hib_gather %arg0, %arg1 {axis = 0 : si32} : (tensor<16x8xf32>, tensor<4xi32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// -----
// CHECK-LABEL: gather_copy
func.func @test_gather_copy(%arg0: tensor<16x8xf32>, %arg1: tensor<4xi32>) -> tensor<4x8xf32> {
  // CHECK: darwinn.gather_copy
  %0 = darwinn.gather_copy %arg0, %arg1 {axis = 0 : si32} : (tensor<16x8xf32>, tensor<4xi32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// -----
// CHECK-LABEL: top_k_lowering_target
func.func @test_top_k_lowering_target(%arg0: tensor<16x8xf32>, %arg1: tensor<4xi32>) -> tensor<4x8xf32> {
  // CHECK: darwinn.gather
  %0 = darwinn.gather %arg0, %arg1 {axis = 0 : si32} : (tensor<16x8xf32>, tensor<4xi32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// -----
// CHECK-LABEL: scatter
func.func @test_scatter(%arg0: tensor<16x8xf32>, %arg1: tensor<4xi32>, %arg2: tensor<4x8xf32>) -> tensor<16x8xf32> {
  // CHECK: darwinn.scatter
  %0 = darwinn.scatter %arg0, %arg1, %arg2 : (tensor<16x8xf32>, tensor<4xi32>, tensor<4x8xf32>) -> tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}
// -----
// CHECK-LABEL: tgc_elementwise_add
func.func @test_tgc_elementwise_add(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK: darwinn.tgc_elementwise_add
  %0 = darwinn.tgc_elementwise_add %arg0, %arg1 : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----
// CHECK-LABEL: bitcast
func.func @test_bitcast(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK: darwinn.bitcast
  %0 = darwinn.bitcast %arg0 : (tensor<8x8xf32>) -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----
// CHECK-LABEL: slice_1d_extent
func.func @test_slice_1d_extent(%arg0: tensor<16xf32>) -> tensor<8xf32> {
  // CHECK: darwinn.slice_1d_extent
  %0 = darwinn.slice_1d_extent %arg0 : (tensor<16xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----
// CHECK-LABEL: scatter_arity_floor
func.func @test_scatter_arity_floor(%arg0: tensor<16x8xf32>, %arg1: tensor<4xi32>) -> tensor<16x8xf32> {
  // CHECK: darwinn.scatter
  %0 = darwinn.scatter %arg0, %arg1 : (tensor<16x8xf32>, tensor<4xi32>) -> tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// -----
// CHECK-LABEL: dive_ref_reduction
func.func @test_dive_ref_reduction(%arg0: tensor<4x8xf32>) -> tensor<4x1xf32> {
  // CHECK: darwinn.dive_ref_reduction
  %0 = darwinn.dive_ref_reduction %arg0 {axes = array<i64: 1>, exclusive = false, reverse = false, acc_type = f32} : (tensor<4x8xf32>) -> tensor<4x1xf32>
  return %0 : tensor<4x1xf32>
}

// -----
// CHECK-LABEL: fill_lower
func.func @test_fill_lower(%arg0: f32) -> tensor<4x4xf32> {
  // CHECK: darwinn.fill
  %0 = darwinn.fill %arg0 : (f32) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}
// -----
// CHECK-LABEL: aux_tensor_type
func.func @test_aux_tensor_type() -> tensor<4xf32> {
  // CHECK: darwinn.aux_tensor_type
  %0 = "darwinn.aux_tensor_type"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: binary_map
func.func @test_binary_map(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: darwinn.binary_map
  %0 = "darwinn.binary_map"(%arg0, %arg1) {function = "add"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: broadcast_shard
func.func @test_broadcast_shard(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: darwinn.broadcast_shard
  %0 = "darwinn.broadcast_shard"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: broadcast_slice
func.func @test_broadcast_slice(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: darwinn.broadcast_slice
  %0 = "darwinn.broadcast_slice"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: cast_in
func.func @test_cast_in(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: darwinn.cast_in
  %0 = "darwinn.cast_in"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: cast_out
func.func @test_cast_out(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: darwinn.cast_out
  %0 = "darwinn.cast_out"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: huffman_decompress
func.func @test_huffman_decompress() -> tensor<4xf32> {
  // CHECK: darwinn.huffman_decompress
  %0 = "darwinn.huffman_decompress"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: huffman_compression
func.func @test_huffman_compression() -> tensor<4xf32> {
  // CHECK: darwinn.huffman_compression
  %0 = "darwinn.huffman_compression"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: chunking_reshape
func.func @test_chunking_reshape(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: darwinn.chunking_reshape
  %0 = "darwinn.chunking_reshape"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: compression_mode
func.func @test_compression_mode() -> tensor<4xf32> {
  // CHECK: darwinn.compression_mode
  %0 = "darwinn.compression_mode"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: compute_lowering_hint
func.func @test_compute_lowering_hint() -> tensor<4xf32> {
  // CHECK: darwinn.compute_lowering_hint
  %0 = "darwinn.compute_lowering_hint"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: compute_op_options
func.func @test_compute_op_options() -> tensor<4xf32> {
  // CHECK: darwinn.compute_op_options
  %0 = "darwinn.compute_op_options"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: compute_type_hint
func.func @test_compute_type_hint() -> tensor<4xf32> {
  // CHECK: darwinn.compute_type_hint
  %0 = "darwinn.compute_type_hint"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: condition_scope
func.func @test_condition_scope() -> tensor<4xf32> {
  // CHECK: darwinn.condition_scope
  %0 = "darwinn.condition_scope"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: const_bias_scale
func.func @test_const_bias_scale() -> tensor<4xf32> {
  // CHECK: darwinn.const_bias_scale
  %0 = "darwinn.const_bias_scale"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: const_type
func.func @test_const_type() -> tensor<4xf32> {
  // CHECK: darwinn.const_type
  %0 = "darwinn.const_type"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: constant_generator
func.func @test_constant_generator() -> tensor<4xf32> {
  // CHECK: darwinn.constant_generator
  %0 = "darwinn.constant_generator"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: copy_from_host
func.func @test_copy_from_host(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: darwinn.copy_from_host
  %0 = "darwinn.copy_from_host"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: copy_using_wide
func.func @test_copy_using_wide(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: darwinn.copy_using_wide
  %0 = "darwinn.copy_using_wide"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: cost_hint
func.func @test_cost_hint() -> tensor<4xf32> {
  // CHECK: darwinn.cost_hint
  %0 = "darwinn.cost_hint"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: create_empty_tensor
func.func @test_create_empty_tensor() -> tensor<4xf32> {
  // CHECK: darwinn.create_empty_tensor
  %0 = "darwinn.create_empty_tensor"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: custom_tiling_options
func.func @test_custom_tiling_options() -> tensor<4xf32> {
  // CHECK: darwinn.custom_tiling_options
  %0 = "darwinn.custom_tiling_options"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: device_type
func.func @test_device_type() -> tensor<4xf32> {
  // CHECK: darwinn.device_type
  %0 = "darwinn.device_type"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_op
func.func @test_dive_op() -> tensor<4xf32> {
  // CHECK: darwinn.dive_op
  %0 = "darwinn.dive_op"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_ref_cwise
func.func @test_dive_ref_cwise() -> tensor<4xf32> {
  // CHECK: darwinn.dive_ref_cwise
  %0 = "darwinn.dive_ref_cwise"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dtc_info
func.func @test_dtc_info() -> tensor<4xf32> {
  // CHECK: darwinn.dtc_info
  %0 = "darwinn.dtc_info"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dynamic_slice
func.func @test_dynamic_slice(%arg0: tensor<4xf32>, %arg1: tensor<1xi32>) -> tensor<4xf32> {
  // CHECK: darwinn.dynamic_slice
  %0 = "darwinn.dynamic_slice"(%arg0, %arg1) {slice_sizes = array<i64: 4>} : (tensor<4xf32>, tensor<1xi32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dynamic_slice_with_copy
func.func @test_dynamic_slice_with_copy(%arg0: tensor<4xf32>, %arg1: tensor<1xi32>) -> tensor<4xf32> {
  // CHECK: darwinn.dynamic_slice_with_copy
  %0 = "darwinn.dynamic_slice_with_copy"(%arg0, %arg1) {slice_sizes = array<i64: 4>} : (tensor<4xf32>, tensor<1xi32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dynamic_update_slice
func.func @test_dynamic_update_slice(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>, %arg2: tensor<1xi32>) -> tensor<4xf32> {
  // CHECK: darwinn.dynamic_update_slice
  %0 = "darwinn.dynamic_update_slice"(%arg0, %arg1, %arg2) : (tensor<4xf32>, tensor<4xf32>, tensor<1xi32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dynamic_update_slice_with_offsets
func.func @test_dynamic_update_slice_with_offsets() -> tensor<4xf32> {
  // CHECK: darwinn.dynamic_update_slice_with_offsets
  %0 = "darwinn.dynamic_update_slice_with_offsets"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: fast_walsh_hadamard_transform
func.func @test_fast_walsh_hadamard_transform() -> tensor<4xf32> {
  // CHECK: darwinn.fast_walsh_hadamard_transform
  %0 = "darwinn.fast_walsh_hadamard_transform"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: filter
func.func @test_filter() -> tensor<4xf32> {
  // CHECK: darwinn.filter
  %0 = "darwinn.filter"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: filter_cmp
func.func @test_filter_cmp(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: darwinn.filter_cmp
  %0 = "darwinn.filter_cmp"(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: filter_output
func.func @test_filter_output() -> tensor<4xf32> {
  // CHECK: darwinn.filter_output
  %0 = "darwinn.filter_output"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: filtered_indices_dispatch_mode
func.func @test_filtered_indices_dispatch_mode() -> tensor<4xf32> {
  // CHECK: darwinn.filtered_indices_dispatch_mode
  %0 = "darwinn.filtered_indices_dispatch_mode"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: fully_connected_zout_indexed
func.func @test_fully_connected_zout_indexed() -> tensor<4xf32> {
  // CHECK: darwinn.fully_connected_zout_indexed
  %0 = "darwinn.fully_connected_zout_indexed"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: function_symmetry
func.func @test_function_symmetry() -> tensor<4xf32> {
  // CHECK: darwinn.function_symmetry
  %0 = "darwinn.function_symmetry"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: get_full_tensor_of_dynamic_view
func.func @test_get_full_tensor_of_dynamic_view() -> tensor<4xf32> {
  // CHECK: darwinn.get_full_tensor_of_dynamic_view
  %0 = "darwinn.get_full_tensor_of_dynamic_view"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: get_indexed_slice
func.func @test_get_indexed_slice() -> tensor<4xf32> {
  // CHECK: darwinn.get_indexed_slice
  %0 = "darwinn.get_indexed_slice"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: get_tensor
func.func @test_get_tensor() -> tensor<4xf32> {
  // CHECK: darwinn.get_tensor
  %0 = "darwinn.get_tensor"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
// -----
// CHECK-LABEL: hl_bitcast
func.func @test_hl_bitcast(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: darwinn.hl_bitcast
  %0 = "darwinn.hl_bitcast"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: host_to_ssram
func.func @test_host_to_ssram() -> tensor<4xf32> {
  // CHECK: darwinn.host_to_ssram
  %0 = "darwinn.host_to_ssram"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: host_to_ssram_shard
func.func @test_host_to_ssram_shard() -> tensor<4xf32> {
  // CHECK: darwinn.host_to_ssram_shard
  %0 = "darwinn.host_to_ssram_shard"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: host_to_tile
func.func @test_host_to_tile() -> tensor<4xf32> {
  // CHECK: darwinn.host_to_tile
  %0 = "darwinn.host_to_tile"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: host_to_tile_shard
func.func @test_host_to_tile_shard() -> tensor<4xf32> {
  // CHECK: darwinn.host_to_tile_shard
  %0 = "darwinn.host_to_tile_shard"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: image_format
func.func @test_image_format() -> tensor<4xf32> {
  // CHECK: darwinn.image_format
  %0 = "darwinn.image_format"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: index_filter
func.func @test_index_filter(%arg0: tensor<4x8xf32>) -> (tensor<4x4xf32>, tensor<4x4xi32>) {
  // CHECK: darwinn.index_filter
  %values, %indices = darwinn.index_filter %arg0 {k = 4 : si32} : (tensor<4x8xf32>) -> (tensor<4x4xf32>, tensor<4x4xi32>)
  return %values, %indices : tensor<4x4xf32>, tensor<4x4xi32>
}

// -----
// CHECK-LABEL: infeed
func.func @test_infeed() -> tensor<4xf32> {
  // CHECK: darwinn.infeed
  %0 = "darwinn.infeed"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: inner_op
func.func @test_inner_op() -> tensor<4xf32> {
  // CHECK: darwinn.inner_op
  %0 = "darwinn.inner_op"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: interpolate
func.func @test_interpolate(%arg0: tensor<1x2x4x4xf32>) -> tensor<1x2x2x2xf32> {
  // CHECK: darwinn.interpolate
  %0 = darwinn.interpolate %arg0 {kernel = array<i64: 2, 2>, stride = array<i64: 2, 2>, pad = array<i64: 0, 0, 0, 0>} : (tensor<1x2x4x4xf32>) -> tensor<1x2x2x2xf32>
  return %0 : tensor<1x2x2x2xf32>
}

// -----
// CHECK-LABEL: interpolate_hardware
func.func @test_interpolate_hardware(%arg0: tensor<1x2x4x4xf32>) -> tensor<1x2x2x2xf32> {
  // CHECK: darwinn.interpolate_hardware
  %0 = darwinn.interpolate_hardware %arg0 {kernel = array<i64: 2, 2>, stride = array<i64: 2, 2>, pad = array<i64: 0, 0, 0, 0>} : (tensor<1x2x4x4xf32>) -> tensor<1x2x2x2xf32>
  return %0 : tensor<1x2x2x2xf32>
}

// -----
// CHECK-LABEL: interpolate_method
func.func @test_interpolate_method() -> tensor<4xf32> {
  // CHECK: darwinn.interpolate_method
  %0 = "darwinn.interpolate_method"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: iota
func.func @test_iota() -> tensor<4xf32> {
  // CHECK: darwinn.iota
  %0 = "darwinn.iota"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: is_bool
func.func @test_is_bool() -> tensor<4xf32> {
  // CHECK: darwinn.is_bool
  %0 = "darwinn.is_bool"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: is_parameter
func.func @test_is_parameter() -> tensor<4xf32> {
  // CHECK: darwinn.is_parameter
  %0 = "darwinn.is_parameter"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: is_sparsely_packed
func.func @test_is_sparsely_packed() -> tensor<4xf32> {
  // CHECK: darwinn.is_sparsely_packed
  %0 = "darwinn.is_sparsely_packed"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: join_attributes
func.func @test_join_attributes() -> tensor<4xf32> {
  // CHECK: darwinn.join_attributes
  %0 = "darwinn.join_attributes"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: kernel_level
func.func @test_kernel_level() -> tensor<4xf32> {
  // CHECK: darwinn.kernel_level
  %0 = "darwinn.kernel_level"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: launch_custom_kernel
func.func @test_launch_custom_kernel() -> tensor<4xf32> {
  // CHECK: darwinn.launch_custom_kernel
  %0 = "darwinn.launch_custom_kernel"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: launch_function
func.func @test_launch_function() -> tensor<4xf32> {
  // CHECK: darwinn.launch_function
  %0 = "darwinn.launch_function"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: legacy_interpolate
func.func @test_legacy_interpolate() -> tensor<4xf32> {
  // CHECK: darwinn.legacy_interpolate
  %0 = "darwinn.legacy_interpolate"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: linear_func
func.func @test_linear_func() -> tensor<4xf32> {
  // CHECK: darwinn.linear_func
  %0 = "darwinn.linear_func"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: local_copy_attributes
func.func @test_local_copy_attributes() -> tensor<4xf32> {
  // CHECK: darwinn.local_copy_attributes
  %0 = "darwinn.local_copy_attributes"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: mapping
func.func @test_mapping() -> tensor<4xf32> {
  // CHECK: darwinn.mapping
  %0 = "darwinn.mapping"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: mask_indices
func.func @test_mask_indices(%arg0: tensor<4x8xf32>) -> tensor<4xi32> {
  // CHECK: darwinn.mask_indices
  %0 = darwinn.mask_indices %arg0 {axis = 1 : si32} : (tensor<4x8xf32>) -> tensor<4xi32>
  return %0 : tensor<4xi32>
}

// -----
// CHECK-LABEL: materialize_cast
func.func @test_materialize_cast(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: darwinn.materialize_cast
  %0 = "darwinn.materialize_cast"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: materialize_policy
func.func @test_materialize_policy() -> tensor<4xf32> {
  // CHECK: darwinn.materialize_policy
  %0 = "darwinn.materialize_policy"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: mem_space
func.func @test_mem_space() -> tensor<4xf32> {
  // CHECK: darwinn.mem_space
  %0 = "darwinn.mem_space"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: mesh_pad_slice
func.func @test_mesh_pad_slice(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: darwinn.mesh_pad_slice
  %0 = "darwinn.mesh_pad_slice"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: mma_compute_op
func.func @test_mma_compute_op(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: darwinn.mma_compute_op
  %0 = "darwinn.mma_compute_op"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: multimedia
func.func @test_multimedia() -> tensor<4xf32> {
  // CHECK: darwinn.multimedia
  %0 = "darwinn.multimedia"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: narrow_to_narrow
func.func @test_narrow_to_narrow(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: darwinn.narrow_to_narrow
  %0 = "darwinn.narrow_to_narrow"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: narrow_to_narrow_shard
func.func @test_narrow_to_narrow_shard(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: darwinn.narrow_to_narrow_shard
  %0 = "darwinn.narrow_to_narrow_shard"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: narrow_to_narrow_slice
func.func @test_narrow_to_narrow_slice(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: darwinn.narrow_to_narrow_slice
  %0 = "darwinn.narrow_to_narrow_slice"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: narrow_to_wide
func.func @test_narrow_to_wide(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: darwinn.narrow_to_wide
  %0 = "darwinn.narrow_to_wide"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: narrow_to_wide_shard
func.func @test_narrow_to_wide_shard(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: darwinn.narrow_to_wide_shard
  %0 = "darwinn.narrow_to_wide_shard"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: narrow_to_wide_slice
func.func @test_narrow_to_wide_slice(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: darwinn.narrow_to_wide_slice
  %0 = "darwinn.narrow_to_wide_slice"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: nlu_e8m0_rounding
func.func @test_nlu_e8m0_rounding() -> tensor<4xf32> {
  // CHECK: darwinn.nlu_e8m0_rounding
  %0 = "darwinn.nlu_e8m0_rounding"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: nlu_func
func.func @test_nlu_func() -> tensor<4xf32> {
  // CHECK: darwinn.nlu_func
  %0 = "darwinn.nlu_func"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: nlu_predicate
func.func @test_nlu_predicate() -> tensor<4xf32> {
  // CHECK: darwinn.nlu_predicate
  %0 = "darwinn.nlu_predicate"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: nlu_preprocess
func.func @test_nlu_preprocess() -> tensor<4xf32> {
  // CHECK: darwinn.nlu_preprocess
  %0 = "darwinn.nlu_preprocess"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: outfeed
func.func @test_outfeed() -> tensor<4xf32> {
  // CHECK: darwinn.outfeed
  %0 = "darwinn.outfeed"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: packed_index_options
func.func @test_packed_index_options() -> tensor<4xf32> {
  // CHECK: darwinn.packed_index_options
  %0 = "darwinn.packed_index_options"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: parallel_mesh_copy
func.func @test_parallel_mesh_copy(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: darwinn.parallel_mesh_copy
  %0 = "darwinn.parallel_mesh_copy"(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: preemption_point
func.func @test_preemption_point() -> tensor<4xf32> {
  // CHECK: darwinn.preemption_point
  %0 = "darwinn.preemption_point"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: probe
func.func @test_probe() -> tensor<4xf32> {
  // CHECK: darwinn.probe
  %0 = "darwinn.probe"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: reinterpret_cast
func.func @test_reinterpret_cast(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: darwinn.reinterpret_cast
  %0 = "darwinn.reinterpret_cast"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: relu
func.func @test_relu(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: darwinn.relu
  %0 = "darwinn.relu"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: resampler
func.func @test_resampler(%arg0: tensor<1x2x4x4xf32>) -> tensor<1x2x2x2xf32> {
  // CHECK: darwinn.resampler
  %0 = darwinn.resampler %arg0 {kernel = array<i64: 2, 2>, stride = array<i64: 2, 2>, pad = array<i64: 0, 0, 0, 0>} : (tensor<1x2x4x4xf32>) -> tensor<1x2x2x2xf32>
  return %0 : tensor<1x2x2x2xf32>
}

// -----
// CHECK-LABEL: resampler_options
func.func @test_resampler_options() -> tensor<4xf32> {
  // CHECK: darwinn.resampler_options
  %0 = "darwinn.resampler_options"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: reshape_op
func.func @test_reshape_op(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: darwinn.reshape_op
  %0 = "darwinn.reshape_op"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: ring_to_tile_options
func.func @test_ring_to_tile_options() -> tensor<4xf32> {
  // CHECK: darwinn.ring_to_tile_options
  %0 = "darwinn.ring_to_tile_options"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: ring_to_tile_slice
func.func @test_ring_to_tile_slice(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: darwinn.ring_to_tile_slice
  %0 = "darwinn.ring_to_tile_slice"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: vica_compute_op
func.func @test_vica_compute_op() -> tensor<4xf32> {
  // CHECK: darwinn.vica_compute_op
  %0 = "darwinn.vica_compute_op"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: vica_custom_padding
func.func @test_vica_custom_padding(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: darwinn.vica_custom_padding
  %0 = "darwinn.vica_custom_padding"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: vica_depth_to_space_op
func.func @test_vica_depth_to_space_op(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: darwinn.vica_depth_to_space_op
  %0 = "darwinn.vica_depth_to_space_op"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: vica_residual_add_op
func.func @test_vica_residual_add_op(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: darwinn.vica_residual_add_op
  %0 = "darwinn.vica_residual_add_op"(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: vica_unary_compute_op
func.func @test_vica_unary_compute_op(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: darwinn.vica_unary_compute_op
  %0 = "darwinn.vica_unary_compute_op"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
// -----
// CHECK-LABEL: rng_bit_generator
func.func @test_rng_bit_generator() -> tensor<4xf32> {
  // CHECK: darwinn.rng_bit_generator
  %0 = "darwinn.rng_bit_generator"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: scalar_register_to_host_transfer
func.func @test_scalar_register_to_host_transfer(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: darwinn.scalar_register_to_host_transfer
  %0 = "darwinn.scalar_register_to_host_transfer"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: scalar_register_to_tile_transfer
func.func @test_scalar_register_to_tile_transfer(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: darwinn.scalar_register_to_tile_transfer
  %0 = "darwinn.scalar_register_to_tile_transfer"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: scale
func.func @test_scale() -> tensor<4xf32> {
  // CHECK: darwinn.scale
  %0 = "darwinn.scale"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: scalar_fadd
func.func @test_scalar_fadd(%arg0: bf16, %arg1: bf16) -> bf16 {
  // CHECK: darwinn.scalar.fadd
  %0 = "darwinn.scalar.fadd"(%arg0, %arg1) : (bf16, bf16) -> bf16
  return %0 : bf16
}

// -----
// CHECK-LABEL: scalar_feq
func.func @test_scalar_feq(%arg0: f16, %arg1: f16) -> f16 {
  // CHECK: darwinn.scalar.feq
  %0 = "darwinn.scalar.feq"(%arg0, %arg1) : (f16, f16) -> f16
  return %0 : f16
}

// -----
// CHECK-LABEL: scalar_fgt
func.func @test_scalar_fgt(%arg0: bf16, %arg1: bf16) -> bf16 {
  // CHECK: darwinn.scalar.fgt
  %0 = "darwinn.scalar.fgt"(%arg0, %arg1) : (bf16, bf16) -> bf16
  return %0 : bf16
}

// -----
// CHECK-LABEL: scalar_fgte
func.func @test_scalar_fgte(%arg0: bf16, %arg1: bf16) -> bf16 {
  // CHECK: darwinn.scalar.fgte
  %0 = "darwinn.scalar.fgte"(%arg0, %arg1) : (bf16, bf16) -> bf16
  return %0 : bf16
}

// -----
// CHECK-LABEL: scalar_flt
func.func @test_scalar_flt(%arg0: bf16, %arg1: bf16) -> bf16 {
  // CHECK: darwinn.scalar.flt
  %0 = "darwinn.scalar.flt"(%arg0, %arg1) : (bf16, bf16) -> bf16
  return %0 : bf16
}

// -----
// CHECK-LABEL: scalar_flte
func.func @test_scalar_flte(%arg0: bf16, %arg1: bf16) -> bf16 {
  // CHECK: darwinn.scalar.flte
  %0 = "darwinn.scalar.flte"(%arg0, %arg1) : (bf16, bf16) -> bf16
  return %0 : bf16
}

// -----
// CHECK-LABEL: scalar_inplace_dilatef
func.func @test_scalar_inplace_dilatef() -> tensor<4xf32> {
  // CHECK: darwinn.scalar.inplace_dilatef
  %0 = "darwinn.scalar.inplace_dilatef"() {assigned_register = 3 : i5} : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: scalar_inplace_truncatef
func.func @test_scalar_inplace_truncatef() -> tensor<4xf32> {
  // CHECK: darwinn.scalar.inplace_truncatef
  %0 = "darwinn.scalar.inplace_truncatef"() {assigned_register = 3 : i5} : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: select
func.func @test_select(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>, %arg2: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: darwinn.select
  %0 = "darwinn.select"(%arg0, %arg1, %arg2) : (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: slice_1d_extent_with_padding_info
func.func @test_slice_1d_extent_with_padding_info(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: darwinn.slice_1d_extent_with_padding_info
  %0 = "darwinn.slice_1d_extent_with_padding_info"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: sparse_narrow_to_wide
func.func @test_sparse_narrow_to_wide(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: darwinn.sparse_narrow_to_wide
  %0 = "darwinn.sparse_narrow_to_wide"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: sparse_narrow_to_wide_slice
func.func @test_sparse_narrow_to_wide_slice(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: darwinn.sparse_narrow_to_wide_slice
  %0 = "darwinn.sparse_narrow_to_wide_slice"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: sparse_tensor_op
func.func @test_sparse_tensor_op() -> tensor<4xf32> {
  // CHECK: darwinn.sparse_tensor_op
  %0 = "darwinn.sparse_tensor_op"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: sparsity
func.func @test_sparsity() -> tensor<4xf32> {
  // CHECK: darwinn.sparsity
  %0 = "darwinn.sparsity"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: sparsity_type
func.func @test_sparsity_type() -> tensor<4xf32> {
  // CHECK: darwinn.sparsity_type
  %0 = "darwinn.sparsity_type"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: spline_segment
func.func @test_spline_segment() -> tensor<4xf32> {
  // CHECK: darwinn.spline_segment
  %0 = "darwinn.spline_segment"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: split
func.func @test_split(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: darwinn.split
  %0 = "darwinn.split"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: start_offset_and_stride
func.func @test_start_offset_and_stride() -> tensor<4xf32> {
  // CHECK: darwinn.start_offset_and_stride
  %0 = "darwinn.start_offset_and_stride"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: static_compute_op
func.func @test_static_compute_op() -> tensor<4xf32> {
  // CHECK: darwinn.static_compute_op
  %0 = "darwinn.static_compute_op"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: static_sparse_compute_op
func.func @test_static_sparse_compute_op() -> tensor<4xf32> {
  // CHECK: darwinn.static_sparse_compute_op
  %0 = "darwinn.static_sparse_compute_op"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: static_unary_compute_op
func.func @test_static_unary_compute_op(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: darwinn.static_unary_compute_op
  %0 = "darwinn.static_unary_compute_op"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: streaming_compute_op
func.func @test_streaming_compute_op() -> tensor<4xf32> {
  // CHECK: darwinn.streaming_compute_op
  %0 = "darwinn.streaming_compute_op"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: streaming_copy_op
func.func @test_streaming_copy_op(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: darwinn.streaming_copy_op
  %0 = "darwinn.streaming_copy_op"(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: streaming_sparse_compute_op
func.func @test_streaming_sparse_compute_op() -> tensor<4xf32> {
  // CHECK: darwinn.streaming_sparse_compute_op
  %0 = "darwinn.streaming_sparse_compute_op"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: streaming_sparse_copy_op
func.func @test_streaming_sparse_copy_op(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: darwinn.streaming_sparse_copy_op
  %0 = "darwinn.streaming_sparse_copy_op"(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: streaming_unary_compute_op
func.func @test_streaming_unary_compute_op(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: darwinn.streaming_unary_compute_op
  %0 = "darwinn.streaming_unary_compute_op"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: swizzling
func.func @test_swizzling() -> tensor<4xf32> {
  // CHECK: darwinn.swizzling
  %0 = "darwinn.swizzling"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: synchronized_compute_op
func.func @test_synchronized_compute_op() -> tensor<4xf32> {
  // CHECK: darwinn.synchronized_compute_op
  %0 = "darwinn.synchronized_compute_op"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: synchronized_copy_op
func.func @test_synchronized_copy_op(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: darwinn.synchronized_copy_op
  %0 = "darwinn.synchronized_copy_op"(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: synchronized_sparse_compute_op
func.func @test_synchronized_sparse_compute_op() -> tensor<4xf32> {
  // CHECK: darwinn.synchronized_sparse_compute_op
  %0 = "darwinn.synchronized_sparse_compute_op"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: synchronized_sparse_copy_op
func.func @test_synchronized_sparse_copy_op(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: darwinn.synchronized_sparse_copy_op
  %0 = "darwinn.synchronized_sparse_copy_op"(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: synchronized_unary_compute_op
func.func @test_synchronized_unary_compute_op(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: darwinn.synchronized_unary_compute_op
  %0 = "darwinn.synchronized_unary_compute_op"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: tensor_op
func.func @test_tensor_op() -> tensor<4xf32> {
  // CHECK: darwinn.tensor_op
  %0 = "darwinn.tensor_op"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: tensor_op_shard
func.func @test_tensor_op_shard() -> tensor<4xf32> {
  // CHECK: darwinn.tensor_op_shard
  %0 = "darwinn.tensor_op_shard"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: tensor_op_slice
func.func @test_tensor_op_slice(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: darwinn.tensor_op_slice
  %0 = "darwinn.tensor_op_slice"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: terminate
func.func @test_terminate() -> tensor<4xf32> {
  // CHECK: darwinn.terminate
  %0 = "darwinn.terminate"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: tgc_elementwise_mul
func.func @test_tgc_elementwise_mul(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK: darwinn.tgc_elementwise_mul
  %0 = darwinn.tgc_elementwise_mul %arg0, %arg1 : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----
// CHECK-LABEL: tgc_elementwise_sub
func.func @test_tgc_elementwise_sub(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK: darwinn.tgc_elementwise_sub
  %0 = darwinn.tgc_elementwise_sub %arg0, %arg1 : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----
// CHECK-LABEL: tile_to_host
func.func @test_tile_to_host(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: darwinn.tile_to_host
  %0 = "darwinn.tile_to_host"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: tile_to_host_shard
func.func @test_tile_to_host_shard(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: darwinn.tile_to_host_shard
  %0 = "darwinn.tile_to_host_shard"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: tile_to_ring_slice
func.func @test_tile_to_ring_slice(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: darwinn.tile_to_ring_slice
  %0 = "darwinn.tile_to_ring_slice"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: tile_to_scalar_register_transfer
func.func @test_tile_to_scalar_register_transfer(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: darwinn.tile_to_scalar_register_transfer
  %0 = "darwinn.tile_to_scalar_register_transfer"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: tile_to_tile
func.func @test_tile_to_tile(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: darwinn.tile_to_tile
  %0 = "darwinn.tile_to_tile"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: tile_to_tile_shard
func.func @test_tile_to_tile_shard(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: darwinn.tile_to_tile_shard
  %0 = "darwinn.tile_to_tile_shard"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: unary_map
func.func @test_unary_map(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: darwinn.unary_map
  %0 = "darwinn.unary_map"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: unary_tensor_op
func.func @test_unary_tensor_op(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: darwinn.unary_tensor_op
  %0 = "darwinn.unary_tensor_op"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: vex_info
func.func @test_vex_info() -> tensor<4xf32> {
  // CHECK: darwinn.vex_info
  %0 = "darwinn.vex_info"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: vrgkh_operation_mode
func.func @test_vrgkh_operation_mode() -> tensor<4xf32> {
  // CHECK: darwinn.vrgkh_operation_mode
  %0 = "darwinn.vrgkh_operation_mode"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: while
func.func @test_while() -> tensor<4xf32> {
  // CHECK: darwinn.while
  %0 = "darwinn.while"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: wide_to_narrow
func.func @test_wide_to_narrow(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: darwinn.wide_to_narrow
  %0 = "darwinn.wide_to_narrow"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: wide_to_narrow_slice
func.func @test_wide_to_narrow_slice(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: darwinn.wide_to_narrow_slice
  %0 = "darwinn.wide_to_narrow_slice"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: yield
func.func @test_yield() -> tensor<4xf32> {
  // CHECK: darwinn.yield
  %0 = "darwinn.yield"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: zero_point
func.func @test_zero_point() -> tensor<4xf32> {
  // CHECK: darwinn.zero_point
  %0 = "darwinn.zero_point"() : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
