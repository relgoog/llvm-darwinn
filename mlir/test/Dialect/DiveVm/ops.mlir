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
// -----
// CHECK-LABEL: sub
func.func @test_sub(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  // CHECK: dive_vm.sub
  %0 = dive_vm.sub %arg0, %arg1 : (tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
// CHECK-LABEL: load
func.func @test_load(%arg0: tensor<4xi32>) -> tensor<4xf32> {
  // CHECK: dive_vm.load
  %0 = dive_vm.load %arg0 : (tensor<4xi32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: br
func.func @test_br() {
  // CHECK: dive_vm.br
  dive_vm.br : () -> ()
  return
}

// -----
// CHECK-LABEL: convert_yuv_to_rgb
func.func @test_convert_yuv_to_rgb(%arg0: tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32> {
  // CHECK: edgetpu.convert_yuv_to_rgb
  %0 = edgetpu.convert_yuv_to_rgb %arg0 : (tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32>
  return %0 : tensor<1x4x4x4xf32>
}
// -----
// CHECK-LABEL: address_space
func.func @test_address_space(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.address_space
  %0 = dive_vm.address_space %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: allocate
func.func @test_allocate(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.allocate
  %0 = dive_vm.allocate %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: benchmark
func.func @test_benchmark(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.benchmark
  %0 = dive_vm.benchmark %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: buffer_dir
func.func @test_buffer_dir(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.buffer_dir
  %0 = dive_vm.buffer_dir %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: cache_clean_invalidate
func.func @test_cache_clean_invalidate(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.cache_clean_invalidate
  %0 = dive_vm.cache_clean_invalidate %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: chunk_offsets
func.func @test_chunk_offsets(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.chunk_offsets
  %0 = dive_vm.chunk_offsets %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: chunk_sizes
func.func @test_chunk_sizes(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.chunk_sizes
  %0 = dive_vm.chunk_sizes %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: compute_norm_stats_for_rkhy
func.func @test_compute_norm_stats_for_rkhy(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.compute_norm_stats_for_rkhy
  %0 = dive_vm.compute_norm_stats_for_rkhy %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: cond_br
func.func @test_cond_br(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.cond_br
  %0 = dive_vm.cond_br %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: const
func.func @test_const(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.const
  %0 = dive_vm.const %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: const_bytes
func.func @test_const_bytes(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.const_bytes
  %0 = dive_vm.const_bytes %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: core_id
func.func @test_core_id(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.core_id
  %0 = dive_vm.core_id %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: cuda_emu_custom_op
func.func @test_cuda_emu_custom_op(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.cuda_emu_custom_op
  %0 = dive_vm.cuda_emu_custom_op %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: cumsum
func.func @test_cumsum(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.cumsum
  %0 = dive_vm.cumsum %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: custom_on_addresses
func.func @test_custom_on_addresses(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.custom_on_addresses
  %0 = dive_vm.custom_on_addresses %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: custom_on_buffers
func.func @test_custom_on_buffers(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.custom_on_buffers
  %0 = dive_vm.custom_on_buffers %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: disable_itc_tracing
func.func @test_disable_itc_tracing(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.disable_itc_tracing
  %0 = dive_vm.disable_itc_tracing %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dispatch_hardware_instruction
func.func @test_dispatch_hardware_instruction(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.dispatch_hardware_instruction
  %0 = dive_vm.dispatch_hardware_instruction %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dive_vm_module
func.func @test_dive_vm_module(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.dive_vm_module
  %0 = dive_vm.dive_vm_module %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dtc_mode
func.func @test_dtc_mode(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.dtc_mode
  %0 = dive_vm.dtc_mode %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dvfs
func.func @test_dvfs(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.dvfs
  %0 = dive_vm.dvfs %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dynamic_slice_y
func.func @test_dynamic_slice_y(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.dynamic_slice_y
  %0 = dive_vm.dynamic_slice_y %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: enable_itc_tracing
func.func @test_enable_itc_tracing(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.enable_itc_tracing
  %0 = dive_vm.enable_itc_tracing %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: epilogue
func.func @test_epilogue(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.epilogue
  %0 = dive_vm.epilogue %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: executable
func.func @test_executable(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.executable
  %0 = dive_vm.executable %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
// -----
// CHECK-LABEL: extract_slice
func.func @test_extract_slice(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.extract_slice
  %0 = dive_vm.extract_slice %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: get_const
func.func @test_get_const(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.get_const
  %0 = dive_vm.get_const %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: has_idp
func.func @test_has_idp(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.has_idp
  %0 = dive_vm.has_idp %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: heap_size_bytes
func.func @test_heap_size_bytes(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.heap_size_bytes
  %0 = dive_vm.heap_size_bytes %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: idp
func.func @test_idp(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.idp
  %0 = dive_vm.idp %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: idp_instance
func.func @test_idp_instance(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.idp_instance
  %0 = dive_vm.idp_instance %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: insert_slice
func.func @test_insert_slice(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.insert_slice
  %0 = dive_vm.insert_slice %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: instruction_id
func.func @test_instruction_id(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.instruction_id
  %0 = dive_vm.instruction_id %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: instruction_id_live_tensors
func.func @test_instruction_id_live_tensors(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.instruction_id_live_tensors
  %0 = dive_vm.instruction_id_live_tensors %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: legacy_scalar
func.func @test_legacy_scalar(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.legacy_scalar
  %0 = dive_vm.legacy_scalar %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: live_tensor
func.func @test_live_tensor(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.live_tensor
  %0 = dive_vm.live_tensor %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: mark_dive_for_host_resident
func.func @test_mark_dive_for_host_resident(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.mark_dive_for_host_resident
  %0 = dive_vm.mark_dive_for_host_resident %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: mark_value_with_shape
func.func @test_mark_value_with_shape(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.mark_value_with_shape
  %0 = dive_vm.mark_value_with_shape %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: mask_indices
func.func @test_mask_indices(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.mask_indices
  %0 = dive_vm.mask_indices %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: module_group
func.func @test_module_group(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.module_group
  %0 = dive_vm.module_group %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: multinomial
func.func @test_multinomial(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.multinomial
  %0 = dive_vm.multinomial %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: num_input_tensors
func.func @test_num_input_tensors(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.num_input_tensors
  %0 = dive_vm.num_input_tensors %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: num_output_tensors
func.func @test_num_output_tensors(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.num_output_tensors
  %0 = dive_vm.num_output_tensors %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: one_hot
func.func @test_one_hot(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.one_hot
  %0 = dive_vm.one_hot %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: pad
func.func @test_pad(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.pad
  %0 = dive_vm.pad %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: parameter_array
func.func @test_parameter_array(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.parameter_array
  %0 = dive_vm.parameter_array %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: patch_instruction_for_strided_io
func.func @test_patch_instruction_for_strided_io(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.patch_instruction_for_strided_io
  %0 = dive_vm.patch_instruction_for_strided_io %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: perform_software_preemption_if_requested
func.func @test_perform_software_preemption_if_requested(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.perform_software_preemption_if_requested
  %0 = dive_vm.perform_software_preemption_if_requested %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: print
func.func @test_print(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.print
  %0 = dive_vm.print %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: program_tensor_mapping_table
func.func @test_program_tensor_mapping_table(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.program_tensor_mapping_table
  %0 = dive_vm.program_tensor_mapping_table %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
// -----
// CHECK-LABEL: prologue
func.func @test_prologue(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.prologue
  %0 = dive_vm.prologue %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: put_bits
func.func @test_put_bits(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.put_bits
  %0 = dive_vm.put_bits %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: read_gn_stats
func.func @test_read_gn_stats(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.read_gn_stats
  %0 = dive_vm.read_gn_stats %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: reduction
func.func @test_reduction(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.reduction
  %0 = dive_vm.reduction %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: reduction_type
func.func @test_reduction_type(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.reduction_type
  %0 = dive_vm.reduction_type %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: roll
func.func @test_roll(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.roll
  %0 = dive_vm.roll %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: scalar_type
func.func @test_scalar_type(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.scalar_type
  %0 = dive_vm.scalar_type %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: serialized_model_format_type
func.func @test_serialized_model_format_type(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.serialized_model_format_type
  %0 = dive_vm.serialized_model_format_type %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: set_dtc_mode
func.func @test_set_dtc_mode(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.set_dtc_mode
  %0 = dive_vm.set_dtc_mode %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: shape_of_activation
func.func @test_shape_of_activation(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.shape_of_activation
  %0 = dive_vm.shape_of_activation %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: shared_tensor
func.func @test_shared_tensor(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.shared_tensor
  %0 = dive_vm.shared_tensor %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: top_k
func.func @test_top_k(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.top_k
  %0 = dive_vm.top_k %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: tpu_offload
func.func @test_tpu_offload(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.tpu_offload
  %0 = dive_vm.tpu_offload %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: transition_dtc_power_island
func.func @test_transition_dtc_power_island(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.transition_dtc_power_island
  %0 = dive_vm.transition_dtc_power_island %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: translate_sram_address
func.func @test_translate_sram_address(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.translate_sram_address
  %0 = dive_vm.translate_sram_address %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: view_on_address
func.func @test_view_on_address(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.view_on_address
  %0 = dive_vm.view_on_address %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: wait_for_fence_completion
func.func @test_wait_for_fence_completion(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.wait_for_fence_completion
  %0 = dive_vm.wait_for_fence_completion %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: wait_for_power_island_transition_complete
func.func @test_wait_for_power_island_transition_complete(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.wait_for_power_island_transition_complete
  %0 = dive_vm.wait_for_power_island_transition_complete %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: wait_for_rkhy_completion
func.func @test_wait_for_rkhy_completion(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.wait_for_rkhy_completion
  %0 = dive_vm.wait_for_rkhy_completion %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: write_hib_data
func.func @test_write_hib_data(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.write_hib_data
  %0 = dive_vm.write_hib_data %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: write_scalar_arch_register
func.func @test_write_scalar_arch_register(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.write_scalar_arch_register
  %0 = dive_vm.write_scalar_arch_register %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
// -----
// CHECK-LABEL: add_imm
func.func @test_add_imm(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.add_imm
  %0 = dive_vm.add_imm %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: arithmetic_left_shift
func.func @test_arithmetic_left_shift(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.arithmetic_left_shift
  %0 = dive_vm.arithmetic_left_shift %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: arithmetic_left_shift_imm
func.func @test_arithmetic_left_shift_imm(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.arithmetic_left_shift_imm
  %0 = dive_vm.arithmetic_left_shift_imm %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: arithmetic_right_shift
func.func @test_arithmetic_right_shift(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.arithmetic_right_shift
  %0 = dive_vm.arithmetic_right_shift %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: arithmetic_right_shift_imm
func.func @test_arithmetic_right_shift_imm(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.arithmetic_right_shift_imm
  %0 = dive_vm.arithmetic_right_shift_imm %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: bitcast
func.func @test_bitcast(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.bitcast
  %0 = dive_vm.bitcast %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: bitwise_and
func.func @test_bitwise_and(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.bitwise_and
  %0 = dive_vm.bitwise_and %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: bitwise_and_imm
func.func @test_bitwise_and_imm(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.bitwise_and_imm
  %0 = dive_vm.bitwise_and_imm %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: bitwise_or
func.func @test_bitwise_or(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.bitwise_or
  %0 = dive_vm.bitwise_or %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: bitwise_or_imm
func.func @test_bitwise_or_imm(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.bitwise_or_imm
  %0 = dive_vm.bitwise_or_imm %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: bitwise_xor
func.func @test_bitwise_xor(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.bitwise_xor
  %0 = dive_vm.bitwise_xor %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: bitwise_xor_imm
func.func @test_bitwise_xor_imm(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.bitwise_xor_imm
  %0 = dive_vm.bitwise_xor_imm %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: cast
func.func @test_cast(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.cast
  %0 = dive_vm.cast %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: copy_imm
func.func @test_copy_imm(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.copy_imm
  %0 = dive_vm.copy_imm %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: div_imm
func.func @test_div_imm(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.div_imm
  %0 = dive_vm.div_imm %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: equal
func.func @test_equal(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.equal
  %0 = dive_vm.equal %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: equal_imm
func.func @test_equal_imm(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.equal_imm
  %0 = dive_vm.equal_imm %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: fill
func.func @test_fill(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.fill
  %0 = dive_vm.fill %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: greater
func.func @test_greater(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.greater
  %0 = dive_vm.greater %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: greater_equal
func.func @test_greater_equal(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.greater_equal
  %0 = dive_vm.greater_equal %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: greater_equal_imm
func.func @test_greater_equal_imm(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.greater_equal_imm
  %0 = dive_vm.greater_equal_imm %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: greater_imm
func.func @test_greater_imm(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.greater_imm
  %0 = dive_vm.greater_imm %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: less
func.func @test_less(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.less
  %0 = dive_vm.less %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: less_equal
func.func @test_less_equal(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.less_equal
  %0 = dive_vm.less_equal %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: less_equal_imm
func.func @test_less_equal_imm(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.less_equal_imm
  %0 = dive_vm.less_equal_imm %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: less_imm
func.func @test_less_imm(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.less_imm
  %0 = dive_vm.less_imm %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: load_indirect
func.func @test_load_indirect(%arg0: tensor<4xi32>) -> tensor<4xf32> {
  // CHECK: dive_vm.load_indirect
  %0 = dive_vm.load_indirect %arg0 : (tensor<4xi32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: logical_and
func.func @test_logical_and(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.logical_and
  %0 = dive_vm.logical_and %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: logical_and_imm
func.func @test_logical_and_imm(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.logical_and_imm
  %0 = dive_vm.logical_and_imm %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: logical_right_shift
func.func @test_logical_right_shift(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.logical_right_shift
  %0 = dive_vm.logical_right_shift %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: logical_right_shift_imm
func.func @test_logical_right_shift_imm(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.logical_right_shift_imm
  %0 = dive_vm.logical_right_shift_imm %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
// -----
// CHECK-LABEL: max
func.func @test_max(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.max
  %0 = dive_vm.max %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: max_imm
func.func @test_max_imm(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.max_imm
  %0 = dive_vm.max_imm %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: min_imm
func.func @test_min_imm(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.min_imm
  %0 = dive_vm.min_imm %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: mul
func.func @test_mul(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.mul
  %0 = dive_vm.mul %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: mul_imm
func.func @test_mul_imm(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.mul_imm
  %0 = dive_vm.mul_imm %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: not_equal
func.func @test_not_equal(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.not_equal
  %0 = dive_vm.not_equal %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: not_equal_imm
func.func @test_not_equal_imm(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.not_equal_imm
  %0 = dive_vm.not_equal_imm %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: pow
func.func @test_pow(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.pow
  %0 = dive_vm.pow %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: pow_imm
func.func @test_pow_imm(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.pow_imm
  %0 = dive_vm.pow_imm %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: rem
func.func @test_rem(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.rem
  %0 = dive_vm.rem %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: rem_imm
func.func @test_rem_imm(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.rem_imm
  %0 = dive_vm.rem_imm %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: select
func.func @test_select(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>, %arg2: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.select
  %0 = dive_vm.select %arg0, %arg1, %arg2 : (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: sign
func.func @test_sign(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.sign
  %0 = dive_vm.sign %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: store
func.func @test_store(%arg0: tensor<4xi32>) {
  // CHECK: dive_vm.store
  dive_vm.store %arg0 : (tensor<4xi32>) -> ()
  return
}

// -----
// CHECK-LABEL: store_indirect
func.func @test_store_indirect(%arg0: tensor<4xi32>) {
  // CHECK: dive_vm.store_indirect
  dive_vm.store_indirect %arg0 : (tensor<4xi32>) -> ()
  return
}

// -----
// CHECK-LABEL: sub_imm
func.func @test_sub_imm(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.sub_imm
  %0 = dive_vm.sub_imm %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dma_hint
func.func @test_dma_hint(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.dma_hint
  %0 = dive_vm.dma_hint %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: dma_queue
func.func @test_dma_queue(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.dma_queue
  %0 = dive_vm.dma_queue %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: gather
func.func @test_gather(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.gather
  %0 = dive_vm.gather %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: gather_nd
func.func @test_gather_nd(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.gather_nd
  %0 = dive_vm.gather_nd %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: hib_gather_edit
func.func @test_hib_gather_edit(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.hib_gather_edit
  %0 = dive_vm.hib_gather_edit %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: scatter_nd
func.func @test_scatter_nd(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.scatter_nd
  %0 = dive_vm.scatter_nd %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: write_dma_descriptor
func.func @test_write_dma_descriptor(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: dive_vm.write_dma_descriptor
  %0 = dive_vm.write_dma_descriptor %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
// -----
// CHECK-LABEL: convolution_sub_channel_drq
func.func @test_convolution_sub_channel_drq(%arg0: tensor<1x4x4x4xf32>, %arg1: tensor<8x1x1x4xf32>) -> tensor<1x4x4x8xf32> {
  // CHECK: edgetpu.convolution_sub_channel_drq
  %0 = edgetpu.convolution_sub_channel_drq %arg0, %arg1 : (tensor<1x4x4x4xf32>, tensor<8x1x1x4xf32>) -> tensor<1x4x4x8xf32>
  return %0 : tensor<1x4x4x8xf32>
}

// -----
// CHECK-LABEL: depthwise_convolution_fp8
func.func @test_depthwise_convolution_fp8(%arg0: tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32> {
  // CHECK: edgetpu.depthwise_convolution_fp8
  %0 = edgetpu.depthwise_convolution_fp8 %arg0 : (tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32>
  return %0 : tensor<1x4x4x4xf32>
}

// -----
// CHECK-LABEL: fast_walsh_hadamard_transform
func.func @test_fast_walsh_hadamard_transform(%arg0: tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32> {
  // CHECK: edgetpu.fast_walsh_hadamard_transform
  %0 = edgetpu.fast_walsh_hadamard_transform %arg0 : (tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32>
  return %0 : tensor<1x4x4x4xf32>
}

// -----
// CHECK-LABEL: fully_connected_fp8
func.func @test_fully_connected_fp8(%arg0: tensor<1x19xf32>) -> tensor<1x28xf32> {
  // CHECK: edgetpu.fully_connected_fp8
  %0 = edgetpu.fully_connected_fp8 %arg0 : (tensor<1x19xf32>) -> tensor<1x28xf32>
  return %0 : tensor<1x28xf32>
}

// -----
// CHECK-LABEL: fully_connected_sub_channel_drq
func.func @test_fully_connected_sub_channel_drq(%arg0: tensor<1x19xf32>, %arg1: tensor<28x19xf32>) -> tensor<1x28xf32> {
  // CHECK: edgetpu.fully_connected_sub_channel_drq
  %0 = edgetpu.fully_connected_sub_channel_drq %arg0, %arg1 : (tensor<1x19xf32>, tensor<28x19xf32>) -> tensor<1x28xf32>
  return %0 : tensor<1x28xf32>
}

// -----
// CHECK-LABEL: matrix_multiply_sub_channel_drq
func.func @test_matrix_multiply_sub_channel_drq(%arg0: tensor<1x14x19xf32>, %arg1: tensor<1x19x28xf32>) -> tensor<1x14x28xf32> {
  // CHECK: edgetpu.matrix_multiply_sub_channel_drq
  %0 = edgetpu.matrix_multiply_sub_channel_drq %arg0, %arg1 : (tensor<1x14x19xf32>, tensor<1x19x28xf32>) -> tensor<1x14x28xf32>
  return %0 : tensor<1x14x28xf32>
}

// -----
// CHECK-LABEL: transpose_convolution_fp8
func.func @test_transpose_convolution_fp8(%arg0: tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32> {
  // CHECK: edgetpu.transpose_convolution_fp8
  %0 = edgetpu.transpose_convolution_fp8 %arg0 : (tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32>
  return %0 : tensor<1x4x4x4xf32>
}

// -----
// CHECK-LABEL: transposed_convolution_sub_channel_drq
func.func @test_transposed_convolution_sub_channel_drq(%arg0: tensor<1x4x4x4xf32>, %arg1: tensor<8x1x1x4xf32>) -> tensor<1x4x4x8xf32> {
  // CHECK: edgetpu.transposed_convolution_sub_channel_drq
  %0 = edgetpu.transposed_convolution_sub_channel_drq %arg0, %arg1 : (tensor<1x4x4x4xf32>, tensor<8x1x1x4xf32>) -> tensor<1x4x4x8xf32>
  return %0 : tensor<1x4x4x8xf32>
}
