#ifndef MLIR_DARWINN_INSTRUCTION_OPS
#define MLIR_DARWINN_INSTRUCTION_OPS

include "mlir/Dialect/Darwinn/IR/InstructionAttrs.td"
include "mlir/IR/SymbolInterfaces.td"

def DarwinnIsa_ProgramOp
    : DarwinnIsa_Op<"program", [Symbol, SymbolTable, IsolatedFromAbove,
                                SingleBlock, NoTerminator, NoRegionArguments]> {
  let arguments = (ins SymbolNameAttr:$sym_name,
      DarwinnIsa_TargetConfigAttr:$target_config);
  let regions = (region AnyRegion:$body);
  let assemblyFormat = "$sym_name $target_config attr-dict-with-keyword $body";
  let hasVerifier = 1;
}

def DarwinnIsa_ChunkOp
    : DarwinnIsa_Op<"chunk", [HasParent<"ProgramOp">, IsolatedFromAbove,
                              SingleBlock, NoTerminator, NoRegionArguments]> {
  let arguments = (ins I64Attr:$index);
  let regions = (region AnyRegion:$body);
  let assemblyFormat = "$index attr-dict-with-keyword $body";
  let hasVerifier = 1;
}

def DarwinnIsa_FragmentOp
    : DarwinnIsa_Op<"fragment", [HasParent<"ChunkOp">, SingleBlock,
                                 NoTerminator, NoRegionArguments]> {
  let arguments = (ins I64Attr:$index);
  let regions = (region AnyRegion:$body);
  let assemblyFormat = "$index attr-dict-with-keyword $body";
  let hasVerifier = 1;
}

def DarwinnIsa_NoOp : DarwinnIsa_Instruction<"nop"> {
  let arguments = (ins DarwinnIsa_HeaderAttr:$header);
}

def DarwinnIsa_HaltOp : DarwinnIsa_Instruction<"halt"> {
  let arguments = (ins DarwinnIsa_HeaderAttr:$header, BoolAttr:$goto_halt,
      BoolAttr:$interrupt, I64Attr:$interrupt_id, BoolAttr:$goto_sleep);
}

def DarwinnIsa_InterruptOp : DarwinnIsa_Instruction<"interrupt"> {
  let arguments = (ins DarwinnIsa_HeaderAttr:$header, I64Attr:$interrupt_id);
}

def DarwinnIsa_LoadProgramOp : DarwinnIsa_Instruction<"load_program"> {
  let arguments = (ins DarwinnIsa_HeaderAttr:$header, I64Attr:$start_pc,
      I64Attr:$number_chunks);
}

def DarwinnIsa_ExecuteOp : DarwinnIsa_Instruction<"execute"> {
  let arguments = (ins DarwinnIsa_HeaderAttr:$header, I64Attr:$start_pc,
      BoolAttr:$load_pc);
}

def DarwinnIsa_TileFenceOp : DarwinnIsa_Instruction<"tile_fence"> {
  let arguments = (ins DarwinnIsa_TileHeaderAttr:$header,
      DenseBoolArrayAttr:$reset_sync_flag,
      DenseBoolArrayAttr:$reset_sync_flag_thread_ids, I64Attr:$sync_reset_value,
      DenseBoolArrayAttr:$wait_idle, DenseBoolArrayAttr:$wait_idle_thread_ids,
      DenseBoolArrayAttr:$wait_producer_count_valid,
      I64Attr:$producer_count_value, BoolAttr:$increment, BoolAttr:$halt,
      DarwinnIsa_MemoryPowerStateAttr:$narrow_mem_power_state,
      DarwinnIsa_MemoryPowerStateAttr:$wide_mem_power_state,
      DenseBoolArrayAttr:$virtual_channel_subscription,
      BoolAttr:$virtual_channel_subscription_valid,
      BoolAttr:$increment_global_token_a, BoolAttr:$increment_global_token_b,
      BoolAttr:$discard_remaining_bytes);
}

def DarwinnIsa_ScalarFenceOp : DarwinnIsa_Instruction<"scalar_fence"> {
  let arguments = (ins DarwinnIsa_HeaderAttr:$header, I64Attr:$tag,
      DenseBoolArrayAttr:$reset_tile_fence, DenseBoolArrayAttr:$reset_sync_flag,
      I64Attr:$reset_value, I64Attr:$expected_tile_fence_count,
      DenseBoolArrayAttr:$tile_fence_valid,
      DarwinnIsa_ExpectedSyncFlagsAttr:$expected_scalar_sync_flag,
      DenseBoolArrayAttr:$wait_idle, BoolAttr:$increment_global_token_a,
      BoolAttr:$increment_global_token_b, BoolAttr:$preemptable,
      BoolAttr:$discard_remaining_bytes,
      DenseBoolArrayAttr:$virtual_channel_subscription,
      BoolAttr:$virtual_channel_subscription_valid, BoolAttr:$send_interrupt);
}

def DarwinnIsa_TileLoadStoreOp : DarwinnIsa_Instruction<"tile_load_store"> {
  let arguments = (ins DarwinnIsa_TileHeaderAttr:$header,
      DarwinnIsa_TileMemoryOperationAttr:$memory_operation,
      I64Attr:$row_address, I64Attr:$register_burst_length,
      I64Attr:$base_register, BoolAttr:$use_immediate, I64Attr:$immediate_value,
      BoolAttr:$use_trace_registers, I64Attr:$scalar_register,
      DarwinnIsa_TileScalarReplacementAttr:$replace_field,
      DenseBoolArrayAttr:$thread_bitmap);
}

def DarwinnIsa_CoefficientTablesOp
    : DarwinnIsa_Instruction<"coefficient_tables"> {
  let arguments = (ins DarwinnIsa_TileHeaderAttr:$header,
      DarwinnIsa_OverwriteInfoAttr:$overwrite,
      DenseBoolArrayAttr:$register_sourced_operands,
      DenseF32ArrayAttr:$spline_segments,
      DenseF32ArrayAttr:$segment_lower_bounds, I64Attr:$polynomial_degree,
      DenseBoolArrayAttr:$enhanced_square_bitmap,
      DenseBoolArrayAttr:$thread_multicast_bitmap);
}

#endif
