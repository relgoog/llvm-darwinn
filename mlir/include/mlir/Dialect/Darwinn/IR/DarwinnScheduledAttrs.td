#ifndef MLIR_DIALECT_DARWINN_IR_DARWINNSCHEDULEDATTRS_TD
#define MLIR_DIALECT_DARWINN_IR_DARWINNSCHEDULEDATTRS_TD

include "mlir/Dialect/Darwinn/IR/DarwinnDialect.td"
include "mlir/IR/EnumAttr.td"

def Darwinn_AuxTensorKind
    : I32EnumAttr<"AuxTensorKind", "auxiliary tensor kind",
                  [I32EnumAttrCase<"Bias", 0, "BIAS">,
                   I32EnumAttrCase<"Scale", 1, "SCALE">,
                   I32EnumAttrCase<"BiasScale", 2, "BIAS_SCALE">,
                   I32EnumAttrCase<"RhsZeroPoint", 3, "RHS_ZERO_POINT">,
                   I32EnumAttrCase<"Indices", 4, "INDICES">,
                   I32EnumAttrCase<"Bitmask", 5, "BITMASK">,
                   I32EnumAttrCase<"Alpha", 6, "ALPHA">,
                   I32EnumAttrCase<"Beta", 7, "BETA">,
                   I32EnumAttrCase<"SubChannelActivationScale", 8,
                                   "SUB_CHANNEL_ACTIVATION_SCALE">,
                   I32EnumAttrCase<"SubChannelParameterScale", 9,
                                   "SUB_CHANNEL_PARAMETER_SCALE">,
                   I32EnumAttrCase<"Parameter", 10, "PARAMETER">,
                   I32EnumAttrCase<"ParameterLut", 11, "PARAMETER_LUT">,
                   I32EnumAttrCase<"ParameterLutZeroPoint", 12,
                                   "PARAMETER_LUT_ZERO_POINT">]> {
  let cppNamespace = "::mlir::darwinn";
  let genSpecializedAttr = 0;
}

def Darwinn_ConstKind
    : I32EnumAttr<"ConstKind", "constant tensor kind",
                  [I32EnumAttrCase<"None", 0, "NONE">,
                   I32EnumAttrCase<"Parameter", 1, "PARAMETER">,
                   I32EnumAttrCase<"BiasOrScale", 2, "BIAS_OR_SCALE">,
                   I32EnumAttrCase<"SparseParameter", 3, "SPARSE_PARAMETER">,
                   I32EnumAttrCase<"SparseBitmap", 4, "SPARSE_BITMAP">,
                   I32EnumAttrCase<"ParameterLut", 5, "PARAMETER_LUT">,
                   I32EnumAttrCase<"ParameterLutZeroPoint", 6,
                                   "PARAMETER_LUT_ZERO_POINT">]> {
  let cppNamespace = "::mlir::darwinn";
  let genSpecializedAttr = 0;
}

def Darwinn_MemSpaceKind
    : I32EnumAttr<"MemSpaceKind", "memory space of a streamed view",
                  [I32EnumAttrCase<"HostMemory", 0, "HOST_MEMORY">,
                   I32EnumAttrCase<"TileMemory", 1, "TILE_MEMORY">,
                   I32EnumAttrCase<"TileRegisters", 2, "TILE_REGISTERS">]> {
  let cppNamespace = "::mlir::darwinn";
  let genSpecializedAttr = 0;
}

def Darwinn_MemSpaceAttr
    : EnumAttr<Darwinn_Dialect, Darwinn_MemSpaceKind, "mem_space"> {
  let cppClassName = "MemSpaceAttr";
  let assemblyFormat = "`<` $value `>`";
}

def Darwinn_AuxTensorTypeAttr
    : EnumAttr<Darwinn_Dialect, Darwinn_AuxTensorKind, "aux_tensor_type"> {
  let cppClassName = "AuxTensorTypeAttr";
  let assemblyFormat = "`<` $value `>`";
}

def Darwinn_ConstTypeAttr
    : EnumAttr<Darwinn_Dialect, Darwinn_ConstKind, "const_type"> {
  let cppClassName = "ConstTypeAttr";
  let assemblyFormat = "`<` $value `>`";
}

def Darwinn_MappingAttr : AttrDef<Darwinn_Dialect, "Mapping"> {
  let mnemonic = "mapping";
  let parameters = (ins "::mlir::AffineMapAttr":$forward_index_transformation,
      "::mlir::AffineMapAttr":$reverse_index_transformation);
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

def Darwinn_StartOffsetAndStrideAttr
    : AttrDef<Darwinn_Dialect, "StartOffsetAndStride"> {
  let mnemonic = "start_offset_and_stride";
  let parameters = (ins "uint32_t":$start_offset, "uint32_t":$stride,
      "int32_t":$num_fractional_bits);
  let assemblyFormat = "`<` struct(params) `>`";
}

def Darwinn_TensorOpSliceAttr : AttrDef<Darwinn_Dialect, "TensorOpSlice"> {
  let mnemonic = "tensor_op_slice";
  let parameters = (ins OptionalParameter<"::mlir::ArrayAttr">:$id,
      OptionalParameter<"::mlir::ArrayAttr">:$read_domain,
      OptionalParameter<"::mlir::IntegerAttr">:$read_alternate_limit,
      OptionalParameter<"::mlir::IntegerAttr">:$read_alternate_limit_loop,
      OptionalParameter<"::mlir::AffineMapAttr">:$lhs_read_affine_map,
      OptionalParameter<"::mlir::IntegerAttr">:$lhs_read_base_address,
      OptionalParameter<"::mlir::IntegerAttr">:$lhs_read_bytes,
      OptionalParameter<"::mlir::IntegerAttr">:$lhs_last_read_bytes,
      OptionalParameter<"::mlir::IntegerAttr">:$lhs_last_read_bytes_loop,
      OptionalParameter<"::mlir::AffineMapAttr">:$lhs_gather_index_affine_map,
      OptionalParameter<"::mlir::AffineMapAttr">:$lhs_gather_index_to_lhs_map,
      OptionalParameter<"::mlir::"
                        "AffineMapAttr">:$lhs_gather_indexed_dim_filter_begin,
      OptionalParameter<"::mlir::"
                        "AffineMapAttr">:$lhs_gather_indexed_dim_filter_end,
      OptionalParameter<"::mlir::IntegerAttr">:$lhs_gather_index_base_address,
      OptionalParameter<"::mlir::AffineMapAttr">:$rhs_read_affine_map,
      OptionalParameter<"::mlir::IntegerAttr">:$rhs_read_base_address,
      OptionalParameter<"::mlir::IntegerAttr">:$rhs_read_bytes,
      OptionalParameter<"::mlir::IntegerAttr">:$rhs_zout,
      OptionalParameter<"::mlir::IntegerAttr">:$rhs_last_zout,
      OptionalParameter<"::mlir::IntegerAttr">:$rhs_last_zout_loop,
      OptionalParameter<"::mlir::IntegerAttr">:$rhs_double_buffer_loop,
      OptionalParameter<"::mlir::IntegerAttr">:$rhs_double_buffer_offset,
      OptionalParameter<"::mlir::IntegerAttr">:$rhs_sync_loop,
      OptionalParameter<"::mlir::AffineMapAttr">:$psum_read_affine_map,
      OptionalParameter<"::mlir::IntegerAttr">:$psum_read_base_address,
      OptionalParameter<"::mlir::IntegerAttr">:$index_base_address,
      OptionalParameter<"::mlir::AffineMapAttr">:$index_affine_map,
      OptionalParameter<"::mlir::ArrayAttr">:$write_domain,
      OptionalParameter<"::mlir::AffineMapAttr">:$result_write_affine_map,
      OptionalParameter<"::mlir::IntegerAttr">:$result_write_base_address,
      OptionalParameter<"::mlir::IntegerAttr">:$result_write_bytes,
      OptionalParameter<"::mlir::IntegerAttr">:$result_last_write_bytes,
      OptionalParameter<"::mlir::IntegerAttr">:$result_last_write_bytes_loop,
      OptionalParameter<"::mlir::IntegerAttr">:$result_last_write_loop_limit,
      OptionalParameter<"::mlir::IntegerAttr">:$result_last_write_loop_trigger,
      OptionalParameter<"::mlir::IntegerAttr">:$result_self_sync_loop);
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}
def Darwinn_NarrowToWideSliceAttr
    : AttrDef<Darwinn_Dialect, "NarrowToWideSlice"> {
  let mnemonic = "narrow_to_wide_slice";
  let parameters =
      (ins OptionalParameter<"::mlir::ArrayAttr">:$id,
          OptionalParameter<"::mlir::IntegerAttr">:$read_base_address,
          OptionalParameter<"::mlir::ArrayAttr">:$read_domain,
          OptionalParameter<"::mlir::AffineMapAttr">:$read_affine_map,
          OptionalParameter<"::mlir::IntegerAttr">:$read_bytes,
          OptionalParameter<"::mlir::IntegerAttr">:$last_read_bytes,
          OptionalParameter<"::mlir::IntegerAttr">:$last_read_bytes_loop,
          OptionalParameter<"::mlir::IntegerAttr">:$read_alternate_limit,
          OptionalParameter<
              "::mlir::IntegerAttr">:$read_alternate_limit_applied_loop,
          OptionalParameter<
              "::mlir::IntegerAttr">:$read_alternate_limit_trigger_loop,
          OptionalParameter<"::mlir::IntegerAttr">:$read_double_buffer_loop,
          OptionalParameter<"::mlir::IntegerAttr">:$read_sync_loop,
          OptionalParameter<"::mlir::IntegerAttr">:$read_prologue_loop,
          OptionalParameter<"::mlir::IntegerAttr">:$read_prologue_base_address,
          OptionalParameter<"::mlir::IntegerAttr">:$read_prologue_inner_limit,
          OptionalParameter<"::mlir::IntegerAttr">:$read_prologue_outer_stride,
          OptionalParameter<"::mlir::IntegerAttr">:$read_prologue_outer_limit,
          OptionalParameter<"::mlir::IntegerAttr">:$read_prologue_bytes,
          OptionalParameter<"::mlir::IntegerAttr">:$write_base_address,
          OptionalParameter<"::mlir::ArrayAttr">:$write_domain,
          OptionalParameter<"::mlir::AffineMapAttr">:$write_affine_map,
          OptionalParameter<"::mlir::IntegerAttr">:$write_bytes,
          OptionalParameter<"::mlir::IntegerAttr">:$last_write_bytes,
          OptionalParameter<"::mlir::IntegerAttr">:$last_write_bytes_loop,
          OptionalParameter<"::mlir::IntegerAttr">:$stride_unit_granules,
          OptionalParameter<"::mlir::IntegerAttr">:$last_stride_unit_granules,
          OptionalParameter<"::mlir::IntegerAttr">:$cell_stride,
          OptionalParameter<"::mlir::IntegerAttr">:$cell_stride_group_count,
          OptionalParameter<
              "::mlir::IntegerAttr">:$last_cell_stride_group_count,
          OptionalParameter<
              "::mlir::IntegerAttr">:$last_cell_stride_group_count_loop,
          OptionalParameter<"::mlir::IntegerAttr">:$write_double_buffer_loop,
          OptionalParameter<"::mlir::IntegerAttr">:$write_double_buffer_offset,
          OptionalParameter<"::mlir::IntegerAttr">:$write_sync_loop,
          OptionalParameter<"::mlir::IntegerAttr">:$trigger_loop,
          OptionalParameter<"::mlir::BoolAttr">:$transpose,
          OptionalParameter<"::mlir::IntegerAttr">:$write_prologue_loop,
          OptionalParameter<"::mlir::IntegerAttr">:$write_prologue_base_address,
          OptionalParameter<"::mlir::IntegerAttr">:$write_prologue_inner_limit,
          OptionalParameter<"::mlir::IntegerAttr">:$write_prologue_outer_stride,
          OptionalParameter<"::mlir::IntegerAttr">:$write_prologue_outer_limit,
          OptionalParameter<"::mlir::IntegerAttr">:$write_prologue_bytes);
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

def Darwinn_TensorOpShardAttr : AttrDef<Darwinn_Dialect, "TensorOpShard"> {
  let mnemonic = "tensor_op_shard";
  let parameters = (ins OptionalParameter<"::mlir::ArrayAttr">:$shard_id,
      OptionalParameter<"::mlir::ArrayAttr">:$slices);
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

def Darwinn_NarrowToWideShardAttr
    : AttrDef<Darwinn_Dialect, "NarrowToWideShard"> {
  let mnemonic = "narrow_to_wide_shard";
  let parameters = (ins OptionalParameter<"::mlir::ArrayAttr">:$shard_id,
      OptionalParameter<"::mlir::ArrayAttr">:$slices);
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

#endif
