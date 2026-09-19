//===-- DarwinnDialect.td - Darwinn dialect ----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the Darwinn dialect, its base classes, and the
// attribute and type definitions whose mnemonics read as data, not ops.
//
//===----------------------------------------------------------------------===//

#ifndef DARWINN_DIALECT
#define DARWINN_DIALECT

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// The Darwinn Dialect.
//===----------------------------------------------------------------------===//

def Darwinn_Dialect : Dialect {
  let name = "darwinn";
  let cppNamespace = "::mlir::darwinn";
  let description = [{
    Skeleton of the Darwinn dialect. Every op is boilerplate with generic
    operands and results until real semantics land.
  }];
  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Darwinn base classes, mirroring the Tosa layout.
//===----------------------------------------------------------------------===//

class Darwinn_Op<string mnemonic, list<Trait> traits = []>
    : Op<Darwinn_Dialect, mnemonic, traits> {}

class Darwinn_Attr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<Darwinn_Dialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

class Darwinn_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Darwinn_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// Darwinn Attributes.
//===----------------------------------------------------------------------===//

def Darwinn_CompressionModeAttr : Darwinn_Attr<"CompressionMode", "compression_mode"> {
  let summary = "Darwinn compression_mode attribute.";
}

def Darwinn_ComputeLoweringHintAttr : Darwinn_Attr<"ComputeLoweringHint", "compute_lowering_hint"> {
  let summary = "Darwinn compute_lowering_hint attribute.";
}

def Darwinn_ComputeTypeHintAttr : Darwinn_Attr<"ComputeTypeHint", "compute_type_hint"> {
  let summary = "Darwinn compute_type_hint attribute.";
}

def Darwinn_CostHintAttr : Darwinn_Attr<"CostHint", "cost_hint"> {
  let summary = "Darwinn cost_hint attribute.";
}

def Darwinn_ComputeOpOptionsAttr : Darwinn_Attr<"ComputeOpOptions", "compute_op_options"> {
  let summary = "Darwinn compute_op_options attribute.";
}

def Darwinn_CustomTilingOptionsAttr : Darwinn_Attr<"CustomTilingOptions", "custom_tiling_options"> {
  let summary = "Darwinn custom_tiling_options attribute.";
}

def Darwinn_PackedIndexOptionsAttr : Darwinn_Attr<"PackedIndexOptions", "packed_index_options"> {
  let summary = "Darwinn packed_index_options attribute.";
}

def Darwinn_ResamplerOptionsAttr : Darwinn_Attr<"ResamplerOptions", "resampler_options"> {
  let summary = "Darwinn resampler_options attribute.";
}

def Darwinn_RingToTileOptionsAttr : Darwinn_Attr<"RingToTileOptions", "ring_to_tile_options"> {
  let summary = "Darwinn ring_to_tile_options attribute.";
}

def Darwinn_DtcInfoAttr : Darwinn_Attr<"DtcInfo", "dtc_info"> {
  let summary = "Darwinn dtc_info attribute.";
}

def Darwinn_VexInfoAttr : Darwinn_Attr<"VexInfo", "vex_info"> {
  let summary = "Darwinn vex_info attribute.";
}

def Darwinn_FilteredIndicesDispatchModeAttr : Darwinn_Attr<"FilteredIndicesDispatchMode", "filtered_indices_dispatch_mode"> {
  let summary = "Darwinn filtered_indices_dispatch_mode attribute.";
}

def Darwinn_VrgkhOperationModeAttr : Darwinn_Attr<"VrgkhOperationMode", "vrgkh_operation_mode"> {
  let summary = "Darwinn vrgkh_operation_mode attribute.";
}

def Darwinn_FunctionSymmetryAttr : Darwinn_Attr<"FunctionSymmetry", "function_symmetry"> {
  let summary = "Darwinn function_symmetry attribute.";
}

def Darwinn_ImageFormatAttr : Darwinn_Attr<"ImageFormat", "image_format"> {
  let summary = "Darwinn image_format attribute.";
}

def Darwinn_JoinAttributesAttr : Darwinn_Attr<"JoinAttributes", "join_attributes"> {
  let summary = "Darwinn join_attributes attribute.";
}

def Darwinn_LocalCopyAttributesAttr : Darwinn_Attr<"LocalCopyAttributes", "local_copy_attributes"> {
  let summary = "Darwinn local_copy_attributes attribute.";
}

def Darwinn_KernelLevelAttr : Darwinn_Attr<"KernelLevel", "kernel_level"> {
  let summary = "Darwinn kernel_level attribute.";
}

def Darwinn_MaterializePolicyAttr : Darwinn_Attr<"MaterializePolicy", "materialize_policy"> {
  let summary = "Darwinn materialize_policy attribute.";
}

def Darwinn_MemSpaceAttr : Darwinn_Attr<"MemSpace", "mem_space"> {
  let summary = "Darwinn mem_space attribute.";
}

//===----------------------------------------------------------------------===//
// Darwinn Types.
//===----------------------------------------------------------------------===//

def Darwinn_AuxTensor : Darwinn_Type<"AuxTensor", "aux_tensor_type"> {
  let summary = "Darwinn aux tensor type.";
}

def Darwinn_Const : Darwinn_Type<"Const", "const_type"> {
  let summary = "Darwinn const type.";
}

def Darwinn_Device : Darwinn_Type<"Device", "device_type"> {
  let summary = "Darwinn device type.";
}

def Darwinn_Sparsity : Darwinn_Type<"Sparsity", "sparsity_type"> {
  let summary = "Darwinn sparsity type.";
}

def Darwinn_Packed : Darwinn_Type<"Packed", "packed", []> {
  let summary = "Darwinn packed type.";
  let parameters = (ins "unsigned":$width);
  let assemblyFormat = "`<` $width `>`";
}

#endif // DARWINN_DIALECT
