#ifndef MLIR_DARWINN_INSTRUCTION_ATTRS
#define MLIR_DARWINN_INSTRUCTION_ATTRS

include "mlir/Dialect/Darwinn/IR/InstructionDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

def DarwinnIsa_TargetKind
    : DarwinnIsa_Enum<"TargetKind", [I32EnumAttrCase<"G5", 0, "g5">]>;

def DarwinnIsa_MemoryPowerState
    : DarwinnIsa_Enum<
          "MemoryPowerState", [I32EnumAttrCase<"Active", 0, "active">,
                               I32EnumAttrCase<"DeepSleep", 1, "deep_sleep">]>;

def DarwinnIsa_MemoryPowerStateAttr
    : EnumAttr<DarwinnIsa_Dialect, DarwinnIsa_MemoryPowerState,
               "memory_power_state">;

def DarwinnIsa_TileMemoryOperation
    : DarwinnIsa_Enum<
          "TileMemoryOperation", [I32EnumAttrCase<"Load", 0, "load">,
                                  I32EnumAttrCase<"Store", 1, "store">]>;

def DarwinnIsa_TileMemoryOperationAttr
    : EnumAttr<DarwinnIsa_Dialect, DarwinnIsa_TileMemoryOperation,
               "tile_memory_operation">;

def DarwinnIsa_TileScalarReplacement
    : DarwinnIsa_Enum<
          "TileScalarReplacement",
          [I32EnumAttrCase<"None", 0, "none">,
           I32EnumAttrCase<"Address", 1, "address">,
           I32EnumAttrCase<"Immediate", 2, "immediate">,
           I32EnumAttrCase<"AddressAndImmediate", 3, "address_and_immediate">]>;

def DarwinnIsa_TileScalarReplacementAttr
    : EnumAttr<DarwinnIsa_Dialect, DarwinnIsa_TileScalarReplacement,
               "tile_scalar_replacement">;

def DarwinnIsa_TargetConfigAttr : DarwinnIsa_Attr<"TargetConfig", "target"> {
  let parameters = (ins EnumParameter<DarwinnIsa_TargetKind>:$target,
      "unsigned":$scalarPcBits, "bool":$sequencerOverwrite);
  let genVerifyDecl = 1;
}

def DarwinnIsa_PredicateAttr : DarwinnIsa_Attr<"Predicate", "predicate"> {
  let parameters = (ins "unsigned":$registerId, "bool":$polarity);
  let genVerifyDecl = 1;
}

def DarwinnIsa_HeaderAttr : DarwinnIsa_Attr<"Header", "header"> {
  let parameters = (ins OptionalParameter<"PredicateAttr">:$predicate,
      "bool":$enableTrace, "bool":$enableSequencerOverwrite);
}

def DarwinnIsa_TileHeaderAttr : DarwinnIsa_Attr<"TileHeader", "tile_header"> {
  let parameters = (ins "HeaderAttr":$header,
      "::mlir::DenseBoolArrayAttr":$multicastBitmap, "uint32_t":$tag);
  let genVerifyDecl = 1;
}

def DarwinnIsa_OverwriteInfoAttr
    : DarwinnIsa_Attr<"OverwriteInfo", "overwrite"> {
  let parameters = (ins "unsigned":$baseAddressSource,
      "unsigned":$multicastBitmapSource, "bool":$overwriteMulticastBitmap);
  let genVerifyDecl = 1;
}

def DarwinnIsa_ExpectedSyncFlagsAttr
    : DarwinnIsa_Attr<"ExpectedSyncFlags", "expected_sync_flags"> {
  let parameters = (ins "::mlir::DenseBoolArrayAttr":$valid,
      "::mlir::DenseI32ArrayAttr":$values);
  let genVerifyDecl = 1;
}

include "mlir/Dialect/Darwinn/IR/InstructionTraversal.td"
include "mlir/Dialect/Darwinn/IR/InstructionTensor.td"
include "mlir/Dialect/Darwinn/IR/InstructionTransport.td"
include "mlir/Dialect/Darwinn/IR/InstructionDma.td"

#endif
