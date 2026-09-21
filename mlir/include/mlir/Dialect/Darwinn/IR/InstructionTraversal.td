#ifndef MLIR_DIALECT_DARWINN_IR_INSTRUCTIONTRAVERSAL_TD
#define MLIR_DIALECT_DARWINN_IR_INSTRUCTIONTRAVERSAL_TD

include "mlir/Dialect/Darwinn/IR/InstructionDialect.td"

class DarwinnIsa_TraversalArray<string element> : ArrayRefParameter<element> {
  let parser = [{ [&]() -> ::mlir::FailureOr<::llvm::SmallVector<}] # element # [{>> {
    ::llvm::SmallVector<}] # element # [{> values;
    if ($_parser.parseCommaSeparatedList(::mlir::AsmParser::Delimiter::Square, [&]() {
      auto value = ::mlir::FieldParser<}] # element # [{>::parse($_parser);
      if (::mlir::failed(value))
        return ::mlir::failure();
      values.push_back(*value);
      return ::mlir::success();
    }))
      return ::mlir::failure();
    return values;
  }() }];
  let printer = [{
    $_printer << '[';
    ::llvm::interleaveComma($_self, $_printer);
    $_printer << ']';
  }];
}

def DarwinnIsa_PrologueTarget : I32EnumAttr<"PrologueTarget", "prologue destination", [
  I32EnumAttrCase<"WideRegisters", 0, "wide_registers">,
  I32EnumAttrCase<"ParameterScalesBuffer", 1, "parameter_scales_buffer">
]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::darwinn::isa";
}

def DarwinnIsa_TileSyncFlag : I32EnumAttr<"TileSyncFlag", "tile synchronization flag", [
  I32EnumAttrCase<"ActivationWrite", 0, "activation_write">,
  I32EnumAttrCase<"ParameterRead", 1, "parameter_read">,
  I32EnumAttrCase<"PartialSumWrite", 2, "partial_sum_write">,
  I32EnumAttrCase<"MeshInboundFromNorth", 3, "mesh_inbound_from_north">,
  I32EnumAttrCase<"MeshInboundFromEast", 4, "mesh_inbound_from_east">,
  I32EnumAttrCase<"MeshInboundFromSouth", 5, "mesh_inbound_from_south">,
  I32EnumAttrCase<"MeshInboundFromWest", 6, "mesh_inbound_from_west">,
  I32EnumAttrCase<"MeshOutboundForNorth", 7, "mesh_outbound_for_north">,
  I32EnumAttrCase<"MeshOutboundForEast", 8, "mesh_outbound_for_east">,
  I32EnumAttrCase<"MeshOutboundForSouth", 9, "mesh_outbound_for_south">,
  I32EnumAttrCase<"MeshOutboundForWest", 10, "mesh_outbound_for_west">,
  I32EnumAttrCase<"WideToNarrowWrite", 11, "wide_to_narrow_write">,
  I32EnumAttrCase<"WideToScaling", 12, "wide_to_scaling">,
  I32EnumAttrCase<"NarrowToWideWrite", 13, "narrow_to_wide_write">,
  I32EnumAttrCase<"RingBusReadA", 14, "ring_bus_read_a">,
  I32EnumAttrCase<"RingBusReadB", 15, "ring_bus_read_b">,
  I32EnumAttrCase<"RingBusWrite", 16, "ring_bus_write">,
  I32EnumAttrCase<"RingBusProducerA", 17, "ring_bus_producer_a">,
  I32EnumAttrCase<"RingBusProducerB", 18, "ring_bus_producer_b">,
  I32EnumAttrCase<"NarrowToNarrowRead", 19, "narrow_to_narrow_read">,
  I32EnumAttrCase<"NarrowToNarrowWrite", 20, "narrow_to_narrow_write">,
  I32EnumAttrCase<"ActivationRead", 21, "activation_read">
]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::darwinn::isa";
}

def DarwinnIsa_CounterAttr : DarwinnIsa_Attr<"Counter", "counter"> {
  let parameters = (ins "int64_t":$end, "int64_t":$step, "bool":$mask);
  let genVerifyDecl = 1;
}

def DarwinnIsa_PrologueAttr : DarwinnIsa_Attr<"Prologue", "prologue"> {
  let parameters = (ins
    "unsigned":$loopId,
    "int64_t":$baseAddress,
    "uint64_t":$innerLimit,
    "uint64_t":$outerLimit,
    "uint64_t":$outerStride,
    "uint64_t":$accessBytes,
    EnumParameter<DarwinnIsa_PrologueTarget>:$prologueTarget
  );
  let genVerifyDecl = 1;
}

def DarwinnIsa_ByteAddressModeAttr : DarwinnIsa_Attr<"ByteAddressMode", "byte_address_mode"> {
  let parameters = (ins
    "uint32_t":$accessBytesLoopMap,
    "uint64_t":$lastAccessBytes,
    "uint64_t":$defaultAccessBytes
  );
  let genVerifyDecl = 1;
}

def DarwinnIsa_SyncProducerAttr : DarwinnIsa_Attr<"SyncProducer", "sync_producer"> {
  let parameters = (ins "bool":$increment, "unsigned":$syncFlagLoopDepth);
  let genVerifyDecl = 1;
}

def DarwinnIsa_SyncWatcherAttr : DarwinnIsa_Attr<"SyncWatcher", "sync_watcher"> {
  let parameters = (ins
    EnumParameter<DarwinnIsa_TileSyncFlag>:$syncFlag,
    "unsigned":$threadId,
    "bool":$bykjWtLmxpJcgmyqKq,
    "unsigned":$syncFlagLoopDepth,
    "int32_t":$initialExpectedSyncFlagValue,
    "uint32_t":$waitSyncFlagStride,
    "bool":$syncWaitValid
  );
  let genVerifyDecl = 1;
}

def DarwinnIsa_MainOperationAttr : DarwinnIsa_Attr<"MainOperation", "main_operation"> {
  let parameters = (ins
    DarwinnIsa_TraversalArray<"uint16_t">:$counter,
    "unsigned":$alternateInnerLimit,
    "unsigned":$alternateLimitLoopId
  );
  let genVerifyDecl = 1;
}

def DarwinnIsa_TraversalAttr : DarwinnIsa_Attr<"Traversal", "traversal"> {
  let parameters = (ins
    "uint64_t":$baseAddress,
    DarwinnIsa_TraversalArray<"::mlir::darwinn::isa::CounterAttr">:$counter,
    DarwinnIsa_TraversalArray<"::mlir::darwinn::isa::SyncProducerAttr">:$syncProducer,
    "unsigned":$doubleBufferLoop,
    "int64_t":$secondBufferOffset,
    "bool":$initializeInPongState,
    "int64_t":$alternateInnerLimit,
    "unsigned":$alternateLimitLoopId,
    OptionalParameter<"::mlir::darwinn::isa::PrologueAttr">:$prologue,
    DarwinnIsa_TraversalArray<"::mlir::darwinn::isa::PrologueAttr">:$additionalPrologues,
    OptionalParameter<"::mlir::darwinn::isa::ByteAddressModeAttr">:$byteAddressMode
  );
  let genVerifyDecl = 1;
}

#endif
