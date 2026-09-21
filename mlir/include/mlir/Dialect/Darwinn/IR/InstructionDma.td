#ifndef MLIR_DIALECT_DARWINN_IR_INSTRUCTIONDMA_TD
#define MLIR_DIALECT_DARWINN_IR_INSTRUCTIONDMA_TD

include "mlir/Dialect/Darwinn/IR/InstructionDialect.td"
include "mlir/Dialect/Darwinn/IR/InstructionTraversal.td"

def DarwinnIsa_RingConsumerId
    : DarwinnIsa_Enum<
          "RingConsumerId", [I32EnumAttrCase<"ConsumerA", 0, "consumer_a">,
                             I32EnumAttrCase<"ConsumerB", 1, "consumer_b">]>;

def DarwinnIsa_RingDestination
    : DarwinnIsa_Enum<"RingDestination",
                      [I32EnumAttrCase<"NarrowMemory", 0, "narrow_memory">,
                       I32EnumAttrCase<"WideMemory", 1, "wide_memory">]>;

def DarwinnIsa_MeshDirection
    : DarwinnIsa_Enum<
          "MeshDirection", [I32EnumAttrCase<"OutboundNorthInboundSouth", 0,
                                            "outbound_north_inbound_south">,
                            I32EnumAttrCase<"OutboundEastInboundWest", 1,
                                            "outbound_east_inbound_west">,
                            I32EnumAttrCase<"OutboundWestInboundEast", 2,
                                            "outbound_west_inbound_east">,
                            I32EnumAttrCase<"OutboundSouthInboundNorth", 3,
                                            "outbound_south_inbound_north">]>;

def DarwinnIsa_MeshReduction
    : DarwinnIsa_Enum<
          "MeshReduction", [I32EnumAttrCase<"None", 0, "none">,
                            I32EnumAttrCase<"Forward", 1, "forward">,
                            I32EnumAttrCase<"Consume", 2, "consume">]>;

def DarwinnIsa_ByteFilterAttr : DarwinnIsa_Attr<"ByteFilter", "byte_filter"> {
  let parameters = (ins "unsigned":$firstDiscardByteLoopMap,
      "unsigned":$discardByteLoopMap, "uint32_t":$firstDiscardByteCount,
      "uint32_t":$discardByteCount, "uint32_t":$lastDiscardByteCount);
  let genVerifyDecl = 1;
}

def DarwinnIsa_DmaWatcherAttr : DarwinnIsa_Attr<"DmaWatcher", "dma_watcher"> {
  let parameters = (ins "::mlir::darwinn::isa::SyncWatcherAttr":$watcher,
      "bool":$stallTtuId);
  let genVerifyDecl = 1;
}

def DarwinnIsa_RingConsumerAttr
    : DarwinnIsa_Attr<"RingConsumer", "ring_consumer"> {
  let parameters = (ins EnumParameter<DarwinnIsa_RingConsumerId>:$consumer,
      "::mlir::darwinn::isa::TraversalAttr":$traversal,
      "bool":$baseAddressOverride,
      DarwinnIsa_TraversalArray<
                        "::mlir::darwinn::isa::DmaWatcherAttr">:$watchers,
      "::mlir::DenseBoolArrayAttr":$virtualChannelSubscription,
      EnumParameter<DarwinnIsa_RingDestination>:$destination,
      "::mlir::darwinn::isa::ByteFilterAttr":$filter,
      "::mlir::DenseBoolArrayAttr":$threadMulticastBitmap);
  let genVerifyDecl = 1;
}

def DarwinnIsa_RingProducerAttr
    : DarwinnIsa_Attr<"RingProducer", "ring_producer"> {
  let parameters = (ins "::mlir::darwinn::isa::TraversalAttr":$traversal,
      "bool":$baseAddressOverride,
      DarwinnIsa_TraversalArray<
                        "::mlir::darwinn::isa::DmaWatcherAttr">:$watchers,
      "::mlir::DenseBoolArrayAttr":$virtualChannelSubscription,
      "::mlir::DenseBoolArrayAttr":$targets,
      EnumParameter<DarwinnIsa_RingConsumerId>:$consumer);
  let genVerifyDecl = 1;
}

def DarwinnIsa_MeshAttr : DarwinnIsa_Attr<"Mesh", "mesh"> {
  let parameters = (ins EnumParameter<DarwinnIsa_MeshDirection>:$direction,
      "::mlir::darwinn::isa::TraversalAttr":$read,
      "::mlir::darwinn::isa::TraversalAttr":$write, "bool":$baseAddressOverride,
      DarwinnIsa_TraversalArray<
                        "::mlir::darwinn::isa::DmaWatcherAttr">:$readWatchers,
      DarwinnIsa_TraversalArray<
                        "::mlir::darwinn::isa::DmaWatcherAttr">:$writeWatchers,
      "::mlir::DenseI8ArrayAttr":$immediateValue, "unsigned":$validBytes,
      "bool":$forwardingMode,
      EnumParameter<DarwinnIsa_MeshReduction>:$reduction,
      "bool":$dataTypeFloat);
  let genVerifyDecl = 1;
}

def DarwinnIsa_WideToNarrowAttr
    : DarwinnIsa_Attr<"WideToNarrow", "wide_to_narrow"> {
  let parameters = (ins "::mlir::darwinn::isa::TraversalAttr":$read,
      "::mlir::darwinn::isa::TraversalAttr":$write, "bool":$baseAddressOverride,
      DarwinnIsa_TraversalArray<
                        "::mlir::darwinn::isa::DmaWatcherAttr">:$readWatchers,
      DarwinnIsa_TraversalArray<
                        "::mlir::darwinn::isa::DmaWatcherAttr">:$writeWatchers,
      "bool":$transpose, "bool":$doubleOperandMode,
      "unsigned":$wideMemoryLoadStoreLoopId, "bool":$isScalingFactorBias,
      "unsigned":$zInBundleValidCount,
      "::mlir::DenseBoolArrayAttr":$threadMulticastBitmap);
  let genVerifyDecl = 1;
}

def DarwinnIsa_WideByteAddressModeAttr
    : DarwinnIsa_Attr<"WideByteAddressMode", "wide_byte_address_mode"> {
  let parameters = (ins "unsigned":$strideUnitGranulesLoopMap,
      "unsigned":$defaultStrideUnitGranules, "unsigned":$lastStrideUnitGranules,
      "unsigned":$cellStride, "unsigned":$cellStrideGroupCountLoopMap,
      "unsigned":$defaultCellStrideGroupCount,
      "unsigned":$lastCellStrideGroupCount, "uint32_t":$immediateValue);
  let genVerifyDecl = 1;
}

def DarwinnIsa_NarrowToWideAttr
    : DarwinnIsa_Attr<"NarrowToWide", "narrow_to_wide"> {
  let parameters = (ins "::mlir::darwinn::isa::TraversalAttr":$read,
      "::mlir::darwinn::isa::TraversalAttr":$write, "bool":$baseAddressOverride,
      DarwinnIsa_TraversalArray<
                        "::mlir::darwinn::isa::DmaWatcherAttr">:$readWatchers,
      DarwinnIsa_TraversalArray<
                        "::mlir::darwinn::isa::DmaWatcherAttr">:$writeWatchers,
      "::mlir::darwinn::isa::WideByteAddressModeAttr":$byteAddress,
      "::mlir::DenseBoolArrayAttr":$threadMulticastBitmap);
  let genVerifyDecl = 1;
}

class DarwinnIsa_DmaInstruction<string mnemonic, Attr config>
    : DarwinnIsa_Instruction<mnemonic> {
  let arguments = (ins DarwinnIsa_TileHeaderAttr:$header,
      DarwinnIsa_OverwriteInfoAttr:$overwrite,
      DenseBoolArrayAttr:$register_sourced_operands, config:$config);
}

def DarwinnIsa_RingConsumerOp
    : DarwinnIsa_DmaInstruction<"ring_consumer", DarwinnIsa_RingConsumerAttr>;
def DarwinnIsa_RingProducerOp
    : DarwinnIsa_DmaInstruction<"ring_producer", DarwinnIsa_RingProducerAttr>;
def DarwinnIsa_MeshOp : DarwinnIsa_DmaInstruction<"mesh", DarwinnIsa_MeshAttr>;
def DarwinnIsa_WideToNarrowOp
    : DarwinnIsa_DmaInstruction<"wide_to_narrow", DarwinnIsa_WideToNarrowAttr>;
def DarwinnIsa_NarrowToWideOp
    : DarwinnIsa_DmaInstruction<"narrow_to_wide", DarwinnIsa_NarrowToWideAttr>;

#endif
