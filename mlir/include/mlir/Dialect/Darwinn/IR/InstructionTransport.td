#ifndef MLIR_DIALECT_DARWINN_IR_INSTRUCTIONTRANSPORT_TD
#define MLIR_DIALECT_DARWINN_IR_INSTRUCTIONTRANSPORT_TD

include "mlir/Dialect/Darwinn/IR/InstructionDialect.td"
include "mlir/Dialect/Darwinn/IR/InstructionTraversal.td"

def DarwinnIsa_DmaQueue : DarwinnIsa_Enum<"DmaQueue", [
    I32EnumAttrCase<"Instruction", 0, "instruction">,
    I32EnumAttrCase<"Activation", 1, "activation">,
    I32EnumAttrCase<"Parameter", 2, "parameter">,
    I32EnumAttrCase<"Output", 3, "output">]>;

def DarwinnIsa_DmaQueueAttr : EnumAttr<
    DarwinnIsa_Dialect, DarwinnIsa_DmaQueue, "dma_queue">;

def DarwinnIsa_InputFifo : DarwinnIsa_Enum<"InputFifo", [
    I32EnumAttrCase<"Parameter", 0, "parameter">,
    I32EnumAttrCase<"Activation", 1, "activation">]>;

def DarwinnIsa_InputFifoAttr : EnumAttr<
    DarwinnIsa_Dialect, DarwinnIsa_InputFifo, "input_fifo">;

def DarwinnIsa_FieldSource : DarwinnIsa_Enum<"FieldSource", [
    I32EnumAttrCase<"Instruction", 0, "instruction">,
    I32EnumAttrCase<"ScalarRegister", 1, "scalar_register">]>;

def DarwinnIsa_FieldSourceAttr : EnumAttr<
    DarwinnIsa_Dialect, DarwinnIsa_FieldSource, "field_source">;

def DarwinnIsa_ScalarSyncFlag : DarwinnIsa_Enum<"ScalarSyncFlag", [
    I32EnumAttrCase<"ActivationPop", 0, "activation_pop">,
    I32EnumAttrCase<"ParameterPop", 1, "parameter_pop">,
    I32EnumAttrCase<"ActivationInfeed", 2, "activation_infeed">,
    I32EnumAttrCase<"ParameterInfeed", 3, "parameter_infeed">,
    I32EnumAttrCase<"ScalarInfeed", 4, "scalar_infeed">,
    I32EnumAttrCase<"RingProducerA", 5, "ring_producer_a">,
    I32EnumAttrCase<"RingProducerB", 6, "ring_producer_b">,
    I32EnumAttrCase<"RingOutfeed", 7, "ring_outfeed">,
    I32EnumAttrCase<"Pipeline", 8, "pipeline">,
    I32EnumAttrCase<"Software0", 9, "software0">,
    I32EnumAttrCase<"Software1", 10, "software1">,
    I32EnumAttrCase<"Software2", 11, "software2">,
    I32EnumAttrCase<"Software3", 12, "software3">,
    I32EnumAttrCase<"Software4", 13, "software4">,
    I32EnumAttrCase<"Software5", 14, "software5">,
    I32EnumAttrCase<"Software6", 15, "software6">,
    I32EnumAttrCase<"Software7", 16, "software7">]>;

def DarwinnIsa_ScalarSyncWatcherAttr : DarwinnIsa_Attr<
    "ScalarSyncWatcher", "scalar_sync_watcher"> {
  let parameters = (ins EnumParameter<DarwinnIsa_ScalarSyncFlag>:$flag,
      "unsigned":$loopDepth, "uint32_t":$initialValue,
      "uint32_t":$stride, "bool":$valid);
  let genVerifyDecl = 1;
}

def DarwinnIsa_FieldSourcesAttr : DarwinnIsa_Attr<"FieldSources", "field_sources"> {
  let parameters = (ins DarwinnIsa_TraversalArray<
      "::mlir::darwinn::isa::FieldSourceAttr">:$values);
  let genVerifyDecl = 1;
}

def DarwinnIsa_InfeedConversion : DarwinnIsa_Enum<"InfeedConversion", [
    I32EnumAttrCase<"None", 0, "none">,
    I32EnumAttrCase<"Unsigned8ToBfloat", 1, "unsigned8_to_bfloat">]>;

def DarwinnIsa_InfeedConversionAttr : DarwinnIsa_Attr<
    "InfeedConversion", "infeed_conversion"> {
  let parameters = (ins EnumParameter<DarwinnIsa_InfeedConversion>:$mode,
      "unsigned":$zeroPoint);
  let genVerifyDecl = 1;
}

def DarwinnIsa_HibDmaOp : DarwinnIsa_Instruction<"hib_dma"> {
  let arguments = (ins DarwinnIsa_HeaderAttr:$header, I64Attr:$tag,
      DarwinnIsa_DmaQueueAttr:$queue, DarwinnIsa_TraversalAttr:$traversal);
}

def DarwinnIsa_PopInputOp : DarwinnIsa_Instruction<"pop_input"> {
  let arguments = (ins DarwinnIsa_HeaderAttr:$header, I64Attr:$tag,
      DarwinnIsa_InputFifoAttr:$fifo, DarwinnIsa_TraversalAttr:$traversal,
      OptionalAttr<DarwinnIsa_ScalarSyncWatcherAttr>:$watcher);
}

def DarwinnIsa_RingInfeedOp : DarwinnIsa_Instruction<"ring_infeed"> {
  let arguments = (ins DarwinnIsa_HeaderAttr:$header, I64Attr:$tag,
      DarwinnIsa_InputFifoAttr:$fifo, DarwinnIsa_TraversalAttr:$traversal,
      OptionalAttr<DarwinnIsa_ScalarSyncWatcherAttr>:$watcher0,
      OptionalAttr<DarwinnIsa_ScalarSyncWatcherAttr>:$watcher1,
      I64Attr:$sync_increment, DenseBoolArrayAttr:$virtual_channels,
      DenseBoolArrayAttr:$targets, DarwinnIsa_FieldSourceAttr:$bitmap_source,
      DarwinnIsa_FieldSourceAttr:$base_address_source,
      DarwinnIsa_FieldSourcesAttr:$counter_sources,
      DarwinnIsa_InfeedConversionAttr:$conversion);
}

def DarwinnIsa_RingOutfeedOp : DarwinnIsa_Instruction<"ring_outfeed"> {
  let arguments = (ins DarwinnIsa_HeaderAttr:$header, I64Attr:$tag,
      DarwinnIsa_TraversalAttr:$traversal,
      OptionalAttr<DarwinnIsa_ScalarSyncWatcherAttr>:$watcher,
      DarwinnIsa_FieldSourceAttr:$base_address_source,
      DarwinnIsa_FieldSourcesAttr:$counter_sources,
      DenseBoolArrayAttr:$virtual_channels, I64Attr:$bytes_to_pop);
}

#endif
