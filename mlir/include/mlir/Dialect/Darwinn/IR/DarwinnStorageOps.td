#ifndef MLIR_DIALECT_DARWINN_IR_DARWINNSTORAGEOPS_TD
#define MLIR_DIALECT_DARWINN_IR_DARWINNSTORAGEOPS_TD

include "mlir/Dialect/Darwinn/IR/DarwinnDialect.td"
include "mlir/Dialect/Darwinn/IR/DarwinnScheduledAttrs.td"

class Darwinn_StorageOp<string mnemonic> : Darwinn_Op<mnemonic> {
  let hasVerifier = 1;
  let assemblyFormat =
      "operands prop-dict attr-dict `:` functional-type(operands, results)";
}

def Darwinn_CreateEmptyTensorOp : Darwinn_StorageOp<"create_empty_tensor"> {
  let arguments = (ins AffineMapAttr:$slicing_begins,
      I32ArrayAttr:$slicing_domain, AffineMapAttr:$slicing_ends);
  let results = (outs Darwinn_DistributedTensor:$output);
}

def Darwinn_FillOp : Darwinn_StorageOp<"fill"> {
  let arguments = (ins ElementsAttr:$value,
      OptionalAttr<Darwinn_ConstTypeAttr>:$const_type,
      AffineMapAttr:$slicing_begins, I32ArrayAttr:$slicing_domain,
      AffineMapAttr:$slicing_ends);
  let results = (outs Darwinn_DistributedTensor:$output);
}

def Darwinn_GetTensorOp : Darwinn_StorageOp<"get_tensor"> {
  let arguments = (ins Darwinn_DistributedTensor:$input);
  let results = (outs Darwinn_DistributedTensor:$output);
}

def Darwinn_ReshapeOpOp : Darwinn_StorageOp<"reshape_op"> {
  let arguments = (ins Darwinn_DistributedTensor:$input,
      OptionalAttr<AffineMapAttr>:$forward_index_transformation,
      OptionalAttr<AffineMapAttr>:$reverse_index_transformation);
  let results = (outs Darwinn_DistributedTensor:$output);
}

def Darwinn_CopyOpOp : Darwinn_StorageOp<"copy_op"> {
  let arguments = (ins Darwinn_DistributedTensor:$input,
      AffineMapAttr:$forward_index_transformation,
      AffineMapAttr:$reverse_index_transformation,
      AffineMapAttr:$slicing_begins, I32ArrayAttr:$slicing_domain,
      AffineMapAttr:$slicing_ends, AffineMapAttr:$traversal);
  let results = (outs Darwinn_DistributedTensor:$output);
}

def Darwinn_PreemptionPointOp : Darwinn_StorageOp<"preemption_point"> {
  let arguments = (ins BoolAttr:$hardware_preemption_point);
  let results = (outs);
  let hasVerifier = 0;
}

def Darwinn_DistributedCreateViewOp
    : Darwinn_StorageOp<"distributed.create_view"> {
  let arguments = (ins Darwinn_DistributedTensor:$input,
      OptionalAttr<AffineMapAttr>:$forward_index_transformation,
      OptionalAttr<AffineMapAttr>:$reverse_index_transformation);
  let results = (outs Darwinn_DistributedView:$output);
}

def Darwinn_RedistributeOp : Darwinn_StorageOp<"redistribute"> {
  let arguments = (ins AnyTypeOf<[Darwinn_DistributedTensor,
                                  Darwinn_DistributedView]>:$input,
      Optional<Darwinn_WriteView>:$destination,
      OptionalAttr<Darwinn_MappingAttr>:$mapping, AffineMapAttr:$slicing_begins,
      I32ArrayAttr:$slicing_domain, AffineMapAttr:$slicing_ends);
  let results =
      (outs AnyTypeOf<[Darwinn_DistributedTensor, Darwinn_FilledView]>:$output);
}

def Darwinn_CommunicatedCreateEmptyTensorOp
    : Darwinn_StorageOp<"communicated.create_empty_tensor"> {
  let arguments = (ins AffineMapAttr:$slicing_begins,
      I32ArrayAttr:$slicing_domain, AffineMapAttr:$slicing_ends);
  let results = (outs Darwinn_EmptyTensor:$output);
}

def Darwinn_CommunicatedCreateWriteViewOp
    : Darwinn_StorageOp<"communicated.create_write_view"> {
  let arguments = (ins Darwinn_EmptyTensor:$input,
      AffineMapAttr:$reverse_index_transformation);
  let results = (outs Darwinn_WriteView:$output);
}

def Darwinn_CommunicatedJoinViewsOp
    : Darwinn_StorageOp<"communicated.join_views"> {
  let arguments = (ins Variadic<Darwinn_FilledView>:$inputs);
  let results = (outs Darwinn_DistributedTensor:$output);
}

#endif
