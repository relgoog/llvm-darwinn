#ifndef MLIR_DIALECT_DARWINN_IR_DARWINNTRAVERSALPLAN_TD
#define MLIR_DIALECT_DARWINN_IR_DARWINNTRAVERSALPLAN_TD

include "mlir/Dialect/Darwinn/IR/DarwinnDialect.td"

class Darwinn_PlanArray<string element> : ArrayRefParameter<element> {
  let parser =
      [{ [&]() -> ::mlir::FailureOr<::llvm::SmallVector<}]#element#[{>> {
    ::llvm::SmallVector<}]#element#[{> values;
    if ($_parser.parseCommaSeparatedList(::mlir::AsmParser::Delimiter::Square, [&]() {
      auto value = ::mlir::FieldParser<}]#element#[{>::parse($_parser);
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

class Darwinn_TraversalPlanAttr<string name, string spelling>
    : AttrDef<Darwinn_Dialect, name> {
  let mnemonic = spelling;
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

def Darwinn_LoopCounterAttr
    : Darwinn_TraversalPlanAttr<"LoopCounter", "loop_counter"> {
  let parameters = (ins "int64_t":$tripCount, "int64_t":$byteStride);
}

def Darwinn_ByteAccessPlanAttr
    : Darwinn_TraversalPlanAttr<"ByteAccessPlan", "byte_access_plan"> {
  let parameters = (ins "uint32_t":$loopMap, "int64_t":$lastBytes,
      "int64_t":$defaultBytes);
}

def Darwinn_MemoryTraversalPlanAttr
    : Darwinn_TraversalPlanAttr<"MemoryTraversalPlan",
                                "memory_traversal_plan"> {
  let parameters =
      (ins Darwinn_PlanArray<"::mlir::darwinn::LoopCounterAttr">:$counters,
          OptionalParameter<"::mlir::darwinn::ByteAccessPlanAttr">:$access);
}

def Darwinn_TensorTraversalPlanAttr
    : Darwinn_TraversalPlanAttr<"TensorTraversalPlan",
                                "tensor_traversal_plan"> {
  let parameters = (ins Darwinn_PlanArray<"int64_t">:$mainTripCounts,
      "::mlir::darwinn::MemoryTraversalPlanAttr":$narrowRead,
      "::mlir::darwinn::MemoryTraversalPlanAttr":$narrowWrite,
      OptionalParameter<"::mlir::darwinn::MemoryTraversalPlanAttr">:$parameters,
      "::mlir::darwinn::MemoryTraversalPlanAttr":$sums);
}

#endif
