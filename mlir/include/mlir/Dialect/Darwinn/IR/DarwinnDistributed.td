#ifndef MLIR_DIALECT_DARWINN_IR_DARWINNDISTRIBUTED_TD
#define MLIR_DIALECT_DARWINN_IR_DARWINNDISTRIBUTED_TD

include "mlir/Dialect/Darwinn/IR/DarwinnDialect.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

class Darwinn_DistributedType<string name, string mnemonic>
    : Darwinn_Type<name, mnemonic, [ShapedTypeInterface]> {
  let parameters = (ins ArrayRefParameter<"int64_t">:$shape,
      "::mlir::Type":$elementType,
      "::mlir::darwinn::DistributedMemorySpace":$memorySpace);
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    bool hasRank() const { return true; }

    ::mlir::ShapedType cloneWith(
        ::std::optional<::llvm::ArrayRef<int64_t>> shape,
        ::mlir::Type elementType) const {
      return get(getContext(), shape.value_or(getShape()), elementType,
                 getMemorySpace());
    }
  }];
}

def Darwinn_DistributedTensor
    : Darwinn_DistributedType<"DistributedTensor", "distributed_tensor">;

def Darwinn_DistributedView
    : Darwinn_DistributedType<"DistributedView", "distributed_view">;

def Darwinn_EmptyTensor : Darwinn_DistributedType<"EmptyTensor", "EmptyTensor">;

def Darwinn_WriteView : Darwinn_DistributedType<"WriteView", "WriteView">;

def Darwinn_FilledView : Darwinn_DistributedType<"FilledView", "FilledView">;

#endif
