#ifndef MLIR_DARWINN_INSTRUCTION_DIALECT
#define MLIR_DARWINN_INSTRUCTION_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

def DarwinnIsa_Dialect : Dialect {
  let name = "darwinn_isa";
  let cppNamespace = "::mlir::darwinn::isa";
  let useDefaultAttributePrinterParser = 1;
}

class DarwinnIsa_Op<string mnemonic, list<Trait> traits = []>
    : Op<DarwinnIsa_Dialect, mnemonic, traits>;

class DarwinnIsa_Attr<string name, string attrMnemonic>
    : AttrDef<DarwinnIsa_Dialect, name> {
  let mnemonic = attrMnemonic;
  let assemblyFormat = "`<` struct(params) `>`";
}

class DarwinnIsa_Enum<string name, list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, name, cases> {
  let cppNamespace = "::mlir::darwinn::isa";
  let genSpecializedAttr = 0;
}

class DarwinnIsa_Instruction<string mnemonic>
    : DarwinnIsa_Op<mnemonic, [HasParent<"FragmentOp">]> {
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

#endif
