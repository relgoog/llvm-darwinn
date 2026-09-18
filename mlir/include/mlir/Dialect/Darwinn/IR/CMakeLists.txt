add_mlir_dialect(DarwinnOps darwinn)
add_mlir_doc(DarwinnOps DarwinnOps Dialects/ -gen-op-doc -dialect=darwinn)

set(LLVM_TARGET_DEFINITIONS DarwinnDialect.td)
mlir_tablegen(DarwinnAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=darwinn)
mlir_tablegen(DarwinnAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=darwinn)
mlir_tablegen(DarwinnTypes.h.inc -gen-typedef-decls -typedefs-dialect=darwinn)
mlir_tablegen(DarwinnTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=darwinn)
add_mlir_dialect_tablegen_target(MLIRDarwinnAttributesIncGen)
