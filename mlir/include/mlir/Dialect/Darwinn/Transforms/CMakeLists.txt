set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Dwc)
add_mlir_dialect_tablegen_target(MLIRDarwinnTransformsIncGen)
add_mlir_doc(Passes DarwinnPasses Dialects/ -gen-pass-doc)
