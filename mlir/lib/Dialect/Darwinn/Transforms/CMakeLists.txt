set(LLVM_TARGET_DEFINITIONS DwcPasses.td)
mlir_tablegen(DwcPasses.h.inc -gen-pass-decls -name Dwc)
add_public_tablegen_target(MLIRDwcPassIncGen)

add_mlir_dialect_library(MLIRDwcTransforms
  DwcPasses.cpp

  DEPENDS
  MLIRDwcPassIncGen

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRPass
  )

