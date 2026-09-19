set(LLVM_TARGET_DEFINITIONS DwcPasses.td)
mlir_tablegen(DwcPasses.h.inc -gen-pass-decls -name Dwc)
add_mlir_dialect_tablegen_target(MLIRDwcPassIncGen)

add_mlir_dialect_library(MLIRDwcTransforms
  DwcPasses.cpp
  LowerConvert.cpp
  LowerCopySlice.cpp

  DEPENDS
  MLIRDwcPassIncGen

  LINK_LIBS PUBLIC
  MLIRControlFlowToLLVM
  MLIRDiveVmDialect
  MLIRFuncDialect
  MLIRFuncToLLVM
  MLIRLinalgTransforms
  MLIRMathToLLVM
  MLIRMathToLibm
  MLIRPass
  MLIRSCFToControlFlow
  MLIRTransformUtils
  )

