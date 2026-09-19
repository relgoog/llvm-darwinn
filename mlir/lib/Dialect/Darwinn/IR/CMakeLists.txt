add_mlir_dialect_library(MLIRDarwinnDialect
  DarwinnOps.cpp
  DwcOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Darwinn

  DEPENDS
  MLIRDarwinnOpsIncGen
  MLIRDwcOpsIncGen
  MLIRDwcAttrsEnumsGen
  MLIRDarwinnAttributesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRDialect
  )
