add_mlir_dialect_library(MLIRDarwinnDialect
  DarwinnOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Darwinn

  DEPENDS
  MLIRDarwinnOpsIncGen
  MLIRDarwinnAttributesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRDialect
  )
