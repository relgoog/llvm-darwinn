add_mlir_dialect_library(MLIRDiveVmDialect
  DiveVmOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/DiveVm

  DEPENDS
  MLIRDiveVmOpsIncGen
  MLIREdgeTpuOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRDialect
  )
