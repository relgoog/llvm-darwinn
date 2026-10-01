add_mlir_dialect_library(MLIRDarwinnCodegen
  Allocation.cpp
  Builders.cpp
  Compute.cpp
  Convolution.cpp
  Gather.cpp
  Generate.cpp
  Geometry.cpp
  Interpolate.cpp
  LegalizeDwc.cpp
  Movement.cpp
  Permute.cpp
  Program.cpp
  Schedule.cpp
  Transfer.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Darwinn/Codegen

  LINK_LIBS PUBLIC
  MLIRDarwinnDialect
  MLIRDarwinnEncoding
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  )
