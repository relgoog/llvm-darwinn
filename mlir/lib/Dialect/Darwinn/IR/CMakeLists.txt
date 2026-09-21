add_mlir_dialect_library(MLIRDarwinnDialect
  PARTIAL_SOURCES_INTENDED
  DarwinnComputeAttrs.cpp
  DarwinnComputeOps.cpp
  DarwinnDistributed.cpp
  DarwinnOps.cpp
  DarwinnScheduledAttrs.cpp
  DarwinnStorageOps.cpp
  DarwinnTraversalPlan.cpp
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
  MLIRQuantDialect
  )

add_mlir_dialect_library(MLIRDarwinnISADialect
  PARTIAL_SOURCES_INTENDED
  InstructionDma.cpp
  InstructionOps.cpp
  InstructionTensor.cpp
  InstructionTransport.cpp
  InstructionTraversal.cpp

  DEPENDS
  MLIRInstructionOpsIncGen
  MLIRDarwinnISAAttrsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)
