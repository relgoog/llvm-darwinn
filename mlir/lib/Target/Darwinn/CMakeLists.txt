add_mlir_library(MLIRDarwinnEncoding
  PARTIAL_SOURCES_INTENDED
  Allocation.cpp
  Compression.cpp
  Dma.cpp
  Encoding.cpp
  ExecutionPlan.cpp
  Memory.cpp
  ParameterLayout.cpp
  Parameters.cpp
  Serialization.cpp
  Tensor.cpp
  Transport.cpp
  Traversal.cpp

  LINK_COMPONENTS
  Support
)

add_mlir_translation_library(MLIRDarwinnToBinary
  PARTIAL_SOURCES_INTENDED
  DmaAttributes.cpp
  EncodingAttributes.cpp
  TensorAttributes.cpp
  Translation.cpp
  TransportAttributes.cpp
  TraversalAttributes.cpp

  LINK_LIBS PUBLIC
  MLIRDarwinnEncoding
  MLIRDarwinnISADialect
  MLIRIR
  MLIRTranslateLib
)

add_mlir_translation_library(MLIRDarwinnToParameters
  PARTIAL_SOURCES_INTENDED
  ParameterTranslation.cpp

  LINK_LIBS PUBLIC
  MLIRDarwinnEncoding
  MLIRDarwinnDialect
  MLIRFuncDialect
  MLIRIR
  MLIRTranslateLib
)
