add_mlir_library(MLIRDarwinnEncoding
  PARTIAL_SOURCES_INTENDED
  Compression.cpp
  Dma.cpp
  Encoding.cpp
  Memory.cpp
  Parameters.cpp
  Serialization.cpp
  Tensor.cpp
  Transport.cpp
  Traversal.cpp

  LINK_COMPONENTS
  Support
)
