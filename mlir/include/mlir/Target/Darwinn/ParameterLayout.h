#ifndef MLIR_TARGET_DARWINN_PARAMETERLAYOUT_H
#define MLIR_TARGET_DARWINN_PARAMETERLAYOUT_H

#include "mlir/Target/Darwinn/Parameters.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>

namespace mlir::darwinn {

enum class ConvolutionLoopAxis {
  Batch,
  OutputY,
  OutputX,
  KernelY,
  KernelX,
  InputChannel,
  OutputChannel
};

struct ConvolutionLoop {
  ConvolutionLoopAxis axis;
  uint64_t begin;
  uint64_t endExclusive;
  uint64_t step;
};

llvm::Expected<size_t> projectConvolutionInputChannelTile(
    ConvolutionShape shape, llvm::ArrayRef<ConvolutionLoop> innerToOuter);

}

#endif
