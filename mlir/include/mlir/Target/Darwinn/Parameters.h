#ifndef MLIR_TARGET_DARWINN_PARAMETERS_H
#define MLIR_TARGET_DARWINN_PARAMETERS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace mlir::darwinn {

class ConvolutionShape {
public:
  static llvm::Expected<ConvolutionShape>
  create(std::array<size_t, 4> dimensions);

  size_t getOutputChannels() const { return dimensions[0]; }
  size_t getKernelHeight() const { return dimensions[1]; }
  size_t getKernelWidth() const { return dimensions[2]; }
  size_t getInputChannels() const { return dimensions[3]; }
  size_t getElementCount() const { return elements; }

private:
  ConvolutionShape(std::array<size_t, 4> dimensions, size_t elements)
      : dimensions(dimensions), elements(elements) {}

  std::array<size_t, 4> dimensions;
  size_t elements;
};

class DepthwiseShape {
public:
  static llvm::Expected<DepthwiseShape>
  create(std::array<size_t, 3> dimensions);

  size_t getKernelElements() const { return kernelElements; }
  size_t getChannels() const { return channels; }
  size_t getElementCount() const { return elements; }

private:
  DepthwiseShape(size_t kernelElements, size_t channels, size_t elements)
      : kernelElements(kernelElements), channels(channels), elements(elements) {
  }

  size_t kernelElements;
  size_t channels;
  size_t elements;
};

struct ComposedConvolution {
  ConvolutionShape shape;
  llvm::SmallVector<float> filter;
  llvm::SmallVector<float> bias;
};

llvm::Expected<llvm::SmallVector<uint8_t>>
packConvolution(ConvolutionShape shape, llvm::ArrayRef<float> filter,
                std::optional<llvm::ArrayRef<float>> bias,
                size_t inputChannelTile);

llvm::Expected<llvm::SmallVector<uint8_t>>
packDepthwise(DepthwiseShape shape, llvm::ArrayRef<float> filter,
              std::optional<llvm::ArrayRef<float>> bias);

llvm::Expected<llvm::SmallVector<uint8_t>>
packBfloat(llvm::ArrayRef<float> values);

llvm::Expected<ComposedConvolution>
composePointwise(ConvolutionShape firstShape, llvm::ArrayRef<float> firstFilter,
                 std::optional<llvm::ArrayRef<float>> firstBias,
                 ConvolutionShape secondShape,
                 llvm::ArrayRef<float> secondFilter,
                 std::optional<llvm::ArrayRef<float>> secondBias);

}

#endif
