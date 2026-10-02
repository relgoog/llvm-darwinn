#include "mlir/Target/Darwinn/Parameters.h"

#include "llvm/ADT/bit.h"
#include "llvm/Support/Endian.h"

#include <algorithm>
#include <limits>

using namespace mlir::darwinn;

namespace {

llvm::Expected<size_t> elementCount(llvm::ArrayRef<size_t> dimensions) {
  size_t count = 1;

  for (size_t dimension : dimensions) {
    if (dimension == 0)
      return llvm::createStringError("Weight dimensions must be positive");
    if (count > std::numeric_limits<size_t>::max() / dimension)
      return llvm::createStringError("Weight element count overflow");
    count *= dimension;
  }

  return count;
}

llvm::Expected<size_t> alignedCount(size_t count, size_t alignment) {
  size_t padding = (alignment - count % alignment) % alignment;
  if (count > std::numeric_limits<size_t>::max() - padding)
    return llvm::createStringError("Weight alignment overflow");
  return count + padding;
}

llvm::Expected<llvm::SmallVector<uint8_t>> packedBuffer(size_t elements,
                                                        size_t biasChannels) {
  constexpr size_t maximum = std::numeric_limits<size_t>::max();
  if (elements > maximum / 2 || biasChannels > maximum / 4)
    return llvm::createStringError("Packed weight size overflow");
  size_t weightBytes = elements * 2;
  size_t biasBytes = biasChannels * 4;
  if (weightBytes > maximum - biasBytes)
    return llvm::createStringError("Packed weight size overflow");

  llvm::SmallVector<uint8_t> output;
  if (weightBytes + biasBytes > output.max_size())
    return llvm::createStringError(
        "Packed weights exceed the buffer size limit");
  output.reserve(weightBytes + biasBytes);
  return output;
}

llvm::Expected<std::optional<llvm::ArrayRef<float>>>
validateValues(llvm::ArrayRef<float> filter, size_t expectedElements,
               std::optional<llvm::ArrayRef<float>> bias,
               size_t expectedChannels) {
  if (filter.size() != expectedElements)
    return llvm::createStringError("Expected %zu filter values, received %zu",
                                   expectedElements, filter.size());
  if (!bias)
    return std::nullopt;
  if (bias->size() != expectedChannels)
    return llvm::createStringError("Expected %zu bias values, received %zu",
                                   expectedChannels, bias->size());
  uint32_t firstBits = llvm::bit_cast<uint32_t>(bias->front());
  if ((firstBits & 0x7FFFFFFF) == 0 &&
      std::all_of(bias->begin(), bias->end(), [firstBits](float value) {
        return llvm::bit_cast<uint32_t>(value) == firstBits;
      }))
    return std::nullopt;
  return bias;
}

uint16_t bfloatBits(float value) {
  uint32_t bits = llvm::bit_cast<uint32_t>(value);
  uint16_t sign = static_cast<uint16_t>(bits >> 16) & 0x8000;
  uint32_t magnitude = bits & 0x7FFFFFFF;
  if (magnitude > 0x7F800000)
    return sign | 0x7FC0;
  uint32_t rounding = 0x7FFF + ((magnitude >> 16) & 1);
  uint16_t rounded = static_cast<uint16_t>((magnitude + rounding) >> 16);
  return sign | std::min<uint16_t>(rounded, 0x7F7F);
}

uint32_t biasBits(float value) {
  uint32_t bits = llvm::bit_cast<uint32_t>(value);
  if ((bits & 0x7FFFFFFF) == 0x7F800000)
    return (bits & 0x80000000) | 0x7F7FFFFF;
  return bits;
}

}

llvm::Expected<ConvolutionShape>
ConvolutionShape::create(std::array<size_t, 4> dimensions) {
  llvm::Expected<size_t> elements = elementCount(dimensions);
  if (!elements)
    return elements.takeError();
  return ConvolutionShape(dimensions, *elements);
}

llvm::Expected<DepthwiseShape>
DepthwiseShape::create(std::array<size_t, 3> dimensions) {
  llvm::Expected<size_t> elements = elementCount(dimensions);
  if (!elements)
    return elements.takeError();
  return DepthwiseShape(dimensions[0] * dimensions[1], dimensions[2],
                        *elements);
}

llvm::Expected<llvm::SmallVector<uint8_t>> mlir::darwinn::packConvolution(
    ConvolutionShape shape, llvm::ArrayRef<float> filter,
    std::optional<llvm::ArrayRef<float>> bias, size_t inputChannelTile) {
  if (inputChannelTile == 0 || inputChannelTile % 2 != 0)
    return llvm::createStringError(
        "Input channel tile must be positive and even");

  auto validatedBias = validateValues(filter, shape.getElementCount(), bias,
                                      shape.getOutputChannels());
  if (!validatedBias)
    return validatedBias.takeError();
  llvm::Expected<size_t> paddedOutputs = alignedCount(
      shape.getOutputChannels(), shape.getOutputChannels() > 32 ? 32 : 8);
  if (!paddedOutputs)
    return paddedOutputs.takeError();
  llvm::Expected<size_t> paddedInputs =
      alignedCount(shape.getInputChannels(), inputChannelTile);
  if (!paddedInputs)
    return paddedInputs.takeError();
  llvm::Expected<size_t> packedElements =
      elementCount({*paddedOutputs, shape.getKernelHeight(),
                    shape.getKernelWidth(), *paddedInputs});
  if (!packedElements)
    return packedElements.takeError();
  auto output =
      packedBuffer(*packedElements, *validatedBias ? *paddedOutputs : 0);
  if (!output)
    return output.takeError();

  for (size_t outputStart = 0; outputStart < *paddedOutputs;) {
    size_t outputEnd =
        outputStart + std::min<size_t>(*paddedOutputs - outputStart, 32);

    if (*validatedBias) {
      llvm::ArrayRef<float> values = **validatedBias;
      for (size_t channel = outputStart; channel < outputEnd; ++channel) {
        float value = channel < values.size() ? values[channel] : 0.0f;
        uint8_t encoded[4];
        llvm::support::endian::write32le(encoded, biasBits(value));
        output->append(encoded, encoded + 4);
      }
    }

    for (size_t tileStart = 0; tileStart < *paddedInputs;
         tileStart += inputChannelTile) {
      for (size_t kernelY = 0; kernelY < shape.getKernelHeight(); ++kernelY) {
        for (size_t kernelX = 0; kernelX < shape.getKernelWidth(); ++kernelX) {
          for (size_t inputStart = tileStart;
               inputStart < tileStart + inputChannelTile; inputStart += 2) {
            for (size_t outputChannel = outputStart; outputChannel < outputEnd;
                 ++outputChannel) {
              for (size_t inputChannel = inputStart;
                   inputChannel < inputStart + 2; ++inputChannel) {
                float value = 0.0f;
                if (outputChannel < shape.getOutputChannels() &&
                    inputChannel < shape.getInputChannels()) {
                  size_t index =
                      ((outputChannel * shape.getKernelHeight() + kernelY) *
                           shape.getKernelWidth() +
                       kernelX) *
                          shape.getInputChannels() +
                      inputChannel;
                  value = filter[index];
                }
                uint8_t encoded[2];
                llvm::support::endian::write16le(encoded, bfloatBits(value));
                output->append(encoded, encoded + 2);
              }
            }
          }
        }
      }
    }

    outputStart = outputEnd;
  }

  return std::move(*output);
}

llvm::Expected<llvm::SmallVector<uint8_t>>
mlir::darwinn::packDepthwise(DepthwiseShape shape, llvm::ArrayRef<float> filter,
                             std::optional<llvm::ArrayRef<float>> bias) {
  auto validatedBias = validateValues(filter, shape.getElementCount(), bias,
                                      shape.getChannels());
  if (!validatedBias)
    return validatedBias.takeError();
  llvm::Expected<size_t> paddedChannels = alignedCount(shape.getChannels(), 8);
  if (!paddedChannels)
    return paddedChannels.takeError();
  llvm::Expected<size_t> paddedKernel =
      alignedCount(shape.getKernelElements(), 2);
  if (!paddedKernel)
    return paddedKernel.takeError();
  llvm::Expected<size_t> packedElements =
      elementCount({*paddedChannels, *paddedKernel});
  if (!packedElements)
    return packedElements.takeError();
  auto output =
      packedBuffer(*packedElements, *validatedBias ? *paddedChannels : 0);
  if (!output)
    return output.takeError();

  for (size_t channelStart = 0; channelStart < *paddedChannels;
       channelStart += 8) {
    if (*validatedBias) {
      llvm::ArrayRef<float> values = **validatedBias;
      for (size_t channel = channelStart; channel < channelStart + 8;
           ++channel) {
        float value = channel < values.size() ? values[channel] : 0.0f;
        uint8_t encoded[4];
        llvm::support::endian::write32le(encoded, biasBits(value));
        output->append(encoded, encoded + 4);
      }
    }

    for (size_t kernelStart = 0; kernelStart < *paddedKernel;
         kernelStart += 2) {
      for (size_t channel = channelStart; channel < channelStart + 8;
           ++channel) {
        for (size_t kernelElement = kernelStart;
             kernelElement < kernelStart + 2; ++kernelElement) {
          float value = 0.0f;
          if (channel < shape.getChannels() &&
              kernelElement < shape.getKernelElements())
            value = filter[kernelElement * shape.getChannels() + channel];
          uint8_t encoded[2];
          llvm::support::endian::write16le(encoded, bfloatBits(value));
          output->append(encoded, encoded + 2);
        }
      }
    }
  }

  return std::move(*output);
}

llvm::Expected<llvm::SmallVector<uint8_t>>
mlir::darwinn::packBfloat(llvm::ArrayRef<float> values) {
  auto output = packedBuffer(values.size(), 0);
  if (!output)
    return output.takeError();

  for (float value : values) {
    uint8_t encoded[2];
    llvm::support::endian::write16le(encoded, bfloatBits(value));
    output->append(encoded, encoded + 2);
  }

  return std::move(*output);
}

llvm::Expected<ComposedConvolution> mlir::darwinn::composePointwise(
    ConvolutionShape firstShape, llvm::ArrayRef<float> firstFilter,
    std::optional<llvm::ArrayRef<float>> firstBias,
    ConvolutionShape secondShape, llvm::ArrayRef<float> secondFilter,
    std::optional<llvm::ArrayRef<float>> secondBias) {
  if (firstShape.getKernelHeight() != 1 || firstShape.getKernelWidth() != 1 ||
      secondShape.getKernelHeight() != 1 || secondShape.getKernelWidth() != 1 ||
      firstShape.getOutputChannels() != secondShape.getInputChannels())
    return llvm::createStringError("Pointwise composition requires matching "
                                   "channels and one-element kernels");

  auto validatedFirstBias =
      validateValues(firstFilter, firstShape.getElementCount(), firstBias,
                     firstShape.getOutputChannels());
  if (!validatedFirstBias)
    return validatedFirstBias.takeError();
  auto validatedSecondBias =
      validateValues(secondFilter, secondShape.getElementCount(), secondBias,
                     secondShape.getOutputChannels());
  if (!validatedSecondBias)
    return validatedSecondBias.takeError();
  auto shape = ConvolutionShape::create(
      {secondShape.getOutputChannels(), 1, 1, firstShape.getInputChannels()});
  if (!shape)
    return shape.takeError();
  llvm::SmallVector<float> filter;
  llvm::SmallVector<float> bias;
  if (shape->getElementCount() > filter.max_size() ||
      shape->getOutputChannels() > bias.max_size())
    return llvm::createStringError(
        "Composed weights exceed the buffer size limit");
  filter.resize(shape->getElementCount(), 0.0f);
  bias.resize(shape->getOutputChannels(), 0.0f);

  for (size_t outputChannel = 0; outputChannel < shape->getOutputChannels();
       ++outputChannel) {
    for (size_t intermediateChannel = 0;
         intermediateChannel < firstShape.getOutputChannels();
         ++intermediateChannel) {
      float coefficient =
          secondFilter[outputChannel * secondShape.getInputChannels() +
                       intermediateChannel];

      for (size_t inputChannel = 0; inputChannel < shape->getInputChannels();
           ++inputChannel) {
        volatile float product =
            coefficient *
            firstFilter[intermediateChannel * firstShape.getInputChannels() +
                        inputChannel];
        filter[outputChannel * shape->getInputChannels() + inputChannel] +=
            product;
      }

      if (*validatedFirstBias) {
        volatile float product =
            coefficient * (**validatedFirstBias)[intermediateChannel];
        bias[outputChannel] += product;
      }
    }

    if (*validatedSecondBias)
      bias[outputChannel] += (**validatedSecondBias)[outputChannel];
  }

  return ComposedConvolution{*shape, std::move(filter), std::move(bias)};
}
