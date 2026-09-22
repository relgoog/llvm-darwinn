#include "mlir/Target/Darwinn/ParameterLayout.h"

#include <algorithm>
#include <array>
#include <limits>

using namespace mlir::darwinn;

llvm::Expected<size_t> mlir::darwinn::projectConvolutionInputChannelTile(
    ConvolutionShape shape, llvm::ArrayRef<ConvolutionLoop> innerToOuter) {
  std::array<uint64_t, 4> spans{1, 1, 1, 1};
  std::array<bool, 4> present{};
  uint64_t inputTile = 1;
  uint64_t outputTile = 1;
  unsigned phase = 0;

  for (const ConvolutionLoop &loop : innerToOuter) {
    if (loop.axis < ConvolutionLoopAxis::Batch ||
        loop.axis > ConvolutionLoopAxis::OutputChannel)
      return llvm::createStringError("Unknown convolution loop axis");
    if (loop.step == 0 || loop.begin >= loop.endExclusive)
      return llvm::createStringError("Convolution loop range must be positive");
    uint64_t distance = loop.endExclusive - loop.begin;
    uint64_t count = distance / loop.step + (distance % loop.step != 0);
    unsigned axisIndex = 0;

    switch (loop.axis) {
    case ConvolutionLoopAxis::Batch:
    case ConvolutionLoopAxis::OutputY:
    case ConvolutionLoopAxis::OutputX:
      continue;
    case ConvolutionLoopAxis::OutputChannel:
      axisIndex = 0;
      break;
    case ConvolutionLoopAxis::InputChannel:
      axisIndex = 1;
      break;
    case ConvolutionLoopAxis::KernelX:
      axisIndex = 2;
      break;
    case ConvolutionLoopAxis::KernelY:
      axisIndex = 3;
      break;
    }

    if (loop.begin != 0 || loop.step != spans[axisIndex])
      return llvm::createStringError(
          "Parameter loops must decompose contiguous coordinates from zero");
    if (count > std::numeric_limits<uint64_t>::max() / spans[axisIndex])
      return llvm::createStringError("Convolution loop span overflow");
    spans[axisIndex] *= count;
    present[axisIndex] = true;
    if (count == 1)
      continue;

    unsigned nextPhase = phase;

    switch (loop.axis) {
    case ConvolutionLoopAxis::OutputChannel:
      nextPhase = phase == 0 ? 0 : 5;
      if (phase == 0)
        outputTile = spans[0];
      break;
    case ConvolutionLoopAxis::InputChannel:
      nextPhase = phase <= 1 ? 1 : 4;
      if (phase <= 1)
        inputTile = spans[1];
      break;
    case ConvolutionLoopAxis::KernelX:
      nextPhase = 2;
      break;
    case ConvolutionLoopAxis::KernelY:
      nextPhase = 3;
      break;
    case ConvolutionLoopAxis::Batch:
    case ConvolutionLoopAxis::OutputY:
    case ConvolutionLoopAxis::OutputX:
      return llvm::createStringError("Spatial axis in parameter projection");
    }

    if (nextPhase < phase)
      return llvm::createStringError(
          "Convolution loop order cannot use the parameter packing layout");
    phase = nextPhase;
  }

  std::array<uint64_t, 4> logical{
      shape.getOutputChannels(), shape.getInputChannels(),
      shape.getKernelWidth(), shape.getKernelHeight()};

  for (unsigned index = 0; index < spans.size(); ++index) {
    if (!present[index] || spans[index] < logical[index])
      return llvm::createStringError(
          "Parameter loops do not cover the convolution filter");
    if (index >= 2 && spans[index] != logical[index])
      return llvm::createStringError("Padded kernel loops are unsupported");
  }

  if (spans[0] - logical[0] >= outputTile || spans[1] - logical[1] >= inputTile)
    return llvm::createStringError("Excess parameter loop padding");
  if (outputTile < logical[0] && outputTile % 8 != 0)
    return llvm::createStringError(
        "Output channel loop tiles must preserve eight-channel groups");
  if (inputTile < logical[1] && inputTile % 2 != 0)
    return llvm::createStringError(
        "Input channel loop tiles must preserve channel pairs");

  uint64_t tile = std::min(inputTile, logical[1]);
  if (tile > std::numeric_limits<size_t>::max() - (tile % 2))
    return llvm::createStringError("Input channel tile alignment overflow");
  return static_cast<size_t>(tile + tile % 2);
}
