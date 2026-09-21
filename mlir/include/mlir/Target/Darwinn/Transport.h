#ifndef MLIR_TARGET_DARWINN_TRANSPORT_H
#define MLIR_TARGET_DARWINN_TRANSPORT_H

#include "mlir/Target/Darwinn/Encoding.h"
#include "mlir/Target/Darwinn/Traversal.h"
#include <array>
#include <cstdint>
#include <optional>

namespace mlir::darwinn {

enum class DmaQueue : uint8_t {
  Instruction = 0,
  Activation = 1,
  Parameter = 2,
  Output = 3,
};

enum class InputFifo : uint8_t {
  Parameter = 0,
  Activation = 1,
};

enum class FieldSource : uint8_t {
  Instruction = 0,
  ScalarRegister = 1,
};

enum class ScalarSyncFlag : uint8_t {
  ActivationPop = 0,
  ParameterPop = 1,
  ActivationInfeed = 2,
  ParameterInfeed = 3,
  ScalarInfeed = 4,
  RingProducerA = 5,
  RingProducerB = 6,
  RingOutfeed = 7,
  Pipeline = 8,
  Software0 = 9,
  Software1 = 10,
  Software2 = 11,
  Software3 = 12,
  Software4 = 13,
  Software5 = 14,
  Software6 = 15,
  Software7 = 16,
};

struct ScalarSyncWatcher {
  ScalarSyncFlag flag = ScalarSyncFlag::ActivationPop;
  uint8_t loopDepth = 0;
  uint32_t initialValue = 0;
  uint32_t stride = 1;
  bool valid = false;
};

struct HibDma {
  DmaQueue queue = DmaQueue::Activation;
  Traversal traversal;
};

struct PopInput {
  InputFifo fifo = InputFifo::Activation;
  Traversal traversal;
  std::optional<ScalarSyncWatcher> watcher;
};

struct RingInfeed {
  InputFifo fifo = InputFifo::Activation;
  Traversal traversal;
  std::array<std::optional<ScalarSyncWatcher>, 2> watchers;
  uint32_t syncIncrement = 0;
  std::array<bool, 8> virtualChannels{};
  std::array<bool, 17> targets{};
  FieldSource bitmapSource = FieldSource::Instruction;
  FieldSource baseAddressSource = FieldSource::Instruction;
  std::array<FieldSource, 10> counterSources{};
  std::optional<uint8_t> unsigned8ToBfloatZeroPoint;
};

struct RingOutfeed {
  Traversal traversal;
  std::optional<ScalarSyncWatcher> watcher;
  FieldSource baseAddressSource = FieldSource::Instruction;
  std::array<FieldSource, 10> counterSources{};
  std::array<bool, 8> virtualChannels{};
  uint32_t bytesToPop = 0;
};

llvm::Expected<InstructionBytes> encodeHibDma(Header header, uint32_t tag,
                                              const HibDma &dma);
llvm::Expected<InstructionBytes> encodePopInput(Header header, uint32_t tag,
                                                const PopInput &input);
llvm::Expected<InstructionBytes> encodeRingInfeed(Header header, uint32_t tag,
                                                  const RingInfeed &infeed);
llvm::Expected<InstructionBytes> encodeRingOutfeed(Header header, uint32_t tag,
                                                   const RingOutfeed &outfeed);

}

#endif
