#ifndef MLIR_TARGET_DARWINN_ENCODING_H
#define MLIR_TARGET_DARWINN_ENCODING_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <variant>

namespace mlir::darwinn {

using InstructionBytes = llvm::SmallVector<uint8_t, 64>;

class Predicate {
public:
  static llvm::Expected<Predicate> create(uint8_t registerId, bool polarity);
  uint8_t getRegister() const { return registerId; }
  bool getPolarity() const { return polarity; }

private:
  Predicate(uint8_t registerId, bool polarity)
      : registerId(registerId), polarity(polarity) {}
  uint8_t registerId;
  bool polarity;
};

struct Header {
  std::optional<Predicate> predicate;
  bool enableTrace = false;
  bool enableSequencerOverwrite = false;
};

struct TileHeader {
  Header header;
  std::array<bool, 16> multicastBitmap{};
  uint32_t tag = 0;
};

struct OverwriteInfo {
  uint8_t baseAddressSource = 0;
  uint8_t multicastBitmapSource = 0;
  bool overwriteMulticastBitmap = false;
};

class BitWriter {
public:
  llvm::Error write(uint64_t value, unsigned width);
  llvm::Error writeBitmap(llvm::ArrayRef<bool> values);
  size_t bitLen() const { return bitLength; }
  llvm::ArrayRef<uint8_t> bytes() const { return data; }
  InstructionBytes takeBytes();
  InstructionBytes finish();

private:
  InstructionBytes data;
  size_t bitLength = 0;
};

llvm::Error writeCommonHeader(BitWriter &writer, Header header, uint8_t opcode,
                              bool sequencerOverwrite);
llvm::Error writeTileHeader(BitWriter &writer, const TileHeader &header,
                            uint8_t opcode);
llvm::Error
writeTileComputeHeader(BitWriter &writer, const TileHeader &header,
                       uint8_t opcode, OverwriteInfo overwrite,
                       llvm::ArrayRef<bool> registerSourcedOperands);

enum class MemoryPowerState { Active, DeepSleep };

struct TileFence {
  std::array<bool, 22> resetSyncFlag{};
  std::array<bool, 4> resetSyncFlagThreadIds{};
  uint32_t syncResetValue = 0;
  std::array<bool, 13> waitIdle{};
  std::array<bool, 4> waitIdleThreadIds{};
  std::array<bool, 2> waitProducerCountValid{};
  uint32_t producerCountValue = 0;
  bool increment = false;
  bool halt = false;
  MemoryPowerState narrowMemPowerState = MemoryPowerState::Active;
  MemoryPowerState wideMemPowerState = MemoryPowerState::Active;
  std::array<bool, 8> virtualChannelSubscription{};
  bool virtualChannelSubscriptionValid = false;
  bool incrementGlobalTokenA = false;
  bool incrementGlobalTokenB = false;
  bool discardRemainingBytes = false;

  llvm::Expected<InstructionBytes> encode(const TileHeader &header) const;
};

struct ScalarFence {
  std::array<bool, 20> resetTileFence{};
  std::array<bool, 17> resetSyncFlag{};
  uint32_t resetValue = 0;
  uint32_t expectedTileFenceCount = 0;
  std::array<bool, 20> tileFenceValid{};
  std::array<std::optional<uint32_t>, 17> expectedScalarSyncFlag{};
  std::array<bool, 5> waitIdle{};
  bool incrementGlobalTokenA = false;
  bool incrementGlobalTokenB = false;
  bool preemptable = false;
  bool discardRemainingBytes = false;
  std::array<bool, 8> virtualChannelSubscription{};
  bool virtualChannelSubscriptionValid = false;
  bool sendInterrupt = false;

  llvm::Expected<InstructionBytes> encode(Header header, uint32_t tag) const;
};

struct NoOp {};
struct Halt {
  bool gotoHalt = false;
  bool interrupt = false;
  uint8_t interruptId = 0;
  bool gotoSleep = false;
};
struct Interrupt {
  uint8_t interruptId = 0;
};
struct LoadProgram {
  uint32_t startPc = 0;
  uint32_t numberChunks = 0;
};
struct Execute {
  uint32_t startPc = 0;
  bool loadPc = false;
};
using ScalarInstruction =
    std::variant<NoOp, Halt, Interrupt, LoadProgram, Execute>;

class ScalarEncoder {
public:
  static llvm::Expected<ScalarEncoder> create(unsigned programCounterBits,
                                              bool sequencerOverwrite);
  llvm::Expected<InstructionBytes>
  encode(Header header, const ScalarInstruction &instruction) const;

private:
  ScalarEncoder(unsigned programCounterBits, bool sequencerOverwrite)
      : programCounterBits(programCounterBits),
        sequencerOverwrite(sequencerOverwrite) {}
  unsigned programCounterBits;
  bool sequencerOverwrite;
};

}

#endif
