#include "mlir/Target/Darwinn/Encoding.h"
#include "llvm/ADT/Twine.h"
#include <limits>
#include <type_traits>
#include <utility>

namespace mlir::darwinn {

llvm::Expected<Predicate> Predicate::create(uint8_t registerId, bool polarity) {
  if (registerId >= 8)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Predicate register exceeds the V1 register bank");
  return Predicate(registerId, polarity);
}

llvm::Error BitWriter::write(uint64_t value, unsigned width) {
  if (width > 64 || (width < 64 && value >= (uint64_t{1} << width)))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Instruction value " + llvm::Twine(value) +
                                       " does not fit in " +
                                       llvm::Twine(width) + " bits");
  if (width > std::numeric_limits<size_t>::max() - bitLength)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Instruction bit count overflows");

  size_t end = bitLength + width;
  data.resize(end / 8 + (end % 8 != 0), 0);

  for (unsigned index = 0; index < width; ++index) {
    size_t offset = bitLength + index;
    data[offset / 8] |= static_cast<uint8_t>((value >> index) & 1)
                        << (offset % 8);
  }

  bitLength = end;
  return llvm::Error::success();
}

llvm::Error BitWriter::writeBitmap(llvm::ArrayRef<bool> values) {
  for (bool value : values)
    if (llvm::Error error = write(value, 1))
      return error;
  return llvm::Error::success();
}

InstructionBytes BitWriter::takeBytes() {
  bitLength = 0;
  return std::exchange(data, {});
}

InstructionBytes BitWriter::finish() {
  data.resize((bitLength / 128 + (bitLength % 128 != 0)) * 16, 0);
  return takeBytes();
}

llvm::Error writeCommonHeader(BitWriter &writer, Header header, uint8_t opcode,
                              bool sequencerOverwrite) {
  if (header.enableSequencerOverwrite && !sequencerOverwrite)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Sequencer overwrite is unavailable for this V1 configuration");

  uint64_t predicate = 0;
  if (header.predicate)
    predicate = 1 | (uint64_t(header.predicate->getRegister()) << 1) |
                (uint64_t(header.predicate->getPolarity()) << 4);

  if (llvm::Error error = writer.write(predicate, 5))
    return error;
  if (llvm::Error error = writer.write(header.enableTrace, 1))
    return error;
  if (sequencerOverwrite)
    if (llvm::Error error = writer.write(header.enableSequencerOverwrite, 1))
      return error;
  return writer.write(opcode, 6);
}

llvm::Error writeTileHeader(BitWriter &writer, const TileHeader &header,
                            uint8_t opcode) {
  if (llvm::Error error =
          writeCommonHeader(writer, header.header, opcode, false))
    return error;
  if (llvm::Error error = writer.writeBitmap(header.multicastBitmap))
    return error;
  return writer.write(header.tag, 20);
}

llvm::Error
writeTileComputeHeader(BitWriter &writer, const TileHeader &header,
                       uint8_t opcode, OverwriteInfo overwrite,
                       llvm::ArrayRef<bool> registerSourcedOperands) {
  if (registerSourcedOperands.size() != 8)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Register operand bitmap must contain 8 bits");
  if (llvm::Error error =
          writeCommonHeader(writer, header.header, opcode, false))
    return error;
  if (llvm::Error error = writer.writeBitmap(header.multicastBitmap))
    return error;
  if (llvm::Error error = writer.write(overwrite.baseAddressSource, 5))
    return error;
  if (llvm::Error error = writer.write(overwrite.multicastBitmapSource, 5))
    return error;
  if (llvm::Error error = writer.write(overwrite.overwriteMulticastBitmap, 1))
    return error;
  if (llvm::Error error = writer.write(0, 9))
    return error;
  if (llvm::Error error = writer.write(header.tag, 20))
    return error;
  return writer.writeBitmap(registerSourcedOperands);
}

llvm::Expected<InstructionBytes>
TileFence::encode(const TileHeader &header) const {
  BitWriter writer;
  if (llvm::Error error = writeTileHeader(writer, header, 26))
    return error;
  if (llvm::Error error = writer.writeBitmap(resetSyncFlag))
    return error;
  if (llvm::Error error = writer.writeBitmap(resetSyncFlagThreadIds))
    return error;
  if (llvm::Error error = writer.write(syncResetValue, 25))
    return error;
  if (llvm::Error error = writer.writeBitmap(waitIdle))
    return error;
  if (llvm::Error error = writer.writeBitmap(waitIdleThreadIds))
    return error;
  if (llvm::Error error = writer.writeBitmap(waitProducerCountValid))
    return error;
  if (llvm::Error error = writer.write(producerCountValue, 25))
    return error;
  if (llvm::Error error = writer.write(increment, 1))
    return error;
  if (llvm::Error error = writer.write(halt, 1))
    return error;

  for (MemoryPowerState state : {narrowMemPowerState, wideMemPowerState}) {
    if (state != MemoryPowerState::Active && state != MemoryPowerState::DeepSleep)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "Invalid memory power state");

    if (llvm::Error error = writer.write(state == MemoryPowerState::DeepSleep, 1))
      return error;
  }

  if (llvm::Error error = writer.writeBitmap(virtualChannelSubscription))
    return error;
  if (llvm::Error error = writer.write(virtualChannelSubscriptionValid, 1))
    return error;
  if (llvm::Error error = writer.write(incrementGlobalTokenA, 1))
    return error;
  if (llvm::Error error = writer.write(incrementGlobalTokenB, 1))
    return error;
  if (llvm::Error error = writer.write(discardRemainingBytes, 1))
    return error;
  return writer.finish();
}

llvm::Expected<InstructionBytes> ScalarFence::encode(Header header,
                                                     uint32_t tag) const {
  BitWriter writer;
  if (llvm::Error error = writeCommonHeader(writer, header, 36, false))
    return error;
  if (llvm::Error error = writer.write(tag, 20))
    return error;
  if (llvm::Error error = writer.writeBitmap(resetTileFence))
    return error;
  if (llvm::Error error = writer.writeBitmap(resetSyncFlag))
    return error;
  if (llvm::Error error = writer.write(resetValue, 25))
    return error;
  if (llvm::Error error = writer.write(expectedTileFenceCount, 25))
    return error;
  if (llvm::Error error = writer.writeBitmap(tileFenceValid))
    return error;

  for (std::optional<uint32_t> expected : expectedScalarSyncFlag)
    if (llvm::Error error = writer.write(expected.value_or(0), 25))
      return error;

  for (std::optional<uint32_t> expected : expectedScalarSyncFlag)
    if (llvm::Error error = writer.write(expected.has_value(), 1))
      return error;

  if (llvm::Error error = writer.writeBitmap(waitIdle))
    return error;
  if (llvm::Error error = writer.write(incrementGlobalTokenA, 1))
    return error;
  if (llvm::Error error = writer.write(incrementGlobalTokenB, 1))
    return error;
  if (llvm::Error error = writer.write(preemptable, 1))
    return error;
  if (llvm::Error error = writer.write(discardRemainingBytes, 1))
    return error;
  if (llvm::Error error = writer.writeBitmap(virtualChannelSubscription))
    return error;
  if (llvm::Error error = writer.write(virtualChannelSubscriptionValid, 1))
    return error;
  if (llvm::Error error = writer.write(sendInterrupt, 1))
    return error;
  return writer.finish();
}

llvm::Expected<ScalarEncoder> ScalarEncoder::create(unsigned programCounterBits,
                                                    bool sequencerOverwrite) {
  if (programCounterBits == 0 || programCounterBits > 31)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Invalid V1 program counter width");
  return ScalarEncoder(programCounterBits, sequencerOverwrite);
}

llvm::Expected<InstructionBytes>
ScalarEncoder::encode(Header header,
                      const ScalarInstruction &instruction) const {
  BitWriter writer;
  llvm::Error result = std::visit(
      [&](const auto &operation) -> llvm::Error {
        using Operation = std::decay_t<decltype(operation)>;
        constexpr uint8_t opcode = [] {
          if constexpr (std::is_same_v<Operation, NoOp>)
            return 0;
          if constexpr (std::is_same_v<Operation, Halt>)
            return 33;
          if constexpr (std::is_same_v<Operation, Interrupt>)
            return 40;
          if constexpr (std::is_same_v<Operation, LoadProgram>)
            return 62;
          if constexpr (std::is_same_v<Operation, Execute>)
            return 63;
        }();

        if (llvm::Error error =
                writeCommonHeader(writer, header, opcode, sequencerOverwrite))
          return error;

        if constexpr (std::is_same_v<Operation, Halt>) {
          if (llvm::Error error = writer.write(operation.gotoHalt, 1))
            return error;
          if (llvm::Error error = writer.write(operation.interrupt, 1))
            return error;
          if (llvm::Error error = writer.write(operation.interruptId, 2))
            return error;
          return writer.write(operation.gotoSleep, 1);
        } else if constexpr (std::is_same_v<Operation, Interrupt>) {
          return writer.write(operation.interruptId, 2);
        } else if constexpr (std::is_same_v<Operation, LoadProgram>) {
          if (llvm::Error error =
                  writer.write(operation.startPc, programCounterBits))
            return error;
          return writer.write(operation.numberChunks, programCounterBits + 1);
        } else if constexpr (std::is_same_v<Operation, Execute>) {
          if (llvm::Error error =
                  writer.write(operation.startPc, programCounterBits))
            return error;
          return writer.write(operation.loadPc, 1);
        }

        return llvm::Error::success();
      },
      instruction);

  if (result)
    return std::move(result);
  if (writer.bitLen() > 128)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Scalar instruction exceeds 128 bits");
  return writer.finish();
}

}
