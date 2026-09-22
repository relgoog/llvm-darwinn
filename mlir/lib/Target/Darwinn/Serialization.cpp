#include "mlir/Target/Darwinn/Serialization.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include <algorithm>
#include <limits>
#include <tuple>

using namespace mlir::darwinn;

const InstructionLayout *
EncodedProgram::findInstruction(InstructionRef instruction) const {
  auto key = std::tie(instruction.chunk, instruction.fragment,
                      instruction.instruction);
  auto found =
      std::lower_bound(layout.begin(), layout.end(), key,
                       [](const InstructionLayout &entry, const auto &value) {
                         const auto &reference = entry.instruction;
                         return std::tie(reference.chunk, reference.fragment,
                                         reference.instruction) < value;
                       });

  if (found == layout.end() || !(found->instruction == instruction))
    return nullptr;
  return &*found;
}

llvm::Expected<InstructionBytes>
FragmentEncoder::encode(const Instruction &instruction) {
  return std::visit([&](const auto &packet) { return encode(packet); },
                    instruction);
}

llvm::Expected<InstructionBytes>
FragmentEncoder::encode(const ScalarPacket &packet) {
  return scalarEncoder.encode(packet.header, packet.instruction);
}

llvm::Expected<InstructionBytes>
FragmentEncoder::encode(const TaggedPacket &packet) {
  return std::visit(
      llvm::makeVisitor(
          [&](const ScalarFence &fence) {
            return fence.encode(packet.header, packet.tag);
          },
          [&](const HibDma &dma) {
            return encodeHibDma(packet.header, packet.tag, dma);
          },
          [&](const PopInput &input) {
            return encodePopInput(packet.header, packet.tag, input);
          },
          [&](const RingInfeed &infeed) {
            return encodeRingInfeed(packet.header, packet.tag, infeed);
          },
          [&](const RingOutfeed &outfeed) {
            return encodeRingOutfeed(packet.header, packet.tag, outfeed);
          }),
      packet.instruction);
}

llvm::Expected<InstructionBytes>
FragmentEncoder::encode(const TilePacket &packet) {
  return std::visit(
      [&](const auto &instruction) {
        return instruction.encode(packet.header);
      },
      packet.instruction);
}

llvm::Expected<InstructionBytes>
FragmentEncoder::encode(const ComputePacket &packet) {
  return std::visit(
      llvm::makeVisitor(
          [&](const TensorOp &tensor) {
            return tensorEncoder.encode(packet.header, packet.overwrite,
                                        packet.registerSourcedOperands, tensor);
          },
          [&](const CoefficientTables &coefficients) {
            return coefficients.encode(packet.header, packet.overwrite,
                                       packet.registerSourcedOperands);
          }),
      packet.instruction);
}

llvm::Expected<InstructionBytes>
FragmentEncoder::encode(const DmaInstruction &instruction) {
  return dmaEncoder.encode(instruction);
}

llvm::Expected<llvm::SmallVector<InstructionBytes, 0>>
mlir::darwinn::serializeChunks(
    llvm::ArrayRef<ProgramChunk> chunks, const ScalarEncoder &scalarEncoder,
    std::optional<CompressionLayout> dmaCompression) {
  auto result = serializeProgram(chunks, scalarEncoder, dmaCompression);
  if (!result)
    return result.takeError();
  return std::move(result->chunks);
}

llvm::Expected<EncodedProgram> mlir::darwinn::serializeProgram(
    llvm::ArrayRef<ProgramChunk> chunks, const ScalarEncoder &scalarEncoder,
    std::optional<CompressionLayout> dmaCompression) {
  EncodedProgram result;
  result.chunks.reserve(chunks.size());

  for (auto [chunkIndex, chunk] : llvm::enumerate(chunks)) {
    if (chunkIndex > UINT32_MAX)
      return llvm::createStringError("Darwinn chunk index exceeds 32 bits");
    InstructionBytes chunkBytes;

    for (auto [fragmentIndex, fragment] : llvm::enumerate(chunk.fragments)) {
      if (fragmentIndex > UINT32_MAX)
        return llvm::createStringError(
            "Darwinn fragment index exceeds 32 bits");
      FragmentEncoder encoder(scalarEncoder, dmaCompression);

      for (auto [instructionIndex, instruction] :
           llvm::enumerate(fragment.instructions)) {
        if (instructionIndex > UINT32_MAX)
          return llvm::createStringError(
              "Darwinn instruction index exceeds 32 bits");
        auto bytes = encoder.encode(instruction);

        if (!bytes)
          return llvm::createStringError(llvm::formatv(
              "Darwinn chunk {0} fragment {1} instruction {2} {3}", chunkIndex,
              fragmentIndex, instructionIndex,
              llvm::toString(bytes.takeError())));

        InstructionLayout entry{{static_cast<uint32_t>(chunkIndex),
                                 static_cast<uint32_t>(fragmentIndex),
                                 static_cast<uint32_t>(instructionIndex)},
                                chunkBytes.size(),
                                bytes->size(),
                                std::nullopt,
                                std::nullopt};
        auto *tagged = std::get_if<TaggedPacket>(&instruction);

        if (tagged && std::holds_alternative<HibDma>(tagged->instruction)) {
          if (entry.chunkByteOffset >
              (std::numeric_limits<uint64_t>::max() - 37) / 8)
            return llvm::createStringError("Darwinn patch offset overflows");
          entry.hibAddressBitOffset = entry.chunkByteOffset * 8 + 37;
        }

        if (tagged) {
          if (auto *fence = std::get_if<ScalarFence>(&tagged->instruction))
            entry.scalarFence = InstructionLayout::ScalarFenceInfo{
                tagged->tag, fence->sendInterrupt};
        }

        result.layout.push_back(entry);
        chunkBytes.append(bytes->begin(), bytes->end());
      }
    }

    result.chunks.push_back(std::move(chunkBytes));
  }

  return result;
}
