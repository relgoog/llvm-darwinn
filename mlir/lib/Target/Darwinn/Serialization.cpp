#include "mlir/Target/Darwinn/Serialization.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir::darwinn;

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
  llvm::SmallVector<InstructionBytes, 0> result;
  result.reserve(chunks.size());

  for (auto [chunkIndex, chunk] : llvm::enumerate(chunks)) {
    InstructionBytes chunkBytes;

    for (auto [fragmentIndex, fragment] : llvm::enumerate(chunk.fragments)) {
      FragmentEncoder encoder(scalarEncoder, dmaCompression);
      for (auto [instructionIndex, instruction] :
           llvm::enumerate(fragment.instructions)) {
        auto bytes = encoder.encode(instruction);

        if (!bytes)
          return llvm::createStringError(llvm::formatv(
              "Darwinn chunk {0} fragment {1} instruction {2} {3}", chunkIndex,
              fragmentIndex, instructionIndex,
              llvm::toString(bytes.takeError())));

        chunkBytes.append(bytes->begin(), bytes->end());
      }
    }

    result.push_back(std::move(chunkBytes));
  }

  return result;
}
