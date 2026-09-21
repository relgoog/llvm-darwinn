#include "mlir/Target/Darwinn/Serialization.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir::darwinn;

namespace {

class ChunkEncoder {
public:
  ChunkEncoder(const ScalarEncoder &scalarEncoder,
               std::optional<CompressionLayout> dmaCompression)
      : scalarEncoder(scalarEncoder), dmaEncoder(dmaCompression) {}

  llvm::Expected<InstructionBytes> encode(const ScalarPacket &packet) {
    return scalarEncoder.encode(packet.header, packet.instruction);
  }

  llvm::Expected<InstructionBytes> encode(const TaggedPacket &packet) {
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

  llvm::Expected<InstructionBytes> encode(const TilePacket &packet) {
    return std::visit(
        [&](const auto &instruction) {
          return instruction.encode(packet.header);
        },
        packet.instruction);
  }

  llvm::Expected<InstructionBytes> encode(const ComputePacket &packet) {
    return std::visit(
        llvm::makeVisitor(
            [&](const TensorOp &tensor) {
              return tensorEncoder.encode(packet.header, packet.overwrite,
                                          packet.registerSourcedOperands,
                                          tensor);
            },
            [&](const CoefficientTables &coefficients) {
              return coefficients.encode(packet.header, packet.overwrite,
                                         packet.registerSourcedOperands);
            }),
        packet.instruction);
  }

  llvm::Expected<InstructionBytes> encode(const DmaInstruction &instruction) {
    return dmaEncoder.encode(instruction);
  }

private:
  const ScalarEncoder &scalarEncoder;
  TensorEncoder tensorEncoder;
  DmaEncoder dmaEncoder;
};

}

llvm::Expected<llvm::SmallVector<InstructionBytes, 0>>
mlir::darwinn::serializeChunks(
    llvm::ArrayRef<ProgramChunk> chunks, const ScalarEncoder &scalarEncoder,
    std::optional<CompressionLayout> dmaCompression) {
  llvm::SmallVector<InstructionBytes, 0> result;
  result.reserve(chunks.size());

  for (auto [chunkIndex, chunk] : llvm::enumerate(chunks)) {
    ChunkEncoder encoder(scalarEncoder, dmaCompression);
    InstructionBytes chunkBytes;

    for (auto [instructionIndex, instruction] :
         llvm::enumerate(chunk.instructions)) {
      auto bytes =
          std::visit([&](const auto &packet) { return encoder.encode(packet); },
                     instruction);

      if (!bytes)
        return llvm::createStringError(
            llvm::formatv("Darwinn chunk {0} instruction {1} {2}", chunkIndex,
                          instructionIndex, llvm::toString(bytes.takeError())));

      chunkBytes.append(bytes->begin(), bytes->end());
    }

    result.push_back(std::move(chunkBytes));
  }

  return result;
}
