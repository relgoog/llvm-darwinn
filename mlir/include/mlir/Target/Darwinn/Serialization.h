#ifndef MLIR_TARGET_DARWINN_SERIALIZATION_H
#define MLIR_TARGET_DARWINN_SERIALIZATION_H

#include "mlir/Target/Darwinn/Compression.h"
#include "mlir/Target/Darwinn/Dma.h"
#include "mlir/Target/Darwinn/Encoding.h"
#include "mlir/Target/Darwinn/Memory.h"
#include "mlir/Target/Darwinn/Tensor.h"
#include "mlir/Target/Darwinn/Transport.h"

#include <optional>
#include <variant>

namespace mlir::darwinn {

struct ScalarPacket {
  Header header;
  ScalarInstruction instruction;
};

struct TaggedPacket {
  Header header;
  uint32_t tag = 0;
  std::variant<ScalarFence, HibDma, PopInput, RingInfeed, RingOutfeed>
      instruction;
};

struct TilePacket {
  TileHeader header;
  std::variant<TileFence, TileLoadStore> instruction;
};

struct ComputePacket {
  TileHeader header;
  OverwriteInfo overwrite;
  std::array<bool, 8> registerSourcedOperands{};
  std::variant<TensorOp, CoefficientTables> instruction;
};

using Instruction = std::variant<ScalarPacket, TaggedPacket, TilePacket,
                                 ComputePacket, DmaInstruction>;

struct ProgramFragment {
  llvm::SmallVector<Instruction, 0> instructions;
};

struct ProgramChunk {
  llvm::SmallVector<ProgramFragment, 0> fragments;
};

class FragmentEncoder {
public:
  explicit FragmentEncoder(
      ScalarEncoder scalarEncoder,
      std::optional<CompressionLayout> dmaCompression = std::nullopt)
      : scalarEncoder(scalarEncoder), dmaEncoder(dmaCompression) {}

  llvm::Expected<InstructionBytes> encode(const Instruction &instruction);

private:
  llvm::Expected<InstructionBytes> encode(const ScalarPacket &packet);
  llvm::Expected<InstructionBytes> encode(const TaggedPacket &packet);
  llvm::Expected<InstructionBytes> encode(const TilePacket &packet);
  llvm::Expected<InstructionBytes> encode(const ComputePacket &packet);
  llvm::Expected<InstructionBytes> encode(const DmaInstruction &instruction);

  ScalarEncoder scalarEncoder;
  TensorEncoder tensorEncoder;
  DmaEncoder dmaEncoder;
};

llvm::Expected<llvm::SmallVector<InstructionBytes, 0>>
serializeChunks(llvm::ArrayRef<ProgramChunk> chunks,
                const ScalarEncoder &scalarEncoder,
                std::optional<CompressionLayout> dmaCompression = std::nullopt);

}

#endif
