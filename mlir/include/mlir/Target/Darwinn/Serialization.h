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

struct InstructionRef {
  uint32_t chunk = 0;
  uint32_t fragment = 0;
  uint32_t instruction = 0;

  bool operator==(const InstructionRef &other) const {
    return chunk == other.chunk && fragment == other.fragment &&
           instruction == other.instruction;
  }
};

struct InstructionLayout {
  struct ScalarFenceInfo {
    uint32_t tag;
    bool sendInterrupt;
  };

  InstructionRef instruction;
  uint64_t chunkByteOffset = 0;
  uint64_t byteLength = 0;
  std::optional<uint64_t> hibAddressBitOffset;
  std::optional<ScalarFenceInfo> scalarFence;
};

class EncodedProgram {
public:
  EncodedProgram(EncodedProgram &&) = default;
  EncodedProgram &operator=(EncodedProgram &&) = delete;
  EncodedProgram(const EncodedProgram &) = delete;
  EncodedProgram &operator=(const EncodedProgram &) = delete;

  llvm::ArrayRef<InstructionBytes> getChunks() const { return chunks; }
  llvm::ArrayRef<InstructionLayout> getLayout() const { return layout; }
  const InstructionLayout *findInstruction(InstructionRef instruction) const;

private:
  EncodedProgram() = default;
  friend llvm::Expected<EncodedProgram>
  serializeProgram(llvm::ArrayRef<ProgramChunk>, const ScalarEncoder &,
                   std::optional<CompressionLayout>);
  friend llvm::Expected<llvm::SmallVector<InstructionBytes, 0>>
  serializeChunks(llvm::ArrayRef<ProgramChunk>, const ScalarEncoder &,
                  std::optional<CompressionLayout>);

  llvm::SmallVector<InstructionBytes, 0> chunks;
  llvm::SmallVector<InstructionLayout, 0> layout;
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

llvm::Expected<EncodedProgram> serializeProgram(
    llvm::ArrayRef<ProgramChunk> chunks, const ScalarEncoder &scalarEncoder,
    std::optional<CompressionLayout> dmaCompression = std::nullopt);

}

#endif
