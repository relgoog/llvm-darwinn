#ifndef MLIR_TARGET_DARWINN_MEMORY_H
#define MLIR_TARGET_DARWINN_MEMORY_H

#include "mlir/Target/Darwinn/Encoding.h"

#include <array>
#include <cstdint>

namespace mlir::darwinn {

enum class TileMemoryOperation : uint8_t { Load, Store };

enum class TileScalarReplacement : uint8_t {
  None,
  Address,
  Immediate,
  AddressAndImmediate
};

struct TileLoadStore {
  TileMemoryOperation operation = TileMemoryOperation::Load;
  uint32_t rowAddress = 0;
  uint8_t registerBurstLength = 1;
  uint8_t baseRegister = 0;
  bool useImmediate = false;
  uint32_t immediateValue = 0;
  bool useTraceRegisters = false;
  uint8_t scalarRegister = 0;
  TileScalarReplacement replaceField = TileScalarReplacement::None;
  std::array<bool, 4> threadBitmap{};

  llvm::Expected<InstructionBytes> encode(const TileHeader &header) const;
};

struct CoefficientTables {
  std::array<std::array<float, 5>, 8> splineSegments{};
  std::array<float, 7> segmentLowerBounds{};
  uint8_t polynomialDegree = 0;
  std::array<bool, 8> enhancedSquareBitmap{};
  std::array<bool, 4> threadMulticastBitmap{};

  llvm::Expected<InstructionBytes>
  encode(const TileHeader &header, OverwriteInfo overwrite,
         const std::array<bool, 8> &registerSourcedOperandBitmap) const;
};

}

#endif
