#include "mlir/Target/Darwinn/Memory.h"

#include "llvm/ADT/bit.h"

using namespace mlir::darwinn;

llvm::Expected<InstructionBytes>
TileLoadStore::encode(const TileHeader &header) const {
  if (rowAddress >= 196608)
    return llvm::createStringError("Tile row address exceeds G5 narrow memory");
  if (registerBurstLength == 0 || registerBurstLength > 64)
    return llvm::createStringError(
        "G5 tile register burst must contain 1 to 64 registers");

  uint64_t encodedOperation = static_cast<uint8_t>(operation);
  if (encodedOperation > static_cast<uint8_t>(TileMemoryOperation::Store))
    return llvm::createStringError("Invalid G5 tile memory operation");

  uint64_t encodedReplacement = static_cast<uint8_t>(replaceField);
  if (encodedReplacement >
      static_cast<uint8_t>(TileScalarReplacement::AddressAndImmediate))
    return llvm::createStringError("Invalid G5 tile scalar replacement");

  BitWriter writer;
  if (llvm::Error error = writeTileHeader(writer, header, 27))
    return error;
  if (llvm::Error error = writer.write(encodedOperation, 1))
    return error;
  if (llvm::Error error = writer.write(rowAddress, 18))
    return error;
  if (llvm::Error error = writer.write(registerBurstLength - 1, 6))
    return error;
  if (llvm::Error error = writer.write(baseRegister, 6))
    return error;
  if (llvm::Error error = writer.write(useImmediate, 1))
    return error;
  if (llvm::Error error = writer.write(immediateValue, 32))
    return error;
  if (llvm::Error error = writer.write(useTraceRegisters, 1))
    return error;
  if (llvm::Error error = writer.write(scalarRegister, 5))
    return error;
  if (llvm::Error error = writer.write(encodedReplacement, 2))
    return error;
  if (llvm::Error error = writer.writeBitmap(threadBitmap))
    return error;
  return writer.finish();
}

llvm::Expected<InstructionBytes> CoefficientTables::encode(
    const TileHeader &header, OverwriteInfo overwrite,
    const std::array<bool, 8> &registerSourcedOperandBitmap) const {
  if (polynomialDegree > 4)
    return llvm::createStringError(
        "Polynomial degree exceeds the G5 coefficient table");

  BitWriter writer;
  if (llvm::Error error = writeTileComputeHeader(writer, header, 25, overwrite,
                                                 registerSourcedOperandBitmap))
    return error;

  for (const auto &segment : splineSegments) {
    for (float coefficient : segment) {
      if (llvm::Error error =
              writer.write(llvm::bit_cast<uint32_t>(coefficient), 32))
        return error;
    }
  }

  for (float bound : segmentLowerBounds) {
    if (llvm::Error error = writer.write(llvm::bit_cast<uint32_t>(bound), 32))
      return error;
  }

  if (llvm::Error error = writer.write(polynomialDegree, 3))
    return error;
  if (llvm::Error error = writer.writeBitmap(enhancedSquareBitmap))
    return error;
  if (llvm::Error error = writer.writeBitmap(threadMulticastBitmap))
    return error;
  return writer.finish();
}
