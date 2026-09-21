#include "mlir/Target/Darwinn/Compression.h"
#include "mlir/Target/Darwinn/Encoding.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/bit.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <utility>

using namespace mlir::darwinn;

static uint32_t readBits(llvm::ArrayRef<uint8_t> bytes, size_t offset,
                         unsigned width) {
  uint32_t value = 0;

  for (unsigned index = 0; index < width; ++index) {
    size_t position = offset + index;
    value |= uint32_t((bytes[position / 8] >> (position % 8)) & 1) << index;
  }

  return value;
}

llvm::Expected<CompressionLayout>
CompressionLayout::create(size_t headerBits, size_t maximumBodyBits,
                          size_t maximumInstructionChunks) {
  if (headerBits >= 127 || maximumBodyBits == 0)
    return llvm::createStringError("Invalid compressed instruction dimensions");

  size_t maximum = std::numeric_limits<size_t>::max();
  if (maximumInstructionChunks < 2 || maximumInstructionChunks > maximum / 128)
    return llvm::createStringError(
        "Invalid maximum compressed instruction length");

  size_t maximumChangeCount = maximumInstructionChunks * 4 - 4;
  unsigned countBits = llvm::bit_width(maximumChangeCount - 1);
  if (headerBits + 1 + countBits > 128)
    return llvm::createStringError(
        "Compression header exceeds the first instruction chunk");

  size_t firstCapacity = (128 - headerBits - 1 - countBits) / 32;
  size_t maximumEncodedBits = maximumInstructionChunks * 128;
  if (maximumBodyBits > maximum - headerBits - 1 ||
      headerBits + 1 + maximumBodyBits > maximumEncodedBits)
    return llvm::createStringError(
        "Instruction body exceeds the configured chunk count");

  size_t maximumComparisonBits = (maximumBodyBits + 7) & ~size_t(7);

  for (unsigned valueBits = 31; valueBits != 0; --valueBits) {
    size_t segmentCount = llvm::divideCeil(maximumComparisonBits, valueBits);
    unsigned indexBits = llvm::bit_width(segmentCount - 1);
    if (indexBits + valueBits <= 32)
      return CompressionLayout(headerBits, maximumBodyBits, indexBits,
                               valueBits, countBits, firstCapacity);
  }

  return llvm::createStringError(
      "Instruction body exceeds the compression index space");
}

llvm::Expected<bool>
CompressionLayout::writeBody(BitWriter &writer, const BitWriter *previous,
                             const BitWriter &current) const {
  if (writer.bitLen() % 128 != headerBits)
    return llvm::createStringError(
        "Instruction header does not match its compression layout");

  if (current.bitLen() > maximumBodyBits ||
      (previous && previous->bitLen() > maximumBodyBits))
    return llvm::createStringError(
        "Instruction body exceeds its compression layout");

  llvm::SmallVector<std::pair<size_t, uint32_t>> changes;

  if (previous) {
    size_t previousBits = previous->bytes().size() * 8;
    size_t currentBits = current.bytes().size() * 8;

    for (size_t offset = 0, index = 0; offset < currentBits;
         offset += valueBits, ++index) {
      unsigned width = std::min<size_t>(currentBits - offset, valueBits);
      uint32_t value = readBits(current.bytes(), offset, width);

      if (previousBits < offset + width ||
          readBits(previous->bytes(), offset, width) != value)
        changes.emplace_back(index, value);
    }
  }

  size_t uncompressedChunks =
      llvm::divideCeil(headerBits + 1 + current.bitLen(), 128);
  size_t availableChanges = uncompressedChunks * 4 + firstCapacity;
  bool compressed = previous && availableChanges >= 8 &&
                    changes.size() <= availableChanges - 8 &&
                    unsigned(llvm::bit_width(changes.size())) <= countBits;
  BitWriter encoded = writer;

  if (llvm::Error error = encoded.write(compressed, 1))
    return error;

  if (compressed) {
    if (llvm::Error error = encoded.write(changes.size(), countBits))
      return error;

    if (!changes.empty()) {
      size_t firstOffset = 128 - 32 * firstCapacity;
      unsigned padding = firstOffset - headerBits - 1 - countBits;

      if (llvm::Error error = encoded.write(0, padding))
        return error;

      for (const auto &entry : changes) {
        if (llvm::Error error = encoded.write(entry.first, indexBits))
          return error;
        if (llvm::Error error = encoded.write(entry.second, valueBits))
          return error;
      }
    }
  } else {
    for (size_t offset = 0; offset < current.bitLen(); offset += 32) {
      unsigned width = std::min<size_t>(current.bitLen() - offset, 32);

      if (llvm::Error error =
              encoded.write(readBits(current.bytes(), offset, width), width))
        return error;
    }
  }

  writer = std::move(encoded);
  return compressed;
}
