#ifndef MLIR_TARGET_DARWINN_COMPRESSION_H
#define MLIR_TARGET_DARWINN_COMPRESSION_H

#include "llvm/Support/Error.h"
#include <cstddef>

namespace mlir::darwinn {

class BitWriter;

class CompressionLayout {
public:
  static llvm::Expected<CompressionLayout>
  create(size_t headerBits, size_t maximumBodyBits,
         size_t maximumInstructionChunks);

  llvm::Expected<bool> writeBody(BitWriter &writer, const BitWriter *previous,
                                 const BitWriter &current) const;

private:
  CompressionLayout(size_t headerBits, size_t maximumBodyBits,
                    unsigned indexBits, unsigned valueBits, unsigned countBits,
                    size_t firstCapacity)
      : headerBits(headerBits), maximumBodyBits(maximumBodyBits),
        indexBits(indexBits), valueBits(valueBits), countBits(countBits),
        firstCapacity(firstCapacity) {}

  size_t headerBits;
  size_t maximumBodyBits;
  unsigned indexBits;
  unsigned valueBits;
  unsigned countBits;
  size_t firstCapacity;
};

}

#endif
