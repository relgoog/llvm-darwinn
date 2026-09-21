#include "mlir/Target/Darwinn/Traversal.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
#include <limits>

using namespace mlir::darwinn;
using llvm::createStringError;
using llvm::Error;
using llvm::Expected;

static Expected<unsigned> selectorBits(unsigned count) {
  if (count == 0)
    return createStringError("Traversal loop depth must be positive");
  return llvm::Log2_64_Ceil(count);
}

static Error writeInteger(BitWriter &writer, int64_t value, unsigned width) {
  if (width == 0 || width > 64)
    return createStringError("Invalid signed instruction width %u", width);
  if (value >= 0)
    return writer.write(static_cast<uint64_t>(value), width);
  if (!llvm::isIntN(width, value))
    return createStringError("Signed instruction value does not fit its field");
  uint64_t mask = std::numeric_limits<uint64_t>::max() >> (64 - width);
  return writer.write(static_cast<uint64_t>(value) & mask, width);
}

static Expected<uint64_t> counterCount(const Counter &counter) {
  if (counter.step == 0) {
    if (counter.end != 0)
      return createStringError(
          "Traversal counter has a nonzero limit and zero step");
    return 0;
  }
  if (counter.end == std::numeric_limits<int64_t>::min() && counter.step == -1)
    return createStringError("Traversal counter count overflows");
  int64_t count = counter.end / counter.step;
  if (count < 0 || counter.end % counter.step != 0)
    return createStringError(
        "Traversal counter limit must be a nonnegative multiple of its step");
  return static_cast<uint64_t>(count);
}

static Error writeCounters(BitWriter &writer, const Traversal &traversal,
                           const TraversalEncoding &encoding) {
  bool active =
      !traversal.counter.empty() && traversal.counter.front().step != 0;
  if (!active && llvm::any_of(traversal.counter, [](const Counter &counter) {
        return counter.end != 0 || counter.step != 0 || counter.mask;
      }))
    return createStringError(
        "Disabled traversal counters must have zero limits, steps and masks");
  if (traversal.alternateLimitLoopId == 0 && traversal.alternateInnerLimit != 0)
    return createStringError(
        "Traversal alternate limit requires a loop selector");

  if (!active) {
    for (unsigned index = 0; index < encoding.loopDepth; ++index) {
      if (Error error = writer.write(0, encoding.counterBits + 1))
        return error;
      if (Error error = writer.write(0, encoding.limitBits))
        return error;
    }
    return writer.write(0, encoding.loopDepth);
  }

  if (traversal.counter.size() != encoding.loopDepth)
    return createStringError(
        "Traversal counter count does not match loop depth");

  int64_t previousEnd = 0;
  for (auto [index, counter] : llvm::enumerate(traversal.counter)) {
    int64_t stride =
        counter.mask || (index == 0 && counter.end == 0) ? counter.step : 0;
    int64_t normalized;
    if (llvm::SubOverflow(stride, previousEnd, normalized))
      return createStringError("Traversal counter stride overflows");
    if (!llvm::isIntN(encoding.counterBits + 1, normalized))
      return createStringError(
          "Traversal counter stride exceeds its signed field");
    if (Error error =
            writeInteger(writer, normalized, encoding.counterBits + 1))
      return error;
    Expected<uint64_t> count = counterCount(counter);
    if (!count)
      return count.takeError();
    if (Error error = writer.write(*count, encoding.limitBits))
      return error;
    if (counter.mask &&
        llvm::AddOverflow(previousEnd, counter.end, previousEnd))
      return createStringError("Traversal counter reset overflows");
  }

  for (const Counter &counter : traversal.counter)
    if (Error error = writer.write(counter.mask, 1))
      return error;
  return Error::success();
}

static Error writeAlternateLimit(BitWriter &writer, const Traversal &traversal,
                                 const TraversalEncoding &encoding,
                                 unsigned loopBits) {
  uint64_t alternateCount = 0;
  int64_t alternateReset = 0;
  if (traversal.alternateLimitLoopId != 0) {
    if (traversal.alternateLimitLoopId >= traversal.counter.size())
      return createStringError(
          "Traversal alternate limit selects an unavailable loop");
    if (encoding.alternateCounter >= traversal.counter.size())
      return createStringError(
          "Traversal alternate limit selects an unavailable counter");
    const Counter &counter = traversal.counter[encoding.alternateCounter];
    Expected<uint64_t> count = counterCount(
        {traversal.alternateInnerLimit, counter.step, counter.mask});
    if (!count)
      return count.takeError();
    alternateCount = *count;
    if (counter.mask &&
        llvm::SubOverflow(counter.end, traversal.alternateInnerLimit,
                          alternateReset))
      return createStringError("Traversal alternate reset overflows");
  }

  if (Error error = writer.write(alternateCount, encoding.limitBits))
    return error;
  if (Error error = writer.write(traversal.alternateLimitLoopId, loopBits))
    return error;
  return writeInteger(writer, alternateReset, encoding.counterBits);
}

static Error writePrologue(BitWriter &writer, const Prologue *prologue,
                           const TraversalEncoding &encoding, unsigned loopBits,
                           bool byteAddress) {
  const Prologue absent;
  const Prologue &value = prologue ? *prologue : absent;
  if (value.loopId >= encoding.loopDepth)
    return createStringError("Traversal prologue selects an unavailable loop");
  if (value.accessBytes != 0 && (!encoding.accessBits || !byteAddress))
    return createStringError(
        "Traversal prologue does not support access bytes");

  if (Error error = writer.write(prologue != nullptr, 1))
    return error;
  if (Error error = writer.write(value.loopId, loopBits))
    return error;
  if (Error error =
          writeInteger(writer, value.baseAddress, encoding.paddingBaseBits))
    return error;
  if (Error error = writer.write(value.innerLimit, encoding.counterBits))
    return error;
  if (Error error = writer.write(value.outerLimit, encoding.counterBits))
    return error;
  if (Error error = writer.write(value.outerStride, encoding.counterBits))
    return error;

  if (encoding.accessBits) {
    uint64_t access = 0;
    if (prologue && byteAddress) {
      if (value.accessBytes == 0)
        return createStringError(
            "Prologue access must contain at least one byte");
      access = value.accessBytes - 1;
    }
    if (Error error = writer.write(access, *encoding.accessBits))
      return error;
  }

  if (encoding.paddingTargetBits)
    return writer.write(value.prologueTarget, *encoding.paddingTargetBits);
  if (value.prologueTarget != 0)
    return createStringError("Traversal does not support a prologue target");
  return Error::success();
}

Error mlir::darwinn::writeTraversal(BitWriter &writer,
                                    const Traversal &traversal,
                                    const TraversalEncoding &encoding) {
  Expected<unsigned> loopBits = selectorBits(encoding.loopDepth);
  if (!loopBits)
    return loopBits.takeError();
  if (encoding.loopDepth > 64)
    return createStringError("Traversal loop depth exceeds its mask width");
  if (encoding.counterBits == 0 || encoding.counterBits >= 64)
    return createStringError(
        "Traversal counter width must be between 1 and 63 bits");
  if (encoding.baseBits == 0 || encoding.baseBits > 64 ||
      encoding.limitBits == 0 || encoding.limitBits > 64 ||
      (encoding.accessBits &&
       (*encoding.accessBits == 0 || *encoding.accessBits > 64)))
    return createStringError(
        "Traversal encoding contains an invalid field width");
  unsigned extraPadding = encoding.paddingCount ? encoding.paddingCount - 1 : 0;
  if ((traversal.prologue && encoding.paddingCount == 0) ||
      traversal.additionalPrologues.size() > extraPadding)
    return createStringError("Traversal contains an unavailable prologue");

  if (Error error = writer.write(traversal.baseAddress, encoding.baseBits))
    return error;
  if (Error error = writeCounters(writer, traversal, encoding))
    return error;

  if (encoding.doubleBuffer) {
    if (traversal.doubleBufferLoop >= encoding.loopDepth)
      return createStringError(
          "Traversal double buffer selects an unavailable loop");
    if (Error error = writer.write(traversal.doubleBufferLoop, *loopBits))
      return error;
    if (Error error = writeInteger(writer, traversal.secondBufferOffset,
                                   encoding.counterBits))
      return error;
    if (Error error = writer.write(traversal.initializeInPongState, 1))
      return error;
  } else if (traversal.doubleBufferLoop != 0 ||
             traversal.initializeInPongState ||
             traversal.secondBufferOffset != 0) {
    return createStringError("Traversal does not support double buffering");
  }

  if (Error error = writeAlternateLimit(writer, traversal, encoding, *loopBits))
    return error;
  for (unsigned index = 0; index < encoding.paddingCount; ++index) {
    const Prologue *padding = nullptr;
    if (index == 0) {
      if (traversal.prologue)
        padding = &*traversal.prologue;
    } else if (index - 1 < traversal.additionalPrologues.size()) {
      padding = &traversal.additionalPrologues[index - 1];
    }
    if (Error error = writePrologue(writer, padding, encoding, *loopBits,
                                    traversal.byteAddressMode.has_value()))
      return error;
  }

  if (encoding.accessBits) {
    if (traversal.byteAddressMode) {
      const ByteAddressMode &address = *traversal.byteAddressMode;
      if (address.lastAccessBytes == 0 || address.defaultAccessBytes == 0)
        return createStringError(
            "Traversal access must contain at least one byte");
      if (Error error =
              writer.write(address.accessBytesLoopMap, encoding.loopDepth))
        return error;
      if (Error error =
              writer.write(address.lastAccessBytes - 1, *encoding.accessBits))
        return error;
      if (Error error = writer.write(address.defaultAccessBytes - 1,
                                     *encoding.accessBits))
        return error;
    } else {
      if (Error error = writer.write(0, encoding.loopDepth))
        return error;
      if (Error error = writer.write(0, *encoding.accessBits))
        return error;
      if (Error error = writer.write(0, *encoding.accessBits))
        return error;
    }
  } else if (traversal.byteAddressMode) {
    return createStringError("Traversal does not support byte address mode");
  }
  return Error::success();
}

Error mlir::darwinn::writeMainOperation(BitWriter &writer,
                                        const MainOperation &operation) {
  if (operation.alternateLimitLoopId == 0 && operation.alternateInnerLimit != 0)
    return createStringError(
        "Main operation alternate limit requires a loop selector");
  for (uint16_t limit : operation.counter)
    if (Error error = writer.write(limit, 14))
      return error;
  uint16_t alternate =
      operation.alternateLimitLoopId == 0 ? 0 : operation.alternateInnerLimit;
  if (Error error = writer.write(alternate, 14))
    return error;
  return writer.write(operation.alternateLimitLoopId, 3);
}

Error mlir::darwinn::writeSyncProducer(BitWriter &writer,
                                       const SyncProducer *producer,
                                       uint8_t loopDepth) {
  Expected<unsigned> width = selectorBits(static_cast<unsigned>(loopDepth) + 1);
  if (!width)
    return width.takeError();
  const SyncProducer absent;
  const SyncProducer &value = producer ? *producer : absent;
  if (value.syncFlagLoopDepth > loopDepth)
    return createStringError("Sync producer selects an unavailable loop");
  if (Error error = writer.write(value.increment, 1))
    return error;
  return writer.write(value.syncFlagLoopDepth, *width);
}

Error mlir::darwinn::writeSyncWatcher(BitWriter &writer,
                                      const SyncWatcher *watcher,
                                      uint8_t loopDepth) {
  Expected<unsigned> width = selectorBits(loopDepth);
  if (!width)
    return width.takeError();
  if (!watcher) {
    if (Error error = writer.write(0, 6))
      return error;
    if (Error error = writer.write(0, *width))
      return error;
    return writer.write(0, 53);
  }

  if (watcher->syncFlagLoopDepth > loopDepth)
    return createStringError("Sync watcher selects an unavailable loop");
  unsigned flag = static_cast<unsigned>(watcher->syncFlag);
  if (flag > static_cast<unsigned>(TileSyncFlag::ActivationRead))
    return createStringError("Sync watcher selects an unavailable flag");
  if (Error error = writer.write(flag, 5))
    return error;
  if (Error error = writer.write(watcher->bykjWtLmxpJcgmyqKq, 1))
    return error;
  if (Error error = writer.write(watcher->syncFlagLoopDepth, *width))
    return error;
  if (Error error =
          writeInteger(writer, watcher->initialExpectedSyncFlagValue, 25))
    return error;
  if (Error error = writer.write(watcher->waitSyncFlagStride, 25))
    return error;
  if (Error error = writer.write(watcher->syncWaitValid, 1))
    return error;
  return writer.write(watcher->threadId, 2);
}
