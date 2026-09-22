#include "mlir/Target/Darwinn/ExecutionPlan.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir::darwinn;

static bool isValidBuffer(BufferRef buffer) {
  switch (buffer.kind) {
  case BufferKind::InputActivation:
  case BufferKind::OutputActivation:
  case BufferKind::ParameterRegion:
    return buffer.index <= INT32_MAX;
  case BufferKind::Scratch:
    return buffer.index == 0;
  }
  return false;
}

llvm::Error mlir::darwinn::validateExecutionPlan(const ExecutionPlan &plan) {
  const auto &encoded = plan.program;
  if (plan.entryName.empty())
    return llvm::createStringError("execution entry name must not be empty");
  if (encoded.getChunks().empty() || plan.actions.empty())
    return llvm::createStringError(
        "execution plan requires chunks and actions");
  for (const auto &chunk : encoded.getChunks()) {
    if (chunk.size() > uint64_t(INT64_MAX))
      return llvm::createStringError(
          "instruction chunk exceeds the runtime size range");
  }

  for (auto [index, requirement] : llvm::enumerate(plan.buffers)) {
    if (!isValidBuffer(requirement.buffer))
      return llvm::createStringError("execution buffer reference is invalid");

    for (const auto &previous :
         llvm::ArrayRef(plan.buffers).take_front(index)) {
      if (previous.buffer == requirement.buffer)
        return llvm::createStringError(
            "execution buffer requirement is duplicated");
    }
  }

  std::optional<uint32_t> lastDispatch;
  llvm::SmallVector<InstructionRef, 0> pendingPatches;
  llvm::DenseSet<uint32_t> preemptionTags;

  for (auto [actionIndex, action] : llvm::enumerate(plan.actions)) {
    auto error = std::visit(
        llvm::makeVisitor(
            [&](const HibAddressPatch &patch) -> llvm::Error {
              auto *layout = encoded.findInstruction(patch.instruction);
              if (!layout || !layout->hibAddressBitOffset)
                return llvm::createStringError(
                    "address patch requires an encoded HIB DMA address field");
              uint64_t chunkSize =
                  encoded.getChunks()[patch.instruction.chunk].size();
              uint64_t offset = *layout->hibAddressBitOffset;
              if (offset > uint64_t(INT64_MAX) - 64)
                return llvm::createStringError(
                    "address patch exceeds the signed runtime bit offset "
                    "range");
              if (chunkSize > UINT64_MAX / 8 || offset > chunkSize * 8 ||
                  chunkSize * 8 - offset < 64)
                return llvm::createStringError(
                    "address patch exceeds its chunk");

              auto requirement =
                  llvm::find_if(plan.buffers, [&](const auto &entry) {
                    return entry.buffer == patch.buffer;
                  });
              if (requirement == plan.buffers.end())
                return llvm::createStringError(
                    "address patch references an undeclared buffer");
              if (patch.accessedBytes &&
                  *patch.accessedBytes > UINT64_MAX - patch.byteOffset)
                return llvm::createStringError(
                    "address patch extent overflows");

              if (requirement->byteSize &&
                  (patch.byteOffset > *requirement->byteSize ||
                   (patch.accessedBytes &&
                    *patch.accessedBytes >
                        *requirement->byteSize - patch.byteOffset)))
                return llvm::createStringError(
                    "address patch exceeds its declared buffer");

              if (llvm::is_contained(pendingPatches, patch.instruction))
                return llvm::createStringError(
                    "address field is patched twice before dispatch");
              pendingPatches.push_back(patch.instruction);
              lastDispatch.reset();
              return llvm::Error::success();
            },
            [&](const Dispatch &dispatch) -> llvm::Error {
              if (dispatch.chunk >= encoded.getChunks().size() ||
                  encoded.getChunks()[dispatch.chunk].empty())
                return llvm::createStringError(
                    "dispatch references an absent or empty chunk");

              if (llvm::any_of(pendingPatches, [&](InstructionRef reference) {
                    return reference.chunk != dispatch.chunk;
                  }))
                return llvm::createStringError(
                    "dispatch leaves patches pending on another chunk");

              pendingPatches.clear();
              lastDispatch = dispatch.chunk;
              return llvm::Error::success();
            },
            [&](const FenceWait &wait) -> llvm::Error {
              if (!lastDispatch || *lastDispatch != wait.instruction.chunk)
                return llvm::createStringError(
                    "fence wait must follow its chunk dispatch");
              auto *layout = encoded.findInstruction(wait.instruction);
              if (!layout || !layout->scalarFence ||
                  !layout->scalarFence->sendInterrupt ||
                  layout->scalarFence->tag != wait.tag)
                return llvm::createStringError(
                    "fence wait requires its named interrupting ScalarFence");

              if (layout->chunkByteOffset + layout->byteLength !=
                  encoded.getChunks()[wait.instruction.chunk].size())
                return llvm::createStringError(
                    "fence wait must name the final instruction in its chunk");
              if (wait.softwarePreemption != wait.liveTensors.has_value())
                return llvm::createStringError("software preemption requires "
                                               "explicit live tensor metadata");

              if (wait.softwarePreemption) {
                if (plan.narrowMemoryRows == 0 ||
                    !preemptionTags.insert(wait.tag).second)
                  return llvm::createStringError(
                      "preemption memory capacity is absent or fence tag is "
                      "duplicated");
                llvm::SmallVector<LiveTensorSpan, 0> spans = *wait.liveTensors;
                llvm::sort(spans, [](const auto &left, const auto &right) {
                  return left.startRow < right.startRow;
                });
                uint64_t end = 0;

                for (const auto &span : spans) {
                  if (span.rowCount == 0 || span.startRow < end ||
                      span.startRow > plan.narrowMemoryRows ||
                      span.rowCount > plan.narrowMemoryRows - span.startRow)
                    return llvm::createStringError(
                        "live tensor spans overlap or exceed narrow memory");
                  end = uint64_t(span.startRow) + span.rowCount;
                }
              }

              lastDispatch.reset();
              return llvm::Error::success();
            },
            [&](const CheckPreemption &) -> llvm::Error {
              if (!pendingPatches.empty())
                return llvm::createStringError(
                    "preemption check leaves undispatched patches");
              lastDispatch.reset();
              return llvm::Error::success();
            }),
        action);

    if (error)
      return llvm::createStringError(
          llvm::formatv("execution action {0} {1}", actionIndex,
                        llvm::toString(std::move(error))));
  }

  if (!pendingPatches.empty())
    return llvm::createStringError(
        "execution plan ends with undispatched patches");
  return llvm::Error::success();
}
