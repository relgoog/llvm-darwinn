#include "Codegen.h"
#include <set>

using namespace mlir;
using namespace mlir::darwinn;
using namespace mlir::darwinn::codegen;

namespace {

const HibDma *hibOf(const Instruction &instruction) {
  auto *tagged = std::get_if<TaggedPacket>(&instruction);
  return tagged ? std::get_if<HibDma>(&tagged->instruction) : nullptr;
}

bool inputHib(const Instruction &instruction) {
  const HibDma *hib = hibOf(instruction);
  return hib && hib->queue != DmaQueue::Output;
}

bool outputHib(const Instruction &instruction) {
  const HibDma *hib = hibOf(instruction);
  return hib && hib->queue == DmaQueue::Output;
}

bool interrupt(const Emitted &emitted) {
  auto *tagged = std::get_if<TaggedPacket>(&emitted.instruction);
  if (!tagged)
    return false;
  auto *fence = std::get_if<ScalarFence>(&tagged->instruction);
  return fence && fence->sendInterrupt;
}

} // namespace

FailureOr<GeneratedProgram> codegen::buildProgram(ArrayRef<Segment> segments,
                                                  const Context &context) {
  SmallVector<const Emitted *> flat;
  std::set<size_t> starts{0};
  std::set<size_t> fragmentCandidates;
  std::optional<size_t> prolog;
  for (const Segment &segment : segments) {
    size_t position = flat.size();
    if (!prolog && segment.group)
      prolog = position + 2;
    bool stores = llvm::any_of(segment.instructions, [](const Emitted &item) {
      return outputHib(item.instruction);
    });
    if (stores)
      starts.insert(position + segment.header);
    if (stores || (segment.group && segment.group->kind == GroupKind::Preempt))
      fragmentCandidates.insert(position);
    for (const Emitted &emitted : segment.instructions)
      flat.push_back(&emitted);
  }
  size_t size = flat.size();
  for (size_t k = 0; k < size; ++k) {
    if (inputHib(flat[k]->instruction))
      starts.insert(k);
    if (interrupt(*flat[k]))
      starts.insert(k + 1);
  }
  if (prolog)
    starts.insert(*prolog);
  SmallVector<size_t> added;
  for (size_t k = 0; k < size; ++k) {
    if (!inputHib(flat[k]->instruction) || (k > 0 && interrupt(*flat[k - 1])))
      continue;
    auto following = starts.upper_bound(k);
    if (following != starts.end() && *following < size &&
        inputHib(flat[*following]->instruction))
      added.push_back(k + 1);
  }
  starts.insert(added.begin(), added.end());
  starts.erase(starts.lower_bound(size), starts.end());

  GeneratedProgram program;
  program.hibs = context.resolvedHibs();
  SmallVector<size_t> bounds(starts.begin(), starts.end());
  bounds.push_back(size);
  for (size_t chunkIndex = 0; chunkIndex + 1 < bounds.size(); ++chunkIndex) {
    size_t begin = bounds[chunkIndex], end = bounds[chunkIndex + 1];
    ChunkMeta meta;
    meta.startsWithInputHib = inputHib(flat[begin]->instruction);
    if (interrupt(*flat[end - 1]))
      meta.interrupt = flat[end - 1]->role;
    SmallVector<size_t> splits{begin};
    if (meta.startsWithInputHib)
      for (size_t candidate : fragmentCandidates)
        if (candidate > begin && candidate < end)
          splits.push_back(candidate);
    splits.push_back(end);
    ProgramChunk chunk;
    for (size_t fragmentIndex = 0; fragmentIndex + 1 < splits.size();
         ++fragmentIndex) {
      ProgramFragment fragment;
      for (size_t k = splits[fragmentIndex]; k < splits[fragmentIndex + 1];
           ++k) {
        if (const HibDma *hib = hibOf(flat[k]->instruction))
          meta.hibs.push_back(
              {static_cast<int64_t>(hib->traversal.baseAddress),
               InstructionRef{
                   static_cast<uint32_t>(chunkIndex),
                   static_cast<uint32_t>(fragmentIndex),
                   static_cast<uint32_t>(fragment.instructions.size())}});
        fragment.instructions.push_back(flat[k]->instruction);
      }
      chunk.fragments.push_back(std::move(fragment));
    }
    program.chunks.push_back(std::move(chunk));
    program.meta.push_back(std::move(meta));
  }
  return program;
}
