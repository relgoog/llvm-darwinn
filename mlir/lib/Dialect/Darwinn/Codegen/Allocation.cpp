#include "Codegen.h"
#include "Families.h"
#include "llvm/ADT/STLExtras.h"
#include <limits>

using namespace mlir;
using namespace mlir::darwinn;
using namespace mlir::darwinn::codegen;

namespace {

constexpr int64_t kNarrowSize = 196608;
constexpr int64_t kNarrowCapacity = kNarrowSize - 4;
constexpr int64_t kWideCapacity = 64;

bool isView(Operation *op) {
  return isa<GetTensorOp, DistributedCreateViewOp, ReshapeOpOp>(op);
}

void consumers(Operation *op, SmallVectorImpl<Operation *> &out) {
  for (Operation *user : usersOf(op)) {
    if (isView(user))
      consumers(user, out);
    else
      out.push_back(user);
  }
}

void consumersThroughHost(Operation *op, SmallVectorImpl<Operation *> &out) {
  for (Operation *user : usersOf(op)) {
    if (isa<CommunicatedJoinViewsOp, DistributedCreateViewOp>(user))
      consumersThroughHost(user, out);
    else
      out.push_back(user);
  }
}

Operation *slicingOwner(Operation *op) {
  while (!slicingOf(op)) {
    std::optional<unsigned> index;
    if (isa<TensorOpOp>(op))
      index = 2;
    else if (isa<UnaryTensorOpOp>(op))
      index = 1;
    else if (isa<DistributedCreateViewOp>(op))
      index = 0;
    if (!index || !producer(op, *index))
      return op;
    op = producer(op, *index);
  }
  return op;
}

int64_t narrowBytes(Operation *op, bool clip) {
  Operation *owner = slicingOwner(op);
  TensorInfo info = resultInfo(op);
  Box box = unionBox(*slicingOf(owner), 0, 0);
  int64_t count = 1;
  for (size_t dim = 0; dim < box.lo.size(); ++dim) {
    int64_t lo = box.lo[dim], hi = box.hi[dim];
    if (clip && dim < info.shape.size()) {
      lo = std::max<int64_t>(0, lo);
      hi = std::min(info.shape[dim] - 1, hi);
    }
    count *= std::max<int64_t>(0, hi - lo + 1);
  }
  return ceilDiv(count * info.elementBytes, 4);
}

std::pair<Operation *, SmallVector<Operation *, 2>> hostRoot(Operation *op) {
  if (op->getNumOperands() < 2)
    return {op, {op}};
  Operation *root = producer(op, 1);
  while (isa<CommunicatedCreateWriteViewOp>(root))
    root = producer(root, 0);
  SmallVector<Operation *, 2> writers;
  for (Operation *view : usersOf(root))
    for (Operation *user : usersOf(view))
      if (isa<RedistributeOp>(user) && user->getNumOperands() > 1 &&
          producer(user, 1) == view)
        writers.push_back(user);
  return {root, writers};
}

bool better(const StorageBlock &a, int64_t aPeak, int64_t aAlign,
            const StorageBlock &b, int64_t bPeak, int64_t bAlign) {
  if (aPeak != bPeak)
    return aPeak > bPeak;
  if (aAlign != bAlign)
    return aAlign > bAlign;
  int64_t aLength = a.end - a.start + 1, bLength = b.end - b.start + 1;
  int64_t aArea = a.size * aLength * aLength,
          bArea = b.size * bLength * bLength;
  if (aArea != bArea)
    return aArea > bArea;
  return aLength > bLength;
}

} // namespace

FailureOr<SmallVector<int64_t>>
codegen::tinyMalloc(ArrayRef<StorageBlock> blocks, int64_t alignment,
                    int64_t capacity) {
  int64_t horizon = 0;
  for (const StorageBlock &block : blocks)
    horizon = std::max(horizon, block.end);
  SmallVector<int64_t> usage(horizon + 1, 0);
  for (const StorageBlock &block : blocks)
    for (int64_t step = block.start; step <= block.end; ++step)
      usage[step] += block.size;
  SmallVector<int64_t> peaks;
  SmallVector<SmallVector<size_t>> buckets(horizon + 1);
  for (auto [index, block] : llvm::enumerate(blocks)) {
    int64_t peak = 0;
    for (int64_t step = block.start; step <= block.end; ++step)
      peak = std::max(peak, usage[step]);
    peaks.push_back(peak);
    buckets[block.start].push_back(index);
  }

  SmallVector<int64_t> height(horizon + 1, 0);
  SmallVector<std::optional<int64_t>> placed(blocks.size());
  size_t count = 0;
  auto windows = [&](int64_t level) {
    SmallVector<std::pair<int64_t, int64_t>> out;
    int64_t step = 0;
    while (step <= horizon) {
      if (height[step] != level) {
        ++step;
        continue;
      }
      int64_t stop = step;
      while (stop <= horizon && height[stop] == level)
        ++stop;
      out.push_back({step, stop});
      step = stop;
    }
    return out;
  };
  auto fits = [&](size_t index, int64_t step, int64_t stop) {
    const StorageBlock &block = blocks[index];
    return !placed[index] && step + (block.end - block.start + 1) <= stop;
  };

  while (count < blocks.size()) {
    int64_t level = *llvm::min_element(height);
    while (true) {
      std::optional<size_t> best;
      for (auto [step, stop] : windows(level))
        for (int64_t at = step; at < stop; ++at)
          for (size_t index : buckets[at]) {
            if (!fits(index, at, stop) || level % alignment)
              continue;
            if (!best || better(blocks[index], peaks[index], alignment,
                                blocks[*best], peaks[*best], alignment))
              best = index;
          }
      if (!best)
        break;
      placed[*best] = level;
      ++count;
      for (int64_t step = blocks[*best].start; step <= blocks[*best].end;
           ++step)
        height[step] = level + blocks[*best].size;
      if (count == blocks.size())
        break;
    }
    if (count == blocks.size())
      break;
    int64_t next = capacity;
    for (int64_t value : height)
      if (value > level)
        next = std::min(next, value);
    for (auto [step, stop] : windows(level))
      for (int64_t at = step; at < stop; ++at)
        for (size_t index : buckets[at]) {
          if (!fits(index, at, stop))
            continue;
          int64_t aligned = level % alignment
                                ? ceilDiv(level + 1, alignment) * alignment
                                : level;
          if (level < aligned && aligned < next)
            next = aligned;
        }
    for (int64_t &value : height)
      value = std::max(value, next);
  }

  SmallVector<int64_t> offsets;
  for (auto [index, block] : llvm::enumerate(blocks)) {
    if (*placed[index] + block.size > capacity)
      return failure();
    offsets.push_back(*placed[index]);
  }
  return offsets;
}

FailureOr<StorageProblems> codegen::storageProblems(ArrayRef<Group> groups) {
  llvm::DenseMap<Operation *, int64_t> step;
  llvm::DenseSet<Operation *> passthrough;
  for (const Group &group : groups) {
    switch (group.kind) {
    case GroupKind::Op:
    case GroupKind::Permute:
    case GroupKind::Scatter:
    case GroupKind::Gather:
    case GroupKind::RingReshapeIdentity:
      step.try_emplace(group.op, group.step);
      break;
    case GroupKind::Empty:
      passthrough.insert(group.op);
      break;
    default:
      break;
    }
  }
  std::function<std::optional<int64_t>(Operation *)> useStep =
      [&](Operation *op) -> std::optional<int64_t> {
    auto found = step.find(op);
    if (found != step.end() && !passthrough.contains(op))
      return found->second;
    SmallVector<Operation *> later;
    consumers(op, later);
    std::optional<int64_t> out;
    for (Operation *user : later)
      if (auto value = useStep(user))
        out = std::max(out.value_or(*value), *value);
    return out;
  };
  auto latest = [&](ArrayRef<Operation *> users) {
    std::optional<int64_t> out;
    for (Operation *user : users)
      if (auto value = useStep(user))
        out = std::max(out.value_or(*value), *value);
    return out;
  };

  StorageProblems problems;
  llvm::DenseSet<Operation *> written;
  for (const Group &group : groups) {
    Operation *op = group.op;
    if (group.kind == GroupKind::Init) {
      problems.narrow.push_back(
          {op, Suffix::Init, group.step, group.step, 8, {}, 0});
      continue;
    }
    if (group.kind == GroupKind::Gather) {
      problems.narrow.push_back({op,
                                 Suffix::Dest,
                                 group.step,
                                 group.step + 1,
                                 gatherStagingWords(op),
                                 {},
                                 0});
      SmallVector<Operation *> readers;
      consumers(op, readers);
      problems.narrow.push_back(
          {op,
           Suffix::Dest,
           group.step + 1,
           std::max(group.step + 2, latest(readers).value_or(group.step + 2)),
           narrowBytes(op, false),
           {},
           0});
      continue;
    }
    bool kept = group.kind == GroupKind::Op ||
                group.kind == GroupKind::Permute ||
                group.kind == GroupKind::Scatter ||
                group.kind == GroupKind::RingReshapeIdentity;
    auto first = step.find(op);
    if (!kept || first == step.end() || first->second != group.step)
      continue;
    if (resultSpace(op) == DistributedMemorySpace::HostMemory) {
      if (isModelOutput(op) || written.contains(op))
        continue;
      auto [root, writers] = hostRoot(op);
      written.insert(writers.begin(), writers.end());
      SmallVector<Operation *> readers;
      for (Operation *writer : writers)
        consumersThroughHost(writer, readers);
      TensorInfo info = resultInfo(root);
      std::optional<int64_t> end = latest(readers);
      if (!end)
        return unsupported(op, "a host buffer that is never read");
      problems.host.push_back({op, Suffix::None, group.step, *end,
                               product(info.shape) * info.elementBytes, writers,
                               0});
      continue;
    }
    SmallVector<Operation *> readers;
    consumers(op, readers);
    bool moved = group.kind == GroupKind::Scatter ||
                 group.kind == GroupKind::RingReshapeIdentity;
    problems.narrow.push_back({op,
                               moved ? Suffix::Dest : Suffix::None,
                               group.step,
                               latest(readers).value_or(group.step),
                               narrowBytes(op, !moved),
                               {},
                               0});
  }
  return problems;
}

FailureOr<SmallVector<StorageBlock>>
codegen::wideProblem(ArrayRef<Group> groups) {
  SmallVector<StorageBlock> blocks;
  auto add = [&](Operation *op, Suffix suffix, int64_t start, int64_t end,
                 int64_t size) {
    blocks.push_back({op, suffix, start, end, size, {}, 0});
  };
  for (const Group &group : groups) {
    Operation *op = group.op;
    int64_t step = group.step;
    switch (group.kind) {
    case GroupKind::Init:
      add(op, Suffix::Init, step, step + 1, 1);
      continue;
    case GroupKind::Gather:
      add(op, Suffix::Gather, step, step + 1, gatherWideRows(op));
      continue;
    case GroupKind::Permute:
      add(op, Suffix::Permute, step, step, permuteWideRows(op));
      continue;
    case GroupKind::Op:
      break;
    default:
      continue;
    }
    if (isa<UnaryTensorOpOp, InterpolateHardwareOp>(op)) {
      add(op, Suffix::None, step, step, 1);
      continue;
    }
    if (!isa<TensorOpOp>(op))
      continue;
    switch (*innerOperation(op)) {
    case InnerOperationKind::Elementwise:
      if (linearFunction(op) == LinearFunctionKind::Mac)
        add(op, Suffix::None, step, step, tensorProductRows(op));
      break;
    case InnerOperationKind::Vmc: {
      FailureOr<VmcPlan> plan = vmcPlan(op);
      if (failed(plan))
        return failure();
      if (plan->biasRows)
        add(op, Suffix::Bias, step, step, plan->biasRows);
      add(op, Suffix::None, step, step, plan->weightsRows + plan->sumsRows);
      break;
    }
    case InnerOperationKind::Stencil: {
      int64_t taps = stencilTaps(op);
      if (hasBias(op))
        add(op, Suffix::Bias, step, step, 2);
      add(op, Suffix::None, step, step, taps + 2);
      break;
    }
    default:
      break;
    }
  }
  return blocks;
}

FailureOr<Context> Context::create(func::FuncOp function,
                                   ArrayRef<Group> groups) {
  Context context;
  FailureOr<StorageProblems> problems = storageProblems(groups);
  if (failed(problems))
    return failure();
  auto place = [&](SmallVector<StorageBlock> blocks, int64_t alignment,
                   int64_t capacity) -> FailureOr<SmallVector<StorageBlock>> {
    FailureOr<SmallVector<int64_t>> offsets =
        tinyMalloc(blocks, alignment, capacity);
    if (failed(offsets))
      return failure();
    for (auto [block, offset] : llvm::zip_equal(blocks, *offsets))
      block.offset = offset;
    return blocks;
  };
  auto narrow = place(problems->narrow, 8, kNarrowCapacity);
  auto host = place(problems->host, 1, std::numeric_limits<int64_t>::max());
  FailureOr<SmallVector<StorageBlock>> wideBlocks = wideProblem(groups);
  if (failed(wideBlocks))
    return failure();
  auto wide = place(*wideBlocks, 1, kWideCapacity);
  if (failed(narrow) || failed(host) || failed(wide))
    return function.emitOpError()
           << "codegen storage exceeds tile or wide memory capacity";
  context.narrowBlocks = std::move(*narrow);
  context.hostBlocks = std::move(*host);
  context.wideBlocks = std::move(*wide);
  return context;
}

static int64_t peak(ArrayRef<StorageBlock> blocks) {
  int64_t out = 0;
  for (const StorageBlock &block : blocks)
    out = std::max(out, block.offset + block.size);
  return out;
}

int64_t Context::tileMemoryBytes() const {
  return (peak(narrowBlocks) + kNarrowSize - kNarrowCapacity) * kThreads *
         kTiles;
}

int64_t Context::scratchBytes() const { return peak(hostBlocks); }

FailureOr<int64_t> Context::narrowAddress(Operation *value,
                                          Suffix suffix) const {
  for (const StorageBlock &block : narrowBlocks)
    if (block.value == value && block.suffix == suffix)
      return block.offset * kThreads;
  return failure();
}

FailureOr<int64_t> Context::storageAddress(Operation *op) const {
  Operation *node = op;
  while (true) {
    for (Suffix suffix : {Suffix::None, Suffix::Dest}) {
      const StorageBlock *latest = nullptr;
      for (const StorageBlock &block : narrowBlocks)
        if (block.value == node && block.suffix == suffix &&
            (!latest || block.start > latest->start))
          latest = &block;
      if (latest)
        return latest->offset * kThreads;
    }
    if (isa<GetTensorOp, DistributedCreateViewOp, ReshapeOpOp, RedistributeOp>(
            node) &&
        producer(node, 0)) {
      node = producer(node, 0);
      continue;
    }
    return failure();
  }
}

FailureOr<int64_t> Context::wideAddress(Operation *value, Suffix suffix) const {
  for (const StorageBlock &block : wideBlocks)
    if (block.value == value && block.suffix == suffix)
      return block.offset;
  return failure();
}

FailureOr<int64_t> Context::hostOffset(Operation *writer) const {
  for (const StorageBlock &block : hostBlocks)
    if (llvm::is_contained(block.writers, writer))
      return block.offset;
  return failure();
}

int64_t Context::hib(DmaQueue queue, HibRoot root, int64_t offset, int64_t size,
                     Operation *source) {
  hibs.push_back({queue, root, offset, size, source});
  return static_cast<int64_t>(hibs.size()) - 1;
}

SmallVector<Hib> Context::resolvedHibs() const {
  int64_t fills = 0;
  for (const Hib &hib : hibs)
    if (hib.root == HibRoot::ParameterFill)
      fills += hib.size;
  int64_t fillCursor = 0;
  int64_t parameterCursor = ceilDiv(fills, 4096) * 4096;
  auto constants = [](Operation *op) {
    return std::make_tuple(
        fillBehind(op->getOperand(1)),
        op->getNumOperands() > 3 ? fillBehind(op->getOperand(3)) : nullptr,
        op->getName());
  };
  SmallVector<Hib> out;
  for (Hib hib : hibs) {
    if (hib.root == HibRoot::ParameterFill) {
      hib.offset = fillCursor;
      fillCursor += hib.size;
      hib.root = HibRoot::ParameterRegion;
    } else if (hib.root == HibRoot::Parameter) {
      auto shared = llvm::find_if(out, [&](const Hib &placed) {
        return placed.root == HibRoot::ParameterRegion && placed.source &&
               !isa<FillOp>(placed.source) && placed.size == hib.size &&
               constants(placed.source) == constants(hib.source);
      });
      if (shared != out.end()) {
        hib.offset = shared->offset;
      } else {
        hib.offset = parameterCursor;
        parameterCursor += hib.size;
      }
      hib.root = HibRoot::ParameterRegion;
    }
    out.push_back(hib);
  }
  return out;
}
