#include "Families.h"
#include "mlir/IR/AffineExpr.h"

using namespace mlir;
using namespace mlir::darwinn;
using namespace mlir::darwinn::codegen;

namespace {

constexpr int64_t kRow = 128;
constexpr int64_t kBundle = 32;

struct PermutePlan {
  SmallVector<int64_t> perm;
  Index inStrides;
  Index outStrides;
  Index outExtent;
  int64_t sliceBytes;
  int64_t cell;
  int64_t cellsPerSlice;
  int64_t cellsPerRow;
  int64_t pixels;
  int64_t rows;
  Operation *source;
};

SmallVector<int64_t> permutation(Operation *op) {
  AffineMap map =
      op->getAttrOfType<AffineMapAttr>("forward_index_transformation")
          .getValue();
  SmallVector<int64_t> out;
  for (AffineExpr expr : map.getResults())
    out.push_back(cast<AffineDimExpr>(expr).getPosition());
  return out;
}

PermutePlan plan(Operation *op) {
  Operation *source = producer(op, 0);
  int64_t elem = resultInfo(op).elementBytes;
  Index outExtent = extent(unionBox(*slicingOf(op), 0, 0));
  SmallVector<int64_t> perm = permutation(op);
  Operation *owner = storage(source);
  Index inExtent;
  if (auto slicing = slicingOf(owner)) {
    inExtent = extent(unionBox(*slicing, 0, 0));
  } else {
    for (size_t dim = 0; dim < outExtent.size(); ++dim)
      inExtent.push_back(outExtent[llvm::find(perm, dim) - perm.begin()]);
  }
  PermutePlan out;
  out.perm = perm;
  out.inStrides = strides(inExtent, elem);
  out.outStrides = strides(outExtent, elem);
  out.outExtent = outExtent;
  out.sliceBytes = outExtent.back() / 4 * elem;
  out.cell = std::min(out.sliceBytes, kRow);
  out.cellsPerSlice = out.sliceBytes / out.cell;
  out.cellsPerRow = out.cell <= kRow / 2 ? kRow / out.cell : 1;
  out.pixels = product(ArrayRef(outExtent).drop_back());
  out.rows = out.cellsPerSlice * out.pixels / out.cellsPerRow;
  out.source = owner;
  return out;
}

SmallVector<std::pair<int64_t, int64_t>>
mergePixels(ArrayRef<std::pair<int64_t, int64_t>> loops) {
  SmallVector<std::pair<int64_t, int64_t>> out;
  for (auto [count, stride] : loops) {
    if (count == 1)
      continue;
    if (!out.empty() && stride == out.back().first * out.back().second)
      out.back().first *= count;
    else
      out.push_back({count, stride});
  }
  return out;
}

} // namespace

int64_t codegen::permuteWideRows(Operation *op) {
  int64_t rows = plan(op).rows;
  return rows <= 16 ? rows : 8;
}

Body codegen::permuteCopy(Operation *op, Context &context) {
  PermutePlan p = plan(op);
  int64_t rank = p.outExtent.size();
  SmallVector<std::pair<int64_t, int64_t>> read;
  if (p.cellsPerSlice > 1)
    read.push_back({p.cellsPerSlice, p.cell});
  SmallVector<std::pair<int64_t, int64_t>> pixelLoops;
  for (int64_t dim = rank - 2; dim >= 0; --dim)
    pixelLoops.push_back({p.outExtent[dim], p.inStrides[p.perm[dim]]});
  llvm::append_range(read, mergePixels(pixelLoops));
  bool doubled = p.rows > 16;
  SmallVector<Counter> wideItems =
      doubled ? SmallVector<Counter>{counter(3, 1),
                                     counter(p.rows / 4 - 1, 1, false)}
              : SmallVector<Counter>{counter(p.rows - 1, 1)};
  SmallVector<std::pair<int64_t, int64_t>> scatter{{p.cell / kBundle, 8}};
  if (p.cellsPerRow > 1)
    scatter.push_back({p.cellsPerRow, p.outStrides[rank - 2] / 4});
  if (p.cellsPerSlice > 1)
    scatter.push_back({p.cellsPerSlice, p.cell / 4});
  SmallVector<std::pair<int64_t, int64_t>> pixelOut;
  for (int64_t dim = rank - 2; dim >= 0; --dim)
    pixelOut.push_back({p.outExtent[dim], p.outStrides[dim]});
  pixelOut = mergePixels(pixelOut);
  if (p.cellsPerRow > 1)
    pixelOut[0] = {pixelOut[0].first / p.cellsPerRow,
                   pixelOut[0].second * p.cellsPerRow};
  for (auto [count, stride] : pixelOut)
    scatter.push_back({count, stride / 4});
  int64_t rowLoops = 1 + (p.cellsPerRow > 1);
  size_t depth = 1 + (p.cellsPerSlice > 1 || p.cellsPerRow > 1);
  int64_t perIncrement = 1;
  for (auto [count, stride] : ArrayRef(scatter).take_front(depth))
    perIncrement *= count;
  perIncrement = perIncrement * kBundle / kRow;
  int64_t increments = doubled ? 4 / std::max<int64_t>(perIncrement, 1) : 0;
  FailureOr<int64_t> source = context.storageAddress(p.source);
  FailureOr<int64_t> destination = context.narrowAddress(op);
  if (failed(source) || failed(destination))
    return unsupported(op, "a copy between unplaced narrow blocks");
  std::array<bool, 16> every = bits<16>("1111111111111111");
  SmallVector<Emitted, 0> out;
  for (int64_t thread = 0; thread < kThreads; ++thread) {
    std::array<bool, 4> bitmap = threadBit(thread);
    SmallVector<std::pair<int64_t, int64_t>> threadRead = read;
    if (!(p.cellsPerSlice > 1 || thread == 0))
      threadRead.insert(threadRead.begin(), {1, p.cell});
    SmallVector<Counter> readItems;
    for (auto [count, stride] : threadRead)
      readItems.push_back(counter((count - 1) * stride, stride));
    if (readItems.empty())
      readItems.push_back(counter(0, p.cell));
    NarrowToWide gather;
    gather.read.baseAddress = *source + thread * p.sliceBytes;
    gather.read.counter = padded(readItems, 6);
    gather.read.byteAddressMode = access(p.cell);
    gather.write.counter = padded(wideItems, 4);
    gather.write.syncProducer = {producerSync(true, 1)};
    gather.write.byteAddressMode = access(p.cell * p.cellsPerRow);
    if (doubled) {
      gather.write.doubleBufferLoop = 1;
      gather.write.secondBufferOffset = 4;
      gather.writeWatchers = {dmaWatcher(
          tileWatcher(TileSyncFlag::WideToNarrowWrite,
                      (int32_t(1) << 25) - increments, increments, 1, true),
          true)};
    }
    gather.threadMulticastBitmap = bitmap;
    gather.byteAddress.strideUnitGranulesLoopMap = 1;
    gather.byteAddress.defaultStrideUnitGranules = 2;
    gather.byteAddress.lastStrideUnitGranules = 2;
    gather.byteAddress.cellStride = p.cell == 96 ? 24 : 32;
    gather.byteAddress.defaultCellStrideGroupCount = 1;
    gather.byteAddress.lastCellStrideGroupCount = 1;
    out.push_back(dma(gather, every));
    WideToNarrow scatterOp;
    scatterOp.read.counter = padded(wideItems, 4);
    if (doubled) {
      scatterOp.read.doubleBufferLoop = 1;
      scatterOp.read.secondBufferOffset = 4;
    }
    SmallVector<Counter> writeItems;
    for (auto [count, stride] : scatter)
      writeItems.push_back(counter((count - 1) * stride, stride));
    scatterOp.write.baseAddress = (*destination + thread * p.sliceBytes) / 4;
    scatterOp.write.counter = padded(writeItems, 6);
    scatterOp.write.syncProducer = {producerSync(true, depth)};
    scatterOp.readWatchers = {dmaWatcher(
        tileWatcher(TileSyncFlag::NarrowToWideWrite, 1, 1, 1, true))};
    scatterOp.wideMemoryLoadStoreLoopId = rowLoops;
    scatterOp.zInBundleValidCount = 8;
    scatterOp.threadMulticastBitmap = bitmap;
    out.push_back(dma(scatterOp, every));
  }
  return out;
}
