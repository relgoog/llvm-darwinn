#include "mlir/Target/Darwinn/Allocation.h"
#include "llvm/ADT/APInt.h"
#include <algorithm>
#include <cstddef>
#include <numeric>

using namespace mlir::darwinn;

namespace {

struct PreparedAllocation {
  size_t firstSegment;
  size_t endSegment;
  uint64_t pressure = 0;
  llvm::APInt weight = llvm::APInt(192, 0);
  bool placed = false;
};

}

llvm::Expected<StorageAllocation>
mlir::darwinn::allocateStorage(llvm::ArrayRef<AllocationRecord> records,
                               uint64_t capacity) {
  StorageAllocation output;
  if (records.empty())
    return output;

  llvm::SmallVector<uint64_t> boundaries;
  llvm::SmallVector<uint64_t> identities;
  llvm::SmallVector<PreparedAllocation, 0> prepared;
  llvm::SmallVector<size_t> order;
  llvm::SmallVector<size_t> nonFloor;

  if (records.size() > (boundaries.max_size() - 1) / 2 ||
      records.size() > (nonFloor.max_size() - 2) / 2 ||
      records.size() > prepared.max_size() ||
      records.size() > order.max_size() ||
      records.size() > output.offsets.max_size())
    return llvm::createStringError("Too many storage allocation records");

  uint64_t lastTime = 0;
  identities.reserve(records.size());

  for (size_t index = 0; index < records.size(); ++index) {
    const AllocationRecord &record = records[index];
    if (record.first > record.last)
      return llvm::createStringError("Allocation %zu has a reversed lifetime",
                                     index);
    if (record.size == 0 || record.alignment == 0)
      return llvm::createStringError(
          "Allocation %zu must have positive size and alignment", index);
    if (record.size > capacity)
      return llvm::createStringError("Allocation %zu exceeds storage capacity",
                                     index);
    identities.push_back(record.identity);
    lastTime = std::max(lastTime, record.last);
  }

  std::sort(identities.begin(), identities.end());
  if (std::adjacent_find(identities.begin(), identities.end()) !=
      identities.end())
    return llvm::createStringError(
        "Storage allocation identities must be unique");

  boundaries.reserve(records.size() * 2 + 1);
  boundaries.push_back(0);

  for (const AllocationRecord &record : records) {
    boundaries.push_back(record.first);
    if (record.last < lastTime)
      boundaries.push_back(record.last + 1);
  }

  std::sort(boundaries.begin(), boundaries.end());
  boundaries.erase(std::unique(boundaries.begin(), boundaries.end()),
                   boundaries.end());
  llvm::SmallVector<uint64_t> pressure(boundaries.size(), 0);
  prepared.reserve(records.size());

  for (const AllocationRecord &record : records) {
    size_t firstSegment =
        std::lower_bound(boundaries.begin(), boundaries.end(), record.first) -
        boundaries.begin();
    size_t endSegment =
        std::upper_bound(boundaries.begin(), boundaries.end(), record.last) -
        boundaries.begin();
    prepared.push_back({firstSegment, endSegment});

    for (size_t segment = firstSegment; segment < endSegment; ++segment) {
      if (record.size > capacity - pressure[segment])
        return llvm::createStringError(
            "Overlapping live allocations exceed storage capacity");
      pressure[segment] += record.size;
    }
  }

  for (size_t index = 0; index < records.size(); ++index) {
    const AllocationRecord &record = records[index];
    PreparedAllocation &allocation = prepared[index];
    allocation.pressure =
        *std::max_element(pressure.begin() + allocation.firstSegment,
                          pressure.begin() + allocation.endSegment);
    llvm::APInt duration(192, record.last - record.first);
    ++duration;
    allocation.weight = duration * duration * record.size;
  }

  order.resize(records.size());
  std::iota(order.begin(), order.end(), size_t(0));

  std::sort(order.begin(), order.end(), [&](size_t left, size_t right) {
    if (prepared[left].pressure != prepared[right].pressure)
      return prepared[left].pressure > prepared[right].pressure;
    if (records[left].alignment != records[right].alignment)
      return records[left].alignment > records[right].alignment;
    if (prepared[left].weight != prepared[right].weight)
      return prepared[left].weight.ugt(prepared[right].weight);
    uint64_t leftSpan = records[left].last - records[left].first;
    uint64_t rightSpan = records[right].last - records[right].first;
    if (leftSpan != rightSpan)
      return leftSpan > rightSpan;
    if (records[left].first != records[right].first)
      return records[left].first < records[right].first;
    return left < right;
  });

  output.offsets.reserve(records.size());
  for (const AllocationRecord &record : records)
    output.offsets.push_back({record.identity, 0});

  llvm::SmallVector<uint64_t> skyline(boundaries.size(), 0);
  nonFloor.resize(boundaries.size() + 1, 0);
  size_t remaining = records.size();

  while (remaining != 0) {
    uint64_t floor = *std::min_element(skyline.begin(), skyline.end());
    uint64_t nextFloor = capacity;

    for (size_t segment = 0; segment < skyline.size(); ++segment) {
      nonFloor[segment + 1] = nonFloor[segment] + (skyline[segment] != floor);
      if (skyline[segment] > floor)
        nextFloor = std::min(nextFloor, skyline[segment]);
    }

    size_t selected = records.size();

    for (size_t index : order) {
      const PreparedAllocation &allocation = prepared[index];
      if (allocation.placed ||
          nonFloor[allocation.firstSegment] != nonFloor[allocation.endSegment])
        continue;
      uint64_t alignment = records[index].alignment;
      uint64_t remainder = floor % alignment;

      if (remainder == 0) {
        selected = index;
        break;
      }

      uint64_t padding = alignment - remainder;
      if (padding <= capacity - floor)
        nextFloor = std::min(nextFloor, floor + padding);
    }

    if (selected == records.size()) {
      if (nextFloor <= floor)
        return llvm::createStringError("Storage capacity cannot satisfy "
                                       "allocation alignment and lifetimes");

      for (uint64_t &height : skyline)
        height = std::max(height, nextFloor);
      continue;
    }

    if (records[selected].size > capacity - floor)
      return llvm::createStringError(
          "Storage capacity cannot satisfy allocation alignment and lifetimes");
    uint64_t top = floor + records[selected].size;
    PreparedAllocation &allocation = prepared[selected];
    std::fill(skyline.begin() + allocation.firstSegment,
              skyline.begin() + allocation.endSegment, top);
    output.offsets[selected].offset = floor;
    output.peakUsage = std::max(output.peakUsage, top);
    allocation.placed = true;
    --remaining;
  }

  return output;
}
