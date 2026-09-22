#ifndef MLIR_TARGET_DARWINN_EXECUTIONPLAN_H
#define MLIR_TARGET_DARWINN_EXECUTIONPLAN_H

#include "mlir/Target/Darwinn/Serialization.h"
#include <string>

namespace mlir::darwinn {

enum class BufferKind {
  InputActivation,
  OutputActivation,
  ParameterRegion,
  Scratch
};

struct BufferRef {
  BufferKind kind = BufferKind::InputActivation;
  uint32_t index = 0;

  bool operator==(const BufferRef &other) const {
    return kind == other.kind && index == other.index;
  }
};

struct BufferRequirement {
  BufferRef buffer;
  std::optional<uint64_t> byteSize;
};

struct HibAddressPatch {
  InstructionRef instruction;
  BufferRef buffer;
  uint64_t byteOffset = 0;
  std::optional<uint64_t> accessedBytes;
};

struct Dispatch {
  uint32_t chunk = 0;
};

struct LiveTensorSpan {
  uint32_t startRow = 0;
  uint32_t rowCount = 0;
};

struct FenceWait {
  InstructionRef instruction;
  uint32_t tag = 0;
  bool softwarePreemption = false;
  std::optional<llvm::SmallVector<LiveTensorSpan, 0>> liveTensors;
};

struct CheckPreemption {};

using ControlAction =
    std::variant<HibAddressPatch, Dispatch, FenceWait, CheckPreemption>;

struct ExecutionPlan {
  explicit ExecutionPlan(const EncodedProgram &program) : program(program) {}
  ExecutionPlan(EncodedProgram &&) = delete;

  const EncodedProgram &program;
  std::string entryName;
  llvm::SmallVector<BufferRequirement, 0> buffers;
  llvm::SmallVector<ControlAction, 0> actions;
  uint32_t narrowMemoryRows = 0;
};

llvm::Error validateExecutionPlan(const ExecutionPlan &plan);

}

#endif
