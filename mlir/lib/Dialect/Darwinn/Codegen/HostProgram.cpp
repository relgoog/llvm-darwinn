#include "Codegen.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include <map>

using namespace mlir;
using namespace mlir::darwinn;
using namespace mlir::darwinn::codegen;

namespace {

enum class Root { InputActivation, OutputActivation, ParameterRegion, Scratch };

constexpr std::array<Root, 4> kRoots = {Root::InputActivation,
                                        Root::OutputActivation,
                                        Root::ParameterRegion, Root::Scratch};

Root rootOf(HibRoot root) {
  switch (root) {
  case HibRoot::InputActivation:
    return Root::InputActivation;
  case HibRoot::OutputActivation:
    return Root::OutputActivation;
  case HibRoot::ParameterRegion:
  case HibRoot::Parameter:
  case HibRoot::ParameterFill:
    return Root::ParameterRegion;
  case HibRoot::Scratch:
    return Root::Scratch;
  }
  llvm_unreachable("unknown hib root");
}

} // namespace

FailureOr<SmallVector<HostEvent, 0>>
codegen::hostEvents(const GeneratedProgram &program,
                    const EncodedProgram &encoded) {
  int64_t slotCount = program.hibs.size();
  auto rootSlot = [&](Root root) {
    return slotCount + static_cast<int64_t>(root);
  };
  int64_t nextSlot = slotCount - 1;
  SmallVector<HostEvent, 0> events;
  auto address = [&](int64_t index, bool hoisted) {
    const Hib &hib = program.hibs[index];
    int64_t slot = nextSlot--;
    int64_t root = rootSlot(rootOf(hib.root));
    bool parameter = rootOf(hib.root) == Root::ParameterRegion;
    if (!hoisted || (hib.offset != 0 && parameter)) {
      events.push_back({HostEvent::Kind::Add, root, slot, hib.offset});
    } else {
      events.push_back({HostEvent::Kind::Copy, root, slot, 0});
      if (hib.offset != 0)
        events.push_back({HostEvent::Kind::Add, slot, slot, hib.offset});
    }
    return slot;
  };
  ArrayRef<ChunkMeta> chunks = program.meta;
  auto precededByWait = [&](size_t chunk) {
    return chunk > 0 && chunks[chunk - 1].interrupt.has_value();
  };
  auto bitOffset = [&](InstructionRef reference) -> std::optional<int64_t> {
    const InstructionLayout *layout = encoded.findInstruction(reference);
    if (!layout || !layout->hibAddressBitOffset)
      return std::nullopt;
    return *layout->hibAddressBitOffset;
  };

  events.push_back({HostEvent::Kind::Dispatch, 0, 0, 0});
  for (Root root : kRoots)
    events.push_back(
        {HostEvent::Kind::Root, rootSlot(root), static_cast<int64_t>(root), 0});
  std::map<size_t, int64_t> pending;
  for (size_t chunk = 1; chunk < chunks.size(); ++chunk) {
    if (precededByWait(chunk))
      events.push_back(
          {HostEvent::Kind::Wait,
           *chunks[chunk - 1].interrupt == Role::PreemptionInterrupt, 0, 0});
    SmallVector<std::pair<int64_t, InstructionRef>> puts;
    ArrayRef<std::pair<int64_t, InstructionRef>> own = chunks[chunk].hibs;
    if (auto hoisted = pending.find(chunk); hoisted != pending.end()) {
      puts.push_back({hoisted->second, own.front().second});
      own = own.drop_front();
      pending.erase(hoisted);
    }
    for (const auto &[index, reference] : own)
      puts.push_back({address(index, false), reference});
    if (chunk + 1 < chunks.size() && chunks[chunk + 1].startsWithInputHib &&
        !precededByWait(chunk + 1))
      pending[chunk + 1] = address(chunks[chunk + 1].hibs.front().first, true);
    for (const auto &[slot, reference] : puts) {
      std::optional<int64_t> offset = bitOffset(reference);
      if (!offset)
        return failure();
      events.push_back(
          {HostEvent::Kind::Put, slot, static_cast<int64_t>(chunk), *offset});
    }
    events.push_back(
        {HostEvent::Kind::Dispatch, static_cast<int64_t>(chunk), 0, 0});
  }
  events.push_back({HostEvent::Kind::Wait, 0, 0, 0});
  return events;
}

OwningOpRef<ModuleOp> codegen::hostModule(MLIRContext *context,
                                          ArrayRef<HostEvent> events,
                                          int64_t slotCount,
                                          StringRef programSymbol) {
  Location location = UnknownLoc::get(context);
  OpBuilder builder(context);
  OwningOpRef<ModuleOp> module = ModuleOp::create(location);
  builder.setInsertionPointToEnd(module->getBody());
  auto pointer = LLVM::LLVMPointerType::get(context);
  auto i1 = builder.getI1Type();
  auto i32 = builder.getI32Type();
  auto i64 = builder.getI64Type();
  auto voidType = LLVM::LLVMVoidType::get(context);
  auto declare = [&](StringRef name, Type result, ArrayRef<Type> arguments) {
    return LLVM::LLVMFuncOp::create(
        builder, location, name,
        LLVM::LLVMFunctionType::get(result, arguments));
  };
  auto malloc = declare("malloc", pointer, {i64});
  auto preemption =
      declare("DiveTpu_PerformSoftwarePreemptionIfRequested", voidType, {});
  auto inputAddress = declare("DiveVm_GetAddressOfInputActivation", voidType,
                              {pointer, i32, pointer});
  auto outputAddress = declare("DiveVm_GetAddressOfOutputActivation", voidType,
                               {pointer, i32, pointer});
  auto parameterAddress = declare("DiveVm_GetAddressOfParameterRegion",
                                  voidType, {pointer, i32, pointer});
  auto scratchAddress =
      declare("DiveVm_GetAddressOfScratch", voidType, {pointer, pointer});
  auto wait = declare("DiveTpu_WaitForFenceCompletion", voidType, {i1});
  auto putBits =
      declare("DiveVm_PutBits", voidType, {pointer, i64, i32, i64, i1});
  auto enqueue =
      declare("DiveTpu_EnqueueInstructions", voidType, {pointer, i64, i32});
  auto chunkAddress = declare("DiveVm_GetAddressAndSizeofInstructionChunk",
                              voidType, {pointer, i32, pointer, pointer});
  auto status = declare("DiveVm_GetStatusCode", i32, {});

  auto constant = [&](Type type, int64_t value) -> Value {
    return LLVM::ConstantOp::create(builder, location, type,
                                    builder.getIntegerAttr(type, value));
  };
  auto call = [&](LLVM::LLVMFuncOp callee, ValueRange arguments) {
    return LLVM::CallOp::create(builder, location, callee, arguments);
  };
  auto slotAlloca = [&]() -> Value {
    return LLVM::AllocaOp::create(builder, location, pointer, i64,
                                  constant(i64, 1), 8);
  };
  auto loadValue = [&](Value address) -> Value {
    return LLVM::LoadOp::create(builder, location, i64, address, 4);
  };
  auto storeValue = [&](Value value, Value address) {
    LLVM::StoreOp::create(builder, location, value, address, 4);
  };

  SmallVector<Type> mainArguments;
  for (int root = 0; root < 4; ++root)
    llvm::append_range(mainArguments, SmallVector<Type>{pointer, pointer, i64});
  mainArguments.append({pointer, pointer});
  std::string mainName = (programSymbol + "_main").str();
  auto main = declare(mainName, voidType, mainArguments);
  Block *body = main.addEntryBlock(builder);
  builder.setInsertionPointToStart(body);
  Value info = body->getArgument(12);
  SmallVector<Value> slots(slotCount + kRoots.size());
  for (int64_t slot = slotCount + kRoots.size() - 1; slot >= 0; --slot)
    slots[slot] = slotAlloca();
  struct Resolved {
    int64_t chunk;
    Value bytes;
    Value size;
  };
  std::optional<Resolved> resolved;
  auto resolve = [&](int64_t chunk) -> Resolved & {
    if (resolved && resolved->chunk == chunk)
      return *resolved;
    Value bytes = LLVM::AllocaOp::create(builder, location, pointer, pointer,
                                         constant(i32, 1), 8);
    Value size = LLVM::AllocaOp::create(builder, location, pointer, i64,
                                        constant(i32, 1), 8);
    call(chunkAddress, {info, constant(i32, chunk), bytes, size});
    Value loadedSize = loadValue(size);
    Value loadedBytes =
        LLVM::LoadOp::create(builder, location, pointer, bytes, 8);
    resolved = Resolved{chunk, loadedBytes, loadedSize};
    return *resolved;
  };
  for (const HostEvent &event : events) {
    switch (event.kind) {
    case HostEvent::Kind::Root:
      storeValue(loadValue(body->getArgument(3 * event.second + 1)),
                 slots[event.first]);
      break;
    case HostEvent::Kind::Copy:
      storeValue(loadValue(slots[event.first]), slots[event.second]);
      break;
    case HostEvent::Kind::Add: {
      Value sum =
          LLVM::AddOp::create(builder, location, loadValue(slots[event.first]),
                              constant(i64, event.third));
      storeValue(sum, slots[event.second]);
      break;
    }
    case HostEvent::Kind::Put: {
      Resolved &chunk = resolve(event.second);
      call(putBits,
           {chunk.bytes, loadValue(slots[event.first]), constant(i32, 64),
            constant(i64, event.third), constant(i1, 0)});
      break;
    }
    case HostEvent::Kind::Dispatch: {
      Resolved &chunk = resolve(event.first);
      call(enqueue, {chunk.bytes, chunk.size, constant(i32, event.first)});
      resolved.reset();
      break;
    }
    case HostEvent::Kind::Wait:
      call(wait, {constant(i1, event.first)});
      break;
    }
  }
  LLVM::ReturnOp::create(builder, location, ValueRange{});

  builder.setInsertionPointToEnd(module->getBody());
  auto offload = declare("TPUOffloadFunc", i32, {pointer, pointer});
  Block *entry = offload.addEntryBlock(builder);
  builder.setInsertionPointToStart(entry);
  Value programInfo = entry->getArgument(0), request = entry->getArgument(1);
  Value parameter = slotAlloca(), scratch = slotAlloca(), output = slotAlloca(),
        input = slotAlloca();
  call(scratchAddress, {programInfo, scratch});
  call(parameterAddress, {programInfo, constant(i32, 0), parameter});
  auto activation = [&](LLVM::LLVMFuncOp resolver) -> Value {
    Value raw = call(malloc, {constant(i64, 128)}).getResult();
    Value integer = LLVM::PtrToIntOp::create(builder, location, i64, raw);
    Value bumped =
        LLVM::AddOp::create(builder, location, integer, constant(i64, 63));
    Value remainder =
        LLVM::URemOp::create(builder, location, bumped, constant(i64, 64));
    Value aligned = LLVM::SubOp::create(builder, location, bumped, remainder);
    Value slot = LLVM::IntToPtrOp::create(builder, location, pointer, aligned);
    call(resolver, {request, constant(i32, 0), slot});
    return LLVM::IntToPtrOp::create(builder, location, pointer,
                                    loadValue(slot));
  };
  Value outputPointer = activation(outputAddress);
  Value inputPointer = activation(inputAddress);
  call(preemption, {});
  storeValue(LLVM::PtrToIntOp::create(builder, location, i64, inputPointer),
             input);
  storeValue(LLVM::PtrToIntOp::create(builder, location, i64, outputPointer),
             output);
  Value zero = constant(i64, 0);
  call(main, {input, input, zero, output, output, zero, parameter, parameter,
              zero, scratch, scratch, zero, programInfo, request});
  call(preemption, {});
  Value code = call(status, {}).getResult();
  LLVM::ReturnOp::create(builder, location, code);
  return module;
}
