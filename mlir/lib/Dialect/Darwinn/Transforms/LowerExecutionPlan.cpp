#include "mlir/Dialect/Darwinn/Transforms/LowerExecutionPlan.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DiveVm/IR/DiveVmOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/STLExtras.h"
#include <optional>

using namespace mlir;
using namespace mlir::darwinn;

FailureOr<func::FuncOp>
mlir::darwinn::lowerExecutionPlan(ModuleOp module, Location location,
                                  const ExecutionPlan &plan) {
  if (auto error = validateExecutionPlan(plan))
    return emitError(location, llvm::toString(std::move(error)));
  if (SymbolTable::lookupSymbolIn(module, plan.entryName))
    return emitError(location, "execution entry symbol already exists");

  MLIRContext *context = module.getContext();
  context->loadDialect<arith::ArithDialect, dive_vm::DiveVmDialect,
                       func::FuncDialect, LLVM::LLVMDialect>();
  OpBuilder builder(context);
  auto pointerType = LLVM::LLVMPointerType::get(context);
  auto integerType = builder.getI64Type();
  auto functionType = builder.getFunctionType({pointerType, pointerType},
                                              {builder.getI32Type()});
  OwningOpRef<func::FuncOp> function(
      func::FuncOp::create(location, plan.entryName, functionType));
  Block *entry = function->addEntryBlock();
  builder.setInsertionPointToStart(entry);
  Value programInfo = entry->getArgument(0);
  Value request = entry->getArgument(1);
  llvm::SmallVector<Value, 0> addresses;

  for (const auto &requirement : plan.buffers) {
    auto index = builder.getI32IntegerAttr(requirement.buffer.index);
    Value address;

    switch (requirement.buffer.kind) {
    case BufferKind::InputActivation:
      address = dive_vm::AddressOfInputActivationOp::create(
          builder, location, integerType, request, index);
      break;
    case BufferKind::OutputActivation:
      address = dive_vm::AddressOfOutputActivationOp::create(
          builder, location, integerType, request, index);
      break;
    case BufferKind::ParameterRegion:
      address = dive_vm::AddressOfParameterRegionOp::create(
          builder, location, integerType, programInfo, index);
      break;
    case BufferKind::Scratch:
      address = dive_vm::AddressOfScratchOp::create(builder, location,
                                                    integerType, programInfo);
      break;
    }

    addresses.push_back(address);
  }

  struct ResolvedChunk {
    uint32_t index;
    Value bytes;
    Value size;
    Value buffer;
  };

  std::optional<ResolvedChunk> resolved;
  auto resolveChunk = [&](uint32_t index) -> ResolvedChunk & {
    if (resolved && resolved->index == index)
      return *resolved;
    auto chunk = dive_vm::InstructionChunkOp::create(
        builder, location, pointerType, integerType, programInfo,
        builder.getI64IntegerAttr(index));
    resolved = ResolvedChunk{index, chunk.getBytes(), chunk.getSize(), Value()};
    return *resolved;
  };

  auto resolveBuffer = [&](ResolvedChunk &chunk) -> Value {
    if (chunk.buffer)
      return chunk.buffer;
    auto arrayType = LLVM::LLVMArrayType::get(integerType, 1);
    auto descriptorType = LLVM::LLVMStructType::getLiteral(
        context, {pointerType, pointerType, integerType, arrayType, arrayType});
    Value descriptor =
        LLVM::PoisonOp::create(builder, location, descriptorType);
    Value zero = LLVM::ConstantOp::create(builder, location, integerType,
                                          builder.getI64IntegerAttr(0));
    Value one = LLVM::ConstantOp::create(builder, location, integerType,
                                         builder.getI64IntegerAttr(1));
    descriptor = LLVM::InsertValueOp::create(builder, location, descriptor,
                                             chunk.bytes, ArrayRef<int64_t>{0});
    descriptor = LLVM::InsertValueOp::create(builder, location, descriptor,
                                             chunk.bytes, ArrayRef<int64_t>{1});
    descriptor = LLVM::InsertValueOp::create(builder, location, descriptor,
                                             zero, ArrayRef<int64_t>{2});
    descriptor = LLVM::InsertValueOp::create(
        builder, location, descriptor, chunk.size, ArrayRef<int64_t>{3, 0});
    descriptor = LLVM::InsertValueOp::create(builder, location, descriptor, one,
                                             ArrayRef<int64_t>{4, 0});
    auto bufferType =
        MemRefType::get({ShapedType::kDynamic}, builder.getI8Type());
    chunk.buffer = UnrealizedConversionCastOp::create(builder, location,
                                                      bufferType, descriptor)
                       .getResult(0);
    return chunk.buffer;
  };

  for (const auto &action : plan.actions) {
    std::visit(
        llvm::makeVisitor(
            [&](const HibAddressPatch &patch) {
              auto requirement =
                  llvm::find_if(plan.buffers, [&](const auto &value) {
                    return value.buffer == patch.buffer;
                  });
              Value value = addresses[requirement - plan.buffers.begin()];

              if (patch.byteOffset != 0) {
                Value offset = arith::ConstantOp::create(
                    builder, location, integerType,
                    builder.getIntegerAttr(integerType,
                                           APInt(64, patch.byteOffset)));
                value = arith::AddIOp::create(builder, location, value, offset);
              }

              auto &chunk = resolveChunk(patch.instruction.chunk);
              auto *layout = plan.program.findInstruction(patch.instruction);
              dive_vm::PutBitsOp::create(
                  builder, location, resolveBuffer(chunk), value,
                  builder.getI32IntegerAttr(64),
                  builder.getI64IntegerAttr(*layout->hibAddressBitOffset));
            },
            [&](const Dispatch &dispatch) {
              auto &chunk = resolveChunk(dispatch.chunk);
              dive_vm::DispatchHardwareInstructionOp::create(
                  builder, location, chunk.bytes, chunk.size,
                  builder.getI64IntegerAttr(dispatch.chunk));
              resolved.reset();
            },
            [&](const FenceWait &wait) {
              dive_vm::WaitForFenceCompletionOp::create(
                  builder, location,
                  builder.getBoolAttr(wait.softwarePreemption));
              resolved.reset();
            },
            [&](const CheckPreemption &) {
              dive_vm::PerformSoftwarePreemptionIfRequestedOp::create(builder,
                                                                      location);
              resolved.reset();
            }),
        action);
  }

  Value status =
      dive_vm::GetStatusCodeOp::create(builder, location, builder.getI32Type());
  func::ReturnOp::create(builder, location, status);

  if (failed(verify(function.get())))
    return failure();
  func::FuncOp result = function.release();
  module.push_back(result);
  return result;
}
