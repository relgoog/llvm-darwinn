#include "mlir/Dialect/Darwinn/Transforms/LowerRuntime.h"
#include "mlir/Dialect/DiveVm/IR/DiveVmOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

namespace {

struct RuntimeCall {
  Operation *operation;
  StringRef callee;
  LLVM::LLVMFunctionType type;
  Block *entry;
};

StringRef getRuntimeCallee(Operation *operation) {
  return llvm::TypeSwitch<Operation *, StringRef>(operation)
      .Case<dive_vm::AddressOfInputActivationOp>(
          [](auto) { return "DiveVm_GetAddressOfInputActivation"; })
      .Case<dive_vm::AddressOfOutputActivationOp>(
          [](auto) { return "DiveVm_GetAddressOfOutputActivation"; })
      .Case<dive_vm::AddressOfParameterRegionOp>(
          [](auto) { return "DiveVm_GetAddressOfParameterRegion"; })
      .Case<dive_vm::AddressOfScratchOp>(
          [](auto) { return "DiveVm_GetAddressOfScratch"; })
      .Case<dive_vm::InstructionChunkOp>(
          [](auto) { return "DiveVm_GetAddressAndSizeofInstructionChunk"; })
      .Case<dive_vm::DispatchHardwareInstructionOp>(
          [](auto) { return "DiveTpu_EnqueueInstructions"; })
      .Case<dive_vm::WaitForFenceCompletionOp>(
          [](auto) { return "DiveTpu_WaitForFenceCompletion"; })
      .Case<dive_vm::PerformSoftwarePreemptionIfRequestedOp>(
          [](auto) { return "DiveTpu_PerformSoftwarePreemptionIfRequested"; })
      .Case<dive_vm::GetStatusCodeOp>(
          [](auto) { return "DiveVm_GetStatusCode"; })
      .Default(StringRef());
}

LLVM::LLVMFunctionType getRuntimeType(Operation *operation,
                                      OpBuilder &builder) {
  Type pointer = LLVM::LLVMPointerType::get(builder.getContext());
  Type result = LLVM::LLVMVoidType::get(builder.getContext());
  SmallVector<Type> arguments;

  if (isa<dive_vm::AddressOfInputActivationOp,
          dive_vm::AddressOfOutputActivationOp,
          dive_vm::AddressOfParameterRegionOp>(operation)) {
    arguments = {pointer, builder.getI32Type(), pointer};
  } else if (isa<dive_vm::AddressOfScratchOp>(operation)) {
    arguments = {pointer, pointer};
  } else if (isa<dive_vm::InstructionChunkOp>(operation)) {
    arguments = {pointer, builder.getI32Type(), pointer, pointer};
  } else if (isa<dive_vm::DispatchHardwareInstructionOp>(operation)) {
    arguments = {pointer, builder.getI64Type(), builder.getI32Type()};
  } else if (isa<dive_vm::WaitForFenceCompletionOp>(operation)) {
    arguments = {builder.getI1Type()};
  } else if (isa<dive_vm::GetStatusCodeOp>(operation)) {
    result = builder.getI32Type();
  }

  return LLVM::LLVMFunctionType::get(result, arguments, false);
}

}

LogicalResult mlir::darwinn::lowerDiveVmRuntime(ModuleOp module) {
  OpBuilder builder(module.getContext());
  SmallVector<RuntimeCall> calls;
  WalkResult collected = module.walk([&](Operation *operation) {
    StringRef callee = getRuntimeCallee(operation);
    if (callee.empty())
      return WalkResult::advance();
    if (failed(verify(operation, false)))
      return WalkResult::interrupt();

    auto function = operation->getParentOfType<func::FuncOp>();
    if (!function || function.isDeclaration() ||
        function->getParentOp() != module.getOperation()) {
      operation->emitError("runtime lowering requires a defined function "
                           "directly in the module");
      return WalkResult::interrupt();
    }

    for (Operation *parent = operation->getParentOp(); parent != function;
         parent = parent->getParentOp()) {
      if (parent->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
        operation->emitError("runtime lowering cannot place output storage "
                             "across an isolated region");
        return WalkResult::interrupt();
      }
    }

    LLVM::LLVMFunctionType type = getRuntimeType(operation, builder);
    if (Operation *symbol = SymbolTable::lookupSymbolIn(module, callee)) {
      auto declaration = dyn_cast<LLVM::LLVMFuncOp>(symbol);
      if (!declaration || declaration.getFunctionType() != type ||
          declaration.getCConv() != LLVM::cconv::CConv::C) {
        operation->emitError() << "runtime helper symbol " << callee
                               << " requires LLVM function type " << type
                               << " with the C calling convention";
        return WalkResult::interrupt();
      }
    }

    calls.push_back({operation, callee, type, &function.getBody().front()});
    return WalkResult::advance();
  });

  if (collected.wasInterrupted())
    return failure();

  llvm::StringMap<LLVM::LLVMFuncOp> declarations;
  for (const RuntimeCall &call : calls) {
    if (declarations.contains(call.callee))
      continue;

    auto declaration = module.lookupSymbol<LLVM::LLVMFuncOp>(call.callee);
    if (!declaration) {
      builder.setInsertionPointToStart(module.getBody());
      declaration = LLVM::LLVMFuncOp::create(builder, call.operation->getLoc(),
                                             call.callee, call.type);
    }

    declarations.try_emplace(call.callee, declaration);
  }

  for (const RuntimeCall &call : calls) {
    Operation *operation = call.operation;
    Location location = operation->getLoc();
    LLVM::LLVMFuncOp callee = declarations.lookup(call.callee);
    builder.setInsertionPoint(operation);
    SmallVector<Value> arguments;
    SmallVector<Value> slots;
    bool resolvesAddress =
        isa<dive_vm::AddressOfInputActivationOp,
            dive_vm::AddressOfOutputActivationOp,
            dive_vm::AddressOfParameterRegionOp, dive_vm::AddressOfScratchOp,
            dive_vm::InstructionChunkOp>(operation);

    if (resolvesAddress) {
      arguments.push_back(operation->getOperand(0));
      if (auto index = operation->getAttrOfType<IntegerAttr>("index")) {
        arguments.push_back(LLVM::ConstantOp::create(
            builder, location, builder.getI32Type(),
            builder.getI32IntegerAttr(static_cast<uint32_t>(index.getInt()))));
      }

      {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(call.entry);
        Value one = LLVM::ConstantOp::create(builder, location,
                                             builder.getI64Type(), 1);
        unsigned count = isa<dive_vm::InstructionChunkOp>(operation) ? 2 : 1;

        for (unsigned index = 0; index < count; ++index) {
          slots.push_back(LLVM::AllocaOp::create(
              builder, location,
              LLVM::LLVMPointerType::get(module.getContext()),
              builder.getI64Type(), one, 8));
        }
      }

      llvm::append_range(arguments, slots);
    } else if (auto dispatch = dyn_cast<dive_vm::DispatchHardwareInstructionOp>(
                   operation)) {
      Value index = LLVM::ConstantOp::create(
          builder, location, builder.getI32Type(),
          builder.getI32IntegerAttr(
              static_cast<uint32_t>(dispatch.getIndex())));
      arguments = {dispatch.getBytes(), dispatch.getSize(), index};
    } else if (auto wait =
                   dyn_cast<dive_vm::WaitForFenceCompletionOp>(operation)) {
      arguments.push_back(
          LLVM::ConstantOp::create(builder, location, builder.getI1Type(),
                                   wait.getSoftwarePreemption()));
    }

    auto invocation =
        LLVM::CallOp::create(builder, location, callee, arguments);
    SmallVector<Value> results;
    for (Value slot : slots) {
      results.push_back(LLVM::LoadOp::create(builder, location,
                                             builder.getI64Type(), slot, 8));
    }

    if (isa<dive_vm::InstructionChunkOp>(operation)) {
      results[0] = LLVM::IntToPtrOp::create(
          builder, location, LLVM::LLVMPointerType::get(module.getContext()),
          results[0]);
    } else if (isa<dive_vm::GetStatusCodeOp>(operation)) {
      results.push_back(invocation.getResult());
    }

    operation->replaceAllUsesWith(results);
    operation->erase();
  }

  return success();
}
