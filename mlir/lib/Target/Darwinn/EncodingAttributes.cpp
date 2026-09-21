#include "mlir/Target/Darwinn/EncodingAttributes.h"

#include "mlir/Dialect/Darwinn/IR/InstructionOps.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::darwinn;

namespace {

template <size_t count>
std::array<bool, count> bitmap(DenseBoolArrayAttr attribute) {
  assert(attribute.size() == count);
  std::array<bool, count> result;
  llvm::copy(attribute.asArrayRef(), result.begin());
  return result;
}

MemoryPowerState powerState(isa::MemoryPowerState state) {
  switch (state) {
  case isa::MemoryPowerState::Active:
    return MemoryPowerState::Active;
  case isa::MemoryPowerState::DeepSleep:
    return MemoryPowerState::DeepSleep;
  }
  llvm_unreachable("invalid verified memory power state");
}

}

llvm::Expected<Header> mlir::darwinn::convertHeader(isa::HeaderAttr attribute) {
  if (!attribute)
    return llvm::createStringError("missing instruction header");

  Header result;
  if (auto predicate = attribute.getPredicate()) {
    if (predicate.getRegisterId() >= 8)
      return llvm::createStringError(
          "predicate register must be between 0 and 7");
    auto converted =
        Predicate::create(predicate.getRegisterId(), predicate.getPolarity());
    if (!converted)
      return converted.takeError();
    result.predicate = *converted;
  }

  result.enableTrace = attribute.getEnableTrace();
  result.enableSequencerOverwrite = attribute.getEnableSequencerOverwrite();
  return result;
}

llvm::Expected<TileHeader>
mlir::darwinn::convertTileHeader(isa::TileHeaderAttr attribute) {
  if (!attribute || !attribute.getMulticastBitmap() ||
      attribute.getMulticastBitmap().size() != 16)
    return llvm::createStringError(
        "tile multicast bitmap must contain 16 bits");
  if (attribute.getTag() >= (uint32_t{1} << 20))
    return llvm::createStringError("tile tag must fit in 20 bits");
  auto header = convertHeader(attribute.getHeader());
  if (!header)
    return header.takeError();
  if (header->enableSequencerOverwrite)
    return llvm::createStringError(
        "G5 tile headers do not support sequencer overwrite");
  return TileHeader{*header, bitmap<16>(attribute.getMulticastBitmap()),
                    attribute.getTag()};
}

llvm::Expected<OverwriteInfo>
mlir::darwinn::convertOverwrite(isa::OverwriteInfoAttr attribute) {
  if (!attribute || attribute.getBaseAddressSource() >= 32 ||
      attribute.getMulticastBitmapSource() >= 32)
    return llvm::createStringError(
        "overwrite register sources must fit in 5 bits");
  return OverwriteInfo{
      static_cast<uint8_t>(attribute.getBaseAddressSource()),
      static_cast<uint8_t>(attribute.getMulticastBitmapSource()),
      attribute.getOverwriteMulticastBitmap()};
}

llvm::Expected<Instruction>
mlir::darwinn::convertCoreInstruction(Operation *operation) {
  std::string diagnostic;
  {
    ScopedDiagnosticHandler handler(
        operation->getContext(), [&](Diagnostic &error) {
          llvm::raw_string_ostream stream(diagnostic);
          error.print(stream);
        });
    if (failed(verify(operation)))
      return llvm::createStringError(diagnostic);
  }

  return llvm::TypeSwitch<Operation *, llvm::Expected<Instruction>>(operation)
      .Case<isa::NoOp, isa::HaltOp, isa::InterruptOp, isa::LoadProgramOp,
            isa::ExecuteOp>([](auto scalar) -> llvm::Expected<Instruction> {
        auto header = convertHeader(scalar.getHeader());
        if (!header)
          return header.takeError();

        using Op = decltype(scalar);
        ScalarInstruction instruction;
        if constexpr (std::is_same_v<Op, isa::NoOp>) {
          instruction = NoOp{};
        } else if constexpr (std::is_same_v<Op, isa::HaltOp>) {
          instruction = Halt{scalar.getGotoHalt(), scalar.getInterrupt(),
                             static_cast<uint8_t>(scalar.getInterruptId()),
                             scalar.getGotoSleep()};
        } else if constexpr (std::is_same_v<Op, isa::InterruptOp>) {
          instruction =
              Interrupt{static_cast<uint8_t>(scalar.getInterruptId())};
        } else if constexpr (std::is_same_v<Op, isa::LoadProgramOp>) {
          instruction =
              LoadProgram{static_cast<uint32_t>(scalar.getStartPc()),
                          static_cast<uint32_t>(scalar.getNumberChunks())};
        } else {
          instruction = Execute{static_cast<uint32_t>(scalar.getStartPc()),
                                scalar.getLoadPc()};
        }

        return Instruction(ScalarPacket{*header, instruction});
      })
      .Case([&](isa::TileFenceOp operation) -> llvm::Expected<Instruction> {
        auto header = convertTileHeader(operation.getHeader());
        if (!header)
          return header.takeError();

        TileFence fence;
        fence.resetSyncFlag = bitmap<22>(operation.getResetSyncFlagAttr());
        fence.resetSyncFlagThreadIds =
            bitmap<4>(operation.getResetSyncFlagThreadIdsAttr());
        fence.syncResetValue = operation.getSyncResetValue();
        fence.waitIdle = bitmap<13>(operation.getWaitIdleAttr());
        fence.waitIdleThreadIds =
            bitmap<4>(operation.getWaitIdleThreadIdsAttr());
        fence.waitProducerCountValid =
            bitmap<2>(operation.getWaitProducerCountValidAttr());
        fence.producerCountValue = operation.getProducerCountValue();
        fence.increment = operation.getIncrement();
        fence.halt = operation.getHalt();
        fence.narrowMemPowerState =
            powerState(operation.getNarrowMemPowerState());
        fence.wideMemPowerState = powerState(operation.getWideMemPowerState());
        fence.virtualChannelSubscription =
            bitmap<8>(operation.getVirtualChannelSubscriptionAttr());
        fence.virtualChannelSubscriptionValid =
            operation.getVirtualChannelSubscriptionValid();
        fence.incrementGlobalTokenA = operation.getIncrementGlobalTokenA();
        fence.incrementGlobalTokenB = operation.getIncrementGlobalTokenB();
        fence.discardRemainingBytes = operation.getDiscardRemainingBytes();
        return Instruction(TilePacket{*header, fence});
      })
      .Case([&](isa::ScalarFenceOp operation) -> llvm::Expected<Instruction> {
        auto header = convertHeader(operation.getHeader());
        if (!header)
          return header.takeError();

        ScalarFence fence;
        fence.resetTileFence = bitmap<20>(operation.getResetTileFenceAttr());
        fence.resetSyncFlag = bitmap<17>(operation.getResetSyncFlagAttr());
        fence.resetValue = operation.getResetValue();
        fence.expectedTileFenceCount = operation.getExpectedTileFenceCount();
        fence.tileFenceValid = bitmap<20>(operation.getTileFenceValidAttr());
        auto expected = operation.getExpectedScalarSyncFlag();
        for (auto [index, valid, value] :
             llvm::enumerate(expected.getValid().asArrayRef(),
                             expected.getValues().asArrayRef())) {
          if (valid)
            fence.expectedScalarSyncFlag[index] = value;
        }
        fence.waitIdle = bitmap<5>(operation.getWaitIdleAttr());
        fence.incrementGlobalTokenA = operation.getIncrementGlobalTokenA();
        fence.incrementGlobalTokenB = operation.getIncrementGlobalTokenB();
        fence.preemptable = operation.getPreemptable();
        fence.discardRemainingBytes = operation.getDiscardRemainingBytes();
        fence.virtualChannelSubscription =
            bitmap<8>(operation.getVirtualChannelSubscriptionAttr());
        fence.virtualChannelSubscriptionValid =
            operation.getVirtualChannelSubscriptionValid();
        fence.sendInterrupt = operation.getSendInterrupt();
        return Instruction(TaggedPacket{
            *header, static_cast<uint32_t>(operation.getTag()), fence});
      })
      .Case([&](isa::TileLoadStoreOp operation) -> llvm::Expected<Instruction> {
        auto header = convertTileHeader(operation.getHeader());
        if (!header)
          return header.takeError();

        TileLoadStore memory;
        switch (operation.getMemoryOperation()) {
        case isa::TileMemoryOperation::Load:
          memory.operation = TileMemoryOperation::Load;
          break;
        case isa::TileMemoryOperation::Store:
          memory.operation = TileMemoryOperation::Store;
          break;
        }
        memory.rowAddress = operation.getRowAddress();
        memory.registerBurstLength = operation.getRegisterBurstLength();
        memory.baseRegister = operation.getBaseRegister();
        memory.useImmediate = operation.getUseImmediate();
        memory.immediateValue = operation.getImmediateValue();
        memory.useTraceRegisters = operation.getUseTraceRegisters();
        memory.scalarRegister = operation.getScalarRegister();
        switch (operation.getReplaceField()) {
        case isa::TileScalarReplacement::None:
          memory.replaceField = TileScalarReplacement::None;
          break;
        case isa::TileScalarReplacement::Address:
          memory.replaceField = TileScalarReplacement::Address;
          break;
        case isa::TileScalarReplacement::Immediate:
          memory.replaceField = TileScalarReplacement::Immediate;
          break;
        case isa::TileScalarReplacement::AddressAndImmediate:
          memory.replaceField = TileScalarReplacement::AddressAndImmediate;
          break;
        }
        memory.threadBitmap = bitmap<4>(operation.getThreadBitmapAttr());
        return Instruction(TilePacket{*header, memory});
      })
      .Case([&](isa::CoefficientTablesOp operation)
                -> llvm::Expected<Instruction> {
        auto header = convertTileHeader(operation.getHeader());
        if (!header)
          return header.takeError();
        auto overwrite = convertOverwrite(operation.getOverwrite());
        if (!overwrite)
          return overwrite.takeError();

        CoefficientTables coefficients;
        for (auto [index, segment] :
             llvm::enumerate(coefficients.splineSegments))
          llvm::copy(operation.getSplineSegments().slice(index * 5, 5),
                     segment.begin());
        llvm::copy(operation.getSegmentLowerBounds(),
                   coefficients.segmentLowerBounds.begin());
        coefficients.polynomialDegree = operation.getPolynomialDegree();
        coefficients.enhancedSquareBitmap =
            bitmap<8>(operation.getEnhancedSquareBitmapAttr());
        coefficients.threadMulticastBitmap =
            bitmap<4>(operation.getThreadMulticastBitmapAttr());
        return Instruction(
            ComputePacket{*header, *overwrite,
                          bitmap<8>(operation.getRegisterSourcedOperandsAttr()),
                          coefficients});
      })
      .Default([](Operation *unknown) -> llvm::Expected<Instruction> {
        return llvm::createStringError("unsupported Darwinn instruction " +
                                       unknown->getName().getStringRef());
      });
}
