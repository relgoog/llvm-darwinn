#include "mlir/Dialect/Darwinn/IR/InstructionOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::darwinn::isa;

#include "mlir/Dialect/Darwinn/IR/InstructionEnums.cpp.inc"
#include "mlir/Dialect/Darwinn/IR/InstructionOpsDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/Darwinn/IR/InstructionAttrs.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/Darwinn/IR/InstructionOps.cpp.inc"

void DarwinnIsaDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/Darwinn/IR/InstructionAttrs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Darwinn/IR/InstructionOps.cpp.inc"
      >();
}

LogicalResult
TargetConfigAttr::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                         TargetKind target, unsigned scalarPcBits, bool) {
  if (target != TargetKind::G5)
    return emitError() << "requires the G5 instruction target";
  if (scalarPcBits == 0 || scalarPcBits > 31)
    return emitError()
           << "scalar program counter width must be between 1 and 31";
  return success();
}

LogicalResult
PredicateAttr::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                      unsigned registerId, bool) {
  if (registerId >= 8)
    return emitError() << "predicate register must be between 0 and 7";
  return success();
}

LogicalResult
TileHeaderAttr::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                       HeaderAttr header, DenseBoolArrayAttr multicastBitmap,
                       uint32_t tag) {
  if (!header || !multicastBitmap)
    return emitError()
           << "tile header requires a common header and multicast bitmap";
  if (header.getEnableSequencerOverwrite())
    return emitError() << "G5 tile headers do not support sequencer overwrite";
  if (multicastBitmap.size() != 16)
    return emitError() << "tile multicast bitmap must contain 16 bits";
  if (tag >= (uint32_t{1} << 20))
    return emitError() << "tile tag must fit in 20 bits";
  return success();
}

LogicalResult
OverwriteInfoAttr::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                          unsigned baseAddressSource,
                          unsigned multicastBitmapSource, bool) {
  if (baseAddressSource >= 32 || multicastBitmapSource >= 32)
    return emitError() << "overwrite register sources must fit in 5 bits";
  return success();
}

LogicalResult ExpectedSyncFlagsAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError,
    DenseBoolArrayAttr valid, DenseI32ArrayAttr values) {
  if (!valid || !values || valid.size() != 17 || values.size() != 17)
    return emitError()
           << "expected scalar sync flags require 17 validity bits and values";

  for (auto [enabled, value] :
       llvm::zip(valid.asArrayRef(), values.asArrayRef())) {
    if (value < 0 || value >= (int32_t{1} << 25))
      return emitError()
             << "expected scalar sync flag value must fit in 25 bits";
    if (!enabled && value != 0)
      return emitError() << "inactive scalar sync flag values must be zero";
  }

  return success();
}

LogicalResult ProgramOp::verify() {
  auto config = getTargetConfig();
  if (failed(TargetConfigAttr::verify(
          [&]() { return emitOpError(); }, config.getTarget(),
          config.getScalarPcBits(), config.getSequencerOverwrite())))
    return failure();

  if (getBody().empty())
    return emitOpError("requires a block of independently loaded chunks");

  uint64_t expectedIndex = 0;
  for (Operation &operation : getBody().front()) {
    auto chunk = dyn_cast<ChunkOp>(operation);
    if (!chunk)
      return emitOpError("may contain only darwinn_isa.chunk operations");
    if (chunk.getIndex() != expectedIndex)
      return chunk.emitOpError(
          "index must follow its program order starting at zero");
    ++expectedIndex;
  }

  return success();
}

LogicalResult ChunkOp::verify() {
  if (getIndex() > UINT32_MAX)
    return emitOpError("index must fit in an unsigned 32-bit value");
  if (getBody().empty())
    return emitOpError("requires a block of independently encoded fragments");

  uint64_t expectedIndex = 0;
  for (Operation &operation : getBody().front()) {
    auto fragment = dyn_cast<FragmentOp>(operation);
    if (!fragment)
      return emitOpError("may contain only darwinn_isa.fragment operations");
    if (fragment.getIndex() != expectedIndex)
      return fragment.emitOpError(
          "index must follow its chunk order starting at zero");
    ++expectedIndex;
  }

  return success();
}

LogicalResult FragmentOp::verify() {
  if (getIndex() > UINT32_MAX)
    return emitOpError("index must fit in an unsigned 32-bit value");
  if (getBody().empty())
    return emitOpError("requires an instruction block");

  for (Operation &operation : getBody().front()) {
    if (operation.getName().getDialectNamespace() != "darwinn_isa" ||
        !operation.getName().isRegistered() ||
        !operation.hasTrait<OpTrait::HasParent<FragmentOp>::Impl>() ||
        operation.getNumRegions() != 0 || operation.getNumOperands() != 0 ||
        operation.getNumResults() != 0 || operation.getNumSuccessors() != 0)
      return emitOpError("may contain only registered Darwinn instructions");
  }

  return success();
}

static LogicalResult verifyUnsignedField(Operation *op, StringRef name,
                                         IntegerAttr value, unsigned bits) {
  if (value.getValue().isNegative() || value.getValue().getActiveBits() > bits)
    return op->emitOpError()
           << name << " must fit in " << bits << " unsigned bits";
  return success();
}

static LogicalResult verifyBitmap(Operation *op, StringRef name,
                                  DenseBoolArrayAttr value, int64_t count) {
  if (value.size() != count)
    return op->emitOpError() << name << " must contain " << count << " bits";
  return success();
}

static LogicalResult verifyScalarHeader(Operation *op, HeaderAttr header,
                                        bool scalarEncoder = true) {
  auto program = op->getParentOfType<ProgramOp>();
  if (!program)
    return op->emitOpError("requires an enclosing darwinn_isa.program");
  if (header.getEnableSequencerOverwrite() &&
      (!scalarEncoder || !program.getTargetConfig().getSequencerOverwrite()))
    return op->emitOpError("sequencer overwrite is unavailable for this "
                           "instruction configuration");
  return success();
}

LogicalResult NoOp::verify() { return verifyScalarHeader(*this, getHeader()); }

LogicalResult HaltOp::verify() {
  if (failed(verifyScalarHeader(*this, getHeader())))
    return failure();
  return verifyUnsignedField(*this, "interrupt_id", getInterruptIdAttr(), 2);
}

LogicalResult InterruptOp::verify() {
  if (failed(verifyScalarHeader(*this, getHeader())))
    return failure();
  return verifyUnsignedField(*this, "interrupt_id", getInterruptIdAttr(), 2);
}

LogicalResult LoadProgramOp::verify() {
  if (failed(verifyScalarHeader(*this, getHeader())))
    return failure();
  unsigned bits = getOperation()
                      ->getParentOfType<ProgramOp>()
                      .getTargetConfig()
                      .getScalarPcBits();
  if (failed(verifyUnsignedField(*this, "start_pc", getStartPcAttr(), bits)))
    return failure();
  return verifyUnsignedField(*this, "number_chunks", getNumberChunksAttr(),
                             bits + 1);
}

LogicalResult ExecuteOp::verify() {
  if (failed(verifyScalarHeader(*this, getHeader())))
    return failure();
  unsigned bits = getOperation()
                      ->getParentOfType<ProgramOp>()
                      .getTargetConfig()
                      .getScalarPcBits();
  return verifyUnsignedField(*this, "start_pc", getStartPcAttr(), bits);
}

LogicalResult TileFenceOp::verify() {
  if (!symbolizeMemoryPowerState(
          static_cast<uint32_t>(getNarrowMemPowerState())) ||
      !symbolizeMemoryPowerState(static_cast<uint32_t>(getWideMemPowerState())))
    return emitOpError("invalid memory power state");

  if (failed(
          verifyBitmap(*this, "reset_sync_flag", getResetSyncFlagAttr(), 22)) ||
      failed(verifyBitmap(*this, "reset_sync_flag_thread_ids",
                          getResetSyncFlagThreadIdsAttr(), 4)) ||
      failed(verifyBitmap(*this, "wait_idle", getWaitIdleAttr(), 13)) ||
      failed(verifyBitmap(*this, "wait_idle_thread_ids",
                          getWaitIdleThreadIdsAttr(), 4)) ||
      failed(verifyBitmap(*this, "wait_producer_count_valid",
                          getWaitProducerCountValidAttr(), 2)) ||
      failed(verifyBitmap(*this, "virtual_channel_subscription",
                          getVirtualChannelSubscriptionAttr(), 8)) ||
      failed(verifyUnsignedField(*this, "sync_reset_value",
                                 getSyncResetValueAttr(), 25)))
    return failure();
  return verifyUnsignedField(*this, "producer_count_value",
                             getProducerCountValueAttr(), 25);
}

LogicalResult ScalarFenceOp::verify() {
  auto expected = getExpectedScalarSyncFlag();
  if (failed(ExpectedSyncFlagsAttr::verify([&]() { return emitOpError(); },
                                           expected.getValid(),
                                           expected.getValues())))
    return failure();

  if (failed(verifyScalarHeader(*this, getHeader(), false)) ||
      failed(verifyUnsignedField(*this, "tag", getTagAttr(), 20)) ||
      failed(verifyBitmap(*this, "reset_tile_fence", getResetTileFenceAttr(),
                          20)) ||
      failed(
          verifyBitmap(*this, "reset_sync_flag", getResetSyncFlagAttr(), 17)) ||
      failed(verifyBitmap(*this, "tile_fence_valid", getTileFenceValidAttr(),
                          20)) ||
      failed(verifyBitmap(*this, "wait_idle", getWaitIdleAttr(), 5)) ||
      failed(verifyBitmap(*this, "virtual_channel_subscription",
                          getVirtualChannelSubscriptionAttr(), 8)) ||
      failed(
          verifyUnsignedField(*this, "reset_value", getResetValueAttr(), 25)))
    return failure();
  return verifyUnsignedField(*this, "expected_tile_fence_count",
                             getExpectedTileFenceCountAttr(), 25);
}

LogicalResult TileLoadStoreOp::verify() {
  if (!symbolizeTileMemoryOperation(
          static_cast<uint32_t>(getMemoryOperation())))
    return emitOpError("invalid tile memory operation");
  if (!symbolizeTileScalarReplacement(static_cast<uint32_t>(getReplaceField())))
    return emitOpError("invalid tile scalar replacement");

  if (getRowAddress() >= 196608)
    return emitOpError("row_address exceeds G5 narrow memory");
  if (getRegisterBurstLength() < 1 || getRegisterBurstLength() > 64)
    return emitOpError("register_burst_length must be between 1 and 64");
  if (failed(verifyUnsignedField(*this, "base_register", getBaseRegisterAttr(),
                                 6)) ||
      failed(verifyUnsignedField(*this, "immediate_value",
                                 getImmediateValueAttr(), 32)) ||
      failed(verifyUnsignedField(*this, "scalar_register",
                                 getScalarRegisterAttr(), 5)))
    return failure();
  return verifyBitmap(*this, "thread_bitmap", getThreadBitmapAttr(), 4);
}

LogicalResult CoefficientTablesOp::verify() {
  if (getSplineSegments().size() != 40)
    return emitOpError(
        "spline_segments must contain eight groups of five coefficients");
  if (getSegmentLowerBounds().size() != 7)
    return emitOpError("segment_lower_bounds must contain seven values");
  if (getPolynomialDegree() > 4)
    return emitOpError("polynomial_degree must be between 0 and 4");
  if (failed(verifyBitmap(*this, "register_sourced_operands",
                          getRegisterSourcedOperandsAttr(), 8)) ||
      failed(verifyBitmap(*this, "enhanced_square_bitmap",
                          getEnhancedSquareBitmapAttr(), 8)))
    return failure();
  return verifyBitmap(*this, "thread_multicast_bitmap",
                      getThreadMulticastBitmapAttr(), 4);
}
