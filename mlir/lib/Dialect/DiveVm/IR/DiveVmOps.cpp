//===- DiveVmOps.cpp - MLIR Dialect for DiveVM and EdgeTPU ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/DiveVm/IR/DiveVmOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"

using namespace mlir;
using namespace mlir::dive_vm;
using namespace mlir::edgetpu;
using namespace mlir::dive_vm_tensor;
using namespace mlir::dwg_tensor;

#include "mlir/Dialect/DiveVm/IR/DiveVmOpsDialect.cpp.inc"
#include "mlir/Dialect/DiveVm/IR/EdgeTpuOpsDialect.cpp.inc"
#include "mlir/Dialect/DiveVm/IR/DiveVmTensorOpsDialect.cpp.inc"
#include "mlir/Dialect/DiveVm/IR/DwgTensorOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/DiveVm/IR/DiveVmOps.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/DiveVm/IR/DiveVmTensorOps.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/DiveVm/IR/DwgTensorOps.cpp.inc"

//===----------------------------------------------------------------------===//
// DiveVm and EdgeTpu dialect initialization.
//===----------------------------------------------------------------------===//

void DiveVmDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/DiveVm/IR/DiveVmOps.cpp.inc"
      >();
}

void EdgeTpuDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/DiveVm/IR/EdgeTpuOps.cpp.inc"
      >();
}

void DiveVmTensorDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/DiveVm/IR/DiveVmTensorOps.cpp.inc"
      >();
}

void DwgTensorDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/DiveVm/IR/DwgTensorOps.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "mlir/Dialect/DiveVm/IR/EdgeTpuOps.cpp.inc"

LogicalResult dive_vm_tensor::AllocateOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::ConstOp::verify() {
  if (!(*this)->hasAttr("value"))
    return (*this)->emitOpError(
        "dive_vm_tensor.const op missing value attribute");
  return success();
}

LogicalResult dive_vm_tensor::ExtractSliceOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::InsertSliceOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::TypeCastOp::verify() {
  return success();
}

LogicalResult dwg_tensor::CodegenOp::verify() {
  return success();
}

LogicalResult dwg_tensor::ConstOp::verify() {
  return success();
}

LogicalResult dwg_tensor::DynamicShapeScopeOp::verify() {
  return success();
}

LogicalResult dwg_tensor::EvalWithShapeOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::AbsOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::AddOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::AddImmOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::AddressOfOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::AddressOfActivationOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::AddressOfInputActivationOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::AddressOfOutputActivationOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::AddressOfParameterOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::AddressOfParameterRegionOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::AddressOfScratchOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::ArithmeticLeftShiftOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::ArithmeticLeftShiftImmOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::ArithmeticRightShiftOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::ArithmeticRightShiftImmOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::BitcastOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::BitwiseAndOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::BitwiseAndImmOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::BitwiseOrOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::BitwiseOrImmOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::BitwiseXorOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::BitwiseXorImmOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::BrOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::CacheCleanInvalidateOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::CastOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::ConcatenateOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::CondBrOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::CopyOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::CopyImmOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::CountLeadingZerosOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::CumsumOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::CustomOnBuffersOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::DeclareTensorOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::DivOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::DivImmOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::DmaHintOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::DynamicSliceYOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::EqualOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::EqualImmOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::EvalWithShapeOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::ExtractOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::FillOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::FloorDivOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::GatherOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::GatherNdOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::GreaterOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::GreaterEqualOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::GreaterEqualImmOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::GreaterImmOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::IfOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::InsertOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::LessOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::LessEqualOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::LessEqualImmOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::LessImmOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::LoadOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::LoadIndirectOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::LogicalAndOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::LogicalAndImmOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::LogicalRightShiftOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::LogicalRightShiftImmOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::MarkValueWithShapeOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::MaxOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::MaxImmOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::MinOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::MinImmOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::MulOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::MulImmOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::NotEqualOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::NotEqualImmOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::OneHotOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::PadOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::PerformSoftwarePreemptionIfRequestedOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::PopCountOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::PowOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::PowImmOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::PrintOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::PutBitsOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::ReductionOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::RemOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::RemImmOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::RollOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::RunSerializedModelOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::ScatterNdOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::SelectOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::ShapeOfActivationOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::SignOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::StoreOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::StoreIndirectOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::SubOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::SubImmOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::SwitchOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::TopKOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::TpuOffloadOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::WhileOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::WriteDmaDescriptorOp::verify() {
  return success();
}

LogicalResult dive_vm_tensor::YieldOp::verify() {
  return success();
}

LogicalResult dwg_tensor::AbsOp::verify() {
  return success();
}

LogicalResult dwg_tensor::AddOp::verify() {
  return success();
}

LogicalResult dwg_tensor::ArithmeticLeftShiftOp::verify() {
  return success();
}

LogicalResult dwg_tensor::ArithmeticRightShiftOp::verify() {
  return success();
}

LogicalResult dwg_tensor::BarrierOp::verify() {
  return success();
}

LogicalResult dwg_tensor::BatchMatrixNmsOp::verify() {
  return success();
}

LogicalResult dwg_tensor::BitwiseAndOp::verify() {
  return success();
}

LogicalResult dwg_tensor::BitwiseOrOp::verify() {
  return success();
}

LogicalResult dwg_tensor::BitwiseXorOp::verify() {
  return success();
}

LogicalResult dwg_tensor::CastOp::verify() {
  return success();
}

LogicalResult dwg_tensor::CastFromIndexOp::verify() {
  return success();
}

LogicalResult dwg_tensor::CastToIndexOp::verify() {
  return success();
}

LogicalResult dwg_tensor::ConcatenationOp::verify() {
  return success();
}

LogicalResult dwg_tensor::ConstBytesOp::verify() {
  return success();
}

LogicalResult dwg_tensor::ConstNoneOp::verify() {
  return success();
}

LogicalResult dwg_tensor::CopyOp::verify() {
  return success();
}

LogicalResult dwg_tensor::CopyOuterSliceOp::verify() {
  return success();
}

LogicalResult dwg_tensor::CountLeadingZerosOp::verify() {
  return success();
}

LogicalResult dwg_tensor::CumsumOp::verify() {
  return success();
}

LogicalResult dwg_tensor::DeclareTensorOp::verify() {
  return success();
}

LogicalResult dwg_tensor::DeclareTensorStaticOp::verify() {
  return success();
}

LogicalResult dwg_tensor::DimensionLengthOp::verify() {
  return success();
}

LogicalResult dwg_tensor::DivOp::verify() {
  return success();
}

LogicalResult dwg_tensor::EqOp::verify() {
  return success();
}

LogicalResult dwg_tensor::ForOp::verify() {
  return success();
}

LogicalResult dwg_tensor::GatherOp::verify() {
  return success();
}

LogicalResult dwg_tensor::GatherNdOp::verify() {
  return success();
}

LogicalResult dwg_tensor::GeOp::verify() {
  return success();
}

LogicalResult dwg_tensor::GtOp::verify() {
  return success();
}

LogicalResult dwg_tensor::GumbelRngOp::verify() {
  return success();
}

LogicalResult dwg_tensor::IfOp::verify() {
  return success();
}

LogicalResult dwg_tensor::LandmarksToTransformMatrixOp::verify() {
  return success();
}

LogicalResult dwg_tensor::LeOp::verify() {
  return success();
}

LogicalResult dwg_tensor::LogicalAndOp::verify() {
  return success();
}

LogicalResult dwg_tensor::LtOp::verify() {
  return success();
}

LogicalResult dwg_tensor::MaximumOp::verify() {
  return success();
}

LogicalResult dwg_tensor::MemoryCopyOp::verify() {
  return success();
}

LogicalResult dwg_tensor::MinimumOp::verify() {
  return success();
}

LogicalResult dwg_tensor::ModOp::verify() {
  return success();
}

LogicalResult dwg_tensor::MulOp::verify() {
  return success();
}

LogicalResult dwg_tensor::NeOp::verify() {
  return success();
}

LogicalResult dwg_tensor::NewQosClassOp::verify() {
  return success();
}

LogicalResult dwg_tensor::OneHotOp::verify() {
  return success();
}

LogicalResult dwg_tensor::OneHotV2Op::verify() {
  return success();
}

LogicalResult dwg_tensor::OuterSliceOp::verify() {
  return success();
}

LogicalResult dwg_tensor::PadOp::verify() {
  return success();
}

LogicalResult dwg_tensor::PopCountOp::verify() {
  return success();
}

LogicalResult dwg_tensor::PowOp::verify() {
  return success();
}

LogicalResult dwg_tensor::RaiseErrorOp::verify() {
  return success();
}

LogicalResult dwg_tensor::ReductionOp::verify() {
  return success();
}

LogicalResult dwg_tensor::ReductionV2Op::verify() {
  return success();
}

LogicalResult dwg_tensor::RelayoutOp::verify() {
  return success();
}

LogicalResult dwg_tensor::RoiToTransformMatrixOp::verify() {
  return success();
}

LogicalResult dwg_tensor::RollOp::verify() {
  return success();
}

LogicalResult dwg_tensor::ScatterNdOp::verify() {
  return success();
}

LogicalResult dwg_tensor::SelectOp::verify() {
  return success();
}

LogicalResult dwg_tensor::SelectV2Op::verify() {
  return success();
}

LogicalResult dwg_tensor::SetDimensionLengthOp::verify() {
  return success();
}

LogicalResult dwg_tensor::SignOp::verify() {
  return success();
}

LogicalResult dwg_tensor::SingleChipOffloadOp::verify() {
  return success();
}

LogicalResult dwg_tensor::SliceV2Op::verify() {
  return success();
}

LogicalResult dwg_tensor::SortOp::verify() {
  return success();
}

LogicalResult dwg_tensor::StaticErrorOp::verify() {
  return success();
}

LogicalResult dwg_tensor::SubOp::verify() {
  return success();
}

LogicalResult dwg_tensor::SwitchOp::verify() {
  return success();
}

LogicalResult dwg_tensor::TestOnlyFingerprintCoreOp::verify() {
  return success();
}

LogicalResult dwg_tensor::TopKOp::verify() {
  return success();
}

LogicalResult dwg_tensor::TpuOffloadOp::verify() {
  return success();
}

LogicalResult dwg_tensor::TransformLandmarksOp::verify() {
  return success();
}

LogicalResult dwg_tensor::TransformTensorBilinearOp::verify() {
  return success();
}

LogicalResult dwg_tensor::TypeCastOp::verify() {
  return success();
}

LogicalResult dwg_tensor::TypeCastDynamicOp::verify() {
  return success();
}

LogicalResult dwg_tensor::UniformRandomNumberGenerationOp::verify() {
  return success();
}

LogicalResult dwg_tensor::UnsortedSegmentReduceOp::verify() {
  return success();
}

LogicalResult dwg_tensor::UpdateOuterSliceOp::verify() {
  return success();
}

LogicalResult dwg_tensor::UseMaxConcurrencyModeOp::verify() {
  return success();
}

LogicalResult dwg_tensor::WhileOp::verify() {
  return success();
}

LogicalResult dwg_tensor::YieldOp::verify() {
  return success();
}

static LogicalResult verifyDwcArityN(Operation *op, size_t numOperands,
                                    size_t expected) {
  if (numOperands != expected)
    return op->emitOpError("expects ")
           << expected << " operands, got " << numOperands;
  return success();
}

static LogicalResult verifyDwcArityAtLeast(Operation *op,
                                          size_t numOperands,
                                          size_t min) {
  if (numOperands < min)
    return op->emitOpError("expects at least ")
           << min << " operands, got " << numOperands;
  return success();
}

LogicalResult dive_vm::AddOp::verify() {
  return verifyDwcArityN(*this, getOperands().size(), 2);
}

LogicalResult dive_vm::AddImmOp::verify() {
  return verifyDwcArityN(*this, getOperands().size(), 1);
}

LogicalResult dive_vm::AddressOfActivationOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult dive_vm::AddressOfInputActivationOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult dive_vm::AddressOfOutputActivationOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult dive_vm::AddressOfParameterOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult dive_vm::AddressOfParameterRegionOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult dive_vm::AddressOfScratchOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult dive_vm::AddressSpaceOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult dive_vm::AllocateOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult dive_vm::ArithmeticLeftShiftOp::verify() {
  return verifyDwcArityN(*this, getOperands().size(), 2);
}

LogicalResult dive_vm::ArithmeticLeftShiftImmOp::verify() {
  return verifyDwcArityN(*this, getOperands().size(), 1);
}

LogicalResult dive_vm::ArithmeticRightShiftOp::verify() {
  return verifyDwcArityN(*this, getOperands().size(), 2);
}

LogicalResult dive_vm::ArithmeticRightShiftImmOp::verify() {
  return verifyDwcArityN(*this, getOperands().size(), 1);
}

LogicalResult dive_vm::BenchmarkOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult dive_vm::BitcastOp::verify() {
  return verifyDwcArityN(*this, getOperands().size(), 1);
}

LogicalResult dive_vm::BitwiseAndOp::verify() {
  return verifyDwcArityN(*this, getOperands().size(), 2);
}

LogicalResult dive_vm::BitwiseAndImmOp::verify() {
  return verifyDwcArityN(*this, getOperands().size(), 1);
}

LogicalResult dive_vm::BitwiseOrOp::verify() {
  return verifyDwcArityN(*this, getOperands().size(), 2);
}

LogicalResult dive_vm::BitwiseOrImmOp::verify() {
  return verifyDwcArityN(*this, getOperands().size(), 1);
}

LogicalResult dive_vm::BitwiseXorOp::verify() {
  return verifyDwcArityN(*this, getOperands().size(), 2);
}

LogicalResult dive_vm::BitwiseXorImmOp::verify() {
  return verifyDwcArityN(*this, getOperands().size(), 1);
}

LogicalResult dive_vm::BrOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult dive_vm::BufferDirOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult dive_vm::CacheCleanInvalidateOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult dive_vm::CastOp::verify() {
  return verifyDwcArityN(*this, getOperands().size(), 1);
}

LogicalResult dive_vm::ChunkOffsetsOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult dive_vm::ChunkSizesOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult dive_vm::ComputeNormStatsForVicaOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult dive_vm::CondBrOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult dive_vm::ConstOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult dive_vm::ConstBytesOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult dive_vm::CopyOp::verify() {
  return verifyDwcArityN(*this, getOperands().size(), 1);
}

LogicalResult dive_vm::CopyImmOp::verify() {
  return verifyDwcArityN(*this, getOperands().size(), 1);
}

LogicalResult dive_vm::CoreIdOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult dive_vm::CudaEmuCustomOpOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult dive_vm::CumsumOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult dive_vm::CustomOnAddressesOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult dive_vm::CustomOnBuffersOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult dive_vm::DisableItcTracingOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult dive_vm::DispatchHardwareInstructionOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult dive_vm::DivOp::verify() {
  return verifyDwcArityN(*this, getOperands().size(), 2);
}

LogicalResult dive_vm::DivImmOp::verify() {
  return verifyDwcArityN(*this, getOperands().size(), 1);
}

LogicalResult dive_vm::DiveVmModuleOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult dive_vm::DmaHintOp::verify() {
  return verifyDwcArityAtLeast(*this, getOperands().size(), 2);
}

LogicalResult dive_vm::DmaQueueOp::verify() {
  return verifyDwcArityAtLeast(*this, getOperands().size(), 2);
}

LogicalResult dive_vm::DtcModeOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult dive_vm::DvfsOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult dive_vm::DynamicSliceYOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult dive_vm::EnableItcTracingOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult dive_vm::EpilogueOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult dive_vm::EqualOp::verify() {
  return verifyDwcArityN(*this, getOperands().size(), 2);
}

LogicalResult dive_vm::EqualImmOp::verify() {
  return verifyDwcArityN(*this, getOperands().size(), 1);
}

LogicalResult dive_vm::ExecutableOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult dive_vm::ExtractSliceOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult dive_vm::FillOp::verify() {
  return verifyDwcArityN(*this, getOperands().size(), 1);
}

LogicalResult dive_vm::GatherOp::verify() {
  return verifyDwcArityAtLeast(*this, getOperands().size(), 2);
}

LogicalResult dive_vm::GatherNdOp::verify() {
  return verifyDwcArityAtLeast(*this, getOperands().size(), 2);
}

LogicalResult dive_vm::GetConstOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult dive_vm::GreaterOp::verify() {
  return verifyDwcArityN(*this, getOperands().size(), 2);
}

LogicalResult dive_vm::GreaterEqualOp::verify() {
  return verifyDwcArityN(*this, getOperands().size(), 2);
}

LogicalResult dive_vm::GreaterEqualImmOp::verify() {
  return verifyDwcArityN(*this, getOperands().size(), 1);
}

LogicalResult dive_vm::GreaterImmOp::verify() {
  return verifyDwcArityN(*this, getOperands().size(), 1);
}

LogicalResult dive_vm::HasIdpOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult dive_vm::HeapSizeBytesOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult dive_vm::HibGatherEditOp::verify() {
  return verifyDwcArityAtLeast(*this, getOperands().size(), 2);
}

LogicalResult dive_vm::IdpOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult dive_vm::IdpInstanceOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult dive_vm::InsertSliceOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult dive_vm::InstructionIdOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult dive_vm::InstructionIdLiveTensorsOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult dive_vm::LegacyScalarOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult dive_vm::LessOp::verify() {
  return verifyDwcArityN(*this, getOperands().size(), 2);
}

LogicalResult dive_vm::LessEqualOp::verify() {
  return verifyDwcArityN(*this, getOperands().size(), 2);
}

LogicalResult dive_vm::LessEqualImmOp::verify() {
  return verifyDwcArityN(*this, getOperands().size(), 1);
}

LogicalResult dive_vm::LessImmOp::verify() {
  return verifyDwcArityN(*this, getOperands().size(), 1);
}

LogicalResult dive_vm::LiveTensorOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult dive_vm::LoadOp::verify() {
  return verifyDwcArityN(*this, getOperands().size(), 1);
}

LogicalResult dive_vm::LoadIndirectOp::verify() {
  return verifyDwcArityN(*this, getOperands().size(), 1);
}

LogicalResult dive_vm::LogicalAndOp::verify() {
  return verifyDwcArityN(*this, getOperands().size(), 2);
}

LogicalResult dive_vm::LogicalAndImmOp::verify() {
  return verifyDwcArityN(*this, getOperands().size(), 1);
}

LogicalResult dive_vm::LogicalRightShiftOp::verify() {
  return verifyDwcArityN(*this, getOperands().size(), 2);
}

LogicalResult dive_vm::LogicalRightShiftImmOp::verify() {
  return verifyDwcArityN(*this, getOperands().size(), 1);
}

LogicalResult dive_vm::MarkDiveForHostResidentOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult dive_vm::MarkValueWithShapeOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult dive_vm::MaskIndicesOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult dive_vm::MaxOp::verify() {
  return verifyDwcArityN(*this, getOperands().size(), 2);
}

LogicalResult dive_vm::MaxImmOp::verify() {
  return verifyDwcArityN(*this, getOperands().size(), 1);
}

LogicalResult dive_vm::MinOp::verify() {
  return verifyDwcArityN(*this, getOperands().size(), 2);
}

LogicalResult dive_vm::MinImmOp::verify() {
  return verifyDwcArityN(*this, getOperands().size(), 1);
}

LogicalResult dive_vm::ModuleGroupOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult dive_vm::MulOp::verify() {
  return verifyDwcArityN(*this, getOperands().size(), 2);
}

LogicalResult dive_vm::MulImmOp::verify() {
  return verifyDwcArityN(*this, getOperands().size(), 1);
}

LogicalResult dive_vm::MultinomialOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult dive_vm::NotEqualOp::verify() {
  return verifyDwcArityN(*this, getOperands().size(), 2);
}

LogicalResult dive_vm::NotEqualImmOp::verify() {
  return verifyDwcArityN(*this, getOperands().size(), 1);
}

LogicalResult dive_vm::NumInputTensorsOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult dive_vm::NumOutputTensorsOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult dive_vm::OneHotOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult dive_vm::PadOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult dive_vm::ParameterArrayOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult dive_vm::PatchInstructionForStridedIoOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult dive_vm::PerformSoftwarePreemptionIfRequestedOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult dive_vm::PowOp::verify() {
  return verifyDwcArityN(*this, getOperands().size(), 2);
}

LogicalResult dive_vm::PowImmOp::verify() {
  return verifyDwcArityN(*this, getOperands().size(), 1);
}

LogicalResult dive_vm::PrintOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult dive_vm::ProgramTensorMappingTableOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult dive_vm::PrologueOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult dive_vm::PutBitsOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult dive_vm::ReadGnStatsOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult dive_vm::ReductionOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult dive_vm::ReductionTypeOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult dive_vm::RemOp::verify() {
  return verifyDwcArityN(*this, getOperands().size(), 2);
}

LogicalResult dive_vm::RemImmOp::verify() {
  return verifyDwcArityN(*this, getOperands().size(), 1);
}

LogicalResult dive_vm::RollOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult dive_vm::ScalarTypeOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult dive_vm::ScatterNdOp::verify() {
  return verifyDwcArityAtLeast(*this, getOperands().size(), 2);
}

LogicalResult dive_vm::SelectOp::verify() {
  return verifyDwcArityN(*this, getOperands().size(), 3);
}

LogicalResult dive_vm::SerializedModelFormatTypeOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult dive_vm::SetDtcModeOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult dive_vm::ShapeOfActivationOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult dive_vm::SharedTensorOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult dive_vm::SignOp::verify() {
  return verifyDwcArityN(*this, getOperands().size(), 1);
}

LogicalResult dive_vm::StoreOp::verify() {
  return verifyDwcArityN(*this, getOperands().size(), 1);
}

LogicalResult dive_vm::StoreIndirectOp::verify() {
  return verifyDwcArityN(*this, getOperands().size(), 1);
}

LogicalResult dive_vm::SubOp::verify() {
  return verifyDwcArityN(*this, getOperands().size(), 2);
}

LogicalResult dive_vm::SubImmOp::verify() {
  return verifyDwcArityN(*this, getOperands().size(), 1);
}

LogicalResult dive_vm::TopKOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult dive_vm::TpuOffloadOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult dive_vm::TransitionDtcPowerIslandOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult dive_vm::TranslateSramAddressOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult dive_vm::ViewOnAddressOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult dive_vm::WaitForFenceCompletionOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult dive_vm::WaitForPowerIslandTransitionCompleteOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult dive_vm::WaitForVicaCompletionOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult dive_vm::WriteDmaDescriptorOp::verify() {
  return verifyDwcArityAtLeast(*this, getOperands().size(), 2);
}

LogicalResult dive_vm::WriteHibDataOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult dive_vm::WriteScalarArchRegisterOp::verify() {
  // No shape contract: fully generic operands carry no rank to check.
  return success();
}

LogicalResult edgetpu::AnnotateMaterializePolicyOp::verify() {
  // No shape contract: packet layout absent, policy annotation carries
  // no operand shape to check.
  return success();
}

LogicalResult edgetpu::AttentionV1Op::verify() {
  // No shape contract: packet layout absent, fully generic operands
  // carry no rank to check.
  return success();
}

LogicalResult edgetpu::ConvertYuvToRgbOp::verify() {
  // No shape contract: packet layout absent, fully generic operands
  // carry no rank to check.
  return success();
}

LogicalResult edgetpu::ConvolutionSubChannelOp::verify() {
  // No shape contract: packet layout absent, fully generic operands
  // carry no rank to check.
  return success();
}

LogicalResult edgetpu::ConvolutionSubChannelDrqOp::verify() {
  // No shape contract: packet layout absent, fully generic operands
  // carry no rank to check.
  return success();
}

LogicalResult edgetpu::DepthwiseConvolutionFp8Op::verify() {
  // No shape contract: packet layout absent, fully generic operands
  // carry no rank to check.
  return success();
}

LogicalResult edgetpu::FastWalshHadamardTransformOp::verify() {
  // No shape contract: packet layout absent, fully generic operands
  // carry no rank to check.
  return success();
}

LogicalResult edgetpu::FullyConnectedFp8Op::verify() {
  // No shape contract: packet layout absent, fully generic operands
  // carry no rank to check.
  return success();
}

LogicalResult edgetpu::FullyConnectedSubChannelOp::verify() {
  // No shape contract: packet layout absent, fully generic operands
  // carry no rank to check.
  return success();
}

LogicalResult edgetpu::FullyConnectedSubChannelDrqOp::verify() {
  // No shape contract: packet layout absent, fully generic operands
  // carry no rank to check.
  return success();
}

LogicalResult edgetpu::MatrixMultiplySubChannelOp::verify() {
  // No shape contract: packet layout absent, fully generic operands
  // carry no rank to check.
  return success();
}

LogicalResult edgetpu::MatrixMultiplySubChannelDrqOp::verify() {
  // No shape contract: packet layout absent, fully generic operands
  // carry no rank to check.
  return success();
}

LogicalResult edgetpu::TransposeConvolutionFp8Op::verify() {
  // No shape contract: packet layout absent, fully generic operands
  // carry no rank to check.
  return success();
}

LogicalResult edgetpu::TransposedConvolutionSubChannelOp::verify() {
  // No shape contract: packet layout absent, fully generic operands
  // carry no rank to check.
  return success();
}

LogicalResult edgetpu::TransposedConvolutionSubChannelDrqOp::verify() {
  // No shape contract: packet layout absent, fully generic operands
  // carry no rank to check.
  return success();
}
