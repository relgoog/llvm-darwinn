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

#include "mlir/Dialect/DiveVm/IR/DiveVmOpsDialect.cpp.inc"
#include "mlir/Dialect/DiveVm/IR/EdgeTpuOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/DiveVm/IR/DiveVmOps.cpp.inc"

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

#define GET_OP_CLASSES
#include "mlir/Dialect/DiveVm/IR/EdgeTpuOps.cpp.inc"

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

LogicalResult dive_vm::ComputeNormStatsForRkhyOp::verify() {
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

LogicalResult dive_vm::WaitForRkhyCompletionOp::verify() {
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
