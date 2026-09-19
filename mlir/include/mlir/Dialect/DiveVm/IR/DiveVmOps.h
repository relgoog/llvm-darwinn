//===-- DiveVmOps.h - DiveVM and EdgeTPU dialect declarations ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the DiveVM and EdgeTPU dialects in MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_DIVEVM_IR_DIVEVMOPS_H
#define MLIR_DIALECT_DIVEVM_IR_DIVEVMOPS_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"

//===----------------------------------------------------------------------===//
// DiveVM and EdgeTPU dialect includes.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/DiveVm/IR/DiveVmOpsDialect.h.inc"
#include "mlir/Dialect/DiveVm/IR/EdgeTpuOpsDialect.h.inc"
#include "mlir/Dialect/DiveVm/IR/DiveVmTensorOpsDialect.h.inc"
#include "mlir/Dialect/DiveVm/IR/DwgTensorOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/DiveVm/IR/DiveVmOps.h.inc"
#define GET_OP_CLASSES
#include "mlir/Dialect/DiveVm/IR/EdgeTpuOps.h.inc"
#define GET_OP_CLASSES
#include "mlir/Dialect/DiveVm/IR/DiveVmTensorOps.h.inc"
#define GET_OP_CLASSES
#include "mlir/Dialect/DiveVm/IR/DwgTensorOps.h.inc"

#endif // MLIR_DIALECT_DIVEVM_IR_DIVEVMOPS_H
