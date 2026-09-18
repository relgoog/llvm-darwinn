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
