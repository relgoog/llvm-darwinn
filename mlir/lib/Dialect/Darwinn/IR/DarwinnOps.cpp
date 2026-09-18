//===- DarwinnOps.cpp - MLIR Dialect for Darwinn ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Darwinn/IR/DarwinnOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::darwinn;

#include "mlir/Dialect/Darwinn/IR/DarwinnOpsDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/Darwinn/IR/DarwinnAttributes.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/Darwinn/IR/DarwinnTypes.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/Darwinn/IR/DarwinnOps.cpp.inc"

//===----------------------------------------------------------------------===//
// Darwinn dialect initialization.
//===----------------------------------------------------------------------===//

void DarwinnDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Darwinn/IR/DarwinnOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/Darwinn/IR/DarwinnAttributes.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/Darwinn/IR/DarwinnTypes.cpp.inc"
      >();
}
