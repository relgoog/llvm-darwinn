//===-- DarwinnOps.h - Darwinn dialect declarations --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the Darwinn dialect in MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_DARWINN_IR_DARWINNOPS_H
#define MLIR_DIALECT_DARWINN_IR_DARWINNOPS_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"

//===----------------------------------------------------------------------===//
// Darwinn dialect includes.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Darwinn/IR/DarwinnOpsDialect.h.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/Darwinn/IR/DarwinnAttributes.h.inc"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/Darwinn/IR/DarwinnTypes.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/Darwinn/IR/DarwinnOps.h.inc"

#endif // MLIR_DIALECT_DARWINN_IR_DARWINNOPS_H
