#ifndef MLIR_DIALECT_DARWINN_IR_INSTRUCTIONOPS_H
#define MLIR_DIALECT_DARWINN_IR_INSTRUCTIONOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Darwinn/IR/InstructionAttrs.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"

#define GET_OP_CLASSES
#include "mlir/Dialect/Darwinn/IR/InstructionOps.h.inc"

#endif
