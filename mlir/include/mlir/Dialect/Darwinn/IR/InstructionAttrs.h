#ifndef MLIR_DIALECT_DARWINN_IR_INSTRUCTIONATTRS_H
#define MLIR_DIALECT_DARWINN_IR_INSTRUCTIONATTRS_H

#include "mlir/Dialect/Darwinn/IR/InstructionDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include <cstdint>

#include "mlir/Dialect/Darwinn/IR/InstructionEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/Darwinn/IR/InstructionAttrs.h.inc"

#endif
