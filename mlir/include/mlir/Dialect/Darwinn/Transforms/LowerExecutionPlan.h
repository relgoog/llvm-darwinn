#ifndef MLIR_DIALECT_DARWINN_TRANSFORMS_LOWEREXECUTIONPLAN_H
#define MLIR_DIALECT_DARWINN_TRANSFORMS_LOWEREXECUTIONPLAN_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Target/Darwinn/ExecutionPlan.h"

namespace mlir::darwinn {

FailureOr<func::FuncOp> lowerExecutionPlan(ModuleOp module, Location location,
                                           const ExecutionPlan &plan);

}

#endif
