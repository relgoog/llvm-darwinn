#ifndef MLIR_DIALECT_DARWINN_TRANSFORMS_TENSORCOMPUTECONFIGURATION_H
#define MLIR_DIALECT_DARWINN_TRANSFORMS_TENSORCOMPUTECONFIGURATION_H

#include "mlir/Dialect/Darwinn/IR/DarwinnScheduledAttrs.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/Darwinn/Tensor.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <optional>

namespace mlir {
class Operation;

namespace darwinn {

struct TensorLinearConfiguration {
  LinearOperation operation;
  OperandType activationType;
  OperandType parameterType;
  bool singleFeature;
  MacDisable macDisable;
  bool broadcastSecondOperand;
};

struct TensorNonLinearConfiguration {
  ActivationFunction operation;
  uint32_t activationPipelineScaleBits;
  uint32_t immediateBiasBits;
  uint32_t highClipBits;
  uint32_t lowClipBits;
  OutputType outputType;
  bool useScales;
  bool applyBias;
  bool symmetricFunction;
  bool evenOddFunction;
  Preprocess preprocess;
  NluPredicate biasPredicate;
  NluPredicate scalePredicate;
  uint32_t bitmask;
};

struct TensorComputeConfiguration {
  TensorLinearConfiguration linear;
  TensorNonLinearConfiguration nonLinear;
  llvm::SmallVector<llvm::APFloat, 2> coefficients;
  std::optional<NluFunctionKind> functionTable;
};

FailureOr<TensorComputeConfiguration>
deriveTensorComputeConfiguration(Operation *operation,
                                 TensorTraversalPlanAttr traversal);

}
}

#endif
