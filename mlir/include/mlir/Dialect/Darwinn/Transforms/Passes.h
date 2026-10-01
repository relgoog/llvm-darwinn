//===- Passes.h - Darwinn optimization pass declarations ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_DARWINN_TRANSFORMS_PASSES_H
#define MLIR_DIALECT_DARWINN_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace darwinn {

#define GEN_PASS_DECL
#include "mlir/Dialect/Darwinn/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/Darwinn/Transforms/Passes.h.inc"

void populateLowerCopySlicePatterns(RewritePatternSet &patterns);
void populateLowerConvertPatterns(RewritePatternSet &patterns);

} // namespace darwinn
} // namespace mlir

#endif // MLIR_DIALECT_DARWINN_TRANSFORMS_PASSES_H
