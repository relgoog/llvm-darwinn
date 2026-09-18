//===- DiveVmDialect.td - DiveVM and EdgeTPU dialect base defs -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the dive_vm and edgetpu dialects and their common
// operation base classes. Concrete operations live in DiveVmOps.td and
// EdgeTpuOps.td alongside this file, mirroring how TosaShapeOps.td and
// TosaUtilOps.td extend the Tosa directory.
//
//===----------------------------------------------------------------------===//

#ifndef DIVEVM_DIALECT
#define DIVEVM_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// The DiveVM Dialect.
//===----------------------------------------------------------------------===//

def DiveVm_Dialect : Dialect {
  let name = "dive_vm";

  let description = [{
    The DiveVM dialect models the low-level VM instruction set observed in
    dive binaries, including arithmetic, memory, DMA, and control operations.
  }];

  let cppNamespace = "::mlir::dive_vm";
}

//===----------------------------------------------------------------------===//
// The EdgeTPU Dialect.
//===----------------------------------------------------------------------===//

def EdgeTpu_Dialect : Dialect {
  let name = "edgetpu";

  let description = [{
    The EdgeTPU dialect models the accelerator-side operations observed
    alongside dive_vm, such as sub-channel convolutions and matmuls.
  }];

  let cppNamespace = "::mlir::edgetpu";
}

//===----------------------------------------------------------------------===//
// Operation base classes.
//===----------------------------------------------------------------------===//

class DiveVm_Op<string mnemonic, list<Trait> traits = []>
    : Op<DiveVm_Dialect, mnemonic, traits> {}

class EdgeTpu_Op<string mnemonic, list<Trait> traits = []>
    : Op<EdgeTpu_Dialect, mnemonic, traits> {}

#endif // DIVEVM_DIALECT
