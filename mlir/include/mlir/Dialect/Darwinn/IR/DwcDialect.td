//===-- DwcDialect.td - DWC dialect ----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the DWC dialect and its operation base class.
// Concrete operations live in DwcOps.td alongside this file.
//
//===----------------------------------------------------------------------===//

#ifndef DWC_DIALECT
#define DWC_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// The DWC Dialect.
//===----------------------------------------------------------------------===//

def Dwc_Dialect : Dialect {
  let name = "dwc";
  let cppNamespace = "::mlir::dwc";
  let description = [{
    The DWC dialect models the compute operations observed in dive binaries,
    including convolutions, elementwise ops, reductions, and reshapes.
  }];
}

//===----------------------------------------------------------------------===//
// DWC base classes.
//===----------------------------------------------------------------------===//

class Dwc_Op<string mnemonic, list<Trait> traits = []>
    : Op<Dwc_Dialect, mnemonic, traits> {}

#endif // DWC_DIALECT
