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

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
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
  let useDefaultAttributePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// DWC base classes.
//===----------------------------------------------------------------------===//

class Dwc_Op<string mnemonic, list<Trait> traits = []>
    : Op<Dwc_Dialect, mnemonic, traits> {}

class Dwc_I32Enum<string name, string description, list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, description, cases> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::dwc";
}
class Dwc_I32EnumAttr<string mnemonic, Dwc_I32Enum enumInfo> :
    EnumAttr<Dwc_Dialect, enumInfo, mnemonic> {
  let assemblyFormat = "`<` $value `>`";
}

def Dwc_ComparisonTypeEqual : I32EnumAttrCase<"Equal", 0, "EQUAL">;
def Dwc_ComparisonTypeGreater : I32EnumAttrCase<"Greater", 1, "GREATER">;
def Dwc_ComparisonTypeGreaterEqual : I32EnumAttrCase<"GreaterEqual", 2, "GREATER_EQUAL">;
def Dwc_ComparisonTypeLess : I32EnumAttrCase<"Less", 3, "LESS">;
def Dwc_ComparisonTypeLessEqual : I32EnumAttrCase<"LessEqual", 4, "LESS_EQUAL">;
def Dwc_ComparisonTypeNotEqual : I32EnumAttrCase<"NotEqual", 5, "NOT_EQUAL">;
def Dwc_ComparisonTypeEnum : Dwc_I32Enum<
  "ComparisonType", "DWC comparison type", [
    Dwc_ComparisonTypeEqual,
    Dwc_ComparisonTypeGreater,
    Dwc_ComparisonTypeGreaterEqual,
    Dwc_ComparisonTypeLess,
    Dwc_ComparisonTypeLessEqual,
    Dwc_ComparisonTypeNotEqual
  ]>;
def Dwc_ComparisonTypeAttr : Dwc_I32EnumAttr<"comparison_type", Dwc_ComparisonTypeEnum>;

def Dwc_CumulativeOpTypeSum : I32EnumAttrCase<"Sum", 0, "SUM">;
def Dwc_CumulativeOpTypeEnum : Dwc_I32Enum<
  "CumulativeOpType", "DWC cumulative op type", [
    Dwc_CumulativeOpTypeSum
  ]>;
def Dwc_CumulativeOpTypeAttr : Dwc_I32EnumAttr<"cumulative_op_type", Dwc_CumulativeOpTypeEnum>;

def Dwc_ReductionTypeArgmax : I32EnumAttrCase<"Argmax", 0, "ARGMAX">;
def Dwc_ReductionTypeArgmin : I32EnumAttrCase<"Argmin", 1, "ARGMIN">;
def Dwc_ReductionTypeLogicalAnd : I32EnumAttrCase<"LogicalAnd", 2, "LOGICAL_AND">;
def Dwc_ReductionTypeLogicalOr : I32EnumAttrCase<"LogicalOr", 3, "LOGICAL_OR">;
def Dwc_ReductionTypeMax : I32EnumAttrCase<"Max", 4, "MAX">;
def Dwc_ReductionTypeMean : I32EnumAttrCase<"Mean", 5, "MEAN">;
def Dwc_ReductionTypeMin : I32EnumAttrCase<"Min", 6, "MIN">;
def Dwc_ReductionTypeSum : I32EnumAttrCase<"Sum", 7, "SUM">;
def Dwc_ReductionTypeEnum : Dwc_I32Enum<
  "ReductionType", "DWC reduction type", [
    Dwc_ReductionTypeArgmax,
    Dwc_ReductionTypeArgmin,
    Dwc_ReductionTypeLogicalAnd,
    Dwc_ReductionTypeLogicalOr,
    Dwc_ReductionTypeMax,
    Dwc_ReductionTypeMean,
    Dwc_ReductionTypeMin,
    Dwc_ReductionTypeSum
  ]>;
def Dwc_ReductionTypeAttr : Dwc_I32EnumAttr<"reduction_type", Dwc_ReductionTypeEnum>;

def Dwc_ScalarOpTypeIdentity : I32EnumAttrCase<"Identity", 0, "IDENTITY">;
def Dwc_ScalarOpTypeSubtract : I32EnumAttrCase<"Subtract", 1, "SUBTRACT">;
def Dwc_ScalarOpTypeEnum : Dwc_I32Enum<
  "ScalarOpType", "DWC scalar op type", [
    Dwc_ScalarOpTypeIdentity,
    Dwc_ScalarOpTypeSubtract
  ]>;
def Dwc_ScalarOpTypeAttr : Dwc_I32EnumAttr<"scalar_op_type", Dwc_ScalarOpTypeEnum>;

#endif // DWC_DIALECT
