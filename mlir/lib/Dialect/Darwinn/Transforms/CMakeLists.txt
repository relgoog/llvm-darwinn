set(LLVM_TARGET_DEFINITIONS DwcPasses.td)
mlir_tablegen(DwcPasses.h.inc -gen-pass-decls -name Dwc)
add_mlir_dialect_tablegen_target(MLIRDwcPassIncGen)

add_mlir_dialect_library(MLIRDwcTransforms
  DwcPasses.cpp
  LegalizeBfloat16.cpp
  LowerConvert.cpp
  LowerExecutionPlan.cpp
  LowerCopySlice.cpp
  LowerPutBits.cpp
  LowerRuntime.cpp
  LowerSemanticToDistributed.cpp
  MoveRedistributeToHost.cpp
  NodePartition.cpp
  OptimizePointwise.cpp
  PlanSharding.cpp
  PlanTensorTraversals.cpp
  PropagateDistributedSlices.cpp
  PropagatingSlicing.cpp
  RedistributeOptimization.cpp
  ScheduledStorage.cpp
  SlicingCost.cpp
  SlicingModel.cpp
  SpillFillOptimization.cpp
  TransferCanonicalization.cpp
  TensorComputeConfiguration.cpp

  DEPENDS
  MLIRDwcPassIncGen
  MLIRDarwinnTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRArithDialect
  MLIRControlFlowToLLVM
  MLIRDarwinnDialect
  MLIRDarwinnEncoding
  MLIRDiveVmDialect
  MLIRFuncDialect
  MLIRFuncToLLVM
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRMathDialect
  MLIRMathToLLVM
  MLIRMathToLibm
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRSCFToControlFlow
  MLIRTensorDialect
  MLIRTransformUtils
  )
