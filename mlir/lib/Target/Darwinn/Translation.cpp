#include "mlir/Target/Darwinn/Translation.h"

#include "mlir/Dialect/Darwinn/IR/InstructionOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Target/Darwinn/DmaAttributes.h"
#include "mlir/Target/Darwinn/EncodingAttributes.h"
#include "mlir/Target/Darwinn/TensorAttributes.h"
#include "mlir/Target/Darwinn/TransportAttributes.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::darwinn;

namespace {

template <typename Packet>
llvm::Expected<Instruction> asInstruction(llvm::Expected<Packet> packet) {
  if (!packet)
    return packet.takeError();
  return Instruction(std::move(*packet));
}

llvm::Expected<Instruction> convertInstruction(Operation *operation) {
  return llvm::TypeSwitch<Operation *, llvm::Expected<Instruction>>(operation)
      .Case<isa::HibDmaOp, isa::PopInputOp, isa::RingInfeedOp,
            isa::RingOutfeedOp>([](auto transport) {
        return asInstruction(convertTransport(transport));
      })
      .Case([](isa::TensorOp tensor) {
        return asInstruction(convertTensor(tensor));
      })
      .Case<isa::RingConsumerOp, isa::RingProducerOp, isa::MeshOp,
            isa::WideToNarrowOp, isa::NarrowToWideOp>(
          [](auto dma) { return asInstruction(convertDmaInstruction(dma)); })
      .Default(convertCoreInstruction);
}

LogicalResult translateModule(ModuleOp module, llvm::raw_ostream &output) {
  isa::ProgramOp program;
  for (Operation &operation : module.getBody()->getOperations()) {
    if (program || !llvm::isa<isa::ProgramOp>(operation))
      return module.emitError(
          "binary translation requires exactly one Darwinn program");
    program = cast<isa::ProgramOp>(operation);
  }
  if (!program)
    return module.emitError("binary translation requires a Darwinn program");

  auto chunks = translateProgram(program);
  if (failed(chunks))
    return failure();
  for (const InstructionBytes &chunk : *chunks)
    output.write(reinterpret_cast<const char *>(chunk.data()), chunk.size());
  return success();
}

}

FailureOr<llvm::SmallVector<InstructionBytes, 0>>
mlir::darwinn::translateProgram(isa::ProgramOp program) {
  if (failed(verify(program)))
    return failure();

  auto config = program.getTargetConfig();
  auto scalarEncoder = ScalarEncoder::create(config.getScalarPcBits(),
                                             config.getSequencerOverwrite());
  if (!scalarEncoder)
    return program.emitError(llvm::toString(scalarEncoder.takeError()));

  llvm::SmallVector<InstructionBytes, 0> chunks;
  for (auto chunk : program.getBody().front().getOps<isa::ChunkOp>()) {
    InstructionBytes chunkBytes;
    for (auto fragment : chunk.getBody().front().getOps<isa::FragmentOp>()) {
      FragmentEncoder encoder(*scalarEncoder);
      for (Operation &operation : fragment.getBody().front()) {
        auto instruction = convertInstruction(&operation);
        if (!instruction)
          return operation.emitError(llvm::toString(instruction.takeError()));
        auto bytes = encoder.encode(*instruction);
        if (!bytes)
          return operation.emitError(llvm::toString(bytes.takeError()));
        chunkBytes.append(bytes->begin(), bytes->end());
      }
    }
    chunks.push_back(std::move(chunkBytes));
  }

  return chunks;
}

void mlir::registerToDarwinnTranslation() {
  TranslateFromMLIRRegistration registration(
      "darwinn-to-binary",
      "Serialize Darwinn instruction chunks in program order", translateModule,
      [](DialectRegistry &registry) {
        registry.insert<isa::DarwinnIsaDialect>();
      });
}
