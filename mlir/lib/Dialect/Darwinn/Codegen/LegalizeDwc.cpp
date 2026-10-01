#include "Codegen.h"
#include "mlir/Dialect/Darwinn/Codegen/Codegen.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::darwinn;
using namespace mlir::darwinn::codegen;

namespace {

std::optional<float> number(const llvm::json::Value &value) {
  if (std::optional<double> parsed = value.getAsNumber())
    return static_cast<float>(*parsed);
  std::optional<StringRef> text = value.getAsString();
  if (text == "Infinity")
    return std::numeric_limits<float>::infinity();
  if (text == "-Infinity")
    return -std::numeric_limits<float>::infinity();
  return std::nullopt;
}

std::optional<StringRef> splineName(NluFunctionKind function) {
  switch (function) {
  case NluFunctionKind::Exp:
    return StringRef("exp");
  case NluFunctionKind::Reciprocal:
    return StringRef("recip");
  default:
    return std::nullopt;
  }
}

FailureOr<CoefficientTables>
coefficientTables(const llvm::json::Object &spline) {
  const llvm::json::Array *segments = spline.getArray("segments");
  CoefficientTables tables;
  if (!segments || segments->empty() ||
      segments->size() > tables.splineSegments.size())
    return failure();
  SmallVector<float> lowers;
  SmallVector<SmallVector<float>> coefficients;
  for (const llvm::json::Value &entry : *segments) {
    const llvm::json::Object *segment = entry.getAsObject();
    const llvm::json::Array *values =
        segment ? segment->getArray("coefficients") : nullptr;
    std::optional<float> lower = segment && segment->get("lower")
                                     ? number(*segment->get("lower"))
                                     : std::nullopt;
    if (!values || !lower || values->empty() ||
        values->size() > tables.splineSegments[0].size())
      return failure();
    SmallVector<float> parsed;
    for (const llvm::json::Value &value : *values) {
      std::optional<float> coefficient = number(value);
      if (!coefficient)
        return failure();
      parsed.push_back(*coefficient);
    }
    if (!coefficients.empty() && parsed.size() != coefficients[0].size())
      return failure();
    lowers.push_back(*lower);
    coefficients.push_back(std::move(parsed));
  }
  for (auto [index, segment] : llvm::enumerate(tables.splineSegments)) {
    const SmallVector<float> &source =
        coefficients[std::min(index, coefficients.size() - 1)];
    llvm::copy(source, segment.begin());
  }
  for (auto [index, lower] : llvm::enumerate(tables.segmentLowerBounds))
    lower = index + 1 < lowers.size() ? lowers[index + 1]
                                      : std::numeric_limits<float>::infinity();
  tables.polynomialDegree = coefficients[0].size() - 1;
  return tables;
}

FailureOr<std::map<NluFunctionKind, CoefficientTables>>
readNluSplines(StringRef path, Operation *anchor) {
  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer)
    return anchor->emitError() << "cannot read NLU splines from " << path;
  llvm::Expected<llvm::json::Value> parsed =
      llvm::json::parse((*buffer)->getBuffer());
  if (!parsed)
    return anchor->emitError()
           << "malformed NLU splines: " << llvm::toString(parsed.takeError());
  const llvm::json::Object *splines =
      parsed->getAsObject() ? parsed->getAsObject()->getObject("splines")
                            : nullptr;
  if (!splines)
    return anchor->emitError() << "NLU splines lack a splines object";
  std::map<NluFunctionKind, CoefficientTables> out;
  for (NluFunctionKind function :
       {NluFunctionKind::Exp, NluFunctionKind::Reciprocal}) {
    const llvm::json::Object *spline =
        splines->getObject(*splineName(function));
    if (!spline)
      continue;
    FailureOr<CoefficientTables> tables = coefficientTables(*spline);
    if (failed(tables))
      return anchor->emitError()
             << "malformed NLU spline " << *splineName(function);
    out[function] = *tables;
  }
  return out;
}

std::string hex(ArrayRef<uint8_t> bytes) {
  std::string out;
  llvm::raw_string_ostream stream(out);
  for (uint8_t byte : bytes)
    stream << llvm::format_hex_no_prefix(byte, 2, true);
  return out;
}

StringRef rootName(HibRoot root) {
  switch (root) {
  case HibRoot::InputActivation:
    return "input_activation";
  case HibRoot::OutputActivation:
    return "output_activation";
  case HibRoot::ParameterRegion:
  case HibRoot::Parameter:
  case HibRoot::ParameterFill:
    return "parameter_region";
  case HibRoot::Scratch:
    return "scratch";
  }
  llvm_unreachable("unknown hib root");
}

StringRef kindName(GroupKind kind) {
  switch (kind) {
  case GroupKind::Preempt:
    return "preempt";
  case GroupKind::Init:
    return "init";
  case GroupKind::Op:
    return "op";
  case GroupKind::Permute:
    return "permute";
  case GroupKind::Empty:
    return "empty";
  case GroupKind::Gather:
    return "gather";
  case GroupKind::Scatter:
    return "scatter";
  case GroupKind::RingReshapeIdentity:
    return "ring_reshape_identity";
  case GroupKind::Relayout:
    return "relayout";
  case GroupKind::Padding:
    return "padding";
  }
  llvm_unreachable("unknown group kind");
}

StringRef suffixName(Suffix suffix) {
  switch (suffix) {
  case Suffix::None:
    return "";
  case Suffix::Init:
    return "init";
  case Suffix::Dest:
    return "dest";
  case Suffix::Gather:
    return "gather";
  case Suffix::Permute:
    return "permute";
  case Suffix::Bias:
    return "bias";
  }
  llvm_unreachable("unknown block suffix");
}

LogicalResult writeSchedule(StringRef path, func::FuncOp function,
                            ArrayRef<Group> groups, const Context &context) {
  llvm::DenseMap<Operation *, int64_t> index;
  for (auto [position, op] : llvm::enumerate(function.getBody().front()))
    index[&op] = position;
  llvm::json::Array scheduled;
  for (const Group &group : groups)
    scheduled.push_back(
        llvm::json::Array{kindName(group.kind), index.lookup(group.op)});
  auto blocks = [&](ArrayRef<StorageBlock> entries) {
    llvm::json::Array out;
    for (const StorageBlock &block : entries)
      out.push_back(llvm::json::Array{index.lookup(block.value),
                                      suffixName(block.suffix), block.start,
                                      block.end, block.size, block.offset});
    return out;
  };
  std::error_code error;
  llvm::raw_fd_ostream stream(path, error);
  if (error)
    return function.emitError() << "cannot write " << path;
  stream << llvm::json::Value(
      llvm::json::Object{{"groups", std::move(scheduled)},
                         {"narrow", blocks(context.narrow())},
                         {"host", blocks(context.host())},
                         {"wide", blocks(context.wide())}});
  return success();
}

class LegalizeDwcPass
    : public PassWrapper<LegalizeDwcPass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LegalizeDwcPass)

  LegalizeDwcPass() = default;
  LegalizeDwcPass(const LegalizeDwcPass &other) : PassWrapper(other) {}

  StringRef getArgument() const final { return "darwinn-legalize-dwc"; }

  StringRef getDescription() const final {
    return "Generate the Darwinn instruction program for a tensor operation "
           "function";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<DarwinnDialect>();
  }

  void runOnOperation() final {
    func::FuncOp function = getOperation();
    if (function.getBody().empty())
      return;
    SmallVector<Group> groups = deriveSchedule(function);
    FailureOr<Context> context = Context::create(function, groups);
    if (failed(context))
      return signalPassFailure();
    if (!scheduleOutput.empty() &&
        failed(writeSchedule(scheduleOutput, function, groups, *context)))
      return signalPassFailure();
    if (!nluSplines.empty()) {
      auto tables = readNluSplines(nluSplines, function);
      if (failed(tables))
        return signalPassFailure();
      context->nluTables = std::move(*tables);
    }
    FailureOr<SmallVector<Segment>> segments = generate(groups, *context);
    if (failed(segments))
      return signalPassFailure();
    FailureOr<GeneratedProgram> program = buildProgram(*segments, *context);
    if (failed(program))
      return signalPassFailure();
    if (chunksOutput.empty())
      return;
    auto scalar = ScalarEncoder::create(16, false);
    if (!scalar) {
      function.emitError(llvm::toString(scalar.takeError()));
      return signalPassFailure();
    }
    auto encoded = serializeProgram(program->chunks, *scalar);
    if (!encoded) {
      function.emitError(llvm::toString(encoded.takeError()));
      return signalPassFailure();
    }
    llvm::json::Array chunks, hibs;
    for (const InstructionBytes &chunk : encoded->getChunks())
      chunks.push_back(hex(chunk));
    for (const ChunkMeta &meta : program->meta)
      for (const auto &[index, reference] : meta.hibs) {
        const InstructionLayout *layout = encoded->findInstruction(reference);
        const Hib &hib = program->hibs[index];
        hibs.push_back(llvm::json::Array{reference.chunk,
                                         *layout->hibAddressBitOffset,
                                         rootName(hib.root), hib.offset});
      }
    std::error_code error;
    llvm::raw_fd_ostream stream(chunksOutput, error);
    if (error) {
      function.emitError() << "cannot write " << chunksOutput;
      return signalPassFailure();
    }
    stream << llvm::json::Value(llvm::json::Object{
        {"chunks", std::move(chunks)}, {"hibs", std::move(hibs)}});
  }

  Option<std::string> nluSplines{
      *this, "nlu-splines",
      llvm::cl::desc("JSON spline library for the NLU functions")};
  Option<std::string> scheduleOutput{
      *this, "schedule-output",
      llvm::cl::desc("Write the tensor groups and storage placements as JSON")};
  Option<std::string> chunksOutput{
      *this, "chunks-output",
      llvm::cl::desc("Write the encoded chunks and hib patch offsets as JSON")};
};

} // namespace

std::unique_ptr<Pass> mlir::darwinn::createLegalizeDwcPass() {
  return std::make_unique<LegalizeDwcPass>();
}

void mlir::darwinn::registerLegalizeDwcPass() {
  PassRegistration<LegalizeDwcPass>();
}
