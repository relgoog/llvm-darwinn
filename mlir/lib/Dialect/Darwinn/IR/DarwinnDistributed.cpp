#include "mlir/Dialect/Darwinn/IR/DarwinnDistributed.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/StringSwitch.h"

using namespace mlir;
using namespace mlir::darwinn;

static std::optional<DistributedMemorySpace> parseMemorySpace(StringRef name) {
  return llvm::StringSwitch<std::optional<DistributedMemorySpace>>(name)
      .Case("HOST_MEMORY", DistributedMemorySpace::HostMemory)
      .Case("TILE_MEMORY", DistributedMemorySpace::TileMemory)
      .Case("TILE_REGISTERS", DistributedMemorySpace::TileRegisters)
      .Default(std::nullopt);
}

static StringRef printMemorySpace(DistributedMemorySpace memorySpace) {
  switch (memorySpace) {
  case DistributedMemorySpace::HostMemory:
    return "HOST_MEMORY";
  case DistributedMemorySpace::TileMemory:
    return "TILE_MEMORY";
  case DistributedMemorySpace::TileRegisters:
    return "TILE_REGISTERS";
  }

  llvm_unreachable("invalid distributed memory space");
}

template <typename TypeT>
static Type parseDistributedType(AsmParser &parser) {
  SmallVector<int64_t> shape;
  Type elementType;
  StringAttr memorySpaceName;

  if (parser.parseLess() || parser.parseDimensionList(shape) ||
      parser.parseType(elementType) || parser.parseGreater() ||
      parser.parseSymbolName(memorySpaceName))
    return {};

  auto memorySpace = parseMemorySpace(memorySpaceName.getValue());

  if (!memorySpace) {
    parser.emitError(parser.getNameLoc(), "unknown distributed memory space ")
        << memorySpaceName.getValue();
    return {};
  }

  return TypeT::getChecked(
      [&]() { return parser.emitError(parser.getNameLoc()); },
      parser.getContext(), ArrayRef<int64_t>(shape), elementType, *memorySpace);
}

template <typename TypeT>
static Type parseCommunicatedType(AsmParser &parser) {
  SmallVector<int64_t> shape;
  Type elementType;
  StringRef memorySpaceName;

  auto parseDimension = [&]() -> ParseResult {
    if (succeeded(parser.parseOptionalQuestion())) {
      shape.push_back(ShapedType::kDynamic);
      return success();
    }

    int64_t dimension;

    if (parser.parseInteger(dimension))
      return failure();

    if (dimension < 0)
      return parser.emitError(parser.getCurrentLocation(),
                              "expected a nonnegative dimension or '?'");

    shape.push_back(dimension);
    return success();
  };

  if (parser.parseLParen() ||
      parser.parseCommaSeparatedList(AsmParser::Delimiter::Square,
                                     parseDimension) ||
      parser.parseComma() || parser.parseType(elementType) ||
      parser.parseComma() || parser.parseKeyword(&memorySpaceName) ||
      parser.parseRParen())
    return {};

  auto memorySpace = parseMemorySpace(memorySpaceName);

  if (!memorySpace) {
    parser.emitError(parser.getNameLoc(), "unknown distributed memory space ")
        << memorySpaceName;
    return {};
  }

  return TypeT::getChecked(
      [&]() { return parser.emitError(parser.getNameLoc()); },
      parser.getContext(), ArrayRef<int64_t>(shape), elementType, *memorySpace);
}

static void printDistributedType(AsmPrinter &printer, ArrayRef<int64_t> shape,
                                 Type elementType,
                                 DistributedMemorySpace memorySpace) {
  printer << '<';

  if (!shape.empty()) {
    printer.printDimensionList(shape);
    printer << 'x';
  }

  printer << elementType << ">@" << printMemorySpace(memorySpace);
}

static void printCommunicatedType(AsmPrinter &printer, ArrayRef<int64_t> shape,
                                  Type elementType,
                                  DistributedMemorySpace memorySpace) {
  printer << "([";

  llvm::interleaveComma(shape, printer, [&](int64_t dimension) {
    if (ShapedType::isDynamic(dimension))
      printer << '?';
    else
      printer << dimension;
  });

  printer << "], " << elementType << ", " << printMemorySpace(memorySpace)
          << ')';
}

static LogicalResult
verifyDistributedType(function_ref<InFlightDiagnostic()> emitError,
                      ArrayRef<int64_t> shape, Type elementType,
                      DistributedMemorySpace memorySpace) {
  if (memorySpace != DistributedMemorySpace::HostMemory &&
      memorySpace != DistributedMemorySpace::TileMemory &&
      memorySpace != DistributedMemorySpace::TileRegisters)
    return emitError() << "invalid distributed memory space";

  if (!elementType ||
      (isa<ShapedType>(elementType) && !isa<VectorType>(elementType)))
    return emitError() << "invalid distributed element type " << elementType;

  return RankedTensorType::verify(emitError, shape, elementType, {});
}

Type DistributedTensorType::parse(AsmParser &parser) {
  return parseDistributedType<DistributedTensorType>(parser);
}

void DistributedTensorType::print(AsmPrinter &printer) const {
  printDistributedType(printer, getShape(), getElementType(), getMemorySpace());
}

LogicalResult
DistributedTensorType::verify(function_ref<InFlightDiagnostic()> emitError,
                              ArrayRef<int64_t> shape, Type elementType,
                              DistributedMemorySpace memorySpace) {
  return verifyDistributedType(emitError, shape, elementType, memorySpace);
}

Type DistributedViewType::parse(AsmParser &parser) {
  return parseDistributedType<DistributedViewType>(parser);
}

void DistributedViewType::print(AsmPrinter &printer) const {
  printDistributedType(printer, getShape(), getElementType(), getMemorySpace());
}

LogicalResult
DistributedViewType::verify(function_ref<InFlightDiagnostic()> emitError,
                            ArrayRef<int64_t> shape, Type elementType,
                            DistributedMemorySpace memorySpace) {
  return verifyDistributedType(emitError, shape, elementType, memorySpace);
}

Type EmptyTensorType::parse(AsmParser &parser) {
  return parseCommunicatedType<EmptyTensorType>(parser);
}

void EmptyTensorType::print(AsmPrinter &printer) const {
  printCommunicatedType(printer, getShape(), getElementType(),
                        getMemorySpace());
}

LogicalResult
EmptyTensorType::verify(function_ref<InFlightDiagnostic()> emitError,
                        ArrayRef<int64_t> shape, Type elementType,
                        DistributedMemorySpace memorySpace) {
  return verifyDistributedType(emitError, shape, elementType, memorySpace);
}

Type WriteViewType::parse(AsmParser &parser) {
  return parseCommunicatedType<WriteViewType>(parser);
}

void WriteViewType::print(AsmPrinter &printer) const {
  printCommunicatedType(printer, getShape(), getElementType(),
                        getMemorySpace());
}

LogicalResult
WriteViewType::verify(function_ref<InFlightDiagnostic()> emitError,
                      ArrayRef<int64_t> shape, Type elementType,
                      DistributedMemorySpace memorySpace) {
  return verifyDistributedType(emitError, shape, elementType, memorySpace);
}

Type FilledViewType::parse(AsmParser &parser) {
  return parseCommunicatedType<FilledViewType>(parser);
}

void FilledViewType::print(AsmPrinter &printer) const {
  printCommunicatedType(printer, getShape(), getElementType(),
                        getMemorySpace());
}

LogicalResult
FilledViewType::verify(function_ref<InFlightDiagnostic()> emitError,
                       ArrayRef<int64_t> shape, Type elementType,
                       DistributedMemorySpace memorySpace) {
  return verifyDistributedType(emitError, shape, elementType, memorySpace);
}
