"""Lower sealed CuTe numeric consumers without evaluating example values."""

from collections.abc import Callable
from dataclasses import dataclass

from torch._inductor.runtime._cudagraph._compiler.owned_numeric import OwnedNumeric
from torch._inductor.runtime._cudagraph._compiler.overflow_properties import _NSW, _NUW, _OVERFLOW
from torch._inductor.runtime.cudagraph_arg_mapping import grid_expression_inputs, IntExpr
from torch._inductor.runtime._cudagraph._compiler.values import integer_width, ScalarValue


class NumericDeclined(ValueError):
    pass


@dataclass(frozen=True)
class NumericSource:
    expression: IntExpr
    lower: int | None
    upper: int | None


@dataclass(frozen=True)
class Comparison:
    predicate: str
    llvm_type: str
    left: IntExpr
    right: IntExpr


@dataclass(frozen=True)
class NumericValue:
    llvm_type: str
    expression: IntExpr | Comparison
    lower: int
    upper: int


@dataclass(frozen=True)
class RangeObligation:
    expression: IntExpr
    lower: int
    upper: int
    reason: str


@dataclass(frozen=True)
class LoweredNumeric:
    values: tuple[NumericValue, ...]
    obligations: tuple[RangeObligation, ...]


@dataclass(frozen=True)
class _Source:
    original: int
    llvm_type: str
    leaves: tuple[tuple[tuple[int, ...], str], ...] | None


def _bounds(llvm_type):
    try:
        width = integer_width(llvm_type)
    except ValueError as error:
        raise NumericDeclined("Numeric consumers require supported integer types") from error
    return (0, 1) if width == 1 else (-(1 << (width - 1)), (1 << (width - 1)) - 1)


def _unsigned(value):
    width = integer_width(value.llvm_type)
    if value.lower >= 0:
        return value.lower, value.upper
    if value.upper < 0:
        return value.lower + (1 << width), value.upper + (1 << width)
    return 0, (1 << width) - 1


def lower_numeric(
    program: OwnedNumeric,
    resolve_source: Callable[[int, tuple[int, ...]], NumericSource],
) -> LoweredNumeric:
    """Returned ranges assume every obligation is enforced before any result use."""
    if type(program) is not OwnedNumeric or not callable(resolve_source):
        raise NumericDeclined("Expected an owned numeric program and a source resolver")
    program.check()
    cfg = program._cfg
    if len(cfg.blocks) != 1:
        raise NumericDeclined("Numeric lowering supports only one return block")
    block, = cfg.blocks
    if block.terminator.kind != "return" or block.terminator.edges:
        raise NumericDeclined("Numeric lowering cannot flatten control flow")
    if len(block.arguments) != len(program.source_order) or len(block.arguments) != len(cfg.argument_types):
        raise NumericDeclined("Numeric entry lost its original source ordering")
    flags = dict(program._flags)
    expected_flags = tuple(item.result for item in block.instructions if item.expression.kind in _OVERFLOW)
    if tuple(flags) != expected_flags or any(value & ~(_NSW | _NUW) for value in flags.values()):
        raise NumericDeclined("Numeric overflow properties lack exact instruction coverage")
    predicates = dict(zip(program._predicates, ("eq", "ne", "slt", "sle", "sgt", "sge", "ult", "ule", "ugt", "uge")))
    if len(predicates) != 10:
        raise NumericDeclined("Numeric comparison predicate identities are incomplete")
    slots, resolved, obligations = {}, {}, []
    for slot, original, llvm_type in zip(block.arguments, program.source_order, cfg.argument_types):
        typ = cfg.types[slot]
        if typ.llvm_type != llvm_type or slot in slots:
            raise NumericDeclined("Numeric entry slot types differ from the original signature")
        slots[slot] = _Source(original, llvm_type, typ.leaves)

    def reference(slot, path, llvm_type):
        if slot not in slots:
            raise NumericDeclined("Numeric operands must name earlier CFG slots")
        value = slots[slot]
        if type(value) is NumericValue:
            if path or value.llvm_type != llvm_type:
                raise NumericDeclined("Numeric operand changed its scalar type or field path")
            return value
        if value.leaves is None:
            if path or value.llvm_type != llvm_type:
                raise NumericDeclined("Scalar source changed its exact compiler type")
        elif (path, llvm_type) not in value.leaves:
            raise NumericDeclined("Aggregate source lacks the exact typed scalar leaf")
        minimum, maximum = _bounds(llvm_type)
        key = value.original, path, llvm_type
        if key in resolved:
            return resolved[key]
        source = resolve_source(value.original, path)
        if (type(source) is not NumericSource
                or source.lower is not None and type(source.lower) is not int
                or source.upper is not None and type(source.upper) is not int
                or source.lower is not None and source.upper is not None and source.lower > source.upper):
            raise NumericDeclined("Source resolver must supply an exact expression and inherited interval")
        expr = source.expression
        literal = (type(expr) is IntExpr and expr.op == "constant" and type(expr.value) is int
                   and type(expr.args) is tuple and not expr.args)
        if not literal and grid_expression_inputs(expr) is None:
            raise NumericDeclined("Source expressions require supported integer arithmetic over boxed leaves")
        lower = minimum if source.lower is None else max(minimum, source.lower)
        upper = maximum if source.upper is None else min(maximum, source.upper)
        if expr.op == "constant":
            if not lower <= expr.value <= upper:
                raise NumericDeclined("Literal source is outside its inherited or compiler-width range")
            lower = upper = expr.value
        elif lower > upper:
            raise NumericDeclined("Inherited source domain cannot satisfy its compiler width")
        elif (source.lower is None or source.lower < minimum
              or source.upper is None or source.upper > maximum):
            obligation = RangeObligation(expr, minimum, maximum, "source " + llvm_type + " representation")
            if obligation not in obligations:
                obligations.append(obligation)
        result = NumericValue(llvm_type, expr, lower, upper)
        resolved[key] = result
        return result

    def operand(flow):
        if (flow.kind != "argument" or flow.operands or flow.attributes
                or flow.constant is not None or flow.predicate is not None):
            raise NumericDeclined("Arithmetic requires exact earlier compiler operand slots")
        return reference(flow.argument, flow.path, flow.llvm_type)

    def numeric_operand(flow):
        value = operand(flow)
        if type(value.expression) is Comparison:
            comparison = value.expression
            tag = comparison.predicate
            if tag not in ("eq", "ne"):
                tag = tag[1:]
            expression = IntExpr(tag, args=(comparison.left, comparison.right))
            return NumericValue(value.llvm_type, expression, value.lower, value.upper)
        return value

    def bounded(value, maximum):
        if value.lower < 0 or value.lower > maximum:
            raise NumericDeclined("Arithmetic has no supported nonnegative domain")
        if value.upper <= maximum:
            return value
        expression, factor = value.expression, 1
        while expression.op == "multiply" and len(expression.args) == 2:
            left, right = expression.args
            if left.op == "constant":
                left, right = right, left
            if right.op != "constant" or right.value <= 0:
                raise NumericDeclined("Arithmetic domain requires one boxed leaf and positive constant factors")
            factor *= right.value
            expression = left
        if expression.op != "boxed":
            raise NumericDeclined("Arithmetic domain requires one boxed leaf and positive constant factors")
        upper = maximum // factor
        # Leaf predicates are safe independently of guard ordering and source-width checks.
        obligation = RangeObligation(expression, 0, upper, "arithmetic " + value.llvm_type + " nsw domain")
        if obligation not in obligations:
            obligations.append(obligation)
        return NumericValue(value.llvm_type, value.expression, value.lower, min(value.upper, upper * factor))

    for instruction in block.instructions:
        flow, slot = instruction.expression, instruction.result
        minimum, maximum = _bounds(flow.llvm_type)
        if slot in slots or cfg.types[slot].llvm_type != flow.llvm_type or cfg.types[slot].leaves is not None:
            raise NumericDeclined("Numeric instruction changed its exact result slot type")
        if flow.kind == "argument":
            result = operand(flow)
        elif flow.kind in ("constant", "zero"):
            if flow.argument is not None or flow.path or flow.operands or flow.attributes or flow.predicate is not None:
                raise NumericDeclined("Numeric literal has unexpected operands or properties")
            number = 0
            if flow.kind == "constant":
                if flow.constant is None:
                    raise NumericDeclined("Numeric literal lacks its decoded compiler bytes")
                typ, data, size = flow.constant
                if typ != flow.llvm_type or type(data) is not bytes or len(data) != size:
                    raise NumericDeclined("Numeric literal changed its compiler byte representation")
                value = ScalarValue(typ, int.from_bytes(data, "little"), size)
                if value.data() != data:
                    raise NumericDeclined("Numeric literal changed its compiler byte representation")
                number = value.integer(signed=typ != "i1")
            elif flow.constant is not None:
                raise NumericDeclined("Zero instruction carries a foreign constant")
            result = NumericValue(flow.llvm_type, IntExpr("constant", number), number, number)
        else:
            if flow.argument is not None or flow.path or flow.constant is not None:
                raise NumericDeclined("Numeric instruction carries unsupported source properties")
            operands = tuple(numeric_operand(item) for item in flow.operands)
            if flow.kind in ("llvm.sext", "llvm.zext"):
                if len(operands) != 1 or flow.attributes or flow.predicate is not None:
                    raise NumericDeclined("Integer extension has unsupported operands or properties")
                value, = operands
                before, after = integer_width(value.llvm_type), integer_width(flow.llvm_type)
                if before >= after:
                    raise NumericDeclined("Integer extension must increase the actual type width")
                expr, lower, upper = value.expression, value.lower, value.upper
                if flow.kind == "llvm.zext" and lower < 0 or flow.kind == "llvm.sext" and before == 1 and upper > 0:
                    if expr.op != "constant":
                        raise NumericDeclined("Integer extension needs unsupported dynamic bit reinterpretation")
                    number = expr.value + (1 << before) if flow.kind == "llvm.zext" else -expr.value
                    expr, lower, upper = IntExpr("constant", number), number, number
                result = NumericValue(flow.llvm_type, expr, lower, upper)
            elif flow.kind == "llvm.icmp":
                if (len(operands) != 2 or flow.llvm_type != "i1" or flow.predicate not in predicates
                        or tuple(name for name, _ in flow.attributes) != ("predicate",)
                        or operands[0].llvm_type != operands[1].llvm_type):
                    raise NumericDeclined("Comparison lost its exact decoded predicate or operand types")
                left, right = operands
                predicate = predicates[flow.predicate]
                if integer_width(left.llvm_type) == 1 and predicate not in ("eq", "ne"):
                    raise NumericDeclined("Ordered i1 comparisons require separate signed-bit semantics")
                if predicate.startswith("u") and (left.lower < 0 or right.lower < 0):
                    raise NumericDeclined("Unsigned comparison requires proven nonnegative operands")
                result = NumericValue("i1", Comparison(predicate, left.llvm_type, left.expression, right.expression), 0, 1)
            elif flow.kind == "llvm.sdiv":
                if (len(operands) != 2 or flow.attributes or flow.predicate is not None or flow.llvm_type == "i1"
                        or any(value.llvm_type != flow.llvm_type for value in operands)):
                    raise NumericDeclined("Signed division lost its exact operand widths or properties")
                dividend, divisor = operands
                if (dividend.lower < 0 or divisor.expression.op != "constant" or divisor.expression.value <= 0):
                    raise NumericDeclined("Signed division requires a nonnegative dividend and a positive constant divisor")
                denominator = divisor.expression.value
                result = NumericValue(flow.llvm_type, IntExpr("floordiv", args=(dividend.expression, divisor.expression)),
                                      dividend.lower // denominator, dividend.upper // denominator)
            elif flow.kind == "llvm.and":
                if (len(operands) != 2 or flow.llvm_type != "i1" or flow.attributes or flow.predicate is not None
                        or any(value.llvm_type != "i1" for value in operands)):
                    raise NumericDeclined("Boolean conjunction requires exact i1 operands")
                left, right = operands
                result = NumericValue("i1", IntExpr("and", args=(left.expression, right.expression)),
                                      int(left.lower == right.lower == 1), int(left.upper == right.upper == 1))
            elif flow.kind == "llvm.select":
                if (len(operands) != 3 or flow.predicate is not None
                        or flow.attributes not in ((), (("fastmathFlags", "#llvm.fastmath<none>"),))
                        or operands[0].llvm_type != "i1"
                        or any(value.llvm_type != flow.llvm_type for value in operands[1:])):
                    raise NumericDeclined("Selection lost its exact condition, arm widths or properties")
                condition, yes, no = operands
                result = NumericValue(flow.llvm_type, IntExpr("select", args=(condition.expression, yes.expression, no.expression)),
                                      min(yes.lower, no.lower), max(yes.upper, no.upper))
            elif flow.kind in ("llvm.add", "llvm.mul"):
                if (len(operands) != 2 or flow.attributes or flow.predicate is not None or flow.llvm_type == "i1"
                        or any(value.llvm_type != flow.llvm_type for value in operands)):
                    raise NumericDeclined("Arithmetic lost its exact operand widths or properties")
                left, right = operands
                if flow.kind == "llvm.add":
                    lower, upper = left.lower + right.lower, left.upper + right.upper
                else:
                    products = tuple(a * b for a in (left.lower, left.upper) for b in (right.lower, right.upper))
                    lower, upper = min(products), max(products)
                if lower < minimum or upper > maximum:
                    if flags[slot] & _NSW and left.lower >= 0 and right.lower >= 0:
                        if left.expression.op == "constant":
                            left, right = right, left
                        if right.expression.op != "constant" or right.expression.value <= 0:
                            raise NumericDeclined("Arithmetic domain requires one boxed leaf and positive constants")
                        if flow.kind == "llvm.add":
                            left = bounded(left, maximum - right.expression.value)
                            lower, upper = left.lower + right.lower, left.upper + right.upper
                        else:
                            left = bounded(left, maximum // right.expression.value)
                            lower, upper = left.lower * right.lower, left.upper * right.upper
                    else:
                        raise NumericDeclined("Arithmetic lacks a proof against signed overflow or wrapping")
                if flags[slot] & _NUW:
                    a, b = _unsigned(left), _unsigned(right)
                    unsigned_max = a[1] + b[1] if flow.kind == "llvm.add" else a[1] * b[1]
                    if unsigned_max >= 1 << integer_width(flow.llvm_type):
                        raise NumericDeclined("Arithmetic lacks a proof of its compiler nuw property")
                a, b = left.expression, right.expression
                if a.op == b.op == "constant":
                    number = a.value + b.value if flow.kind == "llvm.add" else a.value * b.value
                    expr = IntExpr("constant", number)
                elif left.lower < 0 or right.lower < 0:
                    raise NumericDeclined("Numeric arithmetic requires proven nonnegative operands")
                elif flow.kind == "llvm.add" and a.op == "constant" and a.value == 0:
                    expr = b
                elif flow.kind == "llvm.add" and b.op == "constant" and b.value == 0:
                    expr = a
                else:
                    expr = IntExpr("add" if flow.kind == "llvm.add" else "multiply", args=(a, b))
                result = NumericValue(flow.llvm_type, expr, lower, upper)
            else:
                raise NumericDeclined("Unsupported numeric instruction: " + flow.kind)
        slots[slot] = result
    if len(block.terminator.values) != len(cfg.result_types):
        raise NumericDeclined("Numeric return lost its compiler result signature")
    values = tuple(reference(slot, (), typ) for slot, typ in zip(block.terminator.values, cfg.result_types))
    program.check()
    return LoweredNumeric(values, tuple(obligations))
