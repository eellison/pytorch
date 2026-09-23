"""Lower symbolic addresses and their exact reuse predicates."""

import sympy

import torch
from torch._inductor.runtime._cudagraph._compiler.fx_adapter.contract import (
    FXTraceDeclined,
    NormalizeEvent,
)
from torch._inductor.runtime.cudagraph_arg_mapping import (
    BufferSource,
    InputSource,
    IntExpr,
    ParameterSource,
    PointerSource,
)
from torch.utils._sympy.functions import FloorDiv, Mod, PythonMod


def symbolic_integer(value, symbols, decline=FXTraceDeclined):
    if type(value) is int:
        return sympy.Integer(value)
    if type(value) is not IntExpr:
        raise decline("Guard lost its exact integer expression")
    if value.op == "constant" and type(value.value) is int and not value.args:
        return sympy.Integer(value.value)
    if value.op == "boxed" and not value.args and value.value in symbols:
        return symbols[value.value]
    if (
        value.op == "storage_offset"
        and not value.args
        and (value.op, value.value) in symbols
    ):
        return symbols[value.op, value.value]
    if (
        value.op in ("size", "stride")
        and len(value.args) == 1
        and value.args[0].op == "constant"
        and (value.op, value.value, value.args[0].value) in symbols
    ):
        # a host trace's tensor metadata load (its tape has no boxed integers)
        return symbols[value.op, value.value, value.args[0].value]
    if value.op == "select" and value.value is None and len(value.args) == 3:
        condition, when_true, when_false = (
            symbolic_integer(arg, symbols, decline) for arg in value.args
        )
        return sympy.Piecewise((when_true, sympy.Eq(condition, 1)), (when_false, True))
    if value.op in ("max", "min") and value.value is None and len(value.args) >= 2:
        operands = tuple(symbolic_integer(arg, symbols, decline) for arg in value.args)
        return (sympy.Max if value.op == "max" else sympy.Min)(*operands)
    if value.value is None and len(value.args) == 2:
        left, right = (symbolic_integer(arg, symbols, decline) for arg in value.args)
        if value.op == "multiply":
            return left * right
        if value.op == "add":
            return left + right
        if value.op == "ceildiv" and isinstance(right, sympy.Integer) and right > 0:
            return FloorDiv(left + right - 1, right)
        if value.op == "floordiv" and (
            not isinstance(right, sympy.Integer) or right > 0
        ):
            return FloorDiv(
                left,
                right,
                evaluate=isinstance(left, sympy.Integer)
                and isinstance(right, sympy.Integer),
            )
        comparisons = {
            "eq": sympy.Eq,
            "ne": sympy.Ne,
            "lt": sympy.Lt,
            "le": sympy.Le,
            "gt": sympy.Gt,
            "ge": sympy.Ge,
        }
        if value.op in comparisons:
            return sympy.Piecewise((1, comparisons[value.op](left, right)), (0, True))
        if value.op == "and":
            return sympy.Piecewise(
                (1, sympy.And(sympy.Eq(left, 1), sympy.Eq(right, 1))), (0, True)
            )
    raise decline("Guard exceeds the supported native integer expressions")


def pointer_alignment_guard(pointer, alignment, root_alignments, trace):
    if (
        type(pointer) is not PointerSource
        or pointer.root not in root_alignments
        or type(alignment) is not int
        or alignment <= 0
        or alignment & (alignment - 1)
    ):
        raise FXTraceDeclined(
            "Pointer alignment requires an exact storage root and power-of-two requirement"
        )
    symbols = {origin.value: symbol for symbol, origin in trace.symbol_sources.items()}
    symbols.update(
        (("storage_offset", binding.index), binding.symbol)
        for binding in trace.storage_offset_bindings
    )
    address = symbolic_integer(pointer.byte_offset, symbols)
    if alignment > root_alignments[pointer.root]:
        normalized = {
            trace.tensor_roots.event(event).root
            for event in trace.events
            if type(event) is NormalizeEvent
        }
        if type(pointer.root) is not InputSource or pointer.root in normalized:
            raise FXTraceDeclined(
                "Pointer specialization exceeds its allocated or normalized root alignment"
            )
        bindings = [
            binding
            for binding in trace.address_bindings
            if binding.root == pointer.root and binding.generation == 0
        ]
        if len(bindings) != 1:
            raise FXTraceDeclined(
                "Pointer alignment lost its original input address binding"
            )
        address += bindings[0].symbol
    return sympy.Eq(sympy.Mod(address, alignment), 0)


def lower_address_scalar(value, abi_type, trace, early_expression):
    """Return a late source and reuse guards, or None for ordinary integers."""
    environment = trace.shape_env
    if type(value) is torch.SymInt:
        if value.node.shape_env is not environment:
            raise FXTraceDeclined(
                "Address scalar belongs to another tracing environment"
            )
        value = value.node.expr
    bindings = {binding.symbol: binding for binding in trace.address_bindings}
    if not isinstance(value, sympy.Expr) or not value.free_symbols.intersection(
        bindings
    ):
        return None
    bits = {"i32": 32, "i64": 64}.get(abi_type)
    if bits is None:
        raise FXTraceDeclined("Address scalar requires a selected signed i32/i64 ABI")
    metadata = {binding.symbol for binding in trace.storage_offset_bindings}
    if not value.free_symbols.issubset(
        bindings.keys() | trace.symbol_sources.keys() | metadata
    ):
        raise FXTraceDeclined("Address scalar has an unrecorded integer source")
    replaced = {
        trace.tensor_roots.event(event).root
        for event in trace.events
        if type(event) is NormalizeEvent
    }
    selected = tuple(
        bindings[symbol] for symbol in value.free_symbols.intersection(bindings)
    )
    if any(binding.generation != 0 or binding.root in replaced for binding in selected):
        raise FXTraceDeclined(
            "Address scalar cannot use an input root replaced by normalization"
        )
    if any(
        type(binding.root) not in (InputSource, BufferSource) for binding in selected
    ):
        raise FXTraceDeclined("Address scalar lost its recorded storage root")

    signed_min, signed_max, unsigned_max = -(1 << 63), (1 << 63) - 1, (1 << 64) - 1
    owned = any(type(binding.root) is BufferSource for binding in selected)
    guards = []
    symbols = {origin.value: symbol for symbol, origin in trace.symbol_sources.items()}
    symbols.update(
        (("storage_offset", binding.index), binding.symbol)
        for binding in trace.storage_offset_bindings
    )

    def bounds(term):
        result = environment.bound_sympy(term)
        if not isinstance(result.lower, sympy.Integer) or not isinstance(
            result.upper, sympy.Integer
        ):
            raise FXTraceDeclined(
                "Address arithmetic requires finite proven integer bounds"
            )
        return int(result.lower), int(result.upper)

    def require_range(term, minimum, maximum, message):
        lower, upper = bounds(term)
        if lower >= minimum and upper <= maximum:
            return
        if owned or upper < minimum or lower > maximum:
            raise FXTraceDeclined(message)
        if lower < minimum:
            guards.append(sympy.Ge(term, minimum))
        if upper > maximum:
            guards.append(sympy.Le(term, maximum))

    def representable(term):
        maximum = signed_max if owned and bounds(term)[0] < 0 else unsigned_max
        require_range(
            term,
            signed_min,
            maximum,
            "Address arithmetic intermediate can overflow its native 64-bit representation",
        )

    def check_early(expression):
        arguments = tuple(check_early(argument) for argument in expression.args)
        value = symbolic_integer(expression, symbols)
        require_range(
            value,
            signed_min,
            signed_max,
            "Early integer operand exceeds signed native transport",
        )
        if expression.op in ("ceildiv", "floordiv"):
            require_range(
                arguments[0],
                0,
                signed_max,
                "Early division requires a nonnegative numerator",
            )
            require_range(
                arguments[1],
                1,
                signed_max,
                "Early division requires a positive denominator",
            )
        elif expression.op in ("and", "select"):
            for condition in arguments if expression.op == "and" else arguments[:1]:
                require_range(
                    condition, 0, 1, "Early boolean operand must be zero or one"
                )
        return value

    def lower(term):
        representable(term)
        if isinstance(term, sympy.Integer):
            return ParameterSource("constant", 64, int(term) & unsigned_max)
        if isinstance(term, sympy.Symbol) and term in bindings:
            return ParameterSource(
                "pointer",
                64,
                PointerSource(bindings[term].root, IntExpr("constant", 0)),
            )
        if not term.free_symbols.intersection(bindings):
            expression = early_expression(term)
            if type(expression) is int:
                expression = IntExpr("constant", expression)
            check_early(expression)
            return ParameterSource("value", 64, expression)
        if isinstance(term, (sympy.Add, sympy.Mul)):
            op = "add" if isinstance(term, sympy.Add) else "mul"
            partial, *remaining = term.args
            result = lower(partial)
            for argument in remaining:
                operand = lower(argument)
                partial = partial + argument if op == "add" else partial * argument
                representable(partial)
                result = ParameterSource(op, 64, args=(result, operand))
            return result
        if isinstance(term, (FloorDiv, Mod, PythonMod, sympy.Mod)):
            dividend, divisor = term.args
            if bounds(dividend)[0] < 0 or bounds(divisor)[0] <= 0:
                raise FXTraceDeclined(
                    "Address division requires a nonnegative dividend and positive divisor"
                )
            return ParameterSource(
                "udiv" if isinstance(term, FloorDiv) else "urem",
                64,
                args=(lower(dividend), lower(divisor)),
            )
        raise FXTraceDeclined(f"Unsupported address scalar arithmetic: {term!r}")

    result = lower(value)
    abi_min, abi_max = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    message = (
        "Owned address scalar lacks a proven selected ABI range"
        if owned
        else "Address scalar lacks a representable selected ABI value"
    )
    require_range(value, abi_min, abi_max, message)
    if bits == 32:
        result = ParameterSource("trunc", 32, args=(result,))
    return result, tuple(dict.fromkeys(guards))
