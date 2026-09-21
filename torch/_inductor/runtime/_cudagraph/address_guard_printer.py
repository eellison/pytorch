"""Checked C++ printers for integer and address predicates."""

from math import gcd

import sympy
from sympy.core.relational import Relational
from sympy.logic.boolalg import BooleanAtom, BooleanFunction

from torch.fx.experimental.symbolic_shapes import _ShapeGuardCppPrinter
from torch.utils._sympy.functions import FloorDiv, Max, Min, Mod, PythonMod
from torch.utils._sympy.value_ranges import bound_sympy, ValueRanges

from ._compiler.tma_dimension import TmaDimension


class GuardExportDeclined(ValueError):
    pass


class UIntGCD(sympy.Function):
    """Integer gcd after converting each operand to uint64."""

    is_integer = True
    is_nonnegative = True

    @classmethod
    def eval(cls, *args):
        if not args or any(arg.is_integer is not True for arg in args):
            raise ValueError("Unsigned gcd requires integer operands")
        if all(isinstance(arg, sympy.Integer) for arg in args):
            return sympy.Integer(gcd(*(int(arg) % (2 ** 64) for arg in args)))


class _BoundedPrinter(_ShapeGuardCppPrinter):
    def __init__(self, *args, domains, sources, integer_bits=64, allow_signed_dividend=False):
        super().__init__(*args)
        self.integer_bits = integer_bits
        self.allow_signed_dividend = allow_signed_dividend
        self.domains = domains
        self.sources = sources
        self.checked = set()
        self.discharged = []

    def print_source(self, source):
        if not any(source is recorded for recorded in self.sources):
            raise GuardExportDeclined("Guard source has no exact original boxed integer")
        return super().print_source(source)

    def bounds(self, expr):
        terms = expr.atoms(UIntGCD, TmaDimension)
        if not terms:
            return bound_sympy(expr, self.domains)
        domains = self.domains.copy()
        replacements = {}
        for term in terms:
            symbol = sympy.Dummy(integer=True, nonnegative=True)
            replacements[term] = symbol
            domains[symbol] = ValueRanges(0, 2 ** 64 - 1)
        return bound_sympy(expr.xreplace(replacements), domains)

    def _print_UIntGCD(self, expr):
        values = [f"static_cast<uint64_t>({self._print(arg)})" for arg in expr.args]
        result = values[0]
        for value in values[1:]:
            result = f"std::gcd<uint64_t>({result}, {value})"
        return f"static_cast<__int128>({result})"

    def _print_TmaDimension(self, expr):
        values = [f"static_cast<uint64_t>({self._print(arg)})" for arg in expr.args]
        if len(values) == 1:
            return f"static_cast<__int128>({values[0]})"
        pairs = ", ".join(f"{{{shape}, {stride}}}" for shape, stride in zip(values[::2], values[1::2], strict=True))
        return """static_cast<__int128>(([&]() -> uint64_t {
  const uint64_t values[][2] = {""" + pairs + """};
  uint64_t shape = 1, stride = 0;
  for (const auto& value : values) {
    const uint64_t divisor = std::gcd(stride, value[1]);
    shape = divisor ? (shape - 1) * (stride / divisor)
                      + (value[0] - 1) * (value[1] / divisor) + 1 : value[0];
    stride = divisor;
  }
  return shape;
})())"""

    def check_arithmetic(self, expr):
        if expr in self.checked:
            return
        if not expr.free_symbols.issubset(self.domains):
            raise GuardExportDeclined("Guard arithmetic has an unresolved symbolic source")
        if isinstance(expr, sympy.Piecewise):
            if not expr.args or expr.args[-1].cond is not sympy.true:
                raise GuardExportDeclined("Integer selection requires an unconditional final arm")
            for value, condition in expr.args:
                self.check_arithmetic(value)
                self.check_arithmetic(condition)
        else:
            for arg in expr.args:
                self.check_arithmetic(arg)
        if isinstance(expr, (Relational, BooleanFunction, BooleanAtom)):
            if not isinstance(expr, (Relational, sympy.And, sympy.Or, sympy.Not, BooleanAtom)):
                raise GuardExportDeclined("Unsupported integer guard predicate")
        else:
            if not isinstance(expr, (sympy.Integer, sympy.Symbol, sympy.Add, sympy.Mul,
                                     FloorDiv, Mod, PythonMod, sympy.Mod, Min, Max, sympy.Min, sympy.Max,
                                     sympy.Piecewise, UIntGCD, TmaDimension)):
                raise GuardExportDeclined(f"Unsupported guard arithmetic: {type(expr).__name__}")
            bounds = self.bounds(expr)
            limit = 2 ** (self.integer_bits - 1)
            if not bounds.is_int or bounds.lower <= -limit or bounds.upper >= limit:
                raise GuardExportDeclined(f"Guard arithmetic is not proven to fit signed int{self.integer_bits}")
            if isinstance(expr, (FloorDiv, Mod, PythonMod, sympy.Mod)):
                left, right = (self.bounds(arg) for arg in expr.args)
                signed = self.allow_signed_dividend and not isinstance(expr, Mod)
                if (not signed and left.lower < 0) or right.lower <= 0 or right.upper > (2 ** (self.integer_bits - 2) - 1):
                    requirement = "a positive bounded divisor" if signed else "nonnegative operands and a positive divisor"
                    raise GuardExportDeclined(f"Guard division requires {requirement}")
            if isinstance(expr, (sympy.Add, sympy.Mul)):
                partial = sympy.Integer(0 if isinstance(expr, sympy.Add) else 1)
                for arg in expr.args:
                    partial = partial + arg if isinstance(expr, sympy.Add) else partial * arg
                    bounds = self.bounds(partial)
                    if bounds.lower <= -limit or bounds.upper >= limit:
                        raise GuardExportDeclined(f"A guard arithmetic intermediate may overflow int{self.integer_bits}")
        self.checked.add(expr)

    def doprint(self, expr):
        if isinstance(expr, sympy.And):
            terms = []
            for term in expr.args:
                bounds = self.bounds(term)
                if bounds.lower is sympy.true and bounds.upper is sympy.true:
                    self.discharged.append(term)
                else:
                    terms.append(term)
            expr = sympy.And(*terms)
        if isinstance(expr, (Relational, BooleanFunction, BooleanAtom)):
            bounds = self.bounds(expr)
            if bounds.lower is sympy.true and bounds.upper is sympy.true:
                self.discharged.append(expr)
                return "true"
        self.check_arithmetic(expr)
        return super().doprint(expr)


class _AddressPrinter(_BoundedPrinter):
    def __init__(self, *args, domains, sources):
        super().__init__(*args, domains=domains, sources=sources, integer_bits=128, allow_signed_dividend=True)

    def bounds(self, expr):
        # bound_sympy maps sympy.Mod to C remainder; predicates use Python semantics.
        proof = expr.replace(lambda node: node.func is sympy.Mod,
                             lambda node: PythonMod(*node.args, evaluate=False))
        return super().bounds(proof)

    def doprint(self, expr):
        self.check_arithmetic(expr)
        return super().doprint(expr)

    def _print_Symbol(self, expr):
        return f"static_cast<__int128>({super()._print_Symbol(expr)})"

    def _print_Integer(self, expr):
        value = int(expr)
        if abs(value) >= 2 ** 127:
            raise GuardExportDeclined("Guard integer literal does not fit the checked signed int128 domain")
        high, low = divmod(abs(value), 2 ** 64)
        result = f"static_cast<__int128>({low}ULL)"
        if high:
            result = f"((static_cast<__int128>({high}ULL) << 64) + {result})"
        return f"(-{result})" if value < 0 else result

    def _print_FloorDiv(self, expr):
        left, right = (self._print(arg) for arg in expr.args)
        return f"((({left}) / ({right})) - ((({left}) % ({right})) < 0))"

    def _print_Mod(self, expr):
        left, right = (self._print(arg) for arg in expr.args)
        remainder = f"(({left}) % ({right}))"
        return f"(({remainder} < 0) ? ({remainder} + ({right})) : {remainder})"

    _print_PythonMod = _print_Mod

    def _print_min_max(self, expr, name):
        arguments = ", ".join(self._print(arg) for arg in expr.args)
        return f"std::{name}<__int128>({{{arguments}}})"
