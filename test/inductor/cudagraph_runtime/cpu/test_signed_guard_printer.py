"""Compile signed address predicates through the existing native guard ABI."""

import ctypes
from math import gcd



import sympy
from torch._inductor.runtime._cudagraph.address_guard_printer import (
    _AddressPrinter, _BoundedPrinter, GuardExportDeclined, UIntGCD,
)
from torch._inductor.runtime._cudagraph.guard_export import _AddressTerminalPrinter
from torch._dynamo.source import LocalSource
from torch._inductor.codecache import CppCodeCache
from torch.testing._internal.common_utils import instantiate_parametrized_tests, parametrize, run_tests, TestCase
from torch.utils._sympy.functions import FloorDiv, Mod as TorchMod, PythonMod
from torch.utils._sympy.value_ranges import ValueRanges


UINT64_MAX = (1 << 64) - 1
I64_MAX = (1 << 63) - 1


@instantiate_parametrized_tests
class TestSignedGuardPrinter(TestCase):
    def printer(self, intervals, printer_type=_AddressPrinter):
        symbols = sympy.symbols(f"value0:{len(intervals)}", integer=True)
        sources = tuple(LocalSource(f"source{index}") for index in range(len(intervals)))
        mapping = {symbol: [source] for symbol, source in zip(symbols, sources, strict=True)}
        domains = {symbol: ValueRanges(*interval) for symbol, interval in zip(symbols, intervals, strict=True)}
        if printer_type is _AddressTerminalPrinter:
            printer = printer_type(mapping, lambda source: source.name, mapping, sources=sources,
                source_domains={id(source): domains[symbol] for symbol, source in zip(symbols, sources, strict=True)},
                symbols=frozenset(symbols))
        else:
            printer = printer_type(mapping, lambda source: source.name, mapping, domains=domains, sources=sources)
        return symbols, sources, printer

    def compile_predicate(self, printer, sources, expression, unsigned=()):
        printed = printer.doprint(expression)
        declarations = []
        for index, source in enumerate(sources):
            if source in printer.source_to_symbol:
                dtype = "uint64_t" if index in unsigned else "int64_t"
                name = printer.source_to_symbol[source].name
                declarations.append(f"  const {dtype} {name} = static_cast<{dtype}>(int_values[{index}]);")
        cpp = """#include <algorithm>
#include <cstdint>
#include <numeric>
extern "C" int8_t guard(int64_t* int_values, double*) {
""" + "\n".join(declarations) + f"""
  return {printed};
}}
"""
        library = CppCodeCache.load(cpp)
        address = ctypes.cast(library.guard, ctypes.c_void_p).value
        self.assertIsNotNone(address)
        predicate = ctypes.CFUNCTYPE(ctypes.c_int8, ctypes.POINTER(ctypes.c_int64),
                                    ctypes.POINTER(ctypes.c_double))(address)
        return library, predicate, printed

    def evaluate(self, predicate, values):
        bits = (ctypes.c_uint64 * len(values))(*(value & UINT64_MAX for value in values))
        return predicate(ctypes.cast(bits, ctypes.POINTER(ctypes.c_int64)), None)

    @parametrize("operation", ("floor", "python_mod", "sympy_mod"))
    def test_negative_dividends_and_exact_multiples(self, operation):
        (value, expected), sources, printer = self.printer(((-64, 64), (-8, 32)))
        constructor = {"floor": FloorDiv, "python_mod": PythonMod, "sympy_mod": sympy.Mod}[operation]
        expression = sympy.Eq(constructor(value, 16, evaluate=False), expected)
        library, predicate, printed = self.compile_predicate(printer, sources, expression)
        self.assertNotIn(printed, ("true", "false"))
        for dividend in (-33, -32, -31, -17, -16, -15, -1, 0, 1, 15, 16, 17, 31, 32, 33):
            reference = dividend // 16 if operation == "floor" else dividend % 16
            self.assertEqual(self.evaluate(predicate, (dividend, reference)), 1)
            self.assertEqual(self.evaluate(predicate, (dividend, reference + 1)), 0)
        self.assertIsNotNone(library)

    def test_high_bit_pointer_minus_storage_offset(self):
        (pointer, offset, quotient, remainder), sources, printer = self.printer(
            ((0, UINT64_MAX), (0, I64_MAX), (-I64_MAX, I64_MAX), (0, 15)))
        value = pointer - 4 * offset
        expression = sympy.And(sympy.Eq(FloorDiv(value, 16, evaluate=False), quotient),
                               sympy.Eq(sympy.Mod(value, 16, evaluate=False), remainder))
        library, predicate, _ = self.compile_predicate(printer, sources, expression, unsigned=(0,))
        samples = ((1 << 63, 1 << 61), ((1 << 63) - 1, 1 << 61),
                   ((1 << 63) + 16, (1 << 61) + 8), ((1 << 63) + 15, (1 << 61) + 8),
                   (UINT64_MAX, 1 << 62), (UINT64_MAX, 0), (0, I64_MAX))
        for address, storage_offset in samples:
            dividend = address - 4 * storage_offset
            floor, mod = dividend // 16, dividend % 16
            self.assertEqual(self.evaluate(predicate, (address, storage_offset, floor, mod)), 1)
            self.assertEqual(self.evaluate(predicate, (address, storage_offset, floor, (mod + 1) % 16)), 0)
        self.assertIsNotNone(library)

    def test_sympy_mod_negative_domain_is_not_discharged_true(self):
        (value,), sources, printer = self.printer(((-31, -17),))
        modulo = sympy.Mod(value, 16, evaluate=False)
        expression = modulo < 1
        self.assertEqual(printer.bounds(modulo), printer.bounds(PythonMod(value, 16, evaluate=False)))
        bounds = printer.bounds(expression)
        self.assertFalse(bounds.lower is sympy.true and bounds.upper is sympy.true)
        library, predicate, printed = self.compile_predicate(printer, sources, expression)
        self.assertNotEqual(printed, "true")
        self.assertNotIn(expression, printer.discharged)
        for dividend in range(-31, -16):
            self.assertEqual(self.evaluate(predicate, (dividend,)), int(dividend % 16 < 1))
        self.assertIsNotNone(library)

    @parametrize("fault", ("default_floor", "default_python_mod", "torch_mod", "zero_divisor",
                           "negative_divisor", "large_divisor", "overflowing_prefix"))
    def test_unchanged_domain_and_overflow_refusals(self, fault):
        printer_type = _BoundedPrinter if fault.startswith("default_") else _AddressPrinter
        (value, other), _, printer = self.printer(((-64, 64), (0, 16)), printer_type)
        if fault.startswith("default_"):
            self.assertFalse(printer.allow_signed_dividend)
            self.assertEqual(printer.integer_bits, 64)
        product = sympy.Mul(value, 1 << 126, 0, evaluate=False)
        expressions = {
            "default_floor": sympy.Eq(FloorDiv(value, 16, evaluate=False), other),
            "default_python_mod": sympy.Eq(PythonMod(value, 16, evaluate=False), other),
            "torch_mod": sympy.Eq(TorchMod(value, 16, evaluate=False), other),
            "zero_divisor": FloorDiv(value, other, evaluate=False),
            "negative_divisor": FloorDiv(value, -1, evaluate=False),
            "large_divisor": FloorDiv(value, 1 << 126, evaluate=False),
            "overflowing_prefix": sympy.Eq(product, 0, evaluate=False),
        }
        with self.assertRaises(GuardExportDeclined):
            printer.doprint(expressions[fault])

    def test_real_terminal_printer_mro_keeps_signed_semantics(self):
        (value, expected), sources, printer = self.printer(((-64, 64), (-8, 8)), _AddressTerminalPrinter)
        self.assertEqual(printer.domains, {})
        self.assertEqual(printer.integer_bits, 128)
        self.assertTrue(printer.allow_signed_dividend)
        expression = sympy.Eq(FloorDiv(value, 16, evaluate=False), expected)
        library, predicate, _ = self.compile_predicate(printer, sources, expression)
        self.assertEqual(self.evaluate(predicate, (-17, -2)), 1)
        self.assertEqual(self.evaluate(predicate, (-17, -1)), 0)
        for symbol, source in zip((value, expected), sources, strict=True):
            self.assertIs(printer.bound_sources[symbol], source)
        self.assertIsNotNone(library)

    @parametrize("arity", (1, 2, 3))
    def test_unsigned_gcd_uses_integer_values_and_unsigned_conversion(self, arity):
        symbols, sources, printer = self.printer(((0, UINT64_MAX),) * (arity + 1))
        expression = UIntGCD(*symbols[:arity])
        library, predicate, printed = self.compile_predicate(
            printer, sources, sympy.Eq(expression, symbols[-1]), unsigned=tuple(range(arity + 1)))
        self.assertNotIn(printed, ("true", "false"))
        samples = ((0, 0, 0), (48, 80, 112), (1 << 40, (1 << 40) + 16, 0),
                   (1 << 40, 1 << 40, 1 << 40), (-1, 32, 48), (-16, 48, 96), (-(1 << 63), 0, 16))
        for sample in samples:
            values = sample[:arity]
            expected = gcd(*(value % (1 << 64) for value in values))
            self.assertEqual(expression.subs(dict(zip(symbols[:arity], values, strict=True))), expected)
            self.assertEqual(self.evaluate(predicate, (*values, expected)), 1)
            self.assertEqual(self.evaluate(predicate, (*values, (expected + 1) % (1 << 64))), 0)
        self.assertIsNotNone(library)

    def test_grouped_stride_bound_uses_gcd_not_each_contributor(self):
        (left, right), sources, printer = self.printer(((0, I64_MAX),) * 2, _AddressTerminalPrinter)
        stride = UIntGCD(left, right)
        expression = stride < (1 << 39)
        library, predicate, printed = self.compile_predicate(printer, sources, expression)
        self.assertNotIn(printed, ("true", "false"))
        for values in ((1 << 39, (1 << 39) + 8), (1 << 39, 1 << 39),
                       (0, (1 << 39) - 8), (0, 1 << 39), (0, 0)):
            self.assertEqual(self.evaluate(predicate, values), int(gcd(*values) < (1 << 39)))
        self.assertNotIn(expression, printer.discharged)
        self.assertIsNotNone(library)

    def test_unsigned_gcd_converts_derived_negative_operands(self):
        (left, right), sources, printer = self.printer(((-I64_MAX, I64_MAX), (0, I64_MAX)))
        expression = sympy.Eq(UIntGCD(left - 16, right), 48)
        library, predicate, _ = self.compile_predicate(printer, sources, expression)
        self.assertEqual(self.evaluate(predicate, (0, 48)), 1)
        self.assertEqual(self.evaluate(predicate, (32, 48)), 0)
        self.assertIsNotNone(library)

    def test_unsigned_gcd_checks_arithmetic_before_conversion(self):
        (value,), _, printer = self.printer(((0, I64_MAX),))
        with self.assertRaisesRegex(GuardExportDeclined, "overflow|fit signed"):
            printer.doprint(UIntGCD(value * (1 << 126), 16))

    @parametrize("operand", (sympy.Rational(1, 2), sympy.Float(1.0), sympy.true))
    def test_unsigned_gcd_rejects_noninteger_operands(self, operand):
        with self.assertRaisesRegex(ValueError, "integer operands"):
            UIntGCD(operand)


if __name__ == "__main__":
    run_tests()
