"""Compile the exact unsigned TMA dimension recurrence through the guard printer."""

import ctypes

import sympy
from torch._dynamo.source import LocalSource
from torch._inductor.codecache import CppCodeCache
from torch._inductor.runtime._cudagraph.address_guard_printer import _AddressPrinter, GuardExportDeclined
from torch._inductor.runtime._cudagraph._compiler.tma_dimension import TmaDimension, tma_dimension
from torch.testing._internal.common_utils import instantiate_parametrized_tests, parametrize, run_tests, TestCase
from torch.utils._sympy.value_ranges import ValueRanges


UINT64_MAX = (1 << 64) - 1


@instantiate_parametrized_tests
class TestTmaDimensionPrinter(TestCase):
    @parametrize("arity", (1, 2, 4, 6))
    def test_compiled_unsigned_recurrence(self, arity):
        symbols = sympy.symbols(f"value0:{arity}", integer=True)
        sources = tuple(LocalSource(f"source{index}") for index in range(arity))
        mapping = {symbol: [source] for symbol, source in zip(symbols, sources, strict=True)}
        printer = _AddressPrinter(mapping, lambda source: source.name, mapping,
                                  domains={symbol: ValueRanges(0, UINT64_MAX) for symbol in symbols}, sources=sources)
        expression = TmaDimension(*symbols)
        self.assertEqual(printer.bounds(expression), ValueRanges(0, UINT64_MAX))
        printed = printer.doprint(expression)
        declarations = "\n".join(
            f"  const uint64_t {printer.source_to_symbol[source].name} = values[{index}];"
            for index, source in enumerate(sources))
        library = CppCodeCache.load("""#include <cstdint>
#include <numeric>
extern "C" uint64_t dimension(const uint64_t* values) {
""" + declarations + f"\n  return {printed};\n}}\n")
        address = ctypes.cast(library.dimension, ctypes.c_void_p).value
        self.assertIsNotNone(address)
        evaluate = ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64))(address)
        samples = {
            1: (((0,), 0), ((1,), 1), ((1 << 32,), 1 << 32),
                (((1 << 32) + 1,), (1 << 32) + 1), ((-1,), UINT64_MAX)),
            2: (((0, 0), 0), ((3, 0), 3), ((3, 4), 3), ((-1, 1), UINT64_MAX)),
            4: (((2, 0, 3, 0), 3), ((3, 0, 2, 0), 2), ((2, 0, 3, 4), 3),
                ((2, 4, 3, 0), 2), ((2, 4, 2, 4), 3), ((3, 8, 4, 12), 14),
                ((UINT64_MAX, 1, 3, 1), 1), ((0, 0, 2, 1), 2), ((0, 1, 1, 0), 0),
                ((1 << 32, 8, 1, 0), 1 << 32)),
            6: (((3, 8, 4, 12, 2, 16), 18), ((2, 4, 2, 4, 2, 4), 4),
                ((0, 0, 5, 0, 3, 0), 3), ((UINT64_MAX, 1, 3, 1, 2, 1), 2)),
        }[arity]
        for values, expected in samples:
            with self.subTest(values=values):
                self.assertEqual(tma_dimension(*values), expected)
                self.assertEqual(expression.subs(dict(zip(symbols, values, strict=True))), expected)
                self.assertEqual(evaluate((ctypes.c_uint64 * arity)(*(value & UINT64_MAX for value in values))), expected)

    @parametrize("values", ((), (1, 2, 3), (sympy.Rational(1, 2),), (sympy.Float(1),), (sympy.true,)))
    def test_rejects_invalid_arguments(self, values):
        with self.assertRaisesRegex(ValueError, "integer shape/stride pairs"):
            TmaDimension(*values)

    def test_checks_source_arithmetic_before_unsigned_conversion(self):
        value = sympy.Symbol("value", integer=True)
        source = LocalSource("source")
        mapping = {value: [source]}
        printer = _AddressPrinter(mapping, lambda source: source.name, mapping,
                                  domains={value: ValueRanges(0, (1 << 63) - 1)}, sources=(source,))
        with self.assertRaisesRegex(GuardExportDeclined, "overflow|fit signed"):
            printer.doprint(TmaDimension(value * (1 << 126), 16))


if __name__ == "__main__":
    run_tests()
