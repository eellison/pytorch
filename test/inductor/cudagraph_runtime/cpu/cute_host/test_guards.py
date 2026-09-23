# Owner(s): ["module: inductor"]
"""Consumed compiler comparisons and selects retain compiled reuse predicates."""

import ctypes
from functools import partial

from torch._inductor.runtime._cudagraph._sdk import activate


activate()

import sympy
from cutlass._mlir import ir
from numeric_test_utils import numeric_transports, transport_numeric

from torch._dynamo.source import LocalSource
from torch._inductor.codecache import CppCodeCache
from torch._inductor.runtime._cudagraph._compiler.cfg_values import read_cfg_function
from torch._inductor.runtime._cudagraph._compiler.cute_bridge.numeric import (
    lower_numeric,
    NumericSource,
)
from torch._inductor.runtime._cudagraph._compiler.decoded_values import (
    prepare_decodings,
)
from torch._inductor.runtime._cudagraph._compiler.overflow_properties import (
    bind_properties,
)
from torch._inductor.runtime._cudagraph._compiler.owned_numeric import (
    evaluate_owned,
    freeze_numeric,
)
from torch._inductor.runtime._cudagraph._compiler.values import scalar
from torch._inductor.runtime._cudagraph.address_guard_printer import GuardExportDeclined
from torch._inductor.runtime._cudagraph.cute_adapter import _condition, _symbolic
from torch._inductor.runtime._cudagraph.guard_export import _TerminalPrinter
from torch._inductor.runtime.cudagraph_arg_mapping import IntExpr
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)
from torch.utils._sympy.value_ranges import ValueRanges


@instantiate_parametrized_tests
class TestNumericReuseGuard(TestCase):
    def printer(self):
        symbols = sympy.symbols("left right", integer=True)
        sources = tuple(LocalSource(name) for name in ("left", "right"))
        mapping = {
            symbol: [source] for symbol, source in zip(symbols, sources, strict=True)
        }
        printer = _TerminalPrinter(
            mapping,
            lambda source: source.name,
            mapping,
            sources=sources,
            source_domains={id(source): ValueRanges(2, 9) for source in sources},
            symbols=frozenset(symbols),
        )
        return symbols, sources, printer

    @numeric_transports
    @parametrize("inputs", ((2, 5), (5, 2), (8, 7), (5, 8)))
    def test_consumed_comparison_select_compiles(self, transport, inputs):
        source = """module { llvm.func @probe(%left: i32, %right: i32) -> i1 {
          %two = llvm.mlir.constant(2 : i32) : i32
          %four = llvm.mlir.constant(4 : i32) : i32
          %false = llvm.mlir.constant(false) : i1
          %less = llvm.icmp "slt" %left, %right : i32
          %enabled = llvm.icmp "ne" %less, %false : i1
          %shifted = llvm.add %right, %two : i32
          %selected = llvm.select %enabled, %left, %shifted : i1, i32
          %guard = llvm.icmp "sle" %selected, %four : i32
          llvm.return %guard : i1
        } }"""
        with ir.Context(), ir.Location.unknown(), ir.raw_values():
            module = ir.Module.parse(source)
            self.assertTrue(module.operation.verify())
            cfg = read_cfg_function(module, "probe")
            numeric = freeze_numeric(
                bind_properties(cfg), prepare_decodings(cfg), (0, 1)
            )
        numeric = transport_numeric(numeric, transport)
        lowered = lower_numeric(
            numeric, lambda index, path: NumericSource(IntExpr("boxed", index), 2, 9)
        )
        self.assertEqual(lowered.obligations, ())
        symbols, sources, printer = self.printer()
        expression = _condition(
            lowered.values[0], partial(_symbolic, symbols=dict(enumerate(symbols)))
        )
        self.assertTrue(expression.has(sympy.Piecewise))
        printed = printer.doprint(expression)
        self.assertNotIn(printed, ("true", "false"))
        self.assertIn("?", printed)
        declarations = "\n".join(
            f"  const int64_t {printer.source_to_symbol[source].name} = values[{index}];"
            for index, source in enumerate(sources)
        )
        cpp = f"""#include <cstdint>
extern "C" int8_t guard(int64_t* values, double*) {{
{declarations}
  return {printed};
}}
"""
        library = CppCodeCache.load(cpp)
        address = ctypes.cast(library.guard, ctypes.c_void_p).value
        self.assertIsNotNone(address)
        guard = ctypes.CFUNCTYPE(
            ctypes.c_int8,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_double),
        )(address)
        expected = evaluate_owned(
            numeric, tuple(scalar("i32", value) for value in inputs)
        )[0].integer(signed=False)
        self.assertEqual(guard((ctypes.c_int64 * 2)(*inputs), None), expected)
        self.assertEqual(
            expected, int((inputs[0] if inputs[0] < inputs[1] else inputs[1] + 2) <= 4)
        )

    @parametrize("fault", ("value", "condition", "missing_else"))
    def test_all_arms_and_conditions_keep_arithmetic_checks(self, fault):
        (left, _), _, printer = self.printer()
        overflow = 2**62 * left
        if fault == "value":
            expression = sympy.Piecewise(
                (overflow, left < 2), (left, True), evaluate=False
            )
        elif fault == "condition":
            expression = sympy.Piecewise(
                (left, sympy.Lt(overflow, 0, evaluate=False)),
                (left + 1, True),
                evaluate=False,
            )
        else:
            expression = sympy.Piecewise((left, left < 5), evaluate=False)
        message = (
            "unconditional final arm" if fault == "missing_else" else "fit signed int64"
        )
        with self.assertRaisesRegex(GuardExportDeclined, message):
            printer.doprint(expression)


if __name__ == "__main__":
    run_tests()
