# Owner(s): ["module: inductor"]

import ctypes
from contextlib import nullcontext
from dataclasses import replace
from unittest import mock

import sympy

import torch
from torch._dynamo.source import LocalSource
from torch._inductor.codecache import CppCodeCache
from torch._inductor.runtime._cudagraph._compiler.fx_adapter.contract import (
    IntegerRange,
)
from torch._inductor.runtime._cudagraph.api import InputContract, TensorInput
from torch._inductor.runtime._cudagraph.direct_host import _direct_origin
from torch._inductor.runtime._cudagraph.extraction import (
    ComputedIntegerBinding,
    trace_host,
)
from torch._inductor.runtime._cudagraph.frontend import lower_terminal
from torch._inductor.runtime._cudagraph.guard_export import (
    export_guards,
    GuardExportDeclined,
    prepare_guard,
)
from torch._inductor.runtime.cudagraph_arg_mapping import BufferSource, IntExpr
from torch.fx.experimental.symbolic_shapes import DimDynamic
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)
from torch.utils._sympy.functions import FloorDiv


def host(box):
    n, tensor = box
    box.clear()
    output = torch.empty_strided((n,), (1,), dtype=tensor.dtype, device=tensor.device)
    output.data_ptr()
    return (output,)


def extract():
    n = IntExpr("boxed", 0)
    contract = InputContract(
        ("integer", "tensor"),
        (TensorInput(1, torch.float32, (n,), (1,)),),
        (IntegerRange(0, 1, 2**63 - 1),),
        device_index=0,
    )
    origin, _ = _direct_origin(host, contract)
    with mock.patch.object(torch.cuda, "is_available", return_value=False):
        trace = trace_host(
            host,
            contract,
            [8, torch.empty(8)],
            (),
            None,
            direct=True,
            context_factory=lambda state: nullcontext(),
        )
    return replace(trace, compiler_binding=origin)


@instantiate_parametrized_tests
class TestComputedIntegerGuards(TestCase):
    def setUp(self):
        super().setUp()
        self.trace = extract()
        self.n = self.trace.integer_placeholders[0].node._expr
        self.libraries = []

    def callback(self, arguments, expected, body="return args[0];", *, kind="rebind"):
        library = CppCodeCache.load(
            """#include <cstdint>
#include <stdexcept>
#include <vector>
static int64_t calls = 0;
extern "C" int64_t count() { return calls; }
extern "C" void reset() { calls = 0; }
extern "C" int64_t callback(const std::vector<int64_t>& args) {
  ++calls;
"""
            + body
            + "\n}\n"
        )
        library.count.restype = ctypes.c_int64
        library.reset()
        self.libraries.append(library)
        environment = self.trace.shape_env
        source = LocalSource(f"computed_{len(self.trace.computed_integer_bindings)}")
        symbol = environment.create_unspecified_symbol(
            expected, source, DimDynamic.DYNAMIC
        )
        environment.constrain_symbol_range(symbol, -(2**63), 2**63 - 1)
        value = environment.create_symintnode(symbol, hint=expected, source=source)
        binding = ComputedIntegerBinding(
            value,
            source,
            symbol,
            tuple(arguments),
            ctypes.cast(library.callback, ctypes.c_void_p).value,
            library,
            kind,
            expected,
        )
        self.trace = replace(
            self.trace,
            computed_integer_bindings=(*self.trace.computed_integer_bindings, binding),
        )
        return symbol, library

    def compile(self, *predicates):
        program = lower_terminal(self.trace, (), extra_guards=tuple(predicates))
        self.addCleanup(program.close)
        guard = prepare_guard(program, [8, torch.empty(8)])
        self.assertIsNotNone(guard)
        for library in self.libraries:
            library.reset()
        return guard

    def evaluate(self, guard, n):
        self.assertEqual(guard.boxed_pointer_indices, ())
        self.assertEqual(guard.boxed_storage_offset_indices, ())
        self.assertTrue(all(index == 0 for index in guard.boxed_integer_indices))
        values = (ctypes.c_int64 * len(guard.boxed_integer_indices))(
            *[n for _ in guard.boxed_integer_indices]
        )
        predicate = ctypes.CFUNCTYPE(
            ctypes.c_int8,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_double),
        )(guard.function_address)
        return predicate(values, None)

    def test_chained_rebinds_use_runtime_values_and_short_circuit(self):
        divisor, first = self.callback((self.n,), 2, "return args[0] - 6;")
        quotient, second = self.callback(
            (FloorDiv(10, divisor, evaluate=False),), 5, "return args[0];", kind="guard"
        )
        guard = self.compile(sympy.Gt(divisor, 0, evaluate=False))
        self.assertEqual(self.evaluate(guard, 8), 1)
        self.assertEqual(self.evaluate(guard, 9), 0)
        self.assertEqual(self.evaluate(guard, 6), 0)
        self.assertEqual(self.evaluate(guard, 5), 0)
        self.assertEqual(first.count(), 4)
        self.assertEqual(second.count(), 2)
        self.assertNotEqual(divisor, quotient)
        self.assertIn(first, guard.library_owner)
        self.assertIn(second, guard.library_owner)

    def test_equal_hints_have_independent_callback_symbols(self):
        first, _ = self.callback((self.n,), 8, "return args[0];")
        second, _ = self.callback((self.n,), 8, "return 16 - args[0];")
        self.assertNotEqual(first, second)
        guard = self.compile(sympy.Eq(first, second, evaluate=False))
        self.assertEqual(self.evaluate(guard, 8), 1)
        self.assertEqual(self.evaluate(guard, 7), 0)
        self.assertEqual(self.evaluate(guard, 9), 0)

    def test_selector_miss_skips_later_callback(self):
        first, _ = self.callback((self.n,), 1, "return args[0] / 8;", kind="guard")
        second, library = self.callback((first,), 1, "return args[0];")
        guard = self.compile(sympy.Eq(second, 1, evaluate=False))
        self.assertEqual(self.evaluate(guard, 16), 0)
        self.assertEqual(library.count(), 0)
        self.assertEqual(self.evaluate(guard, 9), 1)
        self.assertEqual(library.count(), 1)

    @parametrize("fault", ("zero_divisor", "narrowing", "overflow", "exception"))
    def test_invalid_callback_argument_misses_before_call(self, fault):
        argument, body, valid, invalid = self.n, "return 1;", 8, 7
        if fault == "zero_divisor":
            argument = FloorDiv(12, self.n - 7, evaluate=False)
        elif fault == "narrowing":
            argument, invalid = self.n * 2, 2**63 - 1
        elif fault == "overflow":
            argument, invalid = self.n**3, 2**63 - 1
        else:
            body = 'if (args[0] == 7) throw std::runtime_error("argument"); return 1;'
        _, library = self.callback((argument,), 1, body, kind="guard")
        guard = self.compile()
        self.assertEqual(self.evaluate(guard, valid), 1)
        self.assertEqual(self.evaluate(guard, invalid), 0)
        self.assertEqual(library.count(), 2 if fault == "exception" else 1)

    def test_forward_callback_reference_declines(self):
        first, _ = self.callback((self.n,), 8)
        second, _ = self.callback((self.n,), 8)
        bindings = list(self.trace.computed_integer_bindings)
        bindings[0] = replace(bindings[0], arguments=(second,))
        self.trace = replace(self.trace, computed_integer_bindings=tuple(bindings))
        with self.assertRaisesRegex(GuardExportDeclined, "before allocation"):
            export_guards(self.trace)

    def test_owned_address_callback_argument_declines(self):
        address = next(
            binding.symbol
            for binding in self.trace.address_bindings
            if type(binding.root) is BufferSource
        )
        self.callback((address,), 4096)
        with self.assertRaisesRegex(GuardExportDeclined, "before allocation"):
            export_guards(self.trace)

    def test_callback_replacement_is_checked_without_hint_substitution(self):
        result, _ = self.callback((self.n,), 8, "return 16 - args[0];")
        self.trace.shape_env.replacements[result] = self.n
        guard = self.compile()
        self.assertEqual(self.evaluate(guard, 8), 1)
        self.assertEqual(self.evaluate(guard, 7), 0)
        self.assertEqual(self.evaluate(guard, 9), 0)

    def test_callback_range_guards_are_retained(self):
        result, _ = self.callback((self.n,), 8)
        self.trace.shape_env.constrain_symbol_range(result, 4, 10)
        guard = self.compile()
        self.assertEqual(self.evaluate(guard, 3), 0)
        self.assertEqual(self.evaluate(guard, 8), 1)
        self.assertEqual(self.evaluate(guard, 11), 0)


if __name__ == "__main__":
    run_tests()
