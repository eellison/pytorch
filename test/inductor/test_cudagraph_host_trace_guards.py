# Owner(s): ["module: inductor"]
"""Ordered native host-trace guards retain metadata and address dependencies."""

import ctypes
from types import SimpleNamespace
from unittest import mock

import sympy

import torch
from torch._dynamo.source import LocalSource
from torch._inductor.codecache import CppCodeCache
from torch._inductor.runtime._cudagraph.host_trace_guards import (
    compile_host_trace_guard,
)
from torch._inductor.runtime.cudagraph_arg_mapping import BufferSource
from torch._inductor.runtime.cudagraph_host_trace_mapping import HostTraceSymbolMapping
from torch._inductor.runtime.cudagraph_launch_association import UnsupportedCapture
from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)
from torch.utils._sympy.functions import (
    FloatPow,
    FloatTrueDiv,
    FloorDiv,
    Identity,
    Max,
    Min,
    Mod,
    PythonMod,
    ToFloat,
)


def tensor_metadata(tensor, kind, dimension=None):
    if kind == "size":
        return tensor.size(dimension)
    if kind == "stride":
        return tensor.stride(dimension)
    if kind == "dtype":
        return {torch.float32: 6, torch.float64: 7, torch.int64: 4, torch.complex64: 9}[
            tensor.dtype
        ]
    return {
        "device": -1 if tensor.device.type == "cpu" else tensor.device.index,
        "rank": tensor.dim(),
        "neg": int(tensor.is_neg()),
        "conj": int(tensor.is_conj()),
        "layout": 0,
        "pinned": int(tensor.is_pinned()),
    }[kind]


@instantiate_parametrized_tests
class TestHostTraceGuards(TestCase):
    def setUp(self):
        super().setUp()
        self.environment = ShapeEnv(duck_shape=False, specialize_zero_one=False)
        self.tensor = torch.empty(9)
        self.record = SimpleNamespace(
            position=0,
            name="arg0",
            dtype=self.tensor.dtype,
            device=self.tensor.device,
            pinned=False,
            sizes=[self.symbol("size", 9)],
            strides=[self.symbol("stride", 1)],
            offset=self.symbol("offset", 0),
            root=SimpleNamespace(
                name="p0",
                itemsize=4,
                sym=self.symbol("base", self.tensor.untyped_storage().data_ptr()),
            ),
        )
        self.tape = SimpleNamespace(
            shape_env=self.environment,
            inputs=[self.record],
            allocs=[],
            nargs=1,
            guards=[],
            opaque=[],
            device=torch.device("cuda", 0),
        )
        metadata = mock.patch.object(
            torch._C, "_cuda_boxed_tensor_metadata", tensor_metadata, create=True
        )
        metadata.start()
        self.addCleanup(metadata.stop)

    def symbol(self, name, hint):
        source = LocalSource(name)
        expression = self.environment.create_unspecified_symbol(
            hint, source, DimDynamic.DYNAMIC
        )
        return self.environment.create_symintnode(expression, hint=hint, source=source)

    def compile(self, extra=()):
        mapping = HostTraceSymbolMapping(self.tape)
        return compile_host_trace_guard(self.tape, mapping, [self.tensor], extra)

    def evaluate(self, guard, tensor=None, overrides=None):
        tensor = self.tensor if tensor is None else tensor
        overrides = {} if overrides is None else overrides
        values = [
            overrides.get(("pointer", index, None), tensor.data_ptr())
            for index in guard.boxed_pointer_indices
        ]
        values.extend(
            overrides.get(("storage_offset", index, None), tensor.storage_offset())
            for index in guard.boxed_storage_offset_indices
        )
        values.extend(
            overrides.get(binding, tensor_metadata(tensor, binding[0], binding[2]))
            for binding in guard.metadata_bindings
        )
        bits = (ctypes.c_uint64 * len(values))(*(value % (1 << 64) for value in values))
        predicate = ctypes.CFUNCTYPE(
            ctypes.c_int8,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_double),
        )(guard.function_address)
        return predicate(ctypes.cast(bits, ctypes.POINTER(ctypes.c_int64)), None)

    def opaque(self, name, args, expected, body, *, kind="guard"):
        source = (
            """#include <cstdint>
#include <stdexcept>
#include <vector>
extern "C" int64_t callback(const std::vector<int64_t>& args) {
"""
            + body
            + "\n}\n"
        )
        library = CppCodeCache.load(source)
        symbol = self.symbol(name, expected)
        record = {
            "sym": symbol,
            "args": args,
            "expected": expected,
            "kind": kind,
            "seq": len(self.tape.opaque),
            "fn": name,
            "impl": ctypes.cast(library.callback, ctypes.c_void_p).value,
            "call": library.callback,
        }
        self.tape.opaque.append(record)
        return symbol

    @parametrize("size,expected", ((8, 1), (9, 1), (15, 1), (16, 0)))
    def test_opaque_selector_recomputes_choice_on_each_call(self, size, expected):
        choice = self.opaque("choice", [self.record.sizes[0]], 1, "return args[0] / 8;")
        self.assertTrue(choice == 1)
        self.tape.guards = [guard.expr for guard in self.environment.guards]
        guard = self.compile()
        self.assertEqual(self.evaluate(guard, torch.empty(size)), expected)
        self.assertIn(choice.node._expr, guard.mapping.opaque_symbols)

    def test_rebind_is_evaluated_before_its_guard_and_next_call(self):
        divisor = self.opaque(
            "divisor", [self.record.sizes[0]], 2, "return args[0] - 7;", kind="rebind"
        )
        original = divisor.node._expr
        self.tape.guards = [sympy.Gt(original, 0)]
        self.opaque("quotient", [FloorDiv(10, original)], 5, "return args[0];")
        guard = self.compile()
        self.assertEqual(self.evaluate(guard), 1)
        self.assertEqual(self.evaluate(guard, torch.empty(7)), 0)
        self.assertEqual(self.evaluate(guard, torch.empty(6)), 0)
        self.assertEqual(self.evaluate(guard, torch.empty(8)), 0)

    def test_callback_exception_is_a_miss_before_replay(self):
        self.opaque(
            "choice",
            [self.record.sizes[0]],
            1,
            'if (args[0] < 8) throw std::runtime_error("size"); return 1;',
        )
        guard = self.compile()
        self.assertEqual(self.evaluate(guard), 1)
        self.assertEqual(self.evaluate(guard, torch.empty(7)), 0)

    def test_opaque_argument_overflow_misses_before_narrowing(self):
        size = self.record.sizes[0].node._expr
        self.opaque("choice", [size * 2], 18, "return args[0];")
        guard = self.compile()
        self.assertEqual(self.evaluate(guard), 1)
        self.assertEqual(self.evaluate(guard, overrides={("size", 0, 0): 1 << 62}), 0)

    def test_large_product_matches_unbounded_integer_reference(self):
        size = self.record.sizes[0].node._expr
        stride = self.record.strides[0].node._expr
        offset = self.record.offset.node._expr
        product = size * stride * (offset + 1)
        self.tape.guards = [sympy.Eq(PythonMod(product, 97), PythonMod(size, 97))]
        guard = self.compile()
        results = set()
        for shape, step, start in (
            (9, 1, 0),
            (9, 2, 0),
            (1 << 40, 1 << 40, (1 << 40) - 1),
            ((1 << 63) - 1, 1, 0),
            ((1 << 63) - 1, (1 << 63) - 1, 0),
        ):
            expected = int((shape * step * (start + 1)) % 97 == shape % 97)
            results.add(expected)
            self.assertEqual(
                self.evaluate(
                    guard,
                    overrides={
                        ("size", 0, 0): shape,
                        ("stride", 0, 0): step,
                        ("storage_offset", 0, None): start,
                    },
                ),
                expected,
            )
        self.assertEqual(results, {0, 1})

    @parametrize("operation", ("add", "mul"))
    def test_intermediate_overflow_cannot_hide_behind_cancellation(self, operation):
        size = self.record.sizes[0].node._expr
        if operation == "add":
            high = 1 << 126
            expression = sympy.Add(
                high, high - 10, size, -high, 10 - high, evaluate=False
            )
            overflow_size = 10
        else:
            positive = sympy.Mul(size, 1 << 120, evaluate=False)
            negative = sympy.Mul(size, -(1 << 120), evaluate=False)
            expression = sympy.Add(positive, negative, size, evaluate=False)
            overflow_size = 1 << 62
        self.tape.guards = [sympy.Eq(expression, size, evaluate=False)]
        guard = self.compile()
        self.assertEqual(self.evaluate(guard), 1)
        self.assertEqual(sympy.simplify(expression - size), 0)
        self.assertEqual(
            self.evaluate(guard, overrides={("size", 0, 0): overflow_size}), 0
        )

    def test_short_circuit_does_not_evaluate_overflowing_arm(self):
        size = self.record.sizes[0].node._expr
        stride = self.record.strides[0].node._expr
        offset = self.record.offset.node._expr
        product = size * stride * (offset + 1)
        first = sympy.Eq(size, 9)
        predicate = sympy.Or(first, sympy.Eq(product, 18))
        self.assertEqual(predicate.args[0], first)
        self.tape.guards = [predicate]
        guard = self.compile()
        for shape, step, start, expected in (
            (9, (1 << 63) - 1, (1 << 63) - 1, 1),
            (10, (1 << 63) - 1, (1 << 63) - 1, 0),
            (18, 1, 0, 1),
            (10, 1, 0, 0),
        ):
            self.assertEqual(
                self.evaluate(
                    guard,
                    overrides={
                        ("size", 0, 0): shape,
                        ("stride", 0, 0): step,
                        ("storage_offset", 0, None): start,
                    },
                ),
                expected,
            )

    @parametrize("fault", ("int64_narrowing", "int128_overflow"))
    def test_product_overflow_misses_before_opaque_callback(self, fault):
        library = CppCodeCache.load(
            """#include <cstdint>
#include <vector>
static int64_t calls = 0;
extern "C" int64_t count() { return calls; }
extern "C" void reset() { calls = 0; }
extern "C" int64_t callback(const std::vector<int64_t>&) { ++calls; return 1; }
"""
        )
        library.count.restype = ctypes.c_int64
        size = self.record.sizes[0].node._expr
        stride = self.record.strides[0].node._expr
        offset = self.record.offset.node._expr
        self.tape.opaque.append(
            {
                "sym": self.symbol("choice", 1),
                "args": [size * stride * (offset + 1)],
                "expected": 1,
                "kind": "guard",
                "seq": 0,
                "fn": "choice",
                "impl": ctypes.cast(library.callback, ctypes.c_void_p).value,
                "call": library.callback,
            }
        )
        guard = self.compile()
        library.reset()
        self.assertEqual(self.evaluate(guard), 1)
        self.assertEqual(library.count(), 1)
        shape, step, start = (
            (1 << 40, 1 << 40, (1 << 40) - 1)
            if fault == "int64_narrowing"
            else (1 << 62, 1 << 62, 15)
        )
        self.assertEqual(
            self.evaluate(
                guard,
                overrides={
                    ("size", 0, 0): shape,
                    ("stride", 0, 0): step,
                    ("storage_offset", 0, None): start,
                },
            ),
            0,
        )
        self.assertEqual(library.count(), 1)

    @parametrize("fault", ("unknown_op", "noninteger"))
    def test_checked_product_does_not_admit_other_invalid_arithmetic(self, fault):
        size = self.record.sizes[0].node._expr
        stride = self.record.strides[0].node._expr
        offset = self.record.offset.node._expr
        product = size * stride * (offset + 1)
        invalid = {
            "unknown_op": sympy.Abs(product, evaluate=False),
            "noninteger": product * sympy.Float(0.5),
        }[fault]
        self.tape.guards = [sympy.Eq(invalid, 9)]
        with self.assertRaisesRegex(UnsupportedCapture, "checked native arithmetic"):
            self.compile()

    @parametrize("operation", ("floordiv", "pythonmod", "sympymod"))
    def test_signed_division_matches_python(self, operation):
        size = self.record.sizes[0].node._expr
        stride = self.record.strides[0].node._expr
        constructor = {
            "floordiv": FloorDiv,
            "pythonmod": PythonMod,
            "sympymod": sympy.Mod,
        }[operation]
        expression = constructor(size - 10, stride - 3, evaluate=False)
        reference = 0 if operation == "floordiv" else -1
        self.tape.guards = [sympy.Eq(expression, reference)]
        guard = self.compile()
        for numerator in (-7, -1, 0, 1, 7):
            for denominator in (-3, -2, -1, 0, 1, 2, 3, 11):
                with self.subTest(numerator=numerator, denominator=denominator):
                    expected = 0
                    if denominator:
                        result = (
                            numerator // denominator
                            if operation == "floordiv"
                            else numerator % denominator
                        )
                        expected = int(result == reference)
                    self.assertEqual(
                        self.evaluate(
                            guard,
                            overrides={
                                ("size", 0, 0): numerator + 10,
                                ("stride", 0, 0): denominator + 3,
                            },
                        ),
                        expected,
                    )

    def test_nonnegative_mod_preserves_its_domain(self):
        size = self.record.sizes[0].node._expr
        stride = self.record.strides[0].node._expr
        self.tape.guards = [sympy.Eq(Mod(stride - 1, size - 7, evaluate=False), 0)]
        guard = self.compile()
        for numerator in (-1, 0, 1, 6):
            for denominator in (-3, -1, 0, 1, 2, 3):
                with self.subTest(numerator=numerator, denominator=denominator):
                    expected = int(
                        numerator >= 0
                        and denominator > 0
                        and numerator % denominator == 0
                    )
                    self.assertEqual(
                        self.evaluate(
                            guard,
                            overrides={
                                ("size", 0, 0): denominator + 7,
                                ("stride", 0, 0): numerator + 1,
                            },
                        ),
                        expected,
                    )

    @parametrize("operation", (FloorDiv, PythonMod, sympy.Mod))
    def test_signed_minimum_division_boundary(self, operation):
        size = self.record.sizes[0].node._expr
        stride = self.record.strides[0].node._expr
        numerator = sympy.Mul(-(1 << 65), stride + 1, evaluate=False)
        expression = operation(numerator, size - 10, evaluate=False)
        relation = sympy.Ge if operation is FloorDiv else sympy.Eq
        self.tape.guards = [relation(expression, 0, evaluate=False)]
        guard = self.compile()
        self.assertEqual(self.evaluate(guard), 1)
        self.assertEqual(
            self.evaluate(guard, overrides={("stride", 0, 0): (1 << 62) - 1}),
            int(operation is not FloorDiv),
        )
        self.assertEqual(self.evaluate(guard, overrides={("size", 0, 0): 10}), 0)

    @parametrize("selection", ("or", "piecewise"))
    def test_unselected_division_by_zero_is_not_evaluated(self, selection):
        size = self.record.sizes[0].node._expr
        stride = self.record.strides[0].node._expr
        condition = sympy.Eq(size, 9)
        quotient = FloorDiv(12, stride - 1)
        if selection == "or":
            expression = sympy.Or(condition, sympy.Eq(quotient, 6))
            self.assertEqual(expression.args[0], condition)
        else:
            expression = sympy.Eq(sympy.Piecewise((6, condition), (quotient, True)), 6)
        self.tape.guards = [expression]
        guard = self.compile()
        self.assertEqual(self.evaluate(guard), 1)
        self.assertEqual(self.evaluate(guard, overrides={("size", 0, 0): 10}), 0)
        self.assertEqual(
            self.evaluate(guard, overrides={("size", 0, 0): 10, ("stride", 0, 0): 3}), 1
        )

    @parametrize("operation", (Min, Max, sympy.Min, sympy.Max))
    def test_selection_does_not_hide_child_overflow(self, operation):
        size = self.record.sizes[0].node._expr
        stride = self.record.strides[0].node._expr
        product = sympy.Mul(size, 1 << 120, evaluate=False)
        expression = operation(product, stride, evaluate=False)
        reference = expression.subs({size: 9, stride: 1})
        self.tape.guards = [sympy.Eq(expression, reference, evaluate=False)]
        guard = self.compile()
        self.assertEqual(self.evaluate(guard), 1)
        self.assertEqual(self.evaluate(guard, overrides={("size", 0, 0): 1 << 62}), 0)

    @parametrize("operation", (FloorDiv, PythonMod, sympy.Mod))
    def test_constant_result_does_not_hide_zero_divisor(self, operation):
        stride = self.record.strides[0].node._expr
        expression = operation(0, stride - 2, evaluate=False)
        self.tape.guards = [sympy.Eq(expression, 0, evaluate=False)]
        guard = self.compile()
        self.assertEqual(self.evaluate(guard), 1)
        self.assertEqual(self.evaluate(guard, overrides={("stride", 0, 0): 2}), 0)
        self.assertEqual(self.evaluate(guard, overrides={("stride", 0, 0): 3}), 1)

    def test_mathematical_proof_can_eliminate_large_integer_computation(self):
        size = self.record.sizes[0].node._expr
        product = sympy.Mul(size, 1 << 120, evaluate=False)
        expression = Min(product, sympy.Integer(0), evaluate=False)
        self.tape.guards = [sympy.Eq(expression, 0, evaluate=False)]
        guard = self.compile()
        self.assertEqual(self.evaluate(guard, overrides={("size", 0, 0): 1 << 62}), 1)

    @parametrize("operation", (FloorDiv, PythonMod, sympy.Mod, Mod))
    def test_nonnegative_divisor_proof_keeps_defined_domain(self, operation):
        size = self.record.sizes[0].node._expr
        stride = self.record.strides[0].node._expr
        numerator = 12 if operation is FloorDiv else stride - 1
        divisor = size - 8
        expression = operation(numerator, divisor, evaluate=False)
        self.tape.guards = [
            sympy.Ge(divisor, 0, evaluate=False),
            sympy.Ge(expression, 0, evaluate=False),
        ]
        guard = self.compile()
        self.assertEqual(self.evaluate(guard), 1)
        self.assertEqual(self.evaluate(guard, overrides={("size", 0, 0): 8}), 0)
        self.assertEqual(self.evaluate(guard, overrides={("size", 0, 0): 10}), 1)
        self.assertEqual(
            self.evaluate(guard, overrides={("stride", 0, 0): 0}),
            int(operation is not Mod),
        )

    @parametrize("selection", ("or", "piecewise", "and_not"))
    def test_defined_domain_obligation_follows_lazy_branch(self, selection):
        size = self.record.sizes[0].node._expr
        stride = self.record.strides[0].node._expr
        divisor = size - 8
        quotient = FloorDiv(12, divisor)
        if selection == "or":
            expression = sympy.Or(
                sympy.Eq(size, 8), sympy.Ge(quotient, 0, evaluate=False)
            )
        elif selection == "piecewise":
            expression = sympy.Ge(
                sympy.Piecewise((quotient, sympy.Eq(size, 9)), (0, True)),
                0,
                evaluate=False,
            )
        else:
            condition = sympy.And(
                sympy.Eq(stride, 2), sympy.Ge(quotient, 0, evaluate=False)
            )
            self.assertEqual(condition.args[0], sympy.Eq(stride, 2))
            expression = sympy.Not(condition, evaluate=False)
        self.tape.guards = [sympy.Ge(divisor, 0), expression]
        guard = self.compile()
        self.assertEqual(self.evaluate(guard), 1)
        self.assertEqual(self.evaluate(guard, overrides={("size", 0, 0): 8}), 1)
        self.assertEqual(self.evaluate(guard, overrides={("size", 0, 0): 10}), 1)
        if selection == "and_not":
            self.assertEqual(
                self.evaluate(
                    guard, overrides={("size", 0, 0): 8, ("stride", 0, 0): 2}
                ),
                0,
            )

    def test_piecewise_condition_keeps_its_defined_domain(self):
        size = self.record.sizes[0].node._expr
        divisor = size - 8
        condition = sympy.Gt(FloorDiv(12, divisor), 1)
        expression = sympy.Piecewise((0, condition), (1, True))
        self.tape.guards = [
            sympy.Ge(divisor, 0, evaluate=False),
            sympy.Ge(expression, 0, evaluate=False),
        ]
        guard = self.compile()
        self.assertEqual(self.evaluate(guard), 1)
        self.assertEqual(self.evaluate(guard, overrides={("size", 0, 0): 8}), 0)

    def test_defined_domain_allows_eliminating_overflowing_numerator(self):
        size = self.record.sizes[0].node._expr
        numerator = sympy.Mul(size, 1 << 120, evaluate=False)
        divisor = size - 8
        expression = FloorDiv(numerator, divisor, evaluate=False)
        self.tape.guards = [
            sympy.Ge(divisor, 0, evaluate=False),
            sympy.Ge(expression, 0, evaluate=False),
        ]
        guard = self.compile()
        self.assertEqual(self.evaluate(guard), 1)
        self.assertEqual(self.evaluate(guard, overrides={("size", 0, 0): 8}), 0)
        self.assertEqual(self.evaluate(guard, overrides={("size", 0, 0): 1 << 62}), 1)

    @parametrize("source", ("input", "owned"))
    @parametrize("positive_offset", (False, True))
    def test_live_address_guard_translation_preserves_defined_domain(
        self, source, positive_offset
    ):
        if source == "owned":
            quotient = self.symbol("owned_quotient", 2)
            self.tape.allocs = [
                SimpleNamespace(
                    name="alloc0",
                    q=quotient,
                    dtype=torch.float32,
                    root=SimpleNamespace(name="a0", sym=256 * quotient, itemsize=4),
                )
            ]
            address = 256 * quotient
        else:
            address = self.record.root.sym + 4 * self.record.offset
        self.assertTrue(12 // (address // 509 + int(positive_offset)) >= 0)
        self.tape.guards = [guard.expr for guard in self.environment.guards]
        self.assertTrue(any(guard.has(FloorDiv) for guard in self.tape.guards))
        if source == "owned" and not positive_offset:
            with self.assertRaisesRegex(UnsupportedCapture, "domain is not proven"):
                self.compile()
            return
        guard = self.compile()
        self.assertEqual(self.evaluate(guard, overrides={("pointer", 0, None): 512}), 1)
        self.assertEqual(
            self.evaluate(guard, overrides={("pointer", 0, None): 256}),
            int(positive_offset),
        )

    def test_translated_piecewise_retains_conditional_domain(self):
        size = self.record.sizes[0].node._expr
        address = self.record.root.sym.node._expr + 4 * self.record.offset.node._expr
        value = FloorDiv(12, FloorDiv(address, 509))
        expression = sympy.Piecewise((value, sympy.Eq(size, 9)), (0, True))
        self.tape.guards = [sympy.Ge(expression, 0, evaluate=False)]
        guard = self.compile()
        self.assertEqual(self.evaluate(guard, overrides={("pointer", 0, None): 512}), 1)
        self.assertEqual(self.evaluate(guard, overrides={("pointer", 0, None): 256}), 0)
        self.assertEqual(
            self.evaluate(
                guard, overrides={("pointer", 0, None): 256, ("size", 0, 0): 10}
            ),
            1,
        )

    @parametrize("selection", ("or", "and_not"))
    def test_translated_boolean_retains_short_circuit_order(self, selection):
        size = self.record.sizes[0].node._expr
        address = self.record.root.sym.node._expr + 4 * self.record.offset.node._expr
        condition = sympy.Eq(size, 10)
        value = sympy.Ge(FloorDiv(12, FloorDiv(address, 509)), 0, evaluate=False)
        expression = (
            sympy.Or(condition, value)
            if selection == "or"
            else sympy.Not(sympy.And(condition, value), evaluate=False)
        )
        self.tape.guards = [expression]
        guard = self.compile()
        self.assertEqual(self.evaluate(guard), 1)
        self.assertEqual(
            self.evaluate(guard, overrides={("pointer", 0, None): 256}),
            int(selection == "and_not"),
        )
        self.assertEqual(
            self.evaluate(
                guard, overrides={("pointer", 0, None): 256, ("size", 0, 0): 10}
            ),
            int(selection == "or"),
        )

    def test_runtime_checked_divisor_can_be_wider_than_int64(self):
        size = self.record.sizes[0].node._expr
        stride = self.record.strides[0].node._expr
        offset = self.record.offset.node._expr
        divisor = size * stride * (offset + 1) + 1
        self.tape.guards = [sympy.Eq(FloorDiv(size, divisor), 0)]
        guard = self.compile()
        self.assertEqual(self.evaluate(guard), 1)
        self.assertEqual(
            self.evaluate(
                guard,
                overrides={
                    ("size", 0, 0): 1 << 40,
                    ("stride", 0, 0): 1 << 40,
                    ("storage_offset", 0, None): (1 << 40) - 1,
                },
            ),
            1,
        )
        self.assertEqual(
            self.evaluate(
                guard,
                overrides={
                    ("size", 0, 0): 1 << 62,
                    ("stride", 0, 0): 1 << 62,
                    ("storage_offset", 0, None): 15,
                },
            ),
            0,
        )

    def test_nested_opaque_reduction_divisor_is_runtime_checked(self):
        first = self.opaque(
            "first", [self.record.sizes[0]], 9, "return args[0];", kind="rebind"
        )
        second = self.opaque(
            "second", [self.record.strides[0]], 1, "return args[0];", kind="rebind"
        )
        expression = FloorDiv(
            512, Min(512, second.node._expr, FloorDiv(512, Min(32, first.node._expr)))
        )
        self.tape.guards = [sympy.Eq(expression, 512)]
        guard = self.compile()
        self.assertEqual(self.evaluate(guard), 1)
        self.assertEqual(self.evaluate(guard, overrides={("stride", 0, 0): 0}), 0)
        self.assertEqual(self.evaluate(guard, overrides={("stride", 0, 0): 2}), 0)
        self.assertEqual(self.evaluate(guard, overrides={("size", 0, 0): 33}), 1)

    def test_opaque_forward_reference_declines(self):
        first = self.opaque("first", [self.record.sizes[0]], 9, "return args[0];")
        second = self.opaque("second", [first], 9, "return args[0];")
        self.tape.opaque[0]["args"] = [second]
        with self.assertRaisesRegex(UnsupportedCapture, "unbound symbolic source"):
            self.compile()

    def test_original_specialization_and_native_metadata_registration(self):
        size = self.record.sizes[0]
        original = size.node._expr
        self.assertTrue(size == 9)
        self.tape.guards = [guard.expr for guard in self.environment.guards]
        guard = self.compile()
        self.assertEqual(len(guard.registration), 6)
        self.assertEqual(guard.registration[0], ())
        self.assertEqual(guard.registration[3:5], ((0,), (0,)))
        self.assertIn(("size", 0, 0), guard.metadata_bindings)
        self.assertIn(original, guard.mapping.metadata_symbols)
        self.assertEqual(size.node.expr, 9)
        self.assertEqual(self.evaluate(guard, torch.empty(9)), 1)
        self.assertEqual(self.evaluate(guard, torch.empty(10)), 0)
        self.assertEqual(self.evaluate(guard, torch.empty(0)), 0)

    def test_original_range_and_replacement_guards_are_not_dropped(self):
        size = self.record.sizes[0].node._expr
        self.environment.constrain_symbol_range(size, 3, 12)
        guard = self.compile()
        self.assertEqual(self.evaluate(guard, torch.empty(3)), 1)
        self.assertEqual(self.evaluate(guard, torch.empty(12)), 1)
        self.assertEqual(self.evaluate(guard, torch.empty(2)), 0)
        self.assertEqual(self.evaluate(guard, torch.empty(13)), 0)
        self.environment.replacements[size] = sympy.Integer(9)
        guard = self.compile()
        self.assertEqual(self.evaluate(guard, torch.empty(9)), 1)
        self.assertEqual(self.evaluate(guard, torch.empty(8)), 0)

    def test_zero_divisor_misses_before_dependent_arithmetic(self):
        size = self.record.sizes[0].node._expr
        divisor = size - 7
        self.tape.guards = [sympy.Gt(divisor, 0), sympy.Eq(FloorDiv(10, divisor), 5)]
        guard = self.compile()
        self.assertEqual(self.evaluate(guard, torch.empty(9)), 1)
        self.assertEqual(self.evaluate(guard, torch.empty(7)), 0)
        self.assertEqual(self.evaluate(guard, torch.empty(6)), 0)
        self.assertEqual(self.evaluate(guard, torch.empty(8)), 0)

    def test_division_without_an_earlier_domain_guard_is_checked(self):
        divisor = self.record.sizes[0].node._expr - 7
        self.tape.guards = [sympy.Eq(FloorDiv(10, divisor), 5)]
        guard = self.compile()
        self.assertEqual(self.evaluate(guard, torch.empty(9)), 1)
        self.assertEqual(self.evaluate(guard, torch.empty(7)), 0)
        self.assertEqual(self.evaluate(guard, torch.empty(6)), 0)
        self.assertEqual(self.evaluate(guard, torch.empty(8)), 0)

    def test_ordered_product_bounds_protect_later_intermediates(self):
        size = self.record.sizes[0].node._expr
        stride = self.record.strides[0].node._expr
        offset = self.record.offset.node._expr
        product = size * stride
        self.tape.guards = [
            sympy.Le(product, (1 << 63) - 1),
            sympy.Le(product * offset, (1 << 63) - 1),
        ]
        guard = self.compile()
        self.assertEqual(self.evaluate(guard), 1)
        self.assertEqual(
            self.evaluate(
                guard,
                overrides={
                    ("size", 0, 0): 1 << 62,
                    ("stride", 0, 0): 8,
                    ("storage_offset", 0, None): 1 << 62,
                },
            ),
            0,
        )

    def test_extra_guards_use_translated_unsigned_addresses(self):
        mapping = HostTraceSymbolMapping(self.tape)
        (address,) = mapping.address_symbols
        extra = (sympy.Eq(sympy.Mod(address, 16), 0),)
        guard = compile_host_trace_guard(self.tape, mapping, [self.tensor], extra)
        self.assertEqual(
            self.evaluate(guard, overrides={("pointer", 0, None): (1 << 63) + 32}), 1
        )
        self.assertEqual(
            self.evaluate(guard, overrides={("pointer", 0, None): (1 << 63) + 36}), 0
        )
        self.assertIn("const uint64_t pointer_0", guard.cpp_source)

    def test_input_storage_base_guard_accounts_for_current_offset(self):
        base = self.record.root.sym.node._expr
        self.tape.guards = [sympy.Eq(sympy.Mod(base, 16), 0)]
        guard = self.compile()
        overrides = {
            ("pointer", 0, None): (1 << 45) + 4,
            ("storage_offset", 0, None): 1,
        }
        self.assertEqual(self.evaluate(guard, overrides=overrides), 1)
        overrides[("storage_offset", 0, None)] = 2
        self.assertEqual(self.evaluate(guard, overrides=overrides), 0)

    @parametrize("property", ("dtype", "device", "rank", "neg", "conj", "layout"))
    def test_tensor_contract_is_guarded(self, property):
        guard = self.compile()
        value = tensor_metadata(self.tensor, property)
        self.assertEqual(self.evaluate(guard), 1)
        self.assertEqual(
            self.evaluate(guard, overrides={(property, 0, 0): value + 1}), 0
        )

    @parametrize("mathbit", ("negative", "conjugate"))
    def test_mathbit_examples_decline_during_preparation(self, mathbit):
        if mathbit == "negative":
            self.tensor = torch._neg_view(self.tensor)
        else:
            self.tensor = torch.empty(9, dtype=torch.complex64).conj()
            self.record.dtype = self.tensor.dtype
            self.record.root.itemsize = self.tensor.element_size()
        with self.assertRaisesRegex(
            UnsupportedCapture, "plain strided Tensor contract"
        ):
            self.compile()

    @parametrize("property", ("size", "stride", "storage_offset"))
    def test_metadata_domain_checks_precede_symbolic_arithmetic(self, property):
        guard = self.compile()
        dimension = None if property == "storage_offset" else 0
        self.assertEqual(
            self.evaluate(guard, overrides={(property, 0, dimension): -1}), 0
        )

    def test_owned_address_guards_require_allocator_proof(self):
        quotient = self.symbol("owned_quotient", 0)
        self.tape.allocs = [
            SimpleNamespace(
                name="alloc0",
                q=quotient,
                dtype=torch.float32,
                root=SimpleNamespace(name="a0", sym=256 * quotient, itemsize=4),
            )
        ]
        mapping = HostTraceSymbolMapping(self.tape)
        owned = next(
            symbol
            for symbol, source in mapping.address_symbols.items()
            if type(source) is BufferSource
        )
        guard = compile_host_trace_guard(
            self.tape,
            mapping,
            [self.tensor],
            (sympy.Eq(sympy.Mod(owned, 256), 0, evaluate=False),),
        )
        self.assertEqual(self.evaluate(guard), 1)
        self.assertEqual(guard.boxed_pointer_indices, (0,))
        with self.assertRaisesRegex(UnsupportedCapture, "not proven before allocation"):
            compile_host_trace_guard(
                self.tape, mapping, [self.tensor], (sympy.Gt(owned, 4096),)
            )
        with self.assertRaises(UnsupportedCapture):
            compile_host_trace_guard(
                self.tape,
                mapping,
                [self.tensor],
                (
                    sympy.Ge(
                        FloorDiv(
                            12, PythonMod(owned, 512, evaluate=False), evaluate=False
                        ),
                        0,
                        evaluate=False,
                    ),
                ),
            )
        defined = sympy.Ge(FloorDiv(12, PythonMod(owned, 512) + 1), 0, evaluate=False)
        guard = compile_host_trace_guard(self.tape, mapping, [self.tensor], (defined,))
        self.assertEqual(self.evaluate(guard), 1)

    def test_typed_float_attention_scale_uses_recorded_guard(self):
        dimension = self.record.sizes[0]
        scale = 1.0 / (torch.sym_float(dimension) ** 0.5)
        self.assertEqual(float(scale), 1.0 / 3.0)
        self.tape.guards = [guard.expr for guard in self.environment.guards]
        self.assertTrue(any(guard.has(FloatTrueDiv) for guard in self.tape.guards))
        guard = self.compile()
        self.assertIn("guard_float_power", guard.cpp_source)
        for size in (8, 9, 10, 63, 64):
            self.assertEqual(
                self.evaluate(guard, torch.empty(size)),
                int(1.0 / (float(size) ** 0.5) == 1.0 / 3.0),
            )

    def test_typed_float_integer_conversion_preserves_binary64_rounding(self):
        stride = self.record.strides[0].node._expr
        boundary = 1 << 53
        self.tape.guards = [
            sympy.Ge(stride, 1),
            sympy.Eq(
                ToFloat(stride + boundary), sympy.Float(float(boundary)), evaluate=False
            ),
        ]
        guard = self.compile()
        for value in (1, 2, 3, 4):
            self.assertEqual(
                self.evaluate(guard, overrides={("stride", 0, 0): value}),
                int(float(value + boundary) == float(boundary)),
            )

    @parametrize("operation", ("divide", "power"))
    def test_typed_float_lazy_boolean_skips_invalid_branch(self, operation):
        stride = self.record.strides[0].node._expr
        if operation == "divide":
            value = FloatTrueDiv(sympy.Float(1), ToFloat(stride - 1))
        else:
            value = FloatPow(ToFloat(stride - 1), sympy.Float(-1))
        condition = sympy.Eq(value, sympy.Float(1), evaluate=False)
        self.tape.guards = [sympy.Or(sympy.Eq(stride, 1), condition, evaluate=False)]
        guard = self.compile()
        for step, expected in ((1, 1), (2, 1), (3, 0)):
            self.assertEqual(
                self.evaluate(guard, overrides={("stride", 0, 0): step}), expected
            )

    def test_typed_float_lazy_piecewise_skips_invalid_condition(self):
        stride = self.record.strides[0].node._expr
        value = FloatTrueDiv(sympy.Float(1), ToFloat(stride - 1))
        selected = sympy.Piecewise(
            (1, sympy.Eq(stride, 1)),
            (2, sympy.Eq(value, sympy.Float(1), evaluate=False)),
            (3, sympy.true),
            evaluate=False,
        )
        self.tape.guards = [sympy.Ge(selected, 1, evaluate=False)]
        guard = self.compile()
        for step in (1, 2, 3):
            self.assertEqual(
                self.evaluate(guard, overrides={("stride", 0, 0): step}), 1
            )

    @parametrize("operation", ("divide", "sqrt", "inverse", "overflow"))
    def test_typed_float_invalid_domain_is_a_miss(self, operation):
        size = self.record.sizes[0].node._expr
        operand = ToFloat(size - 8)
        if operation == "divide":
            value = FloatTrueDiv(sympy.Float(1), operand)
            invalid = 8
        else:
            exponent, invalid = {
                "sqrt": (0.5, 7),
                "inverse": (-1.0, 8),
                "overflow": (1024.0, 10),
            }[operation]
            value = FloatPow(operand, sympy.Float(exponent))
        self.tape.guards = [sympy.Eq(value, sympy.Float(1), evaluate=False)]
        guard = self.compile()
        self.assertEqual(self.evaluate(guard), 1)
        self.assertEqual(self.evaluate(guard, torch.empty(invalid)), 0)

    def test_typed_float_mixed_integer_comparison_declines(self):
        stride = self.record.strides[0].node._expr
        integer = stride + (1 << 53)
        self.tape.guards = [sympy.Eq(ToFloat(integer), integer, evaluate=False)]
        with self.assertRaisesRegex(UnsupportedCapture, "typed operands"):
            self.compile()

    def test_typed_float_owned_address_requires_preallocation_proof(self):
        quotient = self.symbol("owned_quotient", 1)
        self.tape.allocs = [
            SimpleNamespace(
                name="alloc0",
                q=quotient,
                dtype=torch.float32,
                root=SimpleNamespace(name="a0", sym=256 * quotient, itemsize=4),
            )
        ]
        self.tape.guards = [
            sympy.Eq(
                ToFloat(256 * quotient.node._expr), sympy.Float(256), evaluate=False
            )
        ]
        with self.assertRaisesRegex(UnsupportedCapture, "not proven before allocation"):
            self.compile()

    @parametrize("identity", (False, True))
    def test_dynamic_float_predicates_decline(self, identity):
        value = sympy.Float(0.5) * self.record.sizes[0].node._expr
        if identity:
            value = Identity(value)
        self.tape.guards = [sympy.Gt(value, 0)]
        with self.assertRaisesRegex(UnsupportedCapture, "checked native arithmetic"):
            self.compile()


if __name__ == "__main__":
    run_tests()
