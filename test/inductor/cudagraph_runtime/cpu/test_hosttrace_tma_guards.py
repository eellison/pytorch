# Owner(s): ["module: inductor"]

import ctypes
from types import SimpleNamespace

import sympy

import torch
from torch._dynamo.source import LocalSource
from torch._inductor.codecache import CppCodeCache
from torch._inductor.runtime._cudagraph._compiler.tma_dimension import TmaDimension
from torch._inductor.runtime._cudagraph.address_guard_printer import UIntGCD
from torch._inductor.runtime._cudagraph.direct_hosttrace import (
    _Lowering,
    _PREDICATE_PREAMBLE,
    HostTraceLoweringDeclined,
    lower_tape,
)
from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)
from torch.utils._sympy.functions import FloorDiv


@instantiate_parametrized_tests
class TestHostTraceTmaGuards(TestCase):
    def compile(self, expression, symbols):
        lowering = _Lowering(SimpleNamespace(opaque={}))
        names = {symbol: f"value{index}" for index, symbol in enumerate(symbols)}
        printed = lowering.predicate_cpp(expression, names)
        source = "\n".join(
            (
                *_PREDICATE_PREAMBLE,
                'extern "C" int8_t evaluate(const int64_t* values) {',
                "  bool bad = false;",
                *(
                    f"  const int64_t {name} = values[{index}];"
                    for index, name in enumerate(names.values())
                ),
                f"  return ({printed}) && !bad;",
                "}",
            )
        )
        library = CppCodeCache.load(source)
        function = ctypes.CFUNCTYPE(ctypes.c_int8, ctypes.POINTER(ctypes.c_int64))(
            ctypes.cast(library.evaluate, ctypes.c_void_p).value
        )
        return library, function

    @parametrize("stride", (0, 8, (1 << 38) - 1, 1 << 38, 1 << 62, -1))
    def test_stride_predicate_does_not_narrow_unsigned_product(self, stride):
        value = sympy.Symbol("stride", integer=True)
        expression = sympy.Lt(4 * UIntGCD(value), 1 << 40, evaluate=False)
        _library, predicate = self.compile(expression, (value,))
        expected = int(4 * (stride % (1 << 64)) < 1 << 40)
        self.assertEqual(predicate((ctypes.c_int64 * 1)(stride)), expected)

    @parametrize("dimension", (0, 1, 1 << 32, (1 << 32) + 1, -1))
    def test_dimension_predicate_preserves_unsigned_domain(self, dimension):
        value = sympy.Symbol("dimension", integer=True)
        encoded = TmaDimension(value)
        expression = sympy.And(
            sympy.Ge(encoded, 1, evaluate=False),
            sympy.Le(encoded, 1 << 32, evaluate=False),
            evaluate=False,
        )
        _library, predicate = self.compile(expression, (value,))
        self.assertEqual(
            predicate((ctypes.c_int64 * 1)(dimension)),
            int(1 <= dimension % (1 << 64) <= 1 << 32),
        )

    @parametrize(
        "values,expected",
        (((2, 0, 3, 0), 3), ((3, 0, 2, 0), 2), ((3, 8, 4, 12), 14), ((-1, 1, 3, 1), 1)),
    )
    def test_grouped_dimension_keeps_order_and_uint64_wrap(self, values, expected):
        symbols = sympy.symbols("value0:4", integer=True)
        expression = sympy.Eq(TmaDimension(*symbols), expected, evaluate=False)
        _library, predicate = self.compile(expression, symbols)
        self.assertEqual(predicate((ctypes.c_int64 * 4)(*values)), 1)

    @parametrize("inside_recast", (False, True))
    def test_unproven_int128_intermediate_declines(self, inside_recast):
        symbols = sympy.symbols("value0:3", integer=True)
        if inside_recast:
            expression = UIntGCD(sympy.Mul(*symbols, evaluate=False))
        else:
            expression = sympy.Mul(
                *(UIntGCD(value) for value in symbols), evaluate=False
            )
        predicate = sympy.Lt(expression, 1 << 63, evaluate=False)
        with self.assertRaisesRegex(HostTraceLoweringDeclined, "fit signed|overflow"):
            self.compile(predicate, symbols)

    def test_unproven_divisor_declines_before_compilation(self):
        numerator, divisor = sympy.symbols("numerator divisor", integer=True)
        quotient = FloorDiv(UIntGCD(numerator), divisor, evaluate=False)
        expression = sympy.Eq(quotient, 1, evaluate=False)
        with self.assertRaisesRegex(HostTraceLoweringDeclined, "TMA predicate"):
            self.compile(expression, (numerator, divisor))

    def test_missing_original_slot_is_not_inferred(self):
        value = sympy.Symbol("missing", integer=True)
        with self.assertRaisesRegex(HostTraceLoweringDeclined, "no boxed source"):
            self.compile(sympy.Lt(UIntGCD(value), 1024, evaluate=False), ())

    def test_ordinary_predicate_keeps_int64_overflow_check(self):
        value = sympy.Symbol("value", integer=True)
        expression = sympy.Lt(4 * value, 1 << 40, evaluate=False)
        _library, predicate = self.compile(expression, (value,))
        self.assertEqual(predicate((ctypes.c_int64 * 1)(8)), 1)
        self.assertEqual(predicate((ctypes.c_int64 * 1)(1 << 62)), 0)

    def test_lower_tape_uses_whole_tma_predicate(self):
        environment = ShapeEnv(duck_shape=False, specialize_zero_one=False)

        def symbol(name, hint):
            source = LocalSource(name)
            expression = environment.create_unspecified_symbol(
                hint, source, DimDynamic.DYNAMIC
            )
            return environment.create_symintnode(expression, hint=hint, source=source)

        stride = symbol("stride", 1)
        record = SimpleNamespace(
            position=0,
            root=SimpleNamespace(name="p0", itemsize=4, sym=symbol("base", 4096)),
            sizes=[symbol("size", 8)],
            strides=[stride],
            offset=symbol("offset", 0),
            dtype=torch.float32,
            device=torch.device("cuda", 0),
            pinned=False,
        )
        tape = SimpleNamespace(
            shape_env=environment,
            inputs=[record],
            allocs=[],
            nargs=1,
            guards=[sympy.Lt(4 * UIntGCD(stride.node._expr), 1 << 40, evaluate=False)],
            opaque=[],
            launches=[],
            outputs=[],
            memsets=[],
            memcpys=[],
            host_buffers=[],
            rng_increment=None,
        )
        lowered = lower_tape(tape, device=0)
        self.assertEqual(lowered.pointer_indices, ())
        self.assertEqual(lowered.offset_indices, ())
        tensor = torch.empty(8)
        predicate = ctypes.CFUNCTYPE(
            ctypes.c_int8,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_double),
        )(lowered.predicate_address)
        for step, expected in ((1, 1), (1 << 38, 0), (1 << 62, 0)):
            values = [
                step
                if fact.kind == "stride"
                else torch._C._cuda_boxed_tensor_metadata(tensor, fact.kind, fact.dim)
                for fact in lowered.facts
            ]
            # Only predicate metadata is evaluated; these strides never reach a kernel.
            for index, fact in enumerate(lowered.facts):
                if fact.kind == "device":
                    values[index] = 0
            self.assertEqual(
                predicate((ctypes.c_int64 * len(values))(*values), None), expected
            )


if __name__ == "__main__":
    run_tests()
