# Owner(s): ["module: inductor"]

import ctypes
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

import sympy

import torch
from torch._dynamo.source import LocalSource
from torch._inductor.runtime._cudagraph._compiler.fx_adapter.contract import (
    InputContract,
    IntegerRange,
    TensorInput,
)
from torch._inductor.runtime._cudagraph.cuda_tape_import import TapeSources
from torch._inductor.runtime._cudagraph.direct_cuda_host import (
    CudaInvocation,
    DirectCudaHost,
)
from torch._inductor.runtime._cudagraph.direct_host import (
    _direct_origin,
    _TracedInvocations,
)
from torch._inductor.runtime._cudagraph.direct_hosttrace import (
    _bind_capture_rng,
    _RngSlot,
)
from torch._inductor.runtime._cudagraph.direct_invocation import activate
from torch._inductor.runtime._cudagraph.extraction import trace_host
from torch._inductor.runtime._cudagraph.frontend import (
    DirectPhysicalCall,
    lower_terminal,
)
from torch._inductor.runtime._cudagraph.guard_export import prepare_guard
from torch._inductor.runtime.cudagraph_arg_mapping import (
    InputSource,
    IntExpr,
    PointerSource,
)
from torch._inductor.runtime.cudagraph_boxed_replay import (
    _NumericProgram,
    _PhysicalCall,
    _PhysicalField,
)
from torch._inductor.runtime.cudagraph_compiled_evaluation import (
    compile_numeric,
    EarlyStatus,
)
from torch._inductor.runtime.cudagraph_host_trace_mapping import HostTraceSymbolMapping
from torch._inductor.runtime.cudagraph_launch_association import UnsupportedCapture
from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


ZERO = IntExpr("constant", 0)
ONE = IntExpr("constant", 1)
ADAPTER = EMPTY = None


def ordinary(x):
    raise AssertionError("No ordinary calls during symbolic extraction")


def host(box):
    _, _, x, y = box
    box.clear()
    return ADAPTER(x), ADAPTER(y)


def host_with_empty(box):
    _, _, x, y = box
    box.clear()
    first = ADAPTER(x)
    middle = EMPTY(x)
    return first, middle, ADAPTER(y)


def symbol(env, name, hint):
    source = LocalSource(name)
    value = env.create_unspecified_symbol(hint, source, DimDynamic.DYNAMIC)
    return env.create_symintnode(value, hint=hint, source=source)


def owner(adapter, launches=1, fields=1):
    env = ShapeEnv(duck_shape=False, specialize_zero_one=False)
    n = symbol(env, "n", 8)
    root = SimpleNamespace(name="p0", itemsize=4, sym=symbol(env, "address", 4096))
    inp = SimpleNamespace(
        position=0,
        dtype=torch.float32,
        sizes=[n],
        strides=[symbol(env, "stride", 1)],
        offset=symbol(env, "offset", 0),
        root=root,
    )
    tape = SimpleNamespace(
        nargs=1,
        shape_env=env,
        inputs=[inp],
        allocs=[],
        launches=[{"seq": i} for i in range(launches)],
        outputs=[
            SimpleNamespace(
                root=root,
                sizes=[n],
                strides=inp.strides,
                offset=inp.offset,
                dtype=torch.float32,
            )
        ],
        guards=[sympy.Ge(n.node._expr, 7, evaluate=False)] if not launches else [],
        opaque=[],
        constants=(),
        host_buffers=[],
        memcpys=[],
        memsets=[],
        rng_increment=4 * n * launches if launches else None,
        rng_slots=[
            {"launch": i, "offset": 8 + 40 * j + 16, "size": 8, "increment": 4 * n}
            for i in range(launches)
            for j in range(fields)
        ],
    )
    module = SimpleNamespace(check=lambda: None, parameter_sizes=(8, 40 * fields))
    calls = tuple(
        _PhysicalCall(
            (_PhysicalField(0, 0, "pointer", PointerSource(InputSource(0), ZERO)),),
            module,
            (ONE, ONE, ONE),
            (),
            constants=((1, 0, bytes(range(40 * fields))),),
        )
        for _ in range(launches)
    )
    lowered = SimpleNamespace(
        tape=tape,
        symbols=SimpleNamespace(mapping=HostTraceSymbolMapping(tape)),
        calls=calls,
        memsets=(),
        host_tables=(),
        memcpys=(),
        extra_guards=(),
        rng=ONE if launches else None,
        rng_fields=tuple(
            (i, 1, 40 * j) for i in range(launches) for j in range(fields)
        ),
        rng_slots=tuple(
            _RngSlot(i, 1, 40 * j + 16, 8, 4 * n * i)
            for i in range(launches)
            for j in range(fields)
        ),
    )
    return CudaInvocation(adapter, tape, lowered, torch.Tensor, (("argument", 0),))


@instantiate_parametrized_tests
class TestMixedRng(TestCase):
    def program(self, launches=1, fields=1, empty=False):
        adapter = DirectCudaHost(ordinary)
        first = owner(adapter, launches, fields)
        second = CudaInvocation(
            adapter, first.tape, first.lowered, torch.Tensor, (("argument", 0),)
        )
        empty_adapter = DirectCudaHost(ordinary)
        empty_owner = owner(empty_adapter, 0)
        self.enterContext(
            mock.patch.dict(globals(), ADAPTER=adapter, EMPTY=empty_adapter)
        )
        n, m = IntExpr("boxed", 0), IntExpr("boxed", 1)
        contract = InputContract(
            ("integer", "integer", "tensor", "tensor"),
            (
                TensorInput(2, torch.float32, (n,), (1,)),
                TensorInput(3, torch.float32, (m,), (1,)),
            ),
            (IntegerRange(0, 4, 2**62), IntegerRange(1, 4, 2**62)),
            0,
        )
        body = host_with_empty if empty else host
        origin, _ = _direct_origin(body, contract)
        observations = (first, empty_owner, second) if empty else (first, second)
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            trace = trace_host(
                body,
                contract,
                [8, 8, torch.empty(8), torch.empty(8)],
                (),
                None,
                direct=True,
                context_factory=lambda state: activate(
                    _TracedInvocations(state, observations)
                ),
            )
        program = lower_terminal(replace(trace, compiler_binding=origin), ())
        return program, first

    @parametrize("launches", (1, 2))
    @parametrize("fields", (1, 2))
    def test_repeated_invocations_compose_prefix_once(self, launches, fields):
        program, first = self.program(launches, fields)
        calls = [event for event in program.events if type(event) is DirectPhysicalCall]
        self.assertEqual(len(calls), 2 * launches)
        self.assertIsNot(calls[0].owner, calls[-1].owner)
        numeric = _NumericProgram(program, [17, 23, torch.empty(17), torch.empty(23)])
        self.assertEqual(
            numeric.values[numeric.add(program.rng)], 4 * (17 + 23) * launches
        )
        for index, event in enumerate(calls):
            expected = (
                4 * 17 * index
                if index < launches
                else 4 * 17 * launches + 4 * 23 * (index - launches)
            )
            scalars = [field for field in event.bound.fields if field.kind == "i64"]
            self.assertEqual(len(scalars), fields)
            self.assertEqual(
                [
                    numeric.values[numeric.add(field.source.expression)]
                    for field in scalars
                ],
                [expected] * fields,
            )
            for field in scalars:
                for parameter, offset, data in event.bound.constants:
                    self.assertFalse(
                        parameter == field.parameter
                        and offset < field.byte_offset + 8
                        and field.byte_offset < offset + len(data)
                    )
        self.assertEqual(
            first.lowered.calls[0].constants, ((1, 0, bytes(range(40 * fields))),)
        )

    def test_zero_launch_completion_preserves_guard_and_prefix(self):
        program, _ = self.program(empty=True)
        calls = [event for event in program.events if type(event) is DirectPhysicalCall]
        self.assertEqual(len(calls), 2)
        numeric = _NumericProgram(program, [8, 8, torch.empty(8), torch.empty(8)])
        self.assertEqual(numeric.values[numeric.add(program.rng)], 64)
        guard = prepare_guard(program, [8, 8, torch.empty(8), torch.empty(8)])
        predicate = ctypes.CFUNCTYPE(
            ctypes.c_int8,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_double),
        )(guard.function_address)
        for rows, expected in ((8, True), (6, False), (10, True)):
            values = {0: rows, 1: 8}
            arguments = (ctypes.c_int64 * len(guard.boxed_integer_indices))(
                *(values[i] for i in guard.boxed_integer_indices)
            )
            self.assertEqual(predicate(arguments, None) == 1, expected)

    @parametrize("width,value", ((4, 2**31 + 4), (8, 2**32 + 4)))
    def test_recorded_width_preserves_full_offset(self, width, value):
        importer = SimpleNamespace(
            translate=sympy.sympify, _lower=lambda x: IntExpr("constant", int(x))
        )
        call = _PhysicalCall(
            (),
            SimpleNamespace(),
            (ONE, ONE, ONE),
            (),
            constants=((0, 0, bytes(range(40))),),
        )
        bound, guards = TapeSources.rng_call(
            importer, call, (_RngSlot(0, 0, 16, width, 4),), sympy.Integer(value - 4)
        )
        numeric = _NumericProgram(
            SimpleNamespace(input_names=(), integer_inputs=()), ()
        )
        actual = numeric.values[numeric.add(bound.fields[0].source.expression)]
        self.assertEqual(actual & ((1 << (width * 8)) - 1), value)
        self.assertTrue(all(bool(guard) for guard in guards))
        self.assertEqual(
            bound.constants,
            ((0, 0, bytes(range(16))), (0, 16 + width, bytes(range(16 + width, 40)))),
        )

    @parametrize("width,value", ((4, -4), (4, 2**32), (8, -4), (8, 2**63)))
    def test_prefix_range_is_a_reuse_obligation(self, width, value):
        importer = SimpleNamespace(
            translate=sympy.sympify, _lower=lambda x: IntExpr("constant", int(x))
        )
        call = _PhysicalCall(
            (), SimpleNamespace(), (ONE, ONE, ONE), (), constants=((0, 0, bytes(40)),)
        )
        _, guards = TapeSources.rng_call(
            importer, call, (_RngSlot(0, 0, 16, width, 0),), sympy.Integer(value)
        )
        self.assertFalse(all(bool(guard) for guard in guards))

    def test_capture_bindings_are_preparation_local(self):
        original = bytes(range(80))
        call = _PhysicalCall(
            (), SimpleNamespace(), (ONE, ONE, ONE), (), constants=((0, 0, original),)
        )
        prepared = []
        for philox in ((4096, 8192, 0), (12288, 16384, 0)):
            images = [bytearray(original)]
            bound = _bind_capture_rng(call, images, ((0, 0), (0, 40)), philox)
            expected = bytearray(original)
            payload = philox[0].to_bytes(8, "little") + philox[1].to_bytes(8, "little")
            expected[:16] = payload
            expected[40:56] = payload
            self.assertEqual(images, [expected])
            self.assertEqual(bound.constants, ((0, 0, bytes(expected)),))
            self.assertEqual(call.constants, ((0, 0, original),))
            prepared.append(bound)
        self.assertNotEqual(prepared[0].constants, prepared[1].constants)

    def test_aggregate_uses_checked_early_arithmetic(self):
        program, _ = self.program()
        numeric = _NumericProgram(program, [8, 8, torch.empty(8), torch.empty(8)])
        output = numeric.add(program.rng)
        compiled = compile_numeric(numeric)
        status, values = compiled.evaluate_leaves(
            compiled.bind_inputs([17, 23, torch.empty(17), torch.empty(23)])
        )
        self.assertEqual(status, EarlyStatus.SUCCESS)
        self.assertEqual(values[output], 160)
        status, _ = compiled.evaluate_leaves(
            compiled.bind_inputs([2**61, 8, torch.empty(8), torch.empty(8)])
        )
        self.assertNotEqual(status, EarlyStatus.SUCCESS)

    def test_missing_slots_remain_an_explicit_decline(self):
        adapter = DirectCudaHost(ordinary)
        invocation = owner(adapter)
        invocation.tape.rng_slots = []
        with self.assertRaisesRegex(UnsupportedCapture, "recorded RNG slots"):
            TapeSources(invocation.tape, {0: None}, None, None, None)


if __name__ == "__main__":
    run_tests()
