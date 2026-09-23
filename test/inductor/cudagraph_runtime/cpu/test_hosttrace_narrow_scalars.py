# Owner(s): ["module: inductor"]

import ctypes
import dataclasses
from types import SimpleNamespace
from unittest import mock

import sympy

import torch
from torch._dynamo.source import LocalSource
from torch._inductor.runtime import cudagraph_compiled_evaluation as evaluation
from torch._inductor.runtime._cudagraph import direct_hosttrace
from torch._inductor.runtime.cudagraph_arg_mapping import (
    ExpressionSource,
    IntegerInput,
    IntegerSource,
    IntExpr,
    ParameterSource,
)
from torch._inductor.runtime.cudagraph_boxed_replay import (
    _bind_physical_call,
    _KernelModule,
    _make_replay,
    _NumericProgram,
    _ParameterProgram,
    _PhysicalCall,
    _PhysicalField,
)
from torch._inductor.runtime.cudagraph_launch_association import (
    CapturedKernelLaunch,
    RecordedKernelLaunch,
    UnsupportedCapture,
)
from torch.cuda._host_trace import _pack
from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


class _Module(_KernelModule):
    function = 17
    parameter_layout = ((0, 8),)
    shared = 0

    def check(self):
        pass

    def _borrow_for_cudagraph(self):
        return self


class _Graph:
    def __init__(self, image):
        self.image = image
        self.fields = None

    def _inspect_captured_kernel_nodes(self, nodes):
        kernel = (40, 17, 0, 0, (1, 1, 1), (32, 1, 1), 0, False, ((0, 8, self.image),))
        return 30, 20, (40,), (kernel,)

    def _prepare_kernel_replay_updates(
        self, pointers, count, fields, grids, value_count, **kwargs
    ):
        self.fields = fields
        return object()

    def _make_boxed_replay(self, *args, **kwargs):
        return self


@instantiate_parametrized_tests
class TestHostTraceNarrowScalars(TestCase):
    def fixture(self, kind, value, source=None):
        width = {"u8": 1, "i16": 2, "i32": 4}[kind]
        image = b"\xa5\x3c" + _pack(kind, value) + b"\x5a" * (6 - width)
        one = IntExpr("constant", 1)
        field = _PhysicalField(
            0, 2, f"i{width * 8}", source or ExpressionSource(IntExpr("boxed", 0))
        )
        call = _PhysicalCall(
            (field,),
            _Module(),
            (one, one, one),
            (),
            ((0, 0, image[:2]), (0, 2 + width, image[2 + width :])),
        )
        graph = _Graph(image)
        launch = RecordedKernelLaunch(
            10, (1, 20, 30, ()), (1, 20, 30, ((40, bytes(8)),)), 17, (image,)
        )
        captured = CapturedKernelLaunch(
            0, graph._inspect_captured_kernel_nodes(())[3][0]
        )
        numeric = _NumericProgram(
            SimpleNamespace(
                input_names=("value",), integer_inputs=(IntegerInput("value", 0),)
            ),
            (value,),
        )
        return call, graph, launch, captured, numeric

    @parametrize("kind", ("u8", "i16"))
    def test_physical_binding_preserves_int64_until_store(self, kind):
        samples = (
            0,
            127,
            128,
            255,
            256,
            32767,
            32768,
            65535,
            65536,
            -1,
            -129,
            -32769,
            -(2**63),
            2**63 - 1,
        )
        for initial in samples:
            call, graph, launch, captured, numeric = self.fixture(kind, initial)
            compiled = []
            compile_evaluation = evaluation.compile_evaluation

            def remember(*args, **kwargs):
                result = compile_evaluation(*args, **kwargs)
                compiled.append(result)
                return result

            with mock.patch.object(
                evaluation, "compile_evaluation", side_effect=remember
            ):
                _make_replay(
                    graph,
                    1,
                    (),
                    (),
                    (),
                    (call,),
                    (launch,),
                    {},
                    None,
                    numeric=numeric,
                    capture_inputs=(initial,),
                )
            width = 1 if kind == "u8" else 2
            self.assertEqual(graph.fields, [(40, 0, 2, width, 0)])
            self.assertEqual(launch.argument_bytes[0][:2], b"\xa5\x3c")
            self.assertEqual(
                launch.argument_bytes[0][2 + width :], b"\x5a" * (6 - width)
            )
            program = compiled[0].early
            for value in samples:
                status, values = program.evaluate_leaves(program.bind_inputs((value,)))
                self.assertEqual(status, evaluation.EarlyStatus.SUCCESS)
                self.assertEqual(values[graph.fields[0][-1]], value)

    @parametrize("kind", ("u8", "i16"))
    @parametrize("damage", ("bytes", "overlap", "extent"))
    def test_packed_field_validation_is_preserved(self, kind, damage):
        call, graph, launch, captured, numeric = self.fixture(kind, -1)
        if damage == "bytes":
            launch = dataclasses.replace(launch, argument_bytes=(bytes(8),))
            message = "differs from its recorded source"
        elif damage == "overlap":
            call = dataclasses.replace(call, fields=call.fields * 2)
            message = "overlap"
        else:
            call = dataclasses.replace(
                call, fields=(dataclasses.replace(call.fields[0], byte_offset=8),)
            )
            message = "exceeds"
        with self.assertRaisesRegex(UnsupportedCapture, message):
            _bind_physical_call(call, launch, captured, 1, {}, {}, numeric)

    @parametrize("kind", ("u8", "i16", "i32"))
    def test_existing_numeric_domain_is_not_narrowed_or_bypassed(self, kind):
        limit = 2**31 if kind == "i32" else 2**63
        for value in (-limit - 1, limit):
            call, graph, launch, captured, numeric = self.fixture(
                kind, 0, IntegerSource(value)
            )
            with self.assertRaisesRegex(
                UnsupportedCapture, "exceeds its selected ABI width"
            ):
                _bind_physical_call(call, launch, captured, 1, {}, {}, numeric)

    @parametrize("kind", ("u8", "i16"))
    def test_typed_late_parameter_scope_is_unchanged(self, kind):
        width = 8 if kind == "u8" else 16
        call, graph, launch, captured, numeric = self.fixture(
            kind, 0, ParameterSource("constant", width, 0)
        )
        with self.assertRaisesRegex(
            UnsupportedCapture, "Late parameter lost its typed expression"
        ):
            _bind_physical_call(
                call,
                launch,
                captured,
                1,
                {},
                {},
                numeric,
                _ParameterProgram(numeric, 1, {}),
                [],
            )

    def tape(self, kind, expression):
        environment = ShapeEnv(duck_shape=False, specialize_zero_one=False)

        def symbol(name, hint):
            source = LocalSource(name)
            value = environment.create_unspecified_symbol(
                hint, source, DimDynamic.DYNAMIC
            )
            return environment.create_symintnode(value, hint=hint, source=source)

        size, stride = symbol("size", 9), symbol("stride", 1)
        value = expression(size, stride)
        width = 1 if kind == "u8" else 2
        record = SimpleNamespace(
            position=0,
            name="arg0",
            dtype=torch.float32,
            device=torch.device("cuda:0"),
            sizes=[size],
            strides=[stride],
            offset=symbol("offset", 0),
            root=SimpleNamespace(name="p0", itemsize=4, sym=symbol("base", 4096)),
        )
        image = b"\xa5\x3c" + _pack(kind, value.node.hint) + b"\x5a" * (6 - width)
        launch = dict(
            param_layout=((0, 8),),
            hint_image=image,
            params=[dict(offset=2, size=width, kind=kind, value=value)],
            block_expr=(32, 1, 1),
            block=(32, 1, 1),
            smem=0,
            grid=(1, 1, 1),
            func=17,
            kernel="narrow",
        )
        return SimpleNamespace(
            shape_env=environment,
            inputs=[record],
            allocs=[],
            opaque=[],
            nargs=1,
            guards=[],
            launches=[launch],
            outputs=[],
            rng_increment=None,
            device=torch.device("cuda:0"),
        )

    def lower(self, tape):
        with mock.patch.object(
            direct_hosttrace, "_HostTraceKernelModule", return_value=_Module()
        ):
            return direct_hosttrace.lower_tape(tape)

    @parametrize("kind", ("u8", "i16"))
    def test_recorded_symbolic_fields_lower_without_signedness_inference(self, kind):
        tape = self.tape(kind, lambda size, stride: size - 65546)
        lowered = self.lower(tape)
        (field,) = lowered.calls[0].fields
        self.assertEqual(
            (field.parameter, field.byte_offset, field.kind),
            (0, 2, "i8" if kind == "u8" else "i16"),
        )
        numeric = _NumericProgram(lowered.records, (torch.empty(9),))
        self.assertEqual(numeric.values[numeric.add(field.source.expression)], -65537)
        self.assertEqual(
            lowered.calls[0].constants,
            (
                (0, 0, b"\xa5\x3c"),
                (0, 3 if kind == "u8" else 4, b"\x5a" * (5 if kind == "u8" else 4)),
            ),
        )

    @parametrize("kind", ("u8", "i16"))
    def test_narrow_store_retains_original_division_domain(self, kind):
        tape = self.tape(kind, lambda size, stride: torch.sym_min(size // stride, 0))
        lowered = self.lower(tape)
        predicate = ctypes.CFUNCTYPE(
            ctypes.c_int8,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_double),
        )(lowered.predicate_address)
        self.assertEqual(lowered.pointer_indices, ())
        self.assertEqual(lowered.offset_indices, ())
        for stride in (1, 0, 2):
            facts = {
                "rank": 1,
                "dtype": direct_hosttrace._dtype_code(torch.float32),
                "device": 0,
                "neg": 0,
                "conj": 0,
                "size": 9,
                "stride": stride,
            }
            values = (ctypes.c_int64 * len(lowered.facts))(
                *(facts[f.kind] for f in lowered.facts)
            )
            self.assertEqual(predicate(values, None), int(stride != 0))
        self.assertTrue(any(guard.has(sympy.Gt) for guard in lowered.extra_guards))

    @parametrize("kind", ("u8", "i16"))
    def test_recorded_width_must_match_kind(self, kind):
        tape = self.tape(kind, lambda size, stride: size)
        tape.launches[0]["params"][0]["size"] = 2 if kind == "u8" else 1
        with self.assertRaisesRegex(
            direct_hosttrace.HostTraceLoweringDeclined, "integer field width differs"
        ):
            self.lower(tape)


if __name__ == "__main__":
    run_tests()
