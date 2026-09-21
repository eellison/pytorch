# Owner(s): ["module: inductor"]

from dataclasses import replace
from types import SimpleNamespace

import sympy

import torch
from torch._dynamo.source import LocalSource
from torch._inductor.runtime._cudagraph.direct_hosttrace import lower_tape
from torch._inductor.runtime.cudagraph_arg_mapping import (
    bind_output_slots,
    BorrowedInputOutput,
    InputSource,
    OutputReference,
    TensorViewOutput,
)
from torch._inductor.runtime.cudagraph_launch_association import UnsupportedCapture
from torch.cuda._host_trace import _OutputRec, _TraceShapeEnv
from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


@instantiate_parametrized_tests
class TestHostTraceOutputReference(TestCase):
    def setUp(self):
        super().setUp()
        env = ShapeEnv(duck_shape=False, specialize_zero_one=False)

        def symbol(name, hint):
            source = LocalSource(name)
            expr = env.create_unspecified_symbol(hint, source, DimDynamic.DYNAMIC)
            return env.create_symintnode(expr, hint=hint, source=source)

        n = symbol("n", 8)
        self.input = SimpleNamespace(
            position=1,
            root=SimpleNamespace(name="p1", itemsize=4, sym=symbol("ptr", 4096)),
            sizes=[n],
            strides=[symbol("stride", 1)],
            offset=symbol("offset", 2),
            dtype=torch.float32,
            device=torch.device("cuda", 0),
            pinned=False,
        )
        q = symbol("alloc_q", 32)
        self.allocation = SimpleNamespace(
            name="alloc0",
            seq=0,
            q=q,
            root=SimpleNamespace(name="a0", itemsize=4, sym=256 * q),
            sizes=[n],
            strides=[1],
            dtype=torch.float32,
        )
        self.tape = SimpleNamespace(
            shape_env=env,
            inputs=[self.input],
            allocs=[self.allocation],
            nargs=2,
            guards=[],
            opaque=[],
            launches=[],
            outputs=[],
            memsets=[],
            memcpys=[],
            host_buffers=[],
            rng_increment=None,
        )
        self.input_output = _OutputRec(
            "input",
            self.input.root,
            [n],
            self.input.strides,
            self.input.offset,
            torch.float32,
        )
        self.owned_output = _OutputRec(
            "owned", self.allocation.root, [n], [1], 0, torch.float32
        )

    def test_explicit_references_keep_distinct_whole_views(self):
        self.tape.outputs = [
            self.owned_output,
            replace(self.owned_output, name="view"),
            replace(self.owned_output, identity=("output", 1)),
            replace(self.input_output, identity=("argument", 1)),
            replace(self.input_output, identity=("output", 3)),
        ]
        lowered = lower_tape(self.tape)
        slots = bind_output_slots(
            lowered.outputs, lowered.allocations, ("input",), set(), symbolic=True
        )
        self.assertIsNotNone(slots)
        self.assertEqual(slots[0], 0)
        self.assertIs(type(slots[1]), TensorViewOutput)
        self.assertEqual(slots[2], OutputReference(1))
        self.assertEqual(slots[3], BorrowedInputOutput(InputSource(0)))
        self.assertEqual(slots[4], OutputReference(3))

    def test_missing_metadata_does_not_infer_input_identity(self):
        record = vars(self.input_output).copy()
        del record["identity"]
        self.tape.outputs = [SimpleNamespace(**record)]
        (output,) = lower_tape(self.tape).outputs
        self.assertIs(type(output), TensorViewOutput)
        self.assertEqual(output.source, InputSource(0))

    @parametrize(
        "identity",
        (
            ("output", -1),
            ("output", 0),
            ("output", 1),
            ("output", True),
            ("argument", 0),
            ("argument", True),
            ("unknown", 1),
            ["argument", 1],
        ),
    )
    def test_invalid_identity_is_rejected_before_preparation(self, identity):
        self.tape.outputs = [replace(self.input_output, identity=identity)]
        with self.assertRaisesRegex(
            UnsupportedCapture, "Invalid host trace output identity"
        ):
            lower_tape(self.tape)

    @parametrize("field", ("root", "dtype", "sizes", "strides", "offset"))
    @parametrize("kind", ("argument", "output"))
    def test_reference_requires_the_recorded_layout(self, field, kind):
        changed = {
            "root": self.allocation.root,
            "dtype": torch.int32,
            "sizes": [7],
            "strides": [2],
            "offset": 0,
        }[field]
        index = 1 if kind == "argument" else 0
        output = replace(self.input_output, identity=(kind, index), **{field: changed})
        self.tape.outputs = (
            [output] if kind == "argument" else [self.input_output, output]
        )
        with self.assertRaisesRegex(
            UnsupportedCapture, "identity lost its recorded layout"
        ):
            lower_tape(self.tape)

    @parametrize("pin_in_snapshot", (False, True))
    def test_reference_uses_only_recorded_guard_pins(self, pin_in_snapshot):
        env = _TraceShapeEnv()

        def symbol(name, hint):
            source = LocalSource(name)
            expr = env.create_unspecified_symbol(hint, source, DimDynamic.DYNAMIC)
            return env.create_symintnode(expr, hint=hint, source=source)

        n, m = symbol("n", 8), symbol("m", 8)
        self.input.root.sym = symbol("ptr", 4096)
        self.input.sizes = [n, m]
        self.input.strides = [symbol("stride0", 8), symbol("stride1", 1)]
        self.input.offset = symbol("offset", 0)
        self.tape.shape_env = env
        self.tape.allocs = []
        self.tape.outputs = [
            _OutputRec(
                "input",
                self.input.root,
                [n, n],
                self.input.strides,
                self.input.offset,
                torch.float32,
                ("argument", 1),
            )
        ]
        equality = sympy.Eq(n.node.expr, m.node.expr, evaluate=False)
        if pin_in_snapshot:
            env.evaluate_expr(equality, hint=True)
        self.tape.guards = env.tape_guards()[0]
        if not pin_in_snapshot:
            env.evaluate_expr(equality, hint=True)

        if pin_in_snapshot:
            self.assertEqual(
                lower_tape(self.tape).outputs,
                (BorrowedInputOutput(InputSource(0)),),
            )
        else:
            with self.assertRaisesRegex(
                UnsupportedCapture, "identity lost its recorded layout"
            ):
                lower_tape(self.tape)


if __name__ == "__main__":
    run_tests()
