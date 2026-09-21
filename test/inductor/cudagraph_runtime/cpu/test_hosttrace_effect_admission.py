# Owner(s): ["module: inductor"]

from types import SimpleNamespace

import torch
from torch._dynamo.source import LocalSource
from torch._inductor.runtime._cudagraph.direct_hosttrace import lower_tape
from torch._inductor.runtime.cudagraph_arg_mapping import BufferSource, PointerSource
from torch._inductor.runtime.cudagraph_launch_association import UnsupportedCapture
from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


@instantiate_parametrized_tests
class TestHostTraceEffectAdmission(TestCase):
    def setUp(self):
        super().setUp()
        env = ShapeEnv(duck_shape=False, specialize_zero_one=False)

        def symbol(name, hint):
            source = LocalSource(name)
            expr = env.create_unspecified_symbol(hint, source, DimDynamic.DYNAMIC)
            return env.create_symintnode(expr, hint=hint, source=source)

        n = symbol("n", 8)
        inp = SimpleNamespace(
            position=0,
            root=SimpleNamespace(name="p0", itemsize=4, sym=symbol("ptr", 4096)),
            sizes=[n],
            strides=[symbol("stride", 1)],
            offset=symbol("offset", 2),
            dtype=torch.float32,
            device=torch.device("cpu"),
            pinned=True,
        )
        q = symbol("alloc_q", 32)
        allocation = SimpleNamespace(
            name="alloc0",
            seq=0,
            q=q,
            root=SimpleNamespace(name="a0", itemsize=4, sym=256 * q),
            sizes=[n],
            strides=[1],
            dtype=torch.float32,
        )
        self.copy = {
            "seq": 1,
            "src": inp.root.sym + inp.offset * 4,
            "dst": allocation.root.sym,
            "bytes": n * 4,
        }
        self.tape = SimpleNamespace(
            shape_env=env,
            inputs=[inp],
            allocs=[allocation],
            nargs=1,
            guards=[],
            opaque=[],
            launches=[],
            outputs=[],
            memsets=[],
            memcpys=[self.copy],
            host_buffers=[],
            rng_increment=None,
            device=torch.device("cuda", 0),
        )

    @parametrize("kind", (None, "h2d"))
    def test_existing_h2d_records_remain_admitted(self, kind):
        if kind is not None:
            self.copy["kind"] = kind
        self.tape.regions = []
        lowered = lower_tape(self.tape)
        self.assertEqual(len(lowered.memcpys), 1)
        self.assertEqual(lowered.memcpy_kinds, ("h2d",))

    def test_closed_regions_are_not_silently_omitted(self):
        # a region record the recorder did not write declines by name, never dropped
        self.tape.regions = [SimpleNamespace(seq=2, op="mm")]
        with self.assertRaisesRegex(UnsupportedCapture, "closed regions"):
            lower_tape(self.tape)

    @parametrize("kind", ("d2h", "unknown"))
    def test_unsupported_direction_is_not_reinterpreted_as_h2d(self, kind):
        self.copy["kind"] = kind
        with self.assertRaisesRegex(UnsupportedCapture, "copy kind"):
            lower_tape(self.tape)

    def test_d2d_record_lowers_from_a_device_source(self):
        # a declared device-to-device copy: the source is a pointer over a device
        # input (or an owned allocation), no host table and no pinned position
        self.tape.inputs[0].device = torch.device("cuda", 0)
        self.tape.inputs[0].pinned = False
        self.copy["kind"] = "d2d"
        self.tape.regions = []
        lowered = lower_tape(self.tape)
        (record,) = lowered.memcpys
        self.assertIsInstance(record[1], PointerSource)
        self.assertIsInstance(record[2], PointerSource)
        self.assertIsInstance(record[2].root, BufferSource)
        self.assertEqual(lowered.memcpy_kinds, ("d2d",))
        self.assertEqual(lowered.pinned_positions, ())

    def test_d2d_record_with_a_pinned_source_declines(self):
        # the declared kind is never reinterpreted: a "d2d" record over a host source
        # is a recorder contradiction and declines by name
        self.copy["kind"] = "d2d"
        self.tape.regions = []
        with self.assertRaisesRegex(UnsupportedCapture, "reads a pinned input"):
            lower_tape(self.tape)

    def test_unlabelled_device_source_already_declines(self):
        self.tape.inputs[0].device = torch.device("cuda", 0)
        self.tape.inputs[0].pinned = False
        self.tape.regions = []
        with self.assertRaisesRegex(
            UnsupportedCapture, "neither a host table nor a pinned input"
        ):
            lower_tape(self.tape)


if __name__ == "__main__":
    run_tests()
