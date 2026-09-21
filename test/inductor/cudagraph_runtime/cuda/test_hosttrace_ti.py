# Owner(s): ["module: inductor"]
"""Elementwise ops through the traced TensorIterator sibling, lowered into the shared native replay."""

import ctypes
import gc

import sympy

import torch
import torch.nn.functional as F
from torch.testing._internal.common_utils import run_tests, TestCase


def _pair(M, N, dtype=torch.bfloat16, offset=0):
    flat = torch.randn(2 * (M * N + offset), device="cuda").to(dtype)
    x = flat[offset : offset + M * N].view(M, N)
    y = flat[M * N + 2 * offset : 2 * M * N + 2 * offset].view(M, N)
    return x, y


class TestHostTraceTI(TestCase):
    def setUp(self):
        super().setUp()
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        from torch._inductor.runtime._cudagraph import direct_hosttrace

        self.module = direct_hosttrace

    def _replay(self, fn, args):
        gc.collect()
        replay = self.module.HostTraceReplay(fn, args)
        self.addCleanup(replay.close)
        return replay

    def _check(self, replay, args, fn):
        got = replay(*args)
        want = fn(*args)
        self.assertEqual(got.shape, want.shape)
        self.assertEqual(got, want, atol=0, rtol=0)

    def test_add_serves_new_shapes_bitwise(self):
        replay = self._replay(torch.add, _pair(64, 4096))
        self.assertEqual(len(replay.lowered.calls), 1)
        self.assertEqual(len(replay.lowered.allocations), 1)
        for args in (
            _pair(48, 4096),
            _pair(64, 2048),
            _pair(7, 1000),
            _pair(1024, 1024),
            _pair(3, 8),
        ):
            self._check(replay, args, torch.add)
        self.assertEqual(replay.misses, 0)

    def test_a_copy_on_write_input_in_a_written_position_materializes(self):
        # eager's in-place add on a lazy clone materializes it (the mutable accessor)
        # and leaves the storage it shared untouched; the replay does the same before
        # it binds the address (the tape names the positions it wrote), and a lazy
        # clone in a read-only position stays lazy
        def add_(x, y):
            return x.add_(y)

        replay = self._replay(add_, _pair(64, 4096))
        self.assertEqual(replay.lowered.tape.written_inputs, (0,))
        base = torch.randn(64, 4096, device="cuda").to(torch.bfloat16)
        y = torch.randn(64, 4096, device="cuda").to(torch.bfloat16)
        keep = base.clone()
        lazy_x, lazy_y = torch._lazy_clone(base), torch._lazy_clone(y)
        self.assertTrue(torch._C._is_cow_tensor(lazy_x))
        out = replay(lazy_x, lazy_y)
        self.assertEqual(replay.misses, 0)
        self.assertIs(out, lazy_x)
        self.assertFalse(torch._C._is_cow_tensor(lazy_x))
        self.assertTrue(torch._C._is_cow_tensor(lazy_y))
        self.assertEqual(base, keep, atol=0, rtol=0)
        self.assertEqual(lazy_x, keep.clone().add_(y), atol=0, rtol=0)

    def test_lazy_clones_in_read_positions_stay_lazy_off_the_hit_path(self):
        # the hit path reads a read position through the const accessor (A286); the
        # preparation, the miss path and the arena check read the same way (round 8's
        # F5: they used to materialize every lazy clone in the box), and a written
        # position materializes at the preparation as it does at a served call
        def add(x, y):
            return x + y

        def add_(x, y):
            return x.add_(y)

        def lazy_pair(*shape):
            base = torch.randn(*shape, device="cuda").to(torch.bfloat16)
            y = torch.randn(*shape, device="cuda").to(torch.bfloat16)
            return base, y, [torch._lazy_clone(base), torch._lazy_clone(y)]

        cow = torch._C._is_cow_tensor
        base, y, lazy = lazy_pair(64, 4096)
        make = self.module.HostTraceReplay
        replay = make(add, tuple(lazy), warm_up=False, arena_check=True)
        self.addCleanup(replay.close)
        self.assertEqual([cow(t) for t in lazy], [True, True])
        base, y, lazy = lazy_pair(64, 4096)
        self.assertEqual(replay(*lazy), base + y, atol=0, rtol=0)
        self.assertEqual(replay.misses, 0)
        self.assertEqual([cow(t) for t in lazy], [True, True])
        base, y, lazy = lazy_pair(4, 64, 4096)
        self.assertEqual(replay(*lazy), base + y, atol=0, rtol=0)
        self.assertEqual((replay.misses, len(replay.variants)), (1, 2))
        self.assertEqual([cow(t) for t in lazy], [True, True])
        base, y, lazy = lazy_pair(64, 4096)
        keep = base.clone()
        inplace = make(add_, tuple(lazy), warm_up=False)
        self.addCleanup(inplace.close)
        self.assertEqual([cow(t) for t in lazy], [False, True])
        self.assertEqual(base, keep, atol=0, rtol=0)

    def test_written_positions_index_the_box_behind_a_closed_scalar(self):
        # a non-tensor argument ahead of the tensors compacts the box: the written
        # position is the box's, not the argument list's
        def scaled_add_(k, x, y):
            return x.add_(y, alpha=k)

        replay = self._replay(scaled_add_, (2, *_pair(64, 4096)))
        self.assertEqual(replay.lowered.tape.written_inputs, (1,))
        self.assertEqual(replay.lowered.written_positions, (0,))
        base = torch.randn(64, 4096, device="cuda").to(torch.bfloat16)
        y = torch.randn(64, 4096, device="cuda").to(torch.bfloat16)
        keep = base.clone()
        lazy_x, lazy_y = torch._lazy_clone(base), torch._lazy_clone(y)
        out = replay(2, lazy_x, lazy_y)
        self.assertEqual(replay.misses, 0)
        self.assertIs(out, lazy_x)
        self.assertFalse(torch._C._is_cow_tensor(lazy_x))
        self.assertTrue(torch._C._is_cow_tensor(lazy_y))
        self.assertEqual(base, keep, atol=0, rtol=0)
        self.assertEqual(lazy_x, keep.clone().add_(y, alpha=2), atol=0, rtol=0)

    def test_a_power_in_a_payload_is_an_obligation_the_predicate_checks(self):
        # a launch field the host computed as a cube of one size (the numel of the
        # (n, n, n) broadcast product) lowers to a power: its int64 range is an
        # obligation of the lowering, evaluated with checked arithmetic ahead of the
        # plan, so a value the plan's evaluator would overflow on misses instead
        def cube(x):
            return x[:, None, None] * x[None, :, None] * x[None, None, :]

        x = torch.randn(8, device="cuda").to(torch.bfloat16)
        replay = self._replay(cube, (x,))
        lowered = replay.lowered
        powers = [
            g
            for g in lowered.extra_guards
            if isinstance(g, sympy.Le)
            and isinstance(g.lhs, sympy.Pow)
            and g.lhs.exp == 3
            and g.rhs == 2**63 - 1
        ]
        self.assertEqual(len(powers), 1)
        self.assertEqual(replay(x), cube(x), atol=0, rtol=0)
        # the compiled facts predicate at the served call's own values with the size
        # alone moved to 2**21 (the cube is exactly 2**63): a miss, not an exception
        module = self.module
        boxed = replay._families[0].box((x,))
        values = [boxed[i].data_ptr() for i in lowered.pointer_indices]
        values.extend(boxed[i].storage_offset() for i in lowered.offset_indices)
        facts = lowered.facts
        values.extend(module._fact_value(fact, boxed[fact.index]) for fact in facts)
        sizes = [i for i, f in enumerate(facts) if f.kind == "size" and f.dim == 0]
        size_slot = len(values) - len(facts) + sizes[0]
        predicate = ctypes.CFUNCTYPE(
            ctypes.c_int8,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_double),
        )(lowered.facts_address)

        def evaluate(size):
            values[size_slot] = size
            bits = (ctypes.c_uint64 * len(values))(*(v % (2**64) for v in values))
            return predicate(ctypes.cast(bits, ctypes.POINTER(ctypes.c_int64)), None)

        self.assertEqual(evaluate(8), 1)
        self.assertEqual(evaluate(2**21), 0)

    def test_equal_opaque_calls_are_one_plan_node(self):
        # two strided launches over the same shape record IntDivider's shift and magic
        # for the same divisors twice; the plan computes each (function, arguments)
        # once (a traced host's opaque function is pure in its integer arguments)
        def twice(x, b):
            return torch.add(torch.add(x, b), b)

        x, _ = _pair(64, 4096)
        b = torch.randn(4096, device="cuda", dtype=torch.bfloat16)
        replay = self._replay(twice, (x, b))
        self.assertEqual(len(replay.lowered.calls), 2)
        nodes = {}
        for call in replay.lowered.calls:
            for field in call.fields:
                if field.kind == "pointer":
                    continue
                for expr in _walk(field.source.expression):
                    if expr.op == "pcall":
                        nodes[id(expr)] = (
                            expr.value[0],
                            tuple(id(a) for a in expr.args),
                        )
        rebinds = [r for r in replay.lowered.tape.opaque if r["kind"] == "rebind"]
        self.assertGreater(len(nodes), 0)
        self.assertEqual(len(nodes), len(set(nodes.values())))
        self.assertLess(len(nodes), len(rebinds))
        for M, N in ((48, 3000), (33, 2049), (64, 128)):
            x, _ = _pair(M, N)
            b = torch.randn(N, device="cuda", dtype=torch.bfloat16)
            self._check(replay, (x, b), twice)

    def test_the_boxer_builds_the_box_in_one_pass(self):
        # the family's box: the tensors at the tape's positions from the call's tuple,
        # the written positions materialized, the arena appended
        boxer = torch._C._HostTraceBoxer
        base = torch.randn(8, 8, device="cuda")
        lazy_w, lazy_r = torch._lazy_clone(base), torch._lazy_clone(base)
        arena = torch.empty(16, device="cuda", dtype=torch.uint8)
        box = boxer(None, [0])((lazy_w, lazy_r), arena)
        self.assertEqual([id(t) for t in box], [id(lazy_w), id(lazy_r), id(arena)])
        self.assertFalse(torch._C._is_cow_tensor(lazy_w))
        self.assertTrue(torch._C._is_cow_tensor(lazy_r))
        box = boxer([2, 0], [])((lazy_r, 3, base))
        self.assertEqual([id(t) for t in box], [id(base), id(lazy_r)])
        self.assertTrue(torch._C._is_cow_tensor(lazy_r))
        with self.assertRaises(RuntimeError):
            boxer(None, [])([base])
        with self.assertRaises(IndexError):
            boxer([1], [])((base,))
        replay = self._replay(torch.add, _pair(64, 4096))
        family = replay._hot
        args = _pair(64, 4096)
        self.assertEqual([id(t) for t in family.box(args)][:2], [id(t) for t in args])

    def test_strided_broadcast_uses_call_rebinds(self):
        # the strided path: IntDivider's magic and shift are the host's own function,
        # re-run natively per call, so a size sweep across powers of two keeps serving
        def bcast(x, b):
            return torch.add(x, b)

        x, _ = _pair(64, 4096)
        b = torch.randn(4096, device="cuda", dtype=torch.bfloat16)
        replay = self._replay(bcast, (x, b))
        ops = {
            expr.op
            for call in replay.lowered.calls
            for field in call.fields
            if field.kind != "pointer"
            for expr in _walk(field.source.expression)
        }
        self.assertIn("pcall", ops)
        served = 0
        for M, N in (
            (32, 4096),
            (128, 4096),
            (48, 3000),
            (33, 2049),
            (97, 1009),
            (64, 128),
        ):
            x, _ = _pair(M, N)
            b = torch.randn(N, device="cuda", dtype=torch.bfloat16)
            before = replay.misses
            self._check(replay, (x, b), bcast)
            served += replay.misses == before
        self.assertGreaterEqual(served, 4)

    def test_vectorization_width_is_guarded(self):
        base = _pair(64, 4096, offset=8)
        replay = self._replay(torch.add, base)
        self._check(replay, _pair(64, 4096, offset=16), torch.add)
        self.assertEqual(replay.misses, 0)
        self._check(replay, _pair(64, 4096, offset=4), torch.add)
        self.assertEqual(replay.misses, 1)

    def test_unary_ops_and_a_two_op_function(self):
        for fn in (
            F.silu,
            F.gelu,
            lambda t: F.gelu(t, approximate="tanh"),
            lambda t: torch.mul(t, t),
        ):
            x, _ = _pair(64, 4096)
            replay = self._replay(fn, (x,))
            for args in ((_pair(48, 3000)[0],), (_pair(5, 7)[0],)):
                self._check(replay, args, fn)
            self.assertEqual(replay.misses, 0)

        def two(x, y):
            return F.silu(torch.add(x, y))

        replay = self._replay(two, _pair(32, 4096))
        self.assertEqual(len(replay.lowered.calls), 2)
        for args in (_pair(48, 4096), _pair(9, 1000)):
            self._check(replay, args, two)
        self.assertEqual(replay.misses, 0)

    def test_dtype_and_broadcast_changes_miss(self):
        replay = self._replay(torch.add, _pair(64, 4096))
        x, y = _pair(64, 4096, dtype=torch.float16)
        self._check(replay, (x, y), torch.add)
        x, _ = _pair(64, 4096)
        self._check(replay, (x, x[:1]), torch.add)
        self.assertEqual(replay.misses, 2)

    def test_agrees_with_the_interim_replay(self):
        from torch.cuda import _host_trace

        base = _pair(64, 4096)
        replay = self._replay(torch.add, base)
        interim = _host_trace.build(replay.tape, torch.add, base)
        for args in (_pair(48, 4096), _pair(7, 1000), _pair(3, 8)):
            native = replay(*args)
            ours = interim.replay(args)[0]
            self.assertEqual(native, ours, atol=0, rtol=0)


def _walk(expr):
    yield expr
    for arg in expr.args:
        yield from _walk(arg)


if __name__ == "__main__":
    run_tests()
