# Owner(s): ["module: inductor"]
"""Layer norm traced by torch.cuda._host_trace, lowered into the shared native replay."""

import threading
import time
import unittest

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


def _ln(x, w, b):
    return torch.nn.functional.layer_norm(x, (4096,), w, b, 1e-5)


class TestHostTraceLayerNorm(TestCase):
    def setUp(self):
        super().setUp()
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        from torch._inductor.runtime._cudagraph import direct_hosttrace

        self.module = direct_hosttrace

    def _inputs(self, m, n=4096, offset=0, device_index=None):
        device = "cuda" if device_index is None else torch.device("cuda", device_index)
        base = torch.randn(m * n + offset, device=device)
        x = base[offset:].view(m, n)
        w = torch.randn(n, device=device)
        b = torch.randn(n, device=device)
        return x, w, b

    def test_lowering_shape(self):
        replay = self.module.HostTraceReplay(_ln, self._inputs(64))
        try:
            lowered = replay.lowered
            self.assertEqual(len(lowered.calls), 1)
            # three allocations: the output stays a runtime buffer, mean and rstd
            # (temporaries) are blocks of the planned arena
            self.assertEqual(len(lowered.allocations) + len(lowered.arena.blocks), 3)
            self.assertEqual(len(lowered.allocations), 1)
            call = lowered.calls[0]
            kinds = sorted(field.kind for field in call.fields)
            self.assertEqual(kinds, ["pointer"] * 6)
            self.assertEqual(len(call.constants), 2)  # N and eps, captured bytes
            # the grid is a native size load of the input; nothing is boxed but tensors
            self.assertEqual(call.grid[0].op, "size")
            self.assertEqual(lowered.registration[0], ())
            self.assertEqual(lowered.input_count, 4)  # the three tensors and the arena
            kinds = {fact.kind for fact in lowered.facts}
            self.assertTrue({"rank", "dtype", "device", "size"} <= kinds)
            self.assertIn("int_values", lowered.guard_source)
        finally:
            replay.close()

    def test_constant_facts_are_one_memcmp(self):
        # the predicate compares its constant facts (rank, dtype, device, math bits and
        # the sizes and strides the tape fixed) as one memcmp of the leading fact
        # slots against a static array; the relations among the rest stay statements
        replay = self.module.HostTraceReplay(_ln, self._inputs(64))
        try:
            lowered = replay.lowered
            source = lowered.guard_source
            self.assertEqual(source.count("static const int64_t expected_facts["), 1)
            start = len(lowered.pointer_indices) + len(lowered.offset_indices)
            self.assertIn(f"std::memcmp(int_values + {start}, expected_facts", source)
            for kind in ("rank", "dtype", "device", "neg", "conj"):
                self.assertNotIn(f"if (!({kind}0_0 == ", source)
            count = int(source.split("expected_facts[")[1].split("]")[0])
            self.assertGreaterEqual(count, 5 * lowered.input_count)
            self.assertEqual(
                [f.kind for f in lowered.facts[:5]],
                ["rank", "dtype", "device", "neg", "conj"],
            )
            x, w, b = self._inputs(8)
            replay(x, w, b)
            self.assertEqual(replay.misses, 0)
            replay(*(t.to(torch.bfloat16) for t in (x, w, b)))
            self.assertEqual(replay.misses, 1)
        finally:
            replay.close()

    def test_serves_six_shapes_bitwise(self):
        replay = self.module.HostTraceReplay(_ln, self._inputs(64))
        try:
            for m in (64, 8, 3, 1, 128, 257):
                x, w, b = self._inputs(m)
                got = replay(x, w, b)
                self.assertEqual(got, _ln(x, w, b), atol=0, rtol=0)
            self.assertEqual(replay.misses, 0)
            self.assertEqual(replay.served, 6)
        finally:
            replay.close()

    def test_misses_go_to_the_ordinary_host(self):
        replay = self.module.HostTraceReplay(_ln, self._inputs(64))
        try:
            x, w, b = self._inputs(
                16, offset=1
            )  # a 4-byte aligned, not 16-byte aligned, input
            self.assertEqual(replay(x, w, b), _ln(x, w, b), atol=0, rtol=0)
            self.assertEqual(replay.misses, 1)
            x, w, b = self._inputs(16)
            xt = x.t().contiguous().t()  # a stride the tape did not trace
            self.assertEqual(replay(xt, w, b), _ln(xt, w, b), atol=0, rtol=0)
            self.assertEqual(replay.misses, 2)
            # dtype, rank and device are native predicate facts: the entry boxes tensors only
            xh, wh, bh = (v.half() for v in self._inputs(16))
            self.assertTrue(replay.lowered.contract_holds((xh, wh, bh)))
            self.assertEqual(replay(xh, wh, bh), _ln(xh, wh, bh), atol=0, rtol=0)
            self.assertEqual(replay.misses, 3)
            x3, w, b = self._inputs(16)
            self.assertEqual(
                replay(x3.view(2, 8, 4096), w, b),
                _ln(x3.view(2, 8, 4096), w, b),
                atol=0,
                rtol=0,
            )
            self.assertEqual(replay.misses, 4)
            x0, w, b = self._inputs(16)
            self.assertEqual(replay(x0[:0], w, b), _ln(x0[:0], w, b), atol=0, rtol=0)
            self.assertEqual(
                replay.misses, 5
            )  # an empty input: the size fact rejects it
            xn = torch._neg_view(x0)
            self.assertEqual(replay(xn, w, b), _ln(xn, w, b), atol=0, rtol=0)
            self.assertEqual(replay.misses, 6)  # a math bit: the neg fact rejects it
        finally:
            replay.close()

    def test_the_call_path_derives_no_shapes(self):
        replay = self.module.HostTraceReplay(_ln, self._inputs(64))
        try:
            args = self._inputs(8)
            arena = replay._hot.arena.tensor  # the family's arena is boxed last
            self.assertEqual(replay.lowered.box(args, arena), [*args, arena])
            # the argument contract is the only Python check left: arity and constants
            self.assertTrue(replay.lowered.contract_holds(args))
            self.assertFalse(replay.lowered.contract_holds(args[:2]))
        finally:
            replay.close()

    def test_argument_contract_edges(self):
        class Sub(torch.Tensor):
            pass

        def head(x, w, b, k):
            return _ln(x, w, b)[:k]

        replay = self.module.HostTraceReplay(head, (*self._inputs(64), 2))
        try:
            x, w, b = self._inputs(8)
            self.assertEqual(replay(x, w, b, 2), head(x, w, b, 2), atol=0, rtol=0)
            self.assertEqual(replay.misses, 0)
            # a constant that changed value or type, a tensor in its position, and a
            # non-tensor in a tensor position are contract misses: served as written
            for args in (
                (x, w, b, 3),
                (x, w, b, True),
                (x, w, b, torch.tensor(2)),
                (x, None, b, 2),
            ):
                self.assertFalse(replay.lowered.contract_holds(args))
                self.assertEqual(replay(*args), head(*args), atol=0, rtol=0)
            self.assertEqual(replay.misses, 4)
            self.assertFalse(replay.lowered.contract_holds((x, w, b)))
            self.assertFalse(replay.lowered.contract_holds((x, w, b, 2, 2)))
            with self.assertRaisesRegex(TypeError, "positional argument"):
                replay(x, w, b)
            # a Parameter is a plain tensor to the dispatcher; another subclass is not,
            # and the contract (which admits it) is not what refuses it
            p = torch.nn.Parameter(x, requires_grad=False)
            self.assertEqual(replay(p, w, b, 2), head(x, w, b, 2), atol=0, rtol=0)
            self.assertTrue(
                replay.lowered.contract_holds((x.as_subclass(Sub), w, b, 2))
            )
            with self.assertRaisesRegex(TypeError, "Tensors or Parameters"):
                replay(x.as_subclass(Sub), w, b, 2)
            self.assertEqual(replay(x, w, b, 2), head(x, w, b, 2), atol=0, rtol=0)
            self.assertEqual(replay.misses, 5)
            self.assertEqual(replay.calls, 9)
        finally:
            replay.close()

    def test_agrees_with_eager_at_other_shapes(self):
        # the oracle: eager on copies of the call's tensors beside the entry,
        # bitwise output for output and argument for argument
        oracle = self._oracle(_ln, self._inputs(64))
        for m in (8, 3, 1, 128):
            oracle.check(self._inputs(m))

    def test_held_outputs_survive_later_replays(self):
        replay = self.module.HostTraceReplay(_ln, self._inputs(64))
        try:
            x, w, b = self._inputs(32)
            first = replay(x, w, b)
            expected = first.clone()
            for m in (8, 32, 64):
                replay(*self._inputs(m))
            self.assertEqual(first, expected, atol=0, rtol=0)
        finally:
            replay.close()

    # ---- retirement stage B (hosttrace_review/interim_retire): one tape, the native
    # entry prepared from it, compared to eager (the oracle)

    def _oracle(self, fn, args, **kw):
        from torch.testing._internal.host_trace_oracle import Oracle

        oracle = Oracle(fn, args, **kw)
        self.addCleanup(oracle.close)
        return oracle

    def test_a_forked_side_stream_is_refused_by_name(self):
        # the recorder accepts a side stream forked from and joined to the capture
        # stream (O33); the native preparation replays one stream in host order, so
        # it refuses such a tape by name with the tape's stream facts (the eager
        # form of the stack's suites serves it; the DAG replay is the runtime's)
        side = torch.cuda.Stream()

        def forked(x, w, b):
            cur = torch.cuda.current_stream()
            side.wait_stream(cur)
            with torch.cuda.stream(side):
                out = _ln(x, w, b)
            cur.wait_stream(side)
            return out

        oracle = self._oracle(forked, self._inputs(8))
        self.assertIs(oracle.tape.all_on_capture_stream, False)
        self.assertIsNone(oracle.native)
        self.assertIn("all_on_capture_stream=False; 1 launches", oracle.refused)
        # the class is still served at other shapes on the fallback (the tape's
        # own predicate with eager as the executor), bitwise eager
        for m in (4, 16):
            oracle.check(self._inputs(m))
        # an entry constructed at such a call: the constructor's warm-up ran (E24: it
        # is the entry's first call), so the decline leaves the entry in the declined
        # state, warned by name (A306), and the ordinary host serves the class
        with self.assertWarnsRegex(RuntimeWarning, "all_on_capture_stream=False"):
            replay = self.module.HostTraceReplay(forked, self._inputs(8))
        self.addCleanup(replay.close)
        self.assertEqual((len(replay.variants), len(replay.declines)), (0, 1))

    def test_dtype_changing_output_views(self):
        # view_as_real of a complex input: the input's offset is in complex elements,
        # the view's in floats (an odd complex offset may miss on the alignment guard,
        # an even one serves); view_as_complex of the output: an output view with a
        # dtype of its own over the allocation, bound with that dtype and its units
        def real_view(z, w, b):
            real = torch.view_as_real(z)
            return torch.nn.functional.layer_norm(real, (2,), w, b, 1e-5)

        first = torch.randn(8 * 16 + 4, device="cuda", dtype=torch.complex64)
        w, b = torch.randn(2, device="cuda"), torch.randn(2, device="cuda")
        oracle = self._oracle(real_view, (first[: 8 * 16].view(8, 16), w, b))
        self.assertIsNone(oracle.refused)
        second = torch.randn_like(first)
        served = []
        for start in (1, 2, 4):
            changed = (second[start : start + 8 * 16].view(8, 16), w, b)
            out = oracle.try_check(changed)  # served bitwise, or a miss
            if out is not None:
                served.append(start)
                self.assertEqual(out[0], real_view(*changed), atol=0, rtol=0)
        self.assertTrue({2, 4} <= set(served), served)

        def complex_out(x, w, b):
            return torch.view_as_complex(_ln(x, w, b).view(x.shape[0], 2048, 2))

        oracle = self._oracle(complex_out, self._inputs(8))
        self.assertIsNone(oracle.refused)
        (view,) = oracle.native.lowered.outputs
        self.assertEqual(view.dtype, torch.complex64)
        for m in (8, 3, 1):
            (out,) = oracle.check(self._inputs(m))
            self.assertEqual((out.dtype, out.shape), (torch.complex64, (m, 2048)))

    def test_preparation_survives_traces_on_other_threads(self):
        # the preparation captures on its own stream in relaxed mode with no
        # device-wide synchronize: traces in flight on other threads survive it,
        # and several threads trace, prepare and serve at once
        from torch.cuda import _host_trace

        stop = threading.Event()
        errors: list = []

        def builder():
            try:
                args = self._inputs(8)
                tape = _host_trace.trace(_ln, args)
                while not stop.is_set():
                    self.module.HostTraceReplay(_ln, args, tape=tape).close()
            except Exception as e:
                errors.append(e)

        thread = threading.Thread(target=builder)
        thread.start()
        try:
            time.sleep(0.2)
            for m in range(9, 19):
                tape = _host_trace.trace(_ln, self._inputs(m))
                self.assertEqual(tape.num_launches, 1)
        finally:
            stop.set()
            thread.join()
        self.assertEqual(errors, [])

        results: list = []

        def worker(k):
            try:
                replay = self.module.HostTraceReplay(_ln, self._inputs(8 + k))
                for j in range(5):
                    x, w, b = self._inputs(8 + k + j)
                    results.append((replay(x, w, b), _ln(x, w, b)))
                replay.close()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(k,)) for k in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        torch.cuda.synchronize()
        self.assertEqual(errors, [])
        self.assertEqual(len(results), 15)
        for got, want in results:
            self.assertEqual(got, want, atol=0, rtol=0)

    @unittest.skipIf(torch.cuda.device_count() < 2, "two GPUs")
    def test_an_entry_on_a_second_device(self):
        # an entry traced and prepared on cuda:1 after one on cuda:0: the graph, the
        # buffers, the arena and the events are cuda:1's; it is called under its device
        first = self.module.HostTraceReplay(_ln, self._inputs(16))
        self.addCleanup(first.close)
        with torch.cuda.device(1):
            oracle = self._oracle(_ln, self._inputs(16))
            self.assertIsNone(oracle.refused)
            self.assertEqual(oracle.native.lowered.device, 1)
            for m in (40, 8):
                (out,) = oracle.check(self._inputs(m))
                self.assertEqual(out.device, torch.device("cuda", 1))
            torch.cuda.synchronize(1)
        x, w, b = self._inputs(8)
        self.assertEqual(first(x, w, b), _ln(x, w, b), atol=0, rtol=0)

    @unittest.skipIf(torch.cuda.device_count() < 2, "two GPUs")
    def test_a_tape_traced_on_the_first_device_prepared_on_the_second(self):
        # a tape traced on cuda:0, prepared with device=1: the variant's device is
        # where its exec, buffers and events live, and its device facts are cuda:1's
        from torch.cuda import _host_trace

        tape = _host_trace.trace(_ln, self._inputs(16))
        with torch.cuda.device(1):
            oracle = self._oracle(_ln, self._inputs(16), tape=tape, device=1)
            self.assertIsNone(oracle.refused)
            self.assertEqual(oracle.native.lowered.device, 1)
            (out,) = oracle.check(self._inputs(24))
            self.assertEqual(out.device, torch.device("cuda", 1))
        # cuda:0 inputs miss the device fact
        with torch.cuda.device(1):
            oracle.expect_miss(self._inputs(24, device_index=0))

    def test_an_output_no_launch_writes_is_a_runtime_buffer(self):
        # an allocation the host returns that no launch writes (eager's at::empty as it
        # is), at a symbolic shape: a per-call runtime buffer of the variant, its
        # metadata eager's, its values indeterminate on every path
        def with_scratch(x, w, b):
            scratch = torch.empty(x.shape[0], 8, device=x.device, dtype=x.dtype)
            return _ln(x, w, b), scratch

        oracle = self._oracle(with_scratch, self._inputs(16))
        self.assertIsNone(oracle.refused)
        lowered = oracle.native.lowered
        self.assertEqual(lowered.unwritten_outputs, (1,))
        self.assertEqual(len(lowered.outputs), 2)
        for m in (16, 3, 40):
            _, scratch = oracle.check(self._inputs(m))
            self.assertEqual((scratch.shape, scratch.dtype), ((m, 8), torch.float32))
        self.assertEqual(oracle.native.misses, 0)


if __name__ == "__main__":
    run_tests()
