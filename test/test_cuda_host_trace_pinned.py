# Owner(s): ["module: cuda graphs"]

import gc
import unittest

import torch
import torch.cuda._host_trace_replay as _host_trace_replay
from torch.cuda._host_trace_capture import graph_nodes, MemcpyNode
from torch.cuda._host_trace_lower_tape import LoweredMemcpy
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    requires_cuda_python_bindings,
    run_tests,
    TEST_CUDA,
    TestCase,
)


class HostTraceReplay(_host_trace_replay.HostTraceReplay):
    # traces at its first call
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._called = True


def _pinned(n, dtype=torch.int64):
    return torch.zeros(n, dtype=dtype, pin_memory=True)


def _values(n, k, dtype=torch.int64):
    return (torch.arange(n) * 7 + 13 * k).to(dtype)


def h2d(h, d):
    d.copy_(h, non_blocking=True)
    return d * 2


def d2h(x, h):
    y = x + 1
    h.copy_(y, non_blocking=True)
    return y


class Staged:
    # an engine object whose pinned staging and output buffers rotate per step
    def __init__(self, staging, out):
        self.staging, self.out = staging, out


def staged(d, staging, out):
    y = d.copy_(staging, non_blocking=True) * 2
    out.copy_(y, non_blocking=True)
    return y


@requires_cuda_python_bindings
@unittest.skipIf(not TEST_CUDA, "requires CUDA")
class TestPinnedMemcpy(TestCase):
    def assertOneGraph(self, f, kinds):
        # every call replays one segment holding the memcpy nodes, and nothing runs outside it
        self.assertEqual((f.eager, f.declines), (0, []))
        (v,) = f.variants
        lowered = v.captured.lowered
        self.assertTrue(all(isinstance(s, range) for s in lowered.steps), lowered.steps)
        self.assertEqual(len(v.captured.segments), 1)
        copies = [lo.launch.kind for lo in lowered.launches if isinstance(lo, LoweredMemcpy)]
        self.assertEqual(copies, kinds)
        nodes = graph_nodes(v.captured.segments[0].graph.raw_cuda_graph(), memcpy=True)
        self.assertEqual([n.kind for n in nodes if isinstance(n, MemcpyNode)], kinds)

    @parametrize("n", [1, 7, 4096, 1 << 20])
    @parametrize("dtype", [torch.int64, torch.float16])
    def test_h2d_argument(self, n, dtype):
        f = HostTraceReplay(h2d)
        h, d = _pinned(n, dtype), torch.zeros(n, dtype=dtype, device="cuda")
        for k in range(4):
            h.copy_(_values(n, k, dtype))
            out = f(h, d)
            torch.cuda.synchronize()
            self.assertEqual(out, _values(n, k, dtype).cuda() * 2, atol=0, rtol=0)
            self.assertEqual(d.cpu(), h, atol=0, rtol=0)
        self.assertEqual(f.replays, 3)
        self.assertOneGraph(f, ["H2D"])

    @parametrize("n", [1, 7, 4096, 1 << 20])
    @parametrize("dtype", [torch.int64, torch.float16])
    def test_d2h_argument(self, n, dtype):
        f = HostTraceReplay(d2h)
        x, h = torch.zeros(n, dtype=dtype, device="cuda"), _pinned(n, dtype)
        for k in range(4):
            x.copy_(_values(n, k, dtype))
            h.fill_(-1)
            out = f(x, h)
            torch.cuda.synchronize()
            want = x + 1
            self.assertEqual(out, want, atol=0, rtol=0)
            self.assertEqual(h, want.cpu(), atol=0, rtol=0)
        self.assertEqual(f.replays, 3)
        self.assertOneGraph(f, ["D2H"])

    def test_rotating_pinned_arguments(self):
        # a pool of staging and output buffers, one of each per call, as an engine's
        n = 1000
        f = HostTraceReplay(lambda h, d, out: out.copy_(d.copy_(h, non_blocking=True) + 1, non_blocking=True))
        ins, outs = [_pinned(n) for _ in range(3)], [_pinned(n) for _ in range(3)]
        d = torch.zeros(n, dtype=torch.int64, device="cuda")
        for k in range(9):
            h, out = ins[k % 3], outs[k % 3]
            h.copy_(_values(n, k))
            out.fill_(-1)
            self.assertIs(f(h, d, out), out)
            torch.cuda.synchronize()
            self.assertEqual(out, _values(n, k) + 1, atol=0, rtol=0)
        self.assertOneGraph(f, ["H2D", "D2H"])

    def test_sliced_pinned_arguments(self):
        # views at offsets into one pinned block, as staging slots carved from one buffer
        n = 256
        block = _pinned(4 * n)
        d = torch.zeros(n, dtype=torch.int64, device="cuda")
        f = HostTraceReplay(h2d)
        for k in range(4):
            h = block[k * n : (k + 1) * n]
            h.copy_(_values(n, k))
            self.assertEqual(f(h, d), _values(n, k).cuda() * 2, atol=0, rtol=0)
        self.assertOneGraph(f, ["H2D"])

    def test_to_device(self):
        f = HostTraceReplay(lambda h, x: h.to("cuda", non_blocking=True) + x)
        h, x = _pinned(100), torch.ones(100, dtype=torch.int64, device="cuda")
        for k in range(3):
            h.copy_(_values(100, k))
            self.assertEqual(f(h, x), _values(100, k).cuda() + 1, atol=0, rtol=0)
        self.assertOneGraph(f, ["H2D"])

    def test_vllm_like_step(self):
        # staging in, an embedding and a greedy sampler, the sampled tokens out: one graph
        R, V, H = 8, 512, 64
        table = torch.randn(V, H, device="cuda")
        w = torch.randn(H, device="cuda")

        def step(staging, sampled, idx, table, w):
            idx.copy_(staging, non_blocking=True)
            scores = torch.nn.functional.embedding(idx, table) * w
            tokens = scores.max(dim=-1).indices
            sampled.copy_(tokens, non_blocking=True)
            return tokens

        f = HostTraceReplay(step)
        idx = torch.zeros(R, dtype=torch.int64, device="cuda")
        g = torch.Generator().manual_seed(0)
        for k in range(5):
            staging, sampled = _pinned(R), _pinned(R)
            staging.copy_(torch.randint(0, V, (R,), generator=g))
            sampled_ref, idx_ref = _pinned(R), torch.zeros_like(idx)
            want = step(staging, sampled_ref, idx_ref, table, w)
            out = f(staging, sampled, idx, table, w)
            torch.cuda.synchronize()
            self.assertEqual(out, want, atol=0, rtol=0)
            self.assertEqual(sampled, sampled_ref, atol=0, rtol=0)
        self.assertOneGraph(f, ["H2D", "D2H"])

    def test_copies_run_when_the_graph_does(self):
        # as eager's non_blocking copies: the H2D reads the host buffer and the D2H writes
        # its own when they run on the stream, after the work queued before them
        n = 4096
        def step(h, d, out):
            y = d.copy_(h, non_blocking=True) * 3
            out.copy_(y, non_blocking=True)
            return y

        f = HostTraceReplay(step)
        d = torch.zeros(n, dtype=torch.int64, device="cuda")
        h, out = _pinned(n), _pinned(n)
        f(h, d, out)
        f(h, d, out)
        for fn in (f, step):
            h.copy_(_values(n, 1))
            out.fill_(-1)
            torch.cuda._sleep(200_000_000)
            fn(h, d, out)
            # the stream is still sleeping: nothing ran yet
            self.assertEqual(out, torch.full((n,), -1), atol=0, rtol=0)
            h.copy_(_values(n, 2))
            torch.cuda.synchronize()
            self.assertEqual(out, _values(n, 2) * 3, atol=0, rtol=0)
        self.assertEqual((f.eager, f.replays), (0, 2))
        self.assertOneGraph(f, ["H2D", "D2H"])

    def test_a_view_of_a_pinned_argument_is_no_output(self):
        f = HostTraceReplay(lambda h, d: h.copy_(d, non_blocking=True)[1:])
        h, d = _pinned(8), torch.arange(8, device="cuda")
        out = f(h, d)
        torch.cuda.synchronize()
        self.assertEqual(out, torch.arange(1, 8))
        self.assertEqual(f.variants, [])
        self.assertIn("output 0 is a view of a pinned host argument", f.declines[0])

    def test_a_freed_argument_outlives_the_replay(self):
        # the host allocator does not hand out a pinned block the graph still reads (record_event)
        n = 1 << 16
        f = HostTraceReplay(h2d)
        d = torch.zeros(n, dtype=torch.int64, device="cuda")
        f(_pinned(n), d)
        torch.cuda.synchronize()
        h = _pinned(n)
        h.copy_(_values(n, 5))
        ptr = h.data_ptr()
        torch.cuda._sleep(200_000_000)
        out = f(h, d)
        del h
        again = _pinned(n)
        self.assertNotEqual(again.data_ptr(), ptr)
        torch.cuda.synchronize()
        self.assertEqual(out, _values(n, 5).cuda() * 2, atol=0, rtol=0)
        self.assertEqual(f.replays, 1)

    @unittest.skipIf(not hasattr(torch._C, "_HostTraceBound"), "requires the bound call")
    def test_a_bound_call_reads_rotating_buffers_per_call(self):
        # buffers held on an object, not passed: the bound call reads the attributes at every call
        n = 512
        d = torch.zeros(n, dtype=torch.int64, device="cuda")
        ins, outs = [_pinned(n) for _ in range(2)], [_pinned(n) for _ in range(2)]
        f = HostTraceReplay(staged, static_prefix=1, opaque=())
        obj = Staged(ins[0], outs[0])
        for _ in range(2):
            f(d, obj.staging, obj.out)
        b = torch._C._HostTraceBound((d,), 0, 1, ())
        b.add(f, (Staged,), (), (), (), ((0, "staging"), (0, "out")), (), None, ())
        replays = f.replays
        for k in range(6):
            obj.staging, obj.out = ins[k % 2], outs[k % 2]
            obj.staging.copy_(_values(n, k))
            obj.out.fill_(-1)
            y = b(obj)
            torch.cuda.synchronize()
            self.assertEqual(y, _values(n, k).cuda() * 2, atol=0, rtol=0)
            self.assertEqual(obj.out, _values(n, k) * 2, atol=0, rtol=0)
        self.assertEqual((b.hits(), f.replays), ([6], replays + 6))
        # a pageable buffer is no plan's: nothing runs
        obj.staging = _values(n, 9)
        self.assertIs(b(obj), NotImplemented)
        self.assertOneGraph(f, ["H2D", "D2H"])

    @parametrize("direction", ["d2h", "h2d", "d2d"])
    def test_a_form_instantiates_after_the_traced_buffers_are_freed(self, direction):
        # F24: a new mm key instantiates a form from a clone of the variant's graph, whose memcpy node
        # still held its capture's operands; with the pinned (and device) caches emptied the driver
        # crashed on the freed host pointer
        def f(a0, a1, buf):
            v1 = torch.sigmoid(a1)
            v2 = a0.t() @ a0
            if direction == "d2h":
                buf.copy_(v1, non_blocking=True)
            elif direction == "h2d":
                v1 = v1 + a1.new_empty(a1.shape).copy_(buf, non_blocking=True)
            else:
                v1 = v1 + torch.empty_like(a1).copy_(buf)
            return torch.abs(a1), v2, v1

        def mk(n, k, m, seed):
            g = torch.Generator(device="cuda").manual_seed(seed)
            a0 = torch.randn(n, k, device="cuda", generator=g).bfloat16()
            a1 = torch.randn(k, m, device="cuda", generator=g).bfloat16()
            buf = torch.randn(k, m, device="cuda", generator=g).bfloat16()
            return a0, a1, buf if direction == "d2d" else buf.cpu().pin_memory()

        r = _host_trace_replay.HostTraceReplay(f)
        for i, shape in enumerate([(58, 1024, 50), (58, 1024, 50), (2, 1025, 50), (2, 1025, 50)]):
            args = mk(*shape, i)
            want = f(*(a.clone() for a in args))
            out = r(*args)
            torch.cuda.synchronize()
            self.assertEqual(out[0], want[0], atol=0, rtol=0)
            self.assertEqual(out[2], want[2], atol=0, rtol=0)
            if direction == "d2h":
                self.assertEqual(args[2], torch.sigmoid(args[1]).cpu(), atol=0, rtol=0)
            del args, out, want
            gc.collect()
            torch.cuda.empty_cache()
            torch._C._host_emptyCache()
        self.assertGreater(r.replays, 0)

    def test_a_replay_on_the_same_buffers_after_a_new_trace(self):
        # F24's other form: the same tensors replayed again after a trace at a new size
        def f(a0, a1, p0):
            v1 = torch.sigmoid(a1)
            v2 = a0.t() @ a0
            p0.copy_(v1, non_blocking=True)
            return torch.abs(a1), a0.sum(), v2

        r = _host_trace_replay.HostTraceReplay(f)
        for i, (n, k, reuse) in enumerate([(58, 1024, False), (58, 1024, False), (0, 0, True), (2, 1025, False), (0, 0, True)]):
            if not reuse:
                g = torch.Generator(device="cuda").manual_seed(i)
                args = (torch.randn(n, k, device="cuda", generator=g).bfloat16(), torch.randn(k, 50, device="cuda", generator=g).bfloat16(),
                        torch.empty(k, 50, dtype=torch.bfloat16, pin_memory=True))
            out = r(*args)
            torch.cuda.synchronize()
            self.assertEqual(out[0], torch.abs(args[1]), atol=0, rtol=0)
            self.assertEqual(args[2], torch.sigmoid(args[1]).cpu(), atol=0, rtol=0)

    @parametrize("direction", ["h2d", "d2h"])
    def test_pageable_declines(self, direction):
        # eager's copy_ to or from pageable memory is synchronous
        n = 64
        f = HostTraceReplay(h2d if direction == "h2d" else d2h)
        dev = torch.zeros(n, dtype=torch.int64, device="cuda")
        for k in range(3):
            host = _values(n, k)
            out = f(host, dev) if direction == "h2d" else f(dev, host)
            torch.cuda.synchronize()
            if direction == "h2d":
                self.assertEqual(out, _values(n, k).cuda() * 2)
            else:
                self.assertEqual(host, dev.cpu() + 1)
        self.assertEqual(f.variants, [])
        self.assertTrue(f.declines and all("only CUDA tensors on one device and pinned CPU tensors" in d for d in f.declines), f.declines)

    def test_a_pageable_call_misses_a_pinned_variant(self):
        n = 64
        f = HostTraceReplay(h2d)
        d = torch.zeros(n, dtype=torch.int64, device="cuda")
        f(_pinned(n), d)
        self.assertEqual(len(f.variants), 1)
        self.assertEqual(f(_values(n, 3), d), _values(n, 3).cuda() * 2)
        self.assertEqual(f.replays, 0)
        self.assertEqual(len(f.variants), 1)

    def test_blocking_copy_declines(self):
        f = HostTraceReplay(lambda h, d: d.copy_(h) * 2)
        h, d = _pinned(64), torch.zeros(64, dtype=torch.int64, device="cuda")
        h.copy_(_values(64, 1))
        self.assertEqual(f(h, d), _values(64, 1).cuda() * 2)
        self.assertEqual(f.variants, [])
        self.assertIn("a blocking aten.copy_.default to or from a pinned host argument", f.declines[0])

    def test_rotating_closure_slots_decline(self):
        # a closure's buffer is Python state a replay does not re-read: as a closure's CUDA
        # tensor, it is not traced, so a rotation the graph would miss cannot happen
        n = 64
        slots, state = [_pinned(n), _pinned(n)], [0]
        d = torch.zeros(n, dtype=torch.int64, device="cuda")

        def fn(d):
            k = state[0] = (state[0] + 1) % 2
            d.copy_(slots[k], non_blocking=True)
            return d * 2

        f = HostTraceReplay(fn)
        for k in range(4):
            slots[(state[0] + 1) % 2].copy_(_values(n, k))
            self.assertEqual(f(d), _values(n, k).cuda() * 2, atol=0, rtol=0)
        self.assertEqual(f.variants, [])
        self.assertIn("a tensor the trace does not track", f.declines[0])

    def test_other_ops_of_a_pinned_argument_decline(self):
        f = HostTraceReplay(lambda h, d: d + h.sum())
        h, d = _pinned(8), torch.zeros(8, dtype=torch.int64, device="cuda")
        self.assertEqual(f(h, d), d + h.sum())
        self.assertEqual(f.variants, [])
        self.assertIn("of a pinned host argument; only its copy_ to or from the device is traced", f.declines[0])

    def test_non_contiguous_copy_declines(self):
        f = HostTraceReplay(lambda h, d: d.copy_(h, non_blocking=True) * 2)
        h, d = _pinned(16).view(4, 4).t(), torch.zeros(4, 4, dtype=torch.int64, device="cuda")
        h.copy_(torch.arange(16).view(4, 4))
        self.assertEqual(f(h, d), h.cuda() * 2)
        self.assertEqual(f.variants, [])
        self.assertIn("not both contiguous", f.declines[0])


instantiate_parametrized_tests(TestPinnedMemcpy)


if __name__ == "__main__":
    run_tests()
