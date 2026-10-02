# Owner(s): ["module: cuda graphs"]

import ctypes
import gc
import itertools
import os
import sys
import threading
import unittest
import weakref
from unittest import mock

import torch
from torch.cuda import _host_trace_capture, _host_trace_replay
from torch.cuda._host_trace import declined
from torch.cuda._host_trace_capture import launch_images, launch_slots
from torch.cuda._host_trace_lower_tape import LoweredMemcpy, LoweredMemset
from torch.cuda._host_trace_memory import MemoryPlan, place, StepMemory
from torch.cuda._host_trace_replay import _exact_class
from torch.cuda._host_trace_tape import argument_contract, EagerCall, trace
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_CUDNN_ATTENTION
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    requires_cuda_python_bindings,
    run_tests,
    TEST_CUDA,
    TestCase,
)
from torch.utils._triton import has_triton


class HostTraceReplay(_host_trace_replay.HostTraceReplay):
    # traces at its first call: these are tests of the trace; an entry's first
    # call runs eagerly (test_the_first_call_runs_eagerly)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._called = True


if has_triton():
    import triton
    import triton.language as tl

    @triton.jit
    def _add(x_ptr, y_ptr, n, s, B: tl.constexpr):
        i = tl.program_id(0) * B + tl.arange(0, B)
        m = i < n
        tl.store(y_ptr + i, tl.load(x_ptr + i, mask=m) + s, mask=m)

    @triton.autotune(
        configs=[triton.Config({"B": 128}), triton.Config({"B": 256})], key=["n"]
    )
    @triton.jit
    def _add_tuned(x_ptr, y_ptr, n, s, B: tl.constexpr):
        i = tl.program_id(0) * B + tl.arange(0, B)
        m = i < n
        tl.store(y_ptr + i, tl.load(x_ptr + i, mask=m) + s, mask=m)

    @triton.heuristics(
        {
            "B": lambda a: triton.next_power_of_2(a["n"]),
            "EVEN": lambda a: a["n"] % 16 == 0,
        }
    )
    @triton.jit
    def _add_heuristic(x_ptr, y_ptr, n, s, B: tl.constexpr, EVEN: tl.constexpr):
        i = tl.arange(0, B)
        if EVEN:
            tl.store(y_ptr + i, tl.load(x_ptr + i, mask=i < n) + s, mask=i < n)
        else:
            tl.store(y_ptr + i, tl.load(x_ptr + i, mask=i < n) + 2 * s, mask=i < n)

    @triton.jit
    def _scale2d(x_ptr, y_ptr, M, N, sxm, sxn, BM: tl.constexpr, BN: tl.constexpr):
        m = tl.program_id(0) * BM + tl.arange(0, BM)[:, None]
        n = tl.program_id(1) * BN + tl.arange(0, BN)[None, :]
        mask = (m < M) & (n < N)
        x = tl.load(x_ptr + m * sxm + n * sxn, mask=mask)
        tl.store(y_ptr + m * N + n, x * 2, mask=mask)


def add(x, s=3):
    y = torch.empty_like(x)
    n = x.numel()
    _add[(triton.cdiv(n, 128),)](x, y, n, s, B=128)
    return y


def two_step(x):
    return add(add(x), 1)


def matmul_chain(x, w):
    # triton -> mm (eager; matmul folds the batch with _unsafe_view) -> triton
    return add(torch.matmul(add(x).view(4, -1, w.shape[0]), w), 1)


# CUDA kernels the trace does not follow: eager calls, whose metadata comes
# from their fake kernels
_lib = torch.library.Library("host_trace_test", "FRAGMENT")
_lib.define("transposed_fake(Tensor x) -> Tensor")
_lib.impl("transposed_fake", lambda x: x.clone(), "CUDA")
_lib.define("transposed_fake_at_4(Tensor x) -> Tensor")
_lib.impl("transposed_fake_at_4", lambda x: x.clone(), "CUDA")
transposed_fake = torch.ops.host_trace_test.transposed_fake
transposed_fake_at_4 = torch.ops.host_trace_test.transposed_fake_at_4


@torch.library.register_fake("host_trace_test::transposed_fake")
def _(x):
    return x.new_empty(x.shape[::-1]).t()


@torch.library.register_fake("host_trace_test::transposed_fake_at_4")
def _(x):
    if x.shape[0] == 4:
        return x.new_empty(x.shape[::-1]).t()
    return torch.empty_like(x)


@torch.library.custom_op("host_trace_test::traced_lying_fake", mutates_args=())
def traced_lying_fake(x: torch.Tensor) -> torch.Tensor:
    return x.clone()


@traced_lying_fake.register_fake
def _(x):
    return x.new_empty(x.shape[::-1]).t()


@torch.library.custom_op("host_trace_test::size_1_stride", mutates_args=())
def size_1_stride(x: torch.Tensor) -> torch.Tensor:
    # a single-query head as flash and cuDNN write it: the size-1 dim's
    # stride is not its fake's
    n, _, k = x.shape
    return torch.empty_strided(x.shape, (k, n * k, 1), device=x.device).copy_(x)


@size_1_stride.register_fake
def _(x):
    return torch.empty_like(x, memory_format=torch.contiguous_format)


# the entry reenters() calls back into, and whether lies_when_small lies
REENTERED: list = []
LIE = [False]


# an eager call: as a custom_op its clone would trace (the copy is a memcpy)
_lib.define("lies_when_small(Tensor x) -> Tensor")
_lib.impl("lies_when_small", lambda x: x[1:].clone() if LIE[0] and x.numel() <= 1024 else x.clone(), "CUDA")
lies_when_small = torch.ops.host_trace_test.lies_when_small


@torch.library.register_fake("host_trace_test::lies_when_small")
def _(x):
    return torch.empty_like(x)


@torch.library.custom_op("host_trace_test::reenters", mutates_args=())
def reenters(x: torch.Tensor) -> torch.Tensor:
    if x.numel() <= 1024:
        return x * 2
    h = x.numel() // 2
    return torch.cat([REENTERED[0](x[:h].contiguous()), x[h:] * 2])


@reenters.register_fake
def _(x):
    return torch.empty_like(x)


def scale2d(x):
    M, N = x.shape
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(M, 16), triton.cdiv(N, 32))
    _scale2d[grid](x, y, M, N, x.stride(0), x.stride(1), BM=16, BN=32)
    return y


def _contract(args, global_state=False):
    return argument_contract(args, global_state)


class TestContract(TestCase):
    def test_contract(self):
        x = torch.randn(4, 4)
        self.assertEqual(_contract((x, 3, 2.0)), _contract((x[:2], 5, 2.0)))
        self.assertNotEqual(_contract((x, 3)), _contract((x, 3.0)))
        self.assertNotEqual(_contract((x, 3)), _contract((x, True)))
        self.assertNotEqual(_contract((x,)), _contract((x.double(),)))
        self.assertNotEqual(_contract((x,)), _contract((x[0],)))
        self.assertNotEqual(_contract((x, 2.0)), _contract((x, -2.0)))
        self.assertEqual(_contract((x, [1, 2])), _contract((x, (1, 2))))
        c = _contract((x, 3))
        self.assertEqual(_exact_class(c, (x, 3)), _exact_class(c, (x.clone(), 3)))
        self.assertNotEqual(_exact_class(c, (x, 3)), _exact_class(c, (x.t(), 3)))
        self.assertNotEqual(_exact_class(c, (x, 3)), _exact_class(c, (x, 4)))

    def test_cpu_arguments_run_eagerly(self):
        f = HostTraceReplay(lambda x: x + 1)
        x = torch.randn(3)
        self.assertEqual(f(x), x + 1)
        self.assertEqual(f(x), x + 1)
        self.assertEqual((f.traces, f.replays, f.eager), (1, 0, 2))
        self.assertEqual(len(f.declines), 1)


@unittest.skipIf(not TEST_CUDA, "requires CUDA")
@requires_cuda_python_bindings
@unittest.skipIf(not has_triton(), "requires triton")
class TestHostTraceReplay(TestCase):
    def test_numerics_across_sizes(self):
        f = HostTraceReplay(add)
        sizes = [1000, 3000, 1, 2, 17, 128, 129, 1024, 4096, 5000, 1000]
        for n in sizes:
            x = torch.randn(n, device="cuda")
            self.assertEqual(f(x), x + 3)
        # sizes share variants: Triton specializes n only on %16 and ==1
        self.assertLess(f.traces, len(sizes))
        self.assertEqual(f.eager, 0)
        self.assertEqual(f.replays, len(sizes) - 1)

    @parametrize("layout", ["contiguous", "transposed", "sliced", "strided"])
    def test_numerics_across_strides(self, layout):
        f = HostTraceReplay(scale2d)
        for M, N in [(64, 96), (37, 50), (1, 33), (130, 7), (64, 96)]:
            base = torch.randn(2 * M + 3, 2 * N + 5, device="cuda")
            x = {
                "contiguous": base[:M, :N].contiguous(),
                "transposed": base[:N, :M].t(),
                "sliced": base[1 : M + 1, 2 : N + 2],
                "strided": base[::2, ::2][:M, :N],
            }[layout]
            self.assertEqual(f(x), x * 2)
        self.assertEqual(f.eager, 0)

    def test_hits_and_misses(self):
        calls = []

        def fn(x):
            calls.append(x.numel())
            return add(x)

        f = HostTraceReplay(fn)
        f(torch.randn(1000, device="cuda"))
        # the first call warms up and traces
        self.assertEqual((len(calls), f.traces, f.replays), (2, 1, 0))
        x = torch.randn(3000, device="cuda")
        self.assertEqual(f(x), x + 3)
        # a hit runs no Python of fn
        self.assertEqual((len(calls), f.traces, f.replays), (2, 1, 1))
        x = torch.randn(1024, device="cuda")
        self.assertEqual(f(x), x + 3)
        # n % 16 flips Triton's specialization, the launch's own guard: a miss
        # runs that launch again alone, as an entry of the variant
        self.assertEqual((len(calls), f.traces, f.replays), (2, 1, 2))
        self.assertEqual((len(f.variants), f.redispatches), (1, 1))
        for n in (1000, 1024, 2048, 3000):
            x = torch.randn(n, device="cuda")
            self.assertEqual(f(x), x + 3)
        self.assertEqual((len(calls), f.traces), (2, 1))

    def test_the_first_call_runs_eagerly(self):
        calls = []

        def fn(x):
            calls.append(x.numel())
            return add(x)

        f = _host_trace_replay.HostTraceReplay(fn)
        x = torch.randn(1000, device="cuda")
        # as cudagraph trees': the first call is eager, the second traces (its
        # warm-up is the call), the third replays
        for expected in ((1, 0, 0, 1), (3, 1, 0, 1), (3, 1, 1, 1)):
            self.assertEqual(f(x), x + 3)
            self.assertEqual((len(calls), f.traces, f.replays, f.eager), expected)

    def test_len_compared_to_a_constant_guards_the_comparison(self):
        def fn(x):
            return x if len(x) == 0 else add(x)

        f = HostTraceReplay(fn)
        for n in (1000, 3000, 1000):
            x = torch.randn(n, device="cuda")
            self.assertEqual(f(x), x + 3)
        self.assertEqual((f.traces, f.replays), (1, 2))

    def test_a_misaligned_input_redispatches(self):
        f = HostTraceReplay(add)
        base = torch.randn(4000, device="cuda")
        self.assertEqual(f(base[:1000]), base[:1000] + 3)
        x = base[1:1001]
        self.assertEqual(f(x), x + 3)
        self.assertEqual((f.traces, len(f.variants), f.redispatches), (1, 1, 1))
        # the offset-4 entry also serves an offset-8 input
        x = base[2:1002]
        self.assertEqual(f(x), x + 3)
        self.assertEqual((f.traces, f.redispatches), (1, 1))

    def test_repeated_replays_keep_no_stale_state(self):
        f = HostTraceReplay(add)
        inputs = [torch.randn(n, device="cuda") for n in (1000, 3000, 1024) * 4]
        outputs = [f(x) for x in inputs]
        torch.cuda.synchronize()
        for x, y in zip(inputs, outputs):
            self.assertEqual(y, x + 3)
        ptrs = {y.data_ptr() for y in outputs}
        self.assertEqual(len(ptrs), len(outputs))
        # a replay after its input changes reads the new values
        x = inputs[1]
        x.fill_(1.0)
        self.assertEqual(f(x), torch.full_like(x, 4.0))

    def test_a_decline_falls_back_once(self):
        calls = []

        def fn(x):
            calls.append(1)
            return x.cumsum(0)

        f = HostTraceReplay(fn)
        x = torch.randn(100, device="cuda")
        # the warm-up is the call: fn runs once for it and once for the trace
        self.assertEqual(f(x), x.cumsum(0))
        self.assertEqual((len(calls), f.traces, f.eager), (2, 1, 1))
        # no traced launches: uncaptured, not a decline
        self.assertEqual((f.uncaptured, f.declines), (1, []))
        # the declined class runs eagerly without a trace
        self.assertEqual(f(x), x.cumsum(0))
        self.assertEqual((len(calls), f.traces, f.eager), (3, 1, 2))
        # another class traces again, with its warm-up
        y = torch.randn(7, device="cuda")
        self.assertEqual(f(y), y.cumsum(0))
        self.assertEqual((len(calls), f.traces, f.eager), (5, 2, 3))
        self.assertEqual((f.variants, f.uncaptured, f.declines), ([], 2, []))

    def test_a_tensor_constant_runs_eagerly(self):
        f = HostTraceReplay(lambda xs: add(xs[0]))
        for n in (100, 200):
            x = torch.randn(n, device="cuda")
            self.assertEqual(f([x]), x + 3)
        # the contract holds the list's shape, not its tensor: one decline
        self.assertEqual((f.traces, f.eager), (1, 2))
        self.assertIn("arg0 holds a tensor inside a list", f.declines[0])

    def test_heuristics(self):
        def fn(x, n, s):
            y = torch.empty_like(x)
            _add_heuristic[(1,)](x, y, n, s)
            return y

        f = HostTraceReplay(fn)
        for n in (1000, 1001, 1008, 1024, 1000, 1008, 1025, 1):
            x = torch.randn(n, device="cuda")
            want = x + (3 if n % 16 == 0 else 6)
            self.assertEqual(f(x, n, 3), want)
        # B (next_power_of_2) and EVEN select the variants
        self.assertEqual(f.traces, 4)
        self.assertEqual(f.eager, 0)

    def test_families_by_contract(self):
        f = HostTraceReplay(add)
        x = torch.randn(1000, device="cuda")
        self.assertEqual(f(x), x + 3)
        self.assertEqual(f(x, 4), x + 4)
        self.assertEqual(f(x, 4), x + 4)
        h = x.half()
        self.assertEqual(f(h), h + 3)
        self.assertEqual(f(x, 3), x + 3)
        self.assertEqual(len(f._families), 3)
        self.assertEqual((f.traces, f.replays), (3, 2))

    def test_a_branch_the_trace_takes_otherwise_declines(self):
        # an int argument is a SymInt in the trace, which fails isinstance:
        # the trace's operator calls are not the warm-up's
        def fn(x, n):
            return x * n if isinstance(n, int) else x - 1

        x = torch.randn(64, device="cuda")
        f = HostTraceReplay(fn)
        for n in (2, 10, 2):
            self.assertEqual(f(x, n), fn(x, n))
        # the first decline may be a lazy initialization, and a decline is of its ints
        self.assertEqual((f.traces, f.replays, f.eager), (3, 0, 3))
        self.assertIn("fn's call 0 is aten.sub.Tensor", f.declines[0])
        self.assertIn("at the warm-up aten.mul.Tensor", f.declines[0])
        # an int the trace takes symbolically is not a new witness
        g = HostTraceReplay(lambda x, n: x[:n] * 2)
        for n in (2, 3, 4):
            self.assertEqual(g(x, n), x[:n] * 2)
        self.assertEqual((g.traces, g.replays, g.eager), (1, 2, 0))

    def test_result_structures(self):
        def fn(x):
            y = add(x)
            return y, x, y

        f = HostTraceReplay(fn)
        for n in (1000, 3000, 3000):
            x = torch.randn(n, device="cuda")
            y, x2, y2 = f(x)
            self.assertIs(x2, x)
            self.assertIs(y2, y)
            self.assertEqual(y, x + 3)

        f = HostTraceReplay(lambda x: [add(x)])
        for n in (1000, 3000):
            x = torch.randn(n, device="cuda")
            out = f(x)
            self.assertIsInstance(out, list)
            self.assertEqual(out[0], x + 3)

        def inplace(x):
            n = x.numel()
            _add[(triton.cdiv(n, 128),)](x, x, n, 1, B=128)

        f = HostTraceReplay(inplace)
        x = torch.zeros(1000, device="cuda")
        for i in range(1, 4):
            self.assertIsNone(f(x))
            self.assertEqual(x, torch.full_like(x, float(i)))
        self.assertEqual((f.traces, f.replays), (1, 2))

    def test_chained_launches(self):
        def fn(x):
            t = add(x, 1)
            return add(t, 2)

        f = HostTraceReplay(fn)
        for n in (1000, 3000, 5000, 3000):
            x = torch.randn(n, device="cuda")
            self.assertEqual(f(x), x + 3)
        self.assertEqual((f.traces, f.replays), (1, 3))

    def test_a_zeroed_workspace_is_a_patched_memset(self):
        def fn(x):
            n = x.numel()
            w = torch.empty(n, device=x.device).zero_()
            _add[(triton.cdiv(n, 128),)](w, w, n, 2, B=128)
            _add[(triton.cdiv(n, 128),)](w, w, n, 1, B=128)
            return w

        f = HostTraceReplay(fn)
        sizes = (1000, 3000, 3000, 1000, 5000)
        outs = [f(torch.randn(n, device="cuda")) for n in sizes]
        for n, out in zip(sizes, outs):
            self.assertEqual(out, torch.full((n,), 3.0, device="cuda"), atol=0, rtol=0)
        self.assertEqual((f.traces, f.eager), (1, 0))
        (variant,) = f.variants
        memset = torch._C._host_trace_held_images(variant.native)[0]
        self.assertEqual(memset[1], 5000 * 4)

    def test_int_outputs(self):
        f = HostTraceReplay(lambda x: (add(x), x.shape[0] * 2, 7))
        for n in (1000, 3000, 5000):
            x = torch.randn(n, device="cuda")
            y, m, k = f(x)
            self.assertEqual(y, x + 3)
            self.assertEqual((m, k), (2 * n, 7))
            self.assertIs(type(m), int)
        self.assertEqual((f.traces, f.eager, f.declines), (1, 0, []))

    def test_replays_on_the_current_stream(self):
        f = HostTraceReplay(add)
        f(torch.randn(1000, device="cuda"))
        s = torch.cuda.Stream()
        x = torch.randn(3000, device="cuda")
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            y = f(x)
        torch.cuda.current_stream().wait_stream(s)
        self.assertEqual(y, x + 3)

    def test_a_call_under_capture_is_captured_as_eager(self):
        f = HostTraceReplay(add)
        x = torch.randn(1000, device="cuda")
        f(x)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            y = f(x)
        x.fill_(1.0)
        g.replay()
        self.assertEqual(y, torch.full_like(x, 4.0))
        self.assertEqual((f.traces, f.replays, f.eager), (1, 0, 1))
        self.assertIn("an outer capture", f.declines[0])

    def test_output_views(self):
        def views(x):
            y, n = add(x), x.numel()
            return y[1:], y.view(-1, 2), x[: n // 2]

        f = HostTraceReplay(views)
        for n in (1000, 2000, 4096, 1000):
            x = torch.randn(n, device="cuda")
            a, b, c = f(x)
            self.assertEqual(a, (x + 3)[1:])
            self.assertEqual(b, (x + 3).view(-1, 2))
            self.assertEqual(c, x[: n // 2])
            self.assertIs(a._base, b._base)
            self.assertIs(c._base, x)
        self.assertEqual(f.eager, 0)
        self.assertGreater(f.replays, 1)

        def reshaped(x):
            y = add(x)
            return y, y.view(-1, 10)

        f = HostTraceReplay(reshaped)
        for n in (1000, 3000):
            x = torch.randn(n, device="cuda")
            y, v = f(x)
            self.assertEqual(v, (x + 3).view(-1, 10))
            self.assertIs(v._base, y)
        self.assertEqual((f.replays, f.eager), (1, 0))

    def test_keyword_arguments(self):
        f = HostTraceReplay(add)
        x = torch.randn(1000, device="cuda")
        self.assertEqual(f(x, s=4), x + 4)
        self.assertEqual(f(x=x, s=4), x + 4)
        self.assertEqual((f.traces, f.replays, f.eager), (1, 1, 0))
        g = HostTraceReplay(lambda x, *, s: add(x, s))
        self.assertEqual(g(x, s=4), x + 4)
        self.assertEqual((g.traces, g.eager), (0, 1))
        self.assertIn("keyword arguments", g.declines[0])

    def test_a_variant_retains_no_tensors(self):
        f = HostTraceReplay(add)
        refs = []
        # the warm-up call, then a replay
        for n in (1000, 3000):
            x = torch.randn(n, device="cuda")
            y = f(x)
            refs += [weakref.ref(x), weakref.ref(y)]
            del x, y
        self.assertEqual((f.traces, f.replays), (1, 1))
        self.assertEqual([r() for r in refs], [None] * 4)
        (v,) = f.variants
        self.assertIsNone(v.tape.warm_up_result)
        self.assertFalse(any(isinstance(a, torch.Tensor) for a in v.tape.args))

    def test_only_plain_tensors_are_traced(self):
        class Sub(torch.Tensor):
            pass

        f = HostTraceReplay(add)
        x = torch.randn(1000, device="cuda")
        for t in (x.as_subclass(Sub), torch.nn.Parameter(x)):
            self.assertEqual(f(t), x + 3)
            self.assertEqual(type(f(t)), type(add(t)))
        self.assertEqual((f.traces, f.replays), (2, 0))
        self.assertIn("arg0 is a Sub; only torch.Tensor", f.declines[0])

    def test_storage_reads_decline(self):
        def fn(x):
            return add(x, x.untyped_storage().nbytes() // 4 - x.numel())

        f = HostTraceReplay(fn)
        base = torch.randn(2000, device="cuda")
        for x in (torch.randn(1000, device="cuda"), base[:1000], base[:1000]):
            self.assertEqual(f(x), fn(x))
        self.assertEqual((f.replays, len(f.declines)), (0, 1))
        self.assertIn("storage of a traced tensor", f.declines[0])

    def test_an_address_read_as_an_int_declines(self):
        # a guard would pin the input's address: declined, not baked
        def fn(x):
            return add(x, int(x.data_ptr()) % 7)

        f = HostTraceReplay(fn)
        base = torch.randn(4096 + 64, device="cuda")
        for i in range(4):
            x = base[i : i + 4096]
            self.assertEqual(f(x), fn(x))
        self.assertEqual(f.replays, 0)
        for why in f.declines:
            self.assertIn("the host read a traced address as an int", why)

    def test_an_output_that_requires_grad_declines(self):
        def fn(x):
            y = torch.empty(x.shape, device=x.device, requires_grad=True)
            _add[(8,)](x, y, x.numel(), 3, B=128)
            return y

        f = HostTraceReplay(fn)
        x = torch.randn(1000, device="cuda")
        for _ in range(2):
            y = f(x)
            self.assertTrue(y.requires_grad)
            self.assertEqual(y.detach(), x + 3)
        self.assertEqual((f.replays, f.eager), (0, 2))
        why = "output 0 requires grad; a replay's outputs do not"
        self.assertEqual(f.declines, [f"host_trace: {why} (declined)"])

    def test_an_argument_that_requires_grad_declines_under_grad_mode(self):
        def fn(x):
            return torch.tanh(x) * 2

        f = HostTraceReplay(fn)
        x = torch.randn(1000, device="cuda")
        f(x)
        f(x)
        xg = x.clone().requires_grad_()
        y = f(xg)
        self.assertTrue(y.requires_grad)
        self.assertEqual(y, fn(xg))
        with torch.no_grad():
            self.assertEqual(f(xg), fn(x))
        self.assertEqual((f.traces, f.eager), (3, 1))
        why = "arg0 requires grad under grad mode; a replay records no autograd graph"
        self.assertEqual(f.declines, [f"host_trace: {why} (declined)"])

    def test_a_traced_tensor_fn_keeps_declines(self):
        kept = []

        def fn(x):
            y = x * 2
            kept.append(y)
            return y + 1

        f = HostTraceReplay(fn, check_escapes=True)
        x = torch.randn(1000, device="cuda")
        for _ in range(3):
            self.assertEqual(f(x), x * 2 + 1)
        self.assertEqual((f.traces, f.replays, f.eager), (1, 0, 3))
        why = "fn kept a traced tensor past the trace, in a list; fn must return the tensors it makes and store none"
        self.assertEqual(f.declines, [f"host_trace: {why} (declined)"])
        # the trace's append is now a view of the warm-up's
        self.assertEqual([type(t) for t in kept], [torch.Tensor] * 4)
        self.assertEqual(kept[1].data_ptr(), kept[0].data_ptr())
        self.assertEqual(kept[1] + 1, x * 2 + 1)

    def test_a_traced_tensor_in_an_argument_object_declines(self):
        class Cache:
            def __init__(self):
                self.kv = []

        def fn(x, cache):
            cache.kv.append(x.sin())
            return cache.kv[-1] * 2

        f = HostTraceReplay(fn, check_escapes=True)
        x, cache = torch.randn(1000, device="cuda"), Cache()
        f(x, cache)
        self.assertEqual((f.traces, f.replays), (1, 0))
        self.assertIn("fn kept a traced tensor past the trace", f.declines[0])
        self.assertEqual(cache.kv[1] * 2, x.sin() * 2)

    def test_a_kept_argument_is_the_argument(self):
        kept = []

        def fn(x):
            kept.append(x)
            return x * 2

        f = HostTraceReplay(fn, check_escapes=True)
        x = torch.randn(1000, device="cuda")
        self.assertEqual(f(x), x * 2)
        self.assertIn("fn kept a traced tensor past the trace", f.declines[0])
        self.assertEqual(type(kept[1]), torch.Tensor)
        self.assertEqual(kept[1].data_ptr(), x.data_ptr())

    def test_a_cycle_fn_leaves_is_not_kept(self):
        def fn(x):
            y = x * 2
            box = {"y": y}
            box["self"] = box
            return y + 1

        f = HostTraceReplay(fn, check_escapes=True)
        x = torch.randn(1000, device="cuda")
        for _ in range(3):
            self.assertEqual(f(x), x * 2 + 1)
        self.assertEqual((f.traces, f.replays, f.eager), (1, 2, 0))

    @parametrize("check_global_state", [False, True])
    def test_first_use_initialization_in_the_warm_up_is_the_traced_state(self, check_global_state):
        def fn(x):
            torch.backends.cuda.enable_math_sdp(False)
            return two_step(x)

        f = HostTraceReplay(fn, check_global_state=check_global_state)
        x = torch.randn(1000, device="cuda")
        try:
            for _ in range(3):
                self.assertEqual(f(x), two_step(x))
        finally:
            torch.backends.cuda.enable_math_sdp(True)
        self.assertEqual((f.traces, f.replays, f.eager), (1, 2, 0))

    @mock.patch("torch.cuda._host_trace.raise_unexpected", False)
    def test_errors_of_the_trace_decline(self):
        def fn(x):
            # the traced input is a subclass: an error only in the trace
            if type(x) is not torch.Tensor:
                raise ValueError("traced")
            return add(x)

        f = HostTraceReplay(fn)
        for n in (1000, 1000, 1024):
            x = torch.randn(n, device="cuda")
            self.assertEqual(f(x), x + 3)
        self.assertEqual((f.traces, f.eager), (2, 3))
        self.assertIn("ValueError: traced", f.declines[0])
        # the warm-up's own error is the call's
        g = HostTraceReplay(lambda x: add(x.no_such_attribute))
        with self.assertRaises(AttributeError):
            g(x)
        # an error of the lowering or capture declines
        h = HostTraceReplay(add)
        boom = RuntimeError("boom")
        with mock.patch("torch.cuda._host_trace_replay.lower_tape", side_effect=boom):
            self.assertEqual(h(x), x + 3)
        self.assertIn("RuntimeError: boom", h.declines[0])
        self.assertTrue(h.declines[0].endswith("(declined)"))
        # its assertion is a bug, not a decline
        bug = AssertionError("bug")
        with mock.patch("torch.cuda._host_trace_replay.lower_tape", side_effect=bug):
            with self.assertRaisesRegex(AssertionError, "bug"):
                HostTraceReplay(add)(x)
        # the suites' strict mode raises the others too
        with mock.patch("torch.cuda._host_trace.raise_unexpected", True):
            with self.assertRaisesRegex(ValueError, "traced"):
                HostTraceReplay(fn)(x)
            with mock.patch("torch.cuda._host_trace_replay.lower_tape", side_effect=boom):
                with self.assertRaisesRegex(RuntimeError, "boom"):
                    HostTraceReplay(add)(x)

    def test_an_int_outside_int64_declines(self):
        f = HostTraceReplay(lambda x, s: add(x, s % 7))
        x = torch.randn(1000, device="cuda")
        for s in (2**64 + 10, 3, 2**64 + 10, -(2**63) - 1, 4):
            self.assertEqual(f(x, s), x + s % 7)
        # the second big int misses, a new class declines, 4 replays the second trace
        self.assertEqual((f.traces, f.replays), (3, 1))
        self.assertIn("outside int64", f.declines[0])

    def test_a_misaligned_allocation_runs_eagerly(self):
        f = HostTraceReplay(add)
        f(torch.randn(1000, device="cuda"))
        x = torch.randn(3000, device="cuda")
        # a pool whose segments are 16 bytes past a 512-byte boundary
        backing = torch.empty(64 << 20, dtype=torch.uint8, device="cuda")
        at = [backing.data_ptr() + 16]

        def alloc(size, device, stream):
            p, at[0] = at[0], at[0] + (size + 511) // 512 * 512
            return p

        c_alloc = ctypes.CFUNCTYPE(
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p
        )(alloc)
        c_free = ctypes.CFUNCTYPE(
            None, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p
        )(lambda *args: None)
        allocator = torch._C._cuda_customAllocator(
            ctypes.cast(c_alloc, ctypes.c_void_p).value,
            ctypes.cast(c_free, ctypes.c_void_p).value,
        )
        pool = torch.cuda.MemPool(allocator)
        with torch.cuda.use_mem_pool(pool):
            self.assertEqual(f(x), x + 3)
        torch.cuda.synchronize()
        del pool
        self.assertEqual((f.replays, f.eager), (0, 1))
        self.assertIn("not 256-byte aligned", f.declines[0])
        self.assertEqual(f(x), x + 3)
        self.assertEqual(f.replays, 1)

    @parametrize("op", ["pad", "pad_crop", "diag_embed", "block_diag", "unbind_copy", "slice_backward", "new_full"])
    def test_a_composite_runs_as_its_parts(self, op):
        # a CompositeExplicitAutograd(NonFunctional) op of no route of its own: its body's parts
        fn = {
            "pad": lambda x: torch.nn.functional.pad(x, (1, 2, 0, 1), value=3.0),
            "pad_crop": lambda x: torch.nn.functional.pad(x, (-1, 2)),
            "diag_embed": lambda x: torch.diag_embed(x),
            "block_diag": lambda x: torch.block_diag(x, x[1:]),
            "unbind_copy": lambda x: torch.unbind_copy(x, 1),
            "slice_backward": lambda x: torch.ops.aten.slice_backward(x, [x.shape[0], x.shape[1] + 3], 1, 1, x.shape[1] + 1, 1),
            "new_full": lambda x: x.new_full((2, 3), 2.5),
        }[op]
        f = HostTraceReplay(fn)
        for n in (6, 6, 6, 9, 9):
            x = torch.randn(4, n, device="cuda")
            self.assertEqual(f(x), fn(x), atol=0, rtol=0)
        self.assertEqual(f.declines, [])
        self.assertGreaterEqual(f.replays, 2)
        for v in f.variants:
            self.assertFalse([r for _, r in v.tape.launches if isinstance(r, EagerCall)])

    def test_a_composite_with_an_eager_part_is_one_eager_step(self):
        # cat.out, stack.out's part, has no route: the parts are not split around
        # it, and a plain eager part adds no reason
        x, o = torch.randn(4, 6, device="cuda"), torch.empty(2, 4, 6, device="cuda")
        tape = trace(lambda x, o: torch.stack([x, x], out=o), (x, o))
        (call,) = [r for _, r in tape.launches]
        self.assertIsInstance(call, EagerCall)
        self.assertEqual(call.reason, "aten.stack.out's pointwise host declines: its tensors is not a traced tensor")

    def test_an_empty_copy_keeps_its_sources_strides(self):
        # an empty tensor is contiguous (c10), so empty_like keeps a broadcast's zero strides
        f = HostTraceReplay(lambda t: t.to(torch.float32))
        for n in (0, 0, 0, 5, 0, 7):
            t = torch.ones((), dtype=torch.bfloat16, device="cuda").expand(1, n, 3)
            out, want = f(t), t.to(torch.float32)
            self.assertEqual((out.shape, out.stride()), (want.shape, want.stride()))
            self.assertEqual(out, want, atol=0, rtol=0)
        self.assertEqual(f.declines, [])
        self.assertEqual(f.traces, 2)

    @parametrize("op", ["prod", "var", "std", "var_mean", "std_mean"])
    def test_a_zero_element_reduction_replays_as_its_outputs_fill(self, op):
        fn = {
            "prod": lambda t: t.prod(1),
            "var": lambda t: t.var(1),
            "std": lambda t: t.std(1),
            "var_mean": lambda t: torch.var_mean(t, 1),
            "std_mean": lambda t: torch.std_mean(t, 1),
        }[op]
        f = HostTraceReplay(fn)
        for n, k in ((3, 0), (5, 0), (3, 0), (4, 2), (3, 0)):
            t = torch.randn(n, k, device="cuda")
            self.assertEqual(f(t), fn(t), atol=0, rtol=0, equal_nan=True)
        self.assertEqual(f.declines, [])
        self.assertEqual(f.traces, 2)
        for v in f.variants:
            self.assertFalse([r for _, r in v.tape.launches if isinstance(r, EagerCall)])

    @parametrize("nbytes", [257, 4096, 64 << 20])
    def test_a_device_copy_of_any_size_replays(self, nbytes):
        # its capture copies within a caching-allocator block as large as the copy
        f = HostTraceReplay(lambda x: x.clone())
        for _ in range(3):
            x = torch.randint(0, 256, (nbytes,), dtype=torch.uint8, device="cuda")
            self.assertEqual(f(x), x, atol=0, rtol=0)
        self.assertEqual((f.traces, f.replays, f.eager), (1, 2, 0))
        self.assertEqual(f.declines, [])
        (v,) = f.variants
        self.assertEqual([type(lo) for lo in v.captured.lowered.launches], [LoweredMemcpy])

    @parametrize("nbytes", [257, 4096, 64 << 20])
    def test_a_device_copy_of_any_alignment_or_allocation_replays(self, nbytes):
        from cuda.bindings import runtime

        from torch.cuda._utils import _check_cuda_bindings

        class Blob:
            def __init__(self, n):
                self.ptr = int(_check_cuda_bindings(runtime.cudaMalloc(n)))
                self.__cuda_array_interface__ = {"shape": (n,), "typestr": "|u1", "data": (self.ptr, False), "version": 3}

            def __del__(self):
                runtime.cudaFree(self.ptr)

        blobs = [Blob(nbytes + 3) for _ in range(2)]
        f = HostTraceReplay(lambda x: x.clone())
        srcs = [torch.empty(nbytes + 3, dtype=torch.uint8, device="cuda")[off:][:nbytes] for off in (0, 1, 3, 0)]
        # a cudaMalloc operand after allocator ones is refused by an expandable-segments capture (DESIGN_NOTES 36)
        if "expandable_segments:True" not in (os.environ.get("PYTORCH_CUDA_ALLOC_CONF") or ""):
            srcs += [torch.as_tensor(b, device="cuda")[off:][:nbytes] for b, off in zip(blobs, (0, 1))]
        for x in srcs:
            x.copy_(torch.randint(0, 256, (nbytes,), dtype=torch.uint8, device="cuda"))
            self.assertEqual(f(x), x, atol=0, rtol=0)
        self.assertEqual((f.traces, f.replays, f.eager), (1, len(srcs) - 1, 0))
        self.assertEqual(f.declines, [])

    def test_an_autotune_miss_retraces(self):
        def tuned(x):
            y, n = torch.empty_like(x), x.numel()
            _add_tuned[lambda meta: (triton.cdiv(n, meta["B"]),)](x, y, n, 3)
            return y

        f = _host_trace_replay.HostTraceReplay(tuned)
        # the first call, eager, tunes its size; a later size's warm-up tunes
        # it: the benchmark's calls are not the trace's, so that call runs
        # eagerly and the next traces (a witness decline is retried once per
        # class, and logged)
        for n in (1000, 1000, 2000, 2000, 2000):
            x = torch.randn(n, device="cuda")
            self.assertEqual(f(x), x + 3)
        self.assertEqual((f.traces, f.replays, f.eager, f.declines), (3, 1, 2, []))

    @unittest.skipIf(torch.cuda.device_count() < 2, "requires two GPUs")
    def test_a_capture_on_the_tapes_device(self):
        def on_device(x):
            with torch.cuda.device(x.device):
                return add(x)

        f = HostTraceReplay(on_device)
        f(torch.randn(1000, device="cuda:1"))
        x = torch.randn(1000, device="cuda:1")
        g = torch.cuda.CUDAGraph()
        with torch.cuda.device(1), torch.cuda.stream(torch.cuda.Stream(1)):
            g.capture_begin()
            # the current device's stream is not capturing, the tape's is
            with torch.cuda.device(0):
                y = f(x)
            g.capture_end()
        x.fill_(1.0)
        g.replay()
        self.assertEqual(y, torch.full_like(x, 4.0))
        self.assertEqual((f.replays, f.eager), (0, 1))
        self.assertIn("a capture on cuda:1", f.declines[0])


@unittest.skipIf(not TEST_CUDA, "requires CUDA")
@requires_cuda_python_bindings
@unittest.skipIf(not has_triton(), "requires triton")
class TestChainReplay(TestCase):
    def test_a_chain_matches_eager(self):
        f = HostTraceReplay(matmul_chain)
        w = torch.randn(16, 8, device="cuda")
        side = torch.cuda.Stream()
        for n in (1024, 4096, 2048, 4096, 1024):
            x = torch.randn(n, device="cuda")
            self.assertEqual(f(x, w), matmul_chain(x, w))
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                y = f(x, w)
            torch.cuda.current_stream().wait_stream(side)
            self.assertEqual(y, matmul_chain(x, w))
        self.assertEqual((f.traces, f.eager), (len(f.variants), 0))
        for v in f.variants:
            self.assertEqual(len(v.captured.segments), 2)

    @mock.patch("torch.cuda._host_trace.raise_unexpected", False)
    def test_a_segment_that_fails_to_capture_runs_eagerly(self):
        # the segment after the matmul fails at its second launch, the mul: the
        # call traces again with the mul an eager call, and the rest replays
        def fn(x, w):
            return (matmul_chain(x, w) * 2).sin()

        launch, calls = _host_trace_capture._launch, []

        def fail_second(launches, *args):
            calls.append(len(launches))
            if len(calls) == 2:
                launch(launches[:1], *args)
                raise RuntimeError("injected")
            return launch(launches, *args)

        f = HostTraceReplay(fn)
        w = torch.randn(16, 8, device="cuda")
        with mock.patch.object(_host_trace_capture, "_launch", fail_second):
            for _ in range(3):
                x = torch.randn(1024, device="cuda")
                self.assertEqual(f(x, w), fn(x, w), atol=0, rtol=0)
        self.assertEqual((f.traces, f.replays, f.eager), (2, 2, 0))
        self.assertEqual(calls, [1, 3, 1, 1, 1])
        (v,) = f.variants
        self.assertEqual(len(v.captured.segments), 3)
        self.assertEqual(len(f.declines), 1)
        self.assertIn("RuntimeError: injected", f.declines[0])
        # the suites' strict mode raises it
        calls.clear()
        with mock.patch("torch.cuda._host_trace.raise_unexpected", True):
            with mock.patch.object(_host_trace_capture, "_launch", fail_second):
                with self.assertRaisesRegex(RuntimeError, "injected"):
                    HostTraceReplay(fn)(x, w)

    @mock.patch("torch.cuda._host_trace.raise_unexpected", True)
    def test_a_segment_that_fails_to_verify_runs_eagerly(self):
        # a decline is local in strict mode too; the Triton launch of the
        # segment after the matmul runs eagerly
        verify, calls = _host_trace_capture._verify, []

        def refuse_second(*args):
            calls.append(None)
            if len(calls) == 2:
                raise declined("refused")
            return verify(*args)

        f = HostTraceReplay(matmul_chain)
        w = torch.randn(16, 8, device="cuda")
        with mock.patch.object(_host_trace_capture, "_verify", refuse_second):
            for _ in range(3):
                x = torch.randn(1024, device="cuda")
                self.assertEqual(f(x, w), matmul_chain(x, w), atol=0, rtol=0)
        self.assertEqual((f.traces, f.replays, f.eager), (2, 2, 0))
        (v,) = f.variants
        self.assertEqual(len(v.captured.segments), 1)
        self.assertEqual(len(f.declines), 1)
        self.assertIn("the capture of _add failed: refused", f.declines[0])

    def test_an_inplace_eager_op(self):
        def fn(x):
            x.mul_(2)
            y = add(x)
            y.add_(1)
            return add(y)

        f = HostTraceReplay(fn)
        for n in (1000, 1000, 1000, 3000):
            x = torch.randn(n, device="cuda")
            want = x.clone()
            self.assertEqual(f(x), fn(want))
            self.assertEqual(x, want)
        self.assertGreater(f.replays, 0)
        self.assertEqual(f.eager, 0)

    def test_an_out_argument_that_is_also_an_input(self):
        def fn(x):
            y, z = add(x), add(x, 1)
            torch.add(y, 1, out=y)
            torch.mul(z, z, out=z)
            return add(y), add(z)

        f = HostTraceReplay(fn)
        for n in (1000, 1000, 3000):
            x = torch.randn(n, device="cuda")
            self.assertEqual(f(x), fn(x))
        self.assertEqual((f.replays, f.eager), (2, 0))

    def test_inplace_foreach_ops(self):
        def fn(x):
            a, b = add(x), add(x, 1)
            torch._foreach_mul_([a, b], 2.0)
            torch._foreach_add_([a, b], [b, a])
            return add(a), add(b)

        f = HostTraceReplay(fn)
        for n in (1000, 1000, 3000):
            x = torch.randn(n, device="cuda")
            self.assertEqual(f(x), fn(x))
        self.assertEqual((f.replays, f.eager), (2, 0))

    def test_an_eager_op_with_none_outputs(self):
        def fn(x, w):
            stats = x[:, :1]
            mask = [True, False, False]
            dx, dw, db = torch.ops.aten.native_layer_norm_backward(x, x, [64], stats, stats, w, w, mask)
            return add(dx)

        f = HostTraceReplay(fn)
        for n in (8, 8, 16):
            x, w = torch.randn(n, 64, device="cuda"), torch.randn(64, device="cuda")
            self.assertEqual(f(x, w), fn(x, w))
        self.assertEqual((f.replays, f.eager, f.declines), (2, 0, []))

    @unittest.skipIf(not PLATFORM_SUPPORTS_CUDNN_ATTENTION, "requires cuDNN attention")
    def test_cudnn_attention(self):
        # its outputs include Nones and ints
        def fn(q, k, v):
            out, lse = torch.ops.aten._scaled_dot_product_cudnn_attention(q, k, v, None, True)[:2]
            return add(out), lse

        f = HostTraceReplay(fn)
        for s in (128, 128, 256):
            q, k, v = (torch.randn(2, 4, s, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3))
            self.assertEqual(f(q, k, v), fn(q, k, v))
        self.assertEqual((f.replays, f.eager, f.declines), (2, 0, []))

    @unittest.skipIf(not PLATFORM_SUPPORTS_CUDNN_ATTENTION, "requires cuDNN attention")
    def test_cudnn_attention_of_a_shape_it_does_not_take_raises(self):
        # a key of another head dim than the query: eager's cuDNN raises (and
        # at a second such call in the process crashes), a replay would run it
        def fn(q, k, v):
            return add(torch.nn.functional.scaled_dot_product_attention(q, k, v))

        f = HostTraceReplay(fn)
        q, k, v = (torch.randn(2, 4, 64, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3))
        f(q, k, v), f(q, k, v)
        k = torch.randn(2, 4, 64, 32, device="cuda", dtype=torch.bfloat16)
        with self.assertRaises(RuntimeError):
            f(q, k, v)
        torch.cuda.synchronize()

    @parametrize("symbolic", ["ir", "sympy"])
    def test_masked_attention_picks_eagers_backend(self, symbolic):
        # the selector's mask broadcast check reads sizes; under the trace it
        # must decide by the hint, as eager does, not decline the fused backends
        def fn(q, k, v, mask):
            return add(torch.nn.functional.scaled_dot_product_attention(q, k, v, mask))

        f = HostTraceReplay(fn)
        with mock.patch("torch.cuda._host_trace.symbolic", symbolic):
            for s in (128, 128, 256):
                q, k, v = (torch.randn(2, 4, s, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3))
                mask = torch.randn(2, 1, s, s, device="cuda", dtype=torch.bfloat16)
                self.assertEqual(f(q, k, v, mask), fn(q, k, v, mask), atol=0, rtol=0)
        self.assertEqual(f.eager, 0)

    def test_a_view_of_an_eager_output(self):
        def fn(x):
            z = add(x).sin()
            return z.view(-1, 4)[1:], z

        f = HostTraceReplay(fn)
        for n in (1024, 1024, 4096):
            x = torch.randn(n, device="cuda")
            view, z = f(x)
            self.assertEqual((view, z), fn(x))
            self.assertEqual(view.data_ptr(), z.data_ptr() + 16)
        self.assertEqual((f.replays, f.eager), (2, 0))

    def test_a_triton_fallback_in_a_chain(self):
        def fn(x):
            y = add(x)
            z = torch.empty_like(y)
            n = y.numel()
            _add[(triton.cdiv(n, 128),)](y, z, n, 2, B=128, launch_pdl=True)
            return add(z, 1)

        f = HostTraceReplay(fn)
        with mock.patch("torch.cuda._host_trace_replay.trace_structured") as log:
            for n in (1000, 1000, 3000, 1, 1):
                x = torch.randn(n, device="cuda")
                self.assertEqual(f(x), x + 6)
        self.assertEqual(f.eager, 0)
        self.assertGreater(len(f.variants), 1)
        (why,) = f.triton_fallbacks
        self.assertIn("programmatic-dependent", why)
        self.assertEqual(log.call_count, 1)
        self.assertEqual(log.call_args.kwargs["payload_fn"](), why)

    def test_a_fake_kernels_wrong_metadata_declines_at_the_warm_up(self):
        def fn(x):
            return add(transposed_fake(add(x)))

        f = HostTraceReplay(fn)
        for shape in ((8, 16), (8, 16), (4, 16)):
            x = torch.randn(shape, device="cuda")
            self.assertEqual(f(x), x + 6)
        self.assertEqual((f.traces, f.replays, f.eager, f.variants), (2, 0, 3, []))
        self.assertRegex(
            f.declines[0],
            r"transposed_fake.default output 0 .*\(16, 1\).* at the warm-up; its fake kernel predicted .*\(1, 8\)",
        )
        # the trace at (4, 16) warms up too
        self.assertRegex(f.declines[1], r"output 0 .*\(16, 1\).* at the warm-up; its fake kernel predicted .*\(1, 4\)")

    def test_a_fake_kernels_wrong_layout_at_a_later_trace_declines(self):
        def fn(x):
            return add(transposed_fake_at_4(add(x)))

        f = HostTraceReplay(fn)
        x = torch.randn(8, 16, device="cuda")
        self.assertEqual(f(x), x + 6)
        # the trace at (4, 16) warms up too: its warm-up finds the disagreement
        y = torch.randn(4, 16, device="cuda")
        self.assertEqual(f(y), y + 6)
        self.assertEqual(f(y), y + 6)
        self.assertEqual(f(x), x + 6)
        self.assertEqual((f.traces, f.replays, f.eager, len(f.variants)), (2, 1, 2, 1))
        self.assertRegex(
            f.declines[0],
            r"transposed_fake_at_4.default output 0 .*\(16, 1\).* at the warm-up; its fake kernel predicted .*\(1, 4\)",
        )

    def test_a_traced_kernels_fake_is_not_read(self):
        # the trace follows the op's Python kernel, not its fake kernel
        def fn(x):
            return add(traced_lying_fake(add(x)))

        f = HostTraceReplay(fn)
        for shape in ((8, 16), (8, 16), (4, 16)):
            x = torch.randn(shape, device="cuda")
            self.assertEqual(f(x), x + 6)
        self.assertEqual((f.eager, f.declines), (0, []))

    def test_a_size_1_dims_stride_is_not_compared(self):
        def fn(x):
            return add(size_1_stride(add(x)))

        f = HostTraceReplay(fn)
        # n=8's warm-up predicts the size-1 dim's stride as 128; n=4 writes 64
        for n in (8, 8, 4, 4, 8, 4):
            x = torch.randn(n, 1, 16, device="cuda")
            self.assertEqual(f(x), x + 6)
        self.assertEqual((f.traces, f.replays, f.eager, f.declines), (1, 5, 0, []))

    def test_warm_up_calls_pair_by_operator_and_arguments(self):
        first = [True]

        def fn(x):
            if first.pop() if first else False:
                x.t().clone()  # a lazy initialization: the warm-up only
            y = add(x)
            z = y.t().clone().sin().sin().sin()
            return add(z), add(y.clone())

        f = HostTraceReplay(fn)
        for _ in range(3):
            x = torch.randn(8, 16, device="cuda")
            z, y = f(x)
            self.assertEqual((z, y), ((x + 3).t().sin().sin().sin() + 3, x + 6))
            self.assertEqual(z.stride(), (1, 16))
        # the lazy initialization is a call the trace does not make: the first
        # call runs eagerly and the next traces (the witness decline's retry)
        self.assertEqual((f.traces, f.replays, f.eager, f.declines), (2, 1, 1, []))


@unittest.skipIf(not TEST_CUDA, "requires CUDA")
@requires_cuda_python_bindings
@unittest.skipIf(not has_triton(), "requires triton")
class TestReplayMemory(TestCase):
    def test_the_plan_follows_the_tape(self):
        def fn(x):
            torch.empty_like(x)
            return two_step(x)

        f = HostTraceReplay(fn)
        x = torch.randn(1000, device="cuda")
        self.assertEqual(f(x), x + 4)
        (v,) = f.variants
        lowered = v.captured.lowered
        seqs = [a.seq for a in lowered.allocations]
        first, second = (lo.seq for lo in lowered.launches)
        # the unused allocation's last use is its own allocation, add(x)'s the
        # second launch, which reads it; both are the run's temporaries
        temporaries = ((0, seqs[0], seqs[0]), (1, seqs[1], second))
        # eager order: the unused one is freed at once, add(x)'s after the
        # second launch, which follows the output's allocation; x's last use
        # is the run
        run = StepMemory((2,), temporaries, (), (0, -1, 1, 2, -2), (0,))
        after = StepMemory((), (), ())
        self.assertEqual(v.memory, MemoryPlan((2,), (run, after)))
        self.assertLess(seqs[1], first)
        self.assertLess(seqs[2], second)
        g = HostTraceReplay(fn, memory="run_buffer")
        self.assertEqual(g(x), x + 4)
        run = StepMemory((2,), temporaries, (), (), (0,))
        self.assertEqual(g.variants[0].memory, MemoryPlan((2,), (run, after)))
        # the unused allocation is dead when add(x)'s is made: they share bytes
        self.assertEqual(place(temporaries, [4000, 4000]), ([0, 0], 4096))

    def test_placement(self):
        # (allocation, seq, last use): 1 dies before 2 is made, 0 outlives both
        temporaries = ((0, 0, 5), (1, 1, 2), (2, 3, 6), (3, 4, 7))
        offsets, total = place(temporaries, [1024, 500, 512, 2048])
        self.assertEqual(offsets, [0, 1024, 1024, 1536])
        # the size is the live peak (0, 2 and 3 at seq 4), each in 512-byte blocks
        self.assertEqual(total, 1024 + 512 + 2048)
        # an empty temporary takes no bytes
        self.assertEqual(place(temporaries, [0, 0, 512, 0]), ([0, 0, 0, 0], 512))
        # the first gap that fits, below the live ones
        temporaries = ((0, 0, 9), (1, 1, 2), (2, 3, 9), (3, 4, 9))
        self.assertEqual(
            place(temporaries, [512, 1024, 512, 512])[0], [0, 512, 512, 1024]
        )

    def measure(self, fn, x):
        # an earlier test's cyclic garbage (a declined trace's) must not be
        # freed inside the measured call
        gc.collect()
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        y = fn(x)
        torch.cuda.synchronize()
        after = torch.cuda.memory_allocated()
        return y, after - before, torch.cuda.max_memory_allocated() - before

    def test_retained_and_peak_memory_follow_eager(self):
        f = HostTraceReplay(two_step)
        # the trace, two replays of the same shape, a smaller size, a new
        # variant (1 specializes), the first again
        for n in (1 << 20, 1 << 20, 1 << 20, 4096, 1, 1 << 18, 1 << 20):
            x = torch.randn(n, device="cuda")
            want, eager_retained, eager_peak = self.measure(two_step, x)
            y, retained, peak = self.measure(f, x)
            self.assertEqual(y, want)
            # only the output stays allocated, and only while it is referenced
            self.assertEqual(retained, eager_retained)
            if f.traces + f.replays > 1:  # the warm-up call is eager's
                self.assertLessEqual(peak, eager_peak)
            before = torch.cuda.memory_allocated()
            del y
            self.assertEqual(torch.cuda.memory_allocated(), before - eager_retained)
        self.assertEqual((f.traces, f.eager), (2, 0))

    def test_host_step_keeps_the_runs_peak(self):
        # the host step runs first, out of tape order (HF LayerDrop's
        # torch.rand([])): the run before it still frees its temporaries
        def fn(x):
            for _ in range(4):
                x = add(add(x))
            torch.rand([])
            return add(lies_when_small(x))

        f = HostTraceReplay(fn)
        for _ in range(3):
            x = torch.randn(1 << 20, device="cuda")
            want, _, eager_peak = self.measure(fn, x)
            y, _, peak = self.measure(f, x)
            self.assertEqual(y, want)
        self.assertEqual((f.traces, f.replays, f.eager), (1, 2, 0))
        self.assertLessEqual(peak, eager_peak)

    def test_call_boxed_drops_an_argument_after_its_last_use(self):
        # a caller that gives its references up (Inductor's boxed call): x is
        # freed after the first run, before the clone, as eager frees it
        def fn(x):
            return add(lies_when_small(add(x)))

        def boxed(args):
            return add(lies_when_small(add(args.pop())))

        f = HostTraceReplay(fn)
        for _ in range(3):
            want, eager_retained, eager_peak = self.measure(boxed, [torch.randn(1 << 20, device="cuda")])
            y, retained, peak = self.measure(f.call_boxed, [torch.randn(1 << 20, device="cuda")])
            self.assertEqual(retained, eager_retained)
        self.assertEqual((f.replays, f.eager), (2, 0))
        self.assertEqual(peak, eager_peak)
        self.assertLess(peak, self.measure(f, torch.randn(1 << 20, device="cuda"))[2])
        # an argument an output views is not dropped
        g = HostTraceReplay(lambda x: (add(lies_when_small(add(x))), x[1:]))
        for _ in range(3):
            x = torch.randn(1 << 20, device="cuda")
            args = [x]
            y, view = g.call_boxed(args)
            self.assertEqual(args, [])
            self.assertIs(view._base, x)
        self.assertEqual(g.replays, 2)

    @parametrize("memory", ["eager", "run_buffer"])
    def test_an_output_after_a_dead_temporary(self, memory):
        # add(x) is dead once add(add(x)) is made, and eager's output takes
        # its bytes; a run buffer holds both temporaries through the replay
        def three_step(x):
            return add(add(add(x)), 1)

        def requests(fn, x):
            before = torch.cuda.memory_stats()["allocation.all.allocated"]
            y = fn(x)
            return y, torch.cuda.memory_stats()["allocation.all.allocated"] - before

        f = HostTraceReplay(three_step, memory=memory)
        x = torch.randn(1 << 20, device="cuda")
        for _ in range(3):
            want, eager_retained, eager_peak = self.measure(three_step, x)
            y, retained, peak = self.measure(f, x)
            self.assertEqual(y, want)
            self.assertEqual(retained, eager_retained)
        self.assertEqual(f.replays, 2)
        eager_requests, replay_requests = requests(three_step, x)[1], requests(f, x)[1]
        if memory == "eager":
            self.assertEqual(peak, eager_peak)
            self.assertEqual(replay_requests, eager_requests)
        else:
            # the documented bound: the outputs made after the temporaries' peak
            self.assertLessEqual(peak, eager_peak + y.nbytes)
            self.assertEqual(replay_requests, 2)

    def test_auto_memory_takes_a_run_buffer_within_its_margin(self):
        # a run buffer holds add(x) and add(add(x)) together and the output
        # beside them: the output's bytes over eager's peak. At 128 MiB both
        # that and the buffer pass the 64 MiB margin
        def three_step(x):
            return add(add(add(x)), 1)

        for n, eager_order in ((1 << 20, False), (1 << 25, True)):
            f = HostTraceReplay(three_step, memory="auto")
            x = torch.randn(n, device="cuda")
            for _ in range(3):
                self.assertEqual(f(x), x + 7)
            (v,) = f.variants
            self.assertEqual(any(s.order for s in v.memory.steps), eager_order)

    def test_a_run_splits_where_an_eager_calls_output_dies(self):
        # the clone is dead once add reads it, but it is held to the end of
        # its run; the run splits before the output's allocation
        def fn(x):
            return add(add(lies_when_small(add(x))), 1)

        f = HostTraceReplay(fn)
        x = torch.randn(1 << 20, device="cuda")
        for _ in range(3):
            want, eager_retained, eager_peak = self.measure(fn, x)
            y, retained, peak = self.measure(f, x)
            self.assertEqual(y, want)
            self.assertEqual(retained, eager_retained)
        self.assertEqual((f.replays, f.eager), (2, 0))
        self.assertEqual(peak, eager_peak)
        (v,) = f.variants
        runs = [s for s in v.captured.lowered.steps if isinstance(s, range)]
        self.assertEqual([len(r) for r in runs], [1, 1, 1])

    def test_a_run_splits_where_a_freed_argument_dies(self):
        # x is dead once add(x) is made, but a boxed call drops it only after
        # its run; the run splits before the output's allocation
        def fn(x):
            return add(add(x), 1)

        f = HostTraceReplay(fn, freed_arguments=(0,))
        n = 1 << 22
        for _ in range(3):
            args = [torch.randn(n, device="cuda")]
            want = args[0] + 4
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            y = f.call_boxed(args)
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() - before
            self.assertEqual(y, want)
        self.assertEqual((f.replays, f.eager), (2, 0))
        self.assertEqual(peak, 4 * n)
        (v,) = f.variants
        runs = [s for s in v.captured.lowered.steps if isinstance(s, range)]
        self.assertEqual([len(r) for r in runs], [1, 1])

    def test_distinct_shapes_reserve_as_eager(self):
        # each replay frees and reuses blocks as eager does, so the cache
        # grows as eager's over a stream of distinct sizes
        def three_step(x):
            return add(add(add(x)), 1)

        f = HostTraceReplay(three_step)
        f(torch.randn(4096, device="cuda"))
        sizes = [4096 * k for k in (700, 3, 1500, 64, 2900, 900, 5000, 1, 2048, 777)]

        def reserved(fn):
            gc.collect()
            torch.cuda.empty_cache()
            for n in sizes:
                self.assertEqual(fn(torch.ones(n, device="cuda")), torch.full((n,), 8.0))
            return torch.cuda.memory_reserved()

        eager_reserved = reserved(three_step)
        self.assertLessEqual(reserved(f), eager_reserved)
        self.assertEqual((f.traces, f.replays, f.eager), (1, len(sizes), 0))

    @unittest.skipIf(
        not hasattr(torch._C, "_HostTraceVariant"), "requires the native path"
    )
    def test_a_released_segment_falls_back_to_a_run_buffer(self):
        # a is freed before c is made, while the graph that uses it is not
        # queued yet; c's allocation over the cap releases every free cached
        # segment, a's with it, so the run's temporaries go in a run buffer
        def fn(x, z):
            b = add(add(x))
            return b, add(z)

        mib = 1 << 20
        f = HostTraceReplay(fn)
        # sizes no segment an earlier test left partly used can hold, and each
        # its own segment (over 10 MiB, 2 MiB multiples: no rounding or split)
        x = torch.randn(48 * mib, device="cuda")  # 192 MiB
        z = torch.randn(64 * mib, device="cuda")  # 256 MiB
        f(x, z)
        gc.collect()
        torch.cuda.empty_cache()
        torch.empty(40 * mib, device="cuda")  # a cached 160 MiB segment
        # c over the cap: without the release a, b and c need 640 MiB more;
        # releasing 160 + 192, then b, c and the run buffer need 480
        cap = torch.cuda.memory_reserved() + 544 * mib
        total = torch.cuda.get_device_properties(x.device).total_memory
        retries = torch.cuda.memory_stats()["num_alloc_retries"]
        buffered = torch._C._host_trace_buffered_runs()
        torch.cuda.set_per_process_memory_fraction(cap / total)
        try:
            b, c = f(x, z)
        finally:
            torch.cuda.set_per_process_memory_fraction(1.0)
        self.assertEqual(torch.cuda.memory_stats()["num_alloc_retries"], retries + 1)
        self.assertEqual(torch._C._host_trace_buffered_runs(), buffered + 1)
        self.assertEqual(b, x + 6)
        self.assertEqual(c, z + 3)
        self.assertEqual((f.traces, f.replays, f.eager), (1, 1, 0))

    def test_an_output_held_across_replays_keeps_its_values(self):
        f = HostTraceReplay(two_step)
        xs = [torch.randn(n, device="cuda") for n in (4096, 4096, 2048, 8192, 4096)]
        ys = [f(x) for x in xs]
        del ys[1]  # its bytes may be the next output's, as eager's
        ys.insert(1, f(xs[1]))
        for x, y in zip(xs, ys):
            self.assertEqual(y, x + 4)
        self.assertEqual(len({y.data_ptr() for y in ys}), len(ys))
        self.assertEqual((f.traces, f.eager), (1, 0))

    def test_an_eager_call_between_runs(self):
        def fn(x):
            return add(add(x).sin())

        f = HostTraceReplay(fn)
        for n in (1 << 20, 1 << 20, 1 << 18, 1 << 20):
            x = torch.randn(n, device="cuda")
            want, eager_retained, eager_peak = self.measure(fn, x)
            y, retained, peak = self.measure(f, x)
            self.assertEqual(y, want)
            self.assertEqual(retained, eager_retained)
            if f.replays:
                self.assertEqual(peak, eager_peak)
        self.assertEqual((f.traces, f.eager), (1, 0))

    def test_a_replay_on_another_stream(self):
        f = HostTraceReplay(two_step)
        x = torch.randn(4096, device="cuda")
        self.assertEqual(f(x), x + 4)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            y = f(x)
        torch.cuda.current_stream().wait_stream(s)
        self.assertEqual(y, x + 4)
        self.assertEqual((f.traces, f.replays, f.eager), (1, 1, 0))


def held_images(f):
    return [torch._C._host_trace_held_images(v.native) for v in f.variants]


def three_outputs(x):
    y = add(x)
    return y, add(y, 1), torch.empty_like(x).zero_()


@unittest.skipIf(not TEST_CUDA, "requires CUDA")
@requires_cuda_python_bindings
@unittest.skipIf(not has_triton(), "requires triton")
@unittest.skipIf(not hasattr(torch._C, "_HostTraceVariant"), "requires the native path")
class TestNativeReplay(TestCase):
    def test_the_native_nodes_hold_launch_images(self):
        # 12 variants (dtype, n % 16, a 16-byte aligned input or not), two
        # sizes each, so a hit patches or not; every allocation is an output,
        # so the serving variant's nodes must hold launch_images at the
        # outputs' addresses
        f = HostTraceReplay(three_outputs)
        # a variant per trace: the founders' rows give the images
        f._fold = f._redispatch = lambda *args: False
        inputs = []
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for n in (1000, 1024, 3000, 4096):
                base = torch.randn(n + 1, device="cuda", dtype=dtype)
                inputs += [base[:n], base[1:]]
        for x in (*inputs, *reversed(inputs), *inputs[::3], *inputs):
            replays = f.replays
            outs = f(x)
            self.assertEqual(outs[1], x + 4)
            self.assertEqual(outs[2], torch.zeros_like(x), atol=0, rtol=0)
            if f.replays == replays:
                continue  # a trace's outputs are its run's, not the nodes'
            lowered, values = next(
                (v.captured.lowered, vals)
                for v in f._families[_contract((x,))]
                if (vals := v.captured.lowered.evaluate((x,))) is not None
            )
            bases = [0] * len(lowered.allocations)
            for out, ref in zip(outs, lowered.outputs):
                bases[ref[1]] = out.data_ptr()
            want = []
            for lo in lowered.launches:
                if isinstance(lo, LoweredMemset):
                    (dst,) = launch_slots(lo, values, bases)
                    want.append((dst, values[lo.width], values[lo.height], values[lo.pitch]))
                else:
                    grid = tuple(values[r] for r in lo.grid)
                    want.append((grid, tuple(launch_images(lo, launch_slots(lo, values, bases)))))
            variant = next(v for v in f.variants if v.captured.lowered is lowered)
            self.assertEqual(torch._C._host_trace_held_images(variant.native), want)
        self.assertEqual(len(f.variants), 12)
        self.assertEqual((f.traces, f.eager), (12, 0))

    def test_an_error_after_a_setter_leaves_the_node_unknown(self):
        f = HostTraceReplay(two_step)
        for n in (1000, 3000):
            f(torch.randn(n, device="cuda"))
        self.assertEqual(len(f.variants), 1)
        before = held_images(f)[0]
        x = torch.randn(5000, device="cuda")
        torch._C._host_trace_fail_after_setter(0)
        try:
            with self.assertRaisesRegex(RuntimeError, "injected after a setter"):
                f(x)
        finally:
            torch._C._host_trace_fail_after_setter(-1)
        # the first launch's node was patched: its bytes are unknown; the
        # second was not reached
        self.assertEqual(held_images(f)[0], [None, before[1]])
        self.assertEqual(f(x), x + 4)
        self.assertNotIn(None, held_images(f)[0])
        self.assertEqual((f.traces, f.replays, f.eager), (1, 2, 0))

    def test_memcpy_and_memset_nodes_are_set_every_replay(self):
        # such a node holds the allocation its address was in when set, which
        # a free and reallocation at the same address leaves stale; a kernel
        # node's parameters are bytes, set only when they change
        def fn(x):
            y = x.clone()
            return y, add(y), torch.empty_like(x).zero_()

        f = HostTraceReplay(fn)
        x = torch.randn(4096, device="cuda")
        want = (x, x + 3, torch.zeros_like(x))
        self.assertEqual(f(x), want)
        # the first replay too, though its nodes hold the trace's values
        for _ in range(3):
            sets = torch._C._host_trace_memory_node_sets()
            self.assertEqual(f(x), want)
            self.assertEqual(torch._C._host_trace_memory_node_sets(), sets + 2)
        self.assertEqual((f.traces, f.replays), (1, 3))
        (before,) = held_images(f)
        # the memcpy is set first, then (the kernel unchanged) the memset
        for n, held in ((0, [None, *before[1:]]), (1, [*before[:2], None])):
            torch._C._host_trace_fail_after_setter(n)
            try:
                with self.assertRaisesRegex(RuntimeError, "injected after a setter"):
                    f(x)
            finally:
                torch._C._host_trace_fail_after_setter(-1)
            self.assertEqual(held_images(f), [held])
            self.assertEqual(f(x), want)
        self.assertEqual((f.traces, f.eager), (1, 0))

    @parametrize("check_global_state", [False, True])
    def test_the_native_key_partitions_as_the_contract(self, check_global_state):
        f = HostTraceReplay(two_step, check_global_state=check_global_state)
        x = torch.randn(4, 6, device="cuda")
        c = torch.randn(4, 6, device="cuda", dtype=torch.complex64)
        tensors = [x, x[:2], x.t(), x.half(), x[0], x._neg_view(), c, c.conj()]
        tensors += [c.conj()._neg_view(), x.cpu(), x.to_sparse()]
        cases = [(t,) for t in tensors]
        cases += [(x, s) for s in (3, 5, 3.0, True, 2.0, -2.0, None, "a")]
        keys = []
        for grad in (True, False):
            with torch.set_grad_enabled(grad):
                keys += [(f._native_key(a), _contract(a, check_global_state)) for a in cases]
        # a None key (an argument with no native kind) takes the Python path
        for (k1, c1), (k2, c2) in itertools.product(keys, keys):
            if k1 is not None and k2 is not None:
                self.assertEqual(k1 == k2, c1 == c2, msg=f"{c1} {c2}")

    @parametrize("check_global_state", [False, True])
    def test_global_state_is_in_the_contract_on_request(self, check_global_state):
        f = HostTraceReplay(two_step, check_global_state=check_global_state)
        x = torch.randn(1000, device="cuda")
        f(x)
        f(x)
        prior = torch.backends.cuda.matmul.fp32_precision
        try:
            torch.backends.cuda.matmul.fp32_precision = "ieee" if prior == "tf32" else "tf32"
            self.assertEqual(f(x), two_step(x))
        finally:
            torch.backends.cuda.matmul.fp32_precision = prior
        self.assertEqual((f.traces, f.replays), (2, 1) if check_global_state else (1, 2))

    def test_grad_mode_splits_an_argument_that_requires_grad(self):
        # autograd records the call under grad mode only: a replay traced under
        # no_grad does not serve it (without check_global_state too)
        def fn(x):
            return (x * 2).sin()

        f = HostTraceReplay(fn)
        x = torch.randn(1000, device="cuda", requires_grad=True)
        with torch.no_grad():
            f(x)
            f(x)
        y = f(x)
        self.assertIsNotNone(y.grad_fn)
        self.assertEqual(y, fn(x))
        self.assertEqual((f.traces, f.replays, f.eager), (2, 1, 1))

    def test_a_hit_runs_no_python(self):
        f = HostTraceReplay(two_step)
        x = torch.randn(3000, device="cuda")
        f(x)
        frames = []
        sys.setprofile(
            lambda frame, event, arg: event == "call" and frames.append(frame)
        )
        try:
            y = f(x)
        finally:
            sys.setprofile(None)
        self.assertEqual(frames, [])
        self.assertEqual(y, x + 4)
        self.assertEqual((f.traces, f.replays), (1, 1))

    def test_an_aten_step_runs_no_python(self):
        # an OpOverload step goes through the boxed dispatcher, its outputs
        # checked in C++
        f = HostTraceReplay(matmul_chain)
        x, w = torch.randn(8, 64, device="cuda"), torch.randn(64, 32, device="cuda")
        f(x, w)
        (v,) = f.variants
        self.assertEqual(v.native.python_steps, 0)
        frames = []
        sys.setprofile(
            lambda frame, event, arg: event == "call" and frames.append(frame)
        )
        try:
            y = f(x, w)
        finally:
            sys.setprofile(None)
        self.assertEqual(frames, [])
        self.assertEqual(y, matmul_chain(x, w))
        self.assertEqual((f.traces, f.replays, v.native.python_calls), (1, 1, 0))

    def test_two_threads_miss_and_hit_one_entry(self):
        # each thread's sizes miss (a trace, a new variant) and hit, while the
        # other holds the entry
        f = HostTraceReplay(two_step)
        errors = []

        def run(sizes):
            try:
                for _ in range(20):
                    for n in sizes:
                        x = torch.randn(n, device="cuda")
                        torch.testing.assert_close(f(x), x + 4)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=run, args=(s,))
            for s in ((1000, 3000, 1024), (2048, 5000, 1001))
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        self.assertEqual(f.eager, 0)
        self.assertGreater(f.replays, 0)

    def test_replay_hooks_run_on_a_native_hit(self):
        f = HostTraceReplay(two_step)
        x = torch.randn(3000, device="cuda")
        f(x)
        (v,) = f.variants
        (segment,) = v.captured.segments
        seen = []
        handle = torch.cuda.graphs.register_graph_replay_start_hook(seen.append)
        try:
            self.assertEqual(f(x), x + 4)
        finally:
            handle.remove()
        segment.graph.register_replay_end_hook(seen.append)
        self.assertEqual(f(x), x + 4)
        self.assertEqual(seen, [segment.graph, segment.graph])
        self.assertEqual(f.replays, 2)


instantiate_parametrized_tests(TestHostTraceReplay)
instantiate_parametrized_tests(TestChainReplay)
instantiate_parametrized_tests(TestReplayMemory)
instantiate_parametrized_tests(TestNativeReplay)


def setUpModule():
    import torch.cuda._host_trace_capture as capture

    capture.raise_trace_disagreements = True
    torch.cuda._host_trace.raise_unexpected = True


if __name__ == "__main__":
    run_tests()
