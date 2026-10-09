# Owner(s): ["module: cuda graphs"]

import builtins
import ctypes
import dataclasses
import gc
import itertools
import os
import random
import sys
import threading
import unittest
import weakref
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

import sympy

import torch
from torch._prims.rng_prims import graphsafe_run_with_rng_state
from torch._subclasses.fake_tensor import FakeTensor
from torch.cuda import (
    _host_trace_capture,
    _host_trace_ir as _ir,
    _host_trace_native,
    _host_trace_replay,
    _host_trace_tape,
)
from torch.cuda._host_trace import declined
from torch.cuda._host_trace_capture import launch_images, launch_slots
from torch.cuda._host_trace_launch import KernelLaunch
from torch.cuda._host_trace_lower_tape import FoldRefused, LoweredMemcpy, LoweredMemset
from torch.cuda._host_trace_memory import _AFTER, _layout, MemoryPlan, place, StepMemory
from torch.cuda._host_trace_program import compile_program, IntegerProgram
from torch.cuda._host_trace_replay import _exact_class, Handback
from torch.cuda._host_trace_tape import argument_contract, current_trace, EagerCall, trace, TrustedInputs
from torch.cuda._utils import _check_cuda_bindings
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_CUDNN_ATTENTION, PLATFORM_SUPPORTS_MEM_EFF_ATTENTION
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    requires_cuda_python_bindings,
    run_tests,
    subtest,
    TEST_CUDA,
    TestCase,
)
from torch.utils._sympy.value_ranges import ValueRanges
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

    @triton.jit
    def _second_row(x_ptr, y_ptr, N, sxm, B: tl.constexpr):
        n = tl.program_id(0) * B + tl.arange(0, B)
        tl.store(y_ptr + n, tl.load(x_ptr + sxm + n, mask=n < N), mask=n < N)


# host code that looks sizes up in tables, as SGLang's GEMM choices do
_TACTICS = {(8, 32): 2.0, (24, 32): 5.0}


@torch.library.custom_op("host_trace_test::by_tactic", mutates_args=())
def by_tactic(x: torch.Tensor) -> torch.Tensor:
    return x * _TACTICS.get(tuple(x.shape), 3.0)


@by_tactic.register_fake
def _(x):
    return torch.empty_like(x)


@torch.library.custom_op("host_trace_test::by_large", mutates_args=())
def by_large(x: torch.Tensor) -> torch.Tensor:
    return x * {True: 2.0, False: 3.0}[x.shape[0] >= 16]


@by_large.register_fake
def _(x):
    return torch.empty_like(x)


def add(x, s=3):
    y = torch.empty_like(x)
    n = x.numel()
    _add[(triton.cdiv(n, 128),)](x, y, n, s, B=128)
    return y


def two_step(x):
    return add(add(x), 1)


def five_step(x):
    return add(add(add(add(x))), 1)


def matmul_chain(x, w):
    # triton -> mm (eager; matmul folds the batch with _unsafe_view) -> triton
    return add(torch.matmul(add(x).view(4, -1, w.shape[0]), w), 1)


# CUDA kernels the trace does not follow with library_impls off: eager calls,
# whose metadata comes from their fake kernels
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


# eager calls: one that calls REENTERED[0], one whose output views its input
_lib.define("calls_entry(Tensor x) -> Tensor")
_lib.impl("calls_entry", lambda x: REENTERED[0](x), "CUDA")
_lib.define("aliases(Tensor x) -> Tensor")
_lib.impl("aliases", lambda x: x.view_as(x), "CUDA")
calls_entry = torch.ops.host_trace_test.calls_entry
aliases = torch.ops.host_trace_test.aliases


@torch.library.register_fake("host_trace_test::calls_entry")
def _(x):
    return torch.empty_like(x)


@torch.library.register_fake("host_trace_test::aliases")
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


# ops registered as vLLM's and SGLang's direct_register_custom_op registers
# them: Library.define of the inferred schema and impl("CUDA"), no custom_op
def _direct_scale(x: torch.Tensor, s: float) -> torch.Tensor:
    return x * s + 1


def _direct_scale_into(out: torch.Tensor, x: torch.Tensor) -> None:
    torch.mul(x, 2, out=out)


_direct_lib = torch.library.Library("host_trace_direct", "FRAGMENT")
for _name, _fn, _mutates in (("scale", _direct_scale, []), ("scale_into", _direct_scale_into, ["out"])):
    _direct_lib.define(_name + torch.library.infer_schema(_fn, mutates_args=_mutates))
    _direct_lib.impl(_name, _fn, "CUDA")
_direct_lib._register_fake("scale", lambda x, s: torch.empty_like(x))
_direct_lib._register_fake("scale_into", lambda out, x: None)
# a Library.impl kernel that calls REENTERED[0]
_direct_lib.define("calls_reentered(Tensor x) -> Tensor")
_direct_lib.impl("calls_reentered", lambda x: REENTERED[0](x), "CUDA")
_direct_lib._register_fake("calls_reentered", lambda x: torch.empty_like(x))
calls_reentered = torch.ops.host_trace_direct.calls_reentered


@torch.library.custom_op("host_trace_test::calls_reentered_op", mutates_args=())
def calls_reentered_op(x: torch.Tensor) -> torch.Tensor:
    return REENTERED[0](x)


@calls_reentered_op.register_fake
def _(x):
    return torch.empty_like(x)


# a library op whose trace needs state set up outside the trace's capture (the
# FlashInfer fork's kernel modules), which its caller's wrapper sets up only
# under a dispatch mode (a trace's warm-up), inside a Library.impl body
FIRST_USE: list = []


@torch.library.custom_op("host_trace_test::set_up_at_warm_up", mutates_args=())
def set_up_at_warm_up(x: torch.Tensor) -> torch.Tensor:
    if current_trace() is not None and not FIRST_USE:
        raise current_trace().decline("not set up at the warm-up")
    return x * 2


@set_up_at_warm_up.register_fake
def _(x):
    return torch.empty_like(x)


def _sets_up_under_a_mode(x: torch.Tensor) -> torch.Tensor:
    if torch._C._len_torch_dispatch_stack() and current_trace() is None:
        FIRST_USE.append(x.numel())
    return set_up_at_warm_up(x)


_direct_lib.define("sets_up_inside(Tensor x) -> Tensor")
_direct_lib.impl("sets_up_inside", _sets_up_under_a_mode, "CUDA")
_direct_lib._register_fake("sets_up_inside", lambda x: torch.empty_like(x))

# per body kind: the op whose body calls REENTERED[0], and its re-entry switch
REENTRY_OPS = {"Library.impl": (calls_reentered, "library_impls_reentry"), "custom_op": (torch.ops.host_trace_test.calls_reentered_op, "custom_op_reentry")}

def scale2d(x):
    M, N = x.shape
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(M, 16), triton.cdiv(N, 32))
    _scale2d[grid](x, y, M, N, x.stride(0), x.stride(1), BM=16, BN=32)
    return y


def tactic_add(x):
    # a library's kernel choice in Python, a chain of comparisons on the row
    # count (SGLang's GEMM tactic, prep C2): classes 2..16, 17..32, 48, 65..
    # and the rest
    m = x.shape[0]
    B = 128 if m <= 16 else 256 if m <= 32 else 512 if m == 48 else 1024 if m <= 64 else 2048
    y = torch.empty_like(x)
    n = x.numel()
    _add[(triton.cdiv(n, B),)](x, y, n, 1, B=B)
    return y


@torch.library.custom_op("host_trace_test::tactic_op", mutates_args=())
def tactic_op(x: torch.Tensor) -> torch.Tensor:
    return tactic_add(x)


@tactic_op.register_fake
def _(x):
    return torch.empty_like(x)


def split_add(x):
    # a library's split on the row count (a split reduction's): one launch up
    # to 16 rows, else two through a temporary, twice the size past 64
    m, n = x.shape[0], x.numel()
    y = torch.empty_like(x)
    if m <= 16:
        _add[(triton.cdiv(n, 128),)](x, y, n, 3, B=128)
        return y
    t = torch.empty(n if m <= 64 else 2 * n, device=x.device, dtype=x.dtype)
    _add[(triton.cdiv(n, 128),)](x, t, n, 1, B=128)
    _add[(triton.cdiv(n, 128),)](t, y, n, 2, B=128)
    return y


@torch.library.custom_op("host_trace_test::split_op", mutates_args=())
def split_op(x: torch.Tensor) -> torch.Tensor:
    return split_add(x)


@split_op.register_fake
def _(x):
    return torch.empty_like(x)


@torch.library.custom_op("host_trace_test::enqueues_past_32", mutates_args=())
def enqueues_past_32(x: torch.Tensor) -> torch.Tensor:
    # a launch the tracer does not see past 32 rows
    if x.shape[0] > 32:
        torch.cuda._sleep(1)
    return tactic_add(x)


@enqueues_past_32.register_fake
def _(x):
    return torch.empty_like(x)


# an extension op (sgl_kernel's): a CUDA kernel the tracer cannot see into
_ext_lib = torch.library.Library("host_trace_test", "FRAGMENT")
_ext_lib.define("tactic_ext(Tensor x) -> Tensor")
_ext_lib.impl("tactic_ext", tactic_add, "CUDA")
torch.library.register_fake("host_trace_test::tactic_ext", lambda x: torch.empty_like(x), lib=_ext_lib)


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

    @parametrize("fresh", [True, False])
    def test_fresh_stack_chunk_switch(self, fresh):
        on_ballast = []

        def fn(x):
            frame = sys._getframe()
            while frame is not None and frame.f_code.co_name != "ballast":
                frame = frame.f_back
            on_ballast.append(frame is not None)
            return add(x) * 2

        with mock.patch.object(torch.cuda._host_trace, "fresh_stack_chunk", fresh):
            f = HostTraceReplay(fn)
            for n in (1000, 3000, 5000):
                x = torch.randn(n, device="cuda")
                self.assertEqual(f(x), (x + 3) * 2, atol=0, rtol=0)
            self.assertEqual((f.traces, f.replays, f.eager), (1, 2, 0))
            # a miss runs its warm-up and trace on the ballast, trace() its trace only
            ballast = fresh and sys.version_info < (3, 13)
            self.assertEqual(on_ballast, [ballast, ballast])
            trace(fn, (torch.randn(1000, device="cuda"),))
            self.assertEqual(on_ballast[2:], [False, ballast])

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

    def test_a_size_looked_up_in_an_op_is_the_ops_guard(self):
        # hash() of a SymInt pins its value: inside an op, a pin the op owns
        f = HostTraceReplay(lambda x: by_tactic(x) + 1)
        for n in (8, 8, 24, 24, 12, 8):
            x = torch.randn(n, 32, device="cuda")
            self.assertEqual(f(x), x * _TACTICS.get((n, 32), 3.0) + 1)
        self.assertEqual((f.traces, f.replays, f.redispatches, f.eager, f.declines), (1, 5, 2, 0, []))
        (v,) = f.variants
        pins = [o for g, o in zip(v.tape.guards, v.tape.owners) if str(g).startswith("Eq(") and str(g).endswith(", 8)")]
        self.assertEqual(pins, [0])

    def test_a_comparison_as_a_dict_key_guards_the_comparison(self):
        def model(x):
            return x * {True: 2.0, False: 3.0}[x.shape[0] >= 16] + 1

        for fn, traces, redispatches in ((model, 2, 0), (lambda x: by_large(x) + 1, 1, 1)):
            f = HostTraceReplay(fn)
            for n in (8, 12, 24, 40, 8):
                x = torch.randn(n, 32, device="cuda")
                self.assertEqual(f(x), x * (2.0 if n >= 16 else 3.0) + 1)
            self.assertEqual((f.traces, f.replays, f.redispatches, f.eager, f.declines), (traces, 5 - traces, redispatches, 0, []))

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

    def test_a_trusted_misaligned_input_redispatches(self):
        # trust vouches for layouts, never addresses: the launch owns its alignment guard
        f = HostTraceReplay(add, trusted=TrustedInputs((((1000,), (1,)),)))
        base = torch.randn(4000, device="cuda")
        for k in (0, 1, 4, 2, 8, 3):
            x = base[k : k + 1000]
            self.assertEqual(f(x), x + 3, atol=0, rtol=0)
        self.assertEqual((f.traces, f.replays, f.redispatches, f.eager), (1, 5, 1, 0))

    def test_a_trusted_view_at_a_size_dependent_offset_redispatches(self):
        def fn(x, n):
            t = x[: 2 * n] * 3
            return add(t[n : 2 * n])

        s = sympy.Symbol("s0", integer=True, positive=True)
        f = HostTraceReplay(fn, trusted=TrustedInputs((((8192,), (1,)), s), {s: ValueRanges(1, 4096)}))
        x = torch.randn(8192, device="cuda")
        for n in (1000, 1001, 1004, 1003, 1002, 1008):
            self.assertEqual(f(x, n), fn(x, n), atol=0, rtol=0)
        # t[n:]'s alignment (n % 4) and n % 16 (1008) each select a redispatched entry
        self.assertEqual((f.traces, f.replays, f.redispatches, f.eager), (1, 5, 2, 0))

    @parametrize("shared", [False, True])
    def test_a_guard_two_ops_share_redispatches_both(self, shared):
        # both launches specialize on n % 16: graph-level, a flip traces; with
        # shared_op_guards each op's own, a flip dispatches both again
        def fn(x):
            return add(x), add(x)

        with mock.patch.object(torch.cuda._host_trace, "shared_op_guards", shared):
            f = HostTraceReplay(fn)
            for n in (1000, 3000, 1024, 2048, 1000):
                x = torch.randn(n, device="cuda")
                self.assertEqual(f(x), (x + 3, x + 3), atol=0, rtol=0)
        self.assertEqual((f.traces, f.redispatches), (1, 1) if shared else (2, 0))

    @parametrize("how", ["model", "op", "op_shared"])
    def test_a_kernel_choice_in_an_op_redispatches(self, how):
        # four layers of a tactic choice on the row count over 2..130: in the
        # model's Python its guards are the graph's (a trace per class); in a
        # custom op the op's, but four layers recording one guard make it the
        # graph's again unless shared_op_guards keeps it each op's
        layer = tactic_add if how == "model" else tactic_op

        def fn(x):
            for _ in range(4):
                x = layer(x)
            return x

        shared = how == "op_shared"
        with mock.patch.object(torch.cuda._host_trace, "shared_op_guards", shared):
            f = HostTraceReplay(fn)
            for m in range(2, 131):
                x = torch.randn(m, 64, device="cuda")
                self.assertEqual(f(x), fn(x), atol=0, rtol=0)
        self.assertEqual((f.traces, f.redispatches, f.eager), (1, 4, 0) if shared else (5, 0, 0))

    def test_a_redispatch_renames_ir_records(self):
        # an IR variant's entry guards are records in its own context, renamed
        # from the fresh trace's with no sympy export of its guards
        def fn(x):
            for _ in range(4):
                x = tactic_op(x)
            return x

        exports = []
        export = _ir.Env.guards

        def counted(env):
            exports.append(env)
            return export.fget(env)

        with mock.patch.object(torch.cuda._host_trace, "shared_op_guards", True):
            f = HostTraceReplay(fn)
            x = torch.randn(2, 64, device="cuda")
            self.assertEqual(f(x), fn(x), atol=0, rtol=0)
            with mock.patch.object(_ir.Env, "guards", property(counted)):
                for m in (20, 40, 100, 130):
                    x = torch.randn(m, 64, device="cuda")
                    self.assertEqual(f(x), fn(x), atol=0, rtol=0)
        self.assertEqual((f.traces, f.redispatches, exports), (1, 3, []))
        lowering = f.variants[0].captured.lowered.lowering
        records = [g for _, guards in lowering.entry_guards for g in guards]
        self.assertTrue(records)
        symbols = lowering.lowering.ctx.symbols
        for node, _ in records:
            self.assertIsInstance(node, _ir.Node)
            self.assertLessEqual(node.free_symbols, symbols.keys())

    @parametrize("case", ["fill_expanded", "zero_flipped", "fill_", "zero_", "add_", "copy_"])
    @parametrize("returns", ["base", "slice", "both"])
    def test_an_inplace_op_on_a_slice_of_an_eager_output_is_its_argument(self, case, returns):
        # sum (int32) and flip (complex64) run as eager steps; an in-place op on a slice of their output
        # returns its argument, not a new eager output: a replay that took it for one returned the slice for
        # the base (the base's tensor replaced by the slice's). The bf16 copy of a complex input is the
        # fuzzer's (it makes zero_ an eager step on the symbolic TensorIterator build)
        def fn(a0, w):
            if case == "zero_flipped":
                v0 = a0.t().flip(0)
                s = v0[v0.shape[0] // 2 :]
            else:
                v0 = a0.sum(1)
                torch.add(a0, 1, out=a0)
                s = v0.unsqueeze(0).expand(3, *v0.shape)[:, :1] if case == "fill_expanded" else v0[v0.shape[0] // 4 : v0.shape[0] // 2]
            if case in ("fill_expanded", "fill_"):
                s.fill_(3)
            elif case in ("zero_", "zero_flipped"):
                s.zero_()
            elif case == "add_":
                s.add_(2)
            else:
                s.copy_(w[: s.shape[0]])
            out = {"base": (v0,), "slice": (s,), "both": (v0, s)}[returns]
            return (*out, a0.to(torch.bfloat16)) if case == "zero_flipped" else out

        f = HostTraceReplay(fn)
        dtype = torch.complex64 if case == "zero_flipped" else torch.int32
        for m in (128, 128, 64):
            a = torch.randint(-8, 9, (m, 18), device="cuda").to(dtype)
            w = torch.randint(-8, 9, (m,), device="cuda", dtype=torch.int64)
            got, want = f(a.clone(), w), fn(a.clone(), w)
            for x, y in zip(torch.utils._pytree.tree_leaves(got), torch.utils._pytree.tree_leaves(want)):
                self.assertEqual((x.shape, x.stride(), x.storage_offset()), (y.shape, y.stride(), y.storage_offset()))
                self.assertEqual(x, y, atol=0, rtol=0)
        self.assertEqual(f.eager, 0)
        self.assertGreater(f.replays, 0)

    def test_a_native_rejection_declines_the_trace(self):
        # a spec the native variant rejects is the lowering's bug: the suites raise it, a user's call declines
        # (noted) and runs eagerly, never an AssertionError into user code
        def fn(x):
            return x * 2 + 1

        reject = mock.patch.object(torch._C, "_HostTraceVariant", side_effect=ValueError("a site's record 1"))
        f = HostTraceReplay(fn)
        with reject, mock.patch.object(torch.cuda._host_trace, "raise_unexpected", False):
            for _ in range(2):
                x = torch.randn(64, device="cuda")
                self.assertEqual(f(x), fn(x), atol=0, rtol=0)
        self.assertEqual((f.traces, f.replays, f.eager), (1, 0, 2))
        self.assertIn("the native variant rejects the lowering (a site's record 1)", " ".join(f.declines))
        f = HostTraceReplay(fn)
        with reject, mock.patch.object(torch.cuda._host_trace, "raise_unexpected", True):
            with self.assertRaisesRegex(AssertionError, "the native variant rejects: a site's record 1"):
                f(torch.randn(64, device="cuda"))

    @parametrize("what", ["row", "entry"])
    def test_a_native_rejection_at_a_miss_declines(self, what):
        # a key's row (a new GEMM key) or a redispatch's entry (tactic_op's new class) the native variant rejects:
        # the suites raise it; a user's call drops the variant, noted, and traces again
        def fn(x, w):
            return (x @ w + 1) if what == "row" else tactic_op(x) * 2

        w = torch.randn(64, 64, device="cuda")
        method = "add_row" if what == "row" else "add_entry"
        for strict in (False, True):
            f = HostTraceReplay(fn)
            for m in (8, 8):
                x = torch.randn(m, 64, device="cuda")
                self.assertEqual(f(x, w), fn(x, w), atol=0, rtol=0)
            x = torch.randn(96 if what == "row" else 20, 64, device="cuda")
            with mock.patch.object(torch._C._HostTraceVariant, method, side_effect=ValueError("rejected by the test")):
                with mock.patch.object(torch.cuda._host_trace, "raise_unexpected", strict):
                    if strict:
                        with self.assertRaisesRegex(AssertionError, f"the native variant rejects an? {what}: rejected by the test"):
                            f(x, w)
                        continue
                    self.assertEqual(f(x, w), fn(x, w), atol=0, rtol=0)
            self.assertEqual((f.traces, f.eager), (2, 0))
            self.assertIn(f"the native variant rejects a{'n' if what == 'entry' else ''} {what} (rejected by the test)", " ".join(f.declines))

    def test_a_redo_takes_a_dispatch_whose_guards_hold(self):
        # a redo takes a dispatch of the same reads whose guards hold at it: (2, 3000)'s redispatch and (1, 2016)'s
        # dispatch one softmax for the four layers, (1, 1568)'s none; (1, 1000) traces (the sum dispatches otherwise)
        def fn(q, page_table, kv, *ws):
            k = kv[page_table].flatten(1, 2)
            for w in ws:
                h = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + 1e-6)
                p = torch.softmax((h[:, None] * k).sum(-1), -1)
                q = q + (p[..., None] * k).sum(1) @ w
            return q

        kv = torch.randn(64, 64, 64, device="cuda")
        ws = [torch.randn(64, 64, device="cuda") / 8 for _ in range(4)]
        eager_call, softmaxes = _host_trace_tape._Trace.eager_call, []

        def counted(tr, func, a, k):
            softmaxes.append(func is torch.ops.aten._softmax.default)
            return eager_call(tr, func, a, k)

        f, counts = HostTraceReplay(fn), []
        with mock.patch.object(_host_trace_tape._Trace, "eager_call", counted):
            for b, n in ((2, 1000), (2, 3000), (1, 1000), (1, 2016), (1, 1568)):
                args = (torch.randn(b, 64, device="cuda"), torch.randint(0, 64, (b, -(-n // 64)), device="cuda"), kv, *ws)
                del softmaxes[:]
                self.assertEqual(f(*args), fn(*args), atol=0, rtol=0)
                counts.append(sum(softmaxes))
        self.assertEqual((counts, f.traces, f.redispatches), ([4, 1, 4, 1, 0], 2, 3))
        self.assertEqual([c[:2] for c in f.retrace_causes], [("dispatch", "aten.sum.dim_IntList")])

    def test_a_redo_agrees_with_the_variants_layout_under_its_guards(self):
        # at b == 1 the eps add returns [1, 1], the variant [b, 1]: one layout under the redo's guard b == 1
        def fn(x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)

        f = HostTraceReplay(fn)
        for b in (2, 3, 1):
            x = torch.randn(b, 64, device="cuda")
            self.assertEqual(f(x), fn(x), atol=0, rtol=0)
        self.assertEqual((f.traces, f.redispatches), (1, 1))

    @parametrize("adjacent", [False, True])
    def test_a_split_in_an_op_traces_its_class(self, adjacent):
        # the op's launches and temporary change with the row count: its class
        # traces (adjacent: two such ops)
        def fn(x):
            return add(split_op(split_op(x) if adjacent else add(x)), 1)

        f = HostTraceReplay(fn)
        for m in (8, 40, 100, 12, 50, 128, 3, 64, 65):
            x = torch.randn(m, 64, device="cuda")
            self.assertEqual(f(x), fn(x), atol=0, rtol=0)
        self.assertEqual((f.traces, len(f.variants), f.replays, f.eager), (3, 3, 6, 0))
        # each new class's trace: split_op's own guards fail and its redispatch refuses
        self.assertEqual([c[0] for c, n in f.retrace_causes.items() for _ in range(n)], ["dispatch"] * 2)
        self.assertEqual({c[1] for c in f.retrace_causes}, {"host_trace_test.split_op.default"})

    def test_a_retrace_records_a_guard_no_op_recorded(self):
        # fn's own branch on a size is a graph guard of no op's: a call where it fails traces again
        def fn(x):
            return x + 1 if x.shape[0] > 4 else x * 2

        f = HostTraceReplay(fn)
        for m in (8, 16, 2):
            x = torch.randn(m, 64, device="cuda")
            self.assertEqual(f(x), fn(x), atol=0, rtol=0)
        ((kind, op, _, where, refusal),) = f.retrace_causes
        line = fn.__code__.co_firstlineno + 1
        self.assertEqual((kind, op, where, refusal), ("graph", None, f"test/test_cuda_host_trace_replay.py:{line}", None))
        self.assertEqual((f.traces, sum(f.retrace_causes.values())), (2, 1))

    def test_a_refused_redispatch_tries_the_refusing_op_first(self):
        # the split op refuses at every new class: once it has refused for a
        # variant, its redispatch tries it before the tactic op, so a refusal
        # again dispatches nothing else
        def fn(x, y):
            return tactic_op(x), split_op(y)

        redone = []
        redo = torch.cuda._host_trace_redispatch._redo
        with mock.patch.object(torch.cuda._host_trace_redispatch, "_redo", lambda lowered, selector, **k: redone.append(selector.op) or redo(lowered, selector, **k)):
            f = HostTraceReplay(fn)
            for m, k in ((8, 8), (20, 40), (40, 100), (60, 20)):
                x, y = torch.randn(m, 64, device="cuda"), torch.randn(k, 64, device="cuda")
                self.assertEqual(f(x, y), fn(x, y), atol=0, rtol=0)
        self.assertEqual((f.traces, f.redispatches, sum(f.redispatch_refusals.values())), (3, 1, 4))
        self.assertEqual(redone, [0, 1, 1, 0, 1, 1, 0])

    def test_a_redispatch_refusal_names_its_op(self):
        # the redispatch's redos share one capture: the op that enqueues a
        # launch the trace does not record is the one its refusal names
        def fn(x, y):
            return tactic_op(x), enqueues_past_32(y)

        f = HostTraceReplay(fn)
        for m, k in ((8, 8), (20, 40)):
            x, y = torch.randn(m, 64, device="cuda"), torch.randn(k, 64, device="cuda")
            self.assertEqual(f(x, y), fn(x, y), atol=0, rtol=0)
        self.assertEqual(f.redispatches, 0)
        (refusal,) = f.redispatch_refusals
        self.assertRegex(refusal, r"^host_trace_test\.enqueues_past_32\.default declines again: .*enqueued 1 operations")

    def test_a_redispatch_refusal_leaves_no_cycle(self):
        # a refusal kept with its traceback holds the call's frames, so the replay, until a gc
        def fn(x, y):
            return tactic_op(x), enqueues_past_32(y)

        f = HostTraceReplay(fn)
        gc.collect()
        gc.disable()
        try:
            for m, k in ((8, 8), (20, 40)):
                x, y = torch.randn(m, 64, device="cuda"), torch.randn(k, 64, device="cuda")
                self.assertEqual(f(x, y), fn(x, y), atol=0, rtol=0)
            self.assertEqual(len(f.redispatch_refusals), 1)
            graphs = [weakref.ref(s.graph) for v in f.variants for s in v.captured.segments]
            self.assertTrue(graphs)
            replay = weakref.ref(f)
            del f
            self.assertIsNone(replay())
            self.assertEqual([g() for g in graphs], [None] * len(graphs))
        finally:
            gc.enable()

    def test_redos_on_another_thread_have_their_own_fake_mode(self):
        # a thread's redos share one FakeTensorMode (its init walks the stack),
        # so the background learner's thread never shares the caller's
        def fn(x):
            return tactic_op(x)

        modes = {}
        redo = torch.cuda._host_trace_redispatch._redo

        def recorded(*a, **k):
            out = redo(*a, **k)
            modes.setdefault(threading.get_ident(), set()).add(id(torch.cuda._host_trace_redispatch._redo_fakes.mode))
            return out

        def run():
            f = HostTraceReplay(fn)
            for m in (8, 20, 40):
                x = torch.randn(m, 64, device="cuda")
                self.assertEqual(f(x), fn(x), atol=0, rtol=0)
            self.assertEqual((f.traces, f.redispatches), (1, 2))

        with mock.patch.object(torch.cuda._host_trace_redispatch, "_redo", recorded):
            with ThreadPoolExecutor(1) as worker:
                worker.submit(run).result()
            run()
        self.assertEqual(sorted(map(len, modes.values())), [1, 1])
        self.assertEqual(len(set.union(*modes.values())), 2)

    @parametrize("case", ["k_shrinks", "k_one_aliased"])
    def test_a_bound_tape_at_k_one_allocates_each_buffer_once(self, case):
        # K shrinks to 1 at the second call of a tape bind_opaque bound (its GEMMs learned out of band).
        # Respec's _splice recorded the later GEMMs' outputs twice there (the bound tape's ops of no
        # allocation had spans past every allocation): a launch on a buffer its plan allocated a step
        # later, an illegal address (on GH200 a write into host memory). Respec is gone; whatever serves
        # the K == 1 call holds each allocation once and matches eager
        def k_shrinks(a0, a1, a2, a3, i):
            v0 = a2 @ a2.t()
            v1 = a2.t() @ a2
            v2 = v0 * v0
            a2 @ v1.t()
            v4 = a1 / 3
            v5 = a2 * i
            a1.copy_(v4)
            v6 = a2 @ v1.t()
            v6.t() @ v5
            torch.addmm(a2, v2, v5)
            a2 @ v1.t()
            return v6

        def k_one_aliased(a0, a1, a2, a3):
            v0 = a0 * a0
            v1 = a0.t() @ a2
            v2 = v0 @ v0.t()
            a3.sum()
            v4 = a1 @ v1.t()
            a3 * 3
            v6 = torch.softmax(v2, 0)
            v7 = v6.t() @ a0
            v9 = v6 @ v7
            v9 @ v4.t()
            return v1

        def calls():
            if case == "k_shrinks":
                for k, i in ((10, 47), (1, 46)):
                    yield tuple(torch.randn(*s, device="cuda", dtype=torch.bfloat16) for s in ((47, k), (k, 5), (47, 5), (3, 47, k))) + (i,)
            else:
                yield tuple(torch.randn(*s, device="cuda") for s in ((96, 65), (65, 16), (96, 16), (3, 96, 65)))
                shared = torch.randn(112, device="cuda")
                yield shared[64:].view(48, 1), shared[:27].view(1, 27), torch.randn(48, 27, device="cuda"), torch.randn(3, 48, 1, device="cuda")

        fn = k_shrinks if case == "k_shrinks" else k_one_aliased
        f = HostTraceReplay(fn)
        for args in calls():
            want = fn(*(a.clone() if isinstance(a, torch.Tensor) else a for a in args))
            self.assertEqual(f(*args), want, atol=0, rtol=0)
            for v in f.variants:
                self.assertEqual(len({id(a.root) for a in v.tape.allocs}), len(v.tape.allocs))
        self.assertEqual(f.eager, 0)

    @parametrize("how", ["off", "impl", "impl_shared"])
    @mock.patch("torch.cuda._host_trace.library_impls", False)
    def test_a_traced_impl_owns_its_kernel_choice(self, how):
        # an extension op runs eagerly (an eager step per layer); its traced
        # impl puts its launch in the graph with the tactic's guards the op's,
        # which redispatch under shared_op_guards
        op = torch.ops.host_trace_test.tactic_ext.default

        def fn(x):
            for _ in range(4):
                x = op(x) * 2
            return x

        shared = how == "impl_shared"
        torch.cuda._host_trace.register_traced_impl(op, tactic_add)
        self.addCleanup(torch.cuda._host_trace._TRACED_IMPLS.pop, op)
        with (
            mock.patch.object(torch.cuda._host_trace, "traced_impls", how != "off"),
            mock.patch.object(torch.cuda._host_trace, "shared_op_guards", shared),
        ):
            f = HostTraceReplay(fn)
            for m in range(2, 131):
                x = torch.randn(m, 64, device="cuda")
                self.assertEqual(f(x), fn(x), atol=0, rtol=0)
        eager = [r for v in f.variants for _, r in v.tape.launches if isinstance(r, EagerCall)]
        counts = {"off": (1, 0, 4), "impl": (5, 0, 0), "impl_shared": (1, 4, 0)}[how]
        self.assertEqual((f.traces, f.redispatches, len(eager)), counts)
        self.assertGreater(f.replays, 100)

    @parametrize("keep", [True, False])
    @mock.patch("torch.cuda._host_trace.library_impls", False)
    def test_an_all_eager_region_traces_once(self, keep):
        # every op runs eagerly: no variant, and off, each call of a new class
        # traced again to find nothing to capture
        op = torch.ops.host_trace_test.tactic_ext.default

        def fn(x):
            for _ in range(4):
                x = op(x)
            return x

        with mock.patch.object(torch.cuda._host_trace, "keep_eager_regions", keep):
            f = HostTraceReplay(fn)
            for m in range(2, 20):
                x = torch.randn(m, 64, device="cuda")
                self.assertEqual(f(x), fn(x), atol=0, rtol=0)
        self.assertEqual((f.traces, f.uncaptured, f.eager, f.eager_region_calls, f.replays), (1, 1, 18, 17, 0) if keep else (18, 18, 18, 0, 0))

    @mock.patch.object(torch.cuda._host_trace, "keep_eager_regions", True)
    @mock.patch("torch.cuda._host_trace.library_impls", False)
    def test_an_eager_regions_guards_select_it(self):
        # above 10 rows a launch: the region's guard fails, and the call traces
        op = torch.ops.host_trace_test.tactic_ext.default

        def fn(x):
            x = op(x)
            return x * 2 if x.shape[0] > 10 else x

        f = HostTraceReplay(fn)
        for m in (2, 5, 9, 11, 13, 3, 12):
            x = torch.randn(m, 64, device="cuda")
            self.assertEqual(f(x), fn(x), atol=0, rtol=0)
        self.assertEqual((f.traces, f.uncaptured, f.eager_region_calls, f.replays, len(f.variants)), (2, 1, 3, 2, 1))

    @parametrize("library_impls", [False, True])
    @mock.patch.object(torch.cuda._host_trace, "traced_impls", True)
    def test_a_declining_traced_impl_leaves_an_eager_call(self, library_impls):
        op = torch.ops.host_trace_test.tactic_ext.default

        def declines(x):
            raise current_trace().decline("the port has no launch")

        def fn(x):
            return op(x) * 2

        torch.cuda._host_trace.register_traced_impl(op, declines)
        self.addCleanup(torch.cuda._host_trace._TRACED_IMPLS.pop, op)
        with mock.patch.object(torch.cuda._host_trace, "library_impls", library_impls):
            ((_, call), _) = trace(fn, (torch.randn(48, 64, device="cuda"),)).launches
            f = HostTraceReplay(fn)
            for m in (48, 20, 48):
                x = torch.randn(m, 64, device="cuda")
                self.assertEqual(f(x), fn(x), atol=0, rtol=0)
        if library_impls:
            self.assertIsInstance(call, KernelLaunch)
        else:
            self.assertIsInstance(call, EagerCall)
            self.assertIn("the port has no launch", call.reason)
        self.assertEqual((f.traces, f.replays, f.eager), (1, 2, 0))

    def test_fast_twins_trace_and_replay(self):
        # split_with_sizes's twin (view) and the extension op's (its fake kernel)
        op = torch.ops.host_trace_test.tactic_ext.default

        def fn(x):
            a, b = x[1:].split([16, 48], dim=1)
            return op(b) * 2, a * 3

        made = []

        def wrapped(*args):
            made.append(args[1])
            return FakeTensor(*args)

        with mock.patch.object(_host_trace_tape, "FakeTensor", wrapped):
            trace(fn, (torch.randn(40, 64, device="cuda"),))
            self.assertGreater(len(made), 0)
            f = HostTraceReplay(fn)
            for m in (40, 3, 17, 40, 130):
                x = torch.randn(m, 64, device="cuda")
                self.assertEqual(f(x), fn(x), atol=0, rtol=0)
        self.assertEqual((f.traces, f.replays, f.eager), (1, 4, 0))

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
        self.assertEqual((f.traces, f.replays, f.eager), (1, 12, 0))

    def test_a_gather_over_an_allocation_redispatches(self):
        # index_select's gather (more than 16 indices) restrides the allocation with as_strided
        def fn(x, i):
            return (x * 2).index_select(0, i)

        f = HostTraceReplay(fn)
        x = torch.randn(300, 64, device="cuda")
        for n in (8, 8, 32, 32, 8):
            i = torch.randint(0, 300, (n,), device="cuda")
            self.assertEqual(f(x, i), fn(x, i))
        self.assertEqual((f.traces, f.replays, f.redispatches, f.eager, f.redispatch_refusals), (1, 4, 1, 0, {}))

    def test_an_eager_steps_guards_are_its_own(self):
        # index is an eager step; its fake kernel's broadcast of i against j picks the output's size
        def fn(x, i, j):
            return x[i, j] * 2

        x = torch.randn(64, 64, device="cuda")
        # traced at len(j) == 1: the output is len(i) long; at len(i) == 1 it is len(j)
        # long, so the redispatch returns other metadata and refuses (a full trace)
        cases = {((8, 1), (8, 1), (1, 8), (1, 8)): (2, 2, 0, {"aten.index.Tensor returned other metadata": 1})}
        # traced at len(i) == len(j) (one symbol): at (1, 8) the output is that long too, so it redispatches
        cases[((8, 8), (1, 8), (8, 8), (4, 4))] = (1, 3, 1, {})
        for sizes, want in cases.items():
            f = HostTraceReplay(fn)
            for a, b in sizes:
                i, j = torch.randint(0, 64, (a,), device="cuda"), torch.randint(0, 64, (b,), device="cuda")
                self.assertEqual(f(x, i, j), fn(x, i, j), atol=0, rtol=0)
            self.assertEqual((f.traces, f.replays, f.redispatches, f.redispatch_refusals), want)

    def test_a_size_one_view_of_a_contiguous_slice_takes_no_trace(self):
        # flatten's contiguity check guarded len(i) != 1; the slice's stride is the contiguous
        # one by construction, so the check needs no guard and len(i) == 1 takes no trace
        def fn(cache, i):
            return cache[i].flatten(0, 1) * 2

        cache = torch.randn(64, 4, 8, device="cuda")
        f = HostTraceReplay(fn)
        for n in (8, 1, 3, 1):
            i = torch.randint(0, 64, (n,), device="cuda")
            self.assertEqual(f(cache, i), fn(cache, i), atol=0, rtol=0)
        self.assertEqual((f.traces, f.replays), (1, 3))

    def test_a_python_error_of_a_host_route_raises_in_the_suites(self):
        # not a TORCH_CHECK, which eager would raise too: the route's bug
        def route(*args):
            raise TypeError("incompatible function arguments")

        def fn(x, i):
            return (x * 2).index_select(0, i)

        x, i = torch.randn(300, 64, device="cuda"), torch.randint(0, 300, (8,), device="cuda")
        with mock.patch.dict("torch.cuda._host_trace_tape._TRACED_ATEN", {torch.ops.aten.index_select.default: route}):
            with self.assertRaisesRegex(TypeError, "incompatible function arguments"):
                trace(fn, (x, i))
            # outside the suites it declines by name
            with mock.patch("torch.cuda._host_trace.raise_unexpected", False):
                f = HostTraceReplay(fn)
                for _ in range(3):
                    self.assertEqual(f(x, i), fn(x, i))
        self.assertEqual((f.traces, f.replays, f.eager), (1, 0, 3))
        self.assertEqual(f.declines, ["host_trace: aten.index_select.default raised TypeError: incompatible function arguments (declined)"])  # noqa: B950

    @parametrize("keep", [True, False])
    def test_a_decline_falls_back_once(self, keep):
        calls = []

        def fn(x):
            calls.append(1)
            return x.cumsum(0)

        self.enterContext(mock.patch.object(torch.cuda._host_trace, "keep_eager_regions", keep))
        f = HostTraceReplay(fn)
        x = torch.randn(100, device="cuda")
        # the warm-up is the call: fn runs once for it and once for the trace
        self.assertEqual(f(x), x.cumsum(0))
        self.assertEqual((len(calls), f.traces, f.eager), (2, 1, 1))
        # no traced launches: uncaptured, not a structural decline
        why = "host_trace: no traced launches: every operation runs eagerly (aten.cumsum.default) (declined)"
        self.assertEqual((f.uncaptured, f.structural, f.declines), (1, 0, [why]))
        # the declined class runs eagerly without a trace
        self.assertEqual(f(x), x.cumsum(0))
        self.assertEqual((len(calls), f.traces, f.eager), (3, 1, 2))
        # another class the trace's guards hold for runs eagerly too; off, it
        # traces again, with its warm-up
        y = torch.randn(7, device="cuda")
        self.assertEqual(f(y), y.cumsum(0))
        self.assertEqual((len(calls), f.traces, f.eager), (4, 1, 3) if keep else (5, 2, 3))
        self.assertEqual((f.variants, f.uncaptured, f.declines), ([], 1 if keep else 2, [why]))

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
        self.assertEqual((f.traces, f.replays, f.eager), (1, 1, 0))

    def test_a_side_stream_declines_once_and_warns(self):
        side = torch.cuda.Stream()

        def fork_join(x):
            y = add(x)
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                y = add(y)
            torch.cuda.current_stream().wait_stream(side)
            return y

        f = HostTraceReplay(fork_join)
        with self.assertLogs("torch.cuda._host_trace_replay", "WARNING") as logs:
            for n in (64, 64, 128, 256, 64):
                x = torch.randn(n, device="cuda")
                self.assertEqual(f(x), fork_join(x), atol=0, rtol=0)
        self.assertEqual(len(logs.output), 1)
        self.assertIn("_add: launched on a stream other than the trace's", logs.output[0])
        self.assertEqual((f.traces, f.replays, f.eager, f.structural), (1, 0, 5, 1))

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

    def test_output_views_match_eager_autograd_metadata(self):
        def views(x, x2):
            y = add(x)
            return y, y[1:], y.view(-1, 2), x[2:], x2[2:]

        def check(outs, ref, args):
            known = {id(t): i for i, t in enumerate((*outs, *args))}
            known_ref = {id(t): i for i, t in enumerate((*ref, *args))}
            for o, r in zip(outs, ref):
                self.assertTrue(torch.equal(o, r))
                self.assertEqual((o._is_view(), o.requires_grad, o.is_inference()), (r._is_view(), r.requires_grad, r.is_inference()))
                if r._is_view():
                    self.assertEqual(torch._C._autograd._get_creation_meta(o), torch._C._autograd._get_creation_meta(r))
                self.assertEqual(known.get(id(o._base), o._base), known_ref.get(id(r._base), r._base))
            for o in outs[1:]:
                if not o.is_inference():
                    v = o._base._version
                    o.add_(0)
                    self.assertEqual((o._version, o._base._version), (v + 1, v + 1))

        for mode in (torch.enable_grad, torch.no_grad, torch.inference_mode):
            f = HostTraceReplay(views)
            for n in (1000, 2000, 1000):
                big = torch.randn(n + 1, device="cuda", requires_grad=mode is torch.no_grad)
                x = torch.randn(n, device="cuda")
                with mode():
                    x2 = big[1:]
                    outs, ref = f(x, x2), views(x, x2)
                    check(outs, ref, (x, big))
            self.assertEqual(f.eager, 0)
            self.assertGreater(f.replays, 1)

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

    @parametrize("keep", [False, True])
    def test_an_eager_steps_output_is_kept_only_if_fn_keeps_it(self, keep):
        # index is an eager step: the tape's own records of its output are no tensor fn kept
        # a first trace that imports torch._dynamo is kept by torch.fx.wrap's frame, until a gc
        import torch._dynamo  # noqa: F401

        kept = []

        def fn(x, i):
            k = x[i]
            if keep:
                kept.append(k)
            return k.sum(-1)

        f = HostTraceReplay(fn, check_escapes=True)
        x, i = torch.randn(64, 64, device="cuda"), torch.randint(0, 64, (8,), device="cuda")
        for _ in range(3):
            self.assertEqual(f(x, i), x[i].sum(-1))
        if keep:
            self.assertEqual((f.traces, f.replays, f.eager), (1, 0, 3))
            self.assertIn("fn kept a traced tensor past the trace, in a list", f.declines[-1])
        else:
            note = "host_trace: an eager step in a variant: aten.index.Tensor (no traced implementation)"
            self.assertEqual((f.traces, f.declines), (1, [note]))
            self.assertGreater(f.replays, 0)

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

    def test_a_cycle_fn_leaves_is_kept(self):
        # no collection inside the capture: one could finalize a CUDAGraph there
        def fn(x):
            y = x * 2
            box = {"y": y}
            box["self"] = box
            return y + 1

        f = HostTraceReplay(fn, check_escapes=True)
        x = torch.randn(1000, device="cuda")
        for _ in range(3):
            self.assertEqual(f(x), x * 2 + 1)
        self.assertEqual((f.traces, f.replays, f.eager), (1, 0, 3))
        self.assertIn("fn kept a traced tensor past the trace, in a dict", f.declines[0])

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

    @parametrize("op", ["new_full", "new_zeros", "new_ones"])
    def test_a_new_factory_of_symbolic_sizes_is_traced(self, op):
        # its C++ body takes int sizes: the factory it calls at self's options
        fn = {
            "new_full": lambda x: x.new_full((x.shape[0], 2 * x.shape[1]), 2.5),
            "new_zeros": lambda x: x.new_zeros((x.shape[1] + 1,), dtype=torch.int64),
            "new_ones": lambda x: x.new_ones((2 * x.shape[1], 3)),
        }[op]
        f = HostTraceReplay(fn)
        for n in (6, 6, 6, 9, 9):
            x = torch.randn(4, n, device="cuda")
            self.assertEqual(f(x), fn(x), atol=0, rtol=0)
        self.assertEqual((f.traces, f.replays, f.eager, f.declines), (1, 4, 0, []))
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
        # the (4, 2) call's prod, var_mean and std_mean have no traced launch
        uncaptured = {"prod": "aten.prod.dim_int", "var_mean": "aten.var_mean.correction", "std_mean": "aten.std_mean.correction"}.get(op)
        self.assertEqual(f.declines, [f"host_trace: no traced launches: every operation runs eagerly ({uncaptured}) (declined)"] if uncaptured else [])
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

        bench = _add_tuned._bench

        def ranked(*args, config, **meta):
            # 2000 at the other config than 1000's: at 1000's, its call redispatches 1000's trace
            bench(*args, config=config, **meta)
            return [float((config.kwargs["B"] == 128) != (args[2] == 1000))]

        f = _host_trace_replay.HostTraceReplay(tuned)
        # the first call, eager, tunes its size; a later size's warm-up tunes
        # it: the benchmark's calls are not the trace's, so that call runs
        # eagerly and the next traces (a witness decline is retried once per
        # class, and noted)
        with mock.patch.object(_add_tuned, "_bench", ranked):
            for n in (1000, 1000, 2000, 2000, 2000):
                x = torch.randn(n, device="cuda")
                self.assertEqual(f(x), x + 3)
        self.assertEqual((f.traces, f.replays, f.eager, [d[-9:] for d in f.declines]), (3, 1, 2, ["(retried)"]))

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
        f = HostTraceReplay(matmul_chain, opaque=())
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
        # call traces again with the mul an eager call, and the rest replays.
        # The first trace's run is split before the sin, a later peak at
        # varying sizes, so the failing launch holds two
        def fn(x, w):
            return (matmul_chain(x, w) * 2).sin()

        launch, calls = _host_trace_capture._launch, []

        def fail_second(launches, *args):
            calls.append(len(launches))
            if len(calls) == 2:
                launch(launches[:1], *args)
                raise RuntimeError("injected")
            return launch(launches, *args)

        f = HostTraceReplay(fn, splits="walk", opaque=())
        w = torch.randn(16, 8, device="cuda")
        with mock.patch.object(_host_trace_capture, "_launch", fail_second):
            for _ in range(3):
                x = torch.randn(1024, device="cuda")
                self.assertEqual(f(x, w), fn(x, w), atol=0, rtol=0)
        self.assertEqual((f.traces, f.replays, f.eager), (2, 2, 0))
        self.assertEqual(calls, [1, 2, 1, 1, 1])
        (v,) = f.variants
        self.assertEqual(len(v.captured.segments), 3)
        self.assertEqual(len(f.declines), 2)
        self.assertIn("RuntimeError: injected", f.declines[0])
        self.assertEqual(f.declines[1], "host_trace: an eager step in a variant: aten.mm.default (no traced implementation)")
        # the suites' strict mode raises it
        calls.clear()
        with mock.patch("torch.cuda._host_trace.raise_unexpected", True):
            with mock.patch.object(_host_trace_capture, "_launch", fail_second):
                with self.assertRaisesRegex(RuntimeError, "injected"):
                    HostTraceReplay(fn, opaque=())(x, w)

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

        f = HostTraceReplay(matmul_chain, opaque=())
        w = torch.randn(16, 8, device="cuda")
        with mock.patch.object(_host_trace_capture, "_verify", refuse_second):
            for _ in range(3):
                x = torch.randn(1024, device="cuda")
                self.assertEqual(f(x, w), matmul_chain(x, w), atol=0, rtol=0)
        self.assertEqual((f.traces, f.replays, f.eager), (2, 2, 0))
        (v,) = f.variants
        self.assertEqual(len(v.captured.segments), 1)
        self.assertEqual(len(f.declines), 2)
        self.assertIn("the capture of _add failed: refused", f.declines[0])
        self.assertEqual(f.declines[1], "host_trace: an eager step in a variant: aten.mm.default (no traced implementation)")

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
        self.assertEqual((f.replays, f.eager), (2, 0))
        self.assertEqual(f.declines, ["host_trace: an eager step in a variant: aten.native_layer_norm_backward.default (no traced implementation)"])

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
        self.assertEqual((f.replays, f.eager), (2, 0))
        self.assertEqual(f.declines, ["host_trace: an eager step in a variant: aten._scaled_dot_product_cudnn_attention.default (no traced implementation)"])

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

    @parametrize("shared", [False, True])
    def test_a_triton_fallback_in_a_chain(self, shared):
        def fn(x):
            y = add(x)
            z = torch.empty_like(y)
            n = y.numel()
            _add[(triton.cdiv(n, 128),)](y, z, n, 2, B=128, launch_cooperative_grid=True)
            return add(z, 1)

        f = HostTraceReplay(fn)
        with (
            mock.patch("torch.cuda._host_trace_replay.trace_structured") as log,
            mock.patch.object(torch.cuda._host_trace, "shared_op_guards", shared),
        ):
            for n in (1000, 1000, 3000, 1, 1):
                x = torch.randn(n, device="cuda")
                self.assertEqual(f(x), fn(x), atol=0, rtol=0)
        self.assertEqual(f.eager, 0)
        self.assertGreater(f.replays, 0)
        if shared:
            self.assertEqual(len(f.variants), 1)
        else:
            self.assertGreater(len(f.variants), 1)
        (why,) = f.triton_fallbacks
        self.assertIn("cooperative launches", why)
        (fallback,) = [c for c in log.call_args_list if c.kwargs["metadata_fn"]()["name"] == "host_trace_triton_fallback"]
        self.assertEqual(fallback.kwargs["payload_fn"](), why)

    @mock.patch("torch.cuda._host_trace.library_impls", False)
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

    @mock.patch("torch.cuda._host_trace.library_impls", False)
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
            f.declines[-1],
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

    def test_a_size_1_dims_eager_stride_holds_only_at_size_1(self):
        # the diagonal of a (128, 1) has length 1 and stride s0 + s1 (2): sort's fake kernel predicts its
        # output contiguous (1), eager's keeps the input's 2, which the warm-up check wrote into the tape as
        # the constant 2. The (31, 12) call, served by a redispatch, then predicted stride 2 at length 12
        # (_Disagreement: eager's is 1): eager's size-1 stride holds where the length is 1, a select
        def fn(a0, a1):
            v0 = a1.argmax(0)
            v1 = a0.diagonal()
            return v0 + v0, v1.scatter(0, v1.argsort(0), v1)

        f = HostTraceReplay(fn)
        for n, m in ((128, 1), (128, 1), (31, 12), (31, 12)):
            a0 = torch.randint(-50, 50, (n, m), device="cuda", dtype=torch.int32)
            a1 = torch.randint(-50, 50, (3 if m == 1 else 2, n, m), device="cuda", dtype=torch.int32)
            self.assertEqual(f(a0, a1), fn(a0, a1), atol=0, rtol=0)
        self.assertEqual((f.traces, f.redispatches, f.replays, f.eager), (1, 1, 3, 0))

    def test_a_full_width_slice_is_one_view_at_every_size(self):
        # buf[:, :n] at n == width: indexing's return-self shortcut changes only identity, so the
        # trace at n = 8 serves n = 64 (vLLM block_table's slot_mappings[:, :num_tokens_padded]); the
        # mul alone dispatches again there (its operand is contiguous at the full width, which its
        # TensorIterator collapses: an op guard), no graph guard
        def fn(buf, x):
            return buf[:, : x.shape[0]] * 2 + x.sum()

        f = HostTraceReplay(fn)
        buf = torch.randn(16, 64, device="cuda")
        for n in (8, 8, 64, 64, 32, 64):
            x = torch.randn(n, device="cuda")
            self.assertEqual(f(buf, x), fn(buf, x), atol=0, rtol=0)
        self.assertEqual((f.traces, f.redispatches, f.replays, f.retrace_causes), (1, 1, 5, {}))

    @parametrize("checks", [True, False])
    def test_an_as_strided_bound_is_a_check_of_the_calls(self, checks):
        # eager's checkInBoundsForStorage: with validity_checks a condition the variant's calls meet
        # (Tape.checks), no guard; a call past the bound misses and its warm-up raises eager's error
        def fn(x, n):
            return torch.zeros(n, device="cuda").as_strided((x.shape[0],), (1,)) + x

        f = HostTraceReplay(fn)
        with mock.patch("torch.cuda._host_trace.validity_checks", checks):
            for m, n in ((4, 8), (6, 8), (5, 9)):
                x = torch.randn(m, device="cuda")
                self.assertEqual(f(x, n), fn(x, n), atol=0, rtol=0)
            self.assertEqual((f.traces, f.replays), (1, 2))
            with self.assertRaisesRegex(RuntimeError, "out of bounds"):
                f(torch.randn(9, device="cuda"), 8)
        self.assertEqual(f.replays, 2)
        (tape,) = [v.captured.lowered.tape for v in f.variants]
        self.assertEqual(bool(tape.checks), checks)

    def test_a_zero_element_eager_output_keeps_eagers_strides(self):
        # sort's zero-element outputs take the input's strides where its fake kernel's are
        # contiguous: they address nothing, so the replay keeps eager's output, no disagreement
        def fn(x, y):
            return x.argsort(2), y + 1

        f = HostTraceReplay(fn)
        y = torch.randn(8, device="cuda")
        for x in (torch.randn(1, 12, 0, device="cuda"), torch.empty_strided((1, 12, 0), (12, 1, 12), device="cuda")):
            got, want = f(x, y), fn(x, y)
            self.assertEqual(got, want, atol=0, rtol=0)
            self.assertEqual([t.stride() for t in got], [t.stride() for t in want])
        self.assertEqual((f.traces, f.replays, f.eager), (1, 1, 0))

    @parametrize("free", [True, False])
    def test_a_size_1_views_stride_is_free(self, free):
        # qkv[:, :256] viewed (T, 256): eager's computeStride strides dim 0 by
        # 256 at T=1 and by qkv's 384 above; scale2d never steps it at T=1
        def fn(qkv):
            return scale2d(qkv[:, :256].view(qkv.shape[0], 256))

        f = HostTraceReplay(fn)
        base = torch.randn(9, 384, device="cuda")
        with mock.patch("torch.cuda._host_trace.free_size_one_strides", free):
            for t in (2, 1, 3, 1, 8):
                self.assertEqual(f(base[:t]), fn(base[:t]), atol=0, rtol=0)
        self.assertEqual((f.traces, f.eager), (1 if free else 2, 0))

    @parametrize("inferred", [True, False])
    @mock.patch("torch.cuda._host_trace.free_size_one_strides", True)
    def test_a_view_inferring_a_size_is_free_too(self, inferred):
        # view(-1, 256) went to the fake kernel, which guards T > 1 as eager's
        # computeStride would for view(T, 256)
        def fn(qkv):
            return scale2d(qkv[:, :256].view(-1, 256))

        f = HostTraceReplay(fn)
        base = torch.randn(9, 384, device="cuda")
        with mock.patch("torch.cuda._host_trace.free_inferred_views", inferred):
            for t in (2, 1, 3, 1, 8):
                self.assertEqual(f(base[:t]), fn(base[:t]), atol=0, rtol=0)
        self.assertEqual((f.traces, f.replays, f.eager), (1, 4, 0) if inferred else (2, 3, 0))
        # 6T // 4 times 4 is not 6T at every T: the fake kernel's view
        g = HostTraceReplay(lambda x: scale2d(x.view(-1, 4)))
        with mock.patch("torch.cuda._host_trace.free_inferred_views", inferred):
            for t in (2, 4, 6, 2, 4):
                x = torch.randn(t, 6, device="cuda")
                self.assertEqual(g(x), scale2d(x.view(-1, 4)), atol=0, rtol=0)
        self.assertEqual((g.traces + g.replays, g.eager), (5, 0))
        self.assertGreater(g.replays, 0)

    @parametrize("free", [True, False])
    def test_a_kernel_stepping_a_size_1_dim_sees_eagers_stride(self, free):
        # a kernel that indexes a size-1 dim past its size reads where the
        # dim's stride points: the free view's selects eager's (256) at T == 1,
        # not the T > 1 trace's (384)
        def fn(qkv):
            x = qkv[:, :256].view(qkv.shape[0], 256)
            y = torch.empty(256, device=x.device)
            _second_row[(2,)](x, y, 256, x.stride(0), B=128)
            return y

        f = HostTraceReplay(fn)
        base = torch.randn(3, 384, device="cuda")
        with mock.patch("torch.cuda._host_trace.free_size_one_strides", free):
            self.assertEqual(f(base[:2]), base[1, :256], atol=0, rtol=0)
            got = f(base[:1])
            if not free:
                # the T == 1 call traced (its result is the witness's): compare a replay
                got = f(base[:1])
        self.assertEqual((f.traces, f.replays, f.eager), (1, 1, 0) if free else (2, 1, 0))
        self.assertEqual(got, fn(base[:1]), atol=0, rtol=0)

    def test_a_gemm_on_a_size_1_view_reads_eagers_strides(self):
        # matmul reshapes a (4, 1, 46) of strides (46, 1, 1) for its bmm: eager's computeStride strides
        # the size-1 dim 46, the free view kept the input's 1, and the bmm's binding was harvested at a
        # layout cuBLAS runs with another kernel than eager's (gemvNSP, not gemvx): 1 ulp apart
        def fn(a):
            return a @ a.transpose(-1, -2)

        f = HostTraceReplay(fn)
        for _ in range(4):
            a = torch.randn(4 * 46 + 64, device="cuda")[:184].as_strided((4, 1, 46), (46, 1, 1))
            self.assertEqual(f(a), fn(a), atol=0, rtol=0)
        self.assertEqual((f.traces, f.replays, f.eager), (1, 3, 0))
        ((site,),) = [v.captured.lowered.sites for v in f.variants]
        self.assertEqual(site.site.key.strides[:2], ((46, 46, 1), (46, 1, 1)))

    @mock.patch("torch.cuda._host_trace.library_impls", True)
    def test_a_library_impl_kernel_is_traced(self):
        def fn(x):
            y = torch.empty_like(x)
            torch.ops.host_trace_direct.scale_into(y, torch.ops.host_trace_direct.scale(x, 3.0))
            return y

        f = HostTraceReplay(fn)
        for n in (8, 8, 16, 16, 8):
            x = torch.randn(n, 32, device="cuda")
            self.assertEqual(f(x), (x * 3 + 1) * 2)
        self.assertEqual((f.traces, f.replays, f.eager, f.declines), (1, 4, 0, []))
        (v,) = f.variants
        self.assertFalse([r for _, r in v.tape.launches if isinstance(r, EagerCall)])

    @parametrize("name", ["tactic_ext", "transposed_fake", "transposed_fake_at_4", "lies_when_small"])
    @mock.patch("torch.cuda._host_trace.library_impls", True)
    def test_a_fixture_kernel_traces_with_library_impls(self, name):
        op = getattr(torch.ops.host_trace_test, name).default

        def fn(x):
            return add(op(add(x)))

        f = HostTraceReplay(fn)
        for m in (8, 8, 4, 12, 4, 8):
            x = torch.randn(m, 16, device="cuda")
            self.assertEqual(f(x), fn(x), atol=0, rtol=0)
        self.assertEqual((f.eager, f.declines), (0, []))
        self.assertGreater(f.replays, 0)
        self.assertFalse([r for v in f.variants for _, r in v.tape.launches if isinstance(r, EagerCall)])

    @mock.patch("torch.cuda._host_trace.library_impls", False)
    def test_a_library_impl_kernel_is_a_named_eager_step(self):
        tape = trace(lambda x: torch.ops.host_trace_direct.scale(x, 3.0), (torch.randn(8, 32, device="cuda"),))
        (call,) = [r for _, r in tape.launches]
        self.assertIsInstance(call, EagerCall)
        self.assertEqual(call.reason, "host_trace_direct.scale.default's CUDA kernel is a Library.impl Python function (library_impls is off)")  # noqa: B950

    @parametrize("kind", list(REENTRY_OPS))
    @parametrize("checked", [True, False])
    @mock.patch("torch.cuda._host_trace.library_impls", True)
    def test_a_library_impl_calling_an_entry_is_an_eager_step(self, checked, kind):
        # the trace would inline the call at the trace's Python state
        g = HostTraceReplay(lambda x: add(x, 2))
        op, switch = REENTRY_OPS[kind]

        def fn(x):
            return add(op(add(x)), 1)

        REENTERED[:] = [g]
        self.addCleanup(REENTERED.clear)
        f = HostTraceReplay(fn)
        with mock.patch.object(torch.cuda._host_trace, switch, checked):
            calls = [r for _, r in trace(fn, (torch.randn(4096, device="cuda"),)).launches]
            for n in (4096, 4096, 2048, 4096):
                x = torch.randn(n, device="cuda")
                self.assertEqual(f(x), add(add(add(x), 2), 1), atol=0, rtol=0)
        self.assertEqual((f.traces, f.replays, f.eager), (1, 3, 0))
        # checked, g replays in each of f's replays, as its eager step
        self.assertEqual(g.replays, 4 if checked else 1)
        if checked:
            (call,) = [r for r in calls if isinstance(r, EagerCall)]
            name = op.default
            self.assertEqual(call.reason, f"{name}'s kernel declines: host_trace: {name}'s {kind} body calls a host-trace entry (declined)")
        else:
            self.assertFalse([r for r in calls if isinstance(r, EagerCall)])
            self.assertEqual(len(calls), 3)

    @mock.patch("torch.cuda._host_trace.library_impls", True)
    def test_a_library_impl_bodys_calls_see_the_warm_up(self):
        # the warm-up runs a body the trace descends under its witness: the
        # body's calls see a dispatch mode there, as the trace's do
        FIRST_USE.clear()
        self.addCleanup(FIRST_USE.clear)

        def fn(x):
            return add(torch.ops.host_trace_direct.sets_up_inside(add(x)), 1)

        f = HostTraceReplay(fn)
        for n in (4096, 4096, 2048, 4096):
            x = torch.randn(n, device="cuda")
            self.assertEqual(f(x), add(add(x) * 2, 1), atol=0, rtol=0)
        self.assertEqual(FIRST_USE, [4096])
        self.assertEqual((f.traces, f.eager, f.declines), (1, 0, []))
        self.assertFalse([r for v in f.variants for _, r in v.tape.launches if isinstance(r, EagerCall)])

    @mock.patch("torch.cuda._host_trace.library_impls", True)
    def test_a_library_impl_registered_after_a_trace_is_on_the_stack(self):
        x = torch.randn(4096, device="cuda")
        trace(add, (x,))
        tapes = []
        lib = torch.library.Library("host_trace_late", "FRAGMENT")
        self.addCleanup(lib._destroy)
        lib.define("body(Tensor x) -> Tensor")
        lib.impl("body", lambda y: tapes.append(trace(lambda z: add(lies_when_small(z)), (y,))) or y.clone(), "CUDA")
        torch.ops.host_trace_late.body(x)
        (call,) = [r for _, r in tapes[0].launches if isinstance(r, EagerCall)]
        self.assertIn("the trace runs inside a Library.impl body", call.reason)

    @parametrize("kind", list(REENTRY_OPS))
    @parametrize("checked", [True, False])
    @mock.patch("torch.cuda._host_trace.library_impls", True)
    def test_a_trace_inside_a_library_impl_body_runs_library_impls_eagerly(self, checked, kind):
        # the body may read state its caller set: a nested trace leaves it eager
        tapes = []

        def body(x):
            tapes.append(trace(lambda y: add(lies_when_small(y)), (x,)))
            return x.clone()

        op, switch = REENTRY_OPS[kind]
        REENTERED[:] = [body]
        self.addCleanup(REENTERED.clear)
        with mock.patch.object(torch.cuda._host_trace, switch, checked):
            op(torch.randn(4096, device="cuda"))
        (tape,) = tapes
        eager = [r for _, r in tape.launches if isinstance(r, EagerCall)]
        if checked:
            (call,) = eager
            self.assertEqual(call.reason, f"host_trace_test.lies_when_small.default's kernel declines: the trace runs inside a {kind} body, whose state its body may read")  # noqa: B950
        else:
            self.assertFalse(eager)

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
        self.assertEqual((f.traces, f.replays, f.eager, [d[-9:] for d in f.declines]), (2, 1, 1, ["(retried)"]))


@unittest.skipIf(not TEST_CUDA, "requires CUDA")
@requires_cuda_python_bindings
@unittest.skipIf(not has_triton(), "requires triton")
class TestReplayMemory(TestCase):
    def test_an_entry_reads_a_root_its_launch_did_not(self):
        # traced at n = 1, argmax reads nothing of v (one element: its launch only writes the index),
        # but its op holds v: its redispatch entry at n = 13 reads v, which the plan must keep live
        # through argmax, not free after the Triton add for the GEMM's output to take
        def fn(a1, a2):
            v = add(a2, 2)
            a1 @ a1.t()
            return v.argmax(0)

        def randn(*size):
            return torch.randn(size, device="cuda", dtype=torch.float16)

        f = HostTraceReplay(fn)
        for a1, a2 in ((randn(17, 1), randn(1)), (randn(2048, 13), randn(13)), (randn(2048, 13), randn(13))):
            self.assertEqual(f(a1, a2), fn(a1, a2), atol=0, rtol=0)
        self.assertEqual((f.traces, f.redispatches, f.eager, f.retrace_causes), (1, 1, 0, {}))
        tape, memory = f.variants[0].tape, f.variants[0].memory
        lives = {k: last for step in memory.steps for k, _, last in step.temporaries}
        # v is the tape's first allocation (the host's empty_like), argmax its last op
        self.assertGreaterEqual(lives[0], tape.launches[tape.ops[-1].launches.start][0])

    def test_an_entry_reads_a_root_its_traced_launch_did_not(self):
        # argmax over a size-1 dim at the trace writes zeros and reads nothing of v: its redispatch
        # entry at size 3 reads v, which the stack's and the add's temporaries must not take
        def fn(a, b):
            v = b.sum(2, keepdim=True)
            return v.argmax(0), torch.stack([a, a]) + 1.5

        f = HostTraceReplay(fn)
        for n, m, k in ((1, 200, 25), (1, 200, 25), (3, 96, 13)):
            a, b = (torch.randn(s, device="cuda", dtype=torch.bfloat16) for s in ((m, k), (n, m, k)))
            self.assertEqual(f(a, b), fn(a, b), atol=0, rtol=0)
        self.assertEqual((f.traces, f.redispatches, f.eager), (1, 1, 0))

    def test_the_plan_follows_the_tape(self):
        def fn(x):
            torch.empty_like(x)
            return two_step(x)

        f = HostTraceReplay(fn, memory="eager")
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
        # the first call traced (its result is the witness's): replay the plan
        self.assertEqual(f(x), x + 4)
        self.assertEqual((f.traces, f.replays, f.eager), (1, 1, 0))
        g = HostTraceReplay(fn, memory="run_buffer")
        self.assertEqual(g(x), x + 4)
        run = StepMemory((2,), temporaries, (), (), (0,))
        self.assertEqual(g.variants[0].memory, MemoryPlan((2,), (run, after)))
        self.assertEqual(g(x), x + 4)
        self.assertEqual((g.traces, g.replays, g.eager), (1, 1, 0))
        # the default, "auto", takes the run buffer here; the output is its own allocation
        h = HostTraceReplay(fn)
        self.assertEqual(h(x), x + 4)
        self.assertEqual(h(x), x + 4)
        self.assertEqual((h.traces, h.replays, h.eager), (1, 1, 0))
        self.assertEqual(h.variants[0].memory, MemoryPlan((2,), (run, after)))
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

    @parametrize("shared", [False, True])
    def test_retained_and_peak_memory_follow_eager(self, shared):
        f = HostTraceReplay(two_step)
        # the trace, two replays of the same shape, a smaller size, a new
        # variant (1 specializes; with shared guards the variant redispatches),
        # the first again
        for n in (1 << 20, 1 << 20, 1 << 20, 4096, 1, 1 << 18, 1 << 20):
            x = torch.randn(n, device="cuda")
            want, eager_retained, eager_peak = self.measure(two_step, x)
            with mock.patch.object(torch.cuda._host_trace, "shared_op_guards", shared):
                y, retained, peak = self.measure(f, x)
            self.assertEqual(y, want, atol=0, rtol=0)
            # only the output stays allocated, and only while it is referenced
            self.assertEqual(retained, eager_retained)
            if f.traces + f.replays > 1:  # the warm-up call is eager's
                self.assertLessEqual(peak, eager_peak)
            before = torch.cuda.memory_allocated()
            del y
            self.assertEqual(torch.cuda.memory_allocated(), before - eager_retained)
        self.assertEqual((f.traces, f.eager), (1 if shared else 2, 0))
        self.assertGreater(f.replays, 0)

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

    @mock.patch("torch.cuda._host_trace.library_impls", False)
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
            self.assertEqual((f.traces, f.replays, f.eager), (1, 2, 0))
            (v,) = f.variants
            self.assertEqual(any(s.order for s in v.memory.steps), eager_order)

    @mock.patch("torch.cuda._host_trace.library_impls", False)
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

    def test_a_run_splits_past_a_peak_it_cannot_lower(self):
        # traced with a large x, the peak is add(x)'s temporary beside its
        # output, which no split lowers; the later one, where y is dead but
        # held to its run's end, is the peak with a small x, so the run
        # splits there too
        def fn(x, y, c):
            return add(add(x)), add(y), add(c)

        f = HostTraceReplay(fn, memory="eager", freed_arguments=(1,), splits="walk")
        m = 1 << 18
        for n in (1 << 22, 1 << 22, 4096, 4096):
            args = [torch.randn(k, device="cuda") for k in (n, m, m)]
            want = (args[0] + 6, args[1] + 3, args[2] + 3)
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            y = f.call_boxed(args)
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() - before
            self.assertEqual(y, want)
        self.assertEqual((f.replays, f.eager), (3, 0))
        self.assertEqual(peak, 4 * (4096 + m))
        (v,) = f.variants
        runs = [s for s in v.captured.lowered.steps if isinstance(s, range)]
        self.assertEqual([len(r) for r in runs], [3, 1])

    def test_fixed_sizes_split_only_at_the_peak(self):
        # the tape above at fixed sizes: y held to its run's end is live only
        # past the peak (add(x)'s temporary beside its output), which it
        # cannot become, so the run stays one graph
        def fn(x, y, c):
            return add(add(x)), add(y), add(c)

        n, m = 1 << 22, 1 << 18
        layouts = tuple(((k,), (1,)) for k in (n, m, m))
        f = HostTraceReplay(fn, memory="eager", trusted=TrustedInputs(layouts), freed_arguments=(1,), splits="walk")
        for _ in range(3):
            args = [torch.randn(k, device="cuda") for k in (n, m, m)]
            want = (args[0] + 6, args[1] + 3, args[2] + 3)
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            y = f.call_boxed(args)
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() - before
            self.assertEqual(y, want)
        self.assertEqual((f.replays, f.eager), (2, 0))
        self.assertEqual(peak, 4 * 2 * n)
        (v,) = f.variants
        runs = [s for s in v.captured.lowered.steps if isinstance(s, range)]
        self.assertEqual([len(r) for r in runs], [4])

    def test_fixed_sizes_run_buffer_splits_past_the_peak(self):
        # the tape above under "auto": a run buffer holds y and add(y)'s
        # output past "eager"'s peak too, and the split there lowers its own
        def fn(x, y, c):
            return add(add(x)), add(y), add(c)

        n, m = 1 << 22, 1 << 18
        layouts = tuple(((k,), (1,)) for k in (n, m, m))
        f = HostTraceReplay(fn, trusted=TrustedInputs(layouts), memory="auto", freed_arguments=(1,), splits="walk")
        for _ in range(3):
            args = [torch.randn(k, device="cuda") for k in (n, m, m)]
            want = (args[0] + 6, args[1] + 3, args[2] + 3)
            self.assertEqual(f.call_boxed(args), want)
        self.assertEqual((f.replays, f.eager), (2, 0))
        (v,) = f.variants
        self.assertFalse(any(s.order for s in v.memory.steps))
        runs = [s for s in v.captured.lowered.steps if isinstance(s, range)]
        self.assertEqual([len(r) for r in runs], [3, 1])

    def test_split_before_the_peak_allocation_first(self):
        # every argument freed, at fixed sizes, under "auto": the split just
        # before add(c), the peak's allocation, is taken first. The last-use
        # split just after add(x) lowers "eager"'s peak as much, but puts
        # add(add(x))'s dead output in add(c)'s run buffer beside add(c)'s,
        # as splits="peak" (the candidate that lowers it most) takes
        def fn(x, y, c):
            a = add(x)
            add(a)
            return a, add(c)

        k = 4096
        sizes = (k, k, 2 * k)
        layouts = tuple(((n,), (1,)) for n in sizes)
        for splits, want_peak, want_runs in (("walk", 4 * 2 * k, [2, 1]), ("peak", 4 * 3 * k, [1, 2])):
            f = HostTraceReplay(fn, trusted=TrustedInputs(layouts), memory="auto", freed_arguments=(0, 1, 2), splits=splits)
            for _ in range(3):
                args = [torch.randn(n, device="cuda") for n in sizes]
                want = (args[0] + 3, args[2] + 3)
                torch.cuda.synchronize()
                before = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                y = f.call_boxed(args)
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated() - before
                self.assertEqual(y, want)
            self.assertEqual((f.replays, f.eager), (2, 0))
            (v,) = f.variants
            runs = [s for s in v.captured.lowered.steps if isinstance(s, range)]
            self.assertEqual(peak, want_peak)
            self.assertEqual([len(r) for r in runs], want_runs)

    def test_a_split_lowering_the_peak_by_little_is_undone(self):
        # splitting before add(x) drops s and add(s)'s dead output at the
        # peak, 8 * k bytes: for k = n / 256 that is within 1/200 of the
        # replay's peak (x, s and both outputs), too little for "coarse" to
        # pay a graph launch for
        def fn(x, s):
            add(s)
            return add(x)

        n = 1 << 20
        for splits, k, split in (("coarse", n // 256, False), ("coarse", n // 64, True), ("walk", n // 256, True)):
            sizes = (n, k)
            layouts = tuple(((m,), (1,)) for m in sizes)
            f = HostTraceReplay(fn, trusted=TrustedInputs(layouts), memory="auto", freed_arguments=(0, 1), splits=splits)
            for _ in range(3):
                args = [torch.randn(m, device="cuda") for m in sizes]
                want = args[0] + 3
                torch.cuda.synchronize()
                before = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                y = f.call_boxed(args)
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated() - before
                self.assertEqual(y, want)
            self.assertEqual((f.replays, f.eager), (2, 0))
            (v,) = f.variants
            self.assertEqual(peak, 4 * (n - k if split else n + k))
            runs = [len(r) for r in v.captured.lowered.steps if isinstance(r, range)]
            self.assertEqual(runs, [1, 1] if split else [2])

    @parametrize("redispatch", [True, False])
    def test_a_slow_call_that_does_not_run_hands_the_call_back(self, redispatch):
        # under trust only the first trace runs its call: a later slow call (a
        # misaligned x's redispatch, or a trace where it refuses) hands the
        # arguments back, and the call again through call_boxed drops x after
        # its last use, which Python's reference would hold to the end
        def fn(x):
            return add(add(x), 1)

        n = 1 << 22
        layouts = (((n,), (1,)),)
        f = HostTraceReplay(fn, trusted=TrustedInputs(layouts), freed_arguments=(0,), handback=True)
        if not redispatch:
            f._redispatch = lambda *args: FoldRefused("")
        handed = []
        for offset in (0, 0, 1, 1):
            args = [torch.randn(n + 1, device="cuda")[offset : offset + n]]
            want = args[0] + 4
            y = f.call_boxed(args)
            if isinstance(y, Handback):
                handed.append(offset)
                args, y = list(y.args), None
                torch.cuda.synchronize()
                before = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                y = f.call_boxed(args)
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated() - before
            self.assertEqual(y, want)
        self.assertEqual(handed, [1])
        self.assertEqual(peak, 4 * n)
        self.assertEqual((f.traces, f.redispatches, f.eager), (1, 1, 0) if redispatch else (2, 0, 0))

    def test_distinct_shapes_reserve_as_eager(self):
        # each replay frees and reuses blocks as eager does, so the cache
        # grows as eager's over a stream of distinct sizes
        def three_step(x):
            return add(add(add(x)), 1)

        f = HostTraceReplay(three_step, memory="eager")
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

        f = HostTraceReplay(fn, memory="eager")
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
        f._fold = lambda *args: False
        f._redispatch = lambda *args: FoldRefused("")
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

    def test_memcpy_and_memset_nodes_are_set_when_they_change(self):
        # such a node holds its destination's address, not the allocation
        # there: after a free and reallocation (or a VMM remap) at the address
        # the unchanged node writes the new memory (core/learncost/
        # probe_memset_alloc.py), so it is set only when its parameters
        # change, as a kernel node is
        def fn(x):
            y = x.clone()
            return y, add(y), torch.empty_like(x).zero_()

        f = HostTraceReplay(fn)
        x = torch.randn(4096, device="cuda")
        want = (x, x + 3, torch.zeros_like(x))
        self.assertEqual(f(x), want)
        # the first replay sets both (its nodes hold the trace's values), the
        # next ones at the same addresses neither
        for n in (2, 0, 0):
            sets = torch._C._host_trace_memory_node_sets()
            self.assertEqual(f(x), want)
            self.assertEqual(torch._C._host_trace_memory_node_sets() - sets, n)
        # the cached segments released: the outputs' new allocations, at the
        # same addresses or not, are written
        for _ in range(3):
            torch.cuda.empty_cache()
            self.assertEqual(f(x), want)
        # another input moves the memcpy's source: it is set, the rest is not
        x2 = x.clone()
        sets = torch._C._host_trace_memory_node_sets()
        self.assertEqual(f(x2), want)
        self.assertGreaterEqual(torch._C._host_trace_memory_node_sets() - sets, 1)
        # a setter that fails leaves its node unknown, so the next call sets it
        (before,) = held_images(f)
        torch._C._host_trace_fail_after_setter(0)
        try:
            with self.assertRaisesRegex(RuntimeError, "injected after a setter"):
                f(x)
        finally:
            torch._C._host_trace_fail_after_setter(-1)
        self.assertEqual(held_images(f), [[None, *before[1:]]])
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
        # constants by value: a list is the tuple of its elements
        cases += [(x, s) for s in ("bb", (1,), [1], (1, 2), (1, 2.0), torch.float32, torch.float16)]
        keys = []
        for grad in (True, False):
            with torch.set_grad_enabled(grad):
                keys += [(f._native_key(a), _contract(a, check_global_state)) for a in cases]
        # a plain CUDA tensor has a native key: the pairs below compare keys at all
        self.assertIsNotNone(f._native_key((x,)))
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

    def _boxed_parity(self, fn, args, gen):
        # each call draws what eager's would from where eager's would; the
        # boxed step's hit enters no Python, and the Python step's counts and
        # bits are the same
        results = []
        for boxed in (True, False):
            step = _host_trace_native._boxed if boxed else lambda step: None
            with mock.patch.object(_host_trace_native, "_boxed", step):
                f = HostTraceReplay(fn)
                outs = []
                for i in range(4):
                    offset = gen.get_offset()
                    frames = []
                    if boxed and i == 3:
                        sys.setprofile(lambda frame, event, arg: event == "call" and frames.append(frame))
                    try:
                        outs.append(f(*args))
                    finally:
                        sys.setprofile(None)
                    self.assertEqual(frames, [])
                    taken = gen.get_offset() - offset
                    gen.set_offset(offset)
                    self.assertEqual(outs[-1], fn(*args), atol=0, rtol=0)
                    self.assertEqual(taken, gen.get_offset() - offset)
                    gen.set_offset(offset)
                (v,) = f.variants
                self.assertEqual(v.native.python_steps, 0 if boxed else 1)
                self.assertEqual(v.native.python_calls, 0 if boxed else f.replays)
                results.append(((f.traces, f.replays, f.eager), outs))
        self.assertEqual(results[0][0], (1, 3, 0))
        self.assertEqual(results[0], results[1], atol=0, rtol=0)

    def test_a_graphsafe_rng_step_runs_no_python(self):
        # it draws from its generator state, not the default generator's
        x = torch.randn(4096, device="cuda")
        state = torch.cuda.default_generators[0].clone_state()
        offset = torch.cuda.default_generators[0].get_offset()

        def fn(x):
            out, mask = graphsafe_run_with_rng_state(torch.ops.aten.native_dropout.default, add(x), 0.5, True, rng_state=state)
            return add(out), mask

        self._boxed_parity(fn, (x,), state)
        self.assertEqual(torch.cuda.default_generators[0].get_offset(), offset)

    @unittest.skipIf(not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION, "requires efficient attention")
    def test_a_host_seed_offset_step_runs_no_python(self):
        # its host seed and offset go to the device, as the trace predicts
        def fn(q, k, v):
            o, lse, seed, offset = torch.ops.aten._scaled_dot_product_efficient_attention(add(q), k, v, None, True, 0.125, True)
            return add(o), lse, seed, offset

        qkv = [torch.randn(2, 4, 128, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
        self._boxed_parity(fn, qkv, torch.cuda.default_generators[0])

    def test_an_index_with_a_none_runs_no_python(self):
        # its Tensor?[] indices, a None among them, are a boxed list
        def fn(x, idx):
            return add(torch.ops.aten.index.Tensor(add(x), [None, idx]))

        x = torch.randn(8, 64, 32, device="cuda")
        idx = torch.randint(0, 64, (5, 3), device="cuda")
        self._boxed_parity(fn, (x, idx), torch.cuda.default_generators[0])

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

    def test_pool_streams_capturing_elsewhere(self):
        # another thread captures on every stream of the pool
        # torch.cuda.Stream() hands out: the trace's and the replay's
        # captures are on streams of their own
        from cuda.bindings import runtime

        ready, stop = threading.Event(), threading.Event()

        def hold():
            streams = [torch.cuda.Stream() for _ in range(32)]
            relaxed = runtime.cudaStreamCaptureMode.cudaStreamCaptureModeRelaxed
            for s in streams:
                _check_cuda_bindings(runtime.cudaStreamBeginCapture(s.cuda_stream, relaxed))
            ready.set()
            stop.wait()
            for s in streams:
                _check_cuda_bindings(runtime.cudaGraphDestroy(_check_cuda_bindings(runtime.cudaStreamEndCapture(s.cuda_stream))))

        thread = threading.Thread(target=hold)
        thread.start()
        ready.wait()
        f = HostTraceReplay(two_step)
        try:
            for n in (1000, 3000, 1000):
                x = torch.randn(n, device="cuda")
                self.assertEqual(f(x), x + 4)
        finally:
            stop.set()
            thread.join()
        self.assertEqual((f.traces, f.replays, f.eager), (1, 2, 0))

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
        handle = segment.graph.register_replay_end_hook(seen.append)
        try:
            self.assertEqual(f(x), x + 4)
        finally:
            handle.remove()
        self.assertEqual(seen, [segment.graph, segment.graph])
        self.assertEqual(f.replays, 2)


def counter_bytes(x, n):
    # FlashInfer's get_trtllm_gen_multi_ctas_kv_counter_bytes: max(batch * heads, sm_count)
    return x[: max(32 * n, 152)] * 2


def affine(w, x):
    return add(x @ w)


def _restride(w):
    w.as_strided_(w.shape, (1, w.shape[0]))


def _reoffset(w):
    w.set_(w.untyped_storage(), 1, w.shape, w.stride())


def _resize(w):
    w.resize_(w.shape[0], w.shape[1] // 2)


def _reseat(w):
    w.set_(torch.randn_like(w))


def entry(path, fn, **kwargs):
    with mock.patch.object(torch.cuda._host_trace, "cpp_entry", path == "cpp"):
        return HostTraceReplay(fn, **kwargs)


def by_constant(x, c):
    return add(x, len(repr(c)))


def keyword_only(x, *, s=3):
    return add(x, s)


@unittest.skipIf(not TEST_CUDA, "requires CUDA")
@requires_cuda_python_bindings
@unittest.skipIf(not has_triton(), "requires triton")
@unittest.skipIf(not hasattr(torch._C, "_HostTraceVariant"), "requires the native path")
class TestEntryParity(TestCase):
    """The same calls on the C++ entry and on HostTraceReplay's Python dispatch
    (cpp_entry=False): outputs bitwise eager's, equal counters, and the calls
    each hands to Python from `steady` on."""

    def run_both(self, fn, calls, steady):
        counts, slow = {}, {}
        for path in ("python", "cpp"):
            f = entry(path, fn)
            for i, (args, kwargs) in enumerate(calls):
                if i == steady:
                    slow[path] = f.slow_calls
                self.assertEqual(f(*args, **kwargs), fn(*args, **kwargs), atol=0, rtol=0)
            counts[path] = (f.traces, f.replays, f.eager)
            slow[path] = f.slow_calls - slow[path]
        self.assertEqual(counts["python"], counts["cpp"])
        self.assertEqual(slow["python"], len(calls) - steady)
        return counts["cpp"], slow["cpp"]

    def test_the_steady_state_is_served_in_cpp(self):
        x = torch.randn(4096, device="cuda")
        self.assertEqual(self.run_both(two_step, [((x,), {})] * 4, 1), ((1, 3, 0), 0))

    def test_keyword_arguments(self):
        x = torch.randn(4096, device="cuda")
        calls = [((x,), {"s": 2}), ((x,), {"s": 5}), ((x, 7), {}), ((x,), {"s": 2})]
        self.assertEqual(self.run_both(add, calls, 1), ((1, 3, 0), 0))

    @parametrize("constants", [("a", "bb"), ((1,), (22,)), ([1], [22]), (torch.float32, torch.bfloat16), ((1, "a"), (1, 2.0))])
    def test_constant_arguments(self, constants):
        x = torch.randn(4096, device="cuda")
        calls = [((x, c), {}) for c in (*constants, *constants, constants[0])]
        self.assertEqual(self.run_both(by_constant, calls, 2), ((2, 3, 0), 0))

    def test_a_changed_list_is_another_constant(self):
        x = torch.randn(4096, device="cuda")
        for path in ("python", "cpp"):
            f = entry(path, by_constant)
            c = [1]
            for i in range(4):
                if i == 2:
                    c[0] = 22
                self.assertEqual(f(x, c), by_constant(x, c), atol=0, rtol=0)
            self.assertEqual((f.traces, f.replays, f.eager), (2, 2, 0))

    def test_a_keyword_only_argument_runs_eagerly(self):
        # _positional binds no keyword-only parameter: both paths run the call eagerly
        x = torch.randn(4096, device="cuda")
        self.assertEqual(self.run_both(keyword_only, [((x,), {"s": 2})] * 3, 0), ((0, 0, 3), 3))


@unittest.skipIf(not TEST_CUDA, "requires CUDA")
@requires_cuda_python_bindings
@unittest.skipIf(not has_triton(), "requires triton")
class TestBuiltinMinMax(TestCase):
    @parametrize("symbolic", ["ir", "sympy"])
    def test_max_of_a_symbolic_int_takes_no_guard(self, symbolic):
        f = HostTraceReplay(counter_bytes)
        x = torch.randn(4096, device="cuda")
        with mock.patch("torch.cuda._host_trace.symbolic", symbolic):
            # 32 * n < 152 at 2 and 3, not at 6 or 64
            for n in (2, 6, 3, 64):
                self.assertEqual(f(x, n), counter_bytes(x, n), atol=0, rtol=0)
        self.assertEqual((f.traces, len(f.variants), f.replays, f.eager), (1, 1, 3, 0))
        self.assertIs(builtins.max, _host_trace_replay._builtin_max)
        self.assertIs(builtins.min, _host_trace_replay._builtin_min)

    @parametrize("symbolic", ["ir", "sympy"])
    def test_without_the_switch_max_guards_and_says_where(self, symbolic):
        f = HostTraceReplay(counter_bytes, trace_builtin_minmax=False)
        x = torch.randn(4096, device="cuda")
        with mock.patch("torch.cuda._host_trace.symbolic", symbolic):
            for n in (2, 6, 3, 64):
                self.assertEqual(f(x, n), counter_bytes(x, n), atol=0, rtol=0)
        self.assertEqual((f.traces, len(f.variants), f.replays), (2, 2, 2))
        line = counter_bytes.__code__.co_firstlineno + 2
        note = f"builtin max/min at test_cuda_host_trace_replay.py:{line} with trace_builtin_minmax=False; use torch.sym_max"
        # only the comparison's guard: not the slice's or its kernel's on the same line
        noted = [g for g in f.variants[0].tape.shape_env.guards if g.sloc.framework_loc is not None]
        self.assertEqual([g.sloc.framework_loc for g in noted], [note])
        self.assertRegex(str(noted[0].expr), r"^32\*s\d+ < 152$")

    def test_other_calls_run_the_builtin(self):
        seen = []

        def fn(x, n):
            seen.append((max("ab", "b"), min([3.5, 2]), max([], default=None), min(v for v in (4, 2))))
            with self.assertRaisesRegex(TypeError, "not iterable"):
                max(n)
            # a key compares, so it guards as without the switch
            return x[: max(32 * n, 152, key=lambda v: v)] * 2

        f = HostTraceReplay(fn)
        x = torch.randn(4096, device="cuda")
        for n in (2, 6, 3):
            self.assertEqual(f(x, n), counter_bytes(x, n), atol=0, rtol=0)
        self.assertEqual((f.traces, len(f.variants), f.replays), (2, 2, 1))
        self.assertEqual(set(seen), {("b", 2, None, 2)})

    def test_nested_holds_restore_the_builtins(self):
        hold = _host_trace_replay._builtin_minmax
        with hold:
            wrapped = builtins.max
            self.assertIsNot(wrapped, _host_trace_replay._builtin_max)
            with hold:
                self.assertIs(builtins.max, wrapped)
            self.assertIs(builtins.max, wrapped)
            self.assertEqual((max(3, 5), min([4, 1]), max([], default=0)), (5, 1, 0))
        self.assertIs(builtins.max, _host_trace_replay._builtin_max)
@unittest.skipIf(not hasattr(torch._C, "_HostTraceVariant"), "requires the native path")
class TestStaticPrefix(TestCase):
    def check(self, f, w, x):
        self.assertEqual(f(w, x), affine(w, x), atol=0, rtol=0)

    def test_unchanged_statics_skip_their_rows(self):
        f = HostTraceReplay(affine, static_prefix=1)
        w = torch.randn(65, 64, device="cuda")[1:]
        for _ in range(6):
            self.check(f, w, torch.randn(8, 64, device="cuda"))
        # the trace, then a full hit that records w; every later hit skips it
        self.assertEqual((f.traces, f.replays, f.static_hits), (1, 5, 4))
        (variant,) = f.variants
        dynamic, total = torch._C._host_trace_static_split(variant.native)
        self.assertGreater(dynamic, 0)
        self.assertLess(dynamic, total)
        # neither the version counter nor the values are in the key: the
        # replay reads w's new values
        w.mul_(2)
        self.check(f, w, torch.randn(8, 64, device="cuda"))
        self.assertEqual(f.static_hits, 5)
        # a new size of x is no change to w
        for m in (24, 24, 8):
            self.check(f, w, torch.randn(m, 64, device="cuda"))
        self.assertGreaterEqual(f.static_hits, 6)

    def test_unchanged_dynamic_leaves_evaluate_no_row(self):
        f = HostTraceReplay(affine, static_prefix=1)
        w = torch.randn(64, 64, device="cuda")
        x = torch.randn(8, 64, device="cuda")
        y = torch.randn(8, 64, device="cuda")
        for _ in range(3):
            self.check(f, w, x)
        (variant,) = f.variants
        dynamic = torch._C._host_trace_static_split(variant.native)[0]
        before = torch._C._host_trace_delta_counts()
        for _ in range(3):
            self.check(f, w, x)
        skips, partials, fulls, rows = (
            b - a for a, b in zip(before, torch._C._host_trace_delta_counts())
        )
        self.assertEqual((skips, partials, fulls, rows), (3, 0, 0, 0))
        # y differs from x in its address alone (input 1, a pointer leaf)
        before = torch._C._host_trace_delta_counts()
        for z in (y, x, y):
            self.check(f, w, z)
        skips, partials, fulls, rows = (
            b - a for a, b in zip(before, torch._C._host_trace_delta_counts())
        )
        self.assertEqual((skips, partials, fulls), (0, 3, 0))
        self.assertLess(rows, 3 * dynamic)
        changed = torch._C._host_trace_delta_changed(variant.native)
        self.assertEqual([c[:2] for c in changed], [(1, 1)])
        torch._C._host_trace_delta(False)
        try:
            self.check(f, w, y)
        finally:
            torch._C._host_trace_delta(True)
        # a new size of x, and back
        for m in (24, 24, 8, 8):
            self.check(f, w, torch.randn(m, 64, device="cuda"))
        self.check(f, w, x)

    def test_without_a_prefix_no_hit_is_static(self):
        f = HostTraceReplay(affine)
        w = torch.randn(64, 64, device="cuda")
        for _ in range(4):
            self.check(f, w, torch.randn(8, 64, device="cuda"))
        self.assertEqual((f.replays, f.static_hits), (3, 0))
        (variant,) = f.variants
        self.assertEqual(torch._C._host_trace_static_split(variant.native)[0], 0)

    @parametrize(
        "change",
        [
            subtest(lambda w: w.clone(), name="new_tensor"),
            subtest(lambda w: w.half(), name="dtype"),
            subtest(lambda w: w.t(), name="view"),
            subtest(_restride, name="strides_in_place"),
            subtest(_reoffset, name="offset_in_place"),
            subtest(_resize, name="resize_in_place"),
            subtest(_reseat, name="storage_in_place"),
        ],
    )
    def test_a_changed_static_takes_the_full_path(self, change):
        f = HostTraceReplay(affine, static_prefix=1)
        base = torch.randn(64 * 64 + 1, device="cuda")
        w = base[: 64 * 64].view(64, 64)
        x = torch.randn(8, 64, device="cuda")
        for _ in range(3):
            self.check(f, w, x)
        self.assertEqual(f.static_hits, 1)
        w2 = change(w)
        if w2 is None:
            w2 = w
        x2 = x.to(w2.dtype)
        # a trace or a full hit; then a full hit or the first static one
        self.check(f, w2, x2)
        self.assertEqual(f.static_hits, 1)
        self.check(f, w2, x2)
        hits = f.static_hits
        self.check(f, w2, x2)
        self.assertEqual(f.static_hits, hits + 1)
        if w2 is not w:
            self.check(f, w, x)
            self.assertEqual(f.static_hits, hits + 1)
            self.check(f, w, x)
            self.assertEqual(f.static_hits, hits + 2)

    def test_a_changed_grad_mode_takes_the_full_path(self):
        f = HostTraceReplay(affine, static_prefix=1)
        w = torch.randn(64, 64, device="cuda")
        x = torch.randn(8, 64, device="cuda")
        with torch.no_grad():
            for _ in range(3):
                self.check(f, w, x)
        self.assertEqual(f.static_hits, 1)
        with torch.enable_grad():
            self.check(f, w, x)
        self.assertEqual(f.static_hits, 1)

    def test_changed_global_state_takes_the_full_path(self):
        f = HostTraceReplay(affine, static_prefix=1, check_global_state=True)
        w = torch.randn(64, 64, device="cuda")
        x = torch.randn(8, 64, device="cuda")
        for _ in range(3):
            self.check(f, w, x)
        self.assertEqual(f.static_hits, 1)
        prior = torch.backends.cuda.matmul.fp32_precision
        try:
            torch.backends.cuda.matmul.fp32_precision = "ieee" if prior == "tf32" else "tf32"
            self.check(f, w, x)
            self.assertEqual(f.static_hits, 1)
        finally:
            torch.backends.cuda.matmul.fp32_precision = prior
        self.check(f, w, x)
        self.assertEqual(f.traces, 2)

    def test_a_staged_copy_past_its_buffer_is_eager(self):
        # a static staging buffer sliced to a dynamic length: a longer input
        # misses and raises as eager does, never a clamped copy
        def staged(buf, x):
            return buf[: x.shape[0]].copy_(x) * 2

        f = HostTraceReplay(staged, static_prefix=1)
        buf = torch.zeros(64, device="cuda")
        for n in (8, 16, 8, 16, 32, 64):
            x = torch.randn(n, device="cuda")
            self.assertEqual(f(buf, x), x * 2, atol=0, rtol=0)
        replays = f.replays
        with self.assertRaisesRegex(RuntimeError, "size"):
            f(buf, torch.randn(100, device="cuda"))
        self.assertEqual(f.replays, replays)
        x = torch.randn(16, device="cuda")
        self.assertEqual(f(buf, x), x * 2, atol=0, rtol=0)
        self.assertEqual(f.replays, replays + 1)

    def test_fewer_arguments_than_the_prefix_take_the_full_path(self):
        f = HostTraceReplay(two_step, static_prefix=2)
        x = torch.randn(1000, device="cuda")
        for _ in range(3):
            self.assertEqual(f(x), x + 4)
        self.assertEqual((f.replays, f.static_hits), (2, 0))

    def test_alternating_sizes_try_the_variant_that_served_them(self):
        # one family, a variant per size; each call tries the variant that
        # last served its sizes first, so no evaluate misses
        def grows(w, x):
            y = add(x @ w)
            for _ in range(x.shape[0] // 8):
                y = add(y, 1)
            return y

        f = HostTraceReplay(grows, static_prefix=1)
        w = torch.randn(64, 64, device="cuda")
        sizes = (8, 16, 24, 32)
        for m in sizes * 3:
            x = torch.randn(m, 64, device="cuda")
            self.assertEqual(f(w, x), grows(w, x), atol=0, rtol=0)
        self.assertEqual((len(f.variants), len(f._families)), (4, 1))
        torch._C._host_trace_phase_times(True)
        try:
            for m in sizes[::-1] * 2 + sizes * 2:
                x = torch.randn(m, 64, device="cuda")
                self.assertEqual(f(w, x), grows(w, x), atol=0, rtol=0)
        finally:
            phases = torch._C._host_trace_phase_times(False)
        self.assertEqual(phases[9], 0)

    def test_switches_off_take_the_earlier_paths(self):
        # C2's variant order, C6's placement reuse and C9's delta evaluate are
        # on by default; each switched off, calls take the path before it
        def grows(w, x):
            y = add(x @ w)
            for _ in range(x.shape[0] // 8):
                y = add(y, 1)
            return y

        w = torch.randn(64, 64, device="cuda")
        sizes = (8, 16, 24, 32)
        switches = (torch._C._host_trace_variant_order, torch._C._host_trace_placement_reuse, torch._C._host_trace_delta)
        for switch in switches:
            f = HostTraceReplay(grows, static_prefix=1, memory="run_buffer")
            switch(False)
            try:
                for m in sizes * 3:
                    x = torch.randn(m, 64, device="cuda")
                    self.assertEqual(f(w, x), grows(w, x), atol=0, rtol=0)
                torch._C._host_trace_phase_times(True)
                try:
                    for m in sizes[::-1] * 2 + sizes:
                        x = torch.randn(m, 64, device="cuda")
                        self.assertEqual(f(w, x), grows(w, x), atol=0, rtol=0)
                finally:
                    phases = torch._C._host_trace_phase_times(False)
            finally:
                switch(True)
            self.assertEqual((len(f.variants), f.eager), (4, 0))
            # registration order: a later size misses the variants before it
            self.assertEqual(phases[9] > 0, switch is torch._C._host_trace_variant_order)


@unittest.skipIf(not TEST_CUDA, "requires CUDA")
@requires_cuda_python_bindings
@unittest.skipIf(not has_triton(), "requires triton")
@unittest.skipIf(not hasattr(torch._C, "_HostTraceVariant"), "requires the native path")
class TestPlannedMemory(TestCase):
    def setUp(self):
        super().setUp()
        torch.cuda.synchronize()
        torch._C._host_trace_release_held_buffers()

    def check(self, f, fn, *args):
        self.assertEqual(f(*args), fn(*args), atol=0, rtol=0)

    def test_layout_never_overlaps(self):
        rng = random.Random(0)
        choices = (0, 1, 100, 512, 513, 4096, 70000)
        for _ in range(200):
            n = rng.randint(1, 12)
            program = IntegerProgram([rng.choice(choices) for _ in range(n)])
            spans = {}
            for k in range(n):
                start, end = sorted(rng.sample(range(40), 2))
                spans[k] = (program.emit("boxed", k), (0, start), (0, end) if rng.random() < 0.7 else (1, _AFTER))
            # first fit's at the traced sizes, or it raises
            offsets, total = _layout(program, spans)
            compiled = compile_program(program)
            for _ in range(5):
                sizes = [rng.choice(choices) for _ in range(n)]
                _, v = compiled.evaluate_inputs(sizes)
                ranges = [(v[offsets[k]], v[offsets[k]] + -(-sizes[k] // 512) * 512) for k in range(n)]
                for j in range(n):
                    self.assertEqual(ranges[j][0] % 512, 0)
                    self.assertLessEqual(ranges[j][1], v[total])
                    for k in range(j):
                        if spans[j][1] < spans[k][2] and spans[k][1] < spans[j][2]:
                            apart = ranges[j][1] <= ranges[k][0] or ranges[k][1] <= ranges[j][0]
                            self.assertTrue(apart or sizes[j] == 0 or sizes[k] == 0)

    def test_only_the_outputs_are_allocated(self):
        f = HostTraceReplay(five_step, memory="held")
        x = torch.randn(1 << 20, device="cuda")
        for _ in range(3):
            self.check(f, five_step, x)
        (v,) = f.variants
        self.assertEqual([k for k, *_ in v.memory.planned], [0, 1, 2])
        self.assertEqual(v.memory.outputs, (3,))
        images = torch._C._host_trace_held_images(v.native)
        before = torch.cuda.memory_stats()["allocation.all.allocated"]
        y = f(x)
        self.assertEqual(torch.cuda.memory_stats()["allocation.all.allocated"], before + 1)
        self.assertEqual(y, five_step(x), atol=0, rtol=0)
        # the temporaries kept their addresses: only the output's node may move
        self.assertEqual(torch._C._host_trace_held_images(v.native)[:3], images[:3])
        # add(add(add(x))) takes add(x)'s bytes
        self.assertEqual(torch._C._host_trace_held_buffer_bytes(), 2 * x.nbytes)
        self.assertEqual((f.traces, f.replays, f.eager), (1, 3, 0))

    def test_offsets_follow_the_shapes(self):
        f = HostTraceReplay(five_step, memory="held")
        for n in (1 << 20, 1 << 20, 4096, 1 << 22, 1000, 1 << 20, 3):
            self.check(f, five_step, torch.randn(n, device="cuda"))
        self.assertEqual(f.eager, 0)
        # grown to the largest call's, and kept
        self.assertEqual(torch._C._host_trace_held_buffer_bytes(), 2 * 4 * (1 << 22))

    def test_eager_steps_read_and_write_planned_tensors(self):
        def fn(x, w):
            a = add(x)
            a.mul_(2)
            b = add(a).sin()
            return add(torch.matmul(add(b).view(4, -1, w.shape[0]), w), 1)

        f = HostTraceReplay(fn, memory="held", opaque=())
        w = torch.randn(64, 32, device="cuda")
        for m in (64, 64, 128, 64):
            self.check(f, fn, torch.randn(4 * m, 64, device="cuda"), w)
        self.assertEqual(f.eager, 0)
        self.assertTrue(any(tensor for v in f.variants for *_, tensor in v.memory.planned))

    def test_entries_share_a_buffer_per_stream(self):
        f = HostTraceReplay(five_step, memory="held")
        g = HostTraceReplay(two_step, memory="held")
        x = torch.randn(1 << 20, device="cuda")
        for _ in range(3):
            self.assertEqual(g(f(x)), two_step(five_step(x)), atol=0, rtol=0)
        self.assertEqual(torch._C._host_trace_held_buffer_bytes(), 2 * x.nbytes)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            y = f(x)
        torch.cuda.current_stream().wait_stream(s)
        self.assertEqual(y, five_step(x), atol=0, rtol=0)
        self.assertEqual(torch._C._host_trace_held_buffer_bytes(), 4 * x.nbytes)
        torch.cuda.synchronize()
        torch._C._host_trace_release_held_buffers()
        self.assertEqual(torch._C._host_trace_held_buffer_bytes(), 0)
        self.check(f, five_step, x)

    def test_a_nested_replay_takes_its_own_buffer(self):
        # b is in the outer replay's buffer across the inner replay
        def fn(x):
            b = add(add(x))
            return b + calls_entry(add(x, 1))

        REENTERED[:] = [HostTraceReplay(five_step, memory="held")]
        try:
            f = HostTraceReplay(fn, memory="held")
            for _ in range(4):
                self.check(f, fn, torch.randn(1 << 16, device="cuda"))
            self.assertEqual(f.eager, 0)
            self.assertGreater(REENTERED[0].replays, 0)
        finally:
            REENTERED.clear()

    def test_an_eager_output_viewing_a_planned_tensor_disagrees(self):
        def fn(x):
            return add(aliases(add(x)))

        f = HostTraceReplay(fn, memory="held")
        x = torch.randn(4096, device="cuda")
        with self.assertRaisesRegex(AssertionError, "returned a view of an allocation"):
            for _ in range(3):
                self.check(f, fn, x)
        # the op's metadata is not its fake's: later traces decline
        for _ in range(3):
            self.check(f, fn, x)


@unittest.skipIf(not TEST_CUDA, "requires CUDA")
@requires_cuda_python_bindings
@unittest.skipIf(not has_triton(), "requires triton")
@unittest.skipIf(not hasattr(torch._C, "_HostTraceVariant"), "requires the native path")
class TestDirtyRecords(TestCase):
    def test_unchanged_records_are_not_packed(self):
        f = HostTraceReplay(five_step)
        xs = [torch.randn(n, device="cuda") for n in (4096, 8192)]
        for x in xs * 2:
            self.assertEqual(f(x), five_step(x), atol=0, rtol=0)
        traces = f.traces
        # the allocator hands out the same addresses: no launch's rows change
        deltas = []
        for _ in range(4):
            packs = torch._C._host_trace_kernel_packs()
            self.assertEqual(f(xs[1]), five_step(xs[1]), atol=0, rtol=0)
            deltas.append(torch._C._host_trace_kernel_packs() - packs)
        self.assertEqual(min(deltas), 0)
        # every launch reads the size, and its node changes
        packs, sets = torch._C._host_trace_kernel_packs(), torch._C._host_trace_kernel_node_sets()
        self.assertEqual(f(xs[0]), five_step(xs[0]), atol=0, rtol=0)
        self.assertEqual(torch._C._host_trace_kernel_packs() - packs, 4)
        self.assertEqual(torch._C._host_trace_kernel_node_sets() - sets, 4)
        # kept outputs move the later launches' pointers
        outs = [f(xs[0]) for _ in range(3)]
        for out in outs:
            self.assertEqual(out, five_step(xs[0]), atol=0, rtol=0)
        self.assertEqual(f.traces, traces)

    def test_alternating_statics_repatch_their_readers(self):
        f = HostTraceReplay(affine, static_prefix=1)
        ws = [torch.randn(64, 64, device="cuda") for _ in range(2)]
        x = torch.randn(8, 64, device="cuda")
        for w in [ws[0]] * 3 + ws * 3 + [ws[1]] * 3:
            self.assertEqual(f(w, x), affine(w, x), atol=0, rtol=0)
        self.assertGreater(f.static_hits, 0)

    @parametrize("library_impls", [False, True])
    def test_a_nested_replay_of_the_variant_repatches_the_outer(self, library_impls):
        # the inner call patches the outer's later launches at its own rows; with library_impls on, calls_entry's
        # body re-enters f, so it is an eager step (library_impls_reentry)
        def fn(x):
            return add(calls_entry(add(x)), 1)

        nested = [False]

        def inner(x):
            if nested[0]:
                return x * 2
            nested[0] = True
            try:
                h = x.numel() // 2
                return torch.cat([f(x[:h].contiguous()), x[h:] * 2])
            finally:
                nested[0] = False

        def ref(x, depth=0):
            y = add(x)
            h = y.numel() // 2
            z = y * 2 if depth else torch.cat([ref(y[:h].contiguous(), 1), y[h:] * 2])
            return add(z, 1)

        REENTERED[:] = [inner]
        try:
            f = HostTraceReplay(fn)
            with mock.patch.object(torch.cuda._host_trace, "library_impls", library_impls):
                for n in (8192, 4096) * 4:
                    x = torch.randn(n, device="cuda")
                    self.assertEqual(f(x), ref(x), atol=0, rtol=0)
            self.assertEqual((f.traces, f.replays, f.eager), (2, 14, 0))
        finally:
            REENTERED.clear()

    @mock.patch("torch.cuda._host_trace.library_impls", True)
    def test_a_nested_replay_inside_a_library_impl_is_an_eager_step(self):
        # inner reads `nested`, which its own call sets: traced, calls_entry would
        # replay the state of the call it was traced at
        def fn(x):
            return add(calls_entry(add(x)), 1)

        nested = [False]

        def inner(x):
            if nested[0]:
                return x * 2
            nested[0] = True
            try:
                h = x.numel() // 2
                return torch.cat([f(x[:h].contiguous()), x[h:] * 2])
            finally:
                nested[0] = False

        def ref(x, depth=0):
            y = add(x)
            h = y.numel() // 2
            z = y * 2 if depth else torch.cat([ref(y[:h].contiguous(), 1), y[h:] * 2])
            return add(z, 1)

        REENTERED[:] = [inner]
        self.addCleanup(REENTERED.clear)
        f = HostTraceReplay(fn)
        for n in (8192, 4096) * 4:
            x = torch.randn(n, device="cuda")
            self.assertEqual(f(x), ref(x), atol=0, rtol=0)
        self.assertGreater(f.replays, 4)
        reasons = {r.reason.split(": ", 1)[1] for v in f.variants for _, r in v.tape.launches if isinstance(r, EagerCall)}
        self.assertEqual(reasons, {
            "host_trace: host_trace_test.calls_entry.default's Library.impl body calls a host-trace entry (declined)",
            "the trace runs inside a Library.impl body, whose state its body may read",
        })



@dataclasses.dataclass
class Batch:
    x: torch.Tensor
    n: int
    mode: object
    note: object = None
    total: int = 0


@dataclasses.dataclass
class BoundOut:
    y: object = None
    z: object = None
    tag: str = "out"


class BoundRoot:
    def __init__(self):
        self.flag = "on"


def bound_fn(w, a, x, n):
    return affine(w, a), add(x, n)


@unittest.skipIf(not TEST_CUDA, "requires CUDA")
@requires_cuda_python_bindings
@unittest.skipIf(not has_triton(), "requires triton")
@unittest.skipIf(not hasattr(torch._C, "_HostTraceBound"), "requires the bound call")
class TestBoundCall(TestCase):
    MODE, OTHER = object(), object()

    def setUp(self):
        super().setUp()
        self.w = torch.randn(64, 64, device="cuda")
        self.a = torch.randn(8, 64, device="cuda")
        self.x = torch.randn(4096, device="cuda")
        self.root = BoundRoot()

    def bound(self, f, limits=None):
        b = torch._C._HostTraceBound((self.w,), 1, 1, (self.root,))
        below = ((0, "total", 0, "n", tuple(limits)),) if limits else ()
        same = ((0, "mode", self.MODE), (1, "flag", "on"))
        b.add(f, (Batch,), same, below, ((0, "note"),), ((0, "x"),), ((0, "n"),), BoundOut(), ("y", "z"))
        return b

    def entry(self):
        f = HostTraceReplay(bound_fn, static_prefix=1, opaque=())
        for _ in range(2):
            f(self.w, self.a, self.x, 3)
        return f

    def check(self, out, a, x, n):
        self.assertIs(type(out), BoundOut)
        self.assertEqual(out.tag, "out")
        y, z = bound_fn(self.w, a, x, n)
        self.assertEqual(out.y, y, atol=0, rtol=0)
        self.assertEqual(out.z, z, atol=0, rtol=0)

    def test_a_bound_call_is_the_entry_call(self):
        f = self.entry()
        b = self.bound(f)
        replays = f.replays
        for n in (3, 3, 5):
            a, x = torch.randn_like(self.a), torch.randn_like(self.x)
            self.check(b(a, Batch(x, n, self.MODE)), a, x, n)
        self.assertEqual(b.hits(), [3])
        self.assertEqual(f.replays, replays + 3)
        self.assertGreater(f.static_hits, 0)
        # the template is copied, not shared
        out = b(self.a, Batch(self.x, 3, self.MODE))
        out.tag = "changed"
        self.assertEqual(b(self.a, Batch(self.x, 3, self.MODE)).tag, "out")

    def test_a_call_no_guard_holds_for_runs_nothing(self):
        f = self.entry()
        b = self.bound(f, limits=[100] * 8)
        replays = f.replays

        class Sub(Batch):
            pass

        x = self.x
        cases = [
            lambda: b(self.a, Batch(x, 3, self.OTHER)),
            lambda: b(self.a, Batch(x, 3, self.MODE, note=x)),
            lambda: b(self.a, Batch(x.cpu(), 3, self.MODE)),
            lambda: b(self.a, Batch(None, 3, self.MODE)),
            lambda: b(self.a, Batch(x, 3.0, self.MODE)),
            lambda: b(self.a, Batch(x, True, self.MODE)),
            lambda: b(self.a, Batch(x, 3, self.MODE, total=100)),
            lambda: b(self.a, Batch(x, 8, self.MODE)),
            lambda: b(self.a, Batch(x, 3, self.MODE, total=1 << 70)),
            lambda: b(self.a, Sub(x, 3, self.MODE)),
            lambda: b(self.a, 3),
            lambda: b(self.a),
            lambda: b(self.a, Batch(x, 3, self.MODE), Batch(x, 3, self.MODE)),
            lambda: b(self.a, batch=Batch(x, 3, self.MODE)),
        ]
        for case in cases:
            self.assertIs(case(), NotImplemented)
        self.root.flag = "off"
        self.assertIs(b(self.a, Batch(x, 3, self.MODE)), NotImplemented)
        self.root.flag = "on"
        self.assertEqual(f.replays, replays)
        self.assertEqual(b.hits(), [0])
        self.check(b(self.a, Batch(x, 3, self.MODE, total=99)), self.a, x, 3)
        self.assertEqual(b.hits(), [1])

    def test_a_call_that_is_no_replay_hit_runs_nothing(self):
        f = self.entry()
        b = self.bound(f)
        traces, replays = f.traces, f.replays
        x = torch.randn(4096, device="cuda", dtype=torch.float64)
        self.assertIs(b(self.a, Batch(x, 3, self.MODE)), NotImplemented)
        self.assertEqual((f.traces, f.replays), (traces, replays))
        f(self.w, self.a, x, 3)
        self.check(b(self.a, Batch(x, 3, self.MODE)), self.a, x, 3)
        self.assertEqual(b.hits(), [1])

    @mock.patch("torch.cuda._host_trace.library_impls", False)
    def test_a_bound_call_from_within_a_bound_call(self):
        INNER = object()

        def outer_fn(w, a, x, n):
            return affine(w, a), add(calls_entry(add(x, n)), 1)

        def inner(x):
            out = b(self.a, Batch(x, 2, INNER))
            self.assertIsNot(out, NotImplemented)
            return out.z

        def ref(x, n):
            return add(add(add(x, n), 2), 1)

        REENTERED[:] = [lambda x: add(x, 2)]
        try:
            f = self.entry()
            g = HostTraceReplay(outer_fn, static_prefix=1)
            for _ in range(2):
                g(self.w, self.a, self.x, 3)
            b = self.bound(g)
            b.add(f, (Batch,), ((0, "mode", INNER),), (), (), ((0, "x"),), ((0, "n"),), BoundOut(), ("y", "z"))
            REENTERED[:] = [inner]
            for n in (3, 3, 4):
                x = torch.randn_like(self.x)
                out = b(self.a, Batch(x, n, self.MODE))
                self.assertEqual(out.y, affine(self.w, self.a), atol=0, rtol=0)
                self.assertEqual(out.z, ref(x, n), atol=0, rtol=0)
            self.assertEqual(b.hits(), [3, 3])
        finally:
            REENTERED.clear()

    @mock.patch("torch.cuda._host_trace.library_impls", True)
    def test_a_bound_call_inside_a_library_impl_is_an_eager_step(self):
        # traced, calls_entry would inline the inner call: no hit of f's plan
        INNER = object()

        def outer_fn(w, a, x, n):
            return affine(w, a), add(calls_entry(add(x, n)), 1)

        def inner(x):
            # under a trace x is a traced tensor, which no plan takes: the
            # caller's own path calls f
            out = b(self.a, Batch(x, 2, INNER))
            return f(self.w, self.a, x, 2)[1] if out is NotImplemented else out.z

        REENTERED[:] = [inner]
        self.addCleanup(REENTERED.clear)
        f = self.entry()
        g = HostTraceReplay(outer_fn, static_prefix=1, opaque=())
        b = self.bound(g)
        b.add(f, (Batch,), ((0, "mode", INNER),), (), (), ((0, "x"),), ((0, "n"),), BoundOut(), ("y", "z"))
        # g's trace calls b, which calls f
        for _ in range(2):
            g(self.w, self.a, self.x, 3)
        for n in (3, 3, 4, 4):
            x = torch.randn_like(self.x)
            out = b(self.a, Batch(x, n, self.MODE))
            self.assertEqual(out.y, affine(self.w, self.a), atol=0, rtol=0)
            self.assertEqual(out.z, add(add(add(x, n), 2), 1), atol=0, rtol=0)
        # g's first call and its warm-up hit f's plan, then each replay's eager step
        self.assertEqual(b.hits(), [4, 6])
        self.assertGreater(g.replays, 0)
        reasons = [r.reason for v in g.variants for _, r in v.tape.launches if isinstance(r, EagerCall) and r.reason]
        self.assertEqual(reasons, ["host_trace_test.calls_entry.default's kernel declines: host_trace: host_trace_test.calls_entry.default's Library.impl body calls a host-trace entry (declined)"])  # noqa: B950

    def test_flat_outputs_without_a_template(self):
        f = self.entry()
        b = torch._C._HostTraceBound((self.w,), 1, 1, ())
        b.add(f, (Batch,), (), (), (), ((0, "x"),), ((0, "n"),), None, ())
        y, z = b(self.a, Batch(self.x, 3, self.MODE))
        self.assertEqual(y, affine(self.w, self.a), atol=0, rtol=0)
        self.assertEqual(z, add(self.x, 3), atol=0, rtol=0)

    def trusted(self, f):
        b = torch._C._HostTraceBound((self.w,), 1, 1, (self.root,), trust_statics=True)
        b.add(f, (Batch,), ((0, "mode", self.MODE),), (), (), ((0, "x"),), ((0, "n"),), BoundOut(), ("y", "z"))
        return b

    @unittest.skipIf(not hasattr(torch._C._HostTraceBound, "statics_changed"), "requires the one-call options")
    def test_the_one_call_options(self):
        f = self.entry()
        calls = []

        def fallback(*args, **kwargs):
            calls.append((args, kwargs))
            return "fallback"

        # sources: the batch read off the root (0), the root (1)
        b = torch._C._HostTraceBound(
            (self.w,), 1, 0, (self.root,), derived=((0, "batch"),), none_kwargs=("proxy",), resets=((1, "flag", "reset"),), fallback=fallback
        )
        b.add(f, (Batch,), ((0, "mode", self.MODE),), (), (), ((0, "x"),), ((0, "n"),), BoundOut(), ("y", "z"))
        x = torch.randn_like(self.x)
        self.root.batch = Batch(x, 3, self.MODE)
        self.check(b(self.a, proxy=None), self.a, x, 3)
        self.assertEqual(self.root.flag, "reset")
        self.check(b(self.a), self.a, x, 3)
        self.assertEqual((b.hits(), calls), ([2], []))
        self.root.flag = "on"
        misses = [((self.a,), {"proxy": x}), ((self.a,), {"other": None}), ((self.a, 1), {})]
        for args, kwargs in misses:
            self.assertEqual(b(*args, **kwargs), "fallback")
        self.root.batch = Batch(x, 3, self.OTHER)
        self.assertEqual(b(self.a), "fallback")
        del self.root.batch
        self.assertEqual(b(self.a), "fallback")
        self.assertEqual(len(calls), 5)
        self.assertIs(calls[0][1]["proxy"], x)
        self.assertEqual([len(a) for a, _ in calls], [1, 1, 2, 1, 1])
        self.assertEqual((self.root.flag, b.hits()), ("on", [2]))

    @unittest.skipIf(not hasattr(torch._C._HostTraceBound, "statics_changed"), "requires the one-call options")
    def test_trusted_statics_changed_in_place_take_the_full_path(self):
        f = self.entry()
        b = self.trusted(f)
        for _ in range(3):
            self.check(b(self.a, Batch(self.x, 3, self.MODE)), self.a, self.x, 3)
        hits = f.static_hits
        self.w.set_(torch.randn_like(self.w))
        b.statics_changed()
        # the full path: a full hit that records the new w, then static hits
        self.check(b(self.a, Batch(self.x, 3, self.MODE)), self.a, self.x, 3)
        self.assertEqual(f.static_hits, hits)
        for _ in range(2):
            self.check(b(self.a, Batch(self.x, 3, self.MODE)), self.a, self.x, 3)
        self.assertEqual(f.static_hits, hits + 2)

    @unittest.skipIf(not hasattr(torch._C, "_host_trace_trusted_statics_check"), "requires the native safety checks")
    @parametrize("change", ["set_", "data", "swap", "resize", "t_", "as_strided_", "strided_set_"])
    def test_trusted_statics_changed_without_notice_take_the_full_path(self, change):
        torch._C._host_trace_trusted_statics_check(True)
        self.addCleanup(torch._C._host_trace_trusted_statics_check, False)
        f = self.entry()
        b = self.trusted(f)
        for _ in range(3):
            self.check(b(self.a, Batch(self.x, 3, self.MODE)), self.a, self.x, 3)
        hits = f.static_hits
        new = torch.randn_like(self.w)
        if change == "set_":
            self.w.set_(new)
        elif change == "data":
            self.w.data = new
        elif change == "swap":
            torch.utils.swap_tensors(self.w, new)
        elif change == "resize":
            # a new allocation at the same size
            self.w.resize_(2 * self.w.numel()).resize_(new.shape).copy_(new)
        elif change == "t_":
            self.w.t_()
        elif change == "as_strided_":
            self.w.as_strided_((64, 64), (1, 64))
        else:
            self.w.set_(torch.randn(64, 128, device="cuda").untyped_storage(), 0, (64, 64), (128, 1))
        # no statics_changed(): the check finds the storage or layout changed
        self.check(b(self.a, Batch(self.x, 3, self.MODE)), self.a, self.x, 3)
        self.assertEqual(f.static_hits, hits)
        self.check(b(self.a, Batch(self.x, 3, self.MODE)), self.a, self.x, 3)
        self.assertEqual(f.static_hits, hits + 1)

    @unittest.skipIf(not hasattr(torch._C, "_host_trace_trusted_statics_check"), "requires the native safety checks")
    def test_trusted_statics_unchecked_replay_the_old_layout(self):
        f = self.entry()
        b = self.trusted(f)
        for _ in range(3):
            self.check(b(self.a, Batch(self.x, 3, self.MODE)), self.a, self.x, 3)
        hits, old = f.static_hits, self.w.detach().clone()
        self.w.t_()
        out = b(self.a, Batch(self.x, 3, self.MODE))
        # unsupported without statics_changed(): a static hit at the old strides
        self.assertEqual(f.static_hits, hits + 1)
        self.assertEqual(out.y, bound_fn(old, self.a, self.x, 3)[0], atol=0, rtol=0)
        self.assertNotEqual(out.y, bound_fn(self.w, self.a, self.x, 3)[0])

    @unittest.skipIf(not hasattr(torch._C, "_host_trace_trusted_statics_check"), "requires the native safety checks")
    def test_trusted_statics_repointed_without_notice_replay_the_old_address(self):
        def scale(w, x, n):
            return w * 2 + x[:n].sum()

        w, x = torch.randn(4096, device="cuda"), torch.randn(64, device="cuda")
        f = HostTraceReplay(scale, static_prefix=1)
        for _ in range(2):
            f(w, x, 3)
        b = torch._C._HostTraceBound((w,), 2, 0, (self.root,), trust_statics=True)
        b.add(f, (), (), (), (), (), (), None, ())
        for _ in range(3):
            b(x, 3)
        hits, old, keep = f.static_hits, w.clone(), w.untyped_storage()
        w.set_(torch.randn_like(w))
        # unsupported without statics_changed(): the kernel's address is a static row
        out = b(x, 3)
        self.assertEqual(f.static_hits, hits + 1)
        self.assertEqual(out, scale(old, x, 3), atol=0, rtol=0)
        b.statics_changed()
        self.assertEqual(b(x, 3), scale(w, x, 3), atol=0, rtol=0)

    @unittest.skipIf(not hasattr(torch._C, "_host_trace_trusted_statics_check"), "requires the native safety checks")
    @parametrize("change", ["data", "swap", "t_"])
    def test_trusted_statics_changed_with_notice(self, change):
        f = self.entry()
        b = self.trusted(f)
        for _ in range(3):
            self.check(b(self.a, Batch(self.x, 3, self.MODE)), self.a, self.x, 3)
        hits = f.static_hits
        if change == "data":
            self.w.data = torch.randn_like(self.w)
        elif change == "swap":
            torch.utils.swap_tensors(self.w, torch.randn_like(self.w))
        else:
            self.w.t_()
        b.statics_changed()
        self.check(b(self.a, Batch(self.x, 3, self.MODE)), self.a, self.x, 3)
        self.assertEqual(f.static_hits, hits)
        self.check(b(self.a, Batch(self.x, 3, self.MODE)), self.a, self.x, 3)
        self.assertEqual(f.static_hits, hits + 1)

    @unittest.skipIf(not hasattr(torch._C._HostTraceBound, "statics_changed"), "requires the one-call options")
    def test_trusted_statics_another_call_recorded_are_checked(self):
        f = self.entry()
        b = self.trusted(f)
        for _ in range(3):
            self.check(b(self.a, Batch(self.x, 3, self.MODE)), self.a, self.x, 3)
        # the entry called directly on other leading arguments records them
        w2 = torch.randn_like(self.w)
        for _ in range(2):
            self.assertEqual(f(w2, self.a, self.x, 3), bound_fn(w2, self.a, self.x, 3), atol=0, rtol=0)
        hits = f.static_hits
        self.check(b(self.a, Batch(self.x, 3, self.MODE)), self.a, self.x, 3)
        self.assertEqual(f.static_hits, hits)


instantiate_parametrized_tests(TestHostTraceReplay)
instantiate_parametrized_tests(TestChainReplay)
instantiate_parametrized_tests(TestReplayMemory)
instantiate_parametrized_tests(TestNativeReplay)
instantiate_parametrized_tests(TestEntryParity)
instantiate_parametrized_tests(TestBuiltinMinMax)
instantiate_parametrized_tests(TestStaticPrefix)
instantiate_parametrized_tests(TestBoundCall)
instantiate_parametrized_tests(TestDirtyRecords)


def setUpModule():
    from torch.cuda import _host_trace_hint_audit
    import torch.cuda._host_trace_capture as capture

    _host_trace_hint_audit.enable_for_tests()
    capture.raise_trace_disagreements = True
    torch.cuda._host_trace.raise_unexpected = True


if __name__ == "__main__":
    run_tests()
