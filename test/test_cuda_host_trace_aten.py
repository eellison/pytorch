# Owner(s): ["module: cuda graphs"]

import unittest

import torch
import torch.nn.functional as F
from torch.cuda._host_trace_capture import (
    capture_kernel_nodes,
    KernelNode,
    MemsetNode,
)
from torch.cuda._host_trace_harvest import _zero_init
from torch.cuda._host_trace_launch import KernelLaunch
from torch.cuda._host_trace_replay import HostTraceReplay
from torch.cuda._host_trace_tape import _hint, EagerCall, Memset, trace, TrustedInputs
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    requires_cuda_python_bindings,
    run_tests,
    TEST_CUDA,
    TestCase,
)


def silu_mul(x, y):
    return F.silu(x) * y


def _other(dtype):
    return torch.float16 if dtype == torch.float32 else torch.float32


# name: (fn, operand shapes, functor bytes the kernel reads); an operand of
# shape None is a bool condition
CASES = {
    "silu_mul": (silu_mul, ("mh", "mh"), 0),
    "silu_mul_broadcast": (silu_mul, ("mh", "h"), 0),
    "add": (lambda x, y: torch.add(x, y, alpha=2), ("mh", "h"), 4),
    "gelu": (F.gelu, ("mh",), 0),
    "gelu_tanh": (lambda x: F.gelu(x, approximate="tanh"), ("mh",), 0),
    "rsqrt": (torch.rsqrt, ("mh",), 0),
    "where": (torch.where, (None, "mh", "h"), 0),
    "cast": (lambda x: x.to(_other(x.dtype)), ("mh",), 0),
    "cast_f64": (lambda x: x.double(), ("mh",), 0),
    "cast_t": (lambda x: x.t().to(_other(x.dtype), memory_format=torch.contiguous_format), ("mh",), 0),
    "cast_f64_t": (lambda x: x.t().to(torch.float64, memory_format=torch.contiguous_format), ("mh",), 0),
    "contiguous_t": (lambda x: x.t().contiguous(), ("mh",), 0),
}


# name: (fn over an (m, h) input, max traces over the sweep, ops functor bytes,
# whether the accumulator is the input dtype)
REDUCTIONS = {
    f"{op.__name__}_{form}": (lambda x, op=op, f=f: f(op, x), traces, ops, same)
    for op, ops, same in ((torch.sum, 0, False), (torch.mean, 4, False), (torch.amax, 0, True))
    for form, f, traces in (
        ("last", lambda op, x: op(x, -1), 2),
        ("first", lambda op, x: op(x, 0), 2),
        ("keepdim", lambda op, x: op(x, -1, keepdim=True), 2),
        ("all", lambda op, x: op(x), 3),
        ("t", lambda op, x: op(x.t(), -1), 2),
    )
}


# bytes of ReduceConfig, OffsetCalculator<1, uint32_t> and OffsetCalculator<2, uint32_t>
REDUCE_CONFIG_BYTES, OFFSET_CALC_1_BYTES, OFFSET_CALC_2_BYTES = 64, 404, 504
# OffsetCalculator<N>: dims, then sizes_ (MAX_DIMS IntDividers), then strides_
# (MAX_DIMS x N uint32)
OFFSET_CALC_STRIDES_AT, OFFSET_CALC_STRIDES_BYTES_PER_OPERAND = 304, 100


def _shared_root_copy(x):
    t = x * x
    return t[:32].copy_(t[32:])


# name: (fn, args from (x f16 [64, 4096], y f32 [4096]), the tape's step types)
DECLINES = {
    "integer_sum": (lambda x: x.sum(-1), lambda x, y: (x.long(),), [EagerCall]),
    "sum_with_dtype": (lambda x: x.sum(-1, dtype=torch.float32), lambda x, y: (x,), [EagerCall]),
    "double_mean": (lambda x: x.mean(-1), lambda x, y: (x.double(),), [EagerCall]),
    "empty_sum": (lambda x: x[:0].sum(-1), lambda x, y: (x,), [EagerCall]),
    "mixed_dtypes": (silu_mul, lambda x, y: (x, y), [KernelLaunch, EagerCall]),
    "memcpy_clone": (torch.clone, lambda x, y: (x,), [EagerCall]),
    "python_scalar": (lambda x: x + 2, lambda x, y: (x,), [EagerCall]),
    "integer_rsqrt": (torch.rsqrt, lambda x, y: (x.long(),), [EagerCall]),
    "shared_root_copy": (_shared_root_copy, lambda x, y: (x,), [KernelLaunch, EagerCall]),
}


def _reduce_unread(functor_bytes: int, arg_bytes: int) -> set[tuple[int, int]]:
    # (param, byte) of a ReduceOp that eager copies from stack garbage despite
    # the harvest memset: an empty ops functor's byte and the struct padding
    ident = -(-max(functor_bytes, 1) // arg_bytes) * arg_bytes
    config = -(-(ident + arg_bytes) // 4) * 4
    calcs_end = config + REDUCE_CONFIG_BYTES + OFFSET_CALC_1_BYTES + OFFSET_CALC_2_BYTES
    src = -(-calcs_end // 8) * 8
    skipped = [
        *range(functor_bytes, ident),
        *range(ident + arg_bytes, config),
        *range(config + 57, config + 60),
        *range(calcs_end, src),
        src + 58,
        src + 59,
    ]
    return {(0, b) for b in skipped}


def _inputs(case, m, h, dtype):
    shapes = {"mh": (m, h), "h": (h,), None: (m, h)}
    args = [torch.randn(shapes[s], device="cuda", dtype=dtype) for s in CASES[case][1]]
    return [a > 0 if s is None else a for a, s in zip(args, CASES[case][1])]


def _unread(node: KernelNode, launch: KernelLaunch, functor_bytes: int) -> set[tuple[int, int]]:
    # (param, byte) eager leaves uninitialized and the kernel never reads: a
    # functor's tail (an empty functor's byte), a cast loader's and storer's
    # padding, and a strided op's padding and OffsetCalculator entries past dims
    if "vectorized" in node.name or "unrolled" in node.name:
        tail = {(1, b) for b in range(functor_bytes, len(node.images[1]))}
        if "vectorized" in node.name:
            return tail
        return tail | {(p, b) for p in (5, 6) if len(node.images[p]) == 8 for b in range(1, 4)}
    n = len(launch.pointers)
    cast = "StridedCastOp" in node.name
    at = 8 * n + (-(-n // 4) * 4 if cast else 0)
    image = node.images[1]
    dims = int.from_bytes(image[at : at + 4], "little")
    sizes, strides = at + 4, at + OFFSET_CALC_STRIDES_AT
    end = strides + OFFSET_CALC_STRIDES_BYTES_PER_OPERAND * n
    skipped = [
        *range(9 * n if cast else at, at),
        *range(sizes + 12 * dims, strides),
        *range(strides + 4 * n * dims, end),
        *range(end + functor_bytes, len(image)),
    ]
    return {(1, b) for b in skipped}


@unittest.skipIf(not TEST_CUDA, "requires CUDA")
@requires_cuda_python_bindings
@unittest.skipIf(not hasattr(torch._C, "_cuda_hostTraceMul"), "needs traced hosts")
class TestHostTraceAten(TestCase):
    @parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    @parametrize("case", list(CASES))
    def test_replays_new_shapes(self, dtype, case):
        fn = CASES[case][0]
        entry = HostTraceReplay(fn)
        for h in (4096, 768):
            for m in (64, 200, 7, 1):
                args = _inputs(case, m, h, dtype)
                torch.cuda.synchronize()
                base = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                out = entry(*args)
                torch.cuda.synchronize()
                replay_peak = torch.cuda.max_memory_allocated() - base
                ref = fn(*args)
                self.assertEqual(out, ref, atol=0, rtol=0)
                self.assertEqual(out.stride(), ref.stride())
                del out, ref
                torch.cuda.reset_peak_memory_stats()
                fn(*args)
                torch.cuda.synchronize()
                self.assertEqual(replay_peak, torch.cuda.max_memory_allocated() - base)
        # at most one variant for m > 1 and one for m == 1
        self.assertLessEqual(entry.traces, 2)
        self.assertEqual(entry.eager, 0)
        for v in entry.variants:
            self.assertTrue(all(isinstance(s, range) for s in v.captured.lowered.steps))

    def _assert_launch_matches(self, launch, node, declared):
        self.assertEqual(launch.function, node.function)
        self.assertEqual(tuple(int(_hint(g)) for g in launch.grid), node.grid)
        self.assertEqual((launch.block, launch.smem), (node.block, node.smem))
        self.assertEqual(tuple(launch.layout), tuple(node.layout))
        fields = zip(launch.fields, launch.slots)
        for i, ((param, at, width), v) in enumerate(fields):
            declared |= {(param, b) for b in range(at, at + width)}
            if i not in launch.pointers:
                eager = node.images[param][at : at + width]
                eager = int.from_bytes(eager, "little", signed=True)
                self.assertEqual(int(_hint(v)), eager)
        for p, images in enumerate(zip(launch.images, node.images)):
            pairs = enumerate(zip(*images))
            diff = [b for b, (o, e) in pairs if o != e and (p, b) not in declared]
            self.assertEqual(diff, [], msg=f"{launch.name} param {p}")

    @parametrize("dtype", [torch.float16, torch.float32])
    @parametrize("case", list(CASES))
    @parametrize("m", [64, 7, 1])
    def test_records_match_eager(self, dtype, case, m):
        fn, _, functor_bytes = CASES[case]
        args = _inputs(case, m, 4096, dtype)
        tape = trace(fn, tuple(args))
        launches = [c for _, c in tape.launches]
        self.assertTrue(all(isinstance(c, KernelLaunch) for c in launches))
        nodes = capture_kernel_nodes(lambda s: fn(*args))
        self.assertEqual(len(launches), len(nodes))
        for launch, node in zip(launches, nodes):
            self._assert_launch_matches(launch, node, _unread(node, launch, functor_bytes))

    @parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    @parametrize("case", list(REDUCTIONS))
    def test_reduction_replays_new_shapes(self, dtype, case):
        fn, most = REDUCTIONS[case][:2]
        entry = HostTraceReplay(fn)
        for h in (4096, 768):
            for m in (64, 200, 7, 1, 150, 5):
                x = torch.randn(m, h, device="cuda", dtype=dtype)
                torch.cuda.synchronize()
                base = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                out = entry(x)
                torch.cuda.synchronize()
                replay_peak = torch.cuda.max_memory_allocated() - base
                ref = fn(x)
                self.assertEqual(out, ref, atol=0, rtol=0)
                self.assertEqual(out.stride(), ref.stride())
                del out, ref
                torch.cuda.reset_peak_memory_stats()
                fn(x)
                torch.cuda.synchronize()
                self.assertEqual(replay_peak, torch.cuda.max_memory_allocated() - base)
        self.assertLessEqual(entry.traces, most)
        self.assertEqual(entry.eager, 0)

    def _assert_replays_like_eager(self, entry, fn, x):
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        out = entry(x)
        torch.cuda.synchronize()
        replay_peak = torch.cuda.max_memory_allocated() - base
        ref = fn(x)
        self.assertEqual(out, ref, atol=0, rtol=0)
        del out, ref
        torch.cuda.reset_peak_memory_stats()
        fn(x)
        torch.cuda.synchronize()
        self.assertEqual(replay_peak, torch.cuda.max_memory_allocated() - base)

    @parametrize("dtype", [torch.float16, torch.float32])
    @parametrize("case", ["sum_last", "amax_last", "mean_first", "sum_t"])
    def test_reduction_one_trace_across_brackets(self, dtype, case):
        # the launch config's power-of-two brackets of both sizes are patched,
        # not guarded
        fn = REDUCTIONS[case][0]
        entry = HostTraceReplay(fn)
        for m, h in ((64, 768), (2, 768), (37, 96), (129, 4096), (600, 200), (9, 1500), (300, 32)):
            self._assert_replays_like_eager(entry, fn, torch.randn(m, h, device="cuda", dtype=dtype))
        self.assertEqual(entry.traces, 1)
        self.assertEqual(entry.eager, 0)

    def test_global_reduction_replays_new_shapes(self):
        # a reduction across CTAs: its staging buffer, semaphores and CTAs per
        # output are sizes of the call
        fn = REDUCTIONS["sum_last"][0]
        entry = HostTraceReplay(fn)
        for m, h in ((2, 1 << 20), (3, 3 << 19), (5, 1 << 21)):
            self._assert_replays_like_eager(entry, fn, torch.randn(m, h, device="cuda"))
        self.assertEqual(entry.traces, 1)
        self.assertEqual(entry.eager, 0)

    @parametrize("dtype", [torch.float16, torch.float32])
    @parametrize("case", list(REDUCTIONS))
    @parametrize("m", [4096, 64, 7, 1])
    def test_reduction_records_match_eager(self, dtype, case, m):
        fn, _, functor_bytes, same = REDUCTIONS[case]
        unread = _reduce_unread(functor_bytes, dtype.itemsize if same else 4)
        x = torch.randn(m, 768, device="cuda", dtype=dtype)
        launches = [c for _, c in trace(fn, (x,)).launches]
        with _zero_init():
            nodes = capture_kernel_nodes(lambda s: fn(x))
        kinds = [Memset if isinstance(n, MemsetNode) else KernelLaunch for n in nodes]
        self.assertEqual([type(c) for c in launches], kinds)
        for launch, node in zip(launches, nodes):
            if isinstance(node, MemsetNode):
                self.assertEqual(launch.value, node.value)
                width = int(_hint(launch.width)) * launch.element_size
                self.assertEqual(width, node.width * node.element_size)
            else:
                self._assert_launch_matches(launch, node, set(unread))

    @parametrize("case", list(DECLINES))
    def test_decline_is_an_eager_call(self, case):
        x = torch.randn(64, 4096, device="cuda", dtype=torch.float16)
        y = torch.randn(4096, device="cuda", dtype=torch.float32)
        fn, args_fn, kinds = DECLINES[case]
        args = args_fn(x, y)
        tape = trace(fn, args)
        self.assertEqual([type(c) for _, c in tape.launches], kinds)
        entry = HostTraceReplay(fn)
        entry(*args)
        self.assertEqual(entry(*args), fn(*args), atol=0, rtol=0)

    def test_copy_between_arguments_checks_overlap_at_replay(self):
        # arguments disjoint at the trace share storage at the third call: the
        # variant's argument pair overlaps, so the call runs eagerly
        def pair(shared):
            buf = torch.arange(64 * 64 + 4, device="cuda", dtype=torch.float32)
            dst = buf[:4096] if shared else torch.zeros(4096, device="cuda")
            return dst.view(64, 64).t(), buf[4:].view(64, 64)

        def fn(dst, src):
            return dst.copy_(src)

        self.assertTrue(all(isinstance(c, KernelLaunch) for _, c in trace(fn, pair(False)).launches))
        entry = HostTraceReplay(fn)
        for _ in range(2):
            self.assertEqual(entry(*pair(False)), fn(*pair(False)), atol=0, rtol=0)
        self.assertEqual(entry.eager, 0)
        overlap = "refer to a single memory location"
        with self.assertRaisesRegex(RuntimeError, overlap):
            fn(*pair(True))
        with self.assertRaisesRegex(RuntimeError, overlap):
            entry(*pair(True))
        self.assertEqual(entry(*pair(False)), fn(*pair(False)), atol=0, rtol=0)
        self.assertEqual((entry.traces, entry.eager, len(entry.variants)), (1, 1, 1))

    def test_a_trusted_trace_records_no_argument_pairs(self):
        # the caller vouches for trusted inputs' aliasing: no replay overlap check
        def fn(dst, src):
            return dst.copy_(src)

        args = (torch.zeros(64, 64, device="cuda").t(), torch.randn(64, 64, device="cuda"))
        self.assertEqual(trace(fn, args).argument_pairs, ((0, 1),))
        trusted = TrustedInputs(layouts=tuple((tuple(a.shape), a.stride()) for a in args))
        self.assertEqual(trace(fn, args, trusted=trusted).argument_pairs, ())

    def test_inplace_between_arguments_partially_overlapping_at_replay(self):
        def fn(a, b):
            return a.add_(b) * b

        def pair(shift):
            buf = torch.randn(8192, device="cuda")
            return buf[:4096], (buf[shift : shift + 4096] if shift else torch.randn(4096, device="cuda"))

        entry = HostTraceReplay(fn)
        for _ in range(2):
            a, b = pair(0)
            self.assertEqual(entry(a.clone(), b), fn(a.clone(), b), atol=0, rtol=0)
        self.assertEqual(entry.eager, 0)
        overlap = "refer to a single memory location"
        with self.assertRaisesRegex(RuntimeError, overlap):
            fn(*pair(1024))
        with self.assertRaisesRegex(RuntimeError, overlap):
            entry(*pair(1024))
        # disjoint views of one storage hold the disjoint variant
        buf = torch.randn(8192, device="cuda")
        ref = buf.clone()
        self.assertEqual(entry(buf[:4096], buf[4096:]), fn(ref[:4096], ref[4096:]), atol=0, rtol=0)

    @parametrize("traced_alias", [True, False])
    def test_self_alias_at_trace_or_replay(self, traced_alias):
        # a variant traced at an overlap runs that step eagerly at every call;
        # one traced disjoint runs a call whose arguments overlap eagerly
        def fn(a, b):
            return a.copy_(b) * b

        def call(f, alias):
            a = torch.arange(8192, device="cuda", dtype=torch.float32)[::2]
            return f(a, a) if alias else f(a, torch.ones(8192, device="cuda")[::2])

        entry = HostTraceReplay(fn)
        for alias in (traced_alias, traced_alias, not traced_alias, not traced_alias, traced_alias):
            self.assertEqual(call(entry, alias), call(fn, alias), atol=0, rtol=0)
        steps = [[c.reason for _, c in v.captured.lowered.tape.launches if type(c) is EagerCall] for v in entry.variants]
        aliased = ["aten.copy_.default writes a storage another operand is of"]
        self.assertEqual((entry.traces, entry.eager, steps), (1, 0, [aliased]) if traced_alias else (1, 2, [[]]))

    def test_self_alias_decline_is_its_own_class(self):
        # a tape of only an eager step declines its call's class, which
        # includes which arguments overlap
        def fn(a, b):
            return a.copy_(b)

        def call(f, alias):
            a = torch.arange(8192, device="cuda", dtype=torch.float32)[::2]
            return f(a, a) if alias else f(a, torch.ones(8192, device="cuda")[::2])

        entry = HostTraceReplay(fn)
        for alias, eager in ((True, 1), (False, 1), (True, 2), (False, 2)):
            self.assertEqual(call(entry, alias), call(fn, alias), atol=0, rtol=0)
            self.assertEqual(entry.eager, eager)
        self.assertEqual((entry.traces, len(entry.variants)), (2, 1))

    def test_size_one_dim_strides_follow_eager(self):
        # x [1, h] has strides (1, 1): eager's output keeps stride 1 on the
        # size-1 dim, where the fake's is h
        def fn(x, r):
            return x.float() * r

        entry = HostTraceReplay(fn)
        for h in (64, 4096, 768):
            x = torch.randn(h, 1, device="cuda", dtype=torch.bfloat16).t()
            r = torch.ones(1, 1, device="cuda")
            out, ref = entry(x, r), fn(x, r)
            self.assertEqual(out, ref, atol=0, rtol=0)
            self.assertEqual(out.stride(), ref.stride())
        self.assertEqual(entry.eager, 0)
        self.assertTrue(all(isinstance(c, KernelLaunch) for _, c in trace(fn, (x, r)).launches))


instantiate_parametrized_tests(TestHostTraceAten)

if __name__ == "__main__":
    run_tests()
