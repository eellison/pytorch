# Owner(s): ["module: cuda"]

import itertools
import math
import re
import unittest

from host_trace_testing import (
    assert_eager_function_handles,
    bits,
    build,
    EntryMode,
    HostTraceTestCase,
)

import torch
import torch.nn.functional as F
from torch.testing._internal.common_utils import (
    run_tests,
    skipIfRocm,
    TEST_CUDA_PYTHON_BINDINGS,
)


if torch.cuda.is_available():
    import host_trace_two_hint as two_hint

    from torch.cuda import _host_trace as ht, _host_trace_ti

C = torch._C


def _assert_fused_rms_norm_route(case, fn, args):
    # F.rms_norm's _fused_rms_norm is a converted host, never decomposed
    # (DECISIONS O40 / A190; the core suite's test_unconverted_hosts_decline_by_name
    # has the same shape): traced as ATen's launches where eager runs ATen; where
    # eager's route is torch._native's override (the vendored QuACK CuTe DSL rms
    # norm: torch/cuda/_host_trace_native.py) it is one launch of the override's
    # own program recorded from its launch descriptor when the program has one
    # (A410: built at its compile under the runtime SDK, or read beside QuACK's
    # cached object; a fact of the cache, not of the call), else one closed region
    from torch._native import registry

    override = any(
        n.active for n in registry._graphs.get(("_fused_rms_norm", "CUDA"), ())
    )
    tape = ht.trace(fn, args)
    kernels = [L["kernel"] for L in tape.launches]
    if not override:
        case.assertEqual(tape.num_regions, 0)
        case.assertGreater(tape.num_launches, 0)
        return
    if tape.num_regions:
        case.assertEqual((tape.num_launches, tape.num_regions), (0, 1), kernels)
        case.assertEqual(tape.regions[0].op, "_fused_rms_norm")
    else:
        case.assertEqual(len(kernels), 1, kernels)
        case.assertIn("quack", kernels[0].lower(), kernels)
        case.assertIn("rmsnorm", kernels[0].lower(), kernels)


# a 0-dim CPU tensor operand (not a wrapped number): an implicit CPU scalar
# input the entry declines by name; made outside the traced function, since
# torch.tensor inside one is a lift_fresh the trace declines
_CPU_TWO = torch.tensor(2.0)
_CPU_ONES = torch.ones(1)

# the real op and the sibling entry, both as functions of the same arguments
OPS = {
    "add": (
        lambda x, y, alpha=1.0: torch.add(x, y, alpha=alpha),
        lambda x, y, alpha=1.0: C._host_trace_ti_add(x, y, alpha),
    ),
    "mul": (torch.mul, C._host_trace_ti_mul) if torch.cuda.is_available() else None,
    "div": (torch.div, C._host_trace_ti_div) if torch.cuda.is_available() else None,
    "silu": (F.silu, C._host_trace_ti_silu) if torch.cuda.is_available() else None,
    "gelu": (lambda x: F.gelu(x), lambda x: C._host_trace_ti_gelu(x, "none"))
    if torch.cuda.is_available()
    else None,
    "gelu_tanh": (
        lambda x: F.gelu(x, approximate="tanh"),
        lambda x: C._host_trace_ti_gelu(x, "tanh"),
    )
    if torch.cuda.is_available()
    else None,
    "tanh": (torch.tanh, C._host_trace_ti_tanh) if torch.cuda.is_available() else None,
    "sqrt": (torch.sqrt, C._host_trace_ti_sqrt) if torch.cuda.is_available() else None,
    "reciprocal": (torch.reciprocal, C._host_trace_ti_reciprocal)
    if torch.cuda.is_available()
    else None,
}

# the kernel template and its leading integer arguments: the functor differs
# between a lambda in the real op and the sibling's named functor, everything
# before it (vector width, unroll) must not
_FAMILY = re.compile(
    r"^(_ZN2at6native\d+(?:vectorized_elementwise_kernel|unrolled_elementwise_kernel|elementwise_kernel)I(?:Li\d+E)*)"
)


def _family(name):
    m = _FAMILY.match(name)
    return m.group(1) if m else name


def _eager_work(fn, args):
    # the device work one eager call issues, in order: kernels by name,
    # memsets and memcpys as such
    from torch.profiler import profile, ProfilerActivity

    fn(*args)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        fn(*args)
        torch.cuda.synchronize()
    work = []
    for e in p.events():
        if e.device_type != torch.autograd.DeviceType.CUDA:
            continue
        kind = "kernel"
        for other in ("Memset", "Memcpy"):
            if e.name.startswith(other):
                kind = other.lower()
        work.append((kind, e.name if kind == "kernel" else None))
    return work


def _tape_work(tape):
    # the tape's launches, memsets and memcpys in issue order (one seq counter)
    items = [("kernel", L["seq"], C._demangle(L["kernel"])) for L in tape.launches]
    # the memset and memcpy records of later commits' tapes
    items += [("memset", m["seq"], None) for m in getattr(tape, "memsets", [])]
    items += [("memcpy", m["seq"], None) for m in getattr(tape, "memcpys", [])]
    return [(kind, name) for kind, _, name in sorted(items, key=lambda i: i[1])]


# the rows of the fidelity test whose launch is a twin of eager's by construction (none since
# E36 stage 3: every converted host launches eager's own kernel); a row that must be one
# again is listed here with its reason, literal equality is asserted to FAIL for it, and it
# leaves the table the moment its host launches eager's function
_TWINS_PENDING: dict[str, str] = {}


def _assert_or_pending(case, pending, check):
    if not pending:
        check()
        return
    with case.assertRaises(
        AssertionError, msg="launches eager's kernel now: assert equality for it"
    ):
        check()


def _gen(name):
    # a generated binding as a function of the tensors and values, the
    # allocating form
    binding = getattr(C, f"_host_trace_ti_gen_{name}")
    return lambda *args: binding(*args, None)


_FLOATS = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
_INTS = (torch.int32, torch.int64, torch.uint8)
aten = torch.ops.aten

# the generated siblings under test: the real op and the generated entry as
# functions of the tensor operands (the values bound; a string names the
# binding), the tensor arity, the dtypes eager's kernel serves, how the
# operands are prepared ("positive": the domain of log / rsqrt; "mask": a bool
# second operand), and the size of the functor eager's lambda captures per
# dtype (the launch image's data pointers follow it)
GENERATED = {
    "sigmoid": (torch.sigmoid, "sigmoid", 1, _FLOATS, None, lambda d: 1),
    "silu_backward": (
        aten.silu_backward,
        "silu_backward",
        2,
        _FLOATS,
        None,
        lambda d: 1,
    ),
    "remainder": (
        torch.remainder,
        "remainder_Tensor",
        2,
        _FLOATS + _INTS,
        "nonzero",
        lambda d: 1,
    ),
    "abs": (torch.abs, "abs", 1, _FLOATS + _INTS + (torch.bool,), None, lambda d: 1),
    "log": (torch.log, "log", 1, _FLOATS, "positive", lambda d: 1),
    "leaky_relu(0.2)": (
        lambda x: F.leaky_relu(x, 0.2),
        lambda x: _gen("leaky_relu")(x, 0.2),
        1,
        _FLOATS,
        None,
        lambda d: 8 if d is torch.float64 else 4,
    ),
    "maximum": (
        torch.maximum,
        "maximum",
        2,
        _FLOATS + _INTS + (torch.bool,),
        None,
        lambda d: 1,
    ),
    "minimum": (
        torch.minimum,
        "minimum",
        2,
        _FLOATS + _INTS + (torch.bool,),
        None,
        lambda d: 1,
    ),
    "tanh_backward": (
        aten.tanh_backward,
        "tanh_backward",
        2,
        _FLOATS,
        None,
        lambda d: 1,
    ),
    "sigmoid_backward": (
        aten.sigmoid_backward,
        "sigmoid_backward",
        2,
        _FLOATS,
        None,
        lambda d: 1,
    ),
    "mse_loss(none)": (
        lambda a, b: F.mse_loss(a, b, reduction="none"),
        "mse_loss",
        2,
        _FLOATS,
        None,
        lambda d: 1,
    ),
    "gelu_backward": (
        lambda g, x: aten.gelu_backward(g, x),
        lambda g, x: _gen("gelu_backward")(g, x, "none"),
        2,
        _FLOATS,
        None,
        lambda d: 1,
    ),
    "gelu_backward(tanh)": (
        lambda g, x: aten.gelu_backward(g, x, approximate="tanh"),
        lambda g, x: _gen("gelu_backward")(g, x, "tanh"),
        2,
        _FLOATS,
        None,
        lambda d: 1,
    ),
    "threshold_backward(0.5)": (
        lambda g, x: aten.threshold_backward(g, x, 0.5),
        lambda g, x: _gen("threshold_backward")(g, x, 0.5),
        2,
        _FLOATS,
        None,
        lambda d: 16 if d is torch.float64 else 8,
    ),
    "threshold_backward(1) int": (
        lambda g, x: aten.threshold_backward(g, x, 1),
        lambda g, x: _gen("threshold_backward")(g, x, 1),
        2,
        _INTS,
        None,
        lambda d: 2 * d.itemsize,
    ),
    "hardtanh_backward(-0.5, 0.5)": (
        lambda g, x: aten.hardtanh_backward(g, x, -0.5, 0.5),
        lambda g, x: _gen("hardtanh_backward")(g, x, -0.5, 0.5),
        2,
        _FLOATS,
        None,
        lambda d: 16 if d is torch.float64 else 8,
    ),
    "lerp(0.3)": (
        lambda a, b: torch.lerp(a, b, 0.3),
        lambda a, b: _gen("lerp_Scalar")(a, b, 0.3),
        2,
        _FLOATS,
        None,
        lambda d: 8 if d is torch.float64 else 4,
    ),
    "lerp(0.7)": (
        lambda a, b: torch.lerp(a, b, 0.7),
        lambda a, b: _gen("lerp_Scalar")(a, b, 0.7),
        2,
        _FLOATS,
        None,
        lambda d: 8 if d is torch.float64 else 4,
    ),
    "native_dropout_backward(1.25)": (
        lambda g, m: aten.native_dropout_backward(g, m, 1.25),
        lambda g, m: _gen("native_dropout_backward")(g, m, 1.25),
        2,
        _FLOATS,
        "mask",
        lambda d: 8 if d is torch.float64 else 4,
    ),
    "lerp.Tensor": (torch.lerp, "lerp_Tensor", 3, _FLOATS, None, lambda d: 1),
    "addcmul(value=2)": (
        lambda a, b, c: torch.addcmul(a, b, c, value=2),
        lambda a, b, c: _gen("addcmul")(a, b, c, 2),
        3,
        _FLOATS + _INTS,
        None,
        lambda d: 8
        if d in (torch.float64, torch.int32, torch.int64, torch.uint8)
        else 4,
    ),
}


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
@skipIfRocm(msg="host tracing is CUDA-only in this version")
class TestCudaHostTraceTI(HostTraceTestCase):
    def _assert_parity(self, real, entry, args, ntensors, functor_bytes, f_size=1):
        want = real(*args)
        got = entry(*args)
        torch.cuda.synchronize()
        self._assert_bitwise(got, want, stride=True)
        real_nodes, _ = self._capture(lambda: real(*args))
        ours, _ = self._capture(lambda: entry(*args))
        self.assertEqual(len(ours), len(real_nodes))
        for (name_r, grid_r, block_r, smem_r, image_r), (
            name_o,
            grid_o,
            block_o,
            smem_o,
            image_o,
        ) in zip(real_nodes, ours):
            self.assertEqual(_family(name_o), _family(name_r))
            self.assertEqual((grid_o, block_o, smem_o), (grid_r, block_r, smem_r))
            if "elementwise_kernelILi128E" in name_r:
                # the strided path: the real op's closure and the sibling's named
                # functor lay out (data, offset_calc, f) with different padding
                continue
            # vectorized / unrolled: N at 0, the functor after it, then the
            # pointers (the output's is each capture's own allocation, the
            # inputs' agree)
            self.assertEqual(len(image_o), len(image_r))
            self.assertEqual(image_o[:4], image_r[:4])
            f_off = 4 if f_size <= 4 else 8
            data = (f_off + f_size + 7) // 8 * 8
            self.assertEqual(
                image_o[data + 8 : data + 8 * ntensors],
                image_r[data + 8 : data + 8 * ntensors],
            )
            if functor_bytes:
                self.assertEqual(
                    image_o[f_off : f_off + f_size], image_r[f_off : f_off + f_size]
                )

    def _matrix(self, dtype, binary):
        d = "cuda"
        M, N = 48, 3000
        x = torch.randn(M, N, device=d).to(dtype)
        y = torch.randn(M, N, device=d).to(dtype)
        cases = {
            "contiguous": (x, y),
            "transposed": (x.t().contiguous().t(), y),
            "broadcast": (x, y[0]),
            "expanded": (x, y[:1].expand(M, N)),
            "slice": (x[:, ::2], y[:, ::2]),
            "scalar": (x, y[0, 0]),
            "misaligned": (x.flatten()[1:], y.flatten()[1:]),
            "batch1": (x[:1], y[:1]),
        }
        if not binary:
            cases = {k: (v[0],) for k, v in cases.items()}
        return cases

    def test_parity_with_the_real_op(self):
        for name, pair in OPS.items():
            real, entry = pair
            binary = name in ("add", "mul", "div")
            dtypes = [torch.float16, torch.bfloat16, torch.float32, torch.float64]
            if name == "mul":
                dtypes += [torch.int32, torch.int64, torch.bool]
            for dtype in dtypes:
                for case, args in self._matrix(dtype, binary).items():
                    with self.subTest(op=name, dtype=dtype, case=case):
                        f_size = (
                            (8 if dtype is torch.float64 else 4) if name == "add" else 1
                        )
                        self._assert_parity(
                            real,
                            entry,
                            args,
                            3 if binary else 2,
                            functor_bytes=(name == "add"),
                            f_size=f_size,
                        )
        # add's alpha travels in the functor
        x = torch.randn(64, 4096, device="cuda", dtype=torch.bfloat16)
        y = torch.randn_like(x)
        real, entry = OPS["add"]
        self._assert_parity(real, entry, (x, y, 2.5), 3, functor_bytes=True, f_size=4)

    def test_copy_parity_with_the_real_op(self):
        for dtype in (torch.bfloat16, torch.float32):
            src = torch.randn(64, 4096, device="cuda").to(dtype).t()
            want = torch.empty(4096, 64, device="cuda", dtype=dtype)
            got = torch.empty_like(want)
            want.copy_(src)
            C._host_trace_ti_copy_(got, src)
            self.assertTrue(torch.equal(bits(got), bits(want)))
            real_nodes, _ = self._capture(lambda: want.copy_(src))
            ours, _ = self._capture(lambda: C._host_trace_ti_copy_(got, src))
            self.assertEqual(len(ours), 1)
            self.assertEqual(_family(ours[0][0]), _family(real_nodes[0][0]))
            self.assertEqual(ours[0][1:4], real_nodes[0][1:4])
        # a contiguous copy is a memcpy in the real op, and in the sibling
        # (copy_d2d): one cudaMemcpyAsync in ordinary mode, a memcpy record and
        # no launch under a trace
        a = torch.randn(64, device="cuda")
        b = C._host_trace_ti_copy_(torch.empty_like(a), a)
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(b, a))
        tape = ht.trace(lambda t: torch.empty_like(t).copy_(t), (a,))
        self.assertEqual((tape.num_launches, len(tape.memcpys)), (0, 1))

    # ---- traced

    def _roundtrip(self, fn, base_args, new_args_list, atol=0, rtol=0):
        tape, variant, cases = self._replay_cases(
            fn, base_args, new_args_list, atol=atol, rtol=rtol
        )
        return tape, variant, sum(c.out is not None for c in cases)

    def _pair(self, M, N, dtype=torch.bfloat16, offset=0):
        flat = torch.randn(2 * (M * N + offset), device="cuda").to(dtype)
        x = flat[offset : offset + M * N].view(M, N)
        y = flat[M * N + 2 * offset : 2 * M * N + 2 * offset].view(M, N)
        return x, y

    def test_add_replays_at_new_shapes(self):
        fn = torch.add
        base = self._pair(64, 4096)
        news = [
            self._pair(48, 4096),
            self._pair(64, 2048),
            self._pair(7, 1000),
            self._pair(1024, 1024),
            self._pair(3, 8),
        ]
        tape, _, served = self._roundtrip(fn, base, news)
        self.assertEqual(tape.num_launches, 1)
        self.assertEqual(tape.num_allocations, 1)
        self.assertEqual(served, len(news))

    def test_add_strided_and_broadcast_replay_without_pinning_sizes(self):
        # the strided path: IntDivider's magic and shift are opaque rebinds of
        # the divisor, so a size sweep crossing powers of two keeps serving
        def bcast(x, b):
            return torch.add(x, b)

        x, _ = self._pair(64, 4096)
        b = torch.randn(4096, device="cuda", dtype=torch.bfloat16)
        news = []
        for M, N in [
            (48, 3000),
            (33, 2049),
            (64, 4096),
            (32, 4096),
            (64, 2048),
            (7, 1000),
            (1024, 512),
            (129, 65),
            (2, 3),
            (500, 4097),
        ]:
            news.append(
                (
                    self._pair(M, N)[0],
                    torch.randn(N, device="cuda", dtype=torch.bfloat16),
                )
            )
        tape, _, served = self._roundtrip(bcast, (x, b), news)
        self.assertEqual(served, len(news))
        self.assertTrue(any(o["fn"] == "intdivider_m1" for o in tape.opaque))

        def transposed(x, y):
            return torch.add(x, y)

        xt = torch.randn(4096, 64, device="cuda", dtype=torch.bfloat16).t()
        yt = self._pair(64, 4096)[1]
        news = [
            (
                torch.randn(K, M, device="cuda", dtype=torch.bfloat16).t(),
                self._pair(M, K)[1],
            )
            for (M, K) in [(48, 3000), (33, 2049), (128, 512), (5, 7)]
        ]
        _, _, served = self._roundtrip(transposed, (xt, yt), news)
        self.assertEqual(served, len(news))

    def test_size_one_dims_miss_or_serve_correctly(self):
        # a trace base with sizes >= 2; a size-1 replay is a miss or the right
        # answer, never a wrong shape
        x, y = self._pair(8, 4096)
        b = torch.randn(4096, device="cuda", dtype=torch.bfloat16)
        for fn, base, new in [
            (torch.add, (x, y), self._pair(1, 4096)),
            (torch.add, (x, b), (self._pair(1, 4096)[0], b)),
            (F.silu, (x,), (self._pair(1, 4096)[0],)),
        ]:
            self._roundtrip(fn, base, [new])

    def test_dtype_broadcast_and_alpha_changes_miss(self):
        x, y = self._pair(16, 4096)
        tape = ht.trace(torch.add, (x, y))
        variant = build(tape, torch.add, (x, y))
        half = self._pair(16, 4096, dtype=torch.float16)
        self.assertIsNone(variant.try_replay(half))
        # a broadcast shape after a same-shape trace: the strides guard misses
        self.assertIsNone(variant.try_replay((x, y[0])))
        # alpha is a constant of the variant
        fn = lambda a, b, alpha: torch.add(a, b, alpha=alpha)  # noqa: E731
        tape = ht.trace(fn, (x, y, 2.0))
        variant = build(tape, fn, (x, y, 2.0))
        self.assertIsNotNone(variant.try_replay((x, y, 2.0)))
        self.assertIsNone(variant.try_replay((x, y, 3.0)))

    def test_vectorization_width_is_guarded(self):
        # traced with 16-byte aligned operands (vector width 8 for bf16); an
        # operand aligned to 8 bytes only replays through a miss, one aligned
        # to 32 bytes serves
        base = self._pair(64, 4096, offset=8)
        tape = ht.trace(torch.add, base)
        variant = build(tape, torch.add, base)
        self.assertIsNotNone(variant.try_replay(self._pair(64, 4096, offset=16)))
        self.assertIsNone(variant.try_replay(self._pair(64, 4096, offset=4)))
        # and the other way round: traced at width 4, an aligned pair misses
        base4 = self._pair(64, 4096, offset=4)
        tape = ht.trace(torch.add, base4)
        variant = build(tape, torch.add, base4)
        self.assertIsNotNone(variant.try_replay(self._pair(64, 4096, offset=12)))
        self.assertIsNone(variant.try_replay(self._pair(64, 4096, offset=8)))

    def test_unary_ops_replay(self):
        for fn in (
            F.silu,
            F.gelu,
            lambda t: F.gelu(t, approximate="tanh"),
            lambda t: torch.mul(t, t),
            torch.tanh,
            lambda t: torch.sqrt(t * t),
        ):
            x, _ = self._pair(64, 4096)
            news = [
                (self._pair(48, 3000)[0],),
                (self._pair(5, 7)[0],),
                (self._pair(64, 4096, dtype=torch.bfloat16)[0].t().contiguous().t(),),
            ]
            _, _, served = self._roundtrip(fn, (x,), news)
            self.assertGreaterEqual(served, 2)

    def test_two_ops_trace_as_one_function(self):
        def fn(x, y):
            return F.silu(torch.add(x, y))

        base = self._pair(32, 4096)
        news = [self._pair(48, 4096), self._pair(9, 1000)]
        tape, _, served = self._roundtrip(fn, base, news)
        self.assertEqual(tape.num_launches, 2)
        self.assertEqual(tape.num_allocations, 2)
        self.assertEqual(served, 2)

    def test_copy_through_contiguous_replays(self):
        def fn(x):
            return x.t().contiguous()

        x = torch.randn(64, 4096, device="cuda", dtype=torch.bfloat16)
        news = [
            (torch.randn(48, 3000, device="cuda", dtype=torch.bfloat16),),
            (torch.randn(5, 4097, device="cuda", dtype=torch.bfloat16),),
        ]
        tape, _, served = self._roundtrip(fn, (x,), news)
        self.assertEqual(tape.num_launches, 1)
        self.assertEqual(served, 2)

    def test_declines_by_name(self):
        x, y = self._pair(8, 4096)
        with self.assertRaisesRegex(ht.Declined, "aten.atan2.default"):
            ht.trace(torch.atan2, (x, y))
        # a CPU tensor that is not a scalar: eager refuses it (the warm-up
        # would raise its text); the entry declines it by name
        with self.assertRaisesRegex(ht.Declined, "aten.add.Tensor with a cpu tensor"):
            ht.trace(lambda t: t + _CPU_ONES, (x,), warm_up=False)
        # a bf16 and a float32 operand promote through the cast kernel
        self.assertEqual(ht.trace(torch.add, (x, y.float())).num_launches, 1)
        # a CPU scalar that would promote the CUDA operand (an int tensor
        # times a float) is type promotion, declined like a dtype pair
        xi = torch.arange(64, device="cuda").view(8, 8)
        with self.assertRaisesRegex(ht.Declined, "promotes its CUDA operand"):
            ht.trace(lambda t: t * 0.5, (xi,))
        with self.assertRaisesRegex(ht.Declined, "div on Long"):
            ht.trace(lambda t: t / 2, (xi,))
        # a contiguous clone is one memcpy record (copy_d2d), no launch
        tape = ht.trace(lambda t: t.clone(), (x,))
        self.assertEqual((tape.num_launches, len(tape.memcpys)), (0, 1))
        self.assertFalse(C._host_trace_tracing())
        tape = ht.trace(torch.add, (x, y))
        self.assertEqual(tape.num_launches, 1)

    # ---- a CPU scalar operand (a Python number the dispatcher wrapped)

    # the four ops with a scalar on either side, as Python spells them and as
    # torch.* spells the scalar-first forms; (name, real, kwargs for alpha)
    _SCALAR_FORMS = {
        "x * 0.5": lambda x, s: x * s,
        "0.5 * x": lambda x, s: s * x,
        "x + 1.5": lambda x, s: x + s,
        "1.5 + x": lambda x, s: s + x,
        "x - 2.0": lambda x, s: x - s,
        "2.0 - x": lambda x, s: s - x,
        "x / 2.0": lambda x, s: x / s,
        "2.0 / x": lambda x, s: s / x,
        "torch.add(2.0, x)": lambda x, s: torch.add(s, x),
        "torch.sub(2.0, x)": lambda x, s: torch.sub(s, x),
        "torch.div(2.0, x)": lambda x, s: torch.div(s, x),
        "torch.add(x, 2.0, alpha=0.5)": lambda x, s: torch.add(x, s, alpha=0.5),
        "torch.sub(x, 2.0, alpha=2)": lambda x, s: torch.sub(x, s, alpha=2),
        "x * 2 (int)": lambda x, s: x * int(s),
    }

    def _scalar_bytes(self, name, dtype):
        # where the scalar sits in the launch image of the vectorized /
        # unrolled kernel: N (4 bytes), then the functor at its alignment. The
        # add functors hold (scalar, alpha) in opmath; AUnaryFunctor /
        # BUnaryFunctor hold the (empty) inner functor first and the scalar
        # at the opmath alignment; div's scalar-first form stores it in
        # scalar_t. Padding is compared only where the sibling zeroes it and
        # eager's stack bytes would differ, so it is skipped.
        op = 4 if dtype is not torch.float64 else 8
        if "+" in name or "add" in name or "-" in name or "sub" in name:
            return op, 2 * op  # (offset, size): scalar then alpha
        if name == "2.0 / x":
            return None  # reciprocal then mul: the scalar rides in the mul's functor
        if name == "torch.div(2.0, x)":
            # AUnaryFunctor over DivFunctor<scalar_t>: the functor at the opmath
            # alignment, its empty inner functor first, the scalar in scalar_t
            # after it (6 / 8 / 16 bytes in for bf16 / fp32 / fp64)
            return op + dtype.itemsize, dtype.itemsize
        return 2 * op, op

    def test_scalar_operand_parity_with_the_real_op(self):
        # bitwise the real op's output, the same kernel instantiation, grid
        # and block, and the same scalar bytes in the image, for both orders
        # of the four ops on bf16 / fp32 (fp16 / fp64 for a few)
        for dtype in (torch.bfloat16, torch.float32, torch.float16, torch.float64):
            x = torch.randn(48, 3000, device="cuda").to(dtype) + 1.5
            for name, fn in self._SCALAR_FORMS.items():
                if dtype in (torch.float16, torch.float64) and name not in (
                    "x * 0.5",
                    "x + 1.5",
                    "x / 2.0",
                    "torch.div(2.0, x)",
                ):
                    continue
                scalar = 0.5 if "0.5" in name else (1.5 if "1.5" in name else 2.0)
                with self.subTest(op=name, dtype=dtype):
                    want = fn(x, scalar)
                    real_nodes, _ = self._capture(lambda: fn(x, scalar))
                    tape = ht.trace(fn, (x, scalar))
                    variant = build(tape, fn, (x, scalar))
                    (got,) = variant.replay((x, scalar))
                    torch.cuda.synchronize()
                    self.assertEqual(got.dtype, want.dtype)
                    self.assertEqual(got.stride(), want.stride())
                    self.assertTrue(
                        torch.equal(bits(got), bits(want)), f"{name} differs bitwise"
                    )
                    ours, _ = self._capture(
                        lambda: ht._TRACED_ENTRIES[self._op_of(name)](
                            *self._entry_args(name, x, scalar)
                        )
                    )
                    self.assertEqual(len(ours), len(real_nodes))
                    for (name_r, grid_r, block_r, smem_r, image_r), (
                        name_o,
                        grid_o,
                        block_o,
                        smem_o,
                        image_o,
                    ) in zip(real_nodes, ours):
                        self.assertEqual(_family(name_o), _family(name_r))
                        self.assertEqual(
                            (grid_o, block_o, smem_o), (grid_r, block_r, smem_r)
                        )
                        if "elementwise_kernelILi128E" in name_r:
                            continue
                        self.assertEqual(len(image_o), len(image_r))
                        self.assertEqual(image_o[:4], image_r[:4])
                        where = self._scalar_bytes(name, dtype)
                        if where is not None and len(ours) == 1:
                            off, size = where
                            self.assertEqual(
                                image_o[off : off + size],
                                image_r[off : off + size],
                                f"{name}: scalar bytes",
                            )

    def _op_of(self, name):
        aten = torch.ops.aten
        if "2.0 - x" in name:
            return aten.rsub.Scalar
        if "2.0 / x" in name:
            return aten.mul.Tensor
        if "+" in name or "add" in name:
            return aten.add.Tensor
        if "-" in name or "sub" in name:
            return aten.sub.Tensor
        if "/" in name or "div" in name:
            return aten.div.Tensor
        return aten.mul.Tensor

    def _entry_args(self, name, x, scalar):
        # the entry's arguments as the dispatcher hands them: the Python number
        # as a Python number (the dispatcher unwraps the wrapped tensor)
        number = int(scalar) if "(int)" in name else scalar
        if "2.0 - x" in name:
            return (x, scalar)
        if "2.0 / x" in name:
            return (x.reciprocal(), number)
        if "alpha=0.5" in name:
            return (x, number, 0.5)
        if "alpha=2" in name:
            return (x, number, 2)
        scalar_first = name.startswith(("0.5", "1.5", "2.0", "torch."))
        return (number, x) if scalar_first else (x, number)

    def test_scalar_operands_replay_at_new_shapes_as_constants(self):
        # the scalar is a trace-time constant: the tape serves other shapes
        # bitwise, and a call with another value of a scalar argument misses
        # on the argument contract before any GPU work
        forms = [
            "x * 0.5",
            "0.5 * x",
            "x + 1.5",
            "2.0 - x",
            "x - 2.0",
            "x / 2.0",
            "2.0 / x",
            "torch.sub(2.0, x)",
        ]
        for dtype in (torch.bfloat16, torch.float32):
            for name in forms:
                fn = self._SCALAR_FORMS[name]
                scalar = 0.5 if "0.5" in name else (1.5 if "1.5" in name else 2.0)
                x = self._pair(64, 4096, dtype=dtype)[0] + 1.0
                news = [
                    (self._pair(48, 3000, dtype=dtype)[0] + 1.0, scalar),
                    (self._pair(7, 1000, dtype=dtype)[0] + 1.0, scalar),
                    (self._pair(3, 8, dtype=dtype)[0] + 1.0, scalar),
                ]
                with self.subTest(op=name, dtype=dtype):
                    tape, variant, served = self._roundtrip(fn, (x, scalar), news)
                    self.assertEqual(served, len(news))
                    self.assertIsNone(variant.try_replay((x, scalar * 2)))
                    self.assertIsNotNone(variant.try_replay((x, scalar)))
        # the add forms record the scalar as a named constant field beside alpha
        x = self._pair(8, 4096)[0]
        tape = ht.trace(lambda t: t + 1.5, (x,))
        launch = tape.to_json()
        self.assertIn('"other_"', launch)
        tape = ht.trace(lambda t: 2.0 - t, (x,))
        self.assertIn('"self_"', tape.to_json())
        # a transposed single operand is coalesced to one dim and takes the
        # vectorized kernel as in eager; a slice-step or an expanded operand
        # cannot be coalesced with the contiguous output and takes the strided
        # kernel (elementwise_kernel<128, 4>, the scalar in the StridedOp
        # proxy). Each serves another shape of the same layout
        bf16 = torch.bfloat16
        layouts = {
            "transposed": (
                lambda M, N: torch.randn(N, M, device="cuda", dtype=bf16).t(),
                "vectorized_elementwise_kernel",
            ),
            "slice-step": (
                lambda M, N: torch.randn(M, 2 * N, device="cuda", dtype=bf16)[:, ::2],
                "elementwise_kernelILi128ELi4E",
            ),
            "expanded": (
                lambda M, N: torch.randn(M, 1, device="cuda", dtype=bf16).expand(M, N),
                "elementwise_kernelILi128ELi4E",
            ),
        }
        for layout, (make, kernel) in layouts.items():
            for name in ("x * 0.5", "x + 1.5", "2.0 - x", "x / 2.0"):
                fn = self._SCALAR_FORMS[name]
                scalar = 0.5 if "0.5" in name else (1.5 if "1.5" in name else 2.0)
                xl = make(64, 4096)
                with self.subTest(op=name, layout=layout):
                    real_nodes, _ = self._capture(lambda: fn(xl, scalar))
                    entry = ht._TRACED_ENTRIES[self._op_of(name)]
                    args = self._entry_args(name, xl, scalar)
                    ours, _ = self._capture(lambda: entry(*args))
                    self.assertEqual(
                        [_family(n[0]) for n in ours],
                        [_family(n[0]) for n in real_nodes],
                    )
                    self.assertIn(kernel, ours[0][0])
                    _, _, served = self._roundtrip(
                        fn, (xl, scalar), [(make(48, 3000), scalar)]
                    )
                    self.assertEqual(served, 1)
        self.assertFalse(C._host_trace_tracing())

    def test_a_0dim_cpu_tensor_operand_is_an_implicit_input_and_declines(self):
        # only the Python number the dispatcher unwrapped takes the constant
        # route; a 0-dim CPU tensor the caller made (a closure, a list entry,
        # a pinned or pageable argument) is a host value with no record on
        # the tape: read once at the trace, it would go stale when mutated
        x = self._pair(4, 8, dtype=torch.float32)[0]
        c = torch.tensor(0.5)
        message = "implicit CPU scalar input"
        forms = {
            "x * c": lambda t: t * c,
            "c * x": lambda t: c * t,
            "x + c": lambda t: t + c,
            "c - x": lambda t: c - t,
            "x / c": lambda t: t / c,
            "torch.add(x, c)": lambda t: torch.add(t, c),
        }
        for name, fn in forms.items():
            with self.subTest(form=name):
                with self.assertRaisesRegex(ht.Declined, message):
                    ht.trace(fn, (x,))
                with self.assertRaisesRegex(ht.Declined, message):
                    with EntryMode():
                        fn(x)
        pinned = torch.tensor(0.5).pin_memory()
        with self.assertRaisesRegex(ht.Declined, message):
            ht.trace(lambda t: t * pinned, (x,))
        # a CPU tensor argument declines at the trace entry (only CUDA inputs
        # are traced here); once pinned CPU inputs are admitted, a pinned one
        # declines by name at the sibling's entry and a pageable one as
        # pageable memory
        entry = "|only CUDA tensors are traced"
        with self.assertRaisesRegex(ht.Declined, message + entry):
            ht.trace(lambda t, u: t * u, (x, pinned))
        with self.assertRaisesRegex(ht.Declined, "pageable CPU memory" + entry):
            ht.trace(lambda t, u: t * u, (x, c))
        with self.assertRaisesRegex(ht.Declined, message):
            ht.trace(lambda t, l: t * l[0], (x, [c]))
        with self.assertRaisesRegex(ht.Declined, message):
            ht._TRACED_ENTRIES[torch.ops.aten.mul.Tensor](x, _CPU_TWO)
        self.assertFalse(C._host_trace_tracing())
        # a 0-dim CUDA tensor is read through its pointer at the replay: a
        # closure over one follows a later mutation, as eager does
        d = torch.tensor(0.5, device="cuda")

        def times_d(t):
            return t * d

        tape = ht.trace(times_d, (x,))
        variant = build(tape, times_d, (x,))
        d.fill_(0.75)
        (got,) = variant.replay((x,))
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(got, times_d(x)))
        # wrapped numbers are unaffected
        for fn in (lambda t: t * 0.5, lambda t: 0.5 * t):
            self.assertEqual(ht.trace(fn, (x,)).num_launches, 1)

    def test_alpha_beyond_the_opmath_range_raises_like_the_real_add(self):
        # add_kernel converts alpha with the checked alpha.to<opmath_t>(): an
        # alpha that overflows float raises in eager, in the entry, and in a
        # trace with or without the warm-up (the sibling computed with inf);
        # fp64's opmath is double and takes it
        message = "cannot be converted to type float without overflow"
        add = ht._TRACED_ENTRIES[torch.ops.aten.add.Tensor]
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            x, y = self._pair(8, 64, dtype=dtype)
            forms = {
                "add(x, 1.0, alpha=1e300)": lambda t: torch.add(t, 1.0, alpha=1e300),
                "add(x, y, alpha=1e300)": lambda t: torch.add(t, y, alpha=1e300),
                "sub(x, 1.0, alpha=-1e300)": lambda t: torch.sub(t, 1.0, alpha=-1e300),
                "rsub(x, 1.0, alpha=1e300)": lambda t: torch.rsub(t, 1.0, alpha=1e300),
                "add(x, 1e300, alpha=1e300)": lambda t: torch.add(
                    t, 1e300, alpha=1e300
                ),
            }
            for name, fn in forms.items():
                with self.subTest(form=name, dtype=dtype):
                    with self.assertRaisesRegex(RuntimeError, message):
                        fn(x)
                    with self.assertRaisesRegex(RuntimeError, message):
                        with EntryMode():
                            fn(x)
                    for warm_up in (False, True):
                        with self.assertRaisesRegex(RuntimeError, message):
                            ht.trace(fn, (x,), warm_up=warm_up)
                    self.assertFalse(C._host_trace_tracing())
            with self.assertRaisesRegex(RuntimeError, message):
                add(x, 1.0, alpha=1e300)
        x, y = self._pair(8, 64, dtype=torch.float64)
        want = torch.add(x, y, alpha=1e300)
        self.assertTrue(torch.equal(bits(add(x, y, alpha=1e300)), bits(want)))
        tape = ht.trace(lambda t: torch.add(t, 1.0, alpha=1e300), (x,), warm_up=False)
        self.assertEqual(tape.num_launches, 1)

    def test_integer_alpha_rounds_once(self):
        # the runtime team's case (convergence_20260918/robustness_r16/scalar):
        # eager keeps a Python int alpha as an int64 Scalar and converts it to
        # opmath once; 2**62 + 2**38 + 1 sits just above a float32 midpoint,
        # so int64 -> float rounds up while int64 -> double -> float loses the
        # +1 first and rounds to even, down. The sibling carries the int and
        # converts once in C++; 2**54 (add) / -2**54 (sub) as the tensor puts
        # the sum on a bf16 midpoint too, so both dtypes tell the two apart
        alpha = 2**62 + 2**38 + 1
        big = 2.0**54
        forms = {
            "add(x, y, alpha)": (lambda a, b, al: torch.add(a, b, alpha=al), big),
            "sub(x, y, alpha)": (lambda a, b, al: torch.sub(a, b, alpha=al), -big),
            "add_(x, 1, alpha)": (lambda a, b, al: a.mul(1).add_(1, alpha=al), big),
        }
        for (name, (fn, fill)), dtype in itertools.product(
            forms.items(), (torch.float32, torch.bfloat16)
        ):
            with self.subTest(form=name, dtype=dtype):
                x = torch.full((8, 64), fill, device="cuda", dtype=dtype)
                y = torch.ones_like(x)
                want = fn(x, y, alpha)
                self.assertFalse(torch.equal(want, fn(x, y, float(alpha))))
                with EntryMode():
                    got = fn(x, y, alpha)
                self.assertTrue(torch.equal(bits(got), bits(want)))
                tape = ht.trace(fn, (x, y, alpha), warm_up=False)
                variant = build(tape, fn, (x, y, alpha))
                for rows in (8, 12):
                    a = torch.full((rows, 64), fill, device="cuda", dtype=dtype)
                    b = torch.ones_like(a)
                    (out,) = variant.replay((a, b, alpha))
                    self.assertTrue(torch.equal(bits(out), bits(fn(a, b, alpha))))
                # the same value as a float is another constant: a miss
                self.assertIsNone(variant.try_replay((x, y, float(alpha))))
        self.assertNotEqual(ht._constant(2), ht._constant(2.0))

    def test_signed_zero_and_nan_scalar_operands_are_their_own_constants(self):
        # the scalar is baked into the launch as written: a tape traced with
        # 0.0 must not serve -0.0 (x / -0.0 is -inf where x / 0.0 is inf; the
        # zero results of the other ops differ in their sign bit), and a fresh
        # nan is the traced nan
        x = torch.arange(1, 9, device="cuda", dtype=torch.float32).view(2, 4)
        forms = {
            "x / s": lambda t, s: t / s,
            "x * s": lambda t, s: t * s,
            "x + s": lambda t, s: t + s,
            "s - x": lambda t, s: s - t,
        }
        for name, fn in forms.items():
            with self.subTest(form=name):
                tape = ht.trace(fn, (x, 0.0))
                variant = build(tape, fn, (x, 0.0))
                self.assertIsNone(variant.try_replay((x, -0.0)))
                self.assertIsNotNone(variant.try_replay((x, 0.0)))
                other = build(ht.trace(fn, (x, -0.0)), fn, (x, -0.0))
                (got,) = other.replay((x, -0.0))
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(bits(got), bits(fn(x, -0.0))))
        fn = self._SCALAR_FORMS["x * 0.5"]
        nan = float("nan")
        variant = build(ht.trace(fn, (x, nan)), fn, (x, float("nan")))
        self.assertIsNotNone(variant.try_replay((x, float("nan"))))
        self.assertIsNone(variant.try_replay((x, -nan)))

    def test_empty_outputs_take_eagers_strides(self):
        # an empty result of the sibling has the strides eager's iterator
        # gives it: the contiguous fast path (an empty tensor is contiguous)
        # and c10::contiguous_strides' (1, 1) for (16, 0), not the general
        # path's (0, 1)
        x = torch.randn(16, 1024, device="cuda")
        forms = {
            "x[:, :0] / 2": lambda t: t[:, :0] / 2,
            "x[:, :0] * 0.5": lambda t: t[:, :0] * 0.5,
            "tanh(x[:, :0])": lambda t: torch.tanh(t[:, :0]),
            "x[:0] * 0.5": lambda t: t[:0] * 0.5,
            "x[:0, :0] / 2": lambda t: t[:0, :0] / 2,
            "x[:, :0].t() / 2": lambda t: t[:, :0].t() / 2,
        }
        for name, fn in forms.items():
            with self.subTest(form=name):
                want = fn(x)
                with EntryMode():
                    sibling = fn(x)
                tape = ht.trace(fn, (x,))
                self.assertEqual(tape.num_launches, 0)
                (got,) = build(tape, fn, (x,)).replay((x,))
                for out in (sibling, got):
                    self.assertEqual(
                        (out.shape, out.stride(), out.dtype),
                        (want.shape, want.stride(), want.dtype),
                    )

    def test_gelu_new_formula_traces(self):
        # Hugging Face's NewGELUActivation as written: pow, tanh and four
        # scalar operands; the scalars ride as constants, eight launches

        def gelu_new(t):
            inner = math.sqrt(2.0 / math.pi) * (t + 0.044715 * torch.pow(t, 3.0))
            return 0.5 * t * (1.0 + torch.tanh(inner))

        for dtype in (torch.bfloat16, torch.float32):
            x = self._pair(4, 3072, dtype=dtype)[0]
            tape = ht.trace(gelu_new, (x,))
            self.assertEqual(tape.num_launches, 8)
            variant = build(tape, gelu_new, (x,))
            for y in (
                x,
                self._pair(6, 3072, dtype=dtype)[0],
                self._pair(1, 3072, dtype=dtype)[0],
            ):
                got = variant.try_replay((y,))
                if got is None:
                    continue
                self.assertTrue(torch.equal(bits(got[0]), bits(gelu_new(y))))

    def test_pow_tensor_scalar_follows_the_kernel_host(self):
        # PowKernel.cu routes 0.5 / -0.5 / -1 to sqrt / rsqrt / reciprocal,
        # 2, 3, -2 to closed forms and the rest to pow_ with the exponent
        # captured: bitwise the real op with the same kernel family
        for dtype in (torch.bfloat16, torch.float32, torch.float16):
            x = torch.rand(48, 3000, device="cuda").to(dtype) + 0.5
            for exp in (0.5, -0.5, -1.0, 2.0, 3.0, -2.0, 2.5, 3, 2, -3):
                with self.subTest(exp=exp, dtype=dtype):
                    fn = lambda t: torch.pow(t, exp)  # noqa: E731
                    want = fn(x)
                    real_nodes, _ = self._capture(lambda: fn(x))
                    ours, _ = self._capture(
                        lambda: C._host_trace_ti_pow_tensor_scalar(x, exp)
                    )
                    self.assertEqual(len(ours), 1)
                    self.assertEqual(_family(ours[0][0]), _family(real_nodes[0][0]))
                    self.assertEqual(ours[0][1:4], real_nodes[0][1:4])
                    tape = ht.trace(fn, (x,))
                    (got,) = build(tape, fn, (x,)).replay((x,))
                    self.assertTrue(
                        torch.equal(bits(got), bits(want)),
                        f"pow {exp} differs bitwise",
                    )
        # exponent 0 is eager's fill_(1) into the structured allocation (one
        # fill launch, an int base too), exponent 1 its copy_(base)
        x = torch.rand(8, 64, device="cuda", dtype=torch.bfloat16) + 0.5
        xi = torch.arange(64, device="cuda").view(8, 8)
        for base, exp in ((x, 0.0), (x, 0), (xi, 0), (x, False)):
            with self.subTest(exp=exp, dtype=base.dtype):
                fn = lambda t: torch.pow(t, exp)  # noqa: E731
                eager = assert_eager_function_handles(self, fn, (base,), launches=1)
                self.assertEqual(len(eager), 1)
                (got,) = build(ht.trace(fn, (base,)), fn, (base,)).replay((base,))
                self.assertTrue(torch.equal(got, fn(base)))
        for base, exp in ((x, 1.0), (x, 1), (xi, 1), (x[:, ::2], 1.0), (x, True)):
            with self.subTest(
                exp=exp, dtype=base.dtype, strided=not base.is_contiguous()
            ):
                fn = lambda t: torch.pow(t, exp)  # noqa: E731
                tape = ht.trace(fn, (base,))
                self.assertEqual(
                    (tape.num_launches, tape.num_memcpys),
                    (0, 1) if base.is_contiguous() else (1, 0),
                )
                (got,) = build(tape, fn, (base,)).replay((base,))
                self.assertTrue(torch.equal(got, fn(base)))
        with self.assertRaisesRegex(ht.Declined, "promotes the base"):
            ht.trace(lambda t: torch.pow(t, 2.5), (xi,))
        with self.assertRaisesRegex(RuntimeError, "negative integer powers"):
            ht.trace(lambda t: torch.pow(t, -2), (xi,), warm_up=False)
        self.assertFalse(C._host_trace_tracing())

    def test_reshape_of_a_non_viewable_tensor_copies_like_eager(self):
        # reshape's composite decides between a view and a copy on the
        # symbolic metadata as eager decides on the concrete one: a transposed
        # source flattened is a clone (the sibling copy) then a view
        def flat(t):
            return t.t().reshape(-1)

        x = self._pair(8, 4096)[0]
        tape = ht.trace(flat, (x,))
        self.assertEqual(tape.num_launches, 1)
        variant = build(tape, flat, (x,))
        y = self._pair(6, 3000)[0]
        (got,) = variant.replay((y,))
        self.assertEqual(got.stride(), flat(y).stride())
        self.assertTrue(torch.equal(got, flat(y)))

        # a viewable reshape is a view: no launch, eager's strides
        def head(t):
            return t[:, :4096].reshape(t.shape[0], 64, 64)

        tape = ht.trace(head, (x,))
        self.assertEqual(tape.num_launches, 0)
        (out,) = build(tape, head, (x,)).replay((x,))
        self.assertEqual(out.stride(), head(x).stride())

    # ---- fills, arange, comparisons, masked_fill, clamp and the in-place /
    # out= / .Scalar forms (the Llama-architecture run's declines): each
    # traced at one shape, replayed at two others and at new addresses,
    # bitwise eager

    _MORE_DTYPES = (
        torch.float32,
        torch.bfloat16,
        torch.float16,
        torch.int64,
        torch.bool,
    )

    def _values(self, M, N, dtype, offset=0):
        # varied values of the dtype at a fresh address (`offset` moves it)
        flat = torch.randn(M * N + offset, device="cuda") * 3
        x = flat[offset:].view(M, N)
        if dtype is torch.bool:
            return x > 0
        return x.to(dtype)

    def _replays(self, fn, base, news, warm_up=True):
        # fn does not mutate its inputs: trace and build at base, replay at
        # each of news bitwise eager (shape, strides, dtype, bits)
        tape = ht.trace(fn, base, warm_up=warm_up)
        variant = build(tape, fn, base)
        for new in news:
            out = variant.try_replay(new)
            shapes = [tuple(a.shape) for a in new if isinstance(a, torch.Tensor)]
            self.assertIsNotNone(out, f"miss at {shapes}")
            want = fn(*new)
            want = (want,) if isinstance(want, torch.Tensor) else tuple(want)
            torch.cuda.synchronize()
            self.assertEqual(len(out), len(want))
            for o, w in zip(out, want):
                self.assertEqual(
                    (o.shape, o.stride(), o.dtype), (w.shape, w.stride(), w.dtype)
                )
                self.assertTrue(torch.equal(bits(o), bits(w)), "outputs differ bitwise")
        return tape, variant

    def _inplace_replays(self, fn, make, shapes):
        # fn mutates its first argument: the trace and the build get their own
        # copies (the build runs nothing); each replay runs on a fresh input
        # beside eager on a clone of it, and the input's storage is what
        # changed on both sides
        tape = ht.trace(fn, make(*shapes[0]))
        variant = build(tape, fn, make(*shapes[0]))
        for shape in shapes:
            args = make(*shape)
            clones = tuple(a.clone() for a in args)
            out = variant.try_replay(args)
            self.assertIsNotNone(out, f"miss at {shape}")
            want = fn(*clones)
            torch.cuda.synchronize()
            self.assertEqual(
                (out[0].shape, out[0].stride()), (want.shape, want.stride())
            )
            self.assertTrue(torch.equal(bits(out[0]), bits(want)))
            self.assertEqual(
                out[0].untyped_storage().data_ptr(),
                args[0].untyped_storage().data_ptr(),
            )
            self.assertTrue(torch.equal(bits(args[0]), bits(clones[0])))
        return tape

    def test_fills_replay_at_new_shapes(self):
        # fill_ / zero_ and the factories over a traced tensor's shape (a
        # SymInt size): one launch into one allocation, at every dtype; the
        # zero forms are eager's memset over a dense tensor (a memset record,
        # no launch) and its fill launch over a strided one
        memset_forms = {
            "zeros",
            "zeros_like",
            "new_zeros",
            "zero_",
            "zero_ of a transposed allocation",
        }
        for dtype in self._MORE_DTYPES:
            v = True if dtype is torch.bool else 3
            forms = {
                "full": lambda x: torch.full(x.shape, v, dtype=dtype, device=x.device),
                "full symint tuple": lambda x: torch.full(
                    (x.shape[0], x.shape[1] + 1), v, dtype=dtype, device=x.device
                ),
                "zeros": lambda x: torch.zeros(x.shape, dtype=dtype, device=x.device),
                "ones": lambda x: torch.ones(x.shape, dtype=dtype, device=x.device),
                "zeros_like": torch.zeros_like,
                "ones_like": torch.ones_like,
                "full_like": lambda x: torch.full_like(x, v),
                "new_zeros": lambda x: x.new_zeros((x.shape[0], 3)),
                "new_ones": lambda x: x.new_ones((x.shape[1],)),
                "new_full": lambda x: x.new_full((2, x.shape[0]), v),
                "fill_": lambda x: torch.empty_like(x).fill_(v),
                "zero_": lambda x: torch.empty_like(x).zero_(),
                "fill": lambda x: torch.fill(x, v),
                "fill_ of a transposed allocation": lambda x: torch.empty_like(x)
                .t()
                .fill_(v),
                "zero_ of a transposed allocation": lambda x: torch.empty_like(x)
                .t()
                .zero_(),
                "zero_ of a strided view": lambda x: torch.empty(
                    x.shape[0], 2 * x.shape[1], dtype=dtype, device=x.device
                )[:, ::2].zero_(),
            }
            for name, fn in forms.items():
                with self.subTest(op=name, dtype=dtype):
                    base = (self._values(64, 4096, dtype),)
                    news = [
                        (self._values(48, 3000, dtype, offset=8),),
                        (self._values(7, 1000, dtype),),
                    ]
                    tape, _ = self._replays(fn, base, news)
                    self.assertEqual(
                        (tape.num_launches, len(tape.memsets), tape.num_allocations),
                        (0, 1, 1) if name in memset_forms else (1, 0, 1),
                    )
        # the dtype eager infers from the value (infer_full_options), and the
        # value as a constant of the tape: another value misses on the
        # argument contract
        x = self._values(8, 64, torch.float32)
        for value, dtype in (
            (3, torch.int64),
            (0.5, torch.float32),
            (True, torch.bool),
        ):
            fn = lambda t, v: torch.full((t.shape[0], 5), v, device=t.device)  # noqa: E731
            tape = ht.trace(fn, (x, value))
            self.assertEqual(tape.outputs[0].dtype, dtype)
            variant = build(tape, fn, (x, value))
            self.assertIsNone(variant.try_replay((x, value + 1)))
            self.assertTrue(torch.equal(variant.replay((x, value))[0], fn(x, value)))
        # the fill kernel is eager's: same family, grid and block, contiguous
        # (vectorized) and strided
        for make in (
            lambda: torch.empty(64, 4096, device="cuda"),
            lambda: torch.empty(64, 8192, device="cuda")[:, ::2],
        ):
            t = make()
            real_nodes, _ = self._capture(lambda: t.fill_(2.5))
            ours, _ = self._capture(lambda: C._host_trace_ti_fill_(t, 2.5))
            self.assertEqual(len(ours), 1)
            self.assertEqual(_family(ours[0][0]), _family(real_nodes[0][0]))
            self.assertEqual(ours[0][1:4], real_nodes[0][1:4])
        # a factory off the trace's device declines by name
        with self.assertRaisesRegex(ht.Declined, "aten.zeros.default on cpu"):
            ht.trace(lambda t: torch.zeros(3), (x,))
        with self.assertRaisesRegex(ht.Declined, "pinned"):
            ht.trace(lambda t: torch.ones_like(t, pin_memory=True), (x,), warm_up=False)
        self.assertFalse(C._host_trace_tracing())

    @unittest.skipIf(not TEST_CUDA_PYTHON_BINDINGS, "cuda.bindings reads the nodes")
    def test_arange_replays_eager_function_handles(self):
        # the entry is compiled into RangeFactories.cu and launches that file's
        # elementwise_kernel_with_index over eager's ArangeFunctor: the entry's
        # node, the tape's launch and the replay's node hold the function
        # handle eager's capture holds (E36): an integral arange over a
        # symbolic length, one with a step, a floating and a bf16 one
        x = torch.randn(4, 4096, device="cuda")

        def entry(start, step, n, dtype):
            def call(k):
                out = torch.empty(n(k), device=k.device, dtype=dtype)
                return C._host_trace_ti_arange(start, step, out)

            return call

        i32, i64, f32, bf16 = torch.int32, torch.int64, torch.float32, torch.bfloat16
        cases = {
            "i64, symbolic length": (
                lambda k: torch.arange(3, 3 + k.shape[-1], device=k.device),
                entry(3, 1, lambda k: k.shape[-1], i64),
            ),
            "i32, step 2": (
                lambda k: torch.arange(
                    0, 2 * k.shape[-1], 2, device=k.device, dtype=i32
                ),
                entry(0, 2, lambda k: k.shape[-1], i32),
            ),
            "f32": (
                lambda k: torch.arange(0.5, 1024.5, 0.25, device=k.device),
                entry(0.5, 0.25, lambda k: 4096, f32),
            ),
            "bf16": (
                lambda k: torch.arange(1, 1025, device=k.device, dtype=bf16),
                entry(1, 1, lambda k: 1024, bf16),
            ),
        }
        for name, (real, ours) in cases.items():
            with self.subTest(case=name):
                eager = assert_eager_function_handles(
                    self, real, (x,), ours, launches=1
                )
                self.assertEqual(len(eager), 1)

    def test_arange_over_symint_bounds_replays(self):
        # the model-derived cache_position: torch.arange(past, past + q) over
        # the cache's SymInt length. The bounds stay values of the tape (the
        # length a size, start and step fields of the launch), so a trace at
        # one cache length serves the others
        def cache_position(k):
            past = k.shape[-2]
            return torch.arange(past, past + 1, device=k.device)

        def kv(L, B=4):
            return (torch.randn(B, 2, L, 8, device="cuda", dtype=torch.bfloat16),)

        tape, variant = self._replays(
            cache_position, kv(16), [kv(24), kv(40, B=1), kv(1)]
        )
        self.assertEqual((tape.num_launches, tape.num_allocations), (1, 1))
        self.assertIn('"xstart"', tape.to_json())
        self.assertEqual(variant.replay(kv(33))[0].tolist(), [33])
        forms = {
            "arange(L)": lambda k: torch.arange(k.shape[-2], device=k.device),
            "arange(0, L, 2)": lambda k: torch.arange(
                0, k.shape[-2], 2, device=k.device
            ),
            "arange(L, 0, -3)": lambda k: torch.arange(
                k.shape[-2], 0, -3, device=k.device
            ),
            "arange(L - 2, 2 * L + 1)": lambda k: torch.arange(
                k.shape[-2] - 2, 2 * k.shape[-2] + 1, device=k.device
            ),
            "arange(L) int32": lambda k: torch.arange(
                k.shape[-2], device=k.device, dtype=torch.int32
            ),
            "arange(0, L) fp32": lambda k: torch.arange(
                0, k.shape[-2], device=k.device, dtype=torch.float32
            ),
            "arange(0, L, 2) bf16": lambda k: torch.arange(
                0, k.shape[-2], 2, device=k.device, dtype=torch.bfloat16
            ),
            "arange(L, 0, -1) fp16": lambda k: torch.arange(
                k.shape[-2], 0, -1, device=k.device, dtype=torch.float16
            ),
            "arange(8) constant": lambda k: torch.arange(8, device=k.device),
            "arange(-3, 3) constant": lambda k: torch.arange(-3, 3, device=k.device),
            "arange(0.5, 3.25, 0.5) float constants": lambda k: torch.arange(
                0.5, 3.25, 0.5, device=k.device
            ),
            "arange(0, 1000, 3) constant": lambda k: torch.arange(
                0, 1000, 3, device=k.device
            ),
        }
        for name, fn in forms.items():
            with self.subTest(op=name):
                tape, _ = self._replays(fn, kv(16), [kv(24), kv(1000, B=1), kv(3)])
                self.assertEqual((tape.num_launches, tape.num_allocations), (1, 1))
        # the index kernel's launch shape is eager's (64 threads, one element
        # each); the values bitwise
        for dtype in (torch.int64, torch.float32, torch.bfloat16):
            entry = ht._TRACED_ENTRIES[torch.ops.aten.arange.start_step]
            real_nodes, want = self._capture(
                lambda: torch.arange(3, 1000, 2, device="cuda", dtype=dtype)
            )
            ours, got = self._capture(
                lambda: entry(3, 1000, 2, device="cuda", dtype=dtype)
            )
            self.assertEqual(len(ours), 1)
            self.assertEqual(ours[0][1:4], real_nodes[0][1:4])
            self.assertTrue(torch.equal(bits(got), bits(want)))
        # eager's checks keep their texts; a floating bound into an integral
        # dtype (a truncating conversion) declines by name
        k = kv(16)
        with self.assertRaisesRegex(RuntimeError, "step must be nonzero"):
            ht.trace(lambda t: torch.arange(0, t.shape[-2], 0, device=t.device), k)
        with self.assertRaisesRegex(
            RuntimeError, "upper bound and lower bound inconsistent with step sign"
        ):
            ht.trace(lambda t: torch.arange(t.shape[-2], 0, device=t.device), k)
        with self.assertRaisesRegex(RuntimeError, "step must be nonzero"):
            ht.trace(
                lambda t: torch.arange(0, t.shape[-2], 0, device=t.device),
                k,
                warm_up=False,
            )
        with self.assertRaisesRegex(ht.Declined, "floating bound into torch.int64"):
            ht.trace(
                lambda t: torch.arange(
                    0.5, t.shape[-2], dtype=torch.int64, device=t.device
                ),
                k,
            )
        with self.assertRaisesRegex(ht.Declined, "aten.arange.start_out"):
            ht.trace(
                lambda t: torch.arange(
                    0,
                    t.shape[-2],
                    out=torch.empty(16, device=t.device, dtype=torch.int64),
                ),
                k,
            )
        self.assertFalse(C._host_trace_tracing())

    def test_comparisons_with_a_scalar_produce_bool(self):
        # eq / ne / lt / le / gt / ge with a Python number on either side and
        # the Tensor forms: a bool output over the operand's dtype
        # (a bool operand compares with a bool: an int scalar promotes it to
        # int64 in eager, which the sibling declines as promotion)
        scalar_forms = {
            "x == 1": lambda x, one: x == one,
            "x != 1": lambda x, one: x != one,
            "x < 1": lambda x, one: x < one,
            "x <= 1": lambda x, one: x <= one,
            "x > 1": lambda x, one: x > one,
            "x >= 1": lambda x, one: x >= one,
            "1 < x": lambda x, one: one < x,
            "torch.eq(x, 1)": lambda x, one: torch.eq(x, one),
            "torch.ge(x, 1)": lambda x, one: torch.ge(x, one),
        }
        tensor_forms = {
            "x == y": lambda x, y: x == y,
            "x != y": lambda x, y: x != y,
            "x < y": lambda x, y: x < y,
            "x >= y[0]": lambda x, y: x >= y[0],
            "torch.gt(x, y)": lambda x, y: torch.gt(x, y),
        }
        for dtype in self._MORE_DTYPES:
            one = True if dtype is torch.bool else 1
            for name, fn in scalar_forms.items():
                with self.subTest(op=name, dtype=dtype):
                    tape, _ = self._replays(
                        fn,
                        (self._values(64, 4096, dtype), one),
                        [
                            (self._values(48, 3000, dtype, offset=8), one),
                            (self._values(7, 1000, dtype), one),
                        ],
                    )
                    self.assertEqual(
                        (tape.num_launches, tape.outputs[0].dtype), (1, torch.bool)
                    )
            for name, fn in tensor_forms.items():
                with self.subTest(op=name, dtype=dtype):
                    tape, _ = self._replays(
                        fn,
                        (self._values(64, 4096, dtype), self._values(64, 4096, dtype)),
                        [
                            (
                                self._values(48, 3000, dtype, offset=8),
                                self._values(48, 3000, dtype),
                            ),
                            (
                                self._values(7, 1000, dtype),
                                self._values(7, 1000, dtype, offset=16),
                            ),
                        ],
                    )
                    self.assertEqual(
                        (tape.num_launches, tape.outputs[0].dtype), (1, torch.bool)
                    )
        # floating scalars against floating operands, with the rounding eager
        # applies (the scalar is converted to the operand's dtype)
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            for fn in (
                lambda x: x < 0.3,
                lambda x: x == 0.30078125,
                lambda x: x >= -1.7,
            ):
                self._replays(
                    fn,
                    (self._values(64, 4096, dtype),),
                    [(self._values(48, 3000, dtype, offset=8),)],
                )
        # a scalar that would promote the operand and a dtype pair decline
        xi = self._values(8, 64, torch.int64)
        with self.assertRaisesRegex(ht.Declined, "promotes its CUDA operand"):
            ht.trace(lambda t: t == 1.5, (xi,))
        with self.assertRaisesRegex(
            ht.Declined, "promotes its CUDA operand from Bool to Long"
        ):
            ht.trace(lambda t: t == 1, (xi > 0,))
        # a dtype pair promotes through the cast kernel
        self.assertEqual(ht.trace(lambda t, u: t < u, (xi, xi.float())).num_launches, 1)
        # the comparison kernels are eager's families
        x = self._values(64, 4096, torch.bfloat16)
        for fn, entry_args in (
            (lambda: x < 1, (x, 1, "lt")),
            (lambda: x == 1, (x, 1, "eq")),
            (lambda: x[:, ::2] > 1, (x[:, ::2], 1, "gt")),
        ):
            real_nodes, _ = self._capture(fn)
            ours, _ = self._capture(lambda: C._host_trace_ti_compare(*entry_args))
            self.assertEqual(len(ours), 1)
            self.assertEqual(_family(ours[0][0]), _family(real_nodes[0][0]))
            self.assertEqual(ours[0][1:4], real_nodes[0][1:4])
        self.assertFalse(C._host_trace_tracing())

    def test_mask_through_eq_scalar_and_masked_fill(self):
        # an attention-mask idiom: the int64 mask compared with a scalar, the
        # bool result masking scores in place; padded and unpadded masks,
        # broadcast over heads, at two batch / length pairs
        def scores(x, m):
            return (x * 1.0).masked_fill_(m == 0, float("-inf"))

        def scores_over_heads(x, m):
            return (x * 1.0).masked_fill_((m == 0)[:, None, :], -1e4)

        def make(B, L, dtype, padded, H=None):
            x = torch.randn((B, L) if H is None else (B, H, L), device="cuda").to(dtype)
            m = torch.ones(B, L, device="cuda", dtype=torch.int64)
            if padded:
                for b in range(B):
                    m[b, : b % L] = 0
            return x, m

        for dtype in (torch.bfloat16, torch.float32):
            for padded in (False, True):
                with self.subTest(dtype=dtype, padded=padded):
                    tape, _ = self._replays(
                        scores,
                        make(4, 16, dtype, padded),
                        [make(4, 24, dtype, padded), make(2, 40, dtype, not padded)],
                    )
                    self.assertEqual((tape.num_launches, tape.num_allocations), (3, 2))
                    tape, _ = self._replays(
                        scores_over_heads,
                        make(4, 16, dtype, padded, H=8),
                        [
                            make(4, 24, dtype, padded, H=8),
                            make(2, 40, dtype, not padded, H=3),
                        ],
                    )
                    self.assertEqual(tape.num_launches, 3)
        # masked_fill_ on a view of an input writes the input's storage
        x, m = make(4, 16, torch.float32, True)
        self._inplace_replays(
            lambda t, mask: t[:, 1:].masked_fill_(mask[:, 1:] == 0, 0.0),
            lambda B, L: make(B, L, torch.float32, True),
            [(4, 16), (4, 24), (2, 40)],
        )
        # the out-of-place form: eager clones self first, a memcpy record for
        # a contiguous self (copy_d2d) and a kernel copy for a strided one
        tape, _ = self._replays(
            lambda t, mask: t.masked_fill(mask == 0, 0.0),
            (x, m),
            [make(4, 24, torch.float32, True), make(2, 40, torch.float32, False)],
        )
        self.assertEqual((tape.num_launches, len(tape.memcpys)), (2, 1))
        tape, _ = self._replays(
            lambda t, mask: t.t().masked_fill(mask.t() == 0, 0.0),
            (x, m),
            [make(4, 24, torch.float32, True), make(2, 40, torch.float32, False)],
        )
        self.assertEqual(tape.num_launches, 3)
        # eager's checks keep their texts
        with self.assertRaisesRegex(
            RuntimeError, "masked_fill only supports boolean masks"
        ):
            ht.trace(lambda t, mask: (t * 1.0).masked_fill_(mask, 0.0), (x, m))
        with self.assertRaisesRegex(
            RuntimeError, "masked_fill only supports boolean masks"
        ):
            ht.trace(
                lambda t, mask: (t * 1.0).masked_fill_(mask, 0.0), (x, m), warm_up=False
            )
        # the eq of HF's mask test and the all() behind it trace (the
        # comparison and the reduction); the bool() of the result is a host
        # read and declines by name
        tape = ht.trace(lambda mask: torch.all(mask == 1), (m,))
        self.assertEqual(tape.num_launches, 2)
        with self.assertRaisesRegex(ht.Declined, r"bool\(\)"):
            ht.trace(lambda mask: bool(torch.all(mask == 1)), (m,))
        self.assertFalse(C._host_trace_tracing())

    def test_in_place_scalar_ops_write_the_input_storage(self):
        # add_ / sub_ / mul_ / div_ with a number (the Tensor overload with a
        # wrapped number, and the .Scalar overload C++ callers use), out= and
        # the out-of-place .Scalar overloads: the same kernels as the
        # allocating forms, the destination's storage written
        aten = torch.ops.aten
        inplace = {
            "t.add_(1.5)": lambda t: t.add_(1.5),
            "t.sub_(2.0)": lambda t: t.sub_(2.0),
            "t.mul_(0.5)": lambda t: t.mul_(0.5),
            "t.div_(2.0)": lambda t: t.div_(2.0),
            "t.add_(1.0, alpha=2)": lambda t: t.add_(1.0, alpha=2),
            "add_.Scalar": lambda t: aten.add_.Scalar(t, 1.5),
            "sub_.Scalar": lambda t: aten.sub_.Scalar(t, 1.5),
            "mul_.Scalar": lambda t: aten.mul_.Scalar(t, 0.5),
            "div_.Scalar": lambda t: aten.div_.Scalar(t, 4.0),
            "t += 1.5": lambda t: t.__iadd__(1.5),
            "t[:, 1:].mul_(0.5) (a view of the input)": lambda t: t[:, 1:].mul_(0.5),
            "t.t().add_(1.0) (strided)": lambda t: t.t().add_(1.0),
        }
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            for name, fn in inplace.items():
                with self.subTest(op=name, dtype=dtype):
                    tape = self._inplace_replays(
                        fn,
                        lambda M, N: (self._values(M, N, dtype),),
                        [(64, 4096), (48, 3000), (7, 1000)],
                    )
                    self.assertEqual((tape.num_launches, tape.num_allocations), (1, 0))
        # a tensor other, in place
        self._inplace_replays(
            lambda t, u: t.add_(u, alpha=0.5),
            lambda M, N: (
                self._values(M, N, torch.float32),
                self._values(M, N, torch.float32, offset=8),
            ),
            [(64, 4096), (48, 3000)],
        )
        # in place on an allocation: no new allocation, the second launch
        # writes the first's output
        tape, _ = self._replays(
            lambda t: (t * 2).add_(1.0),
            (self._values(64, 4096, torch.float32),),
            [(self._values(48, 3000, torch.float32),)],
        )
        self.assertEqual((tape.num_launches, tape.num_allocations), (2, 1))
        # the out-of-place .Scalar overloads and out=
        outplace = {
            "add.Scalar": lambda t: aten.add.Scalar(t, 1.5),
            "sub.Scalar": lambda t: aten.sub.Scalar(t, 1.5),
            "mul.Scalar": lambda t: aten.mul.Scalar(t, 0.5),
            "div.Scalar": lambda t: aten.div.Scalar(t, 4.0),
            "torch.add(t, 1.5, out=)": lambda t: torch.add(
                t, 1.5, out=torch.empty_like(t)
            ),
            "torch.mul(t, 0.5, out=)": lambda t: torch.mul(
                t, 0.5, out=torch.empty_like(t)
            ),
            "torch.sub(t, 1.0, alpha=2, out=)": lambda t: torch.sub(
                t, 1.0, alpha=2, out=torch.empty_like(t)
            ),
            "torch.div(t, 2.0, out=)": lambda t: torch.div(
                t, 2.0, out=torch.empty_like(t)
            ),
        }
        for name, fn in outplace.items():
            with self.subTest(op=name):
                tape, _ = self._replays(
                    fn,
                    (self._values(64, 4096, torch.bfloat16),),
                    [
                        (self._values(48, 3000, torch.bfloat16, offset=8),),
                        (self._values(7, 1000, torch.bfloat16),),
                    ],
                )
                self.assertEqual((tape.num_launches, tape.num_allocations), (1, 1))
        # mul_ and add_ on integers (both kernels take them: add's through
        # the generated entry in UfuncCUDA_add.cu, E36 stage 2)
        for fn in (lambda t: t.mul_(3), lambda t: t.add_(1)):
            self._inplace_replays(
                fn,
                lambda M, N: (self._values(M, N, torch.int64),),
                [(64, 4096), (7, 1000)],
            )
        # an in-place op whose other would broadcast the destination keeps
        # eager's text; an out= tensor of another shape would be resized,
        # which declines by name
        x, u = (
            self._values(8, 64, torch.float32),
            self._values(8, 64, torch.float32, offset=8),
        )
        with self.assertRaisesRegex(RuntimeError, "doesn't match the broadcast shape"):
            ht.trace(lambda t, o: t[0].add_(o), (x, u))
        with self.assertRaisesRegex(RuntimeError, "doesn't match the broadcast shape"):
            ht.trace(lambda t, o: t[0].add_(o), (x, u), warm_up=False)
        with self.assertRaisesRegex(ht.Declined, "does not match the result's shape"):
            ht.trace(
                lambda t: torch.add(t, 1.0, out=torch.empty(3, device=t.device)),
                (x,),
                warm_up=False,
            )
        # F.rms_norm's _fused_rms_norm has a CUDA kernel beside a
        # CompositeImplicit decomposition eager never runs on CUDA: the mode
        # follows eager's route (DECISIONS O40 / A190) instead of tracing add_
        # and the rest of that decomposition
        w = torch.ones(4096, device="cuda") * 1.5
        x = self._values(64, 4096, torch.float32)
        _assert_fused_rms_norm_route(
            self, lambda t, g: F.rms_norm(t, (4096,), g, 1e-5), (x, w)
        )
        self.assertFalse(C._host_trace_tracing())

    def _assert_same_work(self, tape, eager):
        ours = _tape_work(tape)
        self.assertEqual(len(ours), len(eager), (ours, eager))
        for (kind, name), (kind_e, name_e) in zip(ours, eager):
            self.assertEqual(kind, kind_e, (ours, eager))
            if kind != "kernel":
                continue
            # a converted host launches eager's own kernel (E36)
            self.assertEqual(name, name_e)

    def test_tape_holds_the_device_work_eager_issues(self):
        # one op per family: the tape's launches and memsets, in issue order,
        # are the device work eager issues for the same call (the profiler's
        # device events): a sibling's launch is eager's kernel template with
        # the same leading integer arguments and operand array, a converted
        # host's launch is eager's kernel by its full name, a memset is a
        # memset. An op that decomposes under the mode into kernels eager does
        # not launch (F.rms_norm's _fused_rms_norm, DECISIONS O40 / A190)
        # declines instead of failing here
        x = self._values(64, 4096, torch.float32)
        m = x > 0
        w = torch.ones(4096, device="cuda") * 1.5
        ln_stats = torch.ops.aten.native_layer_norm.default(x, [4096], w, w, 1e-5)[1:]
        logp = torch.log_softmax(x, -1)
        tg = torch.randint(0, 4096, (64,), device="cuda")
        seed = torch.ones((), device="cuda")
        total_weight = torch.ops.aten.nll_loss_forward.default(x, tg, None, 1, -100)[1]
        ids = torch.randint(0, 1000, (512,), device="cuda")
        g = self._values(512, 64, torch.bfloat16)
        ops = {
            "add.Scalar": (lambda t: t + 1.5, (x,)),
            "add.Tensor": (lambda t, u: t + u, (x, w)),
            "silu": (F.silu, (x,)),
            "sin": (torch.sin, (x,)),
            "pow": (lambda t: t.pow(2), (x,)),
            "copy_ (strided)": (lambda t: t.t().contiguous(), (x,)),
            "_to_copy (cast)": (lambda t: t.to(torch.bfloat16), (x,)),
            "full": (lambda t: torch.full((t.shape[0], 8), 2.0, device=t.device), (x,)),
            "zeros_like (memset)": (torch.zeros_like, (x,)),
            "zero_ (strided view)": (
                lambda t: torch.empty(t.shape[0], 8192, device=t.device)[
                    :, ::2
                ].zero_(),
                (x,),
            ),
            "arange": (
                lambda t: torch.arange(3, 3 + t.shape[0], device=t.device),
                (x,),
            ),
            "eq.Scalar": (lambda t: t == 0.0, (x,)),
            "gt.Tensor": (lambda t, u: t > u, (x, w)),
            "masked_fill_": (lambda t, k: (t * 2.0).masked_fill_(k, -1.0), (x, m)),
            "clamp": (lambda t: t.clamp(-0.5, 0.5), (x,)),
            "sum": (lambda t: t.sum(1), (x,)),
            "amax": (lambda t: t.amax(1), (x,)),
            "softmax (converted host)": (lambda t: t.softmax(-1), (x,)),
            # the training hosts: layer norm backward (commit 1), the
            # log_softmax backward and the nll_loss pair (commit 7); the nll
            # backward zeroes its gradient with a memset before its kernel
            "native_layer_norm_backward (converted host)": (
                lambda t, g: torch.ops.aten.native_layer_norm_backward.default(
                    t, t, [4096], *ln_stats, g, g, [True, True, True]
                ),
                (x, w),
            ),
            "_log_softmax_backward_data (converted host)": (
                lambda t: torch.ops.aten._log_softmax_backward_data.default(
                    t, logp, -1, t.dtype
                ),
                (x,),
            ),
            "nll_loss_forward mean (converted host)": (
                lambda t: torch.ops.aten.nll_loss_forward.default(t, tg, None, 1, -100),
                (x,),
            ),
            "nll_loss_forward none (converted host)": (
                lambda t: torch.ops.aten.nll_loss_forward.default(t, tg, None, 0, -100),
                (x,),
            ),
            "nll_loss_backward mean (converted host)": (
                lambda t: torch.ops.aten.nll_loss_backward.default(
                    seed, t, tg, None, 1, -100, total_weight
                ),
                (x,),
            ),
            "embedding_dense_backward (memset + feature kernel)": (
                lambda t, i: torch.ops.aten.embedding_dense_backward(
                    t, i, 1000, -1, False
                ),
                (g, ids),
            ),
        }
        for name, (fn, args) in ops.items():
            with self.subTest(op=name):
                tape, eager = ht.trace(fn, args), _eager_work(fn, args)
                _assert_or_pending(
                    self,
                    name in _TWINS_PENDING,
                    lambda: self._assert_same_work(tape, eager),
                )
        _assert_fused_rms_norm_route(
            self, lambda t, g: F.rms_norm(t, (4096,), g, 1e-5), (x, w)
        )
        self.assertFalse(C._host_trace_tracing())

    def test_clamp_with_scalar_bounds(self):
        # clamp / clamp_min / clamp_max / relu with scalar bounds, in place
        # too; a nan bound fills; integral bounds that cannot change an
        # element are dropped and one that would change every element raises
        forms = {
            "clamp(-0.5, 0.5)": lambda x: x.clamp(-0.5, 0.5),
            "clamp(min=-0.5)": lambda x: x.clamp(min=-0.5),
            "clamp(max=0.5)": lambda x: x.clamp(max=0.5),
            "clamp_min(0.0)": lambda x: x.clamp_min(0.0),
            "clamp_max(0.0)": lambda x: x.clamp_max(0.0),
            "torch.clamp(x, 0, 1)": lambda x: torch.clamp(x, 0, 1),
            "relu": torch.relu,
            "F.relu": F.relu,
            "hardtanh": F.hardtanh,
            "hardtanh(-0.5, 0.5)": lambda x: F.hardtanh(x, -0.5, 0.5),
            "hardtanh_ on an allocation": lambda x: F.hardtanh_(x * 1.0),
            "clamp_ on an allocation": lambda x: (x * 1.0).clamp_(-0.5, 0.5),
            "relu_ on an allocation": lambda x: (x * 1.0).relu_(),
            "clamp(nan)": lambda x: x.clamp(min=float("nan")),
            "clamp of a transposed operand": lambda x: x.t().clamp(-0.5, 0.5),
        }
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            for name, fn in forms.items():
                with self.subTest(op=name, dtype=dtype):
                    x = self._values(64, 4096, dtype)
                    x[0, :8] = float("nan")
                    self._replays(
                        fn,
                        (x,),
                        [
                            (self._values(48, 3000, dtype, offset=8),),
                            (self._values(7, 1000, dtype),),
                        ],
                    )
        int_forms = {
            "clamp(-1, 1)": lambda x: x.clamp(-1, 1),
            "clamp_min(0)": lambda x: x.clamp_min(0),
            "clamp_max(2)": lambda x: x.clamp_max(2),
            "relu": torch.relu,
            "hardtanh(-1, 1)": lambda x: F.hardtanh(x, -1, 1),
        }
        for name, fn in int_forms.items():
            with self.subTest(op=name, dtype=torch.int64):
                self._replays(
                    fn,
                    (self._values(64, 4096, torch.int64),),
                    [
                        (self._values(48, 3000, torch.int64, offset=8),),
                        (self._values(7, 1000, torch.int64),),
                    ],
                )
        # in place on an input
        self._inplace_replays(
            lambda t: t.clamp_(-0.5, 0.5),
            lambda M, N: (self._values(M, N, torch.float32),),
            [(64, 4096), (48, 3000)],
        )
        self._inplace_replays(
            lambda t: t[:, 1:].relu_(),
            lambda M, N: (self._values(M, N, torch.bfloat16),),
            [(64, 4096), (7, 1000)],
        )
        # int32 bounds beyond the dtype's range: both dropped is a copy (in
        # place: nothing), a min above the range raises eager's text
        xi32 = self._values(8, 64, torch.int32)
        tape, _ = self._replays(
            lambda t: (t * 1).clamp_(-(2**40), 2**40),
            (xi32,),
            [(self._values(3, 8, torch.int32),)],
        )
        self.assertEqual(tape.num_launches, 1)
        # out of place with both bounds dropped: eager's result.copy_(self), a
        # memcpy record (copy_d2d) and no launch
        tape = ht.trace(lambda t: t.clamp(-(2**40), 2**40), (xi32,))
        self.assertEqual((tape.num_launches, len(tape.memcpys)), (0, 1))
        with self.assertRaisesRegex(
            RuntimeError, "Clamp min value .* is outside the representable range of Int"
        ):
            ht.trace(lambda t: t.clamp_min(2**40), (xi32,))
        with self.assertRaisesRegex(
            RuntimeError, "Clamp min value .* is outside the representable range of Int"
        ):
            ht.trace(lambda t: t.clamp_min(2**40), (xi32,), warm_up=False)
        # the meta's rules: a bound, no complex, no promotion of an integral
        # self by a floating bound; tensor bounds are another overload
        x = self._values(8, 64, torch.float32)
        xi = self._values(8, 64, torch.int64)
        with self.assertRaisesRegex(
            RuntimeError, "At least one of 'min' or 'max' must not be None"
        ):
            ht.trace(lambda t: torch.ops.aten.clamp.default(t), (x,), warm_up=False)
        with self.assertRaisesRegex(ht.Declined, "promotes it"):
            ht.trace(lambda t: t.clamp(-0.5, 0.5), (xi,))
        with self.assertRaisesRegex(ht.Declined, "aten.clamp.Tensor"):
            ht.trace(lambda t: t.clamp(min=t[0, 0]), (x,))
        with self.assertRaisesRegex(
            NotImplementedError, "Boolean inputs not supported for relu"
        ):
            ht.trace(torch.relu, (x > 0,))
        with self.assertRaisesRegex(
            ht.Declined, "Boolean inputs not supported for relu"
        ):
            ht.trace(torch.relu, (x > 0,), warm_up=False)
        # eager's clamp kernel family, contiguous and strided
        x = self._values(64, 4096, torch.bfloat16)
        for fn, entry_args in (
            (lambda: x.clamp(-0.5, 0.5), (x, -0.5, 0.5)),
            (lambda: x[:, ::2].clamp_min(0.0), (x[:, ::2], 0.0, None)),
        ):
            real_nodes, _ = self._capture(fn)
            ours, _ = self._capture(lambda: C._host_trace_ti_clamp(*entry_args))
            self.assertEqual(len(ours), 1)
            self.assertEqual(_family(ours[0][0]), _family(real_nodes[0][0]))
            self.assertEqual(ours[0][1:4], real_nodes[0][1:4])
        self.assertFalse(C._host_trace_tracing())

    def test_internal_overlap_of_the_destination_is_refused_like_the_real_op(self):
        # the real copy_ refuses a destination in which several elements share
        # one memory location; the sibling's build performs the same check
        # from the strides, so it never writes where eager raises
        message = "more than one element of the written-to tensor"
        x = torch.randn(8, 16, device="cuda", dtype=torch.bfloat16)
        src = torch.ones(8, 16, device="cuda", dtype=torch.bfloat16)
        with self.assertRaisesRegex(RuntimeError, message):
            x[:, :1].expand(8, 16).copy_(src)
        with self.assertRaisesRegex(RuntimeError, message):
            C._host_trace_ti_copy_(x[:, :1].expand(8, 16), src)

        def fn(x, y):
            x[:, :1].expand(8, 16).copy_(y)
            return x

        args = (x.clone(), src)
        # with warm-up eager refuses first; without it the sibling's own
        # check refuses inside the trace with the same text (Declined is a
        # RuntimeError, so this holds whether or not the recorder wraps it)
        with self.assertRaisesRegex(RuntimeError, message):
            ht.trace(fn, args)
        with self.assertRaisesRegex(RuntimeError, message):
            ht.trace(fn, args, warm_up=False)
        self.assertFalse(C._host_trace_tracing())
        # partial overlap between source and destination is what the real
        # copy_ allows (TooHard), and the sibling allows it the same way
        y = torch.randn(8, 16, device="cuda", dtype=torch.bfloat16)
        want, got = y.clone(), y.clone()
        want[:, 1:].copy_(want[:, :-1])
        C._host_trace_ti_copy_(got[:, 1:], got[:, :-1])
        self.assertTrue(torch.equal(bits(got), bits(want)))

    def test_alpha_is_checked_like_the_real_add(self):
        x, y = self._pair(8, 64)
        for alpha, message in (
            (True, "Boolean alpha only supported for Boolean results"),
            (2j, "argument alpha must not be a complex number"),
        ):
            with self.assertRaisesRegex(RuntimeError, message):
                torch.add(x, y, alpha=alpha)
            with self.assertRaisesRegex(RuntimeError, message):
                ht._TRACED_ENTRIES[torch.ops.aten.add.Tensor](x, y, alpha=alpha)
            with self.assertRaisesRegex(RuntimeError, message):
                ht.trace(
                    lambda a, b: torch.add(a, b, alpha=alpha), (x, y), warm_up=False
                )
        xi = torch.arange(64, device="cuda").view(8, 8)
        with self.assertRaisesRegex(
            RuntimeError, "must not be a floating point number"
        ):
            ht._TRACED_ENTRIES[torch.ops.aten.add.Tensor](xi, xi, alpha=2.0)
        self.assertFalse(C._host_trace_tracing())

    def test_registry_contract(self):
        # atan2 has no sibling: the op the registry tests stand an entry in for
        atan2 = torch.ops.aten.atan2.default
        x, y = self._pair(8, 64)
        with self.assertRaisesRegex(ValueError, "already has a traced entry"):
            ht.register_traced_entry(torch.ops.aten.add.Tensor, lambda *a, **k: None)
        # an entry that dispatches the op it stands for is a decline, not a
        # RecursionError, under the trace and in ordinary mode
        ht.register_traced_entry(atan2, lambda a, b: torch.atan2(a, b))
        try:
            with self.assertRaisesRegex(
                ht.Declined, "aten.atan2.default.*dispatched.*itself"
            ):
                ht.trace(torch.atan2, (x, y))
            self.assertFalse(C._host_trace_tracing())
            with self.assertRaisesRegex(ht.Declined, "dispatched"):
                with EntryMode():
                    torch.atan2(x, y)
            # a replacement entry runs its sibling (mul stands in: the registry
            # mechanics are what is tested here)
            ht.register_traced_entry(
                atan2, lambda a, b: C._host_trace_ti_mul(a, b), replace=True
            )
            tape = ht.trace(torch.atan2, (x, y))
            self.assertEqual(tape.num_launches, 1)
        finally:
            ht._TRACED_ENTRIES.pop(atan2, None)
        with self.assertRaisesRegex(ht.Declined, "aten.atan2.default"):
            ht.trace(torch.atan2, (x, y))

    def test_a_copy_on_write_input_stays_lazy(self):
        # inputs are read through the const accessor: a lazy clone stays
        # copy-on-write through an ordinary opted add, the trace's warm-up
        # and a replay; the output goes through the mutable form.
        x, y = self._pair(64, 4096)
        lazy = torch._lazy_clone(x)
        self.assertTrue(torch._C._is_cow_tensor(lazy))
        want = torch.add(x, y)
        out = torch.add(lazy, y)
        self.assertTrue(torch._C._is_cow_tensor(lazy))
        self.assertTrue(torch.equal(out, want))
        args = (lazy, y)
        tape = ht.trace(torch.add, args)
        variant = build(tape, torch.add, args)
        self.assertTrue(torch._C._is_cow_tensor(lazy))
        self.assertTrue(torch.equal(variant.replay(args)[0], want))
        self.assertTrue(torch._C._is_cow_tensor(lazy))

    def test_ordinary_ops_are_untouched(self):
        # the sibling never runs on the ordinary path: the real op's kernel
        x, y = self._pair(64, 4096)
        nodes, _ = self._capture(lambda: torch.add(x, y))
        self.assertEqual(len(nodes), 1)
        self.assertIn(b"CUDAFunctor_add".decode(), nodes[0][0])

    def test_copies_between_allocations_record_a_memcpy(self):
        # copy_'s `src != dst` between two allocations is decided by root
        # identity, never by the address hints (placeholders): a clone of an
        # intermediate, empty_like(a).copy_(a) and a clone chain record the
        # memcpy eager issues, name the pair as a root fact, and replay at
        # other shapes
        def clone_of_product(x):
            return (x * 2).clone()

        def clone_of_sum(x):
            y = x + 1
            return y.clone()

        def empty_like_copy(x):
            a = x * 2
            return torch.empty_like(a).copy_(a)

        def chain(x):
            y = x * 2
            for _ in range(4):
                y = y.clone()
            return y

        x = torch.randn(8, 256, device="cuda")
        news = [
            (torch.randn(13, 256, device="cuda"),),
            (torch.randn(8, 100, device="cuda"),),
        ]
        cases = (
            (clone_of_product, 1),
            (clone_of_sum, 1),
            (empty_like_copy, 1),
            (chain, 4),
        )
        for fn, memcpys in cases:
            with self.subTest(fn=fn.__name__):
                tape = two_hint.trace_twice(fn, (x,))
                self.assertEqual((tape.num_launches, tape.num_memcpys), (1, memcpys))
                pairs = [r for r in tape.root_facts if r[0] != "domain"]
                self.assertEqual(len(pairs), memcpys)
                variant = build(tape, fn, (x,))
                for args in ((x,), *news):
                    got = variant.replay(args)[0]
                    self.assertTrue(torch.equal(bits(got), bits(fn(*args))))

        # an empty view into the sibling launches nothing, as eager
        def empty_add(x):
            return x[:0] + x[:0]

        tape = ht.trace(empty_add, (x,))
        self.assertEqual(tape.num_launches, 0)
        variant = build(tape, empty_add, (x,))
        self.assertEqual(tuple(variant.replay((x,))[0].shape), (0, 256))

    def test_two_hints_name_a_value_taken_from_a_hint(self):
        # an entry that sizes its output from the batch's hint instead of its
        # symbol traces once without complaint; under two assignments the
        # allocation differs, and the difference names it (the registry is
        # this commit's, so the test lives here and not in commit 1's file)
        relu = torch.ops.aten.relu.default

        def entry(x):
            n = x.shape[0].node.hint
            return torch.empty((n, x.shape[1]), device=x.device, dtype=x.dtype)

        previous = ht._TRACED_ENTRIES.get(relu)
        ht.register_traced_entry(relu, entry, replace=True)
        try:
            x = torch.randn(64, 32, device="cuda")
            with self.assertRaisesRegex(
                two_hint.HintDependence, r"allocations\[0\]\.sizes\[0\]: 64 vs 65"
            ):
                two_hint.trace_twice(torch.relu, (x,))
        finally:
            if previous is None:
                del ht._TRACED_ENTRIES[relu]
            else:
                ht.register_traced_entry(relu, previous, replace=True)

    # ---- the generated siblings (torchgen over add's ufunc_inner_loop and
    # ti/siblings.yaml: HostTraceSibling_<op>.cu, DECISIONS A192): bitwise
    # the real op with the same kernel family, grid, block and launch image
    # over the layout matrix and the dtypes eager serves; traced through the
    # registry's generic entry (the functional, in-place and out= overloads)
    # and replayed at new shapes; holding the device work eager issues

    def _assert_same_launch(self, a, b, args, f_size, strided_ok=True):
        # two entries: bitwise outputs, one launch each of the same kernel
        # family, grid, block and image up to the output pointer (each
        # capture's own allocation) on the vectorized / unrolled paths
        want, got = a(*args), b(*args)
        torch.cuda.synchronize()
        self.assertEqual(
            (got.shape, got.stride(), got.dtype),
            (want.shape, want.stride(), want.dtype),
        )
        self.assertTrue(torch.equal(bits(got), bits(want)), "outputs differ bitwise")
        ((name_a, grid_a, block_a, smem_a, image_a),) = self._capture(lambda: a(*args))[
            0
        ]
        ((name_b, grid_b, block_b, smem_b, image_b),) = self._capture(lambda: b(*args))[
            0
        ]
        self.assertEqual(_family(name_b), _family(name_a))
        self.assertEqual((grid_b, block_b, smem_b), (grid_a, block_a, smem_a))
        self.assertEqual(len(image_b), len(image_a))
        if "elementwise_kernelILi128E" in name_a:
            return
        f_off = 4 if f_size <= 4 else 8
        data = (f_off + f_size + 7) // 8 * 8
        self.assertEqual(image_b[:data], image_a[:data])
        self.assertEqual(image_b[data + 8 :], image_a[data + 8 :])

    def _assert_same_work(self, tape, eager):
        ours = _tape_work(tape)
        self.assertEqual(len(ours), len(eager), (ours, eager))
        for (kind, name), (kind_e, name_e) in zip(ours, eager):
            self.assertEqual(kind, kind_e, (ours, eager))
            if kind != "kernel":
                continue
            # a converted host launches eager's own kernel (E36)
            self.assertEqual(name, name_e)

    def test_add_entry_is_the_generated_one(self):
        # the hand binding (bound_arg alpha: an int stays an int64 Scalar, a
        # symbolic number pins, A208) and the generated binding call the one
        # entry compiled into UfuncCUDA_add.cu (E36 stage 2): the same launch
        # and image for every form, the integer dtypes served as add_kernel
        # serves them; the registry keeps the hand entries (their alpha checks)
        # for the whole group
        gen, hand = _gen("add"), C._host_trace_ti_add
        for dtype in (torch.bfloat16, torch.float32, torch.float16, torch.float64):
            op = 8 if dtype is torch.float64 else 4
            for case, (x, y) in self._matrix(dtype, True).items():
                for args, f_size in (
                    ((x, y, 1.0), op),
                    ((x, y, 2.5), op),
                    ((x, 1.5, 1.0), 2 * op),
                    ((2.0, y, 0.5), 2 * op),
                ):
                    if not isinstance(args[0], torch.Tensor) and case != "contiguous":
                        continue
                    with self.subTest(
                        dtype=dtype, case=case, form=[type(v).__name__ for v in args]
                    ):
                        self._assert_same_launch(hand, gen, args, f_size)
        xi = torch.arange(64 * 64, device="cuda", dtype=torch.int32).view(64, 64)
        for entry in (gen, hand):
            self.assertTrue(torch.equal(entry(xi, xi, 3), torch.add(xi, xi, alpha=3)))
            self.assertTrue(torch.equal(entry(xi, 7, 1), xi + 7))
        xj = torch.arange(48 * 3000, device="cuda", dtype=torch.int32).view(48, 3000)
        tape, _, served = self._roundtrip(torch.add, (xi, xi), [(xj, xj)])
        self.assertEqual((tape.num_launches, served), (1, 1))
        self.assertIs(ht._TRACED_ENTRIES[aten.add.Tensor], _host_trace_ti._add)
        inplace = ht._TRACED_ENTRIES[aten.add_.Tensor]
        self.assertTrue(inplace.__qualname__.startswith("_inplace."), inplace)

    def _operands(self, dtype, case, ntensors, prep):
        # the matrix case's operands for an op of `ntensors` inputs, prepared
        # for its domain
        x, y = self._matrix(dtype, True)[case]
        if ntensors == 3:
            z = torch.randn(x.shape, device="cuda").to(dtype)
            if case == "broadcast":
                z = z[0]
            elif case == "slice":
                z = torch.randn(x.shape[0], 2 * x.shape[1], device="cuda").to(dtype)[
                    :, ::2
                ]
        if prep == "positive":
            x = x.abs() + 0.5
        if prep == "mask":
            y = y > 0
        if prep == "nonzero":
            y = y.abs() % 5 + 1
        return (x,) if ntensors == 1 else (x, y) if ntensors == 2 else (x, y, z)

    def _assert_generated_parity(self, real, entry, args, ntensors, f_size):
        # bitwise the real op, one launch of the same kernel family, grid,
        # block and shared memory, and the same image: eager's host launches
        # the generated functor the sibling launches (E36 stage 2), so the
        # functor bytes and the pointer array sit at the same offsets
        # (an empty functor's one byte is whatever eager's stack held: not compared)
        self._assert_parity(
            real, entry, args, ntensors, functor_bytes=f_size > 1, f_size=f_size
        )

    def test_generated_siblings_parity_with_the_real_op(self):
        # every generated op: bitwise the real op with the same kernel family,
        # grid, block and launch image (the functor bytes aside) over the layout
        # matrix, on every dtype eager's kernel serves
        for name, (real, entry, ntensors, dtypes, prep, f_size) in GENERATED.items():
            if isinstance(entry, str):
                entry = _gen(entry)
            for dtype in dtypes:
                cases = ("contiguous", "transposed", "broadcast", "slice", "misaligned")
                if ntensors == 1:
                    cases = (
                        "contiguous",
                        "transposed",
                        "slice",
                        "misaligned",
                        "batch1",
                    )
                for case in cases:
                    if ntensors == 3 and case == "misaligned":
                        continue
                    args = self._operands(dtype, case, ntensors, prep)
                    with self.subTest(op=name, dtype=dtype, case=case):
                        self._assert_generated_parity(
                            real, entry, args, ntensors, f_size(dtype)
                        )

    def test_generated_siblings_replay_at_new_shapes(self):
        # traced through the registry's generic entry, one launch, replayed at
        # two new shapes bitwise eager; the in-place and out= forms of a
        # generated op write their destination
        for name, (real, _, ntensors, dtypes, prep, _) in GENERATED.items():
            for dtype in (torch.bfloat16, torch.float32):
                if dtype not in dtypes:
                    continue
                shapes = [(64, 4096), (48, 3000), (7, 1000)]
                args = [
                    self._operands(dtype, "contiguous", ntensors, prep) for _ in shapes
                ]
                args = [
                    tuple(
                        torch.randn(shape, device="cuda").to(a.dtype)
                        if a.dtype is not torch.bool
                        else torch.randn(shape, device="cuda") > 0
                        for a in base
                    )
                    for shape, base in zip(shapes, args)
                ]
                if prep == "positive":
                    args = [(a[0].abs() + 0.5, *a[1:]) for a in args]
                if prep == "nonzero":
                    args = [(a[0], a[1].abs() % 5 + 1) for a in args]
                with self.subTest(op=name, dtype=dtype):
                    tape, _, served = self._roundtrip(real, args[0], args[1:])
                    self.assertEqual(tape.num_launches, 1)
                    self.assertEqual(served, 2)
        for fn in (
            lambda t: torch.maximum(t, t[0]),
            lambda t: torch.maximum(t.t(), t.t()),
            lambda t: torch.sigmoid(t, out=torch.empty_like(t)),
            lambda t: aten.gelu_backward.grad_input(
                t, t, grad_input=torch.empty_like(t)
            ),
            lambda t: torch.abs(t, out=torch.empty_like(t)),
        ):
            for dtype in (torch.bfloat16, torch.float32):
                x = self._pair(64, 4096, dtype=dtype)[0]
                with self.subTest(fn="out= / views", dtype=dtype):
                    tape, _, served = self._roundtrip(
                        fn,
                        (x,),
                        [
                            (self._pair(48, 3000, dtype=dtype)[0],),
                            (self._pair(3, 8, dtype=dtype)[0],),
                        ],
                    )
                    self.assertEqual(tape.num_launches, 1)
                    self.assertEqual(served, 2)

    def test_generated_in_place_forms_write_the_input(self):
        # sigmoid_, abs_, leaky_relu_, addcmul_ and lerp_.Scalar through the
        # generic entry: the trace writes its input, the replay writes the new
        # one, bitwise eager on a clone
        forms = {
            "sigmoid_": (lambda t: t.sigmoid_(), 1),
            "abs_": (lambda t: t.abs_(), 1),
            "leaky_relu_": (lambda t: F.leaky_relu_(t, 0.3), 1),
            "addcmul_": (lambda t, u, v: t.addcmul_(u, v, value=0.5), 3),
            "lerp_.Scalar": (lambda t, u: t.lerp_(u, 0.25), 2),
        }
        for name, (fn, n) in forms.items():
            for dtype in (torch.bfloat16, torch.float32):
                base = tuple(
                    torch.randn(64, 4096, device="cuda").to(dtype) for _ in range(n)
                )
                with self.subTest(op=name, dtype=dtype):
                    tape = ht.trace(fn, tuple(t.clone() for t in base))
                    self.assertEqual(tape.num_launches, 1)
                    variant = build(tape, fn, tuple(t.clone() for t in base))
                    for shape in ((48, 3000), (7, 1000)):
                        new = tuple(
                            torch.randn(shape, device="cuda").to(dtype)
                            for _ in range(n)
                        )
                        want = tuple(t.clone() for t in new)
                        fn(*want)
                        out = variant.try_replay(new)
                        torch.cuda.synchronize()
                        self.assertIsNotNone(out)
                        self.assertTrue(torch.equal(bits(new[0]), bits(want[0])))
                        self.assertEqual(out[0].data_ptr(), new[0].data_ptr())

    def test_generated_siblings_hold_the_device_work_eager_issues(self):
        # the tape's launches, in issue order, are the device work eager issues
        # for the same call (the profiler's device events): eager's kernel by
        # its full name (the fidelity check of DECISIONS A190 / A191 over the
        # generated family)
        x = torch.randn(64, 4096, device="cuda")
        y = torch.randn(64, 4096, device="cuda")
        m = x > 0
        ops = {
            name: (real, ntensors, prep)
            for name, (real, _, ntensors, _, prep, _) in GENERATED.items()
        }
        for name, (real, ntensors, prep) in ops.items():
            args = (x.abs() + 0.5,) if prep == "positive" else (x,)
            if ntensors >= 2:
                args = args + (m if prep == "mask" else y,)
            if ntensors == 3:
                args = args + (y * 0.5,)
            with self.subTest(op=name):
                # the generated entry is compiled into eager's translation
                # unit and launches eager's functor (E36 stage 2): literal
                # name equality, as for every converted host
                self._assert_same_work(ht.trace(real, args), _eager_work(real, args))

    def test_generated_maximum_and_minimum_scalar_operands(self):
        # a CPU scalar on either side (the symmetric AUnaryFunctor form) is
        # bitwise the real op incl. a nan scalar read on the host in the tensor
        # dtype and a NaN payload eager passes through unchanged (its scalar_t
        # lambda returns the operand's bits; an opmath round trip would
        # canonicalize it, DECISIONS A192)
        for name in ("maximum", "minimum"):
            real, gen = getattr(torch, name), _gen(name)
            for dtype in (torch.bfloat16, torch.float32, torch.float16, torch.int64):
                x = self._matrix(dtype, False)["contiguous"][0]
                if dtype is torch.float16:
                    x = x.clone()
                    x.view(torch.int16)[0, 0] = 0x7E01
                for s in (0.5, 2, -3, 1.0004, float("nan")):
                    if dtype is torch.int64 and isinstance(s, float):
                        continue
                    cpu = torch.tensor(s)
                    for real_args, ours in (((x, cpu), (x, s)), ((cpu, x), (s, x))):
                        with self.subTest(
                            op=name,
                            dtype=dtype,
                            scalar=s,
                            first=isinstance(ours[0], torch.Tensor),
                        ):
                            want = real(*real_args)
                            got = gen(*ours)
                            torch.cuda.synchronize()
                            self.assertTrue(torch.equal(bits(got), bits(want)))
            # the two-tensor path keeps the payload too
            x = self._matrix(torch.float16, True)["contiguous"][0].clone()
            x.view(torch.int16)[0, :4] = 0x7E01
            y = torch.full_like(x, 1.0)
            for args in ((x, y), (y, x)):
                want, got = real(*args), gen(*args)
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(bits(got), bits(want)))
                self.assertEqual(got.view(torch.int16)[0, 0].item(), 0x7E01)

    def test_generated_siblings_decline_and_raise_like_eager(self):
        x, y = self._pair(8, 64, dtype=torch.float32)
        xi = torch.arange(64, device="cuda").view(8, 8)
        # a dtype outside the sibling's set declines by name
        with self.assertRaisesRegex(ht.Declined, "sigmoid on Long"):
            ht.trace(torch.sigmoid, (xi,))
        with self.assertRaisesRegex(ht.Declined, "exp on Long"):
            ht.trace(torch.exp, (xi,))
        # a CPU scalar operand of an op eager launches through the plain
        # gpu_kernel (which asserts on one) declines by name
        with self.assertRaisesRegex(ht.Declined, "addcmul with a CPU scalar operand"):
            _gen("addcmul")(x, y, 2.0, 1)
        with self.assertRaisesRegex(
            ht.Declined, "gelu_backward with a CPU scalar operand"
        ):
            _gen("gelu_backward")(x, 2.0, "none")
        # the op's own errors: eager's texts, before any launch
        with self.assertRaisesRegex(
            RuntimeError, "approximate argument must be either none or tanh"
        ):
            ht.trace(
                lambda g, t: aten.gelu_backward(g, t, approximate="erf"),
                (x, y),
                warm_up=False,
            )
        with self.assertRaisesRegex(
            RuntimeError, "Mask should be Bool Scalar TypeFloat"
        ):
            ht.trace(
                lambda g, t: aten.native_dropout_backward(g, t, 2.0),
                (x, y),
                warm_up=False,
            )
        with self.assertRaisesRegex(
            ht.Declined, "native_dropout_backward with a Byte mask"
        ):
            _gen("native_dropout_backward")(x, (y > 0).to(torch.uint8), 2.0)
        # a 0-dim CPU tensor operand (not a wrapped number) is an implicit
        # CPU scalar input and declines by name (A167)
        with self.assertRaisesRegex(ht.Declined, "implicit CPU scalar input"):
            ht.trace(lambda t: torch.maximum(t, _CPU_TWO), (x,))
        # a promoted binary pair traces through the cast kernel (lerp.Scalar
        # too); a ternary op's promotion (lerp.Tensor) still declines
        self.assertEqual(
            ht.trace(torch.maximum, (x, y.to(torch.bfloat16))).num_launches, 1
        )
        with self.assertRaisesRegex(ht.Declined, "type promotion"):
            ht.trace(
                lambda a, b: torch.lerp(a, b, a * 0.5),
                (x, y.to(torch.bfloat16)),
                warm_up=False,
            )
        self.assertFalse(C._host_trace_tracing())
        # the table names every generated op and the overloads it serves
        table = {row[0]: row for row in C._host_trace_ti_gen_siblings()}
        self.assertEqual(table["sigmoid"][2], ["sigmoid_", "sigmoid.out"])
        self.assertEqual(table["lerp.Scalar"][2], ["lerp_.Scalar", "lerp.Scalar_out"])
        self.assertEqual(table["gelu_backward"][2], ["gelu_backward.grad_input"])
        self.assertEqual(table["mse_loss"][4], True)
        self.assertIs(
            ht._TRACED_ENTRIES[aten.mse_loss.default], _host_trace_ti._mse_loss
        )
        for op in (
            aten.sigmoid_.default,
            aten.sigmoid.out,
            aten.leaky_relu_.default,
            aten.addcmul.out,
            aten.maximum.out,
        ):
            self.assertIn(op, ht._TRACED_ENTRIES)

    def test_partial_overlap_between_operands_declines_and_is_guarded_like_eager(self):
        # eager's get_overlap_status runs on every call: two dense operands over
        # one storage whose byte intervals intersect, without being one tensor,
        # refuse (add_, copy_, out=, masked_fill_'s mask). The sibling evaluates
        # eager's test on the trace's values: a call eager refuses declines by
        # name at the trace, and a tape built on separate inputs keeps the two
        # apart with one address guard, so a replay on overlapping views
        # misses where eager raises instead of being served
        message = "some elements of the input tensor and the written-to tensor"

        def add_(a, b):
            a.add_(b)
            return a

        def copy_(a, b):
            a.copy_(b)
            return a

        def add_out(a, b, c):
            torch.add(a, b, out=c)
            return c

        def masked_fill_(a, m):
            a.masked_fill_(m, True)
            return a

        def rand(M, N, dtype=torch.float32):
            t = torch.randn(M, N, device="cuda")
            return t > 0 if dtype is torch.bool else t

        def transposed(b):
            # a dense, non-contiguous source: the copy kernel, not the memcpy
            return b.reshape(b.shape[1], b.shape[0]).t()

        # op: (a pair of separate operands, the same pair carved from one
        # storage `x` as overlapping views, the operands' dtype)
        ops = {
            add_: (lambda a, b: (a, b), lambda x: (x[:-1], x[1:]), torch.float32),
            copy_: (
                lambda a, b: (a, transposed(b)),
                lambda x: (x[:-1], transposed(x[1:])),
                torch.float32,
            ),
            add_out: (
                lambda a, b: (a, rand(*a.shape), b),
                lambda x: (x[1:], rand(*x[1:].shape), x[:-1]),
                torch.float32,
            ),
            masked_fill_: (lambda a, b: (a, b), lambda x: (x[:-1], x[1:]), torch.bool),
        }
        for fn, (separate, alias, dtype) in ops.items():

            def args(M=8, N=64):
                return separate(rand(M, N, dtype), rand(M, N, dtype))

            def overlapping(M=8, N=64):
                return alias(rand(M + 1, N, dtype))

            with self.subTest(op=fn.__name__):
                with self.assertRaisesRegex(RuntimeError, message):
                    fn(*overlapping())
                # the recorder alone (no warm-up) refuses by name, typed
                with self.assertRaisesRegex(ht.Declined, message):
                    ht.trace(fn, overlapping(), warm_up=False)
                with self.assertRaisesRegex(RuntimeError, message):
                    ht.trace(fn, overlapping())
                self.assertFalse(C._host_trace_tracing())
                base = args()
                tape = ht.trace(fn, base)
                variant = build(tape, fn, base)
                # the non-overlapping class serves, in either order in memory
                flat = rand(2 * 8 * 64, 1, dtype).view(-1)
                lo, hi = flat[: 8 * 64].view(8, 64), flat[8 * 64 :].view(8, 64)
                for new in (
                    args(),
                    args(5, 64),
                    args(8, 96),
                    separate(lo, hi),
                    separate(hi, lo),
                ):
                    want = fn(*[t.clone() for t in new])
                    (got,) = variant.replay(new)
                    self.assertTrue(torch.equal(got, want))
                # overlapping views miss (eager raises there)
                with self.assertRaises(ht.Miss):
                    variant.replay(overlapping())
                self.assertIsNone(variant.try_replay(overlapping()))
                self.assertIsNone(variant.try_replay(overlapping(5, 64)))
        # one storage passed twice is eager's Full case, which add_ allows: the
        # tape records the identity of the two inputs' addresses (a replay with
        # two tensors misses, one traced on two tensors misses on one)
        y = rand(8, 64)
        tape = ht.trace(add_, (y, y))
        variant = build(tape, add_, (y, y))
        z = rand(8, 64)
        want = add_(z.clone(), z.clone())
        (got,) = variant.replay((z, z))
        self.assertTrue(torch.equal(got, want))
        self.assertIsNone(variant.try_replay((rand(8, 64), rand(8, 64))))
        base = (rand(8, 64), rand(8, 64))
        variant = build(ht.trace(add_, base), add_, base)
        self.assertIsNone(variant.try_replay((z, z)))
        self.assertFalse(C._host_trace_tracing())

    def test_overlap_inside_one_operand_declines_typed(self):
        # two views of one input whose intervals intersect (eager's Partial
        # over one storage) decline at the trace with eager's text, a typed
        # Declined rather than the RuntimeError the build would raise running
        # eager; a pair eager allows (strided views, TooHard there) traces
        message = "some elements of the input tensor and the written-to tensor"
        cases = {
            "add_ of the transpose": lambda x: x.add_(x.t()),
            "flat copy_ shifted": lambda x: x.view(-1)[:-1].copy_(x.view(-1)[1:]),
            "flat add_ shifted": lambda x: x.view(-1)[:-1].add_(x.view(-1)[1:]),
            "out= over a shifted view": lambda x: torch.add(x[1:], 1.0, out=x[:-1]),
            "copy_ from a shifted view": lambda x: x[:-1].copy_(x[1:]),
            "square view add_ its transpose": lambda x: x[:, :8].add_(x[:, :8].t()),
        }
        for name, op in cases.items():
            with self.subTest(case=name):

                def fn(x):
                    op(x)
                    return x

                x = torch.randn(8, 8, device="cuda")
                with self.assertRaisesRegex(RuntimeError, message):
                    fn(x.clone())
                with self.assertRaisesRegex(ht.Declined, message):
                    ht.trace(fn, (x.clone(),), warm_up=False)
                with self.assertRaisesRegex(RuntimeError, message):
                    ht.trace(fn, (x.clone(),))
                self.assertFalse(C._host_trace_tracing())

        def halves(x):
            x[:, :8].add_(x[:, 8:])
            return x

        x = torch.randn(8, 16, device="cuda")
        tape = ht.trace(halves, (x.clone(),), warm_up=False)
        variant = build(tape, halves, (x.clone(),))
        for t in (x, torch.randn(5, 16, device="cuda")):
            (got,) = variant.replay((t.clone(),))
            self.assertTrue(torch.equal(got, halves(t.clone())))
        self.assertFalse(C._host_trace_tracing())

    def test_out_of_another_shape_declines_by_name(self):
        # TensorIterator resizes an out= whose shape is not the result's
        # (a deprecation warning when it had elements); a traced tensor keeps
        # its storage, so the recorder declines by name. With the warm-up on,
        # eager's resize happens to the call's own out= before the symbolic
        # run, and trace() declines there: the warm-up changed the argument's
        # metadata, and a tape traced after it would describe the resized call
        def fn(x, y, o):
            torch.add(x, y, out=o)
            return o

        def args():
            return (
                torch.randn(8, 64, device="cuda"),
                torch.randn(8, 64, device="cuda"),
                torch.randn(8, 32, device="cuda"),
            )

        with self.assertWarnsRegex(UserWarning, "resized"):
            self.assertEqual(fn(*args()).shape, (8, 64))
        with self.assertRaisesRegex(
            ht.Declined, r"out= of shape .* does not match the result's shape"
        ):
            ht.trace(fn, args(), warm_up=False)
        self.assertFalse(C._host_trace_tracing())
        warmed = args()
        with self.assertWarnsRegex(UserWarning, "resized"):
            with self.assertRaisesRegex(
                ht.Declined,
                r"the warm-up changed the metadata of arg2 .*eager resized it",
            ):
                ht.trace(fn, warmed)
        self.assertEqual(tuple(warmed[2].shape), (8, 64))
        self.assertFalse(C._host_trace_tracing())
        # an out= of the result's shape traces, builds and replays
        x, y, o = args()
        o = torch.empty(8, 64, device="cuda")
        variant = build(ht.trace(fn, (x, y, o)), fn, (x, y, o))
        (got,) = variant.replay((x, y, torch.empty(8, 64, device="cuda")))
        self.assertTrue(torch.equal(got, x + y))
        self.assertFalse(C._host_trace_tracing())

    def test_where_with_a_python_scalar_traces_through_scalar_tensor(self):
        # torch.where(mask, min_value, x) (Gemma-2's sliding-window mask): the
        # ScalarOther / ScalarSelf composites build their scalar operand with
        # scalar_tensor (an allocation and eager's fill_ kernel over one
        # element) and call where.self, whose entry launches eager's opaque
        # where kernel over WhereFunctor: eager's function handles, bitwise,
        # replayed at new shapes; a non-bool condition raises eager's text
        minv = torch.finfo(torch.bfloat16).min
        B, T = 4, 64

        def args(b, t, dtype):
            x = torch.randn(b, 1, 1, t, device="cuda").to(dtype)
            y = torch.randn(b, 1, 1, t, device="cuda").to(dtype)
            return torch.randn(t, device="cuda") > 0, x, y

        cases = {
            "where(mask, min, x)": (
                lambda m, x, y: torch.where(m, minv, x),
                torch.bfloat16,
            ),
            "where(mask, x, 0.0)": (
                lambda m, x, y: torch.where(m, x, 0.0),
                torch.float32,
            ),
            "where(mask, x, y)": (lambda m, x, y: torch.where(m, x, y), torch.bfloat16),
            "where(mask, 2, x) int": (
                lambda m, x, y: torch.where(m, 2, x),
                torch.int64,
            ),
        }
        for name, (fn, dtype) in cases.items():
            base = args(B, T, dtype)
            with self.subTest(case=name):
                eager = assert_eager_function_handles(self, fn, base)
                self.assertEqual(len(eager), 1 if name == "where(mask, x, y)" else 2)
                tape, _, served = self._roundtrip(
                    fn, base, [args(2, 40, dtype), args(3, 8, dtype)]
                )
                self.assertEqual(served, 2)
        # the entry beside the real op: strides and bits over the layout matrix
        for dtype in (torch.bfloat16, torch.float32, torch.int64):
            for case, (x, y) in self._matrix(dtype, True).items():
                if case == "scalar":
                    continue
                m = torch.randn(x.shape, device="cuda") > 0
                with self.subTest(dtype=dtype, case=case):
                    want = torch.where(m, x, y)
                    got = C._host_trace_ti_where(m, x, y)
                    torch.cuda.synchronize()
                    self._assert_bitwise(got, want, stride=True)
        m, x, y = args(B, T, torch.bfloat16)
        with self.assertRaisesRegex(
            RuntimeError, "where expected condition to be a boolean tensor"
        ):
            ht.trace(lambda c, a, b: torch.where(c, a, b), (m.float(), x, y))
        # scalar_tensor on the CPU (no device given) is not traced
        with self.assertRaisesRegex(ht.Declined, "scalar_tensor.default on cpu"):
            ht.trace(lambda a: a * torch.scalar_tensor(2.0).cuda(), (x,), warm_up=False)
        self.assertFalse(C._host_trace_tracing())

    def test_every_case_traces_the_same_program_under_other_hints(self):
        # the recorder never reads a hint: every trace this class makes, made
        # again under other hints, is the same program (host_trace_two_hint)
        two_hint.assert_family(self)


if __name__ == "__main__":
    run_tests()
