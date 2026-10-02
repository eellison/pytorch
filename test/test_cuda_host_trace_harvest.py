# Owner(s): ["module: cuda graphs"]

import dataclasses
import os
import subprocess
import sys
import threading
import unittest
from unittest import mock

import torch
import torch.utils._pytree as pytree
from torch._inductor import config as inductor_config
from torch.testing._internal.common_cuda import tf32_off, tf32_on
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TEST_CUDA,
    TestCase,
)


if TEST_CUDA:
    from cuda.bindings import driver

    import torch._inductor.inductor_prims  # noqa: F401 (prims.inductor_seeds)
    import torch.cuda._host_trace_harvest as harvest
    from torch._inductor.cudagraph_host_trace import _harvest_budget
    from torch.cuda._host_trace_harvest import _launch, HarvestProvider
    from torch.cuda._host_trace_opaque import library_state, OpaqueKernel, OpaqueKey
    from torch.cuda._host_trace_replay import HostTraceReplay

aten = torch.ops.aten
bf16 = torch.bfloat16


def _fresh(op, out):
    rets = [out] if len(op._schema.returns) == 1 else list(out)
    return [
        o
        for r, ret in zip(rets, op._schema.returns)
        if ret.alias_info is None
        for o in pytree.tree_leaves(r)
        if isinstance(o, torch.Tensor)
    ]


def _key(op, args, kwargs):
    # a functional, in-place or out= call: its operands are its tensor
    # leaves, then its fresh outputs
    leaves = pytree.tree_leaves((args, kwargs))
    operands = [t for t in leaves if isinstance(t, torch.Tensor)]
    operands += _fresh(op, op(*args, **kwargs))
    key = OpaqueKey(
        op,
        tuple(t.dtype for t in operands),
        tuple(tuple(t.shape) for t in operands),
        tuple(t.stride() for t in operands),
        tuple(t.data_ptr() % 256 for t in operands),
        tuple(x for x in leaves if not isinstance(x, torch.Tensor)),
        operands[-1].device.index,
        library_state(),
    )
    return key, operands


def _learn(provider, op, args, kwargs=None):
    kwargs = kwargs or {}
    if (why := provider.accepts(op, args, kwargs)) is not None:
        return None, why
    key, operands = _key(op, args, kwargs)
    binding = provider.learn(key, args, kwargs, operands)
    return binding, provider.refused.get(provider._normal(key))


def _replay(binding, operands):
    # the binding launched on these operands into a fresh output
    o = operands[-1]
    out = torch.empty_strided(o.shape, o.stride(), dtype=o.dtype, device=o.device)
    out.fill_(float("nan"))
    scratch = [
        torch.empty(n, dtype=torch.uint8, device="cuda") for n in binding.scratch
    ]
    addresses = [t.data_ptr() for t in operands[:-1]] + [out.data_ptr()]
    stream = torch.cuda.current_stream().cuda_stream
    _launch(binding, addresses, [t.data_ptr() for t in scratch], stream)
    return out


def _private_bytes():
    # bytes the caching allocator holds in private pools (a harvest's pool
    # keeps at most _POOL_KEEP bytes when the harvest is over)
    return sum(
        seg["total_size"]
        for seg in torch.cuda.memory_snapshot(include_traces=False)
        if tuple(seg["segment_pool_id"]) != (0, 0)
    )


def _names(binding):
    return tuple(
        driver.cuFuncGetName(n.function)[1].decode()[:48]
        for n in binding.nodes
        if isinstance(n, OpaqueKernel)
    )


@unittest.skipIf(not TEST_CUDA, "CUDA not available")
@instantiate_parametrized_tests
class TestHostTraceHarvest(TestCase):
    def assertBitwise(self, a, b):
        self.assertTrue(torch.equal(a.view(torch.uint8), b.view(torch.uint8)))

    def assertRebinds(self, provider, op, make, trials=2):
        # learned on one set of operands, replayed on fresh ones at other
        # addresses of the same alignment class
        binding, why = _learn(provider, op, *make())
        self.assertIsNotNone(binding, why)
        for _ in range(trials):
            args, kwargs = make()
            ref = op(*args, **kwargs)
            key, operands = _key(op, args, kwargs)
            self.assertIs(provider.bind(key), binding)
            self.assertBitwise(_replay(binding, operands), ref)
        return binding

    def test_mm_sweep(self):
        # nvjet's choice changes across M (at 73, 80 and 96 among others) and
        # small M takes split-K (nvjet into the workspace, then cublasLt's
        # splitKreduce); every M binds and verifies
        p = HarvestProvider()
        b = torch.randn(4096, 4096, device="cuda", dtype=bf16)
        chosen = set()
        for m in (1, 8, 72, 73, 80, 96, 256):
            a = torch.randn(m, 4096, device="cuda", dtype=bf16)
            binding, why = _learn(p, aten.mm.default, (a, b))
            self.assertIsNotNone(binding, f"M={m}: {why}")
            chosen.add(_names(binding))
        self.assertGreater(len(chosen), 3)
        self.assertEqual(p.refused, {})

    def _captures(self, provider, op, args, siblings=True):
        smeared = torch._C._cuda_hostTraceHarvestCapture
        with (
            mock.patch.object(harvest, "_SIBLINGS", siblings),
            mock.patch.object(torch._C, "_cuda_hostTraceHarvestCapture", side_effect=smeared) as calls,
        ):
            binding, why = _learn(provider, op, args)
        self.assertIsNotNone(binding, why)
        return binding, calls.call_count

    def test_sibling_one_capture(self):
        # a key launching a harvested key's kernels takes its slots, read
        # from one capture, and verifies; a full harvest is the pair's two
        b = torch.randn(768, 3072, device="cuda", dtype=bf16)
        a1, a2 = (torch.randn(m, 768, device="cuda", dtype=bf16) for m in (3072, 4096))
        bindings = []
        for siblings, captures in ((True, 1), (False, 2)):
            p = HarvestProvider()
            first, _ = self._captures(p, aten.mm.default, (a1, b))
            binding, n = self._captures(p, aten.mm.default, (a2, b), siblings)
            self.assertEqual(_names(binding), _names(first))
            self.assertEqual(n, captures)
            bindings.append(binding)
        self.assertEqual(bindings[0], bindings[1])
        a3 = torch.randn(4096, 768, device="cuda", dtype=bf16)
        key, operands = _key(aten.mm.default, (a3, b), {})
        self.assertBitwise(_replay(p.bind(key), operands), a3 @ b)

    def test_sibling_unverified(self):
        # _VERIFY_SIBLINGS off: the sibling binds from its one capture without
        # the check, to the binding the check passes
        b = torch.randn(768, 3072, device="cuda", dtype=bf16)
        a1, a2 = (torch.randn(m, 768, device="cuda", dtype=bf16) for m in (3072, 4096))
        bindings = []
        for verify in (True, False):
            p = HarvestProvider()
            self._captures(p, aten.mm.default, (a1, b))
            with mock.patch.object(harvest, "_VERIFY_SIBLINGS", verify):
                binding, n = self._captures(p, aten.mm.default, (a2, b))
            self.assertEqual(n, 1)
            bindings.append(binding)
        self.assertEqual(bindings[0], bindings[1])
        key, operands = _key(aten.mm.default, (a2, b), {})
        self.assertBitwise(_replay(p.bind(key), operands), a2 @ b)

    def test_sibling_miss_primes(self):
        # no harvested key launches the kernels: the one capture is the full
        # harvest's A (the pair's) or its priming capture (the four's)
        b = torch.randn(768, 3072, device="cuda", dtype=bf16)
        a1, a2 = (torch.randn(m, 768, device="cuda", dtype=bf16) for m in (2048, 8))
        for pair, captures in ((True, 2), (False, 5)):
            p = HarvestProvider()
            first, _ = self._captures(p, aten.mm.default, (a1, b))
            with mock.patch.object(harvest, "_PAIR", pair):
                binding, n = self._captures(p, aten.mm.default, (a2, b))
            self.assertNotEqual(_names(binding), _names(first))
            self.assertEqual(n, captures)
            self.assertEqual(p.refused, {})

    @parametrize("op", [aten.mm.default, aten.addmm.default, aten.bmm.default])
    def test_pair(self, op):
        # a full harvest from two captures binds as from the four, and
        # rebinds bitwise at other addresses
        def r(*shape):
            return torch.randn(*shape, device="cuda", dtype=bf16)

        def make():
            if op is aten.bmm.default:
                return (r(4, 1000, 768), r(4, 768, 2304)), {}
            args = (r(1000, 768), r(768, 2304))
            return ((r(2304), *args) if op is aten.addmm.default else args), {}

        bindings = []
        for pair, captures in ((True, 2), (False, 5)):
            with mock.patch.object(harvest, "_PAIR", pair):
                binding, n = self._captures(HarvestProvider(), op, make()[0])
            self.assertEqual(n, captures)
            bindings.append(binding)
        self.assertEqual(bindings[0], bindings[1])
        self.assertRebinds(HarvestProvider(), op, make)

    def test_pair_refused_harvests_in_four(self):
        # a refusal of the pair (a qword it cannot explain) is no refusal of
        # the key: the four captures harvest it
        full = HarvestProvider._harvest

        def ambiguous(self, *args, pair=False, **kwargs):
            if pair:
                raise harvest._Refused("ambiguous")
            return full(self, *args, **kwargs)

        p = HarvestProvider()
        b = torch.randn(768, 3072, device="cuda", dtype=bf16)
        a = torch.randn(1000, 768, device="cuda", dtype=bf16)
        with mock.patch.object(HarvestProvider, "_harvest", ambiguous):
            binding, n = self._captures(p, aten.mm.default, (a, b))
        self.assertEqual(n, 5)
        self.assertEqual(p.refused, {})
        key, operands = _key(aten.mm.default, (a, b), {})
        self.assertBitwise(_replay(p.bind(key), operands), a @ b)

    def test_fast_division_constant_in_a_segment(self):
        # a vocabulary projection's fast division magic and shift (0x3aaaaaaab
        # at M=24576) is a constant, though it may fall in a segment
        b = torch.randn(768, 30528, device="cuda", dtype=bf16)
        a = torch.randn(24576, 768, device="cuda", dtype=bf16)
        p = HarvestProvider()
        binding, why = _learn(p, aten.mm.default, (a, b))
        self.assertIsNotNone(binding, why)

    @parametrize(
        "m,n,k",
        [
            (8, 4096, 4096),
            (80, 4096, 4096),
            (256, 256, 8192),
            (1, 1024, 65536),
            (33, 520, 136),
        ],
    )
    @parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_mm_rebinds_bitwise(self, m, n, k, dtype):
        def make():
            a = torch.randn(m, k, device="cuda", dtype=dtype)
            return (a, torch.randn(k, n, device="cuda", dtype=dtype)), {}

        self.assertRebinds(HarvestProvider(), aten.mm.default, make)

    def test_mm_transposed_operands(self):
        def make():
            a = torch.randn(512, 96, device="cuda", dtype=bf16).t()
            return (a, torch.randn(256, 512, device="cuda", dtype=bf16).t()), {}

        self.assertRebinds(HarvestProvider(), aten.mm.default, make)

    @parametrize("offset", [1, 3, 4])
    def test_align_one_cutlass(self, offset):
        # an operand one element off a 16-byte boundary takes a cutlass 2.x
        # align1 kernel instead of nvjet: its pointers still sit at 8-byte
        # offsets, so it binds; `offset` 1 and 3 share alignment class 2
        def make():
            base = torch.randn(64 * 512 + 64 + offset, device="cuda", dtype=bf16)
            a = base[64 + offset :].view(64, 512)
            return (a, torch.randn(512, 512, device="cuda", dtype=bf16)), {}

        binding = self.assertRebinds(HarvestProvider(), aten.mm.default, make)
        self.assertTrue(all("cutlass" in n or "splitK" in n for n in _names(binding)))

    def test_alignment_class_shared(self):
        p = HarvestProvider()
        b = torch.randn(512, 512, device="cuda", dtype=bf16)
        base = torch.randn(64 * 512 + 8, device="cuda", dtype=bf16)
        binding, why = _learn(
            p, aten.mm.default, (base[1 : 1 + 64 * 512].view(64, 512), b)
        )
        self.assertIsNotNone(binding, why)
        # address % 256 differs (2 vs 6), the class (2) does not
        a = base[3 : 3 + 64 * 512].view(64, 512)
        key, _ = _key(aten.mm.default, (a, b), {})
        self.assertIs(p.bind(key), binding)
        self.assertEqual(p.harvests, 1)

    def test_workspace_cache_on(self):
        # ATen reads TORCH_CUBLAS_WORKSPACE_CACHE once: a fresh process. Each
        # capture's cached workspace is its scratch; none stays cached in the
        # harvest's pool, so eager mms after the harvest reuse its memory
        code = f"""
import sys, torch
sys.path.insert(0, {os.path.dirname(os.path.abspath(__file__))!r})
from test_cuda_host_trace_harvest import _learn, _private_bytes, _replay, aten
from torch.cuda._host_trace_harvest import _POOL_KEEP, HarvestProvider
p = HarvestProvider()
# split-K shapes: the call takes a workspace
a, b = (torch.randn(*s, device="cuda", dtype=torch.bfloat16) for s in ((8, 4096), (4096, 4096)))
refs = [torch.mm(a, b), torch.mm(a[:4], b)]
for x, want in ((a, refs[0]), (a[:4], refs[1])):
    binding, why = _learn(p, aten.mm.default, (x, b))
    assert binding is not None and binding.scratch, why
    assert torch.equal(_replay(binding, [x, b, want]), want)
assert _private_bytes() <= _POOL_KEEP
for s in (torch.cuda.current_stream(), torch.cuda.Stream()):
    with torch.cuda.stream(s):
        fill = [torch.full((1 << 20,), float("nan"), device="cuda") for _ in range(64)]
        assert all(torch.equal(torch.mm(a, b), refs[0]) for _ in range(4))
        del fill
print("ok", p.harvests, p.refused)
"""
        env = dict(os.environ, TORCH_CUBLAS_WORKSPACE_CACHE="1")
        out = subprocess.check_output([sys.executable, "-c", code], env=env, text=True)
        self.assertIn("ok 2 {}", out)

    def test_the_key_carries_the_operands_device(self):
        # a binding is keyed on its operands' device, not the current one
        p = HarvestProvider()
        a, b = (torch.randn(64, 64, device="cuda", dtype=bf16) for _ in range(2))
        binding, why = _learn(p, aten.mm.default, (a, b))
        self.assertIsNotNone(binding, why)
        key, _ = _key(aten.mm.default, (a, b), {})
        self.assertEqual(key.device, a.device.index)
        self.assertIs(p.bind(key), binding)
        self.assertIsNone(p.bind(dataclasses.replace(key, device=key.device + 1)))

    def test_settings_in_key(self):
        # a binding learned under one split-K setting does not serve another
        m = torch.backends.cuda.matmul
        backend = torch.backends.cuda.preferred_blas_library()
        reduction = m.allow_bf16_reduced_precision_reduction
        p = HarvestProvider()
        a = torch.randn(1, 4096, device="cuda", dtype=bf16)
        b = torch.randn(4096, 4096, device="cuda", dtype=bf16)
        try:
            torch.backends.cuda.preferred_blas_library("cublaslt")
            split, why = _learn(p, aten.mm.default, (a, b))
            self.assertIsNotNone(split, why)
            m.allow_bf16_reduced_precision_reduction = (False, False)
            key, _ = _key(aten.mm.default, (a, b), {})
            self.assertIsNone(p.bind(key))
            m.allow_bf16_reduced_precision_reduction = reduction
            self.assertIs(p.bind(dataclasses.replace(key, state=library_state())), split)
        finally:
            m.allow_bf16_reduced_precision_reduction = reduction
            torch.backends.cuda.preferred_blas_library(backend)
        self.assertEqual(p.harvests, 1)

    def test_scratch_only_when_referenced(self):
        p = HarvestProvider()
        b = torch.randn(4096, 4096, device="cuda", dtype=bf16)
        a = torch.randn(256, 4096, device="cuda", dtype=bf16)
        plain, why = _learn(p, aten.mm.default, (a, b))
        self.assertIsNotNone(plain, why)
        self.assertEqual(plain.scratch, ())
        split, why = _learn(p, aten.mm.default, (a[:1], b))
        self.assertIsNotNone(split, why)
        self.assertTrue(any("splitK" in n for n in _names(split)))
        self.assertNotEqual(split.scratch, ())

    def test_zero_operand(self):
        # verified on the call's operands, where the zero one hides a wrong
        # binding: it replays bitwise on random ones too
        def make():
            a = torch.randn(64, 128, device="cuda", dtype=bf16)
            return (a, torch.zeros(128, 96, device="cuda", dtype=bf16)), {}

        binding = self.assertRebinds(HarvestProvider(), aten.mm.default, make)
        a = torch.randn(64, 128, device="cuda", dtype=bf16)
        b = torch.randn(128, 96, device="cuda", dtype=bf16)
        self.assertBitwise(_replay(binding, [a, b, torch.mm(a, b)]), torch.mm(a, b))

    def test_other_thread_allocating(self):
        # the allocation count is the process's: another thread's
        # allocations refuse a harvest, but not the key for good
        stop = threading.Event()

        def churn():
            with torch.cuda.stream(torch.cuda.Stream()):
                while not stop.is_set():
                    torch.empty(1 << 16, device="cuda")

        p = HarvestProvider()
        args = [
            (
                torch.randn(n, 128, device="cuda", dtype=bf16),
                torch.randn(128, n, device="cuda", dtype=bf16),
            )
            for n in (64, 96, 128, 160)
        ]
        thread = threading.Thread(target=churn)
        thread.start()
        try:
            for x in args:
                _learn(p, aten.mm.default, x)
        finally:
            stop.set()
            thread.join()
        self.assertTrue(p.transient)
        for x in args:
            binding, why = _learn(p, aten.mm.default, x)
            self.assertIsNotNone(binding, why)
        self.assertEqual(p.refused, {})

    def test_private_pool_bytes_after_harvest(self):
        p = HarvestProvider(budget=lambda: 64)
        b = torch.randn(4096, 4096, device="cuda", dtype=bf16)
        reserved = []
        for m in range(1, 49):
            a = torch.randn(m, 4096, device="cuda", dtype=bf16)
            binding, why = _learn(p, aten.mm.default, (a, b))
            self.assertIsNotNone(binding, f"M={m}: {why}")
            self.assertLessEqual(_private_bytes(), harvest._POOL_KEEP)
            reserved.append(torch.cuda.memory_reserved())
        self.assertLessEqual(reserved[-1], reserved[len(reserved) // 2])

    def test_gemm_harvest_peak_is_eagers(self):
        # the captures' cuBLAS workspace is the arena's and verify launches on
        # the call's own tensors: past eager's peak, only the digests; the
        # pool doesn't grow
        p = HarvestProvider()
        b = torch.randn(4096, 4096, device="cuda", dtype=bf16)
        pool = []
        for m in (256, 1, 8, 1024, 2):
            a = torch.randn(m, 4096, device="cuda", dtype=bf16)
            torch.cuda.synchronize()
            base = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            key, operands = _key(aten.mm.default, (a, b), {})
            eager = torch.cuda.max_memory_allocated() - base
            base = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            binding = p.learn(key, (a, b), {}, operands)
            self.assertIsNotNone(binding, p.refused.get(p._normal(key)))
            self.assertLessEqual(torch.cuda.max_memory_allocated() - base, eager + (1 << 20))
            pool.append(_private_bytes())
        self.assertEqual(len(set(pool)), 1)

    @parametrize("wrong", ["output", "input"])
    def test_gemm_verifies_in_place(self, wrong):
        # a GEMM's binding is launched into the call's output: a wrong output
        # refuses the key and eager reruns into it; a written input raises
        a, b = (torch.randn(256, 256, device="cuda", dtype=bf16) for _ in range(2))
        key, operands = _key(aten.mm.default, (a, b), {})
        want = operands[-1].clone()

        def swapped(binding, addresses, *rest):
            addresses = list(addresses)
            if wrong == "output":
                addresses[0], addresses[1] = addresses[1], addresses[0]
            else:
                addresses[2] = addresses[0]
            return _launch(binding, addresses, *rest)

        p = HarvestProvider()
        with mock.patch.object(harvest, "_launch", swapped):
            if wrong == "input":
                with self.assertRaisesRegex(RuntimeError, "wrote input 0"):
                    p.learn(key, (a, b), {}, operands)
                return
            self.assertIsNone(p.learn(key, (a, b), {}, operands))
        self.assertIn("differs from eager at operand 2", p.refused[p._normal(key)])
        self.assertBitwise(operands[-1], want)

    @parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_addmm_bias(self, dtype):
        def make():
            bias = torch.randn(384, device="cuda", dtype=dtype)
            a = torch.randn(96, 256, device="cuda", dtype=dtype)
            return (bias, a, torch.randn(256, 384, device="cuda", dtype=dtype)), {}

        self.assertRebinds(HarvestProvider(), aten.addmm.default, make)

    @parametrize("op", ["mm", "addmm"])
    def test_out_padded_rows(self, op):
        # Inductor pads an output's row stride (a vocabulary projection's
        # odd N): the out= operand binds at its strides like an input
        def make():
            a = torch.randn(96, 256, device="cuda", dtype=bf16)
            b = torch.randn(256, 1001, device="cuda", dtype=bf16)
            out = torch.empty(96, 1008, device="cuda", dtype=bf16)[:, :1001]
            args = (a, b) if op == "mm" else (torch.randn(1001, device="cuda", dtype=bf16), a, b)
            return args, {"out": out}

        self.assertRebinds(HarvestProvider(), getattr(aten, op).out, make)

    @parametrize("tf32", [False, True])
    @parametrize("op", ["mm", "addmm"])
    def test_fp32_rebinds_bitwise(self, op, tf32):
        # bitwise against eager at the run's TF32 setting
        def make():
            a = torch.randn(96, 256, device="cuda")
            b = torch.randn(256, 384, device="cuda")
            args = (a, b) if op == "mm" else (torch.randn(384, device="cuda"), a, b)
            return args, {}

        with tf32_on(self) if tf32 else tf32_off():
            self.assertRebinds(HarvestProvider(), getattr(aten, op).default, make)

    def test_math_mode_per_call(self):
        # each call keeps the TF32 setting it was traced under, whatever the
        # setting at a replay's harvest
        def fn(x, w):
            torch.backends.cuda.matmul.allow_tf32 = False
            y = x @ w
            torch.backends.cuda.matmul.allow_tf32 = True
            z = y @ w
            torch.backends.cuda.matmul.allow_tf32 = False
            # a traced launch: a trace of only eager steps runs the call eagerly
            return y, z, x * w

        # a refused key's eager step too: the replay that refuses it runs its
        # opaque step eagerly, and the retrace records a plain eager step
        harvest_ = HarvestProvider._harvest

        def refusing(self, key, *args, **kwargs):
            if key.sizes[0] == (192, 192):
                raise harvest._Refused("refused by the test")
            return harvest_(self, key, *args, **kwargs)

        p = HarvestProvider()
        f = HostTraceReplay(fn, opaque=(p,))
        with tf32_off(), mock.patch.object(HarvestProvider, "_harvest", refusing):
            for n in (128, 128, 256, 256, 192, 192, 192, 128):
                a, b = (torch.randn(n, n, device="cuda") for _ in range(2))
                want, got = fn(a, b), f(a, b)
                self.assertEqual(got, want, atol=0, rtol=0)
        self.assertEqual(p.harvests, 6)
        self.assertEqual(len(p.refused), 2)
        # the first call runs eagerly; a new size's harvested keys bind by
        # relowering the tape; the refusal retraces
        self.assertEqual((f.traces, f.relowers, f.replays), (2, 1, 5))

    def test_addmm_declines(self):
        p = HarvestProvider()
        a = torch.randn(32, 64, device="cuda", dtype=bf16)
        b = torch.randn(64, 48, device="cuda", dtype=bf16)
        bias = torch.randn(48, device="cuda", dtype=bf16)
        op = aten.addmm.default
        self.assertIn(
            "not 1-D",
            p.accepts(op, (torch.randn(32, 48, device="cuda", dtype=bf16), a, b), {}),
        )
        self.assertIn("beta or alpha", p.accepts(op, (bias, a, b), {"beta": 2}))
        self.assertIsNone(p.accepts(op, (bias, a, b), {"alpha": 1.0}))
        self.assertIn("dtypes", p.accepts(op, (bias.float(), a, b), {}))
        self.assertIn("dtypes", p.accepts(aten.mm.default, (a.double(), b.double()), {}))
        self.assertIsNone(p.accepts(aten.mm.default, (a.float(), b.float()), {}))
        strided = torch.randn(96, device="cuda", dtype=bf16)[::2]
        binding, why = _learn(p, op, (strided, a, b))
        self.assertIsNone(binding)
        self.assertIn("stride", why)

    def test_bmm(self):
        def make():
            a = torch.randn(6, 72, 128, device="cuda", dtype=bf16)
            return (a, torch.randn(6, 128, 200, device="cuda", dtype=bf16)), {}

        self.assertRebinds(HarvestProvider(), aten.bmm.default, make)

    def test_bmm_transposed_batch(self):
        def make():
            a = torch.randn(64, 4, 96, device="cuda", dtype=torch.float16).transpose(
                0, 1
            )
            return (a, torch.randn(4, 96, 32, device="cuda", dtype=torch.float16)), {}

        self.assertRebinds(HarvestProvider(), aten.bmm.default, make)

    def test_refused_key_cached(self):
        p = HarvestProvider()
        a = torch.randn(32, 64, device="cuda", dtype=bf16)
        b = torch.randn(64, 48, device="cuda", dtype=bf16)
        strided = torch.randn(96, device="cuda", dtype=bf16)[::2]
        for _ in range(3):
            self.assertIsNone(_learn(p, aten.addmm.default, (strided, a, b))[0])
        self.assertEqual((p.harvests, len(p.refused)), (1, 1))

    def test_budget(self):
        p = HarvestProvider(budget=lambda: 3)
        b = torch.randn(64, 64, device="cuda", dtype=bf16)
        keys = []
        for m in (8, 16, 24, 32):
            a = torch.randn(m, 64, device="cuda", dtype=bf16)
            keys.append(_key(aten.mm.default, (a, b), {})[0])
            _learn(p, aten.mm.default, (a, b))
        # the fourth key is past the budget
        self.assertEqual(p.harvests, 3)
        self.assertEqual([p.bind(k) is not None for k in keys], [True, True, True, False])
        self.assertEqual(p.refused, {})

    def test_inductor_budget_is_the_knob_at_each_harvest(self):
        p = HarvestProvider(budget=_harvest_budget)
        b = torch.randn(64, 64, device="cuda", dtype=bf16)
        a8, a16 = (torch.randn(m, 64, device="cuda", dtype=bf16) for m in (8, 16))
        with inductor_config.patch({"triton.cudagraph_host_trace_harvest_budget": 1}):
            self.assertIsNotNone(_learn(p, aten.mm.default, (a8, b))[0])
            self.assertIsNone(_learn(p, aten.mm.default, (a16, b))[0])
        self.assertIsNotNone(_learn(p, aten.mm.default, (a16, b))[0])
        self.assertEqual((p.harvests, p.refused), (2, {}))


_SEEDED = {
    "bernoulli_p": (aten.bernoulli_.float, lambda x: ((x, 0.3), {})),
    "bernoulli": (aten.bernoulli.default, lambda x: ((x,), {})),
    "rand_like": (aten.rand_like.default, lambda x: ((x,), {})),
    "randn_like": (aten.randn_like.default, lambda x: ((x,), {})),
    "normal_": (aten.normal_.default, lambda x: ((x,), {})),
    "exponential_": (aten.exponential_.default, lambda x: ((x,), {})),
    "random_": (getattr(aten.random_, "from"), lambda x: ((x, 0, 10), {})),
    "log_normal_": (aten.log_normal_.default, lambda x: ((x,), {})),
    "cauchy_": (aten.cauchy_.default, lambda x: ((x,), {})),
    "geometric_": (aten.geometric_.default, lambda x: ((x, 0.3), {})),
}


@unittest.skipIf(not TEST_CUDA, "CUDA not available")
@instantiate_parametrized_tests
class TestHostTraceHarvestAttentionAndRng(TestCase):
    def assertReplays(self, op, make, rng=False, every=False):
        # learned on one call, launched on another's operands (fresh outputs
        # allocated here, filled with 0xFF), bitwise against the eager call
        # from the same generator offset
        p = HarvestProvider(("attention", "rng"), rng_ops=(torch.ops.prims.inductor_seeds.default,))
        args, kwargs = make()
        self.assertIsNone(p.accepts(op, args, kwargs))
        key, operands = _key(op, args, kwargs)
        gen = torch.cuda.default_generators[torch.cuda.current_device()]
        before = gen.get_offset()
        binding = p.learn(key, args, kwargs, operands)
        self.assertIsNotNone(binding, p.refused.get(p._normal(key)))
        self.assertEqual(binding.rng_increment > 0, rng)
        # the harvest leaves the generator where it was, and no pool behind
        self.assertEqual(gen.get_offset(), before)
        self.assertLessEqual(_private_bytes(), harvest._POOL_KEEP)
        args, kwargs = make()
        start = gen.get_offset()
        leaves = [t for t in pytree.tree_leaves((args, kwargs)) if isinstance(t, torch.Tensor)]
        placed = [t.clone() for t in leaves]
        ref = op(*args, **kwargs)
        end = gen.get_offset()
        self.assertEqual(end - start, binding.rng_increment)
        outs = [torch.empty_like(o).fill_(0) for o in _fresh(op, ref)]
        for o in outs:
            o.untyped_storage().fill_(255)
        seed = gen.initial_seed()
        seed = torch.tensor([seed - (seed >> 63 << 64)], device="cuda")
        offset = torch.tensor([start], device="cuda")
        scratch = [torch.empty(n, dtype=torch.uint8, device="cuda") for n in binding.scratch]
        addresses = [t.data_ptr() for t in (*placed, *outs)]
        stream = torch.cuda.current_stream().cuda_stream
        philox = (seed.data_ptr(), offset.data_ptr(), 0)
        _launch(binding, addresses, [t.data_ptr() for t in scratch], stream, philox)
        # the first output (the in-place or out= operand, else the first
        # fresh one) is what every caller reads
        pairs = [(placed[-1], leaves[-1])] if not outs else list(zip(outs, _fresh(op, ref)))
        for got, want in pairs if every else pairs[:1]:
            self.assertTrue(torch.equal(got.view(torch.uint8), want.view(torch.uint8)))
        return binding

    @parametrize("backend", ["cudnn", "flash", "efficient"])
    def test_attention_forward(self, backend):
        ops = {
            "cudnn": (aten._scaled_dot_product_cudnn_attention.default, (None, False, 0.0, True, False)),
            "flash": (aten._scaled_dot_product_flash_attention.default, (0.0, True, False)),
            "efficient": (aten._scaled_dot_product_efficient_attention.default, (None, False, 0.0, True)),
        }
        op, rest = ops[backend]

        def make():
            q, k, v = (torch.randn(2, 4, 128, 64, device="cuda", dtype=bf16) for _ in range(3))
            return (q, k, v, *rest), {}

        self.assertReplays(op, make)

    @parametrize("backend", ["cudnn", "flash"])
    def test_attention_dropout(self, backend):
        ops = {
            "cudnn": (aten._scaled_dot_product_cudnn_attention.default, (None, False, 0.125, True, False)),
            "flash": (aten._scaled_dot_product_flash_attention.default, (0.125, True, False)),
        }
        op, rest = ops[backend]

        def make():
            q, k, v = (torch.randn(2, 4, 128, 64, device="cuda", dtype=bf16) for _ in range(3))
            return (q, k, v, *rest), {}

        self.assertReplays(op, make, rng=True)

    @parametrize("dropout_p", [0.0, 0.125])
    def test_cudnn_attention_backward(self, dropout_p):
        # the backward reads the forward's saved seed and offset operands
        def make():
            q, k, v = (torch.randn(2, 4, 128, 64, device="cuda", dtype=bf16) for _ in range(3))
            fwd = aten._scaled_dot_product_cudnn_attention(q, k, v, None, True, dropout_p, True, False)
            o, lse, cq, ck, mq, mk, seed, offset = fwd[:8]
            return (torch.randn_like(o), q, k, v, o, lse, seed, offset, None, cq, ck, mq, mk, dropout_p, True), {}

        self.assertReplays(aten._scaled_dot_product_cudnn_attention_backward.default, make)

    @parametrize("backend,dropout_p", [("flash", 0.0), ("flash", 0.125), ("efficient", 0.0)])
    def test_attention_backward(self, backend, dropout_p):
        # temporaries the call frees coalesce into one block: one scratch
        # buffer as far as they reach
        def make():
            q, k, v = (torch.randn(2, 4, 128, 64, device="cuda", dtype=bf16) for _ in range(3))
            if backend == "flash":
                fwd = aten._scaled_dot_product_flash_attention(q, k, v, dropout_p, True, False)
                o, lse, cq, ck, mq, mk, seed, offset, _ = fwd
                return (torch.randn_like(o), q, k, v, o, lse, cq, ck, mq, mk, dropout_p, True, seed, offset), {}
            o, lse, seed, offset = aten._scaled_dot_product_efficient_attention(q, k, v, None, True, dropout_p, True)
            grads = (True, True, True, False)
            return (torch.randn_like(o), q, k, v, None, o, lse, seed, offset, dropout_p, grads, True), {}

        op = getattr(aten, f"_scaled_dot_product_{backend}_attention_backward").default
        binding = self.assertReplays(op, make, every=True)
        # eager's temporaries: flash's two copies, softmax_d and dq_accum
        # in two blocks, efficient's two in one
        self.assertEqual(len(binding.scratch), 2 if backend == "flash" else 1)

    @parametrize("which", ["cudnn", "cudnn_backward", "efficient"])
    def test_attention_sibling_one_capture(self, which):
        # another batch size launches the harvested key's kernels (cuDNN's by
        # name: each plan loads its own copy): one capture, the full binding
        def make(b):
            q, k, v = (torch.randn(b, 12, 256, 64, device="cuda", dtype=bf16) for _ in range(3))
            if which == "efficient":
                return aten._scaled_dot_product_efficient_attention.default, (q, k, v, None, False, 0.0, True)
            fwd = (q, k, v, None, True, 0.0, True, False)
            if which == "cudnn":
                return aten._scaled_dot_product_cudnn_attention.default, fwd
            o, lse, cq, ck, mq, mk, seed, offset = aten._scaled_dot_product_cudnn_attention(*fwd)[:8]
            bwd = (torch.randn_like(o), q, k, v, o, lse, seed, offset, None, cq, ck, mq, mk, 0.0, True)
            return aten._scaled_dot_product_cudnn_attention_backward.default, bwd

        smeared = torch._C._cuda_hostTraceHarvestCapture
        bindings = []
        for siblings, captures in ((True, 1), (False, 2)):
            p = HarvestProvider(("attention",))
            self.assertIsNotNone(_learn(p, *make(4))[0])
            with (
                mock.patch.object(harvest, "_SIBLINGS", siblings),
                mock.patch.object(torch._C, "_cuda_hostTraceHarvestCapture", side_effect=smeared) as calls,
            ):
                binding, why = _learn(p, *make(16))
            self.assertIsNotNone(binding, why)
            self.assertEqual(calls.call_count, captures)
            bindings.append(binding)
        self.assertEqual(bindings[0], bindings[1])

    @parametrize(
        "backend,kernel",
        [("efficient", "PyTorchMemEffAttention"), ("dropout", "fused_dropout_kernel")],
    )
    def test_unexplained_varying_parameter_refuses(self, backend, kernel):
        # without the harvest flag ATen leaves these kernels' unused parameter
        # bytes uninitialized (Params padding, TensorInfo's dims past the
        # tensor's): they vary between captures
        q, k, v = (torch.randn(2, 4, 128, 64, device="cuda", dtype=bf16) for _ in range(3))
        if backend == "efficient":
            o, lse, seed, offset = aten._scaled_dot_product_efficient_attention(q, k, v, None, True, 0.0, True)
            args = (torch.randn_like(o), q, k, v, None, o, lse, seed, offset, 0.0, (True, True, True, False), True)
            op = aten._scaled_dot_product_efficient_attention_backward.default
        else:
            args = (torch.randn(64, 256, device="cuda"), 0.1, True)
            op = aten.native_dropout.default
        p = HarvestProvider(("attention", "rng"))
        key, operands = _key(op, args, {})
        with mock.patch.object(harvest, "_ZERO_INIT", False):
            self.assertIsNone(p.learn(key, args, {}, operands))
        why = p.refused[p._normal(key)]
        self.assertIn("unexplained varying parameter", why)
        self.assertIn(kernel, why)
        self.assertIsNotNone(HarvestProvider(("attention", "rng")).learn(key, args, {}, operands))

    def test_blocks_out_of_address_order(self):
        # after a first harvest the allocator places flash's softmax_d and
        # dq_accum blocks in another address order in the later captures
        def make():
            q, k, v = (torch.randn(4, 16, 256, 64, device="cuda", dtype=bf16) for _ in range(3))
            fwd = aten._scaled_dot_product_flash_attention(q, k, v, 0.0, True, False)
            o, lse, cq, ck, mq, mk, seed, offset, _ = fwd
            return (torch.randn_like(o), q, k, v, o, lse, cq, ck, mq, mk, 0.0, True, seed, offset), {}

        for _ in range(3):
            self.assertReplays(aten._scaled_dot_product_flash_attention_backward.default, make, every=True)

    def test_flash_backward_dq_within_eager_spread(self):
        # at S=512 eager's own dq (fp32 atomics) differs run to run, dk and
        # dv don't; operands 8, 9 are dq and dk
        q, k, v = (torch.randn(4, 16, 512, 64, device="cuda", dtype=bf16) for _ in range(3))
        o, lse, cq, ck, mq, mk, seed, offset, _ = aten._scaled_dot_product_flash_attention(q, k, v, 0.0, True, False)
        args = (torch.randn_like(o), q, k, v, o, lse, cq, ck, mq, mk, 0.0, True, seed, offset)
        op = aten._scaled_dot_product_flash_attention_backward.default
        p = HarvestProvider(families=("attention",))
        key, operands = _key(op, args, {})
        binding = p.learn(key, args, {}, operands)
        self.assertIsNotNone(binding, p.refused.get(p._normal(key)))

        def swapped(b, addresses, *rest):
            addresses = list(addresses)
            addresses[8], addresses[9] = addresses[9], addresses[8]
            return _launch(b, addresses, *rest)

        p = HarvestProvider(families=("attention",))
        with mock.patch.object(harvest, "_launch", swapped):
            self.assertIsNone(p.learn(key, args, {}, operands))
        self.assertIn("differs from eager at operand 8", p.refused[p._normal(key)])

    def test_coalesced_under_recorded_history(self):
        # while the process records its own memory history the harvest reads
        # the pool's allocations from it, and leaves it on
        p, ref = (HarvestProvider(families=("attention",)) for _ in range(2))
        q, k, v = (torch.randn(2, 4, 128, 64, device="cuda", dtype=bf16) for _ in range(3))
        o, lse, seed, offset = aten._scaled_dot_product_efficient_attention(q, k, v, None, True, 0.0, True)
        args = (torch.randn_like(o), q, k, v, None, o, lse, seed, offset, 0.0, (True, True, True, False), True)
        op = aten._scaled_dot_product_efficient_attention_backward.default
        key, operands = _key(op, args, {})
        torch.cuda.memory._record_memory_history(context="all", stacks="all")
        try:
            binding = p.learn(key, args, {}, operands)
            self.assertIsNotNone(binding, p.refused.get(p._normal(key)))
            self.assertTrue(torch._C._cuda_isHistoryEnabled())
        finally:
            torch.cuda.memory._record_memory_history(enabled=None)
        self.assertEqual(binding.scratch, ref.learn(key, args, {}, operands).scratch)

    def test_inductor_seeds(self):
        # Inductor's seeds for its dropout and rand kernels
        def make():
            out = torch.empty(3, dtype=torch.int64, device="cuda")
            return (-(2**63), 2**63 - 1, [3]), {"out": out}

        self.assertReplays(aten.randint.low_out, make, rng=True)

    def test_inductor_seeds_prim(self):
        # the seeds under graphsafe RNG; its output reuses a temporary's bytes
        seeds = torch.ops.prims.inductor_seeds.default
        self.assertReplays(seeds, lambda: ((3, torch.device("cuda")), {}), rng=True)

    def test_uniform(self):
        self.assertReplays(aten.uniform_.default, lambda: ((torch.empty(1000, device="cuda"),), {}), rng=True)

    def test_other_generator_declined(self):
        g = torch.Generator("cuda")
        x = torch.empty(1000, device="cuda")
        why = HarvestProvider(families=("rng",)).accepts(aten.uniform_.default, (x,), {"generator": g})
        self.assertIn("generator", why)

    def test_native_dropout(self):
        def make():
            return (torch.randn(64, 256, device="cuda"), 0.1, True), {}

        self.assertReplays(aten.native_dropout.default, make, rng=True, every=True)

    @parametrize("case", list(_SEEDED))
    def test_seeded_op(self, case):
        # any ATen op tagged nondeterministic_seeded is in the family
        op, make = _SEEDED[case]
        self.assertReplays(op, lambda: make(torch.rand(64, 256, device="cuda")), rng=True)


def setUpModule():
    import torch.cuda._host_trace_capture as capture

    capture.raise_trace_disagreements = True
    torch.cuda._host_trace.raise_unexpected = True


_CONVS = {
    # (in channels, out channels, groups, stride, dilation, bias)
    "plain": (32, 64, 1, 1, 1, True),
    "no_bias": (32, 64, 1, 1, 1, False),
    "groups": (32, 64, 4, 1, 1, True),
    "depthwise": (32, 32, 32, 1, 1, False),
    "strided_dilated": (32, 64, 1, 2, 2, True),
}

# refusals of stale bytes that depend on what ran before (not on the smears):
# ATen's bias add and bias grad (TensorIterator structs, until they are zeroed)
_STALE_REFUSALS = ("elementwise_kernel", "reduce_kernel")


def _conv(case, dtype, channels_last, batch=4):
    cin, cout, groups, stride, dilation, bias = _CONVS[case]
    x = torch.randn(batch, cin, 20, 20, device="cuda", dtype=dtype)
    w = torch.randn(cout, cin // groups, 3, 3, device="cuda", dtype=dtype)
    if channels_last:
        x = x.to(memory_format=torch.channels_last)
        w = w.to(memory_format=torch.channels_last)
    b = torch.randn(cout, device="cuda", dtype=dtype) if bias else None
    return x, w, b, [stride] * 2, [dilation] * 2, [dilation] * 2, False, [0, 0], groups


def _conv_backward(case, dtype, channels_last, batch=4):
    x, w, b, *rest = _conv(case, dtype, channels_last, batch)
    grad = torch.randn_like(aten.convolution(x, w, b, *rest))
    bias_sizes = None if b is None else list(b.shape)
    return grad, x, w, bias_sizes, *rest, [True, True, b is not None]


@unittest.skipIf(not TEST_CUDA, "CUDA not available")
@instantiate_parametrized_tests
class TestHostTraceHarvestConv(TestCase):
    def assertReplays(self, op, make, tf32=False):
        # learned on one call, launched twice on fresh operands into outputs
        # filled with 0xFF, every output bitwise against eager's; eager is
        # deterministic here (dgrad's atomics are not otherwise)
        flags = torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=True, allow_tf32=tf32)
        with flags, tf32_on(self) if tf32 else tf32_off():
            p = HarvestProvider(("conv",))
            binding, why = _learn(p, op, make())
            self.assertLessEqual(_private_bytes(), harvest._POOL_KEEP)
            if binding is None and any(k in why for k in _STALE_REFUSALS):
                self.skipTest(f"stale bytes the harvest cannot prove dead: {why}")
            self.assertIsNotNone(binding, why)
            for _ in range(2):
                args = make()
                ref = _fresh(op, op(*args))
                leaves = [t for t in pytree.tree_leaves(args) if isinstance(t, torch.Tensor)]
                outs = [torch.empty_like(o) for o in ref]
                for o in outs:
                    o.untyped_storage().fill_(255)
                scratch = [torch.empty(n, dtype=torch.uint8, device="cuda") for n in binding.scratch]
                addresses = [t.data_ptr() for t in (*leaves, *outs)]
                stream = torch.cuda.current_stream().cuda_stream
                _launch(binding, addresses, [t.data_ptr() for t in scratch], stream)
                for got, want in zip(outs, ref):
                    self.assertEqual(got.stride(), want.stride())
                    self.assertTrue(torch.equal(got.contiguous().view(torch.uint8), want.contiguous().view(torch.uint8)))
        return binding

    @parametrize("channels_last", [False, True])
    @parametrize("dtype", ["fp16", "bf16", "fp32", "tf32"])
    @parametrize("case", list(_CONVS))
    def test_forward(self, case, dtype, channels_last):
        dt = {"fp16": torch.half, "bf16": bf16}.get(dtype, torch.float)
        make = lambda: _conv(case, dt, channels_last)  # noqa: E731
        self.assertReplays(aten.convolution.default, make, tf32=dtype == "tf32")

    @parametrize("channels_last", [False, True])
    @parametrize("dtype", ["fp16", "bf16", "fp32", "tf32"])
    @parametrize("case", list(_CONVS))
    def test_backward(self, case, dtype, channels_last):
        # the first capture after an uncaptured run launches another parameter
        # image (the harvest primes), and dgrad frees its workspace before it
        # allocates grad_weight (each output gets a hole of its size)
        dt = {"fp16": torch.half, "bf16": bf16}.get(dtype, torch.float)
        make = lambda: _conv_backward(case, dt, channels_last)  # noqa: E731
        self.assertReplays(aten.convolution_backward.default, make, tf32=dtype == "tf32")

    def test_pair(self):
        # harvested in full from the pair, after the priming capture
        smeared = torch._C._cuda_hostTraceHarvestCapture
        make = lambda: _conv_backward("plain", bf16, True)  # noqa: E731
        with mock.patch.object(torch._C, "_cuda_hostTraceHarvestCapture", side_effect=smeared) as calls:
            self.assertReplays(aten.convolution_backward.default, make)
        self.assertEqual(calls.call_count, 3)

    def test_batch_sizes_are_keys(self):
        # a dynamic batch is one key per size, each learned once
        p = HarvestProvider(("conv",))
        for batch in (1, 3, 8, 3):
            args = _conv("plain", bf16, True, batch)
            binding, why = _learn(p, aten.convolution.default, args)
            self.assertIsNotNone(binding, why)
        self.assertEqual(len(p.bindings), 3)

    def test_transposed(self):
        def make():
            x = torch.randn(4, 64, 10, 10, device="cuda", dtype=bf16)
            w = torch.randn(64, 32, 4, 4, device="cuda", dtype=bf16)
            return x, w, None, [2, 2], [1, 1], [1, 1], True, [0, 0], 1

        self.assertReplays(aten.convolution.default, make)

    def test_mixed_dtypes_declined(self):
        x, w, _, *rest = _conv("plain", bf16, True)
        b = torch.randn(64, device="cuda")
        why = HarvestProvider(("conv",)).accepts(aten.convolution.default, (x, w, b, *rest), {})
        self.assertIn("not all bf16, fp16 or fp32", why)


if __name__ == "__main__":
    run_tests()
