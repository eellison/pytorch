# Owner(s): ["module: cuda graphs"]

import collections
import dataclasses
import functools
import os
import subprocess
import sys
import tempfile
import threading
import time
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
    import torch.cuda._host_trace_memory as memory
    from torch.cuda._host_trace_harvest import _launch, HarvestProvider
    from torch.cuda._host_trace_opaque import library_state, OpaqueBinding, OpaqueKernel, OpaqueKey, OpaqueMemset, Slot
    from torch.cuda._host_trace_replay import _BACKGROUND, HostTraceReplay
    from torch.cuda._host_trace_tape import _cuda_address, _Witness, EagerCall, trace

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


def _backed_peak(f, args):
    # peak live bytes of a call backed by memory, from the allocator's trace:
    # the captures' blocks (address space only, never mapped) left out
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.memory._record_memory_history(enabled="all", context=None, stacks="python", max_entries=1_000_000)
    try:
        f(*args)
        torch.cuda.synchronize()
        trace = torch.cuda.memory._snapshot()["device_traces"][torch.cuda.current_device()]
    finally:
        torch.cuda.memory._record_memory_history(enabled=None)
    a = harvest._address_only()
    unmapped = [(r[0], r[0] + a._SPAN) for r in a.ranges.values()]
    live, cur, peak = {}, 0, 0
    for e in trace:
        if e["action"] == "alloc" and not any(lo <= e["addr"] < hi for lo, hi in unmapped):
            live[e["addr"]] = e["size"]
            cur += e["size"]
            peak = max(peak, cur)
        elif e["action"] == "free_completed" and e["addr"] in live:
            cur -= live.pop(e["addr"])
    return peak


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

    def test_bindings_share_the_parts_they_repeat(self):
        # keys that differ in M take the same kernels with the same slots and
        # parameter layouts: their bindings hold them once (_compact), and
        # each still replays bitwise
        p = HarvestProvider()
        b = torch.randn(768, 1024, device="cuda", dtype=bf16)
        bindings = [
            self.assertRebinds(p, aten.mm.default, lambda m=m: ((torch.randn(m, 768, device="cuda", dtype=bf16), b), {}))
            for m in (200, 201)
        ]
        pairs = [(x, y) for x, y in zip(*(binding.nodes for binding in bindings)) if isinstance(x, OpaqueKernel) and x.slots == y.slots]
        self.assertTrue(pairs)
        for x, y in pairs:
            self.assertIs(x.slots, y.slots)
            self.assertIs(x.params, y.params)

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
# any shapes: every cuBLAS binding holds the call's workspace
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

    @parametrize("fill", [True, False])
    def test_learns_and_replays_under_deterministic_algorithms(self, fill):
        # use_deterministic_algorithms(True) makes torch.empty fill its memory
        # with a kernel: the harvest's own allocations in its address-only pool
        # (reserved, never mapped) must launch none, else an illegal address (a
        # fresh process: the error is sticky). Each family's key learns and
        # replays bitwise; a user's empty output under the fill replays eager's
        # fill (NaN), bitwise as int views.
        code = f"""
import torch
from torch.cuda._host_trace_harvest import HarvestProvider
from torch.cuda._host_trace_replay import HostTraceReplay
torch.use_deterministic_algorithms(True)
torch.utils.deterministic.fill_uninitialized_memory = {fill}
x = torch.randn(8, 64, 28, 28, device="cuda", dtype=torch.bfloat16).to(memory_format=torch.channels_last)
w = torch.randn(128, 64, 3, 3, device="cuda", dtype=torch.bfloat16).to(memory_format=torch.channels_last)
a, b = torch.randn(512, 768, device="cuda"), torch.randn(768, 1024, device="cuda")
q, k, v = (torch.randn(2, 8, 256, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3))
cases = (
    ("conv", lambda x, w: torch.nn.functional.conv2d(x, w, padding=1), (x, w)),
    ("blas", torch.mm, (a, b)),
    ("attention", torch.nn.functional.scaled_dot_product_attention, (q, k, v)),
)
for family, f, args in cases:
    p = HarvestProvider((family,))
    r = HostTraceReplay(f, opaque=(p,))
    for _ in range(4):  # eager, trace, then two replays
        out = r(*args)
    torch.cuda.synchronize()
    assert torch.equal(out, f(*args)), family
    assert r.replays == 2 and p.harvests >= 1 and not p.refused, (family, r.replays, p.harvests, p.refused)
if {fill}:
    def g(y, a, b):
        return y + torch.empty(y.shape, device=y.device), torch.mm(a, b)
    y = torch.randn(1000, device="cuda")
    r = HostTraceReplay(g, opaque=(HarvestProvider(("blas",)),))
    for _ in range(4):
        got = r(y, a, b)
    want = g(y, a, b)
    assert all(torch.equal(u.view(torch.int32), e.view(torch.int32)) for u, e in zip(got, want))
    assert r.replays == 2, r.replays
print("ok")
"""
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertIn("ok", out.stdout, out.stderr[-3000:])

    def test_learns_beside_an_nccl_process_group(self):
        # the "nccl" backend registers every new allocator segment with its
        # communicators; the harvest's captures allocate address space with no
        # memory behind it, which NCCL cannot register (a fresh process: the
        # process group is global)
        code = f"""
import sys, torch, torch.distributed as dist
sys.path.insert(0, {os.path.dirname(os.path.abspath(__file__))!r})
from test_cuda_host_trace_harvest import _learn, _replay, aten
from torch.cuda._host_trace_harvest import HarvestProvider
dist.init_process_group("nccl", store=dist.HashStore(), rank=0, world_size=1, device_id=torch.device("cuda", 0))
p = HarvestProvider()
a, b = (torch.randn(*s, device="cuda") for s in ((64, 512), (512, 256)))
binding, why = _learn(p, aten.mm.default, (a, b))
assert binding is not None, why
want = torch.mm(a, b)
assert torch.equal(_replay(binding, [a, b, want]), want)
dist.destroy_process_group()
print("ok", p.harvests, p.refused)
"""
        out = subprocess.check_output([sys.executable, "-c", code], text=True)
        self.assertIn("ok 1 {}", out)

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

    def test_a_plan_that_frees_an_allocation_before_its_use_fails_at_build(self):
        # the build checks that every allocation a launch reaches (by a slot or its keyed site's
        # table: the workspace) is live there in the memory plan
        plan_memory = memory.plan_memory

        def early(lowered, mode="eager", **kwargs):
            plan = plan_memory(lowered, mode, **kwargs)
            steps = [dataclasses.replace(s, temporaries=tuple((k, seq, seq) for k, seq, _ in s.temporaries)) for s in plan.steps]
            return dataclasses.replace(plan, steps=tuple(steps))

        def fn(x):
            return (x * 2).relu() + 1

        # no harvest: the default provider's pool would outlive the test (the pool-bytes tests sum every private pool)
        r = HostTraceReplay(fn, opaque=())
        r(torch.randn(32, 64, device="cuda"))  # eager
        with mock.patch.object(memory, "plan_memory", early), self.assertRaisesRegex(AssertionError, "outside their lifetimes"):
            r(torch.randn(32, 64, device="cuda"))

    def test_every_cublas_binding_has_its_workspace(self):
        # eager allocates the call's workspace whether or not its kernel reads it, on any GPU
        p = HarvestProvider()
        b = torch.randn(4096, 4096, device="cuda", dtype=bf16)
        for m in (256, 8, 1):
            binding, why = _learn(p, aten.mm.default, (torch.randn(m, 4096, device="cuda", dtype=bf16), b))
            self.assertIsNotNone(binding, why)
            self.assertNotEqual(binding.scratch, ())

    def test_the_plan_weighs_a_workspace_at_zero_bytes(self):
        # eager allocates and frees it per call: its run's sites share one slot of the run buffer,
        # which no split or eager order is chosen to save
        def fn(a, b):
            return (a @ b).relu() @ b

        r = HostTraceReplay(fn)
        a, b = torch.randn(64, 4096, device="cuda", dtype=bf16), torch.randn(4096, 4096, device="cuda", dtype=bf16)
        for _ in range(3):
            self.assertEqual(r(a, b), fn(a, b))
        self.assertEqual(r.traces, 1)
        (v,) = r.variants
        lowered = v.captured.lowered
        scratch = [k for site in lowered.sites for _, k in site.scratch.values()]
        self.assertEqual(len(scratch), 2)
        self.assertTrue(all(lowered.program.values[lowered.allocations[k].nbytes] > 0 for k in scratch))
        self.assertEqual([memory._nbytes(lowered)[k] for k in scratch], [0, 0])

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

    @parametrize("history", ["off", "process"])
    def test_other_thread_allocating(self, history):
        # the pool's log holds its own allocations alone, whatever the
        # process's history: another thread's do not refuse a harvest
        stop = threading.Event()

        def churn():
            # on every stream of the pool torch.cuda.Stream() hands out
            streams = [torch.cuda.Stream() for _ in range(32)]
            while not stop.is_set():
                for s in streams:
                    with torch.cuda.stream(s):
                        torch.empty(1 << 16, device="cuda")

        p = HarvestProvider()
        args = [
            (
                torch.randn(n, 128, device="cuda", dtype=bf16),
                torch.randn(128, n, device="cuda", dtype=bf16),
            )
            for n in (64, 96, 128, 160)
        ]
        if history == "process":
            torch.cuda.memory._record_memory_history(max_entries=1 << 20)
        thread = threading.Thread(target=churn)
        thread.start()
        try:
            for x in args:
                _learn(p, aten.mm.default, x)
        finally:
            stop.set()
            thread.join()
            if history == "process":
                torch.cuda.memory._record_memory_history(None)
        self.assertEqual((len(p.bindings), p.transient, p.refused), (len(args), {}, {}))

    def test_concurrent_harvests(self):
        # two threads' harvests, each in its own provider, take turns at the
        # pool and each learns all its keys
        shapes = {0: (64, 96, 128, 160), 1: (72, 104, 136, 168)}
        args = {i: [(torch.randn(n, 128, device="cuda", dtype=bf16), torch.randn(128, n, device="cuda", dtype=bf16)) for n in ns] for i, ns in shapes.items()}
        providers = {i: HarvestProvider() for i in shapes}
        errors = []

        def learn(i):
            try:
                for x in args[i]:
                    _learn(providers[i], aten.mm.default, x)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=learn, args=(i,)) for i in shapes]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        for i, p in providers.items():
            self.assertEqual((len(p.bindings), p.transient, p.refused), (len(args[i]), {}, {}))
            for a, b in args[i]:
                key, operands = _key(aten.mm.default, (a, b), {})
                self.assertBitwise(_replay(p.bind(key), operands), torch.mm(a, b))

    @parametrize("when", ["before", "during"])
    def test_user_history_untouched(self, when):
        # the process's memory history, on before the harvest or turned on
        # during it, is on after it with its entries; off, the harvest never
        # turns it on
        p = HarvestProvider()
        a = torch.randn(96, 128, device="cuda", dtype=bf16)
        b = torch.randn(128, 96, device="cuda", dtype=bf16)
        real = torch._C._cuda_hostTraceHarvestCapture
        seen, mine = [], []

        def user():
            torch.cuda.memory._record_memory_history(context="all", stacks="python", max_entries=1 << 20)
            mine.append(torch.empty(1 << 16, device="cuda"))

        def capture(*args, **kwargs):
            seen.append(torch._C._cuda_isHistoryEnabled())
            if when == "during" and len(seen) == 1:
                user()
            return real(*args, **kwargs)

        if when == "before":
            user()
        try:
            with mock.patch.object(torch._C, "_cuda_hostTraceHarvestCapture", capture):
                binding, why = _learn(p, aten.mm.default, (a, b))
            self.assertIsNotNone(binding, why)
            self.assertEqual(seen[0], when == "before")
            self.assertTrue(torch._C._cuda_isHistoryEnabled())
            traces = torch.cuda.memory._snapshot()["device_traces"][a.device.index]
            entry = next(e for e in traces if e["action"] == "alloc" and e["addr"] == mine[0].data_ptr())
            self.assertTrue(entry["frames"])
        finally:
            torch.cuda.memory._record_memory_history(None)

    def test_private_pool_bytes_after_harvest(self):
        p = HarvestProvider()
        b = torch.randn(4096, 4096, device="cuda", dtype=bf16)
        reserved = []
        for m in range(1, 49):
            a = torch.randn(m, 4096, device="cuda", dtype=bf16)
            binding, why = _learn(p, aten.mm.default, (a, b))
            self.assertIsNotNone(binding, f"M={m}: {why}")
            self.assertLessEqual(_private_bytes(), harvest._POOL_KEEP)
            reserved.append(torch.cuda.memory_reserved())
        self.assertLessEqual(reserved[-1], reserved[len(reserved) // 2])

    def test_gemm_harvest_peak_is_two_workspaces(self):
        # each capture's cuBLAS workspace is its call's allocation, as eager's,
        # held to the harvest's end (two captures here), and verify launches
        # on the call's own tensors: past them, only the digests; the pool
        # doesn't grow
        ws = torch._C._cuda_getCublasWorkspaceSize()
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
            self.assertLess(eager, 2 * ws)
            self.assertLessEqual(torch.cuda.max_memory_allocated() - base, 2 * ws + (1 << 20))
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
                # the raised error's frames held verify's buffers past its
                # pool: their segment is cached, released here
                torch.cuda.empty_cache()
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

    @parametrize("shape", [(16, 768, 768), (8192, 768, 768), (8192, 3072, 768)])
    def test_tf32_addmm_earlier_workspace(self, shape):
        # GoogleFnet's tf32 nn.Linear: cuBLASLt's CUTLASS 3 bias kernel keeps
        # an earlier call's workspace address in its parameters (dead). Every
        # learn binds, and the trace replays it bitwise with no eager step
        m, k, n = shape

        def make():
            return (torch.randn(n, device="cuda"), torch.randn(m, k, device="cuda"), torch.randn(n, k, device="cuda").t()), {}

        with tf32_on(self):
            for _ in range(3):
                p = HarvestProvider()
                self.assertRebinds(p, aten.addmm.default, make)
                self.assertEqual(p.refused, {})
            p = HarvestProvider()
            r = HostTraceReplay(torch.addmm, opaque=(p,))
            for _ in range(5):
                args, _ = make()
                self.assertBitwise(r(*args), torch.addmm(*args))
        self.assertEqual(p.refused, {})
        self.assertGreater(r.replays, 0)
        self.assertEqual(r.eager, 1)
        steps = [rec for v in r.variants for _, rec in v.tape.launches if type(rec) is EagerCall]
        self.assertEqual(steps, [])

    def test_same_is_bitwise_by_a_chunked_digest(self):
        # verify's compare: bits in index order (+0/-0 and NaN payloads differ), any
        # strides, across chunk boundaries, with temporaries of chunks, not of the tensor
        a = torch.randn(3000, 1700, device="cuda")
        self.assertTrue(harvest._same(a, a.clone()))
        self.assertTrue(harvest._same(a.t(), a.t().contiguous()))
        b = a.clone()
        b[2999, 1699] = torch.nextafter(b[2999, 1699], torch.tensor(float("inf"), device="cuda"))
        self.assertFalse(harvest._same(a, b))
        z = torch.zeros(5, device="cuda")
        self.assertFalse(harvest._same(z, -z))
        n = torch.full((5,), float("nan"), device="cuda")
        m = n.clone()
        m.view(torch.int32)[2] += 1
        self.assertFalse(harvest._same(n, m))
        self.assertFalse(harvest._same(a, a.flip(0)))
        h = torch.randn(1 << 28, device="cuda", dtype=bf16)  # 512 MiB
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        self.assertTrue(harvest._same(h, h))
        self.assertLess(torch.cuda.max_memory_allocated() - base, 128 << 20)

    def test_the_fused_digest_is_the_chunked_one(self):
        # one kernel over a dense tensor in memory order (any order of its
        # dims), chunk copies for a view with gaps: the same digest as torch
        # ops on index-order chunks, for every dtype width and layout
        base = torch.randn(37, 129, 5, device="cuda") * 100
        cl = torch.randn(8, 16, 9, 7, device="cuda").to(memory_format=torch.channels_last)
        for dt in (torch.float32, bf16, torch.int64, torch.int16, torch.uint8, torch.bool, torch.complex64):
            t = base > 0 if dt == torch.bool else base.to(dt)
            for x in (t, t.transpose(0, 2), t.permute(2, 0, 1), t[:, ::3], t[0], t[:0], t.unsqueeze(1).expand(37, 2, 129, 5), cl.to(dt) if dt != torch.bool else cl > 0):
                with mock.patch.object(harvest, "has_triton", lambda: False):
                    want = harvest._digest(x)
                self.assertEqual(harvest._digest(x), want)

    def test_a_digest_holds_one_piece_at_a_time(self):
        # a broadcast or gapped view (no single kernel over its memory) is
        # digested through copies of _DIGEST_CHUNK-word pieces, one alive at a
        # time: not a copy of the tensor (a 384 MiB broadcast mask here)
        mask = torch.zeros(64, 1, 512, 512, device="cuda", dtype=bf16).expand(64, 12, 512, 512)
        gapped = torch.randn(4096, 8192, device="cuda")[:, ::2]
        for t in (mask, gapped):
            torch.cuda.synchronize()
            base = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            harvest._digest(t)
            torch.cuda.synchronize()
            self.assertLessEqual(torch.cuda.max_memory_allocated() - base, 2 * 8 * harvest._DIGEST_CHUNK)

    def test_a_stale_constant_is_found(self):
        # the first constant qword the predicate holds, skipping a slot's
        # neighbors (qs[1]) and loose bytes (qs[4]); the binding is not touched
        qs = [0xFF11_0000_0020, 0xFF11_0000_0030, 0x8_0000_0FFF, 0xFF11_0000_0040, 0xFF11_0000_0050]
        image = b"".join(q.to_bytes(8, "little") for q in qs)
        kernel = OpaqueKernel(0, (1, 1, 1), (1, 1, 1), 0, (), ((0, len(image)),), (image,), (Slot(0, 12, "operand", 0, 0),), ((0, 32, 8),))
        b = OpaqueBinding((kernel,), ())
        self.assertEqual(harvest._stale_constant(b, lambda q: q >> 32 == 0xFF11), (0, 0, 0, qs[0]))
        self.assertEqual(harvest._stale_constant(b, lambda q: q in qs[1:]), (0, 0, 24, qs[3]))
        self.assertIsNone(harvest._stale_constant(b, lambda q: q >> 32 == 0xFF12))
        self.assertEqual(b.nodes[0].images, (image,))
        self.assertFalse(hasattr(harvest, "_canonical"))

    def test_a_tensor_map_inside_a_params_struct_refuses_untouched(self):
        # CUTLASS 3 kernels (cuDNN's sm100 dgrad among them) take their TMA
        # descriptors inside one Params struct: a word of the embedded map that
        # reads as a stale address's high half is the map's encoding. Zeroed
        # (the old _canonical) the kernel's TMA traps (Flash-Next, 10-16): the
        # binding is refused with its bytes as captured, never rewritten
        from cuda.bindings import driver

        from torch.cuda._utils import _check_cuda_bindings

        x = torch.zeros(256, 256, device="cuda", dtype=torch.bfloat16)
        tmap = _check_cuda_bindings(
            driver.cuTensorMapEncodeTiled(
                driver.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, x.data_ptr(), [driver.cuuint64_t(256)] * 2, [driver.cuuint64_t(512)],
                [driver.cuuint32_t(64)] * 2, [driver.cuuint32_t(1)] * 2,
                driver.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
                driver.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,
                driver.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE,
                driver.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
            )
        )
        map_bytes = b"".join(int(w).to_bytes(8, "little") for w in tmap.opaque)
        words = [int.from_bytes(map_bytes[o : o + 8], "little") for o in range(0, 128, 8)]
        hi = next(w for w in words if w >> 32)
        params = bytes(64) + map_bytes + bytes(64)  # header, the map at 64 (alignas(64)), trailer
        kernel = OpaqueKernel(0, (1, 1, 1), (1, 1, 1), 0, (), ((0, len(params)),), (params,), (), ())
        b = OpaqueBinding((kernel,), ())
        x_, param, off, q = harvest._stale_constant(b, lambda q: q == hi)
        self.assertEqual((x_, param, q), (0, 0, hi))
        self.assertTrue(64 <= off < 192)
        self.assertEqual(b.nodes[0].images[0][64:192], map_bytes)

    def test_a_cudnn_binding_with_a_stale_constant_refuses_before_any_launch(self):
        # the refusal comes before verify: no launch of the binding (a stale
        # flag can hang the kernel, which verify cannot recover from)
        args = _conv("plain", bf16, False)
        launched = []
        with (
            mock.patch.object(harvest, "_stale_constant", lambda b, stale: (0, 0, 0, 0xFFFE_0000_0000)),
            mock.patch.object(harvest, "_launch", lambda *a, **k: launched.append(a)),
        ):
            binding, why = _learn(HarvestProvider(("conv",)), aten.convolution.default, args)
        self.assertIsNone(binding)
        self.assertIn(harvest._STALE_STRUCT, why)
        self.assertEqual(launched, [])

    def test_a_cublas_binding_is_never_checked_for_stale_constants(self):
        # the stale-constant refusal is for cuDNN's heap-built parameter
        # structs; a cuBLAS binding's constants are its own (an nvjet kernel's
        # tensor maps among them) and bind as captured
        calls = []
        real = harvest._stale_constant

        def spy(binding, *args, **kwargs):
            calls.append(binding)
            return real(binding, *args, **kwargs)

        p = HarvestProvider()
        a, b = (torch.randn(*s, device="cuda", dtype=torch.bfloat16) for s in ((64, 512), (512, 256)))
        with mock.patch.object(harvest, "_stale_constant", spy):
            binding, why = _learn(p, aten.mm.default, (a, b))
        self.assertIsNotNone(binding, why)
        self.assertEqual(calls, [])

    def test_scratch_blocks_share_as_eager_frees_them(self):
        def shared(uses, sizes):
            nodes = tuple(OpaqueMemset(Slot(0, 0, "scratch", j, 0), 0, 1, 1, 1, 1) for j in uses)
            b = harvest._shared(OpaqueBinding(nodes, sizes))
            return [n.dst.index for n in b.nodes], b.scratch

        # one after the other: one block of the larger's bytes
        self.assertEqual(shared((0, 0, 1, 1), (8, 16)), ([0, 0, 0, 0], (16,)))
        # live together: each its own; a later one joins the first free block
        self.assertEqual(shared((0, 1, 0), (8, 16)), ([0, 1, 0], (8, 16)))
        self.assertEqual(shared((0, 1, 0, 2), (8, 16, 4)), ([0, 1, 0, 0], (8, 16)))
        # a block no node reads keeps its own
        self.assertEqual(shared((0,), (8, 16)), ([0], (8, 16)))

    def test_a_learn_leaves_no_reserved_memory(self):
        # nothing of verify's outlives the learn: its buffers come from a pool
        # of their own, which goes back to the device when the harvest is
        # over, so reserved memory is where it was but for the capture pool's
        # cache (at most _POOL_KEEP bytes)
        a = torch.randn(2048, 4096, device="cuda", dtype=bf16)
        b = torch.randn(4096, 4096, device="cuda", dtype=bf16)
        key, operands = _key(aten.mm.default, (a, b), {})
        want = operands[-1].clone()
        torch.cuda.synchronize()
        before, private = torch.cuda.memory_reserved(), _private_bytes()
        p = HarvestProvider()
        binding = p.learn(key, (a, b), {}, operands)
        self.assertIsNotNone(binding, p.refused)
        torch.cuda.synchronize()
        self.assertEqual(torch.cuda.memory_reserved() - before, _private_bytes() - private)
        self.assertLessEqual(_private_bytes(), harvest._POOL_KEEP)
        self.assertBitwise(operands[-1], want)
        self.assertBitwise(_replay(binding, [a, b, want]), want)

    def test_verify_reuses_the_caching_allocators_blocks(self):
        # verify allocates from the caching allocator (the captures touch no
        # memory, so no pool of its own): with a free cached block that fits
        # its buffers, a learn maps no memory-backed segment
        x = torch.randn(64, 64, 128, 128, device="cuda", dtype=bf16)
        w = torch.randn(64, 64, 3, 3, device="cuda", dtype=bf16)
        g = torch.randn(64, 64, 128, 128, device="cuda", dtype=bf16)
        args = (g, x, w, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        key, operands = _key(aten.convolution_backward.default, args, {})
        torch.empty(1 << 30, dtype=torch.uint8, device="cuda")  # freed: a cached block
        p = HarvestProvider(("conv",))
        with torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=True):
            torch.cuda.synchronize()
            torch.cuda.memory._record_memory_history(enabled="all", context=None, stacks="python", max_entries=1_000_000)
            try:
                binding = p.learn(key, args, {}, operands)
                torch.cuda.synchronize()
                trace = torch.cuda.memory._snapshot()["device_traces"][torch.cuda.current_device()]
            finally:
                torch.cuda.memory._record_memory_history(enabled=None)
        why = next(iter(p.refused.values()), None)
        if binding is None and harvest._STALE_STRUCT in why:
            self.skipTest(why)
        self.assertIsNotNone(binding, p.refused)
        a = harvest._address_only()
        unmapped = [(r[0], r[0] + a._SPAN) for r in a.ranges.values()]
        mapped = [e for e in trace if e["action"] in ("segment_alloc", "segment_map") and not any(lo <= e["addr"] < hi for lo, hi in unmapped)]
        self.assertEqual(mapped, [])

    def test_a_retired_qword_into_a_released_pool_is_poisoned(self):
        # the earlier workspace address cuBLASLt's CUTLASS 3 struct keeps may
        # point into memory no allocation maps (a released pool). The address
        # shape suffices to retire it only because cuBLAS is closed source and
        # verify mirrors and poisons every retired qword, mapped or not: the
        # keys bind because the poisoned launches match eager
        def make():
            return (torch.randn(768, device="cuda"), torch.randn(16, 768, device="cuda"), torch.randn(768, 768, device="cuda").t()), {}

        with tf32_on(self):
            for _ in range(3):
                p = HarvestProvider()
                self.assertRebinds(p, aten.addmm.default, make)
                self.assertEqual(p.refused, {})
                self.assertLessEqual(_private_bytes(), harvest._POOL_KEEP)
            p = HarvestProvider()
            r = HostTraceReplay(torch.addmm, opaque=(p,))
            for _ in range(5):
                args, _ = make()
                self.assertBitwise(r(*args), torch.addmm(*args))
        self.assertEqual(p.refused, {})
        self.assertGreater(r.replays, 0)
        # a mirror of an address in no allocation (reserved, never mapped)
        size = 1 << 30
        va = int(harvest._check_cuda_bindings(driver.cuMemAddressReserve(size, 1 << 21, 0, 0)))
        try:
            q = va + (512 << 20)
            self.assertFalse(harvest._cuda_address(q))
            mirrors = harvest._Mirrors(torch.cuda.current_device(), [q])
            try:
                lo, hi = mirrors.windows[q]
                self.assertEqual((lo, hi - lo), (q - harvest._MIRROR_BELOW, harvest._MIRROR_BELOW + harvest._MIRROR_ABOVE))
                self.assertEqual((q + mirrors.shift[q]) & 0xFFFFFFFF, q & 0xFFFFFFFF)
                mirrors.fill(0xFF)
                self.assertFalse(mirrors.touched(0xFF))
            finally:
                mirrors.close()
        finally:
            harvest._check_cuda_bindings(driver.cuMemAddressFree(va, size))

    def test_a_live_qword_taken_as_retired_is_refused(self):
        # addmm's bias pointer forced to no role is retired (address-shaped,
        # in the harvest's unmapped operand arena): verify mirrors and poisons
        # it, the poisoned launch and the one owning it as scratch mismatch
        # eager, and the key refuses, with no fault
        def classified(role, qs):
            return False if isinstance(role, tuple) and role[:2] == ("operand", 0) else role

        args = (torch.randn(640, device="cuda"), torch.randn(24, 512, device="cuda"), torch.randn(640, 512, device="cuda").t())
        with tf32_on(self), mock.patch.object(harvest, "_classified", classified):
            p = HarvestProvider()
            binding, why = _learn(p, aten.addmm.default, args)
        self.assertIsNone(binding)
        self.assertIn("constant addresses no channel explains", why)
        torch.cuda.synchronize()
        self.assertEqual(torch.addmm(*args), torch.addmm(*args), atol=0, rtol=0)

    def learned_on(self, fn, *args):
        # the operand positions each trace-time learn read live tensors at,
        # and the calls' outputs
        lents = []
        learn_ = HostTraceReplay._learn

        def spy(replay, op, provider, call, key, real=None, **kwargs):
            lents.append(sorted(real or {}))
            return learn_(replay, op, provider, call, key, real, **kwargs)

        r = HostTraceReplay(fn)
        # the trace's own learn (bind_opaque's, as a trace without a warm-up has it)
        with mock.patch.object(HostTraceReplay, "_learn", spy), mock.patch.object(_Witness, "_learn", lambda *a: None):
            outs = [r(*args) for _ in range(3)]
        self.assertGreater(r.replays, 0)
        return lents, outs

    def test_a_trace_learns_at_the_warm_up_call(self):
        # the warm-up's call learns its key on its live inputs and output: the
        # trace binds it, and no learn of stand-ins runs
        calls = []
        learn_ = HostTraceReplay._learn

        def spy(replay, *args, **kwargs):
            calls.append(args[0])
            return learn_(replay, *args, **kwargs)

        x, w = torch.randn(256, 512, device="cuda"), torch.randn(512, 384, device="cuda")
        r = HostTraceReplay(torch.mm)
        with mock.patch.object(HostTraceReplay, "_learn", spy):
            for _ in range(3):
                self.assertBitwise(r(x, w), torch.mm(x, w))
        self.assertEqual(calls, [])
        self.assertGreater(r.replays, 0)
        self.assertEqual(r.eager, 1)

    def test_a_trace_learns_on_its_live_operands(self):
        # at the trace a GEMM's operands that are the call's arguments and its
        # output are live: the learn reads them, the op does not run again,
        # and the output, checked into in place, ends as eager's
        x, w = torch.randn(256, 512, device="cuda"), torch.randn(512, 384, device="cuda")
        lents, outs = self.learned_on(torch.mm, x, w)
        self.assertEqual(lents, [[0, 1, 2]])
        for out in outs:
            self.assertBitwise(out, torch.mm(x, w))

    def test_a_trace_lends_no_output_a_later_step_changes(self):
        # the output's live value is not the op's result: the learn makes its own
        def fn(x, w):
            y = torch.mm(x, w)
            return y.add_(1)

        x, w = torch.randn(256, 512, device="cuda"), torch.randn(512, 384, device="cuda")
        lents, outs = self.learned_on(fn, x, w)
        self.assertEqual(lents, [[0, 1]])
        for out in outs:
            self.assertBitwise(out, fn(x, w))

    @mock.patch("torch.cuda._host_trace_replay.oob_learn", False)
    @parametrize("eager_steps", [True, False])
    def test_math_mode_per_call(self, eager_steps):
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
        # opaque step eagerly, and the variant built at it (or the retrace)
        # records a plain eager step
        harvest_ = HarvestProvider._harvest

        def refusing(self, key, *args, **kwargs):
            if key.sizes[0] == (192, 192):
                raise harvest._Refused("refused by the test")
            return harvest_(self, key, *args, **kwargs)

        p = HarvestProvider()
        f = HostTraceReplay(fn, opaque=(p,))
        with tf32_off(), mock.patch.object(HarvestProvider, "_harvest", refusing), mock.patch("torch.cuda._host_trace_replay.refused_eager_steps", eager_steps):
            for n in (128, 128, 256, 256, 192, 192, 192, 128):
                a, b = (torch.randn(n, n, device="cuda") for _ in range(2))
                want, got = fn(a, b), f(a, b)
                self.assertEqual(got, want, atol=0, rtol=0)
        self.assertEqual(p.harvests, 6)
        self.assertEqual(len(p.refused), 2)
        # the first call runs eagerly; a new size's harvested keys bind by
        # relowering the tape; the refusal builds a variant (both mm keys
        # eager steps), or retraces
        want = (1, 1, 6, 1) if eager_steps else (2, 1, 5, 0)
        self.assertEqual((f.traces, f.relowers, f.replays, f.eager_sites), want)

    def test_the_default_harvests_blas(self):
        # by default an entry harvests cuBLAS's keys; opaque=() runs the mm as an eager step
        def fn(x, w):
            return torch.mm(x, w) + 1

        w = torch.randn(64, 64, device="cuda")
        for kw, harvests, eager_steps in (({}, [4], 0), ({"opaque": ()}, [], 1)):
            f = HostTraceReplay(fn, **kw)
            with tf32_off():
                for m in (8, 16, 32, 8, 16, 32, 24):
                    x = torch.randn(m, 64, device="cuda")
                    self.assertBitwise(f(x, w), fn(x, w))
            self.assertEqual([p.harvests for p in f.opaque], harvests)
            self.assertEqual((f.traces, f.replays, f.eager), (1, 5, 1))
            self.assertEqual([sum(not isinstance(s, range) for s in v.captured.lowered.steps) for v in f.variants], [eager_steps])

    def test_an_op_outside_the_harvested_set_names_no_reason(self):
        # the default provider declines cumsum without a reason of its own: the
        # step's reason is its CUDA host's alone
        def fn(x):
            return x.cumsum(-1) + 1

        tape = trace(fn, (torch.randn(6, 32, device="cuda"),), opaque=(HarvestProvider(),))
        steps = [rec for _, rec in tape.launches if type(rec) is EagerCall]
        self.assertEqual([rec.name for rec in steps], ["aten.cumsum.default"])
        self.assertTrue(steps[0].reason.startswith("aten.cumsum.default's CUDA host declines: "), steps[0].reason)

    def test_pointwise_padding_is_loose(self):
        # an ATen pointwise launch's functor padding, by its type as the launch
        # site reports it, holds whatever the call left there (here another
        # byte at each call): loose bytes, not state of the call
        from test_cuda_host_trace_aten import _functor_ops

        _, ops = _functor_ops()
        op = ops.add_padded.default
        binding, why = _learn(HarvestProvider(("extern",), extern_ops=(op,)), op, (torch.randn(4096, device="cuda"),))
        self.assertIsNone(why)
        (n,) = binding.nodes
        self.assertIn((1, 4, 4), n.loose)

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

    @parametrize("out_variant", [False, True])
    def test_mm_out_dtype_float32(self, out_variant):
        # a router GEMM: bf16 operands into a float32 output, as torch.mm(..., out_dtype=torch.float32) (mm.dtype) or
        # Inductor's out= call (mm.dtype_out); harvested through mm.dtype_out, rebinding bitwise
        def make():
            a = torch.randn(64, 4096, device="cuda", dtype=bf16)
            b = torch.randn(256, 4096, device="cuda", dtype=bf16).t()
            kwargs = {"out": torch.empty(64, 256, device="cuda", dtype=torch.float32)} if out_variant else {}
            return (a, b, torch.float32), kwargs

        p = HarvestProvider()
        self.assertRebinds(p, aten.mm.dtype_out if out_variant else aten.mm.dtype, make)
        self.assertEqual(p.refused, {})

    def test_mm_out_dtype_replays(self):
        # a traced torch.mm(..., out_dtype=torch.float32) replays bitwise with no eager step
        def fn(a, b):
            return torch.mm(a, b, out_dtype=torch.float32)

        p = HarvestProvider()
        r = HostTraceReplay(fn, opaque=(p,))
        b = torch.randn(4096, 256, device="cuda", dtype=bf16)
        for m in (8, 8, 8):
            a = torch.randn(m, 4096, device="cuda", dtype=bf16)
            self.assertBitwise(r(a, b), fn(a, b))
        self.assertEqual(p.refused, {})
        self.assertGreater(r.replays, 0)
        steps = [rec for v in r.variants for _, rec in v.tape.launches if type(rec) is EagerCall]
        self.assertEqual(steps, [])

    def test_mm_out_dtype_declines(self):
        # only bf16 or fp16 operands into float32 are harvested; any other pair is refused with a named reason
        p = HarvestProvider()
        a = torch.randn(32, 64, device="cuda", dtype=bf16)
        b = torch.randn(64, 48, device="cuda", dtype=bf16)
        self.assertIsNone(p.accepts(aten.mm.dtype, (a, b, torch.float32), {}))
        self.assertIsNone(p.accepts(aten.mm.dtype, (a.half(), b.half(), torch.float32), {}))
        for args in ((a.float(), b.float(), torch.float32), (a, b, torch.bfloat16), (a, b.half(), torch.float32)):
            self.assertIn("only bf16 or fp16 into float32", p.accepts(aten.mm.dtype, args, {}))
        out = torch.empty(32, 48, device="cuda", dtype=bf16)
        self.assertIn("only bf16 or fp16 into float32", p.accepts(aten.mm.dtype_out, (a, b, torch.bfloat16), {"out": out}))

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
        why = p.refused.get(p._normal(key))
        if binding is None and harvest._STALE_STRUCT in why:
            self.skipTest(why)
        self.assertIsNotNone(binding, why)
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

    def test_a_hanging_form_is_released_before_the_learn_returns(self):
        # efficient attention's backward spins on a semaphore in its
        # workspace with some forms: zeroed from the other stream after 20x
        # the op's eager time, the launch finishes, and the learn returns with
        # no kernel left on any stream, long before the (here 60 s) ceiling
        q, k, v = (torch.randn(16, 12, 512, 64, device="cuda", dtype=bf16) for _ in range(3))
        o, lse, seed, offset = aten._scaled_dot_product_efficient_attention(q, k, v, None, True, 0.0, False)
        args = (torch.randn_like(o), q, k, v, None, o, lse, seed, offset, 0.0, (True, True, True, False), False)
        op = aten._scaled_dot_product_efficient_attention_backward.default
        key, operands = _key(op, args, {})
        p = HarvestProvider(families=("attention",))
        start = time.monotonic()
        with mock.patch.object(harvest, "_HANG", 60.0):
            binding = p.learn(key, args, {}, operands)
        elapsed = time.monotonic() - start
        self.assertIsNotNone(binding, p.refused.get(p._normal(key)))
        streams = (torch.cuda.current_stream(), *harvest._DEVICES[torch.cuda.current_device()].streams)
        self.assertTrue(all(s.query() for s in streams))
        self.assertLess(elapsed, 10.0)

    def test_coalesced_under_recorded_history(self):
        # while the process records its own memory history the harvest learns
        # the same, and leaves it on
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

    @parametrize("forbid_learners", [False, True])
    @mock.patch("torch.cuda._host_trace_replay.oob_learn", False)
    def test_an_rng_key_learns_out_of_band(self, forbid_learners):
        # a new size of a bound dropout is learned at its call (with
        # forbid_learners the first at its trace too: no variant learns), and
        # each call draws what eager's would from where eager's would (the
        # learn's own eager run takes nothing from the generator)
        def fn(x):
            return torch.nn.functional.dropout(x * 2, 0.5, training=True) + 1

        gen = torch.cuda.default_generators[0]
        f = HostTraceReplay(fn, opaque=(HarvestProvider(("rng",)),), fullgraph=forbid_learners, forbid_learners=forbid_learners)
        learned = (0, 1, 2, 2, 3, 3, 3, 4) if forbid_learners else (0, 0, 0, 1, 2, 2, 2, 3)
        for n, k in zip((4096, 4096, 8192, 8192, 12288, 12288, 4096, 16384), learned):
            x = torch.randn(n, device="cuda")
            offset = gen.get_offset()
            got = f(x)
            taken = gen.get_offset() - offset
            gen.set_offset(offset)
            self.assertEqual(got, fn(x), atol=0, rtol=0)
            self.assertEqual(taken, gen.get_offset() - offset)
            self.assertEqual(f.learned, k)
        self.assertEqual((f.traces, f.replays, f.eager, f.learners, f.declines), (1, 6, 1, int(not forbid_learners), []))

    @parametrize("case", list(_SEEDED))
    def test_seeded_op(self, case):
        # any ATen op tagged nondeterministic_seeded is in the family
        op, make = _SEEDED[case]
        self.assertReplays(op, lambda: make(torch.rand(64, 256, device="cuda")), rng=True)


def setUpModule():
    from torch.cuda import _host_trace_hint_audit
    import torch.cuda._host_trace_capture as capture

    _host_trace_hint_audit.enable_for_tests()
    torch.cuda._host_trace.verify_redo_templates = True
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
_STALE_REFUSALS = ("elementwise_kernel", "reduce_kernel", harvest._STALE_STRUCT)


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

    def test_strided_dgrad_binds_no_harvest_high_half(self):
        # CUTLASS strided dgrad holds an int32 under a stale address's high
        # half (parameter byte 1424 here), which it reads as a flag: the
        # harvest's own high halves there (0xff11) hang it. Bound, it holds no
        # constant with one of the harvest's ranges' high halves; refused, it
        # says why
        make = lambda: _conv_backward("strided_dilated", bf16, True)  # noqa: E731
        p = HarvestProvider(("conv",))
        with torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=True), tf32_off():
            binding, why = _learn(p, aten.convolution_backward.default, make())
        if binding is None:
            self.assertIn(harvest._STALE_STRUCT, why)
            return
        highs = harvest._own_highs()
        self.assertTrue(highs)
        for n in binding.nodes:
            if not isinstance(n, OpaqueKernel):
                continue
            near = {(s.param, s.offset + d) for s in n.slots for d in range(-7, 8)}
            loose = {(p, o + i) for p, o, size in n.loose for i in range(size)}
            for param, image in enumerate(n.images):
                for i, q in enumerate(harvest._qwords(image, 0)):
                    if (param, 8 * i) in near or any((param, 8 * i + j) in loose for j in range(8)):
                        continue
                    self.assertFalse(q >> 32 in highs and not _cuda_address(q), f"parameter {param} byte {8 * i}: {q:#x}")

    @staticmethod
    def _wide_backward():
        # cuDNN's dgrad and wgrad each take a workspace at these sizes
        x, g = (torch.randn(32, 64, 128, 128, device="cuda", dtype=bf16) for _ in range(2))
        w = torch.randn(64, 64, 3, 3, device="cuda", dtype=bf16)
        return g, x, w, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]

    def test_dgrad_and_wgrad_workspaces_share_a_block(self):
        # eager frees dgrad's workspace before wgrad takes its own: the
        # binding holds one block of the larger's bytes, both at its start
        # (wgrad's kernel keeps a stale pointer into dgrad's)
        op = aten.convolution_backward.default
        with mock.patch.object(harvest, "_shared", lambda b: b):
            unshared = self.assertReplays(op, self._wide_backward)
        if len(unshared.scratch) != 2:
            self.skipTest(f"workspaces {unshared.scratch} on this cuDNN")
        self.assertEqual(self.assertReplays(op, self._wide_backward).scratch, (max(unshared.scratch),))

    def test_a_refused_shared_form_binds_unshared(self):
        # a shared form verify refuses (here its last two kernels swapped:
        # wgrad's reduction before the partials it reads) gives way to the
        # binding with each block its own
        shared = harvest._shared

        def broken(b):
            s = shared(b)
            return s if s is b else dataclasses.replace(s, nodes=(*s.nodes[:-2], s.nodes[-1], s.nodes[-2]))

        op = aten.convolution_backward.default
        with mock.patch.object(harvest, "_shared", broken):
            binding = self.assertReplays(op, self._wide_backward)
        if len(binding.scratch) == 1:
            self.skipTest("one workspace on this cuDNN")
        self.assertEqual(len(binding.scratch), 2)

    def test_pair(self):
        # harvested in full from the pair, after the priming capture
        smeared = torch._C._cuda_hostTraceHarvestCapture
        make = lambda: _conv_backward("plain", bf16, True)  # noqa: E731
        with mock.patch.object(torch._C, "_cuda_hostTraceHarvestCapture", side_effect=smeared) as calls:
            self.assertReplays(aten.convolution_backward.default, make)
        self.assertEqual(calls.call_count, 3)

    def test_a_batch_sibling_takes_one_capture(self):
        # a new batch whose plan launches a harvested batch's kernels is its
        # sibling: one capture, read with the harvested key's slots, checked,
        # and bitwise on fresh operands
        op = aten.convolution.default
        smeared = torch._C._cuda_hostTraceHarvestCapture
        with torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=True), tf32_off():
            p = HarvestProvider(("conv",))
            bindings, counts = [], []
            for batch in (4, 6):
                with mock.patch.object(torch._C, "_cuda_hostTraceHarvestCapture", side_effect=smeared) as calls:
                    binding, why = _learn(p, op, _conv("plain", bf16, False, batch))
                self.assertIsNotNone(binding, why)
                bindings.append(binding)
                counts.append(calls.call_count)
            if _names(bindings[1]) != _names(bindings[0]):
                self.skipTest("cuDNN picks other kernels at batch 6")
            self.assertEqual(counts[1], 1)
            args = _conv("plain", bf16, False, 6)
            key, operands = _key(op, args, {})
            got, want = _replay(p.bind(key), operands), op(*args)
            self.assertTrue(torch.equal(got.contiguous().view(torch.uint8), want.contiguous().view(torch.uint8)))

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


# SGLang-style custom ops (Library.define + a CUDA impl + a fake): q and k are
# views of one qkv storage, written in place
_extern_lib = torch.library.Library("ht_extern", "DEF")
_extern_lib.define("scale_qk(Tensor(a!) q, Tensor(b!) k, Tensor w) -> ()")
_extern_lib.define("rope_qk(Tensor(a!) q, Tensor(b!) k, Tensor pos, Tensor table) -> ()")
_extern_lib.define("store_kv(Tensor k, Tensor v, Tensor(a!) k_cache, Tensor(b!) v_cache, Tensor loc) -> ()")
_extern_lib.define("double_into(Tensor x, Tensor(a!) out) -> ()")
_extern_lib.define("store_paged(Tensor k, Tensor v, Tensor(a!) k_cache, Tensor(b!) v_cache, Tensor slot) -> ()")
_extern_lib.define("to_fp8(Tensor x, Tensor(a!) out) -> ()")
_extern_lib.define("copy_twice(Tensor x, Tensor(a!) out) -> ()")


def _scale_qk(q, k, w):
    q.mul_(w)
    k.mul_(w)


def _rope_qk(q, k, pos, table):
    rows = table.index_select(0, pos)
    q.add_(rows)
    k.sub_(rows)


def _store_kv(k, v, k_cache, v_cache, loc):
    k_cache.index_copy_(0, loc, k)
    v_cache.index_copy_(0, loc, v)


def _store_paged(k, v, k_cache, v_cache, slot):
    # vLLM reshape_and_cache_flash: a slot is a page (dim 0) and a row in it (dim 1)
    page, row = slot // k_cache.shape[1], slot % k_cache.shape[1]
    k_cache[page, row] = k
    v_cache[page, row] = v


def _double_into(x, out):
    torch.mul(x, 2, out=out)


_extern_lib.impl("double_into", _double_into, "CUDA")


def _to_fp8(x, out):
    # vLLM's static_scaled_fp8_quant: a float8 output written, not read
    out.copy_(x)


_extern_lib.impl("to_fp8", _to_fp8, "CUDA")
torch.library.register_fake("ht_extern::to_fp8", lambda x, out: None, lib=_extern_lib)
def _copy_twice(x, out):
    # a copy_ into out is a memcpy node, which an extern binding cannot hold: its key refuses at any size
    out.copy_(x * 2)


_extern_lib.impl("copy_twice", _copy_twice, "CUDA")
torch.library.register_fake("ht_extern::copy_twice", lambda x, out: None, lib=_extern_lib)
torch.library.register_fake("ht_extern::double_into", lambda x, out: None, lib=_extern_lib)
_extern_lib.impl("scale_qk", _scale_qk, "CUDA")
_extern_lib.impl("store_kv", _store_kv, "CUDA")
torch.library.register_fake("ht_extern::store_kv", lambda k, v, k_cache, v_cache, loc: None, lib=_extern_lib)
_extern_lib.impl("rope_qk", _rope_qk, "CUDA")
_extern_lib.impl("store_paged", _store_paged, "CUDA")
torch.library.register_fake("ht_extern::store_paged", lambda k, v, k_cache, v_cache, slot: None, lib=_extern_lib)

# a library op whose Python CUDA kernel runs a harvested GEMM (a keyed site inside the op) and then branches on a size
# (the op's own guard, so a selector over its launches): Flash-Next's attention op around a harvested extern, in small
_nested_lib = torch.library.Library("ht_nested", "DEF")
_nested_lib.define("mm_then_branch(Tensor x, Tensor w) -> Tensor")


def _mm_then_branch(x, w):
    y = torch.mm(x, w)
    return y * 2 if x.shape[0] > 16 else y + 1


_nested_lib.impl("mm_then_branch", _mm_then_branch, "CUDA")
torch.library.register_fake("ht_nested::mm_then_branch", lambda x, w: x.new_empty(x.shape[0], w.shape[1]), lib=_nested_lib)
_nested_lib.define("mm_by_branch(Tensor x, Tensor w) -> Tensor")


def _mm_by_branch(x, w):
    # the GEMM's operand is another view on each side of the op's own guard
    return torch.mm(x, w if x.shape[0] > 16 else w.t()) + 1


_nested_lib.impl("mm_by_branch", _mm_by_branch, "CUDA")
torch.library.register_fake("ht_nested::mm_by_branch", lambda x, w: x.new_empty(x.shape[0], w.shape[1]), lib=_nested_lib)
torch.library.register_fake("ht_extern::scale_qk", lambda q, k, w: None, lib=_extern_lib)
torch.library.register_fake("ht_extern::rope_qk", lambda q, k, pos, table: None, lib=_extern_lib)


@instantiate_parametrized_tests
@unittest.skipIf(not TEST_CUDA, "CUDA not available")
class TestExternHarvest(TestCase):
    @staticmethod
    def scale(qkv, w, at):
        d = w.shape[0]
        torch.ops.ht_extern.scale_qk(qkv[:, :d], qkv[:, at : at + d], w)
        return qkv * 2

    @staticmethod
    def rope(qkv, pos, table):
        d = table.shape[1]
        torch.ops.ht_extern.rope_qk(qkv[:, :d], qkv[:, d : 2 * d], pos, table)
        return qkv + 1

    @staticmethod
    def store(qkv, k_cache, v_cache, loc):
        d = k_cache.shape[1]
        torch.ops.ht_extern.store_kv(qkv[:, d : 2 * d], qkv[:, 2 * d :], k_cache, v_cache, loc)
        return qkv + 1

    def run_cases(self, f, cases, indexed=None, lendable=(), learn_pool=False, **provider_options):
        ops = torch.ops.ht_extern
        extern = (ops.scale_qk.default, ops.rope_qk.default, ops.store_kv.default, ops.store_paged.default)
        p = HarvestProvider(("extern",), extern_ops=extern, indexed=indexed, lendable=lendable, **provider_options)
        r = HostTraceReplay(f, opaque=(p,), learn_pool=learn_pool)
        for args in cases:
            # the floating arguments are what the calls write
            ref = [a.clone() if isinstance(a, torch.Tensor) and a.is_floating_point() else a for a in args]
            self.assertEqual(r(*args), f(*ref), atol=0, rtol=0)
            self.assertEqual(args, ref, atol=0, rtol=0)
        return p, r

    def eager_calls(self, r):
        return [rec for v in r.variants for _, rec in v.tape.launches if type(rec) is EagerCall]

    def test_aliased_written_operands(self):
        # q and k as views of one storage: harvested and replayed bitwise, the
        # distance between them part of the key (k after q, then after v)
        w = torch.randn(64, device="cuda", dtype=bf16)
        cases = [(torch.randn(t, 192, device="cuda", dtype=bf16), w, at) for t, at in ((5, 64), (17, 64), (33, 64), (9, 128), (21, 128), (5, 64))]
        p, r = self.run_cases(self.scale, cases)
        self.assertEqual(len(p.refused), 0, p.refused)
        self.assertEqual({k[0] for k in p.bindings}, {((1, 0, 128),), ((1, 0, 256),)})
        self.assertEqual(self.eager_calls(r), [])
        self.assertEqual(r.eager, 1)

    def test_a_dropped_write_is_refused(self):
        # a binding missing the write to k (the second kernel) differs from
        # eager on the group's block: refused, the call stays eager and exact
        launch = harvest._launch

        def first_kernel(b, *args):
            return launch(dataclasses.replace(b, nodes=b.nodes[:1]), *args)

        w = torch.randn(64, device="cuda", dtype=bf16)
        cases = [(torch.randn(t, 192, device="cuda", dtype=bf16), w, 64) for t in (5, 17, 33)]
        with mock.patch.object(harvest, "_launch", first_kernel):
            p, r = self.run_cases(self.scale, cases)
        self.assertEqual(p.bindings, {})
        self.assertTrue(p.refused)
        self.assertTrue(all("differs from eager" in why for why in p.refused.values()), p.refused)

    def test_index_operand(self):
        # the read-only operands are the live ones: the positions index the
        # table, which is past the cap (only written operands are copied)
        table = torch.randn(4096, 64, device="cuda", dtype=bf16)
        cases = [(torch.randn(t, 192, device="cuda", dtype=bf16), torch.randint(4000, 4096, (t,), device="cuda"), table) for t in (7, 30, 7, 12)]
        p, r = self.run_cases(self.rope, cases, extern_cap=64 << 10)
        self.assertEqual(len(p.refused), 0, p.refused)
        self.assertEqual(self.eager_calls(r), [])

    def test_a_size_sweep_harvests_every_key(self):
        # vLLM's range warm-up: every GEMM and extern key is harvested, none refused
        def fn(x, w, pos, table):
            qkv = torch.mm(x, w)
            torch.ops.ht_extern.rope_qk(qkv[:, :64], qkv[:, 64:128], pos, table)
            return qkv + 1

        p = HarvestProvider(("blas", "extern"), extern_ops=(torch.ops.ht_extern.rope_qk.default,), extern_cap=64 << 10)
        r = HostTraceReplay(fn, opaque=(p,))
        w, table = torch.randn(64, 192, device="cuda", dtype=bf16), torch.randn(4096, 64, device="cuda", dtype=bf16)
        for t in (5, 9, 17, 33, 5, 9, 17, 33):
            x, pos = torch.randn(t, 64, device="cuda", dtype=bf16), torch.randint(4000, 4096, (t,), device="cuda")
            self.assertEqual(r(x, w, pos, table), fn(x, w, pos, table), atol=0, rtol=0)
        self.assertEqual(self.eager_calls(r), [])
        self.assertEqual((p.harvests, len(p.bindings), p.refused, r.traces, r.eager), (8, 8, {}, 1, 1))

    @parametrize("case", ["larger", "smaller"])
    def test_a_key_refused_past_the_traced_call_is_exact(self, case):
        # the extern op's key refused at a call past the traced one, the GEMM's key bound: exact,
        # and the GEMM writes no row past its output (one bound at the traced key would at a smaller call)
        def fn(x, w, buf, pos, table):
            qkv = torch.mm(x, w, out=buf[: x.shape[0]])
            torch.ops.ht_extern.rope_qk(qkv[:, :64], qkv[:, 64:128], pos, table)
            return qkv + 1

        sizes = {"larger": (5, 9, 17, 33), "smaller": (5, 17, 33, 9)}[case]
        harvest_ = HarvestProvider._harvest

        def refusing(self, key, *args, **kwargs):
            if key.op is torch.ops.ht_extern.rope_qk.default and key.sizes[0][0] == sizes[-1]:
                raise harvest._Refused("refused by the test")
            return harvest_(self, key, *args, **kwargs)

        p = HarvestProvider(("blas", "extern"), extern_ops=(torch.ops.ht_extern.rope_qk.default,), extern_cap=64 << 10)
        r = HostTraceReplay(fn, opaque=(p,))
        w, table = torch.randn(64, 192, device="cuda", dtype=bf16), torch.randn(4096, 64, device="cuda", dtype=bf16)
        with mock.patch.object(HarvestProvider, "_harvest", refusing):
            for t in sizes * 2:
                x, pos = torch.randn(t, 64, device="cuda", dtype=bf16), torch.randint(4000, 4096, (t,), device="cuda")
                got, want = (torch.full((64, 192), 7.0, device="cuda", dtype=bf16) for _ in range(2))
                self.assertEqual(r(x, w, got, pos, table), fn(x, w, want, pos, table), atol=0, rtol=0)
                self.assertEqual(got, want, atol=0, rtol=0)
        self.assertEqual(list(p.refused.values()), ["refused by the test"])

    def test_a_keyed_site_inside_an_op_with_its_own_guards(self):
        # the GEMM's node is its keyed site's (patched per key); the op's selector takes only its other launches, so
        # no record has two owners (the native variant rejected the spec: "a site's record")
        def fn(x, w):
            return (torch.ops.ht_nested.mm_then_branch(x + 1, w) - 1,)

        p = HarvestProvider(("blas",))
        r = HostTraceReplay(fn, opaque=(p,))
        w = torch.randn(64, 64, device="cuda")
        with tf32_off():
            for m in (8, 40, 8, 40, 100):
                x = torch.randn(m, 64, device="cuda")
                self.assertEqual(r(x, w), fn(x, w), atol=0, rtol=0)
        lowered = r.variants[0].captured.lowered
        site_nodes = {n for s in lowered.sites for n in s.nodes}
        self.assertTrue(site_nodes)
        self.assertTrue(all(site_nodes.isdisjoint(s.nodes) for s in lowered.selectors))
        self.assertEqual(r.eager, 1)
        self.assertGreater(r.replays, 0)

    def test_an_op_holding_a_keyed_site_redispatches(self):
        # the op's own guard flips: its redo's GEMM is the variant's keyed site in the same role, which serves its key
        # per call, so the entry holds only the op's other launches: one trace, every flip redispatched
        def fn(x, w):
            return (torch.ops.ht_nested.mm_then_branch(x + 1, w) - 1,)

        r = HostTraceReplay(fn, opaque=(HarvestProvider(("blas",)),))
        w = torch.randn(64, 64, device="cuda")
        with tf32_off():
            for m in (8, 40, 8, 40, 100, 4):
                x = torch.randn(m, 64, device="cuda")
                self.assertEqual(r(x, w), fn(x, w), atol=0, rtol=0)
        self.assertEqual(r.traces, 1)
        self.assertGreater(r.redispatches, 0)
        self.assertEqual(r.retrace_causes, {})
        for v in r.variants:
            memory.check_plan(v.captured.lowered, v.memory)

    def test_layers_holding_keyed_sites_share_a_redo_template(self):
        # four layers of the op flip together: one full redo per distinct read (the first layer reads an argument,
        # the others the layer before's output), the others its template (verify mode, on in this suite, redoes each
        # in full and compares). One weight: a key holds each argument's own size symbols
        def fn(x, w):
            for _ in range(4):
                x = torch.ops.ht_nested.mm_then_branch(x, w) * 0.5
            return (x,)

        r = HostTraceReplay(fn, opaque=(HarvestProvider(("blas",)),))
        w = torch.randn(64, 64, device="cuda") / 8
        with tf32_off():
            for m in (8, 40, 8, 40):
                x = torch.randn(m, 64, device="cuda")
                self.assertEqual(r(x, w), fn(x, w), atol=0, rtol=0)
        self.assertEqual(r.traces, 1)
        self.assertEqual((r.redos, r.redo_instances), (2, 2))

    def test_an_op_whose_keyed_site_changes_role_retraces(self):
        # the GEMM reads w on one side of the op's guard and w.t() on the other: the redo's nested call is not the
        # variant's site in the same role, so the flip refuses and traces again, counted
        def fn(x, w):
            return (torch.ops.ht_nested.mm_by_branch(x * 2, w) - 1,)

        r = HostTraceReplay(fn, opaque=(HarvestProvider(("blas",)),))
        w = torch.randn(64, 64, device="cuda")
        with tf32_off():
            for m in (8, 40, 8):
                x = torch.randn(m, 64, device="cuda")
                self.assertEqual(r(x, w), fn(x, w), atol=0, rtol=0)
        self.assertEqual(r.traces, 2)
        self.assertEqual(r.redispatches, 0)
        ((cause, n),) = r.retrace_causes.items()
        self.assertEqual((cause[0], cause[1], n), ("dispatch", "ht_nested.mm_by_branch.default", 1))
        self.assertIn("another keyed site in ht_nested.mm_by_branch.default", cause[4])

    def test_a_refused_key_at_a_redispatched_call_retraces(self):
        # the extern key refused (past extern_cap) at a call the variant holds through the bmm's
        # redispatch entry (its trace's m == 1 fails): the twin from the trace's tape would fail
        # that guard, so it refuses and the call traces again, eagerly exact
        def fn(x, w, qkv, pos, table):
            torch.ops.ht_extern.rope_qk(qkv[:, :64], qkv[:, 64:128], pos, table)
            return torch.bmm(x, w).sum(-1), qkv + 1

        p = HarvestProvider(("blas", "extern"), extern_ops=(torch.ops.ht_extern.rope_qk.default,), extern_cap=64 << 10)
        r = HostTraceReplay(fn, opaque=(p,))
        w, table = torch.randn(8, 96, 192, device="cuda", dtype=bf16), torch.randn(4096, 64, device="cuda", dtype=bf16)
        for m, n in ((32, 40), (1, 40), (32, 40), (65, 300), (65, 300)):
            x, qkv = torch.randn(8, m, 96, device="cuda", dtype=bf16), torch.randn(n, 192, device="cuda", dtype=bf16)
            pos, want = torch.randint(0, 4096, (n,), device="cuda"), qkv.clone()
            self.assertEqual(r(x, w, qkv, pos, table), fn(x, w, want, pos, table), atol=0, rtol=0)
            self.assertEqual(qkv, want, atol=0, rtol=0)
        self.assertEqual(len(p.refused), 1)
        self.assertEqual(r.eager_sites_refusals, {"an op's own guards fail at the call: an entry selects it": 1})

    @mock.patch("torch.cuda._host_trace_replay.oob_learn", False)
    def test_a_written_input_is_refused(self):
        # a binding that also writes a live read-only input (a table row no
        # position reads) is refused with every output right
        launch = harvest._launch
        table = torch.randn(4096, 64, device="cuda", dtype=bf16)

        def writes_table(*args):
            why = launch(*args)
            table[0].add_(1)
            return why

        # once (_DIGEST_ONCE) or after each launch
        for digest, once in ((d, o) for d in ("full", "rows") for o in (True, False)):
            p = HarvestProvider(("extern",), extern_ops=(torch.ops.ht_extern.rope_qk.default,), extern_cap=64 << 10, extern_digest=digest)
            r = HostTraceReplay(self.rope, opaque=(p,))
            with mock.patch.object(harvest, "_launch", writes_table), mock.patch.object(harvest, "_DIGEST_ONCE", once):
                for t in (7, 30, 7):
                    qkv, pos = torch.randn(t, 192, device="cuda", dtype=bf16), torch.randint(4000, 4096, (t,), device="cuda")
                    want = self.rope(qkv.clone(), pos, table.clone())
                    self.assertEqual(r(qkv, pos, table), want, atol=0, rtol=0)
            self.assertEqual(p.bindings, {})
            self.assertTrue(p.refused)
            self.assertTrue(all("differs from eager at operand 3" in why for why in p.refused.values()), p.refused)

    @mock.patch("torch.cuda._host_trace_replay.oob_learn", False)
    def test_a_permuted_input_row(self):
        # a binding that also rolls a live read-only input's row (one no
        # position reads): its column sums move, its row sums do not
        launch = harvest._launch
        table = torch.randn(4096, 64, device="cuda", dtype=bf16)

        def rolls_table(*args):
            why = launch(*args)
            table[0] = table[0].roll(1)
            return why

        for digest, refused in (("full", True), ("rows", False)):
            p = HarvestProvider(("extern",), extern_ops=(torch.ops.ht_extern.rope_qk.default,), extern_cap=64 << 10, extern_digest=digest)
            r = HostTraceReplay(self.rope, opaque=(p,))
            with mock.patch.object(harvest, "_launch", rolls_table):
                for t in (7, 30, 7):
                    qkv, pos = torch.randn(t, 192, device="cuda", dtype=bf16), torch.randint(4000, 4096, (t,), device="cuda")
                    want = self.rope(qkv.clone(), pos, table.clone())
                    self.assertEqual(r(qkv, pos, table), want, atol=0, rtol=0)
            self.assertEqual(bool(p.refused), refused, p.refused)
            self.assertEqual(not p.bindings, refused)

    def test_declared_kv_writer(self):
        # caches past the cap: verified on small caches and distinct rows in
        # them, as declared; replayed into the live caches at the live rows
        k_cache, v_cache = (torch.randn(8192, 64, device="cuda", dtype=bf16) for _ in range(2))
        cases = [(torch.randn(t, 192, device="cuda", dtype=bf16), k_cache, v_cache, torch.randperm(8192, device="cuda")[:t]) for t in (3, 11, 40, 11)]
        indexed = {torch.ops.ht_extern.store_kv.default: ("loc", "k_cache", "v_cache")}
        p, r = self.run_cases(self.store, cases, indexed, extern_cap=256 << 10)
        self.assertEqual(len(p.refused), 0, p.refused)
        self.assertTrue(p.bindings)
        self.assertEqual(self.eager_calls(r), [])
        # undeclared, the written caches are past the cap
        p, r = self.run_cases(self.store, cases, extern_cap=256 << 10)
        self.assertEqual(p.bindings, {})
        self.assertTrue(all("past the extern cap" in why for why in p.refused.values()), p.refused)

    @staticmethod
    def store_paged(qkv, kv, slot):
        # kv is [pages, heads, rows, 2 * d], K and V interleaved in each row: the
        # caches are [pages, rows, heads, d] views of it, d elements apart (vLLM's
        # FlashInfer layout on SM100)
        t, (_, h, _, d2) = qkv.shape[0], kv.shape
        k_cache, v_cache = kv.transpose(1, 2).split(d2 // 2, -1)
        k, v = qkv.unflatten(1, (2, h, d2 // 2)).unbind(1)
        torch.ops.ht_extern.store_paged(k, v, k_cache, v_cache, slot)
        return qkv + 1

    def paged_cases(self):
        kv = torch.randn(4096, 2, 16, 128, device="cuda", dtype=bf16)
        return [(torch.randn(t, 2 * 2 * 64, device="cuda", dtype=bf16), kv, torch.randperm(4096 * 16, device="cuda")[:t]) for t in (3, 11, 40, 11)]

    def test_declared_interleaved_kv_writer(self):
        # K and V caches interleaved in one page, past the cap: verified on
        # small caches in one block at their distance, distinct slots spanning
        # the page and row dims, as declared; replayed into the live caches
        indexed = {torch.ops.ht_extern.store_paged.default: ("slot", "k_cache", "v_cache", 2)}
        with mock.patch.object(harvest, "_EXTERN_CAP", 1 << 20):
            p, r = self.run_cases(self.store_paged, self.paged_cases(), indexed)
            self.assertEqual(len(p.refused), 0, p.refused)
            # k and v 256 bytes apart in qkv, v_cache 128 bytes past k_cache
            self.assertEqual({k[0] for k in p.bindings}, {((1, 0, 256), (3, 2, 128))})
            self.assertEqual(self.eager_calls(r), [])
            self.assertEqual(r.eager, 1)
            self.assertGreater(r.replays, 0)
            # undeclared, the written caches are past the cap
            p, r = self.run_cases(self.store_paged, self.paged_cases())
            self.assertEqual(p.bindings, {})
            self.assertTrue(all("past the extern cap" in why for why in p.refused.values()), p.refused)

    def test_interleaved_kv_writer_dropped_v_write_is_refused(self):
        # a binding without its last kernel (the V write) leaves V's bytes in
        # the shared block as they were: differs from eager, refused
        launch = harvest._launch
        indexed = {torch.ops.ht_extern.store_paged.default: ("slot", "k_cache", "v_cache", 2)}
        with mock.patch.object(harvest, "_launch", lambda b, *args: launch(dataclasses.replace(b, nodes=b.nodes[:-1]), *args)):
            p, r = self.run_cases(self.store_paged, self.paged_cases(), indexed)
        self.assertEqual(p.bindings, {})
        self.assertTrue(any("differs from eager" in why for why in p.refused.values()), p.refused)

    def double(self, launch=None, dtype=bf16):
        # double_into's harvests, each with its peak allocation past its start
        p = HarvestProvider(("extern",), extern_ops=(torch.ops.ht_extern.double_into.default,))
        learn, peaks = p.learn, []

        def measured(*args):
            torch.cuda.reset_peak_memory_stats()
            start = torch.cuda.memory_allocated()
            try:
                return learn(*args)
            finally:
                peaks.append(torch.cuda.max_memory_allocated() - start)

        def f(x, out):
            torch.ops.ht_extern.double_into(x, out)
            return out + 1

        p.learn = measured
        r = HostTraceReplay(f, opaque=(p,))
        with mock.patch.object(harvest, "_launch", launch or harvest._launch):
            for t in (2048, 1024, 2048):
                x = torch.randn(t, 8192, device="cuda").to(dtype) if dtype.is_floating_point else torch.randint(-1000, 1000, (t, 8192), device="cuda", dtype=dtype)
                out = torch.empty_like(x)
                self.assertEqual(r(x, out), x * 2 + 1, atol=0, rtol=0)
                self.assertEqual(out, x * 2, atol=0, rtol=0)
        return p, r, peaks

    @mock.patch("torch.cuda._host_trace_replay.oob_learn", False)
    def test_written_not_read_one_copy(self):
        # eager on the random copy of an output it writes without reading
        # gives the live result: the binding is checked against the live
        # bytes, on that one copy (not a reference copy and a second)
        p, r, peaks = self.double()
        self.assertEqual(len(p.refused), 0, p.refused)
        self.assertEqual(len(p.bindings), 1)
        self.assertEqual(self.eager_calls(r), [])
        # one copy of out and a 16 MiB comparison piece; a second copy would pass 2x
        self.assertLess(max(peaks), 1.75 * 2048 * 8192 * 2)

    @mock.patch.object(harvest, "_EXTERN_FAST", False)
    @mock.patch("torch.cuda._host_trace_replay.oob_learn", False)
    def test_written_not_read_fast_path_off(self):
        # without the fast path the binding is checked on fresh copies: still bound, two copies' peak
        p, r, peaks = self.double()
        self.assertEqual(len(p.refused), 0, p.refused)
        self.assertEqual(len(p.bindings), 1)
        self.assertEqual(self.eager_calls(r), [])
        self.assertGreaterEqual(max(peaks), 2 * 2048 * 8192 * 2)

    def test_a_float8_output_is_harvested(self):
        # a written float8 operand starts the verify random, drawn in float32 (float8 has no uniform_)
        p = HarvestProvider(("extern",), extern_ops=(torch.ops.ht_extern.to_fp8.default,))

        def f(x):
            out = torch.empty(x.shape, device=x.device, dtype=torch.float8_e4m3fn)
            torch.ops.ht_extern.to_fp8(x * 0.5, out)
            return out

        r = HostTraceReplay(f, opaque=(p,))
        for t in (64, 256, 64, 256):
            x = torch.randn(t, 128, device="cuda")
            self.assertEqual(r(x).view(torch.uint8), f(x).view(torch.uint8), atol=0, rtol=0)
        self.assertEqual(p.refused, {})
        self.assertEqual(len(p.bindings), 2)
        self.assertEqual(self.eager_calls(r), [])
        # the first call runs eagerly
        self.assertEqual((r.traces, r.replays, r.eager), (1, 2, 1))

    def test_written_not_read_dropped_write_is_refused(self):
        # a binding that writes nothing leaves the random copy: refused
        launch = harvest._launch
        p, r, _ = self.double(lambda b, *args: launch(dataclasses.replace(b, nodes=()), *args))
        self.assertEqual(p.bindings, {})
        self.assertTrue(p.refused)
        self.assertTrue(all("differs from eager" in why for why in p.refused.values()), p.refused)

    @mock.patch("torch.cuda._host_trace_replay.oob_learn", False)
    def test_written_int_one_copy(self):
        # an integer output written without reading is zeros in the copy:
        # eager on it gives the live result, so still one copy
        p, r, _ = self.double(dtype=torch.int32)
        self.assertEqual(len(p.refused), 0, p.refused)
        self.assertEqual(len(p.bindings), 1)
        self.assertEqual(self.eager_calls(r), [])

    @mock.patch("torch.cuda._host_trace_replay.oob_learn", False)
    def test_written_int_dropped_write_is_refused(self):
        # a binding that writes nothing leaves the zeros, not the live result: refused
        launch = harvest._launch
        p, r, _ = self.double(lambda b, *args: launch(dataclasses.replace(b, nodes=()), *args), torch.int32)
        self.assertEqual(p.bindings, {})
        self.assertTrue(p.refused)
        self.assertTrue(all("differs from eager" in why for why in p.refused.values()), p.refused)

    def test_not_allowlisted(self):
        # an op not in extern_ops stays an eager step: it writes a storage another operand is of
        w = torch.randn(64, device="cuda", dtype=bf16)
        f = HostTraceReplay(self.scale, opaque=(HarvestProvider(("extern",)),))
        for t in (5, 17, 33):
            qkv = torch.randn(t, 192, device="cuda", dtype=bf16)
            want = self.scale(qkv.clone(), w, 64)
            self.assertEqual(f(qkv, w, 64), want, atol=0, rtol=0)
        reasons = {rec.reason for rec in self.eager_calls(f)}
        self.assertEqual(reasons, {"ht_extern.scale_qk.default writes a storage another operand is of"})

    @mock.patch("torch.cuda._host_trace_replay.oob_learn", False)
    def test_learned_without_its_call(self):
        # a key at a new size is learned without a run of the model: the
        # positions and rows zeros, q and k views of one buffer, the table and
        # caches (past _LENT) views of the lendable tensors, left untouched
        table = torch.randn(4096, 64, device="cuda", dtype=bf16)
        k_cache, v_cache = (torch.randn(8192, 64, device="cuda", dtype=bf16) for _ in range(2))

        def f(qkv, pos, table, k_cache, v_cache, loc):
            return self.store(self.rope(qkv, pos, table), k_cache, v_cache, loc)

        def case(t):
            return torch.randn(t, 192, device="cuda", dtype=bf16), torch.randint(4096, (t,), device="cuda"), table, k_cache, v_cache, torch.randperm(8192, device="cuda")[:t]

        indexed = {torch.ops.ht_extern.store_kv.default: ("loc", "k_cache", "v_cache")}
        sizes = (5, 5, 17, 33, 9, 17, 40)
        with mock.patch.object(harvest, "_LENT", 64 << 10):
            for lendable, learned in (((table, k_cache, v_cache), 4), ((), 0)):
                p, r = self.run_cases(f, [case(t) for t in sizes], indexed, lendable)
                self.assertEqual(len(p.refused), 0, p.refused)
                self.assertEqual(self.eager_calls(r), [])
                self.assertEqual(r.learned, learned)

    @parametrize("unguarded", [False, True])
    @mock.patch("torch.cuda._host_trace_replay.oob_learn", False)
    def test_learned_interleaved_kv_writer(self, unguarded):
        # K and V caches as views of one lendable tensor past _LENT (an alias
        # group): once the variant is keyed (the fourth call), a new size's key
        # is learned without its call under lend_alias_groups, at the key's
        # distance; refused otherwise, the learning variant runs it eagerly.
        # With bound_addresses_unguarded off the trace guards the slots'
        # numel() == 0 and the first new size traces again
        indexed = {torch.ops.ht_extern.store_paged.default: ("slot", "k_cache", "v_cache", 2)}
        kv = torch.randn(4096, 2, 16, 128, device="cuda", dtype=bf16)
        cases = [(torch.randn(t, 2 * 2 * 64, device="cuda", dtype=bf16), kv, torch.randperm(4096 * 16, device="cuda")[:t]) for t in (3, 3, 3, 3, 11, 40, 17)]
        with (
            mock.patch.object(harvest, "_EXTERN_CAP", 1 << 20),
            mock.patch.object(harvest, "_LENT", 64 << 10),
            mock.patch.object(torch.cuda._host_trace, "bound_addresses_unguarded", unguarded),
        ):
            for groups, learned in ((True, 3), (False, 0)):
                p, r = self.run_cases(self.store_paged, cases, indexed, (kv,), lend_alias_groups=groups)
                self.assertEqual(len(p.refused), 0, p.refused)
                self.assertEqual(self.eager_calls(r), [])
                self.assertEqual((r.learned, r.traces, r.replays, r.eager), (learned, 1, 5, 1) if unguarded else (learned, 2, 4, 1))

    @mock.patch("torch.cuda._host_trace_replay.oob_learn", False)
    def test_learned_in_learning_variant(self):
        # rope's key bound by another entry: f's trace keys rope and leaves
        # store opaque (a learning variant). A new size's rope key is not
        # learned in it: f traces again
        table = torch.randn(4096, 64, device="cuda", dtype=bf16)
        k_cache, v_cache = (torch.randn(8192, 64, device="cuda", dtype=bf16) for _ in range(2))

        def f(qkv, pos, table, k_cache, v_cache, loc):
            return self.store(self.rope(qkv, pos, table), k_cache, v_cache, loc)

        def case(t):
            return torch.randn(t, 192, device="cuda", dtype=bf16), torch.randint(4096, (t,), device="cuda"), table, k_cache, v_cache, torch.randperm(8192, device="cuda")[:t]

        ops = torch.ops.ht_extern
        indexed = {ops.store_kv.default: ("loc", "k_cache", "v_cache")}
        with mock.patch.object(harvest, "_LENT", 64 << 10):
            p = HarvestProvider(("extern",), extern_ops=(ops.rope_qk.default, ops.store_kv.default), indexed=indexed, lendable=(table, k_cache, v_cache))
            rope = HostTraceReplay(self.rope, opaque=(p,))
            for qkv, pos, *_ in [case(5) for _ in range(3)]:
                rope(qkv, pos, table)
            r = HostTraceReplay(f, opaque=(p,))
            for args in [case(t) for t in (5, 5, 17, 33, 9, 40)]:
                ref = [a.clone() if a.is_floating_point() else a for a in args]
                self.assertEqual(r(*args), f(*ref), atol=0, rtol=0)
                self.assertEqual(args, ref, atol=0, rtol=0)
            self.assertEqual(len(p.refused), 0, p.refused)
            self.assertEqual(self.eager_calls(r), [])
            self.assertEqual((r.learned, r.traces), (0, 2))

    @mock.patch("torch.cuda._host_trace_replay.oob_learn", False)
    def test_learner_redispatches_selector(self):
        # rope and store opaque at every size (a learning variant); the
        # embedding's index_select takes its small-index kernel at <= 16
        # indices, a selector. Traced at 17, the learning variant does not
        # redispatch it at 9: 9 traces again
        table = torch.randn(4096, 64, device="cuda", dtype=bf16)
        emb = torch.randn(1000, 192, device="cuda", dtype=bf16)
        k_cache, v_cache = (torch.randn(8192, 64, device="cuda", dtype=bf16) for _ in range(2))

        def f(ids, pos, table, k_cache, v_cache, loc, emb):
            return self.store(self.rope(emb.index_select(0, ids), pos, table), k_cache, v_cache, loc)

        def case(t):
            return torch.randint(1000, (t,), device="cuda"), torch.randint(4096, (t,), device="cuda"), table, k_cache, v_cache, torch.randperm(8192, device="cuda")[:t], emb

        ops = torch.ops.ht_extern
        indexed = {ops.store_kv.default: ("loc", "k_cache", "v_cache")}
        p = HarvestProvider(("extern",), extern_ops=(ops.rope_qk.default, ops.store_kv.default), indexed=indexed, lendable=(table, k_cache, v_cache))
        r = HostTraceReplay(f, opaque=(p,))
        for args in [case(t) for t in (5, 17, 33, 9, 40)]:
            ref = [a.clone() if a.is_floating_point() else a for a in args]
            self.assertEqual(r(*args), f(*ref), atol=0, rtol=0)
            self.assertEqual(args, ref, atol=0, rtol=0)
        self.assertTrue(all(v.learns for v in r.variants))
        # the entry's first call (5) runs eagerly; 17 traces
        self.assertEqual((r.traces, r.redispatches, r.eager), (2, 0, 1))

    @parametrize("oob,traces", [(True, 1), (False, 4)])
    def test_new_shapes_are_local(self, oob, traces):
        # a shape sweep over one variant's class traces once. A new length is
        # learned (rope, keyed by another entry), redispatched (the embedding's
        # index_select small-index kernel at <= 16 indices), by oob_learn; the
        # sum's split reduction is a section
        table = torch.randn(4096, 64, device="cuda", dtype=bf16)
        emb = torch.randn(1000, 192, device="cuda", dtype=bf16)
        k_cache, v_cache = (torch.randn(8192, 64, device="cuda", dtype=bf16) for _ in range(2))

        def f(ids, pos, table, k_cache, v_cache, loc, emb):
            h = emb.index_select(0, ids)
            return self.store(self.rope(h, pos, table), k_cache, v_cache, loc), h.sum(0)

        def case(t):
            return torch.randint(1000, (t,), device="cuda"), torch.randint(4096, (t,), device="cuda"), table, k_cache, v_cache, torch.randperm(8192, device="cuda")[:t], emb

        ops = torch.ops.ht_extern
        indexed = {ops.store_kv.default: ("loc", "k_cache", "v_cache")}
        with mock.patch("torch.cuda._host_trace_replay.oob_learn", oob):
            p = HarvestProvider(("extern",), extern_ops=(ops.rope_qk.default, ops.store_kv.default), indexed=indexed, lendable=(table, k_cache, v_cache))
            rope = HostTraceReplay(self.rope, opaque=(p,))
            for qkv, pos in [(torch.randn(5, 192, device="cuda", dtype=bf16), torch.randint(4096, (5,), device="cuda")) for _ in range(3)]:
                rope(qkv, pos, table)
            r = HostTraceReplay(f, opaque=(p,))
            for args in [case(t) for t in (5, 5, 17, 33, 9, 40, 300, 12, 2000, 64, 4000, 7)]:
                ref = [a.clone() if a.is_floating_point() else a for a in args]
                self.assertEqual(r(*args), f(*ref), atol=0, rtol=0)
                self.assertEqual(args, ref, atol=0, rtol=0)
        self.assertEqual((len(p.refused), self.eager_calls(r), r.traces), (0, [], traces))
        if oob:
            self.assertEqual((r.redispatches, r.sections, r.relowers, r.eager, len(r.variants), r.learners), (2, 1, 0, 1, 1, 0))

    @mock.patch("torch.cuda._host_trace_replay.oob_learn", False)
    def test_learn_pool(self):
        # with learn_pool the out-of-band learns' own operands are in a
        # private pool, and its segments are gone after the call; without it
        # they are in the default pool
        table = torch.randn(4096, 64, device="cuda", dtype=bf16)
        k_cache, v_cache = (torch.randn(8192, 64, device="cuda", dtype=bf16) for _ in range(2))

        def f(qkv, pos, table, k_cache, v_cache, loc):
            return self.store(self.rope(qkv, pos, table), k_cache, v_cache, loc)

        def case(t):
            return torch.randn(t, 192, device="cuda", dtype=bf16), torch.randint(4096, (t,), device="cuda"), table, k_cache, v_cache, torch.randperm(8192, device="cuda")[:t]

        def pool_of(t):
            segments = torch.cuda.memory_snapshot(include_traces=False)
            return next(tuple(s["segment_pool_id"]) for s in segments if s["address"] <= t.data_ptr() < s["address"] + s["total_size"])

        pools, out_of_band = [], []
        learn, replay_learn = HarvestProvider.learn, HostTraceReplay._learn

        def recording(self, key, args, kwargs, operands):
            lent = (table, k_cache, v_cache)
            if out_of_band:
                pools.extend(pool_of(t) for t in operands if not any(t.untyped_storage().data_ptr() == x.data_ptr() for x in lent))
            return learn(self, key, args, kwargs, operands)

        def learning(self, *args):
            out_of_band.append(True)
            try:
                return replay_learn(self, *args)
            finally:
                out_of_band.pop()

        indexed = {torch.ops.ht_extern.store_kv.default: ("loc", "k_cache", "v_cache")}
        for learn_pool in (True, False):
            pools.clear()
            with (
                mock.patch.object(harvest, "_LENT", 64 << 10),
                mock.patch.object(HarvestProvider, "learn", recording),
                mock.patch.object(HostTraceReplay, "_learn", learning),
            ):
                p, r = self.run_cases(f, [case(t) for t in (5, 5, 17, 33, 9, 17, 40)], indexed, (table, k_cache, v_cache), learn_pool=learn_pool)
            self.assertEqual(r.learned, 4)
            if learn_pool:
                self.assertTrue(pools and (0, 0) not in pools, pools)
                live = {tuple(s["segment_pool_id"]) for s in torch.cuda.memory_snapshot(include_traces=False)}
                self.assertFalse(live & set(pools), pools)
            else:
                self.assertEqual(set(pools), {(0, 0)})

    def test_cap(self):
        # operands past the cap are refused and run eagerly, still exact; under it they bind
        w = torch.randn(64, device="cuda", dtype=bf16)
        cases = [(torch.randn(t, 192, device="cuda", dtype=bf16), w, 64) for t in (5, 17, 33)]
        p, r = self.run_cases(self.scale, cases, extern_cap=1024)
        self.assertEqual(p.bindings, {})
        self.assertTrue(all("past the extern cap" in why for why in p.refused.values()), p.refused)
        p, r = self.run_cases(self.scale, cases, extern_cap=1 << 20)
        self.assertEqual(p.refused, {})
        self.assertTrue(p.bindings)
        # extern_cap None reads the module's _EXTERN_CAP (1 GiB) at each harvest
        with mock.patch.object(harvest, "_EXTERN_CAP", 1024):
            p, r = self.run_cases(self.scale, cases)
        self.assertEqual(p.bindings, {})
        self.assertTrue(all("past the extern cap 1024" in why for why in p.refused.values()), p.refused)


# an extern op whose kernel's parameter struct has padding the host leaves as
# the stack's bytes, and a field holding a stack address the kernel never reads
# (as cuBLAS's split-K reduce takes host-mode alpha and beta pointers);
# set_pattern smears the stack below the op first, as another caller's frames
# would
_PADDED_OP = r"""
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>
#include <cstring>

namespace {

struct Params {
  const float* x;
  int n;
  float* out;
  float scale;
  const void* host;
};

__global__ void scale_kernel(Params p) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < p.n) {
    p.out[i] = p.x[i] * p.scale;
  }
}

int64_t pattern = 0;

__attribute__((noinline)) void smear() {
  unsigned char buf[1 << 14];
  std::memset(buf, static_cast<int>(pattern), sizeof(buf));
  asm volatile("" : : "r"(buf) : "memory");
}

__attribute__((noinline)) void launch(const at::Tensor& x, const at::Tensor& out, double scale) {
  Params p;
  p.x = x.const_data_ptr<float>();
  p.n = static_cast<int>(x.numel());
  p.out = out.mutable_data_ptr<float>();
  p.scale = static_cast<float>(scale);
  p.host = &p;
  scale_kernel<<<(p.n + 255) / 256, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(p);
}

void scale_into(const at::Tensor& x, const at::Tensor& out, double scale) {
  if (pattern) {
    smear();
  }
  launch(x, out, scale);
}

void set_pattern(int64_t p) {
  pattern = p;
}

} // namespace

TORCH_LIBRARY(ht_padded, m) {
  m.def("scale_into(Tensor x, Tensor(a!) out, float scale) -> ()");
  m.def("set_pattern(int pattern) -> ()", set_pattern);
}

TORCH_LIBRARY_IMPL(ht_padded, CUDA, m) {
  m.impl("scale_into", scale_into);
}
"""


@functools.cache
def _padded_op():
    from torch.utils.cpp_extension import load_inline

    load_inline("ht_padded_params", cpp_sources="", cuda_sources=_PADDED_OP, is_python_module=False)
    torch.library.register_fake("ht_padded::scale_into", lambda x, out, scale: None)
    return torch.ops.ht_padded


def _scale_plus_one(x, out):
    torch.ops.ht_padded.scale_into(x, out, 3.0)
    return out + 1


@instantiate_parametrized_tests
class TestOutOfBandLearn(TestCase):
    @staticmethod
    def f(x, w):
        return torch.mm(x, w) + 1

    def calls(self, r, sizes, w):
        for t in sizes:
            x = torch.randn(t, 64, device="cuda", dtype=bf16)
            self.assertEqual(r(x, w), self.f(x, w), atol=0, rtol=0)

    def test_prepare(self):
        # prepare learns a new size's keys without running the call, which then replays
        w = torch.randn(64, 48, device="cuda", dtype=bf16)
        r = HostTraceReplay(self.f, opaque=(HarvestProvider(),))
        self.calls(r, (32, 32, 40, 40), w)
        learned, replays, eager = r.learned, r.replays, r.eager
        self.assertTrue(r.prepare(torch.empty(72, 64, device="cuda", dtype=bf16), w))
        self.assertGreater(r.learned, learned)
        self.calls(r, (72,), w)
        self.assertEqual((r.learned - learned, r.replays, r.eager), (1, replays + 1, eager))

    @parametrize("oob", [False, True])
    def test_a_late_key_is_learned_locally(self, oob):
        # a new size after many replays learns its key at the call: no trace.
        # With oob_learn the trace bound its own key; without, its learning
        # variant relowers once
        w = torch.randn(64, 48, device="cuda", dtype=bf16)
        r = HostTraceReplay(self.f, opaque=(HarvestProvider(),))
        with mock.patch("torch.cuda._host_trace_replay.oob_learn", oob):
            self.calls(r, (32,) * 20, w)
            traces, learned, replays = r.traces, r.learned, r.replays
            self.calls(r, (40, 40, 48, 32), w)
        self.assertEqual((r.traces - traces, r.learned - learned, r.relowers, r.eager), (0, 2, 0 if oob else 1, 1))

    def test_many_rows_keep_each_rows_users(self):
        # each new size adds a row to the GEMM's table, after which the call
        # indexes which nodes read which rows from the site's reads (not from
        # every row): early and late rows both replay at fresh operand
        # addresses, bitwise
        w = torch.randn(64, 48, device="cuda", dtype=bf16)
        p = HarvestProvider()
        r = HostTraceReplay(self.f, opaque=(p,))
        self.calls(r, (32, 32), w)
        self.calls(r, range(33, 113), w)
        learned, traces = r.learned, r.traces
        self.calls(r, (33, 112, 70, 33, 32), w)
        self.assertEqual((r.learned - learned, r.traces - traces, r.eager), (0, 0, 1))
        # a binding's topology is computed once, not per site and arm that reads it
        b = p.bindings[next(iter(p.bindings))]
        self.assertIs(b.topology, b.topology)

    @parametrize("case", ["conv", "conv_backward", "attention", "mm"])
    def test_a_learn_costs_at_most_its_workspace(self, case):
        # the call that learns its key (the trace's warm-up call) holds no
        # more memory than eager's call but the binding's workspace and small
        # terms (digests, mirror windows): no copy of an operand or output
        # (128 MiB each here), captures into address space only
        def conv(x, w):
            return torch.nn.functional.conv2d(x, w, padding=1)

        def attention(q, k, v):
            return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

        def backward(g, x, w):
            return aten.convolution_backward(g, x, w, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])[:2]

        fn, args = {
            "conv": (conv, (torch.randn(64, 64, 128, 128, device="cuda", dtype=bf16), torch.randn(64, 64, 3, 3, device="cuda", dtype=bf16))),
            "attention": (attention, tuple(torch.randn(32, 16, 2048, 64, device="cuda", dtype=bf16) for _ in range(3))),
            "mm": (torch.mm, (torch.randn(16384, 4096, device="cuda", dtype=bf16), torch.randn(4096, 4096, device="cuda", dtype=bf16))),
            "conv_backward": (backward, tuple(torch.randn(64, 64, 128, 128, device="cuda", dtype=bf16) for _ in range(2)) + (torch.randn(64, 64, 3, 3, device="cuda", dtype=bf16),)),
        }[case]
        providers = (HarvestProvider(), HarvestProvider(("attention",)), HarvestProvider(("conv",)))
        with torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=True):
            eager = _backed_peak(fn, args)
            r = HostTraceReplay(fn, opaque=providers)
            r(*args)
            learn = _backed_peak(r, args)
            whys = [w for p in providers for w in p.refused.values()]
            if r.learned == 0 and whys and all(harvest._STALE_STRUCT in w for w in whys):
                self.skipTest(whys[0])
            self.assertGreater(r.learned, 0)
            workspace = sum(sum(b.scratch) for p in providers for b in p.bindings.values())
            self.assertLessEqual(learn - eager, workspace + (64 << 20))
            self.assertEqual(r(*args), fn(*args), atol=0, rtol=0)

    def test_a_new_row_packs_only_what_it_changes(self):
        # the call after a new key's row is added packs the nodes a changed row
        # or site reads, not every node: z's kernel (inputs at fixed
        # addresses, no key) is not packed again
        def f(x, w, z, zout):
            torch.mul(z, 2, out=zout)
            return torch.mm(x, w) + 1

        w = torch.randn(64, 48, device="cuda", dtype=bf16)
        z = torch.randn(4096, device="cuda", dtype=bf16)
        zout = torch.empty_like(z)
        r = HostTraceReplay(f, opaque=(HarvestProvider(),))
        xs = {t: torch.randn(t, 64, device="cuda", dtype=bf16) for t in (32, 40, 48)}
        for t in (32, 32, 40, 32, 40):
            self.assertEqual(r(xs[t], w, z, zout), f(xs[t], w, z, zout), atol=0, rtol=0)
        before = torch._C._host_trace_kernel_packs()
        self.assertEqual(r(xs[48], w, z, zout), f(xs[48], w, z, zout), atol=0, rtol=0)
        packs = torch._C._host_trace_kernel_packs() - before
        records = len(torch._C._host_trace_held_images(r.variants[0].native))
        self.assertLess(packs, records)
        self.assertEqual((r.traces, r.learned), (1, 3))

    def test_equal_keys_bind_once_per_call(self):
        # one GEMM key in three layers: a new size's key is built, bound and
        # learned once, then taken by each layer's site
        def f(x, w):
            for _ in range(3):
                x = torch.mm(x, w)
            return x

        w = torch.randn(64, 64, device="cuda", dtype=bf16) * 0.1
        r = HostTraceReplay(f, opaque=(HarvestProvider(),))
        self.calls_with(r, f, (32, 32), w)
        with mock.patch.object(HarvestProvider, "bind", autospec=True, side_effect=HarvestProvider.bind) as bind:
            self.calls_with(r, f, (40,), w)
        self.assertEqual((bind.call_count, r.learned, r.traces), (1, 2, 1))

    def calls_with(self, r, f, sizes, w):
        for t in sizes:
            x = torch.randn(t, 64, device="cuda", dtype=bf16)
            self.assertEqual(r(x, w), f(x, w), atol=0, rtol=0)

    def test_a_structural_refusal_holds_for_the_layout_class(self):
        # the op's kernels refuse (a memcpy node) at its first key, learned at
        # the warm-up's call; a key at another size of its layout class refuses
        # too, with no harvest, and the call stays a plain eager step of one
        # variant, traced once
        op = torch.ops.ht_extern.copy_twice.default

        def f(x):
            out = torch.empty_like(x)
            op(x, out)
            return out + 1

        p = HarvestProvider(("extern",), extern_ops=(op,))
        r = HostTraceReplay(f, opaque=(p,))
        for t in (8, 16, 24, 32, 40, 48):
            x = torch.randn(t, 64, device="cuda")
            self.assertEqual(r(x), f(x), atol=0, rtol=0)
        self.assertEqual(len(p.refused_class), 1)
        self.assertIn("node of type", next(iter(p.refused_class.values())))
        # a key of another layout (a transposed input) is not refused by it
        f32 = torch.float32
        transposed = OpaqueKey(op, (f32, f32), ((8, 64), (8, 64)), ((1, 8), (64, 1)), (0, 0), (), torch.cuda.current_device(), library_state())
        self.assertIsNone(p.refusal(transposed))
        self.assertIsNotNone(p.refusal(dataclasses.replace(transposed, strides=((64, 1), (64, 1)))))
        self.assertEqual((p.harvests, r.traces), (1, 1))

    def test_prepare_range(self):
        # every size of a range prepared: a first call at any of them learns
        # nothing and replays, bitwise equal to eager
        w = torch.randn(64, 48, device="cuda", dtype=bf16)
        r = HostTraceReplay(self.f, opaque=(HarvestProvider(),))
        self.calls(r, (32, 32, 40, 40), w)
        sizes = range(41, 300)
        self.assertTrue(all(r.prepare(torch.empty(t, 64, device="cuda", dtype=bf16), w) for t in sizes))
        learned, traces, replays, eager = r.learned, r.traces, r.replays, r.eager
        self.calls(r, sizes, w)
        self.assertEqual((r.learned - learned, r.traces - traces, r.replays - replays, r.eager - eager), (0, 0, len(sizes), 0))

    def test_background(self):
        # a new size runs eagerly while its key is learned on the worker; the
        # main thread allocates, frees and synchronizes meanwhile; later calls replay
        w = torch.randn(64, 48, device="cuda", dtype=bf16)
        p = HarvestProvider()
        r = HostTraceReplay(self.f, opaque=(p,), background=True)
        # holding the worker keeps 40 unlearned for its second call
        with _BACKGROUND.learning:
            self.calls(r, (32, 32, 40, 40), w)
        sizes = range(48, 400, 8)
        self.calls(r, sizes, w)
        # and 40's two calls: the trace learned only 32 (oob_learn)
        self.assertEqual(r.deferred, len(sizes) + 2)
        while not _BACKGROUND.idle():
            junk = [torch.empty(n << 20, dtype=torch.uint8, device="cuda") for n in (3, 17, 33)]
            del junk
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        self.assertEqual((r.background_errors, p.refused), ([], {}))
        replays, eager = r.replays, r.eager
        self.calls(r, sizes, w)
        self.assertEqual((r.replays - replays, r.eager - eager, r.deferred), (len(sizes), 0, len(sizes) + 2))

    def test_background_under_fullgraph(self):
        # a deferral is a learner: fullgraph runs the call eagerly and counts it;
        # forbid_learners learns the key at the call instead
        w = torch.randn(64, 48, device="cuda", dtype=bf16)
        sizes = range(48, 112, 8)
        r = HostTraceReplay(self.f, opaque=(HarvestProvider(),), background=True, fullgraph=True)
        with _BACKGROUND.learning:
            self.calls(r, (32, 32, 40, 40), w)
        eager = r.eager
        self.calls(r, sizes, w)
        # 40's two calls deferred too (oob_learn)
        self.assertEqual((r.deferred, r.eager - eager), (len(sizes) + 2, len(sizes)))
        while not _BACKGROUND.idle():
            time.sleep(0.01)
        self.assertEqual(r.background_errors, [])
        replays = r.replays
        self.calls(r, sizes, w)
        self.assertEqual((r.replays - replays, r.eager - eager, r.deferred), (len(sizes), len(sizes), len(sizes) + 2))
        g = HostTraceReplay(self.f, opaque=(HarvestProvider(),), background=True, fullgraph=True, forbid_learners=True)
        self.calls(g, (32, 32, 40, 40), w)
        eager, replays = g.eager, g.replays
        self.calls(g, sizes, w)
        self.assertEqual((g.deferred, g.eager - eager, g.replays - replays, g.learners), (0, 0, len(sizes), 0))

    def test_background_while_the_main_thread_allocates(self):
        # the main thread's allocations during a worker's capture are not the
        # call's: every deferred size replays after idle, run after run
        w = torch.randn(64, 48, device="cuda", dtype=bf16)
        sizes = range(48, 120, 8)
        for _ in range(20):
            p = HarvestProvider()
            r = HostTraceReplay(self.f, opaque=(p,), background=True)
            with _BACKGROUND.learning:
                self.calls(r, (32, 32, 40, 40), w)
            self.calls(r, sizes, w)
            # 40's two calls too: the trace learned only 32 (oob_learn)
            self.assertEqual(r.deferred, len(sizes) + 2)
            while not _BACKGROUND.idle():
                junk = [torch.empty(n << 20, dtype=torch.uint8, device="cuda") for n in (3, 17, 33)]
                del junk
                torch.cuda.synchronize()
            replays, eager = r.replays, r.eager
            self.calls(r, sizes, w)
            self.assertEqual((r.replays - replays, r.eager - eager), (len(sizes), 0))
            self.assertEqual((r.eager, r.background_errors, p.refused, p.transient), (3 + len(sizes), [], {}, {}))

    def test_a_failed_background_learn_is_an_error(self):
        # a worker learn that binds nothing and refuses nothing is in
        # background_errors; the key's next call learns it in the foreground
        w = torch.randn(64, 48, device="cuda", dtype=bf16)
        p = HarvestProvider()
        r = HostTraceReplay(self.f, opaque=(p,), background=True)
        learn, failing = HarvestProvider.learn, [72]

        def flaky(self, key, *args):
            if threading.current_thread() is not threading.main_thread() and key.sizes[0][0] in failing:
                failing.clear()
                return None
            return learn(self, key, *args)

        with mock.patch.object(HarvestProvider, "learn", flaky):
            self.calls(r, (32, 32, 40, 40, 72), w)
            while not _BACKGROUND.idle():
                torch.cuda.synchronize()
            self.assertEqual(len(r.background_errors), 1)
            self.assertIn("bound nothing and refused nothing", r.background_errors[0])
            replays, eager, deferred = r.replays, r.eager, r.deferred
            self.calls(r, (72, 72), w)
        self.assertEqual((r.replays - replays, r.eager - eager, r.deferred - deferred, p.refused), (2, 0, 0, {}))

    def test_persist(self):
        # saved bindings bind in a new process with no harvest: each is
        # checked against one plain capture of the call (its kernels and
        # their parameters past the operand slots)
        w = torch.randn(64, 48, device="cuda", dtype=bf16)
        p = HarvestProvider()
        r = HostTraceReplay(self.f, opaque=(p,))
        self.calls(r, (32, 32, 40, 72, 72), w)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "bindings.pickle")
            saved = p.save(path)
            self.assertEqual(saved, len(p.bindings))
            code = f"""
import sys, torch
sys.path.insert(0, {os.path.dirname(os.path.abspath(__file__))!r})
from torch.cuda._host_trace_harvest import HarvestProvider
from torch.cuda._host_trace_replay import HostTraceReplay
f = lambda x, w: torch.mm(x, w) + 1
p = HarvestProvider()
loaded = p.load({path!r})
r = HostTraceReplay(f, opaque=(p,))
w = torch.randn(64, 48, device="cuda", dtype=torch.bfloat16)
for t in (32, 32, 40, 72, 72):
    x = torch.randn(t, 64, device="cuda", dtype=torch.bfloat16)
    assert torch.equal(r(x, w), f(x, w))
print("ok", loaded, p.harvests, p.restored, p.stale)
"""
            out = subprocess.check_output([sys.executable, "-c", code], text=True)
        self.assertIn(f"ok {saved} 0 {saved} []", out)

    def test_persist_stale(self):
        # a stored binding whose kernels differ from the plain capture is
        # dropped and the key harvested
        w = torch.randn(64, 48, device="cuda", dtype=bf16)
        p = HarvestProvider()
        self.calls(HostTraceReplay(self.f, opaque=(p,)), (32, 32, 40), w)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "bindings.pickle")
            p.save(path)
            q = HarvestProvider()
            self.assertEqual(q.load(path), len(p.bindings))
            self.assertEqual(q.load(path, versions={"other": 1}), 0)
            q.load(path)
            q.stored = {k: dataclasses.replace(b, nodes=tuple(dataclasses.replace(n, grid=(n.grid[0] + 1, *n.grid[1:])) if hasattr(n, "grid") else n for n in b.nodes)) for k, b in q.stored.items()}
            self.calls(HostTraceReplay(self.f, opaque=(q,)), (32, 32, 40), w)
        self.assertEqual((q.restored, len(q.stale) > 0, q.harvests), (0, True, len(p.bindings)))


    def test_a_restore_visits_only_the_keys_its_functions_complete(self):
        # load's stored keys wait for their kernel functions by name; a restore that loads a function visits only the keys
        # waiting for it (index_visits), and holds every stored key whose functions are then all loaded (what a scan of
        # every stored key gives: verify_redo_templates checks it at each restore)
        from torch.cuda._host_trace_opaque import OpaqueKernel

        w = torch.randn(64, 48, device="cuda", dtype=bf16)
        p = HarvestProvider()
        self.calls(HostTraceReplay(self.f, opaque=(p,)), (32, 32, 40, 72, 72, 200, 200), w)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "bindings.pickle")
            p.save(path)
            q = HarvestProvider()
            self.assertEqual(q.load(path), len(p.bindings))
        before = dict(q.stored)
        uses = collections.Counter((k[7], n.function) for k, b in before.items() for n in {n.function: n for n in b.nodes if isinstance(n, OpaqueKernel)}.values())
        self.calls(HostTraceReplay(self.f, opaque=(q,)), (32, 32), w)
        self.assertGreater(q.restored, 0)
        loaded = {f for f in uses if f in q.functions}
        complete = {k for k, b in before.items() if all((k[7], n.function) in q.functions for n in b.nodes if isinstance(n, OpaqueKernel))}
        self.assertEqual(complete, set(before) - set(q.stored))
        self.assertEqual(q.index_visits, sum(uses[f] for f in loaded))

    def test_preload_restores_before_any_call(self):
        # preload restores one stored key per kernel function on synthetic operands and holds every stored key: the calls at
        # the stored sizes then restore nothing, harvest nothing, and replay bitwise
        w = torch.randn(64, 48, device="cuda", dtype=bf16)
        p = HarvestProvider()
        sizes = (32, 32, 40, 72, 72, 200, 200)
        self.calls(HostTraceReplay(self.f, opaque=(p,)), sizes, w)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "bindings.pickle")
            saved = p.save(path)
            q = HarvestProvider()
            self.assertEqual(q.load(path), saved)
        restored = q.preload()
        self.assertGreater(restored, 0)
        self.assertEqual((len(q.stored), len(q.bindings), q.stale), (0, saved, []))
        before = (q.restored, q.harvests)
        self.calls(HostTraceReplay(self.f, opaque=(q,)), sizes, w)
        self.assertEqual((q.restored, q.harvests), before)

    @parametrize("pattern", [0, 0x3C])
    def test_persist_padding(self, pattern):
        # a stored kernel's padding holds the harvest's smear and its unread
        # stack address capture A's frame; the restoring process's plain
        # capture has other bytes there (with `pattern`, its own smear): each
        # stored key restores, harvests nothing and replays bitwise
        ops = _padded_op()
        sizes = (1000, 1000, 3000, 5000, 7000, 7000)
        p = HarvestProvider(("extern",), extern_ops=(ops.scale_into.default,))
        r = HostTraceReplay(_scale_plus_one, opaque=(p,))
        for t in sizes:
            x, out = torch.randn(t, device="cuda"), torch.empty(t, device="cuda")
            self.assertEqual(r(x, out), x * 3 + 1, atol=0, rtol=0)
        self.assertEqual((p.refused, len(p.bindings)), ({}, 4))
        # Params' padding after n and scale, and its stack address
        for b in p.bindings.values():
            (n,) = b.nodes
            self.assertEqual(n.loose, ((0, 12, 4), (0, 28, 12)))
            self.assertEqual(n.images[0][12:16] + n.images[0][28:32], b"\xa5" * 8)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "bindings.pickle")
            saved = p.save(path)
            self.assertEqual(saved, 4)
            code = f"""
import sys, torch
sys.path.insert(0, {os.path.dirname(os.path.abspath(__file__))!r})
from test_cuda_host_trace_harvest import _padded_op, _scale_plus_one
from torch.cuda._host_trace_harvest import HarvestProvider
from torch.cuda._host_trace_replay import HostTraceReplay
ops = _padded_op()
ops.set_pattern({pattern})
p = HarvestProvider(("extern",), extern_ops=(ops.scale_into.default,))
loaded = p.load({path!r})
r = HostTraceReplay(_scale_plus_one, opaque=(p,))
for t in {sizes!r}:
    x, out = torch.randn(t, device="cuda"), torch.empty(t, device="cuda")
    assert torch.equal(r(x, out), x * 3 + 1) and torch.equal(out, x * 3)
print("ok", loaded, p.harvests, p.restored, p.stale, r.learned, r.traces, r.replays, r.eager)
"""
            out = subprocess.check_output([sys.executable, "-c", code], text=True)
        # the restoring process's trace restores its key out of band (oob_learn), a learn
        self.assertIn(f"ok {saved} 0 {saved} [] 1 1 {len(sizes) - 2} 1", out)

# A stand-in for trtllm-gen's paged attention launcher (ISSUES R8): a library
# op whose C++ host code picks the kernel (persistent or multi-CTA KV), its
# tile and its KV split from the batch (batch_size, max_q_len, the query's
# rows, the SM count) and passes them in a parameter struct. Each kernel
# writes its kind, split, tile, grid and scalar arguments into `report`, so
# a replay that ran a choice made for another shape differs from eager's.
_CHOICE_OP = r"""
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <torch/library.h>
#include <algorithm>

namespace {

struct Params {
  const __nv_bfloat16* q;
  const int* cu_seqlens_q;
  float* out;
  int* report;
  int batch_size, max_q_len, tokens, max_kv_len, heads, split;
};

template <int KIND, int TILE>
__global__ void attn(Params p) {
  int t = blockIdx.x, h = blockIdx.y, b = blockIdx.z / p.split, s = blockIdx.z % p.split;
  int start = p.cu_seqlens_q[b], end = p.cu_seqlens_q[b + 1];
  for (int i = threadIdx.x; i < TILE * 64; i += blockDim.x) {
    int row = start + t * TILE + i / 64, col = i % 64;
    if (row < end && col % p.split == s) {
      int at = (row * p.heads + h) * 64 + col;
      p.out[at] = __bfloat162float(p.q[at]) * (KIND * 16 + p.split) + TILE;
    }
  }
  if (t == 0 && h == 0 && blockIdx.z == 0 && threadIdx.x == 0) {
    int v[10] = {KIND, p.split, TILE, (int)gridDim.x, (int)gridDim.y, (int)gridDim.z, p.batch_size, p.max_q_len, p.tokens, p.max_kv_len};
    for (int i = 0; i < 10; ++i) p.report[i] = v[i];
  }
}

void mixed_attn(const at::Tensor& q, const at::Tensor& cu_seqlens_q, const at::Tensor& out, const at::Tensor& report,
                int64_t max_q_len, int64_t batch_size, int64_t max_kv_len) {
  int heads = q.size(1), tile = max_q_len <= 16 ? 16 : 64, tiles = (max_q_len + tile - 1) / tile;
  int64_t ctas = batch_size * tiles * heads, sm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  int split = std::max<int64_t>(1, std::min<int64_t>({4, sm / ctas, (max_kv_len + 1023) / 1024}));
  Params p{reinterpret_cast<const __nv_bfloat16*>(q.const_data_ptr()), cu_seqlens_q.const_data_ptr<int>(),
           out.mutable_data_ptr<float>(), report.mutable_data_ptr<int>(), (int)batch_size, (int)max_q_len,
           (int)q.size(0), (int)max_kv_len, heads, split};
  dim3 grid(tiles, heads, batch_size * split);
  auto stream = c10::cuda::getCurrentCUDAStream();
  if (split > 1) {
    tile == 16 ? attn<1, 16><<<grid, 128, 0, stream>>>(p) : attn<1, 64><<<grid, 128, 0, stream>>>(p);
  } else {
    tile == 16 ? attn<0, 16><<<grid, 128, 0, stream>>>(p) : attn<0, 64><<<grid, 128, 0, stream>>>(p);
  }
}

}  // namespace

TORCH_LIBRARY(ht_choice, m) {
  m.def("mixed_attn(Tensor q, Tensor cu_seqlens_q, Tensor(a!) out, Tensor(b!) report, SymInt max_q_len, SymInt batch_size, int max_kv_len) -> ()");
}

TORCH_LIBRARY_IMPL(ht_choice, CUDA, m) {
  m.impl("mixed_attn", &mixed_attn);
}
"""


def _choice_op():
    from torch.utils.cpp_extension import load_inline

    if not hasattr(torch.ops.ht_choice, "mixed_attn"):
        load_inline("ht_choice_mixed_attn", cpp_sources="", cuda_sources=_CHOICE_OP, is_python_module=False)
        torch.library.register_fake("ht_choice::mixed_attn", lambda *a: None)
    return torch.ops.ht_choice.mixed_attn.default


@unittest.skipIf(not TEST_CUDA, "CUDA not available")
class TestExternHostChoice(TestCase):
    @staticmethod
    def step(q, cu_seqlens_q, max_q_len):
        out = torch.empty(q.shape, device=q.device)
        report = torch.empty(10, device=q.device, dtype=torch.int32)
        _choice_op()(q, cu_seqlens_q, out, report, max_q_len, cu_seqlens_q.shape[0] - 1, 4096)
        return out, report

    @staticmethod
    def batch(nd, chunk):
        # nd decode requests (one token each) mixed with one prefill chunk
        lens = [1] * nd + ([chunk] if chunk else [])
        cu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], device="cuda", dtype=torch.int32)
        return torch.randn(sum(lens), 4, 64, device="cuda", dtype=bf16), cu, max(lens)

    def test_mixed_batch_choice_is_eagers(self):
        # at each (nd, chunk) point the replayed kernel, its split, tile, grid
        # and scalar arguments are eager's: one trace, each new key learned on
        # its miss, and a second sweep replays every point with no eager run
        points = [(1, 8), (8, 8), (16, 16), (32, 64), (2, 200), (64, 0), (4, 0), (8, 300), (1, 8)]
        p = HarvestProvider(("extern",), extern_ops=(_choice_op(),))
        r = HostTraceReplay(self.step, opaque=(p,))
        choices = set()
        for sweep in range(2):
            replays, eager, traces = r.replays, r.eager, r.traces
            for nd, chunk in points:
                q, cu, max_q = self.batch(nd, chunk)
                want = self.step(q, cu, max_q)
                got = r(q, cu, max_q)
                self.assertEqual(got[1], want[1], atol=0, rtol=0, msg=f"report at nd={nd} chunk={chunk}")
                self.assertEqual(got[0], want[0], atol=0, rtol=0)
                choices.add(tuple(want[1][:3].tolist()))
            if sweep:
                self.assertEqual((r.replays - replays, r.eager - eager, r.traces - traces), (len(points), 0, 0))
        self.assertEqual(len(p.refused), 0, p.refused)
        self.assertEqual((r.traces, r.eager), (1, 1))
        self.assertEqual(len(p.bindings), len(set(points)))
        # both kernels and several splits and tiles were taken
        self.assertEqual({c[0] for c in choices}, {0, 1})
        self.assertGreaterEqual(len(choices), 4)


if __name__ == "__main__":
    run_tests()
