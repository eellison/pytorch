# Owner(s): ["module: cuda"]

import statistics
import time
import unittest

from host_trace_h2d_probe import probe
from host_trace_testing import (
    assert_eager_function_handles,
    build,
    HostTraceTestCase,
    wait_for_h2d,
)

import torch
import torch.nn.functional as F
from torch.testing._internal.common_cuda import (
    PLATFORM_SUPPORTS_CUDNN_ATTENTION,
    PLATFORM_SUPPORTS_FLASH_ATTENTION,
    PLATFORM_SUPPORTS_MEM_EFF_ATTENTION,
)
from torch.testing._internal.common_utils import run_tests, skipIfRocm


if torch.cuda.is_available():
    import host_trace_two_hint as two_hint

    from torch.cuda import _host_trace as ht

C = torch._C

# one decode step, sized so the split heuristic and the flash kernel choice do
# not change over the cache lengths below (one 128-key block)
H, DH = 8, 64
D = H * DH
V = 1024
FF = 4096  # the MLP width: its down projection has K = 4096, where cuBLAS splits K at small M
LMAX = 64
DTYPE = torch.bfloat16


def rotate_half(x, neg_one):
    # the rotary rotation: cat([-x2, x1], -1) over the two halves of the head
    # dim (x1 a non-contiguous view; the negation is a mul by a device scalar)
    h = x.shape[-1] // 2
    return torch.cat([x[..., h:] * neg_one, x[..., :h]], -1)


def decode_step(
    ids,
    table,
    ln_w,
    ln_b,
    wq,
    wk,
    wv,
    w_up,
    w_down,
    w_out,
    cos,
    sin,
    neg_one,
    k_view,
    v_view,
):
    # ids: pinned int64 (B,); table (V, D); wq / wk / wv (D, D), w_up (FF, D),
    # w_down (D, FF), w_out (V, D); cos / sin (DH,); k_view / v_view
    # (B, H, L, DH) views of the caller's caches, position L - 1 is written
    B = ids.shape[0]
    L = k_view.shape[2]
    ids_dev = torch.empty(B, dtype=torch.int64, device="cuda")
    probe().copy_into(ids_dev, ids)  # the ids cross through the H2D path
    x = F.embedding(ids_dev, table)  # the traced index_select sibling
    h = F.layer_norm(x, (D,), ln_w, ln_b)  # the converted CUDA host
    q = F.linear(h, wq).view(B, H, 1, DH)  # projections: closed cuBLAS regions
    q = q * cos + rotate_half(q, neg_one) * sin  # rotary: mul / cat / add
    k = F.linear(h, wk).view(B, H, 1, DH)
    k = k * cos + rotate_half(k, neg_one) * sin
    k_view[:, :, L - 1 : L].copy_(k)  # cache write
    v_view[:, :, L - 1 : L].copy_(F.linear(h, wv).view(B, H, 1, DH))
    attn = F.scaled_dot_product_attention(q, k_view, v_view)  # SDPA -> flash
    y = F.silu(attn.reshape(B, D) + x)  # residual add, activation
    y = y + F.linear(F.silu(F.linear(y, w_up)), w_down)  # MLP: K = 4096 on the way down
    # a float32 softmax over the row and back: two cast copies (sibling) and
    # the converted softmax host
    p = torch.softmax(y.to(torch.float32), -1).to(DTYPE)
    logits = F.linear(p, w_out)  # the logit projection
    last = k_view[:, :, L - 1].sum(-1)  # last-position read (SymInt index)
    return logits, last


def _eager_kernels(fn, args):
    # the kernels one eager call launches, in order, by the profiler's name
    from torch.profiler import profile, ProfilerActivity

    fn(*args)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn(*args)
        torch.cuda.synchronize()
    return [
        e.name
        for e in prof.events()
        if e.device_type == torch.autograd.DeviceType.CUDA
        and not e.name.startswith(("Memset", "Memcpy"))
    ]


def _tape_kernels(tape):
    return [torch._C._demangle(L["kernel"]) for L in tape.launches]


def _assert_same_kernels(case, tape, fn, args):
    # fidelity: the tape's launches are the profiler's kernels of one eager
    # call, by name and in order; the sibling's copies name their functor
    # where eager's is a lambda, so for those the template and its integer
    # arguments (elementwise_kernel<128, 4, ...>) agree
    got, want = _tape_kernels(tape), _eager_kernels(fn, args)
    case.assertEqual(len(got), len(want))
    for g, w in zip(got, want):
        if "elementwise_kernel" in w:
            case.assertEqual(g.split(",")[:2], w.split(",")[:2])
        else:
            case.assertEqual(g, w)


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
@unittest.skipIf(not PLATFORM_SUPPORTS_FLASH_ATTENTION, "flash attention not supported")
@skipIfRocm(msg="host tracing is CUDA-only in this version")
class TestCudaHostTraceDecode(HostTraceTestCase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(0)
        self.table = torch.randn(V, D, device="cuda", dtype=DTYPE)
        self.ln_w = 1 + 0.1 * torch.randn(D, device="cuda", dtype=DTYPE)
        self.ln_b = 0.1 * torch.randn(D, device="cuda", dtype=DTYPE)
        self.wq, self.wk, self.wv = (
            torch.randn(D, D, device="cuda", dtype=DTYPE) / D**0.5 for _ in range(3)
        )
        self.w_up = torch.randn(FF, D, device="cuda", dtype=DTYPE) / D**0.5
        self.w_down = torch.randn(D, FF, device="cuda", dtype=DTYPE) / FF**0.5
        self.w_out = torch.randn(V, D, device="cuda", dtype=DTYPE) / D**0.5
        # rotary tables for one position, laid out as [half, half] over DH
        ang = torch.arange(DH // 2, device="cuda", dtype=torch.float32) * 0.1
        self.cos = torch.cat([ang.cos(), ang.cos()]).to(DTYPE)
        self.sin = torch.cat([ang.sin(), ang.sin()]).to(DTYPE)
        self.neg_one = torch.tensor(-1.0, device="cuda", dtype=DTYPE)
        # the default backend order on the box may prefer cuDNN, whose host is
        # not converted; a decode pins flash the same way
        self._flash = torch.nn.attention.sdpa_kernel(
            torch.nn.attention.SDPBackend.FLASH_ATTENTION
        )
        self._flash.__enter__()
        # the variants built from one tape, by the first one's identity
        self._pool: dict[int, list] = {}

    def tearDown(self):
        self._flash.__exit__(None, None, None)
        super().tearDown()

    def _caches(self, B):
        k = torch.randn(B, H, LMAX, DH, device="cuda", dtype=DTYPE)
        v = torch.randn(B, H, LMAX, DH, device="cuda", dtype=DTYPE)
        return k, v

    def _ids(self, B):
        return torch.randint(0, V, (B,), dtype=torch.int64).pin_memory()

    def _args(self, ids, caches, L):
        k, v = caches
        return (
            ids,
            self.table,
            self.ln_w,
            self.ln_b,
            self.wq,
            self.wk,
            self.wv,
            self.w_up,
            self.w_down,
            self.w_out,
            self.cos,
            self.sin,
            self.neg_one,
            k[:, :, :L],
            v[:, :, :L],
        )

    def _trace(self, B, L, **kw):
        caches = self._caches(B)
        args = self._args(self._ids(B), caches, L)
        tape = ht.trace(decode_step, args, **kw)
        variant = build(tape, decode_step, args)
        return tape, variant

    def _serve(self, variant, args):
        # what an entry does with one tape's variants: the first that serves
        # the call does; a TopologyMiss (a projection's cuBLAS node chain at
        # this batch is not the one a variant's graph holds) names the tape,
        # which built at these inputs (no trace) is a variant with that chain,
        # kept beside the others; any other Miss is the call's (the eager form
        # holds no chain: one variant serves every batch its guards admit)
        variants = self._pool.setdefault(id(variant), [variant])
        topology, plain = None, None
        for v in variants:
            try:
                return v.replay(args)
            except ht.TopologyMiss as e:
                topology = e
            except ht.Miss as e:
                plain = e
        if topology is None:
            raise plain
        variants.append(build(topology.tape, decode_step, args))
        return variants[-1].replay(args)

    def _step(self, variant, B, L, eager_caches, replay_caches):
        # the same ids and the same cache contents on both sides; the replay
        # writes position L - 1 of the caller's cache like the ordinary call
        ids = self._ids(B)
        want = decode_step(*self._args(ids, eager_caches, L))
        try:
            got = self._serve(variant, self._args(ids, replay_caches, L))
        except ht.Miss:
            return False
        for w, g in zip(want, got):
            self._assert_bitwise(g, w)
        for e, r in zip(eager_caches, replay_caches):
            self.assertTrue(torch.equal(e[:, :, :L], r[:, :, :L]))
        return True

    def _run_steps(self, variant, B, lengths):
        e_caches = self._caches(B)
        r_caches = tuple(c.clone() for c in e_caches)
        served = [L for L in lengths if self._step(variant, B, L, e_caches, r_caches)]
        return served

    def test_the_chain_traces_end_to_end(self):
        tape, _ = self._trace(4, 16)
        # the ids' H2D copy, embedding (index_select), layer norm, two
        # rotaries (mul, mul, cat as two copies, mul, add), two cache copies,
        # flash, add, silu, silu, add, two casts, softmax, a sum: every launch
        # a converted or sibling host; the six projections are closed regions
        self.assertEqual(tape.num_memcpys, 1)  # the ids
        self.assertEqual(tape.num_regions, 6)
        self.assertGreaterEqual(tape.num_launches, 18)
        names = " ".join(rec["kernel"] for rec in tape.launches)
        for needle in (
            "layer_norm",
            "flash_fwd",
            "indexSelectSmallIndex",
            "reduce_kernel",
            "softmax_warp_forward",
        ):
            self.assertIn(needle, names)

    def test_steps_of_a_growing_cache_serve_without_new_pins(self):
        tape, variant = self._trace(4, 16)
        served = self._run_steps(variant, 4, range(16, 41))
        self.assertEqual(served, list(range(16, 41)))
        # the guard set is the tape's, nothing accumulates per step
        self.assertEqual(tape.num_guards, len(tape.guards))

    def test_other_batch_sizes_from_one_trace(self):
        _, variant = self._trace(4, 16)
        for B in (3, 8):
            served = self._run_steps(variant, B, range(20, 26))
            self.assertEqual(served, list(range(20, 26)), f"batch {B}")
        # batch 1 is a size-1 branch of the elementwise sibling (dimension
        # coalescing changes the launch): the batch-4 tape misses it by name
        caches = self._caches(1)
        with self.assertRaisesRegex(ht.Miss, "!= 1"):
            variant.replay(self._args(self._ids(1), caches, 20))

    def test_batch_sweep_across_gemm_variants(self):
        # one batch-4 tape over B = 1..64: each B is served bitwise or missed
        # by name. Within the tape's own batch guard (an earlier host pins
        # B <= 16) the projections cross the cuBLAS tile-M boundaries at 12
        # and 16 in place; where a projection's node chain changes the same
        # tape is built again at that batch into another exec (_serve), one
        # exec per chain vector (on this box the K = 4096 down projection
        # keeps its split-K chain up to B = 24, so one exec serves 2..16);
        # batch 1 misses on the elementwise sibling's coalescing
        _, variant = self._trace(4, 16)
        h0 = ht.gemm_harvests()
        served, missed = [], {}
        for B in range(1, 65):
            e_caches = self._caches(B)
            r_caches = tuple(c.clone() for c in e_caches)
            try:
                self._serve(variant, self._args(self._ids(B), r_caches, 20))
            except ht.Miss as e:
                missed[B] = str(e).split(":")[0][:50]
                continue
            self.assertTrue(self._step(variant, B, 20, e_caches, r_caches))
            served.append(B)
        variants = self._pool[id(variant)]
        reasons = sorted(set(missed.values()))
        if variant.native is not None:
            stats = [v.native.region_stats() for v in variants]
            chains = [
                tuple(
                    tuple(zip(s["kinds"], s.get("programmatic", [False] * s["nodes"])))
                    for s in st["sites"]
                )
                for st in stats
            ]
            print(
                f"\n[decode batch sweep 1..64] served {served} missed {sorted(missed)} reasons {reasons} "
                f"harvests {ht.gemm_harvests() - h0} variants {len(variants)} "
                f"chains {[[len(t) for t in c] for c in chains]} "
                f"applies {[s['applies'] for s in stats]} "
                f"graph updates {[s['graph_updates'] for s in stats]}"
            )
            # one variant per chain vector
            self.assertEqual(len(set(chains)), len(variants))
        else:
            print(
                f"\n[decode batch sweep 1..64] served {served} missed {sorted(missed)} reasons {reasons}"
            )
        self.assertEqual(served, list(range(2, 17)))
        self.assertIn(1, missed)
        self.assertEqual(sorted(k for k in missed if k > 16), list(range(17, 65)))

    def test_batch_one_has_its_own_tape(self):
        _, variant = self._trace(1, 16)
        served = self._run_steps(variant, 1, range(16, 30))
        self.assertEqual(served, list(range(16, 30)))
        caches = self._caches(4)
        with self.assertRaisesRegex(ht.Miss, "== 1"):
            variant.replay(self._args(self._ids(4), caches, 20))

    def test_two_variants_alive_interleaved(self):
        _, v4 = self._trace(4, 16)
        _, v2 = self._trace(2, 16)
        e4, r4 = self._caches(4), None
        r4 = tuple(c.clone() for c in e4)
        e2 = self._caches(2)
        r2 = tuple(c.clone() for c in e2)
        for L in range(17, 33):
            self.assertTrue(self._step(v4, 4, L, e4, r4))
            self.assertTrue(self._step(v2, 2, L, e2, r2))

    def test_without_warm_up(self):
        _, variant = self._trace(4, 16, warm_up=False)
        served = self._run_steps(variant, 4, range(16, 24))
        self.assertEqual(served, list(range(16, 24)))

    def test_step_cost(self):
        _, variant = self._trace(4, 16)
        caches = self._caches(4)
        args = self._args(self._ids(4), caches, 24)
        for _ in range(5):
            decode_step(*args)
            variant.replay(args)
        torch.cuda.synchronize()

        def timed(fn, n=50):
            xs = []
            for _ in range(n):
                t = time.perf_counter()
                fn()
                xs.append((time.perf_counter() - t) * 1e6)
            torch.cuda.synchronize()
            return statistics.median(xs)

        eager = timed(lambda: decode_step(*args))
        replay = timed(lambda: variant.replay(args))
        # stepping the cache length: every step patches the kernels' bytes
        lengths = iter(range(17, 41))

        def stepping():
            L = next(lengths)
            variant.replay(self._args(args[0], caches, L))

        step_replay = timed(stepping, n=24)
        print(
            f"\n[decode step CPU us] eager {eager:.1f} replay fixed L {replay:.1f} "
            f"replay stepping L {step_replay:.1f}"
        )
        self.assertGreater(eager, 0)

    def test_in_place_pinned_ids_need_wait_for_h2d(self):
        # A decode loop written the natural way keeps one pinned ids buffer and
        # rewrites it per step. The replay's copy reads that buffer
        # asynchronously, so a caller running ahead of the GPU must call
        # wait_for_h2d(ids) before each rewrite; nothing detects a missing
        # wait (a contract, like the synchronous-copy rule). A fresh pinned
        # tensor per step needs no wait: the replay holds it until its copy
        # ran. (The eager form copies nothing asynchronously.)
        _, variant = self._trace(4, 16)
        big = torch.randn(8192, 8192, device="cuda", dtype=DTYPE)

        def run(pattern):
            e_caches = self._caches(4)
            r_caches = tuple(c.clone() for c in e_caches)
            ids = self._ids(4)
            pairs = []
            for L in range(17, 27):
                if pattern == "in_place_with_wait":
                    wait_for_h2d(ids)
                if pattern == "fresh":
                    ids = self._ids(4)
                else:
                    ids.copy_(torch.randint(0, V, (4,), dtype=torch.int64))
                # the ordinary call takes its own pinned copy of this step's ids
                want = decode_step(*self._args(ids.clone().pin_memory(), e_caches, L))
                for _ in range(8):  # keep the GPU behind the CPU
                    big @ big
                got = variant.replay(self._args(ids, r_caches, L))
                pairs.append(([g.clone() for g in got], want))
            torch.cuda.synchronize()
            wrong = sum(
                1 for g, w in pairs if not all(torch.equal(a, b) for a, b in zip(g, w))
            )
            caches_equal = all(
                torch.equal(e[:, :, :26], r[:, :, :26])
                for e, r in zip(e_caches, r_caches)
            )
            return wrong, caches_equal

        wrong, caches_equal = run("in_place_with_wait")
        self.assertEqual(wrong, 0)
        self.assertTrue(caches_equal)
        wrong, caches_equal = run("fresh")
        self.assertEqual(wrong, 0)
        self.assertTrue(caches_equal)
        # without the wait the same loop is a race: the copy may read the next
        # step's ids. Observed, not asserted (a race is not a deterministic test).
        wrong, _ = run("in_place_without_wait")
        print(
            f"\n[in-place pinned ids without wait_for_h2d] {wrong}/10 steps read later ids"
        )

    @unittest.skipIf(
        not PLATFORM_SUPPORTS_CUDNN_ATTENTION, "cuDNN attention not supported"
    )
    def test_the_selector_reads_a_masked_calls_shape_as_eager_does(self):
        # check_attn_mask_shape (sdp_utils_cpp.h) compares the mask's dims with
        # the query's, the key's, the batch and the heads. Under a trace the
        # mask's dims are its own symbols, and a "statically equal or
        # concretely 1" test (E42's dodge) rejected the cuDNN arm eager takes,
        # so the trace went on to the next backend: another program. Now a
        # hinted dim is compared as the concrete one is, a guard, and the trace
        # takes eager's arm: today a decline naming the unconverted cuDNN host
        # (the cuDNN region, once it lands, keeps the other arms' kernels off
        # the tape)
        q, k, v = (
            torch.randn(2, H, 8, DH, device="cuda", dtype=DTYPE) for _ in range(3)
        )
        mask = torch.zeros(2, 1, 8, 8, device="cuda", dtype=DTYPE)
        args = (q, k, v, mask)
        cudnn = int(torch.nn.attention.SDPBackend.CUDNN_ATTENTION)
        # the class pins flash for the decode chain (setUp); this call is the
        # selector's own choice among every backend
        backends = [
            torch.nn.attention.SDPBackend.CUDNN_ATTENTION,
            torch.nn.attention.SDPBackend.FLASH_ATTENTION,
            torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
            torch.nn.attention.SDPBackend.MATH,
        ]
        with torch.nn.attention.sdpa_kernel(backends):
            if torch._fused_sdp_choice(*args) != cudnn:
                self.skipTest(
                    "eager's selector does not give this masked call to cuDNN"
                )
            try:
                tape = ht.trace(F.scaled_dot_product_attention, args)
            except ht.Declined as e:
                self.assertIn("cudnn", str(e))
            else:
                other = [
                    k for k in _tape_kernels(tape) if "fmha" in k or "softmax" in k
                ]
                self.assertEqual(other, [])

    @unittest.skipIf(
        not PLATFORM_SUPPORTS_CUDNN_ATTENTION, "cuDNN attention not supported"
    )
    def test_the_cudnn_sdpa_route_is_a_closed_region(self):
        # the default backend order on this box picks cuDNN attention: a closed
        # region beside the six projections (torch/cuda/_host_trace_cudnn.py),
        # served bitwise over the cache lengths as the flash chain is
        caches = self._caches(4)
        args = self._args(self._ids(4), caches, 16)
        with torch.nn.attention.sdpa_kernel(
            torch.nn.attention.SDPBackend.CUDNN_ATTENTION
        ):
            tape = ht.trace(decode_step, args)
            self.assertEqual([r.op for r in tape.regions].count("cudnn_sdpa"), 1)
            self.assertEqual(tape.num_regions, 7)
            self.assertNotIn(
                "flash_fwd", " ".join(rec["kernel"] for rec in tape.launches)
            )
            variant = build(tape, decode_step, args)
            served = self._run_steps(variant, 4, range(16, 25))
            self.assertEqual(served, list(range(16, 25)))

    def test_odd_head_dims_pad_through_the_composite(self):
        # flash takes head dims in multiples of 8: the SDPA composite pads the
        # inputs (constant_pad_nd, a CompositeExplicit body on the int
        # signature, which the mode runs with its ints pinned) and slices the
        # output; the pad's fill and copy are on the tape as eager launches
        # them. The body reads its sizes as ints, so the tape is that shape's
        # (another batch misses by name); a symbolic constant_pad_nd body
        # would lift the pins
        def sdpa(q, k, v):
            return F.scaled_dot_product_attention(q, k, v)

        def case(B, seed):
            torch.manual_seed(seed)
            return tuple(
                torch.randn(B, H, 16, 100, device="cuda", dtype=DTYPE) for _ in range(3)
            )

        args = case(2, 0)
        tape = ht.trace(sdpa, args)
        self.assertIn(
            "flash_fwd_kernel", " ".join(rec["kernel"] for rec in tape.launches)
        )
        variant = build(tape, sdpa, args)
        new = case(2, 1)
        self._assert_bitwise(variant.replay(new)[0], sdpa(*new))
        with self.assertRaises(ht.Miss):
            variant.replay(case(3, 2))

    def test_expected_named_misses_of_the_sdpa_path(self):
        # the documented over-pins of the flash host, each a named miss at a
        # boundary the kernel or the host branches on (never a wrong result):
        #   seqlen_k % 128 (the Is_even_MN template)   -> miss at multiples of 128
        #   seqlen_q != 1 (the host's causal flip)     -> miss at a single query
        #   is_causal square lengths (SDPA-level)       -> eager refuses Lq != Lk
        #   a shape-derived window size                 -> pins (the aten op)
        #   batch 1 (elementwise sibling coalescing)    -> test_other_batch_sizes
        def sdpa(q, k, v):
            return F.scaled_dot_product_attention(q, k, v)

        def qkv(lq, lk):
            return tuple(
                torch.randn(2, H, n, DH, device="cuda", dtype=DTYPE)
                for n in (lq, lk, lk)
            )

        args = qkv(32, 32)
        tape = ht.trace(sdpa, args)
        variant = build(tape, sdpa, args)
        for lq, lk in ((32, 48), (48, 48), (33, 47)):
            a = qkv(lq, lk)
            self.assertTrue(torch.equal(variant.replay(a)[0], sdpa(*a)), f"{lq}/{lk}")
        with self.assertRaisesRegex(ht.Miss, "% 128"):
            variant.replay(qkv(32, 128))
        with self.assertRaisesRegex(ht.Miss, "!= 1"):
            variant.replay(qkv(1, 32))

    def test_what_a_real_decode_still_needs(self):
        # the ops a real decode reaches, each traced since the commit named
        # (an op this stack does not trace declines by name, never a wrong
        # result; nothing of the decode path is left in this list)
        x = torch.randn(4, D, device="cuda", dtype=DTYPE)
        w2 = torch.randn(D, D, device="cuda", dtype=DTYPE)
        ids = torch.randint(0, V, (4,), device="cuda")
        # the 2-D GEMM is a closed region since commit 10, and so is the batched
        # one (bmm over expanded operands: batch stride 0)
        self.assertEqual(ht.trace(lambda a, b: a @ b, (x, w2)).num_regions, 1)
        tape = ht.trace(lambda a, b: a @ b, (x.expand(2, 4, D), w2.expand(2, D, D)))
        self.assertEqual((tape.num_regions, tape.num_launches), (1, 0))
        # softmax, dtype casts and sin / cos trace since commit 7
        for fn in (
            lambda a: torch.softmax(a, -1),
            lambda a: a.float(),
            lambda a: a.sin() * a.cos(),
        ):
            self.assertGreaterEqual(ht.trace(fn, (x,)).num_launches, 1)
        # traced since commit 8: embedding (the index_select sibling) and cat
        tape = ht.trace(lambda t, i: F.embedding(i, t), (self.table, ids))
        self.assertEqual(tape.num_launches, 1)
        tape = ht.trace(lambda a: torch.cat([a, a], -1), (x,))
        self.assertEqual(tape.num_launches, 1)

    def _sdpa_fwd_bwd(self, p):
        # create_graph=False, as loss.backward() runs the backward (the flash
        # backward launches the same kernels either way; the traced step and
        # the profiled eager step run the same way regardless)
        def fwd_bwd(q, k, v, cot):
            leaves = [t.detach().requires_grad_(True) for t in (q, k, v)]
            out = F.scaled_dot_product_attention(*leaves, dropout_p=p, is_causal=True)
            gq, gk, gv = torch.autograd.grad(out, leaves, cot, create_graph=False)
            return out.detach(), gq, gk, gv

        return fwd_bwd

    def _sdpa_case(self, B, L, seed):
        torch.manual_seed(seed)
        q, k, v = (
            torch.randn(B, H, L, DH, device="cuda", dtype=DTYPE) for _ in range(3)
        )
        return (q, k, v, torch.randn(B, H, L, DH, device="cuda", dtype=DTYPE))

    def test_forward_and_backward_in_one_trace(self):
        # torch.autograd.grad on the main thread: the SDPA flash forward, then
        # the engine runs the backward node on its device worker thread, where
        # the trace mode brings the recorder along; the tape holds the forward
        # launch, the two transpose copies the backward entry makes (the
        # cotangent and the saved output are (B, H, L, D) contiguous) and the
        # backward's three launches, the kernels eager launches, with the
        # sequence lengths symbolic through the _symint backward kernels, and
        # replays the gradients bitwise at other shapes (batch 1 is a guard of
        # the copies' iterator)
        fwd_bwd = self._sdpa_fwd_bwd(0.0)
        args = self._sdpa_case(2, 128, 0)
        tape = ht.trace(fwd_bwd, args)
        names = " ".join(rec["kernel"] for rec in tape.launches)
        self.assertEqual(tape.num_launches, 6)
        for needle in (
            "flash_fwd_kernel",
            "Copy",
            "flash_bwd_dot_do_o",
            "seqk_parallel",
            "convert_dq",
        ):
            self.assertIn(needle, names)
        _assert_same_kernels(self, tape, fwd_bwd, args)
        self.assertIsNone(tape.rng_increment)
        variant = build(tape, fwd_bwd, args)
        for B, L, seed in [(2, 128, 1), (3, 256, 2), (4, 128, 3)]:
            new_args = self._sdpa_case(B, L, seed)
            got = variant.replay(new_args)
            want = fwd_bwd(*new_args)
            torch.cuda.synchronize()
            for g, w in zip(got, want):
                self.assertEqual(g.shape, w.shape)
                self.assertTrue(torch.equal(g, w))

    def test_forward_and_backward_with_dropout_in_one_trace(self):
        # the forward declares its philox increment (b * h * 32); the backward
        # re-derives the mask from the rng_state the forward kernel stored and
        # declares nothing: per replay the generator advances by one eager
        # call's amount and the gradients equal eager's from the same seed
        fwd_bwd = self._sdpa_fwd_bwd(0.1)
        args = self._sdpa_case(2, 128, 0)
        tape = ht.trace(fwd_bwd, args)
        self.assertIsNotNone(tape.rng_increment)
        _assert_same_kernels(self, tape, fwd_bwd, args)
        variant = build(tape, fwd_bwd, args)
        gen = torch.cuda.default_generators[torch.cuda.current_device()]
        for B, L, seed in [(2, 128, 1), (3, 256, 2), (4, 128, 3)]:
            new_args = self._sdpa_case(B, L, seed)
            torch.manual_seed(seed)
            before = gen.get_offset()
            want = fwd_bwd(*new_args)
            advance = gen.get_offset() - before
            self.assertEqual(advance, 32 * B * H)
            torch.manual_seed(seed)
            got = variant.replay(new_args)
            torch.cuda.synchronize()
            self.assertEqual(gen.get_offset() - before, advance)
            for g, w in zip(got, want):
                self.assertTrue(torch.equal(g, w))

    def test_every_case_traces_the_same_program_under_other_hints(self):
        # the recorder never reads a hint: every trace this class makes, made
        # again under other hints, is the same program (host_trace_two_hint)
        two_hint.assert_family(self)


# ---- the memory-efficient attention host (the CUTLASS fmha forward and
# backward), eager's own route for an additive 4-D mask (flash takes none): a
# padded batch under SDPA. The kernel-choice ladder of _efficient_attention_forward
# reads the head dims, the strides and the addresses' alignment classes, all
# guards under a trace. On recent GPUs cuDNN attention sits ahead of it in
# eager's order for the smaller head dims and its host is not converted, so
# the tests run under the selector's own order without cuDNN (what a caller
# with a padded batch uses), except where they read the selector itself.
EFF_H, EFF_DH = 4, 256
NO_CUDNN = [
    torch.nn.attention.SDPBackend.FLASH_ATTENTION,
    torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
    torch.nn.attention.SDPBackend.MATH,
]
EFFICIENT = int(torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION)
CUDNN = int(torch.nn.attention.SDPBackend.CUDNN_ATTENTION)


def masked_sdpa(q, k, v, mask):
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


def masked_fwd_bwd(q, k, v, mask, cot):
    leaves = [t.detach().requires_grad_(True) for t in (q, k, v)]
    out = F.scaled_dot_product_attention(*leaves, attn_mask=mask)
    gq, gk, gv = torch.autograd.grad(out, leaves, cot, create_graph=False)
    return out.detach(), gq, gk, gv


def _padded_mask(lengths, lq, lk, dtype):
    # a left-padded batch: row b attends to its last lengths[b] keys (0 / min
    # additive, transformers' 4-D form, one head broadcast to all)
    m = torch.zeros(len(lengths), 1, lq, lk, device="cuda", dtype=dtype)
    for b, n in enumerate(lengths):
        m[b, :, :, : lk - n] = torch.finfo(dtype).min
    return m


def _masked_case(lengths, lq, lk, dh=EFF_DH, dtype=DTYPE, seed=0):
    torch.manual_seed(seed)
    B = len(lengths)
    q, k, v = (
        torch.randn(B, EFF_H, n, dh, device="cuda", dtype=dtype) for n in (lq, lk, lk)
    )
    return q, k, v, _padded_mask(lengths, lq, lk, dtype)


def _bwd_case(lengths, n, seed, cot_layout="bhld"):
    # the cotangent in the (B, H, L, D) layout of the output (the backward
    # entry copies it to (B, L, H, D)), or already in that layout as a view
    q, k, v, mask = _masked_case(lengths, n, n, seed=seed)
    cot = torch.randn_like(q)
    if cot_layout == "blhd":
        B, H, L, D = q.shape
        cot = torch.randn(B, L, H, D, device="cuda", dtype=q.dtype).transpose(1, 2)
    return (q, k, v, mask, cot)


def without_cudnn():
    return torch.nn.attention.sdpa_kernel(NO_CUDNN)


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
@unittest.skipIf(
    not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION, "memory-efficient attention not supported"
)
@skipIfRocm(msg="host tracing is CUDA-only in this version")
class TestCudaHostTraceEfficientAttention(HostTraceTestCase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(0)

    def test_the_selector_reaches_the_kernel_by_itself_for_a_masked_call(self):
        # with no pin at all: eager's own selector gives a masked call at this
        # head dim to the memory-efficient kernel (flash refuses the mask) or,
        # on the GPUs that prefer it, to cuDNN attention, whose host declines
        # by name; never a wrong result
        args = _masked_case([32, 29, 17, 5], 1, 32)
        choice = torch._fused_sdp_choice(*args)
        if choice == EFFICIENT:
            tape = ht.trace(masked_sdpa, args)
            self.assertIn("fmha_cutlassF", tape.launches[0]["kernel"])
        else:
            self.assertEqual(choice, CUDNN)
            with self.assertRaisesRegex(ht.Declined, "cudnn"):
                ht.trace(masked_sdpa, args)

    def test_a_masked_decode_call_traces_and_replays_at_other_lengths(self):
        # one tape at (B=4, Lq=1, Lk=32) serves other key lengths and batch
        # sizes bitwise; the tape's kernel is the profiler's fmha forward of
        # the same eager call (the two seed / offset zeros are memsets); the
        # aligned mask is not padded, as eager does not pad it
        with without_cudnn():
            args = _masked_case([32, 29, 17, 5], 1, 32)
            tape = ht.trace(masked_sdpa, args)
            self.assertEqual(tape.num_launches, 1)
            self.assertIn("fmha_cutlassF", tape.launches[0]["kernel"])
            self.assertEqual(len(tape.memsets), 2)
            _assert_same_kernels(self, tape, masked_sdpa, args)
            variant = build(tape, masked_sdpa, args)
            cases = (
                ([48, 40, 9, 1], 48),
                ([64, 3], 64),
                ([16] * 8, 16),
                ([32, 20, 20, 31, 2, 7], 32),
            )
            for lengths, lk in cases:
                new = _masked_case(lengths, 1, lk, seed=lk + len(lengths))
                self._assert_bitwise(
                    variant.replay(new)[0], masked_sdpa(*new), f"{lengths}"
                )

    def test_the_named_misses_of_the_host(self):
        # the branches the kernel ladder and the host's checks take: batch 1
        # (the bias stride checks are skipped at one batch), more than one
        # query (the query-block grid and the strideM check), a key length
        # the composite pads (the mask's stride alignment class)
        with without_cudnn():
            args = _masked_case([32, 29, 17, 5], 1, 32)
            variant = build(ht.trace(masked_sdpa, args), masked_sdpa, args)
            for lengths, lq, lk in (([32], 1, 32), ([32, 5], 2, 32), ([20, 7], 1, 20)):
                with self.assertRaises(ht.Miss, msg=f"{lengths} {lq} {lk}"):
                    variant.replay(_masked_case(lengths, lq, lk))

    def test_a_prefill_call_with_a_square_mask(self):
        # Lq = Lk (a padded prefill): the query-block grid and the bias strideM
        # check are on the tape; other lengths and batch sizes serve bitwise
        with without_cudnn():
            args = _masked_case([32, 32, 20], 32, 32)
            tape = ht.trace(masked_sdpa, args)
            self.assertEqual(tape.num_launches, 1)
            variant = build(tape, masked_sdpa, args)
            for lengths, n in (([64, 10], 64), ([128] * 3, 128), ([48, 1, 48, 30], 48)):
                new = _masked_case(lengths, n, n, seed=n)
                self._assert_bitwise(
                    variant.replay(new)[0], masked_sdpa(*new), f"{lengths}"
                )

    def test_a_smaller_head_dim_reaches_the_kernel_without_cudnn(self):
        # the Llama-family head dims under the selector's order without cuDNN:
        # flash refuses the mask, the memory-efficient kernel takes it; the
        # 64x64 kernel of the ladder (kMaxK 64) rather than the head-dim-256 one
        with without_cudnn():
            args = _masked_case([32, 29, 17, 5], 1, 32, dh=64)
            tape = ht.trace(masked_sdpa, args)
            self.assertIn("64x64", tape.launches[0]["kernel"])
            variant = build(tape, masked_sdpa, args)
            new = _masked_case([48, 2, 48], 1, 48, dh=64, seed=3)
            self._assert_bitwise(variant.replay(new)[0], masked_sdpa(*new))

    def test_float32_traces_under_eagers_own_selection(self):
        # float32 has no flash or cuDNN route: the selector reaches the
        # memory-efficient kernel with no pin (the f32 aligned kernels)
        args = _masked_case([32, 29, 17, 5], 1, 32, dh=64, dtype=torch.float32)
        self.assertEqual(torch._fused_sdp_choice(*args), EFFICIENT)
        tape = ht.trace(masked_sdpa, args)
        self.assertIn("f32", tape.launches[0]["kernel"])
        variant = build(tape, masked_sdpa, args)
        new = _masked_case([64, 7], 1, 64, dh=64, dtype=torch.float32, seed=5)
        self._assert_bitwise(variant.replay(new)[0], masked_sdpa(*new))

    def test_the_replay_launches_eagers_function(self):
        # E36: the tape's kernel is the node of an eager capture of the same
        # call, by name here and by function handle on the native replay
        with without_cudnn():
            args = _masked_case([32, 29, 17, 5], 1, 32)
            assert_eager_function_handles(self, masked_sdpa, args, launches=1)

    def test_dropout_declines_by_name(self):
        # the generator increment of the fmha forward is not recorded in this
        # commit: a decline naming the host, before the generator is advanced
        q, k, v, mask = _masked_case([32, 29, 17, 5], 1, 32)

        def dropout_sdpa(q, k, v, mask):
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=0.1
            )

        with without_cudnn():
            with self.assertRaisesRegex(
                ht.Declined, "memory-efficient attention with dropout"
            ):
                ht.trace(dropout_sdpa, (q, k, v, mask))

    def test_forward_and_backward_in_one_trace(self):
        # torch.autograd.grad through the masked forward: the engine runs the
        # backward node on its device worker thread, where the trace mode
        # brings the recorder along; the tape holds the fmha forward, the
        # copy the backward entry makes (the cotangent transposed contiguous)
        # and the fmha backward, the kernels eager launches, the lengths
        # symbolic through the _symint backward kernel; the gradients replay
        # bitwise at other lengths and batch sizes
        with without_cudnn():
            args = _bwd_case([32, 32, 20], 32, 0)
            tape = ht.trace(masked_fwd_bwd, args)
            names = " ".join(rec["kernel"] for rec in tape.launches)
            self.assertIn("fmha_cutlassF", names)
            self.assertIn("fmha_cutlassB", names)
            _assert_same_kernels(self, tape, masked_fwd_bwd, args)
            variant = build(tape, masked_fwd_bwd, args)
            for lengths, n, seed in (
                ([32, 32, 20], 32, 1),
                ([64, 9], 64, 2),
                ([48] * 4, 48, 3),
            ):
                new = _bwd_case(lengths, n, seed)
                got, want = variant.replay(new), masked_fwd_bwd(*new)
                for g, w in zip(got, want):
                    self._assert_bitwise(g, w, f"{lengths} {n}")

    def test_the_backward_launches_eagers_function(self):
        # E36 over the forward and the backward: the two fmha kernels are the
        # nodes of an eager capture (the cotangent comes in the layout the
        # backward reads, so the entry's strided copy, whose function object is
        # the copy commit's, is not on this tape)
        with without_cudnn():
            args = _bwd_case([32, 32, 20], 32, 0, cot_layout="blhd")
            assert_eager_function_handles(self, masked_fwd_bwd, args, launches=2)

    def test_every_case_traces_the_same_program_under_other_hints(self):
        # the recorder never reads a hint: every trace this class makes, made
        # again under other hints, is the same program (host_trace_two_hint)
        two_hint.assert_family(self)


if __name__ == "__main__":
    run_tests()
