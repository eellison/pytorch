# Owner(s): ["module: cuda"]

import statistics
import time
import unittest

from host_trace_h2d_probe import probe
from host_trace_testing import HostTraceTestCase

import torch
import torch.nn.functional as F
from torch.testing._internal.common_cuda import (
    PLATFORM_SUPPORTS_CUDNN_ATTENTION,
    PLATFORM_SUPPORTS_FLASH_ATTENTION,
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
LMAX = 64
DTYPE = torch.bfloat16


def rotate_half(x, neg_one):
    # the rotary rotation: cat([-x2, x1], -1) over the two halves of the head
    # dim (x1 a non-contiguous view; the negation is a mul by a device scalar)
    h = x.shape[-1] // 2
    return torch.cat([x[..., h:] * neg_one, x[..., :h]], -1)


def decode_step(
    ids, table, ln_w, ln_b, wq, wk, wv, w_out, cos, sin, neg_one, k_view, v_view
):
    # ids: pinned int64 (B,); table (V, D); the weights (D,); cos / sin (DH,);
    # k_view / v_view (B, H, L, DH) views of the caller's caches, position L - 1
    # is written
    B = ids.shape[0]
    L = k_view.shape[2]
    ids_dev = torch.empty(B, dtype=torch.int64, device="cuda")
    probe().copy_into(ids_dev, ids)  # the ids cross through the H2D path
    x = F.embedding(ids_dev, table)  # the traced index_select sibling
    h = F.layer_norm(x, (D,), ln_w, ln_b)  # the converted CUDA host
    q = (h * wq).view(B, H, 1, DH)  # projection stand-ins: opted elementwise
    q = q * cos + rotate_half(q, neg_one) * sin  # rotary: mul / cat / add
    k = h.mul(wk).view(B, H, 1, DH)
    k = k * cos + rotate_half(k, neg_one) * sin
    k_view[:, :, L - 1 : L].copy_(k)  # cache write
    v_view[:, :, L - 1 : L].copy_(h.mul(wv).view(B, H, 1, DH))
    attn = F.scaled_dot_product_attention(q, k_view, v_view)  # SDPA -> flash
    y = F.silu(attn.reshape(B, D) + x)  # residual add, activation
    # a float32 softmax over the row and back: two cast copies (sibling) and
    # the converted softmax host
    p = torch.softmax(y.to(torch.float32), -1).to(DTYPE)
    logits = (p * w_out).sum(-1)  # logit stand-in (no GEMM: closed)
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
        self.wq, self.wk, self.wv, self.w_out = (
            torch.randn(D, device="cuda", dtype=DTYPE) for _ in range(4)
        )
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
        variant = ht.build(tape, decode_step, args)
        return tape, variant

    def _step(self, variant, B, L, eager_caches, replay_caches):
        # the same ids and the same cache contents on both sides; the replay
        # writes position L - 1 of the caller's cache like the ordinary call
        ids = self._ids(B)
        want = decode_step(*self._args(ids, eager_caches, L))
        got = variant.try_replay(self._args(ids, replay_caches, L))
        if got is None:
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
        # the ids' H2D copy, embedding (index_select), layer norm, the
        # projection muls, two rotaries (mul, mul, cat as two copies, mul,
        # add), two cache copies, flash, add, silu, two casts, softmax, mul,
        # two sums: every launch a converted or sibling host
        self.assertEqual(tape.num_memcpys, 1)  # the ids
        self.assertGreaterEqual(tape.num_launches, 20)
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
        # ht.wait_for_h2d(ids) before each rewrite; nothing detects a missing
        # wait (a contract, like the synchronous-copy rule). A fresh pinned
        # tensor per step needs no wait: the replay holds it until its copy ran.
        _, variant = self._trace(4, 16)
        big = torch.randn(8192, 8192, device="cuda", dtype=DTYPE)

        def run(pattern):
            e_caches = self._caches(4)
            r_caches = tuple(c.clone() for c in e_caches)
            ids = self._ids(4)
            pairs = []
            for L in range(17, 27):
                if pattern == "in_place_with_wait":
                    ht.wait_for_h2d(ids)
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
    def test_the_cudnn_sdpa_host_declines_by_name(self):
        # the default backend order may pick cuDNN, whose host is not converted:
        # the trace declines naming it, never a wrong result; a decode pins flash
        caches = self._caches(4)
        args = self._args(self._ids(4), caches, 16)
        with torch.nn.attention.sdpa_kernel(
            torch.nn.attention.SDPBackend.CUDNN_ATTENTION
        ):
            with self.assertRaisesRegex(ht.Declined, "cudnn"):
                ht.trace(decode_step, args)

    def test_odd_head_dims_decline_by_name(self):
        # flash pads head dims to a multiple of 8 inside its host
        # (constant_pad_nd, an unconverted op inside a converted host)
        q, k, v = (
            torch.randn(2, H, 16, 100, device="cuda", dtype=DTYPE) for _ in range(3)
        )
        with self.assertRaisesRegex(ht.Declined, "constant_pad_nd"):
            ht.trace(F.scaled_dot_product_attention, (q, k, v))

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
        variant = ht.build(tape, sdpa, args)
        for lq, lk in ((32, 48), (48, 48), (33, 47)):
            a = qkv(lq, lk)
            self.assertTrue(torch.equal(variant.replay(a)[0], sdpa(*a)), f"{lq}/{lk}")
        with self.assertRaisesRegex(ht.Miss, "% 128"):
            variant.replay(qkv(32, 128))
        with self.assertRaisesRegex(ht.Miss, "!= 1"):
            variant.replay(qkv(1, 32))

    def test_what_a_real_decode_still_needs(self):
        # the ops a real decode reaches that this stack does not trace yet: each
        # declines by name (never a wrong result); the list is the work queue
        x = torch.randn(4, D, device="cuda", dtype=DTYPE)
        w2 = torch.randn(D, D, device="cuda", dtype=DTYPE)
        ids = torch.randint(0, V, (4,), device="cuda")
        cases = {
            "matmul (closed GEMM)": (lambda a, b: a @ b, (x, w2)),
        }
        for name, (fn, args) in cases.items():
            with self.assertRaises(ht.Declined, msg=name):
                ht.trace(fn, args)
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
        variant = ht.build(tape, fwd_bwd, args)
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
        variant = ht.build(tape, fwd_bwd, args)
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


if __name__ == "__main__":
    run_tests()
