# Owner(s): ["module: inductor"]
"""The decode-step composition (commit 6) lowered into the shared native replay."""

import gc
import os
import sys

import torch
import torch.nn.functional as F
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_FLASH_ATTENTION
from torch.testing._internal.common_utils import run_tests, TestCase


sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
)
from host_trace_h2d_probe import probe


C = torch._C
H, DH = 8, 64
D = H * DH
V = 1024
LMAX = 48
DTYPE = torch.bfloat16


def decode_step(ids, table, ln_w, ln_b, wq, wk, wv, w_out, k_view, v_view):
    x = probe().gather(table, ids)
    B = x.shape[0]
    L = k_view.shape[2]
    h = F.layer_norm(x, (D,), ln_w, ln_b)
    q = (h * wq).view(B, H, 1, DH)
    k_view[:, :, L - 1 : L].copy_(h.mul(wk).view(B, H, 1, DH))
    v_view[:, :, L - 1 : L].copy_(h.mul(wv).view(B, H, 1, DH))
    attn = F.scaled_dot_product_attention(q, k_view, v_view)
    y = F.silu(attn.reshape(B, D) + x)
    logits = (y * w_out).sum(-1)
    last = k_view[:, :, L - 1].sum(-1)
    return logits, last


class TestHostTraceDecode(TestCase):
    def setUp(self):
        super().setUp()
        if not torch.cuda.is_available() or not PLATFORM_SUPPORTS_FLASH_ATTENTION:
            self.skipTest("CUDA with flash attention required")
        from torch._inductor.runtime._cudagraph import direct_hosttrace

        self.module = direct_hosttrace
        torch.manual_seed(0)
        self.table = torch.randn(V, D, device="cuda", dtype=DTYPE)
        self.ln_w = 1 + 0.1 * torch.randn(D, device="cuda", dtype=DTYPE)
        self.ln_b = 0.1 * torch.randn(D, device="cuda", dtype=DTYPE)
        self.wq, self.wk, self.wv, self.w_out = (
            torch.randn(D, device="cuda", dtype=DTYPE) for _ in range(4)
        )
        # the default backend order may prefer cuDNN, whose host is not converted
        self._flash = torch.nn.attention.sdpa_kernel(
            torch.nn.attention.SDPBackend.FLASH_ATTENTION
        )
        self._flash.__enter__()
        self.addCleanup(self._flash.__exit__, None, None, None)

    def _caches(self, B):
        return (
            torch.randn(B, H, LMAX, DH, device="cuda", dtype=DTYPE),
            torch.randn(B, H, LMAX, DH, device="cuda", dtype=DTYPE),
        )

    @staticmethod
    def _ids(B):
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
            k[:, :, :L],
            v[:, :, :L],
        )

    def _replay(self, B, L):
        gc.collect()
        replay = self.module.HostTraceReplay(
            decode_step, self._args(self._ids(B), self._caches(B), L)
        )
        self.addCleanup(replay.close)
        return replay

    def _step(self, replay, B, L, eager_caches, replay_caches):
        ids = self._ids(B)
        want = decode_step(*self._args(ids, eager_caches, L))
        before = replay.misses
        got = replay(*self._args(ids, replay_caches, L))
        served = replay.misses == before
        for g, w in zip(got, want):
            self.assertEqual(g, w, atol=0, rtol=0)
        for e, r in zip(eager_caches, replay_caches):
            self.assertEqual(e[:, :, :L], r[:, :, :L], atol=0, rtol=0)
        return served

    def test_steps_of_a_growing_cache_serve_bitwise(self):
        replay = self._replay(4, 16)
        e_caches = self._caches(4)
        r_caches = tuple(c.clone() for c in e_caches)
        served = [self._step(replay, 4, L, e_caches, r_caches) for L in range(16, 33)]
        self.assertTrue(all(served))
        self.assertEqual(replay.misses, 0)

    def test_other_batch_sizes_from_one_trace(self):
        replay = self._replay(4, 16)
        for B in (3, 8):
            e_caches = self._caches(B)
            r_caches = tuple(c.clone() for c in e_caches)
            self.assertTrue(self._step(replay, B, 20, e_caches, r_caches))
        # batch 1 changes the sibling's coalescing: a named miss, its own variant
        e_caches = self._caches(1)
        self.assertFalse(
            self._step(replay, 1, 20, e_caches, tuple(c.clone() for c in e_caches))
        )

    def test_two_variants_interleaved(self):
        r4, r2 = self._replay(4, 16), self._replay(2, 16)
        e4, e2 = self._caches(4), self._caches(2)
        c4, c2 = tuple(c.clone() for c in e4), tuple(c.clone() for c in e2)
        for L in range(17, 25):
            self.assertTrue(self._step(r4, 4, L, e4, c4))
            self.assertTrue(self._step(r2, 2, L, e2, c2))

    def test_agrees_with_eager_at_other_lengths(self):
        # the oracle clones the caches for eager: the same cache contents on
        # both sides, the cache writes compared after each step
        from torch.testing._internal.host_trace_oracle import Oracle

        oracle = Oracle(decode_step, self._args(self._ids(4), self._caches(4), 16))
        self.addCleanup(oracle.close)
        self.assertIsNone(oracle.refused)
        caches = self._caches(4)
        for L in (18, 24, 30):
            oracle.check(self._args(self._ids(4), caches, L))


if __name__ == "__main__":
    run_tests()
