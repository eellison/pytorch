# The FlashInfer trtllm-gen fork (flashinfer/trtllm_trace.py) under host-trace replay at vLLM's Qwen3-8B shapes
# (32/8 heads, head dim 128, page 16, HND K/V), through the public API vLLM calls:
#   ATTN_FORK=1 CUDA_VISIBLE_DEVICES=1 bash land/core/attn/python_vllm_attn.sh land/core/attn/tests/test_trtllm_fork_trace.py
import math
import os
import time

import torch

import flashinfer
import flashinfer.trtllm_trace as T
from torch.cuda._host_trace_replay import HostTraceReplay
from torch.cuda._host_trace_tape import EagerCall
from torch.testing._internal.common_utils import run_tests, TestCase

HQ, HKV, D, PAGE, PAGES, WIDTH = 32, 8, 128, 16, 4096, 256
LAYERS = int(os.environ.get("FORK_TEST_LAYERS", "4"))
bf16 = torch.bfloat16


def caches(seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    k = torch.randn(PAGES, HKV, PAGE, D, device="cuda", generator=g).to(bf16)
    v = torch.randn(PAGES, HKV, PAGE, D, device="cuda", generator=g).to(bf16)
    return k, v


def tables(lens, seed=0):
    g = torch.Generator().manual_seed(seed)
    bt = torch.zeros(len(lens), WIDTH, dtype=torch.int32)
    perm = torch.randperm(PAGES, generator=g)
    at = 0
    for i, n in enumerate(lens):
        p = math.ceil(n / PAGE)
        bt[i, :p] = perm[at : at + p]
        at += p
    return bt.cuda(), torch.tensor(lens, dtype=torch.int32, device="cuda")


def context_step(q, k, v, ws, bt, seq_lens, cu_q, cu_kv, out, max_q, max_kv, bs):
    # vLLM's prefill call (v1/attention/backends/flashinfer.py), once per layer
    for _ in range(LAYERS):
        flashinfer.prefill.trtllm_batch_context_with_kv_cache(
            query=q, kv_cache=(k, v), workspace_buffer=ws, block_tables=bt, seq_lens=seq_lens, max_q_len=max_q, max_kv_len=max_kv,
            bmm1_scale=1 / math.sqrt(D), bmm2_scale=1.0, batch_size=bs, cum_seq_lens_q=cu_q, cum_seq_lens_kv=cu_kv, window_left=-1,
            out=out, kv_layout="HND")
    return out


def decode_step(q, k, v, ws, bt, seq_lens, out, max_seq_len):
    for _ in range(LAYERS):
        flashinfer.decode.trtllm_batch_decode_with_kv_cache(
            query=q, kv_cache=(k, v), workspace_buffer=ws, block_tables=bt, seq_lens=seq_lens, max_seq_len=max_seq_len,
            bmm1_scale=1 / math.sqrt(D), bmm2_scale=1.0, window_left=-1, out=out, kv_layout="HND", backend="trtllm-gen", q_len_per_req=1)
    return out


class TestTrtllmFork(TestCase):
    def setUp(self):
        self.k, self.v = caches()
        self.ws = torch.zeros(256 << 20, dtype=torch.uint8, device="cuda")

    def eager_steps(self, r):
        return [rec.name for v in r.variants for _, rec in v.tape.launches if isinstance(rec, EagerCall)]

    def context_args(self, q_lens, kv_lens, seed):
        bt, seq_lens = tables(kv_lens, seed)
        g = torch.Generator(device="cuda").manual_seed(seed)
        q = torch.randn(sum(q_lens), HQ, D, device="cuda", generator=g).to(bf16)
        cu_q = torch.tensor([0, *torch.tensor(q_lens).cumsum(0).tolist()], dtype=torch.int32, device="cuda")
        cu_kv = torch.tensor([0, *torch.tensor(kv_lens).cumsum(0).tolist()], dtype=torch.int32, device="cuda")
        out = torch.empty_like(q)
        return [q, self.k, self.v, self.ws, bt, seq_lens, cu_q, cu_kv, out, max(q_lens), max(kv_lens), len(q_lens)]

    def test_context_max_kv_moves_no_dispatch(self):
        # a context call reads max_kv only into its params: every max_kv (vLLM's stock value in a mixed step, the
        # batch's longest sequence) replays one variant, no redispatch, bitwise
        r = HostTraceReplay(context_step)
        cases = [([64, 64], [64, 300]), ([64, 64], [64, 301]), ([64, 64], [100, 420]), ([64, 64], [64, 2000]), ([64, 64], [700, 64]),
                 ([64, 64], [64, 301])]
        times = []
        for i, (q_lens, kv_lens) in enumerate(cases):
            args = self.context_args(q_lens, kv_lens, i)
            ref = list(args)
            ref[8] = torch.empty_like(args[8])
            context_step(*ref)
            start = time.perf_counter()
            r(*args)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
            self.assertEqual(args[8], ref[8], atol=0, rtol=0)
        print(f"context: traces {r.traces} replays {r.replays} redispatches {r.redispatches} ({r.redispatch_s * 1e3:.1f} ms) "
              f"retrace_causes {r.retrace_causes} call ms {[round(t * 1e3, 1) for t in times]}")
        self.assertEqual(self.eager_steps(r), [])
        if os.environ.get("FORK_TEST_EXPECT_PINNED"):
            # the original fork: max_kv pinned, a redispatch (of each layer's op) per new max_kv
            self.assertGreater(r.redispatches, 0)
            return
        # (the entry's first call runs eagerly: r.eager 1)
        self.assertEqual((r.traces, r.redispatches, r.eager), (1, 0, 1))

    def test_decode_split_regions(self):
        # decode at bs 4: max_seq_len moves the split (min(ceil(max_kv / 512), 19 // 4 ...)) in 512-token regions; a
        # redispatch per region first seen, none for a region seen before, bitwise
        r = HostTraceReplay(decode_step)
        lens = [300, 260, 270, 280]
        plans = []
        for i, m in enumerate((300, 400, 600, 1100, 1500, 400, 2100, 600)):
            bt, seq_lens = tables([min(x, m) for x in lens[:-1]] + [m], i)
            q = torch.randn(4, HQ, D, device="cuda").to(bf16)
            out, ref = torch.empty_like(q), torch.empty_like(q)
            decode_step(q, self.k, self.v, self.ws, bt, seq_lens, ref, m)
            r(q, self.k, self.v, self.ws, bt, seq_lens, out, m)
            self.assertEqual(out, ref, atol=0, rtol=0)
            plans.append(min(-(-m // 512), max(1, 152 // (8 * 4))))
        print(f"decode: traces {r.traces} redispatches {r.redispatches} folds {r.folds} retrace_causes {r.retrace_causes} splits {plans}")
        self.assertEqual(self.eager_steps(r), [])
        self.assertLessEqual(r.traces + r.redispatches + r.folds, len(set(plans)) + 1)


if __name__ == "__main__":
    run_tests()
