# The closed trtllm-gen decode/context ops (vllm/armV trtllm_shim's vllm_ht ops over stock FlashInfer) harvested with
# their KV length patched (HarvestProvider(patched=trtllm_shim.patched())): a key per split class, the length a scalar
# slot, bitwise vs eager at vLLM's Qwen3-8B shapes (32/8 heads, head dim 128, page 16, HND K/V):
#   CUDA_VISIBLE_DEVICES=1 INTEG_ARMV=land/core/attn/armV bash python_vllm_attn.sh tests/test_closed_trtllm_patched.py
import math
import os
import sys

import torch

sys.path.insert(0, os.environ.get("INTEG_ARMV", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "armV")))
import trtllm_shim  # noqa: E402
from torch.cuda._host_trace_harvest import HarvestProvider  # noqa: E402
from torch.cuda._host_trace_replay import HostTraceReplay  # noqa: E402
from torch.cuda._host_trace_tape import EagerCall  # noqa: E402
from torch.testing._internal.common_utils import run_tests, TestCase  # noqa: E402

HQ, HKV, D, PAGE, PAGES, WIDTH = 32, 8, 128, 16, 4096, 256
bf16 = torch.bfloat16
ops = torch.ops.vllm_ht


def decode(q, k, v, ws, counters, bt, seq_lens, out, max_seq_len):
    ops.trtllm_decode(q, k, v, ws, counters, bt, seq_lens, out, 1 / math.sqrt(D), 1.0, max_seq_len, -1)
    return out


def context(q, k, v, ws, counters, bt, seq_lens, cu_q, cu_kv, out, max_q, max_kv, bs):
    ops.trtllm_context(q, k, v, ws, counters, bt, seq_lens, cu_q, cu_kv, out, 1 / math.sqrt(D), 1.0, max_q, max_kv, bs, -1)
    return out


def tables(lens, seed):
    g = torch.Generator().manual_seed(seed)
    bt = torch.zeros(len(lens), WIDTH, dtype=torch.int32)
    perm, at = torch.randperm(PAGES, generator=g), 0
    for i, n in enumerate(lens):
        p = math.ceil(n / PAGE)
        bt[i, :p] = perm[at : at + p]
        at += p
    return bt.cuda(), torch.tensor(lens, dtype=torch.int32, device="cuda")


class TestClosedPatched(TestCase):
    def setUp(self):
        g = torch.Generator(device="cuda").manual_seed(0)
        self.k = torch.randn(PAGES, HKV, PAGE, D, device="cuda", generator=g).to(bf16)
        self.v = torch.randn(PAGES, HKV, PAGE, D, device="cuda", generator=g).to(bf16)
        self.ws = torch.zeros(128 << 20, dtype=torch.uint8, device="cuda")
        self.counters = torch.zeros(1 << 20, dtype=torch.uint8, device="cuda")

    def provider(self):
        return HarvestProvider(("extern",), extern_ops=(ops.trtllm_decode.default, ops.trtllm_context.default), patched=trtllm_shim.patched())

    def eager_steps(self, r):
        return [rec.name for v in r.variants for _, rec in v.tape.launches if type(rec) is EagerCall]

    def test_decode(self):
        # bs 1: split min(ceil(max_seq_len / 512), 19); bs 16: always 1. Every step's max_seq_len (stock vLLM: the
        # batch's longest sequence) moves; keys learned only per split class
        for bs, steps in ((1, (300, 301, 302, 450, 513, 700, 1024, 1025, 1500, 301, 2049)), (16, (300, 301, 900, 2000, 3000))):
            p = self.provider()
            r = HostTraceReplay(decode, opaque=(p,))
            for i, m in enumerate(steps):
                bt, seq_lens = tables([m] + [min(m, 200 + 7 * j) for j in range(bs - 1)], i)
                q = torch.randn(bs, HQ, D, device="cuda").to(bf16)
                out, ref = torch.empty_like(q), torch.empty_like(q)
                decode(q, self.k, self.v, self.ws, self.counters, bt, seq_lens, ref, m)
                r(q, self.k, self.v, self.ws, self.counters, bt, seq_lens, out, m)
                self.assertEqual(out, ref, atol=0, rtol=0)
            classes = {min(-(-m // 512), max(1, 152 // (8 * bs))) for m in steps}
            slots = sorted({(t.param, t.offset, t.width, t.delta) for b in p.bindings.values() for n in b.nodes for t in getattr(n, "slots", ()) if t.kind == "scalar"})
            print(f"decode bs {bs}: harvests {p.harvests} bindings {len(p.bindings)} classes {sorted(classes)} refused {list(p.refused.values())[:2]} "
                  f"traces {r.traces} learned {r.learned} scalar slots {slots}")
            self.assertEqual(p.refused, {})
            self.assertEqual(len(p.bindings), len(classes))
            self.assertEqual(self.eager_steps(r), [])

    def test_context(self):
        # a context call's max_kv: its params only, one key for every value
        p = self.provider()
        r = HostTraceReplay(context, opaque=(p,))
        for i, kv_lens in enumerate(([64, 300], [64, 301], [100, 420], [64, 2000], [700, 64])):
            bt, seq_lens = tables(kv_lens, i)
            q = torch.randn(128, HQ, D, device="cuda").to(bf16)
            cu_q = torch.tensor([0, 64, 128], dtype=torch.int32, device="cuda")
            cu_kv = torch.tensor([0, kv_lens[0], sum(kv_lens)], dtype=torch.int32, device="cuda")
            out, ref = torch.empty_like(q), torch.empty_like(q)
            args = (q, self.k, self.v, self.ws, self.counters, bt, seq_lens, cu_q, cu_kv)
            context(*args, ref, 64, max(kv_lens), 2)
            r(*args, out, 64, max(kv_lens), 2)
            self.assertEqual(out, ref, atol=0, rtol=0)
        print(f"context: harvests {p.harvests} bindings {len(p.bindings)} refused {list(p.refused.values())[:2]} traces {r.traces}")
        self.assertEqual((len(p.bindings), p.refused), (1, {}))
        self.assertEqual(self.eager_steps(r), [])


if __name__ == "__main__":
    run_tests()
