# Parity of the FlashInfer trtllm-gen fork (flashinfer/trtllm_trace.py, a Python port of the launcher's host code)
# with FlashInfer's real C++ launcher, so the port cannot drift: across a grid of shapes (batch, q and KV lengths, head
# dims 64/128/256, bf16 and fp8 Q/KV, pages 16/32/64, decode and context, sinks and LSE), each public API call runs
# eagerly, and trtllm_trace.parity compares the port's kernel, grid (split count), block, smem, cluster attributes and
# every KernelParams byte (ctas_kv, max_kv, tiles' TMA descriptors...) with the binding's own launch, captured on the
# call's tensors. A config whose cubin is not on this box is skipped (the launcher cannot run it either).
#   ATTN_FORK=1 FLASHINFER_CUBIN_DIR=land/core/attn/cubins CUDA_VISIBLE_DEVICES=1 bash python_vllm_attn.sh tests/test_trtllm_fork_parity.py
import itertools
import math
import unittest.mock

import torch

import flashinfer
import flashinfer.trtllm_trace as T
import flashinfer.trtllm_witness as tw
from torch.testing._internal.common_utils import instantiate_parametrized_tests, parametrize, run_tests, TestCase

PAGES = 2048
fp8 = torch.float8_e4m3fn

# (dtype of Q and KV, head dim, page, (q heads, KV heads)): the cubin sets on this box (vLLM's and SGLang's caches), at
# 4 q heads per KV head (Qwen3-8B) and 6 (DeepSeek-R1-like grouping)
FAMILIES = {"bf16_h128_p16": (torch.bfloat16, 128, 16, (32, 8)), "bf16_h128_p64": (torch.bfloat16, 128, 64, (32, 8)),
            "bf16_h64_p64": (torch.bfloat16, 64, 64, (32, 8)), "fp8_h256_p32": (fp8, 256, 32, (32, 8)),
            "bf16_h128_p16_g6": (torch.bfloat16, 128, 16, (48, 8)), "fp8_h256_p32_g6": (fp8, 256, 32, (48, 8))}
DECODE = [dict(bs=b, kv=k, max_seq=m) for b, k, m in itertools.product((1, 4, 13, 64, 130), (257, 1500), (None, 4096, 40960))]
DECODE += [dict(bs=8, kv=258, q=2), dict(bs=5, kv=600, q=2), dict(bs=8, kv=257, sinks=True), dict(bs=3, kv=700, lse=True)]
CONTEXT = [dict(bs=b, q=q, kv=k) for b, q, k in ((1, 256, 256), (2, 64, 300), (4, 128, 640), (8, 64, 64), (1, 300, 300), (3, 17, 1000))]
CONTEXT += [dict(bs=2, q=256, kv=256, sinks=True), dict(bs=2, q=100, kv=500, lse=True)]
CASES = [(f, "decode", i) for f in FAMILIES for i in range(len(DECODE))] + [(f, "context", i) for f in FAMILIES for i in range(len(CONTEXT))]
_pools: dict = {}


def pools(dtype, d, page, hkv):
    key = (dtype, d, page, hkv)
    if key not in _pools:
        g = torch.Generator(device="cuda").manual_seed(0)
        k = torch.randn(PAGES, hkv, page, d, device="cuda", generator=g).to(dtype)
        v = torch.randn(PAGES, hkv, page, d, device="cuda", generator=g).to(dtype)
        _pools[key] = (k, v)
    return _pools[key]


def call(family, kind, cfg, workspace):
    dtype, d, page, (hq, hkv) = FAMILIES[family]
    bs, kv, q = cfg["bs"], cfg["kv"], cfg.get("q", 1 if kind == "decode" else None)
    g = torch.Generator().manual_seed(bs * 1000 + kv)
    pages = math.ceil(kv / page)
    width = math.ceil(max(kv, cfg.get("max_seq") or 0, 4096) / page)
    bt = torch.zeros(bs, width, dtype=torch.int32)
    # pages may repeat across requests: the calls only read the pools
    bt[:, :pages] = torch.randint(0, PAGES, (bs, pages), generator=g, dtype=torch.int32)
    kw = dict(kv_cache=pools(dtype, d, page, hkv), workspace_buffer=workspace, block_tables=bt.cuda(),
              seq_lens=torch.full((bs,), kv, dtype=torch.int32, device="cuda"), bmm1_scale=1 / math.sqrt(d), bmm2_scale=1.0,
              window_left=-1, kv_layout="HND")
    query = torch.randn(bs * q, hq, d, generator=g).to(dtype).cuda()
    if cfg.get("sinks"):
        kw["sinks"] = torch.randn(hq, generator=g).float().cuda()
    if cfg.get("lse"):
        kw["return_lse"] = True
    with tw.record_binding_calls() as calls:
        if kind == "decode":
            flashinfer.decode.trtllm_batch_decode_with_kv_cache(query=query, max_seq_len=cfg.get("max_seq") or kv, q_len_per_req=q,
                                                                out_dtype=torch.bfloat16, backend="trtllm-gen", **kw)
        else:
            cu_q = torch.arange(bs + 1, dtype=torch.int32, device="cuda") * q
            cu_kv = torch.arange(bs + 1, dtype=torch.int32, device="cuda") * kv
            flashinfer.prefill.trtllm_batch_context_with_kv_cache(query=query, max_q_len=q, max_kv_len=kv, batch_size=bs, cum_seq_lens_q=cu_q,
                                                                  cum_seq_lens_kv=cu_kv, out_dtype=torch.bfloat16, **kw)
    torch.cuda.synchronize()
    (c,) = calls
    return c


@instantiate_parametrized_tests
class TestTrtllmForkParity(TestCase):
    workspace = None

    @parametrize("case", CASES, name_fn=lambda c: f"{c[0]}_{c[1]}_{c[2]}")
    def test_parity(self, case):
        family, kind, i = case
        cfg = (DECODE if kind == "decode" else CONTEXT)[i]
        if TestTrtllmForkParity.workspace is None:
            TestTrtllmForkParity.workspace = torch.zeros(256 << 20, dtype=torch.uint8, device="cuda")
        try:
            c = call(family, kind, cfg, self.workspace)
        except Exception as e:  # the launcher itself: no cubin for the selection on this box
            if "cubin" not in str(e).lower() and "not found" not in str(e).lower():
                raise
            self.skipTest(f"the launcher cannot run {family} {kind} {cfg}: {str(e).splitlines()[0][:160]}")
        try:
            bad = T.parity(c.kind, c.binding, c.args)
        except T.NotFound as e:
            # the launcher only logs a missing cubin (CUDA_ERROR_INVALID_HANDLE) and launches nothing
            if str(e).startswith("no cubin"):
                self.skipTest(f"no cubin on this box for {family} {kind} {cfg}: {str(e)[:120]}")
            self.fail(f"{family} {kind} {cfg}: the launcher runs, the port has no kernel: {e}")
        self.assertEqual(bad, [], f"{family} {kind} {cfg}")

    def test_parity_detects_a_drift(self):
        # the check is not vacuous: a port whose divisor field drifts by one is reported, by name
        if TestTrtllmForkParity.workspace is None:
            TestTrtllmForkParity.workspace = torch.zeros(256 << 20, dtype=torch.uint8, device="cuda")
        c = call("bf16_h128_p16", "decode", DECODE[0], self.workspace)
        self.assertEqual(T.parity(c.kind, c.binding, c.args), [])
        with unittest.mock.patch.object(T, "fast_mod_div", lambda d, f=T.fast_mod_div: f(d + 1)):
            bad = T.parity(c.kind, c.binding, c.args)
        self.assertTrue(bad and all(b.startswith("heads_q_per_kv_divisor") for b in bad), bad)


if __name__ == "__main__":
    run_tests()
