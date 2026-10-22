# OPEN_PLAN.md GEMM scoping: GPU time of Qwen3-8B's bf16 projections per M (a CUDA graph cycling over weight copies
# that together exceed L2, so host launch cost is out and weights come from HBM, as in a step), cuBLAS
# against open (traceable-launch) GEMMs. Backends (--backends): cublas (F.linear), triton_bi (vLLM's batch-invariant
# matmul_persistent, one fixed config), inductor (torch.compile max-autotune, TRITON templates only, per (shape, M)),
# inductor_tma (inductor plus triton.enable_persistent_tma_matmul), fi_cute (FlashInfer mm_bf16 backend="cute-dsl"),
# tgv (SGLang's CuTe TGV, sglang env only: python_sgltgv.sh). Tables: python3 bench/gemm_open_table.py.
#   CUDA_VISIBLE_DEVICES=2 bash python_vllm_cpp.sh bench/gemm_open.py --out bench/out/gemm_open_v.json --backends cublas triton_bi fi_cute
import argparse, json, os, sys

import torch
import torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
p.add_argument("--backends", nargs="+", default=["cublas", "triton_bi", "fi_cute"])
p.add_argument("--ms", type=int, nargs="+", default=[1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])
p.add_argument("--shapes", nargs="+", default=["qkv", "o", "gate_up", "down", "lm_head"])
a = p.parse_args()

SHAPES = {"qkv": (6144, 4096), "o": (4096, 4096), "gate_up": (24576, 4096), "down": (4096, 12288), "lm_head": (151936, 4096)}
LM_MS = {1, 8, 16, 32, 64, 128, 256}  # lm_head runs on one row per request
dev = torch.device("cuda")
torch.manual_seed(0)


def make(name):
    if name == "cublas":
        return lambda x, w: F.linear(x, w)
    if name == "triton_bi":
        from vllm.model_executor.determinism.batch_invariant import matmul_persistent
        return lambda x, w: matmul_persistent(x, w.t())
    if name == "fi_cute":
        from flashinfer.gemm import mm_bf16
        return lambda x, w: mm_bf16(x, w.t(), backend="cute-dsl")
    if name == "tgv":
        from sglang.kernels.ops.gemm.cutedsl_bf16_gemm import _tgv_bf16_gemm_run
        return lambda x, w: _tgv_bf16_gemm_run(x, w, None)
    if name in ("inductor", "inductor_tma"):
        import torch._inductor.config as ic
        ic.max_autotune_gemm_backends = "TRITON"
        ic.max_autotune = True
        ic.triton.enable_persistent_tma_matmul = name == "inductor_tma"
        cache = {}

        def run(x, w):
            k = (x.shape, w.shape)
            if k not in cache:
                torch._dynamo.reset()
                cache[k] = torch.compile(lambda x, w: F.linear(x, w), dynamic=False, mode="max-autotune-no-cudagraphs")
            return cache[k](x, w)
        return run
    raise ValueError(name)


fns = {b: make(b) for b in a.backends}


def gpu_us(fn, x, ws, reps=20):
    fn(x, ws[0])
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for w in ws:
            fn(x, w)
    g.replay()
    torch.cuda.synchronize()
    t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(5):
        t0.record()
        for _ in range(reps):
            g.replay()
        t1.record()
        t1.synchronize()
        times.append(t0.elapsed_time(t1) * 1e3 / (reps * len(ws)))
    return sorted(times)[2]

res = []
for s in a.shapes:
    N, K = SHAPES[s]
    w = torch.randn(N, K, device=dev, dtype=torch.bfloat16) / K ** 0.5
    ws = [w] + [w.clone() for _ in range(max(0, -(-(320 << 20) // (N * K * 2)) - 1))]
    for M in a.ms:
        if s == "lm_head" and M not in LM_MS:
            continue
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        ref = F.linear(x, w).float()
        row = {"shape": s, "M": M, "N": N, "K": K}
        for b, fn in fns.items():
            try:
                y = fn(x, w)
                row[f"{b}_err"] = (y.float() - ref).abs().max().item()
                row[b] = gpu_us(fn, x, ws)
            except Exception as e:  # noqa: BLE001  a backend that does not take the shape
                row[b] = None
                row[f"{b}_why"] = f"{type(e).__name__}: {str(e)[:160]}"
        res.append(row)
        print(json.dumps(row), flush=True)
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
with open(a.out, "w") as f:
    json.dump(res, f, indent=1)
