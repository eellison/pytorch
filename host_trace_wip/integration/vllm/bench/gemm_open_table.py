# Per-shape and per-step GPU-time tables for OPEN_PLAN.md from bench/out/gemm_open_{v,tgv,ind,tma}.json.
import json, os
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
cell = {}
for f in ("gemm_open_v", "gemm_open_tgv", "gemm_open_ind", "gemm_open_tma"):
    for r in json.load(open(os.path.join(D, f + ".json"))):
        c = cell.setdefault((r["shape"], r["M"]), {})
        for b in ("cublas", "triton_bi", "fi_cute", "tgv", "inductor", "inductor_tma"):
            if r.get(b) is not None and (b != "cublas" or f == "gemm_open_v"):
                c[b] = r[b]
OPEN = ("tgv", "fi_cute", "inductor", "inductor_tma")
MS = (1, 32, 128, 512, 2048, 8192)
fmt = lambda v: "-" if v is None else f"{v:.1f}"
print("| shape | M | cuBLAS | TGV | FI cute-dsl | Inductor Triton | Inductor Triton TMA | vLLM BI Triton | best open / cuBLAS |")
print("|---|---|---|---|---|---|---|---|---|")
for s in ("qkv", "o", "gate_up", "down", "lm_head"):
    for M in MS:
        c = cell.get((s, M))
        if not c:
            continue
        best = min((c[b], b) for b in OPEN if b in c)
        print(f"| {s} | {M} | {fmt(c.get('cublas'))} | {fmt(c.get('tgv'))} | {fmt(c.get('fi_cute'))} | {fmt(c.get('inductor'))} | "
              f"{fmt(c.get('inductor_tma'))} | {fmt(c.get('triton_bi'))} | {best[0] / c['cublas']:.2f} ({best[1]}) |")
print()
print("| step | M (lm_head M) | cuBLAS us | best open us | ratio | TGV<=128 + Inductor TMA >128 us | ratio |")
print("|---|---|---|---|---|---|---|")
for name, M, lm in (("decode", 1, 1), ("decode", 32, 32), ("decode", 128, 128), ("prefill", 512, 1), ("prefill", 2048, 1), ("prefill", 8192, 1)):
    tot = {"cublas": 0.0, "best": 0.0, "pol": 0.0}
    for s, n, m in (("qkv", 36, M), ("o", 36, M), ("gate_up", 36, M), ("down", 36, M), ("lm_head", 1, lm)):
        c = cell[(s, m)]
        tot["cublas"] += n * c["cublas"]
        tot["best"] += n * min(c[b] for b in OPEN if b in c)
        tot["pol"] += n * (c["tgv"] if m <= 128 else c["inductor_tma"])
    print(f"| {name} | {M} ({lm}) | {tot['cublas']:.0f} | {tot['best']:.0f} | {tot['best'] / tot['cublas']:.2f} | {tot['pol']:.0f} | {tot['pol'] / tot['cublas']:.2f} |")
