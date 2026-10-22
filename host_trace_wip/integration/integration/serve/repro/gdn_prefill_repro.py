# Is FlashInfer's GDN chunked prefill (the kernel vLLM 0.29 picks for Qwen3.5-9B on GB300: "Using FlashInfer GDN prefill kernel")
# reproducible run to run? No model, no host trace: the exact call vLLM makes (fi_chunk_gated_delta_rule: l2norm'd q/k, exp(g),
# fp32 beta and state, int64 cu_seqlens), Qwen3.5-9B head shapes (16 key heads x 128, 32 value heads x 128), on identical inputs,
# N times; prints per case how many distinct outputs / final states, and which rows (by sequence, chunk of 64) differ.
#   CUDA_VISIBLE_DEVICES=N CAND_LINE=c4 bash python_vllm_cand2.sh serve/repro/gdn_prefill_repro.py
import torch
from flashinfer.gdn_prefill import chunk_gated_delta_rule

torch.manual_seed(0)
HK, HV, D = 16, 32, 128
RUNS = 8


def case(name, lens, init_nonzero=False):
    T = sum(lens)
    q = torch.nn.functional.normalize(torch.randn(T, HK, D, device="cuda", dtype=torch.float32), dim=-1).bfloat16()
    k = torch.nn.functional.normalize(torch.randn(T, HK, D, device="cuda", dtype=torch.float32), dim=-1).bfloat16()
    v = torch.randn(T, HV, D, device="cuda", dtype=torch.bfloat16)
    g = torch.exp(-torch.rand(T, HV, device="cuda", dtype=torch.float32) * 0.1)
    beta = torch.rand(T, HV, device="cuda", dtype=torch.float32)
    n = len(lens)
    state = (torch.randn(n, HV, D, D, device="cuda", dtype=torch.float32) * 0.01) if init_nonzero else torch.zeros(n, HV, D, D, device="cuda", dtype=torch.float32)
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0).tolist()), device="cuda", dtype=torch.int64)
    outs = []
    for _ in range(RUNS):
        o, s = chunk_gated_delta_rule(q=q, k=k, v=v, g=g, beta=beta, initial_state=state.clone(), output_final_state=True, cu_seqlens=cu)
        torch.cuda.synchronize()
        outs.append((o.clone(), s.clone()))
    eq = lambda a, b: bool(((a == b) | (a.isnan() & b.isnan())).all())
    d_o, d_s = [], []
    for o, s in outs:
        if not any(eq(o, x) for x in d_o):
            d_o.append(o)
        if not any(eq(s, x) for x in d_s):
            d_s.append(s)
    o0, o1 = outs[0][0], next((o for o, _ in outs if not eq(o, outs[0][0])), outs[0][0])
    rows = ((o0 != o1) & ~(o0.isnan() & o1.isnan())).reshape(T, -1).any(dim=1).nonzero().flatten().tolist()
    where = {}
    starts = cu.tolist()
    for r in rows:
        si = max(i for i in range(n) if starts[i] <= r)
        where.setdefault(si, set()).add((r - starts[si]) // 64)
    nan = [int(o.isnan().any(dim=-1).any(dim=-1).sum()) for o, _ in outs[:3]]
    print(f"{name}: lens {lens} init {'rand' if init_nonzero else 'zero'}: distinct outputs {len(d_o)}/{RUNS}, distinct final states {len(d_s)}/{RUNS}, "
          f"differing rows {len(rows)} (seq -> chunks of 64: {{{', '.join(f'{k}: {sorted(v)[:8]}' for k, v in where.items())}}}), NaN rows {nan}, "
          f"max |o0-o1| {float((o0.float()-o1.float()).nan_to_num().abs().max()):.3g}", flush=True)


def unwritten(name, lens):
    """The same call with the output buffer pre-filled with two different values: rows that keep the fill were never written
    by the kernel (eager then returns whatever the allocator's memory held)."""
    T = sum(lens)
    q = torch.nn.functional.normalize(torch.randn(T, HK, D, device="cuda"), dim=-1).bfloat16()
    k = torch.nn.functional.normalize(torch.randn(T, HK, D, device="cuda"), dim=-1).bfloat16()
    v = torch.randn(T, HV, D, device="cuda", dtype=torch.bfloat16)
    g = torch.exp(-torch.rand(T, HV, device="cuda") * 0.1)
    beta = torch.rand(T, HV, device="cuda")
    n = len(lens)
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0).tolist()), device="cuda", dtype=torch.int64)
    res = []
    for fill in (7.0, -3.0):
        out = torch.full((T, HV, D), fill, device="cuda", dtype=torch.bfloat16)
        chunk_gated_delta_rule(q=q, k=k, v=v, g=g, beta=beta, initial_state=torch.zeros(n, HV, D, D, device="cuda"), output_final_state=True,
                               cu_seqlens=cu, output=out)
        torch.cuda.synchronize()
        res.append(out.clone())
    keep = (res[0] == 7.0).all(dim=-1).all(dim=-1) & (res[1] == -3.0).all(dim=-1).all(dim=-1)
    rows = keep.nonzero().flatten().tolist()
    starts = cu.tolist()
    per = {}
    for r in rows:
        si = max(i for i in range(n) if starts[i] <= r)
        per.setdefault(si, []).append(r - starts[si])
    print(f"UNWRITTEN {name}: lens {lens}: {len(rows)} of {T} rows never written; per seq (first..last offset, count): "
          f"{ {k: (min(v), max(v), len(v)) for k, v in list(per.items())[:8]} }; NaN rows with fill {int(res[0].isnan().any(dim=-1).any(dim=-1).sum())}", flush=True)


for name, lens in (("short", [256]), ("one 5103", [5103]), ("batch", [5103, 5229, 64, 64, 2048]), ("many small", [64] * 32), ("odd", [100, 300, 1000])):
    unwritten(name, lens)

for name, lens, init in (("short", [256], False), ("one 2048", [2048], False), ("one 5103", [5103], False), ("one 8000", [8000], False),
                         ("batch like q35s (new prefills)", [5103, 5229, 64, 64, 2048], False), ("continuing chunks (nonzero state)", [1710, 6126], True),
                         ("many small", [64] * 32, False)):
    case(name, lens, init)
