# gdn_outfill.py: does FlashInfer's GDN prefill write every element of its output / final state? Same inputs, output and output_state
# passed in pre-filled with a sentinel (NaN, 12345.0, 0); counts of elements still equal to the sentinel after the call, and whether the
# outputs differ between sentinels. FI_FENCE=1 loads PR #5335.
import os, sys
if os.environ.get("FI_FENCE") == "1":
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fi_fence"))
    import fi_fence_hook  # noqa: F401
import torch
from flashinfer.gdn_prefill import chunk_gated_delta_rule

dev, HK, HV, D = "cuda", 16, 48, 128
for spec in os.environ.get("GDN_SHAPES", "1x64 1x512 1x2048 4x128").split():
    seqs, T = map(int, spec.split("x"))
    g = torch.Generator(device=dev).manual_seed(0)
    N = seqs * T
    q = torch.nn.functional.normalize(torch.randn(N, HK, D, device=dev, generator=g), dim=-1).to(torch.bfloat16)
    k = torch.nn.functional.normalize(torch.randn(N, HK, D, device=dev, generator=g), dim=-1).to(torch.bfloat16)
    v = torch.randn(N, HV, D, device=dev, generator=g).to(torch.bfloat16)
    gate = torch.exp(-torch.rand(N, HV, device=dev, generator=g))
    beta = torch.rand(N, HV, device=dev, generator=g)
    state = torch.zeros(seqs, HV, D, D, device=dev)
    cu = torch.arange(0, N + 1, T, dtype=torch.int64, device=dev)
    res = {}
    for name, val in (("nan", float("nan")), ("12345", 12345.0), ("zero", 0.0)):
        out = torch.full((N, HV, D), val, dtype=torch.bfloat16, device=dev)
        ost = torch.full((seqs, HV, D, D), val, dtype=torch.float32, device=dev)
        o, f = chunk_gated_delta_rule(q=q, k=k, v=v, g=gate, beta=beta, initial_state=state.clone(), output_final_state=True, cu_seqlens=cu,
                                      output=out, output_state=ost)
        torch.cuda.synchronize()
        left_o = int(o.isnan().sum()) if name == "nan" else int((o == val).sum())
        left_f = int(f.isnan().sum()) if name == "nan" else int((f == val).sum())
        res[name] = (o.clone(), f.clone())
        # per-token rows still at the sentinel (which tokens the kernel never wrote)
        rows = (o.isnan() if name == "nan" else (o == val)).view(N, -1).all(dim=1).nonzero().view(-1).tolist()
        print(f"{spec} sentinel {name}: output elements left {left_o}/{o.numel()}, state elements left {left_f}/{f.numel()}, "
              f"untouched tokens {len(rows)} {rows[:3]}..{rows[-3:] if rows else ''}", flush=True)
    a, b = res["12345"][0], res["zero"][0]
    m = ~((a == 12345.0) | (b == 0.0))
    print(f"{spec}: outputs where both written agree: {bool(torch.equal(a[m], b[m]))}", flush=True)
