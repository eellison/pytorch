# gdn_rep2.py: FlashInfer 0.6.18 chunk_gated_delta_rule (Blackwell CuTe chunked GDN prefill) at Qwen3.8-27B shapes, fixed finite inputs,
# zero initial state, sequences GDN_SHAPES (default 1x64 1x512 1x2048 4x128), each run twice: NaN counts in output / final state, whether
# the two runs agree bitwise, and an output digest (to compare processes). FI_FENCE=1 loads flashinfer#5334's fenced kernel copy.
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
    outs = []
    for _ in range(2):
        o, f = chunk_gated_delta_rule(q=q, k=k, v=v, g=gate, beta=beta, initial_state=state.clone(), output_final_state=True, cu_seqlens=cu)
        torch.cuda.synchronize()
        outs.append((o.clone(), f.clone()))
    (o1, f1), (o2, f2) = outs
    same = torch.equal(o1.nan_to_num(7.0), o2.nan_to_num(7.0)) and torch.equal(f1.nan_to_num(7.0), f2.nan_to_num(7.0))
    dig = int((o1.nan_to_num(0).float().view(-1)[:4096] * torch.arange(4096, device=dev)).sum().item() * 1000) if o1.numel() >= 4096 else 0
    print(f"{spec}: nan out {int(o1.isnan().sum())}/{o1.numel()} nan state {int(f1.isnan().sum())} runs-agree {same} digest {dig}", flush=True)
