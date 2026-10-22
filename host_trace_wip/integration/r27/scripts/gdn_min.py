# minimal: one FlashInfer GDN prefill call per shape, NaN tokens per sequence; env CUTE_DSL_* printed
import os, torch
from flashinfer.gdn_prefill import chunk_gated_delta_rule
dev = "cuda"; HK, HV, D = 16, 48, 128
print({k: v for k, v in os.environ.items() if "CUTE" in k or "CUTLASS" in k or "TVM" in k or "FLASHINFER_WORK" in k})
for seqs, T in ((1, 64), (1, 512), (4, 128), (1, 100)):
    g = torch.Generator(device=dev).manual_seed(0); N = seqs * T
    q = torch.nn.functional.normalize(torch.randn(N, HK, D, device=dev, generator=g), dim=-1).to(torch.bfloat16)
    k = torch.nn.functional.normalize(torch.randn(N, HK, D, device=dev, generator=g), dim=-1).to(torch.bfloat16)
    v = torch.randn(N, HV, D, device=dev, generator=g).to(torch.bfloat16)
    gate = torch.exp(-torch.rand(N, HV, device=dev, generator=g)); beta = torch.rand(N, HV, device=dev, generator=g)
    cu = torch.arange(0, N + 1, T, dtype=torch.int64, device=dev)
    for st_dtype in (torch.float32,):
        out, fin = chunk_gated_delta_rule(q=q, k=k, v=v, g=gate, beta=beta, initial_state=torch.zeros(seqs, HV, D, D, device=dev, dtype=st_dtype),
                                          output_final_state=True, cu_seqlens=cu)
        torch.cuda.synchronize()
        bad = out.isnan().reshape(N, -1).any(-1).nonzero().flatten().tolist()
        print(f"seqs {seqs} T {T}: nan tokens {len(bad)} first/last {bad[:2]} {bad[-2:]} final-state nan {int(fin.isnan().sum())}", flush=True)
