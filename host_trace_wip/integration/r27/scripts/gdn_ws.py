# gdn_ws.py: per shape, the GDN prefill's NaN flag next to the addresses of its cached workspaces (flashinfer gdn_prefill's compiled-config
# cache) and of its output / state buffers, modulo a few alignments: is NaN a function of an address?
import gc, os
import torch
from flashinfer.gdn_prefill import chunk_gated_delta_rule
import flashinfer.gdn_kernels.blackwell.gdn_prefill as gp

dev, HK, HV, D = "cuda", 16, 48, 128


def workspaces():
    out = []
    for ref in gc.get_referents(gp._get_compiled_cache):
        if isinstance(ref, dict):
            for key, cfg in ref.items():
                if isinstance(cfg, dict):
                    out += [(str(key)[:40], kk, cfg[kk]) for kk in cfg if isinstance(kk, str) and kk.startswith("workspace_")]
    return out


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
    o, f = chunk_gated_delta_rule(q=q, k=k, v=v, g=gate, beta=beta, initial_state=state, output_final_state=True, cu_seqlens=cu)
    torch.cuda.synchronize()
    nan = bool(o.isnan().any())
    ws = [(kk, hex(w.data_ptr() % (1 << 16)), w.numel()) for _, kk, w in workspaces()]
    print(f"{spec}: {'NaN' if nan else 'ok '} out%64K {hex(o.data_ptr() % (1 << 16))} state%64K {hex(f.data_ptr() % (1 << 16))} "
          f"q%64K {hex(q.data_ptr() % (1 << 16))} workspaces {ws}", flush=True)
