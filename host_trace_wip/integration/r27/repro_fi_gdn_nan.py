# Standalone: FlashInfer 0.6.18 chunk_gated_delta_rule (the Blackwell CuTe chunked GDN prefill vLLM uses for Qwen3.5 / Qwen3.8
# GDN layers on sm_10x) at Qwen3.8-27B's shapes (16 q/k heads, 48 v heads, head dim 128) and vLLM's warm-up shape (4 sequences x
# 128 tokens), from identical finite inputs and a zero initial state, with the kernel's cached global workspace and the caching
# allocator's free memory filled with different byte patterns before each call. A correct kernel gives the same output for
# every pattern. No vLLM, no host tracing.
#   python repro_fi_gdn_nan.py
import gc
import torch
from flashinfer.gdn_prefill import chunk_gated_delta_rule
import flashinfer.gdn_kernels.blackwell.gdn_prefill as gp

dev = "cuda"
HK, HV, D = 16, 48, 128


def inputs(seqs, T, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    N = seqs * T
    q = torch.nn.functional.normalize(torch.randn(N, HK, D, device=dev, generator=g), dim=-1).to(torch.bfloat16)
    k = torch.nn.functional.normalize(torch.randn(N, HK, D, device=dev, generator=g), dim=-1).to(torch.bfloat16)
    v = torch.randn(N, HV, D, device=dev, generator=g).to(torch.bfloat16)
    gate = torch.exp(-torch.rand(N, HV, device=dev, generator=g))
    beta = torch.rand(N, HV, device=dev, generator=g)
    state = torch.zeros(seqs, HV, D, D, device=dev)
    cu = torch.arange(0, N + 1, T, dtype=torch.int64, device=dev)
    return q, k, v, gate, beta, state, cu


def workspaces():
    out = []
    for ref in gc.get_referents(gp._get_compiled_cache):
        if isinstance(ref, dict):
            for cfg in ref.values():
                if isinstance(cfg, dict):
                    out += [cfg[k] for k in cfg if isinstance(k, str) and k.startswith("workspace_")]
    return out


def fill(t, pattern, seed):
    b = t.view(torch.uint8)
    if pattern == "zero":
        b.zero_()
    elif pattern == "ff":
        b.fill_(0xFF)
    elif pattern == "random":
        b.copy_(torch.randint(0, 256, b.shape, dtype=torch.uint8, device=dev, generator=torch.Generator(device=dev).manual_seed(seed)))
    elif pattern == "nanf32":
        b.view(torch.float32)[: b.numel() // 4].fill_(float("nan"))


def free_memory(pattern, seed):
    # hand the allocator blocks holding `pattern`, then free them (the kernel's own temporaries may land there)
    hold = []
    for n in (1 << 16, 1 << 20, 1 << 22, 1 << 24, 1 << 26):
        t = torch.empty(n, dtype=torch.uint8, device=dev)
        fill(t, pattern, seed + n)
        hold.append(t)
    del hold


def run(args, ws_pattern, mem_pattern, seed):
    q, k, v, gate, beta, state, cu = args
    for w in workspaces():
        fill(w, ws_pattern, seed)
    free_memory(mem_pattern, seed)
    out, final = chunk_gated_delta_rule(q=q, k=k, v=v, g=gate, beta=beta, initial_state=state.clone(), output_final_state=True, cu_seqlens=cu)
    torch.cuda.synchronize()
    return out.clone(), final.clone()


for seqs, T in ((4, 128), (1, 64), (1, 512)):
    args = inputs(seqs, T)
    ref = run(args, "zero", "zero", 0)
    print(f"seqs {seqs} T {T}: reference nan out {int(ref[0].isnan().sum())} / {ref[0].numel()}, workspaces {[w.numel() for w in workspaces()]}", flush=True)
    for ws_pattern in ("zero", "ff", "random", "nanf32"):
        for mem_pattern in ("zero", "random", "nanf32"):
            bad = 0
            nan = 0
            for seed in range(4):
                o, f = run(args, ws_pattern, mem_pattern, seed)
                same = torch.equal(o, ref[0]) and torch.equal(f, ref[1])
                bad += not same
                nan = max(nan, int(o.isnan().sum()))
            print(f"   workspace {ws_pattern:7s} free memory {mem_pattern:7s}: differs {bad}/4, max nan {nan}", flush=True)
