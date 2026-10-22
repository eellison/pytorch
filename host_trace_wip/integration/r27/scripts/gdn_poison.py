# r27: does FlashInfer's GDN chunked prefill (the call vLLM makes for Qwen3.8-27B, gdn_prefill.chunk_gated_delta_rule) read memory
# it did not write? Same inputs, run with the caching allocator's free blocks pre-filled with NaN vs with zeros (the API allocates
# its output / final state / workspace with torch.empty), and with its cached workspace poisoned. Eager only, no host trace.
import torch
from flashinfer.gdn_prefill import chunk_gated_delta_rule
import flashinfer.gdn_kernels.blackwell.gdn_prefill as gp

torch.manual_seed(0)
dev = "cuda"
HK, HV, D = 16, 48, 128


def inputs(T):
    g = torch.Generator(device=dev).manual_seed(T)
    q = torch.nn.functional.normalize(torch.randn(T, HK, D, device=dev, generator=g), dim=-1).to(torch.bfloat16)
    k = torch.nn.functional.normalize(torch.randn(T, HK, D, device=dev, generator=g), dim=-1).to(torch.bfloat16)
    v = torch.randn(T, HV, D, device=dev, generator=g).to(torch.bfloat16)
    gate = torch.exp(-torch.rand(T, HV, device=dev, generator=g))
    beta = torch.rand(T, HV, device=dev, generator=g)
    state = torch.zeros(1, HV, D, D, device=dev)
    cu = torch.tensor([0, T], dtype=torch.int64, device=dev)
    return q, k, v, gate, beta, state, cu


def run(args, fill):
    q, k, v, gate, beta, state, cu = args
    torch.cuda.synchronize()
    # free blocks of the sizes the API allocates, holding `fill`
    junk = [torch.full((n,), fill, device=dev) for n in (1 << 16, 1 << 18, 1 << 20, 1 << 22, 1 << 24)]
    del junk
    out, final = chunk_gated_delta_rule(q=q, k=k, v=v, g=gate, beta=beta, initial_state=state.clone(), output_final_state=True, cu_seqlens=cu)
    torch.cuda.synchronize()
    return out.clone(), final.clone()


def poison_workspace(fill):
    for ref in __import__("gc").get_referents(gp._get_compiled_cache):
        if isinstance(ref, dict):
            for cfg in ref.values():
                if not isinstance(cfg, dict):
                    continue
                for key in [k for k in cfg if isinstance(k, str) and k.startswith("workspace_")]:
                    cfg[key].view(torch.uint8).fill_(0xFF if fill != fill else 0)


def same(a, b):
    return bool(((a == b) | (a.isnan() & b.isnan())).all())


for T in (17, 64, 100, 512, 1000):
    args = inputs(T)
    ref = run(args, 0.0)
    rows = []
    for name, fill, ws in (("zero", 0.0, None), ("nan", float("nan"), None), ("zero+ws0", 0.0, 0.0), ("nan+wsFF", float("nan"), float("nan")), ("zero again", 0.0, None)):
        if ws is not None:
            poison_workspace(ws)
        o, f = run(args, fill)
        rows.append((name, same(o, ref[0]), same(f, ref[1]), int(o.isnan().sum()), int(f.isnan().sum())))
    print(f"T={T}", rows, flush=True)
