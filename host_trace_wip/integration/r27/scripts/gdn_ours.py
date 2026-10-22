# gdn_ours.py: FlashInfer GDN prefill under OUR host-trace replay (HostTraceReplay, CuTe route armed), no vLLM: per shape, the replayed
# output vs an eager call in the same process (bitwise, NaN counts), plus plain eager twice. Isolates our replay of this one kernel.
import os
import torch
import cutlass  # noqa: F401
import torch.cuda._host_trace_cute as htc
htc.install()  # before FlashInfer compiles its CuTe kernels
from torch.cuda._host_trace_replay import HostTraceReplay
from flashinfer.gdn_prefill import chunk_gated_delta_rule

dev, HK, HV, D = "cuda", 16, 48, 128


def gdn(q, k, v, gate, beta, st, cu):
    o, f = chunk_gated_delta_rule(q=q, k=k, v=v, g=gate, beta=beta, initial_state=st, output_final_state=True, cu_seqlens=cu)
    return o, f


for spec in os.environ.get("GDN_SHAPES", "1x512").split():
    seqs, T = map(int, spec.split("x"))
    g = torch.Generator(device=dev).manual_seed(0)
    N = seqs * T
    q = torch.nn.functional.normalize(torch.randn(N, HK, D, device=dev, generator=g), dim=-1).to(torch.bfloat16)
    k = torch.nn.functional.normalize(torch.randn(N, HK, D, device=dev, generator=g), dim=-1).to(torch.bfloat16)
    v = torch.randn(N, HV, D, device=dev, generator=g).to(torch.bfloat16)
    gate = torch.exp(-torch.rand(N, HV, device=dev, generator=g))
    beta = torch.rand(N, HV, device=dev, generator=g)
    state0 = torch.zeros(seqs, HV, D, D, device=dev)
    cu = torch.arange(0, N + 1, T, dtype=torch.int64, device=dev)
    e1 = [t.clone() for t in gdn(q, k, v, gate, beta, state0.clone(), cu)]
    e2 = [t.clone() for t in gdn(q, k, v, gate, beta, state0.clone(), cu)]
    r = HostTraceReplay(gdn)
    outs = []
    for i in range(5):
        res = r(q, k, v, gate, beta, state0.clone(), cu)
        torch.cuda.synchronize()
        outs.append([t.clone() for t in res])
    eq = lambda a, b: all(torch.equal(x.nan_to_num(7.0), y.nan_to_num(7.0)) for x, y in zip(a, b))
    print(f"{spec}: traces {r.traces} replays {r.replays} eager {r.eager} | eager-eager {eq(e1, e2)} | call-vs-eager "
          f"{[eq(o, e1) for o in outs]} | nan eager {int(e1[0].isnan().sum())} nan calls {[int(o[0].isnan().sum()) for o in outs]}", flush=True)
    if r.replays == 0:
        print("   declines:", sorted({str(d)[:300] for d in r.declines})[:4], "reasons:", list(getattr(r, "_reasons", {}))[:4], flush=True)
