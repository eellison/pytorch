# Repro hunt 2 for retrace cause ('other', None, 'its guards hold', None, None). Seen with stock SGLang decode
# (gh_fluct1): at bs 16 (the batch shrinking 17 -> 16) the call missed with no candidate variant (every variant's _fill
# returned None), a family of 2 variants, no redispatch refusal recorded, and the closest variant's graph guards holding.
# bs 16 is where aten.embedding's index_select switches to its small-index kernel (numIndices <= 16, an op-owned guard
# s <= 16), and each new bs is a new cuBLAS key (M = bs) for the final variant's keyed sites. Here: embedding -> 3
# harvested linears, M swept 24 -> 8 twice, every call checked against eager; exits 3 if 'its guards hold' is seen.
#   launcher guards_hold_retrace2.py
import sys

import torch
import torch.cuda._host_trace_replay as R
from torch.cuda._host_trace_harvest import HarvestProvider

dev = torch.device("cuda")
torch.manual_seed(0)
emb = torch.randn(151936, 4096, device=dev, dtype=torch.bfloat16) * 0.02
ws = [torch.randn(n, 4096, device=dev, dtype=torch.bfloat16) * 0.02 for n in (6144, 4096, 4096)]


def step(ids, emb, w0, w1, w2):
    h = torch.nn.functional.embedding(ids, emb)
    q = torch.nn.functional.linear(h, w0)[:, :4096]
    return (torch.nn.functional.linear(torch.nn.functional.linear(q, w1), w2),)


e = R.HostTraceReplay(step, opaque=(HarvestProvider(("blas",)),))
order = list(range(24, 7, -1))
for rnd in range(2):
    for m in order:
        ids = torch.randint(0, 151936, (m,), device=dev)
        for _ in range(2):
            (out,) = e(ids, emb, *ws)
            assert torch.equal(out, step(ids, emb, *ws)[0]), (rnd, m)
    print(f"round {rnd}: traces {e.traces} replays {e.replays} eager {e.eager} variants {len(e.variants)} "
          f"relowers {getattr(e, 'relowers', None)} redispatches {getattr(e, 'redispatches', None)} learned {getattr(e, 'learned', None)}", flush=True)
    for c, n in dict(getattr(e, "retrace_causes", {})).items():
        print(f"  {n}x {c}", flush=True)
hit = any(c[2] == "its guards hold" for c in getattr(e, "retrace_causes", {}))
print("its guards hold seen:", hit)
sys.exit(3 if hit else 0)
