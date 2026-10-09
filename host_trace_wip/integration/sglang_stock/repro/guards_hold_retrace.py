# Repro hunt for retrace cause ('other', None, 'its guards hold', None, None) (seen once in the stock SGLang decode
# sweep s2b_fluct, decode bs 256 -> 1): _retrace_cause found the closest variant's graph guards holding and no
# redispatch refusal recorded, yet the call traced again. Candidate path: a learning variant (cuBLAS keys learned out of
# band) whose relower at a new key returns None (_call_held -> _relower -> None -> _miss). Here: Qwen3-8B-shaped decode
# GEMMs (K 4096; N 4096 / 6144 / 24576 / 151936) under the blas harvest, M swept 256 -> 1 as the decode batch shrinks,
# twice; every call checked against eager. Prints counters and retrace_causes; exits 3 if 'its guards hold' is seen.
#   launcher guards_hold_retrace.py
import sys

import torch
import torch.cuda._host_trace_replay as R
from torch.cuda._host_trace_harvest import HarvestProvider

dev = torch.device("cuda")
torch.manual_seed(0)
ws = [torch.randn(n, 4096, device=dev, dtype=torch.bfloat16) * 0.02 for n in (4096, 6144, 24576, 151936)]


def step(x, w0, w1, w2, w3):
    h = torch.nn.functional.linear(x, w1)[:, :4096] + torch.nn.functional.linear(x, w0)
    g = torch.nn.functional.linear(h, w2)[:, :4096]
    return (torch.nn.functional.linear(g, w3),)


e = R.HostTraceReplay(step, opaque=(HarvestProvider(("blas",)),))
for rnd in range(2):
    for m in range(256, 0, -1):
        x = torch.randn(m, 4096, device=dev, dtype=torch.bfloat16)
        (out,) = e(x, *ws)
        ref = step(x, *ws)[0]
        assert torch.equal(out, ref), (rnd, m)
    causes = dict(getattr(e, "retrace_causes", {}))
    print(f"round {rnd}: traces {e.traces} replays {e.replays} eager {e.eager} variants {len(e.variants)} "
          f"relowers {getattr(e, 'relowers', None)} redispatches {getattr(e, 'redispatches', None)} learned {getattr(e, 'learned', None)}", flush=True)
    for c, n in causes.items():
        print(f"  {n}x {c}", flush=True)
hit = any(c[2] == "its guards hold" for c in getattr(e, "retrace_causes", {}))
print("its guards hold seen:", hit)
sys.exit(3 if hit else 0)
