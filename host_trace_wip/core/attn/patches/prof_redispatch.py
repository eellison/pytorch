# Where a redispatch's time goes: the context op of LAYERS layers traced at one max_kv, then called at new max_kv values
# (the original fork pins a context call's max_kv: each new value redispatches every layer's op). cProfile of the calls.
#   ATTN_FORK=1 ATTN_FORK_DIR=land/core/attn/fork_orig FORK_TEST_LAYERS=36 python_vllm_attn.sh tests/prof_redispatch.py
import cProfile
import io
import os
import pstats
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_trtllm_fork_trace as F  # noqa: E402
from torch.cuda._host_trace_replay import HostTraceReplay  # noqa: E402

t = F.TestTrtllmFork()
t.setUp()
r = HostTraceReplay(F.context_step)
for i, kv in enumerate((300, 300, 300)):
    r(*t.context_args([64, 64], [64, kv], i))
torch.cuda.synchronize()
print("warm: traces", r.traces, "redispatches", r.redispatches)
prof = cProfile.Profile()
times = []
for i, kv in enumerate((301, 420, 2000, 302)):
    args = t.context_args([64, 64], [64, kv], 10 + i)
    torch.cuda.synchronize()
    start = time.perf_counter()
    prof.enable()
    r(*args)
    prof.disable()
    torch.cuda.synchronize()
    times.append(time.perf_counter() - start)
print("layers", F.LAYERS, "call ms", [round(x * 1e3, 1) for x in times], "traces", r.traces, "redispatches", r.redispatches,
      f"redispatch_s {r.redispatch_s * 1e3:.1f} ms", "refusals", r.redispatch_refusals)
s = io.StringIO()
pstats.Stats(prof, stream=s).sort_stats("cumulative").print_stats(45)
print(s.getvalue())
s = io.StringIO()
pstats.Stats(prof, stream=s).sort_stats("tottime").print_stats(30)
print(s.getvalue())
