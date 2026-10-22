import os, sys
T = "/data/eellison/src/pytorch/agent_space/paramgraph/land/core/trtllm_cpp/tests"
sys.path.insert(0, T)
import test_trtllm_cpp_trace as t
import torch
from torch.cuda._host_trace_tape import EagerCall
c = t.TestR1("test_r1_context"); c.setUp()
r = t.HostTraceReplay(c.context)
for i, (q_lens, kv_lens) in enumerate((([64, 64], [64, 300]), ([64, 64], [64, 301]), ([100, 28], [700, 28]), ([64, 64], [64, 2000]))):
    bt, seq_lens = c.tables(kv_lens, i)
    q = torch.randn(sum(q_lens), c.HQ, c.D, device="cuda").to(torch.float8_e4m3fn)
    cu_q = torch.tensor([0, *torch.tensor(q_lens).cumsum(0).tolist()], dtype=torch.int32, device="cuda")
    cu_kv = torch.tensor([0, *torch.tensor(kv_lens).cumsum(0).tolist()], dtype=torch.int32, device="cuda")
    out = torch.empty(sum(q_lens), c.HQ, c.D, device="cuda", dtype=torch.bfloat16)
    r(c.k, c.v, c.ws, q, bt, seq_lens, cu_q, cu_kv, out, max(q_lens), max(kv_lens), len(q_lens))
    print("call", i, "traces", r.traces, "replays", r.replays, "variants", len(r.variants), flush=True)
print("declines", dict(t.DECLINES))
for v in r.variants:
    for _, rec in v.tape.launches:
        print("REC", type(rec).__name__, getattr(rec, "name", "")[:100])
        if isinstance(rec, EagerCall):
            print("  REASON", (rec.reason or "")[:1500])
