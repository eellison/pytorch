# Aliased tensor arguments under HostTraceReplay: write through a full buffer, read through a slice of it.
import torch
import torch.cuda._host_trace_replay as R
full = torch.zeros(16, device="cuda")
src = torch.zeros(16, device="cuda")
def fn(full, view, src):
    full.copy_(src)
    return (view * 2,)
e = R.HostTraceReplay(fn, memory="run_buffer")
ok = True
for i in range(8):
    src.fill_(i)
    (out,) = e(full, full[:4], src)
    ok &= bool((out == 2 * i).all())
print("replays", e.replays, "traces", e.traces, "ok", ok)
