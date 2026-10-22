# Two retrace causes from the vLLM range sweep, standalone (torch only). Prints each call's traces/redispatches and the
# entry's retrace_causes.
# (i)  a copy_ whose host picks a memcpy at n == 1 (the strided source is contiguous there) and a copy kernel above:
#      a redispatch / fold refuses a memcpy record (_host_trace_lower_tape.py: FoldRefused "a memcpy in ..."), so a flip
#      retraces with cause ('dispatch', 'aten.copy_.default', (n - 1) == 0, ..., 'a memcpy in aten.copy_.default').
# (ii) a full-width slice: buf[:, :n] at n == width (vLLM block_table.py:222 slot_mappings[:, :num_tokens]) guards
#      n < width at graph level, where a symbolic Min(n, width) end would serve both.
import torch
import torch.cuda._host_trace_replay as R

dev = torch.device("cuda")


def memcpy_flip(x, out):
    col = x[:, 0]  # stride x.shape[1]: contiguous only when n == 1
    o = out[: x.shape[0]]
    o.copy_(col)
    return (o * 2,)


def full_width(buf, x):
    s = buf[:, : x.shape[0]]
    return (s * 2 + x.sum(),)


def run(name, fn, make, sizes):
    e = R.HostTraceReplay(fn)
    for n in sizes:
        args = make(n)
        (out,) = e(*args)
        (ref,) = fn(*args)
        torch.cuda.synchronize()
        print(f"{name} n={n}: {'bitwise' if torch.equal(out, ref) else 'DIFFERS'} traces {e.traces} redispatches {getattr(e, 'redispatches', '?')} variants {len(e.variants)}", flush=True)
    print(f"{name} retrace_causes: {getattr(e, 'retrace_causes', 'n/a (no retrace_causes in this build)')}")
    print(f"{name} declines: {sorted({str(d)[:200] for d in e.declines})[:3]}")


out = torch.zeros(64, device=dev)
run("memcpy_flip", memcpy_flip, lambda n: (torch.randn(n, 8, device=dev), out), [4, 4, 4, 1, 1, 6, 1])
buf = torch.randn(16, 64, device=dev)
run("full_width", full_width, lambda n: (buf, torch.randn(n, device=dev)), [8, 8, 8, 64, 64, 32, 64])
