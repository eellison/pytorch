# Does host tracing record pinned-host <-> device copies (aten.copy_ with non_blocking) as graph memcpy nodes, and do
# replays read / write the pinned host buffer when the graph executes (eager non_blocking semantics)?
#   python memcpy_probe.py
import traceback

import torch
import torch.cuda._host_trace_replay as R


def report(name, entry):
    vs = entry.variants
    segs = [len(v.captured.segments) if getattr(v, "captured", None) is not None else None for v in vs]
    print(f"{name}: traces {entry.traces} replays {entry.replays} eager {entry.eager} variants {len(vs)} segments {segs}", flush=True)
    print(f"  declines: {sorted({str(d)[:300] for d in entry.declines})[:4]}", flush=True)


def case(name, fn, make_args, check, calls=4):
    try:
        entry = R.HostTraceReplay(fn)
        for i in range(calls):
            args = make_args(i)
            out = entry(*args)
            torch.cuda.synchronize()
            ok = check(i, args, out)
            print(f"  {name} call {i}: {'OK' if ok else 'WRONG'}", flush=True)
        report(name, entry)
    except Exception as e:
        print(f"{name}: raised {type(e).__name__}: {str(e)[:300]}", flush=True)
        traceback.print_exc(limit=3)


dev = torch.device("cuda")
N = 64
# 1. H2D: a persistent pinned host buffer (same object every call, contents change) copied into a device buffer, then used
h = torch.zeros(N, dtype=torch.int64, pin_memory=True)
d = torch.zeros(N, dtype=torch.int64, device=dev)


def h2d(h, d):
    d.copy_(h, non_blocking=True)
    return (d * 2,)


def h2d_args(i):
    h.copy_(torch.arange(N) + 100 * i)  # host writes before the call, as vLLM's staging
    return (h, d)


case("h2d_persistent_pinned", h2d, h2d_args, lambda i, a, o: torch.equal(o[0].cpu(), (torch.arange(N) + 100 * i) * 2))

# 2. H2D from a different pinned buffer each call (rotating pool, as vLLM's UvaBufferPool / hostcuts' K slots)
pool = [torch.zeros(N, dtype=torch.int64, pin_memory=True) for _ in range(3)]


def h2d_rot_args(i):
    b = pool[i % 3]
    b.copy_(torch.arange(N) + 7 * i)
    return (b, d)


case("h2d_rotating_pinned", h2d, h2d_rot_args, lambda i, a, o: torch.equal(o[0].cpu(), (torch.arange(N) + 7 * i) * 2))

# 3. D2H: a device result copied into a persistent pinned host buffer (the sampler's async output copy)
out_h = torch.zeros(N, dtype=torch.int64, pin_memory=True)
x = torch.zeros(N, dtype=torch.int64, device=dev)


def d2h(x, out_h):
    y = x + 1
    out_h.copy_(y, non_blocking=True)
    return (y,)


def d2h_args(i):
    x.copy_(torch.arange(N, device=dev) * (i + 1))
    return (x, out_h)


case("d2h_persistent_pinned", d2h, d2h_args, lambda i, a, o: torch.equal(out_h, torch.arange(N) * (i + 1) + 1))

# 4. D2D control
d2 = torch.zeros(N, dtype=torch.int64, device=dev)


def d2d(x, d2):
    d2.copy_(x)
    return (d2 + 1,)


case("d2d_control", d2d, lambda i: (x.copy_(torch.arange(N, device=dev) + i), d2), lambda i, a, o: torch.equal(o[0].cpu(), torch.arange(N) + i + 1))

# 5/6. The pinned buffer not an argument (a closure / module global, as vLLM's persistent staging): only device args
hc = torch.zeros(N, dtype=torch.int64, pin_memory=True)


def h2d_closure(d):
    d.copy_(hc, non_blocking=True)
    return (d * 2,)


def h2d_closure_args(i):
    hc.copy_(torch.arange(N) + 3 * i)
    return (d,)


case("h2d_closure_pinned", h2d_closure, h2d_closure_args, lambda i, a, o: torch.equal(o[0].cpu(), (torch.arange(N) + 3 * i) * 2))
oc = torch.zeros(N, dtype=torch.int64, pin_memory=True)


def d2h_closure(x):
    y = x + 1
    oc.copy_(y, non_blocking=True)
    return (y,)


case("d2h_closure_pinned", d2h_closure, lambda i: (x.copy_(torch.arange(N, device=dev) * (i + 2)),), lambda i, a, o: torch.equal(oc, torch.arange(N) * (i + 2) + 1))
