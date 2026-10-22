# r27: a traced region with eager steps between graph runs (the 27B prefill shape: proj -> custom op that reads the proj output and
# writes a state pool + an output -> proj), replayed with memory="run_buffer" vs "eager", with the caching allocator's state
# perturbed between calls (junk blocks of the run's sizes held / freed). Bitwise vs the function run eagerly, state included.
import sys
import torch
import torch.cuda._host_trace as ht
from torch.cuda._host_trace_replay import HostTraceReplay
from torch.cuda._host_trace_tape import EagerCall

lib = torch.library.Library("r27rb", "DEF")
lib.define("core(Tensor x, Tensor(a!) out, Tensor(b!) pool, Tensor idx) -> ()")


def core_impl(x, out, pool, idx):
    # data-dependent host work (no traced host): an eager step; reads x, writes pool rows and out
    rows = idx.nonzero().flatten()
    st = pool[rows]
    st.mul_(0.5).add_(x[-1:].float().sum(0, keepdim=True))
    pool[rows] = st
    out.copy_(torch.tanh(x) * st.mean())


lib.impl("core", core_impl, "CUDA")
D, L = 1024, int(sys.argv[2]) if len(sys.argv) > 2 else 4
W = [torch.randn(D, D, device="cuda") / 32 for _ in range(L)]


def f(x, pool, idx):
    h = x
    for w in W:
        a = h * 1.5 + 0.25          # graph run
        out = torch.zeros_like(a)
        torch.ops.r27rb.core(a, out, pool, idx)   # eager step
        h = out * 2 - a             # graph run
    return h


def run(memory, perturb):
    torch.manual_seed(0)
    pool = torch.randn(16, D, device="cuda")
    r = HostTraceReplay(f, memory=memory)
    bad = 0
    junk = []
    for i in range(12):
        T = 64
        x = torch.randn(T, D, device="cuda")
        idx = torch.zeros(16, dtype=torch.int32, device="cuda")
        idx[i % 16] = 1
        ref_pool = pool.clone()
        want = f(x.clone(), ref_pool, idx)
        got = r(x, pool, idx)
        ok = torch.equal(got, want) and torch.equal(pool, ref_pool)
        bad += not ok
        pool.copy_(ref_pool)
        if perturb:  # hold / free blocks of the run's sizes, as a model's other steps would
            junk.append(torch.empty((i % 5 + 1) * T * D, device="cuda"))
            if len(junk) > 3:
                junk.pop(0)
    eager = sum(isinstance(rec, EagerCall) for v in r.variants for _, rec in v.tape.launches)
    print(f"memory={memory} perturb={perturb}: traces {r.traces} replays {r.replays} variants {len(r.variants)} eager_steps {eager} bad {bad}/12", flush=True)


for memory in ("run_buffer", "eager", "held"):
    for perturb in (False, True):
        run(memory, perturb)
