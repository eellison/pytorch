# r27: harvested cuBLAS mm at the 27B prefill shapes (GDN in_proj_qkvz: [M,5120] x [5120,16384]^T, in_proj_ba: N=96, bf16),
# HostTraceReplay with the blas family only (SOUND), followed by an eager step (custom op) as in the 27B prefill variant.
# Learn happens at the trace; between replays the caching allocator is shifted (fillers held / freed, empty_cache) so every
# op-scoped allocation moves. Each replay is compared with eager; the bindings' slots and stale bytes are dumped.
import sys
import torch
import torch.cuda._host_trace as ht
from torch.cuda._host_trace_harvest import HarvestProvider
from torch.cuda._host_trace_replay import HostTraceReplay
from torch.cuda._host_trace_tape import EagerCall

torch.manual_seed(0)
dev = "cuda"
K, N1, N2 = 5120, 16384, 96
W1 = (torch.randn(N1, K, device=dev) / 64).to(torch.bfloat16)
W2 = (torch.randn(N2, K, device=dev) / 64).to(torch.bfloat16)
lib = torch.library.Library("r27bs", "DEF")
lib.define("consume(Tensor a, Tensor b, Tensor(a!) out) -> ()")


def consume(a, b, out):
    n = int((a[:, :8].float().abs().sum() > -1).item())  # host read: an eager step
    out.copy_(a[:, : out.shape[1]].float() * n + b.float().sum(-1, keepdim=True))


lib.impl("consume", consume, "CUDA")
EAGER_STEP = len(sys.argv) > 1 and sys.argv[1] == "eager_step"


def f(x, w1, w2):
    y = torch.nn.functional.linear(x, w1)
    z = torch.nn.functional.linear(x, w2)
    if EAGER_STEP:
        out = torch.empty(x.shape[0], 256, device=x.device)
        torch.ops.r27bs.consume(y, z, out)
        return y, z, out
    return y, z


def shift(i):
    # move the allocator: hold fillers of assorted sizes, free some, and empty the cache every other call
    global HOLD
    HOLD = [torch.empty((1 + (i * 7919) % 13) << 20, dtype=torch.uint8, device=dev) for _ in range(1 + i % 4)]
    if i % 2:
        torch.cuda.empty_cache()


HOLD = []
for M in (64, 63, 128):
    provider = HarvestProvider(("blas",))
    r = HostTraceReplay(f, opaque=(provider,), memory="run_buffer")
    bad = []
    for i in range(10):
        x = torch.randn(M, K, device=dev).to(torch.bfloat16)
        want = f(x, W1, W2)
        got = r(x, W1, W2)
        ok = all(torch.equal(a, b) for a, b in zip(got, want))
        if not ok:
            bad.append((i, [int(a.isnan().sum()) for a in got], [float((a.float() - b.float()).nan_to_num().abs().max()) for a, b in zip(got, want)]))
        shift(i)
    eager = sum(isinstance(rec, EagerCall) for v in r.variants for _, rec in v.tape.launches)
    print(f"M={M} eager_step={EAGER_STEP}: traces {r.traces} replays {r.replays} variants {len(r.variants)} eager_steps {eager} "
          f"bindings {len(provider.bindings)} refused {getattr(provider, 'refused', None) and list(provider.refused)[:3]} stale {provider.stale[:3]} bad {len(bad)}/10 {bad[:3]}", flush=True)
    for key, b in list(provider.bindings.items())[:4]:
        nodes = getattr(b, "nodes", ())
        print("   BINDING", str(key)[:160], [(getattr(n, "name", type(n).__name__)[:70] if hasattr(n, "name") else type(n).__name__) for n in nodes][:4], flush=True)
