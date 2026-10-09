# run_buffer liveness after a respec (vLLM open-variant NaN, standalone): a step with a dispatch_unit kernel choice
# whose branches differ in launches and allocations (so the replay respecs, not redispatches), two cuBLAS mms harvested
# as opaque calls, and traced pointwise temporaries. The first call's size is traced; another index size makes the variant's selector unselected, so the
# replay builds a respec'd variant (no trace), whose mm sites rebind at the new M. In vLLM, the respec'd variant's
# re-bound opaque launches kept their sequence numbers while the allocations they use got new, later ones, so the
# run_buffer plan saw those allocations live over an empty or inverted interval and gave their bytes to other
# temporaries (NaN outputs). This script checks each call bitwise against eager and, for every variant, the plan's
# intervals: an allocation used by a launch outside [its seq, its recorded last use].
#   python runbuffer_nan_repro.py [memory] [sizes]   (memory: run_buffer (default) or eager; sizes: comma-separated
#   index sizes, first one traced). Traced at 32 rows, the mms are single-node cuBLAS kernels; at <= 4 rows cuBLAS picks
#   split-K (2 nodes + a 32 MiB scratch), so their redispatch is refused for topology and allocations too.
import sys

import torch
import torch.cuda._host_trace as ht
import torch.cuda._host_trace_replay as R
import triton
import triton.language as tl
from torch.cuda._host_trace_harvest import HarvestProvider
from torch.cuda._host_trace_lower_tape import LoweredLaunch, PointerSlot
from torch.cuda._host_trace_memory import plan_memory

memory = sys.argv[1] if len(sys.argv) > 1 else "run_buffer"
dev = torch.device("cuda")
torch.manual_seed(0)
V, D, F = 4096, 4096, 4096  # Qwen3-8B o_proj-like K/N: cuBLAS picks split-K (2 nodes + scratch) at small M
emb = torch.randn(V, D, device=dev, dtype=torch.bfloat16)
# nn.Linear layout [out, in] as vLLM's weights (F.linear: cuBLAS TN)
w1 = torch.randn(2 * F, D, device=dev, dtype=torch.bfloat16) / D ** 0.5
w2 = torch.randn(D, F, device=dev, dtype=torch.bfloat16) / F ** 0.5
wo = torch.randn(D, D, device=dev, dtype=torch.bfloat16) / D ** 0.5


@triton.jit
def _scale(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(y_ptr + i, tl.load(x_ptr + i, mask=i < n) * 2, mask=i < n)


@ht.dispatch_unit
def choice(x):
    """A Python kernel choice with a different launch topology and allocations per branch (as the trtllm fork's split
    count): one launch at <= 4 rows, a temporary and two launches above. A redispatch refuses it where it flips, so the
    replay respecs the variant."""
    n = x.numel()
    if x.shape[0] <= 4:
        y = torch.empty_like(x)
        _scale[(triton.cdiv(n, 1024),)](x, y, n, BLOCK=1024)
        return y
    t = torch.empty_like(x)
    _scale[(triton.cdiv(n, 1024),)](x, t, n, BLOCK=1024)
    y = torch.empty_like(x)
    _scale[(triton.cdiv(n, 1024),)](t, y, n, BLOCK=1024)
    return y


def step(idx, emb, w1, w2, wo):
    x = choice(torch.nn.functional.embedding(idx, emb))
    x = torch.nn.functional.linear(x, wo)  # cuBLAS right after the respec'd op (the o_proj after attention in vLLM)
    gu = torch.nn.functional.linear(x, w1)  # cuBLAS, harvested (opaque) at M = idx size
    g, u = gu.chunk(2, dim=-1)
    a = torch.nn.functional.silu(g) * u  # traced pointwise temporaries
    y = torch.nn.functional.linear(a, w2)  # cuBLAS, harvested
    return (y + x,)


def check_plan(variant):
    lo = variant.captured.lowered
    plan = plan_memory(lo, "run_buffer")
    live = {k: (seq, last) for st in plan.steps for k, seq, last in st.temporaries}
    bad = []
    for la in lo.launches:
        if not isinstance(la, LoweredLaunch):
            continue
        for sl in la.slots:
            if isinstance(sl, PointerSlot) and sl.base in live:
                seq, last = live[sl.base]
                if not seq <= la.seq <= max(seq, last):
                    bad.append((la.launch.name[:40], la.seq, sl.base, seq, last))
    order = [la.seq for la in lo.launches]
    n_alloc = len(lo.allocations)
    early = []
    for la in lo.launches:
        if isinstance(la, LoweredLaunch):
            for sl in la.slots:
                if isinstance(sl, PointerSlot) and sl.base is not None and sl.base < n_alloc and la.seq < lo.allocations[sl.base].seq:
                    early.append((la.launch.name[:30], la.seq, sl.base, lo.allocations[sl.base].seq))
    return bad + early, order != sorted(order)


torch.set_grad_enabled(False)  # vLLM runs under inference_mode
provider = HarvestProvider(("blas",))
entry = R.HostTraceReplay(step, opaque=(provider,), memory=memory)
ok = True
SIZES = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [32, 32, 32] + [m for m in (4, 2, 1, 3, 8, 16, 5, 64) for _ in (0, 1)]
for i, n in enumerate(SIZES):
    idx = torch.randint(0, V, (n,), device=dev)
    (out,) = entry(idx, emb, w1, w2, wo)
    (ref,) = step(idx, emb, w1, w2, wo)
    torch.cuda.synchronize()
    same = torch.equal(out, ref)
    ok &= same
    print(f"call {i} n={n}: {'bitwise' if same else 'DIFFERS'} nan={bool(out.isnan().any())} traces {entry.traces} variants {len(entry.variants)} "
          f"respecs {getattr(entry, 'respecs', '?')} redispatches {getattr(entry, 'redispatches', '?')}", flush=True)
for j, v in enumerate(entry.variants):
    bad, unordered = check_plan(v)
    lo = v.captured.lowered
    print(f"variant {j} launches: {[getattr(getattr(la, 'launch', None), 'name', type(la).__name__)[:28] for la in lo.launches]}")
    print(f"variant {j} allocation bytes: {[lo.program.values[a.nbytes] if isinstance(a.nbytes, int) and a.nbytes < len(lo.program.values) else '?' for a in lo.allocations]}")
    print(f"variant {j}: allocations {len(lo.allocations)} launches {len(lo.launches)}; launches out of seq order: {unordered}; "
          f"uses outside their allocation's planned interval: {len(bad)} {bad[:4]}")
print("ALL BITWISE" if ok else "SOME CALLS DIFFER")
