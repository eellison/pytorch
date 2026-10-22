# Torch-level probe for the Vc trace-call mismatch (fn_vc1 / fn_vc2): under vLLM's compile mode a disabled CustomOp inside an
# opaque custom op (vllm.moe_forward_shared) runs its forward_native through torch.compile (CustomOp.maybe_compile,
# dynamic=True). Under a host trace that compiled function is called with traced tensors and declines ("torch.compile
# cannot JIT compile during CUDA graph capture"), so the custom op is an eager step. Question: is the TRACE call's output
# eager's, and are replays eager's? A custom op whose body calls a torch.compile'd silu-and-mul (warmed up eagerly first),
# followed by a traced op on its result; HostTraceReplay at a few sizes; each call compared with a fresh eager call.
#   gpu3.sh r_nc bash python_fn.sh tests/repro_nested_compile.py
import collections

import torch
import torch.nn.functional as F

import torch.cuda._host_trace as ht
import torch.cuda._host_trace_replay as R

if __import__("os").environ.get("FN_WITNESS_INFRA") == "1":  # the fix (b) prototype (scripts/witness_infra.py)
    import sys as _sys

    _sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))), "scripts"))
    import witness_infra

    witness_infra.install()
import torch.cuda._host_trace_tape as T

DECLINES = collections.Counter()
_init = ht.Declined.__init__


def _rec(exc, *a, **k):
    _init(exc, *a, **k)
    DECLINES[str(exc)[:200]] += 1


ht.Declined.__init__ = _rec


def silu_mul_native(x):
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]


silu_mul_compiled = torch.compile(silu_mul_native, dynamic=True)
STATE = {"calls": 0}


@torch.library.custom_op("fnrepro::shared_mlp", mutates_args=())
def shared_mlp(x: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor) -> torch.Tensor:
    STATE["calls"] += 1
    return silu_mul_compiled(x @ w_up) @ w_down


@shared_mlp.register_fake
def _(x, w_up, w_down):
    return x.new_empty(x.shape[0], w_down.shape[1])


def step(x, w_up, w_down, bias):
    y = torch.ops.fnrepro.shared_mlp(x, w_up, w_down)
    return (y * 2 + bias,)


torch.manual_seed(0)
H, I = 256, 512
w_up = torch.randn(H, 2 * I, device="cuda", dtype=torch.bfloat16) * 0.05
w_down = torch.randn(I, H, device="cuda", dtype=torch.bfloat16) * 0.05
bias = torch.randn(H, device="cuda", dtype=torch.bfloat16)
for n in (3, 17):  # warm the compiled function eagerly (dynamic), as vLLM's warm-up does
    step(torch.randn(n, H, device="cuda", dtype=torch.bfloat16), w_up, w_down, bias)
entry = R.HostTraceReplay(step)
rows = []
for n in (4, 8, 4, 8, 64, 4, 64):
    x = torch.randn(n, H, device="cuda", dtype=torch.bfloat16)
    want = step(x, w_up, w_down, bias)[0]
    want_native = silu_mul_native(x @ w_up) @ w_down * 2 + bias
    t0, r0, e0 = entry.traces, entry.replays, entry.eager
    got = entry(x, w_up, w_down, bias)[0]
    torch.cuda.synchronize()
    path = "trace" if entry.traces > t0 else "replay" if entry.replays > r0 else "eager"
    rows.append((n, path, torch.equal(got, want), torch.equal(got, want_native), torch.equal(want, want_native)))
    print("CALL", n, path, "== compiled eager:", rows[-1][2], " == uncompiled eager:", rows[-1][3],
          " (compiled == uncompiled:", rows[-1][4], ")", flush=True)
steps = collections.Counter(getattr(rec, "name", "?") for v in entry.variants for _, rec in v.tape.launches if isinstance(rec, T.EagerCall))
print("traces", entry.traces, "replays", entry.replays, "eager", entry.eager, "eager steps", dict(steps), flush=True)
for k, c in DECLINES.most_common(6):
    print("DECLINE", c, k, flush=True)
