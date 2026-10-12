# the three new vLLM _C launchers (scaled_fp4_quant.out, static_scaled_fp8_quant, reshape_and_cache_flash fp8) and silu_and_mul
# under HostTraceReplay with no provider (SOUND): per op, calls over several token counts, outputs bitwise vs the op run eagerly,
# and the variants' eager steps (must be none: each launch is the port's record, drift-checked against the real op's capture).
import collections, sys
import os
sys.path.insert(0, os.environ["VLLM_PORTS_DIR"])
import torch
import vllm._custom_ops  # noqa: F401  (registers the _C ops)
import torch.cuda._host_trace as ht
from torch.cuda._host_trace_replay import HostTraceReplay
from torch.cuda._host_trace_tape import EagerCall
import vllm_ops_trace

print("installed", [str(o) for o in vllm_ops_trace.install()], flush=True)
DECL = collections.Counter()
_init = ht.Declined.__init__
def _rec(exc, *a, **k):
    _init(exc, *a, **k); DECL[str(exc)[-700:] if "the op launched" in str(exc) else str(exc)[:240]] += 1
ht.Declined.__init__ = _rec
dev = "cuda"
K = 5120
def fp4(x, s, out, sf):
    torch.ops._C.scaled_fp4_quant.out(x, s, True, output=out, output_scale=sf)
    return out, sf
def fp4_args(m, g):
    x = torch.randn(m, K, device=dev, generator=g).to(torch.bfloat16)
    s = torch.tensor([448 * 6 / 3.0], device=dev)
    return [x, s, torch.empty(m, K // 2, dtype=torch.uint8, device=dev), torch.empty((m + 127) // 128 * 128, K // 64, dtype=torch.int32, device=dev)]
def fp4_rowmajor(x, s, out, sf):
    torch.ops._C.scaled_fp4_quant.out(x, s, False, output=out, output_scale=sf)
    return out, sf
def fp4r_args(m, g):
    x, s, out, _ = fp4_args(m, g)
    return [x, s, out, torch.empty(m, K // 16, dtype=torch.uint8, device=dev)]
def fp8(out, x, s):
    torch.ops._C.static_scaled_fp8_quant(out, x, s)
    return out
def fp8_args(m, g):
    x = torch.randn(m, 6144, device=dev, generator=g).to(torch.bfloat16)
    return [torch.empty(m, 6144, dtype=torch.float8_e4m3fn, device=dev), x, torch.tensor([0.05], device=dev)]
NB, B, H, D = 64, 32, 4, 256
KV = torch.zeros(NB, 2, B, H, D, dtype=torch.uint8, device=dev)
def cache(kv, k, v, slots, ks, vs):
    torch.ops._C_cache_ops.reshape_and_cache_flash(k, v, kv[:, 0], kv[:, 1], slots, "fp8", ks, vs)
    return kv
def cache_args(m, g):
    k = torch.randn(m, H, D, device=dev, generator=g).to(torch.bfloat16)
    v = torch.randn(m, H, D, device=dev, generator=g).to(torch.bfloat16)
    slots = torch.randperm(NB * B, generator=torch.Generator().manual_seed(m))[:m].to(dev)
    return [KV, k, v, slots, torch.tensor([0.1], device=dev), torch.tensor([0.2], device=dev)]
def silu(out, x):
    torch.ops._C.silu_and_mul(out, x)
    return out
def silu_args(m, g):
    return [torch.empty(m, 17408, dtype=torch.bfloat16, device=dev), torch.randn(m, 34816, device=dev, generator=g).to(torch.bfloat16)]
def clone(r):
    return tuple(t.clone() for t in r) if isinstance(r, tuple) else r.clone()
for name, f, mk in (("fp4_swizzled", fp4, fp4_args), ("fp4_rowmajor", fp4_rowmajor, fp4r_args), ("fp8_static", fp8, fp8_args),
                    ("cache_fp8", cache, cache_args), ("silu_and_mul", silu, silu_args)):
    DECL.clear()
    r = HostTraceReplay(f)
    ok = []
    for m in (1, 7, 8, 64, 100, 129, 1000, 2056):
        g = torch.Generator(device=dev).manual_seed(m)
        args = mk(m, g)
        KV.zero_()
        want = clone(f(*[a.clone() if isinstance(a, torch.Tensor) and a is not KV else a for a in args]))
        KV.zero_()
        got = clone(r(*args))
        ok.append(all(torch.equal(a.view(torch.uint8) if a.dtype == torch.float8_e4m3fn else a, b.view(torch.uint8) if b.dtype == torch.float8_e4m3fn else b)
                      for a, b in zip(got if isinstance(got, tuple) else (got,), want if isinstance(want, tuple) else (want,))))
    eager = [rec.name for v in r.variants for _, rec in v.tape.launches if isinstance(rec, EagerCall)]
    reasons = {(rec.reason or "")[:300] for v in r.variants for _, rec in v.tape.launches if isinstance(rec, EagerCall)}
    print(f"PORT {name}: bitwise {sum(ok)}/{len(ok)} traces {r.traces} replays {r.replays} variants {len(r.variants)} redispatches {getattr(r, 'redispatches', None)} "
          f"eager {eager} reasons {reasons} retrace {dict(r.retrace_causes)} declines {dict(DECL)}", flush=True)
