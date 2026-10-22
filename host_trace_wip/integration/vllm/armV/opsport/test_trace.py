# The five vLLM _C launchers (armV/vllm_ops_trace.py) under a host trace, at Qwen3-8B's shapes and arm V's operand
# forms: per case, trace -> KernelLaunch records; replay over a T sweep, bitwise vs eager; traces = guard regions.
# "off" runs the same with traced_impls off (the default path). TS="all": T 1..8192.
#   CUDA_VISIBLE_DEVICES=0 flock .../gpu0.lock taskset -c 0-71 bash python_vllm_cpp.sh armV/opsport/test_trace.py [CASE...] [off]
import os
import sys
import traceback

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import vllm._custom_ops  # noqa: E402,F401  (registers torch.ops._C*, loads _C)
import vllm_ops_trace  # noqa: E402

import torch.cuda._host_trace as ht  # noqa: E402
from torch.cuda import _host_trace_replay  # noqa: E402
from torch.cuda._host_trace_launch import KernelLaunch  # noqa: E402
from torch.cuda._host_trace_tape import trace  # noqa: E402

HID, Q, K, D, INTER, POS, BLOCK, NB = 4096, 32, 8, 128, 12288, 40960, 16, 1024
QKV = (Q + 2 * K) * D
_TS = os.environ.get("TS", "1 2 3 7 8 13 64 100 128 129 255 256 257 512 1000 2048 4096 8192")
TS = list(range(1, 8193)) if _TS == "all" else [int(t) for t in _TS.split()]
g = torch.Generator(device="cuda").manual_seed(0)
EPS = 1e-6


def randn(*shape, dtype=torch.bfloat16):
    return torch.randn(*shape, device="cuda", generator=g).to(dtype)


def empty(*shape):
    return torch.empty(*shape, device="cuda", dtype=torch.bfloat16)


# input_layernorm of layer 0, the final norm: rms_norm on [T, hidden]
def rms_norm_case(t):
    return (empty(t, HID), randn(t, HID), randn(HID)), 0


def rms_norm_f(out, x, w):
    torch.ops._C.rms_norm(out, x, w, EPS)


# q_norm / k_norm: rms_norm on the [T, heads, D] view of qkv's q slice (rank 3, row stride QKV)
def qk_norm_case(t):
    return (empty(t, Q, D), randn(t, QKV), randn(D)), 0


def qk_norm_f(out, qkv, w):
    torch.ops._C.rms_norm(out, qkv[:, : Q * D].view(-1, Q, D), w, EPS)


def fused_add_case(t):
    return (randn(t, HID), randn(t, HID), randn(HID)), (0, 1)


def fused_add_f(x, res, w):
    torch.ops._C.fused_add_rms_norm(x, res, w, EPS)


def rotary_case(t):
    pos = torch.randint(0, POS, (t,), device="cuda", generator=g)
    return (pos, randn(t, Q * D), randn(t, K * D), randn(POS, D)), (1, 2)


def rotary_f(pos, q, k, cache):
    torch.ops._C.rotary_embedding(pos, q, k, D, cache, True)


def silu_case(t):
    return (empty(t, INTER), randn(t, 2 * INTER)), 0


def silu_f(out, x):
    torch.ops._C.silu_and_mul(out, x)


# FlashInferImpl.do_kv_cache_update: key/value the [T, K, D] views of qkv, caches the split of kv_cache (B, H, N, 2D)
def cache_case(t):
    slots = torch.randperm(NB * BLOCK, device="cuda", generator=g)[:t]
    kv = torch.zeros(NB, K, BLOCK, 2 * D, device="cuda", dtype=torch.bfloat16)
    one = torch.ones((), device="cuda")
    return (randn(t, QKV), kv, slots, one, one.clone()), 1


def cache_f(qkv, kv, slots, ks, vs):
    k = qkv[:, Q * D : (Q + K) * D].view(-1, K, D)
    v = qkv[:, (Q + K) * D :].view(-1, K, D)
    kc, vc = kv.transpose(1, 2).split(D, dim=-1)
    torch.ops._C_cache_ops.reshape_and_cache_flash(k, v, kc, vc, slots, "auto", ks, vs)


OPS = {"rms_norm": (rms_norm_case, rms_norm_f), "qk_norm": (qk_norm_case, qk_norm_f), "fused_add": (fused_add_case, fused_add_f),
       "rotary": (rotary_case, rotary_f), "silu": (silu_case, silu_f), "cache": (cache_case, cache_f)}


class Replay(_host_trace_replay.HostTraceReplay):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._called = True


def clone(args):
    return [a.clone() if isinstance(a, torch.Tensor) else a for a in args]


def run(name, on):
    case, f = OPS[name]
    ht.traced_impls = on
    args, outs = case(TS[0])
    tape = trace(f, clone(args))
    kinds = [type(r).__name__ + ("" if isinstance(r, KernelLaunch) else f"({getattr(r, 'reason', '')})") for _, r in tape.launches]
    print(name, "on" if on else "off", "trace:", [k[:400] for k in kinds], flush=True)
    traced = bool(tape.launches) and all(isinstance(r, KernelLaunch) for _, r in tape.launches)
    r = Replay(f)
    ok = True
    outs = (outs,) if isinstance(outs, int) else outs
    firsts = []
    for t in TS:
        args, _ = case(t)
        ref, mine = clone(args), clone(args)
        f(*ref)
        before = r.traces
        r(*mine)
        torch.cuda.synchronize()
        same = all(torch.equal(mine[i], ref[i]) for i in outs)
        ok &= same
        if r.traces != before:
            firsts.append(t)
        if not same or len(TS) <= 64:
            print("  T", t, "bitwise" if same else "DIFFERENT", "(traced)" if r.traces != before else "")
    print(f"  {len(TS)} Ts, all bitwise: {ok}; traced at T {firsts}")
    print("  traces/replays/eager/uncaptured/structural", (r.traces, r.replays, r.eager, r.uncaptured, r.structural), flush=True)
    variants = sum(len(f) for f in r._families.values())
    print("  variants/folds/relowers/redispatches/guards", (variants, r.folds, r.relowers, r.redispatches, r.guards), flush=True)
    for why in r._reasons:
        print("  reason:", str(why)[:600])
    return ok and (traced or not on), r.traces


def main():
    import logging

    logging.basicConfig()
    names = [n for n in sys.argv[1:] if n != "off"] or list(OPS)
    vllm_ops_trace.install()
    results = {}
    for on in (True, False) if "off" in sys.argv else (True,):
        for n in names:
            try:
                results[(n, on)] = run(n, on)
            except Exception:
                traceback.print_exc()
                results[(n, on)] = (False, None)
    print("TRACE", results)


if __name__ == "__main__":
    main()
