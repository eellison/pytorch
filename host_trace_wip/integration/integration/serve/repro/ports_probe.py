# Why the vllm_ops_trace launcher ports do not engage under vLLM 0.29 (q8p: every port op an eager step, reason None =
# "no traced host"). No model: vLLM's _C ops called the way 0.29's layers call them, traced with the ports registered.
# Prints per case: traces / replays / variants, bitwise vs eager, and each variant's eager steps with their reasons.
#   CUDA_VISIBLE_DEVICES=N CAND_LINE=c4 bash python_vllm_cand2.sh serve/repro/ports_probe.py
import os
import sys

import torch
import torch.cuda._host_trace_replay as R
import torch.cuda._host_trace_tape as T
from torch.cuda._host_trace_harvest import HarvestProvider

import vllm._custom_ops as ops  # noqa: F401  (loads _C, registers the ops)
from vllm import ir
import vllm.kernels  # noqa: F401  (registers the vllm_c IR impls)

sys.path.insert(0, os.environ.get("PORTS_DIR", "/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration/vllm/armV_stock_fast_m"))
import vllm_ops_trace  # noqa: E402

print("registered", [str(o) for o in vllm_ops_trace.install()], "torch_wrap", ir.op._ENABLE_TORCH_WRAP, flush=True)
H = 4096
W = torch.randn(H, device="cuda", dtype=torch.bfloat16)


def c_rms(x, w):
    out = torch.empty_like(x)
    torch.ops._C.rms_norm(out, x * 1, w, 1e-6)
    return (out + 0,)


def c_silu(x, w):
    out = torch.empty(x.shape[0], H // 2, device=x.device, dtype=x.dtype)
    torch.ops._C.silu_and_mul(out, x * 1)
    return (out + 0,)


def ir_rms(x, w):
    with ir.op.enable_torch_wrap(False):
        return (ir.ops.rms_norm(x * 1, w, 1e-6) + 0,)


def ir_rms_wrapped(x, w):
    with ir.op.enable_torch_wrap(True):
        return (ir.ops.rms_norm(x * 1, w, 1e-6) + 0,)


def ir_add_wrapped(x, w):
    r = x * 2
    with ir.op.enable_torch_wrap(True):
        y, r2 = ir.ops.fused_add_rms_norm(x * 1, r, w, 1e-6)
    return (y + 0, r2 + 0)


def eager_steps(entry):
    out = []
    for v in entry.variants:
        for _, rec in v.tape.launches:
            if isinstance(rec, T.EagerCall) and not rec.host:
                out.append((rec.name, rec.reason))
    return out


with ir.ops.rms_norm.set_priority(["vllm_c", "native"]), ir.ops.fused_add_rms_norm.set_priority(["vllm_c", "native"]):
    for name, fn in (("c_rms", c_rms), ("c_silu", c_silu), ("ir_rms", ir_rms), ("ir_rms_wrapped", ir_rms_wrapped), ("ir_add_wrapped", ir_add_wrapped)):
        entry = R.HostTraceReplay(fn, opaque=(HarvestProvider(("blas",)),), memory="run_buffer")
        ok, err = [], None
        for m in (8, 40, 8, 40, 100):
            x = torch.randn(m, H, device="cuda", dtype=torch.bfloat16)
            try:
                got = entry(x, W)
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                break
            ok.append(all(torch.equal(g, h) for g, h in zip(got, fn(x, W))))
        print(name, "traces", entry.traces, "replays", entry.replays, "eager", entry.eager, "variants", len(entry.variants), "bitwise", ok,
              "eager steps", sorted(set(eager_steps(entry)))[:6], "declines", sorted({str(d)[:300] for d in entry.declines})[:3], "escaped", err, flush=True)
