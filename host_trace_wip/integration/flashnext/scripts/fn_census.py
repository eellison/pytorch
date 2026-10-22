"""Census of the GPU launches of one engine step of stock vLLM (Qwen3.8-Flash-Next NVFP4 by default), per step kind, by route.
Flash-Next lane copy of trtllm_moe_cpp/scripts/mimo_census.py: "registered" is split by dispatcher op into aten (ATen
native), c-port (a vLLM _C op with a launcher port in integration/vllm/vllm_ports: rms_norm, fused_add_rms_norm,
rotary_embedding, silu_and_mul, reshape_and_cache_flash, scaled_fp4_quant, static_scaled_fp8_quant,
silu_and_mul_nvfp4_quant, and the vllm_ir ops wrapping the norms) and c-open (any other vLLM / custom-op launch under a
dispatcher op: flagged, it needs a registered launcher port, never a harvest). Steps: decode bs 1/8/64 (KV 256), decode bs
4 at KV 6000 (QSA selects past 2048), prefill 512/2048/8192, mixed 8 decoders + 512.

Each profiled step's Kineto trace (torch.profiler, CPU + CUDA, with_stack) is saved to out/census/<model>_<step>.json.
Every kernel / memcpy / memset is joined to its runtime or driver launch call by correlation id, and the launch call to
the innermost dispatcher op and the Python frames enclosing it on the same thread. Route of a launch:
  triton      a Python frame in triton/ (the Triton launcher)
  cute        a Python frame in nvidia_cutlass_dsl / cutlass/ (CuTe DSL)
  closed      a closed library's kernel (cuBLAS / cuBLASLt nvjet / cuDNN / NCCL) under a dispatcher op
  registered  under a dispatcher op (ATen native or a vLLM / FlashInfer op registered with the dispatcher)
  fi-moe      FlashInfer's trtllm-gen fused MoE launcher (this lane's C++ route, land/core/trtllm_moe_cpp)
  fi-fmha     FlashInfer's trtllm-gen attention launcher (land/core/r27's C++ route)
  deepgemm    DeepGEMM's launcher (the DeepGEMM lane's route)
  flashinfer  any other FlashInfer launch (flagged)
  other       none of the above (flagged)
Flagged launches are listed by kernel name with their dispatcher op and innermost Python frame.

    python_moe.sh scripts/mimo_census.py [--model M] [--llm-kw JSON]
"""

import argparse
import collections
import json
import os
import re
import sys

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
import torch
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ROUTES = ["triton", "cute", "closed", "aten", "c-port", "c-open", "fi-moe", "fi-fmha", "deepgemm", "flashinfer", "other"]
FLAGGED = {"flashinfer", "other", "c-open"}
C_PORTED = {"_C::rms_norm", "_C::fused_add_rms_norm", "_C::rotary_embedding", "_C::silu_and_mul",
            "_C_cache_ops::reshape_and_cache_flash", "_C::scaled_fp4_quant", "_C::static_scaled_fp8_quant",
            "_C::silu_and_mul_nvfp4_quant", "vllm_ir::rms_norm", "vllm_ir::fused_add_rms_norm"}
CLOSED = re.compile(r"nvjet|cublas|cudnn|nccl|^sm\d+_xmma|gemv|gemm.*cublas|ampere_|cutlass_80_tensorop|splitKreduce", re.I)


def route(kernel, op, frames):
    files = " ".join(frames)
    if re.search(r"(^|[ /])triton/", files):
        return "triton"
    # CuTe DSL kernels are named kernel_cutlass_<module>...; launched from Python or from a cached compiled function
    if kernel.startswith("kernel_cutlass_") or re.search(r"nvidia_cutlass_dsl|cutlass/base_dsl|cutlass/cute/", files):
        return "cute"
    if "deep_gemm" in files:
        return "deepgemm"
    if "flashinfer/" in files:
        if op is not None and op.startswith("aten::") and not CLOSED.search(kernel) and "flashinfer/fused_moe" not in files:
            return "aten"  # an ATen op FlashInfer's Python calls (gdn_prefill.py's int casts): ATen's own route
        if "flashinfer/fused_moe" in files:
            return "fi-moe"
        if re.search(r"flashinfer/(decode|prefill|trtllm_cpp)\.py", files):
            return "fi-fmha"
        return "flashinfer"
    if op is not None:
        if CLOSED.search(kernel):
            return "closed"
        if op.startswith("aten::"):
            return "aten"
        return "c-port" if op in C_PORTED else "c-open"
    return "other"


def analyze(path):
    ev = json.load(open(path))["traceEvents"]
    gpu = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    launch = {e["args"]["correlation"]: e for e in ev if e.get("ph") == "X" and e.get("cat") in ("cuda_runtime", "cuda_driver")
              and "correlation" in e.get("args", {})}
    # enclosing dispatcher ops / Python frames of each launch call: one sweep per thread with a stack of open intervals
    spans = collections.defaultdict(list)
    for e in ev:
        if e.get("ph") == "X" and e.get("cat") in ("cpu_op", "python_function"):
            spans[e["tid"]].append((e["ts"], 0, e))
    for L in launch.values():
        spans[L["tid"]].append((L["ts"], 1, L))
    enclosing = {}
    for items in spans.values():
        items.sort(key=lambda x: (x[0], x[1], -x[2]["dur"]))
        stack = []
        for ts, is_launch, e in items:
            while stack and stack[-1]["ts"] + stack[-1]["dur"] < ts:
                stack.pop()
            if is_launch:
                t1 = ts + e["dur"]
                enclosing[e["args"]["correlation"]] = [x for x in stack if x["ts"] + x["dur"] >= t1]
            else:
                stack.append(e)
    rows = []
    for k in gpu:
        L = launch.get(k["args"].get("correlation"))
        op, frames = None, []
        if L is not None:
            enc = enclosing.get(L["args"]["correlation"], [])
            ops = [e for e in enc if e["cat"] == "cpu_op"]
            if ops:
                op = ops[-1]["name"]
            frames = [e["name"] for e in enc if e["cat"] == "python_function"]
        r = route(k["name"], op, frames)
        inner = next((f for f in reversed(frames) if not re.search(r"torch/(nn|_ops|utils|autograd|profiler)|<built-in", f)), frames[-1] if frames else "")
        rows.append({"kernel": k["name"], "cat": k["cat"], "route": r, "op": op, "frame": inner,
                     "launch": L["name"] if L else None})
    return rows




def summarize(name, rows, tokens):
    counts = collections.Counter(r["route"] for r in rows)
    flagged = collections.Counter((r["route"], r["kernel"][:120], r["op"], r["frame"][:160]) for r in rows if r["route"] in FLAGGED)
    print(f"CENSUS {name} tokens {tokens} launches {len(rows)} " + " ".join(f"{k} {counts.get(k, 0)}" for k in ROUTES), flush=True)
    for (rt, kern, op, frame), n in flagged.most_common():
        print(f"  FLAG {rt} x{n} {kern} | op {op} | {frame}", flush=True)
    return {"tokens": tokens, "launches": len(rows), "routes": {k: counts.get(k, 0) for k in ROUTES},
            "flagged": [list(k) + [n] for k, n in flagged.most_common()],
            "by_route_kernels": {rt: collections.Counter(r["kernel"][:100] for r in rows if r["route"] == rt).most_common()
                                 for rt in ROUTES if counts.get(rt)}}


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[1] == "--analyze":
    # offline: python3 mimo_census.py --analyze OUT.json TRACE.json...  (step name = trace file's suffix)
    out = {}
    for path in sys.argv[3:]:
        name = os.path.basename(path).rsplit("_", 1)[1].removesuffix(".json")
        out[name] = summarize(name, analyze(path), None)
    with open(sys.argv[2], "w") as f:
        json.dump(out, f, indent=1)
    sys.exit(0)
p = argparse.ArgumentParser()
p.add_argument("--model", default="/data/eellison/models/Qwen3.8-Flash-Next-NVFP4")
p.add_argument("--llm-kw", default='{"language_model_only": true}')
p.add_argument("--out", default=os.path.join(ROOT, "out/census"))
a = p.parse_args()
if os.environ.get("FN_QSA_DET") == "1":  # DIAGNOSIS CONFIG: deterministic QSA top-k order (scripts/qsa_det.py)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import qsa_det

    qsa_det.install()
kw = dict(model=a.model, tensor_parallel_size=1, enforce_eager=True, enable_prefix_caching=False, max_model_len=16384,
          gpu_memory_utilization=float(os.environ.get("INTEG_GPU_UTIL", "0.9")), seed=0, trust_remote_code=True)
kw.update(json.loads(a.llm_kw))
# one repr per distinct (token count) of FlashInfer's trtllm_fp4_block_scale_moe call kwargs (for the MoE lane's NVFP4 test)
import flashinfer.fused_moe as _fim

_moe = _fim.trtllm_fp4_block_scale_moe
MOE_CALLS = {}


def _desc(v):
    if isinstance(v, torch.Tensor):
        d = f"T{tuple(v.shape)}:{v.dtype}:s{tuple(v.stride())}"
        return d + (f"={v.item()!r}" if v.numel() == 1 else "")
    return repr(v)


def _moe_logged(*args, **kwargs):
    n = int(kwargs["hidden_states"].shape[0]) if "hidden_states" in kwargs else -1
    if n not in MOE_CALLS:
        MOE_CALLS[n] = {"args": [_desc(x) for x in args], "kwargs": {k: _desc(v) for k, v in kwargs.items()}}
        print("MOE_CALL", n, json.dumps(MOE_CALLS[n]), flush=True)
    return _moe(*args, **kwargs)


_fim.trtllm_fp4_block_scale_moe = _moe_logged
llm = LLM(**kw)
eng = llm.llm_engine
runner = eng.model_executor.driver_worker.worker.model_runner
# MoE layer facts (first FusedMoE layer)
for _name, _m in runner.model.named_modules():
    if type(_m).__name__ == "FusedMoE":
        _facts = {"layer": _name, "class": type(_m).__name__}
        for _k in ("routing_method_type", "top_k", "global_num_experts", "renormalize", "use_grouped_topk", "num_expert_group",
                   "topk_group", "scoring_func", "routed_scaling_factor", "activation", "n_shared_experts", "fuse_shared_experts",
                   "is_fused_shared_expert_enabled", "router_logits_dtype", "intermediate_size_per_partition", "hidden_size"):
            if hasattr(_m, _k):
                _facts[_k] = repr(getattr(_m, _k))
        for _k in ("quant_method", "runner", "shared_experts", "_shared_experts"):
            if getattr(_m, _k, None) is not None:
                _facts[_k] = type(getattr(_m, _k)).__name__
        _qm = getattr(_m, "quant_method", None)
        for _k in ("moe_kernel", "fused_experts", "experts", "kernel"):
            if getattr(_qm, _k, None) is not None:
                _o = getattr(_qm, _k)
                _facts[f"qm.{_k}"] = type(_o).__name__
                for _kk in ("per_token_activation", "routing_method_type", "topk", "intermediate_size_per_partition"):
                    if hasattr(_o, _kk):
                        _facts[f"qm.{_k}.{_kk}"] = repr(getattr(_o, _kk))
                _ex = getattr(_o, "fused_experts", None) or getattr(_o, "experts", None)
                if _ex is not None:
                    _facts[f"qm.{_k}.experts"] = type(_ex).__name__
                    for _kk in ("per_token_activation", "routing_method_type", "topk"):
                        if hasattr(_ex, _kk):
                            _facts[f"qm.{_k}.experts.{_kk}"] = repr(getattr(_ex, _kk))
        print("MOE_FACTS", json.dumps(_facts), flush=True)
        break
TOKENS = []
_fwd = runner.model.forward


def fwd(input_ids, positions, *args, **kwargs):
    TOKENS.append(int(input_ids.shape[0]) if input_ids is not None else int(positions.shape[-1]))
    return _fwd(input_ids, positions, *args, **kwargs)


runner.model.forward = fwd
g = torch.Generator().manual_seed(0)
rid = [0]


def add(n, max_tokens=64):
    rid[0] += 1
    ids = torch.randint(1000, 100000, (n,), generator=g).tolist()
    eng.add_request(str(rid[0]), TokensPrompt(prompt_token_ids=ids), SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True))


def drain():
    while eng.has_unfinished_requests():
        eng.step()


def until_decoding(n):
    while True:
        eng.step()
        if TOKENS and TOKENS[-1] == n:
            return


os.makedirs(a.out, exist_ok=True)
tag = os.path.basename(a.model.rstrip("/"))
census = {}


def profile(name):
    TOKENS.clear()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                with_stack=True) as prof:
        eng.step()
        torch.cuda.synchronize()
    path = os.path.join(a.out, f"{tag}_{name}.json")
    prof.export_chrome_trace(path)
    census[name] = summarize(name, analyze(path), list(TOKENS))


# warm-up pass of every shape below (JIT, autotune), then the profiled pass
for rep in range(2):
    prof_or_step = profile if rep == 1 else (lambda name: eng.step())
    for bs, plen in ((1, 256), (8, 256), (64, 256), (4, 6000)):
        for _ in range(bs):
            add(plen)
        until_decoding(bs)
        eng.step()
        prof_or_step(f"decode{bs}" + ("" if plen == 256 else f"kv{plen}"))
        drain()
    for L in (512, 2048, 8192):
        add(L, max_tokens=1)
        prof_or_step(f"prefill{L}")
        drain()
    for _ in range(8):
        add(256)
    until_decoding(8)
    add(512)
    prof_or_step("mixed8x512")
    drain()

census["moe_calls"] = MOE_CALLS
with open(os.path.join(a.out, f"{tag}_census.json"), "w") as f:
    json.dump(census, f, indent=1)
print("CENSUS_DONE", os.path.join(a.out, f"{tag}_census.json"), flush=True)
os._exit(0)
