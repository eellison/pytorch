"""Qwen3.8-Flash-Next NVFP4 under the armV adapter (arm Vc or V) with every traced forward checked against a fresh eager
reference (V.check), on shapes interleaved round by round. Flash-Next lane copy of trtllm_moe_cpp/scripts/mimo_interleave.py
(the MiMo lane's vc_ilv4 driver): each round runs prefill 64 / 512 / 2048, decode bursts at bs 8, 1 and 64 (KV 256),
a long-KV decode burst (bs 4 at KV 6000: QSA's indexer selects 2048 of the KV past 2048 tokens) and a mixed step
(8 decoders + one 512-token prompt), each with its own fill steps. --kinds limits the shapes (decode / prefill / mixed).
--llm-kw: extra LLM kwargs (e.g. {"gdn_prefill_backend": "triton"}, the diagnosis config).

    run_fn_arm.sh TAG Vc|V [--rounds N] [--kinds decode prefill mixed]   (writes out/TAG.json)
"""

import argparse
import collections
import json
import os
import sys
import time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
import torch

p = argparse.ArgumentParser()
p.add_argument("--arm", choices=["V", "Vc"], required=True)
p.add_argument("--model", required=True)
p.add_argument("--out", required=True)
p.add_argument("--max-model-len", type=int, default=16384)
p.add_argument("--llm-kw", default="{}")
p.add_argument("--kinds", nargs="*", default=["prefill", "decode", "longdecode", "mixed"])
p.add_argument("--rounds", type=int, default=9)
p.add_argument("--trust-remote-code", action="store_true")
p.add_argument("--language-model-only", action="store_true")
p.add_argument("--check", action="store_true")
a, _ = p.parse_known_args()
sys.path.insert(0, os.environ["INTEG_ARMV"])
import adapter as armv_adapter  # noqa: E402

if os.environ.get("ARMV_CUTE_OBSERVE") == "1":
    import cutlass  # noqa: F401
    import torch.cuda._host_trace_cute as _cute

    _cute.install()
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402

extra = {"compilation_config": {"cudagraph_mode": "NONE"}} if a.arm == "Vc" else {}
t0 = time.perf_counter()
llm = LLM(model=a.model, tensor_parallel_size=1, enforce_eager=a.arm == "V", trust_remote_code=a.trust_remote_code,
          **({"language_model_only": True} if a.language_model_only else {}), **extra, enable_prefix_caching=False,
          gpu_memory_utilization=float(os.environ.get("INTEG_GPU_UTIL", "0.9")),
          **({"kv_cache_memory_bytes": int(os.environ["INTEG_KV_BYTES"])} if os.environ.get("INTEG_KV_BYTES") else {}),
          max_model_len=a.max_model_len, seed=0, **json.loads(a.llm_kw))
init_s = time.perf_counter() - t0
eng = llm.llm_engine
runner = eng.model_executor.driver_worker.worker.model_runner
import adapter_cpp  # noqa: E402

# layout of the PLE layer's state (the short conv's conv_state: its decode index_select / slice-assign declines; repro input)
for _n, _m in runner.model.named_modules():
    if "ple" in type(_m).__name__.lower() and hasattr(_m, "kv_cache"):
        _kv = _m.kv_cache if isinstance(_m.kv_cache, (list, tuple)) else [_m.kv_cache]
        for _i, _t in enumerate(x for kv in _kv for x in (kv if isinstance(kv, (list, tuple)) else [kv])):
            if isinstance(_t, torch.Tensor):
                print("PLE_STATE", _n, type(_m).__name__, _i, tuple(_t.shape), tuple(_t.stride()), _t.dtype, _t.storage_offset(), flush=True)

V = (adapter_cpp.ArmVcpp if adapter_cpp.BOUND else armv_adapter.ArmV)(runner)
V.install()
V.check = a.check
runner.model.forward = V.forward
g = torch.Generator().manual_seed(0)


def prompt(n):
    return TokensPrompt(prompt_token_ids=torch.randint(1000, 100000, (n,), generator=g).tolist())


sp = lambda n: SamplingParams(max_tokens=n, ignore_eos=True, temperature=0.0)
llm.generate([prompt(128) for _ in range(4)], sp(8), use_tqdm=False)  # warm-up (the gc.freeze hook runs after it)
PHASE = ["warmup"]
ROUND = [0]
rid = [0]
phase_of_check = ["warmup"] * len(V.checks)  # the phase of each V.checks row, appended as the rows appear


def step():
    n = len(V.checks)
    print(f"STEP round {ROUND[0]} {PHASE[0]} calls {sum(V.calls.values())} checks {n}", flush=True)
    outs = eng.step()
    phase_of_check.extend([PHASE[0]] * (len(V.checks) - n))
    for c in V.checks[n:]:
        print("CHECKROW", PHASE[0], c.get("tokens"), c.get("path"), "bitwise", c.get("bitwise"), "state", c.get("state_bitwise"),
              "nan_ref", c.get("nan"), "nan_v", c.get("nan_v"), "eager2", c.get("eager_twice_bitwise"),
              *(("max_abs", c.get("max_abs"), "state_bad", c.get("state_bad"), "detail", c.get("state_first_bad_detail")) if not c.get("bitwise") else ()),
              flush=True)
    return outs


def add(n, max_tokens):
    rid[0] += 1
    eng.add_request(f"r{rid[0]}", prompt(n), sp(max_tokens))
    return f"r{rid[0]}"


def drain():
    while eng.has_unfinished_requests():
        step()


for r in range(a.rounds):
    ROUND[0] = r
    for L in (64, 512, 2048) if "prefill" in a.kinds else ():
        PHASE[0] = f"prefill{L}"
        add(L, 1)
        drain()
    # requests end by length (no aborts): the fill samples token 1, each decode step one more
    bursts = ((8, 3, 256), (1, 3, 256), (64, 3, 256)) if "decode" in a.kinds else ()
    bursts += ((4, 3, 6000),) if "longdecode" in a.kinds else ()
    for bs, steps, plen in bursts:
        tag = f"{bs}" if plen == 256 else f"{bs}kv{plen}"
        PHASE[0] = f"fill{tag}"
        ids = {add(plen, steps + 1) for _ in range(bs)}
        sampled = set()
        while sampled != ids and eng.has_unfinished_requests():  # the fill (chunked past max_num_batched_tokens)
            sampled |= {o.request_id for o in step() if len(o.outputs[0].token_ids) >= 1}
        PHASE[0] = f"decode{tag}"
        drain()
    if "mixed" not in a.kinds:
        continue
    PHASE[0] = "mixed_fill"
    for _ in range(8):
        add(256, 3)
    step()
    PHASE[0] = "mixed_decode8"
    step()
    PHASE[0] = "mixed8x512"
    add(512, 1)
    step()
    PHASE[0] = "drain"
    drain()

res = V.summary()
rows = []
index = collections.Counter()
for c, ph in zip(V.checks, phase_of_check + ["?"] * (len(V.checks) - len(phase_of_check))):
    key = (ph, c.get("tokens"))
    rows.append({"phase": ph, "tokens": c.get("tokens"), "path": c.get("path"), "index": index[key], "bitwise": c.get("bitwise"),
                 "nan": c.get("nan"), "nan_v": c.get("nan_v"), "eager_twice_bitwise": c.get("eager_twice_bitwise"),
                 "sampled_equal": c.get("sampled_equal"), "max_abs": c.get("max_abs")})
    index[key] += 1
table = collections.defaultdict(lambda: {"calls": 0, "replays": 0, "bitwise": 0, "nan_ref": 0, "first_bad_index": None, "paths": collections.Counter()})
for row in rows:
    t = table[f"{row['phase']}/{row['tokens']}"]
    t["calls"] += 1
    t["replays"] += row["path"] == "replay"
    t["bitwise"] += bool(row["bitwise"])
    t["nan_ref"] += bool(row["nan"])
    t["paths"][row["path"]] += 1
    if not row["bitwise"] and t["first_bad_index"] is None:
        t["first_bad_index"] = row["index"]
out = {"arm": a.arm, "init_s": init_s, "rounds": a.rounds, "V": res, "rows": rows,
       "by_shape": {k: {**v, "paths": dict(v["paths"])} for k, v in table.items()}}
with open(a.out, "w") as f:
    json.dump(out, f, indent=1, default=str)
for k, v in sorted(out["by_shape"].items()):
    flags = "".join(("R" if r["path"] == "replay" else (r["path"] or "?")[0].upper()) + ("" if r["bitwise"] else "!") for r in rows if f"{r['phase']}/{r['tokens']}" == k)
    print("SHAPE", k, {kk: vv for kk, vv in v.items()}, "by index:", flags, flush=True)
print("TOTALS", {k: res.get(k) for k in ("traces", "replays", "eager", "variants", "learned_by_op", "unsound_learned", "sections_added", "no_sections_refused", "no_lib_redo_refused")},
      "checks", len(rows), "bitwise", sum(bool(r["bitwise"]) for r in rows), "nan_ref", sum(bool(r["nan"]) for r in rows), flush=True)
os._exit(0)
