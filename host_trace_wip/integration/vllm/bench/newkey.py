# Phase 6 step 4b: first-call cost of a new key under arm V (python_vllm_cpp.sh). Each model forward is timed (synced),
# with the trace / variant / harvest / eager counters before and after it, and filed under its exact key (kind, T,
# split, launches) and its entry key. Phases: warm (decode bs 1, 4), a decode bs sweep, a prefill length sweep, mixed
# steps with varied prompt lengths, then a check pass (V.check: every traced forward against eager, bitwise).
#   CUDA_VISIBLE_DEVICES=2 bash python_vllm_cpp.sh bench/newkey.py --out bench/out/newkey/X.json
import argparse, json, os, statistics, sys, time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
import torch
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
p.add_argument("--model", default="/data/eellison/models/Qwen3-8B")
p.add_argument("--warm-bs", type=int, nargs="*", default=[1, 4])
p.add_argument("--bs", type=int, nargs="*", default=[2, 3, 5, 6, 7, 8, 9, 12, 16, 17, 24, 31, 32, 33, 48, 63, 64, 65])
p.add_argument("--prefill-lens", type=int, nargs="*", default=[100, 200, 300, 513, 1000, 1500])
p.add_argument("--mixed", type=int, nargs="*", default=[40, 90, 130, 260, 333])
p.add_argument("--check-bs", type=int, nargs="*", default=[1, 4, 9, 33])
a = p.parse_args()
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "armV"))
import adapter
import adapter_cpp

llm = LLM(model=a.model, tensor_parallel_size=1, enforce_eager=True, attention_backend="FLASHINFER", enable_prefix_caching=False,
          gpu_memory_utilization=0.6, max_model_len=4096, seed=0)
runner = llm.llm_engine.model_executor.driver_worker.worker.model_runner
V = (adapter_cpp.ArmVcpp if adapter_cpp.BOUND else adapter.ArmV)(runner)
V.install()
print("arm", type(V).__name__, "family", adapter.FAMILY, "split", adapter.BS_SPLIT, "lendable", len(getattr(V.provider, "lendable", ())), flush=True)
PHASE = ["warm"]
CALLS = []  # (phase, exact key str, entry key str, ms, d_traces, d_variants, d_harvests, d_eager, d_replays)
fwd = V.forward


def counts():
    es = list(V.entries.values())
    return (sum(e.traces for e in es), sum(len(e.variants) for e in es), V.provider.harvests, sum(e.eager for e in es), sum(e.replays for e in es))


def forward(input_ids, positions, *args, **kw):
    V.last_exact = None
    c0 = counts()
    torch.cuda.synchronize()
    t = time.perf_counter()
    out = fwd(input_ids, positions, *args, **kw)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t) * 1e3
    c1 = counts()
    ex = V.last_exact
    exs = f"{ex[0]}/T{ex[1]}/{ex[2]}" if ex else f"bound/T{input_ids.shape[0]}"
    CALLS.append((PHASE[0], exs, ex[0] if ex else "bound", ex[1] if ex else input_ids.shape[0], round(dt, 2), *[y - x for x, y in zip(c0, c1)]))
    return out


runner.model.forward = forward
g = torch.Generator().manual_seed(0)


def prompts(n, length):
    return [TokensPrompt(prompt_token_ids=torch.randint(1000, 100000, (length,), generator=g).tolist()) for _ in range(n)]


sp = lambda n: SamplingParams(max_tokens=n, ignore_eos=True, temperature=0.0)
t_all = time.perf_counter()
for bs in a.warm_bs:
    llm.generate(prompts(bs, 64), sp(6), use_tqdm=False)
PHASE[0] = "decode_sweep"
for bs in a.bs:
    llm.generate(prompts(bs, 64), sp(4), use_tqdm=False)
PHASE[0] = "prefill_sweep"
for L in a.prefill_lens:
    llm.generate(prompts(1, L), sp(2), use_tqdm=False)
PHASE[0] = "mixed"
eng = llm.llm_engine
for i, pr in enumerate(prompts(8, 64)):
    eng.add_request(f"m{i}", pr, sp(1000))
for _ in range(3):
    eng.step()
for j, L in enumerate(a.mixed):
    eng.add_request(f"n{j}", prompts(1, L)[0], sp(2))
    eng.step()
    eng.step()
for i in range(8):
    eng.abort_request([f"m{i}"])
while eng.has_unfinished_requests():
    eng.step()
sweep_s = time.perf_counter() - t_all
PHASE[0] = "check"
V.check = True
for bs in a.check_bs:
    llm.generate(prompts(bs, 64), sp(3), use_tqdm=False)
V.check = False

first = {}
for ph, ex, kind, T, ms, dtr, dv, dh, de, dr in CALLS:
    if ex.startswith("bound"):
        continue
    first.setdefault(ex, []).append((ph, ms, dtr, dv, dh, de, dr))
rows = []
for ex, cs in first.items():
    ph, ms, dtr, dv, dh, de, dr = cs[0]
    rows.append({"key": ex, "phase": ph, "first_ms": ms, "first_traces": dtr, "first_harvests": dh, "first_eager": de,
                 "next_ms": [c[1] for c in cs[1:4]], "next_traces": sum(c[2] for c in cs[1:4])})
chk = V.checks
res = {"arm": type(V).__name__, "family": adapter.FAMILY, "split": adapter.BS_SPLIT, "sweep_s": sweep_s, "counters": V.counters(),
       "harvests": V.provider.harvests, "entries": len(V.entries), "bad": [str(k) for k in V.bad], "keys": rows,
       "checks": {"n": len(chk), "bitwise": sum(c.get("bitwise", False) for c in chk)}, "calls": CALLS,
       "mem": {"reserved_mib": torch.cuda.memory_reserved() >> 20, "allocated_mib": torch.cuda.memory_allocated() >> 20}}
try:
    res["summary_entries"] = {k: {kk: v[kk] for kk in ("traces", "replays", "eager", "variants", "declines", "reasons")} for k, v in V.summary()["entries"].items()}
except Exception as e:  # noqa: BLE001
    res["summary_error"] = repr(e)
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
with open(a.out, "w") as f:
    json.dump(res, f, indent=1, default=str)
print("sweep_s", round(sweep_s, 1), "counters", res["counters"], "checks", res["checks"], "bad", len(V.bad), flush=True)
for r in rows:
    print(f"{r['phase']:14s} {r['key'][:70]:70s} first {r['first_ms']:9.1f} ms tr {r['first_traces']} h {r['first_harvests']} e {r['first_eager']} | next {r['next_ms']} tr {r['next_traces']}", flush=True)
