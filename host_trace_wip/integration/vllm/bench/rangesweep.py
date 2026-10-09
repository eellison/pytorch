# Phase 6 step 4c: how many traces a range warm-up needs under arm V (python_vllm_cpp.sh). Drives decode bs over
# --dec-bs, single- and multi-request prefills over --pre-lens, and mixed steps (nd decoding requests plus one new
# prompt, chunked to the 8192-token budget) over --mixed-nd x --mixed-lens. Every model forward is timed (synced) with
# the trace / variant / harvest / eager deltas and filed under its exact key; at the end, per family entry: traces,
# variants, and the guards (with their symbols' sources) that tell its variants apart.
#   CUDA_VISIBLE_DEVICES=2 bash python_vllm_cpp.sh bench/rangesweep.py --out bench/out/range/X.json
import argparse, collections, json, os, sys, time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
import torch
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
p.add_argument("--model", default="/data/eellison/models/Qwen3-8B")
p.add_argument("--dec-bs", type=int, nargs="*", default=list(range(1, 257)))
p.add_argument("--pre-lens", type=int, nargs="*", default=[1, 2, 3, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 127, 128, 129, 255, 256, 257,
                                                           511, 512, 513, 1000, 1023, 1024, 1025, 2047, 2048, 2049, 3000, 4095, 4096, 4097, 6000, 8191, 8192])
p.add_argument("--pre-multi", type=str, nargs="*", default=["2x8", "2x100", "3x500", "8x64", "16x300", "64x100", "128x64", "2x4096"])
p.add_argument("--mixed-nd", type=int, nargs="*", default=[1, 2, 3, 8, 16, 17, 32, 33, 64, 128, 255])
p.add_argument("--mixed-lens", type=int, nargs="*", default=[1, 2, 7, 16, 17, 40, 128, 500, 1000, 4000, 8192])
p.add_argument("--check", action="store_true", help="compare every forward against eager (V.check), bitwise")
p.add_argument("--diag", action="store_true", help="print the guards that tell each entry's variants apart")
a = p.parse_args()
sys.path.insert(0, os.environ.get("INTEG_ARMV", "/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/vllm/armV"))
import adapter
import adapter_cpp

llm = LLM(model=a.model, tensor_parallel_size=1, enforce_eager=True, attention_backend="FLASHINFER", enable_prefix_caching=False,
          gpu_memory_utilization=float(os.environ.get("INTEG_GPU_UTIL", "0.6")), **({"kv_cache_memory_bytes": int(os.environ["INTEG_KV_BYTES"])} if os.environ.get("INTEG_KV_BYTES") else {}), max_model_len=int(os.environ.get("INTEG_MAX_MODEL_LEN", "16384")), max_num_batched_tokens=8192, max_num_seqs=256, seed=0)
runner = llm.llm_engine.model_executor.driver_worker.worker.model_runner
V = (adapter_cpp.ArmVcpp if adapter_cpp.BOUND else adapter.ArmV)(runner)
V.install()
V.check = a.check
print("arm", type(V).__name__, "family", adapter.FAMILY, "split", adapter.BS_SPLIT, flush=True)
PHASE = ["decode"]
CALLS = []
fwd = V.forward


def counts():
    es = list(V.entries.values())
    return (sum(e.traces for e in es), sum(len(e.variants) for e in es), V.provider.harvests, sum(e.eager for e in es), sum(e.replays for e in es),
            sum(getattr(e, "relowers", 0) for e in es))


def forward(input_ids, positions, *args, **kw):
    V.last_exact = None
    c0 = counts()
    n_eager0 = sum(v for k, v in V.calls.items() if str(k).startswith("eager"))
    torch.cuda.synchronize()
    t = time.perf_counter()
    out = fwd(input_ids, positions, *args, **kw)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t) * 1e3
    d = [y - x for x, y in zip(c0, counts())]
    ex = V.last_exact
    exs = f"{ex[0]}/T{ex[1]}/{ex[2]}" if ex else f"bound/T{input_ids.shape[0]}"
    fam = str(V._entry_key(ex[0], ex[1], ex[2], ex[4]))[:40] if ex else "bound"
    ad_eager = sum(v for k, v in V.calls.items() if str(k).startswith("eager")) - n_eager0
    CALLS.append((PHASE[0], exs, fam, round(dt, 2), *d, ad_eager))
    return out


runner.model.forward = forward
g = torch.Generator().manual_seed(0)


def prompt(length):
    return TokensPrompt(prompt_token_ids=torch.randint(1000, 100000, (length,), generator=g).tolist())


sp = lambda n: SamplingParams(max_tokens=n, ignore_eos=True, temperature=0.0)
eng = llm.llm_engine
t_all = time.perf_counter()
for bs in a.dec_bs:
    llm.generate([prompt(16) for _ in range(bs)], sp(3), use_tqdm=False)
t_dec = time.perf_counter()
PHASE[0] = "prefill"
for L in a.pre_lens:
    llm.generate([prompt(L)], sp(1), use_tqdm=False)
for spec in a.pre_multi:
    n, L = map(int, spec.split("x"))
    llm.generate([prompt(L) for _ in range(n)], sp(1), use_tqdm=False)
t_pre = time.perf_counter()
PHASE[0] = "mixed"
for nd in a.mixed_nd:
    for i in range(nd):
        eng.add_request(f"d{nd}_{i}", prompt(8), sp(100000))
    eng.step()
    while not (CALLS[-1][1].startswith("decode") or CALLS[-1][1].startswith("bound")):
        eng.step()
    for j, L in enumerate(a.mixed_lens):
        eng.add_request(f"p{nd}_{j}", prompt(L), sp(1))
        while True:
            eng.step()
            if not any(r == f"p{nd}_{j}" for r in eng.output_processor.request_states):
                break
    eng.abort_request([f"d{nd}_{i}" for i in range(nd)])
    while eng.has_unfinished_requests():
        eng.step()
t_mix = time.perf_counter()

per = collections.defaultdict(lambda: collections.Counter())
first = {}
for ph, ex, fam, ms, dtr, dv, dh, de, dr, drl, ade in CALLS:
    c = per[ph]
    c["calls"] += 1
    c["traces"] += dtr
    c["relowers"] += drl
    c["learns"] += dh
    c["steps_with_learns"] += dh > 0
    c["eager_entry"] += de
    c["eager_adapter"] += ade
    if ex.startswith("bound"):
        continue
    if ex not in first:
        first[ex] = (ph, fam, ms, dtr, dh, de, ade)
        c["new_keys"] += 1
        c["new_keys_traced"] += dtr > 0
        c["new_keys_eager"] += (de + ade) > 0
summary = {}
for ph in ("decode", "prefill", "mixed"):
    ms = sorted(f[2] for f in first.values() if f[0] == ph and f[3] == 0 and f[5] == 0 and f[6] == 0)
    summary[ph] = dict(per[ph], first_ms_med=ms[len(ms) // 2] if ms else None, first_ms_p90=ms[int(len(ms) * 0.9)] if ms else None,
                       first_ms_max=ms[-1] if ms else None)
ents = {str(k)[:60]: {"traces": e.traces, "variants": len(e.variants), "relowers": getattr(e, "relowers", 0), "eager": e.eager} for k, e in V.entries.items()}
res = {"arm": type(V).__name__, "family": adapter.FAMILY, "split": adapter.BS_SPLIT, "secs": {"decode": t_dec - t_all, "prefill": t_pre - t_dec, "mixed": t_mix - t_pre},
       "summary": summary, "entries": ents, "bad": [str(k) for k in V.bad], "adapter_calls": dict(V.calls), "first": first, "calls": CALLS,
       "checks": {"n": len(V.checks), "bitwise": sum(c["bitwise"] for c in V.checks), "bad": [c for c in V.checks if not c["bitwise"]][:20]},
       "mem": {"reserved_mib": torch.cuda.memory_reserved() >> 20, "allocated_mib": torch.cuda.memory_allocated() >> 20}}
# census over every variant: the ops left opaque (each a learn per new key), eager calls and boundaries in the lowering
VS = V.summary()
census = {"learned_by_op": VS["learned_by_op"], "traced_impls": VS["traced_impls"], "trtllm_fork": VS["trtllm_fork"], "opaque_ops": collections.Counter(),
          "eager_ops": collections.Counter(), "boundaries": collections.Counter(), "variants": 0, "refused": VS["refused"], "decline_sites": VS["decline_sites"]}
for e in VS["entries"].values():
    for pv in e["per_variant"]:
        census["variants"] += 1
        census["opaque_ops"].update(pv["opaque_ops"])
        census["eager_ops"].update(pv["eager_ops"])
        census["boundaries"][pv["boundaries"]] += 1
res["census"] = census
res["retrace"] = {k: {"retrace_causes": e.get("retrace_causes"), "respecs": e.get("respecs"), "redispatches": e.get("redispatches"), "traces": e.get("traces")}
                  for k, e in VS["entries"].items()}
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
with open(a.out, "w") as f:
    json.dump(res, f, indent=1, default=str)
print("secs", {k: round(v, 1) for k, v in res["secs"].items()}, "mem", res["mem"], flush=True)
for ph, s in summary.items():
    print(ph, dict(s), flush=True)
for k, v in ents.items():
    print("entry", k, v, flush=True)
print("checks", res["checks"]["n"], "bitwise", res["checks"]["bitwise"], res["checks"]["bad"][:5], flush=True)
print("bad", len(V.bad), [str(k)[:80] for k in list(V.bad)[:10]], flush=True)
print("adapter calls", dict(V.calls), flush=True)
print("CENSUS learned_by_op", census["learned_by_op"], flush=True)
print("CENSUS opaque_ops (summed over variants)", dict(census["opaque_ops"]), flush=True)
print("CENSUS eager_ops", dict(census["eager_ops"]), "boundaries per variant", dict(census["boundaries"]), "variants", census["variants"], flush=True)
print("CENSUS refused", census["refused"][:10], "decline_sites", census["decline_sites"][:10], flush=True)
for ex, f in first.items():
    if f[3] or f[5] or f[6] or f[2] > 150:
        print("slow-first", f[0], ex[:80], f[1:], flush=True)
if a.diag:
    for k, e in V.entries.items():
        vs = e.variants
        if len(vs) < 2:
            continue
        print("=== entry", str(k)[:80], "variants", len(vs), flush=True)
        sets = [[str(x.expr) for x in v.tape.shape_env.guards] for v in vs]
        common = set.intersection(*map(set, sets))
        for i, (v, gs) in enumerate(zip(vs, sets)):
            own = [x for x in v.tape.shape_env.guards if str(x.expr) not in common]
            env = v.tape.shape_env
            syms = sorted({str(s) for x in own for s in x.expr.free_symbols})
            print(f"  v{i} learns {v.learns} guards {len(gs)} own {len(own)}", flush=True)
            for x in own[:12]:
                print("     ", str(x.expr)[:160], flush=True)
            print("      syms", [(s, env.names.get(s), getattr(env.ctx.symbols.get(s), "hint", None)) for s in syms][:12], flush=True)
