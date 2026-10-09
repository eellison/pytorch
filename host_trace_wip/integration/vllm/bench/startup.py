# Cold/warm start to ready-to-serve for vLLM arms (eager, default, V), in-process engine, same engine config as drive.py.
# Seconds from process start (/proc/self/stat) to: imports done, LLM() returned (model load; default: compile + graph
# capture, split from vLLM's own log lines), then arm V's warm-up (one forward per structural class: the shapes behind
# the range sweep's traces), = ready. After ready, the first request of each probe shape class (shapes no arm warmed
# exactly) is timed with its trace / harvest deltas. --save writes V's bindings after the probes (HarvestProvider.save);
# --load reads them before the warm-up (the "warm" start).
#   python startup.py --arm V --out X.json [--save B.pkl | --load B.pkl]
import os, time

T_PROC = None
with open("/proc/self/stat") as f, open("/proc/uptime") as g:
    _ticks = int(f.read().rsplit(")", 1)[1].split()[19])
    T_PROC = time.time() - (float(g.read().split()[0]) - _ticks / os.sysconf("SC_CLK_TCK"))
import argparse, json, logging, re, sys

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
p = argparse.ArgumentParser()
p.add_argument("--arm", choices=["eager", "default", "V"], required=True)
p.add_argument("--out", required=True)
p.add_argument("--save", default=None)
p.add_argument("--load", default=None)
p.add_argument("--model", default="/data/eellison/models/Qwen3-8B")
a = p.parse_args()
import torch
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

R = {"arm": a.arm, "util": float(os.environ.get("INTEG_GPU_UTIL", "0.6")), "stamps": {}, "vllm_log": {}, "probes": {}, "warmup": {}}
stamp = lambda k: R["stamps"].__setitem__(k, round(time.time() - T_PROC, 3))
stamp("imports")
LOGS = []


class _Grab(logging.Handler):
    def emit(self, rec):
        LOGS.append((time.time() - T_PROC, rec.getMessage()))


logging.getLogger("vllm").addHandler(_Grab())
llm = LLM(model=a.model, tensor_parallel_size=1, enforce_eager=a.arm != "default", attention_backend="FLASHINFER", enable_prefix_caching=False,
          gpu_memory_utilization=R["util"], **({"kv_cache_memory_bytes": int(os.environ["INTEG_KV_BYTES"])} if os.environ.get("INTEG_KV_BYTES") else {}), max_model_len=4096, seed=0)
stamp("llm_init")
for t, m in LOGS:
    for key, pat in (("model_load_s", r"Model loading took [\d.]+ GiB memory and ([\d.]+) seconds"), ("compile_s", r"torch.compile took ([\d.]+) s in total"),
                     ("dynamo_s", r"Dynamo bytecode transform time: ([\d.]+) s"), ("init_engine_s", r"init engine .* took ([\d.]+) s"),
                     ("kv_tokens", r"GPU KV cache size: ([\d,]+) tokens")):
        if (mm := re.search(pat, m)):
            R["vllm_log"][key] = float(mm.group(1).replace(",", ""))
    if (mm := re.search(r"Graph capturing finished in ([\d.]+) secs, took ([\d.]+) GiB", m)):
        R["vllm_log"].setdefault("capture", []).append((float(mm.group(1)), float(mm.group(2))))
V = None
runner = llm.llm_engine.model_executor.driver_worker.worker.model_runner
if a.arm == "V":
    sys.path.insert(0, os.environ["INTEG_ARMV"])
    import adapter
    import adapter_cpp

    V = (adapter_cpp.ArmVcpp if adapter_cpp.BOUND else adapter.ArmV)(runner)
    V.install()
    if a.load:
        R["loaded"] = V.provider.load(a.load)
    stamp("installed")
g = torch.Generator().manual_seed(0)
prompt = lambda n: TokensPrompt(prompt_token_ids=torch.randint(1000, 100000, (n,), generator=g).tolist())
sp = lambda n: SamplingParams(max_tokens=n, ignore_eos=True, temperature=0.0)
eng = llm.llm_engine


def counts():
    if V is None:
        return {}
    es = list(V.entries.values())
    return {"traces": sum(e.traces for e in es), "variants": sum(len(e.variants) for e in es), "harvests": V.provider.harvests,
            "restored": getattr(V.provider, "restored", 0), "eager": sum(e.eager for e in es)}


def timed(name, fn, sink):
    c0 = counts()
    torch.cuda.synchronize()
    t = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    c1 = counts()
    sink[name] = {"s": round(time.perf_counter() - t, 4), **{k: c1[k] - c0[k] for k in c1}}


def mixed(nd, pl):
    def run():
        for i in range(nd):
            eng.add_request(f"d{nd}_{pl}_{i}", prompt(8), sp(4))
        eng.step()
        eng.add_request(f"p{nd}_{pl}", prompt(pl), sp(1))
        while eng.has_unfinished_requests():
            eng.step()
    return run


# arm V's warm-up: decode in each bucket (BS_SPLIT 2,32) and both forms, prefill n==1 / n>1 per bucket, mixed nd==1 / nd>1
WARM = [("decode1", lambda: llm.generate([prompt(16)], sp(3), use_tqdm=False)),
        ("decode2", lambda: llm.generate([prompt(16) for _ in range(2)], sp(3), use_tqdm=False)),
        ("decode33", lambda: llm.generate([prompt(16) for _ in range(33)], sp(3), use_tqdm=False)),
        ("prefill1", lambda: llm.generate([prompt(1)], sp(1), use_tqdm=False)),
        ("prefill17", lambda: llm.generate([prompt(17)], sp(1), use_tqdm=False)),
        ("prefill64", lambda: llm.generate([prompt(64)], sp(1), use_tqdm=False)),
        ("prefill2x8", lambda: llm.generate([prompt(8) for _ in range(2)], sp(1), use_tqdm=False)),
        ("prefill8x64", lambda: llm.generate([prompt(64) for _ in range(8)], sp(1), use_tqdm=False)),
        ("mixed1x16", mixed(1, 16)), ("mixed2x500", mixed(2, 500)), ("mixed33x100", mixed(33, 100))]
if a.arm == "V":
    for name, fn in WARM:
        timed(name, fn, R["warmup"])
stamp("ready")
PROBES = [("decode5", lambda: llm.generate([prompt(16) for _ in range(5)], sp(3), use_tqdm=False)),
          ("decode48", lambda: llm.generate([prompt(16) for _ in range(48)], sp(3), use_tqdm=False)),
          ("decode200", lambda: llm.generate([prompt(16) for _ in range(200)], sp(3), use_tqdm=False)),
          ("prefill300", lambda: llm.generate([prompt(300)], sp(1), use_tqdm=False)),
          ("prefill3000", lambda: llm.generate([prompt(3000)], sp(1), use_tqdm=False)),
          ("prefill4x700", lambda: llm.generate([prompt(700) for _ in range(4)], sp(1), use_tqdm=False)),
          ("mixed16x700", mixed(16, 700)), ("mixed64x40", mixed(64, 40))]
for name, fn in PROBES:
    timed(name, fn, R["probes"])
    timed(name + "_again", fn, R["probes"])
stamp("probes_done")
if a.save and V is not None:
    R["saved"] = V.provider.save(a.save)
R["counts"] = counts()
R["mem"] = {"reserved_mib": torch.cuda.memory_reserved() >> 20, "allocated_mib": torch.cuda.memory_allocated() >> 20,
            "max_allocated_mib": torch.cuda.max_memory_allocated() >> 20}
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
json.dump(R, open(a.out, "w"), indent=1)
print("STARTUP", json.dumps({k: R[k] for k in ("arm", "stamps", "vllm_log", "counts", "mem")}), flush=True)
print("WARMUP", json.dumps(R["warmup"]), flush=True)
print("PROBES", json.dumps(R["probes"]), flush=True)
