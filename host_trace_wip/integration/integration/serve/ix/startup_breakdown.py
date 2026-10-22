# Engine startup breakdown per arm from the run logs (vLLM's own INFO lines) and the run JSONs (init_s, our warm-up / bindings load):
# weight load, model load, torch.compile (and whether the graph came from the cache), profiling run, each "Graph capturing finished",
# KV cache size, FlashInfer autotune (configs loaded / saved, seconds), engine init total, our bindings load and untimed warm-up.
#   python3 serve/ix/startup_breakdown.py SET [arm ...]       (vllm/logs/t_<SET>_<arm>_r*.log + vllm/out/t_<SET>_<arm>_r*.json)
import glob, json, os, re, statistics as st, sys
from datetime import datetime

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SET = sys.argv[1]
ARMS = sys.argv[2:] or ["default", "fullnc", "bnc", "vcsl", "vrs", "vcs"]
PAT = {"weights_s": r"Loading weights took ([\d.]+) seconds", "model_load_s": r"Model loading took [\d.]+ GiB memory and ([\d.]+) seconds",
       "compile_s": r"torch.compile took ([\d.]+) s in total", "profile_run_s": r"Initial profiling/warmup run took ([\d.]+) s",
       "kv_tokens": r"GPU KV cache size: ([\d,]+) tokens", "init_engine_s": r"init engine \(profile, create kv cache, warmup model\) took ([\d.]+) s"}


def parse(log):
    t = open(log, errors="replace").read()
    d = {}
    for k, p in PAT.items():
        m = re.findall(p, t)
        if m:
            d[k] = float(m[-1].replace(",", ""))
    d["graph_captures"] = [(int(a), float(b)) for a, b in re.findall(r"Graph capturing finished in (\d+) secs, took ([\d.]+) GiB", t)]
    d["compile_cache_hit"] = bool(re.search(r"Directly load AOT compilation|Directly load the compiled graph|from the cache, took", t))
    ld = re.findall(r"Autotuner\]: Loaded (\d+) configs", t)
    sv = re.findall(r"Autotuner\]: Saved (\d+) configs", t)
    d["fi_autotune_loaded"], d["fi_autotune_saved"] = (int(ld[-1]) if ld else None), (int(sv[-1]) if sv else None)
    ts = re.findall(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ - INFO - autotuner.py:\d+ - flashinfer.jit: \[Autotuner\]: Autotuning process (starts|ends)", t)
    if len(ts) >= 2:
        s0 = next((datetime.fromisoformat(x) for x, w in ts if w == "starts"), None)
        e0 = next((datetime.fromisoformat(x) for x, w in reversed(ts) if w == "ends"), None)
        d["fi_autotune_s"] = (e0 - s0).total_seconds() if s0 and e0 else None
    return d


rows = []
for a in ARMS:
    for log in sorted(glob.glob(f"{HERE}/vllm/logs/t_{SET}_{a}_r*.log")):
        js = log.replace("/logs/", "/out/").replace(".log", ".json")
        d = parse(log)
        if os.path.exists(js):
            R = json.load(open(js))
            d["init_s"] = R.get("init_s")
            d["bindings_load_s"] = R.get("V", {}).get("bindings_load_s")
            d["warmup_s"] = R.get("cb_warmup_s")
            d["capture_model_s"] = R.get("capture", {}).get("capture_model_s")
            d["graph_gib"] = (R.get("capture", {}).get("graph_bytes") or 0) / 2**30
            d["cuda_graphs"] = R.get("capture", {}).get("capture_begins")
        rows.append((a, os.path.basename(log)[:-4], d))
f = lambda x, n=1: (f"{x:.{n}f}" if isinstance(x, (int, float)) else str(x)) if x is not None else "-"
print(f"| arm | run | engine ready s (LLM()) | weights s | model load s | torch.compile s (cache hit) | profiling run s | graph captures (s, GiB) | CUDA graphs | KV tokens | FlashInfer autotune (loaded/saved, s) | init engine s | our bindings load s | untimed warm-up s |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for a, run, d in rows:
    print(f"| {a} | {run} | {f(d.get('init_s'))} | {f(d.get('weights_s'), 2)} | {f(d.get('model_load_s'), 2)} | {f(d.get('compile_s'), 2)} ({'hit' if d.get('compile_cache_hit') else 'miss/none'}) | "
          f"{f(d.get('profile_run_s'), 2)} | {', '.join(f'{s}s/{g}GiB' for s, g in d.get('graph_captures', [])) or '-'} | {f(d.get('cuda_graphs'), 0)} | {f(d.get('kv_tokens'), 0)} | "
          f"{d.get('fi_autotune_loaded')}/{d.get('fi_autotune_saved')}, {f(d.get('fi_autotune_s'), 2)} | {f(d.get('init_engine_s'), 2)} | {f(d.get('bindings_load_s'), 2)} | {f(d.get('warmup_s'))} |")
