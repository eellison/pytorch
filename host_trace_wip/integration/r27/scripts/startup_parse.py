# startup_parse.py LOG...: engine start-up breakdown from vLLM's own log lines (and our summary in the run's JSON when present):
# weight load, model load, torch.compile (cache hit = "Directly load"), profiling/warm-up run, FlashInfer autotune (start/end stamps),
# CUDA graph capture ("Graph capturing finished in X secs, took Y GiB"), init engine total; ours: bindings load, learn seconds, traces.
import json, os, re, sys
from datetime import datetime

def ts(line):
    m = re.search(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3})", line)
    return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S,%f") if m else None

for f in sys.argv[1:]:
    d = {}
    at0 = at1 = None
    for line in open(f, errors="replace"):
        if m := re.search(r"Loading weights took ([\d.]+) seconds", line): d["weights_s"] = float(m.group(1))
        if m := re.search(r"Model loading took ([\d.]+) GiB memory and ([\d.]+) seconds", line): d["model_load_s"] = float(m.group(2))
        if "Directly load AOT compilation" in line or "Directly load the compiled graph" in line: d["compile_cache"] = "hit"
        if m := re.search(r"torch.compile took ([\d.]+) s in total", line): d["torch_compile_s"] = float(m.group(1))
        if m := re.search(r"Initial profiling/warmup run took ([\d.]+) s", line): d["profile_run_s"] = float(m.group(1))
        if "Autotuning process starts" in line: at0 = ts(line)
        if "Autotuning process ends" in line and at0: at1 = ts(line)
        if m := re.search(r"Loaded (\d+) configs from .*flashinfer_autotune", line): d["fi_autotune_cached_configs"] = int(m.group(1))
        if m := re.search(r"Graph capturing finished in ([\d.]+) secs, took ([\d.]+) GiB", line):
            d["graph_capture_s"], d["graph_GiB"] = float(m.group(1)), float(m.group(2))
        if m := re.search(r"init engine \(profile, create kv cache, warmup model\) took ([\d.]+) s", line): d["init_engine_s"] = float(m.group(1))
        if m := re.search(r"GPU KV cache size: ([\d,]+) tokens", line): d["kv_tokens"] = int(m.group(1).replace(",", ""))
    if at0 and at1:
        d["fi_autotune_s"] = round((at1 - at0).total_seconds(), 2)
    d.setdefault("compile_cache", "miss" if "torch_compile_s" in d else None)
    j = os.path.join(os.path.dirname(f), "..", "..", "out", "timing", os.path.basename(f).replace(".log", ".json"))
    if os.path.exists(j):
        r = json.load(open(j))
        d["ready_s"] = round(r.get("init_s", 0), 1)
        if "capture" in r:
            d["capture_begins"], d["graphs"] = r["capture"].get("capture_begins"), r["capture"].get("graphs")
        V = r.get("V") or {}
        if V:
            d.update({"ours_bindings_load_s": V.get("bindings_load_s"), "ours_learn_s": V.get("learn_s"), "ours_harvests": V.get("harvests"),
                      "ours_traces": V.get("traces"), "ours_trace_s": V.get("trace_time_s"), "ours_graph_warmup_s": V.get("graph_warmup_s")})
        else:
            d["vllm_graph_warmup_s"] = d.get("graph_capture_s")
    print(os.path.basename(f), d)
