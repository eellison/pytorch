# The last K kernels of the middle profiled step, in GPU order (in-graph or not), with durations and gaps:
# python prof_tail.py trace.json [K]
import json, sys
ev = json.load(open(sys.argv[1]))["traceEvents"]
K = int(sys.argv[2]) if len(sys.argv) > 2 else 20
step = next(e for e in ev if e.get("name") == "ENGINE_STEP_1" and e.get("ph") == "X" and e.get("cat") in ("user_annotation", "cpu_op"))
rt = {e["args"]["correlation"]: e for e in ev if e.get("cat") in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {})}
gpu = sorted((e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and e.get("ph") == "X"
              and (r := rt.get(e.get("args", {}).get("correlation"))) is not None and step["ts"] <= r["ts"] <= step["ts"] + step["dur"]), key=lambda e: e["ts"])
tail = gpu[-K:]
prev = None
tot = 0.0
for e in tail:
    gap = e["ts"] - (prev["ts"] + prev["dur"]) if prev else 0
    g = "graph" if "GraphLaunch" in rt[e["args"]["correlation"]]["name"] else "eager"
    print(f"{e['dur']:8.1f} us  gap {gap:7.1f}  {g:5s} {e['name'][:90]}")
    tot += e["dur"]
    prev = e
print(f"sum {tot:.1f} us over {len(tail)}; span {tail[-1]['ts'] + tail[-1]['dur'] - tail[0]['ts']:.1f} us")
