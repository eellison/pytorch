# prof_outside.py DIR...: per (arm, bs), the GPU work NOT launched by a graph launch (kernels / memcpy / memset whose launching
# runtime call is not cudaGraphLaunch), by name: count per step and total time per step (median over the profiled steps)
import collections, glob, json, os, re, statistics, sys

for d in sys.argv[1:]:
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    for f in sorted(glob.glob(os.path.join(d, "*_bs*_*.json"))):
        m = re.match(r"(\w+?)_bs(\d+)_(\d+)\.json", os.path.basename(f))
        arm, bs = m.group(1), int(m.group(2))
        ev = [e for e in json.load(open(f)).get("traceEvents", []) if e.get("ph") == "X"]
        corr_api = {e["args"].get("correlation"): e["name"] for e in ev if e.get("cat") in ("cuda_runtime", "cuda_driver") and "args" in e}
        step = collections.defaultdict(lambda: [0, 0.0])
        for e in ev:
            if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset"):
                api = corr_api.get(e.get("args", {}).get("correlation"), "?")
                if "GraphLaunch" in api:
                    continue
                name = (e["name"][:90] + f"  [{api}]")
                step[name][0] += 1
                step[name][1] += e["dur"]
        for name, (n, t) in step.items():
            per[(arm, bs)][name].append((n, t))
    for (arm, bs), names in sorted(per.items()):
        print(f"== {arm} bs {bs}: outside the graph")
        for name, xs in sorted(names.items(), key=lambda kv: -statistics.median(t for _, t in kv[1])):
            print(f"   {statistics.median(n for n, _ in xs):5.0f} x  {statistics.median(t for _, t in xs):8.1f} us  {name}")
