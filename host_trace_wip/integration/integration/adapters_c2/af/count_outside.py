# python3 count_outside.py <chrome trace>...: GPU ops of one profiled decode step (the profile wraps the whole mr.decode
# call) launched outside CUDA graphs, split into before / after the step's first graph launch (by host launch time),
# with their names; graph launches and in-graph kernels counted too. Ops = kernels, memcpys, memsets.
import gzip
import json
import sys

for path in sys.argv[1:]:
    with gzip.open(path) if path.endswith(".gz") else open(path) as f:
        ev = json.load(f)["traceEvents"]
    rt = {e["args"].get("correlation"): e for e in ev if e.get("cat") in ("cuda_runtime", "cuda_driver") and "args" in e}
    ops = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    launches = sorted(r["ts"] for r in rt.values() if "GraphLaunch" in r["name"])
    first = launches[0] if launches else float("inf")
    before, after, in_graph = [], [], 0
    for k in sorted(ops, key=lambda e: e["ts"]):
        r = rt.get(k["args"].get("correlation"))
        if r is not None and "GraphLaunch" in r["name"]:
            in_graph += 1
            continue
        t = r["ts"] if r is not None else k["ts"]
        (before if t < first else after).append(f'{k["cat"]}:{k["name"][:70]}')
    print(json.dumps({"trace": path, "graph_launches": len(launches), "in_graph": in_graph, "outside": len(before) + len(after),
                      "before": len(before), "after": len(after), "before_names": before, "after_names": after}))
