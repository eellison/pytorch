# python3 armV/span.py DIR: per Kineto step trace, GPU span first-to-last op (kernel/memcpy/memset), and the span and busy
# time of the ops launched by cudaGraphLaunch (matched by correlation id). Median over the files per (arm, bs).
import collections, glob, json, os, statistics, sys


def one(path):
    ev = json.load(open(path))["traceEvents"]
    gl = {e["args"]["correlation"] for e in ev if e.get("ph") == "X" and e.get("name") == "cudaGraphLaunch"}
    k = sorted((e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")), key=lambda e: e["ts"])
    g = [x for x in k if x["args"].get("correlation") in gl]
    def span(xs):
        busy, end = 0.0, xs[0]["ts"]
        for x in xs:
            busy += max(0.0, x["ts"] + x["dur"] - max(x["ts"], end))
            end = max(end, x["ts"] + x["dur"])
        return end - xs[0]["ts"], busy
    s, b = span(k)
    gs, gb = span(g) if g else (0.0, 0.0)
    return s, b, gs, gb, len(k), len(g)


rows = collections.defaultdict(list)
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.json"))):
    arm, bs, _ = os.path.basename(f)[:-5].rsplit("_", 2)
    rows[(arm, int(bs[2:]))].append(one(f))
for (arm, bs), r in sorted(rows.items(), key=lambda kv: (kv[0][1], kv[0][0])):
    m = [statistics.median(c) for c in zip(*r)]
    print(f"bs{bs:<3} {arm:8} step_span {m[0]:7.0f} us busy {m[1]:6.0f} | graph_span {m[2]:6.0f} busy {m[3]:6.0f} | ops {m[4]:.0f} in_graph {m[5]:.0f}")
