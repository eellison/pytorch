# prof_step.py DIR...: per profiled decode step (drive_fn --profile-decode chrome traces <arm>_bs<N>_<k>.json): kernels launched,
# GPU busy (union of kernel/memcpy/memset intervals), wall (first CPU event -> last GPU end), host time from the step start to the
# first launch API call and to the first GPU activity, graph launches vs direct kernel launches, memcpys; median per (arm, bs).
import collections, glob, json, os, re, statistics, sys

LAUNCH = re.compile(r"cudaLaunchKernel|cuLaunchKernel|cudaLaunchCooperativeKernel|cuLaunchCooperativeKernel")
rows = collections.defaultdict(list)
for d in sys.argv[1:]:
    for f in sorted(glob.glob(os.path.join(d, "*_bs*_*.json"))):
        m = re.match(r"(\w+?)_bs(\d+)_(\d+)\.json", os.path.basename(f))
        arm, bs = m.group(1), int(m.group(2))
        ev = [e for e in json.load(open(f)).get("traceEvents", []) if e.get("ph") == "X"]
        gpu = [e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
        cpu = [e for e in ev if e.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset")]
        rt = [e for e in cpu if e.get("cat") in ("cuda_runtime", "cuda_driver")]
        if not gpu or not cpu:
            continue
        t0 = min(e["ts"] for e in cpu)
        tend = max(e["ts"] + e["dur"] for e in gpu + cpu)
        iv = sorted((e["ts"], e["ts"] + e["dur"]) for e in gpu)
        busy, cur = 0.0, None
        for a, b in iv:
            if cur is None or a > cur[1]:
                if cur:
                    busy += cur[1] - cur[0]
                cur = [a, b]
            else:
                cur[1] = max(cur[1], b)
        busy += cur[1] - cur[0]
        graphs = [e for e in rt if "GraphLaunch" in e["name"]]
        direct = [e for e in rt if LAUNCH.search(e["name"])]
        first_api = min((e["ts"] for e in graphs + direct), default=None)
        rows[(arm, bs)].append({
            "kernels": sum(e.get("cat") == "kernel" for e in gpu), "memcpy": sum(e.get("cat") == "gpu_memcpy" for e in gpu),
            "memset": sum(e.get("cat") == "gpu_memset" for e in gpu), "graph_launches": len(graphs), "direct_launches": len(direct),
            "wall_us": tend - t0, "gpu_busy_us": busy, "host_to_first_launch_us": (first_api - t0) if first_api else None,
            "host_to_first_gpu_us": iv[0][0] - t0,
            "direct_kernels_us": sum(e["dur"] for e in direct)})
med = lambda xs: statistics.median(xs) if xs else None
print(f"{'arm':8s} {'bs':>3s} {'n':>2s} {'kernels':>8s} {'memcpy':>6s} {'memset':>6s} {'graphL':>6s} {'directL':>7s} {'wall_ms':>8s} {'gpu_busy_ms':>11s} {'host->1st launch ms':>19s} {'host->1st GPU ms':>16s}")
for (arm, bs), rs in sorted(rows.items()):
    g = lambda k: med([r[k] for r in rs if r[k] is not None])
    f = lambda v, s=1e-3: f"{v * s:.2f}" if v is not None else "-"
    print(f"{arm:8s} {bs:3d} {len(rs):2d} {g('kernels'):8.0f} {g('memcpy'):6.0f} {g('memset'):6.0f} {g('graph_launches'):6.0f} {g('direct_launches'):7.0f} "
          f"{f(g('wall_us')):>8s} {f(g('gpu_busy_us')):>11s} {f(g('host_to_first_launch_us')):>19s} {f(g('host_to_first_gpu_us')):>16s}")
