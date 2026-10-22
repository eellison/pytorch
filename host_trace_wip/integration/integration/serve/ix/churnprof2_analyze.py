# churnprof2 (main 10-14): truly unsynced churn C = 64, vcsl vs default. (1) where the host blocks inside a step (long CUDA API calls,
# relative to the previous step's GPU end); (2) our launch vs the wait; (3) GPU idle gap between consecutive steps, by class; (4) vcsl:
# per-call wall of cuGraphExecKernelNodeSetParams by what changed (function / grid / args / no-op) and whether the stream was busy.
#   python3 serve/ix/churnprof2_analyze.py
import collections, json, os, re, statistics as st

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SYNCY = ("Synchronize", "cudaMemcpy ", "cudaStreamWaitEvent", "cudaEventQuery")


def steps(path):
    ev = json.load(open(path))["traceEvents"]
    X = [e for e in ev if e.get("ph") == "X"]
    ex0 = sorted([e for e in X if e["name"] == "PH_execute_model"], key=lambda e: e["ts"])
    ex = [e for e in ex0 if not any(o is not e and o["ts"] <= e["ts"] and e["ts"] + e["dur"] <= o["ts"] + o["dur"] for o in ex0)]
    lab = sorted([e for e in X if e["name"].startswith("STEP_")], key=lambda e: e["ts"])
    rtl = sorted([e for e in X if e.get("cat") in ("cuda_runtime", "cuda_driver")], key=lambda e: e["ts"])
    rt = {e.get("args", {}).get("correlation"): e for e in rtl if e.get("args", {}).get("correlation") is not None}
    gpu = sorted([e for e in X if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")], key=lambda e: e["ts"])
    out = []
    for i, e in enumerate(ex):
        t0, t1 = e["ts"], (ex[i + 1]["ts"] if i + 1 < len(ex) else e["ts"] + e["dur"])
        lb = next((l["name"] for l in lab if t0 <= l["ts"] <= t0 + e["dur"]), None)
        size = tuple(map(int, re.match(r"STEP_(\d+)x(\d+)", lb).groups())) if lb else None
        mine = [g for g in gpu if (s := rt.get(g.get("args", {}).get("correlation"))) is not None and t0 <= s["ts"] < t1]
        launch = next((r for r in rtl if t0 <= r["ts"] <= t0 + e["dur"] and "GraphLaunch" in r["name"]), None)
        waits = [(r["name"], round(r["ts"] - t0), round(r["dur"])) for r in rtl if t0 <= r["ts"] <= t1 and r["dur"] > 100 and "GraphLaunch" not in r["name"]]
        out.append({"size": size, "t0": t0, "exec": e["dur"], "launch_at": (launch["ts"] - t0) if launch else None, "launch_dur": launch["dur"] if launch else None,
                    "gpu_first": min((g["ts"] for g in mine), default=None), "gpu_last": max((g["ts"] + g["dur"] for g in mine), default=None), "waits": waits})
    prev = None
    for i, s in enumerate(out):
        s["cls"] = "mixed" if s["size"] and s["size"][1] > s["size"][0] else ("decode_changed" if prev and prev["size"] != s["size"] else "decode_same")
        s["prev_gpu_end_rel"] = (prev["gpu_last"] - s["t0"]) if prev and prev["gpu_last"] else None  # > 0: the previous step's GPU work still running at this step's start
        s["gap"] = (s["gpu_first"] - prev["gpu_last"]) if prev and prev["gpu_last"] and s["gpu_first"] else None
        prev = s
    return out


for arm in ("vcsl", "default"):
    p = f"{HERE}/serve/out/prof_churn2/{arm}_cb64_200_260.json"
    if not os.path.exists(p):
        continue
    S = steps(p)
    print(f"\n### {arm}: unsynced churn C = 64, profiled steps 200-260 (us)\n")
    print("| class | steps | interval | execute_model | previous step's GPU still running at start for | graph launch at | GPU idle gap before the step's first kernel |")
    print("|---|---|---|---|---|---|---|")
    for k in ("decode_same", "decode_changed", "mixed"):
        xs = [x for x in S[1:-1] if x["cls"] == k]
        if not xs:
            continue
        m = lambda f: st.median([x[f] for x in xs if x[f] is not None]) if any(x[f] is not None for x in xs) else float("nan")
        iv = st.median([S[S.index(x) + 1]["t0"] - x["t0"] for x in xs])
        print(f"| {k} | {len(xs)} | {iv:.0f} | {m('exec'):.0f} | {m('prev_gpu_end_rel'):.0f} | {m('launch_at'):.0f} | {m('gap'):.0f} |")
    print("\nlong CUDA API calls (> 100 us, not the graph launch) per class, first steps: name, offset into execute_model, us")
    for k in ("decode_same", "decode_changed", "mixed"):
        xs = [x for x in S[1:-1] if x["cls"] == k][:3]
        for x in xs:
            print(f"  {k} {x['size']}: launch at {x['launch_at']}, prev GPU end {x['prev_gpu_end_rel']}: {x['waits'][:6]}")
R = json.load(open(f"{HERE}/vllm/out/t_churnprof2_vcsl_r1.json")) if os.path.exists(f"{HERE}/vllm/out/t_churnprof2_vcsl_r1.json") else None
if R:
    rows = R["cb"]["64"].get("ns_ph", [])
    busy = {r[7]: (r[8][1] if r[8] else None, (r[9] or [None, None])[1]) for r in rows}
    cls, prev = {}, None
    for r in rows:
        cls[r[7]] = "mixed" if r[1] > r[0] else ("decode_changed" if prev and (prev[0], prev[1]) != (r[0], r[1]) else "decode_same")
        prev = r
    E = R.get("setlog", {}).get("cb64", {}).get("entries", [])
    print(f"\n### vcsl: cuGraphExecKernelNodeSetParams wall per call, steps 300-360 ({len(E)} calls; no profiler)\n")
    print("| step class | stream busy at our forward | change | calls | median us | p90 us | max us |")
    print("|---|---|---|---|---|---|---|")
    g = collections.defaultdict(list)
    for e in E:
        kind = "function" if e["func_changed"] else ("grid/block/smem" if e["cfg_changed"] else ("args" if e["bytes_changed"] else "no-op"))
        b = busy.get(e["step"], (None, None))[1]
        g[(cls.get(e["step"], "?"), b, kind)].append(e["dur_ns"] / 1e3)
    for k in sorted(g, key=str):
        v = sorted(g[k])
        print(f"| {k[0]} | {k[1]} | {k[2]} | {len(v)} | {st.median(v):.1f} | {v[int(0.9 * (len(v) - 1))]:.1f} | {v[-1]:.1f} |")
    print("\n### vcsl phase clocks per class (unsynced)\n")
    for k in ("decode_same", "decode_changed", "mixed"):
        rs = [r for r in rows if cls[r[7]] == k]
        if rs:
            m = lambda f: st.median(f(r) for r in rs)
            print(f"{k}: n {len(rs)}, busy at entry {sum(1 for r in rs if r[8] and r[8][1])}/{len(rs)}, busy at forward {sum(1 for r in rs if r[9] and r[9][1])}/{len(rs)}, "
                  f"host ms {m(lambda r: r[6] * 1e3):.2f}, dispatch {m(lambda r: r[2].get('ph_us', {}).get('dispatch', 0)):.0f}, evaluate {m(lambda r: r[2].get('ph_us', {}).get('evaluate', 0)):.0f}, "
                  f"patch {m(lambda r: r[2].get('ph_us', {}).get('patch', 0)):.0f}, launch {m(lambda r: r[2].get('ph_us', {}).get('launch', 0)):.0f}, sets {m(lambda r: r[3]):.0f}")
