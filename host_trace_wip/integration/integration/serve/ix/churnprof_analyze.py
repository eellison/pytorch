# churnprof analysis (main 10-14): unsynced churn C = 64, vrs vs bnc. Per step class (decode same size / decode size changed / mixed):
# ours' phase clocks and node sets (ns_ph), setlog classification of the sets (no-op / args only / grid-block-smem / function), and from
# the timelines: execute_model host time, GPU busy attributed to the step (kernels launched inside its execute_model .. sample_tokens),
# the GPU idle time inside the step's window.   python3 serve/ix/churnprof_analyze.py
import collections, json, os, re, statistics as st

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def cls_of(rows):
    out, prev = [], None
    for r in rows:
        k = "mixed" if r[1] > r[0] else ("decode_changed" if prev is not None and (prev[0], prev[1]) != (r[0], r[1]) else "decode_same")
        out.append(k)
        prev = r
    return out


R = json.load(open(f"{HERE}/vllm/out/t_churnprof_vrs_r1.json"))
rows = R["cb"]["64"]["ns_ph"]
cl = cls_of(rows)
print("### vrs phase clocks per step class (unsynced churn, C = 64; execute_model only)\n")
print("| class | steps | host ms (execute_model) | dispatch us | evaluate us | patch us | launch us | kernel nodes set | memory nodes set |")
print("|---|---|---|---|---|---|---|---|---|")
for k in ("decode_same", "decode_changed", "mixed"):
    rs = [r for r, c in zip(rows, cl) if c == k]
    if not rs:
        continue
    m = lambda f: st.median(f(r) for r in rs)
    print(f"| {k} | {len(rs)} | {m(lambda r: r[6] * 1e3):.2f} | {m(lambda r: r[2].get('ph_us', {}).get('dispatch', 0)):.0f} | {m(lambda r: r[2].get('ph_us', {}).get('evaluate', 0)):.0f} | "
          f"{m(lambda r: r[2].get('ph_us', {}).get('patch', 0)):.0f} | {m(lambda r: r[2].get('ph_us', {}).get('launch', 0)):.0f} | {m(lambda r: r[3]):.0f} | {m(lambda r: r[4]):.0f} |")
step_cls = {r[7]: c for r, c in zip(rows, cl)}
SL = R.get("setlog", {}).get("cb64")
if SL:
    by = collections.defaultdict(list)
    for e in SL["entries"]:
        by[e["step"]].append(e)
    agg = collections.defaultdict(lambda: collections.Counter())
    n = collections.Counter()
    for s_, es in by.items():
        k = step_cls.get(s_, "?")
        n[k] += 1
        for e in es:
            agg[k]["sets"] += 1
            if e["func_changed"]:
                agg[k]["function"] += 1
            elif e["cfg_changed"]:
                agg[k]["grid/block/smem"] += 1
            elif e["bytes_changed"]:
                agg[k]["args only"] += 1
            else:
                agg[k]["no-op (identical)"] += 1
    print("\n### vrs kernel node sets per step, by what changed vs the node's previous set (setlog, steps 200-260)\n")
    print("| class | steps | sets / step | function | grid/block/smem only | args only | no-op (identical bytes) |")
    print("|---|---|---|---|---|---|---|")
    for k in ("decode_same", "decode_changed", "mixed"):
        if n[k]:
            a = agg[k]
            print(f"| {k} | {n[k]} | {a['sets'] / n[k]:.0f} | {a['function'] / n[k]:.0f} | {a['grid/block/smem'] / n[k]:.0f} | {a['args only'] / n[k]:.0f} | {a['no-op (identical)'] / n[k]:.0f} |")


def timeline(path):
    ev = json.load(open(path))["traceEvents"]
    X = [e for e in ev if e.get("ph") == "X"]
    ex0 = sorted([e for e in X if e["name"] == "PH_execute_model"], key=lambda e: e["ts"])
    ex = [e for e in ex0 if not any(o is not e and o["ts"] <= e["ts"] and e["ts"] + e["dur"] <= o["ts"] + o["dur"] for o in ex0)]
    lab = sorted([e for e in X if e["name"].startswith("STEP_")], key=lambda e: e["ts"])
    rt = {e.get("args", {}).get("correlation"): e for e in X if e.get("cat") in ("cuda_runtime", "cuda_driver") and e.get("args", {}).get("correlation") is not None}
    gpu = sorted([e for e in X if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")], key=lambda e: e["ts"])
    out, prev = [], None
    for i, e in enumerate(ex):
        t0, t1 = e["ts"], (ex[i + 1]["ts"] if i + 1 < len(ex) else e["ts"] + e["dur"])
        lb = next((l["name"] for l in lab if t0 <= l["ts"] <= t0 + e["dur"]), None)
        r_, t_ = map(int, re.match(r"STEP_(\d+)x(\d+)", lb).groups()) if lb else (None, None)
        mine = [g for g in gpu if (src := rt.get(g.get("args", {}).get("correlation"))) is not None and t0 <= src["ts"] < t1]
        busy = sum(g["dur"] for g in mine)
        inwin = [g for g in gpu if g["ts"] < t1 and g["ts"] + g["dur"] > t0]
        cov, cur = 0.0, None
        for g in inwin:
            a, b = max(g["ts"], t0), min(g["ts"] + g["dur"], t1)
            if cur is None or a > cur[1]:
                if cur:
                    cov += cur[1] - cur[0]
                cur = [a, b]
            else:
                cur[1] = max(cur[1], b)
        if cur:
            cov += cur[1] - cur[0]
        k = "mixed" if (t_ or 0) > (r_ or 0) else ("decode_changed" if prev is not None and prev != (r_, t_) else "decode_same")
        prev = (r_, t_)
        out.append({"cls": k, "size": (r_, t_), "interval_us": t1 - t0, "exec_us": e["dur"], "gpu_busy_launched_us": busy, "gpu_idle_in_window_us": (t1 - t0) - cov})
    return out


for arm in ("vrs", "bnc"):
    p = f"{HERE}/serve/out/prof_churn/{arm}_cb64_200_260.json"
    if not os.path.exists(p):
        continue
    T = timeline(p)
    print(f"\n### {arm} timeline per step class (profiled; unsynced churn C = 64, steps 200-260)\n")
    print("| class | steps | interval us | execute_model host us | GPU busy launched by the step us | GPU idle inside the step's interval us |")
    print("|---|---|---|---|---|---|")
    for k in ("decode_same", "decode_changed", "mixed"):
        xs = [x for x in T if x["cls"] == k]
        if xs:
            m = lambda f: st.median(x[f] for x in xs)
            print(f"| {k} | {len(xs)} | {m('interval_us'):.0f} | {m('exec_us'):.0f} | {m('gpu_busy_launched_us'):.0f} | {m('gpu_idle_in_window_us'):.0f} |")
