# Per-step out-of-graph breakdown from drive_ix2.py's profiled full steps (DRIVE_PROFILE_DIR): for each chrome trace (one step,
# execute_model through sample_tokens), the CUDA runtime/driver calls by kind (kernel launch, graph launch, memcpy by direction,
# memset) with count and host us, and the device work they issued: kernels and copies outside any graph (count, bytes, device us)
# vs kernels inside graph launches; each split by phase (input prep, metadata tables, metadata build, rest of execute_model =
# forward, sample_tokens) from the PH_* annotations (a call is in the innermost PH range enclosing it on its thread).
#   python3 serve/ix/prof_breakdown.py DIR [DIR ...]
import collections
import glob
import json
import os
import sys

PHASES = ("PH_input_prep", "PH_metadata_tables", "PH_metadata_build", "PH_sample_tokens", "PH_execute_model")


def kind_of(name):
    n = name.lower()
    if "graphlaunch" in n:
        return "graph launch"
    if "launchkernel" in n or "launchcooperativekernel" in n:
        return "kernel launch"
    if "memcpy" in n:
        return "memcpy"
    if "memset" in n:
        return "memset"
    return None


def analyze(path):
    ev = json.load(open(path))["traceEvents"]
    ranges = [e for e in ev if e.get("ph") == "X" and e.get("name") in PHASES]
    rt = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("cuda_runtime", "cuda_driver") and kind_of(e["name"])]
    dev = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    corr = {}
    for e in rt:
        c = e.get("args", {}).get("correlation")
        if c is not None:
            corr[c] = e

    def phase(e):
        best = None
        for r in ranges:
            if r.get("tid") == e.get("tid") and r["ts"] <= e["ts"] <= r["ts"] + r["dur"]:
                if best is None or r["dur"] < best["dur"]:
                    best = r
        if best is None:
            return "other"
        return {"PH_execute_model": "forward (rest of execute_model)"}.get(best["name"], best["name"][3:])

    host = collections.defaultdict(lambda: [0, 0.0])
    for e in rt:
        k = (phase(e), kind_of(e["name"]) + (" " + memdir(e) if kind_of(e["name"]) == "memcpy" else ""))
        host[k][0] += 1
        host[k][1] += e["dur"]
    devd = collections.defaultdict(lambda: [0, 0.0, 0])
    names = collections.Counter()
    for d in dev:
        c = d.get("args", {}).get("correlation")
        src = corr.get(c)
        in_graph = src is not None and kind_of(src["name"]) == "graph launch"
        ph = phase(src) if src is not None else "other"
        cat = "in graph" if in_graph else {"kernel": "kernel outside graph", "gpu_memcpy": "memcpy outside graph", "gpu_memset": "memset outside graph"}[d["cat"]]
        if cat.startswith("memcpy"):
            cat += " " + d["name"].split(" ")[-1] if " " in d["name"] else ""
        key = (ph, cat)
        devd[key][0] += 1
        devd[key][1] += d["dur"]
        devd[key][2] += int(d.get("args", {}).get("bytes", 0) or 0)
        if not in_graph and d["cat"] == "kernel":
            names[d["name"][:70]] += 1
    return host, devd, names


def memdir(e):
    a = e.get("args", {})
    for k in ("kind", "copy_kind"):
        if k in a:
            return str(a[k])
    return ""


for arg in sys.argv[1:]:
    for path in sorted(glob.glob(os.path.join(arg, "*.json"))):
        host, devd, names = analyze(path)
        print(f"\n### {os.path.basename(path)[:-5]}\n")
        print("| phase | runtime call | count | host us |")
        print("|---|---|---|---|")
        for (ph, k), (n, us) in sorted(host.items()):
            print(f"| {ph} | {k} | {n} | {us:.0f} |")
        print("\n| phase | device work | count | device us | bytes |")
        print("|---|---|---|---|---|")
        for (ph, k), (n, us, b) in sorted(devd.items()):
            print(f"| {ph} | {k} | {n} | {us:.0f} | {b} |")
        out_k = sum(n for (ph, k), (n, us, b) in devd.items() if k != "in graph")
        out_us = sum(us for (ph, k), (n, us, b) in devd.items() if k != "in graph")
        print(f"\nOutside our/vLLM's graphs per step: {out_k} device ops, {out_us:.0f} us device. Top kernels outside graphs: "
              + "; ".join(f"{k} x{n}" for k, n in names.most_common(6)))
