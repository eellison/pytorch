# Capture-size study tables (10-13): vLLM with the capture range raised past g4's mixed shapes vs ours (vcs) and g4f's vLLM
# configs; step ms medians [min-max] over runs, the cudagraph mode each row's steps dispatched to, and startup / memory.
#   python3 serve/ix/capsz_table.py
import glob, json, math, os, statistics

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "vllm", "out")
ARMS = [("g4f", "default", "vLLM default (capture <= 1024)"), ("capsz", "dc2304", "vLLM default, capture <= 2304"),
        ("capsz", "dc4096", "vLLM default, capture <= 4096"), ("g4f", "fullnc", "vLLM default-nc (FULL_DECODE_ONLY)"),
        ("capsz", "bnc2304", "vLLM no compile, breakable FULL_AND_PIECEWISE <= 2304"), ("capsz", "bnc4096", "vLLM no compile, breakable <= 4096"),
        ("capsz", "fnc2304", "vLLM no compile, FULL requested <= 2304"), ("g4f", "vcs", "ours: Vc + sampling")]
runs = {a: [json.load(open(f)) for f in sorted(glob.glob(os.path.join(OUT, f"t_{s}_{a}_r*.json")))] for s, a, _ in ARMS}
runs = {a: v for a, v in runs.items() if v}
R0 = runs["vcs"][0]
ROWS = [(f"decode bs {b}", ("decode", b)) for b in sorted(R0["decode"], key=int)] + [(k[6:].replace("_", " "), ("ix", k)) for k in R0["ix"]]


def cell(a, path):
    v = [R[path[0]][path[1]]["step_ms"] for R in runs[a] if R.get(path[0], {}).get(path[1], {}).get("step_ms")]
    return (statistics.median(v), min(v), max(v)) if v else None


def modes(a, path):
    m = {}
    for R in runs[a]:
        for k, n in R.get(path[0], {}).get(path[1], {}).get("cg_modes", {}).items():
            m[k.split(".")[-1]] = m.get(k.split(".")[-1], 0) + n
    return "/".join(sorted(m)) if m else "?"


arms = [a for _, a, _ in ARMS if a in runs]
print("| workload | " + " | ".join(arms) + " |")
print("|---|" + "---|" * len(arms))
for lab, path in ROWS:
    cs = []
    for a in arms:
        c = cell(a, path)
        cs.append(f"{c[0]:.2f} [{c[1]:.2f}-{c[2]:.2f}] {modes(a, path)}" if c else "-")
    print(f"| {lab} | " + " | ".join(cs) + " |")
print("\nRatio vLLM ms / ours (vcs) ms (> 1: ours faster):\n")
print("| workload | " + " | ".join(a for a in arms if a != "vcs") + " |")
print("|---|" + "---|" * (len(arms) - 1))
gm = {a: [] for a in arms}
for lab, path in ROWS:
    o = cell("vcs", path)
    row = []
    for a in arms:
        if a == "vcs":
            continue
        c = cell(a, path)
        row.append(f"{c[0] / o[0]:.2f}" if c and o else "-")
        if c and o:
            gm[a].append(c[0] / o[0])
    print(f"| {lab} | " + " | ".join(row) + " |")
print("| geomean | " + " | ".join(f"{math.exp(sum(map(math.log, gm[a])) / len(gm[a])):.2f}" if gm[a] else "-" for a in arms if a != "vcs") + " |")
print("\n| measure | " + " | ".join(arms) + " |")
print("|---|" + "---|" * len(arms))


def med(a, f):
    v = [x for R in runs[a] if (x := f(R)) is not None]
    return statistics.median(v) if v else None


for lab, f, fmt in (("engine ready (init) s", lambda R: R["init_s"], "{:.0f}"),
                    ("capture_model s", lambda R: R.get("capture", {}).get("capture_model_s"), "{:.0f}"),
                    ("CUDA graph captures begun", lambda R: R.get("capture", {}).get("capture_begins"), "{:.0f}"),
                    ("descriptors FULL", lambda R: R.get("capture", {}).get("descs", {}).get("FULL"), "{:.0f}"),
                    ("descriptors PIECEWISE", lambda R: R.get("capture", {}).get("descs", {}).get("PIECEWISE"), "{:.0f}"),
                    ("graph memory GiB (capture free-mem delta)", lambda R: R["capture"]["graph_bytes"] / 2**30 if "capture" in R else None, "{:.2f}"),
                    ("graph memory estimate GiB (taken from KV budget)", lambda R: (R.get("capture", {}).get("cudagraph_memory_estimate") or 0) / 2**30 if "capture" in R else None, "{:.2f}"),
                    ("KV cache tokens", lambda R: R["mem"]["kv_cache_tokens"], "{:.0f}"),
                    ("max capture size", lambda R: R["cg_config"]["max_capture_size"], "{:.0f}"),
                    ("cudagraph_mode (resolved)", lambda R: None, "")):
    vals = []
    for a in arms:
        if lab.startswith("cudagraph_mode"):
            vals.append(runs[a][0]["cg_config"]["cudagraph_mode"].split(".")[-1])
            continue
        m = med(a, f)
        vals.append(fmt.format(m) if m is not None else "-")
    print(f"| {lab} | " + " | ".join(vals) + " |")
print("\nruns: " + ", ".join(f"{a} {len(runs[a])}" for a in arms))
