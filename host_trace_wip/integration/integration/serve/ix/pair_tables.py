# Reporting tables for a full-step set (the user's rules, 10-13): (1) headline = ONE vLLM config per model, the faster overall
# across the timed rows (geomean), vs ONE config of ours (the best: compiled forward + sampling in the graph), labelled; (2) the
# matched pairs below, compile matched to compile: compiled (vLLM default vs ours over the compiled forward) and uncompiled (vLLM
# default-nc vs ours on the eager forward), each with ours with and without sampling in the step. Ratios = vLLM ms / ours ms (> 1:
# ours faster); no cross-compile ratios.
#   python3 serve/ix/pair_tables.py SET [--ours vcs]     (vllm/out/t_<SET>_<arm>_r*.json)
import argparse
import glob
import json
import math
import os
import statistics

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "vllm", "out")
ap = argparse.ArgumentParser()
ap.add_argument("set")
ap.add_argument("--ours", default="vcs")
a = ap.parse_args()
NAMES = {"default": "vLLM default (compile + graphs)", "fullnc": "vLLM default-nc (no compile, FULL decode graphs)",
         "vc": "ours: compiled forward", "vcs": "ours: compiled forward + sampling in graph",
         "vreg": "ours: eager forward", "vrs": "ours: eager forward + sampling in graph"}
runs = {k: [json.load(open(f)) for f in sorted(glob.glob(os.path.join(OUT, f"t_{a.set}_{k}_r*.json")))] for k in NAMES}
runs = {k: v for k, v in runs.items() if v}


def rows_of():
    R = next(iter(runs.values()))[0]
    rows = [(f"decode bs {b}", lambda R, b=b: R["decode"][b]["step_ms"]) for b in sorted(R.get("decode", {}), key=int)]
    keys = sorted(R.get("ix", {}), key=lambda k: (not k.startswith("mixed"), k))
    rows += [(k.replace("_", " "), lambda R, k=k: R["ix"][k]["step_ms"]) for k in keys]
    return rows


ROWS = rows_of()


def med(arm, get):
    v = []
    for R in runs.get(arm, []):
        try:
            x = get(R)
        except (KeyError, TypeError):
            x = None
        if x is not None:
            v.append(x)
    return (statistics.median(v), min(v), max(v)) if v else None


def geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


vals = {arm: [med(arm, g) for _, g in ROWS] for arm in runs}
vllm = [x for x in ("default", "fullnc") if x in runs]
gm = {x: geomean([c[0] for c in vals[x] if c]) for x in vllm}
base = min(gm, key=gm.get)
f = lambda c: f"{c[0]:.2f} [{c[1]:.2f}-{c[2]:.2f}]" if c else "-"
r = lambda b, o: f"{b[0] / o[0]:.2f}" if b and o else "-"
print(f"## Headline: {NAMES[base]} vs {NAMES[a.ours]}\n")
print(f"vLLM config: {NAMES[base]}, the faster vLLM config overall (geomean step ms over the rows: "
      + ", ".join(f"{k} {v:.2f}" for k, v in gm.items()) + f"), used for every row. Ours: {NAMES[a.ours]}.\n")
print(f"| workload | {base} ms | {a.ours} ms | {base} / {a.ours} |")
print("|---|---|---|---|")
for (lab, _), b, o in zip(ROWS, vals[base], vals[a.ours]):
    print(f"| {lab} | {f(b)} | {f(o)} | {r(b, o)} |")
print(f"| geomean | | | {geomean([b[0] / o[0] for b, o in zip(vals[base], vals[a.ours]) if b and o]):.2f} |")
for title, vl, ours in (("Compiled pair (compile on for both)", "default", ("vcs", "vc")),
                        ("Uncompiled pair (compile off for both)", "fullnc", ("vrs", "vreg"))):
    if vl not in runs or not all(o in runs for o in ours):
        continue
    print(f"\n## {title}: {NAMES[vl]} vs {NAMES[ours[0]]} (secondary: {NAMES[ours[1]]})\n")
    print(f"| workload | {vl} ms | {ours[0]} ms | {ours[1]} ms | {vl} / {ours[0]} | {vl} / {ours[1]} |")
    print("|---|---|---|---|---|---|")
    for (lab, _), b, o, s in zip(ROWS, vals[vl], vals[ours[0]], vals[ours[1]]):
        print(f"| {lab} | {f(b)} | {f(o)} | {f(s)} | {r(b, o)} | {r(b, s)} |")
    print(f"| geomean | | | | {geomean([b[0] / o[0] for b, o in zip(vals[vl], vals[ours[0]]) if b and o]):.2f} | "
          f"{geomean([b[0] / s[0] for b, s in zip(vals[vl], vals[ours[1]]) if b and s]):.2f} |")
print("\n## Memory and startup\n")
arms = list(runs)
print("| measure | " + " | ".join(arms) + " |")
print("|---|" + "---|" * len(arms))
for lab, get in (("bench peak over init MiB", lambda R: R["mem"]["bench_peak_over_init_mib"]), ("init s", lambda R: R["init_s"]),
                 ("model forwards at init", lambda R: R["model_forwards"]["init"])):
    print(f"| {lab} | " + " | ".join(f"{c[0]:.0f}" if (c := med(x, get)) else "-" for x in arms) + " |")
print("\nruns per arm: " + ", ".join(f"{k} {len(v)}" for k, v in runs.items()))
