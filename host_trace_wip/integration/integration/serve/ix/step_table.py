# Controlled-step table (the g4 / q1 shape of vllm/c2tables.py) for one rounds set, plus V-stock counts per run.
#   python3 serve/ix/step_table.py q1p [--ref q1]      (vllm/out/t_<set>_{default,fullnc,vstock}_r*.json)
# --ref SET adds that set's V-stock column (e.g. q1: the same workload with ARMV_HOST_ARGS unset) for comparison.
import argparse
import glob
import json
import os
import statistics

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "vllm", "out")
ap = argparse.ArgumentParser()
ap.add_argument("set")
ap.add_argument("--ref", default=None)
a = ap.parse_args()


def runs(st, arm):
    return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(OUT, f"t_{st}_{arm}_r*.json")))]


arms = {"default": runs(a.set, "default"), "default-nc": runs(a.set, "fullnc"), "V-stock": runs(a.set, "vstock"), "V-registered": runs(a.set, "vreg"), "V-compiled (Vc)": runs(a.set, "vc"), "V-registered + sampling in step (vrs)": runs(a.set, "vrs"), "V-compiled + sampling in step (vcs)": runs(a.set, "vcs")}
arms = {k: v for k, v in arms.items() if v}
if a.ref:
    arms[f"V-stock ({a.ref})"] = runs(a.ref, "vstock")


def cell(rs, get):
    v = []
    for R in rs:
        try:
            x = get(R)
        except (KeyError, TypeError):
            x = None
        if x is not None:
            v.append(x)
    return (statistics.median(v), min(v), max(v)) if v else None


keys = sorted({k for rs in arms.values() for R in rs for k in R.get("ix", {})}, key=lambda k: (not k.startswith("mixed"), k))
dec_bs = sorted({b for rs in arms.values() for R in rs for b in R.get("decode", {})}, key=int)
asy_bs = sorted({b for rs in arms.values() for R in rs for b in R.get("async", {})}, key=int)
rows = [(f"decode synced {b}", lambda R, b=b: R["decode"][b]["step_ms"], None) for b in dec_bs]
rows += [(f"decode async {b}", lambda R, b=b: R["async"][b]["step_ms"], None) for b in asy_bs]
rows += [(k.replace("_", " "), lambda R, k=k: R["ix"][k]["step_ms"], k) for k in keys]
names = list(arms)
print("Step ms (execute_model, synced unless async), median over rounds [min-max]; async = (wall(48 tok) - wall(16 tok)) / 32.")
vs = [n for n in names if n.startswith("V-")]
print("| workload | " + " | ".join(names) + " | " + " | ".join(f"default / {v} | default-nc / {v}" for v in vs) + " | decoder KV min/med/max |")
print("|---|" + "---|" * len(names) + "---|---|" * len(vs) + "---|")
for lab, get, k in rows:
    m = {n: cell(arms[n], get) for n in names}
    f = lambda c: f"{c[0]:.2f} [{c[1]:.2f}-{c[2]:.2f}]" if c else "-"
    r = lambda n, v: f"{m[n][0] / m[v][0]:.2f}" if m.get(n) and m.get(v) else "-"
    kvs = [R["ix"][k].get("kv_min_med_max") for rs in arms.values() for R in rs if k and k in R.get("ix", {})] if k else []
    kvs = [x for x in kvs if x]
    kv = f"{min(x[0] for x in kvs)}/{int(statistics.median(x[1] for x in kvs))}/{max(x[2] for x in kvs)}" if kvs else "-"
    print(f"| {lab} | " + " | ".join(f(m[n]) for n in names) + " | " + " | ".join(f"{r('default', v)} | {r('default-nc', v)}" for v in vs) + f" | {kv} |")
for lab, get in (("KV tokens (k)", lambda R: R["mem"]["kv_cache_tokens"] / 1e3), ("bench peak over init MiB", lambda R: R["mem"]["bench_peak_over_init_mib"]),
                 ("init s (engine ready; V-stock traces at first call)", lambda R: R["init_s"]), ("model forwards at init", lambda R: R["model_forwards"]["init"])):
    print(f"| {lab} | " + " | ".join(f"{c[0]:.0f}" if (c := cell(arms[n], get)) else "-" for n in names) + " |" + " | |" * len(vs) + " |")
print("rounds:", {n: len(r) for n, r in arms.items()})
print()
print("V arms: counts per run (whole run: warm-up + measured steps):")
for n in names:
    if not n.startswith("V-"):
        continue
    for i, R in enumerate(arms[n]):
        V = R["V"]
        ent = {k: (e["traces"], e["replays"], e["eager"], e["variants"], [pv["eager_calls"] for pv in e["per_variant"]]) for k, e in V["entries"].items()}
        cpu = sum(int(s.split("x ", 1)[0]) for s in V.get("decline_sites", []) if "is on cpu; only CUDA tensors" in s or "an allocation on meta in a trace" in s)
        print(f"- {n} run {i + 1}: traces {V['traces']}, replays {V['replays']}, eager calls {V['eager']}, variants {V['variants']}, calls {V['calls']}"
              + (f", MISCONFIGURED ({cpu} CPU-argument / meta-allocation declines)" if cpu else ""))
        if i == 0:
            for k, (t, rp, eg, nv, ec) in ent.items():
                print(f"  - {k}: traces {t}, replays {rp}, eager {eg}, variants {nv}, eager steps per variant {ec}")
