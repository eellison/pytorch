# Where the tail stalls come from, per point, from the worker's step timeline (ARMV_STEPLOG=1 servers; ix_run.py saves
# *_steps.json for the timed run).
#   python3 serve/ix/ix_tails.py TAG [TAG ...]            (or explicit *_steps.json paths)
# Each step gets one class, first match: trace (a host trace ran), learn (an out-of-band harvest), redispatch, eager (an entry
# or adapter eager call, no replay), replay, else vLLM's dispatch mode (FULL / PIECEWISE / NONE: the default arms).
# Per request (worker step indices): prefill window = admit step start .. first-token step end; decode window = first-token
# step end .. finish step end. The step running when a request arrived is not in its window (the worker cannot see arrival);
# "prev" counts the class of the step before the admit step. Step seconds are host time inside execute_model; "between" is
# the time between consecutive steps (vLLM's sample_tokens / output processing / scheduling; under async scheduling
# also the wait for the GPU), so a window's classes sum to 100%.
# Rows: all requests vs the slowest 10% by prefill window (TTFT tail) and by decode time per token (TPOT tail): mean window,
# and the share of the window in each step class (trace / learn / redispatch / eager / replay / dispatch modes).
import collections
import glob
import json
import os
import statistics
import sys

SERVE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORDER = ("trace", "learn", "redispatch", "eager", "replay")


def step_class(s, c):
    for k in ("traces", "learns", "redispatches", "eager", "replays"):
        if s[c[k]]:
            return {"traces": "trace", "learns": "learn", "redispatches": "redispatch", "eager": "eager", "replays": "replay"}[k]
    return s[c["cg_mode"]] or "none"


def shares(win, steps, cls):
    """Seconds of each class over the steps in [lo, hi] (step indices), and the window's seconds."""
    lo, hi = win
    out = collections.Counter()
    for i in range(lo, hi + 1):
        out[cls[i]] += steps[i][1] - steps[i][0]
        if i < hi:
            out["between"] += steps[i + 1][0] - steps[i][1]
    return out


def report(path):
    d = json.load(open(path))
    c = {k: i for i, k in enumerate(d["cols"])}
    steps = d["steps"]
    if not steps:
        return
    cls = [step_class(s, c) for s in steps]
    span = steps[-1][1] - steps[0][0]
    per = collections.defaultdict(list)
    for s, k in zip(steps, cls):
        per[k].append(s[1] - s[0])
    name = os.path.basename(path).replace("_steps.json", "")
    print(f"\n### {name}\n")
    print(f"{len(steps)} steps over {span:.1f} s. Learn host s {sum(s[c['learn_s']] for s in steps):.1f}, redispatch host s "
          f"{sum(s[c['redispatch_s']] for s in steps):.1f}.\n")
    print("| step class | steps | seconds (share of run) | median ms | p99 ms | max ms |")
    print("|---|---|---|---|---|---|")
    for k, v in sorted(per.items(), key=lambda kv: -sum(kv[1])):
        v = sorted(v)
        print(f"| {k} | {len(v)} | {sum(v):.1f} ({100 * sum(v) / span:.0f}%) | {1e3 * statistics.median(v):.1f} | {1e3 * v[min(len(v) - 1, int(0.99 * len(v)))]:.1f} | {1e3 * v[-1]:.1f} |")
    reqs = []
    for rid, (a, f, z) in d["reqs"].items():
        if f is None:
            continue
        pre = shares((a, f), steps, cls)
        r = {"pre_s": steps[f][1] - steps[a][0], "pre": pre, "prev": cls[a - 1] if a > 0 else None}
        if z is not None and z > f:
            dec = shares((f + 1, z), steps, cls)
            r.update(tpot=(steps[z][1] - steps[f][1]) / (z - f), dec=dec, dec_s=steps[z][1] - steps[f][1])
        reqs.append(r)
    if not reqs:
        return
    classes = [k for k in ORDER if k in per] + sorted(k for k in per if k not in ORDER) + ["between"]

    def row(label, rs, win, key):
        tot = collections.Counter()
        for r in rs:
            tot.update(r[key])
        t = sum(tot.values()) or 1
        prev = collections.Counter(r["prev"] for r in rs if r["prev"] is not None)
        mean = statistics.mean(r[win] for r in rs)
        print(f"| {label} | {len(rs)} | {1e3 * mean:.1f} | " + " | ".join(f"{100 * tot[k] / t:.0f}%" for k in classes)
              + f" | {', '.join(f'{k} {n}' for k, n in prev.most_common(3))} |")

    print("\n| requests | n | mean window ms | " + " | ".join(classes) + " | prev step (prefill rows) |")
    print("|---|---|---|" + "---|" * len(classes) + "---|")
    n10 = max(1, len(reqs) // 10)
    row("all, prefill window", reqs, "pre_s", "pre")
    row("TTFT tail 10%, prefill window", sorted(reqs, key=lambda r: -r["pre_s"])[:n10], "pre_s", "pre")
    dec = [r for r in reqs if "tpot" in r]
    if dec:
        row("all, decode window (per token)", [dict(r, w=r["tpot"]) for r in dec], "w", "dec")
        row("TPOT tail 10%, decode window (per token)", [dict(r, w=r["tpot"]) for r in sorted(dec, key=lambda r: -r["tpot"])[:max(1, len(dec) // 10)]], "w", "dec")


for arg in sys.argv[1:]:
    paths = [arg] if arg.endswith(".json") else sorted(glob.glob(os.path.join(SERVE, "out", "ix", arg, "*_steps.json")))
    for p in paths:
        report(p)
