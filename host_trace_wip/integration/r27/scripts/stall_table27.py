"""stall_table27.py NEWSET [OLDSET]: decode-curve stalls per arm (median over rounds where per-point), and new vs old per-token means.
Per arm and context: geomean of per-point mean step ms (new, old, ratio); points / steps > 10x median and how many of those steps had a
GC pass inside (GC) or not; total GC ms inside measured steps; flagged points (PSI > 2% of window or outlier excess > 1%); max and
p99 step; refreezes (ours); GC passes by generation over the run."""
import collections, glob, json, os, re, statistics as st, sys

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out", "timing")
NEW, OLD = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else None)
ARMS = ("default", "vcvp", "fullnc", "breakable", "breakablegb", "vsvp")


def runs(s, arm):
    return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(D, f"t_{s}_{arm}_r*.json")))]


def gm(xs):
    xs = [x for x in xs if x]
    return st.geometric_mean(xs) if xs else None


print(f"{NEW} decode-curve stalls (synced full steps; per point: 16 warm + 128 measured)" + (f"; old = {OLD}" if OLD else ""))
print(f"{'arm':10s} {'ctx':>4s} {'rounds':>6s} {'gm new ms':>9s} {'gm old ms':>9s} {'new/old':>7s} {'10x steps':>9s} {'w/ GC':>6s} {'no GC':>6s} "
      f"{'GC ms in steps':>14s} {'flagged pts':>11s} {'max ms':>8s} {'p99 med':>8s}")
for arm in ARMS:
    R = runs(NEW, arm)
    if not R:
        continue
    O = runs(OLD, arm) if OLD else []
    for ctx in (1024, 8192):
        keys = sorted({k for j in R for k in j.get("dcurve", {}) if k.startswith(f"{ctx}_")}, key=lambda k: int(k.split("_")[1]))
        if not keys:
            continue
        new = [st.median([j["dcurve"][k]["mean_ms"] for j in R if k in j.get("dcurve", {}) and j["dcurve"][k].get("mean_ms")]) for k in keys]
        old = [st.median(v) if (v := [j["dcurve"][k]["mean_ms"] for j in O if k in j.get("dcurve", {}) and j["dcurve"][k].get("mean_ms")]) else None for k in keys]
        n10 = gc10 = 0
        gcms, flagged, mx, p99 = 0.0, 0, 0.0, []
        for j in R:
            for k in keys:
                s_ = j["dcurve"].get(k, {}).get("stalls") or {}
                for _, _, g in s_.get("outliers_10x", []):
                    n10 += 1
                    gc10 += bool(g)
                gcms += s_.get("gc_ms_in_window") or 0
                flagged += bool((s_.get("flag") or {}).get("flagged"))
                mx = max(mx, s_.get("max_ms") or 0)
                if s_.get("p99_ms"):
                    p99.append(s_["p99_ms"] / s_["median_ms"])
        g_new, g_old = gm(new), gm([o for o in old if o])
        print(f"{arm:10s} {ctx:4d} {len(R):6d} {g_new:9.3f} {(g_old or 0):9.3f} {(g_new / g_old if g_old else 0):7.3f} {n10:9d} {gc10:6d} {n10 - gc10:6d} "
              f"{gcms:14.1f} {flagged:11d} {mx:8.1f} {(st.median(p99) if p99 else 0):8.2f}")
    gcr = collections.Counter()
    for j in R:
        for g, n in (j.get("gc_events") or {}).get("n_by_gen", {}).items():
            gcr[g] += n
    refz = [len((j.get("V") or {}).get("refreezes") or []) for j in R]
    print(f"           GC passes by generation over the runs: {dict(gcr)}" + (f"; refreezes per run: {refz}" if arm.startswith("v") else ""))
