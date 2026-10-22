"""ab_refreeze27.py: n27ab re-freeze A/B. Per point, each round's per-step median, then the median over rounds, for ours with
(vcvp / vsvp) and without (vcvpnr / vsvpnr) ARMV_REFREEZE; ratio with/without (> 1: the re-freeze arm is slower), per round and overall."""
import glob, json, os, re, statistics as st

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out", "timing")
v = {}
for f in glob.glob(os.path.join(D, "t_n27ab_*_r*.json")):
    m = re.match(r"t_n27ab_(vcvp|vcvpnr|vsvp|vsvpnr)_r(\d+)\.json", os.path.basename(f))
    if not m:
        continue
    j = json.load(open(f))
    for k, d in j["dcurve"].items():
        v.setdefault((m.group(1), k), {})[int(m.group(2))] = (d["stalls"]["median_ms"], d["mean_ms"], d["stalls"]["flag"]["flagged"])
print(f"{'point':10s} {'arm':5s} {'with ms':>8s} {'without ms':>10s} {'with/without':>12s}  per-round with/without")
allr = {"vcvp": [], "vsvp": []}
for (arm, k) in sorted(v, key=lambda x: (x[0], int(x[1].split("_")[0]), int(x[1].split("_")[1]))):
    if arm.endswith("nr") or (arm + "nr", k) not in v:
        continue
    a, b = v[(arm, k)], v[(arm + "nr", k)]
    ma, mb = st.median(x[0] for x in a.values()), st.median(x[0] for x in b.values())
    per = [round(a[r][0] / b[r][0], 4) for r in sorted(a) if r in b]
    allr[arm].append(ma / mb)
    print(f"{k:10s} {arm:5s} {ma:8.3f} {mb:10.3f} {ma / mb:12.4f}  {per}")
for arm, xs in allr.items():
    if xs:
        print(f"{arm}: geomean with/without re-freeze {st.geometric_mean(xs):.4f} over {len(xs)} points")
