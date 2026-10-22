"""flag_points27.py SET ARM [MIN]: the decode-curve points of ARM flagged (PSI > 2% of the window or outlier excess > 1%) in at least
MIN (default 2) of its rounds, printed as "CTX BS BS ..." lines, one per context."""
import collections, glob, json, os, sys

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out", "timing")
SET, ARM, MIN = sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 2
n = collections.Counter()
for f in glob.glob(os.path.join(D, f"t_{SET}_{ARM}_r*.json")):
    for k, v in json.load(open(f)).get("dcurve", {}).items():
        if ((v.get("stalls") or {}).get("flag") or {}).get("flagged"):
            n[k] += 1
by = collections.defaultdict(list)
for k, c in n.items():
    if c >= MIN:
        ctx, bs = map(int, k.split("_"))
        by[ctx].append(bs)
for ctx in sorted(by):
    print(ctx, " ".join(map(str, sorted(by[ctx]))))
