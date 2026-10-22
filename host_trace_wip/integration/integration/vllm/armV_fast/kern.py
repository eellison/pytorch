# python3 armV/kern.py A.json B.json: per-kernel GPU time of two Kineto traces of one step (A vs B), grouped by kernel
# name, plus GPU span (first start to last end), busy time, kernel count, and early starts (a kernel starting before
# the previous one ends: PDL overlap).
import collections, json, sys


def load(path):
    ev = json.load(open(path))["traceEvents"]
    k = sorted((e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")), key=lambda e: e["ts"])
    busy, end, early, gaps = 0.0, k[0]["ts"], 0, 0.0
    for x in k:
        if x["ts"] < end:
            early += 1
        else:
            gaps += x["ts"] - end
        busy += max(0.0, x["ts"] + x["dur"] - max(x["ts"], end))
        end = max(end, x["ts"] + x["dur"])
    by = collections.defaultdict(lambda: [0, 0.0])
    for x in k:
        r = by[x["name"][:90]]
        r[0] += 1
        r[1] += x["dur"]
    head = {"span_us": round(end - k[0]["ts"], 1), "busy_us": round(busy, 1), "gaps_us": round(gaps, 1), "ops": len(k), "early_starts": early}
    return head, by


(ha, a), (hb, b) = load(sys.argv[1]), load(sys.argv[2])
print("A", sys.argv[1], ha)
print("B", sys.argv[2], hb)
rows = sorted(set(a) | set(b), key=lambda n: -abs(a.get(n, [0, 0.0])[1] - b.get(n, [0, 0.0])[1]))
print(f"{'A n':>5} {'A us':>8} {'B n':>5} {'B us':>8} {'A-B':>8}  kernel")
for n in rows[: int(sys.argv[3]) if len(sys.argv) > 3 else 25]:
    an, au = a.get(n, [0, 0.0])
    bn, bu = b.get(n, [0, 0.0])
    print(f"{an:5d} {au:8.1f} {bn:5d} {bu:8.1f} {au - bu:8.1f}  {n}")
