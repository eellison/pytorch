# Host time before the graph launch in each DECODE_STEP of a drive_ix3 trace: inclusive CPU time of python functions/ops that end
# before the step's cudaGraphLaunch, aggregated per step; prints both arms' top entries and the largest differences.
import collections, json, sys


def host(p):
    ev = [e for e in json.load(open(p))["traceEvents"] if e.get("ph") == "X"]
    steps = sorted((e for e in ev if e["name"].startswith("DECODE_STEP_") and e.get("cat") == "user_annotation"), key=lambda e: e["ts"])
    gl = sorted((e for e in ev if e["name"] in ("cudaGraphLaunch", "cuGraphLaunch")), key=lambda e: e["ts"])
    agg, pre = collections.defaultdict(float), []
    for s in steps:
        t0, t1 = s["ts"], s["ts"] + s["dur"]
        g = next(e for e in gl if t0 <= e["ts"] <= t1)
        pre.append(g["ts"] - t0)
        for e in ev:
            if e.get("cat") in ("python_function", "cpu_op") and t0 <= e["ts"] and e["ts"] + e["dur"] <= g["ts"]:
                agg[e["name"][:120]] += e["dur"] / len(steps)
    return pre, agg


(pa, A), (pb, B) = host(sys.argv[1]), host(sys.argv[2])
print("pre-launch us per step: A", [round(x) for x in pa], " B", [round(x) for x in pb])
print("\nlargest A - B (inclusive us per step):")
for n in sorted(set(A) | set(B), key=lambda n: -(A.get(n, 0) - B.get(n, 0)))[:25]:
    print(f"  {A.get(n, 0):8.0f} {B.get(n, 0):8.0f}  {A.get(n, 0) - B.get(n, 0):+8.0f}  {n}")
