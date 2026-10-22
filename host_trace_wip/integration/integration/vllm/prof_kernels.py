# Kernel-time totals by (shortened) kernel name for the middle profiled step: python prof_kernels.py a.json b.json
import collections, json, re, sys


def load(path):
    ev = json.load(open(path))["traceEvents"]
    step = next(e for e in ev if e.get("name") == "ENGINE_STEP_1" and e.get("ph") == "X" and e.get("cat") in ("user_annotation", "cpu_op"))
    rt = {e["args"]["correlation"]: e for e in ev if e.get("cat") in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {})}
    tot, cnt = collections.Counter(), collections.Counter()
    for e in ev:
        if e.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset") or e.get("ph") != "X":
            continue
        r = rt.get(e.get("args", {}).get("correlation"))
        if r is None or not (step["ts"] <= r["ts"] <= step["ts"] + step["dur"]):
            continue
        name = re.sub(r"<.*", "", e["name"])[:60]
        tot[name] += e["dur"]
        cnt[name] += 1
    return tot, cnt


a, ac = load(sys.argv[1])
b, bc = load(sys.argv[2])
print(f"total kernel us: A {sum(a.values()):.0f} ({sum(ac.values())} ops)  B {sum(b.values()):.0f} ({sum(bc.values())} ops)")
names = sorted(set(a) | set(b), key=lambda n: -abs(b[n] - a[n]))
for n in names[:25]:
    print(f"{b[n] - a[n]:9.1f} us  A {a[n]:8.1f} ({ac[n]:3d})  B {b[n]:8.1f} ({bc[n]:3d})  {n}")
