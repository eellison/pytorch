# Compare two drive_ix3 decode traces: per step wall / GPU busy / first-kernel delay, per-kernel GPU time (per step), and the kernel
# names only one arm runs.  python prof_dec_cmp.py A.json B.json
import collections, json, sys


def load(p):
    ev = [e for e in json.load(open(p))["traceEvents"] if e.get("ph") == "X"]
    steps = sorted((e for e in ev if e["name"].startswith("DECODE_STEP_")), key=lambda e: e["ts"])
    kern = sorted((e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")), key=lambda e: e["ts"])
    rt = [e for e in ev if e.get("cat") in ("cuda_runtime", "cuda_driver")]
    out, per = [], collections.defaultdict(lambda: [0, 0.0])
    for s in steps:
        t0, t1 = s["ts"], s["ts"] + s["dur"]
        ks = [k for k in kern if t0 <= k["ts"] <= t1]
        busy, end = 0.0, -1
        for k in ks:
            a, b = k["ts"], k["ts"] + k["dur"]
            if b > end:
                busy += b - max(a, end); end = b
        gl = [e for e in rt if t0 <= e["ts"] <= t1 and e["name"] in ("cudaGraphLaunch", "cuGraphLaunch")]
        first_launch = gl[0]["ts"] - t0 if gl else None
        gpu_first = ks[0]["ts"] - t0 if ks else None
        gpu_span = (ks[-1]["ts"] + ks[-1]["dur"] - ks[0]["ts"]) if ks else 0
        rc = collections.Counter(e["name"] for e in rt if t0 <= e["ts"] <= t1)
        out.append(dict(wall=s["dur"], busy=busy, n=len(ks), span=gpu_span, graph_launch_at=first_launch, gpu_first_at=gpu_first, gl=len(gl),
                        launches=sum(v for k_, v in rc.items() if "aunchKernel" in k_), memcpy=sum(v for k_, v in rc.items() if "emcpy" in k_)))
        for k in ks:
            per[k["name"]][0] += 1; per[k["name"]][1] += k["dur"]
    return out, {k: (c / len(steps), d / len(steps)) for k, (c, d) in per.items()}


A, B = sys.argv[1], sys.argv[2]
(sa, ka), (sb, kb) = load(A), load(B)
for name, st in ((A, sa), (B, sb)):
    print(name)
    for s in st:
        print("  wall {wall:8.0f} us  GPU busy {busy:8.0f}  GPU span {span:8.0f}  kernels {n}  graph launches {gl}  kernel launches {launches}  memcpy {memcpy}  "
              "graph launch at +{graph_launch_at}  first GPU op at +{gpu_first_at}".format(**s))
print("\nper-kernel GPU us per step (A vs B), largest |diff| first:")
names = set(ka) | set(kb)
rows = sorted(names, key=lambda n: -abs(ka.get(n, (0, 0))[1] - kb.get(n, (0, 0))[1]))
for n in rows[:30]:
    ca, da = ka.get(n, (0, 0)); cb, db = kb.get(n, (0, 0))
    print(f"  A {ca:5.0f}x {da:8.1f}  B {cb:5.0f}x {db:8.1f}  diff {da - db:+8.1f}  {n[:150]}")
print("\nGPU total per step: A %.0f us, B %.0f us" % (sum(d for _, d in ka.values()), sum(d for _, d in kb.values())))
