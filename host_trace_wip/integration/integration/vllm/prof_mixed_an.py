# Per measured mixed step of a drive_prof trace: host (step wall), GPU kernel sum and busy union, launches (graph vs kernel),
# top GPU kernels, top CPU ops by inclusive time inside the step.  python prof_mixed_an.py trace.json [top]
import collections, json, sys

T = json.load(open(sys.argv[1]))["traceEvents"]
top = int(sys.argv[2]) if len(sys.argv) > 2 else 12
ev = [e for e in T if e.get("ph") == "X"]
steps = sorted((e for e in ev if e["name"].startswith("MIXED_STEP_")), key=lambda e: e["ts"])
kern = [e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
rt = [e for e in ev if e.get("cat") == "cuda_runtime" or e.get("cat") == "cuda_driver"]
cpu = [e for e in ev if e.get("cat") in ("cpu_op", "python_function", "user_annotation")]
for s in steps:
    t0, t1 = s["ts"], s["ts"] + s["dur"]
    ks = sorted((k for k in kern if t0 <= k["ts"] <= t1), key=lambda k: k["ts"])
    busy, end = 0.0, -1
    for k in ks:
        a, b = k["ts"], k["ts"] + k["dur"]
        if b > end:
            busy += b - max(a, end); end = b
    r = collections.Counter(e["name"] for e in rt if t0 <= e["ts"] <= t1)
    print(f"{s['name']}: wall {s['dur'] / 1e3:.2f} ms, GPU kernel sum {sum(k['dur'] for k in ks) / 1e3:.2f} ms, busy {busy / 1e3:.2f} ms, "
          f"kernels {len(ks)}, graph launches {r.get('cudaGraphLaunch', 0)}, kernel launches {r.get('cudaLaunchKernel', 0) + r.get('cuLaunchKernel', 0) + r.get('cudaLaunchKernelExC', 0) + r.get('cuLaunchKernelEx', 0)}")
s = steps[len(steps) // 2]
t0, t1 = s["ts"], s["ts"] + s["dur"]
kc = collections.defaultdict(lambda: [0, 0.0])
for k in kern:
    if t0 <= k["ts"] <= t1:
        kc[k["name"][:110]][0] += 1; kc[k["name"][:110]][1] += k["dur"]
print(f"\n{s['name']} top GPU kernels (count, ms):")
for n, (c, d) in sorted(kc.items(), key=lambda x: -x[1][1])[:top]:
    print(f"  {c:4d} {d / 1e3:7.2f}  {n}")
cc = collections.defaultdict(lambda: [0, 0.0])
for e in cpu:
    if t0 <= e["ts"] <= t1 and e is not s:
        cc[e["name"][:110]][0] += 1; cc[e["name"][:110]][1] += e["dur"]
print(f"\n{s['name']} top CPU events by inclusive time (count, ms):")
for n, (c, d) in sorted(cc.items(), key=lambda x: -x[1][1])[:top * 3]:
    print(f"  {c:4d} {d / 1e3:7.2f}  {n}")
