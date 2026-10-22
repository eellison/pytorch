# prof_one.py TRACE...: one profiled full step (drive_sweep27 DRIVE_PROFILE_DIR chrome trace): wall (PH_execute_model start ->
# PH_sample_tokens end), GPU busy (union of kernel / memcpy / memset intervals), kernels, launch API calls, graph launches, and
# GPU busy per annotated phase window.
import json, re, sys

LAUNCH = re.compile(r"cudaLaunchKernel|cuLaunchKernel|cudaLaunchCooperativeKernel|cuLaunchKernelEx|cudaLaunchKernelExC")
for f in sys.argv[1:]:
    ev = [e for e in json.load(open(f)).get("traceEvents", []) if e.get("ph") == "X"]
    gpu = [e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    ph = {e["name"]: e for e in ev if str(e.get("name", "")).startswith("PH_")}
    t0 = ph["PH_execute_model"]["ts"] if "PH_execute_model" in ph else min(e["ts"] for e in ev)
    t1 = max(e["ts"] + e["dur"] for e in ([ph["PH_sample_tokens"]] if "PH_sample_tokens" in ph else ev))
    iv = sorted((max(e["ts"], t0), min(e["ts"] + e["dur"], t1)) for e in gpu if e["ts"] + e["dur"] > t0 and e["ts"] < t1)
    busy, cur = 0.0, None
    for a, b in iv:
        if cur is None or a > cur[1]:
            if cur:
                busy += cur[1] - cur[0]
            cur = [a, b]
        else:
            cur[1] = max(cur[1], b)
    if cur:
        busy += cur[1] - cur[0]
    rt = [e for e in ev if e.get("cat") in ("cuda_runtime", "cuda_driver")]
    launches = sum(1 for e in rt if LAUNCH.search(e["name"]) and t0 <= e["ts"] <= t1)
    graphs = sum(1 for e in rt if "GraphLaunch" in e["name"] and t0 <= e["ts"] <= t1)
    kernels = sum(1 for e in gpu if e.get("cat") == "kernel" and t0 <= e["ts"] <= t1)
    phases = {n: round(e["dur"] / 1e3, 2) for n, e in ph.items()}
    print(f"{f}: wall {(t1 - t0) / 1e3:.2f} ms, GPU busy {busy / 1e3:.2f} ms, host-bound share {1 - busy / (t1 - t0):.0%}, kernels {kernels}, "
          f"launch calls {launches}, graph launches {graphs}, phases (ms) {phases}")
