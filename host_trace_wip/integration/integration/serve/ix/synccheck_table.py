# synccheck (main 10-14): CUDA synchronize calls inside one profiled full step (execute_model .. sample_tokens) per arm, prefill 512 and
# mixed 1024 + 64, swph1's timed configuration.   python3 serve/ix/synccheck_table.py
import collections, glob, json, os

D = "serve/out/prof_synccheck"
print("| arm | step | cudaDeviceSynchronize | cudaStreamSynchronize | cudaEventSynchronize | other *Synchronize | graph launches | step wall us |")
print("|---|---|---|---|---|---|---|---|")
for arm in ("default", "fullnc", "bnc", "vcsl", "vrs"):
    for cls, name in (("1x512", "prefill 512"), ("65x1088", "mixed 1024+64")):
        p = f"{D}/{arm}_{cls}.json"
        if not os.path.exists(p):
            print(f"| {arm} | {name} | - | - | - | - | - | - |")
            continue
        ev = [e for e in json.load(open(p))["traceEvents"] if e.get("ph") == "X"]
        rt = [e for e in ev if e.get("cat") in ("cuda_runtime", "cuda_driver")]
        c = collections.Counter(e["name"] for e in rt if "Synchronize" in e["name"])
        other = sum(v for k, v in c.items() if k not in ("cudaDeviceSynchronize", "cudaStreamSynchronize", "cudaEventSynchronize"))
        gl = sum(1 for e in rt if "GraphLaunch" in e["name"])
        ex = [e for e in ev if e["name"] in ("PH_execute_model", "PH_sample_tokens")]
        wall = (max(e["ts"] + e["dur"] for e in ex) - min(e["ts"] for e in ex)) if ex else 0
        print(f"| {arm} | {name} | {c['cudaDeviceSynchronize']} | {c['cudaStreamSynchronize']} | {c['cudaEventSynchronize']} | {other} | {gl} | {wall:.0f} |")
