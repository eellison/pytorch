"""mktable27.py LABEL=TAG... : per-phase table for 27B startup runs (learnfast/out/startup27_h3t_vt_rw_sec_r<TAG>.json + logs/rw_vt<TAG>.log),
reference columns from the 10-15 GPU 0 vLLM runs (fullnc / breakable+gdnbreak)."""
import json
import subprocess
import sys

L = "/data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast"
REF = {"vLLM fullnc": (f"{L}/out/startup27_h2t_fullnc_r1.json", f"{L}/logs/st27h2_fullnc_r1.log"),
       "vLLM breakable+gdnbreak": (f"{L}/out/startup27_h2t_brk_r1.json", f"{L}/logs/st27h2_brk_r1.log")}


def rows(js, log):
    out = subprocess.run([sys.executable, f"{L}/breakdown27.py", js, log], capture_output=True, text=True, check=True).stdout
    r = {}
    for line in out.splitlines():
        k, v = (x.strip() for x in line.split("\t", 1))
        if k.startswith("ours warm-up"):
            k = k.split(":")[0]
        elif k.startswith("stream first-use"):
            k, v = "stream first-use (traces / redispatches)", k.split(": ", 1)[1].replace("harvests 0", "").strip(", ")
        elif k.startswith("ready to steady"):
            k = "ready to steady (last event)"
        elif k.startswith("FlashInfer autotune"):
            k = "FlashInfer autotune"
        r[k] = v
    d = json.load(open(js))["startup"]
    if d.get("range_warm"):
        r["peak allocated GiB (16K prefill / mixed / stream)"] = " / ".join(f"{d['range_warm']['phases'][p]['max_alloc_gib']:.2f}" for p in ("prefill1", "mixed")) + f" / {d['stream_max_alloc_gib']:.2f}"
        r["events: warm-up redispatches / stream traces + redispatches"] = f"{sum(p['events'][3] for p in d['range_warm']['phases'].values())} / {d['events_total'][0]} + {d['events_total'][3]}"
    return r


cols = dict((k, rows(*v)) for k, v in REF.items())
for spec in sys.argv[1:]:
    label, tag = spec.split("=")
    cols[label] = rows(f"{L}/out/startup27_h3t_vt_rw_sec_r{tag}.json", f"{L}/logs/rw_vt{tag}.log")
keys = []
for c in cols.values():
    for k in c:
        if k not in keys:
            keys.append(k)
print("| phase | " + " | ".join(cols) + " |")
print("|---|" + "---|" * len(cols))
for k in keys:
    print(f"| {k} | " + " | ".join(cols[c].get(k, "") for c in cols) + " |")
