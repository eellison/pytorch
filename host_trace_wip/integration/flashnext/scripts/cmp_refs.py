"""Compare stock_ref.py outputs across processes: per-forward digests (and per-row digests), NaN/inf, tokens, quality.
    python3 scripts/cmp_refs.py out/a.json out/b.json ..."""
import collections
import json
import sys

runs = [json.load(open(p)) for p in sys.argv[1:]]
base = runs[0]
print("forwards", [r["forwards"] for r in runs], "nan rows", [r["nan_rows"] for r in runs],
      "inf rows", [sum(x["inf"] for x in r["rows"]) for r in runs], "init_s", [round(r["init_s"], 1) for r in runs])
phases = collections.OrderedDict()
for i, row in enumerate(base["rows"]):
    phases.setdefault(row["phase"], []).append(i)
for ph, idx in phases.items():
    eq = [all(r["rows"][i]["digest"] == base["rows"][i]["digest"] and r["rows"][i]["rows"] == base["rows"][i]["rows"]
              for i in idx) for r in runs[1:]]
    toks = sorted(set(base["rows"][i]["tokens"] for i in idx))
    print(f"{ph:18s} forwards {len(idx):3d} tokens {toks} identical to run 1: {eq}")
print("sampled tokens identical:", [r["tokens"] == base["tokens"] for r in runs[1:]])
q = [r.get("quality") for r in runs]
if all(q):
    print("generations identical:", [x["generations"] == q[0]["generations"] for x in q[1:]])
    for k in range(len(q[0]["ppl"])):
        print("ppl", q[0]["ppl"][k]["file"], [round(x["ppl"][k]["ppl"], 4) for x in q], "nll", [round(x["ppl"][k]["nll"], 6) for x in q])
