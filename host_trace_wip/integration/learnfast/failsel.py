"""failsel.py DIAG.jsonl: per trace, every failing selector (by op) of the redispatch attempts at that call."""
import json, sys
L = [json.loads(l) for l in open(sys.argv[1])]
pending = []
for d in L:
    ev = d.get("ev")
    if ev in ("redispatch_refused", "redispatch_ok"):
        pending.append(d)
    elif ev == "miss":
        if d.get("traces_added"):
            tok = d["toks"][-1] if d["toks"] else None
            graph = sum(1 for c in d["causes"] if c.get("class") == "graph")
            atts = [p for p in pending if (p["toks"][-1] if p["toks"] else None) == tok]
            ops = {}
            for p in atts:
                for k, v in (p.get("unselected_ops") or {}).items():
                    k = k.replace("vllm.", "").replace(".default", "")
                    ops[k] = max(ops.get(k, 0), v)
            refused = sorted({p["op"].replace("vllm.", "").replace(".default", "") for p in atts if p["ev"] == "redispatch_refused"})
            print(f"trace at {tok}: variants failing on graph guards {graph}; dispatch attempts {len(atts)}; failing selectors {ops}; refused by {refused}")
        pending = []
