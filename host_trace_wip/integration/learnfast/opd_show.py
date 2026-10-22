"""opd_show.py DIAG.jsonl: opdeps summaries of every trace (top-level and nested), merged per func over the run."""
import collections, json, sys
L = [json.loads(l) for l in open(sys.argv[1])]
for ev in ("opdeps", "opdeps_nested"):
    tot = collections.defaultdict(collections.Counter)
    names = collections.defaultdict(lambda: collections.defaultdict(set))
    for d in L:
        if d.get("ev") == ev:
            for f, v in d["summary"].items():
                for k, c in v.items():
                    if k != "names":
                        tot[f][k] += c
                for k, ns in v.get("names", {}).items():
                    names[f][k] |= set(ns)
    print("==", ev, "traces", sum(1 for d in L if d.get("ev") == ev))
    for f, c in sorted(tot.items(), key=lambda x: -sum(v for k, v in x[1].items() if k in ("args_only", "args_or_declared", "reads_outside", "opaque_records"))):
        if f.startswith("<triton.runtime.autotuner") or (ev == "opdeps_nested" and c.get("with_guards", 0) == 0):
            continue
        print(f"  {f[:58]:58s} {dict(c)} {dict((k, sorted(v)[:4]) for k, v in names[f].items())}")
for d in L:
    if d.get("ev") == "opdeps_error":
        print("ERROR", d["err"])
