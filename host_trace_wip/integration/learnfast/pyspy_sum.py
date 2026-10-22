"""pyspy_sum.py SPEEDSCOPE.json: inclusive / self sample shares of the main thread, overall and within the trace phase (stacks
holding diag27's _trace wrapper), for the functions that matter to the core agent."""
import collections
import json
import sys

d = json.load(open(sys.argv[1]))
frames = d["shared"]["frames"]
name = [f"{f['name']} ({f.get('file', '').rsplit('/', 1)[-1]}:{f.get('line', '')})" for f in frames]
prof = max(d["profiles"], key=lambda p: len(p["samples"]))  # the main thread
tot = sum(prof["weights"])
incl, selfc, tr_incl, tr_self, tr_tot = collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter(), 0
for stack, w in zip(prof["samples"], prof["weights"]):
    names = [frames[i]["name"] for i in stack]
    seen = set()
    for i in stack:
        if i not in seen:
            incl[i] += w
            seen.add(i)
    if stack:
        selfc[stack[-1]] += w
    if "_trace" in names and any(frames[i].get("file", "").endswith("diag27.py") and frames[i]["name"] == "_trace" for i in stack):
        tr_tot += w
        for i in seen:
            tr_incl[i] += w
        if stack:
            tr_self[stack[-1]] += w
print(f"samples {tot}; in trace {tr_tot} ({100 * tr_tot / max(tot, 1):.1f}%)")
print("--- trace phase, inclusive % of trace samples (top 45)")
for i, c in tr_incl.most_common(45):
    print(f"{100 * c / max(tr_tot, 1):6.1f}%  {name[i][:150]}")
print("--- trace phase, self % (top 25)")
for i, c in tr_self.most_common(25):
    print(f"{100 * c / max(tr_tot, 1):6.1f}%  {name[i][:150]}")
