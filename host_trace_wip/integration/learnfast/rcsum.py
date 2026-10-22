"""rcsum.py DIAG.jsonl [LOG]: counts of a diag27 run (traces, redispatch/redo-context counts, sections, errors, end timers)."""
import collections, json, re, sys
L = [json.loads(l) for l in open(sys.argv[1])]
c = collections.Counter(d.get("ev") for d in L)
print("events", dict(c))
tr = [d for d in L if d.get("ev") == "trace"]
print("traces", len(tr), "trace_s", round(sum(d["s"] for d in tr), 1), "toks", [d["toks"][-1] for d in tr])
end = [d for d in L if d.get("ev") == "end"]
if end:
    print("end acc", end[-1]["acc"], "cnt", end[-1]["cnt"], "t", end[-1]["t"])
causes = collections.Counter(str(x)[:120] for d in L if d.get("ev") == "miss" for x in d.get("causes", []))
print("miss causes", causes.most_common(8))
for d in [d for d in L if d.get("ev") == "redo_context_error"][:3]:
    print("REDO_CONTEXT_ERROR", d["err"], d["tb"][-600:])
op = [d for d in L if d.get("ev") == "opspan_mem" and "peak_base" in d]
if op:
    print("opspan peak max base/opspan", max(d["peak_base"] for d in op), max(d["peak_opspan"] for d in op))
if len(sys.argv) > 2:
    s = open(sys.argv[2], errors="replace").read()
    for pat in ("illegal memory access", "CUDA error", "Traceback", "rc=-?\\d+", "lib_redo_refused[^}]*}", "'traces': \\d+", "'redispatches': \\d+", "'sections_added': \\d+"):
        m = re.findall(pat, s)
        print(pat, len(m), m[-1][:200] if m else "")
