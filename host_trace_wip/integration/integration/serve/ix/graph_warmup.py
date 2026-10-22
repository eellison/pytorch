# CUDA-graph warm-up time per arm (cold-start slide): vLLM arms = the sum of vLLM's "Graph capturing finished in X secs" lines (the
# memory-profiling capture at KV sizing plus the real capture) and graph memory; ours = the seconds of the calls that added a trace
# (adapter trace_call_s, recorded from 10-13 ~20:30) and the cuBLAS bindings load. Reads every run log / JSON of the given sets.
#   python3 serve/ix/graph_warmup.py SET [SET ...]
import glob, json, os, re, statistics as st, sys

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
rows = {}
for SET in sys.argv[1:]:
    for log in sorted(glob.glob(f"{HERE}/vllm/logs/t_{SET}_*_r*.log")):
        name = os.path.basename(log)[:-4]
        arm = name[len(f"t_{SET}_"):].rsplit("_r", 1)[0]
        t = open(log, errors="replace").read()
        caps = [(int(a), float(b)) for a, b in re.findall(r"Graph capturing finished in (\d+) secs, took ([\d.]+) GiB", t)]
        js = log.replace("/logs/", "/out/").replace(".log", ".json")
        R = json.load(open(js)) if os.path.exists(js) else {}
        V = R.get("V", {})
        tr = [e.get("trace_call_s") for e in V.get("entries", {}).values() if e.get("trace_call_s") is not None]
        rows.setdefault(arm, []).append({"set": SET, "run": name, "capture_lines_s": sum(c[0] for c in caps), "capture_lines": caps,
                                         "capture_model_s": R.get("capture", {}).get("capture_model_s"), "graph_gib": (R.get("capture", {}).get("graph_bytes") or 0) / 2**30,
                                         "cuda_graphs": R.get("capture", {}).get("capture_begins"), "trace_s": sum(tr) if tr else None,
                                         "traces": V.get("traces"), "bindings_load_s": V.get("bindings_load_s"), "init_s": R.get("init_s")})
f = lambda x, n=1: f"{x:.{n}f}" if isinstance(x, (int, float)) else "-"
L = ["| arm | runs | graph warm-up s (sum of vLLM capture lines; ours: trace calls) | per capture line (s, GiB) | capture_model s | CUDA graphs | graph memory GiB | our traces | bindings load s | engine init s |",
     "|---|---|---|---|---|---|---|---|---|---|"]
for arm, rs in rows.items():
    m = lambda k: st.median([r[k] for r in rs if r[k] is not None]) if any(r[k] is not None for r in rs) else None
    warm = m("capture_lines_s") if any(r["capture_lines"] for r in rs) else m("trace_s")
    L.append(f"| {arm} | {len(rs)} ({', '.join(sorted({r['set'] for r in rs}))}) | {f(warm)} | {rs[0]['capture_lines'] or '-'} | {f(m('capture_model_s'))} | {f(m('cuda_graphs'), 0)} | "
             f"{f(m('graph_gib'), 2)} | {f(m('traces'), 0)} | {f(m('bindings_load_s'), 2)} | {f(m('init_s'))} |")
print("\n".join(L))
