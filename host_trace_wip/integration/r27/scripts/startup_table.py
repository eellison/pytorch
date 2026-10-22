"""startup_table.py SET [WARMUP_SET]: engine start-up per arm from startup_parse.py (cold-cache round 0 and warm rounds 1-3).
vLLM's graph warm-up = its capture seconds. Ours: ready (engine up) and graph warm-up = trace seconds + learns + bindings load,
summed over the run (lazy first-call traces), taken from WARMUP_SET (a run with the trace timer: n27wfs for n27wf)."""
import glob, os, re, statistics as st, subprocess, sys

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SET, WSET = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else None)
logs = sorted(glob.glob(os.path.join(R, "logs", "timing", f"t_{SET}_*_r*.log")) + (glob.glob(os.path.join(R, "logs", "timing", f"t_{WSET}_*_r*.log")) if WSET else []))
out = subprocess.run([sys.executable, os.path.join(R, "scripts", "startup_parse.py"), *logs], capture_output=True, text=True).stdout
rows = {}
for line in out.splitlines():
    name, d = line.split(" ", 1)
    m = re.match(r"t_(\w+?)_(default|fullnc|breakable|vcvp|vsvp)(cold)?_r(\d+)\.log", name)
    if m:
        rows.setdefault((m.group(2), m.group(1), "cold" if m.group(3) else "warm"), []).append(eval(d))
f = lambda xs, k: (f"{st.mean(v):7.1f}" if (v := [x[k] for x in xs if x.get(k) is not None]) else f"{'-':>7s}")
print(f"{SET} start-up (seconds; warm = mean of rounds 1-3, cold = round 0 with fresh vLLM / Inductor caches and no cuBLAS bindings)")
print(f"{'arm':10s} {'cache':5s} {'ready':>7s} {'compile':>7s} {'FI tune':>7s} {'capture':>7s} {'graphs':>7s} {'graphGiB':>8s} {'KV tok':>9s}  graph warm-up")
for arm in ("default", "fullnc", "breakable", "vcvp", "vsvp"):
    for cache in ("cold", "warm"):
        xs = rows.get((arm, SET, cache), [])
        if not xs:
            continue
        if arm.startswith("v"):
            w = rows.get((arm, WSET, "warm"), []) if WSET else []
            gw = f"{f(w, 'ours_graph_warmup_s').strip()} s ({w[0].get('ours_traces')} traces, learns {f(w, 'ours_learn_s').strip()} s, bindings load {f(w, 'ours_bindings_load_s').strip()} s; from {WSET})" if w and cache == "warm" else "-"
        else:
            gw = f"{f(xs, 'vllm_graph_warmup_s').strip()} s (capture)"
        print(f"{arm:10s} {cache:5s} {f(xs, 'ready_s')} {f(xs, 'torch_compile_s')} {f(xs, 'fi_autotune_s')} {f(xs, 'graph_capture_s')} {f(xs, 'graphs')} "
              f"{f(xs, 'graph_GiB'):>8s} {f(xs, 'kv_tokens'):>9s}  {gw}")
