# Per-point table of ix_run.py tags: python ix/ix_table.py TAG [TAG ...] [--sku SKU ...]. One row per point, columns in
# the morning-report order: counts first, then throughput / latency, then memory.
#   steps            execute_model calls over the client run (InferenceX's 2x conc warm-ups included)
#   new keys / learns  exact keys first seen in the run / out-of-band harvests (ours)
#   learn s (share)  host seconds in out-of-band learns, and their share of the client's wall time ("learn-dominated" > 25%)
#   traces / eager   new traces, eager-path steps (entry first calls + declined/eager forms) in the run (ours)
#   redispatches     op-owned re-dispatches in the run and their host seconds (ours; e.g. the trtllm context op at a new max_kv)
#   graphed          ours: replayed steps / steps; default: FULL / PIECEWISE / NONE step fractions
#   bnd              eager boundaries in our variants (static)
#   out-of-graph     kernel launches outside graphs per step (profiler window, separate 2x conc burst after the point)
#   tok/s, TPOT p50/p90, TTFT p90 from InferenceX's client result; KV tokens from vLLM's startup line.
import glob
import json
import os
import re
import sys

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out", "ix")
args = sys.argv[1:]
skus = set(args[args.index("--sku") + 1:]) if "--sku" in args else set()
tags = args[:args.index("--sku")] if "--sku" in args else args
rows = []
for tag in tags:
    for p in sorted(glob.glob(os.path.join(OUT, tag, "*_point.json"))):
        d = json.load(open(p))
        if skus and d["sku"] not in skus:
            continue
        r = json.load(open(p.replace("_point.json", ".json"))) if os.path.exists(p.replace("_point.json", ".json")) else {}
        b0, b1 = d["breakdown_before"] or {}, d["breakdown_after"] or {}
        delta = lambda k: (b1.get(k) or 0) - (b0.get(k) or 0)
        steps = max(delta("steps"), 1)
        ours = "replays" in b1
        modes = {m: (b1.get("cg_modes", {}).get(m, 0) - b0.get("cg_modes", {}).get(m, 0)) / steps for m in ("FULL", "PIECEWISE", "NONE")}
        eager = (delta("entry_eager") + sum(b1.get("eager_path_calls", {}).values()) - sum(b0.get("eager_path_calls", {}).values())) if ours else None
        log = p.replace("_point.json", "_server.log")
        kv = None
        if os.path.exists(log):
            for line in open(log, errors="replace"):
                m = re.search(r"GPU KV cache size: ([\d,]+) tokens", line)  # bench reporting only
                if m:
                    kv = int(m.group(1).replace(",", ""))
        learn_s = delta("learn_s")
        prof = (d.get("profile") or {}).get("per_step", {})
        rows.append(dict(sku=d["sku"], shape=d["scenario"], conc=d["conc"], rc=d["rc"], steps=steps,
                         keys=delta("exact_keys") if ours else None, learns=delta("learns") if ours else None,
                         learn_s=learn_s if ours else None, share=learn_s / d["client_wall_s"] if ours and d["client_wall_s"] else None,
                         traces=delta("traces") if ours else None, eager=eager,
                         redisp=f"{delta('redispatches')} ({delta('redispatch_s'):.1f} s)" if "redispatches" in b1 else None,
                         graphed=f"{delta('replays') / steps:.2f}" if ours else " / ".join(f"{modes[k]:.2f}" for k in ("FULL", "PIECEWISE", "NONE")),
                         bnd=b1.get("boundaries_static"), og=prof.get("kernel_launches_outside_graphs"),
                         tps=r.get("output_throughput"), tpot50=r.get("median_tpot_ms"), tpot90=r.get("p90_tpot_ms"), ttft90=r.get("p90_ttft_ms"),
                         kv=kv, ready=d["server"]["ready_s"], warm=d["server"].get("warm_s")))
f = lambda x, n=2: "-" if x is None else f"{x:.{n}f}" if isinstance(x, float) else str(x)
print("| sku | shape | conc | steps | new keys | learns | learn s (share) | traces | eager steps | redispatches (s) | graphed (ours: replayed; default: FULL/PW/NONE) | bnd | out-of-graph launches/step | tok/s | TPOT p50 / p90 ms | TTFT p90 ms | KV tokens | ready s (+warm s) | rc |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for x in sorted(rows, key=lambda x: (x["shape"], x["conc"], x["sku"])):
    share = "" if x["share"] is None else f" ({100 * x['share']:.0f}%{', learn-dominated' if x['share'] > 0.25 else ''})"
    print(f"| {x['sku']} | {x['shape']} | {x['conc']} | {x['steps']} | {f(x['keys'])} | {f(x['learns'])} | {f(x['learn_s'], 1)}{share} | {f(x['traces'])} | {f(x['eager'])} | {f(x['redisp'])} | "
          f"{x['graphed']} | {f(x['bnd'])} | {f(x['og'], 1)} | {f(x['tps'], 1)} | {f(x['tpot50'])} / {f(x['tpot90'])} | {f(x['ttft90'], 1)} | {f(x['kv'])} | "
          f"{f(x['ready'], 0)}{'' if x['warm'] is None else ' (+' + f(x['warm'], 0) + ')'} | {x['rc']} |")
