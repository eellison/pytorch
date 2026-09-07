import json, collections, sys
path = sys.argv[1] if len(sys.argv) > 1 else "agent_space/peak_cmp/results.jsonl"
rows = [json.loads(l) for l in open(path) if l.strip()]
by = collections.defaultdict(dict)
for r in rows:
    by[(r["workload"], r["M"], r["K"], r.get("N"))][(r["impl"], r["mode"])] = r
def us(r): return f"{r['median_us']:7.2f}" if r and "median_us" in r else "      -"
order = ["rmsnorm", "rmsnorm_nvfp4", "rmsnorm_mxfp4", "rmsnorm_mxfp8", "quant_nvfp4", "quant_mxfp4", "quant_mxfp8", "gemm_nvfp4", "gemm_mxfp8"]
for wl in order:
    keys = sorted(k for k in by if k[0] == wl)
    if not keys: continue
    print(f"\n=== {wl} (median us; ours modes: default / cd / persistent / persistent_cd / mk) ===")
    for key in keys:
        cells = by[key]
        ours = {m: cells.get(("ours", m)) for m in ("default", "cd", "persistent", "persistent_cd", "mk")}
        best = min((r for r in ours.values() if r and "median_us" in r), key=lambda r: r["median_us"], default=None)
        theirs = {k[0]: v for k, v in cells.items() if k[0] != "ours"}
        their_best = min((r for r in theirs.values() if "median_us" in r), key=lambda r: r["median_us"], default=None)
        shape = f"{key[1]}x{key[2]}" + (f"x{key[3]}" if key[3] else "")
        ratio = f"{best['median_us']/their_best['median_us']:.2f}x" if best and their_best else "-"
        line = f"{shape:16s} ours: " + " ".join(f"{m[:4]}={us(r).strip()}" for m, r in ours.items() if r) + f" | best={us(best).strip()} ({best['mode']}, k={best.get('kernels')})" if best else f"{shape:16s} ours: -"
        line += " || " + " ".join(f"{k}={us(r).strip()}" for k, r in theirs.items()) + f" | ours_best/their_best={ratio}"
        print(line)
        errs = {k: r["error"][:100] for k, r in cells.items() if "error" in r}
        if errs: print("      ERRORS:", errs)
        if best and "quant_mismatch_frac_vs_fi" in best:
            print(f"      ours vs fi bytes: quant mismatch {best['quant_mismatch_frac_vs_fi']:.4%}, scale mismatch {best['scale_mismatch_frac_vs_fi']:.4%}; dequant mean|err| ours={best['dequant_mean_abs_err']:.4g} theirs={their_best.get('dequant_mean_abs_err', float('nan')):.4g}")
        elif best and "dequant_mean_abs_err" in best and their_best:
            print(f"      dequant mean|err| ours={best['dequant_mean_abs_err']:.4g} theirs={their_best.get('dequant_mean_abs_err', float('nan')):.4g}")
        if best and best.get("kernels", 1) != 1:
            print(f"      NOTE: ours best uses {best['kernels']} kernels: {best.get('kernel_names')}")
