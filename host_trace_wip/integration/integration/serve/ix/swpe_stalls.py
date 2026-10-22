# swpe stall attribution (main 10-15): per arm / run / context, the decode-curve points whose tail is long, with what was inside the slow
# steps (GC generation and ms from gc.callbacks, host memory pressure); flagged points (PSI > 2% of the point or > 10x-median-step excess
# > 1% of the measured steps).   python3 serve/ix/swpe_stalls.py SET
import glob, json, os, sys

SET = sys.argv[1] if len(sys.argv) > 1 else "swpe1"
print("| arm | run | ctx | bs | TPOT mean ms | median step ms | p99 ms | max ms | steps > 3x (excess ms) | steps > 10x | GC in the slow steps | GC passes > 1 ms in the window | PSI s | flagged |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
tot = {}
for f in sorted(glob.glob(f"vllm/out/t_{SET}_*_r[0-9].json")):
    arm, r = os.path.basename(f)[len(f"t_{SET}_"):-5].rsplit("_r", 1)
    R = json.load(open(f))
    for ctx, pts in R.get("dec_ctx", {}).items():
        for bs, d in pts.items():
            t = d.get("tail")
            if not t:
                continue
            k = (arm, r)
            tot.setdefault(k, [0, 0, 0])
            tot[k][0] += 1
            tot[k][1] += t["flagged"]
            tot[k][2] += t.get("steps_over_3x", 0) > 0
            if t["flagged"] or t.get("steps_over_3x", 0) or t["max_ms"] > 2 * t["median_ms"]:
                gcs = [o["gc"] for o in t.get("outliers", []) if o.get("gc")]
                print(f"| {arm} | r{r} | {ctx} | {bs} | {d['tpot_ms_mean']:.2f} | {t['median_ms']:.2f} | {t['p99_ms']:.2f} | {t['max_ms']:.1f} | {t.get('steps_over_3x', 0)} ({t.get('excess_over_3x_ms', 0):.0f}) | "
                      f"{t['outlier_steps']} | {gcs or '-'} | {t.get('gc_in_window') or '-'} | {t['host_mem_pressure_full_s'] if t['host_mem_pressure_full_s'] is not None else '-'} | {t['flagged']} |")
print("\n| arm | run | points | flagged points | points with a step > 3x the median |\n|---|---|---|---|---|")
for (a, r), v in sorted(tot.items()):
    print(f"| {a} | r{r} | {v[0]} | {v[1]} | {v[2]} |")
