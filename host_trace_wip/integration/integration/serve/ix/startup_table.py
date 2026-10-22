# Fair startup table (main 10-14): engine ready, TTFT of the first 8 requests, ready -> steady (last stream step with a trace or a cuBLAS
# learn; first-use binds of stored keys excluded), per arm and run; ours' minimal warm-up v1 (decode / prefill / mixed traces) vs v2
# (+ an 8-request batched prefill and bs-8 decodes); the one-time offline cost (seed run + learnfast learn_range) as its own row.
#   python3 serve/ix/startup_table.py
import glob, json, os, re, statistics as st

rows = []
for f in sorted(glob.glob("vllm/out/t_startup_*_r[0-9].json")):
    arm, r = re.match(r".*t_startup_(\w+?)_r(\d)\.json", f).groups()
    if arm == "seed":
        continue
    R = json.load(open(f))
    S = R.get("startup", {})
    tl = [e for e in S.get("event_steps", []) if e[2][0] > 0 or e[2][1] > 0]
    steady = tl[-1][1] if tl else 0.0
    warm = ("v3 (v2 + multi-prefill mixed)" if S.get("min_warm_v3") else ("v2 (+ burst of %d)" % S["min_warm_burst"])) if S.get("min_warm_burst") else ("v1" if arm in ("vcsl", "vrs") else "-")
    tt = S.get("ttft_first8_ms") or []
    rows.append((arm, r, warm, S.get("engine_ready_s"), S.get("init_s"), S.get("min_warm_s"), st.median(tt) if tt else None, max(tt) if tt else None, steady,
                 (S.get("engine_ready_s") or 0) + steady, S.get("events_total"), S.get("host_mem_pressure_full_s"),
                 S.get("outlier_steps") if "outlier_excess_s" in S else "-",  # older runs counted prefill steps too: not shown
                 # flag on PSI only: the stream's long steps are ours' traces (part of the result), not host noise
                 ((S.get("host_mem_pressure_full_s") or 0) > 0.02 * S.get("stream_wall_s", 1e9))))
print("| arm | run | ours' minimal warm-up | engine ready s | init s | warm-up s | TTFT first 8 median / max ms | ready -> steady s | ready-to-steady s | stream traces / learns / binds | PSI s | decode outliers | flagged |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
f2 = lambda x: f"{x:.1f}" if isinstance(x, (int, float)) else "-"
for a in rows:
    print(f"| {a[0]} | r{a[1]} | {a[2]} | {f2(a[3])} | {f2(a[4])} | {f2(a[5])} | {f2(a[6])} / {f2(a[7])} | {f2(a[8])} | {f2(a[9])} | {a[10]} | {f2(a[11])} | {a[12]} | {a[13]} |")
print("\n| one-time offline cost (ours) | vcsl | vrs |\n|---|---|---|")
print("| seed run (minimal warm-up + 20 stream steps, keys saved) | 48.1 s | 48.7 s |")
print("| learnfast learn_range (token counts 1-16384, sampled-row GEMMs 1-128; 8 processes, GPU 0, fresh processes) | 384.6 s | 381.9 s |")
print("| bindings saved and loaded | 81,920 | 81,920 |")
