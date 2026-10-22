# Continuous-batching churn tables (10-13): vllm/out/t_<SET>_<arm>_r*.json (drive_sweep.py --cb). Headline: ours with warm GEMMs
# (vcsl: cuBLAS bindings preloaded) vs the faster vLLM config overall (geomean end-to-end ms per token over C); then every arm with
# churn counters; the cold-learn arm (vcs) as a diagnostic; startup accounting.   python3 serve/ix/cb_table.py [SET=cbh] [GPU label]
import glob, json, math, os, statistics as st, sys

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SET = sys.argv[1] if len(sys.argv) > 1 else "cbh"
GPU = sys.argv[2] if len(sys.argv) > 2 else "0"
NAMES = {"default": "vLLM default", "fullnc": "vLLM default-nc", "bnc": "vLLM breakable graphs", "vcsl": "ours compiled, warm GEMMs (preloaded cuBLAS)",
         "vrs": "ours uncompiled (V-registered + sampling)", "vcs": "ours compiled, cold cuBLAS learns (diagnostic)",
         "vcsl_h": "ours compiled on the new head (ms3)", "vrs_h": "ours uncompiled on the new head (ms3)",
         "vcsl_fh": "ours compiled, new head + form_held", "vrs_fh": "ours uncompiled, new head + form_held",
         "vcsl_h7": "ours compiled, head ms7 (split off)", "vcsl_h7s": "ours compiled, head ms7 (split 64)"}
GROUPS = [("compiled forward: vLLM graphs vs ours", ["default"], "vcsl"), ("eager forward: vLLM graphs vs ours", ["fullnc", "bnc"], "vrs"),
          ("compiled forward: vLLM graphs vs ours on the new head", ["default"], "vcsl_h"), ("eager forward: vLLM graphs vs ours on the new head", ["fullnc", "bnc"], "vrs_h"),
          ("compiled forward: vLLM graphs vs ours (new head + form_held)", ["default"], "vcsl_fh"), ("eager forward: vLLM graphs vs ours (new head + form_held)", ["fullnc", "bnc"], "vrs_fh"),
          ("compiled forward: vLLM graphs vs ours (head ms7, split off)", ["default"], "vcsl_h7"), ("compiled forward: vLLM graphs vs ours (head ms7, split 64)", ["default"], "vcsl_h7s")]
runs = {a: [json.load(open(f)) for f in sorted(glob.glob(f"{HERE}/vllm/out/t_{SET}_{a}_r[0-9].json"))] for a in NAMES}
runs = {a: v for a, v in runs.items() if v}
Cs = sorted({int(c) for v in runs.values() for R in v for c in R.get("cb", {})})


def med(a, C, k):
    v = [R["cb"][str(C)][k] for R in runs.get(a, []) if str(C) in R.get("cb", {}) and R["cb"][str(C)].get(k) is not None]
    return (st.median(v), min(v), max(v)) if v else None


def sub(a, C, part, k):
    v = [R["cb"][str(C)][part][k] for R in runs.get(a, []) if str(C) in R.get("cb", {}) and R["cb"][str(C)].get(part)]
    return st.median(v) if v else None


geo = lambda xs: math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float("nan")
f = lambda c, n=3: (f"{c[0]:.{n}f} [{c[1]:.{n}f}-{c[2]:.{n}f}]" if isinstance(c, tuple) else f"{c:.{n}f}") if c is not None else "-"
L = [f"## Continuous batching, Qwen3-8B, c5h, GPU {GPU} (all arms on this GPU)\n",
     "C requests in flight, outputs 256-2048 and prompts 64-512 (seeded, the same request sequence in every arm), finished requests replaced at once; "
     "400 warm-up steps, 3000 measured. End-to-end ms per token = wall time of the measured engine steps / tokens produced. Runs per arm: "
     + ", ".join(f"{a} {len(v)}" for a, v in runs.items()) + ".\n"]
for g, bl, o in GROUPS:  # ratios within a forward only (same kernels): graphing methods compared, never across compile settings
    bl = [x for x in bl if x in runs]
    if o not in runs or not bl:
        continue
    L.append(f"### {g}\n")
    L.append("| C | " + " | ".join(f"{x} ms/token" for x in bl) + f" | ours ({o}) ms/token | " + " | ".join(f"{x} / ours" for x in bl) + " |")
    L.append("|---|" + "---|" * (2 * len(bl) + 1))
    rs = {x: [] for x in bl}
    for C in Cs:
        o_ = med(o, C, "wall_ms_per_token")
        bs_ = [med(x, C, "wall_ms_per_token") for x in bl]
        for x, b_ in zip(bl, bs_):
            if b_ and o_:
                rs[x].append(b_[0] / o_[0])
        L.append(f"| {C} | " + " | ".join(f(b_) for b_ in bs_) + f" | {f(o_)} | " + " | ".join(f"{b_[0] / o_[0]:.2f}" if b_ and o_ else "-" for b_ in bs_) + " |")
    L.append("| geomean | " + " | ".join([""] * (len(bl) + 1)) + " | " + " | ".join(f"{geo(rs[x]):.2f}" for x in bl) + " |\n")
L.append("### Every arm\n")
L.append("| C | arm | ms/token (wall) | ms/token (step sum) | decode step ms (median) | mixed step ms (median) | mixed steps | shape changed | decode bs changed | vLLM pad rows/step | nodes set/step | patch us/step | learns in window | traces in window |")
L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for C in Cs:
    for a in runs:
        g = lambda k: med(a, C, k)
        L.append(f"| {C} | {a} | {f(g('wall_ms_per_token'))} | {f(g('step_ms_per_token'))} | {f(sub(a, C, 'decode', 'step_ms'), 2)} | {f(sub(a, C, 'mixed', 'step_ms'), 2)} | "
                 f"{f(g('mixed_steps'), 0)} | {f(g('shape_changed_frac'))} | {f(g('decode_bs_changed_frac'))} | {f(g('pad_tokens_mean'), 2)} | {f(g('nodes_set_mean'), 1)} | "
                 f"{f(g('patch_us_mean'), 0)} | {f(g('learns_in_window'), 0)} | {f(g('traces_in_window'), 0)} |")
if any(R["cb"][str(C)].get("ns_split") for v in runs.values() for R in v for C in Cs if str(C) in R.get("cb", {})):
    L.append("\n### Unsynced steps by size change (async scheduling as shipped; per step: execute_model entry to the next one)\n")
    L.append("Interval = pipeline period per step (host and GPU overlapped); size changed = the step's (requests, tokens) differ from the previous step's. "
             "Ours re-freezes the GC heap after post-warm-up traces (ARMV_REFREEZE) from cbh4 on. Medians over runs.\n")
    L.append("| C | arm | decode same: n, interval ms | decode size changed: n, interval ms | changed - same ms | mixed: n, interval ms | execute_model host ms (decode same / changed / mixed) | split calls per step (same / changed / mixed) |")
    L.append("|---|---|---|---|---|---|---|---|")
    def nsm(a, C, k, f_):
        v = [R["cb"][str(C)]["ns_split"][k][f_] for R in runs.get(a, []) if str(C) in R.get("cb", {}) and f_ in ((R["cb"][str(C)].get("ns_split") or {}).get(k) or {})]
        return st.median(v) if v else None
    for C in Cs:
        for a in runs:
            ds, dc = nsm(a, C, "decode_same", "interval_ms_median"), nsm(a, C, "decode_changed", "interval_ms_median")
            L.append(f"| {C} | {a} | {f(nsm(a, C, 'decode_same', 'n'), 0)}, {f(ds, 2)} | {f(nsm(a, C, 'decode_changed', 'n'), 0)}, {f(dc, 2)} | "
                     f"{f(dc - ds, 2) if ds is not None and dc is not None else '-'} | {f(nsm(a, C, 'mixed_changed', 'n'), 0)}, {f(nsm(a, C, 'mixed_changed', 'interval_ms_median'), 2)} | "
                     f"{f(nsm(a, C, 'decode_same', 'exec_host_ms_median'), 2)} / {f(nsm(a, C, 'decode_changed', 'exec_host_ms_median'), 2)} / {f(nsm(a, C, 'mixed_changed', 'exec_host_ms_median'), 2)} | "
                     f"{f(nsm(a, C, 'decode_same', 'split_calls_mean'), 2)} / {f(nsm(a, C, 'decode_changed', 'split_calls_mean'), 2)} / {f(nsm(a, C, 'mixed_changed', 'split_calls_mean'), 2)} |")
L.append("\n### Background on each run's cores (own-core gate 0.30 on gb300-box from cbh4b; foreign busy fraction > 0.15 flagged)\n")
L.append("Host-noise flag (unified, 10-14): a window with host memory pressure (PSI full, system-wide) > 2% of its wall (1% before 21:14), or decode steps > 10x the window median interval whose excess is > 1% of the wall; flagged runs are rerun at the end of the set.\n")
L.append("| arm | run | foreign busy fraction | busy cores on own range (mean) | own process cores | PSI full s (C = 16 / 64 / 128) | outlier steps (C = 16 / 64 / 128) | reserved GiB before / after empty_cache |")
L.append("|---|---|---|---|---|---|---|---|")
for a in runs:
    for i, R in enumerate(runs[a]):
        pl = R.get("placement", {})
        fb = pl.get("foreign_busy_frac")
        psi = " / ".join(f"{R['cb'][str(C)].get('host_mem_pressure_full_s'):.2f}" if R["cb"].get(str(C), {}).get("host_mem_pressure_full_s") is not None else "-" for C in Cs)
        out = " / ".join(str((R["cb"].get(str(C), {}).get("host_flag") or {}).get("outlier_steps", "-")) for C in Cs)
        ec = R.get("empty_cache_gib")
        L.append(f"| {a} | r{i + 1} | {fb if fb is not None else '-'}{' (busy)' if fb is not None and fb > 0.15 else ''} | {pl.get('busy_cores_own_range_mean', '-')} | {pl.get('own_process_cores_mean', '-')} | {psi} | {out} | "
                 + (f"{ec[0]:.1f} / {ec[1]:.1f}" if ec else "-") + " |")
_notes = f"{HERE}/serve/out/{SET}_flag_notes.md"
if os.path.exists(_notes):
    L.append("\nFlag-rule history:" + open(_notes).read())
L.append("\n### Startup accounting\n")
L.append("| arm | engine init s (median) | bindings loaded | bindings load s |")
L.append("|---|---|---|---|")
for a in runs:
    ini = st.median(R["init_s"] for R in runs[a])
    bl = [R.get("V", {}).get("bindings_loaded") for R in runs[a]]
    bs = [R.get("V", {}).get("bindings_load_s") for R in runs[a] if R.get("V", {}).get("bindings_load_s") is not None]
    L.append(f"| {a} | {ini:.1f} | {bl[0] if bl and bl[0] is not None else '-'} | {st.median(bs):.2f} |" if bs else f"| {a} | {ini:.1f} | - | - |")
lf = f"{HERE}/vllm/out/t_{SET}_learn.json"
if os.path.exists(lf):
    Lr = json.load(open(lf))
    L.append(f"\nOffline learn pass (once, same build and config): {Lr.get('total_wall_s', float('nan')):.0f} s wall, {Lr.get('V', {}).get('bindings_saved')} bindings saved "
             f"({Lr.get('V', {}).get('harvests')} cuBLAS keys learned); file vllm/out/bindings_{SET}.pkl.")
open(f"{HERE}/serve/out/{SET}_table.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))

# chart (house style): end-to-end ms per token per C, the headline arms; labels: churn and per-step counters
if os.environ.get("CB_CHART", "1") == "1":
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(HERE), "talk"))
        import charts as Ch
    except ImportError:
        Ch = None
    if Ch is not None and Cs:
        import textwrap
        plt = Ch.plt
        arms = [("default", "vLLM default graphs (compiled)", Ch.DEFAULT), ("vcsl", "ours (compiled)", Ch.OURS), ("fullnc", "vLLM FULL decode graphs (eager)", Ch.DEFNC),
                ("bnc", "vLLM breakable graphs (eager)", Ch.VIOLET), ("vrs", "ours (eager)", "#7fb2ea"),
                ("vcsl_h", "ours (compiled), new head", "#174a8c"), ("vrs_h", "ours (eager), new head", "#a9cdf2"),
                ("vcsl_fh", "ours (compiled), new head + form_held", "#174a8c"), ("vrs_fh", "ours (eager), new head + form_held", "#a9cdf2"),
                ("vcsl_h7", "ours (compiled), head ms7", "#2a78d6"), ("vcsl_h7s", "ours (compiled), head ms7 + split", "#174a8c")]
        arms = [x for x in arms if x[0] in runs]
        pl = {(R.get("placement", {}).get("cuda_visible_devices"), R.get("placement", {}).get("cpus")) for v in runs.values() for R in v if R.get("placement")}
        where = "; ".join(f"GPU {g}, CPUs {c}" for g, c in sorted(pl, key=str)) or f"GPU {GPU}"
        fig, ax = Ch.slide("Serving-style churn: requests join and finish continuously", "End-to-end time per output token, C requests in flight",
                           f"Qwen3-8B bf16, GB300 ({where}; every arm on this GPU), vLLM 0.29 (V2 runner, FlashInfer), host-trace c5h. C requests in flight, output lengths 256-2048 and "
                           "prompts 64-512 (seeded; the same request sequence in every arm), a finished request replaced at once (its prefill lands as a mixed step). Untimed warm-up in every arm "
                           "(one prefill per token count, a short decode per batch size, three mixed steps; vLLM also captures its graphs), then 400 warm-up and 3,000 measured engine steps per C. "
                           "Compare bars within a forward only (compiled: vLLM default vs ours; eager: vLLM default-nc and breakable vs ours), the same kernels under different graphing. "
                           "Ours: 0 traces and 0 learns inside every measured window (enforced); compiled: cuBLAS bindings preloaded. Bars: median of 3 runs; whiskers: min-max. "
                           + (("Unsynced: no syncs in the measured windows (vLLM's async scheduling as shipped). Ours re-freezes the GC heap after post-warm-up traces; "
                               "ours' warm-up also covers short-chunk mixed steps (the variants our replay needs; vLLM captures at init). Runs gated on host memory pressure; "
                               "windows with host-noise flags rerun (see the table's background section). "
                               + ("Every arm: torch.cuda.empty_cache() once after the warm-up. " if any(R.get("empty_cache_gib") for v in runs.values() for R in v) else "")
                               + ("Ours (form_held): the runtime keeps each form's held node state. " if any(R.get("form_held") for v in runs.values() for R in v) else ""))
                              if any(R["cb"][str(C)].get("synced_steps") is False for v in runs.values() for R in v for C in Cs if str(C) in R.get("cb", {})) else "") + "|| "
                           "Runs per arm: " + ", ".join(f"{a} {len(v)}" for a, v in runs.items() if a in [x[0] for x in arms]) + ".")
        w = 0.8 / len(arms)
        for j, (a, lab, col) in enumerate(arms):
            cs = [med(a, C, "wall_ms_per_token") for C in Cs]
            xs = [i - 0.4 + (j + 0.5) * w for i in range(len(Cs))]
            ax.bar(xs, [c[0] * 1e3 for c in cs], w, color=col, label=lab, yerr=[[ (c[0] - c[1]) * 1e3 for c in cs], [(c[2] - c[0]) * 1e3 for c in cs]], capsize=3, ecolor=Ch.INK2)
            for x, c in zip(xs, cs):
                ax.text(x, c[2] * 1e3 + 6, f"{c[0] * 1e3:.0f}", ha="center", fontsize=10, color=Ch.INK2)
        ax.set_xticks(range(len(Cs)))
        ax.set_xticklabels([f"C = {C}\n{med(arms[0][0], C, 'mixed_steps')[0]:.0f} mixed steps / 3000" for C in Cs], fontsize=12)
        ax.set_ylabel("us per output token (end to end)")
        ax.legend(frameon=False, fontsize=11)
        Ch.hgrid(ax, "y")
        ft = fig.texts[-1]
        ft.set_text("\n".join(textwrap.fill(p_, 150) for p_ in ft.get_text().replace("\n", " ").split(" || ")))
        fig.subplots_adjust(top=0.80, bottom=0.34)
        out = f"{HERE}/serve/out/charts/fig_{SET}_churn.png"
        fig.savefig(out, dpi=150)
        print("wrote", out)
