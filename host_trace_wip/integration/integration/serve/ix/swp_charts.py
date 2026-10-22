# Slide sweeps (10-13): decode every bs 1-128, prefill 64-16000, mixed chunk x decodes, alternating decode; ours (Vc + sampling)
# vs vLLM default and default-nc, full-step metric. House style from land/scratch/talk/charts.py. Writes serve/out/charts/*.png,
# serve/out/swp_tables.md and serve/out/charts/swp_data.json.
#   python3 serve/ix/swp_charts.py [SET=swp]
import glob, json, math, os, statistics as st, sys

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "talk"))
import charts as C  # noqa: E402

plt = C.plt
SET = sys.argv[1] if len(sys.argv) > 1 else "swp"
OUTD = os.path.join(HERE, "serve", "out", "charts")
ARMS = [("default", "vLLM default (compile + graphs)", C.DEFAULT), ("fullnc", "vLLM default-nc (FULL decode graphs)", C.DEFNC),
        ("bnc", "vLLM breakable graphs (no compile)", C.VIOLET),
        ("vcsl", "ours: compiled + sampling, one graph", C.OURS)]
OURS_ARM = "vcsl"
runs = {a: [json.load(open(f)) for f in sorted(glob.glob(f"{HERE}/vllm/out/t_{SET}_{a}_r*.json"))] for a in [x[0] for x in ARMS] + ["vcs"]}
if not runs.get("vcsl"):  # sets made before the warm-GEMM arm (swp): the cold arm is "ours"
    ARMS[3] = ("vcs", "ours: compiled + sampling, one graph", C.OURS)
    OURS_ARM = "vcs"
runs = {a: v for a, v in runs.items() if v}
NRUN = {a: len(v) for a, v in runs.items()}
CAP = next((R["cg_config"]["capture_sizes"] for a in ("default", "fullnc") for R in runs.get(a, []) if R["cg_config"]["capture_sizes"]), [])
DATA = {}


def get(arm, path):
    v = []
    for R in runs.get(arm, []):
        x = R
        try:
            for p in path:
                x = x[p]
        except (KeyError, TypeError):
            continue
        if x is not None:
            v.append(x)
    return v


def med(arm, path):
    v = get(arm, path)
    return st.median(v) if v else None


def save(fig, name, **kw):
    ft = fig.texts[-1]
    import textwrap
    ft.set_text("\n".join(textwrap.fill(p_, 150) for p_ in ft.get_text().replace("\n", " ").split(" || ")))
    fig.subplots_adjust(**({"top": 0.80, "bottom": 0.2} | kw))
    path = os.path.join(OUTD, f"{name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("wrote", path)


runs_note = "Runs per arm: " + ", ".join(f"{a} {n}" for a, n in NRUN.items() if a in [x[0] for x in ARMS]) + ". Median over runs (line); min-max over runs (band); each run's point is the median of its measured steps."
LINE = os.environ.get("SWP_LINE", "c5h: snap/step324 + build_cpp12" if SET.startswith("swph") else "c5g: snap/step318 + build_cpp11")
# placement from the runs' metadata (drive_sweep R["placement"]); sets made before it was recorded: SWP_GPU / SWP_CPUS, else the chain's
_pl = {(R.get("placement", {}).get("cuda_visible_devices"), R.get("placement", {}).get("cpus")) for v in runs.values() for R in v if R.get("placement")}
if _pl:
    GPU_LABEL = "; ".join(f"GPU {g}, CPUs {c}" for g, c in sorted(_pl, key=str))
else:
    GPU_LABEL = f"GPU {os.environ.get('SWP_GPU', '3' if SET == 'swph' else '0')}, CPUs {os.environ.get('SWP_CPUS', '108-143' if SET == 'swph' else '0-71')}"
common_note = (f"Qwen3-8B bf16, GB300 ({GPU_LABEL}; every arm on this GPU, idle-gated), vLLM 0.29 (V2 runner, FlashInfer), host-trace candidate 5 ({LINE}). "
               "Step = execute_model + sample_tokens, synced. Ours: the sound config (only cuBLAS harvested), replays checked bitwise in separate --check runs"
               + ("; warm GEMMs (cuBLAS bindings preloaded, 0 learns in measured steps). " if OURS_ARM == "vcsl" else ". "))

# ---- decode: every batch size (median with a min-max band; vLLM minus ours)
bss = sorted({int(b) for a in runs for R in runs[a] for b in R["decode"]})
nograph = sorted({b for a in ("default", "fullnc") for b in bss for R in runs.get(a, []) if "NONE" in str(R["decode"].get(str(b), {}).get("cg_modes", {}))})
fig, (ax, ax2) = C.slide("Decode step time at every batch size 1-128", "vLLM pads each batch to its next capture size; ours replays at the exact size",
                         common_note + f"Decode at short KV (256-token prompts), one engine per arm-run, {36 if SET == 'swph' else 20} steps per batch size, first 4 dropped. Lines: median over runs; band: min-max. "
                         "Grey lines: vLLM capture sizes" + (f"; vLLM ran without a graph at bs {nograph}." if nograph else "; vLLM graphed every decode step in this range (FULL).")
                         + " Right: vLLM minus ours, per step (median of runs). || " + runs_note, 1, 2, gridspec_kw={"width_ratios": [1.35, 1]})
dec = {}
for a, lab, col in ARMS:
    if a not in runs:
        continue
    vals = [get(a, ("decode", str(b), "step_ms")) for b in bss]
    ys = [st.median(v) if v else None for v in vals]
    dec[a] = ys
    ax.plot(bss, ys, color=col, lw=1.8, label=lab)
    if NRUN[a] > 1:
        ax.fill_between(bss, [min(v) if v else None for v in vals], [max(v) if v else None for v in vals], color=col, alpha=0.15, lw=0)
for c in CAP:
    if c <= max(bss):
        ax.axvline(c, color=C.GRID, lw=0.8, zorder=0)
        ax2.axvline(c, color=C.GRID, lw=0.8, zorder=0)
ax.set_xlabel("decode batch size (requests)")
ax.set_ylabel("step ms")
ax.set_xlim(0, max(bss) + 1)
ax.legend(frameon=False, loc="upper left", fontsize=11)
C.hgrid(ax, "y")
diff = {}
if OURS_ARM in dec:
    for a, lab, col in ARMS[:3]:
        if a in dec:
            d = [(x - o) * 1e3 if x is not None and o is not None else None for x, o in zip(dec[a], dec[OURS_ARM])]
            diff[a] = d
            ax2.plot(bss, d, color=col, lw=1.6, label=lab.split(" (")[0] + " - ours")
    ax2.axhline(0, color=C.MUTED, lw=1)
ax2.set_xlim(0, max(bss) + 1)
ax2.set_xlabel("decode batch size")
ax2.set_ylabel("vLLM ms - ours ms (us)")
ax2.legend(frameon=False, loc="upper right", fontsize=10.5)
C.hgrid(ax2, "y")
save(fig, f"fig_{SET}_decode", wspace=0.28, bottom=0.27)
DATA["decode"] = {"bs": bss, **dec, "diff_us": diff, "capture_sizes": [c for c in CAP if c <= max(bss)], "vllm_nograph_bs": nograph}

# ---- decode at ~1K / ~8K context (InferenceX-style TPOT over 128 measured steps, KV growing over the span)
ctxs = sorted({int(c) for a in runs for R in runs[a] for c in R.get("dec_ctx", {})})
if ctxs:
    fig, axs = C.slide("Decode time per output token vs batch size, ~1K and ~8K context", "Mean step time over 128 decode steps per point (KV grows over the span), as TPOT",
                       common_note + "Each point: bs requests with 1,100- or 8,300-token prompts, the full-batch decode steps after the staggered prefill, 4 dropped, 128 measured (mean). "
                       "Every batch size 1-128. Grey lines: vLLM capture sizes. || " + runs_note.replace("is the median of its measured steps", "is the mean of its 128 measured steps"), 1, len(ctxs))
    axs = axs if len(ctxs) > 1 else [axs]
    for ax, ctx in zip(axs, ctxs):
        bss_c = sorted({int(b) for a in runs for R in runs[a] for b in R.get("dec_ctx", {}).get(str(ctx), {})})
        for a, lab, col in ARMS:
            if a not in runs:
                continue
            vals = [get(a, ("dec_ctx", str(ctx), str(b), "tpot_ms_mean")) for b in bss_c]
            ys = [st.median(v) if v else None for v in vals]
            DATA.setdefault("dec_ctx", {}).setdefault(str(ctx), {})[a] = ys
            ax.plot(bss_c, ys, color=col, lw=1.8, label=lab)
            if NRUN[a] > 1:
                ax.fill_between(bss_c, [min(v) if v else None for v in vals], [max(v) if v else None for v in vals], color=col, alpha=0.15, lw=0)
        for c in CAP:
            if c <= max(bss_c):
                ax.axvline(c, color=C.GRID, lw=0.8, zorder=0)
        ax.set_title(f"~{ctx // 1000}K context ({ctx}-token prompts)")
        ax.set_xlabel("decode batch size")
        C.hgrid(ax, "y")
        DATA["dec_ctx"][str(ctx)]["bs"] = bss_c
    axs[0].set_ylabel("ms per decode step (TPOT)")
    axs[0].legend(frameon=False, loc="upper left", fontsize=10)
    save(fig, f"fig_{SET}_decode_ctx", wspace=0.2, bottom=0.27)

# ---- prefill
lens = sorted({int(L) for a in runs for R in runs[a] for L in R["prefill"]})
fig, ax = C.slide("Prefill step time, 64 to 16K tokens", "One request per step", common_note + "Single-request prefill, 9 steps per length, first 4 dropped. || " + runs_note)
pre = {}
for a, lab, col in ARMS:
    if a not in runs:
        continue
    vals = [get(a, ("prefill", str(L), "step_ms")) for L in lens]
    ys = [st.median(v) if v else None for v in vals]
    pre[a] = ys
    ax.plot(lens, ys, color=col, lw=2.2, marker="o", ms=4, label=lab)
    if NRUN[a] > 1:
        ax.fill_between(lens, [min(v) if v else None for v in vals], [max(v) if v else None for v in vals], color=col, alpha=0.15, lw=0)
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xticks(lens)
ax.set_xticklabels([str(L) for L in lens])
ax.axvline(1024, color=C.MUTED, lw=1, ls="--")
ax.text(1024 * 1.05, ax.get_ylim()[1] * 0.7, "vLLM capture limit 1,024", color=C.MUTED, fontsize=11)
ax.set_xlabel("prompt tokens")
ax.set_ylabel("step ms (log)")
ax.legend(frameon=False, loc="upper left", fontsize=13)
C.hgrid(ax, "y")
save(fig, f"fig_{SET}_prefill", bottom=0.26)
DATA["prefill"] = {"lens": lens, **pre}

# ---- mixed: chunk x decodes
mk = sorted({k for a in runs for R in runs[a] for k in R["ix"] if k.startswith("mixed_")})
spec = lambda k: tuple(map(int, k[6:].split("@")[0].split("x")))
nds = sorted({spec(k)[1] for k in mk})
if nds:
    fig, axes = C.slide("Mixed steps: one prefill chunk + decodes", "Short KV (1-2K) decodes; dashed line: chunk + decodes = vLLM's 1,024-token capture limit",
                        common_note + "Each step: N decodes + one P-token prefill chunk; 6 steps per shape, first 2 dropped. || " + runs_note, 1, len(nds), sharey=False)
    mix = {}
    for ax, nd in zip(axes if len(nds) > 1 else [axes], nds):
        ks = sorted([k for k in mk if spec(k)[1] == nd], key=lambda k: spec(k)[0])
        ps = [spec(k)[0] for k in ks]
        for a, lab, col in ARMS:
            if a not in runs:
                continue
            vals = [get(a, ("ix", k, "step_ms")) for k in ks]
            ys = [st.median(v) if v else None for v in vals]
            mix[f"{a}_{nd}"] = ys
            ax.plot(ps, ys, color=col, lw=2.2, marker="o", ms=4, label=lab)
            if NRUN[a] > 1:
                ax.fill_between(ps, [min(v) if v else None for v in vals], [max(v) if v else None for v in vals], color=col, alpha=0.15, lw=0)
        ax.set_xscale("log", base=2)
        ax.set_xticks(ps)
        ax.set_xticklabels([str(p) for p in ps], fontsize=11)
        ax.axvline(1024 - nd, color=C.MUTED, lw=1, ls="--")
        ax.set_title(f"{nd} decodes")
        ax.set_xlabel("prefill chunk tokens")
        C.hgrid(ax, "y")
    (axes[0] if len(nds) > 1 else axes).set_ylabel("step ms")
    (axes[0] if len(nds) > 1 else axes).legend(frameon=False, loc="upper left", fontsize=9)
    save(fig, f"fig_{SET}_mixed", wspace=0.25, bottom=0.28)
    DATA["mixed"] = {"keys": mk, **mix}

# ---- alternating decode vs fixed at the same KV
altk = sorted({k for a in runs for R in runs[a] for k in R["ix"] if k.startswith("alt_")}, key=lambda k: int(k.split("_")[1]))
if altk:
    fig, ax = C.slide("Batch size changing every step", "Alternating decode A <-> B vs the same sizes held fixed (same KV, 1-2K)",
                      common_note + "Alternating: B decoders, the scheduler's token budget alternates A / B, so steps run A, B, A, B, ... (16 measured steps of each). "
                      "Fixed: the same sizes held for 20 steps (first 4 dropped), same KV spread. Labels: vLLM's padded rows per step (default) and ours' kernel nodes set per step. || " + runs_note)
    xs, w = list(range(len(altk))), 0.13
    alt = {}
    for j, (a, lab, col) in enumerate(ARMS):
        if a not in runs:
            continue
        fx, al = [], []
        for k in altk:
            A, B = map(int, k[4:].split("_"))
            f2 = [med(a, ("ix", f"decode_{x}", "step_ms")) for x in (A, B)]
            fx.append(st.fmean(f2) if all(f2) else None)
            al.append(med(a, ("ix", k, "step_ms")))
        alt[a] = {"fixed": fx, "alternating": al}
        off = (j - 1) * 2 * w
        ax.bar([x + off - w / 2 for x in xs], [y or 0 for y in fx], w, color=col, alpha=0.45, label=f"{lab}: fixed")
        ax.bar([x + off + w / 2 for x in xs], [y or 0 for y in al], w, color=col, label=f"{lab}: alternating")
        for x, k, y in zip(xs, altk, al):
            if a == "default":
                pad = med(a, ("ix", k, "pad_tokens_mean"))
                ax.text(x + off + w / 2, (y or 0) + 0.1, f"pad {pad:.1f}" if pad is not None else "", ha="center", fontsize=9, color=C.INK2)
            if a == OURS_ARM:
                pv = get(a, ("ix", k, "patch"))
                ax.text(x + off + w / 2, (y or 0) + 0.1, f"{st.median(p[1] for p in pv):.0f} nodes" if pv else "", ha="center", fontsize=9, color=C.INK2)
    ax.set_xticks(xs)
    ax.set_xticklabels([k[4:].replace("_", " <-> ") for k in altk])
    ax.set_ylabel("step ms (mean of the two sizes)")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper left")
    C.hgrid(ax, "y")
    save(fig, f"fig_{SET}_alternating", bottom=0.26)
    DATA["alternating"] = {"pairs": altk, **alt}

# ---- tables
L = []
geo = lambda xs: math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float("nan")


def ratio_rows(title, rows, key="step_ms"):
    L.append(f"\n### {title}\n")
    L.append(f"| point | default ms | default-nc ms | breakable ms | ours ({OURS_ARM}) ms | default / ours | default-nc / ours | breakable / ours | default / breakable | vLLM pad rows (default) | ours nodes set | ours patch us | ours learns in measured steps (all runs) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    rs = {"default": [], "fullnc": [], "bnc": [], "db": []}
    for lab, path in rows:
        d, n, o = med("default", path + (key,)), med("fullnc", path + (key,)), med(OURS_ARM, path + (key,))
        bk = med("bnc", path + (key,))
        pad = med("default", path + ("pad_tokens_mean",))
        nodes = med("vcs", path + ("patch",)) if False else None
        pv = get(OURS_ARM, path + ("patch",))
        nodes = st.median(p[1] for p in pv) if pv else None
        pu = med(OURS_ARM, path + ("ph_us", "patch"))
        lr = get(OURS_ARM, path + ("learns_in_rows",))
        f = lambda x: f"{x:.2f}" if x is not None else "-"
        if d and o:
            rs["default"].append(d / o)
        if n and o:
            rs["fullnc"].append(n / o)
        if bk and o:
            rs["bnc"].append(bk / o)
        if d and bk:
            rs["db"].append(d / bk)
        L.append(f"| {lab} | {f(d)} | {f(n)} | {f(bk)} | {f(o)} | {f(d / o) if d and o else '-'} | {f(n / o) if n and o else '-'} | {f(bk / o) if bk and o else '-'} | {f(d / bk) if d and bk else '-'} | {f(pad)} | {f(nodes)} | {f(pu)} | {sum(lr) if lr else '-'} |")
    L.append(f"| geomean | | | | | {geo(rs['default']):.2f} | {geo(rs['fullnc']):.2f} | {geo(rs['bnc']):.2f} | {geo(rs['db']):.2f} | | | | |")
    return rs


ratio_rows("Decode, every batch size", [(f"bs {b}", ("decode", str(b))) for b in bss])
for ctx in ctxs:
    ratio_rows(f"Decode at ~{ctx // 1000}K context (mean step ms over 128 steps)", [(f"bs {b}", ("dec_ctx", str(ctx), str(b))) for b in sorted({int(b) for a in runs for R in runs[a] for b in R.get("dec_ctx", {}).get(str(ctx), {})})], key="tpot_ms_mean")
ratio_rows("Prefill", [(f"{L_} tokens", ("prefill", str(L_))) for L_ in lens])
ratio_rows("Mixed", [(k[6:], ("ix", k)) for k in sorted(mk, key=lambda k: (spec(k)[1], spec(k)[0]))])
alt = sorted({k for a in runs for R in runs[a] for k in R["ix"] if k.startswith("alt_")})
rows = []
for k in alt:
    A, B = map(int, k[4:].split("_"))
    rows += [(f"alt {A}<->{B} (both)", ("ix", k)), (f"alt {A}<->{B}: bs {A} steps", ("ix", k, f"bs{A}")), (f"alt {A}<->{B}: bs {B} steps", ("ix", k, f"bs{B}")),
             (f"fixed bs {A} (same KV)", ("ix", f"decode_{A}")), (f"fixed bs {B} (same KV)", ("ix", f"decode_{B}"))]
ratio_rows("Alternating decode (batch size changes every step)", rows)
# ours: traces / variants per entry over the whole run
L.append("\n### Ours: traces and variants per entry over the whole run (all sweeps in one engine)\n")
L.append("| run | entry | traces | replays | eager | variants |")
L.append("|---|---|---|---|---|---|")
for i, R in enumerate(runs.get(OURS_ARM, [])):
    for key, e in R.get("V", {}).get("entries", {}).items():
        L.append(f"| r{i + 1} | {key[:40]} | {e.get('traces')} | {e.get('replays')} | {e.get('eager')} | {e.get('variants')} |")
    L.append(f"| r{i + 1} | learned cuBLAS keys | {R.get('V', {}).get('learned_by_op')} | | | |")
open(os.path.join(HERE, "serve", "out", f"{SET}_tables.md"), "w").write("\n".join(L) + "\n")
json.dump(DATA, open(os.path.join(OUTD, f"{SET}_data.json"), "w"), indent=1, default=str)
print("\n".join(L[:12]))
