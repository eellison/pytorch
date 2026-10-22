# Slide sweeps by graphing method (10-13, user: compare cudagraphing methods on the same forward, no cross-group ratios):
#   compiled forward:  vLLM default (compile + its graphs)        vs ours compiled (vcsl: Vc + sampling, warm GEMMs)
#   eager forward:     vLLM default-nc (FULL decode graphs) and vLLM breakable graphs  vs ours uncompiled (vrs: V-registered + sampling)
# Charts: one panel row per forward. Tables: ratios within a group only. Writes serve/out/charts/fig_<SET>_*.png, serve/out/<SET>_tables.md.
#   PYTHONPATH=presentation/slide_pack/pylib python3 serve/ix/swp_charts2.py SET
import glob, json, math, os, statistics as st, sys, textwrap

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "talk"))
import charts as C  # noqa: E402

plt = C.plt
SET = sys.argv[1]
OUTD = os.path.join(HERE, "serve", "out", "charts")
NAME = {"default": "vLLM default graphs", "vcsl": "ours", "fullnc": "vLLM FULL decode graphs (default-nc)", "bnc": "vLLM breakable graphs", "vrs": "ours"}
COL = {"default": C.DEFAULT, "vcsl": C.OURS, "fullnc": C.DEFNC, "bnc": C.VIOLET, "vrs": C.OURS}
GROUPS = [("compiled forward (torch.compile)", ["default"], "vcsl"), ("eager forward (no compile)", ["fullnc", "bnc"], "vrs")]
FILES = {a: sorted(glob.glob(f"{HERE}/vllm/out/t_{SET}_{a}_r*.json")) for a in NAME}
runs = {a: [json.load(open(f)) for f in FILES[a]] for a in NAME}
runs = {a: v for a, v in runs.items() if v}
GROUPS = [(g, [b for b in bl if b in runs], o) for g, bl, o in GROUPS if o in runs or any(b in runs for b in bl)]
NRUN = {a: len(v) for a, v in runs.items()}
CAP = next((R["cg_config"]["capture_sizes"] for a in ("default", "fullnc") for R in runs.get(a, []) if R["cg_config"]["capture_sizes"]), [])
pl = {(R.get("placement", {}).get("cuda_visible_devices"), R.get("placement", {}).get("cpus")) for v in runs.values() for R in v if R.get("placement")}
WHERE = "; ".join(f"GPU {g}, CPUs {c}" for g, c in sorted(pl, key=str)) or "GPU ?"
NOTE = (f"Qwen3-8B bf16, GB300 ({WHERE}; every arm on this GPU), vLLM 0.29 (V2 runner, FlashInfer), host-trace c5h (snap/step324 + build_cpp12). Step = execute_model + "
        "sample_tokens, synced. Ours runs with phase clocks on; any overhead counts against ours. Each panel compares graphing methods running the same forward (same kernels); no ratio crosses panels. Ours: one graph per step "
        "including sampling, only cuBLAS harvested, 0 traces and 0 learns in measured steps (enforced)" + ("; compiled: cuBLAS bindings preloaded. " if any(g.startswith("compiled") for g, _, _ in GROUPS) else ". "))
PENDING = ("PENDING MEMSET FIX: ours' decode and mixed rows (any step with decode attention) are pending a core fix: FlashInfer's multi-CTA counter "
           "zeroing replays a wrong size (constant per variant instead of following the batch); re-verified bitwise and spot-retimed after the fix. ")
PEND_C = "Ours' decode and mixed rows: PENDING a core memset fix (counter-zeroing size); re-verified and spot-retimed after it. "
if os.environ.get("SWP_PENDING", "1") == "0":  # 10-14: the held-memset spot check matched FlashInfer's formula; no pending label
    PENDING = PEND_C = ""
MSL = ("Decode differences at small bs: vLLM's static graphs force worst-case max_seq_len (= max_model_len 16,384) into the attention kernel; ours uses the actual value. ")
GCS = 10  # a step this many times the run's median step is a one-step stall; on ours these are gen-2 GC passes over trace garbage (gcv round)
if any(R.get("empty_cache_gib") for v in runs.values() for R in v):
    NOTE += ("Ours re-freezes the GC heap after post-warm-up traces (no gen-2 collection landed in any of ours' measured windows); vLLM's tail "
             "spikes are gen-2 GC passes inside its windows (vLLM as shipped does not re-freeze after its warm-up); every arm empties the allocator "
             "cache before the first point; 3 rounds, gated on host memory pressure. ")
if os.environ.get("SWP_DROP_FLAGGED") == "1":
    NOTE += ("Per round, points flagged for host noise (memory pressure > 2% of the point, or steps > 10x the median adding > 1%) are dropped. "
             "Ours re-freezes the GC heap after post-warm-up traces; every arm empties the allocator cache before the first point. ")
RUNS = "Runs per arm: " + ", ".join(f"{a} {n}" for a, n in NRUN.items()) + ". Line: median over runs; band: min-max over runs."


DROP_FLAGGED = os.environ.get("SWP_DROP_FLAGGED") == "1"  # 10-15: drop a round's point flagged for host noise (tail.flagged)


def get(a, path):
    out = []
    for R in runs.get(a, []):
        x = R
        try:
            for p in path:
                parent, x = x, x[p]
        except (KeyError, TypeError):
            continue
        if DROP_FLAGGED and isinstance(parent, dict) and (parent.get("tail") or {}).get("flagged"):
            continue
        if x is not None:
            out.append(x)
    return out


def med(a, path):
    v = get(a, path)
    return st.median(v) if v else None


def save(fig, name, bottom=0.27, **kw):
    ft = fig.texts[-1]
    ft.set_text("\n".join(textwrap.fill(p_, 160) for p_ in ft.get_text().replace("\n", " ").split(" || ")))
    fig.subplots_adjust(top=0.82, bottom=bottom, **kw)
    path = os.path.join(OUTD, f"{name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("wrote", path)


def line(ax, a, xs, path_of, lw=1.8, marker=None):
    vals = [get(a, path_of(x)) for x in xs]
    ys = [st.median(v) if v else None for v in vals]
    ax.plot(xs, ys, color=COL[a], lw=lw, label=NAME[a], marker=marker, ms=3)
    if NRUN.get(a, 0) > 1:
        ax.fill_between(xs, [min(v) if v else None for v in vals], [max(v) if v else None for v in vals], color=COL[a], alpha=0.15, lw=0)
    return ys


def stalls(a, path):
    # per run of arm a at this point: the slowest step's ms where it is >= GCS x the run's median step
    out = []
    for v in get(a, path):
        sl = v.get("slowest_step") if isinstance(v, dict) else None
        if sl and v.get("step_ms") and sl["step_ms"] >= GCS * v["step_ms"]:
            out.append(sl["step_ms"])
    return out


def caps(ax, xmax):
    for c in CAP:
        if c <= xmax:
            ax.axvline(c, color=C.GRID, lw=0.8, zorder=0)


# ---- decode at ~1K / ~8K context: rows = forward group, cols = context
ctxs = sorted({int(c) for v in runs.values() for R in v for c in R.get("dec_ctx", {})})
if ctxs:
    fig, axs = C.slide("Decode time per output token vs batch size", "TPOT over 128 decode steps per point; " + ("top row: compiled forward, bottom row: eager forward" if len(GROUPS) > 1 else GROUPS[0][0]),
                       NOTE + "Each point: bs requests with 1,100- or 8,300-token prompts; the full-batch decode steps after the staggered prefill, 4 dropped, 128 measured (mean). "
                       "Every batch size 1-128. Grey lines: vLLM capture sizes. x: a point where an ours run had a one-step stall (>= 10x the median step; a gen-2 GC pass over "
                       "trace garbage), which lifts that run's mean. " + MSL + "|| " + PEND_C + RUNS, len(GROUPS), len(ctxs), sharex=True)
    axs = [[axs]] if len(GROUPS) == 1 and len(ctxs) == 1 else ([axs] if len(GROUPS) == 1 else ([[r] for r in axs] if len(ctxs) == 1 else axs))
    for gi, (g, bl, o) in enumerate(GROUPS):
        for ci, ctx in enumerate(ctxs):
            ax = axs[gi][ci]
            bss = sorted({int(b) for v in runs.values() for R in v for b in R.get("dec_ctx", {}).get(str(ctx), {})})
            for a in bl + [o]:
                if a in runs:
                    ys = line(ax, a, bss, lambda b: ("dec_ctx", str(ctx), str(b), "tpot_ms_mean"))
                    if a == o:
                        for b, y in zip(bss, ys):
                            if stalls(o, ("dec_ctx", str(ctx), str(b))):
                                ax.plot([b], [y], marker="x", color=C.INK2, ms=7, mew=1.5)
            caps(ax, max(bss))
            C.hgrid(ax, "y")
            if gi == 0:
                ax.set_title(f"~{ctx // 1000}K context", fontsize=14)
            if ci == 0:
                ax.set_ylabel(f"{g.split(' (')[0]}\nms per step", fontsize=12)
                ax.legend(frameon=False, fontsize=9, loc="upper left")
            if gi == len(GROUPS) - 1:
                ax.set_xlabel("decode batch size")
    save(fig, f"fig_{SET}_decode_ctx", bottom=0.33, hspace=0.12, wspace=0.15)

# ---- prefill: one panel per group
lens = sorted({int(L) for v in runs.values() for R in v for L in R.get("prefill", {}) if str(L).isdigit()})
if lens:
    fig, axs = C.slide("Prefill step time, 64 to 16K tokens", "One request per step; left: compiled forward, right: eager forward",
                       NOTE + "Single-request prefill, 9 steps per length, first 4 dropped. Dashed: vLLM's 1,024-token capture limit. || " + RUNS, 1, len(GROUPS), sharey=True)
    axs = [axs] if len(GROUPS) == 1 else axs
    for ax, (g, bl, o) in zip(axs, GROUPS):
        for a in bl + [o]:
            if a in runs:
                line(ax, a, lens, lambda L: ("prefill", str(L), "step_ms"), lw=2.0, marker="o")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(lens)
        ax.set_xticklabels([str(L) for L in lens], fontsize=9, rotation=45)
        ax.axvline(1024, color=C.MUTED, lw=1, ls="--")
        ax.set_title(g, fontsize=14)
        ax.set_xlabel("prompt tokens")
        ax.legend(frameon=False, fontsize=10, loc="upper left")
        C.hgrid(ax, "y")
    axs[0].set_ylabel("step ms (log)")
    save(fig, f"fig_{SET}_prefill", wspace=0.08)

# ---- mixed: rows = group, cols = decodes
mk = sorted({k for v in runs.values() for R in v for k in R.get("ix", {}) if k.startswith("mixed_")})
spec = lambda k: tuple(map(int, k[6:].split("@")[0].split("x")))
nds = sorted({spec(k)[1] for k in mk})
if nds:
    fig, axs = C.slide("Mixed steps: one prefill chunk + decodes", "Top row: compiled forward, bottom row: eager forward; dashed: chunk + decodes = 1,024",
                       NOTE + "Each step: N decodes + one P-token prefill chunk; 6 steps per shape, first 2 dropped. || " + PEND_C + RUNS, len(GROUPS), len(nds), sharex=True)
    axs = [axs] if len(GROUPS) == 1 else axs
    for gi, (g, bl, o) in enumerate(GROUPS):
        for ci, nd in enumerate(nds):
            ax = axs[gi][ci]
            ks = sorted([k for k in mk if spec(k)[1] == nd], key=lambda k: spec(k)[0])
            ps = [spec(k)[0] for k in ks]
            for a in bl + [o]:
                if a in runs:
                    vals = [get(a, ("ix", k, "step_ms")) for k in ks]
                    ax.plot(ps, [st.median(v) if v else None for v in vals], color=COL[a], lw=2, marker="o", ms=3.5, label=NAME[a])
                    if NRUN[a] > 1:
                        ax.fill_between(ps, [min(v) if v else None for v in vals], [max(v) if v else None for v in vals], color=COL[a], alpha=0.15, lw=0)
            ax.set_xscale("log", base=2)
            ax.set_xticks(ps)
            ax.set_xticklabels([str(p) for p in ps], fontsize=9)
            ax.axvline(1024 - nd, color=C.MUTED, lw=1, ls="--")
            C.hgrid(ax, "y")
            if gi == 0:
                ax.set_title(f"{nd} decodes", fontsize=14)
            if ci == 0:
                ax.set_ylabel(f"{g.split(' (')[0]}\nstep ms", fontsize=12)
                ax.legend(frameon=False, fontsize=8.5, loc="upper left")
            if gi == len(GROUPS) - 1:
                ax.set_xlabel("prefill chunk tokens")
    save(fig, f"fig_{SET}_mixed", bottom=0.31, hspace=0.12, wspace=0.2)

# ---- alternating vs fixed (same KV): one panel per group
altk = sorted({k for v in runs.values() for R in v for k in R.get("ix", {}) if k.startswith("alt_")}, key=lambda k: int(k.split("_")[1]))
if altk:
    fig, axs = C.slide("Batch size changing every step", "Alternating A <-> B vs the same sizes held fixed; left: compiled, right: eager forward",
                       NOTE + "Alternating: B decoders, the scheduler's token budget alternates A / B (16 measured steps of each). Fixed: the same sizes held, 20 steps, first 4 dropped. "
                       "Labels: vLLM's padded rows per step; ours' kernel nodes set per step. || " + PEND_C + RUNS, 1, len(GROUPS), sharey=True)
    axs = [axs] if len(GROUPS) == 1 else axs
    for ax, (g, bl, o) in zip(axs, GROUPS):
        arms = [a for a in bl + [o] if a in runs]
        w = 0.8 / (2 * len(arms))
        xs = list(range(len(altk)))
        for j, a in enumerate(arms):
            fx, al = [], []
            for k in altk:
                A, B = map(int, k[4:].split("_"))
                f2 = [med(a, ("ix", f"decode_{x}", "step_ms")) for x in (A, B)]
                fx.append(st.fmean(f2) if all(f2) else 0)
                al.append(med(a, ("ix", k, "step_ms")) or 0)
            off = -0.4 + (2 * j) * w
            ax.bar([x + off + w / 2 for x in xs], fx, w, color=COL[a], alpha=0.45, label=f"{NAME[a]}: fixed")
            ax.bar([x + off + 1.5 * w for x in xs], al, w, color=COL[a], label=f"{NAME[a]}: alternating")
            for x, k, y in zip(xs, altk, al):
                if a == o:
                    pv = get(a, ("ix", k, "patch"))
                    lab = f"{st.median(p[1] for p in pv):.0f} nodes" if pv else ""
                else:
                    pad = med(a, ("ix", k, "pad_tokens_mean"))
                    lab = f"pad {pad:.1f}" if pad is not None else ""
                ax.text(x + off + 1.5 * w, y + 0.1, lab, ha="center", fontsize=8, color=C.INK2, rotation=90, va="bottom")
        ax.set_xticks(xs)
        ax.set_xticklabels([k[4:].replace("_", " <-> ") for k in altk])
        ax.set_title(g, fontsize=14)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.35)
        ax.legend(frameon=False, fontsize=8, ncol=1, loc="upper left")
        C.hgrid(ax, "y")
    axs[0].set_ylabel("step ms (mean of the two sizes)")
    save(fig, f"fig_{SET}_alternating", bottom=0.31, wspace=0.06)

# ---- tables: ratios within a group only
L = [f"# {SET}: graphing methods on the same forward ({WHERE})\n", "Ratio = vLLM ms / ours ms (> 1: ours faster). No ratio crosses the compiled / eager groups. " + RUNS + "\n",
     PENDING + "\n", "Pad rows: vLLM's padded tokens per step (its graph sizes round the batch up); ours runs the exact size.\n",
     "Ours runs with phase clocks on (the C++ replay entry's phase timers and patch counters); any overhead counts against ours.\n"]
geo = lambda xs: math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float("nan")
f2 = lambda x: f"{x:.2f}" if x is not None else "-"


def section(title, rows, key="step_ms", note="", stall=False):
    for g, bl, o in GROUPS:
        L.append(f"\n### {title}: {g}\n")
        if note:
            L.append(note + "\n")
        hdr = " | ".join([f"{b} ms" for b in bl] + [f"ours ({o}) ms"] + [f"{b} / ours" for b in bl] + [f"{b} pad rows" for b in bl] + ["ours nodes set", "ours learns+traces in measured rows"]
                         + (["ours one-step stalls (runs: ms)"] if stall else []))
        L.append(f"| point | {hdr} |")
        L.append("|---|" + "---|" * (3 * len(bl) + 3 + stall))
        rs = {b: [] for b in bl}
        for lab, path in rows:
            o_ = med(o, path + (key,))
            bs_ = [med(b, path + (key,)) for b in bl]
            for b, x in zip(bl, bs_):
                if x and o_:
                    rs[b].append(x / o_)
            pv = get(o, path + ("patch",))
            lr = sum(get(o, path + ("learns_in_rows",))) + sum(get(o, path + ("traces_in_rows",)))
            cells = [f2(x) for x in bs_] + [f2(o_)] + [f2(x / o_) if x and o_ else "-" for x in bs_] + [f2(med(b, path + ("pad_tokens_mean",))) for b in bl] + \
                    [f2(st.median(p[1] for p in pv)) if pv else "-", str(lr) if o in runs else "-"]
            if stall:
                sv = stalls(o, path)
                cells.append(f"{len(sv)}/{NRUN.get(o, 0)}: " + ", ".join(f"{x:.0f}" for x in sv) + " (GC pass)" if sv else "-")
            L.append(f"| {lab} | " + " | ".join(cells) + " |")
        L.append("| geomean | " + " | ".join([""] * (len(bl) + 1) + [f"{geo(rs[b]):.2f}" for b in bl] + [""] * (len(bl) + 2 + stall)) + " |")


for ctx in ctxs:
    bss = sorted({int(b) for v in runs.values() for R in v for b in R.get("dec_ctx", {}).get(str(ctx), {})})
    section(f"Decode at ~{ctx // 1000}K context (TPOT, mean over 128 steps)", [(f"bs {b}", ("dec_ctx", str(ctx), str(b))) for b in bss], key="tpot_ms_mean",
            note=PENDING + MSL + "Stalls column: ours runs with a step >= 10x the median step (a gen-2 GC pass over trace garbage), which lifts that run's TPOT mean.", stall=True)
section("Prefill", [(f"{L_} tokens", ("prefill", str(L_))) for L_ in lens],
        note="Single request: FlashInfer's counter is max(152, 32 x batch) = 152 semaphores at batch 1, so the memset size issue does not change these rows.")
section("Mixed", [(k[6:], ("ix", k)) for k in sorted(mk, key=lambda k: (spec(k)[1], spec(k)[0]))], note=PENDING)
rows = []
for k in altk:
    A, B = map(int, k[4:].split("_"))
    rows += [(f"alt {A}<->{B}", ("ix", k)), (f"fixed bs {A} (same KV)", ("ix", f"decode_{A}")), (f"fixed bs {B} (same KV)", ("ix", f"decode_{B}"))]
section("Alternating decode", rows, note=PENDING + MSL)
bss0 = sorted({int(b) for v in runs.values() for R in v for b in R.get("decode", {})})
section("Decode at short KV (256-token prompts; supporting)", [(f"bs {b}", ("decode", str(b))) for b in bss0], note=PENDING + MSL)
L.append("\n### Startup and graphs (median over runs)\n")
L.append("| arm | engine init s | capture_model s | CUDA graphs captured | graph memory GiB | KV cache tokens | bindings preloaded |")
L.append("|---|---|---|---|---|---|---|")
for a in runs:
    cap = lambda k: med(a, ("capture", k))
    gb = cap("graph_bytes")
    L.append(f"| {a} | {f2(med(a, ('init_s',)))} | {f2(cap('capture_model_s'))} | {f2(cap('capture_begins'))} | {f2(gb / 2**30) if gb else '-'} | {f2(med(a, ('mem', 'kv_cache_tokens')))} | {f2(med(a, ('V', 'bindings_loaded')))} |")
L.append("\n### Background on the run's cores (foreign busy fraction over the run; > 0.15 flagged)\n")
L.append("| arm | run | foreign busy fraction | busy cores on own range (mean) | own process cores | note |")
L.append("|---|---|---|---|---|---|")
for a in runs:
    for i, R in enumerate(runs[a]):
        pl = R.get("placement", {})
        rname = os.path.basename(FILES[a][i])[:-5].rsplit("_", 1)[-1]
        fb = pl.get("foreign_busy_frac")
        note = ""
        if fb is None and pl.get("busy_cores_own_range_mean") is not None:  # runs before own-process accounting: own ~1 core
            fb = max(0.0, pl["busy_cores_own_range_mean"] - 1.0) / 36
            note = "est. (own process ~1 core)"
        flag = " FLAGGED" if fb is not None and fb > 0.15 else ""
        L.append(f"| {a} | {rname} | {f2(fb)}{flag} | {f2(pl.get('busy_cores_own_range_mean'))} | {f2(pl.get('own_process_cores_mean'))} | {note} |")
open(os.path.join(HERE, "serve", "out", f"{SET}_tables.md"), "w").write("\n".join(L) + "\n")
print("tables:", os.path.join(HERE, "serve", "out", f"{SET}_tables.md"))
