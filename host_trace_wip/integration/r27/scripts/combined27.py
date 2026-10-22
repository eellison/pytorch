# combined27.py [out/timing] [out_dev3]: the 27B headline across the final sets (c5h, Qwen3.8-27B NVFP4, vLLM 0.29).
#   decode curves: n27d (GPU 1; pure decode runs FLA's fused_recurrent GDN decode, no GDN prefill kernel in any backend)
#   FlashInfer GDN (vLLM's GB300 default): n27wf (prefill / mixed, GPU 2), n27cbf (churn, GPU 2); NaN processes listed
#   Triton GDN (bitwise-checked): n27wt (prefill / mixed), n27cbt (churn), both dev-box GPU 3
# Each group has two forward sections (no ratio crosses them), the faster forward's first. Ratio = vLLM / ours (> 1: ours faster).
# Runs with window_violations (a trace or learn inside a measured window) or section entries > 0 are excluded and counted.
import collections, glob, json, os, re, statistics as st, sys

here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "out", "timing")
D3 = sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, "out_dev3")
FI = ("vLLM default GDN backend (FlashInfer 0.6.18): not bitwise-checkable; its prefill is nondeterministic across call paths "
      "within a process, and NaNs in some processes without vLLM or host trace (flashinfer#4704)")
GROUPS = [
    ("DECODE CURVES (n27dg, GPU 2, 3 rounds + PSI-gated reruns of points flagged in 2+ rounds; 16 warm + 128 measured steps, per-token mean). "
     "Ours re-freezes the GC after its warm-up traces. Decode attention: vLLM's static graphs force worst-case max_seq_len; ours uses the "
     "actual value. breakable+gdnbreak is the breakable decode arm (plain breakable: padded sizes only, a cross-check). A point flagged "
     "(PSI > 2% or outlier excess > 1%) is a label; every point = median over its rounds and reruns of each round's per-step median. Kernel identity (out/kid/KID_d.txt): only that label and vLLM "
     "padding differ", [(D, "n27dg"), (D, "n27dgf1024"), (D, "n27dgf8192")], False),
    # churn: n27cbn / n27cbng (truly unsynced windows; the earlier n27cbf / n27cbfg windows had a harness sync pair per step: withdrawn)
    ("PREFILL / MIXED / CHURN, " + FI + " (n27wf, n27cbn: GPU 2)", [(D, "n27wf"), (D, "n27wfg"), (D, "n27cbn"), (D, "n27cbng")], True),
    ("PREFILL / MIXED, Triton GDN, bitwise-checked (n27wt: dev-box GPU 3)", [(D3, "n27wt"), (D3, "n27wtg")], False),
]
SECS = [("COMPILED FORWARD: vLLM default (piecewise + FULL decode) vs ours Vc+VP", ["default"], "vcvp"),
        ("EAGER FORWARD: vLLM default-nc (FULL decode), vLLM breakable and breakable+gdnbreak vs ours V+VP", ["fullnc", "breakable", "breakablegb"], "vsvp")]
# vLLM 0.29 breakable replays the GDN decode path for prefill / mixed steps of <= 1024 tokens (its PIECEWISE sizes): wrong outputs
# (out/BREAKABLE_GDN.md). Those breakable rows, and its churn rows (every churn window has such mixed steps), are INVALID; the
# correctness-forced arm breakable+gdnbreak (Qwen GDN as an eager break; sets n27wfg / n27cbfg / n27wtg / n27cbtg) replaces them.
BRK_INVALID = "vLLM 0.29 breakable replays the GDN decode path for prefill / mixed steps <= 1024 tokens (wrong outputs, out/BREAKABLE_GDN.md)"


# 10-15 (out/dp27/CMP.txt, after Flash-Next's finding): breakable's FULL decode graphs give wrong tokens at UNPADDED capture sizes (bs 32 /
# 64 at 1K: 0/32, 0/64 sequences match default-nc, degenerate; bs 8 at 8K: 0/8) and correct ones where the batch is padded (bs 31, 9).
# Unpadded decode sizes of the n27d config (pad 0 in breakable's n27d rows): 1, 2, 4, 8, 16, then every multiple of 8 to 128.
BRK_DEC_UNPADDED = {1, 2, 4, 8, 16} | set(range(24, 129, 8))
BRK_DEC_INVALID = "vLLM 0.29 breakable's FULL decode graphs give wrong tokens at unpadded capture sizes (out/dp27/CMP.txt)"


def brk_invalid(row):
    m = re.match(r"prefill\s+(\d+)$", row)
    if m:
        return int(m.group(1)) <= 1024
    m = re.match(r"mixed_(\d+)x(\d+)", row)
    if m:
        return int(m.group(1)) + int(m.group(2)) <= 1024
    m = re.match(r"decode \dK ctx bs\s+(\d+)", row)
    if m:
        return int(m.group(1)) in BRK_DEC_UNPADDED
    return row.startswith("churn")


def load(sets):
    rows, nan, dropped = collections.OrderedDict(), [], []
    flags = collections.defaultdict(list)  # (row, arm) -> flagged per value (decode-curve points with stall stats)
    add = lambda r, arm, v, fl=False: (rows.setdefault(r, collections.defaultdict(list))[arm].append(v), flags[(r, arm)].append(fl))
    for d, setn, *only in sets:
        for f in sorted(glob.glob(os.path.join(d, f"t_{setn}_*_r{only[0] if only else '*'}.json"))):
            m = re.match(rf"t_{re.escape(setn)}_(default|fullnc|vcvp|breakable|breakablegb|vsvp)_r(\d+)\.json", os.path.basename(f))
            if not m:
                continue
            arm, j = m.group(1), json.load(open(f))
            secs = (j.get("V") or {}).get("sections_added") or 0
            if j.get("window_violations") or secs:
                dropped.append(f"{os.path.basename(f)} (violations {len(j.get('window_violations') or [])}, sections {secs})")
                continue
            if j.get("nan_probe") is not None:
                nan.append((setn, arm, m.group(2), j["nan_probe"]))
            for L, v in (j.get("prefill", {}) if not setn.startswith("n27d") else {}).items():  # n27d's prefill row is its warm-up probe
                v.get("step_ms") and add(f"prefill {int(L):5d}", arm, v["step_ms"])
            for k, v in j.get("dcurve", {}).items():
                ctx, bs = map(int, k.split("_"))
                # each round's value at a point is its per-step MEDIAN (a single stall cannot move it; sets without step stats: the mean)
                pv = (v.get("stalls") or {}).get("median_ms") or v.get("mean_ms")
                pv and add(f"decode {ctx // 1024}K ctx bs {bs:3d} (per token)", arm, pv, bool(((v.get("stalls") or {}).get("flag") or {}).get("flagged")))
            for k, v in j.get("ix", {}).items():
                if v.get("step_ms") and k.startswith(("mixed_", "alt_")):
                    add(k, arm, v["step_ms"])
            for C, v in j.get("cb", {}).items():
                v.get("wall_ms_per_token") and add(f"churn C={C} (per token)", arm, v["wall_ms_per_token"])
    # a point's value is the median over all its rounds and reruns, flagged or not (10-16, the slides session's spikes: dropping clean
    # flagged rounds kept stall-inflated ones); PSI / outlier flags are labels, counted per group below
    med = {r: {a: st.median(x) for a, x in d.items()} for r, d in rows.items()}
    load.flagged = sum(fl for v in flags.values() for fl in v)
    load.values = sum(len(v) for v in flags.values())
    invalid = {r: m.pop("breakable") for r, m in med.items() if "breakable" in m and brk_invalid(r)}
    return med, nan, dropped, invalid


print("27B (c5h, Qwen3.8-27B NVFP4, vLLM 0.29). Every arm of a set ran on that set's GPU. Every ratio compares two CUDA-graphing")
print("methods on the same forward (the same kernels except the labelled cases: vLLM padding, decode max_seq_len); no ratio crosses the two forward sections.")
print("No Triton-GDN churn: n27cbt's windows had the harness sync pair per step (and rounds 2-3 KID profiling); dev-box unreachable for a rerun.")
print("CAVEAT: vLLM's prefill / mixed step is host time (prefill-64 default: GPU busy 10.5 ms of a 216 ms step); graphing removes it.")
print("SYNCS (measured, out/syncstep): every step row is synced at the step level in every arm. On top of that, ours and vLLM's PIECEWISE")
print("steps carry one forward-level sync pair (the harness's synced() wrapper; no per-piece syncs); vLLM's NONE-mode steps (default-nc")
print("prefill / mixed, default and breakable above 1024 tokens) carry none, so ratios against those rows are slightly conservative for ours.")
for gtitle, sets, show_nan in GROUPS:
    med, nan, dropped, invalid = load(sets)
    print("\n" + "=" * 100 + "\n" + gtitle)
    if not med:
        print("  (no runs yet)")
        continue
    secs = list(SECS)
    both = [m for m in med.values() if "vcvp" in m and "vsvp" in m]
    if both:
        eager_faster = st.geometric_mean([m["vsvp"] for m in both]) < st.geometric_mean([m["vcvp"] for m in both])
        if eager_faster:
            secs.reverse()
        print(f"Faster forward (ours on each, geomean over {len(both)} rows): {'eager' if eager_faster else 'compiled'}.")
    for title, vl, ours in secs:
        print(f"\n{title}")
        print(f"{'row':44s} " + " ".join(f"{a:>10s}" for a in vl) + f" {'ours':>10s} " + " ".join(f"{a + '/ours':>13s}" for a in vl))
        for r, m in med.items():
            if ours not in m or not any(a in m for a in vl):
                continue
            print(f"{r:44s} " + " ".join(f"{m[a]:10.2f}" if a in m else f"{'-':>10s}" for a in vl) + f" {m[ours]:10.2f} "
                  + " ".join(f"{m[a] / m[ours]:12.2f}x" if a in m else f"{'-':>13s}" for a in vl))
        for grp in ("decode 1K", "decode 8K", "prefill", "mixed", "alt", "churn"):
            for a in vl:
                g = [m[a] / m[ours] for r, m in med.items() if r.startswith(grp) and a in m and ours in m]
                if g:
                    print(f"  {grp:10s} {a:10s} n={len(g):3d} geomean {a}/ours {st.geometric_mean(g):.2f}x")
    if show_nan and nan:
        print("\nNaN processes (one-request prefill 64 / 512 / 2048 with logprobs after the measurements; True = NaN logprob):")
        for setn, arm, rnd, pr in sorted(nan):
            print(f"  {setn} {arm:10s} r{rnd}: " + " ".join(f"{L}:{'NaN' if v else 'ok'}" for L, v in pr.items()))
        print(f"  {sum(any(p.values()) for *_, p in nan)} of {len(nan)} processes had a NaN.")
    if getattr(load, "flagged", 0):
        print(f"\nHost-noise labels: {load.flagged} of {load.values} per-round point values are PSI / outlier flagged (values kept; medians).")
    if invalid:
        print(f"\nINVALID (not in any ratio): breakable at {len(invalid)} rows; {BRK_INVALID}; decode rows: {BRK_DEC_INVALID}:")
        print("  " + ", ".join(f"{r.strip()} {v:.2f} ms" for r, v in invalid.items()))
    if dropped:
        print("\nExcluded runs: " + "; ".join(dropped))
