"""Flash-Next timing tables from the gb300-box sets (g2w: w on GPU 2; g0d / g0d20: d on GPU 0).
    python3 scripts/fn_tables.py > out/timing/TABLES_gb300-box.md
Per point and arm: the round values (step_ms, synced full step incl. sampling), shown as median [min-max] over rounds; ratios are
vLLM arm / ours on the medians. A round more than 5% off the median of its rounds is flagged (rerun rule). Breakable PIECEWISE
prefill / mixed points of at most 1024 tokens are INVALID (GDN decode branch replayed, r27/out/BREAKABLE_GDN.md)."""
import json
import math
import os
import re
import statistics
import sys

T = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out", "timing")
ARMS = ("fullnc", "breakable", "v")
INVALID_BRK = {"p64", "p128", "p256", "p512", "p1024", "mixed_256x16", "mixed_256x64", "mixed_512x16", "mixed_512x64"}


def load(name):
    p = os.path.join(T, name + ".json")
    return json.load(open(p)) if os.path.exists(p) else None


def fmt(vals):
    m = statistics.median(vals)
    return m, f"{m:.1f} [{min(vals):.1f}-{max(vals):.1f}]"


def dev_flag(vals):
    m = statistics.median(vals)
    return any(abs(x - m) / m > 0.05 for x in vals) if len(vals) > 1 else False


def w_points(d):
    pts = {f"p{k}": (v["step_ms"], v.get("cg_modes")) for k, v in d["prefill"].items()}
    for k, v in d["ix"].items():
        pts[k.split("@")[0]] = (v["step_ms"], None)
    return pts


WSET, DSET = os.environ.get("FN_WSET", "g2w"), os.environ.get("FN_DSET", "")
ROUNDS = (1, 2, 3)


def psi_flags():  # PSI_WINDOW lines from the lane's timing log: tag -> (full_pct, flagged)
    out = {}
    for line in open(os.path.join(T, "..", "..", "logs", "timing", "timing_idle.log")):
        m = re.search(r"PSI_WINDOW (\S+) full_pct ([\d.]+)", line)
        if m:
            out[m.group(1)] = (float(m.group(2)), float(m.group(2)) > 2)
    return out


def table_w():
    runs = {a: [load(f"t_{WSET}_{a}_r{r}") for r in ROUNDS] for a in ARMS}
    keys = list(w_points(runs["v"][0]).keys())
    brk_modes = {k: m for k, (_, m) in w_points(runs["breakable"][0]).items()}
    print("| point | fullnc | breakable | ours | fullnc / ours | breakable / ours | flags |")
    print("|---|---|---|---|---|---|---|")
    geo = {"fullnc": [], "breakable": []}
    for k in keys:
        cells, med, flags = {}, {}, []
        for a in ARMS:
            vals = [w_points(d)[k][0] for d in runs[a] if d]
            med[a], cells[a] = fmt(vals)
            if dev_flag(vals):
                flags.append(f"{a} round >5% off")
        brk_inv = k in INVALID_BRK
        mode = brk_modes.get(k)
        bcell = cells["breakable"] + (" INVALID" if brk_inv else "") + (f" ({next(iter(mode))})" if mode and not brk_inv else "")
        r_f = med["fullnc"] / med["v"]
        r_b = "-" if brk_inv else f"{med['breakable'] / med['v']:.2f}"
        if not k.startswith("alt"):
            geo["fullnc"].append(r_f)
            if not brk_inv:
                geo["breakable"].append(med["breakable"] / med["v"])
        print(f"| {k} | {cells['fullnc']} | {bcell} | {cells['v']} | {r_f:.2f} | {r_b} | {'; '.join(flags)} |")
    g = lambda xs: math.exp(statistics.fmean(math.log(x) for x in xs)) if xs else float("nan")
    print(f"\nGeomean over prefill + mixed points: fullnc / ours {g(geo['fullnc']):.2f} ({len(geo['fullnc'])} points); "
          f"breakable / ours {g(geo['breakable']):.2f} ({len(geo['breakable'])} valid points)")


def table_d_set():  # decode: FN_DSET rounds 1-3 at the grid; breakable INVALID where it did not pad (unpadded capture size, BREAKABLE_PLE.md)
    runs = {a: [load(f"t_{DSET}_{a}_r{r}") for r in ROUNDS] for a in ARMS}
    grid = list(runs["v"][0]["dcurve"])
    print("| ctx_bs | fullnc | breakable | ours | fullnc / ours | breakable / ours | flags |")
    print("|---|---|---|---|---|---|---|")
    gf, gb = {"1024": [], "8192": []}, {"1024": [], "8192": []}
    for k in grid:
        cells, med, flags = {}, {}, []
        for a in ARMS:
            vals = [d["dcurve"][k]["step_ms"] for d in runs[a] if d]
            med[a], cells[a] = fmt(vals)
            if dev_flag(vals):
                flags.append(f"{a} round >5% off")
        brk_valid = all((d["dcurve"][k].get("pad_tokens_mean") or 0) > 0 for d in runs["breakable"] if d)
        rb = f"{med['breakable'] / med['v']:.2f}" if brk_valid else "-"
        bcell = cells["breakable"] + ("" if brk_valid else " INVALID")
        ctx = k.split("_")[0]
        gf[ctx].append(med["fullnc"] / med["v"])
        if brk_valid:
            gb[ctx].append(med["breakable"] / med["v"])
        print(f"| {k} | {cells['fullnc']} | {bcell} | {cells['v']} | {med['fullnc'] / med['v']:.2f} | {rb} | {'; '.join(flags)} |")
    g = lambda xs: math.exp(statistics.fmean(math.log(x) for x in xs)) if xs else float("nan")
    for ctx in gf:
        print(f"\nctx {ctx}: geomean fullnc / ours {g(gf[ctx]):.3f} ({len(gf[ctx])} points); breakable / ours {g(gb[ctx]):.3f} ({len(gb[ctx])} valid padded points)")


def table_d():
    r1 = {a: load(f"t_g0d_{a}_r1") for a in ARMS}
    r23 = {a: [load(f"t_g0d20_{a}_r{r}") for r in (2, 3)] for a in ARMS}
    grid = [k for k in r23["v"][0]["dcurve"]]
    print("| ctx_bs | fullnc | breakable | ours | fullnc / ours | breakable / ours | flags |")
    print("|---|---|---|---|---|---|---|")
    for k in grid:
        cells, med, flags = {}, {}, []
        for a in ARMS:
            vals = [r1[a]["dcurve"][k]["step_ms"]] + [d["dcurve"][k]["step_ms"] for d in r23[a] if d]
            med[a], cells[a] = fmt(vals)
            if dev_flag(vals):
                flags.append(f"{a} round >5% off")
        print(f"| {k} | {cells['fullnc']} | {cells['breakable']} | {cells['v']} | {med['fullnc'] / med['v']:.2f} | "
              f"{med['breakable'] / med['v']:.2f} | {'; '.join(flags)} |")
    for ctx in ("1024", "8192"):
        ks = [k for k in r1["v"]["dcurve"] if k.startswith(ctx + "_")]
        rf = [r1["fullnc"]["dcurve"][k]["step_ms"] / r1["v"]["dcurve"][k]["step_ms"] for k in ks]
        rb = [r1["breakable"]["dcurve"][k]["step_ms"] / r1["v"]["dcurve"][k]["step_ms"] for k in ks]
        g = lambda xs: math.exp(statistics.fmean(math.log(x) for x in xs))
        print(f"\nd r1 full curve ctx {ctx} ({len(ks)} points, bs {ks[0].split('_')[1]}-{ks[-1].split('_')[1]}): geomean fullnc / ours "
              f"{g(rf):.3f} (min {min(rf):.2f}, max {max(rf):.2f}); breakable / ours {g(rb):.3f} (min {min(rb):.2f}, max {max(rb):.2f})")


def full_curve_csv():
    r1 = {a: load(f"t_g0d_{a}_r1") for a in ARMS}
    with open(os.path.join(T, "g0d_r1_curve.csv"), "w") as f:
        f.write("ctx,bs,fullnc_ms,breakable_ms,ours_ms\n")
        for k in r1["v"]["dcurve"]:
            c, b = k.split("_")
            f.write(f"{c},{b}," + ",".join(f"{r1[a]['dcurve'][k]['step_ms']:.3f}" for a in ARMS) + "\n")


if __name__ == "__main__":
    if DSET:  # finals: FN_WSET / FN_DSET sets, rounds 1-3 each, PSI per run
        print(f"## w (prefill / mixed), set {WSET}, rounds 1-3, step ms median [min-max]\n")
        table_w()
        print(f"\n## d (decode), set {DSET}, rounds 1-3 at the 20-point grid, step ms median [min-max]\n")
        table_d_set()
        psi = psi_flags()
        tags = [f"t_{st}_{a}_r{r}" for st in (WSET, DSET) for a in ARMS for r in ROUNDS]
        print("\nPSI over each run window (full %, flagged > 2%): " + ", ".join(f"{t[2:]} {psi[t][0]:.2f}{' FLAG' if psi[t][1] else ''}" for t in tags if t in psi))
    else:
        print("## w (prefill / mixed), gb300-box GPU 2, set g2w, 3 rounds, step ms median [min-max]\n")
        table_w()
        print("\n## d (decode), gb300-box GPU 0: d r1 (set g0d, full curve) + d20 r2 / r3 (set g0d20), step ms median [min-max] over 3 rounds at the 20-point grid\n")
        table_d()
        full_curve_csv()
