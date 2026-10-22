"""Clean-window columns for the Flash-Next finals (coordinator 10-15; the MiMo lane's tools/clean_table.py + flag_runs.py logic).
A window (one prefill / mixed / alt / decode-curve point of one run) is clean unless host-pressure flagged: its memory "full" stall
(psi_full_s) exceeds 2% of its wall (n x median step), or its steps beyond 10x the median exceed 1% of the wall. Runs from before
the driver recorded per-window PSI (g1w r1 / r2) fall back to the run-level PSI_WINDOW line in logs/timing/timing_idle.log: a run
flagged there has every window flagged.
    python3 scripts/fn_clean.py SET [SET...]            per row: all-rounds median and clean median (n) per arm, ratios vs ours
    python3 scripts/fn_clean.py --check N SET [SET...]  exit 0 when every arm has >= N clean windows in every row
"""
import collections
import glob
import json
import os
import re
import statistics
import sys

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
T, LOG = os.path.join(D, "out", "timing"), os.path.join(D, "logs", "timing", "timing_idle.log")
ARMS = ("fullnc", "breakable", "v")
INVALID_BRK = {"prefill 64", "prefill 128", "prefill 256", "prefill 512", "prefill 1024",
               "mixed 256x16", "mixed 256x64", "mixed 512x16", "mixed 512x64"}
args = sys.argv[1:]
check = None
if args[0] == "--check":
    check, args = int(args[1]), args[2:]
run_psi = {}
for line in open(LOG):
    m = re.search(r"PSI_WINDOW (\S+) full_pct ([\d.]+)", line)
    if m:
        run_psi[m.group(1)] = float(m.group(2))


def flagged(w, run_flag):
    if w.get("psi_full_s") is None:
        return run_flag
    wall = (w.get("n") or 0) * w["step_ms"] / 1e3
    ex = w.get("outlier_excess_ms")
    return (wall > 0 and w["psi_full_s"] / wall > 0.02) or (ex is not None and ex / 1e3 > 0.01 * wall)


rows = collections.defaultdict(lambda: collections.defaultdict(lambda: ([], [])))  # row -> arm -> (clean, all)
pads = collections.defaultdict(list)  # breakable decode rows: pad_tokens_mean per run
order = []
for SET in args:
    for f in sorted(glob.glob(os.path.join(T, f"t_{SET}_*_r*.json"))):
        m = re.match(rf"t_{SET}_(.+)_r\d+\.json", os.path.basename(f))
        if not m or m.group(1) not in ARMS:
            continue
        arm, tag = m.group(1), os.path.basename(f)[:-5]
        j = json.load(open(f))
        run_flag = run_psi.get(tag, 0.0) > 2
        wins = [(f"prefill {k}", w) for k, w in (j.get("prefill") or {}).items()]
        for k, w in (j.get("ix") or {}).items():
            wins.append((f"mixed {k[6:].split('@')[0]}" if k.startswith("mixed_") else f"alt {k[4:].replace('_', '<->')}", w))
        wins += [(f"dc {k}", w) for k, w in (j.get("dcurve") or {}).items()]
        for row, w in wins:
            if not isinstance(w, dict) or not w.get("step_ms"):
                continue
            if row not in order:
                order.append(row)
            c, a = rows[row][arm]
            a.append(w["step_ms"])
            if not flagged(w, run_flag):
                c.append(w["step_ms"])
            if arm == "breakable" and row.startswith("dc "):
                pads[row].append(w.get("pad_tokens_mean") or 0)


def brk_valid(row):
    return row not in INVALID_BRK and (not row.startswith("dc ") or all(p > 0 for p in pads[row]))


if check is not None:
    short = [(row, a, len(rows[row][a][0])) for row in order for a in ARMS if (a != "breakable" or brk_valid(row)) and len(rows[row][a][0]) < check]
    print(f"clean-window check (>= {check}): {len(short)} cells short" + (f", e.g. {short[:6]}" if short else ""))
    sys.exit(0 if not short else 1)
med = lambda xs: statistics.median(xs) if xs else None
print("| row | fullnc all | fullnc clean (n) | breakable all | breakable clean (n) | ours all | ours clean (n) | fullnc / ours all | clean | breakable / ours all | clean | note |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for row in order:
    cell, ma, mc = {}, {}, {}
    for a in ARMS:
        c, al = rows[row][a]
        ma[a], mc[a] = med(al), med(c)
        cell[a] = (f"{ma[a]:.1f}" if ma[a] else "-", f"{mc[a]:.1f} ({len(c)})" if mc[a] else f"- (0)")
    note = []
    rf_a = ma["fullnc"] / ma["v"]
    rf_c = mc["fullnc"] / mc["v"] if mc["fullnc"] and mc["v"] else None
    if rf_c and abs(rf_c / rf_a - 1) > 0.03:
        note.append(f"fullnc ratio clean vs all {rf_c / rf_a - 1:+.1%}")
    if brk_valid(row):
        rb_a = f"{ma['breakable'] / ma['v']:.2f}"
        rb_c_v = mc["breakable"] / mc["v"] if mc["breakable"] and mc["v"] else None
        rb_c = f"{rb_c_v:.2f}" if rb_c_v else "-"
        if rb_c_v and abs(rb_c_v / (ma['breakable'] / ma['v']) - 1) > 0.03:
            note.append(f"breakable ratio clean vs all {rb_c_v / (ma['breakable'] / ma['v']) - 1:+.1%}")
        bcols = (cell["breakable"][0], cell["breakable"][1], rb_a, rb_c)
    else:
        bcols = (cell["breakable"][0] + " INVALID", "-", "-", "-")
    print(f"| {row} | {cell['fullnc'][0]} | {cell['fullnc'][1]} | {bcols[0]} | {bcols[1]} | {cell['v'][0]} | {cell['v'][1]} | "
          f"{rf_a:.2f} | {f'{rf_c:.2f}' if rf_c else '-'} | {bcols[2]} | {bcols[3]} | {'; '.join(note)} |")
# geomeans over the headline's point sets (prefill + mixed rows; breakable at its valid rows), all rounds and clean windows
import math
geo = lambda xs: math.exp(statistics.fmean(math.log(x) for x in xs)) if xs else float("nan")
pm = [r for r in order if r.startswith(("prefill ", "mixed "))]
if pm:
    for a in ("fullnc", "breakable"):
        pts = [r for r in pm if a == "fullnc" or brk_valid(r)]
        al = [med(rows[r][a][1]) / med(rows[r]["v"][1]) for r in pts]
        cl = [med(rows[r][a][0]) / med(rows[r]["v"][0]) for r in pts if rows[r][a][0] and rows[r]["v"][0]]
        print(f"\nGeomean {a} / ours over {len(pts)} prefill + mixed points: all rounds {geo(al):.2f}; clean windows {geo(cl):.2f} ({len(cl)} of {len(pts)} points have clean windows in both arms)")
