"""Kernel identity across timing arms (KIND=id of round_fn_line.sh): per decode batch size, the multiset of GPU kernel names
in the profiled decode steps (drive_sweep27 --profile-decode: Kineto traces <arm>_bs<B>_<k>.json) of each arm, compared with
ours (V). Kernel names carry template arguments, not grids, so a graph arm's padded size (bs 13 runs the 16-row graph) still
compares by name. Memcpy / memset nodes are reported separately (graph input staging differs by design).
    python3 scripts/kernel_identity.py out/timing/prof_<SET>_fullnc out/timing/prof_<SET>_breakable out/timing/prof_<SET>_v"""
import collections
import glob
import json
import os
import re
import sys


def kernels(path):
    ev = json.load(open(path))["traceEvents"]
    k = collections.Counter(e["name"] for e in ev if e.get("ph") == "X" and e.get("cat") == "kernel")
    m = collections.Counter(e["cat"] for e in ev if e.get("ph") == "X" and e.get("cat") in ("gpu_memcpy", "gpu_memset"))
    return k, m


arms = {}
for d in sys.argv[1:]:
    for f in sorted(glob.glob(os.path.join(d, "*_bs*_*.json"))):
        arm, bs, k = re.match(r"(.+)_bs(\d+)_(\d+)\.json", os.path.basename(f)).groups()
        arms.setdefault(os.path.basename(d.rstrip("/")), {}).setdefault(int(bs), []).append(kernels(f))
names = list(arms)
ours = next(n for n in names if n.endswith("_v"))
bad = 0
for bs in sorted({b for a in arms.values() for b in a}):
    ko, mo = arms[ours].get(bs, [(None, None)])[-1]
    for n in names:
        if n == ours or bs not in arms[n] or ko is None:
            continue
        kn, mn = arms[n][bs][-1]
        only_ours, only_theirs = ko - kn, kn - ko
        same = not only_ours and not only_theirs
        bad += not same
        print(f"bs {bs:3d} {ours} vs {n}: kernels {sum(ko.values())} vs {sum(kn.values())}; identical multiset: {same}; "
              f"memcpy/memset {dict(mo)} vs {dict(mn)}")
        for name, c in list(only_ours.items())[:8]:
            print(f"    only {ours} x{c}: {name[:150]}")
        for name, c in list(only_theirs.items())[:8]:
            print(f"    only {n} x{c}: {name[:150]}")
print("KERNEL_IDENTITY", "PASS" if not bad else f"FAIL ({bad} comparisons differ)")
