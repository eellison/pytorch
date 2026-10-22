"""Kernel identity at every step class (with_kprof.py files of the arms of one set): for each class ours (v) ran, its GPU
kernel-name multiset against each stock arm's. Graph arms pad decode batches to a capture size (vLLM pads, we don't): a
difference there whose kernels differ only in size-dependent library choices (cuBLAS nvjet tile names, a MoE tactic) is the
labelled padding difference; every differing class is listed with its kernels so it can be read.
    python3 scripts/kernel_identity_all.py OURS.json STOCK1.json [STOCK2.json ...]"""
import collections
import json
import os
import sys


def label(name):
    import re
    return re.sub(r"\s+", " ", name)[:140]


ours_path, *stock_paths = sys.argv[1:]
ours = json.load(open(ours_path))
stocks = {os.path.basename(p).removesuffix(".json"): json.load(open(p)) for p in stock_paths}
summary = collections.Counter()
for key in sorted(ours, key=lambda k: (k.split("|")[0], [int("".join(c for c in x if c.isdigit()) or 0) for x in k.split("|")[1:]])):
    ko = collections.Counter(ours[key]["kernels"])
    for arm, st in stocks.items():
        if key not in st:
            summary[(arm, "missing")] += 1
            print(f"{key:40s} vs {arm}: not profiled there")
            continue
        ks = collections.Counter(st[key]["kernels"])
        if ko == ks:
            summary[(arm, "identical")] += 1
            continue
        summary[(arm, "differ")] += 1
        print(f"{key:40s} vs {arm}: ours {sum(ko.values())} kernels, {arm} {sum(ks.values())}")
        for n, c in (ko - ks).items():
            print(f"    only ours x{c}: {label(n)}")
        for n, c in (ks - ko).items():
            print(f"    only {arm} x{c}: {label(n)}")
for (arm, what), n in sorted(summary.items()):
    print(f"SUMMARY vs {arm}: {what} {n}")
print("classes profiled by ours:", len(ours))
