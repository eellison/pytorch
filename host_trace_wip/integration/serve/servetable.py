# Table of a sweep: python servetable.py TAG [shape ...]. Per shape and concurrency: output tok/s, TTFT / TPOT / ITL
# median and p99 for arms default and V, the V/default ratio, and for V the exact keys first seen at that point and the
# host seconds of their first calls (the out-of-band learns), from the ctr files' first_calls records.
import json
import os
import sys

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
tag = sys.argv[1]
shapes = sys.argv[2:] or ["1k1k", "8k1k"]


def load(name):
    p = os.path.join(OUT, name)
    return json.load(open(p)) if os.path.exists(p) else None


for shape in shapes:
    print(f"\n### {tag} {shape}\n")
    print("| conc | default tok/s | V tok/s | V/default | TTFT med ms d / V | TPOT med ms d / V | TPOT p99 ms d / V | ITL p99 ms d / V | V new keys | V first-call s | V traces |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    prev = load(f"{tag}_V_ctr_warm.json")
    seen = {r["key"] for r in prev["first_calls"]} if prev and "first_calls" in prev else set()
    for c in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        d, v = load(f"{tag}_default_{shape}_c{c}.json"), load(f"{tag}_V_{shape}_c{c}.json")
        if d is None and v is None:
            continue
        ctr = load(f"{tag}_V_ctr_{shape}_c{c}.json")
        new, first_s, traces = "", "", ""
        if ctr and "first_calls" in ctr:
            recs = [r for r in ctr["first_calls"] if r["key"] not in seen]
            new, first_s = len(recs), round(sum(r["calls"][0][0] for r in recs if r["calls"]), 1)
            seen |= {r["key"] for r in recs}
            traces = ctr.get("traces")
        f = lambda x, k: f"{x[k]:.2f}" if x else "-"
        ratio = f"{v['output_throughput'] / d['output_throughput']:.2f}" if d and v else "-"
        print(f"| {c} | {f(d, 'output_throughput')} | {f(v, 'output_throughput')} | {ratio} | {f(d, 'median_ttft_ms')} / {f(v, 'median_ttft_ms')} | "
              f"{f(d, 'median_tpot_ms')} / {f(v, 'median_tpot_ms')} | {f(d, 'p99_tpot_ms')} / {f(v, 'p99_tpot_ms')} | {f(d, 'p99_itl_ms')} / {f(v, 'p99_itl_ms')} | {new} | {first_s} | {traces} |")
