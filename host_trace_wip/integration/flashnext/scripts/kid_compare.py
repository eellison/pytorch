"""Kernel identity per timed point from the drivers' own R["kid"] (drive_sweep27 DRIVE_KID, default on: one profiled step after
each point's measured window, every arm and round): ours (v, clean replays only) against each stock arm, key by key.
A difference is LABELLED PADDING when the stock arm padded that point (its summary has pad_tokens_mean > 0: vLLM pads to a
capture size, we don't) and every differing kernel is a size-dependent library choice (cuBLAS / cuBLASLt nvjet, split-K
reduce, gemv; FlashInfer MoE batched-GEMM tactic / routing / finalize variant; CuTe skinny GEMM vs cuBLAS tier). Anything else
is a DIFF and voids the point. At points the stock arm did not pad, launch shapes (name + grid + block, the flashnext
driver snapshot) must match too: a same-name config change is "graph_settings" (labelled, like padding; user 10-14) when
the stock arm ran the point in a CUDA graph, else DIFF_SHAPES (voids the point). Writes the per-point verdict JSON next to OURS (for the table tool).
    python3 scripts/kid_compare.py OURS.json STOCK.json [STOCK2.json ...]"""
import collections
import json
import re
import sys

LIB = re.compile(r"nvjet|cublas|splitKreduce|gemv|dot_kernel|reduce_1Block|^bmm_|routingIndices|routing::|finalizeKernel|CuteSkinnyGemm", re.I)


def summary_of(R, key):
    point = key.rsplit("_", 1)[0]
    if point.startswith("d") and point[1:].isdigit():
        ctx, bs = point[1:], key.rsplit("_", 1)[1].split("x")[0]
        return R.get("dcurve", {}).get(f"{ctx}_{bs}", {})
    if point.startswith("m") and "x" in point:  # ix mixed: R["ix"]["mixed_<P>x<ND>[@LO-HI]"]
        return next((v for k, v in R.get("ix", {}).items() if k.startswith(f"mixed_{point[1:]}")), {})
    if point.startswith("a") and "_" in point:  # alternating decode: R["ix"]["alt_<A>_<B>"]
        return R.get("ix", {}).get(f"alt_{point[1:]}", {})
    if point.startswith("p"):
        return R.get("prefill", {}).get(point[1:], R.get("prefill", {}).get(int(point[1:]) if point[1:].isdigit() else point, {}))
    return {}


ours_path, *stocks = sys.argv[1:]
O = json.load(open(ours_path))
verdicts = {}
counts = collections.Counter()
for sp in stocks:
    S = json.load(open(sp))
    arm = S.get("arm") or sp
    for key, ov in sorted(O.get("kid", {}).items()):
        if not ov.get("clean", True) or ov.get("path") not in (None, "replay"):
            counts[(arm, "ours not a clean replay")] += 1
            continue
        sv = S.get("kid", {}).get(key)
        if sv is None:
            counts[(arm, "missing in stock")] += 1
            continue
        # graph memset / memcpy nodes show as "kernels" named memset32 / memcpy32_post / memcpy128 in a vLLM graph and as
        # memset / memcpy activity in ours: compared apart (memops), not as kernels
        mem = lambda k: {n: c for n, c in k.items() if not re.match(r"memset|memcpy", n)}
        ko, ks = collections.Counter(mem(ov["kernels"])), collections.Counter(mem(sv["kernels"]))
        padded = (summary_of(S, key).get("pad_tokens_mean") or 0) > 0
        lo, ls = collections.Counter(mem(ov.get("launch_shapes", {}))), collections.Counter(mem(sv.get("launch_shapes", {})))
        if ko == ks and (padded or not lo or not ls or lo == ls):
            v = "identical"
        elif ko == ks:
            # same kernels at a point neither arm padded, other grids / blocks: a same-name config change. If the stock arm ran
            # that point in a CUDA graph, it is vLLM's graph-forced setting (worst-case max_seq_len, fixed splits), labelled like
            # padding (user 10-14: our arms keep per-call settings); in an eager stock step it voids the point.
            graphed = any(m in ("FULL", "PIECEWISE") for m in (summary_of(S, key).get("cg_modes") or {}))
            v = "graph_settings" if graphed else "DIFF_SHAPES"
            print(f"{v} {key} vs {arm}:")
            for n, c in list((lo - ls).items())[:6]:
                print(f"    only ours x{c}: {n[:170]}")
            for n, c in list((ls - lo).items())[:6]:
                print(f"    only stock x{c}: {n[:170]}")
        else:
            diff = list((ko - ks).keys()) + list((ks - ko).keys())
            # at a padded point the graph arm also fills its padding (a FillFunctor on the padded input rows)
            v = "padding" if padded and all(LIB.search(n) or "FillFunctor" in n for n in diff) else "DIFF"
            if v == "DIFF":
                print(f"DIFF {key} vs {arm}: ours {sum(ko.values())} kernels, stock {sum(ks.values())}, stock padded: {padded}")
                for n, c in (ko - ks).items():
                    print(f"    only ours x{c}: {n[:140]}")
                for n, c in (ks - ko).items():
                    print(f"    only stock x{c}: {n[:140]}")
        verdicts.setdefault(key, {})[arm] = v
        counts[(arm, v)] += 1
for (arm, v), n in sorted(counts.items()):
    print(f"SUMMARY vs {arm}: {v} {n}")
out = ours_path.removesuffix(".json") + "_kid_verdicts.json"
json.dump(verdicts, open(out, "w"), indent=1)
print("wrote", out)
