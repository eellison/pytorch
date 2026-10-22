"""Audit of kid_compare's labels (coordinator 10-14) on one set's records: OURS.json STOCK.json [STOCK2.json ...].
(1) Every point a stock arm padded: each differing kernel classified as
      gemm     a GEMM whose kernel choice follows M (cuBLAS / cuBLASLt nvjet, split-K reduce, gemv dot / reduce, CuTe skinny GEMM)
      padfill  the graph arm's pad fill on padded input rows (FillFunctor)
      moe      a FlashInfer MoE kernel (batched-GEMM tactic, routing, finalize), whose choice follows the token count
      other    anything else (attention, GDN, norm, quant, an M-independent kernel with another count): makes the point DIFF
    M-independent kernels (all but gemm / padfill / moe) must have equal counts.
(2) Memory ops per point: graph memset / memcpy nodes, which both sides' traces show as kernels named memset* / memcpy*
    with a grid (block 512 / 256), compared by (name, grid, block) multiset; and over the decode ladder, every memset class
    whose size (grid) grows with bs on the stock side but stays constant on ours is flagged.
Writes OURS_kid_audit.json."""
import collections
import json
import re
import sys

GEMM = re.compile(r"nvjet|cublas|splitKreduce|gemv|dot_kernel|reduce_1Block|CuteSkinnyGemm", re.I)
MOE = re.compile(r"^bmm_|routingIndices|moe::dev::routing|routingCustom|finalizeKernel|moe::dev::finalize")
MEM = re.compile(r"^(memset|memcpy)")


def cls(name):
    if MEM.match(name):
        return "mem"
    if "FillFunctor" in name:
        return "padfill"
    if GEMM.search(name):
        return "gemm"
    if MOE.search(name):
        return "moe"
    return "other"


def pad_of(R, key):
    point, shape = key.rsplit("_", 1)
    if point.startswith("d") and point[1:].isdigit():
        s = R.get("dcurve", {}).get(f"{point[1:]}_{shape.split('x')[0]}", {})
    elif point.startswith("m") and "x" in point:
        s = next((v for k, v in R.get("ix", {}).items() if k.startswith(f"mixed_{point[1:]}")), {})
    elif point.startswith("a") and "_" in point:
        s = R.get("ix", {}).get(f"alt_{point[1:]}", {})
    elif point.startswith("p"):
        s = R.get("prefill", {}).get(point[1:], {})
    else:
        s = {}
    return (s.get("pad_tokens_mean") or 0) > 0


O = json.load(open(sys.argv[1]))
out = {"padding_audit": {}, "memops": {}, "ladder_memset_flags": [],
       "notes": ["padding (MoE variant by token count): counted as padding (coordinator 10-14). At every unpadded point the MoE "
                 "kernels match exactly, so both sides run the same selection function and only the input count differs."]}
counts = collections.Counter()
for sp in sys.argv[2:]:
    S = json.load(open(sp))
    arm = S.get("arm")
    for key, ov in sorted(O.get("kid", {}).items()):
        sv = S.get("kid", {}).get(key)
        if sv is None or not ov.get("clean", True) or ov.get("path") not in (None, "replay"):
            continue
        ko, ks = collections.Counter(ov["kernels"]), collections.Counter(sv["kernels"])
        # (2) memops by (name, grid, block)
        mo = collections.Counter({k: c for k, c in ov.get("launch_shapes", {}).items() if MEM.match(k)})
        ms = collections.Counter({k: c for k, c in sv.get("launch_shapes", {}).items() if MEM.match(k)})
        memv = "equal" if mo == ms else "differ"
        out["memops"].setdefault(key, {})[arm] = {"verdict": memv, "only_ours": dict(mo - ms), "only_stock": dict(ms - mo)}
        counts[(arm, "memops " + memv)] += 1
        if not pad_of(S, key):
            continue
        diff = {n: ("ours", c) for n, c in (ko - ks).items()} | {n: ("stock", c) for n, c in (ks - ko).items()}
        by = collections.defaultdict(list)
        for n, (side, c) in diff.items():
            by[cls(n)].append([side, c, n[:150]])
        indep_o = collections.Counter({n: c for n, c in ko.items() if cls(n) in ("other",)})
        indep_s = collections.Counter({n: c for n, c in ks.items() if cls(n) in ("other",)})
        verdict = "padding" if not by.get("other") and indep_o == indep_s else "DIFF"
        if verdict == "padding" and by.get("moe"):
            verdict = "padding (MoE variant by token count)"
        out["padding_audit"].setdefault(key, {})[arm] = {"verdict": verdict, **{k: v for k, v in by.items() if k != "mem"}}
        counts[(arm, verdict)] += 1
    # ladder: memset classes per bs on each side (decode keys d<ctx>_<bs>x<bs>)
    for ctx in sorted({k.split("_")[0] for k in O.get("kid", {}) if re.match(r"d\d+_", k)}):
        def series(K):
            ser = {}
            for k, v in K.get("kid", {}).items():
                m = re.match(rf"{ctx}_(\d+)x\1$", k)
                if m:
                    ser[int(m.group(1))] = sorted((n, c) for n, c in v.get("launch_shapes", {}).items() if n.startswith("memset"))
            return ser
        so, ss = series(O), series(S)
        common = sorted(set(so) & set(ss))
        if len(common) < 2:
            continue
        o_const = len({tuple(so[b]) for b in common}) == 1
        s_const = len({tuple(ss[b]) for b in common}) == 1
        if o_const and not s_const:
            out["ladder_memset_flags"].append({"arm": arm, "ctx": ctx, "ours_constant": so[common[0]], "stock_bs1": ss[common[0]], "stock_bsmax": ss[common[-1]]})
        counts[(arm, f"ladder {ctx}: ours memsets {'constant' if o_const else 'vary with bs'}, stock {'constant' if s_const else 'vary with bs'}")] += 1
for (arm, what), n in sorted(counts.items()):
    print(f"AUDIT vs {arm}: {what}: {n}")
print("ladder memset flags:", out["ladder_memset_flags"])
path = sys.argv[1].removesuffix(".json") + "_kid_audit.json"
json.dump(out, open(path, "w"), indent=1)
print("wrote", path)
