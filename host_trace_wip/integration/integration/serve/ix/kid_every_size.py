# Kernel identity at every size (kidall): per point, the kernel multiset of (name, grid, block) of one profiled step (DRIVE_KREC), ours
# vs the stock arm on the same forward. Classes per point: SAME (identical multisets), SAME_NAMES (names equal, launch shapes differ;
# "padding" when the stock arm padded that step, DIFF_SHAPES otherwise), DIFF_NAMES (a kernel name on one side only, or a count differs).
#   python3 serve/ix/kid_every_size.py [SET=kidall]
import collections, glob, json, os, re, sys

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SET = sys.argv[1] if len(sys.argv) > 1 else "kidall"
R = {}
for f in glob.glob(f"{HERE}/vllm/out/t_{SET}_*_r1.json"):
    R[os.path.basename(f)[len(f"t_{SET}_"):-len("_r1.json")]] = json.load(open(f))
PAIRS = [("default", "vcsl"), ("default", "vcslm"), ("fullnc", "vrs"), ("fullnc", "vrsm"), ("bnc", "vrs"), ("bnc", "vrsm")]


def points(r):
    for ctx, v in r.get("dec_ctx", {}).items():
        for bs, d in v.items():
            yield (f"decode ~{int(ctx) // 1000}K bs {bs}", d)
    for L, d in r.get("prefill", {}).items():
        if str(L).isdigit():
            yield (f"prefill {L}", d)
    for k, d in r.get("ix", {}).items():
        if k.startswith("mixed_"):
            yield (k[6:], d)


def names(m):
    c = collections.Counter()
    for k, n in (m or {}).items():
        c[k.split("|grid=")[0]] += n
    return c


norm = lambda s: re.sub(r"\s+", " ", s)


PAD_FAMILIES = ("nvjet", "void cublasLt", "void at::native::unrolled_elementwise_kernel<at::native::FillFunctor<bool>",
                "void at::native::vectorized_elementwise_kernel<2, at::native::FillFunctor<bool>", "void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<bool>")


def category(k, point, padded, name_count_differs):
    """Why a kernel entry differs: an explained class, or None (unexplained). At a padded point a launch-shape difference is expected for
    any kernel; a kernel-name difference only for cuBLAS (another M picks another kernel), the padding fills, memsets / memcpys and attention."""
    n = k.split("|grid=")[0]
    if padded and (not name_count_differs or n.startswith(PAD_FAMILIES)):
        return "padding (vLLM ran this step at its padded capture size)"
    if n.startswith("fmha") and point.startswith("decode"):
        return "worst-case max_seq_len (vLLM's static graphs force max_model_len; ours uses the actual value)"
    if n.startswith(("memset", "memcpy", "GPU_MEMSET", "GPU_MEMCPY")):
        return "memset / memcpy shown as a kernel on one side (graph memset/memcpy node vs captured op; the multi-CTA counter memset follows the attention mode)"
    return None


for stock, ours in PAIRS:
    if stock not in R or ours not in R:
        continue
    P_s, P_o = dict(points(R[stock])), dict(points(R[ours]))
    cls, unexplained, expl = collections.Counter(), [], collections.Counter()
    for p, ds in P_s.items():
        do = P_o.get(p)
        if not do or not ds.get("kernels") or not do.get("kernels"):
            cls["missing record"] += 1
            continue
        ks, ko = collections.Counter(ds["kernels"]), collections.Counter(do["kernels"])
        if ks == ko:
            cls["SAME"] += 1
            continue
        padded = (ds.get("pad_tokens_mean") or 0) > 0
        diff = [k for k in set(ks) | set(ko) if ks[k] != ko[k]]
        ns, no = names(ks), names(ko)
        ncd = lambda k: ns[k.split("|grid=")[0]] != no[k.split("|grid=")[0]]
        cats = {category(k, p, padded, ncd(k)) for k in diff}
        if None in cats:
            cls["DIFF (unexplained)"] += 1
            unexplained.append((p, sorted(norm(k)[:150] for k in diff if category(k, p, padded, ncd(k)) is None)[:6]))
        else:
            cls["explained: " + " + ".join(sorted(c.split(" (")[0] for c in cats))] += 1
            for c in cats:
                expl[c] += 1
    print(f"\n## {ours} vs {stock} ({len(P_s)} points)")
    for k, v in sorted(cls.items()):
        print(f"- {k}: {v}")
    for c, n in expl.most_common():
        print(f"  - {c}: {n} points")
    for p, d in unexplained[:40]:
        print(f"  UNEXPLAINED {p}: {d}")
