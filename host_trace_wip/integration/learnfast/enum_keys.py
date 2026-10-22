# enum_keys.py --seed SEED.pkl --out KEYS.pkl --max-tokens T [--max-seqs S --seqs-n N [N ...]]
# The warm-up range's cuBLAS keys from the keys a short traced run harvested (SEED: a bindings file saved with
# ARMV_BINDINGS_SAVE, e.g. after the startup traces): the harvested sites' keys, grouped by everything but the token dim
# M (op incl. overload, dtypes incl. out dtype, the other sizes, layout, alignment classes, scalars, device, library state),
# each group enumerated over M in [1, T] ([1, S] for groups whose N is in --seqs-n: the logits GEMM over sampled rows).
# M is operand 0's dim 0 and the output's dim 0 (mm / addmm / mm.dtype_out); bmm: dim 1. A stride equal to the seed's M
# (an operand laid out (K, M)) follows M when the group's seeds show it varying with M; with one seed a stride equal to M
# that is not the stride of a size-1 dim is followed too, and reported. Keys only: whatever the replay then asks for that
# this set lacks is a counted miss (learned on first use), never a wrong binding.
import argparse, collections, pickle, sys
p = argparse.ArgumentParser()
p.add_argument("--seed", required=True, nargs="+")
p.add_argument("--out", required=True)
p.add_argument("--max-tokens", type=int, required=True)
p.add_argument("--max-seqs", type=int, default=0)
p.add_argument("--seqs-n", type=int, nargs="*", default=[])
a = p.parse_args()
entries, meta = [], None
for s in a.seed:
    d = pickle.load(open(s, "rb"))
    meta = meta or d["meta"]
    entries += [k for k, _ in d["entries"]]
def mdim(k):
    op = k[1] if isinstance(k[1], str) else k[1][0]
    return 1 if op.endswith("bmm") else 0
groups = collections.defaultdict(dict)
for k in entries:
    md, sizes = mdim(k), k[3]
    M = sizes[0][md]
    shape_rest = tuple(tuple(x for j, x in enumerate(z) if not (i in (0, len(sizes) - 1) and j == md)) for i, z in enumerate(sizes))
    groups[(k[0], k[1], k[2], shape_rest, k[5], k[6], k[7], k[8], len(sizes))][M] = k
out, report = [], []
for g, byM in groups.items():
    ref_M = max(byM)
    ref = byM[ref_M]
    md = mdim(ref)
    n_out = ref[3][-1][-1]
    top = a.max_seqs if (a.max_seqs and n_out in a.seqs_n) else a.max_tokens
    # strides that follow M: those equal to M at every seed of the group (not a size-1 dim's)
    follow = set()
    for i, (z, st) in enumerate(zip(ref[3], ref[4])):
        for j, s in enumerate(st):
            if s == ref_M and z[j] != 1 and all(byM[m][4][i][j] == m for m in byM):
                follow.add((i, j))
    if follow:
        report.append((g[1], ref[3], "strides following M", sorted(follow)))
    for M in range(1, top + 1):
        if M in byM:
            out.append(byM[M]); continue
        sizes = tuple(tuple(M if (i in (0, len(ref[3]) - 1) and j == md) else x for j, x in enumerate(z)) for i, z in enumerate(ref[3]))
        strides = tuple(tuple(M if (i, j) in follow else s for j, s in enumerate(st)) for i, st in enumerate(ref[4]))
        out.append((ref[0], ref[1], ref[2], sizes, strides, *ref[5:]))
    report.append((g[1], ref[3][1:], "M 1..%d" % top, "seeds", sorted(byM)[:4], len(byM)))
pickle.dump({"meta": meta, "entries": [(k, None) for k in out]}, open(a.out, "wb"))
for r in report: print(*r)
print("groups", len(groups), "keys", len(out))
