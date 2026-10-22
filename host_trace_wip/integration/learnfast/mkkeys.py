# mkkeys.py SRC.pkl OUT.pkl MAXT MAXSEQS: the warm-up range's keys from SRC's key formulas: for each (op, N, K, layout) group of
# SRC, every M in [1, MAXT] (the lm_head group, M <= SRC's max there: [1, MAXSEQS]); each key SRC's key of that group with M
# replaced (strides follow the observed pattern; a size-1 dim keeps the stride SRC shows at M = 1).
import pickle, sys, collections
src, out, maxt, maxseqs = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
d = pickle.load(open(src, "rb"))
groups = collections.defaultdict(dict)
for k, b in d["entries"]:
    groups[(k[1], k[2], k[3][1], k[4][1])][k[3][0][0]] = k
keys = []
for g, byM in groups.items():
    top = maxt if max(byM) > maxseqs else maxseqs
    ref = byM[max(byM)]
    for M in range(1, top + 1):
        if M in byM:
            keys.append(byM[M]); continue
        (m0, kk), w, (_, n) = ref[3]
        sizes = ((M, kk), w, (M, n))
        keys.append((ref[0], ref[1], ref[2], sizes, ref[4], *ref[5:]))
    print(g[0], g[2], "Ms", top, "observed", len(byM), "strides at M=1", byM.get(1, ref)[4])
pickle.dump({"meta": d["meta"], "entries": [(k, None) for k in keys]}, open(out, "wb"))
print("keys", len(keys))
