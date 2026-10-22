# compare.py A.pkl B.pkl: per key, whether the saved bindings are equal (dataclass ==, images bytes included); for the
# differing ones, which fields differ, and whether they are equal with loose (host) byte ranges masked.
import collections, dataclasses, pickle, sys
def load(p):
    d = pickle.load(open(p, "rb"))
    return d["meta"], dict(d["entries"])
ma, A = load(sys.argv[1]); mb, B = load(sys.argv[2])
print("meta equal:", {k: ma[k] == mb[k] for k in ma})
common = A.keys() & B.keys()
print("keys", len(A), len(B), "common", len(common))
def masked(b):
    nodes = []
    for n in b.nodes:
        if hasattr(n, "images"):
            imgs = [bytearray(i) for i in n.images]
            for p, o, s in n.loose:
                imgs[p][o:o + s] = bytes(s)
            nodes.append(dataclasses.replace(n, images=tuple(map(bytes, imgs)), loose=()))
        else:
            nodes.append(n)
    return dataclasses.replace(b, nodes=tuple(nodes))
eq = 0; eqm = 0; why = collections.Counter(); ex = None
for k in common:
    a, b = A[k], B[k]
    if a == b:
        eq += 1; continue
    if masked(a) == masked(b):
        eqm += 1
    f = []
    if len(a.nodes) != len(b.nodes): f.append("node count")
    else:
        for x, y in zip(a.nodes, b.nodes):
            for fld in dataclasses.fields(x):
                if getattr(x, fld.name) != getattr(y, fld.name): f.append(type(x).__name__ + "." + fld.name)
    for fld in ("scratch", "rng_increment", "owner"):
        if getattr(a, fld) != getattr(b, fld): f.append(fld)
    why[tuple(sorted(set(f)))] += 1
    ex = ex or k
print("identical", eq, "equal with loose bytes masked", eqm, "differ", len(common) - eq)
for w, n in why.most_common(10): print(" ", n, w)
if ex: print("example", ex[3])
