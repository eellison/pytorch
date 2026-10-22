import pickle, sys, collections, json
d = pickle.load(open(sys.argv[1], "rb"))
E = d["entries"]
print("bindings", len(E))
c = collections.Counter(); ms = collections.defaultdict(set)
for k, b in E:
    g = (k[1], str(k[2][0]), tuple(k[3][1]), k[4][1], k[4][0])
    c[g] += 1; ms[g].add(k[3][0][0])
for g, n in c.most_common(): print(n, g, "M range", min(ms[g]), max(ms[g]))
