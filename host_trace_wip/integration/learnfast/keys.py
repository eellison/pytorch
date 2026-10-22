import pickle, sys, collections
d = pickle.load(open(sys.argv[1], "rb"))
print(d["meta"])
E = d["entries"]
print(len(E))
k0, b0 = E[0]
print(k0); print(b0)
c = collections.Counter()
for k, b in E:
    sizes = k[3]
    c[(k[1], tuple(sizes[1]), k[4], k[5])] += 1
for x, n in c.most_common(): print(n, x)
