# merge.py OUT IN...: one bindings file of the shards' entries (meta of the first; each shard's meta must agree but for libraries)
import pickle, sys, os
out, ins = sys.argv[1], sys.argv[2:]
metas, entries = [], []
for p in ins:
    d = pickle.load(open(p, "rb")); metas.append(d["meta"]); entries += d["entries"]
assert all({k: v for k, v in m.items() if k != "libraries"} == {k: v for k, v in metas[0].items() if k != "libraries"} for m in metas)
keys = [k for k, _ in entries]; assert len(set(keys)) == len(keys), "duplicate keys across shards"
with open(out + ".tmp", "wb") as f:
    pickle.dump({"meta": metas[0], "entries": entries}, f)
os.replace(out + ".tmp", out)
print("merged", len(entries))
