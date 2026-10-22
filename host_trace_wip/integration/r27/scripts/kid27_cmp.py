"""Kernel identity (27B): kernel names per profiled full step (execute_model through sample_tokens), vLLM arm vs ours.
    python3 kid27_cmp.py DIR VLLM_TAG OURS_TAG       (DIR/<tag>_<reqs>x<tokens>.json from drive_sweep27 DRIVE_PROFILE_DIR)
Prints, per step class: kernel launches and distinct names on each side, names on one side only, and count differences."""
import collections, glob, json, os, re, sys

D, A, B = sys.argv[1:4]
norm = lambda k: re.sub(r"_object_at__[0-9a-fx_]+", "_object_at_", k)  # CuTe DSL names carry an object address


def kernels(p):
    ev = json.load(open(p))["traceEvents"]
    k = collections.Counter(norm(e["name"]) for e in ev if e.get("ph") == "X" and e.get("cat") == "kernel")
    mc = collections.Counter(e["cat"] for e in ev if e.get("ph") == "X" and e.get("cat") in ("gpu_memcpy", "gpu_memset"))
    gl = sum(1 for e in ev if e.get("ph") == "X" and e.get("cat") in ("cuda_runtime", "cuda_driver") and "GraphLaunch" in e["name"])
    return k, mc, gl


ok = True
for pa in sorted(glob.glob(os.path.join(D, f"{A}_*x*.json"))):
    cls = pa[len(os.path.join(D, A)) + 1:-5]
    pb = os.path.join(D, f"{B}_{cls}.json")
    if not os.path.exists(pb):
        print(f"== {cls}: no {B} trace")
        continue
    (ka, ma, ga), (kb, mb, gb) = kernels(pa), kernels(pb)
    only_a, only_b = sorted(set(ka) - set(kb)), sorted(set(kb) - set(ka))
    diff = sorted((k, ka[k], kb[k]) for k in set(ka) & set(kb) if ka[k] != kb[k])
    same = not only_a and not only_b and not diff
    ok &= same
    print(f"== {cls}: {A} {sum(ka.values())} kernels / {len(ka)} names / graph launches {ga} / {dict(ma)}; "
          f"{B} {sum(kb.values())} / {len(kb)} / {gb} / {dict(mb)}: {'IDENTICAL kernel multiset' if same else 'DIFFERENT'}")
    for k in only_a:
        print(f"   only {A}: x{ka[k]} {k[:150]}")
    for k in only_b:
        print(f"   only {B}: x{kb[k]} {k[:150]}")
    for k, x, y in diff:
        print(f"   count {x} vs {y}: {k[:150]}")
print("ALL IDENTICAL" if ok else "DIFFERENCES FOUND")
