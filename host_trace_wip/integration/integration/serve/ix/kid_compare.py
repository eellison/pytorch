# Kernel-identity check: kernel-name multisets of one profiled full step per class (decode 64x64, prefill 1x512, mixed 65x1088), ours vs
# vLLM on the same forward (vcsl vs default, vrs vs fullnc); names present on one side only, and count differences.
#   python3 serve/ix/kid_compare.py
import collections, glob, json, os, re

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = f"{HERE}/serve/out/kid"


def names(path):
    ev = json.load(open(path))["traceEvents"]
    c = collections.Counter()
    for e in ev:
        if e.get("ph") == "X" and e.get("cat") == "kernel":
            c[re.sub(r"\s+", " ", e["name"])[:160]] += 1
    return c


for ours, base in (("vcsl", "default"), ("vrs", "fullnc")):
    print(f"\n## {ours} vs {base}")
    for cls in ("64x64", "1x512", "65x1088"):
        po, pb = glob.glob(f"{D}/{ours}/{ours}_{cls}.json"), glob.glob(f"{D}/{base}/{base}_{cls}.json")
        if not po or not pb:
            print(f"### {cls}: missing trace ({'ours' if not po else base})")
            continue
        o, b = names(po[0]), names(pb[0])
        only_o = sorted(set(o) - set(b))
        only_b = sorted(set(b) - set(o))
        diff = {k: (o[k], b[k]) for k in set(o) & set(b) if o[k] != b[k]}
        print(f"### {cls}: kernels ours {sum(o.values())} ({len(o)} names), vLLM {sum(b.values())} ({len(b)} names)")
        for k in only_o:
            print(f"  only ours  x{o[k]}: {k}")
        for k in only_b:
            print(f"  only vLLM  x{b[k]}: {k}")
        for k, (x, y) in sorted(diff.items()):
            print(f"  count differs ours {x} vs vLLM {y}: {k}")
