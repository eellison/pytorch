# peaks (main 10-14): vrs vs fullnc per-step peak allocated above the step's start and reserved growth, by step kind and token
# bucket; memprobe's init / after-warm-up / end. python3 serve/ix/peaks_table.py
import collections, json, os

G = 2 ** 30
for a in ("vrs", "fullnc"):
    p = f"vllm/out/t_peaks_{a}_r1.json"
    if not os.path.exists(p):
        continue
    R = json.load(open(p))
    S = R.get("peaks", {}).get("steps", [])
    M = R.get("memprobe", {})
    print(f"\n### {a}: {len(S)} steps after the warm-up; reserved init {M.get('init', {}).get('reserved', 0) / G:.2f}, after warm-up "
          f"{M.get('after_warmup', {}).get('reserved', 0) / G:.2f}, end {M.get('end', {}).get('reserved', 0) / G:.2f} GiB; allocated end "
          f"{M.get('end', {}).get('allocated', 0) / G:.2f}; reserved before install {R['reserved_after_init_mib'] / 1024:.2f}\n")
    agg = collections.defaultdict(lambda: [0, 0.0, 0.0])
    for r_, t_, pk, dres, res in S:
        kind = "decode" if r_ == t_ else ("prefill" if r_ == 1 else "mixed")
        b = 1 << max(0, (t_ - 1).bit_length())
        k = (kind, b)
        agg[k][0] += 1
        agg[k][1] = max(agg[k][1], pk)
        agg[k][2] += max(0, dres)
    print("| step kind | tokens up to | steps | max peak above start GiB | reserved growth in these steps GiB |")
    print("|---|---|---|---|---|")
    for (kind, b), (n, pk, g) in sorted(agg.items()):
        print(f"| {kind} | {b} | {n} | {pk / G:.2f} | {g / G:.2f} |")
