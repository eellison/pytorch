# memset spot check table: each checked replay's held memset widths vs FlashInfer's counter formula
# round_up(max(batch * 32, 152), 8) * 4 (Qwen3-8B: 32 query heads; GB300: 152 SMs). Decode steps: batch = decodes; mixed: the decode
# counter at batch = decodes, the prefill counter at batch = prefill requests (1).
#   python3 serve/ix/memset_spot_table.py vllm/out/t_memspot_vcsl_r1.json
import collections, json, sys

R = json.load(open(sys.argv[1]))
f = lambda b: -(-max(b * 32, 152) // 8) * 8 * 4
bad, n = [], 0
print("| step (reqs, tokens, decodes) | entry, variant | held widths (count) | formula: decode / prefill | match |")
print("|---|---|---|---|---|")
for c in R.get("memset_check", []):
    reqs, toks, nd = c["cls"]
    byv = collections.defaultdict(collections.Counter)
    for r in c["rows"]:
        byv[(r[0], r[1])][(r[3], r[6] == "placeholder")] += 1
    for (e, v), ws in byv.items():
        want = {f(nd)} if reqs == toks else {f(nd), f(reqs - nd)}
        got = {w for (w, ph) in ws}
        ok = got <= want and not any(ph for (_, ph) in ws)
        n += 1
        if not ok:
            bad.append((c["cls"], e, v, dict(ws)))
        print(f"| {reqs}, {toks}, {nd} | {e}, {v} | " + ", ".join(f"{w}{' placeholder' if ph else ''} (x{k})" for (w, ph), k in sorted(ws.items())) +
              f" | {f(nd)} / {f(reqs - nd) if reqs != toks else '-'} | {'yes' if ok else 'NO'} |")
print(f"\nMEMSPOT checked {n} (step, variant) groups; mismatches {len(bad)}: {bad[:10]}")
