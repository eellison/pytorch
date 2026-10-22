"""churn_stats27.py SET...: churn window stats per arm (median over rounds): wall ms per output token, steps, mixed-step fraction,
shape-change rate (step class differs from the previous step), vLLM's padded tokens per step, ours' graph nodes patched per step."""
import collections, glob, json, os, re, statistics as st, sys

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out", "timing")
rows = collections.defaultdict(lambda: collections.defaultdict(list))
for s in sys.argv[1:]:
    for f in sorted(glob.glob(os.path.join(D, f"t_{s}_*_r*.json"))):
        m = re.match(rf"t_{s}_(default|fullnc|breakable|breakablegb|vcvp|vsvp)_r\d+\.json", os.path.basename(f))
        if not m:
            continue
        for C, d in json.load(open(f)).get("cb", {}).items():
            r = rows[(int(C), m.group(1))]
            for k in ("wall_ms_per_token", "steps", "mixed_frac", "shape_changed_frac", "decode_bs_changed_frac", "pad_tokens_mean", "nodes_set_mean", "mem_nodes_set_mean"):
                if d.get(k) is not None:
                    r[k].append(d[k])
md = lambda r, k, fmt: (fmt % st.median(r[k])) if r.get(k) else "-"
print(f"{'C':>3s} {'arm':12s} {'ms/token':>9s} {'steps':>6s} {'mixed':>6s} {'shape chg':>9s} {'dec bs chg':>10s} {'pad tok/step':>12s} {'nodes patched/step':>18s}")
for (C, arm), r in sorted(rows.items()):
    print(f"{C:3d} {arm:12s} {md(r, 'wall_ms_per_token', '%9.4f')} {md(r, 'steps', '%6d')} {md(r, 'mixed_frac', '%6.3f')} {md(r, 'shape_changed_frac', '%9.3f')} "
          f"{md(r, 'decode_bs_changed_frac', '%10.3f')} {md(r, 'pad_tokens_mean', '%12.2f')} {md(r, 'nodes_set_mean', '%18.1f')}")
