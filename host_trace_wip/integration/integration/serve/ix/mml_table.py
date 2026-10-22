# max_model_len A/B (set mml): vLLM default / default-nc at 16384 vs 40960, decode TPOT at ~1K / ~8K for bs 1 / 8 / 64; delta vs
# run-to-run spread; startup and KV tokens.   python3 serve/ix/mml_table.py
import glob, json, os, statistics as st

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
runs = {a: [json.load(open(f)) for f in sorted(glob.glob(f"{HERE}/vllm/out/t_mml_{a}_r*.json"))] for a in ("default16", "default40", "fullnc16", "fullnc40")}
print("| arm | ctx | bs | 16384 ms (runs) | 40960 ms (runs) | delta % (40960 vs 16384) | spread % (max-min over runs, larger of the two) |")
print("|---|---|---|---|---|---|---|")
for base in ("default", "fullnc"):
    for ctx in ("1100", "8300"):
        for bs in ("1", "8", "64"):
            v = {m: [R["dec_ctx"][ctx][bs]["tpot_ms_mean"] for R in runs[f"{base}{m}"] if R["dec_ctx"].get(ctx, {}).get(bs)] for m in ("16", "40")}
            if not v["16"] or not v["40"]:
                continue
            m16, m40 = st.median(v["16"]), st.median(v["40"])
            spread = max((max(x) - min(x)) / st.median(x) * 100 for x in v.values())
            print(f"| {base} | {ctx} | {bs} | {m16:.3f} ({', '.join(f'{x:.2f}' for x in v['16'])}) | {m40:.3f} ({', '.join(f'{x:.2f}' for x in v['40'])}) | {(m40 / m16 - 1) * 100:+.1f} | {spread:.1f} |")
print("\n| arm | init s (median) | KV cache tokens | max capture size | graph memory GiB |")
print("|---|---|---|---|---|")
for a, v in runs.items():
    if v:
        print(f"| {a} | {st.median(R['init_s'] for R in v):.1f} | {st.median(R['mem']['kv_cache_tokens'] for R in v):.0f} | {v[0]['cg_config']['max_capture_size']} | {st.median((R.get('capture', {}).get('graph_bytes') or 0) / 2**30 for R in v):.2f} |")
