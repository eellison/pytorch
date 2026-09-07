import json, collections
rows = [json.loads(l) for l in open("agent_space/quack_quant/results.jsonl") if l.strip()]
by = collections.defaultdict(dict)
for r in rows:
    by[(r["case"], r["M"], r["K"])][r["mode"]] = r
def us(r): return f"{r['median_us']:8.1f}" if r and "median_us" in r else "       -"
print(f"{'case':28s} {'shape':12s} {'eager':>9s} {'off':>9s} {'on':>9s} {'on_mk':>9s} | k(off/on/mk) nested | on/off | eq(on,off) eq(on,mk)")
for (case, M, K), modes in sorted(by.items(), key=lambda kv: (kv[0][1], kv[0][0])):
    e, off, on, mk = (modes.get(m) for m in ("eager", "off", "on", "on_mk"))
    if on is None and "on_dynamic" in modes:
        continue
    ks = "/".join(str(m["kernels"]) if m else "-" for m in (off, on, mk))
    ratio = f"{on['median_us']/off['median_us']:6.2f}x" if on and off else "      -"
    eq1 = (on["out_hash"] == off["out_hash"]) if on and off else "-"
    eq2 = (on["out_hash"] == mk["out_hash"]) if on and mk else "-"
    print(f"{case:28s} {M}x{K:<6d} {us(e)} {us(off)} {us(on)} {us(mk)} | {ks:12s} {on['nested'] if on else '-':>6} | {ratio} | {eq1!s:9s} {eq2!s}")
print("\n=== dynamic vs static (mxfp8_e4m3) ===")
for (case, M, K), modes in sorted(by.items(), key=lambda kv: (kv[0][1], kv[0][0])):
    if "on_dynamic" in modes:
        base = by[(case, M, K)]
        print(f"{case:28s} {M}x{K:<6d} static on={us(base.get('on'))} off={us(base.get('off'))} | dynamic on={us(modes['on_dynamic'])} (k={modes['on_dynamic']['kernels']}) off={us(modes['off_dynamic'])} (k={modes['off_dynamic']['kernels']})")
