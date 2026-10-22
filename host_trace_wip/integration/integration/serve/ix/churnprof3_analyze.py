# churnprof3 (main 10-14): per-call wall of cuGraphExecKernelNodeSetParams by change type and stream state, per step class; per-step
# kernel packs / row switches / TMA counter deltas; ours' phase clocks.   python3 serve/ix/churnprof3_analyze.py vllm/out/t_churnprof3_vcsl_r1.json [...]
import collections, json, statistics as st, sys

for path in sys.argv[1:]:
    R = json.load(open(path))
    rows = R["cb"]["64"].get("ns_ph", [])
    cls, prev = {}, None
    for r in rows:
        cls[r[7]] = "mixed" if r[1] > r[0] else ("decode_changed" if prev and (prev[0], prev[1]) != (r[0], r[1]) else "decode_same")
        prev = r
    busy = {r[7]: (r[9] or [None, None])[1] for r in rows}
    print(f"\n## {path}  form_held={R.get('form_held')}\n")
    print("| class | steps | busy at forward | host ms | dispatch | evaluate | patch | launch us | node sets | packs | row switches | TMA deltas |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k in ("decode_same", "decode_changed", "mixed"):
        rs = [r for r in rows if cls[r[7]] == k]
        if not rs:
            continue
        m = lambda f: st.median(f(r) for r in rs)
        ex = [r[10] for r in rs if len(r) > 10 and r[10]]
        packs = st.median(x[0] for x in ex) if ex else None
        rsw = st.median(x[1] for x in ex) if ex and len(ex[0]) > 1 else None
        tma = [st.median(x[j] for x in ex) for j in range(2, len(ex[0]))] if ex else None
        print(f"| {k} | {len(rs)} | {sum(1 for r in rs if r[9] and r[9][1])}/{len(rs)} | {m(lambda r: r[6] * 1e3):.2f} | {m(lambda r: r[2].get('ph_us', {}).get('dispatch', 0)):.0f} | "
              f"{m(lambda r: r[2].get('ph_us', {}).get('evaluate', 0)):.0f} | {m(lambda r: r[2].get('ph_us', {}).get('patch', 0)):.0f} | {m(lambda r: r[2].get('ph_us', {}).get('launch', 0)):.0f} | "
              f"{m(lambda r: r[3]):.0f} | {packs} | {rsw} | {tma} |")
    E = R.get("setlog", {}).get("cb64", {}).get("entries", [])
    if E:
        g = collections.defaultdict(list)
        execs = collections.defaultdict(set)
        for e in E:
            kind = "function" if e["func_changed"] else ("grid/block/smem" if e["cfg_changed"] else ("args" if e["bytes_changed"] else "no-op"))
            g[(cls.get(e["step"], "?"), busy.get(e["step"]), kind)].append(e["dur_ns"] / 1e3)
            execs[cls.get(e["step"], "?")].add(e["exec"])
        print(f"\n{len(E)} kernel node set calls (steps 100-250); distinct execs per class: " + ", ".join(f"{k} {len(v)}" for k, v in execs.items()))
        print("\n| step class | stream busy | change | calls | median us | p90 us | max us |")
        print("|---|---|---|---|---|---|---|")
        for k in sorted(g, key=str):
            v = sorted(g[k])
            print(f"| {k[0]} | {k[1]} | {k[2]} | {len(v)} | {st.median(v):.2f} | {v[int(0.9 * (len(v) - 1))]:.2f} | {v[-1]:.1f} |")
