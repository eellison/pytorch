import json, statistics, sys
L = "/data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast/out/startup27_h3t_vt_rw_sec_r%s.diag.jsonl"
for t in sys.argv[1:]:
    rows = [json.loads(l) for l in open(L % t) if '"gcu"' in l]
    end = [json.loads(l) for l in open(L % t) if '"ev": "end"' in l]
    full = [r for i, r in enumerate(rows) if i == 0 or r["full_walks"] != rows[i - 1]["full_walks"]]
    inc = [r for r in rows if r not in full]
    st = json.load(open(L.replace(".diag.jsonl", ".json") % t))["startup"]
    print(f"{t}: walk {sum(r['s'] for r in rows):.1f} s / {len(rows)} calls; full {len(full)} calls {sum(r['s'] for r in full):.1f} s; "
          f"incremental {len(inc)} calls {sum(r['s'] for r in inc):.1f} s, median {statistics.median(r['s'] for r in inc):.3f} s / "
          f"{statistics.median(r['walked'] for r in inc):,.0f} walked" + (f" / {statistics.median(r['sealed_skipped'] for r in inc):,.0f} sealed skipped" if inc and inc[0].get('sealed_skipped') is not None else "")
          + f"; ready {st['engine_ready_s']:.1f} steady {st['engine_ready_s'] + st.get('steady_after_ready_s', 0):.1f}")
