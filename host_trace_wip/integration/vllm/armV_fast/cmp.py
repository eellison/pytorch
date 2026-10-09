# armV/cmp.py tag:arm ...: step / fwd ms, V pre-launch host, sections per case.
import json, sys
for spec in sys.argv[1:]:
    tag, arm = spec.split(":")
    d = json.load(open(f"bench/out/{tag}_{arm}.json"))
    print(f"== {tag}_{arm}", "host_ms" in d.get("V", {}) and {k: round(v[0], 3) for k, v in d["V"]["host_ms"].items()} or "")
    for ph in ("decode", "prefill"):
        for k, r in d[ph].items():
            sec = {s: round(v) for s, v in sorted(r.get("sections_us", {}).items(), key=lambda x: -x[1])[:14]}
            print(f"  {ph} {k}: step {r['step_ms']:.2f} fwd {r['fwd_ms']:.2f} min {r['step_min_ms']:.2f} {sec}")
    r = d["mixed"]
    print(f"  mixed: step {r['step_ms']:.2f} fwd {r['fwd_ms']:.2f}", {s: round(v) for s, v in sorted(r.get("sections_us", {}).items(), key=lambda x: -x[1])[:14]})
    m = d["mem"]; print("  mem", round(m["bench_peak_over_init_mib"]), m["kv_cache_tokens"])
