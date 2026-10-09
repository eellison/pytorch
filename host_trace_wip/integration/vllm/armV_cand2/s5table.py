# python3 armV/s5table.py TAG...: step ms and host sections (us) per row for each TAG_{V,default,eager}.json present.
import json, os, sys
B = "/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/vllm/bench/out"
PRE = ("add_requests", "update_requests", "gather_batch_req_state", "prepare_inputs", "prepare_attn", "build_slot_mappings_by_layer")
for tag in sys.argv[1:]:
    for arm in ("V", "default", "eager"):
        p = f"{B}/{tag}_{arm}.json"
        if not os.path.exists(p):
            continue
        R = json.load(open(p))
        hm = R.get("V", {}).get("host_ms", {})
        rows = [("decode", k, v) for k, v in R["decode"].items()] + [("prefill", k, v) for k, v in R["prefill"].items()] + ([("mixed", "", R["mixed"])] if "step_ms" in R.get("mixed", {}) else [])
        print(f"== {tag} {arm}  probe {R.get('host_probe_us')}")
        for kind, k, v in rows:
            s = v.get("sections_us", {})
            pre = sum(s.get(x, 0) for x in PRE)
            print(f"{kind:7s} {k:6s} step {v['step_ms']:6.2f} min {v['step_min_ms']:6.2f} | pre-fwd {pre:6.0f} = " + " ".join(f"{x.split('_')[0][:6]}:{s.get(x,0):.0f}" for x in PRE if s.get(x)))
