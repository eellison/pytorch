# python3 armV/wtable.py TAG...: pre-launch window rows (drive.py --launch-stamps) per TAG_V/default json, with sections when stamped.
import json, os, sys
B = "/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/vllm/bench/out"
for tag in sys.argv[1:]:
    for arm in ("V", "Vcpp", "default"):
        p = f"{B}/{tag}_{arm}.json"
        if not os.path.exists(p):
            continue
        R = json.load(open(p))
        print(f"== {tag} {arm} hostcuts={R.get('V', {}).get('hostcuts')} replay_kw={R.get('V', {}).get('replay_kw')}")
        for kind in ("decode", "prefill"):
            for k, v in R[kind].items():
                s = v.get("sections_us", {})
                extra = f" fwd_entry {v['fwd_entry_us']:.0f} fwd->launch {v['fwd_to_launch_us']:.0f}" if "fwd_entry_us" in v else ""
                pl = v.get("pre_launch_us")
                print(f"{kind:7s} {k:6s} step {v['step_ms']:6.2f} pre-launch {pl if pl is None else round(pl)}{extra} " + " ".join(f"{a}:{b:.0f}" for a, b in s.items() if b >= 3))
