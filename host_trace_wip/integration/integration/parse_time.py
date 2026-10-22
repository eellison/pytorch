# Per-run timing tables from run_time_c2.sh outputs: python3 parse_time.py <json>...
# steady (drive.py / driveF*.py): fwd_median_ms per prefill T, ragged, decode bs; mixed (driveM.py): fwd_median_ms per
# chunk x decodes and per ext-decode bs (dec / ext); startup (startup_sgl.py): seconds to imports / load / capture / ready.
import json
import sys


def ms(d, key="fwd_median_ms"):
    return None if not isinstance(d, dict) or d.get(key) is None else round(d[key], 3)


for path in sys.argv[1:]:
    R = json.load(open(path))
    print("####", path)
    if "classes" in R:
        cap = R.get("capture") or {}
        print(f" startup arm={R['arm']} imports={R['t_imports']:.1f}s load_end={R['t_load_end']:.1f}s"
              f" capture={cap.get('seconds', 0):.1f}s installed={R['t_installed']:.1f}s ready={R['t_ready']:.1f}s"
              f" warmup={R['warmup_s']:.2f}s kv_tokens={R.get('max_total_num_tokens')} loaded={R.get('loaded_bindings')}"
              f" saved={R.get('saved_bindings')} peak={R.get('peak')}")
        for name, c in R["classes"].items():
            print(f"  {name}: first={c['first_s']*1e3:.1f}ms warm={c['warm_s']*1e3:.1f}ms calls={c['n_calls']}"
                  f" traces={c['traces']} eager={c['eager']} variants={c['variants']}")
        print(" summary", R.get("summary"))
        continue
    if "mixed" in R:
        print(" mixed", {c: ms(v) for c, v in R["mixed"].items()})
        print(" extdecode", {b: (ms(v.get("dec")), ms(v.get("ext"))) for b, v in R.get("extdecode", {}).items()})
    else:
        print(" prefill", {T: ms(v) for T, v in R.get("prefill", {}).items()}, "ragged", ms(R.get("ragged")))
        print(" decode", {b: ms(v) for b, v in R.get("decode", {}).items()})
        print(" decode host", {b: ms(v, "fwd_host_median_ms") for b, v in R.get("decode", {}).items()})
    print(" capture", R.get("capture"), "peak", R.get("peak"), "kv_tokens", R.get("max_total_num_tokens"))
