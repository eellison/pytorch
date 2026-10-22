# nan_summary.py CHECK.json...: per process, check rows by kind with NaN treated as unequal (strict): rows equal and NaN-free, rows whose
# eager reference has NaN, rows where ours has NaN, and the two cross directions (ours NaN with a clean reference; reference NaN with
# ours clean); our counts (traces, replays, eager, learned ops).
import collections, json, sys

tot = collections.Counter()
for f in sys.argv[1:]:
    try:
        d = json.load(open(f))
    except Exception as e:
        print(f, "unreadable", type(e).__name__)
        continue
    rows = d["V"].get("checks", [])
    c = collections.Counter()
    for r in rows:
        ref_nan, ours_nan = bool(r.get("nan")), (r.get("nan_v") or 0) > 0
        strict = r["bitwise"] and not ref_nan and not ours_nan and r.get("state_bitwise", True) is not False
        c[(r["kind"], "strict_equal" if strict else "unequal")] += 1
        c["ref_nan_rows"] += ref_nan
        c["ours_nan_rows"] += ours_nan
        c["ours_nan_ref_clean"] += ours_nan and not ref_nan
        c["ref_nan_ours_clean"] += ref_nan and not ours_nan
        c["vp_tokens_unequal"] += r.get("vp_tokens_equal") is False
        # the reference's own nondeterminism: an eager-path row (ours not involved) that differs, or eager run twice that differs
        c["eager_path_rows_unequal"] += r.get("path") == "eager" and not strict and not ref_nan
        c["ref_twice_unequal"] += r.get("eager_twice_bitwise") is False
    V = d["V"]
    print(f.split("/")[-1], dict(sorted(c.items(), key=str)), {k: V.get(k) for k in ("traces", "replays", "eager", "unsound_learned")},
          "NaN-reference process" if c["ref_nan_rows"] else "clean process")
    tot.update(c)
    tot["processes"] += 1
    tot["nan_ref_processes"] += c["ref_nan_rows"] > 0
    tot["nan_ours_processes"] += c["ours_nan_rows"] > 0
print("TOTAL", dict(sorted(tot.items(), key=str)))
