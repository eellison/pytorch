# Count table for drive.py / rangesweep.py JSONs: python summ.py X.json ...
import collections, json, sys

for path in sys.argv[1:]:
    R = json.load(open(path))
    print("==", path)
    if "summary" in R:  # rangesweep
        S = R["summary"]
        print(" traces", {p: S[p].get("traces") for p in S}, "total", sum(S[p].get("traces", 0) for p in S),
              "relowers", sum(S[p].get("relowers", 0) for p in S), "learns", sum(S[p].get("learns", 0) for p in S))
        print(" eager_entry", {p: S[p].get("eager_entry") for p in S}, "eager_adapter", {p: S[p].get("eager_adapter") for p in S})
        print(" first_ms_med", {p: S[p].get("first_ms_med") for p in S}, "new_keys", {p: S[p].get("new_keys") for p in S})
        for k, e in R["entries"].items():
            print("  entry", k, e)
        C = R.get("census", {"boundaries": None, "variants": None, "eager_ops": None, "refused": []})
        print(" boundaries/variant", C["boundaries"], "variants", C["variants"], "eager_ops", C["eager_ops"], "refused", C["refused"][:5])
        print(" checks", R["checks"]["n"], R["checks"]["bitwise"], "bad keys", len(R["bad"]), "mem", R["mem"])
        continue
    m = R["mem"]
    print(" mem peak_over_init %.0f MiB end_over_init %.1f kv_tokens %d after_init %.0f" % (m["bench_peak_over_init_mib"], m["end_over_init_mib"], m["kv_cache_tokens"], m["after_init_mib"]))
    if "V" not in R:
        continue
    V = R["V"]
    print(" traces", V["traces"], "variants", V["variants"], "replays", V["replays"], "eager", V["eager"], "harvests", V["harvests"], "bad_keys", V["bad_keys"], "refused", V["refused"][:3], "decline_sites", V["decline_sites"][:3])
    ch = V["checks"]
    print(" checks", len(ch), "bitwise", sum(c["bitwise"] for c in ch), "bitwise_vs_stock", sum(bool(c.get("bitwise_vs_stock")) for c in ch))
    bad = [c for c in ch if not c["bitwise"]]
    if bad:
        print(" BAD", bad[:5])
    bnd, seg, learn = collections.Counter(), collections.Counter(), 0
    for k, e in V["entries"].items():
        pv = e["per_variant"]
        serving = [v for v in pv if not v["learns"]]
        bnd.update(v["boundaries"] for v in serving)
        seg.update(v["segments"] for v in serving)
        learn += len(pv) - len(serving)
        print("  %-48s tr %s var %s rep %s eager %s kernels %s bnd %s seg %s eager_ops %s" % (k[:48], e["traces"], e["variants"], e["replays"], e["eager"],
              [v["kernels"] for v in serving], [v["boundaries"] for v in serving], [v["segments"] for v in serving], e["eager_ops"]))
    print(" serving variants: boundaries", dict(bnd), "segments", dict(seg), "learner variants", learn)
