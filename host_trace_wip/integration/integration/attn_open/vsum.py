# V summary of a drive_attn JSON: python vsum.py out/X.json
import collections, json, sys
for f in sys.argv[1:]:
    d = json.load(open(f))
    V = d.get("V")
    print("==", f, d.get("attn"))
    if not V:
        print("no V"); continue
    print({k: V.get(k) for k in ("traces", "variants", "replays", "eager", "calls", "harvests", "bindings", "host_ms")})
    print("learned", dict(list(V.get("learned_by_op", {}).items())[:8]), "refused", V.get("refused", [])[:4])
    for k, e in V["entries"].items():
        pv = e["per_variant"]
        print(" ENTRY", k[:120], {x: e[x] for x in ("traces", "replays", "eager", "variants", "redispatches")})
        print("   variants", [(v["kernels"], v["eager_calls"], v["boundaries"], v["segments"]) for v in pv][:12])
        eo = collections.Counter()
        for v in pv:
            eo.update(v["eager_ops"])
        if eo: print("   eager_ops", dict(eo))
        for r in e["eager_reasons"][:6]: print("   eager_reason", r[:300])
        for r in e["declines"][:6]: print("   decline", r[:400])
        for c, n in e["retrace_causes"].items(): print("   retrace", n, c[:300])
    print(" decline_sites", V.get("decline_sites", [])[:8])
    ch = V["checks"]
    print(" checks", len(ch), "bitwise", sum(bool(c["bitwise"]) for c in ch), collections.Counter((c["kind"], c["bitwise"]) for c in ch))
    print(" step_ms", {k: round(v["step_ms"], 2) for k, v in d.get("decode", {}).items()}, {k: round(v["step_ms"], 2) for k, v in d.get("prefill", {}).items() if "step_ms" in v}, round(d["mixed"].get("step_ms", 0), 2) if d.get("mixed") else None)
