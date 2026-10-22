# Counts from SGLang driver JSONs: python3 parse_sgl.py <json>... (F / Fc / D: driveF/drive; M*: driveM; seq_*: PREP_SEQCHECK)
import json, sys

def final_variants(summary):
    out = {}
    for name, e in summary.get("entries", {}).items():
        pv = e.get("per_variant", [])
        fin = [v for v in pv if not v.get("learns")]
        out[name] = {"traces": e.get("traces"), "replays": e.get("replays"), "eager": e.get("eager"), "variants": len(pv),
                     "final": len(fin), "learners": len(pv) - len(fin),
                     "final_boundaries": sorted({v["boundaries"] for v in fin}), "final_segments": sorted({v["segments"] for v in fin}),
                     "final_eager_calls": sum(v.get("eager_calls", 0) for v in fin), "eager_reasons": e.get("eager_reasons"),
                     "declines": e.get("declines"), "reasons": e.get("reasons")}
    return out

def summ(s):
    keys = ("traces", "replays", "eager", "uncaptured", "structural", "variants", "harvests", "bindings")
    return {k: s.get(k) for k in keys} | {"families": s.get("families"), "refused": s.get("refused"), "decline_sites": s.get("decline_sites"),
            "calls": s.get("calls"), "extern": s.get("extern")}

for path in sys.argv[1:]:
    R = json.load(open(path))
    print("####", path)
    print("peak", R.get("peak"), "max_total_num_tokens", R.get("max_total_num_tokens"))
    if "seqcheck" in R:
        for bs, d in R["seqcheck"].items():
            print(" seq bs", bs, "steps", len(d["tokens"]), "calls", d["calls"], "counters", d["counters"])
    A = R.get("armF")
    if A and "summary" in A:
        s = A["summary"]
        print(" summary", json.dumps(summ(s), default=str))
        for k, v in final_variants(s).items():
            print(" entry", k, json.dumps(v, default=str))
        checks = s.get("checks", [])
        if checks:
            print(" checks bitwise %d/%d" % (sum(bool(c.get("bitwise")) for c in checks), len(checks)),
                  [(c.get("kind"), c.get("shape"), c.get("max_abs")) for c in checks if not c.get("bitwise")])
        for case, v in A.items():
            if isinstance(v, dict) and "check" in v:
                c = v["check"]
                print("  case", case, "traces", v.get("traces"), "replays", v.get("replays"), "eager", v.get("eager"),
                      "variants", v.get("variants"), "bitwise", c.get("bitwise") if isinstance(c, dict) else c)
        if "warmup" in A:
            print(" warmup", json.dumps(A["warmup"], default=str)[:600])
    if "mixed" in R:
        for sec in ("mixed", "extdecode"):
            for case, v in R.get(sec, {}).items():
                c = v.get("check") or {}
                print(" ", sec, case, "paths", v.get("paths"), "all_calls", v.get("all_calls"),
                      "bitwise", c.get("bitwise") if isinstance(c, dict) else c, "dec_vs_ext", (v.get("dec_vs_ext") or {}).get("bitwise"))
        print(" prewarm", R.get("prewarm"))
    for case, p in (R.get("profile") or {}).items():
        if isinstance(p, dict):
            print("  prof", case, {k: p.get(k) for k in ("kernels", "in_graph_kernels", "eager_kernels", "graph_launches")})
