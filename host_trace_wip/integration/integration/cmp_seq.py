# python3 cmp_seq.py ref.json other.json...: per bs, steps whose tokens and logits sha1 match the reference (PREP_SEQCHECK files)
import json, sys
ref = json.load(open(sys.argv[1]))["seqcheck"]
for path in sys.argv[2:]:
    R = json.load(open(path))
    S = R["seqcheck"]
    out = []
    for bs, d in ref.items():
        o = S.get(bs)
        if o is None:
            out.append(f"bs{bs}: missing"); continue
        tok = sum(a == b for a, b in zip(d["tokens"], o["tokens"]))
        sha = sum(a == b for a, b in zip(d["logits_sha1"], o["logits_sha1"]))
        out.append(f"bs{bs}: tokens {tok}/{len(d['tokens'])} sha1 {sha}/{len(d['logits_sha1'])} calls {o['calls']} counters {o['counters']}")
    s = R.get("armF", {}).get("summary", {})
    print(path, "| summary traces", s.get("traces"), "replays", s.get("replays"), "eager", s.get("eager"), "variants", s.get("variants"),
          "refused", s.get("refused"), "decline_sites", s.get("decline_sites"), "| peak", R.get("peak"))
    for e, v in s.get("entries", {}).items():
        fin = [x for x in v.get("per_variant", []) if not x.get("learns")]
        print("  entry", e, "traces", v.get("traces"), "eager", v.get("eager"), "variants", len(v.get("per_variant", [])),
              "final boundaries", sorted({x["boundaries"] for x in fin}), "segments", sorted({x["segments"] for x in fin}))
    print("  " + "\n  ".join(out))
