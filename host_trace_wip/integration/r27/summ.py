# summ.py OUT.json: counts of a drive_fn arm-V --check run (traces / replays / eager / variants per entry, eager steps and segments per
# variant, learned keys, unsound keys, check rows by kind / path / tokens / bitwise / state / eager-twice control)
import collections, json, sys
d = json.load(open(sys.argv[1])); V = d["V"]
print("TOTAL", {k: V.get(k) for k in ("traces", "replays", "eager", "variants", "calls")})
print("LEARNED", V.get("learned_by_op"), "UNSOUND", V.get("unsound_learned"))
ch = V.get("checks", [])
print("CHECKS", len(ch), "bitwise", sum(bool(c.get("bitwise")) for c in ch), "state_bitwise", sum(bool(c.get("state_bitwise")) for c in ch))
rows = collections.Counter((c.get("kind"), c.get("path"), c.get("tokens", c.get("shape")), c.get("bitwise"), c.get("state_bitwise"), c.get("eager_twice_bitwise")) for c in ch)
print("  CHECK (kind, path, tokens, bitwise, state_bitwise, eager_twice_bitwise): n")
for k, n in sorted(rows.items(), key=str):
    print("  CHECK", k, n)
for c in ch:
    if not (c.get("bitwise") and c.get("state_bitwise", True)):
        print("  BAD", {k: c.get(k) for k in ("kind", "path", "tokens", "shape", "nan", "nan_v", "max_abs", "state_bitwise", "state_bad", "state_first_bad_detail", "eager_twice_bitwise", "eager_twice_max_abs")})
for k, e in V.get("entries", {}).items():
    print("ENTRY", str(k)[:140], {x: e.get(x) for x in ("traces", "replays", "eager", "variants", "redispatches")})
    for pv in e.get("per_variant", [])[:6]:
        print("   VARIANT", {x: pv.get(x) for x in ("kernels", "eager_calls", "boundaries", "segments")}, list(collections.Counter(pv.get("eager_ops", [])).items())[:10])
    rc = e.get("retrace_causes")
    if rc:
        print("   RETRACE", str(rc)[:400])
    if e.get("declines") and not e.get("variants"):
        print("   DECLINES", e.get("declines", [])[:3])
