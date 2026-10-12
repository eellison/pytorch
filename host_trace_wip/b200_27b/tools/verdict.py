# verdict.py OUT.json: the pass criteria of a 27B V-stock --check.
#   PASS needs: every decode check row bitwise (output and recurrent state); every decode variant with 0 eager steps and 1 segment;
#   learned (harvested) keys cuBLAS only (unsound_learned empty); at least one decode replay.
# Prefill / mixed rows are reported (counts by path / tokens / bitwise, eager-twice control) but do not decide PASS.
import collections, json, sys

d = json.load(open(sys.argv[1])); V = d["V"]
ok, why = True, []
dec = [c for c in V.get("checks", []) if c.get("kind") == "decode"]
bad = [c for c in dec if not (c.get("bitwise") and c.get("state_bitwise", True))]
if bad:
    ok = False; why.append(f"{len(bad)} of {len(dec)} decode rows not bitwise")
if not any(c.get("path") == "replay" for c in dec):
    ok = False; why.append("no decode replay")
for k, e in V.get("entries", {}).items():
    if "decode" in k:
        for pv in e.get("per_variant", []):
            if pv.get("eager_calls") != 0 or pv.get("segments") != 1:
                ok = False; why.append(f"decode variant {k}: eager_calls {pv.get('eager_calls')} segments {pv.get('segments')} {dict(collections.Counter(pv.get('eager_ops', {})))}")
if V.get("unsound_learned"):
    ok = False; why.append(f"non-cuBLAS learned keys {V['unsound_learned']}")
print("VERDICT", "PASS" if ok else "FAIL", why)
print("learned", V.get("learned_by_op"), "totals", {k: V.get(k) for k in ("traces", "replays", "eager", "variants", "calls")})
rows = collections.Counter((c.get("kind"), c.get("path"), c.get("tokens"), bool(c.get("bitwise")), c.get("state_bitwise"), c.get("eager_twice_bitwise"))
                           for c in V.get("checks", []) if c.get("kind") != "decode")
print("prefill/mixed rows (kind, path, tokens, bitwise, state_bitwise, eager_twice_bitwise): n  [reported, not part of PASS]")
for k, n in sorted(rows.items(), key=str):
    print("  ", k, n)
