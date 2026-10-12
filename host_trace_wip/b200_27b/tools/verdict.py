# verdict.py OUT.json: the pass criteria of a 27B V-stock --check.
#   PASS needs: every decode check row bitwise (output and recurrent state); every decode variant with 0 eager steps and 1 segment;
#   learned (harvested) keys cuBLAS only (unsound_learned empty); at least one decode replay whose eager reference is free of NaN.
#   The row compare is NaN-aware, so rows whose eager reference has NaN are excluded (listed as not verified).
# Prefill / mixed rows are reported (counts by path / tokens / bitwise, eager-twice control) but do not decide PASS.
import collections, json, sys

d = json.load(open(sys.argv[1])); V = d["V"]
ok, why = True, []
dec = [c for c in V.get("checks", []) if c.get("kind") == "decode"]
bad = [c for c in dec if not (c.get("bitwise") and c.get("state_bitwise", True))]
if bad:
    ok = False; why.append(f"{len(bad)} of {len(dec)} decode rows not bitwise")
# the compare is NaN-aware (equal where both are NaN): a row whose eager reference has NaN proves nothing. Count only clean rows.
clean = [c for c in dec if not c.get("nan") and not c.get("nan_v")]
by_bs = collections.Counter((c.get("tokens"), c.get("path")) for c in clean)
print("clean decode rows (tokens, path): n", dict(sorted(by_bs.items(), key=str)), "| decode rows with NaN in the eager reference:",
      dict(collections.Counter(c.get("tokens") for c in dec if c.get("nan"))))
if not any(c.get("path") == "replay" for c in clean):
    ok = False; why.append("no clean decode replay (every decode reference has NaN: inconclusive, rerun in a fresh process)")
missing = [bs for bs in (1, 8, 64) if not any(c.get("tokens") == bs and c.get("path") == "replay" for c in clean)]
if missing:
    why.append(f"no clean decode replay at bs {missing} (not verified there)")
for k, e in V.get("entries", {}).items():
    if "decode" in k:
        for pv in e.get("per_variant", []):
            if pv.get("eager_calls") != 0 or pv.get("segments") != 1:
                ok = False; why.append(f"decode variant {k}: eager_calls {pv.get('eager_calls')} segments {pv.get('segments')} {dict(collections.Counter(pv.get('eager_ops', {})))}")
if V.get("unsound_learned"):
    ok = False; why.append(f"non-cuBLAS learned keys {V['unsound_learned']}")
print("VERDICT", ("PARTIAL" if missing else "PASS") if ok else "FAIL", why)
print("learned", V.get("learned_by_op"), "totals", {k: V.get(k) for k in ("traces", "replays", "eager", "variants", "calls")})
rows = collections.Counter((c.get("kind"), c.get("path"), c.get("tokens"), bool(c.get("bitwise")), c.get("state_bitwise"), c.get("eager_twice_bitwise"), "nan_ref" if c.get("nan") else "clean")
                           for c in V.get("checks", []) if c.get("kind") != "decode")
print("prefill/mixed rows (kind, path, tokens, bitwise, state_bitwise, eager_twice_bitwise, reference): n  [reported, not part of PASS]")
for k, n in sorted(rows.items(), key=str):
    print("  ", k, n)
