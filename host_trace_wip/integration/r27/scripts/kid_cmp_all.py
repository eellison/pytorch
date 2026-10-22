"""Every-size kernel identity (27B): R["kid"] of drive_sweep27 runs (one profiled full step per point and step class, after the
point's measured window), vLLM arm vs ours per forward.
    python3 kid_cmp_all.py DIR SET [SET...]      (DIR/t_<SET>_<arm>_r<k>.json; every round merged, rounds cross-checked)
Pairs: default vs vcvp; fullnc vs vsvp; breakable vs vsvp. Per pair: sizes compared, identical kernel multisets, differing ones
split into "padding" (the vLLM step ran padded: padded_tokens > num_toks) and "other"; then each differing kernel name with the
sizes where it differs. Ours counts only clean replays (no trace / learn in the profiled step). Tactics: FlashInfer AutoTuner
choices (op -> runner:tactic) per arm (DRIVE_TACTIC_LOG runs)."""
import collections, glob, json, os, re, sys

D, SETS = sys.argv[1], sys.argv[2:]
PAIRS = [("default", "vcvp"), ("fullnc", "vsvp"), ("breakable", "vsvp"), ("breakablegb", "vsvp")]
norm = lambda k: re.sub(r"_object_at__[0-9a-fx_]+", "_object_at_", k)


def load(arm):
    kid, tac, rounds_bad = {}, collections.defaultdict(set), []
    for s in SETS:
        for f in sorted(glob.glob(os.path.join(D, f"t_{s}_{arm}_r*.json"))):
            j = json.load(open(f))
            for op, d in (j.get("tactics") or {}).items():
                tac[op.split(" ", 1)[0]] |= set(d)
            for k, v in (j.get("kid") or {}).items():
                if "kernels" not in v or (arm.startswith("v") and not (v.get("clean") and v.get("path") == "replay")):
                    continue
                # a copy / memset can run as a DMA op or as a driver kernel (memcpy32_post, memcpy128, memset32, ...) depending on
                # how it was issued (eager call or graph node): those kernels are folded into the copy / memset counts
                ker = collections.Counter({norm(n): c for n, c in v["kernels"].items()})
                cp = sum(c for n, c in ker.items() if re.match(r"memcpy", n)) + v.get("memcpy", 0)
                ms = sum(c for n, c in ker.items() if re.match(r"memset", n)) + v.get("memset", 0)
                ker = collections.Counter({n: c for n, c in ker.items() if not re.match(r"mem(cpy|set)", n)})
                shp = collections.Counter({norm(n): c for n, c in (v.get("shapes") or {}).items() if not re.match(r"mem(cpy|set)", n)})
                v = dict(v, kernels=ker, shapes=shp, copies=(cp, ms), src=os.path.basename(f))
                key = f"{s}:{k}"
                if key in kid and kid[key]["kernels"] != v["kernels"]:
                    rounds_bad.append((key, kid[key]["src"], v["src"]))
                kid.setdefault(key, v)
    return kid, tac, rounds_bad


arms = {a: load(a) for a in {x for p in PAIRS for x in p}}
for a, (kid, tac, bad) in sorted(arms.items()):
    print(f"{a:10s}: {len(kid)} profiled sizes" + (f"; ROUNDS DIFFER at {len(bad)} sizes, e.g. {bad[:3]}" if bad else ""))
for vl, ours in PAIRS:
    kv, ko = arms[vl][0], arms[ours][0]
    both = sorted(set(kv) & set(ko))
    if not both:
        continue
    same, pad, other, diff_shapes, copies_diff = [], [], [], [], []
    names = collections.defaultdict(lambda: {"padding": [], "other": []})
    shape_names = collections.defaultdict(lambda: {"padding": [], "DIFF_SHAPES": []})
    for k in both:
        a, b = kv[k]["kernels"], ko[k]["kernels"]
        padded = (kv[k].get("padded_tokens") or 0) > (kv[k].get("num_toks") or 0)
        if kv[k]["copies"] != ko[k]["copies"]:
            copies_diff.append((k, kv[k]["copies"], ko[k]["copies"], padded))
        if a == b:
            same.append(k)
            sa, sb = kv[k]["shapes"], ko[k]["shapes"]
            if sa and sb and sa != sb:  # same kernel names, different launch shapes (grid / block): split counts, max_seq_len grids
                diff_shapes.append((k, padded))
                for n in set(sa) | set(sb):
                    if sa[n] != sb[n]:
                        shape_names[n]["padding" if padded else "DIFF_SHAPES"].append((k, sa[n], sb[n]))
            continue
        (pad if padded else other).append(k)
        for n in set(a) | set(b):
            if a[n] != b[n]:
                names[n]["padding" if padded else "other"].append((k, a[n], b[n]))
    print(f"\n===== {vl} vs {ours}: {len(both)} sizes in both ({len(set(kv) - set(ko))} only {vl}, {len(set(ko) - set(kv))} only {ours}); "
          f"identical {len(same)} (of which launch shapes differ: {sum(not p for _, p in diff_shapes)} DIFF_SHAPES unpadded, "
          f"{sum(p for _, p in diff_shapes)} padded), differ with vLLM padding {len(pad)}, differ otherwise {len(other)}")
    secs = collections.Counter(re.match(r"[a-z]*", k.split(":", 1)[1]).group(0) for k in both)
    print("   sizes by section: " + ", ".join(f"{s or '?'} {n}" for s, n in sorted(secs.items())))
    for n, d in sorted(names.items(), key=lambda x: -len(x[1]["other"]) - len(x[1]["padding"])):
        for lab in ("other", "padding"):
            if d[lab]:
                ex = ", ".join(f"{k.split(':', 1)[1]} ({x} vs {y})" for k, x, y in d[lab][:4])
                print(f"   [{lab}] {len(d[lab]):4d} sizes: {n[:120]}  e.g. {ex}")
    if copies_diff:
        print(f"   copy / memset counts (DMA + copy kernels) differ at {len(copies_diff)} sizes, e.g. "
              + ", ".join(f"{k.split(':', 1)[1]} {x} vs {y}{' (padded)' if p else ''}" for k, x, y, p in copies_diff[:4]))
    for n, d in sorted(shape_names.items(), key=lambda x: -len(x[1]["DIFF_SHAPES"])):
        for lab in ("DIFF_SHAPES", "padding"):
            if d[lab]:
                ex = ", ".join(f"{k.split(':', 1)[1]} ({x} vs {y})" for k, x, y in d[lab][:3])
                print(f"   [{lab} shapes] {len(d[lab]):4d} sizes: {n[:160]}  e.g. {ex}")
print("\nTactics (FlashInfer AutoTuner op -> runner:tactic seen in the run):")
for a, (_, tac, _) in sorted(arms.items()):
    print(f"  {a:10s} " + "; ".join(f"{op}: {sorted(v)}" for op, v in sorted(tac.items())))
