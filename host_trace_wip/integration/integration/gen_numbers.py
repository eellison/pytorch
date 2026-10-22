# Tables for NUMBERS_cand2.md from out_c2/ (counts, correctness, memory) and time_c2/ (timing, startup).
# python3 gen_numbers.py > numbers_tables.md   (per-run rows; raw JSON path on every row)
import glob
import json
import os

X = os.path.dirname(os.path.abspath(__file__))


def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def parts(path):
    # c2_<line>_<backend>_<arm>[_<round>].json; seq_c2_<line>_<backend>_<arm>.json
    b = os.path.basename(path)[:-5].replace("seq_", "")
    f = b.split("_")
    return f[1], f[2], "_".join(f[3:])


def fin(entries):
    out = []
    for name, e in entries.items():
        pv = e.get("per_variant", [])
        f = [v for v in pv if not v.get("learns")]
        out.append(f"{name}: tr {e.get('traces')} eager {e.get('eager')} var {len(pv)} (final {len(f)}) bnd {sorted({v['boundaries'] for v in f})} seg {sorted({v['segments'] for v in f})}")
    return "; ".join(out)


print("## Counts and correctness: driveF / drive (decode bs 1 8 64, prefill T 64 512 2048 + ragged)\n")
print("| line | backend | arm | bitwise | traces | eager steps | variants | refused / declines | graphs per profiled step (eager kernels) | peak alloc MiB (vs D) | KV tokens | per entry | raw |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
peaks = {}
for p in sorted(glob.glob(f"{X}/out_c2/c2_*.json")):
    line, be, arm = parts(p)
    R = load(p)
    if R and arm in ("D", "DT"):
        peaks[(line, be, arm)] = R.get("peak", {}).get("max_allocated_mib")
for p in sorted(glob.glob(f"{X}/out_c2/c2_*.json")):
    line, be, arm = parts(p)
    R = load(p)
    if not R or "mixed" in R or arm.startswith(("M", "KV")):
        continue
    s = (R.get("armF") or {}).get("summary") or {}
    ch = s.get("checks", [])
    prof = R.get("profile") or {}
    g = ", ".join(f"{k} {v.get('graph_launches')} ({v.get('eager_kernels')})" for k, v in prof.items() if isinstance(v, dict))
    pk = R.get("peak", {}).get("max_allocated_mib")
    base = peaks.get((line, be, "DT" if arm.endswith("T") else "D"))
    d = f"{pk:.1f} ({pk - base:+.1f})" if pk is not None and base is not None else (f"{pk:.1f}" if pk else "")
    print(f"| {line} | {be} | {arm} | {sum(bool(c.get('bitwise')) for c in ch)}/{len(ch)} | {s.get('traces', '')} | {s.get('eager', '')} | {s.get('variants', '')} | "
          f"{len(s.get('refused') or [])} / {len(s.get('decline_sites') or [])} | {g} | {d} | {R.get('max_total_num_tokens')} | {fin(s.get('entries', {}))} | {os.path.relpath(p, X)} |")

print("\n## Arm P sequences: prefill + 13 decode steps, tokens and logits sha1 vs the same line's eager (SD / SDT; L = PREP_SEQ_LEN 250)\n")
print("| line | backend | arm | ref | bs: tokens, sha1 (traces, eager, calls) | summary traces / eager / variants | raw |")
print("|---|---|---|---|---|---|---|")
for p in sorted(glob.glob(f"{X}/out_c2/seq_c2_*.json")):
    line, be, arm = parts(p)
    if arm.startswith("SD"):
        continue
    ref = "SDT" if arm == "SM" else ("SDL" if arm.endswith("L") else "SD")
    rp = f"{X}/out_c2/seq_c2_{line}_{be}_{ref}.json"
    R, RR = load(p), load(rp)
    if not R:
        continue
    cells = []
    for bs, d in R["seqcheck"].items():
        r = (RR or {}).get("seqcheck", {}).get(bs)
        tok = sum(a == b for a, b in zip(d["tokens"], r["tokens"])) if r else "?"
        sha = sum(a == b for a, b in zip(d["logits_sha1"], r["logits_sha1"])) if r else "?"
        c = d["counters"]
        cells.append(f"bs{bs}: {tok}/{len(d['tokens'])}, {sha}/{len(d['logits_sha1'])} ({c['traces']}, {c['eager']}, {d['calls']})")
    s = R.get("armF", {}).get("summary", {})
    print(f"| {line} | {be} | {arm} | {ref} | {'; '.join(cells)} | {s.get('traces')} / {s.get('eager')} / {s.get('variants')} | {os.path.relpath(p, X)} |")

print("\n## Mixed (driveM: chunk C x running decodes, and decode-through-extend)\n")
print("| line | backend | arm | in-process bitwise | mixed digests = eager | traces / eager / variants | final bnd / seg | graphs per profiled step | peak alloc MiB (vs eager) | raw |")
print("|---|---|---|---|---|---|---|---|---|---|")
for p in sorted(glob.glob(f"{X}/out_c2/c2_*_M*.json")):
    line, be, arm = parts(p)
    R = load(p)
    if not R or "mixed" not in R:
        continue
    ref = load(f"{X}/out_c2/c2_{line}_{be}_{'MDT' if arm.endswith('T') else 'MD'}.json")
    dig = f"{sum(v['digests'] == ref['mixed'][c]['digests'] for c, v in R['mixed'].items())}/{len(R['mixed'])}" if ref else "?"
    s = (R.get("armF") or {}).get("summary") or {}
    ch = s.get("checks", [])
    f = [v for e in s.get("entries", {}).values() for v in e.get("per_variant", []) if not v.get("learns")]
    g = ", ".join(f"{k} {v.get('graph_launches')}" for k, v in (R.get("profile") or {}).items() if isinstance(v, dict))
    pk = R["peak"]["max_allocated_mib"]
    d = f"{pk:.1f} ({pk - ref['peak']['max_allocated_mib']:+.1f})" if ref else f"{pk:.1f}"
    print(f"| {line} | {be} | {arm} | {sum(bool(c.get('bitwise')) for c in ch)}/{len(ch)} | {dig} | {s.get('traces', '')} / {s.get('eager', '')} / {s.get('variants', '')} | "
          f"{sorted({v['boundaries'] for v in f})} / {sorted({v['segments'] for v in f})} | {g} | {d} | {os.path.relpath(p, X)} |")

print("\n## KV capacity at --mem-fraction-static 0.8, no token cap\n")
for p in sorted(glob.glob(f"{X}/out_c2/c2_*_KV*.json")):
    R = load(p)
    if R:
        print(f"- {parts(p)}: max_total_num_tokens {R.get('max_total_num_tokens')}, capture {R.get('capture')}, mem_after_load {R.get('mem_after_load')}, peak {R.get('peak')} ({os.path.relpath(p, X)})")

print("\n## Timing: steady state (fwd median ms per forward; GPU 0)\n")
rows = sorted(glob.glob(f"{X}/time_c2/c2_*_t*_r*.json"))
print("| run | prefill 64 / 512 / 2048 / ragged | decode bs 1 / 8 / 13 / 50 / 64 / 130 | decode host ms (same bs) | capture | raw |")
print("|---|---|---|---|---|---|")
for p in rows:
    R = load(p)
    if not R:
        continue
    pf = R.get("prefill", {})
    dc = R.get("decode", {})
    f = lambda d, k="fwd_median_ms": "-" if not isinstance(d, dict) or d.get(k) is None else f"{d[k]:.3f}"
    print(f"| {os.path.basename(p)[3:-5]} | {' / '.join(f(pf.get(str(t))) for t in (64, 512, 2048))} / {f(R.get('ragged'))} | "
          f"{' / '.join(f(dc.get(str(b))) for b in (1, 8, 13, 50, 64, 130))} | {' / '.join(f(dc.get(str(b)), 'fwd_host_median_ms') for b in (1, 8, 13, 50, 64, 130))} | "
          f"{(R.get('capture') or {}).get('seconds', '')} | {os.path.relpath(p, X)} |")

print("\n## Timing: mixed (fwd median ms; GPU 0)\n")
for p in sorted(glob.glob(f"{X}/time_c2/c2_*_m*_r*.json")):
    R = load(p)
    if not R:
        continue
    mx = {c: round(v.get("fwd_median_ms", 0), 3) for c, v in R.get("mixed", {}).items()}
    ed = {b: tuple(round((v.get(k) or {}).get("fwd_median_ms", 0), 3) for k in ("dec", "ext")) for b, v in R.get("extdecode", {}).items()}
    print(f"- {os.path.basename(p)[3:-5]}: mixed {mx}; ext-decode (dec, ext) {ed} ({os.path.relpath(p, X)})")

print("\n## Startup (seconds from process start; GPU 0)\n")
print("| run | imports | load end | SGLang capture | installed | ready | our warm-up | first call per class ms | traces / eager | loaded / saved bindings | raw |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for p in sorted(glob.glob(f"{X}/time_c2/c2_*_s*_r*.json")):
    R = load(p)
    if not R or "classes" not in R:
        continue
    cap = (R.get("capture") or {}).get("seconds")
    fc = ", ".join(f"{k} {v['first_s'] * 1e3:.0f}" for k, v in R["classes"].items())
    s = R.get("summary") or {}
    print(f"| {os.path.basename(p)[3:-5]} | {R['t_imports']:.1f} | {R['t_load_end']:.1f} | {'' if cap is None else f'{cap:.1f}'} | {R['t_installed']:.1f} | {R['t_ready']:.1f} | "
          f"{R['warmup_s']:.2f} | {fc} | {s.get('traces', '')} / {s.get('eager', '')} | {R.get('loaded_bindings', '')} / {R.get('saved_bindings', '')} | {os.path.relpath(p, X)} |")
