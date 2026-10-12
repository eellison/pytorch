# timing_table.py DIR: median [min-max] over rounds of the decode step time (ms, execute_model, synced) per arm and row, and the
# V-stock speedup vs default and vs default-nc; V-stock replay / trace / eager counts per round (from its summary).
import collections, glob, json, os, re, statistics, sys

d = sys.argv[1]
rows = collections.defaultdict(lambda: collections.defaultdict(list))
counts = []
for f in sorted(glob.glob(os.path.join(d, "t_*_r*.json"))):
    arm = re.match(r"t_(\w+?)_r\d+\.json", os.path.basename(f)).group(1)
    j = json.load(open(f))
    for bs, v in j.get("decode", {}).items():
        if v.get("step_ms"):
            rows[f"decode bs {bs} (KV ~256)"][arm].append(v["step_ms"])
    for k, v in j.get("ix", {}).items():
        if k.startswith("decode_") and v.get("step_ms"):
            rows[f"decode bs {k[7:]} (KV {v.get('kv_min_med_max')})"][arm].append(v["step_ms"])
    if arm == "vstock" and "V" in j:
        V = j["V"]
        counts.append({k: V.get(k) for k in ("traces", "replays", "eager", "variants")} | {"unsound": V.get("unsound_learned")})
fmt = lambda xs: f"{statistics.median(xs):8.2f} [{min(xs):.2f}-{max(xs):.2f}] n={len(xs)}" if xs else "        -"
print(f"{'row':42s} {'default':28s} {'default-nc':28s} {'V-stock':28s} V/def V/nc")
for r, arms in rows.items():
    m = {a: statistics.median(v) for a, v in arms.items() if v}
    sp = lambda a: f"{m[a] / m['vstock']:.2f}x" if a in m and "vstock" in m else "-"
    print(f"{r:42s} {fmt(arms.get('default', [])):28s} {fmt(arms.get('fullnc', [])):28s} {fmt(arms.get('vstock', [])):28s} {sp('default')} {sp('fullnc')}")
print("V-stock counts per round:", counts)
