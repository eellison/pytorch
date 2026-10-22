# memprobe comparison (main 10-14): vrs vs fullnc reserved / allocated / driver-side memory at init, after warm-range and at the end;
# pools; top active allocation sites. python3 serve/ix/memprobe_table.py
import json, os

G = 2 ** 30
runs = {a: json.load(open(f"vllm/out/t_memprobe_{a}_r1.json")).get("memprobe", {}) for a in ("vrs", "fullnc") if os.path.exists(f"vllm/out/t_memprobe_{a}_r1.json")}
print("| arm | point | reserved GiB | allocated GiB | reserved - allocated GiB | inactive split GiB | segments | driver used - torch reserved GiB | variants |")
print("|---|---|---|---|---|---|---|---|---|")
for a, M in runs.items():
    for t in ("init", "after_warmup", "end"):
        m = M.get(t)
        if m:
            print(f"| {a} | {t} | {m['reserved'] / G:.2f} | {m['allocated'] / G:.2f} | {(m['reserved'] - m['allocated']) / G:.2f} | {(m['inactive_split'] or 0) / G:.2f} | "
                  f"{m['segments']} | {m['driver_minus_reserved'] / G:.2f} | {m['variants']} |")
for a, M in runs.items():
    for t in ("after_warmup", "end"):
        m = M.get(t)
        if not m:
            continue
        print(f"\n#### {a} {t}: pools (reserved / allocated GiB, segments)\n")
        for pid, p in list(m["pools"].items())[:8]:
            print(f"- {pid}: {p['reserved'] / G:.2f} / {p['allocated'] / G:.2f}, {p['segments']}")
        print(f"\n#### {a} {t}: active bytes by allocation site (pool, site, GiB)\n")
        for pid, site, b in m["top_sites"][:15]:
            print(f"- {pid} {site}: {b / G:.3f}")
