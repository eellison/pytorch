# Kernel time of one profiled vLLM step (drive_attn --profile-kinds): python prof_attn.py DIR [--top N]
# Attention = kernels matching ATTN (FlashInfer trtllm-gen fmha, FA4 CuTe DSL, Triton unified attention, flex/Inductor
# attention templates, split-KV combines); cache = KV-cache writes. Prints per trace: total, attention, cache, top kernels.
import collections, glob, json, os, re, sys

ATTN = re.compile(r"fmha|trtllm|Fmha|flash_fwd|FlashAttention|flash_attn|unified_attention|reduce_segments|flex_attention|flex_decoding|triton_tem_fused|FwdCombine|fwd_combine|BatchDecode|BatchPrefill", re.I)
CACHE = re.compile(r"reshape_and_cache|concat_and_cache", re.I)


def load(path):
    ev = json.load(open(path))["traceEvents"]
    tot, cnt = collections.Counter(), collections.Counter()
    for e in ev:
        if e.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset") or e.get("ph") != "X":
            continue
        name = e["name"]
        tot[name] += e["dur"]
        cnt[name] += 1
    return tot, cnt


def summarize(path, top=8):
    tot, cnt = load(path)
    a = {n: v for n, v in tot.items() if ATTN.search(n)}
    c = {n: v for n, v in tot.items() if CACHE.search(n)}
    return {"total_us": round(sum(tot.values()), 1), "kernels": sum(cnt.values()), "attn_us": round(sum(a.values()), 1), "attn_launches": sum(cnt[n] for n in a),
            "cache_us": round(sum(c.values()), 1), "attn_kernels": {n[:110]: (round(v, 1), cnt[n]) for n, v in sorted(a.items(), key=lambda x: -x[1])},
            "top": [(n[:90], round(v, 1), cnt[n]) for n, v in tot.most_common(top)]}


if __name__ == "__main__":
    top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 8
    out = {}
    for d in [x for x in sys.argv[1:] if os.path.isdir(x)]:
        for f in sorted(glob.glob(os.path.join(d, "*.json"))):
            s = out[os.path.relpath(f)] = summarize(f, top)
            print(f"{os.path.relpath(f)}: total {s['total_us']} us ({s['kernels']} kernels), attention {s['attn_us']} us ({s['attn_launches']}), cache write {s['cache_us']} us")
            for n, (v, k) in s["attn_kernels"].items():
                print(f"    attn {v:9.1f} us x{k:4d}  {n}")
            for n, v, k in s["top"]:
                print(f"    top  {v:9.1f} us x{k:4d}  {n}")
    if "--json" in sys.argv:
        json.dump(out, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)
