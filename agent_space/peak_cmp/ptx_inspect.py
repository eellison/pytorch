import re, os, sys, glob, torch
sys.argv = ["x", "rmsnorm_nvfp4", "8192", "4096", "ours", "default"]
src = open("agent_space/peak_cmp/peak_cell.py").read().split("torch.manual_seed(0)")[0]
exec(src)
from torch._inductor.codecache import PyCodeCache
M, K = 8192, 4096
x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16); w = torch.randn(K, device="cuda", dtype=torch.bfloat16)
cases = {"rms_nvfp4": (lambda x, w: QUANT["nvfp4"](rms(x, w)), (x, w)), "rms_mxfp8": (lambda x, w: QUANT["mxfp8"](rms(x, w)), (x, w)),
         "quant_nvfp4": (QUANT["nvfp4"], (x,)), "quant_mxfp8": (QUANT["mxfp8"], (x,))}
which = sys.argv[1:] if len(sys.argv) > 1 else list(cases)
for name in cases:
    fn, args = cases[name]
    cache = f"/tmp/triton_ptx_{name}"; os.environ["TRITON_CACHE_DIR"] = cache
    import shutil; shutil.rmtree(cache, ignore_errors=True)
    torch._dynamo.reset(); metrics.reset(); start = len(PyCodeCache.modules)
    with fresh_inductor_cache(), config.patch({"triton.nested_reduction": True, "triton.cudagraphs": False, "fx_graph_cache": False}):
        compiled = torch.compile(fn, fullgraph=True, dynamic=False); out, srcs = run_and_get_code(compiled, *args); compiled(*args); torch.cuda.synchronize()
    cfgs = []; ptx_texts = {}
    for mod in PyCodeCache.modules[start:]:
        for n, kern in vars(mod).items():
            if n.startswith("triton_") and hasattr(kern, "compile_results"):
                for cr in kern.compile_results:
                    cfg = cr.config
                    cfgs.append((dict(cfg.kwargs), cfg.num_warps, cfg.num_stages, cr.kernel.metadata.num_regs if hasattr(cr.kernel, "metadata") else None))
                    ptx_texts[f"{n[:24]}|{dict(cfg.kwargs)}|w{cfg.num_warps}"] = cr.kernel.asm.get("ptx", "")
    k = [s for s in srcs if "@triton_heuristics" in s][0]
    kind = "persistent" if "persistent_reduction" in k else "looped"
    stats = {}
    for key, t in ptx_texts.items():
        c = lambda pat: len(re.findall(pat, t))
        stats[key] = dict(ld_g_v4=c(r"ld\.global\.[\w.]*v4"), ld_g_v2=c(r"ld\.global\.[\w.]*v2"), ld_g_scalar=c(r"ld\.global\.(?!.*v[24])\S+"),
            st_g_v4=c(r"st\.global\.[\w.]*v4"), st_g_v2=c(r"st\.global\.[\w.]*v2"), st_g_u8=c(r"st\.global\.(?:\w+\.)?u8 "), st_g_b8=c(r"st\.global\.(?:\w+\.)?b8 "), st_g_scalar=c(r"st\.global\.(?!.*v[24])\S+"),
            ld_shared=c(r"ld\.shared"), st_shared=c(r"st\.shared"), bar_sync=c(r"bar\.sync|barrier\.sync"), shfl=c(r"shfl\.sync"), cvt_e2m1=c(r"e2m1x2"), cvt_e4m3=c(r"cvt\.rn\.satfinite\.e4m3x2"), regs=re.search(r"\.reg \.b32 \t%r<(\d+)>", t).group(1) if re.search(r"\.reg \.b32 \t%r<(\d+)>", t) else None, lines=t.count("\n"))
    print(f"== {name}: {kind} k={metrics.generated_kernel_count} cfg={cfgs}")
    for f, st in stats.items(): print(f"   {f}: {st}")
