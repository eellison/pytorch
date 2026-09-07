import re, sys, torch
sys.argv = ["x", "quant_nvfp4", "8192", "4096", "ours", "default"]
src = open("agent_space/peak_cmp/peak_cell.py").read().split("torch.manual_seed(0)")[0]
exec(src)
from torch._inductor.codecache import PyCodeCache
x = torch.randn(8192, 4096, device="cuda", dtype=torch.bfloat16)
for name, fn in (("nvfp4", QUANT["nvfp4"]), ("mxfp8", QUANT["mxfp8"])):
    torch._dynamo.reset(); metrics.reset(); start = len(PyCodeCache.modules)
    with fresh_inductor_cache(), config.patch({"triton.nested_reduction": True, "triton.cudagraphs": False, "fx_graph_cache": False}):
        out, srcs = run_and_get_code(torch.compile(fn, fullgraph=True, dynamic=False), x)
    k = [s for s in srcs if "@triton_heuristics" in s][0]
    body = k[k.index("@triton.jit"):]
    loads = re.findall(r"tl\.load\((\w+)", body)
    cfgs = []
    for mod in PyCodeCache.modules[start:]:
        for n, kern in vars(mod).items():
            if n.startswith("triton_") and hasattr(kern, "launchers"):
                for l in kern.launchers: cfgs.append((dict(l.config.kwargs), l.config.num_warps, l.n_regs, l.n_spills))
    print(f"== {name}: kind={'persistent' if 'persistent_reduction' in k else 'looped'} loads={loads} tl.split={body.count('tl.split(')} inline_asm={body.count('inline_asm')} stores={body.count('tl.store(')} cfg={cfgs}")
    print("   size_hints:", re.search(r"size_hints=(\{[^}]*\})", k).group(1), "| min_rblock:", re.search(r"'min_rblock': (\d+)", k).group(1) if "'min_rblock'" in k else None)
    for line in body.splitlines():
        if "tl.load(" in line or "for r0_offset" in line or "tl.split(" in line or "reshape(" in line: print("     ", line.strip()[:150])
