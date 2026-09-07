import re, os, sys, glob, time, torch
case = sys.argv[1]
T0 = time.time()
sys.argv = ["x", "rmsnorm_nvfp4", "8192", "4096", "ours", "default"]
exec(open("agent_space/peak_cmp/peak_cell.py").read().split("torch.manual_seed(0)")[0])
M, K = 8192, 4096
x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16); w = torch.randn(K, device="cuda", dtype=torch.bfloat16)
cases = {"rms_nvfp4": (lambda x, w: QUANT["nvfp4"](rms(x, w)), (x, w)), "rms_mxfp8": (lambda x, w: QUANT["mxfp8"](rms(x, w)), (x, w)),
         "quant_nvfp4": (QUANT["nvfp4"], (x,)), "quant_mxfp8": (QUANT["mxfp8"], (x,))}
fn, args = cases[case]
with config.patch({"triton.nested_reduction": True, "triton.cudagraphs": False, "fx_graph_cache": False}):
    compiled = torch.compile(fn, fullgraph=True, dynamic=False); out, srcs = run_and_get_code(compiled, *args); torch.cuda.synchronize()
k = [s for s in srcs if "@triton_heuristics" in s][0]
kind = "persistent" if "persistent_reduction" in k else "looped"
cache = "/tmp/torchinductor_eellison"
seen = set()
print(f"== {case}: {kind}")
for p in sorted(glob.glob(f"{cache}/**/*.ptx", recursive=True)):
    if os.path.getmtime(p) < T0: continue
    t = open(p).read()
    if "triton_" not in t[:4000]: continue
    meta = {}
    for j in glob.glob(os.path.dirname(p) + "/*.json"):
        try:
            import json; d = json.load(open(j)); meta = {k2: d.get(k2) for k2 in ("num_warps", "num_stages", "shared")}
        except Exception: pass
    c = lambda pat: len(re.findall(pat, t))
    regs = re.search(r"\.reg \.b32 \t%r<(\d+)>", t)
    st = dict(ld_g_v4=c(r"ld\.global\.[\w.]*\.v4"), ld_g_v2=c(r"ld\.global\.[\w.]*\.v2"), ld_g_1=c(r"ld\.global\.(?:[\w]+\.)*(?:u16|b16|u8|b8|u32|b32|f32) "), st_g_v4=c(r"st\.global\.[\w.]*\.v4"), st_g_v2=c(r"st\.global\.[\w.]*\.v2"), st_g_u8=c(r"st\.global\.(?:[\w]+\.)*(?:u8|b8) "), st_g_u16=c(r"st\.global\.(?:[\w]+\.)*(?:u16|b16) "), st_g_u32=c(r"st\.global\.(?:[\w]+\.)*(?:u32|b32|f32) "),
              ld_sh=c(r"ld\.shared"), st_sh=c(r"st\.shared"), bar=c(r"bar\.sync|barrier\.sync"), shfl=c(r"shfl\.sync"), e2m1=c(r"e2m1x2"), e4m3=c(r"e4m3x2"), prmt=c(r"prmt\.b32"), bfe=c(r"bfe\.|bfi\."), lines=t.count("\n"))
    print(f"   {meta} regs<{regs.group(1) if regs else '?'}> {st}")
