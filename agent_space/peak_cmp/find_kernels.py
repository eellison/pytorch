import sys, torch
sys.argv = ["x", "rmsnorm_nvfp4", "8192", "4096", "ours", "default"]
exec(open("agent_space/peak_cmp/peak_cell.py").read().split("torch.manual_seed(0)")[0])
from torch._inductor.codecache import PyCodeCache
x = torch.randn(8192, 4096, device="cuda", dtype=torch.bfloat16); w = torch.randn(4096, device="cuda", dtype=torch.bfloat16)
fn = lambda x, w: QUANT["nvfp4"](rms(x, w))
n0 = len(PyCodeCache.modules)
with fresh_inductor_cache(), config.patch({"triton.nested_reduction": True, "triton.cudagraphs": False, "fx_graph_cache": False}):
    compiled = torch.compile(fn, fullgraph=True, dynamic=False); compiled(x, w); torch.cuda.synchronize()
    print("modules before/after:", n0, len(PyCodeCache.modules))
    for i, m in enumerate(PyCodeCache.modules[n0:]):
        names = [n for n in vars(m) if n.startswith("triton_") or n == "async_compile"]
        print(i, getattr(m, "__file__", "?")[-60:], names[:5])
        for n in names:
            obj = getattr(m, n)
            if hasattr(obj, "compile_results") or hasattr(obj, "launchers"):
                print("   ", n[:30], type(obj).__name__, "launchers:", len(getattr(obj, "launchers", [])), "compile_results:", len(getattr(obj, "compile_results", [])), [a for a in dir(obj) if "config" in a.lower()][:8])
