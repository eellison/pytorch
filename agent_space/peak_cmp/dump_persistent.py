import sys, torch, contextlib, re
sys.argv = ["x", "rmsnorm_nvfp4", "8192", "4096", "ours", "default"]
exec(open("agent_space/peak_cmp/peak_cell.py").read().split("torch.manual_seed(0)")[0])
M, K = 8192, 4096
x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16); w = torch.randn(K, device="cuda", dtype=torch.bfloat16)
fn = lambda x, w: QUANT["nvfp4"](rms(x, w))
class _P(InductorChoices):
    @staticmethod
    def should_use_persistent_reduction(*a, **k): return True
with config.patch({"triton.nested_reduction": True, "triton.cudagraphs": False, "fx_graph_cache": False}), V.set_choices_handler(_P()):
    compiled = torch.compile(fn, fullgraph=True, dynamic=False); out, srcs = run_and_get_code(compiled, x, w)
src = [s for s in srcs if "@triton_heuristics" in s][0]
open("agent_space/peak_cmp/gen_persistent_nvfp4.py", "w").write(src)
body = src[src.index("@triton.jit"):src.index("''', device_str")]
print(body)
