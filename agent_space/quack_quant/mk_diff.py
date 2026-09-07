import re, torch, torch.nn.functional as F, difflib
from torch._inductor import config, metrics
from torch._inductor.utils import fresh_inductor_cache, run_and_get_code
from quack.blockscaled import quantize as qz
M, K = 8192, 4096
x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16); w = torch.randn(K, device="cuda", dtype=torch.bfloat16)
fn = lambda x, w: qz.to_mx(F.rms_norm(x, (K,), w))
srcs = {}
for mode in ("on", "on_mk"):
    torch._dynamo.reset(); metrics.reset()
    with fresh_inductor_cache(), config.patch({"triton.nested_reduction": True, "triton.cudagraphs": False, "fx_graph_cache": False, "triton.multi_kernel": 1 if mode == "on_mk" else 0}):
        out, s = run_and_get_code(torch.compile(fn, fullgraph=True, dynamic=False), x, w)
    src = "\n".join(s)
    kernels = re.findall(r"(@triton_heuristics\.\w+\(.*?\n\)\n@triton\.jit\ndef triton_\w+\(.*?)(?=\n''', device_str)", src, re.S)
    srcs[mode] = kernels
    print(f"{mode}: {len(kernels)} kernels: {[re.search(r'def (triton_\w+)', k).group(1)[:20] for k in kernels]}")
    open(f"agent_space/quack_quant/mk_diff_{mode}.py", "w").write(src)
per_on = [k for k in srcs["on"] if "persistent_reduction" in k][0]
per_mk = [k for k in srcs["on_mk"] if "persistent_reduction" in k][0]
norm = lambda k: [re.sub(r"triton_\w+_\d+", "K", l) for l in k.splitlines()]
d = list(difflib.unified_diff(norm(per_on), norm(per_mk), "on/persistent", "on_mk/persistent", lineterm="", n=1))
print("\n".join(l[:200] for l in d[:80]))
