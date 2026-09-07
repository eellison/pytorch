import re, sys, torch
from torch._inductor import config, metrics
from torch._inductor.utils import fresh_inductor_cache, run_and_get_code
from quack.blockscaled import quantize as qz
M, K = 2048, 3072
x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
for dyn in (True, False):
    torch._dynamo.reset(); metrics.reset()
    with fresh_inductor_cache(), config.patch({"triton.nested_reduction": True, "triton.cudagraphs": False, "fx_graph_cache": False}):
        out, srcs = run_and_get_code(torch.compile((qz.to_mx if sys.argv[1] == "mx" else qz.to_mxfp4_byte), fullgraph=True, dynamic=dyn), x)
    print(f"===== dynamic={dyn} kernels={metrics.generated_kernel_count}")
    src = "\n".join(srcs)
    for m in re.finditer(r"@triton_heuristics\.(\w+)\(\n(.*?)\n\)\n@triton\.jit\ndef (triton_\w+)\((.*?)\):", src, re.S):
        kind, body, name, sig = m.groups()
        hints = re.search(r"size_hints=(\{[^}]*\})", body); rh = re.search(r"reduction_hint=(\S+),", body)
        print(f"  {kind:22s} {name[:40]:40s} size_hints={hints.group(1) if hints else '?'} {rh.group(1) if rh else ''}")
        print(f"      args: {sig[:150]}")
    for line in src.splitlines():
        if re.match(r"\s+triton_\w+\.run\(", line) or "xnumel = " in line and "def " not in line:
            print("   ", line.strip()[:160])
