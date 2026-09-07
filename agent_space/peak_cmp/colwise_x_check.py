import sys, torch
exec(open("agent_space/peak_cmp/colwise_probe.py").read().split("M, K = int(sys.argv[1])")[0])
M, K = 8192, 4096
torch.manual_seed(0)
x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16); w = torch.randn(K, device="cuda", dtype=torch.bfloat16)
fn = lambda x, w: mxfp8_colwise(rms(x, w))
outs = {}
for label, nested, allow in (("unnested", False, False), ("nested_R_only", True, False), ("nested_X", True, True)):
    torch._dynamo.reset(); metrics.reset()
    with config.patch({"triton.nested_reduction": nested, "triton.nested_reduction_allow_x": allow, "triton.cudagraphs": False, "fx_graph_cache": False}):
        out, srcs = run_and_get_code(torch.compile(fn, fullgraph=True, dynamic=False), x, w)
    outs[label] = (out, srcs)
    print(f"{label}: kernels={metrics.generated_kernel_count} nested={metrics.codegen_nested_reduction}")
q_ref, s_ref = outs["unnested"][0]; q_x, s_x = outs["nested_X"][0]
print("scale exact:", torch.equal(s_ref.view(torch.uint8), s_x.view(torch.uint8)), "| scale bytes differing:", int((s_ref.view(torch.uint8) != s_x.view(torch.uint8)).sum()), "of", s_ref.numel())
dq = (q_ref.view(torch.uint8) != q_x.view(torch.uint8)); print("q bytes differing:", int(dq.sum()), "of", q_ref.numel(), f"({100*dq.float().mean().item():.4f}%)")
rows = dq.any(dim=1).nonzero().flatten(); cols = dq.any(dim=0).nonzero().flatten()
print("differing rows:", rows.numel(), "first:", rows[:8].tolist(), "| differing cols:", cols.numel(), "first:", cols[:8].tolist())
# dequant compare against fp32 reference
y = torch.nn.functional.rms_norm(x.float(), (K,), w.float(), eps=1e-6)
def deq(q, s):
    sf = torch.ldexp(torch.ones_like(s.view(torch.uint8), dtype=torch.float32), s.view(torch.uint8).to(torch.int32) - 127)  # (M/32, K)
    return q.float().view(M // 32, 32, K) * sf.unsqueeze(1)
e_ref = (deq(q_ref, s_ref).view(M, K) - y).abs(); e_x = (deq(q_x, s_x).view(M, K) - y).abs()
print(f"dequant mean|err| unnested={e_ref.mean().item():.5g} nested_X={e_x.mean().item():.5g}; max unnested={e_ref.max().item():.4g} nested_X={e_x.max().item():.4g}")
if dq.any():
    r, c = dq.nonzero()[0].tolist()
    print(f"example [{r},{c}]: x={x[r,c].item():.5g} y={y[r,c].item():.5g} q_ref={q_ref[r,c].float().item()} q_x={q_x[r,c].float().item()} scale_ref={s_ref.view(torch.uint8)[r//32,c].item()} scale_x={s_x.view(torch.uint8)[r//32,c].item()}")
src = [s for s in outs["nested_X"][1] if "@triton_heuristics" in s][0]
body = src[src.index("@triton.jit"):]
print("---- nested_X kernel (structure) ----")
for l in body.splitlines():
    t = l.strip()
    if any(k in t for k in ("def triton_", "XBLOCK", "R0_BLOCK", "for r0_offset", "tl.load(", "tl.store(", "max2(", "tl.sum(", "rsqrt", "reshape(", "min_xblock", "size_hints")) and "constexpr" not in t: print("   ", t[:150])
import re; print("   meta:", re.search(r"'min_xblock': \d+|'min_rblock': \d+", src), re.search(r"size_hints=(\{[^}]*\})", src).group(1))
