import sys, torch
from torch._inductor import scheduler as S
exec(open("agent_space/peak_cmp/colwise_probe.py").read().split("M, K = int(sys.argv[1])")[0])
NR = S.NestedReduction
def wrap(name):
    orig = NR.__dict__[name]
    fn = orig.__func__ if isinstance(orig, (classmethod, staticmethod)) else orig
    is_cls = isinstance(orig, classmethod)
    def inner(*a, **k):
        args = a[1:] if is_cls else a
        r = fn(*a, **k) if is_cls else fn(*a, **k)
        desc = ", ".join(x.get_name() if hasattr(x, "get_name") else (str(x)[:30]) for x in args)
        print(f"    {name}({desc[:110]}) -> {str(r)[:120] if not hasattr(r, 'parent_nodes') else 'PLAN'}", flush=True)
        return r
    setattr(NR, name, classmethod(inner) if is_cls else (staticmethod(inner) if isinstance(orig, staticmethod) else inner))
for n in ("_get_grouped_reduction_and_size", "get_grouped_axis", "_min_block_unprofitable_for_kernel", "plan_from_topology", "plan"):
    if n in NR.__dict__: wrap(n)
    else: print("missing", n)
M, K = 2048, 4096
torch.manual_seed(0)
x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16); w = torch.randn(K, device="cuda", dtype=torch.bfloat16)
torch._dynamo.reset(); metrics.reset()
with config.patch({"triton.nested_reduction": True, "triton.nested_reduction_allow_x": True, "triton.cudagraphs": False, "fx_graph_cache": False}):
    out, srcs = run_and_get_code(torch.compile(lambda x, w: mxfp8_colwise(rms(x, w)), fullgraph=True, dynamic=False), x, w)
print(f"RESULT kernels={metrics.generated_kernel_count} nested={metrics.codegen_nested_reduction}")
