import sys, torch
from torch._inductor import scheduler as S
exec(open("agent_space/peak_cmp/colwise_probe.py").read().split("M, K = int(sys.argv[1])")[0])
NR = S.NestedReduction
for name in ("is_candidate", "plan", "_is_dependent_reduction_pair", "sub_parent_epilogue_plan"):
    orig = getattr(NR, name)
    def make(name, orig):
        def inner(cls, *a, **k):
            r = orig.__func__(cls, *a, **k) if hasattr(orig, "__func__") else orig(*a, **k)
            nodes = [x for x in a if hasattr(x, "get_name")]
            print(f"    {name}({', '.join(n.get_name() for n in nodes)}) -> {r if not hasattr(r, 'parent_nodes') else 'PLAN'}")
            return r
        return inner
    setattr(NR, name, classmethod(make(name, orig)))
orig_grouped = NR._get_grouped_reduction_and_size
def grouped(cls, *a, **k):
    r = orig_grouped.__func__(cls, *a, **k)
    print(f"    _get_grouped_reduction_and_size -> {r}")
    return r
NR._get_grouped_reduction_and_size = classmethod(grouped)
M, K = 2048, 4096
torch.manual_seed(0)
x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16); w = torch.randn(K, device="cuda", dtype=torch.bfloat16)
torch._dynamo.reset(); metrics.reset()
with config.patch({"triton.nested_reduction": True, "triton.cudagraphs": False, "fx_graph_cache": False}):
    out, srcs = run_and_get_code(torch.compile(lambda x, w: mxfp8_colwise(rms(x, w)), fullgraph=True, dynamic=False), x, w)
print(f"RESULT kernels={metrics.generated_kernel_count} nested={metrics.codegen_nested_reduction}")
