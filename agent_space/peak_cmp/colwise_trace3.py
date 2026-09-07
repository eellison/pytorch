import sys, torch, inspect
from torch._inductor import scheduler as S
exec(open("agent_space/peak_cmp/colwise_probe.py").read().split("M, K = int(sys.argv[1])")[0])
NR = S.NestedReduction
depth = [0]
for name, attr in list(NR.__dict__.items()):
    if name.startswith("__") or not isinstance(attr, (classmethod, staticmethod)): continue
    fn = attr.__func__; is_cls = isinstance(attr, classmethod)
    def make(name, fn, is_cls):
        def inner(*a, **k):
            depth[0] += 1
            try: r = fn(*a, **k)
            except Exception as ex:
                print(f"{'  '*depth[0]}{name} RAISED {type(ex).__name__}: {str(ex)[:100]}", flush=True); depth[0] -= 1; raise
            args = a[1:] if is_cls else a
            if r is None or r is False or (isinstance(r, tuple) and len(r) == 0):
                desc = ", ".join(x.get_name() if hasattr(x, "get_name") else type(x).__name__ for x in args)
                print(f"{'  '*depth[0]}{name}({desc[:90]}) -> {r!r}", flush=True)
            depth[0] -= 1
            return r
        return classmethod(inner) if is_cls else staticmethod(inner)
    setattr(NR, name, make(name, fn, is_cls))
M, K = 2048, 4096
torch.manual_seed(0)
x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16); w = torch.randn(K, device="cuda", dtype=torch.bfloat16)
torch._dynamo.reset(); metrics.reset()
with config.patch({"triton.nested_reduction": True, "triton.nested_reduction_allow_x": True, "triton.cudagraphs": False, "fx_graph_cache": False}):
    out, srcs = run_and_get_code(torch.compile(lambda x, w: mxfp8_colwise(rms(x, w)), fullgraph=True, dynamic=False), x, w)
print(f"RESULT kernels={metrics.generated_kernel_count} nested={metrics.codegen_nested_reduction}")
