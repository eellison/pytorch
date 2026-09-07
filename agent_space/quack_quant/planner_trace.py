import functools, torch
from torch._inductor import config, metrics, scheduler as S
from torch._inductor.utils import fresh_inductor_cache, run_and_get_code
from quack.blockscaled import quantize as qz

M, K, BS = 2048, 3072, 32
x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
NR = S.NestedReduction
STEPS = ["_mutations_survive_hoisting", "_sub_parent_epilogue_candidate_nodes", "_sub_parent_internal_access_relations",
         "_try_get_sub_parent_access_relations", "_sub_parent_broadcast_access_relations", "_order_sub_parent_parent_nodes",
         "_sub_parent_epilogue_outputs_unread"]
def wrap(name):
    orig = getattr(NR, name)
    def inner(*a, **k):
        r = orig(*a, **k)
        flag = "OK  " if r not in (None, False, ()) else "NONE"
        print(f"    {flag} {name}")
        if name == "_sub_parent_epilogue_candidate_nodes" and r is None:
            nodes = a[0]
            for n in nodes:
                print(f"         node {n.get_name()} group={n.group} ranges={n.get_ranges()} reduction={n.is_reduction()}")
        if name in ("_try_get_sub_parent_access_relations", "_sub_parent_broadcast_access_relations") and r in (None, ()):
            parent_nodes, epilogue_nodes = a[0], a[1]
            for n in epilogue_nodes:
                print(f"         epilogue {n.get_name()} ranges={n.get_ranges()}")
                for d in n.read_writes.reads:
                    print(f"            read {d}")
        return r
    return inner
orig_plan = NR.sub_parent_epilogue_plan
def plan(cls, nodes, numel, rnumel):
    print(f"  sub_parent_epilogue_plan nodes={[n.get_name() for n in nodes]} numel={numel} rnumel={rnumel}")
    r = orig_plan.__func__(cls, nodes, numel, rnumel)
    print(f"  -> {'PLAN' if r is not None else 'None'}")
    return r
NR.sub_parent_epilogue_plan = classmethod(plan)
for s in STEPS:
    setattr(NR, s, staticmethod(wrap(s)) if isinstance(NR.__dict__[s], staticmethod) else classmethod(lambda cls, *a, _f=wrap(s), **k: _f(*a, **k)))

def scale_and_hp(x):
    data_hp = x.reshape(M, K // BS, BS)
    max_abs = torch.amax(torch.abs(data_hp), -1).unsqueeze(-1).to(torch.float32)
    sb = qz._compute_e8m0_scale_floor(max_abs, qz.F4_E2M1_MAX_POW2)
    scale_fp32 = torch.clamp((sb.to(torch.int32) << qz.MBITS_F32).view(torch.float32), min=qz.F32_MIN_NORMAL)
    return data_hp.to(torch.float32) / scale_fp32, sb.view(torch.float8_e8m0fnu).squeeze(-1)
def v4(x):
    data_lp, scale = scale_and_hp(x)
    codes = data_lp.reshape(M, K).clamp(0, 15).to(torch.uint8)
    c = codes.view(M, K // 2, 2)
    return c[..., 0] | (c[..., 1] << 4), scale
def v5(x):
    data_lp, scale = scale_and_hp(x)
    c = data_lp.view(M, K // BS, BS // 2, 2).clamp(0, 15).to(torch.uint8)
    return (c[..., 0] | (c[..., 1] << 4)).view(M, K // 2), scale
def v1(x):
    return qz.to_mxfp4(x)
for fn in (v4, v5, v1):
    print(f"=== {fn.__name__}")
    torch._dynamo.reset(); metrics.reset()
    with fresh_inductor_cache(), config.patch({"triton.nested_reduction": True, "triton.cudagraphs": False, "fx_graph_cache": False}):
        run_and_get_code(torch.compile(fn, fullgraph=True, dynamic=False), x)
    print(f"  kernels={metrics.generated_kernel_count} nested={metrics.codegen_nested_reduction}")
