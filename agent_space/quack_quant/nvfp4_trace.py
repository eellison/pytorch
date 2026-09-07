import torch
from torch._inductor import config, metrics, scheduler as S
from torch._inductor.utils import fresh_inductor_cache, run_and_get_code
from torch._dynamo.utils import dynamo_timed
from quack.blockscaled import quantize as qz
def fuse_nodes_fixpoint(self, nodes):
    with dynamo_timed("Scheduler.fused_nodes", log_pt2_compile_event=True, log_waitcounter=True):
        for i in range(10):
            old_len = len(nodes); nodes = self.fuse_nodes_once(nodes, is_reorder_round=False)
            if len(nodes) == old_len or len(nodes) == 1: break
        if config.loop_ordering_after_fusion or config.loop_index_inversion_in_fusion:
            for i in range(10):
                old_len = len(nodes); nodes = self.fuse_nodes_once(nodes, is_reorder_round=True)
                if len(nodes) == old_len or len(nodes) == 1: break
        return nodes
S.Scheduler.fuse_nodes = fuse_nodes_fixpoint
NR = S.NestedReduction
STEPS = ["_sub_parent_epilogue_candidate_nodes", "_sub_parent_internal_access_relations", "_try_get_sub_parent_access_relations",
         "_sub_parent_broadcast_access_relations", "_order_sub_parent_parent_nodes", "_sub_parent_epilogue_outputs_unread"]
def wrap(name):
    orig = getattr(NR, name)
    def inner(*a, **k):
        r = orig(*a, **k)
        print(f"    {'OK  ' if r is not None and r is not False else 'NONE'} {name} -> {type(r).__name__}{'' if r is None else ' len=' + str(len(r)) if hasattr(r, '__len__') else ''}")
        if r is None and name in ("_try_get_sub_parent_access_relations", "_sub_parent_broadcast_access_relations", "_order_sub_parent_parent_nodes"):
            parent_nodes, epilogue_nodes = a[0], a[1]
            print(f"         parent_source/internal names arg: {a[4] if len(a) > 4 else a[2]}")
            for n in list(parent_nodes) + list(epilogue_nodes):
                print(f"         {'P' if n in parent_nodes else 'E'} {n.get_name()} ranges={n.get_ranges()}")
                for d in n.read_writes.reads: print(f"            read  {d}")
                for d in n.read_writes.writes: print(f"            write {d}")
        return r
    return inner
for s in STEPS:
    setattr(NR, s, staticmethod(wrap(s)) if isinstance(NR.__dict__[s], staticmethod) else classmethod(lambda cls, *a, _f=wrap(s), **k: _f(*a, **k)))
orig_plan = NR.sub_parent_epilogue_plan
def plan(cls, nodes, numel, rnumel):
    print(f"  plan nodes={[n.get_name() for n in nodes]} numel={numel} rnumel={rnumel}")
    r = orig_plan.__func__(cls, nodes, numel, rnumel); print(f"  -> {'PLAN' if r is not None else 'None'}"); return r
NR.sub_parent_epilogue_plan = classmethod(plan)
x = torch.randn(2048, 3072, device="cuda", dtype=torch.bfloat16)
torch._dynamo.reset(); metrics.reset()
with fresh_inductor_cache(), config.patch({"triton.nested_reduction": True, "triton.cudagraphs": False, "fx_graph_cache": False}):
    run_and_get_code(torch.compile(qz.to_nvfp4, fullgraph=True, dynamic=False), x)
print(f"RESULT kernels={metrics.generated_kernel_count} nested={metrics.codegen_nested_reduction}")
