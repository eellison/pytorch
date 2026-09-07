import sys, torch
from torch._inductor import config, metrics, scheduler as S
from torch._inductor.utils import fresh_inductor_cache, run_and_get_code
from torch._dynamo.utils import dynamo_timed
from quack.blockscaled import quantize as qz

def fuse_nodes_fixpoint(self, nodes):
    with dynamo_timed("Scheduler.fused_nodes", log_pt2_compile_event=True, log_waitcounter=True):
        for i in range(10):
            old_len = len(nodes)
            nodes = self.fuse_nodes_once(nodes, is_reorder_round=False)
            if len(nodes) == old_len or len(nodes) == 1:
                break
        if config.loop_ordering_after_fusion or config.loop_index_inversion_in_fusion:
            for i in range(10):
                old_len = len(nodes)
                nodes = self.fuse_nodes_once(nodes, is_reorder_round=True)
                if len(nodes) == old_len or len(nodes) == 1:
                    break
        return nodes

M, K = 2048, 3072
x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
def as_bytes(t): return t.view(torch.uint8) if t.dtype != torch.float32 else t.view(torch.int32)
def exact(a, b):
    a = list(a) if isinstance(a, (tuple, list)) else [a]; b = list(b) if isinstance(b, (tuple, list)) else [b]
    return all(torch.equal(as_bytes(p), as_bytes(q)) for p, q in zip(a, b))
cases = {"mxfp4": (qz.to_mxfp4, False), "nvfp4": (qz.to_nvfp4, False), "mxfp8_e4m3": (qz.to_mx, False),
         "mxfp6_e2m3_packed": (qz.to_mxfp6_e2m3_packed, True), "mxfp6_e2m3_packed_static": (qz.to_mxfp6_e2m3_packed, False),
         "mxfp4_byte": (qz.to_mxfp4_byte, True), "mxfp8_dim0": (qz.to_mx_dim0, False)}
orig = S.Scheduler.fuse_nodes
for name, (fn, dyn) in cases.items():
    ref = fn(x)
    line = f"{name:26s} dyn={dyn!s:5s}"
    for label, impl in (("baseline", orig), ("fixpoint", fuse_nodes_fixpoint)):
        S.Scheduler.fuse_nodes = impl
        torch._dynamo.reset(); metrics.reset()
        with fresh_inductor_cache(), config.patch({"triton.nested_reduction": True, "triton.cudagraphs": False, "fx_graph_cache": False}):
            out, srcs = run_and_get_code(torch.compile(fn, fullgraph=True, dynamic=dyn), x)
        names = [l[4:l.index("(")][:16] for s in srcs for l in s.splitlines() if l.startswith("def triton_")]
        line += f" | {label}: k={metrics.generated_kernel_count} n={metrics.codegen_nested_reduction} eq={exact(out, ref)} {names}"
    print(line, flush=True)
S.Scheduler.fuse_nodes = orig
