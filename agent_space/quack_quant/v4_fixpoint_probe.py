import torch
from torch._inductor import config, metrics, scheduler as S
from torch._inductor.utils import run_and_get_code
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
M, K = 2048, 3072
x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
for name, fn in (("quack to_mxfp4", qz.to_mxfp4),):
    torch._dynamo.reset(); metrics.reset()
    with config.patch({"triton.nested_reduction": True, "triton.cudagraphs": False, "fx_graph_cache": False}):
        out, srcs = run_and_get_code(torch.compile(fn, fullgraph=True, dynamic=False), x)
    print(f"{name}: kernels={metrics.generated_kernel_count} nested={metrics.codegen_nested_reduction}", flush=True)
