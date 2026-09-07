import statistics, torch
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
def graph_bench(fn, warmup=20, samples=50, calls=100):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(calls): fn()
    g.replay(); torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True); vals = []
    for _ in range(samples):
        s.record(); g.replay(); e.record(); e.synchronize(); vals.append(s.elapsed_time(e) * 1000 / calls)
    return statistics.median(vals)
orig = S.Scheduler.fuse_nodes
for M, K in ((2048, 3072), (8192, 4096), (65536, 2048)):
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    ref = qz.to_mxfp4(x)
    line = f"to_mxfp4 {M}x{K}:"
    for label, impl in (("baseline", orig), ("fixpoint", fuse_nodes_fixpoint)):
        S.Scheduler.fuse_nodes = impl
        torch._dynamo.reset(); metrics.reset()
        with fresh_inductor_cache(), config.patch({"triton.nested_reduction": True, "triton.cudagraphs": False, "fx_graph_cache": False}):
            compiled = torch.compile(qz.to_mxfp4, fullgraph=True, dynamic=False)
            out, _ = run_and_get_code(compiled, x)
            eq = all(torch.equal(a.view(torch.uint8), b.view(torch.uint8)) for a, b in zip(out, ref))
            us = graph_bench(lambda: compiled(x))
        line += f"  {label}: {us:.1f}us k={metrics.generated_kernel_count} n={metrics.codegen_nested_reduction} eq={eq}"
    print(line, flush=True)
S.Scheduler.fuse_nodes = orig
