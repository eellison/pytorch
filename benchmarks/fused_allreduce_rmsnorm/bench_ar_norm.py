"""Benchmark Lamport vs Kraken vs FlashInfer with CUDA graphs."""
import gc, os
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from push_poll_allreduce_rmsnorm import lamport_ar_bias_rmsnorm, LamportWorkspace
from kraken.fused.one_shot_all_reduce_bias_rms_norm import one_shot_all_reduce_bias_rms_norm
from kraken.fused.rms_norm import rms_norm as kraken_rms_norm
import flashinfer
from flashinfer.comm.allreduce import AllReduceFusionPattern, allreduce_fusion, create_allreduce_fusion_workspace


def bench_cudagraph(fn, iters=200, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    l2 = torch.empty(40*1024*1024, dtype=torch.int8, device="cuda")
    start = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        l2.zero_(); start[i].record(); g.replay(); end[i].record()
    torch.cuda.synchronize()
    times = sorted([s.elapsed_time(e)*1000 for s, e in zip(start, end)])
    return times[len(times)//2]


def bench_eager(fn, iters=200, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    l2 = torch.empty(40*1024*1024, dtype=torch.int8, device="cuda")
    start = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        l2.zero_(); start[i].record(); fn(); end[i].record()
    torch.cuda.synchronize()
    times = sorted([s.elapsed_time(e)*1000 for s, e in zip(start, end)])
    return times[len(times)//2]


def try_bench(fn, name, rank):
    try:
        return bench_cudagraph(fn)
    except Exception as e:
        if rank == 0:
            print(f"  [cudagraph failed {name}: {e}]")
        return bench_eager(fn)


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    symm_mem.enable_symm_mem_for_group(dist.group.WORLD.group_name)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dtype = torch.bfloat16; D = 5120; eps = 1e-5

    ws = LamportWorkspace(max_N=64, D=D, dtype=dtype, device=device)
    fi_ws = create_allreduce_fusion_workspace(
        backend="trtllm", world_size=world_size, rank=rank,
        max_token_num=64, hidden_dim=D, dtype=dtype, force_oneshot_support=True)
    torch.cuda.synchronize(); dist.barrier()

    if rank == 0:
        print(f"4x {torch.cuda.get_device_name(device)}, D={D}, CUDA graphs")
        print(f"\n{'b':>4} | {'nccl':>8} {'kraken':>8} {'lamport':>8} {'flashinfer':>10}")
        print("-" * 52)

    for b in [1, 2, 4, 8, 16, 32, 64]:
        gc.disable()
        torch.manual_seed(42)
        w = torch.randn(D, dtype=dtype, device=device)
        bias = torch.randn(b, D, dtype=dtype, device=device)
        torch.manual_seed(42 + rank)
        x = torch.randn(b, D, dtype=dtype, device=device)

        # NCCL
        x_n = x.clone(); y_n = torch.empty_like(x)
        def run_nccl():
            x_n.copy_(x); dist.all_reduce(x_n); x_n.add_(bias)
            y_n.copy_(kraken_rms_norm(x_n, w, eps))
        t_nccl = try_bench(run_nccl, "nccl", rank)

        # Kraken
        kr_buf = symm_mem.empty(b, D, dtype=dtype, device=device)
        symm_mem.rendezvous(kr_buf, dist.group.WORLD.group_name)
        y_kr = torch.empty_like(x)
        def run_kr():
            one_shot_all_reduce_bias_rms_norm(kr_buf, x, bias, w, y_kr, eps)
        t_kr = try_bench(run_kr, "kraken", rank)

        # Lamport
        y_lp = torch.empty_like(x)
        def run_lp():
            lamport_ar_bias_rmsnorm(ws, x, bias, w, y_lp, eps)
        t_lp = try_bench(run_lp, "lamport", rank)

        # FlashInfer
        x_fi = x.clone()
        norm_out = torch.empty_like(x_fi); res_out = torch.empty_like(x_fi)
        def run_fi():
            x_fi.copy_(x)
            allreduce_fusion(input=x_fi, workspace=fi_ws,
                pattern=AllReduceFusionPattern.kARResidualRMSNorm,
                residual_out=res_out, norm_out=norm_out,
                residual_in=bias, rms_gamma=w, rms_eps=eps, use_oneshot=True)
        t_fi = try_bench(run_fi, "flashinfer", rank)

        if rank == 0:
            print(f"{b:>4} | {t_nccl:>8.2f} {t_kr:>8.2f} {t_lp:>8.2f} {t_fi:>10.2f}")
        gc.enable()

    fi_ws.destroy()
    os._exit(0)

if __name__ == "__main__":
    main()
