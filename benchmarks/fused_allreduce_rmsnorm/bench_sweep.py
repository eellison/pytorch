"""
Comprehensive benchmark: Lamport (one-shot & two-shot), push_sync, Kraken, FlashInfer, NCCL
across common LLM inference hidden dims, batch sizes, and world sizes.

Usage:
  # world_size=8 (all GPUs)
  torchrun --nproc-per-node 8 bench_sweep.py

  # world_size=4
  torchrun --nproc-per-node 4 bench_sweep.py

  # world_size=2
  torchrun --nproc-per-node 2 bench_sweep.py

  # Run all world sizes automatically:
  python bench_sweep.py --sweep-world-sizes
"""
import gc, os, sys, subprocess

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from push_poll_allreduce_rmsnorm import (
    lamport_ar_bias_rmsnorm,
    lamport_twoshot_ar_bias_rmsnorm,
    push_sync_ar_bias_rmsnorm,
    push_sync_twoshot_ar_bias_rmsnorm,
    butterfly_ar_bias_rmsnorm,
    inplace_sync_ar_bias_rmsnorm,
    LamportWorkspace,
    init_sentinel_buffer,
)
from kraken.fused.one_shot_all_reduce_bias_rms_norm import one_shot_all_reduce_bias_rms_norm
from kraken.fused.rms_norm import rms_norm as kraken_rms_norm
from flashinfer.comm.allreduce import (
    AllReduceFusionPattern,
    allreduce_fusion,
    create_allreduce_fusion_workspace,
)

# Common LLM hidden dimensions:
#   4096  - LLaMA-2 7B, Mistral 7B
#   5120  - LLaMA-2 13B
#   6656  - LLaMA-2 34B (rounded)
#   8192  - LLaMA-2 70B, Falcon 40B
#  12288  - GPT-3 175B
#  16384  - LLaMA-3 405B
HIDDEN_DIMS = [4096, 5120, 7168, 8192, 16384]
BATCH_SIZES = [1, 4, 8, 16, 32, 64, 128, 512, 1024, 2048]
EPS = 1e-5
DTYPE = torch.bfloat16
ITERS = 200


def bench_cudagraph(fn, iters=ITERS, warmup=3):
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
    l2 = torch.empty(40 * 1024 * 1024, dtype=torch.int8, device="cuda")
    start = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        l2.zero_()
        start[i].record()
        g.replay()
        end[i].record()
    torch.cuda.synchronize()
    times = sorted([s.elapsed_time(e) * 1000 for s, e in zip(start, end)])
    return times[len(times) // 2]


def bench_eager(fn, iters=ITERS, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    l2 = torch.empty(40 * 1024 * 1024, dtype=torch.int8, device="cuda")
    start = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        l2.zero_()
        start[i].record()
        fn()
        end[i].record()
    torch.cuda.synchronize()
    times = sorted([s.elapsed_time(e) * 1000 for s, e in zip(start, end)])
    return times[len(times) // 2]


def try_bench(fn, name, rank):
    dist.barrier()
    torch.cuda.synchronize()
    try:
        return bench_cudagraph(fn)
    except Exception as e:
        if rank == 0:
            print(f"  [cudagraph failed for {name}: {e}]")
        return bench_eager(fn)
    finally:
        torch.cuda.synchronize()
        dist.barrier()


def run_benchmark():
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    symm_mem.enable_symm_mem_for_group(dist.group.WORLD.group_name)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    max_b = max(BATCH_SIZES)
    max_D = max(HIDDEN_DIMS)

    if rank == 0:
        gpu_name = torch.cuda.get_device_name(device)
        print(f"\n{'='*90}")
        print(f"  {world_size}x {gpu_name}  |  world_size={world_size}  |  dtype=bf16  |  CUDA graphs")
        print(f"{'='*90}")

    @torch.compile(fullgraph=True)
    def _compiled_bias_rmsnorm(x_in, bias_in, w_in, eps):
        x_in = x_in + bias_in
        variance = x_in.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x_in = x_in * torch.rsqrt(variance + eps)
        return (w_in * x_in).to(torch.bfloat16)

    for D in HIDDEN_DIMS:
        # Pre-allocate workspaces for this D
        lamport_ws = LamportWorkspace(
            max_N=max_b, D=D, dtype=DTYPE, device=device, twoshot=False
        )
        lamport_2s_ws = LamportWorkspace(
            max_N=max_b, D=D, dtype=DTYPE, device=device, twoshot=True
        )
        fi_ws = create_allreduce_fusion_workspace(
            backend="trtllm",
            world_size=world_size,
            rank=rank,
            max_token_num=max_b,
            hidden_dim=D,
            dtype=DTYPE,
            force_oneshot_support=True,
        )
        torch.cuda.synchronize()
        dist.barrier()

        if rank == 0:
            hdr = (
                f"\n  D={D}"
                f"\n  {'b':>4} | {'nccl':>9} {'kraken':>9} {'push_sync':>10} "
                f"{'lamport':>9} {'2shot':>9} {'bfly':>9} {'inplace':>9} {'symm_1s':>9} {'symm_2s':>9} {'fi_1shot':>11} {'fi_2shot':>11}   (us)"
            )
            print(hdr)
            print("  " + "-" * 100)

        for b in BATCH_SIZES:
            gc.disable()
            torch.manual_seed(42)
            w = torch.randn(D, dtype=DTYPE, device=device)
            bias = torch.randn(b, D, dtype=DTYPE, device=device)
            torch.manual_seed(42 + rank)
            x = torch.randn(b, D, dtype=DTYPE, device=device)

            results = {}

            # --- NCCL baseline (torch.compiled bias+rmsnorm) ---
            x_n = x.clone()
            y_n = torch.empty_like(x)

            def run_nccl():
                x_n.copy_(x)
                dist.all_reduce(x_n)
                y_n.copy_(_compiled_bias_rmsnorm(x_n, bias, w, EPS))

            results["nccl"] = try_bench(run_nccl, "nccl", rank)

            # --- symm_mem one-shot allreduce + compiled rmsnorm ---
            sa_buf = symm_mem.empty(b, D, dtype=DTYPE, device=device)
            symm_mem.rendezvous(sa_buf, dist.group.WORLD.group_name)
            sa_buf.copy_(x)
            y_sa = torch.empty_like(x)

            def run_symm_1s():
                sa_out = torch.ops.symm_mem.one_shot_all_reduce(
                    sa_buf, "sum", dist.group.WORLD.group_name)
                y_sa.copy_(_compiled_bias_rmsnorm(sa_out, bias, w, EPS))

            results["symm_1s"] = try_bench(run_symm_1s, "symm_1s", rank)

            # --- symm_mem two-shot allreduce + compiled rmsnorm ---
            sa2_buf = symm_mem.empty(b, D, dtype=DTYPE, device=device)
            symm_mem.rendezvous(sa2_buf, dist.group.WORLD.group_name)
            sa2_buf.copy_(x)
            y_sa2 = torch.empty_like(x)

            def run_symm_2s():
                torch.ops.symm_mem.two_shot_all_reduce_(
                    sa2_buf, "sum", dist.group.WORLD.group_name)
                y_sa2.copy_(_compiled_bias_rmsnorm(sa2_buf, bias, w, EPS))

            results["symm_2s"] = try_bench(run_symm_2s, "symm_2s", rank)

            # --- Kraken ---
            kr_buf = symm_mem.empty(b, D, dtype=DTYPE, device=device)
            symm_mem.rendezvous(kr_buf, dist.group.WORLD.group_name)
            y_kr = torch.empty_like(x)

            def run_kr():
                one_shot_all_reduce_bias_rms_norm(kr_buf, x, bias, w, y_kr, EPS)

            results["kraken"] = try_bench(run_kr, "kraken", rank)

            # --- Inplace (input already in symm_mem, no copy) ---
            ip_buf = symm_mem.empty(b, D, dtype=DTYPE, device=device)
            symm_mem.rendezvous(ip_buf, dist.group.WORLD.group_name)
            ip_buf.copy_(x)
            y_ip = torch.empty_like(x)

            def run_ip():
                inplace_sync_ar_bias_rmsnorm(ip_buf, bias, w, y_ip, EPS)

            results["inplace"] = try_bench(run_ip, "inplace", rank)

            # --- Push+Sync (Variant A) ---
            ps_buf = symm_mem.empty(world_size * b, D, dtype=DTYPE, device=device)
            symm_mem.rendezvous(ps_buf, dist.group.WORLD.group_name)
            y_ps = torch.empty_like(x)

            def run_ps():
                push_sync_ar_bias_rmsnorm(ps_buf, x, bias, w, y_ps, EPS)

            results["push_sync"] = try_bench(run_ps, "push_sync", rank)

            # --- Lamport one-shot ---
            init_sentinel_buffer(lamport_ws.triple_buf)
            lamport_ws.counter.zero_()
            lamport_ws.phase.zero_()
            torch.cuda.synchronize()
            dist.barrier()
            y_lp = torch.empty_like(x)

            def run_lp():
                lamport_ar_bias_rmsnorm(lamport_ws, x, bias, w, y_lp, EPS)

            results["lamport"] = try_bench(run_lp, "lamport", rank)

            # --- Lamport two-shot (needs b >= world_size) ---
            if b >= world_size:
                init_sentinel_buffer(lamport_2s_ws.triple_buf)
                lamport_2s_ws.counter.zero_()
                lamport_2s_ws.phase.zero_()
                torch.cuda.synchronize()
                dist.barrier()
                y_2s = torch.empty_like(x)

                def run_2s():
                    lamport_twoshot_ar_bias_rmsnorm(lamport_2s_ws, x, bias, w, y_2s, EPS)

                results["2shot"] = try_bench(run_2s, "2shot", rank)
            else:
                results["2shot"] = float("nan")

            # --- Butterfly (recursive-doubling) ---
            bf_buf = symm_mem.empty(2 * b, D, dtype=DTYPE, device=device)
            symm_mem.rendezvous(bf_buf, dist.group.WORLD.group_name)
            y_bf = torch.empty_like(x)

            def run_bf():
                butterfly_ar_bias_rmsnorm(bf_buf, x, bias, w, y_bf, EPS)

            results["butterfly"] = try_bench(run_bf, "butterfly", rank)

            # --- FlashInfer ---
            x_fi = x.clone()
            norm_out = torch.empty_like(x_fi)
            res_out = torch.empty_like(x_fi)

            def run_fi():
                x_fi.copy_(x)
                allreduce_fusion(
                    input=x_fi,
                    workspace=fi_ws,
                    pattern=AllReduceFusionPattern.kARResidualRMSNorm,
                    residual_out=res_out,
                    norm_out=norm_out,
                    residual_in=bias,
                    rms_gamma=w,
                    rms_eps=EPS,
                    use_oneshot=True,
                )

            results["fi_1shot"] = try_bench(run_fi, "fi_1shot", rank)

            # --- FlashInfer two-shot (needs b > world_size) ---
            if b > world_size:
                x_fi2 = x.clone()
                norm_out2 = torch.empty_like(x_fi2)
                res_out2 = torch.empty_like(x_fi2)

                def run_fi2():
                    x_fi2.copy_(x)
                    allreduce_fusion(
                        input=x_fi2,
                        workspace=fi_ws,
                        pattern=AllReduceFusionPattern.kARResidualRMSNorm,
                        residual_out=res_out2,
                        norm_out=norm_out2,
                        residual_in=bias,
                        rms_gamma=w,
                        rms_eps=EPS,
                        use_oneshot=False,
                    )

                results["fi_2shot"] = try_bench(run_fi2, "fi_2shot", rank)
            else:
                results["fi_2shot"] = float("nan")

            if rank == 0:
                def fmt(v):
                    if v != v:  # nan
                        return "n/a"
                    return f"{v:.2f}"

                print(
                    f"  {b:>4} | {fmt(results['nccl']):>9} {fmt(results['kraken']):>9} "
                    f"{fmt(results['push_sync']):>10} {fmt(results['lamport']):>9} "
                    f"{fmt(results['2shot']):>9} {fmt(results['butterfly']):>9} "
                    f"{fmt(results['inplace']):>9} {fmt(results['symm_1s']):>9} {fmt(results['symm_2s']):>9} "
                    f"{fmt(results['fi_1shot']):>11} {fmt(results['fi_2shot']):>11}"
                )

            gc.enable()
            dist.barrier()

        fi_ws.destroy()
        dist.barrier()

    os._exit(0)


def sweep_world_sizes():
    script = os.path.abspath(__file__)
    for ws in [2, 4, 8]:
        print(f"\n>>> Launching world_size={ws} <<<")
        subprocess.run(
            ["torchrun", f"--nproc-per-node={ws}", script],
            env={**os.environ},
        )


if __name__ == "__main__":
    if "--sweep-world-sizes" in sys.argv:
        sweep_world_sizes()
    else:
        run_benchmark()
