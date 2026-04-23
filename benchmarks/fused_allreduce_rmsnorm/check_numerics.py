import os, torch, torch.distributed as dist, torch.distributed._symmetric_memory as symm_mem
from push_poll_allreduce_rmsnorm import (
    lamport_ar_bias_rmsnorm, LamportWorkspace,
    butterfly_ar_bias_rmsnorm,
)
from kraken.fused.one_shot_all_reduce_bias_rms_norm import one_shot_all_reduce_bias_rms_norm

local_rank = int(os.environ['LOCAL_RANK'])
device = torch.device(f'cuda:{local_rank}')
torch.cuda.set_device(device)
dist.init_process_group('nccl', device_id=device)
rank = dist.get_rank()
W = dist.get_world_size()

for D in [4096, 5120, 7168, 8192, 16384]:
    for b in [1, 8, 32, 128]:
        eps = 1e-5
        torch.manual_seed(42 + rank)
        x = torch.randn(b, D, dtype=torch.bfloat16, device=device)
        torch.manual_seed(42)
        w = torch.randn(D, dtype=torch.bfloat16, device=device)
        bias = torch.randn(b, D, dtype=torch.bfloat16, device=device)

        # Reference: Kraken
        kr_buf = symm_mem.empty(b, D, dtype=torch.bfloat16, device=device)
        symm_mem.rendezvous(kr_buf, dist.group.WORLD.group_name)
        y_kr = torch.empty_like(x)
        one_shot_all_reduce_bias_rms_norm(kr_buf, x, bias, w, y_kr, eps)

        # Lamport one-shot
        ws = LamportWorkspace(max_N=b, D=D, dtype=torch.bfloat16, device=device)
        y_lp = torch.empty_like(x)
        lamport_ar_bias_rmsnorm(ws, x, bias, w, y_lp, eps)

        # Butterfly
        bf_buf = symm_mem.empty(2 * b, D, dtype=torch.bfloat16, device=device)
        symm_mem.rendezvous(bf_buf, dist.group.WORLD.group_name)
        y_bf = torch.empty_like(x)
        butterfly_ar_bias_rmsnorm(bf_buf, x, bias, w, y_bf, eps)

        if rank == 0:
            d_lp = (y_kr.float() - y_lp.float()).abs().max().item()
            d_bf = (y_kr.float() - y_bf.float()).abs().max().item()
            ok = d_lp < 0.1 and d_bf < 0.1
            status = "PASS" if ok else "FAIL"
            print(f"  {status}  D={D:>5} b={b:>3} | lamport={d_lp:.4f} butterfly={d_bf:.4f}")

        dist.barrier()

os._exit(0)
