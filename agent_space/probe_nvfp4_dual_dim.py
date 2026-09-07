"""Probe spills, kernel-only time, and a reciprocal-multiply variant."""

import sys

import torch
import triton
from triton.testing import do_bench

sys.path.insert(0, "/data/users/eellison/pytorch/agent_space")
from nvfp4_dual_dim_kernel import nvfp4_dual_dim, nvfp4_dual_dim_kernel  # noqa: E402


def kernel_self_time(fn, iters=30):
    from torch.profiler import profile, ProfilerActivity

    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    total = 0.0
    for evt in prof.key_averages():
        if evt.self_device_time_total > 0 and "memcpy" not in evt.key.lower():
            total += evt.self_device_time_total / iters
    return total


M, K = 16384, 7168
x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
ideal = M * K * 2 + 2 * (M * K // 2 + M * K // 16)

for transposed in (False, True):
    for bm, bk, w in [(16, 256, 2), (16, 512, 4), (32, 128, 2), (16, 512, 2), (16, 1024, 8), (16, 1024, 4)]:
        nvfp4_dual_dim(x, bm, bk, w, dim1_transposed=transposed)
        dev = torch.cuda.current_device()
        kern = None
        for key, k in nvfp4_dual_dim_kernel.device_caches[dev][0].items():
            kern = k
        wall = do_bench(lambda: nvfp4_dual_dim(x, bm, bk, w, dim1_transposed=transposed), return_mode="median") * 1e3
        kus = kernel_self_time(lambda: nvfp4_dual_dim(x, bm, bk, w, dim1_transposed=transposed))
        tbs = ideal / (kus * 1e-6) / 1e12
        spill = getattr(kern, "n_spills", "?") if kern is not None else "?"
        regs = getattr(kern, "n_regs", "?") if kern is not None else "?"
        print(
            f"t={int(transposed)} {bm}x{bk}w{w}: wall={wall:.1f}us kernel={kus:.1f}us "
            f"({tbs:.2f} TB/s) regs={regs} spills={spill}",
            flush=True,
        )
        nvfp4_dual_dim_kernel.device_caches[dev][0].clear()
