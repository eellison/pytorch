"""Per-kernel breakdown of the compiled padded-swizzle rmsnorm vs FlashInfer.

The padding strategy is a wash across variants, so this attributes the residual
gap to FlashInfer: how much is the fused rmsnorm+quant+scatter kernel, how much
is the separate pad kernel, and how many kernels FlashInfer itself launches.
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import torch
from torch.profiler import ProfilerActivity, profile

from bench_padded_variants import rmsnorm_fp4_padded
from bench_main_cuda import swizzled_scale_size
from bench_main_cuda import compile_main
from blocked_variants import VARIANTS


def kernel_times(fn, iters=200):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    totals = defaultdict(float)
    counts = defaultdict(int)
    for e in prof.key_averages():
        if e.device_time_total <= 0 or e.device_type.name != "CUDA":
            continue
        totals[e.key] += e.device_time_total / iters
        counts[e.key] += e.count // iters
    return totals, counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="predicated")
    p.add_argument("--shapes", default="129x4096,989x4096")
    p.add_argument("--suites", default="nvfp4,mxfp4")
    args = p.parse_args()

    for spec in args.shapes.split(","):
        rows, hidden = map(int, spec.split("x"))
        for name in args.suites.split(","):
            block, fmt = (16, "e4m3") if name == "nvfp4" else (32, "ue8m0")
            x = torch.randn(rows, hidden, dtype=torch.bfloat16, device="cuda")
            weight = torch.randn(hidden, dtype=torch.bfloat16, device="cuda")
            to_blocked = VARIANTS[args.variant]
            compiled, _out, _meta = compile_main(
                lambda a, w: rmsnorm_fp4_padded(a, w, block, fmt, to_blocked),
                x, weight, coordinate_descent=False,
            )
            torch.cuda.synchronize()
            from flashinfer.cute_dsl import rmsnorm_fp4quant
            sdt = torch.float8_e4m3fn if fmt == "e4m3" else torch.uint8
            fi_quant = torch.empty(rows, hidden // 2, device="cuda", dtype=torch.float4_e2m1fn_x2)
            fi_scale = torch.empty(swizzled_scale_size(rows, hidden, block), device="cuda", dtype=sdt)
            gs = torch.ones(1, device="cuda", dtype=torch.float32)
            fi_fn = lambda: rmsnorm_fp4quant(x, weight, y_fp4=fi_quant, block_scale=fi_scale,
                global_scale=gs, eps=1e-6, block_size=block, scale_format=fmt,
                is_sf_swizzled_layout=True, enable_pdl=False)
            fi_fn()
            torch.cuda.synchronize()

            for label, fn in (("inductor", lambda: compiled(x, weight)), ("flashinfer", fi_fn)):
                totals, counts = kernel_times(fn)
                total = sum(totals.values())
                print(f"\n{name} {rows}x{hidden}  {label} ({args.variant if label=='inductor' else ''})"
                      f"  total={total:.2f}us over {sum(counts.values())} kernels")
                for k in sorted(totals, key=totals.get, reverse=True):
                    print(f"    {totals[k]:7.2f}us  x{counts[k]}  {k[:78]}")
            torch._dynamo.reset()


if __name__ == "__main__":
    main()
