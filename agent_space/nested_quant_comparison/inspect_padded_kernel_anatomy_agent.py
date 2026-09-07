from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from bench_main_cuda import (
    FP4_MAX,
    FP8_MAX,
    PACK_E2M1X2_ASM,
    graph_bench,
    recip_ue8m0,
)
from bench_padded_quant_main import rmsnorm_fp4_padded, rmsnorm_mxfp8_padded
from torch._dynamo.utils import counters
from torch._higher_order_ops.flex_gemm import to_blocked
from torch._higher_order_ops.inline_asm_elementwise import inline_asm_elementwise
from torch._inductor import config, inductor_prims, metrics


def rmsnorm_fp4_custom(
    x: torch.Tensor, weight: torch.Tensor, block: int, scale_format: str
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, hidden = x.shape
    normed = F.rms_norm(x, (hidden,), weight).view(rows, hidden // block, block)
    amax = normed.abs().amax(dim=-1)
    if scale_format == "e4m3":
        scale = (amax / FP4_MAX).clamp(min=1e-12, max=FP8_MAX).to(
            torch.float8_e4m3fn
        )
        inv_scale = scale.float().reciprocal()
    else:
        scale = inductor_prims.cvt_e8m0_rceil(
            (amax / FP4_MAX).clamp_min(1e-12)
        )
        inv_scale = recip_ue8m0(scale)
    pairs = normed.view(rows, hidden // block, block // 2, 2)
    packed = inline_asm_elementwise(
        pairs[..., 0].float() * inv_scale.unsqueeze(-1),
        pairs[..., 1].float() * inv_scale.unsqueeze(-1),
        asm_str=PACK_E2M1X2_ASM,
        constraints="=r,f,f",
        dtype=torch.int32,
        is_pure=True,
        pack=1,
    ).to(torch.uint8).view(rows, hidden // 2)
    return packed, to_blocked(scale)


def rmsnorm_mxfp8_custom(
    x: torch.Tensor, weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, hidden = x.shape
    normed = F.rms_norm(x, (hidden,), weight)
    groups = normed.view(rows, hidden // 32, 32)
    amax = groups.abs().float().amax(dim=-1)
    raw_scale = (amax / FP8_MAX).clamp_min(torch.finfo(torch.float32).tiny)
    scale = inductor_prims.cvt_e8m0_rceil(raw_scale)
    scale_f32 = torch.ldexp(torch.ones_like(raw_scale), scale.to(torch.int32) - 127)
    quant = (
        (groups.float() / scale_f32.unsqueeze(-1))
        .clamp(-FP8_MAX, FP8_MAX)
        .to(torch.float8_e4m3fn)
        .view(rows, hidden)
    )
    return quant, to_blocked(scale)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("format", choices=("nvfp4", "mxfp4", "mxfp8"))
    parser.add_argument("rows", type=int)
    parser.add_argument("hidden", type=int)
    parser.add_argument("--transform", choices=("explicit", "custom"), default="explicit")
    parser.add_argument("--force-pointwise-cat", action="store_true")
    parser.add_argument("--coordinate-descent", action="store_true")
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    x = torch.randn(args.rows, args.hidden, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(args.hidden, device="cuda", dtype=torch.bfloat16)

    if args.format == "nvfp4":
        fp4_fn = rmsnorm_fp4_padded if args.transform == "explicit" else rmsnorm_fp4_custom
        fn = lambda x, weight: fp4_fn(x, weight, 16, "e4m3")
    elif args.format == "mxfp4":
        fp4_fn = rmsnorm_fp4_padded if args.transform == "explicit" else rmsnorm_fp4_custom
        fn = lambda x, weight: fp4_fn(x, weight, 32, "ue8m0")
    else:
        fn = rmsnorm_mxfp8_padded if args.transform == "explicit" else rmsnorm_mxfp8_custom

    metrics.reset()
    counters.clear()
    with config.patch(
        {
            "triton.nested_reduction": True,
            "triton.cudagraphs": False,
            "fx_graph_cache": False,
            "emulate_precision_casts": True,
            "coordinate_descent_tuning": args.coordinate_descent,
            "force_pointwise_cat": args.force_pointwise_cat,
        }
    ):
        compiled = torch.compile(fn, fullgraph=True, dynamic=False)
        output = compiled(x, weight)
        torch.cuda.synchronize()
        bench = None
        if args.bench:
            bench = graph_bench(
                lambda: compiled(x, weight), warmup=20, samples=50, calls_per_graph=100
            )
        profile_kernels = None
        if args.profile:
            from torch.profiler import ProfilerActivity, profile

            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                compiled(x, weight)
                torch.cuda.synchronize()
            profile_kernels = [
                event.name
                for event in prof.events()
                if str(event.device_type).endswith("CUDA")
            ]

    print(f"torch={torch.__version__}")
    print(f"git={torch.version.git_version}")
    print(f"torch_file={torch.__file__}")
    print(
        f"format={args.format} shape={args.rows}x{args.hidden} "
        f"transform={args.transform} force_pointwise_cat={args.force_pointwise_cat} "
        f"coordinate_descent={args.coordinate_descent}"
    )
    print(f"kernel_count={metrics.generated_kernel_count}")
    print(f"nested_reduction_count={metrics.codegen_nested_reduction}")
    print(f"outputs={[tuple(value.shape) for value in output]}")
    print(f"inductor_counters={dict(counters['inductor'])}")
    if bench is not None:
        print(f"bench={bench}")
    if profile_kernels is not None:
        print(f"profile_kernels={profile_kernels}")


if __name__ == "__main__":
    main()
