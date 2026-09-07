from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import triton

import torch._dynamo as dynamo
import torch._inductor.config as inductor_config
from torch._higher_order_ops.inline_asm_elementwise import inline_asm_elementwise
from torch._inductor import inductor_prims, metrics


FP8_MAX = 448.0
FP4_MAX = 6.0
PACK_E2M1X2_ASM = (
    "{.reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, $2, $1; cvt.u32.u8 $0, t;}"
)
RECIP_UE8M0_ASM = (
    "{.reg .pred p_zero; .reg .s32 neg_exp; .reg .f32 neg_exp_f, result; "
    "setp.eq.u32 p_zero, $1, 0; sub.s32 neg_exp, 127, $1; "
    "cvt.rn.f32.s32 neg_exp_f, neg_exp; ex2.approx.f32 result, neg_exp_f; "
    "selp.f32 $0, 0f00000000, result, p_zero;}"
)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def graph_bench(
    fn: Callable[[], Any], *, warmup: int, samples: int, calls_per_graph: int
) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls_per_graph):
            fn()
    graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    values = []
    for _ in range(samples):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) * 1000.0 / calls_per_graph)
    return {
        "median_us": statistics.median(values),
        "p20_us": percentile(values, 0.2),
        "p80_us": percentile(values, 0.8),
        "min_us": min(values),
        "max_us": max(values),
    }


def compile_main(
    fn: Callable[..., Any], *args: torch.Tensor, coordinate_descent: bool = False
) -> tuple[Callable[..., Any], Any, dict[str, int]]:
    dynamo.reset()
    metrics.reset()
    with inductor_config.patch(
        {
            "triton.nested_reduction": True,
            "triton.cudagraphs": False,
            "fx_graph_cache": False,
            "emulate_precision_casts": True,
            "coordinate_descent_tuning": coordinate_descent,
        }
    ):
        compiled = torch.compile(fn, fullgraph=True, dynamic=False)
        output = compiled(*args)
        torch.cuda.synchronize()
    return compiled, output, {
        "kernel_count": metrics.generated_kernel_count,
        "nested_reduction_count": metrics.codegen_nested_reduction,
    }


def swizzle_scale(scale: torch.Tensor) -> torch.Tensor:
    rows, cols = scale.shape
    blocks = scale.view(rows // 128, 128, cols // 4, 4).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(rows, cols)


def unswizzle_scale(scale: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    blocks = scale.reshape(-1, 32, 4, 4).transpose(1, 2)
    return blocks.reshape(rows // 128, cols // 4, 128, 4).permute(0, 2, 1, 3).reshape(rows, cols)


def swizzled_scale_size(rows: int, hidden: int, block: int) -> int:
    return ((rows + 127) // 128) * ((hidden // block + 3) // 4) * 512


def tensor_diff(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float | bool]:
    actual_f = actual.float()
    expected_f = expected.float()
    diff = (actual_f - expected_f).abs()
    denom = expected_f.abs().clamp_min(1e-12)
    return {
        "exact": bool(torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))),
        "max_abs": float(diff.max()),
        "max_rel": float((diff / denom).max()),
        "mean_abs": float(diff.mean()),
    }


def byte_diff(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float | int | bool]:
    actual_bytes = actual.view(torch.uint8).reshape(-1)
    expected_bytes = expected.view(torch.uint8).reshape(-1)
    mismatches = int((actual_bytes != expected_bytes).sum())
    return {
        "exact": mismatches == 0,
        "mismatched_bytes": mismatches,
        "mismatch_fraction": mismatches / actual_bytes.numel(),
    }


def base_row(
    suite: str,
    implementation: str,
    rows: int,
    hidden: int,
    dtype: torch.dtype,
    layout: str,
    config: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "suite": suite,
        "implementation": implementation,
        "shape": [rows, hidden],
        "dtype": str(dtype),
        "layout": layout,
        "warmup": config.warmup,
        "samples": config.samples,
        "calls_per_graph": config.calls_per_graph,
    }


def manual_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    xf = x.float()
    return xf * torch.rsqrt((xf * xf).mean(dim=-1, keepdim=True) + eps) * weight.float()


def static_rmsnorm_fp8(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (manual_rmsnorm(x, weight) / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)


def fused_add_static_fp8(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    summed_f = x.float() + residual.float()
    summed = summed_f.to(x.dtype)
    normed = summed_f * torch.rsqrt((summed_f * summed_f).mean(dim=-1, keepdim=True) + 1e-6)
    quant = (normed * weight.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return quant, summed


def bench_static(config: argparse.Namespace) -> list[dict[str, Any]]:
    from flashinfer import norm

    results = []
    shapes = [(1, 4096), (19, 4096), (99, 4096), (128, 4096), (989, 4096), (989, 8192), (989, 16384)]
    dtypes = [torch.bfloat16, torch.float16]
    for dtype in dtypes:
        for rows, hidden in shapes:
            if dtype == torch.float16 and (rows, hidden) not in ((99, 4096), (989, 4096), (989, 8192)):
                continue
            x = torch.randn(rows, hidden, device="cuda", dtype=dtype)
            weight = torch.randn(hidden, device="cuda", dtype=dtype)
            scale = torch.ones(1, device="cuda", dtype=torch.float32)
            compiled, pt_out, meta = compile_main(static_rmsnorm_fp8, x, weight, scale, coordinate_descent=config.coordinate_descent)
            fi_out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
            norm.rmsnorm_quant(fi_out, x, weight, scale, 1e-6, enable_pdl=False)
            torch.cuda.synchronize()
            cross = tensor_diff(pt_out, fi_out)
            pt_impl = "inductor_main_coordesc" if config.coordinate_descent else "inductor_main"
            pt_row = base_row("static_rmsnorm_fp8", pt_impl, rows, hidden, dtype, "scalar", config)
            pt_row.update(meta)
            pt_row["correctness_vs_flashinfer"] = cross
            pt_row.update(graph_bench(lambda: compiled(x, weight, scale), warmup=config.warmup, samples=config.samples, calls_per_graph=config.calls_per_graph))
            results.append(pt_row)
            for pdl in ((False, True) if config.include_pdl else (False,)):
                def fi_fn(pdl: bool = pdl) -> None:
                    norm.rmsnorm_quant(fi_out, x, weight, scale, 1e-6, enable_pdl=pdl)

                row = base_row("static_rmsnorm_fp8", f"flashinfer_main_pdl_{str(pdl).lower()}", rows, hidden, dtype, "scalar", config)
                row["correctness_vs_inductor"] = cross
                row.update(graph_bench(fi_fn, warmup=config.warmup, samples=config.samples, calls_per_graph=config.calls_per_graph))
                results.append(row)

            zero = torch.zeros_like(x)
            residual = torch.randn_like(x)
            compiled_add, pt_add, add_meta = compile_main(fused_add_static_fp8, zero, residual, weight, scale, coordinate_descent=config.coordinate_descent)
            fi_residual = residual.clone()
            fi_quant = torch.empty_like(x, dtype=torch.float8_e4m3fn)
            norm.fused_add_rmsnorm_quant(fi_quant, zero, fi_residual, weight, scale, 1e-6, enable_pdl=False)
            torch.cuda.synchronize()
            add_cross = {
                "quant": tensor_diff(pt_add[0], fi_quant),
                "residual": tensor_diff(pt_add[1], fi_residual),
            }
            pt_row = base_row("fused_add_static_fp8", pt_impl, rows, hidden, dtype, "scalar", config)
            pt_row.update(add_meta)
            pt_row["correctness_vs_flashinfer"] = add_cross
            pt_row.update(graph_bench(lambda: compiled_add(zero, residual, weight, scale), warmup=config.warmup, samples=config.samples, calls_per_graph=config.calls_per_graph))
            results.append(pt_row)
            for pdl in ((False, True) if config.include_pdl else (False,)):
                fi_residual.copy_(residual)

                def fi_add_fn(pdl: bool = pdl) -> None:
                    norm.fused_add_rmsnorm_quant(fi_quant, zero, fi_residual, weight, scale, 1e-6, enable_pdl=pdl)

                row = base_row("fused_add_static_fp8", f"flashinfer_main_pdl_{str(pdl).lower()}", rows, hidden, dtype, "scalar", config)
                row["correctness_vs_inductor"] = add_cross
                row.update(graph_bench(fi_add_fn, warmup=config.warmup, samples=config.samples, calls_per_graph=config.calls_per_graph))
                results.append(row)
    return results


def fused_add_block_fp8(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rows, hidden = x.shape
    summed_f = x.float() + residual.float()
    summed = summed_f.to(x.dtype)
    normed = (summed_f * torch.rsqrt((summed_f * summed_f).mean(dim=-1, keepdim=True) + 1e-6) * weight.float()).to(x.dtype)
    groups = normed.float().view(rows, hidden // 128, 128)
    scale = groups.abs().amax(dim=-1).clamp_min(1e-4) / FP8_MAX
    quant = (groups / scale.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).view(rows, hidden)
    padded_rows = (rows + 3) // 4 * 4
    scale = F.pad(scale, (0, 0, 0, padded_rows - rows)).t().contiguous()
    return quant, scale, normed, summed


def bench_block_fp8(config: argparse.Namespace) -> list[dict[str, Any]]:
    from flashinfer import norm

    results = []
    shapes = [(1, 4096), (19, 4096), (99, 4096), (128, 4096), (989, 4096), (989, 8192), (989, 16384)]
    for dtype in (torch.bfloat16, torch.float16):
        for rows, hidden in shapes:
            if dtype == torch.float16 and (rows, hidden) not in ((99, 4096), (989, 4096), (989, 8192)):
                continue
            zero = torch.zeros(rows, hidden, device="cuda", dtype=dtype)
            residual = torch.randn_like(zero) * 0.1
            weight = torch.randn(hidden, device="cuda", dtype=dtype)
            compiled, pt_out, meta = compile_main(fused_add_block_fp8, zero, residual, weight, coordinate_descent=config.coordinate_descent)
            fi_quant = torch.empty_like(zero, dtype=torch.float8_e4m3fn)
            fi_scale = torch.empty(hidden // 128, (rows + 3) // 4 * 4, device="cuda", dtype=torch.float32)
            fi_normed = torch.empty_like(zero)
            fi_residual = residual.clone()
            norm.fused_add_rmsnorm_fp8_block_quant(fi_quant, fi_scale, fi_normed, zero, fi_residual, weight, 1e-6, enable_pdl=False)
            torch.cuda.synchronize()
            pt_scale_logical = pt_out[1].t()[:rows]
            fi_scale_logical = fi_scale.t()[:rows]
            pt_dequant = pt_out[0].float().view(rows, hidden // 128, 128) * pt_scale_logical.unsqueeze(-1)
            fi_dequant = fi_quant.float().view(rows, hidden // 128, 128) * fi_scale_logical.unsqueeze(-1)
            correctness = {
                "quant": tensor_diff(pt_out[0], fi_quant),
                "scale": tensor_diff(pt_scale_logical, fi_scale_logical),
                "dequant": tensor_diff(pt_dequant, fi_dequant),
                "normed": tensor_diff(pt_out[2], fi_normed),
                "residual": tensor_diff(pt_out[3], fi_residual),
            }
            pt_impl = "inductor_main_coordesc" if config.coordinate_descent else "inductor_main"
            row = base_row("fused_add_block_fp8_g128", pt_impl, rows, hidden, dtype, "column_major_padded", config)
            row.update(meta)
            row["correctness_vs_flashinfer"] = correctness
            row.update(graph_bench(lambda: compiled(zero, residual, weight), warmup=config.warmup, samples=config.samples, calls_per_graph=config.calls_per_graph))
            results.append(row)
            for pdl in ((False, True) if config.include_pdl else (False,)):
                fi_residual.copy_(residual)

                def fi_fn(pdl: bool = pdl) -> None:
                    norm.fused_add_rmsnorm_fp8_block_quant(fi_quant, fi_scale, fi_normed, zero, fi_residual, weight, 1e-6, enable_pdl=pdl)

                row = base_row("fused_add_block_fp8_g128", f"flashinfer_main_pdl_{str(pdl).lower()}", rows, hidden, dtype, "column_major_padded", config)
                row["correctness_vs_inductor"] = correctness
                row.update(graph_bench(fi_fn, warmup=config.warmup, samples=config.samples, calls_per_graph=config.calls_per_graph))
                results.append(row)
    return results


def recip_ue8m0(scale: torch.Tensor) -> torch.Tensor:
    return inline_asm_elementwise(scale.to(torch.int32), asm_str=RECIP_UE8M0_ASM, constraints="=f,r", dtype=torch.float32, is_pure=True, pack=1)


def dequant_fp4(
    packed: torch.Tensor,
    scale: torch.Tensor,
    *,
    rows: int,
    hidden: int,
    block: int,
    scale_format: str,
    swizzled: bool,
) -> torch.Tensor:
    packed_bytes = packed.view(torch.uint8).reshape(rows, hidden // 2)
    codes = torch.stack((packed_bytes & 0x0F, packed_bytes >> 4), dim=-1).reshape(rows, hidden).long()
    positive = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=packed.device)
    values = positive[codes & 7] * torch.where(codes < 8, 1.0, -1.0)
    scale = scale.reshape(-1)
    if swizzled:
        scale = unswizzle_scale(scale, rows, hidden // block).reshape(-1)
    scale = scale.reshape(rows, hidden // block)
    if scale_format == "e4m3":
        scale_f32 = scale.float()
    else:
        scale_f32 = torch.ldexp(torch.ones_like(scale, dtype=torch.float32), scale.to(torch.int32) - 127)
    return (values.view(rows, hidden // block, block) * scale_f32.unsqueeze(-1)).view(rows, hidden)


def rmsnorm_fp4(
    x: torch.Tensor, weight: torch.Tensor, block: int, scale_format: str, swizzled: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, hidden = x.shape
    normed = F.rms_norm(x, (hidden,), weight).view(rows, hidden // block, block)
    amax = normed.abs().amax(dim=-1)
    if scale_format == "e4m3":
        scale = (amax / FP4_MAX).clamp(min=1e-12, max=FP8_MAX).to(torch.float8_e4m3fn)
        inv_scale = 1.0 / scale.float()
    else:
        scale = inductor_prims.cvt_e8m0_rceil((amax / FP4_MAX).clamp_min(1e-12))
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
    if swizzled:
        scale = swizzle_scale(scale)
    return packed, scale


def bench_fp4(config: argparse.Namespace) -> list[dict[str, Any]]:
    from flashinfer.cute_dsl import rmsnorm_fp4quant

    results = []
    shapes = [(128, 4096), (1024, 4096), (256, 8192), (128, 16384)]
    for dtype in (torch.bfloat16, torch.float16):
        for rows, hidden in shapes:
            if dtype == torch.float16 and (rows, hidden) not in ((128, 4096), (1024, 4096)):
                continue
            x = torch.randn(rows, hidden, device="cuda", dtype=dtype)
            weight = torch.randn(hidden, device="cuda", dtype=dtype)
            global_scale = torch.ones(1, device="cuda", dtype=torch.float32)
            for name, block, scale_format, scale_dtype in (
                ("nvfp4", 16, "e4m3", torch.float8_e4m3fn),
                ("mxfp4", 32, "ue8m0", torch.uint8),
            ):
                for swizzled in (False, True):
                    def pt_fn(x: torch.Tensor, weight: torch.Tensor, block: int = block, scale_format: str = scale_format, swizzled: bool = swizzled):
                        return rmsnorm_fp4(x, weight, block, scale_format, swizzled)

                    compiled, pt_out, meta = compile_main(pt_fn, x, weight, coordinate_descent=config.coordinate_descent)
                    fi_quant = torch.empty(rows, hidden // 2, device="cuda", dtype=torch.float4_e2m1fn_x2)
                    scale_shape = swizzled_scale_size(rows, hidden, block) if swizzled else (rows, hidden // block)
                    fi_scale = torch.empty(scale_shape, device="cuda", dtype=scale_dtype)

                    def fi_fn() -> Any:
                        return rmsnorm_fp4quant(
                            x,
                            weight,
                            y_fp4=fi_quant,
                            block_scale=fi_scale,
                            global_scale=global_scale,
                            eps=1e-6,
                            block_size=block,
                            scale_format=scale_format,
                            is_sf_swizzled_layout=swizzled,
                            enable_pdl=False,
                        )

                    fi_fn()
                    torch.cuda.synchronize()
                    reference = F.rms_norm(x, (hidden,), weight).float()
                    pt_dequant = dequant_fp4(pt_out[0], pt_out[1], rows=rows, hidden=hidden, block=block, scale_format=scale_format, swizzled=swizzled)
                    fi_dequant = dequant_fp4(fi_quant, fi_scale, rows=rows, hidden=hidden, block=block, scale_format=scale_format, swizzled=swizzled)
                    correctness = {
                        "quant": byte_diff(pt_out[0], fi_quant),
                        "scale": tensor_diff(pt_out[1].reshape(-1), fi_scale.reshape(-1)),
                        "inductor_dequant_vs_reference": tensor_diff(pt_dequant, reference),
                        "flashinfer_dequant_vs_reference": tensor_diff(fi_dequant, reference),
                    }
                    layout = "swizzled_128x4" if swizzled else "row_major"
                    pt_impl = "inductor_main_coordesc" if config.coordinate_descent else "inductor_main"
                    row = base_row(f"rmsnorm_{name}", pt_impl, rows, hidden, dtype, layout, config)
                    row.update(meta)
                    row["correctness_vs_flashinfer"] = correctness
                    row.update(graph_bench(lambda: compiled(x, weight), warmup=config.warmup, samples=config.samples, calls_per_graph=config.calls_per_graph))
                    results.append(row)
                    row = base_row(f"rmsnorm_{name}", "flashinfer_main", rows, hidden, dtype, layout, config)
                    row["correctness_vs_inductor"] = correctness
                    row.update(graph_bench(fi_fn, warmup=config.warmup, samples=config.samples, calls_per_graph=config.calls_per_graph))
                    results.append(row)
    return results


def rmsnorm_mxfp8(x: torch.Tensor, weight: torch.Tensor, swizzled: bool) -> tuple[torch.Tensor, torch.Tensor]:
    rows, hidden = x.shape
    normed = F.rms_norm(x, (hidden,), weight)
    groups = normed.view(rows, hidden // 32, 32)
    amax = groups.abs().float().amax(dim=-1)
    raw_scale = (amax / FP8_MAX).clamp_min(torch.finfo(torch.float32).tiny)
    scale = inductor_prims.cvt_e8m0_rceil(raw_scale)
    scale_f32 = torch.ldexp(torch.ones_like(raw_scale), scale.to(torch.int32) - 127)
    quant = (groups.float() / scale_f32.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).view(rows, hidden)
    if swizzled:
        scale = swizzle_scale(scale)
    return quant, scale


def bench_mxfp8(config: argparse.Namespace) -> list[dict[str, Any]]:
    from flashinfer import norm
    from flashinfer.quantization import mxfp8_quantize
    from flashinfer.tllm_enums import SfLayout

    results = []
    for rows, hidden in ((128, 4096), (1024, 4096), (256, 8192), (128, 16384)):
        x = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(hidden, device="cuda", dtype=torch.bfloat16)
        fi_normed = torch.empty_like(x)
        for swizzled in (False, True):
            def pt_fn(x: torch.Tensor, weight: torch.Tensor, swizzled: bool = swizzled):
                return rmsnorm_mxfp8(x, weight, swizzled)

            compiled, pt_out, meta = compile_main(pt_fn, x, weight, coordinate_descent=config.coordinate_descent)
            sf_layout = SfLayout.layout_128x4 if swizzled else SfLayout.layout_linear

            def fi_fn() -> tuple[torch.Tensor, torch.Tensor]:
                norm.rmsnorm(x, weight, 1e-6, out=fi_normed, enable_pdl=False)
                return mxfp8_quantize(fi_normed, sf_swizzle_layout=sf_layout, enable_pdl=False)

            fi_out = fi_fn()
            torch.cuda.synchronize()
            correctness = {
                "quant": tensor_diff(pt_out[0], fi_out[0]),
                "scale": tensor_diff(pt_out[1].reshape(-1), fi_out[1].reshape(-1)),
            }
            layout = "swizzled_128x4" if swizzled else "row_major"
            pt_impl = "inductor_main_fused_coordesc" if config.coordinate_descent else "inductor_main_fused"
            row = base_row("rmsnorm_mxfp8", pt_impl, rows, hidden, torch.bfloat16, layout, config)
            row.update(meta)
            row["correctness_vs_flashinfer_composed"] = correctness
            row.update(graph_bench(lambda: compiled(x, weight), warmup=config.warmup, samples=config.samples, calls_per_graph=config.calls_per_graph))
            results.append(row)
            row = base_row("rmsnorm_mxfp8", "flashinfer_main_composed", rows, hidden, torch.bfloat16, layout, config)
            row["correctness_vs_inductor"] = correctness
            row.update(graph_bench(fi_fn, warmup=config.warmup, samples=config.samples, calls_per_graph=config.calls_per_graph))
            results.append(row)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suites", nargs="+", choices=("static", "block_fp8", "fp4", "mxfp8"), default=("static", "block_fp8", "fp4", "mxfp8"))
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--calls-per-graph", type=int, default=100)
    parser.add_argument("--include-pdl", action="store_true")
    parser.add_argument("--coordinate-descent", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    config = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    started = time.time()
    results = []
    runners = {
        "static": bench_static,
        "block_fp8": bench_block_fp8,
        "fp4": bench_fp4,
        "mxfp8": bench_mxfp8,
    }
    for suite in config.suites:
        print(f"running {suite}", flush=True)
        try:
            results.extend(runners[suite](config))
        except Exception as error:
            results.append({"suite": suite, "error": f"{type(error).__name__}: {error}"})
            print(f"{suite} failed: {type(error).__name__}: {error}", flush=True)
    payload = {
        "environment": {
            "torch_version": torch.__version__,
            "torch_git_version": torch.version.git_version,
            "torch_file": torch.__file__,
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "capability": list(torch.cuda.get_device_capability()),
            "elapsed_seconds": time.time() - started,
        },
        "results": results,
    }
    config.output.parent.mkdir(parents=True, exist_ok=True)
    config.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
