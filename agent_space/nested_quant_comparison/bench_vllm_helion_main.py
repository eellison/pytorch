from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

import torch._dynamo as dynamo
import torch._inductor.config as inductor_config
from torch._inductor import metrics
from vllm.kernels.helion.ops.rms_norm_dynamic_per_token_quant import (
    rms_norm_dynamic_per_token_quant,
)


FP8_MAX = 448.0
MIN_SCALE = 1.0 / (FP8_MAX * 512.0)


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


def dynamic_rmsnorm_fp8(
    x: torch.Tensor, weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden = x.shape[-1]
    xf = x.float()
    rms = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    normed = (xf * rms).to(x.dtype) * weight
    scale = (normed.abs().amax(dim=-1, keepdim=True).float() / FP8_MAX).clamp_min(MIN_SCALE)
    quant = (normed / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return quant.view(-1, hidden), scale


def fused_add_dynamic_rmsnorm_fp8(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    summed_f = x.float() + residual.float()
    summed = summed_f.to(x.dtype)
    rms = torch.rsqrt(summed_f.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    normed = (summed_f * rms).to(x.dtype) * weight
    scale = (normed.abs().amax(dim=-1, keepdim=True).float() / FP8_MAX).clamp_min(MIN_SCALE)
    quant = (normed / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return quant, scale, summed


def diff(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float | bool]:
    delta = (actual.float() - expected.float()).abs()
    return {
        "exact": bool(torch.equal(actual, expected)),
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--calls-per-graph", type=int, default=100)
    parser.add_argument("--coordinate-descent", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    started = time.time()
    results = []
    for rows, hidden in ((1, 4096), (128, 4096), (1024, 4096), (4096, 4096), (1024, 5120)):
        weight = torch.randn(hidden, device="cuda", dtype=torch.bfloat16)
        for fused_add in (False, True):
            if fused_add:
                x = torch.zeros(rows, hidden, device="cuda", dtype=torch.bfloat16)
                residual = torch.randn_like(x)
                pt_fn = fused_add_dynamic_rmsnorm_fp8
                compile_args = (x, residual, weight)
            else:
                x = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
                residual = None
                pt_fn = dynamic_rmsnorm_fp8
                compile_args = (x, weight)

            dynamo.reset()
            metrics.reset()
            with inductor_config.patch(
                {
                    "triton.nested_reduction": True,
                    "triton.cudagraphs": False,
                    "fx_graph_cache": False,
                    "emulate_precision_casts": True,
                    "coordinate_descent_tuning": args.coordinate_descent,
                }
            ):
                compiled = torch.compile(pt_fn, fullgraph=True, dynamic=False)
                pt_out = compiled(*compile_args)
                torch.cuda.synchronize()
            kernel_count = metrics.generated_kernel_count
            nested_count = metrics.codegen_nested_reduction

            vllm_quant = torch.empty_like(x, dtype=torch.float8_e4m3fn)
            vllm_scale = torch.empty(rows, 1, device="cuda", dtype=torch.float32)
            vllm_residual = residual.clone() if residual is not None else None

            def vllm_fn() -> None:
                rms_norm_dynamic_per_token_quant(
                    vllm_quant,
                    x,
                    weight,
                    vllm_scale,
                    1e-6,
                    residual=vllm_residual,
                )

            vllm_fn()
            torch.cuda.synchronize()
            pt_dequant = pt_out[0].float() * pt_out[1]
            vllm_dequant = vllm_quant.float() * vllm_scale
            if residual is None:
                ref_input = x.float()
            else:
                ref_input = x.float() + residual.float()
            ref_normed = (
                ref_input
                * torch.rsqrt(ref_input.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
            ).to(x.dtype) * weight
            correctness = {
                "quant": diff(pt_out[0], vllm_quant),
                "scale": diff(pt_out[1], vllm_scale),
                "dequant": diff(pt_dequant, vllm_dequant),
                "inductor_dequant_vs_reference": diff(pt_dequant, ref_normed),
                "vllm_dequant_vs_reference": diff(vllm_dequant, ref_normed),
            }
            if fused_add:
                correctness["residual"] = diff(pt_out[2], vllm_residual)
            suite = "fused_add_dynamic_per_token_fp8" if fused_add else "dynamic_per_token_fp8"
            common = {
                "suite": suite,
                "shape": [rows, hidden],
                "dtype": str(x.dtype),
                "layout": "row_major_scale",
                "warmup": args.warmup,
                "samples": args.samples,
                "calls_per_graph": args.calls_per_graph,
                "correctness": correctness,
            }
            pt_impl = "inductor_main_coordesc" if args.coordinate_descent else "inductor_main"
            pt_row = dict(common, implementation=pt_impl)
            pt_row.update({"kernel_count": kernel_count, "nested_reduction_count": nested_count})
            pt_row.update(graph_bench(lambda: compiled(*compile_args), warmup=args.warmup, samples=args.samples, calls_per_graph=args.calls_per_graph))
            results.append(pt_row)
            vllm_row = dict(common, implementation="vllm_main_helion_1.4.0")
            vllm_row.update(graph_bench(vllm_fn, warmup=args.warmup, samples=args.samples, calls_per_graph=args.calls_per_graph))
            results.append(vllm_row)

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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
