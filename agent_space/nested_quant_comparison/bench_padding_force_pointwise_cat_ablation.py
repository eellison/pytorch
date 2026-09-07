from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

import torch

import torch._dynamo as dynamo
import torch._inductor.config as inductor_config
from bench_main_cuda import (
    base_row,
    byte_diff,
    graph_bench,
    swizzled_scale_size,
)
from bench_padded_quant_main import (
    logical_e8m0_scale_diff,
    logical_float_scale_diff,
    rmsnorm_fp4_padded,
    rmsnorm_mxfp8_padded,
)
from torch._dynamo.utils import counters
from torch._inductor import metrics


def compile_ablation(
    fn: Callable[..., Any],
    *args: torch.Tensor,
    force_pointwise_cat: bool,
) -> tuple[Callable[..., Any], Any, dict[str, int | float | bool]]:
    dynamo.reset()
    metrics.reset()
    counters.clear()
    started = time.time()
    with inductor_config.patch(
        {
            "triton.nested_reduction": True,
            "triton.cudagraphs": False,
            "fx_graph_cache": False,
            "emulate_precision_casts": True,
            "coordinate_descent_tuning": True,
            "force_pointwise_cat": force_pointwise_cat,
        }
    ):
        compiled = torch.compile(fn, fullgraph=True, dynamic=False)
        output = compiled(*args)
        torch.cuda.synchronize()
    return compiled, output, {
        "coordinate_descent": True,
        "force_pointwise_cat": force_pointwise_cat,
        "compile_seconds": time.time() - started,
        "kernel_count": metrics.generated_kernel_count,
        "nested_reduction_count": metrics.codegen_nested_reduction,
        "pad_rewritten_as_cat": int(
            counters["inductor"]["pad_rewritten_as_cat"]
        ),
    }


def compare_scale(
    actual: torch.Tensor,
    expected: torch.Tensor,
    rows: int,
    cols: int,
    scale_format: str,
) -> dict[str, float | int | bool | str]:
    if scale_format == "e4m3":
        return logical_float_scale_diff(actual, expected, rows, cols)
    return logical_e8m0_scale_diff(actual, expected, rows, cols)


def bench_fp4_case(
    config: argparse.Namespace,
    rows: int,
    hidden: int,
    name: str,
    block: int,
    scale_format: str,
    scale_dtype: torch.dtype,
) -> list[dict[str, Any]]:
    from flashinfer.cute_dsl import rmsnorm_fp4quant

    x = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(hidden, device="cuda", dtype=torch.bfloat16)
    global_scale = torch.ones(1, device="cuda", dtype=torch.float32)

    def pt_fn(x: torch.Tensor, weight: torch.Tensor):
        return rmsnorm_fp4_padded(x, weight, block, scale_format)

    compiled_fns = {}
    outputs = {}
    metadata = {}
    for force_pointwise_cat in (False, True):
        key = (
            "inductor_main_coordesc_force_pointwise_cat"
            if force_pointwise_cat
            else "inductor_main_coordesc_default_cat"
        )
        compiled, output, meta = compile_ablation(
            pt_fn, x, weight, force_pointwise_cat=force_pointwise_cat
        )
        compiled_fns[key], outputs[key], metadata[key] = compiled, output, meta

    fi_quant = torch.empty(
        rows, hidden // 2, device="cuda", dtype=torch.float4_e2m1fn_x2
    )
    fi_scale = torch.empty(
        swizzled_scale_size(rows, hidden, block), device="cuda", dtype=scale_dtype
    )

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
            is_sf_swizzled_layout=True,
            enable_pdl=False,
        )

    fi_fn()
    torch.cuda.synchronize()
    scale_cols = hidden // block
    reference = outputs["inductor_main_coordesc_default_cat"]
    results = []
    for key, compiled in compiled_fns.items():
        output = outputs[key]
        row = base_row(
            f"rmsnorm_{name}",
            key,
            rows,
            hidden,
            torch.bfloat16,
            "padded_128x4",
            config,
        )
        row.update(metadata[key])
        row["correctness_vs_default"] = {
            "quant": byte_diff(output[0], reference[0]),
            "scale": compare_scale(
                output[1], reference[1], rows, scale_cols, scale_format
            ),
        }
        row["correctness_vs_flashinfer"] = {
            "quant": byte_diff(output[0], fi_quant),
            "scale": compare_scale(
                output[1], fi_scale, rows, scale_cols, scale_format
            ),
        }
        row.update(
            graph_bench(
                lambda compiled=compiled: compiled(x, weight),
                warmup=config.warmup,
                samples=config.samples,
                calls_per_graph=config.calls_per_graph,
            )
        )
        results.append(row)

    row = base_row(
        f"rmsnorm_{name}",
        "flashinfer_main_pdl_false",
        rows,
        hidden,
        torch.bfloat16,
        "padded_128x4",
        config,
    )
    row.update(
        graph_bench(
            fi_fn,
            warmup=config.warmup,
            samples=config.samples,
            calls_per_graph=config.calls_per_graph,
        )
    )
    results.append(row)
    return results


def bench_mxfp8_case(
    config: argparse.Namespace, rows: int, hidden: int
) -> list[dict[str, Any]]:
    from flashinfer import norm
    from flashinfer.quantization import mxfp8_quantize
    from flashinfer.tllm_enums import SfLayout

    x = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(hidden, device="cuda", dtype=torch.bfloat16)
    compiled_fns = {}
    outputs = {}
    metadata = {}
    for force_pointwise_cat in (False, True):
        key = (
            "inductor_main_coordesc_force_pointwise_cat"
            if force_pointwise_cat
            else "inductor_main_coordesc_default_cat"
        )
        compiled, output, meta = compile_ablation(
            rmsnorm_mxfp8_padded,
            x,
            weight,
            force_pointwise_cat=force_pointwise_cat,
        )
        compiled_fns[key], outputs[key], metadata[key] = compiled, output, meta

    fi_normed = torch.empty_like(x)

    def fi_fn() -> tuple[torch.Tensor, torch.Tensor]:
        norm.rmsnorm(x, weight, 1e-6, out=fi_normed, enable_pdl=False)
        return mxfp8_quantize(
            fi_normed, sf_swizzle_layout=SfLayout.layout_128x4, enable_pdl=False
        )

    fi_output = fi_fn()
    torch.cuda.synchronize()
    scale_cols = hidden // 32
    reference = outputs["inductor_main_coordesc_default_cat"]
    results = []
    for key, compiled in compiled_fns.items():
        output = outputs[key]
        row = base_row(
            "rmsnorm_mxfp8",
            key,
            rows,
            hidden,
            torch.bfloat16,
            "padded_128x4",
            config,
        )
        row.update(metadata[key])
        row["correctness_vs_default"] = {
            "quant": byte_diff(output[0], reference[0]),
            "scale": compare_scale(
                output[1], reference[1], rows, scale_cols, "ue8m0"
            ),
        }
        row["correctness_vs_flashinfer_composed"] = {
            "quant": byte_diff(output[0], fi_output[0]),
            "scale": compare_scale(
                output[1], fi_output[1].reshape(-1), rows, scale_cols, "ue8m0"
            ),
        }
        row.update(
            graph_bench(
                lambda compiled=compiled: compiled(x, weight),
                warmup=config.warmup,
                samples=config.samples,
                calls_per_graph=config.calls_per_graph,
            )
        )
        results.append(row)

    row = base_row(
        "rmsnorm_mxfp8",
        "flashinfer_main_composed_pdl_false",
        rows,
        hidden,
        torch.bfloat16,
        "padded_128x4",
        config,
    )
    row.update(
        graph_bench(
            fi_fn,
            warmup=config.warmup,
            samples=config.samples,
            calls_per_graph=config.calls_per_graph,
        )
    )
    results.append(row)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--calls-per-graph", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-control", action="store_true")
    config = parser.parse_args()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    started = time.time()
    shapes = [(129, 4096), (989, 4096)]
    if config.include_control:
        shapes.append((129, 4128))

    results = []
    for rows, hidden in shapes:
        print(f"running force_pointwise_cat ablation at {rows}x{hidden}", flush=True)
        for name, block, scale_format, scale_dtype in (
            ("nvfp4", 16, "e4m3", torch.float8_e4m3fn),
            ("mxfp4", 32, "ue8m0", torch.uint8),
        ):
            print(f"  compiling {name}", flush=True)
            results.extend(
                bench_fp4_case(
                    config,
                    rows,
                    hidden,
                    name,
                    block,
                    scale_format,
                    scale_dtype,
                )
            )
        print("  compiling mxfp8", flush=True)
        results.extend(bench_mxfp8_case(config, rows, hidden))

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
