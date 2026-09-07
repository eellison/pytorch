from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

import torch

import torch._dynamo as dynamo
import torch._inductor.config as inductor_config
from bench_main_cuda import base_row, byte_diff, graph_bench, swizzled_scale_size
from bench_padded_quant_main import (
    logical_e8m0_scale_diff,
    logical_float_scale_diff,
    rmsnorm_fp4_padded,
    rmsnorm_mxfp8_padded,
)
from torch._dynamo.utils import counters
from torch._inductor import metrics
from torch._inductor.codecache import PyCodeCache


def codegen_metadata(previous_module_ids: set[int]) -> dict[str, Any]:
    modules = [module for module in PyCodeCache.modules if id(module) not in previous_module_ids]
    source_files = []
    kernels = []
    seen_launchers = set()
    for module in modules:
        source_path = Path(module.__file__)
        if source_path.exists():
            source_files.append(
                {
                    "path": str(source_path),
                    "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                }
            )
        for name, value in vars(module).items():
            launchers = getattr(value, "launchers", None)
            if not name.startswith("triton_") or not launchers:
                continue
            launcher = launchers[0]
            if id(launcher) in seen_launchers:
                continue
            seen_launchers.add(id(launcher))
            triton_config = launcher.config
            kernels.append(
                {
                    "name": name,
                    "config": dict(triton_config.kwargs),
                    "num_warps": triton_config.num_warps,
                    "num_stages": triton_config.num_stages,
                    "num_ctas": triton_config.num_ctas,
                    "binary_cache_hash": launcher.cache_hash,
                    "registers": launcher.n_regs,
                    "spills": launcher.n_spills,
                    "shared_memory_bytes": launcher.shared,
                }
            )
    return {"source_files": source_files, "selected_kernels": kernels}


def compile_fresh(
    fn: Callable[..., Any],
    *args: torch.Tensor,
    coordinate_descent: bool,
    force_pointwise_cat: bool,
) -> tuple[Callable[..., Any], Any, dict[str, Any]]:
    dynamo.reset()
    metrics.reset()
    counters.clear()
    previous_module_ids = {id(module) for module in PyCodeCache.modules}
    started = time.time()
    with inductor_config.patch(
        {
            "triton.nested_reduction": True,
            "triton.cudagraphs": False,
            "fx_graph_cache": False,
            "emulate_precision_casts": True,
            "coordinate_descent_tuning": coordinate_descent,
            "force_pointwise_cat": force_pointwise_cat,
        }
    ):
        compiled = torch.compile(fn, fullgraph=True, dynamic=False)
        output = compiled(*args)
        torch.cuda.synchronize()
    metadata = {
        "coordinate_descent": coordinate_descent,
        "force_pointwise_cat": force_pointwise_cat,
        "compile_seconds": time.time() - started,
        "kernel_count": metrics.generated_kernel_count,
        "nested_reduction_count": metrics.codegen_nested_reduction,
        "pad_rewritten_as_cat": int(counters["inductor"]["pad_rewritten_as_cat"]),
    }
    metadata.update(codegen_metadata(previous_module_ids))
    return compiled, output, metadata


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

    compiled, output, metadata = compile_fresh(
        pt_fn,
        x,
        weight,
        coordinate_descent=config.mode != "default",
        force_pointwise_cat=config.mode == "coordesc_force_cat",
    )
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
    if scale_format == "e4m3":
        scale_diff = logical_float_scale_diff(
            output[1], fi_scale, rows, hidden // block
        )
    else:
        scale_diff = logical_e8m0_scale_diff(
            output[1], fi_scale, rows, hidden // block
        )
    implementation = {
        "default": "inductor_main_default_fresh",
        "coordesc": "inductor_main_coordesc_fresh",
        "coordesc_force_cat": "inductor_main_coordesc_force_pointwise_cat_fresh",
    }[config.mode]
    row = base_row(
        f"rmsnorm_{name}",
        implementation,
        rows,
        hidden,
        torch.bfloat16,
        "padded_128x4",
        config,
    )
    row.update(metadata)
    row["correctness_vs_flashinfer"] = {
        "quant": byte_diff(output[0], fi_quant),
        "scale": scale_diff,
    }
    row.update(
        graph_bench(
            lambda: compiled(x, weight),
            warmup=config.warmup,
            samples=config.samples,
            calls_per_graph=config.calls_per_graph,
        )
    )
    fi_row = base_row(
        f"rmsnorm_{name}",
        "flashinfer_main_pdl_false",
        rows,
        hidden,
        torch.bfloat16,
        "padded_128x4",
        config,
    )
    fi_row.update(
        graph_bench(
            fi_fn,
            warmup=config.warmup,
            samples=config.samples,
            calls_per_graph=config.calls_per_graph,
        )
    )
    return [row, fi_row]


def bench_mxfp8_case(
    config: argparse.Namespace, rows: int, hidden: int
) -> list[dict[str, Any]]:
    from flashinfer import norm
    from flashinfer.quantization import mxfp8_quantize
    from flashinfer.tllm_enums import SfLayout

    x = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(hidden, device="cuda", dtype=torch.bfloat16)
    compiled, output, metadata = compile_fresh(
        rmsnorm_mxfp8_padded,
        x,
        weight,
        coordinate_descent=config.mode != "default",
        force_pointwise_cat=config.mode == "coordesc_force_cat",
    )
    fi_normed = torch.empty_like(x)

    def fi_fn() -> tuple[torch.Tensor, torch.Tensor]:
        norm.rmsnorm(x, weight, 1e-6, out=fi_normed, enable_pdl=False)
        return mxfp8_quantize(
            fi_normed, sf_swizzle_layout=SfLayout.layout_128x4, enable_pdl=False
        )

    fi_output = fi_fn()
    torch.cuda.synchronize()
    implementation = {
        "default": "inductor_main_default_fresh",
        "coordesc": "inductor_main_coordesc_fresh",
        "coordesc_force_cat": "inductor_main_coordesc_force_pointwise_cat_fresh",
    }[config.mode]
    row = base_row(
        "rmsnorm_mxfp8",
        implementation,
        rows,
        hidden,
        torch.bfloat16,
        "padded_128x4",
        config,
    )
    row.update(metadata)
    row["correctness_vs_flashinfer_composed"] = {
        "quant": byte_diff(output[0], fi_output[0]),
        "scale": logical_e8m0_scale_diff(
            output[1], fi_output[1].reshape(-1), rows, hidden // 32
        ),
    }
    row.update(
        graph_bench(
            lambda: compiled(x, weight),
            warmup=config.warmup,
            samples=config.samples,
            calls_per_graph=config.calls_per_graph,
        )
    )
    fi_row = base_row(
        "rmsnorm_mxfp8",
        "flashinfer_main_composed_pdl_false",
        rows,
        hidden,
        torch.bfloat16,
        "padded_128x4",
        config,
    )
    fi_row.update(
        graph_bench(
            fi_fn,
            warmup=config.warmup,
            samples=config.samples,
            calls_per_graph=config.calls_per_graph,
        )
    )
    return [row, fi_row]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("default", "coordesc", "coordesc_force_cat"),
        required=True,
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--calls-per-graph", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--shape-set", choices=("priority", "remaining", "all"), default="priority"
    )
    config = parser.parse_args()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    started = time.time()
    results = []
    priority_shapes = ((129, 4096), (989, 4096))
    remaining_shapes = ((1, 4096), (19, 4096), (99, 4096), (129, 4128))
    shapes = {
        "priority": priority_shapes,
        "remaining": remaining_shapes,
        "all": priority_shapes + remaining_shapes,
    }[config.shape_set]
    for rows, hidden in shapes:
        print(f"running fresh {config.mode} at {rows}x{hidden}", flush=True)
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
            "mode": config.mode,
            "elapsed_seconds": time.time() - started,
        },
        "results": results,
    }
    config.output.parent.mkdir(parents=True, exist_ok=True)
    config.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
