from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

import torch._dynamo as dynamo
import torch._inductor.config as inductor_config
from bench_main_cuda import (
    FP4_MAX,
    FP8_MAX,
    PACK_E2M1X2_ASM,
    base_row,
    byte_diff,
    graph_bench,
    recip_ue8m0,
    swizzled_scale_size,
)
from bench_padded_quant_main import (
    logical_e8m0_scale_diff,
    logical_float_scale_diff,
)
from torch._dynamo.utils import counters
from torch._higher_order_ops.inline_asm_elementwise import inline_asm_elementwise
from torch._inductor import inductor_prims, metrics
from torch._inductor.codecache import PyCodeCache


def to_blocked_128x4_aten(scale: torch.Tensor) -> torch.Tensor:
    rows, cols = scale.shape
    padded_rows = (rows + 127) // 128 * 128
    padded_cols = (cols + 3) // 4 * 4
    row = torch.arange(rows, device=scale.device)[:, None]
    col = torch.arange(cols, device=scale.device)[None, :]
    offset = row // 128 * (padded_cols // 4) * 512
    offset = offset + col // 4 * 512
    offset = offset + row % 32 * 16
    offset = offset + row // 32 % 4 * 4
    offset = offset + col % 4
    output = torch.empty(
        padded_rows * padded_cols, dtype=scale.dtype, device=scale.device
    )
    output = torch.ops.aten._unsafe_index_put.default(
        output, [offset], scale, False
    )

    linear = torch.arange(output.numel(), device=scale.device)
    col_inner = linear % 4
    linear = linear // 4
    row_outer = linear % 4
    linear = linear // 4
    row_lane = linear % 32
    linear = linear // 32
    col_outer = linear % (padded_cols // 4)
    row_chunk = linear // (padded_cols // 4)
    logical_row = row_chunk * 128 + row_outer * 32 + row_lane
    logical_col = col_outer * 4 + col_inner
    padding_mask = (logical_row >= rows) | (logical_col >= cols)
    padding = torch.zeros((), dtype=scale.dtype, device=scale.device)
    return torch.ops.aten.index_put.default(output, [padding_mask], padding, False)


def rmsnorm_fp4_aten_padded(
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
    return packed, to_blocked_128x4_aten(scale)


def rmsnorm_mxfp8_aten_padded(
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
    return quant, to_blocked_128x4_aten(scale)


def source_kernel_stats(source: str, name: str) -> dict[str, int] | None:
    pattern = re.compile(
        rf"{re.escape(name)} = async_compile\.triton\([^\n]*?'''(.*?)'''",
        re.DOTALL,
    )
    match = pattern.search(source)
    if match is None:
        return None
    kernel = match.group(1)
    return {
        "tl_load_count": kernel.count("tl.load("),
        "tl_store_count": kernel.count("tl.store("),
        "device_assert_count": kernel.count("tl.device_assert("),
        "gdc_wait_count": kernel.count("tl.extra.cuda.gdc_wait()"),
        "gdc_launch_count": kernel.count("tl.extra.cuda.gdc_launch_dependents()"),
    }


def codegen_metadata(previous_module_ids: set[int]) -> dict[str, Any]:
    modules = [module for module in PyCodeCache.modules if id(module) not in previous_module_ids]
    source_files = []
    kernels = []
    seen_launchers = set()
    for module in modules:
        source_path = Path(module.__file__)
        source = source_path.read_text() if source_path.exists() else ""
        if source:
            source_files.append(
                {
                    "path": str(source_path),
                    "sha256": hashlib.sha256(source.encode()).hexdigest(),
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
            kernel = {
                "name": name,
                "config": dict(triton_config.kwargs),
                "num_warps": triton_config.num_warps,
                "num_stages": triton_config.num_stages,
                "num_ctas": triton_config.num_ctas,
                "binary_cache_hash": launcher.cache_hash,
                "registers": launcher.n_regs,
                "spills": launcher.n_spills,
                "shared_memory_bytes": launcher.shared,
                "launch_pdl": bool(value.inductor_meta.get("launch_pdl", False)),
            }
            stats = source_kernel_stats(source, name)
            if stats is not None:
                kernel.update(stats)
            kernels.append(kernel)
    padding_candidates = [kernel for kernel in kernels if "index_put" in kernel["name"]]
    return {
        "source_files": source_files,
        "selected_kernels": kernels,
        "padding_kernel_candidates": padding_candidates,
    }


def compile_main(
    fn: Callable[..., Any], *args: torch.Tensor, coordinate_descent: bool
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
        }
    ):
        compiled = torch.compile(fn, fullgraph=True, dynamic=False)
        output = compiled(*args)
        torch.cuda.synchronize()
    metadata = {
        "coordinate_descent": coordinate_descent,
        "enable_fuse_auxiliary_writes": False,
        "compile_seconds": time.time() - started,
        "kernel_count": metrics.generated_kernel_count,
        "nested_reduction_count": metrics.codegen_nested_reduction,
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
        return rmsnorm_fp4_aten_padded(x, weight, block, scale_format)

    compiled, output, metadata = compile_main(
        pt_fn, x, weight, coordinate_descent=not config.no_coordinate_descent
    )
    fi_quant = torch.empty(
        rows, hidden // 2, device="cuda", dtype=torch.float4_e2m1fn_x2
    )
    fi_scale = torch.empty(
        swizzled_scale_size(rows, hidden, block), device="cuda", dtype=scale_dtype
    )

    def fi_fn(enable_pdl: bool = False) -> Any:
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
            enable_pdl=enable_pdl,
        )

    fi_fn()
    torch.cuda.synchronize()
    scale_diff = (
        logical_float_scale_diff(output[1], fi_scale, rows, hidden // block)
        if scale_format == "e4m3"
        else logical_e8m0_scale_diff(output[1], fi_scale, rows, hidden // block)
    )
    row = base_row(
        f"rmsnorm_{name}",
        config.label,
        rows,
        hidden,
        torch.bfloat16,
        "aten_scatter_padded_128x4",
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
    rows_out = [row]
    for pdl in ((False, True) if config.include_pdl else (False,)):
        fi_row = base_row(
            f"rmsnorm_{name}",
            f"flashinfer_main_pdl_{str(pdl).lower()}",
            rows,
            hidden,
            torch.bfloat16,
            "padded_128x4",
            config,
        )
        fi_row.update(
            graph_bench(
                lambda pdl=pdl: fi_fn(pdl),
                warmup=config.warmup,
                samples=config.samples,
                calls_per_graph=config.calls_per_graph,
            )
        )
        rows_out.append(fi_row)
    return rows_out


def bench_mxfp8_case(
    config: argparse.Namespace, rows: int, hidden: int
) -> list[dict[str, Any]]:
    from flashinfer import norm
    from flashinfer.quantization import mxfp8_quantize
    from flashinfer.tllm_enums import SfLayout

    x = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(hidden, device="cuda", dtype=torch.bfloat16)
    compiled, output, metadata = compile_main(
        rmsnorm_mxfp8_aten_padded,
        x,
        weight,
        coordinate_descent=not config.no_coordinate_descent,
    )
    fi_normed = torch.empty_like(x)

    def fi_fn(enable_pdl: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        norm.rmsnorm(x, weight, 1e-6, out=fi_normed, enable_pdl=enable_pdl)
        return mxfp8_quantize(
            fi_normed,
            sf_swizzle_layout=SfLayout.layout_128x4,
            enable_pdl=enable_pdl,
        )

    fi_output = fi_fn()
    torch.cuda.synchronize()
    row = base_row(
        "rmsnorm_mxfp8",
        config.label,
        rows,
        hidden,
        torch.bfloat16,
        "aten_scatter_padded_128x4",
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
    rows_out = [row]
    for pdl in ((False, True) if config.include_pdl else (False,)):
        fi_row = base_row(
            "rmsnorm_mxfp8",
            f"flashinfer_main_composed_pdl_{str(pdl).lower()}",
            rows,
            hidden,
            torch.bfloat16,
            "padded_128x4",
            config,
        )
        fi_row.update(
            graph_bench(
                lambda pdl=pdl: fi_fn(pdl),
                warmup=config.warmup,
                samples=config.samples,
                calls_per_graph=config.calls_per_graph,
            )
        )
        rows_out.append(fi_row)
    return rows_out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--calls-per-graph", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-coordinate-descent", action="store_true")
    parser.add_argument("--include-pdl", action="store_true")
    parser.add_argument(
        "--case-set",
        choices=("all", "anomalies", "pdl_fp4", "pdl_large", "mxfp8_large"),
        default="all",
    )
    config = parser.parse_args()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    started = time.time()
    results = []
    all_shapes = (
        (1, 4096),
        (19, 4096),
        (99, 4096),
        (129, 4096),
        (989, 4096),
        (129, 4128),
    )
    cases = (
        ((19, 4096), "mxfp8", 32, "ue8m0", torch.uint8),
        ((129, 4096), "mxfp4", 32, "ue8m0", torch.uint8),
        ((989, 4096), "nvfp4", 16, "e4m3", torch.float8_e4m3fn),
    )
    if config.case_set == "anomalies":
        for (rows, hidden), name, block, scale_format, scale_dtype in cases:
            print(f"running ATen padded scatter at {rows}x{hidden} {name}", flush=True)
            if name == "mxfp8":
                results.extend(bench_mxfp8_case(config, rows, hidden))
            else:
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
    elif config.case_set == "all":
        for rows, hidden in all_shapes:
            print(f"running ATen padded scatter at {rows}x{hidden}", flush=True)
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
    elif config.case_set in ("pdl_fp4", "pdl_large"):
        pdl_shapes = (
            (1, 4096),
            (19, 4096),
            (99, 4096),
            (129, 4096),
            (989, 4096),
        )
        if config.case_set == "pdl_large":
            pdl_shapes = ((129, 4096), (989, 4096))
        for rows, hidden in pdl_shapes:
            for name, block, scale_format, scale_dtype in (
                ("nvfp4", 16, "e4m3", torch.float8_e4m3fn),
                ("mxfp4", 32, "ue8m0", torch.uint8),
            ):
                print(
                    f"running PDL comparison at {rows}x{hidden} {name}",
                    flush=True,
                )
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
    else:
        for rows in (129, 989):
            print(f"running PDL comparison at {rows}x4096 mxfp8", flush=True)
            results.extend(bench_mxfp8_case(config, rows, 4096))
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
        "protocol": {
            "warmup": config.warmup,
            "samples": config.samples,
            "calls_per_graph": config.calls_per_graph,
            "external_cuda_graph": True,
            "inductor_internal_cudagraphs": False,
            "coordinate_descent": not config.no_coordinate_descent,
            "enable_fuse_auxiliary_writes": False,
            "flashinfer_pdl": [False, True] if config.include_pdl else [False],
            "scale_representation": "ATen _unsafe_index_put plus scalar boolean index_put",
        },
        "results": results,
    }
    config.output.parent.mkdir(parents=True, exist_ok=True)
    config.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
