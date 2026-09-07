from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from bench_d117_aten_padded_quant import codegen_metadata, compile_main
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
from torch._higher_order_ops.inline_asm_elementwise import inline_asm_elementwise
from torch._inductor import inductor_prims


def to_blocked_128x4_valid_only(scale: torch.Tensor) -> torch.Tensor:
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
    return torch.ops.aten._unsafe_index_put.default(output, [offset], scale, False)


def rmsnorm_fp4_ceiling(
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
    return packed, to_blocked_128x4_valid_only(scale)


def rmsnorm_mxfp8_ceiling(
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
    return quant, to_blocked_128x4_valid_only(scale)


def wrapper_source_checks(metadata: dict[str, Any]) -> dict[str, Any]:
    sources = []
    for entry in metadata["source_files"]:
        path = Path(entry["path"])
        if path.exists():
            sources.append(path.read_text())
    return {
        "external_op_call_count": sum(
            source.count("extern_kernels.") for source in sources
        ),
        "launch_pdl_true_count": sum(
            source.count("'launch_pdl': True") for source in sources
        ),
        "gdc_wait_count": sum(
            source.count("tl.extra.cuda.gdc_wait()") for source in sources
        ),
        "gdc_launch_count": sum(
            source.count("tl.extra.cuda.gdc_launch_dependents()")
            for source in sources
        ),
    }


def bench_fp4(
    config: argparse.Namespace,
    rows: int,
    name: str,
    block: int,
    scale_format: str,
    scale_dtype: torch.dtype,
) -> list[dict[str, Any]]:
    from flashinfer.cute_dsl import rmsnorm_fp4quant

    hidden = 4096
    x = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(hidden, device="cuda", dtype=torch.bfloat16)
    global_scale = torch.ones(1, device="cuda", dtype=torch.float32)

    def pt_fn(x: torch.Tensor, weight: torch.Tensor):
        return rmsnorm_fp4_ceiling(x, weight, block, scale_format)

    compiled, output, metadata = compile_main(
        pt_fn, x, weight, coordinate_descent=True
    )
    metadata["generated_source_checks"] = wrapper_source_checks(metadata)
    if metadata["kernel_count"] != 1:
        raise AssertionError(f"expected one kernel, got {metadata['kernel_count']}")
    if metadata["generated_source_checks"]["external_op_call_count"] != 0:
        raise AssertionError("unexpected external op in generated wrapper")

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
            enable_pdl=True,
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
        "d117_inductor_pdl_on_one_kernel_uninitialized_padding",
        rows,
        hidden,
        torch.bfloat16,
        "aten_scatter_padded_128x4",
        config,
    )
    row.update(metadata)
    row["padding_bytes"] = "unspecified"
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
        "flashinfer_main_pdl_true",
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


def bench_mxfp8(config: argparse.Namespace, rows: int) -> list[dict[str, Any]]:
    from flashinfer import norm
    from flashinfer.quantization import mxfp8_quantize
    from flashinfer.tllm_enums import SfLayout

    hidden = 4096
    x = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(hidden, device="cuda", dtype=torch.bfloat16)
    compiled, output, metadata = compile_main(
        rmsnorm_mxfp8_ceiling, x, weight, coordinate_descent=True
    )
    metadata["generated_source_checks"] = wrapper_source_checks(metadata)
    if metadata["kernel_count"] != 1:
        raise AssertionError(f"expected one kernel, got {metadata['kernel_count']}")
    if metadata["generated_source_checks"]["external_op_call_count"] != 0:
        raise AssertionError("unexpected external op in generated wrapper")
    fi_normed = torch.empty_like(x)

    def fi_fn() -> tuple[torch.Tensor, torch.Tensor]:
        norm.rmsnorm(x, weight, 1e-6, out=fi_normed, enable_pdl=True)
        return mxfp8_quantize(
            fi_normed, sf_swizzle_layout=SfLayout.layout_128x4, enable_pdl=True
        )

    fi_output = fi_fn()
    torch.cuda.synchronize()
    row = base_row(
        "rmsnorm_mxfp8",
        "d117_inductor_pdl_on_one_kernel_uninitialized_padding",
        rows,
        hidden,
        torch.bfloat16,
        "aten_scatter_padded_128x4",
        config,
    )
    row.update(metadata)
    row["padding_bytes"] = "unspecified"
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
        "flashinfer_main_composed_pdl_true",
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
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--calls-per-graph", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    config = parser.parse_args()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    started = time.time()
    results = []
    for rows in (129, 989):
        print(f"running uninitialized-padding ceiling at {rows}x4096", flush=True)
        results.extend(
            bench_fp4(
                config, rows, "nvfp4", 16, "e4m3", torch.float8_e4m3fn
            )
        )
        results.extend(bench_fp4(config, rows, "mxfp4", 32, "ue8m0", torch.uint8))
        results.extend(bench_mxfp8(config, rows))
    payload = {
        "environment": {
            "torch_version": torch.__version__,
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
            "coordinate_descent": True,
            "inductor_pdl": True,
            "flashinfer_pdl": True,
            "padding_bytes": "unspecified",
            "correctness_region": "logical scale values after unswizzle",
        },
        "results": results,
    }
    config.output.parent.mkdir(parents=True, exist_ok=True)
    config.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
