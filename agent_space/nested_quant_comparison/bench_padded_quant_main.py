from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from bench_main_cuda import (
    FP4_MAX,
    FP8_MAX,
    PACK_E2M1X2_ASM,
    base_row,
    byte_diff,
    compile_main,
    graph_bench,
    recip_ue8m0,
    swizzled_scale_size,
    tensor_diff,
)
from torch._higher_order_ops.inline_asm_elementwise import inline_asm_elementwise
from torch._inductor import inductor_prims


def to_blocked_padded(scale: torch.Tensor) -> torch.Tensor:
    rows, cols = scale.shape
    padded_rows = (rows + 127) // 128 * 128
    padded_cols = (cols + 3) // 4 * 4
    scale = F.pad(scale, (0, padded_cols - cols, 0, padded_rows - rows))
    blocks = scale.view(padded_rows // 128, 128, padded_cols // 4, 4)
    return blocks.permute(0, 2, 1, 3).reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1)


def from_blocked_padded(scale: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    padded_rows = (rows + 127) // 128 * 128
    padded_cols = (cols + 3) // 4 * 4
    blocks = scale.reshape(-1, 32, 4, 4).transpose(1, 2)
    padded = blocks.reshape(padded_rows // 128, padded_cols // 4, 128, 4)
    return padded.permute(0, 2, 1, 3).reshape(padded_rows, padded_cols)[:rows, :cols]


def logical_float_scale_diff(
    actual: torch.Tensor, expected: torch.Tensor, rows: int, cols: int
) -> dict[str, float | int | bool | str]:
    actual = from_blocked_padded(actual, rows, cols)
    expected = from_blocked_padded(expected, rows, cols)
    result = tensor_diff(actual, expected)
    bytes_result = byte_diff(actual, expected)
    result.update(
        {
            "format": "e4m3_scale_value",
            "comparison_region": "logical_after_unswizzle",
            "logical_values": actual.numel(),
            "mismatched_values": bytes_result["mismatched_bytes"],
            "mismatch_fraction": bytes_result["mismatch_fraction"],
        }
    )
    return result


def logical_e8m0_scale_diff(
    actual: torch.Tensor, expected: torch.Tensor, rows: int, cols: int
) -> dict[str, float | int | bool | str]:
    actual = from_blocked_padded(actual, rows, cols)
    expected = from_blocked_padded(expected, rows, cols)
    delta = actual.to(torch.int16) - expected.to(torch.int16)
    abs_delta = delta.abs()
    mismatches = int((delta != 0).sum())
    max_delta = int(abs_delta.max())
    return {
        "format": "ue8m0_exponent_code",
        "comparison_region": "logical_after_unswizzle",
        "exact": mismatches == 0,
        "logical_values": actual.numel(),
        "mismatched_values": mismatches,
        "mismatch_fraction": mismatches / actual.numel(),
        "max_exponent_delta": max_delta,
        "mean_abs_exponent_delta": float(abs_delta.float().mean()),
        "max_scale_factor": float(2**max_delta),
    }


def fp8_payload_diff(
    actual: torch.Tensor, expected: torch.Tensor
) -> dict[str, float | int | bool]:
    result = tensor_diff(actual, expected)
    bytes_result = byte_diff(actual, expected)
    result.update(
        {
            "mismatched_values": bytes_result["mismatched_bytes"],
            "mismatch_fraction": bytes_result["mismatch_fraction"],
        }
    )
    return result


def rmsnorm_fp4_padded(
    x: torch.Tensor, weight: torch.Tensor, block: int, scale_format: str
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, hidden = x.shape
    normed = F.rms_norm(x, (hidden,), weight).view(rows, hidden // block, block)
    amax = normed.abs().amax(dim=-1)
    if scale_format == "e4m3":
        scale = (amax / FP4_MAX).clamp(min=1e-12, max=FP8_MAX).to(torch.float8_e4m3fn)
        inv_scale = scale.float().reciprocal()
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
    return packed, to_blocked_padded(scale)


def rmsnorm_mxfp8_padded(
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
    return quant, to_blocked_padded(scale)


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
    outputs = {}
    compiled_fns = {}
    metadata = {}
    for coordinate_descent in (False, True):
        def pt_fn(x, weight, block=block, scale_format=scale_format):
            return rmsnorm_fp4_padded(x, weight, block, scale_format)

        compiled, output, meta = compile_main(
            pt_fn, x, weight, coordinate_descent=coordinate_descent
        )
        key = "inductor_main_coordesc" if coordinate_descent else "inductor_main"
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
    results = []
    for key, compiled in compiled_fns.items():
        output = outputs[key]
        scale_cols = hidden // block
        if scale_format == "e4m3":
            scale_correctness = logical_float_scale_diff(
                output[1], fi_scale, rows, scale_cols
            )
        else:
            scale_correctness = logical_e8m0_scale_diff(
                output[1], fi_scale, rows, scale_cols
            )
        correctness = {
            "quant": byte_diff(output[0], fi_quant),
            "scale": scale_correctness,
        }
        row = base_row(
            f"rmsnorm_{name}", key, rows, hidden, torch.bfloat16, "padded_128x4", config
        )
        row.update(metadata[key])
        row["correctness_vs_flashinfer"] = correctness
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
        f"rmsnorm_{name}", "flashinfer_main_pdl_false", rows, hidden, torch.bfloat16, "padded_128x4", config
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
    outputs = {}
    compiled_fns = {}
    metadata = {}
    for coordinate_descent in (False, True):
        compiled, output, meta = compile_main(
            rmsnorm_mxfp8_padded, x, weight, coordinate_descent=coordinate_descent
        )
        key = "inductor_main_coordesc" if coordinate_descent else "inductor_main"
        compiled_fns[key], outputs[key], metadata[key] = compiled, output, meta

    fi_normed = torch.empty_like(x)

    def fi_fn() -> tuple[torch.Tensor, torch.Tensor]:
        norm.rmsnorm(x, weight, 1e-6, out=fi_normed, enable_pdl=False)
        return mxfp8_quantize(
            fi_normed, sf_swizzle_layout=SfLayout.layout_128x4, enable_pdl=False
        )

    fi_out = fi_fn()
    torch.cuda.synchronize()
    results = []
    for key, compiled in compiled_fns.items():
        output = outputs[key]
        correctness = {
            "quant": fp8_payload_diff(output[0], fi_out[0]),
            "scale": logical_e8m0_scale_diff(
                output[1], fi_out[1].reshape(-1), rows, hidden // 32
            ),
        }
        row = base_row(
            "rmsnorm_mxfp8", key, rows, hidden, torch.bfloat16, "padded_128x4", config
        )
        row.update(metadata[key])
        row["correctness_vs_flashinfer_composed"] = correctness
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
        "rmsnorm_mxfp8", "flashinfer_main_composed_pdl_false", rows, hidden, torch.bfloat16, "padded_128x4", config
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
    config = parser.parse_args()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    started = time.time()
    results = []
    shapes = ((1, 4096), (19, 4096), (99, 4096), (129, 4096), (989, 4096), (129, 4128))
    for rows, hidden in shapes:
        print(f"running padded formats at {rows}x{hidden}", flush=True)
        for name, block, scale_format, scale_dtype in (
            ("nvfp4", 16, "e4m3", torch.float8_e4m3fn),
            ("mxfp4", 32, "ue8m0", torch.uint8),
        ):
            try:
                results.extend(
                    bench_fp4_case(
                        config, rows, hidden, name, block, scale_format, scale_dtype
                    )
                )
            except Exception as error:
                results.append(
                    {
                        "suite": f"rmsnorm_{name}",
                        "shape": [rows, hidden],
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
        try:
            results.extend(bench_mxfp8_case(config, rows, hidden))
        except Exception as error:
            results.append(
                {
                    "suite": "rmsnorm_mxfp8",
                    "shape": [rows, hidden],
                    "error": f"{type(error).__name__}: {error}",
                }
            )

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
