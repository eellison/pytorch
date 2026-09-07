"""Padded 128x4 scale swizzle: F.pad vs index_put vs predicated-store, against FlashInfer.

Same cudagraph methodology as bench_padded_quant_main.py (graph_bench over
calls_per_graph replays), so numbers are comparable with
results/pr191974_padded_after_all.json.
"""
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
)
from bench_padded_quant_main import logical_e8m0_scale_diff, logical_float_scale_diff
from blocked_variants import VARIANTS

import torch._inductor.config as inductor_config

# per-variant extra inductor config; fpad_cat pins the 2-kernel pointwise pad
# topology instead of the ConcatKernel one that row-only padding otherwise takes
VARIANT_CONFIG = {"fpad_cat": {"force_pointwise_cat": True}}
from torch._higher_order_ops.inline_asm_elementwise import inline_asm_elementwise
from torch._inductor import inductor_prims


def rmsnorm_fp4_padded(x, weight, block, scale_format, to_blocked):
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
    return packed, to_blocked(scale)


def bench_case(config, rows, hidden, name, block, scale_format, scale_dtype):
    from flashinfer.cute_dsl import rmsnorm_fp4quant

    x = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(hidden, device="cuda", dtype=torch.bfloat16)
    global_scale = torch.ones(1, device="cuda", dtype=torch.float32)

    compiled_fns, outputs, metadata = {}, {}, {}
    for variant, to_blocked in VARIANTS.items():
        def pt_fn(x, weight, to_blocked=to_blocked):
            return rmsnorm_fp4_padded(x, weight, block, scale_format, to_blocked)

        with inductor_config.patch(VARIANT_CONFIG.get(variant, {})):
            compiled, output, meta = compile_main(
                pt_fn, x, weight, coordinate_descent=config.coordinate_descent
            )
        compiled_fns[variant], outputs[variant], metadata[variant] = compiled, output, meta

    fi_quant = torch.empty(rows, hidden // 2, device="cuda", dtype=torch.float4_e2m1fn_x2)
    fi_scale = torch.empty(
        swizzled_scale_size(rows, hidden, block), device="cuda", dtype=scale_dtype
    )

    def fi_fn():
        return rmsnorm_fp4quant(
            x, weight, y_fp4=fi_quant, block_scale=fi_scale, global_scale=global_scale,
            eps=1e-6, block_size=block, scale_format=scale_format,
            is_sf_swizzled_layout=True, enable_pdl=False,
        )

    fi_fn()
    torch.cuda.synchronize()

    scale_cols = hidden // block
    reference = outputs[next(iter(VARIANTS))][1]

    # Timing is order-sensitive (GPU clocks ramp across a case), so sweep the
    # variants forward and then in reverse and keep each one's best.
    callables = [(v, lambda c=c: c(x, weight)) for v, c in compiled_fns.items()]
    callables.append(("flashinfer", fi_fn))
    timings = {}
    for sweep in (callables, list(reversed(callables))):
        for label, fn in sweep:
            got = graph_bench(fn, warmup=config.warmup, samples=config.samples,
                              calls_per_graph=config.calls_per_graph)
            prev = timings.get(label)
            if prev is None or got["median_us"] < prev["median_us"]:
                timings[label] = got

    results = []
    for variant in compiled_fns:
        output = outputs[variant]
        diff = logical_float_scale_diff if scale_format == "e4m3" else logical_e8m0_scale_diff
        row = base_row(f"rmsnorm_{name}", variant, rows, hidden, torch.bfloat16, "padded_128x4", config)
        row.update(metadata[variant])
        row["correctness_vs_flashinfer"] = {
            "quant": byte_diff(output[0], fi_quant),
            "scale": diff(output[1], fi_scale, rows, scale_cols),
        }
        row["identical_to_fpad_including_pad"] = bool(torch.equal(output[1], reference))
        row.update(timings[variant])
        results.append(row)

    row = base_row(f"rmsnorm_{name}", "flashinfer", rows, hidden, torch.bfloat16, "padded_128x4", config)
    row.update(timings["flashinfer"])
    results.append(row)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--calls-per-graph", type=int, default=100)
    parser.add_argument("--coordinate-descent", action="store_true")
    parser.add_argument("--variants", default="")
    parser.add_argument("--shapes", default="")
    parser.add_argument("--output", type=Path, required=True)
    config = parser.parse_args()
    if config.variants:
        keep = set(config.variants.split(","))
        for name in list(VARIANTS):
            if name not in keep:
                del VARIANTS[name]
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    started = time.time()
    results = []
    shapes = ((1, 4096), (19, 4096), (99, 4096), (129, 4096), (989, 4096), (129, 4128))
    if config.shapes:
        shapes = tuple(
            tuple(map(int, shape.split("x"))) for shape in config.shapes.split(",")
        )
    for rows, hidden in shapes:
        for name, block, scale_format, scale_dtype in (
            ("nvfp4", 16, "e4m3", torch.float8_e4m3fn),
            ("mxfp4", 32, "ue8m0", torch.uint8),
        ):
            print(f"running {name} {rows}x{hidden}", flush=True)
            try:
                results.extend(bench_case(config, rows, hidden, name, block, scale_format, scale_dtype))
            except Exception as error:
                import traceback; traceback.print_exc()
                results.append({"suite": f"rmsnorm_{name}", "shape": [rows, hidden],
                                "error": f"{type(error).__name__}: {error}"})
    payload = {
        "environment": {
            "torch_version": torch.__version__,
            "torch_git_version": torch.version.git_version,
            "torch_file": torch.__file__,
            "gpu": torch.cuda.get_device_name(),
            "coordinate_descent": config.coordinate_descent,
            "elapsed_seconds": time.time() - started,
        },
        "results": results,
    }
    config.output.parent.mkdir(parents=True, exist_ok=True)
    config.output.write_text(json.dumps(payload, indent=2) + "\n")
    print("wrote", config.output, flush=True)


if __name__ == "__main__":
    main()
