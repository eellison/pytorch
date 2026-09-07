from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from bench_main_cuda import FP8_MAX, base_row, compile_main, graph_bench
from bench_padded_quant_main import fp8_payload_diff, logical_e8m0_scale_diff
from blocked_variants import VARIANTS
from torch._inductor import inductor_prims


def rmsnorm_mxfp8_padded(x, weight, to_blocked):
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


def bench_case(config, rows, hidden):
    from flashinfer import norm
    from flashinfer.quantization import mxfp8_quantize
    from flashinfer.tllm_enums import SfLayout

    x = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(hidden, device="cuda", dtype=torch.bfloat16)
    compiled_fns, outputs, metadata = {}, {}, {}
    for variant, to_blocked in VARIANTS.items():

        def pt_fn(x, weight, to_blocked=to_blocked):
            return rmsnorm_mxfp8_padded(x, weight, to_blocked)

        compiled, output, meta = compile_main(
            pt_fn, x, weight, coordinate_descent=config.coordinate_descent
        )
        compiled_fns[variant], outputs[variant], metadata[variant] = (
            compiled,
            output,
            meta,
        )

    fi_normed = torch.empty_like(x)

    def fi_fn():
        norm.rmsnorm(x, weight, 1e-6, out=fi_normed, enable_pdl=False)
        return mxfp8_quantize(
            fi_normed, sf_swizzle_layout=SfLayout.layout_128x4, enable_pdl=False
        )

    fi_out = fi_fn()
    torch.cuda.synchronize()
    reference = outputs[next(iter(VARIANTS))][1]
    callables = [
        (variant, lambda compiled=compiled: compiled(x, weight))
        for variant, compiled in compiled_fns.items()
    ]
    callables.append(("flashinfer", fi_fn))
    timings = {}
    for sweep in (callables, list(reversed(callables))):
        for label, fn in sweep:
            timing = graph_bench(
                fn,
                warmup=config.warmup,
                samples=config.samples,
                calls_per_graph=config.calls_per_graph,
            )
            previous = timings.get(label)
            if previous is None or timing["median_us"] < previous["median_us"]:
                timings[label] = timing

    results = []
    for variant, compiled in compiled_fns.items():
        output = outputs[variant]
        row = base_row(
            "rmsnorm_mxfp8",
            variant,
            rows,
            hidden,
            torch.bfloat16,
            "padded_128x4",
            config,
        )
        row.update(metadata[variant])
        row["correctness_vs_flashinfer"] = {
            "quant": fp8_payload_diff(output[0], fi_out[0]),
            "scale": logical_e8m0_scale_diff(
                output[1], fi_out[1].reshape(-1), rows, hidden // 32
            ),
        }
        row["identical_to_fpad_including_pad"] = bool(
            torch.equal(output[1], reference)
        )
        row.update(timings[variant])
        results.append(row)

    row = base_row(
        "rmsnorm_mxfp8",
        "flashinfer",
        rows,
        hidden,
        torch.bfloat16,
        "padded_128x4",
        config,
    )
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
    shapes = (
        (1, 4096),
        (19, 4096),
        (99, 4096),
        (129, 4096),
        (989, 4096),
        (129, 4128),
    )
    if config.shapes:
        shapes = tuple(
            tuple(map(int, shape.split("x"))) for shape in config.shapes.split(",")
        )
    for rows, hidden in shapes:
        print(f"running mxfp8 {rows}x{hidden}", flush=True)
        try:
            results.extend(bench_case(config, rows, hidden))
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
