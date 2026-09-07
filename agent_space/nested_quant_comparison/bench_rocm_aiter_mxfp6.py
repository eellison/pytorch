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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--calls-per-graph", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if torch.version.hip is None:
        raise RuntimeError("This benchmark requires a ROCm PyTorch main build")

    import aiter.ops.gemm_op_a6w6 as aiter_mxfp6
    from aiter.jit.utils.chip_info import get_gfx_runtime

    if get_gfx_runtime() != "gfx950":
        raise RuntimeError("AITER's native MXFP6 packer requires gfx950")

    aiter_mxfp6._QUANT_BACKEND = "hip"
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    results = []
    started = time.time()
    for dtype in (torch.bfloat16, torch.float16):
        for rows, hidden in ((256, 4096), (1024, 4096), (4096, 4096), (256, 8192)):
            x = torch.randn(rows, hidden, device="cuda", dtype=dtype)
            packed_size, scale_size = aiter_mxfp6.mxfp6_gemm_pack_size(rows, hidden)
            aiter_packed = torch.zeros(packed_size, device="cuda", dtype=torch.uint8)
            aiter_scale = torch.zeros(scale_size, device="cuda", dtype=torch.uint8)

            def inductor_reference(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                codes, scales = aiter_mxfp6.quant_mxfp6_torch(x)
                return (
                    aiter_mxfp6.pack_big_torch(codes),
                    aiter_mxfp6.pack_scale_torch(scales, rows),
                )

            dynamo.reset()
            metrics.reset()
            with inductor_config.patch(
                {
                    "triton.nested_reduction": True,
                    "triton.cudagraphs": False,
                    "fx_graph_cache": False,
                }
            ):
                compiled = torch.compile(inductor_reference, fullgraph=True, dynamic=False)
                pt_out = compiled(x)
                torch.cuda.synchronize()
            kernel_count = metrics.generated_kernel_count
            nested_count = metrics.codegen_nested_reduction

            def aiter_fn() -> tuple[torch.Tensor, torch.Tensor]:
                return aiter_mxfp6.quant_mxfp6_gemm_out(x, aiter_packed, aiter_scale)

            aiter_fn()
            torch.cuda.synchronize()
            packed_mismatch = int((pt_out[0] != aiter_packed).sum())
            scale_mismatch = int((pt_out[1] != aiter_scale).sum())
            common = {
                "suite": "mxfp6_e2m3_hadamard_gemm_pack",
                "shape": [rows, hidden],
                "dtype": str(dtype),
                "layout": "aiter_mxfp6_c0c1_256_padk2",
                "warmup": args.warmup,
                "samples": args.samples,
                "calls_per_graph": args.calls_per_graph,
                "packed_mismatch_count": packed_mismatch,
                "scale_mismatch_count": scale_mismatch,
            }
            pt_row = dict(common, implementation="inductor_main")
            pt_row.update(
                {
                    "kernel_count": kernel_count,
                    "nested_reduction_count": nested_count,
                }
            )
            pt_row.update(
                graph_bench(
                    lambda: compiled(x),
                    warmup=args.warmup,
                    samples=args.samples,
                    calls_per_graph=args.calls_per_graph,
                )
            )
            results.append(pt_row)
            aiter_row = dict(common, implementation="aiter_main_hip")
            aiter_row.update(
                graph_bench(
                    aiter_fn,
                    warmup=args.warmup,
                    samples=args.samples,
                    calls_per_graph=args.calls_per_graph,
                )
            )
            results.append(aiter_row)

    payload = {
        "environment": {
            "torch_version": torch.__version__,
            "torch_git_version": torch.version.git_version,
            "torch_file": torch.__file__,
            "rocm_version": torch.version.hip,
            "gpu": torch.cuda.get_device_name(),
            "gfx": get_gfx_runtime(),
            "elapsed_seconds": time.time() - started,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
