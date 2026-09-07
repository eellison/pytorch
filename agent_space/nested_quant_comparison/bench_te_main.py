from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import time
from pathlib import Path


# Ignore stale wheel metadata while loading the locally built TE main extensions.
_distribution = metadata.distribution


def _isolated_distribution(name: str):
    normalized = name.lower().replace("-", "_")
    if normalized in {
        "transformer_engine_torch",
        "transformer_engine_cu12",
        "transformer_engine_cu13",
    }:
        raise metadata.PackageNotFoundError(name)
    return _distribution(name)


metadata.distribution = _isolated_distribution

import torch

import transformer_engine
import transformer_engine_torch as tex
from transformer_engine.pytorch.constants import DType, TE_DType
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

from bench_main_cuda import (
    byte_diff,
    compile_main,
    graph_bench,
    manual_rmsnorm,
    rmsnorm_mxfp8,
    tensor_diff,
    unswizzle_scale,
)


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
    for rows, hidden in ((128, 4096), (1024, 4096), (256, 8192), (128, 16384)):
        x = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(hidden, device="cuda", dtype=torch.bfloat16)
        reference = manual_rmsnorm(x, weight)
        for swizzled in (False, True):

            def pt_fn(
                x: torch.Tensor,
                weight: torch.Tensor,
                swizzled: bool = swizzled,
            ):
                return rmsnorm_mxfp8(x, weight, swizzled)

            compiled, pt_out, compile_meta = compile_main(
                pt_fn, x, weight, coordinate_descent=args.coordinate_descent
            )
            quantizer = MXFP8Quantizer(
                DType.kFloat8E4M3, rowwise=True, columnwise=False
            )
            quantizer.optimize_for_gemm = swizzled

            def te_fn():
                output, _, _ = tex.rmsnorm_fwd(
                    x,
                    weight,
                    1e-6,
                    None,
                    quantizer,
                    TE_DType[torch.bfloat16],
                    0,
                    False,
                )
                return output

            te_out = te_fn()
            torch.cuda.synchronize()
            pt_scale = pt_out[1]
            if swizzled:
                pt_scale = unswizzle_scale(pt_scale, rows, hidden // 32)
            pt_scale_f32 = torch.ldexp(
                torch.ones_like(pt_scale, dtype=torch.float32),
                pt_scale.to(torch.int32) - 127,
            )
            pt_dequant = (
                pt_out[0].view(rows, hidden // 32, 32).float()
                * pt_scale_f32.unsqueeze(-1)
            ).reshape(rows, hidden)
            correctness = {
                "payload_bytes": byte_diff(
                    pt_out[0].view(torch.uint8), te_out._rowwise_data
                ),
                "scale_bytes": byte_diff(
                    pt_out[1].reshape(-1), te_out._rowwise_scale_inv.reshape(-1)
                ),
                "dequant": tensor_diff(pt_dequant, te_out.dequantize()),
                "inductor_dequant_vs_reference": tensor_diff(pt_dequant, reference),
                "te_dequant_vs_reference": tensor_diff(te_out.dequantize(), reference),
            }
            common = {
                "suite": "rmsnorm_mxfp8",
                "shape": [rows, hidden],
                "dtype": str(x.dtype),
                "layout": "swizzled_128x4" if swizzled else "row_major",
                "warmup": args.warmup,
                "samples": args.samples,
                "calls_per_graph": args.calls_per_graph,
                "correctness": correctness,
            }
            pt_impl = (
                "inductor_main_coordesc"
                if args.coordinate_descent
                else "inductor_main"
            )
            pt_row = dict(common, implementation=pt_impl, **compile_meta)
            pt_row.update(
                graph_bench(
                    lambda: compiled(x, weight),
                    warmup=args.warmup,
                    samples=args.samples,
                    calls_per_graph=args.calls_per_graph,
                )
            )
            results.append(pt_row)
            te_row = dict(common, implementation="transformer_engine_main")
            te_row.update(
                graph_bench(
                    te_fn,
                    warmup=args.warmup,
                    samples=args.samples,
                    calls_per_graph=args.calls_per_graph,
                )
            )
            results.append(te_row)

    payload = {
        "environment": {
            "torch_version": torch.__version__,
            "torch_git_version": torch.version.git_version,
            "torch_file": torch.__file__,
            "transformer_engine_version": transformer_engine.__version__,
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
