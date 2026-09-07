from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torchao.prototype.mx_formats.kernels import triton_to_mxfp8_dim0

from bench_main_cuda import (
    byte_diff,
    compile_main,
    graph_bench,
    manual_rmsnorm,
    rmsnorm_mxfp8,
    tensor_diff,
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

        def pt_fn(x: torch.Tensor, weight: torch.Tensor):
            return rmsnorm_mxfp8(x, weight, False)

        compiled, pt_out, compile_meta = compile_main(
            pt_fn, x, weight, coordinate_descent=args.coordinate_descent
        )
        def torchao_fn():
            normalized = F.rms_norm(x, (hidden,), weight, 1e-6)
            return triton_to_mxfp8_dim0(normalized, 32, "rceil")

        ao_out = torchao_fn()
        torch.cuda.synchronize()
        pt_scale_f32 = torch.ldexp(
            torch.ones_like(pt_out[1], dtype=torch.float32),
            pt_out[1].to(torch.int32) - 127,
        )
        ao_scale_u8 = ao_out[1].view(torch.uint8)
        ao_scale_f32 = torch.ldexp(
            torch.ones_like(ao_scale_u8, dtype=torch.float32),
            ao_scale_u8.to(torch.int32) - 127,
        )
        pt_dequant = (
            pt_out[0].view(rows, hidden // 32, 32).float()
            * pt_scale_f32.unsqueeze(-1)
        ).reshape(rows, hidden)
        ao_dequant = (
            ao_out[0].view(rows, hidden // 32, 32).float()
            * ao_scale_f32.unsqueeze(-1)
        ).reshape(rows, hidden)
        reference = manual_rmsnorm(x, weight)
        correctness = {
            "payload_bytes": byte_diff(pt_out[0], ao_out[0]),
            "scale_bytes": byte_diff(pt_out[1], ao_out[1]),
            "dequant": tensor_diff(pt_dequant, ao_dequant),
            "inductor_dequant_vs_reference": tensor_diff(pt_dequant, reference),
            "torchao_dequant_vs_reference": tensor_diff(ao_dequant, reference),
        }
        common = {
            "suite": "rmsnorm_mxfp8",
            "shape": [rows, hidden],
            "dtype": str(x.dtype),
            "layout": "row_major",
            "warmup": args.warmup,
            "samples": args.samples,
            "calls_per_graph": args.calls_per_graph,
            "correctness": correctness,
        }
        pt_impl = (
            "inductor_main_coordesc" if args.coordinate_descent else "inductor_main"
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
        ao_row = dict(common, implementation="torchao_main_composed")
        ao_row.update(
            graph_bench(
                torchao_fn,
                warmup=args.warmup,
                samples=args.samples,
                calls_per_graph=args.calls_per_graph,
            )
        )
        results.append(ao_row)

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
