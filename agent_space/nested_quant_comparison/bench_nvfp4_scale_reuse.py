from __future__ import annotations

import argparse
import types
from pathlib import Path

import torch

from bench_main_cuda import graph_bench


SOURCE = Path(
    "/tmp/torchinductor_eellison/yq/"
    "cyqyc7obdj3lfur7zejb5xb2ohhjzdzyk46y63k4szdqsykkmfyj.py"
)

OLD = """        tmp62 = tmp50.to(tl.bfloat16)
        tmp63 = tmp62.to(tl.float32)
        tmp64 = tmp63 * tmp30
        tmp65 = tmp64.to(tl.bfloat16)
        tmp66 = tmp65.to(tl.float32)
        tmp67 = tmp66.to(tl.float32)
        tmp68 = tl.maximum(tmp67, tmp35, tl.PropagateNan.ALL)
        tmp69 = tl.minimum(tmp68, tmp37, tl.PropagateNan.ALL)
        tmp70 = tmp69.to(tl.bfloat16)
        tmp71 = tmp70.to(tl.float32)
        tmp72 = tmp71.to(tl.bfloat16)
        tmp73 = tmp72.to(tl.float32)
        tmp74 = tmp73.to(tl.bfloat16)
        tmp75 = tmp74.to(tl.float32)
        tmp76 = tmp75.to(tl.float8e4nv)
        tmp77 = tmp76.to(tl.float32)
        tmp78 = tl.full([1, 1], 1.0, tl.float32)
        tmp79 = (tmp78 / tmp77)
"""

NEW = """        tmp77 = tmp45.to(tl.float32)
        tmp78 = tl.full([1, 1], 1.0, tl.float32)
        tmp79_reduced = (tmp78 / tmp77)
        tmp79 = tl.reshape(tl.broadcast_to(tmp79_reduced[:, :, None], [XBLOCK, nested_R0_REDUCED_BLOCK, (nested_R0_LOCAL_REDUCTION_SIZE//2)]), [XBLOCK, (R0_BLOCK//2)])
"""


def load_module(name: str, source: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__file__ = str(SOURCE)
    exec(compile(source, str(SOURCE), "exec"), module.__dict__)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--calls-per-graph", type=int, default=20)
    args = parser.parse_args()

    source = SOURCE.read_text()
    if source.count(OLD) != 1:
        raise RuntimeError("expected exactly one NVFP4 replay block")

    base = load_module("nvfp4_base", source)
    optimized_source = source.replace(
        "triton_red_fused__fused_rms_norm_1",
        "triton_red_fused__fused_rms_norm_reuse_1",
    ).replace(OLD, NEW)
    optimized = load_module("nvfp4_reuse", optimized_source)

    x = torch.randn(16000, 8192, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(8192, device="cuda", dtype=torch.bfloat16)
    base_scale = torch.empty(8192000, device="cuda", dtype=torch.float8_e4m3fn)
    base_quant = torch.empty((16000, 512, 8), device="cuda", dtype=torch.uint8)
    reuse_scale = torch.empty_like(base_scale)
    reuse_quant = torch.empty_like(base_quant)
    base_kernel = base.triton_red_fused__fused_rms_norm_1
    reuse_kernel = optimized.triton_red_fused__fused_rms_norm_reuse_1

    def run_base() -> None:
        stream = torch.cuda.current_stream().cuda_stream
        base_kernel.run(x, weight, base_scale, base_quant, 16000, 8192, stream=stream)

    def run_reuse() -> None:
        stream = torch.cuda.current_stream().cuda_stream
        reuse_kernel.run(x, weight, reuse_scale, reuse_quant, 16000, 8192, stream=stream)

    run_base()
    run_reuse()
    torch.cuda.synchronize()
    print("quant_equal", torch.equal(base_quant, reuse_quant))
    print("scale_equal", torch.equal(base_scale, reuse_scale))
    for label, fn in (("base", run_base), ("reuse", run_reuse)):
        print(label, graph_bench(fn, warmup=args.warmup, samples=args.samples,
                                 calls_per_graph=args.calls_per_graph))


if __name__ == "__main__":
    main()
