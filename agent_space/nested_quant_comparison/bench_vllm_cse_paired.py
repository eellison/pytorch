from __future__ import annotations

import json
import statistics
from collections.abc import Callable

import torch

import generated_vllm_gap_coordesc as mixed
import generated_vllm_gap_default as original
from torch._C import _cuda_getCurrentRawStream
from vllm.kernels.helion.ops.rms_norm_dynamic_per_token_quant import (
    rms_norm_dynamic_per_token_quant,
)


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


torch.manual_seed(0)
rows = hidden = 4096
x = torch.zeros(rows, hidden, device="cuda", dtype=torch.bfloat16)
residual = torch.randn_like(x)
weight = torch.randn(hidden, device="cuda", dtype=torch.bfloat16)
kernel_name = (
    "triton_red_fused__to_copy_abs_add_amax_clamp_clamp_min_div_"
    "mean_mul_pow_rsqrt_0"
)


def make_inductor_runner(module) -> Callable[[], None]:
    scale = torch.empty(rows, 1, device="cuda", dtype=torch.float32)
    quant = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    summed = torch.empty_like(x)
    kernel = getattr(module, kernel_name)

    def run() -> None:
        kernel.run(
            x,
            residual,
            weight,
            scale,
            quant,
            summed,
            rows,
            hidden,
            stream=_cuda_getCurrentRawStream(0),
        )

    return run


def make_vllm_runner() -> Callable[[], None]:
    quant = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scale = torch.empty(rows, 1, device="cuda", dtype=torch.float32)
    output_residual = residual.clone()

    def run() -> None:
        rms_norm_dynamic_per_token_quant(
            quant, x, weight, scale, 1e-6, residual=output_residual
        )

    return run


runners = {
    "inductor_original": make_inductor_runner(original),
    "inductor_mixed": make_inductor_runner(mixed),
    "vllm_helion": make_vllm_runner(),
}
for runner in runners.values():
    for _ in range(20):
        runner()
torch.cuda.synchronize()

graphs = {}
for name, runner in runners.items():
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(100):
            runner()
    graphs[name] = graph
torch.cuda.synchronize()

values = {name: [] for name in runners}
orders = [list(runners), list(reversed(runners))]
for sample in range(100):
    for name in orders[sample % 2]:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graphs[name].replay()
        end.record()
        end.synchronize()
        values[name].append(start.elapsed_time(end) * 10.0)

results = {}
for name, times in values.items():
    results[name] = {
        "median_us": statistics.median(times),
        "p20_us": percentile(times, 0.2),
        "p80_us": percentile(times, 0.8),
        "min_us": min(times),
        "max_us": max(times),
    }
print(json.dumps(results, indent=2))
