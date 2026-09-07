from __future__ import annotations

import json
import statistics
from collections.abc import Callable

import torch

import generated_vllm_gap_default as original
import torch._inductor.config as inductor_config
from bench_vllm_helion_main import fused_add_dynamic_rmsnorm_fp8
from vllm.kernels.helion.ops.rms_norm_dynamic_per_token_quant import (
    rms_norm_dynamic_per_token_quant,
)


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] * (upper - position) + values[upper] * (position - lower)


def main() -> None:
    torch.manual_seed(0)
    rows = hidden = 4096
    x = torch.zeros(rows, hidden, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    weight = torch.randn(hidden, device="cuda", dtype=torch.bfloat16)

    with inductor_config.patch(
        {
            "triton.nested_reduction": True,
            "triton.cudagraphs": False,
            "fx_graph_cache": False,
            "emulate_precision_casts": True,
            "coordinate_descent_tuning": True,
            "triton.multi_loop_reduction_blocks": True,
        }
    ):
        patched = torch.compile(
            fused_add_dynamic_rmsnorm_fp8, fullgraph=True, dynamic=False
        )
        patched(x, residual, weight)
        torch.cuda.synchronize()

    vllm_quant = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    vllm_scale = torch.empty(rows, 1, device="cuda", dtype=torch.float32)
    vllm_residual = residual.clone()

    def run_original() -> None:
        original.call([x, residual, weight])

    def run_patched() -> None:
        patched(x, residual, weight)

    def run_vllm() -> None:
        rms_norm_dynamic_per_token_quant(
            vllm_quant, x, weight, vllm_scale, 1e-6, residual=vllm_residual
        )

    runners: dict[str, Callable[[], None]] = {
        "inductor_main_original": run_original,
        "inductor_per_loop": run_patched,
        "vllm_helion": run_vllm,
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

    results = {
        name: {
            "median_us": statistics.median(times),
            "p20_us": percentile(times, 0.2),
            "p80_us": percentile(times, 0.8),
            "min_us": min(times),
            "max_us": max(times),
        }
        for name, times in values.items()
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
