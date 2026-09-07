from __future__ import annotations

import json
import statistics
from collections.abc import Callable

import torch

import generated_vllm_gap_coordesc as hoisted
import generated_vllm_gap_default as unhoisted
from torch._C import _cuda_getCurrentRawStream
from triton import Config
from vllm.kernels.helion.ops.rms_norm_dynamic_per_token_quant import (
    rms_norm_dynamic_per_token_quant,
)


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def graph_bench(fn: Callable[[], None]) -> dict[str, float]:
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(100):
            fn()
    graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    values = []
    for _ in range(100):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) * 10.0)
    return {
        "median_us": statistics.median(values),
        "p20_us": percentile(values, 0.2),
        "p80_us": percentile(values, 0.8),
        "min_us": min(values),
        "max_us": max(values),
    }


def main() -> None:
    torch.manual_seed(0)
    rows = hidden = 4096
    x = torch.zeros(rows, hidden, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    weight = torch.randn(hidden, device="cuda", dtype=torch.bfloat16)
    outputs = {}
    runners = {}
    kernel_name = (
        "triton_red_fused__to_copy_abs_add_amax_clamp_clamp_min_div_"
        "mean_mul_pow_rsqrt_0"
    )
    for name, module in (
        ("inductor_original", unhoisted),
        ("mixed_blocks_helion_cache", hoisted),
    ):
        scale = torch.empty(rows, 1, device="cuda", dtype=torch.float32)
        quant = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        summed = torch.empty_like(x)
        kernel = getattr(module, kernel_name)

        def run(
            kernel=kernel, scale=scale, quant=quant, summed=summed
        ) -> None:
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

        run()
        torch.cuda.synchronize()
        outputs[name] = (quant.clone(), scale.clone(), summed.clone())
        runners[name] = run

    base_kernel = getattr(unhoisted, kernel_name)
    for rblock in (256, 512, 1024, 2048, 4096):
        for num_warps in (4, 8, 16):
            config = Config(
                {"XBLOCK": 1, "R0_BLOCK": rblock},
                num_warps=num_warps,
                num_stages=1,
            )
            try:
                compiled = base_kernel._precompile_config(config)
                launcher = compiled.make_launcher()
            except Exception as error:
                runners[f"rblock_{rblock}_warps_{num_warps}"] = str(error)
                continue

            def run_forced(launcher=launcher) -> None:
                launcher(
                    x,
                    residual,
                    weight,
                    outputs["inductor_original"][1],
                    outputs["inductor_original"][0],
                    outputs["inductor_original"][2],
                    rows,
                    hidden,
                    _cuda_getCurrentRawStream(0),
                )

            runners[f"rblock_{rblock}_warps_{num_warps}"] = run_forced

    mixed_kernel = getattr(hoisted, kernel_name)
    for num_stages in (1, 2, 4, 8):
        for num_warps in (4, 8, 16):
            config = Config(
                {"XBLOCK": 1, "R0_BLOCK": 512},
                num_warps=num_warps,
                num_stages=num_stages,
            )
            try:
                compiled = mixed_kernel._precompile_config(config)
                launcher = compiled.make_launcher()
            except Exception as error:
                runners[f"mixed_stages_{num_stages}_warps_{num_warps}"] = str(error)
                continue

            def run_mixed(launcher=launcher) -> None:
                launcher(
                    x,
                    residual,
                    weight,
                    outputs["mixed_blocks_helion_cache"][1],
                    outputs["mixed_blocks_helion_cache"][0],
                    outputs["mixed_blocks_helion_cache"][2],
                    rows,
                    hidden,
                    _cuda_getCurrentRawStream(0),
                )

            runners[f"mixed_stages_{num_stages}_warps_{num_warps}"] = run_mixed

    old_fp_fusion = mixed_kernel.triton_meta["enable_fp_fusion"]
    mixed_kernel.triton_meta["enable_fp_fusion"] = True
    fp_fusion_config = Config(
        {"XBLOCK": 1, "R0_BLOCK": 512}, num_warps=8, num_stages=8
    )
    fp_fusion_compiled = mixed_kernel._precompile_config(fp_fusion_config)
    mixed_kernel.triton_meta["enable_fp_fusion"] = old_fp_fusion
    fp_fusion_launcher = fp_fusion_compiled.make_launcher()
    fp_fusion_scale = torch.empty(rows, 1, device="cuda", dtype=torch.float32)
    fp_fusion_quant = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    fp_fusion_summed = torch.empty_like(x)

    def run_fp_fusion() -> None:
        fp_fusion_launcher(
            x,
            residual,
            weight,
            fp_fusion_scale,
            fp_fusion_quant,
            fp_fusion_summed,
            rows,
            hidden,
            _cuda_getCurrentRawStream(0),
        )

    run_fp_fusion()
    torch.cuda.synchronize()
    outputs["mixed_fp_fusion"] = (
        fp_fusion_quant.clone(),
        fp_fusion_scale.clone(),
        fp_fusion_summed.clone(),
    )
    runners["mixed_stages_8_warps_8_fp_fusion"] = run_fp_fusion

    vllm_quant = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    vllm_scale = torch.empty(rows, 1, device="cuda", dtype=torch.float32)
    vllm_residual = residual.clone()

    def run_vllm() -> None:
        rms_norm_dynamic_per_token_quant(
            vllm_quant,
            x,
            weight,
            vllm_scale,
            1e-6,
            residual=vllm_residual,
        )

    run_vllm()
    torch.cuda.synchronize()
    outputs["vllm_helion"] = (
        vllm_quant.clone(),
        vllm_scale.clone(),
        vllm_residual.clone(),
    )
    runners["vllm_helion"] = run_vllm

    correctness = {}
    for index, output_name in enumerate(("quant", "scale", "summed")):
        actual = outputs["mixed_blocks_helion_cache"][index]
        expected = outputs["inductor_original"][index]
        delta = (actual.float() - expected.float()).abs()
        correctness[output_name] = {
            "exact": bool(torch.equal(actual, expected)),
            "max_abs": float(delta.max()),
            "mean_abs": float(delta.mean()),
        }
    correctness["mixed_vs_vllm_dequant"] = {
        "max_abs": float(
            (
                outputs["mixed_blocks_helion_cache"][0].float()
                * outputs["mixed_blocks_helion_cache"][1]
                - outputs["vllm_helion"][0].float() * outputs["vllm_helion"][1]
            )
            .abs()
            .max()
        ),
        "mean_abs": float(
            (
                outputs["mixed_blocks_helion_cache"][0].float()
                * outputs["mixed_blocks_helion_cache"][1]
                - outputs["vllm_helion"][0].float() * outputs["vllm_helion"][1]
            )
            .abs()
            .mean()
        ),
    }
    correctness["fp_fusion_vs_vllm_dequant"] = {
        "max_abs": float(
            (
                outputs["mixed_fp_fusion"][0].float()
                * outputs["mixed_fp_fusion"][1]
                - outputs["vllm_helion"][0].float() * outputs["vllm_helion"][1]
            )
            .abs()
            .max()
        ),
        "mean_abs": float(
            (
                outputs["mixed_fp_fusion"][0].float()
                * outputs["mixed_fp_fusion"][1]
                - outputs["vllm_helion"][0].float() * outputs["vllm_helion"][1]
            )
            .abs()
            .mean()
        ),
    }

    results = {
        "torch": torch.__version__,
        "torch_git": torch.version.git_version,
        "gpu": torch.cuda.get_device_name(),
        "shape": [rows, hidden],
        "dtype": str(x.dtype),
        "calls_per_graph": 100,
        "samples": 100,
        "correctness": correctness,
        "timings": {
            name: graph_bench(runner) if callable(runner) else {"error": runner}
            for name, runner in runners.items()
        },
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
