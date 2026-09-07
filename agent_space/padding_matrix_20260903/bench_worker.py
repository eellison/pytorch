from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from torch._higher_order_ops.inline_asm_elementwise import inline_asm_elementwise
from torch._inductor import config, inductor_prims, metrics
from torch._inductor.codecache import PyCodeCache
from torch._inductor.utils import fresh_inductor_cache, run_and_get_code


FP4_MAX = 6.0
FP6_MAX = 7.5
FP8_MAX = 448.0
MX_GROUP = 32
DCN_LOGICAL_ROWS = 96
DCN_PHYSICAL_ROWS = 128
DCN_XDL = 32
DCN_COL_CHUNK = 4
DCN_COL_INNER = 2
DCN_PAD_VALUE = 127
PACK_E2M1X2_ASM = (
    "{.reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, $2, $1; cvt.u32.u8 $0, t;}"
)
RECIP_UE8M0_ASM = (
    "{.reg .pred p_zero; .reg .s32 neg_exp; .reg .f32 neg_exp_f, result; "
    "setp.eq.u32 p_zero, $1, 0; sub.s32 neg_exp, 127, $1; "
    "cvt.rn.f32.s32 neg_exp_f, neg_exp; ex2.approx.f32 result, neg_exp_f; "
    "selp.f32 $0, 0f00000000, result, p_zero;}"
)
E2M3X2_UNPACK_ASM = (
    "{.reg .b16 pair; .reg .b32 bits; "
    "cvt.rn.satfinite.e2m3x2.f32 pair, $3, $2; "
    "cvt.u32.u16 bits, pair; and.b32 $0, bits, 63; "
    "shr.u32 $1, bits, 8; and.b32 $1, $1, 63;}"
)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] * (upper - position) + ordered[upper] * (
        position - lower
    )


def graph_bench(
    fn: Callable[[], Any], warmup: int, samples: int, calls_per_graph: int
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


def tensor_hash(value: torch.Tensor) -> str:
    payload = value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def output_metadata(output: Any) -> list[dict[str, Any]]:
    values = output if isinstance(output, (tuple, list)) else (output,)
    return [
        {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "bytes": value.numel() * value.element_size(),
            "sha256": tensor_hash(value),
        }
        for value in values
    ]


def byte_diff(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    actual = actual.contiguous().view(torch.uint8).reshape(-1)
    expected = expected.contiguous().view(torch.uint8).reshape(-1)
    mismatches = int((actual != expected).sum())
    return {
        "exact": mismatches == 0,
        "mismatched_bytes": mismatches,
        "mismatch_fraction": mismatches / actual.numel(),
    }


def recip_ue8m0(scale: torch.Tensor) -> torch.Tensor:
    return inline_asm_elementwise(
        scale.to(torch.int32),
        asm_str=RECIP_UE8M0_ASM,
        constraints="=f,r",
        dtype=torch.float32,
        is_pure=True,
        pack=1,
    )


def native_e2m3_values(values: torch.Tensor) -> torch.Tensor:
    return inline_asm_elementwise(
        values,
        asm_str=E2M3X2_UNPACK_ASM,
        constraints="=r,=r,f,f",
        dtype=torch.int32,
        is_pure=True,
        pack=2,
    )


def pack_e2m3(values: torch.Tensor) -> torch.Tensor:
    values = torch.ops._inductor_test.realize(values.to(torch.int32) & 0x3F)
    values = values.reshape(*values.shape[:-1], values.shape[-1] // 4, 4)
    low = values[..., 0] | ((values[..., 1] & 0x03) << 6)
    middle = ((values[..., 1] >> 2) & 0x0F) | ((values[..., 2] & 0x0F) << 4)
    high = ((values[..., 2] >> 4) & 0x03) | (values[..., 3] << 2)
    low = torch.ops._inductor_test.realize(low)
    middle = torch.ops._inductor_test.realize(middle)
    high = torch.ops._inductor_test.realize(high)
    return torch.stack((low, middle, high), dim=-1).to(torch.uint8)


def blocked_shape(rows: int, cols: int) -> tuple[int, int]:
    return (rows + 127) // 128 * 128, (cols + 3) // 4 * 4


def blocked_offsets(rows: torch.Tensor, cols: torch.Tensor, padded_cols: int):
    row = rows[:, None]
    col = cols[None, :]
    return (
        row // 128 * (padded_cols // 4) * 512
        + col // 4 * 512
        + row % 32 * 16
        + row // 32 % 4 * 4
        + col % 4
    )


def blocked_fpad(scale: torch.Tensor, padding_value: int = 0) -> torch.Tensor:
    rows, cols = scale.shape
    padded_rows, padded_cols = blocked_shape(rows, cols)
    scale = F.pad(
        scale,
        (0, padded_cols - cols, 0, padded_rows - rows),
        value=padding_value,
    )
    blocks = scale.view(padded_rows // 128, 128, padded_cols // 4, 4)
    return (
        blocks.permute(0, 2, 1, 3)
        .reshape(-1, 4, 32, 4)
        .transpose(1, 2)
        .reshape(-1)
    )


def blocked_fill_scatter(
    scale: torch.Tensor, padding_value: int = 0
) -> torch.Tensor:
    rows, cols = scale.shape
    padded_rows, padded_cols = blocked_shape(rows, cols)
    row = torch.arange(rows, device=scale.device)
    col = torch.arange(cols, device=scale.device)
    offsets = blocked_offsets(row, col, padded_cols)
    out = torch.full(
        (padded_rows * padded_cols,),
        padding_value,
        dtype=scale.dtype,
        device=scale.device,
    )
    return torch.ops.aten._unsafe_index_put.default(out, [offsets], scale, False)


def blocked_pad_scatter(
    scale: torch.Tensor, padding_value: int = 0
) -> torch.Tensor:
    rows, cols = scale.shape
    padded_rows, padded_cols = blocked_shape(rows, cols)
    device = scale.device
    out = torch.empty(padded_rows * padded_cols, dtype=scale.dtype, device=device)
    offsets = blocked_offsets(
        torch.arange(rows, device=device),
        torch.arange(cols, device=device),
        padded_cols,
    )
    out = torch.ops.aten._unsafe_index_put.default(out, [offsets], scale, False)
    for row_lo, row_hi, col_lo, col_hi in (
        (rows, padded_rows, 0, padded_cols),
        (0, rows, cols, padded_cols),
    ):
        if row_lo >= row_hi or col_lo >= col_hi:
            continue
        pad_offsets = blocked_offsets(
            torch.arange(row_lo, row_hi, device=device),
            torch.arange(col_lo, col_hi, device=device),
            padded_cols,
        )
        values = torch.full(
            (row_hi - row_lo, col_hi - col_lo),
            padding_value,
            dtype=scale.dtype,
            device=device,
        )
        out = torch.ops.aten._unsafe_index_put.default(
            out, [pad_offsets], values, False
        )
    return out


def blocked_predicated(
    scale: torch.Tensor, padding_value: int = 0
) -> torch.Tensor:
    if not hasattr(inductor_prims, "predicated_masked_fill"):
        raise RuntimeError("predicated_masked_fill is unavailable in this worktree")
    rows, cols = scale.shape
    padded_rows, padded_cols = blocked_shape(rows, cols)
    device = scale.device
    row = torch.arange(rows, device=device)
    col = torch.arange(cols, device=device)
    offsets = blocked_offsets(row, col, padded_cols)
    out = torch.empty(padded_rows * padded_cols, dtype=scale.dtype, device=device)
    out = torch.ops.aten._unsafe_index_put.default(out, [offsets], scale, False)
    valid = blocked_fpad(torch.ones_like(scale, dtype=torch.bool)).bool()
    value = torch.full((), padding_value, dtype=scale.dtype, device=device)
    return inductor_prims.predicated_masked_fill(out, ~valid, value)


BLOCKED_VARIANTS = {
    "fpad": blocked_fpad,
    "fill_scatter": blocked_fill_scatter,
    "pad_scatter": blocked_pad_scatter,
    "predicated": blocked_predicated,
}


def xdl_shape(rows: int, cols: int) -> tuple[int, int]:
    chunks = (rows + DCN_LOGICAL_ROWS - 1) // DCN_LOGICAL_ROWS
    return chunks * DCN_PHYSICAL_ROWS, (cols + 3) // 4 * 4


def xdl_fpad(scale: torch.Tensor, padding_value: int = DCN_PAD_VALUE):
    rows, cols = scale.shape
    padded_rows, padded_cols = xdl_shape(rows, cols)
    chunks = padded_rows // DCN_PHYSICAL_ROWS
    scale = F.pad(
        scale,
        (0, padded_cols - cols, 0, chunks * DCN_LOGICAL_ROWS - rows),
        value=padding_value,
    )
    scale = scale.view(chunks, DCN_LOGICAL_ROWS, padded_cols)
    scale = F.pad(
        scale,
        (0, 0, 0, DCN_PHYSICAL_ROWS - DCN_LOGICAL_ROWS),
        value=padding_value,
    ).reshape(padded_rows, padded_cols)
    blocks = scale.view(
        padded_rows // (2 * DCN_XDL),
        2,
        DCN_XDL,
        padded_cols // DCN_COL_CHUNK,
        2,
        DCN_COL_INNER,
    )
    return blocks.permute(0, 3, 5, 2, 4, 1).reshape(-1)


def xdl_offsets(scale: torch.Tensor, padded_cols: int) -> torch.Tensor:
    rows, cols = scale.shape
    row = torch.arange(rows, device=scale.device)[:, None]
    col = torch.arange(cols, device=scale.device)[None, :]
    physical_row = (
        row // DCN_LOGICAL_ROWS * DCN_PHYSICAL_ROWS + row % DCN_LOGICAL_ROWS
    )
    row_outer = physical_row // (2 * DCN_XDL)
    row_inner = physical_row % (2 * DCN_XDL)
    col_outer = col // DCN_COL_CHUNK
    col_inner = col % DCN_COL_CHUNK
    return (
        (
            (
                (row_outer * (padded_cols // DCN_COL_CHUNK) + col_outer)
                * DCN_COL_INNER
                + col_inner % DCN_COL_INNER
            )
            * DCN_XDL
            + row_inner % DCN_XDL
        )
        * 2
        + col_inner // DCN_COL_INNER
    ) * 2 + row_inner // DCN_XDL


def xdl_fill_scatter(scale: torch.Tensor, padding_value: int = DCN_PAD_VALUE):
    rows, cols = scale.shape
    padded_rows, padded_cols = xdl_shape(rows, cols)
    out = torch.full(
        (padded_rows * padded_cols,),
        padding_value,
        dtype=scale.dtype,
        device=scale.device,
    )
    return torch.ops.aten._unsafe_index_put.default(
        out, [xdl_offsets(scale, padded_cols)], scale, False
    )


def xdl_auxiliary(scale: torch.Tensor, padding_value: int = DCN_PAD_VALUE):
    if not hasattr(inductor_prims, "padded_xdl_scale_scatter"):
        raise RuntimeError("padded_xdl_scale_scatter is unavailable in this worktree")
    rows, cols = scale.shape
    padded_rows, padded_cols = xdl_shape(rows, cols)
    return inductor_prims.padded_xdl_scale_scatter(
        scale,
        padded_rows,
        padded_cols,
        DCN_LOGICAL_ROWS,
        DCN_PHYSICAL_ROWS,
        DCN_XDL,
        DCN_COL_CHUNK,
        DCN_COL_INNER,
        padding_value,
    )


def xdl_prim(scale: torch.Tensor, padding_value: int = DCN_PAD_VALUE):
    if not hasattr(inductor_prims, "to_padded_blocked"):
        raise RuntimeError("to_padded_blocked is unavailable in this worktree")
    return inductor_prims.to_padded_blocked(
        scale,
        DCN_LOGICAL_ROWS,
        DCN_PHYSICAL_ROWS,
        DCN_XDL,
        DCN_COL_INNER,
        padding_value,
    )


XDL_VARIANTS = {
    "fpad": xdl_fpad,
    "prim": xdl_prim,
    "fill_scatter": xdl_fill_scatter,
    "auxiliary": xdl_auxiliary,
}


def mxfp4_quantize(x: torch.Tensor, pad: Callable) -> tuple[torch.Tensor, torch.Tensor]:
    rows, hidden = x.shape
    groups = x.view(rows, hidden // MX_GROUP, MX_GROUP)
    amax = groups.abs().amax(dim=-1)
    scale = inductor_prims.cvt_e8m0_rceil((amax / FP4_MAX).clamp_min(1e-12))
    inverse = recip_ue8m0(scale)
    pairs = groups.view(rows, hidden // MX_GROUP, MX_GROUP // 2, 2)
    packed = inline_asm_elementwise(
        pairs[..., 0].float() * inverse.unsqueeze(-1),
        pairs[..., 1].float() * inverse.unsqueeze(-1),
        asm_str=PACK_E2M1X2_ASM,
        constraints="=r,f,f",
        dtype=torch.int32,
        is_pure=True,
        pack=1,
    ).to(torch.uint8).view(rows, hidden // 2)
    return packed, pad(scale)


def nvfp4_quantize(x: torch.Tensor, pad: Callable) -> tuple[torch.Tensor, torch.Tensor]:
    rows, hidden = x.shape
    groups = x.view(rows, hidden // 16, 16)
    amax = groups.abs().amax(dim=-1)
    scale = (amax / FP4_MAX).clamp(min=1e-12, max=FP8_MAX).to(torch.float8_e4m3fn)
    inverse = scale.float().reciprocal()
    pairs = groups.view(rows, hidden // 16, 8, 2)
    packed = inline_asm_elementwise(
        pairs[..., 0].float() * inverse.unsqueeze(-1),
        pairs[..., 1].float() * inverse.unsqueeze(-1),
        asm_str=PACK_E2M1X2_ASM,
        constraints="=r,f,f",
        dtype=torch.int32,
        is_pure=True,
        pack=1,
    ).to(torch.uint8).view(rows, hidden // 2)
    return packed, pad(scale)


def rmsnorm_mxfp4(
    x: torch.Tensor, weight: torch.Tensor, pad: Callable
) -> tuple[torch.Tensor, torch.Tensor]:
    return mxfp4_quantize(F.rms_norm(x, (x.shape[-1],), weight, eps=1e-6), pad)


def rmsnorm_nvfp4(
    x: torch.Tensor, weight: torch.Tensor, pad: Callable
) -> tuple[torch.Tensor, torch.Tensor]:
    return nvfp4_quantize(F.rms_norm(x, (x.shape[-1],), weight, eps=1e-6), pad)


def mxfp8_quantize(
    x: torch.Tensor, pad: Callable
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, hidden = x.shape
    groups = x.view(rows, hidden // MX_GROUP, MX_GROUP)
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
    return quant, pad(scale)


def rmsnorm_mxfp8(
    x: torch.Tensor, weight: torch.Tensor, pad: Callable
) -> tuple[torch.Tensor, torch.Tensor]:
    return mxfp8_quantize(F.rms_norm(x, (x.shape[-1],), weight, eps=1e-6), pad)


def mxfp6_quantize(
    x: torch.Tensor, pad: Callable
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, hidden = x.shape
    blocks = x.reshape(rows, hidden // MX_GROUP, MX_GROUP).float()
    amax = blocks.abs().amax(dim=-1)
    exponent = torch.ceil(torch.log2((amax / FP6_MAX).clamp(min=2.0**-127)))
    exponent = torch.where(amax == 0, torch.zeros_like(exponent), exponent)
    exponent = exponent.clamp(min=-127.0, max=127.0)
    scaled = blocks / torch.pow(2.0, exponent).unsqueeze(-1)
    codes = native_e2m3_values(scaled)
    codes = torch.where(scaled == 0, torch.zeros_like(codes), codes)
    packed = pack_e2m3(codes).reshape(rows, hidden * 3 // 4)
    scale = (exponent.to(torch.int32) + 127).to(torch.uint8)
    return packed, pad(scale)


def rmsnorm_mxfp6(
    x: torch.Tensor, weight: torch.Tensor, pad: Callable
) -> tuple[torch.Tensor, torch.Tensor]:
    return mxfp6_quantize(F.rms_norm(x, (x.shape[-1],), weight, eps=1e-6), pad)


def dcn_mxfp6(
    x: torch.Tensor, other: torch.Tensor, pad: Callable
) -> tuple[torch.Tensor, torch.Tensor]:
    return mxfp6_quantize(torch.addcmul(x, x, other), pad)


def make_inputs(workload: str, rows: int, hidden: int):
    if workload == "swizzle_mxfp4":
        return (torch.randint(0, 255, (rows, hidden // 32), device="cuda", dtype=torch.uint8),)
    if workload == "swizzle_dcn":
        return (torch.randint(0, 255, (rows, hidden // 32), device="cuda", dtype=torch.uint8),)
    if workload in ("mxfp4_quant", "mxfp8_quant"):
        return (torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16),)
    if workload in ("rmsnorm_mxfp4", "rmsnorm_nvfp4", "rmsnorm_mxfp8"):
        return (
            torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16),
            torch.randn(hidden, device="cuda", dtype=torch.bfloat16),
        )
    if workload == "mxfp6_quant":
        return (torch.randn(rows, hidden, device="cuda", dtype=torch.float16),)
    if workload == "rmsnorm_mxfp6":
        return (
            torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16),
            torch.randn(hidden, device="cuda", dtype=torch.bfloat16),
        )
    if workload == "dcn_mxfp6":
        x = torch.randn(rows, hidden, device="cuda", dtype=torch.float16)
        return x, torch.randn_like(x)
    raise ValueError(f"unknown workload: {workload}")


def make_inductor_fn(workload: str, variant: str):
    if workload in (
        "swizzle_mxfp4",
        "mxfp4_quant",
        "rmsnorm_mxfp4",
        "rmsnorm_nvfp4",
        "mxfp8_quant",
        "rmsnorm_mxfp8",
    ):
        pad = BLOCKED_VARIANTS[variant]
    else:
        pad = XDL_VARIANTS[variant]
    if workload.startswith("swizzle"):
        return lambda scale: pad(scale)
    if workload == "mxfp4_quant":
        return lambda x: mxfp4_quantize(x, pad)
    if workload == "rmsnorm_mxfp4":
        return lambda x, weight: rmsnorm_mxfp4(x, weight, pad)
    if workload == "rmsnorm_nvfp4":
        return lambda x, weight: rmsnorm_nvfp4(x, weight, pad)
    if workload == "mxfp8_quant":
        return lambda x: mxfp8_quantize(x, pad)
    if workload == "rmsnorm_mxfp8":
        return lambda x, weight: rmsnorm_mxfp8(x, weight, pad)
    if workload == "mxfp6_quant":
        return lambda x: mxfp6_quantize(x, pad)
    if workload == "rmsnorm_mxfp6":
        return lambda x, weight: rmsnorm_mxfp6(x, weight, pad)
    if workload == "dcn_mxfp6":
        return lambda x, other: dcn_mxfp6(x, other, pad)
    raise ValueError(workload)


def scale_output(output: Any) -> torch.Tensor:
    return output if isinstance(output, torch.Tensor) else output[-1]


def padding_stats(workload: str, rows: int, hidden: int, output: Any):
    scale = scale_output(output).reshape(-1)
    logical_cols = hidden // (16 if workload == "rmsnorm_nvfp4" else MX_GROUP)
    if workload in (
        "swizzle_dcn",
        "mxfp6_quant",
        "rmsnorm_mxfp6",
        "dcn_mxfp6",
    ):
        logical = torch.ones((rows, logical_cols), dtype=torch.bool, device="cuda")
        valid = xdl_fpad(logical, False).bool()
        padding_value = DCN_PAD_VALUE
    else:
        logical = torch.ones((rows, logical_cols), dtype=torch.bool, device="cuda")
        valid = blocked_fpad(logical, False).bool()
        padding_value = 0
    padding = ~valid
    wrong = int((scale[padding] != padding_value).sum()) if bool(padding.any()) else 0
    return {
        "logical_scale_values": rows * logical_cols,
        "padded_scale_values": scale.numel(),
        "padding_values": int(padding.sum()),
        "padding_fraction": float(padding.float().mean()),
        "padding_wrong": wrong,
    }


def compile_inductor(fn: Callable, args: tuple[torch.Tensor, ...], tuning: str):
    torch._dynamo.reset()
    metrics.reset()
    module_start = len(PyCodeCache.modules)
    patches = {
        "triton.nested_reduction": True,
        "triton.cudagraphs": False,
        "fx_graph_cache": False,
        "emulate_precision_casts": True,
        "coordinate_descent_tuning": tuning == "coordesc",
    }
    if hasattr(config.triton, "enable_fuse_auxiliary_writes"):
        patches["triton.enable_fuse_auxiliary_writes"] = True
    with fresh_inductor_cache(), config.patch(patches):
        compiled = torch.compile(fn, fullgraph=True, dynamic=False)
        output, sources = run_and_get_code(compiled, *args)
        torch.cuda.synchronize()
    kernel_sources = [
        source
        for source in sources
        if "@triton_heuristics" in source and "@triton.jit" in source
    ]
    runtime_kernels = []
    for module in PyCodeCache.modules[module_start:]:
        for name, kernel in vars(module).items():
            if not name.startswith("triton_") or not hasattr(kernel, "launchers"):
                continue
            for launcher in kernel.launchers:
                launch_config = launcher.config
                runtime_kernels.append(
                    {
                        "name": name,
                        "config": dict(launch_config.kwargs),
                        "num_warps": launch_config.num_warps,
                        "num_stages": launch_config.num_stages,
                        "registers": launcher.n_regs,
                        "spills": launcher.n_spills,
                    }
                )
    metadata = {
        "kernel_count": metrics.generated_kernel_count,
        "nested_reduction_count": metrics.codegen_nested_reduction,
        "source_count": len(kernel_sources),
        "store_count": sum(source.count("tl.store(") for source in kernel_sources),
        "masked_store_count": sum(
            1
            for source in kernel_sources
            for line in source.splitlines()
            if "tl.store(" in line and not line.rstrip().endswith(", None)")
        ),
        "auxiliary_loop_count": sum(
            source.count("for auxiliary_") for source in kernel_sources
        ),
        "runtime_kernels": runtime_kernels,
    }
    return compiled, output, metadata


def flashinfer_case(workload: str, args: tuple[torch.Tensor, ...], rows: int, hidden: int):
    if workload == "mxfp4_quant":
        from flashinfer.quantization import mxfp4_quantize as fi_mxfp4_quantize
        from flashinfer.tllm_enums import SfLayout

        def fn():
            return fi_mxfp4_quantize(
                args[0], backend="cuda", enable_pdl=False, sfLayout=SfLayout.layout_128x4
            )

    elif workload in ("rmsnorm_mxfp4", "rmsnorm_nvfp4"):
        from flashinfer.cute_dsl import rmsnorm_fp4quant

        x, weight = args
        block = 16 if workload == "rmsnorm_nvfp4" else 32
        scale_format = "e4m3" if workload == "rmsnorm_nvfp4" else "ue8m0"
        scale_dtype = torch.float8_e4m3fn if workload == "rmsnorm_nvfp4" else torch.uint8
        quant = torch.empty(
            rows, hidden // 2, device="cuda", dtype=torch.float4_e2m1fn_x2
        )
        padded_rows, padded_cols = blocked_shape(rows, hidden // block)
        scale = torch.empty(padded_rows * padded_cols, device="cuda", dtype=scale_dtype)
        global_scale = torch.ones(1, device="cuda", dtype=torch.float32)

        def fn():
            return rmsnorm_fp4quant(
                x,
                weight,
                y_fp4=quant,
                block_scale=scale,
                global_scale=global_scale,
                eps=1e-6,
                block_size=block,
                scale_format=scale_format,
                is_sf_swizzled_layout=True,
                enable_pdl=False,
            )

    elif workload in ("mxfp8_quant", "rmsnorm_mxfp8"):
        from flashinfer.quantization import mxfp8_quantize as fi_mxfp8_quantize
        from flashinfer.tllm_enums import SfLayout

        if workload == "mxfp8_quant":
            def fn():
                return fi_mxfp8_quantize(
                    args[0],
                    sf_swizzle_layout=SfLayout.layout_128x4,
                    enable_pdl=False,
                )
        else:
            from flashinfer import norm

            x, weight = args
            normed = torch.empty_like(x)

            def fn():
                norm.rmsnorm(x, weight, 1e-6, out=normed, enable_pdl=False)
                return fi_mxfp8_quantize(
                    normed,
                    sf_swizzle_layout=SfLayout.layout_128x4,
                    enable_pdl=False,
                )

    else:
        raise ValueError(f"FlashInfer baseline unavailable for {workload}")
    output = fn()
    torch.cuda.synchronize()
    return fn, output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--tuning", choices=("default", "coordesc"), default="default")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--calls-per-graph", type=int, default=100)
    parser.add_argument("--cooldown", type=float, default=0.5)
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    inputs = make_inputs(args.workload, args.rows, args.hidden)
    started = time.time()
    if args.variant == "flashinfer":
        timed_fn, output = flashinfer_case(
            args.workload, inputs, args.rows, args.hidden
        )
        compile_metadata = {
            "kernel_count": 2 if args.workload == "rmsnorm_mxfp8" else 1,
            "nested_reduction_count": 0,
            "source_count": None,
            "store_count": None,
            "masked_store_count": None,
            "auxiliary_loop_count": None,
            "runtime_kernels": [],
        }
    else:
        fn = make_inductor_fn(args.workload, args.variant)
        compiled, output, compile_metadata = compile_inductor(
            fn, inputs, args.tuning
        )
        timed_fn = lambda: compiled(*inputs)

    time.sleep(args.cooldown)
    timing = graph_bench(
        timed_fn, args.warmup, args.samples, args.calls_per_graph
    )
    worktree = Path(torch.__file__).parents[1]
    worktree_status = subprocess.check_output(
        ["git", "-C", str(worktree), "status", "--porcelain=v1"], text=True
    )
    worktree_diff = subprocess.check_output(
        ["git", "-C", str(worktree), "diff", "HEAD", "--binary"]
    )
    result = {
        "workload": args.workload,
        "variant": args.variant,
        "shape": [args.rows, args.hidden],
        "tuning": args.tuning,
        "worktree_git": subprocess.check_output(
            ["git", "-C", str(worktree), "rev-parse", "HEAD"],
            text=True,
        ).strip(),
        "worktree_dirty": bool(worktree_status),
        "worktree_diff_sha256": hashlib.sha256(worktree_diff).hexdigest(),
        "torch_build_git": torch.version.git_version,
        "torch_file": torch.__file__,
        "gpu": torch.cuda.get_device_name(),
        "warmup": args.warmup,
        "samples": args.samples,
        "calls_per_graph": args.calls_per_graph,
        "elapsed_seconds": time.time() - started,
        "input_bytes": sum(x.numel() * x.element_size() for x in inputs),
        "outputs": output_metadata(output),
        "padding": padding_stats(
            args.workload, args.rows, args.hidden, output
        ),
        **compile_metadata,
        **timing,
    }
    if args.variant == "flashinfer":
        result["pdl"] = False
    print("PADDING_BENCH_RESULT=" + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
