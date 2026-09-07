"""Probe: can rmsnorm+quant+padded-swizzle be a single kernel?

Idea: run the reduction over padded_rows (pad rows take masked/zero loads), store
0 into the scale for pad rows, and predicate off the packed store so the packed
buffer stays rows x hidden/2. That folds the pad write into the epilogue and
removes the second launch.

Variant A (`pad_input`): pad x itself and slice the packed result -- checks
whether Inductor keeps it to one kernel or reintroduces a copy.
Variant B (`pad_scale_domain`): keep x unpadded, scatter the valid scales and
let the scale buffer's pad lanes be written by the same epilogue.
"""
from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from bench_main_cuda import FP4_MAX, FP8_MAX, PACK_E2M1X2_ASM, compile_main, graph_bench, recip_ue8m0
from blocked_variants import _padded_shape, _offsets, VARIANTS

from torch._higher_order_ops.inline_asm_elementwise import inline_asm_elementwise
from torch._inductor import inductor_prims


def _quantize(normed, rows, hidden, block, scale_format):
    normed = normed.view(rows, hidden // block, block)
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
        asm_str=PACK_E2M1X2_ASM, constraints="=r,f,f", dtype=torch.int32,
        is_pure=True, pack=1,
    ).to(torch.uint8).view(rows, hidden // 2)
    return packed, scale


def baseline(x, weight, block, scale_format):
    rows, hidden = x.shape
    normed = F.rms_norm(x, (hidden,), weight)
    packed, scale = _quantize(normed, rows, hidden, block, scale_format)
    return packed, VARIANTS["predicated"](scale)


def pad_input(x, weight, block, scale_format):
    """Reduction runs over padded_rows; packed is sliced back to rows."""
    rows, hidden = x.shape
    cols = hidden // block
    padded_rows, padded_cols = _padded_shape(rows, cols)
    xp = F.pad(x, (0, 0, 0, padded_rows - rows))
    normed = F.rms_norm(xp, (hidden,), weight)
    packed, scale = _quantize(normed, padded_rows, hidden, block, scale_format)
    if padded_cols != cols:
        scale = F.pad(scale, (0, padded_cols - cols))
    # pad rows produced garbage scales; zero them before the swizzle
    keep = torch.arange(padded_rows, device=x.device)[:, None] < rows
    if padded_cols != cols:
        keep = keep & (torch.arange(padded_cols, device=x.device)[None, :] < cols)
    scale = torch.where(keep, scale, torch.zeros((), dtype=scale.dtype, device=x.device))
    offset = _offsets(torch.arange(padded_rows, device=x.device),
                      torch.arange(padded_cols, device=x.device), padded_cols)
    out = torch.empty(padded_rows * padded_cols, dtype=scale.dtype, device=x.device)
    out = torch.ops.aten._unsafe_index_put.default(out, [offset], scale, False)
    return packed[:rows], out


IMPLS = {"baseline": baseline, "pad_input": pad_input}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--impl", required=True)
    p.add_argument("--shapes", default="129x4096,989x4096")
    p.add_argument("--suites", default="nvfp4,mxfp4")
    p.add_argument("--no-verify", dest="verify", action="store_false")
    p.add_argument("--coordinate-descent", action="store_true")
    args = p.parse_args()
    fn = IMPLS[args.impl]
    for spec in args.shapes.split(","):
        rows, hidden = map(int, spec.split("x"))
        for name in args.suites.split(","):
            block, fmt = (16, "e4m3") if name == "nvfp4" else (32, "ue8m0")
            x = torch.randn(rows, hidden, dtype=torch.bfloat16, device="cuda")
            w = torch.randn(hidden, dtype=torch.bfloat16, device="cuda")
            compiled, out, meta = compile_main(
                lambda a, b: fn(a, b, block, fmt), x, w,
                coordinate_descent=args.coordinate_descent)
            if args.verify:
                ref_c, ref_o, _ = compile_main(
                    lambda a, b: baseline(a, b, block, fmt), x, w,
                    coordinate_descent=args.coordinate_descent)
                ok_q = torch.equal(out[0], ref_o[0])
                ok_s = torch.equal(out[1].view(torch.uint8), ref_o[1].view(torch.uint8))
            else:
                ok_q = ok_s = None
            t = graph_bench(lambda: compiled(x, w), warmup=20, samples=50, calls_per_graph=100)
            print(f"{name:<6}{rows}x{hidden:<6} {args.impl:<12} {t['median_us']:7.2f}us "
                  f"kernels={meta.get('kernel_count')}  quant_ok={ok_q} scale_ok={ok_s}", flush=True)
            torch._dynamo.reset()


if __name__ == "__main__":
    main()
