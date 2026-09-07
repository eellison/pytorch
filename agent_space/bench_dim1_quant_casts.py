"""Dim1 (columnwise) MXFP8 / NVFP4 quantization casts under Inductor.

Run through agent_space/run_wt.py with PYTORCH_WORKTREE=/tmp/dim1_wt.

Workloads (x is [M, K] bf16, row-major, cuda):
  mxfp8_dim0     rowwise control: 1x32 groups along K, E8M0 scales, fp8 payload
                 row-major [M, K], scale [M, K//32].
  mxfp8_dim1_t   columnwise, torchao layout: 32x1 groups along M, payload
                 stored as [K, M] contiguous (returned as [M, K] col-major
                 view), scale [K, M//32]. Mirrors to_mx_dim1_reference.
  mxfp8_dim1_rm  columnwise, natural layout: 32x1 groups along M, payload
                 row-major [M, K], scale [M//32, K].
  nvfp4_dim0     rowwise control (the stack flagship): 16 groups along K,
                 fp8 e4m3 scales, adjacent COLUMN pairs packed to one byte,
                 payload [M, K//2], scale [M, K//16].
  nvfp4_dim1_rm  columnwise: 16 groups along M, adjacent ROW pairs packed to
                 one byte, payload row-major [M//2, K], scale [M//16, K].
  nvfp4_dim1_t   same but payload stored [K, M//2] contiguous and scale
                 [K, M//16] contiguous (transposed layout for a GEMM consumer).
  ao_mxfp8_dim1  torchao dedicated Triton dim1 kernel (baseline, no compile).
  ao_mxfp8_dim0  torchao dedicated Triton dim0 kernel (baseline, no compile).
"""

import argparse
import os
import re

import torch
import torch._inductor.config as inductor_config
from torch._higher_order_ops.inline_asm_elementwise import inline_asm_elementwise
from torch._inductor import metrics
from torch._inductor.utils import run_and_get_code
from triton.testing import do_bench

from torchao.prototype.mx_formats.config import ScaleCalculationMode
from torchao.prototype.mx_formats.mx_tensor import to_mx


E2M1X2_PACK_ASM = (
    "{.reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, $2, $1; cvt.u32.u8 $0, t;}"
)
RCEIL = ScaleCalculationMode.RCEIL
FP8 = torch.float8_e4m3fn


def mxfp8_dim0(x):
    scale, data = to_mx(x, FP8, 32, scaling_mode=RCEIL)
    return data, scale


def mxfp8_dim1_t(x):
    xt = x.t().contiguous()
    scale, data = to_mx(xt, FP8, 32, scaling_mode=RCEIL)
    return data.t(), scale


def mxfp8_dim1_rm(x):
    M, K = x.shape
    xg = x.view(M // 32, 32, K).permute(0, 2, 1).reshape(-1, 32)
    scale, data = to_mx(xg, FP8, 32, scaling_mode=RCEIL)
    data = data.view(M // 32, K, 32).permute(0, 2, 1).reshape(M, K)
    return data, scale.view(M // 32, K)


def nvfp4_dim0(x):
    M, K = x.shape
    xg = x.view(M, K // 16, 16)
    amax = xg.float().abs().amax(dim=-1)
    scale = (amax / 6.0).clamp(min=1e-12, max=448.0).to(FP8)
    xp = xg.view(M, K // 16, 8, 2)
    scale_f = scale.float().unsqueeze(-1)
    even = xp[..., 0].float() / scale_f
    odd = xp[..., 1].float() / scale_f
    packed = inline_asm_elementwise(
        even, odd, asm_str=E2M1X2_PACK_ASM, constraints="=r,f,f",
        dtype=torch.int32, is_pure=True, pack=1,
    )
    return packed.to(torch.uint8).view(M, K // 2), scale


def _nvfp4_dim1(x):
    M, K = x.shape
    xg = x.view(M // 16, 16, K)
    amax = xg.float().abs().amax(dim=1)
    scale = (amax / 6.0).clamp(min=1e-12, max=448.0).to(FP8)
    xp = x.view(M // 16, 8, 2, K)
    scale_f = scale.float().unsqueeze(1)
    even = xp[:, :, 0, :].float() / scale_f
    odd = xp[:, :, 1, :].float() / scale_f
    packed = inline_asm_elementwise(
        even, odd, asm_str=E2M1X2_PACK_ASM, constraints="=r,f,f",
        dtype=torch.int32, is_pure=True, pack=1,
    )
    return packed.to(torch.uint8).view(M // 2, K), scale


def nvfp4_dim1_rm(x):
    return _nvfp4_dim1(x)


def nvfp4_dim1_t(x):
    payload, scale = _nvfp4_dim1(x)
    return payload.t().contiguous(), scale.t().contiguous()


def ao_mxfp8_dim1(x):
    from torchao.prototype.mx_formats.kernels import triton_to_mxfp8_dim1

    return triton_to_mxfp8_dim1(x, 32, "rceil")


def ao_mxfp8_dim0(x):
    from torchao.prototype.mx_formats.kernels import triton_to_mxfp8_dim0

    return triton_to_mxfp8_dim0(x, 32, "rceil")


WORKLOADS = {
    "mxfp8_dim0": mxfp8_dim0,
    "mxfp8_dim1_t": mxfp8_dim1_t,
    "mxfp8_dim1_rm": mxfp8_dim1_rm,
    "nvfp4_dim0": nvfp4_dim0,
    "nvfp4_dim1_rm": nvfp4_dim1_rm,
    "nvfp4_dim1_t": nvfp4_dim1_t,
}


def ideal_bytes(name, M, K):
    if "mxfp8" in name:
        return M * K * 2 + M * K + M * K // 32
    return M * K * 2 + M * K // 2 + M * K // 16


def bitwise_equal(a, b):
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if a.dtype in (FP8, torch.float8_e8m0fnu):
        a, b = a.view(torch.uint8), b.view(torch.uint8)
    return torch.equal(a.contiguous(), b.contiguous())


def kernel_profile(fn, iters=20):
    from torch.profiler import profile, ProfilerActivity

    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    rows = []
    for evt in prof.key_averages():
        if evt.self_device_time_total > 0 and "memcpy" not in evt.key.lower():
            rows.append((evt.key, evt.self_device_time_total / iters, evt.count // iters))
    rows.sort(key=lambda r: -r[1])
    return rows


def run_compiled_case(name, fn, x, nested, dump_dir, cd, do_profile):
    torch._dynamo.reset()
    metrics.reset()
    M, K = x.shape
    patch = {"triton.nested_reduction": nested}
    if cd:
        patch["coordinate_descent_tuning"] = True
    with inductor_config.patch(patch):
        cfn = torch.compile(fn, fullgraph=True)
        result, codes = run_and_get_code(cfn, x)
        kernels = [k for code in codes for k in re.findall(r"def (triton_\w+)\(", code)]
        staged = metrics.codegen_nested_reduction
        tag = f"{name}_{M}x{K}_nested{int(nested)}" + ("_cd" if cd else "")
        with open(os.path.join(dump_dir, tag + ".py"), "w") as f:
            f.write("\n\n".join(codes))
        us = do_bench(lambda: cfn(x), return_mode="median") * 1e3
        gbs = ideal_bytes(name, M, K) / (us * 1e-6) / 1e9
        prof_rows = kernel_profile(lambda: cfn(x)) if do_profile else []
    print(
        f"RESULT {name} {M}x{K} nested={int(nested)} cd={int(cd)} "
        f"kernels={len(kernels)} staged={staged} us={us:.2f} eff_TB/s={gbs / 1e3:.2f}",
        flush=True,
    )
    for kname, t_us, cnt in prof_rows:
        print(f"  PROF {kname[:100]} {t_us:.1f}us x{cnt}", flush=True)
    return result


def run_ao_case(name, x):
    fn = {"ao_mxfp8_dim1": ao_mxfp8_dim1, "ao_mxfp8_dim0": ao_mxfp8_dim0}[name]
    M, K = x.shape
    out = fn(x)
    us = do_bench(lambda: fn(x), return_mode="median") * 1e3
    gbs = ideal_bytes(name, M, K) / (us * 1e-6) / 1e9
    print(
        f"RESULT {name} {M}x{K} kernels=1 us={us:.2f} eff_TB/s={gbs / 1e3:.2f}",
        flush=True,
    )
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workload", required=True)
    p.add_argument("--shapes", default="1024x1024,4096x4096,16384x7168")
    p.add_argument("--dump-dir", default="/tmp/dim1_wt/agent_space/dim1_dumps")
    p.add_argument("--cd", action="store_true")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--compare-ao", action="store_true")
    args = p.parse_args()
    os.makedirs(args.dump_dir, exist_ok=True)
    torch.manual_seed(0)

    for shape in args.shapes.split(","):
        M, K = map(int, shape.split("x"))
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        if args.workload.startswith("ao_"):
            run_ao_case(args.workload, x)
            continue
        fn = WORKLOADS[args.workload]
        big = M * K >= 4096 * 4096
        r_off = run_compiled_case(
            args.workload, fn, x, False, args.dump_dir, args.cd, args.profile and big
        )
        r_on = run_compiled_case(
            args.workload, fn, x, True, args.dump_dir, args.cd, args.profile and big
        )
        match = all(bitwise_equal(a, b) for a, b in zip(r_off, r_on))
        print(f"NESTED_MATCH {args.workload} {M}x{K}: {match}", flush=True)
        if args.compare_ao and args.workload == "mxfp8_dim1_t":
            ao_payload, ao_scale = ao_mxfp8_dim1(x)
            payload, scale = r_on
            pm = (payload.view(torch.uint8) != ao_payload.view(torch.uint8)).sum().item()
            sm = (scale.view(torch.uint8) != ao_scale.view(torch.uint8).reshape(scale.shape)).sum().item()
            print(f"AO_DIFF {M}x{K}: payload_mismatch={pm}/{M * K} scale_mismatch={sm}/{K * M // 32}", flush=True)


if __name__ == "__main__":
    main()
