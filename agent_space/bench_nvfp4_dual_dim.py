"""Combined dim0+dim1 NVFP4 kernel vs status quo.

Run through agent_space/run_wt.py with
PYTORCH_WORKTREE=/data/users/eellison/pytorch/agent_space/nvfp4_dual_dim_wt
and TORCHINDUCTOR_FORCE_DISABLE_CACHES=1.

Baselines (semantics from agent_space/bench_dim1_quant_casts.py):
  status_quo_rm  compiled nvfp4_dim0 (staged, nested=True) + compiled
                 nvfp4_dim1_rm run back-to-back (3 kernels, 3 reads of x).
  status_quo_t   same with nvfp4_dim1_t (transposed dim1 outputs).
  dual_graph_*   ONE compiled graph returning all four outputs (what Inductor
                 does today if you ask for both dims at once).
  combined_*     the handwritten one-load kernel (nvfp4_dual_dim_kernel.py).
  ao_dim0        torchao NVFP4Tensor.to_nvfp4 eager (no dedicated kernel
                 available: mslk extension not importable in this env).
Floor: one x read + all four outputs = 3.125 B/elem at 8 TB/s B200 HBM.
"""

import argparse
import importlib.util
import os
import re
import sys

import torch
import torch._inductor.config as inductor_config
from torch._inductor import metrics
from torch._inductor.utils import run_and_get_code
from triton.testing import do_bench

AGENT_SPACE = "/data/users/eellison/pytorch/agent_space"
sys.path.insert(0, AGENT_SPACE)

from nvfp4_dual_dim_kernel import nvfp4_dual_dim  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "bench_dim1", os.path.join(AGENT_SPACE, "bench_dim1_quant_casts.py")
)
bench_dim1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench_dim1)

FP8 = torch.float8_e4m3fn
CONFIGS = [
    (16, 128, 2),
    (16, 256, 2),
    (16, 256, 4),
    (16, 512, 4),
    (16, 512, 8),
    (16, 1024, 8),
    (32, 64, 2),
    (32, 128, 2),
    (32, 256, 4),
    (64, 128, 4),
]


def ideal_bytes(M, K):
    return M * K * 2 + 2 * (M * K // 2 + M * K // 16)


def bitwise_equal(a, b):
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if a.dtype == FP8:
        a, b = a.view(torch.uint8), b.view(torch.uint8)
    return torch.equal(a.contiguous(), b.contiguous())


def dual_ref(x):
    q0, s0 = bench_dim1.nvfp4_dim0(x)
    q1, s1 = bench_dim1.nvfp4_dim1_rm(x)
    return q0, s0, q1, s1


def dual_ref_t(x):
    q0, s0 = bench_dim1.nvfp4_dim0(x)
    q1, s1 = bench_dim1.nvfp4_dim1_t(x)
    return q0, s0, q1, s1


def bench_us(fn):
    return do_bench(fn, return_mode="median") * 1e3


def report(name, M, K, us, extra=""):
    tbs = ideal_bytes(M, K) / (us * 1e-6) / 1e12
    print(f"RESULT {name} {M}x{K} us={us:.2f} eff_TB/s={tbs:.2f} {extra}", flush=True)


def check_correctness(x, cd, configs):
    M, K = x.shape
    with inductor_config.patch({"triton.nested_reduction": True}):
        torch._dynamo.reset()
        d0 = torch.compile(bench_dim1.nvfp4_dim0, fullgraph=True)
        d1rm = torch.compile(bench_dim1.nvfp4_dim1_rm, fullgraph=True)
        d1t = torch.compile(bench_dim1.nvfp4_dim1_t, fullgraph=True)
        rq0, rs0 = d0(x)
        rq1, rs1 = d1rm(x)
        rq1t, rs1t = d1t(x)
    for bm, bk, w in configs:
        for precise in (True, False):
            q0, s0, q1, s1 = nvfp4_dual_dim(x, bm, bk, w, dim1_transposed=False, precise=precise)
            ok_rm = [bitwise_equal(a, b) for a, b in zip((q0, s0, q1, s1), (rq0, rs0, rq1, rs1))]
            q0, s0, q1, s1 = nvfp4_dual_dim(x, bm, bk, w, dim1_transposed=True, precise=precise)
            ok_t = [bitwise_equal(a, b) for a, b in zip((q0, s0, q1, s1), (rq0, rs0, rq1t, rs1t))]
            status = "OK" if all(ok_rm) and all(ok_t) else f"MISMATCH rm={ok_rm} t={ok_t}"
            print(f"CORRECTNESS {M}x{K} bm={bm} bk={bk} w={w} precise={int(precise)}: {status}", flush=True)
    # independent near-oracle: torchao pure-torch quantize (reciprocal-multiply
    # numerics, so report mismatch counts instead of asserting)
    try:
        from torchao.prototype.mx_formats.nvfp4_tensor import nvfp4_quantize

        q0, s0, q1t, s1t = nvfp4_dual_dim(x, dim1_transposed=True)
        aos0, aoq0 = nvfp4_quantize(x, 16)
        aos1, aoq1 = nvfp4_quantize(x.t().contiguous(), 16)
        pm0 = (q0 != aoq0).sum().item()
        sm0 = (s0.view(torch.uint8) != aos0.view(torch.uint8)).sum().item()
        pm1 = (q1t != aoq1).sum().item()
        sm1 = (s1t.view(torch.uint8) != aos1.view(torch.uint8)).sum().item()
        print(
            f"AO_DIFF {M}x{K}: dim0 payload {pm0}/{q0.numel()} scale {sm0}/{s0.numel()}"
            f" | dim1t payload {pm1}/{q1t.numel()} scale {sm1}/{s1t.numel()}",
            flush=True,
        )
    except Exception as e:
        print(f"AO_DIFF {M}x{K}: unavailable ({type(e).__name__}: {e})", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shapes", default="1024x1024,4096x4096,16384x7168")
    p.add_argument("--edge-shapes", default="528x1040,2064x4112")
    p.add_argument("--cd", action="store_true")
    p.add_argument("--skip-correctness", action="store_true")
    p.add_argument("--skip-status-quo", action="store_true")
    p.add_argument("--sweep", action="store_true", help="print every config, not just best")
    p.add_argument("--dump-dir", default=os.path.join(AGENT_SPACE, "nvfp4_dual_dumps"))
    args = p.parse_args()
    os.makedirs(args.dump_dir, exist_ok=True)
    torch.manual_seed(0)

    if not args.skip_correctness:
        for shape in args.edge_shapes.split(","):
            if not shape:
                continue
            M, K = map(int, shape.split("x"))
            x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
            check_correctness(x, args.cd, [(32, 128, 4), (16, 256, 4)])

    for shape in args.shapes.split(","):
        M, K = map(int, shape.split("x"))
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        floor_us = ideal_bytes(M, K) / 8e12 * 1e6
        print(f"FLOOR {M}x{K} bytes={ideal_bytes(M, K)} us={floor_us:.2f} (8 TB/s)", flush=True)

        if not args.skip_correctness:
            check_correctness(x, args.cd, [(32, 128, 4)])

        # combined kernel sweep
        for transposed in (False, True):
            for precise in (True, False):
                tag = "combined_t" if transposed else "combined_rm"
                tag += "" if precise else "_fast"
                best = None
                for bm, bk, w in CONFIGS:
                    us = bench_us(lambda: nvfp4_dual_dim(x, bm, bk, w, dim1_transposed=transposed, precise=precise))
                    if args.sweep:
                        report(f"{tag}[{bm}x{bk}w{w}]", M, K, us)
                    if best is None or us < best[0]:
                        best = (us, bm, bk, w)
                us, bm, bk, w = best
                report(tag, M, K, us, extra=f"best_config={bm}x{bk}w{w}")

        if args.skip_status_quo:
            continue

        # status quo: separate compiled dim0 + dim1 graphs
        torch._dynamo.reset()
        metrics.reset()
        with inductor_config.patch(
            {"triton.nested_reduction": True, **({"coordinate_descent_tuning": True} if args.cd else {})}
        ):
            d0 = torch.compile(bench_dim1.nvfp4_dim0, fullgraph=True)
            d0(x)
            staged = metrics.codegen_nested_reduction
            d1rm = torch.compile(bench_dim1.nvfp4_dim1_rm, fullgraph=True)
            d1t = torch.compile(bench_dim1.nvfp4_dim1_t, fullgraph=True)
            d1rm(x)
            d1t(x)
            us0 = bench_us(lambda: d0(x))
            report("inductor_dim0", M, K, us0, extra=f"staged={staged}")
            us1rm = bench_us(lambda: d1rm(x))
            report("inductor_dim1_rm", M, K, us1rm)
            us1t = bench_us(lambda: d1t(x))
            report("inductor_dim1_t", M, K, us1t)
            us = bench_us(lambda: (d0(x), d1rm(x)))
            report("status_quo_rm", M, K, us)
            us = bench_us(lambda: (d0(x), d1t(x)))
            report("status_quo_t", M, K, us)

            # single compiled graph producing all four outputs
            for name, fn in (("dual_graph_rm", dual_ref), ("dual_graph_t", dual_ref_t)):
                torch._dynamo.reset()
                metrics.reset()
                cfn = torch.compile(fn, fullgraph=True)
                _, codes = run_and_get_code(cfn, x)
                kernels = [k for c in codes for k in re.findall(r"def (triton_\w+)\(", c)]
                with open(os.path.join(args.dump_dir, f"{name}_{M}x{K}.py"), "w") as f:
                    f.write("\n\n".join(codes))
                us = bench_us(lambda: cfn(x))
                report(name, M, K, us, extra=f"kernels={len(kernels)}")

        # torchao eager rowwise (no dedicated NVFP4 kernel available)
        try:
            from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

            NVFP4Tensor.to_nvfp4(x)
            us = bench_us(lambda: NVFP4Tensor.to_nvfp4(x))
            report("ao_dim0_eager", M, K, us)
        except Exception as e:
            print(f"ao_dim0_eager {M}x{K}: unavailable ({e})", flush=True)


if __name__ == "__main__":
    main()
