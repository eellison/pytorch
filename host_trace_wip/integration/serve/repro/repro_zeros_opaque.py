# Candidate repro for R1-V's decline "sNNNN is not read from the call's inputs" (symbol source allocK.base/256, created
# by an allocation inside the tape's composite redispatch of a factory such as aten.zeros, _host_trace_tape._eager_call
# `func in _ZEROS` branch). Cases, each a HostTraceReplay called at 3 sizes, printing traces / replays / declines and a
# bitwise check against eager:
#   zeros_mm      torch.zeros((m, k)) as an operand of a harvested cuBLAS mm (opaque call)
#   zeros_triton  torch.zeros((round_up(m, 128), 4), int32) written by a Triton kernel (traced launch)
#   zeros_ret     torch.zeros((round_up(m, 128), 4), int32) filled pointwise and returned
#   reshape_copy_triton  reshape of a non-contiguous split view (copy) fed to a Triton kernel (vLLM's _rms_norm_gated_cuda)
#   zeros_scale_into_mm  zeros((round_up(m,128),4), int32) written by Triton next to a harvested mm (create_fp4_scale_tensor)
#   empty_mm      control: torch.empty + fill_ as the mm operand
# Run: CUDA_VISIBLE_DEVICES=1 bash serve/python_vllm_cand3.sh serve/repro/repro_zeros_opaque.py
import torch
import torch.cuda._host_trace_replay as R
import triton
import triton.language as tl
from torch.cuda._host_trace_harvest import HarvestProvider


@triton.jit
def _fill_rows(out, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(out + i * 4, i.to(tl.int32), mask=i < n)


def round_up(m, a):
    return (m + a - 1) // a * a


def zeros_mm(x, w):
    a = torch.zeros((x.shape[0], w.shape[0]), device=x.device, dtype=x.dtype)
    a += x[:, : w.shape[0]]
    return (torch.mm(a, w),)


def zeros_triton(x, w):
    m = x.shape[0]
    s = torch.zeros((round_up(m, 128), 4), device=x.device, dtype=torch.int32)
    _fill_rows[(triton.cdiv(m, 64),)](s, m, BLOCK=64)
    return (s, x * 2)


def zeros_ret(x, w):
    s = torch.zeros((round_up(x.shape[0], 128), 4), device=x.device, dtype=torch.int32)
    s += 1
    return (s,)


@triton.jit
def _scale_rows(x, z, out, n, D: tl.constexpr):
    r = tl.program_id(0)
    c = tl.arange(0, D)
    tl.store(out + r * D + c, tl.load(x + r * D + c) * tl.load(z + r * D + c), mask=r < n)


def reshape_copy_triton(x, w):
    # vLLM qwen_gdn_linear_attn._rms_norm_gated_cuda: a reshape of a non-contiguous split view (a copy), fed to Triton
    m = x.shape[0]
    a, g = x.split([256, 256], dim=-1)
    g3 = g.view(m, 2, 128)
    g2 = g3.reshape(-1, 128)  # contiguous rows of a strided view: reshape copies
    a2 = a.reshape(-1, 128)
    out = torch.empty_like(a2)
    _scale_rows[(2 * m,)](a2, g2, out, 2 * m, D=128)
    return (out,)


def zeros_scale_into_mm(x, w):
    # vLLM create_fp4_scale_tensor: torch.zeros((round_up(m, 128), n // 64), int32) written by a kernel, read by the GEMM
    m = x.shape[0]
    s = torch.zeros((round_up(m, 128), 4), device=x.device, dtype=torch.int32)
    _fill_rows[(triton.cdiv(m, 64),)](s, m, BLOCK=64)
    y = torch.mm(x[:, :256], w)
    return (y + s[:m, :1].to(y.dtype),)


def empty_mm(x, w):
    a = torch.empty((x.shape[0], w.shape[0]), device=x.device, dtype=x.dtype)
    a.fill_(0)
    a += x[:, : w.shape[0]]
    return (torch.mm(a, w),)


torch.manual_seed(0)
w = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16)
for fn in (zeros_mm, zeros_triton, zeros_ret, reshape_copy_triton, zeros_scale_into_mm, empty_mm):
    entry = R.HostTraceReplay(fn, opaque=(HarvestProvider(("blas",)),), memory="run_buffer")
    ok = []
    for m in (7, 33, 200, 33, 7):
        x = torch.randn(m, 512, device="cuda", dtype=torch.bfloat16)
        got = entry(x, w)
        want = fn(x, w)
        ok.append(all(torch.equal(g, h) for g, h in zip(got, want)))
    print(fn.__name__, "traces", entry.traces, "replays", entry.replays, "eager", entry.eager, "variants", len(entry.variants),
          "bitwise", ok, "declines", sorted({str(d)[:200] for d in entry.declines})[:3],
          "retrace_causes", dict(list(getattr(entry, "retrace_causes", {}).items())[:4]), flush=True)
