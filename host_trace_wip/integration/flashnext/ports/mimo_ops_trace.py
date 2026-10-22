# Python port of the host launcher of vLLM's _C.per_token_group_fp8_quant_packed (csrc/libtorch_stable/quantization/w8a8/fp8/
# per_token_group_quant.cu, per_token_group_quant_8bit_packed): DeepGEMM's activation quantization on Blackwell (fp8 values,
# UE8M0 scales packed 4 per int32, TMA-aligned column-major scale layout), which MiMo-V2.6-Flash's FP8 block linears call
# before each DeepGEMM GEMM. Same form as the shared ports (integration/vllm/vllm_ports/vllm_ops_trace.py, whose _Params
# record-and-check is used as is): the launch is recorded with its parameters as expressions of the trace's values and
# checked byte for byte against the real op's launch captured on stand-ins; any difference declines.
from __future__ import annotations

import functools
import math
import os
import sys
from typing import Any

import torch

sys.path.insert(0, os.environ.get("VLLM_PORTS_DIR", "/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration/vllm/vllm_ports"))
import vllm_ops_trace as V  # noqa: E402

_KERNEL = ("void per_token_group_quant_8bit_packed_register_kernel<{t}, {d}, 128, {kx}, {ry}>({t} const*, void*, unsigned int*, "
           "int, int, int, int, int, long, float, float, float)")
_CXX = {torch.bfloat16: "c10::BFloat16", torch.float16: "c10::Half", torch.float8_e4m3fn: "c10::Float8_e4m3fn", torch.int8: "signed char"}


@functools.cache
def kernel(demangled: str) -> tuple[int, int, str]:
    """V.kernel for a kernel outside namespace vllm (this one is global): the static __global__ stub from the .symtab."""
    import ctypes

    from cuda.bindings import driver

    from torch.cuda._utils import _check_cuda_bindings

    path = V._loaded(V.MODULE)
    symbols = V._symtab(path)
    mangled = [n for n in symbols if n.startswith("_Z") and "per_token_group_quant_8bit_packed_register_kernel" in n and V._demangle(n) == demangled]
    if len(mangled) != 1:
        raise LookupError(f"{demangled} in {path}: {mangled}")
    lib = ctypes.CDLL(path, mode=os.RTLD_NOLOAD | os.RTLD_LAZY)
    stub = ctypes.cast(getattr(lib, V._EXPORT), ctypes.c_void_p).value - symbols[V._EXPORT] + symbols[mangled[0]]
    with open("/proc/self/maps") as maps:
        copies = sorted({line.split()[-1] for line in maps if "/libcudart" in line})
    for copy in copies:
        func = ctypes.c_void_p()
        if ctypes.CDLL(copy, mode=os.RTLD_NOLOAD).cudaGetFuncBySymbol(ctypes.byref(func), ctypes.c_void_p(stub)) == 0:
            break
    else:
        raise LookupError(f"cudaGetFuncBySymbol({mangled[0]}) in {copies}")
    return func.value, stub, _check_cuda_bindings(driver.cuFuncGetName(func.value)).decode()


def _groups_per_block_x(padded: int) -> int:
    return 16 if padded % 16 == 0 else 8 if padded % 8 == 0 else 4


# per_token_group_fp8_quant_packed(Tensor input, Tensor! output_q, Tensor! output_s_packed, int group_size, float eps,
# float fp8_min, float fp8_max) -> ()
def per_token_group_fp8_quant_packed(*args: Any, **kwargs: Any) -> None:
    from torch.cuda._host_trace import select

    op = torch.ops._C.per_token_group_fp8_quant_packed.default
    c = V._Params(op, args, kwargs)
    a = V._bind(op, args, kwargs)
    x, q, s = a["input"], a["output_q"], a["output_s_packed"]
    c.require(a["group_size"] == 128, "group_size 128")
    c.require(x.dtype in (torch.bfloat16, torch.float16), "bf16 / fp16 input")
    c.require(q.dtype in (torch.float8_e4m3fn, torch.int8), "an fp8 e4m3 (or int8) output")
    c.require(V._dense(x) and V._dense(q), "contiguous input and output_q")
    k = int(x.shape[-1])  # the kernel instantiation follows it: a guard
    c.require(k % 128 == 0, "k a multiple of the group size")
    mn = math.prod(x.shape) // k
    groups_per_row = k // 128
    k_num_packed_sfk = (groups_per_row + 3) // 4
    tma_aligned_mn = (mn + 3) // 4 * 4
    q_rows = math.prod(q.shape) // k
    c.require(q_rows >= mn, "output_q has at least mn rows")
    extent = select(q_rows < tma_aligned_mn, q_rows, tma_aligned_mn) if isinstance(q_rows, torch.SymInt) else min(q_rows, tma_aligned_mn)
    c.require(s.dtype == torch.int32, "int32 packed scales")
    c.require(s.dim() == 2 and s.shape[0] == mn and s.shape[1] == k_num_packed_sfk, "scales [mn, ceil(groups / 4)]")
    c.require(s.stride(0) == 1 and s.stride(1) == tma_aligned_mn, "scale strides [1, tma_aligned_mn]")
    padded = k_num_packed_sfk * 4
    num_scale_elems = mn + (k_num_packed_sfk - 1) * tma_aligned_mn
    kx = _groups_per_block_x(padded)
    ry = 16 // kx
    row_blocks = (tma_aligned_mn + ry - 1) // ry
    sf_k_blocks = padded // kx
    fn = kernel(_KERNEL.format(t=_CXX[x.dtype], d=_CXX[q.dtype], kx=kx, ry=ry))
    params = [("P", x.data_ptr()), ("P", q.data_ptr()), ("P", s.data_ptr()), ("i", padded), ("i", groups_per_row), ("i", mn),
              ("i", extent), ("i", tma_aligned_mn), ("q", num_scale_elems), ("f", a["eps"]), ("f", a["fp8_min"]), ("f", a["fp8_max"])]
    c.launch(fn, params, (row_blocks, sf_k_blocks, 1), (kx * ry * 8, 1, 1), pdl=True)


def install() -> list[Any]:
    import torch.cuda._host_trace as ht

    o = torch.ops._C.per_token_group_fp8_quant_packed.default
    ht.register_traced_impl(o, V._loud(o, per_token_group_fp8_quant_packed))
    ht.traced_impls = True
    return [o]
