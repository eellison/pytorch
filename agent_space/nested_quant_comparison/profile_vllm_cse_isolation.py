import torch

import generated_vllm_gap_coordesc as mixed
import generated_vllm_gap_default as original
from torch._C import _cuda_getCurrentRawStream
from vllm.kernels.helion.ops.rms_norm_dynamic_per_token_quant import (
    rms_norm_dynamic_per_token_quant,
)


torch.manual_seed(0)
rows = hidden = 4096
x = torch.zeros(rows, hidden, device="cuda", dtype=torch.bfloat16)
residual = torch.randn_like(x)
weight = torch.randn(hidden, device="cuda", dtype=torch.bfloat16)
scale = torch.empty(rows, 1, device="cuda", dtype=torch.float32)
quant = torch.empty_like(x, dtype=torch.float8_e4m3fn)
summed = torch.empty_like(x)
vllm_residual = residual.clone()
kernel_name = (
    "triton_red_fused__to_copy_abs_add_amax_clamp_clamp_min_div_"
    "mean_mul_pow_rsqrt_0"
)


def run(module) -> None:
    getattr(module, kernel_name).run(
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


run(original)
run(mixed)
rms_norm_dynamic_per_token_quant(
    quant, x, weight, scale, 1e-6, residual=vllm_residual
)
torch.cuda.synchronize()

for name, fn in (
    ("inductor_original", lambda: run(original)),
    ("inductor_mixed", lambda: run(mixed)),
    (
        "vllm_helion",
        lambda: rms_norm_dynamic_per_token_quant(
            quant, x, weight, scale, 1e-6, residual=vllm_residual
        ),
    ),
):
    torch.cuda.nvtx.range_push(name)
    fn()
    torch.cuda.nvtx.range_pop()
torch.cuda.synchronize()
