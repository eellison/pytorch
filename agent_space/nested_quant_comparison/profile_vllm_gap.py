import torch

import torch._inductor.config as inductor_config
from vllm.kernels.helion.ops.rms_norm_dynamic_per_token_quant import (
    rms_norm_dynamic_per_token_quant,
)

from bench_vllm_helion_main import fused_add_dynamic_rmsnorm_fp8


torch.manual_seed(0)
rows, hidden = 4096, 4096
x = torch.zeros(rows, hidden, device="cuda", dtype=torch.bfloat16)
residual = torch.randn_like(x)
weight = torch.randn(hidden, device="cuda", dtype=torch.bfloat16)
compiled_variants = {}
for name, coordinate_descent in (("default", False), ("coordesc", True)):
    torch._dynamo.reset()
    with inductor_config.patch(
        {
            "triton.nested_reduction": True,
            "triton.cudagraphs": False,
            "fx_graph_cache": False,
            "emulate_precision_casts": True,
            "coordinate_descent_tuning": coordinate_descent,
        }
    ):
        compiled = torch.compile(
            fused_add_dynamic_rmsnorm_fp8, fullgraph=True, dynamic=False
        )
        compiled(x, residual, weight)
    compiled_variants[name] = compiled

vllm_quant = torch.empty_like(x, dtype=torch.float8_e4m3fn)
vllm_scale = torch.empty(rows, 1, device="cuda", dtype=torch.float32)
vllm_residual = residual.clone()
rms_norm_dynamic_per_token_quant(
    vllm_quant,
    x,
    weight,
    vllm_scale,
    1e-6,
    residual=vllm_residual,
)
torch.cuda.synchronize()

for name, compiled in compiled_variants.items():
    torch.cuda.nvtx.range_push(f"inductor_main_{name}")
    compiled(x, residual, weight)
    torch.cuda.nvtx.range_pop()
torch.cuda.nvtx.range_push("vllm_main_helion")
rms_norm_dynamic_per_token_quant(
    vllm_quant,
    x,
    weight,
    vllm_scale,
    1e-6,
    residual=vllm_residual,
)
torch.cuda.nvtx.range_pop()
torch.cuda.synchronize()
