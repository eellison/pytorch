import torch

import torch._inductor.config as inductor_config
from flashinfer import norm

from bench_main_cuda import fused_add_block_fp8


torch.manual_seed(0)
rows, hidden = 128, 4096
x = torch.zeros(rows, hidden, device="cuda", dtype=torch.bfloat16)
residual = torch.randn_like(x) * 0.1
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
        compiled = torch.compile(fused_add_block_fp8, fullgraph=True, dynamic=False)
        compiled(x, residual, weight)
    compiled_variants[name] = compiled

fi_quant = torch.empty_like(x, dtype=torch.float8_e4m3fn)
fi_scale = torch.empty(hidden // 128, rows, device="cuda", dtype=torch.float32)
fi_normed = torch.empty_like(x)
fi_residual = residual.clone()
norm.fused_add_rmsnorm_fp8_block_quant(
    fi_quant,
    fi_scale,
    fi_normed,
    x,
    fi_residual,
    weight,
    1e-6,
    enable_pdl=False,
)
torch.cuda.synchronize()

for name, compiled in compiled_variants.items():
    torch.cuda.nvtx.range_push(f"inductor_main_{name}")
    compiled(x, residual, weight)
    torch.cuda.nvtx.range_pop()
torch.cuda.nvtx.range_push("flashinfer_main_pdl_false")
norm.fused_add_rmsnorm_fp8_block_quant(
    fi_quant,
    fi_scale,
    fi_normed,
    x,
    fi_residual,
    weight,
    1e-6,
    enable_pdl=False,
)
torch.cuda.nvtx.range_pop()
torch.cuda.synchronize()
