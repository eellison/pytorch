from pathlib import Path

import torch

import torch._inductor.config as inductor_config
from torch._inductor.utils import run_and_get_code

from bench_vllm_helion_main import fused_add_dynamic_rmsnorm_fp8


torch.manual_seed(0)
rows, hidden = 4096, 4096
x = torch.zeros(rows, hidden, device="cuda", dtype=torch.bfloat16)
residual = torch.randn_like(x)
weight = torch.randn(hidden, device="cuda", dtype=torch.bfloat16)

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
        _, code = run_and_get_code(compiled, x, residual, weight)
        torch.cuda.synchronize()
    Path(f"generated_vllm_gap_{name}.py").write_text("\n".join(code))
