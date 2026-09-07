from pathlib import Path

import torch

import torch._inductor.config as inductor_config
from torch._inductor import metrics
from torch._inductor.utils import run_and_get_code

from bench_main_cuda import fused_add_block_fp8


torch.manual_seed(0)
rows, hidden = 128, 4096
x = torch.zeros(rows, hidden, device="cuda", dtype=torch.bfloat16)
residual = torch.randn_like(x) * 0.1
weight = torch.randn(hidden, device="cuda", dtype=torch.bfloat16)
for coordinate_descent in (False, True):
    metrics.reset()
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
        _, sources = run_and_get_code(compiled, x, residual, weight)
        torch.cuda.synchronize()

    suffix = "coordesc" if coordinate_descent else "default"
    output = Path(__file__).with_name(f"generated_nonpadded_block_fp8_{suffix}.py")
    output.write_text("\n\n".join(sources))
    print(f"variant={suffix}")
    print(f"output={output}")
    print(f"kernel_count={metrics.generated_kernel_count}")
    print(f"nested_reduction_count={metrics.codegen_nested_reduction}")
