from pathlib import Path

import torch

import torch._inductor.config as inductor_config
from bench_main_cuda import rmsnorm_fp4
from torch._inductor import metrics
from torch._inductor.utils import run_and_get_code


torch.manual_seed(0)
rows, hidden = 1024, 4096
x = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
weight = torch.randn(hidden, device="cuda", dtype=torch.bfloat16)

for swizzled in (False, True):
    for coordinate_descent in (False, True):
        torch._dynamo.reset()
        metrics.reset()

        def f(x, weight, swizzled=swizzled):
            return rmsnorm_fp4(x, weight, 16, "e4m3", swizzled)

        with inductor_config.patch(
            {
                "triton.nested_reduction": True,
                "triton.cudagraphs": False,
                "fx_graph_cache": False,
                "emulate_precision_casts": True,
                "coordinate_descent_tuning": coordinate_descent,
            }
        ):
            compiled = torch.compile(f, fullgraph=True, dynamic=False)
            _, sources = run_and_get_code(compiled, x, weight)
            torch.cuda.synchronize()
        layout = "swizzled" if swizzled else "row"
        tuning = "coordesc" if coordinate_descent else "default"
        Path(f"generated_nvfp4_1024_{layout}_{tuning}.py").write_text(
            "\n".join(sources)
        )
        print(layout, tuning, metrics.generated_kernel_count, metrics.codegen_nested_reduction)
