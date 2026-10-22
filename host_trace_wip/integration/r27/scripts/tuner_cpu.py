import sys, torch
import torch.cuda._host_trace as ht
import torch.cuda._host_trace_ir as ir
sys.path.insert(0, "/data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27/armV_r27")
from flashinfer.autotuner import AutoTuner
from flashinfer.gemm.gemm_base import _MM_FP4_TUNING_CONFIG_128x4 as cfg
from flashinfer.fused_moe.utils import map_to_hybrid_bucket_uncapped
import adapter
for M in (1, 5, 64, 300, 1100, 3000, 9000):
    env = ir.Env()
    s = env.symbol(M, "M", positive=True)
    shapes = ((s, 2560), (5120, 2560), (s, 160), (5120, 160), (), (), (s, 5120), (), (), (1 << 25,))
    adapter._install_fi_tuner_local()
    try:
        prof = AutoTuner._find_nearest_profile(shapes, cfg)
        print(M, "profile", prof[0], prof[2], prof[6], "eager", map_to_hybrid_bucket_uncapped(M), "guards", len(env.records),
              [ir.render(n) if hasattr(ir, "render") else n for n, _ in env.records][:8])
    except Exception as e:
        print(M, "ERR", type(e).__name__, str(e)[:300])
# second "trace" with the same value: no cross-trace comparison
print("lru size", AutoTuner._find_nearest_profile_cached.cache_info())
