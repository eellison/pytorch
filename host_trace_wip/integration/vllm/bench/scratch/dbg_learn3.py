import sys, unittest.mock as mock
sys.path.insert(0, "src_vllmcpp/test")
sys.argv = sys.argv[:1]
import torch
import test_cuda_host_trace_harvest as T
from torch.cuda import _host_trace_harvest as harvest
t = T.TestExternHarvest("test_learned_in_learning_variant")
bf16 = torch.bfloat16
ops = torch.ops.ht_extern
def show(tag, p, r):
    print(tag, "learned", r.learned, "traces", r.traces, "relowers", r.relowers, "eager", r.eager, "replays", r.replays, "refused", list(p.refused.values())[:2],
          "variants", [(v.learns, len(v.captured.lowered.sites), len(v.captured.lowered.opaque)) for v in r.variants], flush=True)
indexed = {ops.store_paged.default: ("slot", "k_cache", "v_cache", 2)}
kv = torch.randn(4096, 2, 16, 128, device="cuda", dtype=bf16)
for sizes in ((3, 3, 3, 11, 40, 17), (3, 3, 3, 3, 11, 40, 17)):
    cases = [(torch.randn(n, 256, device="cuda", dtype=bf16), kv, torch.randperm(4096 * 16, device="cuda")[:n]) for n in sizes]
    for groups in (True, False):
        with mock.patch.object(harvest, "_EXTERN_CAP", 1 << 20), mock.patch.object(harvest, "_LENT", 64 << 10):
            p, r = t.run_cases(t.store_paged, cases, indexed, (kv,), lend_alias_groups=groups)
        show(f"{sizes} groups={groups}", p, r)
