import sys, unittest.mock as mock
sys.path.insert(0, "src_vllmcpp/test")
sys.argv = sys.argv[:1]
import torch
import test_cuda_host_trace_harvest as T
from torch.cuda import _host_trace_harvest as harvest
t = T.TestExternHarvest("test_learned_in_learning_variant")
bf16 = torch.bfloat16
table = torch.randn(4096, 64, device="cuda", dtype=bf16)
k_cache, v_cache = (torch.randn(8192, 64, device="cuda", dtype=bf16) for _ in range(2))
def f(qkv, pos, table, k_cache, v_cache, loc):
    return t.store(t.rope(qkv, pos, table), k_cache, v_cache, loc)
def case(n):
    return torch.randn(n, 192, device="cuda", dtype=bf16), torch.randint(4096, (n,), device="cuda"), table, k_cache, v_cache, torch.randperm(8192, device="cuda")[:n]
indexed = {torch.ops.ht_extern.store_kv.default: ("loc", "k_cache", "v_cache")}
for sizes in ((5, 17, 33, 9, 40), (5, 5, 17, 33, 9, 17, 40)):
    for on in (True, False):
        with mock.patch.object(harvest, "_LENT", 64 << 10):
            p, r = t.run_cases(f, [case(n) for n in sizes], indexed, (table, k_cache, v_cache), learn_in_learning_variants=on)
        print(sizes, on, "learned", r.learned, "traces", r.traces, "relowers", r.relowers, "eager", r.eager, "replays", r.replays,
              "variants", [(v.learns, len(v.captured.lowered.sites)) for v in r.variants], "reasons", list(r._reasons)[:4], flush=True)
