import sys, unittest.mock as mock
sys.path.insert(0, "src_vllmcpp/test")
sys.argv = sys.argv[:1]
import torch
import test_cuda_host_trace_harvest as T
from torch.cuda import _host_trace_harvest as harvest
from torch.cuda._host_trace_harvest import HarvestProvider
from torch.cuda._host_trace_replay import HostTraceReplay
t = T.TestExternHarvest("test_learned_in_learning_variant")
bf16 = torch.bfloat16
ops = torch.ops.ht_extern
def show(tag, p, r):
    print(tag, "learned", r.learned, "traces", r.traces, "relowers", r.relowers, "eager", r.eager, "replays", r.replays, "refused", list(p.refused.values())[:2],
          "variants", [(v.learns, len(v.captured.lowered.sites), len(v.captured.lowered.opaque)) for v in r.variants], flush=True)
# interleaved
indexed = {ops.store_paged.default: ("slot", "k_cache", "v_cache", 2)}
kv = torch.randn(4096, 2, 16, 128, device="cuda", dtype=bf16)
cases = [(torch.randn(n, 256, device="cuda", dtype=bf16), kv, torch.randperm(4096 * 16, device="cuda")[:n]) for n in (3, 3, 11, 40, 17)]
for groups in (True, False):
    with mock.patch.object(harvest, "_EXTERN_CAP", 1 << 20), mock.patch.object(harvest, "_LENT", 64 << 10):
        p, r = t.run_cases(t.store_paged, cases, indexed, (kv,), lend_alias_groups=groups)
    show(f"paged groups={groups}", p, r)
# learning variant with a keyed site: rope's keys prebound by another entry
table = torch.randn(4096, 64, device="cuda", dtype=bf16)
k_cache, v_cache = (torch.randn(8192, 64, device="cuda", dtype=bf16) for _ in range(2))
def f(qkv, pos, table, k_cache, v_cache, loc):
    return t.store(t.rope(qkv, pos, table), k_cache, v_cache, loc)
def case(n):
    return torch.randn(n, 192, device="cuda", dtype=bf16), torch.randint(4096, (n,), device="cuda"), table, k_cache, v_cache, torch.randperm(8192, device="cuda")[:n]
idx2 = {ops.store_kv.default: ("loc", "k_cache", "v_cache")}
for on in (True, False):
    with mock.patch.object(harvest, "_LENT", 64 << 10):
        p = HarvestProvider(("extern",), extern_ops=(ops.rope_qk.default, ops.store_kv.default), indexed=idx2, lendable=(table, k_cache, v_cache))
        r0 = HostTraceReplay(t.rope, opaque=(p,))
        for n in (5, 5, 5):
            c = case(n)
            r0(c[0], c[1], table)
        show("prewarm", p, r0)
        r = HostTraceReplay(f, opaque=(p,), learn_in_learning_variants=on)
        for n in (5, 5, 17, 33, 9, 40):
            args = case(n)
            ref = [a.clone() if a.is_floating_point() else a for a in args]
            assert torch.equal(r(*args), f(*ref))
        show(f"learner on={on}", p, r)
