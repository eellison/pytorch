# gdn_ours3.py (gdn_ours2.py + TMA descriptor bytes, eager witness vs our encode): FlashInfer GDN prefill under OUR host-trace replay with its module-global workspace lifted as an argument (as the adapter's
# ARMV_GDN_WS does), no vLLM: per shape and process, eager twice, then 6 HostTraceReplay calls (eager, trace, replays), each compared
# bitwise with eager; NaN counts. Isolates our capture and replay of this one CuTe kernel.
import gc, os
import torch
import cutlass  # noqa: F401
import torch.cuda._host_trace_cute as htc
htc.install()
from torch.cuda._host_trace_replay import HostTraceReplay
# r27 diagnosis: the TMA descriptor parameter bytes, eager (the DSL's own launch, as the trace's witness capture holds it) vs ours (the
# driver encode a replay launches)
import hashlib
import torch.cuda._host_trace_launch as htl
LOGB = []
_orig_describe = htc._describe


def _describe_log(tr, launch, node, roots, owner):
    rec = _orig_describe(tr, launch, node, roots, owner)
    for i, p in enumerate(launch.params):
        if getattr(p, "tma", None):
            img = node.images[i]
            LOGB.append(("eager", node.name[:30], i, img[:64].hex()[:24], hashlib.md5(img[64:]).hexdigest()[:8], img[64:].count(0)))
    return rec
htc._describe = _describe_log
_orig_encode = htl.TmaDescriptor.encode


def _encode_log(self, slots, edits=None):
    b = _orig_encode(self, slots, edits)
    LOGB.append(("ours", self.param, bytes(b)[:64].hex()[:24], hashlib.md5(bytes(b)[64:]).hexdigest()[:8], bytes(b)[64:].count(0)))
    return b
htl.TmaDescriptor.encode = _encode_log

from flashinfer.gdn_prefill import chunk_gated_delta_rule
import flashinfer.gdn_kernels.blackwell.gdn_prefill as gp

dev, HK, HV, D = "cuda", 16, 48, 128


def cfgs():
    return [(cfg, k) for ref in gc.get_referents(gp._get_compiled_cache) if isinstance(ref, dict) for cfg in ref.values()
            if isinstance(cfg, dict) for k in list(cfg) if isinstance(k, str) and k.startswith("workspace_")]


def gdn(q, k, v, gate, beta, st, cu, ws):
    saved = [(c, kk, c[kk]) for c, kk in cfgs()]
    for c, kk, _ in saved:
        c[kk] = ws
    try:
        return chunk_gated_delta_rule(q=q, k=k, v=v, g=gate, beta=beta, initial_state=st, output_final_state=True, cu_seqlens=cu)
    finally:
        for c, kk, old in saved:
            c[kk] = old


for spec in os.environ.get("GDN_SHAPES", "1x64 1x512 1x2048").split():
    seqs, T = map(int, spec.split("x"))
    g = torch.Generator(device=dev).manual_seed(0)
    N = seqs * T
    q = torch.nn.functional.normalize(torch.randn(N, HK, D, device=dev, generator=g), dim=-1).to(torch.bfloat16)
    k = torch.nn.functional.normalize(torch.randn(N, HK, D, device=dev, generator=g), dim=-1).to(torch.bfloat16)
    v = torch.randn(N, HV, D, device=dev, generator=g).to(torch.bfloat16)
    gate = torch.exp(-torch.rand(N, HV, device=dev, generator=g))
    beta = torch.rand(N, HV, device=dev, generator=g)
    state0 = torch.zeros(seqs, HV, D, D, device=dev)
    cu = torch.arange(0, N + 1, T, dtype=torch.int64, device=dev)
    e1 = [t.clone() for t in chunk_gated_delta_rule(q=q, k=k, v=v, g=gate, beta=beta, initial_state=state0.clone(), output_final_state=True, cu_seqlens=cu)]
    e2 = [t.clone() for t in chunk_gated_delta_rule(q=q, k=k, v=v, g=gate, beta=beta, initial_state=state0.clone(), output_final_state=True, cu_seqlens=cu)]
    ws = cfgs()[0][0][cfgs()[0][1]] if cfgs() else torch.empty(1 << 20, dtype=torch.int8, device=dev)
    ws = torch.empty_like(ws)  # the compiled function takes the workspace at its compiled size
    r = HostTraceReplay(gdn)
    outs, paths = [], []
    for i in range(6):
        t0, p0 = r.traces, r.replays
        res = r(q, k, v, gate, beta, state0.clone(), cu, ws)
        torch.cuda.synchronize()
        paths.append("R" if r.replays > p0 else "T" if r.traces > t0 else "E")
        outs.append([t.clone() for t in res])
    eq = lambda a, b: all(torch.equal(x.nan_to_num(7.0), y.nan_to_num(7.0)) for x, y in zip(a, b))
    print(f"{spec}: paths {''.join(paths)} | eager-eager {eq(e1, e2)} nan eager {int(e1[0].isnan().sum())} | call-vs-eager {[eq(o, e1) for o in outs]} "
          f"nan calls {[int(o[0].isnan().sum()) for o in outs]}", flush=True)
    seen = set()
    for x in LOGB:
        if x not in seen:
            seen.add(x); print("   TMA", x, flush=True)
    LOGB.clear()
    if r.replays == 0:
        print("   declines:", sorted({str(d)[:300] for d in r.declines})[:3], flush=True)
