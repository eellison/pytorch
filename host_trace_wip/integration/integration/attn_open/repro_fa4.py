# Engine-free repro: vLLM's FA4 (CuTe DSL) paged attention call under HostTraceReplay, as FlashAttentionImpl.forward makes
# it (flash_attn_varlen_func(fa_version=4), paged KV 16-token blocks, int lengths as arguments). Prints per case: traces,
# variants, eager steps with their full reasons, declines, retrace causes, bitwise vs eager.
#   CUDA_VISIBLE_DEVICES=1 CAND_LINE=pinned bash python_vllm_cand2.sh attn_open/repro_fa4.py
import collections, os, sys
import cutlass  # noqa: F401
import torch
import torch.cuda._host_trace_cute as C

C.install()
import torch.cuda._host_trace_replay as R
import torch.cuda._host_trace_tape as T
from torch.cuda._host_trace_launch import KernelLaunch
from vllm.vllm_flash_attn import flash_attn_varlen_func

torch.manual_seed(0)
H, KVH, D, BS, NB = 32, 8, 128, 16, 4096
kv = torch.randn(2, NB, BS, KVH, D, device="cuda", dtype=torch.bfloat16)
kc, vc = kv.unbind(0)


def attn(q, kc, vc, out, cu, seqused, bt, max_q, max_k):
    flash_attn_varlen_func(q=q, k=kc, v=vc, out=out, cu_seqlens_q=cu, max_seqlen_q=max_q, seqused_k=seqused, max_seqlen_k=max_k,
                           softmax_scale=D ** -0.5, causal=True, alibi_slopes=None, window_size=None, block_table=bt, softcap=0,
                           scheduler_metadata=None, fa_version=4, q_descale=None, k_descale=None, v_descale=None, num_splits=0, s_aux=None)
    return (out,)


def inputs(qlens, klens):
    B = len(qlens)
    nblk = max(-(-k // BS) for k in klens)
    bt = torch.randperm(NB, device="cuda", dtype=torch.int32)[: B * nblk].view(B, nblk).contiguous()
    cu = torch.tensor([0] + list(torch.tensor(qlens).cumsum(0)), device="cuda", dtype=torch.int32)
    q = torch.randn(sum(qlens), H, D, device="cuda", dtype=torch.bfloat16)
    return q, kc, vc, torch.empty_like(q), cu, torch.tensor(klens, device="cuda", dtype=torch.int32), bt, max(qlens), max(klens)


def report(name, entry, ok):
    print(f"== {name}: traces {entry.traces} replays {entry.replays} eager {entry.eager} variants {len(entry.variants)} bitwise {ok}", flush=True)
    for i, v in enumerate(entry.variants):
        recs = [rec for _, rec in v.tape.launches]
        k = sum(isinstance(r, KernelLaunch) for r in recs)
        eg = collections.Counter((r.reason or "?") for r in recs if isinstance(r, T.EagerCall))
        print(f"  variant {i}: {k} kernel launches, {sum(eg.values())} eager steps, names {[getattr(r, 'name', '?')[-60:] for r in recs if isinstance(r, KernelLaunch)]}")
        for why, n in eg.items():
            print(f"    {n}x eager: {why}")
    for d in sorted({str(d) for d in entry.declines})[:6]:
        print("  decline:", d[:1500])
    for c, n in getattr(entry, "retrace_causes", {}).items():
        print("  retrace", n, c)


cases = {"decode": [([1] * b, [300 + 7 * i for i in range(b)]) for b in (4, 4, 8, 8, 1, 1, 64, 64, 8)],
         "prefill": [([L], [L]) for L in (512, 512, 300, 300, 2048, 2048, 64, 64)],
         "mixed": [([1] * 8 + [128], [256 + i for i in range(8)] + [128])] * 4}
only = sys.argv[1:] or list(cases)
for name in only:
    entry = R.HostTraceReplay(attn, opaque=(), memory="run_buffer")
    ok = []
    for qlens, klens in cases[name]:
        args = inputs(qlens, klens)
        (got,) = entry(*args)
        got = got.clone()
        a2 = list(args); a2[3] = torch.empty_like(args[3])
        (want,) = attn(*a2)
        torch.cuda.synchronize()
        ok.append(bool(torch.equal(got, want)))
    report(name, entry, ok)
