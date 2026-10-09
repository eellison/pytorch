# Torch-level repros of the core gaps seen with stock SGLang + the hook (Qwen3-8B, attn line). Each case is a
# HostTraceReplay over plain torch code shaped as SGLang's; prints traces / replays / variants / segments, eager-step
# declines and retrace causes, and checks every call against eager.   launcher core_gaps_sglang.py
# 1. prefill last-token gather, SGLang's LogitsProcessor._get_pruned_states (logits_processor.py:677-678):
#    last_index = torch.cumsum(extend_seq_lens, dim=0) - 1; pruned = hidden_states[last_index]
#    seen: 2 eager steps per extend variant (aten.cumsum "pointwise host declines: an iterator writes a temporary",
#    aten.index.Tensor "no traced host"), plus a retrace at batch 1 ("aten.cumsum.default returned other metadata").
# 3. an op-owned kernel choice whose two outcomes are different kernels (the fork's trtllm-gen decode: the multi-CTA
#    split count from max(sm_count // (heads_kv * bs), 1) picks another cubin / grid z): seen as retrace causes
#    ('dispatch', flashinfer_ht.trtllm_paged_attention_decode, (max(19 // bs, 1) - 4) == 0, trtllm_trace.py:470,
#    'another launch topology in ...'), one whole-graph trace per split class. Here a dispatch_unit picks kernel A
#    (one program) below 16 rows and kernel B (one program per row) above: a redispatch finds another function.
# 2. decode embedding at batch 1 (VocabParallelEmbedding -> F.embedding, unquant.py:429): retrace cause
#    ('meta', aten.embedding.default, (s - 1) == 0) between bs 1 and bs > 1.
import torch
import torch.cuda._host_trace as ht
import torch.cuda._host_trace_replay as R
import triton
import triton.language as tl

dev = torch.device("cuda")
H = 4096


def pruned(hidden_states, extend_seq_lens):
    last_index = torch.cumsum(extend_seq_lens, dim=0) - 1
    return (hidden_states[last_index],)


def embed(weight, ids):
    return (torch.nn.functional.embedding(ids, weight),)


@triton.jit
def _scale_one(x, y, n, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    tl.store(y + i, tl.load(x + i, mask=i < n) * 2, mask=i < n)


@triton.jit
def _scale_rows(x, y, n):
    i = tl.program_id(0)
    tl.store(y + i, tl.load(x + i) * 2)


@ht.dispatch_unit
def _scaled(x, y):
    n = x.shape[0]
    if n < 16:
        _scale_one[(1,)](x, y, n, BLOCK=16)
    else:
        _scale_rows[(n,)](x, y, n)
    return y


@ht.dispatch_unit
def _scaled_split(x, y):
    # as the trtllm-gen decode's multi-CTA path: above 16 rows a partial buffer and a second (reduce) launch
    n = x.shape[0]
    if n < 16:
        _scale_one[(1,)](x, y, n, BLOCK=16)
    else:
        part = torch.empty_like(x)
        _scale_rows[(n,)](x, part, n)
        _scale_rows[(n,)](part, y, n)
    return y


def choice_split(x):
    y = torch.empty_like(x)
    return (_scaled_split(x * 1, y) + 1,)


def choice(x):
    y = torch.empty_like(x)
    return (_scaled(x * 1, y) + 1,)


def report(name, e):
    v = e.variants[-1] if e.variants else None
    print(f"{name}: traces {e.traces} replays {e.replays} eager {e.eager} variants {len(e.variants)}"
          f" segments {len(v.captured.segments) if v else None}")
    print(f"  declines {sorted({str(d)[:200] for d in e.declines})[:4]}")
    print(f"  retrace_causes {dict(getattr(e, 'retrace_causes', {}))}")


e = R.HostTraceReplay(pruned)
for lens in ([97, 333, 517, 1053], [64, 64], [512], [7, 9, 11], [2048]):
    t = torch.tensor(lens, device=dev, dtype=torch.int32)
    h = torch.randn(sum(lens), H, device=dev, dtype=torch.bfloat16)
    for _ in range(3):
        (out,) = e(h, t)
        assert torch.equal(out, pruned(h, t)[0])
report("pruned_states (cumsum + index.Tensor)", e)

w = torch.randn(151936, H, device=dev, dtype=torch.bfloat16)
e = R.HostTraceReplay(embed)
for bs in (8, 64, 1, 13, 1):
    ids = torch.randint(0, 151936, (bs,), device=dev)
    for _ in range(3):
        (out,) = e(w, ids)
        assert torch.equal(out, embed(w, ids)[0])
report("decode embedding (bs 8, 64, 1, 13, 1)", e)

e = R.HostTraceReplay(choice)
for n in (32, 64, 8, 4, 40, 8):
    x = torch.randn(n, device=dev)
    for _ in range(3):
        (out,) = e(x)
        assert torch.equal(out, choice(x)[0])
report("op-owned kernel choice (n 32, 64, 8, 4, 40, 8)", e)
print("  redispatches", getattr(e, "redispatches", None))

e = R.HostTraceReplay(choice_split)
for n in (32, 64, 8, 4, 40, 8):
    x = torch.randn(n, device=dev)
    for _ in range(3):
        (out,) = e(x)
        assert torch.equal(out, choice_split(x)[0])
report("op-owned choice, split path with a scratch buffer and a 2nd launch (n 32, 64, 8, 4, 40, 8)", e)
print("  redispatches", getattr(e, "redispatches", None))
