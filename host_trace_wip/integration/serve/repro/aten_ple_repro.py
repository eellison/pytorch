# Torch-level repro of the ATen ops that run as eager steps in Qwen3.8-Flash-Next's decode variant under V-stock (fs1):
# they come from vLLM's per-layer-embedding layer (vllm/models/qwen4_exp/nvidia/ple_layer.py: compute_ngram_ids,
# _shift_precompute / _shift_apply, the decode short conv). Mirrors those op sequences with the same dtypes; each function
# is a HostTraceReplay called at a few (num_reqs, num_tokens); prints the variant's eager steps with their reasons and a
# bitwise check against eager.
# Run: CUDA_VISIBLE_DEVICES=1 bash serve/python_vllm_s280.sh serve/repro/aten_ple_repro.py
import torch
import torch.cuda._host_trace_replay as R
import torch.cuda._host_trace_tape as T
import torch.nn.functional as F

EOS = 7
POS = torch.arange(1024, device="cuda", dtype=torch.int64)
PADDED = torch.full((64, 1024), EOS, device="cuda", dtype=torch.int64)


def ngram_ids(input_ids, qsl, positions_buffer, padded_buffer):
    # compute_ngram_ids: searchsorted, clamp_ with a symbolic max, index.Tensor, clamp(0, W-1), index_put_
    num_reqs, num_tokens = qsl.shape[0] - 1, input_ids.shape[0]
    positions = positions_buffer[:num_tokens]
    packed = padded_buffer[:num_reqs, :num_tokens]
    packed.fill_(EOS)
    request_indices = torch.searchsorted(qsl, positions, right=True) - 1
    request_indices.clamp_(max=num_reqs - 1)
    columns = (positions - qsl[request_indices]).clamp(0, packed.shape[1] - 1)
    packed[request_indices, columns] = input_ids
    # _shift_precompute / _shift_apply: cummax, gather
    pos = torch.arange(packed.shape[1], device=packed.device, dtype=torch.int64)
    eos_positions = torch.where(packed == EOS, pos, -1)
    prev = torch.cummax(eos_positions, dim=1).values
    gather_indices = (pos - 1).clamp_min(0).unsqueeze(0).expand(packed.shape[0], -1)
    shifted = packed.gather(1, gather_indices)
    return (packed.clone(), prev, shifted)


def short_conv(x, conv_state, state_indices, w):
    # the decode short conv: index_select, depthwise F.conv1d, index_copy_
    cached = conv_state.index_select(0, state_indices)
    history = torch.cat((cached[..., :3], x.unsqueeze(-1)), dim=-1)
    out = F.silu(F.conv1d(history, w.unsqueeze(1).contiguous(), groups=history.size(1)).squeeze(-1))
    cached[..., :3] = history[..., -3:]
    conv_state.index_copy_(0, state_indices, cached)
    return (out,)


def eager_steps(entry):
    rows = {}
    for v in entry.variants:
        for _, rec in v.tape.launches:
            if isinstance(rec, T.EagerCall) and not rec.host:
                name = getattr(rec.target, "__name__", str(rec.target))
                rows[f"{name}: {(rec.reason or '')[:200]}"] = rows.get(f"{name}: {(rec.reason or '')[:200]}", 0) + 1
    return rows


torch.manual_seed(0)
for name, fn, mk in (
    ("ngram_ids", ngram_ids, lambda n, t: (torch.randint(0, 1000, (t,), device="cuda"),
                                            torch.linspace(0, t, n + 1, device="cuda").long(), POS, PADDED)),
    ("short_conv", short_conv, lambda n, t: (torch.randn(n, 256, device="cuda", dtype=torch.bfloat16),
                                              CONV, torch.randperm(64, device="cuda")[:n], W)),
):
    CONV = torch.randn(64, 256, 4, device="cuda", dtype=torch.bfloat16)
    W = torch.randn(256, 4, device="cuda", dtype=torch.bfloat16)
    entry = R.HostTraceReplay(fn)
    ok, err = [], None
    for n, t in ((3, 3), (5, 5), (3, 3), (8, 8), (5, 5)):
        args = mk(n, t)
        try:
            got = entry(*args)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            break
        want = fn(*mk(n, t)) if fn is ngram_ids else None
        ok.append(None if want is None else all(torch.equal(a, b) for a, b in zip(got, want)))
    print(name, "traces", entry.traces, "replays", entry.replays, "eager", entry.eager, "variants", len(entry.variants), "bitwise", ok,
          "declines", sorted({str(d)[:200] for d in entry.declines})[:2], "escaped", err, flush=True)
    for k, c in eager_steps(entry).items():
        print("   EAGER", c, k, flush=True)
