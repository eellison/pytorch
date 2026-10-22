# Torch-level repro of the ATen eager steps in Qwen3.8-Flash-Next's traced forward (fn_v2 / fn_v3) from vLLM 0.29's PLE
# short conv (vllm/models/qwen4_exp/nvidia/ple_layer.py, _short_conv_dilated_decode_batched and the vstack in
# _short_conv_dilated_dispatch), on the real conv_state layout (printed by fn_v3: kv_cache[0] (432, 9, 10240) bf16, strides
# (19869696, 10240, 1): one page per block; _short_conv transposes it to (432, 10240, 9) strides (19869696, 1, 10240)).
# Each function is a HostTraceReplay called at a few request counts; prints traces / replays / eager steps with their
# reasons, the trace declines, and a bitwise check against eager on cloned state.
#   gpu3.sh r_ple bash python_fn.sh tests/repro_ple_conv.py
import collections

import torch
import torch.nn.functional as F

import torch.cuda._host_trace as ht
import torch.cuda._host_trace_replay as R
import torch.cuda._host_trace_tape as T

NULL_BLOCK_ID = 0  # vLLM's padded-row block id (value irrelevant here: no padded rows are passed)
BLOCKS, PAGE, H, L = 112, 19869696, 10240, 9  # 432 blocks in the run; 112 here (4.5 GB) so the span passes 2**31 elements as there (16 blocks: no repro); conv_state_len = (kernel 4 - 1) * dilation 3
DECLINES = collections.Counter()
_init = ht.Declined.__init__


def _rec(exc, *a, **k):
    _init(exc, *a, **k)
    DECLINES[str(exc)[:260]] += 1


ht.Declined.__init__ = _rec


def conv_state_pool():
    raw = torch.randn(BLOCKS * PAGE + 16, device="cuda", dtype=torch.bfloat16)
    kv0 = raw.as_strided((BLOCKS, L, H), (PAGE, H, 1))
    return raw, kv0


def decode(x_d, kv0, state_indices_tensor_d, w):
    # _short_conv: conv_state = kv_cache[0].transpose(-1, -2); then _short_conv_dilated_decode_batched (has_initial_states None)
    conv_state = kv0.transpose(-1, -2)
    # _short_conv (ple_layer.py:1134-1147): the state capacity slice (conv_state_len + num_spec_tokens = 9), which makes the
    # traced view's storage offset s_stride * (s_size - 9) + s_offset (fn_v4a FN_DIAG_PLE)
    if conv_state.size(-1) < L:
        raise ValueError("capacity")
    conv_state = conv_state[..., -L:]
    state_indices = state_indices_tensor_d.to(device=conv_state.device, dtype=torch.int64)
    valid_state = state_indices != NULL_BLOCK_ID
    state_indices = torch.where(valid_state, state_indices, torch.zeros_like(state_indices))
    has_initial_state = valid_state
    cached_state = conv_state.index_select(0, state_indices)  # ple_layer.py:694
    state = cached_state[..., :L].to(x_d.dtype)
    initial_state = torch.where(has_initial_state.view(-1, 1, 1), state, torch.zeros_like(state))
    history = torch.cat((initial_state, x_d.unsqueeze(-1)), dim=-1)
    conv_output = F.conv1d(history, w.unsqueeze(1).contiguous(), groups=history.size(1), dilation=3).squeeze(-1)
    output = F.silu(conv_output)
    output = output * valid_state.view(-1, 1).to(output.dtype)  # ple_layer.py:713
    next_state = history[..., -L:]
    existing_base_state = cached_state[..., :L]
    safe_next_state = torch.where(valid_state.view(-1, 1, 1), next_state.to(conv_state.dtype), existing_base_state)
    cached_state[..., :L] = safe_next_state  # ple_layer.py:725
    conv_state.index_copy_(0, state_indices, cached_state)
    return (output,)


def decode_then_vstack(x_d, kv0, state_indices_tensor_d, w, prefill_part):
    # _short_conv_dilated_dispatch's mixed batch: the decode part and a prefill part stacked (ple_layer.py:1085)
    (out_d,) = decode(x_d, kv0, state_indices_tensor_d, w)
    return (torch.vstack([out_d, prefill_part]),)


def eager_steps(entry):
    rows = collections.Counter()
    for v in entry.variants:
        for _, rec in v.tape.launches:
            if isinstance(rec, T.EagerCall) and not rec.host:
                rows[f"{getattr(rec.target, '__name__', str(rec.target))}: {(rec.reason or '')[:220]}"] += 1
    return rows


torch.manual_seed(0)
raw, kv0 = conv_state_pool()
w = torch.randn(H, 4, device="cuda", dtype=torch.bfloat16)
# static_prefix=2: kv0 and w as the adapter passes module state (HostTraceReplay(static_prefix=len(prefix)): params,
# buffers and kv_cache first), the other arguments as call arguments
def decode_sp(kv0, w, x_d, idx):
    return decode(x_d, kv0, idx, w)


def vstack_sp(kv0, w, x_d, idx, prefill_part):
    return decode_then_vstack(x_d, kv0, idx, w, prefill_part)


# vLLM's form: the short conv runs inside the custom op vllm::qwen4_exp_ple_short_conv(inputs, output, layer_name)
# (mutates_args=["output"]), whose Python body reads the layer's conv_state and the step's metadata from module /
# forward-context state, not from its arguments (the adapter swaps the traced tensors into that state).
STATE = {}


@torch.library.custom_op("fnrepro::ple_short_conv", mutates_args=["output"])
def ple_short_conv(inputs: torch.Tensor, output: torch.Tensor, layer_name: str) -> None:
    kv0, w, idx = STATE[layer_name]
    (result,) = decode(inputs, kv0, idx, w)
    output[: result.shape[0]].copy_(result)


@ple_short_conv.register_fake
def _(inputs, output, layer_name):
    return None


def decode_op(kv0, w, x_d, idx, out):
    STATE["l"] = (kv0, w, idx)
    torch.ops.fnrepro.ple_short_conv(x_d, out, "l")
    return (out,)


for name, fn, extra, sp in (("decode in a custom op body static_prefix=2", decode_op,
                             lambda n: (torch.empty(n, H, device="cuda", dtype=torch.bfloat16),), 2),
                            ("decode", decode, lambda n: (), 0),
                            ("decode_then_vstack", decode_then_vstack, lambda n: (torch.randn(2 * n + 1, H, device="cuda", dtype=torch.bfloat16),), 0),
                            ("decode static_prefix=2", decode_sp, lambda n: (), 2),
                            ("decode_then_vstack static_prefix=2", vstack_sp, lambda n: (torch.randn(2 * n + 1, H, device="cuda", dtype=torch.bfloat16),), 2)):
    DECLINES.clear()
    entry = R.HostTraceReplay(fn, static_prefix=sp)
    bitwise = []
    for n in (3, 5, 3, 8, 5, 3):  # n < BLOCKS
        g = torch.Generator(device="cuda").manual_seed(n)
        x = torch.randn(n, H, device="cuda", dtype=torch.bfloat16, generator=g)
        idx = (torch.randperm(BLOCKS - 1, device="cuda", generator=g)[:n] + 1).to(torch.int32)
        ext = extra(n)
        before = raw.clone()
        call = (kv0, w, x, idx, *ext) if sp else (x, kv0, idx, w, *ext)
        want = fn(*(c.clone() if any(c is e for e in ext) else c for c in call))  # its own output buffers
        want_state = raw.clone()
        raw.copy_(before)
        got = entry(*call)
        torch.cuda.synchronize()
        bitwise.append(all(torch.equal(a, b) for a, b in zip(got, want)) and torch.equal(raw, want_state))
    print(name, "traces", entry.traces, "replays", entry.replays, "eager", entry.eager, "variants", len(entry.variants),
          "bitwise", bitwise, flush=True)
    for k, c in eager_steps(entry).items():
        print("   EAGER STEP", c, k, flush=True)
    for k, c in DECLINES.most_common(8):
        print("   DECLINE", c, k, flush=True)
