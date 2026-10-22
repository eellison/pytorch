# A redispatch re-runs a custom op's body alone; module state the body reads (vLLM: unified_attention_with_output ->
# get_attention_context(layer_name) -> layer.kv_cache, the trtllm workspace global, the forward context's metadata) was swapped
# to the step's traced tensors only while the traced function ran, so the redo sees the real tensor and declines
# ("... of a tensor the trace does not track"), and the call traces again. Torch only.
import torch
import torch.cuda._host_trace as ht
import torch.cuda._host_trace_replay as R

DECLINES = []
_init = ht.Declined.__init__


def _logged(exc, *a, **k):  # every decline raised, including the redo's (caught and turned into a retrace)
    _init(exc, *a, **k)
    DECLINES.append(str(exc)[:200])


ht.Declined.__init__ = _logged

STATE = {}


@torch.library.custom_op("repro_redo::attn", mutates_args=("out",))
def attn(q: torch.Tensor, out: torch.Tensor) -> None:
    kv = STATE["kv"]  # module state, as layer.kv_cache
    # an op-owned host branch with the same kernels on both sides (vLLM utils/torch_utils.py:145 `1 not in t.shape`)
    q2 = q.reshape(1, -1).view(q.shape) if q.shape[0] == 1 else q
    torch.add(q2, kv[: q.shape[0]], out=out)


@attn.register_fake
def _(q, out):
    return None


def step(kv, q):
    saved = STATE.get("kv")
    STATE["kv"] = kv  # the adapter's attribute swap: the traced kv while the step runs
    try:
        out = torch.empty_like(q)
        attn(q, out)
        return (out * 1,)
    finally:
        STATE["kv"] = saved


dev = torch.device("cuda")
kv = torch.randn(64, 8, device=dev)
STATE["kv"] = kv
e = R.HostTraceReplay(step)
for n in (4, 4, 4, 1, 1, 6, 1):
    q = torch.randn(n, 8, device=dev)
    (out,) = e(kv, q)
    (ref,) = step(kv, q)
    torch.cuda.synchronize()
    print(f"n={n}: {'bitwise' if torch.equal(out, ref) else 'DIFFERS'} traces {e.traces} redispatches {getattr(e, 'redispatches', '?')} "
          f"variants {len(e.variants)}", flush=True)
print("retrace_causes:", getattr(e, "retrace_causes", "n/a"))
print("declines raised:", sorted(set(DECLINES)))
print("redispatch refusals:", getattr(e, "redispatch_refusals", None))


# control: the same op with kv as an argument redispatches at the flip and does not trace again
@torch.library.custom_op("repro_redo::attn_arg", mutates_args=("out",))
def attn_arg(q: torch.Tensor, kv: torch.Tensor, out: torch.Tensor) -> None:
    q2 = q.reshape(1, -1).view(q.shape) if q.shape[0] == 1 else q
    torch.add(q2, kv[: q.shape[0]], out=out)


@attn_arg.register_fake
def _(q, kv, out):
    return None


def step_arg(kv, q):
    out = torch.empty_like(q)
    attn_arg(q, kv, out)
    return (out * 1,)


DECLINES.clear()
e = R.HostTraceReplay(step_arg)
for n in (4, 4, 4, 1, 1, 6, 1):
    q = torch.randn(n, 8, device=dev)
    (out,) = e(kv, q)
    (ref,) = step_arg(kv, q)
    torch.cuda.synchronize()
    print(f"control n={n}: {'bitwise' if torch.equal(out, ref) else 'DIFFERS'} traces {e.traces} redispatches {getattr(e, 'redispatches', '?')}", flush=True)
print("control declines raised:", sorted(set(DECLINES)))
print("control retrace_causes:", getattr(e, "retrace_causes", None))
print("control redispatch refusals:", getattr(e, "redispatch_refusals", None))
