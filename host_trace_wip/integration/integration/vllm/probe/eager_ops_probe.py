# The three ops VP's sampler tail needed that a trace runs eagerly (each an EagerCall boundary): vLLM's
# gumbel_sample end (local_argmax.gather(-1, local_max.argmax(-1, keepdim=True))) and the two stand-ins tried.
#   python eager_ops_probe.py
import torch
import torch.cuda._host_trace_replay as R

dev = torch.device("cuda")
n, blocks = 8, 149  # decode bs 8; Qwen3 vocab 151936 / BLOCK_SIZE 1024


def gather(local_argmax, local_max):
    return (local_argmax.gather(-1, local_max.argmax(dim=-1, keepdim=True)).view(-1),)


def index(local_argmax, local_max):
    return (local_argmax[torch.arange(local_argmax.shape[0], device=local_argmax.device), local_max.argmax(dim=-1)],)


def long_sum(local_argmax, local_max):
    sel = torch.arange(local_argmax.shape[1], device=local_argmax.device) == local_max.argmax(dim=-1, keepdim=True)
    return ((local_argmax * sel).sum(dim=-1),)


for name, fn in (("gather", gather), ("index", index), ("long_sum", long_sum)):
    e = R.HostTraceReplay(fn)
    for i in range(3):
        la = torch.randint(0, 151936, (n, blocks), device=dev, dtype=torch.int64)
        lm = torch.randn(n, blocks, device=dev, dtype=torch.float32)
        (out,) = e(la, lm)
        assert torch.equal(out, fn(la, lm)[0])
    v = e.variants[0] if e.variants else None
    print(f"{name}: traces {e.traces} replays {e.replays} variants {len(e.variants)} segments {len(v.captured.segments) if v else None}")
    print(f"  declines {sorted({str(d)[:240] for d in e.declines})[:3]}")
