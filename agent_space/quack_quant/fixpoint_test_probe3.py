import torch, contextlib
from torch._inductor import config, metrics
from torch._inductor.utils import run_and_get_code
from torch._inductor.virtualized import V
from torch._inductor.choices import InductorChoices
M, K, G = 2048, 3072, 32
def f(x):
    blocks = x.view(M, K // G, G)
    amax = blocks.abs().amax(dim=-1).unsqueeze(-1).float()
    exponent = ((amax.view(torch.int32) >> 23) & 0xFF) - 127 - 2
    biased = (exponent.clamp(-127, 128) + 127).to(torch.uint8)
    scale = ((biased.to(torch.int32) << 23).view(torch.float32)).clamp_min(2.0**-126)
    codes = (blocks.float() / scale).reshape(M, K).clamp(0, 15).to(torch.uint8)
    flat = codes.contiguous().view(-1)
    return (flat[::2] | (flat[1::2] << 4)).view(M, K // 2), biased.squeeze(-1)
def choices(force):
    class _C(InductorChoices):
        @staticmethod
        def should_use_cooperative_reduction(*a, **k): return False
        @staticmethod
        def should_use_persistent_reduction(*a, **k): return force
    return V.set_choices_handler(_C())
x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
for label, ctx, compile_kwargs in (("no handler, fullgraph static", contextlib.nullcontext(), dict(fullgraph=True, dynamic=False)), ("no handler, plain compile", contextlib.nullcontext(), {}), ("force persistent", choices(True), {}), ("force looped", choices(False), {})):
    torch._dynamo.reset(); metrics.reset()
    with config.patch({"triton.nested_reduction": True, "split_reductions": False, "loop_ordering_after_fusion": True, "triton.cudagraphs": False, "fx_graph_cache": False}), ctx:
        out, srcs = run_and_get_code(torch.compile(f, **compile_kwargs), x)
    names = [l[4:l.index("(")][:24] for s in srcs for l in s.splitlines() if l.startswith("def triton_")]
    print(f"{label:30s}: kernels={metrics.generated_kernel_count} nested={metrics.codegen_nested_reduction} {names}", flush=True)
