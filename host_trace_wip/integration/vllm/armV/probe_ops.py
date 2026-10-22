# Which dispatcher ops vLLM's eager Qwen3 forward calls (direct-call attention), per stage: python_vllm_v.sh armV/probe_ops.py
import collections, os, sys
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
from vllm.forward_context import get_forward_context

OPS = {}


class Log(TorchDispatchMode):
    def __init__(self, c):
        super().__init__()
        self.c = c

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.c[str(func)] += 1
        return func(*args, **(kwargs or {}))


orig = Qwen3ForCausalLM.forward


def fwd(self, *a, **k):
    from vllm.model_executor.layers.attention.attention import Attention
    for m in self.modules():
        if isinstance(m, Attention):
            m.use_direct_call = True
    amd = get_forward_context().attn_metadata
    if not amd:
        return orig(self, *a, **k)
    md = next(iter(amd.values()))
    tag = ("decode" if md.num_prefills == 0 else "prefill" if md.num_decodes == 0 else "mixed", md.num_actual_tokens)
    c = collections.Counter()
    with Log(c):
        out = orig(self, *a, **k)
    if tag not in OPS:
        OPS[tag] = c
        print("FWD", tag, {kk: v for kk, v in self.__dict__.items() if False}, flush=True)
        print("  kwargs", {kk: (type(v).__name__, getattr(v, "shape", None)) for kk, v in k.items()}, "md", type(md).__name__,
              {f: getattr(md, f) for f in ("num_decodes", "num_prefills", "num_decode_tokens", "num_prefill_tokens")},
              type(md.decode).__name__, type(md.prefill).__name__, flush=True)
        if md.decode is not None:
            print("  decode", md.decode.kernel, md.decode.max_seq_len, md.decode.q_len_per_req, flush=True)
        if md.prefill is not None:
            print("  prefill", md.prefill.max_q_len, md.prefill.max_seq_len, flush=True)
        for kk, v in c.most_common():
            print(f"  {v:6d} {kk}", flush=True)
    return out


Qwen3ForCausalLM.forward = fwd
llm = LLM(model="/data/eellison/models/Qwen3-8B", enforce_eager=True, attention_backend="FLASHINFER", enable_prefix_caching=False,
          gpu_memory_utilization=0.6, max_model_len=4096, seed=0)
g = torch.Generator().manual_seed(0)
P = lambda n, L: [TokensPrompt(prompt_token_ids=torch.randint(1000, 100000, (L,), generator=g).tolist()) for _ in range(n)]
llm.generate(P(2, 64), SamplingParams(max_tokens=3, ignore_eos=True, temperature=0.0), use_tqdm=False)
# Show the schema of the ops we will declare
for n in ("_C_cache_ops.reshape_and_cache_flash", "_C.rms_norm", "_C.fused_add_rms_norm", "_C.rotary_embedding", "_C.silu_and_mul"):
    ns, op = n.split(".")
    try:
        print("SCHEMA", getattr(getattr(torch.ops, ns), op).default._schema, flush=True)
    except Exception as e:
        print("SCHEMA?", n, e)
from vllm.model_executor.layers.attention.attention import Attention
m = next(m for m in llm.llm_engine.model_executor.driver_worker.worker.model_runner.model.modules() if isinstance(m, Attention))
print("LAYER", type(m.kv_cache), getattr(m.kv_cache, "shape", None), getattr(m.kv_cache, "stride", lambda: None)(), m._k_scale.device, type(m._k_scale))
print("ATTRS", [(k, tuple(v.shape), v.device.type) for k, v in vars(m).items() if isinstance(v, torch.Tensor)])
