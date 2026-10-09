# cProfile of execute_model's prep sections (eager arm; prep does not depend on the arm):
# GPUModelRunner.add_requests / prepare_inputs / prepare_attn and DefaultModelState.prepare_attn, for decode bs 8 and prefill 512.
# python_vllm.sh armV/probe_prep.py
import cProfile, io, os, pstats, sys

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
import torch
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.model_states.default import DefaultModelState

PROF = {}
ON = [None]


def profiled(cls, name):
    fn = getattr(cls, name)

    def wrapper(*a, **kw):
        if ON[0] is None:
            return fn(*a, **kw)
        p = PROF.setdefault((ON[0], name), cProfile.Profile())
        p.enable()
        try:
            return fn(*a, **kw)
        finally:
            p.disable()
    setattr(cls, name, wrapper)


for n in ("add_requests", "prepare_inputs", "prepare_attn"):
    profiled(GPUModelRunner, n)
profiled(DefaultModelState, "prepare_attn")

llm = LLM(model="/data/eellison/models/Qwen3-8B", enforce_eager=True, attention_backend="FLASHINFER", enable_prefix_caching=False,
          gpu_memory_utilization=0.6, max_model_len=4096, seed=0)
g = torch.Generator().manual_seed(0)
pr = lambda n, L: [TokensPrompt(prompt_token_ids=torch.randint(1000, 100000, (L,), generator=g).tolist()) for _ in range(n)]
sp = lambda n: SamplingParams(max_tokens=n, ignore_eos=True, temperature=0.0)
llm.generate(pr(4, 128), sp(8), use_tqdm=False)
ON[0] = "decode8"
llm.generate(pr(8, 256), sp(25), use_tqdm=False)  # 1 prefill step + 24 decode steps
ON[0] = "prefill512"
for _ in range(5):
    llm.generate(pr(1, 512), sp(1), use_tqdm=False)
ON[0] = None
for (stage, name), p in sorted(PROF.items()):
    s = io.StringIO()
    st = pstats.Stats(p, stream=s)
    print(f"===== {stage} {name}: {st.total_calls} calls, {st.total_tt * 1e3:.2f} ms total")
    st.sort_stats("cumulative").print_stats(28)
    print("\n".join(l for l in s.getvalue().splitlines() if l.strip())[:6000])
