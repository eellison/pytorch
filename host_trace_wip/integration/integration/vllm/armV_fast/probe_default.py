# What the default arm (FULL_AND_PIECEWISE) runs for a short prefill: python_vllm.sh armV/probe_default.py [T ...]
# Per forward: CUDAGraphWrapper calls (replay vs runnable), host time in the eager split ops (FlashInfer forward and KV
# update), forward wall and GPU time; then one torch.profiler step, top CPU self time.
import collections, os, sys, time
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
import torch
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from vllm.compilation.cuda_graph import CUDAGraphWrapper
from vllm.v1.attention.backends.flashinfer import FlashInferImpl
from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
from vllm.forward_context import get_forward_context

ST = collections.defaultdict(float)
CNT = collections.Counter()


def timed(cls, name, tag):
    f = getattr(cls, name)

    def w(*a, **k):
        t = time.perf_counter()
        try:
            return f(*a, **k)
        finally:
            ST[tag] += time.perf_counter() - t
            CNT[tag] += 1
    setattr(cls, name, w)


timed(FlashInferImpl, "forward", "fi_forward")
timed(FlashInferImpl, "do_kv_cache_update", "kv_update")
orig_call = CUDAGraphWrapper.__call__


def gw(self, *a, **k):
    fc = get_forward_context()
    mode = fc.cudagraph_runtime_mode
    replay = mode == self.runtime_mode and mode.name != "NONE"
    CNT[f"wrapper_{self.runtime_mode.name}_{'replay' if replay else 'runnable'}_ctx{mode.name}"] += 1
    t = time.perf_counter()
    try:
        return orig_call(self, *a, **k)
    finally:
        ST[f"wrapper_{self.runtime_mode.name}"] += time.perf_counter() - t


CUDAGraphWrapper.__call__ = gw
orig_pw = CudaGraphManager.run_pw_graph


def pw(self, model, inputs):
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t = time.perf_counter()
    e0.record()
    out = orig_pw(self, model, inputs)
    e1.record()
    host = time.perf_counter() - t
    torch.cuda.synchronize()
    ST["pw_host"] += host
    ST["pw_wall"] += time.perf_counter() - t
    ST["pw_gpu"] += e0.elapsed_time(e1) / 1e3
    CNT["pw"] += 1
    CNT[f"pw_breakable_{self.use_breakable_cg}"] += 1
    return out


CudaGraphManager.run_pw_graph = pw
llm = LLM(model="/data/eellison/models/Qwen3-8B", attention_backend="FLASHINFER", enable_prefix_caching=False,
          gpu_memory_utilization=0.6, max_model_len=4096, seed=0)
g = torch.Generator().manual_seed(0)
P = lambda L: [TokensPrompt(prompt_token_ids=torch.randint(1000, 100000, (L,), generator=g).tolist())]
sp = SamplingParams(max_tokens=1, ignore_eos=True, temperature=0.0)
llm.generate(P(128), sp, use_tqdm=False)
for L in [int(x) for x in sys.argv[1:]] or [64, 512]:
    for _ in range(3):
        llm.generate(P(L), sp, use_tqdm=False)
    ST.clear(); CNT.clear()
    for _ in range(5):
        llm.generate(P(L), sp, use_tqdm=False)
    n = max(CNT["pw"], 1)
    print(f"T={L} per forward (n={CNT['pw']}):", {k: round(v / n * 1e3, 3) for k, v in sorted(ST.items())}, "ms;",
          {k: v / n for k, v in sorted(CNT.items())}, flush=True)
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        llm.generate(P(L), sp, use_tqdm=False)
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=25), flush=True)
