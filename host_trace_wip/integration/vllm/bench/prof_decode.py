# One profiled decode step per batch size: torch.profiler (CPU + CUDA, Python stacks) around 3 consecutive engine steps of a
# steady pure-decode batch, after the batch size has been seen (default: its graphs; V: its trace and binding).
#   python prof_decode.py --arm default|V --bs 8 64 --out-dir D
import argparse, json, os, sys, time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
p = argparse.ArgumentParser()
p.add_argument("--arm", choices=["default", "V", "eager", "VAf"], required=True)  # VP = V with ARMV_VP=1
p.add_argument("--bs", type=int, nargs="+", default=[8, 64])
p.add_argument("--out-dir", required=True)
p.add_argument("--model", default="/data/eellison/models/Qwen3-8B")
a = p.parse_args()
import torch
from torch.profiler import ProfilerActivity, profile, record_function
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

if a.arm == "VAf":
    sys.path.insert(0, os.environ["INTEG_ARMV"])
    import hostcuts

    _orig = GPUModelRunner.capture_model

    def _capture(self):
        hostcuts.install(self)
        return _orig(self)
    GPUModelRunner.capture_model = _capture

llm = LLM(model=a.model, tensor_parallel_size=1, enforce_eager=a.arm in ("V", "eager"), attention_backend="FLASHINFER", enable_prefix_caching=False,
          gpu_memory_utilization=float(os.environ.get("INTEG_GPU_UTIL", "0.6")), **({"kv_cache_memory_bytes": int(os.environ["INTEG_KV_BYTES"])} if os.environ.get("INTEG_KV_BYTES") else {}), max_model_len=4096, seed=0)
V = None
if a.arm == "V":
    sys.path.insert(0, os.environ["INTEG_ARMV"])
    import adapter
    import adapter_cpp

    runner = llm.llm_engine.model_executor.driver_worker.worker.model_runner
    V = (adapter_cpp.ArmVcpp if adapter_cpp.BOUND else adapter.ArmV)(runner)
    V.install()
g = torch.Generator().manual_seed(0)
prompt = lambda n: TokensPrompt(prompt_token_ids=torch.randint(1000, 100000, (n,), generator=g).tolist())
eng = llm.llm_engine
os.makedirs(a.out_dir, exist_ok=True)
info = {"arm": a.arm, "async_scheduling": getattr(llm.llm_engine.vllm_config.scheduler_config, "async_scheduling", None), "bs": {}}
for bs in a.bs:
    llm.generate([prompt(256) for _ in range(bs)], SamplingParams(max_tokens=24, ignore_eos=True, temperature=0.0), use_tqdm=False)  # this bs seen
    for i in range(bs):
        eng.add_request(f"p{bs}_{i}", prompt(256), SamplingParams(max_tokens=1000, ignore_eos=True, temperature=0.0))
    for _ in range(12):
        eng.step()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True, record_shapes=False) as prof:
        for k in range(3):
            with record_function(f"ENGINE_STEP_{k}"):
                eng.step()
        torch.cuda.synchronize()
    path = os.path.join(a.out_dir, f"decode_{bs}.json")
    prof.export_chrome_trace(path)
    eng.abort_request([f"p{bs}_{i}" for i in range(bs)])
    while eng.has_unfinished_requests():
        eng.step()
    info["bs"][bs] = path
    print("profiled", bs, path, flush=True)
if V is not None:
    info["V"] = {k: v for k, v in V.summary().items() if k in ("traces", "replays", "eager", "variants", "calls", "bound")}
json.dump(info, open(os.path.join(a.out_dir, "info.json"), "w"), indent=1, default=str)
