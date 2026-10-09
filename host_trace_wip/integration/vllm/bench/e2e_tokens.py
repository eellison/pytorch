# End-to-end greedy tokens per arm (VP vs V, VAf vs default): python e2e_tokens.py --arm V|VAf|default|eager --out X.json
# Same prompts (seed 0, 128 tokens) at bs 1/8/64/128, 32 greedy tokens each; arm VP = --arm V with ARMV_VP=1.
import argparse, json, os, sys

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
p = argparse.ArgumentParser()
p.add_argument("--arm", choices=["eager", "default", "V", "VAf", "fullnc", "fullnc_full"], required=True)
p.add_argument("--out", required=True)
p.add_argument("--bs", type=int, nargs="+", default=[1, 8, 64, 128])
p.add_argument("--model", default="/data/eellison/models/Qwen3-8B")
a = p.parse_args()
import torch
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

sys.path.insert(0, os.environ["INTEG_ARMV"])
if a.arm == "VAf":
    import hostcuts

    _orig = GPUModelRunner.capture_model

    def capture(self):
        hostcuts.install(self)
        return _orig(self)
    GPUModelRunner.capture_model = capture
extra = {"compilation_config": {"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY" if a.arm == "fullnc" else "FULL"}} if a.arm.startswith("fullnc") else {}
llm = LLM(model=a.model, tensor_parallel_size=1, enforce_eager=a.arm in ("eager", "V"), **extra, attention_backend="FLASHINFER", enable_prefix_caching=False,
          gpu_memory_utilization=float(os.environ.get("INTEG_GPU_UTIL", "0.6")), **({"kv_cache_memory_bytes": int(os.environ["INTEG_KV_BYTES"])} if os.environ.get("INTEG_KV_BYTES") else {}), max_model_len=4096, seed=0)
V = None
if a.arm == "V":
    import adapter
    import adapter_cpp

    runner = llm.llm_engine.model_executor.driver_worker.worker.model_runner
    V = (adapter_cpp.ArmVcpp if adapter_cpp.BOUND else adapter.ArmV)(runner)
    V.install()
g = torch.Generator().manual_seed(0)
R = {"arm": a.arm, "vp": os.environ.get("ARMV_VP") == "1", "tokens": {}}
for bs in a.bs:
    prompts = [TokensPrompt(prompt_token_ids=torch.randint(1000, 100000, (128,), generator=g).tolist()) for _ in range(bs)]
    for rep in (0, 1):  # rep 1: every key seen (V: replays)
        outs = llm.generate(prompts, SamplingParams(max_tokens=32, ignore_eos=True, temperature=0.0), use_tqdm=False)
        R["tokens"][f"{bs}_{rep}"] = [list(o.outputs[0].token_ids) for o in outs]
if V is not None:
    R["V"] = {k: v for k, v in V.summary().items() if k in ("traces", "replays", "eager", "variants", "calls", "bound", "bad_keys")}
json.dump(R, open(a.out, "w"))
print("E2E", a.arm, R["vp"], {k: len(v) for k, v in R["tokens"].items()}, R.get("V"), flush=True)
