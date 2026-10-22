# brk_bitwise.py OUT ARM (breakable | fullnc): one-request prefills of every length 33..80 (each its own step: PIECEWISE capture sizes 40,
# 48, 56, 64, 72, 80 pad lengths in between), max_tokens 1, top-20 logprobs; prompts seeded identically in both arms. Compare the two
# outputs with --compare A B. max_num_seqs 64: capture range min(2 x 64, 1024) = 128 tokens covers 33..80 (8 gave 16: all eager). Report-only check of vLLM's breakable graphs (Qwen's GDN op is captured inside its segments).
import json, os, sys
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
if sys.argv[1] == "--compare":
    a, b = (json.load(open(f)) for f in sys.argv[2:4])
    for L in a:
        same_tok = a[L]["token"] == b[L]["token"]
        same_lp = a[L]["logprobs"] == b[L]["logprobs"]
        print(L, "token", "same" if same_tok else f"DIFF {a[L]['token']} vs {b[L]['token']}", "| top-20 logprobs", "bitwise" if same_lp else
              f"DIFF max |d| {max(abs(x - y) for x, y in zip(sorted(a[L]['logprobs'].values()), sorted(b[L]['logprobs'].values()))):.3g}")
    sys.exit(0)
import torch
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

out, arm = sys.argv[1], sys.argv[2]
cc = {"mode": 0, "cudagraph_mode": "FULL_AND_PIECEWISE"} if arm == "breakable" else {"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"}
llm = LLM(model="/data/eellison/models/Qwen3.8-27B-NVFP4", language_model_only=True, max_model_len=4096, kv_cache_dtype="fp8", max_num_seqs=64,
          gdn_prefill_backend="triton", compilation_config=cc, enable_prefix_caching=False, gpu_memory_utilization=0.5,
          kv_cache_memory_bytes=16 << 30, seed=0)
g = torch.Generator().manual_seed(7)
res = {}
for L in range(33, 81):
    ids = torch.randint(1000, 100000, (L,), generator=g).tolist()
    o = llm.generate([TokensPrompt(prompt_token_ids=ids)], SamplingParams(max_tokens=1, temperature=0.0, logprobs=20), use_tqdm=False)[0].outputs[0]
    res[L] = {"token": o.token_ids[0], "logprobs": {str(t): lp.logprob for t, lp in o.logprobs[0].items()}}
json.dump(res, open(out, "w"))
print("done", arm, len(res))
