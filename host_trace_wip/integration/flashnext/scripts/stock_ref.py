"""Stock vLLM eager reference run (no host trace): every model forward's output digest and NaN flag, for decode, prefill
and mixed steps driven through the engine one step at a time. Three processes of this with the same arguments must
give the same digests and no NaN for the references to count as clean (DECISIONS 2026-10-12).

    python_fn.sh scripts/stock_ref.py --model M --out out/X.json [--llm-kw JSON] [--decode-bs 1 8 64] ...

Flash-Next lane copy of trtllm_moe_cpp/scripts/stock_ref.py: MiMo backport removed; max len 16384 (user-approved for every
arm); --long-decode (decode at long KV: QSA's indexer selects 2048 of the KV only past 2048 tokens); prints the resolved
compile / cudagraph / MoE / GDN-prefill configuration.
"""

import argparse
import json
import os
import sys
import time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
import torch
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--out", required=True)
p.add_argument("--llm-kw", default="{}")
p.add_argument("--max-model-len", type=int, default=16384)
p.add_argument("--decode-bs", type=int, nargs="*", default=[1, 8, 64])
p.add_argument("--decode-prompt", type=int, default=256)
p.add_argument("--decode-steps", type=int, default=8)
p.add_argument("--long-decode", nargs="*", default=["4x6000"], help="BSxPROMPT: BS requests of PROMPT tokens, then --decode-steps decode steps")
p.add_argument("--prefill-lens", type=int, nargs="*", default=[64, 512, 2048, 8192])
p.add_argument("--mixed", nargs="*", default=["8x128", "32x512"], help="NDxPL: ND decoders, then one PL-token prompt per step, 3 steps")
p.add_argument("--enforce-eager", type=int, default=1)
p.add_argument("--quality", nargs="*", default=[], help="text files: greedy generations on 5 prompts + perplexity of each one's first 2048 tokens")
p.add_argument("--qsa-det", action="store_true", help="DIAGNOSIS CONFIG: deterministic QSA top-k order (scripts/qsa_det.py)")
a = p.parse_args()
if a.qsa_det:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import qsa_det

    qsa_det.install()

kw = dict(model=a.model, tensor_parallel_size=1, enforce_eager=bool(a.enforce_eager), enable_prefix_caching=False,
          gpu_memory_utilization=float(os.environ.get("INTEG_GPU_UTIL", "0.85")), max_model_len=a.max_model_len, seed=0,
          trust_remote_code=True)
kw.update(json.loads(a.llm_kw))
t0 = time.perf_counter()
llm = LLM(**kw)
init_s = time.perf_counter() - t0
eng = llm.llm_engine
runner = eng.model_executor.driver_worker.worker.model_runner
cfg = eng.vllm_config
CONFIG = {"compilation_mode": str(cfg.compilation_config.mode), "cudagraph_mode": str(cfg.compilation_config.cudagraph_mode),
          "max_num_seqs": cfg.scheduler_config.max_num_seqs, "max_num_batched_tokens": cfg.scheduler_config.max_num_batched_tokens,
          "gdn_prefill_backend": __import__("vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn", fromlist=["x"])._resolve_gdn_prefill_backend(cfg),
          "kv_cache_dtype": str(cfg.cache_config.cache_dtype)}
print("CONFIG", CONFIG, flush=True)
ROWS: list[dict] = []
PHASE = ["init"]
_fwd = runner.model.forward


def fwd(input_ids, positions, *args, **kw):
    out = _fwd(input_ids, positions, *args, **kw)
    h = out[0] if isinstance(out, tuple) else out
    if PHASE[0] != "init" and isinstance(h, torch.Tensor):
        b = h.detach().contiguous().view(h.shape[0], -1)
        bits = b.view(torch.int16).to(torch.int64) if b.element_size() == 2 else b.view(torch.int32).to(torch.int64)
        rows = (bits * torch.arange(1, bits.shape[1] + 1, device=bits.device)).sum(dim=1)
        nanr = torch.isnan(h.view(h.shape[0], -1)).any(dim=1)
        nidx = nanr.nonzero().flatten()
        ROWS.append({"phase": PHASE[0], "tokens": int(h.shape[0]), "nan": bool(torch.isnan(h).any()),
                     "nan_rows_n": int(nidx.numel()), "nan_rows_idx": nidx[:16].tolist(), "shape": list(h.shape),
                     "nan_cols_n": int(torch.isnan(h.view(h.shape[0], -1)).any(dim=0).sum()),
                     "inf": bool(torch.isinf(h).any()), "digest": int(rows.sum()), "rows": rows.tolist()})
    return out


runner.model.forward = fwd
g = torch.Generator().manual_seed(0)


def prompt(n):
    return TokensPrompt(prompt_token_ids=torch.randint(1000, 100000, (n,), generator=g).tolist())


# requests end by length (a few steps past each workload), so no abort is needed between workloads
SP = SamplingParams(temperature=0.0, max_tokens=a.decode_steps + 4, ignore_eos=True)
rid = [0]


def add(n, sp=SP):
    rid[0] += 1
    eng.add_request(str(rid[0]), prompt(n), sp)
    return str(rid[0])


def drain():
    while eng.has_unfinished_requests():
        eng.step()


toks = {}
decodes = [(bs, a.decode_prompt, f"{bs}") for bs in a.decode_bs]
decodes += [(int(s.split("x")[0]), int(s.split("x")[1]), s) for s in a.long_decode]
for bs, plen, tag in decodes:
    PHASE[0] = f"decode_fill{tag}"
    for _ in range(bs):
        add(plen)
    while True:  # until every request decodes
        outs = eng.step()
        if ROWS and ROWS[-1]["tokens"] == bs:
            break
    PHASE[0] = f"decode{tag}"
    for _ in range(a.decode_steps):
        for o in eng.step():
            toks.setdefault(f"decode{tag}", {})[o.request_id] = list(o.outputs[0].token_ids)
    drain()
for L in a.prefill_lens:
    PHASE[0] = f"prefill{L}"
    add(L, SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True))
    while eng.has_unfinished_requests():
        eng.step()
for spec in a.mixed:
    nd, pl = map(int, spec.split("x"))
    PHASE[0] = f"mixed_fill{spec}"
    for _ in range(nd):
        add(256)
    while True:
        eng.step()
        if ROWS and ROWS[-1]["tokens"] == nd:
            break
    PHASE[0] = f"mixed{spec}"
    for _ in range(3):
        add(pl)
        eng.step()
    drain()

quality = None
if a.quality:  # model-quality sanity (not recorded in ROWS): coherent greedy text and a plain-text perplexity
    PHASE[0] = "init"
    prompts = ["The capital of France is", "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
               "Water boils at sea level at a temperature of", "The three primary colors are",
               "Translate to French: 'Good morning, how are you?'\nFrench:"]
    gens = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=48), use_tqdm=False)
    tok = llm.get_tokenizer()
    quality = {"generations": [[p, g.outputs[0].text] for p, g in zip(prompts, gens)], "ppl": []}
    for p_, t_ in quality["generations"]:
        print("GEN", repr(p_), "->", repr(t_), flush=True)
    for path in a.quality:
        ids = tok(open(path).read(), add_special_tokens=False)["input_ids"][:2048]
        out = llm.generate([TokensPrompt(prompt_token_ids=ids)], SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0),
                           use_tqdm=False)[0]
        lps = [d[t].logprob for t, d in zip(ids[1:], out.prompt_logprobs[1:])]
        bad = [i for i, x in enumerate(lps) if x != x]
        print("PPL_NAN_POSITIONS", path, len(bad), bad[:32], flush=True)
        nll = -sum(lps) / len(lps)
        quality["ppl"].append({"file": path, "tokens": len(ids), "nll": nll, "ppl": float(torch.tensor(nll).exp())})
        print("PPL", path, "tokens", len(ids), "nll", round(nll, 4), "ppl", round(quality["ppl"][-1]["ppl"], 3), flush=True)

res = {"model": a.model, "config": CONFIG, "quality": quality, "init_s": init_s, "llm_kw": kw, "moe_backend": str(cfg.kernel_config.moe_backend) if hasattr(cfg, "kernel_config") else None,
       "rows": ROWS, "tokens": toks, "nan_rows": sum(r["nan"] for r in ROWS), "forwards": len(ROWS)}
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
with open(a.out, "w") as f:
    json.dump(res, f)
print("STOCK_REF", a.out, "forwards", len(ROWS), "nan", res["nan_rows"], "init_s", round(init_s, 1), flush=True)
os._exit(0)
