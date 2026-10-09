# Stock SGLang eager smoke + structure dump for a model (no hook): load once, run a few eager prefill / decode steps
# (logits digests, argmax runs to spot degenerate output), and dump what a stock hook arm must pass as arguments:
# the attention backend tree, each sub-backend's forward_metadata fields after a prefill and after a decode step
# (CUDA tensors, CPU tensors, ints, other), the ForwardBatch fields, and the KV / mamba pools' CUDA tensors.
#   launcher [lockwrap.py] q38_probe.py <sglang server args> --batch-size 1 --out X.json
import argparse
import dataclasses
import hashlib
import json
import sys
import time

import numpy as np
import torch
import sglang.benchmark.one_batch as ob

p0 = argparse.ArgumentParser(add_help=False)
p0.add_argument("--out", required=True)
p0.add_argument("--prefill-lens", type=int, nargs="+", default=[64, 512])
p0.add_argument("--decode-bs", type=int, nargs="+", default=[1, 8])
p0.add_argument("--decode-steps", type=int, default=12)
mine, rest = p0.parse_known_args()
sys.argv = [sys.argv[0]] + rest
R = {"argv": rest, "phases": {}, "errors": []}
MB = 2**20


def kind(v):
    if isinstance(v, torch.Tensor):
        return f"{'cuda' if v.is_cuda else 'cpu'}{'(pinned)' if (not v.is_cuda and v.is_pinned()) else ''} {str(v.dtype).replace('torch.', '')} {list(v.shape)}"
    if isinstance(v, (bool, int, float, str)) or v is None:
        return f"{type(v).__name__} {v!r}"[:60]
    if isinstance(v, (list, tuple)):
        inner = sorted({kind(x).split(' ')[0] for x in v[:8]})
        return f"{type(v).__name__}[{len(v)}] of {inner}"
    return type(v).__name__


def fields(obj, depth=0):
    out = {}
    items = [(f.name, getattr(obj, f.name, None)) for f in dataclasses.fields(obj)] if dataclasses.is_dataclass(obj) else list(vars(obj).items()) if hasattr(obj, "__dict__") else []
    for n, v in items:
        out[n] = kind(v)
        if depth < 1 and (dataclasses.is_dataclass(v) or (hasattr(v, "__dict__") and not isinstance(v, (torch.Tensor, torch.nn.Module)) and type(v).__module__.startswith("sglang"))):
            out[n] = {"_type": type(v).__name__, **fields(v, depth + 1)}
    return out


def backends(b):
    subs = [b] + list(getattr(b, "attn_backend_list", []) or [])
    for name in ("full_attn_backend", "linear_attn_backend", "primary", "secondary"):
        s = getattr(b, name, None)
        if s is not None and s not in subs:
            subs.append(s)
    return subs


def dump_metadata(inner, tag):
    d = {}
    for i, sb in enumerate(backends(inner.attn_backend)):
        md = getattr(sb, "forward_metadata", "<no attribute>")
        d[f"{i}:{type(sb).__name__}"] = {"_md_type": type(md).__name__, **(fields(md) if md is not None and not isinstance(md, str) else {})}
    R["phases"].setdefault("metadata", {})[tag] = d


def digest(t):
    return hashlib.sha1(t.float().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:16]


def my_test(server_args, port_args, bench_args, gpu_id, tp_rank):
    ob.publish(server_args, role="scheduler", ranks=ob.SpawnRanks(world_rank=ob.spawn_world_rank(server_args, tp_rank=tp_rank, pp_rank=0), gpu_id=gpu_id))
    ob.initialize_moe_config(); ob.initialize_fp8_gemm_config(); ob.initialize_fp4_gemm_config()
    t = time.perf_counter()
    mr, _ = ob.load_model(server_args, port_args, gpu_id, tp_rank)
    R["load_seconds"] = time.perf_counter() - t
    inner = mr.torch_runner
    R["max_total_num_tokens"] = inner.max_total_num_tokens
    R["attention_backends"] = [type(b).__name__ for b in backends(inner.attn_backend)]
    R["model_class"] = type(inner.model).__name__
    R["mem_after_load_mib"] = torch.cuda.memory_allocated() / MB
    R["pools"] = {}
    for pn in ("token_to_kv_pool", "req_to_token_pool"):
        pool = getattr(inner, pn, None)
        R["pools"][pn] = {"_type": type(pool).__name__, **{k: kind(v) for k, v in vars(pool).items() if isinstance(v, (torch.Tensor, list, tuple)) or dataclasses.is_dataclass(v) or (hasattr(v, "__dict__") and type(v).__module__.startswith("sglang"))}} if pool is not None else None
    for k in ("server_args",):
        pass
    np.random.seed(0)
    orig_fwd = type(inner).forward
    seen = {}

    def spy(self, forward_batch, *a, **k):
        tag = ("decode" if forward_batch.forward_mode.is_decode() else "extend")
        out = orig_fwd(self, forward_batch, *a, **k)
        if tag not in seen:
            seen[tag] = True
            R["phases"].setdefault("forward_batch", {})[tag] = fields(forward_batch)
            dump_metadata(self, tag)
        return out

    type(inner).forward = spy
    try:
        for T in mine.prefill_lens:
            mr.clear()
            reqs = ob.prepare_synthetic_inputs_for_latency_test(1, T, [list(np.random.randint(0, 10000, T))])
            nt, logits, _ = mr.extend(reqs)
            torch.cuda.synchronize()
            R["phases"][f"prefill_{T}"] = {"digest": digest(logits), "argmax": nt.tolist()}
        for bs in mine.decode_bs:
            mr.clear()
            reqs = ob.prepare_synthetic_inputs_for_latency_test(bs, 128, [list(np.random.randint(0, 10000, 128)) for _ in range(bs)])
            nt, logits, batch = mr.extend(reqs)
            toks = [nt.tolist()]
            for _ in range(mine.decode_steps):
                nt, logits = mr.decode(nt, batch)
                toks.append(nt.tolist())
            torch.cuda.synchronize()
            rows = list(zip(*toks))
            R["phases"][f"decode_{bs}"] = {"tokens_row0": list(rows[0]), "distinct_per_row": [len(set(r)) for r in rows], "last_digest": digest(logits)}
    except Exception as e:  # keep what was dumped
        import traceback

        R["errors"].append(traceback.format_exc()[-3000:])
    finally:
        type(inner).forward = orig_fwd
    R["peak_mib"] = torch.cuda.max_memory_allocated() / MB
    with open(mine.out, "w") as f:
        json.dump(R, f, indent=1, default=str)
    print("WROTE", mine.out, flush=True)


ob.latency_test = my_test
ob.cli_main()
