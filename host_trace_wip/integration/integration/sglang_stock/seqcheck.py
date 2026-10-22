# seqcheck.py: stock SGLang eager vs stock SGLang + the hook (adapter/adapter_hook.py ArmS), one process, one load.
# Each case is a fresh batch: prefill (one extend) then --decode-steps decode steps; per step the argmax tokens and a sha1
# of the logits bytes. Phases: E1 eager, E2 eager again (run-to-run determinism control), H hooked (counters per case).
# Compares H and E2 against E1. Recurrent state needs no snapshot: every case starts from fresh requests.
#   launcher seqcheck.py <sglang server args> --batch-size 1 --out X.json [--bs 1 8 64] [--prompt-len 128] [--prefill-lens 64 512]
import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "adapter"))
import adapter_hook  # noqa: E402  before sglang (the attention import)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import sglang.benchmark.one_batch as ob  # noqa: E402

p0 = argparse.ArgumentParser(add_help=False)
p0.add_argument("--out", required=True)
p0.add_argument("--bs", type=int, nargs="+", default=[1, 8])
p0.add_argument("--prompt-len", type=int, default=128)
p0.add_argument("--prefill-lens", type=int, nargs="*", default=[64, 512])
p0.add_argument("--ragged", type=int, nargs="*", default=[97, 333, 517])
p0.add_argument("--decode-steps", type=int, default=12)
p0.add_argument("--hook-reps", type=int, default=2, help="hooked repetitions of the case list (the second shows replays)")
mine, rest = p0.parse_known_args()
sys.argv = [sys.argv[0]] + rest
R = {"argv": rest, "phases": {}, "compare": {}, "errors": []}


def sha(t):
    return hashlib.sha1(t.float().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:16]


def cases():
    out = [(f"prefill_{T}", [T], 0) for T in mine.prefill_lens]
    if mine.ragged:
        out.append(("ragged", list(mine.ragged), 0))
    out += [(f"decode_{bs}", [mine.prompt_len] * bs, mine.decode_steps) for bs in mine.bs]
    return out


def run_case(mr, lens, steps, seed):
    rs = np.random.RandomState(seed)
    mr.clear()
    reqs = ob.prepare_synthetic_inputs_for_latency_test(len(lens), max(lens), [list(rs.randint(0, 10000, n)) for n in lens])
    nt, logits, batch = mr.extend(reqs)
    rec = {"tokens": [nt.tolist()], "sha": [sha(logits)]}
    for _ in range(steps):
        nt, logits = mr.decode(nt, batch)
        rec["tokens"].append(nt.tolist())
        rec["sha"].append(sha(logits))
    torch.cuda.synchronize()
    return rec


def my_test(server_args, port_args, bench_args, gpu_id, tp_rank):
    ob.publish(server_args, role="scheduler", ranks=ob.SpawnRanks(world_rank=ob.spawn_world_rank(server_args, tp_rank=tp_rank, pp_rank=0), gpu_id=gpu_id))
    ob.initialize_moe_config(); ob.initialize_fp8_gemm_config(); ob.initialize_fp4_gemm_config()
    t = time.perf_counter()
    mr, _ = ob.load_model(server_args, port_args, gpu_id, tp_rank)
    R["load_seconds"] = time.perf_counter() - t
    inner = mr.torch_runner
    R["max_total_num_tokens"] = inner.max_total_num_tokens
    R["attention_backends"] = [type(b).__name__ for b in adapter_hook._backends(inner.attn_backend)]
    cs = cases()
    for ph in ("E1", "E2"):
        R["phases"][ph] = {}
        for k, (name, lens, steps) in enumerate(cs):
            try:
                R["phases"][ph][name] = run_case(mr, lens, steps, 1000 + k)
            except Exception:
                R["errors"].append(f"{ph} {name}: " + __import__("traceback").format_exc()[-2500:])
    arm = adapter_hook.ArmS(inner)
    R["static_slots"] = len(arm.obj_slots)
    arm.install()
    for rep in range(mine.hook_reps):
        ph = f"H{rep}"
        R["phases"][ph] = {}
        for k, (name, lens, steps) in enumerate(cs):
            c0 = arm.counters()
            t0 = time.perf_counter()
            try:
                rec = run_case(mr, lens, steps, 1000 + k)
            except Exception:
                R["errors"].append(f"{ph} {name}: " + __import__("traceback").format_exc()[-2500:])
                continue
            c1 = arm.counters()
            rec["counters"] = {kk: c1[kk] - c0[kk] for kk in c1 if kk != "calls"}
            rec["seconds"] = time.perf_counter() - t0
            R["phases"][ph][name] = rec
            print(ph, name, rec["counters"], flush=True)
    arm.uninstall()
    for ph in ["E2"] + [f"H{r}" for r in range(mine.hook_reps)]:
        cmp = {}
        for name, ref in R["phases"]["E1"].items():
            got = R["phases"].get(ph, {}).get(name)
            if got is None:
                cmp[name] = "missing"
                continue
            cmp[name] = {"tokens": f"{sum(a == b for a, b in zip(ref['tokens'], got['tokens']))}/{len(ref['tokens'])}",
                         "sha": f"{sum(a == b for a, b in zip(ref['sha'], got['sha']))}/{len(ref['sha'])}"}
        R["compare"][f"{ph}_vs_E1"] = cmp
    R["summary"] = arm.summary()
    R["peak_mib"] = torch.cuda.max_memory_allocated() / 2**20
    with open(mine.out, "w") as f:
        json.dump(R, f, indent=1, default=str)
    print("COMPARE", json.dumps(R["compare"]), flush=True)
    print("WROTE", mine.out, flush=True)


ob.latency_test = my_test
ob.cli_main()
