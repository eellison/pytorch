# Startup to ready, per arm: wall seconds from process start (STARTUP_T0, the shell's clock just before exec) to the end
# of imports, of model load (SGLang's own graph capture is inside load for arm A; its seconds are split out), and of
# our warm-up: per shape class (prefill T, decode bs) the first call's seconds, then calls until two in a row neither
# trace nor run eagerly (driveF.py's warm rule, at most 10), with the counter deltas. "ready" = every class quiet.
# STARTUP_ARM: A (SGLang default graphs) | D (eager) | F | Fc | P | Pc | FT | M | AT | DT (T / M: SGLang's bf16 GEMM
# backend initialized, i.e. TGV). STARTUP_SAVE=path: HarvestProvider.save after warm-up; STARTUP_LOAD=path: load before
# the first call ("warm" = our persisted bindings reused).
#   launcher startup_sgl.py <sglang server args> --batch-size <bs...> --prefill-lens <T...> --out X.json
import argparse
import json
import os
import sys
import time

T0 = float(os.environ.get("STARTUP_T0", time.time()))
ARM = os.environ["STARTUP_ARM"]
A = "/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/sglang/armF"
C2 = "/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration/adapters_c2"
sys.path[:0] = [A, f"{A}/prep"]
MOD = {"F": (A, "adapter", "ArmF"), "Fc": (A, "adapter_cpp", "ArmF"), "P": (f"{A}/prep", "adapter_prep", "PrepF"),
       "Pc": (C2, "adapter_prep_cpp", "PrepF"), "M": (C2, "adapter_prep_cpp", "PrepF"), "FT": (f"{C2}/tgv", "adapter", "ArmF")}
if ARM == "M":
    os.environ["PREP_M"] = "1"
if ARM == "FT":
    os.environ.setdefault("ARMF_TGV", "1")
    os.environ.setdefault("ARMF_STATIC", "1")
if ARM in ("P", "Pc", "M"):
    os.environ.setdefault("ARMF_PREP", "1")
adapter = None
if ARM in MOD:
    d, name, _ = MOD[ARM]
    sys.path.insert(0, d)
    adapter = __import__(name)  # before sglang: arms the CuTe route before any CuTe compile
import numpy as np
import torch
import sglang.benchmark.one_batch as ob
from sglang.srt.model_executor.model_runner import ModelRunner

p0 = argparse.ArgumentParser(add_help=False)
p0.add_argument("--out", required=True)
p0.add_argument("--prefill-lens", type=int, nargs="+", default=[64, 512, 2048])
mine, rest = p0.parse_known_args()
sys.argv = [sys.argv[0]] + rest
R = {"arm": ARM, "argv": rest, "t_imports": time.time() - T0, "capture": None, "classes": {}}

orig_init = ModelRunner.init_cuda_graphs


def timed_init(self, *a, **k):
    torch.cuda.synchronize()
    t = time.time()
    out = orig_init(self, *a, **k)
    torch.cuda.synchronize()
    R["capture"] = {"seconds": time.time() - t, "start": t - T0}
    return out


ModelRunner.init_cuda_graphs = timed_init


def my_test(server_args, port_args, bench_args, gpu_id, tp_rank):
    ob.publish(server_args, role="scheduler", ranks=ob.SpawnRanks(world_rank=ob.spawn_world_rank(server_args, tp_rank=tp_rank, pp_rank=0), gpu_id=gpu_id))
    ob.initialize_moe_config(); ob.initialize_fp8_gemm_config(); ob.initialize_fp4_gemm_config()
    if ARM in ("FT", "M", "AT", "DT"):
        from sglang.srt.layers.quantization import unquant
        unquant.initialize_bf16_gemm_config()
        R["bf16_gemm_backend"] = str(unquant.get_bf16_gemm_backend())
    R["t_load_start"] = time.time() - T0
    mr, _ = ob.load_model(server_args, port_args, gpu_id, tp_rank)
    torch.cuda.synchronize()
    R["t_load_end"] = time.time() - T0
    inner = mr.torch_runner
    R["max_total_num_tokens"] = inner.max_total_num_tokens
    arm = None
    if adapter is not None:
        arm = getattr(adapter, MOD[ARM][2])(inner)
        if os.environ.get("STARTUP_LOAD"):
            R["loaded_bindings"] = arm.provider.load(os.environ["STARTUP_LOAD"])
        arm.install()
    R["t_installed"] = time.time() - T0
    counters = arm.counters if arm else (lambda: {"traces": 0, "replays": 0, "eager": 0, "variants": 0})
    np.random.seed(0)

    def prefill(lens):
        mr.clear()
        reqs = ob.prepare_synthetic_inputs_for_latency_test(len(lens), max(lens), [list(np.random.randint(0, 10000, n)) for n in lens])
        nt, _, batch = mr.extend(reqs)
        return nt, batch

    def warm(name, fn):
        calls, quiet, first = [], 0, None
        c0 = counters()
        for i in range(10):
            c = counters()
            torch.cuda.synchronize()
            t = time.time()
            fn()
            torch.cuda.synchronize()
            calls.append(time.time() - t)
            d = counters()
            quiet = quiet + 1 if d["traces"] == c["traces"] and d["eager"] == c["eager"] else 0
            if i >= 1 and quiet >= 2:
                break
        c1 = counters()
        R["classes"][name] = {"first_s": calls[0], "calls_s": calls, "warm_s": sum(calls), "n_calls": len(calls),
                              **{k: c1[k] - c0[k] for k in ("traces", "replays", "eager", "variants")}}
        print("class", name, json.dumps(R["classes"][name]), flush=True)

    for T in mine.prefill_lens:
        warm(f"prefill_{T}", lambda: prefill([T]))
    for bs in bench_args.batch_size:
        nt, batch = prefill([256] * bs)
        state = {"nt": nt}

        def step():
            state["nt"], _ = mr.decode(state["nt"], batch)

        warm(f"decode_{bs}", step)
    torch.cuda.synchronize()
    R["t_ready"] = time.time() - T0
    R["warmup_s"] = sum(c["warm_s"] for c in R["classes"].values())
    if arm is not None:
        if os.environ.get("STARTUP_SAVE"):
            R["saved_bindings"] = arm.provider.save(os.environ["STARTUP_SAVE"])
        s = arm.summary()
        R["summary"] = {k: s.get(k) for k in ("traces", "replays", "eager", "variants", "harvests", "bindings", "refused", "decline_sites")}
    R["peak"] = {"max_allocated_mib": torch.cuda.max_memory_allocated() / 2**20, "reserved_mib": torch.cuda.memory_reserved() / 2**20}
    with open(mine.out, "w") as f:
        json.dump(R, f, indent=1, default=str)
    print("WROTE", mine.out, "ready", R["t_ready"], flush=True)


ob.latency_test = my_test
ob.cli_main()
