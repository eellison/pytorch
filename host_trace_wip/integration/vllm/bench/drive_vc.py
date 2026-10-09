# Baseline driver for vLLM's own arms (no host trace), mirroring sglang/bench/drive.py:
# decode: bs requests of 256 prompt tokens, 24 steps, steps[4:]; prefill: one request of L tokens, 7 reps, reps[2:].
# step = V2 GPUModelRunner.execute_model (input prep + metadata + forward, not sampling), synced;
# fwd = the forward alone (model call, run_pw_graph or run_fullgraph), synced.
# python_vllm.sh bench/drive.py --arm eager|default --out X.json; arm V (host trace, armV/adapter.py) runs through python_vllm_v.sh.
# mixed: 8 requests decoding, then one 128-token request added per step (decode + prefill in one forward), 6 rounds.
import argparse, json, os, statistics, sys, time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
import torch
if os.environ.get("INTEG_NUMEL_PROFILE"):  # drive_vc: the Python caller of any numel()/sizes()/stride() that raises (builtin c_exception events)
    import threading, traceback
    _seen = set()

    def _numel_prof(frame, event, arg):
        if event == "c_exception" and getattr(arg, "__name__", "") in ("numel", "size", "stride", "dim", "nelement", "sym_numel", "data_ptr"):
            st = "".join(traceback.format_stack(frame, limit=25))
            if st not in _seen and len(_seen) < 8:
                _seen.add(st)
                print(f"NUMEL_TRAP {arg.__name__} raised; Python caller stack:\n{st}", flush=True)
    sys.setprofile(_numel_prof)
    threading.setprofile(_numel_prof)
if os.environ.get("INTEG_TRAP_NUMEL"):  # diagnostics: the Python stack of a numel() on a symbolic tensor
    import traceback as _tb

    _numel = torch.Tensor.numel

    def _trap_numel(self):
        try:
            return _numel(self)
        except RuntimeError:
            print("TRAP_NUMEL stack:\n" + "".join(_tb.format_stack(limit=40)), flush=True)
            raise
    torch.Tensor.numel = _trap_numel
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager, ModelCudaGraphManager
from vllm.v1.worker.gpu.model_runner import GPUModelRunner  # model runner V2, v0.29's default

p = argparse.ArgumentParser()
p.add_argument("--arm", choices=["eager", "default", "V", "fullnc", "fullnc_full", "VAf", "Vc", "compnc"], required=True)  # Vc: V around vLLM's compiled forward (compile ON, cudagraph_mode NONE); compnc: that compiled forward alone  # VAf: default + armV hostcuts (installed before capture); VP = V with ARMV_VP=1  # fullnc: FULL decode cudagraphs of the uncompiled model
p.add_argument("--dump-decode", default=None, help="torch.save the compute_logits inputs (bf16) of every pure decode step")
p.add_argument("--check", action="store_true", help="arm V: compare every traced forward against eager (slow; not for timing)")
p.add_argument("--mixed", type=int, default=8, help="mixed rounds (0: skip)")
p.add_argument("--max-model-len", type=int, default=4096)
p.add_argument("--ix-mixed", nargs="*", default=[], help="drive_ix: InferenceX-shaped mixed steps PxND: one P-token prefill chunk + ND decoders at KV --ix-mixed-kv")
p.add_argument("--ix-mixed-kv", type=int, nargs=2, default=[4096, 8000], help="decoder prompt lengths spread evenly over [lo, hi]")
p.add_argument("--ix-decode", nargs="*", default=[], help="drive_ix: pure decode BS:LO-HI (decoder prompt lengths spread over [LO, HI])")
p.add_argument("--ix-reps", type=int, default=6, help="measured steps = reps - 2 (the first two are dropped)")
p.add_argument("--prof-mixed", default="", help="drive_prof: torch.profiler trace of the measured mixed steps into this dir")
p.add_argument("--stamps", action="store_true", help="host time of execute_model's sections per step (unsynced inside the step)")
p.add_argument("--stamps-fine", action="store_true", help="with --stamps: also prepare_inputs' and add_requests' callees")
p.add_argument("--out", required=True)
p.add_argument("--launch-stamps", action="store_true", help="pre-launch window: execute_model entry -> first cudaGraphLaunch entry (LD_PRELOAD launchshim.so); no syncs inside the step")
p.add_argument("--profile-decode", type=int, nargs="*", default=[], help="Kineto-profile 3 single decode steps at these batch sizes")
p.add_argument("--profile-dir", default=None)
p.add_argument("--batch-size", type=int, nargs="+", default=[1, 8, 64])
p.add_argument("--decode-tokens", type=int, default=25)
p.add_argument("--prefill-lens", type=int, nargs="+", default=[64, 512, 2048])
p.add_argument("--prefill-reqs", type=int, nargs="*", default=[], help="multi-request prefill: 6 rounds of N prompts of --prefill-req-len")
p.add_argument("--prefill-req-len", type=int, default=128)
p.add_argument("--model", default="/data/eellison/models/Qwen3-8B")
p.add_argument("--sweep-tokens", type=int, nargs="*", default=[], help="integration: tokens-per-step sweep totals (with --sweep-decode splits)")
p.add_argument("--sweep-decode", type=int, nargs="*", default=[0], help="decoding requests in each sweep step; prefill fills the rest")
p.add_argument("--sweep-reps", type=int, default=6)
p.add_argument("--max-batched", type=int, default=None, help="engine max_num_batched_tokens (default: vLLM's)")
p.add_argument("--max-seqs", type=int, default=None, help="engine max_num_seqs (default: vLLM's)")
p.add_argument("--async-decode", type=int, nargs="*", default=[], help="integration: decode step time with no sync inside the step (async scheduling as shipped): (wall(48 tokens) - wall(16)) / 32")
p.add_argument("--shrink", type=int, nargs="*", default=[], help="integration: fluctuating decode batch: N requests, one finishing per step (bs N, N-1, .., 1), 2 passes; pass 2 = settled")
p.add_argument("--mixed-spec", nargs="*", default=[], help="integration: extra mixed workloads NDxPL (ND decoding requests, PL-token prompts added one per 2 steps, --mixed rounds)")
a = p.parse_args()
ARM_V = a.arm in ("V", "Vc")
if ARM_V:
    sys.path.insert(0, os.environ.get("INTEG_ARMV", "/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/vllm/armV"))
    import adapter as armv_adapter

LOG = []  # (num_reqs, num_tokens, step_s, fwd_s, requests with one scheduled token)
FWD = []
DEPTH = [0]


def synced(fn, sink):
    # time the outermost call only (run_pw_graph calls the model), never during capture
    def wrapper(*args, **kw):
        if DEPTH[0] or torch.cuda.is_current_stream_capturing():
            return fn(*args, **kw)
        DEPTH[0] += 1
        try:
            torch.cuda.synchronize()
            t = time.perf_counter()
            out = fn(*args, **kw)
            torch.cuda.synchronize()
            sink(time.perf_counter() - t, args)
            return out
        finally:
            DEPTH[0] -= 1
    return wrapper


fwd_sink = lambda dt, args: FWD.append(dt)
if a.launch_stamps:  # the step's own host path only: no synchronize between execute_model's entry and the launch
    synced = lambda fn, sink: fn
    import ctypes
    _shim = ctypes.CDLL(next(x for x in os.environ["LD_PRELOAD"].split(":") if "launchshim" in x))
    L_NS, L_N = (ctypes.c_int64 * 256).in_dll(_shim, "ht_launch_ns"), ctypes.c_int64.in_dll(_shim, "ht_launch_count")
    assert time.get_clock_info("perf_counter").implementation == "clock_gettime(CLOCK_MONOTONIC)"
NFWD = {"init": 0, "bench": 0}
PHASE = ["init"]
_cls_fwd = Qwen3ForCausalLM.forward


def _counted_fwd(*args, **kw):
    NFWD[PHASE[0]] += 1
    return _cls_fwd(*args, **kw)


Qwen3ForCausalLM.forward = _counted_fwd
if not ARM_V:  # arm V times the adapter's forward (installed below): the traced step calls the class forward
    Qwen3ForCausalLM.forward = synced(Qwen3ForCausalLM.forward, fwd_sink)
CudaGraphManager.run_pw_graph = synced(CudaGraphManager.run_pw_graph, fwd_sink)
ModelCudaGraphManager.run_fullgraph = synced(ModelCudaGraphManager.run_fullgraph, fwd_sink)
orig_exec = GPUModelRunner.execute_model


def timed_exec(self, so, *args, **kw):
    if kw.get("dummy_run"):
        return orig_exec(self, so, *args, **kw)
    torch.cuda.synchronize()
    n = len(FWD)
    n0 = L_N.value if a.launch_stamps else 0
    t = time.perf_counter()
    out = orig_exec(self, so, *args, **kw)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    if a.launch_stamps:  # field 3: pre-launch seconds (None without a graph launch); field 5: launches in the step
        fwd = L_NS[n0 & 255] / 1e9 - t if L_N.value > n0 else None
        fe = STAMPS.pop("@fwd", None)
        LOG.append((len(so.num_scheduled_tokens), so.total_num_scheduled_tokens, t1 - t, fwd, sum(v == 1 for v in so.num_scheduled_tokens.values()),
                    {"launches": L_N.value - n0, "fwd_entry": fe - t if fe else None}))
        return out
    LOG.append((len(so.num_scheduled_tokens), so.total_num_scheduled_tokens, t1 - t, FWD[n] if len(FWD) > n else None,
                sum(v == 1 for v in so.num_scheduled_tokens.values())))
    return out


GPUModelRunner.execute_model = timed_exec
NOSYNC = [False]
_timed_sync = timed_exec


def timed_exec(self, so, *args, **kw):  # --async-decode: no synchronize around the step
    if NOSYNC[0]:
        NOSYNC.append(1)
        return orig_exec(self, so, *args, **kw)
    return _timed_sync(self, so, *args, **kw)


GPUModelRunner.execute_model = timed_exec
if a.arm == "VAf":  # vLLM default graphs with V's host cuts in the input prep / request-state writes (installed before capture)
    sys.path.insert(0, os.environ["INTEG_ARMV"])
    import hostcuts as _hostcuts

    _orig_capture = GPUModelRunner.capture_model

    def _capture_with_cuts(self):
        R_HOSTCUTS.extend(_hostcuts.install(self))
        return _orig_capture(self)
    GPUModelRunner.capture_model = _capture_with_cuts
R_HOSTCUTS = []
PROF = {}  # bs -> decode steps seen; steps 12-14 of a profiled bs run under torch.profiler, one trace each
if a.profile_decode:
    _untraced = timed_exec

    def timed_exec(self, so, *args, **kw):
        bs = len(so.num_scheduled_tokens)
        if kw.get("dummy_run") or bs not in a.profile_decode or so.total_num_scheduled_tokens != bs:
            return _untraced(self, so, *args, **kw)
        PROF[bs] = PROF.get(bs, 0) + 1
        if not 12 <= PROF[bs] <= 14:
            return _untraced(self, so, *args, **kw)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
            out = _untraced(self, so, *args, **kw)
        os.makedirs(a.profile_dir, exist_ok=True)
        prof.export_chrome_trace(os.path.join(a.profile_dir, f"{a.arm}_bs{bs}_{PROF[bs]}.json"))
        return out
    GPUModelRunner.execute_model = timed_exec
STAMPS = {}  # section -> host seconds in the current step; with --stamps a copy becomes LOG row field 5


def stamped(fn, name):
    def wrapper(*args, **kw):
        t = time.perf_counter()
        try:
            return fn(*args, **kw)
        finally:
            STAMPS[name] = STAMPS.get(name, 0.0) + time.perf_counter() - t
    return wrapper


if a.stamps:
    import vllm.v1.worker.gpu.model_runner as mr
    from vllm.v1.worker.gpu.model_states.default import DefaultModelState
    for n in ("finish_requests", "free_states", "add_requests", "update_requests", "gather_batch_req_state", "prepare_inputs", "prepare_attn"):
        setattr(GPUModelRunner, n, stamped(getattr(GPUModelRunner, n), n))
    for n in ("prepare_inputs", "prepare_attn"):
        setattr(DefaultModelState, n, stamped(getattr(DefaultModelState, n), "ms." + n))
    mr.build_slot_mappings_by_layer = stamped(mr.build_slot_mappings_by_layer, "build_slot_mappings_by_layer")
    if a.stamps_fine:
        for n in ("async_copy_to_gpu", "prepare_pos_seq_lens", "combine_sampled_and_draft_tokens", "prepare_prefill_inputs"):
            setattr(mr, n, stamped(getattr(mr, n), "pi." + n))
        from vllm.v1.worker.gpu.sample.sampler import Sampler
        from vllm.v1.worker.gpu.states import RequestState
        Sampler.apply_staged_writes = stamped(Sampler.apply_staged_writes, "ar.sampler_staged")
        RequestState.apply_staged_writes = stamped(RequestState.apply_staged_writes, "ar.req_states_staged")
        RequestState.add_request = stamped(RequestState.add_request, "ar.req_states_add")
    _timed = timed_exec

    def timed_exec(self, so, *args, **kw):
        STAMPS.clear()
        n = len(LOG)
        out = _timed(self, so, *args, **kw)
        if len(LOG) > n:
            LOG[-1] += (dict(STAMPS),)
        return out
    GPUModelRunner.execute_model = timed_exec

# drive_prof: per-step cudagraph dispatch, the scheduled request mix and Dynamo/Inductor counter deltas (field "diag" of R["rows_diag"])
import vllm.v1.worker.gpu.model_runner as _mr
from torch._dynamo.utils import counters as _ctr
DIAG = []


class Diag(dict):  # the step row's last field; summ's section medians skip it
    pass
_disp = _mr.dispatch_cg_and_sync_dp


def _disp_logged(mgr, num_reqs, num_toks, uniform, *args, **kw):
    out = _disp(mgr, num_reqs, num_toks, uniform, *args, **kw)
    bd = out[0]
    DIAG.append({"num_reqs": num_reqs, "num_toks": num_toks, "uniform_tok_count": uniform, "max_query_len": kw.get("max_query_len"),
                 "cg_mode": str(bd.cg_mode), "padded_tokens": bd.num_tokens, "padded_reqs": getattr(bd, "num_reqs", None)})
    return out
_mr.dispatch_cg_and_sync_dp = _disp_logged
_ctr_flat = lambda: {f"{k}.{k2}": v for k, d in _ctr.items() for k2, v in d.items()}
_exec_diag = GPUModelRunner.execute_model


def _exec_with_diag(self, so, *args, **kw):
    if kw.get("dummy_run"):
        return _exec_diag(self, so, *args, **kw)
    c0, nd = _ctr_flat(), len(DIAG)
    out = _exec_diag(self, so, *args, **kw)
    c1 = _ctr_flat()
    comp = {}
    for r in so.scheduled_new_reqs:
        comp[r.req_id] = r.num_computed_tokens
    cr = so.scheduled_cached_reqs
    for rid, nc in zip(cr.req_ids, cr.num_computed_tokens):
        comp[rid] = nc
    mix = sorted((so.num_scheduled_tokens[r], comp.get(r)) for r in so.num_scheduled_tokens)
    d = Diag(DIAG[nd]) if len(DIAG) > nd else Diag()
    d["mix"] = mix
    d["counters"] = {k: c1[k] - c0.get(k, 0) for k in c1 if c1[k] != c0.get(k, 0)}
    if LOG:
        LOG[-1] = LOG[-1] + (d,)
    return out
GPUModelRunner.execute_model = _exec_with_diag

t0 = time.perf_counter()
extra = {"compilation_config": {"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"}} if a.arm == "fullnc" else {}
if a.arm == "fullnc_full":  # default-nc with FULL graphs for every batch vLLM can graph (compile off)
    extra = {"compilation_config": {"mode": 0, "cudagraph_mode": "FULL"}}
if a.arm in ("Vc", "compnc"):
    extra = {"compilation_config": {"cudagraph_mode": "NONE"}}
if a.max_batched:
    extra["max_num_batched_tokens"] = a.max_batched
if a.max_seqs:
    extra["max_num_seqs"] = a.max_seqs
llm = LLM(model=a.model, tensor_parallel_size=1, enforce_eager=a.arm not in ("default", "fullnc", "fullnc_full", "VAf", "Vc", "compnc"), attention_backend="FLASHINFER", **extra,
          enable_prefix_caching=False, gpu_memory_utilization=float(os.environ.get("INTEG_GPU_UTIL", "0.6")), **({"kv_cache_memory_bytes": int(os.environ["INTEG_KV_BYTES"])} if os.environ.get("INTEG_KV_BYTES") else {}), max_model_len=a.max_model_len, seed=0)
R = {"arm": a.arm, "init_s": time.perf_counter() - t0, "decode": {}, "prefill": {}, "mixed": {}, "mixed_specs": {}, "rows": {}}
R["util"] = float(os.environ.get("INTEG_GPU_UTIL", "0.6"))
_cc = llm.llm_engine.vllm_config.compilation_config
_runner = llm.llm_engine.model_executor.driver_worker.worker.model_runner
R["cg_config"] = {"mode": str(_cc.mode), "cudagraph_mode": str(_cc.cudagraph_mode), "capture_sizes": list(_cc.cudagraph_capture_sizes or []),
                  "max_capture_size": _cc.max_cudagraph_capture_size, "use_breakable_cg": getattr(_runner.cudagraph_manager, "use_breakable_cg", None),
                  "max_num_batched_tokens": llm.llm_engine.vllm_config.scheduler_config.max_num_batched_tokens,
                  "chunked_prefill": llm.llm_engine.vllm_config.scheduler_config.enable_chunked_prefill}
R["reserved_after_init_mib"] = torch.cuda.memory_reserved() / 2**20
PHASE[0] = "bench"
V = None
if ARM_V:
    runner = llm.llm_engine.model_executor.driver_worker.worker.model_runner
    import adapter_cpp  # install_sgl_cpp (python_vllm_cpp.sh): the bound decode entry unless ARMV_BOUND=0
    V = (adapter_cpp.ArmVcpp if adapter_cpp.BOUND else armv_adapter.ArmV)(runner)
    V.install()
    V.check = a.check
    runner.model.forward = synced(V.forward, fwd_sink)
if ARM_V and a.stamps_fine:  # instance-level patches (V.install, armV/hostcuts.py) hide the class-level stamps
    import vllm.v1.worker.gpu.model_runner as mr
    for owner, n, tag in [(runner, "prepare_inputs", "prepare_inputs"), (runner.req_states, "apply_staged_writes", "ar.req_states_staged"),
                          (runner.sampler, "apply_staged_writes", "ar.sampler_staged"), (runner.model_state, "preprocess_state", "ms.preprocess_state"),
                          (runner.model_state, "prepare_inputs", "ms.prepare_inputs"), (runner.step_timing, "record_batch", "x.step_timing"),
                          (runner.eplb, "prepare_forward", "x.eplb"), (mr, "dispatch_cg_and_sync_dp", "x.dispatch_cg"), (mr, "set_forward_context", "x.set_fc_create"),
                          (runner.block_tables, "apply_staged_writes", "x.bt_staged"), (runner, "prepare_attn", "x.prepare_attn"),
                          (runner.model_state, "prepare_attn", "ms.prepare_attn"), (runner, "update_pp_decode_requests", "x.update_pp")]:
        setattr(owner, n, stamped(getattr(owner, n), tag))
    import hostcuts
    hostcuts.PROF = STAMPS
if ARM_V and a.launch_stamps and a.stamps:  # forward entry stamp (absolute; timed_exec turns it into an offset)
    _m = llm.llm_engine.model_executor.driver_worker.worker.model_runner.model
    _mf = _m.forward

    def _fwd_stamped(*args, **kw):
        STAMPS.setdefault("@fwd", time.perf_counter())
        return _mf(*args, **kw)
    _m.forward = _fwd_stamped
DUMP = []
if a.dump_decode:
    _model = llm.llm_engine.model_executor.driver_worker.worker.model_runner.model
    _cl = _model.compute_logits

    def compute_logits(h, *rest, **kw):
        if LOG and LOG[-1][0] == LOG[-1][1] == h.shape[0]:
            DUMP.append(h.detach().cpu().clone())
        return _cl(h, *rest, **kw)
    _model.compute_logits = compute_logits
g = torch.Generator().manual_seed(0)


def prompts(n, length):
    return [TokensPrompt(prompt_token_ids=torch.randint(1000, 100000, (length,), generator=g).tolist()) for _ in range(n)]


def summ(rows):
    st, fw = [r[2] * 1e3 for r in rows], [r[3] * 1e3 for r in rows if r[3] is not None]
    out = {"n": len(rows), "step_ms": statistics.median(st), "fwd_ms": statistics.median(fw) if fw else None, "step_min_ms": min(st)}
    if a.launch_stamps:
        out["pre_launch_us"] = out.pop("fwd_ms") * 1e3 if fw else None
        out["launches"] = statistics.median(r[5]["launches"] for r in rows)
        fe = [r[5]["fwd_entry"] for r in rows if r[5]["fwd_entry"] is not None]
        if fe:  # execute_model entry -> model forward entry, and forward entry -> launch
            out["fwd_entry_us"] = statistics.median(fe) * 1e6
            out["fwd_to_launch_us"] = statistics.median(r[3] - r[5]["fwd_entry"] for r in rows if r[3] is not None and r[5]["fwd_entry"] is not None) * 1e6
        secs = [r[6] for r in rows if len(r) > 6 and not isinstance(r[6], Diag)]
        if secs:
            out["sections_us"] = {k: round(statistics.median(d.get(k, 0.0) for d in secs) * 1e6, 1) for k in sorted({k for d in secs for k in d})}
        return out
    secs = [r[5] for r in rows if len(r) > 5 and not isinstance(r[5], Diag)]
    if secs:
        out["sections_us"] = {k: round(statistics.median(d.get(k, 0.0) for d in secs) * 1e6, 1) for k in sorted({k for d in secs for k in d})}
    return out


torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
mem0 = torch.cuda.memory_allocated()
sp = lambda n: SamplingParams(max_tokens=n, ignore_eos=True, temperature=0.0)
llm.generate(prompts(4, 128), sp(8), use_tqdm=False)  # warm-up
for bs in a.batch_size:
    LOG.clear()
    llm.generate(prompts(bs, 256), sp(a.decode_tokens), use_tqdm=False)
    dec = [r for r in LOG if r[0] == bs and r[1] == bs]
    if len(dec) < 8:
        print("LOG", LOG[:8], flush=True)
    R["decode"][bs] = summ(dec[4:])
    R["rows"][f"decode{bs}"] = [list(r[:4]) for r in LOG]
    print("decode", bs, R["decode"][bs], flush=True)
for L in a.prefill_lens:
    rows = []
    for _ in range(9):
        LOG.clear()
        llm.generate(prompts(1, L), sp(1), use_tqdm=False)
        rows += [r for r in LOG if r[1] == L]
    R["prefill"][L] = summ(rows[4:])
    R["rows"][f"prefill{L}"] = [list(r[:4]) for r in rows]
    print("prefill", L, R["prefill"][L], flush=True)
for N in a.prefill_reqs:
    rows = []
    for _ in range(9):  # arm V: a key's first 4 calls are eager, trace, learner run, retrace
        LOG.clear()
        llm.generate(prompts(N, a.prefill_req_len), sp(1), use_tqdm=False)
        rows += [r for r in LOG if r[0] == N and r[1] == N * a.prefill_req_len]
    R["prefill"][f"{N}x{a.prefill_req_len}"] = summ(rows[4:]) if len(rows) > 4 else {"rows": [list(r) for r in rows]}
    print("prefill", N, R["prefill"][f"{N}x{a.prefill_req_len}"], flush=True)
if a.mixed:
    eng, sp_m = llm.llm_engine, sp(1000)
    LOG.clear()
    for i, pr in enumerate(prompts(8, 256)):
        eng.add_request(f"m{i}", pr, sp_m)
    for _ in range(4):
        eng.step()
    prof = None
    for j, pr in enumerate(prompts(a.mixed, 128)):
        if a.prof_mixed and j == 3:  # rows[3:] are the measured mixed steps
            from torch.profiler import ProfilerActivity, profile, record_function
            torch.cuda.synchronize()
            prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True, record_shapes=False)
            prof.__enter__()
        eng.add_request(f"n{j}", pr, sp(2))
        if prof is not None:
            with record_function(f"MIXED_STEP_{j}"):
                eng.step()
            with record_function(f"AFTER_STEP_{j}"):
                eng.step()
        else:
            eng.step()
            eng.step()
    if prof is not None:
        torch.cuda.synchronize()
        prof.__exit__(None, None, None)
        os.makedirs(a.prof_mixed, exist_ok=True)
        prof.export_chrome_trace(os.path.join(a.prof_mixed, f"mixed_{a.arm}.json"))
    rows = [r for r in LOG if 0 < r[4] < r[0]]
    for i in range(8):
        eng.abort_request([f"m{i}"])
    for _ in range(500):  # a request that never finishes (lost sampled tokens) fails the run instead of hanging it
        if not eng.has_unfinished_requests():
            break
        eng.step()
    else:
        raise RuntimeError("mixed: requests still unfinished after 500 steps")
    R["mixed"] = summ(rows[3:]) if len(rows) > 3 else {"rows": [list(r) for r in rows]}
    R["mixed"]["shapes"] = sorted({(r[0], r[1]) for r in rows})
    R["rows"]["mixed"] = [list(r[:4]) for r in rows]
    R["rows_diag_mixed"] = [r[-1] for r in rows if isinstance(r[-1], dict)]
    R["rows_diag_all"] = [[r[0], r[1], r[2], r[-1]] for r in LOG if isinstance(r[-1], dict)]
    print("mixed", R["mixed"], flush=True)
for spec in a.mixed_spec:  # same form as above at other decode:prefill ratios
    nd, pl = map(int, spec.split("x"))
    eng, sp_m = llm.llm_engine, sp(1000)
    LOG.clear()
    for i, pr in enumerate(prompts(nd, 64)):
        eng.add_request(f"ms{spec}_{i}", pr, sp_m)
    for _ in range(4):
        eng.step()
    for j, pr in enumerate(prompts(max(a.mixed, 6), pl)):
        eng.add_request(f"ns{spec}_{j}", pr, sp(2))
        eng.step()
        eng.step()
    rows = [r for r in LOG if 0 < r[4] < r[0]]
    eng.abort_request([f"ms{spec}_{i}" for i in range(nd)])
    while eng.has_unfinished_requests():
        eng.step()
    R["mixed_specs"][spec] = summ(rows[3:]) if len(rows) > 3 else {"rows": [list(r) for r in rows]}
    R["mixed_specs"][spec]["shapes"] = sorted({(r[0], r[1]) for r in rows})
    R["rows"][f"mixed{spec}"] = [list(r[:4]) for r in rows]
    print("mixed", spec, R["mixed_specs"][spec], flush=True)
R["hostcuts"] = R_HOSTCUTS
R["async"] = {}
for bs in a.async_decode:
    walls = {}
    for n_tok in (16, 48, 16, 48):
        NOSYNC[1:] = []
        NOSYNC[0] = True
        torch.cuda.synchronize()
        t = time.perf_counter()
        llm.generate(prompts(bs, 256), sp(n_tok), use_tqdm=False)
        torch.cuda.synchronize()
        walls.setdefault(n_tok, []).append((time.perf_counter() - t, len(NOSYNC) - 1))
        NOSYNC[0] = False
    w16, w48 = min(w[0] for w in walls[16]), min(w[0] for w in walls[48])
    R["async"][bs] = {"step_ms": (w48 - w16) / 32 * 1e3, "walls": walls}
    print("async", bs, R["async"][bs], flush=True)
R["sweep"] = {}
for nd in a.sweep_decode:  # one step of exactly T tokens: nd decoding requests + prefill prompts (<= 4000 tokens each) filling T - nd
    for T in a.sweep_tokens:
        if nd >= T:
            continue
        eng, P = llm.llm_engine, T - nd
        k = -(-P // 4000)
        lens = [P // k + (i < P % k) for i in range(k)]
        LOG.clear()
        for i, pr in enumerate(prompts(nd, 64)):
            eng.add_request(f"sw{T}_{nd}_d{i}", pr, sp(100000))
        for _ in range(3):
            eng.step()
        for j in range(a.sweep_reps):
            for i, pr in enumerate([TokensPrompt(prompt_token_ids=torch.randint(1000, 100000, (L,), generator=g).tolist()) for L in lens]):
                eng.add_request(f"sw{T}_{nd}_p{j}_{i}", pr, sp(1))
            eng.step()
            eng.step()
        rows = [r for r in LOG if r[1] == T and r[0] == nd + k]
        eng.abort_request([f"sw{T}_{nd}_d{i}" for i in range(nd)])
        while eng.has_unfinished_requests():
            eng.step()
        key = f"{T}_{nd}"
        R["sweep"][key] = (summ(rows[2:]) if len(rows) > 2 else {"step_ms": None}) | {"n_rows": len(rows), "first_ms": [round(r[2] * 1e3, 2) for r in rows[:3]],
                                                                                       "prompts": lens, "shapes": sorted({(r[0], r[1]) for r in LOG if r[1] > nd})}
        print("sweep", key, {k_: v for k_, v in R["sweep"][key].items() if k_ != "shapes"}, flush=True)
R["ix"] = {}
spread = lambda n, lo, hi: [lo + (hi - lo) * i // max(1, n - 1) for i in range(n)]
kv_prompt = lambda L: TokensPrompt(prompt_token_ids=torch.randint(1000, 100000, (L,), generator=g).tolist())


def _ix_fill(tag, lens):  # decoders with long prompts, stepped until every one of them decodes (one scheduled token each)
    eng = llm.llm_engine
    for i, L in enumerate(lens):
        eng.add_request(f"{tag}_d{i}", kv_prompt(L), sp(100000))
    for _ in range(2000):
        LOG.clear()
        eng.step()
        if LOG and LOG[-1][0] == len(lens) and LOG[-1][1] == len(lens):
            return
    raise RuntimeError(f"{tag}: decoders never all decoding")


def _ix_drain(tag, n):
    eng = llm.llm_engine
    eng.abort_request([f"{tag}_d{i}" for i in range(n)])
    while eng.has_unfinished_requests():
        eng.step()


def _ix_kv(rows):
    kv = [c + s_ for r in rows if isinstance(r[-1], dict) for s_, c in r[-1].get("mix", []) if s_ == 1 and c is not None]
    return [min(kv), int(statistics.median(kv)), max(kv)] if kv else None


for spec in a.ix_mixed:  # one step = ND decodes + one P-token chunk (P + ND <= max_num_batched_tokens), then one decode-only step
    P, nd = map(int, spec.split("x"))
    tag, eng = f"ixm{P}_{nd}", llm.llm_engine
    _ix_fill(tag, spread(nd, *a.ix_mixed_kv))
    LOG.clear()
    for j in range(a.ix_reps):
        eng.add_request(f"{tag}_p{j}", kv_prompt(P), sp(1))
        eng.step()
        eng.step()
    rows = [r for r in LOG if r[1] == P + nd and r[0] == nd + 1]
    R["ix"][f"mixed_{spec}"] = (summ(rows[2:]) if len(rows) > 2 else {"step_ms": None}) | {"n_rows": len(rows), "first_ms": [round(r[2] * 1e3, 2) for r in rows[:2]],
                                                                                         "kv_min_med_max": _ix_kv(rows), "shapes": sorted({(r[0], r[1]) for r in LOG})}
    _ix_drain(tag, nd)
    print("ix", spec, {k: v for k, v in R["ix"][f"mixed_{spec}"].items() if k != "shapes"}, flush=True)
for spec in a.ix_decode:
    bs, rng = spec.split(":")
    bs, (lo, hi) = int(bs), map(int, rng.split("-"))
    tag, eng = f"ixd{bs}", llm.llm_engine
    _ix_fill(tag, spread(bs, lo, hi))
    LOG.clear()
    for _ in range(a.ix_reps + 14):
        eng.step()
    rows = [r for r in LOG if r[0] == r[1] == bs]
    R["ix"][f"decode_{bs}"] = (summ(rows[4:]) if len(rows) > 4 else {"step_ms": None}) | {"n_rows": len(rows), "kv_min_med_max": _ix_kv(rows)}
    _ix_drain(tag, bs)
    print("ix decode", bs, {k: v for k, v in R["ix"][f"decode_{bs}"].items()}, flush=True)
R["shrink"] = {}
for N in a.shrink:  # every decode step a new batch size
    for ps in (1, 2):
        LOG.clear()
        llm.generate(prompts(N, 64), [sp(k + 1) for k in range(N)], use_tqdm=False)
        rows = [r for r in LOG if r[0] == r[1]]
        R["shrink"][f"{N}_pass{ps}"] = summ(rows) | {"bs": [r[0] for r in rows], "step_ms_all": [round(r[2] * 1e3, 3) for r in rows],
                                                      "mean_ms": statistics.fmean(r[2] * 1e3 for r in rows), "sum_ms": sum(r[2] * 1e3 for r in rows)}
        print("shrink", N, ps, {k: v for k, v in R["shrink"][f"{N}_pass{ps}"].items() if k not in ("bs", "step_ms_all")}, flush=True)
import gc, numpy as np  # host probe: the same CPU-only code in every arm's process, and the collector's state
_arr, _t = np.arange(1025, dtype=np.int32), torch.zeros(1024, dtype=torch.int32)
def _probe(fn, n=3000):
    ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return round(sorted(ts)[n // 2] * 1e6, 2)
R["host_probe_us"] = {"python": _probe(lambda: sum(i * i for i in range(200))), "numpy": _probe(lambda: np.cumsum(_arr[:65])),
                      "torch_cpu": _probe(lambda: torch.from_numpy(_arr)[:64].clone()), "slice": _probe(lambda: _t[:64]),
                      "cuda_empty": _probe(lambda: torch.empty(64, device="cuda"))}
R["gc"] = {"enabled": gc.isenabled(), "threshold": gc.get_threshold(), "count": gc.get_count(), "frozen": gc.get_freeze_count(), "tracked": len(gc.get_objects())}
import threading
gc.disable()
R["host_probe_us"]["numpy_gc_off"] = _probe(lambda: np.cumsum(_arr[:65]))
gc.enable()
R["host_env"] = {"profile": repr(sys.getprofile()), "trace": repr(sys.gettrace()), "threads": [t.name for t in threading.enumerate()],
                 "monitoring": [sys.monitoring.get_tool(i) for i in range(6)] if hasattr(sys, "monitoring") else None,
                 "tf_modes": torch._C._len_torch_function_stack(), "dispatch_modes": torch._C._len_torch_dispatch_stack(),
                 "grad": torch.is_grad_enabled(), "inference": torch.is_inference_mode_enabled()}
print("host_probe", R["host_probe_us"], R["gc"], R["host_env"], flush=True)
if os.environ.get("PROBE_PERF"):  # flat perf profile of this process while it runs the torch_cpu probe snippet for 4 s
    import subprocess
    ev = "cycles,instructions,L1-icache-load-misses,L1-dcache-load-misses,iTLB-load-misses,dTLB-load-misses,LLC-load-misses,page-faults,context-switches,cpu-migrations"
    cmd = ["perf", "stat", "-e", ev, "-t", str(threading.get_native_id()), "-o", os.environ["PROBE_PERF"], "--", "sleep", "3"]
    pp = subprocess.Popen(cmd if os.environ.get("PROBE_STAT") else ["perf", "record", "-F", "4999", "-p", str(os.getpid()), "-o", os.environ["PROBE_PERF"], "--", "sleep", "3"])
    t0 = time.perf_counter()
    it = 0
    while pp.poll() is None:
        torch.from_numpy(_arr)[:64].clone()
        it += 1
    R["probe_perf"] = {"iters": it, "secs": time.perf_counter() - t0}
    print("probe_perf", R["probe_perf"], flush=True)
if V is not None:
    R["V"] = V.summary()
R["model_forwards"] = dict(NFWD)
R["mem"] = {}
R["mem"] |= {"reserved_after_init_mib": R["reserved_after_init_mib"], "max_allocated_mib": torch.cuda.max_memory_allocated() / 2**20, "reserved_mib": torch.cuda.memory_reserved() / 2**20,
            "after_init_mib": mem0 / 2**20, "bench_peak_over_init_mib": (torch.cuda.max_memory_allocated() - mem0) / 2**20,
            "end_over_init_mib": (torch.cuda.memory_allocated() - mem0) / 2**20,
            "kv_cache_tokens": llm.llm_engine.vllm_config.cache_config.num_gpu_blocks * llm.llm_engine.vllm_config.cache_config.block_size}
if a.dump_decode:
    torch.save(DUMP, a.dump_decode)
json.dump(R, open(a.out, "w"), indent=1, default=str)
print(json.dumps(R, default=str)[:20000], flush=True)
