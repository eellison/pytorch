# drive_sweep.py: a frozen copy of drive_ix2.py (10-13 05:55, serve lane) for the slide sweeps; drive_ix2.py is shared with another instance.
# Baseline driver for vLLM's own arms (no host trace), mirroring sglang/bench/drive.py:
# decode: bs requests of 256 prompt tokens, 24 steps, steps[4:]; prefill: one request of L tokens, 7 reps, reps[2:].
# step = V2 GPUModelRunner.execute_model (input prep + metadata + forward, not sampling), synced;
# fwd = the forward alone (model call, run_pw_graph or run_fullgraph), synced.
# python_vllm.sh bench/drive.py --arm eager|default --out X.json; arm V (host trace, armV/adapter.py) runs through python_vllm_v.sh.
# mixed: 8 requests decoding, then one 128-token request added per step (decode + prefill in one forward), 6 rounds.
import argparse, collections, json, os, statistics, sys, time
T_START = time.perf_counter()


def _cpu_snap():  # (total, idle) jiffies per CPU in this process's affinity, from /proc/stat
    aff, out = os.sched_getaffinity(0), {}
    for line in open("/proc/stat"):
        if line.startswith("cpu") and line[3].isdigit():
            f = line.split()
            c = int(f[0][3:])
            if c in aff:
                v = list(map(int, f[1:9]))
                out[c] = (sum(v), v[3] + v[4])
    return out


CPU0 = _cpu_snap()
_RU0 = os.times()
# DRIVE_GC_LOG=1: every cyclic-GC pass (generation, seconds), so a slow step can be matched to a collection
GC_EVENTS = []


def _psi_mem_s():  # host memory pressure "full" total (us), /proc/pressure/memory
    try:
        for ln in open("/proc/pressure/memory"):
            if ln.startswith("full"):
                return int(ln.split("total=")[1])
    except OSError:
        return None
if os.environ.get("DRIVE_GC_LOG") == "1":
    import gc as _gc
    _gc_t0 = [0.0]

    def _gc_cb(phase, info):
        if phase == "start":
            _gc_t0[0] = time.perf_counter()
        else:
            GC_EVENTS.append((info["generation"], time.perf_counter() - _gc_t0[0], info.get("collected", 0), time.perf_counter()))
    _gc.callbacks.append(_gc_cb)

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
import torch
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
p.add_argument("--ix-mixed", nargs="*", default=[], help="drive_ix: InferenceX-shaped mixed steps PxND[@LO-HI]: one P-token prefill chunk + ND decoders at KV LO-HI (default --ix-mixed-kv)")
p.add_argument("--ix-mixed-kv", type=int, nargs=2, default=[4096, 8000], help="decoder prompt lengths spread evenly over [lo, hi]")
p.add_argument("--dec-ctx", type=int, nargs="*", default=[], help="decode at long context (10-13, InferenceX-style TPOT): prompt length per curve")
p.add_argument("--dec-ctx-sizes", type=int, nargs="*", default=[], help="batch sizes of the --dec-ctx curves")
p.add_argument("--dec-ctx-steps", type=int, default=128, help="measured full-batch decode steps per point (after 4 dropped)")
p.add_argument("--prefill-reps", type=int, default=9, help="generate calls per prefill length (the first 4 rows are dropped)")
p.add_argument("--warm-range", type=int, nargs=3, default=None, metavar=("LO", "HI", "BSMAX"),
               help="untimed warm-up before the churn windows (every arm): one prefill per token count LO..HI and a short decode at every bs 1..BSMAX (first-time traces, cuBLAS binds)")
p.add_argument("--cb", type=int, nargs="*", default=[], help="continuous batching (10-13): keep C requests in flight; a finished one is replaced by a new one (its prefill lands as a mixed step)")
p.add_argument("--cb-steps", type=int, default=2000, help="measured engine steps per C (after --cb-warm)")
p.add_argument("--cb-warm", type=int, default=300)
p.add_argument("--cb-out", type=int, nargs=2, default=[256, 2048], help="output lengths, seeded uniform over [lo, hi]")
p.add_argument("--cb-prompt", type=int, nargs=2, default=[64, 512], help="prompt lengths, seeded uniform over [lo, hi]")
p.add_argument("--ifx-c", type=int, nargs="*", default=[], help="10-19 InferenceX-shaped fixed-seq (fixed_seq.py semantics; as r27 drive_sweep27): concurrency values, one window each")
p.add_argument("--ifx-shape", type=int, nargs=2, default=[1024, 1024], help="fixed-seq input / output tokens L_in L_out; lengths seeded uniform over [ratio * L, L]")
p.add_argument("--ifx-ratio", type=float, default=0.8, help="fixed-seq range ratio (InferenceX 0.8)")
p.add_argument("--ifx-mult", type=int, default=10, help="fixed-seq prompts per window = mult * C (InferenceX 10)")
p.add_argument("--ifx-prewarm-frac", type=float, default=1.0, help="untimed pass over this fraction (at least C prompts; 0 = none) of the same seeded request list")
p.add_argument("--ifx-wave-reps", type=int, default=0, help="before the untimed pass: the first C prompts (the initial admission wave) this many times, outputs capped at --ifx-wave-out")
p.add_argument("--ifx-wave-out", type=int, default=64, help="output cap of the wave reps: the wave's prefill steps are the sequence's while no request of the wave has finished")
p.add_argument("--alt-decode", nargs="*", default=[], help="churn diagnostic (10-13): pure decode alternating batch sizes A:B[@LO-HI] every step (scheduler token budget A / B over B decoders)")
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
        # 10-14: no sync in an unsynced window (NOSYNC: cbh4 / startup streams); before this fix the forward was synced in every arm
        if DEPTH[0] or NOSYNC[0] or torch.cuda.is_current_stream_capturing():
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
# entry profile (serve lane, 10-13), DRIVE_PHASES=1: the C++ replay entry's phase clocks (_host_trace_phase_times) and patch counters
# over each full step (reset at execute_model's start, read after sample_tokens), in the step's Diag; summ reports their medians
PHASES_ON = os.environ.get("DRIVE_PHASES") == "1" and hasattr(torch._C, "_host_trace_phase_times")
PH_NAMES = ("p0", "key", "bind", "evaluate", "allocate", "patch", "launch", "outputs", "dispatch", "miss", "a_plan", "eager_a", "eager_b", "c_misc", "k_capture", "p15")
PATCH_CTRS = ("kernel_packs", "kernel_node_sets", "memory_node_sets")
_patch0 = [None]


def _patch_now():
    return [getattr(torch._C, "_host_trace_" + n)() for n in PATCH_CTRS] + list(getattr(torch._C, "_host_trace_delta_counts", lambda: ())())


def _phase_reset():
    torch._C._host_trace_phase_times(True)
    _patch0[0] = _patch_now()


def _phase_read():
    ph = torch._C._host_trace_phase_times(True)
    now = _patch_now()
    return {"ph_us": {n: x / 1e3 for n, x in zip(PH_NAMES, ph) if x}, "patch": [b - a for a, b in zip(_patch0[0] or now, now)]}


# per-point kernel record (DRIVE_KREC=1, 10-14): after a point's measured window, one extra step of the point's shape runs under
# torch.profiler (CUDA activity only; CUPTI reports kernels inside graph launches) and its kernels' (name, grid, block) multiset goes
# into the point's JSON ("kernels"), for the kernel-identity check against the stock arm at every size.
KREC = {"on": os.environ.get("DRIVE_KREC") == "1", "target": None, "skip": 0, "seen": 0, "prof": None, "result": None}


def _krec_arm(shape, skip=0):
    KREC.update(target=shape if KREC["on"] else None, skip=skip, seen=0, prof=None, result=None)


def _krec_pre(so):
    if KREC["target"] is None or KREC["prof"] is not None:
        return
    if (len(so.num_scheduled_tokens), so.total_num_scheduled_tokens) != KREC["target"]:
        return
    KREC["seen"] += 1
    if KREC["seen"] == KREC["skip"] + 1:
        torch.cuda.synchronize()
        KREC["prof"] = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA])
        KREC["prof"].__enter__()


def _krec_post():
    if KREC["prof"] is None:
        return
    torch.cuda.synchronize()
    prof, KREC["prof"] = KREC["prof"], None
    prof.__exit__(None, None, None)
    path = f"/tmp/krec_{os.getpid()}.json"
    prof.export_chrome_trace(path)
    c = collections.Counter()
    for e in json.load(open(path))["traceEvents"]:
        if e.get("ph") == "X" and e.get("cat") == "kernel":
            ar = e.get("args", {})
            c[f"{e['name'][:200]}|grid={ar.get('grid')}|block={ar.get('block')}"] += 1
        elif e.get("ph") == "X" and e.get("cat") in ("gpu_memset", "gpu_memcpy"):  # memset / memcpy nodes and ops, by bytes
            c[f"{e['cat'].upper()} {e['name'][:60]}|bytes={e.get('args', {}).get('bytes')}"] += 1
    os.remove(path)
    KREC["result"], KREC["target"] = dict(c), None


if os.environ.get("DRIVE_STEP_WITH_SAMPLE") == "1":  # full step (serve lane, 10-12): sample_tokens' time added to the step's row (synced)
    _orig_sample_tokens = GPUModelRunner.sample_tokens

    def timed_sample_tokens(self, *args, **kw):
        if NOSYNC[0]:  # unsynced window (async scheduling as shipped): no sync, no row
            return _orig_sample_tokens(self, *args, **kw)
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = _orig_sample_tokens(self, *args, **kw)
        torch.cuda.synchronize()
        _krec_post()
        if LOG and not NOSYNC[0]:
            r = LOG[-1]
            LOG[-1] = (*r[:2], r[2] + time.perf_counter() - t, *r[3:])
            if PHASES_ON and isinstance(r[-1], Diag):
                r[-1].update(_phase_read())
        return out
    GPUModelRunner.sample_tokens = timed_sample_tokens
NOSYNC = [False]
# 10-14 (main): a synchronize from this harness inside an unsynced window is counted; a cb window with any fails the run (*_SYNCFAIL.json)
HSYNC = [0]
_orig_cuda_sync = torch.cuda.synchronize


def _hsync_guard(*a_, **k_):
    if NOSYNC[0] and sys._getframe(1).f_code.co_filename.endswith(("drive_sweep.py", "drive_diag.py")):
        HSYNC[0] += 1
    return _orig_cuda_sync(*a_, **k_)
torch.cuda.synchronize = _hsync_guard
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


def _split_counts():  # (split calls, split builds, catch-up sets) of the head's executor-level split; zeros on older builds
    c = tuple(torch._C._host_trace_split_counts()) if hasattr(torch._C, "_host_trace_split_counts") else (0, 0, 0)
    if hasattr(torch._C, "_host_trace_split_dirty_wholes"):  # 10-18 (ms20+ counters, 2-tuple above): + whole-exec calls under dirty_min, graph launches
        c += (torch._C._host_trace_split_dirty_wholes(), torch._C._host_trace_graph_launches())
    return c

_PRB = {}


def _ctx_probe():  # 10-18 (split-cost agent, SPLIT_CUT_COST.md "Window-2 trigger"): a fixed 1000-kernel graph replayed on the current stream
    # (warmed there, so it gives work to no extra stream); ~0.93-0.95 ms = the context's fast state, ~1.33 ms = slow (> 8 working streams)
    if os.environ.get("DRIVE_CTX_PROBE", "1") != "1":
        return None
    if not _PRB:
        x = torch.zeros(1024, device="cuda")
        for _ in range(3):
            x.add_(1.0)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=torch.cuda.Stream()):
            for _ in range(1000):
                x.add_(1.0)
        g.replay()  # one untimed replay first: the first replay of a fresh exec reads slow (upload)
        torch.cuda.synchronize()
        _PRB.update(x=x, g=g, e0=torch.cuda.Event(enable_timing=True), e1=torch.cuda.Event(enable_timing=True))
    torch.cuda.synchronize()
    _PRB["e0"].record()
    _PRB["g"].replay()
    _PRB["e1"].record()
    torch.cuda.synchronize()
    return round(_PRB["e0"].elapsed_time(_PRB["e1"]), 4)



NS_ROWS = []  # unsynced window rows: (num_reqs, num_tokens, cg_mode, padded_tokens, kernel nodes set, memory nodes set, learns, traces,
# execute_model entry time, execute_model host seconds (10-14: per-step intervals split by size changed / same, no sync added)


def _exec_with_diag(self, so, *args, **kw):
    if kw.get("dummy_run"):
        return _exec_diag(self, so, *args, **kw)
    if NOSYNC[0]:
        t_in = time.perf_counter()
        _V = globals().get("V")
        nd0 = len(DIAG)
        k0 = (torch._C._host_trace_kernel_node_sets(), torch._C._host_trace_memory_node_sets()) if hasattr(torch._C, "_host_trace_kernel_node_sets") else (0, 0)
        h0 = (_V.provider.harvests, sum(e.traces for e in _V.entries.values())) if _V is not None else (0, 0)
        sp0 = _split_counts()
        out = _exec_diag(self, so, *args, **kw)
        dg = DIAG[nd0] if len(DIAG) > nd0 else {}
        k1 = (torch._C._host_trace_kernel_node_sets(), torch._C._host_trace_memory_node_sets()) if hasattr(torch._C, "_host_trace_kernel_node_sets") else (0, 0)
        h1 = (_V.provider.harvests, sum(e.traces for e in _V.entries.values())) if _V is not None else (0, 0)
        NS_ROWS.append((len(so.num_scheduled_tokens), so.total_num_scheduled_tokens, dg.get("cg_mode"), dg.get("padded_tokens"),
                        k1[0] - k0[0], k1[1] - k0[1], h1[0] - h0[0], h1[1] - h0[1], t_in, time.perf_counter() - t_in,
                        [x - y for x, y in zip(_split_counts(), sp0)]))
        return out
    t_step0 = time.perf_counter()  # 10-15: the step's start, to match GC passes to slow steps
    c0, nd = _ctr_flat(), len(DIAG)
    _krec_pre(so)
    if PHASES_ON:
        _phase_reset()
    _V = globals().get("V")
    h0 = (_V.provider.harvests, sum(e.traces for e in _V.entries.values())) if _V is not None else None
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
    d["t0"] = t_step0
    d["mix"] = mix
    d["counters"] = {k: c1[k] - c0.get(k, 0) for k in c1 if c1[k] != c0.get(k, 0)}
    if h0 is not None:  # ours: cuBLAS keys learned and traces taken during this step
        d["learns"] = _V.provider.harvests - h0[0]
        d["traces"] = sum(e.traces for e in _V.entries.values()) - h0[1]
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
if os.environ.get("DRIVE_CC_JSON"):  # capture-size study (10-13): merged into the arm's compilation_config, e.g. {"max_cudagraph_capture_size": 2304}
    extra.setdefault("compilation_config", {}).update(json.loads(os.environ["DRIVE_CC_JSON"]))
# capture stats: capture_model's wall seconds and returned graph bytes (free-memory delta), and every CUDA graph capture begun
CAP = {"capture_model_s": 0.0, "graph_bytes": 0, "capture_begins": 0}
_orig_capture_model = GPUModelRunner.capture_model


def _timed_capture_model(self):
    t = time.perf_counter()
    out = _orig_capture_model(self)
    CAP["capture_model_s"] += time.perf_counter() - t
    CAP["graph_bytes"] += int(out or 0)
    return out
GPUModelRunner.capture_model = _timed_capture_model
_orig_cb = torch.cuda.CUDAGraph.capture_begin


def _counted_cb(self, *a_, **k_):
    CAP["capture_begins"] += 1
    return _orig_cb(self, *a_, **k_)
torch.cuda.CUDAGraph.capture_begin = _counted_cb
if a.max_batched:
    extra["max_num_batched_tokens"] = a.max_batched
if a.max_seqs:
    extra["max_num_seqs"] = a.max_seqs
if os.environ.get("ARMV_INDUCTOR_HOSTS") == "1":  # Vc (serve lane, 10-12): Inductor's host helpers + Triton launches under the trace
    sys.path.insert(0, os.environ.get("VLLM_PORTS_DIR", "/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration/vllm/vllm_ports"))
    import inductor_hosts

    print("INDUCTOR_HOSTS installed", inductor_hosts.install(), flush=True)
llm = LLM(model=a.model, tensor_parallel_size=1, enforce_eager=a.arm not in ("default", "fullnc", "fullnc_full", "VAf", "Vc", "compnc"), attention_backend="FLASHINFER", **extra,
          enable_prefix_caching=False, gpu_memory_utilization=float(os.environ.get("INTEG_GPU_UTIL", "0.6")), **({"kv_cache_memory_bytes": int(os.environ["INTEG_KV_BYTES"])} if os.environ.get("INTEG_KV_BYTES") else {}), max_model_len=a.max_model_len, seed=0)
R = {"arm": a.arm, "init_s": time.perf_counter() - t0, "decode": {}, "prefill": {}, "mixed": {}, "mixed_specs": {}, "rows": {}}
R["util"] = float(os.environ.get("INTEG_GPU_UTIL", "0.6"))
_cc = llm.llm_engine.vllm_config.compilation_config
_runner = llm.llm_engine.model_executor.driver_worker.worker.model_runner
if os.environ.get("DRIVE_PROFILE_DIR"):
    # breakdown (serve lane, 10-13): one profiled full step (execute_model through sample_tokens) per class "reqs:tokens" in
    # DRIVE_PROFILE_CLASSES, at its DRIVE_PROFILE_AT-th occurrence; phases annotated: input prep, metadata (prepare_attn and the
    # model state's attention build), execute_model (forward = execute_model minus those), sample_tokens; chrome trace per class (serve/ix/prof_breakdown.py)
    import torch.profiler as _tp

    _PDIR = os.environ["DRIVE_PROFILE_DIR"]
    _PCLS = {tuple(map(int, c.split(":"))) for c in os.environ.get("DRIVE_PROFILE_CLASSES", "1:1,64:64,65:1088").split(",")}
    _PAT = int(os.environ.get("DRIVE_PROFILE_AT", "6"))
    _pseen, _pcur = {}, [None]

    def _annot(obj, name, tag):
        f = getattr(obj, name)

        def g(*a, **k):
            with _tp.record_function(tag):
                return f(*a, **k)
        setattr(obj, name, g)
    _annot(_runner, "prepare_inputs", "PH_input_prep")
    _annot(_runner, "prepare_attn", "PH_metadata_tables")
    _annot(_runner.model_state, "prepare_attn", "PH_metadata_build")
    _pexec, _psample = GPUModelRunner.execute_model, GPUModelRunner.sample_tokens

    def _prof_exec(self, so, *a, **k):
        if not k.get("dummy_run"):
            c = (len(so.num_scheduled_tokens), so.total_num_scheduled_tokens)
            if c in _PCLS:
                _pseen[c] = _pseen.get(c, 0) + 1
                if _pseen[c] == _PAT and _pcur[0] is None:
                    torch.cuda.synchronize()
                    prof = _tp.profile(activities=[_tp.ProfilerActivity.CPU, _tp.ProfilerActivity.CUDA])
                    prof.__enter__()
                    _pcur[0] = (c, prof)
        if _pcur[0] is not None:
            with _tp.record_function("PH_execute_model"):
                return _pexec(self, so, *a, **k)
        return _pexec(self, so, *a, **k)

    def _prof_sample(self, *a, **k):
        if _pcur[0] is None:
            return _psample(self, *a, **k)
        with _tp.record_function("PH_sample_tokens"):
            out = _psample(self, *a, **k)
        torch.cuda.synchronize()
        c, prof = _pcur[0]
        _pcur[0] = None
        prof.__exit__(None, None, None)
        os.makedirs(_PDIR, exist_ok=True)
        prof.export_chrome_trace(os.path.join(_PDIR, f"{a_arm_tag()}_{c[0]}x{c[1]}.json"))
        return out
    a_arm_tag = lambda: os.environ.get("DRIVE_PROFILE_TAG", a.arm)
    GPUModelRunner.execute_model, GPUModelRunner.sample_tokens = _prof_exec, _prof_sample
R["cg_config"] = {"mode": str(_cc.mode), "cudagraph_mode": str(_cc.cudagraph_mode), "capture_sizes": list(_cc.cudagraph_capture_sizes or []),
                  "max_capture_size": _cc.max_cudagraph_capture_size, "use_breakable_cg": getattr(_runner.cudagraph_manager, "use_breakable_cg", None),
                  "max_num_batched_tokens": llm.llm_engine.vllm_config.scheduler_config.max_num_batched_tokens,
                  "chunked_prefill": llm.llm_engine.vllm_config.scheduler_config.enable_chunked_prefill}
R["reserved_after_init_mib"] = torch.cuda.memory_reserved() / 2**20
_mgr = _runner.cudagraph_manager
R["capture"] = dict(CAP, descs={str(m): len(d) for m, d in getattr(_mgr, "_capture_descs", {}).items()}, graphs=len(getattr(_mgr, "graphs", {})),
                    max_num_seqs=llm.llm_engine.vllm_config.scheduler_config.max_num_seqs,
                    cudagraph_memory_estimate=getattr(llm.llm_engine.model_executor.driver_worker.worker, "cudagraph_memory_estimate", None),
                    breakable_env=os.environ.get("VLLM_USE_BREAKABLE_CUDAGRAPH"))
print("capture", R["capture"], flush=True)
PHASE[0] = "bench"
V = None
if ARM_V:
    runner = llm.llm_engine.model_executor.driver_worker.worker.model_runner
    import adapter_cpp  # install_sgl_cpp (python_vllm_cpp.sh): the bound decode entry unless ARMV_BOUND=0
    V = (adapter_cpp.ArmVcpp if adapter_cpp.BOUND else armv_adapter.ArmV)(runner)
    V.install()
    V.check = a.check
    runner.model.forward = synced(V.forward, fwd_sink)
    if os.environ.get("DRIVE_SPLIT"):  # 10-15: the executor-level split (head ms7+): torch._C._host_trace_split(threshold); -1 = off
        torch._C._host_trace_split(int(os.environ["DRIVE_SPLIT"]))
        if os.environ.get("DRIVE_SPLIT_CATCH_UP"):
            torch._C._host_trace_split_catch_up(int(os.environ["DRIVE_SPLIT_CATCH_UP"]))
        R["split"] = {"threshold": int(os.environ["DRIVE_SPLIT"]), "catch_up": torch._C._host_trace_split_catch_up(None)}
    if os.environ.get("DRIVE_FORM_HELD") == "1":  # 10-14 (cbh4c): the runtime keeps each form's held node state (form_held_enabled)
        torch._C._host_trace_form_held(True)
        R["form_held"] = True
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
    cg = collections.Counter(r[-1].get("cg_mode") for r in rows if isinstance(r[-1], Diag))
    if cg:
        out["cg_modes"] = dict(cg)
    pads = [r[-1]["padded_tokens"] - r[-1]["num_toks"] for r in rows if isinstance(r[-1], Diag) and r[-1].get("padded_tokens") is not None and r[-1].get("num_toks") is not None]
    if pads:
        out["pad_tokens_mean"] = statistics.fmean(pads)
    lr = [r[-1] for r in rows if isinstance(r[-1], Diag) and "learns" in r[-1]]
    if lr:  # ours: learns and traces inside the summarized (measured) steps
        out["learns_in_rows"] = sum(x["learns"] for x in lr)
        out["traces_in_rows"] = sum(x["traces"] for x in lr)
    phs = [r[-1] for r in rows if isinstance(r[-1], Diag) and "ph_us" in r[-1]]
    if phs:
        out["ph_us"] = {k: round(statistics.median(d["ph_us"].get(k, 0.0) for d in phs), 1) for k in sorted({k for d in phs for k in d["ph_us"]})}
        out["patch"] = [statistics.median(d["patch"][i] for d in phs) for i in range(len(phs[0]["patch"]))]
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


WARMED = [False]


def _range_warmup():
    """One prefill per token count lo..hi, a short decode at every bs 1..bsmax, three mixed steps in a row (an entry's first call runs
    eagerly, its second traces): first-time traces and cuBLAS binds before any measured step. Returns its seconds."""
    t_w = time.perf_counter()
    if a.warm_range:
        lo_, hi_, bsmax_ = a.warm_range
        for L in range(lo_, hi_ + 1):
            llm.generate(prompts(1, L), sp(1), use_tqdm=False)
        for bs in range(1, bsmax_ + 1):
            llm.generate(prompts(bs, 64), sp(3), use_tqdm=False)
    _ix_fill("cbwarm", [256] * 4)
    for j in range(3):
        llm.llm_engine.add_request(f"cbwarm_p{j}", kv_prompt(300 + j), sp(1))
        llm.llm_engine.step()
    _ix_drain("cbwarm", 4)
    if os.environ.get("DRIVE_MIXED_WARM") == "1":  # cbh4+ ours (10-14): mixed steps with 16 decoders and short prefill chunks (the churn's
        _ix_fill("cbwarm2", [256] * 16)            # 64-512-token prompts; cbh4 vrs r1 traced a 15-decode + 82-token mixed step in a window)
        for j, L in enumerate((64, 80, 96, 112, 128, 160, 192, 256, 384, 512)):
            llm.llm_engine.add_request(f"cbwarm2_p{j}", kv_prompt(L), sp(1))
            llm.llm_engine.step()
        _ix_drain("cbwarm2", 16)
    WARMED[0] = True
    return time.perf_counter() - t_w


def _gc_freeze_ours(when):
    """Ours only (DRIVE_GC_FREEZE=1, user decision 10-13): gc.freeze() once after our warm-up, as vLLM's freeze_gc_heap() does after its own;
    tracked-object counts before and after in R["gc_freeze"]."""
    import gc as _g
    if V is None or os.environ.get("DRIVE_GC_FREEZE") != "1" or "gc_freeze" in R:
        return
    before = {"count": _g.get_count(), "tracked": len(_g.get_objects()), "frozen": _g.get_freeze_count()}
    _g.freeze()
    if hasattr(V, "refreeze_armed"):  # ARMV_REFREEZE: post-warm-up traces / learns re-freeze from here on
        V.refreeze_armed = True
    R["gc_freeze"] = {"when": when, "before": before, "after": {"count": _g.get_count(), "tracked": len(_g.get_objects()), "frozen": _g.get_freeze_count()}}
    print("gc_freeze", R["gc_freeze"], flush=True)


if a.warm_range and not a.cb:  # sweeps with --warm-range: the same untimed warm-up before the first measured section, then the freeze
    R["sweep_warmup_s"] = _range_warmup()
    _gc_freeze_ours("after the sweep warm-up")
if os.environ.get("DRIVE_EMPTY_CACHE") == "1" and not a.cb:  # 10-15: every arm, once before the first measured section (after ours' warm-up)
    R["empty_cache_gib"] = [torch.cuda.memory_reserved() / 2**30]
    torch.cuda.empty_cache()
    R["empty_cache_gib"].append(torch.cuda.memory_reserved() / 2**30)
for bs in a.batch_size:
    LOG.clear()
    llm.generate(prompts(bs, 256), sp(a.decode_tokens), use_tqdm=False)
    dec = [r for r in LOG if r[0] == bs and r[1] == bs]
    if len(dec) < 8:
        print("LOG", LOG[:8], flush=True)
    R["decode"][bs] = summ(dec[4:]) if len(dec) > 4 else {"step_ms": None, "n": len(dec)}
    R["rows"][f"decode{bs}"] = [list(r[:4]) for r in LOG]
    print("decode", bs, R["decode"][bs], flush=True)
    if str(bs) in os.environ.get("DRIVE_PERF_DECODE", "").split(","):  # entry profile (10-13): perf samples of a long decode at bs, main thread
        import subprocess, threading as _th
        _pf = f"{os.environ['DRIVE_PERF_OUT']}_bs{bs}.data"
        _pp = subprocess.Popen(["perf", "record", "-F", "1999", "--call-graph", "dwarf,32768", "-t", str(_th.get_native_id()), "-D", "3000", "-o", _pf, "--", "sleep", "6"])
        LOG.clear()
        llm.generate(prompts(bs, 256), sp(int(os.environ.get("DRIVE_PERF_TOKENS", "1500"))), use_tqdm=False)
        _pp.wait()
        R.setdefault("perf_decode", {})[bs] = {"file": _pf, "steps": len(LOG), **summ([r for r in LOG if r[0] == bs and r[1] == bs][4:])}
        print("perf_decode", bs, R["perf_decode"][bs], flush=True)
import gc as _gcm
R["gc_after_warmup"] = {"tracked": len(_gcm.get_objects()), "frozen": _gcm.get_freeze_count(), "count": _gcm.get_count(), "threshold": _gcm.get_threshold()}
print("gc_after_warmup", R["gc_after_warmup"], flush=True)
R["dec_ctx"] = {}
# DRIVE_GC_DIAG=1 (untimed diagnostic, 10-14): over the measured --dec-ctx curves, gc.DEBUG_SAVEALL keeps every collected object in
# gc.garbage; after each pass a callback histograms it by type, samples allocation sites (tracemalloc) and referrer types for the top
# types, times gc.garbage.clear() (the deallocation / finalizers DEBUG_SAVEALL deferred) and clears it. Pass time without finalizers
# (the collection itself) vs the clear time (finalizers / deallocation) separates the two costs.
GC_DIAG = []
if os.environ.get("DRIVE_GC_DIAG") == "1":
    import gc as _gd
    import tracemalloc as _tm
    _tm.start(4)
    _gd_t = [0.0]
    _gd_refs_left = [8]  # gc.get_referrers walks every tracked object, frozen ones included: a few calls per run only

    def _gd_cb(phase, info):
        if phase == "start":
            _gd_t[0] = time.perf_counter()
            return
        dt = time.perf_counter() - _gd_t[0]
        g = list(_gd.garbage)
        hist = collections.Counter(type(o).__module__ + "." + type(o).__qualname__ for o in g)
        top = hist.most_common(8)
        sites = {}
        for tname, _ in top[:4]:
            objs = [o for o in g if type(o).__module__ + "." + type(o).__qualname__ == tname][:200]
            tb = collections.Counter()
            for o in objs:
                t = _tm.get_object_traceback(o)
                if t is not None:
                    fr = [f for f in t if "drive_sweep" not in f.filename][:3]
                    tb[" <- ".join(f"{os.path.basename(f.filename)}:{f.lineno}" for f in fr)] += 1
            refs = collections.Counter()
            if _gd_refs_left[0] > 0 and objs:
                _gd_refs_left[0] -= 1
                refs = collections.Counter(type(r).__module__ + "." + type(r).__qualname__ for r in _gd.get_referrers(objs[0]) if r is not g)
            sites[tname] = {"alloc": tb.most_common(4), "referrers": refs.most_common(5)}
        t1 = time.perf_counter()
        del g
        _gd.garbage.clear()
        clear_s = time.perf_counter() - t1
        GC_DIAG.append({"gen": info["generation"], "collected": info.get("collected"), "pass_ms": round(dt * 1e3, 2), "clear_ms": round(clear_s * 1e3, 2),
                        "n_garbage": sum(hist.values()), "top": top, "sites": sites})
    _gd.callbacks.append(_gd_cb)
    _gd.set_debug(_gd.DEBUG_SAVEALL)
for ctx in a.dec_ctx:  # bs requests of ctx-token prompts; the full-batch decode steps after the staggered prefill: 4 dropped, then N measured
    R["dec_ctx"][ctx] = {}
    for bs in a.dec_ctx_sizes:
        LOG.clear()
        n_out = 4 + a.dec_ctx_steps + -(-bs * ctx // 16384) + 4
        t0c = time.perf_counter()
        psi_p0 = _psi_mem_s()
        ms0 = torch.cuda.memory_stats()
        _krec_arm((bs, bs), skip=4 + a.dec_ctx_steps)
        llm.generate(prompts(bs, ctx), sp(n_out), use_tqdm=False)
        ms1 = torch.cuda.memory_stats()
        dec = [r for r in LOG if r[0] == bs and r[1] == bs]
        meas = dec[4:4 + a.dec_ctx_steps]
        d = (summ(meas) if meas else {"step_ms": None}) | {"n_full_steps": len(dec), "wall_s": time.perf_counter() - t0c}
        if meas:
            d["tpot_ms_mean"] = statistics.fmean(r[2] for r in meas) * 1e3  # mean step time over the span (KV grows by one per step)
            d["ms_per_token_mean"] = d["tpot_ms_mean"] / bs
        if GC_EVENTS:
            d["gc_passes"] = [(g_, round(s_ * 1e3, 1), c_) for g_, s_, c_, t_ in GC_EVENTS if t_ >= t0c and s_ > 0.005]  # (generation, ms, objects collected)
        if meas:  # the slowest measured step: step ms, forward ms (execute_model to the model's return), our entry us; allocator growth over the point
            mx = max(meas, key=lambda r: r[2])
            dg = mx[-1] if isinstance(mx[-1], Diag) else {}
            d["slowest_step"] = {"step_ms": round(mx[2] * 1e3, 2), "fwd_ms": round(mx[3] * 1e3, 2) if mx[3] is not None else None,
                                 "entry_us": (dg.get("ph_us") or {}).get("dispatch"), "index": meas.index(mx)}
        d["alloc_growth"] = {k: ms1.get(k, 0) - ms0.get(k, 0) for k in ("segment.all.allocated", "num_device_alloc", "num_alloc_retries")}
        if meas:  # 10-15: step-time tail and its attribution: p99, max, steps > 10x the median, GC passes inside those steps, host memory pressure
            sts = sorted(r[2] * 1e3 for r in meas)
            med_ = statistics.median(sts)
            outl = [r for r in meas if r[2] * 1e3 > 10 * med_]
            win0 = next((r[-1].get("t0") for r in meas if isinstance(r[-1], Diag) and r[-1].get("t0")), None)
            win1 = max((r[-1]["t0"] + r[2] for r in meas if isinstance(r[-1], Diag) and r[-1].get("t0")), default=None)

            def _gc_in(t_a, t_b):  # passes ending inside [t_a, t_b + 2 ms] whose start is after t_a - their duration
                return [(g_, round(s_ * 1e3, 1)) for g_, s_, c_, t_ in GC_EVENTS if t_a <= t_ <= t_b + 0.002 and s_ > 0.001]
            psi_p1 = _psi_mem_s()
            d["tail"] = {"p99_ms": sts[min(len(sts) - 1, int(0.99 * len(sts)))], "max_ms": sts[-1], "median_ms": med_, "outlier_steps": len(outl), "steps_over_3x": sum(1 for x in sts if x > 3 * med_),
                         "excess_over_3x_ms": sum(x - med_ for x in sts if x > 3 * med_),
                         "outlier_excess_ms": sum(r[2] * 1e3 - med_ for r in outl),
                         "outliers": [{"step_ms": round(r[2] * 1e3, 2), "gc": _gc_in(r[-1]["t0"], r[-1]["t0"] + r[2]) if isinstance(r[-1], Diag) and r[-1].get("t0") else None}
                                      for r in outl[:8]],
                         "gc_in_window": _gc_in(win0, win1) if win0 and win1 else None,
                         "host_mem_pressure_full_s": (psi_p1 - psi_p0) / 1e6 if psi_p0 is not None and psi_p1 is not None else None}
            mw = sum(r[2] for r in meas)
            psi_s_ = d["tail"]["host_mem_pressure_full_s"]
            d["tail"]["flagged"] = (psi_s_ is not None and psi_s_ > 0.02 * (time.perf_counter() - t0c)) or d["tail"]["outlier_excess_ms"] / 1e3 > 0.01 * mw
        if KREC["on"]:
            d["kernels"] = KREC["result"]
            _krec_arm(None)
        R["dec_ctx"][ctx][bs] = d
        print("dec_ctx", ctx, bs, {k: v for k, v in d.items() if k in ("n", "step_ms", "tpot_ms_mean", "learns_in_rows", "traces_in_rows", "cg_modes", "pad_tokens_mean")}, flush=True)
if os.environ.get("DRIVE_GC_DIAG") == "1":
    _gd.set_debug(0)
    _gd.callbacks.remove(_gd_cb)
    R["gc_diag"] = GC_DIAG
    print("gc_diag passes", len(GC_DIAG), [(x["gen"], x["collected"], x["pass_ms"], x["clear_ms"]) for x in GC_DIAG][:20], flush=True)
for L in a.prefill_lens:
    rows = []
    for _ in range(a.prefill_reps):
        LOG.clear()
        llm.generate(prompts(1, L), sp(1), use_tqdm=False)
        rows += [r for r in LOG if r[1] == L]
    R["prefill"][L] = summ(rows[4:]) if len(rows) > 4 else {"step_ms": None, "n": len(rows)}
    if KREC["on"]:
        _krec_arm((1, L))
        LOG.clear()
        llm.generate(prompts(1, L), sp(1), use_tqdm=False)
        R["prefill"][L]["kernels"] = KREC["result"]
        _krec_arm(None)
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



for spec in a.ix_mixed:  # one step = ND decodes + one P-token chunk (P + ND <= max_num_batched_tokens), then one decode-only step
    shape, _, rng = spec.partition("@")  # PxND[@LO-HI]: decoder KV spread over [LO, HI] (default --ix-mixed-kv)
    P, nd = map(int, shape.split("x"))
    lo_hi = tuple(map(int, rng.split("-"))) if rng else tuple(a.ix_mixed_kv)
    tag, eng = f"ixm{P}_{nd}_{lo_hi[0]}", llm.llm_engine
    _ix_fill(tag, spread(nd, *lo_hi))
    LOG.clear()
    for j in range(a.ix_reps):
        eng.add_request(f"{tag}_p{j}", kv_prompt(P), sp(1))
        eng.step()
        eng.step()
    rows = [r for r in LOG if r[1] == P + nd and r[0] == nd + 1]
    R["ix"][f"mixed_{spec}"] = (summ(rows[2:]) if len(rows) > 2 else {"step_ms": None}) | {"n_rows": len(rows), "first_ms": [round(r[2] * 1e3, 2) for r in rows[:2]],
                                                                                         "kv_min_med_max": _ix_kv(rows), "shapes": sorted({(r[0], r[1]) for r in LOG})}
    if KREC["on"]:
        _krec_arm((nd + 1, P + nd))
        eng.add_request(f"{tag}_pk", kv_prompt(P), sp(1))
        eng.step()
        eng.step()
        R["ix"][f"mixed_{spec}"]["kernels"] = KREC["result"]
        _krec_arm(None)
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
for spec in a.alt_decode:  # the first A decoders run every step, the other B - A every second step: shapes A, B, A, B, ...
    pair, _, rng = spec.partition("@")
    A, B = map(int, pair.split(":"))
    lo, hi = map(int, (rng or "1024-2048").split("-"))
    tag, eng = f"alt{A}_{B}", llm.llm_engine
    _ix_fill(tag, spread(max(A, B), lo, hi))
    sched = eng.engine_core.engine_core.scheduler
    budget0 = sched.max_num_scheduled_tokens
    LOG.clear()
    try:
        for k in range(2 * (a.ix_reps + 14)):
            sched.max_num_scheduled_tokens = A if k % 2 == 0 else B
            eng.step()
    finally:
        sched.max_num_scheduled_tokens = budget0
    rows = [r for r in LOG if r[0] == r[1] and r[0] in (A, B)]
    R["ix"][f"alt_{A}_{B}"] = (summ(rows[8:]) if len(rows) > 8 else {"step_ms": None}) | {"n_rows": len(rows), "shapes_seq": [r[0] for r in rows[:12]],
                                                                                     "kv_min_med_max": _ix_kv(rows)}
    for x in (A, B):
        sub = [r for r in rows[8:] if r[0] == x]
        if sub:
            R["ix"][f"alt_{A}_{B}"][f"bs{x}"] = summ(sub)
    _ix_drain(tag, max(A, B))
    print("ix alt", A, B, {k: v for k, v in R["ix"][f"alt_{A}_{B}"].items()}, flush=True)
R["cb"] = {}
CB_NOSYNC = os.environ.get("DRIVE_CB_NOSYNC") == "1"  # churn window with no syncs (async scheduling as shipped); wall time only


def _psi_avg60():
    try:
        for ln in open("/proc/pressure/memory"):
            if ln.startswith("full"):
                return float(ln.split()[2].split("=")[1])
    except OSError:
        return None


def _psi_mem():  # 10-14: host memory pressure (kswapd / swap on the shared box): /proc/pressure/memory "full" total, us
    try:
        for ln in open("/proc/pressure/memory"):
            if ln.startswith("full"):
                return int(ln.split("total=")[1])
    except OSError:
        return None
import faulthandler
_STALL_F = open(a.out.replace(".json", "_stalls.txt"), "w") if CB_NOSYNC else None
SYNC_DEBUG = os.environ.get("DRIVE_SYNC_DEBUG") == "1"  # torch.cuda sync debug mode in the unsynced window: count syncing calls
if (a.cb or a.ifx_c) and not WARMED[0]:  # untimed warm-up (every arm), our equivalent of vLLM's capture warm-up: every family traced, every cuBLAS size bound
    R["cb_warmup_s"] = _range_warmup()
    if V is not None:  # 10-14: what ours' warm-up paid (traces, cuBLAS learns / binds, seconds in trace and slow calls)
        R["cb_warmup_ours"] = {"traces": sum(e.traces for e in V.entries.values()), "harvests": V.provider.harvests, "bindings": len(V.provider.bindings),
                               "bindings_loaded": getattr(V, "bindings_loaded", None), "trace_call_s": sum(getattr(V, "trace_call_s", {}).values()),
                               "slow_call_s": sum(getattr(V, "slow_s", {}).values())}
        print("cb_warmup_ours", R["cb_warmup_s"], R["cb_warmup_ours"], flush=True)
    _gc_freeze_ours("after the churn warm-up")
    if os.environ.get("DRIVE_EMPTY_CACHE") == "1":  # 10-14 A/B: release the allocator's cached blocks once after the warm-up
        R["empty_cache_gib"] = [torch.cuda.memory_reserved() / 2**30]
        torch.cuda.empty_cache()
        R["empty_cache_gib"].append(torch.cuda.memory_reserved() / 2**30)
_CB_SEQ = [int(x) for x in os.environ.get("DRIVE_CB_SPLIT_SEQ", "").split(",") if x]  # 10-18: split off/on per window in one process (--cb 64 64 64 64 + "0,1,0,1"; the 2nd pair is the reported one)
for _wi, C in enumerate(a.cb):  # continuous batching: C in flight, seeded output / prompt lengths, the same request sequence in every arm
    eng, rng = llm.llm_engine, torch.Generator().manual_seed(1000 + C)
    nxt = [0]
    _cbp = f"cb{C}_w{_wi}" if _CB_SEQ else f"cb{C}"
    if _CB_SEQ:
        torch._C._host_trace_split(bool(_CB_SEQ[_wi]))
        _sc0 = _split_counts()

    def admit():
        i = nxt[0]
        nxt[0] += 1
        L = int(torch.randint(a.cb_prompt[0], a.cb_prompt[1] + 1, (1,), generator=rng))
        n = int(torch.randint(a.cb_out[0], a.cb_out[1] + 1, (1,), generator=rng))
        ids = torch.randint(1000, 100000, (L,), generator=rng).tolist()
        eng.add_request(f"{_cbp}_{i}", TokensPrompt(prompt_token_ids=ids), sp(n))
    for _ in range(C):
        admit()
    LOG.clear()
    done = 0
    t_win = toks = 0
    for k in range(a.cb_warm + a.cb_steps):
        if k == a.cb_warm:
            _prb0 = _ctx_probe()  # before the window (synced; outside its wall)
            LOG.clear()
            if CB_NOSYNC:
                torch.cuda.synchronize()
                NS_ROWS.clear()
                NOSYNC[0] = True
                if SYNC_DEBUG:
                    import warnings as _w
                    _wctx = _w.catch_warnings(record=True)
                    _wrec = _wctx.__enter__()
                    _w.simplefilter("always")
                    torch.cuda.set_sync_debug_mode(1)
            t0w, done0 = time.perf_counter(), done
            hs0 = HSYNC[0]
            ms_w0 = torch.cuda.memory_stats()  # 10-14: allocator events in the window (OOM retries free the cache and sync)
            psi0 = _psi_mem()
            psi60_0 = _psi_avg60()
            v0 = (V.provider.harvests, sum(e.traces for e in V.entries.values()), len(V.provider.bindings)) if V is not None else None
        if CB_NOSYNC and k >= a.cb_warm:  # 10-14: a step over 0.5 s dumps every thread's Python stack (stall diagnosis; faulthandler's C thread)
            faulthandler.dump_traceback_later(0.5, repeat=False, file=_STALL_F)
        outs = eng.step()
        if CB_NOSYNC and k >= a.cb_warm:
            faulthandler.cancel_dump_traceback_later()
        fin = sum(1 for o in outs if o.finished)
        done += fin
        for _ in range(fin):
            admit()
    hsyncs = HSYNC[0] - hs0  # before the window-end drain below (part of the wall measurement)
    ms_w1 = torch.cuda.memory_stats()
    psi1 = _psi_mem()
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0w
    sync_warn = None
    if CB_NOSYNC:
        NOSYNC[0] = False
        if SYNC_DEBUG:
            torch.cuda.set_sync_debug_mode(0)
            _wctx.__exit__(None, None, None)
            sync_warn = dict(collections.Counter(str(x.message)[:120] for x in _wrec).most_common(10))
        # synthetic rows in LOG's layout for the counters below: (reqs, tokens, step s unknown -> 0, None, Diag)
        ns = list(NS_ROWS)
        LOG[:] = [(r[0], r[1], 0.0, None, Diag({"cg_mode": r[2], "padded_tokens": r[3], "num_toks": r[1], "patch": [0, r[4], r[5]], "learns": r[6], "traces": r[7]}))
                  for r in NS_ROWS]
    _prb1 = _ctx_probe()  # after the window (its wall is already taken)
    rows = [r for r in LOG if r[0] > 0]
    # tokens produced per step: one per scheduled request whose prompt is complete after the step (decodes, plus a prefill's last chunk)
    out_tok = sum(r[0] for r in rows)
    dec = [r for r in rows if r[0] == r[1]]
    mixed = [r for r in rows if r[1] > r[0]]
    changed = sum(1 for p_, q_ in zip(rows, rows[1:]) if (p_[0], p_[1]) != (q_[0], q_[1]))
    dchanged = sum(1 for p_, q_ in zip(dec, dec[1:]) if p_[0] != q_[0])
    step_sum = sum(r[2] for r in rows)
    d = {"C": C, "steps": len(rows), "wall_s": wall, "synced_steps": not CB_NOSYNC, "harness_syncs_in_window": hsyncs,
         "alloc_in_window": {k: ms_w1.get(k, 0) - ms_w0.get(k, 0) for k in ("num_alloc_retries", "num_device_alloc", "num_device_free", "num_ooms", "num_sync_all_streams")}
         | {"reserved_start_gib": ms_w0.get("reserved_bytes.all.current", 0) / 2**30, "reserved_end_gib": ms_w1.get("reserved_bytes.all.current", 0) / 2**30},
         "host_mem_pressure_full_s": (psi1 - psi0) / 1e6 if psi0 is not None and psi1 is not None else None,  # PSI: all tasks stalled on memory
         "psi_full_avg60_at_window_start": psi60_0, "sync_debug_warnings": sync_warn, "out_tokens_approx": out_tok, "wall_ms_per_token": wall * 1e3 / max(out_tok, 1),
         "step_ms_per_token": step_sum * 1e3 / max(out_tok, 1), "step_ms_mean": step_sum * 1e3 / max(len(rows), 1), "finished": done - done0,
         "mixed_steps": len(mixed), "mixed_frac": len(mixed) / max(len(rows), 1), "shape_changed_frac": changed / max(len(rows) - 1, 1),
         "decode_bs_changed_frac": dchanged / max(len(dec) - 1, 1), "decode_steps": len(dec),
         "decode": summ(dec) if dec else None, "mixed": summ(mixed) if mixed else None}
    pads = [r[-1]["padded_tokens"] - r[-1]["num_toks"] for r in rows if isinstance(r[-1], Diag) and r[-1].get("padded_tokens") is not None and r[-1].get("num_toks") is not None]
    d["pad_tokens_mean"] = statistics.fmean(pads) if pads else None
    d["cg_modes"] = dict(collections.Counter(r[-1].get("cg_mode") for r in rows if isinstance(r[-1], Diag)))
    phs = [r[-1] for r in rows if isinstance(r[-1], Diag) and "patch" in r[-1]]
    if phs and V is not None:
        d["nodes_set_mean"] = statistics.fmean(x["patch"][1] for x in phs)
        d["mem_nodes_set_mean"] = statistics.fmean(x["patch"][2] for x in phs)
        if all("ph_us" in x for x in phs):
            d["patch_us_mean"] = statistics.fmean(x["ph_us"].get("patch", 0.0) for x in phs)
            d["entry_us_mean"] = statistics.fmean(x["ph_us"].get("dispatch", 0.0) for x in phs)
    if V is not None:  # ours: cuBLAS keys learned and traces taken inside the measured window
        d["learns_in_window"] = V.provider.harvests - v0[0]
        d["traces_in_window"] = sum(e.traces for e in V.entries.values()) - v0[1]
        d["binds_in_window"] = len(V.provider.bindings) - v0[2]  # keys newly bound in the window (a learn, or a stored key's first use)
    eng.abort_request([f"{_cbp}_{i}" for i in range(nxt[0])])
    while eng.has_unfinished_requests():
        eng.step()
    # every measured step: requests, tokens, step ms, cudagraph mode, padded tokens, kernel / memory nodes set, entry patch / total us
    d["steps_log"] = [[r[0], r[1], round(r[2] * 1e3, 3), r[-1].get("cg_mode") if isinstance(r[-1], Diag) else None,
                       r[-1].get("padded_tokens") if isinstance(r[-1], Diag) else None,
                       (r[-1].get("patch") or [None, None, None])[1] if isinstance(r[-1], Diag) else None,
                       (r[-1].get("patch") or [None, None, None])[2] if isinstance(r[-1], Diag) else None,
                       round((r[-1].get("ph_us") or {}).get("patch", 0.0), 1) if isinstance(r[-1], Diag) else None,
                       round((r[-1].get("ph_us") or {}).get("dispatch", 0.0), 1) if isinstance(r[-1], Diag) else None] for r in rows]
    if CB_NOSYNC and len(ns) > 2:  # per-step interval (execute_model entry to the next one) and execute_model host time, by size changed vs same
        st_ = [[r[0], r[1], round((q[8] - r[8]) * 1e3, 3), round(r[9] * 1e3, 3), int((r[0], r[1]) != (p[0], p[1])), (r[10] if len(r) > 10 else [0, 0, 0])]
               for p, r, q in zip(ns, ns[1:], ns[2:])]
        d["ns_steps"] = st_  # [reqs, tokens, interval ms, execute_model host ms, size changed vs previous step]
        split = {}
        for lab, sel in (("all", lambda x: True), ("decode", lambda x: x[0] == x[1]), ("mixed", lambda x: x[1] > x[0])):
            for ch in (0, 1):
                xs = [x for x in st_ if sel(x) and x[4] == ch]
                if xs:
                    split[f"{lab}_{'changed' if ch else 'same'}"] = {"n": len(xs), "interval_ms_median": statistics.median(x[2] for x in xs),
                                                                    "interval_ms_mean": statistics.fmean(x[2] for x in xs),
                                                                    "exec_host_ms_median": statistics.median(x[3] for x in xs),
                                                                    "split_calls_mean": statistics.fmean(x[5][0] for x in xs),
                                                                    "split_counts_sum": [sum(x[5][i] for x in xs) for i in range(len(xs[0][5]))]}
        d["ns_split"] = split
        # 10-14 unified host-noise flag (all lanes): host memory pressure > 1% of the window's wall, or a step > 10x the window median
        # interval (decode steps; mixed steps carry a prefill chunk and are structurally several x longer)
        # (flag on outliers only when their excess over the median is > 1% of the wall, the same bar as PSI)
        med_iv = statistics.median(x[2] for x in st_)
        outl = [x for x in st_ if x[0] == x[1] and x[2] > 10 * med_iv]
        excess = sum(x[2] - med_iv for x in outl) / 1e3
        psi_s = d.get("host_mem_pressure_full_s")
        d["host_flag"] = {"psi_frac": (psi_s / wall) if psi_s is not None else None, "outlier_steps": len(outl), "outlier_excess_s": excess,
                          "flagged": excess > 0.01 * wall or (psi_s is not None and psi_s > 0.02 * wall)}  # PSI bar 2% (main 10-14 21:15)
    d["window_t0_from_start_s"] = t0w - T_START  # 10-18: window position in the process (2nd-window transient check)
    d["ctx_probe_ms"] = [_prb0, _prb1]  # fast ~0.93-0.95, slow ~1.33 (GB300)
    if _CB_SEQ:
        d["split_on"] = bool(_CB_SEQ[_wi]); d["window_index"] = _wi; d["split_counts_delta"] = [y - x for x, y in zip(_sc0, _split_counts())]
    R["cb"][f"{C}_w{_wi}_s{_CB_SEQ[_wi]}" if _CB_SEQ else str(C)] = d
    print("cb", C, {k: v for k, v in d.items() if k not in ("decode", "mixed")}, flush=True)
R["ifx"] = {}
if a.ifx_c:  # 10-19 InferenceX-shaped fixed-seq (r27 drive_sweep27's section): random prompts, lengths uniform in [ratio * L, L], ignore_eos,
    # mult * C prompts submitted at once with at most C in flight; streamed (DELTA) outputs; per request TTFT = first token - admission,
    # TPOT = (finish - first token) / (n - 1). Window = first admission .. last finish + the end barrier (unsynced with DRIVE_CB_NOSYNC).
    from vllm.sampling_params import RequestOutputKind
    import numpy as _np
    L_in, L_out = a.ifx_shape
    sp_ifx = lambda n: SamplingParams(max_tokens=n, ignore_eos=True, temperature=0.0, output_kind=RequestOutputKind.DELTA)
for C in a.ifx_c:
    N = a.ifx_mult * C
    rng = torch.Generator().manual_seed(3000 + 7 * C + L_in + L_out)  # the same request list in every arm and pass
    reqs = []
    for i in range(N):
        L = int(torch.randint(int(a.ifx_ratio * L_in), L_in + 1, (1,), generator=rng))
        n = int(torch.randint(int(a.ifx_ratio * L_out), L_out + 1, (1,), generator=rng))
        reqs.append((torch.randint(1000, 100000, (L,), generator=rng).tolist(), n))
    eng = llm.llm_engine

    def ifx_pass(n_req, tag, out_cap=None):
        nxt, done, t_adm, t_first, t_fin, ntok = [0], 0, {}, {}, {}, collections.Counter()

        def admit():
            i = nxt[0]
            nxt[0] += 1
            eng.add_request(f"{tag}_{i}", TokensPrompt(prompt_token_ids=reqs[i][0]), sp_ifx(min(reqs[i][1], out_cap or reqs[i][1])))
            t_adm[i] = time.perf_counter()
        for _ in range(min(C, n_req)):
            admit()
        while done < n_req:
            if not eng.has_unfinished_requests():
                raise RuntimeError(f"ifx {tag}: engine idle with {n_req - done} requests unfinished")
            outs = eng.step()
            now = time.perf_counter()
            for o in outs:
                i = int(o.request_id.rsplit("_", 1)[1])
                k = len(o.outputs[0].token_ids) if o.outputs else 0
                if k and i not in t_first:
                    t_first[i] = now
                ntok[i] += k
                if o.finished:
                    t_fin[i] = now
                    done += 1
            for _ in range(sum(1 for o in outs if o.finished)):
                if nxt[0] < n_req:
                    admit()
        return t_adm, t_first, t_fin, ntok
    t_p = time.perf_counter()
    for w_ in range(a.ifx_wave_reps):  # untimed: a step shape seen once per pass is still on its first (eager / trace) calls otherwise
        ifx_pass(C, f"ifx{C}v{w_}", a.ifx_wave_out)
    if a.ifx_wave_reps:
        print("ifx", C, f"wave reps done ({a.ifx_wave_reps} x {C} prompts, {time.perf_counter() - t_p:.1f} s)", flush=True)
    n_pre = min(N, max(C, round(a.ifx_prewarm_frac * N))) if a.ifx_prewarm_frac > 0 else 0
    if n_pre:
        t_p = time.perf_counter()
        ifx_pass(n_pre, f"ifx{C}w")
        print("ifx", C, f"prewarm pass done ({n_pre} prompts, {time.perf_counter() - t_p:.1f} s)", flush=True)
    _prb0 = _ctx_probe()
    LOG.clear()
    v0 = (V.provider.harvests, sum(e.traces for e in V.entries.values()), len(V.provider.bindings)) if V is not None else None
    torch.cuda.synchronize()
    NS_ROWS.clear()
    psi0, mono_w0 = _psi_mem(), time.monotonic()
    hs0 = HSYNC[0]
    t0w = time.perf_counter()
    NOSYNC[0] = CB_NOSYNC
    t_adm, t_first, t_fin, ntok = ifx_pass(N, f"ifx{C}")
    hsyncs = HSYNC[0] - hs0
    NOSYNC[0] = False
    torch.cuda.synchronize()  # the window's one end barrier
    wall = time.perf_counter() - t0w
    psi1 = _psi_mem()
    rows = [(r[0], r[1]) for r in NS_ROWS] if CB_NOSYNC else [(r[0], r[1]) for r in LOG if r[0] > 0]
    out_tok, in_tok = sum(ntok.values()), sum(len(reqs[i][0]) for i in range(N))
    ttft = [(t_first[i] - t_adm[i]) * 1e3 for i in range(N)]
    tpot = [(t_fin[i] - t_first[i]) * 1e3 / (ntok[i] - 1) for i in range(N) if ntok[i] > 1]
    e2e = [(t_fin[i] - t_adm[i]) * 1e3 for i in range(N)]
    pct = lambda xs, q: float(_np.percentile(xs, q))
    d = {"C": C, "n_prompts": N, "shape": [L_in, L_out], "ratio": a.ifx_ratio, "prewarm_prompts": n_pre, "wave_reps": [a.ifx_wave_reps, a.ifx_wave_out], "wall_s": wall,
         "out_tokens": out_tok, "out_tokens_expected": sum(r[1] for r in reqs), "in_tokens": in_tok,
         "output_tok_s": out_tok / wall, "total_tok_s": (out_tok + in_tok) / wall, "wall_ms_per_token": wall * 1e3 / max(out_tok, 1),
         "ttft_ms": {"median": statistics.median(ttft), "p90": pct(ttft, 90), "p99": pct(ttft, 99), "mean": statistics.fmean(ttft)},
         "tpot_ms": {"median": statistics.median(tpot), "p90": pct(tpot, 90), "mean": statistics.fmean(tpot)},
         "e2e_ms": {"median": statistics.median(e2e), "p90": pct(e2e, 90)},
         "steps": len(rows), "decode_steps": sum(1 for r in rows if r[0] == r[1]), "mixed_steps": sum(1 for r in rows if r[1] > r[0]),
         "step_tokens_max": max((r[1] for r in rows), default=0), "step_tokens_mean": statistics.fmean(r[1] for r in rows) if rows else None,
         "psi_full_s": (psi1 - psi0) / 1e6 if psi0 is not None and psi1 is not None else None, "window_mono": [mono_w0, time.monotonic()],
         "nosync": CB_NOSYNC, "harness_syncs_in_window": hsyncs if CB_NOSYNC else None, "ctx_probe_ms": [_prb0, _ctx_probe()],
         "window_t0_from_start_s": t0w - T_START,
         "per_request": {"in": [len(r[0]) for r in reqs], "out": [ntok[i] for i in range(N)], "ttft_ms": [round(x, 3) for x in ttft],
                         "e2e_ms": [round(x, 3) for x in e2e]}}
    if CB_NOSYNC:
        d["cg_modes"] = dict(collections.Counter(r[2] for r in NS_ROWS))
        pads = [r[3] - r[1] for r in NS_ROWS if r[3] is not None]
        d["pad_tokens_mean"] = statistics.fmean(pads) if pads else None
    if V is not None:
        d["learns_in_window"] = V.provider.harvests - v0[0]
        d["traces_in_window"] = sum(e.traces for e in V.entries.values()) - v0[1]
        d["binds_in_window"] = len(V.provider.bindings) - v0[2]
    R["ifx"][str(C)] = d
    print("ifx", C, {k: v for k, v in d.items() if k != "per_request"}, flush=True)
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
if V is not None and os.environ.get("DRIVE_CUT_REPORT"):  # 10-16 (memset lane): where the split auto policy would cut each entry's graphs (CPU only)
    sys.path.insert(0, "/data/eellison/src/pytorch/agent_space/paramgraph/land/core/memset/split")
    from cut_report import cut_report as _cut_report
    R["cut_report_files"] = []
    for _key, _entry in V.entries.items():
        _name = f"8b_{_key[0]}{'_vps' if _key[-1] else ''}"
        _path = os.path.join(os.environ["DRIVE_CUT_REPORT"], f"{_name}.json")
        try:
            json.dump(_cut_report(_entry), open(_path, "w"), default=str)
            R["cut_report_files"].append(_path)
        except Exception as _e:  # report the failure, keep the run's own output
            R["cut_report_files"].append(f"{_name}: {type(_e).__name__}: {_e}")
    print("cut_report", R["cut_report_files"], flush=True)
R["total_wall_s"] = time.perf_counter() - T_START
if V is not None and os.environ.get("DRIVE_STRICT_WINDOW") == "1":  # no first-time trace or cuBLAS learn in any measured window or row
    bad = [(f"{s_} {C}", k, d.get(k)) for s_ in ("cb", "ifx") for C, d in R.get(s_, {}).items() for k in ("traces_in_window", "learns_in_window") if d.get(k)]

    def _walk(x, path):
        if isinstance(x, dict):
            for k in ("learns_in_rows", "traces_in_rows"):
                if x.get(k):
                    bad.append((path, k, x[k]))
            for k, v in x.items():
                if isinstance(v, dict):
                    _walk(v, f"{path}/{k}")
    for sec in ("decode", "prefill", "ix", "dec_ctx"):
        _walk(R.get(sec, {}), sec)
    R["window_violations"] = bad
_aff = sorted(os.sched_getaffinity(0))
_c1 = _cpu_snap()
_busy = sum(1 - (_c1[c][1] - CPU0[c][1]) / max(1, _c1[c][0] - CPU0[c][0]) for c in _c1 if c in CPU0)
_ru1 = os.times()
_own_cores = ((_ru1.user + _ru1.system + _ru1.children_user + _ru1.children_system) - (_RU0.user + _RU0.system + _RU0.children_user + _RU0.children_system)) / max(1e-9, _ru1.elapsed - _RU0.elapsed)
R["placement"] = {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "gpu": torch.cuda.get_device_name(0),
                  "cpus": f"{_aff[0]}-{_aff[-1]}" if _aff == list(range(_aff[0], _aff[-1] + 1)) else ",".join(map(str, _aff)),
                  "busy_cores_own_range_mean": round(_busy, 2), "own_process_cores_mean": round(_own_cores, 2),
                  "foreign_busy_frac": round(max(0.0, _busy - _own_cores) / max(1, len(_aff)), 3)}  # > 0.15: flag the run (coordinator 10-13)  # mean busy cores over the whole run on this process's CPUs (itself included)
if V is not None:  # RD2 section-entry soundness bug (10-13): a run of ours that used any section entry is invalid
    _secs = {str(k)[:60]: e.get("sections") for k, e in R.get("V", {}).get("entries", {}).items() if e.get("sections")}
    R["section_entries"] = _secs
    if _secs:
        json.dump(R, open(a.out.replace(".json", "_SECTIONFAIL.json"), "w"), indent=1, default=str)
        print("SECTION_ENTRIES_USED (run invalid)", _secs, flush=True)
        sys.exit(4)
_hs_bad = {C: d.get("harness_syncs_in_window") for C, d in R.get("cb", {}).items() if not d.get("synced_steps") and d.get("harness_syncs_in_window")}
_hs_bad |= {f"ifx{C}": d.get("harness_syncs_in_window") for C, d in R.get("ifx", {}).items() if d.get("harness_syncs_in_window")}
_host_bad = {C: d["host_flag"] for C, d in R.get("cb", {}).items() if (d.get("host_flag") or {}).get("flagged")}
if _host_bad and os.environ.get("DRIVE_HOSTFLAG_FAIL", "1") == "1":  # rerun at the end of the set (set_chain reruns a run without its JSON)
    json.dump(R, open(a.out.replace(".json", f"_HOSTFLAG_{int(time.time())}.json"), "w"), indent=1, default=str)
    print("HOST_NOISE_FLAGGED (rerun)", _host_bad, flush=True)
    sys.exit(6)
if _hs_bad:  # 10-14: an "unsynced" window this harness synced is not one
    json.dump(R, open(a.out.replace(".json", "_SYNCFAIL.json"), "w"), indent=1, default=str)
    print("HARNESS_SYNCS_IN_UNSYNCED_WINDOW (run invalid)", _hs_bad, flush=True)
    sys.exit(5)
if R.get("window_violations"):
    json.dump(R, open(a.out.replace(".json", "_WINDOWFAIL.json"), "w"), indent=1, default=str)
    print("WINDOW_VIOLATIONS", R["window_violations"][:20], flush=True)
    sys.exit(3)
json.dump(R, open(a.out, "w"), indent=1, default=str)
print(json.dumps(R, default=str)[:20000], flush=True)
