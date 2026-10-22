# learnfast copy of r27/bench/drive_sweep27.py (10-15) + the DRIVE_STARTUP block (fair startup, ported from vllm/bench/drive_diag.py)
# drive_sweep27.py (r27 lane, 10-13): integration/vllm/bench/drive_fn.py (10-13 copy) + from drive_sweep.py (the Qwen3-8B slide sweeps):
# --alt-decode, DRIVE_STEP_WITH_SAMPLE (sample_tokens' time added to the step row, every arm), cg_modes / pad tokens per row set, and
# (10-13 06:27 copy) --cb continuous-batching churn with DRIVE_PHASES patch counters.
# drive_fn.py (serve lane, 10-12): drive_q35.py + --llm-kw (JSON LLM(...) kwargs, e.g. Flash-Next's recipe sizes) + CuTe DSL
# observation before the engine compiles any CuTe kernel when ARMV_CUTE_OBSERVE=1 (V-stock on NVFP4 models).
# Baseline driver for vLLM's own arms (no host trace), mirroring sglang/bench/drive.py:
# decode: bs requests of 256 prompt tokens, 24 steps, steps[4:]; prefill: one request of L tokens, 7 reps, reps[2:].
# step = V2 GPUModelRunner.execute_model (input prep + metadata + forward, not sampling), synced;
# fwd = the forward alone (model call, run_pw_graph or run_fullgraph), synced.
# python_vllm.sh bench/drive.py --arm eager|default --out X.json; arm V (host trace, armV/adapter.py) runs through python_vllm_v.sh.
# mixed: 8 requests decoding, then one 128-token request added per step (decode + prefill in one forward), 6 rounds.
import argparse, collections, json, os, statistics, sys, time
# r27 (coordinator 10-15), DRIVE_GC_LOG=1: every cyclic-GC pass (generation, ms, collected, start time, engine-step index at its start),
# so a slow step can be attributed to a collection (or not); every arm
STEP_IDX = [0]
GC_EVENTS = []
if os.environ.get("DRIVE_GC_LOG") == "1":
    import gc as _gc
    _gc_t0 = [0.0]

    def _gc_cb(phase, info):
        if phase == "start":
            _gc_t0[0] = time.perf_counter()
        else:
            GC_EVENTS.append((info["generation"], (time.perf_counter() - _gc_t0[0]) * 1e3, info.get("collected", 0), _gc_t0[0], STEP_IDX[0]))
    _gc.callbacks.append(_gc_cb)


def _psi_full_us():  # host memory pressure: /proc/pressure/memory "full" total (us)
    try:
        for ln in open("/proc/pressure/memory"):
            if ln.startswith("full"):
                return int(ln.split("total=")[1])
    except OSError:
        return None


def step_stats(meas, wall_s, psi_us):
    """Per-step intervals of a measured window (synced full steps): median / max / p99, steps > 10x the median, each such step's GC
    passes, every GC pass in the window, and the host-noise flag (PSI > 2% of the window or outlier excess > 1% of it)."""
    ms = [r[2] * 1e3 for r in meas]
    if not ms:
        return {}
    med, srt = statistics.median(ms), sorted(ms)
    p99 = srt[min(len(srt) - 1, int(0.99 * len(srt)))]
    spans = [(r[-1].get("t0"), r[2]) if isinstance(r[-1], Diag) else (None, r[2]) for r in meas]

    def gcs(t0, dt):
        return [(g, round(d, 2)) for g, d, _, ts, _ in GC_EVENTS if t0 is not None and t0 <= ts <= t0 + dt]
    outl = [(i, round(x, 3), gcs(*spans[i])) for i, x in enumerate(ms) if x > 10 * med]
    in_win = [g for (t0, dt) in spans for g in gcs(t0, dt)]
    excess = sum(x - med for _, x, _ in outl) / 1e3
    slow = [(i, round(x, 3), gcs(*spans[i])) for i, x in enumerate(ms) if x > 2 * med]
    return {"step_ms_all": [round(x, 3) for x in ms], "median_ms": med, "max_ms": max(ms), "p99_ms": p99, "n_10x": len(outl),
            "outliers_10x": outl, "slow_2x": slow[:20], "n_slow_2x": len(slow), "gc_in_window": in_win,
            "gc_ms_in_window": round(sum(d for _, d in in_win), 2), "wall_s": wall_s, "psi_full_s": psi_us / 1e6 if psi_us is not None else None,
            "flag": {"psi_frac": (psi_us / 1e6 / wall_s) if psi_us is not None and wall_s else None, "outlier_excess_frac": excess / wall_s if wall_s else None,
                     "flagged": bool((psi_us is not None and wall_s and psi_us / 1e6 > 0.02 * wall_s) or (wall_s and excess > 0.01 * wall_s))}}



os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
if os.environ.get("FI_FENCE") == "1":  # r27 diagnosis only: FlashInfer's GDN chunked kernel with flashinfer#5334's two TMEM fences
    sys.path.insert(0, os.path.join("/data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27", "fi_fence"))
    import fi_fence_hook  # noqa: F401
import torch
if os.environ.get("VLLM_GDN_EAGER_BREAK") == "1":  # r27 arm breakable+gdnbreak (10-14): Qwen GDN as an eager break under breakable graphs
    sys.path.insert(0, os.path.join("/data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27", "patches", "vllm_gdnbreak"))
    import vllm_qwen_gdn_eager_break as _gdnbreak
    print("gdnbreak patched", _gdnbreak.PATCHED, flush=True)
    if not _gdnbreak.PATCHED:
        raise SystemExit("VLLM_GDN_EAGER_BREAK=1 but nothing patched (breakable graphs off?)")
TACTICS = None
if os.environ.get("DRIVE_TACTIC_LOG") == "1":  # r27 kernel identity: FlashInfer AutoTuner choices per (op, input shapes), every call
    from flashinfer.autotuner import AutoTuner as _AT

    TACTICS, _choose = collections.defaultdict(collections.Counter), _AT.choose_one

    def _choose_logged(self, custom_op, runners, tuning_config, inputs, **kw):
        runner, tactic = _choose(self, custom_op, runners, tuning_config, inputs, **kw)
        shapes = tuple(tuple(str(d) for d in t.shape) for t in inputs if isinstance(t, torch.Tensor))
        TACTICS[f"{custom_op} {shapes}"][f"{type(runner).__name__}:{tactic}"] += 1
        return runner, tactic
    _AT.choose_one = _choose_logged
if os.environ.get("INTEG_KEEP_TRACEBACKS"):  # keep the frames of exceptions raised inside the host-trace entry (core drops them), Python entry
    import importlib
    for _m in ("torch.cuda._host_trace", "torch.cuda._host_trace_replay", "torch.cuda._host_trace_redispatch", "torch.cuda._host_trace_tape"):
        try:
            _mod = importlib.import_module(_m)
        except Exception:
            continue
        if hasattr(_mod, "_drop_tracebacks"):
            _mod._drop_tracebacks = lambda *a, **k: None
    import torch.cuda._host_trace as _hte
    _hte._drop_tracebacks.__code__ = (lambda e: None).__code__  # every module's imported binding too
    if os.environ.get("INTEG_RAISE_UNEXPECTED") == "1":  # an unexpected error inside a trace escapes with its traceback
        _hte.raise_unexpected = True
    if hasattr(_hte, "cpp_entry"):
        _hte.cpp_entry = False
if os.environ.get("INTEG_CEXC_PROFILE"):  # diagnosis: every C function that raises when called from Python under compile/inductor code, all threads
    import threading as _th, traceback as _tb2
    _cexc_seen = set()

    def _cexc(frame, event, arg):
        if event != "c_exception":
            return
        f, files = frame, []
        for _ in range(12):
            if f is None:
                break
            files.append(f.f_code.co_filename)
            f = f.f_back
        if globals().get("PHASE", ["init"])[0] != "bench" or any("importlib" in x or "functools" in x for x in files[:2]):
            return
        key = (getattr(arg, "__qualname__", str(arg)), frame.f_code.co_filename, frame.f_lineno)
        if key in _cexc_seen or len(_cexc_seen) >= 60:
            return
        _cexc_seen.add(key)
        print(f"CEXC {key[0]} at {key[1]}:{key[2]}\n" + "".join(_tb2.format_stack(frame, limit=10)), flush=True)
    _th.setprofile_all_threads(_cexc)
    sys.setprofile(_cexc)
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
p.add_argument("--attention-backend", default="auto", help="drive_q35: vLLM attention_backend; auto = vLLM picks (the Qwen3.5-9B recipe sets none)")
p.add_argument("--trust-remote-code", action="store_true", help="drive_q35: the recipe base arg")
p.add_argument("--language-model-only", action="store_true", help="drive_q35: the Qwen3.5 family guide's --language-model-only (text-only, no vision tower)")
p.add_argument("--ix-mixed", nargs="*", default=[], help="drive_ix: InferenceX-shaped mixed steps PxND[@LO-HI]: one P-token prefill chunk + ND decoders at KV LO-HI (default --ix-mixed-kv)")
p.add_argument("--ix-mixed-kv", type=int, nargs=2, default=[4096, 8000], help="decoder prompt lengths spread evenly over [lo, hi]")
p.add_argument("--cb", type=int, nargs="*", default=[], help="continuous batching (10-13): keep C requests in flight; a finished one is replaced by a new one (its prefill lands as a mixed step)")
p.add_argument("--cb-steps", type=int, default=2000, help="measured engine steps per C (after --cb-warm)")
p.add_argument("--cb-warm", type=int, default=300)
p.add_argument("--cb-out", type=int, nargs=2, default=[256, 2048], help="output lengths, seeded uniform over [lo, hi]")
p.add_argument("--cb-prewarm", action="store_true", help="r27: run each C's whole seeded request sequence once untimed first (first-time traces and cuBLAS learns), then the same sequence measured")
p.add_argument("--cb-prompt", type=int, nargs=2, default=[64, 512], help="prompt lengths, seeded uniform over [lo, hi]")
p.add_argument("--host-loop", type=int, nargs=2, default=None, help="r27 host profiling: N single-request prefills of L tokens back to back (args: L N), after a HOSTLOOP_START line, for py-spy")
p.add_argument("--learn-range", type=int, nargs=2, default=None, help="r27 (learn runs): one single-request prefill of every length in [LO, HI] first, so every cuBLAS key of a step of that many tokens is learned (and saved with ARMV_BINDINGS_SAVE)")
p.add_argument("--dcurve-ctx", type=int, nargs="*", default=[], help="r27 decode curves (InferenceX / vllm bench latency convention): context lengths the decoders start at")
p.add_argument("--dcurve-bs", type=int, nargs="*", default=[], help="r27 decode curves: batch sizes (one fill of bs decoders at the context per point)")
p.add_argument("--dcurve-steps", type=int, default=128, help="r27 decode curves: measured decode steps per point (after --dcurve-warm)")
p.add_argument("--dcurve-warm", type=int, default=16)
p.add_argument("--alt-decode", nargs="*", default=[], help="churn diagnostic (10-13): pure decode alternating batch sizes A:B[@LO-HI] every step (scheduler token budget A / B over B decoders)")
p.add_argument("--ix-decode", nargs="*", default=[], help="drive_ix: pure decode BS:LO-HI (decoder prompt lengths spread over [LO, HI])")
p.add_argument("--ix-reps", type=int, default=6, help="measured steps = reps - 2 (the first two are dropped)")
p.add_argument("--prof-mixed", default="", help="drive_prof: torch.profiler trace of the measured mixed steps into this dir")
p.add_argument("--stamps", action="store_true", help="host time of execute_model's sections per step (unsynced inside the step)")
p.add_argument("--stamps-fine", action="store_true", help="with --stamps: also prepare_inputs' and add_requests' callees")
p.add_argument("--out", required=True)
p.add_argument("--llm-kw", default=None, help="drive_fn: JSON dict of extra LLM(...) kwargs (merged last)")
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
FWD_T0 = [0.0, 0.0]  # the outermost timed forward's entry, and its start after the wrapper's synchronize
DEPTH = [0]


def synced(fn, sink):
    # time the outermost call only (run_pw_graph calls the model), never during capture
    def wrapper(*args, **kw):
        # r27 (10-14): no sync inside the unsynced churn window (CBNS): this wrapper synced around run_pw_graph / run_fullgraph, which
        # V2's runner calls once per step in every arm (ours too), so the "unsynced" window had a sync pair per step
        if DEPTH[0] or CBNS[0] or torch.cuda.is_current_stream_capturing():
            return fn(*args, **kw)
        DEPTH[0] += 1
        try:
            FWD_T0[0] = time.perf_counter()  # forward entry, before the wrapper's own synchronize
            torch.cuda.synchronize()
            t = time.perf_counter()
            FWD_T0[1] = t
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
# r27: churn timed without per-step syncs (DRIVE_CB_NOSYNC=1): rows keep their shapes (time 0), the window's wall clock is the metric
CBNS = [False]
# r27 (10-14): torch.cuda.synchronize calls made by this harness while CBNS is on (must stay 0: the churn window is unsynced)
HSYNC = [0]
_hs_orig = torch.cuda.synchronize


def _hs_sync(*a_, **k_):
    if CBNS[0] and sys._getframe(1).f_code.co_filename == __file__:
        HSYNC[0] += 1
    return _hs_orig(*a_, **k_)
torch.cuda.synchronize = _hs_sync
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


if os.environ.get("DRIVE_STEP_WITH_SAMPLE") == "1":  # full step (drive_sweep.py): sample_tokens' time added to the step's row (synced)
    _orig_sample_tokens = GPUModelRunner.sample_tokens

    def timed_sample_tokens(self, *args, **kw):
        if CBNS[0]:
            out = _orig_sample_tokens(self, *args, **kw)
            if LOG and PHASES_ON and isinstance(LOG[-1][-1], Diag):
                LOG[-1][-1].update(_phase_read())
            return out
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = _orig_sample_tokens(self, *args, **kw)
        torch.cuda.synchronize()
        if LOG and not NOSYNC[0]:
            r = LOG[-1]
            LOG[-1] = (*r[:2], r[2] + time.perf_counter() - t, *r[3:])
            if PHASES_ON and isinstance(r[-1], Diag):
                r[-1].update(_phase_read())
        return out
    GPUModelRunner.sample_tokens = timed_sample_tokens
_timed_sync = timed_exec


def timed_exec(self, so, *args, **kw):  # --async-decode: no synchronize around the step
    if CBNS[0] and not kw.get("dummy_run"):
        out = orig_exec(self, so, *args, **kw)
        LOG.append((len(so.num_scheduled_tokens), so.total_num_scheduled_tokens, 0.0, None, sum(v == 1 for v in so.num_scheduled_tokens.values())))
        return out
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
    if PHASES_ON:
        _phase_reset()
    # r27: cuBLAS learns and traces this step took (ours only), so every row set reports learns / traces inside its measured steps
    v = globals().get("V")
    lt0 = (v.provider.harvests, sum(e.traces for e in v.entries.values())) if v is not None else None
    t_step0, idx0 = time.perf_counter(), STEP_IDX[0]
    STEP_IDX[0] += 1
    out = _exec_diag(self, so, *args, **kw)
    lt1 = (v.provider.harvests, sum(e.traces for e in v.entries.values())) if v is not None else None
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
    d["t0"], d["idx"] = t_step0, idx0
    d["counters"] = {k: c1[k] - c0.get(k, 0) for k in c1 if c1[k] != c0.get(k, 0)}
    if lt0 is not None:
        d["learns"], d["traces"] = lt1[0] - lt0[0], lt1[1] - lt0[1]
    if LOG:
        LOG[-1] = LOG[-1] + (d,)
    return out
GPUModelRunner.execute_model = _exec_with_diag

t0 = time.perf_counter()
_T0_WALL = time.time()
extra = {"compilation_config": {"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"}} if a.arm == "fullnc" else {}
if a.arm == "fullnc_full":  # default-nc with FULL graphs for every batch vLLM can graph (compile off)
    extra = {"compilation_config": {"mode": 0, "cudagraph_mode": "FULL"}}
if a.arm in ("Vc", "compnc"):
    extra = {"compilation_config": {"cudagraph_mode": "NONE"}}
if a.max_batched:
    extra["max_num_batched_tokens"] = a.max_batched
if a.max_seqs:
    extra["max_num_seqs"] = a.max_seqs
if a.llm_kw:
    extra.update(json.loads(a.llm_kw))
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
if os.environ.get("ARMV_TRACE_PDL") == "0":  # diagnostics: no PDL in replays (programmatic launches run eagerly, no programmatic edges)
    import torch.cuda._host_trace as _ht
    import torch.cuda._host_trace_capture as _htc

    _ht.trace_pdl = False
    _htc.keep_programmatic_edges = False
if os.environ.get("DIAG_AUTOTUNE") == "1":  # diagnosis: each CachingAutotuner's tuning and every run under / outside a trace with its launcher count
    import collections as _colat
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner as _CA
    from torch.cuda._host_trace_tape import current_trace as _ct_at
    ATLOG = _colat.Counter()
    _orig_a2o = _CA.autotune_to_one_config
    _orig_run_at = _CA.run

    def _a2o(self, *a, **k):
        ATLOG[("autotune_to_one_config", self.fn.__name__, len(self.launchers), _ct_at() is not None)] += 1
        return _orig_a2o(self, *a, **k)

    def _run_at(self, *a, **k):
        ATLOG[("run", self.fn.__name__, len(self.launchers), _ct_at() is not None, id(self) % 100000)] += 1
        return _orig_run_at(self, *a, **k)
    _CA.autotune_to_one_config = _a2o
    _CA.run = _run_at
    import atexit as _ate

    @_ate.register
    def _dump_at():
        for k, n in sorted(ATLOG.items(), key=str)[:60]:
            if "rms_norm_0" in k[1] or k[0] != "run":
                print("DIAG_AT", k, n, flush=True)
if os.environ.get("ARMV_INDUCTOR_HOSTS") == "1":  # Vc: Inductor's C++ wrapper helpers handled for traced tensors (vllm_ports/inductor_hosts.py)
    sys.path.insert(0, os.environ.get("VLLM_PORTS_DIR", "/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration/vllm/vllm_ports"))
    import inductor_hosts

    print("INDUCTOR_HOSTS installed", inductor_hosts.install(), flush=True)
MEMP = None
if os.environ.get("ARMV_MEM_PROBE") == "1":  # diagnostics: peak allocated over the call's start, per HostTraceReplay call kind
    import collections as _col
    import torch.cuda._host_trace_replay as _HR

    MEMP = {"peak_over_start_mib": _col.defaultdict(float), "calls": _col.Counter(), "global_peak_mib": 0.0}
    _orig_call = _HR.HostTraceReplay.__call__

    def _mem_call(self, *args, **kw):
        MEMP["global_peak_mib"] = max(MEMP["global_peak_mib"], torch.cuda.max_memory_allocated() / 2**20)
        m0, t0, r0 = torch.cuda.memory_allocated(), self.traces, self.replays
        torch.cuda.reset_peak_memory_stats()
        tt = time.perf_counter()
        try:
            return _orig_call(self, *args, **kw)
        finally:
            what = "trace" if self.traces > t0 else "replay" if self.replays > r0 else "eager"
            MEMP["calls"][what] += 1
            toks = next((x.shape[0] for x in args if isinstance(x, torch.Tensor) and x.dim() == 1 and x.dtype in (torch.int32, torch.int64)), None)
            MEMP.setdefault("log", []).append((what, toks, round(1e3 * (time.perf_counter() - tt), 2), round((torch.cuda.max_memory_allocated() - m0) / 2**20, 1)))
            MEMP["peak_over_start_mib"][what] = max(MEMP["peak_over_start_mib"][what], (torch.cuda.max_memory_allocated() - m0) / 2**20)
    _HR.HostTraceReplay.__call__ = _mem_call
if os.environ.get("ARMV_CUTE_OBSERVE") == "1":  # before the engine compiles FlashInfer's CuTe DSL kernels (profile run)
    import cutlass  # noqa: F401
    import torch.cuda._host_trace_cute as _cute

    _cute.install()
llm = LLM(model=a.model, tensor_parallel_size=1, enforce_eager=a.arm not in ("default", "fullnc", "fullnc_full", "VAf", "Vc", "compnc"), **({} if a.attention_backend == "auto" else {"attention_backend": a.attention_backend}), **({"language_model_only": True} if a.language_model_only else {}), trust_remote_code=a.trust_remote_code, **extra,
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
_mgr = _runner.cudagraph_manager
R["capture"] = dict(CAP, descs={str(m): len(d) for m, d in getattr(_mgr, "_capture_descs", {}).items()}, graphs=len(getattr(_mgr, "graphs", {})),
                    max_num_seqs=llm.llm_engine.vllm_config.scheduler_config.max_num_seqs,
                    cudagraph_memory_estimate=getattr(llm.llm_engine.model_executor.driver_worker.worker, "cudagraph_memory_estimate", None),
                    breakable_env=os.environ.get("VLLM_USE_BREAKABLE_CUDAGRAPH"))
_cc_ = llm.llm_engine.vllm_config.cache_config
R["capture"]["kv_tokens"] = (_cc_.num_gpu_blocks or 0) * _cc_.block_size  # KV pool in tokens (what graph memory costs shows here)
print("capture", R["capture"], flush=True)
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
# r27 kernel identity (user / coordinator 10-14: every size, every arm and round; DRIVE_KID=0 turns it off): after each point's
# measured window, one extra full step (execute_model through sample_tokens) of that point is profiled (CUDA activity only: kernels,
# graph nodes included, memcpy / memset, graph launches) into R["kid"]["<point>_<reqs>x<tokens>"]; churn: a third, untimed pass of the
# same seeded sequence profiles the first occurrence of every step class of its window. Ours: "path" says replay / trace / eager
# and "clean" is False when the step traced or learned (comparisons use clean replays only).
KID_ON = os.environ.get("DRIVE_KID", "0") == "1"  # off by default (10-14): a profiler session slows every later step of the process (churn rounds 2-3 +3-10% in every arm)
KID_ARM = [None, False]  # point prefix, persistent (churn: every new class) or one step
R_KID = {}
if KID_ON:
    import tempfile
    import torch.profiler as _ktp

    _kcur = [None]
    _kexec, _ksample = GPUModelRunner.execute_model, GPUModelRunner.sample_tokens

    def _kid_v():
        return (sum(e.traces for e in V.entries.values()), V.provider.harvests) if V is not None else None

    def _kid_exec(self, so, *a_, **k_):
        if KID_ARM[0] is not None and not k_.get("dummy_run") and _kcur[0] is None:
            key = f"{KID_ARM[0]}_{len(so.num_scheduled_tokens)}x{so.total_num_scheduled_tokens}"
            if not KID_ARM[1]:
                KID_ARM[0] = None
            if key not in R_KID:
                torch.cuda.synchronize()
                prof = _ktp.profile(activities=[_ktp.ProfilerActivity.CUDA])
                prof.__enter__()
                _kcur[0] = (key, prof, _kid_v())
        return _kexec(self, so, *a_, **k_)

    def _kid_sample(self, *a_, **k_):
        out = _ksample(self, *a_, **k_)
        if _kcur[0] is not None:
            torch.cuda.synchronize()
            key, prof, v0 = _kcur[0]
            _kcur[0] = None
            try:  # never fails the run: an error is recorded in place of the step's kernels
                prof.__exit__(None, None, None)
                fd, tmp = tempfile.mkstemp(suffix=".json", dir="/dev/shm")
                os.close(fd)
                try:
                    prof.export_chrome_trace(tmp)
                    ev = json.load(open(tmp))["traceEvents"]
                finally:
                    os.unlink(tmp)
            except Exception as e_:
                R_KID[key] = {"error": repr(e_)}
                return out
            X = [e for e in ev if e.get("ph") == "X"]
            R_KID[key] = {"kernels": dict(collections.Counter(e["name"] for e in X if e.get("cat") == "kernel")),
                          "shapes": dict(collections.Counter(f'{e["name"]}|{e.get("args", {}).get("grid")}|{e.get("args", {}).get("block")}'
                                                             for e in X if e.get("cat") == "kernel")),
                          "memcpy": sum(e.get("cat") == "gpu_memcpy" for e in X), "memset": sum(e.get("cat") == "gpu_memset" for e in X),
                          "memset_bytes": dict(collections.Counter(str(e.get("args", {}).get("bytes")) for e in X if e.get("cat") == "gpu_memset")),
                          "syncs": dict(collections.Counter(e["name"] for e in X if e.get("cat") in ("cuda_runtime", "cuda_driver") and "ynchronize" in e["name"])),
                          "graph_launches": sum(e.get("cat") in ("cuda_runtime", "cuda_driver") and "GraphLaunch" in e["name"] for e in X),
                          "path": getattr(V, "_path", None) if V is not None else "vllm", "clean": V is None or _kid_v() == v0}
            dg = LOG[-1][-1] if LOG and isinstance(LOG[-1][-1], Diag) else {}
            R_KID[key].update({k: dg.get(k) for k in ("cg_mode", "num_toks", "padded_tokens", "padded_reqs")})
        return out
    GPUModelRunner.execute_model, GPUModelRunner.sample_tokens = _kid_exec, _kid_sample


def kid_step(point, fn):  # one extra profiled step of a point, after its measured window
    if not KID_ON:
        return
    KID_ARM[:] = [point, False]
    try:
        fn()
    finally:
        KID_ARM[:] = [None, False]


PHASE[0] = "bench"
DIGEST = None
if os.environ.get("DRIVE_DIGEST"):  # control (diagnosis): per-row digests of every big model forward (>= 2048 tokens), any arm, no hook
    DIGEST = []
    _dr = llm.llm_engine.model_executor.driver_worker.worker.model_runner
    _dfwd = _dr.model.forward

    def _digest_fwd(input_ids, positions, *args, **kw):
        out = _dfwd(input_ids, positions, *args, **kw)
        if input_ids is not None and input_ids.shape[0] >= 2048 and isinstance(out, torch.Tensor):
            b = out.detach().contiguous().view(out.shape[0], -1).view(torch.int16).to(torch.int64)
            rows = (b * torch.arange(1, b.shape[1] + 1, device=b.device)).sum(dim=1)
            DIGEST.append({"tokens": int(input_ids.shape[0]), "ids": int(input_ids.to(torch.int64).sum()), "pos": int(positions.to(torch.int64).sum()),
                           "rows": rows.tolist()})
        return out
    _dr.model.forward = _digest_fwd
V = None
if ARM_V:
    runner = llm.llm_engine.model_executor.driver_worker.worker.model_runner
    import adapter_cpp  # install_sgl_cpp (python_vllm_cpp.sh): the bound decode entry unless ARMV_BOUND=0
    _ti = time.perf_counter()
    V = (adapter_cpp.ArmVcpp if adapter_cpp.BOUND else armv_adapter.ArmV)(runner)
    V.install()
    R["install_s"] = time.perf_counter() - _ti  # ours: the adapter's install incl. its bindings load (V.bindings_load_s)
    if os.environ.get("DRIVE_ENTRY_FORMS") == "1":  # diagnostic: what distinguishes each new entry key from its kind's first
        _FORMS = R["entry_forms"] = []

        def _kdiff(a, b, path="", out=None):
            out = [] if out is None else out
            if len(out) >= 12:
                return out
            if isinstance(a, tuple) and isinstance(b, tuple):
                if len(a) != len(b):
                    out.append(f"{path}: len {len(a)} vs {len(b)}")
                    return out
                for i, (x, y) in enumerate(zip(a, b)):
                    _kdiff(x, y, f"{path}/{a[i - 1] if i and isinstance(a[i - 1], str) else i}", out)
            elif a != b:
                out.append(f"{path}: {str(a)[:40]} vs {str(b)[:40]}")
            return out

        class _LoggedEntries(dict):
            def __setitem__(self, k, v):
                first = next((x for x in self if isinstance(x, tuple) and isinstance(k, tuple) and x[:2] == k[:2]), None)
                _FORMS.append({"kind": str(k[1]) if isinstance(k, tuple) and len(k) > 1 else str(k)[:30], "n": len(self),
                               "diff_vs_first": _kdiff(first, k) if first is not None else None})
                super().__setitem__(k, v)

        V.entries = _LoggedEntries(V.entries)
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
    cg = collections.Counter(r[-1].get("cg_mode") for r in rows if isinstance(r[-1], Diag))
    if cg:
        out["cg_modes"] = dict(cg)
    pads = [r[-1]["padded_tokens"] - r[-1]["num_toks"] for r in rows if isinstance(r[-1], Diag) and r[-1].get("padded_tokens") is not None and r[-1].get("num_toks") is not None]
    if pads:
        out["pad_tokens_mean"] = statistics.fmean(pads)
    if any(isinstance(r[-1], Diag) and "learns" in r[-1] for r in rows):
        out["learns"] = sum(r[-1].get("learns", 0) for r in rows if isinstance(r[-1], Diag))
        out["traces"] = sum(r[-1].get("traces", 0) for r in rows if isinstance(r[-1], Diag))
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
# learnfast DRIVE_STARTUP=1 (10-15; the 8B lane's fair-startup harness, vllm/bench/drive_diag.py, ported to r27's driver for 27B):
# engine ready = process start -> LLM init (vLLM: incl. its graph capture) + our install with bindings loaded + ours' minimal
# warm-up (a prefill + bs-1 decode; an 8-request batched prefill (seeded 128-2048-token prompts) with bs-8 decodes; a mixed step
# where 4 new requests join the 8 decoders), then a request stream as served (unsynced): 8 requests at once (their TTFT), then up to
# DRIVE_STARTUP_C in flight (seeded prompts 128-2048, outputs 64-512, finished ones replaced) for DRIVE_STARTUP_STEPS steps; per
# step our trace / harvest / binding deltas; steady = after the last step with one (reached if >= 200 quiet steps follow).
# Writes the JSON and exits.
# DRIVE_SPLIT_AB=1 (coordinator 10-16, the split pipeline on 27B; needs torch._C._host_trace_split, step_ms16+): ours with the
# executor-level split off then on, per scenario (pure decode at a few batch sizes, an alternating pair, one mixed shape): per step
# the synced forward wall, the device busy time (kernel durations from a short torch.profiler window), exposed host = wall - busy,
# split section launches and slow_hits (must stay 0 on hits); and each variant's section bounds per run. Writes JSON and exits.
if os.environ.get("DRIVE_SPLIT_AB") == "1" and V is not None:
    import random as _rs
    from torch.profiler import profile as _prof, ProfilerActivity as _PA
    _e, _rg = llm.llm_engine, _rs.Random(5)
    _sched = _e.engine_core.engine_core.scheduler
    _SAB = R["split_ab"] = {"scenarios": {}, "bounds": {}}
    _toks2 = lambda n: TokensPrompt(prompt_token_ids=[_rg.randint(1000, 100000) for _ in range(n)])
    _slow = lambda: sum(getattr(x, "slow_hits", 0) for x in V.entries.values())
    _N, _W, _P = int(os.environ.get("DRIVE_SPLIT_STEPS", "32")), 8, 6

    def _drain_all():
        for rid in list(_e.engine_core.engine_core.scheduler.requests):
            _e.abort_request([rid])
        while _e.has_unfinished_requests():
            _e.step()

    def _decoders(n, ctx=1024):
        for j in range(n):
            _e.add_request(f"sab{n}_{j}_{_rg.random()}", _toks2(ctx), sp(4000))
        while any(r.num_computed_tokens < r.num_prompt_tokens for r in _sched.requests.values()):
            _e.step()

    def _timed(step_fn, i):
        torch.cuda.synchronize()
        t = time.perf_counter()
        step_fn(i)
        torch.cuda.synchronize()
        return (time.perf_counter() - t) * 1e3

    def _measure(step_fn, busy_pass):
        out = _measure1(step_fn, busy_pass)
        print("SPLIT_AB_ROW", busy_pass, json.dumps(out), flush=True)
        json.dump(R, open(a.out, "w"), indent=1, default=str)
        return out

    def _measure1(step_fn, busy_pass):
        for _ in range(_W):
            step_fn(0)
        if busy_pass:  # CUPTI attached from here on: run after every wall measurement (harness pitfall 10-14)
            torch.cuda.synchronize()
            with _prof(activities=[_PA.CUDA]) as pr:
                for i in range(_P):
                    step_fn(i)
                torch.cuda.synchronize()
            busy = sum(ev.device_time for ev in pr.events() if "CUDA" in str(ev.device_type) and ev.device_time > 0) / 1e3 / _P
            if os.environ.get("DRIVE_SPLIT_TRACE"):  # the window's chrome trace (kernels with their graph launch's correlation id)
                _TR[0] += 1
                pr.export_chrome_trace(f"{os.environ['DRIVE_SPLIT_TRACE']}_{_TR[0]}.json")
            return {"busy_ms": round(busy, 3), "trace": _TR[0] if os.environ.get("DRIVE_SPLIT_TRACE") else None}
        _EVT["on"] = True
        for _l in ("first", "repeat", "sets_first", "sets_repeat", "launch_first", "launch_repeat"):
            _EVT[_l].clear()
        rows, per = [], []
        _ctr = lambda: (torch._C._host_trace_graph_launches(), torch._C._host_trace_kernel_node_sets(), torch._C._host_trace_memory_node_sets(),
                        torch._C._host_trace_piece_skips())
        for i in range(_N):
            h0, c0, k0, s0 = _slow(), torch._C._host_trace_split_counts()[0], _ctr(), (_SHIM.ht2_count.value if _SHIM else 0)
            torch.cuda.synchronize()
            torch._C._host_trace_phase_times(True)
            t = time.perf_counter()
            if _NSYS and i == _N - _NSYS:
                torch.cuda.cudart().cudaProfilerStart()
            step_fn(i)
            torch.cuda.synchronize()
            w = (time.perf_counter() - t) * 1e3
            if _NSYS and i == _N - 1:
                torch.cuda.cudart().cudaProfilerStop()
            ph = torch._C._host_trace_phase_times(True)
            k1 = _ctr()
            rows.append((w, _slow() - h0, torch._C._host_trace_split_counts()[0] - c0, *[b - a for a, b in zip(k0, k1)]))
            if _SHIM:  # this step's graph launches: time inside each cudaGraphLaunch, and the host gap before each (patching)
                n1 = _SHIM.ht2_count.value
                ins = [_SHIM.ht2_in[j & 4095] / 1e9 for j in range(s0, n1)]
                outs = [_SHIM.ht2_out[j & 4095] / 1e9 for j in range(s0, n1)]
                per.append({"fwd_entry_us": round((FWD_T0[0] - t) * 1e6, 1), "fwd_sync_us": round((FWD_T0[1] - FWD_T0[0]) * 1e6, 1),
                            "fwd_to_launch_us": round((ins[0] - FWD_T0[1]) * 1e6, 1) if ins else None,
                            "ph_us": {nm: round(x / 1e3, 1) for nm, x in zip(PH_NAMES, ph) if x},
                            "launch_us": [round((o - i_) * 1e6, 1) for i_, o in zip(ins, outs)],
                            "gap_before_us": [round((ins[0] - t) * 1e6, 1)] + [round((ins[j] - outs[j - 1]) * 1e6, 1) for j in range(1, len(ins))]})
        _EVT["on"] = False
        out = {"step_wall_ms_med": statistics.median(r[0] for r in rows), "slow_hits": sum(r[1] for r in rows),
               "split_calls_per_step": statistics.median(r[2] for r in rows), "graph_launches_per_step": statistics.median(r[3] for r in rows),
               "kernel_node_sets_per_step": statistics.median(r[4] for r in rows), "memory_node_sets_per_step": statistics.median(r[5] for r in rows),
               "piece_skips_per_step": statistics.median(r[6] for r in rows), "n": len(rows)}
        for _l in ("first", "repeat"):
            if _EVT[_l]:
                out[f"evt_{_l}_ms_med"] = round(statistics.median(_EVT[_l]), 3)
                out[f"evt_{_l}_ms_minmax"] = (round(min(_EVT[_l]), 3), round(max(_EVT[_l]), 3))
                out[f"evt_{_l}_node_sets_med"] = statistics.median(_EVT["sets_" + _l])
                out[f"evt_{_l}_calls"] = len(_EVT[_l])
                if _EVT["launch_" + _l]:
                    ls = [x for x in _EVT["launch_" + _l] if x]
                    nl = collections.Counter(len(x) for x in ls).most_common(1)[0][0] if ls else 0
                    out[f"launch_span_{_l}_ms_med_by_index"] = [round(statistics.median(x[j] for x in ls if len(x) == nl), 3) for j in range(nl)]
        if per:
            nl = collections.Counter(len(x["launch_us"]) for x in per).most_common(1)[0][0]
            same = [x for x in per if len(x["launch_us"]) == nl]
            out["launches_seen_per_step"] = nl
            out["launch_us_med_by_index"] = [round(statistics.median(x["launch_us"][j] for x in same), 1) for j in range(nl)]
            out["gap_before_us_med_by_index"] = [round(statistics.median(x["gap_before_us"][j] for x in same), 1) for j in range(nl)]
            # step start -> first launch, split: vLLM host work before the forward, the timing wrapper's synchronize, ours (adapter +
            # entry up to the launch), with ours' native phase times (key / bind / evaluate / allocate / patch / launch / dispatch)
            for f in ("fwd_entry_us", "fwd_sync_us", "fwd_to_launch_us"):
                vals = [x[f] for x in same if x[f] is not None]
                out[f + "_med"] = round(statistics.median(vals), 1) if vals else None
            names = sorted({k for x in same for k in x["ph_us"]})
            out["phase_us_med"] = {k: round(statistics.median(x["ph_us"].get(k, 0.0) for x in same), 1) for k in names}
        return out

    _NSYS = int(os.environ.get("DRIVE_SPLIT_NSYS_STEPS", "0"))
    _TR = [0]  # the last N measured steps of each scenario inside a cudaProfiler range
    _SHIM = None
    if any("launchshim2" in x for x in os.environ.get("LD_PRELOAD", "").split(":")):
        import ctypes as _ct
        _lib = _ct.CDLL(next(x for x in os.environ["LD_PRELOAD"].split(":") if "launchshim2" in x))

        class _Sh:
            ht2_in = (_ct.c_int64 * 4096).in_dll(_lib, "ht2_in_ns")
            ht2_out = (_ct.c_int64 * 4096).in_dll(_lib, "ht2_out_ns")
            ht2_count = _ct.c_int64.in_dll(_lib, "ht2_count")
        _SHIM = _Sh
    # DRIVE_SPLIT_EVT=1 (coordinator 10-16, CUPTI-free device span): every HostTraceReplay call during the measured steps is
    # bracketed by CUDA events on the current stream (GPU idle before: synchronize first): device time from the call's first queued
    # work (its graph launch) to completion; with DRIVE_SPLIT_EVT_REPEAT=1 each call is made a second time with the same arguments
    # right after (nothing to patch: kernel node sets counted), its span recorded apart. Diagnostic: the repeat re-runs the step's
    # in-place state updates, so outputs after it are not meaningful.
    _EVT = {"on": False, "first": [], "repeat": [], "sets_first": [], "sets_repeat": [], "launch_first": [], "launch_repeat": []}
    _S3 = None
    if any("launchshim3" in x for x in os.environ.get("LD_PRELOAD", "").split(":")):
        import ctypes as _ct3
        _S3 = _ct3.CDLL(next(x for x in os.environ["LD_PRELOAD"].split(":") if "launchshim3" in x))
        _S3.ht3_elapsed_ms.restype = _ct3.c_float
        _S3.ht3_elapsed_ms.argtypes = [_ct3.c_int64]
        _S3_COUNT = _ct3.c_int64.in_dll(_S3, "ht3_count")
    if os.environ.get("DRIVE_SPLIT_EVT") == "1":
        import torch.cuda._host_trace_replay as _hre
        _orig_call = _hre.HostTraceReplay.__call__

        def _evt_call(self, *a, **k):
            if not _EVT["on"]:
                return _orig_call(self, *a, **k)
            outs = None
            for which in (("first", "repeat") if os.environ.get("DRIVE_SPLIT_EVT_REPEAT") == "1" else ("first",)):
                torch.cuda.synchronize()
                s0 = torch._C._host_trace_kernel_node_sets()
                n0 = _S3_COUNT.value if _S3 else 0
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                o = _orig_call(self, *a, **k)
                e1.record()
                torch.cuda.synchronize()
                _EVT[which].append(e0.elapsed_time(e1))
                if _S3:  # device time from just before each graph launch to its completion (events inside the shim)
                    _EVT["launch_" + which].append([round(_S3.ht3_elapsed_ms(j), 3) for j in range(n0, _S3_COUNT.value)])
                _EVT["sets_" + which].append(torch._C._host_trace_kernel_node_sets() - s0)
                outs = o if outs is None else outs
            return outs

        _hre.HostTraceReplay.__call__ = _evt_call
    _bs = [int(x) for x in os.environ.get("DRIVE_SPLIT_BS", "1,8,32").split(",") if x]
    _alt = tuple(int(x) for x in os.environ.get("DRIVE_SPLIT_ALT", "9:16").split(":"))
    _mix = tuple(int(x) for x in os.environ.get("DRIVE_SPLIT_MIXED", "8x512").split("x"))
    _passes = {"walls": (False,), "busy": (True,)}.get(os.environ.get("DRIVE_SPLIT_PASS", ""), (False, True))
    _modes = os.environ.get("DRIVE_SPLIT_MODES", "off,on").split(",")
    for _busy in _passes:
        for _mi, _mode0 in enumerate(_modes):
            _mode = _mode0 if _modes.count(_mode0) == 1 else f"{_mode0}{_modes[:_mi + 1].count(_mode0)}"  # off1, on1, off2, ...
            torch._C._host_trace_split(_mode0 == "on")
            for b in (_bs if os.environ.get("DRIVE_SPLIT_BS", "x") else []):
                _decoders(b)
                _SAB["scenarios"].setdefault(f"{_mode}/decode{b}", {}).update(_measure(lambda i: _e.step(), _busy))
                _drain_all()
            if os.environ.get("DRIVE_SPLIT_ONLY_DECODE") == "1":
                continue
            A, B = _alt
            _decoders(B)
            b0 = _sched.max_num_scheduled_tokens

            def _alt_step(i):
                _sched.max_num_scheduled_tokens = A if i % 2 == 0 else B
                _e.step()
            try:
                _SAB["scenarios"].setdefault(f"{_mode}/alt{A}_{B}", {}).update(_measure(_alt_step, _busy))
            finally:
                _sched.max_num_scheduled_tokens = b0
            _drain_all()
            nd, pl = _mix
            _decoders(nd)
            _k = [0]

            def _mix_step(i):
                _k[0] += 1
                _e.add_request(f"sabm{_k[0]}_{_busy}", _toks2(pl), sp(1))
                _e.step()
            _SAB["scenarios"].setdefault(f"{_mode}/mixed{nd}x{pl}", {}).update(_measure(_mix_step, _busy))
            _drain_all()
    for _sc in _SAB["scenarios"].values():
        if "step_wall_ms_med" in _sc and "busy_ms" in _sc:
            _sc["exposed_host_ms"] = round(_sc["step_wall_ms_med"] - _sc["busy_ms"], 3)
    print("SPLIT_AB", json.dumps(_SAB["scenarios"]), flush=True)
    torch._C._host_trace_split(False)
    for _k2, _ent in V.entries.items():
        for vi, v in enumerate(getattr(_ent, "variants", [])):
            try:
                runs = sum(isinstance(st_, range) for st_ in v.captured.lowered.steps)
                _SAB["bounds"][f"{_k2[1] if isinstance(_k2, tuple) else _k2}/v{vi}"] = [list(torch._C._host_trace_split_bounds(v.native, g, 0)) for g in range(runs)]
            except Exception as _x:
                _SAB["bounds"][f"{_k2[1]}/v{vi}"] = repr(_x)[:120]
    json.dump(R, open(a.out, "w"), indent=1, default=str)
    import atexit as _axa
    _axa._run_exitfuncs()
    os._exit(0)
if os.environ.get("DRIVE_STARTUP") == "1":
    import random as _rnd
    ST = R["startup"] = {}
    _stat = open("/proc/self/stat").read().rsplit(")", 1)[1].split()
    _btime = float(next(ln.split()[1] for ln in open("/proc/stat") if ln.startswith("btime")))
    _pstart = _btime + int(_stat[19]) / os.sysconf("SC_CLK_TCK")
    ST["init_s"] = R["init_s"]
    ST["install_s"] = R.get("install_s", 0.0)
    ST["pre_llm_s"] = _T0_WALL - _pstart  # process start -> LLM() start (imports, CuTe observe install)
    _tw = time.perf_counter()
    # (traces, harvests, bindings, redispatches); steady ignores bindings (first use of a preloaded, already-learned key binds it)
    _sctr = (lambda: (sum(getattr(e, "traces", 0) for e in V.entries.values()), V.provider.harvests, len(V.provider.bindings),
                      sum(getattr(e, "redispatches", 0) for e in V.entries.values()))) if V is not None else (lambda: (0, 0, 0, 0))
    # DRIVE_RANGE_WARM=1 (the user's range warm-up, 10-15): before engine ready, drive the size classes the server can see, per
    # entry form: decode bs over bucket edges up to max_num_seqs (each also a batched prefill of bs short prompts), single- and
    # two-request prefills over token-count edges up to max_num_batched_tokens, and mixed steps (8 decoders + one joining prompt)
    # over the same edges. Per phase: wall, our counters' deltas and diag27's trace / redispatch / learn seconds.
    if (V is not None or os.environ.get("DRIVE_RANGE_ANY") == "1") and os.environ.get("DRIVE_RANGE_WARM") == "1":
        _sc = llm.llm_engine.vllm_config.scheduler_config
        _mt, _ms = int(os.environ.get("DRIVE_RANGE_MAX_TOKENS", _sc.max_num_batched_tokens)), int(os.environ.get("DRIVE_RANGE_MAX_SEQS", _sc.max_num_seqs))
        _mt = min(_mt, llm.llm_engine.vllm_config.model_config.max_model_len - 8)
        _dg = sys.modules.get("diag27_live")

        def _edges(hi, lo=1):
            pts = {x for k in range(hi.bit_length() + 1) for x in (1 << k, (1 << k) + 1) if lo <= x <= hi}
            return sorted(pts | {hi})
        _rw, _eng0, _ph = _rnd.Random(13), llm.llm_engine, {}
        _toks = lambda n: TokensPrompt(prompt_token_ids=[_rw.randint(1000, 100000) for _ in range(n)])

        def _drain():
            while _eng0.has_unfinished_requests():
                _eng0.step()

        def _phase(name, fn):
            snap = os.environ.get("DRIVE_MEM_SNAPSHOT") if os.environ.get("DRIVE_MEM_SNAPSHOT_PHASE") == name else None
            if snap:  # diagnostic: the allocator's history over this phase, dumped at its end (torch.cuda.memory snapshot)
                torch.cuda.memory._record_memory_history(max_entries=int(os.environ.get("DRIVE_MEM_SNAPSHOT_N", "300000")))
            torch.cuda.reset_peak_memory_stats()
            t, c = time.perf_counter(), _sctr()
            _tp = sys.modules.get("tpl_layer") or sys.modules.get("tpl_closure")
            s0 = dict(_tp.STATS) if _tp else {}
            a0 = dict(_dg.ACC) if _dg else {}
            n0 = dict(_dg.CNT) if _dg else {}
            fn()
            torch.cuda.synchronize()
            _ph[name] = {"s": round(time.perf_counter() - t, 2), "events": [x - y for x, y in zip(_sctr(), c)],
                         "max_alloc_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2), "alloc_gib": round(torch.cuda.memory_allocated() / 2**30, 2),
                         "max_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2), "reserved_gib": round(torch.cuda.memory_reserved() / 2**30, 2)}
            if _dg:
                _ph[name]["acc"] = {k: round(v - a0[k], 2) for k, v in _dg.ACC.items() if v - a0[k] > 0.005}
                _ph[name]["cnt"] = {k: v - n0[k] for k, v in _dg.CNT.items() if v - n0[k]}
            if sys.modules.get("refusal_memo"):
                _ph[name]["refusal_memo"] = dict(sys.modules["refusal_memo"].STATS)
            if _tp:
                _ph[name]["tpl"] = {k: round(v - s0.get(k, 0), 2) for k, v in _tp.STATS.items() if v != s0.get(k, 0) and not k.startswith("diffs_shown")}
            if snap:
                torch.cuda.memory._dump_snapshot(snap)
                torch.cuda.memory._record_memory_history(enabled=None)
            print("RANGE_WARM", name, json.dumps(_ph[name]), flush=True)

        def _decode():
            for b in _edges(_ms):
                for j in range(b):
                    _eng0.add_request(f"rwd{b}_{j}", _toks(16), sp(3))
                _drain()

        def _prefill(nreq):
            def f():
                for T in _edges(_mt, lo=2 * nreq):
                    for j in range(nreq):
                        _eng0.add_request(f"rwp{nreq}_{T}_{j}", _toks(T // nreq), sp(1))
                    _drain()
            return f

        def _mixed():
            for T in _edges(_mt - 8, lo=2):
                for j in range(8):
                    _eng0.add_request(f"rwm{T}_{j}", _toks(16), sp(1000))
                for _ in range(3):
                    _eng0.step()
                _eng0.add_request(f"rwm{T}_new", _toks(T), sp(2))
                _eng0.step()
                _eng0.step()
                _eng0.abort_request([f"rwm{T}_{j}" for j in range(8)])
                _drain()
        # derived from the class boundaries the failing guards record (fg2, 10-15): attention decode's split count
        # min(ceil(L / 512), ...) over the batch's longest context L; mixed steps over mm_fp4's ceil(M / 512) tile classes, its
        # pow2 buckets, GDN's prefill-token residue mod 16, its decode-count (1 / more) and prefill-count (1 / 2 / more) equalities
        _ctx = int(os.environ.get("DRIVE_RANGE_MAX_CTX", "4096"))

        def _attn():
            for k in range(0, -(-_ctx // 512) + 1):
                L = max(1, 512 * k + 1)
                for bs in (1, 8, 32):
                    _eng0.add_request(f"rwa{k}_{bs}_long", _toks(L), sp(3))
                    for j in range(bs - 1):
                        _eng0.add_request(f"rwa{k}_{bs}_{j}", _toks(16), sp(3))
                    _drain()

        def _mixedx():
            for k in range(1, int(os.environ.get("DRIVE_RANGE_MIXED_TILES", "16")) + 1):
                d, npf = (1, 8, 32)[k % 3], (1, 2, 3)[k % 3]
                p = max(npf * 2, 512 * k - d - (3 if k % 2 else 0))
                for j in range(d):
                    _eng0.add_request(f"rwx{k}_{j}", _toks(16), sp(1000))
                for _ in range(3):
                    _eng0.step()
                for j in range(npf):
                    _eng0.add_request(f"rwx{k}_new{j}", _toks(p // npf + (p % npf if j == 0 else 0)), sp(2))
                _eng0.step()
                _eng0.step()
                _eng0.abort_request([f"rwx{k}_{j}" for j in range(d)])
                _drain()
        def _psweep():
            # memory diagnostic: one single-request prefill per T, its own peak; the snapshot (if any) around the largest
            _ph["psweep_peaks"] = {}
            Ts = [int(x) for x in os.environ.get("DRIVE_PSWEEP_T", f"1024,4096,{_mt}").split(",")]
            for T in Ts:
                torch.cuda.reset_peak_memory_stats()
                snap = os.environ.get("DRIVE_MEM_SNAPSHOT") if T == Ts[-1] and os.environ.get("DRIVE_MEM_SNAPSHOT_PHASE") == "psweep_last" else None
                if snap:
                    torch.cuda.memory._record_memory_history(max_entries=int(os.environ.get("DRIVE_MEM_SNAPSHOT_N", "300000")))
                _eng0.add_request(f"rwps{T}", _toks(T), sp(1))
                _drain()
                torch.cuda.synchronize()
                if snap:
                    torch.cuda.memory._dump_snapshot(snap)
                    torch.cuda.memory._record_memory_history(enabled=None)
                _ph["psweep_peaks"][T] = (round(torch.cuda.max_memory_allocated() / 2**30, 2), round(torch.cuda.max_memory_reserved() / 2**30, 2))
            print("PSWEEP", _ph["psweep_peaks"], flush=True)
        ST["range_warm"] = {"max_tokens": _mt, "max_seqs": _ms, "decode_bs": _edges(_ms), "token_edges": _edges(_mt), "max_ctx": _ctx}
        for _n in os.environ.get("DRIVE_RANGE_PHASES", "decode,prefill1,prefill2,mixed").split(","):
            _phase(_n, {"decode": _decode, "prefill1": _prefill(1), "prefill2": _prefill(2), "mixed": _mixed, "attn": _attn, "mixedx": _mixedx, "psweep": _psweep}[_n])
        ST["range_warm"]["phases"] = _ph
        if V is not None and os.environ.get("ARMV_GC_FREEZE") == "1" and hasattr(V, "freeze_after_warmup"):
            V.freeze_after_warmup()
    elif V is not None and os.environ.get("DRIVE_MIN_WARM", "1") == "1":
        llm.generate(prompts(1, 64), sp(3), use_tqdm=False)
        _rw = _rnd.Random(11)
        for j in range(8):
            llm.llm_engine.add_request(f"mwb{j}", TokensPrompt(prompt_token_ids=[_rw.randint(1000, 100000) for _ in range(_rw.randint(128, 2048))]), sp(8))
        for _ in range(2):
            llm.llm_engine.step()
        for j in range(4):
            llm.llm_engine.add_request(f"mwb3_{j}", TokensPrompt(prompt_token_ids=[_rw.randint(1000, 100000) for _ in range(_rw.randint(128, 2048))]), sp(4))
        while llm.llm_engine.has_unfinished_requests():
            llm.llm_engine.step()
        if os.environ.get("ARMV_GC_FREEZE") == "1" and hasattr(V, "freeze_after_warmup"):
            V.freeze_after_warmup()
    ST["min_warm_s"] = time.perf_counter() - _tw
    ST["engine_ready_s"] = time.time() - _pstart
    ST["ctr_at_ready"] = _sctr()
    if _dga := sys.modules.get("diag27_live"):
        ST["acc_at_ready"] = {k: round(v, 2) for k, v in _dga.ACC.items()}
        ST["cnt_at_ready"] = dict(_dga.CNT)
    _eng, _rg = llm.llm_engine, _rnd.Random(7)
    _added, _first, _fin = {}, {}, set()
    _nxt = [0]

    def _sadd():
        i = _nxt[0]
        _nxt[0] += 1
        L, n = _rg.randint(128, 2048), _rg.randint(64, 512)
        _eng.add_request(f"st{i}", TokensPrompt(prompt_token_ids=[_rg.randint(1000, 100000) for _ in range(L)]), sp(n))
        _added[f"st{i}"] = time.perf_counter()
    _C, _N = int(os.environ.get("DRIVE_STARTUP_C", "32")), int(os.environ.get("DRIVE_STARTUP_STEPS", "800"))
    torch.cuda.synchronize()
    ST["alloc_at_ready_gib"] = round(torch.cuda.memory_allocated() / 2**30, 1)
    torch.cuda.reset_peak_memory_stats()
    NOSYNC[0] = True
    _t0 = time.perf_counter()
    for _ in range(8):
        _sadd()
    _steps, _cprev, _gen, _gcum = [], _sctr(), {}, []
    for k in range(_N):
        outs = _eng.step()
        now = time.perf_counter()
        for o in outs:
            if o.outputs:  # cumulative outputs (SamplingParams' default output kind)
                _gen[o.request_id] = max(_gen.get(o.request_id, 0), len(o.outputs[0].token_ids))
            if o.request_id not in _first and o.outputs and len(o.outputs[0].token_ids) > 0:
                _first[o.request_id] = now
            if o.finished:
                _fin.add(o.request_id)
        c = _sctr()
        _steps.append([round(now - _t0, 4), [x - y for x, y in zip(c, _cprev)]])
        _gcum.append(sum(_gen.values()))
        _cprev = c
        if all(f"st{i}" in _first for i in range(8)):
            for _ in range(min(4, _C - (_nxt[0] - len(_fin)))):
                _sadd()
    torch.cuda.synchronize()
    NOSYNC[0] = False
    busy = [i for i, (_, d) in enumerate(_steps) if d[0] or d[1] or d[3]]
    last = busy[-1] if busy else None
    ST["ttft_first8_ms"] = [round((_first[f"st{i}"] - _added[f"st{i}"]) * 1e3, 1) if f"st{i}" in _first else None for i in range(8)]
    ST["steps"] = len(_steps)
    ST["stream_max_alloc_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    ST["stream_max_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 2**30, 2)
    ST["steps_with_trace_or_learn"] = len(busy)
    ST["events_total"] = [sum(d[j] for _, d in _steps) for j in range(4)]
    ST["events_fields"] = ["traces", "harvests", "bindings", "redispatches"]
    _wall = _steps[-1][0]
    ST["gen_tokens"] = _gcum[-1]
    ST["stream_tok_s"] = round(_gcum[-1] / _wall, 1)
    _k0 = last + 1 if last is not None else 0  # throughput over the quiet tail (after the last trace / harvest / redispatch)
    ST["steady_tok_s"] = round((_gcum[-1] - _gcum[_k0 - 1]) / (_wall - _steps[_k0 - 1][0]), 1) if 0 < _k0 < len(_steps) - 1 else (ST["stream_tok_s"] if _k0 == 0 else None)
    if _dg_on := sys.modules.get("diag27_live"):
        ST["acc_at_end"] = {k: round(v, 2) for k, v in _dg_on.ACC.items()}
        ST["cnt_at_end"] = dict(_dg_on.CNT)
    ST["steady_after_ready_s"] = _steps[last][0] if last is not None else 0.0
    ST["quiet_steps_after_last_event"] = len(_steps) - 1 - last if last is not None else len(_steps)
    ST["steady_reached"] = ST["quiet_steps_after_last_event"] >= 200
    ST["ready_to_steady_s"] = ST["engine_ready_s"] + ST["steady_after_ready_s"]
    ST["stream_wall_s"] = time.perf_counter() - _t0
    ST["requests_added"], ST["requests_finished"] = _nxt[0], len(_fin)
    ST["event_steps"] = [[i, _steps[i][0], _steps[i][1]] for i in busy[:100]]
    # memset lane's split-pipeline report (10-16, CPU only: reads the lowered tapes): every entry the adapter holds, after the
    # stream has replayed them; on with DRIVE_CUT_REPORT=1 or the marker file learnfast/cut_report.on
    if V is not None and (os.environ.get("DRIVE_CUT_REPORT") == "1" or os.path.exists("/data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast/cut_report.on")):
        try:
            sys.path.insert(0, "/data/eellison/src/pytorch/agent_space/paramgraph/land/core/memset/split")
            from cut_report import cut_report as _cut_report
            _tag = os.path.basename(a.out).replace(".json", "")
            R["cut_reports"] = []
            for _i, (_k, _e) in enumerate(V.entries.items()):
                _name = f"{_tag}_{_k[1] if isinstance(_k, tuple) and len(_k) > 1 else 'entry'}{_i}"
                _path = f"/data/eellison/src/pytorch/agent_space/paramgraph/land/core/memset/split/out/27b_{_name}.json"
                json.dump(_cut_report(_e), open(_path, "w"), default=str)
                R["cut_reports"].append(_path)
            print("CUT_REPORTS", R["cut_reports"], flush=True)
        except Exception as _e:
            R["cut_report_error"] = repr(_e)[:300]
            print("CUT_REPORT_ERROR", repr(_e)[:300], flush=True)
    if V is not None:
        try:
            R["V"] = V.summary()
        except Exception as _e:
            R["V_summary_error"] = repr(_e)[:200]
    json.dump(R, open(a.out, "w"), indent=1, default=str)
    print("STARTUP", json.dumps({k: v for k, v in ST.items() if k != "event_steps"}, default=str), flush=True)
    sys.stdout.flush()
    import atexit as _axs
    _axs._run_exitfuncs()
    os._exit(0)
llm.generate(prompts(4, 128), sp(8), use_tqdm=False)  # warm-up
if V is not None and os.environ.get("ARMV_GC_FREEZE") == "1" and hasattr(V, "freeze_after_warmup"):
    V.freeze_after_warmup()  # ours: gc.freeze() after our warm-up, as vLLM's freeze_gc_heap() after its own
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
    kid_step(f"p{L}", lambda: llm.generate(prompts(1, L), sp(1), use_tqdm=False))
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
    eng.add_request(f"{tag}_pk", kv_prompt(P), sp(1))
    kid_step(f"m{P}x{nd}", eng.step)
    eng.step()
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
    psi0, tw0 = _psi_full_us(), time.perf_counter()
    try:
        for k in range(2 * (a.ix_reps + 14)):
            sched.max_num_scheduled_tokens = A if k % 2 == 0 else B
            eng.step()
        for x in (A, B):
            sched.max_num_scheduled_tokens = x
            kid_step(f"a{A}_{B}", eng.step)
    finally:
        sched.max_num_scheduled_tokens = budget0
    psi1, tw1 = _psi_full_us(), time.perf_counter()
    rows = [r for r in LOG if r[0] == r[1] and r[0] in (A, B)]
    R["ix"][f"alt_{A}_{B}"] = (summ(rows[8:]) if len(rows) > 8 else {"step_ms": None}) | {"n_rows": len(rows), "shapes_seq": [r[0] for r in rows[:12]],
                                                                                     "kv_min_med_max": _ix_kv(rows)}
    for x in (A, B):
        sub = [r for r in rows[8:] if r[0] == x]
        if sub:
            R["ix"][f"alt_{A}_{B}"][f"bs{x}"] = summ(sub)
    R["ix"][f"alt_{A}_{B}"]["stalls"] = step_stats(rows[8:], tw1 - tw0, (psi1 - psi0) if psi0 is not None and psi1 is not None else None)
    _ix_drain(tag, max(A, B))
    print("ix alt", A, B, {k: v for k, v in R["ix"][f"alt_{A}_{B}"].items() if k != "stalls"}, flush=True)
if a.host_loop:  # host profiling: a steady stream of one-request prefill steps (py-spy attaches after HOSTLOOP_START)
    eng, (L_, N_) = llm.llm_engine, a.host_loop
    for i in range(5):  # warm
        eng.add_request(f"hlw{i}", kv_prompt(L_), sp(1))
        while eng.has_unfinished_requests():
            eng.step()
    print("HOSTLOOP_START", os.getpid(), flush=True)
    LOG.clear()
    t_hl = time.perf_counter()
    for i in range(N_):
        eng.add_request(f"hl{i}", kv_prompt(L_), sp(1))
        while eng.has_unfinished_requests():
            eng.step()
    R["host_loop"] = {"L": L_, "N": N_, "wall_s": time.perf_counter() - t_hl, "steps": len(LOG),
                      "step_ms": statistics.median(r[2] * 1e3 for r in LOG) if LOG else None}
    print("HOSTLOOP_END", R["host_loop"], flush=True)
if a.learn_range:  # every step size in the range once: a linear layer's cuBLAS key is the step's token count, whatever its mix
    t_lr, eng = time.perf_counter(), llm.llm_engine
    for i, M in enumerate(range(a.learn_range[0], a.learn_range[1] + 1)):
        eng.add_request(f"lr{i}", kv_prompt(M), sp(1))
        while eng.has_unfinished_requests():
            eng.step()
        if M % 100 == 0:
            print(f"learn_range at {M}: {time.perf_counter() - t_lr:.0f} s, harvests {V.provider.harvests if V is not None else None}, "
                  f"traces {sum(e.traces for e in V.entries.values()) if V is not None else None}", flush=True)
    R["learn_range"] = {"range": a.learn_range, "wall_s": time.perf_counter() - t_lr, "harvests": V.provider.harvests if V is not None else None}
    print("learn_range", R["learn_range"], flush=True)
R["dcurve"] = {}
for ctx in a.dcurve_ctx:  # decode curves: bs decoders filled at ctx prompt tokens, then warm + measured pure-decode steps (TPOT = step time)
    for bs in a.dcurve_bs:
        tag, eng = f"dc{ctx}_{bs}", llm.llm_engine
        _ix_fill(tag, [ctx] * bs)
        LOG.clear()
        psi0, tw0 = _psi_full_us(), time.perf_counter()
        for _ in range(a.dcurve_warm + a.dcurve_steps):
            eng.step()
        psi1, tw1 = _psi_full_us(), time.perf_counter()
        rows = [r for r in LOG if r[0] == r[1] == bs]
        meas = rows[a.dcurve_warm:]
        kid_step(f"d{ctx}", eng.step)
        R["dcurve"][f"{ctx}_{bs}"] = (summ(meas) if meas else {"step_ms": None}) | {
            "mean_ms": statistics.fmean(r[2] * 1e3 for r in meas) if meas else None, "n_rows": len(rows), "kv_min_med_max": _ix_kv(meas)}
        # the PSI / wall window spans the point's warm + measured steps
        R["dcurve"][f"{ctx}_{bs}"]["stalls"] = step_stats(meas, tw1 - tw0, (psi1 - psi0) if psi0 is not None and psi1 is not None else None)
        _ix_drain(tag, bs)
        _st = R["dcurve"][f"{ctx}_{bs}"].get("stalls", {})
        print("dcurve", ctx, bs, {k: v for k, v in R["dcurve"][f"{ctx}_{bs}"].items() if k != "stalls"},
              {k: _st.get(k) for k in ("median_ms", "max_ms", "p99_ms", "n_10x", "gc_ms_in_window", "flag")}, flush=True)
R["cb"] = {}
for C, cb_pass in [(C, p_) for C in a.cb for p_ in ((0, 1) if a.cb_prewarm else (1,))]:  # pass 0: the sequence untimed (--cb-prewarm)
    # continuous batching: C in flight, seeded output / prompt lengths, the same request sequence in every arm (and in both passes)
    eng, rng = llm.llm_engine, torch.Generator().manual_seed(1000 + C)
    nxt = [0]
    pfx = f"cb{C}w" if cb_pass == 0 else f"cb{C}"

    def admit():
        i = nxt[0]
        nxt[0] += 1
        L = int(torch.randint(a.cb_prompt[0], a.cb_prompt[1] + 1, (1,), generator=rng))
        n = int(torch.randint(a.cb_out[0], a.cb_out[1] + 1, (1,), generator=rng))
        ids = torch.randint(1000, 100000, (L,), generator=rng).tolist()
        eng.add_request(f"{pfx}_{i}", TokensPrompt(prompt_token_ids=ids), sp(n))
    for _ in range(C):
        admit()
    LOG.clear()
    done = 0
    t_win = toks = 0
    for k in range(a.cb_warm + a.cb_steps):
        if k == a.cb_warm:
            LOG.clear()
            t0w, done0 = time.perf_counter(), done
            v0 = (V.provider.harvests, sum(e.traces for e in V.entries.values())) if V is not None else None
            t_win0 = time.perf_counter()
            CBNS[0] = cb_pass == 1 and os.environ.get("DRIVE_CB_NOSYNC") == "1"
            HSYNC[0] = 0
            if cb_pass == 1 and os.environ.get("DRIVE_SYNC_PROF") == "1":  # r27 check (10-14): sync API calls inside the churn window
                import torch.profiler as _stp
                _sprof = _stp.profile(activities=[_stp.ProfilerActivity.CPU, _stp.ProfilerActivity.CUDA])
                _sprof.__enter__()
                # the Python call sites of torch.cuda.synchronize inside the window (a sync's origin)
                import traceback as _tb
                _SYNC_SITES, _orig_sync = collections.Counter(), torch.cuda.synchronize

                def _sync_counted(*a_, **k_):
                    st_ = _tb.extract_stack(limit=4)[:-1]
                    _SYNC_SITES[" <- ".join(f"{os.path.basename(f.filename)}:{f.lineno}" for f in reversed(st_))] += 1
                    return _orig_sync(*a_, **k_)
                torch.cuda.synchronize = _sync_counted
        outs = eng.step()
        fin = sum(1 for o in outs if o.finished)
        done += fin
        for _ in range(fin):
            admit()
    nosync, CBNS[0] = CBNS[0], False
    torch.cuda.synchronize()  # the window's one end barrier (its wall clock includes the queued GPU work); not a per-step sync
    sync_counts = None
    if cb_pass == 1 and os.environ.get("DRIVE_SYNC_PROF") == "1":
        import tempfile
        torch.cuda.synchronize = _orig_sync
        _sprof.__exit__(None, None, None)
        fd_, tmp_ = tempfile.mkstemp(suffix=".json", dir="/dev/shm")
        os.close(fd_)
        try:
            _sprof.export_chrome_trace(tmp_)
            _sev = json.load(open(tmp_))["traceEvents"]
        finally:
            os.unlink(tmp_)
        sync_counts = dict(collections.Counter(e["name"] for e in _sev if e.get("ph") == "X" and e.get("cat") in ("cuda_runtime", "cuda_driver")
                                               and "ynchronize" in e["name"]))
    if cb_pass == 0:
        eng.abort_request([f"{pfx}_{i}" for i in range(nxt[0])])
        while eng.has_unfinished_requests():
            eng.step()
        print("cb", C, "prewarm pass done", flush=True)
        continue
    wall = time.perf_counter() - t0w
    rows = [r for r in LOG if r[0] > 0]
    # tokens produced per step: one per scheduled request whose prompt is complete after the step (decodes, plus a prefill's last chunk)
    out_tok = sum(r[0] for r in rows)
    dec = [r for r in rows if r[0] == r[1]]
    mixed = [r for r in rows if r[1] > r[0]]
    changed = sum(1 for p_, q_ in zip(rows, rows[1:]) if (p_[0], p_[1]) != (q_[0], q_[1]))
    dchanged = sum(1 for p_, q_ in zip(dec, dec[1:]) if p_[0] != q_[0])
    step_sum = sum(r[2] for r in rows)
    d = {"C": C, "steps": len(rows), "wall_s": wall, "out_tokens_approx": out_tok, "wall_ms_per_token": wall * 1e3 / max(out_tok, 1),
         "step_ms_per_token": step_sum * 1e3 / max(out_tok, 1), "step_ms_mean": step_sum * 1e3 / max(len(rows), 1), "finished": done - done0,
         "mixed_steps": len(mixed), "mixed_frac": len(mixed) / max(len(rows), 1), "shape_changed_frac": changed / max(len(rows) - 1, 1),
         "decode_bs_changed_frac": dchanged / max(len(dec) - 1, 1), "decode_steps": len(dec),
         "decode": summ(dec) if dec else None, "mixed": summ(mixed) if mixed else None}
    d["nosync"] = nosync  # step times are 0 in an unsynced window: only the wall clock and the shapes count
    d["harness_syncs_in_window"] = HSYNC[0] if nosync else None
    if sync_counts is not None:
        d["sync_counts"], d["sync_per_step"] = sync_counts, {k: v / max(len(rows), 1) for k, v in sync_counts.items()}
        d["sync_sites"] = dict(_SYNC_SITES)
    pads = [r[-1]["padded_tokens"] - r[-1]["num_toks"] for r in rows if isinstance(r[-1], Diag) and r[-1].get("padded_tokens") is not None and r[-1].get("num_toks") is not None]
    d["pad_tokens_mean"] = statistics.fmean(pads) if pads else None
    d["cg_modes"] = dict(collections.Counter(r[-1].get("cg_mode") for r in rows if isinstance(r[-1], Diag)))
    phs = [r[-1] for r in rows if isinstance(r[-1], Diag) and "patch" in r[-1]]
    if phs:
        d["nodes_set_mean"] = statistics.fmean(x["patch"][1] for x in phs)
        d["mem_nodes_set_mean"] = statistics.fmean(x["patch"][2] for x in phs)
        d["patch_us_mean"] = statistics.fmean(x["ph_us"].get("patch", 0.0) for x in phs)
        d["entry_us_mean"] = statistics.fmean(x["ph_us"].get("dispatch", 0.0) for x in phs)
    if V is not None:  # ours: cuBLAS keys learned and traces taken inside the measured window
        d["learns_in_window"] = V.provider.harvests - v0[0]
        d["traces_in_window"] = sum(e.traces for e in V.entries.values()) - v0[1]
        d["learned_in_window"] = [x for x in getattr(V, "learn_log", []) if x[0] >= t_win0]
    eng.abort_request([f"cb{C}_{i}" for i in range(nxt[0])])
    while eng.has_unfinished_requests():
        eng.step()
    if KID_ON:  # kernel identity: the same seeded sequence once more, untimed; every step class of its window profiled once
        win_cls = collections.Counter((r[0], r[1]) for r in rows)
        rng, nxt, pfx = torch.Generator().manual_seed(1000 + C), [0], f"cb{C}k"
        t_k = time.perf_counter()
        try:
            for _ in range(C):
                admit()
            for k in range(a.cb_warm + a.cb_steps):
                if k == a.cb_warm:
                    KID_ARM[:] = [f"cb{C}", True]
                outs = eng.step()
                for _ in range(sum(1 for o in outs if o.finished)):
                    admit()
        except Exception as e_:
            d["kid_error"] = repr(e_)
        KID_ARM[:] = [None, False]
        eng.abort_request([f"{pfx}_{i}" for i in range(nxt[0])])
        while eng.has_unfinished_requests():
            eng.step()
        d["kid_pass_s"] = time.perf_counter() - t_k
        got = {tuple(map(int, k.rsplit("_", 1)[1].split("x"))) for k in R_KID if k.startswith(f"cb{C}_")}
        d["kid_classes"] = len(win_cls)
        d["kid_window_steps_covered"] = sum(n for c, n in win_cls.items() if c in got) / max(sum(win_cls.values()), 1)
    R["cb"][str(C)] = d
    print("cb", C, {k: v for k, v in d.items() if k not in ("decode", "mixed")}, flush=True)
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
if MEMP is not None:
    R["mem_probe"] = MEMP
if DIGEST is not None:
    json.dump(DIGEST, open(os.environ["DRIVE_DIGEST"], "w"))
if os.environ.get("DRIVE_NAN_PROBE") == "1":  # r27 (FlashInfer GDN, 10-13): after the measurements, is this process one whose GDN prefill
    # NaNs? one-request prefills of 64 / 512 / 2048 tokens, the sampled token's logprob (NaN logits give a NaN logprob)
    import math
    R["nan_probe"] = {}
    for L in (64, 512, 2048):
        o = llm.generate(prompts(1, L), SamplingParams(max_tokens=1, temperature=0.0, logprobs=1), use_tqdm=False)[0].outputs[0]
        lp = o.logprobs[0][o.token_ids[0]].logprob if o.logprobs else None
        R["nan_probe"][L] = lp is None or math.isnan(lp)
    print("nan_probe", R["nan_probe"], flush=True)
if os.environ.get("DRIVE_LOGIT_PROBE"):  # r27 (10-14, breakable GDN check): one-request prefills of the listed lengths, the same seeded
    # prompt per length in every arm; the step's last-row logits (fp32) saved to DRIVE_LOGIT_PROBE_DIR/<tag>_<L>.pt (vLLM arms)
    _lp_dir, _lp_cur, _lp_cl = os.environ["DRIVE_LOGIT_PROBE_DIR"], [None], _runner.model.compute_logits
    os.makedirs(_lp_dir, exist_ok=True)

    def _lp_compute_logits(h, *a_, **k_):
        out = _lp_cl(h, *a_, **k_)
        if _lp_cur[0] is not None:
            torch.save(out[-1].float().cpu(), os.path.join(_lp_dir, f"{os.environ.get('DRIVE_LOGIT_PROBE_TAG', a.arm)}_{_lp_cur[0]}.pt"))
            _lp_cur[0] = None
        return out
    _runner.model.compute_logits = _lp_compute_logits
    for L in map(int, os.environ["DRIVE_LOGIT_PROBE"].split()):
        _lp_cur[0] = L
        ids = torch.randint(1000, 100000, (L,), generator=torch.Generator().manual_seed(4242 + L)).tolist()
        llm.generate([TokensPrompt(prompt_token_ids=ids)], SamplingParams(max_tokens=1, temperature=0.0), use_tqdm=False)
    _runner.model.compute_logits = _lp_cl
    print("logit_probe saved", os.environ["DRIVE_LOGIT_PROBE"], flush=True)
if os.environ.get("DRIVE_DECODE_PROBE"):  # r27 (10-15, after Flash-Next's dp_brk): greedy decode of BS seeded prompts of LEN tokens, 24 new
    # tokens each, entries "BS:LEN"; token ids to DRIVE_LOGIT_PROBE_DIR/<tag>_dec<BS>_<LEN>.json (breakable decode check at capture sizes)
    for _ent in os.environ["DRIVE_DECODE_PROBE"].split():
        BS, PL = map(int, _ent.split(":"))
        _ps = [TokensPrompt(prompt_token_ids=torch.randint(1000, 100000, (PL,), generator=torch.Generator().manual_seed(777 + i)).tolist()) for i in range(BS)]
        _outs = llm.generate(_ps, SamplingParams(max_tokens=24, temperature=0.0, ignore_eos=True), use_tqdm=False)
        json.dump([list(o.outputs[0].token_ids) for o in _outs], open(os.path.join(os.environ["DRIVE_LOGIT_PROBE_DIR"],
                  f"{os.environ.get('DRIVE_LOGIT_PROBE_TAG', a.arm)}_dec{BS}_{PL}.json"), "w"))
    print("decode_probe saved", os.environ["DRIVE_DECODE_PROBE"], flush=True)
# r27 (user rule 10-13): no first-time trace or cuBLAS learn inside any measured window of ours; with DRIVE_REQUIRE_WARM=1 the run
# fails (exit 3, after writing its JSON with window_violations) if one did
viol = []
def _walk_viol(x, path):
    if isinstance(x, dict):
        for k_, kk in (("learns", "traces"), ("learns_in_window", "traces_in_window")):
            if (x.get(k_) or 0) > 0 or (x.get(kk) or 0) > 0:
                viol.append((path, x.get(k_), x.get(kk)))
        for k_, v_ in x.items():
            if k_ not in ("V", "rows"):
                _walk_viol(v_, f"{path}/{k_}")
if V is not None:
    for sec in ("decode", "prefill", "ix", "cb", "async", "dcurve"):
        _walk_viol(R.get(sec, {}), sec)
    R["window_violations"] = viol
if os.environ.get("DRIVE_GC_LOG") == "1":
    R["gc_events"] = {"n_by_gen": dict(collections.Counter(g for g, *_ in GC_EVENTS)), "ms_by_gen": {g: round(sum(d for g2, d, *_ in GC_EVENTS if g2 == g), 2) for g in {e[0] for e in GC_EVENTS}},
                      "events": [(g, round(d, 2), c, i) for g, d, c, _, i in GC_EVENTS][-5000:]}
if KID_ON:
    R["kid"] = R_KID
if TACTICS is not None:
    R["tactics"] = {k: dict(v) for k, v in TACTICS.items()}
json.dump(R, open(a.out, "w"), indent=1, default=str)
if viol:
    print("WINDOW VIOLATIONS (traces / learns inside measured steps):", viol[:20], flush=True)
    if os.environ.get("DRIVE_REQUIRE_WARM") == "1":
        sys.exit(3)
_hs = {C: d.get("harness_syncs_in_window") for C, d in R.get("cb", {}).items() if d.get("harness_syncs_in_window")}
if _hs:  # r27 (10-14): a harness sync inside an unsynced churn window fails the run (after its JSON is written)
    print("HARNESS SYNCS INSIDE THE UNSYNCED CHURN WINDOW:", _hs, flush=True)
    sys.exit(4)
print(json.dumps(R, default=str)[:20000], flush=True)
