# Arm V under `vllm serve` (phase 6 step 4): a vLLM general plugin (entry point in armv_serve-0.0.dist-info; on the path only
# through VLLM_EXTRA_PYTHONPATH in bench/run_serve.sh). It loads in every vLLM process. In the engine core it wraps
# Worker.compile_or_warm_up_model so that, once warmup is done, the model runner gets the same install that bench/drive.py
# does: ArmVcpp (the bound decode entry) on python_vllm_cpp.sh, otherwise ArmV.
# ARMV_SERVE_LOG=path: once a second at most, from the forward, rewrite path with the counters and a per-key record of
# the first 6 Python-path calls (host seconds, unsynced, plus the entry's traces/replays/eager after each call). The
# record covers keys that the bound never sees: first-seen prefill and mixed T, and decode before the bound has its plan.
# ARMV_SERVE_LOG_EVERY=s: also rewrite it every s seconds from a daemon thread, so the file is current while the server
# is idle (between warm-up and serving). The record adds relowers, learned, harvests and torch.cuda.memory_reserved.
# ARMV_SERVE_MODE=policy (arm eager-model, stock torch): no host trace; only arm V's max_seq_len policy (ArmV._policy:
# decode and prefill max_seq_len = max_model_len, as vLLM's FULL capture uses) on the eager metadata. V's bitwise reference.
# ARMV_SERVE_TRACE=decode (arm Vd): steps with a prefill (prefill and mixed) run V's eager path (ArmV._eager, same policy),
# so only decode keys are traced.
import json
import os
import sys
import threading
import time


def register():
    from vllm.v1.worker.gpu_worker import Worker

    if getattr(Worker, "_armv_serve", False):
        return
    Worker._armv_serve = True
    orig = Worker.compile_or_warm_up_model

    def compile_or_warm_up_model(self):
        res = orig(self)
        (_policy_only if os.environ.get("ARMV_SERVE_MODE") == "policy" else _install)(self.model_runner)
        return res
    Worker.compile_or_warm_up_model = compile_or_warm_up_model


class _LastKey(dict):
    last = None

    def get(self, key, default=None):
        self.last = key
        return dict.get(self, key, default)


def _policy_only(runner):
    ms, L = runner.model_state, runner.max_model_len
    orig = ms.prepare_attn

    def prepare_attn(*args, **kw):
        amd = orig(*args, **kw)
        if not kw.get("for_capture") and isinstance(amd, dict):
            for md in {id(m): m for m in amd.values()}.values():
                for part in (getattr(md, "decode", None), getattr(md, "prefill", None)):
                    if part is not None:
                        part.max_seq_len = L
        return amd
    ms.prepare_attn = prepare_attn
    print(f"armv_serve: max_seq_len policy only, max_model_len={L}", flush=True)


def _install(runner):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    import adapter
    import adapter_cpp
    from vllm.forward_context import get_forward_context

    V = (adapter_cpp.ArmVcpp if adapter_cpp.BOUND else adapter.ArmV)(runner)
    V.install()
    V.entries = _LastKey(V.entries)
    path = os.environ.get("ARMV_SERVE_LOG")
    first, n_calls, t_dump = {}, [0], [time.perf_counter()]
    fwd = V.forward
    decode_only = os.environ.get("ARMV_SERVE_TRACE") == "decode"

    def forward(input_ids, positions, *args, **kw):
        V.entries.last = V.last_exact = None
        t0 = time.perf_counter()
        fc = get_forward_context()
        if decode_only and isinstance(fc.attn_metadata, adapter._Pending) and fc.attn_metadata.input_batch.has_prefill:
            out = V._eager("serve_eager_prefill", fc, input_ids, positions, *args)
        else:
            out = fwd(input_ids, positions, *args, **kw)
        t1 = time.perf_counter()
        n_calls[0] += 1
        key = V.entries.last
        if key is not None:
            rec = first.setdefault(V.last_exact or key, [])  # per exact sizes (an ARMV_FAMILY entry holds many)
            if len(rec) < 6:
                e = dict.get(V.entries, key)
                rec.append((t1 - t0, getattr(e, "traces", None), getattr(e, "replays", None), getattr(e, "eager", None)))
        if path and t1 - t_dump[0] > 1.0:
            t_dump[0] = t1
            _dump(V, path, first, n_calls[0])
        return out
    V.model.forward = forward
    every = float(os.environ.get("ARMV_SERVE_LOG_EVERY", "0"))
    if path and every > 0:
        def loop():
            while True:
                time.sleep(every)
                try:
                    _dump(V, path, first, n_calls[0])
                except RuntimeError:  # a dict changed size under the forward; next round
                    pass
        threading.Thread(target=loop, daemon=True).start()
    print(f"armv_serve: installed {type(V).__name__} replay_kw={V.replay_kw}", flush=True)


def _dump(V, path, first, n_calls):
    res = {"arm": type(V).__name__, "forward_calls": n_calls, "keys": len(V.entries)}
    res.update(V.counters())
    es = list(dict.values(V.entries))
    res.update(relowers=sum(getattr(e, "relowers", 0) for e in es), learned=sum(getattr(e, "learned", 0) for e in es), harvests=V.provider.harvests)
    import torch

    res["reserved_mib"] = torch.cuda.memory_reserved() >> 20
    if hasattr(V, "bound"):
        res["bound"] = {"hits": sum(V.bound.hits()), "plans": len(V.dec_keys)}
    res["first_calls"] = [[str(k[0]), k[1], v] for k, v in list(first.items())]
    tmp = f"{path}.{threading.get_ident()}.tmp"
    with open(tmp, "w") as f:
        json.dump(res, f)
    os.replace(tmp, path)
