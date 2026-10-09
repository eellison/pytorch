# Arm V under `vllm serve`, through vLLM's own extension points only (no patching of vLLM's server or worker classes):
# - worker side: `--worker-extension-cls armv_serve.ArmVWorkerExt` mixes the armv_* methods into the GPU worker, so they
#   run in the engine-core/worker process via collective_rpc (vLLM serializes them with the engine's steps);
# - HTTP side: the `vllm.endpoint_plugins` entry point `armv_endpoints` (dist-info here; loaded only when VLLM_PLUGINS
#   names it) adds /armv/* routes that call those methods through the API server's EngineClient.collective_rpc.
# Routes: POST /armv/install {"mode": "trace"|"policy", "first_calls": bool}; GET /armv/counters[?first_calls=1];
# POST /armv/check {"on": bool} (V compares every replayed step against the model's own eager forward: adapter checks);
# GET /armv/summary (the adapter's full summary(): per-entry variants, boundaries, declines).
# GET /armv/breakdown: cumulative per-win counters (_Stats; the bench runner diffs them per point): steps, the cudagraph
# dispatch modes vLLM chose (FULL / PIECEWISE / NONE) and the padded token / request rows, and for V: replays, eager and
# trace calls, new exact keys, out-of-band learns and their host seconds (separate from step time), static boundaries.
# POST /armv/profile {"steps": N} then GET /armv/profile: a torch.profiler window over the next N execute_model calls;
# per step, CUDA runtime kernel launches vs graph launches (launches outside graphs).
# mode "observe" (default / eager arms): only _Stats, no host trace.
# mode "policy" = arm eagerm (stock torch path, enforce_eager): no host trace, only arm V's max_seq_len policy on vLLM's
# eager metadata (V's bitwise reference). The adapter dir is ARMV_SERVE_ADAPTER (default ../armV_serve).
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))

# ARMV_CUTE_OBSERVE=1 (set by the host-trace SKUs only): observe cute.compile from the moment vLLM imports this module,
# i.e. at worker init (--worker-extension-cls) and in the API server (endpoint plugin), before the engine compiles any
# CuTe DSL kernel (FlashInfer's NVFP4 GEMM compiles at the profile run). A kernel compiled before observation has no
# captured host program / binder spec, so under a trace it is only a foreign TVM-FFI call (an eager step).
if os.environ.get("ARMV_CUTE_OBSERVE") == "1":
    import cutlass  # noqa: F401
    import torch.cuda._host_trace_cute as _cute

    _cute.install()


class _Stats:
    """Cumulative per-win counters of one worker (installed once with the arm)."""

    def __init__(self, runner, V):
        import collections

        self.V = V
        self.steps = 0  # execute_model calls
        self.cg_modes = collections.Counter()  # per step: the cudagraph mode vLLM dispatched (V runs NONE)
        self.padded_token_rows = 0  # descriptor num_tokens - real tokens, summed over steps
        self.padded_req_rows = 0
        self.learns = 0  # out-of-band harvests (provider.learn calls that harvested)
        self.learn_s = 0.0  # their host seconds (includes their syncs)
        self.forward_s = 0.0  # V: host seconds in the model forward (step entry to return, unsynced)
        self.profile = None  # (steps wanted, steps seen, torch.profiler.profile) while a window is armed
        self.profile_result = None
        orig_exec = runner.execute_model

        self.last_desc = None  # the step's dispatch result (dispatch can run more than once per step: the last one is used)

        def execute_model(*a, **k):
            self.steps += 1
            self._profile_tick()
            self.last_desc = None
            try:
                return orig_exec(*a, **k)
            finally:
                if self.last_desc is not None:
                    desc, num_reqs, num_tokens = self.last_desc
                    self.cg_modes[desc.cg_mode.name] += 1
                    self.padded_token_rows += max(desc.num_tokens - num_tokens, 0)
                    if desc.num_reqs is not None:
                        self.padded_req_rows += max(desc.num_reqs - num_reqs, 0)
        runner.execute_model = execute_model
        cgm = getattr(runner, "cudagraph_manager", None)
        if cgm is not None:
            orig_dispatch = cgm.dispatch

            def dispatch(num_reqs, num_tokens, *a, **k):
                desc = orig_dispatch(num_reqs, num_tokens, *a, **k)
                self.last_desc = (desc, num_reqs, num_tokens)
                return desc
            cgm.dispatch = dispatch
        if V is not None:
            prov = V.provider
            orig_learn = prov.learn

            def learn(*a, **k):
                h, t0 = prov.harvests, time.perf_counter()
                try:
                    return orig_learn(*a, **k)
                finally:
                    if prov.harvests > h:
                        self.learns += prov.harvests - h
                        self.learn_s += time.perf_counter() - t0
            prov.learn = learn
            fwd = V.model.forward

            def forward(*a, **k):
                t0 = time.perf_counter()
                try:
                    return fwd(*a, **k)
                finally:
                    self.forward_s += time.perf_counter() - t0
            V.model.forward = forward

    def snapshot(self):
        res = {"steps": self.steps, "cg_modes": dict(self.cg_modes), "padded_token_rows": self.padded_token_rows,
               "padded_req_rows": self.padded_req_rows, "learns": self.learns, "learn_s": self.learn_s, "forward_s": self.forward_s}
        V = self.V
        if V is not None:
            es = list(V.entries.values())
            c = V.counters()
            res.update(replays=c["replays"], traces=c["traces"], entry_eager=c["eager"], variants=c["variants"],
                       eager_path_calls={k: n for k, n in V.calls.items() if k.startswith("eager")}, exact_keys=len(V.seen),
                       boundaries_static=sum(_boundaries(v) for e in es for v in e.variants),
                       pad_rows=dict(getattr(V, "pad_rows", {})), vp_sampled=V.calls.get("vp_sampled", 0),
                       redispatches=sum(getattr(e, "redispatches", 0) for e in es), redispatch_s=sum(getattr(e, "redispatch_s", 0.0) for e in es))
            if hasattr(V, "bound"):
                res["bound_hits"] = sum(V.bound.hits())
        return res

    def _profile_tick(self):
        if self.profile is None:
            return
        want, seen, prof = self.profile
        if prof is None:
            import torch

            prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
            prof.__enter__()
            self.profile = (want, 0, prof)
            return
        seen += 1
        self.profile = (want, seen, prof)
        if seen < want:
            return
        prof.__exit__(None, None, None)
        self.profile = None
        # measurement only: CUDA runtime/driver API events by name (cudaLaunchKernel, cuLaunchKernelEx, cudaGraphLaunch, ...)
        launches = graph = copies = 0
        for e in prof.events():
            n = e.name
            if "GraphLaunch" in n:
                graph += 1
            elif "LaunchKernel" in n or n.startswith("cudaLaunchCooperativeKernel"):
                launches += 1
            elif n.startswith(("cudaMemcpy", "cudaMemset", "cuMemcpy", "cuMemset")):
                copies += 1
        self.profile_result = {"steps": want, "kernel_launches_outside_graphs": launches, "graph_launches": graph, "memcpy_memset": copies,
                               "per_step": {"kernel_launches_outside_graphs": launches / want, "graph_launches": graph / want, "memcpy_memset": copies / want}}


class ArmVWorkerExt:
    def armv_install(self, mode="trace", first_calls=False):
        if getattr(self, "_armv", None) is not None or getattr(self, "_armv_policy", False) or getattr(self, "_armv_stats", None) is not None:
            return {"installed": False, "why": "already installed"}
        runner = self.model_runner
        if mode == "observe":
            self._armv_stats = _Stats(runner, None)
            return {"installed": True, "mode": mode}
        if mode == "policy":
            _policy_only(runner)
            self._armv_policy = True
            self._armv_stats = _Stats(runner, None)
            return {"installed": True, "mode": mode, "max_model_len": runner.max_model_len}
        sys.path.insert(0, os.environ.get("ARMV_SERVE_ADAPTER", os.path.join(_HERE, "..", "armV_serve")))
        import adapter
        import adapter_cpp

        t0 = time.perf_counter()
        # the bound decode entry needs traced metadata: not with ARMV_METADATA=eager (V-stock) or a hybrid model
        cpp = adapter_cpp.BOUND and adapter.METADATA == "trace" and not (getattr(adapter, "HYBRID", False) and len(runner.attn_groups) > 1)
        V = (adapter_cpp.ArmVcpp if cpp else adapter.ArmV)(runner)
        V.install()
        self._armv = V
        self._armv_first = _record_first_calls(V) if first_calls else None
        self._armv_stats = _Stats(runner, V)
        return {"installed": True, "mode": mode, "arm": type(V).__name__, "replay_kw": {k: str(v) for k, v in V.replay_kw.items()},
                "install_s": time.perf_counter() - t0, "adapter": adapter.__file__, "hybrid": getattr(V, "hybrid", False),
                "attrs": len(V.attrs), "params": len(V.params)}

    def armv_counters(self, first_calls=False):
        import torch

        V = getattr(self, "_armv", None)
        res = {"reserved_mib": torch.cuda.memory_reserved() >> 20, "allocated_mib": torch.cuda.memory_allocated() >> 20,
               "max_allocated_mib": torch.cuda.max_memory_allocated() >> 20, "policy_only": getattr(self, "_armv_policy", False)}
        if V is None:
            return res
        es = list(V.entries.values())
        res.update(V.counters())
        res.update(keys=len(es), exact_keys=len(V.seen), bad_keys=len(V.bad), harvests=V.provider.harvests, bindings=len(V.provider.bindings),
                   relowers=sum(getattr(e, "relowers", 0) for e in es), learned=sum(getattr(e, "learned", 0) for e in es))
        variants = [v for e in es for v in e.variants]
        res["boundaries"] = sum(_boundaries(v) for v in variants)
        res["segments"] = sum(len(v.captured.segments) for v in variants if v.captured is not None)
        res["learner_variants"] = sum(bool(getattr(v, "learns", False)) for v in variants)
        if hasattr(V, "bound"):
            res["bound"] = {"hits": sum(V.bound.hits()), "plans": len(V.dec_keys)}
        checks = V.checks
        res["checks"] = {"n": len(checks), "bitwise": sum(c["bitwise"] for c in checks),
                         "bitwise_vs_stock": sum(c.get("bitwise_vs_stock", False) for c in checks),
                         "with_stock": sum("bitwise_vs_stock" in c for c in checks),
                         "first_bad": [c for c in checks if not c["bitwise"]][:5]}
        if first_calls and getattr(self, "_armv_first", None) is not None:
            res["first_calls"] = [{"key": repr(k), "calls": v} for k, v in self._armv_first.items()]
        return res

    def armv_check(self, on=True):
        V = self._armv
        V.check = bool(on)
        return {"check": V.check, "checks_so_far": len(V.checks)}

    def armv_summary(self):
        return self._armv.summary()

    def armv_breakdown(self):
        st = getattr(self, "_armv_stats", None)
        return None if st is None else st.snapshot()

    def armv_profile(self, steps=0):
        """steps > 0 arms a window over the next `steps` execute_model calls; 0 returns the last result (None while armed)."""
        st = self._armv_stats
        if steps > 0:
            st.profile, st.profile_result = (int(steps), 0, None), None
            return {"armed": int(steps)}
        return st.profile_result


def _boundaries(v):
    if v.captured is None:
        return 0
    return sum(not isinstance(s, range) and not s.call.host for s in v.captured.lowered.steps)


def _record_first_calls(V):
    """Per exact key (V.last_exact, set by the deferred path), the first 6 calls: host seconds (unsynced) and the entry's
    traces / replays / eager after the call. Keys the bound decode path serves never reach here after their plan exists."""
    first = {}
    fwd = V.forward

    def forward(input_ids, positions, *args, **kw):
        V.last_exact = None
        t0 = time.perf_counter()
        out = fwd(input_ids, positions, *args, **kw)
        dt = time.perf_counter() - t0
        exact = V.last_exact
        if exact is not None:
            rec = first.setdefault(exact, [])
            if len(rec) < 6:
                e = V.entries.get(V._entry_key(exact[0], exact[1], exact[2], exact[4]))
                rec.append((dt, getattr(e, "traces", None), getattr(e, "replays", None), getattr(e, "eager", None)))
        return out
    V.model.forward = forward
    return first


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


class ArmVEndpoints:
    name = "armv_endpoints"
    required_tasks = ("generate",)

    def __init__(self):
        self.engine = None

    def attach_router(self, app):
        from fastapi import Request

        async def rpc(method, **kw):
            (res,) = await self.engine.collective_rpc(method, kwargs=kw)  # one worker (TP1)
            return res

        @app.post("/armv/install")
        async def install(request: Request):
            body = await request.json() if await request.body() else {}
            return await rpc("armv_install", mode=body.get("mode", "trace"), first_calls=bool(body.get("first_calls", False)))

        @app.get("/armv/counters")
        async def counters(first_calls: int = 0):
            return await rpc("armv_counters", first_calls=bool(first_calls))

        @app.post("/armv/check")
        async def check(request: Request):
            body = await request.json()
            return await rpc("armv_check", on=bool(body.get("on", True)))

        @app.get("/armv/summary")
        async def summary():
            return await rpc("armv_summary")

        @app.get("/armv/breakdown")
        async def breakdown():
            return await rpc("armv_breakdown")

        @app.post("/armv/profile")
        async def profile_arm(request: Request):
            body = await request.json()
            return await rpc("armv_profile", steps=int(body.get("steps", 20)))

        @app.get("/armv/profile")
        async def profile_get():
            return await rpc("armv_profile", steps=0)

    async def init_state(self, engine_client, state, args):
        self.engine = engine_client
