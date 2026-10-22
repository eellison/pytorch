# R1 (Qwen3.8-27B NVFP4) arm V probe, in process (no server): LLM(recipe args + --language-model-only, enforce_eager),
# arm V from armV_serve installed, a few prefills / decodes / a mixed step, then the adapter's summary (per entry: traces,
# variants, declines, eager reasons, decline sites) and the check rows. Same adapter and config as session.sh's V arm.
#   python r1_probe.py --out X.json [--check] [--kv-gib 24]
import argparse
import json
import os
import sys

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
p.add_argument("--model", default="/data/eellison/models/Qwen3.8-27B-NVFP4")
p.add_argument("--check", action="store_true")
p.add_argument("--kv-gib", type=float, default=24)
p.add_argument("--max-model-len", type=int, default=262144)
p.add_argument("--arm", default="V", choices=["V", "eager"])
p.add_argument("--sym-origins", action="store_true")
p.add_argument("--zero-kv", action="store_true", help="diagnosis: zero every layer kv_cache tensor (attention KV, GDN conv/ssm state) after init")
p.add_argument("--kv-dtype", default="fp8")
p.add_argument("--gdn-prefill", default=None)
p.add_argument("--prompts", default="std", choices=["std", "many"])
p.add_argument("--cute-observe", action="store_true", help="observe cute.compile from process start (torch.cuda._host_trace_cute.install before the engine compiles anything)")
p.add_argument("--adapter-dir", default="armV_serve", help="adapter package dir under serve/ (armV_stock: V-stock + hybrid)")
p.add_argument("--moe-backend", default=None, help="vLLM --moe-backend (diagnosis; default: vLLM picks)")
p.add_argument("--attn", default=None, help="vLLM attention_backend (diagnosis; default: vLLM picks)")
p.add_argument("--llm-kw", default=None, help="JSON dict of extra LLM(...) kwargs (they override the probe's defaults)")
p.add_argument("--disable-kernels", default=None, help="diagnosis: VLLM_DISABLED_KERNELS (e.g. FlashInferCuteDslNvFp4LinearKernel)")
p.add_argument("--decode-only", action="store_true")
a = p.parse_args()
if a.disable_kernels:
    os.environ["VLLM_DISABLED_KERNELS"] = a.disable_kernels

import torch

if a.cute_observe:
    import cutlass  # noqa: F401  (install() observes only with cutlass loaded)
    import torch.cuda._host_trace_cute as _cute

    _cute.install()
    print("cute observing", _cute.observing(), flush=True)
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

KW = dict(model=a.model, tensor_parallel_size=1, enforce_eager=True, kv_cache_dtype=a.kv_dtype, language_model_only=True,
          max_model_len=a.max_model_len, kv_cache_memory_bytes=int(a.kv_gib * (1 << 30)), max_num_seqs=64,
          gpu_memory_utilization=0.2, enable_prefix_caching=False, seed=0,
          **({"gdn_prefill_backend": a.gdn_prefill} if a.gdn_prefill else {}), **({"moe_backend": a.moe_backend} if a.moe_backend else {}),
          **({"attention_backend": a.attn} if a.attn else {}))
KW.update(json.loads(a.llm_kw) if a.llm_kw else {})
print("LLM kwargs", KW, flush=True)
llm = LLM(**KW)
runner = llm.llm_engine.model_executor.driver_worker.worker.model_runner
if a.zero_kv:
    n = 0
    for m in runner.model.modules():
        kv = vars(m).get("kv_cache")
        for t in (kv if isinstance(kv, (list, tuple)) else [kv]):
            if isinstance(t, torch.Tensor):
                t.zero_()
                n += 1
    print("zeroed kv tensors", n, flush=True)
V = None
SYMS = {}  # diagnosis (--sym-origins): symbol name -> (source string, creation stack) from the tracer's Env.symbol
if a.sym_origins:
    import traceback

    import torch.cuda._host_trace_ir as _ir

    _orig_symbol = getattr(getattr(_ir, "Env", None), "symbol", None)

    def _symbol(self, value, source, *, positive=False):
        out = _orig_symbol(self, value, source, positive=positive)
        name = next(reversed(self.names))  # Env.symbol just inserted it
        stack = [f"{os.path.basename(f.filename)}:{f.lineno} {f.name}" for f in traceback.extract_stack()[-16:-1]]
        SYMS[name] = (source, stack)
        return out
    if _orig_symbol is not None:
        _ir.Env.symbol = _symbol
if a.arm == "V":
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), a.adapter_dir))
    import adapter

    if a.sym_origins:  # diagnosis: the full stack (torch frames too) where an unsourced-symbol decline is raised
        import torch.cuda._host_trace as _ht

        DECL_STACKS = []
        _prev_init = _ht.Declined.__init__

        def _decl_init(exc, *args, **kw):
            _prev_init(exc, *args, **kw)
            if "is not read from the call's inputs" in str(exc) and len(DECL_STACKS) < 4:
                import traceback

                DECL_STACKS.append((str(exc)[:120], [f"{os.path.basename(f.filename)}:{f.lineno} {f.name}: {(f.line or '')[:100]}" for f in traceback.extract_stack()[-30:-1]]))
        _ht.Declined.__init__ = _decl_init
        import torch.cuda._host_trace_lower_tape as _lt
        from torch.cuda._host_trace_ir import render as _ir_render

        EAGER_FAIL = []
        _prev_eager = getattr(getattr(_lt, "_TapeLowering", None), "eager_call", None)

        def _eager_call(self, seq, call):
            try:
                return _prev_eager(self, seq, call)
            except _ht.Declined as e:
                if len(EAGER_FAIL) < 6:
                    tgt = call.target
                    desc = tgt if not isinstance(tgt, tuple) else (tgt[0], getattr(tgt[1], "__name__", type(tgt[1]).__name__), tgt[2:4] if tgt[0] == "cute" else None)
                    syms = [(i, str(v.node.expr) if hasattr(v.node, "expr") else _ir_render(v.node.node)) for i, v in enumerate(torch.utils._pytree.tree_leaves((call.args, call.kwargs))) if isinstance(v, torch.SymInt)]
                    EAGER_FAIL.append((str(e)[:120], repr(desc)[:400], syms[:12], (call.reason or "")[:300]))
                raise
        if _prev_eager is not None:
            _lt._TapeLowering.eager_call = _eager_call
        # a native variant that rejects its spec: which keyed sites / selectors claim the same launch record
        import collections as _co
        import torch.cuda._host_trace_replay as _rp

        SPEC_FAIL = []
        _prev_flatten, _prev_native = getattr(_rp, "flatten_variant", None), getattr(_rp, "native_variant", None)
        _last = {}

        def _flatten(captured, *a, **k):
            _last["captured"] = captured
            return _prev_flatten(captured, *a, **k)

        def _native(spec):
            try:
                return _prev_native(spec)
            except AssertionError as e:
                cap = _last.get("captured")
                if cap is not None and len(SPEC_FAIL) < 3:
                    low = cap.lowered
                    owners = _co.defaultdict(list)
                    for i, st in enumerate(low.sites):
                        for n in st.nodes:
                            owners[n].append(("site", i, str(st.site.op), len(st.nodes)))
                    for i, sl in enumerate(low.selectors):
                        op = low.tape.ops[sl.op] if isinstance(sl.op, int) and sl.op < len(low.tape.ops) else sl.op
                        for n in sl.nodes:
                            owners[n].append(("selector", i, repr(getattr(op, "name", op))[:120], len(sl.nodes)))
                    dup = {n: o for n, o in owners.items() if len(o) > 1}
                    names = {n: repr(getattr(cap.launches[n].launch, "name", type(cap.launches[n].launch).__name__))[:120] for n in list(dup)[:6] if n < len(cap.launches)}
                    SPEC_FAIL.append((str(e)[:160], len(low.sites), len(low.selectors), len(cap.launches), {n: dup[n] for n in list(dup)[:6]}, names))
                    print("SPECFAIL", SPEC_FAIL[-1], flush=True)  # now: the error escapes and ends the engine
                raise
        if hasattr(_rp, "flatten_variant") and hasattr(_rp, "native_variant"):
            _rp.flatten_variant, _rp.native_variant = _flatten, _native
    V = adapter.ArmV(runner)
    V.install()
    V.check = a.check
g = torch.Generator().manual_seed(0)
prompt = lambda n: TokensPrompt(prompt_token_ids=torch.randint(1000, 100000, (n,), generator=g).tolist())
sp = lambda n: SamplingParams(max_tokens=n, ignore_eos=True, temperature=0.0)
R = {"arm": a.arm, "outputs": {}}
WORK = [("p100", [prompt(100)], 6), ("p17x3", [prompt(17) for _ in range(3)], 6), ("p1000", [prompt(1000)], 4)]
if a.prompts == "many":  # the serve_check lengths, one request each, 16 tokens
    WORK = [(f"p{n}", [prompt(n)], 16) for n in (1, 5, 16, 17, 33, 100, 129, 257, 500, 1000, 1025, 2048, 3000, 4097, 6000, 8000)]
for name, ps, n in WORK:
    outs = llm.generate(ps, sp(n), use_tqdm=False)
    R["outputs"][name] = [list(o.outputs[0].token_ids) for o in outs]
    print(name, R["outputs"][name], flush=True)
if V is not None:
    s = V.summary()
    s["decline_sites"] = adapter.DECLINE_SITES.most_common(30) and [f"{n}x {m} @ {w}" for (m, w), n in adapter.DECLINE_SITES.most_common(30)]
    R["summary"] = s
    print("COUNTERS", json.dumps(V.counters()), flush=True)
    print("LEARNED", {"harvests": V.provider.harvests, "bindings": len(V.provider.bindings), "by_op": dict(list(s["learned_by_op"].items())[:12]),
                      "refused": s.get("refused", [])[:6]}, flush=True)
    for k, e in s["entries"].items():
        pv = e.get("per_variant", [])
        if pv:
            print("VARIANTS", k[:100], [{x: v.get(x) for x in ("kernels", "eager_calls", "boundaries", "opaque", "segments", "eager_ops")} for v in pv][:4], flush=True)
    for k, e in s["entries"].items():
        print("ENTRY", k[:160], {x: e[x] for x in ("traces", "replays", "eager", "variants")}, "declines", e["declines"][:3], "reasons", e["reasons"][:3], flush=True)
    def _kdiff(a, b, path=""):  # diagnosis: where two entry keys first differ
        if type(a) is not type(b):
            return f"{path}: {type(a).__name__} vs {type(b).__name__}"
        if isinstance(a, tuple):
            if len(a) != len(b):
                return f"{path}: len {len(a)} vs {len(b)}"
            for i, (x, y) in enumerate(zip(a, b)):
                if x != y:
                    return _kdiff(x, y, f"{path}[{i}]{'=' + repr(a[0])[:40] if a and isinstance(a[0], str) else ''}")
            return None
        return f"{path}: {repr(a)[:120]} vs {repr(b)[:120]}"
    by_kind = {}
    for k in V.entries:
        by_kind.setdefault(k[:2], []).append(k)
    for kind, ks in by_kind.items():
        if len(ks) > 1:
            print("KEYDIFF", kind, len(ks), [_kdiff(ks[0], k) for k in ks[1:4]], flush=True)
    for k, e in V.entries.items():  # candidate 3+: why each call traced again
        rc = getattr(e, "retrace_causes", None)
        if rc:
            print("RETRACE", str(k)[:120], [(c, n) for c, n in rc.items()][:10], flush=True)
    print("DECLINE_SITES", s["decline_sites"][:15], flush=True)
    if a.sym_origins:
        import re  # diagnosis only: the symbol names in the decline texts
        for d in {d for e in s["entries"].values() for d in e["declines"]}:
            for name in re.findall(r"\b[sz]f?\d+\b", d):
                print("SYM", name, SYMS.get(name), flush=True)
    if a.sym_origins:
        for row in EAGER_FAIL:
            print("EAGERFAIL", row, flush=True)
        for row in SPEC_FAIL:
            print("SPECFAIL", row, flush=True)
        for msg, st in DECL_STACKS:
            print("DECLSTACK", msg, flush=True)
            for fr in st:
                print("   ", fr, flush=True)
    print("CHECKS", [(c["kind"], c["bitwise"], c.get("state_bitwise")) for c in s["checks"]][:40], flush=True)
deg = [k for k, v in R["outputs"].items() for o in v if len(set(o)) == 1]
print("DEGENERATE", len(deg), "of", sum(len(v) for v in R["outputs"].values()), deg, flush=True)
json.dump(R, open(a.out, "w"), indent=1, default=str)
