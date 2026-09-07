"""One benchmark cell: (case, M, K, mode). Prints a single JSON line last."""
import hashlib, json, statistics, sys, traceback
import torch
import torch.nn.functional as F
from torch._inductor import config, metrics
from torch._inductor.utils import fresh_inductor_cache, run_and_get_code
from quack.blockscaled import quantize as qz

case, M, K, mode = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
WARMUP, SAMPLES, CALLS = 20, 50, 100
DYNAMIC_DEFAULT = {"mxfp6_e2m3", "mxfp6_e3m2", "mxfp6_e2m3_packed", "mxfp6_e3m2_packed", "mxfp4_byte"}

def percentile(values, q):
    s = sorted(values); return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]

def graph_bench(fn):
    for _ in range(WARMUP): fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(CALLS): fn()
    g.replay(); torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    vals = []
    for _ in range(SAMPLES):
        s.record(); g.replay(); e.record(); e.synchronize()
        vals.append(s.elapsed_time(e) * 1000.0 / CALLS)
    return vals, "graph"

def event_bench(fn):
    for _ in range(WARMUP): fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    vals = []
    for _ in range(SAMPLES):
        s.record()
        for _ in range(CALLS): fn()
        e.record(); e.synchronize()
        vals.append(s.elapsed_time(e) * 1000.0 / CALLS)
    return vals, "events"

def out_hash(out):
    outs = list(out) if isinstance(out, (tuple, list)) else [out]
    h = 0
    for t in outs:
        b = t.view(torch.uint8) if t.dtype != torch.float32 else t.view(torch.int32)
        h ^= int(hashlib.sha1(f"{tuple(t.shape)}|{t.dtype}|{b.long().sum().item()}|{b.flatten()[::7919].long().sum().item()}".encode()).hexdigest()[:15], 16)
    return h

torch.manual_seed(0)
x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
w = torch.randn(K, device="cuda", dtype=torch.bfloat16)
fmt = case.split("+")[-1]
if fmt == "mxfp8_dim0":
    quant = qz.to_mx_dim0
else:
    quant = qz.QUANTIZERS[fmt][0]
if case.startswith("rmsnorm+"):
    fn = lambda x, w: quant(F.rms_norm(x, (K,), w)); args = (x, w)
else:
    fn = lambda x: quant(x); args = (x,)

dynamic = fmt in DYNAMIC_DEFAULT
if mode.endswith("_dynamic"): dynamic = True; mode_base = mode[: -len("_dynamic")]
elif mode.endswith("_static"): dynamic = False; mode_base = mode[: -len("_static")]
else: mode_base = mode

result = {"case": case, "M": M, "K": K, "mode": mode, "dynamic": dynamic}
try:
    if mode_base == "eager":
        out = fn(*args)
        result.update(kernels=None, nested=None)
        timed = lambda: fn(*args)
    else:
        nested = mode_base in ("on", "on_mk", "on_looped", "on_persistent")
        patches = {"triton.nested_reduction": nested, "triton.cudagraphs": False, "fx_graph_cache": False,
                   "triton.multi_kernel": 1 if mode_base == "on_mk" else 0,
                   "triton.persistent_reductions": mode_base != "on_looped"}
        torch._dynamo.reset(); metrics.reset()
        import contextlib
        from torch._inductor.virtualized import V
        from torch._inductor.choices import InductorChoices
        class _Persistent(InductorChoices):
            @staticmethod
            def should_use_persistent_reduction(*a, **k): return True
        choices_ctx = V.set_choices_handler(_Persistent()) if mode_base == "on_persistent" else contextlib.nullcontext()
        with fresh_inductor_cache(), config.patch(patches), choices_ctx:
            compiled = torch.compile(fn, fullgraph=True, dynamic=dynamic)
            out, sources = run_and_get_code(compiled, *args)
            torch.cuda.synchronize()
            result.update(kernels=metrics.generated_kernel_count, nested=metrics.codegen_nested_reduction,
                          multi_kernel_defs=sum(s.count("async_compile.multi_kernel(") for s in sources))
            timed = lambda: compiled(*args)
            try:
                vals, how = graph_bench(timed)
            except Exception:
                vals, how = event_bench(timed)
            result.update(median_us=statistics.median(vals), p20_us=percentile(vals, 0.2), p80_us=percentile(vals, 0.8), timing=how)
    if mode_base == "eager":
        try:
            vals, how = graph_bench(timed)
        except Exception:
            vals, how = event_bench(timed)
        result.update(median_us=statistics.median(vals), p20_us=percentile(vals, 0.2), p80_us=percentile(vals, 0.8), timing=how)
    result["out_hash"] = out_hash(out)
except Exception as e:
    result["error"] = "".join(traceback.format_exception_only(type(e), e))[-500:].strip()
print("CELL_RESULT " + json.dumps(result), flush=True)
