"""Compile every QuACK pure-PyTorch quantizer with nested reduction on/off.

Reports fusion (codegen_nested_reduction), kernel counts, byte-exactness of
nested vs unnested vs eager, and any compile crash. No timing here.
"""
import sys, traceback, json
import torch
import torch.nn.functional as F
from torch._inductor import config, metrics
from torch._inductor.utils import fresh_inductor_cache, run_and_get_code
from quack.blockscaled import quantize as qz

M, K = int(sys.argv[1]) if len(sys.argv) > 1 else 2048, int(sys.argv[2]) if len(sys.argv) > 2 else 3072
DYNAMIC = {"mxfp6_e2m3", "mxfp6_e3m2", "mxfp6_e2m3_packed", "mxfp6_e3m2_packed", "mxfp4_byte"}

def as_bytes(t):
    return t.view(torch.uint8) if t.dtype != torch.float32 else t.view(torch.int32)

def flat_outputs(out):
    return list(out) if isinstance(out, (tuple, list)) else [out]

def exact(a, b):
    return all(torch.equal(as_bytes(x), as_bytes(y)) for x, y in zip(flat_outputs(a), flat_outputs(b)))

def compile_and_run(fn, args, nested, dynamic):
    torch._dynamo.reset(); metrics.reset()
    patches = {"triton.nested_reduction": nested, "triton.cudagraphs": False, "fx_graph_cache": False}
    with fresh_inductor_cache(), config.patch(patches):
        compiled = torch.compile(fn, fullgraph=True, dynamic=dynamic)
        out, sources = run_and_get_code(compiled, *args)
        torch.cuda.synchronize()
    names = []
    for s in sources:
        for line in s.splitlines():
            if line.startswith("def triton_"):
                names.append(line[4:line.index("(")])
    return out, {"nested": metrics.codegen_nested_reduction, "kernels": metrics.generated_kernel_count, "names": names}

def build_cases():
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(K, device="cuda", dtype=torch.bfloat16)
    cases = []
    for name, (eager_fn, _) in qz.QUANTIZERS.items():
        dyn = name in DYNAMIC
        cases.append((f"{name}", eager_fn, (x,), dyn))
        cases.append((f"rmsnorm+{name}", lambda x, w, f=eager_fn: f(F.rms_norm(x, (K,), w)), (x, w), dyn))
        if name != "nvfp4":
            def with_blocked(x, f=eager_fn):
                q, s = f(x)
                return q, qz.to_blocked(s.view(M, -1))
            cases.append((f"{name}+to_blocked", with_blocked, (x,), dyn))
    cases.append(("mxfp8_dim0", qz.to_mx_dim0, (x,), False))
    cases.append(("rmsnorm+mxfp8_dim0", lambda x, w: qz.to_mx_dim0(F.rms_norm(x, (K,), w)), (x, w), False))
    def nvfp4_pts(x):
        pts = qz.nvfp4_per_tensor_scale(x.float().abs().amax())
        return qz.to_nvfp4(x, 16, pts)
    cases.append(("nvfp4_per_tensor", nvfp4_pts, (x,), False))
    cases.append(("rmsnorm+nvfp4_per_tensor", lambda x, w: nvfp4_pts(F.rms_norm(x, (K,), w)), (x, w), False))
    return cases

rows = []
for label, fn, args, dyn in build_cases():
    row = {"case": label, "dynamic": dyn}
    try:
        ref = fn(*args)
    except Exception as e:
        row["eager_error"] = repr(e)[:200]; rows.append(row); continue
    for nested in (True, False):
        key = "on" if nested else "off"
        try:
            out, meta = compile_and_run(fn, args, nested, dyn)
            row[f"{key}_nested"] = meta["nested"]; row[f"{key}_kernels"] = meta["kernels"]
            row[f"{key}_names"] = meta["names"]
            row[f"{key}_eq_eager"] = exact(out, ref)
            if nested: on_out = out
            else: row["on_eq_off"] = exact(on_out, out) if "on_kernels" in row else None
        except Exception as e:
            row[f"{key}_error"] = "".join(traceback.format_exception_only(type(e), e))[-400:].strip()
    rows.append(row)
    print(f"{label:34s} dyn={dyn!s:5s} on: nested={row.get('on_nested','ERR')} k={row.get('on_kernels','ERR')} eq_eager={row.get('on_eq_eager','ERR')} | off: k={row.get('off_kernels','ERR')} eq_eager={row.get('off_eq_eager','ERR')} | on==off {row.get('on_eq_off','ERR')}", flush=True)
    for key in ("on_error", "off_error"):
        if key in row: print(f"    {key}: {row[key]}", flush=True)

json.dump(rows, open(f"agent_space/quack_quant/probe_{M}x{K}.json", "w"), indent=1)
