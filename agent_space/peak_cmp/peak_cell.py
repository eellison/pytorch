"""One peak-vs-peak cell: (workload, M, K[, N], impl, mode). Prints CELL_RESULT json."""
import hashlib, json, statistics, sys, traceback, contextlib
import torch, torch.nn.functional as F
from torch._higher_order_ops.inline_asm_elementwise import inline_asm_elementwise
from torch._inductor import config, inductor_prims, metrics
from torch._inductor.utils import fresh_inductor_cache, run_and_get_code
from torch._inductor.virtualized import V
from torch._inductor.choices import InductorChoices

workload, M, K = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
N = int(sys.argv[4]) if workload.startswith("gemm") else None
impl, mode = sys.argv[-2], sys.argv[-1]
WARMUP, SAMPLES, CALLS = 20, 50, 100
FP4_MAX, FP8_MAX = 6.0, 448.0
PACK_E2M1X2_ASM = "{.reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, $2, $1; cvt.u32.u8 $0, t;}"
RECIP_UE8M0_ASM = ("{.reg .pred p_zero; .reg .s32 neg_exp; .reg .f32 neg_exp_f, result; setp.eq.u32 p_zero, $1, 0; "
                   "sub.s32 neg_exp, 127, $1; cvt.rn.f32.s32 neg_exp_f, neg_exp; ex2.approx.f32 result, neg_exp_f; selp.f32 $0, 0f00000000, result, p_zero;}")

def recip_ue8m0(scale):
    return inline_asm_elementwise(scale.to(torch.int32), asm_str=RECIP_UE8M0_ASM, constraints="=f,r", dtype=torch.float32, is_pure=True, pack=1)
def swizzle_scale(scale):
    rows, cols = scale.shape
    blocks = scale.view(rows // 128, 128, cols // 4, 4).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1)
def unswizzle_scale(flat, rows, cols):
    blocks = flat.reshape(-1, 32, 4, 4).transpose(1, 2)
    return blocks.reshape(rows // 128, cols // 4, 128, 4).permute(0, 2, 1, 3).reshape(rows, cols)

# ---- our formulations (fused, swizzled scales) ----
def fp4_quant(normed, block, scale_format):
    rows, hidden = normed.shape
    g = normed.view(rows, hidden // block, block)
    amax = g.abs().amax(dim=-1)
    if scale_format == "e4m3":
        scale = (amax / FP4_MAX).clamp(min=1e-12, max=FP8_MAX).to(torch.float8_e4m3fn); inv = 1.0 / scale.float()
    else:
        scale = inductor_prims.cvt_e8m0_rceil((amax / FP4_MAX).clamp_min(1e-12)); inv = recip_ue8m0(scale)
    pairs = g.view(rows, hidden // block, block // 2, 2)
    packed = inline_asm_elementwise(pairs[..., 0].float() * inv.unsqueeze(-1), pairs[..., 1].float() * inv.unsqueeze(-1),
                                    asm_str=PACK_E2M1X2_ASM, constraints="=r,f,f", dtype=torch.int32, is_pure=True, pack=1)
    return packed.to(torch.uint8).view(rows, hidden // 2), swizzle_scale(scale)
def mxfp8_quant(normed):
    rows, hidden = normed.shape
    g = normed.view(rows, hidden // 32, 32)
    amax = g.abs().float().amax(dim=-1)
    raw = (amax / FP8_MAX).clamp_min(torch.finfo(torch.float32).tiny)
    scale = inductor_prims.cvt_e8m0_rceil(raw)
    # Multiply by the reciprocal: exact for power-of-two scales and avoids a
    # full fp32 division per element.
    q = (g.float() * recip_ue8m0(scale).unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).view(rows, hidden)
    return q, swizzle_scale(scale)
QUANT = {"nvfp4": lambda y: fp4_quant(y, 16, "e4m3"), "mxfp4": lambda y: fp4_quant(y, 32, "ue8m0"), "mxfp8": mxfp8_quant}
def rms(x, w): return F.rms_norm(x, (x.shape[-1],), w, eps=1e-6)

# ---- references / dequant ----
def dequant(q, sflat, fmt, rows, cols):
    block = 16 if fmt == "nvfp4" else 32
    scale = unswizzle_scale(sflat.reshape(-1).view(torch.uint8), rows, cols // block)
    if fmt == "nvfp4": s = scale.view(torch.float8_e4m3fn).float()
    else: s = torch.ldexp(torch.ones_like(scale, dtype=torch.float32), scale.to(torch.int32) - 127)
    if fmt == "mxfp8":
        vals = q.view(torch.float8_e4m3fn).float().view(rows, cols)
    else:
        b = q.view(torch.uint8).reshape(rows, cols // 2)
        codes = torch.stack((b & 0x0F, b >> 4), dim=-1).reshape(rows, cols).long()
        table = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=q.device)
        vals = table[codes & 7] * torch.where(codes < 8, 1.0, -1.0)
    return (vals.view(rows, cols // block, block) * s.unsqueeze(-1)).view(rows, cols)
def sha(t): return hashlib.sha1(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:12]

def graph_bench(fn):
    for _ in range(WARMUP): fn()
    torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(CALLS): fn()
    g.replay(); torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True); vals = []
    for _ in range(SAMPLES):
        s.record(); g.replay(); e.record(); e.synchronize(); vals.append(s.elapsed_time(e) * 1000 / CALLS)
    return vals, "graph"
def event_bench(fn):
    for _ in range(WARMUP): fn()
    torch.cuda.synchronize(); s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True); vals = []
    for _ in range(SAMPLES):
        s.record()
        for _ in range(CALLS): fn()
        e.record(); e.synchronize(); vals.append(s.elapsed_time(e) * 1000 / CALLS)
    return vals, "events"
def bench(fn):
    try: return graph_bench(fn)
    except Exception: return event_bench(fn)

torch.manual_seed(0)
x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16); w = torch.randn(K, device="cuda", dtype=torch.bfloat16)
fmt = workload.split("_")[-1] if workload != "rmsnorm" else None
result = {"workload": workload, "M": M, "K": K, "N": N, "impl": impl, "mode": mode}
try:
    if workload.startswith("gemm"):
        Bm = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
        ref = (x.float() @ Bm.float())
        rows, cols = M, N
        if impl == "ours":
            fn = lambda a, b: QUANT[fmt](a @ b); args = (x, Bm)
        elif impl == "quack":
            from quack.gemm_interface import gemm
            from quack.blockscaled.quantize import unpack_scale_blocked_to_2d
            fn = lambda: gemm(x, Bm, out_dtype=fmt); args = ()
        elif impl == "mm_bf16":
            fn = lambda: x @ Bm; args = ()
        elif impl == "quack_bf16":
            from quack.gemm_interface import gemm
            fn = lambda: gemm(x, Bm); args = ()
    else:
        rows, cols = M, K
        if workload == "rmsnorm":
            ref = rms(x.float(), w.float())
            if impl == "ours": fn = rms; args = (x, w)
            elif impl == "quack":
                from quack.rmsnorm import rmsnorm_fwd
                fn = lambda: rmsnorm_fwd(x, w, eps=1e-6)[0]; args = ()
            elif impl == "fi":
                from flashinfer import norm
                out = torch.empty_like(x); fn = lambda: norm.rmsnorm(x, w, 1e-6, out=out, enable_pdl=False); args = ()
        elif workload.startswith("rmsnorm_"):
            ref = rms(x, w).float()
            if impl == "ours": fn = lambda x, w: QUANT[fmt](rms(x, w)); args = (x, w)
            elif impl == "fi_fused":
                from flashinfer.cute_dsl import rmsnorm_fp4quant
                gs = torch.ones(1, device="cuda", dtype=torch.float32); blk = 16 if fmt == "nvfp4" else 32; sf = "e4m3" if fmt == "nvfp4" else "ue8m0"
                fn = lambda: rmsnorm_fp4quant(x, w, global_scale=gs, eps=1e-6, block_size=blk, scale_format=sf, is_sf_swizzled_layout=True, enable_pdl=False); args = ()
            elif impl in ("fi_composed", "quack_fi"):
                from flashinfer.quantization import fp4_quantize, mxfp4_quantize, mxfp8_quantize
                from flashinfer.tllm_enums import SfLayout
                gs = torch.ones(1, device="cuda", dtype=torch.float32)
                if fmt == "nvfp4": q = lambda y: fp4_quantize(y, gs, sf_vec_size=16, sf_use_ue8m0=False, is_sf_swizzled_layout=True, enable_pdl=False)
                elif fmt == "mxfp4": q = lambda y: mxfp4_quantize(y, backend="cuda", enable_pdl=False, sfLayout=SfLayout.layout_128x4)
                else: q = lambda y: mxfp8_quantize(y, sf_swizzle_layout=SfLayout.layout_128x4, enable_pdl=False)
                if impl == "fi_composed":
                    from flashinfer import norm
                    normed = torch.empty_like(x)
                    def fn(): norm.rmsnorm(x, w, 1e-6, out=normed, enable_pdl=False); return q(normed)
                else:
                    from quack.rmsnorm import rmsnorm_fwd
                    fn = lambda: q(rmsnorm_fwd(x, w, eps=1e-6)[0])
                args = ()
        else:  # standalone quant
            ref = x.float()
            if impl == "ours": fn = QUANT[fmt]; args = (x,)
            elif impl == "fi":
                from flashinfer.quantization import fp4_quantize, mxfp4_quantize, mxfp8_quantize
                from flashinfer.tllm_enums import SfLayout
                gs = torch.ones(1, device="cuda", dtype=torch.float32)
                if fmt == "nvfp4": fn = lambda: fp4_quantize(x, gs, sf_vec_size=16, sf_use_ue8m0=False, is_sf_swizzled_layout=True, enable_pdl=False)
                elif fmt == "mxfp4": fn = lambda: mxfp4_quantize(x, backend="cuda", enable_pdl=False, sfLayout=SfLayout.layout_128x4)
                else: fn = lambda: mxfp8_quantize(x, sf_swizzle_layout=SfLayout.layout_128x4, enable_pdl=False)
                args = ()
    if impl == "ours":
        class _Persistent(InductorChoices):
            @staticmethod
            def should_use_persistent_reduction(*a, **k): return True
        patches = {"triton.nested_reduction": True, "triton.cudagraphs": False, "fx_graph_cache": False,
                   "coordinate_descent_tuning": mode.endswith("cd"), "triton.multi_kernel": 1 if mode == "mk" else 0}
        ctx = V.set_choices_handler(_Persistent()) if mode.startswith("persistent") else contextlib.nullcontext()
        torch._dynamo.reset(); metrics.reset()
        with fresh_inductor_cache(), config.patch(patches), ctx:
            compiled = torch.compile(fn, fullgraph=True, dynamic=False)
            out, sources = run_and_get_code(compiled, *args); torch.cuda.synchronize()
            result.update(kernels=metrics.generated_kernel_count, nested=metrics.codegen_nested_reduction,
                          kernel_names=[l[4:l.index("(")][:40] for s in sources for l in s.splitlines() if l.startswith("def triton_")])
            timed = lambda: compiled(*args)
            vals, how = bench(timed)
    else:
        out = fn(); torch.cuda.synchronize()
        timed = fn
        vals, how = bench(timed)
    result.update(median_us=statistics.median(vals), p20_us=sorted(vals)[len(vals) // 5], p80_us=sorted(vals)[4 * len(vals) // 5], timing=how)
    # correctness vs fp32 reference
    if workload == "rmsnorm":
        result.update(max_abs_err=(out.float() - ref).abs().max().item(), out_sha=sha(out))
    elif workload.startswith("gemm") and impl in ("mm_bf16", "quack_bf16"):
        result.update(max_abs_err=(out.float() - ref).abs().max().item())
    else:
        if impl == "quack":
            q, sflat = out.qdata, out.scale.contiguous().view(torch.uint8).reshape(-1)
        else:
            q, sflat = out[0], out[1]
        dq = dequant(q, sflat, fmt, rows, cols)
        err = (dq - ref).abs()
        result.update(dequant_mean_abs_err=err.mean().item(), dequant_max_abs_err=err.max().item(),
                      quant_sha=sha(q.view(torch.uint8)), scale_sha=sha(sflat.view(torch.uint8)), quant_bytes=q.view(torch.uint8).numel())
        if impl == "ours" and not workload.startswith("gemm"):
            # byte comparison against flashinfer's kernel for the same op
            from flashinfer.quantization import fp4_quantize, mxfp4_quantize, mxfp8_quantize
            from flashinfer.tllm_enums import SfLayout
            gs = torch.ones(1, device="cuda", dtype=torch.float32)
            y = rms(x, w) if workload.startswith("rmsnorm_") else x
            if fmt == "nvfp4": fo = fp4_quantize(y, gs, sf_vec_size=16, sf_use_ue8m0=False, is_sf_swizzled_layout=True, enable_pdl=False)
            elif fmt == "mxfp4": fo = mxfp4_quantize(y, backend="cuda", enable_pdl=False, sfLayout=SfLayout.layout_128x4)
            else: fo = mxfp8_quantize(y, sf_swizzle_layout=SfLayout.layout_128x4, enable_pdl=False)
            qb, fb = q.view(torch.uint8).reshape(-1), fo[0].view(torch.uint8).reshape(-1)
            sb, fsb = sflat.view(torch.uint8).reshape(-1), fo[1].view(torch.uint8).reshape(-1)
            result.update(quant_mismatch_frac_vs_fi=(qb != fb).float().mean().item(), scale_mismatch_frac_vs_fi=(sb != fsb).float().mean().item())
except Exception as e:
    result["error"] = "".join(traceback.format_exception_only(type(e), e))[-600:].strip()
print("CELL_RESULT " + json.dumps(result), flush=True)
