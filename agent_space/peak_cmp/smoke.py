import torch, time
M, K, N = 2048, 3072, 4096
x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16); w = torch.randn(K, device="cuda", dtype=torch.bfloat16)
def t(fn, n=20):
    fn(); torch.cuda.synchronize(); s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n): fn()
    e.record(); e.synchronize(); return s.elapsed_time(e) * 1000 / n
def desc(o):
    if isinstance(o, torch.Tensor): return f"{tuple(o.shape)} {str(o.dtype).replace('torch.','')}"
    if isinstance(o, (tuple, list)): return "(" + ", ".join(desc(v) for v in o) + ")"
    return f"{type(o).__name__}:" + ", ".join(f"{k}={desc(getattr(o,k))}" for k in ("qdata","scale","per_tensor_scale") if hasattr(o,k))
# QuACK rmsnorm CuTe
from quack.rmsnorm import rmsnorm_fwd
out = rmsnorm_fwd(x, w, eps=1e-6); print("quack rmsnorm_fwd:", desc(out), f"{t(lambda: rmsnorm_fwd(x, w, eps=1e-6)):.1f}us", "max|d| vs F.rms_norm:", (out[0].float() - torch.nn.functional.rms_norm(x, (K,), w, eps=1e-6).float()).abs().max().item())
# flashinfer
from flashinfer.cute_dsl import rmsnorm_fp4quant
from flashinfer.quantization import fp4_quantize, mxfp4_quantize, mxfp8_quantize
from flashinfer.tllm_enums import SfLayout
from flashinfer import norm
gs = torch.ones(1, device="cuda", dtype=torch.float32)
for fmt, blk, sf in (("nvfp4", 16, "e4m3"), ("mxfp4", 32, "ue8m0")):
    o = rmsnorm_fp4quant(x, w, global_scale=gs, eps=1e-6, block_size=blk, scale_format=sf, is_sf_swizzled_layout=True, enable_pdl=False)
    print(f"fi rmsnorm_fp4quant {fmt}:", desc(o), f"{t(lambda: rmsnorm_fp4quant(x, w, global_scale=gs, eps=1e-6, block_size=blk, scale_format=sf, is_sf_swizzled_layout=True, enable_pdl=False)):.1f}us")
o = fp4_quantize(x, gs, sf_vec_size=16, sf_use_ue8m0=False, is_sf_swizzled_layout=True, enable_pdl=False); print("fi fp4_quantize nvfp4:", desc(o), f"{t(lambda: fp4_quantize(x, gs, sf_vec_size=16, sf_use_ue8m0=False, is_sf_swizzled_layout=True, enable_pdl=False)):.1f}us")
o = mxfp4_quantize(x, backend='cuda', enable_pdl=False, sfLayout=SfLayout.layout_128x4); print("fi mxfp4_quantize:", desc(o), f"{t(lambda: mxfp4_quantize(x, backend='cuda', enable_pdl=False, sfLayout=SfLayout.layout_128x4)):.1f}us")
o = mxfp8_quantize(x, sf_swizzle_layout=SfLayout.layout_128x4, enable_pdl=False); print("fi mxfp8_quantize:", desc(o), f"{t(lambda: mxfp8_quantize(x, sf_swizzle_layout=SfLayout.layout_128x4, enable_pdl=False)):.1f}us")
o = norm.rmsnorm(x, w, 1e-6, enable_pdl=False); print("fi rmsnorm:", desc(o), f"{t(lambda: norm.rmsnorm(x, w, 1e-6, enable_pdl=False)):.1f}us")
# QuACK GEMM quant-out
from quack.gemm_interface import gemm
B = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
Bt = torch.randn(N, K, device="cuda", dtype=torch.bfloat16).mT
for fmt in ("nvfp4", "mxfp8"):
    for bname, Bm in (("B(K,N) row-major", B), ("B=(N,K).mT k-major", Bt)):
        try:
            o = gemm(x, Bm, out_dtype=fmt); torch.cuda.synchronize()
            print(f"quack gemm out={fmt} {bname}:", desc(o), f"{t(lambda: gemm(x, Bm, out_dtype=fmt)):.1f}us")
        except Exception as e:
            print(f"quack gemm out={fmt} {bname}: ERROR {type(e).__name__}: {str(e)[:160]}")
print("torch.mm bf16:", f"{t(lambda: x @ B):.1f}us", "| quack gemm bf16 out:", f"{t(lambda: gemm(x, Bt)):.1f}us")
