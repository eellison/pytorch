import sys, contextlib, torch
IMPL = sys.argv[1]; M, K = int(sys.argv[2]), int(sys.argv[3])
sys.argv = ["x", "rmsnorm_nvfp4", str(M), str(K), "ours", "default"]
exec(open("agent_space/peak_cmp/peak_cell.py").read().split("torch.manual_seed(0)")[0])
impl = IMPL
def mxfp8_quant_recip(normed):
    rows, hidden = normed.shape
    g = normed.view(rows, hidden // 32, 32)
    amax = g.abs().float().amax(dim=-1)
    raw = (amax / FP8_MAX).clamp_min(torch.finfo(torch.float32).tiny)
    scale = inductor_prims.cvt_e8m0_rceil(raw)
    q = (g.float() * recip_ue8m0(scale).unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).view(rows, hidden)
    return q, swizzle_scale(scale)
QUANT["mxfp8r"] = mxfp8_quant_recip
torch.manual_seed(0)
x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16); w = torch.randn(K, device="cuda", dtype=torch.bfloat16)
if impl.startswith("ours") or impl.startswith("quant"):
    fmt, mode = impl.split("_")[1], impl.split("_", 2)[2]
    if impl.startswith("quant"):
        fn0 = QUANT[fmt]; fn = lambda x, w: fn0(x)
    else:
        fn = lambda x, w: QUANT[fmt](rms(x, w))
    class _P(InductorChoices):
        @staticmethod
        def should_use_persistent_reduction(*a, **k): return True
    with config.patch({"triton.nested_reduction": True, "triton.cudagraphs": False, "fx_graph_cache": False, "coordinate_descent_tuning": mode.endswith("cd")}), \
         (V.set_choices_handler(_P()) if mode.startswith("persistent") else contextlib.nullcontext()):
        run = torch.compile(fn, fullgraph=True, dynamic=False)
        for _ in range(5): run(x, w)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push("profile"); run(x, w); torch.cuda.nvtx.range_pop(); torch.cuda.synchronize()
else:
    from flashinfer.cute_dsl import rmsnorm_fp4quant
    gs = torch.ones(1, device="cuda", dtype=torch.float32)
    blk, sf = (16, "e4m3") if impl.endswith("nvfp4") else (32, "ue8m0")
    run = lambda: rmsnorm_fp4quant(x, w, global_scale=gs, eps=1e-6, block_size=blk, scale_format=sf, is_sf_swizzled_layout=True, enable_pdl=False)
    for _ in range(5): run()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push("profile"); run(); torch.cuda.nvtx.range_pop(); torch.cuda.synchronize()
print("DONE")
