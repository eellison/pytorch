import sys, statistics, torch, triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
RECIP = gl.constexpr("{.reg .pred p_zero; .reg .s32 neg_exp; .reg .f32 neg_exp_f, result; setp.eq.u32 p_zero, $1, 0; sub.s32 neg_exp, 127, $1; cvt.rn.f32.s32 neg_exp_f, neg_exp; ex2.approx.f32 result, neg_exp_f; selp.f32 $0, 0f00000000, result, p_zero;}")
E8M0 = gl.constexpr("cvt.rp.satfinite.ue8m0x2.f32 $0, 0.0, $1;")

@gluon.jit
def rms_colwise_band(x_ptr, w_ptr, q_ptr, sf_ptr, K: gl.constexpr, RPT: gl.constexpr, CPT: gl.constexpr, TPW_R: gl.constexpr, NUM_WARPS: gl.constexpr):
    # One 32-row band per program. Tile per K-step: [32, KC] with each thread owning RPT rows x CPT cols.
    TPW_C: gl.constexpr = 32 // TPW_R
    KC: gl.constexpr = CPT * TPW_C * NUM_WARPS
    gl.static_assert(RPT * TPW_R == 32, "threads must cover the 32-row band")
    L: gl.constexpr = gl.BlockedLayout(size_per_thread=[RPT, CPT], threads_per_warp=[TPW_R, TPW_C], warps_per_cta=[1, NUM_WARPS], order=[1, 0])
    band = gl.program_id(0)
    r = gl.arange(0, 32, layout=gl.SliceLayout(1, L))
    c = gl.arange(0, KC, layout=gl.SliceLayout(0, L))
    rows = band * 32 + gl.expand_dims(r, 1)            # [32, 1]
    cols = gl.expand_dims(c, 0)                          # [1, KC]
    # pass 1: per-row sum of squares over K
    acc = gl.zeros([32, KC], gl.float32, L)
    for k0 in range(0, K, KC):
        xt = gl.load(x_ptr + rows * K + (k0 + cols)).to(gl.float32)
        acc += xt * xt
    ss = gl.sum(acc, axis=1)                             # [32], layout SliceLayout(1, L)
    rstd = gl.inline_asm_elementwise("rsqrt.approx.f32 $0, $1;", "=f,f", [ss * (1.0 / K) + 1e-6], dtype=gl.float32, is_pure=True, pack=1)
    rstd2 = gl.expand_dims(rstd, 1)                      # [32, 1]
    # pass 2: normalize, 32-row group amax per column, scale, quantize, store
    for k0 in range(0, K, KC):
        xt = gl.load(x_ptr + rows * K + (k0 + cols)).to(gl.float32)
        wt = gl.load(w_ptr + (k0 + cols)).to(gl.float32)  # [1, KC]
        y = xt * rstd2 * wt
        amax = gl.max(gl.maximum(y, -y), axis=0)         # [KC], layout SliceLayout(0, L)
        raw = gl.maximum(amax * (1.0 / 448.0), 1.1754943508222875e-38)
        sf = gl.inline_asm_elementwise(E8M0, "=h,r", [raw], dtype=gl.uint16, is_pure=True, pack=1).to(gl.uint8)
        inv = gl.inline_asm_elementwise(RECIP, "=f,r", [sf.to(gl.int32)], dtype=gl.float32, is_pure=True, pack=1)
        q = y * gl.expand_dims(inv, 0)
        q = gl.minimum(gl.maximum(q, -448.0), 448.0)
        gl.store(q_ptr + rows * K + (k0 + cols), q.to(gl.float8e4nv))
        gl.store(sf_ptr + band * K + (k0 + c), sf)       # row-major (M/32, K) scales

def graph_bench(fn, warmup=10, samples=30, calls=50):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        for _ in range(calls): fn()
    gr.replay(); torch.cuda.synchronize(); s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True); vals = []
    for _ in range(samples):
        s.record(); gr.replay(); e.record(); e.synchronize(); vals.append(s.elapsed_time(e) * 1000 / calls)
    return statistics.median(vals)

M, K = int(sys.argv[1]), int(sys.argv[2])
torch.manual_seed(0)
x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16); w = torch.randn(K, device="cuda", dtype=torch.bfloat16)
def ref_fn(x, w):
    y = torch.nn.functional.rms_norm(x.float(), (K,), w.float(), eps=1e-6)
    g = y.view(M // 32, 32, K); amax = g.abs().amax(1); raw = (amax / 448.0).clamp_min(torch.finfo(torch.float32).tiny)
    e = torch.ceil(torch.log2(raw)).clamp(-127, 127); sf = (e + 127).to(torch.uint8)
    q = (g * torch.ldexp(torch.ones_like(e), -e).unsqueeze(1)).clamp(-448, 448).to(torch.float8_e4m3fn).view(M, K)
    return q, sf
q_ref, sf_ref = ref_fn(x, w)
for RPT, CPT, TPW_R, NW in ((8, 8, 4, 4), (8, 8, 4, 8), (4, 8, 8, 4), (16, 8, 2, 4), (8, 4, 4, 8), (8, 16, 4, 2)):
    q = torch.empty(M, K, device="cuda", dtype=torch.float8_e4m3fn); sf = torch.empty(M // 32, K, device="cuda", dtype=torch.uint8)
    try:
        ck = rms_colwise_band[(M // 32,)](x, w, q, sf, K=K, RPT=RPT, CPT=CPT, TPW_R=TPW_R, NUM_WARPS=NW, num_warps=NW); torch.cuda.synchronize()
    except Exception as ex:
        print(f"RPT={RPT} CPT={CPT} TPW_R={TPW_R} NW={NW}: ERROR {str(ex)[:250]}"); continue
    us = graph_bench(lambda: rms_colwise_band[(M // 32,)](x, w, q, sf, K=K, RPT=RPT, CPT=CPT, TPW_R=TPW_R, NUM_WARPS=NW, num_warps=NW))
    sf_mis = int((sf != sf_ref).sum()); q_mis = int((q.view(torch.uint8) != q_ref.view(torch.uint8)).sum())
    print(f"{M}x{K} band RPT={RPT} CPT={CPT} TPW_R={TPW_R} NW={NW} (KC={CPT*(32//TPW_R)*NW}): {us:7.1f}us regs={ck.n_regs} spills={ck.n_spills} | sf mismatches={sf_mis} q mismatches={q_mis} ({100*q_mis/q_ref.numel():.5f}%)", flush=True)
