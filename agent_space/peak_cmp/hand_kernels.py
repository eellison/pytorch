import statistics, sys, torch, triton, triton.language as tl
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
PACK = tl.constexpr('{.reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, $2, $1; cvt.u32.u8 $0, t;}')

@triton.jit
def v0_generated(in_ptr0, in_ptr1, out_ptr2, out_ptr4, xnumel, r0_numel, XBLOCK: tl.constexpr, R0_BLOCK: tl.constexpr):
    G: tl.constexpr = R0_BLOCK // 16
    xindex = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_1 = r0_index; x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 4096 * x0), None, eviction_policy='evict_first').to(tl.float32)
    tmp2 = tmp0 * tmp0
    tmp5 = tl.sum(tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK]), 1)[:, None]
    r0_4 = tl.arange(0, G)[None, :]
    r0_7 = tl.arange(0, R0_BLOCK // 2)[None, :]
    tmp12 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last').to(tl.float32)
    tmp10 = libdevice.rsqrt(tmp5 / 4096.0 + 1e-06)
    tmp14 = tmp0 * tmp10 * tmp12
    tmp17 = tl.reshape(tl_math.abs(tmp14), [XBLOCK, G, 16])
    tmp18 = triton_helpers.max2(tmp17, 2)
    tmp23 = tl.maximum(tmp18 * 0.16666666666666666, 1e-12, tl.PropagateNan.ALL)
    tmp25 = tl.minimum(tmp23, 448.0, tl.PropagateNan.ALL)
    tmp27 = tmp25.to(tl.float8e4nv)
    tmp28, tmp29 = tl.split(tl.reshape(tmp0, [XBLOCK, (R0_BLOCK // 2), 2]))
    tmp30, tmp31 = tl.split(tl.reshape(tmp12, [1, (R0_BLOCK // 2), 2]))
    tmp37 = tmp28 * tmp10 * tmp30
    tmp40 = 1.0 / tmp27.to(tl.float32)
    tmp41 = tl.reshape(tl.broadcast_to(tmp40[:, :, None], [XBLOCK, G, 8]), [XBLOCK, (R0_BLOCK // 2)])
    tmp42 = tmp37 * tmp41
    tmp48 = tmp29 * tmp10 * tmp31
    tmp49 = tmp48 * tmp41
    tmp50 = tl.inline_asm_elementwise(PACK, '=r,f,f', [tmp42, tmp49], dtype=tl.int32, is_pure=True, pack=1)
    tl.store(out_ptr2 + (4 * ((x0 // 32) % 4) + 16 * (x0 % 32) + 512 * (r0_4 // 4) + 32768 * (x0 // 128) + (r0_4 % 4)), tmp27, None)
    tl.store(out_ptr4 + (r0_7 + 2048 * x0), tmp50.to(tl.uint8), None)

@triton.jit
def v1_grouped(in_ptr0, in_ptr1, out_ptr2, out_ptr4, xnumel, r0_numel, XBLOCK: tl.constexpr, R0_BLOCK: tl.constexpr):
    G: tl.constexpr = R0_BLOCK // 16
    xindex = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    x0 = xindex
    x = tl.load(in_ptr0 + (r0_index + 4096 * x0), None, eviction_policy='evict_first').to(tl.float32)
    w = tl.load(in_ptr1 + r0_index, None, eviction_policy='evict_last').to(tl.float32)
    ss = tl.sum(x * x, 1)[:, None]
    rstd = libdevice.rsqrt(ss / 4096.0 + 1e-06)
    y = x * rstd * w
    y4 = tl.reshape(y, [XBLOCK, G, 8, 2])
    amax = tl.max(tl.max(tl_math.abs(y4), 3), 2)  # [XBLOCK, G]
    sc = tl.minimum(tl.maximum(amax * 0.16666666666666666, 1e-12, tl.PropagateNan.ALL), 448.0, tl.PropagateNan.ALL).to(tl.float8e4nv)
    inv = (1.0 / sc.to(tl.float32))[:, :, None]  # [XBLOCK, G, 1]
    ev, od = tl.split(y4)  # [XBLOCK, G, 8]
    packed = tl.inline_asm_elementwise(PACK, '=r,f,f', [ev * inv, od * inv], dtype=tl.int32, is_pure=True, pack=1)
    r0_4 = tl.arange(0, G)[None, :]
    tl.store(out_ptr2 + (4 * ((x0 // 32) % 4) + 16 * (x0 % 32) + 512 * (r0_4 // 4) + 32768 * (x0 // 128) + (r0_4 % 4)), sc, None)
    r0_7 = tl.arange(0, R0_BLOCK // 2)[None, :]
    tl.store(out_ptr4 + (r0_7 + 2048 * x0), tl.reshape(packed, [XBLOCK, R0_BLOCK // 2]).to(tl.uint8), None)

@triton.jit
def v1b_grouped_recompute(in_ptr0, in_ptr1, out_ptr2, out_ptr4, xnumel, r0_numel, XBLOCK: tl.constexpr, R0_BLOCK: tl.constexpr):
    G: tl.constexpr = R0_BLOCK // 16
    xindex = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    x0 = xindex
    x = tl.load(in_ptr0 + (r0_index + 4096 * x0), None, eviction_policy='evict_first').to(tl.float32)
    w = tl.load(in_ptr1 + r0_index, None, eviction_policy='evict_last').to(tl.float32)
    ss = tl.sum(x * x, 1)[:, None]
    rstd = libdevice.rsqrt(ss / 4096.0 + 1e-06)
    y = x * rstd * w
    amax = tl.max(tl.reshape(tl_math.abs(y), [XBLOCK, G, 16]), 2)
    sc = tl.minimum(tl.maximum(amax * 0.16666666666666666, 1e-12, tl.PropagateNan.ALL), 448.0, tl.PropagateNan.ALL).to(tl.float8e4nv)
    inv = (1.0 / sc.to(tl.float32))[:, :, None]
    xe, xo = tl.split(tl.reshape(x, [XBLOCK, G, 8, 2]))
    we, wo = tl.split(tl.reshape(w, [1, G, 8, 2]))
    ev = xe * rstd[:, :, None] * we
    od = xo * rstd[:, :, None] * wo
    packed = tl.inline_asm_elementwise(PACK, '=r,f,f', [ev * inv, od * inv], dtype=tl.int32, is_pure=True, pack=1)
    r0_4 = tl.arange(0, G)[None, :]
    tl.store(out_ptr2 + (4 * ((x0 // 32) % 4) + 16 * (x0 % 32) + 512 * (r0_4 // 4) + 32768 * (x0 // 128) + (r0_4 % 4)), sc, None)
    r0_7 = tl.arange(0, R0_BLOCK // 2)[None, :]
    tl.store(out_ptr4 + (r0_7 + 2048 * x0), tl.reshape(packed, [XBLOCK, R0_BLOCK // 2]).to(tl.uint8), None)

@triton.jit
def v2_split_normed_2d(in_ptr0, in_ptr1, out_ptr2, out_ptr4, xnumel, r0_numel, XBLOCK: tl.constexpr, R0_BLOCK: tl.constexpr):
    G: tl.constexpr = R0_BLOCK // 16
    xindex = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    x0 = xindex
    x = tl.load(in_ptr0 + (r0_index + 4096 * x0), None, eviction_policy='evict_first').to(tl.float32)
    w = tl.load(in_ptr1 + r0_index, None, eviction_policy='evict_last').to(tl.float32)
    ss = tl.sum(x * x, 1)[:, None]
    rstd = libdevice.rsqrt(ss / 4096.0 + 1e-06)
    y = x * rstd * w
    amax = tl.max(tl.reshape(tl_math.abs(y), [XBLOCK, G, 16]), 2)
    sc = tl.minimum(tl.maximum(amax * 0.16666666666666666, 1e-12, tl.PropagateNan.ALL), 448.0, tl.PropagateNan.ALL).to(tl.float8e4nv)
    inv2d = tl.reshape(tl.broadcast_to((1.0 / sc.to(tl.float32))[:, :, None], [XBLOCK, G, 8]), [XBLOCK, R0_BLOCK // 2])
    ev, od = tl.split(tl.reshape(y, [XBLOCK, R0_BLOCK // 2, 2]))
    packed = tl.inline_asm_elementwise(PACK, '=r,f,f', [ev * inv2d, od * inv2d], dtype=tl.int32, is_pure=True, pack=1)
    r0_4 = tl.arange(0, G)[None, :]
    tl.store(out_ptr2 + (4 * ((x0 // 32) % 4) + 16 * (x0 % 32) + 512 * (r0_4 // 4) + 32768 * (x0 // 128) + (r0_4 % 4)), sc, None)
    r0_7 = tl.arange(0, R0_BLOCK // 2)[None, :]
    tl.store(out_ptr4 + (r0_7 + 2048 * x0), packed.to(tl.uint8), None)

def graph_bench(fn, warmup=20, samples=50, calls=100):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(calls): fn()
    g.replay(); torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True); vals = []
    for _ in range(samples):
        s.record(); g.replay(); e.record(); e.synchronize(); vals.append(s.elapsed_time(e) * 1000 / calls)
    return statistics.median(vals)

M, K = 8192, 4096
torch.manual_seed(0)
x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16); w = torch.randn(K, device="cuda", dtype=torch.bfloat16)
outs = {}
for name, kern in (("v0_generated", v0_generated), ("v1_grouped", v1_grouped), ("v1b_grp_recomp", v1b_grouped_recompute), ("v2_split_normed", v2_split_normed_2d)):
    for XBLOCK, nw in ((1, 8), (1, 4), (2, 4), (1, 2), (4, 4)):
        q = torch.empty(M, K // 2, device="cuda", dtype=torch.uint8); sc = torch.empty(M * K // 16, device="cuda", dtype=torch.float8_e4m3fn)
        grid = (M // XBLOCK,)
        try:
            ck = kern[grid](x, w, sc, q, M, K, XBLOCK=XBLOCK, R0_BLOCK=K, num_warps=nw)
            torch.cuda.synchronize()
        except Exception as ex:
            import traceback; print(f"{name} XBLOCK={XBLOCK} nw={nw}: ERROR"); traceback.print_exc(limit=1); print(str(ex)[:1500]); break
        us = graph_bench(lambda: kern[grid](x, w, sc, q, M, K, XBLOCK=XBLOCK, R0_BLOCK=K, num_warps=nw))
        outs[(name, XBLOCK, nw)] = (q.clone(), sc.clone())
        print(f"{name:13s} XBLOCK={XBLOCK} nw={nw}: {us:6.2f}us regs={ck.n_regs} spills={ck.n_spills} smem={ck.metadata.shared}", flush=True)
for k, (q, sc) in outs.items():
    if k[0] != "v0_generated" and ("v0_generated",) + k[1:] in outs:
        ref = outs[("v0_generated",) + k[1:]]
        print(f"  {k[0]} {k[1:]} vs v0 same cfg: quant exact={torch.equal(q, ref[0])} scale exact={torch.equal(sc.view(torch.uint8), ref[1].view(torch.uint8))}")
