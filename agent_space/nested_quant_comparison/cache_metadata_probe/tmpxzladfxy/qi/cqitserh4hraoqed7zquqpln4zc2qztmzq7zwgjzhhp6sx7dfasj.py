
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4096, 'r0_': 32},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'out_ptr2': '*u8', 'out_ptr3': '*fp8e4nv', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_per_fused__fused_rms_norm__to_copy__unsafe_index_put_abs_add_amax_arange_clamp_clamp_min_div_floor_divide_inductor_cvt_e8m0_rceil_ldexp_mul_ones_like_remainder_sub_unsqueeze_view_1', 'mutated_arg_names': ['out_ptr2'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 1, 'autotune_hints': set(), 'tiling_scores': {'x': 4864, 'r0_': 319488}, 'backend_hash': 'B49614C0BBA23CA71245E046F2A6ABFCFD211E8FB0FFF5563191062F952B4CD3', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_per_fused__fused_rms_norm__to_copy__unsafe_index_put_abs_add_amax_arange_clamp_clamp_min_div_floor_divide_inductor_cvt_e8m0_rceil_ldexp_mul_ones_like_remainder_sub_unsqueeze_view_1(in_ptr0, in_ptr1, in_ptr2, out_ptr2, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 2432
    r0_numel = 32
    R0_BLOCK: tl.constexpr = 32
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x3 = xindex
    x1 = xindex // 128
    x0 = (xindex % 128)
    tmp0 = tl.load(in_ptr0 + (r0_2 + 32*x3), xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (r0_2 + 32*x0), xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tl.full([1, 1], 4096.0, tl.float32)
    tmp4 = (tmp2 / tmp3)
    tmp5 = tl.full([1, 1], 1.1920928955078125e-07, tl.float32)
    tmp6 = tmp4 + tmp5
    tmp7 = libdevice.rsqrt(tmp6)
    tmp8 = tmp1 * tmp7
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp8 * tmp10
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tl_math.abs(tmp12)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
    tmp17 = tl.where(xmask, tmp15, float("-inf"))
    tmp18 = triton_helpers.max2(tmp17, 1)[:, None].to(tl.float32)
    tmp19 = tl.full([1, 1], 0.002232142857142857, tl.float32)
    tmp20 = tmp18 * tmp19
    tmp21 = tl.full([1, 1], 1.1754943508222875e-38, tl.float32)
    tmp22 = tl.maximum(tmp20, tmp21, tl.PropagateNan.ALL)
    tmp23 = tl.inline_asm_elementwise('cvt.rp.satfinite.ue8m0x2.f32 $0, 0.0, $1;', '=h,r', [tmp22], dtype=tl.uint16, is_pure=True, pack=1)
    tmp24 = tmp23.to(tl.int16).to(tl.uint8)
    tmp25 = tmp12.to(tl.float32)
    tmp26 = tmp24.to(tl.int32)
    tmp27 = tl.full([1, 1], 127, tl.int32)
    tmp28 = tmp26 - tmp27
    tmp29 = tl.full([1, 1], 1.0, tl.float32)
    tmp30 = libdevice.ldexp(tmp29, tmp28.to(tl.int32))
    tmp31 = (tmp25 / tmp30)
    tmp32 = tl.full([1, 1], -448.0, tl.float32)
    tmp33 = tl.maximum(tmp31, tmp32, tl.PropagateNan.ALL)
    tmp34 = tl.full([1, 1], 448.0, tl.float32)
    tmp35 = tl.minimum(tmp33, tmp34, tl.PropagateNan.ALL)
    tmp36 = tmp35.to(tl.float8e4nv)
    tl.store(out_ptr2 + (16*x1 + 512*(x0 // 4) + ((x0 % 4))), tmp24, xmask)
    tl.store(out_ptr3 + (r0_2 + 32*x3), tmp36, xmask)
