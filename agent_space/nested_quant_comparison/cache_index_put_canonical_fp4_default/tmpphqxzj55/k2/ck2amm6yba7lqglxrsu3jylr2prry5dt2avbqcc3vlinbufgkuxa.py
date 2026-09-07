
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 256, 'r0_': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr1': '*bf16', 'out_ptr3': '*u8', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__fused_rms_norm_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 2, 'autotune_hints': set(), 'tiling_scores': {'x': 1032, 'r0_': 1065024}, 'min_rblock': 16, 'backend_hash': '855470BEF4251187CB5023D695885C65615617FA1EB3C786F01CCF7A5DBD6E40', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused__fused_rms_norm_0(in_ptr0, in_ptr1, out_ptr1, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 129
    r0_numel = 4128
    nested_R0_LOCAL_REDUCTION_SIZE: tl.constexpr = 16
    nested_R0_REDUCED_BLOCK: tl.constexpr = R0_BLOCK // nested_R0_LOCAL_REDUCTION_SIZE
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp4 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 4128*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tmp1 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(r0_mask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        reduced_r0_index = r0_offset // nested_R0_LOCAL_REDUCTION_SIZE + tl.arange(0, nested_R0_REDUCED_BLOCK)[None, :]
        reduced_r0_index_mask = reduced_r0_index < 258
        r0_4 = reduced_r0_index
        lane2_r0_index = r0_offset // 2 + tl.arange(0, R0_BLOCK // 2)[None, :]
        lane2_r0_index_mask = lane2_r0_index < 2064
        r0_7 = lane2_r0_index
        tmp6 = tl.load(in_ptr0 + (r0_1 + 4128*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp14 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tl.full([1, 1], 4128.0, tl.float32)
        tmp9 = (tmp4 / tmp8)
        tmp10 = tl.full([1, 1], 1.1920928955078125e-07, tl.float32)
        tmp11 = tmp9 + tmp10
        tmp12 = libdevice.rsqrt(tmp11)
        tmp13 = tmp7 * tmp12
        tmp15 = tmp14.to(tl.float32)
        tmp16 = tmp13 * tmp15
        tmp17 = tmp16.to(tl.float32)
        tmp18 = tl_math.abs(tmp17)
        tmp19 = tl.reshape(tmp18, [XBLOCK, nested_R0_REDUCED_BLOCK, nested_R0_LOCAL_REDUCTION_SIZE])
        tmp20 = triton_helpers.max2(tmp19, 2)
        tmp21, tmp22 = tl.split(tl.reshape(tmp6, [XBLOCK, (R0_BLOCK//2), 2]))
        tmp23, tmp24 = tl.split(tl.reshape(tmp14, [1, (R0_BLOCK//2), 2]))
        tmp25 = tmp21.to(tl.float32)
        tmp26 = tmp25 * tmp12
        tmp27 = tmp23.to(tl.float32)
        tmp28 = tmp26 * tmp27
        tmp29 = tmp28.to(tl.float32)
        tmp30 = tmp29.to(tl.float32)
        tmp31 = tl.full([1, 1], 0.16666666666666666, tl.float32)
        tmp32 = tmp20 * tmp31
        tmp33 = tmp32.to(tl.float32)
        tmp34 = tl.full([1, 1], 1e-12, tl.float32)
        tmp35 = tl.maximum(tmp33, tmp34, tl.PropagateNan.ALL)
        tmp36 = tl.full([1, 1], 448.0, tl.float32)
        tmp37 = tl.minimum(tmp35, tmp36, tl.PropagateNan.ALL)
        tmp38 = tmp37.to(tl.float32)
        tmp39 = tmp38.to(tl.float8e4nv)
        tmp40 = tmp39.to(tl.float32)
        tmp41 = tl.full([1, 1], 1.0, tl.float32)
        tmp42 = (tmp41 / tmp40)
        tmp43 = tl.reshape(tl.broadcast_to(tmp42[:, :, None], [XBLOCK, nested_R0_REDUCED_BLOCK, (nested_R0_LOCAL_REDUCTION_SIZE//2)]), [XBLOCK, (R0_BLOCK//2)])
        tmp44 = tmp30 * tmp43
        tmp45 = tmp22.to(tl.float32)
        tmp46 = tmp45 * tmp12
        tmp47 = tmp24.to(tl.float32)
        tmp48 = tmp46 * tmp47
        tmp49 = tmp48.to(tl.float32)
        tmp50 = tmp49.to(tl.float32)
        tmp51 = tmp50 * tmp43
        tmp52 = tl.inline_asm_elementwise('{.reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, $2, $1; cvt.u32.u8 $0, t;}', '=r,f,f', [tmp44, tmp51], dtype=tl.int32, is_pure=True, pack=1)
        tmp53 = tmp52.to(tl.uint8)
        tl.store(out_ptr1 + (r0_4 + 258*x0), tmp20, reduced_r0_index_mask & xmask)
        tl.store(out_ptr3 + (r0_7 + 2064*x0), tmp53, lane2_r0_index_mask & xmask)
