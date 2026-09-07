
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr2': '*fp8e4nv', 'out_ptr5': '*u8', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': False, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__fused_rms_norm_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 2, 'autotune_hints': set(), 'tiling_scores': {'x': 1032, 'r0_': 1065024}, 'min_rblock': 16, 'backend_hash': '855470BEF4251187CB5023D695885C65615617FA1EB3C786F01CCF7A5DBD6E40', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False}
)
@triton.jit
def triton_red_fused__fused_rms_norm_1(in_ptr0, in_ptr1, out_ptr2, out_ptr5, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
        tmp17 = tmp16.to(tl.bfloat16)
        tmp18 = tmp17.to(tl.float32)
        tmp19 = tmp18.to(tl.bfloat16)
        tmp20 = tmp19.to(tl.float32)
        tmp21 = tmp20.to(tl.bfloat16)
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tl_math.abs(tmp22)
        tmp24 = tmp23.to(tl.bfloat16)
        tmp25 = tmp24.to(tl.float32)
        tmp26 = tl.reshape(tmp25, [XBLOCK, nested_R0_REDUCED_BLOCK, nested_R0_LOCAL_REDUCTION_SIZE])
        tmp27 = triton_helpers.max2(tmp26, 2)
        tmp28 = tmp27.to(tl.bfloat16)
        tmp29 = tmp28.to(tl.float32)
        tmp30 = tl.full([1, 1], 0.16666666666666666, tl.float32)
        tmp31 = tmp29 * tmp30
        tmp32 = tmp31.to(tl.bfloat16)
        tmp33 = tmp32.to(tl.float32)
        tmp34 = tmp33.to(tl.float32)
        tmp35 = tl.full([1, 1], 1e-12, tl.float32)
        tmp36 = tl.maximum(tmp34, tmp35, tl.PropagateNan.ALL)
        tmp37 = tl.full([1, 1], 448.0, tl.float32)
        tmp38 = tl.minimum(tmp36, tmp37, tl.PropagateNan.ALL)
        tmp39 = tmp38.to(tl.bfloat16)
        tmp40 = tmp39.to(tl.float32)
        tmp41 = tmp40.to(tl.bfloat16)
        tmp42 = tmp41.to(tl.float32)
        tmp43 = tmp42.to(tl.bfloat16)
        tmp44 = tmp43.to(tl.float32)
        tmp45 = tmp44.to(tl.float8e4nv)
        tmp46, tmp47 = tl.split(tl.reshape(tmp6, [XBLOCK, (R0_BLOCK//2), 2]))
        tmp48, tmp49 = tl.split(tl.reshape(tmp14, [1, (R0_BLOCK//2), 2]))
        tmp50 = tmp46.to(tl.float32)
        tmp51 = tmp50 * tmp12
        tmp52 = tmp48.to(tl.float32)
        tmp53 = tmp51 * tmp52
        tmp54 = tmp53.to(tl.bfloat16)
        tmp55 = tmp54.to(tl.float32)
        tmp56 = tmp55.to(tl.bfloat16)
        tmp57 = tmp56.to(tl.float32)
        tmp58 = tmp57.to(tl.bfloat16)
        tmp59 = tmp58.to(tl.float32)
        tmp60 = tmp59.to(tl.float32)
        tmp61 = tmp45.to(tl.float32)
        tmp62 = tl.full([1, 1], 1.0, tl.float32)
        tmp63 = (tmp62 / tmp61)
        tmp64 = tl.reshape(tl.broadcast_to(tmp63[:, :, None], [XBLOCK, nested_R0_REDUCED_BLOCK, (nested_R0_LOCAL_REDUCTION_SIZE//2)]), [XBLOCK, (R0_BLOCK//2)])
        tmp65 = tmp60 * tmp64
        tmp66 = tmp47.to(tl.float32)
        tmp67 = tmp66 * tmp12
        tmp68 = tmp49.to(tl.float32)
        tmp69 = tmp67 * tmp68
        tmp70 = tmp69.to(tl.bfloat16)
        tmp71 = tmp70.to(tl.float32)
        tmp72 = tmp71.to(tl.bfloat16)
        tmp73 = tmp72.to(tl.float32)
        tmp74 = tmp73.to(tl.bfloat16)
        tmp75 = tmp74.to(tl.float32)
        tmp76 = tmp75.to(tl.float32)
        tmp77 = tmp76 * tmp64
        tmp78 = tl.inline_asm_elementwise('{.reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, $2, $1; cvt.u32.u8 $0, t;}', '=r,f,f', [tmp65, tmp77], dtype=tl.int32, is_pure=True, pack=1)
        tmp79 = tmp78.to(tl.uint8)
        tl.store(out_ptr2 + (4*(((x0 // 32) % 4)) + 16*((x0 % 32)) + 512*(r0_4 // 4) + 33280*(x0 // 128) + ((r0_4 % 4))), tmp45, reduced_r0_index_mask & xmask)
        tl.store(out_ptr5 + (r0_7 + 2064*x0), tmp79, lane2_r0_index_mask & xmask)
