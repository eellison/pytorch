
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 256, 'r0_': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'out_ptr3': '*u8', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': False, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_per_fused__fused_rms_norm__to_copy_abs_amax_clamp_div_mul_reciprocal_select_unsqueeze_view_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 1, 'autotune_hints': set(), 'min_rblock': 2, 'backend_hash': '855470BEF4251187CB5023D695885C65615617FA1EB3C786F01CCF7A5DBD6E40', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False}
)
@triton.jit
def triton_per_fused__fused_rms_norm__to_copy_abs_amax_clamp_div_mul_reciprocal_select_unsqueeze_view_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 256
    r0_numel = 16
    R0_BLOCK: tl.constexpr = 16
    nested_R0_LOCAL_REDUCTION_SIZE: tl.constexpr = 16
    nested_R0_REDUCED_BLOCK: tl.constexpr = R0_BLOCK // nested_R0_LOCAL_REDUCTION_SIZE
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 16*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (0))
    tmp3 = tl.broadcast_to(tmp2, [1, 1])
    tmp10 = tl.load(in_ptr2 + (r0_1 + 16*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp4 = tl.full([1, 1], 4096.0, tl.float32)
    tmp5 = (tmp3 / tmp4)
    tmp6 = tl.full([1, 1], 1.1920928955078125e-07, tl.float32)
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp1 * tmp8
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 * tmp11
    tmp13 = tmp12.to(tl.bfloat16)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp14.to(tl.bfloat16)
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp16.to(tl.bfloat16)
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tl_math.abs(tmp18)
    tmp20 = tmp19.to(tl.bfloat16)
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, R0_BLOCK])
    tmp24 = tl.where(r0_mask & xmask, tmp22, float("-inf"))
    tmp25 = triton_helpers.max2(tmp24, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp25, xmask)
    lane2_r0_index = r0_offset // 2 + tl.arange(0, R0_BLOCK // 2)[None, :]
    lane2_r0_index_mask = lane2_r0_index < 8
    r0_2 = lane2_r0_index
    tmp26, tmp27 = tl.split(tl.reshape(tmp0, [XBLOCK, (R0_BLOCK//2), 2]))
    tmp28 = tmp26.to(tl.float32)
    tmp29 = tmp28 * tmp8
    tmp30, tmp31 = tl.split(tl.reshape(tmp10, [XBLOCK, (R0_BLOCK//2), 2]))
    tmp32 = tmp30.to(tl.float32)
    tmp33 = tmp29 * tmp32
    tmp34 = tmp33.to(tl.bfloat16)
    tmp35 = tmp34.to(tl.float32)
    tmp36 = tmp35.to(tl.bfloat16)
    tmp37 = tmp36.to(tl.float32)
    tmp38 = tmp37.to(tl.bfloat16)
    tmp39 = tmp38.to(tl.float32)
    tmp40 = tmp39.to(tl.float32)
    tmp41 = tmp25.to(tl.bfloat16)
    tmp42 = tmp41.to(tl.float32)
    tmp43 = tl.full([1, 1], 0.16666666666666666, tl.float32)
    tmp44 = tmp42 * tmp43
    tmp45 = tmp44.to(tl.bfloat16)
    tmp46 = tmp45.to(tl.float32)
    tmp47 = tmp46.to(tl.float32)
    tmp48 = tl.full([1, 1], 1e-12, tl.float32)
    tmp49 = tl.maximum(tmp47, tmp48, tl.PropagateNan.ALL)
    tmp50 = tl.full([1, 1], 448.0, tl.float32)
    tmp51 = tl.minimum(tmp49, tmp50, tl.PropagateNan.ALL)
    tmp52 = tmp51.to(tl.bfloat16)
    tmp53 = tmp52.to(tl.float32)
    tmp54 = tmp53.to(tl.bfloat16)
    tmp55 = tmp54.to(tl.float32)
    tmp56 = tmp55.to(tl.bfloat16)
    tmp57 = tmp56.to(tl.float32)
    tmp58 = tmp57.to(tl.float8e4nv)
    tmp59 = tmp58.to(tl.float32)
    tmp60 = tl.full([1, 1], 1.0, tl.float32)
    tmp61 = (tmp60 / tmp59)
    tmp62 = tmp40 * tmp61
    tmp63 = tmp27.to(tl.float32)
    tmp64 = tmp63 * tmp8
    tmp65 = tmp31.to(tl.float32)
    tmp66 = tmp64 * tmp65
    tmp67 = tmp66.to(tl.bfloat16)
    tmp68 = tmp67.to(tl.float32)
    tmp69 = tmp68.to(tl.bfloat16)
    tmp70 = tmp69.to(tl.float32)
    tmp71 = tmp70.to(tl.bfloat16)
    tmp72 = tmp71.to(tl.float32)
    tmp73 = tmp72.to(tl.float32)
    tmp74 = tmp73 * tmp61
    tmp75 = tl.inline_asm_elementwise('{.reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, $2, $1; cvt.u32.u8 $0, t;}', '=r,f,f', [tmp62, tmp74], dtype=tl.int32, is_pure=True, pack=1)
    tmp76 = tmp75.to(tl.uint8)
    tl.store(out_ptr3 + (r0_2 + 8*x0), tmp76, lane2_r0_index_mask & xmask)
