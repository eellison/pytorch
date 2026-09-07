
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp8e4nv', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': False, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_put_add_arange_clamp_div_fill_floor_divide_mul_remainder_unsqueeze_3', 'mutated_arg_names': ['out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 1024}, 'backend_hash': '855470BEF4251187CB5023D695885C65615617FA1EB3C786F01CCF7A5DBD6E40', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False},
    min_elem_per_thread=2
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_put_add_arange_clamp_div_fill_floor_divide_mul_remainder_unsqueeze_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.bfloat16)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tl.full([1], 0.16666666666666666, tl.float32)
    tmp4 = tmp2 * tmp3
    tmp5 = tmp4.to(tl.bfloat16)
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tl.full([1], 1e-12, tl.float32)
    tmp9 = tl.maximum(tmp7, tmp8, tl.PropagateNan.ALL)
    tmp10 = tl.full([1], 448.0, tl.float32)
    tmp11 = tl.minimum(tmp9, tmp10, tl.PropagateNan.ALL)
    tmp12 = tmp11.to(tl.bfloat16)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp13.to(tl.bfloat16)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp15.to(tl.bfloat16)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp17.to(tl.float8e4nv)
    tl.store(out_ptr0 + (512*(x0 // 4) + ((x0 % 4))), tmp18, xmask)
