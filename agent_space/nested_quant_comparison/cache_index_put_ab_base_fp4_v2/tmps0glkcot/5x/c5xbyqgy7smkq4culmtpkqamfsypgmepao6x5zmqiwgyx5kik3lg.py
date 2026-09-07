
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp8e4nv', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__to_copy__unsafe_view_clamp_clone_constant_pad_nd_div_permute_transpose_view_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 199684}, 'backend_hash': '855470BEF4251187CB5023D695885C65615617FA1EB3C786F01CCF7A5DBD6E40', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=2
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_view_clamp_clone_constant_pad_nd_div_permute_transpose_view_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 66560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 4) % 4)
    x2 = ((xindex // 16) % 32)
    x3 = xindex // 512
    x0 = (xindex % 4)
    x5 = xindex
    tmp0 = (x2 + 32*x1 + 128*(x3 // 65)).to(tl.int32)
    tmp1 = tl.full([1], 129, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = (x0 + 4*((x3 % 65))).to(tl.int32)
    tmp4 = tl.full([1], 258, tl.int64)
    tmp5 = tmp3 < tmp4
    tmp6 = tmp2 & tmp5
    tmp7 = tl.load(in_ptr0 + (x0 + 4*((x3 % 65)) + 258*x2 + 8256*x1 + 33024*(x3 // 65)), tmp6 & xmask, other=0.0).to(tl.float32)
    tmp8 = tl.full([1], 0.16666666666666666, tl.float32)
    tmp9 = tmp7 * tmp8
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tl.full([1], 1e-12, tl.float32)
    tmp12 = tl.maximum(tmp10, tmp11, tl.PropagateNan.ALL)
    tmp13 = tl.full([1], 448.0, tl.float32)
    tmp14 = tl.minimum(tmp12, tmp13, tl.PropagateNan.ALL)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp15.to(tl.float8e4nv)
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp6, tmp16, tmp17)
    tl.store(out_ptr0 + (x5), tmp18, xmask)
