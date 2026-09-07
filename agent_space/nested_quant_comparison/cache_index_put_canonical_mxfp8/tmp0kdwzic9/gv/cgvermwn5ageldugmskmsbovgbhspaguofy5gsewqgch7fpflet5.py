# AOT ID: ['38_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._C._dynamo.guards import copy_if_misaligned
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_size_stride_grouped = torch._C._dynamo.guards.assert_size_stride_grouped
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /data/users/eellison/pytorch/agent_space/nested_quant_comparison/cache_index_put_canonical_mxfp8/tmp0kdwzic9/qg/cqgtp2clwpx62tmvkfdlhih3heezystsym6oalyzelff7kofmfo6.py
# Topologically Sorted Source Nodes: [normed], Original ATen: [aten._fused_rms_norm]
# Source node to ATen node mapping:
#   normed => convert_element_type, mean, pow_1
# Graph fragment:
#   %arg0_1 : Tensor "bf16[129, 4128][4128, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %convert_element_type : Tensor "f32[129, 4128][4128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.float32), kwargs = {})
#   %pow_1 : Tensor "f32[129, 4128][4128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type, 2), kwargs = {})
#   %mean : Tensor "f32[129, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [1], True), kwargs = {})
#   return %buf0
triton_red_fused__fused_rms_norm_0 = async_compile.triton('triton_red_fused__fused_rms_norm_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr3': '*u8', 'out_ptr4': '*fp8e4nv', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': False, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__fused_rms_norm_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 2, 'autotune_hints': set(), 'tiling_scores': {'x': 1032, 'r0_': 1065024}, 'min_rblock': 32, 'backend_hash': '855470BEF4251187CB5023D695885C65615617FA1EB3C786F01CCF7A5DBD6E40', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False}
)
@triton.jit
def triton_red_fused__fused_rms_norm_0(in_ptr0, in_ptr1, out_ptr3, out_ptr4, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 129
    r0_numel = 4128
    nested_R0_LOCAL_REDUCTION_SIZE: tl.constexpr = 32
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
        reduced_r0_index_mask = reduced_r0_index < 129
        r0_4 = reduced_r0_index
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
        tmp26 = tmp25.to(tl.bfloat16)
        tmp27 = tmp26.to(tl.float32)
        tmp28 = tmp27.to(tl.float32)
        tmp29 = tl.reshape(tmp28, [XBLOCK, nested_R0_REDUCED_BLOCK, nested_R0_LOCAL_REDUCTION_SIZE])
        tmp30 = triton_helpers.max2(tmp29, 2)
        tmp31 = tl.full([1, 1], 0.002232142857142857, tl.float32)
        tmp32 = tmp30 * tmp31
        tmp33 = tl.full([1, 1], 1.1754943508222875e-38, tl.float32)
        tmp34 = tl.maximum(tmp32, tmp33, tl.PropagateNan.ALL)
        tmp35 = tl.inline_asm_elementwise('cvt.rp.satfinite.ue8m0x2.f32 $0, 0.0, $1;', '=h,r', [tmp34], dtype=tl.uint16, is_pure=True, pack=1)
        tmp36 = tmp35.to(tl.int16).to(tl.uint8)
        tmp37 = tmp22.to(tl.float32)
        tmp38 = tl.reshape(tl.broadcast_to(tmp36[:, :, None], [XBLOCK, nested_R0_REDUCED_BLOCK, nested_R0_LOCAL_REDUCTION_SIZE]), [XBLOCK, R0_BLOCK])
        tmp39 = tmp38.to(tl.int32)
        tmp40 = tl.full([1, 1], 127, tl.int32)
        tmp41 = tmp39 - tmp40
        tmp42 = tl.full([1, 1], 1.0, tl.float32)
        tmp43 = libdevice.ldexp(tmp42, tmp41.to(tl.int32))
        tmp44 = (tmp37 / tmp43)
        tmp45 = tl.full([1, 1], -448.0, tl.float32)
        tmp46 = tl.maximum(tmp44, tmp45, tl.PropagateNan.ALL)
        tmp47 = tl.full([1, 1], 448.0, tl.float32)
        tmp48 = tl.minimum(tmp46, tmp47, tl.PropagateNan.ALL)
        tmp49 = tmp48.to(tl.float8e4nv)
        tl.store(out_ptr3 + (4*(((x0 // 32) % 4)) + 16*((x0 % 32)) + 512*(r0_4 // 4) + 16896*(x0 // 128) + ((r0_4 % 4))), tmp36, reduced_r0_index_mask & xmask)
        tl.store(out_ptr4 + (r0_1 + 4128*x0), tmp49, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /data/users/eellison/pytorch/agent_space/nested_quant_comparison/cache_index_put_canonical_mxfp8/tmp0kdwzic9/7u/c7ucnr5i6rupbqnxg7hyjj6fz7jvfqgz2vvpw6trjmsd6o73vnxs.py
# Topologically Sorted Source Nodes: [arange_2, r_1, floordiv_3, mul_5, mul_6, arange_3, c_1, floordiv_4, mul_7, add_4, mod_3, mul_8, add_5, floordiv_5, mod_4, mul_9, add_6, mod_5, pad_offset, zeros, out_2], Original ATen: [aten.arange, aten.unsqueeze, aten.floor_divide, aten.mul, aten.add, aten.remainder, aten.zeros, aten._unsafe_index_put]
# Source node to ATen node mapping:
#   add_4 => add_5
#   add_5 => add_6
#   add_6 => add_7
#   arange_2 => iota_2
#   arange_3 => iota_3
#   c_1 => unsqueeze_4
#   floordiv_3 => div_5
#   floordiv_4 => div_6
#   floordiv_5 => div_7
#   mod_3 => remainder_3
#   mod_4 => remainder_4
#   mod_5 => remainder_5
#   mul_5 => mul_7
#   mul_6 => mul_8
#   mul_7 => mul_9
#   mul_8 => mul_10
#   mul_9 => mul_11
#   out_2 => _unsafe_index_put_1
#   pad_offset => add_8
#   r_1 => unsqueeze_3
#   zeros => full_1
# Graph fragment:
#   %buf5 : Tensor "u8[33792][1]cuda:0" = PlaceHolder[target=buf5]
#   %iota_2 : Tensor "i64[127][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (127,), kwargs = {start: 129, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %unsqueeze_3 : Tensor "i64[127, 1][1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota_2, 1), kwargs = {})
#   %div_5 : Tensor "i64[127, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor_mode](args = (%unsqueeze_3, 128), kwargs = {rounding_mode: floor})
#   %mul_7 : Tensor "i64[127, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_5, 33), kwargs = {})
#   %mul_8 : Tensor "i64[127, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, 512), kwargs = {})
#   %iota_3 : Tensor "i64[132][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (132,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %unsqueeze_4 : Tensor "i64[1, 132][132, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota_3, 0), kwargs = {})
#   %div_6 : Tensor "i64[1, 132][132, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor_mode](args = (%unsqueeze_4, 4), kwargs = {rounding_mode: floor})
#   %mul_9 : Tensor "i64[1, 132][132, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_6, 512), kwargs = {})
#   %add_5 : Tensor "i64[127, 132][132, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %mul_9), kwargs = {})
#   %remainder_3 : Tensor "i64[127, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.remainder.Scalar](args = (%unsqueeze_3, 32), kwargs = {})
#   %mul_10 : Tensor "i64[127, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%remainder_3, 16), kwargs = {})
#   %add_6 : Tensor "i64[127, 132][132, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_5, %mul_10), kwargs = {})
#   %div_7 : Tensor "i64[127, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor_mode](args = (%unsqueeze_3, 32), kwargs = {rounding_mode: floor})
#   %remainder_4 : Tensor "i64[127, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.remainder.Scalar](args = (%div_7, 4), kwargs = {})
#   %mul_11 : Tensor "i64[127, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%remainder_4, 4), kwargs = {})
#   %add_7 : Tensor "i64[127, 132][132, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_6, %mul_11), kwargs = {})
#   %remainder_5 : Tensor "i64[1, 132][132, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.remainder.Scalar](args = (%unsqueeze_4, 4), kwargs = {})
#   %add_8 : Tensor "i64[127, 132][132, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_7, %remainder_5), kwargs = {})
#   %full_1 : Tensor "u8[127, 132][132, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([127, 132], 0), kwargs = {dtype: torch.uint8, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %_unsafe_index_put_1 : Tensor "u8[33792][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims._unsafe_index_put_.default](args = (%_unsafe_index_put, [%add_8], %full_1), kwargs = {})
#   return %buf6
triton_poi_fused__unsafe_index_put_add_arange_floor_divide_mul_remainder_unsqueeze_zeros_1 = async_compile.triton('triton_poi_fused__unsafe_index_put_add_arange_floor_divide_mul_remainder_unsqueeze_zeros_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*u8', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': False, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__unsafe_index_put_add_arange_floor_divide_mul_remainder_unsqueeze_zeros_1', 'mutated_arg_names': ['out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 33528}, 'backend_hash': '855470BEF4251187CB5023D695885C65615617FA1EB3C786F01CCF7A5DBD6E40', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_put_add_arange_floor_divide_mul_remainder_unsqueeze_zeros_1(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16764
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 132)
    x1 = xindex // 132
    tmp0 = tl.full([1], 0, tl.uint8)
    tl.store(out_ptr0 + (4*((((129 + x1) // 32) % 4)) + 16*(((129 + x1) % 32)) + 512*(x0 // 4) + 16896*((129 + x1) // 128) + ((x0 % 4))), tmp0, xmask)
''', device_str='cuda')


# kernel path: /data/users/eellison/pytorch/agent_space/nested_quant_comparison/cache_index_put_canonical_mxfp8/tmp0kdwzic9/6e/c6e3csscnke4kblxdu6ssemell6dcdn5ypkcderwykqnznblc7fm.py
# Topologically Sorted Source Nodes: [arange_4, r_2, floordiv_6, mul_10, mul_11, arange_5, c_2, floordiv_7, mul_12, add_8, mod_6, mul_13, add_9, floordiv_8, mod_7, mul_14, add_10, mod_8, pad_offset_1, zeros_2, out_3], Original ATen: [aten.arange, aten.unsqueeze, aten.floor_divide, aten.mul, aten.add, aten.remainder, aten.zeros, aten._unsafe_index_put]
# Source node to ATen node mapping:
#   add_10 => add_11
#   add_8 => add_9
#   add_9 => add_10
#   arange_4 => iota_4
#   arange_5 => iota_5
#   c_2 => unsqueeze_6
#   floordiv_6 => div_8
#   floordiv_7 => div_9
#   floordiv_8 => div_10
#   mod_6 => remainder_6
#   mod_7 => remainder_7
#   mod_8 => remainder_8
#   mul_10 => mul_12
#   mul_11 => mul_13
#   mul_12 => mul_14
#   mul_13 => mul_15
#   mul_14 => mul_16
#   out_3 => _unsafe_index_put_2
#   pad_offset_1 => add_12
#   r_2 => unsqueeze_5
#   zeros_2 => full_2
# Graph fragment:
#   %buf6 : Tensor "u8[33792][1]cuda:0" = PlaceHolder[target=buf6]
#   %iota_4 : Tensor "i64[129][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (129,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %unsqueeze_5 : Tensor "i64[129, 1][1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota_4, 1), kwargs = {})
#   %div_8 : Tensor "i64[129, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor_mode](args = (%unsqueeze_5, 128), kwargs = {rounding_mode: floor})
#   %mul_12 : Tensor "i64[129, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_8, 33), kwargs = {})
#   %mul_13 : Tensor "i64[129, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_12, 512), kwargs = {})
#   %iota_5 : Tensor "i64[3][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (3,), kwargs = {start: 129, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %unsqueeze_6 : Tensor "i64[1, 3][3, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota_5, 0), kwargs = {})
#   %div_9 : Tensor "i64[1, 3][3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor_mode](args = (%unsqueeze_6, 4), kwargs = {rounding_mode: floor})
#   %mul_14 : Tensor "i64[1, 3][3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_9, 512), kwargs = {})
#   %add_9 : Tensor "i64[129, 3][3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_13, %mul_14), kwargs = {})
#   %remainder_6 : Tensor "i64[129, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.remainder.Scalar](args = (%unsqueeze_5, 32), kwargs = {})
#   %mul_15 : Tensor "i64[129, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%remainder_6, 16), kwargs = {})
#   %add_10 : Tensor "i64[129, 3][3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_9, %mul_15), kwargs = {})
#   %div_10 : Tensor "i64[129, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor_mode](args = (%unsqueeze_5, 32), kwargs = {rounding_mode: floor})
#   %remainder_7 : Tensor "i64[129, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.remainder.Scalar](args = (%div_10, 4), kwargs = {})
#   %mul_16 : Tensor "i64[129, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%remainder_7, 4), kwargs = {})
#   %add_11 : Tensor "i64[129, 3][3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_10, %mul_16), kwargs = {})
#   %remainder_8 : Tensor "i64[1, 3][3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.remainder.Scalar](args = (%unsqueeze_6, 4), kwargs = {})
#   %add_12 : Tensor "i64[129, 3][3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_11, %remainder_8), kwargs = {})
#   %full_2 : Tensor "u8[129, 3][3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([129, 3], 0), kwargs = {dtype: torch.uint8, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %_unsafe_index_put_2 : Tensor "u8[33792][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims._unsafe_index_put_.default](args = (%_unsafe_index_put_1, [%add_12], %full_2), kwargs = {})
#   return %buf7
triton_poi_fused__unsafe_index_put_add_arange_floor_divide_mul_remainder_unsqueeze_zeros_2 = async_compile.triton('triton_poi_fused__unsafe_index_put_add_arange_floor_divide_mul_remainder_unsqueeze_zeros_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 512}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*u8', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': False, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__unsafe_index_put_add_arange_floor_divide_mul_remainder_unsqueeze_zeros_2', 'mutated_arg_names': ['out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 774}, 'backend_hash': '855470BEF4251187CB5023D695885C65615617FA1EB3C786F01CCF7A5DBD6E40', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_put_add_arange_floor_divide_mul_remainder_unsqueeze_zeros_2(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 387
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 3)
    x1 = xindex // 3
    tmp0 = tl.full([1], 0, tl.uint8)
    tl.store(out_ptr0 + (4*(((x1 // 32) % 4)) + 16*((x1 % 32)) + 512*((129 + x0) // 4) + 16896*(x1 // 128) + (((129 + x0) % 4))), tmp0, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, arg1_1 = args
        args.clear()
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf4 = empty_strided_cuda((33792, ), (1, ), torch.uint8)
            assert_size_stride_grouped((arg0_1, arg1_1), ((129, 4128), (4128, )), ((4128, 1), (1, )), 'input')
            arg0_1 = copy_if_misaligned(arg0_1)
            arg1_1 = copy_if_misaligned(arg1_1)
            buf3 = empty_strided_cuda((129, 129, 32), (4128, 32, 1), torch.float8_e4m3fn)
            # Topologically Sorted Source Nodes: [normed], Original ATen: [aten._fused_rms_norm]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__fused_rms_norm_0.run(arg0_1, arg1_1, buf4, buf3, 129, 4128, stream=raw_stream0)
            del arg0_1
            del arg1_1
            # Topologically Sorted Source Nodes: [arange_2, r_1, floordiv_3, mul_5, mul_6, arange_3, c_1, floordiv_4, mul_7, add_4, mod_3, mul_8, add_5, floordiv_5, mod_4, mul_9, add_6, mod_5, pad_offset, zeros, out_2], Original ATen: [aten.arange, aten.unsqueeze, aten.floor_divide, aten.mul, aten.add, aten.remainder, aten.zeros, aten._unsafe_index_put]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_index_put_add_arange_floor_divide_mul_remainder_unsqueeze_zeros_1.run(buf4, 16764, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [arange_4, r_2, floordiv_6, mul_10, mul_11, arange_5, c_2, floordiv_7, mul_12, add_8, mod_6, mul_13, add_9, floordiv_8, mod_7, mul_14, add_10, mod_8, pad_offset_1, zeros_2, out_3], Original ATen: [aten.arange, aten.unsqueeze, aten.floor_divide, aten.mul, aten.add, aten.remainder, aten.zeros, aten._unsafe_index_put]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_index_put_add_arange_floor_divide_mul_remainder_unsqueeze_zeros_2.run(buf4, 387, stream=raw_stream0)
        return (reinterpret_tensor(buf3, (129, 4128), (4128, 1), 0), buf4, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((129, 4128), (4128, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1_1 = rand_strided((4128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    return [arg0_1, arg1_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
