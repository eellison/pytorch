# AOT ID: ['9_inference']
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


# kernel path: /data/users/eellison/pytorch/agent_space/nested_quant_comparison/cache_index_put_control_mxfp8_v2/tmppw1_fidp/cj/ccjz4litrwwvkpebfs4x22f47palsd5ltcftlxbxljuvae3oinkr.py
# Topologically Sorted Source Nodes: [normed], Original ATen: [aten._fused_rms_norm]
# Source node to ATen node mapping:
#   normed => convert_element_type, mean, pow_1
# Graph fragment:
#   %arg0_1 : Tensor "bf16[989, 4096][4096, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %convert_element_type : Tensor "f32[989, 4096][4096, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.float32), kwargs = {})
#   %pow_1 : Tensor "f32[989, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type, 2), kwargs = {})
#   %mean : Tensor "f32[989, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [1], True), kwargs = {})
#   return %buf0
triton_red_fused__fused_rms_norm_0 = async_compile.triton('triton_red_fused__fused_rms_norm_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r0_': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__fused_rms_norm_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'autotune_hints': set(), 'tiling_scores': {'x': 7912, 'r0_': 8101888}, 'backend_hash': '855470BEF4251187CB5023D695885C65615617FA1EB3C786F01CCF7A5DBD6E40', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused__fused_rms_norm_0(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 989
    r0_numel = 4096
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
        tmp0 = tl.load(in_ptr0 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tmp1 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(r0_mask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /data/users/eellison/pytorch/agent_space/nested_quant_comparison/cache_index_put_control_mxfp8_v2/tmppw1_fidp/j7/cj75z2mzdeupechwphfp5s6i6dsiy2buuzt57me3qoa3vhahw4l3.py
# Topologically Sorted Source Nodes: [normed, groups, float_2, ones_like, abs_1, float_1, amax, truediv, raw_scale, scale, to, sub, scale_f32, unsqueeze, truediv_1, clamp, to_1, arange, row, floordiv, mul, offset, arange_1, col, floordiv_1, mul_2, offset_1, mod, mul_3, offset_2, floordiv_2, mod_1, mul_4, offset_3, mod_2, offset_4, out_1], Original ATen: [aten._fused_rms_norm, aten.view, aten._to_copy, aten.ones_like, aten.abs, aten.amax, aten.div, aten.clamp_min, prims.inductor_cvt_e8m0_rceil, aten.sub, aten.ldexp, aten.unsqueeze, aten.clamp, aten.arange, aten.floor_divide, aten.mul, aten.add, aten.remainder, aten._unsafe_index_put]
# Source node to ATen node mapping:
#   abs_1 => abs_1
#   amax => amax
#   arange => iota
#   arange_1 => iota_1
#   clamp => clamp_max, clamp_min_1
#   col => unsqueeze_2
#   float_1 => convert_element_type_2
#   float_2 => convert_element_type_4
#   floordiv => div_2
#   floordiv_1 => div_3
#   floordiv_2 => div_4
#   groups => view
#   mod => remainder
#   mod_1 => remainder_1
#   mod_2 => remainder_2
#   mul => mul_2
#   mul_2 => mul_4
#   mul_3 => mul_5
#   mul_4 => mul_6
#   normed => add, convert_element_type, convert_element_type_1, mean, mul, mul_1, pow_1, rsqrt
#   offset => mul_3
#   offset_1 => add_1
#   offset_2 => add_2
#   offset_3 => add_3
#   offset_4 => add_4
#   ones_like => full_default
#   out_1 => _unsafe_index_put
#   raw_scale => clamp_min
#   row => unsqueeze_1
#   scale => inductor_cvt_e8m0_rceil
#   scale_f32 => ldexp
#   sub => sub
#   to => convert_element_type_3
#   to_1 => convert_element_type_5
#   truediv => div
#   truediv_1 => div_1
#   unsqueeze => unsqueeze
# Graph fragment:
#   %arg0_1 : Tensor "bf16[989, 4096][4096, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %buf0 : Tensor "f32[989, 1][1, 989]cuda:0" = PlaceHolder[target=buf0]
#   %arg1_1 : Tensor "bf16[4096][1]cuda:0" = PlaceHolder[target=arg1_1]
#   %amax : Tensor "f32[989, 128][128, 1]cuda:0" = PlaceHolder[target=amax]
#   %inductor_cvt_e8m0_rceil : Tensor "u8[989, 128][128, 1]cuda:0" = PlaceHolder[target=inductor_cvt_e8m0_rceil]
#   %inductor_predicated_masked_fill : Tensor "u8[131072][1]cuda:0" = PlaceHolder[target=inductor_predicated_masked_fill]
#   %convert_element_type : Tensor "f32[989, 4096][4096, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.float32), kwargs = {})
#   %pow_1 : Tensor "f32[989, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type, 2), kwargs = {})
#   %mean : Tensor "f32[989, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [1], True), kwargs = {})
#   %add : Tensor "f32[989, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt : Tensor "f32[989, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul : Tensor "f32[989, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, %rsqrt), kwargs = {})
#   %mul_1 : Tensor "f32[989, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %arg1_1), kwargs = {})
#   %convert_element_type_1 : Tensor "bf16[989, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_1, torch.bfloat16), kwargs = {})
#   %view : Tensor "bf16[989, 128, 32][4096, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_1, [989, 128, 32]), kwargs = {})
#   %convert_element_type_4 : Tensor "f32[989, 128, 32][4096, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view, torch.float32), kwargs = {})
#   %full_default : Tensor "f32[989, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([989, 128], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %abs_1 : Tensor "bf16[989, 128, 32][4096, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%view,), kwargs = {})
#   %convert_element_type_2 : Tensor "f32[989, 128, 32][4096, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%abs_1, torch.float32), kwargs = {})
#   %amax : Tensor "f32[989, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%convert_element_type_2, [-1]), kwargs = {})
#   %div : Tensor "f32[989, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%amax, 448.0), kwargs = {})
#   %clamp_min : Tensor "f32[989, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%div, 1.1754943508222875e-38), kwargs = {})
#   %inductor_cvt_e8m0_rceil : Tensor "u8[989, 128][128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.inductor_cvt_e8m0_rceil.default](args = (%clamp_min,), kwargs = {})
#   %convert_element_type_3 : Tensor "i32[989, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%inductor_cvt_e8m0_rceil, torch.int32), kwargs = {})
#   %sub : Tensor "i32[989, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_3, 127), kwargs = {})
#   %ldexp : Tensor "f32[989, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ldexp.Tensor](args = (%full_default, %sub), kwargs = {})
#   %unsqueeze : Tensor "f32[989, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%ldexp, -1), kwargs = {})
#   %div_1 : Tensor "f32[989, 128, 32][4096, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_4, %unsqueeze), kwargs = {})
#   %clamp_min_1 : Tensor "f32[989, 128, 32][4096, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%div_1, -448.0), kwargs = {})
#   %clamp_max : Tensor "f32[989, 128, 32][4096, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_1, 448.0), kwargs = {})
#   %convert_element_type_5 : Tensor "f8e4m3fn[989, 128, 32][4096, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_max, torch.float8_e4m3fn), kwargs = {})
#   %iota : Tensor "i64[989][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (989,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %unsqueeze_1 : Tensor "i64[989, 1][1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota, 1), kwargs = {})
#   %div_2 : Tensor "i64[989, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor_mode](args = (%unsqueeze_1, 128), kwargs = {rounding_mode: floor})
#   %mul_2 : Tensor "i64[989, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_2, 32), kwargs = {})
#   %mul_3 : Tensor "i64[989, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, 512), kwargs = {})
#   %iota_1 : Tensor "i64[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (128,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %unsqueeze_2 : Tensor "i64[1, 128][128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota_1, 0), kwargs = {})
#   %div_3 : Tensor "i64[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor_mode](args = (%unsqueeze_2, 4), kwargs = {rounding_mode: floor})
#   %mul_4 : Tensor "i64[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_3, 512), kwargs = {})
#   %add_1 : Tensor "i64[989, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %mul_4), kwargs = {})
#   %remainder : Tensor "i64[989, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.remainder.Scalar](args = (%unsqueeze_1, 32), kwargs = {})
#   %mul_5 : Tensor "i64[989, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%remainder, 16), kwargs = {})
#   %add_2 : Tensor "i64[989, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1, %mul_5), kwargs = {})
#   %div_4 : Tensor "i64[989, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor_mode](args = (%unsqueeze_1, 32), kwargs = {rounding_mode: floor})
#   %remainder_1 : Tensor "i64[989, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.remainder.Scalar](args = (%div_4, 4), kwargs = {})
#   %mul_6 : Tensor "i64[989, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%remainder_1, 4), kwargs = {})
#   %add_3 : Tensor "i64[989, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, %mul_6), kwargs = {})
#   %remainder_2 : Tensor "i64[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.remainder.Scalar](args = (%unsqueeze_2, 4), kwargs = {})
#   %add_4 : Tensor "i64[989, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_3, %remainder_2), kwargs = {})
#   %_unsafe_index_put : Tensor "u8[131072][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims._unsafe_index_put_.default](args = (%empty, [%add_4], %inductor_cvt_e8m0_rceil), kwargs = {})
#   return %amax,%inductor_cvt_e8m0_rceil,%buf5,%convert_element_type_5
triton_per_fused__fused_rms_norm__to_copy__unsafe_index_put_abs_add_amax_arange_clamp_clamp_min_div_floor_divide_inductor_cvt_e8m0_rceil_ldexp_mul_ones_like_remainder_sub_unsqueeze_view_1 = async_compile.triton('triton_per_fused__fused_rms_norm__to_copy__unsafe_index_put_abs_add_amax_arange_clamp_clamp_min_div_floor_divide_inductor_cvt_e8m0_rceil_ldexp_mul_ones_like_remainder_sub_unsqueeze_view_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 131072, 'r0_': 32},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'out_ptr2': '*u8', 'out_ptr3': '*fp8e4nv', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_per_fused__fused_rms_norm__to_copy__unsafe_index_put_abs_add_amax_arange_clamp_clamp_min_div_floor_divide_inductor_cvt_e8m0_rceil_ldexp_mul_ones_like_remainder_sub_unsqueeze_view_1', 'mutated_arg_names': ['out_ptr2'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 1, 'autotune_hints': set(), 'tiling_scores': {'x': 253184, 'r0_': 16211968}, 'backend_hash': '855470BEF4251187CB5023D695885C65615617FA1EB3C786F01CCF7A5DBD6E40', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_per_fused__fused_rms_norm__to_copy__unsafe_index_put_abs_add_amax_arange_clamp_clamp_min_div_floor_divide_inductor_cvt_e8m0_rceil_ldexp_mul_ones_like_remainder_sub_unsqueeze_view_1(in_ptr0, in_ptr1, in_ptr2, out_ptr2, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 126592
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
    tl.store(out_ptr2 + (4*(((x1 // 32) % 4)) + 16*((x1 % 32)) + 512*(x0 // 4) + 16384*(x1 // 128) + ((x0 % 4))), tmp24, xmask)
    tl.store(out_ptr3 + (r0_2 + 32*x3), tmp36, xmask)
''', device_str='cuda')


# kernel path: /data/users/eellison/pytorch/agent_space/nested_quant_comparison/cache_index_put_control_mxfp8_v2/tmppw1_fidp/i4/ci4mxsihw7dxh2bmtsvvhw3ezpcnjv3ldxqua5iwjfycj5jszkre.py
# Topologically Sorted Source Nodes: [linear, linear_1, linear_2, linear_3, row_chunk, mul_5, row_outer, mul_6, add_4, row_lane, logical_row, ge, col_outer, mul_7, col_inner, logical_col, ge_1, mask, zero, inductor_predicated_masked_fill_default], Original ATen: [aten.arange, aten.floor_divide, aten.mul, aten.remainder, aten.add, aten.ge, aten.bitwise_or, aten.zeros, prims.inductor_predicated_masked_fill]
# Source node to ATen node mapping:
#   add_4 => add_5
#   col_inner => remainder_3
#   col_outer => remainder_6
#   ge => ge
#   ge_1 => ge_1
#   inductor_predicated_masked_fill_default => inductor_predicated_masked_fill
#   linear => iota_2
#   linear_1 => div_5
#   linear_2 => div_6
#   linear_3 => div_7
#   logical_col => add_7
#   logical_row => add_6
#   mask => bitwise_or
#   mul_5 => mul_7
#   mul_6 => mul_8
#   mul_7 => mul_9
#   row_chunk => div_8
#   row_lane => remainder_5
#   row_outer => remainder_4
#   zero => full_1
# Graph fragment:
#   %buf5 : Tensor "u8[131072][1]cuda:0" = PlaceHolder[target=buf5]
#   %iota_2 : Tensor "i64[131072][1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.iota.default](args = (131072,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %div_5 : Tensor "i64[131072][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor_mode](args = (%iota_2, 4), kwargs = {rounding_mode: floor})
#   %div_6 : Tensor "i64[131072][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor_mode](args = (%div_5, 4), kwargs = {rounding_mode: floor})
#   %div_7 : Tensor "i64[131072][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor_mode](args = (%div_6, 32), kwargs = {rounding_mode: floor})
#   %div_8 : Tensor "i64[131072][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor_mode](args = (%div_7, 32), kwargs = {rounding_mode: floor})
#   %mul_7 : Tensor "i64[131072][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_8, 128), kwargs = {})
#   %remainder_4 : Tensor "i64[131072][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.remainder.Scalar](args = (%div_5, 4), kwargs = {})
#   %mul_8 : Tensor "i64[131072][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%remainder_4, 32), kwargs = {})
#   %add_5 : Tensor "i64[131072][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_7, %mul_8), kwargs = {})
#   %remainder_5 : Tensor "i64[131072][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.remainder.Scalar](args = (%div_6, 32), kwargs = {})
#   %add_6 : Tensor "i64[131072][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_5, %remainder_5), kwargs = {})
#   %ge : Tensor "b8[131072][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%add_6, 989), kwargs = {})
#   %remainder_6 : Tensor "i64[131072][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.remainder.Scalar](args = (%div_7, 32), kwargs = {})
#   %mul_9 : Tensor "i64[131072][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%remainder_6, 4), kwargs = {})
#   %remainder_3 : Tensor "i64[131072][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.remainder.Scalar](args = (%iota_2, 4), kwargs = {})
#   %add_7 : Tensor "i64[131072][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_9, %remainder_3), kwargs = {})
#   %ge_1 : Tensor "b8[131072][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%add_7, 128), kwargs = {})
#   %bitwise_or : Tensor "b8[131072][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%ge, %ge_1), kwargs = {})
#   %full_1 : Tensor "u8[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.uint8, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %inductor_predicated_masked_fill : Tensor "u8[131072][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.inductor_predicated_masked_fill_.default](args = (%_unsafe_index_put, %bitwise_or, %full_1), kwargs = {})
#   return %buf6
triton_poi_fused_add_arange_bitwise_or_floor_divide_ge_inductor_predicated_masked_fill_mul_remainder_zeros_2 = async_compile.triton('triton_poi_fused_add_arange_bitwise_or_floor_divide_ge_inductor_predicated_masked_fill_mul_remainder_zeros_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*u8', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_arange_bitwise_or_floor_divide_ge_inductor_predicated_masked_fill_mul_remainder_zeros_2', 'mutated_arg_names': ['out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 262144}, 'backend_hash': '855470BEF4251187CB5023D695885C65615617FA1EB3C786F01CCF7A5DBD6E40', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_arange_bitwise_or_floor_divide_ge_inductor_predicated_masked_fill_mul_remainder_zeros_2(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = (32*(((x0 // 4) % 4)) + 128*(x0 // 16384) + (((x0 // 16) % 32))).to(tl.int32)
    tmp1 = tl.full([1], 989, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = (4*(((x0 // 512) % 32)) + ((x0 % 4))).to(tl.int32)
    tmp4 = tl.full([1], 128, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp6 = tmp2 | tmp5
    tmp7 = tl.full([1], 0, tl.uint8)
    tmp8 = tl.full([1], 0, tl.int32)
    tmp9 = tl.full(tmp8.shape, 0, tmp8.dtype)
    tmp10 = tl.where(tmp6, tmp8, tmp9)
    tl.store(out_ptr0 + (x0), tmp7, tmp6)
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
        assert_size_stride(arg0_1, (989, 4096), (4096, 1), 'input')
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            arg0_1 = copy_if_misaligned(arg0_1)
            buf0 = empty_strided_cuda((989, 1), (1, 989), torch.float32)
            # Topologically Sorted Source Nodes: [normed], Original ATen: [aten._fused_rms_norm]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__fused_rms_norm_0.run(arg0_1, buf0, 989, 4096, stream=raw_stream0)
            buf4 = empty_strided_cuda((131072, ), (1, ), torch.uint8)
            assert_size_stride(arg1_1, (4096, ), (1, ), 'input')
            arg1_1 = copy_if_misaligned(arg1_1)
            buf3 = empty_strided_cuda((989, 128, 32), (4096, 32, 1), torch.float8_e4m3fn)
            # Topologically Sorted Source Nodes: [normed, groups, float_2, ones_like, abs_1, float_1, amax, truediv, raw_scale, scale, to, sub, scale_f32, unsqueeze, truediv_1, clamp, to_1, arange, row, floordiv, mul, offset, arange_1, col, floordiv_1, mul_2, offset_1, mod, mul_3, offset_2, floordiv_2, mod_1, mul_4, offset_3, mod_2, offset_4, out_1], Original ATen: [aten._fused_rms_norm, aten.view, aten._to_copy, aten.ones_like, aten.abs, aten.amax, aten.div, aten.clamp_min, prims.inductor_cvt_e8m0_rceil, aten.sub, aten.ldexp, aten.unsqueeze, aten.clamp, aten.arange, aten.floor_divide, aten.mul, aten.add, aten.remainder, aten._unsafe_index_put]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm__to_copy__unsafe_index_put_abs_add_amax_arange_clamp_clamp_min_div_floor_divide_inductor_cvt_e8m0_rceil_ldexp_mul_ones_like_remainder_sub_unsqueeze_view_1.run(arg0_1, buf0, arg1_1, buf4, buf3, 126592, 32, stream=raw_stream0)
            del arg0_1
            del arg1_1
            del buf0
            # Topologically Sorted Source Nodes: [linear, linear_1, linear_2, linear_3, row_chunk, mul_5, row_outer, mul_6, add_4, row_lane, logical_row, ge, col_outer, mul_7, col_inner, logical_col, ge_1, mask, zero, inductor_predicated_masked_fill_default], Original ATen: [aten.arange, aten.floor_divide, aten.mul, aten.remainder, aten.add, aten.ge, aten.bitwise_or, aten.zeros, prims.inductor_predicated_masked_fill]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_arange_bitwise_or_floor_divide_ge_inductor_predicated_masked_fill_mul_remainder_zeros_2.run(buf4, 131072, stream=raw_stream0)
        return (reinterpret_tensor(buf3, (989, 4096), (4096, 1), 0), buf4, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((989, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    return [arg0_1, arg1_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
