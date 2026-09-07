# AOT ID: ['2_inference']
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


# kernel path: /data/users/eellison/pytorch/agent_space/nested_quant_comparison/cache_pr191974_padded_review_pair/tmpnotbpim2/bl/cbl537s3biwp5yudfabwkgfxwizr62kszqpasezvyn6uouyfdzkn.py
# Topologically Sorted Source Nodes: [rms_norm], Original ATen: [aten._fused_rms_norm]
# Source node to ATen node mapping:
#   rms_norm => convert_element_type, mean, pow_1
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr2': '*fp8e4nv', 'out_ptr5': '*u8', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': False, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__fused_rms_norm_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 2, 'autotune_hints': set(), 'tiling_scores': {'x': 7912, 'r0_': 8101888}, 'min_rblock': 16, 'backend_hash': '855470BEF4251187CB5023D695885C65615617FA1EB3C786F01CCF7A5DBD6E40', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False}
)
@triton.jit
def triton_red_fused__fused_rms_norm_0(in_ptr0, in_ptr1, out_ptr2, out_ptr5, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 989
    r0_numel = 4096
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
        tmp0 = tl.load(in_ptr0 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
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
        reduced_r0_index_mask = reduced_r0_index < 256
        r0_4 = reduced_r0_index
        lane2_r0_index = r0_offset // 2 + tl.arange(0, R0_BLOCK // 2)[None, :]
        lane2_r0_index_mask = lane2_r0_index < 2048
        r0_7 = lane2_r0_index
        tmp6 = tl.load(in_ptr0 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp14 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tl.full([1, 1], 4096.0, tl.float32)
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
        tl.store(out_ptr2 + (4*(((x0 // 32) % 4)) + 16*((x0 % 32)) + 512*(r0_4 // 4) + 32768*(x0 // 128) + ((r0_4 % 4))), tmp45, reduced_r0_index_mask & xmask)
        tl.store(out_ptr5 + (r0_7 + 2048*x0), tmp79, lane2_r0_index_mask & xmask)
''', device_str='cuda')


# kernel path: /data/users/eellison/pytorch/agent_space/nested_quant_comparison/cache_pr191974_padded_review_pair/tmpnotbpim2/7b/c7bflv4ldaze56szcnnv7u4jgsjzm4qe6ek65vwe3ljferio7vus.py
# Topologically Sorted Source Nodes: [linear, linear_1, linear_2, linear_3, row_chunk, mul_7, row_outer, mul_8, add_4, row_lane, logical_row, ge, col_outer, mul_9, col_inner, logical_col, ge_1, padding_mask, padding, index_put_default], Original ATen: [aten.arange, aten.floor_divide, aten.mul, aten.remainder, aten.add, aten.ge, aten.bitwise_or, aten.zeros, aten.index_put]
# Source node to ATen node mapping:
#   add_4 => add_5
#   col_inner => remainder_3
#   col_outer => remainder_6
#   ge => ge
#   ge_1 => ge_1
#   index_put_default => index_put_1
#   linear => iota_2
#   linear_1 => div_4
#   linear_2 => div_5
#   linear_3 => div_6
#   logical_col => add_7
#   logical_row => add_6
#   mul_7 => mul_9
#   mul_8 => mul_10
#   mul_9 => mul_11
#   padding => full_default
#   padding_mask => bitwise_or
#   row_chunk => div_7
#   row_lane => remainder_5
#   row_outer => remainder_4
# Graph fragment:
#   %buf6 : Tensor "f8e4m3fn[262144][1]cuda:0" = PlaceHolder[target=buf6]
#   %iota_2 : Tensor "i64[262144][1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.iota.default](args = (262144,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %div_4 : Tensor "i64[262144][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor_mode](args = (%iota_2, 4), kwargs = {rounding_mode: floor})
#   %div_5 : Tensor "i64[262144][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor_mode](args = (%div_4, 4), kwargs = {rounding_mode: floor})
#   %div_6 : Tensor "i64[262144][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor_mode](args = (%div_5, 32), kwargs = {rounding_mode: floor})
#   %div_7 : Tensor "i64[262144][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor_mode](args = (%div_6, 64), kwargs = {rounding_mode: floor})
#   %mul_9 : Tensor "i64[262144][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_7, 128), kwargs = {})
#   %remainder_4 : Tensor "i64[262144][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.remainder.Scalar](args = (%div_4, 4), kwargs = {})
#   %mul_10 : Tensor "i64[262144][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%remainder_4, 32), kwargs = {})
#   %add_5 : Tensor "i64[262144][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_9, %mul_10), kwargs = {})
#   %remainder_5 : Tensor "i64[262144][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.remainder.Scalar](args = (%div_5, 32), kwargs = {})
#   %add_6 : Tensor "i64[262144][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_5, %remainder_5), kwargs = {})
#   %ge : Tensor "b8[262144][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%add_6, 989), kwargs = {})
#   %remainder_6 : Tensor "i64[262144][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.remainder.Scalar](args = (%div_6, 64), kwargs = {})
#   %mul_11 : Tensor "i64[262144][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%remainder_6, 4), kwargs = {})
#   %remainder_3 : Tensor "i64[262144][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.remainder.Scalar](args = (%iota_2, 4), kwargs = {})
#   %add_7 : Tensor "i64[262144][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, %remainder_3), kwargs = {})
#   %ge_1 : Tensor "b8[262144][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%add_7, 256), kwargs = {})
#   %bitwise_or : Tensor "b8[262144][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%ge, %ge_1), kwargs = {})
#   %full_default : Tensor "f8e4m3fn[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.float8_e4m3fn, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put_1 : Tensor "f8e4m3fn[262144][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%index_put, [%bitwise_or], %full_default), kwargs = {})
#   return %buf7
triton_poi_fused_add_arange_bitwise_or_floor_divide_ge_index_put_mul_remainder_zeros_1 = async_compile.triton('triton_poi_fused_add_arange_bitwise_or_floor_divide_ge_index_put_mul_remainder_zeros_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp8e4nv', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': False, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_arange_bitwise_or_floor_divide_ge_index_put_mul_remainder_zeros_1', 'mutated_arg_names': ['out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 524288}, 'backend_hash': '855470BEF4251187CB5023D695885C65615617FA1EB3C786F01CCF7A5DBD6E40', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_arange_bitwise_or_floor_divide_ge_index_put_mul_remainder_zeros_1(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = (32*(((x0 // 4) % 4)) + 128*(x0 // 32768) + (((x0 // 16) % 32))).to(tl.int32)
    tmp1 = tl.full([1], 989, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = (4*(((x0 // 512) % 64)) + ((x0 % 4))).to(tl.int32)
    tmp4 = tl.full([1], 256, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp6 = tmp2 | tmp5
    tmp7 = tl.full([1], 0.0, tl.float8e4nv)
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
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf5 = empty_strided_cuda((262144, ), (1, ), torch.float8_e4m3fn)
            assert_size_stride_grouped((arg0_1, arg1_1), ((989, 4096), (4096, )), ((4096, 1), (1, )), 'input')
            arg0_1 = copy_if_misaligned(arg0_1)
            arg1_1 = copy_if_misaligned(arg1_1)
            buf4 = empty_strided_cuda((989, 256, 8), (2048, 8, 1), torch.uint8)
            # Topologically Sorted Source Nodes: [rms_norm], Original ATen: [aten._fused_rms_norm]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__fused_rms_norm_0.run(arg0_1, arg1_1, buf5, buf4, 989, 4096, stream=raw_stream0)
            del arg0_1
            del arg1_1
            # Topologically Sorted Source Nodes: [linear, linear_1, linear_2, linear_3, row_chunk, mul_7, row_outer, mul_8, add_4, row_lane, logical_row, ge, col_outer, mul_9, col_inner, logical_col, ge_1, padding_mask, padding, index_put_default], Original ATen: [aten.arange, aten.floor_divide, aten.mul, aten.remainder, aten.add, aten.ge, aten.bitwise_or, aten.zeros, aten.index_put]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_arange_bitwise_or_floor_divide_ge_index_put_mul_remainder_zeros_1.run(buf5, 262144, stream=raw_stream0)
        return (reinterpret_tensor(buf4, (989, 2048), (2048, 1), 0), buf5, )

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
