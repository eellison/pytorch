# AOT ID: ['7_inference']
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


# kernel path: /data/users/eellison/pytorch/agent_space/nested_quant_comparison/cache_index_put_integrated_fp4/tmpfe2bn3ut/l2/cl2fh56ewncllxrvr4a76aqrxj7ywrupbbwyiypdopnmiwkrp3hz.py
# Topologically Sorted Source Nodes: [rms_norm], Original ATen: [aten._fused_rms_norm]
# Source node to ATen node mapping:
#   rms_norm => convert_element_type, mean, pow_1
# Graph fragment:
#   %arg0_1 : Tensor "bf16[1, 4096][4096, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %convert_element_type : Tensor "f32[1, 4096][4096, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.float32), kwargs = {})
#   %pow_1 : Tensor "f32[1, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type, 2), kwargs = {})
#   %mean : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [1], True), kwargs = {})
#   return %buf0
triton_red_fused__fused_rms_norm_0 = async_compile.triton('triton_red_fused__fused_rms_norm_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r0_': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {'xnumel': 1}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__fused_rms_norm_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'autotune_hints': set(), 'tiling_scores': {'r0_': 8192}, 'backend_hash': '855470BEF4251187CB5023D695885C65615617FA1EB3C786F01CCF7A5DBD6E40', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused__fused_rms_norm_0(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 4096
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    _tmp4 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tmp1 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(r0_mask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (tl.full([1, 1], 0, tl.int32).broadcast_to(XBLOCK, 1)), tmp4, None)
''', device_str='cuda')


# kernel path: /data/users/eellison/pytorch/agent_space/nested_quant_comparison/cache_index_put_integrated_fp4/tmpfe2bn3ut/yk/cykxfxqvblo7uh26opv4i4l5h7okx5oc2m4eep5zl6kf2e7qcpfc.py
# Topologically Sorted Source Nodes: [rms_norm, normed, pairs, getitem, float_2, abs_1, amax, truediv, clamp, scale, float_1, inv_scale, unsqueeze, mul, getitem_1, float_3, unsqueeze_1, mul_1, inline_asm_elementwise, to_1], Original ATen: [aten._fused_rms_norm, aten.view, aten.select, aten._to_copy, aten.abs, aten.amax, aten.div, aten.clamp, aten.reciprocal, aten.unsqueeze, aten.mul]
# Source node to ATen node mapping:
#   abs_1 => abs_1
#   amax => amax
#   clamp => clamp_max, clamp_min, convert_element_type_2, convert_element_type_3
#   float_1 => convert_element_type_5
#   float_2 => convert_element_type_6
#   float_3 => convert_element_type_7
#   getitem => select
#   getitem_1 => select_1
#   inline_asm_elementwise => inline_asm_elementwise
#   inv_scale => reciprocal
#   mul => mul_2
#   mul_1 => mul_3
#   normed => view
#   pairs => view_1
#   rms_norm => add, convert_element_type, convert_element_type_1, mean, mul, mul_1, pow_1, rsqrt
#   scale => convert_element_type_4
#   to_1 => convert_element_type_8
#   truediv => div
#   unsqueeze => unsqueeze
#   unsqueeze_1 => unsqueeze_1
# Graph fragment:
#   %arg0_1 : Tensor "bf16[1, 4096][4096, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %buf0 : Tensor "f32[1, 1][1, 1]cuda:0" = PlaceHolder[target=buf0]
#   %arg1_1 : Tensor "bf16[4096][1]cuda:0" = PlaceHolder[target=arg1_1]
#   %amax : Tensor "bf16[1, 256][256, 1]cuda:0" = PlaceHolder[target=amax]
#   %inline_asm_elementwise : Tensor "i32[1, 256, 8][2048, 8, 1]cuda:0" = PlaceHolder[target=inline_asm_elementwise]
#   %convert_element_type : Tensor "f32[1, 4096][4096, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.float32), kwargs = {})
#   %pow_1 : Tensor "f32[1, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type, 2), kwargs = {})
#   %mean : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [1], True), kwargs = {})
#   %add : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul : Tensor "f32[1, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, %rsqrt), kwargs = {})
#   %mul_1 : Tensor "f32[1, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %arg1_1), kwargs = {})
#   %convert_element_type_1 : Tensor "bf16[1, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_1, torch.bfloat16), kwargs = {})
#   %view : Tensor "bf16[1, 256, 16][4096, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_1, [1, 256, 16]), kwargs = {})
#   %view_1 : Tensor "bf16[1, 256, 8, 2][4096, 16, 2, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view, [1, 256, 8, 2]), kwargs = {})
#   %select : Tensor "bf16[1, 256, 8][4096, 16, 2]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%view_1, 3, 0), kwargs = {})
#   %convert_element_type_6 : Tensor "f32[1, 256, 8][2048, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%select, torch.float32), kwargs = {})
#   %abs_1 : Tensor "bf16[1, 256, 16][4096, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%view,), kwargs = {})
#   %amax : Tensor "bf16[1, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%abs_1, [-1]), kwargs = {})
#   %div : Tensor "bf16[1, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%amax, 6.0), kwargs = {})
#   %convert_element_type_2 : Tensor "f32[1, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div, torch.float32), kwargs = {})
#   %clamp_min : Tensor "f32[1, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type_2, 1e-12), kwargs = {})
#   %clamp_max : Tensor "f32[1, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min, 448.0), kwargs = {})
#   %convert_element_type_3 : Tensor "bf16[1, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_max, torch.bfloat16), kwargs = {})
#   %convert_element_type_4 : Tensor "f8e4m3fn[1, 256][256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_3, torch.float8_e4m3fn), kwargs = {})
#   %convert_element_type_5 : Tensor "f32[1, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_4, torch.float32), kwargs = {})
#   %reciprocal : Tensor "f32[1, 256][256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reciprocal.default](args = (%convert_element_type_5,), kwargs = {})
#   %unsqueeze : Tensor "f32[1, 256, 1][256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%reciprocal, -1), kwargs = {})
#   %mul_2 : Tensor "f32[1, 256, 8][2048, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_6, %unsqueeze), kwargs = {})
#   %select_1 : Tensor "bf16[1, 256, 8][4096, 16, 2]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%view_1, 3, 1), kwargs = {})
#   %convert_element_type_7 : Tensor "f32[1, 256, 8][2048, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%select_1, torch.float32), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[1, 256, 1][256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%reciprocal, -1), kwargs = {})
#   %mul_3 : Tensor "f32[1, 256, 8][2048, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_7, %unsqueeze_1), kwargs = {})
#   %inline_asm_elementwise : Tensor "i32[1, 256, 8][2048, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.higher_order.inline_asm_elementwise](args = (%mul_2, %mul_3), kwargs = {asm_str: {.reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, $2, $1; cvt.u32.u8 $0, t;}, constraints: =r,f,f, dtype: torch.int32, is_pure: True, pack: 1})
#   %convert_element_type_8 : Tensor "u8[1, 256, 8][2048, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%inline_asm_elementwise, torch.uint8), kwargs = {})
#   return %amax,%inline_asm_elementwise,%convert_element_type_8
triton_per_fused__fused_rms_norm__to_copy_abs_amax_clamp_div_mul_reciprocal_select_unsqueeze_view_1 = async_compile.triton('triton_per_fused__fused_rms_norm__to_copy_abs_amax_clamp_div_mul_reciprocal_select_unsqueeze_view_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'out_ptr2': '*u8', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_per_fused__fused_rms_norm__to_copy_abs_amax_clamp_div_mul_reciprocal_select_unsqueeze_view_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 1, 'autotune_hints': set(), 'min_rblock': 2, 'backend_hash': '855470BEF4251187CB5023D695885C65615617FA1EB3C786F01CCF7A5DBD6E40', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_per_fused__fused_rms_norm__to_copy_abs_amax_clamp_div_mul_reciprocal_select_unsqueeze_view_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tl_math.abs(tmp13)
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
    tmp17 = tl.where(r0_mask & xmask, tmp15, float("-inf"))
    tmp18 = triton_helpers.max2(tmp17, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp18, xmask)
    lane2_r0_index = r0_offset // 2 + tl.arange(0, R0_BLOCK // 2)[None, :]
    lane2_r0_index_mask = lane2_r0_index < 8
    r0_2 = lane2_r0_index
    tmp19, tmp20 = tl.split(tl.reshape(tmp0, [XBLOCK, (R0_BLOCK//2), 2]))
    tmp21 = tmp19.to(tl.float32)
    tmp22 = tmp21 * tmp8
    tmp23, tmp24 = tl.split(tl.reshape(tmp10, [XBLOCK, (R0_BLOCK//2), 2]))
    tmp25 = tmp23.to(tl.float32)
    tmp26 = tmp22 * tmp25
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tl.full([1, 1], 0.16666666666666666, tl.float32)
    tmp30 = tmp18 * tmp29
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tl.full([1, 1], 1e-12, tl.float32)
    tmp33 = tl.maximum(tmp31, tmp32, tl.PropagateNan.ALL)
    tmp34 = tl.full([1, 1], 448.0, tl.float32)
    tmp35 = tl.minimum(tmp33, tmp34, tl.PropagateNan.ALL)
    tmp36 = tmp35.to(tl.float32)
    tmp37 = tmp36.to(tl.float8e4nv)
    tmp38 = tmp37.to(tl.float32)
    tmp39 = tl.full([1, 1], 1.0, tl.float32)
    tmp40 = (tmp39 / tmp38)
    tmp41 = tmp28 * tmp40
    tmp42 = tmp20.to(tl.float32)
    tmp43 = tmp42 * tmp8
    tmp44 = tmp24.to(tl.float32)
    tmp45 = tmp43 * tmp44
    tmp46 = tmp45.to(tl.float32)
    tmp47 = tmp46.to(tl.float32)
    tmp48 = tmp47 * tmp40
    tmp49 = tl.inline_asm_elementwise('{.reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, $2, $1; cvt.u32.u8 $0, t;}', '=r,f,f', [tmp41, tmp48], dtype=tl.int32, is_pure=True, pack=1)
    tmp50 = tmp49.to(tl.uint8)
    tl.store(out_ptr2 + (r0_2 + 8*x0), tmp50, lane2_r0_index_mask & xmask)
''', device_str='cuda')


# kernel path: /data/users/eellison/pytorch/agent_space/nested_quant_comparison/cache_index_put_integrated_fp4/tmpfe2bn3ut/um/cumwtioy5iczaymm7elyxzhh6ck7n6k7twb5curphld5ovk3wg6o.py
# Topologically Sorted Source Nodes: [truediv, clamp, scale], Original ATen: [aten.div, aten.clamp, aten._to_copy]
# Source node to ATen node mapping:
#   clamp => clamp_max, clamp_min, convert_element_type_2, convert_element_type_3
#   scale => convert_element_type_4
#   truediv => div
# Graph fragment:
#   %amax : Tensor "bf16[1, 256][256, 1]cuda:0" = PlaceHolder[target=amax]
#   %div : Tensor "bf16[1, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%amax, 6.0), kwargs = {})
#   %convert_element_type_2 : Tensor "f32[1, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div, torch.float32), kwargs = {})
#   %clamp_min : Tensor "f32[1, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type_2, 1e-12), kwargs = {})
#   %clamp_max : Tensor "f32[1, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min, 448.0), kwargs = {})
#   %convert_element_type_3 : Tensor "bf16[1, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_max, torch.bfloat16), kwargs = {})
#   %convert_element_type_4 : Tensor "f8e4m3fn[1, 256][256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_3, torch.float8_e4m3fn), kwargs = {})
#   return %convert_element_type_4
triton_poi_fused__to_copy_clamp_div_2 = async_compile.triton('triton_poi_fused__to_copy_clamp_div_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp8e4nv', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__to_copy_clamp_div_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 1024}, 'backend_hash': '855470BEF4251187CB5023D695885C65615617FA1EB3C786F01CCF7A5DBD6E40', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=2
)
@triton.jit
def triton_poi_fused__to_copy_clamp_div_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tl.full([1], 0.16666666666666666, tl.float32)
    tmp2 = tmp0 * tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tl.full([1], 1e-12, tl.float32)
    tmp5 = tl.maximum(tmp3, tmp4, tl.PropagateNan.ALL)
    tmp6 = tl.full([1], 448.0, tl.float32)
    tmp7 = tl.minimum(tmp5, tmp6, tl.PropagateNan.ALL)
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp8.to(tl.float8e4nv)
    tl.store(out_ptr0 + (x0), tmp9, xmask)
''', device_str='cuda')


# kernel path: /data/users/eellison/pytorch/agent_space/nested_quant_comparison/cache_index_put_integrated_fp4/tmpfe2bn3ut/2p/c2pil5clr262iqjsdo7q53fpu4czgdlaivwe3sf4vd55gtp7rnu3.py
# Topologically Sorted Source Nodes: [truediv, clamp, scale, scale_1], Original ATen: [aten.div, aten.clamp, aten._to_copy, aten.constant_pad_nd]
# Source node to ATen node mapping:
#   clamp => clamp_max, clamp_min, convert_element_type_2, convert_element_type_3
#   scale => convert_element_type_4
#   scale_1 => constant_pad_nd
#   truediv => div
# Graph fragment:
#   %div : Tensor "bf16[1, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%amax, 6.0), kwargs = {})
#   %convert_element_type_2 : Tensor "f32[1, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div, torch.float32), kwargs = {})
#   %clamp_min : Tensor "f32[1, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type_2, 1e-12), kwargs = {})
#   %clamp_max : Tensor "f32[1, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min, 448.0), kwargs = {})
#   %convert_element_type_3 : Tensor "bf16[1, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_max, torch.bfloat16), kwargs = {})
#   %convert_element_type_4 : Tensor "f8e4m3fn[1, 256][256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_3, torch.float8_e4m3fn), kwargs = {})
#   %constant_pad_nd : Tensor "f8e4m3fn[128, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%convert_element_type_4, [0, 0, 0, 127], 0.0), kwargs = {})
#   return %buf5
triton_poi_fused__to_copy_clamp_constant_pad_nd_div_3 = async_compile.triton('triton_poi_fused__to_copy_clamp_constant_pad_nd_div_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp8e4nv', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__to_copy_clamp_constant_pad_nd_div_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 65024}, 'backend_hash': '855470BEF4251187CB5023D695885C65615617FA1EB3C786F01CCF7A5DBD6E40', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_clamp_constant_pad_nd_div_3(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.full([1], 0.0, tl.float8e4nv)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /data/users/eellison/pytorch/agent_space/nested_quant_comparison/cache_index_put_integrated_fp4/tmpfe2bn3ut/6l/c6lcs24rmz24ctowcyzom3mjlovjn35q7zmmrksbvarbf7fen3qd.py
# Topologically Sorted Source Nodes: [blocks, permute, reshape, transpose, reshape_1], Original ATen: [aten.view, aten.permute, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   blocks => view_3
#   permute => permute
#   reshape => view_4
#   reshape_1 => clone
#   transpose => permute_1
# Graph fragment:
#   %constant_pad_nd : Tensor "f8e4m3fn[128, 256][256, 1]cuda:0" = PlaceHolder[target=constant_pad_nd]
#   %view_3 : Tensor "f8e4m3fn[1, 128, 64, 4][32768, 256, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%constant_pad_nd, [1, 128, 64, 4]), kwargs = {})
#   %permute : Tensor "f8e4m3fn[1, 64, 128, 4][32768, 4, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_3, [0, 2, 1, 3]), kwargs = {})
#   %view_4 : Tensor "f8e4m3fn[64, 4, 32, 4][4, 8192, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute, [64, 4, 32, 4]), kwargs = {})
#   %permute_1 : Tensor "f8e4m3fn[64, 32, 4, 4][4, 256, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_4, [0, 2, 1, 3]), kwargs = {})
#   %clone : Tensor "f8e4m3fn[64, 32, 4, 4][512, 16, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_1,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone
triton_poi_fused_clone_permute_transpose_view_4 = async_compile.triton('triton_poi_fused_clone_permute_transpose_view_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp8e4nv', 'out_ptr0': '*fp8e4nv', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_clone_permute_transpose_view_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 98304}, 'backend_hash': '855470BEF4251187CB5023D695885C65615617FA1EB3C786F01CCF7A5DBD6E40', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_permute_transpose_view_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 4)
    x1 = ((xindex // 4) % 4)
    x2 = ((xindex // 16) % 32)
    x3 = xindex // 512
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 4*x3 + 256*x2 + 8192*x1), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
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
        assert_size_stride(arg0_1, (1, 4096), (4096, 1), 'input')
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            arg0_1 = copy_if_misaligned(arg0_1)
            buf0 = empty_strided_cuda((1, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [rms_norm], Original ATen: [aten._fused_rms_norm]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__fused_rms_norm_0.run(arg0_1, buf0, 1, 4096, stream=raw_stream0)
            assert_size_stride(arg1_1, (4096, ), (1, ), 'input')
            arg1_1 = copy_if_misaligned(arg1_1)
            buf1 = empty_strided_cuda((1, 256), (256, 1), torch.bfloat16)
            buf3 = empty_strided_cuda((1, 256, 8), (2048, 8, 1), torch.uint8)
            # Topologically Sorted Source Nodes: [rms_norm, normed, pairs, getitem, float_2, abs_1, amax, truediv, clamp, scale, float_1, inv_scale, unsqueeze, mul, getitem_1, float_3, unsqueeze_1, mul_1, inline_asm_elementwise, to_1], Original ATen: [aten._fused_rms_norm, aten.view, aten.select, aten._to_copy, aten.abs, aten.amax, aten.div, aten.clamp, aten.reciprocal, aten.unsqueeze, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm__to_copy_abs_amax_clamp_div_mul_reciprocal_select_unsqueeze_view_1.run(arg0_1, buf0, arg1_1, buf1, buf3, 256, 16, stream=raw_stream0)
            del arg0_1
            del arg1_1
            del buf0
            buf6 = empty_strided_cuda((128, 256), (256, 1), torch.float8_e4m3fn)
            buf4 = reinterpret_tensor(buf6, (1, 256), (256, 1), 0)  # alias
            # Topologically Sorted Source Nodes: [truediv, clamp, scale], Original ATen: [aten.div, aten.clamp, aten._to_copy]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_clamp_div_2.run(buf1, buf4, 256, stream=raw_stream0)
            del buf1
            buf5 = reinterpret_tensor(buf6, (127, 256), (256, 1), 256)  # alias
            # Topologically Sorted Source Nodes: [truediv, clamp, scale, scale_1], Original ATen: [aten.div, aten.clamp, aten._to_copy, aten.constant_pad_nd]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_clamp_constant_pad_nd_div_3.run(buf5, 32512, stream=raw_stream0)
            buf7 = empty_strided_cuda((64, 32, 4, 4), (512, 16, 4, 1), torch.float8_e4m3fn)
            # Topologically Sorted Source Nodes: [blocks, permute, reshape, transpose, reshape_1], Original ATen: [aten.view, aten.permute, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_permute_transpose_view_4.run(buf6, buf7, 32768, stream=raw_stream0)
            del buf4
            del buf5
            del buf6
        return (reinterpret_tensor(buf3, (1, 2048), (2048, 1), 0), reinterpret_tensor(buf7, (32768, ), (1, ), 0), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((1, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
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
