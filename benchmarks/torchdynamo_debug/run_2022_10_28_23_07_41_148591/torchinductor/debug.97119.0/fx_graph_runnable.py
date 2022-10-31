import torch._inductor.overrides

import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
from torch.fx.experimental.proxy_tensor import make_fx

# torch version: 1.14.0a0+gitd13b678
# torch cuda version: 11.6
# torch git version: d13b6781d8b7353919ee06378636773f762b880e


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2022 NVIDIA Corporation 
# Built on Thu_Feb_10_18:23:41_PST_2022 
# Cuda compilation tools, release 11.6, V11.6.112 
# Build cuda_11.6.r11.6/compiler.30978841_0 

# GPU Hardware Info: 
# NVIDIA A100-SXM4-40GB : 8 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('_tensor_constant0', torch.randn([], dtype=torch.float32))
        self.register_buffer('_tensor_constant1', torch.randn([], dtype=torch.float32))
        self.register_buffer('_tensor_constant2', torch.randn([], dtype=torch.float32))

    
    
    def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193):
        view = torch.ops.aten.view.default(primals_191, [-1, 128]);  primals_191 = None
        embedding = torch.ops.aten.embedding.default(primals_43, view)
        ones = torch.ops.aten.ones.default([2, 128], device = device(type='cuda', index=0), pin_memory = False)
        alias = torch.ops.aten.alias.default(ones);  ones = None
        alias_1 = torch.ops.aten.alias.default(alias);  alias = None
        slice_1 = torch.ops.aten.slice.Tensor(alias_1, 0, 0, 9223372036854775807);  alias_1 = None
        unsqueeze = torch.ops.aten.unsqueeze.default(slice_1, 1);  slice_1 = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
        slice_2 = torch.ops.aten.slice.Tensor(unsqueeze_1, 3, 0, 9223372036854775807);  unsqueeze_1 = None
        _tensor_constant0 = self._tensor_constant0
        lift_fresh_copy = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None
        sub = torch.ops.aten.sub.Tensor(lift_fresh_copy, slice_2);  lift_fresh_copy = None
        mul = torch.ops.aten.mul.Tensor(sub, -3.4028234663852886e+38);  sub = None
        rand_like = torch.ops.aten.rand_like.default(embedding, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_2 = torch.ops.aten.alias.default(rand_like);  rand_like = None
        gt = torch.ops.aten.gt.Scalar(alias_2, 0.1);  alias_2 = None
        mul_1 = torch.ops.aten.mul.Tensor(gt, embedding)
        mul_2 = torch.ops.aten.mul.Tensor(mul_1, 1.1111111111111112);  mul_1 = None
        pow_1 = torch.ops.aten.pow.Tensor_Scalar(mul_2, 2)
        mean = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
        add = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
        sqrt = torch.ops.aten.sqrt.default(add);  add = None
        reciprocal = torch.ops.aten.reciprocal.default(sqrt);  sqrt = None
        mul_3 = torch.ops.aten.mul.Tensor(mul_2, reciprocal)
        mul_4 = torch.ops.aten.mul.Tensor(primals_1, mul_3);  mul_3 = None
        permute = torch.ops.aten.permute.default(primals_44, [1, 0]);  primals_44 = None
        view_1 = torch.ops.aten.view.default(mul_4, [256, 512]);  mul_4 = None
        mm = torch.ops.aten.mm.default(view_1, permute)
        _unsafe_view = torch.ops.aten._unsafe_view.default(mm, [2, 128, 384]);  mm = None
        view_2 = torch.ops.aten.view.default(_unsafe_view, [2, -1, 6, 64]);  _unsafe_view = None
        permute_1 = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        permute_2 = torch.ops.aten.permute.default(primals_45, [1, 0]);  primals_45 = None
        mm_1 = torch.ops.aten.mm.default(view_1, permute_2)
        _unsafe_view_1 = torch.ops.aten._unsafe_view.default(mm_1, [2, 128, 384]);  mm_1 = None
        view_4 = torch.ops.aten.view.default(_unsafe_view_1, [2, -1, 6, 64]);  _unsafe_view_1 = None
        permute_3 = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
        permute_4 = torch.ops.aten.permute.default(primals_46, [1, 0]);  primals_46 = None
        mm_2 = torch.ops.aten.mm.default(view_1, permute_4);  view_1 = None
        _unsafe_view_2 = torch.ops.aten._unsafe_view.default(mm_2, [2, 128, 384]);  mm_2 = None
        view_6 = torch.ops.aten.view.default(_unsafe_view_2, [2, -1, 6, 64]);  _unsafe_view_2 = None
        permute_5 = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
        permute_6 = torch.ops.aten.permute.default(permute_3, [0, 1, 3, 2]);  permute_3 = None
        expand = torch.ops.aten.expand.default(permute_1, [2, 6, 128, 64]);  permute_1 = None
        clone = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        _unsafe_view_3 = torch.ops.aten._unsafe_view.default(clone, [12, 128, 64]);  clone = None
        expand_1 = torch.ops.aten.expand.default(permute_6, [2, 6, 64, 128]);  permute_6 = None
        clone_1 = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        _unsafe_view_4 = torch.ops.aten._unsafe_view.default(clone_1, [12, 64, 128]);  clone_1 = None
        bmm = torch.ops.aten.bmm.default(_unsafe_view_3, _unsafe_view_4)
        _unsafe_view_5 = torch.ops.aten._unsafe_view.default(bmm, [2, 6, 128, 128]);  bmm = None
        arange = torch.ops.aten.arange.default(128, dtype = torch.int64, device = device(type='cuda', index=0), pin_memory = False)
        alias_6 = torch.ops.aten.alias.default(arange);  arange = None
        alias_7 = torch.ops.aten.alias.default(alias_6);  alias_6 = None
        slice_3 = torch.ops.aten.slice.Tensor(alias_7, 0, 0, 9223372036854775807)
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(slice_3, 1);  slice_3 = None
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(alias_7, 0);  alias_7 = None
        slice_4 = torch.ops.aten.slice.Tensor(unsqueeze_3, 1, 0, 9223372036854775807);  unsqueeze_3 = None
        sub_1 = torch.ops.aten.sub.Tensor(slice_4, unsqueeze_2);  slice_4 = unsqueeze_2 = None
        gt_1 = torch.ops.aten.gt.Scalar(sub_1, 0)
        _to_copy = torch.ops.aten._to_copy.default(gt_1, dtype = torch.int64);  gt_1 = None
        mul_5 = torch.ops.aten.mul.Tensor(_to_copy, 16);  _to_copy = None
        add_1 = torch.ops.aten.add.Tensor(mul_5, 0);  mul_5 = None
        abs_1 = torch.ops.aten.abs.default(sub_1)
        lt = torch.ops.aten.lt.Scalar(abs_1, 8)
        _to_copy_1 = torch.ops.aten._to_copy.default(abs_1, dtype = torch.float32)
        div = torch.ops.aten.div.Tensor(_to_copy_1, 8);  _to_copy_1 = None
        log = torch.ops.aten.log.default(div);  div = None
        div_1 = torch.ops.aten.div.Tensor(log, 2.772588722239781);  log = None
        mul_6 = torch.ops.aten.mul.Tensor(div_1, 8);  div_1 = None
        _to_copy_2 = torch.ops.aten._to_copy.default(mul_6, dtype = torch.int64);  mul_6 = None
        add_2 = torch.ops.aten.add.Tensor(_to_copy_2, 8);  _to_copy_2 = None
        full_like = torch.ops.aten.full_like.default(add_2, 15, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_10 = torch.ops.aten.alias.default(full_like);  full_like = None
        alias_11 = torch.ops.aten.alias.default(alias_10);  alias_10 = None
        minimum = torch.ops.aten.minimum.default(add_2, alias_11);  add_2 = alias_11 = None
        where = torch.ops.aten.where.self(lt, abs_1, minimum);  lt = abs_1 = minimum = None
        add_3 = torch.ops.aten.add.Tensor(add_1, where);  add_1 = where = None
        embedding_1 = torch.ops.aten.embedding.default(primals_47, add_3);  primals_47 = None
        permute_7 = torch.ops.aten.permute.default(embedding_1, [2, 0, 1]);  embedding_1 = None
        unsqueeze_4 = torch.ops.aten.unsqueeze.default(permute_7, 0);  permute_7 = None
        add_4 = torch.ops.aten.add.Tensor(unsqueeze_4, mul);  unsqueeze_4 = mul = None
        add_5 = torch.ops.aten.add.Tensor(_unsafe_view_5, add_4);  _unsafe_view_5 = None
        amax = torch.ops.aten.amax.default(add_5, [-1], True)
        sub_2 = torch.ops.aten.sub.Tensor(add_5, amax);  add_5 = amax = None
        exp = torch.ops.aten.exp.default(sub_2);  sub_2 = None
        sum_1 = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div_2 = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        philox_seed_like = torch.ops.prims.philox_seed_like.default(div_2)
        philox_rand_like = torch.ops.prims.philox_rand_like.default(div_2, philox_seed_like, 0)
        gt_2 = torch.ops.aten.gt.Scalar(philox_rand_like, 0.1);  philox_rand_like = None
        _to_copy_3 = torch.ops.aten._to_copy.default(gt_2, dtype = torch.float32);  gt_2 = None
        mul_7 = torch.ops.aten.mul.Tensor(_to_copy_3, div_2);  _to_copy_3 = None
        mul_8 = torch.ops.aten.mul.Tensor(mul_7, 1.1111111111111112);  mul_7 = None
        expand_2 = torch.ops.aten.expand.default(mul_8, [2, 6, 128, 128]);  mul_8 = None
        view_7 = torch.ops.aten.view.default(expand_2, [12, 128, 128]);  expand_2 = None
        expand_3 = torch.ops.aten.expand.default(permute_5, [2, 6, 128, 64]);  permute_5 = None
        clone_2 = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
        _unsafe_view_6 = torch.ops.aten._unsafe_view.default(clone_2, [12, 128, 64]);  clone_2 = None
        bmm_1 = torch.ops.aten.bmm.default(view_7, _unsafe_view_6)
        _unsafe_view_7 = torch.ops.aten._unsafe_view.default(bmm_1, [2, 6, 128, 64]);  bmm_1 = None
        permute_8 = torch.ops.aten.permute.default(_unsafe_view_7, [0, 2, 1, 3]);  _unsafe_view_7 = None
        clone_3 = torch.ops.aten.clone.default(permute_8, memory_format = torch.contiguous_format);  permute_8 = None
        view_8 = torch.ops.aten.view.default(clone_3, [2, -1, 384]);  clone_3 = None
        permute_9 = torch.ops.aten.permute.default(primals_48, [1, 0]);  primals_48 = None
        view_9 = torch.ops.aten.view.default(view_8, [256, 384]);  view_8 = None
        mm_3 = torch.ops.aten.mm.default(view_9, permute_9)
        _unsafe_view_8 = torch.ops.aten._unsafe_view.default(mm_3, [2, 128, 512]);  mm_3 = None
        rand_like_1 = torch.ops.aten.rand_like.default(_unsafe_view_8, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_15 = torch.ops.aten.alias.default(rand_like_1);  rand_like_1 = None
        gt_3 = torch.ops.aten.gt.Scalar(alias_15, 0.1);  alias_15 = None
        mul_9 = torch.ops.aten.mul.Tensor(gt_3, _unsafe_view_8);  _unsafe_view_8 = None
        mul_10 = torch.ops.aten.mul.Tensor(mul_9, 1.1111111111111112);  mul_9 = None
        add_6 = torch.ops.aten.add.Tensor(mul_2, mul_10);  mul_2 = mul_10 = None
        pow_2 = torch.ops.aten.pow.Tensor_Scalar(add_6, 2)
        mean_1 = torch.ops.aten.mean.dim(pow_2, [-1], True);  pow_2 = None
        add_7 = torch.ops.aten.add.Tensor(mean_1, 1e-06);  mean_1 = None
        sqrt_1 = torch.ops.aten.sqrt.default(add_7);  add_7 = None
        reciprocal_1 = torch.ops.aten.reciprocal.default(sqrt_1);  sqrt_1 = None
        mul_11 = torch.ops.aten.mul.Tensor(add_6, reciprocal_1)
        mul_12 = torch.ops.aten.mul.Tensor(primals_2, mul_11);  mul_11 = None
        permute_10 = torch.ops.aten.permute.default(primals_49, [1, 0]);  primals_49 = None
        view_10 = torch.ops.aten.view.default(mul_12, [256, 512]);  mul_12 = None
        mm_4 = torch.ops.aten.mm.default(view_10, permute_10)
        _unsafe_view_9 = torch.ops.aten._unsafe_view.default(mm_4, [2, 128, 1024])
        mul_13 = torch.ops.aten.mul.Tensor(_unsafe_view_9, 0.5)
        pow_3 = torch.ops.aten.pow.Tensor_Scalar(_unsafe_view_9, 3.0)
        mul_14 = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
        add_8 = torch.ops.aten.add.Tensor(_unsafe_view_9, mul_14);  _unsafe_view_9 = mul_14 = None
        mul_15 = torch.ops.aten.mul.Tensor(add_8, 0.7978845608028654);  add_8 = None
        mul_16 = torch.ops.aten.mul.Tensor(mul_15, -2.0);  mul_15 = None
        exp_1 = torch.ops.aten.exp.default(mul_16);  mul_16 = None
        add_9 = torch.ops.aten.add.Tensor(exp_1, 1.0);  exp_1 = None
        reciprocal_2 = torch.ops.aten.reciprocal.default(add_9);  add_9 = None
        mul_17 = torch.ops.aten.mul.Tensor(reciprocal_2, 2.0);  reciprocal_2 = None
        sub_3 = torch.ops.aten.sub.Tensor(mul_17, 1.0);  mul_17 = None
        add_10 = torch.ops.aten.add.Tensor(sub_3, 1.0)
        mul_18 = torch.ops.aten.mul.Tensor(mul_13, add_10);  mul_13 = add_10 = None
        permute_11 = torch.ops.aten.permute.default(primals_50, [1, 0]);  primals_50 = None
        mm_5 = torch.ops.aten.mm.default(view_10, permute_11);  view_10 = None
        _unsafe_view_10 = torch.ops.aten._unsafe_view.default(mm_5, [2, 128, 1024])
        mul_19 = torch.ops.aten.mul.Tensor(mul_18, _unsafe_view_10);  mul_18 = _unsafe_view_10 = None
        rand_like_2 = torch.ops.aten.rand_like.default(mul_19, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_22 = torch.ops.aten.alias.default(rand_like_2);  rand_like_2 = None
        gt_4 = torch.ops.aten.gt.Scalar(alias_22, 0.1);  alias_22 = None
        mul_20 = torch.ops.aten.mul.Tensor(gt_4, mul_19);  mul_19 = None
        mul_21 = torch.ops.aten.mul.Tensor(mul_20, 1.1111111111111112);  mul_20 = None
        permute_12 = torch.ops.aten.permute.default(primals_51, [1, 0]);  primals_51 = None
        view_12 = torch.ops.aten.view.default(mul_21, [256, 1024]);  mul_21 = None
        mm_6 = torch.ops.aten.mm.default(view_12, permute_12)
        _unsafe_view_11 = torch.ops.aten._unsafe_view.default(mm_6, [2, 128, 512]);  mm_6 = None
        rand_like_3 = torch.ops.aten.rand_like.default(_unsafe_view_11, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_23 = torch.ops.aten.alias.default(rand_like_3);  rand_like_3 = None
        gt_5 = torch.ops.aten.gt.Scalar(alias_23, 0.1);  alias_23 = None
        mul_22 = torch.ops.aten.mul.Tensor(gt_5, _unsafe_view_11);  _unsafe_view_11 = None
        mul_23 = torch.ops.aten.mul.Tensor(mul_22, 1.1111111111111112);  mul_22 = None
        add_11 = torch.ops.aten.add.Tensor(add_6, mul_23);  mul_23 = None
        pow_4 = torch.ops.aten.pow.Tensor_Scalar(add_11, 2)
        mean_2 = torch.ops.aten.mean.dim(pow_4, [-1], True);  pow_4 = None
        add_12 = torch.ops.aten.add.Tensor(mean_2, 1e-06);  mean_2 = None
        sqrt_2 = torch.ops.aten.sqrt.default(add_12);  add_12 = None
        reciprocal_3 = torch.ops.aten.reciprocal.default(sqrt_2);  sqrt_2 = None
        mul_24 = torch.ops.aten.mul.Tensor(add_11, reciprocal_3)
        mul_25 = torch.ops.aten.mul.Tensor(primals_3, mul_24);  mul_24 = None
        permute_13 = torch.ops.aten.permute.default(primals_52, [1, 0]);  primals_52 = None
        view_13 = torch.ops.aten.view.default(mul_25, [256, 512]);  mul_25 = None
        mm_7 = torch.ops.aten.mm.default(view_13, permute_13)
        _unsafe_view_12 = torch.ops.aten._unsafe_view.default(mm_7, [2, 128, 384]);  mm_7 = None
        view_14 = torch.ops.aten.view.default(_unsafe_view_12, [2, -1, 6, 64]);  _unsafe_view_12 = None
        permute_14 = torch.ops.aten.permute.default(view_14, [0, 2, 1, 3]);  view_14 = None
        permute_15 = torch.ops.aten.permute.default(primals_53, [1, 0]);  primals_53 = None
        mm_8 = torch.ops.aten.mm.default(view_13, permute_15)
        _unsafe_view_13 = torch.ops.aten._unsafe_view.default(mm_8, [2, 128, 384]);  mm_8 = None
        view_16 = torch.ops.aten.view.default(_unsafe_view_13, [2, -1, 6, 64]);  _unsafe_view_13 = None
        permute_16 = torch.ops.aten.permute.default(view_16, [0, 2, 1, 3]);  view_16 = None
        permute_17 = torch.ops.aten.permute.default(primals_54, [1, 0]);  primals_54 = None
        mm_9 = torch.ops.aten.mm.default(view_13, permute_17);  view_13 = None
        _unsafe_view_14 = torch.ops.aten._unsafe_view.default(mm_9, [2, 128, 384]);  mm_9 = None
        view_18 = torch.ops.aten.view.default(_unsafe_view_14, [2, -1, 6, 64]);  _unsafe_view_14 = None
        permute_18 = torch.ops.aten.permute.default(view_18, [0, 2, 1, 3]);  view_18 = None
        permute_19 = torch.ops.aten.permute.default(permute_16, [0, 1, 3, 2]);  permute_16 = None
        expand_4 = torch.ops.aten.expand.default(permute_14, [2, 6, 128, 64]);  permute_14 = None
        clone_4 = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
        _unsafe_view_15 = torch.ops.aten._unsafe_view.default(clone_4, [12, 128, 64]);  clone_4 = None
        expand_5 = torch.ops.aten.expand.default(permute_19, [2, 6, 64, 128]);  permute_19 = None
        clone_5 = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
        _unsafe_view_16 = torch.ops.aten._unsafe_view.default(clone_5, [12, 64, 128]);  clone_5 = None
        bmm_2 = torch.ops.aten.bmm.default(_unsafe_view_15, _unsafe_view_16)
        _unsafe_view_17 = torch.ops.aten._unsafe_view.default(bmm_2, [2, 6, 128, 128]);  bmm_2 = None
        add_13 = torch.ops.aten.add.Tensor(_unsafe_view_17, add_4);  _unsafe_view_17 = None
        amax_1 = torch.ops.aten.amax.default(add_13, [-1], True)
        sub_4 = torch.ops.aten.sub.Tensor(add_13, amax_1);  add_13 = amax_1 = None
        exp_2 = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        sum_2 = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_3 = torch.ops.aten.div.Tensor(exp_2, sum_2);  exp_2 = sum_2 = None
        philox_rand_like_1 = torch.ops.prims.philox_rand_like.default(div_3, philox_seed_like, 196608)
        gt_6 = torch.ops.aten.gt.Scalar(philox_rand_like_1, 0.1);  philox_rand_like_1 = None
        _to_copy_4 = torch.ops.aten._to_copy.default(gt_6, dtype = torch.float32);  gt_6 = None
        mul_26 = torch.ops.aten.mul.Tensor(_to_copy_4, div_3);  _to_copy_4 = None
        mul_27 = torch.ops.aten.mul.Tensor(mul_26, 1.1111111111111112);  mul_26 = None
        expand_6 = torch.ops.aten.expand.default(mul_27, [2, 6, 128, 128]);  mul_27 = None
        view_19 = torch.ops.aten.view.default(expand_6, [12, 128, 128]);  expand_6 = None
        expand_7 = torch.ops.aten.expand.default(permute_18, [2, 6, 128, 64]);  permute_18 = None
        clone_6 = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format);  expand_7 = None
        _unsafe_view_18 = torch.ops.aten._unsafe_view.default(clone_6, [12, 128, 64]);  clone_6 = None
        bmm_3 = torch.ops.aten.bmm.default(view_19, _unsafe_view_18)
        _unsafe_view_19 = torch.ops.aten._unsafe_view.default(bmm_3, [2, 6, 128, 64]);  bmm_3 = None
        permute_20 = torch.ops.aten.permute.default(_unsafe_view_19, [0, 2, 1, 3]);  _unsafe_view_19 = None
        clone_7 = torch.ops.aten.clone.default(permute_20, memory_format = torch.contiguous_format);  permute_20 = None
        view_20 = torch.ops.aten.view.default(clone_7, [2, -1, 384]);  clone_7 = None
        permute_21 = torch.ops.aten.permute.default(primals_55, [1, 0]);  primals_55 = None
        view_21 = torch.ops.aten.view.default(view_20, [256, 384]);  view_20 = None
        mm_10 = torch.ops.aten.mm.default(view_21, permute_21)
        _unsafe_view_20 = torch.ops.aten._unsafe_view.default(mm_10, [2, 128, 512]);  mm_10 = None
        rand_like_4 = torch.ops.aten.rand_like.default(_unsafe_view_20, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_30 = torch.ops.aten.alias.default(rand_like_4);  rand_like_4 = None
        gt_7 = torch.ops.aten.gt.Scalar(alias_30, 0.1);  alias_30 = None
        mul_28 = torch.ops.aten.mul.Tensor(gt_7, _unsafe_view_20);  _unsafe_view_20 = None
        mul_29 = torch.ops.aten.mul.Tensor(mul_28, 1.1111111111111112);  mul_28 = None
        add_14 = torch.ops.aten.add.Tensor(add_11, mul_29);  mul_29 = None
        pow_5 = torch.ops.aten.pow.Tensor_Scalar(add_14, 2)
        mean_3 = torch.ops.aten.mean.dim(pow_5, [-1], True);  pow_5 = None
        add_15 = torch.ops.aten.add.Tensor(mean_3, 1e-06);  mean_3 = None
        sqrt_3 = torch.ops.aten.sqrt.default(add_15);  add_15 = None
        reciprocal_4 = torch.ops.aten.reciprocal.default(sqrt_3);  sqrt_3 = None
        mul_30 = torch.ops.aten.mul.Tensor(add_14, reciprocal_4)
        mul_31 = torch.ops.aten.mul.Tensor(primals_4, mul_30);  mul_30 = None
        permute_22 = torch.ops.aten.permute.default(primals_56, [1, 0]);  primals_56 = None
        view_22 = torch.ops.aten.view.default(mul_31, [256, 512]);  mul_31 = None
        mm_11 = torch.ops.aten.mm.default(view_22, permute_22)
        _unsafe_view_21 = torch.ops.aten._unsafe_view.default(mm_11, [2, 128, 1024])
        mul_32 = torch.ops.aten.mul.Tensor(_unsafe_view_21, 0.5)
        pow_6 = torch.ops.aten.pow.Tensor_Scalar(_unsafe_view_21, 3.0)
        mul_33 = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
        add_16 = torch.ops.aten.add.Tensor(_unsafe_view_21, mul_33);  _unsafe_view_21 = mul_33 = None
        mul_34 = torch.ops.aten.mul.Tensor(add_16, 0.7978845608028654);  add_16 = None
        mul_35 = torch.ops.aten.mul.Tensor(mul_34, -2.0);  mul_34 = None
        exp_3 = torch.ops.aten.exp.default(mul_35);  mul_35 = None
        add_17 = torch.ops.aten.add.Tensor(exp_3, 1.0);  exp_3 = None
        reciprocal_5 = torch.ops.aten.reciprocal.default(add_17);  add_17 = None
        mul_36 = torch.ops.aten.mul.Tensor(reciprocal_5, 2.0);  reciprocal_5 = None
        sub_5 = torch.ops.aten.sub.Tensor(mul_36, 1.0);  mul_36 = None
        add_18 = torch.ops.aten.add.Tensor(sub_5, 1.0)
        mul_37 = torch.ops.aten.mul.Tensor(mul_32, add_18);  mul_32 = add_18 = None
        permute_23 = torch.ops.aten.permute.default(primals_57, [1, 0]);  primals_57 = None
        mm_12 = torch.ops.aten.mm.default(view_22, permute_23);  view_22 = None
        _unsafe_view_22 = torch.ops.aten._unsafe_view.default(mm_12, [2, 128, 1024])
        mul_38 = torch.ops.aten.mul.Tensor(mul_37, _unsafe_view_22);  mul_37 = _unsafe_view_22 = None
        rand_like_5 = torch.ops.aten.rand_like.default(mul_38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_37 = torch.ops.aten.alias.default(rand_like_5);  rand_like_5 = None
        gt_8 = torch.ops.aten.gt.Scalar(alias_37, 0.1);  alias_37 = None
        mul_39 = torch.ops.aten.mul.Tensor(gt_8, mul_38);  mul_38 = None
        mul_40 = torch.ops.aten.mul.Tensor(mul_39, 1.1111111111111112);  mul_39 = None
        permute_24 = torch.ops.aten.permute.default(primals_58, [1, 0]);  primals_58 = None
        view_24 = torch.ops.aten.view.default(mul_40, [256, 1024]);  mul_40 = None
        mm_13 = torch.ops.aten.mm.default(view_24, permute_24)
        _unsafe_view_23 = torch.ops.aten._unsafe_view.default(mm_13, [2, 128, 512]);  mm_13 = None
        rand_like_6 = torch.ops.aten.rand_like.default(_unsafe_view_23, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_38 = torch.ops.aten.alias.default(rand_like_6);  rand_like_6 = None
        gt_9 = torch.ops.aten.gt.Scalar(alias_38, 0.1);  alias_38 = None
        mul_41 = torch.ops.aten.mul.Tensor(gt_9, _unsafe_view_23);  _unsafe_view_23 = None
        mul_42 = torch.ops.aten.mul.Tensor(mul_41, 1.1111111111111112);  mul_41 = None
        add_19 = torch.ops.aten.add.Tensor(add_14, mul_42);  mul_42 = None
        pow_7 = torch.ops.aten.pow.Tensor_Scalar(add_19, 2)
        mean_4 = torch.ops.aten.mean.dim(pow_7, [-1], True);  pow_7 = None
        add_20 = torch.ops.aten.add.Tensor(mean_4, 1e-06);  mean_4 = None
        sqrt_4 = torch.ops.aten.sqrt.default(add_20);  add_20 = None
        reciprocal_6 = torch.ops.aten.reciprocal.default(sqrt_4);  sqrt_4 = None
        mul_43 = torch.ops.aten.mul.Tensor(add_19, reciprocal_6)
        mul_44 = torch.ops.aten.mul.Tensor(primals_5, mul_43);  mul_43 = None
        permute_25 = torch.ops.aten.permute.default(primals_59, [1, 0]);  primals_59 = None
        view_25 = torch.ops.aten.view.default(mul_44, [256, 512]);  mul_44 = None
        mm_14 = torch.ops.aten.mm.default(view_25, permute_25)
        _unsafe_view_24 = torch.ops.aten._unsafe_view.default(mm_14, [2, 128, 384]);  mm_14 = None
        view_26 = torch.ops.aten.view.default(_unsafe_view_24, [2, -1, 6, 64]);  _unsafe_view_24 = None
        permute_26 = torch.ops.aten.permute.default(view_26, [0, 2, 1, 3]);  view_26 = None
        permute_27 = torch.ops.aten.permute.default(primals_60, [1, 0]);  primals_60 = None
        mm_15 = torch.ops.aten.mm.default(view_25, permute_27)
        _unsafe_view_25 = torch.ops.aten._unsafe_view.default(mm_15, [2, 128, 384]);  mm_15 = None
        view_28 = torch.ops.aten.view.default(_unsafe_view_25, [2, -1, 6, 64]);  _unsafe_view_25 = None
        permute_28 = torch.ops.aten.permute.default(view_28, [0, 2, 1, 3]);  view_28 = None
        permute_29 = torch.ops.aten.permute.default(primals_61, [1, 0]);  primals_61 = None
        mm_16 = torch.ops.aten.mm.default(view_25, permute_29);  view_25 = None
        _unsafe_view_26 = torch.ops.aten._unsafe_view.default(mm_16, [2, 128, 384]);  mm_16 = None
        view_30 = torch.ops.aten.view.default(_unsafe_view_26, [2, -1, 6, 64]);  _unsafe_view_26 = None
        permute_30 = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None
        permute_31 = torch.ops.aten.permute.default(permute_28, [0, 1, 3, 2]);  permute_28 = None
        expand_8 = torch.ops.aten.expand.default(permute_26, [2, 6, 128, 64]);  permute_26 = None
        clone_8 = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
        _unsafe_view_27 = torch.ops.aten._unsafe_view.default(clone_8, [12, 128, 64]);  clone_8 = None
        expand_9 = torch.ops.aten.expand.default(permute_31, [2, 6, 64, 128]);  permute_31 = None
        clone_9 = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
        _unsafe_view_28 = torch.ops.aten._unsafe_view.default(clone_9, [12, 64, 128]);  clone_9 = None
        bmm_4 = torch.ops.aten.bmm.default(_unsafe_view_27, _unsafe_view_28)
        _unsafe_view_29 = torch.ops.aten._unsafe_view.default(bmm_4, [2, 6, 128, 128]);  bmm_4 = None
        add_21 = torch.ops.aten.add.Tensor(_unsafe_view_29, add_4);  _unsafe_view_29 = None
        amax_2 = torch.ops.aten.amax.default(add_21, [-1], True)
        sub_6 = torch.ops.aten.sub.Tensor(add_21, amax_2);  add_21 = amax_2 = None
        exp_4 = torch.ops.aten.exp.default(sub_6);  sub_6 = None
        sum_3 = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_4 = torch.ops.aten.div.Tensor(exp_4, sum_3);  exp_4 = sum_3 = None
        philox_rand_like_2 = torch.ops.prims.philox_rand_like.default(div_4, philox_seed_like, 393216)
        gt_10 = torch.ops.aten.gt.Scalar(philox_rand_like_2, 0.1);  philox_rand_like_2 = None
        _to_copy_5 = torch.ops.aten._to_copy.default(gt_10, dtype = torch.float32);  gt_10 = None
        mul_45 = torch.ops.aten.mul.Tensor(_to_copy_5, div_4);  _to_copy_5 = None
        mul_46 = torch.ops.aten.mul.Tensor(mul_45, 1.1111111111111112);  mul_45 = None
        expand_10 = torch.ops.aten.expand.default(mul_46, [2, 6, 128, 128]);  mul_46 = None
        view_31 = torch.ops.aten.view.default(expand_10, [12, 128, 128]);  expand_10 = None
        expand_11 = torch.ops.aten.expand.default(permute_30, [2, 6, 128, 64]);  permute_30 = None
        clone_10 = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
        _unsafe_view_30 = torch.ops.aten._unsafe_view.default(clone_10, [12, 128, 64]);  clone_10 = None
        bmm_5 = torch.ops.aten.bmm.default(view_31, _unsafe_view_30)
        _unsafe_view_31 = torch.ops.aten._unsafe_view.default(bmm_5, [2, 6, 128, 64]);  bmm_5 = None
        permute_32 = torch.ops.aten.permute.default(_unsafe_view_31, [0, 2, 1, 3]);  _unsafe_view_31 = None
        clone_11 = torch.ops.aten.clone.default(permute_32, memory_format = torch.contiguous_format);  permute_32 = None
        view_32 = torch.ops.aten.view.default(clone_11, [2, -1, 384]);  clone_11 = None
        permute_33 = torch.ops.aten.permute.default(primals_62, [1, 0]);  primals_62 = None
        view_33 = torch.ops.aten.view.default(view_32, [256, 384]);  view_32 = None
        mm_17 = torch.ops.aten.mm.default(view_33, permute_33)
        _unsafe_view_32 = torch.ops.aten._unsafe_view.default(mm_17, [2, 128, 512]);  mm_17 = None
        rand_like_7 = torch.ops.aten.rand_like.default(_unsafe_view_32, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_45 = torch.ops.aten.alias.default(rand_like_7);  rand_like_7 = None
        gt_11 = torch.ops.aten.gt.Scalar(alias_45, 0.1);  alias_45 = None
        mul_47 = torch.ops.aten.mul.Tensor(gt_11, _unsafe_view_32);  _unsafe_view_32 = None
        mul_48 = torch.ops.aten.mul.Tensor(mul_47, 1.1111111111111112);  mul_47 = None
        add_22 = torch.ops.aten.add.Tensor(add_19, mul_48);  mul_48 = None
        pow_8 = torch.ops.aten.pow.Tensor_Scalar(add_22, 2)
        mean_5 = torch.ops.aten.mean.dim(pow_8, [-1], True);  pow_8 = None
        add_23 = torch.ops.aten.add.Tensor(mean_5, 1e-06);  mean_5 = None
        sqrt_5 = torch.ops.aten.sqrt.default(add_23);  add_23 = None
        reciprocal_7 = torch.ops.aten.reciprocal.default(sqrt_5);  sqrt_5 = None
        mul_49 = torch.ops.aten.mul.Tensor(add_22, reciprocal_7)
        mul_50 = torch.ops.aten.mul.Tensor(primals_6, mul_49);  mul_49 = None
        permute_34 = torch.ops.aten.permute.default(primals_63, [1, 0]);  primals_63 = None
        view_34 = torch.ops.aten.view.default(mul_50, [256, 512]);  mul_50 = None
        mm_18 = torch.ops.aten.mm.default(view_34, permute_34)
        _unsafe_view_33 = torch.ops.aten._unsafe_view.default(mm_18, [2, 128, 1024])
        mul_51 = torch.ops.aten.mul.Tensor(_unsafe_view_33, 0.5)
        pow_9 = torch.ops.aten.pow.Tensor_Scalar(_unsafe_view_33, 3.0)
        mul_52 = torch.ops.aten.mul.Tensor(pow_9, 0.044715);  pow_9 = None
        add_24 = torch.ops.aten.add.Tensor(_unsafe_view_33, mul_52);  _unsafe_view_33 = mul_52 = None
        mul_53 = torch.ops.aten.mul.Tensor(add_24, 0.7978845608028654);  add_24 = None
        mul_54 = torch.ops.aten.mul.Tensor(mul_53, -2.0);  mul_53 = None
        exp_5 = torch.ops.aten.exp.default(mul_54);  mul_54 = None
        add_25 = torch.ops.aten.add.Tensor(exp_5, 1.0);  exp_5 = None
        reciprocal_8 = torch.ops.aten.reciprocal.default(add_25);  add_25 = None
        mul_55 = torch.ops.aten.mul.Tensor(reciprocal_8, 2.0);  reciprocal_8 = None
        sub_7 = torch.ops.aten.sub.Tensor(mul_55, 1.0);  mul_55 = None
        add_26 = torch.ops.aten.add.Tensor(sub_7, 1.0)
        mul_56 = torch.ops.aten.mul.Tensor(mul_51, add_26);  mul_51 = add_26 = None
        permute_35 = torch.ops.aten.permute.default(primals_64, [1, 0]);  primals_64 = None
        mm_19 = torch.ops.aten.mm.default(view_34, permute_35);  view_34 = None
        _unsafe_view_34 = torch.ops.aten._unsafe_view.default(mm_19, [2, 128, 1024])
        mul_57 = torch.ops.aten.mul.Tensor(mul_56, _unsafe_view_34);  mul_56 = _unsafe_view_34 = None
        rand_like_8 = torch.ops.aten.rand_like.default(mul_57, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_52 = torch.ops.aten.alias.default(rand_like_8);  rand_like_8 = None
        gt_12 = torch.ops.aten.gt.Scalar(alias_52, 0.1);  alias_52 = None
        mul_58 = torch.ops.aten.mul.Tensor(gt_12, mul_57);  mul_57 = None
        mul_59 = torch.ops.aten.mul.Tensor(mul_58, 1.1111111111111112);  mul_58 = None
        permute_36 = torch.ops.aten.permute.default(primals_65, [1, 0]);  primals_65 = None
        view_36 = torch.ops.aten.view.default(mul_59, [256, 1024]);  mul_59 = None
        mm_20 = torch.ops.aten.mm.default(view_36, permute_36)
        _unsafe_view_35 = torch.ops.aten._unsafe_view.default(mm_20, [2, 128, 512]);  mm_20 = None
        rand_like_9 = torch.ops.aten.rand_like.default(_unsafe_view_35, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_53 = torch.ops.aten.alias.default(rand_like_9);  rand_like_9 = None
        gt_13 = torch.ops.aten.gt.Scalar(alias_53, 0.1);  alias_53 = None
        mul_60 = torch.ops.aten.mul.Tensor(gt_13, _unsafe_view_35);  _unsafe_view_35 = None
        mul_61 = torch.ops.aten.mul.Tensor(mul_60, 1.1111111111111112);  mul_60 = None
        add_27 = torch.ops.aten.add.Tensor(add_22, mul_61);  mul_61 = None
        pow_10 = torch.ops.aten.pow.Tensor_Scalar(add_27, 2)
        mean_6 = torch.ops.aten.mean.dim(pow_10, [-1], True);  pow_10 = None
        add_28 = torch.ops.aten.add.Tensor(mean_6, 1e-06);  mean_6 = None
        sqrt_6 = torch.ops.aten.sqrt.default(add_28);  add_28 = None
        reciprocal_9 = torch.ops.aten.reciprocal.default(sqrt_6);  sqrt_6 = None
        mul_62 = torch.ops.aten.mul.Tensor(add_27, reciprocal_9)
        mul_63 = torch.ops.aten.mul.Tensor(primals_7, mul_62);  mul_62 = None
        permute_37 = torch.ops.aten.permute.default(primals_66, [1, 0]);  primals_66 = None
        view_37 = torch.ops.aten.view.default(mul_63, [256, 512]);  mul_63 = None
        mm_21 = torch.ops.aten.mm.default(view_37, permute_37)
        _unsafe_view_36 = torch.ops.aten._unsafe_view.default(mm_21, [2, 128, 384]);  mm_21 = None
        view_38 = torch.ops.aten.view.default(_unsafe_view_36, [2, -1, 6, 64]);  _unsafe_view_36 = None
        permute_38 = torch.ops.aten.permute.default(view_38, [0, 2, 1, 3]);  view_38 = None
        permute_39 = torch.ops.aten.permute.default(primals_67, [1, 0]);  primals_67 = None
        mm_22 = torch.ops.aten.mm.default(view_37, permute_39)
        _unsafe_view_37 = torch.ops.aten._unsafe_view.default(mm_22, [2, 128, 384]);  mm_22 = None
        view_40 = torch.ops.aten.view.default(_unsafe_view_37, [2, -1, 6, 64]);  _unsafe_view_37 = None
        permute_40 = torch.ops.aten.permute.default(view_40, [0, 2, 1, 3]);  view_40 = None
        permute_41 = torch.ops.aten.permute.default(primals_68, [1, 0]);  primals_68 = None
        mm_23 = torch.ops.aten.mm.default(view_37, permute_41);  view_37 = None
        _unsafe_view_38 = torch.ops.aten._unsafe_view.default(mm_23, [2, 128, 384]);  mm_23 = None
        view_42 = torch.ops.aten.view.default(_unsafe_view_38, [2, -1, 6, 64]);  _unsafe_view_38 = None
        permute_42 = torch.ops.aten.permute.default(view_42, [0, 2, 1, 3]);  view_42 = None
        permute_43 = torch.ops.aten.permute.default(permute_40, [0, 1, 3, 2]);  permute_40 = None
        expand_12 = torch.ops.aten.expand.default(permute_38, [2, 6, 128, 64]);  permute_38 = None
        clone_12 = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
        _unsafe_view_39 = torch.ops.aten._unsafe_view.default(clone_12, [12, 128, 64]);  clone_12 = None
        expand_13 = torch.ops.aten.expand.default(permute_43, [2, 6, 64, 128]);  permute_43 = None
        clone_13 = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
        _unsafe_view_40 = torch.ops.aten._unsafe_view.default(clone_13, [12, 64, 128]);  clone_13 = None
        bmm_6 = torch.ops.aten.bmm.default(_unsafe_view_39, _unsafe_view_40)
        _unsafe_view_41 = torch.ops.aten._unsafe_view.default(bmm_6, [2, 6, 128, 128]);  bmm_6 = None
        add_29 = torch.ops.aten.add.Tensor(_unsafe_view_41, add_4);  _unsafe_view_41 = None
        amax_3 = torch.ops.aten.amax.default(add_29, [-1], True)
        sub_8 = torch.ops.aten.sub.Tensor(add_29, amax_3);  add_29 = amax_3 = None
        exp_6 = torch.ops.aten.exp.default(sub_8);  sub_8 = None
        sum_4 = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_5 = torch.ops.aten.div.Tensor(exp_6, sum_4);  exp_6 = sum_4 = None
        philox_rand_like_3 = torch.ops.prims.philox_rand_like.default(div_5, philox_seed_like, 589824)
        gt_14 = torch.ops.aten.gt.Scalar(philox_rand_like_3, 0.1);  philox_rand_like_3 = None
        _to_copy_6 = torch.ops.aten._to_copy.default(gt_14, dtype = torch.float32);  gt_14 = None
        mul_64 = torch.ops.aten.mul.Tensor(_to_copy_6, div_5);  _to_copy_6 = None
        mul_65 = torch.ops.aten.mul.Tensor(mul_64, 1.1111111111111112);  mul_64 = None
        expand_14 = torch.ops.aten.expand.default(mul_65, [2, 6, 128, 128]);  mul_65 = None
        view_43 = torch.ops.aten.view.default(expand_14, [12, 128, 128]);  expand_14 = None
        expand_15 = torch.ops.aten.expand.default(permute_42, [2, 6, 128, 64]);  permute_42 = None
        clone_14 = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
        _unsafe_view_42 = torch.ops.aten._unsafe_view.default(clone_14, [12, 128, 64]);  clone_14 = None
        bmm_7 = torch.ops.aten.bmm.default(view_43, _unsafe_view_42)
        _unsafe_view_43 = torch.ops.aten._unsafe_view.default(bmm_7, [2, 6, 128, 64]);  bmm_7 = None
        permute_44 = torch.ops.aten.permute.default(_unsafe_view_43, [0, 2, 1, 3]);  _unsafe_view_43 = None
        clone_15 = torch.ops.aten.clone.default(permute_44, memory_format = torch.contiguous_format);  permute_44 = None
        view_44 = torch.ops.aten.view.default(clone_15, [2, -1, 384]);  clone_15 = None
        permute_45 = torch.ops.aten.permute.default(primals_69, [1, 0]);  primals_69 = None
        view_45 = torch.ops.aten.view.default(view_44, [256, 384]);  view_44 = None
        mm_24 = torch.ops.aten.mm.default(view_45, permute_45)
        _unsafe_view_44 = torch.ops.aten._unsafe_view.default(mm_24, [2, 128, 512]);  mm_24 = None
        rand_like_10 = torch.ops.aten.rand_like.default(_unsafe_view_44, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_60 = torch.ops.aten.alias.default(rand_like_10);  rand_like_10 = None
        gt_15 = torch.ops.aten.gt.Scalar(alias_60, 0.1);  alias_60 = None
        mul_66 = torch.ops.aten.mul.Tensor(gt_15, _unsafe_view_44);  _unsafe_view_44 = None
        mul_67 = torch.ops.aten.mul.Tensor(mul_66, 1.1111111111111112);  mul_66 = None
        add_30 = torch.ops.aten.add.Tensor(add_27, mul_67);  mul_67 = None
        pow_11 = torch.ops.aten.pow.Tensor_Scalar(add_30, 2)
        mean_7 = torch.ops.aten.mean.dim(pow_11, [-1], True);  pow_11 = None
        add_31 = torch.ops.aten.add.Tensor(mean_7, 1e-06);  mean_7 = None
        sqrt_7 = torch.ops.aten.sqrt.default(add_31);  add_31 = None
        reciprocal_10 = torch.ops.aten.reciprocal.default(sqrt_7);  sqrt_7 = None
        mul_68 = torch.ops.aten.mul.Tensor(add_30, reciprocal_10)
        mul_69 = torch.ops.aten.mul.Tensor(primals_8, mul_68);  mul_68 = None
        permute_46 = torch.ops.aten.permute.default(primals_70, [1, 0]);  primals_70 = None
        view_46 = torch.ops.aten.view.default(mul_69, [256, 512]);  mul_69 = None
        mm_25 = torch.ops.aten.mm.default(view_46, permute_46)
        _unsafe_view_45 = torch.ops.aten._unsafe_view.default(mm_25, [2, 128, 1024])
        mul_70 = torch.ops.aten.mul.Tensor(_unsafe_view_45, 0.5)
        pow_12 = torch.ops.aten.pow.Tensor_Scalar(_unsafe_view_45, 3.0)
        mul_71 = torch.ops.aten.mul.Tensor(pow_12, 0.044715);  pow_12 = None
        add_32 = torch.ops.aten.add.Tensor(_unsafe_view_45, mul_71);  _unsafe_view_45 = mul_71 = None
        mul_72 = torch.ops.aten.mul.Tensor(add_32, 0.7978845608028654);  add_32 = None
        mul_73 = torch.ops.aten.mul.Tensor(mul_72, -2.0);  mul_72 = None
        exp_7 = torch.ops.aten.exp.default(mul_73);  mul_73 = None
        add_33 = torch.ops.aten.add.Tensor(exp_7, 1.0);  exp_7 = None
        reciprocal_11 = torch.ops.aten.reciprocal.default(add_33);  add_33 = None
        mul_74 = torch.ops.aten.mul.Tensor(reciprocal_11, 2.0);  reciprocal_11 = None
        sub_9 = torch.ops.aten.sub.Tensor(mul_74, 1.0);  mul_74 = None
        add_34 = torch.ops.aten.add.Tensor(sub_9, 1.0)
        mul_75 = torch.ops.aten.mul.Tensor(mul_70, add_34);  mul_70 = add_34 = None
        permute_47 = torch.ops.aten.permute.default(primals_71, [1, 0]);  primals_71 = None
        mm_26 = torch.ops.aten.mm.default(view_46, permute_47);  view_46 = None
        _unsafe_view_46 = torch.ops.aten._unsafe_view.default(mm_26, [2, 128, 1024])
        mul_76 = torch.ops.aten.mul.Tensor(mul_75, _unsafe_view_46);  mul_75 = _unsafe_view_46 = None
        rand_like_11 = torch.ops.aten.rand_like.default(mul_76, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_67 = torch.ops.aten.alias.default(rand_like_11);  rand_like_11 = None
        gt_16 = torch.ops.aten.gt.Scalar(alias_67, 0.1);  alias_67 = None
        mul_77 = torch.ops.aten.mul.Tensor(gt_16, mul_76);  mul_76 = None
        mul_78 = torch.ops.aten.mul.Tensor(mul_77, 1.1111111111111112);  mul_77 = None
        permute_48 = torch.ops.aten.permute.default(primals_72, [1, 0]);  primals_72 = None
        view_48 = torch.ops.aten.view.default(mul_78, [256, 1024]);  mul_78 = None
        mm_27 = torch.ops.aten.mm.default(view_48, permute_48)
        _unsafe_view_47 = torch.ops.aten._unsafe_view.default(mm_27, [2, 128, 512]);  mm_27 = None
        rand_like_12 = torch.ops.aten.rand_like.default(_unsafe_view_47, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_68 = torch.ops.aten.alias.default(rand_like_12);  rand_like_12 = None
        gt_17 = torch.ops.aten.gt.Scalar(alias_68, 0.1);  alias_68 = None
        mul_79 = torch.ops.aten.mul.Tensor(gt_17, _unsafe_view_47);  _unsafe_view_47 = None
        mul_80 = torch.ops.aten.mul.Tensor(mul_79, 1.1111111111111112);  mul_79 = None
        add_35 = torch.ops.aten.add.Tensor(add_30, mul_80);  mul_80 = None
        pow_13 = torch.ops.aten.pow.Tensor_Scalar(add_35, 2)
        mean_8 = torch.ops.aten.mean.dim(pow_13, [-1], True);  pow_13 = None
        add_36 = torch.ops.aten.add.Tensor(mean_8, 1e-06);  mean_8 = None
        sqrt_8 = torch.ops.aten.sqrt.default(add_36);  add_36 = None
        reciprocal_12 = torch.ops.aten.reciprocal.default(sqrt_8);  sqrt_8 = None
        mul_81 = torch.ops.aten.mul.Tensor(add_35, reciprocal_12)
        mul_82 = torch.ops.aten.mul.Tensor(primals_9, mul_81);  mul_81 = None
        permute_49 = torch.ops.aten.permute.default(primals_73, [1, 0]);  primals_73 = None
        view_49 = torch.ops.aten.view.default(mul_82, [256, 512]);  mul_82 = None
        mm_28 = torch.ops.aten.mm.default(view_49, permute_49)
        _unsafe_view_48 = torch.ops.aten._unsafe_view.default(mm_28, [2, 128, 384]);  mm_28 = None
        view_50 = torch.ops.aten.view.default(_unsafe_view_48, [2, -1, 6, 64]);  _unsafe_view_48 = None
        permute_50 = torch.ops.aten.permute.default(view_50, [0, 2, 1, 3]);  view_50 = None
        permute_51 = torch.ops.aten.permute.default(primals_74, [1, 0]);  primals_74 = None
        mm_29 = torch.ops.aten.mm.default(view_49, permute_51)
        _unsafe_view_49 = torch.ops.aten._unsafe_view.default(mm_29, [2, 128, 384]);  mm_29 = None
        view_52 = torch.ops.aten.view.default(_unsafe_view_49, [2, -1, 6, 64]);  _unsafe_view_49 = None
        permute_52 = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None
        permute_53 = torch.ops.aten.permute.default(primals_75, [1, 0]);  primals_75 = None
        mm_30 = torch.ops.aten.mm.default(view_49, permute_53);  view_49 = None
        _unsafe_view_50 = torch.ops.aten._unsafe_view.default(mm_30, [2, 128, 384]);  mm_30 = None
        view_54 = torch.ops.aten.view.default(_unsafe_view_50, [2, -1, 6, 64]);  _unsafe_view_50 = None
        permute_54 = torch.ops.aten.permute.default(view_54, [0, 2, 1, 3]);  view_54 = None
        permute_55 = torch.ops.aten.permute.default(permute_52, [0, 1, 3, 2]);  permute_52 = None
        expand_16 = torch.ops.aten.expand.default(permute_50, [2, 6, 128, 64]);  permute_50 = None
        clone_16 = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
        _unsafe_view_51 = torch.ops.aten._unsafe_view.default(clone_16, [12, 128, 64]);  clone_16 = None
        expand_17 = torch.ops.aten.expand.default(permute_55, [2, 6, 64, 128]);  permute_55 = None
        clone_17 = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
        _unsafe_view_52 = torch.ops.aten._unsafe_view.default(clone_17, [12, 64, 128]);  clone_17 = None
        bmm_8 = torch.ops.aten.bmm.default(_unsafe_view_51, _unsafe_view_52)
        _unsafe_view_53 = torch.ops.aten._unsafe_view.default(bmm_8, [2, 6, 128, 128]);  bmm_8 = None
        add_37 = torch.ops.aten.add.Tensor(_unsafe_view_53, add_4);  _unsafe_view_53 = None
        amax_4 = torch.ops.aten.amax.default(add_37, [-1], True)
        sub_10 = torch.ops.aten.sub.Tensor(add_37, amax_4);  add_37 = amax_4 = None
        exp_8 = torch.ops.aten.exp.default(sub_10);  sub_10 = None
        sum_5 = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_6 = torch.ops.aten.div.Tensor(exp_8, sum_5);  exp_8 = sum_5 = None
        philox_rand_like_4 = torch.ops.prims.philox_rand_like.default(div_6, philox_seed_like, 786432)
        gt_18 = torch.ops.aten.gt.Scalar(philox_rand_like_4, 0.1);  philox_rand_like_4 = None
        _to_copy_7 = torch.ops.aten._to_copy.default(gt_18, dtype = torch.float32);  gt_18 = None
        mul_83 = torch.ops.aten.mul.Tensor(_to_copy_7, div_6);  _to_copy_7 = None
        mul_84 = torch.ops.aten.mul.Tensor(mul_83, 1.1111111111111112);  mul_83 = None
        expand_18 = torch.ops.aten.expand.default(mul_84, [2, 6, 128, 128]);  mul_84 = None
        view_55 = torch.ops.aten.view.default(expand_18, [12, 128, 128]);  expand_18 = None
        expand_19 = torch.ops.aten.expand.default(permute_54, [2, 6, 128, 64]);  permute_54 = None
        clone_18 = torch.ops.aten.clone.default(expand_19, memory_format = torch.contiguous_format);  expand_19 = None
        _unsafe_view_54 = torch.ops.aten._unsafe_view.default(clone_18, [12, 128, 64]);  clone_18 = None
        bmm_9 = torch.ops.aten.bmm.default(view_55, _unsafe_view_54)
        _unsafe_view_55 = torch.ops.aten._unsafe_view.default(bmm_9, [2, 6, 128, 64]);  bmm_9 = None
        permute_56 = torch.ops.aten.permute.default(_unsafe_view_55, [0, 2, 1, 3]);  _unsafe_view_55 = None
        clone_19 = torch.ops.aten.clone.default(permute_56, memory_format = torch.contiguous_format);  permute_56 = None
        view_56 = torch.ops.aten.view.default(clone_19, [2, -1, 384]);  clone_19 = None
        permute_57 = torch.ops.aten.permute.default(primals_76, [1, 0]);  primals_76 = None
        view_57 = torch.ops.aten.view.default(view_56, [256, 384]);  view_56 = None
        mm_31 = torch.ops.aten.mm.default(view_57, permute_57)
        _unsafe_view_56 = torch.ops.aten._unsafe_view.default(mm_31, [2, 128, 512]);  mm_31 = None
        rand_like_13 = torch.ops.aten.rand_like.default(_unsafe_view_56, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_75 = torch.ops.aten.alias.default(rand_like_13);  rand_like_13 = None
        gt_19 = torch.ops.aten.gt.Scalar(alias_75, 0.1);  alias_75 = None
        mul_85 = torch.ops.aten.mul.Tensor(gt_19, _unsafe_view_56);  _unsafe_view_56 = None
        mul_86 = torch.ops.aten.mul.Tensor(mul_85, 1.1111111111111112);  mul_85 = None
        add_38 = torch.ops.aten.add.Tensor(add_35, mul_86);  mul_86 = None
        pow_14 = torch.ops.aten.pow.Tensor_Scalar(add_38, 2)
        mean_9 = torch.ops.aten.mean.dim(pow_14, [-1], True);  pow_14 = None
        add_39 = torch.ops.aten.add.Tensor(mean_9, 1e-06);  mean_9 = None
        sqrt_9 = torch.ops.aten.sqrt.default(add_39);  add_39 = None
        reciprocal_13 = torch.ops.aten.reciprocal.default(sqrt_9);  sqrt_9 = None
        mul_87 = torch.ops.aten.mul.Tensor(add_38, reciprocal_13)
        mul_88 = torch.ops.aten.mul.Tensor(primals_10, mul_87);  mul_87 = None
        permute_58 = torch.ops.aten.permute.default(primals_77, [1, 0]);  primals_77 = None
        view_58 = torch.ops.aten.view.default(mul_88, [256, 512]);  mul_88 = None
        mm_32 = torch.ops.aten.mm.default(view_58, permute_58)
        _unsafe_view_57 = torch.ops.aten._unsafe_view.default(mm_32, [2, 128, 1024])
        mul_89 = torch.ops.aten.mul.Tensor(_unsafe_view_57, 0.5)
        pow_15 = torch.ops.aten.pow.Tensor_Scalar(_unsafe_view_57, 3.0)
        mul_90 = torch.ops.aten.mul.Tensor(pow_15, 0.044715);  pow_15 = None
        add_40 = torch.ops.aten.add.Tensor(_unsafe_view_57, mul_90);  _unsafe_view_57 = mul_90 = None
        mul_91 = torch.ops.aten.mul.Tensor(add_40, 0.7978845608028654);  add_40 = None
        mul_92 = torch.ops.aten.mul.Tensor(mul_91, -2.0);  mul_91 = None
        exp_9 = torch.ops.aten.exp.default(mul_92);  mul_92 = None
        add_41 = torch.ops.aten.add.Tensor(exp_9, 1.0);  exp_9 = None
        reciprocal_14 = torch.ops.aten.reciprocal.default(add_41);  add_41 = None
        mul_93 = torch.ops.aten.mul.Tensor(reciprocal_14, 2.0);  reciprocal_14 = None
        sub_11 = torch.ops.aten.sub.Tensor(mul_93, 1.0);  mul_93 = None
        add_42 = torch.ops.aten.add.Tensor(sub_11, 1.0)
        mul_94 = torch.ops.aten.mul.Tensor(mul_89, add_42);  mul_89 = add_42 = None
        permute_59 = torch.ops.aten.permute.default(primals_78, [1, 0]);  primals_78 = None
        mm_33 = torch.ops.aten.mm.default(view_58, permute_59);  view_58 = None
        _unsafe_view_58 = torch.ops.aten._unsafe_view.default(mm_33, [2, 128, 1024])
        mul_95 = torch.ops.aten.mul.Tensor(mul_94, _unsafe_view_58);  mul_94 = _unsafe_view_58 = None
        rand_like_14 = torch.ops.aten.rand_like.default(mul_95, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_82 = torch.ops.aten.alias.default(rand_like_14);  rand_like_14 = None
        gt_20 = torch.ops.aten.gt.Scalar(alias_82, 0.1);  alias_82 = None
        mul_96 = torch.ops.aten.mul.Tensor(gt_20, mul_95);  mul_95 = None
        mul_97 = torch.ops.aten.mul.Tensor(mul_96, 1.1111111111111112);  mul_96 = None
        permute_60 = torch.ops.aten.permute.default(primals_79, [1, 0]);  primals_79 = None
        view_60 = torch.ops.aten.view.default(mul_97, [256, 1024]);  mul_97 = None
        mm_34 = torch.ops.aten.mm.default(view_60, permute_60)
        _unsafe_view_59 = torch.ops.aten._unsafe_view.default(mm_34, [2, 128, 512]);  mm_34 = None
        rand_like_15 = torch.ops.aten.rand_like.default(_unsafe_view_59, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_83 = torch.ops.aten.alias.default(rand_like_15);  rand_like_15 = None
        gt_21 = torch.ops.aten.gt.Scalar(alias_83, 0.1);  alias_83 = None
        mul_98 = torch.ops.aten.mul.Tensor(gt_21, _unsafe_view_59);  _unsafe_view_59 = None
        mul_99 = torch.ops.aten.mul.Tensor(mul_98, 1.1111111111111112);  mul_98 = None
        add_43 = torch.ops.aten.add.Tensor(add_38, mul_99);  mul_99 = None
        pow_16 = torch.ops.aten.pow.Tensor_Scalar(add_43, 2)
        mean_10 = torch.ops.aten.mean.dim(pow_16, [-1], True);  pow_16 = None
        add_44 = torch.ops.aten.add.Tensor(mean_10, 1e-06);  mean_10 = None
        sqrt_10 = torch.ops.aten.sqrt.default(add_44);  add_44 = None
        reciprocal_15 = torch.ops.aten.reciprocal.default(sqrt_10);  sqrt_10 = None
        mul_100 = torch.ops.aten.mul.Tensor(add_43, reciprocal_15)
        mul_101 = torch.ops.aten.mul.Tensor(primals_11, mul_100);  mul_100 = None
        permute_61 = torch.ops.aten.permute.default(primals_80, [1, 0]);  primals_80 = None
        view_61 = torch.ops.aten.view.default(mul_101, [256, 512]);  mul_101 = None
        mm_35 = torch.ops.aten.mm.default(view_61, permute_61)
        _unsafe_view_60 = torch.ops.aten._unsafe_view.default(mm_35, [2, 128, 384]);  mm_35 = None
        view_62 = torch.ops.aten.view.default(_unsafe_view_60, [2, -1, 6, 64]);  _unsafe_view_60 = None
        permute_62 = torch.ops.aten.permute.default(view_62, [0, 2, 1, 3]);  view_62 = None
        permute_63 = torch.ops.aten.permute.default(primals_81, [1, 0]);  primals_81 = None
        mm_36 = torch.ops.aten.mm.default(view_61, permute_63)
        _unsafe_view_61 = torch.ops.aten._unsafe_view.default(mm_36, [2, 128, 384]);  mm_36 = None
        view_64 = torch.ops.aten.view.default(_unsafe_view_61, [2, -1, 6, 64]);  _unsafe_view_61 = None
        permute_64 = torch.ops.aten.permute.default(view_64, [0, 2, 1, 3]);  view_64 = None
        permute_65 = torch.ops.aten.permute.default(primals_82, [1, 0]);  primals_82 = None
        mm_37 = torch.ops.aten.mm.default(view_61, permute_65);  view_61 = None
        _unsafe_view_62 = torch.ops.aten._unsafe_view.default(mm_37, [2, 128, 384]);  mm_37 = None
        view_66 = torch.ops.aten.view.default(_unsafe_view_62, [2, -1, 6, 64]);  _unsafe_view_62 = None
        permute_66 = torch.ops.aten.permute.default(view_66, [0, 2, 1, 3]);  view_66 = None
        permute_67 = torch.ops.aten.permute.default(permute_64, [0, 1, 3, 2]);  permute_64 = None
        expand_20 = torch.ops.aten.expand.default(permute_62, [2, 6, 128, 64]);  permute_62 = None
        clone_20 = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
        _unsafe_view_63 = torch.ops.aten._unsafe_view.default(clone_20, [12, 128, 64]);  clone_20 = None
        expand_21 = torch.ops.aten.expand.default(permute_67, [2, 6, 64, 128]);  permute_67 = None
        clone_21 = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
        _unsafe_view_64 = torch.ops.aten._unsafe_view.default(clone_21, [12, 64, 128]);  clone_21 = None
        bmm_10 = torch.ops.aten.bmm.default(_unsafe_view_63, _unsafe_view_64)
        _unsafe_view_65 = torch.ops.aten._unsafe_view.default(bmm_10, [2, 6, 128, 128]);  bmm_10 = None
        add_45 = torch.ops.aten.add.Tensor(_unsafe_view_65, add_4);  _unsafe_view_65 = None
        amax_5 = torch.ops.aten.amax.default(add_45, [-1], True)
        sub_12 = torch.ops.aten.sub.Tensor(add_45, amax_5);  add_45 = amax_5 = None
        exp_10 = torch.ops.aten.exp.default(sub_12);  sub_12 = None
        sum_6 = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_7 = torch.ops.aten.div.Tensor(exp_10, sum_6);  exp_10 = sum_6 = None
        philox_rand_like_5 = torch.ops.prims.philox_rand_like.default(div_7, philox_seed_like, 983040)
        gt_22 = torch.ops.aten.gt.Scalar(philox_rand_like_5, 0.1);  philox_rand_like_5 = None
        _to_copy_8 = torch.ops.aten._to_copy.default(gt_22, dtype = torch.float32);  gt_22 = None
        mul_102 = torch.ops.aten.mul.Tensor(_to_copy_8, div_7);  _to_copy_8 = None
        mul_103 = torch.ops.aten.mul.Tensor(mul_102, 1.1111111111111112);  mul_102 = None
        expand_22 = torch.ops.aten.expand.default(mul_103, [2, 6, 128, 128]);  mul_103 = None
        view_67 = torch.ops.aten.view.default(expand_22, [12, 128, 128]);  expand_22 = None
        expand_23 = torch.ops.aten.expand.default(permute_66, [2, 6, 128, 64]);  permute_66 = None
        clone_22 = torch.ops.aten.clone.default(expand_23, memory_format = torch.contiguous_format);  expand_23 = None
        _unsafe_view_66 = torch.ops.aten._unsafe_view.default(clone_22, [12, 128, 64]);  clone_22 = None
        bmm_11 = torch.ops.aten.bmm.default(view_67, _unsafe_view_66)
        _unsafe_view_67 = torch.ops.aten._unsafe_view.default(bmm_11, [2, 6, 128, 64]);  bmm_11 = None
        permute_68 = torch.ops.aten.permute.default(_unsafe_view_67, [0, 2, 1, 3]);  _unsafe_view_67 = None
        clone_23 = torch.ops.aten.clone.default(permute_68, memory_format = torch.contiguous_format);  permute_68 = None
        view_68 = torch.ops.aten.view.default(clone_23, [2, -1, 384]);  clone_23 = None
        permute_69 = torch.ops.aten.permute.default(primals_83, [1, 0]);  primals_83 = None
        view_69 = torch.ops.aten.view.default(view_68, [256, 384]);  view_68 = None
        mm_38 = torch.ops.aten.mm.default(view_69, permute_69)
        _unsafe_view_68 = torch.ops.aten._unsafe_view.default(mm_38, [2, 128, 512]);  mm_38 = None
        rand_like_16 = torch.ops.aten.rand_like.default(_unsafe_view_68, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_90 = torch.ops.aten.alias.default(rand_like_16);  rand_like_16 = None
        gt_23 = torch.ops.aten.gt.Scalar(alias_90, 0.1);  alias_90 = None
        mul_104 = torch.ops.aten.mul.Tensor(gt_23, _unsafe_view_68);  _unsafe_view_68 = None
        mul_105 = torch.ops.aten.mul.Tensor(mul_104, 1.1111111111111112);  mul_104 = None
        add_46 = torch.ops.aten.add.Tensor(add_43, mul_105);  mul_105 = None
        pow_17 = torch.ops.aten.pow.Tensor_Scalar(add_46, 2)
        mean_11 = torch.ops.aten.mean.dim(pow_17, [-1], True);  pow_17 = None
        add_47 = torch.ops.aten.add.Tensor(mean_11, 1e-06);  mean_11 = None
        sqrt_11 = torch.ops.aten.sqrt.default(add_47);  add_47 = None
        reciprocal_16 = torch.ops.aten.reciprocal.default(sqrt_11);  sqrt_11 = None
        mul_106 = torch.ops.aten.mul.Tensor(add_46, reciprocal_16)
        mul_107 = torch.ops.aten.mul.Tensor(primals_12, mul_106);  mul_106 = None
        permute_70 = torch.ops.aten.permute.default(primals_84, [1, 0]);  primals_84 = None
        view_70 = torch.ops.aten.view.default(mul_107, [256, 512]);  mul_107 = None
        mm_39 = torch.ops.aten.mm.default(view_70, permute_70)
        _unsafe_view_69 = torch.ops.aten._unsafe_view.default(mm_39, [2, 128, 1024])
        mul_108 = torch.ops.aten.mul.Tensor(_unsafe_view_69, 0.5)
        pow_18 = torch.ops.aten.pow.Tensor_Scalar(_unsafe_view_69, 3.0)
        mul_109 = torch.ops.aten.mul.Tensor(pow_18, 0.044715);  pow_18 = None
        add_48 = torch.ops.aten.add.Tensor(_unsafe_view_69, mul_109);  _unsafe_view_69 = mul_109 = None
        mul_110 = torch.ops.aten.mul.Tensor(add_48, 0.7978845608028654);  add_48 = None
        mul_111 = torch.ops.aten.mul.Tensor(mul_110, -2.0);  mul_110 = None
        exp_11 = torch.ops.aten.exp.default(mul_111);  mul_111 = None
        add_49 = torch.ops.aten.add.Tensor(exp_11, 1.0);  exp_11 = None
        reciprocal_17 = torch.ops.aten.reciprocal.default(add_49);  add_49 = None
        mul_112 = torch.ops.aten.mul.Tensor(reciprocal_17, 2.0);  reciprocal_17 = None
        sub_13 = torch.ops.aten.sub.Tensor(mul_112, 1.0);  mul_112 = None
        add_50 = torch.ops.aten.add.Tensor(sub_13, 1.0)
        mul_113 = torch.ops.aten.mul.Tensor(mul_108, add_50);  mul_108 = add_50 = None
        permute_71 = torch.ops.aten.permute.default(primals_85, [1, 0]);  primals_85 = None
        mm_40 = torch.ops.aten.mm.default(view_70, permute_71);  view_70 = None
        _unsafe_view_70 = torch.ops.aten._unsafe_view.default(mm_40, [2, 128, 1024])
        mul_114 = torch.ops.aten.mul.Tensor(mul_113, _unsafe_view_70);  mul_113 = _unsafe_view_70 = None
        rand_like_17 = torch.ops.aten.rand_like.default(mul_114, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_97 = torch.ops.aten.alias.default(rand_like_17);  rand_like_17 = None
        gt_24 = torch.ops.aten.gt.Scalar(alias_97, 0.1);  alias_97 = None
        mul_115 = torch.ops.aten.mul.Tensor(gt_24, mul_114);  mul_114 = None
        mul_116 = torch.ops.aten.mul.Tensor(mul_115, 1.1111111111111112);  mul_115 = None
        permute_72 = torch.ops.aten.permute.default(primals_86, [1, 0]);  primals_86 = None
        view_72 = torch.ops.aten.view.default(mul_116, [256, 1024]);  mul_116 = None
        mm_41 = torch.ops.aten.mm.default(view_72, permute_72)
        _unsafe_view_71 = torch.ops.aten._unsafe_view.default(mm_41, [2, 128, 512]);  mm_41 = None
        rand_like_18 = torch.ops.aten.rand_like.default(_unsafe_view_71, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_98 = torch.ops.aten.alias.default(rand_like_18);  rand_like_18 = None
        gt_25 = torch.ops.aten.gt.Scalar(alias_98, 0.1);  alias_98 = None
        mul_117 = torch.ops.aten.mul.Tensor(gt_25, _unsafe_view_71);  _unsafe_view_71 = None
        mul_118 = torch.ops.aten.mul.Tensor(mul_117, 1.1111111111111112);  mul_117 = None
        add_51 = torch.ops.aten.add.Tensor(add_46, mul_118);  mul_118 = None
        pow_19 = torch.ops.aten.pow.Tensor_Scalar(add_51, 2)
        mean_12 = torch.ops.aten.mean.dim(pow_19, [-1], True);  pow_19 = None
        add_52 = torch.ops.aten.add.Tensor(mean_12, 1e-06);  mean_12 = None
        sqrt_12 = torch.ops.aten.sqrt.default(add_52);  add_52 = None
        reciprocal_18 = torch.ops.aten.reciprocal.default(sqrt_12);  sqrt_12 = None
        mul_119 = torch.ops.aten.mul.Tensor(add_51, reciprocal_18)
        mul_120 = torch.ops.aten.mul.Tensor(primals_13, mul_119);  mul_119 = None
        permute_73 = torch.ops.aten.permute.default(primals_87, [1, 0]);  primals_87 = None
        view_73 = torch.ops.aten.view.default(mul_120, [256, 512]);  mul_120 = None
        mm_42 = torch.ops.aten.mm.default(view_73, permute_73)
        _unsafe_view_72 = torch.ops.aten._unsafe_view.default(mm_42, [2, 128, 384]);  mm_42 = None
        view_74 = torch.ops.aten.view.default(_unsafe_view_72, [2, -1, 6, 64]);  _unsafe_view_72 = None
        permute_74 = torch.ops.aten.permute.default(view_74, [0, 2, 1, 3]);  view_74 = None
        permute_75 = torch.ops.aten.permute.default(primals_88, [1, 0]);  primals_88 = None
        mm_43 = torch.ops.aten.mm.default(view_73, permute_75)
        _unsafe_view_73 = torch.ops.aten._unsafe_view.default(mm_43, [2, 128, 384]);  mm_43 = None
        view_76 = torch.ops.aten.view.default(_unsafe_view_73, [2, -1, 6, 64]);  _unsafe_view_73 = None
        permute_76 = torch.ops.aten.permute.default(view_76, [0, 2, 1, 3]);  view_76 = None
        permute_77 = torch.ops.aten.permute.default(primals_89, [1, 0]);  primals_89 = None
        mm_44 = torch.ops.aten.mm.default(view_73, permute_77);  view_73 = None
        _unsafe_view_74 = torch.ops.aten._unsafe_view.default(mm_44, [2, 128, 384]);  mm_44 = None
        view_78 = torch.ops.aten.view.default(_unsafe_view_74, [2, -1, 6, 64]);  _unsafe_view_74 = None
        permute_78 = torch.ops.aten.permute.default(view_78, [0, 2, 1, 3]);  view_78 = None
        permute_79 = torch.ops.aten.permute.default(permute_76, [0, 1, 3, 2]);  permute_76 = None
        expand_24 = torch.ops.aten.expand.default(permute_74, [2, 6, 128, 64]);  permute_74 = None
        clone_24 = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
        _unsafe_view_75 = torch.ops.aten._unsafe_view.default(clone_24, [12, 128, 64]);  clone_24 = None
        expand_25 = torch.ops.aten.expand.default(permute_79, [2, 6, 64, 128]);  permute_79 = None
        clone_25 = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
        _unsafe_view_76 = torch.ops.aten._unsafe_view.default(clone_25, [12, 64, 128]);  clone_25 = None
        bmm_12 = torch.ops.aten.bmm.default(_unsafe_view_75, _unsafe_view_76)
        _unsafe_view_77 = torch.ops.aten._unsafe_view.default(bmm_12, [2, 6, 128, 128]);  bmm_12 = None
        add_53 = torch.ops.aten.add.Tensor(_unsafe_view_77, add_4);  _unsafe_view_77 = None
        amax_6 = torch.ops.aten.amax.default(add_53, [-1], True)
        sub_14 = torch.ops.aten.sub.Tensor(add_53, amax_6);  add_53 = amax_6 = None
        exp_12 = torch.ops.aten.exp.default(sub_14);  sub_14 = None
        sum_7 = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
        div_8 = torch.ops.aten.div.Tensor(exp_12, sum_7);  exp_12 = sum_7 = None
        philox_rand_like_6 = torch.ops.prims.philox_rand_like.default(div_8, philox_seed_like, 1179648)
        gt_26 = torch.ops.aten.gt.Scalar(philox_rand_like_6, 0.1);  philox_rand_like_6 = None
        _to_copy_9 = torch.ops.aten._to_copy.default(gt_26, dtype = torch.float32);  gt_26 = None
        mul_121 = torch.ops.aten.mul.Tensor(_to_copy_9, div_8);  _to_copy_9 = None
        mul_122 = torch.ops.aten.mul.Tensor(mul_121, 1.1111111111111112);  mul_121 = None
        expand_26 = torch.ops.aten.expand.default(mul_122, [2, 6, 128, 128]);  mul_122 = None
        view_79 = torch.ops.aten.view.default(expand_26, [12, 128, 128]);  expand_26 = None
        expand_27 = torch.ops.aten.expand.default(permute_78, [2, 6, 128, 64]);  permute_78 = None
        clone_26 = torch.ops.aten.clone.default(expand_27, memory_format = torch.contiguous_format);  expand_27 = None
        _unsafe_view_78 = torch.ops.aten._unsafe_view.default(clone_26, [12, 128, 64]);  clone_26 = None
        bmm_13 = torch.ops.aten.bmm.default(view_79, _unsafe_view_78)
        _unsafe_view_79 = torch.ops.aten._unsafe_view.default(bmm_13, [2, 6, 128, 64]);  bmm_13 = None
        permute_80 = torch.ops.aten.permute.default(_unsafe_view_79, [0, 2, 1, 3]);  _unsafe_view_79 = None
        clone_27 = torch.ops.aten.clone.default(permute_80, memory_format = torch.contiguous_format);  permute_80 = None
        view_80 = torch.ops.aten.view.default(clone_27, [2, -1, 384]);  clone_27 = None
        permute_81 = torch.ops.aten.permute.default(primals_90, [1, 0]);  primals_90 = None
        view_81 = torch.ops.aten.view.default(view_80, [256, 384]);  view_80 = None
        mm_45 = torch.ops.aten.mm.default(view_81, permute_81)
        _unsafe_view_80 = torch.ops.aten._unsafe_view.default(mm_45, [2, 128, 512]);  mm_45 = None
        rand_like_19 = torch.ops.aten.rand_like.default(_unsafe_view_80, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_105 = torch.ops.aten.alias.default(rand_like_19);  rand_like_19 = None
        gt_27 = torch.ops.aten.gt.Scalar(alias_105, 0.1);  alias_105 = None
        mul_123 = torch.ops.aten.mul.Tensor(gt_27, _unsafe_view_80);  _unsafe_view_80 = None
        mul_124 = torch.ops.aten.mul.Tensor(mul_123, 1.1111111111111112);  mul_123 = None
        add_54 = torch.ops.aten.add.Tensor(add_51, mul_124);  mul_124 = None
        pow_20 = torch.ops.aten.pow.Tensor_Scalar(add_54, 2)
        mean_13 = torch.ops.aten.mean.dim(pow_20, [-1], True);  pow_20 = None
        add_55 = torch.ops.aten.add.Tensor(mean_13, 1e-06);  mean_13 = None
        sqrt_13 = torch.ops.aten.sqrt.default(add_55);  add_55 = None
        reciprocal_19 = torch.ops.aten.reciprocal.default(sqrt_13);  sqrt_13 = None
        mul_125 = torch.ops.aten.mul.Tensor(add_54, reciprocal_19)
        mul_126 = torch.ops.aten.mul.Tensor(primals_14, mul_125);  mul_125 = None
        permute_82 = torch.ops.aten.permute.default(primals_91, [1, 0]);  primals_91 = None
        view_82 = torch.ops.aten.view.default(mul_126, [256, 512]);  mul_126 = None
        mm_46 = torch.ops.aten.mm.default(view_82, permute_82)
        _unsafe_view_81 = torch.ops.aten._unsafe_view.default(mm_46, [2, 128, 1024])
        mul_127 = torch.ops.aten.mul.Tensor(_unsafe_view_81, 0.5)
        pow_21 = torch.ops.aten.pow.Tensor_Scalar(_unsafe_view_81, 3.0)
        mul_128 = torch.ops.aten.mul.Tensor(pow_21, 0.044715);  pow_21 = None
        add_56 = torch.ops.aten.add.Tensor(_unsafe_view_81, mul_128);  _unsafe_view_81 = mul_128 = None
        mul_129 = torch.ops.aten.mul.Tensor(add_56, 0.7978845608028654);  add_56 = None
        mul_130 = torch.ops.aten.mul.Tensor(mul_129, -2.0);  mul_129 = None
        exp_13 = torch.ops.aten.exp.default(mul_130);  mul_130 = None
        add_57 = torch.ops.aten.add.Tensor(exp_13, 1.0);  exp_13 = None
        reciprocal_20 = torch.ops.aten.reciprocal.default(add_57);  add_57 = None
        mul_131 = torch.ops.aten.mul.Tensor(reciprocal_20, 2.0);  reciprocal_20 = None
        sub_15 = torch.ops.aten.sub.Tensor(mul_131, 1.0);  mul_131 = None
        add_58 = torch.ops.aten.add.Tensor(sub_15, 1.0)
        mul_132 = torch.ops.aten.mul.Tensor(mul_127, add_58);  mul_127 = add_58 = None
        permute_83 = torch.ops.aten.permute.default(primals_92, [1, 0]);  primals_92 = None
        mm_47 = torch.ops.aten.mm.default(view_82, permute_83);  view_82 = None
        _unsafe_view_82 = torch.ops.aten._unsafe_view.default(mm_47, [2, 128, 1024])
        mul_133 = torch.ops.aten.mul.Tensor(mul_132, _unsafe_view_82);  mul_132 = _unsafe_view_82 = None
        rand_like_20 = torch.ops.aten.rand_like.default(mul_133, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_112 = torch.ops.aten.alias.default(rand_like_20);  rand_like_20 = None
        gt_28 = torch.ops.aten.gt.Scalar(alias_112, 0.1);  alias_112 = None
        mul_134 = torch.ops.aten.mul.Tensor(gt_28, mul_133);  mul_133 = None
        mul_135 = torch.ops.aten.mul.Tensor(mul_134, 1.1111111111111112);  mul_134 = None
        permute_84 = torch.ops.aten.permute.default(primals_93, [1, 0]);  primals_93 = None
        view_84 = torch.ops.aten.view.default(mul_135, [256, 1024]);  mul_135 = None
        mm_48 = torch.ops.aten.mm.default(view_84, permute_84)
        _unsafe_view_83 = torch.ops.aten._unsafe_view.default(mm_48, [2, 128, 512]);  mm_48 = None
        rand_like_21 = torch.ops.aten.rand_like.default(_unsafe_view_83, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_113 = torch.ops.aten.alias.default(rand_like_21);  rand_like_21 = None
        gt_29 = torch.ops.aten.gt.Scalar(alias_113, 0.1);  alias_113 = None
        mul_136 = torch.ops.aten.mul.Tensor(gt_29, _unsafe_view_83);  _unsafe_view_83 = None
        mul_137 = torch.ops.aten.mul.Tensor(mul_136, 1.1111111111111112);  mul_136 = None
        add_59 = torch.ops.aten.add.Tensor(add_54, mul_137);  mul_137 = None
        pow_22 = torch.ops.aten.pow.Tensor_Scalar(add_59, 2)
        mean_14 = torch.ops.aten.mean.dim(pow_22, [-1], True);  pow_22 = None
        add_60 = torch.ops.aten.add.Tensor(mean_14, 1e-06);  mean_14 = None
        sqrt_14 = torch.ops.aten.sqrt.default(add_60);  add_60 = None
        reciprocal_21 = torch.ops.aten.reciprocal.default(sqrt_14);  sqrt_14 = None
        mul_138 = torch.ops.aten.mul.Tensor(add_59, reciprocal_21)
        mul_139 = torch.ops.aten.mul.Tensor(primals_15, mul_138);  mul_138 = None
        permute_85 = torch.ops.aten.permute.default(primals_94, [1, 0]);  primals_94 = None
        view_85 = torch.ops.aten.view.default(mul_139, [256, 512]);  mul_139 = None
        mm_49 = torch.ops.aten.mm.default(view_85, permute_85)
        _unsafe_view_84 = torch.ops.aten._unsafe_view.default(mm_49, [2, 128, 384]);  mm_49 = None
        view_86 = torch.ops.aten.view.default(_unsafe_view_84, [2, -1, 6, 64]);  _unsafe_view_84 = None
        permute_86 = torch.ops.aten.permute.default(view_86, [0, 2, 1, 3]);  view_86 = None
        permute_87 = torch.ops.aten.permute.default(primals_95, [1, 0]);  primals_95 = None
        mm_50 = torch.ops.aten.mm.default(view_85, permute_87)
        _unsafe_view_85 = torch.ops.aten._unsafe_view.default(mm_50, [2, 128, 384]);  mm_50 = None
        view_88 = torch.ops.aten.view.default(_unsafe_view_85, [2, -1, 6, 64]);  _unsafe_view_85 = None
        permute_88 = torch.ops.aten.permute.default(view_88, [0, 2, 1, 3]);  view_88 = None
        permute_89 = torch.ops.aten.permute.default(primals_96, [1, 0]);  primals_96 = None
        mm_51 = torch.ops.aten.mm.default(view_85, permute_89);  view_85 = None
        _unsafe_view_86 = torch.ops.aten._unsafe_view.default(mm_51, [2, 128, 384]);  mm_51 = None
        view_90 = torch.ops.aten.view.default(_unsafe_view_86, [2, -1, 6, 64]);  _unsafe_view_86 = None
        permute_90 = torch.ops.aten.permute.default(view_90, [0, 2, 1, 3]);  view_90 = None
        permute_91 = torch.ops.aten.permute.default(permute_88, [0, 1, 3, 2]);  permute_88 = None
        expand_28 = torch.ops.aten.expand.default(permute_86, [2, 6, 128, 64]);  permute_86 = None
        clone_28 = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
        _unsafe_view_87 = torch.ops.aten._unsafe_view.default(clone_28, [12, 128, 64]);  clone_28 = None
        expand_29 = torch.ops.aten.expand.default(permute_91, [2, 6, 64, 128]);  permute_91 = None
        clone_29 = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
        _unsafe_view_88 = torch.ops.aten._unsafe_view.default(clone_29, [12, 64, 128]);  clone_29 = None
        bmm_14 = torch.ops.aten.bmm.default(_unsafe_view_87, _unsafe_view_88)
        _unsafe_view_89 = torch.ops.aten._unsafe_view.default(bmm_14, [2, 6, 128, 128]);  bmm_14 = None
        add_61 = torch.ops.aten.add.Tensor(_unsafe_view_89, add_4);  _unsafe_view_89 = add_4 = None
        amax_7 = torch.ops.aten.amax.default(add_61, [-1], True)
        sub_16 = torch.ops.aten.sub.Tensor(add_61, amax_7);  add_61 = amax_7 = None
        exp_14 = torch.ops.aten.exp.default(sub_16);  sub_16 = None
        sum_8 = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
        div_9 = torch.ops.aten.div.Tensor(exp_14, sum_8);  exp_14 = sum_8 = None
        philox_rand_like_7 = torch.ops.prims.philox_rand_like.default(div_9, philox_seed_like, 1376256)
        gt_30 = torch.ops.aten.gt.Scalar(philox_rand_like_7, 0.1);  philox_rand_like_7 = None
        _to_copy_10 = torch.ops.aten._to_copy.default(gt_30, dtype = torch.float32);  gt_30 = None
        mul_140 = torch.ops.aten.mul.Tensor(_to_copy_10, div_9);  _to_copy_10 = None
        mul_141 = torch.ops.aten.mul.Tensor(mul_140, 1.1111111111111112);  mul_140 = None
        expand_30 = torch.ops.aten.expand.default(mul_141, [2, 6, 128, 128]);  mul_141 = None
        view_91 = torch.ops.aten.view.default(expand_30, [12, 128, 128]);  expand_30 = None
        expand_31 = torch.ops.aten.expand.default(permute_90, [2, 6, 128, 64]);  permute_90 = None
        clone_30 = torch.ops.aten.clone.default(expand_31, memory_format = torch.contiguous_format);  expand_31 = None
        _unsafe_view_90 = torch.ops.aten._unsafe_view.default(clone_30, [12, 128, 64]);  clone_30 = None
        bmm_15 = torch.ops.aten.bmm.default(view_91, _unsafe_view_90)
        _unsafe_view_91 = torch.ops.aten._unsafe_view.default(bmm_15, [2, 6, 128, 64]);  bmm_15 = None
        permute_92 = torch.ops.aten.permute.default(_unsafe_view_91, [0, 2, 1, 3]);  _unsafe_view_91 = None
        clone_31 = torch.ops.aten.clone.default(permute_92, memory_format = torch.contiguous_format);  permute_92 = None
        view_92 = torch.ops.aten.view.default(clone_31, [2, -1, 384]);  clone_31 = None
        permute_93 = torch.ops.aten.permute.default(primals_97, [1, 0]);  primals_97 = None
        view_93 = torch.ops.aten.view.default(view_92, [256, 384]);  view_92 = None
        mm_52 = torch.ops.aten.mm.default(view_93, permute_93)
        _unsafe_view_92 = torch.ops.aten._unsafe_view.default(mm_52, [2, 128, 512]);  mm_52 = None
        rand_like_22 = torch.ops.aten.rand_like.default(_unsafe_view_92, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_120 = torch.ops.aten.alias.default(rand_like_22);  rand_like_22 = None
        gt_31 = torch.ops.aten.gt.Scalar(alias_120, 0.1);  alias_120 = None
        mul_142 = torch.ops.aten.mul.Tensor(gt_31, _unsafe_view_92);  _unsafe_view_92 = None
        mul_143 = torch.ops.aten.mul.Tensor(mul_142, 1.1111111111111112);  mul_142 = None
        add_62 = torch.ops.aten.add.Tensor(add_59, mul_143);  mul_143 = None
        pow_23 = torch.ops.aten.pow.Tensor_Scalar(add_62, 2)
        mean_15 = torch.ops.aten.mean.dim(pow_23, [-1], True);  pow_23 = None
        add_63 = torch.ops.aten.add.Tensor(mean_15, 1e-06);  mean_15 = None
        sqrt_15 = torch.ops.aten.sqrt.default(add_63);  add_63 = None
        reciprocal_22 = torch.ops.aten.reciprocal.default(sqrt_15);  sqrt_15 = None
        mul_144 = torch.ops.aten.mul.Tensor(add_62, reciprocal_22)
        mul_145 = torch.ops.aten.mul.Tensor(primals_16, mul_144);  mul_144 = None
        permute_94 = torch.ops.aten.permute.default(primals_98, [1, 0]);  primals_98 = None
        view_94 = torch.ops.aten.view.default(mul_145, [256, 512]);  mul_145 = None
        mm_53 = torch.ops.aten.mm.default(view_94, permute_94)
        _unsafe_view_93 = torch.ops.aten._unsafe_view.default(mm_53, [2, 128, 1024])
        mul_146 = torch.ops.aten.mul.Tensor(_unsafe_view_93, 0.5)
        pow_24 = torch.ops.aten.pow.Tensor_Scalar(_unsafe_view_93, 3.0)
        mul_147 = torch.ops.aten.mul.Tensor(pow_24, 0.044715);  pow_24 = None
        add_64 = torch.ops.aten.add.Tensor(_unsafe_view_93, mul_147);  _unsafe_view_93 = mul_147 = None
        mul_148 = torch.ops.aten.mul.Tensor(add_64, 0.7978845608028654);  add_64 = None
        mul_149 = torch.ops.aten.mul.Tensor(mul_148, -2.0);  mul_148 = None
        exp_15 = torch.ops.aten.exp.default(mul_149);  mul_149 = None
        add_65 = torch.ops.aten.add.Tensor(exp_15, 1.0);  exp_15 = None
        reciprocal_23 = torch.ops.aten.reciprocal.default(add_65);  add_65 = None
        mul_150 = torch.ops.aten.mul.Tensor(reciprocal_23, 2.0);  reciprocal_23 = None
        sub_17 = torch.ops.aten.sub.Tensor(mul_150, 1.0);  mul_150 = None
        add_66 = torch.ops.aten.add.Tensor(sub_17, 1.0)
        mul_151 = torch.ops.aten.mul.Tensor(mul_146, add_66);  mul_146 = add_66 = None
        permute_95 = torch.ops.aten.permute.default(primals_99, [1, 0]);  primals_99 = None
        mm_54 = torch.ops.aten.mm.default(view_94, permute_95);  view_94 = None
        _unsafe_view_94 = torch.ops.aten._unsafe_view.default(mm_54, [2, 128, 1024])
        mul_152 = torch.ops.aten.mul.Tensor(mul_151, _unsafe_view_94);  mul_151 = _unsafe_view_94 = None
        rand_like_23 = torch.ops.aten.rand_like.default(mul_152, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_127 = torch.ops.aten.alias.default(rand_like_23);  rand_like_23 = None
        gt_32 = torch.ops.aten.gt.Scalar(alias_127, 0.1);  alias_127 = None
        mul_153 = torch.ops.aten.mul.Tensor(gt_32, mul_152);  mul_152 = None
        mul_154 = torch.ops.aten.mul.Tensor(mul_153, 1.1111111111111112);  mul_153 = None
        permute_96 = torch.ops.aten.permute.default(primals_100, [1, 0]);  primals_100 = None
        view_96 = torch.ops.aten.view.default(mul_154, [256, 1024]);  mul_154 = None
        mm_55 = torch.ops.aten.mm.default(view_96, permute_96)
        _unsafe_view_95 = torch.ops.aten._unsafe_view.default(mm_55, [2, 128, 512]);  mm_55 = None
        rand_like_24 = torch.ops.aten.rand_like.default(_unsafe_view_95, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_128 = torch.ops.aten.alias.default(rand_like_24);  rand_like_24 = None
        gt_33 = torch.ops.aten.gt.Scalar(alias_128, 0.1);  alias_128 = None
        mul_155 = torch.ops.aten.mul.Tensor(gt_33, _unsafe_view_95);  _unsafe_view_95 = None
        mul_156 = torch.ops.aten.mul.Tensor(mul_155, 1.1111111111111112);  mul_155 = None
        add_67 = torch.ops.aten.add.Tensor(add_62, mul_156);  mul_156 = None
        pow_25 = torch.ops.aten.pow.Tensor_Scalar(add_67, 2)
        mean_16 = torch.ops.aten.mean.dim(pow_25, [-1], True);  pow_25 = None
        add_68 = torch.ops.aten.add.Tensor(mean_16, 1e-06);  mean_16 = None
        sqrt_16 = torch.ops.aten.sqrt.default(add_68);  add_68 = None
        reciprocal_24 = torch.ops.aten.reciprocal.default(sqrt_16);  sqrt_16 = None
        mul_157 = torch.ops.aten.mul.Tensor(add_67, reciprocal_24)
        mul_158 = torch.ops.aten.mul.Tensor(primals_17, mul_157);  mul_157 = None
        rand_like_25 = torch.ops.aten.rand_like.default(mul_158, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_132 = torch.ops.aten.alias.default(rand_like_25);  rand_like_25 = None
        gt_34 = torch.ops.aten.gt.Scalar(alias_132, 0.1);  alias_132 = None
        mul_159 = torch.ops.aten.mul.Tensor(gt_34, mul_158);  mul_158 = None
        mul_160 = torch.ops.aten.mul.Tensor(mul_159, 1.1111111111111112);  mul_159 = None
        view_97 = torch.ops.aten.view.default(primals_192, [-1, 128]);  primals_192 = None
        embedding_2 = torch.ops.aten.embedding.default(primals_43, view_97);  primals_43 = None
        ones_2 = torch.ops.aten.ones.default([2, 128], dtype = torch.int64, device = device(type='cuda', index=0), pin_memory = False)
        alias_135 = torch.ops.aten.alias.default(ones_2);  ones_2 = None
        alias_136 = torch.ops.aten.alias.default(alias_135);  alias_135 = None
        arange_2 = torch.ops.aten.arange.default(128, device = device(type='cuda', index=0), pin_memory = False)
        alias_137 = torch.ops.aten.alias.default(arange_2);  arange_2 = None
        alias_138 = torch.ops.aten.alias.default(alias_137);  alias_137 = None
        unsqueeze_5 = torch.ops.aten.unsqueeze.default(alias_138, 0);  alias_138 = None
        unsqueeze_6 = torch.ops.aten.unsqueeze.default(unsqueeze_5, 1)
        slice_5 = torch.ops.aten.slice.Tensor(unsqueeze_6, 2, 0, 9223372036854775807);  unsqueeze_6 = None
        repeat = torch.ops.aten.repeat.default(slice_5, [2, 128, 1]);  slice_5 = None
        slice_6 = torch.ops.aten.slice.Tensor(unsqueeze_5, 1, 0, 9223372036854775807);  unsqueeze_5 = None
        unsqueeze_8 = torch.ops.aten.unsqueeze.default(slice_6, 2);  slice_6 = None
        le = torch.ops.aten.le.Tensor(repeat, unsqueeze_8);  repeat = unsqueeze_8 = None
        _to_copy_11 = torch.ops.aten._to_copy.default(le, dtype = torch.float32);  le = None
        slice_7 = torch.ops.aten.slice.Tensor(_to_copy_11, 0, 0, 9223372036854775807);  _to_copy_11 = None
        unsqueeze_9 = torch.ops.aten.unsqueeze.default(slice_7, 1);  slice_7 = None
        slice_8 = torch.ops.aten.slice.Tensor(unsqueeze_9, 2, 0, 9223372036854775807);  unsqueeze_9 = None
        slice_9 = torch.ops.aten.slice.Tensor(slice_8, 3, 0, 9223372036854775807);  slice_8 = None
        mul_161 = torch.ops.aten.mul.Tensor(slice_9, slice_2);  slice_9 = slice_2 = None
        _tensor_constant1 = self._tensor_constant1
        lift_fresh_copy_1 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant1);  _tensor_constant1 = None
        sub_18 = torch.ops.aten.sub.Tensor(lift_fresh_copy_1, mul_161);  lift_fresh_copy_1 = mul_161 = None
        mul_162 = torch.ops.aten.mul.Tensor(sub_18, -3.4028234663852886e+38);  sub_18 = None
        slice_12 = torch.ops.aten.slice.Tensor(alias_136, 0, 0, 9223372036854775807);  alias_136 = None
        unsqueeze_12 = torch.ops.aten.unsqueeze.default(slice_12, 1);  slice_12 = None
        unsqueeze_13 = torch.ops.aten.unsqueeze.default(unsqueeze_12, 2);  unsqueeze_12 = None
        slice_13 = torch.ops.aten.slice.Tensor(unsqueeze_13, 3, 0, 9223372036854775807);  unsqueeze_13 = None
        _to_copy_12 = torch.ops.aten._to_copy.default(slice_13, dtype = torch.float32);  slice_13 = None
        _tensor_constant2 = self._tensor_constant2
        lift_fresh_copy_2 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant2);  _tensor_constant2 = None
        sub_19 = torch.ops.aten.sub.Tensor(lift_fresh_copy_2, _to_copy_12);  lift_fresh_copy_2 = _to_copy_12 = None
        mul_163 = torch.ops.aten.mul.Tensor(sub_19, -3.4028234663852886e+38);  sub_19 = None
        rand_like_26 = torch.ops.aten.rand_like.default(embedding_2, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_139 = torch.ops.aten.alias.default(rand_like_26);  rand_like_26 = None
        gt_35 = torch.ops.aten.gt.Scalar(alias_139, 0.1);  alias_139 = None
        mul_164 = torch.ops.aten.mul.Tensor(gt_35, embedding_2)
        mul_165 = torch.ops.aten.mul.Tensor(mul_164, 1.1111111111111112);  mul_164 = None
        pow_26 = torch.ops.aten.pow.Tensor_Scalar(mul_165, 2)
        mean_17 = torch.ops.aten.mean.dim(pow_26, [-1], True);  pow_26 = None
        add_69 = torch.ops.aten.add.Tensor(mean_17, 1e-06);  mean_17 = None
        sqrt_17 = torch.ops.aten.sqrt.default(add_69);  add_69 = None
        reciprocal_25 = torch.ops.aten.reciprocal.default(sqrt_17);  sqrt_17 = None
        mul_166 = torch.ops.aten.mul.Tensor(mul_165, reciprocal_25)
        mul_167 = torch.ops.aten.mul.Tensor(primals_18, mul_166);  mul_166 = None
        permute_97 = torch.ops.aten.permute.default(primals_101, [1, 0]);  primals_101 = None
        view_98 = torch.ops.aten.view.default(mul_167, [256, 512]);  mul_167 = None
        mm_56 = torch.ops.aten.mm.default(view_98, permute_97)
        _unsafe_view_96 = torch.ops.aten._unsafe_view.default(mm_56, [2, 128, 384]);  mm_56 = None
        view_99 = torch.ops.aten.view.default(_unsafe_view_96, [2, -1, 6, 64]);  _unsafe_view_96 = None
        permute_98 = torch.ops.aten.permute.default(view_99, [0, 2, 1, 3]);  view_99 = None
        permute_99 = torch.ops.aten.permute.default(primals_102, [1, 0]);  primals_102 = None
        mm_57 = torch.ops.aten.mm.default(view_98, permute_99)
        _unsafe_view_97 = torch.ops.aten._unsafe_view.default(mm_57, [2, 128, 384]);  mm_57 = None
        view_101 = torch.ops.aten.view.default(_unsafe_view_97, [2, -1, 6, 64]);  _unsafe_view_97 = None
        permute_100 = torch.ops.aten.permute.default(view_101, [0, 2, 1, 3]);  view_101 = None
        permute_101 = torch.ops.aten.permute.default(primals_103, [1, 0]);  primals_103 = None
        mm_58 = torch.ops.aten.mm.default(view_98, permute_101);  view_98 = None
        _unsafe_view_98 = torch.ops.aten._unsafe_view.default(mm_58, [2, 128, 384]);  mm_58 = None
        view_103 = torch.ops.aten.view.default(_unsafe_view_98, [2, -1, 6, 64]);  _unsafe_view_98 = None
        permute_102 = torch.ops.aten.permute.default(view_103, [0, 2, 1, 3]);  view_103 = None
        permute_103 = torch.ops.aten.permute.default(permute_100, [0, 1, 3, 2])
        expand_32 = torch.ops.aten.expand.default(permute_98, [2, 6, 128, 64]);  permute_98 = None
        clone_32 = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
        _unsafe_view_99 = torch.ops.aten._unsafe_view.default(clone_32, [12, 128, 64]);  clone_32 = None
        expand_33 = torch.ops.aten.expand.default(permute_103, [2, 6, 64, 128]);  permute_103 = None
        clone_33 = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
        _unsafe_view_100 = torch.ops.aten._unsafe_view.default(clone_33, [12, 64, 128]);  clone_33 = None
        bmm_16 = torch.ops.aten.bmm.default(_unsafe_view_99, _unsafe_view_100)
        _unsafe_view_101 = torch.ops.aten._unsafe_view.default(bmm_16, [2, 6, 128, 128]);  bmm_16 = None
        zeros_like = torch.ops.aten.zeros_like.default(sub_1, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_147 = torch.ops.aten.alias.default(zeros_like);  zeros_like = None
        alias_148 = torch.ops.aten.alias.default(alias_147);  alias_147 = None
        minimum_1 = torch.ops.aten.minimum.default(sub_1, alias_148);  sub_1 = alias_148 = None
        neg = torch.ops.aten.neg.default(minimum_1);  minimum_1 = None
        lt_1 = torch.ops.aten.lt.Scalar(neg, 16)
        _to_copy_13 = torch.ops.aten._to_copy.default(neg, dtype = torch.float32)
        div_10 = torch.ops.aten.div.Tensor(_to_copy_13, 16);  _to_copy_13 = None
        log_1 = torch.ops.aten.log.default(div_10);  div_10 = None
        div_11 = torch.ops.aten.div.Tensor(log_1, 2.0794415416798357);  log_1 = None
        mul_168 = torch.ops.aten.mul.Tensor(div_11, 16);  div_11 = None
        _to_copy_14 = torch.ops.aten._to_copy.default(mul_168, dtype = torch.int64);  mul_168 = None
        add_70 = torch.ops.aten.add.Tensor(_to_copy_14, 16);  _to_copy_14 = None
        full_like_1 = torch.ops.aten.full_like.default(add_70, 31, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_149 = torch.ops.aten.alias.default(full_like_1);  full_like_1 = None
        alias_150 = torch.ops.aten.alias.default(alias_149);  alias_149 = None
        minimum_2 = torch.ops.aten.minimum.default(add_70, alias_150);  add_70 = alias_150 = None
        where_1 = torch.ops.aten.where.self(lt_1, neg, minimum_2);  lt_1 = neg = minimum_2 = None
        add_71 = torch.ops.aten.add.Tensor(where_1, 0);  where_1 = None
        embedding_3 = torch.ops.aten.embedding.default(primals_104, add_71);  primals_104 = None
        permute_104 = torch.ops.aten.permute.default(embedding_3, [2, 0, 1]);  embedding_3 = None
        unsqueeze_16 = torch.ops.aten.unsqueeze.default(permute_104, 0);  permute_104 = None
        add_72 = torch.ops.aten.add.Tensor(unsqueeze_16, mul_162);  unsqueeze_16 = mul_162 = None
        add_73 = torch.ops.aten.add.Tensor(_unsafe_view_101, add_72);  _unsafe_view_101 = None
        amax_8 = torch.ops.aten.amax.default(add_73, [-1], True)
        sub_21 = torch.ops.aten.sub.Tensor(add_73, amax_8);  add_73 = amax_8 = None
        exp_16 = torch.ops.aten.exp.default(sub_21);  sub_21 = None
        sum_9 = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
        div_12 = torch.ops.aten.div.Tensor(exp_16, sum_9);  exp_16 = sum_9 = None
        philox_rand_like_8 = torch.ops.prims.philox_rand_like.default(div_12, philox_seed_like, 1572864)
        gt_36 = torch.ops.aten.gt.Scalar(philox_rand_like_8, 0.1);  philox_rand_like_8 = None
        _to_copy_15 = torch.ops.aten._to_copy.default(gt_36, dtype = torch.float32);  gt_36 = None
        mul_169 = torch.ops.aten.mul.Tensor(_to_copy_15, div_12);  _to_copy_15 = None
        mul_170 = torch.ops.aten.mul.Tensor(mul_169, 1.1111111111111112);  mul_169 = None
        expand_34 = torch.ops.aten.expand.default(mul_170, [2, 6, 128, 128]);  mul_170 = None
        view_104 = torch.ops.aten.view.default(expand_34, [12, 128, 128]);  expand_34 = None
        expand_35 = torch.ops.aten.expand.default(permute_102, [2, 6, 128, 64])
        clone_34 = torch.ops.aten.clone.default(expand_35, memory_format = torch.contiguous_format);  expand_35 = None
        _unsafe_view_102 = torch.ops.aten._unsafe_view.default(clone_34, [12, 128, 64]);  clone_34 = None
        bmm_17 = torch.ops.aten.bmm.default(view_104, _unsafe_view_102)
        _unsafe_view_103 = torch.ops.aten._unsafe_view.default(bmm_17, [2, 6, 128, 64]);  bmm_17 = None
        permute_105 = torch.ops.aten.permute.default(_unsafe_view_103, [0, 2, 1, 3]);  _unsafe_view_103 = None
        clone_35 = torch.ops.aten.clone.default(permute_105, memory_format = torch.contiguous_format);  permute_105 = None
        view_105 = torch.ops.aten.view.default(clone_35, [2, -1, 384]);  clone_35 = None
        permute_106 = torch.ops.aten.permute.default(primals_105, [1, 0]);  primals_105 = None
        view_106 = torch.ops.aten.view.default(view_105, [256, 384]);  view_105 = None
        mm_59 = torch.ops.aten.mm.default(view_106, permute_106)
        _unsafe_view_104 = torch.ops.aten._unsafe_view.default(mm_59, [2, 128, 512]);  mm_59 = None
        rand_like_27 = torch.ops.aten.rand_like.default(_unsafe_view_104, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_154 = torch.ops.aten.alias.default(rand_like_27);  rand_like_27 = None
        gt_37 = torch.ops.aten.gt.Scalar(alias_154, 0.1);  alias_154 = None
        mul_171 = torch.ops.aten.mul.Tensor(gt_37, _unsafe_view_104);  _unsafe_view_104 = None
        mul_172 = torch.ops.aten.mul.Tensor(mul_171, 1.1111111111111112);  mul_171 = None
        add_74 = torch.ops.aten.add.Tensor(mul_165, mul_172);  mul_165 = mul_172 = None
        pow_27 = torch.ops.aten.pow.Tensor_Scalar(add_74, 2)
        mean_18 = torch.ops.aten.mean.dim(pow_27, [-1], True);  pow_27 = None
        add_75 = torch.ops.aten.add.Tensor(mean_18, 1e-06);  mean_18 = None
        sqrt_18 = torch.ops.aten.sqrt.default(add_75);  add_75 = None
        reciprocal_26 = torch.ops.aten.reciprocal.default(sqrt_18);  sqrt_18 = None
        mul_173 = torch.ops.aten.mul.Tensor(add_74, reciprocal_26)
        mul_174 = torch.ops.aten.mul.Tensor(primals_19, mul_173);  mul_173 = None
        permute_107 = torch.ops.aten.permute.default(primals_106, [1, 0]);  primals_106 = None
        view_107 = torch.ops.aten.view.default(mul_174, [256, 512]);  mul_174 = None
        mm_60 = torch.ops.aten.mm.default(view_107, permute_107);  view_107 = None
        _unsafe_view_105 = torch.ops.aten._unsafe_view.default(mm_60, [2, 128, 384]);  mm_60 = None
        view_108 = torch.ops.aten.view.default(_unsafe_view_105, [2, -1, 6, 64]);  _unsafe_view_105 = None
        permute_108 = torch.ops.aten.permute.default(view_108, [0, 2, 1, 3]);  view_108 = None
        permute_109 = torch.ops.aten.permute.default(primals_107, [1, 0]);  primals_107 = None
        view_109 = torch.ops.aten.view.default(mul_160, [256, 512])
        mm_61 = torch.ops.aten.mm.default(view_109, permute_109)
        _unsafe_view_106 = torch.ops.aten._unsafe_view.default(mm_61, [2, 128, 384]);  mm_61 = None
        view_110 = torch.ops.aten.view.default(_unsafe_view_106, [2, -1, 6, 64]);  _unsafe_view_106 = None
        permute_110 = torch.ops.aten.permute.default(view_110, [0, 2, 1, 3]);  view_110 = None
        permute_111 = torch.ops.aten.permute.default(primals_108, [1, 0]);  primals_108 = None
        mm_62 = torch.ops.aten.mm.default(view_109, permute_111)
        _unsafe_view_107 = torch.ops.aten._unsafe_view.default(mm_62, [2, 128, 384]);  mm_62 = None
        view_112 = torch.ops.aten.view.default(_unsafe_view_107, [2, -1, 6, 64]);  _unsafe_view_107 = None
        permute_112 = torch.ops.aten.permute.default(view_112, [0, 2, 1, 3]);  view_112 = None
        permute_113 = torch.ops.aten.permute.default(permute_110, [0, 1, 3, 2])
        expand_36 = torch.ops.aten.expand.default(permute_108, [2, 6, 128, 64]);  permute_108 = None
        clone_36 = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
        _unsafe_view_108 = torch.ops.aten._unsafe_view.default(clone_36, [12, 128, 64]);  clone_36 = None
        expand_37 = torch.ops.aten.expand.default(permute_113, [2, 6, 64, 128]);  permute_113 = None
        clone_37 = torch.ops.aten.clone.default(expand_37, memory_format = torch.contiguous_format);  expand_37 = None
        _unsafe_view_109 = torch.ops.aten._unsafe_view.default(clone_37, [12, 64, 128]);  clone_37 = None
        bmm_18 = torch.ops.aten.bmm.default(_unsafe_view_108, _unsafe_view_109)
        _unsafe_view_110 = torch.ops.aten._unsafe_view.default(bmm_18, [2, 6, 128, 128]);  bmm_18 = None
        zeros = torch.ops.aten.zeros.default([1, 6, 128, 128], dtype = torch.float32, device = device(type='cuda', index=0), pin_memory = False)
        alias_158 = torch.ops.aten.alias.default(zeros);  zeros = None
        alias_159 = torch.ops.aten.alias.default(alias_158);  alias_158 = None
        add_76 = torch.ops.aten.add.Tensor(alias_159, mul_163);  alias_159 = mul_163 = None
        add_77 = torch.ops.aten.add.Tensor(_unsafe_view_110, add_76);  _unsafe_view_110 = None
        amax_9 = torch.ops.aten.amax.default(add_77, [-1], True)
        sub_22 = torch.ops.aten.sub.Tensor(add_77, amax_9);  add_77 = amax_9 = None
        exp_17 = torch.ops.aten.exp.default(sub_22);  sub_22 = None
        sum_10 = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
        div_13 = torch.ops.aten.div.Tensor(exp_17, sum_10);  exp_17 = sum_10 = None
        philox_rand_like_9 = torch.ops.prims.philox_rand_like.default(div_13, philox_seed_like, 1769472)
        gt_38 = torch.ops.aten.gt.Scalar(philox_rand_like_9, 0.1);  philox_rand_like_9 = None
        _to_copy_16 = torch.ops.aten._to_copy.default(gt_38, dtype = torch.float32);  gt_38 = None
        mul_175 = torch.ops.aten.mul.Tensor(_to_copy_16, div_13);  _to_copy_16 = None
        mul_176 = torch.ops.aten.mul.Tensor(mul_175, 1.1111111111111112);  mul_175 = None
        expand_38 = torch.ops.aten.expand.default(mul_176, [2, 6, 128, 128]);  mul_176 = None
        view_113 = torch.ops.aten.view.default(expand_38, [12, 128, 128]);  expand_38 = None
        expand_39 = torch.ops.aten.expand.default(permute_112, [2, 6, 128, 64])
        clone_38 = torch.ops.aten.clone.default(expand_39, memory_format = torch.contiguous_format);  expand_39 = None
        _unsafe_view_111 = torch.ops.aten._unsafe_view.default(clone_38, [12, 128, 64]);  clone_38 = None
        bmm_19 = torch.ops.aten.bmm.default(view_113, _unsafe_view_111)
        _unsafe_view_112 = torch.ops.aten._unsafe_view.default(bmm_19, [2, 6, 128, 64]);  bmm_19 = None
        permute_114 = torch.ops.aten.permute.default(_unsafe_view_112, [0, 2, 1, 3]);  _unsafe_view_112 = None
        clone_39 = torch.ops.aten.clone.default(permute_114, memory_format = torch.contiguous_format);  permute_114 = None
        view_114 = torch.ops.aten.view.default(clone_39, [2, -1, 384]);  clone_39 = None
        permute_115 = torch.ops.aten.permute.default(primals_109, [1, 0]);  primals_109 = None
        view_115 = torch.ops.aten.view.default(view_114, [256, 384]);  view_114 = None
        mm_63 = torch.ops.aten.mm.default(view_115, permute_115)
        _unsafe_view_113 = torch.ops.aten._unsafe_view.default(mm_63, [2, 128, 512]);  mm_63 = None
        rand_like_28 = torch.ops.aten.rand_like.default(_unsafe_view_113, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_163 = torch.ops.aten.alias.default(rand_like_28);  rand_like_28 = None
        gt_39 = torch.ops.aten.gt.Scalar(alias_163, 0.1);  alias_163 = None
        mul_177 = torch.ops.aten.mul.Tensor(gt_39, _unsafe_view_113);  _unsafe_view_113 = None
        mul_178 = torch.ops.aten.mul.Tensor(mul_177, 1.1111111111111112);  mul_177 = None
        add_78 = torch.ops.aten.add.Tensor(add_74, mul_178);  mul_178 = None
        pow_28 = torch.ops.aten.pow.Tensor_Scalar(add_78, 2)
        mean_19 = torch.ops.aten.mean.dim(pow_28, [-1], True);  pow_28 = None
        add_79 = torch.ops.aten.add.Tensor(mean_19, 1e-06);  mean_19 = None
        sqrt_19 = torch.ops.aten.sqrt.default(add_79);  add_79 = None
        reciprocal_27 = torch.ops.aten.reciprocal.default(sqrt_19);  sqrt_19 = None
        mul_179 = torch.ops.aten.mul.Tensor(add_78, reciprocal_27)
        mul_180 = torch.ops.aten.mul.Tensor(primals_20, mul_179);  mul_179 = None
        permute_116 = torch.ops.aten.permute.default(primals_110, [1, 0]);  primals_110 = None
        view_116 = torch.ops.aten.view.default(mul_180, [256, 512]);  mul_180 = None
        mm_64 = torch.ops.aten.mm.default(view_116, permute_116)
        _unsafe_view_114 = torch.ops.aten._unsafe_view.default(mm_64, [2, 128, 1024])
        mul_181 = torch.ops.aten.mul.Tensor(_unsafe_view_114, 0.5)
        pow_29 = torch.ops.aten.pow.Tensor_Scalar(_unsafe_view_114, 3.0)
        mul_182 = torch.ops.aten.mul.Tensor(pow_29, 0.044715);  pow_29 = None
        add_80 = torch.ops.aten.add.Tensor(_unsafe_view_114, mul_182);  _unsafe_view_114 = mul_182 = None
        mul_183 = torch.ops.aten.mul.Tensor(add_80, 0.7978845608028654);  add_80 = None
        mul_184 = torch.ops.aten.mul.Tensor(mul_183, -2.0);  mul_183 = None
        exp_18 = torch.ops.aten.exp.default(mul_184);  mul_184 = None
        add_81 = torch.ops.aten.add.Tensor(exp_18, 1.0);  exp_18 = None
        reciprocal_28 = torch.ops.aten.reciprocal.default(add_81);  add_81 = None
        mul_185 = torch.ops.aten.mul.Tensor(reciprocal_28, 2.0);  reciprocal_28 = None
        sub_23 = torch.ops.aten.sub.Tensor(mul_185, 1.0);  mul_185 = None
        add_82 = torch.ops.aten.add.Tensor(sub_23, 1.0)
        mul_186 = torch.ops.aten.mul.Tensor(mul_181, add_82);  mul_181 = add_82 = None
        permute_117 = torch.ops.aten.permute.default(primals_111, [1, 0]);  primals_111 = None
        mm_65 = torch.ops.aten.mm.default(view_116, permute_117);  view_116 = None
        _unsafe_view_115 = torch.ops.aten._unsafe_view.default(mm_65, [2, 128, 1024])
        mul_187 = torch.ops.aten.mul.Tensor(mul_186, _unsafe_view_115);  mul_186 = _unsafe_view_115 = None
        rand_like_29 = torch.ops.aten.rand_like.default(mul_187, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_170 = torch.ops.aten.alias.default(rand_like_29);  rand_like_29 = None
        gt_40 = torch.ops.aten.gt.Scalar(alias_170, 0.1);  alias_170 = None
        mul_188 = torch.ops.aten.mul.Tensor(gt_40, mul_187);  mul_187 = None
        mul_189 = torch.ops.aten.mul.Tensor(mul_188, 1.1111111111111112);  mul_188 = None
        permute_118 = torch.ops.aten.permute.default(primals_112, [1, 0]);  primals_112 = None
        view_118 = torch.ops.aten.view.default(mul_189, [256, 1024]);  mul_189 = None
        mm_66 = torch.ops.aten.mm.default(view_118, permute_118)
        _unsafe_view_116 = torch.ops.aten._unsafe_view.default(mm_66, [2, 128, 512]);  mm_66 = None
        rand_like_30 = torch.ops.aten.rand_like.default(_unsafe_view_116, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_171 = torch.ops.aten.alias.default(rand_like_30);  rand_like_30 = None
        gt_41 = torch.ops.aten.gt.Scalar(alias_171, 0.1);  alias_171 = None
        mul_190 = torch.ops.aten.mul.Tensor(gt_41, _unsafe_view_116);  _unsafe_view_116 = None
        mul_191 = torch.ops.aten.mul.Tensor(mul_190, 1.1111111111111112);  mul_190 = None
        add_83 = torch.ops.aten.add.Tensor(add_78, mul_191);  mul_191 = None
        pow_30 = torch.ops.aten.pow.Tensor_Scalar(add_83, 2)
        mean_20 = torch.ops.aten.mean.dim(pow_30, [-1], True);  pow_30 = None
        add_84 = torch.ops.aten.add.Tensor(mean_20, 1e-06);  mean_20 = None
        sqrt_20 = torch.ops.aten.sqrt.default(add_84);  add_84 = None
        reciprocal_29 = torch.ops.aten.reciprocal.default(sqrt_20);  sqrt_20 = None
        mul_192 = torch.ops.aten.mul.Tensor(add_83, reciprocal_29)
        mul_193 = torch.ops.aten.mul.Tensor(primals_21, mul_192);  mul_192 = None
        permute_119 = torch.ops.aten.permute.default(primals_113, [1, 0]);  primals_113 = None
        view_119 = torch.ops.aten.view.default(mul_193, [256, 512]);  mul_193 = None
        mm_67 = torch.ops.aten.mm.default(view_119, permute_119)
        _unsafe_view_117 = torch.ops.aten._unsafe_view.default(mm_67, [2, 128, 384]);  mm_67 = None
        view_120 = torch.ops.aten.view.default(_unsafe_view_117, [2, -1, 6, 64]);  _unsafe_view_117 = None
        permute_120 = torch.ops.aten.permute.default(view_120, [0, 2, 1, 3]);  view_120 = None
        permute_121 = torch.ops.aten.permute.default(primals_114, [1, 0]);  primals_114 = None
        mm_68 = torch.ops.aten.mm.default(view_119, permute_121)
        _unsafe_view_118 = torch.ops.aten._unsafe_view.default(mm_68, [2, 128, 384]);  mm_68 = None
        view_122 = torch.ops.aten.view.default(_unsafe_view_118, [2, -1, 6, 64]);  _unsafe_view_118 = None
        permute_122 = torch.ops.aten.permute.default(view_122, [0, 2, 1, 3]);  view_122 = None
        permute_123 = torch.ops.aten.permute.default(primals_115, [1, 0]);  primals_115 = None
        mm_69 = torch.ops.aten.mm.default(view_119, permute_123);  view_119 = None
        _unsafe_view_119 = torch.ops.aten._unsafe_view.default(mm_69, [2, 128, 384]);  mm_69 = None
        view_124 = torch.ops.aten.view.default(_unsafe_view_119, [2, -1, 6, 64]);  _unsafe_view_119 = None
        permute_124 = torch.ops.aten.permute.default(view_124, [0, 2, 1, 3]);  view_124 = None
        permute_125 = torch.ops.aten.permute.default(permute_122, [0, 1, 3, 2])
        expand_40 = torch.ops.aten.expand.default(permute_120, [2, 6, 128, 64]);  permute_120 = None
        clone_40 = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
        _unsafe_view_120 = torch.ops.aten._unsafe_view.default(clone_40, [12, 128, 64]);  clone_40 = None
        expand_41 = torch.ops.aten.expand.default(permute_125, [2, 6, 64, 128]);  permute_125 = None
        clone_41 = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
        _unsafe_view_121 = torch.ops.aten._unsafe_view.default(clone_41, [12, 64, 128]);  clone_41 = None
        bmm_20 = torch.ops.aten.bmm.default(_unsafe_view_120, _unsafe_view_121)
        _unsafe_view_122 = torch.ops.aten._unsafe_view.default(bmm_20, [2, 6, 128, 128]);  bmm_20 = None
        add_85 = torch.ops.aten.add.Tensor(_unsafe_view_122, add_72);  _unsafe_view_122 = None
        amax_10 = torch.ops.aten.amax.default(add_85, [-1], True)
        sub_24 = torch.ops.aten.sub.Tensor(add_85, amax_10);  add_85 = amax_10 = None
        exp_19 = torch.ops.aten.exp.default(sub_24);  sub_24 = None
        sum_11 = torch.ops.aten.sum.dim_IntList(exp_19, [-1], True)
        div_14 = torch.ops.aten.div.Tensor(exp_19, sum_11);  exp_19 = sum_11 = None
        philox_rand_like_10 = torch.ops.prims.philox_rand_like.default(div_14, philox_seed_like, 1966080)
        gt_42 = torch.ops.aten.gt.Scalar(philox_rand_like_10, 0.1);  philox_rand_like_10 = None
        _to_copy_17 = torch.ops.aten._to_copy.default(gt_42, dtype = torch.float32);  gt_42 = None
        mul_194 = torch.ops.aten.mul.Tensor(_to_copy_17, div_14);  _to_copy_17 = None
        mul_195 = torch.ops.aten.mul.Tensor(mul_194, 1.1111111111111112);  mul_194 = None
        expand_42 = torch.ops.aten.expand.default(mul_195, [2, 6, 128, 128]);  mul_195 = None
        view_125 = torch.ops.aten.view.default(expand_42, [12, 128, 128]);  expand_42 = None
        expand_43 = torch.ops.aten.expand.default(permute_124, [2, 6, 128, 64])
        clone_42 = torch.ops.aten.clone.default(expand_43, memory_format = torch.contiguous_format);  expand_43 = None
        _unsafe_view_123 = torch.ops.aten._unsafe_view.default(clone_42, [12, 128, 64]);  clone_42 = None
        bmm_21 = torch.ops.aten.bmm.default(view_125, _unsafe_view_123)
        _unsafe_view_124 = torch.ops.aten._unsafe_view.default(bmm_21, [2, 6, 128, 64]);  bmm_21 = None
        permute_126 = torch.ops.aten.permute.default(_unsafe_view_124, [0, 2, 1, 3]);  _unsafe_view_124 = None
        clone_43 = torch.ops.aten.clone.default(permute_126, memory_format = torch.contiguous_format);  permute_126 = None
        view_126 = torch.ops.aten.view.default(clone_43, [2, -1, 384]);  clone_43 = None
        permute_127 = torch.ops.aten.permute.default(primals_116, [1, 0]);  primals_116 = None
        view_127 = torch.ops.aten.view.default(view_126, [256, 384]);  view_126 = None
        mm_70 = torch.ops.aten.mm.default(view_127, permute_127)
        _unsafe_view_125 = torch.ops.aten._unsafe_view.default(mm_70, [2, 128, 512]);  mm_70 = None
        rand_like_31 = torch.ops.aten.rand_like.default(_unsafe_view_125, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_178 = torch.ops.aten.alias.default(rand_like_31);  rand_like_31 = None
        gt_43 = torch.ops.aten.gt.Scalar(alias_178, 0.1);  alias_178 = None
        mul_196 = torch.ops.aten.mul.Tensor(gt_43, _unsafe_view_125);  _unsafe_view_125 = None
        mul_197 = torch.ops.aten.mul.Tensor(mul_196, 1.1111111111111112);  mul_196 = None
        add_86 = torch.ops.aten.add.Tensor(add_83, mul_197);  mul_197 = None
        pow_31 = torch.ops.aten.pow.Tensor_Scalar(add_86, 2)
        mean_21 = torch.ops.aten.mean.dim(pow_31, [-1], True);  pow_31 = None
        add_87 = torch.ops.aten.add.Tensor(mean_21, 1e-06);  mean_21 = None
        sqrt_21 = torch.ops.aten.sqrt.default(add_87);  add_87 = None
        reciprocal_30 = torch.ops.aten.reciprocal.default(sqrt_21);  sqrt_21 = None
        mul_198 = torch.ops.aten.mul.Tensor(add_86, reciprocal_30)
        mul_199 = torch.ops.aten.mul.Tensor(primals_22, mul_198);  mul_198 = None
        permute_128 = torch.ops.aten.permute.default(primals_117, [1, 0]);  primals_117 = None
        view_128 = torch.ops.aten.view.default(mul_199, [256, 512]);  mul_199 = None
        mm_71 = torch.ops.aten.mm.default(view_128, permute_128);  view_128 = None
        _unsafe_view_126 = torch.ops.aten._unsafe_view.default(mm_71, [2, 128, 384]);  mm_71 = None
        view_129 = torch.ops.aten.view.default(_unsafe_view_126, [2, -1, 6, 64]);  _unsafe_view_126 = None
        permute_129 = torch.ops.aten.permute.default(view_129, [0, 2, 1, 3]);  view_129 = None
        permute_130 = torch.ops.aten.permute.default(primals_118, [1, 0]);  primals_118 = None
        mm_72 = torch.ops.aten.mm.default(view_109, permute_130)
        _unsafe_view_127 = torch.ops.aten._unsafe_view.default(mm_72, [2, 128, 384]);  mm_72 = None
        view_131 = torch.ops.aten.view.default(_unsafe_view_127, [2, -1, 6, 64]);  _unsafe_view_127 = None
        permute_131 = torch.ops.aten.permute.default(view_131, [0, 2, 1, 3]);  view_131 = None
        permute_132 = torch.ops.aten.permute.default(primals_119, [1, 0]);  primals_119 = None
        mm_73 = torch.ops.aten.mm.default(view_109, permute_132)
        _unsafe_view_128 = torch.ops.aten._unsafe_view.default(mm_73, [2, 128, 384]);  mm_73 = None
        view_133 = torch.ops.aten.view.default(_unsafe_view_128, [2, -1, 6, 64]);  _unsafe_view_128 = None
        permute_133 = torch.ops.aten.permute.default(view_133, [0, 2, 1, 3]);  view_133 = None
        permute_134 = torch.ops.aten.permute.default(permute_131, [0, 1, 3, 2])
        expand_44 = torch.ops.aten.expand.default(permute_129, [2, 6, 128, 64]);  permute_129 = None
        clone_44 = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
        _unsafe_view_129 = torch.ops.aten._unsafe_view.default(clone_44, [12, 128, 64]);  clone_44 = None
        expand_45 = torch.ops.aten.expand.default(permute_134, [2, 6, 64, 128]);  permute_134 = None
        clone_45 = torch.ops.aten.clone.default(expand_45, memory_format = torch.contiguous_format);  expand_45 = None
        _unsafe_view_130 = torch.ops.aten._unsafe_view.default(clone_45, [12, 64, 128]);  clone_45 = None
        bmm_22 = torch.ops.aten.bmm.default(_unsafe_view_129, _unsafe_view_130)
        _unsafe_view_131 = torch.ops.aten._unsafe_view.default(bmm_22, [2, 6, 128, 128]);  bmm_22 = None
        add_88 = torch.ops.aten.add.Tensor(_unsafe_view_131, add_76);  _unsafe_view_131 = None
        amax_11 = torch.ops.aten.amax.default(add_88, [-1], True)
        sub_25 = torch.ops.aten.sub.Tensor(add_88, amax_11);  add_88 = amax_11 = None
        exp_20 = torch.ops.aten.exp.default(sub_25);  sub_25 = None
        sum_12 = torch.ops.aten.sum.dim_IntList(exp_20, [-1], True)
        div_15 = torch.ops.aten.div.Tensor(exp_20, sum_12);  exp_20 = sum_12 = None
        philox_rand_like_11 = torch.ops.prims.philox_rand_like.default(div_15, philox_seed_like, 2162688)
        gt_44 = torch.ops.aten.gt.Scalar(philox_rand_like_11, 0.1);  philox_rand_like_11 = None
        _to_copy_18 = torch.ops.aten._to_copy.default(gt_44, dtype = torch.float32);  gt_44 = None
        mul_200 = torch.ops.aten.mul.Tensor(_to_copy_18, div_15);  _to_copy_18 = None
        mul_201 = torch.ops.aten.mul.Tensor(mul_200, 1.1111111111111112);  mul_200 = None
        expand_46 = torch.ops.aten.expand.default(mul_201, [2, 6, 128, 128]);  mul_201 = None
        view_134 = torch.ops.aten.view.default(expand_46, [12, 128, 128]);  expand_46 = None
        expand_47 = torch.ops.aten.expand.default(permute_133, [2, 6, 128, 64])
        clone_46 = torch.ops.aten.clone.default(expand_47, memory_format = torch.contiguous_format);  expand_47 = None
        _unsafe_view_132 = torch.ops.aten._unsafe_view.default(clone_46, [12, 128, 64]);  clone_46 = None
        bmm_23 = torch.ops.aten.bmm.default(view_134, _unsafe_view_132)
        _unsafe_view_133 = torch.ops.aten._unsafe_view.default(bmm_23, [2, 6, 128, 64]);  bmm_23 = None
        permute_135 = torch.ops.aten.permute.default(_unsafe_view_133, [0, 2, 1, 3]);  _unsafe_view_133 = None
        clone_47 = torch.ops.aten.clone.default(permute_135, memory_format = torch.contiguous_format);  permute_135 = None
        view_135 = torch.ops.aten.view.default(clone_47, [2, -1, 384]);  clone_47 = None
        permute_136 = torch.ops.aten.permute.default(primals_120, [1, 0]);  primals_120 = None
        view_136 = torch.ops.aten.view.default(view_135, [256, 384]);  view_135 = None
        mm_74 = torch.ops.aten.mm.default(view_136, permute_136)
        _unsafe_view_134 = torch.ops.aten._unsafe_view.default(mm_74, [2, 128, 512]);  mm_74 = None
        rand_like_32 = torch.ops.aten.rand_like.default(_unsafe_view_134, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_185 = torch.ops.aten.alias.default(rand_like_32);  rand_like_32 = None
        gt_45 = torch.ops.aten.gt.Scalar(alias_185, 0.1);  alias_185 = None
        mul_202 = torch.ops.aten.mul.Tensor(gt_45, _unsafe_view_134);  _unsafe_view_134 = None
        mul_203 = torch.ops.aten.mul.Tensor(mul_202, 1.1111111111111112);  mul_202 = None
        add_89 = torch.ops.aten.add.Tensor(add_86, mul_203);  mul_203 = None
        pow_32 = torch.ops.aten.pow.Tensor_Scalar(add_89, 2)
        mean_22 = torch.ops.aten.mean.dim(pow_32, [-1], True);  pow_32 = None
        add_90 = torch.ops.aten.add.Tensor(mean_22, 1e-06);  mean_22 = None
        sqrt_22 = torch.ops.aten.sqrt.default(add_90);  add_90 = None
        reciprocal_31 = torch.ops.aten.reciprocal.default(sqrt_22);  sqrt_22 = None
        mul_204 = torch.ops.aten.mul.Tensor(add_89, reciprocal_31)
        mul_205 = torch.ops.aten.mul.Tensor(primals_23, mul_204);  mul_204 = None
        permute_137 = torch.ops.aten.permute.default(primals_121, [1, 0]);  primals_121 = None
        view_137 = torch.ops.aten.view.default(mul_205, [256, 512]);  mul_205 = None
        mm_75 = torch.ops.aten.mm.default(view_137, permute_137)
        _unsafe_view_135 = torch.ops.aten._unsafe_view.default(mm_75, [2, 128, 1024])
        mul_206 = torch.ops.aten.mul.Tensor(_unsafe_view_135, 0.5)
        pow_33 = torch.ops.aten.pow.Tensor_Scalar(_unsafe_view_135, 3.0)
        mul_207 = torch.ops.aten.mul.Tensor(pow_33, 0.044715);  pow_33 = None
        add_91 = torch.ops.aten.add.Tensor(_unsafe_view_135, mul_207);  _unsafe_view_135 = mul_207 = None
        mul_208 = torch.ops.aten.mul.Tensor(add_91, 0.7978845608028654);  add_91 = None
        mul_209 = torch.ops.aten.mul.Tensor(mul_208, -2.0);  mul_208 = None
        exp_21 = torch.ops.aten.exp.default(mul_209);  mul_209 = None
        add_92 = torch.ops.aten.add.Tensor(exp_21, 1.0);  exp_21 = None
        reciprocal_32 = torch.ops.aten.reciprocal.default(add_92);  add_92 = None
        mul_210 = torch.ops.aten.mul.Tensor(reciprocal_32, 2.0);  reciprocal_32 = None
        sub_26 = torch.ops.aten.sub.Tensor(mul_210, 1.0);  mul_210 = None
        add_93 = torch.ops.aten.add.Tensor(sub_26, 1.0)
        mul_211 = torch.ops.aten.mul.Tensor(mul_206, add_93);  mul_206 = add_93 = None
        permute_138 = torch.ops.aten.permute.default(primals_122, [1, 0]);  primals_122 = None
        mm_76 = torch.ops.aten.mm.default(view_137, permute_138);  view_137 = None
        _unsafe_view_136 = torch.ops.aten._unsafe_view.default(mm_76, [2, 128, 1024])
        mul_212 = torch.ops.aten.mul.Tensor(mul_211, _unsafe_view_136);  mul_211 = _unsafe_view_136 = None
        rand_like_33 = torch.ops.aten.rand_like.default(mul_212, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_192 = torch.ops.aten.alias.default(rand_like_33);  rand_like_33 = None
        gt_46 = torch.ops.aten.gt.Scalar(alias_192, 0.1);  alias_192 = None
        mul_213 = torch.ops.aten.mul.Tensor(gt_46, mul_212);  mul_212 = None
        mul_214 = torch.ops.aten.mul.Tensor(mul_213, 1.1111111111111112);  mul_213 = None
        permute_139 = torch.ops.aten.permute.default(primals_123, [1, 0]);  primals_123 = None
        view_139 = torch.ops.aten.view.default(mul_214, [256, 1024]);  mul_214 = None
        mm_77 = torch.ops.aten.mm.default(view_139, permute_139)
        _unsafe_view_137 = torch.ops.aten._unsafe_view.default(mm_77, [2, 128, 512]);  mm_77 = None
        rand_like_34 = torch.ops.aten.rand_like.default(_unsafe_view_137, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_193 = torch.ops.aten.alias.default(rand_like_34);  rand_like_34 = None
        gt_47 = torch.ops.aten.gt.Scalar(alias_193, 0.1);  alias_193 = None
        mul_215 = torch.ops.aten.mul.Tensor(gt_47, _unsafe_view_137);  _unsafe_view_137 = None
        mul_216 = torch.ops.aten.mul.Tensor(mul_215, 1.1111111111111112);  mul_215 = None
        add_94 = torch.ops.aten.add.Tensor(add_89, mul_216);  mul_216 = None
        pow_34 = torch.ops.aten.pow.Tensor_Scalar(add_94, 2)
        mean_23 = torch.ops.aten.mean.dim(pow_34, [-1], True);  pow_34 = None
        add_95 = torch.ops.aten.add.Tensor(mean_23, 1e-06);  mean_23 = None
        sqrt_23 = torch.ops.aten.sqrt.default(add_95);  add_95 = None
        reciprocal_33 = torch.ops.aten.reciprocal.default(sqrt_23);  sqrt_23 = None
        mul_217 = torch.ops.aten.mul.Tensor(add_94, reciprocal_33)
        mul_218 = torch.ops.aten.mul.Tensor(primals_24, mul_217);  mul_217 = None
        permute_140 = torch.ops.aten.permute.default(primals_124, [1, 0]);  primals_124 = None
        view_140 = torch.ops.aten.view.default(mul_218, [256, 512]);  mul_218 = None
        mm_78 = torch.ops.aten.mm.default(view_140, permute_140)
        _unsafe_view_138 = torch.ops.aten._unsafe_view.default(mm_78, [2, 128, 384]);  mm_78 = None
        view_141 = torch.ops.aten.view.default(_unsafe_view_138, [2, -1, 6, 64]);  _unsafe_view_138 = None
        permute_141 = torch.ops.aten.permute.default(view_141, [0, 2, 1, 3]);  view_141 = None
        permute_142 = torch.ops.aten.permute.default(primals_125, [1, 0]);  primals_125 = None
        mm_79 = torch.ops.aten.mm.default(view_140, permute_142)
        _unsafe_view_139 = torch.ops.aten._unsafe_view.default(mm_79, [2, 128, 384]);  mm_79 = None
        view_143 = torch.ops.aten.view.default(_unsafe_view_139, [2, -1, 6, 64]);  _unsafe_view_139 = None
        permute_143 = torch.ops.aten.permute.default(view_143, [0, 2, 1, 3]);  view_143 = None
        permute_144 = torch.ops.aten.permute.default(primals_126, [1, 0]);  primals_126 = None
        mm_80 = torch.ops.aten.mm.default(view_140, permute_144);  view_140 = None
        _unsafe_view_140 = torch.ops.aten._unsafe_view.default(mm_80, [2, 128, 384]);  mm_80 = None
        view_145 = torch.ops.aten.view.default(_unsafe_view_140, [2, -1, 6, 64]);  _unsafe_view_140 = None
        permute_145 = torch.ops.aten.permute.default(view_145, [0, 2, 1, 3]);  view_145 = None
        permute_146 = torch.ops.aten.permute.default(permute_143, [0, 1, 3, 2])
        expand_48 = torch.ops.aten.expand.default(permute_141, [2, 6, 128, 64]);  permute_141 = None
        clone_48 = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
        _unsafe_view_141 = torch.ops.aten._unsafe_view.default(clone_48, [12, 128, 64]);  clone_48 = None
        expand_49 = torch.ops.aten.expand.default(permute_146, [2, 6, 64, 128]);  permute_146 = None
        clone_49 = torch.ops.aten.clone.default(expand_49, memory_format = torch.contiguous_format);  expand_49 = None
        _unsafe_view_142 = torch.ops.aten._unsafe_view.default(clone_49, [12, 64, 128]);  clone_49 = None
        bmm_24 = torch.ops.aten.bmm.default(_unsafe_view_141, _unsafe_view_142)
        _unsafe_view_143 = torch.ops.aten._unsafe_view.default(bmm_24, [2, 6, 128, 128]);  bmm_24 = None
        add_96 = torch.ops.aten.add.Tensor(_unsafe_view_143, add_72);  _unsafe_view_143 = None
        amax_12 = torch.ops.aten.amax.default(add_96, [-1], True)
        sub_27 = torch.ops.aten.sub.Tensor(add_96, amax_12);  add_96 = amax_12 = None
        exp_22 = torch.ops.aten.exp.default(sub_27);  sub_27 = None
        sum_13 = torch.ops.aten.sum.dim_IntList(exp_22, [-1], True)
        div_16 = torch.ops.aten.div.Tensor(exp_22, sum_13);  exp_22 = sum_13 = None
        philox_rand_like_12 = torch.ops.prims.philox_rand_like.default(div_16, philox_seed_like, 2359296)
        gt_48 = torch.ops.aten.gt.Scalar(philox_rand_like_12, 0.1);  philox_rand_like_12 = None
        _to_copy_19 = torch.ops.aten._to_copy.default(gt_48, dtype = torch.float32);  gt_48 = None
        mul_219 = torch.ops.aten.mul.Tensor(_to_copy_19, div_16);  _to_copy_19 = None
        mul_220 = torch.ops.aten.mul.Tensor(mul_219, 1.1111111111111112);  mul_219 = None
        expand_50 = torch.ops.aten.expand.default(mul_220, [2, 6, 128, 128]);  mul_220 = None
        view_146 = torch.ops.aten.view.default(expand_50, [12, 128, 128]);  expand_50 = None
        expand_51 = torch.ops.aten.expand.default(permute_145, [2, 6, 128, 64])
        clone_50 = torch.ops.aten.clone.default(expand_51, memory_format = torch.contiguous_format);  expand_51 = None
        _unsafe_view_144 = torch.ops.aten._unsafe_view.default(clone_50, [12, 128, 64]);  clone_50 = None
        bmm_25 = torch.ops.aten.bmm.default(view_146, _unsafe_view_144)
        _unsafe_view_145 = torch.ops.aten._unsafe_view.default(bmm_25, [2, 6, 128, 64]);  bmm_25 = None
        permute_147 = torch.ops.aten.permute.default(_unsafe_view_145, [0, 2, 1, 3]);  _unsafe_view_145 = None
        clone_51 = torch.ops.aten.clone.default(permute_147, memory_format = torch.contiguous_format);  permute_147 = None
        view_147 = torch.ops.aten.view.default(clone_51, [2, -1, 384]);  clone_51 = None
        permute_148 = torch.ops.aten.permute.default(primals_127, [1, 0]);  primals_127 = None
        view_148 = torch.ops.aten.view.default(view_147, [256, 384]);  view_147 = None
        mm_81 = torch.ops.aten.mm.default(view_148, permute_148)
        _unsafe_view_146 = torch.ops.aten._unsafe_view.default(mm_81, [2, 128, 512]);  mm_81 = None
        rand_like_35 = torch.ops.aten.rand_like.default(_unsafe_view_146, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_200 = torch.ops.aten.alias.default(rand_like_35);  rand_like_35 = None
        gt_49 = torch.ops.aten.gt.Scalar(alias_200, 0.1);  alias_200 = None
        mul_221 = torch.ops.aten.mul.Tensor(gt_49, _unsafe_view_146);  _unsafe_view_146 = None
        mul_222 = torch.ops.aten.mul.Tensor(mul_221, 1.1111111111111112);  mul_221 = None
        add_97 = torch.ops.aten.add.Tensor(add_94, mul_222);  mul_222 = None
        pow_35 = torch.ops.aten.pow.Tensor_Scalar(add_97, 2)
        mean_24 = torch.ops.aten.mean.dim(pow_35, [-1], True);  pow_35 = None
        add_98 = torch.ops.aten.add.Tensor(mean_24, 1e-06);  mean_24 = None
        sqrt_24 = torch.ops.aten.sqrt.default(add_98);  add_98 = None
        reciprocal_34 = torch.ops.aten.reciprocal.default(sqrt_24);  sqrt_24 = None
        mul_223 = torch.ops.aten.mul.Tensor(add_97, reciprocal_34)
        mul_224 = torch.ops.aten.mul.Tensor(primals_25, mul_223);  mul_223 = None
        permute_149 = torch.ops.aten.permute.default(primals_128, [1, 0]);  primals_128 = None
        view_149 = torch.ops.aten.view.default(mul_224, [256, 512]);  mul_224 = None
        mm_82 = torch.ops.aten.mm.default(view_149, permute_149);  view_149 = None
        _unsafe_view_147 = torch.ops.aten._unsafe_view.default(mm_82, [2, 128, 384]);  mm_82 = None
        view_150 = torch.ops.aten.view.default(_unsafe_view_147, [2, -1, 6, 64]);  _unsafe_view_147 = None
        permute_150 = torch.ops.aten.permute.default(view_150, [0, 2, 1, 3]);  view_150 = None
        permute_151 = torch.ops.aten.permute.default(primals_129, [1, 0]);  primals_129 = None
        mm_83 = torch.ops.aten.mm.default(view_109, permute_151)
        _unsafe_view_148 = torch.ops.aten._unsafe_view.default(mm_83, [2, 128, 384]);  mm_83 = None
        view_152 = torch.ops.aten.view.default(_unsafe_view_148, [2, -1, 6, 64]);  _unsafe_view_148 = None
        permute_152 = torch.ops.aten.permute.default(view_152, [0, 2, 1, 3]);  view_152 = None
        permute_153 = torch.ops.aten.permute.default(primals_130, [1, 0]);  primals_130 = None
        mm_84 = torch.ops.aten.mm.default(view_109, permute_153)
        _unsafe_view_149 = torch.ops.aten._unsafe_view.default(mm_84, [2, 128, 384]);  mm_84 = None
        view_154 = torch.ops.aten.view.default(_unsafe_view_149, [2, -1, 6, 64]);  _unsafe_view_149 = None
        permute_154 = torch.ops.aten.permute.default(view_154, [0, 2, 1, 3]);  view_154 = None
        permute_155 = torch.ops.aten.permute.default(permute_152, [0, 1, 3, 2])
        expand_52 = torch.ops.aten.expand.default(permute_150, [2, 6, 128, 64]);  permute_150 = None
        clone_52 = torch.ops.aten.clone.default(expand_52, memory_format = torch.contiguous_format);  expand_52 = None
        _unsafe_view_150 = torch.ops.aten._unsafe_view.default(clone_52, [12, 128, 64]);  clone_52 = None
        expand_53 = torch.ops.aten.expand.default(permute_155, [2, 6, 64, 128]);  permute_155 = None
        clone_53 = torch.ops.aten.clone.default(expand_53, memory_format = torch.contiguous_format);  expand_53 = None
        _unsafe_view_151 = torch.ops.aten._unsafe_view.default(clone_53, [12, 64, 128]);  clone_53 = None
        bmm_26 = torch.ops.aten.bmm.default(_unsafe_view_150, _unsafe_view_151)
        _unsafe_view_152 = torch.ops.aten._unsafe_view.default(bmm_26, [2, 6, 128, 128]);  bmm_26 = None
        add_99 = torch.ops.aten.add.Tensor(_unsafe_view_152, add_76);  _unsafe_view_152 = None
        amax_13 = torch.ops.aten.amax.default(add_99, [-1], True)
        sub_28 = torch.ops.aten.sub.Tensor(add_99, amax_13);  add_99 = amax_13 = None
        exp_23 = torch.ops.aten.exp.default(sub_28);  sub_28 = None
        sum_14 = torch.ops.aten.sum.dim_IntList(exp_23, [-1], True)
        div_17 = torch.ops.aten.div.Tensor(exp_23, sum_14);  exp_23 = sum_14 = None
        philox_rand_like_13 = torch.ops.prims.philox_rand_like.default(div_17, philox_seed_like, 2555904)
        gt_50 = torch.ops.aten.gt.Scalar(philox_rand_like_13, 0.1);  philox_rand_like_13 = None
        _to_copy_20 = torch.ops.aten._to_copy.default(gt_50, dtype = torch.float32);  gt_50 = None
        mul_225 = torch.ops.aten.mul.Tensor(_to_copy_20, div_17);  _to_copy_20 = None
        mul_226 = torch.ops.aten.mul.Tensor(mul_225, 1.1111111111111112);  mul_225 = None
        expand_54 = torch.ops.aten.expand.default(mul_226, [2, 6, 128, 128]);  mul_226 = None
        view_155 = torch.ops.aten.view.default(expand_54, [12, 128, 128]);  expand_54 = None
        expand_55 = torch.ops.aten.expand.default(permute_154, [2, 6, 128, 64])
        clone_54 = torch.ops.aten.clone.default(expand_55, memory_format = torch.contiguous_format);  expand_55 = None
        _unsafe_view_153 = torch.ops.aten._unsafe_view.default(clone_54, [12, 128, 64]);  clone_54 = None
        bmm_27 = torch.ops.aten.bmm.default(view_155, _unsafe_view_153)
        _unsafe_view_154 = torch.ops.aten._unsafe_view.default(bmm_27, [2, 6, 128, 64]);  bmm_27 = None
        permute_156 = torch.ops.aten.permute.default(_unsafe_view_154, [0, 2, 1, 3]);  _unsafe_view_154 = None
        clone_55 = torch.ops.aten.clone.default(permute_156, memory_format = torch.contiguous_format);  permute_156 = None
        view_156 = torch.ops.aten.view.default(clone_55, [2, -1, 384]);  clone_55 = None
        permute_157 = torch.ops.aten.permute.default(primals_131, [1, 0]);  primals_131 = None
        view_157 = torch.ops.aten.view.default(view_156, [256, 384]);  view_156 = None
        mm_85 = torch.ops.aten.mm.default(view_157, permute_157)
        _unsafe_view_155 = torch.ops.aten._unsafe_view.default(mm_85, [2, 128, 512]);  mm_85 = None
        rand_like_36 = torch.ops.aten.rand_like.default(_unsafe_view_155, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_207 = torch.ops.aten.alias.default(rand_like_36);  rand_like_36 = None
        gt_51 = torch.ops.aten.gt.Scalar(alias_207, 0.1);  alias_207 = None
        mul_227 = torch.ops.aten.mul.Tensor(gt_51, _unsafe_view_155);  _unsafe_view_155 = None
        mul_228 = torch.ops.aten.mul.Tensor(mul_227, 1.1111111111111112);  mul_227 = None
        add_100 = torch.ops.aten.add.Tensor(add_97, mul_228);  mul_228 = None
        pow_36 = torch.ops.aten.pow.Tensor_Scalar(add_100, 2)
        mean_25 = torch.ops.aten.mean.dim(pow_36, [-1], True);  pow_36 = None
        add_101 = torch.ops.aten.add.Tensor(mean_25, 1e-06);  mean_25 = None
        sqrt_25 = torch.ops.aten.sqrt.default(add_101);  add_101 = None
        reciprocal_35 = torch.ops.aten.reciprocal.default(sqrt_25);  sqrt_25 = None
        mul_229 = torch.ops.aten.mul.Tensor(add_100, reciprocal_35)
        mul_230 = torch.ops.aten.mul.Tensor(primals_26, mul_229);  mul_229 = None
        permute_158 = torch.ops.aten.permute.default(primals_132, [1, 0]);  primals_132 = None
        view_158 = torch.ops.aten.view.default(mul_230, [256, 512]);  mul_230 = None
        mm_86 = torch.ops.aten.mm.default(view_158, permute_158)
        _unsafe_view_156 = torch.ops.aten._unsafe_view.default(mm_86, [2, 128, 1024])
        mul_231 = torch.ops.aten.mul.Tensor(_unsafe_view_156, 0.5)
        pow_37 = torch.ops.aten.pow.Tensor_Scalar(_unsafe_view_156, 3.0)
        mul_232 = torch.ops.aten.mul.Tensor(pow_37, 0.044715);  pow_37 = None
        add_102 = torch.ops.aten.add.Tensor(_unsafe_view_156, mul_232);  _unsafe_view_156 = mul_232 = None
        mul_233 = torch.ops.aten.mul.Tensor(add_102, 0.7978845608028654);  add_102 = None
        mul_234 = torch.ops.aten.mul.Tensor(mul_233, -2.0);  mul_233 = None
        exp_24 = torch.ops.aten.exp.default(mul_234);  mul_234 = None
        add_103 = torch.ops.aten.add.Tensor(exp_24, 1.0);  exp_24 = None
        reciprocal_36 = torch.ops.aten.reciprocal.default(add_103);  add_103 = None
        mul_235 = torch.ops.aten.mul.Tensor(reciprocal_36, 2.0);  reciprocal_36 = None
        sub_29 = torch.ops.aten.sub.Tensor(mul_235, 1.0);  mul_235 = None
        add_104 = torch.ops.aten.add.Tensor(sub_29, 1.0)
        mul_236 = torch.ops.aten.mul.Tensor(mul_231, add_104);  mul_231 = add_104 = None
        permute_159 = torch.ops.aten.permute.default(primals_133, [1, 0]);  primals_133 = None
        mm_87 = torch.ops.aten.mm.default(view_158, permute_159);  view_158 = None
        _unsafe_view_157 = torch.ops.aten._unsafe_view.default(mm_87, [2, 128, 1024])
        mul_237 = torch.ops.aten.mul.Tensor(mul_236, _unsafe_view_157);  mul_236 = _unsafe_view_157 = None
        rand_like_37 = torch.ops.aten.rand_like.default(mul_237, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_214 = torch.ops.aten.alias.default(rand_like_37);  rand_like_37 = None
        gt_52 = torch.ops.aten.gt.Scalar(alias_214, 0.1);  alias_214 = None
        mul_238 = torch.ops.aten.mul.Tensor(gt_52, mul_237);  mul_237 = None
        mul_239 = torch.ops.aten.mul.Tensor(mul_238, 1.1111111111111112);  mul_238 = None
        permute_160 = torch.ops.aten.permute.default(primals_134, [1, 0]);  primals_134 = None
        view_160 = torch.ops.aten.view.default(mul_239, [256, 1024]);  mul_239 = None
        mm_88 = torch.ops.aten.mm.default(view_160, permute_160)
        _unsafe_view_158 = torch.ops.aten._unsafe_view.default(mm_88, [2, 128, 512]);  mm_88 = None
        rand_like_38 = torch.ops.aten.rand_like.default(_unsafe_view_158, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_215 = torch.ops.aten.alias.default(rand_like_38);  rand_like_38 = None
        gt_53 = torch.ops.aten.gt.Scalar(alias_215, 0.1);  alias_215 = None
        mul_240 = torch.ops.aten.mul.Tensor(gt_53, _unsafe_view_158);  _unsafe_view_158 = None
        mul_241 = torch.ops.aten.mul.Tensor(mul_240, 1.1111111111111112);  mul_240 = None
        add_105 = torch.ops.aten.add.Tensor(add_100, mul_241);  mul_241 = None
        pow_38 = torch.ops.aten.pow.Tensor_Scalar(add_105, 2)
        mean_26 = torch.ops.aten.mean.dim(pow_38, [-1], True);  pow_38 = None
        add_106 = torch.ops.aten.add.Tensor(mean_26, 1e-06);  mean_26 = None
        sqrt_26 = torch.ops.aten.sqrt.default(add_106);  add_106 = None
        reciprocal_37 = torch.ops.aten.reciprocal.default(sqrt_26);  sqrt_26 = None
        mul_242 = torch.ops.aten.mul.Tensor(add_105, reciprocal_37)
        mul_243 = torch.ops.aten.mul.Tensor(primals_27, mul_242);  mul_242 = None
        permute_161 = torch.ops.aten.permute.default(primals_135, [1, 0]);  primals_135 = None
        view_161 = torch.ops.aten.view.default(mul_243, [256, 512]);  mul_243 = None
        mm_89 = torch.ops.aten.mm.default(view_161, permute_161)
        _unsafe_view_159 = torch.ops.aten._unsafe_view.default(mm_89, [2, 128, 384]);  mm_89 = None
        view_162 = torch.ops.aten.view.default(_unsafe_view_159, [2, -1, 6, 64]);  _unsafe_view_159 = None
        permute_162 = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None
        permute_163 = torch.ops.aten.permute.default(primals_136, [1, 0]);  primals_136 = None
        mm_90 = torch.ops.aten.mm.default(view_161, permute_163)
        _unsafe_view_160 = torch.ops.aten._unsafe_view.default(mm_90, [2, 128, 384]);  mm_90 = None
        view_164 = torch.ops.aten.view.default(_unsafe_view_160, [2, -1, 6, 64]);  _unsafe_view_160 = None
        permute_164 = torch.ops.aten.permute.default(view_164, [0, 2, 1, 3]);  view_164 = None
        permute_165 = torch.ops.aten.permute.default(primals_137, [1, 0]);  primals_137 = None
        mm_91 = torch.ops.aten.mm.default(view_161, permute_165);  view_161 = None
        _unsafe_view_161 = torch.ops.aten._unsafe_view.default(mm_91, [2, 128, 384]);  mm_91 = None
        view_166 = torch.ops.aten.view.default(_unsafe_view_161, [2, -1, 6, 64]);  _unsafe_view_161 = None
        permute_166 = torch.ops.aten.permute.default(view_166, [0, 2, 1, 3]);  view_166 = None
        permute_167 = torch.ops.aten.permute.default(permute_164, [0, 1, 3, 2])
        expand_56 = torch.ops.aten.expand.default(permute_162, [2, 6, 128, 64]);  permute_162 = None
        clone_56 = torch.ops.aten.clone.default(expand_56, memory_format = torch.contiguous_format);  expand_56 = None
        _unsafe_view_162 = torch.ops.aten._unsafe_view.default(clone_56, [12, 128, 64]);  clone_56 = None
        expand_57 = torch.ops.aten.expand.default(permute_167, [2, 6, 64, 128]);  permute_167 = None
        clone_57 = torch.ops.aten.clone.default(expand_57, memory_format = torch.contiguous_format);  expand_57 = None
        _unsafe_view_163 = torch.ops.aten._unsafe_view.default(clone_57, [12, 64, 128]);  clone_57 = None
        bmm_28 = torch.ops.aten.bmm.default(_unsafe_view_162, _unsafe_view_163)
        _unsafe_view_164 = torch.ops.aten._unsafe_view.default(bmm_28, [2, 6, 128, 128]);  bmm_28 = None
        add_107 = torch.ops.aten.add.Tensor(_unsafe_view_164, add_72);  _unsafe_view_164 = None
        amax_14 = torch.ops.aten.amax.default(add_107, [-1], True)
        sub_30 = torch.ops.aten.sub.Tensor(add_107, amax_14);  add_107 = amax_14 = None
        exp_25 = torch.ops.aten.exp.default(sub_30);  sub_30 = None
        sum_15 = torch.ops.aten.sum.dim_IntList(exp_25, [-1], True)
        div_18 = torch.ops.aten.div.Tensor(exp_25, sum_15);  exp_25 = sum_15 = None
        philox_rand_like_14 = torch.ops.prims.philox_rand_like.default(div_18, philox_seed_like, 2752512)
        gt_54 = torch.ops.aten.gt.Scalar(philox_rand_like_14, 0.1);  philox_rand_like_14 = None
        _to_copy_21 = torch.ops.aten._to_copy.default(gt_54, dtype = torch.float32);  gt_54 = None
        mul_244 = torch.ops.aten.mul.Tensor(_to_copy_21, div_18);  _to_copy_21 = None
        mul_245 = torch.ops.aten.mul.Tensor(mul_244, 1.1111111111111112);  mul_244 = None
        expand_58 = torch.ops.aten.expand.default(mul_245, [2, 6, 128, 128]);  mul_245 = None
        view_167 = torch.ops.aten.view.default(expand_58, [12, 128, 128]);  expand_58 = None
        expand_59 = torch.ops.aten.expand.default(permute_166, [2, 6, 128, 64])
        clone_58 = torch.ops.aten.clone.default(expand_59, memory_format = torch.contiguous_format);  expand_59 = None
        _unsafe_view_165 = torch.ops.aten._unsafe_view.default(clone_58, [12, 128, 64]);  clone_58 = None
        bmm_29 = torch.ops.aten.bmm.default(view_167, _unsafe_view_165)
        _unsafe_view_166 = torch.ops.aten._unsafe_view.default(bmm_29, [2, 6, 128, 64]);  bmm_29 = None
        permute_168 = torch.ops.aten.permute.default(_unsafe_view_166, [0, 2, 1, 3]);  _unsafe_view_166 = None
        clone_59 = torch.ops.aten.clone.default(permute_168, memory_format = torch.contiguous_format);  permute_168 = None
        view_168 = torch.ops.aten.view.default(clone_59, [2, -1, 384]);  clone_59 = None
        permute_169 = torch.ops.aten.permute.default(primals_138, [1, 0]);  primals_138 = None
        view_169 = torch.ops.aten.view.default(view_168, [256, 384]);  view_168 = None
        mm_92 = torch.ops.aten.mm.default(view_169, permute_169)
        _unsafe_view_167 = torch.ops.aten._unsafe_view.default(mm_92, [2, 128, 512]);  mm_92 = None
        rand_like_39 = torch.ops.aten.rand_like.default(_unsafe_view_167, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_222 = torch.ops.aten.alias.default(rand_like_39);  rand_like_39 = None
        gt_55 = torch.ops.aten.gt.Scalar(alias_222, 0.1);  alias_222 = None
        mul_246 = torch.ops.aten.mul.Tensor(gt_55, _unsafe_view_167);  _unsafe_view_167 = None
        mul_247 = torch.ops.aten.mul.Tensor(mul_246, 1.1111111111111112);  mul_246 = None
        add_108 = torch.ops.aten.add.Tensor(add_105, mul_247);  mul_247 = None
        pow_39 = torch.ops.aten.pow.Tensor_Scalar(add_108, 2)
        mean_27 = torch.ops.aten.mean.dim(pow_39, [-1], True);  pow_39 = None
        add_109 = torch.ops.aten.add.Tensor(mean_27, 1e-06);  mean_27 = None
        sqrt_27 = torch.ops.aten.sqrt.default(add_109);  add_109 = None
        reciprocal_38 = torch.ops.aten.reciprocal.default(sqrt_27);  sqrt_27 = None
        mul_248 = torch.ops.aten.mul.Tensor(add_108, reciprocal_38)
        mul_249 = torch.ops.aten.mul.Tensor(primals_28, mul_248);  mul_248 = None
        permute_170 = torch.ops.aten.permute.default(primals_139, [1, 0]);  primals_139 = None
        view_170 = torch.ops.aten.view.default(mul_249, [256, 512]);  mul_249 = None
        mm_93 = torch.ops.aten.mm.default(view_170, permute_170);  view_170 = None
        _unsafe_view_168 = torch.ops.aten._unsafe_view.default(mm_93, [2, 128, 384]);  mm_93 = None
        view_171 = torch.ops.aten.view.default(_unsafe_view_168, [2, -1, 6, 64]);  _unsafe_view_168 = None
        permute_171 = torch.ops.aten.permute.default(view_171, [0, 2, 1, 3]);  view_171 = None
        permute_172 = torch.ops.aten.permute.default(primals_140, [1, 0]);  primals_140 = None
        mm_94 = torch.ops.aten.mm.default(view_109, permute_172)
        _unsafe_view_169 = torch.ops.aten._unsafe_view.default(mm_94, [2, 128, 384]);  mm_94 = None
        view_173 = torch.ops.aten.view.default(_unsafe_view_169, [2, -1, 6, 64]);  _unsafe_view_169 = None
        permute_173 = torch.ops.aten.permute.default(view_173, [0, 2, 1, 3]);  view_173 = None
        permute_174 = torch.ops.aten.permute.default(primals_141, [1, 0]);  primals_141 = None
        mm_95 = torch.ops.aten.mm.default(view_109, permute_174)
        _unsafe_view_170 = torch.ops.aten._unsafe_view.default(mm_95, [2, 128, 384]);  mm_95 = None
        view_175 = torch.ops.aten.view.default(_unsafe_view_170, [2, -1, 6, 64]);  _unsafe_view_170 = None
        permute_175 = torch.ops.aten.permute.default(view_175, [0, 2, 1, 3]);  view_175 = None
        permute_176 = torch.ops.aten.permute.default(permute_173, [0, 1, 3, 2])
        expand_60 = torch.ops.aten.expand.default(permute_171, [2, 6, 128, 64]);  permute_171 = None
        clone_60 = torch.ops.aten.clone.default(expand_60, memory_format = torch.contiguous_format);  expand_60 = None
        _unsafe_view_171 = torch.ops.aten._unsafe_view.default(clone_60, [12, 128, 64]);  clone_60 = None
        expand_61 = torch.ops.aten.expand.default(permute_176, [2, 6, 64, 128]);  permute_176 = None
        clone_61 = torch.ops.aten.clone.default(expand_61, memory_format = torch.contiguous_format);  expand_61 = None
        _unsafe_view_172 = torch.ops.aten._unsafe_view.default(clone_61, [12, 64, 128]);  clone_61 = None
        bmm_30 = torch.ops.aten.bmm.default(_unsafe_view_171, _unsafe_view_172)
        _unsafe_view_173 = torch.ops.aten._unsafe_view.default(bmm_30, [2, 6, 128, 128]);  bmm_30 = None
        add_110 = torch.ops.aten.add.Tensor(_unsafe_view_173, add_76);  _unsafe_view_173 = None
        amax_15 = torch.ops.aten.amax.default(add_110, [-1], True)
        sub_31 = torch.ops.aten.sub.Tensor(add_110, amax_15);  add_110 = amax_15 = None
        exp_26 = torch.ops.aten.exp.default(sub_31);  sub_31 = None
        sum_16 = torch.ops.aten.sum.dim_IntList(exp_26, [-1], True)
        div_19 = torch.ops.aten.div.Tensor(exp_26, sum_16);  exp_26 = sum_16 = None
        philox_rand_like_15 = torch.ops.prims.philox_rand_like.default(div_19, philox_seed_like, 2949120)
        gt_56 = torch.ops.aten.gt.Scalar(philox_rand_like_15, 0.1);  philox_rand_like_15 = None
        _to_copy_22 = torch.ops.aten._to_copy.default(gt_56, dtype = torch.float32);  gt_56 = None
        mul_250 = torch.ops.aten.mul.Tensor(_to_copy_22, div_19);  _to_copy_22 = None
        mul_251 = torch.ops.aten.mul.Tensor(mul_250, 1.1111111111111112);  mul_250 = None
        expand_62 = torch.ops.aten.expand.default(mul_251, [2, 6, 128, 128]);  mul_251 = None
        view_176 = torch.ops.aten.view.default(expand_62, [12, 128, 128]);  expand_62 = None
        expand_63 = torch.ops.aten.expand.default(permute_175, [2, 6, 128, 64])
        clone_62 = torch.ops.aten.clone.default(expand_63, memory_format = torch.contiguous_format);  expand_63 = None
        _unsafe_view_174 = torch.ops.aten._unsafe_view.default(clone_62, [12, 128, 64]);  clone_62 = None
        bmm_31 = torch.ops.aten.bmm.default(view_176, _unsafe_view_174)
        _unsafe_view_175 = torch.ops.aten._unsafe_view.default(bmm_31, [2, 6, 128, 64]);  bmm_31 = None
        permute_177 = torch.ops.aten.permute.default(_unsafe_view_175, [0, 2, 1, 3]);  _unsafe_view_175 = None
        clone_63 = torch.ops.aten.clone.default(permute_177, memory_format = torch.contiguous_format);  permute_177 = None
        view_177 = torch.ops.aten.view.default(clone_63, [2, -1, 384]);  clone_63 = None
        permute_178 = torch.ops.aten.permute.default(primals_142, [1, 0]);  primals_142 = None
        view_178 = torch.ops.aten.view.default(view_177, [256, 384]);  view_177 = None
        mm_96 = torch.ops.aten.mm.default(view_178, permute_178)
        _unsafe_view_176 = torch.ops.aten._unsafe_view.default(mm_96, [2, 128, 512]);  mm_96 = None
        rand_like_40 = torch.ops.aten.rand_like.default(_unsafe_view_176, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_229 = torch.ops.aten.alias.default(rand_like_40);  rand_like_40 = None
        gt_57 = torch.ops.aten.gt.Scalar(alias_229, 0.1);  alias_229 = None
        mul_252 = torch.ops.aten.mul.Tensor(gt_57, _unsafe_view_176);  _unsafe_view_176 = None
        mul_253 = torch.ops.aten.mul.Tensor(mul_252, 1.1111111111111112);  mul_252 = None
        add_111 = torch.ops.aten.add.Tensor(add_108, mul_253);  mul_253 = None
        pow_40 = torch.ops.aten.pow.Tensor_Scalar(add_111, 2)
        mean_28 = torch.ops.aten.mean.dim(pow_40, [-1], True);  pow_40 = None
        add_112 = torch.ops.aten.add.Tensor(mean_28, 1e-06);  mean_28 = None
        sqrt_28 = torch.ops.aten.sqrt.default(add_112);  add_112 = None
        reciprocal_39 = torch.ops.aten.reciprocal.default(sqrt_28);  sqrt_28 = None
        mul_254 = torch.ops.aten.mul.Tensor(add_111, reciprocal_39)
        mul_255 = torch.ops.aten.mul.Tensor(primals_29, mul_254);  mul_254 = None
        permute_179 = torch.ops.aten.permute.default(primals_143, [1, 0]);  primals_143 = None
        view_179 = torch.ops.aten.view.default(mul_255, [256, 512]);  mul_255 = None
        mm_97 = torch.ops.aten.mm.default(view_179, permute_179)
        _unsafe_view_177 = torch.ops.aten._unsafe_view.default(mm_97, [2, 128, 1024])
        mul_256 = torch.ops.aten.mul.Tensor(_unsafe_view_177, 0.5)
        pow_41 = torch.ops.aten.pow.Tensor_Scalar(_unsafe_view_177, 3.0)
        mul_257 = torch.ops.aten.mul.Tensor(pow_41, 0.044715);  pow_41 = None
        add_113 = torch.ops.aten.add.Tensor(_unsafe_view_177, mul_257);  _unsafe_view_177 = mul_257 = None
        mul_258 = torch.ops.aten.mul.Tensor(add_113, 0.7978845608028654);  add_113 = None
        mul_259 = torch.ops.aten.mul.Tensor(mul_258, -2.0);  mul_258 = None
        exp_27 = torch.ops.aten.exp.default(mul_259);  mul_259 = None
        add_114 = torch.ops.aten.add.Tensor(exp_27, 1.0);  exp_27 = None
        reciprocal_40 = torch.ops.aten.reciprocal.default(add_114);  add_114 = None
        mul_260 = torch.ops.aten.mul.Tensor(reciprocal_40, 2.0);  reciprocal_40 = None
        sub_32 = torch.ops.aten.sub.Tensor(mul_260, 1.0);  mul_260 = None
        add_115 = torch.ops.aten.add.Tensor(sub_32, 1.0)
        mul_261 = torch.ops.aten.mul.Tensor(mul_256, add_115);  mul_256 = add_115 = None
        permute_180 = torch.ops.aten.permute.default(primals_144, [1, 0]);  primals_144 = None
        mm_98 = torch.ops.aten.mm.default(view_179, permute_180);  view_179 = None
        _unsafe_view_178 = torch.ops.aten._unsafe_view.default(mm_98, [2, 128, 1024])
        mul_262 = torch.ops.aten.mul.Tensor(mul_261, _unsafe_view_178);  mul_261 = _unsafe_view_178 = None
        rand_like_41 = torch.ops.aten.rand_like.default(mul_262, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_236 = torch.ops.aten.alias.default(rand_like_41);  rand_like_41 = None
        gt_58 = torch.ops.aten.gt.Scalar(alias_236, 0.1);  alias_236 = None
        mul_263 = torch.ops.aten.mul.Tensor(gt_58, mul_262);  mul_262 = None
        mul_264 = torch.ops.aten.mul.Tensor(mul_263, 1.1111111111111112);  mul_263 = None
        permute_181 = torch.ops.aten.permute.default(primals_145, [1, 0]);  primals_145 = None
        view_181 = torch.ops.aten.view.default(mul_264, [256, 1024]);  mul_264 = None
        mm_99 = torch.ops.aten.mm.default(view_181, permute_181)
        _unsafe_view_179 = torch.ops.aten._unsafe_view.default(mm_99, [2, 128, 512]);  mm_99 = None
        rand_like_42 = torch.ops.aten.rand_like.default(_unsafe_view_179, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_237 = torch.ops.aten.alias.default(rand_like_42);  rand_like_42 = None
        gt_59 = torch.ops.aten.gt.Scalar(alias_237, 0.1);  alias_237 = None
        mul_265 = torch.ops.aten.mul.Tensor(gt_59, _unsafe_view_179);  _unsafe_view_179 = None
        mul_266 = torch.ops.aten.mul.Tensor(mul_265, 1.1111111111111112);  mul_265 = None
        add_116 = torch.ops.aten.add.Tensor(add_111, mul_266);  mul_266 = None
        pow_42 = torch.ops.aten.pow.Tensor_Scalar(add_116, 2)
        mean_29 = torch.ops.aten.mean.dim(pow_42, [-1], True);  pow_42 = None
        add_117 = torch.ops.aten.add.Tensor(mean_29, 1e-06);  mean_29 = None
        sqrt_29 = torch.ops.aten.sqrt.default(add_117);  add_117 = None
        reciprocal_41 = torch.ops.aten.reciprocal.default(sqrt_29);  sqrt_29 = None
        mul_267 = torch.ops.aten.mul.Tensor(add_116, reciprocal_41)
        mul_268 = torch.ops.aten.mul.Tensor(primals_30, mul_267);  mul_267 = None
        permute_182 = torch.ops.aten.permute.default(primals_146, [1, 0]);  primals_146 = None
        view_182 = torch.ops.aten.view.default(mul_268, [256, 512]);  mul_268 = None
        mm_100 = torch.ops.aten.mm.default(view_182, permute_182)
        _unsafe_view_180 = torch.ops.aten._unsafe_view.default(mm_100, [2, 128, 384]);  mm_100 = None
        view_183 = torch.ops.aten.view.default(_unsafe_view_180, [2, -1, 6, 64]);  _unsafe_view_180 = None
        permute_183 = torch.ops.aten.permute.default(view_183, [0, 2, 1, 3]);  view_183 = None
        permute_184 = torch.ops.aten.permute.default(primals_147, [1, 0]);  primals_147 = None
        mm_101 = torch.ops.aten.mm.default(view_182, permute_184)
        _unsafe_view_181 = torch.ops.aten._unsafe_view.default(mm_101, [2, 128, 384]);  mm_101 = None
        view_185 = torch.ops.aten.view.default(_unsafe_view_181, [2, -1, 6, 64]);  _unsafe_view_181 = None
        permute_185 = torch.ops.aten.permute.default(view_185, [0, 2, 1, 3]);  view_185 = None
        permute_186 = torch.ops.aten.permute.default(primals_148, [1, 0]);  primals_148 = None
        mm_102 = torch.ops.aten.mm.default(view_182, permute_186);  view_182 = None
        _unsafe_view_182 = torch.ops.aten._unsafe_view.default(mm_102, [2, 128, 384]);  mm_102 = None
        view_187 = torch.ops.aten.view.default(_unsafe_view_182, [2, -1, 6, 64]);  _unsafe_view_182 = None
        permute_187 = torch.ops.aten.permute.default(view_187, [0, 2, 1, 3]);  view_187 = None
        permute_188 = torch.ops.aten.permute.default(permute_185, [0, 1, 3, 2])
        expand_64 = torch.ops.aten.expand.default(permute_183, [2, 6, 128, 64]);  permute_183 = None
        clone_64 = torch.ops.aten.clone.default(expand_64, memory_format = torch.contiguous_format);  expand_64 = None
        _unsafe_view_183 = torch.ops.aten._unsafe_view.default(clone_64, [12, 128, 64]);  clone_64 = None
        expand_65 = torch.ops.aten.expand.default(permute_188, [2, 6, 64, 128]);  permute_188 = None
        clone_65 = torch.ops.aten.clone.default(expand_65, memory_format = torch.contiguous_format);  expand_65 = None
        _unsafe_view_184 = torch.ops.aten._unsafe_view.default(clone_65, [12, 64, 128]);  clone_65 = None
        bmm_32 = torch.ops.aten.bmm.default(_unsafe_view_183, _unsafe_view_184)
        _unsafe_view_185 = torch.ops.aten._unsafe_view.default(bmm_32, [2, 6, 128, 128]);  bmm_32 = None
        add_118 = torch.ops.aten.add.Tensor(_unsafe_view_185, add_72);  _unsafe_view_185 = None
        amax_16 = torch.ops.aten.amax.default(add_118, [-1], True)
        sub_33 = torch.ops.aten.sub.Tensor(add_118, amax_16);  add_118 = amax_16 = None
        exp_28 = torch.ops.aten.exp.default(sub_33);  sub_33 = None
        sum_17 = torch.ops.aten.sum.dim_IntList(exp_28, [-1], True)
        div_20 = torch.ops.aten.div.Tensor(exp_28, sum_17);  exp_28 = sum_17 = None
        philox_rand_like_16 = torch.ops.prims.philox_rand_like.default(div_20, philox_seed_like, 3145728)
        gt_60 = torch.ops.aten.gt.Scalar(philox_rand_like_16, 0.1);  philox_rand_like_16 = None
        _to_copy_23 = torch.ops.aten._to_copy.default(gt_60, dtype = torch.float32);  gt_60 = None
        mul_269 = torch.ops.aten.mul.Tensor(_to_copy_23, div_20);  _to_copy_23 = None
        mul_270 = torch.ops.aten.mul.Tensor(mul_269, 1.1111111111111112);  mul_269 = None
        expand_66 = torch.ops.aten.expand.default(mul_270, [2, 6, 128, 128]);  mul_270 = None
        view_188 = torch.ops.aten.view.default(expand_66, [12, 128, 128]);  expand_66 = None
        expand_67 = torch.ops.aten.expand.default(permute_187, [2, 6, 128, 64])
        clone_66 = torch.ops.aten.clone.default(expand_67, memory_format = torch.contiguous_format);  expand_67 = None
        _unsafe_view_186 = torch.ops.aten._unsafe_view.default(clone_66, [12, 128, 64]);  clone_66 = None
        bmm_33 = torch.ops.aten.bmm.default(view_188, _unsafe_view_186)
        _unsafe_view_187 = torch.ops.aten._unsafe_view.default(bmm_33, [2, 6, 128, 64]);  bmm_33 = None
        permute_189 = torch.ops.aten.permute.default(_unsafe_view_187, [0, 2, 1, 3]);  _unsafe_view_187 = None
        clone_67 = torch.ops.aten.clone.default(permute_189, memory_format = torch.contiguous_format);  permute_189 = None
        view_189 = torch.ops.aten.view.default(clone_67, [2, -1, 384]);  clone_67 = None
        permute_190 = torch.ops.aten.permute.default(primals_149, [1, 0]);  primals_149 = None
        view_190 = torch.ops.aten.view.default(view_189, [256, 384]);  view_189 = None
        mm_103 = torch.ops.aten.mm.default(view_190, permute_190)
        _unsafe_view_188 = torch.ops.aten._unsafe_view.default(mm_103, [2, 128, 512]);  mm_103 = None
        rand_like_43 = torch.ops.aten.rand_like.default(_unsafe_view_188, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_244 = torch.ops.aten.alias.default(rand_like_43);  rand_like_43 = None
        gt_61 = torch.ops.aten.gt.Scalar(alias_244, 0.1);  alias_244 = None
        mul_271 = torch.ops.aten.mul.Tensor(gt_61, _unsafe_view_188);  _unsafe_view_188 = None
        mul_272 = torch.ops.aten.mul.Tensor(mul_271, 1.1111111111111112);  mul_271 = None
        add_119 = torch.ops.aten.add.Tensor(add_116, mul_272);  mul_272 = None
        pow_43 = torch.ops.aten.pow.Tensor_Scalar(add_119, 2)
        mean_30 = torch.ops.aten.mean.dim(pow_43, [-1], True);  pow_43 = None
        add_120 = torch.ops.aten.add.Tensor(mean_30, 1e-06);  mean_30 = None
        sqrt_30 = torch.ops.aten.sqrt.default(add_120);  add_120 = None
        reciprocal_42 = torch.ops.aten.reciprocal.default(sqrt_30);  sqrt_30 = None
        mul_273 = torch.ops.aten.mul.Tensor(add_119, reciprocal_42)
        mul_274 = torch.ops.aten.mul.Tensor(primals_31, mul_273);  mul_273 = None
        permute_191 = torch.ops.aten.permute.default(primals_150, [1, 0]);  primals_150 = None
        view_191 = torch.ops.aten.view.default(mul_274, [256, 512]);  mul_274 = None
        mm_104 = torch.ops.aten.mm.default(view_191, permute_191);  view_191 = None
        _unsafe_view_189 = torch.ops.aten._unsafe_view.default(mm_104, [2, 128, 384]);  mm_104 = None
        view_192 = torch.ops.aten.view.default(_unsafe_view_189, [2, -1, 6, 64]);  _unsafe_view_189 = None
        permute_192 = torch.ops.aten.permute.default(view_192, [0, 2, 1, 3]);  view_192 = None
        permute_193 = torch.ops.aten.permute.default(primals_151, [1, 0]);  primals_151 = None
        mm_105 = torch.ops.aten.mm.default(view_109, permute_193)
        _unsafe_view_190 = torch.ops.aten._unsafe_view.default(mm_105, [2, 128, 384]);  mm_105 = None
        view_194 = torch.ops.aten.view.default(_unsafe_view_190, [2, -1, 6, 64]);  _unsafe_view_190 = None
        permute_194 = torch.ops.aten.permute.default(view_194, [0, 2, 1, 3]);  view_194 = None
        permute_195 = torch.ops.aten.permute.default(primals_152, [1, 0]);  primals_152 = None
        mm_106 = torch.ops.aten.mm.default(view_109, permute_195)
        _unsafe_view_191 = torch.ops.aten._unsafe_view.default(mm_106, [2, 128, 384]);  mm_106 = None
        view_196 = torch.ops.aten.view.default(_unsafe_view_191, [2, -1, 6, 64]);  _unsafe_view_191 = None
        permute_196 = torch.ops.aten.permute.default(view_196, [0, 2, 1, 3]);  view_196 = None
        permute_197 = torch.ops.aten.permute.default(permute_194, [0, 1, 3, 2])
        expand_68 = torch.ops.aten.expand.default(permute_192, [2, 6, 128, 64]);  permute_192 = None
        clone_68 = torch.ops.aten.clone.default(expand_68, memory_format = torch.contiguous_format);  expand_68 = None
        _unsafe_view_192 = torch.ops.aten._unsafe_view.default(clone_68, [12, 128, 64]);  clone_68 = None
        expand_69 = torch.ops.aten.expand.default(permute_197, [2, 6, 64, 128]);  permute_197 = None
        clone_69 = torch.ops.aten.clone.default(expand_69, memory_format = torch.contiguous_format);  expand_69 = None
        _unsafe_view_193 = torch.ops.aten._unsafe_view.default(clone_69, [12, 64, 128]);  clone_69 = None
        bmm_34 = torch.ops.aten.bmm.default(_unsafe_view_192, _unsafe_view_193)
        _unsafe_view_194 = torch.ops.aten._unsafe_view.default(bmm_34, [2, 6, 128, 128]);  bmm_34 = None
        add_121 = torch.ops.aten.add.Tensor(_unsafe_view_194, add_76);  _unsafe_view_194 = None
        amax_17 = torch.ops.aten.amax.default(add_121, [-1], True)
        sub_34 = torch.ops.aten.sub.Tensor(add_121, amax_17);  add_121 = amax_17 = None
        exp_29 = torch.ops.aten.exp.default(sub_34);  sub_34 = None
        sum_18 = torch.ops.aten.sum.dim_IntList(exp_29, [-1], True)
        div_21 = torch.ops.aten.div.Tensor(exp_29, sum_18);  exp_29 = sum_18 = None
        philox_rand_like_17 = torch.ops.prims.philox_rand_like.default(div_21, philox_seed_like, 3342336)
        gt_62 = torch.ops.aten.gt.Scalar(philox_rand_like_17, 0.1);  philox_rand_like_17 = None
        _to_copy_24 = torch.ops.aten._to_copy.default(gt_62, dtype = torch.float32);  gt_62 = None
        mul_275 = torch.ops.aten.mul.Tensor(_to_copy_24, div_21);  _to_copy_24 = None
        mul_276 = torch.ops.aten.mul.Tensor(mul_275, 1.1111111111111112);  mul_275 = None
        expand_70 = torch.ops.aten.expand.default(mul_276, [2, 6, 128, 128]);  mul_276 = None
        view_197 = torch.ops.aten.view.default(expand_70, [12, 128, 128]);  expand_70 = None
        expand_71 = torch.ops.aten.expand.default(permute_196, [2, 6, 128, 64])
        clone_70 = torch.ops.aten.clone.default(expand_71, memory_format = torch.contiguous_format);  expand_71 = None
        _unsafe_view_195 = torch.ops.aten._unsafe_view.default(clone_70, [12, 128, 64]);  clone_70 = None
        bmm_35 = torch.ops.aten.bmm.default(view_197, _unsafe_view_195)
        _unsafe_view_196 = torch.ops.aten._unsafe_view.default(bmm_35, [2, 6, 128, 64]);  bmm_35 = None
        permute_198 = torch.ops.aten.permute.default(_unsafe_view_196, [0, 2, 1, 3]);  _unsafe_view_196 = None
        clone_71 = torch.ops.aten.clone.default(permute_198, memory_format = torch.contiguous_format);  permute_198 = None
        view_198 = torch.ops.aten.view.default(clone_71, [2, -1, 384]);  clone_71 = None
        permute_199 = torch.ops.aten.permute.default(primals_153, [1, 0]);  primals_153 = None
        view_199 = torch.ops.aten.view.default(view_198, [256, 384]);  view_198 = None
        mm_107 = torch.ops.aten.mm.default(view_199, permute_199)
        _unsafe_view_197 = torch.ops.aten._unsafe_view.default(mm_107, [2, 128, 512]);  mm_107 = None
        rand_like_44 = torch.ops.aten.rand_like.default(_unsafe_view_197, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_251 = torch.ops.aten.alias.default(rand_like_44);  rand_like_44 = None
        gt_63 = torch.ops.aten.gt.Scalar(alias_251, 0.1);  alias_251 = None
        mul_277 = torch.ops.aten.mul.Tensor(gt_63, _unsafe_view_197);  _unsafe_view_197 = None
        mul_278 = torch.ops.aten.mul.Tensor(mul_277, 1.1111111111111112);  mul_277 = None
        add_122 = torch.ops.aten.add.Tensor(add_119, mul_278);  mul_278 = None
        pow_44 = torch.ops.aten.pow.Tensor_Scalar(add_122, 2)
        mean_31 = torch.ops.aten.mean.dim(pow_44, [-1], True);  pow_44 = None
        add_123 = torch.ops.aten.add.Tensor(mean_31, 1e-06);  mean_31 = None
        sqrt_31 = torch.ops.aten.sqrt.default(add_123);  add_123 = None
        reciprocal_43 = torch.ops.aten.reciprocal.default(sqrt_31);  sqrt_31 = None
        mul_279 = torch.ops.aten.mul.Tensor(add_122, reciprocal_43)
        mul_280 = torch.ops.aten.mul.Tensor(primals_32, mul_279);  mul_279 = None
        permute_200 = torch.ops.aten.permute.default(primals_154, [1, 0]);  primals_154 = None
        view_200 = torch.ops.aten.view.default(mul_280, [256, 512]);  mul_280 = None
        mm_108 = torch.ops.aten.mm.default(view_200, permute_200)
        _unsafe_view_198 = torch.ops.aten._unsafe_view.default(mm_108, [2, 128, 1024])
        mul_281 = torch.ops.aten.mul.Tensor(_unsafe_view_198, 0.5)
        pow_45 = torch.ops.aten.pow.Tensor_Scalar(_unsafe_view_198, 3.0)
        mul_282 = torch.ops.aten.mul.Tensor(pow_45, 0.044715);  pow_45 = None
        add_124 = torch.ops.aten.add.Tensor(_unsafe_view_198, mul_282);  _unsafe_view_198 = mul_282 = None
        mul_283 = torch.ops.aten.mul.Tensor(add_124, 0.7978845608028654);  add_124 = None
        mul_284 = torch.ops.aten.mul.Tensor(mul_283, -2.0);  mul_283 = None
        exp_30 = torch.ops.aten.exp.default(mul_284);  mul_284 = None
        add_125 = torch.ops.aten.add.Tensor(exp_30, 1.0);  exp_30 = None
        reciprocal_44 = torch.ops.aten.reciprocal.default(add_125);  add_125 = None
        mul_285 = torch.ops.aten.mul.Tensor(reciprocal_44, 2.0);  reciprocal_44 = None
        sub_35 = torch.ops.aten.sub.Tensor(mul_285, 1.0);  mul_285 = None
        add_126 = torch.ops.aten.add.Tensor(sub_35, 1.0)
        mul_286 = torch.ops.aten.mul.Tensor(mul_281, add_126);  mul_281 = add_126 = None
        permute_201 = torch.ops.aten.permute.default(primals_155, [1, 0]);  primals_155 = None
        mm_109 = torch.ops.aten.mm.default(view_200, permute_201);  view_200 = None
        _unsafe_view_199 = torch.ops.aten._unsafe_view.default(mm_109, [2, 128, 1024])
        mul_287 = torch.ops.aten.mul.Tensor(mul_286, _unsafe_view_199);  mul_286 = _unsafe_view_199 = None
        rand_like_45 = torch.ops.aten.rand_like.default(mul_287, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_258 = torch.ops.aten.alias.default(rand_like_45);  rand_like_45 = None
        gt_64 = torch.ops.aten.gt.Scalar(alias_258, 0.1);  alias_258 = None
        mul_288 = torch.ops.aten.mul.Tensor(gt_64, mul_287);  mul_287 = None
        mul_289 = torch.ops.aten.mul.Tensor(mul_288, 1.1111111111111112);  mul_288 = None
        permute_202 = torch.ops.aten.permute.default(primals_156, [1, 0]);  primals_156 = None
        view_202 = torch.ops.aten.view.default(mul_289, [256, 1024]);  mul_289 = None
        mm_110 = torch.ops.aten.mm.default(view_202, permute_202)
        _unsafe_view_200 = torch.ops.aten._unsafe_view.default(mm_110, [2, 128, 512]);  mm_110 = None
        rand_like_46 = torch.ops.aten.rand_like.default(_unsafe_view_200, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_259 = torch.ops.aten.alias.default(rand_like_46);  rand_like_46 = None
        gt_65 = torch.ops.aten.gt.Scalar(alias_259, 0.1);  alias_259 = None
        mul_290 = torch.ops.aten.mul.Tensor(gt_65, _unsafe_view_200);  _unsafe_view_200 = None
        mul_291 = torch.ops.aten.mul.Tensor(mul_290, 1.1111111111111112);  mul_290 = None
        add_127 = torch.ops.aten.add.Tensor(add_122, mul_291);  mul_291 = None
        pow_46 = torch.ops.aten.pow.Tensor_Scalar(add_127, 2)
        mean_32 = torch.ops.aten.mean.dim(pow_46, [-1], True);  pow_46 = None
        add_128 = torch.ops.aten.add.Tensor(mean_32, 1e-06);  mean_32 = None
        sqrt_32 = torch.ops.aten.sqrt.default(add_128);  add_128 = None
        reciprocal_45 = torch.ops.aten.reciprocal.default(sqrt_32);  sqrt_32 = None
        mul_292 = torch.ops.aten.mul.Tensor(add_127, reciprocal_45)
        mul_293 = torch.ops.aten.mul.Tensor(primals_33, mul_292);  mul_292 = None
        permute_203 = torch.ops.aten.permute.default(primals_157, [1, 0]);  primals_157 = None
        view_203 = torch.ops.aten.view.default(mul_293, [256, 512]);  mul_293 = None
        mm_111 = torch.ops.aten.mm.default(view_203, permute_203)
        _unsafe_view_201 = torch.ops.aten._unsafe_view.default(mm_111, [2, 128, 384]);  mm_111 = None
        view_204 = torch.ops.aten.view.default(_unsafe_view_201, [2, -1, 6, 64]);  _unsafe_view_201 = None
        permute_204 = torch.ops.aten.permute.default(view_204, [0, 2, 1, 3]);  view_204 = None
        permute_205 = torch.ops.aten.permute.default(primals_158, [1, 0]);  primals_158 = None
        mm_112 = torch.ops.aten.mm.default(view_203, permute_205)
        _unsafe_view_202 = torch.ops.aten._unsafe_view.default(mm_112, [2, 128, 384]);  mm_112 = None
        view_206 = torch.ops.aten.view.default(_unsafe_view_202, [2, -1, 6, 64]);  _unsafe_view_202 = None
        permute_206 = torch.ops.aten.permute.default(view_206, [0, 2, 1, 3]);  view_206 = None
        permute_207 = torch.ops.aten.permute.default(primals_159, [1, 0]);  primals_159 = None
        mm_113 = torch.ops.aten.mm.default(view_203, permute_207);  view_203 = None
        _unsafe_view_203 = torch.ops.aten._unsafe_view.default(mm_113, [2, 128, 384]);  mm_113 = None
        view_208 = torch.ops.aten.view.default(_unsafe_view_203, [2, -1, 6, 64]);  _unsafe_view_203 = None
        permute_208 = torch.ops.aten.permute.default(view_208, [0, 2, 1, 3]);  view_208 = None
        permute_209 = torch.ops.aten.permute.default(permute_206, [0, 1, 3, 2])
        expand_72 = torch.ops.aten.expand.default(permute_204, [2, 6, 128, 64]);  permute_204 = None
        clone_72 = torch.ops.aten.clone.default(expand_72, memory_format = torch.contiguous_format);  expand_72 = None
        _unsafe_view_204 = torch.ops.aten._unsafe_view.default(clone_72, [12, 128, 64]);  clone_72 = None
        expand_73 = torch.ops.aten.expand.default(permute_209, [2, 6, 64, 128]);  permute_209 = None
        clone_73 = torch.ops.aten.clone.default(expand_73, memory_format = torch.contiguous_format);  expand_73 = None
        _unsafe_view_205 = torch.ops.aten._unsafe_view.default(clone_73, [12, 64, 128]);  clone_73 = None
        bmm_36 = torch.ops.aten.bmm.default(_unsafe_view_204, _unsafe_view_205)
        _unsafe_view_206 = torch.ops.aten._unsafe_view.default(bmm_36, [2, 6, 128, 128]);  bmm_36 = None
        add_129 = torch.ops.aten.add.Tensor(_unsafe_view_206, add_72);  _unsafe_view_206 = None
        amax_18 = torch.ops.aten.amax.default(add_129, [-1], True)
        sub_36 = torch.ops.aten.sub.Tensor(add_129, amax_18);  add_129 = amax_18 = None
        exp_31 = torch.ops.aten.exp.default(sub_36);  sub_36 = None
        sum_19 = torch.ops.aten.sum.dim_IntList(exp_31, [-1], True)
        div_22 = torch.ops.aten.div.Tensor(exp_31, sum_19);  exp_31 = sum_19 = None
        philox_rand_like_18 = torch.ops.prims.philox_rand_like.default(div_22, philox_seed_like, 3538944)
        gt_66 = torch.ops.aten.gt.Scalar(philox_rand_like_18, 0.1);  philox_rand_like_18 = None
        _to_copy_25 = torch.ops.aten._to_copy.default(gt_66, dtype = torch.float32);  gt_66 = None
        mul_294 = torch.ops.aten.mul.Tensor(_to_copy_25, div_22);  _to_copy_25 = None
        mul_295 = torch.ops.aten.mul.Tensor(mul_294, 1.1111111111111112);  mul_294 = None
        expand_74 = torch.ops.aten.expand.default(mul_295, [2, 6, 128, 128]);  mul_295 = None
        view_209 = torch.ops.aten.view.default(expand_74, [12, 128, 128]);  expand_74 = None
        expand_75 = torch.ops.aten.expand.default(permute_208, [2, 6, 128, 64])
        clone_74 = torch.ops.aten.clone.default(expand_75, memory_format = torch.contiguous_format);  expand_75 = None
        _unsafe_view_207 = torch.ops.aten._unsafe_view.default(clone_74, [12, 128, 64]);  clone_74 = None
        bmm_37 = torch.ops.aten.bmm.default(view_209, _unsafe_view_207)
        _unsafe_view_208 = torch.ops.aten._unsafe_view.default(bmm_37, [2, 6, 128, 64]);  bmm_37 = None
        permute_210 = torch.ops.aten.permute.default(_unsafe_view_208, [0, 2, 1, 3]);  _unsafe_view_208 = None
        clone_75 = torch.ops.aten.clone.default(permute_210, memory_format = torch.contiguous_format);  permute_210 = None
        view_210 = torch.ops.aten.view.default(clone_75, [2, -1, 384]);  clone_75 = None
        permute_211 = torch.ops.aten.permute.default(primals_160, [1, 0]);  primals_160 = None
        view_211 = torch.ops.aten.view.default(view_210, [256, 384]);  view_210 = None
        mm_114 = torch.ops.aten.mm.default(view_211, permute_211)
        _unsafe_view_209 = torch.ops.aten._unsafe_view.default(mm_114, [2, 128, 512]);  mm_114 = None
        rand_like_47 = torch.ops.aten.rand_like.default(_unsafe_view_209, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_266 = torch.ops.aten.alias.default(rand_like_47);  rand_like_47 = None
        gt_67 = torch.ops.aten.gt.Scalar(alias_266, 0.1);  alias_266 = None
        mul_296 = torch.ops.aten.mul.Tensor(gt_67, _unsafe_view_209);  _unsafe_view_209 = None
        mul_297 = torch.ops.aten.mul.Tensor(mul_296, 1.1111111111111112);  mul_296 = None
        add_130 = torch.ops.aten.add.Tensor(add_127, mul_297);  mul_297 = None
        pow_47 = torch.ops.aten.pow.Tensor_Scalar(add_130, 2)
        mean_33 = torch.ops.aten.mean.dim(pow_47, [-1], True);  pow_47 = None
        add_131 = torch.ops.aten.add.Tensor(mean_33, 1e-06);  mean_33 = None
        sqrt_33 = torch.ops.aten.sqrt.default(add_131);  add_131 = None
        reciprocal_46 = torch.ops.aten.reciprocal.default(sqrt_33);  sqrt_33 = None
        mul_298 = torch.ops.aten.mul.Tensor(add_130, reciprocal_46)
        mul_299 = torch.ops.aten.mul.Tensor(primals_34, mul_298);  mul_298 = None
        permute_212 = torch.ops.aten.permute.default(primals_161, [1, 0]);  primals_161 = None
        view_212 = torch.ops.aten.view.default(mul_299, [256, 512]);  mul_299 = None
        mm_115 = torch.ops.aten.mm.default(view_212, permute_212);  view_212 = None
        _unsafe_view_210 = torch.ops.aten._unsafe_view.default(mm_115, [2, 128, 384]);  mm_115 = None
        view_213 = torch.ops.aten.view.default(_unsafe_view_210, [2, -1, 6, 64]);  _unsafe_view_210 = None
        permute_213 = torch.ops.aten.permute.default(view_213, [0, 2, 1, 3]);  view_213 = None
        permute_214 = torch.ops.aten.permute.default(primals_162, [1, 0]);  primals_162 = None
        mm_116 = torch.ops.aten.mm.default(view_109, permute_214)
        _unsafe_view_211 = torch.ops.aten._unsafe_view.default(mm_116, [2, 128, 384]);  mm_116 = None
        view_215 = torch.ops.aten.view.default(_unsafe_view_211, [2, -1, 6, 64]);  _unsafe_view_211 = None
        permute_215 = torch.ops.aten.permute.default(view_215, [0, 2, 1, 3]);  view_215 = None
        permute_216 = torch.ops.aten.permute.default(primals_163, [1, 0]);  primals_163 = None
        mm_117 = torch.ops.aten.mm.default(view_109, permute_216)
        _unsafe_view_212 = torch.ops.aten._unsafe_view.default(mm_117, [2, 128, 384]);  mm_117 = None
        view_217 = torch.ops.aten.view.default(_unsafe_view_212, [2, -1, 6, 64]);  _unsafe_view_212 = None
        permute_217 = torch.ops.aten.permute.default(view_217, [0, 2, 1, 3]);  view_217 = None
        permute_218 = torch.ops.aten.permute.default(permute_215, [0, 1, 3, 2])
        expand_76 = torch.ops.aten.expand.default(permute_213, [2, 6, 128, 64]);  permute_213 = None
        clone_76 = torch.ops.aten.clone.default(expand_76, memory_format = torch.contiguous_format);  expand_76 = None
        _unsafe_view_213 = torch.ops.aten._unsafe_view.default(clone_76, [12, 128, 64]);  clone_76 = None
        expand_77 = torch.ops.aten.expand.default(permute_218, [2, 6, 64, 128]);  permute_218 = None
        clone_77 = torch.ops.aten.clone.default(expand_77, memory_format = torch.contiguous_format);  expand_77 = None
        _unsafe_view_214 = torch.ops.aten._unsafe_view.default(clone_77, [12, 64, 128]);  clone_77 = None
        bmm_38 = torch.ops.aten.bmm.default(_unsafe_view_213, _unsafe_view_214)
        _unsafe_view_215 = torch.ops.aten._unsafe_view.default(bmm_38, [2, 6, 128, 128]);  bmm_38 = None
        add_132 = torch.ops.aten.add.Tensor(_unsafe_view_215, add_76);  _unsafe_view_215 = None
        amax_19 = torch.ops.aten.amax.default(add_132, [-1], True)
        sub_37 = torch.ops.aten.sub.Tensor(add_132, amax_19);  add_132 = amax_19 = None
        exp_32 = torch.ops.aten.exp.default(sub_37);  sub_37 = None
        sum_20 = torch.ops.aten.sum.dim_IntList(exp_32, [-1], True)
        div_23 = torch.ops.aten.div.Tensor(exp_32, sum_20);  exp_32 = sum_20 = None
        philox_rand_like_19 = torch.ops.prims.philox_rand_like.default(div_23, philox_seed_like, 3735552)
        gt_68 = torch.ops.aten.gt.Scalar(philox_rand_like_19, 0.1);  philox_rand_like_19 = None
        _to_copy_26 = torch.ops.aten._to_copy.default(gt_68, dtype = torch.float32);  gt_68 = None
        mul_300 = torch.ops.aten.mul.Tensor(_to_copy_26, div_23);  _to_copy_26 = None
        mul_301 = torch.ops.aten.mul.Tensor(mul_300, 1.1111111111111112);  mul_300 = None
        expand_78 = torch.ops.aten.expand.default(mul_301, [2, 6, 128, 128]);  mul_301 = None
        view_218 = torch.ops.aten.view.default(expand_78, [12, 128, 128]);  expand_78 = None
        expand_79 = torch.ops.aten.expand.default(permute_217, [2, 6, 128, 64])
        clone_78 = torch.ops.aten.clone.default(expand_79, memory_format = torch.contiguous_format);  expand_79 = None
        _unsafe_view_216 = torch.ops.aten._unsafe_view.default(clone_78, [12, 128, 64]);  clone_78 = None
        bmm_39 = torch.ops.aten.bmm.default(view_218, _unsafe_view_216)
        _unsafe_view_217 = torch.ops.aten._unsafe_view.default(bmm_39, [2, 6, 128, 64]);  bmm_39 = None
        permute_219 = torch.ops.aten.permute.default(_unsafe_view_217, [0, 2, 1, 3]);  _unsafe_view_217 = None
        clone_79 = torch.ops.aten.clone.default(permute_219, memory_format = torch.contiguous_format);  permute_219 = None
        view_219 = torch.ops.aten.view.default(clone_79, [2, -1, 384]);  clone_79 = None
        permute_220 = torch.ops.aten.permute.default(primals_164, [1, 0]);  primals_164 = None
        view_220 = torch.ops.aten.view.default(view_219, [256, 384]);  view_219 = None
        mm_118 = torch.ops.aten.mm.default(view_220, permute_220)
        _unsafe_view_218 = torch.ops.aten._unsafe_view.default(mm_118, [2, 128, 512]);  mm_118 = None
        rand_like_48 = torch.ops.aten.rand_like.default(_unsafe_view_218, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_273 = torch.ops.aten.alias.default(rand_like_48);  rand_like_48 = None
        gt_69 = torch.ops.aten.gt.Scalar(alias_273, 0.1);  alias_273 = None
        mul_302 = torch.ops.aten.mul.Tensor(gt_69, _unsafe_view_218);  _unsafe_view_218 = None
        mul_303 = torch.ops.aten.mul.Tensor(mul_302, 1.1111111111111112);  mul_302 = None
        add_133 = torch.ops.aten.add.Tensor(add_130, mul_303);  mul_303 = None
        pow_48 = torch.ops.aten.pow.Tensor_Scalar(add_133, 2)
        mean_34 = torch.ops.aten.mean.dim(pow_48, [-1], True);  pow_48 = None
        add_134 = torch.ops.aten.add.Tensor(mean_34, 1e-06);  mean_34 = None
        sqrt_34 = torch.ops.aten.sqrt.default(add_134);  add_134 = None
        reciprocal_47 = torch.ops.aten.reciprocal.default(sqrt_34);  sqrt_34 = None
        mul_304 = torch.ops.aten.mul.Tensor(add_133, reciprocal_47)
        mul_305 = torch.ops.aten.mul.Tensor(primals_35, mul_304);  mul_304 = None
        permute_221 = torch.ops.aten.permute.default(primals_165, [1, 0]);  primals_165 = None
        view_221 = torch.ops.aten.view.default(mul_305, [256, 512]);  mul_305 = None
        mm_119 = torch.ops.aten.mm.default(view_221, permute_221)
        _unsafe_view_219 = torch.ops.aten._unsafe_view.default(mm_119, [2, 128, 1024])
        mul_306 = torch.ops.aten.mul.Tensor(_unsafe_view_219, 0.5)
        pow_49 = torch.ops.aten.pow.Tensor_Scalar(_unsafe_view_219, 3.0)
        mul_307 = torch.ops.aten.mul.Tensor(pow_49, 0.044715);  pow_49 = None
        add_135 = torch.ops.aten.add.Tensor(_unsafe_view_219, mul_307);  _unsafe_view_219 = mul_307 = None
        mul_308 = torch.ops.aten.mul.Tensor(add_135, 0.7978845608028654);  add_135 = None
        mul_309 = torch.ops.aten.mul.Tensor(mul_308, -2.0);  mul_308 = None
        exp_33 = torch.ops.aten.exp.default(mul_309);  mul_309 = None
        add_136 = torch.ops.aten.add.Tensor(exp_33, 1.0);  exp_33 = None
        reciprocal_48 = torch.ops.aten.reciprocal.default(add_136);  add_136 = None
        mul_310 = torch.ops.aten.mul.Tensor(reciprocal_48, 2.0);  reciprocal_48 = None
        sub_38 = torch.ops.aten.sub.Tensor(mul_310, 1.0);  mul_310 = None
        add_137 = torch.ops.aten.add.Tensor(sub_38, 1.0)
        mul_311 = torch.ops.aten.mul.Tensor(mul_306, add_137);  mul_306 = add_137 = None
        permute_222 = torch.ops.aten.permute.default(primals_166, [1, 0]);  primals_166 = None
        mm_120 = torch.ops.aten.mm.default(view_221, permute_222);  view_221 = None
        _unsafe_view_220 = torch.ops.aten._unsafe_view.default(mm_120, [2, 128, 1024])
        mul_312 = torch.ops.aten.mul.Tensor(mul_311, _unsafe_view_220);  mul_311 = _unsafe_view_220 = None
        rand_like_49 = torch.ops.aten.rand_like.default(mul_312, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_280 = torch.ops.aten.alias.default(rand_like_49);  rand_like_49 = None
        gt_70 = torch.ops.aten.gt.Scalar(alias_280, 0.1);  alias_280 = None
        mul_313 = torch.ops.aten.mul.Tensor(gt_70, mul_312);  mul_312 = None
        mul_314 = torch.ops.aten.mul.Tensor(mul_313, 1.1111111111111112);  mul_313 = None
        permute_223 = torch.ops.aten.permute.default(primals_167, [1, 0]);  primals_167 = None
        view_223 = torch.ops.aten.view.default(mul_314, [256, 1024]);  mul_314 = None
        mm_121 = torch.ops.aten.mm.default(view_223, permute_223)
        _unsafe_view_221 = torch.ops.aten._unsafe_view.default(mm_121, [2, 128, 512]);  mm_121 = None
        rand_like_50 = torch.ops.aten.rand_like.default(_unsafe_view_221, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_281 = torch.ops.aten.alias.default(rand_like_50);  rand_like_50 = None
        gt_71 = torch.ops.aten.gt.Scalar(alias_281, 0.1);  alias_281 = None
        mul_315 = torch.ops.aten.mul.Tensor(gt_71, _unsafe_view_221);  _unsafe_view_221 = None
        mul_316 = torch.ops.aten.mul.Tensor(mul_315, 1.1111111111111112);  mul_315 = None
        add_138 = torch.ops.aten.add.Tensor(add_133, mul_316);  mul_316 = None
        pow_50 = torch.ops.aten.pow.Tensor_Scalar(add_138, 2)
        mean_35 = torch.ops.aten.mean.dim(pow_50, [-1], True);  pow_50 = None
        add_139 = torch.ops.aten.add.Tensor(mean_35, 1e-06);  mean_35 = None
        sqrt_35 = torch.ops.aten.sqrt.default(add_139);  add_139 = None
        reciprocal_49 = torch.ops.aten.reciprocal.default(sqrt_35);  sqrt_35 = None
        mul_317 = torch.ops.aten.mul.Tensor(add_138, reciprocal_49)
        mul_318 = torch.ops.aten.mul.Tensor(primals_36, mul_317);  mul_317 = None
        permute_224 = torch.ops.aten.permute.default(primals_168, [1, 0]);  primals_168 = None
        view_224 = torch.ops.aten.view.default(mul_318, [256, 512]);  mul_318 = None
        mm_122 = torch.ops.aten.mm.default(view_224, permute_224)
        _unsafe_view_222 = torch.ops.aten._unsafe_view.default(mm_122, [2, 128, 384]);  mm_122 = None
        view_225 = torch.ops.aten.view.default(_unsafe_view_222, [2, -1, 6, 64]);  _unsafe_view_222 = None
        permute_225 = torch.ops.aten.permute.default(view_225, [0, 2, 1, 3]);  view_225 = None
        permute_226 = torch.ops.aten.permute.default(primals_169, [1, 0]);  primals_169 = None
        mm_123 = torch.ops.aten.mm.default(view_224, permute_226)
        _unsafe_view_223 = torch.ops.aten._unsafe_view.default(mm_123, [2, 128, 384]);  mm_123 = None
        view_227 = torch.ops.aten.view.default(_unsafe_view_223, [2, -1, 6, 64]);  _unsafe_view_223 = None
        permute_227 = torch.ops.aten.permute.default(view_227, [0, 2, 1, 3]);  view_227 = None
        permute_228 = torch.ops.aten.permute.default(primals_170, [1, 0]);  primals_170 = None
        mm_124 = torch.ops.aten.mm.default(view_224, permute_228);  view_224 = None
        _unsafe_view_224 = torch.ops.aten._unsafe_view.default(mm_124, [2, 128, 384]);  mm_124 = None
        view_229 = torch.ops.aten.view.default(_unsafe_view_224, [2, -1, 6, 64]);  _unsafe_view_224 = None
        permute_229 = torch.ops.aten.permute.default(view_229, [0, 2, 1, 3]);  view_229 = None
        permute_230 = torch.ops.aten.permute.default(permute_227, [0, 1, 3, 2])
        expand_80 = torch.ops.aten.expand.default(permute_225, [2, 6, 128, 64]);  permute_225 = None
        clone_80 = torch.ops.aten.clone.default(expand_80, memory_format = torch.contiguous_format);  expand_80 = None
        _unsafe_view_225 = torch.ops.aten._unsafe_view.default(clone_80, [12, 128, 64]);  clone_80 = None
        expand_81 = torch.ops.aten.expand.default(permute_230, [2, 6, 64, 128]);  permute_230 = None
        clone_81 = torch.ops.aten.clone.default(expand_81, memory_format = torch.contiguous_format);  expand_81 = None
        _unsafe_view_226 = torch.ops.aten._unsafe_view.default(clone_81, [12, 64, 128]);  clone_81 = None
        bmm_40 = torch.ops.aten.bmm.default(_unsafe_view_225, _unsafe_view_226)
        _unsafe_view_227 = torch.ops.aten._unsafe_view.default(bmm_40, [2, 6, 128, 128]);  bmm_40 = None
        add_140 = torch.ops.aten.add.Tensor(_unsafe_view_227, add_72);  _unsafe_view_227 = None
        amax_20 = torch.ops.aten.amax.default(add_140, [-1], True)
        sub_39 = torch.ops.aten.sub.Tensor(add_140, amax_20);  add_140 = amax_20 = None
        exp_34 = torch.ops.aten.exp.default(sub_39);  sub_39 = None
        sum_21 = torch.ops.aten.sum.dim_IntList(exp_34, [-1], True)
        div_24 = torch.ops.aten.div.Tensor(exp_34, sum_21);  exp_34 = sum_21 = None
        philox_rand_like_20 = torch.ops.prims.philox_rand_like.default(div_24, philox_seed_like, 3932160)
        gt_72 = torch.ops.aten.gt.Scalar(philox_rand_like_20, 0.1);  philox_rand_like_20 = None
        _to_copy_27 = torch.ops.aten._to_copy.default(gt_72, dtype = torch.float32);  gt_72 = None
        mul_319 = torch.ops.aten.mul.Tensor(_to_copy_27, div_24);  _to_copy_27 = None
        mul_320 = torch.ops.aten.mul.Tensor(mul_319, 1.1111111111111112);  mul_319 = None
        expand_82 = torch.ops.aten.expand.default(mul_320, [2, 6, 128, 128]);  mul_320 = None
        view_230 = torch.ops.aten.view.default(expand_82, [12, 128, 128]);  expand_82 = None
        expand_83 = torch.ops.aten.expand.default(permute_229, [2, 6, 128, 64])
        clone_82 = torch.ops.aten.clone.default(expand_83, memory_format = torch.contiguous_format);  expand_83 = None
        _unsafe_view_228 = torch.ops.aten._unsafe_view.default(clone_82, [12, 128, 64]);  clone_82 = None
        bmm_41 = torch.ops.aten.bmm.default(view_230, _unsafe_view_228)
        _unsafe_view_229 = torch.ops.aten._unsafe_view.default(bmm_41, [2, 6, 128, 64]);  bmm_41 = None
        permute_231 = torch.ops.aten.permute.default(_unsafe_view_229, [0, 2, 1, 3]);  _unsafe_view_229 = None
        clone_83 = torch.ops.aten.clone.default(permute_231, memory_format = torch.contiguous_format);  permute_231 = None
        view_231 = torch.ops.aten.view.default(clone_83, [2, -1, 384]);  clone_83 = None
        permute_232 = torch.ops.aten.permute.default(primals_171, [1, 0]);  primals_171 = None
        view_232 = torch.ops.aten.view.default(view_231, [256, 384]);  view_231 = None
        mm_125 = torch.ops.aten.mm.default(view_232, permute_232)
        _unsafe_view_230 = torch.ops.aten._unsafe_view.default(mm_125, [2, 128, 512]);  mm_125 = None
        rand_like_51 = torch.ops.aten.rand_like.default(_unsafe_view_230, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_288 = torch.ops.aten.alias.default(rand_like_51);  rand_like_51 = None
        gt_73 = torch.ops.aten.gt.Scalar(alias_288, 0.1);  alias_288 = None
        mul_321 = torch.ops.aten.mul.Tensor(gt_73, _unsafe_view_230);  _unsafe_view_230 = None
        mul_322 = torch.ops.aten.mul.Tensor(mul_321, 1.1111111111111112);  mul_321 = None
        add_141 = torch.ops.aten.add.Tensor(add_138, mul_322);  mul_322 = None
        pow_51 = torch.ops.aten.pow.Tensor_Scalar(add_141, 2)
        mean_36 = torch.ops.aten.mean.dim(pow_51, [-1], True);  pow_51 = None
        add_142 = torch.ops.aten.add.Tensor(mean_36, 1e-06);  mean_36 = None
        sqrt_36 = torch.ops.aten.sqrt.default(add_142);  add_142 = None
        reciprocal_50 = torch.ops.aten.reciprocal.default(sqrt_36);  sqrt_36 = None
        mul_323 = torch.ops.aten.mul.Tensor(add_141, reciprocal_50)
        mul_324 = torch.ops.aten.mul.Tensor(primals_37, mul_323);  mul_323 = None
        permute_233 = torch.ops.aten.permute.default(primals_172, [1, 0]);  primals_172 = None
        view_233 = torch.ops.aten.view.default(mul_324, [256, 512]);  mul_324 = None
        mm_126 = torch.ops.aten.mm.default(view_233, permute_233);  view_233 = None
        _unsafe_view_231 = torch.ops.aten._unsafe_view.default(mm_126, [2, 128, 384]);  mm_126 = None
        view_234 = torch.ops.aten.view.default(_unsafe_view_231, [2, -1, 6, 64]);  _unsafe_view_231 = None
        permute_234 = torch.ops.aten.permute.default(view_234, [0, 2, 1, 3]);  view_234 = None
        permute_235 = torch.ops.aten.permute.default(primals_173, [1, 0]);  primals_173 = None
        mm_127 = torch.ops.aten.mm.default(view_109, permute_235)
        _unsafe_view_232 = torch.ops.aten._unsafe_view.default(mm_127, [2, 128, 384]);  mm_127 = None
        view_236 = torch.ops.aten.view.default(_unsafe_view_232, [2, -1, 6, 64]);  _unsafe_view_232 = None
        permute_236 = torch.ops.aten.permute.default(view_236, [0, 2, 1, 3]);  view_236 = None
        permute_237 = torch.ops.aten.permute.default(primals_174, [1, 0]);  primals_174 = None
        mm_128 = torch.ops.aten.mm.default(view_109, permute_237)
        _unsafe_view_233 = torch.ops.aten._unsafe_view.default(mm_128, [2, 128, 384]);  mm_128 = None
        view_238 = torch.ops.aten.view.default(_unsafe_view_233, [2, -1, 6, 64]);  _unsafe_view_233 = None
        permute_238 = torch.ops.aten.permute.default(view_238, [0, 2, 1, 3]);  view_238 = None
        permute_239 = torch.ops.aten.permute.default(permute_236, [0, 1, 3, 2])
        expand_84 = torch.ops.aten.expand.default(permute_234, [2, 6, 128, 64]);  permute_234 = None
        clone_84 = torch.ops.aten.clone.default(expand_84, memory_format = torch.contiguous_format);  expand_84 = None
        _unsafe_view_234 = torch.ops.aten._unsafe_view.default(clone_84, [12, 128, 64]);  clone_84 = None
        expand_85 = torch.ops.aten.expand.default(permute_239, [2, 6, 64, 128]);  permute_239 = None
        clone_85 = torch.ops.aten.clone.default(expand_85, memory_format = torch.contiguous_format);  expand_85 = None
        _unsafe_view_235 = torch.ops.aten._unsafe_view.default(clone_85, [12, 64, 128]);  clone_85 = None
        bmm_42 = torch.ops.aten.bmm.default(_unsafe_view_234, _unsafe_view_235)
        _unsafe_view_236 = torch.ops.aten._unsafe_view.default(bmm_42, [2, 6, 128, 128]);  bmm_42 = None
        add_143 = torch.ops.aten.add.Tensor(_unsafe_view_236, add_76);  _unsafe_view_236 = None
        amax_21 = torch.ops.aten.amax.default(add_143, [-1], True)
        sub_40 = torch.ops.aten.sub.Tensor(add_143, amax_21);  add_143 = amax_21 = None
        exp_35 = torch.ops.aten.exp.default(sub_40);  sub_40 = None
        sum_22 = torch.ops.aten.sum.dim_IntList(exp_35, [-1], True)
        div_25 = torch.ops.aten.div.Tensor(exp_35, sum_22);  exp_35 = sum_22 = None
        philox_rand_like_21 = torch.ops.prims.philox_rand_like.default(div_25, philox_seed_like, 4128768)
        gt_74 = torch.ops.aten.gt.Scalar(philox_rand_like_21, 0.1);  philox_rand_like_21 = None
        _to_copy_28 = torch.ops.aten._to_copy.default(gt_74, dtype = torch.float32);  gt_74 = None
        mul_325 = torch.ops.aten.mul.Tensor(_to_copy_28, div_25);  _to_copy_28 = None
        mul_326 = torch.ops.aten.mul.Tensor(mul_325, 1.1111111111111112);  mul_325 = None
        expand_86 = torch.ops.aten.expand.default(mul_326, [2, 6, 128, 128]);  mul_326 = None
        view_239 = torch.ops.aten.view.default(expand_86, [12, 128, 128]);  expand_86 = None
        expand_87 = torch.ops.aten.expand.default(permute_238, [2, 6, 128, 64])
        clone_86 = torch.ops.aten.clone.default(expand_87, memory_format = torch.contiguous_format);  expand_87 = None
        _unsafe_view_237 = torch.ops.aten._unsafe_view.default(clone_86, [12, 128, 64]);  clone_86 = None
        bmm_43 = torch.ops.aten.bmm.default(view_239, _unsafe_view_237)
        _unsafe_view_238 = torch.ops.aten._unsafe_view.default(bmm_43, [2, 6, 128, 64]);  bmm_43 = None
        permute_240 = torch.ops.aten.permute.default(_unsafe_view_238, [0, 2, 1, 3]);  _unsafe_view_238 = None
        clone_87 = torch.ops.aten.clone.default(permute_240, memory_format = torch.contiguous_format);  permute_240 = None
        view_240 = torch.ops.aten.view.default(clone_87, [2, -1, 384]);  clone_87 = None
        permute_241 = torch.ops.aten.permute.default(primals_175, [1, 0]);  primals_175 = None
        view_241 = torch.ops.aten.view.default(view_240, [256, 384]);  view_240 = None
        mm_129 = torch.ops.aten.mm.default(view_241, permute_241)
        _unsafe_view_239 = torch.ops.aten._unsafe_view.default(mm_129, [2, 128, 512]);  mm_129 = None
        rand_like_52 = torch.ops.aten.rand_like.default(_unsafe_view_239, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_295 = torch.ops.aten.alias.default(rand_like_52);  rand_like_52 = None
        gt_75 = torch.ops.aten.gt.Scalar(alias_295, 0.1);  alias_295 = None
        mul_327 = torch.ops.aten.mul.Tensor(gt_75, _unsafe_view_239);  _unsafe_view_239 = None
        mul_328 = torch.ops.aten.mul.Tensor(mul_327, 1.1111111111111112);  mul_327 = None
        add_144 = torch.ops.aten.add.Tensor(add_141, mul_328);  mul_328 = None
        pow_52 = torch.ops.aten.pow.Tensor_Scalar(add_144, 2)
        mean_37 = torch.ops.aten.mean.dim(pow_52, [-1], True);  pow_52 = None
        add_145 = torch.ops.aten.add.Tensor(mean_37, 1e-06);  mean_37 = None
        sqrt_37 = torch.ops.aten.sqrt.default(add_145);  add_145 = None
        reciprocal_51 = torch.ops.aten.reciprocal.default(sqrt_37);  sqrt_37 = None
        mul_329 = torch.ops.aten.mul.Tensor(add_144, reciprocal_51)
        mul_330 = torch.ops.aten.mul.Tensor(primals_38, mul_329);  mul_329 = None
        permute_242 = torch.ops.aten.permute.default(primals_176, [1, 0]);  primals_176 = None
        view_242 = torch.ops.aten.view.default(mul_330, [256, 512]);  mul_330 = None
        mm_130 = torch.ops.aten.mm.default(view_242, permute_242)
        _unsafe_view_240 = torch.ops.aten._unsafe_view.default(mm_130, [2, 128, 1024])
        mul_331 = torch.ops.aten.mul.Tensor(_unsafe_view_240, 0.5)
        pow_53 = torch.ops.aten.pow.Tensor_Scalar(_unsafe_view_240, 3.0)
        mul_332 = torch.ops.aten.mul.Tensor(pow_53, 0.044715);  pow_53 = None
        add_146 = torch.ops.aten.add.Tensor(_unsafe_view_240, mul_332);  _unsafe_view_240 = mul_332 = None
        mul_333 = torch.ops.aten.mul.Tensor(add_146, 0.7978845608028654);  add_146 = None
        mul_334 = torch.ops.aten.mul.Tensor(mul_333, -2.0);  mul_333 = None
        exp_36 = torch.ops.aten.exp.default(mul_334);  mul_334 = None
        add_147 = torch.ops.aten.add.Tensor(exp_36, 1.0);  exp_36 = None
        reciprocal_52 = torch.ops.aten.reciprocal.default(add_147);  add_147 = None
        mul_335 = torch.ops.aten.mul.Tensor(reciprocal_52, 2.0);  reciprocal_52 = None
        sub_41 = torch.ops.aten.sub.Tensor(mul_335, 1.0);  mul_335 = None
        add_148 = torch.ops.aten.add.Tensor(sub_41, 1.0)
        mul_336 = torch.ops.aten.mul.Tensor(mul_331, add_148);  mul_331 = add_148 = None
        permute_243 = torch.ops.aten.permute.default(primals_177, [1, 0]);  primals_177 = None
        mm_131 = torch.ops.aten.mm.default(view_242, permute_243);  view_242 = None
        _unsafe_view_241 = torch.ops.aten._unsafe_view.default(mm_131, [2, 128, 1024])
        mul_337 = torch.ops.aten.mul.Tensor(mul_336, _unsafe_view_241);  mul_336 = _unsafe_view_241 = None
        rand_like_53 = torch.ops.aten.rand_like.default(mul_337, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_302 = torch.ops.aten.alias.default(rand_like_53);  rand_like_53 = None
        gt_76 = torch.ops.aten.gt.Scalar(alias_302, 0.1);  alias_302 = None
        mul_338 = torch.ops.aten.mul.Tensor(gt_76, mul_337);  mul_337 = None
        mul_339 = torch.ops.aten.mul.Tensor(mul_338, 1.1111111111111112);  mul_338 = None
        permute_244 = torch.ops.aten.permute.default(primals_178, [1, 0]);  primals_178 = None
        view_244 = torch.ops.aten.view.default(mul_339, [256, 1024]);  mul_339 = None
        mm_132 = torch.ops.aten.mm.default(view_244, permute_244)
        _unsafe_view_242 = torch.ops.aten._unsafe_view.default(mm_132, [2, 128, 512]);  mm_132 = None
        rand_like_54 = torch.ops.aten.rand_like.default(_unsafe_view_242, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_303 = torch.ops.aten.alias.default(rand_like_54);  rand_like_54 = None
        gt_77 = torch.ops.aten.gt.Scalar(alias_303, 0.1);  alias_303 = None
        mul_340 = torch.ops.aten.mul.Tensor(gt_77, _unsafe_view_242);  _unsafe_view_242 = None
        mul_341 = torch.ops.aten.mul.Tensor(mul_340, 1.1111111111111112);  mul_340 = None
        add_149 = torch.ops.aten.add.Tensor(add_144, mul_341);  mul_341 = None
        pow_54 = torch.ops.aten.pow.Tensor_Scalar(add_149, 2)
        mean_38 = torch.ops.aten.mean.dim(pow_54, [-1], True);  pow_54 = None
        add_150 = torch.ops.aten.add.Tensor(mean_38, 1e-06);  mean_38 = None
        sqrt_38 = torch.ops.aten.sqrt.default(add_150);  add_150 = None
        reciprocal_53 = torch.ops.aten.reciprocal.default(sqrt_38);  sqrt_38 = None
        mul_342 = torch.ops.aten.mul.Tensor(add_149, reciprocal_53)
        mul_343 = torch.ops.aten.mul.Tensor(primals_39, mul_342);  mul_342 = None
        permute_245 = torch.ops.aten.permute.default(primals_179, [1, 0]);  primals_179 = None
        view_245 = torch.ops.aten.view.default(mul_343, [256, 512]);  mul_343 = None
        mm_133 = torch.ops.aten.mm.default(view_245, permute_245)
        _unsafe_view_243 = torch.ops.aten._unsafe_view.default(mm_133, [2, 128, 384]);  mm_133 = None
        view_246 = torch.ops.aten.view.default(_unsafe_view_243, [2, -1, 6, 64]);  _unsafe_view_243 = None
        permute_246 = torch.ops.aten.permute.default(view_246, [0, 2, 1, 3]);  view_246 = None
        permute_247 = torch.ops.aten.permute.default(primals_180, [1, 0]);  primals_180 = None
        mm_134 = torch.ops.aten.mm.default(view_245, permute_247)
        _unsafe_view_244 = torch.ops.aten._unsafe_view.default(mm_134, [2, 128, 384]);  mm_134 = None
        view_248 = torch.ops.aten.view.default(_unsafe_view_244, [2, -1, 6, 64]);  _unsafe_view_244 = None
        permute_248 = torch.ops.aten.permute.default(view_248, [0, 2, 1, 3]);  view_248 = None
        permute_249 = torch.ops.aten.permute.default(primals_181, [1, 0]);  primals_181 = None
        mm_135 = torch.ops.aten.mm.default(view_245, permute_249);  view_245 = None
        _unsafe_view_245 = torch.ops.aten._unsafe_view.default(mm_135, [2, 128, 384]);  mm_135 = None
        view_250 = torch.ops.aten.view.default(_unsafe_view_245, [2, -1, 6, 64]);  _unsafe_view_245 = None
        permute_250 = torch.ops.aten.permute.default(view_250, [0, 2, 1, 3]);  view_250 = None
        permute_251 = torch.ops.aten.permute.default(permute_248, [0, 1, 3, 2])
        expand_88 = torch.ops.aten.expand.default(permute_246, [2, 6, 128, 64]);  permute_246 = None
        clone_88 = torch.ops.aten.clone.default(expand_88, memory_format = torch.contiguous_format);  expand_88 = None
        _unsafe_view_246 = torch.ops.aten._unsafe_view.default(clone_88, [12, 128, 64]);  clone_88 = None
        expand_89 = torch.ops.aten.expand.default(permute_251, [2, 6, 64, 128]);  permute_251 = None
        clone_89 = torch.ops.aten.clone.default(expand_89, memory_format = torch.contiguous_format);  expand_89 = None
        _unsafe_view_247 = torch.ops.aten._unsafe_view.default(clone_89, [12, 64, 128]);  clone_89 = None
        bmm_44 = torch.ops.aten.bmm.default(_unsafe_view_246, _unsafe_view_247)
        _unsafe_view_248 = torch.ops.aten._unsafe_view.default(bmm_44, [2, 6, 128, 128]);  bmm_44 = None
        add_151 = torch.ops.aten.add.Tensor(_unsafe_view_248, add_72);  _unsafe_view_248 = add_72 = None
        amax_22 = torch.ops.aten.amax.default(add_151, [-1], True)
        sub_42 = torch.ops.aten.sub.Tensor(add_151, amax_22);  add_151 = amax_22 = None
        exp_37 = torch.ops.aten.exp.default(sub_42);  sub_42 = None
        sum_23 = torch.ops.aten.sum.dim_IntList(exp_37, [-1], True)
        div_26 = torch.ops.aten.div.Tensor(exp_37, sum_23);  exp_37 = sum_23 = None
        philox_rand_like_22 = torch.ops.prims.philox_rand_like.default(div_26, philox_seed_like, 4325376)
        gt_78 = torch.ops.aten.gt.Scalar(philox_rand_like_22, 0.1);  philox_rand_like_22 = None
        _to_copy_29 = torch.ops.aten._to_copy.default(gt_78, dtype = torch.float32);  gt_78 = None
        mul_344 = torch.ops.aten.mul.Tensor(_to_copy_29, div_26);  _to_copy_29 = None
        mul_345 = torch.ops.aten.mul.Tensor(mul_344, 1.1111111111111112);  mul_344 = None
        expand_90 = torch.ops.aten.expand.default(mul_345, [2, 6, 128, 128]);  mul_345 = None
        view_251 = torch.ops.aten.view.default(expand_90, [12, 128, 128]);  expand_90 = None
        expand_91 = torch.ops.aten.expand.default(permute_250, [2, 6, 128, 64])
        clone_90 = torch.ops.aten.clone.default(expand_91, memory_format = torch.contiguous_format);  expand_91 = None
        _unsafe_view_249 = torch.ops.aten._unsafe_view.default(clone_90, [12, 128, 64]);  clone_90 = None
        bmm_45 = torch.ops.aten.bmm.default(view_251, _unsafe_view_249)
        _unsafe_view_250 = torch.ops.aten._unsafe_view.default(bmm_45, [2, 6, 128, 64]);  bmm_45 = None
        permute_252 = torch.ops.aten.permute.default(_unsafe_view_250, [0, 2, 1, 3]);  _unsafe_view_250 = None
        clone_91 = torch.ops.aten.clone.default(permute_252, memory_format = torch.contiguous_format);  permute_252 = None
        view_252 = torch.ops.aten.view.default(clone_91, [2, -1, 384]);  clone_91 = None
        permute_253 = torch.ops.aten.permute.default(primals_182, [1, 0]);  primals_182 = None
        view_253 = torch.ops.aten.view.default(view_252, [256, 384]);  view_252 = None
        mm_136 = torch.ops.aten.mm.default(view_253, permute_253)
        _unsafe_view_251 = torch.ops.aten._unsafe_view.default(mm_136, [2, 128, 512]);  mm_136 = None
        rand_like_55 = torch.ops.aten.rand_like.default(_unsafe_view_251, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_310 = torch.ops.aten.alias.default(rand_like_55);  rand_like_55 = None
        gt_79 = torch.ops.aten.gt.Scalar(alias_310, 0.1);  alias_310 = None
        mul_346 = torch.ops.aten.mul.Tensor(gt_79, _unsafe_view_251);  _unsafe_view_251 = None
        mul_347 = torch.ops.aten.mul.Tensor(mul_346, 1.1111111111111112);  mul_346 = None
        add_152 = torch.ops.aten.add.Tensor(add_149, mul_347);  mul_347 = None
        pow_55 = torch.ops.aten.pow.Tensor_Scalar(add_152, 2)
        mean_39 = torch.ops.aten.mean.dim(pow_55, [-1], True);  pow_55 = None
        add_153 = torch.ops.aten.add.Tensor(mean_39, 1e-06);  mean_39 = None
        sqrt_39 = torch.ops.aten.sqrt.default(add_153);  add_153 = None
        reciprocal_54 = torch.ops.aten.reciprocal.default(sqrt_39);  sqrt_39 = None
        mul_348 = torch.ops.aten.mul.Tensor(add_152, reciprocal_54)
        mul_349 = torch.ops.aten.mul.Tensor(primals_40, mul_348);  mul_348 = None
        permute_254 = torch.ops.aten.permute.default(primals_183, [1, 0]);  primals_183 = None
        view_254 = torch.ops.aten.view.default(mul_349, [256, 512]);  mul_349 = None
        mm_137 = torch.ops.aten.mm.default(view_254, permute_254);  view_254 = None
        _unsafe_view_252 = torch.ops.aten._unsafe_view.default(mm_137, [2, 128, 384]);  mm_137 = None
        view_255 = torch.ops.aten.view.default(_unsafe_view_252, [2, -1, 6, 64]);  _unsafe_view_252 = None
        permute_255 = torch.ops.aten.permute.default(view_255, [0, 2, 1, 3]);  view_255 = None
        permute_256 = torch.ops.aten.permute.default(primals_184, [1, 0]);  primals_184 = None
        mm_138 = torch.ops.aten.mm.default(view_109, permute_256)
        _unsafe_view_253 = torch.ops.aten._unsafe_view.default(mm_138, [2, 128, 384]);  mm_138 = None
        view_257 = torch.ops.aten.view.default(_unsafe_view_253, [2, -1, 6, 64]);  _unsafe_view_253 = None
        permute_257 = torch.ops.aten.permute.default(view_257, [0, 2, 1, 3]);  view_257 = None
        permute_258 = torch.ops.aten.permute.default(primals_185, [1, 0]);  primals_185 = None
        mm_139 = torch.ops.aten.mm.default(view_109, permute_258);  view_109 = None
        _unsafe_view_254 = torch.ops.aten._unsafe_view.default(mm_139, [2, 128, 384]);  mm_139 = None
        view_259 = torch.ops.aten.view.default(_unsafe_view_254, [2, -1, 6, 64]);  _unsafe_view_254 = None
        permute_259 = torch.ops.aten.permute.default(view_259, [0, 2, 1, 3]);  view_259 = None
        permute_260 = torch.ops.aten.permute.default(permute_257, [0, 1, 3, 2])
        expand_92 = torch.ops.aten.expand.default(permute_255, [2, 6, 128, 64]);  permute_255 = None
        clone_92 = torch.ops.aten.clone.default(expand_92, memory_format = torch.contiguous_format);  expand_92 = None
        _unsafe_view_255 = torch.ops.aten._unsafe_view.default(clone_92, [12, 128, 64]);  clone_92 = None
        expand_93 = torch.ops.aten.expand.default(permute_260, [2, 6, 64, 128]);  permute_260 = None
        clone_93 = torch.ops.aten.clone.default(expand_93, memory_format = torch.contiguous_format);  expand_93 = None
        _unsafe_view_256 = torch.ops.aten._unsafe_view.default(clone_93, [12, 64, 128]);  clone_93 = None
        bmm_46 = torch.ops.aten.bmm.default(_unsafe_view_255, _unsafe_view_256)
        _unsafe_view_257 = torch.ops.aten._unsafe_view.default(bmm_46, [2, 6, 128, 128]);  bmm_46 = None
        add_154 = torch.ops.aten.add.Tensor(_unsafe_view_257, add_76);  _unsafe_view_257 = add_76 = None
        amax_23 = torch.ops.aten.amax.default(add_154, [-1], True)
        sub_43 = torch.ops.aten.sub.Tensor(add_154, amax_23);  add_154 = amax_23 = None
        exp_38 = torch.ops.aten.exp.default(sub_43);  sub_43 = None
        sum_24 = torch.ops.aten.sum.dim_IntList(exp_38, [-1], True)
        div_27 = torch.ops.aten.div.Tensor(exp_38, sum_24);  exp_38 = sum_24 = None
        philox_rand_like_23 = torch.ops.prims.philox_rand_like.default(div_27, philox_seed_like, 4521984)
        gt_80 = torch.ops.aten.gt.Scalar(philox_rand_like_23, 0.1);  philox_rand_like_23 = None
        _to_copy_30 = torch.ops.aten._to_copy.default(gt_80, dtype = torch.float32);  gt_80 = None
        mul_350 = torch.ops.aten.mul.Tensor(_to_copy_30, div_27);  _to_copy_30 = None
        mul_351 = torch.ops.aten.mul.Tensor(mul_350, 1.1111111111111112);  mul_350 = None
        expand_94 = torch.ops.aten.expand.default(mul_351, [2, 6, 128, 128]);  mul_351 = None
        view_260 = torch.ops.aten.view.default(expand_94, [12, 128, 128]);  expand_94 = None
        expand_95 = torch.ops.aten.expand.default(permute_259, [2, 6, 128, 64])
        clone_94 = torch.ops.aten.clone.default(expand_95, memory_format = torch.contiguous_format);  expand_95 = None
        _unsafe_view_258 = torch.ops.aten._unsafe_view.default(clone_94, [12, 128, 64]);  clone_94 = None
        bmm_47 = torch.ops.aten.bmm.default(view_260, _unsafe_view_258)
        _unsafe_view_259 = torch.ops.aten._unsafe_view.default(bmm_47, [2, 6, 128, 64]);  bmm_47 = None
        permute_261 = torch.ops.aten.permute.default(_unsafe_view_259, [0, 2, 1, 3]);  _unsafe_view_259 = None
        clone_95 = torch.ops.aten.clone.default(permute_261, memory_format = torch.contiguous_format);  permute_261 = None
        view_261 = torch.ops.aten.view.default(clone_95, [2, -1, 384]);  clone_95 = None
        permute_262 = torch.ops.aten.permute.default(primals_186, [1, 0]);  primals_186 = None
        view_262 = torch.ops.aten.view.default(view_261, [256, 384]);  view_261 = None
        mm_140 = torch.ops.aten.mm.default(view_262, permute_262)
        _unsafe_view_260 = torch.ops.aten._unsafe_view.default(mm_140, [2, 128, 512]);  mm_140 = None
        rand_like_56 = torch.ops.aten.rand_like.default(_unsafe_view_260, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_317 = torch.ops.aten.alias.default(rand_like_56);  rand_like_56 = None
        gt_81 = torch.ops.aten.gt.Scalar(alias_317, 0.1);  alias_317 = None
        mul_352 = torch.ops.aten.mul.Tensor(gt_81, _unsafe_view_260);  _unsafe_view_260 = None
        mul_353 = torch.ops.aten.mul.Tensor(mul_352, 1.1111111111111112);  mul_352 = None
        add_155 = torch.ops.aten.add.Tensor(add_152, mul_353);  mul_353 = None
        pow_56 = torch.ops.aten.pow.Tensor_Scalar(add_155, 2)
        mean_40 = torch.ops.aten.mean.dim(pow_56, [-1], True);  pow_56 = None
        add_156 = torch.ops.aten.add.Tensor(mean_40, 1e-06);  mean_40 = None
        sqrt_40 = torch.ops.aten.sqrt.default(add_156);  add_156 = None
        reciprocal_55 = torch.ops.aten.reciprocal.default(sqrt_40);  sqrt_40 = None
        mul_354 = torch.ops.aten.mul.Tensor(add_155, reciprocal_55)
        mul_355 = torch.ops.aten.mul.Tensor(primals_41, mul_354);  mul_354 = None
        permute_263 = torch.ops.aten.permute.default(primals_187, [1, 0]);  primals_187 = None
        view_263 = torch.ops.aten.view.default(mul_355, [256, 512]);  mul_355 = None
        mm_141 = torch.ops.aten.mm.default(view_263, permute_263)
        _unsafe_view_261 = torch.ops.aten._unsafe_view.default(mm_141, [2, 128, 1024])
        mul_356 = torch.ops.aten.mul.Tensor(_unsafe_view_261, 0.5)
        pow_57 = torch.ops.aten.pow.Tensor_Scalar(_unsafe_view_261, 3.0)
        mul_357 = torch.ops.aten.mul.Tensor(pow_57, 0.044715);  pow_57 = None
        add_157 = torch.ops.aten.add.Tensor(_unsafe_view_261, mul_357);  _unsafe_view_261 = mul_357 = None
        mul_358 = torch.ops.aten.mul.Tensor(add_157, 0.7978845608028654);  add_157 = None
        mul_359 = torch.ops.aten.mul.Tensor(mul_358, -2.0);  mul_358 = None
        exp_39 = torch.ops.aten.exp.default(mul_359);  mul_359 = None
        add_158 = torch.ops.aten.add.Tensor(exp_39, 1.0);  exp_39 = None
        reciprocal_56 = torch.ops.aten.reciprocal.default(add_158);  add_158 = None
        mul_360 = torch.ops.aten.mul.Tensor(reciprocal_56, 2.0);  reciprocal_56 = None
        sub_44 = torch.ops.aten.sub.Tensor(mul_360, 1.0);  mul_360 = None
        add_159 = torch.ops.aten.add.Tensor(sub_44, 1.0)
        mul_361 = torch.ops.aten.mul.Tensor(mul_356, add_159);  mul_356 = add_159 = None
        permute_264 = torch.ops.aten.permute.default(primals_188, [1, 0]);  primals_188 = None
        mm_142 = torch.ops.aten.mm.default(view_263, permute_264);  view_263 = None
        _unsafe_view_262 = torch.ops.aten._unsafe_view.default(mm_142, [2, 128, 1024])
        mul_362 = torch.ops.aten.mul.Tensor(mul_361, _unsafe_view_262);  mul_361 = _unsafe_view_262 = None
        rand_like_57 = torch.ops.aten.rand_like.default(mul_362, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_324 = torch.ops.aten.alias.default(rand_like_57);  rand_like_57 = None
        gt_82 = torch.ops.aten.gt.Scalar(alias_324, 0.1);  alias_324 = None
        mul_363 = torch.ops.aten.mul.Tensor(gt_82, mul_362);  mul_362 = None
        mul_364 = torch.ops.aten.mul.Tensor(mul_363, 1.1111111111111112);  mul_363 = None
        permute_265 = torch.ops.aten.permute.default(primals_189, [1, 0]);  primals_189 = None
        view_265 = torch.ops.aten.view.default(mul_364, [256, 1024]);  mul_364 = None
        mm_143 = torch.ops.aten.mm.default(view_265, permute_265)
        _unsafe_view_263 = torch.ops.aten._unsafe_view.default(mm_143, [2, 128, 512]);  mm_143 = None
        rand_like_58 = torch.ops.aten.rand_like.default(_unsafe_view_263, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_325 = torch.ops.aten.alias.default(rand_like_58);  rand_like_58 = None
        gt_83 = torch.ops.aten.gt.Scalar(alias_325, 0.1);  alias_325 = None
        mul_365 = torch.ops.aten.mul.Tensor(gt_83, _unsafe_view_263);  _unsafe_view_263 = None
        mul_366 = torch.ops.aten.mul.Tensor(mul_365, 1.1111111111111112);  mul_365 = None
        add_160 = torch.ops.aten.add.Tensor(add_155, mul_366);  mul_366 = None
        pow_58 = torch.ops.aten.pow.Tensor_Scalar(add_160, 2)
        mean_41 = torch.ops.aten.mean.dim(pow_58, [-1], True);  pow_58 = None
        add_161 = torch.ops.aten.add.Tensor(mean_41, 1e-06);  mean_41 = None
        sqrt_41 = torch.ops.aten.sqrt.default(add_161);  add_161 = None
        reciprocal_57 = torch.ops.aten.reciprocal.default(sqrt_41);  sqrt_41 = None
        mul_367 = torch.ops.aten.mul.Tensor(add_160, reciprocal_57)
        mul_368 = torch.ops.aten.mul.Tensor(primals_42, mul_367);  mul_367 = None
        rand_like_59 = torch.ops.aten.rand_like.default(mul_368, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_329 = torch.ops.aten.alias.default(rand_like_59);  rand_like_59 = None
        gt_84 = torch.ops.aten.gt.Scalar(alias_329, 0.1);  alias_329 = None
        mul_369 = torch.ops.aten.mul.Tensor(gt_84, mul_368);  mul_368 = None
        mul_370 = torch.ops.aten.mul.Tensor(mul_369, 1.1111111111111112);  mul_369 = None
        permute_266 = torch.ops.aten.permute.default(primals_190, [1, 0]);  primals_190 = None
        view_266 = torch.ops.aten.view.default(mul_370, [256, 512]);  mul_370 = None
        mm_144 = torch.ops.aten.mm.default(view_266, permute_266)
        _unsafe_view_264 = torch.ops.aten._unsafe_view.default(mm_144, [2, 128, 250112]);  mm_144 = None
        view_267 = torch.ops.aten.view.default(_unsafe_view_264, [-1, 250112])
        view_268 = torch.ops.aten.view.default(primals_193, [-1]);  primals_193 = None
        amax_24 = torch.ops.aten.amax.default(view_267, [1], True)
        sub_45 = torch.ops.aten.sub.Tensor(view_267, amax_24);  view_267 = amax_24 = None
        exp_40 = torch.ops.aten.exp.default(sub_45)
        sum_25 = torch.ops.aten.sum.dim_IntList(exp_40, [1], True);  exp_40 = None
        log_2 = torch.ops.aten.log.default(sum_25);  sum_25 = None
        sub_46 = torch.ops.aten.sub.Tensor(sub_45, log_2);  sub_45 = log_2 = None
        unsqueeze_17 = torch.ops.aten.unsqueeze.default(view_268, 1);  view_268 = None
        gather = torch.ops.aten.gather.default(sub_46, 1, unsqueeze_17)
        squeeze = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg_1 = torch.ops.aten.neg.default(squeeze);  squeeze = None
        mean_42 = torch.ops.aten.mean.default(neg_1);  neg_1 = None
        permute_269 = torch.ops.aten.permute.default(permute_266, [1, 0]);  permute_266 = None
        permute_273 = torch.ops.aten.permute.default(permute_265, [1, 0]);  permute_265 = None
        permute_277 = torch.ops.aten.permute.default(permute_264, [1, 0]);  permute_264 = None
        permute_281 = torch.ops.aten.permute.default(permute_263, [1, 0]);  permute_263 = None
        permute_285 = torch.ops.aten.permute.default(permute_262, [1, 0]);  permute_262 = None
        permute_288 = torch.ops.aten.permute.default(view_260, [0, 2, 1]);  view_260 = None
        permute_289 = torch.ops.aten.permute.default(_unsafe_view_258, [0, 2, 1]);  _unsafe_view_258 = None
        permute_290 = torch.ops.aten.permute.default(_unsafe_view_255, [0, 2, 1]);  _unsafe_view_255 = None
        permute_291 = torch.ops.aten.permute.default(_unsafe_view_256, [0, 2, 1]);  _unsafe_view_256 = None
        permute_296 = torch.ops.aten.permute.default(permute_258, [1, 0]);  permute_258 = None
        permute_301 = torch.ops.aten.permute.default(permute_256, [1, 0]);  permute_256 = None
        permute_306 = torch.ops.aten.permute.default(permute_254, [1, 0]);  permute_254 = None
        permute_310 = torch.ops.aten.permute.default(permute_253, [1, 0]);  permute_253 = None
        permute_313 = torch.ops.aten.permute.default(view_251, [0, 2, 1]);  view_251 = None
        permute_314 = torch.ops.aten.permute.default(_unsafe_view_249, [0, 2, 1]);  _unsafe_view_249 = None
        permute_315 = torch.ops.aten.permute.default(_unsafe_view_246, [0, 2, 1]);  _unsafe_view_246 = None
        permute_316 = torch.ops.aten.permute.default(_unsafe_view_247, [0, 2, 1]);  _unsafe_view_247 = None
        permute_321 = torch.ops.aten.permute.default(permute_249, [1, 0]);  permute_249 = None
        permute_326 = torch.ops.aten.permute.default(permute_247, [1, 0]);  permute_247 = None
        permute_331 = torch.ops.aten.permute.default(permute_245, [1, 0]);  permute_245 = None
        permute_335 = torch.ops.aten.permute.default(permute_244, [1, 0]);  permute_244 = None
        permute_339 = torch.ops.aten.permute.default(permute_243, [1, 0]);  permute_243 = None
        permute_343 = torch.ops.aten.permute.default(permute_242, [1, 0]);  permute_242 = None
        permute_347 = torch.ops.aten.permute.default(permute_241, [1, 0]);  permute_241 = None
        permute_350 = torch.ops.aten.permute.default(view_239, [0, 2, 1]);  view_239 = None
        permute_351 = torch.ops.aten.permute.default(_unsafe_view_237, [0, 2, 1]);  _unsafe_view_237 = None
        permute_352 = torch.ops.aten.permute.default(_unsafe_view_234, [0, 2, 1]);  _unsafe_view_234 = None
        permute_353 = torch.ops.aten.permute.default(_unsafe_view_235, [0, 2, 1]);  _unsafe_view_235 = None
        permute_358 = torch.ops.aten.permute.default(permute_237, [1, 0]);  permute_237 = None
        permute_363 = torch.ops.aten.permute.default(permute_235, [1, 0]);  permute_235 = None
        permute_368 = torch.ops.aten.permute.default(permute_233, [1, 0]);  permute_233 = None
        permute_372 = torch.ops.aten.permute.default(permute_232, [1, 0]);  permute_232 = None
        permute_375 = torch.ops.aten.permute.default(view_230, [0, 2, 1]);  view_230 = None
        permute_376 = torch.ops.aten.permute.default(_unsafe_view_228, [0, 2, 1]);  _unsafe_view_228 = None
        permute_377 = torch.ops.aten.permute.default(_unsafe_view_225, [0, 2, 1]);  _unsafe_view_225 = None
        permute_378 = torch.ops.aten.permute.default(_unsafe_view_226, [0, 2, 1]);  _unsafe_view_226 = None
        permute_383 = torch.ops.aten.permute.default(permute_228, [1, 0]);  permute_228 = None
        permute_388 = torch.ops.aten.permute.default(permute_226, [1, 0]);  permute_226 = None
        permute_393 = torch.ops.aten.permute.default(permute_224, [1, 0]);  permute_224 = None
        permute_397 = torch.ops.aten.permute.default(permute_223, [1, 0]);  permute_223 = None
        permute_401 = torch.ops.aten.permute.default(permute_222, [1, 0]);  permute_222 = None
        permute_405 = torch.ops.aten.permute.default(permute_221, [1, 0]);  permute_221 = None
        permute_409 = torch.ops.aten.permute.default(permute_220, [1, 0]);  permute_220 = None
        permute_412 = torch.ops.aten.permute.default(view_218, [0, 2, 1]);  view_218 = None
        permute_413 = torch.ops.aten.permute.default(_unsafe_view_216, [0, 2, 1]);  _unsafe_view_216 = None
        permute_414 = torch.ops.aten.permute.default(_unsafe_view_213, [0, 2, 1]);  _unsafe_view_213 = None
        permute_415 = torch.ops.aten.permute.default(_unsafe_view_214, [0, 2, 1]);  _unsafe_view_214 = None
        permute_420 = torch.ops.aten.permute.default(permute_216, [1, 0]);  permute_216 = None
        permute_425 = torch.ops.aten.permute.default(permute_214, [1, 0]);  permute_214 = None
        permute_430 = torch.ops.aten.permute.default(permute_212, [1, 0]);  permute_212 = None
        permute_434 = torch.ops.aten.permute.default(permute_211, [1, 0]);  permute_211 = None
        permute_437 = torch.ops.aten.permute.default(view_209, [0, 2, 1]);  view_209 = None
        permute_438 = torch.ops.aten.permute.default(_unsafe_view_207, [0, 2, 1]);  _unsafe_view_207 = None
        permute_439 = torch.ops.aten.permute.default(_unsafe_view_204, [0, 2, 1]);  _unsafe_view_204 = None
        permute_440 = torch.ops.aten.permute.default(_unsafe_view_205, [0, 2, 1]);  _unsafe_view_205 = None
        permute_445 = torch.ops.aten.permute.default(permute_207, [1, 0]);  permute_207 = None
        permute_450 = torch.ops.aten.permute.default(permute_205, [1, 0]);  permute_205 = None
        permute_455 = torch.ops.aten.permute.default(permute_203, [1, 0]);  permute_203 = None
        permute_459 = torch.ops.aten.permute.default(permute_202, [1, 0]);  permute_202 = None
        permute_463 = torch.ops.aten.permute.default(permute_201, [1, 0]);  permute_201 = None
        permute_467 = torch.ops.aten.permute.default(permute_200, [1, 0]);  permute_200 = None
        permute_471 = torch.ops.aten.permute.default(permute_199, [1, 0]);  permute_199 = None
        permute_474 = torch.ops.aten.permute.default(view_197, [0, 2, 1]);  view_197 = None
        permute_475 = torch.ops.aten.permute.default(_unsafe_view_195, [0, 2, 1]);  _unsafe_view_195 = None
        permute_476 = torch.ops.aten.permute.default(_unsafe_view_192, [0, 2, 1]);  _unsafe_view_192 = None
        permute_477 = torch.ops.aten.permute.default(_unsafe_view_193, [0, 2, 1]);  _unsafe_view_193 = None
        permute_482 = torch.ops.aten.permute.default(permute_195, [1, 0]);  permute_195 = None
        permute_487 = torch.ops.aten.permute.default(permute_193, [1, 0]);  permute_193 = None
        permute_492 = torch.ops.aten.permute.default(permute_191, [1, 0]);  permute_191 = None
        permute_496 = torch.ops.aten.permute.default(permute_190, [1, 0]);  permute_190 = None
        permute_499 = torch.ops.aten.permute.default(view_188, [0, 2, 1]);  view_188 = None
        permute_500 = torch.ops.aten.permute.default(_unsafe_view_186, [0, 2, 1]);  _unsafe_view_186 = None
        permute_501 = torch.ops.aten.permute.default(_unsafe_view_183, [0, 2, 1]);  _unsafe_view_183 = None
        permute_502 = torch.ops.aten.permute.default(_unsafe_view_184, [0, 2, 1]);  _unsafe_view_184 = None
        permute_507 = torch.ops.aten.permute.default(permute_186, [1, 0]);  permute_186 = None
        permute_512 = torch.ops.aten.permute.default(permute_184, [1, 0]);  permute_184 = None
        permute_517 = torch.ops.aten.permute.default(permute_182, [1, 0]);  permute_182 = None
        permute_521 = torch.ops.aten.permute.default(permute_181, [1, 0]);  permute_181 = None
        permute_525 = torch.ops.aten.permute.default(permute_180, [1, 0]);  permute_180 = None
        permute_529 = torch.ops.aten.permute.default(permute_179, [1, 0]);  permute_179 = None
        permute_533 = torch.ops.aten.permute.default(permute_178, [1, 0]);  permute_178 = None
        permute_536 = torch.ops.aten.permute.default(view_176, [0, 2, 1]);  view_176 = None
        permute_537 = torch.ops.aten.permute.default(_unsafe_view_174, [0, 2, 1]);  _unsafe_view_174 = None
        permute_538 = torch.ops.aten.permute.default(_unsafe_view_171, [0, 2, 1]);  _unsafe_view_171 = None
        permute_539 = torch.ops.aten.permute.default(_unsafe_view_172, [0, 2, 1]);  _unsafe_view_172 = None
        permute_544 = torch.ops.aten.permute.default(permute_174, [1, 0]);  permute_174 = None
        permute_549 = torch.ops.aten.permute.default(permute_172, [1, 0]);  permute_172 = None
        permute_554 = torch.ops.aten.permute.default(permute_170, [1, 0]);  permute_170 = None
        permute_558 = torch.ops.aten.permute.default(permute_169, [1, 0]);  permute_169 = None
        permute_561 = torch.ops.aten.permute.default(view_167, [0, 2, 1]);  view_167 = None
        permute_562 = torch.ops.aten.permute.default(_unsafe_view_165, [0, 2, 1]);  _unsafe_view_165 = None
        permute_563 = torch.ops.aten.permute.default(_unsafe_view_162, [0, 2, 1]);  _unsafe_view_162 = None
        permute_564 = torch.ops.aten.permute.default(_unsafe_view_163, [0, 2, 1]);  _unsafe_view_163 = None
        permute_569 = torch.ops.aten.permute.default(permute_165, [1, 0]);  permute_165 = None
        permute_574 = torch.ops.aten.permute.default(permute_163, [1, 0]);  permute_163 = None
        permute_579 = torch.ops.aten.permute.default(permute_161, [1, 0]);  permute_161 = None
        permute_583 = torch.ops.aten.permute.default(permute_160, [1, 0]);  permute_160 = None
        permute_587 = torch.ops.aten.permute.default(permute_159, [1, 0]);  permute_159 = None
        permute_591 = torch.ops.aten.permute.default(permute_158, [1, 0]);  permute_158 = None
        permute_595 = torch.ops.aten.permute.default(permute_157, [1, 0]);  permute_157 = None
        permute_598 = torch.ops.aten.permute.default(view_155, [0, 2, 1]);  view_155 = None
        permute_599 = torch.ops.aten.permute.default(_unsafe_view_153, [0, 2, 1]);  _unsafe_view_153 = None
        permute_600 = torch.ops.aten.permute.default(_unsafe_view_150, [0, 2, 1]);  _unsafe_view_150 = None
        permute_601 = torch.ops.aten.permute.default(_unsafe_view_151, [0, 2, 1]);  _unsafe_view_151 = None
        permute_606 = torch.ops.aten.permute.default(permute_153, [1, 0]);  permute_153 = None
        permute_611 = torch.ops.aten.permute.default(permute_151, [1, 0]);  permute_151 = None
        permute_616 = torch.ops.aten.permute.default(permute_149, [1, 0]);  permute_149 = None
        permute_620 = torch.ops.aten.permute.default(permute_148, [1, 0]);  permute_148 = None
        permute_623 = torch.ops.aten.permute.default(view_146, [0, 2, 1]);  view_146 = None
        permute_624 = torch.ops.aten.permute.default(_unsafe_view_144, [0, 2, 1]);  _unsafe_view_144 = None
        permute_625 = torch.ops.aten.permute.default(_unsafe_view_141, [0, 2, 1]);  _unsafe_view_141 = None
        permute_626 = torch.ops.aten.permute.default(_unsafe_view_142, [0, 2, 1]);  _unsafe_view_142 = None
        permute_631 = torch.ops.aten.permute.default(permute_144, [1, 0]);  permute_144 = None
        permute_636 = torch.ops.aten.permute.default(permute_142, [1, 0]);  permute_142 = None
        permute_641 = torch.ops.aten.permute.default(permute_140, [1, 0]);  permute_140 = None
        permute_645 = torch.ops.aten.permute.default(permute_139, [1, 0]);  permute_139 = None
        permute_649 = torch.ops.aten.permute.default(permute_138, [1, 0]);  permute_138 = None
        permute_653 = torch.ops.aten.permute.default(permute_137, [1, 0]);  permute_137 = None
        permute_657 = torch.ops.aten.permute.default(permute_136, [1, 0]);  permute_136 = None
        permute_660 = torch.ops.aten.permute.default(view_134, [0, 2, 1]);  view_134 = None
        permute_661 = torch.ops.aten.permute.default(_unsafe_view_132, [0, 2, 1]);  _unsafe_view_132 = None
        permute_662 = torch.ops.aten.permute.default(_unsafe_view_129, [0, 2, 1]);  _unsafe_view_129 = None
        permute_663 = torch.ops.aten.permute.default(_unsafe_view_130, [0, 2, 1]);  _unsafe_view_130 = None
        permute_668 = torch.ops.aten.permute.default(permute_132, [1, 0]);  permute_132 = None
        permute_673 = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None
        permute_678 = torch.ops.aten.permute.default(permute_128, [1, 0]);  permute_128 = None
        permute_682 = torch.ops.aten.permute.default(permute_127, [1, 0]);  permute_127 = None
        permute_685 = torch.ops.aten.permute.default(view_125, [0, 2, 1]);  view_125 = None
        permute_686 = torch.ops.aten.permute.default(_unsafe_view_123, [0, 2, 1]);  _unsafe_view_123 = None
        permute_687 = torch.ops.aten.permute.default(_unsafe_view_120, [0, 2, 1]);  _unsafe_view_120 = None
        permute_688 = torch.ops.aten.permute.default(_unsafe_view_121, [0, 2, 1]);  _unsafe_view_121 = None
        permute_693 = torch.ops.aten.permute.default(permute_123, [1, 0]);  permute_123 = None
        permute_698 = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None
        permute_703 = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None
        permute_707 = torch.ops.aten.permute.default(permute_118, [1, 0]);  permute_118 = None
        permute_711 = torch.ops.aten.permute.default(permute_117, [1, 0]);  permute_117 = None
        permute_715 = torch.ops.aten.permute.default(permute_116, [1, 0]);  permute_116 = None
        permute_719 = torch.ops.aten.permute.default(permute_115, [1, 0]);  permute_115 = None
        permute_722 = torch.ops.aten.permute.default(view_113, [0, 2, 1]);  view_113 = None
        permute_723 = torch.ops.aten.permute.default(_unsafe_view_111, [0, 2, 1]);  _unsafe_view_111 = None
        permute_724 = torch.ops.aten.permute.default(_unsafe_view_108, [0, 2, 1]);  _unsafe_view_108 = None
        permute_725 = torch.ops.aten.permute.default(_unsafe_view_109, [0, 2, 1]);  _unsafe_view_109 = None
        permute_730 = torch.ops.aten.permute.default(permute_111, [1, 0]);  permute_111 = None
        permute_735 = torch.ops.aten.permute.default(permute_109, [1, 0]);  permute_109 = None
        permute_740 = torch.ops.aten.permute.default(permute_107, [1, 0]);  permute_107 = None
        permute_744 = torch.ops.aten.permute.default(permute_106, [1, 0]);  permute_106 = None
        permute_747 = torch.ops.aten.permute.default(view_104, [0, 2, 1]);  view_104 = None
        permute_748 = torch.ops.aten.permute.default(_unsafe_view_102, [0, 2, 1]);  _unsafe_view_102 = None
        view_560 = torch.ops.aten.view.default(add_71, [16384]);  add_71 = None
        permute_750 = torch.ops.aten.permute.default(_unsafe_view_99, [0, 2, 1]);  _unsafe_view_99 = None
        permute_751 = torch.ops.aten.permute.default(_unsafe_view_100, [0, 2, 1]);  _unsafe_view_100 = None
        permute_756 = torch.ops.aten.permute.default(permute_101, [1, 0]);  permute_101 = None
        permute_761 = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None
        permute_766 = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
        view_572 = torch.ops.aten.view.default(view_97, [256]);  view_97 = None
        permute_770 = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None
        permute_774 = torch.ops.aten.permute.default(permute_95, [1, 0]);  permute_95 = None
        permute_778 = torch.ops.aten.permute.default(permute_94, [1, 0]);  permute_94 = None
        permute_782 = torch.ops.aten.permute.default(permute_93, [1, 0]);  permute_93 = None
        permute_785 = torch.ops.aten.permute.default(view_91, [0, 2, 1]);  view_91 = None
        permute_786 = torch.ops.aten.permute.default(_unsafe_view_90, [0, 2, 1]);  _unsafe_view_90 = None
        permute_787 = torch.ops.aten.permute.default(_unsafe_view_87, [0, 2, 1]);  _unsafe_view_87 = None
        permute_788 = torch.ops.aten.permute.default(_unsafe_view_88, [0, 2, 1]);  _unsafe_view_88 = None
        permute_793 = torch.ops.aten.permute.default(permute_89, [1, 0]);  permute_89 = None
        permute_798 = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None
        permute_803 = torch.ops.aten.permute.default(permute_85, [1, 0]);  permute_85 = None
        permute_807 = torch.ops.aten.permute.default(permute_84, [1, 0]);  permute_84 = None
        permute_811 = torch.ops.aten.permute.default(permute_83, [1, 0]);  permute_83 = None
        permute_815 = torch.ops.aten.permute.default(permute_82, [1, 0]);  permute_82 = None
        permute_819 = torch.ops.aten.permute.default(permute_81, [1, 0]);  permute_81 = None
        permute_822 = torch.ops.aten.permute.default(view_79, [0, 2, 1]);  view_79 = None
        permute_823 = torch.ops.aten.permute.default(_unsafe_view_78, [0, 2, 1]);  _unsafe_view_78 = None
        permute_824 = torch.ops.aten.permute.default(_unsafe_view_75, [0, 2, 1]);  _unsafe_view_75 = None
        permute_825 = torch.ops.aten.permute.default(_unsafe_view_76, [0, 2, 1]);  _unsafe_view_76 = None
        permute_830 = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
        permute_835 = torch.ops.aten.permute.default(permute_75, [1, 0]);  permute_75 = None
        permute_840 = torch.ops.aten.permute.default(permute_73, [1, 0]);  permute_73 = None
        permute_844 = torch.ops.aten.permute.default(permute_72, [1, 0]);  permute_72 = None
        permute_848 = torch.ops.aten.permute.default(permute_71, [1, 0]);  permute_71 = None
        permute_852 = torch.ops.aten.permute.default(permute_70, [1, 0]);  permute_70 = None
        permute_856 = torch.ops.aten.permute.default(permute_69, [1, 0]);  permute_69 = None
        permute_859 = torch.ops.aten.permute.default(view_67, [0, 2, 1]);  view_67 = None
        permute_860 = torch.ops.aten.permute.default(_unsafe_view_66, [0, 2, 1]);  _unsafe_view_66 = None
        permute_861 = torch.ops.aten.permute.default(_unsafe_view_63, [0, 2, 1]);  _unsafe_view_63 = None
        permute_862 = torch.ops.aten.permute.default(_unsafe_view_64, [0, 2, 1]);  _unsafe_view_64 = None
        permute_867 = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
        permute_872 = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
        permute_877 = torch.ops.aten.permute.default(permute_61, [1, 0]);  permute_61 = None
        permute_881 = torch.ops.aten.permute.default(permute_60, [1, 0]);  permute_60 = None
        permute_885 = torch.ops.aten.permute.default(permute_59, [1, 0]);  permute_59 = None
        permute_889 = torch.ops.aten.permute.default(permute_58, [1, 0]);  permute_58 = None
        permute_893 = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None
        permute_896 = torch.ops.aten.permute.default(view_55, [0, 2, 1]);  view_55 = None
        permute_897 = torch.ops.aten.permute.default(_unsafe_view_54, [0, 2, 1]);  _unsafe_view_54 = None
        permute_898 = torch.ops.aten.permute.default(_unsafe_view_51, [0, 2, 1]);  _unsafe_view_51 = None
        permute_899 = torch.ops.aten.permute.default(_unsafe_view_52, [0, 2, 1]);  _unsafe_view_52 = None
        permute_904 = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
        permute_909 = torch.ops.aten.permute.default(permute_51, [1, 0]);  permute_51 = None
        permute_914 = torch.ops.aten.permute.default(permute_49, [1, 0]);  permute_49 = None
        permute_918 = torch.ops.aten.permute.default(permute_48, [1, 0]);  permute_48 = None
        permute_922 = torch.ops.aten.permute.default(permute_47, [1, 0]);  permute_47 = None
        permute_926 = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None
        permute_930 = torch.ops.aten.permute.default(permute_45, [1, 0]);  permute_45 = None
        permute_933 = torch.ops.aten.permute.default(view_43, [0, 2, 1]);  view_43 = None
        permute_934 = torch.ops.aten.permute.default(_unsafe_view_42, [0, 2, 1]);  _unsafe_view_42 = None
        permute_935 = torch.ops.aten.permute.default(_unsafe_view_39, [0, 2, 1]);  _unsafe_view_39 = None
        permute_936 = torch.ops.aten.permute.default(_unsafe_view_40, [0, 2, 1]);  _unsafe_view_40 = None
        permute_941 = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
        permute_946 = torch.ops.aten.permute.default(permute_39, [1, 0]);  permute_39 = None
        permute_951 = torch.ops.aten.permute.default(permute_37, [1, 0]);  permute_37 = None
        permute_955 = torch.ops.aten.permute.default(permute_36, [1, 0]);  permute_36 = None
        permute_959 = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
        permute_963 = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
        permute_967 = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
        permute_970 = torch.ops.aten.permute.default(view_31, [0, 2, 1]);  view_31 = None
        permute_971 = torch.ops.aten.permute.default(_unsafe_view_30, [0, 2, 1]);  _unsafe_view_30 = None
        permute_972 = torch.ops.aten.permute.default(_unsafe_view_27, [0, 2, 1]);  _unsafe_view_27 = None
        permute_973 = torch.ops.aten.permute.default(_unsafe_view_28, [0, 2, 1]);  _unsafe_view_28 = None
        permute_978 = torch.ops.aten.permute.default(permute_29, [1, 0]);  permute_29 = None
        permute_983 = torch.ops.aten.permute.default(permute_27, [1, 0]);  permute_27 = None
        permute_988 = torch.ops.aten.permute.default(permute_25, [1, 0]);  permute_25 = None
        permute_992 = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
        permute_996 = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
        permute_1000 = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
        permute_1004 = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
        permute_1007 = torch.ops.aten.permute.default(view_19, [0, 2, 1]);  view_19 = None
        permute_1008 = torch.ops.aten.permute.default(_unsafe_view_18, [0, 2, 1]);  _unsafe_view_18 = None
        permute_1009 = torch.ops.aten.permute.default(_unsafe_view_15, [0, 2, 1]);  _unsafe_view_15 = None
        permute_1010 = torch.ops.aten.permute.default(_unsafe_view_16, [0, 2, 1]);  _unsafe_view_16 = None
        permute_1015 = torch.ops.aten.permute.default(permute_17, [1, 0]);  permute_17 = None
        permute_1020 = torch.ops.aten.permute.default(permute_15, [1, 0]);  permute_15 = None
        permute_1025 = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
        permute_1029 = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
        permute_1033 = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
        permute_1037 = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
        permute_1041 = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
        permute_1044 = torch.ops.aten.permute.default(view_7, [0, 2, 1]);  view_7 = None
        permute_1045 = torch.ops.aten.permute.default(_unsafe_view_6, [0, 2, 1]);  _unsafe_view_6 = None
        view_741 = torch.ops.aten.view.default(add_3, [16384]);  add_3 = None
        permute_1047 = torch.ops.aten.permute.default(_unsafe_view_3, [0, 2, 1]);  _unsafe_view_3 = None
        permute_1048 = torch.ops.aten.permute.default(_unsafe_view_4, [0, 2, 1]);  _unsafe_view_4 = None
        permute_1053 = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
        permute_1058 = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        permute_1063 = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        view_753 = torch.ops.aten.view.default(view, [256]);  view = None
        return [mean_42, _unsafe_view_264, permute_100, permute_102, permute_110, permute_112, permute_122, permute_124, permute_131, permute_133, permute_143, permute_145, permute_152, permute_154, permute_164, permute_166, permute_173, permute_175, permute_185, permute_187, permute_194, permute_196, permute_206, permute_208, permute_215, permute_217, permute_227, permute_229, permute_236, permute_238, permute_248, permute_250, permute_257, permute_259, mul_160, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, embedding, gt, reciprocal, div_2, philox_seed_like, view_9, gt_3, add_6, reciprocal_1, mm_4, sub_3, mm_5, gt_4, view_12, gt_5, add_11, reciprocal_3, div_3, view_21, gt_7, add_14, reciprocal_4, mm_11, sub_5, mm_12, gt_8, view_24, gt_9, add_19, reciprocal_6, div_4, view_33, gt_11, add_22, reciprocal_7, mm_18, sub_7, mm_19, gt_12, view_36, gt_13, add_27, reciprocal_9, div_5, view_45, gt_15, add_30, reciprocal_10, mm_25, sub_9, mm_26, gt_16, view_48, gt_17, add_35, reciprocal_12, div_6, view_57, gt_19, add_38, reciprocal_13, mm_32, sub_11, mm_33, gt_20, view_60, gt_21, add_43, reciprocal_15, div_7, view_69, gt_23, add_46, reciprocal_16, mm_39, sub_13, mm_40, gt_24, view_72, gt_25, add_51, reciprocal_18, div_8, view_81, gt_27, add_54, reciprocal_19, mm_46, sub_15, mm_47, gt_28, view_84, gt_29, add_59, reciprocal_21, div_9, view_93, gt_31, add_62, reciprocal_22, mm_53, sub_17, mm_54, gt_32, view_96, gt_33, add_67, reciprocal_24, gt_34, embedding_2, gt_35, reciprocal_25, div_12, view_106, gt_37, add_74, reciprocal_26, div_13, view_115, gt_39, add_78, reciprocal_27, mm_64, sub_23, mm_65, gt_40, view_118, gt_41, add_83, reciprocal_29, div_14, view_127, gt_43, add_86, reciprocal_30, div_15, view_136, gt_45, add_89, reciprocal_31, mm_75, sub_26, mm_76, gt_46, view_139, gt_47, add_94, reciprocal_33, div_16, view_148, gt_49, add_97, reciprocal_34, div_17, view_157, gt_51, add_100, reciprocal_35, mm_86, sub_29, mm_87, gt_52, view_160, gt_53, add_105, reciprocal_37, div_18, view_169, gt_55, add_108, reciprocal_38, div_19, view_178, gt_57, add_111, reciprocal_39, mm_97, sub_32, mm_98, gt_58, view_181, gt_59, add_116, reciprocal_41, div_20, view_190, gt_61, add_119, reciprocal_42, div_21, view_199, gt_63, add_122, reciprocal_43, mm_108, sub_35, mm_109, gt_64, view_202, gt_65, add_127, reciprocal_45, div_22, view_211, gt_67, add_130, reciprocal_46, div_23, view_220, gt_69, add_133, reciprocal_47, mm_119, sub_38, mm_120, gt_70, view_223, gt_71, add_138, reciprocal_49, div_24, view_232, gt_73, add_141, reciprocal_50, div_25, view_241, gt_75, add_144, reciprocal_51, mm_130, sub_41, mm_131, gt_76, view_244, gt_77, add_149, reciprocal_53, div_26, view_253, gt_79, add_152, reciprocal_54, div_27, view_262, gt_81, add_155, reciprocal_55, mm_141, sub_44, mm_142, gt_82, view_265, gt_83, add_160, reciprocal_57, gt_84, view_266, sub_46, unsqueeze_17, permute_269, permute_273, permute_277, permute_281, permute_285, permute_288, permute_289, permute_290, permute_291, permute_296, permute_301, permute_306, permute_310, permute_313, permute_314, permute_315, permute_316, permute_321, permute_326, permute_331, permute_335, permute_339, permute_343, permute_347, permute_350, permute_351, permute_352, permute_353, permute_358, permute_363, permute_368, permute_372, permute_375, permute_376, permute_377, permute_378, permute_383, permute_388, permute_393, permute_397, permute_401, permute_405, permute_409, permute_412, permute_413, permute_414, permute_415, permute_420, permute_425, permute_430, permute_434, permute_437, permute_438, permute_439, permute_440, permute_445, permute_450, permute_455, permute_459, permute_463, permute_467, permute_471, permute_474, permute_475, permute_476, permute_477, permute_482, permute_487, permute_492, permute_496, permute_499, permute_500, permute_501, permute_502, permute_507, permute_512, permute_517, permute_521, permute_525, permute_529, permute_533, permute_536, permute_537, permute_538, permute_539, permute_544, permute_549, permute_554, permute_558, permute_561, permute_562, permute_563, permute_564, permute_569, permute_574, permute_579, permute_583, permute_587, permute_591, permute_595, permute_598, permute_599, permute_600, permute_601, permute_606, permute_611, permute_616, permute_620, permute_623, permute_624, permute_625, permute_626, permute_631, permute_636, permute_641, permute_645, permute_649, permute_653, permute_657, permute_660, permute_661, permute_662, permute_663, permute_668, permute_673, permute_678, permute_682, permute_685, permute_686, permute_687, permute_688, permute_693, permute_698, permute_703, permute_707, permute_711, permute_715, permute_719, permute_722, permute_723, permute_724, permute_725, permute_730, permute_735, permute_740, permute_744, permute_747, permute_748, view_560, permute_750, permute_751, permute_756, permute_761, permute_766, view_572, permute_770, permute_774, permute_778, permute_782, permute_785, permute_786, permute_787, permute_788, permute_793, permute_798, permute_803, permute_807, permute_811, permute_815, permute_819, permute_822, permute_823, permute_824, permute_825, permute_830, permute_835, permute_840, permute_844, permute_848, permute_852, permute_856, permute_859, permute_860, permute_861, permute_862, permute_867, permute_872, permute_877, permute_881, permute_885, permute_889, permute_893, permute_896, permute_897, permute_898, permute_899, permute_904, permute_909, permute_914, permute_918, permute_922, permute_926, permute_930, permute_933, permute_934, permute_935, permute_936, permute_941, permute_946, permute_951, permute_955, permute_959, permute_963, permute_967, permute_970, permute_971, permute_972, permute_973, permute_978, permute_983, permute_988, permute_992, permute_996, permute_1000, permute_1004, permute_1007, permute_1008, permute_1009, permute_1010, permute_1015, permute_1020, permute_1025, permute_1029, permute_1033, permute_1037, permute_1041, permute_1044, permute_1045, view_741, permute_1047, permute_1048, permute_1053, permute_1058, permute_1063, view_753]
        
args = [((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((250112, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((32, 6), (6, 1), torch.float32, 'cuda'), ((512, 384), (384, 1), torch.float32, 'cuda'), ((1024, 512), (512, 1), torch.float32, 'cuda'), ((1024, 512), (512, 1), torch.float32, 'cuda'), ((512, 1024), (1024, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((512, 384), (384, 1), torch.float32, 'cuda'), ((1024, 512), (512, 1), torch.float32, 'cuda'), ((1024, 512), (512, 1), torch.float32, 'cuda'), ((512, 1024), (1024, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((512, 384), (384, 1), torch.float32, 'cuda'), ((1024, 512), (512, 1), torch.float32, 'cuda'), ((1024, 512), (512, 1), torch.float32, 'cuda'), ((512, 1024), (1024, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((512, 384), (384, 1), torch.float32, 'cuda'), ((1024, 512), (512, 1), torch.float32, 'cuda'), ((1024, 512), (512, 1), torch.float32, 'cuda'), ((512, 1024), (1024, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((512, 384), (384, 1), torch.float32, 'cuda'), ((1024, 512), (512, 1), torch.float32, 'cuda'), ((1024, 512), (512, 1), torch.float32, 'cuda'), ((512, 1024), (1024, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((512, 384), (384, 1), torch.float32, 'cuda'), ((1024, 512), (512, 1), torch.float32, 'cuda'), ((1024, 512), (512, 1), torch.float32, 'cuda'), ((512, 1024), (1024, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((512, 384), (384, 1), torch.float32, 'cuda'), ((1024, 512), (512, 1), torch.float32, 'cuda'), ((1024, 512), (512, 1), torch.float32, 'cuda'), ((512, 1024), (1024, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((512, 384), (384, 1), torch.float32, 'cuda'), ((1024, 512), (512, 1), torch.float32, 'cuda'), ((1024, 512), (512, 1), torch.float32, 'cuda'), ((512, 1024), (1024, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((32, 6), (6, 1), torch.float32, 'cuda'), ((512, 384), (384, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((512, 384), (384, 1), torch.float32, 'cuda'), ((1024, 512), (512, 1), torch.float32, 'cuda'), ((1024, 512), (512, 1), torch.float32, 'cuda'), ((512, 1024), (1024, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((512, 384), (384, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((512, 384), (384, 1), torch.float32, 'cuda'), ((1024, 512), (512, 1), torch.float32, 'cuda'), ((1024, 512), (512, 1), torch.float32, 'cuda'), ((512, 1024), (1024, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((512, 384), (384, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((512, 384), (384, 1), torch.float32, 'cuda'), ((1024, 512), (512, 1), torch.float32, 'cuda'), ((1024, 512), (512, 1), torch.float32, 'cuda'), ((512, 1024), (1024, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((512, 384), (384, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((512, 384), (384, 1), torch.float32, 'cuda'), ((1024, 512), (512, 1), torch.float32, 'cuda'), ((1024, 512), (512, 1), torch.float32, 'cuda'), ((512, 1024), (1024, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((512, 384), (384, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((512, 384), (384, 1), torch.float32, 'cuda'), ((1024, 512), (512, 1), torch.float32, 'cuda'), ((1024, 512), (512, 1), torch.float32, 'cuda'), ((512, 1024), (1024, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((512, 384), (384, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((512, 384), (384, 1), torch.float32, 'cuda'), ((1024, 512), (512, 1), torch.float32, 'cuda'), ((1024, 512), (512, 1), torch.float32, 'cuda'), ((512, 1024), (1024, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((512, 384), (384, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((512, 384), (384, 1), torch.float32, 'cuda'), ((1024, 512), (512, 1), torch.float32, 'cuda'), ((1024, 512), (512, 1), torch.float32, 'cuda'), ((512, 1024), (1024, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((512, 384), (384, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((512, 384), (384, 1), torch.float32, 'cuda'), ((1024, 512), (512, 1), torch.float32, 'cuda'), ((1024, 512), (512, 1), torch.float32, 'cuda'), ((512, 1024), (1024, 1), torch.float32, 'cuda'), ((250112, 512), (512, 1), torch.float32, 'cuda'), ((2, 128), (128, 1), torch.int64, 'cuda'), ((2, 128), (128, 1), torch.int64, 'cuda'), ((2, 128), (128, 1), torch.int64, 'cuda')]
args = [rand_strided(sh, st, dt, dev) for (sh, st, dt, dev) in args]
mod = make_fx(Repro().to(device="cuda"))(*args)

from torch._inductor.compile_fx import compile_fx_inner
from torch._dynamo.debug_utils import same_two_models

compiled = compile_fx_inner(mod, args)
compiled(args)
