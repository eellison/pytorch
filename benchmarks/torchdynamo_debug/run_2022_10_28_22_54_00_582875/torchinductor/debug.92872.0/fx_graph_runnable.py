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
        self.register_buffer('_tensor_constant3', torch.randn([], dtype=torch.float32))
        self.register_buffer('_tensor_constant4', torch.randn([], dtype=torch.float32))
        self.register_buffer('_tensor_constant5', torch.randn([], dtype=torch.float32))
        self.register_buffer('_tensor_constant6', torch.randn([], dtype=torch.float32))
        self.register_buffer('_tensor_constant7', torch.randn([], dtype=torch.float32))
        self.register_buffer('_tensor_constant8', torch.randn([], dtype=torch.float32))
        self.register_buffer('_tensor_constant9', torch.randn([], dtype=torch.float32))
        self.register_buffer('_tensor_constant10', torch.randn([], dtype=torch.float32))
        self.register_buffer('_tensor_constant11', torch.randn([], dtype=torch.float32))
        self.register_buffer('_tensor_constant12', torch.randn([], dtype=torch.float32))
        self.register_buffer('_tensor_constant13', torch.randn([], dtype=torch.float32))
        self.register_buffer('_tensor_constant14', torch.randn([], dtype=torch.float32))
        self.register_buffer('_tensor_constant15', torch.randn([], dtype=torch.float32))
        self.register_buffer('_tensor_constant16', torch.randn([], dtype=torch.float32))
        self.register_buffer('_tensor_constant17', torch.randn([], dtype=torch.float32))
        self.register_buffer('_tensor_constant18', torch.randn([], dtype=torch.float32))
        self.register_buffer('_tensor_constant19', torch.randn([], dtype=torch.float32))
        self.register_buffer('_tensor_constant20', torch.randn([], dtype=torch.float32))
        self.register_buffer('_tensor_constant21', torch.randn([], dtype=torch.float32))
        self.register_buffer('_tensor_constant22', torch.randn([], dtype=torch.float32))
        self.register_buffer('_tensor_constant23', torch.randn([], dtype=torch.float32))
        self.register_buffer('_tensor_constant24', torch.randn([], dtype=torch.float32))
        self.register_buffer('_tensor_constant25', torch.randn([], dtype=torch.float32))
        self.register_buffer('_tensor_constant26', torch.randn([], dtype=torch.float32))

    
    
    def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206):
        ones = torch.ops.aten.ones.default([64, 128], device = device(type='cuda', index=0), pin_memory = False)
        alias = torch.ops.aten.alias.default(ones);  ones = None
        alias_1 = torch.ops.aten.alias.default(alias);  alias = None
        slice_1 = torch.ops.aten.slice.Tensor(primals_203, 0, 0, 9223372036854775807);  primals_203 = None
        slice_2 = torch.ops.aten.slice.Tensor(slice_1, 1, 0, 128);  slice_1 = None
        expand = torch.ops.aten.expand.default(slice_2, [64, 128])
        slice_3 = torch.ops.aten.slice.Tensor(alias_1, 0, 0, 9223372036854775807);  alias_1 = None
        unsqueeze = torch.ops.aten.unsqueeze.default(slice_3, 1);  slice_3 = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
        slice_4 = torch.ops.aten.slice.Tensor(unsqueeze_1, 3, 0, 9223372036854775807);  unsqueeze_1 = None
        _tensor_constant0 = self._tensor_constant0
        lift_fresh_copy = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None
        sub = torch.ops.aten.sub.Tensor(lift_fresh_copy, slice_4);  lift_fresh_copy = slice_4 = None
        mul = torch.ops.aten.mul.Tensor(sub, -3.4028234663852886e+38);  sub = None
        slice_5 = torch.ops.aten.slice.Tensor(primals_204, 0, 0, 9223372036854775807);  primals_204 = None
        slice_6 = torch.ops.aten.slice.Tensor(slice_5, 1, 0, 128);  slice_5 = None
        embedding = torch.ops.aten.embedding.default(primals_1, primals_205, 0)
        embedding_1 = torch.ops.aten.embedding.default(primals_2, expand);  primals_2 = expand = None
        add = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
        embedding_2 = torch.ops.aten.embedding.default(primals_3, slice_6);  primals_3 = None
        add_1 = torch.ops.aten.add.Tensor(add, embedding_2);  add = embedding_2 = None
        var_mean = torch.ops.aten.var_mean.correction(add_1, [2], correction = 0, keepdim = True)
        getitem = var_mean[0]
        getitem_1 = var_mean[1];  var_mean = None
        add_2 = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        sqrt = torch.ops.aten.sqrt.default(add_2);  add_2 = None
        reciprocal = torch.ops.aten.reciprocal.default(sqrt);  sqrt = None
        sub_1 = torch.ops.aten.sub.Tensor(add_1, getitem_1);  add_1 = getitem_1 = None
        mul_1 = torch.ops.aten.mul.Tensor(sub_1, reciprocal);  sub_1 = None
        mul_2 = torch.ops.aten.mul.Tensor(mul_1, primals_4)
        add_3 = torch.ops.aten.add.Tensor(mul_2, primals_5);  mul_2 = primals_5 = None
        convert_element_type = torch.ops.prims.convert_element_type.default(add_3, torch.float32);  add_3 = None
        rand_like = torch.ops.aten.rand_like.default(convert_element_type, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_2 = torch.ops.aten.alias.default(rand_like);  rand_like = None
        gt = torch.ops.aten.gt.Scalar(alias_2, 0.1);  alias_2 = None
        mul_3 = torch.ops.aten.mul.Tensor(gt, convert_element_type);  convert_element_type = None
        mul_4 = torch.ops.aten.mul.Tensor(mul_3, 1.1111111111111112);  mul_3 = None
        permute = torch.ops.aten.permute.default(primals_6, [1, 0]);  primals_6 = None
        view = torch.ops.aten.view.default(mul_4, [8192, 768])
        addmm = torch.ops.aten.addmm.default(primals_7, view, permute);  primals_7 = None
        view_1 = torch.ops.aten.view.default(addmm, [64, 128, 768]);  addmm = None
        permute_1 = torch.ops.aten.permute.default(primals_8, [1, 0]);  primals_8 = None
        addmm_1 = torch.ops.aten.addmm.default(primals_9, view, permute_1);  primals_9 = None
        view_3 = torch.ops.aten.view.default(addmm_1, [64, 128, 768]);  addmm_1 = None
        view_4 = torch.ops.aten.view.default(view_3, [64, 128, 12, 64]);  view_3 = None
        permute_2 = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
        permute_3 = torch.ops.aten.permute.default(primals_10, [1, 0]);  primals_10 = None
        addmm_2 = torch.ops.aten.addmm.default(primals_11, view, permute_3);  primals_11 = None
        view_6 = torch.ops.aten.view.default(addmm_2, [64, 128, 768]);  addmm_2 = None
        view_7 = torch.ops.aten.view.default(view_6, [64, 128, 12, 64]);  view_6 = None
        permute_4 = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None
        view_8 = torch.ops.aten.view.default(view_1, [64, 128, 12, 64]);  view_1 = None
        permute_5 = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
        permute_6 = torch.ops.aten.permute.default(permute_2, [0, 1, 3, 2]);  permute_2 = None
        expand_1 = torch.ops.aten.expand.default(permute_5, [64, 12, 128, 64]);  permute_5 = None
        clone = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        _unsafe_view = torch.ops.aten._unsafe_view.default(clone, [768, 128, 64]);  clone = None
        expand_2 = torch.ops.aten.expand.default(permute_6, [64, 12, 64, 128]);  permute_6 = None
        clone_1 = torch.ops.aten.clone.default(expand_2, memory_format = torch.contiguous_format);  expand_2 = None
        _unsafe_view_1 = torch.ops.aten._unsafe_view.default(clone_1, [768, 64, 128]);  clone_1 = None
        bmm = torch.ops.aten.bmm.default(_unsafe_view, _unsafe_view_1)
        _unsafe_view_2 = torch.ops.aten._unsafe_view.default(bmm, [64, 12, 128, 128]);  bmm = None
        div = torch.ops.aten.div.Tensor(_unsafe_view_2, 8.0);  _unsafe_view_2 = None
        add_4 = torch.ops.aten.add.Tensor(div, mul);  div = None
        amax = torch.ops.aten.amax.default(add_4, [-1], True)
        sub_2 = torch.ops.aten.sub.Tensor(add_4, amax);  add_4 = amax = None
        exp = torch.ops.aten.exp.default(sub_2);  sub_2 = None
        sum_1 = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div_1 = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        alias_4 = torch.ops.aten.alias.default(div_1)
        alias_5 = torch.ops.aten.alias.default(alias_4);  alias_4 = None
        rand_like_1 = torch.ops.aten.rand_like.default(div_1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_6 = torch.ops.aten.alias.default(rand_like_1);  rand_like_1 = None
        gt_1 = torch.ops.aten.gt.Scalar(alias_6, 0.1);  alias_6 = None
        mul_5 = torch.ops.aten.mul.Tensor(gt_1, div_1);  div_1 = None
        mul_6 = torch.ops.aten.mul.Tensor(mul_5, 1.1111111111111112);  mul_5 = None
        expand_3 = torch.ops.aten.expand.default(mul_6, [64, 12, 128, 128]);  mul_6 = None
        view_9 = torch.ops.aten.view.default(expand_3, [768, 128, 128]);  expand_3 = None
        expand_4 = torch.ops.aten.expand.default(permute_4, [64, 12, 128, 64]);  permute_4 = None
        clone_2 = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
        _unsafe_view_3 = torch.ops.aten._unsafe_view.default(clone_2, [768, 128, 64]);  clone_2 = None
        bmm_1 = torch.ops.aten.bmm.default(view_9, _unsafe_view_3)
        _unsafe_view_4 = torch.ops.aten._unsafe_view.default(bmm_1, [64, 12, 128, 64]);  bmm_1 = None
        permute_7 = torch.ops.aten.permute.default(_unsafe_view_4, [0, 2, 1, 3]);  _unsafe_view_4 = None
        clone_3 = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
        view_10 = torch.ops.aten.view.default(clone_3, [64, 128, 768]);  clone_3 = None
        permute_8 = torch.ops.aten.permute.default(primals_12, [1, 0]);  primals_12 = None
        view_11 = torch.ops.aten.view.default(view_10, [8192, 768]);  view_10 = None
        addmm_3 = torch.ops.aten.addmm.default(primals_13, view_11, permute_8);  primals_13 = None
        view_12 = torch.ops.aten.view.default(addmm_3, [64, 128, 768]);  addmm_3 = None
        rand_like_2 = torch.ops.aten.rand_like.default(view_12, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_7 = torch.ops.aten.alias.default(rand_like_2);  rand_like_2 = None
        gt_2 = torch.ops.aten.gt.Scalar(alias_7, 0.1);  alias_7 = None
        mul_7 = torch.ops.aten.mul.Tensor(gt_2, view_12);  view_12 = None
        mul_8 = torch.ops.aten.mul.Tensor(mul_7, 1.1111111111111112);  mul_7 = None
        add_5 = torch.ops.aten.add.Tensor(mul_8, mul_4);  mul_8 = mul_4 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(add_5, [2], correction = 0, keepdim = True)
        getitem_2 = var_mean_1[0]
        getitem_3 = var_mean_1[1];  var_mean_1 = None
        add_6 = torch.ops.aten.add.Tensor(getitem_2, 1e-12);  getitem_2 = None
        sqrt_1 = torch.ops.aten.sqrt.default(add_6);  add_6 = None
        reciprocal_1 = torch.ops.aten.reciprocal.default(sqrt_1);  sqrt_1 = None
        sub_3 = torch.ops.aten.sub.Tensor(add_5, getitem_3);  add_5 = getitem_3 = None
        mul_9 = torch.ops.aten.mul.Tensor(sub_3, reciprocal_1);  sub_3 = None
        mul_10 = torch.ops.aten.mul.Tensor(mul_9, primals_14)
        add_7 = torch.ops.aten.add.Tensor(mul_10, primals_15);  mul_10 = primals_15 = None
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(add_7, torch.float32);  add_7 = None
        permute_9 = torch.ops.aten.permute.default(primals_16, [1, 0]);  primals_16 = None
        view_13 = torch.ops.aten.view.default(convert_element_type_1, [8192, 768])
        addmm_4 = torch.ops.aten.addmm.default(primals_17, view_13, permute_9);  primals_17 = None
        view_14 = torch.ops.aten.view.default(addmm_4, [64, 128, 3072]);  addmm_4 = None
        mul_11 = torch.ops.aten.mul.Tensor(view_14, 0.5)
        mul_12 = torch.ops.aten.mul.Tensor(view_14, 0.7071067811865476)
        sign = torch.ops.aten.sign.default(mul_12)
        abs_1 = torch.ops.aten.abs.default(mul_12);  mul_12 = None
        mul_13 = torch.ops.aten.mul.Tensor(abs_1, 0.3275911)
        add_8 = torch.ops.aten.add.Tensor(mul_13, 1.0);  mul_13 = None
        reciprocal_2 = torch.ops.aten.reciprocal.default(add_8);  add_8 = None
        mul_14 = torch.ops.aten.mul.Tensor(reciprocal_2, 1.0);  reciprocal_2 = None
        mul_15 = torch.ops.aten.mul.Tensor(mul_14, 1.061405429)
        add_9 = torch.ops.aten.add.Tensor(mul_15, -1.453152027);  mul_15 = None
        mul_16 = torch.ops.aten.mul.Tensor(add_9, mul_14);  add_9 = None
        add_10 = torch.ops.aten.add.Tensor(mul_16, 1.421413741);  mul_16 = None
        mul_17 = torch.ops.aten.mul.Tensor(add_10, mul_14);  add_10 = None
        add_11 = torch.ops.aten.add.Tensor(mul_17, -0.284496736);  mul_17 = None
        mul_18 = torch.ops.aten.mul.Tensor(add_11, mul_14);  add_11 = None
        add_12 = torch.ops.aten.add.Tensor(mul_18, 0.254829592);  mul_18 = None
        mul_19 = torch.ops.aten.mul.Tensor(add_12, mul_14);  add_12 = mul_14 = None
        neg = torch.ops.aten.neg.default(abs_1)
        mul_20 = torch.ops.aten.mul.Tensor(neg, abs_1);  neg = abs_1 = None
        exp_1 = torch.ops.aten.exp.default(mul_20);  mul_20 = None
        mul_21 = torch.ops.aten.mul.Tensor(mul_19, exp_1);  mul_19 = exp_1 = None
        _tensor_constant1 = self._tensor_constant1
        lift_fresh_copy_1 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant1);  _tensor_constant1 = None
        sub_4 = torch.ops.aten.sub.Tensor(lift_fresh_copy_1, mul_21);  lift_fresh_copy_1 = None
        mul_22 = torch.ops.aten.mul.Tensor(sign, sub_4);  sub_4 = None
        add_13 = torch.ops.aten.add.Tensor(mul_22, 1);  mul_22 = None
        mul_23 = torch.ops.aten.mul.Tensor(mul_11, add_13);  mul_11 = add_13 = None
        permute_10 = torch.ops.aten.permute.default(primals_18, [1, 0]);  primals_18 = None
        view_15 = torch.ops.aten.view.default(mul_23, [8192, 3072]);  mul_23 = None
        addmm_5 = torch.ops.aten.addmm.default(primals_19, view_15, permute_10);  primals_19 = None
        view_16 = torch.ops.aten.view.default(addmm_5, [64, 128, 768]);  addmm_5 = None
        rand_like_3 = torch.ops.aten.rand_like.default(view_16, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_8 = torch.ops.aten.alias.default(rand_like_3);  rand_like_3 = None
        gt_3 = torch.ops.aten.gt.Scalar(alias_8, 0.1);  alias_8 = None
        mul_24 = torch.ops.aten.mul.Tensor(gt_3, view_16);  view_16 = None
        mul_25 = torch.ops.aten.mul.Tensor(mul_24, 1.1111111111111112);  mul_24 = None
        add_14 = torch.ops.aten.add.Tensor(mul_25, convert_element_type_1);  mul_25 = convert_element_type_1 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(add_14, [2], correction = 0, keepdim = True)
        getitem_4 = var_mean_2[0]
        getitem_5 = var_mean_2[1];  var_mean_2 = None
        add_15 = torch.ops.aten.add.Tensor(getitem_4, 1e-12);  getitem_4 = None
        sqrt_2 = torch.ops.aten.sqrt.default(add_15);  add_15 = None
        reciprocal_3 = torch.ops.aten.reciprocal.default(sqrt_2);  sqrt_2 = None
        sub_5 = torch.ops.aten.sub.Tensor(add_14, getitem_5);  add_14 = getitem_5 = None
        mul_26 = torch.ops.aten.mul.Tensor(sub_5, reciprocal_3);  sub_5 = None
        mul_27 = torch.ops.aten.mul.Tensor(mul_26, primals_20)
        add_16 = torch.ops.aten.add.Tensor(mul_27, primals_21);  mul_27 = primals_21 = None
        convert_element_type_2 = torch.ops.prims.convert_element_type.default(add_16, torch.float32);  add_16 = None
        permute_11 = torch.ops.aten.permute.default(primals_22, [1, 0]);  primals_22 = None
        view_17 = torch.ops.aten.view.default(convert_element_type_2, [8192, 768])
        addmm_6 = torch.ops.aten.addmm.default(primals_23, view_17, permute_11);  primals_23 = None
        view_18 = torch.ops.aten.view.default(addmm_6, [64, 128, 768]);  addmm_6 = None
        permute_12 = torch.ops.aten.permute.default(primals_24, [1, 0]);  primals_24 = None
        addmm_7 = torch.ops.aten.addmm.default(primals_25, view_17, permute_12);  primals_25 = None
        view_20 = torch.ops.aten.view.default(addmm_7, [64, 128, 768]);  addmm_7 = None
        view_21 = torch.ops.aten.view.default(view_20, [64, 128, 12, 64]);  view_20 = None
        permute_13 = torch.ops.aten.permute.default(view_21, [0, 2, 1, 3]);  view_21 = None
        permute_14 = torch.ops.aten.permute.default(primals_26, [1, 0]);  primals_26 = None
        addmm_8 = torch.ops.aten.addmm.default(primals_27, view_17, permute_14);  primals_27 = None
        view_23 = torch.ops.aten.view.default(addmm_8, [64, 128, 768]);  addmm_8 = None
        view_24 = torch.ops.aten.view.default(view_23, [64, 128, 12, 64]);  view_23 = None
        permute_15 = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3]);  view_24 = None
        view_25 = torch.ops.aten.view.default(view_18, [64, 128, 12, 64]);  view_18 = None
        permute_16 = torch.ops.aten.permute.default(view_25, [0, 2, 1, 3]);  view_25 = None
        permute_17 = torch.ops.aten.permute.default(permute_13, [0, 1, 3, 2]);  permute_13 = None
        expand_5 = torch.ops.aten.expand.default(permute_16, [64, 12, 128, 64]);  permute_16 = None
        clone_4 = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
        _unsafe_view_5 = torch.ops.aten._unsafe_view.default(clone_4, [768, 128, 64]);  clone_4 = None
        expand_6 = torch.ops.aten.expand.default(permute_17, [64, 12, 64, 128]);  permute_17 = None
        clone_5 = torch.ops.aten.clone.default(expand_6, memory_format = torch.contiguous_format);  expand_6 = None
        _unsafe_view_6 = torch.ops.aten._unsafe_view.default(clone_5, [768, 64, 128]);  clone_5 = None
        bmm_2 = torch.ops.aten.bmm.default(_unsafe_view_5, _unsafe_view_6)
        _unsafe_view_7 = torch.ops.aten._unsafe_view.default(bmm_2, [64, 12, 128, 128]);  bmm_2 = None
        div_2 = torch.ops.aten.div.Tensor(_unsafe_view_7, 8.0);  _unsafe_view_7 = None
        add_17 = torch.ops.aten.add.Tensor(div_2, mul);  div_2 = None
        amax_1 = torch.ops.aten.amax.default(add_17, [-1], True)
        sub_6 = torch.ops.aten.sub.Tensor(add_17, amax_1);  add_17 = amax_1 = None
        exp_2 = torch.ops.aten.exp.default(sub_6);  sub_6 = None
        sum_2 = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_3 = torch.ops.aten.div.Tensor(exp_2, sum_2);  exp_2 = sum_2 = None
        alias_10 = torch.ops.aten.alias.default(div_3)
        alias_11 = torch.ops.aten.alias.default(alias_10);  alias_10 = None
        rand_like_4 = torch.ops.aten.rand_like.default(div_3, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_12 = torch.ops.aten.alias.default(rand_like_4);  rand_like_4 = None
        gt_4 = torch.ops.aten.gt.Scalar(alias_12, 0.1);  alias_12 = None
        mul_28 = torch.ops.aten.mul.Tensor(gt_4, div_3);  div_3 = None
        mul_29 = torch.ops.aten.mul.Tensor(mul_28, 1.1111111111111112);  mul_28 = None
        expand_7 = torch.ops.aten.expand.default(mul_29, [64, 12, 128, 128]);  mul_29 = None
        view_26 = torch.ops.aten.view.default(expand_7, [768, 128, 128]);  expand_7 = None
        expand_8 = torch.ops.aten.expand.default(permute_15, [64, 12, 128, 64]);  permute_15 = None
        clone_6 = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
        _unsafe_view_8 = torch.ops.aten._unsafe_view.default(clone_6, [768, 128, 64]);  clone_6 = None
        bmm_3 = torch.ops.aten.bmm.default(view_26, _unsafe_view_8)
        _unsafe_view_9 = torch.ops.aten._unsafe_view.default(bmm_3, [64, 12, 128, 64]);  bmm_3 = None
        permute_18 = torch.ops.aten.permute.default(_unsafe_view_9, [0, 2, 1, 3]);  _unsafe_view_9 = None
        clone_7 = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
        view_27 = torch.ops.aten.view.default(clone_7, [64, 128, 768]);  clone_7 = None
        permute_19 = torch.ops.aten.permute.default(primals_28, [1, 0]);  primals_28 = None
        view_28 = torch.ops.aten.view.default(view_27, [8192, 768]);  view_27 = None
        addmm_9 = torch.ops.aten.addmm.default(primals_29, view_28, permute_19);  primals_29 = None
        view_29 = torch.ops.aten.view.default(addmm_9, [64, 128, 768]);  addmm_9 = None
        rand_like_5 = torch.ops.aten.rand_like.default(view_29, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_13 = torch.ops.aten.alias.default(rand_like_5);  rand_like_5 = None
        gt_5 = torch.ops.aten.gt.Scalar(alias_13, 0.1);  alias_13 = None
        mul_30 = torch.ops.aten.mul.Tensor(gt_5, view_29);  view_29 = None
        mul_31 = torch.ops.aten.mul.Tensor(mul_30, 1.1111111111111112);  mul_30 = None
        add_18 = torch.ops.aten.add.Tensor(mul_31, convert_element_type_2);  mul_31 = convert_element_type_2 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(add_18, [2], correction = 0, keepdim = True)
        getitem_6 = var_mean_3[0]
        getitem_7 = var_mean_3[1];  var_mean_3 = None
        add_19 = torch.ops.aten.add.Tensor(getitem_6, 1e-12);  getitem_6 = None
        sqrt_3 = torch.ops.aten.sqrt.default(add_19);  add_19 = None
        reciprocal_4 = torch.ops.aten.reciprocal.default(sqrt_3);  sqrt_3 = None
        sub_7 = torch.ops.aten.sub.Tensor(add_18, getitem_7);  add_18 = getitem_7 = None
        mul_32 = torch.ops.aten.mul.Tensor(sub_7, reciprocal_4);  sub_7 = None
        mul_33 = torch.ops.aten.mul.Tensor(mul_32, primals_30)
        add_20 = torch.ops.aten.add.Tensor(mul_33, primals_31);  mul_33 = primals_31 = None
        convert_element_type_3 = torch.ops.prims.convert_element_type.default(add_20, torch.float32);  add_20 = None
        permute_20 = torch.ops.aten.permute.default(primals_32, [1, 0]);  primals_32 = None
        view_30 = torch.ops.aten.view.default(convert_element_type_3, [8192, 768])
        addmm_10 = torch.ops.aten.addmm.default(primals_33, view_30, permute_20);  primals_33 = None
        view_31 = torch.ops.aten.view.default(addmm_10, [64, 128, 3072]);  addmm_10 = None
        mul_34 = torch.ops.aten.mul.Tensor(view_31, 0.5)
        mul_35 = torch.ops.aten.mul.Tensor(view_31, 0.7071067811865476)
        sign_1 = torch.ops.aten.sign.default(mul_35)
        abs_2 = torch.ops.aten.abs.default(mul_35);  mul_35 = None
        mul_36 = torch.ops.aten.mul.Tensor(abs_2, 0.3275911)
        add_21 = torch.ops.aten.add.Tensor(mul_36, 1.0);  mul_36 = None
        reciprocal_5 = torch.ops.aten.reciprocal.default(add_21);  add_21 = None
        mul_37 = torch.ops.aten.mul.Tensor(reciprocal_5, 1.0);  reciprocal_5 = None
        mul_38 = torch.ops.aten.mul.Tensor(mul_37, 1.061405429)
        add_22 = torch.ops.aten.add.Tensor(mul_38, -1.453152027);  mul_38 = None
        mul_39 = torch.ops.aten.mul.Tensor(add_22, mul_37);  add_22 = None
        add_23 = torch.ops.aten.add.Tensor(mul_39, 1.421413741);  mul_39 = None
        mul_40 = torch.ops.aten.mul.Tensor(add_23, mul_37);  add_23 = None
        add_24 = torch.ops.aten.add.Tensor(mul_40, -0.284496736);  mul_40 = None
        mul_41 = torch.ops.aten.mul.Tensor(add_24, mul_37);  add_24 = None
        add_25 = torch.ops.aten.add.Tensor(mul_41, 0.254829592);  mul_41 = None
        mul_42 = torch.ops.aten.mul.Tensor(add_25, mul_37);  add_25 = mul_37 = None
        neg_1 = torch.ops.aten.neg.default(abs_2)
        mul_43 = torch.ops.aten.mul.Tensor(neg_1, abs_2);  neg_1 = abs_2 = None
        exp_3 = torch.ops.aten.exp.default(mul_43);  mul_43 = None
        mul_44 = torch.ops.aten.mul.Tensor(mul_42, exp_3);  mul_42 = exp_3 = None
        _tensor_constant2 = self._tensor_constant2
        lift_fresh_copy_2 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant2);  _tensor_constant2 = None
        sub_8 = torch.ops.aten.sub.Tensor(lift_fresh_copy_2, mul_44);  lift_fresh_copy_2 = None
        mul_45 = torch.ops.aten.mul.Tensor(sign_1, sub_8);  sub_8 = None
        add_26 = torch.ops.aten.add.Tensor(mul_45, 1);  mul_45 = None
        mul_46 = torch.ops.aten.mul.Tensor(mul_34, add_26);  mul_34 = add_26 = None
        permute_21 = torch.ops.aten.permute.default(primals_34, [1, 0]);  primals_34 = None
        view_32 = torch.ops.aten.view.default(mul_46, [8192, 3072]);  mul_46 = None
        addmm_11 = torch.ops.aten.addmm.default(primals_35, view_32, permute_21);  primals_35 = None
        view_33 = torch.ops.aten.view.default(addmm_11, [64, 128, 768]);  addmm_11 = None
        rand_like_6 = torch.ops.aten.rand_like.default(view_33, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_14 = torch.ops.aten.alias.default(rand_like_6);  rand_like_6 = None
        gt_6 = torch.ops.aten.gt.Scalar(alias_14, 0.1);  alias_14 = None
        mul_47 = torch.ops.aten.mul.Tensor(gt_6, view_33);  view_33 = None
        mul_48 = torch.ops.aten.mul.Tensor(mul_47, 1.1111111111111112);  mul_47 = None
        add_27 = torch.ops.aten.add.Tensor(mul_48, convert_element_type_3);  mul_48 = convert_element_type_3 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(add_27, [2], correction = 0, keepdim = True)
        getitem_8 = var_mean_4[0]
        getitem_9 = var_mean_4[1];  var_mean_4 = None
        add_28 = torch.ops.aten.add.Tensor(getitem_8, 1e-12);  getitem_8 = None
        sqrt_4 = torch.ops.aten.sqrt.default(add_28);  add_28 = None
        reciprocal_6 = torch.ops.aten.reciprocal.default(sqrt_4);  sqrt_4 = None
        sub_9 = torch.ops.aten.sub.Tensor(add_27, getitem_9);  add_27 = getitem_9 = None
        mul_49 = torch.ops.aten.mul.Tensor(sub_9, reciprocal_6);  sub_9 = None
        mul_50 = torch.ops.aten.mul.Tensor(mul_49, primals_36)
        add_29 = torch.ops.aten.add.Tensor(mul_50, primals_37);  mul_50 = primals_37 = None
        convert_element_type_4 = torch.ops.prims.convert_element_type.default(add_29, torch.float32);  add_29 = None
        permute_22 = torch.ops.aten.permute.default(primals_38, [1, 0]);  primals_38 = None
        view_34 = torch.ops.aten.view.default(convert_element_type_4, [8192, 768])
        addmm_12 = torch.ops.aten.addmm.default(primals_39, view_34, permute_22);  primals_39 = None
        view_35 = torch.ops.aten.view.default(addmm_12, [64, 128, 768]);  addmm_12 = None
        permute_23 = torch.ops.aten.permute.default(primals_40, [1, 0]);  primals_40 = None
        addmm_13 = torch.ops.aten.addmm.default(primals_41, view_34, permute_23);  primals_41 = None
        view_37 = torch.ops.aten.view.default(addmm_13, [64, 128, 768]);  addmm_13 = None
        view_38 = torch.ops.aten.view.default(view_37, [64, 128, 12, 64]);  view_37 = None
        permute_24 = torch.ops.aten.permute.default(view_38, [0, 2, 1, 3]);  view_38 = None
        permute_25 = torch.ops.aten.permute.default(primals_42, [1, 0]);  primals_42 = None
        addmm_14 = torch.ops.aten.addmm.default(primals_43, view_34, permute_25);  primals_43 = None
        view_40 = torch.ops.aten.view.default(addmm_14, [64, 128, 768]);  addmm_14 = None
        view_41 = torch.ops.aten.view.default(view_40, [64, 128, 12, 64]);  view_40 = None
        permute_26 = torch.ops.aten.permute.default(view_41, [0, 2, 1, 3]);  view_41 = None
        view_42 = torch.ops.aten.view.default(view_35, [64, 128, 12, 64]);  view_35 = None
        permute_27 = torch.ops.aten.permute.default(view_42, [0, 2, 1, 3]);  view_42 = None
        permute_28 = torch.ops.aten.permute.default(permute_24, [0, 1, 3, 2]);  permute_24 = None
        expand_9 = torch.ops.aten.expand.default(permute_27, [64, 12, 128, 64]);  permute_27 = None
        clone_8 = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
        _unsafe_view_10 = torch.ops.aten._unsafe_view.default(clone_8, [768, 128, 64]);  clone_8 = None
        expand_10 = torch.ops.aten.expand.default(permute_28, [64, 12, 64, 128]);  permute_28 = None
        clone_9 = torch.ops.aten.clone.default(expand_10, memory_format = torch.contiguous_format);  expand_10 = None
        _unsafe_view_11 = torch.ops.aten._unsafe_view.default(clone_9, [768, 64, 128]);  clone_9 = None
        bmm_4 = torch.ops.aten.bmm.default(_unsafe_view_10, _unsafe_view_11)
        _unsafe_view_12 = torch.ops.aten._unsafe_view.default(bmm_4, [64, 12, 128, 128]);  bmm_4 = None
        div_4 = torch.ops.aten.div.Tensor(_unsafe_view_12, 8.0);  _unsafe_view_12 = None
        add_30 = torch.ops.aten.add.Tensor(div_4, mul);  div_4 = None
        amax_2 = torch.ops.aten.amax.default(add_30, [-1], True)
        sub_10 = torch.ops.aten.sub.Tensor(add_30, amax_2);  add_30 = amax_2 = None
        exp_4 = torch.ops.aten.exp.default(sub_10);  sub_10 = None
        sum_3 = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_5 = torch.ops.aten.div.Tensor(exp_4, sum_3);  exp_4 = sum_3 = None
        alias_16 = torch.ops.aten.alias.default(div_5)
        alias_17 = torch.ops.aten.alias.default(alias_16);  alias_16 = None
        rand_like_7 = torch.ops.aten.rand_like.default(div_5, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_18 = torch.ops.aten.alias.default(rand_like_7);  rand_like_7 = None
        gt_7 = torch.ops.aten.gt.Scalar(alias_18, 0.1);  alias_18 = None
        mul_51 = torch.ops.aten.mul.Tensor(gt_7, div_5);  div_5 = None
        mul_52 = torch.ops.aten.mul.Tensor(mul_51, 1.1111111111111112);  mul_51 = None
        expand_11 = torch.ops.aten.expand.default(mul_52, [64, 12, 128, 128]);  mul_52 = None
        view_43 = torch.ops.aten.view.default(expand_11, [768, 128, 128]);  expand_11 = None
        expand_12 = torch.ops.aten.expand.default(permute_26, [64, 12, 128, 64]);  permute_26 = None
        clone_10 = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
        _unsafe_view_13 = torch.ops.aten._unsafe_view.default(clone_10, [768, 128, 64]);  clone_10 = None
        bmm_5 = torch.ops.aten.bmm.default(view_43, _unsafe_view_13)
        _unsafe_view_14 = torch.ops.aten._unsafe_view.default(bmm_5, [64, 12, 128, 64]);  bmm_5 = None
        permute_29 = torch.ops.aten.permute.default(_unsafe_view_14, [0, 2, 1, 3]);  _unsafe_view_14 = None
        clone_11 = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
        view_44 = torch.ops.aten.view.default(clone_11, [64, 128, 768]);  clone_11 = None
        permute_30 = torch.ops.aten.permute.default(primals_44, [1, 0]);  primals_44 = None
        view_45 = torch.ops.aten.view.default(view_44, [8192, 768]);  view_44 = None
        addmm_15 = torch.ops.aten.addmm.default(primals_45, view_45, permute_30);  primals_45 = None
        view_46 = torch.ops.aten.view.default(addmm_15, [64, 128, 768]);  addmm_15 = None
        rand_like_8 = torch.ops.aten.rand_like.default(view_46, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_19 = torch.ops.aten.alias.default(rand_like_8);  rand_like_8 = None
        gt_8 = torch.ops.aten.gt.Scalar(alias_19, 0.1);  alias_19 = None
        mul_53 = torch.ops.aten.mul.Tensor(gt_8, view_46);  view_46 = None
        mul_54 = torch.ops.aten.mul.Tensor(mul_53, 1.1111111111111112);  mul_53 = None
        add_31 = torch.ops.aten.add.Tensor(mul_54, convert_element_type_4);  mul_54 = convert_element_type_4 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(add_31, [2], correction = 0, keepdim = True)
        getitem_10 = var_mean_5[0]
        getitem_11 = var_mean_5[1];  var_mean_5 = None
        add_32 = torch.ops.aten.add.Tensor(getitem_10, 1e-12);  getitem_10 = None
        sqrt_5 = torch.ops.aten.sqrt.default(add_32);  add_32 = None
        reciprocal_7 = torch.ops.aten.reciprocal.default(sqrt_5);  sqrt_5 = None
        sub_11 = torch.ops.aten.sub.Tensor(add_31, getitem_11);  add_31 = getitem_11 = None
        mul_55 = torch.ops.aten.mul.Tensor(sub_11, reciprocal_7);  sub_11 = None
        mul_56 = torch.ops.aten.mul.Tensor(mul_55, primals_46)
        add_33 = torch.ops.aten.add.Tensor(mul_56, primals_47);  mul_56 = primals_47 = None
        convert_element_type_5 = torch.ops.prims.convert_element_type.default(add_33, torch.float32);  add_33 = None
        permute_31 = torch.ops.aten.permute.default(primals_48, [1, 0]);  primals_48 = None
        view_47 = torch.ops.aten.view.default(convert_element_type_5, [8192, 768])
        addmm_16 = torch.ops.aten.addmm.default(primals_49, view_47, permute_31);  primals_49 = None
        view_48 = torch.ops.aten.view.default(addmm_16, [64, 128, 3072]);  addmm_16 = None
        mul_57 = torch.ops.aten.mul.Tensor(view_48, 0.5)
        mul_58 = torch.ops.aten.mul.Tensor(view_48, 0.7071067811865476)
        sign_2 = torch.ops.aten.sign.default(mul_58)
        abs_3 = torch.ops.aten.abs.default(mul_58);  mul_58 = None
        mul_59 = torch.ops.aten.mul.Tensor(abs_3, 0.3275911)
        add_34 = torch.ops.aten.add.Tensor(mul_59, 1.0);  mul_59 = None
        reciprocal_8 = torch.ops.aten.reciprocal.default(add_34);  add_34 = None
        mul_60 = torch.ops.aten.mul.Tensor(reciprocal_8, 1.0);  reciprocal_8 = None
        mul_61 = torch.ops.aten.mul.Tensor(mul_60, 1.061405429)
        add_35 = torch.ops.aten.add.Tensor(mul_61, -1.453152027);  mul_61 = None
        mul_62 = torch.ops.aten.mul.Tensor(add_35, mul_60);  add_35 = None
        add_36 = torch.ops.aten.add.Tensor(mul_62, 1.421413741);  mul_62 = None
        mul_63 = torch.ops.aten.mul.Tensor(add_36, mul_60);  add_36 = None
        add_37 = torch.ops.aten.add.Tensor(mul_63, -0.284496736);  mul_63 = None
        mul_64 = torch.ops.aten.mul.Tensor(add_37, mul_60);  add_37 = None
        add_38 = torch.ops.aten.add.Tensor(mul_64, 0.254829592);  mul_64 = None
        mul_65 = torch.ops.aten.mul.Tensor(add_38, mul_60);  add_38 = mul_60 = None
        neg_2 = torch.ops.aten.neg.default(abs_3)
        mul_66 = torch.ops.aten.mul.Tensor(neg_2, abs_3);  neg_2 = abs_3 = None
        exp_5 = torch.ops.aten.exp.default(mul_66);  mul_66 = None
        mul_67 = torch.ops.aten.mul.Tensor(mul_65, exp_5);  mul_65 = exp_5 = None
        _tensor_constant3 = self._tensor_constant3
        lift_fresh_copy_3 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant3);  _tensor_constant3 = None
        sub_12 = torch.ops.aten.sub.Tensor(lift_fresh_copy_3, mul_67);  lift_fresh_copy_3 = None
        mul_68 = torch.ops.aten.mul.Tensor(sign_2, sub_12);  sub_12 = None
        add_39 = torch.ops.aten.add.Tensor(mul_68, 1);  mul_68 = None
        mul_69 = torch.ops.aten.mul.Tensor(mul_57, add_39);  mul_57 = add_39 = None
        permute_32 = torch.ops.aten.permute.default(primals_50, [1, 0]);  primals_50 = None
        view_49 = torch.ops.aten.view.default(mul_69, [8192, 3072]);  mul_69 = None
        addmm_17 = torch.ops.aten.addmm.default(primals_51, view_49, permute_32);  primals_51 = None
        view_50 = torch.ops.aten.view.default(addmm_17, [64, 128, 768]);  addmm_17 = None
        rand_like_9 = torch.ops.aten.rand_like.default(view_50, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_20 = torch.ops.aten.alias.default(rand_like_9);  rand_like_9 = None
        gt_9 = torch.ops.aten.gt.Scalar(alias_20, 0.1);  alias_20 = None
        mul_70 = torch.ops.aten.mul.Tensor(gt_9, view_50);  view_50 = None
        mul_71 = torch.ops.aten.mul.Tensor(mul_70, 1.1111111111111112);  mul_70 = None
        add_40 = torch.ops.aten.add.Tensor(mul_71, convert_element_type_5);  mul_71 = convert_element_type_5 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(add_40, [2], correction = 0, keepdim = True)
        getitem_12 = var_mean_6[0]
        getitem_13 = var_mean_6[1];  var_mean_6 = None
        add_41 = torch.ops.aten.add.Tensor(getitem_12, 1e-12);  getitem_12 = None
        sqrt_6 = torch.ops.aten.sqrt.default(add_41);  add_41 = None
        reciprocal_9 = torch.ops.aten.reciprocal.default(sqrt_6);  sqrt_6 = None
        sub_13 = torch.ops.aten.sub.Tensor(add_40, getitem_13);  add_40 = getitem_13 = None
        mul_72 = torch.ops.aten.mul.Tensor(sub_13, reciprocal_9);  sub_13 = None
        mul_73 = torch.ops.aten.mul.Tensor(mul_72, primals_52)
        add_42 = torch.ops.aten.add.Tensor(mul_73, primals_53);  mul_73 = primals_53 = None
        convert_element_type_6 = torch.ops.prims.convert_element_type.default(add_42, torch.float32);  add_42 = None
        permute_33 = torch.ops.aten.permute.default(primals_54, [1, 0]);  primals_54 = None
        view_51 = torch.ops.aten.view.default(convert_element_type_6, [8192, 768])
        addmm_18 = torch.ops.aten.addmm.default(primals_55, view_51, permute_33);  primals_55 = None
        view_52 = torch.ops.aten.view.default(addmm_18, [64, 128, 768]);  addmm_18 = None
        permute_34 = torch.ops.aten.permute.default(primals_56, [1, 0]);  primals_56 = None
        addmm_19 = torch.ops.aten.addmm.default(primals_57, view_51, permute_34);  primals_57 = None
        view_54 = torch.ops.aten.view.default(addmm_19, [64, 128, 768]);  addmm_19 = None
        view_55 = torch.ops.aten.view.default(view_54, [64, 128, 12, 64]);  view_54 = None
        permute_35 = torch.ops.aten.permute.default(view_55, [0, 2, 1, 3]);  view_55 = None
        permute_36 = torch.ops.aten.permute.default(primals_58, [1, 0]);  primals_58 = None
        addmm_20 = torch.ops.aten.addmm.default(primals_59, view_51, permute_36);  primals_59 = None
        view_57 = torch.ops.aten.view.default(addmm_20, [64, 128, 768]);  addmm_20 = None
        view_58 = torch.ops.aten.view.default(view_57, [64, 128, 12, 64]);  view_57 = None
        permute_37 = torch.ops.aten.permute.default(view_58, [0, 2, 1, 3]);  view_58 = None
        view_59 = torch.ops.aten.view.default(view_52, [64, 128, 12, 64]);  view_52 = None
        permute_38 = torch.ops.aten.permute.default(view_59, [0, 2, 1, 3]);  view_59 = None
        permute_39 = torch.ops.aten.permute.default(permute_35, [0, 1, 3, 2]);  permute_35 = None
        expand_13 = torch.ops.aten.expand.default(permute_38, [64, 12, 128, 64]);  permute_38 = None
        clone_12 = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
        _unsafe_view_15 = torch.ops.aten._unsafe_view.default(clone_12, [768, 128, 64]);  clone_12 = None
        expand_14 = torch.ops.aten.expand.default(permute_39, [64, 12, 64, 128]);  permute_39 = None
        clone_13 = torch.ops.aten.clone.default(expand_14, memory_format = torch.contiguous_format);  expand_14 = None
        _unsafe_view_16 = torch.ops.aten._unsafe_view.default(clone_13, [768, 64, 128]);  clone_13 = None
        bmm_6 = torch.ops.aten.bmm.default(_unsafe_view_15, _unsafe_view_16)
        _unsafe_view_17 = torch.ops.aten._unsafe_view.default(bmm_6, [64, 12, 128, 128]);  bmm_6 = None
        div_6 = torch.ops.aten.div.Tensor(_unsafe_view_17, 8.0);  _unsafe_view_17 = None
        add_43 = torch.ops.aten.add.Tensor(div_6, mul);  div_6 = None
        amax_3 = torch.ops.aten.amax.default(add_43, [-1], True)
        sub_14 = torch.ops.aten.sub.Tensor(add_43, amax_3);  add_43 = amax_3 = None
        exp_6 = torch.ops.aten.exp.default(sub_14);  sub_14 = None
        sum_4 = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_7 = torch.ops.aten.div.Tensor(exp_6, sum_4);  exp_6 = sum_4 = None
        alias_22 = torch.ops.aten.alias.default(div_7)
        alias_23 = torch.ops.aten.alias.default(alias_22);  alias_22 = None
        rand_like_10 = torch.ops.aten.rand_like.default(div_7, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_24 = torch.ops.aten.alias.default(rand_like_10);  rand_like_10 = None
        gt_10 = torch.ops.aten.gt.Scalar(alias_24, 0.1);  alias_24 = None
        mul_74 = torch.ops.aten.mul.Tensor(gt_10, div_7);  div_7 = None
        mul_75 = torch.ops.aten.mul.Tensor(mul_74, 1.1111111111111112);  mul_74 = None
        expand_15 = torch.ops.aten.expand.default(mul_75, [64, 12, 128, 128]);  mul_75 = None
        view_60 = torch.ops.aten.view.default(expand_15, [768, 128, 128]);  expand_15 = None
        expand_16 = torch.ops.aten.expand.default(permute_37, [64, 12, 128, 64]);  permute_37 = None
        clone_14 = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
        _unsafe_view_18 = torch.ops.aten._unsafe_view.default(clone_14, [768, 128, 64]);  clone_14 = None
        bmm_7 = torch.ops.aten.bmm.default(view_60, _unsafe_view_18)
        _unsafe_view_19 = torch.ops.aten._unsafe_view.default(bmm_7, [64, 12, 128, 64]);  bmm_7 = None
        permute_40 = torch.ops.aten.permute.default(_unsafe_view_19, [0, 2, 1, 3]);  _unsafe_view_19 = None
        clone_15 = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
        view_61 = torch.ops.aten.view.default(clone_15, [64, 128, 768]);  clone_15 = None
        permute_41 = torch.ops.aten.permute.default(primals_60, [1, 0]);  primals_60 = None
        view_62 = torch.ops.aten.view.default(view_61, [8192, 768]);  view_61 = None
        addmm_21 = torch.ops.aten.addmm.default(primals_61, view_62, permute_41);  primals_61 = None
        view_63 = torch.ops.aten.view.default(addmm_21, [64, 128, 768]);  addmm_21 = None
        rand_like_11 = torch.ops.aten.rand_like.default(view_63, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_25 = torch.ops.aten.alias.default(rand_like_11);  rand_like_11 = None
        gt_11 = torch.ops.aten.gt.Scalar(alias_25, 0.1);  alias_25 = None
        mul_76 = torch.ops.aten.mul.Tensor(gt_11, view_63);  view_63 = None
        mul_77 = torch.ops.aten.mul.Tensor(mul_76, 1.1111111111111112);  mul_76 = None
        add_44 = torch.ops.aten.add.Tensor(mul_77, convert_element_type_6);  mul_77 = convert_element_type_6 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(add_44, [2], correction = 0, keepdim = True)
        getitem_14 = var_mean_7[0]
        getitem_15 = var_mean_7[1];  var_mean_7 = None
        add_45 = torch.ops.aten.add.Tensor(getitem_14, 1e-12);  getitem_14 = None
        sqrt_7 = torch.ops.aten.sqrt.default(add_45);  add_45 = None
        reciprocal_10 = torch.ops.aten.reciprocal.default(sqrt_7);  sqrt_7 = None
        sub_15 = torch.ops.aten.sub.Tensor(add_44, getitem_15);  add_44 = getitem_15 = None
        mul_78 = torch.ops.aten.mul.Tensor(sub_15, reciprocal_10);  sub_15 = None
        mul_79 = torch.ops.aten.mul.Tensor(mul_78, primals_62)
        add_46 = torch.ops.aten.add.Tensor(mul_79, primals_63);  mul_79 = primals_63 = None
        convert_element_type_7 = torch.ops.prims.convert_element_type.default(add_46, torch.float32);  add_46 = None
        permute_42 = torch.ops.aten.permute.default(primals_64, [1, 0]);  primals_64 = None
        view_64 = torch.ops.aten.view.default(convert_element_type_7, [8192, 768])
        addmm_22 = torch.ops.aten.addmm.default(primals_65, view_64, permute_42);  primals_65 = None
        view_65 = torch.ops.aten.view.default(addmm_22, [64, 128, 3072]);  addmm_22 = None
        mul_80 = torch.ops.aten.mul.Tensor(view_65, 0.5)
        mul_81 = torch.ops.aten.mul.Tensor(view_65, 0.7071067811865476)
        sign_3 = torch.ops.aten.sign.default(mul_81)
        abs_4 = torch.ops.aten.abs.default(mul_81);  mul_81 = None
        mul_82 = torch.ops.aten.mul.Tensor(abs_4, 0.3275911)
        add_47 = torch.ops.aten.add.Tensor(mul_82, 1.0);  mul_82 = None
        reciprocal_11 = torch.ops.aten.reciprocal.default(add_47);  add_47 = None
        mul_83 = torch.ops.aten.mul.Tensor(reciprocal_11, 1.0);  reciprocal_11 = None
        mul_84 = torch.ops.aten.mul.Tensor(mul_83, 1.061405429)
        add_48 = torch.ops.aten.add.Tensor(mul_84, -1.453152027);  mul_84 = None
        mul_85 = torch.ops.aten.mul.Tensor(add_48, mul_83);  add_48 = None
        add_49 = torch.ops.aten.add.Tensor(mul_85, 1.421413741);  mul_85 = None
        mul_86 = torch.ops.aten.mul.Tensor(add_49, mul_83);  add_49 = None
        add_50 = torch.ops.aten.add.Tensor(mul_86, -0.284496736);  mul_86 = None
        mul_87 = torch.ops.aten.mul.Tensor(add_50, mul_83);  add_50 = None
        add_51 = torch.ops.aten.add.Tensor(mul_87, 0.254829592);  mul_87 = None
        mul_88 = torch.ops.aten.mul.Tensor(add_51, mul_83);  add_51 = mul_83 = None
        neg_3 = torch.ops.aten.neg.default(abs_4)
        mul_89 = torch.ops.aten.mul.Tensor(neg_3, abs_4);  neg_3 = abs_4 = None
        exp_7 = torch.ops.aten.exp.default(mul_89);  mul_89 = None
        mul_90 = torch.ops.aten.mul.Tensor(mul_88, exp_7);  mul_88 = exp_7 = None
        _tensor_constant4 = self._tensor_constant4
        lift_fresh_copy_4 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant4);  _tensor_constant4 = None
        sub_16 = torch.ops.aten.sub.Tensor(lift_fresh_copy_4, mul_90);  lift_fresh_copy_4 = None
        mul_91 = torch.ops.aten.mul.Tensor(sign_3, sub_16);  sub_16 = None
        add_52 = torch.ops.aten.add.Tensor(mul_91, 1);  mul_91 = None
        mul_92 = torch.ops.aten.mul.Tensor(mul_80, add_52);  mul_80 = add_52 = None
        permute_43 = torch.ops.aten.permute.default(primals_66, [1, 0]);  primals_66 = None
        view_66 = torch.ops.aten.view.default(mul_92, [8192, 3072]);  mul_92 = None
        addmm_23 = torch.ops.aten.addmm.default(primals_67, view_66, permute_43);  primals_67 = None
        view_67 = torch.ops.aten.view.default(addmm_23, [64, 128, 768]);  addmm_23 = None
        rand_like_12 = torch.ops.aten.rand_like.default(view_67, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_26 = torch.ops.aten.alias.default(rand_like_12);  rand_like_12 = None
        gt_12 = torch.ops.aten.gt.Scalar(alias_26, 0.1);  alias_26 = None
        mul_93 = torch.ops.aten.mul.Tensor(gt_12, view_67);  view_67 = None
        mul_94 = torch.ops.aten.mul.Tensor(mul_93, 1.1111111111111112);  mul_93 = None
        add_53 = torch.ops.aten.add.Tensor(mul_94, convert_element_type_7);  mul_94 = convert_element_type_7 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(add_53, [2], correction = 0, keepdim = True)
        getitem_16 = var_mean_8[0]
        getitem_17 = var_mean_8[1];  var_mean_8 = None
        add_54 = torch.ops.aten.add.Tensor(getitem_16, 1e-12);  getitem_16 = None
        sqrt_8 = torch.ops.aten.sqrt.default(add_54);  add_54 = None
        reciprocal_12 = torch.ops.aten.reciprocal.default(sqrt_8);  sqrt_8 = None
        sub_17 = torch.ops.aten.sub.Tensor(add_53, getitem_17);  add_53 = getitem_17 = None
        mul_95 = torch.ops.aten.mul.Tensor(sub_17, reciprocal_12);  sub_17 = None
        mul_96 = torch.ops.aten.mul.Tensor(mul_95, primals_68)
        add_55 = torch.ops.aten.add.Tensor(mul_96, primals_69);  mul_96 = primals_69 = None
        convert_element_type_8 = torch.ops.prims.convert_element_type.default(add_55, torch.float32);  add_55 = None
        permute_44 = torch.ops.aten.permute.default(primals_70, [1, 0]);  primals_70 = None
        view_68 = torch.ops.aten.view.default(convert_element_type_8, [8192, 768])
        addmm_24 = torch.ops.aten.addmm.default(primals_71, view_68, permute_44);  primals_71 = None
        view_69 = torch.ops.aten.view.default(addmm_24, [64, 128, 768]);  addmm_24 = None
        permute_45 = torch.ops.aten.permute.default(primals_72, [1, 0]);  primals_72 = None
        addmm_25 = torch.ops.aten.addmm.default(primals_73, view_68, permute_45);  primals_73 = None
        view_71 = torch.ops.aten.view.default(addmm_25, [64, 128, 768]);  addmm_25 = None
        view_72 = torch.ops.aten.view.default(view_71, [64, 128, 12, 64]);  view_71 = None
        permute_46 = torch.ops.aten.permute.default(view_72, [0, 2, 1, 3]);  view_72 = None
        permute_47 = torch.ops.aten.permute.default(primals_74, [1, 0]);  primals_74 = None
        addmm_26 = torch.ops.aten.addmm.default(primals_75, view_68, permute_47);  primals_75 = None
        view_74 = torch.ops.aten.view.default(addmm_26, [64, 128, 768]);  addmm_26 = None
        view_75 = torch.ops.aten.view.default(view_74, [64, 128, 12, 64]);  view_74 = None
        permute_48 = torch.ops.aten.permute.default(view_75, [0, 2, 1, 3]);  view_75 = None
        view_76 = torch.ops.aten.view.default(view_69, [64, 128, 12, 64]);  view_69 = None
        permute_49 = torch.ops.aten.permute.default(view_76, [0, 2, 1, 3]);  view_76 = None
        permute_50 = torch.ops.aten.permute.default(permute_46, [0, 1, 3, 2]);  permute_46 = None
        expand_17 = torch.ops.aten.expand.default(permute_49, [64, 12, 128, 64]);  permute_49 = None
        clone_16 = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
        _unsafe_view_20 = torch.ops.aten._unsafe_view.default(clone_16, [768, 128, 64]);  clone_16 = None
        expand_18 = torch.ops.aten.expand.default(permute_50, [64, 12, 64, 128]);  permute_50 = None
        clone_17 = torch.ops.aten.clone.default(expand_18, memory_format = torch.contiguous_format);  expand_18 = None
        _unsafe_view_21 = torch.ops.aten._unsafe_view.default(clone_17, [768, 64, 128]);  clone_17 = None
        bmm_8 = torch.ops.aten.bmm.default(_unsafe_view_20, _unsafe_view_21)
        _unsafe_view_22 = torch.ops.aten._unsafe_view.default(bmm_8, [64, 12, 128, 128]);  bmm_8 = None
        div_8 = torch.ops.aten.div.Tensor(_unsafe_view_22, 8.0);  _unsafe_view_22 = None
        add_56 = torch.ops.aten.add.Tensor(div_8, mul);  div_8 = None
        amax_4 = torch.ops.aten.amax.default(add_56, [-1], True)
        sub_18 = torch.ops.aten.sub.Tensor(add_56, amax_4);  add_56 = amax_4 = None
        exp_8 = torch.ops.aten.exp.default(sub_18);  sub_18 = None
        sum_5 = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_9 = torch.ops.aten.div.Tensor(exp_8, sum_5);  exp_8 = sum_5 = None
        alias_28 = torch.ops.aten.alias.default(div_9)
        alias_29 = torch.ops.aten.alias.default(alias_28);  alias_28 = None
        rand_like_13 = torch.ops.aten.rand_like.default(div_9, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_30 = torch.ops.aten.alias.default(rand_like_13);  rand_like_13 = None
        gt_13 = torch.ops.aten.gt.Scalar(alias_30, 0.1);  alias_30 = None
        mul_97 = torch.ops.aten.mul.Tensor(gt_13, div_9);  div_9 = None
        mul_98 = torch.ops.aten.mul.Tensor(mul_97, 1.1111111111111112);  mul_97 = None
        expand_19 = torch.ops.aten.expand.default(mul_98, [64, 12, 128, 128]);  mul_98 = None
        view_77 = torch.ops.aten.view.default(expand_19, [768, 128, 128]);  expand_19 = None
        expand_20 = torch.ops.aten.expand.default(permute_48, [64, 12, 128, 64]);  permute_48 = None
        clone_18 = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
        _unsafe_view_23 = torch.ops.aten._unsafe_view.default(clone_18, [768, 128, 64]);  clone_18 = None
        bmm_9 = torch.ops.aten.bmm.default(view_77, _unsafe_view_23)
        _unsafe_view_24 = torch.ops.aten._unsafe_view.default(bmm_9, [64, 12, 128, 64]);  bmm_9 = None
        permute_51 = torch.ops.aten.permute.default(_unsafe_view_24, [0, 2, 1, 3]);  _unsafe_view_24 = None
        clone_19 = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
        view_78 = torch.ops.aten.view.default(clone_19, [64, 128, 768]);  clone_19 = None
        permute_52 = torch.ops.aten.permute.default(primals_76, [1, 0]);  primals_76 = None
        view_79 = torch.ops.aten.view.default(view_78, [8192, 768]);  view_78 = None
        addmm_27 = torch.ops.aten.addmm.default(primals_77, view_79, permute_52);  primals_77 = None
        view_80 = torch.ops.aten.view.default(addmm_27, [64, 128, 768]);  addmm_27 = None
        rand_like_14 = torch.ops.aten.rand_like.default(view_80, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_31 = torch.ops.aten.alias.default(rand_like_14);  rand_like_14 = None
        gt_14 = torch.ops.aten.gt.Scalar(alias_31, 0.1);  alias_31 = None
        mul_99 = torch.ops.aten.mul.Tensor(gt_14, view_80);  view_80 = None
        mul_100 = torch.ops.aten.mul.Tensor(mul_99, 1.1111111111111112);  mul_99 = None
        add_57 = torch.ops.aten.add.Tensor(mul_100, convert_element_type_8);  mul_100 = convert_element_type_8 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(add_57, [2], correction = 0, keepdim = True)
        getitem_18 = var_mean_9[0]
        getitem_19 = var_mean_9[1];  var_mean_9 = None
        add_58 = torch.ops.aten.add.Tensor(getitem_18, 1e-12);  getitem_18 = None
        sqrt_9 = torch.ops.aten.sqrt.default(add_58);  add_58 = None
        reciprocal_13 = torch.ops.aten.reciprocal.default(sqrt_9);  sqrt_9 = None
        sub_19 = torch.ops.aten.sub.Tensor(add_57, getitem_19);  add_57 = getitem_19 = None
        mul_101 = torch.ops.aten.mul.Tensor(sub_19, reciprocal_13);  sub_19 = None
        mul_102 = torch.ops.aten.mul.Tensor(mul_101, primals_78)
        add_59 = torch.ops.aten.add.Tensor(mul_102, primals_79);  mul_102 = primals_79 = None
        convert_element_type_9 = torch.ops.prims.convert_element_type.default(add_59, torch.float32);  add_59 = None
        permute_53 = torch.ops.aten.permute.default(primals_80, [1, 0]);  primals_80 = None
        view_81 = torch.ops.aten.view.default(convert_element_type_9, [8192, 768])
        addmm_28 = torch.ops.aten.addmm.default(primals_81, view_81, permute_53);  primals_81 = None
        view_82 = torch.ops.aten.view.default(addmm_28, [64, 128, 3072]);  addmm_28 = None
        mul_103 = torch.ops.aten.mul.Tensor(view_82, 0.5)
        mul_104 = torch.ops.aten.mul.Tensor(view_82, 0.7071067811865476)
        sign_4 = torch.ops.aten.sign.default(mul_104)
        abs_5 = torch.ops.aten.abs.default(mul_104);  mul_104 = None
        mul_105 = torch.ops.aten.mul.Tensor(abs_5, 0.3275911)
        add_60 = torch.ops.aten.add.Tensor(mul_105, 1.0);  mul_105 = None
        reciprocal_14 = torch.ops.aten.reciprocal.default(add_60);  add_60 = None
        mul_106 = torch.ops.aten.mul.Tensor(reciprocal_14, 1.0);  reciprocal_14 = None
        mul_107 = torch.ops.aten.mul.Tensor(mul_106, 1.061405429)
        add_61 = torch.ops.aten.add.Tensor(mul_107, -1.453152027);  mul_107 = None
        mul_108 = torch.ops.aten.mul.Tensor(add_61, mul_106);  add_61 = None
        add_62 = torch.ops.aten.add.Tensor(mul_108, 1.421413741);  mul_108 = None
        mul_109 = torch.ops.aten.mul.Tensor(add_62, mul_106);  add_62 = None
        add_63 = torch.ops.aten.add.Tensor(mul_109, -0.284496736);  mul_109 = None
        mul_110 = torch.ops.aten.mul.Tensor(add_63, mul_106);  add_63 = None
        add_64 = torch.ops.aten.add.Tensor(mul_110, 0.254829592);  mul_110 = None
        mul_111 = torch.ops.aten.mul.Tensor(add_64, mul_106);  add_64 = mul_106 = None
        neg_4 = torch.ops.aten.neg.default(abs_5)
        mul_112 = torch.ops.aten.mul.Tensor(neg_4, abs_5);  neg_4 = abs_5 = None
        exp_9 = torch.ops.aten.exp.default(mul_112);  mul_112 = None
        mul_113 = torch.ops.aten.mul.Tensor(mul_111, exp_9);  mul_111 = exp_9 = None
        _tensor_constant5 = self._tensor_constant5
        lift_fresh_copy_5 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant5);  _tensor_constant5 = None
        sub_20 = torch.ops.aten.sub.Tensor(lift_fresh_copy_5, mul_113);  lift_fresh_copy_5 = None
        mul_114 = torch.ops.aten.mul.Tensor(sign_4, sub_20);  sub_20 = None
        add_65 = torch.ops.aten.add.Tensor(mul_114, 1);  mul_114 = None
        mul_115 = torch.ops.aten.mul.Tensor(mul_103, add_65);  mul_103 = add_65 = None
        permute_54 = torch.ops.aten.permute.default(primals_82, [1, 0]);  primals_82 = None
        view_83 = torch.ops.aten.view.default(mul_115, [8192, 3072]);  mul_115 = None
        addmm_29 = torch.ops.aten.addmm.default(primals_83, view_83, permute_54);  primals_83 = None
        view_84 = torch.ops.aten.view.default(addmm_29, [64, 128, 768]);  addmm_29 = None
        rand_like_15 = torch.ops.aten.rand_like.default(view_84, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_32 = torch.ops.aten.alias.default(rand_like_15);  rand_like_15 = None
        gt_15 = torch.ops.aten.gt.Scalar(alias_32, 0.1);  alias_32 = None
        mul_116 = torch.ops.aten.mul.Tensor(gt_15, view_84);  view_84 = None
        mul_117 = torch.ops.aten.mul.Tensor(mul_116, 1.1111111111111112);  mul_116 = None
        add_66 = torch.ops.aten.add.Tensor(mul_117, convert_element_type_9);  mul_117 = convert_element_type_9 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(add_66, [2], correction = 0, keepdim = True)
        getitem_20 = var_mean_10[0]
        getitem_21 = var_mean_10[1];  var_mean_10 = None
        add_67 = torch.ops.aten.add.Tensor(getitem_20, 1e-12);  getitem_20 = None
        sqrt_10 = torch.ops.aten.sqrt.default(add_67);  add_67 = None
        reciprocal_15 = torch.ops.aten.reciprocal.default(sqrt_10);  sqrt_10 = None
        sub_21 = torch.ops.aten.sub.Tensor(add_66, getitem_21);  add_66 = getitem_21 = None
        mul_118 = torch.ops.aten.mul.Tensor(sub_21, reciprocal_15);  sub_21 = None
        mul_119 = torch.ops.aten.mul.Tensor(mul_118, primals_84)
        add_68 = torch.ops.aten.add.Tensor(mul_119, primals_85);  mul_119 = primals_85 = None
        convert_element_type_10 = torch.ops.prims.convert_element_type.default(add_68, torch.float32);  add_68 = None
        permute_55 = torch.ops.aten.permute.default(primals_86, [1, 0]);  primals_86 = None
        view_85 = torch.ops.aten.view.default(convert_element_type_10, [8192, 768])
        addmm_30 = torch.ops.aten.addmm.default(primals_87, view_85, permute_55);  primals_87 = None
        view_86 = torch.ops.aten.view.default(addmm_30, [64, 128, 768]);  addmm_30 = None
        permute_56 = torch.ops.aten.permute.default(primals_88, [1, 0]);  primals_88 = None
        addmm_31 = torch.ops.aten.addmm.default(primals_89, view_85, permute_56);  primals_89 = None
        view_88 = torch.ops.aten.view.default(addmm_31, [64, 128, 768]);  addmm_31 = None
        view_89 = torch.ops.aten.view.default(view_88, [64, 128, 12, 64]);  view_88 = None
        permute_57 = torch.ops.aten.permute.default(view_89, [0, 2, 1, 3]);  view_89 = None
        permute_58 = torch.ops.aten.permute.default(primals_90, [1, 0]);  primals_90 = None
        addmm_32 = torch.ops.aten.addmm.default(primals_91, view_85, permute_58);  primals_91 = None
        view_91 = torch.ops.aten.view.default(addmm_32, [64, 128, 768]);  addmm_32 = None
        view_92 = torch.ops.aten.view.default(view_91, [64, 128, 12, 64]);  view_91 = None
        permute_59 = torch.ops.aten.permute.default(view_92, [0, 2, 1, 3]);  view_92 = None
        view_93 = torch.ops.aten.view.default(view_86, [64, 128, 12, 64]);  view_86 = None
        permute_60 = torch.ops.aten.permute.default(view_93, [0, 2, 1, 3]);  view_93 = None
        permute_61 = torch.ops.aten.permute.default(permute_57, [0, 1, 3, 2]);  permute_57 = None
        expand_21 = torch.ops.aten.expand.default(permute_60, [64, 12, 128, 64]);  permute_60 = None
        clone_20 = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
        _unsafe_view_25 = torch.ops.aten._unsafe_view.default(clone_20, [768, 128, 64]);  clone_20 = None
        expand_22 = torch.ops.aten.expand.default(permute_61, [64, 12, 64, 128]);  permute_61 = None
        clone_21 = torch.ops.aten.clone.default(expand_22, memory_format = torch.contiguous_format);  expand_22 = None
        _unsafe_view_26 = torch.ops.aten._unsafe_view.default(clone_21, [768, 64, 128]);  clone_21 = None
        bmm_10 = torch.ops.aten.bmm.default(_unsafe_view_25, _unsafe_view_26)
        _unsafe_view_27 = torch.ops.aten._unsafe_view.default(bmm_10, [64, 12, 128, 128]);  bmm_10 = None
        div_10 = torch.ops.aten.div.Tensor(_unsafe_view_27, 8.0);  _unsafe_view_27 = None
        add_69 = torch.ops.aten.add.Tensor(div_10, mul);  div_10 = None
        amax_5 = torch.ops.aten.amax.default(add_69, [-1], True)
        sub_22 = torch.ops.aten.sub.Tensor(add_69, amax_5);  add_69 = amax_5 = None
        exp_10 = torch.ops.aten.exp.default(sub_22);  sub_22 = None
        sum_6 = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_11 = torch.ops.aten.div.Tensor(exp_10, sum_6);  exp_10 = sum_6 = None
        alias_34 = torch.ops.aten.alias.default(div_11)
        alias_35 = torch.ops.aten.alias.default(alias_34);  alias_34 = None
        rand_like_16 = torch.ops.aten.rand_like.default(div_11, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_36 = torch.ops.aten.alias.default(rand_like_16);  rand_like_16 = None
        gt_16 = torch.ops.aten.gt.Scalar(alias_36, 0.1);  alias_36 = None
        mul_120 = torch.ops.aten.mul.Tensor(gt_16, div_11);  div_11 = None
        mul_121 = torch.ops.aten.mul.Tensor(mul_120, 1.1111111111111112);  mul_120 = None
        expand_23 = torch.ops.aten.expand.default(mul_121, [64, 12, 128, 128]);  mul_121 = None
        view_94 = torch.ops.aten.view.default(expand_23, [768, 128, 128]);  expand_23 = None
        expand_24 = torch.ops.aten.expand.default(permute_59, [64, 12, 128, 64]);  permute_59 = None
        clone_22 = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
        _unsafe_view_28 = torch.ops.aten._unsafe_view.default(clone_22, [768, 128, 64]);  clone_22 = None
        bmm_11 = torch.ops.aten.bmm.default(view_94, _unsafe_view_28)
        _unsafe_view_29 = torch.ops.aten._unsafe_view.default(bmm_11, [64, 12, 128, 64]);  bmm_11 = None
        permute_62 = torch.ops.aten.permute.default(_unsafe_view_29, [0, 2, 1, 3]);  _unsafe_view_29 = None
        clone_23 = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
        view_95 = torch.ops.aten.view.default(clone_23, [64, 128, 768]);  clone_23 = None
        permute_63 = torch.ops.aten.permute.default(primals_92, [1, 0]);  primals_92 = None
        view_96 = torch.ops.aten.view.default(view_95, [8192, 768]);  view_95 = None
        addmm_33 = torch.ops.aten.addmm.default(primals_93, view_96, permute_63);  primals_93 = None
        view_97 = torch.ops.aten.view.default(addmm_33, [64, 128, 768]);  addmm_33 = None
        rand_like_17 = torch.ops.aten.rand_like.default(view_97, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_37 = torch.ops.aten.alias.default(rand_like_17);  rand_like_17 = None
        gt_17 = torch.ops.aten.gt.Scalar(alias_37, 0.1);  alias_37 = None
        mul_122 = torch.ops.aten.mul.Tensor(gt_17, view_97);  view_97 = None
        mul_123 = torch.ops.aten.mul.Tensor(mul_122, 1.1111111111111112);  mul_122 = None
        add_70 = torch.ops.aten.add.Tensor(mul_123, convert_element_type_10);  mul_123 = convert_element_type_10 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(add_70, [2], correction = 0, keepdim = True)
        getitem_22 = var_mean_11[0]
        getitem_23 = var_mean_11[1];  var_mean_11 = None
        add_71 = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
        sqrt_11 = torch.ops.aten.sqrt.default(add_71);  add_71 = None
        reciprocal_16 = torch.ops.aten.reciprocal.default(sqrt_11);  sqrt_11 = None
        sub_23 = torch.ops.aten.sub.Tensor(add_70, getitem_23);  add_70 = getitem_23 = None
        mul_124 = torch.ops.aten.mul.Tensor(sub_23, reciprocal_16);  sub_23 = None
        mul_125 = torch.ops.aten.mul.Tensor(mul_124, primals_94)
        add_72 = torch.ops.aten.add.Tensor(mul_125, primals_95);  mul_125 = primals_95 = None
        convert_element_type_11 = torch.ops.prims.convert_element_type.default(add_72, torch.float32);  add_72 = None
        permute_64 = torch.ops.aten.permute.default(primals_96, [1, 0]);  primals_96 = None
        view_98 = torch.ops.aten.view.default(convert_element_type_11, [8192, 768])
        addmm_34 = torch.ops.aten.addmm.default(primals_97, view_98, permute_64);  primals_97 = None
        view_99 = torch.ops.aten.view.default(addmm_34, [64, 128, 3072]);  addmm_34 = None
        mul_126 = torch.ops.aten.mul.Tensor(view_99, 0.5)
        mul_127 = torch.ops.aten.mul.Tensor(view_99, 0.7071067811865476)
        sign_5 = torch.ops.aten.sign.default(mul_127)
        abs_6 = torch.ops.aten.abs.default(mul_127);  mul_127 = None
        mul_128 = torch.ops.aten.mul.Tensor(abs_6, 0.3275911)
        add_73 = torch.ops.aten.add.Tensor(mul_128, 1.0);  mul_128 = None
        reciprocal_17 = torch.ops.aten.reciprocal.default(add_73);  add_73 = None
        mul_129 = torch.ops.aten.mul.Tensor(reciprocal_17, 1.0);  reciprocal_17 = None
        mul_130 = torch.ops.aten.mul.Tensor(mul_129, 1.061405429)
        add_74 = torch.ops.aten.add.Tensor(mul_130, -1.453152027);  mul_130 = None
        mul_131 = torch.ops.aten.mul.Tensor(add_74, mul_129);  add_74 = None
        add_75 = torch.ops.aten.add.Tensor(mul_131, 1.421413741);  mul_131 = None
        mul_132 = torch.ops.aten.mul.Tensor(add_75, mul_129);  add_75 = None
        add_76 = torch.ops.aten.add.Tensor(mul_132, -0.284496736);  mul_132 = None
        mul_133 = torch.ops.aten.mul.Tensor(add_76, mul_129);  add_76 = None
        add_77 = torch.ops.aten.add.Tensor(mul_133, 0.254829592);  mul_133 = None
        mul_134 = torch.ops.aten.mul.Tensor(add_77, mul_129);  add_77 = mul_129 = None
        neg_5 = torch.ops.aten.neg.default(abs_6)
        mul_135 = torch.ops.aten.mul.Tensor(neg_5, abs_6);  neg_5 = abs_6 = None
        exp_11 = torch.ops.aten.exp.default(mul_135);  mul_135 = None
        mul_136 = torch.ops.aten.mul.Tensor(mul_134, exp_11);  mul_134 = exp_11 = None
        _tensor_constant6 = self._tensor_constant6
        lift_fresh_copy_6 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant6);  _tensor_constant6 = None
        sub_24 = torch.ops.aten.sub.Tensor(lift_fresh_copy_6, mul_136);  lift_fresh_copy_6 = None
        mul_137 = torch.ops.aten.mul.Tensor(sign_5, sub_24);  sub_24 = None
        add_78 = torch.ops.aten.add.Tensor(mul_137, 1);  mul_137 = None
        mul_138 = torch.ops.aten.mul.Tensor(mul_126, add_78);  mul_126 = add_78 = None
        permute_65 = torch.ops.aten.permute.default(primals_98, [1, 0]);  primals_98 = None
        view_100 = torch.ops.aten.view.default(mul_138, [8192, 3072]);  mul_138 = None
        addmm_35 = torch.ops.aten.addmm.default(primals_99, view_100, permute_65);  primals_99 = None
        view_101 = torch.ops.aten.view.default(addmm_35, [64, 128, 768]);  addmm_35 = None
        rand_like_18 = torch.ops.aten.rand_like.default(view_101, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_38 = torch.ops.aten.alias.default(rand_like_18);  rand_like_18 = None
        gt_18 = torch.ops.aten.gt.Scalar(alias_38, 0.1);  alias_38 = None
        mul_139 = torch.ops.aten.mul.Tensor(gt_18, view_101);  view_101 = None
        mul_140 = torch.ops.aten.mul.Tensor(mul_139, 1.1111111111111112);  mul_139 = None
        add_79 = torch.ops.aten.add.Tensor(mul_140, convert_element_type_11);  mul_140 = convert_element_type_11 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(add_79, [2], correction = 0, keepdim = True)
        getitem_24 = var_mean_12[0]
        getitem_25 = var_mean_12[1];  var_mean_12 = None
        add_80 = torch.ops.aten.add.Tensor(getitem_24, 1e-12);  getitem_24 = None
        sqrt_12 = torch.ops.aten.sqrt.default(add_80);  add_80 = None
        reciprocal_18 = torch.ops.aten.reciprocal.default(sqrt_12);  sqrt_12 = None
        sub_25 = torch.ops.aten.sub.Tensor(add_79, getitem_25);  add_79 = getitem_25 = None
        mul_141 = torch.ops.aten.mul.Tensor(sub_25, reciprocal_18);  sub_25 = None
        mul_142 = torch.ops.aten.mul.Tensor(mul_141, primals_100)
        add_81 = torch.ops.aten.add.Tensor(mul_142, primals_101);  mul_142 = primals_101 = None
        convert_element_type_12 = torch.ops.prims.convert_element_type.default(add_81, torch.float32);  add_81 = None
        permute_66 = torch.ops.aten.permute.default(primals_102, [1, 0]);  primals_102 = None
        view_102 = torch.ops.aten.view.default(convert_element_type_12, [8192, 768])
        addmm_36 = torch.ops.aten.addmm.default(primals_103, view_102, permute_66);  primals_103 = None
        view_103 = torch.ops.aten.view.default(addmm_36, [64, 128, 768]);  addmm_36 = None
        permute_67 = torch.ops.aten.permute.default(primals_104, [1, 0]);  primals_104 = None
        addmm_37 = torch.ops.aten.addmm.default(primals_105, view_102, permute_67);  primals_105 = None
        view_105 = torch.ops.aten.view.default(addmm_37, [64, 128, 768]);  addmm_37 = None
        view_106 = torch.ops.aten.view.default(view_105, [64, 128, 12, 64]);  view_105 = None
        permute_68 = torch.ops.aten.permute.default(view_106, [0, 2, 1, 3]);  view_106 = None
        permute_69 = torch.ops.aten.permute.default(primals_106, [1, 0]);  primals_106 = None
        addmm_38 = torch.ops.aten.addmm.default(primals_107, view_102, permute_69);  primals_107 = None
        view_108 = torch.ops.aten.view.default(addmm_38, [64, 128, 768]);  addmm_38 = None
        view_109 = torch.ops.aten.view.default(view_108, [64, 128, 12, 64]);  view_108 = None
        permute_70 = torch.ops.aten.permute.default(view_109, [0, 2, 1, 3]);  view_109 = None
        view_110 = torch.ops.aten.view.default(view_103, [64, 128, 12, 64]);  view_103 = None
        permute_71 = torch.ops.aten.permute.default(view_110, [0, 2, 1, 3]);  view_110 = None
        permute_72 = torch.ops.aten.permute.default(permute_68, [0, 1, 3, 2]);  permute_68 = None
        expand_25 = torch.ops.aten.expand.default(permute_71, [64, 12, 128, 64]);  permute_71 = None
        clone_24 = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
        _unsafe_view_30 = torch.ops.aten._unsafe_view.default(clone_24, [768, 128, 64]);  clone_24 = None
        expand_26 = torch.ops.aten.expand.default(permute_72, [64, 12, 64, 128]);  permute_72 = None
        clone_25 = torch.ops.aten.clone.default(expand_26, memory_format = torch.contiguous_format);  expand_26 = None
        _unsafe_view_31 = torch.ops.aten._unsafe_view.default(clone_25, [768, 64, 128]);  clone_25 = None
        bmm_12 = torch.ops.aten.bmm.default(_unsafe_view_30, _unsafe_view_31)
        _unsafe_view_32 = torch.ops.aten._unsafe_view.default(bmm_12, [64, 12, 128, 128]);  bmm_12 = None
        div_12 = torch.ops.aten.div.Tensor(_unsafe_view_32, 8.0);  _unsafe_view_32 = None
        add_82 = torch.ops.aten.add.Tensor(div_12, mul);  div_12 = None
        amax_6 = torch.ops.aten.amax.default(add_82, [-1], True)
        sub_26 = torch.ops.aten.sub.Tensor(add_82, amax_6);  add_82 = amax_6 = None
        exp_12 = torch.ops.aten.exp.default(sub_26);  sub_26 = None
        sum_7 = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
        div_13 = torch.ops.aten.div.Tensor(exp_12, sum_7);  exp_12 = sum_7 = None
        alias_40 = torch.ops.aten.alias.default(div_13)
        alias_41 = torch.ops.aten.alias.default(alias_40);  alias_40 = None
        rand_like_19 = torch.ops.aten.rand_like.default(div_13, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_42 = torch.ops.aten.alias.default(rand_like_19);  rand_like_19 = None
        gt_19 = torch.ops.aten.gt.Scalar(alias_42, 0.1);  alias_42 = None
        mul_143 = torch.ops.aten.mul.Tensor(gt_19, div_13);  div_13 = None
        mul_144 = torch.ops.aten.mul.Tensor(mul_143, 1.1111111111111112);  mul_143 = None
        expand_27 = torch.ops.aten.expand.default(mul_144, [64, 12, 128, 128]);  mul_144 = None
        view_111 = torch.ops.aten.view.default(expand_27, [768, 128, 128]);  expand_27 = None
        expand_28 = torch.ops.aten.expand.default(permute_70, [64, 12, 128, 64]);  permute_70 = None
        clone_26 = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
        _unsafe_view_33 = torch.ops.aten._unsafe_view.default(clone_26, [768, 128, 64]);  clone_26 = None
        bmm_13 = torch.ops.aten.bmm.default(view_111, _unsafe_view_33)
        _unsafe_view_34 = torch.ops.aten._unsafe_view.default(bmm_13, [64, 12, 128, 64]);  bmm_13 = None
        permute_73 = torch.ops.aten.permute.default(_unsafe_view_34, [0, 2, 1, 3]);  _unsafe_view_34 = None
        clone_27 = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
        view_112 = torch.ops.aten.view.default(clone_27, [64, 128, 768]);  clone_27 = None
        permute_74 = torch.ops.aten.permute.default(primals_108, [1, 0]);  primals_108 = None
        view_113 = torch.ops.aten.view.default(view_112, [8192, 768]);  view_112 = None
        addmm_39 = torch.ops.aten.addmm.default(primals_109, view_113, permute_74);  primals_109 = None
        view_114 = torch.ops.aten.view.default(addmm_39, [64, 128, 768]);  addmm_39 = None
        rand_like_20 = torch.ops.aten.rand_like.default(view_114, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_43 = torch.ops.aten.alias.default(rand_like_20);  rand_like_20 = None
        gt_20 = torch.ops.aten.gt.Scalar(alias_43, 0.1);  alias_43 = None
        mul_145 = torch.ops.aten.mul.Tensor(gt_20, view_114);  view_114 = None
        mul_146 = torch.ops.aten.mul.Tensor(mul_145, 1.1111111111111112);  mul_145 = None
        add_83 = torch.ops.aten.add.Tensor(mul_146, convert_element_type_12);  mul_146 = convert_element_type_12 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(add_83, [2], correction = 0, keepdim = True)
        getitem_26 = var_mean_13[0]
        getitem_27 = var_mean_13[1];  var_mean_13 = None
        add_84 = torch.ops.aten.add.Tensor(getitem_26, 1e-12);  getitem_26 = None
        sqrt_13 = torch.ops.aten.sqrt.default(add_84);  add_84 = None
        reciprocal_19 = torch.ops.aten.reciprocal.default(sqrt_13);  sqrt_13 = None
        sub_27 = torch.ops.aten.sub.Tensor(add_83, getitem_27);  add_83 = getitem_27 = None
        mul_147 = torch.ops.aten.mul.Tensor(sub_27, reciprocal_19);  sub_27 = None
        mul_148 = torch.ops.aten.mul.Tensor(mul_147, primals_110)
        add_85 = torch.ops.aten.add.Tensor(mul_148, primals_111);  mul_148 = primals_111 = None
        convert_element_type_13 = torch.ops.prims.convert_element_type.default(add_85, torch.float32);  add_85 = None
        permute_75 = torch.ops.aten.permute.default(primals_112, [1, 0]);  primals_112 = None
        view_115 = torch.ops.aten.view.default(convert_element_type_13, [8192, 768])
        addmm_40 = torch.ops.aten.addmm.default(primals_113, view_115, permute_75);  primals_113 = None
        view_116 = torch.ops.aten.view.default(addmm_40, [64, 128, 3072]);  addmm_40 = None
        mul_149 = torch.ops.aten.mul.Tensor(view_116, 0.5)
        mul_150 = torch.ops.aten.mul.Tensor(view_116, 0.7071067811865476)
        sign_6 = torch.ops.aten.sign.default(mul_150)
        abs_7 = torch.ops.aten.abs.default(mul_150);  mul_150 = None
        mul_151 = torch.ops.aten.mul.Tensor(abs_7, 0.3275911)
        add_86 = torch.ops.aten.add.Tensor(mul_151, 1.0);  mul_151 = None
        reciprocal_20 = torch.ops.aten.reciprocal.default(add_86);  add_86 = None
        mul_152 = torch.ops.aten.mul.Tensor(reciprocal_20, 1.0);  reciprocal_20 = None
        mul_153 = torch.ops.aten.mul.Tensor(mul_152, 1.061405429)
        add_87 = torch.ops.aten.add.Tensor(mul_153, -1.453152027);  mul_153 = None
        mul_154 = torch.ops.aten.mul.Tensor(add_87, mul_152);  add_87 = None
        add_88 = torch.ops.aten.add.Tensor(mul_154, 1.421413741);  mul_154 = None
        mul_155 = torch.ops.aten.mul.Tensor(add_88, mul_152);  add_88 = None
        add_89 = torch.ops.aten.add.Tensor(mul_155, -0.284496736);  mul_155 = None
        mul_156 = torch.ops.aten.mul.Tensor(add_89, mul_152);  add_89 = None
        add_90 = torch.ops.aten.add.Tensor(mul_156, 0.254829592);  mul_156 = None
        mul_157 = torch.ops.aten.mul.Tensor(add_90, mul_152);  add_90 = mul_152 = None
        neg_6 = torch.ops.aten.neg.default(abs_7)
        mul_158 = torch.ops.aten.mul.Tensor(neg_6, abs_7);  neg_6 = abs_7 = None
        exp_13 = torch.ops.aten.exp.default(mul_158);  mul_158 = None
        mul_159 = torch.ops.aten.mul.Tensor(mul_157, exp_13);  mul_157 = exp_13 = None
        _tensor_constant7 = self._tensor_constant7
        lift_fresh_copy_7 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant7);  _tensor_constant7 = None
        sub_28 = torch.ops.aten.sub.Tensor(lift_fresh_copy_7, mul_159);  lift_fresh_copy_7 = None
        mul_160 = torch.ops.aten.mul.Tensor(sign_6, sub_28);  sub_28 = None
        add_91 = torch.ops.aten.add.Tensor(mul_160, 1);  mul_160 = None
        mul_161 = torch.ops.aten.mul.Tensor(mul_149, add_91);  mul_149 = add_91 = None
        permute_76 = torch.ops.aten.permute.default(primals_114, [1, 0]);  primals_114 = None
        view_117 = torch.ops.aten.view.default(mul_161, [8192, 3072]);  mul_161 = None
        addmm_41 = torch.ops.aten.addmm.default(primals_115, view_117, permute_76);  primals_115 = None
        view_118 = torch.ops.aten.view.default(addmm_41, [64, 128, 768]);  addmm_41 = None
        rand_like_21 = torch.ops.aten.rand_like.default(view_118, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_44 = torch.ops.aten.alias.default(rand_like_21);  rand_like_21 = None
        gt_21 = torch.ops.aten.gt.Scalar(alias_44, 0.1);  alias_44 = None
        mul_162 = torch.ops.aten.mul.Tensor(gt_21, view_118);  view_118 = None
        mul_163 = torch.ops.aten.mul.Tensor(mul_162, 1.1111111111111112);  mul_162 = None
        add_92 = torch.ops.aten.add.Tensor(mul_163, convert_element_type_13);  mul_163 = convert_element_type_13 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(add_92, [2], correction = 0, keepdim = True)
        getitem_28 = var_mean_14[0]
        getitem_29 = var_mean_14[1];  var_mean_14 = None
        add_93 = torch.ops.aten.add.Tensor(getitem_28, 1e-12);  getitem_28 = None
        sqrt_14 = torch.ops.aten.sqrt.default(add_93);  add_93 = None
        reciprocal_21 = torch.ops.aten.reciprocal.default(sqrt_14);  sqrt_14 = None
        sub_29 = torch.ops.aten.sub.Tensor(add_92, getitem_29);  add_92 = getitem_29 = None
        mul_164 = torch.ops.aten.mul.Tensor(sub_29, reciprocal_21);  sub_29 = None
        mul_165 = torch.ops.aten.mul.Tensor(mul_164, primals_116)
        add_94 = torch.ops.aten.add.Tensor(mul_165, primals_117);  mul_165 = primals_117 = None
        convert_element_type_14 = torch.ops.prims.convert_element_type.default(add_94, torch.float32);  add_94 = None
        permute_77 = torch.ops.aten.permute.default(primals_118, [1, 0]);  primals_118 = None
        view_119 = torch.ops.aten.view.default(convert_element_type_14, [8192, 768])
        addmm_42 = torch.ops.aten.addmm.default(primals_119, view_119, permute_77);  primals_119 = None
        view_120 = torch.ops.aten.view.default(addmm_42, [64, 128, 768]);  addmm_42 = None
        permute_78 = torch.ops.aten.permute.default(primals_120, [1, 0]);  primals_120 = None
        addmm_43 = torch.ops.aten.addmm.default(primals_121, view_119, permute_78);  primals_121 = None
        view_122 = torch.ops.aten.view.default(addmm_43, [64, 128, 768]);  addmm_43 = None
        view_123 = torch.ops.aten.view.default(view_122, [64, 128, 12, 64]);  view_122 = None
        permute_79 = torch.ops.aten.permute.default(view_123, [0, 2, 1, 3]);  view_123 = None
        permute_80 = torch.ops.aten.permute.default(primals_122, [1, 0]);  primals_122 = None
        addmm_44 = torch.ops.aten.addmm.default(primals_123, view_119, permute_80);  primals_123 = None
        view_125 = torch.ops.aten.view.default(addmm_44, [64, 128, 768]);  addmm_44 = None
        view_126 = torch.ops.aten.view.default(view_125, [64, 128, 12, 64]);  view_125 = None
        permute_81 = torch.ops.aten.permute.default(view_126, [0, 2, 1, 3]);  view_126 = None
        view_127 = torch.ops.aten.view.default(view_120, [64, 128, 12, 64]);  view_120 = None
        permute_82 = torch.ops.aten.permute.default(view_127, [0, 2, 1, 3]);  view_127 = None
        permute_83 = torch.ops.aten.permute.default(permute_79, [0, 1, 3, 2]);  permute_79 = None
        expand_29 = torch.ops.aten.expand.default(permute_82, [64, 12, 128, 64]);  permute_82 = None
        clone_28 = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
        _unsafe_view_35 = torch.ops.aten._unsafe_view.default(clone_28, [768, 128, 64]);  clone_28 = None
        expand_30 = torch.ops.aten.expand.default(permute_83, [64, 12, 64, 128]);  permute_83 = None
        clone_29 = torch.ops.aten.clone.default(expand_30, memory_format = torch.contiguous_format);  expand_30 = None
        _unsafe_view_36 = torch.ops.aten._unsafe_view.default(clone_29, [768, 64, 128]);  clone_29 = None
        bmm_14 = torch.ops.aten.bmm.default(_unsafe_view_35, _unsafe_view_36)
        _unsafe_view_37 = torch.ops.aten._unsafe_view.default(bmm_14, [64, 12, 128, 128]);  bmm_14 = None
        div_14 = torch.ops.aten.div.Tensor(_unsafe_view_37, 8.0);  _unsafe_view_37 = None
        add_95 = torch.ops.aten.add.Tensor(div_14, mul);  div_14 = None
        amax_7 = torch.ops.aten.amax.default(add_95, [-1], True)
        sub_30 = torch.ops.aten.sub.Tensor(add_95, amax_7);  add_95 = amax_7 = None
        exp_14 = torch.ops.aten.exp.default(sub_30);  sub_30 = None
        sum_8 = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
        div_15 = torch.ops.aten.div.Tensor(exp_14, sum_8);  exp_14 = sum_8 = None
        alias_46 = torch.ops.aten.alias.default(div_15)
        alias_47 = torch.ops.aten.alias.default(alias_46);  alias_46 = None
        rand_like_22 = torch.ops.aten.rand_like.default(div_15, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_48 = torch.ops.aten.alias.default(rand_like_22);  rand_like_22 = None
        gt_22 = torch.ops.aten.gt.Scalar(alias_48, 0.1);  alias_48 = None
        mul_166 = torch.ops.aten.mul.Tensor(gt_22, div_15);  div_15 = None
        mul_167 = torch.ops.aten.mul.Tensor(mul_166, 1.1111111111111112);  mul_166 = None
        expand_31 = torch.ops.aten.expand.default(mul_167, [64, 12, 128, 128]);  mul_167 = None
        view_128 = torch.ops.aten.view.default(expand_31, [768, 128, 128]);  expand_31 = None
        expand_32 = torch.ops.aten.expand.default(permute_81, [64, 12, 128, 64]);  permute_81 = None
        clone_30 = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
        _unsafe_view_38 = torch.ops.aten._unsafe_view.default(clone_30, [768, 128, 64]);  clone_30 = None
        bmm_15 = torch.ops.aten.bmm.default(view_128, _unsafe_view_38)
        _unsafe_view_39 = torch.ops.aten._unsafe_view.default(bmm_15, [64, 12, 128, 64]);  bmm_15 = None
        permute_84 = torch.ops.aten.permute.default(_unsafe_view_39, [0, 2, 1, 3]);  _unsafe_view_39 = None
        clone_31 = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
        view_129 = torch.ops.aten.view.default(clone_31, [64, 128, 768]);  clone_31 = None
        permute_85 = torch.ops.aten.permute.default(primals_124, [1, 0]);  primals_124 = None
        view_130 = torch.ops.aten.view.default(view_129, [8192, 768]);  view_129 = None
        addmm_45 = torch.ops.aten.addmm.default(primals_125, view_130, permute_85);  primals_125 = None
        view_131 = torch.ops.aten.view.default(addmm_45, [64, 128, 768]);  addmm_45 = None
        rand_like_23 = torch.ops.aten.rand_like.default(view_131, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_49 = torch.ops.aten.alias.default(rand_like_23);  rand_like_23 = None
        gt_23 = torch.ops.aten.gt.Scalar(alias_49, 0.1);  alias_49 = None
        mul_168 = torch.ops.aten.mul.Tensor(gt_23, view_131);  view_131 = None
        mul_169 = torch.ops.aten.mul.Tensor(mul_168, 1.1111111111111112);  mul_168 = None
        add_96 = torch.ops.aten.add.Tensor(mul_169, convert_element_type_14);  mul_169 = convert_element_type_14 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(add_96, [2], correction = 0, keepdim = True)
        getitem_30 = var_mean_15[0]
        getitem_31 = var_mean_15[1];  var_mean_15 = None
        add_97 = torch.ops.aten.add.Tensor(getitem_30, 1e-12);  getitem_30 = None
        sqrt_15 = torch.ops.aten.sqrt.default(add_97);  add_97 = None
        reciprocal_22 = torch.ops.aten.reciprocal.default(sqrt_15);  sqrt_15 = None
        sub_31 = torch.ops.aten.sub.Tensor(add_96, getitem_31);  add_96 = getitem_31 = None
        mul_170 = torch.ops.aten.mul.Tensor(sub_31, reciprocal_22);  sub_31 = None
        mul_171 = torch.ops.aten.mul.Tensor(mul_170, primals_126)
        add_98 = torch.ops.aten.add.Tensor(mul_171, primals_127);  mul_171 = primals_127 = None
        convert_element_type_15 = torch.ops.prims.convert_element_type.default(add_98, torch.float32);  add_98 = None
        permute_86 = torch.ops.aten.permute.default(primals_128, [1, 0]);  primals_128 = None
        view_132 = torch.ops.aten.view.default(convert_element_type_15, [8192, 768])
        addmm_46 = torch.ops.aten.addmm.default(primals_129, view_132, permute_86);  primals_129 = None
        view_133 = torch.ops.aten.view.default(addmm_46, [64, 128, 3072]);  addmm_46 = None
        mul_172 = torch.ops.aten.mul.Tensor(view_133, 0.5)
        mul_173 = torch.ops.aten.mul.Tensor(view_133, 0.7071067811865476)
        sign_7 = torch.ops.aten.sign.default(mul_173)
        abs_8 = torch.ops.aten.abs.default(mul_173);  mul_173 = None
        mul_174 = torch.ops.aten.mul.Tensor(abs_8, 0.3275911)
        add_99 = torch.ops.aten.add.Tensor(mul_174, 1.0);  mul_174 = None
        reciprocal_23 = torch.ops.aten.reciprocal.default(add_99);  add_99 = None
        mul_175 = torch.ops.aten.mul.Tensor(reciprocal_23, 1.0);  reciprocal_23 = None
        mul_176 = torch.ops.aten.mul.Tensor(mul_175, 1.061405429)
        add_100 = torch.ops.aten.add.Tensor(mul_176, -1.453152027);  mul_176 = None
        mul_177 = torch.ops.aten.mul.Tensor(add_100, mul_175);  add_100 = None
        add_101 = torch.ops.aten.add.Tensor(mul_177, 1.421413741);  mul_177 = None
        mul_178 = torch.ops.aten.mul.Tensor(add_101, mul_175);  add_101 = None
        add_102 = torch.ops.aten.add.Tensor(mul_178, -0.284496736);  mul_178 = None
        mul_179 = torch.ops.aten.mul.Tensor(add_102, mul_175);  add_102 = None
        add_103 = torch.ops.aten.add.Tensor(mul_179, 0.254829592);  mul_179 = None
        mul_180 = torch.ops.aten.mul.Tensor(add_103, mul_175);  add_103 = mul_175 = None
        neg_7 = torch.ops.aten.neg.default(abs_8)
        mul_181 = torch.ops.aten.mul.Tensor(neg_7, abs_8);  neg_7 = abs_8 = None
        exp_15 = torch.ops.aten.exp.default(mul_181);  mul_181 = None
        mul_182 = torch.ops.aten.mul.Tensor(mul_180, exp_15);  mul_180 = exp_15 = None
        _tensor_constant8 = self._tensor_constant8
        lift_fresh_copy_8 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant8);  _tensor_constant8 = None
        sub_32 = torch.ops.aten.sub.Tensor(lift_fresh_copy_8, mul_182);  lift_fresh_copy_8 = None
        mul_183 = torch.ops.aten.mul.Tensor(sign_7, sub_32);  sub_32 = None
        add_104 = torch.ops.aten.add.Tensor(mul_183, 1);  mul_183 = None
        mul_184 = torch.ops.aten.mul.Tensor(mul_172, add_104);  mul_172 = add_104 = None
        permute_87 = torch.ops.aten.permute.default(primals_130, [1, 0]);  primals_130 = None
        view_134 = torch.ops.aten.view.default(mul_184, [8192, 3072]);  mul_184 = None
        addmm_47 = torch.ops.aten.addmm.default(primals_131, view_134, permute_87);  primals_131 = None
        view_135 = torch.ops.aten.view.default(addmm_47, [64, 128, 768]);  addmm_47 = None
        rand_like_24 = torch.ops.aten.rand_like.default(view_135, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_50 = torch.ops.aten.alias.default(rand_like_24);  rand_like_24 = None
        gt_24 = torch.ops.aten.gt.Scalar(alias_50, 0.1);  alias_50 = None
        mul_185 = torch.ops.aten.mul.Tensor(gt_24, view_135);  view_135 = None
        mul_186 = torch.ops.aten.mul.Tensor(mul_185, 1.1111111111111112);  mul_185 = None
        add_105 = torch.ops.aten.add.Tensor(mul_186, convert_element_type_15);  mul_186 = convert_element_type_15 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(add_105, [2], correction = 0, keepdim = True)
        getitem_32 = var_mean_16[0]
        getitem_33 = var_mean_16[1];  var_mean_16 = None
        add_106 = torch.ops.aten.add.Tensor(getitem_32, 1e-12);  getitem_32 = None
        sqrt_16 = torch.ops.aten.sqrt.default(add_106);  add_106 = None
        reciprocal_24 = torch.ops.aten.reciprocal.default(sqrt_16);  sqrt_16 = None
        sub_33 = torch.ops.aten.sub.Tensor(add_105, getitem_33);  add_105 = getitem_33 = None
        mul_187 = torch.ops.aten.mul.Tensor(sub_33, reciprocal_24);  sub_33 = None
        mul_188 = torch.ops.aten.mul.Tensor(mul_187, primals_132)
        add_107 = torch.ops.aten.add.Tensor(mul_188, primals_133);  mul_188 = primals_133 = None
        convert_element_type_16 = torch.ops.prims.convert_element_type.default(add_107, torch.float32);  add_107 = None
        permute_88 = torch.ops.aten.permute.default(primals_134, [1, 0]);  primals_134 = None
        view_136 = torch.ops.aten.view.default(convert_element_type_16, [8192, 768])
        addmm_48 = torch.ops.aten.addmm.default(primals_135, view_136, permute_88);  primals_135 = None
        view_137 = torch.ops.aten.view.default(addmm_48, [64, 128, 768]);  addmm_48 = None
        permute_89 = torch.ops.aten.permute.default(primals_136, [1, 0]);  primals_136 = None
        addmm_49 = torch.ops.aten.addmm.default(primals_137, view_136, permute_89);  primals_137 = None
        view_139 = torch.ops.aten.view.default(addmm_49, [64, 128, 768]);  addmm_49 = None
        view_140 = torch.ops.aten.view.default(view_139, [64, 128, 12, 64]);  view_139 = None
        permute_90 = torch.ops.aten.permute.default(view_140, [0, 2, 1, 3]);  view_140 = None
        permute_91 = torch.ops.aten.permute.default(primals_138, [1, 0]);  primals_138 = None
        addmm_50 = torch.ops.aten.addmm.default(primals_139, view_136, permute_91);  primals_139 = None
        view_142 = torch.ops.aten.view.default(addmm_50, [64, 128, 768]);  addmm_50 = None
        view_143 = torch.ops.aten.view.default(view_142, [64, 128, 12, 64]);  view_142 = None
        permute_92 = torch.ops.aten.permute.default(view_143, [0, 2, 1, 3]);  view_143 = None
        view_144 = torch.ops.aten.view.default(view_137, [64, 128, 12, 64]);  view_137 = None
        permute_93 = torch.ops.aten.permute.default(view_144, [0, 2, 1, 3]);  view_144 = None
        permute_94 = torch.ops.aten.permute.default(permute_90, [0, 1, 3, 2]);  permute_90 = None
        expand_33 = torch.ops.aten.expand.default(permute_93, [64, 12, 128, 64]);  permute_93 = None
        clone_32 = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
        _unsafe_view_40 = torch.ops.aten._unsafe_view.default(clone_32, [768, 128, 64]);  clone_32 = None
        expand_34 = torch.ops.aten.expand.default(permute_94, [64, 12, 64, 128]);  permute_94 = None
        clone_33 = torch.ops.aten.clone.default(expand_34, memory_format = torch.contiguous_format);  expand_34 = None
        _unsafe_view_41 = torch.ops.aten._unsafe_view.default(clone_33, [768, 64, 128]);  clone_33 = None
        bmm_16 = torch.ops.aten.bmm.default(_unsafe_view_40, _unsafe_view_41)
        _unsafe_view_42 = torch.ops.aten._unsafe_view.default(bmm_16, [64, 12, 128, 128]);  bmm_16 = None
        div_16 = torch.ops.aten.div.Tensor(_unsafe_view_42, 8.0);  _unsafe_view_42 = None
        add_108 = torch.ops.aten.add.Tensor(div_16, mul);  div_16 = None
        amax_8 = torch.ops.aten.amax.default(add_108, [-1], True)
        sub_34 = torch.ops.aten.sub.Tensor(add_108, amax_8);  add_108 = amax_8 = None
        exp_16 = torch.ops.aten.exp.default(sub_34);  sub_34 = None
        sum_9 = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
        div_17 = torch.ops.aten.div.Tensor(exp_16, sum_9);  exp_16 = sum_9 = None
        alias_52 = torch.ops.aten.alias.default(div_17)
        alias_53 = torch.ops.aten.alias.default(alias_52);  alias_52 = None
        rand_like_25 = torch.ops.aten.rand_like.default(div_17, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_54 = torch.ops.aten.alias.default(rand_like_25);  rand_like_25 = None
        gt_25 = torch.ops.aten.gt.Scalar(alias_54, 0.1);  alias_54 = None
        mul_189 = torch.ops.aten.mul.Tensor(gt_25, div_17);  div_17 = None
        mul_190 = torch.ops.aten.mul.Tensor(mul_189, 1.1111111111111112);  mul_189 = None
        expand_35 = torch.ops.aten.expand.default(mul_190, [64, 12, 128, 128]);  mul_190 = None
        view_145 = torch.ops.aten.view.default(expand_35, [768, 128, 128]);  expand_35 = None
        expand_36 = torch.ops.aten.expand.default(permute_92, [64, 12, 128, 64]);  permute_92 = None
        clone_34 = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
        _unsafe_view_43 = torch.ops.aten._unsafe_view.default(clone_34, [768, 128, 64]);  clone_34 = None
        bmm_17 = torch.ops.aten.bmm.default(view_145, _unsafe_view_43)
        _unsafe_view_44 = torch.ops.aten._unsafe_view.default(bmm_17, [64, 12, 128, 64]);  bmm_17 = None
        permute_95 = torch.ops.aten.permute.default(_unsafe_view_44, [0, 2, 1, 3]);  _unsafe_view_44 = None
        clone_35 = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
        view_146 = torch.ops.aten.view.default(clone_35, [64, 128, 768]);  clone_35 = None
        permute_96 = torch.ops.aten.permute.default(primals_140, [1, 0]);  primals_140 = None
        view_147 = torch.ops.aten.view.default(view_146, [8192, 768]);  view_146 = None
        addmm_51 = torch.ops.aten.addmm.default(primals_141, view_147, permute_96);  primals_141 = None
        view_148 = torch.ops.aten.view.default(addmm_51, [64, 128, 768]);  addmm_51 = None
        rand_like_26 = torch.ops.aten.rand_like.default(view_148, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_55 = torch.ops.aten.alias.default(rand_like_26);  rand_like_26 = None
        gt_26 = torch.ops.aten.gt.Scalar(alias_55, 0.1);  alias_55 = None
        mul_191 = torch.ops.aten.mul.Tensor(gt_26, view_148);  view_148 = None
        mul_192 = torch.ops.aten.mul.Tensor(mul_191, 1.1111111111111112);  mul_191 = None
        add_109 = torch.ops.aten.add.Tensor(mul_192, convert_element_type_16);  mul_192 = convert_element_type_16 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(add_109, [2], correction = 0, keepdim = True)
        getitem_34 = var_mean_17[0]
        getitem_35 = var_mean_17[1];  var_mean_17 = None
        add_110 = torch.ops.aten.add.Tensor(getitem_34, 1e-12);  getitem_34 = None
        sqrt_17 = torch.ops.aten.sqrt.default(add_110);  add_110 = None
        reciprocal_25 = torch.ops.aten.reciprocal.default(sqrt_17);  sqrt_17 = None
        sub_35 = torch.ops.aten.sub.Tensor(add_109, getitem_35);  add_109 = getitem_35 = None
        mul_193 = torch.ops.aten.mul.Tensor(sub_35, reciprocal_25);  sub_35 = None
        mul_194 = torch.ops.aten.mul.Tensor(mul_193, primals_142)
        add_111 = torch.ops.aten.add.Tensor(mul_194, primals_143);  mul_194 = primals_143 = None
        convert_element_type_17 = torch.ops.prims.convert_element_type.default(add_111, torch.float32);  add_111 = None
        permute_97 = torch.ops.aten.permute.default(primals_144, [1, 0]);  primals_144 = None
        view_149 = torch.ops.aten.view.default(convert_element_type_17, [8192, 768])
        addmm_52 = torch.ops.aten.addmm.default(primals_145, view_149, permute_97);  primals_145 = None
        view_150 = torch.ops.aten.view.default(addmm_52, [64, 128, 3072]);  addmm_52 = None
        mul_195 = torch.ops.aten.mul.Tensor(view_150, 0.5)
        mul_196 = torch.ops.aten.mul.Tensor(view_150, 0.7071067811865476)
        sign_8 = torch.ops.aten.sign.default(mul_196)
        abs_9 = torch.ops.aten.abs.default(mul_196);  mul_196 = None
        mul_197 = torch.ops.aten.mul.Tensor(abs_9, 0.3275911)
        add_112 = torch.ops.aten.add.Tensor(mul_197, 1.0);  mul_197 = None
        reciprocal_26 = torch.ops.aten.reciprocal.default(add_112);  add_112 = None
        mul_198 = torch.ops.aten.mul.Tensor(reciprocal_26, 1.0);  reciprocal_26 = None
        mul_199 = torch.ops.aten.mul.Tensor(mul_198, 1.061405429)
        add_113 = torch.ops.aten.add.Tensor(mul_199, -1.453152027);  mul_199 = None
        mul_200 = torch.ops.aten.mul.Tensor(add_113, mul_198);  add_113 = None
        add_114 = torch.ops.aten.add.Tensor(mul_200, 1.421413741);  mul_200 = None
        mul_201 = torch.ops.aten.mul.Tensor(add_114, mul_198);  add_114 = None
        add_115 = torch.ops.aten.add.Tensor(mul_201, -0.284496736);  mul_201 = None
        mul_202 = torch.ops.aten.mul.Tensor(add_115, mul_198);  add_115 = None
        add_116 = torch.ops.aten.add.Tensor(mul_202, 0.254829592);  mul_202 = None
        mul_203 = torch.ops.aten.mul.Tensor(add_116, mul_198);  add_116 = mul_198 = None
        neg_8 = torch.ops.aten.neg.default(abs_9)
        mul_204 = torch.ops.aten.mul.Tensor(neg_8, abs_9);  neg_8 = abs_9 = None
        exp_17 = torch.ops.aten.exp.default(mul_204);  mul_204 = None
        mul_205 = torch.ops.aten.mul.Tensor(mul_203, exp_17);  mul_203 = exp_17 = None
        _tensor_constant9 = self._tensor_constant9
        lift_fresh_copy_9 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant9);  _tensor_constant9 = None
        sub_36 = torch.ops.aten.sub.Tensor(lift_fresh_copy_9, mul_205);  lift_fresh_copy_9 = None
        mul_206 = torch.ops.aten.mul.Tensor(sign_8, sub_36);  sub_36 = None
        add_117 = torch.ops.aten.add.Tensor(mul_206, 1);  mul_206 = None
        mul_207 = torch.ops.aten.mul.Tensor(mul_195, add_117);  mul_195 = add_117 = None
        permute_98 = torch.ops.aten.permute.default(primals_146, [1, 0]);  primals_146 = None
        view_151 = torch.ops.aten.view.default(mul_207, [8192, 3072]);  mul_207 = None
        addmm_53 = torch.ops.aten.addmm.default(primals_147, view_151, permute_98);  primals_147 = None
        view_152 = torch.ops.aten.view.default(addmm_53, [64, 128, 768]);  addmm_53 = None
        rand_like_27 = torch.ops.aten.rand_like.default(view_152, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_56 = torch.ops.aten.alias.default(rand_like_27);  rand_like_27 = None
        gt_27 = torch.ops.aten.gt.Scalar(alias_56, 0.1);  alias_56 = None
        mul_208 = torch.ops.aten.mul.Tensor(gt_27, view_152);  view_152 = None
        mul_209 = torch.ops.aten.mul.Tensor(mul_208, 1.1111111111111112);  mul_208 = None
        add_118 = torch.ops.aten.add.Tensor(mul_209, convert_element_type_17);  mul_209 = convert_element_type_17 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(add_118, [2], correction = 0, keepdim = True)
        getitem_36 = var_mean_18[0]
        getitem_37 = var_mean_18[1];  var_mean_18 = None
        add_119 = torch.ops.aten.add.Tensor(getitem_36, 1e-12);  getitem_36 = None
        sqrt_18 = torch.ops.aten.sqrt.default(add_119);  add_119 = None
        reciprocal_27 = torch.ops.aten.reciprocal.default(sqrt_18);  sqrt_18 = None
        sub_37 = torch.ops.aten.sub.Tensor(add_118, getitem_37);  add_118 = getitem_37 = None
        mul_210 = torch.ops.aten.mul.Tensor(sub_37, reciprocal_27);  sub_37 = None
        mul_211 = torch.ops.aten.mul.Tensor(mul_210, primals_148)
        add_120 = torch.ops.aten.add.Tensor(mul_211, primals_149);  mul_211 = primals_149 = None
        convert_element_type_18 = torch.ops.prims.convert_element_type.default(add_120, torch.float32);  add_120 = None
        permute_99 = torch.ops.aten.permute.default(primals_150, [1, 0]);  primals_150 = None
        view_153 = torch.ops.aten.view.default(convert_element_type_18, [8192, 768])
        addmm_54 = torch.ops.aten.addmm.default(primals_151, view_153, permute_99);  primals_151 = None
        view_154 = torch.ops.aten.view.default(addmm_54, [64, 128, 768]);  addmm_54 = None
        permute_100 = torch.ops.aten.permute.default(primals_152, [1, 0]);  primals_152 = None
        addmm_55 = torch.ops.aten.addmm.default(primals_153, view_153, permute_100);  primals_153 = None
        view_156 = torch.ops.aten.view.default(addmm_55, [64, 128, 768]);  addmm_55 = None
        view_157 = torch.ops.aten.view.default(view_156, [64, 128, 12, 64]);  view_156 = None
        permute_101 = torch.ops.aten.permute.default(view_157, [0, 2, 1, 3]);  view_157 = None
        permute_102 = torch.ops.aten.permute.default(primals_154, [1, 0]);  primals_154 = None
        addmm_56 = torch.ops.aten.addmm.default(primals_155, view_153, permute_102);  primals_155 = None
        view_159 = torch.ops.aten.view.default(addmm_56, [64, 128, 768]);  addmm_56 = None
        view_160 = torch.ops.aten.view.default(view_159, [64, 128, 12, 64]);  view_159 = None
        permute_103 = torch.ops.aten.permute.default(view_160, [0, 2, 1, 3]);  view_160 = None
        view_161 = torch.ops.aten.view.default(view_154, [64, 128, 12, 64]);  view_154 = None
        permute_104 = torch.ops.aten.permute.default(view_161, [0, 2, 1, 3]);  view_161 = None
        permute_105 = torch.ops.aten.permute.default(permute_101, [0, 1, 3, 2]);  permute_101 = None
        expand_37 = torch.ops.aten.expand.default(permute_104, [64, 12, 128, 64]);  permute_104 = None
        clone_36 = torch.ops.aten.clone.default(expand_37, memory_format = torch.contiguous_format);  expand_37 = None
        _unsafe_view_45 = torch.ops.aten._unsafe_view.default(clone_36, [768, 128, 64]);  clone_36 = None
        expand_38 = torch.ops.aten.expand.default(permute_105, [64, 12, 64, 128]);  permute_105 = None
        clone_37 = torch.ops.aten.clone.default(expand_38, memory_format = torch.contiguous_format);  expand_38 = None
        _unsafe_view_46 = torch.ops.aten._unsafe_view.default(clone_37, [768, 64, 128]);  clone_37 = None
        bmm_18 = torch.ops.aten.bmm.default(_unsafe_view_45, _unsafe_view_46)
        _unsafe_view_47 = torch.ops.aten._unsafe_view.default(bmm_18, [64, 12, 128, 128]);  bmm_18 = None
        div_18 = torch.ops.aten.div.Tensor(_unsafe_view_47, 8.0);  _unsafe_view_47 = None
        add_121 = torch.ops.aten.add.Tensor(div_18, mul);  div_18 = None
        amax_9 = torch.ops.aten.amax.default(add_121, [-1], True)
        sub_38 = torch.ops.aten.sub.Tensor(add_121, amax_9);  add_121 = amax_9 = None
        exp_18 = torch.ops.aten.exp.default(sub_38);  sub_38 = None
        sum_10 = torch.ops.aten.sum.dim_IntList(exp_18, [-1], True)
        div_19 = torch.ops.aten.div.Tensor(exp_18, sum_10);  exp_18 = sum_10 = None
        alias_58 = torch.ops.aten.alias.default(div_19)
        alias_59 = torch.ops.aten.alias.default(alias_58);  alias_58 = None
        rand_like_28 = torch.ops.aten.rand_like.default(div_19, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_60 = torch.ops.aten.alias.default(rand_like_28);  rand_like_28 = None
        gt_28 = torch.ops.aten.gt.Scalar(alias_60, 0.1);  alias_60 = None
        mul_212 = torch.ops.aten.mul.Tensor(gt_28, div_19);  div_19 = None
        mul_213 = torch.ops.aten.mul.Tensor(mul_212, 1.1111111111111112);  mul_212 = None
        expand_39 = torch.ops.aten.expand.default(mul_213, [64, 12, 128, 128]);  mul_213 = None
        view_162 = torch.ops.aten.view.default(expand_39, [768, 128, 128]);  expand_39 = None
        expand_40 = torch.ops.aten.expand.default(permute_103, [64, 12, 128, 64]);  permute_103 = None
        clone_38 = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
        _unsafe_view_48 = torch.ops.aten._unsafe_view.default(clone_38, [768, 128, 64]);  clone_38 = None
        bmm_19 = torch.ops.aten.bmm.default(view_162, _unsafe_view_48)
        _unsafe_view_49 = torch.ops.aten._unsafe_view.default(bmm_19, [64, 12, 128, 64]);  bmm_19 = None
        permute_106 = torch.ops.aten.permute.default(_unsafe_view_49, [0, 2, 1, 3]);  _unsafe_view_49 = None
        clone_39 = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None
        view_163 = torch.ops.aten.view.default(clone_39, [64, 128, 768]);  clone_39 = None
        permute_107 = torch.ops.aten.permute.default(primals_156, [1, 0]);  primals_156 = None
        view_164 = torch.ops.aten.view.default(view_163, [8192, 768]);  view_163 = None
        addmm_57 = torch.ops.aten.addmm.default(primals_157, view_164, permute_107);  primals_157 = None
        view_165 = torch.ops.aten.view.default(addmm_57, [64, 128, 768]);  addmm_57 = None
        rand_like_29 = torch.ops.aten.rand_like.default(view_165, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_61 = torch.ops.aten.alias.default(rand_like_29);  rand_like_29 = None
        gt_29 = torch.ops.aten.gt.Scalar(alias_61, 0.1);  alias_61 = None
        mul_214 = torch.ops.aten.mul.Tensor(gt_29, view_165);  view_165 = None
        mul_215 = torch.ops.aten.mul.Tensor(mul_214, 1.1111111111111112);  mul_214 = None
        add_122 = torch.ops.aten.add.Tensor(mul_215, convert_element_type_18);  mul_215 = convert_element_type_18 = None
        var_mean_19 = torch.ops.aten.var_mean.correction(add_122, [2], correction = 0, keepdim = True)
        getitem_38 = var_mean_19[0]
        getitem_39 = var_mean_19[1];  var_mean_19 = None
        add_123 = torch.ops.aten.add.Tensor(getitem_38, 1e-12);  getitem_38 = None
        sqrt_19 = torch.ops.aten.sqrt.default(add_123);  add_123 = None
        reciprocal_28 = torch.ops.aten.reciprocal.default(sqrt_19);  sqrt_19 = None
        sub_39 = torch.ops.aten.sub.Tensor(add_122, getitem_39);  add_122 = getitem_39 = None
        mul_216 = torch.ops.aten.mul.Tensor(sub_39, reciprocal_28);  sub_39 = None
        mul_217 = torch.ops.aten.mul.Tensor(mul_216, primals_158)
        add_124 = torch.ops.aten.add.Tensor(mul_217, primals_159);  mul_217 = primals_159 = None
        convert_element_type_19 = torch.ops.prims.convert_element_type.default(add_124, torch.float32);  add_124 = None
        permute_108 = torch.ops.aten.permute.default(primals_160, [1, 0]);  primals_160 = None
        view_166 = torch.ops.aten.view.default(convert_element_type_19, [8192, 768])
        addmm_58 = torch.ops.aten.addmm.default(primals_161, view_166, permute_108);  primals_161 = None
        view_167 = torch.ops.aten.view.default(addmm_58, [64, 128, 3072]);  addmm_58 = None
        mul_218 = torch.ops.aten.mul.Tensor(view_167, 0.5)
        mul_219 = torch.ops.aten.mul.Tensor(view_167, 0.7071067811865476)
        sign_9 = torch.ops.aten.sign.default(mul_219)
        abs_10 = torch.ops.aten.abs.default(mul_219);  mul_219 = None
        mul_220 = torch.ops.aten.mul.Tensor(abs_10, 0.3275911)
        add_125 = torch.ops.aten.add.Tensor(mul_220, 1.0);  mul_220 = None
        reciprocal_29 = torch.ops.aten.reciprocal.default(add_125);  add_125 = None
        mul_221 = torch.ops.aten.mul.Tensor(reciprocal_29, 1.0);  reciprocal_29 = None
        mul_222 = torch.ops.aten.mul.Tensor(mul_221, 1.061405429)
        add_126 = torch.ops.aten.add.Tensor(mul_222, -1.453152027);  mul_222 = None
        mul_223 = torch.ops.aten.mul.Tensor(add_126, mul_221);  add_126 = None
        add_127 = torch.ops.aten.add.Tensor(mul_223, 1.421413741);  mul_223 = None
        mul_224 = torch.ops.aten.mul.Tensor(add_127, mul_221);  add_127 = None
        add_128 = torch.ops.aten.add.Tensor(mul_224, -0.284496736);  mul_224 = None
        mul_225 = torch.ops.aten.mul.Tensor(add_128, mul_221);  add_128 = None
        add_129 = torch.ops.aten.add.Tensor(mul_225, 0.254829592);  mul_225 = None
        mul_226 = torch.ops.aten.mul.Tensor(add_129, mul_221);  add_129 = mul_221 = None
        neg_9 = torch.ops.aten.neg.default(abs_10)
        mul_227 = torch.ops.aten.mul.Tensor(neg_9, abs_10);  neg_9 = abs_10 = None
        exp_19 = torch.ops.aten.exp.default(mul_227);  mul_227 = None
        mul_228 = torch.ops.aten.mul.Tensor(mul_226, exp_19);  mul_226 = exp_19 = None
        _tensor_constant10 = self._tensor_constant10
        lift_fresh_copy_10 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant10);  _tensor_constant10 = None
        sub_40 = torch.ops.aten.sub.Tensor(lift_fresh_copy_10, mul_228);  lift_fresh_copy_10 = None
        mul_229 = torch.ops.aten.mul.Tensor(sign_9, sub_40);  sub_40 = None
        add_130 = torch.ops.aten.add.Tensor(mul_229, 1);  mul_229 = None
        mul_230 = torch.ops.aten.mul.Tensor(mul_218, add_130);  mul_218 = add_130 = None
        permute_109 = torch.ops.aten.permute.default(primals_162, [1, 0]);  primals_162 = None
        view_168 = torch.ops.aten.view.default(mul_230, [8192, 3072]);  mul_230 = None
        addmm_59 = torch.ops.aten.addmm.default(primals_163, view_168, permute_109);  primals_163 = None
        view_169 = torch.ops.aten.view.default(addmm_59, [64, 128, 768]);  addmm_59 = None
        rand_like_30 = torch.ops.aten.rand_like.default(view_169, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_62 = torch.ops.aten.alias.default(rand_like_30);  rand_like_30 = None
        gt_30 = torch.ops.aten.gt.Scalar(alias_62, 0.1);  alias_62 = None
        mul_231 = torch.ops.aten.mul.Tensor(gt_30, view_169);  view_169 = None
        mul_232 = torch.ops.aten.mul.Tensor(mul_231, 1.1111111111111112);  mul_231 = None
        add_131 = torch.ops.aten.add.Tensor(mul_232, convert_element_type_19);  mul_232 = convert_element_type_19 = None
        var_mean_20 = torch.ops.aten.var_mean.correction(add_131, [2], correction = 0, keepdim = True)
        getitem_40 = var_mean_20[0]
        getitem_41 = var_mean_20[1];  var_mean_20 = None
        add_132 = torch.ops.aten.add.Tensor(getitem_40, 1e-12);  getitem_40 = None
        sqrt_20 = torch.ops.aten.sqrt.default(add_132);  add_132 = None
        reciprocal_30 = torch.ops.aten.reciprocal.default(sqrt_20);  sqrt_20 = None
        sub_41 = torch.ops.aten.sub.Tensor(add_131, getitem_41);  add_131 = getitem_41 = None
        mul_233 = torch.ops.aten.mul.Tensor(sub_41, reciprocal_30);  sub_41 = None
        mul_234 = torch.ops.aten.mul.Tensor(mul_233, primals_164)
        add_133 = torch.ops.aten.add.Tensor(mul_234, primals_165);  mul_234 = primals_165 = None
        convert_element_type_20 = torch.ops.prims.convert_element_type.default(add_133, torch.float32);  add_133 = None
        permute_110 = torch.ops.aten.permute.default(primals_166, [1, 0]);  primals_166 = None
        view_170 = torch.ops.aten.view.default(convert_element_type_20, [8192, 768])
        addmm_60 = torch.ops.aten.addmm.default(primals_167, view_170, permute_110);  primals_167 = None
        view_171 = torch.ops.aten.view.default(addmm_60, [64, 128, 768]);  addmm_60 = None
        permute_111 = torch.ops.aten.permute.default(primals_168, [1, 0]);  primals_168 = None
        addmm_61 = torch.ops.aten.addmm.default(primals_169, view_170, permute_111);  primals_169 = None
        view_173 = torch.ops.aten.view.default(addmm_61, [64, 128, 768]);  addmm_61 = None
        view_174 = torch.ops.aten.view.default(view_173, [64, 128, 12, 64]);  view_173 = None
        permute_112 = torch.ops.aten.permute.default(view_174, [0, 2, 1, 3]);  view_174 = None
        permute_113 = torch.ops.aten.permute.default(primals_170, [1, 0]);  primals_170 = None
        addmm_62 = torch.ops.aten.addmm.default(primals_171, view_170, permute_113);  primals_171 = None
        view_176 = torch.ops.aten.view.default(addmm_62, [64, 128, 768]);  addmm_62 = None
        view_177 = torch.ops.aten.view.default(view_176, [64, 128, 12, 64]);  view_176 = None
        permute_114 = torch.ops.aten.permute.default(view_177, [0, 2, 1, 3]);  view_177 = None
        view_178 = torch.ops.aten.view.default(view_171, [64, 128, 12, 64]);  view_171 = None
        permute_115 = torch.ops.aten.permute.default(view_178, [0, 2, 1, 3]);  view_178 = None
        permute_116 = torch.ops.aten.permute.default(permute_112, [0, 1, 3, 2]);  permute_112 = None
        expand_41 = torch.ops.aten.expand.default(permute_115, [64, 12, 128, 64]);  permute_115 = None
        clone_40 = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
        _unsafe_view_50 = torch.ops.aten._unsafe_view.default(clone_40, [768, 128, 64]);  clone_40 = None
        expand_42 = torch.ops.aten.expand.default(permute_116, [64, 12, 64, 128]);  permute_116 = None
        clone_41 = torch.ops.aten.clone.default(expand_42, memory_format = torch.contiguous_format);  expand_42 = None
        _unsafe_view_51 = torch.ops.aten._unsafe_view.default(clone_41, [768, 64, 128]);  clone_41 = None
        bmm_20 = torch.ops.aten.bmm.default(_unsafe_view_50, _unsafe_view_51)
        _unsafe_view_52 = torch.ops.aten._unsafe_view.default(bmm_20, [64, 12, 128, 128]);  bmm_20 = None
        div_20 = torch.ops.aten.div.Tensor(_unsafe_view_52, 8.0);  _unsafe_view_52 = None
        add_134 = torch.ops.aten.add.Tensor(div_20, mul);  div_20 = None
        amax_10 = torch.ops.aten.amax.default(add_134, [-1], True)
        sub_42 = torch.ops.aten.sub.Tensor(add_134, amax_10);  add_134 = amax_10 = None
        exp_20 = torch.ops.aten.exp.default(sub_42);  sub_42 = None
        sum_11 = torch.ops.aten.sum.dim_IntList(exp_20, [-1], True)
        div_21 = torch.ops.aten.div.Tensor(exp_20, sum_11);  exp_20 = sum_11 = None
        alias_64 = torch.ops.aten.alias.default(div_21)
        alias_65 = torch.ops.aten.alias.default(alias_64);  alias_64 = None
        rand_like_31 = torch.ops.aten.rand_like.default(div_21, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_66 = torch.ops.aten.alias.default(rand_like_31);  rand_like_31 = None
        gt_31 = torch.ops.aten.gt.Scalar(alias_66, 0.1);  alias_66 = None
        mul_235 = torch.ops.aten.mul.Tensor(gt_31, div_21);  div_21 = None
        mul_236 = torch.ops.aten.mul.Tensor(mul_235, 1.1111111111111112);  mul_235 = None
        expand_43 = torch.ops.aten.expand.default(mul_236, [64, 12, 128, 128]);  mul_236 = None
        view_179 = torch.ops.aten.view.default(expand_43, [768, 128, 128]);  expand_43 = None
        expand_44 = torch.ops.aten.expand.default(permute_114, [64, 12, 128, 64]);  permute_114 = None
        clone_42 = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
        _unsafe_view_53 = torch.ops.aten._unsafe_view.default(clone_42, [768, 128, 64]);  clone_42 = None
        bmm_21 = torch.ops.aten.bmm.default(view_179, _unsafe_view_53)
        _unsafe_view_54 = torch.ops.aten._unsafe_view.default(bmm_21, [64, 12, 128, 64]);  bmm_21 = None
        permute_117 = torch.ops.aten.permute.default(_unsafe_view_54, [0, 2, 1, 3]);  _unsafe_view_54 = None
        clone_43 = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None
        view_180 = torch.ops.aten.view.default(clone_43, [64, 128, 768]);  clone_43 = None
        permute_118 = torch.ops.aten.permute.default(primals_172, [1, 0]);  primals_172 = None
        view_181 = torch.ops.aten.view.default(view_180, [8192, 768]);  view_180 = None
        addmm_63 = torch.ops.aten.addmm.default(primals_173, view_181, permute_118);  primals_173 = None
        view_182 = torch.ops.aten.view.default(addmm_63, [64, 128, 768]);  addmm_63 = None
        rand_like_32 = torch.ops.aten.rand_like.default(view_182, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_67 = torch.ops.aten.alias.default(rand_like_32);  rand_like_32 = None
        gt_32 = torch.ops.aten.gt.Scalar(alias_67, 0.1);  alias_67 = None
        mul_237 = torch.ops.aten.mul.Tensor(gt_32, view_182);  view_182 = None
        mul_238 = torch.ops.aten.mul.Tensor(mul_237, 1.1111111111111112);  mul_237 = None
        add_135 = torch.ops.aten.add.Tensor(mul_238, convert_element_type_20);  mul_238 = convert_element_type_20 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(add_135, [2], correction = 0, keepdim = True)
        getitem_42 = var_mean_21[0]
        getitem_43 = var_mean_21[1];  var_mean_21 = None
        add_136 = torch.ops.aten.add.Tensor(getitem_42, 1e-12);  getitem_42 = None
        sqrt_21 = torch.ops.aten.sqrt.default(add_136);  add_136 = None
        reciprocal_31 = torch.ops.aten.reciprocal.default(sqrt_21);  sqrt_21 = None
        sub_43 = torch.ops.aten.sub.Tensor(add_135, getitem_43);  add_135 = getitem_43 = None
        mul_239 = torch.ops.aten.mul.Tensor(sub_43, reciprocal_31);  sub_43 = None
        mul_240 = torch.ops.aten.mul.Tensor(mul_239, primals_174)
        add_137 = torch.ops.aten.add.Tensor(mul_240, primals_175);  mul_240 = primals_175 = None
        convert_element_type_21 = torch.ops.prims.convert_element_type.default(add_137, torch.float32);  add_137 = None
        permute_119 = torch.ops.aten.permute.default(primals_176, [1, 0]);  primals_176 = None
        view_183 = torch.ops.aten.view.default(convert_element_type_21, [8192, 768])
        addmm_64 = torch.ops.aten.addmm.default(primals_177, view_183, permute_119);  primals_177 = None
        view_184 = torch.ops.aten.view.default(addmm_64, [64, 128, 3072]);  addmm_64 = None
        mul_241 = torch.ops.aten.mul.Tensor(view_184, 0.5)
        mul_242 = torch.ops.aten.mul.Tensor(view_184, 0.7071067811865476)
        sign_10 = torch.ops.aten.sign.default(mul_242)
        abs_11 = torch.ops.aten.abs.default(mul_242);  mul_242 = None
        mul_243 = torch.ops.aten.mul.Tensor(abs_11, 0.3275911)
        add_138 = torch.ops.aten.add.Tensor(mul_243, 1.0);  mul_243 = None
        reciprocal_32 = torch.ops.aten.reciprocal.default(add_138);  add_138 = None
        mul_244 = torch.ops.aten.mul.Tensor(reciprocal_32, 1.0);  reciprocal_32 = None
        mul_245 = torch.ops.aten.mul.Tensor(mul_244, 1.061405429)
        add_139 = torch.ops.aten.add.Tensor(mul_245, -1.453152027);  mul_245 = None
        mul_246 = torch.ops.aten.mul.Tensor(add_139, mul_244);  add_139 = None
        add_140 = torch.ops.aten.add.Tensor(mul_246, 1.421413741);  mul_246 = None
        mul_247 = torch.ops.aten.mul.Tensor(add_140, mul_244);  add_140 = None
        add_141 = torch.ops.aten.add.Tensor(mul_247, -0.284496736);  mul_247 = None
        mul_248 = torch.ops.aten.mul.Tensor(add_141, mul_244);  add_141 = None
        add_142 = torch.ops.aten.add.Tensor(mul_248, 0.254829592);  mul_248 = None
        mul_249 = torch.ops.aten.mul.Tensor(add_142, mul_244);  add_142 = mul_244 = None
        neg_10 = torch.ops.aten.neg.default(abs_11)
        mul_250 = torch.ops.aten.mul.Tensor(neg_10, abs_11);  neg_10 = abs_11 = None
        exp_21 = torch.ops.aten.exp.default(mul_250);  mul_250 = None
        mul_251 = torch.ops.aten.mul.Tensor(mul_249, exp_21);  mul_249 = exp_21 = None
        _tensor_constant11 = self._tensor_constant11
        lift_fresh_copy_11 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant11);  _tensor_constant11 = None
        sub_44 = torch.ops.aten.sub.Tensor(lift_fresh_copy_11, mul_251);  lift_fresh_copy_11 = None
        mul_252 = torch.ops.aten.mul.Tensor(sign_10, sub_44);  sub_44 = None
        add_143 = torch.ops.aten.add.Tensor(mul_252, 1);  mul_252 = None
        mul_253 = torch.ops.aten.mul.Tensor(mul_241, add_143);  mul_241 = add_143 = None
        permute_120 = torch.ops.aten.permute.default(primals_178, [1, 0]);  primals_178 = None
        view_185 = torch.ops.aten.view.default(mul_253, [8192, 3072]);  mul_253 = None
        addmm_65 = torch.ops.aten.addmm.default(primals_179, view_185, permute_120);  primals_179 = None
        view_186 = torch.ops.aten.view.default(addmm_65, [64, 128, 768]);  addmm_65 = None
        rand_like_33 = torch.ops.aten.rand_like.default(view_186, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_68 = torch.ops.aten.alias.default(rand_like_33);  rand_like_33 = None
        gt_33 = torch.ops.aten.gt.Scalar(alias_68, 0.1);  alias_68 = None
        mul_254 = torch.ops.aten.mul.Tensor(gt_33, view_186);  view_186 = None
        mul_255 = torch.ops.aten.mul.Tensor(mul_254, 1.1111111111111112);  mul_254 = None
        add_144 = torch.ops.aten.add.Tensor(mul_255, convert_element_type_21);  mul_255 = convert_element_type_21 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(add_144, [2], correction = 0, keepdim = True)
        getitem_44 = var_mean_22[0]
        getitem_45 = var_mean_22[1];  var_mean_22 = None
        add_145 = torch.ops.aten.add.Tensor(getitem_44, 1e-12);  getitem_44 = None
        sqrt_22 = torch.ops.aten.sqrt.default(add_145);  add_145 = None
        reciprocal_33 = torch.ops.aten.reciprocal.default(sqrt_22);  sqrt_22 = None
        sub_45 = torch.ops.aten.sub.Tensor(add_144, getitem_45);  add_144 = getitem_45 = None
        mul_256 = torch.ops.aten.mul.Tensor(sub_45, reciprocal_33);  sub_45 = None
        mul_257 = torch.ops.aten.mul.Tensor(mul_256, primals_180)
        add_146 = torch.ops.aten.add.Tensor(mul_257, primals_181);  mul_257 = primals_181 = None
        convert_element_type_22 = torch.ops.prims.convert_element_type.default(add_146, torch.float32);  add_146 = None
        permute_121 = torch.ops.aten.permute.default(primals_182, [1, 0]);  primals_182 = None
        view_187 = torch.ops.aten.view.default(convert_element_type_22, [8192, 768])
        addmm_66 = torch.ops.aten.addmm.default(primals_183, view_187, permute_121);  primals_183 = None
        view_188 = torch.ops.aten.view.default(addmm_66, [64, 128, 768]);  addmm_66 = None
        permute_122 = torch.ops.aten.permute.default(primals_184, [1, 0]);  primals_184 = None
        addmm_67 = torch.ops.aten.addmm.default(primals_185, view_187, permute_122);  primals_185 = None
        view_190 = torch.ops.aten.view.default(addmm_67, [64, 128, 768]);  addmm_67 = None
        view_191 = torch.ops.aten.view.default(view_190, [64, 128, 12, 64]);  view_190 = None
        permute_123 = torch.ops.aten.permute.default(view_191, [0, 2, 1, 3]);  view_191 = None
        permute_124 = torch.ops.aten.permute.default(primals_186, [1, 0]);  primals_186 = None
        addmm_68 = torch.ops.aten.addmm.default(primals_187, view_187, permute_124);  primals_187 = None
        view_193 = torch.ops.aten.view.default(addmm_68, [64, 128, 768]);  addmm_68 = None
        view_194 = torch.ops.aten.view.default(view_193, [64, 128, 12, 64]);  view_193 = None
        permute_125 = torch.ops.aten.permute.default(view_194, [0, 2, 1, 3]);  view_194 = None
        view_195 = torch.ops.aten.view.default(view_188, [64, 128, 12, 64]);  view_188 = None
        permute_126 = torch.ops.aten.permute.default(view_195, [0, 2, 1, 3]);  view_195 = None
        permute_127 = torch.ops.aten.permute.default(permute_123, [0, 1, 3, 2]);  permute_123 = None
        expand_45 = torch.ops.aten.expand.default(permute_126, [64, 12, 128, 64]);  permute_126 = None
        clone_44 = torch.ops.aten.clone.default(expand_45, memory_format = torch.contiguous_format);  expand_45 = None
        _unsafe_view_55 = torch.ops.aten._unsafe_view.default(clone_44, [768, 128, 64]);  clone_44 = None
        expand_46 = torch.ops.aten.expand.default(permute_127, [64, 12, 64, 128]);  permute_127 = None
        clone_45 = torch.ops.aten.clone.default(expand_46, memory_format = torch.contiguous_format);  expand_46 = None
        _unsafe_view_56 = torch.ops.aten._unsafe_view.default(clone_45, [768, 64, 128]);  clone_45 = None
        bmm_22 = torch.ops.aten.bmm.default(_unsafe_view_55, _unsafe_view_56)
        _unsafe_view_57 = torch.ops.aten._unsafe_view.default(bmm_22, [64, 12, 128, 128]);  bmm_22 = None
        div_22 = torch.ops.aten.div.Tensor(_unsafe_view_57, 8.0);  _unsafe_view_57 = None
        add_147 = torch.ops.aten.add.Tensor(div_22, mul);  div_22 = mul = None
        amax_11 = torch.ops.aten.amax.default(add_147, [-1], True)
        sub_46 = torch.ops.aten.sub.Tensor(add_147, amax_11);  add_147 = amax_11 = None
        exp_22 = torch.ops.aten.exp.default(sub_46);  sub_46 = None
        sum_12 = torch.ops.aten.sum.dim_IntList(exp_22, [-1], True)
        div_23 = torch.ops.aten.div.Tensor(exp_22, sum_12);  exp_22 = sum_12 = None
        alias_70 = torch.ops.aten.alias.default(div_23)
        alias_71 = torch.ops.aten.alias.default(alias_70);  alias_70 = None
        rand_like_34 = torch.ops.aten.rand_like.default(div_23, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_72 = torch.ops.aten.alias.default(rand_like_34);  rand_like_34 = None
        gt_34 = torch.ops.aten.gt.Scalar(alias_72, 0.1);  alias_72 = None
        mul_258 = torch.ops.aten.mul.Tensor(gt_34, div_23);  div_23 = None
        mul_259 = torch.ops.aten.mul.Tensor(mul_258, 1.1111111111111112);  mul_258 = None
        expand_47 = torch.ops.aten.expand.default(mul_259, [64, 12, 128, 128]);  mul_259 = None
        view_196 = torch.ops.aten.view.default(expand_47, [768, 128, 128]);  expand_47 = None
        expand_48 = torch.ops.aten.expand.default(permute_125, [64, 12, 128, 64]);  permute_125 = None
        clone_46 = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
        _unsafe_view_58 = torch.ops.aten._unsafe_view.default(clone_46, [768, 128, 64]);  clone_46 = None
        bmm_23 = torch.ops.aten.bmm.default(view_196, _unsafe_view_58)
        _unsafe_view_59 = torch.ops.aten._unsafe_view.default(bmm_23, [64, 12, 128, 64]);  bmm_23 = None
        permute_128 = torch.ops.aten.permute.default(_unsafe_view_59, [0, 2, 1, 3]);  _unsafe_view_59 = None
        clone_47 = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None
        view_197 = torch.ops.aten.view.default(clone_47, [64, 128, 768]);  clone_47 = None
        permute_129 = torch.ops.aten.permute.default(primals_188, [1, 0]);  primals_188 = None
        view_198 = torch.ops.aten.view.default(view_197, [8192, 768]);  view_197 = None
        addmm_69 = torch.ops.aten.addmm.default(primals_189, view_198, permute_129);  primals_189 = None
        view_199 = torch.ops.aten.view.default(addmm_69, [64, 128, 768]);  addmm_69 = None
        rand_like_35 = torch.ops.aten.rand_like.default(view_199, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_73 = torch.ops.aten.alias.default(rand_like_35);  rand_like_35 = None
        gt_35 = torch.ops.aten.gt.Scalar(alias_73, 0.1);  alias_73 = None
        mul_260 = torch.ops.aten.mul.Tensor(gt_35, view_199);  view_199 = None
        mul_261 = torch.ops.aten.mul.Tensor(mul_260, 1.1111111111111112);  mul_260 = None
        add_148 = torch.ops.aten.add.Tensor(mul_261, convert_element_type_22);  mul_261 = convert_element_type_22 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(add_148, [2], correction = 0, keepdim = True)
        getitem_46 = var_mean_23[0]
        getitem_47 = var_mean_23[1];  var_mean_23 = None
        add_149 = torch.ops.aten.add.Tensor(getitem_46, 1e-12);  getitem_46 = None
        sqrt_23 = torch.ops.aten.sqrt.default(add_149);  add_149 = None
        reciprocal_34 = torch.ops.aten.reciprocal.default(sqrt_23);  sqrt_23 = None
        sub_47 = torch.ops.aten.sub.Tensor(add_148, getitem_47);  add_148 = getitem_47 = None
        mul_262 = torch.ops.aten.mul.Tensor(sub_47, reciprocal_34);  sub_47 = None
        mul_263 = torch.ops.aten.mul.Tensor(mul_262, primals_190)
        add_150 = torch.ops.aten.add.Tensor(mul_263, primals_191);  mul_263 = primals_191 = None
        convert_element_type_23 = torch.ops.prims.convert_element_type.default(add_150, torch.float32);  add_150 = None
        permute_130 = torch.ops.aten.permute.default(primals_192, [1, 0]);  primals_192 = None
        view_200 = torch.ops.aten.view.default(convert_element_type_23, [8192, 768])
        addmm_70 = torch.ops.aten.addmm.default(primals_193, view_200, permute_130);  primals_193 = None
        view_201 = torch.ops.aten.view.default(addmm_70, [64, 128, 3072]);  addmm_70 = None
        mul_264 = torch.ops.aten.mul.Tensor(view_201, 0.5)
        mul_265 = torch.ops.aten.mul.Tensor(view_201, 0.7071067811865476)
        sign_11 = torch.ops.aten.sign.default(mul_265)
        abs_12 = torch.ops.aten.abs.default(mul_265);  mul_265 = None
        mul_266 = torch.ops.aten.mul.Tensor(abs_12, 0.3275911)
        add_151 = torch.ops.aten.add.Tensor(mul_266, 1.0);  mul_266 = None
        reciprocal_35 = torch.ops.aten.reciprocal.default(add_151);  add_151 = None
        mul_267 = torch.ops.aten.mul.Tensor(reciprocal_35, 1.0);  reciprocal_35 = None
        mul_268 = torch.ops.aten.mul.Tensor(mul_267, 1.061405429)
        add_152 = torch.ops.aten.add.Tensor(mul_268, -1.453152027);  mul_268 = None
        mul_269 = torch.ops.aten.mul.Tensor(add_152, mul_267);  add_152 = None
        add_153 = torch.ops.aten.add.Tensor(mul_269, 1.421413741);  mul_269 = None
        mul_270 = torch.ops.aten.mul.Tensor(add_153, mul_267);  add_153 = None
        add_154 = torch.ops.aten.add.Tensor(mul_270, -0.284496736);  mul_270 = None
        mul_271 = torch.ops.aten.mul.Tensor(add_154, mul_267);  add_154 = None
        add_155 = torch.ops.aten.add.Tensor(mul_271, 0.254829592);  mul_271 = None
        mul_272 = torch.ops.aten.mul.Tensor(add_155, mul_267);  add_155 = mul_267 = None
        neg_11 = torch.ops.aten.neg.default(abs_12)
        mul_273 = torch.ops.aten.mul.Tensor(neg_11, abs_12);  neg_11 = abs_12 = None
        exp_23 = torch.ops.aten.exp.default(mul_273);  mul_273 = None
        mul_274 = torch.ops.aten.mul.Tensor(mul_272, exp_23);  mul_272 = exp_23 = None
        _tensor_constant12 = self._tensor_constant12
        lift_fresh_copy_12 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant12);  _tensor_constant12 = None
        sub_48 = torch.ops.aten.sub.Tensor(lift_fresh_copy_12, mul_274);  lift_fresh_copy_12 = None
        mul_275 = torch.ops.aten.mul.Tensor(sign_11, sub_48);  sub_48 = None
        add_156 = torch.ops.aten.add.Tensor(mul_275, 1);  mul_275 = None
        mul_276 = torch.ops.aten.mul.Tensor(mul_264, add_156);  mul_264 = add_156 = None
        permute_131 = torch.ops.aten.permute.default(primals_194, [1, 0]);  primals_194 = None
        view_202 = torch.ops.aten.view.default(mul_276, [8192, 3072]);  mul_276 = None
        addmm_71 = torch.ops.aten.addmm.default(primals_195, view_202, permute_131);  primals_195 = None
        view_203 = torch.ops.aten.view.default(addmm_71, [64, 128, 768]);  addmm_71 = None
        rand_like_36 = torch.ops.aten.rand_like.default(view_203, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_74 = torch.ops.aten.alias.default(rand_like_36);  rand_like_36 = None
        gt_36 = torch.ops.aten.gt.Scalar(alias_74, 0.1);  alias_74 = None
        mul_277 = torch.ops.aten.mul.Tensor(gt_36, view_203);  view_203 = None
        mul_278 = torch.ops.aten.mul.Tensor(mul_277, 1.1111111111111112);  mul_277 = None
        add_157 = torch.ops.aten.add.Tensor(mul_278, convert_element_type_23);  mul_278 = convert_element_type_23 = None
        var_mean_24 = torch.ops.aten.var_mean.correction(add_157, [2], correction = 0, keepdim = True)
        getitem_48 = var_mean_24[0]
        getitem_49 = var_mean_24[1];  var_mean_24 = None
        add_158 = torch.ops.aten.add.Tensor(getitem_48, 1e-12);  getitem_48 = None
        sqrt_24 = torch.ops.aten.sqrt.default(add_158);  add_158 = None
        reciprocal_36 = torch.ops.aten.reciprocal.default(sqrt_24);  sqrt_24 = None
        sub_49 = torch.ops.aten.sub.Tensor(add_157, getitem_49);  add_157 = getitem_49 = None
        mul_279 = torch.ops.aten.mul.Tensor(sub_49, reciprocal_36);  sub_49 = None
        mul_280 = torch.ops.aten.mul.Tensor(mul_279, primals_196)
        add_159 = torch.ops.aten.add.Tensor(mul_280, primals_197);  mul_280 = primals_197 = None
        convert_element_type_24 = torch.ops.prims.convert_element_type.default(add_159, torch.float32);  add_159 = None
        permute_132 = torch.ops.aten.permute.default(primals_198, [1, 0]);  primals_198 = None
        view_204 = torch.ops.aten.view.default(convert_element_type_24, [8192, 768]);  convert_element_type_24 = None
        addmm_72 = torch.ops.aten.addmm.default(primals_199, view_204, permute_132);  primals_199 = None
        view_205 = torch.ops.aten.view.default(addmm_72, [64, 128, 768]);  addmm_72 = None
        mul_281 = torch.ops.aten.mul.Tensor(view_205, 0.5)
        mul_282 = torch.ops.aten.mul.Tensor(view_205, 0.7071067811865476)
        sign_12 = torch.ops.aten.sign.default(mul_282)
        abs_13 = torch.ops.aten.abs.default(mul_282);  mul_282 = None
        mul_283 = torch.ops.aten.mul.Tensor(abs_13, 0.3275911)
        add_160 = torch.ops.aten.add.Tensor(mul_283, 1.0);  mul_283 = None
        reciprocal_37 = torch.ops.aten.reciprocal.default(add_160);  add_160 = None
        mul_284 = torch.ops.aten.mul.Tensor(reciprocal_37, 1.0);  reciprocal_37 = None
        mul_285 = torch.ops.aten.mul.Tensor(mul_284, 1.061405429)
        add_161 = torch.ops.aten.add.Tensor(mul_285, -1.453152027);  mul_285 = None
        mul_286 = torch.ops.aten.mul.Tensor(add_161, mul_284);  add_161 = None
        add_162 = torch.ops.aten.add.Tensor(mul_286, 1.421413741);  mul_286 = None
        mul_287 = torch.ops.aten.mul.Tensor(add_162, mul_284);  add_162 = None
        add_163 = torch.ops.aten.add.Tensor(mul_287, -0.284496736);  mul_287 = None
        mul_288 = torch.ops.aten.mul.Tensor(add_163, mul_284);  add_163 = None
        add_164 = torch.ops.aten.add.Tensor(mul_288, 0.254829592);  mul_288 = None
        mul_289 = torch.ops.aten.mul.Tensor(add_164, mul_284);  add_164 = mul_284 = None
        neg_12 = torch.ops.aten.neg.default(abs_13)
        mul_290 = torch.ops.aten.mul.Tensor(neg_12, abs_13);  neg_12 = abs_13 = None
        exp_24 = torch.ops.aten.exp.default(mul_290);  mul_290 = None
        mul_291 = torch.ops.aten.mul.Tensor(mul_289, exp_24);  mul_289 = exp_24 = None
        _tensor_constant13 = self._tensor_constant13
        lift_fresh_copy_13 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant13);  _tensor_constant13 = None
        sub_50 = torch.ops.aten.sub.Tensor(lift_fresh_copy_13, mul_291);  lift_fresh_copy_13 = None
        mul_292 = torch.ops.aten.mul.Tensor(sign_12, sub_50);  sub_50 = None
        add_165 = torch.ops.aten.add.Tensor(mul_292, 1);  mul_292 = None
        mul_293 = torch.ops.aten.mul.Tensor(mul_281, add_165);  mul_281 = add_165 = None
        var_mean_25 = torch.ops.aten.var_mean.correction(mul_293, [2], correction = 0, keepdim = True)
        getitem_50 = var_mean_25[0]
        getitem_51 = var_mean_25[1];  var_mean_25 = None
        add_166 = torch.ops.aten.add.Tensor(getitem_50, 1e-12);  getitem_50 = None
        sqrt_25 = torch.ops.aten.sqrt.default(add_166);  add_166 = None
        reciprocal_38 = torch.ops.aten.reciprocal.default(sqrt_25);  sqrt_25 = None
        sub_51 = torch.ops.aten.sub.Tensor(mul_293, getitem_51);  mul_293 = getitem_51 = None
        mul_294 = torch.ops.aten.mul.Tensor(sub_51, reciprocal_38);  sub_51 = None
        mul_295 = torch.ops.aten.mul.Tensor(mul_294, primals_200)
        add_167 = torch.ops.aten.add.Tensor(mul_295, primals_201);  mul_295 = primals_201 = None
        convert_element_type_25 = torch.ops.prims.convert_element_type.default(add_167, torch.float32);  add_167 = None
        permute_133 = torch.ops.aten.permute.default(primals_1, [1, 0]);  primals_1 = None
        view_206 = torch.ops.aten.view.default(convert_element_type_25, [8192, 768]);  convert_element_type_25 = None
        addmm_73 = torch.ops.aten.addmm.default(primals_202, view_206, permute_133);  primals_202 = None
        view_207 = torch.ops.aten.view.default(addmm_73, [64, 128, 30522]);  addmm_73 = None
        view_208 = torch.ops.aten.view.default(view_207, [-1, 30522])
        view_209 = torch.ops.aten.view.default(primals_206, [-1]);  primals_206 = None
        amax_12 = torch.ops.aten.amax.default(view_208, [1], True)
        sub_52 = torch.ops.aten.sub.Tensor(view_208, amax_12);  view_208 = amax_12 = None
        exp_25 = torch.ops.aten.exp.default(sub_52)
        sum_13 = torch.ops.aten.sum.dim_IntList(exp_25, [1], True);  exp_25 = None
        log = torch.ops.aten.log.default(sum_13);  sum_13 = None
        sub_53 = torch.ops.aten.sub.Tensor(sub_52, log);  sub_52 = log = None
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(view_209, 1);  view_209 = None
        gather = torch.ops.aten.gather.default(sub_53, 1, unsqueeze_2)
        squeeze = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg_13 = torch.ops.aten.neg.default(squeeze);  squeeze = None
        mean = torch.ops.aten.mean.default(neg_13);  neg_13 = None
        permute_134 = torch.ops.aten.permute.default(permute_133, [1, 0]);  permute_133 = None
        div_25 = torch.ops.aten.div.Tensor(reciprocal_38, 768);  reciprocal_38 = None
        _tensor_constant14 = self._tensor_constant14
        lift_fresh_copy_14 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant14);  _tensor_constant14 = None
        sub_58 = torch.ops.aten.sub.Tensor(lift_fresh_copy_14, mul_291);  lift_fresh_copy_14 = mul_291 = None
        mul_315 = torch.ops.aten.mul.Tensor(sign_12, sub_58);  sign_12 = sub_58 = None
        add_174 = torch.ops.aten.add.Tensor(mul_315, 1);  mul_315 = None
        mul_316 = torch.ops.aten.mul.Tensor(add_174, 0.5);  add_174 = None
        mul_317 = torch.ops.aten.mul.Tensor(view_205, view_205)
        mul_318 = torch.ops.aten.mul.Tensor(mul_317, -0.5);  mul_317 = None
        exp_28 = torch.ops.aten.exp.default(mul_318);  mul_318 = None
        mul_319 = torch.ops.aten.mul.Tensor(exp_28, 0.3989422804014327);  exp_28 = None
        mul_320 = torch.ops.aten.mul.Tensor(view_205, mul_319);  view_205 = mul_319 = None
        add_175 = torch.ops.aten.add.Tensor(mul_316, mul_320);  mul_316 = mul_320 = None
        permute_138 = torch.ops.aten.permute.default(permute_132, [1, 0]);  permute_132 = None
        div_26 = torch.ops.aten.div.Tensor(reciprocal_36, 768);  reciprocal_36 = None
        permute_142 = torch.ops.aten.permute.default(permute_131, [1, 0]);  permute_131 = None
        _tensor_constant15 = self._tensor_constant15
        lift_fresh_copy_15 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant15);  _tensor_constant15 = None
        sub_62 = torch.ops.aten.sub.Tensor(lift_fresh_copy_15, mul_274);  lift_fresh_copy_15 = mul_274 = None
        mul_341 = torch.ops.aten.mul.Tensor(sign_11, sub_62);  sign_11 = sub_62 = None
        add_181 = torch.ops.aten.add.Tensor(mul_341, 1);  mul_341 = None
        mul_342 = torch.ops.aten.mul.Tensor(add_181, 0.5);  add_181 = None
        mul_343 = torch.ops.aten.mul.Tensor(view_201, view_201)
        mul_344 = torch.ops.aten.mul.Tensor(mul_343, -0.5);  mul_343 = None
        exp_30 = torch.ops.aten.exp.default(mul_344);  mul_344 = None
        mul_345 = torch.ops.aten.mul.Tensor(exp_30, 0.3989422804014327);  exp_30 = None
        mul_346 = torch.ops.aten.mul.Tensor(view_201, mul_345);  view_201 = mul_345 = None
        add_182 = torch.ops.aten.add.Tensor(mul_342, mul_346);  mul_342 = mul_346 = None
        permute_146 = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None
        div_27 = torch.ops.aten.div.Tensor(reciprocal_34, 768);  reciprocal_34 = None
        permute_150 = torch.ops.aten.permute.default(permute_129, [1, 0]);  permute_129 = None
        permute_155 = torch.ops.aten.permute.default(view_196, [0, 2, 1]);  view_196 = None
        permute_156 = torch.ops.aten.permute.default(_unsafe_view_58, [0, 2, 1]);  _unsafe_view_58 = None
        alias_82 = torch.ops.aten.alias.default(alias_71);  alias_71 = None
        alias_83 = torch.ops.aten.alias.default(alias_82);  alias_82 = None
        permute_157 = torch.ops.aten.permute.default(_unsafe_view_55, [0, 2, 1]);  _unsafe_view_55 = None
        permute_158 = torch.ops.aten.permute.default(_unsafe_view_56, [0, 2, 1]);  _unsafe_view_56 = None
        permute_162 = torch.ops.aten.permute.default(permute_124, [1, 0]);  permute_124 = None
        permute_167 = torch.ops.aten.permute.default(permute_122, [1, 0]);  permute_122 = None
        permute_171 = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None
        div_29 = torch.ops.aten.div.Tensor(reciprocal_33, 768);  reciprocal_33 = None
        permute_175 = torch.ops.aten.permute.default(permute_120, [1, 0]);  permute_120 = None
        _tensor_constant16 = self._tensor_constant16
        lift_fresh_copy_16 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant16);  _tensor_constant16 = None
        sub_70 = torch.ops.aten.sub.Tensor(lift_fresh_copy_16, mul_251);  lift_fresh_copy_16 = mul_251 = None
        mul_380 = torch.ops.aten.mul.Tensor(sign_10, sub_70);  sign_10 = sub_70 = None
        add_192 = torch.ops.aten.add.Tensor(mul_380, 1);  mul_380 = None
        mul_381 = torch.ops.aten.mul.Tensor(add_192, 0.5);  add_192 = None
        mul_382 = torch.ops.aten.mul.Tensor(view_184, view_184)
        mul_383 = torch.ops.aten.mul.Tensor(mul_382, -0.5);  mul_382 = None
        exp_32 = torch.ops.aten.exp.default(mul_383);  mul_383 = None
        mul_384 = torch.ops.aten.mul.Tensor(exp_32, 0.3989422804014327);  exp_32 = None
        mul_385 = torch.ops.aten.mul.Tensor(view_184, mul_384);  view_184 = mul_384 = None
        add_193 = torch.ops.aten.add.Tensor(mul_381, mul_385);  mul_381 = mul_385 = None
        permute_179 = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None
        div_30 = torch.ops.aten.div.Tensor(reciprocal_31, 768);  reciprocal_31 = None
        permute_183 = torch.ops.aten.permute.default(permute_118, [1, 0]);  permute_118 = None
        permute_188 = torch.ops.aten.permute.default(view_179, [0, 2, 1]);  view_179 = None
        permute_189 = torch.ops.aten.permute.default(_unsafe_view_53, [0, 2, 1]);  _unsafe_view_53 = None
        alias_84 = torch.ops.aten.alias.default(alias_65);  alias_65 = None
        alias_85 = torch.ops.aten.alias.default(alias_84);  alias_84 = None
        permute_190 = torch.ops.aten.permute.default(_unsafe_view_50, [0, 2, 1]);  _unsafe_view_50 = None
        permute_191 = torch.ops.aten.permute.default(_unsafe_view_51, [0, 2, 1]);  _unsafe_view_51 = None
        permute_195 = torch.ops.aten.permute.default(permute_113, [1, 0]);  permute_113 = None
        permute_200 = torch.ops.aten.permute.default(permute_111, [1, 0]);  permute_111 = None
        permute_204 = torch.ops.aten.permute.default(permute_110, [1, 0]);  permute_110 = None
        div_32 = torch.ops.aten.div.Tensor(reciprocal_30, 768);  reciprocal_30 = None
        permute_208 = torch.ops.aten.permute.default(permute_109, [1, 0]);  permute_109 = None
        _tensor_constant17 = self._tensor_constant17
        lift_fresh_copy_17 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant17);  _tensor_constant17 = None
        sub_78 = torch.ops.aten.sub.Tensor(lift_fresh_copy_17, mul_228);  lift_fresh_copy_17 = mul_228 = None
        mul_419 = torch.ops.aten.mul.Tensor(sign_9, sub_78);  sign_9 = sub_78 = None
        add_203 = torch.ops.aten.add.Tensor(mul_419, 1);  mul_419 = None
        mul_420 = torch.ops.aten.mul.Tensor(add_203, 0.5);  add_203 = None
        mul_421 = torch.ops.aten.mul.Tensor(view_167, view_167)
        mul_422 = torch.ops.aten.mul.Tensor(mul_421, -0.5);  mul_421 = None
        exp_34 = torch.ops.aten.exp.default(mul_422);  mul_422 = None
        mul_423 = torch.ops.aten.mul.Tensor(exp_34, 0.3989422804014327);  exp_34 = None
        mul_424 = torch.ops.aten.mul.Tensor(view_167, mul_423);  view_167 = mul_423 = None
        add_204 = torch.ops.aten.add.Tensor(mul_420, mul_424);  mul_420 = mul_424 = None
        permute_212 = torch.ops.aten.permute.default(permute_108, [1, 0]);  permute_108 = None
        div_33 = torch.ops.aten.div.Tensor(reciprocal_28, 768);  reciprocal_28 = None
        permute_216 = torch.ops.aten.permute.default(permute_107, [1, 0]);  permute_107 = None
        permute_221 = torch.ops.aten.permute.default(view_162, [0, 2, 1]);  view_162 = None
        permute_222 = torch.ops.aten.permute.default(_unsafe_view_48, [0, 2, 1]);  _unsafe_view_48 = None
        alias_86 = torch.ops.aten.alias.default(alias_59);  alias_59 = None
        alias_87 = torch.ops.aten.alias.default(alias_86);  alias_86 = None
        permute_223 = torch.ops.aten.permute.default(_unsafe_view_45, [0, 2, 1]);  _unsafe_view_45 = None
        permute_224 = torch.ops.aten.permute.default(_unsafe_view_46, [0, 2, 1]);  _unsafe_view_46 = None
        permute_228 = torch.ops.aten.permute.default(permute_102, [1, 0]);  permute_102 = None
        permute_233 = torch.ops.aten.permute.default(permute_100, [1, 0]);  permute_100 = None
        permute_237 = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None
        div_35 = torch.ops.aten.div.Tensor(reciprocal_27, 768);  reciprocal_27 = None
        permute_241 = torch.ops.aten.permute.default(permute_98, [1, 0]);  permute_98 = None
        _tensor_constant18 = self._tensor_constant18
        lift_fresh_copy_18 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant18);  _tensor_constant18 = None
        sub_86 = torch.ops.aten.sub.Tensor(lift_fresh_copy_18, mul_205);  lift_fresh_copy_18 = mul_205 = None
        mul_458 = torch.ops.aten.mul.Tensor(sign_8, sub_86);  sign_8 = sub_86 = None
        add_214 = torch.ops.aten.add.Tensor(mul_458, 1);  mul_458 = None
        mul_459 = torch.ops.aten.mul.Tensor(add_214, 0.5);  add_214 = None
        mul_460 = torch.ops.aten.mul.Tensor(view_150, view_150)
        mul_461 = torch.ops.aten.mul.Tensor(mul_460, -0.5);  mul_460 = None
        exp_36 = torch.ops.aten.exp.default(mul_461);  mul_461 = None
        mul_462 = torch.ops.aten.mul.Tensor(exp_36, 0.3989422804014327);  exp_36 = None
        mul_463 = torch.ops.aten.mul.Tensor(view_150, mul_462);  view_150 = mul_462 = None
        add_215 = torch.ops.aten.add.Tensor(mul_459, mul_463);  mul_459 = mul_463 = None
        permute_245 = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
        div_36 = torch.ops.aten.div.Tensor(reciprocal_25, 768);  reciprocal_25 = None
        permute_249 = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None
        permute_254 = torch.ops.aten.permute.default(view_145, [0, 2, 1]);  view_145 = None
        permute_255 = torch.ops.aten.permute.default(_unsafe_view_43, [0, 2, 1]);  _unsafe_view_43 = None
        alias_88 = torch.ops.aten.alias.default(alias_53);  alias_53 = None
        alias_89 = torch.ops.aten.alias.default(alias_88);  alias_88 = None
        permute_256 = torch.ops.aten.permute.default(_unsafe_view_40, [0, 2, 1]);  _unsafe_view_40 = None
        permute_257 = torch.ops.aten.permute.default(_unsafe_view_41, [0, 2, 1]);  _unsafe_view_41 = None
        permute_261 = torch.ops.aten.permute.default(permute_91, [1, 0]);  permute_91 = None
        permute_266 = torch.ops.aten.permute.default(permute_89, [1, 0]);  permute_89 = None
        permute_270 = torch.ops.aten.permute.default(permute_88, [1, 0]);  permute_88 = None
        div_38 = torch.ops.aten.div.Tensor(reciprocal_24, 768);  reciprocal_24 = None
        permute_274 = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None
        _tensor_constant19 = self._tensor_constant19
        lift_fresh_copy_19 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant19);  _tensor_constant19 = None
        sub_94 = torch.ops.aten.sub.Tensor(lift_fresh_copy_19, mul_182);  lift_fresh_copy_19 = mul_182 = None
        mul_497 = torch.ops.aten.mul.Tensor(sign_7, sub_94);  sign_7 = sub_94 = None
        add_225 = torch.ops.aten.add.Tensor(mul_497, 1);  mul_497 = None
        mul_498 = torch.ops.aten.mul.Tensor(add_225, 0.5);  add_225 = None
        mul_499 = torch.ops.aten.mul.Tensor(view_133, view_133)
        mul_500 = torch.ops.aten.mul.Tensor(mul_499, -0.5);  mul_499 = None
        exp_38 = torch.ops.aten.exp.default(mul_500);  mul_500 = None
        mul_501 = torch.ops.aten.mul.Tensor(exp_38, 0.3989422804014327);  exp_38 = None
        mul_502 = torch.ops.aten.mul.Tensor(view_133, mul_501);  view_133 = mul_501 = None
        add_226 = torch.ops.aten.add.Tensor(mul_498, mul_502);  mul_498 = mul_502 = None
        permute_278 = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None
        div_39 = torch.ops.aten.div.Tensor(reciprocal_22, 768);  reciprocal_22 = None
        permute_282 = torch.ops.aten.permute.default(permute_85, [1, 0]);  permute_85 = None
        permute_287 = torch.ops.aten.permute.default(view_128, [0, 2, 1]);  view_128 = None
        permute_288 = torch.ops.aten.permute.default(_unsafe_view_38, [0, 2, 1]);  _unsafe_view_38 = None
        alias_90 = torch.ops.aten.alias.default(alias_47);  alias_47 = None
        alias_91 = torch.ops.aten.alias.default(alias_90);  alias_90 = None
        permute_289 = torch.ops.aten.permute.default(_unsafe_view_35, [0, 2, 1]);  _unsafe_view_35 = None
        permute_290 = torch.ops.aten.permute.default(_unsafe_view_36, [0, 2, 1]);  _unsafe_view_36 = None
        permute_294 = torch.ops.aten.permute.default(permute_80, [1, 0]);  permute_80 = None
        permute_299 = torch.ops.aten.permute.default(permute_78, [1, 0]);  permute_78 = None
        permute_303 = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
        div_41 = torch.ops.aten.div.Tensor(reciprocal_21, 768);  reciprocal_21 = None
        permute_307 = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
        _tensor_constant20 = self._tensor_constant20
        lift_fresh_copy_20 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant20);  _tensor_constant20 = None
        sub_102 = torch.ops.aten.sub.Tensor(lift_fresh_copy_20, mul_159);  lift_fresh_copy_20 = mul_159 = None
        mul_536 = torch.ops.aten.mul.Tensor(sign_6, sub_102);  sign_6 = sub_102 = None
        add_236 = torch.ops.aten.add.Tensor(mul_536, 1);  mul_536 = None
        mul_537 = torch.ops.aten.mul.Tensor(add_236, 0.5);  add_236 = None
        mul_538 = torch.ops.aten.mul.Tensor(view_116, view_116)
        mul_539 = torch.ops.aten.mul.Tensor(mul_538, -0.5);  mul_538 = None
        exp_40 = torch.ops.aten.exp.default(mul_539);  mul_539 = None
        mul_540 = torch.ops.aten.mul.Tensor(exp_40, 0.3989422804014327);  exp_40 = None
        mul_541 = torch.ops.aten.mul.Tensor(view_116, mul_540);  view_116 = mul_540 = None
        add_237 = torch.ops.aten.add.Tensor(mul_537, mul_541);  mul_537 = mul_541 = None
        permute_311 = torch.ops.aten.permute.default(permute_75, [1, 0]);  permute_75 = None
        div_42 = torch.ops.aten.div.Tensor(reciprocal_19, 768);  reciprocal_19 = None
        permute_315 = torch.ops.aten.permute.default(permute_74, [1, 0]);  permute_74 = None
        permute_320 = torch.ops.aten.permute.default(view_111, [0, 2, 1]);  view_111 = None
        permute_321 = torch.ops.aten.permute.default(_unsafe_view_33, [0, 2, 1]);  _unsafe_view_33 = None
        alias_92 = torch.ops.aten.alias.default(alias_41);  alias_41 = None
        alias_93 = torch.ops.aten.alias.default(alias_92);  alias_92 = None
        permute_322 = torch.ops.aten.permute.default(_unsafe_view_30, [0, 2, 1]);  _unsafe_view_30 = None
        permute_323 = torch.ops.aten.permute.default(_unsafe_view_31, [0, 2, 1]);  _unsafe_view_31 = None
        permute_327 = torch.ops.aten.permute.default(permute_69, [1, 0]);  permute_69 = None
        permute_332 = torch.ops.aten.permute.default(permute_67, [1, 0]);  permute_67 = None
        permute_336 = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
        div_44 = torch.ops.aten.div.Tensor(reciprocal_18, 768);  reciprocal_18 = None
        permute_340 = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
        _tensor_constant21 = self._tensor_constant21
        lift_fresh_copy_21 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant21);  _tensor_constant21 = None
        sub_110 = torch.ops.aten.sub.Tensor(lift_fresh_copy_21, mul_136);  lift_fresh_copy_21 = mul_136 = None
        mul_575 = torch.ops.aten.mul.Tensor(sign_5, sub_110);  sign_5 = sub_110 = None
        add_247 = torch.ops.aten.add.Tensor(mul_575, 1);  mul_575 = None
        mul_576 = torch.ops.aten.mul.Tensor(add_247, 0.5);  add_247 = None
        mul_577 = torch.ops.aten.mul.Tensor(view_99, view_99)
        mul_578 = torch.ops.aten.mul.Tensor(mul_577, -0.5);  mul_577 = None
        exp_42 = torch.ops.aten.exp.default(mul_578);  mul_578 = None
        mul_579 = torch.ops.aten.mul.Tensor(exp_42, 0.3989422804014327);  exp_42 = None
        mul_580 = torch.ops.aten.mul.Tensor(view_99, mul_579);  view_99 = mul_579 = None
        add_248 = torch.ops.aten.add.Tensor(mul_576, mul_580);  mul_576 = mul_580 = None
        permute_344 = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
        div_45 = torch.ops.aten.div.Tensor(reciprocal_16, 768);  reciprocal_16 = None
        permute_348 = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
        permute_353 = torch.ops.aten.permute.default(view_94, [0, 2, 1]);  view_94 = None
        permute_354 = torch.ops.aten.permute.default(_unsafe_view_28, [0, 2, 1]);  _unsafe_view_28 = None
        alias_94 = torch.ops.aten.alias.default(alias_35);  alias_35 = None
        alias_95 = torch.ops.aten.alias.default(alias_94);  alias_94 = None
        permute_355 = torch.ops.aten.permute.default(_unsafe_view_25, [0, 2, 1]);  _unsafe_view_25 = None
        permute_356 = torch.ops.aten.permute.default(_unsafe_view_26, [0, 2, 1]);  _unsafe_view_26 = None
        permute_360 = torch.ops.aten.permute.default(permute_58, [1, 0]);  permute_58 = None
        permute_365 = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None
        permute_369 = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
        div_47 = torch.ops.aten.div.Tensor(reciprocal_15, 768);  reciprocal_15 = None
        permute_373 = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
        _tensor_constant22 = self._tensor_constant22
        lift_fresh_copy_22 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant22);  _tensor_constant22 = None
        sub_118 = torch.ops.aten.sub.Tensor(lift_fresh_copy_22, mul_113);  lift_fresh_copy_22 = mul_113 = None
        mul_614 = torch.ops.aten.mul.Tensor(sign_4, sub_118);  sign_4 = sub_118 = None
        add_258 = torch.ops.aten.add.Tensor(mul_614, 1);  mul_614 = None
        mul_615 = torch.ops.aten.mul.Tensor(add_258, 0.5);  add_258 = None
        mul_616 = torch.ops.aten.mul.Tensor(view_82, view_82)
        mul_617 = torch.ops.aten.mul.Tensor(mul_616, -0.5);  mul_616 = None
        exp_44 = torch.ops.aten.exp.default(mul_617);  mul_617 = None
        mul_618 = torch.ops.aten.mul.Tensor(exp_44, 0.3989422804014327);  exp_44 = None
        mul_619 = torch.ops.aten.mul.Tensor(view_82, mul_618);  view_82 = mul_618 = None
        add_259 = torch.ops.aten.add.Tensor(mul_615, mul_619);  mul_615 = mul_619 = None
        permute_377 = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
        div_48 = torch.ops.aten.div.Tensor(reciprocal_13, 768);  reciprocal_13 = None
        permute_381 = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None
        permute_386 = torch.ops.aten.permute.default(view_77, [0, 2, 1]);  view_77 = None
        permute_387 = torch.ops.aten.permute.default(_unsafe_view_23, [0, 2, 1]);  _unsafe_view_23 = None
        alias_96 = torch.ops.aten.alias.default(alias_29);  alias_29 = None
        alias_97 = torch.ops.aten.alias.default(alias_96);  alias_96 = None
        permute_388 = torch.ops.aten.permute.default(_unsafe_view_20, [0, 2, 1]);  _unsafe_view_20 = None
        permute_389 = torch.ops.aten.permute.default(_unsafe_view_21, [0, 2, 1]);  _unsafe_view_21 = None
        permute_393 = torch.ops.aten.permute.default(permute_47, [1, 0]);  permute_47 = None
        permute_398 = torch.ops.aten.permute.default(permute_45, [1, 0]);  permute_45 = None
        permute_402 = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
        div_50 = torch.ops.aten.div.Tensor(reciprocal_12, 768);  reciprocal_12 = None
        permute_406 = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
        _tensor_constant23 = self._tensor_constant23
        lift_fresh_copy_23 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant23);  _tensor_constant23 = None
        sub_126 = torch.ops.aten.sub.Tensor(lift_fresh_copy_23, mul_90);  lift_fresh_copy_23 = mul_90 = None
        mul_653 = torch.ops.aten.mul.Tensor(sign_3, sub_126);  sign_3 = sub_126 = None
        add_269 = torch.ops.aten.add.Tensor(mul_653, 1);  mul_653 = None
        mul_654 = torch.ops.aten.mul.Tensor(add_269, 0.5);  add_269 = None
        mul_655 = torch.ops.aten.mul.Tensor(view_65, view_65)
        mul_656 = torch.ops.aten.mul.Tensor(mul_655, -0.5);  mul_655 = None
        exp_46 = torch.ops.aten.exp.default(mul_656);  mul_656 = None
        mul_657 = torch.ops.aten.mul.Tensor(exp_46, 0.3989422804014327);  exp_46 = None
        mul_658 = torch.ops.aten.mul.Tensor(view_65, mul_657);  view_65 = mul_657 = None
        add_270 = torch.ops.aten.add.Tensor(mul_654, mul_658);  mul_654 = mul_658 = None
        permute_410 = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
        div_51 = torch.ops.aten.div.Tensor(reciprocal_10, 768);  reciprocal_10 = None
        permute_414 = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
        permute_419 = torch.ops.aten.permute.default(view_60, [0, 2, 1]);  view_60 = None
        permute_420 = torch.ops.aten.permute.default(_unsafe_view_18, [0, 2, 1]);  _unsafe_view_18 = None
        alias_98 = torch.ops.aten.alias.default(alias_23);  alias_23 = None
        alias_99 = torch.ops.aten.alias.default(alias_98);  alias_98 = None
        permute_421 = torch.ops.aten.permute.default(_unsafe_view_15, [0, 2, 1]);  _unsafe_view_15 = None
        permute_422 = torch.ops.aten.permute.default(_unsafe_view_16, [0, 2, 1]);  _unsafe_view_16 = None
        permute_426 = torch.ops.aten.permute.default(permute_36, [1, 0]);  permute_36 = None
        permute_431 = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
        permute_435 = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
        div_53 = torch.ops.aten.div.Tensor(reciprocal_9, 768);  reciprocal_9 = None
        permute_439 = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
        _tensor_constant24 = self._tensor_constant24
        lift_fresh_copy_24 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant24);  _tensor_constant24 = None
        sub_134 = torch.ops.aten.sub.Tensor(lift_fresh_copy_24, mul_67);  lift_fresh_copy_24 = mul_67 = None
        mul_692 = torch.ops.aten.mul.Tensor(sign_2, sub_134);  sign_2 = sub_134 = None
        add_280 = torch.ops.aten.add.Tensor(mul_692, 1);  mul_692 = None
        mul_693 = torch.ops.aten.mul.Tensor(add_280, 0.5);  add_280 = None
        mul_694 = torch.ops.aten.mul.Tensor(view_48, view_48)
        mul_695 = torch.ops.aten.mul.Tensor(mul_694, -0.5);  mul_694 = None
        exp_48 = torch.ops.aten.exp.default(mul_695);  mul_695 = None
        mul_696 = torch.ops.aten.mul.Tensor(exp_48, 0.3989422804014327);  exp_48 = None
        mul_697 = torch.ops.aten.mul.Tensor(view_48, mul_696);  view_48 = mul_696 = None
        add_281 = torch.ops.aten.add.Tensor(mul_693, mul_697);  mul_693 = mul_697 = None
        permute_443 = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
        div_54 = torch.ops.aten.div.Tensor(reciprocal_7, 768);  reciprocal_7 = None
        permute_447 = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
        permute_452 = torch.ops.aten.permute.default(view_43, [0, 2, 1]);  view_43 = None
        permute_453 = torch.ops.aten.permute.default(_unsafe_view_13, [0, 2, 1]);  _unsafe_view_13 = None
        alias_100 = torch.ops.aten.alias.default(alias_17);  alias_17 = None
        alias_101 = torch.ops.aten.alias.default(alias_100);  alias_100 = None
        permute_454 = torch.ops.aten.permute.default(_unsafe_view_10, [0, 2, 1]);  _unsafe_view_10 = None
        permute_455 = torch.ops.aten.permute.default(_unsafe_view_11, [0, 2, 1]);  _unsafe_view_11 = None
        permute_459 = torch.ops.aten.permute.default(permute_25, [1, 0]);  permute_25 = None
        permute_464 = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
        permute_468 = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
        div_56 = torch.ops.aten.div.Tensor(reciprocal_6, 768);  reciprocal_6 = None
        permute_472 = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
        _tensor_constant25 = self._tensor_constant25
        lift_fresh_copy_25 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant25);  _tensor_constant25 = None
        sub_142 = torch.ops.aten.sub.Tensor(lift_fresh_copy_25, mul_44);  lift_fresh_copy_25 = mul_44 = None
        mul_731 = torch.ops.aten.mul.Tensor(sign_1, sub_142);  sign_1 = sub_142 = None
        add_291 = torch.ops.aten.add.Tensor(mul_731, 1);  mul_731 = None
        mul_732 = torch.ops.aten.mul.Tensor(add_291, 0.5);  add_291 = None
        mul_733 = torch.ops.aten.mul.Tensor(view_31, view_31)
        mul_734 = torch.ops.aten.mul.Tensor(mul_733, -0.5);  mul_733 = None
        exp_50 = torch.ops.aten.exp.default(mul_734);  mul_734 = None
        mul_735 = torch.ops.aten.mul.Tensor(exp_50, 0.3989422804014327);  exp_50 = None
        mul_736 = torch.ops.aten.mul.Tensor(view_31, mul_735);  view_31 = mul_735 = None
        add_292 = torch.ops.aten.add.Tensor(mul_732, mul_736);  mul_732 = mul_736 = None
        permute_476 = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
        div_57 = torch.ops.aten.div.Tensor(reciprocal_4, 768);  reciprocal_4 = None
        permute_480 = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
        permute_485 = torch.ops.aten.permute.default(view_26, [0, 2, 1]);  view_26 = None
        permute_486 = torch.ops.aten.permute.default(_unsafe_view_8, [0, 2, 1]);  _unsafe_view_8 = None
        alias_102 = torch.ops.aten.alias.default(alias_11);  alias_11 = None
        alias_103 = torch.ops.aten.alias.default(alias_102);  alias_102 = None
        permute_487 = torch.ops.aten.permute.default(_unsafe_view_5, [0, 2, 1]);  _unsafe_view_5 = None
        permute_488 = torch.ops.aten.permute.default(_unsafe_view_6, [0, 2, 1]);  _unsafe_view_6 = None
        permute_492 = torch.ops.aten.permute.default(permute_14, [1, 0]);  permute_14 = None
        permute_497 = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
        permute_501 = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
        div_59 = torch.ops.aten.div.Tensor(reciprocal_3, 768);  reciprocal_3 = None
        permute_505 = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
        _tensor_constant26 = self._tensor_constant26
        lift_fresh_copy_26 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant26);  _tensor_constant26 = None
        sub_150 = torch.ops.aten.sub.Tensor(lift_fresh_copy_26, mul_21);  lift_fresh_copy_26 = mul_21 = None
        mul_770 = torch.ops.aten.mul.Tensor(sign, sub_150);  sign = sub_150 = None
        add_302 = torch.ops.aten.add.Tensor(mul_770, 1);  mul_770 = None
        mul_771 = torch.ops.aten.mul.Tensor(add_302, 0.5);  add_302 = None
        mul_772 = torch.ops.aten.mul.Tensor(view_14, view_14)
        mul_773 = torch.ops.aten.mul.Tensor(mul_772, -0.5);  mul_772 = None
        exp_52 = torch.ops.aten.exp.default(mul_773);  mul_773 = None
        mul_774 = torch.ops.aten.mul.Tensor(exp_52, 0.3989422804014327);  exp_52 = None
        mul_775 = torch.ops.aten.mul.Tensor(view_14, mul_774);  view_14 = mul_774 = None
        add_303 = torch.ops.aten.add.Tensor(mul_771, mul_775);  mul_771 = mul_775 = None
        permute_509 = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
        div_60 = torch.ops.aten.div.Tensor(reciprocal_1, 768);  reciprocal_1 = None
        permute_513 = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
        permute_518 = torch.ops.aten.permute.default(view_9, [0, 2, 1]);  view_9 = None
        permute_519 = torch.ops.aten.permute.default(_unsafe_view_3, [0, 2, 1]);  _unsafe_view_3 = None
        alias_104 = torch.ops.aten.alias.default(alias_5);  alias_5 = None
        alias_105 = torch.ops.aten.alias.default(alias_104);  alias_104 = None
        permute_520 = torch.ops.aten.permute.default(_unsafe_view, [0, 2, 1]);  _unsafe_view = None
        permute_521 = torch.ops.aten.permute.default(_unsafe_view_1, [0, 2, 1]);  _unsafe_view_1 = None
        permute_525 = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
        permute_530 = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        permute_534 = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        div_62 = torch.ops.aten.div.Tensor(reciprocal, 768);  reciprocal = None
        view_506 = torch.ops.aten.view.default(slice_6, [128]);  slice_6 = None
        view_509 = torch.ops.aten.view.default(primals_205, [8192]);  primals_205 = None
        return [mean, view_207, primals_4, primals_14, primals_20, primals_30, primals_36, primals_46, primals_52, primals_62, primals_68, primals_78, primals_84, primals_94, primals_100, primals_110, primals_116, primals_126, primals_132, primals_142, primals_148, primals_158, primals_164, primals_174, primals_180, primals_190, primals_196, primals_200, slice_2, mul_1, gt, view, gt_1, view_11, gt_2, mul_9, view_13, view_15, gt_3, mul_26, view_17, gt_4, view_28, gt_5, mul_32, view_30, view_32, gt_6, mul_49, view_34, gt_7, view_45, gt_8, mul_55, view_47, view_49, gt_9, mul_72, view_51, gt_10, view_62, gt_11, mul_78, view_64, view_66, gt_12, mul_95, view_68, gt_13, view_79, gt_14, mul_101, view_81, view_83, gt_15, mul_118, view_85, gt_16, view_96, gt_17, mul_124, view_98, view_100, gt_18, mul_141, view_102, gt_19, view_113, gt_20, mul_147, view_115, view_117, gt_21, mul_164, view_119, gt_22, view_130, gt_23, mul_170, view_132, view_134, gt_24, mul_187, view_136, gt_25, view_147, gt_26, mul_193, view_149, view_151, gt_27, mul_210, view_153, gt_28, view_164, gt_29, mul_216, view_166, view_168, gt_30, mul_233, view_170, gt_31, view_181, gt_32, mul_239, view_183, view_185, gt_33, mul_256, view_187, gt_34, view_198, gt_35, mul_262, view_200, view_202, gt_36, mul_279, view_204, mul_294, view_206, sub_53, unsqueeze_2, permute_134, div_25, add_175, permute_138, div_26, permute_142, add_182, permute_146, div_27, permute_150, permute_155, permute_156, alias_83, permute_157, permute_158, permute_162, permute_167, permute_171, div_29, permute_175, add_193, permute_179, div_30, permute_183, permute_188, permute_189, alias_85, permute_190, permute_191, permute_195, permute_200, permute_204, div_32, permute_208, add_204, permute_212, div_33, permute_216, permute_221, permute_222, alias_87, permute_223, permute_224, permute_228, permute_233, permute_237, div_35, permute_241, add_215, permute_245, div_36, permute_249, permute_254, permute_255, alias_89, permute_256, permute_257, permute_261, permute_266, permute_270, div_38, permute_274, add_226, permute_278, div_39, permute_282, permute_287, permute_288, alias_91, permute_289, permute_290, permute_294, permute_299, permute_303, div_41, permute_307, add_237, permute_311, div_42, permute_315, permute_320, permute_321, alias_93, permute_322, permute_323, permute_327, permute_332, permute_336, div_44, permute_340, add_248, permute_344, div_45, permute_348, permute_353, permute_354, alias_95, permute_355, permute_356, permute_360, permute_365, permute_369, div_47, permute_373, add_259, permute_377, div_48, permute_381, permute_386, permute_387, alias_97, permute_388, permute_389, permute_393, permute_398, permute_402, div_50, permute_406, add_270, permute_410, div_51, permute_414, permute_419, permute_420, alias_99, permute_421, permute_422, permute_426, permute_431, permute_435, div_53, permute_439, add_281, permute_443, div_54, permute_447, permute_452, permute_453, alias_101, permute_454, permute_455, permute_459, permute_464, permute_468, div_56, permute_472, add_292, permute_476, div_57, permute_480, permute_485, permute_486, alias_103, permute_487, permute_488, permute_492, permute_497, permute_501, div_59, permute_505, add_303, permute_509, div_60, permute_513, permute_518, permute_519, alias_105, permute_520, permute_521, permute_525, permute_530, permute_534, div_62, view_506, view_509]
        
args = [((30522, 768), (768, 1), torch.float32, 'cuda'), ((2, 768), (768, 1), torch.float32, 'cuda'), ((512, 768), (768, 1), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768, 768), (768, 1), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768, 768), (768, 1), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768, 768), (768, 1), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768, 768), (768, 1), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((3072, 768), (768, 1), torch.float32, 'cuda'), ((3072,), (1,), torch.float32, 'cuda'), ((768, 3072), (3072, 1), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768, 768), (768, 1), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768, 768), (768, 1), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768, 768), (768, 1), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768, 768), (768, 1), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((3072, 768), (768, 1), torch.float32, 'cuda'), ((3072,), (1,), torch.float32, 'cuda'), ((768, 3072), (3072, 1), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768, 768), (768, 1), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768, 768), (768, 1), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768, 768), (768, 1), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768, 768), (768, 1), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((3072, 768), (768, 1), torch.float32, 'cuda'), ((3072,), (1,), torch.float32, 'cuda'), ((768, 3072), (3072, 1), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768, 768), (768, 1), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768, 768), (768, 1), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768, 768), (768, 1), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768, 768), (768, 1), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((3072, 768), (768, 1), torch.float32, 'cuda'), ((3072,), (1,), torch.float32, 'cuda'), ((768, 3072), (3072, 1), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768, 768), (768, 1), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768, 768), (768, 1), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768, 768), (768, 1), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768, 768), (768, 1), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((3072, 768), (768, 1), torch.float32, 'cuda'), ((3072,), (1,), torch.float32, 'cuda'), ((768, 3072), (3072, 1), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768, 768), (768, 1), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768, 768), (768, 1), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768, 768), (768, 1), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768, 768), (768, 1), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((3072, 768), (768, 1), torch.float32, 'cuda'), ((3072,), (1,), torch.float32, 'cuda'), ((768, 3072), (3072, 1), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768, 768), (768, 1), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768, 768), (768, 1), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768, 768), (768, 1), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768, 768), (768, 1), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((3072, 768), (768, 1), torch.float32, 'cuda'), ((3072,), (1,), torch.float32, 'cuda'), ((768, 3072), (3072, 1), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768, 768), (768, 1), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768, 768), (768, 1), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768, 768), (768, 1), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768, 768), (768, 1), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((3072, 768), (768, 1), torch.float32, 'cuda'), ((3072,), (1,), torch.float32, 'cuda'), ((768, 3072), (3072, 1), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768, 768), (768, 1), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768, 768), (768, 1), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768, 768), (768, 1), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768, 768), (768, 1), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((3072, 768), (768, 1), torch.float32, 'cuda'), ((3072,), (1,), torch.float32, 'cuda'), ((768, 3072), (3072, 1), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768, 768), (768, 1), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768, 768), (768, 1), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768, 768), (768, 1), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768, 768), (768, 1), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((3072, 768), (768, 1), torch.float32, 'cuda'), ((3072,), (1,), torch.float32, 'cuda'), ((768, 3072), (3072, 1), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768, 768), (768, 1), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768, 768), (768, 1), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768, 768), (768, 1), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768, 768), (768, 1), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((3072, 768), (768, 1), torch.float32, 'cuda'), ((3072,), (1,), torch.float32, 'cuda'), ((768, 3072), (3072, 1), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768, 768), (768, 1), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768, 768), (768, 1), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768, 768), (768, 1), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768, 768), (768, 1), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((3072, 768), (768, 1), torch.float32, 'cuda'), ((3072,), (1,), torch.float32, 'cuda'), ((768, 3072), (3072, 1), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768, 768), (768, 1), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((30522,), (1,), torch.float32, 'cuda'), ((1, 512), (512, 1), torch.int64, 'cuda'), ((1, 512), (512, 1), torch.int64, 'cuda'), ((64, 128), (128, 1), torch.int64, 'cuda'), ((64, 128), (128, 1), torch.int64, 'cuda')]
args = [rand_strided(sh, st, dt, dev) for (sh, st, dt, dev) in args]
mod = make_fx(Repro().to(device="cuda"))(*args)

from torch._inductor.compile_fx import compile_fx_inner
from torch._dynamo.debug_utils import same_two_models

compiled = compile_fx_inner(mod, args)
compiled(args)
