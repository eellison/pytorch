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

    
    
    def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, embedding, gt, reciprocal, div_2, philox_seed_like, view_9, gt_3, add_6, reciprocal_1, mm_4, sub_3, mm_5, gt_4, view_12, gt_5, add_11, reciprocal_3, div_3, view_21, gt_7, add_14, reciprocal_4, mm_11, sub_5, mm_12, gt_8, view_24, gt_9, add_19, reciprocal_6, div_4, view_33, gt_11, add_22, reciprocal_7, mm_18, sub_7, mm_19, gt_12, view_36, gt_13, add_27, reciprocal_9, div_5, view_45, gt_15, add_30, reciprocal_10, mm_25, sub_9, mm_26, gt_16, view_48, gt_17, add_35, reciprocal_12, div_6, view_57, gt_19, add_38, reciprocal_13, mm_32, sub_11, mm_33, gt_20, view_60, gt_21, add_43, reciprocal_15, div_7, view_69, gt_23, add_46, reciprocal_16, mm_39, sub_13, mm_40, gt_24, view_72, gt_25, add_51, reciprocal_18, div_8, view_81, gt_27, add_54, reciprocal_19, mm_46, sub_15, mm_47, gt_28, view_84, gt_29, add_59, reciprocal_21, div_9, view_93, gt_31, add_62, reciprocal_22, mm_53, sub_17, mm_54, gt_32, view_96, gt_33, add_67, reciprocal_24, gt_34, embedding_2, gt_35, reciprocal_25, div_12, view_106, gt_37, add_74, reciprocal_26, div_13, view_115, gt_39, add_78, reciprocal_27, mm_64, sub_23, mm_65, gt_40, view_118, gt_41, add_83, reciprocal_29, div_14, view_127, gt_43, add_86, reciprocal_30, div_15, view_136, gt_45, add_89, reciprocal_31, mm_75, sub_26, mm_76, gt_46, view_139, gt_47, add_94, reciprocal_33, div_16, view_148, gt_49, add_97, reciprocal_34, div_17, view_157, gt_51, add_100, reciprocal_35, mm_86, sub_29, mm_87, gt_52, view_160, gt_53, add_105, reciprocal_37, div_18, view_169, gt_55, add_108, reciprocal_38, div_19, view_178, gt_57, add_111, reciprocal_39, mm_97, sub_32, mm_98, gt_58, view_181, gt_59, add_116, reciprocal_41, div_20, view_190, gt_61, add_119, reciprocal_42, div_21, view_199, gt_63, add_122, reciprocal_43, mm_108, sub_35, mm_109, gt_64, view_202, gt_65, add_127, reciprocal_45, div_22, view_211, gt_67, add_130, reciprocal_46, div_23, view_220, gt_69, add_133, reciprocal_47, mm_119, sub_38, mm_120, gt_70, view_223, gt_71, add_138, reciprocal_49, div_24, view_232, gt_73, add_141, reciprocal_50, div_25, view_241, gt_75, add_144, reciprocal_51, mm_130, sub_41, mm_131, gt_76, view_244, gt_77, add_149, reciprocal_53, div_26, view_253, gt_79, add_152, reciprocal_54, div_27, view_262, gt_81, add_155, reciprocal_55, mm_141, sub_44, mm_142, gt_82, view_265, gt_83, add_160, reciprocal_57, gt_84, view_266, sub_46, unsqueeze_17, permute_269, permute_273, permute_277, permute_281, permute_285, permute_288, permute_289, permute_290, permute_291, permute_296, permute_301, permute_306, permute_310, permute_313, permute_314, permute_315, permute_316, permute_321, permute_326, permute_331, permute_335, permute_339, permute_343, permute_347, permute_350, permute_351, permute_352, permute_353, permute_358, permute_363, permute_368, permute_372, permute_375, permute_376, permute_377, permute_378, permute_383, permute_388, permute_393, permute_397, permute_401, permute_405, permute_409, permute_412, permute_413, permute_414, permute_415, permute_420, permute_425, permute_430, permute_434, permute_437, permute_438, permute_439, permute_440, permute_445, permute_450, permute_455, permute_459, permute_463, permute_467, permute_471, permute_474, permute_475, permute_476, permute_477, permute_482, permute_487, permute_492, permute_496, permute_499, permute_500, permute_501, permute_502, permute_507, permute_512, permute_517, permute_521, permute_525, permute_529, permute_533, permute_536, permute_537, permute_538, permute_539, permute_544, permute_549, permute_554, permute_558, permute_561, permute_562, permute_563, permute_564, permute_569, permute_574, permute_579, permute_583, permute_587, permute_591, permute_595, permute_598, permute_599, permute_600, permute_601, permute_606, permute_611, permute_616, permute_620, permute_623, permute_624, permute_625, permute_626, permute_631, permute_636, permute_641, permute_645, permute_649, permute_653, permute_657, permute_660, permute_661, permute_662, permute_663, permute_668, permute_673, permute_678, permute_682, permute_685, permute_686, permute_687, permute_688, permute_693, permute_698, permute_703, permute_707, permute_711, permute_715, permute_719, permute_722, permute_723, permute_724, permute_725, permute_730, permute_735, permute_740, permute_744, permute_747, permute_748, view_560, permute_750, permute_751, permute_756, permute_761, permute_766, view_572, permute_770, permute_774, permute_778, permute_782, permute_785, permute_786, permute_787, permute_788, permute_793, permute_798, permute_803, permute_807, permute_811, permute_815, permute_819, permute_822, permute_823, permute_824, permute_825, permute_830, permute_835, permute_840, permute_844, permute_848, permute_852, permute_856, permute_859, permute_860, permute_861, permute_862, permute_867, permute_872, permute_877, permute_881, permute_885, permute_889, permute_893, permute_896, permute_897, permute_898, permute_899, permute_904, permute_909, permute_914, permute_918, permute_922, permute_926, permute_930, permute_933, permute_934, permute_935, permute_936, permute_941, permute_946, permute_951, permute_955, permute_959, permute_963, permute_967, permute_970, permute_971, permute_972, permute_973, permute_978, permute_983, permute_988, permute_992, permute_996, permute_1000, permute_1004, permute_1007, permute_1008, permute_1009, permute_1010, permute_1015, permute_1020, permute_1025, permute_1029, permute_1033, permute_1037, permute_1041, permute_1044, permute_1045, view_741, permute_1047, permute_1048, permute_1053, permute_1058, permute_1063, view_753, tangents_1, tangents_2, tangents_3, tangents_4, tangents_5, tangents_6, tangents_7, tangents_8, tangents_9, tangents_10, tangents_11, tangents_12, tangents_13, tangents_14, tangents_15, tangents_16, tangents_17, tangents_18, tangents_19, tangents_20, tangents_21, tangents_22, tangents_23, tangents_24, tangents_25, tangents_26, tangents_27, tangents_28, tangents_29, tangents_30, tangents_31, tangents_32, tangents_33, tangents_34, tangents_35):
        mul_1 = torch.ops.aten.mul.Tensor(gt, embedding);  embedding = None
        mul_2 = torch.ops.aten.mul.Tensor(mul_1, 1.1111111111111112);  mul_1 = None
        alias_4 = torch.ops.aten.alias.default(reciprocal)
        alias_5 = torch.ops.aten.alias.default(alias_4);  alias_4 = None
        mul_3 = torch.ops.aten.mul.Tensor(mul_2, reciprocal)
        mul_4 = torch.ops.aten.mul.Tensor(primals_1, mul_3)
        view_1 = torch.ops.aten.view.default(mul_4, [256, 512]);  mul_4 = None
        alias_13 = torch.ops.aten.alias.default(div_2);  div_2 = None
        alias_14 = torch.ops.aten.alias.default(alias_13);  alias_13 = None
        alias_17 = torch.ops.aten.alias.default(reciprocal_1)
        alias_18 = torch.ops.aten.alias.default(alias_17);  alias_17 = None
        mul_11 = torch.ops.aten.mul.Tensor(add_6, reciprocal_1)
        mul_12 = torch.ops.aten.mul.Tensor(primals_2, mul_11)
        view_10 = torch.ops.aten.view.default(mul_12, [256, 512]);  mul_12 = None
        _unsafe_view_9 = torch.ops.aten._unsafe_view.default(mm_4, [2, 128, 1024]);  mm_4 = None
        mul_13 = torch.ops.aten.mul.Tensor(_unsafe_view_9, 0.5)
        alias_20 = torch.ops.aten.alias.default(sub_3)
        alias_21 = torch.ops.aten.alias.default(alias_20);  alias_20 = None
        add_10 = torch.ops.aten.add.Tensor(sub_3, 1.0);  sub_3 = None
        mul_18 = torch.ops.aten.mul.Tensor(mul_13, add_10)
        _unsafe_view_10 = torch.ops.aten._unsafe_view.default(mm_5, [2, 128, 1024]);  mm_5 = None
        alias_25 = torch.ops.aten.alias.default(reciprocal_3)
        alias_26 = torch.ops.aten.alias.default(alias_25);  alias_25 = None
        mul_24 = torch.ops.aten.mul.Tensor(add_11, reciprocal_3)
        mul_25 = torch.ops.aten.mul.Tensor(primals_3, mul_24)
        view_13 = torch.ops.aten.view.default(mul_25, [256, 512]);  mul_25 = None
        alias_28 = torch.ops.aten.alias.default(div_3);  div_3 = None
        alias_29 = torch.ops.aten.alias.default(alias_28);  alias_28 = None
        alias_32 = torch.ops.aten.alias.default(reciprocal_4)
        alias_33 = torch.ops.aten.alias.default(alias_32);  alias_32 = None
        mul_30 = torch.ops.aten.mul.Tensor(add_14, reciprocal_4)
        mul_31 = torch.ops.aten.mul.Tensor(primals_4, mul_30)
        view_22 = torch.ops.aten.view.default(mul_31, [256, 512]);  mul_31 = None
        _unsafe_view_21 = torch.ops.aten._unsafe_view.default(mm_11, [2, 128, 1024]);  mm_11 = None
        mul_32 = torch.ops.aten.mul.Tensor(_unsafe_view_21, 0.5)
        alias_35 = torch.ops.aten.alias.default(sub_5)
        alias_36 = torch.ops.aten.alias.default(alias_35);  alias_35 = None
        add_18 = torch.ops.aten.add.Tensor(sub_5, 1.0);  sub_5 = None
        mul_37 = torch.ops.aten.mul.Tensor(mul_32, add_18)
        _unsafe_view_22 = torch.ops.aten._unsafe_view.default(mm_12, [2, 128, 1024]);  mm_12 = None
        alias_40 = torch.ops.aten.alias.default(reciprocal_6)
        alias_41 = torch.ops.aten.alias.default(alias_40);  alias_40 = None
        mul_43 = torch.ops.aten.mul.Tensor(add_19, reciprocal_6)
        mul_44 = torch.ops.aten.mul.Tensor(primals_5, mul_43)
        view_25 = torch.ops.aten.view.default(mul_44, [256, 512]);  mul_44 = None
        alias_43 = torch.ops.aten.alias.default(div_4);  div_4 = None
        alias_44 = torch.ops.aten.alias.default(alias_43);  alias_43 = None
        alias_47 = torch.ops.aten.alias.default(reciprocal_7)
        alias_48 = torch.ops.aten.alias.default(alias_47);  alias_47 = None
        mul_49 = torch.ops.aten.mul.Tensor(add_22, reciprocal_7)
        mul_50 = torch.ops.aten.mul.Tensor(primals_6, mul_49)
        view_34 = torch.ops.aten.view.default(mul_50, [256, 512]);  mul_50 = None
        _unsafe_view_33 = torch.ops.aten._unsafe_view.default(mm_18, [2, 128, 1024]);  mm_18 = None
        mul_51 = torch.ops.aten.mul.Tensor(_unsafe_view_33, 0.5)
        alias_50 = torch.ops.aten.alias.default(sub_7)
        alias_51 = torch.ops.aten.alias.default(alias_50);  alias_50 = None
        add_26 = torch.ops.aten.add.Tensor(sub_7, 1.0);  sub_7 = None
        mul_56 = torch.ops.aten.mul.Tensor(mul_51, add_26)
        _unsafe_view_34 = torch.ops.aten._unsafe_view.default(mm_19, [2, 128, 1024]);  mm_19 = None
        alias_55 = torch.ops.aten.alias.default(reciprocal_9)
        alias_56 = torch.ops.aten.alias.default(alias_55);  alias_55 = None
        mul_62 = torch.ops.aten.mul.Tensor(add_27, reciprocal_9)
        mul_63 = torch.ops.aten.mul.Tensor(primals_7, mul_62)
        view_37 = torch.ops.aten.view.default(mul_63, [256, 512]);  mul_63 = None
        alias_58 = torch.ops.aten.alias.default(div_5);  div_5 = None
        alias_59 = torch.ops.aten.alias.default(alias_58);  alias_58 = None
        alias_62 = torch.ops.aten.alias.default(reciprocal_10)
        alias_63 = torch.ops.aten.alias.default(alias_62);  alias_62 = None
        mul_68 = torch.ops.aten.mul.Tensor(add_30, reciprocal_10)
        mul_69 = torch.ops.aten.mul.Tensor(primals_8, mul_68)
        view_46 = torch.ops.aten.view.default(mul_69, [256, 512]);  mul_69 = None
        _unsafe_view_45 = torch.ops.aten._unsafe_view.default(mm_25, [2, 128, 1024]);  mm_25 = None
        mul_70 = torch.ops.aten.mul.Tensor(_unsafe_view_45, 0.5)
        alias_65 = torch.ops.aten.alias.default(sub_9)
        alias_66 = torch.ops.aten.alias.default(alias_65);  alias_65 = None
        add_34 = torch.ops.aten.add.Tensor(sub_9, 1.0);  sub_9 = None
        mul_75 = torch.ops.aten.mul.Tensor(mul_70, add_34)
        _unsafe_view_46 = torch.ops.aten._unsafe_view.default(mm_26, [2, 128, 1024]);  mm_26 = None
        alias_70 = torch.ops.aten.alias.default(reciprocal_12)
        alias_71 = torch.ops.aten.alias.default(alias_70);  alias_70 = None
        mul_81 = torch.ops.aten.mul.Tensor(add_35, reciprocal_12)
        mul_82 = torch.ops.aten.mul.Tensor(primals_9, mul_81)
        view_49 = torch.ops.aten.view.default(mul_82, [256, 512]);  mul_82 = None
        alias_73 = torch.ops.aten.alias.default(div_6);  div_6 = None
        alias_74 = torch.ops.aten.alias.default(alias_73);  alias_73 = None
        alias_77 = torch.ops.aten.alias.default(reciprocal_13)
        alias_78 = torch.ops.aten.alias.default(alias_77);  alias_77 = None
        mul_87 = torch.ops.aten.mul.Tensor(add_38, reciprocal_13)
        mul_88 = torch.ops.aten.mul.Tensor(primals_10, mul_87)
        view_58 = torch.ops.aten.view.default(mul_88, [256, 512]);  mul_88 = None
        _unsafe_view_57 = torch.ops.aten._unsafe_view.default(mm_32, [2, 128, 1024]);  mm_32 = None
        mul_89 = torch.ops.aten.mul.Tensor(_unsafe_view_57, 0.5)
        alias_80 = torch.ops.aten.alias.default(sub_11)
        alias_81 = torch.ops.aten.alias.default(alias_80);  alias_80 = None
        add_42 = torch.ops.aten.add.Tensor(sub_11, 1.0);  sub_11 = None
        mul_94 = torch.ops.aten.mul.Tensor(mul_89, add_42)
        _unsafe_view_58 = torch.ops.aten._unsafe_view.default(mm_33, [2, 128, 1024]);  mm_33 = None
        alias_85 = torch.ops.aten.alias.default(reciprocal_15)
        alias_86 = torch.ops.aten.alias.default(alias_85);  alias_85 = None
        mul_100 = torch.ops.aten.mul.Tensor(add_43, reciprocal_15)
        mul_101 = torch.ops.aten.mul.Tensor(primals_11, mul_100)
        view_61 = torch.ops.aten.view.default(mul_101, [256, 512]);  mul_101 = None
        alias_88 = torch.ops.aten.alias.default(div_7);  div_7 = None
        alias_89 = torch.ops.aten.alias.default(alias_88);  alias_88 = None
        alias_92 = torch.ops.aten.alias.default(reciprocal_16)
        alias_93 = torch.ops.aten.alias.default(alias_92);  alias_92 = None
        mul_106 = torch.ops.aten.mul.Tensor(add_46, reciprocal_16)
        mul_107 = torch.ops.aten.mul.Tensor(primals_12, mul_106)
        view_70 = torch.ops.aten.view.default(mul_107, [256, 512]);  mul_107 = None
        _unsafe_view_69 = torch.ops.aten._unsafe_view.default(mm_39, [2, 128, 1024]);  mm_39 = None
        mul_108 = torch.ops.aten.mul.Tensor(_unsafe_view_69, 0.5)
        alias_95 = torch.ops.aten.alias.default(sub_13)
        alias_96 = torch.ops.aten.alias.default(alias_95);  alias_95 = None
        add_50 = torch.ops.aten.add.Tensor(sub_13, 1.0);  sub_13 = None
        mul_113 = torch.ops.aten.mul.Tensor(mul_108, add_50)
        _unsafe_view_70 = torch.ops.aten._unsafe_view.default(mm_40, [2, 128, 1024]);  mm_40 = None
        alias_100 = torch.ops.aten.alias.default(reciprocal_18)
        alias_101 = torch.ops.aten.alias.default(alias_100);  alias_100 = None
        mul_119 = torch.ops.aten.mul.Tensor(add_51, reciprocal_18)
        mul_120 = torch.ops.aten.mul.Tensor(primals_13, mul_119)
        view_73 = torch.ops.aten.view.default(mul_120, [256, 512]);  mul_120 = None
        alias_103 = torch.ops.aten.alias.default(div_8);  div_8 = None
        alias_104 = torch.ops.aten.alias.default(alias_103);  alias_103 = None
        alias_107 = torch.ops.aten.alias.default(reciprocal_19)
        alias_108 = torch.ops.aten.alias.default(alias_107);  alias_107 = None
        mul_125 = torch.ops.aten.mul.Tensor(add_54, reciprocal_19)
        mul_126 = torch.ops.aten.mul.Tensor(primals_14, mul_125)
        view_82 = torch.ops.aten.view.default(mul_126, [256, 512]);  mul_126 = None
        _unsafe_view_81 = torch.ops.aten._unsafe_view.default(mm_46, [2, 128, 1024]);  mm_46 = None
        mul_127 = torch.ops.aten.mul.Tensor(_unsafe_view_81, 0.5)
        alias_110 = torch.ops.aten.alias.default(sub_15)
        alias_111 = torch.ops.aten.alias.default(alias_110);  alias_110 = None
        add_58 = torch.ops.aten.add.Tensor(sub_15, 1.0);  sub_15 = None
        mul_132 = torch.ops.aten.mul.Tensor(mul_127, add_58)
        _unsafe_view_82 = torch.ops.aten._unsafe_view.default(mm_47, [2, 128, 1024]);  mm_47 = None
        alias_115 = torch.ops.aten.alias.default(reciprocal_21)
        alias_116 = torch.ops.aten.alias.default(alias_115);  alias_115 = None
        mul_138 = torch.ops.aten.mul.Tensor(add_59, reciprocal_21)
        mul_139 = torch.ops.aten.mul.Tensor(primals_15, mul_138)
        view_85 = torch.ops.aten.view.default(mul_139, [256, 512]);  mul_139 = None
        alias_118 = torch.ops.aten.alias.default(div_9);  div_9 = None
        alias_119 = torch.ops.aten.alias.default(alias_118);  alias_118 = None
        alias_122 = torch.ops.aten.alias.default(reciprocal_22)
        alias_123 = torch.ops.aten.alias.default(alias_122);  alias_122 = None
        mul_144 = torch.ops.aten.mul.Tensor(add_62, reciprocal_22)
        mul_145 = torch.ops.aten.mul.Tensor(primals_16, mul_144)
        view_94 = torch.ops.aten.view.default(mul_145, [256, 512]);  mul_145 = None
        _unsafe_view_93 = torch.ops.aten._unsafe_view.default(mm_53, [2, 128, 1024]);  mm_53 = None
        mul_146 = torch.ops.aten.mul.Tensor(_unsafe_view_93, 0.5)
        alias_125 = torch.ops.aten.alias.default(sub_17)
        alias_126 = torch.ops.aten.alias.default(alias_125);  alias_125 = None
        add_66 = torch.ops.aten.add.Tensor(sub_17, 1.0);  sub_17 = None
        mul_151 = torch.ops.aten.mul.Tensor(mul_146, add_66)
        _unsafe_view_94 = torch.ops.aten._unsafe_view.default(mm_54, [2, 128, 1024]);  mm_54 = None
        alias_130 = torch.ops.aten.alias.default(reciprocal_24)
        alias_131 = torch.ops.aten.alias.default(alias_130);  alias_130 = None
        mul_157 = torch.ops.aten.mul.Tensor(add_67, reciprocal_24)
        mul_158 = torch.ops.aten.mul.Tensor(primals_17, mul_157)
        mul_159 = torch.ops.aten.mul.Tensor(gt_34, mul_158);  mul_158 = None
        mul_160 = torch.ops.aten.mul.Tensor(mul_159, 1.1111111111111112);  mul_159 = None
        mul_164 = torch.ops.aten.mul.Tensor(gt_35, embedding_2);  embedding_2 = None
        mul_165 = torch.ops.aten.mul.Tensor(mul_164, 1.1111111111111112);  mul_164 = None
        alias_141 = torch.ops.aten.alias.default(reciprocal_25)
        alias_142 = torch.ops.aten.alias.default(alias_141);  alias_141 = None
        mul_166 = torch.ops.aten.mul.Tensor(mul_165, reciprocal_25)
        mul_167 = torch.ops.aten.mul.Tensor(primals_18, mul_166)
        view_98 = torch.ops.aten.view.default(mul_167, [256, 512]);  mul_167 = None
        alias_152 = torch.ops.aten.alias.default(div_12);  div_12 = None
        alias_153 = torch.ops.aten.alias.default(alias_152);  alias_152 = None
        alias_156 = torch.ops.aten.alias.default(reciprocal_26)
        alias_157 = torch.ops.aten.alias.default(alias_156);  alias_156 = None
        mul_173 = torch.ops.aten.mul.Tensor(add_74, reciprocal_26)
        mul_174 = torch.ops.aten.mul.Tensor(primals_19, mul_173)
        view_107 = torch.ops.aten.view.default(mul_174, [256, 512]);  mul_174 = None
        view_109 = torch.ops.aten.view.default(mul_160, [256, 512]);  mul_160 = None
        alias_161 = torch.ops.aten.alias.default(div_13);  div_13 = None
        alias_162 = torch.ops.aten.alias.default(alias_161);  alias_161 = None
        alias_165 = torch.ops.aten.alias.default(reciprocal_27)
        alias_166 = torch.ops.aten.alias.default(alias_165);  alias_165 = None
        mul_179 = torch.ops.aten.mul.Tensor(add_78, reciprocal_27)
        mul_180 = torch.ops.aten.mul.Tensor(primals_20, mul_179)
        view_116 = torch.ops.aten.view.default(mul_180, [256, 512]);  mul_180 = None
        _unsafe_view_114 = torch.ops.aten._unsafe_view.default(mm_64, [2, 128, 1024]);  mm_64 = None
        mul_181 = torch.ops.aten.mul.Tensor(_unsafe_view_114, 0.5)
        alias_168 = torch.ops.aten.alias.default(sub_23)
        alias_169 = torch.ops.aten.alias.default(alias_168);  alias_168 = None
        add_82 = torch.ops.aten.add.Tensor(sub_23, 1.0);  sub_23 = None
        mul_186 = torch.ops.aten.mul.Tensor(mul_181, add_82)
        _unsafe_view_115 = torch.ops.aten._unsafe_view.default(mm_65, [2, 128, 1024]);  mm_65 = None
        alias_173 = torch.ops.aten.alias.default(reciprocal_29)
        alias_174 = torch.ops.aten.alias.default(alias_173);  alias_173 = None
        mul_192 = torch.ops.aten.mul.Tensor(add_83, reciprocal_29)
        mul_193 = torch.ops.aten.mul.Tensor(primals_21, mul_192)
        view_119 = torch.ops.aten.view.default(mul_193, [256, 512]);  mul_193 = None
        alias_176 = torch.ops.aten.alias.default(div_14);  div_14 = None
        alias_177 = torch.ops.aten.alias.default(alias_176);  alias_176 = None
        alias_180 = torch.ops.aten.alias.default(reciprocal_30)
        alias_181 = torch.ops.aten.alias.default(alias_180);  alias_180 = None
        mul_198 = torch.ops.aten.mul.Tensor(add_86, reciprocal_30)
        mul_199 = torch.ops.aten.mul.Tensor(primals_22, mul_198)
        view_128 = torch.ops.aten.view.default(mul_199, [256, 512]);  mul_199 = None
        alias_183 = torch.ops.aten.alias.default(div_15);  div_15 = None
        alias_184 = torch.ops.aten.alias.default(alias_183);  alias_183 = None
        alias_187 = torch.ops.aten.alias.default(reciprocal_31)
        alias_188 = torch.ops.aten.alias.default(alias_187);  alias_187 = None
        mul_204 = torch.ops.aten.mul.Tensor(add_89, reciprocal_31)
        mul_205 = torch.ops.aten.mul.Tensor(primals_23, mul_204)
        view_137 = torch.ops.aten.view.default(mul_205, [256, 512]);  mul_205 = None
        _unsafe_view_135 = torch.ops.aten._unsafe_view.default(mm_75, [2, 128, 1024]);  mm_75 = None
        mul_206 = torch.ops.aten.mul.Tensor(_unsafe_view_135, 0.5)
        alias_190 = torch.ops.aten.alias.default(sub_26)
        alias_191 = torch.ops.aten.alias.default(alias_190);  alias_190 = None
        add_93 = torch.ops.aten.add.Tensor(sub_26, 1.0);  sub_26 = None
        mul_211 = torch.ops.aten.mul.Tensor(mul_206, add_93)
        _unsafe_view_136 = torch.ops.aten._unsafe_view.default(mm_76, [2, 128, 1024]);  mm_76 = None
        alias_195 = torch.ops.aten.alias.default(reciprocal_33)
        alias_196 = torch.ops.aten.alias.default(alias_195);  alias_195 = None
        mul_217 = torch.ops.aten.mul.Tensor(add_94, reciprocal_33)
        mul_218 = torch.ops.aten.mul.Tensor(primals_24, mul_217)
        view_140 = torch.ops.aten.view.default(mul_218, [256, 512]);  mul_218 = None
        alias_198 = torch.ops.aten.alias.default(div_16);  div_16 = None
        alias_199 = torch.ops.aten.alias.default(alias_198);  alias_198 = None
        alias_202 = torch.ops.aten.alias.default(reciprocal_34)
        alias_203 = torch.ops.aten.alias.default(alias_202);  alias_202 = None
        mul_223 = torch.ops.aten.mul.Tensor(add_97, reciprocal_34)
        mul_224 = torch.ops.aten.mul.Tensor(primals_25, mul_223)
        view_149 = torch.ops.aten.view.default(mul_224, [256, 512]);  mul_224 = None
        alias_205 = torch.ops.aten.alias.default(div_17);  div_17 = None
        alias_206 = torch.ops.aten.alias.default(alias_205);  alias_205 = None
        alias_209 = torch.ops.aten.alias.default(reciprocal_35)
        alias_210 = torch.ops.aten.alias.default(alias_209);  alias_209 = None
        mul_229 = torch.ops.aten.mul.Tensor(add_100, reciprocal_35)
        mul_230 = torch.ops.aten.mul.Tensor(primals_26, mul_229)
        view_158 = torch.ops.aten.view.default(mul_230, [256, 512]);  mul_230 = None
        _unsafe_view_156 = torch.ops.aten._unsafe_view.default(mm_86, [2, 128, 1024]);  mm_86 = None
        mul_231 = torch.ops.aten.mul.Tensor(_unsafe_view_156, 0.5)
        alias_212 = torch.ops.aten.alias.default(sub_29)
        alias_213 = torch.ops.aten.alias.default(alias_212);  alias_212 = None
        add_104 = torch.ops.aten.add.Tensor(sub_29, 1.0);  sub_29 = None
        mul_236 = torch.ops.aten.mul.Tensor(mul_231, add_104)
        _unsafe_view_157 = torch.ops.aten._unsafe_view.default(mm_87, [2, 128, 1024]);  mm_87 = None
        alias_217 = torch.ops.aten.alias.default(reciprocal_37)
        alias_218 = torch.ops.aten.alias.default(alias_217);  alias_217 = None
        mul_242 = torch.ops.aten.mul.Tensor(add_105, reciprocal_37)
        mul_243 = torch.ops.aten.mul.Tensor(primals_27, mul_242)
        view_161 = torch.ops.aten.view.default(mul_243, [256, 512]);  mul_243 = None
        alias_220 = torch.ops.aten.alias.default(div_18);  div_18 = None
        alias_221 = torch.ops.aten.alias.default(alias_220);  alias_220 = None
        alias_224 = torch.ops.aten.alias.default(reciprocal_38)
        alias_225 = torch.ops.aten.alias.default(alias_224);  alias_224 = None
        mul_248 = torch.ops.aten.mul.Tensor(add_108, reciprocal_38)
        mul_249 = torch.ops.aten.mul.Tensor(primals_28, mul_248)
        view_170 = torch.ops.aten.view.default(mul_249, [256, 512]);  mul_249 = None
        alias_227 = torch.ops.aten.alias.default(div_19);  div_19 = None
        alias_228 = torch.ops.aten.alias.default(alias_227);  alias_227 = None
        alias_231 = torch.ops.aten.alias.default(reciprocal_39)
        alias_232 = torch.ops.aten.alias.default(alias_231);  alias_231 = None
        mul_254 = torch.ops.aten.mul.Tensor(add_111, reciprocal_39)
        mul_255 = torch.ops.aten.mul.Tensor(primals_29, mul_254)
        view_179 = torch.ops.aten.view.default(mul_255, [256, 512]);  mul_255 = None
        _unsafe_view_177 = torch.ops.aten._unsafe_view.default(mm_97, [2, 128, 1024]);  mm_97 = None
        mul_256 = torch.ops.aten.mul.Tensor(_unsafe_view_177, 0.5)
        alias_234 = torch.ops.aten.alias.default(sub_32)
        alias_235 = torch.ops.aten.alias.default(alias_234);  alias_234 = None
        add_115 = torch.ops.aten.add.Tensor(sub_32, 1.0);  sub_32 = None
        mul_261 = torch.ops.aten.mul.Tensor(mul_256, add_115)
        _unsafe_view_178 = torch.ops.aten._unsafe_view.default(mm_98, [2, 128, 1024]);  mm_98 = None
        alias_239 = torch.ops.aten.alias.default(reciprocal_41)
        alias_240 = torch.ops.aten.alias.default(alias_239);  alias_239 = None
        mul_267 = torch.ops.aten.mul.Tensor(add_116, reciprocal_41)
        mul_268 = torch.ops.aten.mul.Tensor(primals_30, mul_267)
        view_182 = torch.ops.aten.view.default(mul_268, [256, 512]);  mul_268 = None
        alias_242 = torch.ops.aten.alias.default(div_20);  div_20 = None
        alias_243 = torch.ops.aten.alias.default(alias_242);  alias_242 = None
        alias_246 = torch.ops.aten.alias.default(reciprocal_42)
        alias_247 = torch.ops.aten.alias.default(alias_246);  alias_246 = None
        mul_273 = torch.ops.aten.mul.Tensor(add_119, reciprocal_42)
        mul_274 = torch.ops.aten.mul.Tensor(primals_31, mul_273)
        view_191 = torch.ops.aten.view.default(mul_274, [256, 512]);  mul_274 = None
        alias_249 = torch.ops.aten.alias.default(div_21);  div_21 = None
        alias_250 = torch.ops.aten.alias.default(alias_249);  alias_249 = None
        alias_253 = torch.ops.aten.alias.default(reciprocal_43)
        alias_254 = torch.ops.aten.alias.default(alias_253);  alias_253 = None
        mul_279 = torch.ops.aten.mul.Tensor(add_122, reciprocal_43)
        mul_280 = torch.ops.aten.mul.Tensor(primals_32, mul_279)
        view_200 = torch.ops.aten.view.default(mul_280, [256, 512]);  mul_280 = None
        _unsafe_view_198 = torch.ops.aten._unsafe_view.default(mm_108, [2, 128, 1024]);  mm_108 = None
        mul_281 = torch.ops.aten.mul.Tensor(_unsafe_view_198, 0.5)
        alias_256 = torch.ops.aten.alias.default(sub_35)
        alias_257 = torch.ops.aten.alias.default(alias_256);  alias_256 = None
        add_126 = torch.ops.aten.add.Tensor(sub_35, 1.0);  sub_35 = None
        mul_286 = torch.ops.aten.mul.Tensor(mul_281, add_126)
        _unsafe_view_199 = torch.ops.aten._unsafe_view.default(mm_109, [2, 128, 1024]);  mm_109 = None
        alias_261 = torch.ops.aten.alias.default(reciprocal_45)
        alias_262 = torch.ops.aten.alias.default(alias_261);  alias_261 = None
        mul_292 = torch.ops.aten.mul.Tensor(add_127, reciprocal_45)
        mul_293 = torch.ops.aten.mul.Tensor(primals_33, mul_292)
        view_203 = torch.ops.aten.view.default(mul_293, [256, 512]);  mul_293 = None
        alias_264 = torch.ops.aten.alias.default(div_22);  div_22 = None
        alias_265 = torch.ops.aten.alias.default(alias_264);  alias_264 = None
        alias_268 = torch.ops.aten.alias.default(reciprocal_46)
        alias_269 = torch.ops.aten.alias.default(alias_268);  alias_268 = None
        mul_298 = torch.ops.aten.mul.Tensor(add_130, reciprocal_46)
        mul_299 = torch.ops.aten.mul.Tensor(primals_34, mul_298)
        view_212 = torch.ops.aten.view.default(mul_299, [256, 512]);  mul_299 = None
        alias_271 = torch.ops.aten.alias.default(div_23);  div_23 = None
        alias_272 = torch.ops.aten.alias.default(alias_271);  alias_271 = None
        alias_275 = torch.ops.aten.alias.default(reciprocal_47)
        alias_276 = torch.ops.aten.alias.default(alias_275);  alias_275 = None
        mul_304 = torch.ops.aten.mul.Tensor(add_133, reciprocal_47)
        mul_305 = torch.ops.aten.mul.Tensor(primals_35, mul_304)
        view_221 = torch.ops.aten.view.default(mul_305, [256, 512]);  mul_305 = None
        _unsafe_view_219 = torch.ops.aten._unsafe_view.default(mm_119, [2, 128, 1024]);  mm_119 = None
        mul_306 = torch.ops.aten.mul.Tensor(_unsafe_view_219, 0.5)
        alias_278 = torch.ops.aten.alias.default(sub_38)
        alias_279 = torch.ops.aten.alias.default(alias_278);  alias_278 = None
        add_137 = torch.ops.aten.add.Tensor(sub_38, 1.0);  sub_38 = None
        mul_311 = torch.ops.aten.mul.Tensor(mul_306, add_137)
        _unsafe_view_220 = torch.ops.aten._unsafe_view.default(mm_120, [2, 128, 1024]);  mm_120 = None
        alias_283 = torch.ops.aten.alias.default(reciprocal_49)
        alias_284 = torch.ops.aten.alias.default(alias_283);  alias_283 = None
        mul_317 = torch.ops.aten.mul.Tensor(add_138, reciprocal_49)
        mul_318 = torch.ops.aten.mul.Tensor(primals_36, mul_317)
        view_224 = torch.ops.aten.view.default(mul_318, [256, 512]);  mul_318 = None
        alias_286 = torch.ops.aten.alias.default(div_24);  div_24 = None
        alias_287 = torch.ops.aten.alias.default(alias_286);  alias_286 = None
        alias_290 = torch.ops.aten.alias.default(reciprocal_50)
        alias_291 = torch.ops.aten.alias.default(alias_290);  alias_290 = None
        mul_323 = torch.ops.aten.mul.Tensor(add_141, reciprocal_50)
        mul_324 = torch.ops.aten.mul.Tensor(primals_37, mul_323)
        view_233 = torch.ops.aten.view.default(mul_324, [256, 512]);  mul_324 = None
        alias_293 = torch.ops.aten.alias.default(div_25);  div_25 = None
        alias_294 = torch.ops.aten.alias.default(alias_293);  alias_293 = None
        alias_297 = torch.ops.aten.alias.default(reciprocal_51)
        alias_298 = torch.ops.aten.alias.default(alias_297);  alias_297 = None
        mul_329 = torch.ops.aten.mul.Tensor(add_144, reciprocal_51)
        mul_330 = torch.ops.aten.mul.Tensor(primals_38, mul_329)
        view_242 = torch.ops.aten.view.default(mul_330, [256, 512]);  mul_330 = None
        _unsafe_view_240 = torch.ops.aten._unsafe_view.default(mm_130, [2, 128, 1024]);  mm_130 = None
        mul_331 = torch.ops.aten.mul.Tensor(_unsafe_view_240, 0.5)
        alias_300 = torch.ops.aten.alias.default(sub_41)
        alias_301 = torch.ops.aten.alias.default(alias_300);  alias_300 = None
        add_148 = torch.ops.aten.add.Tensor(sub_41, 1.0);  sub_41 = None
        mul_336 = torch.ops.aten.mul.Tensor(mul_331, add_148)
        _unsafe_view_241 = torch.ops.aten._unsafe_view.default(mm_131, [2, 128, 1024]);  mm_131 = None
        alias_305 = torch.ops.aten.alias.default(reciprocal_53)
        alias_306 = torch.ops.aten.alias.default(alias_305);  alias_305 = None
        mul_342 = torch.ops.aten.mul.Tensor(add_149, reciprocal_53)
        mul_343 = torch.ops.aten.mul.Tensor(primals_39, mul_342)
        view_245 = torch.ops.aten.view.default(mul_343, [256, 512]);  mul_343 = None
        alias_308 = torch.ops.aten.alias.default(div_26);  div_26 = None
        alias_309 = torch.ops.aten.alias.default(alias_308);  alias_308 = None
        alias_312 = torch.ops.aten.alias.default(reciprocal_54)
        alias_313 = torch.ops.aten.alias.default(alias_312);  alias_312 = None
        mul_348 = torch.ops.aten.mul.Tensor(add_152, reciprocal_54)
        mul_349 = torch.ops.aten.mul.Tensor(primals_40, mul_348)
        view_254 = torch.ops.aten.view.default(mul_349, [256, 512]);  mul_349 = None
        alias_315 = torch.ops.aten.alias.default(div_27);  div_27 = None
        alias_316 = torch.ops.aten.alias.default(alias_315);  alias_315 = None
        alias_319 = torch.ops.aten.alias.default(reciprocal_55)
        alias_320 = torch.ops.aten.alias.default(alias_319);  alias_319 = None
        mul_354 = torch.ops.aten.mul.Tensor(add_155, reciprocal_55)
        mul_355 = torch.ops.aten.mul.Tensor(primals_41, mul_354)
        view_263 = torch.ops.aten.view.default(mul_355, [256, 512]);  mul_355 = None
        _unsafe_view_261 = torch.ops.aten._unsafe_view.default(mm_141, [2, 128, 1024]);  mm_141 = None
        mul_356 = torch.ops.aten.mul.Tensor(_unsafe_view_261, 0.5)
        alias_322 = torch.ops.aten.alias.default(sub_44)
        alias_323 = torch.ops.aten.alias.default(alias_322);  alias_322 = None
        add_159 = torch.ops.aten.add.Tensor(sub_44, 1.0);  sub_44 = None
        mul_361 = torch.ops.aten.mul.Tensor(mul_356, add_159)
        _unsafe_view_262 = torch.ops.aten._unsafe_view.default(mm_142, [2, 128, 1024]);  mm_142 = None
        alias_327 = torch.ops.aten.alias.default(reciprocal_57)
        alias_328 = torch.ops.aten.alias.default(alias_327);  alias_327 = None
        mul_367 = torch.ops.aten.mul.Tensor(add_160, reciprocal_57)
        alias_331 = torch.ops.aten.alias.default(sub_46)
        alias_332 = torch.ops.aten.alias.default(alias_331);  alias_331 = None
        full = torch.ops.aten.full.default([], 256.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_333 = torch.ops.aten.alias.default(full);  full = None
        div_28 = torch.ops.aten.div.Tensor(tangents_1, alias_333);  tangents_1 = alias_333 = None
        zeros_like_1 = torch.ops.aten.zeros_like.default(sub_46, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  sub_46 = None
        alias_334 = torch.ops.aten.alias.default(zeros_like_1);  zeros_like_1 = None
        scatter = torch.ops.aten.scatter.value(alias_334, 1, unsqueeze_17, -1.0);  alias_334 = unsqueeze_17 = None
        mul_371 = torch.ops.aten.mul.Tensor(scatter, div_28);  scatter = div_28 = None
        alias_335 = torch.ops.aten.alias.default(alias_332);  alias_332 = None
        alias_336 = torch.ops.aten.alias.default(alias_335);  alias_335 = None
        exp_41 = torch.ops.aten.exp.default(alias_336);  alias_336 = None
        sum_26 = torch.ops.aten.sum.dim_IntList(mul_371, [1], True)
        mul_372 = torch.ops.aten.mul.Tensor(exp_41, sum_26);  exp_41 = sum_26 = None
        sub_47 = torch.ops.aten.sub.Tensor(mul_371, mul_372);  mul_371 = mul_372 = None
        view_269 = torch.ops.aten.view.default(sub_47, [2, 128, 250112]);  sub_47 = None
        add_162 = torch.ops.aten.add.Tensor(tangents_2, view_269);  tangents_2 = view_269 = None
        view_270 = torch.ops.aten.view.default(add_162, [256, 250112]);  add_162 = None
        permute_267 = torch.ops.aten.permute.default(view_270, [1, 0])
        mm_145 = torch.ops.aten.mm.default(permute_267, view_266);  permute_267 = view_266 = None
        permute_268 = torch.ops.aten.permute.default(mm_145, [1, 0]);  mm_145 = None
        mm_146 = torch.ops.aten.mm.default(view_270, permute_269);  view_270 = permute_269 = None
        view_271 = torch.ops.aten.view.default(mm_146, [2, 128, 512]);  mm_146 = None
        permute_270 = torch.ops.aten.permute.default(permute_268, [1, 0]);  permute_268 = None
        _to_copy_31 = torch.ops.aten._to_copy.default(gt_84, dtype = torch.float32);  gt_84 = None
        mul_373 = torch.ops.aten.mul.Tensor(_to_copy_31, 1.1111111111111112);  _to_copy_31 = None
        mul_374 = torch.ops.aten.mul.Tensor(view_271, mul_373);  view_271 = mul_373 = None
        mul_375 = torch.ops.aten.mul.Tensor(mul_374, primals_42);  primals_42 = None
        mul_376 = torch.ops.aten.mul.Tensor(mul_374, mul_367);  mul_374 = mul_367 = None
        sum_27 = torch.ops.aten.sum.dim_IntList(mul_376, [0, 1], True);  mul_376 = None
        view_272 = torch.ops.aten.view.default(sum_27, [512]);  sum_27 = None
        mul_377 = torch.ops.aten.mul.Tensor(mul_375, add_160)
        mul_378 = torch.ops.aten.mul.Tensor(mul_375, reciprocal_57);  mul_375 = reciprocal_57 = None
        sum_28 = torch.ops.aten.sum.dim_IntList(mul_377, [2], True);  mul_377 = None
        alias_337 = torch.ops.aten.alias.default(alias_328);  alias_328 = None
        alias_338 = torch.ops.aten.alias.default(alias_337);  alias_337 = None
        pow_59 = torch.ops.aten.pow.Tensor_Scalar(alias_338, 3);  alias_338 = None
        mul_379 = torch.ops.aten.mul.Scalar(sum_28, -0.5);  sum_28 = None
        mul_380 = torch.ops.aten.mul.Tensor(mul_379, pow_59);  mul_379 = pow_59 = None
        expand_96 = torch.ops.aten.expand.default(mul_380, [2, 128, 512]);  mul_380 = None
        div_29 = torch.ops.aten.div.Scalar(expand_96, 512);  expand_96 = None
        pow_60 = torch.ops.aten.pow.Tensor_Scalar(add_160, 1.0);  add_160 = None
        mul_381 = torch.ops.aten.mul.Scalar(pow_60, 2.0);  pow_60 = None
        mul_382 = torch.ops.aten.mul.Tensor(div_29, mul_381);  div_29 = mul_381 = None
        add_163 = torch.ops.aten.add.Tensor(mul_378, mul_382);  mul_378 = mul_382 = None
        _to_copy_32 = torch.ops.aten._to_copy.default(gt_83, dtype = torch.float32);  gt_83 = None
        mul_383 = torch.ops.aten.mul.Tensor(_to_copy_32, 1.1111111111111112);  _to_copy_32 = None
        mul_384 = torch.ops.aten.mul.Tensor(add_163, mul_383);  mul_383 = None
        view_273 = torch.ops.aten.view.default(mul_384, [256, 512]);  mul_384 = None
        permute_271 = torch.ops.aten.permute.default(view_273, [1, 0])
        mm_147 = torch.ops.aten.mm.default(permute_271, view_265);  permute_271 = view_265 = None
        permute_272 = torch.ops.aten.permute.default(mm_147, [1, 0]);  mm_147 = None
        mm_148 = torch.ops.aten.mm.default(view_273, permute_273);  view_273 = permute_273 = None
        view_274 = torch.ops.aten.view.default(mm_148, [2, 128, 1024]);  mm_148 = None
        permute_274 = torch.ops.aten.permute.default(permute_272, [1, 0]);  permute_272 = None
        _to_copy_33 = torch.ops.aten._to_copy.default(gt_82, dtype = torch.float32);  gt_82 = None
        mul_385 = torch.ops.aten.mul.Tensor(_to_copy_33, 1.1111111111111112);  _to_copy_33 = None
        mul_386 = torch.ops.aten.mul.Tensor(view_274, mul_385);  view_274 = mul_385 = None
        mul_387 = torch.ops.aten.mul.Tensor(mul_386, mul_361);  mul_361 = None
        mul_388 = torch.ops.aten.mul.Tensor(mul_386, _unsafe_view_262);  mul_386 = _unsafe_view_262 = None
        view_275 = torch.ops.aten.view.default(mul_387, [256, 1024]);  mul_387 = None
        permute_275 = torch.ops.aten.permute.default(view_275, [1, 0])
        mm_149 = torch.ops.aten.mm.default(permute_275, view_263);  permute_275 = None
        permute_276 = torch.ops.aten.permute.default(mm_149, [1, 0]);  mm_149 = None
        mm_150 = torch.ops.aten.mm.default(view_275, permute_277);  view_275 = permute_277 = None
        view_276 = torch.ops.aten.view.default(mm_150, [2, 128, 512]);  mm_150 = None
        permute_278 = torch.ops.aten.permute.default(permute_276, [1, 0]);  permute_276 = None
        mul_389 = torch.ops.aten.mul.Tensor(mul_388, mul_356);  mul_356 = None
        mul_390 = torch.ops.aten.mul.Tensor(mul_388, add_159);  mul_388 = add_159 = None
        alias_339 = torch.ops.aten.alias.default(alias_323);  alias_323 = None
        alias_340 = torch.ops.aten.alias.default(alias_339);  alias_339 = None
        mul_391 = torch.ops.aten.mul.Tensor(alias_340, alias_340);  alias_340 = None
        _tensor_constant3 = self._tensor_constant3
        lift_fresh_copy_3 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant3);  _tensor_constant3 = None
        sub_48 = torch.ops.aten.sub.Tensor(lift_fresh_copy_3, mul_391);  lift_fresh_copy_3 = mul_391 = None
        mul_392 = torch.ops.aten.mul.Tensor(mul_389, sub_48);  mul_389 = sub_48 = None
        mul_393 = torch.ops.aten.mul.Tensor(mul_392, 0.7978845608028654);  mul_392 = None
        mul_394 = torch.ops.aten.mul.Tensor(mul_393, 0.044715)
        pow_61 = torch.ops.aten.pow.Tensor_Scalar(_unsafe_view_261, 2.0);  _unsafe_view_261 = None
        mul_395 = torch.ops.aten.mul.Scalar(pow_61, 3.0);  pow_61 = None
        mul_396 = torch.ops.aten.mul.Tensor(mul_394, mul_395);  mul_394 = mul_395 = None
        add_164 = torch.ops.aten.add.Tensor(mul_393, mul_396);  mul_393 = mul_396 = None
        mul_397 = torch.ops.aten.mul.Tensor(mul_390, 0.5);  mul_390 = None
        add_165 = torch.ops.aten.add.Tensor(add_164, mul_397);  add_164 = mul_397 = None
        view_277 = torch.ops.aten.view.default(add_165, [256, 1024]);  add_165 = None
        permute_279 = torch.ops.aten.permute.default(view_277, [1, 0])
        mm_151 = torch.ops.aten.mm.default(permute_279, view_263);  permute_279 = view_263 = None
        permute_280 = torch.ops.aten.permute.default(mm_151, [1, 0]);  mm_151 = None
        mm_152 = torch.ops.aten.mm.default(view_277, permute_281);  view_277 = permute_281 = None
        view_278 = torch.ops.aten.view.default(mm_152, [2, 128, 512]);  mm_152 = None
        add_166 = torch.ops.aten.add.Tensor(view_276, view_278);  view_276 = view_278 = None
        permute_282 = torch.ops.aten.permute.default(permute_280, [1, 0]);  permute_280 = None
        mul_398 = torch.ops.aten.mul.Tensor(add_166, primals_41);  primals_41 = None
        mul_399 = torch.ops.aten.mul.Tensor(add_166, mul_354);  add_166 = mul_354 = None
        sum_29 = torch.ops.aten.sum.dim_IntList(mul_399, [0, 1], True);  mul_399 = None
        view_279 = torch.ops.aten.view.default(sum_29, [512]);  sum_29 = None
        mul_400 = torch.ops.aten.mul.Tensor(mul_398, add_155)
        mul_401 = torch.ops.aten.mul.Tensor(mul_398, reciprocal_55);  mul_398 = reciprocal_55 = None
        sum_30 = torch.ops.aten.sum.dim_IntList(mul_400, [2], True);  mul_400 = None
        add_167 = torch.ops.aten.add.Tensor(add_163, mul_401);  add_163 = mul_401 = None
        alias_341 = torch.ops.aten.alias.default(alias_320);  alias_320 = None
        alias_342 = torch.ops.aten.alias.default(alias_341);  alias_341 = None
        pow_62 = torch.ops.aten.pow.Tensor_Scalar(alias_342, 3);  alias_342 = None
        mul_402 = torch.ops.aten.mul.Scalar(sum_30, -0.5);  sum_30 = None
        mul_403 = torch.ops.aten.mul.Tensor(mul_402, pow_62);  mul_402 = pow_62 = None
        expand_97 = torch.ops.aten.expand.default(mul_403, [2, 128, 512]);  mul_403 = None
        div_30 = torch.ops.aten.div.Scalar(expand_97, 512);  expand_97 = None
        pow_63 = torch.ops.aten.pow.Tensor_Scalar(add_155, 1.0);  add_155 = None
        mul_404 = torch.ops.aten.mul.Scalar(pow_63, 2.0);  pow_63 = None
        mul_405 = torch.ops.aten.mul.Tensor(div_30, mul_404);  div_30 = mul_404 = None
        add_168 = torch.ops.aten.add.Tensor(add_167, mul_405);  add_167 = mul_405 = None
        _to_copy_34 = torch.ops.aten._to_copy.default(gt_81, dtype = torch.float32);  gt_81 = None
        mul_406 = torch.ops.aten.mul.Tensor(_to_copy_34, 1.1111111111111112);  _to_copy_34 = None
        mul_407 = torch.ops.aten.mul.Tensor(add_168, mul_406);  mul_406 = None
        view_280 = torch.ops.aten.view.default(mul_407, [256, 512]);  mul_407 = None
        permute_283 = torch.ops.aten.permute.default(view_280, [1, 0])
        mm_153 = torch.ops.aten.mm.default(permute_283, view_262);  permute_283 = view_262 = None
        permute_284 = torch.ops.aten.permute.default(mm_153, [1, 0]);  mm_153 = None
        mm_154 = torch.ops.aten.mm.default(view_280, permute_285);  view_280 = permute_285 = None
        view_281 = torch.ops.aten.view.default(mm_154, [2, 128, 384]);  mm_154 = None
        permute_286 = torch.ops.aten.permute.default(permute_284, [1, 0]);  permute_284 = None
        view_282 = torch.ops.aten.view.default(view_281, [2, 128, 6, 64]);  view_281 = None
        permute_287 = torch.ops.aten.permute.default(view_282, [0, 2, 1, 3]);  view_282 = None
        clone_96 = torch.ops.aten.clone.default(permute_287, memory_format = torch.contiguous_format);  permute_287 = None
        _unsafe_view_265 = torch.ops.aten._unsafe_view.default(clone_96, [12, 128, 64]);  clone_96 = None
        bmm_48 = torch.ops.aten.bmm.default(permute_288, _unsafe_view_265);  permute_288 = None
        bmm_49 = torch.ops.aten.bmm.default(_unsafe_view_265, permute_289);  _unsafe_view_265 = permute_289 = None
        view_283 = torch.ops.aten.view.default(bmm_48, [2, 6, 128, 64]);  bmm_48 = None
        add_169 = torch.ops.aten.add.Tensor(tangents_34, view_283);  tangents_34 = view_283 = None
        view_284 = torch.ops.aten.view.default(bmm_49, [2, 6, 128, 128]);  bmm_49 = None
        philox_rand_like_24 = torch.ops.prims.philox_rand_like.default(view_284, philox_seed_like, 4521984)
        gt_85 = torch.ops.aten.gt.Scalar(philox_rand_like_24, 0.1);  philox_rand_like_24 = None
        _to_copy_35 = torch.ops.aten._to_copy.default(gt_85, dtype = torch.float32);  gt_85 = None
        mul_408 = torch.ops.aten.mul.Tensor(_to_copy_35, view_284);  _to_copy_35 = view_284 = None
        mul_409 = torch.ops.aten.mul.Tensor(mul_408, 1.1111111111111112);  mul_408 = None
        alias_343 = torch.ops.aten.alias.default(alias_316);  alias_316 = None
        alias_344 = torch.ops.aten.alias.default(alias_343);  alias_343 = None
        mul_410 = torch.ops.aten.mul.Tensor(mul_409, alias_344);  mul_409 = None
        sum_31 = torch.ops.aten.sum.dim_IntList(mul_410, [-1], True)
        mul_411 = torch.ops.aten.mul.Tensor(alias_344, sum_31);  alias_344 = sum_31 = None
        sub_49 = torch.ops.aten.sub.Tensor(mul_410, mul_411);  mul_410 = mul_411 = None
        view_285 = torch.ops.aten.view.default(sub_49, [12, 128, 128]);  sub_49 = None
        bmm_50 = torch.ops.aten.bmm.default(permute_290, view_285);  permute_290 = None
        bmm_51 = torch.ops.aten.bmm.default(view_285, permute_291);  view_285 = permute_291 = None
        view_286 = torch.ops.aten.view.default(bmm_50, [2, 6, 64, 128]);  bmm_50 = None
        view_287 = torch.ops.aten.view.default(bmm_51, [2, 6, 128, 64]);  bmm_51 = None
        permute_292 = torch.ops.aten.permute.default(view_286, [0, 1, 3, 2]);  view_286 = None
        add_170 = torch.ops.aten.add.Tensor(tangents_33, permute_292);  tangents_33 = permute_292 = None
        permute_293 = torch.ops.aten.permute.default(add_169, [0, 2, 1, 3]);  add_169 = None
        clone_97 = torch.ops.aten.clone.default(permute_293, memory_format = torch.contiguous_format);  permute_293 = None
        _unsafe_view_266 = torch.ops.aten._unsafe_view.default(clone_97, [2, 128, 384]);  clone_97 = None
        view_288 = torch.ops.aten.view.default(_unsafe_view_266, [256, 384]);  _unsafe_view_266 = None
        permute_294 = torch.ops.aten.permute.default(view_288, [1, 0])
        mm_155 = torch.ops.aten.mm.default(permute_294, view_109);  permute_294 = None
        permute_295 = torch.ops.aten.permute.default(mm_155, [1, 0]);  mm_155 = None
        mm_156 = torch.ops.aten.mm.default(view_288, permute_296);  view_288 = permute_296 = None
        view_289 = torch.ops.aten.view.default(mm_156, [2, 128, 512]);  mm_156 = None
        add_171 = torch.ops.aten.add.Tensor(tangents_35, view_289);  tangents_35 = view_289 = None
        permute_297 = torch.ops.aten.permute.default(permute_295, [1, 0]);  permute_295 = None
        permute_298 = torch.ops.aten.permute.default(add_170, [0, 2, 1, 3]);  add_170 = None
        clone_98 = torch.ops.aten.clone.default(permute_298, memory_format = torch.contiguous_format);  permute_298 = None
        _unsafe_view_267 = torch.ops.aten._unsafe_view.default(clone_98, [2, 128, 384]);  clone_98 = None
        view_290 = torch.ops.aten.view.default(_unsafe_view_267, [256, 384]);  _unsafe_view_267 = None
        permute_299 = torch.ops.aten.permute.default(view_290, [1, 0])
        mm_157 = torch.ops.aten.mm.default(permute_299, view_109);  permute_299 = None
        permute_300 = torch.ops.aten.permute.default(mm_157, [1, 0]);  mm_157 = None
        mm_158 = torch.ops.aten.mm.default(view_290, permute_301);  view_290 = permute_301 = None
        view_291 = torch.ops.aten.view.default(mm_158, [2, 128, 512]);  mm_158 = None
        add_172 = torch.ops.aten.add.Tensor(add_171, view_291);  add_171 = view_291 = None
        permute_302 = torch.ops.aten.permute.default(permute_300, [1, 0]);  permute_300 = None
        permute_303 = torch.ops.aten.permute.default(view_287, [0, 2, 1, 3]);  view_287 = None
        clone_99 = torch.ops.aten.clone.default(permute_303, memory_format = torch.contiguous_format);  permute_303 = None
        _unsafe_view_268 = torch.ops.aten._unsafe_view.default(clone_99, [2, 128, 384]);  clone_99 = None
        view_292 = torch.ops.aten.view.default(_unsafe_view_268, [256, 384]);  _unsafe_view_268 = None
        permute_304 = torch.ops.aten.permute.default(view_292, [1, 0])
        mm_159 = torch.ops.aten.mm.default(permute_304, view_254);  permute_304 = view_254 = None
        permute_305 = torch.ops.aten.permute.default(mm_159, [1, 0]);  mm_159 = None
        mm_160 = torch.ops.aten.mm.default(view_292, permute_306);  view_292 = permute_306 = None
        view_293 = torch.ops.aten.view.default(mm_160, [2, 128, 512]);  mm_160 = None
        permute_307 = torch.ops.aten.permute.default(permute_305, [1, 0]);  permute_305 = None
        mul_412 = torch.ops.aten.mul.Tensor(view_293, primals_40);  primals_40 = None
        mul_413 = torch.ops.aten.mul.Tensor(view_293, mul_348);  view_293 = mul_348 = None
        sum_32 = torch.ops.aten.sum.dim_IntList(mul_413, [0, 1], True);  mul_413 = None
        view_294 = torch.ops.aten.view.default(sum_32, [512]);  sum_32 = None
        mul_414 = torch.ops.aten.mul.Tensor(mul_412, add_152)
        mul_415 = torch.ops.aten.mul.Tensor(mul_412, reciprocal_54);  mul_412 = reciprocal_54 = None
        sum_33 = torch.ops.aten.sum.dim_IntList(mul_414, [2], True);  mul_414 = None
        add_173 = torch.ops.aten.add.Tensor(add_168, mul_415);  add_168 = mul_415 = None
        alias_345 = torch.ops.aten.alias.default(alias_313);  alias_313 = None
        alias_346 = torch.ops.aten.alias.default(alias_345);  alias_345 = None
        pow_64 = torch.ops.aten.pow.Tensor_Scalar(alias_346, 3);  alias_346 = None
        mul_416 = torch.ops.aten.mul.Scalar(sum_33, -0.5);  sum_33 = None
        mul_417 = torch.ops.aten.mul.Tensor(mul_416, pow_64);  mul_416 = pow_64 = None
        expand_98 = torch.ops.aten.expand.default(mul_417, [2, 128, 512]);  mul_417 = None
        div_31 = torch.ops.aten.div.Scalar(expand_98, 512);  expand_98 = None
        pow_65 = torch.ops.aten.pow.Tensor_Scalar(add_152, 1.0);  add_152 = None
        mul_418 = torch.ops.aten.mul.Scalar(pow_65, 2.0);  pow_65 = None
        mul_419 = torch.ops.aten.mul.Tensor(div_31, mul_418);  div_31 = mul_418 = None
        add_174 = torch.ops.aten.add.Tensor(add_173, mul_419);  add_173 = mul_419 = None
        _to_copy_36 = torch.ops.aten._to_copy.default(gt_79, dtype = torch.float32);  gt_79 = None
        mul_420 = torch.ops.aten.mul.Tensor(_to_copy_36, 1.1111111111111112);  _to_copy_36 = None
        mul_421 = torch.ops.aten.mul.Tensor(add_174, mul_420);  mul_420 = None
        view_295 = torch.ops.aten.view.default(mul_421, [256, 512]);  mul_421 = None
        permute_308 = torch.ops.aten.permute.default(view_295, [1, 0])
        mm_161 = torch.ops.aten.mm.default(permute_308, view_253);  permute_308 = view_253 = None
        permute_309 = torch.ops.aten.permute.default(mm_161, [1, 0]);  mm_161 = None
        mm_162 = torch.ops.aten.mm.default(view_295, permute_310);  view_295 = permute_310 = None
        view_296 = torch.ops.aten.view.default(mm_162, [2, 128, 384]);  mm_162 = None
        permute_311 = torch.ops.aten.permute.default(permute_309, [1, 0]);  permute_309 = None
        view_297 = torch.ops.aten.view.default(view_296, [2, 128, 6, 64]);  view_296 = None
        permute_312 = torch.ops.aten.permute.default(view_297, [0, 2, 1, 3]);  view_297 = None
        clone_100 = torch.ops.aten.clone.default(permute_312, memory_format = torch.contiguous_format);  permute_312 = None
        _unsafe_view_269 = torch.ops.aten._unsafe_view.default(clone_100, [12, 128, 64]);  clone_100 = None
        bmm_52 = torch.ops.aten.bmm.default(permute_313, _unsafe_view_269);  permute_313 = None
        bmm_53 = torch.ops.aten.bmm.default(_unsafe_view_269, permute_314);  _unsafe_view_269 = permute_314 = None
        view_298 = torch.ops.aten.view.default(bmm_52, [2, 6, 128, 64]);  bmm_52 = None
        add_175 = torch.ops.aten.add.Tensor(tangents_32, view_298);  tangents_32 = view_298 = None
        view_299 = torch.ops.aten.view.default(bmm_53, [2, 6, 128, 128]);  bmm_53 = None
        philox_rand_like_25 = torch.ops.prims.philox_rand_like.default(view_299, philox_seed_like, 4325376)
        gt_86 = torch.ops.aten.gt.Scalar(philox_rand_like_25, 0.1);  philox_rand_like_25 = None
        _to_copy_37 = torch.ops.aten._to_copy.default(gt_86, dtype = torch.float32);  gt_86 = None
        mul_422 = torch.ops.aten.mul.Tensor(_to_copy_37, view_299);  _to_copy_37 = view_299 = None
        mul_423 = torch.ops.aten.mul.Tensor(mul_422, 1.1111111111111112);  mul_422 = None
        alias_347 = torch.ops.aten.alias.default(alias_309);  alias_309 = None
        alias_348 = torch.ops.aten.alias.default(alias_347);  alias_347 = None
        mul_424 = torch.ops.aten.mul.Tensor(mul_423, alias_348);  mul_423 = None
        sum_34 = torch.ops.aten.sum.dim_IntList(mul_424, [-1], True)
        mul_425 = torch.ops.aten.mul.Tensor(alias_348, sum_34);  alias_348 = sum_34 = None
        sub_50 = torch.ops.aten.sub.Tensor(mul_424, mul_425);  mul_424 = mul_425 = None
        view_300 = torch.ops.aten.view.default(sub_50, [12, 128, 128])
        bmm_54 = torch.ops.aten.bmm.default(permute_315, view_300);  permute_315 = None
        bmm_55 = torch.ops.aten.bmm.default(view_300, permute_316);  view_300 = permute_316 = None
        view_301 = torch.ops.aten.view.default(bmm_54, [2, 6, 64, 128]);  bmm_54 = None
        view_302 = torch.ops.aten.view.default(bmm_55, [2, 6, 128, 64]);  bmm_55 = None
        permute_317 = torch.ops.aten.permute.default(view_301, [0, 1, 3, 2]);  view_301 = None
        add_176 = torch.ops.aten.add.Tensor(tangents_31, permute_317);  tangents_31 = permute_317 = None
        permute_318 = torch.ops.aten.permute.default(add_175, [0, 2, 1, 3]);  add_175 = None
        clone_101 = torch.ops.aten.clone.default(permute_318, memory_format = torch.contiguous_format);  permute_318 = None
        _unsafe_view_270 = torch.ops.aten._unsafe_view.default(clone_101, [2, 128, 384]);  clone_101 = None
        view_303 = torch.ops.aten.view.default(_unsafe_view_270, [256, 384]);  _unsafe_view_270 = None
        permute_319 = torch.ops.aten.permute.default(view_303, [1, 0])
        mm_163 = torch.ops.aten.mm.default(permute_319, view_245);  permute_319 = None
        permute_320 = torch.ops.aten.permute.default(mm_163, [1, 0]);  mm_163 = None
        mm_164 = torch.ops.aten.mm.default(view_303, permute_321);  view_303 = permute_321 = None
        view_304 = torch.ops.aten.view.default(mm_164, [2, 128, 512]);  mm_164 = None
        permute_322 = torch.ops.aten.permute.default(permute_320, [1, 0]);  permute_320 = None
        permute_323 = torch.ops.aten.permute.default(add_176, [0, 2, 1, 3]);  add_176 = None
        clone_102 = torch.ops.aten.clone.default(permute_323, memory_format = torch.contiguous_format);  permute_323 = None
        _unsafe_view_271 = torch.ops.aten._unsafe_view.default(clone_102, [2, 128, 384]);  clone_102 = None
        view_305 = torch.ops.aten.view.default(_unsafe_view_271, [256, 384]);  _unsafe_view_271 = None
        permute_324 = torch.ops.aten.permute.default(view_305, [1, 0])
        mm_165 = torch.ops.aten.mm.default(permute_324, view_245);  permute_324 = None
        permute_325 = torch.ops.aten.permute.default(mm_165, [1, 0]);  mm_165 = None
        mm_166 = torch.ops.aten.mm.default(view_305, permute_326);  view_305 = permute_326 = None
        view_306 = torch.ops.aten.view.default(mm_166, [2, 128, 512]);  mm_166 = None
        add_177 = torch.ops.aten.add.Tensor(view_304, view_306);  view_304 = view_306 = None
        permute_327 = torch.ops.aten.permute.default(permute_325, [1, 0]);  permute_325 = None
        permute_328 = torch.ops.aten.permute.default(view_302, [0, 2, 1, 3]);  view_302 = None
        clone_103 = torch.ops.aten.clone.default(permute_328, memory_format = torch.contiguous_format);  permute_328 = None
        _unsafe_view_272 = torch.ops.aten._unsafe_view.default(clone_103, [2, 128, 384]);  clone_103 = None
        view_307 = torch.ops.aten.view.default(_unsafe_view_272, [256, 384]);  _unsafe_view_272 = None
        permute_329 = torch.ops.aten.permute.default(view_307, [1, 0])
        mm_167 = torch.ops.aten.mm.default(permute_329, view_245);  permute_329 = view_245 = None
        permute_330 = torch.ops.aten.permute.default(mm_167, [1, 0]);  mm_167 = None
        mm_168 = torch.ops.aten.mm.default(view_307, permute_331);  view_307 = permute_331 = None
        view_308 = torch.ops.aten.view.default(mm_168, [2, 128, 512]);  mm_168 = None
        add_178 = torch.ops.aten.add.Tensor(add_177, view_308);  add_177 = view_308 = None
        permute_332 = torch.ops.aten.permute.default(permute_330, [1, 0]);  permute_330 = None
        mul_426 = torch.ops.aten.mul.Tensor(add_178, primals_39);  primals_39 = None
        mul_427 = torch.ops.aten.mul.Tensor(add_178, mul_342);  add_178 = mul_342 = None
        sum_35 = torch.ops.aten.sum.dim_IntList(mul_427, [0, 1], True);  mul_427 = None
        view_309 = torch.ops.aten.view.default(sum_35, [512]);  sum_35 = None
        mul_428 = torch.ops.aten.mul.Tensor(mul_426, add_149)
        mul_429 = torch.ops.aten.mul.Tensor(mul_426, reciprocal_53);  mul_426 = reciprocal_53 = None
        sum_36 = torch.ops.aten.sum.dim_IntList(mul_428, [2], True);  mul_428 = None
        add_179 = torch.ops.aten.add.Tensor(add_174, mul_429);  add_174 = mul_429 = None
        alias_349 = torch.ops.aten.alias.default(alias_306);  alias_306 = None
        alias_350 = torch.ops.aten.alias.default(alias_349);  alias_349 = None
        pow_66 = torch.ops.aten.pow.Tensor_Scalar(alias_350, 3);  alias_350 = None
        mul_430 = torch.ops.aten.mul.Scalar(sum_36, -0.5);  sum_36 = None
        mul_431 = torch.ops.aten.mul.Tensor(mul_430, pow_66);  mul_430 = pow_66 = None
        expand_99 = torch.ops.aten.expand.default(mul_431, [2, 128, 512]);  mul_431 = None
        div_32 = torch.ops.aten.div.Scalar(expand_99, 512);  expand_99 = None
        pow_67 = torch.ops.aten.pow.Tensor_Scalar(add_149, 1.0);  add_149 = None
        mul_432 = torch.ops.aten.mul.Scalar(pow_67, 2.0);  pow_67 = None
        mul_433 = torch.ops.aten.mul.Tensor(div_32, mul_432);  div_32 = mul_432 = None
        add_180 = torch.ops.aten.add.Tensor(add_179, mul_433);  add_179 = mul_433 = None
        _to_copy_38 = torch.ops.aten._to_copy.default(gt_77, dtype = torch.float32);  gt_77 = None
        mul_434 = torch.ops.aten.mul.Tensor(_to_copy_38, 1.1111111111111112);  _to_copy_38 = None
        mul_435 = torch.ops.aten.mul.Tensor(add_180, mul_434);  mul_434 = None
        view_310 = torch.ops.aten.view.default(mul_435, [256, 512]);  mul_435 = None
        permute_333 = torch.ops.aten.permute.default(view_310, [1, 0])
        mm_169 = torch.ops.aten.mm.default(permute_333, view_244);  permute_333 = view_244 = None
        permute_334 = torch.ops.aten.permute.default(mm_169, [1, 0]);  mm_169 = None
        mm_170 = torch.ops.aten.mm.default(view_310, permute_335);  view_310 = permute_335 = None
        view_311 = torch.ops.aten.view.default(mm_170, [2, 128, 1024]);  mm_170 = None
        permute_336 = torch.ops.aten.permute.default(permute_334, [1, 0]);  permute_334 = None
        _to_copy_39 = torch.ops.aten._to_copy.default(gt_76, dtype = torch.float32);  gt_76 = None
        mul_436 = torch.ops.aten.mul.Tensor(_to_copy_39, 1.1111111111111112);  _to_copy_39 = None
        mul_437 = torch.ops.aten.mul.Tensor(view_311, mul_436);  view_311 = mul_436 = None
        mul_438 = torch.ops.aten.mul.Tensor(mul_437, mul_336);  mul_336 = None
        mul_439 = torch.ops.aten.mul.Tensor(mul_437, _unsafe_view_241);  mul_437 = _unsafe_view_241 = None
        view_312 = torch.ops.aten.view.default(mul_438, [256, 1024]);  mul_438 = None
        permute_337 = torch.ops.aten.permute.default(view_312, [1, 0])
        mm_171 = torch.ops.aten.mm.default(permute_337, view_242);  permute_337 = None
        permute_338 = torch.ops.aten.permute.default(mm_171, [1, 0]);  mm_171 = None
        mm_172 = torch.ops.aten.mm.default(view_312, permute_339);  view_312 = permute_339 = None
        view_313 = torch.ops.aten.view.default(mm_172, [2, 128, 512]);  mm_172 = None
        permute_340 = torch.ops.aten.permute.default(permute_338, [1, 0]);  permute_338 = None
        mul_440 = torch.ops.aten.mul.Tensor(mul_439, mul_331);  mul_331 = None
        mul_441 = torch.ops.aten.mul.Tensor(mul_439, add_148);  mul_439 = add_148 = None
        alias_351 = torch.ops.aten.alias.default(alias_301);  alias_301 = None
        alias_352 = torch.ops.aten.alias.default(alias_351);  alias_351 = None
        mul_442 = torch.ops.aten.mul.Tensor(alias_352, alias_352);  alias_352 = None
        _tensor_constant4 = self._tensor_constant4
        lift_fresh_copy_4 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant4);  _tensor_constant4 = None
        sub_51 = torch.ops.aten.sub.Tensor(lift_fresh_copy_4, mul_442);  lift_fresh_copy_4 = mul_442 = None
        mul_443 = torch.ops.aten.mul.Tensor(mul_440, sub_51);  mul_440 = sub_51 = None
        mul_444 = torch.ops.aten.mul.Tensor(mul_443, 0.7978845608028654);  mul_443 = None
        mul_445 = torch.ops.aten.mul.Tensor(mul_444, 0.044715)
        pow_68 = torch.ops.aten.pow.Tensor_Scalar(_unsafe_view_240, 2.0);  _unsafe_view_240 = None
        mul_446 = torch.ops.aten.mul.Scalar(pow_68, 3.0);  pow_68 = None
        mul_447 = torch.ops.aten.mul.Tensor(mul_445, mul_446);  mul_445 = mul_446 = None
        add_181 = torch.ops.aten.add.Tensor(mul_444, mul_447);  mul_444 = mul_447 = None
        mul_448 = torch.ops.aten.mul.Tensor(mul_441, 0.5);  mul_441 = None
        add_182 = torch.ops.aten.add.Tensor(add_181, mul_448);  add_181 = mul_448 = None
        view_314 = torch.ops.aten.view.default(add_182, [256, 1024]);  add_182 = None
        permute_341 = torch.ops.aten.permute.default(view_314, [1, 0])
        mm_173 = torch.ops.aten.mm.default(permute_341, view_242);  permute_341 = view_242 = None
        permute_342 = torch.ops.aten.permute.default(mm_173, [1, 0]);  mm_173 = None
        mm_174 = torch.ops.aten.mm.default(view_314, permute_343);  view_314 = permute_343 = None
        view_315 = torch.ops.aten.view.default(mm_174, [2, 128, 512]);  mm_174 = None
        add_183 = torch.ops.aten.add.Tensor(view_313, view_315);  view_313 = view_315 = None
        permute_344 = torch.ops.aten.permute.default(permute_342, [1, 0]);  permute_342 = None
        mul_449 = torch.ops.aten.mul.Tensor(add_183, primals_38);  primals_38 = None
        mul_450 = torch.ops.aten.mul.Tensor(add_183, mul_329);  add_183 = mul_329 = None
        sum_37 = torch.ops.aten.sum.dim_IntList(mul_450, [0, 1], True);  mul_450 = None
        view_316 = torch.ops.aten.view.default(sum_37, [512]);  sum_37 = None
        mul_451 = torch.ops.aten.mul.Tensor(mul_449, add_144)
        mul_452 = torch.ops.aten.mul.Tensor(mul_449, reciprocal_51);  mul_449 = reciprocal_51 = None
        sum_38 = torch.ops.aten.sum.dim_IntList(mul_451, [2], True);  mul_451 = None
        add_184 = torch.ops.aten.add.Tensor(add_180, mul_452);  add_180 = mul_452 = None
        alias_353 = torch.ops.aten.alias.default(alias_298);  alias_298 = None
        alias_354 = torch.ops.aten.alias.default(alias_353);  alias_353 = None
        pow_69 = torch.ops.aten.pow.Tensor_Scalar(alias_354, 3);  alias_354 = None
        mul_453 = torch.ops.aten.mul.Scalar(sum_38, -0.5);  sum_38 = None
        mul_454 = torch.ops.aten.mul.Tensor(mul_453, pow_69);  mul_453 = pow_69 = None
        expand_100 = torch.ops.aten.expand.default(mul_454, [2, 128, 512]);  mul_454 = None
        div_33 = torch.ops.aten.div.Scalar(expand_100, 512);  expand_100 = None
        pow_70 = torch.ops.aten.pow.Tensor_Scalar(add_144, 1.0);  add_144 = None
        mul_455 = torch.ops.aten.mul.Scalar(pow_70, 2.0);  pow_70 = None
        mul_456 = torch.ops.aten.mul.Tensor(div_33, mul_455);  div_33 = mul_455 = None
        add_185 = torch.ops.aten.add.Tensor(add_184, mul_456);  add_184 = mul_456 = None
        _to_copy_40 = torch.ops.aten._to_copy.default(gt_75, dtype = torch.float32);  gt_75 = None
        mul_457 = torch.ops.aten.mul.Tensor(_to_copy_40, 1.1111111111111112);  _to_copy_40 = None
        mul_458 = torch.ops.aten.mul.Tensor(add_185, mul_457);  mul_457 = None
        view_317 = torch.ops.aten.view.default(mul_458, [256, 512]);  mul_458 = None
        permute_345 = torch.ops.aten.permute.default(view_317, [1, 0])
        mm_175 = torch.ops.aten.mm.default(permute_345, view_241);  permute_345 = view_241 = None
        permute_346 = torch.ops.aten.permute.default(mm_175, [1, 0]);  mm_175 = None
        mm_176 = torch.ops.aten.mm.default(view_317, permute_347);  view_317 = permute_347 = None
        view_318 = torch.ops.aten.view.default(mm_176, [2, 128, 384]);  mm_176 = None
        permute_348 = torch.ops.aten.permute.default(permute_346, [1, 0]);  permute_346 = None
        view_319 = torch.ops.aten.view.default(view_318, [2, 128, 6, 64]);  view_318 = None
        permute_349 = torch.ops.aten.permute.default(view_319, [0, 2, 1, 3]);  view_319 = None
        clone_104 = torch.ops.aten.clone.default(permute_349, memory_format = torch.contiguous_format);  permute_349 = None
        _unsafe_view_273 = torch.ops.aten._unsafe_view.default(clone_104, [12, 128, 64]);  clone_104 = None
        bmm_56 = torch.ops.aten.bmm.default(permute_350, _unsafe_view_273);  permute_350 = None
        bmm_57 = torch.ops.aten.bmm.default(_unsafe_view_273, permute_351);  _unsafe_view_273 = permute_351 = None
        view_320 = torch.ops.aten.view.default(bmm_56, [2, 6, 128, 64]);  bmm_56 = None
        add_186 = torch.ops.aten.add.Tensor(tangents_30, view_320);  tangents_30 = view_320 = None
        view_321 = torch.ops.aten.view.default(bmm_57, [2, 6, 128, 128]);  bmm_57 = None
        philox_rand_like_26 = torch.ops.prims.philox_rand_like.default(view_321, philox_seed_like, 4128768)
        gt_87 = torch.ops.aten.gt.Scalar(philox_rand_like_26, 0.1);  philox_rand_like_26 = None
        _to_copy_41 = torch.ops.aten._to_copy.default(gt_87, dtype = torch.float32);  gt_87 = None
        mul_459 = torch.ops.aten.mul.Tensor(_to_copy_41, view_321);  _to_copy_41 = view_321 = None
        mul_460 = torch.ops.aten.mul.Tensor(mul_459, 1.1111111111111112);  mul_459 = None
        alias_355 = torch.ops.aten.alias.default(alias_294);  alias_294 = None
        alias_356 = torch.ops.aten.alias.default(alias_355);  alias_355 = None
        mul_461 = torch.ops.aten.mul.Tensor(mul_460, alias_356);  mul_460 = None
        sum_39 = torch.ops.aten.sum.dim_IntList(mul_461, [-1], True)
        mul_462 = torch.ops.aten.mul.Tensor(alias_356, sum_39);  alias_356 = sum_39 = None
        sub_52 = torch.ops.aten.sub.Tensor(mul_461, mul_462);  mul_461 = mul_462 = None
        view_322 = torch.ops.aten.view.default(sub_52, [12, 128, 128]);  sub_52 = None
        bmm_58 = torch.ops.aten.bmm.default(permute_352, view_322);  permute_352 = None
        bmm_59 = torch.ops.aten.bmm.default(view_322, permute_353);  view_322 = permute_353 = None
        view_323 = torch.ops.aten.view.default(bmm_58, [2, 6, 64, 128]);  bmm_58 = None
        view_324 = torch.ops.aten.view.default(bmm_59, [2, 6, 128, 64]);  bmm_59 = None
        permute_354 = torch.ops.aten.permute.default(view_323, [0, 1, 3, 2]);  view_323 = None
        add_187 = torch.ops.aten.add.Tensor(tangents_29, permute_354);  tangents_29 = permute_354 = None
        permute_355 = torch.ops.aten.permute.default(add_186, [0, 2, 1, 3]);  add_186 = None
        clone_105 = torch.ops.aten.clone.default(permute_355, memory_format = torch.contiguous_format);  permute_355 = None
        _unsafe_view_274 = torch.ops.aten._unsafe_view.default(clone_105, [2, 128, 384]);  clone_105 = None
        view_325 = torch.ops.aten.view.default(_unsafe_view_274, [256, 384]);  _unsafe_view_274 = None
        permute_356 = torch.ops.aten.permute.default(view_325, [1, 0])
        mm_177 = torch.ops.aten.mm.default(permute_356, view_109);  permute_356 = None
        permute_357 = torch.ops.aten.permute.default(mm_177, [1, 0]);  mm_177 = None
        mm_178 = torch.ops.aten.mm.default(view_325, permute_358);  view_325 = permute_358 = None
        view_326 = torch.ops.aten.view.default(mm_178, [2, 128, 512]);  mm_178 = None
        add_188 = torch.ops.aten.add.Tensor(add_172, view_326);  add_172 = view_326 = None
        permute_359 = torch.ops.aten.permute.default(permute_357, [1, 0]);  permute_357 = None
        permute_360 = torch.ops.aten.permute.default(add_187, [0, 2, 1, 3]);  add_187 = None
        clone_106 = torch.ops.aten.clone.default(permute_360, memory_format = torch.contiguous_format);  permute_360 = None
        _unsafe_view_275 = torch.ops.aten._unsafe_view.default(clone_106, [2, 128, 384]);  clone_106 = None
        view_327 = torch.ops.aten.view.default(_unsafe_view_275, [256, 384]);  _unsafe_view_275 = None
        permute_361 = torch.ops.aten.permute.default(view_327, [1, 0])
        mm_179 = torch.ops.aten.mm.default(permute_361, view_109);  permute_361 = None
        permute_362 = torch.ops.aten.permute.default(mm_179, [1, 0]);  mm_179 = None
        mm_180 = torch.ops.aten.mm.default(view_327, permute_363);  view_327 = permute_363 = None
        view_328 = torch.ops.aten.view.default(mm_180, [2, 128, 512]);  mm_180 = None
        add_189 = torch.ops.aten.add.Tensor(add_188, view_328);  add_188 = view_328 = None
        permute_364 = torch.ops.aten.permute.default(permute_362, [1, 0]);  permute_362 = None
        permute_365 = torch.ops.aten.permute.default(view_324, [0, 2, 1, 3]);  view_324 = None
        clone_107 = torch.ops.aten.clone.default(permute_365, memory_format = torch.contiguous_format);  permute_365 = None
        _unsafe_view_276 = torch.ops.aten._unsafe_view.default(clone_107, [2, 128, 384]);  clone_107 = None
        view_329 = torch.ops.aten.view.default(_unsafe_view_276, [256, 384]);  _unsafe_view_276 = None
        permute_366 = torch.ops.aten.permute.default(view_329, [1, 0])
        mm_181 = torch.ops.aten.mm.default(permute_366, view_233);  permute_366 = view_233 = None
        permute_367 = torch.ops.aten.permute.default(mm_181, [1, 0]);  mm_181 = None
        mm_182 = torch.ops.aten.mm.default(view_329, permute_368);  view_329 = permute_368 = None
        view_330 = torch.ops.aten.view.default(mm_182, [2, 128, 512]);  mm_182 = None
        permute_369 = torch.ops.aten.permute.default(permute_367, [1, 0]);  permute_367 = None
        mul_463 = torch.ops.aten.mul.Tensor(view_330, primals_37);  primals_37 = None
        mul_464 = torch.ops.aten.mul.Tensor(view_330, mul_323);  view_330 = mul_323 = None
        sum_40 = torch.ops.aten.sum.dim_IntList(mul_464, [0, 1], True);  mul_464 = None
        view_331 = torch.ops.aten.view.default(sum_40, [512]);  sum_40 = None
        mul_465 = torch.ops.aten.mul.Tensor(mul_463, add_141)
        mul_466 = torch.ops.aten.mul.Tensor(mul_463, reciprocal_50);  mul_463 = reciprocal_50 = None
        sum_41 = torch.ops.aten.sum.dim_IntList(mul_465, [2], True);  mul_465 = None
        add_190 = torch.ops.aten.add.Tensor(add_185, mul_466);  add_185 = mul_466 = None
        alias_357 = torch.ops.aten.alias.default(alias_291);  alias_291 = None
        alias_358 = torch.ops.aten.alias.default(alias_357);  alias_357 = None
        pow_71 = torch.ops.aten.pow.Tensor_Scalar(alias_358, 3);  alias_358 = None
        mul_467 = torch.ops.aten.mul.Scalar(sum_41, -0.5);  sum_41 = None
        mul_468 = torch.ops.aten.mul.Tensor(mul_467, pow_71);  mul_467 = pow_71 = None
        expand_101 = torch.ops.aten.expand.default(mul_468, [2, 128, 512]);  mul_468 = None
        div_34 = torch.ops.aten.div.Scalar(expand_101, 512);  expand_101 = None
        pow_72 = torch.ops.aten.pow.Tensor_Scalar(add_141, 1.0);  add_141 = None
        mul_469 = torch.ops.aten.mul.Scalar(pow_72, 2.0);  pow_72 = None
        mul_470 = torch.ops.aten.mul.Tensor(div_34, mul_469);  div_34 = mul_469 = None
        add_191 = torch.ops.aten.add.Tensor(add_190, mul_470);  add_190 = mul_470 = None
        _to_copy_42 = torch.ops.aten._to_copy.default(gt_73, dtype = torch.float32);  gt_73 = None
        mul_471 = torch.ops.aten.mul.Tensor(_to_copy_42, 1.1111111111111112);  _to_copy_42 = None
        mul_472 = torch.ops.aten.mul.Tensor(add_191, mul_471);  mul_471 = None
        view_332 = torch.ops.aten.view.default(mul_472, [256, 512]);  mul_472 = None
        permute_370 = torch.ops.aten.permute.default(view_332, [1, 0])
        mm_183 = torch.ops.aten.mm.default(permute_370, view_232);  permute_370 = view_232 = None
        permute_371 = torch.ops.aten.permute.default(mm_183, [1, 0]);  mm_183 = None
        mm_184 = torch.ops.aten.mm.default(view_332, permute_372);  view_332 = permute_372 = None
        view_333 = torch.ops.aten.view.default(mm_184, [2, 128, 384]);  mm_184 = None
        permute_373 = torch.ops.aten.permute.default(permute_371, [1, 0]);  permute_371 = None
        view_334 = torch.ops.aten.view.default(view_333, [2, 128, 6, 64]);  view_333 = None
        permute_374 = torch.ops.aten.permute.default(view_334, [0, 2, 1, 3]);  view_334 = None
        clone_108 = torch.ops.aten.clone.default(permute_374, memory_format = torch.contiguous_format);  permute_374 = None
        _unsafe_view_277 = torch.ops.aten._unsafe_view.default(clone_108, [12, 128, 64]);  clone_108 = None
        bmm_60 = torch.ops.aten.bmm.default(permute_375, _unsafe_view_277);  permute_375 = None
        bmm_61 = torch.ops.aten.bmm.default(_unsafe_view_277, permute_376);  _unsafe_view_277 = permute_376 = None
        view_335 = torch.ops.aten.view.default(bmm_60, [2, 6, 128, 64]);  bmm_60 = None
        add_192 = torch.ops.aten.add.Tensor(tangents_28, view_335);  tangents_28 = view_335 = None
        view_336 = torch.ops.aten.view.default(bmm_61, [2, 6, 128, 128]);  bmm_61 = None
        philox_rand_like_27 = torch.ops.prims.philox_rand_like.default(view_336, philox_seed_like, 3932160)
        gt_88 = torch.ops.aten.gt.Scalar(philox_rand_like_27, 0.1);  philox_rand_like_27 = None
        _to_copy_43 = torch.ops.aten._to_copy.default(gt_88, dtype = torch.float32);  gt_88 = None
        mul_473 = torch.ops.aten.mul.Tensor(_to_copy_43, view_336);  _to_copy_43 = view_336 = None
        mul_474 = torch.ops.aten.mul.Tensor(mul_473, 1.1111111111111112);  mul_473 = None
        alias_359 = torch.ops.aten.alias.default(alias_287);  alias_287 = None
        alias_360 = torch.ops.aten.alias.default(alias_359);  alias_359 = None
        mul_475 = torch.ops.aten.mul.Tensor(mul_474, alias_360);  mul_474 = None
        sum_42 = torch.ops.aten.sum.dim_IntList(mul_475, [-1], True)
        mul_476 = torch.ops.aten.mul.Tensor(alias_360, sum_42);  alias_360 = sum_42 = None
        sub_53 = torch.ops.aten.sub.Tensor(mul_475, mul_476);  mul_475 = mul_476 = None
        add_193 = torch.ops.aten.add.Tensor(sub_50, sub_53);  sub_50 = None
        view_337 = torch.ops.aten.view.default(sub_53, [12, 128, 128]);  sub_53 = None
        bmm_62 = torch.ops.aten.bmm.default(permute_377, view_337);  permute_377 = None
        bmm_63 = torch.ops.aten.bmm.default(view_337, permute_378);  view_337 = permute_378 = None
        view_338 = torch.ops.aten.view.default(bmm_62, [2, 6, 64, 128]);  bmm_62 = None
        view_339 = torch.ops.aten.view.default(bmm_63, [2, 6, 128, 64]);  bmm_63 = None
        permute_379 = torch.ops.aten.permute.default(view_338, [0, 1, 3, 2]);  view_338 = None
        add_194 = torch.ops.aten.add.Tensor(tangents_27, permute_379);  tangents_27 = permute_379 = None
        permute_380 = torch.ops.aten.permute.default(add_192, [0, 2, 1, 3]);  add_192 = None
        clone_109 = torch.ops.aten.clone.default(permute_380, memory_format = torch.contiguous_format);  permute_380 = None
        _unsafe_view_278 = torch.ops.aten._unsafe_view.default(clone_109, [2, 128, 384]);  clone_109 = None
        view_340 = torch.ops.aten.view.default(_unsafe_view_278, [256, 384]);  _unsafe_view_278 = None
        permute_381 = torch.ops.aten.permute.default(view_340, [1, 0])
        mm_185 = torch.ops.aten.mm.default(permute_381, view_224);  permute_381 = None
        permute_382 = torch.ops.aten.permute.default(mm_185, [1, 0]);  mm_185 = None
        mm_186 = torch.ops.aten.mm.default(view_340, permute_383);  view_340 = permute_383 = None
        view_341 = torch.ops.aten.view.default(mm_186, [2, 128, 512]);  mm_186 = None
        permute_384 = torch.ops.aten.permute.default(permute_382, [1, 0]);  permute_382 = None
        permute_385 = torch.ops.aten.permute.default(add_194, [0, 2, 1, 3]);  add_194 = None
        clone_110 = torch.ops.aten.clone.default(permute_385, memory_format = torch.contiguous_format);  permute_385 = None
        _unsafe_view_279 = torch.ops.aten._unsafe_view.default(clone_110, [2, 128, 384]);  clone_110 = None
        view_342 = torch.ops.aten.view.default(_unsafe_view_279, [256, 384]);  _unsafe_view_279 = None
        permute_386 = torch.ops.aten.permute.default(view_342, [1, 0])
        mm_187 = torch.ops.aten.mm.default(permute_386, view_224);  permute_386 = None
        permute_387 = torch.ops.aten.permute.default(mm_187, [1, 0]);  mm_187 = None
        mm_188 = torch.ops.aten.mm.default(view_342, permute_388);  view_342 = permute_388 = None
        view_343 = torch.ops.aten.view.default(mm_188, [2, 128, 512]);  mm_188 = None
        add_195 = torch.ops.aten.add.Tensor(view_341, view_343);  view_341 = view_343 = None
        permute_389 = torch.ops.aten.permute.default(permute_387, [1, 0]);  permute_387 = None
        permute_390 = torch.ops.aten.permute.default(view_339, [0, 2, 1, 3]);  view_339 = None
        clone_111 = torch.ops.aten.clone.default(permute_390, memory_format = torch.contiguous_format);  permute_390 = None
        _unsafe_view_280 = torch.ops.aten._unsafe_view.default(clone_111, [2, 128, 384]);  clone_111 = None
        view_344 = torch.ops.aten.view.default(_unsafe_view_280, [256, 384]);  _unsafe_view_280 = None
        permute_391 = torch.ops.aten.permute.default(view_344, [1, 0])
        mm_189 = torch.ops.aten.mm.default(permute_391, view_224);  permute_391 = view_224 = None
        permute_392 = torch.ops.aten.permute.default(mm_189, [1, 0]);  mm_189 = None
        mm_190 = torch.ops.aten.mm.default(view_344, permute_393);  view_344 = permute_393 = None
        view_345 = torch.ops.aten.view.default(mm_190, [2, 128, 512]);  mm_190 = None
        add_196 = torch.ops.aten.add.Tensor(add_195, view_345);  add_195 = view_345 = None
        permute_394 = torch.ops.aten.permute.default(permute_392, [1, 0]);  permute_392 = None
        mul_477 = torch.ops.aten.mul.Tensor(add_196, primals_36);  primals_36 = None
        mul_478 = torch.ops.aten.mul.Tensor(add_196, mul_317);  add_196 = mul_317 = None
        sum_43 = torch.ops.aten.sum.dim_IntList(mul_478, [0, 1], True);  mul_478 = None
        view_346 = torch.ops.aten.view.default(sum_43, [512]);  sum_43 = None
        mul_479 = torch.ops.aten.mul.Tensor(mul_477, add_138)
        mul_480 = torch.ops.aten.mul.Tensor(mul_477, reciprocal_49);  mul_477 = reciprocal_49 = None
        sum_44 = torch.ops.aten.sum.dim_IntList(mul_479, [2], True);  mul_479 = None
        add_197 = torch.ops.aten.add.Tensor(add_191, mul_480);  add_191 = mul_480 = None
        alias_361 = torch.ops.aten.alias.default(alias_284);  alias_284 = None
        alias_362 = torch.ops.aten.alias.default(alias_361);  alias_361 = None
        pow_73 = torch.ops.aten.pow.Tensor_Scalar(alias_362, 3);  alias_362 = None
        mul_481 = torch.ops.aten.mul.Scalar(sum_44, -0.5);  sum_44 = None
        mul_482 = torch.ops.aten.mul.Tensor(mul_481, pow_73);  mul_481 = pow_73 = None
        expand_102 = torch.ops.aten.expand.default(mul_482, [2, 128, 512]);  mul_482 = None
        div_35 = torch.ops.aten.div.Scalar(expand_102, 512);  expand_102 = None
        pow_74 = torch.ops.aten.pow.Tensor_Scalar(add_138, 1.0);  add_138 = None
        mul_483 = torch.ops.aten.mul.Scalar(pow_74, 2.0);  pow_74 = None
        mul_484 = torch.ops.aten.mul.Tensor(div_35, mul_483);  div_35 = mul_483 = None
        add_198 = torch.ops.aten.add.Tensor(add_197, mul_484);  add_197 = mul_484 = None
        _to_copy_44 = torch.ops.aten._to_copy.default(gt_71, dtype = torch.float32);  gt_71 = None
        mul_485 = torch.ops.aten.mul.Tensor(_to_copy_44, 1.1111111111111112);  _to_copy_44 = None
        mul_486 = torch.ops.aten.mul.Tensor(add_198, mul_485);  mul_485 = None
        view_347 = torch.ops.aten.view.default(mul_486, [256, 512]);  mul_486 = None
        permute_395 = torch.ops.aten.permute.default(view_347, [1, 0])
        mm_191 = torch.ops.aten.mm.default(permute_395, view_223);  permute_395 = view_223 = None
        permute_396 = torch.ops.aten.permute.default(mm_191, [1, 0]);  mm_191 = None
        mm_192 = torch.ops.aten.mm.default(view_347, permute_397);  view_347 = permute_397 = None
        view_348 = torch.ops.aten.view.default(mm_192, [2, 128, 1024]);  mm_192 = None
        permute_398 = torch.ops.aten.permute.default(permute_396, [1, 0]);  permute_396 = None
        _to_copy_45 = torch.ops.aten._to_copy.default(gt_70, dtype = torch.float32);  gt_70 = None
        mul_487 = torch.ops.aten.mul.Tensor(_to_copy_45, 1.1111111111111112);  _to_copy_45 = None
        mul_488 = torch.ops.aten.mul.Tensor(view_348, mul_487);  view_348 = mul_487 = None
        mul_489 = torch.ops.aten.mul.Tensor(mul_488, mul_311);  mul_311 = None
        mul_490 = torch.ops.aten.mul.Tensor(mul_488, _unsafe_view_220);  mul_488 = _unsafe_view_220 = None
        view_349 = torch.ops.aten.view.default(mul_489, [256, 1024]);  mul_489 = None
        permute_399 = torch.ops.aten.permute.default(view_349, [1, 0])
        mm_193 = torch.ops.aten.mm.default(permute_399, view_221);  permute_399 = None
        permute_400 = torch.ops.aten.permute.default(mm_193, [1, 0]);  mm_193 = None
        mm_194 = torch.ops.aten.mm.default(view_349, permute_401);  view_349 = permute_401 = None
        view_350 = torch.ops.aten.view.default(mm_194, [2, 128, 512]);  mm_194 = None
        permute_402 = torch.ops.aten.permute.default(permute_400, [1, 0]);  permute_400 = None
        mul_491 = torch.ops.aten.mul.Tensor(mul_490, mul_306);  mul_306 = None
        mul_492 = torch.ops.aten.mul.Tensor(mul_490, add_137);  mul_490 = add_137 = None
        alias_363 = torch.ops.aten.alias.default(alias_279);  alias_279 = None
        alias_364 = torch.ops.aten.alias.default(alias_363);  alias_363 = None
        mul_493 = torch.ops.aten.mul.Tensor(alias_364, alias_364);  alias_364 = None
        _tensor_constant5 = self._tensor_constant5
        lift_fresh_copy_5 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant5);  _tensor_constant5 = None
        sub_54 = torch.ops.aten.sub.Tensor(lift_fresh_copy_5, mul_493);  lift_fresh_copy_5 = mul_493 = None
        mul_494 = torch.ops.aten.mul.Tensor(mul_491, sub_54);  mul_491 = sub_54 = None
        mul_495 = torch.ops.aten.mul.Tensor(mul_494, 0.7978845608028654);  mul_494 = None
        mul_496 = torch.ops.aten.mul.Tensor(mul_495, 0.044715)
        pow_75 = torch.ops.aten.pow.Tensor_Scalar(_unsafe_view_219, 2.0);  _unsafe_view_219 = None
        mul_497 = torch.ops.aten.mul.Scalar(pow_75, 3.0);  pow_75 = None
        mul_498 = torch.ops.aten.mul.Tensor(mul_496, mul_497);  mul_496 = mul_497 = None
        add_199 = torch.ops.aten.add.Tensor(mul_495, mul_498);  mul_495 = mul_498 = None
        mul_499 = torch.ops.aten.mul.Tensor(mul_492, 0.5);  mul_492 = None
        add_200 = torch.ops.aten.add.Tensor(add_199, mul_499);  add_199 = mul_499 = None
        view_351 = torch.ops.aten.view.default(add_200, [256, 1024]);  add_200 = None
        permute_403 = torch.ops.aten.permute.default(view_351, [1, 0])
        mm_195 = torch.ops.aten.mm.default(permute_403, view_221);  permute_403 = view_221 = None
        permute_404 = torch.ops.aten.permute.default(mm_195, [1, 0]);  mm_195 = None
        mm_196 = torch.ops.aten.mm.default(view_351, permute_405);  view_351 = permute_405 = None
        view_352 = torch.ops.aten.view.default(mm_196, [2, 128, 512]);  mm_196 = None
        add_201 = torch.ops.aten.add.Tensor(view_350, view_352);  view_350 = view_352 = None
        permute_406 = torch.ops.aten.permute.default(permute_404, [1, 0]);  permute_404 = None
        mul_500 = torch.ops.aten.mul.Tensor(add_201, primals_35);  primals_35 = None
        mul_501 = torch.ops.aten.mul.Tensor(add_201, mul_304);  add_201 = mul_304 = None
        sum_45 = torch.ops.aten.sum.dim_IntList(mul_501, [0, 1], True);  mul_501 = None
        view_353 = torch.ops.aten.view.default(sum_45, [512]);  sum_45 = None
        mul_502 = torch.ops.aten.mul.Tensor(mul_500, add_133)
        mul_503 = torch.ops.aten.mul.Tensor(mul_500, reciprocal_47);  mul_500 = reciprocal_47 = None
        sum_46 = torch.ops.aten.sum.dim_IntList(mul_502, [2], True);  mul_502 = None
        add_202 = torch.ops.aten.add.Tensor(add_198, mul_503);  add_198 = mul_503 = None
        alias_365 = torch.ops.aten.alias.default(alias_276);  alias_276 = None
        alias_366 = torch.ops.aten.alias.default(alias_365);  alias_365 = None
        pow_76 = torch.ops.aten.pow.Tensor_Scalar(alias_366, 3);  alias_366 = None
        mul_504 = torch.ops.aten.mul.Scalar(sum_46, -0.5);  sum_46 = None
        mul_505 = torch.ops.aten.mul.Tensor(mul_504, pow_76);  mul_504 = pow_76 = None
        expand_103 = torch.ops.aten.expand.default(mul_505, [2, 128, 512]);  mul_505 = None
        div_36 = torch.ops.aten.div.Scalar(expand_103, 512);  expand_103 = None
        pow_77 = torch.ops.aten.pow.Tensor_Scalar(add_133, 1.0);  add_133 = None
        mul_506 = torch.ops.aten.mul.Scalar(pow_77, 2.0);  pow_77 = None
        mul_507 = torch.ops.aten.mul.Tensor(div_36, mul_506);  div_36 = mul_506 = None
        add_203 = torch.ops.aten.add.Tensor(add_202, mul_507);  add_202 = mul_507 = None
        _to_copy_46 = torch.ops.aten._to_copy.default(gt_69, dtype = torch.float32);  gt_69 = None
        mul_508 = torch.ops.aten.mul.Tensor(_to_copy_46, 1.1111111111111112);  _to_copy_46 = None
        mul_509 = torch.ops.aten.mul.Tensor(add_203, mul_508);  mul_508 = None
        view_354 = torch.ops.aten.view.default(mul_509, [256, 512]);  mul_509 = None
        permute_407 = torch.ops.aten.permute.default(view_354, [1, 0])
        mm_197 = torch.ops.aten.mm.default(permute_407, view_220);  permute_407 = view_220 = None
        permute_408 = torch.ops.aten.permute.default(mm_197, [1, 0]);  mm_197 = None
        mm_198 = torch.ops.aten.mm.default(view_354, permute_409);  view_354 = permute_409 = None
        view_355 = torch.ops.aten.view.default(mm_198, [2, 128, 384]);  mm_198 = None
        permute_410 = torch.ops.aten.permute.default(permute_408, [1, 0]);  permute_408 = None
        view_356 = torch.ops.aten.view.default(view_355, [2, 128, 6, 64]);  view_355 = None
        permute_411 = torch.ops.aten.permute.default(view_356, [0, 2, 1, 3]);  view_356 = None
        clone_112 = torch.ops.aten.clone.default(permute_411, memory_format = torch.contiguous_format);  permute_411 = None
        _unsafe_view_281 = torch.ops.aten._unsafe_view.default(clone_112, [12, 128, 64]);  clone_112 = None
        bmm_64 = torch.ops.aten.bmm.default(permute_412, _unsafe_view_281);  permute_412 = None
        bmm_65 = torch.ops.aten.bmm.default(_unsafe_view_281, permute_413);  _unsafe_view_281 = permute_413 = None
        view_357 = torch.ops.aten.view.default(bmm_64, [2, 6, 128, 64]);  bmm_64 = None
        add_204 = torch.ops.aten.add.Tensor(tangents_26, view_357);  tangents_26 = view_357 = None
        view_358 = torch.ops.aten.view.default(bmm_65, [2, 6, 128, 128]);  bmm_65 = None
        philox_rand_like_28 = torch.ops.prims.philox_rand_like.default(view_358, philox_seed_like, 3735552)
        gt_89 = torch.ops.aten.gt.Scalar(philox_rand_like_28, 0.1);  philox_rand_like_28 = None
        _to_copy_47 = torch.ops.aten._to_copy.default(gt_89, dtype = torch.float32);  gt_89 = None
        mul_510 = torch.ops.aten.mul.Tensor(_to_copy_47, view_358);  _to_copy_47 = view_358 = None
        mul_511 = torch.ops.aten.mul.Tensor(mul_510, 1.1111111111111112);  mul_510 = None
        alias_367 = torch.ops.aten.alias.default(alias_272);  alias_272 = None
        alias_368 = torch.ops.aten.alias.default(alias_367);  alias_367 = None
        mul_512 = torch.ops.aten.mul.Tensor(mul_511, alias_368);  mul_511 = None
        sum_47 = torch.ops.aten.sum.dim_IntList(mul_512, [-1], True)
        mul_513 = torch.ops.aten.mul.Tensor(alias_368, sum_47);  alias_368 = sum_47 = None
        sub_55 = torch.ops.aten.sub.Tensor(mul_512, mul_513);  mul_512 = mul_513 = None
        view_359 = torch.ops.aten.view.default(sub_55, [12, 128, 128]);  sub_55 = None
        bmm_66 = torch.ops.aten.bmm.default(permute_414, view_359);  permute_414 = None
        bmm_67 = torch.ops.aten.bmm.default(view_359, permute_415);  view_359 = permute_415 = None
        view_360 = torch.ops.aten.view.default(bmm_66, [2, 6, 64, 128]);  bmm_66 = None
        view_361 = torch.ops.aten.view.default(bmm_67, [2, 6, 128, 64]);  bmm_67 = None
        permute_416 = torch.ops.aten.permute.default(view_360, [0, 1, 3, 2]);  view_360 = None
        add_205 = torch.ops.aten.add.Tensor(tangents_25, permute_416);  tangents_25 = permute_416 = None
        permute_417 = torch.ops.aten.permute.default(add_204, [0, 2, 1, 3]);  add_204 = None
        clone_113 = torch.ops.aten.clone.default(permute_417, memory_format = torch.contiguous_format);  permute_417 = None
        _unsafe_view_282 = torch.ops.aten._unsafe_view.default(clone_113, [2, 128, 384]);  clone_113 = None
        view_362 = torch.ops.aten.view.default(_unsafe_view_282, [256, 384]);  _unsafe_view_282 = None
        permute_418 = torch.ops.aten.permute.default(view_362, [1, 0])
        mm_199 = torch.ops.aten.mm.default(permute_418, view_109);  permute_418 = None
        permute_419 = torch.ops.aten.permute.default(mm_199, [1, 0]);  mm_199 = None
        mm_200 = torch.ops.aten.mm.default(view_362, permute_420);  view_362 = permute_420 = None
        view_363 = torch.ops.aten.view.default(mm_200, [2, 128, 512]);  mm_200 = None
        add_206 = torch.ops.aten.add.Tensor(add_189, view_363);  add_189 = view_363 = None
        permute_421 = torch.ops.aten.permute.default(permute_419, [1, 0]);  permute_419 = None
        permute_422 = torch.ops.aten.permute.default(add_205, [0, 2, 1, 3]);  add_205 = None
        clone_114 = torch.ops.aten.clone.default(permute_422, memory_format = torch.contiguous_format);  permute_422 = None
        _unsafe_view_283 = torch.ops.aten._unsafe_view.default(clone_114, [2, 128, 384]);  clone_114 = None
        view_364 = torch.ops.aten.view.default(_unsafe_view_283, [256, 384]);  _unsafe_view_283 = None
        permute_423 = torch.ops.aten.permute.default(view_364, [1, 0])
        mm_201 = torch.ops.aten.mm.default(permute_423, view_109);  permute_423 = None
        permute_424 = torch.ops.aten.permute.default(mm_201, [1, 0]);  mm_201 = None
        mm_202 = torch.ops.aten.mm.default(view_364, permute_425);  view_364 = permute_425 = None
        view_365 = torch.ops.aten.view.default(mm_202, [2, 128, 512]);  mm_202 = None
        add_207 = torch.ops.aten.add.Tensor(add_206, view_365);  add_206 = view_365 = None
        permute_426 = torch.ops.aten.permute.default(permute_424, [1, 0]);  permute_424 = None
        permute_427 = torch.ops.aten.permute.default(view_361, [0, 2, 1, 3]);  view_361 = None
        clone_115 = torch.ops.aten.clone.default(permute_427, memory_format = torch.contiguous_format);  permute_427 = None
        _unsafe_view_284 = torch.ops.aten._unsafe_view.default(clone_115, [2, 128, 384]);  clone_115 = None
        view_366 = torch.ops.aten.view.default(_unsafe_view_284, [256, 384]);  _unsafe_view_284 = None
        permute_428 = torch.ops.aten.permute.default(view_366, [1, 0])
        mm_203 = torch.ops.aten.mm.default(permute_428, view_212);  permute_428 = view_212 = None
        permute_429 = torch.ops.aten.permute.default(mm_203, [1, 0]);  mm_203 = None
        mm_204 = torch.ops.aten.mm.default(view_366, permute_430);  view_366 = permute_430 = None
        view_367 = torch.ops.aten.view.default(mm_204, [2, 128, 512]);  mm_204 = None
        permute_431 = torch.ops.aten.permute.default(permute_429, [1, 0]);  permute_429 = None
        mul_514 = torch.ops.aten.mul.Tensor(view_367, primals_34);  primals_34 = None
        mul_515 = torch.ops.aten.mul.Tensor(view_367, mul_298);  view_367 = mul_298 = None
        sum_48 = torch.ops.aten.sum.dim_IntList(mul_515, [0, 1], True);  mul_515 = None
        view_368 = torch.ops.aten.view.default(sum_48, [512]);  sum_48 = None
        mul_516 = torch.ops.aten.mul.Tensor(mul_514, add_130)
        mul_517 = torch.ops.aten.mul.Tensor(mul_514, reciprocal_46);  mul_514 = reciprocal_46 = None
        sum_49 = torch.ops.aten.sum.dim_IntList(mul_516, [2], True);  mul_516 = None
        add_208 = torch.ops.aten.add.Tensor(add_203, mul_517);  add_203 = mul_517 = None
        alias_369 = torch.ops.aten.alias.default(alias_269);  alias_269 = None
        alias_370 = torch.ops.aten.alias.default(alias_369);  alias_369 = None
        pow_78 = torch.ops.aten.pow.Tensor_Scalar(alias_370, 3);  alias_370 = None
        mul_518 = torch.ops.aten.mul.Scalar(sum_49, -0.5);  sum_49 = None
        mul_519 = torch.ops.aten.mul.Tensor(mul_518, pow_78);  mul_518 = pow_78 = None
        expand_104 = torch.ops.aten.expand.default(mul_519, [2, 128, 512]);  mul_519 = None
        div_37 = torch.ops.aten.div.Scalar(expand_104, 512);  expand_104 = None
        pow_79 = torch.ops.aten.pow.Tensor_Scalar(add_130, 1.0);  add_130 = None
        mul_520 = torch.ops.aten.mul.Scalar(pow_79, 2.0);  pow_79 = None
        mul_521 = torch.ops.aten.mul.Tensor(div_37, mul_520);  div_37 = mul_520 = None
        add_209 = torch.ops.aten.add.Tensor(add_208, mul_521);  add_208 = mul_521 = None
        _to_copy_48 = torch.ops.aten._to_copy.default(gt_67, dtype = torch.float32);  gt_67 = None
        mul_522 = torch.ops.aten.mul.Tensor(_to_copy_48, 1.1111111111111112);  _to_copy_48 = None
        mul_523 = torch.ops.aten.mul.Tensor(add_209, mul_522);  mul_522 = None
        view_369 = torch.ops.aten.view.default(mul_523, [256, 512]);  mul_523 = None
        permute_432 = torch.ops.aten.permute.default(view_369, [1, 0])
        mm_205 = torch.ops.aten.mm.default(permute_432, view_211);  permute_432 = view_211 = None
        permute_433 = torch.ops.aten.permute.default(mm_205, [1, 0]);  mm_205 = None
        mm_206 = torch.ops.aten.mm.default(view_369, permute_434);  view_369 = permute_434 = None
        view_370 = torch.ops.aten.view.default(mm_206, [2, 128, 384]);  mm_206 = None
        permute_435 = torch.ops.aten.permute.default(permute_433, [1, 0]);  permute_433 = None
        view_371 = torch.ops.aten.view.default(view_370, [2, 128, 6, 64]);  view_370 = None
        permute_436 = torch.ops.aten.permute.default(view_371, [0, 2, 1, 3]);  view_371 = None
        clone_116 = torch.ops.aten.clone.default(permute_436, memory_format = torch.contiguous_format);  permute_436 = None
        _unsafe_view_285 = torch.ops.aten._unsafe_view.default(clone_116, [12, 128, 64]);  clone_116 = None
        bmm_68 = torch.ops.aten.bmm.default(permute_437, _unsafe_view_285);  permute_437 = None
        bmm_69 = torch.ops.aten.bmm.default(_unsafe_view_285, permute_438);  _unsafe_view_285 = permute_438 = None
        view_372 = torch.ops.aten.view.default(bmm_68, [2, 6, 128, 64]);  bmm_68 = None
        add_210 = torch.ops.aten.add.Tensor(tangents_24, view_372);  tangents_24 = view_372 = None
        view_373 = torch.ops.aten.view.default(bmm_69, [2, 6, 128, 128]);  bmm_69 = None
        philox_rand_like_29 = torch.ops.prims.philox_rand_like.default(view_373, philox_seed_like, 3538944)
        gt_90 = torch.ops.aten.gt.Scalar(philox_rand_like_29, 0.1);  philox_rand_like_29 = None
        _to_copy_49 = torch.ops.aten._to_copy.default(gt_90, dtype = torch.float32);  gt_90 = None
        mul_524 = torch.ops.aten.mul.Tensor(_to_copy_49, view_373);  _to_copy_49 = view_373 = None
        mul_525 = torch.ops.aten.mul.Tensor(mul_524, 1.1111111111111112);  mul_524 = None
        alias_371 = torch.ops.aten.alias.default(alias_265);  alias_265 = None
        alias_372 = torch.ops.aten.alias.default(alias_371);  alias_371 = None
        mul_526 = torch.ops.aten.mul.Tensor(mul_525, alias_372);  mul_525 = None
        sum_50 = torch.ops.aten.sum.dim_IntList(mul_526, [-1], True)
        mul_527 = torch.ops.aten.mul.Tensor(alias_372, sum_50);  alias_372 = sum_50 = None
        sub_56 = torch.ops.aten.sub.Tensor(mul_526, mul_527);  mul_526 = mul_527 = None
        add_211 = torch.ops.aten.add.Tensor(add_193, sub_56);  add_193 = None
        view_374 = torch.ops.aten.view.default(sub_56, [12, 128, 128]);  sub_56 = None
        bmm_70 = torch.ops.aten.bmm.default(permute_439, view_374);  permute_439 = None
        bmm_71 = torch.ops.aten.bmm.default(view_374, permute_440);  view_374 = permute_440 = None
        view_375 = torch.ops.aten.view.default(bmm_70, [2, 6, 64, 128]);  bmm_70 = None
        view_376 = torch.ops.aten.view.default(bmm_71, [2, 6, 128, 64]);  bmm_71 = None
        permute_441 = torch.ops.aten.permute.default(view_375, [0, 1, 3, 2]);  view_375 = None
        add_212 = torch.ops.aten.add.Tensor(tangents_23, permute_441);  tangents_23 = permute_441 = None
        permute_442 = torch.ops.aten.permute.default(add_210, [0, 2, 1, 3]);  add_210 = None
        clone_117 = torch.ops.aten.clone.default(permute_442, memory_format = torch.contiguous_format);  permute_442 = None
        _unsafe_view_286 = torch.ops.aten._unsafe_view.default(clone_117, [2, 128, 384]);  clone_117 = None
        view_377 = torch.ops.aten.view.default(_unsafe_view_286, [256, 384]);  _unsafe_view_286 = None
        permute_443 = torch.ops.aten.permute.default(view_377, [1, 0])
        mm_207 = torch.ops.aten.mm.default(permute_443, view_203);  permute_443 = None
        permute_444 = torch.ops.aten.permute.default(mm_207, [1, 0]);  mm_207 = None
        mm_208 = torch.ops.aten.mm.default(view_377, permute_445);  view_377 = permute_445 = None
        view_378 = torch.ops.aten.view.default(mm_208, [2, 128, 512]);  mm_208 = None
        permute_446 = torch.ops.aten.permute.default(permute_444, [1, 0]);  permute_444 = None
        permute_447 = torch.ops.aten.permute.default(add_212, [0, 2, 1, 3]);  add_212 = None
        clone_118 = torch.ops.aten.clone.default(permute_447, memory_format = torch.contiguous_format);  permute_447 = None
        _unsafe_view_287 = torch.ops.aten._unsafe_view.default(clone_118, [2, 128, 384]);  clone_118 = None
        view_379 = torch.ops.aten.view.default(_unsafe_view_287, [256, 384]);  _unsafe_view_287 = None
        permute_448 = torch.ops.aten.permute.default(view_379, [1, 0])
        mm_209 = torch.ops.aten.mm.default(permute_448, view_203);  permute_448 = None
        permute_449 = torch.ops.aten.permute.default(mm_209, [1, 0]);  mm_209 = None
        mm_210 = torch.ops.aten.mm.default(view_379, permute_450);  view_379 = permute_450 = None
        view_380 = torch.ops.aten.view.default(mm_210, [2, 128, 512]);  mm_210 = None
        add_213 = torch.ops.aten.add.Tensor(view_378, view_380);  view_378 = view_380 = None
        permute_451 = torch.ops.aten.permute.default(permute_449, [1, 0]);  permute_449 = None
        permute_452 = torch.ops.aten.permute.default(view_376, [0, 2, 1, 3]);  view_376 = None
        clone_119 = torch.ops.aten.clone.default(permute_452, memory_format = torch.contiguous_format);  permute_452 = None
        _unsafe_view_288 = torch.ops.aten._unsafe_view.default(clone_119, [2, 128, 384]);  clone_119 = None
        view_381 = torch.ops.aten.view.default(_unsafe_view_288, [256, 384]);  _unsafe_view_288 = None
        permute_453 = torch.ops.aten.permute.default(view_381, [1, 0])
        mm_211 = torch.ops.aten.mm.default(permute_453, view_203);  permute_453 = view_203 = None
        permute_454 = torch.ops.aten.permute.default(mm_211, [1, 0]);  mm_211 = None
        mm_212 = torch.ops.aten.mm.default(view_381, permute_455);  view_381 = permute_455 = None
        view_382 = torch.ops.aten.view.default(mm_212, [2, 128, 512]);  mm_212 = None
        add_214 = torch.ops.aten.add.Tensor(add_213, view_382);  add_213 = view_382 = None
        permute_456 = torch.ops.aten.permute.default(permute_454, [1, 0]);  permute_454 = None
        mul_528 = torch.ops.aten.mul.Tensor(add_214, primals_33);  primals_33 = None
        mul_529 = torch.ops.aten.mul.Tensor(add_214, mul_292);  add_214 = mul_292 = None
        sum_51 = torch.ops.aten.sum.dim_IntList(mul_529, [0, 1], True);  mul_529 = None
        view_383 = torch.ops.aten.view.default(sum_51, [512]);  sum_51 = None
        mul_530 = torch.ops.aten.mul.Tensor(mul_528, add_127)
        mul_531 = torch.ops.aten.mul.Tensor(mul_528, reciprocal_45);  mul_528 = reciprocal_45 = None
        sum_52 = torch.ops.aten.sum.dim_IntList(mul_530, [2], True);  mul_530 = None
        add_215 = torch.ops.aten.add.Tensor(add_209, mul_531);  add_209 = mul_531 = None
        alias_373 = torch.ops.aten.alias.default(alias_262);  alias_262 = None
        alias_374 = torch.ops.aten.alias.default(alias_373);  alias_373 = None
        pow_80 = torch.ops.aten.pow.Tensor_Scalar(alias_374, 3);  alias_374 = None
        mul_532 = torch.ops.aten.mul.Scalar(sum_52, -0.5);  sum_52 = None
        mul_533 = torch.ops.aten.mul.Tensor(mul_532, pow_80);  mul_532 = pow_80 = None
        expand_105 = torch.ops.aten.expand.default(mul_533, [2, 128, 512]);  mul_533 = None
        div_38 = torch.ops.aten.div.Scalar(expand_105, 512);  expand_105 = None
        pow_81 = torch.ops.aten.pow.Tensor_Scalar(add_127, 1.0);  add_127 = None
        mul_534 = torch.ops.aten.mul.Scalar(pow_81, 2.0);  pow_81 = None
        mul_535 = torch.ops.aten.mul.Tensor(div_38, mul_534);  div_38 = mul_534 = None
        add_216 = torch.ops.aten.add.Tensor(add_215, mul_535);  add_215 = mul_535 = None
        _to_copy_50 = torch.ops.aten._to_copy.default(gt_65, dtype = torch.float32);  gt_65 = None
        mul_536 = torch.ops.aten.mul.Tensor(_to_copy_50, 1.1111111111111112);  _to_copy_50 = None
        mul_537 = torch.ops.aten.mul.Tensor(add_216, mul_536);  mul_536 = None
        view_384 = torch.ops.aten.view.default(mul_537, [256, 512]);  mul_537 = None
        permute_457 = torch.ops.aten.permute.default(view_384, [1, 0])
        mm_213 = torch.ops.aten.mm.default(permute_457, view_202);  permute_457 = view_202 = None
        permute_458 = torch.ops.aten.permute.default(mm_213, [1, 0]);  mm_213 = None
        mm_214 = torch.ops.aten.mm.default(view_384, permute_459);  view_384 = permute_459 = None
        view_385 = torch.ops.aten.view.default(mm_214, [2, 128, 1024]);  mm_214 = None
        permute_460 = torch.ops.aten.permute.default(permute_458, [1, 0]);  permute_458 = None
        _to_copy_51 = torch.ops.aten._to_copy.default(gt_64, dtype = torch.float32);  gt_64 = None
        mul_538 = torch.ops.aten.mul.Tensor(_to_copy_51, 1.1111111111111112);  _to_copy_51 = None
        mul_539 = torch.ops.aten.mul.Tensor(view_385, mul_538);  view_385 = mul_538 = None
        mul_540 = torch.ops.aten.mul.Tensor(mul_539, mul_286);  mul_286 = None
        mul_541 = torch.ops.aten.mul.Tensor(mul_539, _unsafe_view_199);  mul_539 = _unsafe_view_199 = None
        view_386 = torch.ops.aten.view.default(mul_540, [256, 1024]);  mul_540 = None
        permute_461 = torch.ops.aten.permute.default(view_386, [1, 0])
        mm_215 = torch.ops.aten.mm.default(permute_461, view_200);  permute_461 = None
        permute_462 = torch.ops.aten.permute.default(mm_215, [1, 0]);  mm_215 = None
        mm_216 = torch.ops.aten.mm.default(view_386, permute_463);  view_386 = permute_463 = None
        view_387 = torch.ops.aten.view.default(mm_216, [2, 128, 512]);  mm_216 = None
        permute_464 = torch.ops.aten.permute.default(permute_462, [1, 0]);  permute_462 = None
        mul_542 = torch.ops.aten.mul.Tensor(mul_541, mul_281);  mul_281 = None
        mul_543 = torch.ops.aten.mul.Tensor(mul_541, add_126);  mul_541 = add_126 = None
        alias_375 = torch.ops.aten.alias.default(alias_257);  alias_257 = None
        alias_376 = torch.ops.aten.alias.default(alias_375);  alias_375 = None
        mul_544 = torch.ops.aten.mul.Tensor(alias_376, alias_376);  alias_376 = None
        _tensor_constant6 = self._tensor_constant6
        lift_fresh_copy_6 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant6);  _tensor_constant6 = None
        sub_57 = torch.ops.aten.sub.Tensor(lift_fresh_copy_6, mul_544);  lift_fresh_copy_6 = mul_544 = None
        mul_545 = torch.ops.aten.mul.Tensor(mul_542, sub_57);  mul_542 = sub_57 = None
        mul_546 = torch.ops.aten.mul.Tensor(mul_545, 0.7978845608028654);  mul_545 = None
        mul_547 = torch.ops.aten.mul.Tensor(mul_546, 0.044715)
        pow_82 = torch.ops.aten.pow.Tensor_Scalar(_unsafe_view_198, 2.0);  _unsafe_view_198 = None
        mul_548 = torch.ops.aten.mul.Scalar(pow_82, 3.0);  pow_82 = None
        mul_549 = torch.ops.aten.mul.Tensor(mul_547, mul_548);  mul_547 = mul_548 = None
        add_217 = torch.ops.aten.add.Tensor(mul_546, mul_549);  mul_546 = mul_549 = None
        mul_550 = torch.ops.aten.mul.Tensor(mul_543, 0.5);  mul_543 = None
        add_218 = torch.ops.aten.add.Tensor(add_217, mul_550);  add_217 = mul_550 = None
        view_388 = torch.ops.aten.view.default(add_218, [256, 1024]);  add_218 = None
        permute_465 = torch.ops.aten.permute.default(view_388, [1, 0])
        mm_217 = torch.ops.aten.mm.default(permute_465, view_200);  permute_465 = view_200 = None
        permute_466 = torch.ops.aten.permute.default(mm_217, [1, 0]);  mm_217 = None
        mm_218 = torch.ops.aten.mm.default(view_388, permute_467);  view_388 = permute_467 = None
        view_389 = torch.ops.aten.view.default(mm_218, [2, 128, 512]);  mm_218 = None
        add_219 = torch.ops.aten.add.Tensor(view_387, view_389);  view_387 = view_389 = None
        permute_468 = torch.ops.aten.permute.default(permute_466, [1, 0]);  permute_466 = None
        mul_551 = torch.ops.aten.mul.Tensor(add_219, primals_32);  primals_32 = None
        mul_552 = torch.ops.aten.mul.Tensor(add_219, mul_279);  add_219 = mul_279 = None
        sum_53 = torch.ops.aten.sum.dim_IntList(mul_552, [0, 1], True);  mul_552 = None
        view_390 = torch.ops.aten.view.default(sum_53, [512]);  sum_53 = None
        mul_553 = torch.ops.aten.mul.Tensor(mul_551, add_122)
        mul_554 = torch.ops.aten.mul.Tensor(mul_551, reciprocal_43);  mul_551 = reciprocal_43 = None
        sum_54 = torch.ops.aten.sum.dim_IntList(mul_553, [2], True);  mul_553 = None
        add_220 = torch.ops.aten.add.Tensor(add_216, mul_554);  add_216 = mul_554 = None
        alias_377 = torch.ops.aten.alias.default(alias_254);  alias_254 = None
        alias_378 = torch.ops.aten.alias.default(alias_377);  alias_377 = None
        pow_83 = torch.ops.aten.pow.Tensor_Scalar(alias_378, 3);  alias_378 = None
        mul_555 = torch.ops.aten.mul.Scalar(sum_54, -0.5);  sum_54 = None
        mul_556 = torch.ops.aten.mul.Tensor(mul_555, pow_83);  mul_555 = pow_83 = None
        expand_106 = torch.ops.aten.expand.default(mul_556, [2, 128, 512]);  mul_556 = None
        div_39 = torch.ops.aten.div.Scalar(expand_106, 512);  expand_106 = None
        pow_84 = torch.ops.aten.pow.Tensor_Scalar(add_122, 1.0);  add_122 = None
        mul_557 = torch.ops.aten.mul.Scalar(pow_84, 2.0);  pow_84 = None
        mul_558 = torch.ops.aten.mul.Tensor(div_39, mul_557);  div_39 = mul_557 = None
        add_221 = torch.ops.aten.add.Tensor(add_220, mul_558);  add_220 = mul_558 = None
        _to_copy_52 = torch.ops.aten._to_copy.default(gt_63, dtype = torch.float32);  gt_63 = None
        mul_559 = torch.ops.aten.mul.Tensor(_to_copy_52, 1.1111111111111112);  _to_copy_52 = None
        mul_560 = torch.ops.aten.mul.Tensor(add_221, mul_559);  mul_559 = None
        view_391 = torch.ops.aten.view.default(mul_560, [256, 512]);  mul_560 = None
        permute_469 = torch.ops.aten.permute.default(view_391, [1, 0])
        mm_219 = torch.ops.aten.mm.default(permute_469, view_199);  permute_469 = view_199 = None
        permute_470 = torch.ops.aten.permute.default(mm_219, [1, 0]);  mm_219 = None
        mm_220 = torch.ops.aten.mm.default(view_391, permute_471);  view_391 = permute_471 = None
        view_392 = torch.ops.aten.view.default(mm_220, [2, 128, 384]);  mm_220 = None
        permute_472 = torch.ops.aten.permute.default(permute_470, [1, 0]);  permute_470 = None
        view_393 = torch.ops.aten.view.default(view_392, [2, 128, 6, 64]);  view_392 = None
        permute_473 = torch.ops.aten.permute.default(view_393, [0, 2, 1, 3]);  view_393 = None
        clone_120 = torch.ops.aten.clone.default(permute_473, memory_format = torch.contiguous_format);  permute_473 = None
        _unsafe_view_289 = torch.ops.aten._unsafe_view.default(clone_120, [12, 128, 64]);  clone_120 = None
        bmm_72 = torch.ops.aten.bmm.default(permute_474, _unsafe_view_289);  permute_474 = None
        bmm_73 = torch.ops.aten.bmm.default(_unsafe_view_289, permute_475);  _unsafe_view_289 = permute_475 = None
        view_394 = torch.ops.aten.view.default(bmm_72, [2, 6, 128, 64]);  bmm_72 = None
        add_222 = torch.ops.aten.add.Tensor(tangents_22, view_394);  tangents_22 = view_394 = None
        view_395 = torch.ops.aten.view.default(bmm_73, [2, 6, 128, 128]);  bmm_73 = None
        philox_rand_like_30 = torch.ops.prims.philox_rand_like.default(view_395, philox_seed_like, 3342336)
        gt_91 = torch.ops.aten.gt.Scalar(philox_rand_like_30, 0.1);  philox_rand_like_30 = None
        _to_copy_53 = torch.ops.aten._to_copy.default(gt_91, dtype = torch.float32);  gt_91 = None
        mul_561 = torch.ops.aten.mul.Tensor(_to_copy_53, view_395);  _to_copy_53 = view_395 = None
        mul_562 = torch.ops.aten.mul.Tensor(mul_561, 1.1111111111111112);  mul_561 = None
        alias_379 = torch.ops.aten.alias.default(alias_250);  alias_250 = None
        alias_380 = torch.ops.aten.alias.default(alias_379);  alias_379 = None
        mul_563 = torch.ops.aten.mul.Tensor(mul_562, alias_380);  mul_562 = None
        sum_55 = torch.ops.aten.sum.dim_IntList(mul_563, [-1], True)
        mul_564 = torch.ops.aten.mul.Tensor(alias_380, sum_55);  alias_380 = sum_55 = None
        sub_58 = torch.ops.aten.sub.Tensor(mul_563, mul_564);  mul_563 = mul_564 = None
        view_396 = torch.ops.aten.view.default(sub_58, [12, 128, 128]);  sub_58 = None
        bmm_74 = torch.ops.aten.bmm.default(permute_476, view_396);  permute_476 = None
        bmm_75 = torch.ops.aten.bmm.default(view_396, permute_477);  view_396 = permute_477 = None
        view_397 = torch.ops.aten.view.default(bmm_74, [2, 6, 64, 128]);  bmm_74 = None
        view_398 = torch.ops.aten.view.default(bmm_75, [2, 6, 128, 64]);  bmm_75 = None
        permute_478 = torch.ops.aten.permute.default(view_397, [0, 1, 3, 2]);  view_397 = None
        add_223 = torch.ops.aten.add.Tensor(tangents_21, permute_478);  tangents_21 = permute_478 = None
        permute_479 = torch.ops.aten.permute.default(add_222, [0, 2, 1, 3]);  add_222 = None
        clone_121 = torch.ops.aten.clone.default(permute_479, memory_format = torch.contiguous_format);  permute_479 = None
        _unsafe_view_290 = torch.ops.aten._unsafe_view.default(clone_121, [2, 128, 384]);  clone_121 = None
        view_399 = torch.ops.aten.view.default(_unsafe_view_290, [256, 384]);  _unsafe_view_290 = None
        permute_480 = torch.ops.aten.permute.default(view_399, [1, 0])
        mm_221 = torch.ops.aten.mm.default(permute_480, view_109);  permute_480 = None
        permute_481 = torch.ops.aten.permute.default(mm_221, [1, 0]);  mm_221 = None
        mm_222 = torch.ops.aten.mm.default(view_399, permute_482);  view_399 = permute_482 = None
        view_400 = torch.ops.aten.view.default(mm_222, [2, 128, 512]);  mm_222 = None
        add_224 = torch.ops.aten.add.Tensor(add_207, view_400);  add_207 = view_400 = None
        permute_483 = torch.ops.aten.permute.default(permute_481, [1, 0]);  permute_481 = None
        permute_484 = torch.ops.aten.permute.default(add_223, [0, 2, 1, 3]);  add_223 = None
        clone_122 = torch.ops.aten.clone.default(permute_484, memory_format = torch.contiguous_format);  permute_484 = None
        _unsafe_view_291 = torch.ops.aten._unsafe_view.default(clone_122, [2, 128, 384]);  clone_122 = None
        view_401 = torch.ops.aten.view.default(_unsafe_view_291, [256, 384]);  _unsafe_view_291 = None
        permute_485 = torch.ops.aten.permute.default(view_401, [1, 0])
        mm_223 = torch.ops.aten.mm.default(permute_485, view_109);  permute_485 = None
        permute_486 = torch.ops.aten.permute.default(mm_223, [1, 0]);  mm_223 = None
        mm_224 = torch.ops.aten.mm.default(view_401, permute_487);  view_401 = permute_487 = None
        view_402 = torch.ops.aten.view.default(mm_224, [2, 128, 512]);  mm_224 = None
        add_225 = torch.ops.aten.add.Tensor(add_224, view_402);  add_224 = view_402 = None
        permute_488 = torch.ops.aten.permute.default(permute_486, [1, 0]);  permute_486 = None
        permute_489 = torch.ops.aten.permute.default(view_398, [0, 2, 1, 3]);  view_398 = None
        clone_123 = torch.ops.aten.clone.default(permute_489, memory_format = torch.contiguous_format);  permute_489 = None
        _unsafe_view_292 = torch.ops.aten._unsafe_view.default(clone_123, [2, 128, 384]);  clone_123 = None
        view_403 = torch.ops.aten.view.default(_unsafe_view_292, [256, 384]);  _unsafe_view_292 = None
        permute_490 = torch.ops.aten.permute.default(view_403, [1, 0])
        mm_225 = torch.ops.aten.mm.default(permute_490, view_191);  permute_490 = view_191 = None
        permute_491 = torch.ops.aten.permute.default(mm_225, [1, 0]);  mm_225 = None
        mm_226 = torch.ops.aten.mm.default(view_403, permute_492);  view_403 = permute_492 = None
        view_404 = torch.ops.aten.view.default(mm_226, [2, 128, 512]);  mm_226 = None
        permute_493 = torch.ops.aten.permute.default(permute_491, [1, 0]);  permute_491 = None
        mul_565 = torch.ops.aten.mul.Tensor(view_404, primals_31);  primals_31 = None
        mul_566 = torch.ops.aten.mul.Tensor(view_404, mul_273);  view_404 = mul_273 = None
        sum_56 = torch.ops.aten.sum.dim_IntList(mul_566, [0, 1], True);  mul_566 = None
        view_405 = torch.ops.aten.view.default(sum_56, [512]);  sum_56 = None
        mul_567 = torch.ops.aten.mul.Tensor(mul_565, add_119)
        mul_568 = torch.ops.aten.mul.Tensor(mul_565, reciprocal_42);  mul_565 = reciprocal_42 = None
        sum_57 = torch.ops.aten.sum.dim_IntList(mul_567, [2], True);  mul_567 = None
        add_226 = torch.ops.aten.add.Tensor(add_221, mul_568);  add_221 = mul_568 = None
        alias_381 = torch.ops.aten.alias.default(alias_247);  alias_247 = None
        alias_382 = torch.ops.aten.alias.default(alias_381);  alias_381 = None
        pow_85 = torch.ops.aten.pow.Tensor_Scalar(alias_382, 3);  alias_382 = None
        mul_569 = torch.ops.aten.mul.Scalar(sum_57, -0.5);  sum_57 = None
        mul_570 = torch.ops.aten.mul.Tensor(mul_569, pow_85);  mul_569 = pow_85 = None
        expand_107 = torch.ops.aten.expand.default(mul_570, [2, 128, 512]);  mul_570 = None
        div_40 = torch.ops.aten.div.Scalar(expand_107, 512);  expand_107 = None
        pow_86 = torch.ops.aten.pow.Tensor_Scalar(add_119, 1.0);  add_119 = None
        mul_571 = torch.ops.aten.mul.Scalar(pow_86, 2.0);  pow_86 = None
        mul_572 = torch.ops.aten.mul.Tensor(div_40, mul_571);  div_40 = mul_571 = None
        add_227 = torch.ops.aten.add.Tensor(add_226, mul_572);  add_226 = mul_572 = None
        _to_copy_54 = torch.ops.aten._to_copy.default(gt_61, dtype = torch.float32);  gt_61 = None
        mul_573 = torch.ops.aten.mul.Tensor(_to_copy_54, 1.1111111111111112);  _to_copy_54 = None
        mul_574 = torch.ops.aten.mul.Tensor(add_227, mul_573);  mul_573 = None
        view_406 = torch.ops.aten.view.default(mul_574, [256, 512]);  mul_574 = None
        permute_494 = torch.ops.aten.permute.default(view_406, [1, 0])
        mm_227 = torch.ops.aten.mm.default(permute_494, view_190);  permute_494 = view_190 = None
        permute_495 = torch.ops.aten.permute.default(mm_227, [1, 0]);  mm_227 = None
        mm_228 = torch.ops.aten.mm.default(view_406, permute_496);  view_406 = permute_496 = None
        view_407 = torch.ops.aten.view.default(mm_228, [2, 128, 384]);  mm_228 = None
        permute_497 = torch.ops.aten.permute.default(permute_495, [1, 0]);  permute_495 = None
        view_408 = torch.ops.aten.view.default(view_407, [2, 128, 6, 64]);  view_407 = None
        permute_498 = torch.ops.aten.permute.default(view_408, [0, 2, 1, 3]);  view_408 = None
        clone_124 = torch.ops.aten.clone.default(permute_498, memory_format = torch.contiguous_format);  permute_498 = None
        _unsafe_view_293 = torch.ops.aten._unsafe_view.default(clone_124, [12, 128, 64]);  clone_124 = None
        bmm_76 = torch.ops.aten.bmm.default(permute_499, _unsafe_view_293);  permute_499 = None
        bmm_77 = torch.ops.aten.bmm.default(_unsafe_view_293, permute_500);  _unsafe_view_293 = permute_500 = None
        view_409 = torch.ops.aten.view.default(bmm_76, [2, 6, 128, 64]);  bmm_76 = None
        add_228 = torch.ops.aten.add.Tensor(tangents_20, view_409);  tangents_20 = view_409 = None
        view_410 = torch.ops.aten.view.default(bmm_77, [2, 6, 128, 128]);  bmm_77 = None
        philox_rand_like_31 = torch.ops.prims.philox_rand_like.default(view_410, philox_seed_like, 3145728)
        gt_92 = torch.ops.aten.gt.Scalar(philox_rand_like_31, 0.1);  philox_rand_like_31 = None
        _to_copy_55 = torch.ops.aten._to_copy.default(gt_92, dtype = torch.float32);  gt_92 = None
        mul_575 = torch.ops.aten.mul.Tensor(_to_copy_55, view_410);  _to_copy_55 = view_410 = None
        mul_576 = torch.ops.aten.mul.Tensor(mul_575, 1.1111111111111112);  mul_575 = None
        alias_383 = torch.ops.aten.alias.default(alias_243);  alias_243 = None
        alias_384 = torch.ops.aten.alias.default(alias_383);  alias_383 = None
        mul_577 = torch.ops.aten.mul.Tensor(mul_576, alias_384);  mul_576 = None
        sum_58 = torch.ops.aten.sum.dim_IntList(mul_577, [-1], True)
        mul_578 = torch.ops.aten.mul.Tensor(alias_384, sum_58);  alias_384 = sum_58 = None
        sub_59 = torch.ops.aten.sub.Tensor(mul_577, mul_578);  mul_577 = mul_578 = None
        add_229 = torch.ops.aten.add.Tensor(add_211, sub_59);  add_211 = None
        view_411 = torch.ops.aten.view.default(sub_59, [12, 128, 128]);  sub_59 = None
        bmm_78 = torch.ops.aten.bmm.default(permute_501, view_411);  permute_501 = None
        bmm_79 = torch.ops.aten.bmm.default(view_411, permute_502);  view_411 = permute_502 = None
        view_412 = torch.ops.aten.view.default(bmm_78, [2, 6, 64, 128]);  bmm_78 = None
        view_413 = torch.ops.aten.view.default(bmm_79, [2, 6, 128, 64]);  bmm_79 = None
        permute_503 = torch.ops.aten.permute.default(view_412, [0, 1, 3, 2]);  view_412 = None
        add_230 = torch.ops.aten.add.Tensor(tangents_19, permute_503);  tangents_19 = permute_503 = None
        permute_504 = torch.ops.aten.permute.default(add_228, [0, 2, 1, 3]);  add_228 = None
        clone_125 = torch.ops.aten.clone.default(permute_504, memory_format = torch.contiguous_format);  permute_504 = None
        _unsafe_view_294 = torch.ops.aten._unsafe_view.default(clone_125, [2, 128, 384]);  clone_125 = None
        view_414 = torch.ops.aten.view.default(_unsafe_view_294, [256, 384]);  _unsafe_view_294 = None
        permute_505 = torch.ops.aten.permute.default(view_414, [1, 0])
        mm_229 = torch.ops.aten.mm.default(permute_505, view_182);  permute_505 = None
        permute_506 = torch.ops.aten.permute.default(mm_229, [1, 0]);  mm_229 = None
        mm_230 = torch.ops.aten.mm.default(view_414, permute_507);  view_414 = permute_507 = None
        view_415 = torch.ops.aten.view.default(mm_230, [2, 128, 512]);  mm_230 = None
        permute_508 = torch.ops.aten.permute.default(permute_506, [1, 0]);  permute_506 = None
        permute_509 = torch.ops.aten.permute.default(add_230, [0, 2, 1, 3]);  add_230 = None
        clone_126 = torch.ops.aten.clone.default(permute_509, memory_format = torch.contiguous_format);  permute_509 = None
        _unsafe_view_295 = torch.ops.aten._unsafe_view.default(clone_126, [2, 128, 384]);  clone_126 = None
        view_416 = torch.ops.aten.view.default(_unsafe_view_295, [256, 384]);  _unsafe_view_295 = None
        permute_510 = torch.ops.aten.permute.default(view_416, [1, 0])
        mm_231 = torch.ops.aten.mm.default(permute_510, view_182);  permute_510 = None
        permute_511 = torch.ops.aten.permute.default(mm_231, [1, 0]);  mm_231 = None
        mm_232 = torch.ops.aten.mm.default(view_416, permute_512);  view_416 = permute_512 = None
        view_417 = torch.ops.aten.view.default(mm_232, [2, 128, 512]);  mm_232 = None
        add_231 = torch.ops.aten.add.Tensor(view_415, view_417);  view_415 = view_417 = None
        permute_513 = torch.ops.aten.permute.default(permute_511, [1, 0]);  permute_511 = None
        permute_514 = torch.ops.aten.permute.default(view_413, [0, 2, 1, 3]);  view_413 = None
        clone_127 = torch.ops.aten.clone.default(permute_514, memory_format = torch.contiguous_format);  permute_514 = None
        _unsafe_view_296 = torch.ops.aten._unsafe_view.default(clone_127, [2, 128, 384]);  clone_127 = None
        view_418 = torch.ops.aten.view.default(_unsafe_view_296, [256, 384]);  _unsafe_view_296 = None
        permute_515 = torch.ops.aten.permute.default(view_418, [1, 0])
        mm_233 = torch.ops.aten.mm.default(permute_515, view_182);  permute_515 = view_182 = None
        permute_516 = torch.ops.aten.permute.default(mm_233, [1, 0]);  mm_233 = None
        mm_234 = torch.ops.aten.mm.default(view_418, permute_517);  view_418 = permute_517 = None
        view_419 = torch.ops.aten.view.default(mm_234, [2, 128, 512]);  mm_234 = None
        add_232 = torch.ops.aten.add.Tensor(add_231, view_419);  add_231 = view_419 = None
        permute_518 = torch.ops.aten.permute.default(permute_516, [1, 0]);  permute_516 = None
        mul_579 = torch.ops.aten.mul.Tensor(add_232, primals_30);  primals_30 = None
        mul_580 = torch.ops.aten.mul.Tensor(add_232, mul_267);  add_232 = mul_267 = None
        sum_59 = torch.ops.aten.sum.dim_IntList(mul_580, [0, 1], True);  mul_580 = None
        view_420 = torch.ops.aten.view.default(sum_59, [512]);  sum_59 = None
        mul_581 = torch.ops.aten.mul.Tensor(mul_579, add_116)
        mul_582 = torch.ops.aten.mul.Tensor(mul_579, reciprocal_41);  mul_579 = reciprocal_41 = None
        sum_60 = torch.ops.aten.sum.dim_IntList(mul_581, [2], True);  mul_581 = None
        add_233 = torch.ops.aten.add.Tensor(add_227, mul_582);  add_227 = mul_582 = None
        alias_385 = torch.ops.aten.alias.default(alias_240);  alias_240 = None
        alias_386 = torch.ops.aten.alias.default(alias_385);  alias_385 = None
        pow_87 = torch.ops.aten.pow.Tensor_Scalar(alias_386, 3);  alias_386 = None
        mul_583 = torch.ops.aten.mul.Scalar(sum_60, -0.5);  sum_60 = None
        mul_584 = torch.ops.aten.mul.Tensor(mul_583, pow_87);  mul_583 = pow_87 = None
        expand_108 = torch.ops.aten.expand.default(mul_584, [2, 128, 512]);  mul_584 = None
        div_41 = torch.ops.aten.div.Scalar(expand_108, 512);  expand_108 = None
        pow_88 = torch.ops.aten.pow.Tensor_Scalar(add_116, 1.0);  add_116 = None
        mul_585 = torch.ops.aten.mul.Scalar(pow_88, 2.0);  pow_88 = None
        mul_586 = torch.ops.aten.mul.Tensor(div_41, mul_585);  div_41 = mul_585 = None
        add_234 = torch.ops.aten.add.Tensor(add_233, mul_586);  add_233 = mul_586 = None
        _to_copy_56 = torch.ops.aten._to_copy.default(gt_59, dtype = torch.float32);  gt_59 = None
        mul_587 = torch.ops.aten.mul.Tensor(_to_copy_56, 1.1111111111111112);  _to_copy_56 = None
        mul_588 = torch.ops.aten.mul.Tensor(add_234, mul_587);  mul_587 = None
        view_421 = torch.ops.aten.view.default(mul_588, [256, 512]);  mul_588 = None
        permute_519 = torch.ops.aten.permute.default(view_421, [1, 0])
        mm_235 = torch.ops.aten.mm.default(permute_519, view_181);  permute_519 = view_181 = None
        permute_520 = torch.ops.aten.permute.default(mm_235, [1, 0]);  mm_235 = None
        mm_236 = torch.ops.aten.mm.default(view_421, permute_521);  view_421 = permute_521 = None
        view_422 = torch.ops.aten.view.default(mm_236, [2, 128, 1024]);  mm_236 = None
        permute_522 = torch.ops.aten.permute.default(permute_520, [1, 0]);  permute_520 = None
        _to_copy_57 = torch.ops.aten._to_copy.default(gt_58, dtype = torch.float32);  gt_58 = None
        mul_589 = torch.ops.aten.mul.Tensor(_to_copy_57, 1.1111111111111112);  _to_copy_57 = None
        mul_590 = torch.ops.aten.mul.Tensor(view_422, mul_589);  view_422 = mul_589 = None
        mul_591 = torch.ops.aten.mul.Tensor(mul_590, mul_261);  mul_261 = None
        mul_592 = torch.ops.aten.mul.Tensor(mul_590, _unsafe_view_178);  mul_590 = _unsafe_view_178 = None
        view_423 = torch.ops.aten.view.default(mul_591, [256, 1024]);  mul_591 = None
        permute_523 = torch.ops.aten.permute.default(view_423, [1, 0])
        mm_237 = torch.ops.aten.mm.default(permute_523, view_179);  permute_523 = None
        permute_524 = torch.ops.aten.permute.default(mm_237, [1, 0]);  mm_237 = None
        mm_238 = torch.ops.aten.mm.default(view_423, permute_525);  view_423 = permute_525 = None
        view_424 = torch.ops.aten.view.default(mm_238, [2, 128, 512]);  mm_238 = None
        permute_526 = torch.ops.aten.permute.default(permute_524, [1, 0]);  permute_524 = None
        mul_593 = torch.ops.aten.mul.Tensor(mul_592, mul_256);  mul_256 = None
        mul_594 = torch.ops.aten.mul.Tensor(mul_592, add_115);  mul_592 = add_115 = None
        alias_387 = torch.ops.aten.alias.default(alias_235);  alias_235 = None
        alias_388 = torch.ops.aten.alias.default(alias_387);  alias_387 = None
        mul_595 = torch.ops.aten.mul.Tensor(alias_388, alias_388);  alias_388 = None
        _tensor_constant7 = self._tensor_constant7
        lift_fresh_copy_7 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant7);  _tensor_constant7 = None
        sub_60 = torch.ops.aten.sub.Tensor(lift_fresh_copy_7, mul_595);  lift_fresh_copy_7 = mul_595 = None
        mul_596 = torch.ops.aten.mul.Tensor(mul_593, sub_60);  mul_593 = sub_60 = None
        mul_597 = torch.ops.aten.mul.Tensor(mul_596, 0.7978845608028654);  mul_596 = None
        mul_598 = torch.ops.aten.mul.Tensor(mul_597, 0.044715)
        pow_89 = torch.ops.aten.pow.Tensor_Scalar(_unsafe_view_177, 2.0);  _unsafe_view_177 = None
        mul_599 = torch.ops.aten.mul.Scalar(pow_89, 3.0);  pow_89 = None
        mul_600 = torch.ops.aten.mul.Tensor(mul_598, mul_599);  mul_598 = mul_599 = None
        add_235 = torch.ops.aten.add.Tensor(mul_597, mul_600);  mul_597 = mul_600 = None
        mul_601 = torch.ops.aten.mul.Tensor(mul_594, 0.5);  mul_594 = None
        add_236 = torch.ops.aten.add.Tensor(add_235, mul_601);  add_235 = mul_601 = None
        view_425 = torch.ops.aten.view.default(add_236, [256, 1024]);  add_236 = None
        permute_527 = torch.ops.aten.permute.default(view_425, [1, 0])
        mm_239 = torch.ops.aten.mm.default(permute_527, view_179);  permute_527 = view_179 = None
        permute_528 = torch.ops.aten.permute.default(mm_239, [1, 0]);  mm_239 = None
        mm_240 = torch.ops.aten.mm.default(view_425, permute_529);  view_425 = permute_529 = None
        view_426 = torch.ops.aten.view.default(mm_240, [2, 128, 512]);  mm_240 = None
        add_237 = torch.ops.aten.add.Tensor(view_424, view_426);  view_424 = view_426 = None
        permute_530 = torch.ops.aten.permute.default(permute_528, [1, 0]);  permute_528 = None
        mul_602 = torch.ops.aten.mul.Tensor(add_237, primals_29);  primals_29 = None
        mul_603 = torch.ops.aten.mul.Tensor(add_237, mul_254);  add_237 = mul_254 = None
        sum_61 = torch.ops.aten.sum.dim_IntList(mul_603, [0, 1], True);  mul_603 = None
        view_427 = torch.ops.aten.view.default(sum_61, [512]);  sum_61 = None
        mul_604 = torch.ops.aten.mul.Tensor(mul_602, add_111)
        mul_605 = torch.ops.aten.mul.Tensor(mul_602, reciprocal_39);  mul_602 = reciprocal_39 = None
        sum_62 = torch.ops.aten.sum.dim_IntList(mul_604, [2], True);  mul_604 = None
        add_238 = torch.ops.aten.add.Tensor(add_234, mul_605);  add_234 = mul_605 = None
        alias_389 = torch.ops.aten.alias.default(alias_232);  alias_232 = None
        alias_390 = torch.ops.aten.alias.default(alias_389);  alias_389 = None
        pow_90 = torch.ops.aten.pow.Tensor_Scalar(alias_390, 3);  alias_390 = None
        mul_606 = torch.ops.aten.mul.Scalar(sum_62, -0.5);  sum_62 = None
        mul_607 = torch.ops.aten.mul.Tensor(mul_606, pow_90);  mul_606 = pow_90 = None
        expand_109 = torch.ops.aten.expand.default(mul_607, [2, 128, 512]);  mul_607 = None
        div_42 = torch.ops.aten.div.Scalar(expand_109, 512);  expand_109 = None
        pow_91 = torch.ops.aten.pow.Tensor_Scalar(add_111, 1.0);  add_111 = None
        mul_608 = torch.ops.aten.mul.Scalar(pow_91, 2.0);  pow_91 = None
        mul_609 = torch.ops.aten.mul.Tensor(div_42, mul_608);  div_42 = mul_608 = None
        add_239 = torch.ops.aten.add.Tensor(add_238, mul_609);  add_238 = mul_609 = None
        _to_copy_58 = torch.ops.aten._to_copy.default(gt_57, dtype = torch.float32);  gt_57 = None
        mul_610 = torch.ops.aten.mul.Tensor(_to_copy_58, 1.1111111111111112);  _to_copy_58 = None
        mul_611 = torch.ops.aten.mul.Tensor(add_239, mul_610);  mul_610 = None
        view_428 = torch.ops.aten.view.default(mul_611, [256, 512]);  mul_611 = None
        permute_531 = torch.ops.aten.permute.default(view_428, [1, 0])
        mm_241 = torch.ops.aten.mm.default(permute_531, view_178);  permute_531 = view_178 = None
        permute_532 = torch.ops.aten.permute.default(mm_241, [1, 0]);  mm_241 = None
        mm_242 = torch.ops.aten.mm.default(view_428, permute_533);  view_428 = permute_533 = None
        view_429 = torch.ops.aten.view.default(mm_242, [2, 128, 384]);  mm_242 = None
        permute_534 = torch.ops.aten.permute.default(permute_532, [1, 0]);  permute_532 = None
        view_430 = torch.ops.aten.view.default(view_429, [2, 128, 6, 64]);  view_429 = None
        permute_535 = torch.ops.aten.permute.default(view_430, [0, 2, 1, 3]);  view_430 = None
        clone_128 = torch.ops.aten.clone.default(permute_535, memory_format = torch.contiguous_format);  permute_535 = None
        _unsafe_view_297 = torch.ops.aten._unsafe_view.default(clone_128, [12, 128, 64]);  clone_128 = None
        bmm_80 = torch.ops.aten.bmm.default(permute_536, _unsafe_view_297);  permute_536 = None
        bmm_81 = torch.ops.aten.bmm.default(_unsafe_view_297, permute_537);  _unsafe_view_297 = permute_537 = None
        view_431 = torch.ops.aten.view.default(bmm_80, [2, 6, 128, 64]);  bmm_80 = None
        add_240 = torch.ops.aten.add.Tensor(tangents_18, view_431);  tangents_18 = view_431 = None
        view_432 = torch.ops.aten.view.default(bmm_81, [2, 6, 128, 128]);  bmm_81 = None
        philox_rand_like_32 = torch.ops.prims.philox_rand_like.default(view_432, philox_seed_like, 2949120)
        gt_93 = torch.ops.aten.gt.Scalar(philox_rand_like_32, 0.1);  philox_rand_like_32 = None
        _to_copy_59 = torch.ops.aten._to_copy.default(gt_93, dtype = torch.float32);  gt_93 = None
        mul_612 = torch.ops.aten.mul.Tensor(_to_copy_59, view_432);  _to_copy_59 = view_432 = None
        mul_613 = torch.ops.aten.mul.Tensor(mul_612, 1.1111111111111112);  mul_612 = None
        alias_391 = torch.ops.aten.alias.default(alias_228);  alias_228 = None
        alias_392 = torch.ops.aten.alias.default(alias_391);  alias_391 = None
        mul_614 = torch.ops.aten.mul.Tensor(mul_613, alias_392);  mul_613 = None
        sum_63 = torch.ops.aten.sum.dim_IntList(mul_614, [-1], True)
        mul_615 = torch.ops.aten.mul.Tensor(alias_392, sum_63);  alias_392 = sum_63 = None
        sub_61 = torch.ops.aten.sub.Tensor(mul_614, mul_615);  mul_614 = mul_615 = None
        view_433 = torch.ops.aten.view.default(sub_61, [12, 128, 128]);  sub_61 = None
        bmm_82 = torch.ops.aten.bmm.default(permute_538, view_433);  permute_538 = None
        bmm_83 = torch.ops.aten.bmm.default(view_433, permute_539);  view_433 = permute_539 = None
        view_434 = torch.ops.aten.view.default(bmm_82, [2, 6, 64, 128]);  bmm_82 = None
        view_435 = torch.ops.aten.view.default(bmm_83, [2, 6, 128, 64]);  bmm_83 = None
        permute_540 = torch.ops.aten.permute.default(view_434, [0, 1, 3, 2]);  view_434 = None
        add_241 = torch.ops.aten.add.Tensor(tangents_17, permute_540);  tangents_17 = permute_540 = None
        permute_541 = torch.ops.aten.permute.default(add_240, [0, 2, 1, 3]);  add_240 = None
        clone_129 = torch.ops.aten.clone.default(permute_541, memory_format = torch.contiguous_format);  permute_541 = None
        _unsafe_view_298 = torch.ops.aten._unsafe_view.default(clone_129, [2, 128, 384]);  clone_129 = None
        view_436 = torch.ops.aten.view.default(_unsafe_view_298, [256, 384]);  _unsafe_view_298 = None
        permute_542 = torch.ops.aten.permute.default(view_436, [1, 0])
        mm_243 = torch.ops.aten.mm.default(permute_542, view_109);  permute_542 = None
        permute_543 = torch.ops.aten.permute.default(mm_243, [1, 0]);  mm_243 = None
        mm_244 = torch.ops.aten.mm.default(view_436, permute_544);  view_436 = permute_544 = None
        view_437 = torch.ops.aten.view.default(mm_244, [2, 128, 512]);  mm_244 = None
        add_242 = torch.ops.aten.add.Tensor(add_225, view_437);  add_225 = view_437 = None
        permute_545 = torch.ops.aten.permute.default(permute_543, [1, 0]);  permute_543 = None
        permute_546 = torch.ops.aten.permute.default(add_241, [0, 2, 1, 3]);  add_241 = None
        clone_130 = torch.ops.aten.clone.default(permute_546, memory_format = torch.contiguous_format);  permute_546 = None
        _unsafe_view_299 = torch.ops.aten._unsafe_view.default(clone_130, [2, 128, 384]);  clone_130 = None
        view_438 = torch.ops.aten.view.default(_unsafe_view_299, [256, 384]);  _unsafe_view_299 = None
        permute_547 = torch.ops.aten.permute.default(view_438, [1, 0])
        mm_245 = torch.ops.aten.mm.default(permute_547, view_109);  permute_547 = None
        permute_548 = torch.ops.aten.permute.default(mm_245, [1, 0]);  mm_245 = None
        mm_246 = torch.ops.aten.mm.default(view_438, permute_549);  view_438 = permute_549 = None
        view_439 = torch.ops.aten.view.default(mm_246, [2, 128, 512]);  mm_246 = None
        add_243 = torch.ops.aten.add.Tensor(add_242, view_439);  add_242 = view_439 = None
        permute_550 = torch.ops.aten.permute.default(permute_548, [1, 0]);  permute_548 = None
        permute_551 = torch.ops.aten.permute.default(view_435, [0, 2, 1, 3]);  view_435 = None
        clone_131 = torch.ops.aten.clone.default(permute_551, memory_format = torch.contiguous_format);  permute_551 = None
        _unsafe_view_300 = torch.ops.aten._unsafe_view.default(clone_131, [2, 128, 384]);  clone_131 = None
        view_440 = torch.ops.aten.view.default(_unsafe_view_300, [256, 384]);  _unsafe_view_300 = None
        permute_552 = torch.ops.aten.permute.default(view_440, [1, 0])
        mm_247 = torch.ops.aten.mm.default(permute_552, view_170);  permute_552 = view_170 = None
        permute_553 = torch.ops.aten.permute.default(mm_247, [1, 0]);  mm_247 = None
        mm_248 = torch.ops.aten.mm.default(view_440, permute_554);  view_440 = permute_554 = None
        view_441 = torch.ops.aten.view.default(mm_248, [2, 128, 512]);  mm_248 = None
        permute_555 = torch.ops.aten.permute.default(permute_553, [1, 0]);  permute_553 = None
        mul_616 = torch.ops.aten.mul.Tensor(view_441, primals_28);  primals_28 = None
        mul_617 = torch.ops.aten.mul.Tensor(view_441, mul_248);  view_441 = mul_248 = None
        sum_64 = torch.ops.aten.sum.dim_IntList(mul_617, [0, 1], True);  mul_617 = None
        view_442 = torch.ops.aten.view.default(sum_64, [512]);  sum_64 = None
        mul_618 = torch.ops.aten.mul.Tensor(mul_616, add_108)
        mul_619 = torch.ops.aten.mul.Tensor(mul_616, reciprocal_38);  mul_616 = reciprocal_38 = None
        sum_65 = torch.ops.aten.sum.dim_IntList(mul_618, [2], True);  mul_618 = None
        add_244 = torch.ops.aten.add.Tensor(add_239, mul_619);  add_239 = mul_619 = None
        alias_393 = torch.ops.aten.alias.default(alias_225);  alias_225 = None
        alias_394 = torch.ops.aten.alias.default(alias_393);  alias_393 = None
        pow_92 = torch.ops.aten.pow.Tensor_Scalar(alias_394, 3);  alias_394 = None
        mul_620 = torch.ops.aten.mul.Scalar(sum_65, -0.5);  sum_65 = None
        mul_621 = torch.ops.aten.mul.Tensor(mul_620, pow_92);  mul_620 = pow_92 = None
        expand_110 = torch.ops.aten.expand.default(mul_621, [2, 128, 512]);  mul_621 = None
        div_43 = torch.ops.aten.div.Scalar(expand_110, 512);  expand_110 = None
        pow_93 = torch.ops.aten.pow.Tensor_Scalar(add_108, 1.0);  add_108 = None
        mul_622 = torch.ops.aten.mul.Scalar(pow_93, 2.0);  pow_93 = None
        mul_623 = torch.ops.aten.mul.Tensor(div_43, mul_622);  div_43 = mul_622 = None
        add_245 = torch.ops.aten.add.Tensor(add_244, mul_623);  add_244 = mul_623 = None
        _to_copy_60 = torch.ops.aten._to_copy.default(gt_55, dtype = torch.float32);  gt_55 = None
        mul_624 = torch.ops.aten.mul.Tensor(_to_copy_60, 1.1111111111111112);  _to_copy_60 = None
        mul_625 = torch.ops.aten.mul.Tensor(add_245, mul_624);  mul_624 = None
        view_443 = torch.ops.aten.view.default(mul_625, [256, 512]);  mul_625 = None
        permute_556 = torch.ops.aten.permute.default(view_443, [1, 0])
        mm_249 = torch.ops.aten.mm.default(permute_556, view_169);  permute_556 = view_169 = None
        permute_557 = torch.ops.aten.permute.default(mm_249, [1, 0]);  mm_249 = None
        mm_250 = torch.ops.aten.mm.default(view_443, permute_558);  view_443 = permute_558 = None
        view_444 = torch.ops.aten.view.default(mm_250, [2, 128, 384]);  mm_250 = None
        permute_559 = torch.ops.aten.permute.default(permute_557, [1, 0]);  permute_557 = None
        view_445 = torch.ops.aten.view.default(view_444, [2, 128, 6, 64]);  view_444 = None
        permute_560 = torch.ops.aten.permute.default(view_445, [0, 2, 1, 3]);  view_445 = None
        clone_132 = torch.ops.aten.clone.default(permute_560, memory_format = torch.contiguous_format);  permute_560 = None
        _unsafe_view_301 = torch.ops.aten._unsafe_view.default(clone_132, [12, 128, 64]);  clone_132 = None
        bmm_84 = torch.ops.aten.bmm.default(permute_561, _unsafe_view_301);  permute_561 = None
        bmm_85 = torch.ops.aten.bmm.default(_unsafe_view_301, permute_562);  _unsafe_view_301 = permute_562 = None
        view_446 = torch.ops.aten.view.default(bmm_84, [2, 6, 128, 64]);  bmm_84 = None
        add_246 = torch.ops.aten.add.Tensor(tangents_16, view_446);  tangents_16 = view_446 = None
        view_447 = torch.ops.aten.view.default(bmm_85, [2, 6, 128, 128]);  bmm_85 = None
        philox_rand_like_33 = torch.ops.prims.philox_rand_like.default(view_447, philox_seed_like, 2752512)
        gt_94 = torch.ops.aten.gt.Scalar(philox_rand_like_33, 0.1);  philox_rand_like_33 = None
        _to_copy_61 = torch.ops.aten._to_copy.default(gt_94, dtype = torch.float32);  gt_94 = None
        mul_626 = torch.ops.aten.mul.Tensor(_to_copy_61, view_447);  _to_copy_61 = view_447 = None
        mul_627 = torch.ops.aten.mul.Tensor(mul_626, 1.1111111111111112);  mul_626 = None
        alias_395 = torch.ops.aten.alias.default(alias_221);  alias_221 = None
        alias_396 = torch.ops.aten.alias.default(alias_395);  alias_395 = None
        mul_628 = torch.ops.aten.mul.Tensor(mul_627, alias_396);  mul_627 = None
        sum_66 = torch.ops.aten.sum.dim_IntList(mul_628, [-1], True)
        mul_629 = torch.ops.aten.mul.Tensor(alias_396, sum_66);  alias_396 = sum_66 = None
        sub_62 = torch.ops.aten.sub.Tensor(mul_628, mul_629);  mul_628 = mul_629 = None
        add_247 = torch.ops.aten.add.Tensor(add_229, sub_62);  add_229 = None
        view_448 = torch.ops.aten.view.default(sub_62, [12, 128, 128]);  sub_62 = None
        bmm_86 = torch.ops.aten.bmm.default(permute_563, view_448);  permute_563 = None
        bmm_87 = torch.ops.aten.bmm.default(view_448, permute_564);  view_448 = permute_564 = None
        view_449 = torch.ops.aten.view.default(bmm_86, [2, 6, 64, 128]);  bmm_86 = None
        view_450 = torch.ops.aten.view.default(bmm_87, [2, 6, 128, 64]);  bmm_87 = None
        permute_565 = torch.ops.aten.permute.default(view_449, [0, 1, 3, 2]);  view_449 = None
        add_248 = torch.ops.aten.add.Tensor(tangents_15, permute_565);  tangents_15 = permute_565 = None
        permute_566 = torch.ops.aten.permute.default(add_246, [0, 2, 1, 3]);  add_246 = None
        clone_133 = torch.ops.aten.clone.default(permute_566, memory_format = torch.contiguous_format);  permute_566 = None
        _unsafe_view_302 = torch.ops.aten._unsafe_view.default(clone_133, [2, 128, 384]);  clone_133 = None
        view_451 = torch.ops.aten.view.default(_unsafe_view_302, [256, 384]);  _unsafe_view_302 = None
        permute_567 = torch.ops.aten.permute.default(view_451, [1, 0])
        mm_251 = torch.ops.aten.mm.default(permute_567, view_161);  permute_567 = None
        permute_568 = torch.ops.aten.permute.default(mm_251, [1, 0]);  mm_251 = None
        mm_252 = torch.ops.aten.mm.default(view_451, permute_569);  view_451 = permute_569 = None
        view_452 = torch.ops.aten.view.default(mm_252, [2, 128, 512]);  mm_252 = None
        permute_570 = torch.ops.aten.permute.default(permute_568, [1, 0]);  permute_568 = None
        permute_571 = torch.ops.aten.permute.default(add_248, [0, 2, 1, 3]);  add_248 = None
        clone_134 = torch.ops.aten.clone.default(permute_571, memory_format = torch.contiguous_format);  permute_571 = None
        _unsafe_view_303 = torch.ops.aten._unsafe_view.default(clone_134, [2, 128, 384]);  clone_134 = None
        view_453 = torch.ops.aten.view.default(_unsafe_view_303, [256, 384]);  _unsafe_view_303 = None
        permute_572 = torch.ops.aten.permute.default(view_453, [1, 0])
        mm_253 = torch.ops.aten.mm.default(permute_572, view_161);  permute_572 = None
        permute_573 = torch.ops.aten.permute.default(mm_253, [1, 0]);  mm_253 = None
        mm_254 = torch.ops.aten.mm.default(view_453, permute_574);  view_453 = permute_574 = None
        view_454 = torch.ops.aten.view.default(mm_254, [2, 128, 512]);  mm_254 = None
        add_249 = torch.ops.aten.add.Tensor(view_452, view_454);  view_452 = view_454 = None
        permute_575 = torch.ops.aten.permute.default(permute_573, [1, 0]);  permute_573 = None
        permute_576 = torch.ops.aten.permute.default(view_450, [0, 2, 1, 3]);  view_450 = None
        clone_135 = torch.ops.aten.clone.default(permute_576, memory_format = torch.contiguous_format);  permute_576 = None
        _unsafe_view_304 = torch.ops.aten._unsafe_view.default(clone_135, [2, 128, 384]);  clone_135 = None
        view_455 = torch.ops.aten.view.default(_unsafe_view_304, [256, 384]);  _unsafe_view_304 = None
        permute_577 = torch.ops.aten.permute.default(view_455, [1, 0])
        mm_255 = torch.ops.aten.mm.default(permute_577, view_161);  permute_577 = view_161 = None
        permute_578 = torch.ops.aten.permute.default(mm_255, [1, 0]);  mm_255 = None
        mm_256 = torch.ops.aten.mm.default(view_455, permute_579);  view_455 = permute_579 = None
        view_456 = torch.ops.aten.view.default(mm_256, [2, 128, 512]);  mm_256 = None
        add_250 = torch.ops.aten.add.Tensor(add_249, view_456);  add_249 = view_456 = None
        permute_580 = torch.ops.aten.permute.default(permute_578, [1, 0]);  permute_578 = None
        mul_630 = torch.ops.aten.mul.Tensor(add_250, primals_27);  primals_27 = None
        mul_631 = torch.ops.aten.mul.Tensor(add_250, mul_242);  add_250 = mul_242 = None
        sum_67 = torch.ops.aten.sum.dim_IntList(mul_631, [0, 1], True);  mul_631 = None
        view_457 = torch.ops.aten.view.default(sum_67, [512]);  sum_67 = None
        mul_632 = torch.ops.aten.mul.Tensor(mul_630, add_105)
        mul_633 = torch.ops.aten.mul.Tensor(mul_630, reciprocal_37);  mul_630 = reciprocal_37 = None
        sum_68 = torch.ops.aten.sum.dim_IntList(mul_632, [2], True);  mul_632 = None
        add_251 = torch.ops.aten.add.Tensor(add_245, mul_633);  add_245 = mul_633 = None
        alias_397 = torch.ops.aten.alias.default(alias_218);  alias_218 = None
        alias_398 = torch.ops.aten.alias.default(alias_397);  alias_397 = None
        pow_94 = torch.ops.aten.pow.Tensor_Scalar(alias_398, 3);  alias_398 = None
        mul_634 = torch.ops.aten.mul.Scalar(sum_68, -0.5);  sum_68 = None
        mul_635 = torch.ops.aten.mul.Tensor(mul_634, pow_94);  mul_634 = pow_94 = None
        expand_111 = torch.ops.aten.expand.default(mul_635, [2, 128, 512]);  mul_635 = None
        div_44 = torch.ops.aten.div.Scalar(expand_111, 512);  expand_111 = None
        pow_95 = torch.ops.aten.pow.Tensor_Scalar(add_105, 1.0);  add_105 = None
        mul_636 = torch.ops.aten.mul.Scalar(pow_95, 2.0);  pow_95 = None
        mul_637 = torch.ops.aten.mul.Tensor(div_44, mul_636);  div_44 = mul_636 = None
        add_252 = torch.ops.aten.add.Tensor(add_251, mul_637);  add_251 = mul_637 = None
        _to_copy_62 = torch.ops.aten._to_copy.default(gt_53, dtype = torch.float32);  gt_53 = None
        mul_638 = torch.ops.aten.mul.Tensor(_to_copy_62, 1.1111111111111112);  _to_copy_62 = None
        mul_639 = torch.ops.aten.mul.Tensor(add_252, mul_638);  mul_638 = None
        view_458 = torch.ops.aten.view.default(mul_639, [256, 512]);  mul_639 = None
        permute_581 = torch.ops.aten.permute.default(view_458, [1, 0])
        mm_257 = torch.ops.aten.mm.default(permute_581, view_160);  permute_581 = view_160 = None
        permute_582 = torch.ops.aten.permute.default(mm_257, [1, 0]);  mm_257 = None
        mm_258 = torch.ops.aten.mm.default(view_458, permute_583);  view_458 = permute_583 = None
        view_459 = torch.ops.aten.view.default(mm_258, [2, 128, 1024]);  mm_258 = None
        permute_584 = torch.ops.aten.permute.default(permute_582, [1, 0]);  permute_582 = None
        _to_copy_63 = torch.ops.aten._to_copy.default(gt_52, dtype = torch.float32);  gt_52 = None
        mul_640 = torch.ops.aten.mul.Tensor(_to_copy_63, 1.1111111111111112);  _to_copy_63 = None
        mul_641 = torch.ops.aten.mul.Tensor(view_459, mul_640);  view_459 = mul_640 = None
        mul_642 = torch.ops.aten.mul.Tensor(mul_641, mul_236);  mul_236 = None
        mul_643 = torch.ops.aten.mul.Tensor(mul_641, _unsafe_view_157);  mul_641 = _unsafe_view_157 = None
        view_460 = torch.ops.aten.view.default(mul_642, [256, 1024]);  mul_642 = None
        permute_585 = torch.ops.aten.permute.default(view_460, [1, 0])
        mm_259 = torch.ops.aten.mm.default(permute_585, view_158);  permute_585 = None
        permute_586 = torch.ops.aten.permute.default(mm_259, [1, 0]);  mm_259 = None
        mm_260 = torch.ops.aten.mm.default(view_460, permute_587);  view_460 = permute_587 = None
        view_461 = torch.ops.aten.view.default(mm_260, [2, 128, 512]);  mm_260 = None
        permute_588 = torch.ops.aten.permute.default(permute_586, [1, 0]);  permute_586 = None
        mul_644 = torch.ops.aten.mul.Tensor(mul_643, mul_231);  mul_231 = None
        mul_645 = torch.ops.aten.mul.Tensor(mul_643, add_104);  mul_643 = add_104 = None
        alias_399 = torch.ops.aten.alias.default(alias_213);  alias_213 = None
        alias_400 = torch.ops.aten.alias.default(alias_399);  alias_399 = None
        mul_646 = torch.ops.aten.mul.Tensor(alias_400, alias_400);  alias_400 = None
        _tensor_constant8 = self._tensor_constant8
        lift_fresh_copy_8 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant8);  _tensor_constant8 = None
        sub_63 = torch.ops.aten.sub.Tensor(lift_fresh_copy_8, mul_646);  lift_fresh_copy_8 = mul_646 = None
        mul_647 = torch.ops.aten.mul.Tensor(mul_644, sub_63);  mul_644 = sub_63 = None
        mul_648 = torch.ops.aten.mul.Tensor(mul_647, 0.7978845608028654);  mul_647 = None
        mul_649 = torch.ops.aten.mul.Tensor(mul_648, 0.044715)
        pow_96 = torch.ops.aten.pow.Tensor_Scalar(_unsafe_view_156, 2.0);  _unsafe_view_156 = None
        mul_650 = torch.ops.aten.mul.Scalar(pow_96, 3.0);  pow_96 = None
        mul_651 = torch.ops.aten.mul.Tensor(mul_649, mul_650);  mul_649 = mul_650 = None
        add_253 = torch.ops.aten.add.Tensor(mul_648, mul_651);  mul_648 = mul_651 = None
        mul_652 = torch.ops.aten.mul.Tensor(mul_645, 0.5);  mul_645 = None
        add_254 = torch.ops.aten.add.Tensor(add_253, mul_652);  add_253 = mul_652 = None
        view_462 = torch.ops.aten.view.default(add_254, [256, 1024]);  add_254 = None
        permute_589 = torch.ops.aten.permute.default(view_462, [1, 0])
        mm_261 = torch.ops.aten.mm.default(permute_589, view_158);  permute_589 = view_158 = None
        permute_590 = torch.ops.aten.permute.default(mm_261, [1, 0]);  mm_261 = None
        mm_262 = torch.ops.aten.mm.default(view_462, permute_591);  view_462 = permute_591 = None
        view_463 = torch.ops.aten.view.default(mm_262, [2, 128, 512]);  mm_262 = None
        add_255 = torch.ops.aten.add.Tensor(view_461, view_463);  view_461 = view_463 = None
        permute_592 = torch.ops.aten.permute.default(permute_590, [1, 0]);  permute_590 = None
        mul_653 = torch.ops.aten.mul.Tensor(add_255, primals_26);  primals_26 = None
        mul_654 = torch.ops.aten.mul.Tensor(add_255, mul_229);  add_255 = mul_229 = None
        sum_69 = torch.ops.aten.sum.dim_IntList(mul_654, [0, 1], True);  mul_654 = None
        view_464 = torch.ops.aten.view.default(sum_69, [512]);  sum_69 = None
        mul_655 = torch.ops.aten.mul.Tensor(mul_653, add_100)
        mul_656 = torch.ops.aten.mul.Tensor(mul_653, reciprocal_35);  mul_653 = reciprocal_35 = None
        sum_70 = torch.ops.aten.sum.dim_IntList(mul_655, [2], True);  mul_655 = None
        add_256 = torch.ops.aten.add.Tensor(add_252, mul_656);  add_252 = mul_656 = None
        alias_401 = torch.ops.aten.alias.default(alias_210);  alias_210 = None
        alias_402 = torch.ops.aten.alias.default(alias_401);  alias_401 = None
        pow_97 = torch.ops.aten.pow.Tensor_Scalar(alias_402, 3);  alias_402 = None
        mul_657 = torch.ops.aten.mul.Scalar(sum_70, -0.5);  sum_70 = None
        mul_658 = torch.ops.aten.mul.Tensor(mul_657, pow_97);  mul_657 = pow_97 = None
        expand_112 = torch.ops.aten.expand.default(mul_658, [2, 128, 512]);  mul_658 = None
        div_45 = torch.ops.aten.div.Scalar(expand_112, 512);  expand_112 = None
        pow_98 = torch.ops.aten.pow.Tensor_Scalar(add_100, 1.0);  add_100 = None
        mul_659 = torch.ops.aten.mul.Scalar(pow_98, 2.0);  pow_98 = None
        mul_660 = torch.ops.aten.mul.Tensor(div_45, mul_659);  div_45 = mul_659 = None
        add_257 = torch.ops.aten.add.Tensor(add_256, mul_660);  add_256 = mul_660 = None
        _to_copy_64 = torch.ops.aten._to_copy.default(gt_51, dtype = torch.float32);  gt_51 = None
        mul_661 = torch.ops.aten.mul.Tensor(_to_copy_64, 1.1111111111111112);  _to_copy_64 = None
        mul_662 = torch.ops.aten.mul.Tensor(add_257, mul_661);  mul_661 = None
        view_465 = torch.ops.aten.view.default(mul_662, [256, 512]);  mul_662 = None
        permute_593 = torch.ops.aten.permute.default(view_465, [1, 0])
        mm_263 = torch.ops.aten.mm.default(permute_593, view_157);  permute_593 = view_157 = None
        permute_594 = torch.ops.aten.permute.default(mm_263, [1, 0]);  mm_263 = None
        mm_264 = torch.ops.aten.mm.default(view_465, permute_595);  view_465 = permute_595 = None
        view_466 = torch.ops.aten.view.default(mm_264, [2, 128, 384]);  mm_264 = None
        permute_596 = torch.ops.aten.permute.default(permute_594, [1, 0]);  permute_594 = None
        view_467 = torch.ops.aten.view.default(view_466, [2, 128, 6, 64]);  view_466 = None
        permute_597 = torch.ops.aten.permute.default(view_467, [0, 2, 1, 3]);  view_467 = None
        clone_136 = torch.ops.aten.clone.default(permute_597, memory_format = torch.contiguous_format);  permute_597 = None
        _unsafe_view_305 = torch.ops.aten._unsafe_view.default(clone_136, [12, 128, 64]);  clone_136 = None
        bmm_88 = torch.ops.aten.bmm.default(permute_598, _unsafe_view_305);  permute_598 = None
        bmm_89 = torch.ops.aten.bmm.default(_unsafe_view_305, permute_599);  _unsafe_view_305 = permute_599 = None
        view_468 = torch.ops.aten.view.default(bmm_88, [2, 6, 128, 64]);  bmm_88 = None
        add_258 = torch.ops.aten.add.Tensor(tangents_14, view_468);  tangents_14 = view_468 = None
        view_469 = torch.ops.aten.view.default(bmm_89, [2, 6, 128, 128]);  bmm_89 = None
        philox_rand_like_34 = torch.ops.prims.philox_rand_like.default(view_469, philox_seed_like, 2555904)
        gt_95 = torch.ops.aten.gt.Scalar(philox_rand_like_34, 0.1);  philox_rand_like_34 = None
        _to_copy_65 = torch.ops.aten._to_copy.default(gt_95, dtype = torch.float32);  gt_95 = None
        mul_663 = torch.ops.aten.mul.Tensor(_to_copy_65, view_469);  _to_copy_65 = view_469 = None
        mul_664 = torch.ops.aten.mul.Tensor(mul_663, 1.1111111111111112);  mul_663 = None
        alias_403 = torch.ops.aten.alias.default(alias_206);  alias_206 = None
        alias_404 = torch.ops.aten.alias.default(alias_403);  alias_403 = None
        mul_665 = torch.ops.aten.mul.Tensor(mul_664, alias_404);  mul_664 = None
        sum_71 = torch.ops.aten.sum.dim_IntList(mul_665, [-1], True)
        mul_666 = torch.ops.aten.mul.Tensor(alias_404, sum_71);  alias_404 = sum_71 = None
        sub_64 = torch.ops.aten.sub.Tensor(mul_665, mul_666);  mul_665 = mul_666 = None
        view_470 = torch.ops.aten.view.default(sub_64, [12, 128, 128]);  sub_64 = None
        bmm_90 = torch.ops.aten.bmm.default(permute_600, view_470);  permute_600 = None
        bmm_91 = torch.ops.aten.bmm.default(view_470, permute_601);  view_470 = permute_601 = None
        view_471 = torch.ops.aten.view.default(bmm_90, [2, 6, 64, 128]);  bmm_90 = None
        view_472 = torch.ops.aten.view.default(bmm_91, [2, 6, 128, 64]);  bmm_91 = None
        permute_602 = torch.ops.aten.permute.default(view_471, [0, 1, 3, 2]);  view_471 = None
        add_259 = torch.ops.aten.add.Tensor(tangents_13, permute_602);  tangents_13 = permute_602 = None
        permute_603 = torch.ops.aten.permute.default(add_258, [0, 2, 1, 3]);  add_258 = None
        clone_137 = torch.ops.aten.clone.default(permute_603, memory_format = torch.contiguous_format);  permute_603 = None
        _unsafe_view_306 = torch.ops.aten._unsafe_view.default(clone_137, [2, 128, 384]);  clone_137 = None
        view_473 = torch.ops.aten.view.default(_unsafe_view_306, [256, 384]);  _unsafe_view_306 = None
        permute_604 = torch.ops.aten.permute.default(view_473, [1, 0])
        mm_265 = torch.ops.aten.mm.default(permute_604, view_109);  permute_604 = None
        permute_605 = torch.ops.aten.permute.default(mm_265, [1, 0]);  mm_265 = None
        mm_266 = torch.ops.aten.mm.default(view_473, permute_606);  view_473 = permute_606 = None
        view_474 = torch.ops.aten.view.default(mm_266, [2, 128, 512]);  mm_266 = None
        add_260 = torch.ops.aten.add.Tensor(add_243, view_474);  add_243 = view_474 = None
        permute_607 = torch.ops.aten.permute.default(permute_605, [1, 0]);  permute_605 = None
        permute_608 = torch.ops.aten.permute.default(add_259, [0, 2, 1, 3]);  add_259 = None
        clone_138 = torch.ops.aten.clone.default(permute_608, memory_format = torch.contiguous_format);  permute_608 = None
        _unsafe_view_307 = torch.ops.aten._unsafe_view.default(clone_138, [2, 128, 384]);  clone_138 = None
        view_475 = torch.ops.aten.view.default(_unsafe_view_307, [256, 384]);  _unsafe_view_307 = None
        permute_609 = torch.ops.aten.permute.default(view_475, [1, 0])
        mm_267 = torch.ops.aten.mm.default(permute_609, view_109);  permute_609 = None
        permute_610 = torch.ops.aten.permute.default(mm_267, [1, 0]);  mm_267 = None
        mm_268 = torch.ops.aten.mm.default(view_475, permute_611);  view_475 = permute_611 = None
        view_476 = torch.ops.aten.view.default(mm_268, [2, 128, 512]);  mm_268 = None
        add_261 = torch.ops.aten.add.Tensor(add_260, view_476);  add_260 = view_476 = None
        permute_612 = torch.ops.aten.permute.default(permute_610, [1, 0]);  permute_610 = None
        permute_613 = torch.ops.aten.permute.default(view_472, [0, 2, 1, 3]);  view_472 = None
        clone_139 = torch.ops.aten.clone.default(permute_613, memory_format = torch.contiguous_format);  permute_613 = None
        _unsafe_view_308 = torch.ops.aten._unsafe_view.default(clone_139, [2, 128, 384]);  clone_139 = None
        view_477 = torch.ops.aten.view.default(_unsafe_view_308, [256, 384]);  _unsafe_view_308 = None
        permute_614 = torch.ops.aten.permute.default(view_477, [1, 0])
        mm_269 = torch.ops.aten.mm.default(permute_614, view_149);  permute_614 = view_149 = None
        permute_615 = torch.ops.aten.permute.default(mm_269, [1, 0]);  mm_269 = None
        mm_270 = torch.ops.aten.mm.default(view_477, permute_616);  view_477 = permute_616 = None
        view_478 = torch.ops.aten.view.default(mm_270, [2, 128, 512]);  mm_270 = None
        permute_617 = torch.ops.aten.permute.default(permute_615, [1, 0]);  permute_615 = None
        mul_667 = torch.ops.aten.mul.Tensor(view_478, primals_25);  primals_25 = None
        mul_668 = torch.ops.aten.mul.Tensor(view_478, mul_223);  view_478 = mul_223 = None
        sum_72 = torch.ops.aten.sum.dim_IntList(mul_668, [0, 1], True);  mul_668 = None
        view_479 = torch.ops.aten.view.default(sum_72, [512]);  sum_72 = None
        mul_669 = torch.ops.aten.mul.Tensor(mul_667, add_97)
        mul_670 = torch.ops.aten.mul.Tensor(mul_667, reciprocal_34);  mul_667 = reciprocal_34 = None
        sum_73 = torch.ops.aten.sum.dim_IntList(mul_669, [2], True);  mul_669 = None
        add_262 = torch.ops.aten.add.Tensor(add_257, mul_670);  add_257 = mul_670 = None
        alias_405 = torch.ops.aten.alias.default(alias_203);  alias_203 = None
        alias_406 = torch.ops.aten.alias.default(alias_405);  alias_405 = None
        pow_99 = torch.ops.aten.pow.Tensor_Scalar(alias_406, 3);  alias_406 = None
        mul_671 = torch.ops.aten.mul.Scalar(sum_73, -0.5);  sum_73 = None
        mul_672 = torch.ops.aten.mul.Tensor(mul_671, pow_99);  mul_671 = pow_99 = None
        expand_113 = torch.ops.aten.expand.default(mul_672, [2, 128, 512]);  mul_672 = None
        div_46 = torch.ops.aten.div.Scalar(expand_113, 512);  expand_113 = None
        pow_100 = torch.ops.aten.pow.Tensor_Scalar(add_97, 1.0);  add_97 = None
        mul_673 = torch.ops.aten.mul.Scalar(pow_100, 2.0);  pow_100 = None
        mul_674 = torch.ops.aten.mul.Tensor(div_46, mul_673);  div_46 = mul_673 = None
        add_263 = torch.ops.aten.add.Tensor(add_262, mul_674);  add_262 = mul_674 = None
        _to_copy_66 = torch.ops.aten._to_copy.default(gt_49, dtype = torch.float32);  gt_49 = None
        mul_675 = torch.ops.aten.mul.Tensor(_to_copy_66, 1.1111111111111112);  _to_copy_66 = None
        mul_676 = torch.ops.aten.mul.Tensor(add_263, mul_675);  mul_675 = None
        view_480 = torch.ops.aten.view.default(mul_676, [256, 512]);  mul_676 = None
        permute_618 = torch.ops.aten.permute.default(view_480, [1, 0])
        mm_271 = torch.ops.aten.mm.default(permute_618, view_148);  permute_618 = view_148 = None
        permute_619 = torch.ops.aten.permute.default(mm_271, [1, 0]);  mm_271 = None
        mm_272 = torch.ops.aten.mm.default(view_480, permute_620);  view_480 = permute_620 = None
        view_481 = torch.ops.aten.view.default(mm_272, [2, 128, 384]);  mm_272 = None
        permute_621 = torch.ops.aten.permute.default(permute_619, [1, 0]);  permute_619 = None
        view_482 = torch.ops.aten.view.default(view_481, [2, 128, 6, 64]);  view_481 = None
        permute_622 = torch.ops.aten.permute.default(view_482, [0, 2, 1, 3]);  view_482 = None
        clone_140 = torch.ops.aten.clone.default(permute_622, memory_format = torch.contiguous_format);  permute_622 = None
        _unsafe_view_309 = torch.ops.aten._unsafe_view.default(clone_140, [12, 128, 64]);  clone_140 = None
        bmm_92 = torch.ops.aten.bmm.default(permute_623, _unsafe_view_309);  permute_623 = None
        bmm_93 = torch.ops.aten.bmm.default(_unsafe_view_309, permute_624);  _unsafe_view_309 = permute_624 = None
        view_483 = torch.ops.aten.view.default(bmm_92, [2, 6, 128, 64]);  bmm_92 = None
        add_264 = torch.ops.aten.add.Tensor(tangents_12, view_483);  tangents_12 = view_483 = None
        view_484 = torch.ops.aten.view.default(bmm_93, [2, 6, 128, 128]);  bmm_93 = None
        philox_rand_like_35 = torch.ops.prims.philox_rand_like.default(view_484, philox_seed_like, 2359296)
        gt_96 = torch.ops.aten.gt.Scalar(philox_rand_like_35, 0.1);  philox_rand_like_35 = None
        _to_copy_67 = torch.ops.aten._to_copy.default(gt_96, dtype = torch.float32);  gt_96 = None
        mul_677 = torch.ops.aten.mul.Tensor(_to_copy_67, view_484);  _to_copy_67 = view_484 = None
        mul_678 = torch.ops.aten.mul.Tensor(mul_677, 1.1111111111111112);  mul_677 = None
        alias_407 = torch.ops.aten.alias.default(alias_199);  alias_199 = None
        alias_408 = torch.ops.aten.alias.default(alias_407);  alias_407 = None
        mul_679 = torch.ops.aten.mul.Tensor(mul_678, alias_408);  mul_678 = None
        sum_74 = torch.ops.aten.sum.dim_IntList(mul_679, [-1], True)
        mul_680 = torch.ops.aten.mul.Tensor(alias_408, sum_74);  alias_408 = sum_74 = None
        sub_65 = torch.ops.aten.sub.Tensor(mul_679, mul_680);  mul_679 = mul_680 = None
        add_265 = torch.ops.aten.add.Tensor(add_247, sub_65);  add_247 = None
        view_485 = torch.ops.aten.view.default(sub_65, [12, 128, 128]);  sub_65 = None
        bmm_94 = torch.ops.aten.bmm.default(permute_625, view_485);  permute_625 = None
        bmm_95 = torch.ops.aten.bmm.default(view_485, permute_626);  view_485 = permute_626 = None
        view_486 = torch.ops.aten.view.default(bmm_94, [2, 6, 64, 128]);  bmm_94 = None
        view_487 = torch.ops.aten.view.default(bmm_95, [2, 6, 128, 64]);  bmm_95 = None
        permute_627 = torch.ops.aten.permute.default(view_486, [0, 1, 3, 2]);  view_486 = None
        add_266 = torch.ops.aten.add.Tensor(tangents_11, permute_627);  tangents_11 = permute_627 = None
        permute_628 = torch.ops.aten.permute.default(add_264, [0, 2, 1, 3]);  add_264 = None
        clone_141 = torch.ops.aten.clone.default(permute_628, memory_format = torch.contiguous_format);  permute_628 = None
        _unsafe_view_310 = torch.ops.aten._unsafe_view.default(clone_141, [2, 128, 384]);  clone_141 = None
        view_488 = torch.ops.aten.view.default(_unsafe_view_310, [256, 384]);  _unsafe_view_310 = None
        permute_629 = torch.ops.aten.permute.default(view_488, [1, 0])
        mm_273 = torch.ops.aten.mm.default(permute_629, view_140);  permute_629 = None
        permute_630 = torch.ops.aten.permute.default(mm_273, [1, 0]);  mm_273 = None
        mm_274 = torch.ops.aten.mm.default(view_488, permute_631);  view_488 = permute_631 = None
        view_489 = torch.ops.aten.view.default(mm_274, [2, 128, 512]);  mm_274 = None
        permute_632 = torch.ops.aten.permute.default(permute_630, [1, 0]);  permute_630 = None
        permute_633 = torch.ops.aten.permute.default(add_266, [0, 2, 1, 3]);  add_266 = None
        clone_142 = torch.ops.aten.clone.default(permute_633, memory_format = torch.contiguous_format);  permute_633 = None
        _unsafe_view_311 = torch.ops.aten._unsafe_view.default(clone_142, [2, 128, 384]);  clone_142 = None
        view_490 = torch.ops.aten.view.default(_unsafe_view_311, [256, 384]);  _unsafe_view_311 = None
        permute_634 = torch.ops.aten.permute.default(view_490, [1, 0])
        mm_275 = torch.ops.aten.mm.default(permute_634, view_140);  permute_634 = None
        permute_635 = torch.ops.aten.permute.default(mm_275, [1, 0]);  mm_275 = None
        mm_276 = torch.ops.aten.mm.default(view_490, permute_636);  view_490 = permute_636 = None
        view_491 = torch.ops.aten.view.default(mm_276, [2, 128, 512]);  mm_276 = None
        add_267 = torch.ops.aten.add.Tensor(view_489, view_491);  view_489 = view_491 = None
        permute_637 = torch.ops.aten.permute.default(permute_635, [1, 0]);  permute_635 = None
        permute_638 = torch.ops.aten.permute.default(view_487, [0, 2, 1, 3]);  view_487 = None
        clone_143 = torch.ops.aten.clone.default(permute_638, memory_format = torch.contiguous_format);  permute_638 = None
        _unsafe_view_312 = torch.ops.aten._unsafe_view.default(clone_143, [2, 128, 384]);  clone_143 = None
        view_492 = torch.ops.aten.view.default(_unsafe_view_312, [256, 384]);  _unsafe_view_312 = None
        permute_639 = torch.ops.aten.permute.default(view_492, [1, 0])
        mm_277 = torch.ops.aten.mm.default(permute_639, view_140);  permute_639 = view_140 = None
        permute_640 = torch.ops.aten.permute.default(mm_277, [1, 0]);  mm_277 = None
        mm_278 = torch.ops.aten.mm.default(view_492, permute_641);  view_492 = permute_641 = None
        view_493 = torch.ops.aten.view.default(mm_278, [2, 128, 512]);  mm_278 = None
        add_268 = torch.ops.aten.add.Tensor(add_267, view_493);  add_267 = view_493 = None
        permute_642 = torch.ops.aten.permute.default(permute_640, [1, 0]);  permute_640 = None
        mul_681 = torch.ops.aten.mul.Tensor(add_268, primals_24);  primals_24 = None
        mul_682 = torch.ops.aten.mul.Tensor(add_268, mul_217);  add_268 = mul_217 = None
        sum_75 = torch.ops.aten.sum.dim_IntList(mul_682, [0, 1], True);  mul_682 = None
        view_494 = torch.ops.aten.view.default(sum_75, [512]);  sum_75 = None
        mul_683 = torch.ops.aten.mul.Tensor(mul_681, add_94)
        mul_684 = torch.ops.aten.mul.Tensor(mul_681, reciprocal_33);  mul_681 = reciprocal_33 = None
        sum_76 = torch.ops.aten.sum.dim_IntList(mul_683, [2], True);  mul_683 = None
        add_269 = torch.ops.aten.add.Tensor(add_263, mul_684);  add_263 = mul_684 = None
        alias_409 = torch.ops.aten.alias.default(alias_196);  alias_196 = None
        alias_410 = torch.ops.aten.alias.default(alias_409);  alias_409 = None
        pow_101 = torch.ops.aten.pow.Tensor_Scalar(alias_410, 3);  alias_410 = None
        mul_685 = torch.ops.aten.mul.Scalar(sum_76, -0.5);  sum_76 = None
        mul_686 = torch.ops.aten.mul.Tensor(mul_685, pow_101);  mul_685 = pow_101 = None
        expand_114 = torch.ops.aten.expand.default(mul_686, [2, 128, 512]);  mul_686 = None
        div_47 = torch.ops.aten.div.Scalar(expand_114, 512);  expand_114 = None
        pow_102 = torch.ops.aten.pow.Tensor_Scalar(add_94, 1.0);  add_94 = None
        mul_687 = torch.ops.aten.mul.Scalar(pow_102, 2.0);  pow_102 = None
        mul_688 = torch.ops.aten.mul.Tensor(div_47, mul_687);  div_47 = mul_687 = None
        add_270 = torch.ops.aten.add.Tensor(add_269, mul_688);  add_269 = mul_688 = None
        _to_copy_68 = torch.ops.aten._to_copy.default(gt_47, dtype = torch.float32);  gt_47 = None
        mul_689 = torch.ops.aten.mul.Tensor(_to_copy_68, 1.1111111111111112);  _to_copy_68 = None
        mul_690 = torch.ops.aten.mul.Tensor(add_270, mul_689);  mul_689 = None
        view_495 = torch.ops.aten.view.default(mul_690, [256, 512]);  mul_690 = None
        permute_643 = torch.ops.aten.permute.default(view_495, [1, 0])
        mm_279 = torch.ops.aten.mm.default(permute_643, view_139);  permute_643 = view_139 = None
        permute_644 = torch.ops.aten.permute.default(mm_279, [1, 0]);  mm_279 = None
        mm_280 = torch.ops.aten.mm.default(view_495, permute_645);  view_495 = permute_645 = None
        view_496 = torch.ops.aten.view.default(mm_280, [2, 128, 1024]);  mm_280 = None
        permute_646 = torch.ops.aten.permute.default(permute_644, [1, 0]);  permute_644 = None
        _to_copy_69 = torch.ops.aten._to_copy.default(gt_46, dtype = torch.float32);  gt_46 = None
        mul_691 = torch.ops.aten.mul.Tensor(_to_copy_69, 1.1111111111111112);  _to_copy_69 = None
        mul_692 = torch.ops.aten.mul.Tensor(view_496, mul_691);  view_496 = mul_691 = None
        mul_693 = torch.ops.aten.mul.Tensor(mul_692, mul_211);  mul_211 = None
        mul_694 = torch.ops.aten.mul.Tensor(mul_692, _unsafe_view_136);  mul_692 = _unsafe_view_136 = None
        view_497 = torch.ops.aten.view.default(mul_693, [256, 1024]);  mul_693 = None
        permute_647 = torch.ops.aten.permute.default(view_497, [1, 0])
        mm_281 = torch.ops.aten.mm.default(permute_647, view_137);  permute_647 = None
        permute_648 = torch.ops.aten.permute.default(mm_281, [1, 0]);  mm_281 = None
        mm_282 = torch.ops.aten.mm.default(view_497, permute_649);  view_497 = permute_649 = None
        view_498 = torch.ops.aten.view.default(mm_282, [2, 128, 512]);  mm_282 = None
        permute_650 = torch.ops.aten.permute.default(permute_648, [1, 0]);  permute_648 = None
        mul_695 = torch.ops.aten.mul.Tensor(mul_694, mul_206);  mul_206 = None
        mul_696 = torch.ops.aten.mul.Tensor(mul_694, add_93);  mul_694 = add_93 = None
        alias_411 = torch.ops.aten.alias.default(alias_191);  alias_191 = None
        alias_412 = torch.ops.aten.alias.default(alias_411);  alias_411 = None
        mul_697 = torch.ops.aten.mul.Tensor(alias_412, alias_412);  alias_412 = None
        _tensor_constant9 = self._tensor_constant9
        lift_fresh_copy_9 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant9);  _tensor_constant9 = None
        sub_66 = torch.ops.aten.sub.Tensor(lift_fresh_copy_9, mul_697);  lift_fresh_copy_9 = mul_697 = None
        mul_698 = torch.ops.aten.mul.Tensor(mul_695, sub_66);  mul_695 = sub_66 = None
        mul_699 = torch.ops.aten.mul.Tensor(mul_698, 0.7978845608028654);  mul_698 = None
        mul_700 = torch.ops.aten.mul.Tensor(mul_699, 0.044715)
        pow_103 = torch.ops.aten.pow.Tensor_Scalar(_unsafe_view_135, 2.0);  _unsafe_view_135 = None
        mul_701 = torch.ops.aten.mul.Scalar(pow_103, 3.0);  pow_103 = None
        mul_702 = torch.ops.aten.mul.Tensor(mul_700, mul_701);  mul_700 = mul_701 = None
        add_271 = torch.ops.aten.add.Tensor(mul_699, mul_702);  mul_699 = mul_702 = None
        mul_703 = torch.ops.aten.mul.Tensor(mul_696, 0.5);  mul_696 = None
        add_272 = torch.ops.aten.add.Tensor(add_271, mul_703);  add_271 = mul_703 = None
        view_499 = torch.ops.aten.view.default(add_272, [256, 1024]);  add_272 = None
        permute_651 = torch.ops.aten.permute.default(view_499, [1, 0])
        mm_283 = torch.ops.aten.mm.default(permute_651, view_137);  permute_651 = view_137 = None
        permute_652 = torch.ops.aten.permute.default(mm_283, [1, 0]);  mm_283 = None
        mm_284 = torch.ops.aten.mm.default(view_499, permute_653);  view_499 = permute_653 = None
        view_500 = torch.ops.aten.view.default(mm_284, [2, 128, 512]);  mm_284 = None
        add_273 = torch.ops.aten.add.Tensor(view_498, view_500);  view_498 = view_500 = None
        permute_654 = torch.ops.aten.permute.default(permute_652, [1, 0]);  permute_652 = None
        mul_704 = torch.ops.aten.mul.Tensor(add_273, primals_23);  primals_23 = None
        mul_705 = torch.ops.aten.mul.Tensor(add_273, mul_204);  add_273 = mul_204 = None
        sum_77 = torch.ops.aten.sum.dim_IntList(mul_705, [0, 1], True);  mul_705 = None
        view_501 = torch.ops.aten.view.default(sum_77, [512]);  sum_77 = None
        mul_706 = torch.ops.aten.mul.Tensor(mul_704, add_89)
        mul_707 = torch.ops.aten.mul.Tensor(mul_704, reciprocal_31);  mul_704 = reciprocal_31 = None
        sum_78 = torch.ops.aten.sum.dim_IntList(mul_706, [2], True);  mul_706 = None
        add_274 = torch.ops.aten.add.Tensor(add_270, mul_707);  add_270 = mul_707 = None
        alias_413 = torch.ops.aten.alias.default(alias_188);  alias_188 = None
        alias_414 = torch.ops.aten.alias.default(alias_413);  alias_413 = None
        pow_104 = torch.ops.aten.pow.Tensor_Scalar(alias_414, 3);  alias_414 = None
        mul_708 = torch.ops.aten.mul.Scalar(sum_78, -0.5);  sum_78 = None
        mul_709 = torch.ops.aten.mul.Tensor(mul_708, pow_104);  mul_708 = pow_104 = None
        expand_115 = torch.ops.aten.expand.default(mul_709, [2, 128, 512]);  mul_709 = None
        div_48 = torch.ops.aten.div.Scalar(expand_115, 512);  expand_115 = None
        pow_105 = torch.ops.aten.pow.Tensor_Scalar(add_89, 1.0);  add_89 = None
        mul_710 = torch.ops.aten.mul.Scalar(pow_105, 2.0);  pow_105 = None
        mul_711 = torch.ops.aten.mul.Tensor(div_48, mul_710);  div_48 = mul_710 = None
        add_275 = torch.ops.aten.add.Tensor(add_274, mul_711);  add_274 = mul_711 = None
        _to_copy_70 = torch.ops.aten._to_copy.default(gt_45, dtype = torch.float32);  gt_45 = None
        mul_712 = torch.ops.aten.mul.Tensor(_to_copy_70, 1.1111111111111112);  _to_copy_70 = None
        mul_713 = torch.ops.aten.mul.Tensor(add_275, mul_712);  mul_712 = None
        view_502 = torch.ops.aten.view.default(mul_713, [256, 512]);  mul_713 = None
        permute_655 = torch.ops.aten.permute.default(view_502, [1, 0])
        mm_285 = torch.ops.aten.mm.default(permute_655, view_136);  permute_655 = view_136 = None
        permute_656 = torch.ops.aten.permute.default(mm_285, [1, 0]);  mm_285 = None
        mm_286 = torch.ops.aten.mm.default(view_502, permute_657);  view_502 = permute_657 = None
        view_503 = torch.ops.aten.view.default(mm_286, [2, 128, 384]);  mm_286 = None
        permute_658 = torch.ops.aten.permute.default(permute_656, [1, 0]);  permute_656 = None
        view_504 = torch.ops.aten.view.default(view_503, [2, 128, 6, 64]);  view_503 = None
        permute_659 = torch.ops.aten.permute.default(view_504, [0, 2, 1, 3]);  view_504 = None
        clone_144 = torch.ops.aten.clone.default(permute_659, memory_format = torch.contiguous_format);  permute_659 = None
        _unsafe_view_313 = torch.ops.aten._unsafe_view.default(clone_144, [12, 128, 64]);  clone_144 = None
        bmm_96 = torch.ops.aten.bmm.default(permute_660, _unsafe_view_313);  permute_660 = None
        bmm_97 = torch.ops.aten.bmm.default(_unsafe_view_313, permute_661);  _unsafe_view_313 = permute_661 = None
        view_505 = torch.ops.aten.view.default(bmm_96, [2, 6, 128, 64]);  bmm_96 = None
        add_276 = torch.ops.aten.add.Tensor(tangents_10, view_505);  tangents_10 = view_505 = None
        view_506 = torch.ops.aten.view.default(bmm_97, [2, 6, 128, 128]);  bmm_97 = None
        philox_rand_like_36 = torch.ops.prims.philox_rand_like.default(view_506, philox_seed_like, 2162688)
        gt_97 = torch.ops.aten.gt.Scalar(philox_rand_like_36, 0.1);  philox_rand_like_36 = None
        _to_copy_71 = torch.ops.aten._to_copy.default(gt_97, dtype = torch.float32);  gt_97 = None
        mul_714 = torch.ops.aten.mul.Tensor(_to_copy_71, view_506);  _to_copy_71 = view_506 = None
        mul_715 = torch.ops.aten.mul.Tensor(mul_714, 1.1111111111111112);  mul_714 = None
        alias_415 = torch.ops.aten.alias.default(alias_184);  alias_184 = None
        alias_416 = torch.ops.aten.alias.default(alias_415);  alias_415 = None
        mul_716 = torch.ops.aten.mul.Tensor(mul_715, alias_416);  mul_715 = None
        sum_79 = torch.ops.aten.sum.dim_IntList(mul_716, [-1], True)
        mul_717 = torch.ops.aten.mul.Tensor(alias_416, sum_79);  alias_416 = sum_79 = None
        sub_67 = torch.ops.aten.sub.Tensor(mul_716, mul_717);  mul_716 = mul_717 = None
        view_507 = torch.ops.aten.view.default(sub_67, [12, 128, 128]);  sub_67 = None
        bmm_98 = torch.ops.aten.bmm.default(permute_662, view_507);  permute_662 = None
        bmm_99 = torch.ops.aten.bmm.default(view_507, permute_663);  view_507 = permute_663 = None
        view_508 = torch.ops.aten.view.default(bmm_98, [2, 6, 64, 128]);  bmm_98 = None
        view_509 = torch.ops.aten.view.default(bmm_99, [2, 6, 128, 64]);  bmm_99 = None
        permute_664 = torch.ops.aten.permute.default(view_508, [0, 1, 3, 2]);  view_508 = None
        add_277 = torch.ops.aten.add.Tensor(tangents_9, permute_664);  tangents_9 = permute_664 = None
        permute_665 = torch.ops.aten.permute.default(add_276, [0, 2, 1, 3]);  add_276 = None
        clone_145 = torch.ops.aten.clone.default(permute_665, memory_format = torch.contiguous_format);  permute_665 = None
        _unsafe_view_314 = torch.ops.aten._unsafe_view.default(clone_145, [2, 128, 384]);  clone_145 = None
        view_510 = torch.ops.aten.view.default(_unsafe_view_314, [256, 384]);  _unsafe_view_314 = None
        permute_666 = torch.ops.aten.permute.default(view_510, [1, 0])
        mm_287 = torch.ops.aten.mm.default(permute_666, view_109);  permute_666 = None
        permute_667 = torch.ops.aten.permute.default(mm_287, [1, 0]);  mm_287 = None
        mm_288 = torch.ops.aten.mm.default(view_510, permute_668);  view_510 = permute_668 = None
        view_511 = torch.ops.aten.view.default(mm_288, [2, 128, 512]);  mm_288 = None
        add_278 = torch.ops.aten.add.Tensor(add_261, view_511);  add_261 = view_511 = None
        permute_669 = torch.ops.aten.permute.default(permute_667, [1, 0]);  permute_667 = None
        permute_670 = torch.ops.aten.permute.default(add_277, [0, 2, 1, 3]);  add_277 = None
        clone_146 = torch.ops.aten.clone.default(permute_670, memory_format = torch.contiguous_format);  permute_670 = None
        _unsafe_view_315 = torch.ops.aten._unsafe_view.default(clone_146, [2, 128, 384]);  clone_146 = None
        view_512 = torch.ops.aten.view.default(_unsafe_view_315, [256, 384]);  _unsafe_view_315 = None
        permute_671 = torch.ops.aten.permute.default(view_512, [1, 0])
        mm_289 = torch.ops.aten.mm.default(permute_671, view_109);  permute_671 = None
        permute_672 = torch.ops.aten.permute.default(mm_289, [1, 0]);  mm_289 = None
        mm_290 = torch.ops.aten.mm.default(view_512, permute_673);  view_512 = permute_673 = None
        view_513 = torch.ops.aten.view.default(mm_290, [2, 128, 512]);  mm_290 = None
        add_279 = torch.ops.aten.add.Tensor(add_278, view_513);  add_278 = view_513 = None
        permute_674 = torch.ops.aten.permute.default(permute_672, [1, 0]);  permute_672 = None
        permute_675 = torch.ops.aten.permute.default(view_509, [0, 2, 1, 3]);  view_509 = None
        clone_147 = torch.ops.aten.clone.default(permute_675, memory_format = torch.contiguous_format);  permute_675 = None
        _unsafe_view_316 = torch.ops.aten._unsafe_view.default(clone_147, [2, 128, 384]);  clone_147 = None
        view_514 = torch.ops.aten.view.default(_unsafe_view_316, [256, 384]);  _unsafe_view_316 = None
        permute_676 = torch.ops.aten.permute.default(view_514, [1, 0])
        mm_291 = torch.ops.aten.mm.default(permute_676, view_128);  permute_676 = view_128 = None
        permute_677 = torch.ops.aten.permute.default(mm_291, [1, 0]);  mm_291 = None
        mm_292 = torch.ops.aten.mm.default(view_514, permute_678);  view_514 = permute_678 = None
        view_515 = torch.ops.aten.view.default(mm_292, [2, 128, 512]);  mm_292 = None
        permute_679 = torch.ops.aten.permute.default(permute_677, [1, 0]);  permute_677 = None
        mul_718 = torch.ops.aten.mul.Tensor(view_515, primals_22);  primals_22 = None
        mul_719 = torch.ops.aten.mul.Tensor(view_515, mul_198);  view_515 = mul_198 = None
        sum_80 = torch.ops.aten.sum.dim_IntList(mul_719, [0, 1], True);  mul_719 = None
        view_516 = torch.ops.aten.view.default(sum_80, [512]);  sum_80 = None
        mul_720 = torch.ops.aten.mul.Tensor(mul_718, add_86)
        mul_721 = torch.ops.aten.mul.Tensor(mul_718, reciprocal_30);  mul_718 = reciprocal_30 = None
        sum_81 = torch.ops.aten.sum.dim_IntList(mul_720, [2], True);  mul_720 = None
        add_280 = torch.ops.aten.add.Tensor(add_275, mul_721);  add_275 = mul_721 = None
        alias_417 = torch.ops.aten.alias.default(alias_181);  alias_181 = None
        alias_418 = torch.ops.aten.alias.default(alias_417);  alias_417 = None
        pow_106 = torch.ops.aten.pow.Tensor_Scalar(alias_418, 3);  alias_418 = None
        mul_722 = torch.ops.aten.mul.Scalar(sum_81, -0.5);  sum_81 = None
        mul_723 = torch.ops.aten.mul.Tensor(mul_722, pow_106);  mul_722 = pow_106 = None
        expand_116 = torch.ops.aten.expand.default(mul_723, [2, 128, 512]);  mul_723 = None
        div_49 = torch.ops.aten.div.Scalar(expand_116, 512);  expand_116 = None
        pow_107 = torch.ops.aten.pow.Tensor_Scalar(add_86, 1.0);  add_86 = None
        mul_724 = torch.ops.aten.mul.Scalar(pow_107, 2.0);  pow_107 = None
        mul_725 = torch.ops.aten.mul.Tensor(div_49, mul_724);  div_49 = mul_724 = None
        add_281 = torch.ops.aten.add.Tensor(add_280, mul_725);  add_280 = mul_725 = None
        _to_copy_72 = torch.ops.aten._to_copy.default(gt_43, dtype = torch.float32);  gt_43 = None
        mul_726 = torch.ops.aten.mul.Tensor(_to_copy_72, 1.1111111111111112);  _to_copy_72 = None
        mul_727 = torch.ops.aten.mul.Tensor(add_281, mul_726);  mul_726 = None
        view_517 = torch.ops.aten.view.default(mul_727, [256, 512]);  mul_727 = None
        permute_680 = torch.ops.aten.permute.default(view_517, [1, 0])
        mm_293 = torch.ops.aten.mm.default(permute_680, view_127);  permute_680 = view_127 = None
        permute_681 = torch.ops.aten.permute.default(mm_293, [1, 0]);  mm_293 = None
        mm_294 = torch.ops.aten.mm.default(view_517, permute_682);  view_517 = permute_682 = None
        view_518 = torch.ops.aten.view.default(mm_294, [2, 128, 384]);  mm_294 = None
        permute_683 = torch.ops.aten.permute.default(permute_681, [1, 0]);  permute_681 = None
        view_519 = torch.ops.aten.view.default(view_518, [2, 128, 6, 64]);  view_518 = None
        permute_684 = torch.ops.aten.permute.default(view_519, [0, 2, 1, 3]);  view_519 = None
        clone_148 = torch.ops.aten.clone.default(permute_684, memory_format = torch.contiguous_format);  permute_684 = None
        _unsafe_view_317 = torch.ops.aten._unsafe_view.default(clone_148, [12, 128, 64]);  clone_148 = None
        bmm_100 = torch.ops.aten.bmm.default(permute_685, _unsafe_view_317);  permute_685 = None
        bmm_101 = torch.ops.aten.bmm.default(_unsafe_view_317, permute_686);  _unsafe_view_317 = permute_686 = None
        view_520 = torch.ops.aten.view.default(bmm_100, [2, 6, 128, 64]);  bmm_100 = None
        add_282 = torch.ops.aten.add.Tensor(tangents_8, view_520);  tangents_8 = view_520 = None
        view_521 = torch.ops.aten.view.default(bmm_101, [2, 6, 128, 128]);  bmm_101 = None
        philox_rand_like_37 = torch.ops.prims.philox_rand_like.default(view_521, philox_seed_like, 1966080)
        gt_98 = torch.ops.aten.gt.Scalar(philox_rand_like_37, 0.1);  philox_rand_like_37 = None
        _to_copy_73 = torch.ops.aten._to_copy.default(gt_98, dtype = torch.float32);  gt_98 = None
        mul_728 = torch.ops.aten.mul.Tensor(_to_copy_73, view_521);  _to_copy_73 = view_521 = None
        mul_729 = torch.ops.aten.mul.Tensor(mul_728, 1.1111111111111112);  mul_728 = None
        alias_419 = torch.ops.aten.alias.default(alias_177);  alias_177 = None
        alias_420 = torch.ops.aten.alias.default(alias_419);  alias_419 = None
        mul_730 = torch.ops.aten.mul.Tensor(mul_729, alias_420);  mul_729 = None
        sum_82 = torch.ops.aten.sum.dim_IntList(mul_730, [-1], True)
        mul_731 = torch.ops.aten.mul.Tensor(alias_420, sum_82);  alias_420 = sum_82 = None
        sub_68 = torch.ops.aten.sub.Tensor(mul_730, mul_731);  mul_730 = mul_731 = None
        add_283 = torch.ops.aten.add.Tensor(add_265, sub_68);  add_265 = None
        view_522 = torch.ops.aten.view.default(sub_68, [12, 128, 128]);  sub_68 = None
        bmm_102 = torch.ops.aten.bmm.default(permute_687, view_522);  permute_687 = None
        bmm_103 = torch.ops.aten.bmm.default(view_522, permute_688);  view_522 = permute_688 = None
        view_523 = torch.ops.aten.view.default(bmm_102, [2, 6, 64, 128]);  bmm_102 = None
        view_524 = torch.ops.aten.view.default(bmm_103, [2, 6, 128, 64]);  bmm_103 = None
        permute_689 = torch.ops.aten.permute.default(view_523, [0, 1, 3, 2]);  view_523 = None
        add_284 = torch.ops.aten.add.Tensor(tangents_7, permute_689);  tangents_7 = permute_689 = None
        permute_690 = torch.ops.aten.permute.default(add_282, [0, 2, 1, 3]);  add_282 = None
        clone_149 = torch.ops.aten.clone.default(permute_690, memory_format = torch.contiguous_format);  permute_690 = None
        _unsafe_view_318 = torch.ops.aten._unsafe_view.default(clone_149, [2, 128, 384]);  clone_149 = None
        view_525 = torch.ops.aten.view.default(_unsafe_view_318, [256, 384]);  _unsafe_view_318 = None
        permute_691 = torch.ops.aten.permute.default(view_525, [1, 0])
        mm_295 = torch.ops.aten.mm.default(permute_691, view_119);  permute_691 = None
        permute_692 = torch.ops.aten.permute.default(mm_295, [1, 0]);  mm_295 = None
        mm_296 = torch.ops.aten.mm.default(view_525, permute_693);  view_525 = permute_693 = None
        view_526 = torch.ops.aten.view.default(mm_296, [2, 128, 512]);  mm_296 = None
        permute_694 = torch.ops.aten.permute.default(permute_692, [1, 0]);  permute_692 = None
        permute_695 = torch.ops.aten.permute.default(add_284, [0, 2, 1, 3]);  add_284 = None
        clone_150 = torch.ops.aten.clone.default(permute_695, memory_format = torch.contiguous_format);  permute_695 = None
        _unsafe_view_319 = torch.ops.aten._unsafe_view.default(clone_150, [2, 128, 384]);  clone_150 = None
        view_527 = torch.ops.aten.view.default(_unsafe_view_319, [256, 384]);  _unsafe_view_319 = None
        permute_696 = torch.ops.aten.permute.default(view_527, [1, 0])
        mm_297 = torch.ops.aten.mm.default(permute_696, view_119);  permute_696 = None
        permute_697 = torch.ops.aten.permute.default(mm_297, [1, 0]);  mm_297 = None
        mm_298 = torch.ops.aten.mm.default(view_527, permute_698);  view_527 = permute_698 = None
        view_528 = torch.ops.aten.view.default(mm_298, [2, 128, 512]);  mm_298 = None
        add_285 = torch.ops.aten.add.Tensor(view_526, view_528);  view_526 = view_528 = None
        permute_699 = torch.ops.aten.permute.default(permute_697, [1, 0]);  permute_697 = None
        permute_700 = torch.ops.aten.permute.default(view_524, [0, 2, 1, 3]);  view_524 = None
        clone_151 = torch.ops.aten.clone.default(permute_700, memory_format = torch.contiguous_format);  permute_700 = None
        _unsafe_view_320 = torch.ops.aten._unsafe_view.default(clone_151, [2, 128, 384]);  clone_151 = None
        view_529 = torch.ops.aten.view.default(_unsafe_view_320, [256, 384]);  _unsafe_view_320 = None
        permute_701 = torch.ops.aten.permute.default(view_529, [1, 0])
        mm_299 = torch.ops.aten.mm.default(permute_701, view_119);  permute_701 = view_119 = None
        permute_702 = torch.ops.aten.permute.default(mm_299, [1, 0]);  mm_299 = None
        mm_300 = torch.ops.aten.mm.default(view_529, permute_703);  view_529 = permute_703 = None
        view_530 = torch.ops.aten.view.default(mm_300, [2, 128, 512]);  mm_300 = None
        add_286 = torch.ops.aten.add.Tensor(add_285, view_530);  add_285 = view_530 = None
        permute_704 = torch.ops.aten.permute.default(permute_702, [1, 0]);  permute_702 = None
        mul_732 = torch.ops.aten.mul.Tensor(add_286, primals_21);  primals_21 = None
        mul_733 = torch.ops.aten.mul.Tensor(add_286, mul_192);  add_286 = mul_192 = None
        sum_83 = torch.ops.aten.sum.dim_IntList(mul_733, [0, 1], True);  mul_733 = None
        view_531 = torch.ops.aten.view.default(sum_83, [512]);  sum_83 = None
        mul_734 = torch.ops.aten.mul.Tensor(mul_732, add_83)
        mul_735 = torch.ops.aten.mul.Tensor(mul_732, reciprocal_29);  mul_732 = reciprocal_29 = None
        sum_84 = torch.ops.aten.sum.dim_IntList(mul_734, [2], True);  mul_734 = None
        add_287 = torch.ops.aten.add.Tensor(add_281, mul_735);  add_281 = mul_735 = None
        alias_421 = torch.ops.aten.alias.default(alias_174);  alias_174 = None
        alias_422 = torch.ops.aten.alias.default(alias_421);  alias_421 = None
        pow_108 = torch.ops.aten.pow.Tensor_Scalar(alias_422, 3);  alias_422 = None
        mul_736 = torch.ops.aten.mul.Scalar(sum_84, -0.5);  sum_84 = None
        mul_737 = torch.ops.aten.mul.Tensor(mul_736, pow_108);  mul_736 = pow_108 = None
        expand_117 = torch.ops.aten.expand.default(mul_737, [2, 128, 512]);  mul_737 = None
        div_50 = torch.ops.aten.div.Scalar(expand_117, 512);  expand_117 = None
        pow_109 = torch.ops.aten.pow.Tensor_Scalar(add_83, 1.0);  add_83 = None
        mul_738 = torch.ops.aten.mul.Scalar(pow_109, 2.0);  pow_109 = None
        mul_739 = torch.ops.aten.mul.Tensor(div_50, mul_738);  div_50 = mul_738 = None
        add_288 = torch.ops.aten.add.Tensor(add_287, mul_739);  add_287 = mul_739 = None
        _to_copy_74 = torch.ops.aten._to_copy.default(gt_41, dtype = torch.float32);  gt_41 = None
        mul_740 = torch.ops.aten.mul.Tensor(_to_copy_74, 1.1111111111111112);  _to_copy_74 = None
        mul_741 = torch.ops.aten.mul.Tensor(add_288, mul_740);  mul_740 = None
        view_532 = torch.ops.aten.view.default(mul_741, [256, 512]);  mul_741 = None
        permute_705 = torch.ops.aten.permute.default(view_532, [1, 0])
        mm_301 = torch.ops.aten.mm.default(permute_705, view_118);  permute_705 = view_118 = None
        permute_706 = torch.ops.aten.permute.default(mm_301, [1, 0]);  mm_301 = None
        mm_302 = torch.ops.aten.mm.default(view_532, permute_707);  view_532 = permute_707 = None
        view_533 = torch.ops.aten.view.default(mm_302, [2, 128, 1024]);  mm_302 = None
        permute_708 = torch.ops.aten.permute.default(permute_706, [1, 0]);  permute_706 = None
        _to_copy_75 = torch.ops.aten._to_copy.default(gt_40, dtype = torch.float32);  gt_40 = None
        mul_742 = torch.ops.aten.mul.Tensor(_to_copy_75, 1.1111111111111112);  _to_copy_75 = None
        mul_743 = torch.ops.aten.mul.Tensor(view_533, mul_742);  view_533 = mul_742 = None
        mul_744 = torch.ops.aten.mul.Tensor(mul_743, mul_186);  mul_186 = None
        mul_745 = torch.ops.aten.mul.Tensor(mul_743, _unsafe_view_115);  mul_743 = _unsafe_view_115 = None
        view_534 = torch.ops.aten.view.default(mul_744, [256, 1024]);  mul_744 = None
        permute_709 = torch.ops.aten.permute.default(view_534, [1, 0])
        mm_303 = torch.ops.aten.mm.default(permute_709, view_116);  permute_709 = None
        permute_710 = torch.ops.aten.permute.default(mm_303, [1, 0]);  mm_303 = None
        mm_304 = torch.ops.aten.mm.default(view_534, permute_711);  view_534 = permute_711 = None
        view_535 = torch.ops.aten.view.default(mm_304, [2, 128, 512]);  mm_304 = None
        permute_712 = torch.ops.aten.permute.default(permute_710, [1, 0]);  permute_710 = None
        mul_746 = torch.ops.aten.mul.Tensor(mul_745, mul_181);  mul_181 = None
        mul_747 = torch.ops.aten.mul.Tensor(mul_745, add_82);  mul_745 = add_82 = None
        alias_423 = torch.ops.aten.alias.default(alias_169);  alias_169 = None
        alias_424 = torch.ops.aten.alias.default(alias_423);  alias_423 = None
        mul_748 = torch.ops.aten.mul.Tensor(alias_424, alias_424);  alias_424 = None
        _tensor_constant10 = self._tensor_constant10
        lift_fresh_copy_10 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant10);  _tensor_constant10 = None
        sub_69 = torch.ops.aten.sub.Tensor(lift_fresh_copy_10, mul_748);  lift_fresh_copy_10 = mul_748 = None
        mul_749 = torch.ops.aten.mul.Tensor(mul_746, sub_69);  mul_746 = sub_69 = None
        mul_750 = torch.ops.aten.mul.Tensor(mul_749, 0.7978845608028654);  mul_749 = None
        mul_751 = torch.ops.aten.mul.Tensor(mul_750, 0.044715)
        pow_110 = torch.ops.aten.pow.Tensor_Scalar(_unsafe_view_114, 2.0);  _unsafe_view_114 = None
        mul_752 = torch.ops.aten.mul.Scalar(pow_110, 3.0);  pow_110 = None
        mul_753 = torch.ops.aten.mul.Tensor(mul_751, mul_752);  mul_751 = mul_752 = None
        add_289 = torch.ops.aten.add.Tensor(mul_750, mul_753);  mul_750 = mul_753 = None
        mul_754 = torch.ops.aten.mul.Tensor(mul_747, 0.5);  mul_747 = None
        add_290 = torch.ops.aten.add.Tensor(add_289, mul_754);  add_289 = mul_754 = None
        view_536 = torch.ops.aten.view.default(add_290, [256, 1024]);  add_290 = None
        permute_713 = torch.ops.aten.permute.default(view_536, [1, 0])
        mm_305 = torch.ops.aten.mm.default(permute_713, view_116);  permute_713 = view_116 = None
        permute_714 = torch.ops.aten.permute.default(mm_305, [1, 0]);  mm_305 = None
        mm_306 = torch.ops.aten.mm.default(view_536, permute_715);  view_536 = permute_715 = None
        view_537 = torch.ops.aten.view.default(mm_306, [2, 128, 512]);  mm_306 = None
        add_291 = torch.ops.aten.add.Tensor(view_535, view_537);  view_535 = view_537 = None
        permute_716 = torch.ops.aten.permute.default(permute_714, [1, 0]);  permute_714 = None
        mul_755 = torch.ops.aten.mul.Tensor(add_291, primals_20);  primals_20 = None
        mul_756 = torch.ops.aten.mul.Tensor(add_291, mul_179);  add_291 = mul_179 = None
        sum_85 = torch.ops.aten.sum.dim_IntList(mul_756, [0, 1], True);  mul_756 = None
        view_538 = torch.ops.aten.view.default(sum_85, [512]);  sum_85 = None
        mul_757 = torch.ops.aten.mul.Tensor(mul_755, add_78)
        mul_758 = torch.ops.aten.mul.Tensor(mul_755, reciprocal_27);  mul_755 = reciprocal_27 = None
        sum_86 = torch.ops.aten.sum.dim_IntList(mul_757, [2], True);  mul_757 = None
        add_292 = torch.ops.aten.add.Tensor(add_288, mul_758);  add_288 = mul_758 = None
        alias_425 = torch.ops.aten.alias.default(alias_166);  alias_166 = None
        alias_426 = torch.ops.aten.alias.default(alias_425);  alias_425 = None
        pow_111 = torch.ops.aten.pow.Tensor_Scalar(alias_426, 3);  alias_426 = None
        mul_759 = torch.ops.aten.mul.Scalar(sum_86, -0.5);  sum_86 = None
        mul_760 = torch.ops.aten.mul.Tensor(mul_759, pow_111);  mul_759 = pow_111 = None
        expand_118 = torch.ops.aten.expand.default(mul_760, [2, 128, 512]);  mul_760 = None
        div_51 = torch.ops.aten.div.Scalar(expand_118, 512);  expand_118 = None
        pow_112 = torch.ops.aten.pow.Tensor_Scalar(add_78, 1.0);  add_78 = None
        mul_761 = torch.ops.aten.mul.Scalar(pow_112, 2.0);  pow_112 = None
        mul_762 = torch.ops.aten.mul.Tensor(div_51, mul_761);  div_51 = mul_761 = None
        add_293 = torch.ops.aten.add.Tensor(add_292, mul_762);  add_292 = mul_762 = None
        _to_copy_76 = torch.ops.aten._to_copy.default(gt_39, dtype = torch.float32);  gt_39 = None
        mul_763 = torch.ops.aten.mul.Tensor(_to_copy_76, 1.1111111111111112);  _to_copy_76 = None
        mul_764 = torch.ops.aten.mul.Tensor(add_293, mul_763);  mul_763 = None
        view_539 = torch.ops.aten.view.default(mul_764, [256, 512]);  mul_764 = None
        permute_717 = torch.ops.aten.permute.default(view_539, [1, 0])
        mm_307 = torch.ops.aten.mm.default(permute_717, view_115);  permute_717 = view_115 = None
        permute_718 = torch.ops.aten.permute.default(mm_307, [1, 0]);  mm_307 = None
        mm_308 = torch.ops.aten.mm.default(view_539, permute_719);  view_539 = permute_719 = None
        view_540 = torch.ops.aten.view.default(mm_308, [2, 128, 384]);  mm_308 = None
        permute_720 = torch.ops.aten.permute.default(permute_718, [1, 0]);  permute_718 = None
        view_541 = torch.ops.aten.view.default(view_540, [2, 128, 6, 64]);  view_540 = None
        permute_721 = torch.ops.aten.permute.default(view_541, [0, 2, 1, 3]);  view_541 = None
        clone_152 = torch.ops.aten.clone.default(permute_721, memory_format = torch.contiguous_format);  permute_721 = None
        _unsafe_view_321 = torch.ops.aten._unsafe_view.default(clone_152, [12, 128, 64]);  clone_152 = None
        bmm_104 = torch.ops.aten.bmm.default(permute_722, _unsafe_view_321);  permute_722 = None
        bmm_105 = torch.ops.aten.bmm.default(_unsafe_view_321, permute_723);  _unsafe_view_321 = permute_723 = None
        view_542 = torch.ops.aten.view.default(bmm_104, [2, 6, 128, 64]);  bmm_104 = None
        add_294 = torch.ops.aten.add.Tensor(tangents_6, view_542);  tangents_6 = view_542 = None
        view_543 = torch.ops.aten.view.default(bmm_105, [2, 6, 128, 128]);  bmm_105 = None
        philox_rand_like_38 = torch.ops.prims.philox_rand_like.default(view_543, philox_seed_like, 1769472)
        gt_99 = torch.ops.aten.gt.Scalar(philox_rand_like_38, 0.1);  philox_rand_like_38 = None
        _to_copy_77 = torch.ops.aten._to_copy.default(gt_99, dtype = torch.float32);  gt_99 = None
        mul_765 = torch.ops.aten.mul.Tensor(_to_copy_77, view_543);  _to_copy_77 = view_543 = None
        mul_766 = torch.ops.aten.mul.Tensor(mul_765, 1.1111111111111112);  mul_765 = None
        alias_427 = torch.ops.aten.alias.default(alias_162);  alias_162 = None
        alias_428 = torch.ops.aten.alias.default(alias_427);  alias_427 = None
        mul_767 = torch.ops.aten.mul.Tensor(mul_766, alias_428);  mul_766 = None
        sum_87 = torch.ops.aten.sum.dim_IntList(mul_767, [-1], True)
        mul_768 = torch.ops.aten.mul.Tensor(alias_428, sum_87);  alias_428 = sum_87 = None
        sub_70 = torch.ops.aten.sub.Tensor(mul_767, mul_768);  mul_767 = mul_768 = None
        view_544 = torch.ops.aten.view.default(sub_70, [12, 128, 128]);  sub_70 = None
        bmm_106 = torch.ops.aten.bmm.default(permute_724, view_544);  permute_724 = None
        bmm_107 = torch.ops.aten.bmm.default(view_544, permute_725);  view_544 = permute_725 = None
        view_545 = torch.ops.aten.view.default(bmm_106, [2, 6, 64, 128]);  bmm_106 = None
        view_546 = torch.ops.aten.view.default(bmm_107, [2, 6, 128, 64]);  bmm_107 = None
        permute_726 = torch.ops.aten.permute.default(view_545, [0, 1, 3, 2]);  view_545 = None
        add_295 = torch.ops.aten.add.Tensor(tangents_5, permute_726);  tangents_5 = permute_726 = None
        permute_727 = torch.ops.aten.permute.default(add_294, [0, 2, 1, 3]);  add_294 = None
        clone_153 = torch.ops.aten.clone.default(permute_727, memory_format = torch.contiguous_format);  permute_727 = None
        _unsafe_view_322 = torch.ops.aten._unsafe_view.default(clone_153, [2, 128, 384]);  clone_153 = None
        view_547 = torch.ops.aten.view.default(_unsafe_view_322, [256, 384]);  _unsafe_view_322 = None
        permute_728 = torch.ops.aten.permute.default(view_547, [1, 0])
        mm_309 = torch.ops.aten.mm.default(permute_728, view_109);  permute_728 = None
        permute_729 = torch.ops.aten.permute.default(mm_309, [1, 0]);  mm_309 = None
        mm_310 = torch.ops.aten.mm.default(view_547, permute_730);  view_547 = permute_730 = None
        view_548 = torch.ops.aten.view.default(mm_310, [2, 128, 512]);  mm_310 = None
        add_296 = torch.ops.aten.add.Tensor(add_279, view_548);  add_279 = view_548 = None
        permute_731 = torch.ops.aten.permute.default(permute_729, [1, 0]);  permute_729 = None
        permute_732 = torch.ops.aten.permute.default(add_295, [0, 2, 1, 3]);  add_295 = None
        clone_154 = torch.ops.aten.clone.default(permute_732, memory_format = torch.contiguous_format);  permute_732 = None
        _unsafe_view_323 = torch.ops.aten._unsafe_view.default(clone_154, [2, 128, 384]);  clone_154 = None
        view_549 = torch.ops.aten.view.default(_unsafe_view_323, [256, 384]);  _unsafe_view_323 = None
        permute_733 = torch.ops.aten.permute.default(view_549, [1, 0])
        mm_311 = torch.ops.aten.mm.default(permute_733, view_109);  permute_733 = view_109 = None
        permute_734 = torch.ops.aten.permute.default(mm_311, [1, 0]);  mm_311 = None
        mm_312 = torch.ops.aten.mm.default(view_549, permute_735);  view_549 = permute_735 = None
        view_550 = torch.ops.aten.view.default(mm_312, [2, 128, 512]);  mm_312 = None
        add_297 = torch.ops.aten.add.Tensor(add_296, view_550);  add_296 = view_550 = None
        permute_736 = torch.ops.aten.permute.default(permute_734, [1, 0]);  permute_734 = None
        permute_737 = torch.ops.aten.permute.default(view_546, [0, 2, 1, 3]);  view_546 = None
        clone_155 = torch.ops.aten.clone.default(permute_737, memory_format = torch.contiguous_format);  permute_737 = None
        _unsafe_view_324 = torch.ops.aten._unsafe_view.default(clone_155, [2, 128, 384]);  clone_155 = None
        view_551 = torch.ops.aten.view.default(_unsafe_view_324, [256, 384]);  _unsafe_view_324 = None
        permute_738 = torch.ops.aten.permute.default(view_551, [1, 0])
        mm_313 = torch.ops.aten.mm.default(permute_738, view_107);  permute_738 = view_107 = None
        permute_739 = torch.ops.aten.permute.default(mm_313, [1, 0]);  mm_313 = None
        mm_314 = torch.ops.aten.mm.default(view_551, permute_740);  view_551 = permute_740 = None
        view_552 = torch.ops.aten.view.default(mm_314, [2, 128, 512]);  mm_314 = None
        permute_741 = torch.ops.aten.permute.default(permute_739, [1, 0]);  permute_739 = None
        mul_769 = torch.ops.aten.mul.Tensor(view_552, primals_19);  primals_19 = None
        mul_770 = torch.ops.aten.mul.Tensor(view_552, mul_173);  view_552 = mul_173 = None
        sum_88 = torch.ops.aten.sum.dim_IntList(mul_770, [0, 1], True);  mul_770 = None
        view_553 = torch.ops.aten.view.default(sum_88, [512]);  sum_88 = None
        mul_771 = torch.ops.aten.mul.Tensor(mul_769, add_74)
        mul_772 = torch.ops.aten.mul.Tensor(mul_769, reciprocal_26);  mul_769 = reciprocal_26 = None
        sum_89 = torch.ops.aten.sum.dim_IntList(mul_771, [2], True);  mul_771 = None
        add_298 = torch.ops.aten.add.Tensor(add_293, mul_772);  add_293 = mul_772 = None
        alias_429 = torch.ops.aten.alias.default(alias_157);  alias_157 = None
        alias_430 = torch.ops.aten.alias.default(alias_429);  alias_429 = None
        pow_113 = torch.ops.aten.pow.Tensor_Scalar(alias_430, 3);  alias_430 = None
        mul_773 = torch.ops.aten.mul.Scalar(sum_89, -0.5);  sum_89 = None
        mul_774 = torch.ops.aten.mul.Tensor(mul_773, pow_113);  mul_773 = pow_113 = None
        expand_119 = torch.ops.aten.expand.default(mul_774, [2, 128, 512]);  mul_774 = None
        div_52 = torch.ops.aten.div.Scalar(expand_119, 512);  expand_119 = None
        pow_114 = torch.ops.aten.pow.Tensor_Scalar(add_74, 1.0);  add_74 = None
        mul_775 = torch.ops.aten.mul.Scalar(pow_114, 2.0);  pow_114 = None
        mul_776 = torch.ops.aten.mul.Tensor(div_52, mul_775);  div_52 = mul_775 = None
        add_299 = torch.ops.aten.add.Tensor(add_298, mul_776);  add_298 = mul_776 = None
        _to_copy_78 = torch.ops.aten._to_copy.default(gt_37, dtype = torch.float32);  gt_37 = None
        mul_777 = torch.ops.aten.mul.Tensor(_to_copy_78, 1.1111111111111112);  _to_copy_78 = None
        mul_778 = torch.ops.aten.mul.Tensor(add_299, mul_777);  mul_777 = None
        view_554 = torch.ops.aten.view.default(mul_778, [256, 512]);  mul_778 = None
        permute_742 = torch.ops.aten.permute.default(view_554, [1, 0])
        mm_315 = torch.ops.aten.mm.default(permute_742, view_106);  permute_742 = view_106 = None
        permute_743 = torch.ops.aten.permute.default(mm_315, [1, 0]);  mm_315 = None
        mm_316 = torch.ops.aten.mm.default(view_554, permute_744);  view_554 = permute_744 = None
        view_555 = torch.ops.aten.view.default(mm_316, [2, 128, 384]);  mm_316 = None
        permute_745 = torch.ops.aten.permute.default(permute_743, [1, 0]);  permute_743 = None
        view_556 = torch.ops.aten.view.default(view_555, [2, 128, 6, 64]);  view_555 = None
        permute_746 = torch.ops.aten.permute.default(view_556, [0, 2, 1, 3]);  view_556 = None
        clone_156 = torch.ops.aten.clone.default(permute_746, memory_format = torch.contiguous_format);  permute_746 = None
        _unsafe_view_325 = torch.ops.aten._unsafe_view.default(clone_156, [12, 128, 64]);  clone_156 = None
        bmm_108 = torch.ops.aten.bmm.default(permute_747, _unsafe_view_325);  permute_747 = None
        bmm_109 = torch.ops.aten.bmm.default(_unsafe_view_325, permute_748);  _unsafe_view_325 = permute_748 = None
        view_557 = torch.ops.aten.view.default(bmm_108, [2, 6, 128, 64]);  bmm_108 = None
        add_300 = torch.ops.aten.add.Tensor(tangents_4, view_557);  tangents_4 = view_557 = None
        view_558 = torch.ops.aten.view.default(bmm_109, [2, 6, 128, 128]);  bmm_109 = None
        philox_rand_like_39 = torch.ops.prims.philox_rand_like.default(view_558, philox_seed_like, 1572864)
        gt_100 = torch.ops.aten.gt.Scalar(philox_rand_like_39, 0.1);  philox_rand_like_39 = None
        _to_copy_79 = torch.ops.aten._to_copy.default(gt_100, dtype = torch.float32);  gt_100 = None
        mul_779 = torch.ops.aten.mul.Tensor(_to_copy_79, view_558);  _to_copy_79 = view_558 = None
        mul_780 = torch.ops.aten.mul.Tensor(mul_779, 1.1111111111111112);  mul_779 = None
        alias_431 = torch.ops.aten.alias.default(alias_153);  alias_153 = None
        alias_432 = torch.ops.aten.alias.default(alias_431);  alias_431 = None
        mul_781 = torch.ops.aten.mul.Tensor(mul_780, alias_432);  mul_780 = None
        sum_90 = torch.ops.aten.sum.dim_IntList(mul_781, [-1], True)
        mul_782 = torch.ops.aten.mul.Tensor(alias_432, sum_90);  alias_432 = sum_90 = None
        sub_71 = torch.ops.aten.sub.Tensor(mul_781, mul_782);  mul_781 = mul_782 = None
        add_301 = torch.ops.aten.add.Tensor(add_283, sub_71);  add_283 = None
        sum_91 = torch.ops.aten.sum.dim_IntList(add_301, [0], True);  add_301 = None
        squeeze_1 = torch.ops.aten.squeeze.dim(sum_91, 0);  sum_91 = None
        permute_749 = torch.ops.aten.permute.default(squeeze_1, [1, 2, 0]);  squeeze_1 = None
        view_559 = torch.ops.aten.view.default(permute_749, [16384, 6])
        new_zeros = torch.ops.aten.new_zeros.default(permute_749, [32, 6], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  permute_749 = None
        ne = torch.ops.aten.ne.Scalar(view_560, -1)
        unsqueeze_19 = torch.ops.aten.unsqueeze.default(ne, 1);  ne = None
        expand_120 = torch.ops.aten.expand.default(unsqueeze_19, [16384, 6]);  unsqueeze_19 = None
        full_like_2 = torch.ops.aten.full_like.default(view_559, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_433 = torch.ops.aten.alias.default(full_like_2);  full_like_2 = None
        where_2 = torch.ops.aten.where.self(expand_120, view_559, alias_433);  expand_120 = view_559 = alias_433 = None
        index_put = torch.ops.aten.index_put.default(new_zeros, [view_560], where_2, True);  new_zeros = view_560 = where_2 = None
        view_561 = torch.ops.aten.view.default(sub_71, [12, 128, 128]);  sub_71 = None
        bmm_110 = torch.ops.aten.bmm.default(permute_750, view_561);  permute_750 = None
        bmm_111 = torch.ops.aten.bmm.default(view_561, permute_751);  view_561 = permute_751 = None
        view_562 = torch.ops.aten.view.default(bmm_110, [2, 6, 64, 128]);  bmm_110 = None
        view_563 = torch.ops.aten.view.default(bmm_111, [2, 6, 128, 64]);  bmm_111 = None
        permute_752 = torch.ops.aten.permute.default(view_562, [0, 1, 3, 2]);  view_562 = None
        add_302 = torch.ops.aten.add.Tensor(tangents_3, permute_752);  tangents_3 = permute_752 = None
        permute_753 = torch.ops.aten.permute.default(add_300, [0, 2, 1, 3]);  add_300 = None
        clone_157 = torch.ops.aten.clone.default(permute_753, memory_format = torch.contiguous_format);  permute_753 = None
        _unsafe_view_326 = torch.ops.aten._unsafe_view.default(clone_157, [2, 128, 384]);  clone_157 = None
        view_564 = torch.ops.aten.view.default(_unsafe_view_326, [256, 384]);  _unsafe_view_326 = None
        permute_754 = torch.ops.aten.permute.default(view_564, [1, 0])
        mm_317 = torch.ops.aten.mm.default(permute_754, view_98);  permute_754 = None
        permute_755 = torch.ops.aten.permute.default(mm_317, [1, 0]);  mm_317 = None
        mm_318 = torch.ops.aten.mm.default(view_564, permute_756);  view_564 = permute_756 = None
        view_565 = torch.ops.aten.view.default(mm_318, [2, 128, 512]);  mm_318 = None
        permute_757 = torch.ops.aten.permute.default(permute_755, [1, 0]);  permute_755 = None
        permute_758 = torch.ops.aten.permute.default(add_302, [0, 2, 1, 3]);  add_302 = None
        clone_158 = torch.ops.aten.clone.default(permute_758, memory_format = torch.contiguous_format);  permute_758 = None
        _unsafe_view_327 = torch.ops.aten._unsafe_view.default(clone_158, [2, 128, 384]);  clone_158 = None
        view_566 = torch.ops.aten.view.default(_unsafe_view_327, [256, 384]);  _unsafe_view_327 = None
        permute_759 = torch.ops.aten.permute.default(view_566, [1, 0])
        mm_319 = torch.ops.aten.mm.default(permute_759, view_98);  permute_759 = None
        permute_760 = torch.ops.aten.permute.default(mm_319, [1, 0]);  mm_319 = None
        mm_320 = torch.ops.aten.mm.default(view_566, permute_761);  view_566 = permute_761 = None
        view_567 = torch.ops.aten.view.default(mm_320, [2, 128, 512]);  mm_320 = None
        add_303 = torch.ops.aten.add.Tensor(view_565, view_567);  view_565 = view_567 = None
        permute_762 = torch.ops.aten.permute.default(permute_760, [1, 0]);  permute_760 = None
        permute_763 = torch.ops.aten.permute.default(view_563, [0, 2, 1, 3]);  view_563 = None
        clone_159 = torch.ops.aten.clone.default(permute_763, memory_format = torch.contiguous_format);  permute_763 = None
        _unsafe_view_328 = torch.ops.aten._unsafe_view.default(clone_159, [2, 128, 384]);  clone_159 = None
        view_568 = torch.ops.aten.view.default(_unsafe_view_328, [256, 384]);  _unsafe_view_328 = None
        permute_764 = torch.ops.aten.permute.default(view_568, [1, 0])
        mm_321 = torch.ops.aten.mm.default(permute_764, view_98);  permute_764 = view_98 = None
        permute_765 = torch.ops.aten.permute.default(mm_321, [1, 0]);  mm_321 = None
        mm_322 = torch.ops.aten.mm.default(view_568, permute_766);  view_568 = permute_766 = None
        view_569 = torch.ops.aten.view.default(mm_322, [2, 128, 512]);  mm_322 = None
        add_304 = torch.ops.aten.add.Tensor(add_303, view_569);  add_303 = view_569 = None
        permute_767 = torch.ops.aten.permute.default(permute_765, [1, 0]);  permute_765 = None
        mul_783 = torch.ops.aten.mul.Tensor(add_304, primals_18);  primals_18 = None
        mul_784 = torch.ops.aten.mul.Tensor(add_304, mul_166);  add_304 = mul_166 = None
        sum_92 = torch.ops.aten.sum.dim_IntList(mul_784, [0, 1], True);  mul_784 = None
        view_570 = torch.ops.aten.view.default(sum_92, [512]);  sum_92 = None
        mul_785 = torch.ops.aten.mul.Tensor(mul_783, mul_165)
        mul_786 = torch.ops.aten.mul.Tensor(mul_783, reciprocal_25);  mul_783 = reciprocal_25 = None
        sum_93 = torch.ops.aten.sum.dim_IntList(mul_785, [2], True);  mul_785 = None
        add_305 = torch.ops.aten.add.Tensor(add_299, mul_786);  add_299 = mul_786 = None
        alias_434 = torch.ops.aten.alias.default(alias_142);  alias_142 = None
        alias_435 = torch.ops.aten.alias.default(alias_434);  alias_434 = None
        pow_115 = torch.ops.aten.pow.Tensor_Scalar(alias_435, 3);  alias_435 = None
        mul_787 = torch.ops.aten.mul.Scalar(sum_93, -0.5);  sum_93 = None
        mul_788 = torch.ops.aten.mul.Tensor(mul_787, pow_115);  mul_787 = pow_115 = None
        expand_121 = torch.ops.aten.expand.default(mul_788, [2, 128, 512]);  mul_788 = None
        div_53 = torch.ops.aten.div.Scalar(expand_121, 512);  expand_121 = None
        pow_116 = torch.ops.aten.pow.Tensor_Scalar(mul_165, 1.0);  mul_165 = None
        mul_789 = torch.ops.aten.mul.Scalar(pow_116, 2.0);  pow_116 = None
        mul_790 = torch.ops.aten.mul.Tensor(div_53, mul_789);  div_53 = mul_789 = None
        add_306 = torch.ops.aten.add.Tensor(add_305, mul_790);  add_305 = mul_790 = None
        _to_copy_80 = torch.ops.aten._to_copy.default(gt_35, dtype = torch.float32);  gt_35 = None
        mul_791 = torch.ops.aten.mul.Tensor(_to_copy_80, 1.1111111111111112);  _to_copy_80 = None
        mul_792 = torch.ops.aten.mul.Tensor(add_306, mul_791);  add_306 = mul_791 = None
        view_571 = torch.ops.aten.view.default(mul_792, [256, 512])
        new_zeros_1 = torch.ops.aten.new_zeros.default(mul_792, [250112, 512], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  mul_792 = None
        ne_1 = torch.ops.aten.ne.Scalar(view_572, -1)
        unsqueeze_20 = torch.ops.aten.unsqueeze.default(ne_1, 1);  ne_1 = None
        expand_122 = torch.ops.aten.expand.default(unsqueeze_20, [256, 512]);  unsqueeze_20 = None
        full_like_3 = torch.ops.aten.full_like.default(view_571, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_436 = torch.ops.aten.alias.default(full_like_3);  full_like_3 = None
        where_3 = torch.ops.aten.where.self(expand_122, view_571, alias_436);  expand_122 = view_571 = alias_436 = None
        index_put_1 = torch.ops.aten.index_put.default(new_zeros_1, [view_572], where_3, True);  new_zeros_1 = view_572 = where_3 = None
        _to_copy_81 = torch.ops.aten._to_copy.default(gt_34, dtype = torch.float32);  gt_34 = None
        mul_793 = torch.ops.aten.mul.Tensor(_to_copy_81, 1.1111111111111112);  _to_copy_81 = None
        mul_794 = torch.ops.aten.mul.Tensor(add_297, mul_793);  add_297 = mul_793 = None
        mul_795 = torch.ops.aten.mul.Tensor(mul_794, primals_17);  primals_17 = None
        mul_796 = torch.ops.aten.mul.Tensor(mul_794, mul_157);  mul_794 = mul_157 = None
        sum_94 = torch.ops.aten.sum.dim_IntList(mul_796, [0, 1], True);  mul_796 = None
        view_573 = torch.ops.aten.view.default(sum_94, [512]);  sum_94 = None
        mul_797 = torch.ops.aten.mul.Tensor(mul_795, add_67)
        mul_798 = torch.ops.aten.mul.Tensor(mul_795, reciprocal_24);  mul_795 = reciprocal_24 = None
        sum_95 = torch.ops.aten.sum.dim_IntList(mul_797, [2], True);  mul_797 = None
        alias_437 = torch.ops.aten.alias.default(alias_131);  alias_131 = None
        alias_438 = torch.ops.aten.alias.default(alias_437);  alias_437 = None
        pow_117 = torch.ops.aten.pow.Tensor_Scalar(alias_438, 3);  alias_438 = None
        mul_799 = torch.ops.aten.mul.Scalar(sum_95, -0.5);  sum_95 = None
        mul_800 = torch.ops.aten.mul.Tensor(mul_799, pow_117);  mul_799 = pow_117 = None
        expand_123 = torch.ops.aten.expand.default(mul_800, [2, 128, 512]);  mul_800 = None
        div_54 = torch.ops.aten.div.Scalar(expand_123, 512);  expand_123 = None
        pow_118 = torch.ops.aten.pow.Tensor_Scalar(add_67, 1.0);  add_67 = None
        mul_801 = torch.ops.aten.mul.Scalar(pow_118, 2.0);  pow_118 = None
        mul_802 = torch.ops.aten.mul.Tensor(div_54, mul_801);  div_54 = mul_801 = None
        add_307 = torch.ops.aten.add.Tensor(mul_798, mul_802);  mul_798 = mul_802 = None
        _to_copy_82 = torch.ops.aten._to_copy.default(gt_33, dtype = torch.float32);  gt_33 = None
        mul_803 = torch.ops.aten.mul.Tensor(_to_copy_82, 1.1111111111111112);  _to_copy_82 = None
        mul_804 = torch.ops.aten.mul.Tensor(add_307, mul_803);  mul_803 = None
        view_574 = torch.ops.aten.view.default(mul_804, [256, 512]);  mul_804 = None
        permute_768 = torch.ops.aten.permute.default(view_574, [1, 0])
        mm_323 = torch.ops.aten.mm.default(permute_768, view_96);  permute_768 = view_96 = None
        permute_769 = torch.ops.aten.permute.default(mm_323, [1, 0]);  mm_323 = None
        mm_324 = torch.ops.aten.mm.default(view_574, permute_770);  view_574 = permute_770 = None
        view_575 = torch.ops.aten.view.default(mm_324, [2, 128, 1024]);  mm_324 = None
        permute_771 = torch.ops.aten.permute.default(permute_769, [1, 0]);  permute_769 = None
        _to_copy_83 = torch.ops.aten._to_copy.default(gt_32, dtype = torch.float32);  gt_32 = None
        mul_805 = torch.ops.aten.mul.Tensor(_to_copy_83, 1.1111111111111112);  _to_copy_83 = None
        mul_806 = torch.ops.aten.mul.Tensor(view_575, mul_805);  view_575 = mul_805 = None
        mul_807 = torch.ops.aten.mul.Tensor(mul_806, mul_151);  mul_151 = None
        mul_808 = torch.ops.aten.mul.Tensor(mul_806, _unsafe_view_94);  mul_806 = _unsafe_view_94 = None
        view_576 = torch.ops.aten.view.default(mul_807, [256, 1024]);  mul_807 = None
        permute_772 = torch.ops.aten.permute.default(view_576, [1, 0])
        mm_325 = torch.ops.aten.mm.default(permute_772, view_94);  permute_772 = None
        permute_773 = torch.ops.aten.permute.default(mm_325, [1, 0]);  mm_325 = None
        mm_326 = torch.ops.aten.mm.default(view_576, permute_774);  view_576 = permute_774 = None
        view_577 = torch.ops.aten.view.default(mm_326, [2, 128, 512]);  mm_326 = None
        permute_775 = torch.ops.aten.permute.default(permute_773, [1, 0]);  permute_773 = None
        mul_809 = torch.ops.aten.mul.Tensor(mul_808, mul_146);  mul_146 = None
        mul_810 = torch.ops.aten.mul.Tensor(mul_808, add_66);  mul_808 = add_66 = None
        alias_439 = torch.ops.aten.alias.default(alias_126);  alias_126 = None
        alias_440 = torch.ops.aten.alias.default(alias_439);  alias_439 = None
        mul_811 = torch.ops.aten.mul.Tensor(alias_440, alias_440);  alias_440 = None
        _tensor_constant11 = self._tensor_constant11
        lift_fresh_copy_11 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant11);  _tensor_constant11 = None
        sub_72 = torch.ops.aten.sub.Tensor(lift_fresh_copy_11, mul_811);  lift_fresh_copy_11 = mul_811 = None
        mul_812 = torch.ops.aten.mul.Tensor(mul_809, sub_72);  mul_809 = sub_72 = None
        mul_813 = torch.ops.aten.mul.Tensor(mul_812, 0.7978845608028654);  mul_812 = None
        mul_814 = torch.ops.aten.mul.Tensor(mul_813, 0.044715)
        pow_119 = torch.ops.aten.pow.Tensor_Scalar(_unsafe_view_93, 2.0);  _unsafe_view_93 = None
        mul_815 = torch.ops.aten.mul.Scalar(pow_119, 3.0);  pow_119 = None
        mul_816 = torch.ops.aten.mul.Tensor(mul_814, mul_815);  mul_814 = mul_815 = None
        add_308 = torch.ops.aten.add.Tensor(mul_813, mul_816);  mul_813 = mul_816 = None
        mul_817 = torch.ops.aten.mul.Tensor(mul_810, 0.5);  mul_810 = None
        add_309 = torch.ops.aten.add.Tensor(add_308, mul_817);  add_308 = mul_817 = None
        view_578 = torch.ops.aten.view.default(add_309, [256, 1024]);  add_309 = None
        permute_776 = torch.ops.aten.permute.default(view_578, [1, 0])
        mm_327 = torch.ops.aten.mm.default(permute_776, view_94);  permute_776 = view_94 = None
        permute_777 = torch.ops.aten.permute.default(mm_327, [1, 0]);  mm_327 = None
        mm_328 = torch.ops.aten.mm.default(view_578, permute_778);  view_578 = permute_778 = None
        view_579 = torch.ops.aten.view.default(mm_328, [2, 128, 512]);  mm_328 = None
        add_310 = torch.ops.aten.add.Tensor(view_577, view_579);  view_577 = view_579 = None
        permute_779 = torch.ops.aten.permute.default(permute_777, [1, 0]);  permute_777 = None
        mul_818 = torch.ops.aten.mul.Tensor(add_310, primals_16);  primals_16 = None
        mul_819 = torch.ops.aten.mul.Tensor(add_310, mul_144);  add_310 = mul_144 = None
        sum_96 = torch.ops.aten.sum.dim_IntList(mul_819, [0, 1], True);  mul_819 = None
        view_580 = torch.ops.aten.view.default(sum_96, [512]);  sum_96 = None
        mul_820 = torch.ops.aten.mul.Tensor(mul_818, add_62)
        mul_821 = torch.ops.aten.mul.Tensor(mul_818, reciprocal_22);  mul_818 = reciprocal_22 = None
        sum_97 = torch.ops.aten.sum.dim_IntList(mul_820, [2], True);  mul_820 = None
        add_311 = torch.ops.aten.add.Tensor(add_307, mul_821);  add_307 = mul_821 = None
        alias_441 = torch.ops.aten.alias.default(alias_123);  alias_123 = None
        alias_442 = torch.ops.aten.alias.default(alias_441);  alias_441 = None
        pow_120 = torch.ops.aten.pow.Tensor_Scalar(alias_442, 3);  alias_442 = None
        mul_822 = torch.ops.aten.mul.Scalar(sum_97, -0.5);  sum_97 = None
        mul_823 = torch.ops.aten.mul.Tensor(mul_822, pow_120);  mul_822 = pow_120 = None
        expand_124 = torch.ops.aten.expand.default(mul_823, [2, 128, 512]);  mul_823 = None
        div_55 = torch.ops.aten.div.Scalar(expand_124, 512);  expand_124 = None
        pow_121 = torch.ops.aten.pow.Tensor_Scalar(add_62, 1.0);  add_62 = None
        mul_824 = torch.ops.aten.mul.Scalar(pow_121, 2.0);  pow_121 = None
        mul_825 = torch.ops.aten.mul.Tensor(div_55, mul_824);  div_55 = mul_824 = None
        add_312 = torch.ops.aten.add.Tensor(add_311, mul_825);  add_311 = mul_825 = None
        _to_copy_84 = torch.ops.aten._to_copy.default(gt_31, dtype = torch.float32);  gt_31 = None
        mul_826 = torch.ops.aten.mul.Tensor(_to_copy_84, 1.1111111111111112);  _to_copy_84 = None
        mul_827 = torch.ops.aten.mul.Tensor(add_312, mul_826);  mul_826 = None
        view_581 = torch.ops.aten.view.default(mul_827, [256, 512]);  mul_827 = None
        permute_780 = torch.ops.aten.permute.default(view_581, [1, 0])
        mm_329 = torch.ops.aten.mm.default(permute_780, view_93);  permute_780 = view_93 = None
        permute_781 = torch.ops.aten.permute.default(mm_329, [1, 0]);  mm_329 = None
        mm_330 = torch.ops.aten.mm.default(view_581, permute_782);  view_581 = permute_782 = None
        view_582 = torch.ops.aten.view.default(mm_330, [2, 128, 384]);  mm_330 = None
        permute_783 = torch.ops.aten.permute.default(permute_781, [1, 0]);  permute_781 = None
        view_583 = torch.ops.aten.view.default(view_582, [2, 128, 6, 64]);  view_582 = None
        permute_784 = torch.ops.aten.permute.default(view_583, [0, 2, 1, 3]);  view_583 = None
        clone_160 = torch.ops.aten.clone.default(permute_784, memory_format = torch.contiguous_format);  permute_784 = None
        _unsafe_view_329 = torch.ops.aten._unsafe_view.default(clone_160, [12, 128, 64]);  clone_160 = None
        bmm_112 = torch.ops.aten.bmm.default(permute_785, _unsafe_view_329);  permute_785 = None
        bmm_113 = torch.ops.aten.bmm.default(_unsafe_view_329, permute_786);  _unsafe_view_329 = permute_786 = None
        view_584 = torch.ops.aten.view.default(bmm_112, [2, 6, 128, 64]);  bmm_112 = None
        view_585 = torch.ops.aten.view.default(bmm_113, [2, 6, 128, 128]);  bmm_113 = None
        philox_rand_like_40 = torch.ops.prims.philox_rand_like.default(view_585, philox_seed_like, 1376256)
        gt_101 = torch.ops.aten.gt.Scalar(philox_rand_like_40, 0.1);  philox_rand_like_40 = None
        _to_copy_85 = torch.ops.aten._to_copy.default(gt_101, dtype = torch.float32);  gt_101 = None
        mul_828 = torch.ops.aten.mul.Tensor(_to_copy_85, view_585);  _to_copy_85 = view_585 = None
        mul_829 = torch.ops.aten.mul.Tensor(mul_828, 1.1111111111111112);  mul_828 = None
        alias_443 = torch.ops.aten.alias.default(alias_119);  alias_119 = None
        alias_444 = torch.ops.aten.alias.default(alias_443);  alias_443 = None
        mul_830 = torch.ops.aten.mul.Tensor(mul_829, alias_444);  mul_829 = None
        sum_98 = torch.ops.aten.sum.dim_IntList(mul_830, [-1], True)
        mul_831 = torch.ops.aten.mul.Tensor(alias_444, sum_98);  alias_444 = sum_98 = None
        sub_73 = torch.ops.aten.sub.Tensor(mul_830, mul_831);  mul_830 = mul_831 = None
        view_586 = torch.ops.aten.view.default(sub_73, [12, 128, 128])
        bmm_114 = torch.ops.aten.bmm.default(permute_787, view_586);  permute_787 = None
        bmm_115 = torch.ops.aten.bmm.default(view_586, permute_788);  view_586 = permute_788 = None
        view_587 = torch.ops.aten.view.default(bmm_114, [2, 6, 64, 128]);  bmm_114 = None
        view_588 = torch.ops.aten.view.default(bmm_115, [2, 6, 128, 64]);  bmm_115 = None
        permute_789 = torch.ops.aten.permute.default(view_587, [0, 1, 3, 2]);  view_587 = None
        permute_790 = torch.ops.aten.permute.default(view_584, [0, 2, 1, 3]);  view_584 = None
        clone_161 = torch.ops.aten.clone.default(permute_790, memory_format = torch.contiguous_format);  permute_790 = None
        _unsafe_view_330 = torch.ops.aten._unsafe_view.default(clone_161, [2, 128, 384]);  clone_161 = None
        view_589 = torch.ops.aten.view.default(_unsafe_view_330, [256, 384]);  _unsafe_view_330 = None
        permute_791 = torch.ops.aten.permute.default(view_589, [1, 0])
        mm_331 = torch.ops.aten.mm.default(permute_791, view_85);  permute_791 = None
        permute_792 = torch.ops.aten.permute.default(mm_331, [1, 0]);  mm_331 = None
        mm_332 = torch.ops.aten.mm.default(view_589, permute_793);  view_589 = permute_793 = None
        view_590 = torch.ops.aten.view.default(mm_332, [2, 128, 512]);  mm_332 = None
        permute_794 = torch.ops.aten.permute.default(permute_792, [1, 0]);  permute_792 = None
        permute_795 = torch.ops.aten.permute.default(permute_789, [0, 2, 1, 3]);  permute_789 = None
        view_591 = torch.ops.aten.view.default(permute_795, [2, 128, 384]);  permute_795 = None
        clone_162 = torch.ops.aten.clone.default(view_591, memory_format = torch.contiguous_format);  view_591 = None
        _unsafe_view_331 = torch.ops.aten._unsafe_view.default(clone_162, [256, 384]);  clone_162 = None
        permute_796 = torch.ops.aten.permute.default(_unsafe_view_331, [1, 0])
        mm_333 = torch.ops.aten.mm.default(permute_796, view_85);  permute_796 = None
        permute_797 = torch.ops.aten.permute.default(mm_333, [1, 0]);  mm_333 = None
        mm_334 = torch.ops.aten.mm.default(_unsafe_view_331, permute_798);  _unsafe_view_331 = permute_798 = None
        view_592 = torch.ops.aten.view.default(mm_334, [2, 128, 512]);  mm_334 = None
        add_313 = torch.ops.aten.add.Tensor(view_590, view_592);  view_590 = view_592 = None
        permute_799 = torch.ops.aten.permute.default(permute_797, [1, 0]);  permute_797 = None
        permute_800 = torch.ops.aten.permute.default(view_588, [0, 2, 1, 3]);  view_588 = None
        clone_163 = torch.ops.aten.clone.default(permute_800, memory_format = torch.contiguous_format);  permute_800 = None
        _unsafe_view_332 = torch.ops.aten._unsafe_view.default(clone_163, [2, 128, 384]);  clone_163 = None
        view_593 = torch.ops.aten.view.default(_unsafe_view_332, [256, 384]);  _unsafe_view_332 = None
        permute_801 = torch.ops.aten.permute.default(view_593, [1, 0])
        mm_335 = torch.ops.aten.mm.default(permute_801, view_85);  permute_801 = view_85 = None
        permute_802 = torch.ops.aten.permute.default(mm_335, [1, 0]);  mm_335 = None
        mm_336 = torch.ops.aten.mm.default(view_593, permute_803);  view_593 = permute_803 = None
        view_594 = torch.ops.aten.view.default(mm_336, [2, 128, 512]);  mm_336 = None
        add_314 = torch.ops.aten.add.Tensor(add_313, view_594);  add_313 = view_594 = None
        permute_804 = torch.ops.aten.permute.default(permute_802, [1, 0]);  permute_802 = None
        mul_832 = torch.ops.aten.mul.Tensor(add_314, primals_15);  primals_15 = None
        mul_833 = torch.ops.aten.mul.Tensor(add_314, mul_138);  add_314 = mul_138 = None
        sum_99 = torch.ops.aten.sum.dim_IntList(mul_833, [0, 1], True);  mul_833 = None
        view_595 = torch.ops.aten.view.default(sum_99, [512]);  sum_99 = None
        mul_834 = torch.ops.aten.mul.Tensor(mul_832, add_59)
        mul_835 = torch.ops.aten.mul.Tensor(mul_832, reciprocal_21);  mul_832 = reciprocal_21 = None
        sum_100 = torch.ops.aten.sum.dim_IntList(mul_834, [2], True);  mul_834 = None
        add_315 = torch.ops.aten.add.Tensor(add_312, mul_835);  add_312 = mul_835 = None
        alias_445 = torch.ops.aten.alias.default(alias_116);  alias_116 = None
        alias_446 = torch.ops.aten.alias.default(alias_445);  alias_445 = None
        pow_122 = torch.ops.aten.pow.Tensor_Scalar(alias_446, 3);  alias_446 = None
        mul_836 = torch.ops.aten.mul.Scalar(sum_100, -0.5);  sum_100 = None
        mul_837 = torch.ops.aten.mul.Tensor(mul_836, pow_122);  mul_836 = pow_122 = None
        expand_125 = torch.ops.aten.expand.default(mul_837, [2, 128, 512]);  mul_837 = None
        div_56 = torch.ops.aten.div.Scalar(expand_125, 512);  expand_125 = None
        pow_123 = torch.ops.aten.pow.Tensor_Scalar(add_59, 1.0);  add_59 = None
        mul_838 = torch.ops.aten.mul.Scalar(pow_123, 2.0);  pow_123 = None
        mul_839 = torch.ops.aten.mul.Tensor(div_56, mul_838);  div_56 = mul_838 = None
        add_316 = torch.ops.aten.add.Tensor(add_315, mul_839);  add_315 = mul_839 = None
        _to_copy_86 = torch.ops.aten._to_copy.default(gt_29, dtype = torch.float32);  gt_29 = None
        mul_840 = torch.ops.aten.mul.Tensor(_to_copy_86, 1.1111111111111112);  _to_copy_86 = None
        mul_841 = torch.ops.aten.mul.Tensor(add_316, mul_840);  mul_840 = None
        view_596 = torch.ops.aten.view.default(mul_841, [256, 512]);  mul_841 = None
        permute_805 = torch.ops.aten.permute.default(view_596, [1, 0])
        mm_337 = torch.ops.aten.mm.default(permute_805, view_84);  permute_805 = view_84 = None
        permute_806 = torch.ops.aten.permute.default(mm_337, [1, 0]);  mm_337 = None
        mm_338 = torch.ops.aten.mm.default(view_596, permute_807);  view_596 = permute_807 = None
        view_597 = torch.ops.aten.view.default(mm_338, [2, 128, 1024]);  mm_338 = None
        permute_808 = torch.ops.aten.permute.default(permute_806, [1, 0]);  permute_806 = None
        _to_copy_87 = torch.ops.aten._to_copy.default(gt_28, dtype = torch.float32);  gt_28 = None
        mul_842 = torch.ops.aten.mul.Tensor(_to_copy_87, 1.1111111111111112);  _to_copy_87 = None
        mul_843 = torch.ops.aten.mul.Tensor(view_597, mul_842);  view_597 = mul_842 = None
        mul_844 = torch.ops.aten.mul.Tensor(mul_843, mul_132);  mul_132 = None
        mul_845 = torch.ops.aten.mul.Tensor(mul_843, _unsafe_view_82);  mul_843 = _unsafe_view_82 = None
        view_598 = torch.ops.aten.view.default(mul_844, [256, 1024]);  mul_844 = None
        permute_809 = torch.ops.aten.permute.default(view_598, [1, 0])
        mm_339 = torch.ops.aten.mm.default(permute_809, view_82);  permute_809 = None
        permute_810 = torch.ops.aten.permute.default(mm_339, [1, 0]);  mm_339 = None
        mm_340 = torch.ops.aten.mm.default(view_598, permute_811);  view_598 = permute_811 = None
        view_599 = torch.ops.aten.view.default(mm_340, [2, 128, 512]);  mm_340 = None
        permute_812 = torch.ops.aten.permute.default(permute_810, [1, 0]);  permute_810 = None
        mul_846 = torch.ops.aten.mul.Tensor(mul_845, mul_127);  mul_127 = None
        mul_847 = torch.ops.aten.mul.Tensor(mul_845, add_58);  mul_845 = add_58 = None
        alias_447 = torch.ops.aten.alias.default(alias_111);  alias_111 = None
        alias_448 = torch.ops.aten.alias.default(alias_447);  alias_447 = None
        mul_848 = torch.ops.aten.mul.Tensor(alias_448, alias_448);  alias_448 = None
        _tensor_constant12 = self._tensor_constant12
        lift_fresh_copy_12 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant12);  _tensor_constant12 = None
        sub_74 = torch.ops.aten.sub.Tensor(lift_fresh_copy_12, mul_848);  lift_fresh_copy_12 = mul_848 = None
        mul_849 = torch.ops.aten.mul.Tensor(mul_846, sub_74);  mul_846 = sub_74 = None
        mul_850 = torch.ops.aten.mul.Tensor(mul_849, 0.7978845608028654);  mul_849 = None
        mul_851 = torch.ops.aten.mul.Tensor(mul_850, 0.044715)
        pow_124 = torch.ops.aten.pow.Tensor_Scalar(_unsafe_view_81, 2.0);  _unsafe_view_81 = None
        mul_852 = torch.ops.aten.mul.Scalar(pow_124, 3.0);  pow_124 = None
        mul_853 = torch.ops.aten.mul.Tensor(mul_851, mul_852);  mul_851 = mul_852 = None
        add_317 = torch.ops.aten.add.Tensor(mul_850, mul_853);  mul_850 = mul_853 = None
        mul_854 = torch.ops.aten.mul.Tensor(mul_847, 0.5);  mul_847 = None
        add_318 = torch.ops.aten.add.Tensor(add_317, mul_854);  add_317 = mul_854 = None
        view_600 = torch.ops.aten.view.default(add_318, [256, 1024]);  add_318 = None
        permute_813 = torch.ops.aten.permute.default(view_600, [1, 0])
        mm_341 = torch.ops.aten.mm.default(permute_813, view_82);  permute_813 = view_82 = None
        permute_814 = torch.ops.aten.permute.default(mm_341, [1, 0]);  mm_341 = None
        mm_342 = torch.ops.aten.mm.default(view_600, permute_815);  view_600 = permute_815 = None
        view_601 = torch.ops.aten.view.default(mm_342, [2, 128, 512]);  mm_342 = None
        add_319 = torch.ops.aten.add.Tensor(view_599, view_601);  view_599 = view_601 = None
        permute_816 = torch.ops.aten.permute.default(permute_814, [1, 0]);  permute_814 = None
        mul_855 = torch.ops.aten.mul.Tensor(add_319, primals_14);  primals_14 = None
        mul_856 = torch.ops.aten.mul.Tensor(add_319, mul_125);  add_319 = mul_125 = None
        sum_101 = torch.ops.aten.sum.dim_IntList(mul_856, [0, 1], True);  mul_856 = None
        view_602 = torch.ops.aten.view.default(sum_101, [512]);  sum_101 = None
        mul_857 = torch.ops.aten.mul.Tensor(mul_855, add_54)
        mul_858 = torch.ops.aten.mul.Tensor(mul_855, reciprocal_19);  mul_855 = reciprocal_19 = None
        sum_102 = torch.ops.aten.sum.dim_IntList(mul_857, [2], True);  mul_857 = None
        add_320 = torch.ops.aten.add.Tensor(add_316, mul_858);  add_316 = mul_858 = None
        alias_449 = torch.ops.aten.alias.default(alias_108);  alias_108 = None
        alias_450 = torch.ops.aten.alias.default(alias_449);  alias_449 = None
        pow_125 = torch.ops.aten.pow.Tensor_Scalar(alias_450, 3);  alias_450 = None
        mul_859 = torch.ops.aten.mul.Scalar(sum_102, -0.5);  sum_102 = None
        mul_860 = torch.ops.aten.mul.Tensor(mul_859, pow_125);  mul_859 = pow_125 = None
        expand_126 = torch.ops.aten.expand.default(mul_860, [2, 128, 512]);  mul_860 = None
        div_57 = torch.ops.aten.div.Scalar(expand_126, 512);  expand_126 = None
        pow_126 = torch.ops.aten.pow.Tensor_Scalar(add_54, 1.0);  add_54 = None
        mul_861 = torch.ops.aten.mul.Scalar(pow_126, 2.0);  pow_126 = None
        mul_862 = torch.ops.aten.mul.Tensor(div_57, mul_861);  div_57 = mul_861 = None
        add_321 = torch.ops.aten.add.Tensor(add_320, mul_862);  add_320 = mul_862 = None
        _to_copy_88 = torch.ops.aten._to_copy.default(gt_27, dtype = torch.float32);  gt_27 = None
        mul_863 = torch.ops.aten.mul.Tensor(_to_copy_88, 1.1111111111111112);  _to_copy_88 = None
        mul_864 = torch.ops.aten.mul.Tensor(add_321, mul_863);  mul_863 = None
        view_603 = torch.ops.aten.view.default(mul_864, [256, 512]);  mul_864 = None
        permute_817 = torch.ops.aten.permute.default(view_603, [1, 0])
        mm_343 = torch.ops.aten.mm.default(permute_817, view_81);  permute_817 = view_81 = None
        permute_818 = torch.ops.aten.permute.default(mm_343, [1, 0]);  mm_343 = None
        mm_344 = torch.ops.aten.mm.default(view_603, permute_819);  view_603 = permute_819 = None
        view_604 = torch.ops.aten.view.default(mm_344, [2, 128, 384]);  mm_344 = None
        permute_820 = torch.ops.aten.permute.default(permute_818, [1, 0]);  permute_818 = None
        view_605 = torch.ops.aten.view.default(view_604, [2, 128, 6, 64]);  view_604 = None
        permute_821 = torch.ops.aten.permute.default(view_605, [0, 2, 1, 3]);  view_605 = None
        clone_164 = torch.ops.aten.clone.default(permute_821, memory_format = torch.contiguous_format);  permute_821 = None
        _unsafe_view_333 = torch.ops.aten._unsafe_view.default(clone_164, [12, 128, 64]);  clone_164 = None
        bmm_116 = torch.ops.aten.bmm.default(permute_822, _unsafe_view_333);  permute_822 = None
        bmm_117 = torch.ops.aten.bmm.default(_unsafe_view_333, permute_823);  _unsafe_view_333 = permute_823 = None
        view_606 = torch.ops.aten.view.default(bmm_116, [2, 6, 128, 64]);  bmm_116 = None
        view_607 = torch.ops.aten.view.default(bmm_117, [2, 6, 128, 128]);  bmm_117 = None
        philox_rand_like_41 = torch.ops.prims.philox_rand_like.default(view_607, philox_seed_like, 1179648)
        gt_102 = torch.ops.aten.gt.Scalar(philox_rand_like_41, 0.1);  philox_rand_like_41 = None
        _to_copy_89 = torch.ops.aten._to_copy.default(gt_102, dtype = torch.float32);  gt_102 = None
        mul_865 = torch.ops.aten.mul.Tensor(_to_copy_89, view_607);  _to_copy_89 = view_607 = None
        mul_866 = torch.ops.aten.mul.Tensor(mul_865, 1.1111111111111112);  mul_865 = None
        alias_451 = torch.ops.aten.alias.default(alias_104);  alias_104 = None
        alias_452 = torch.ops.aten.alias.default(alias_451);  alias_451 = None
        mul_867 = torch.ops.aten.mul.Tensor(mul_866, alias_452);  mul_866 = None
        sum_103 = torch.ops.aten.sum.dim_IntList(mul_867, [-1], True)
        mul_868 = torch.ops.aten.mul.Tensor(alias_452, sum_103);  alias_452 = sum_103 = None
        sub_75 = torch.ops.aten.sub.Tensor(mul_867, mul_868);  mul_867 = mul_868 = None
        add_322 = torch.ops.aten.add.Tensor(sub_73, sub_75);  sub_73 = None
        view_608 = torch.ops.aten.view.default(sub_75, [12, 128, 128]);  sub_75 = None
        bmm_118 = torch.ops.aten.bmm.default(permute_824, view_608);  permute_824 = None
        bmm_119 = torch.ops.aten.bmm.default(view_608, permute_825);  view_608 = permute_825 = None
        view_609 = torch.ops.aten.view.default(bmm_118, [2, 6, 64, 128]);  bmm_118 = None
        view_610 = torch.ops.aten.view.default(bmm_119, [2, 6, 128, 64]);  bmm_119 = None
        permute_826 = torch.ops.aten.permute.default(view_609, [0, 1, 3, 2]);  view_609 = None
        permute_827 = torch.ops.aten.permute.default(view_606, [0, 2, 1, 3]);  view_606 = None
        clone_165 = torch.ops.aten.clone.default(permute_827, memory_format = torch.contiguous_format);  permute_827 = None
        _unsafe_view_334 = torch.ops.aten._unsafe_view.default(clone_165, [2, 128, 384]);  clone_165 = None
        view_611 = torch.ops.aten.view.default(_unsafe_view_334, [256, 384]);  _unsafe_view_334 = None
        permute_828 = torch.ops.aten.permute.default(view_611, [1, 0])
        mm_345 = torch.ops.aten.mm.default(permute_828, view_73);  permute_828 = None
        permute_829 = torch.ops.aten.permute.default(mm_345, [1, 0]);  mm_345 = None
        mm_346 = torch.ops.aten.mm.default(view_611, permute_830);  view_611 = permute_830 = None
        view_612 = torch.ops.aten.view.default(mm_346, [2, 128, 512]);  mm_346 = None
        permute_831 = torch.ops.aten.permute.default(permute_829, [1, 0]);  permute_829 = None
        permute_832 = torch.ops.aten.permute.default(permute_826, [0, 2, 1, 3]);  permute_826 = None
        view_613 = torch.ops.aten.view.default(permute_832, [2, 128, 384]);  permute_832 = None
        clone_166 = torch.ops.aten.clone.default(view_613, memory_format = torch.contiguous_format);  view_613 = None
        _unsafe_view_335 = torch.ops.aten._unsafe_view.default(clone_166, [256, 384]);  clone_166 = None
        permute_833 = torch.ops.aten.permute.default(_unsafe_view_335, [1, 0])
        mm_347 = torch.ops.aten.mm.default(permute_833, view_73);  permute_833 = None
        permute_834 = torch.ops.aten.permute.default(mm_347, [1, 0]);  mm_347 = None
        mm_348 = torch.ops.aten.mm.default(_unsafe_view_335, permute_835);  _unsafe_view_335 = permute_835 = None
        view_614 = torch.ops.aten.view.default(mm_348, [2, 128, 512]);  mm_348 = None
        add_323 = torch.ops.aten.add.Tensor(view_612, view_614);  view_612 = view_614 = None
        permute_836 = torch.ops.aten.permute.default(permute_834, [1, 0]);  permute_834 = None
        permute_837 = torch.ops.aten.permute.default(view_610, [0, 2, 1, 3]);  view_610 = None
        clone_167 = torch.ops.aten.clone.default(permute_837, memory_format = torch.contiguous_format);  permute_837 = None
        _unsafe_view_336 = torch.ops.aten._unsafe_view.default(clone_167, [2, 128, 384]);  clone_167 = None
        view_615 = torch.ops.aten.view.default(_unsafe_view_336, [256, 384]);  _unsafe_view_336 = None
        permute_838 = torch.ops.aten.permute.default(view_615, [1, 0])
        mm_349 = torch.ops.aten.mm.default(permute_838, view_73);  permute_838 = view_73 = None
        permute_839 = torch.ops.aten.permute.default(mm_349, [1, 0]);  mm_349 = None
        mm_350 = torch.ops.aten.mm.default(view_615, permute_840);  view_615 = permute_840 = None
        view_616 = torch.ops.aten.view.default(mm_350, [2, 128, 512]);  mm_350 = None
        add_324 = torch.ops.aten.add.Tensor(add_323, view_616);  add_323 = view_616 = None
        permute_841 = torch.ops.aten.permute.default(permute_839, [1, 0]);  permute_839 = None
        mul_869 = torch.ops.aten.mul.Tensor(add_324, primals_13);  primals_13 = None
        mul_870 = torch.ops.aten.mul.Tensor(add_324, mul_119);  add_324 = mul_119 = None
        sum_104 = torch.ops.aten.sum.dim_IntList(mul_870, [0, 1], True);  mul_870 = None
        view_617 = torch.ops.aten.view.default(sum_104, [512]);  sum_104 = None
        mul_871 = torch.ops.aten.mul.Tensor(mul_869, add_51)
        mul_872 = torch.ops.aten.mul.Tensor(mul_869, reciprocal_18);  mul_869 = reciprocal_18 = None
        sum_105 = torch.ops.aten.sum.dim_IntList(mul_871, [2], True);  mul_871 = None
        add_325 = torch.ops.aten.add.Tensor(add_321, mul_872);  add_321 = mul_872 = None
        alias_453 = torch.ops.aten.alias.default(alias_101);  alias_101 = None
        alias_454 = torch.ops.aten.alias.default(alias_453);  alias_453 = None
        pow_127 = torch.ops.aten.pow.Tensor_Scalar(alias_454, 3);  alias_454 = None
        mul_873 = torch.ops.aten.mul.Scalar(sum_105, -0.5);  sum_105 = None
        mul_874 = torch.ops.aten.mul.Tensor(mul_873, pow_127);  mul_873 = pow_127 = None
        expand_127 = torch.ops.aten.expand.default(mul_874, [2, 128, 512]);  mul_874 = None
        div_58 = torch.ops.aten.div.Scalar(expand_127, 512);  expand_127 = None
        pow_128 = torch.ops.aten.pow.Tensor_Scalar(add_51, 1.0);  add_51 = None
        mul_875 = torch.ops.aten.mul.Scalar(pow_128, 2.0);  pow_128 = None
        mul_876 = torch.ops.aten.mul.Tensor(div_58, mul_875);  div_58 = mul_875 = None
        add_326 = torch.ops.aten.add.Tensor(add_325, mul_876);  add_325 = mul_876 = None
        _to_copy_90 = torch.ops.aten._to_copy.default(gt_25, dtype = torch.float32);  gt_25 = None
        mul_877 = torch.ops.aten.mul.Tensor(_to_copy_90, 1.1111111111111112);  _to_copy_90 = None
        mul_878 = torch.ops.aten.mul.Tensor(add_326, mul_877);  mul_877 = None
        view_618 = torch.ops.aten.view.default(mul_878, [256, 512]);  mul_878 = None
        permute_842 = torch.ops.aten.permute.default(view_618, [1, 0])
        mm_351 = torch.ops.aten.mm.default(permute_842, view_72);  permute_842 = view_72 = None
        permute_843 = torch.ops.aten.permute.default(mm_351, [1, 0]);  mm_351 = None
        mm_352 = torch.ops.aten.mm.default(view_618, permute_844);  view_618 = permute_844 = None
        view_619 = torch.ops.aten.view.default(mm_352, [2, 128, 1024]);  mm_352 = None
        permute_845 = torch.ops.aten.permute.default(permute_843, [1, 0]);  permute_843 = None
        _to_copy_91 = torch.ops.aten._to_copy.default(gt_24, dtype = torch.float32);  gt_24 = None
        mul_879 = torch.ops.aten.mul.Tensor(_to_copy_91, 1.1111111111111112);  _to_copy_91 = None
        mul_880 = torch.ops.aten.mul.Tensor(view_619, mul_879);  view_619 = mul_879 = None
        mul_881 = torch.ops.aten.mul.Tensor(mul_880, mul_113);  mul_113 = None
        mul_882 = torch.ops.aten.mul.Tensor(mul_880, _unsafe_view_70);  mul_880 = _unsafe_view_70 = None
        view_620 = torch.ops.aten.view.default(mul_881, [256, 1024]);  mul_881 = None
        permute_846 = torch.ops.aten.permute.default(view_620, [1, 0])
        mm_353 = torch.ops.aten.mm.default(permute_846, view_70);  permute_846 = None
        permute_847 = torch.ops.aten.permute.default(mm_353, [1, 0]);  mm_353 = None
        mm_354 = torch.ops.aten.mm.default(view_620, permute_848);  view_620 = permute_848 = None
        view_621 = torch.ops.aten.view.default(mm_354, [2, 128, 512]);  mm_354 = None
        permute_849 = torch.ops.aten.permute.default(permute_847, [1, 0]);  permute_847 = None
        mul_883 = torch.ops.aten.mul.Tensor(mul_882, mul_108);  mul_108 = None
        mul_884 = torch.ops.aten.mul.Tensor(mul_882, add_50);  mul_882 = add_50 = None
        alias_455 = torch.ops.aten.alias.default(alias_96);  alias_96 = None
        alias_456 = torch.ops.aten.alias.default(alias_455);  alias_455 = None
        mul_885 = torch.ops.aten.mul.Tensor(alias_456, alias_456);  alias_456 = None
        _tensor_constant13 = self._tensor_constant13
        lift_fresh_copy_13 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant13);  _tensor_constant13 = None
        sub_76 = torch.ops.aten.sub.Tensor(lift_fresh_copy_13, mul_885);  lift_fresh_copy_13 = mul_885 = None
        mul_886 = torch.ops.aten.mul.Tensor(mul_883, sub_76);  mul_883 = sub_76 = None
        mul_887 = torch.ops.aten.mul.Tensor(mul_886, 0.7978845608028654);  mul_886 = None
        mul_888 = torch.ops.aten.mul.Tensor(mul_887, 0.044715)
        pow_129 = torch.ops.aten.pow.Tensor_Scalar(_unsafe_view_69, 2.0);  _unsafe_view_69 = None
        mul_889 = torch.ops.aten.mul.Scalar(pow_129, 3.0);  pow_129 = None
        mul_890 = torch.ops.aten.mul.Tensor(mul_888, mul_889);  mul_888 = mul_889 = None
        add_327 = torch.ops.aten.add.Tensor(mul_887, mul_890);  mul_887 = mul_890 = None
        mul_891 = torch.ops.aten.mul.Tensor(mul_884, 0.5);  mul_884 = None
        add_328 = torch.ops.aten.add.Tensor(add_327, mul_891);  add_327 = mul_891 = None
        view_622 = torch.ops.aten.view.default(add_328, [256, 1024]);  add_328 = None
        permute_850 = torch.ops.aten.permute.default(view_622, [1, 0])
        mm_355 = torch.ops.aten.mm.default(permute_850, view_70);  permute_850 = view_70 = None
        permute_851 = torch.ops.aten.permute.default(mm_355, [1, 0]);  mm_355 = None
        mm_356 = torch.ops.aten.mm.default(view_622, permute_852);  view_622 = permute_852 = None
        view_623 = torch.ops.aten.view.default(mm_356, [2, 128, 512]);  mm_356 = None
        add_329 = torch.ops.aten.add.Tensor(view_621, view_623);  view_621 = view_623 = None
        permute_853 = torch.ops.aten.permute.default(permute_851, [1, 0]);  permute_851 = None
        mul_892 = torch.ops.aten.mul.Tensor(add_329, primals_12);  primals_12 = None
        mul_893 = torch.ops.aten.mul.Tensor(add_329, mul_106);  add_329 = mul_106 = None
        sum_106 = torch.ops.aten.sum.dim_IntList(mul_893, [0, 1], True);  mul_893 = None
        view_624 = torch.ops.aten.view.default(sum_106, [512]);  sum_106 = None
        mul_894 = torch.ops.aten.mul.Tensor(mul_892, add_46)
        mul_895 = torch.ops.aten.mul.Tensor(mul_892, reciprocal_16);  mul_892 = reciprocal_16 = None
        sum_107 = torch.ops.aten.sum.dim_IntList(mul_894, [2], True);  mul_894 = None
        add_330 = torch.ops.aten.add.Tensor(add_326, mul_895);  add_326 = mul_895 = None
        alias_457 = torch.ops.aten.alias.default(alias_93);  alias_93 = None
        alias_458 = torch.ops.aten.alias.default(alias_457);  alias_457 = None
        pow_130 = torch.ops.aten.pow.Tensor_Scalar(alias_458, 3);  alias_458 = None
        mul_896 = torch.ops.aten.mul.Scalar(sum_107, -0.5);  sum_107 = None
        mul_897 = torch.ops.aten.mul.Tensor(mul_896, pow_130);  mul_896 = pow_130 = None
        expand_128 = torch.ops.aten.expand.default(mul_897, [2, 128, 512]);  mul_897 = None
        div_59 = torch.ops.aten.div.Scalar(expand_128, 512);  expand_128 = None
        pow_131 = torch.ops.aten.pow.Tensor_Scalar(add_46, 1.0);  add_46 = None
        mul_898 = torch.ops.aten.mul.Scalar(pow_131, 2.0);  pow_131 = None
        mul_899 = torch.ops.aten.mul.Tensor(div_59, mul_898);  div_59 = mul_898 = None
        add_331 = torch.ops.aten.add.Tensor(add_330, mul_899);  add_330 = mul_899 = None
        _to_copy_92 = torch.ops.aten._to_copy.default(gt_23, dtype = torch.float32);  gt_23 = None
        mul_900 = torch.ops.aten.mul.Tensor(_to_copy_92, 1.1111111111111112);  _to_copy_92 = None
        mul_901 = torch.ops.aten.mul.Tensor(add_331, mul_900);  mul_900 = None
        view_625 = torch.ops.aten.view.default(mul_901, [256, 512]);  mul_901 = None
        permute_854 = torch.ops.aten.permute.default(view_625, [1, 0])
        mm_357 = torch.ops.aten.mm.default(permute_854, view_69);  permute_854 = view_69 = None
        permute_855 = torch.ops.aten.permute.default(mm_357, [1, 0]);  mm_357 = None
        mm_358 = torch.ops.aten.mm.default(view_625, permute_856);  view_625 = permute_856 = None
        view_626 = torch.ops.aten.view.default(mm_358, [2, 128, 384]);  mm_358 = None
        permute_857 = torch.ops.aten.permute.default(permute_855, [1, 0]);  permute_855 = None
        view_627 = torch.ops.aten.view.default(view_626, [2, 128, 6, 64]);  view_626 = None
        permute_858 = torch.ops.aten.permute.default(view_627, [0, 2, 1, 3]);  view_627 = None
        clone_168 = torch.ops.aten.clone.default(permute_858, memory_format = torch.contiguous_format);  permute_858 = None
        _unsafe_view_337 = torch.ops.aten._unsafe_view.default(clone_168, [12, 128, 64]);  clone_168 = None
        bmm_120 = torch.ops.aten.bmm.default(permute_859, _unsafe_view_337);  permute_859 = None
        bmm_121 = torch.ops.aten.bmm.default(_unsafe_view_337, permute_860);  _unsafe_view_337 = permute_860 = None
        view_628 = torch.ops.aten.view.default(bmm_120, [2, 6, 128, 64]);  bmm_120 = None
        view_629 = torch.ops.aten.view.default(bmm_121, [2, 6, 128, 128]);  bmm_121 = None
        philox_rand_like_42 = torch.ops.prims.philox_rand_like.default(view_629, philox_seed_like, 983040)
        gt_103 = torch.ops.aten.gt.Scalar(philox_rand_like_42, 0.1);  philox_rand_like_42 = None
        _to_copy_93 = torch.ops.aten._to_copy.default(gt_103, dtype = torch.float32);  gt_103 = None
        mul_902 = torch.ops.aten.mul.Tensor(_to_copy_93, view_629);  _to_copy_93 = view_629 = None
        mul_903 = torch.ops.aten.mul.Tensor(mul_902, 1.1111111111111112);  mul_902 = None
        alias_459 = torch.ops.aten.alias.default(alias_89);  alias_89 = None
        alias_460 = torch.ops.aten.alias.default(alias_459);  alias_459 = None
        mul_904 = torch.ops.aten.mul.Tensor(mul_903, alias_460);  mul_903 = None
        sum_108 = torch.ops.aten.sum.dim_IntList(mul_904, [-1], True)
        mul_905 = torch.ops.aten.mul.Tensor(alias_460, sum_108);  alias_460 = sum_108 = None
        sub_77 = torch.ops.aten.sub.Tensor(mul_904, mul_905);  mul_904 = mul_905 = None
        add_332 = torch.ops.aten.add.Tensor(add_322, sub_77);  add_322 = None
        view_630 = torch.ops.aten.view.default(sub_77, [12, 128, 128]);  sub_77 = None
        bmm_122 = torch.ops.aten.bmm.default(permute_861, view_630);  permute_861 = None
        bmm_123 = torch.ops.aten.bmm.default(view_630, permute_862);  view_630 = permute_862 = None
        view_631 = torch.ops.aten.view.default(bmm_122, [2, 6, 64, 128]);  bmm_122 = None
        view_632 = torch.ops.aten.view.default(bmm_123, [2, 6, 128, 64]);  bmm_123 = None
        permute_863 = torch.ops.aten.permute.default(view_631, [0, 1, 3, 2]);  view_631 = None
        permute_864 = torch.ops.aten.permute.default(view_628, [0, 2, 1, 3]);  view_628 = None
        clone_169 = torch.ops.aten.clone.default(permute_864, memory_format = torch.contiguous_format);  permute_864 = None
        _unsafe_view_338 = torch.ops.aten._unsafe_view.default(clone_169, [2, 128, 384]);  clone_169 = None
        view_633 = torch.ops.aten.view.default(_unsafe_view_338, [256, 384]);  _unsafe_view_338 = None
        permute_865 = torch.ops.aten.permute.default(view_633, [1, 0])
        mm_359 = torch.ops.aten.mm.default(permute_865, view_61);  permute_865 = None
        permute_866 = torch.ops.aten.permute.default(mm_359, [1, 0]);  mm_359 = None
        mm_360 = torch.ops.aten.mm.default(view_633, permute_867);  view_633 = permute_867 = None
        view_634 = torch.ops.aten.view.default(mm_360, [2, 128, 512]);  mm_360 = None
        permute_868 = torch.ops.aten.permute.default(permute_866, [1, 0]);  permute_866 = None
        permute_869 = torch.ops.aten.permute.default(permute_863, [0, 2, 1, 3]);  permute_863 = None
        view_635 = torch.ops.aten.view.default(permute_869, [2, 128, 384]);  permute_869 = None
        clone_170 = torch.ops.aten.clone.default(view_635, memory_format = torch.contiguous_format);  view_635 = None
        _unsafe_view_339 = torch.ops.aten._unsafe_view.default(clone_170, [256, 384]);  clone_170 = None
        permute_870 = torch.ops.aten.permute.default(_unsafe_view_339, [1, 0])
        mm_361 = torch.ops.aten.mm.default(permute_870, view_61);  permute_870 = None
        permute_871 = torch.ops.aten.permute.default(mm_361, [1, 0]);  mm_361 = None
        mm_362 = torch.ops.aten.mm.default(_unsafe_view_339, permute_872);  _unsafe_view_339 = permute_872 = None
        view_636 = torch.ops.aten.view.default(mm_362, [2, 128, 512]);  mm_362 = None
        add_333 = torch.ops.aten.add.Tensor(view_634, view_636);  view_634 = view_636 = None
        permute_873 = torch.ops.aten.permute.default(permute_871, [1, 0]);  permute_871 = None
        permute_874 = torch.ops.aten.permute.default(view_632, [0, 2, 1, 3]);  view_632 = None
        clone_171 = torch.ops.aten.clone.default(permute_874, memory_format = torch.contiguous_format);  permute_874 = None
        _unsafe_view_340 = torch.ops.aten._unsafe_view.default(clone_171, [2, 128, 384]);  clone_171 = None
        view_637 = torch.ops.aten.view.default(_unsafe_view_340, [256, 384]);  _unsafe_view_340 = None
        permute_875 = torch.ops.aten.permute.default(view_637, [1, 0])
        mm_363 = torch.ops.aten.mm.default(permute_875, view_61);  permute_875 = view_61 = None
        permute_876 = torch.ops.aten.permute.default(mm_363, [1, 0]);  mm_363 = None
        mm_364 = torch.ops.aten.mm.default(view_637, permute_877);  view_637 = permute_877 = None
        view_638 = torch.ops.aten.view.default(mm_364, [2, 128, 512]);  mm_364 = None
        add_334 = torch.ops.aten.add.Tensor(add_333, view_638);  add_333 = view_638 = None
        permute_878 = torch.ops.aten.permute.default(permute_876, [1, 0]);  permute_876 = None
        mul_906 = torch.ops.aten.mul.Tensor(add_334, primals_11);  primals_11 = None
        mul_907 = torch.ops.aten.mul.Tensor(add_334, mul_100);  add_334 = mul_100 = None
        sum_109 = torch.ops.aten.sum.dim_IntList(mul_907, [0, 1], True);  mul_907 = None
        view_639 = torch.ops.aten.view.default(sum_109, [512]);  sum_109 = None
        mul_908 = torch.ops.aten.mul.Tensor(mul_906, add_43)
        mul_909 = torch.ops.aten.mul.Tensor(mul_906, reciprocal_15);  mul_906 = reciprocal_15 = None
        sum_110 = torch.ops.aten.sum.dim_IntList(mul_908, [2], True);  mul_908 = None
        add_335 = torch.ops.aten.add.Tensor(add_331, mul_909);  add_331 = mul_909 = None
        alias_461 = torch.ops.aten.alias.default(alias_86);  alias_86 = None
        alias_462 = torch.ops.aten.alias.default(alias_461);  alias_461 = None
        pow_132 = torch.ops.aten.pow.Tensor_Scalar(alias_462, 3);  alias_462 = None
        mul_910 = torch.ops.aten.mul.Scalar(sum_110, -0.5);  sum_110 = None
        mul_911 = torch.ops.aten.mul.Tensor(mul_910, pow_132);  mul_910 = pow_132 = None
        expand_129 = torch.ops.aten.expand.default(mul_911, [2, 128, 512]);  mul_911 = None
        div_60 = torch.ops.aten.div.Scalar(expand_129, 512);  expand_129 = None
        pow_133 = torch.ops.aten.pow.Tensor_Scalar(add_43, 1.0);  add_43 = None
        mul_912 = torch.ops.aten.mul.Scalar(pow_133, 2.0);  pow_133 = None
        mul_913 = torch.ops.aten.mul.Tensor(div_60, mul_912);  div_60 = mul_912 = None
        add_336 = torch.ops.aten.add.Tensor(add_335, mul_913);  add_335 = mul_913 = None
        _to_copy_94 = torch.ops.aten._to_copy.default(gt_21, dtype = torch.float32);  gt_21 = None
        mul_914 = torch.ops.aten.mul.Tensor(_to_copy_94, 1.1111111111111112);  _to_copy_94 = None
        mul_915 = torch.ops.aten.mul.Tensor(add_336, mul_914);  mul_914 = None
        view_640 = torch.ops.aten.view.default(mul_915, [256, 512]);  mul_915 = None
        permute_879 = torch.ops.aten.permute.default(view_640, [1, 0])
        mm_365 = torch.ops.aten.mm.default(permute_879, view_60);  permute_879 = view_60 = None
        permute_880 = torch.ops.aten.permute.default(mm_365, [1, 0]);  mm_365 = None
        mm_366 = torch.ops.aten.mm.default(view_640, permute_881);  view_640 = permute_881 = None
        view_641 = torch.ops.aten.view.default(mm_366, [2, 128, 1024]);  mm_366 = None
        permute_882 = torch.ops.aten.permute.default(permute_880, [1, 0]);  permute_880 = None
        _to_copy_95 = torch.ops.aten._to_copy.default(gt_20, dtype = torch.float32);  gt_20 = None
        mul_916 = torch.ops.aten.mul.Tensor(_to_copy_95, 1.1111111111111112);  _to_copy_95 = None
        mul_917 = torch.ops.aten.mul.Tensor(view_641, mul_916);  view_641 = mul_916 = None
        mul_918 = torch.ops.aten.mul.Tensor(mul_917, mul_94);  mul_94 = None
        mul_919 = torch.ops.aten.mul.Tensor(mul_917, _unsafe_view_58);  mul_917 = _unsafe_view_58 = None
        view_642 = torch.ops.aten.view.default(mul_918, [256, 1024]);  mul_918 = None
        permute_883 = torch.ops.aten.permute.default(view_642, [1, 0])
        mm_367 = torch.ops.aten.mm.default(permute_883, view_58);  permute_883 = None
        permute_884 = torch.ops.aten.permute.default(mm_367, [1, 0]);  mm_367 = None
        mm_368 = torch.ops.aten.mm.default(view_642, permute_885);  view_642 = permute_885 = None
        view_643 = torch.ops.aten.view.default(mm_368, [2, 128, 512]);  mm_368 = None
        permute_886 = torch.ops.aten.permute.default(permute_884, [1, 0]);  permute_884 = None
        mul_920 = torch.ops.aten.mul.Tensor(mul_919, mul_89);  mul_89 = None
        mul_921 = torch.ops.aten.mul.Tensor(mul_919, add_42);  mul_919 = add_42 = None
        alias_463 = torch.ops.aten.alias.default(alias_81);  alias_81 = None
        alias_464 = torch.ops.aten.alias.default(alias_463);  alias_463 = None
        mul_922 = torch.ops.aten.mul.Tensor(alias_464, alias_464);  alias_464 = None
        _tensor_constant14 = self._tensor_constant14
        lift_fresh_copy_14 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant14);  _tensor_constant14 = None
        sub_78 = torch.ops.aten.sub.Tensor(lift_fresh_copy_14, mul_922);  lift_fresh_copy_14 = mul_922 = None
        mul_923 = torch.ops.aten.mul.Tensor(mul_920, sub_78);  mul_920 = sub_78 = None
        mul_924 = torch.ops.aten.mul.Tensor(mul_923, 0.7978845608028654);  mul_923 = None
        mul_925 = torch.ops.aten.mul.Tensor(mul_924, 0.044715)
        pow_134 = torch.ops.aten.pow.Tensor_Scalar(_unsafe_view_57, 2.0);  _unsafe_view_57 = None
        mul_926 = torch.ops.aten.mul.Scalar(pow_134, 3.0);  pow_134 = None
        mul_927 = torch.ops.aten.mul.Tensor(mul_925, mul_926);  mul_925 = mul_926 = None
        add_337 = torch.ops.aten.add.Tensor(mul_924, mul_927);  mul_924 = mul_927 = None
        mul_928 = torch.ops.aten.mul.Tensor(mul_921, 0.5);  mul_921 = None
        add_338 = torch.ops.aten.add.Tensor(add_337, mul_928);  add_337 = mul_928 = None
        view_644 = torch.ops.aten.view.default(add_338, [256, 1024]);  add_338 = None
        permute_887 = torch.ops.aten.permute.default(view_644, [1, 0])
        mm_369 = torch.ops.aten.mm.default(permute_887, view_58);  permute_887 = view_58 = None
        permute_888 = torch.ops.aten.permute.default(mm_369, [1, 0]);  mm_369 = None
        mm_370 = torch.ops.aten.mm.default(view_644, permute_889);  view_644 = permute_889 = None
        view_645 = torch.ops.aten.view.default(mm_370, [2, 128, 512]);  mm_370 = None
        add_339 = torch.ops.aten.add.Tensor(view_643, view_645);  view_643 = view_645 = None
        permute_890 = torch.ops.aten.permute.default(permute_888, [1, 0]);  permute_888 = None
        mul_929 = torch.ops.aten.mul.Tensor(add_339, primals_10);  primals_10 = None
        mul_930 = torch.ops.aten.mul.Tensor(add_339, mul_87);  add_339 = mul_87 = None
        sum_111 = torch.ops.aten.sum.dim_IntList(mul_930, [0, 1], True);  mul_930 = None
        view_646 = torch.ops.aten.view.default(sum_111, [512]);  sum_111 = None
        mul_931 = torch.ops.aten.mul.Tensor(mul_929, add_38)
        mul_932 = torch.ops.aten.mul.Tensor(mul_929, reciprocal_13);  mul_929 = reciprocal_13 = None
        sum_112 = torch.ops.aten.sum.dim_IntList(mul_931, [2], True);  mul_931 = None
        add_340 = torch.ops.aten.add.Tensor(add_336, mul_932);  add_336 = mul_932 = None
        alias_465 = torch.ops.aten.alias.default(alias_78);  alias_78 = None
        alias_466 = torch.ops.aten.alias.default(alias_465);  alias_465 = None
        pow_135 = torch.ops.aten.pow.Tensor_Scalar(alias_466, 3);  alias_466 = None
        mul_933 = torch.ops.aten.mul.Scalar(sum_112, -0.5);  sum_112 = None
        mul_934 = torch.ops.aten.mul.Tensor(mul_933, pow_135);  mul_933 = pow_135 = None
        expand_130 = torch.ops.aten.expand.default(mul_934, [2, 128, 512]);  mul_934 = None
        div_61 = torch.ops.aten.div.Scalar(expand_130, 512);  expand_130 = None
        pow_136 = torch.ops.aten.pow.Tensor_Scalar(add_38, 1.0);  add_38 = None
        mul_935 = torch.ops.aten.mul.Scalar(pow_136, 2.0);  pow_136 = None
        mul_936 = torch.ops.aten.mul.Tensor(div_61, mul_935);  div_61 = mul_935 = None
        add_341 = torch.ops.aten.add.Tensor(add_340, mul_936);  add_340 = mul_936 = None
        _to_copy_96 = torch.ops.aten._to_copy.default(gt_19, dtype = torch.float32);  gt_19 = None
        mul_937 = torch.ops.aten.mul.Tensor(_to_copy_96, 1.1111111111111112);  _to_copy_96 = None
        mul_938 = torch.ops.aten.mul.Tensor(add_341, mul_937);  mul_937 = None
        view_647 = torch.ops.aten.view.default(mul_938, [256, 512]);  mul_938 = None
        permute_891 = torch.ops.aten.permute.default(view_647, [1, 0])
        mm_371 = torch.ops.aten.mm.default(permute_891, view_57);  permute_891 = view_57 = None
        permute_892 = torch.ops.aten.permute.default(mm_371, [1, 0]);  mm_371 = None
        mm_372 = torch.ops.aten.mm.default(view_647, permute_893);  view_647 = permute_893 = None
        view_648 = torch.ops.aten.view.default(mm_372, [2, 128, 384]);  mm_372 = None
        permute_894 = torch.ops.aten.permute.default(permute_892, [1, 0]);  permute_892 = None
        view_649 = torch.ops.aten.view.default(view_648, [2, 128, 6, 64]);  view_648 = None
        permute_895 = torch.ops.aten.permute.default(view_649, [0, 2, 1, 3]);  view_649 = None
        clone_172 = torch.ops.aten.clone.default(permute_895, memory_format = torch.contiguous_format);  permute_895 = None
        _unsafe_view_341 = torch.ops.aten._unsafe_view.default(clone_172, [12, 128, 64]);  clone_172 = None
        bmm_124 = torch.ops.aten.bmm.default(permute_896, _unsafe_view_341);  permute_896 = None
        bmm_125 = torch.ops.aten.bmm.default(_unsafe_view_341, permute_897);  _unsafe_view_341 = permute_897 = None
        view_650 = torch.ops.aten.view.default(bmm_124, [2, 6, 128, 64]);  bmm_124 = None
        view_651 = torch.ops.aten.view.default(bmm_125, [2, 6, 128, 128]);  bmm_125 = None
        philox_rand_like_43 = torch.ops.prims.philox_rand_like.default(view_651, philox_seed_like, 786432)
        gt_104 = torch.ops.aten.gt.Scalar(philox_rand_like_43, 0.1);  philox_rand_like_43 = None
        _to_copy_97 = torch.ops.aten._to_copy.default(gt_104, dtype = torch.float32);  gt_104 = None
        mul_939 = torch.ops.aten.mul.Tensor(_to_copy_97, view_651);  _to_copy_97 = view_651 = None
        mul_940 = torch.ops.aten.mul.Tensor(mul_939, 1.1111111111111112);  mul_939 = None
        alias_467 = torch.ops.aten.alias.default(alias_74);  alias_74 = None
        alias_468 = torch.ops.aten.alias.default(alias_467);  alias_467 = None
        mul_941 = torch.ops.aten.mul.Tensor(mul_940, alias_468);  mul_940 = None
        sum_113 = torch.ops.aten.sum.dim_IntList(mul_941, [-1], True)
        mul_942 = torch.ops.aten.mul.Tensor(alias_468, sum_113);  alias_468 = sum_113 = None
        sub_79 = torch.ops.aten.sub.Tensor(mul_941, mul_942);  mul_941 = mul_942 = None
        add_342 = torch.ops.aten.add.Tensor(add_332, sub_79);  add_332 = None
        view_652 = torch.ops.aten.view.default(sub_79, [12, 128, 128]);  sub_79 = None
        bmm_126 = torch.ops.aten.bmm.default(permute_898, view_652);  permute_898 = None
        bmm_127 = torch.ops.aten.bmm.default(view_652, permute_899);  view_652 = permute_899 = None
        view_653 = torch.ops.aten.view.default(bmm_126, [2, 6, 64, 128]);  bmm_126 = None
        view_654 = torch.ops.aten.view.default(bmm_127, [2, 6, 128, 64]);  bmm_127 = None
        permute_900 = torch.ops.aten.permute.default(view_653, [0, 1, 3, 2]);  view_653 = None
        permute_901 = torch.ops.aten.permute.default(view_650, [0, 2, 1, 3]);  view_650 = None
        clone_173 = torch.ops.aten.clone.default(permute_901, memory_format = torch.contiguous_format);  permute_901 = None
        _unsafe_view_342 = torch.ops.aten._unsafe_view.default(clone_173, [2, 128, 384]);  clone_173 = None
        view_655 = torch.ops.aten.view.default(_unsafe_view_342, [256, 384]);  _unsafe_view_342 = None
        permute_902 = torch.ops.aten.permute.default(view_655, [1, 0])
        mm_373 = torch.ops.aten.mm.default(permute_902, view_49);  permute_902 = None
        permute_903 = torch.ops.aten.permute.default(mm_373, [1, 0]);  mm_373 = None
        mm_374 = torch.ops.aten.mm.default(view_655, permute_904);  view_655 = permute_904 = None
        view_656 = torch.ops.aten.view.default(mm_374, [2, 128, 512]);  mm_374 = None
        permute_905 = torch.ops.aten.permute.default(permute_903, [1, 0]);  permute_903 = None
        permute_906 = torch.ops.aten.permute.default(permute_900, [0, 2, 1, 3]);  permute_900 = None
        view_657 = torch.ops.aten.view.default(permute_906, [2, 128, 384]);  permute_906 = None
        clone_174 = torch.ops.aten.clone.default(view_657, memory_format = torch.contiguous_format);  view_657 = None
        _unsafe_view_343 = torch.ops.aten._unsafe_view.default(clone_174, [256, 384]);  clone_174 = None
        permute_907 = torch.ops.aten.permute.default(_unsafe_view_343, [1, 0])
        mm_375 = torch.ops.aten.mm.default(permute_907, view_49);  permute_907 = None
        permute_908 = torch.ops.aten.permute.default(mm_375, [1, 0]);  mm_375 = None
        mm_376 = torch.ops.aten.mm.default(_unsafe_view_343, permute_909);  _unsafe_view_343 = permute_909 = None
        view_658 = torch.ops.aten.view.default(mm_376, [2, 128, 512]);  mm_376 = None
        add_343 = torch.ops.aten.add.Tensor(view_656, view_658);  view_656 = view_658 = None
        permute_910 = torch.ops.aten.permute.default(permute_908, [1, 0]);  permute_908 = None
        permute_911 = torch.ops.aten.permute.default(view_654, [0, 2, 1, 3]);  view_654 = None
        clone_175 = torch.ops.aten.clone.default(permute_911, memory_format = torch.contiguous_format);  permute_911 = None
        _unsafe_view_344 = torch.ops.aten._unsafe_view.default(clone_175, [2, 128, 384]);  clone_175 = None
        view_659 = torch.ops.aten.view.default(_unsafe_view_344, [256, 384]);  _unsafe_view_344 = None
        permute_912 = torch.ops.aten.permute.default(view_659, [1, 0])
        mm_377 = torch.ops.aten.mm.default(permute_912, view_49);  permute_912 = view_49 = None
        permute_913 = torch.ops.aten.permute.default(mm_377, [1, 0]);  mm_377 = None
        mm_378 = torch.ops.aten.mm.default(view_659, permute_914);  view_659 = permute_914 = None
        view_660 = torch.ops.aten.view.default(mm_378, [2, 128, 512]);  mm_378 = None
        add_344 = torch.ops.aten.add.Tensor(add_343, view_660);  add_343 = view_660 = None
        permute_915 = torch.ops.aten.permute.default(permute_913, [1, 0]);  permute_913 = None
        mul_943 = torch.ops.aten.mul.Tensor(add_344, primals_9);  primals_9 = None
        mul_944 = torch.ops.aten.mul.Tensor(add_344, mul_81);  add_344 = mul_81 = None
        sum_114 = torch.ops.aten.sum.dim_IntList(mul_944, [0, 1], True);  mul_944 = None
        view_661 = torch.ops.aten.view.default(sum_114, [512]);  sum_114 = None
        mul_945 = torch.ops.aten.mul.Tensor(mul_943, add_35)
        mul_946 = torch.ops.aten.mul.Tensor(mul_943, reciprocal_12);  mul_943 = reciprocal_12 = None
        sum_115 = torch.ops.aten.sum.dim_IntList(mul_945, [2], True);  mul_945 = None
        add_345 = torch.ops.aten.add.Tensor(add_341, mul_946);  add_341 = mul_946 = None
        alias_469 = torch.ops.aten.alias.default(alias_71);  alias_71 = None
        alias_470 = torch.ops.aten.alias.default(alias_469);  alias_469 = None
        pow_137 = torch.ops.aten.pow.Tensor_Scalar(alias_470, 3);  alias_470 = None
        mul_947 = torch.ops.aten.mul.Scalar(sum_115, -0.5);  sum_115 = None
        mul_948 = torch.ops.aten.mul.Tensor(mul_947, pow_137);  mul_947 = pow_137 = None
        expand_131 = torch.ops.aten.expand.default(mul_948, [2, 128, 512]);  mul_948 = None
        div_62 = torch.ops.aten.div.Scalar(expand_131, 512);  expand_131 = None
        pow_138 = torch.ops.aten.pow.Tensor_Scalar(add_35, 1.0);  add_35 = None
        mul_949 = torch.ops.aten.mul.Scalar(pow_138, 2.0);  pow_138 = None
        mul_950 = torch.ops.aten.mul.Tensor(div_62, mul_949);  div_62 = mul_949 = None
        add_346 = torch.ops.aten.add.Tensor(add_345, mul_950);  add_345 = mul_950 = None
        _to_copy_98 = torch.ops.aten._to_copy.default(gt_17, dtype = torch.float32);  gt_17 = None
        mul_951 = torch.ops.aten.mul.Tensor(_to_copy_98, 1.1111111111111112);  _to_copy_98 = None
        mul_952 = torch.ops.aten.mul.Tensor(add_346, mul_951);  mul_951 = None
        view_662 = torch.ops.aten.view.default(mul_952, [256, 512]);  mul_952 = None
        permute_916 = torch.ops.aten.permute.default(view_662, [1, 0])
        mm_379 = torch.ops.aten.mm.default(permute_916, view_48);  permute_916 = view_48 = None
        permute_917 = torch.ops.aten.permute.default(mm_379, [1, 0]);  mm_379 = None
        mm_380 = torch.ops.aten.mm.default(view_662, permute_918);  view_662 = permute_918 = None
        view_663 = torch.ops.aten.view.default(mm_380, [2, 128, 1024]);  mm_380 = None
        permute_919 = torch.ops.aten.permute.default(permute_917, [1, 0]);  permute_917 = None
        _to_copy_99 = torch.ops.aten._to_copy.default(gt_16, dtype = torch.float32);  gt_16 = None
        mul_953 = torch.ops.aten.mul.Tensor(_to_copy_99, 1.1111111111111112);  _to_copy_99 = None
        mul_954 = torch.ops.aten.mul.Tensor(view_663, mul_953);  view_663 = mul_953 = None
        mul_955 = torch.ops.aten.mul.Tensor(mul_954, mul_75);  mul_75 = None
        mul_956 = torch.ops.aten.mul.Tensor(mul_954, _unsafe_view_46);  mul_954 = _unsafe_view_46 = None
        view_664 = torch.ops.aten.view.default(mul_955, [256, 1024]);  mul_955 = None
        permute_920 = torch.ops.aten.permute.default(view_664, [1, 0])
        mm_381 = torch.ops.aten.mm.default(permute_920, view_46);  permute_920 = None
        permute_921 = torch.ops.aten.permute.default(mm_381, [1, 0]);  mm_381 = None
        mm_382 = torch.ops.aten.mm.default(view_664, permute_922);  view_664 = permute_922 = None
        view_665 = torch.ops.aten.view.default(mm_382, [2, 128, 512]);  mm_382 = None
        permute_923 = torch.ops.aten.permute.default(permute_921, [1, 0]);  permute_921 = None
        mul_957 = torch.ops.aten.mul.Tensor(mul_956, mul_70);  mul_70 = None
        mul_958 = torch.ops.aten.mul.Tensor(mul_956, add_34);  mul_956 = add_34 = None
        alias_471 = torch.ops.aten.alias.default(alias_66);  alias_66 = None
        alias_472 = torch.ops.aten.alias.default(alias_471);  alias_471 = None
        mul_959 = torch.ops.aten.mul.Tensor(alias_472, alias_472);  alias_472 = None
        _tensor_constant15 = self._tensor_constant15
        lift_fresh_copy_15 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant15);  _tensor_constant15 = None
        sub_80 = torch.ops.aten.sub.Tensor(lift_fresh_copy_15, mul_959);  lift_fresh_copy_15 = mul_959 = None
        mul_960 = torch.ops.aten.mul.Tensor(mul_957, sub_80);  mul_957 = sub_80 = None
        mul_961 = torch.ops.aten.mul.Tensor(mul_960, 0.7978845608028654);  mul_960 = None
        mul_962 = torch.ops.aten.mul.Tensor(mul_961, 0.044715)
        pow_139 = torch.ops.aten.pow.Tensor_Scalar(_unsafe_view_45, 2.0);  _unsafe_view_45 = None
        mul_963 = torch.ops.aten.mul.Scalar(pow_139, 3.0);  pow_139 = None
        mul_964 = torch.ops.aten.mul.Tensor(mul_962, mul_963);  mul_962 = mul_963 = None
        add_347 = torch.ops.aten.add.Tensor(mul_961, mul_964);  mul_961 = mul_964 = None
        mul_965 = torch.ops.aten.mul.Tensor(mul_958, 0.5);  mul_958 = None
        add_348 = torch.ops.aten.add.Tensor(add_347, mul_965);  add_347 = mul_965 = None
        view_666 = torch.ops.aten.view.default(add_348, [256, 1024]);  add_348 = None
        permute_924 = torch.ops.aten.permute.default(view_666, [1, 0])
        mm_383 = torch.ops.aten.mm.default(permute_924, view_46);  permute_924 = view_46 = None
        permute_925 = torch.ops.aten.permute.default(mm_383, [1, 0]);  mm_383 = None
        mm_384 = torch.ops.aten.mm.default(view_666, permute_926);  view_666 = permute_926 = None
        view_667 = torch.ops.aten.view.default(mm_384, [2, 128, 512]);  mm_384 = None
        add_349 = torch.ops.aten.add.Tensor(view_665, view_667);  view_665 = view_667 = None
        permute_927 = torch.ops.aten.permute.default(permute_925, [1, 0]);  permute_925 = None
        mul_966 = torch.ops.aten.mul.Tensor(add_349, primals_8);  primals_8 = None
        mul_967 = torch.ops.aten.mul.Tensor(add_349, mul_68);  add_349 = mul_68 = None
        sum_116 = torch.ops.aten.sum.dim_IntList(mul_967, [0, 1], True);  mul_967 = None
        view_668 = torch.ops.aten.view.default(sum_116, [512]);  sum_116 = None
        mul_968 = torch.ops.aten.mul.Tensor(mul_966, add_30)
        mul_969 = torch.ops.aten.mul.Tensor(mul_966, reciprocal_10);  mul_966 = reciprocal_10 = None
        sum_117 = torch.ops.aten.sum.dim_IntList(mul_968, [2], True);  mul_968 = None
        add_350 = torch.ops.aten.add.Tensor(add_346, mul_969);  add_346 = mul_969 = None
        alias_473 = torch.ops.aten.alias.default(alias_63);  alias_63 = None
        alias_474 = torch.ops.aten.alias.default(alias_473);  alias_473 = None
        pow_140 = torch.ops.aten.pow.Tensor_Scalar(alias_474, 3);  alias_474 = None
        mul_970 = torch.ops.aten.mul.Scalar(sum_117, -0.5);  sum_117 = None
        mul_971 = torch.ops.aten.mul.Tensor(mul_970, pow_140);  mul_970 = pow_140 = None
        expand_132 = torch.ops.aten.expand.default(mul_971, [2, 128, 512]);  mul_971 = None
        div_63 = torch.ops.aten.div.Scalar(expand_132, 512);  expand_132 = None
        pow_141 = torch.ops.aten.pow.Tensor_Scalar(add_30, 1.0);  add_30 = None
        mul_972 = torch.ops.aten.mul.Scalar(pow_141, 2.0);  pow_141 = None
        mul_973 = torch.ops.aten.mul.Tensor(div_63, mul_972);  div_63 = mul_972 = None
        add_351 = torch.ops.aten.add.Tensor(add_350, mul_973);  add_350 = mul_973 = None
        _to_copy_100 = torch.ops.aten._to_copy.default(gt_15, dtype = torch.float32);  gt_15 = None
        mul_974 = torch.ops.aten.mul.Tensor(_to_copy_100, 1.1111111111111112);  _to_copy_100 = None
        mul_975 = torch.ops.aten.mul.Tensor(add_351, mul_974);  mul_974 = None
        view_669 = torch.ops.aten.view.default(mul_975, [256, 512]);  mul_975 = None
        permute_928 = torch.ops.aten.permute.default(view_669, [1, 0])
        mm_385 = torch.ops.aten.mm.default(permute_928, view_45);  permute_928 = view_45 = None
        permute_929 = torch.ops.aten.permute.default(mm_385, [1, 0]);  mm_385 = None
        mm_386 = torch.ops.aten.mm.default(view_669, permute_930);  view_669 = permute_930 = None
        view_670 = torch.ops.aten.view.default(mm_386, [2, 128, 384]);  mm_386 = None
        permute_931 = torch.ops.aten.permute.default(permute_929, [1, 0]);  permute_929 = None
        view_671 = torch.ops.aten.view.default(view_670, [2, 128, 6, 64]);  view_670 = None
        permute_932 = torch.ops.aten.permute.default(view_671, [0, 2, 1, 3]);  view_671 = None
        clone_176 = torch.ops.aten.clone.default(permute_932, memory_format = torch.contiguous_format);  permute_932 = None
        _unsafe_view_345 = torch.ops.aten._unsafe_view.default(clone_176, [12, 128, 64]);  clone_176 = None
        bmm_128 = torch.ops.aten.bmm.default(permute_933, _unsafe_view_345);  permute_933 = None
        bmm_129 = torch.ops.aten.bmm.default(_unsafe_view_345, permute_934);  _unsafe_view_345 = permute_934 = None
        view_672 = torch.ops.aten.view.default(bmm_128, [2, 6, 128, 64]);  bmm_128 = None
        view_673 = torch.ops.aten.view.default(bmm_129, [2, 6, 128, 128]);  bmm_129 = None
        philox_rand_like_44 = torch.ops.prims.philox_rand_like.default(view_673, philox_seed_like, 589824)
        gt_105 = torch.ops.aten.gt.Scalar(philox_rand_like_44, 0.1);  philox_rand_like_44 = None
        _to_copy_101 = torch.ops.aten._to_copy.default(gt_105, dtype = torch.float32);  gt_105 = None
        mul_976 = torch.ops.aten.mul.Tensor(_to_copy_101, view_673);  _to_copy_101 = view_673 = None
        mul_977 = torch.ops.aten.mul.Tensor(mul_976, 1.1111111111111112);  mul_976 = None
        alias_475 = torch.ops.aten.alias.default(alias_59);  alias_59 = None
        alias_476 = torch.ops.aten.alias.default(alias_475);  alias_475 = None
        mul_978 = torch.ops.aten.mul.Tensor(mul_977, alias_476);  mul_977 = None
        sum_118 = torch.ops.aten.sum.dim_IntList(mul_978, [-1], True)
        mul_979 = torch.ops.aten.mul.Tensor(alias_476, sum_118);  alias_476 = sum_118 = None
        sub_81 = torch.ops.aten.sub.Tensor(mul_978, mul_979);  mul_978 = mul_979 = None
        add_352 = torch.ops.aten.add.Tensor(add_342, sub_81);  add_342 = None
        view_674 = torch.ops.aten.view.default(sub_81, [12, 128, 128]);  sub_81 = None
        bmm_130 = torch.ops.aten.bmm.default(permute_935, view_674);  permute_935 = None
        bmm_131 = torch.ops.aten.bmm.default(view_674, permute_936);  view_674 = permute_936 = None
        view_675 = torch.ops.aten.view.default(bmm_130, [2, 6, 64, 128]);  bmm_130 = None
        view_676 = torch.ops.aten.view.default(bmm_131, [2, 6, 128, 64]);  bmm_131 = None
        permute_937 = torch.ops.aten.permute.default(view_675, [0, 1, 3, 2]);  view_675 = None
        permute_938 = torch.ops.aten.permute.default(view_672, [0, 2, 1, 3]);  view_672 = None
        clone_177 = torch.ops.aten.clone.default(permute_938, memory_format = torch.contiguous_format);  permute_938 = None
        _unsafe_view_346 = torch.ops.aten._unsafe_view.default(clone_177, [2, 128, 384]);  clone_177 = None
        view_677 = torch.ops.aten.view.default(_unsafe_view_346, [256, 384]);  _unsafe_view_346 = None
        permute_939 = torch.ops.aten.permute.default(view_677, [1, 0])
        mm_387 = torch.ops.aten.mm.default(permute_939, view_37);  permute_939 = None
        permute_940 = torch.ops.aten.permute.default(mm_387, [1, 0]);  mm_387 = None
        mm_388 = torch.ops.aten.mm.default(view_677, permute_941);  view_677 = permute_941 = None
        view_678 = torch.ops.aten.view.default(mm_388, [2, 128, 512]);  mm_388 = None
        permute_942 = torch.ops.aten.permute.default(permute_940, [1, 0]);  permute_940 = None
        permute_943 = torch.ops.aten.permute.default(permute_937, [0, 2, 1, 3]);  permute_937 = None
        view_679 = torch.ops.aten.view.default(permute_943, [2, 128, 384]);  permute_943 = None
        clone_178 = torch.ops.aten.clone.default(view_679, memory_format = torch.contiguous_format);  view_679 = None
        _unsafe_view_347 = torch.ops.aten._unsafe_view.default(clone_178, [256, 384]);  clone_178 = None
        permute_944 = torch.ops.aten.permute.default(_unsafe_view_347, [1, 0])
        mm_389 = torch.ops.aten.mm.default(permute_944, view_37);  permute_944 = None
        permute_945 = torch.ops.aten.permute.default(mm_389, [1, 0]);  mm_389 = None
        mm_390 = torch.ops.aten.mm.default(_unsafe_view_347, permute_946);  _unsafe_view_347 = permute_946 = None
        view_680 = torch.ops.aten.view.default(mm_390, [2, 128, 512]);  mm_390 = None
        add_353 = torch.ops.aten.add.Tensor(view_678, view_680);  view_678 = view_680 = None
        permute_947 = torch.ops.aten.permute.default(permute_945, [1, 0]);  permute_945 = None
        permute_948 = torch.ops.aten.permute.default(view_676, [0, 2, 1, 3]);  view_676 = None
        clone_179 = torch.ops.aten.clone.default(permute_948, memory_format = torch.contiguous_format);  permute_948 = None
        _unsafe_view_348 = torch.ops.aten._unsafe_view.default(clone_179, [2, 128, 384]);  clone_179 = None
        view_681 = torch.ops.aten.view.default(_unsafe_view_348, [256, 384]);  _unsafe_view_348 = None
        permute_949 = torch.ops.aten.permute.default(view_681, [1, 0])
        mm_391 = torch.ops.aten.mm.default(permute_949, view_37);  permute_949 = view_37 = None
        permute_950 = torch.ops.aten.permute.default(mm_391, [1, 0]);  mm_391 = None
        mm_392 = torch.ops.aten.mm.default(view_681, permute_951);  view_681 = permute_951 = None
        view_682 = torch.ops.aten.view.default(mm_392, [2, 128, 512]);  mm_392 = None
        add_354 = torch.ops.aten.add.Tensor(add_353, view_682);  add_353 = view_682 = None
        permute_952 = torch.ops.aten.permute.default(permute_950, [1, 0]);  permute_950 = None
        mul_980 = torch.ops.aten.mul.Tensor(add_354, primals_7);  primals_7 = None
        mul_981 = torch.ops.aten.mul.Tensor(add_354, mul_62);  add_354 = mul_62 = None
        sum_119 = torch.ops.aten.sum.dim_IntList(mul_981, [0, 1], True);  mul_981 = None
        view_683 = torch.ops.aten.view.default(sum_119, [512]);  sum_119 = None
        mul_982 = torch.ops.aten.mul.Tensor(mul_980, add_27)
        mul_983 = torch.ops.aten.mul.Tensor(mul_980, reciprocal_9);  mul_980 = reciprocal_9 = None
        sum_120 = torch.ops.aten.sum.dim_IntList(mul_982, [2], True);  mul_982 = None
        add_355 = torch.ops.aten.add.Tensor(add_351, mul_983);  add_351 = mul_983 = None
        alias_477 = torch.ops.aten.alias.default(alias_56);  alias_56 = None
        alias_478 = torch.ops.aten.alias.default(alias_477);  alias_477 = None
        pow_142 = torch.ops.aten.pow.Tensor_Scalar(alias_478, 3);  alias_478 = None
        mul_984 = torch.ops.aten.mul.Scalar(sum_120, -0.5);  sum_120 = None
        mul_985 = torch.ops.aten.mul.Tensor(mul_984, pow_142);  mul_984 = pow_142 = None
        expand_133 = torch.ops.aten.expand.default(mul_985, [2, 128, 512]);  mul_985 = None
        div_64 = torch.ops.aten.div.Scalar(expand_133, 512);  expand_133 = None
        pow_143 = torch.ops.aten.pow.Tensor_Scalar(add_27, 1.0);  add_27 = None
        mul_986 = torch.ops.aten.mul.Scalar(pow_143, 2.0);  pow_143 = None
        mul_987 = torch.ops.aten.mul.Tensor(div_64, mul_986);  div_64 = mul_986 = None
        add_356 = torch.ops.aten.add.Tensor(add_355, mul_987);  add_355 = mul_987 = None
        _to_copy_102 = torch.ops.aten._to_copy.default(gt_13, dtype = torch.float32);  gt_13 = None
        mul_988 = torch.ops.aten.mul.Tensor(_to_copy_102, 1.1111111111111112);  _to_copy_102 = None
        mul_989 = torch.ops.aten.mul.Tensor(add_356, mul_988);  mul_988 = None
        view_684 = torch.ops.aten.view.default(mul_989, [256, 512]);  mul_989 = None
        permute_953 = torch.ops.aten.permute.default(view_684, [1, 0])
        mm_393 = torch.ops.aten.mm.default(permute_953, view_36);  permute_953 = view_36 = None
        permute_954 = torch.ops.aten.permute.default(mm_393, [1, 0]);  mm_393 = None
        mm_394 = torch.ops.aten.mm.default(view_684, permute_955);  view_684 = permute_955 = None
        view_685 = torch.ops.aten.view.default(mm_394, [2, 128, 1024]);  mm_394 = None
        permute_956 = torch.ops.aten.permute.default(permute_954, [1, 0]);  permute_954 = None
        _to_copy_103 = torch.ops.aten._to_copy.default(gt_12, dtype = torch.float32);  gt_12 = None
        mul_990 = torch.ops.aten.mul.Tensor(_to_copy_103, 1.1111111111111112);  _to_copy_103 = None
        mul_991 = torch.ops.aten.mul.Tensor(view_685, mul_990);  view_685 = mul_990 = None
        mul_992 = torch.ops.aten.mul.Tensor(mul_991, mul_56);  mul_56 = None
        mul_993 = torch.ops.aten.mul.Tensor(mul_991, _unsafe_view_34);  mul_991 = _unsafe_view_34 = None
        view_686 = torch.ops.aten.view.default(mul_992, [256, 1024]);  mul_992 = None
        permute_957 = torch.ops.aten.permute.default(view_686, [1, 0])
        mm_395 = torch.ops.aten.mm.default(permute_957, view_34);  permute_957 = None
        permute_958 = torch.ops.aten.permute.default(mm_395, [1, 0]);  mm_395 = None
        mm_396 = torch.ops.aten.mm.default(view_686, permute_959);  view_686 = permute_959 = None
        view_687 = torch.ops.aten.view.default(mm_396, [2, 128, 512]);  mm_396 = None
        permute_960 = torch.ops.aten.permute.default(permute_958, [1, 0]);  permute_958 = None
        mul_994 = torch.ops.aten.mul.Tensor(mul_993, mul_51);  mul_51 = None
        mul_995 = torch.ops.aten.mul.Tensor(mul_993, add_26);  mul_993 = add_26 = None
        alias_479 = torch.ops.aten.alias.default(alias_51);  alias_51 = None
        alias_480 = torch.ops.aten.alias.default(alias_479);  alias_479 = None
        mul_996 = torch.ops.aten.mul.Tensor(alias_480, alias_480);  alias_480 = None
        _tensor_constant16 = self._tensor_constant16
        lift_fresh_copy_16 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant16);  _tensor_constant16 = None
        sub_82 = torch.ops.aten.sub.Tensor(lift_fresh_copy_16, mul_996);  lift_fresh_copy_16 = mul_996 = None
        mul_997 = torch.ops.aten.mul.Tensor(mul_994, sub_82);  mul_994 = sub_82 = None
        mul_998 = torch.ops.aten.mul.Tensor(mul_997, 0.7978845608028654);  mul_997 = None
        mul_999 = torch.ops.aten.mul.Tensor(mul_998, 0.044715)
        pow_144 = torch.ops.aten.pow.Tensor_Scalar(_unsafe_view_33, 2.0);  _unsafe_view_33 = None
        mul_1000 = torch.ops.aten.mul.Scalar(pow_144, 3.0);  pow_144 = None
        mul_1001 = torch.ops.aten.mul.Tensor(mul_999, mul_1000);  mul_999 = mul_1000 = None
        add_357 = torch.ops.aten.add.Tensor(mul_998, mul_1001);  mul_998 = mul_1001 = None
        mul_1002 = torch.ops.aten.mul.Tensor(mul_995, 0.5);  mul_995 = None
        add_358 = torch.ops.aten.add.Tensor(add_357, mul_1002);  add_357 = mul_1002 = None
        view_688 = torch.ops.aten.view.default(add_358, [256, 1024]);  add_358 = None
        permute_961 = torch.ops.aten.permute.default(view_688, [1, 0])
        mm_397 = torch.ops.aten.mm.default(permute_961, view_34);  permute_961 = view_34 = None
        permute_962 = torch.ops.aten.permute.default(mm_397, [1, 0]);  mm_397 = None
        mm_398 = torch.ops.aten.mm.default(view_688, permute_963);  view_688 = permute_963 = None
        view_689 = torch.ops.aten.view.default(mm_398, [2, 128, 512]);  mm_398 = None
        add_359 = torch.ops.aten.add.Tensor(view_687, view_689);  view_687 = view_689 = None
        permute_964 = torch.ops.aten.permute.default(permute_962, [1, 0]);  permute_962 = None
        mul_1003 = torch.ops.aten.mul.Tensor(add_359, primals_6);  primals_6 = None
        mul_1004 = torch.ops.aten.mul.Tensor(add_359, mul_49);  add_359 = mul_49 = None
        sum_121 = torch.ops.aten.sum.dim_IntList(mul_1004, [0, 1], True);  mul_1004 = None
        view_690 = torch.ops.aten.view.default(sum_121, [512]);  sum_121 = None
        mul_1005 = torch.ops.aten.mul.Tensor(mul_1003, add_22)
        mul_1006 = torch.ops.aten.mul.Tensor(mul_1003, reciprocal_7);  mul_1003 = reciprocal_7 = None
        sum_122 = torch.ops.aten.sum.dim_IntList(mul_1005, [2], True);  mul_1005 = None
        add_360 = torch.ops.aten.add.Tensor(add_356, mul_1006);  add_356 = mul_1006 = None
        alias_481 = torch.ops.aten.alias.default(alias_48);  alias_48 = None
        alias_482 = torch.ops.aten.alias.default(alias_481);  alias_481 = None
        pow_145 = torch.ops.aten.pow.Tensor_Scalar(alias_482, 3);  alias_482 = None
        mul_1007 = torch.ops.aten.mul.Scalar(sum_122, -0.5);  sum_122 = None
        mul_1008 = torch.ops.aten.mul.Tensor(mul_1007, pow_145);  mul_1007 = pow_145 = None
        expand_134 = torch.ops.aten.expand.default(mul_1008, [2, 128, 512]);  mul_1008 = None
        div_65 = torch.ops.aten.div.Scalar(expand_134, 512);  expand_134 = None
        pow_146 = torch.ops.aten.pow.Tensor_Scalar(add_22, 1.0);  add_22 = None
        mul_1009 = torch.ops.aten.mul.Scalar(pow_146, 2.0);  pow_146 = None
        mul_1010 = torch.ops.aten.mul.Tensor(div_65, mul_1009);  div_65 = mul_1009 = None
        add_361 = torch.ops.aten.add.Tensor(add_360, mul_1010);  add_360 = mul_1010 = None
        _to_copy_104 = torch.ops.aten._to_copy.default(gt_11, dtype = torch.float32);  gt_11 = None
        mul_1011 = torch.ops.aten.mul.Tensor(_to_copy_104, 1.1111111111111112);  _to_copy_104 = None
        mul_1012 = torch.ops.aten.mul.Tensor(add_361, mul_1011);  mul_1011 = None
        view_691 = torch.ops.aten.view.default(mul_1012, [256, 512]);  mul_1012 = None
        permute_965 = torch.ops.aten.permute.default(view_691, [1, 0])
        mm_399 = torch.ops.aten.mm.default(permute_965, view_33);  permute_965 = view_33 = None
        permute_966 = torch.ops.aten.permute.default(mm_399, [1, 0]);  mm_399 = None
        mm_400 = torch.ops.aten.mm.default(view_691, permute_967);  view_691 = permute_967 = None
        view_692 = torch.ops.aten.view.default(mm_400, [2, 128, 384]);  mm_400 = None
        permute_968 = torch.ops.aten.permute.default(permute_966, [1, 0]);  permute_966 = None
        view_693 = torch.ops.aten.view.default(view_692, [2, 128, 6, 64]);  view_692 = None
        permute_969 = torch.ops.aten.permute.default(view_693, [0, 2, 1, 3]);  view_693 = None
        clone_180 = torch.ops.aten.clone.default(permute_969, memory_format = torch.contiguous_format);  permute_969 = None
        _unsafe_view_349 = torch.ops.aten._unsafe_view.default(clone_180, [12, 128, 64]);  clone_180 = None
        bmm_132 = torch.ops.aten.bmm.default(permute_970, _unsafe_view_349);  permute_970 = None
        bmm_133 = torch.ops.aten.bmm.default(_unsafe_view_349, permute_971);  _unsafe_view_349 = permute_971 = None
        view_694 = torch.ops.aten.view.default(bmm_132, [2, 6, 128, 64]);  bmm_132 = None
        view_695 = torch.ops.aten.view.default(bmm_133, [2, 6, 128, 128]);  bmm_133 = None
        philox_rand_like_45 = torch.ops.prims.philox_rand_like.default(view_695, philox_seed_like, 393216)
        gt_106 = torch.ops.aten.gt.Scalar(philox_rand_like_45, 0.1);  philox_rand_like_45 = None
        _to_copy_105 = torch.ops.aten._to_copy.default(gt_106, dtype = torch.float32);  gt_106 = None
        mul_1013 = torch.ops.aten.mul.Tensor(_to_copy_105, view_695);  _to_copy_105 = view_695 = None
        mul_1014 = torch.ops.aten.mul.Tensor(mul_1013, 1.1111111111111112);  mul_1013 = None
        alias_483 = torch.ops.aten.alias.default(alias_44);  alias_44 = None
        alias_484 = torch.ops.aten.alias.default(alias_483);  alias_483 = None
        mul_1015 = torch.ops.aten.mul.Tensor(mul_1014, alias_484);  mul_1014 = None
        sum_123 = torch.ops.aten.sum.dim_IntList(mul_1015, [-1], True)
        mul_1016 = torch.ops.aten.mul.Tensor(alias_484, sum_123);  alias_484 = sum_123 = None
        sub_83 = torch.ops.aten.sub.Tensor(mul_1015, mul_1016);  mul_1015 = mul_1016 = None
        add_362 = torch.ops.aten.add.Tensor(add_352, sub_83);  add_352 = None
        view_696 = torch.ops.aten.view.default(sub_83, [12, 128, 128]);  sub_83 = None
        bmm_134 = torch.ops.aten.bmm.default(permute_972, view_696);  permute_972 = None
        bmm_135 = torch.ops.aten.bmm.default(view_696, permute_973);  view_696 = permute_973 = None
        view_697 = torch.ops.aten.view.default(bmm_134, [2, 6, 64, 128]);  bmm_134 = None
        view_698 = torch.ops.aten.view.default(bmm_135, [2, 6, 128, 64]);  bmm_135 = None
        permute_974 = torch.ops.aten.permute.default(view_697, [0, 1, 3, 2]);  view_697 = None
        permute_975 = torch.ops.aten.permute.default(view_694, [0, 2, 1, 3]);  view_694 = None
        clone_181 = torch.ops.aten.clone.default(permute_975, memory_format = torch.contiguous_format);  permute_975 = None
        _unsafe_view_350 = torch.ops.aten._unsafe_view.default(clone_181, [2, 128, 384]);  clone_181 = None
        view_699 = torch.ops.aten.view.default(_unsafe_view_350, [256, 384]);  _unsafe_view_350 = None
        permute_976 = torch.ops.aten.permute.default(view_699, [1, 0])
        mm_401 = torch.ops.aten.mm.default(permute_976, view_25);  permute_976 = None
        permute_977 = torch.ops.aten.permute.default(mm_401, [1, 0]);  mm_401 = None
        mm_402 = torch.ops.aten.mm.default(view_699, permute_978);  view_699 = permute_978 = None
        view_700 = torch.ops.aten.view.default(mm_402, [2, 128, 512]);  mm_402 = None
        permute_979 = torch.ops.aten.permute.default(permute_977, [1, 0]);  permute_977 = None
        permute_980 = torch.ops.aten.permute.default(permute_974, [0, 2, 1, 3]);  permute_974 = None
        view_701 = torch.ops.aten.view.default(permute_980, [2, 128, 384]);  permute_980 = None
        clone_182 = torch.ops.aten.clone.default(view_701, memory_format = torch.contiguous_format);  view_701 = None
        _unsafe_view_351 = torch.ops.aten._unsafe_view.default(clone_182, [256, 384]);  clone_182 = None
        permute_981 = torch.ops.aten.permute.default(_unsafe_view_351, [1, 0])
        mm_403 = torch.ops.aten.mm.default(permute_981, view_25);  permute_981 = None
        permute_982 = torch.ops.aten.permute.default(mm_403, [1, 0]);  mm_403 = None
        mm_404 = torch.ops.aten.mm.default(_unsafe_view_351, permute_983);  _unsafe_view_351 = permute_983 = None
        view_702 = torch.ops.aten.view.default(mm_404, [2, 128, 512]);  mm_404 = None
        add_363 = torch.ops.aten.add.Tensor(view_700, view_702);  view_700 = view_702 = None
        permute_984 = torch.ops.aten.permute.default(permute_982, [1, 0]);  permute_982 = None
        permute_985 = torch.ops.aten.permute.default(view_698, [0, 2, 1, 3]);  view_698 = None
        clone_183 = torch.ops.aten.clone.default(permute_985, memory_format = torch.contiguous_format);  permute_985 = None
        _unsafe_view_352 = torch.ops.aten._unsafe_view.default(clone_183, [2, 128, 384]);  clone_183 = None
        view_703 = torch.ops.aten.view.default(_unsafe_view_352, [256, 384]);  _unsafe_view_352 = None
        permute_986 = torch.ops.aten.permute.default(view_703, [1, 0])
        mm_405 = torch.ops.aten.mm.default(permute_986, view_25);  permute_986 = view_25 = None
        permute_987 = torch.ops.aten.permute.default(mm_405, [1, 0]);  mm_405 = None
        mm_406 = torch.ops.aten.mm.default(view_703, permute_988);  view_703 = permute_988 = None
        view_704 = torch.ops.aten.view.default(mm_406, [2, 128, 512]);  mm_406 = None
        add_364 = torch.ops.aten.add.Tensor(add_363, view_704);  add_363 = view_704 = None
        permute_989 = torch.ops.aten.permute.default(permute_987, [1, 0]);  permute_987 = None
        mul_1017 = torch.ops.aten.mul.Tensor(add_364, primals_5);  primals_5 = None
        mul_1018 = torch.ops.aten.mul.Tensor(add_364, mul_43);  add_364 = mul_43 = None
        sum_124 = torch.ops.aten.sum.dim_IntList(mul_1018, [0, 1], True);  mul_1018 = None
        view_705 = torch.ops.aten.view.default(sum_124, [512]);  sum_124 = None
        mul_1019 = torch.ops.aten.mul.Tensor(mul_1017, add_19)
        mul_1020 = torch.ops.aten.mul.Tensor(mul_1017, reciprocal_6);  mul_1017 = reciprocal_6 = None
        sum_125 = torch.ops.aten.sum.dim_IntList(mul_1019, [2], True);  mul_1019 = None
        add_365 = torch.ops.aten.add.Tensor(add_361, mul_1020);  add_361 = mul_1020 = None
        alias_485 = torch.ops.aten.alias.default(alias_41);  alias_41 = None
        alias_486 = torch.ops.aten.alias.default(alias_485);  alias_485 = None
        pow_147 = torch.ops.aten.pow.Tensor_Scalar(alias_486, 3);  alias_486 = None
        mul_1021 = torch.ops.aten.mul.Scalar(sum_125, -0.5);  sum_125 = None
        mul_1022 = torch.ops.aten.mul.Tensor(mul_1021, pow_147);  mul_1021 = pow_147 = None
        expand_135 = torch.ops.aten.expand.default(mul_1022, [2, 128, 512]);  mul_1022 = None
        div_66 = torch.ops.aten.div.Scalar(expand_135, 512);  expand_135 = None
        pow_148 = torch.ops.aten.pow.Tensor_Scalar(add_19, 1.0);  add_19 = None
        mul_1023 = torch.ops.aten.mul.Scalar(pow_148, 2.0);  pow_148 = None
        mul_1024 = torch.ops.aten.mul.Tensor(div_66, mul_1023);  div_66 = mul_1023 = None
        add_366 = torch.ops.aten.add.Tensor(add_365, mul_1024);  add_365 = mul_1024 = None
        _to_copy_106 = torch.ops.aten._to_copy.default(gt_9, dtype = torch.float32);  gt_9 = None
        mul_1025 = torch.ops.aten.mul.Tensor(_to_copy_106, 1.1111111111111112);  _to_copy_106 = None
        mul_1026 = torch.ops.aten.mul.Tensor(add_366, mul_1025);  mul_1025 = None
        view_706 = torch.ops.aten.view.default(mul_1026, [256, 512]);  mul_1026 = None
        permute_990 = torch.ops.aten.permute.default(view_706, [1, 0])
        mm_407 = torch.ops.aten.mm.default(permute_990, view_24);  permute_990 = view_24 = None
        permute_991 = torch.ops.aten.permute.default(mm_407, [1, 0]);  mm_407 = None
        mm_408 = torch.ops.aten.mm.default(view_706, permute_992);  view_706 = permute_992 = None
        view_707 = torch.ops.aten.view.default(mm_408, [2, 128, 1024]);  mm_408 = None
        permute_993 = torch.ops.aten.permute.default(permute_991, [1, 0]);  permute_991 = None
        _to_copy_107 = torch.ops.aten._to_copy.default(gt_8, dtype = torch.float32);  gt_8 = None
        mul_1027 = torch.ops.aten.mul.Tensor(_to_copy_107, 1.1111111111111112);  _to_copy_107 = None
        mul_1028 = torch.ops.aten.mul.Tensor(view_707, mul_1027);  view_707 = mul_1027 = None
        mul_1029 = torch.ops.aten.mul.Tensor(mul_1028, mul_37);  mul_37 = None
        mul_1030 = torch.ops.aten.mul.Tensor(mul_1028, _unsafe_view_22);  mul_1028 = _unsafe_view_22 = None
        view_708 = torch.ops.aten.view.default(mul_1029, [256, 1024]);  mul_1029 = None
        permute_994 = torch.ops.aten.permute.default(view_708, [1, 0])
        mm_409 = torch.ops.aten.mm.default(permute_994, view_22);  permute_994 = None
        permute_995 = torch.ops.aten.permute.default(mm_409, [1, 0]);  mm_409 = None
        mm_410 = torch.ops.aten.mm.default(view_708, permute_996);  view_708 = permute_996 = None
        view_709 = torch.ops.aten.view.default(mm_410, [2, 128, 512]);  mm_410 = None
        permute_997 = torch.ops.aten.permute.default(permute_995, [1, 0]);  permute_995 = None
        mul_1031 = torch.ops.aten.mul.Tensor(mul_1030, mul_32);  mul_32 = None
        mul_1032 = torch.ops.aten.mul.Tensor(mul_1030, add_18);  mul_1030 = add_18 = None
        alias_487 = torch.ops.aten.alias.default(alias_36);  alias_36 = None
        alias_488 = torch.ops.aten.alias.default(alias_487);  alias_487 = None
        mul_1033 = torch.ops.aten.mul.Tensor(alias_488, alias_488);  alias_488 = None
        _tensor_constant17 = self._tensor_constant17
        lift_fresh_copy_17 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant17);  _tensor_constant17 = None
        sub_84 = torch.ops.aten.sub.Tensor(lift_fresh_copy_17, mul_1033);  lift_fresh_copy_17 = mul_1033 = None
        mul_1034 = torch.ops.aten.mul.Tensor(mul_1031, sub_84);  mul_1031 = sub_84 = None
        mul_1035 = torch.ops.aten.mul.Tensor(mul_1034, 0.7978845608028654);  mul_1034 = None
        mul_1036 = torch.ops.aten.mul.Tensor(mul_1035, 0.044715)
        pow_149 = torch.ops.aten.pow.Tensor_Scalar(_unsafe_view_21, 2.0);  _unsafe_view_21 = None
        mul_1037 = torch.ops.aten.mul.Scalar(pow_149, 3.0);  pow_149 = None
        mul_1038 = torch.ops.aten.mul.Tensor(mul_1036, mul_1037);  mul_1036 = mul_1037 = None
        add_367 = torch.ops.aten.add.Tensor(mul_1035, mul_1038);  mul_1035 = mul_1038 = None
        mul_1039 = torch.ops.aten.mul.Tensor(mul_1032, 0.5);  mul_1032 = None
        add_368 = torch.ops.aten.add.Tensor(add_367, mul_1039);  add_367 = mul_1039 = None
        view_710 = torch.ops.aten.view.default(add_368, [256, 1024]);  add_368 = None
        permute_998 = torch.ops.aten.permute.default(view_710, [1, 0])
        mm_411 = torch.ops.aten.mm.default(permute_998, view_22);  permute_998 = view_22 = None
        permute_999 = torch.ops.aten.permute.default(mm_411, [1, 0]);  mm_411 = None
        mm_412 = torch.ops.aten.mm.default(view_710, permute_1000);  view_710 = permute_1000 = None
        view_711 = torch.ops.aten.view.default(mm_412, [2, 128, 512]);  mm_412 = None
        add_369 = torch.ops.aten.add.Tensor(view_709, view_711);  view_709 = view_711 = None
        permute_1001 = torch.ops.aten.permute.default(permute_999, [1, 0]);  permute_999 = None
        mul_1040 = torch.ops.aten.mul.Tensor(add_369, primals_4);  primals_4 = None
        mul_1041 = torch.ops.aten.mul.Tensor(add_369, mul_30);  add_369 = mul_30 = None
        sum_126 = torch.ops.aten.sum.dim_IntList(mul_1041, [0, 1], True);  mul_1041 = None
        view_712 = torch.ops.aten.view.default(sum_126, [512]);  sum_126 = None
        mul_1042 = torch.ops.aten.mul.Tensor(mul_1040, add_14)
        mul_1043 = torch.ops.aten.mul.Tensor(mul_1040, reciprocal_4);  mul_1040 = reciprocal_4 = None
        sum_127 = torch.ops.aten.sum.dim_IntList(mul_1042, [2], True);  mul_1042 = None
        add_370 = torch.ops.aten.add.Tensor(add_366, mul_1043);  add_366 = mul_1043 = None
        alias_489 = torch.ops.aten.alias.default(alias_33);  alias_33 = None
        alias_490 = torch.ops.aten.alias.default(alias_489);  alias_489 = None
        pow_150 = torch.ops.aten.pow.Tensor_Scalar(alias_490, 3);  alias_490 = None
        mul_1044 = torch.ops.aten.mul.Scalar(sum_127, -0.5);  sum_127 = None
        mul_1045 = torch.ops.aten.mul.Tensor(mul_1044, pow_150);  mul_1044 = pow_150 = None
        expand_136 = torch.ops.aten.expand.default(mul_1045, [2, 128, 512]);  mul_1045 = None
        div_67 = torch.ops.aten.div.Scalar(expand_136, 512);  expand_136 = None
        pow_151 = torch.ops.aten.pow.Tensor_Scalar(add_14, 1.0);  add_14 = None
        mul_1046 = torch.ops.aten.mul.Scalar(pow_151, 2.0);  pow_151 = None
        mul_1047 = torch.ops.aten.mul.Tensor(div_67, mul_1046);  div_67 = mul_1046 = None
        add_371 = torch.ops.aten.add.Tensor(add_370, mul_1047);  add_370 = mul_1047 = None
        _to_copy_108 = torch.ops.aten._to_copy.default(gt_7, dtype = torch.float32);  gt_7 = None
        mul_1048 = torch.ops.aten.mul.Tensor(_to_copy_108, 1.1111111111111112);  _to_copy_108 = None
        mul_1049 = torch.ops.aten.mul.Tensor(add_371, mul_1048);  mul_1048 = None
        view_713 = torch.ops.aten.view.default(mul_1049, [256, 512]);  mul_1049 = None
        permute_1002 = torch.ops.aten.permute.default(view_713, [1, 0])
        mm_413 = torch.ops.aten.mm.default(permute_1002, view_21);  permute_1002 = view_21 = None
        permute_1003 = torch.ops.aten.permute.default(mm_413, [1, 0]);  mm_413 = None
        mm_414 = torch.ops.aten.mm.default(view_713, permute_1004);  view_713 = permute_1004 = None
        view_714 = torch.ops.aten.view.default(mm_414, [2, 128, 384]);  mm_414 = None
        permute_1005 = torch.ops.aten.permute.default(permute_1003, [1, 0]);  permute_1003 = None
        view_715 = torch.ops.aten.view.default(view_714, [2, 128, 6, 64]);  view_714 = None
        permute_1006 = torch.ops.aten.permute.default(view_715, [0, 2, 1, 3]);  view_715 = None
        clone_184 = torch.ops.aten.clone.default(permute_1006, memory_format = torch.contiguous_format);  permute_1006 = None
        _unsafe_view_353 = torch.ops.aten._unsafe_view.default(clone_184, [12, 128, 64]);  clone_184 = None
        bmm_136 = torch.ops.aten.bmm.default(permute_1007, _unsafe_view_353);  permute_1007 = None
        bmm_137 = torch.ops.aten.bmm.default(_unsafe_view_353, permute_1008);  _unsafe_view_353 = permute_1008 = None
        view_716 = torch.ops.aten.view.default(bmm_136, [2, 6, 128, 64]);  bmm_136 = None
        view_717 = torch.ops.aten.view.default(bmm_137, [2, 6, 128, 128]);  bmm_137 = None
        philox_rand_like_46 = torch.ops.prims.philox_rand_like.default(view_717, philox_seed_like, 196608)
        gt_107 = torch.ops.aten.gt.Scalar(philox_rand_like_46, 0.1);  philox_rand_like_46 = None
        _to_copy_109 = torch.ops.aten._to_copy.default(gt_107, dtype = torch.float32);  gt_107 = None
        mul_1050 = torch.ops.aten.mul.Tensor(_to_copy_109, view_717);  _to_copy_109 = view_717 = None
        mul_1051 = torch.ops.aten.mul.Tensor(mul_1050, 1.1111111111111112);  mul_1050 = None
        alias_491 = torch.ops.aten.alias.default(alias_29);  alias_29 = None
        alias_492 = torch.ops.aten.alias.default(alias_491);  alias_491 = None
        mul_1052 = torch.ops.aten.mul.Tensor(mul_1051, alias_492);  mul_1051 = None
        sum_128 = torch.ops.aten.sum.dim_IntList(mul_1052, [-1], True)
        mul_1053 = torch.ops.aten.mul.Tensor(alias_492, sum_128);  alias_492 = sum_128 = None
        sub_85 = torch.ops.aten.sub.Tensor(mul_1052, mul_1053);  mul_1052 = mul_1053 = None
        add_372 = torch.ops.aten.add.Tensor(add_362, sub_85);  add_362 = None
        view_718 = torch.ops.aten.view.default(sub_85, [12, 128, 128]);  sub_85 = None
        bmm_138 = torch.ops.aten.bmm.default(permute_1009, view_718);  permute_1009 = None
        bmm_139 = torch.ops.aten.bmm.default(view_718, permute_1010);  view_718 = permute_1010 = None
        view_719 = torch.ops.aten.view.default(bmm_138, [2, 6, 64, 128]);  bmm_138 = None
        view_720 = torch.ops.aten.view.default(bmm_139, [2, 6, 128, 64]);  bmm_139 = None
        permute_1011 = torch.ops.aten.permute.default(view_719, [0, 1, 3, 2]);  view_719 = None
        permute_1012 = torch.ops.aten.permute.default(view_716, [0, 2, 1, 3]);  view_716 = None
        clone_185 = torch.ops.aten.clone.default(permute_1012, memory_format = torch.contiguous_format);  permute_1012 = None
        _unsafe_view_354 = torch.ops.aten._unsafe_view.default(clone_185, [2, 128, 384]);  clone_185 = None
        view_721 = torch.ops.aten.view.default(_unsafe_view_354, [256, 384]);  _unsafe_view_354 = None
        permute_1013 = torch.ops.aten.permute.default(view_721, [1, 0])
        mm_415 = torch.ops.aten.mm.default(permute_1013, view_13);  permute_1013 = None
        permute_1014 = torch.ops.aten.permute.default(mm_415, [1, 0]);  mm_415 = None
        mm_416 = torch.ops.aten.mm.default(view_721, permute_1015);  view_721 = permute_1015 = None
        view_722 = torch.ops.aten.view.default(mm_416, [2, 128, 512]);  mm_416 = None
        permute_1016 = torch.ops.aten.permute.default(permute_1014, [1, 0]);  permute_1014 = None
        permute_1017 = torch.ops.aten.permute.default(permute_1011, [0, 2, 1, 3]);  permute_1011 = None
        view_723 = torch.ops.aten.view.default(permute_1017, [2, 128, 384]);  permute_1017 = None
        clone_186 = torch.ops.aten.clone.default(view_723, memory_format = torch.contiguous_format);  view_723 = None
        _unsafe_view_355 = torch.ops.aten._unsafe_view.default(clone_186, [256, 384]);  clone_186 = None
        permute_1018 = torch.ops.aten.permute.default(_unsafe_view_355, [1, 0])
        mm_417 = torch.ops.aten.mm.default(permute_1018, view_13);  permute_1018 = None
        permute_1019 = torch.ops.aten.permute.default(mm_417, [1, 0]);  mm_417 = None
        mm_418 = torch.ops.aten.mm.default(_unsafe_view_355, permute_1020);  _unsafe_view_355 = permute_1020 = None
        view_724 = torch.ops.aten.view.default(mm_418, [2, 128, 512]);  mm_418 = None
        add_373 = torch.ops.aten.add.Tensor(view_722, view_724);  view_722 = view_724 = None
        permute_1021 = torch.ops.aten.permute.default(permute_1019, [1, 0]);  permute_1019 = None
        permute_1022 = torch.ops.aten.permute.default(view_720, [0, 2, 1, 3]);  view_720 = None
        clone_187 = torch.ops.aten.clone.default(permute_1022, memory_format = torch.contiguous_format);  permute_1022 = None
        _unsafe_view_356 = torch.ops.aten._unsafe_view.default(clone_187, [2, 128, 384]);  clone_187 = None
        view_725 = torch.ops.aten.view.default(_unsafe_view_356, [256, 384]);  _unsafe_view_356 = None
        permute_1023 = torch.ops.aten.permute.default(view_725, [1, 0])
        mm_419 = torch.ops.aten.mm.default(permute_1023, view_13);  permute_1023 = view_13 = None
        permute_1024 = torch.ops.aten.permute.default(mm_419, [1, 0]);  mm_419 = None
        mm_420 = torch.ops.aten.mm.default(view_725, permute_1025);  view_725 = permute_1025 = None
        view_726 = torch.ops.aten.view.default(mm_420, [2, 128, 512]);  mm_420 = None
        add_374 = torch.ops.aten.add.Tensor(add_373, view_726);  add_373 = view_726 = None
        permute_1026 = torch.ops.aten.permute.default(permute_1024, [1, 0]);  permute_1024 = None
        mul_1054 = torch.ops.aten.mul.Tensor(add_374, primals_3);  primals_3 = None
        mul_1055 = torch.ops.aten.mul.Tensor(add_374, mul_24);  add_374 = mul_24 = None
        sum_129 = torch.ops.aten.sum.dim_IntList(mul_1055, [0, 1], True);  mul_1055 = None
        view_727 = torch.ops.aten.view.default(sum_129, [512]);  sum_129 = None
        mul_1056 = torch.ops.aten.mul.Tensor(mul_1054, add_11)
        mul_1057 = torch.ops.aten.mul.Tensor(mul_1054, reciprocal_3);  mul_1054 = reciprocal_3 = None
        sum_130 = torch.ops.aten.sum.dim_IntList(mul_1056, [2], True);  mul_1056 = None
        add_375 = torch.ops.aten.add.Tensor(add_371, mul_1057);  add_371 = mul_1057 = None
        alias_493 = torch.ops.aten.alias.default(alias_26);  alias_26 = None
        alias_494 = torch.ops.aten.alias.default(alias_493);  alias_493 = None
        pow_152 = torch.ops.aten.pow.Tensor_Scalar(alias_494, 3);  alias_494 = None
        mul_1058 = torch.ops.aten.mul.Scalar(sum_130, -0.5);  sum_130 = None
        mul_1059 = torch.ops.aten.mul.Tensor(mul_1058, pow_152);  mul_1058 = pow_152 = None
        expand_137 = torch.ops.aten.expand.default(mul_1059, [2, 128, 512]);  mul_1059 = None
        div_68 = torch.ops.aten.div.Scalar(expand_137, 512);  expand_137 = None
        pow_153 = torch.ops.aten.pow.Tensor_Scalar(add_11, 1.0);  add_11 = None
        mul_1060 = torch.ops.aten.mul.Scalar(pow_153, 2.0);  pow_153 = None
        mul_1061 = torch.ops.aten.mul.Tensor(div_68, mul_1060);  div_68 = mul_1060 = None
        add_376 = torch.ops.aten.add.Tensor(add_375, mul_1061);  add_375 = mul_1061 = None
        _to_copy_110 = torch.ops.aten._to_copy.default(gt_5, dtype = torch.float32);  gt_5 = None
        mul_1062 = torch.ops.aten.mul.Tensor(_to_copy_110, 1.1111111111111112);  _to_copy_110 = None
        mul_1063 = torch.ops.aten.mul.Tensor(add_376, mul_1062);  mul_1062 = None
        view_728 = torch.ops.aten.view.default(mul_1063, [256, 512]);  mul_1063 = None
        permute_1027 = torch.ops.aten.permute.default(view_728, [1, 0])
        mm_421 = torch.ops.aten.mm.default(permute_1027, view_12);  permute_1027 = view_12 = None
        permute_1028 = torch.ops.aten.permute.default(mm_421, [1, 0]);  mm_421 = None
        mm_422 = torch.ops.aten.mm.default(view_728, permute_1029);  view_728 = permute_1029 = None
        view_729 = torch.ops.aten.view.default(mm_422, [2, 128, 1024]);  mm_422 = None
        permute_1030 = torch.ops.aten.permute.default(permute_1028, [1, 0]);  permute_1028 = None
        _to_copy_111 = torch.ops.aten._to_copy.default(gt_4, dtype = torch.float32);  gt_4 = None
        mul_1064 = torch.ops.aten.mul.Tensor(_to_copy_111, 1.1111111111111112);  _to_copy_111 = None
        mul_1065 = torch.ops.aten.mul.Tensor(view_729, mul_1064);  view_729 = mul_1064 = None
        mul_1066 = torch.ops.aten.mul.Tensor(mul_1065, mul_18);  mul_18 = None
        mul_1067 = torch.ops.aten.mul.Tensor(mul_1065, _unsafe_view_10);  mul_1065 = _unsafe_view_10 = None
        view_730 = torch.ops.aten.view.default(mul_1066, [256, 1024]);  mul_1066 = None
        permute_1031 = torch.ops.aten.permute.default(view_730, [1, 0])
        mm_423 = torch.ops.aten.mm.default(permute_1031, view_10);  permute_1031 = None
        permute_1032 = torch.ops.aten.permute.default(mm_423, [1, 0]);  mm_423 = None
        mm_424 = torch.ops.aten.mm.default(view_730, permute_1033);  view_730 = permute_1033 = None
        view_731 = torch.ops.aten.view.default(mm_424, [2, 128, 512]);  mm_424 = None
        permute_1034 = torch.ops.aten.permute.default(permute_1032, [1, 0]);  permute_1032 = None
        mul_1068 = torch.ops.aten.mul.Tensor(mul_1067, mul_13);  mul_13 = None
        mul_1069 = torch.ops.aten.mul.Tensor(mul_1067, add_10);  mul_1067 = add_10 = None
        alias_495 = torch.ops.aten.alias.default(alias_21);  alias_21 = None
        alias_496 = torch.ops.aten.alias.default(alias_495);  alias_495 = None
        mul_1070 = torch.ops.aten.mul.Tensor(alias_496, alias_496);  alias_496 = None
        _tensor_constant18 = self._tensor_constant18
        lift_fresh_copy_18 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant18);  _tensor_constant18 = None
        sub_86 = torch.ops.aten.sub.Tensor(lift_fresh_copy_18, mul_1070);  lift_fresh_copy_18 = mul_1070 = None
        mul_1071 = torch.ops.aten.mul.Tensor(mul_1068, sub_86);  mul_1068 = sub_86 = None
        mul_1072 = torch.ops.aten.mul.Tensor(mul_1071, 0.7978845608028654);  mul_1071 = None
        mul_1073 = torch.ops.aten.mul.Tensor(mul_1072, 0.044715)
        pow_154 = torch.ops.aten.pow.Tensor_Scalar(_unsafe_view_9, 2.0);  _unsafe_view_9 = None
        mul_1074 = torch.ops.aten.mul.Scalar(pow_154, 3.0);  pow_154 = None
        mul_1075 = torch.ops.aten.mul.Tensor(mul_1073, mul_1074);  mul_1073 = mul_1074 = None
        add_377 = torch.ops.aten.add.Tensor(mul_1072, mul_1075);  mul_1072 = mul_1075 = None
        mul_1076 = torch.ops.aten.mul.Tensor(mul_1069, 0.5);  mul_1069 = None
        add_378 = torch.ops.aten.add.Tensor(add_377, mul_1076);  add_377 = mul_1076 = None
        view_732 = torch.ops.aten.view.default(add_378, [256, 1024]);  add_378 = None
        permute_1035 = torch.ops.aten.permute.default(view_732, [1, 0])
        mm_425 = torch.ops.aten.mm.default(permute_1035, view_10);  permute_1035 = view_10 = None
        permute_1036 = torch.ops.aten.permute.default(mm_425, [1, 0]);  mm_425 = None
        mm_426 = torch.ops.aten.mm.default(view_732, permute_1037);  view_732 = permute_1037 = None
        view_733 = torch.ops.aten.view.default(mm_426, [2, 128, 512]);  mm_426 = None
        add_379 = torch.ops.aten.add.Tensor(view_731, view_733);  view_731 = view_733 = None
        permute_1038 = torch.ops.aten.permute.default(permute_1036, [1, 0]);  permute_1036 = None
        mul_1077 = torch.ops.aten.mul.Tensor(add_379, primals_2);  primals_2 = None
        mul_1078 = torch.ops.aten.mul.Tensor(add_379, mul_11);  add_379 = mul_11 = None
        sum_131 = torch.ops.aten.sum.dim_IntList(mul_1078, [0, 1], True);  mul_1078 = None
        view_734 = torch.ops.aten.view.default(sum_131, [512]);  sum_131 = None
        mul_1079 = torch.ops.aten.mul.Tensor(mul_1077, add_6)
        mul_1080 = torch.ops.aten.mul.Tensor(mul_1077, reciprocal_1);  mul_1077 = reciprocal_1 = None
        sum_132 = torch.ops.aten.sum.dim_IntList(mul_1079, [2], True);  mul_1079 = None
        add_380 = torch.ops.aten.add.Tensor(add_376, mul_1080);  add_376 = mul_1080 = None
        alias_497 = torch.ops.aten.alias.default(alias_18);  alias_18 = None
        alias_498 = torch.ops.aten.alias.default(alias_497);  alias_497 = None
        pow_155 = torch.ops.aten.pow.Tensor_Scalar(alias_498, 3);  alias_498 = None
        mul_1081 = torch.ops.aten.mul.Scalar(sum_132, -0.5);  sum_132 = None
        mul_1082 = torch.ops.aten.mul.Tensor(mul_1081, pow_155);  mul_1081 = pow_155 = None
        expand_138 = torch.ops.aten.expand.default(mul_1082, [2, 128, 512]);  mul_1082 = None
        div_69 = torch.ops.aten.div.Scalar(expand_138, 512);  expand_138 = None
        pow_156 = torch.ops.aten.pow.Tensor_Scalar(add_6, 1.0);  add_6 = None
        mul_1083 = torch.ops.aten.mul.Scalar(pow_156, 2.0);  pow_156 = None
        mul_1084 = torch.ops.aten.mul.Tensor(div_69, mul_1083);  div_69 = mul_1083 = None
        add_381 = torch.ops.aten.add.Tensor(add_380, mul_1084);  add_380 = mul_1084 = None
        _to_copy_112 = torch.ops.aten._to_copy.default(gt_3, dtype = torch.float32);  gt_3 = None
        mul_1085 = torch.ops.aten.mul.Tensor(_to_copy_112, 1.1111111111111112);  _to_copy_112 = None
        mul_1086 = torch.ops.aten.mul.Tensor(add_381, mul_1085);  mul_1085 = None
        view_735 = torch.ops.aten.view.default(mul_1086, [256, 512]);  mul_1086 = None
        permute_1039 = torch.ops.aten.permute.default(view_735, [1, 0])
        mm_427 = torch.ops.aten.mm.default(permute_1039, view_9);  permute_1039 = view_9 = None
        permute_1040 = torch.ops.aten.permute.default(mm_427, [1, 0]);  mm_427 = None
        mm_428 = torch.ops.aten.mm.default(view_735, permute_1041);  view_735 = permute_1041 = None
        view_736 = torch.ops.aten.view.default(mm_428, [2, 128, 384]);  mm_428 = None
        permute_1042 = torch.ops.aten.permute.default(permute_1040, [1, 0]);  permute_1040 = None
        view_737 = torch.ops.aten.view.default(view_736, [2, 128, 6, 64]);  view_736 = None
        permute_1043 = torch.ops.aten.permute.default(view_737, [0, 2, 1, 3]);  view_737 = None
        clone_188 = torch.ops.aten.clone.default(permute_1043, memory_format = torch.contiguous_format);  permute_1043 = None
        _unsafe_view_357 = torch.ops.aten._unsafe_view.default(clone_188, [12, 128, 64]);  clone_188 = None
        bmm_140 = torch.ops.aten.bmm.default(permute_1044, _unsafe_view_357);  permute_1044 = None
        bmm_141 = torch.ops.aten.bmm.default(_unsafe_view_357, permute_1045);  _unsafe_view_357 = permute_1045 = None
        view_738 = torch.ops.aten.view.default(bmm_140, [2, 6, 128, 64]);  bmm_140 = None
        view_739 = torch.ops.aten.view.default(bmm_141, [2, 6, 128, 128]);  bmm_141 = None
        philox_rand_like_47 = torch.ops.prims.philox_rand_like.default(view_739, philox_seed_like, 0);  philox_seed_like = None
        gt_108 = torch.ops.aten.gt.Scalar(philox_rand_like_47, 0.1);  philox_rand_like_47 = None
        _to_copy_113 = torch.ops.aten._to_copy.default(gt_108, dtype = torch.float32);  gt_108 = None
        mul_1087 = torch.ops.aten.mul.Tensor(_to_copy_113, view_739);  _to_copy_113 = view_739 = None
        mul_1088 = torch.ops.aten.mul.Tensor(mul_1087, 1.1111111111111112);  mul_1087 = None
        alias_499 = torch.ops.aten.alias.default(alias_14);  alias_14 = None
        alias_500 = torch.ops.aten.alias.default(alias_499);  alias_499 = None
        mul_1089 = torch.ops.aten.mul.Tensor(mul_1088, alias_500);  mul_1088 = None
        sum_133 = torch.ops.aten.sum.dim_IntList(mul_1089, [-1], True)
        mul_1090 = torch.ops.aten.mul.Tensor(alias_500, sum_133);  alias_500 = sum_133 = None
        sub_87 = torch.ops.aten.sub.Tensor(mul_1089, mul_1090);  mul_1089 = mul_1090 = None
        add_382 = torch.ops.aten.add.Tensor(add_372, sub_87);  add_372 = None
        sum_134 = torch.ops.aten.sum.dim_IntList(add_382, [0], True);  add_382 = None
        squeeze_2 = torch.ops.aten.squeeze.dim(sum_134, 0);  sum_134 = None
        permute_1046 = torch.ops.aten.permute.default(squeeze_2, [1, 2, 0]);  squeeze_2 = None
        view_740 = torch.ops.aten.view.default(permute_1046, [16384, 6])
        new_zeros_2 = torch.ops.aten.new_zeros.default(permute_1046, [32, 6], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  permute_1046 = None
        ne_2 = torch.ops.aten.ne.Scalar(view_741, -1)
        unsqueeze_21 = torch.ops.aten.unsqueeze.default(ne_2, 1);  ne_2 = None
        expand_139 = torch.ops.aten.expand.default(unsqueeze_21, [16384, 6]);  unsqueeze_21 = None
        full_like_4 = torch.ops.aten.full_like.default(view_740, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_501 = torch.ops.aten.alias.default(full_like_4);  full_like_4 = None
        where_4 = torch.ops.aten.where.self(expand_139, view_740, alias_501);  expand_139 = view_740 = alias_501 = None
        index_put_2 = torch.ops.aten.index_put.default(new_zeros_2, [view_741], where_4, True);  new_zeros_2 = view_741 = where_4 = None
        view_742 = torch.ops.aten.view.default(sub_87, [12, 128, 128]);  sub_87 = None
        bmm_142 = torch.ops.aten.bmm.default(permute_1047, view_742);  permute_1047 = None
        bmm_143 = torch.ops.aten.bmm.default(view_742, permute_1048);  view_742 = permute_1048 = None
        view_743 = torch.ops.aten.view.default(bmm_142, [2, 6, 64, 128]);  bmm_142 = None
        view_744 = torch.ops.aten.view.default(bmm_143, [2, 6, 128, 64]);  bmm_143 = None
        permute_1049 = torch.ops.aten.permute.default(view_743, [0, 1, 3, 2]);  view_743 = None
        permute_1050 = torch.ops.aten.permute.default(view_738, [0, 2, 1, 3]);  view_738 = None
        clone_189 = torch.ops.aten.clone.default(permute_1050, memory_format = torch.contiguous_format);  permute_1050 = None
        _unsafe_view_358 = torch.ops.aten._unsafe_view.default(clone_189, [2, 128, 384]);  clone_189 = None
        view_745 = torch.ops.aten.view.default(_unsafe_view_358, [256, 384]);  _unsafe_view_358 = None
        permute_1051 = torch.ops.aten.permute.default(view_745, [1, 0])
        mm_429 = torch.ops.aten.mm.default(permute_1051, view_1);  permute_1051 = None
        permute_1052 = torch.ops.aten.permute.default(mm_429, [1, 0]);  mm_429 = None
        mm_430 = torch.ops.aten.mm.default(view_745, permute_1053);  view_745 = permute_1053 = None
        view_746 = torch.ops.aten.view.default(mm_430, [2, 128, 512]);  mm_430 = None
        permute_1054 = torch.ops.aten.permute.default(permute_1052, [1, 0]);  permute_1052 = None
        permute_1055 = torch.ops.aten.permute.default(permute_1049, [0, 2, 1, 3]);  permute_1049 = None
        view_747 = torch.ops.aten.view.default(permute_1055, [2, 128, 384]);  permute_1055 = None
        clone_190 = torch.ops.aten.clone.default(view_747, memory_format = torch.contiguous_format);  view_747 = None
        _unsafe_view_359 = torch.ops.aten._unsafe_view.default(clone_190, [256, 384]);  clone_190 = None
        permute_1056 = torch.ops.aten.permute.default(_unsafe_view_359, [1, 0])
        mm_431 = torch.ops.aten.mm.default(permute_1056, view_1);  permute_1056 = None
        permute_1057 = torch.ops.aten.permute.default(mm_431, [1, 0]);  mm_431 = None
        mm_432 = torch.ops.aten.mm.default(_unsafe_view_359, permute_1058);  _unsafe_view_359 = permute_1058 = None
        view_748 = torch.ops.aten.view.default(mm_432, [2, 128, 512]);  mm_432 = None
        add_383 = torch.ops.aten.add.Tensor(view_746, view_748);  view_746 = view_748 = None
        permute_1059 = torch.ops.aten.permute.default(permute_1057, [1, 0]);  permute_1057 = None
        permute_1060 = torch.ops.aten.permute.default(view_744, [0, 2, 1, 3]);  view_744 = None
        clone_191 = torch.ops.aten.clone.default(permute_1060, memory_format = torch.contiguous_format);  permute_1060 = None
        _unsafe_view_360 = torch.ops.aten._unsafe_view.default(clone_191, [2, 128, 384]);  clone_191 = None
        view_749 = torch.ops.aten.view.default(_unsafe_view_360, [256, 384]);  _unsafe_view_360 = None
        permute_1061 = torch.ops.aten.permute.default(view_749, [1, 0])
        mm_433 = torch.ops.aten.mm.default(permute_1061, view_1);  permute_1061 = view_1 = None
        permute_1062 = torch.ops.aten.permute.default(mm_433, [1, 0]);  mm_433 = None
        mm_434 = torch.ops.aten.mm.default(view_749, permute_1063);  view_749 = permute_1063 = None
        view_750 = torch.ops.aten.view.default(mm_434, [2, 128, 512]);  mm_434 = None
        add_384 = torch.ops.aten.add.Tensor(add_383, view_750);  add_383 = view_750 = None
        permute_1064 = torch.ops.aten.permute.default(permute_1062, [1, 0]);  permute_1062 = None
        mul_1091 = torch.ops.aten.mul.Tensor(add_384, primals_1);  primals_1 = None
        mul_1092 = torch.ops.aten.mul.Tensor(add_384, mul_3);  add_384 = mul_3 = None
        sum_135 = torch.ops.aten.sum.dim_IntList(mul_1092, [0, 1], True);  mul_1092 = None
        view_751 = torch.ops.aten.view.default(sum_135, [512]);  sum_135 = None
        mul_1093 = torch.ops.aten.mul.Tensor(mul_1091, mul_2)
        mul_1094 = torch.ops.aten.mul.Tensor(mul_1091, reciprocal);  mul_1091 = reciprocal = None
        sum_136 = torch.ops.aten.sum.dim_IntList(mul_1093, [2], True);  mul_1093 = None
        add_385 = torch.ops.aten.add.Tensor(add_381, mul_1094);  add_381 = mul_1094 = None
        alias_502 = torch.ops.aten.alias.default(alias_5);  alias_5 = None
        alias_503 = torch.ops.aten.alias.default(alias_502);  alias_502 = None
        pow_157 = torch.ops.aten.pow.Tensor_Scalar(alias_503, 3);  alias_503 = None
        mul_1095 = torch.ops.aten.mul.Scalar(sum_136, -0.5);  sum_136 = None
        mul_1096 = torch.ops.aten.mul.Tensor(mul_1095, pow_157);  mul_1095 = pow_157 = None
        expand_140 = torch.ops.aten.expand.default(mul_1096, [2, 128, 512]);  mul_1096 = None
        div_70 = torch.ops.aten.div.Scalar(expand_140, 512);  expand_140 = None
        pow_158 = torch.ops.aten.pow.Tensor_Scalar(mul_2, 1.0);  mul_2 = None
        mul_1097 = torch.ops.aten.mul.Scalar(pow_158, 2.0);  pow_158 = None
        mul_1098 = torch.ops.aten.mul.Tensor(div_70, mul_1097);  div_70 = mul_1097 = None
        add_386 = torch.ops.aten.add.Tensor(add_385, mul_1098);  add_385 = mul_1098 = None
        _to_copy_114 = torch.ops.aten._to_copy.default(gt, dtype = torch.float32);  gt = None
        mul_1099 = torch.ops.aten.mul.Tensor(_to_copy_114, 1.1111111111111112);  _to_copy_114 = None
        mul_1100 = torch.ops.aten.mul.Tensor(add_386, mul_1099);  add_386 = mul_1099 = None
        view_752 = torch.ops.aten.view.default(mul_1100, [256, 512])
        new_zeros_3 = torch.ops.aten.new_zeros.default(mul_1100, [250112, 512], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  mul_1100 = None
        ne_3 = torch.ops.aten.ne.Scalar(view_753, -1)
        unsqueeze_22 = torch.ops.aten.unsqueeze.default(ne_3, 1);  ne_3 = None
        expand_141 = torch.ops.aten.expand.default(unsqueeze_22, [256, 512]);  unsqueeze_22 = None
        full_like_5 = torch.ops.aten.full_like.default(view_752, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_504 = torch.ops.aten.alias.default(full_like_5);  full_like_5 = None
        where_5 = torch.ops.aten.where.self(expand_141, view_752, alias_504);  expand_141 = view_752 = alias_504 = None
        index_put_3 = torch.ops.aten.index_put.default(new_zeros_3, [view_753], where_5, True);  new_zeros_3 = view_753 = where_5 = None
        add_387 = torch.ops.aten.add.Tensor(index_put_1, index_put_3);  index_put_1 = index_put_3 = None
        return [view_751, view_734, view_727, view_712, view_705, view_690, view_683, view_668, view_661, view_646, view_639, view_624, view_617, view_602, view_595, view_580, view_573, view_570, view_553, view_538, view_531, view_516, view_501, view_494, view_479, view_464, view_457, view_442, view_427, view_420, view_405, view_390, view_383, view_368, view_353, view_346, view_331, view_316, view_309, view_294, view_279, view_272, add_387, permute_1064, permute_1059, permute_1054, index_put_2, permute_1042, permute_1038, permute_1034, permute_1030, permute_1026, permute_1021, permute_1016, permute_1005, permute_1001, permute_997, permute_993, permute_989, permute_984, permute_979, permute_968, permute_964, permute_960, permute_956, permute_952, permute_947, permute_942, permute_931, permute_927, permute_923, permute_919, permute_915, permute_910, permute_905, permute_894, permute_890, permute_886, permute_882, permute_878, permute_873, permute_868, permute_857, permute_853, permute_849, permute_845, permute_841, permute_836, permute_831, permute_820, permute_816, permute_812, permute_808, permute_804, permute_799, permute_794, permute_783, permute_779, permute_775, permute_771, permute_767, permute_762, permute_757, index_put, permute_745, permute_741, permute_736, permute_731, permute_720, permute_716, permute_712, permute_708, permute_704, permute_699, permute_694, permute_683, permute_679, permute_674, permute_669, permute_658, permute_654, permute_650, permute_646, permute_642, permute_637, permute_632, permute_621, permute_617, permute_612, permute_607, permute_596, permute_592, permute_588, permute_584, permute_580, permute_575, permute_570, permute_559, permute_555, permute_550, permute_545, permute_534, permute_530, permute_526, permute_522, permute_518, permute_513, permute_508, permute_497, permute_493, permute_488, permute_483, permute_472, permute_468, permute_464, permute_460, permute_456, permute_451, permute_446, permute_435, permute_431, permute_426, permute_421, permute_410, permute_406, permute_402, permute_398, permute_394, permute_389, permute_384, permute_373, permute_369, permute_364, permute_359, permute_348, permute_344, permute_340, permute_336, permute_332, permute_327, permute_322, permute_311, permute_307, permute_302, permute_297, permute_286, permute_282, permute_278, permute_274, permute_270, None, None, None]
        
args = [((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.float32, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.bool, 'cuda'), ((2, 128, 1), (128, 1, 256), torch.float32, 'cuda'), ((2, 6, 128, 128), (98304, 16384, 128, 1), torch.float32, 'cuda'), ((), (), torch.int64, 'cuda'), ((256, 384), (384, 1), torch.float32, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.bool, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.float32, 'cuda'), ((2, 128, 1), (128, 1, 256), torch.float32, 'cuda'), ((256, 1024), (1024, 1), torch.float32, 'cuda'), ((2, 128, 1024), (131072, 1024, 1), torch.float32, 'cuda'), ((256, 1024), (1024, 1), torch.float32, 'cuda'), ((2, 128, 1024), (131072, 1024, 1), torch.bool, 'cuda'), ((256, 1024), (1024, 1), torch.float32, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.bool, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.float32, 'cuda'), ((2, 128, 1), (128, 1, 256), torch.float32, 'cuda'), ((2, 6, 128, 128), (98304, 16384, 128, 1), torch.float32, 'cuda'), ((256, 384), (384, 1), torch.float32, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.bool, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.float32, 'cuda'), ((2, 128, 1), (128, 1, 256), torch.float32, 'cuda'), ((256, 1024), (1024, 1), torch.float32, 'cuda'), ((2, 128, 1024), (131072, 1024, 1), torch.float32, 'cuda'), ((256, 1024), (1024, 1), torch.float32, 'cuda'), ((2, 128, 1024), (131072, 1024, 1), torch.bool, 'cuda'), ((256, 1024), (1024, 1), torch.float32, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.bool, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.float32, 'cuda'), ((2, 128, 1), (128, 1, 256), torch.float32, 'cuda'), ((2, 6, 128, 128), (98304, 16384, 128, 1), torch.float32, 'cuda'), ((256, 384), (384, 1), torch.float32, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.bool, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.float32, 'cuda'), ((2, 128, 1), (128, 1, 256), torch.float32, 'cuda'), ((256, 1024), (1024, 1), torch.float32, 'cuda'), ((2, 128, 1024), (131072, 1024, 1), torch.float32, 'cuda'), ((256, 1024), (1024, 1), torch.float32, 'cuda'), ((2, 128, 1024), (131072, 1024, 1), torch.bool, 'cuda'), ((256, 1024), (1024, 1), torch.float32, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.bool, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.float32, 'cuda'), ((2, 128, 1), (128, 1, 256), torch.float32, 'cuda'), ((2, 6, 128, 128), (98304, 16384, 128, 1), torch.float32, 'cuda'), ((256, 384), (384, 1), torch.float32, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.bool, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.float32, 'cuda'), ((2, 128, 1), (128, 1, 256), torch.float32, 'cuda'), ((256, 1024), (1024, 1), torch.float32, 'cuda'), ((2, 128, 1024), (131072, 1024, 1), torch.float32, 'cuda'), ((256, 1024), (1024, 1), torch.float32, 'cuda'), ((2, 128, 1024), (131072, 1024, 1), torch.bool, 'cuda'), ((256, 1024), (1024, 1), torch.float32, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.bool, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.float32, 'cuda'), ((2, 128, 1), (128, 1, 256), torch.float32, 'cuda'), ((2, 6, 128, 128), (98304, 16384, 128, 1), torch.float32, 'cuda'), ((256, 384), (384, 1), torch.float32, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.bool, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.float32, 'cuda'), ((2, 128, 1), (128, 1, 256), torch.float32, 'cuda'), ((256, 1024), (1024, 1), torch.float32, 'cuda'), ((2, 128, 1024), (131072, 1024, 1), torch.float32, 'cuda'), ((256, 1024), (1024, 1), torch.float32, 'cuda'), ((2, 128, 1024), (131072, 1024, 1), torch.bool, 'cuda'), ((256, 1024), (1024, 1), torch.float32, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.bool, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.float32, 'cuda'), ((2, 128, 1), (128, 1, 256), torch.float32, 'cuda'), ((2, 6, 128, 128), (98304, 16384, 128, 1), torch.float32, 'cuda'), ((256, 384), (384, 1), torch.float32, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.bool, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.float32, 'cuda'), ((2, 128, 1), (128, 1, 256), torch.float32, 'cuda'), ((256, 1024), (1024, 1), torch.float32, 'cuda'), ((2, 128, 1024), (131072, 1024, 1), torch.float32, 'cuda'), ((256, 1024), (1024, 1), torch.float32, 'cuda'), ((2, 128, 1024), (131072, 1024, 1), torch.bool, 'cuda'), ((256, 1024), (1024, 1), torch.float32, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.bool, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.float32, 'cuda'), ((2, 128, 1), (128, 1, 256), torch.float32, 'cuda'), ((2, 6, 128, 128), (98304, 16384, 128, 1), torch.float32, 'cuda'), ((256, 384), (384, 1), torch.float32, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.bool, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.float32, 'cuda'), ((2, 128, 1), (128, 1, 256), torch.float32, 'cuda'), ((256, 1024), (1024, 1), torch.float32, 'cuda'), ((2, 128, 1024), (131072, 1024, 1), torch.float32, 'cuda'), ((256, 1024), (1024, 1), torch.float32, 'cuda'), ((2, 128, 1024), (131072, 1024, 1), torch.bool, 'cuda'), ((256, 1024), (1024, 1), torch.float32, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.bool, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.float32, 'cuda'), ((2, 128, 1), (128, 1, 256), torch.float32, 'cuda'), ((2, 6, 128, 128), (98304, 16384, 128, 1), torch.float32, 'cuda'), ((256, 384), (384, 1), torch.float32, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.bool, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.float32, 'cuda'), ((2, 128, 1), (128, 1, 256), torch.float32, 'cuda'), ((256, 1024), (1024, 1), torch.float32, 'cuda'), ((2, 128, 1024), (131072, 1024, 1), torch.float32, 'cuda'), ((256, 1024), (1024, 1), torch.float32, 'cuda'), ((2, 128, 1024), (131072, 1024, 1), torch.bool, 'cuda'), ((256, 1024), (1024, 1), torch.float32, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.bool, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.float32, 'cuda'), ((2, 128, 1), (128, 1, 256), torch.float32, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.bool, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.float32, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.bool, 'cuda'), ((2, 128, 1), (128, 1, 256), torch.float32, 'cuda'), ((2, 6, 128, 128), (98304, 16384, 128, 1), torch.float32, 'cuda'), ((256, 384), (384, 1), torch.float32, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.bool, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.float32, 'cuda'), ((2, 128, 1), (128, 1, 256), torch.float32, 'cuda'), ((2, 6, 128, 128), (98304, 16384, 128, 1), torch.float32, 'cuda'), ((256, 384), (384, 1), torch.float32, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.bool, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.float32, 'cuda'), ((2, 128, 1), (128, 1, 256), torch.float32, 'cuda'), ((256, 1024), (1024, 1), torch.float32, 'cuda'), ((2, 128, 1024), (131072, 1024, 1), torch.float32, 'cuda'), ((256, 1024), (1024, 1), torch.float32, 'cuda'), ((2, 128, 1024), (131072, 1024, 1), torch.bool, 'cuda'), ((256, 1024), (1024, 1), torch.float32, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.bool, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.float32, 'cuda'), ((2, 128, 1), (128, 1, 256), torch.float32, 'cuda'), ((2, 6, 128, 128), (98304, 16384, 128, 1), torch.float32, 'cuda'), ((256, 384), (384, 1), torch.float32, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.bool, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.float32, 'cuda'), ((2, 128, 1), (128, 1, 256), torch.float32, 'cuda'), ((2, 6, 128, 128), (98304, 16384, 128, 1), torch.float32, 'cuda'), ((256, 384), (384, 1), torch.float32, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.bool, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.float32, 'cuda'), ((2, 128, 1), (128, 1, 256), torch.float32, 'cuda'), ((256, 1024), (1024, 1), torch.float32, 'cuda'), ((2, 128, 1024), (131072, 1024, 1), torch.float32, 'cuda'), ((256, 1024), (1024, 1), torch.float32, 'cuda'), ((2, 128, 1024), (131072, 1024, 1), torch.bool, 'cuda'), ((256, 1024), (1024, 1), torch.float32, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.bool, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.float32, 'cuda'), ((2, 128, 1), (128, 1, 256), torch.float32, 'cuda'), ((2, 6, 128, 128), (98304, 16384, 128, 1), torch.float32, 'cuda'), ((256, 384), (384, 1), torch.float32, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.bool, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.float32, 'cuda'), ((2, 128, 1), (128, 1, 256), torch.float32, 'cuda'), ((2, 6, 128, 128), (98304, 16384, 128, 1), torch.float32, 'cuda'), ((256, 384), (384, 1), torch.float32, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.bool, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.float32, 'cuda'), ((2, 128, 1), (128, 1, 256), torch.float32, 'cuda'), ((256, 1024), (1024, 1), torch.float32, 'cuda'), ((2, 128, 1024), (131072, 1024, 1), torch.float32, 'cuda'), ((256, 1024), (1024, 1), torch.float32, 'cuda'), ((2, 128, 1024), (131072, 1024, 1), torch.bool, 'cuda'), ((256, 1024), (1024, 1), torch.float32, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.bool, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.float32, 'cuda'), ((2, 128, 1), (128, 1, 256), torch.float32, 'cuda'), ((2, 6, 128, 128), (98304, 16384, 128, 1), torch.float32, 'cuda'), ((256, 384), (384, 1), torch.float32, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.bool, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.float32, 'cuda'), ((2, 128, 1), (128, 1, 256), torch.float32, 'cuda'), ((2, 6, 128, 128), (98304, 16384, 128, 1), torch.float32, 'cuda'), ((256, 384), (384, 1), torch.float32, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.bool, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.float32, 'cuda'), ((2, 128, 1), (128, 1, 256), torch.float32, 'cuda'), ((256, 1024), (1024, 1), torch.float32, 'cuda'), ((2, 128, 1024), (131072, 1024, 1), torch.float32, 'cuda'), ((256, 1024), (1024, 1), torch.float32, 'cuda'), ((2, 128, 1024), (131072, 1024, 1), torch.bool, 'cuda'), ((256, 1024), (1024, 1), torch.float32, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.bool, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.float32, 'cuda'), ((2, 128, 1), (128, 1, 256), torch.float32, 'cuda'), ((2, 6, 128, 128), (98304, 16384, 128, 1), torch.float32, 'cuda'), ((256, 384), (384, 1), torch.float32, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.bool, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.float32, 'cuda'), ((2, 128, 1), (128, 1, 256), torch.float32, 'cuda'), ((2, 6, 128, 128), (98304, 16384, 128, 1), torch.float32, 'cuda'), ((256, 384), (384, 1), torch.float32, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.bool, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.float32, 'cuda'), ((2, 128, 1), (128, 1, 256), torch.float32, 'cuda'), ((256, 1024), (1024, 1), torch.float32, 'cuda'), ((2, 128, 1024), (131072, 1024, 1), torch.float32, 'cuda'), ((256, 1024), (1024, 1), torch.float32, 'cuda'), ((2, 128, 1024), (131072, 1024, 1), torch.bool, 'cuda'), ((256, 1024), (1024, 1), torch.float32, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.bool, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.float32, 'cuda'), ((2, 128, 1), (128, 1, 256), torch.float32, 'cuda'), ((2, 6, 128, 128), (98304, 16384, 128, 1), torch.float32, 'cuda'), ((256, 384), (384, 1), torch.float32, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.bool, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.float32, 'cuda'), ((2, 128, 1), (128, 1, 256), torch.float32, 'cuda'), ((2, 6, 128, 128), (98304, 16384, 128, 1), torch.float32, 'cuda'), ((256, 384), (384, 1), torch.float32, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.bool, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.float32, 'cuda'), ((2, 128, 1), (128, 1, 256), torch.float32, 'cuda'), ((256, 1024), (1024, 1), torch.float32, 'cuda'), ((2, 128, 1024), (131072, 1024, 1), torch.float32, 'cuda'), ((256, 1024), (1024, 1), torch.float32, 'cuda'), ((2, 128, 1024), (131072, 1024, 1), torch.bool, 'cuda'), ((256, 1024), (1024, 1), torch.float32, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.bool, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.float32, 'cuda'), ((2, 128, 1), (128, 1, 256), torch.float32, 'cuda'), ((2, 6, 128, 128), (98304, 16384, 128, 1), torch.float32, 'cuda'), ((256, 384), (384, 1), torch.float32, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.bool, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.float32, 'cuda'), ((2, 128, 1), (128, 1, 256), torch.float32, 'cuda'), ((2, 6, 128, 128), (98304, 16384, 128, 1), torch.float32, 'cuda'), ((256, 384), (384, 1), torch.float32, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.bool, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.float32, 'cuda'), ((2, 128, 1), (128, 1, 256), torch.float32, 'cuda'), ((256, 1024), (1024, 1), torch.float32, 'cuda'), ((2, 128, 1024), (131072, 1024, 1), torch.float32, 'cuda'), ((256, 1024), (1024, 1), torch.float32, 'cuda'), ((2, 128, 1024), (131072, 1024, 1), torch.bool, 'cuda'), ((256, 1024), (1024, 1), torch.float32, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.bool, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.float32, 'cuda'), ((2, 128, 1), (128, 1, 256), torch.float32, 'cuda'), ((2, 6, 128, 128), (98304, 16384, 128, 1), torch.float32, 'cuda'), ((256, 384), (384, 1), torch.float32, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.bool, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.float32, 'cuda'), ((2, 128, 1), (128, 1, 256), torch.float32, 'cuda'), ((2, 6, 128, 128), (98304, 16384, 128, 1), torch.float32, 'cuda'), ((256, 384), (384, 1), torch.float32, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.bool, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.float32, 'cuda'), ((2, 128, 1), (128, 1, 256), torch.float32, 'cuda'), ((256, 1024), (1024, 1), torch.float32, 'cuda'), ((2, 128, 1024), (131072, 1024, 1), torch.float32, 'cuda'), ((256, 1024), (1024, 1), torch.float32, 'cuda'), ((2, 128, 1024), (131072, 1024, 1), torch.bool, 'cuda'), ((256, 1024), (1024, 1), torch.float32, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.bool, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.float32, 'cuda'), ((2, 128, 1), (128, 1, 256), torch.float32, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.bool, 'cuda'), ((256, 512), (512, 1), torch.float32, 'cuda'), ((256, 250112), (250112, 1), torch.float32, 'cuda'), ((256, 1), (1, 1), torch.int64, 'cuda'), ((250112, 512), (512, 1), torch.float32, 'cuda'), ((512, 1024), (1024, 1), torch.float32, 'cuda'), ((1024, 512), (512, 1), torch.float32, 'cuda'), ((1024, 512), (512, 1), torch.float32, 'cuda'), ((512, 384), (384, 1), torch.float32, 'cuda'), ((12, 128, 128), (16384, 1, 128), torch.float32, 'cuda'), ((12, 64, 128), (8192, 1, 64), torch.float32, 'cuda'), ((12, 64, 128), (8192, 1, 64), torch.float32, 'cuda'), ((12, 128, 64), (8192, 1, 128), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((512, 384), (384, 1), torch.float32, 'cuda'), ((12, 128, 128), (16384, 1, 128), torch.float32, 'cuda'), ((12, 64, 128), (8192, 1, 64), torch.float32, 'cuda'), ((12, 64, 128), (8192, 1, 64), torch.float32, 'cuda'), ((12, 128, 64), (8192, 1, 128), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((512, 1024), (1024, 1), torch.float32, 'cuda'), ((1024, 512), (512, 1), torch.float32, 'cuda'), ((1024, 512), (512, 1), torch.float32, 'cuda'), ((512, 384), (384, 1), torch.float32, 'cuda'), ((12, 128, 128), (16384, 1, 128), torch.float32, 'cuda'), ((12, 64, 128), (8192, 1, 64), torch.float32, 'cuda'), ((12, 64, 128), (8192, 1, 64), torch.float32, 'cuda'), ((12, 128, 64), (8192, 1, 128), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((512, 384), (384, 1), torch.float32, 'cuda'), ((12, 128, 128), (16384, 1, 128), torch.float32, 'cuda'), ((12, 64, 128), (8192, 1, 64), torch.float32, 'cuda'), ((12, 64, 128), (8192, 1, 64), torch.float32, 'cuda'), ((12, 128, 64), (8192, 1, 128), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((512, 1024), (1024, 1), torch.float32, 'cuda'), ((1024, 512), (512, 1), torch.float32, 'cuda'), ((1024, 512), (512, 1), torch.float32, 'cuda'), ((512, 384), (384, 1), torch.float32, 'cuda'), ((12, 128, 128), (16384, 1, 128), torch.float32, 'cuda'), ((12, 64, 128), (8192, 1, 64), torch.float32, 'cuda'), ((12, 64, 128), (8192, 1, 64), torch.float32, 'cuda'), ((12, 128, 64), (8192, 1, 128), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((512, 384), (384, 1), torch.float32, 'cuda'), ((12, 128, 128), (16384, 1, 128), torch.float32, 'cuda'), ((12, 64, 128), (8192, 1, 64), torch.float32, 'cuda'), ((12, 64, 128), (8192, 1, 64), torch.float32, 'cuda'), ((12, 128, 64), (8192, 1, 128), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((512, 1024), (1024, 1), torch.float32, 'cuda'), ((1024, 512), (512, 1), torch.float32, 'cuda'), ((1024, 512), (512, 1), torch.float32, 'cuda'), ((512, 384), (384, 1), torch.float32, 'cuda'), ((12, 128, 128), (16384, 1, 128), torch.float32, 'cuda'), ((12, 64, 128), (8192, 1, 64), torch.float32, 'cuda'), ((12, 64, 128), (8192, 1, 64), torch.float32, 'cuda'), ((12, 128, 64), (8192, 1, 128), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((512, 384), (384, 1), torch.float32, 'cuda'), ((12, 128, 128), (16384, 1, 128), torch.float32, 'cuda'), ((12, 64, 128), (8192, 1, 64), torch.float32, 'cuda'), ((12, 64, 128), (8192, 1, 64), torch.float32, 'cuda'), ((12, 128, 64), (8192, 1, 128), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((512, 1024), (1024, 1), torch.float32, 'cuda'), ((1024, 512), (512, 1), torch.float32, 'cuda'), ((1024, 512), (512, 1), torch.float32, 'cuda'), ((512, 384), (384, 1), torch.float32, 'cuda'), ((12, 128, 128), (16384, 1, 128), torch.float32, 'cuda'), ((12, 64, 128), (8192, 1, 64), torch.float32, 'cuda'), ((12, 64, 128), (8192, 1, 64), torch.float32, 'cuda'), ((12, 128, 64), (8192, 1, 128), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((512, 384), (384, 1), torch.float32, 'cuda'), ((12, 128, 128), (16384, 1, 128), torch.float32, 'cuda'), ((12, 64, 128), (8192, 1, 64), torch.float32, 'cuda'), ((12, 64, 128), (8192, 1, 64), torch.float32, 'cuda'), ((12, 128, 64), (8192, 1, 128), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((512, 1024), (1024, 1), torch.float32, 'cuda'), ((1024, 512), (512, 1), torch.float32, 'cuda'), ((1024, 512), (512, 1), torch.float32, 'cuda'), ((512, 384), (384, 1), torch.float32, 'cuda'), ((12, 128, 128), (16384, 1, 128), torch.float32, 'cuda'), ((12, 64, 128), (8192, 1, 64), torch.float32, 'cuda'), ((12, 64, 128), (8192, 1, 64), torch.float32, 'cuda'), ((12, 128, 64), (8192, 1, 128), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((512, 384), (384, 1), torch.float32, 'cuda'), ((12, 128, 128), (16384, 1, 128), torch.float32, 'cuda'), ((12, 64, 128), (8192, 1, 64), torch.float32, 'cuda'), ((12, 64, 128), (8192, 1, 64), torch.float32, 'cuda'), ((12, 128, 64), (8192, 1, 128), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((512, 1024), (1024, 1), torch.float32, 'cuda'), ((1024, 512), (512, 1), torch.float32, 'cuda'), ((1024, 512), (512, 1), torch.float32, 'cuda'), ((512, 384), (384, 1), torch.float32, 'cuda'), ((12, 128, 128), (16384, 1, 128), torch.float32, 'cuda'), ((12, 64, 128), (8192, 1, 64), torch.float32, 'cuda'), ((12, 64, 128), (8192, 1, 64), torch.float32, 'cuda'), ((12, 128, 64), (8192, 1, 128), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((512, 384), (384, 1), torch.float32, 'cuda'), ((12, 128, 128), (16384, 1, 128), torch.float32, 'cuda'), ((12, 64, 128), (8192, 1, 64), torch.float32, 'cuda'), ((12, 64, 128), (8192, 1, 64), torch.float32, 'cuda'), ((12, 128, 64), (8192, 1, 128), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((512, 1024), (1024, 1), torch.float32, 'cuda'), ((1024, 512), (512, 1), torch.float32, 'cuda'), ((1024, 512), (512, 1), torch.float32, 'cuda'), ((512, 384), (384, 1), torch.float32, 'cuda'), ((12, 128, 128), (16384, 1, 128), torch.float32, 'cuda'), ((12, 64, 128), (8192, 1, 64), torch.float32, 'cuda'), ((12, 64, 128), (8192, 1, 64), torch.float32, 'cuda'), ((12, 128, 64), (8192, 1, 128), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((512, 384), (384, 1), torch.float32, 'cuda'), ((12, 128, 128), (16384, 1, 128), torch.float32, 'cuda'), ((12, 64, 128), (8192, 1, 64), torch.float32, 'cuda'), ((16384,), (1,), torch.int64, 'cuda'), ((12, 64, 128), (8192, 1, 64), torch.float32, 'cuda'), ((12, 128, 64), (8192, 1, 128), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((256,), (1,), torch.int64, 'cuda'), ((512, 1024), (1024, 1), torch.float32, 'cuda'), ((1024, 512), (512, 1), torch.float32, 'cuda'), ((1024, 512), (512, 1), torch.float32, 'cuda'), ((512, 384), (384, 1), torch.float32, 'cuda'), ((12, 128, 128), (16384, 1, 128), torch.float32, 'cuda'), ((12, 64, 128), (8192, 1, 64), torch.float32, 'cuda'), ((12, 64, 128), (8192, 1, 64), torch.float32, 'cuda'), ((12, 128, 64), (8192, 1, 128), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((512, 1024), (1024, 1), torch.float32, 'cuda'), ((1024, 512), (512, 1), torch.float32, 'cuda'), ((1024, 512), (512, 1), torch.float32, 'cuda'), ((512, 384), (384, 1), torch.float32, 'cuda'), ((12, 128, 128), (16384, 1, 128), torch.float32, 'cuda'), ((12, 64, 128), (8192, 1, 64), torch.float32, 'cuda'), ((12, 64, 128), (8192, 1, 64), torch.float32, 'cuda'), ((12, 128, 64), (8192, 1, 128), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((512, 1024), (1024, 1), torch.float32, 'cuda'), ((1024, 512), (512, 1), torch.float32, 'cuda'), ((1024, 512), (512, 1), torch.float32, 'cuda'), ((512, 384), (384, 1), torch.float32, 'cuda'), ((12, 128, 128), (16384, 1, 128), torch.float32, 'cuda'), ((12, 64, 128), (8192, 1, 64), torch.float32, 'cuda'), ((12, 64, 128), (8192, 1, 64), torch.float32, 'cuda'), ((12, 128, 64), (8192, 1, 128), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((512, 1024), (1024, 1), torch.float32, 'cuda'), ((1024, 512), (512, 1), torch.float32, 'cuda'), ((1024, 512), (512, 1), torch.float32, 'cuda'), ((512, 384), (384, 1), torch.float32, 'cuda'), ((12, 128, 128), (16384, 1, 128), torch.float32, 'cuda'), ((12, 64, 128), (8192, 1, 64), torch.float32, 'cuda'), ((12, 64, 128), (8192, 1, 64), torch.float32, 'cuda'), ((12, 128, 64), (8192, 1, 128), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((512, 1024), (1024, 1), torch.float32, 'cuda'), ((1024, 512), (512, 1), torch.float32, 'cuda'), ((1024, 512), (512, 1), torch.float32, 'cuda'), ((512, 384), (384, 1), torch.float32, 'cuda'), ((12, 128, 128), (16384, 1, 128), torch.float32, 'cuda'), ((12, 64, 128), (8192, 1, 64), torch.float32, 'cuda'), ((12, 64, 128), (8192, 1, 64), torch.float32, 'cuda'), ((12, 128, 64), (8192, 1, 128), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((512, 1024), (1024, 1), torch.float32, 'cuda'), ((1024, 512), (512, 1), torch.float32, 'cuda'), ((1024, 512), (512, 1), torch.float32, 'cuda'), ((512, 384), (384, 1), torch.float32, 'cuda'), ((12, 128, 128), (16384, 1, 128), torch.float32, 'cuda'), ((12, 64, 128), (8192, 1, 64), torch.float32, 'cuda'), ((12, 64, 128), (8192, 1, 64), torch.float32, 'cuda'), ((12, 128, 64), (8192, 1, 128), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((512, 1024), (1024, 1), torch.float32, 'cuda'), ((1024, 512), (512, 1), torch.float32, 'cuda'), ((1024, 512), (512, 1), torch.float32, 'cuda'), ((512, 384), (384, 1), torch.float32, 'cuda'), ((12, 128, 128), (16384, 1, 128), torch.float32, 'cuda'), ((12, 64, 128), (8192, 1, 64), torch.float32, 'cuda'), ((12, 64, 128), (8192, 1, 64), torch.float32, 'cuda'), ((12, 128, 64), (8192, 1, 128), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((512, 1024), (1024, 1), torch.float32, 'cuda'), ((1024, 512), (512, 1), torch.float32, 'cuda'), ((1024, 512), (512, 1), torch.float32, 'cuda'), ((512, 384), (384, 1), torch.float32, 'cuda'), ((12, 128, 128), (16384, 1, 128), torch.float32, 'cuda'), ((12, 64, 128), (8192, 1, 64), torch.float32, 'cuda'), ((16384,), (1,), torch.int64, 'cuda'), ((12, 64, 128), (8192, 1, 64), torch.float32, 'cuda'), ((12, 128, 64), (8192, 1, 128), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((384, 512), (512, 1), torch.float32, 'cuda'), ((256,), (1,), torch.int64, 'cuda'), ((), (), torch.float32, 'cuda'), ((2, 128, 250112), (32014336, 250112, 1), torch.float32, 'cuda'), ((2, 6, 128, 64), (49152, 8192, 64, 1), torch.float32, 'cuda'), ((2, 6, 128, 64), (49152, 8192, 64, 1), torch.float32, 'cuda'), ((2, 6, 128, 64), (49152, 8192, 64, 1), torch.float32, 'cuda'), ((2, 6, 128, 64), (49152, 8192, 64, 1), torch.float32, 'cuda'), ((2, 6, 128, 64), (49152, 8192, 64, 1), torch.float32, 'cuda'), ((2, 6, 128, 64), (49152, 8192, 64, 1), torch.float32, 'cuda'), ((2, 6, 128, 64), (49152, 8192, 64, 1), torch.float32, 'cuda'), ((2, 6, 128, 64), (49152, 8192, 64, 1), torch.float32, 'cuda'), ((2, 6, 128, 64), (49152, 8192, 64, 1), torch.float32, 'cuda'), ((2, 6, 128, 64), (49152, 8192, 64, 1), torch.float32, 'cuda'), ((2, 6, 128, 64), (49152, 8192, 64, 1), torch.float32, 'cuda'), ((2, 6, 128, 64), (49152, 8192, 64, 1), torch.float32, 'cuda'), ((2, 6, 128, 64), (49152, 8192, 64, 1), torch.float32, 'cuda'), ((2, 6, 128, 64), (49152, 8192, 64, 1), torch.float32, 'cuda'), ((2, 6, 128, 64), (49152, 8192, 64, 1), torch.float32, 'cuda'), ((2, 6, 128, 64), (49152, 8192, 64, 1), torch.float32, 'cuda'), ((2, 6, 128, 64), (49152, 8192, 64, 1), torch.float32, 'cuda'), ((2, 6, 128, 64), (49152, 8192, 64, 1), torch.float32, 'cuda'), ((2, 6, 128, 64), (49152, 8192, 64, 1), torch.float32, 'cuda'), ((2, 6, 128, 64), (49152, 8192, 64, 1), torch.float32, 'cuda'), ((2, 6, 128, 64), (49152, 8192, 64, 1), torch.float32, 'cuda'), ((2, 6, 128, 64), (49152, 8192, 64, 1), torch.float32, 'cuda'), ((2, 6, 128, 64), (49152, 8192, 64, 1), torch.float32, 'cuda'), ((2, 6, 128, 64), (49152, 8192, 64, 1), torch.float32, 'cuda'), ((2, 6, 128, 64), (49152, 8192, 64, 1), torch.float32, 'cuda'), ((2, 6, 128, 64), (49152, 8192, 64, 1), torch.float32, 'cuda'), ((2, 6, 128, 64), (49152, 8192, 64, 1), torch.float32, 'cuda'), ((2, 6, 128, 64), (49152, 8192, 64, 1), torch.float32, 'cuda'), ((2, 6, 128, 64), (49152, 8192, 64, 1), torch.float32, 'cuda'), ((2, 6, 128, 64), (49152, 8192, 64, 1), torch.float32, 'cuda'), ((2, 6, 128, 64), (49152, 8192, 64, 1), torch.float32, 'cuda'), ((2, 6, 128, 64), (49152, 8192, 64, 1), torch.float32, 'cuda'), ((2, 128, 512), (65536, 512, 1), torch.float32, 'cuda')]
args = [rand_strided(sh, st, dt, dev) for (sh, st, dt, dev) in args]
mod = make_fx(Repro().to(device="cuda"))(*args)

from torch._inductor.compile_fx import compile_fx_inner
from torch._dynamo.debug_utils import same_two_models

compiled = compile_fx_inner(mod, args)
compiled(args)
