
import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
from torch.fx.experimental.proxy_tensor import make_fx

# torch version: 1.14.0a0+gitb05e2e0
# torch cuda version: 11.6
# torch git version: b05e2e0a970d489b404568ed0cb2e09248253e44


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

    
    
    def forward(self, arg16_1, arg18_1, arg20_1, arg22_1, arg24_1, arg26_1, arg28_1, arg30_1, arg32_1, arg34_1, arg36_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg93_1, arg95_1, arg97_1, arg99_1, arg101_1, arg103_1, arg105_1, arg107_1, arg109_1, arg111_1, arg113_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg197_1, arg203_1, arg209_1, arg215_1, arg221_1, arg227_1, arg233_1, arg239_1, arg245_1, arg251_1, arg257_1, arg263_1, arg269_1, arg270_1, arg271_1, arg272_1, arg275_1, arg276_1, arg277_1, arg278_1, arg281_1, arg282_1, arg283_1, arg284_1, arg287_1, arg288_1, arg289_1, arg290_1, arg293_1, arg294_1, arg295_1, arg296_1, arg299_1, arg300_1, arg301_1, arg302_1, arg305_1, arg306_1, arg307_1, arg308_1, arg311_1, arg312_1, arg313_1, arg314_1, arg317_1, arg318_1, arg319_1, arg320_1, arg323_1, arg324_1, arg325_1, arg326_1, arg329_1, arg330_1, arg331_1, arg332_1, arg335_1, arg336_1, arg337_1, arg338_1, arg341_1, arg342_1, arg343_1, arg344_1, arg347_1, arg348_1, arg349_1, arg350_1, arg353_1, arg354_1, arg355_1, arg356_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg368_1, arg369_1, arg370_1, arg371_1, arg374_1, arg375_1, arg376_1, arg377_1, arg381_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg497_1):
        _unsafe_view_6 = torch.ops.aten._unsafe_view.default(arg203_1, [32, 14, 14, 512]);  arg203_1 = None
        add_10 = torch.ops.aten.add.Tensor(_unsafe_view_6, arg93_1);  _unsafe_view_6 = arg93_1 = None
        permute_13 = torch.ops.aten.permute.default(add_10, [0, 3, 1, 2]);  add_10 = None
        view_6 = torch.ops.aten.view.default(arg16_1, [1, 512, 1, 1]);  arg16_1 = None
        mul_11 = torch.ops.aten.mul.Tensor(permute_13, view_6);  permute_13 = view_6 = None
        add_11 = torch.ops.aten.add.Tensor(mul_11, arg197_1);  mul_11 = arg197_1 = None
        _unsafe_view_7 = torch.ops.aten._unsafe_view.default(arg209_1, [32, 14, 14, 512]);  arg209_1 = None
        add_12 = torch.ops.aten.add.Tensor(_unsafe_view_7, arg95_1);  _unsafe_view_7 = arg95_1 = None
        permute_15 = torch.ops.aten.permute.default(add_12, [0, 3, 1, 2]);  add_12 = None
        view_7 = torch.ops.aten.view.default(arg18_1, [1, 512, 1, 1]);  arg18_1 = None
        mul_13 = torch.ops.aten.mul.Tensor(permute_15, view_7);  permute_15 = view_7 = None
        add_13 = torch.ops.aten.add.Tensor(mul_13, add_11);  mul_13 = add_11 = None
        _unsafe_view_8 = torch.ops.aten._unsafe_view.default(arg215_1, [32, 14, 14, 512]);  arg215_1 = None
        add_14 = torch.ops.aten.add.Tensor(_unsafe_view_8, arg97_1);  _unsafe_view_8 = arg97_1 = None
        permute_17 = torch.ops.aten.permute.default(add_14, [0, 3, 1, 2]);  add_14 = None
        view_8 = torch.ops.aten.view.default(arg20_1, [1, 512, 1, 1]);  arg20_1 = None
        mul_15 = torch.ops.aten.mul.Tensor(permute_17, view_8);  permute_17 = view_8 = None
        add_15 = torch.ops.aten.add.Tensor(mul_15, add_13);  mul_15 = add_13 = None
        _unsafe_view_9 = torch.ops.aten._unsafe_view.default(arg221_1, [32, 14, 14, 512]);  arg221_1 = None
        add_16 = torch.ops.aten.add.Tensor(_unsafe_view_9, arg99_1);  _unsafe_view_9 = arg99_1 = None
        permute_19 = torch.ops.aten.permute.default(add_16, [0, 3, 1, 2]);  add_16 = None
        view_9 = torch.ops.aten.view.default(arg22_1, [1, 512, 1, 1]);  arg22_1 = None
        mul_17 = torch.ops.aten.mul.Tensor(permute_19, view_9);  permute_19 = view_9 = None
        add_17 = torch.ops.aten.add.Tensor(mul_17, add_15);  mul_17 = add_15 = None
        _unsafe_view_10 = torch.ops.aten._unsafe_view.default(arg227_1, [32, 14, 14, 512]);  arg227_1 = None
        add_18 = torch.ops.aten.add.Tensor(_unsafe_view_10, arg101_1);  _unsafe_view_10 = arg101_1 = None
        permute_21 = torch.ops.aten.permute.default(add_18, [0, 3, 1, 2]);  add_18 = None
        view_10 = torch.ops.aten.view.default(arg24_1, [1, 512, 1, 1]);  arg24_1 = None
        mul_19 = torch.ops.aten.mul.Tensor(permute_21, view_10);  permute_21 = view_10 = None
        add_19 = torch.ops.aten.add.Tensor(mul_19, add_17);  mul_19 = add_17 = None
        _unsafe_view_11 = torch.ops.aten._unsafe_view.default(arg233_1, [32, 14, 14, 512]);  arg233_1 = None
        add_20 = torch.ops.aten.add.Tensor(_unsafe_view_11, arg103_1);  _unsafe_view_11 = arg103_1 = None
        permute_23 = torch.ops.aten.permute.default(add_20, [0, 3, 1, 2]);  add_20 = None
        view_11 = torch.ops.aten.view.default(arg26_1, [1, 512, 1, 1]);  arg26_1 = None
        mul_21 = torch.ops.aten.mul.Tensor(permute_23, view_11);  permute_23 = view_11 = None
        add_21 = torch.ops.aten.add.Tensor(mul_21, add_19);  mul_21 = add_19 = None
        _unsafe_view_12 = torch.ops.aten._unsafe_view.default(arg239_1, [32, 14, 14, 512]);  arg239_1 = None
        add_22 = torch.ops.aten.add.Tensor(_unsafe_view_12, arg105_1);  _unsafe_view_12 = arg105_1 = None
        permute_25 = torch.ops.aten.permute.default(add_22, [0, 3, 1, 2]);  add_22 = None
        view_12 = torch.ops.aten.view.default(arg28_1, [1, 512, 1, 1]);  arg28_1 = None
        mul_23 = torch.ops.aten.mul.Tensor(permute_25, view_12);  permute_25 = view_12 = None
        add_23 = torch.ops.aten.add.Tensor(mul_23, add_21);  mul_23 = add_21 = None
        _unsafe_view_13 = torch.ops.aten._unsafe_view.default(arg245_1, [32, 14, 14, 512]);  arg245_1 = None
        add_24 = torch.ops.aten.add.Tensor(_unsafe_view_13, arg107_1);  _unsafe_view_13 = arg107_1 = None
        permute_27 = torch.ops.aten.permute.default(add_24, [0, 3, 1, 2]);  add_24 = None
        view_13 = torch.ops.aten.view.default(arg30_1, [1, 512, 1, 1]);  arg30_1 = None
        mul_25 = torch.ops.aten.mul.Tensor(permute_27, view_13);  permute_27 = view_13 = None
        add_25 = torch.ops.aten.add.Tensor(mul_25, add_23);  mul_25 = add_23 = None
        _unsafe_view_14 = torch.ops.aten._unsafe_view.default(arg251_1, [32, 14, 14, 512]);  arg251_1 = None
        add_26 = torch.ops.aten.add.Tensor(_unsafe_view_14, arg109_1);  _unsafe_view_14 = arg109_1 = None
        permute_29 = torch.ops.aten.permute.default(add_26, [0, 3, 1, 2]);  add_26 = None
        view_14 = torch.ops.aten.view.default(arg32_1, [1, 512, 1, 1]);  arg32_1 = None
        mul_27 = torch.ops.aten.mul.Tensor(permute_29, view_14);  permute_29 = view_14 = None
        add_27 = torch.ops.aten.add.Tensor(mul_27, add_25);  mul_27 = add_25 = None
        _unsafe_view_15 = torch.ops.aten._unsafe_view.default(arg257_1, [32, 14, 14, 512]);  arg257_1 = None
        add_28 = torch.ops.aten.add.Tensor(_unsafe_view_15, arg111_1);  _unsafe_view_15 = arg111_1 = None
        permute_31 = torch.ops.aten.permute.default(add_28, [0, 3, 1, 2]);  add_28 = None
        view_15 = torch.ops.aten.view.default(arg34_1, [1, 512, 1, 1]);  arg34_1 = None
        mul_29 = torch.ops.aten.mul.Tensor(permute_31, view_15);  permute_31 = view_15 = None
        add_29 = torch.ops.aten.add.Tensor(mul_29, add_27);  mul_29 = add_27 = None
        _unsafe_view_16 = torch.ops.aten._unsafe_view.default(arg263_1, [32, 14, 14, 512]);  arg263_1 = None
        add_30 = torch.ops.aten.add.Tensor(_unsafe_view_16, arg113_1);  _unsafe_view_16 = arg113_1 = None
        permute_33 = torch.ops.aten.permute.default(add_30, [0, 3, 1, 2]);  add_30 = None
        view_16 = torch.ops.aten.view.default(arg36_1, [1, 512, 1, 1]);  arg36_1 = None
        mul_31 = torch.ops.aten.mul.Tensor(permute_33, view_16);  permute_33 = view_16 = None
        add_31 = torch.ops.aten.add.Tensor(mul_31, add_29);  mul_31 = add_29 = None
        _unsafe_view_17 = torch.ops.aten._unsafe_view.default(arg269_1, [32, 14, 14, 512]);  arg269_1 = None
        add_32 = torch.ops.aten.add.Tensor(_unsafe_view_17, arg115_1);  _unsafe_view_17 = arg115_1 = None
        permute_35 = torch.ops.aten.permute.default(add_32, [0, 3, 1, 2]);  add_32 = None
        view_17 = torch.ops.aten.view.default(arg38_1, [1, 512, 1, 1]);  arg38_1 = None
        mul_33 = torch.ops.aten.mul.Tensor(permute_35, view_17);  permute_35 = view_17 = None
        add_33 = torch.ops.aten.add.Tensor(mul_33, add_31);  mul_33 = add_31 = None
        permute_36 = torch.ops.aten.permute.default(arg270_1, [0, 2, 3, 1]);  arg270_1 = None
        sub_18 = torch.ops.aten.sub.Tensor(permute_36, arg271_1);  permute_36 = arg271_1 = None
        mul_34 = torch.ops.aten.mul.Tensor(sub_18, arg272_1);  sub_18 = arg272_1 = None
        _unsafe_view_18 = torch.ops.aten._unsafe_view.default(arg275_1, [32, 14, 14, 512]);  arg275_1 = None
        add_34 = torch.ops.aten.add.Tensor(_unsafe_view_18, arg117_1);  _unsafe_view_18 = arg117_1 = None
        permute_37 = torch.ops.aten.permute.default(add_34, [0, 3, 1, 2]);  add_34 = None
        view_18 = torch.ops.aten.view.default(arg40_1, [1, 512, 1, 1]);  arg40_1 = None
        mul_35 = torch.ops.aten.mul.Tensor(permute_37, view_18);  permute_37 = view_18 = None
        add_35 = torch.ops.aten.add.Tensor(mul_35, add_33);  mul_35 = add_33 = None
        permute_38 = torch.ops.aten.permute.default(arg276_1, [0, 2, 3, 1]);  arg276_1 = None
        sub_19 = torch.ops.aten.sub.Tensor(permute_38, arg277_1);  permute_38 = arg277_1 = None
        mul_36 = torch.ops.aten.mul.Tensor(sub_19, arg278_1);  sub_19 = arg278_1 = None
        _unsafe_view_19 = torch.ops.aten._unsafe_view.default(arg281_1, [32, 14, 14, 512]);  arg281_1 = None
        add_36 = torch.ops.aten.add.Tensor(_unsafe_view_19, arg119_1);  _unsafe_view_19 = arg119_1 = None
        permute_39 = torch.ops.aten.permute.default(add_36, [0, 3, 1, 2]);  add_36 = None
        view_19 = torch.ops.aten.view.default(arg42_1, [1, 512, 1, 1]);  arg42_1 = None
        mul_37 = torch.ops.aten.mul.Tensor(permute_39, view_19);  permute_39 = view_19 = None
        add_37 = torch.ops.aten.add.Tensor(mul_37, add_35);  mul_37 = add_35 = None
        permute_40 = torch.ops.aten.permute.default(arg282_1, [0, 2, 3, 1]);  arg282_1 = None
        sub_20 = torch.ops.aten.sub.Tensor(permute_40, arg283_1);  permute_40 = arg283_1 = None
        mul_38 = torch.ops.aten.mul.Tensor(sub_20, arg284_1);  sub_20 = arg284_1 = None
        _unsafe_view_20 = torch.ops.aten._unsafe_view.default(arg287_1, [32, 14, 14, 512]);  arg287_1 = None
        add_38 = torch.ops.aten.add.Tensor(_unsafe_view_20, arg121_1);  _unsafe_view_20 = arg121_1 = None
        permute_41 = torch.ops.aten.permute.default(add_38, [0, 3, 1, 2]);  add_38 = None
        view_20 = torch.ops.aten.view.default(arg44_1, [1, 512, 1, 1]);  arg44_1 = None
        mul_39 = torch.ops.aten.mul.Tensor(permute_41, view_20);  permute_41 = view_20 = None
        add_39 = torch.ops.aten.add.Tensor(mul_39, add_37);  mul_39 = add_37 = None
        permute_42 = torch.ops.aten.permute.default(arg288_1, [0, 2, 3, 1]);  arg288_1 = None
        sub_21 = torch.ops.aten.sub.Tensor(permute_42, arg289_1);  permute_42 = arg289_1 = None
        mul_40 = torch.ops.aten.mul.Tensor(sub_21, arg290_1);  sub_21 = arg290_1 = None
        _unsafe_view_21 = torch.ops.aten._unsafe_view.default(arg293_1, [32, 14, 14, 512]);  arg293_1 = None
        add_40 = torch.ops.aten.add.Tensor(_unsafe_view_21, arg123_1);  _unsafe_view_21 = arg123_1 = None
        permute_43 = torch.ops.aten.permute.default(add_40, [0, 3, 1, 2]);  add_40 = None
        view_21 = torch.ops.aten.view.default(arg46_1, [1, 512, 1, 1]);  arg46_1 = None
        mul_41 = torch.ops.aten.mul.Tensor(permute_43, view_21);  permute_43 = view_21 = None
        add_41 = torch.ops.aten.add.Tensor(mul_41, add_39);  mul_41 = add_39 = None
        permute_44 = torch.ops.aten.permute.default(arg294_1, [0, 2, 3, 1]);  arg294_1 = None
        sub_22 = torch.ops.aten.sub.Tensor(permute_44, arg295_1);  permute_44 = arg295_1 = None
        mul_42 = torch.ops.aten.mul.Tensor(sub_22, arg296_1);  sub_22 = arg296_1 = None
        _unsafe_view_22 = torch.ops.aten._unsafe_view.default(arg299_1, [32, 14, 14, 512]);  arg299_1 = None
        add_42 = torch.ops.aten.add.Tensor(_unsafe_view_22, arg125_1);  _unsafe_view_22 = arg125_1 = None
        permute_45 = torch.ops.aten.permute.default(add_42, [0, 3, 1, 2]);  add_42 = None
        view_22 = torch.ops.aten.view.default(arg48_1, [1, 512, 1, 1]);  arg48_1 = None
        mul_43 = torch.ops.aten.mul.Tensor(permute_45, view_22);  permute_45 = view_22 = None
        add_43 = torch.ops.aten.add.Tensor(mul_43, add_41);  mul_43 = add_41 = None
        permute_46 = torch.ops.aten.permute.default(arg300_1, [0, 2, 3, 1]);  arg300_1 = None
        sub_23 = torch.ops.aten.sub.Tensor(permute_46, arg301_1);  permute_46 = arg301_1 = None
        mul_44 = torch.ops.aten.mul.Tensor(sub_23, arg302_1);  sub_23 = arg302_1 = None
        _unsafe_view_23 = torch.ops.aten._unsafe_view.default(arg305_1, [32, 14, 14, 512]);  arg305_1 = None
        add_44 = torch.ops.aten.add.Tensor(_unsafe_view_23, arg127_1);  _unsafe_view_23 = arg127_1 = None
        permute_47 = torch.ops.aten.permute.default(add_44, [0, 3, 1, 2]);  add_44 = None
        view_23 = torch.ops.aten.view.default(arg50_1, [1, 512, 1, 1]);  arg50_1 = None
        mul_45 = torch.ops.aten.mul.Tensor(permute_47, view_23);  permute_47 = view_23 = None
        add_45 = torch.ops.aten.add.Tensor(mul_45, add_43);  mul_45 = add_43 = None
        permute_48 = torch.ops.aten.permute.default(arg306_1, [0, 2, 3, 1]);  arg306_1 = None
        sub_24 = torch.ops.aten.sub.Tensor(permute_48, arg307_1);  permute_48 = arg307_1 = None
        mul_46 = torch.ops.aten.mul.Tensor(sub_24, arg308_1);  sub_24 = arg308_1 = None
        _unsafe_view_24 = torch.ops.aten._unsafe_view.default(arg311_1, [32, 14, 14, 512]);  arg311_1 = None
        add_46 = torch.ops.aten.add.Tensor(_unsafe_view_24, arg129_1);  _unsafe_view_24 = arg129_1 = None
        permute_49 = torch.ops.aten.permute.default(add_46, [0, 3, 1, 2]);  add_46 = None
        view_24 = torch.ops.aten.view.default(arg52_1, [1, 512, 1, 1]);  arg52_1 = None
        mul_47 = torch.ops.aten.mul.Tensor(permute_49, view_24);  permute_49 = view_24 = None
        add_47 = torch.ops.aten.add.Tensor(mul_47, add_45);  mul_47 = add_45 = None
        permute_50 = torch.ops.aten.permute.default(arg312_1, [0, 2, 3, 1]);  arg312_1 = None
        sub_25 = torch.ops.aten.sub.Tensor(permute_50, arg313_1);  permute_50 = arg313_1 = None
        mul_48 = torch.ops.aten.mul.Tensor(sub_25, arg314_1);  sub_25 = arg314_1 = None
        _unsafe_view_25 = torch.ops.aten._unsafe_view.default(arg317_1, [32, 14, 14, 512]);  arg317_1 = None
        add_48 = torch.ops.aten.add.Tensor(_unsafe_view_25, arg131_1);  _unsafe_view_25 = arg131_1 = None
        permute_51 = torch.ops.aten.permute.default(add_48, [0, 3, 1, 2]);  add_48 = None
        view_25 = torch.ops.aten.view.default(arg54_1, [1, 512, 1, 1]);  arg54_1 = None
        mul_49 = torch.ops.aten.mul.Tensor(permute_51, view_25);  permute_51 = view_25 = None
        add_49 = torch.ops.aten.add.Tensor(mul_49, add_47);  mul_49 = add_47 = None
        permute_52 = torch.ops.aten.permute.default(arg318_1, [0, 2, 3, 1]);  arg318_1 = None
        sub_26 = torch.ops.aten.sub.Tensor(permute_52, arg319_1);  permute_52 = arg319_1 = None
        mul_50 = torch.ops.aten.mul.Tensor(sub_26, arg320_1);  sub_26 = arg320_1 = None
        _unsafe_view_26 = torch.ops.aten._unsafe_view.default(arg323_1, [32, 14, 14, 512]);  arg323_1 = None
        add_50 = torch.ops.aten.add.Tensor(_unsafe_view_26, arg133_1);  _unsafe_view_26 = arg133_1 = None
        permute_53 = torch.ops.aten.permute.default(add_50, [0, 3, 1, 2]);  add_50 = None
        view_26 = torch.ops.aten.view.default(arg56_1, [1, 512, 1, 1]);  arg56_1 = None
        mul_51 = torch.ops.aten.mul.Tensor(permute_53, view_26);  permute_53 = view_26 = None
        add_51 = torch.ops.aten.add.Tensor(mul_51, add_49);  mul_51 = add_49 = None
        permute_54 = torch.ops.aten.permute.default(arg324_1, [0, 2, 3, 1]);  arg324_1 = None
        sub_27 = torch.ops.aten.sub.Tensor(permute_54, arg325_1);  permute_54 = arg325_1 = None
        mul_52 = torch.ops.aten.mul.Tensor(sub_27, arg326_1);  sub_27 = arg326_1 = None
        _unsafe_view_27 = torch.ops.aten._unsafe_view.default(arg329_1, [32, 14, 14, 512]);  arg329_1 = None
        add_52 = torch.ops.aten.add.Tensor(_unsafe_view_27, arg135_1);  _unsafe_view_27 = arg135_1 = None
        permute_55 = torch.ops.aten.permute.default(add_52, [0, 3, 1, 2]);  add_52 = None
        view_27 = torch.ops.aten.view.default(arg58_1, [1, 512, 1, 1]);  arg58_1 = None
        mul_53 = torch.ops.aten.mul.Tensor(permute_55, view_27);  permute_55 = view_27 = None
        add_53 = torch.ops.aten.add.Tensor(mul_53, add_51);  mul_53 = add_51 = None
        permute_56 = torch.ops.aten.permute.default(arg330_1, [0, 2, 3, 1]);  arg330_1 = None
        sub_28 = torch.ops.aten.sub.Tensor(permute_56, arg331_1);  permute_56 = arg331_1 = None
        mul_54 = torch.ops.aten.mul.Tensor(sub_28, arg332_1);  sub_28 = arg332_1 = None
        _unsafe_view_28 = torch.ops.aten._unsafe_view.default(arg335_1, [32, 14, 14, 512]);  arg335_1 = None
        add_54 = torch.ops.aten.add.Tensor(_unsafe_view_28, arg137_1);  _unsafe_view_28 = arg137_1 = None
        permute_57 = torch.ops.aten.permute.default(add_54, [0, 3, 1, 2]);  add_54 = None
        view_28 = torch.ops.aten.view.default(arg60_1, [1, 512, 1, 1]);  arg60_1 = None
        mul_55 = torch.ops.aten.mul.Tensor(permute_57, view_28);  permute_57 = view_28 = None
        add_55 = torch.ops.aten.add.Tensor(mul_55, add_53);  mul_55 = add_53 = None
        permute_58 = torch.ops.aten.permute.default(arg336_1, [0, 2, 3, 1]);  arg336_1 = None
        sub_29 = torch.ops.aten.sub.Tensor(permute_58, arg337_1);  permute_58 = arg337_1 = None
        mul_56 = torch.ops.aten.mul.Tensor(sub_29, arg338_1);  sub_29 = arg338_1 = None
        _unsafe_view_29 = torch.ops.aten._unsafe_view.default(arg341_1, [32, 14, 14, 512]);  arg341_1 = None
        add_56 = torch.ops.aten.add.Tensor(_unsafe_view_29, arg139_1);  _unsafe_view_29 = arg139_1 = None
        permute_59 = torch.ops.aten.permute.default(add_56, [0, 3, 1, 2]);  add_56 = None
        view_29 = torch.ops.aten.view.default(arg62_1, [1, 512, 1, 1]);  arg62_1 = None
        mul_57 = torch.ops.aten.mul.Tensor(permute_59, view_29);  permute_59 = view_29 = None
        add_57 = torch.ops.aten.add.Tensor(mul_57, add_55);  mul_57 = add_55 = None
        permute_60 = torch.ops.aten.permute.default(arg342_1, [0, 2, 3, 1]);  arg342_1 = None
        sub_30 = torch.ops.aten.sub.Tensor(permute_60, arg343_1);  permute_60 = arg343_1 = None
        mul_58 = torch.ops.aten.mul.Tensor(sub_30, arg344_1);  sub_30 = arg344_1 = None
        _unsafe_view_30 = torch.ops.aten._unsafe_view.default(arg347_1, [32, 14, 14, 512]);  arg347_1 = None
        add_58 = torch.ops.aten.add.Tensor(_unsafe_view_30, arg141_1);  _unsafe_view_30 = arg141_1 = None
        permute_61 = torch.ops.aten.permute.default(add_58, [0, 3, 1, 2]);  add_58 = None
        view_30 = torch.ops.aten.view.default(arg64_1, [1, 512, 1, 1]);  arg64_1 = None
        mul_59 = torch.ops.aten.mul.Tensor(permute_61, view_30);  permute_61 = view_30 = None
        add_59 = torch.ops.aten.add.Tensor(mul_59, add_57);  mul_59 = add_57 = None
        permute_62 = torch.ops.aten.permute.default(arg348_1, [0, 2, 3, 1]);  arg348_1 = None
        sub_31 = torch.ops.aten.sub.Tensor(permute_62, arg349_1);  permute_62 = arg349_1 = None
        mul_60 = torch.ops.aten.mul.Tensor(sub_31, arg350_1);  sub_31 = arg350_1 = None
        _unsafe_view_31 = torch.ops.aten._unsafe_view.default(arg353_1, [32, 14, 14, 512]);  arg353_1 = None
        add_60 = torch.ops.aten.add.Tensor(_unsafe_view_31, arg143_1);  _unsafe_view_31 = arg143_1 = None
        permute_63 = torch.ops.aten.permute.default(add_60, [0, 3, 1, 2]);  add_60 = None
        view_31 = torch.ops.aten.view.default(arg66_1, [1, 512, 1, 1]);  arg66_1 = None
        mul_61 = torch.ops.aten.mul.Tensor(permute_63, view_31);  permute_63 = view_31 = None
        add_61 = torch.ops.aten.add.Tensor(mul_61, add_59);  mul_61 = add_59 = None
        permute_64 = torch.ops.aten.permute.default(arg354_1, [0, 2, 3, 1]);  arg354_1 = None
        sub_32 = torch.ops.aten.sub.Tensor(permute_64, arg355_1);  permute_64 = arg355_1 = None
        mul_62 = torch.ops.aten.mul.Tensor(sub_32, arg356_1);  sub_32 = arg356_1 = None
        view_32 = torch.ops.aten.view.default(arg68_1, [1, 512, 1, 1]);  arg68_1 = None
        permute_66 = torch.ops.aten.permute.default(arg363_1, [0, 2, 3, 1]);  arg363_1 = None
        sub_33 = torch.ops.aten.sub.Tensor(permute_66, arg364_1);  permute_66 = arg364_1 = None
        mul_63 = torch.ops.aten.mul.Tensor(sub_33, arg365_1);  sub_33 = arg365_1 = None
        _unsafe_view_33 = torch.ops.aten._unsafe_view.default(arg368_1, [32, 7, 7, 1024]);  arg368_1 = None
        add_63 = torch.ops.aten.add.Tensor(_unsafe_view_33, arg148_1);  _unsafe_view_33 = arg148_1 = None
        permute_67 = torch.ops.aten.permute.default(add_63, [0, 3, 1, 2]);  add_63 = None
        view_33 = torch.ops.aten.view.default(arg71_1, [1, 1024, 1, 1]);  arg71_1 = None
        mul_64 = torch.ops.aten.mul.Tensor(permute_67, view_33);  permute_67 = view_33 = None
        add_64 = torch.ops.aten.add.Tensor(mul_64, arg362_1);  mul_64 = arg362_1 = None
        permute_68 = torch.ops.aten.permute.default(arg369_1, [0, 2, 3, 1]);  arg369_1 = None
        sub_34 = torch.ops.aten.sub.Tensor(permute_68, arg370_1);  permute_68 = arg370_1 = None
        mul_65 = torch.ops.aten.mul.Tensor(sub_34, arg371_1);  sub_34 = arg371_1 = None
        _unsafe_view_34 = torch.ops.aten._unsafe_view.default(arg374_1, [32, 7, 7, 1024]);  arg374_1 = None
        add_65 = torch.ops.aten.add.Tensor(_unsafe_view_34, arg150_1);  _unsafe_view_34 = arg150_1 = None
        permute_69 = torch.ops.aten.permute.default(add_65, [0, 3, 1, 2]);  add_65 = None
        view_34 = torch.ops.aten.view.default(arg73_1, [1, 1024, 1, 1]);  arg73_1 = None
        mul_66 = torch.ops.aten.mul.Tensor(permute_69, view_34);  permute_69 = None
        add_66 = torch.ops.aten.add.Tensor(mul_66, add_64);  mul_66 = add_64 = None
        permute_70 = torch.ops.aten.permute.default(arg375_1, [0, 2, 3, 1]);  arg375_1 = None
        sub_35 = torch.ops.aten.sub.Tensor(permute_70, arg376_1);  permute_70 = arg376_1 = None
        mul_67 = torch.ops.aten.mul.Tensor(sub_35, arg377_1);  sub_35 = None
        view_35 = torch.ops.aten.view.default(arg75_1, [1, 1024, 1, 1]);  arg75_1 = None
        mm = torch.ops.aten.mm.default(arg497_1, arg383_1);  arg497_1 = arg383_1 = None
        view_37 = torch.ops.aten.view.default(mm, [32, 1024, 1, 1]);  mm = None
        permute_75 = torch.ops.aten.permute.default(view_37, [0, 2, 3, 1]);  view_37 = None
        mul_68 = torch.ops.aten.mul.Tensor(permute_75, arg76_1);  permute_75 = arg76_1 = None
        mul_69 = torch.ops.aten.mul.Tensor(mul_68, 1024)
        sum_2 = torch.ops.aten.sum.dim_IntList(mul_68, [3], True)
        mul_70 = torch.ops.aten.mul.Tensor(mul_68, arg381_1);  mul_68 = None
        sum_3 = torch.ops.aten.sum.dim_IntList(mul_70, [3], True);  mul_70 = None
        mul_71 = torch.ops.aten.mul.Tensor(arg381_1, sum_3);  arg381_1 = sum_3 = None
        sub_36 = torch.ops.aten.sub.Tensor(mul_69, sum_2);  mul_69 = sum_2 = None
        sub_37 = torch.ops.aten.sub.Tensor(sub_36, mul_71);  sub_36 = mul_71 = None
        mul_72 = torch.ops.aten.mul.Tensor(arg384_1, sub_37);  arg384_1 = sub_37 = None
        permute_76 = torch.ops.aten.permute.default(mul_72, [0, 3, 1, 2]);  mul_72 = None
        squeeze = torch.ops.aten.squeeze.dim(permute_76, 3);  permute_76 = None
        squeeze_1 = torch.ops.aten.squeeze.dim(squeeze, 2);  squeeze = None
        new_zeros = torch.ops.aten.new_zeros.default(squeeze_1, [32768])
        as_strided_scatter = torch.ops.aten.as_strided_scatter.default(new_zeros, squeeze_1, [32, 1024], [1024, 1], 0);  new_zeros = squeeze_1 = None
        as_strided = torch.ops.aten.as_strided.default(as_strided_scatter, [32, 1024, 1, 1], [1024, 1, 1, 1], 0);  as_strided_scatter = None
        expand = torch.ops.aten.expand.default(as_strided, [32, 1024, 7, 7]);  as_strided = None
        div = torch.ops.aten.div.Scalar(expand, 49);  expand = None
        mul_75 = torch.ops.aten.mul.Tensor(div, view_35);  view_35 = None
        permute_77 = torch.ops.aten.permute.default(mul_75, [0, 2, 3, 1]);  mul_75 = None
        clone = torch.ops.aten.clone.default(permute_77, memory_format = torch.contiguous_format);  permute_77 = None
        _unsafe_view_36 = torch.ops.aten._unsafe_view.default(clone, [1568, 1024]);  clone = None
        mm_3 = torch.ops.aten.mm.default(_unsafe_view_36, arg385_1);  _unsafe_view_36 = arg385_1 = None
        view_40 = torch.ops.aten.view.default(mm_3, [32, 7, 7, 4096]);  mm_3 = None
        mul_76 = torch.ops.aten.mul.Tensor(view_40, arg386_1);  view_40 = arg386_1 = None
        view_42 = torch.ops.aten.view.default(mul_76, [1568, 4096]);  mul_76 = None
        mm_5 = torch.ops.aten.mm.default(view_42, arg387_1);  view_42 = arg387_1 = None
        view_43 = torch.ops.aten.view.default(mm_5, [32, 7, 7, 1024]);  mm_5 = None
        mul_77 = torch.ops.aten.mul.Tensor(view_43, arg74_1);  view_43 = arg74_1 = None
        mul_78 = torch.ops.aten.mul.Tensor(mul_77, 1024)
        sum_9 = torch.ops.aten.sum.dim_IntList(mul_77, [3], True)
        mul_79 = torch.ops.aten.mul.Tensor(mul_77, mul_67);  mul_77 = None
        sum_10 = torch.ops.aten.sum.dim_IntList(mul_79, [3], True);  mul_79 = None
        mul_80 = torch.ops.aten.mul.Tensor(mul_67, sum_10);  mul_67 = sum_10 = None
        sub_38 = torch.ops.aten.sub.Tensor(mul_78, sum_9);  mul_78 = sum_9 = None
        sub_39 = torch.ops.aten.sub.Tensor(sub_38, mul_80);  sub_38 = mul_80 = None
        div_1 = torch.ops.aten.div.Tensor(arg377_1, 1024);  arg377_1 = None
        mul_81 = torch.ops.aten.mul.Tensor(div_1, sub_39);  div_1 = sub_39 = None
        permute_84 = torch.ops.aten.permute.default(mul_81, [0, 3, 1, 2]);  mul_81 = None
        convolution_backward = torch.ops.aten.convolution_backward.default(permute_84, add_66, arg151_1, [1024], [1, 1], [3, 3], [1, 1], False, [0, 0], 1024, [True, True, True]);  permute_84 = add_66 = arg151_1 = None
        getitem = convolution_backward[0];  convolution_backward = None
        add_68 = torch.ops.aten.add.Tensor(div, getitem);  div = getitem = None
        mul_84 = torch.ops.aten.mul.Tensor(add_68, view_34);  add_68 = view_34 = None
        return (mul_84,)
        
args = [((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((1024,), (1,), torch.float32, 'cuda'), ((1024,), (1,), torch.float32, 'cuda'), ((1024,), (1,), torch.float32, 'cuda'), ((1024,), (1,), torch.float32, 'cuda'), ((1024,), (1,), torch.float32, 'cuda'), ((1024,), (1,), torch.float32, 'cuda'), ((1024,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512, 1, 7, 7), (49, 49, 7, 1), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512, 1, 7, 7), (49, 49, 7, 1), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512, 1, 7, 7), (49, 49, 7, 1), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512, 1, 7, 7), (49, 49, 7, 1), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512, 1, 7, 7), (49, 49, 7, 1), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512, 1, 7, 7), (49, 49, 7, 1), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512, 1, 7, 7), (49, 49, 7, 1), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512, 1, 7, 7), (49, 49, 7, 1), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512, 1, 7, 7), (49, 49, 7, 1), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512, 1, 7, 7), (49, 49, 7, 1), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512, 1, 7, 7), (49, 49, 7, 1), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512, 1, 7, 7), (49, 49, 7, 1), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512, 1, 7, 7), (49, 49, 7, 1), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512, 1, 7, 7), (49, 49, 7, 1), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512, 1, 7, 7), (49, 49, 7, 1), torch.float32, 'cuda'), ((1024, 512, 2, 2), (2048, 4, 2, 1), torch.float32, 'cuda'), ((1024, 1, 7, 7), (49, 49, 7, 1), torch.float32, 'cuda'), ((1024,), (1,), torch.float32, 'cuda'), ((1024, 1, 7, 7), (49, 49, 7, 1), torch.float32, 'cuda'), ((1024,), (1,), torch.float32, 'cuda'), ((1024, 1, 7, 7), (49, 49, 7, 1), torch.float32, 'cuda'), ((32, 512, 14, 14), (100352, 1, 7168, 512), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((32, 512, 14, 14), (100352, 196, 14, 1), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((32, 512, 14, 14), (100352, 196, 14, 1), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((32, 512, 14, 14), (100352, 196, 14, 1), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((32, 512, 14, 14), (100352, 196, 14, 1), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((32, 512, 14, 14), (100352, 196, 14, 1), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((32, 512, 14, 14), (100352, 196, 14, 1), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((32, 512, 14, 14), (100352, 196, 14, 1), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((32, 512, 14, 14), (100352, 196, 14, 1), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((32, 512, 14, 14), (100352, 196, 14, 1), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((32, 512, 14, 14), (100352, 196, 14, 1), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((32, 512, 14, 14), (100352, 196, 14, 1), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((32, 512, 14, 14), (100352, 196, 14, 1), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((32, 512, 14, 14), (100352, 196, 14, 1), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((32, 512, 14, 14), (100352, 196, 14, 1), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((32, 512, 14, 14), (100352, 196, 14, 1), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((32, 14, 14, 512), (100352, 7168, 512, 1), torch.float32, 'cuda'), ((32, 512, 14, 14), (100352, 1, 7168, 512), torch.float32, 'cuda'), ((32, 1024, 7, 7), (50176, 1, 7168, 1024), torch.float32, 'cuda'), ((32, 1024, 7, 7), (50176, 49, 7, 1), torch.float32, 'cuda'), ((32, 7, 7, 1), (49, 7, 1, 1568), torch.float32, 'cuda'), ((32, 7, 7, 1), (49, 7, 1, 1568), torch.float32, 'cuda'), ((1568, 1024), (1024, 1), torch.float32, 'cuda'), ((32, 1024, 7, 7), (50176, 49, 7, 1), torch.float32, 'cuda'), ((32, 7, 7, 1), (49, 7, 1, 1568), torch.float32, 'cuda'), ((32, 7, 7, 1), (49, 7, 1, 1568), torch.float32, 'cuda'), ((1568, 1024), (1024, 1), torch.float32, 'cuda'), ((32, 1024, 7, 7), (50176, 49, 7, 1), torch.float32, 'cuda'), ((32, 7, 7, 1), (49, 7, 1, 1568), torch.float32, 'cuda'), ((32, 7, 7, 1), (49, 7, 1, 1568), torch.float32, 'cuda'), ((32, 1, 1, 1024), (1024, 32768, 32768, 1), torch.float32, 'cuda'), ((1000, 1024), (1024, 1), torch.float32, 'cuda'), ((32, 1, 1, 1), (1, 32, 32, 32), torch.float32, 'cuda'), ((1024, 4096), (4096, 1), torch.float32, 'cuda'), ((32, 7, 7, 4096), (200704, 28672, 4096, 1), torch.float32, 'cuda'), ((4096, 1024), (1024, 1), torch.float32, 'cuda'), ((1024, 4096), (4096, 1), torch.float32, 'cuda'), ((32, 7, 7, 4096), (200704, 28672, 4096, 1), torch.float32, 'cuda'), ((4096, 1024), (1024, 1), torch.float32, 'cuda'), ((1024, 4096), (4096, 1), torch.float32, 'cuda'), ((32, 7, 7, 4096), (200704, 28672, 4096, 1), torch.float32, 'cuda'), ((4096, 1024), (1024, 1), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((512, 2048), (2048, 1), torch.float32, 'cuda'), ((32, 14, 14, 2048), (401408, 28672, 2048, 1), torch.float32, 'cuda'), ((2048, 512), (512, 1), torch.float32, 'cuda'), ((512, 2048), (2048, 1), torch.float32, 'cuda'), ((32, 14, 14, 2048), (401408, 28672, 2048, 1), torch.float32, 'cuda'), ((2048, 512), (512, 1), torch.float32, 'cuda'), ((512, 2048), (2048, 1), torch.float32, 'cuda'), ((32, 14, 14, 2048), (401408, 28672, 2048, 1), torch.float32, 'cuda'), ((2048, 512), (512, 1), torch.float32, 'cuda'), ((512, 2048), (2048, 1), torch.float32, 'cuda'), ((32, 14, 14, 2048), (401408, 28672, 2048, 1), torch.float32, 'cuda'), ((2048, 512), (512, 1), torch.float32, 'cuda'), ((512, 2048), (2048, 1), torch.float32, 'cuda'), ((32, 14, 14, 2048), (401408, 28672, 2048, 1), torch.float32, 'cuda'), ((2048, 512), (512, 1), torch.float32, 'cuda'), ((512, 2048), (2048, 1), torch.float32, 'cuda'), ((32, 14, 14, 2048), (401408, 28672, 2048, 1), torch.float32, 'cuda'), ((2048, 512), (512, 1), torch.float32, 'cuda'), ((512, 2048), (2048, 1), torch.float32, 'cuda'), ((32, 14, 14, 2048), (401408, 28672, 2048, 1), torch.float32, 'cuda'), ((2048, 512), (512, 1), torch.float32, 'cuda'), ((512, 2048), (2048, 1), torch.float32, 'cuda'), ((32, 14, 14, 2048), (401408, 28672, 2048, 1), torch.float32, 'cuda'), ((2048, 512), (512, 1), torch.float32, 'cuda'), ((512, 2048), (2048, 1), torch.float32, 'cuda'), ((32, 14, 14, 2048), (401408, 28672, 2048, 1), torch.float32, 'cuda'), ((2048, 512), (512, 1), torch.float32, 'cuda'), ((512, 2048), (2048, 1), torch.float32, 'cuda'), ((32, 14, 14, 2048), (401408, 28672, 2048, 1), torch.float32, 'cuda'), ((2048, 512), (512, 1), torch.float32, 'cuda'), ((512, 2048), (2048, 1), torch.float32, 'cuda'), ((32, 14, 14, 2048), (401408, 28672, 2048, 1), torch.float32, 'cuda'), ((2048, 512), (512, 1), torch.float32, 'cuda'), ((512, 2048), (2048, 1), torch.float32, 'cuda'), ((32, 14, 14, 2048), (401408, 28672, 2048, 1), torch.float32, 'cuda'), ((2048, 512), (512, 1), torch.float32, 'cuda'), ((512, 2048), (2048, 1), torch.float32, 'cuda'), ((32, 14, 14, 2048), (401408, 28672, 2048, 1), torch.float32, 'cuda'), ((2048, 512), (512, 1), torch.float32, 'cuda'), ((512, 2048), (2048, 1), torch.float32, 'cuda'), ((32, 14, 14, 2048), (401408, 28672, 2048, 1), torch.float32, 'cuda'), ((2048, 512), (512, 1), torch.float32, 'cuda'), ((512, 2048), (2048, 1), torch.float32, 'cuda'), ((32, 14, 14, 2048), (401408, 28672, 2048, 1), torch.float32, 'cuda'), ((2048, 512), (512, 1), torch.float32, 'cuda'), ((512, 2048), (2048, 1), torch.float32, 'cuda'), ((32, 14, 14, 2048), (401408, 28672, 2048, 1), torch.float32, 'cuda'), ((32, 1000), (1000, 1), torch.float32, 'cuda')]
args = [rand_strided(sh, st, dt, dev) for (sh, st, dt, dev) in args]
mod = make_fx(Repro().to(device="cuda"))(*args)

from torch._dynamo.debug_utils import inductor_fails

if inductor_fails(mod, args):
    exit(1)
else:
    exit(0)
