
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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1):
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
        mul_15 = torch.ops.aten.mul.Tensor(permute_17, view_8);  permute_17 = None
        add_15 = torch.ops.aten.add.Tensor(mul_15, add_13);  mul_15 = add_13 = None
        permute_18 = torch.ops.aten.permute.default(arg216_1, [0, 2, 3, 1]);  arg216_1 = None
        sub_9 = torch.ops.aten.sub.Tensor(permute_18, arg217_1);  permute_18 = arg217_1 = None
        mul_16 = torch.ops.aten.mul.Tensor(sub_9, arg218_1);  sub_9 = None
        _unsafe_view_9 = torch.ops.aten._unsafe_view.default(arg221_1, [32, 14, 14, 512]);  arg221_1 = None
        add_16 = torch.ops.aten.add.Tensor(_unsafe_view_9, arg99_1);  _unsafe_view_9 = arg99_1 = None
        permute_19 = torch.ops.aten.permute.default(add_16, [0, 3, 1, 2]);  add_16 = None
        view_9 = torch.ops.aten.view.default(arg22_1, [1, 512, 1, 1]);  arg22_1 = None
        mul_17 = torch.ops.aten.mul.Tensor(permute_19, view_9);  permute_19 = None
        add_17 = torch.ops.aten.add.Tensor(mul_17, add_15);  mul_17 = None
        permute_20 = torch.ops.aten.permute.default(arg222_1, [0, 2, 3, 1]);  arg222_1 = None
        sub_10 = torch.ops.aten.sub.Tensor(permute_20, arg223_1);  permute_20 = arg223_1 = None
        mul_18 = torch.ops.aten.mul.Tensor(sub_10, arg224_1);  sub_10 = None
        _unsafe_view_10 = torch.ops.aten._unsafe_view.default(arg227_1, [32, 14, 14, 512]);  arg227_1 = None
        add_18 = torch.ops.aten.add.Tensor(_unsafe_view_10, arg101_1);  _unsafe_view_10 = arg101_1 = None
        permute_21 = torch.ops.aten.permute.default(add_18, [0, 3, 1, 2]);  add_18 = None
        view_10 = torch.ops.aten.view.default(arg24_1, [1, 512, 1, 1]);  arg24_1 = None
        mul_19 = torch.ops.aten.mul.Tensor(permute_21, view_10);  permute_21 = None
        add_19 = torch.ops.aten.add.Tensor(mul_19, add_17);  mul_19 = None
        permute_22 = torch.ops.aten.permute.default(arg228_1, [0, 2, 3, 1]);  arg228_1 = None
        sub_11 = torch.ops.aten.sub.Tensor(permute_22, arg229_1);  permute_22 = arg229_1 = None
        mul_20 = torch.ops.aten.mul.Tensor(sub_11, arg230_1);  sub_11 = None
        _unsafe_view_11 = torch.ops.aten._unsafe_view.default(arg233_1, [32, 14, 14, 512]);  arg233_1 = None
        add_20 = torch.ops.aten.add.Tensor(_unsafe_view_11, arg103_1);  _unsafe_view_11 = arg103_1 = None
        permute_23 = torch.ops.aten.permute.default(add_20, [0, 3, 1, 2]);  add_20 = None
        view_11 = torch.ops.aten.view.default(arg26_1, [1, 512, 1, 1]);  arg26_1 = None
        mul_21 = torch.ops.aten.mul.Tensor(permute_23, view_11);  permute_23 = None
        add_21 = torch.ops.aten.add.Tensor(mul_21, add_19);  mul_21 = None
        permute_24 = torch.ops.aten.permute.default(arg234_1, [0, 2, 3, 1]);  arg234_1 = None
        sub_12 = torch.ops.aten.sub.Tensor(permute_24, arg235_1);  permute_24 = arg235_1 = None
        mul_22 = torch.ops.aten.mul.Tensor(sub_12, arg236_1);  sub_12 = None
        _unsafe_view_12 = torch.ops.aten._unsafe_view.default(arg239_1, [32, 14, 14, 512]);  arg239_1 = None
        add_22 = torch.ops.aten.add.Tensor(_unsafe_view_12, arg105_1);  _unsafe_view_12 = arg105_1 = None
        permute_25 = torch.ops.aten.permute.default(add_22, [0, 3, 1, 2]);  add_22 = None
        view_12 = torch.ops.aten.view.default(arg28_1, [1, 512, 1, 1]);  arg28_1 = None
        mul_23 = torch.ops.aten.mul.Tensor(permute_25, view_12);  permute_25 = None
        add_23 = torch.ops.aten.add.Tensor(mul_23, add_21);  mul_23 = None
        permute_26 = torch.ops.aten.permute.default(arg240_1, [0, 2, 3, 1]);  arg240_1 = None
        sub_13 = torch.ops.aten.sub.Tensor(permute_26, arg241_1);  permute_26 = arg241_1 = None
        mul_24 = torch.ops.aten.mul.Tensor(sub_13, arg242_1);  sub_13 = None
        _unsafe_view_13 = torch.ops.aten._unsafe_view.default(arg245_1, [32, 14, 14, 512]);  arg245_1 = None
        add_24 = torch.ops.aten.add.Tensor(_unsafe_view_13, arg107_1);  _unsafe_view_13 = arg107_1 = None
        permute_27 = torch.ops.aten.permute.default(add_24, [0, 3, 1, 2]);  add_24 = None
        view_13 = torch.ops.aten.view.default(arg30_1, [1, 512, 1, 1]);  arg30_1 = None
        mul_25 = torch.ops.aten.mul.Tensor(permute_27, view_13);  permute_27 = None
        add_25 = torch.ops.aten.add.Tensor(mul_25, add_23);  mul_25 = None
        permute_28 = torch.ops.aten.permute.default(arg246_1, [0, 2, 3, 1]);  arg246_1 = None
        sub_14 = torch.ops.aten.sub.Tensor(permute_28, arg247_1);  permute_28 = arg247_1 = None
        mul_26 = torch.ops.aten.mul.Tensor(sub_14, arg248_1);  sub_14 = None
        _unsafe_view_14 = torch.ops.aten._unsafe_view.default(arg251_1, [32, 14, 14, 512]);  arg251_1 = None
        add_26 = torch.ops.aten.add.Tensor(_unsafe_view_14, arg109_1);  _unsafe_view_14 = arg109_1 = None
        permute_29 = torch.ops.aten.permute.default(add_26, [0, 3, 1, 2]);  add_26 = None
        view_14 = torch.ops.aten.view.default(arg32_1, [1, 512, 1, 1]);  arg32_1 = None
        mul_27 = torch.ops.aten.mul.Tensor(permute_29, view_14);  permute_29 = None
        add_27 = torch.ops.aten.add.Tensor(mul_27, add_25);  mul_27 = None
        permute_30 = torch.ops.aten.permute.default(arg252_1, [0, 2, 3, 1]);  arg252_1 = None
        sub_15 = torch.ops.aten.sub.Tensor(permute_30, arg253_1);  permute_30 = arg253_1 = None
        mul_28 = torch.ops.aten.mul.Tensor(sub_15, arg254_1);  sub_15 = None
        _unsafe_view_15 = torch.ops.aten._unsafe_view.default(arg257_1, [32, 14, 14, 512]);  arg257_1 = None
        add_28 = torch.ops.aten.add.Tensor(_unsafe_view_15, arg111_1);  _unsafe_view_15 = arg111_1 = None
        permute_31 = torch.ops.aten.permute.default(add_28, [0, 3, 1, 2]);  add_28 = None
        view_15 = torch.ops.aten.view.default(arg34_1, [1, 512, 1, 1]);  arg34_1 = None
        mul_29 = torch.ops.aten.mul.Tensor(permute_31, view_15);  permute_31 = None
        add_29 = torch.ops.aten.add.Tensor(mul_29, add_27);  mul_29 = None
        permute_32 = torch.ops.aten.permute.default(arg258_1, [0, 2, 3, 1]);  arg258_1 = None
        sub_16 = torch.ops.aten.sub.Tensor(permute_32, arg259_1);  permute_32 = arg259_1 = None
        mul_30 = torch.ops.aten.mul.Tensor(sub_16, arg260_1);  sub_16 = None
        _unsafe_view_16 = torch.ops.aten._unsafe_view.default(arg263_1, [32, 14, 14, 512]);  arg263_1 = None
        add_30 = torch.ops.aten.add.Tensor(_unsafe_view_16, arg113_1);  _unsafe_view_16 = arg113_1 = None
        permute_33 = torch.ops.aten.permute.default(add_30, [0, 3, 1, 2]);  add_30 = None
        view_16 = torch.ops.aten.view.default(arg36_1, [1, 512, 1, 1]);  arg36_1 = None
        mul_31 = torch.ops.aten.mul.Tensor(permute_33, view_16);  permute_33 = None
        add_31 = torch.ops.aten.add.Tensor(mul_31, add_29);  mul_31 = None
        permute_34 = torch.ops.aten.permute.default(arg264_1, [0, 2, 3, 1]);  arg264_1 = None
        sub_17 = torch.ops.aten.sub.Tensor(permute_34, arg265_1);  permute_34 = arg265_1 = None
        mul_32 = torch.ops.aten.mul.Tensor(sub_17, arg266_1);  sub_17 = None
        _unsafe_view_17 = torch.ops.aten._unsafe_view.default(arg269_1, [32, 14, 14, 512]);  arg269_1 = None
        add_32 = torch.ops.aten.add.Tensor(_unsafe_view_17, arg115_1);  _unsafe_view_17 = arg115_1 = None
        permute_35 = torch.ops.aten.permute.default(add_32, [0, 3, 1, 2]);  add_32 = None
        view_17 = torch.ops.aten.view.default(arg38_1, [1, 512, 1, 1]);  arg38_1 = None
        mul_33 = torch.ops.aten.mul.Tensor(permute_35, view_17);  permute_35 = None
        add_33 = torch.ops.aten.add.Tensor(mul_33, add_31);  mul_33 = None
        permute_36 = torch.ops.aten.permute.default(arg270_1, [0, 2, 3, 1]);  arg270_1 = None
        sub_18 = torch.ops.aten.sub.Tensor(permute_36, arg271_1);  permute_36 = arg271_1 = None
        mul_34 = torch.ops.aten.mul.Tensor(sub_18, arg272_1);  sub_18 = None
        _unsafe_view_18 = torch.ops.aten._unsafe_view.default(arg275_1, [32, 14, 14, 512]);  arg275_1 = None
        add_34 = torch.ops.aten.add.Tensor(_unsafe_view_18, arg117_1);  _unsafe_view_18 = arg117_1 = None
        permute_37 = torch.ops.aten.permute.default(add_34, [0, 3, 1, 2]);  add_34 = None
        view_18 = torch.ops.aten.view.default(arg40_1, [1, 512, 1, 1]);  arg40_1 = None
        mul_35 = torch.ops.aten.mul.Tensor(permute_37, view_18);  permute_37 = None
        add_35 = torch.ops.aten.add.Tensor(mul_35, add_33);  mul_35 = None
        permute_38 = torch.ops.aten.permute.default(arg276_1, [0, 2, 3, 1]);  arg276_1 = None
        sub_19 = torch.ops.aten.sub.Tensor(permute_38, arg277_1);  permute_38 = arg277_1 = None
        mul_36 = torch.ops.aten.mul.Tensor(sub_19, arg278_1);  sub_19 = None
        _unsafe_view_19 = torch.ops.aten._unsafe_view.default(arg281_1, [32, 14, 14, 512]);  arg281_1 = None
        add_36 = torch.ops.aten.add.Tensor(_unsafe_view_19, arg119_1);  _unsafe_view_19 = arg119_1 = None
        permute_39 = torch.ops.aten.permute.default(add_36, [0, 3, 1, 2]);  add_36 = None
        view_19 = torch.ops.aten.view.default(arg42_1, [1, 512, 1, 1]);  arg42_1 = None
        mul_37 = torch.ops.aten.mul.Tensor(permute_39, view_19);  permute_39 = None
        add_37 = torch.ops.aten.add.Tensor(mul_37, add_35);  mul_37 = None
        permute_40 = torch.ops.aten.permute.default(arg282_1, [0, 2, 3, 1]);  arg282_1 = None
        sub_20 = torch.ops.aten.sub.Tensor(permute_40, arg283_1);  permute_40 = arg283_1 = None
        mul_38 = torch.ops.aten.mul.Tensor(sub_20, arg284_1);  sub_20 = None
        _unsafe_view_20 = torch.ops.aten._unsafe_view.default(arg287_1, [32, 14, 14, 512]);  arg287_1 = None
        add_38 = torch.ops.aten.add.Tensor(_unsafe_view_20, arg121_1);  _unsafe_view_20 = arg121_1 = None
        permute_41 = torch.ops.aten.permute.default(add_38, [0, 3, 1, 2]);  add_38 = None
        view_20 = torch.ops.aten.view.default(arg44_1, [1, 512, 1, 1]);  arg44_1 = None
        mul_39 = torch.ops.aten.mul.Tensor(permute_41, view_20);  permute_41 = None
        add_39 = torch.ops.aten.add.Tensor(mul_39, add_37);  mul_39 = None
        permute_42 = torch.ops.aten.permute.default(arg288_1, [0, 2, 3, 1]);  arg288_1 = None
        sub_21 = torch.ops.aten.sub.Tensor(permute_42, arg289_1);  permute_42 = arg289_1 = None
        mul_40 = torch.ops.aten.mul.Tensor(sub_21, arg290_1);  sub_21 = None
        _unsafe_view_21 = torch.ops.aten._unsafe_view.default(arg293_1, [32, 14, 14, 512]);  arg293_1 = None
        add_40 = torch.ops.aten.add.Tensor(_unsafe_view_21, arg123_1);  _unsafe_view_21 = arg123_1 = None
        permute_43 = torch.ops.aten.permute.default(add_40, [0, 3, 1, 2]);  add_40 = None
        view_21 = torch.ops.aten.view.default(arg46_1, [1, 512, 1, 1]);  arg46_1 = None
        mul_41 = torch.ops.aten.mul.Tensor(permute_43, view_21);  permute_43 = None
        add_41 = torch.ops.aten.add.Tensor(mul_41, add_39);  mul_41 = None
        permute_44 = torch.ops.aten.permute.default(arg294_1, [0, 2, 3, 1]);  arg294_1 = None
        sub_22 = torch.ops.aten.sub.Tensor(permute_44, arg295_1);  permute_44 = arg295_1 = None
        mul_42 = torch.ops.aten.mul.Tensor(sub_22, arg296_1);  sub_22 = None
        _unsafe_view_22 = torch.ops.aten._unsafe_view.default(arg299_1, [32, 14, 14, 512]);  arg299_1 = None
        add_42 = torch.ops.aten.add.Tensor(_unsafe_view_22, arg125_1);  _unsafe_view_22 = arg125_1 = None
        permute_45 = torch.ops.aten.permute.default(add_42, [0, 3, 1, 2]);  add_42 = None
        view_22 = torch.ops.aten.view.default(arg48_1, [1, 512, 1, 1]);  arg48_1 = None
        mul_43 = torch.ops.aten.mul.Tensor(permute_45, view_22);  permute_45 = None
        add_43 = torch.ops.aten.add.Tensor(mul_43, add_41);  mul_43 = None
        permute_46 = torch.ops.aten.permute.default(arg300_1, [0, 2, 3, 1]);  arg300_1 = None
        sub_23 = torch.ops.aten.sub.Tensor(permute_46, arg301_1);  permute_46 = arg301_1 = None
        mul_44 = torch.ops.aten.mul.Tensor(sub_23, arg302_1);  sub_23 = None
        _unsafe_view_23 = torch.ops.aten._unsafe_view.default(arg305_1, [32, 14, 14, 512]);  arg305_1 = None
        add_44 = torch.ops.aten.add.Tensor(_unsafe_view_23, arg127_1);  _unsafe_view_23 = arg127_1 = None
        permute_47 = torch.ops.aten.permute.default(add_44, [0, 3, 1, 2]);  add_44 = None
        view_23 = torch.ops.aten.view.default(arg50_1, [1, 512, 1, 1]);  arg50_1 = None
        mul_45 = torch.ops.aten.mul.Tensor(permute_47, view_23);  permute_47 = None
        add_45 = torch.ops.aten.add.Tensor(mul_45, add_43);  mul_45 = None
        permute_48 = torch.ops.aten.permute.default(arg306_1, [0, 2, 3, 1]);  arg306_1 = None
        sub_24 = torch.ops.aten.sub.Tensor(permute_48, arg307_1);  permute_48 = arg307_1 = None
        mul_46 = torch.ops.aten.mul.Tensor(sub_24, arg308_1);  sub_24 = None
        _unsafe_view_24 = torch.ops.aten._unsafe_view.default(arg311_1, [32, 14, 14, 512]);  arg311_1 = None
        add_46 = torch.ops.aten.add.Tensor(_unsafe_view_24, arg129_1);  _unsafe_view_24 = arg129_1 = None
        permute_49 = torch.ops.aten.permute.default(add_46, [0, 3, 1, 2]);  add_46 = None
        view_24 = torch.ops.aten.view.default(arg52_1, [1, 512, 1, 1]);  arg52_1 = None
        mul_47 = torch.ops.aten.mul.Tensor(permute_49, view_24);  permute_49 = None
        add_47 = torch.ops.aten.add.Tensor(mul_47, add_45);  mul_47 = None
        permute_50 = torch.ops.aten.permute.default(arg312_1, [0, 2, 3, 1]);  arg312_1 = None
        sub_25 = torch.ops.aten.sub.Tensor(permute_50, arg313_1);  permute_50 = arg313_1 = None
        mul_48 = torch.ops.aten.mul.Tensor(sub_25, arg314_1);  sub_25 = None
        _unsafe_view_25 = torch.ops.aten._unsafe_view.default(arg317_1, [32, 14, 14, 512]);  arg317_1 = None
        add_48 = torch.ops.aten.add.Tensor(_unsafe_view_25, arg131_1);  _unsafe_view_25 = arg131_1 = None
        permute_51 = torch.ops.aten.permute.default(add_48, [0, 3, 1, 2]);  add_48 = None
        view_25 = torch.ops.aten.view.default(arg54_1, [1, 512, 1, 1]);  arg54_1 = None
        mul_49 = torch.ops.aten.mul.Tensor(permute_51, view_25);  permute_51 = None
        add_49 = torch.ops.aten.add.Tensor(mul_49, add_47);  mul_49 = None
        permute_52 = torch.ops.aten.permute.default(arg318_1, [0, 2, 3, 1]);  arg318_1 = None
        sub_26 = torch.ops.aten.sub.Tensor(permute_52, arg319_1);  permute_52 = arg319_1 = None
        mul_50 = torch.ops.aten.mul.Tensor(sub_26, arg320_1);  sub_26 = None
        _unsafe_view_26 = torch.ops.aten._unsafe_view.default(arg323_1, [32, 14, 14, 512]);  arg323_1 = None
        add_50 = torch.ops.aten.add.Tensor(_unsafe_view_26, arg133_1);  _unsafe_view_26 = arg133_1 = None
        permute_53 = torch.ops.aten.permute.default(add_50, [0, 3, 1, 2]);  add_50 = None
        view_26 = torch.ops.aten.view.default(arg56_1, [1, 512, 1, 1]);  arg56_1 = None
        mul_51 = torch.ops.aten.mul.Tensor(permute_53, view_26);  permute_53 = None
        add_51 = torch.ops.aten.add.Tensor(mul_51, add_49);  mul_51 = None
        permute_54 = torch.ops.aten.permute.default(arg324_1, [0, 2, 3, 1]);  arg324_1 = None
        sub_27 = torch.ops.aten.sub.Tensor(permute_54, arg325_1);  permute_54 = arg325_1 = None
        mul_52 = torch.ops.aten.mul.Tensor(sub_27, arg326_1);  sub_27 = None
        _unsafe_view_27 = torch.ops.aten._unsafe_view.default(arg329_1, [32, 14, 14, 512]);  arg329_1 = None
        add_52 = torch.ops.aten.add.Tensor(_unsafe_view_27, arg135_1);  _unsafe_view_27 = arg135_1 = None
        permute_55 = torch.ops.aten.permute.default(add_52, [0, 3, 1, 2]);  add_52 = None
        view_27 = torch.ops.aten.view.default(arg58_1, [1, 512, 1, 1]);  arg58_1 = None
        mul_53 = torch.ops.aten.mul.Tensor(permute_55, view_27);  permute_55 = None
        add_53 = torch.ops.aten.add.Tensor(mul_53, add_51);  mul_53 = None
        permute_56 = torch.ops.aten.permute.default(arg330_1, [0, 2, 3, 1]);  arg330_1 = None
        sub_28 = torch.ops.aten.sub.Tensor(permute_56, arg331_1);  permute_56 = arg331_1 = None
        mul_54 = torch.ops.aten.mul.Tensor(sub_28, arg332_1);  sub_28 = None
        _unsafe_view_28 = torch.ops.aten._unsafe_view.default(arg335_1, [32, 14, 14, 512]);  arg335_1 = None
        add_54 = torch.ops.aten.add.Tensor(_unsafe_view_28, arg137_1);  _unsafe_view_28 = arg137_1 = None
        permute_57 = torch.ops.aten.permute.default(add_54, [0, 3, 1, 2]);  add_54 = None
        view_28 = torch.ops.aten.view.default(arg60_1, [1, 512, 1, 1]);  arg60_1 = None
        mul_55 = torch.ops.aten.mul.Tensor(permute_57, view_28);  permute_57 = None
        add_55 = torch.ops.aten.add.Tensor(mul_55, add_53);  mul_55 = None
        permute_58 = torch.ops.aten.permute.default(arg336_1, [0, 2, 3, 1]);  arg336_1 = None
        sub_29 = torch.ops.aten.sub.Tensor(permute_58, arg337_1);  permute_58 = arg337_1 = None
        mul_56 = torch.ops.aten.mul.Tensor(sub_29, arg338_1);  sub_29 = None
        _unsafe_view_29 = torch.ops.aten._unsafe_view.default(arg341_1, [32, 14, 14, 512]);  arg341_1 = None
        add_56 = torch.ops.aten.add.Tensor(_unsafe_view_29, arg139_1);  _unsafe_view_29 = arg139_1 = None
        permute_59 = torch.ops.aten.permute.default(add_56, [0, 3, 1, 2]);  add_56 = None
        view_29 = torch.ops.aten.view.default(arg62_1, [1, 512, 1, 1]);  arg62_1 = None
        mul_57 = torch.ops.aten.mul.Tensor(permute_59, view_29);  permute_59 = None
        add_57 = torch.ops.aten.add.Tensor(mul_57, add_55);  mul_57 = None
        permute_60 = torch.ops.aten.permute.default(arg342_1, [0, 2, 3, 1]);  arg342_1 = None
        sub_30 = torch.ops.aten.sub.Tensor(permute_60, arg343_1);  permute_60 = arg343_1 = None
        mul_58 = torch.ops.aten.mul.Tensor(sub_30, arg344_1);  sub_30 = None
        _unsafe_view_30 = torch.ops.aten._unsafe_view.default(arg347_1, [32, 14, 14, 512]);  arg347_1 = None
        add_58 = torch.ops.aten.add.Tensor(_unsafe_view_30, arg141_1);  _unsafe_view_30 = arg141_1 = None
        permute_61 = torch.ops.aten.permute.default(add_58, [0, 3, 1, 2]);  add_58 = None
        view_30 = torch.ops.aten.view.default(arg64_1, [1, 512, 1, 1]);  arg64_1 = None
        mul_59 = torch.ops.aten.mul.Tensor(permute_61, view_30);  permute_61 = None
        add_59 = torch.ops.aten.add.Tensor(mul_59, add_57);  mul_59 = None
        permute_62 = torch.ops.aten.permute.default(arg348_1, [0, 2, 3, 1]);  arg348_1 = None
        sub_31 = torch.ops.aten.sub.Tensor(permute_62, arg349_1);  permute_62 = arg349_1 = None
        mul_60 = torch.ops.aten.mul.Tensor(sub_31, arg350_1);  sub_31 = None
        _unsafe_view_31 = torch.ops.aten._unsafe_view.default(arg353_1, [32, 14, 14, 512]);  arg353_1 = None
        add_60 = torch.ops.aten.add.Tensor(_unsafe_view_31, arg143_1);  _unsafe_view_31 = arg143_1 = None
        permute_63 = torch.ops.aten.permute.default(add_60, [0, 3, 1, 2]);  add_60 = None
        view_31 = torch.ops.aten.view.default(arg66_1, [1, 512, 1, 1]);  arg66_1 = None
        mul_61 = torch.ops.aten.mul.Tensor(permute_63, view_31);  permute_63 = None
        add_61 = torch.ops.aten.add.Tensor(mul_61, add_59);  mul_61 = None
        permute_64 = torch.ops.aten.permute.default(arg354_1, [0, 2, 3, 1]);  arg354_1 = None
        sub_32 = torch.ops.aten.sub.Tensor(permute_64, arg355_1);  permute_64 = arg355_1 = None
        mul_62 = torch.ops.aten.mul.Tensor(sub_32, arg356_1);  sub_32 = None
        view_32 = torch.ops.aten.view.default(arg68_1, [1, 512, 1, 1]);  arg68_1 = None
        permute_66 = torch.ops.aten.permute.default(arg363_1, [0, 2, 3, 1]);  arg363_1 = None
        sub_33 = torch.ops.aten.sub.Tensor(permute_66, arg364_1);  permute_66 = arg364_1 = None
        mul_63 = torch.ops.aten.mul.Tensor(sub_33, arg365_1);  sub_33 = None
        _unsafe_view_33 = torch.ops.aten._unsafe_view.default(arg368_1, [32, 7, 7, 1024]);  arg368_1 = None
        add_63 = torch.ops.aten.add.Tensor(_unsafe_view_33, arg148_1);  _unsafe_view_33 = arg148_1 = None
        permute_67 = torch.ops.aten.permute.default(add_63, [0, 3, 1, 2]);  add_63 = None
        view_33 = torch.ops.aten.view.default(arg71_1, [1, 1024, 1, 1]);  arg71_1 = None
        mul_64 = torch.ops.aten.mul.Tensor(permute_67, view_33);  permute_67 = None
        add_64 = torch.ops.aten.add.Tensor(mul_64, arg362_1);  mul_64 = None
        permute_68 = torch.ops.aten.permute.default(arg369_1, [0, 2, 3, 1]);  arg369_1 = None
        sub_34 = torch.ops.aten.sub.Tensor(permute_68, arg370_1);  permute_68 = arg370_1 = None
        mul_65 = torch.ops.aten.mul.Tensor(sub_34, arg371_1);  sub_34 = None
        _unsafe_view_34 = torch.ops.aten._unsafe_view.default(arg374_1, [32, 7, 7, 1024]);  arg374_1 = None
        add_65 = torch.ops.aten.add.Tensor(_unsafe_view_34, arg150_1);  _unsafe_view_34 = arg150_1 = None
        permute_69 = torch.ops.aten.permute.default(add_65, [0, 3, 1, 2]);  add_65 = None
        view_34 = torch.ops.aten.view.default(arg73_1, [1, 1024, 1, 1]);  arg73_1 = None
        mul_66 = torch.ops.aten.mul.Tensor(permute_69, view_34);  permute_69 = None
        add_66 = torch.ops.aten.add.Tensor(mul_66, add_64);  mul_66 = None
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
        mul_84 = torch.ops.aten.mul.Tensor(add_68, view_34);  view_34 = None
        permute_85 = torch.ops.aten.permute.default(mul_84, [0, 2, 3, 1]);  mul_84 = None
        clone_1 = torch.ops.aten.clone.default(permute_85, memory_format = torch.contiguous_format);  permute_85 = None
        _unsafe_view_37 = torch.ops.aten._unsafe_view.default(clone_1, [1568, 1024]);  clone_1 = None
        mm_7 = torch.ops.aten.mm.default(_unsafe_view_37, arg388_1);  _unsafe_view_37 = arg388_1 = None
        view_46 = torch.ops.aten.view.default(mm_7, [32, 7, 7, 4096]);  mm_7 = None
        mul_85 = torch.ops.aten.mul.Tensor(view_46, arg389_1);  view_46 = arg389_1 = None
        view_48 = torch.ops.aten.view.default(mul_85, [1568, 4096]);  mul_85 = None
        mm_9 = torch.ops.aten.mm.default(view_48, arg390_1);  view_48 = arg390_1 = None
        view_49 = torch.ops.aten.view.default(mm_9, [32, 7, 7, 1024]);  mm_9 = None
        mul_86 = torch.ops.aten.mul.Tensor(view_49, arg72_1);  view_49 = arg72_1 = None
        mul_87 = torch.ops.aten.mul.Tensor(mul_86, 1024)
        sum_16 = torch.ops.aten.sum.dim_IntList(mul_86, [3], True)
        mul_88 = torch.ops.aten.mul.Tensor(mul_86, mul_65);  mul_86 = None
        sum_17 = torch.ops.aten.sum.dim_IntList(mul_88, [3], True);  mul_88 = None
        mul_89 = torch.ops.aten.mul.Tensor(mul_65, sum_17);  mul_65 = sum_17 = None
        sub_40 = torch.ops.aten.sub.Tensor(mul_87, sum_16);  mul_87 = sum_16 = None
        sub_41 = torch.ops.aten.sub.Tensor(sub_40, mul_89);  sub_40 = mul_89 = None
        div_2 = torch.ops.aten.div.Tensor(arg371_1, 1024);  arg371_1 = None
        mul_90 = torch.ops.aten.mul.Tensor(div_2, sub_41);  div_2 = sub_41 = None
        permute_92 = torch.ops.aten.permute.default(mul_90, [0, 3, 1, 2]);  mul_90 = None
        convolution_backward_1 = torch.ops.aten.convolution_backward.default(permute_92, add_64, arg149_1, [1024], [1, 1], [3, 3], [1, 1], False, [0, 0], 1024, [True, True, True]);  permute_92 = add_64 = arg149_1 = None
        getitem_3 = convolution_backward_1[0];  convolution_backward_1 = None
        add_69 = torch.ops.aten.add.Tensor(add_68, getitem_3);  add_68 = getitem_3 = None
        mul_93 = torch.ops.aten.mul.Tensor(add_69, view_33);  view_33 = None
        permute_93 = torch.ops.aten.permute.default(mul_93, [0, 2, 3, 1]);  mul_93 = None
        clone_2 = torch.ops.aten.clone.default(permute_93, memory_format = torch.contiguous_format);  permute_93 = None
        _unsafe_view_38 = torch.ops.aten._unsafe_view.default(clone_2, [1568, 1024]);  clone_2 = None
        mm_11 = torch.ops.aten.mm.default(_unsafe_view_38, arg391_1);  _unsafe_view_38 = arg391_1 = None
        view_52 = torch.ops.aten.view.default(mm_11, [32, 7, 7, 4096]);  mm_11 = None
        mul_94 = torch.ops.aten.mul.Tensor(view_52, arg392_1);  view_52 = arg392_1 = None
        view_54 = torch.ops.aten.view.default(mul_94, [1568, 4096]);  mul_94 = None
        mm_13 = torch.ops.aten.mm.default(view_54, arg393_1);  view_54 = arg393_1 = None
        view_55 = torch.ops.aten.view.default(mm_13, [32, 7, 7, 1024]);  mm_13 = None
        mul_95 = torch.ops.aten.mul.Tensor(view_55, arg70_1);  view_55 = arg70_1 = None
        mul_96 = torch.ops.aten.mul.Tensor(mul_95, 1024)
        sum_23 = torch.ops.aten.sum.dim_IntList(mul_95, [3], True)
        mul_97 = torch.ops.aten.mul.Tensor(mul_95, mul_63);  mul_95 = None
        sum_24 = torch.ops.aten.sum.dim_IntList(mul_97, [3], True);  mul_97 = None
        mul_98 = torch.ops.aten.mul.Tensor(mul_63, sum_24);  mul_63 = sum_24 = None
        sub_42 = torch.ops.aten.sub.Tensor(mul_96, sum_23);  mul_96 = sum_23 = None
        sub_43 = torch.ops.aten.sub.Tensor(sub_42, mul_98);  sub_42 = mul_98 = None
        div_3 = torch.ops.aten.div.Tensor(arg365_1, 1024);  arg365_1 = None
        mul_99 = torch.ops.aten.mul.Tensor(div_3, sub_43);  div_3 = sub_43 = None
        permute_100 = torch.ops.aten.permute.default(mul_99, [0, 3, 1, 2]);  mul_99 = None
        convolution_backward_2 = torch.ops.aten.convolution_backward.default(permute_100, arg362_1, arg147_1, [1024], [1, 1], [3, 3], [1, 1], False, [0, 0], 1024, [True, True, True]);  permute_100 = arg362_1 = arg147_1 = None
        getitem_6 = convolution_backward_2[0];  convolution_backward_2 = None
        add_70 = torch.ops.aten.add.Tensor(add_69, getitem_6);  add_69 = getitem_6 = None
        convolution_backward_3 = torch.ops.aten.convolution_backward.default(add_70, arg361_1, arg146_1, [1024], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  add_70 = arg361_1 = arg146_1 = None
        getitem_9 = convolution_backward_3[0];  convolution_backward_3 = None
        permute_101 = torch.ops.aten.permute.default(getitem_9, [0, 2, 3, 1]);  getitem_9 = None
        mul_101 = torch.ops.aten.mul.Tensor(permute_101, arg69_1);  permute_101 = arg69_1 = None
        mul_102 = torch.ops.aten.mul.Tensor(mul_101, 512)
        sum_27 = torch.ops.aten.sum.dim_IntList(mul_101, [3], True)
        mul_103 = torch.ops.aten.mul.Tensor(mul_101, arg360_1);  mul_101 = None
        sum_28 = torch.ops.aten.sum.dim_IntList(mul_103, [3], True);  mul_103 = None
        mul_104 = torch.ops.aten.mul.Tensor(arg360_1, sum_28);  arg360_1 = sum_28 = None
        sub_44 = torch.ops.aten.sub.Tensor(mul_102, sum_27);  mul_102 = sum_27 = None
        sub_45 = torch.ops.aten.sub.Tensor(sub_44, mul_104);  sub_44 = mul_104 = None
        mul_105 = torch.ops.aten.mul.Tensor(arg394_1, sub_45);  arg394_1 = sub_45 = None
        permute_102 = torch.ops.aten.permute.default(mul_105, [0, 3, 1, 2]);  mul_105 = None
        mul_108 = torch.ops.aten.mul.Tensor(permute_102, view_32);  view_32 = None
        permute_103 = torch.ops.aten.permute.default(mul_108, [0, 2, 3, 1]);  mul_108 = None
        view_58 = torch.ops.aten.view.default(permute_103, [6272, 512]);  permute_103 = None
        mm_15 = torch.ops.aten.mm.default(view_58, arg395_1);  view_58 = arg395_1 = None
        view_59 = torch.ops.aten.view.default(mm_15, [32, 14, 14, 2048]);  mm_15 = None
        mul_109 = torch.ops.aten.mul.Tensor(view_59, arg396_1);  view_59 = arg396_1 = None
        view_61 = torch.ops.aten.view.default(mul_109, [6272, 2048]);  mul_109 = None
        mm_17 = torch.ops.aten.mm.default(view_61, arg397_1);  view_61 = arg397_1 = None
        view_62 = torch.ops.aten.view.default(mm_17, [32, 14, 14, 512]);  mm_17 = None
        mul_110 = torch.ops.aten.mul.Tensor(view_62, arg67_1);  view_62 = arg67_1 = None
        mul_111 = torch.ops.aten.mul.Tensor(mul_110, 512)
        sum_34 = torch.ops.aten.sum.dim_IntList(mul_110, [3], True)
        mul_112 = torch.ops.aten.mul.Tensor(mul_110, mul_62);  mul_110 = None
        sum_35 = torch.ops.aten.sum.dim_IntList(mul_112, [3], True);  mul_112 = None
        mul_113 = torch.ops.aten.mul.Tensor(mul_62, sum_35);  mul_62 = sum_35 = None
        sub_46 = torch.ops.aten.sub.Tensor(mul_111, sum_34);  mul_111 = sum_34 = None
        sub_47 = torch.ops.aten.sub.Tensor(sub_46, mul_113);  sub_46 = mul_113 = None
        div_4 = torch.ops.aten.div.Tensor(arg356_1, 512);  arg356_1 = None
        mul_114 = torch.ops.aten.mul.Tensor(div_4, sub_47);  div_4 = sub_47 = None
        permute_110 = torch.ops.aten.permute.default(mul_114, [0, 3, 1, 2]);  mul_114 = None
        convolution_backward_4 = torch.ops.aten.convolution_backward.default(permute_110, add_61, arg144_1, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, True]);  permute_110 = add_61 = arg144_1 = None
        getitem_12 = convolution_backward_4[0];  convolution_backward_4 = None
        add_71 = torch.ops.aten.add.Tensor(permute_102, getitem_12);  permute_102 = getitem_12 = None
        mul_117 = torch.ops.aten.mul.Tensor(add_71, view_31);  view_31 = None
        permute_111 = torch.ops.aten.permute.default(mul_117, [0, 2, 3, 1]);  mul_117 = None
        view_65 = torch.ops.aten.view.default(permute_111, [6272, 512]);  permute_111 = None
        mm_19 = torch.ops.aten.mm.default(view_65, arg398_1);  view_65 = arg398_1 = None
        view_66 = torch.ops.aten.view.default(mm_19, [32, 14, 14, 2048]);  mm_19 = None
        mul_118 = torch.ops.aten.mul.Tensor(view_66, arg399_1);  view_66 = arg399_1 = None
        view_68 = torch.ops.aten.view.default(mul_118, [6272, 2048]);  mul_118 = None
        mm_21 = torch.ops.aten.mm.default(view_68, arg400_1);  view_68 = arg400_1 = None
        view_69 = torch.ops.aten.view.default(mm_21, [32, 14, 14, 512]);  mm_21 = None
        mul_119 = torch.ops.aten.mul.Tensor(view_69, arg65_1);  view_69 = arg65_1 = None
        mul_120 = torch.ops.aten.mul.Tensor(mul_119, 512)
        sum_41 = torch.ops.aten.sum.dim_IntList(mul_119, [3], True)
        mul_121 = torch.ops.aten.mul.Tensor(mul_119, mul_60);  mul_119 = None
        sum_42 = torch.ops.aten.sum.dim_IntList(mul_121, [3], True);  mul_121 = None
        mul_122 = torch.ops.aten.mul.Tensor(mul_60, sum_42);  mul_60 = sum_42 = None
        sub_48 = torch.ops.aten.sub.Tensor(mul_120, sum_41);  mul_120 = sum_41 = None
        sub_49 = torch.ops.aten.sub.Tensor(sub_48, mul_122);  sub_48 = mul_122 = None
        div_5 = torch.ops.aten.div.Tensor(arg350_1, 512);  arg350_1 = None
        mul_123 = torch.ops.aten.mul.Tensor(div_5, sub_49);  div_5 = sub_49 = None
        permute_118 = torch.ops.aten.permute.default(mul_123, [0, 3, 1, 2]);  mul_123 = None
        convolution_backward_5 = torch.ops.aten.convolution_backward.default(permute_118, add_59, arg142_1, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, True]);  permute_118 = add_59 = arg142_1 = None
        getitem_15 = convolution_backward_5[0];  convolution_backward_5 = None
        add_72 = torch.ops.aten.add.Tensor(add_71, getitem_15);  add_71 = getitem_15 = None
        mul_126 = torch.ops.aten.mul.Tensor(add_72, view_30);  view_30 = None
        permute_119 = torch.ops.aten.permute.default(mul_126, [0, 2, 3, 1]);  mul_126 = None
        view_72 = torch.ops.aten.view.default(permute_119, [6272, 512]);  permute_119 = None
        mm_23 = torch.ops.aten.mm.default(view_72, arg401_1);  view_72 = arg401_1 = None
        view_73 = torch.ops.aten.view.default(mm_23, [32, 14, 14, 2048]);  mm_23 = None
        mul_127 = torch.ops.aten.mul.Tensor(view_73, arg402_1);  view_73 = arg402_1 = None
        view_75 = torch.ops.aten.view.default(mul_127, [6272, 2048]);  mul_127 = None
        mm_25 = torch.ops.aten.mm.default(view_75, arg403_1);  view_75 = arg403_1 = None
        view_76 = torch.ops.aten.view.default(mm_25, [32, 14, 14, 512]);  mm_25 = None
        mul_128 = torch.ops.aten.mul.Tensor(view_76, arg63_1);  view_76 = arg63_1 = None
        mul_129 = torch.ops.aten.mul.Tensor(mul_128, 512)
        sum_48 = torch.ops.aten.sum.dim_IntList(mul_128, [3], True)
        mul_130 = torch.ops.aten.mul.Tensor(mul_128, mul_58);  mul_128 = None
        sum_49 = torch.ops.aten.sum.dim_IntList(mul_130, [3], True);  mul_130 = None
        mul_131 = torch.ops.aten.mul.Tensor(mul_58, sum_49);  mul_58 = sum_49 = None
        sub_50 = torch.ops.aten.sub.Tensor(mul_129, sum_48);  mul_129 = sum_48 = None
        sub_51 = torch.ops.aten.sub.Tensor(sub_50, mul_131);  sub_50 = mul_131 = None
        div_6 = torch.ops.aten.div.Tensor(arg344_1, 512);  arg344_1 = None
        mul_132 = torch.ops.aten.mul.Tensor(div_6, sub_51);  div_6 = sub_51 = None
        permute_126 = torch.ops.aten.permute.default(mul_132, [0, 3, 1, 2]);  mul_132 = None
        convolution_backward_6 = torch.ops.aten.convolution_backward.default(permute_126, add_57, arg140_1, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, True]);  permute_126 = add_57 = arg140_1 = None
        getitem_18 = convolution_backward_6[0];  convolution_backward_6 = None
        add_73 = torch.ops.aten.add.Tensor(add_72, getitem_18);  add_72 = getitem_18 = None
        mul_135 = torch.ops.aten.mul.Tensor(add_73, view_29);  view_29 = None
        permute_127 = torch.ops.aten.permute.default(mul_135, [0, 2, 3, 1]);  mul_135 = None
        view_79 = torch.ops.aten.view.default(permute_127, [6272, 512]);  permute_127 = None
        mm_27 = torch.ops.aten.mm.default(view_79, arg404_1);  view_79 = arg404_1 = None
        view_80 = torch.ops.aten.view.default(mm_27, [32, 14, 14, 2048]);  mm_27 = None
        mul_136 = torch.ops.aten.mul.Tensor(view_80, arg405_1);  view_80 = arg405_1 = None
        view_82 = torch.ops.aten.view.default(mul_136, [6272, 2048]);  mul_136 = None
        mm_29 = torch.ops.aten.mm.default(view_82, arg406_1);  view_82 = arg406_1 = None
        view_83 = torch.ops.aten.view.default(mm_29, [32, 14, 14, 512]);  mm_29 = None
        mul_137 = torch.ops.aten.mul.Tensor(view_83, arg61_1);  view_83 = arg61_1 = None
        mul_138 = torch.ops.aten.mul.Tensor(mul_137, 512)
        sum_55 = torch.ops.aten.sum.dim_IntList(mul_137, [3], True)
        mul_139 = torch.ops.aten.mul.Tensor(mul_137, mul_56);  mul_137 = None
        sum_56 = torch.ops.aten.sum.dim_IntList(mul_139, [3], True);  mul_139 = None
        mul_140 = torch.ops.aten.mul.Tensor(mul_56, sum_56);  mul_56 = sum_56 = None
        sub_52 = torch.ops.aten.sub.Tensor(mul_138, sum_55);  mul_138 = sum_55 = None
        sub_53 = torch.ops.aten.sub.Tensor(sub_52, mul_140);  sub_52 = mul_140 = None
        div_7 = torch.ops.aten.div.Tensor(arg338_1, 512);  arg338_1 = None
        mul_141 = torch.ops.aten.mul.Tensor(div_7, sub_53);  div_7 = sub_53 = None
        permute_134 = torch.ops.aten.permute.default(mul_141, [0, 3, 1, 2]);  mul_141 = None
        convolution_backward_7 = torch.ops.aten.convolution_backward.default(permute_134, add_55, arg138_1, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, True]);  permute_134 = add_55 = arg138_1 = None
        getitem_21 = convolution_backward_7[0];  convolution_backward_7 = None
        add_74 = torch.ops.aten.add.Tensor(add_73, getitem_21);  add_73 = getitem_21 = None
        mul_144 = torch.ops.aten.mul.Tensor(add_74, view_28);  view_28 = None
        permute_135 = torch.ops.aten.permute.default(mul_144, [0, 2, 3, 1]);  mul_144 = None
        view_86 = torch.ops.aten.view.default(permute_135, [6272, 512]);  permute_135 = None
        mm_31 = torch.ops.aten.mm.default(view_86, arg407_1);  view_86 = arg407_1 = None
        view_87 = torch.ops.aten.view.default(mm_31, [32, 14, 14, 2048]);  mm_31 = None
        mul_145 = torch.ops.aten.mul.Tensor(view_87, arg408_1);  view_87 = arg408_1 = None
        view_89 = torch.ops.aten.view.default(mul_145, [6272, 2048]);  mul_145 = None
        mm_33 = torch.ops.aten.mm.default(view_89, arg409_1);  view_89 = arg409_1 = None
        view_90 = torch.ops.aten.view.default(mm_33, [32, 14, 14, 512]);  mm_33 = None
        mul_146 = torch.ops.aten.mul.Tensor(view_90, arg59_1);  view_90 = arg59_1 = None
        mul_147 = torch.ops.aten.mul.Tensor(mul_146, 512)
        sum_62 = torch.ops.aten.sum.dim_IntList(mul_146, [3], True)
        mul_148 = torch.ops.aten.mul.Tensor(mul_146, mul_54);  mul_146 = None
        sum_63 = torch.ops.aten.sum.dim_IntList(mul_148, [3], True);  mul_148 = None
        mul_149 = torch.ops.aten.mul.Tensor(mul_54, sum_63);  mul_54 = sum_63 = None
        sub_54 = torch.ops.aten.sub.Tensor(mul_147, sum_62);  mul_147 = sum_62 = None
        sub_55 = torch.ops.aten.sub.Tensor(sub_54, mul_149);  sub_54 = mul_149 = None
        div_8 = torch.ops.aten.div.Tensor(arg332_1, 512);  arg332_1 = None
        mul_150 = torch.ops.aten.mul.Tensor(div_8, sub_55);  div_8 = sub_55 = None
        permute_142 = torch.ops.aten.permute.default(mul_150, [0, 3, 1, 2]);  mul_150 = None
        convolution_backward_8 = torch.ops.aten.convolution_backward.default(permute_142, add_53, arg136_1, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, True]);  permute_142 = add_53 = arg136_1 = None
        getitem_24 = convolution_backward_8[0];  convolution_backward_8 = None
        add_75 = torch.ops.aten.add.Tensor(add_74, getitem_24);  add_74 = getitem_24 = None
        mul_153 = torch.ops.aten.mul.Tensor(add_75, view_27);  view_27 = None
        permute_143 = torch.ops.aten.permute.default(mul_153, [0, 2, 3, 1]);  mul_153 = None
        view_93 = torch.ops.aten.view.default(permute_143, [6272, 512]);  permute_143 = None
        mm_35 = torch.ops.aten.mm.default(view_93, arg410_1);  view_93 = arg410_1 = None
        view_94 = torch.ops.aten.view.default(mm_35, [32, 14, 14, 2048]);  mm_35 = None
        mul_154 = torch.ops.aten.mul.Tensor(view_94, arg411_1);  view_94 = arg411_1 = None
        view_96 = torch.ops.aten.view.default(mul_154, [6272, 2048]);  mul_154 = None
        mm_37 = torch.ops.aten.mm.default(view_96, arg412_1);  view_96 = arg412_1 = None
        view_97 = torch.ops.aten.view.default(mm_37, [32, 14, 14, 512]);  mm_37 = None
        mul_155 = torch.ops.aten.mul.Tensor(view_97, arg57_1);  view_97 = arg57_1 = None
        mul_156 = torch.ops.aten.mul.Tensor(mul_155, 512)
        sum_69 = torch.ops.aten.sum.dim_IntList(mul_155, [3], True)
        mul_157 = torch.ops.aten.mul.Tensor(mul_155, mul_52);  mul_155 = None
        sum_70 = torch.ops.aten.sum.dim_IntList(mul_157, [3], True);  mul_157 = None
        mul_158 = torch.ops.aten.mul.Tensor(mul_52, sum_70);  mul_52 = sum_70 = None
        sub_56 = torch.ops.aten.sub.Tensor(mul_156, sum_69);  mul_156 = sum_69 = None
        sub_57 = torch.ops.aten.sub.Tensor(sub_56, mul_158);  sub_56 = mul_158 = None
        div_9 = torch.ops.aten.div.Tensor(arg326_1, 512);  arg326_1 = None
        mul_159 = torch.ops.aten.mul.Tensor(div_9, sub_57);  div_9 = sub_57 = None
        permute_150 = torch.ops.aten.permute.default(mul_159, [0, 3, 1, 2]);  mul_159 = None
        convolution_backward_9 = torch.ops.aten.convolution_backward.default(permute_150, add_51, arg134_1, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, True]);  permute_150 = add_51 = arg134_1 = None
        getitem_27 = convolution_backward_9[0];  convolution_backward_9 = None
        add_76 = torch.ops.aten.add.Tensor(add_75, getitem_27);  add_75 = getitem_27 = None
        mul_162 = torch.ops.aten.mul.Tensor(add_76, view_26);  view_26 = None
        permute_151 = torch.ops.aten.permute.default(mul_162, [0, 2, 3, 1]);  mul_162 = None
        view_100 = torch.ops.aten.view.default(permute_151, [6272, 512]);  permute_151 = None
        mm_39 = torch.ops.aten.mm.default(view_100, arg413_1);  view_100 = arg413_1 = None
        view_101 = torch.ops.aten.view.default(mm_39, [32, 14, 14, 2048]);  mm_39 = None
        mul_163 = torch.ops.aten.mul.Tensor(view_101, arg414_1);  view_101 = arg414_1 = None
        view_103 = torch.ops.aten.view.default(mul_163, [6272, 2048]);  mul_163 = None
        mm_41 = torch.ops.aten.mm.default(view_103, arg415_1);  view_103 = arg415_1 = None
        view_104 = torch.ops.aten.view.default(mm_41, [32, 14, 14, 512]);  mm_41 = None
        mul_164 = torch.ops.aten.mul.Tensor(view_104, arg55_1);  view_104 = arg55_1 = None
        mul_165 = torch.ops.aten.mul.Tensor(mul_164, 512)
        sum_76 = torch.ops.aten.sum.dim_IntList(mul_164, [3], True)
        mul_166 = torch.ops.aten.mul.Tensor(mul_164, mul_50);  mul_164 = None
        sum_77 = torch.ops.aten.sum.dim_IntList(mul_166, [3], True);  mul_166 = None
        mul_167 = torch.ops.aten.mul.Tensor(mul_50, sum_77);  mul_50 = sum_77 = None
        sub_58 = torch.ops.aten.sub.Tensor(mul_165, sum_76);  mul_165 = sum_76 = None
        sub_59 = torch.ops.aten.sub.Tensor(sub_58, mul_167);  sub_58 = mul_167 = None
        div_10 = torch.ops.aten.div.Tensor(arg320_1, 512);  arg320_1 = None
        mul_168 = torch.ops.aten.mul.Tensor(div_10, sub_59);  div_10 = sub_59 = None
        permute_158 = torch.ops.aten.permute.default(mul_168, [0, 3, 1, 2]);  mul_168 = None
        convolution_backward_10 = torch.ops.aten.convolution_backward.default(permute_158, add_49, arg132_1, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, True]);  permute_158 = add_49 = arg132_1 = None
        getitem_30 = convolution_backward_10[0];  convolution_backward_10 = None
        add_77 = torch.ops.aten.add.Tensor(add_76, getitem_30);  add_76 = getitem_30 = None
        mul_171 = torch.ops.aten.mul.Tensor(add_77, view_25);  view_25 = None
        permute_159 = torch.ops.aten.permute.default(mul_171, [0, 2, 3, 1]);  mul_171 = None
        view_107 = torch.ops.aten.view.default(permute_159, [6272, 512]);  permute_159 = None
        mm_43 = torch.ops.aten.mm.default(view_107, arg416_1);  view_107 = arg416_1 = None
        view_108 = torch.ops.aten.view.default(mm_43, [32, 14, 14, 2048]);  mm_43 = None
        mul_172 = torch.ops.aten.mul.Tensor(view_108, arg417_1);  view_108 = arg417_1 = None
        view_110 = torch.ops.aten.view.default(mul_172, [6272, 2048]);  mul_172 = None
        mm_45 = torch.ops.aten.mm.default(view_110, arg418_1);  view_110 = arg418_1 = None
        view_111 = torch.ops.aten.view.default(mm_45, [32, 14, 14, 512]);  mm_45 = None
        mul_173 = torch.ops.aten.mul.Tensor(view_111, arg53_1);  view_111 = arg53_1 = None
        mul_174 = torch.ops.aten.mul.Tensor(mul_173, 512)
        sum_83 = torch.ops.aten.sum.dim_IntList(mul_173, [3], True)
        mul_175 = torch.ops.aten.mul.Tensor(mul_173, mul_48);  mul_173 = None
        sum_84 = torch.ops.aten.sum.dim_IntList(mul_175, [3], True);  mul_175 = None
        mul_176 = torch.ops.aten.mul.Tensor(mul_48, sum_84);  mul_48 = sum_84 = None
        sub_60 = torch.ops.aten.sub.Tensor(mul_174, sum_83);  mul_174 = sum_83 = None
        sub_61 = torch.ops.aten.sub.Tensor(sub_60, mul_176);  sub_60 = mul_176 = None
        div_11 = torch.ops.aten.div.Tensor(arg314_1, 512);  arg314_1 = None
        mul_177 = torch.ops.aten.mul.Tensor(div_11, sub_61);  div_11 = sub_61 = None
        permute_166 = torch.ops.aten.permute.default(mul_177, [0, 3, 1, 2]);  mul_177 = None
        convolution_backward_11 = torch.ops.aten.convolution_backward.default(permute_166, add_47, arg130_1, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, True]);  permute_166 = add_47 = arg130_1 = None
        getitem_33 = convolution_backward_11[0];  convolution_backward_11 = None
        add_78 = torch.ops.aten.add.Tensor(add_77, getitem_33);  add_77 = getitem_33 = None
        mul_180 = torch.ops.aten.mul.Tensor(add_78, view_24);  view_24 = None
        permute_167 = torch.ops.aten.permute.default(mul_180, [0, 2, 3, 1]);  mul_180 = None
        view_114 = torch.ops.aten.view.default(permute_167, [6272, 512]);  permute_167 = None
        mm_47 = torch.ops.aten.mm.default(view_114, arg419_1);  view_114 = arg419_1 = None
        view_115 = torch.ops.aten.view.default(mm_47, [32, 14, 14, 2048]);  mm_47 = None
        mul_181 = torch.ops.aten.mul.Tensor(view_115, arg420_1);  view_115 = arg420_1 = None
        view_117 = torch.ops.aten.view.default(mul_181, [6272, 2048]);  mul_181 = None
        mm_49 = torch.ops.aten.mm.default(view_117, arg421_1);  view_117 = arg421_1 = None
        view_118 = torch.ops.aten.view.default(mm_49, [32, 14, 14, 512]);  mm_49 = None
        mul_182 = torch.ops.aten.mul.Tensor(view_118, arg51_1);  view_118 = arg51_1 = None
        mul_183 = torch.ops.aten.mul.Tensor(mul_182, 512)
        sum_90 = torch.ops.aten.sum.dim_IntList(mul_182, [3], True)
        mul_184 = torch.ops.aten.mul.Tensor(mul_182, mul_46);  mul_182 = None
        sum_91 = torch.ops.aten.sum.dim_IntList(mul_184, [3], True);  mul_184 = None
        mul_185 = torch.ops.aten.mul.Tensor(mul_46, sum_91);  mul_46 = sum_91 = None
        sub_62 = torch.ops.aten.sub.Tensor(mul_183, sum_90);  mul_183 = sum_90 = None
        sub_63 = torch.ops.aten.sub.Tensor(sub_62, mul_185);  sub_62 = mul_185 = None
        div_12 = torch.ops.aten.div.Tensor(arg308_1, 512);  arg308_1 = None
        mul_186 = torch.ops.aten.mul.Tensor(div_12, sub_63);  div_12 = sub_63 = None
        permute_174 = torch.ops.aten.permute.default(mul_186, [0, 3, 1, 2]);  mul_186 = None
        convolution_backward_12 = torch.ops.aten.convolution_backward.default(permute_174, add_45, arg128_1, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, True]);  permute_174 = add_45 = arg128_1 = None
        getitem_36 = convolution_backward_12[0];  convolution_backward_12 = None
        add_79 = torch.ops.aten.add.Tensor(add_78, getitem_36);  add_78 = getitem_36 = None
        mul_189 = torch.ops.aten.mul.Tensor(add_79, view_23);  view_23 = None
        permute_175 = torch.ops.aten.permute.default(mul_189, [0, 2, 3, 1]);  mul_189 = None
        view_121 = torch.ops.aten.view.default(permute_175, [6272, 512]);  permute_175 = None
        mm_51 = torch.ops.aten.mm.default(view_121, arg422_1);  view_121 = arg422_1 = None
        view_122 = torch.ops.aten.view.default(mm_51, [32, 14, 14, 2048]);  mm_51 = None
        mul_190 = torch.ops.aten.mul.Tensor(view_122, arg423_1);  view_122 = arg423_1 = None
        view_124 = torch.ops.aten.view.default(mul_190, [6272, 2048]);  mul_190 = None
        mm_53 = torch.ops.aten.mm.default(view_124, arg424_1);  view_124 = arg424_1 = None
        view_125 = torch.ops.aten.view.default(mm_53, [32, 14, 14, 512]);  mm_53 = None
        mul_191 = torch.ops.aten.mul.Tensor(view_125, arg49_1);  view_125 = arg49_1 = None
        mul_192 = torch.ops.aten.mul.Tensor(mul_191, 512)
        sum_97 = torch.ops.aten.sum.dim_IntList(mul_191, [3], True)
        mul_193 = torch.ops.aten.mul.Tensor(mul_191, mul_44);  mul_191 = None
        sum_98 = torch.ops.aten.sum.dim_IntList(mul_193, [3], True);  mul_193 = None
        mul_194 = torch.ops.aten.mul.Tensor(mul_44, sum_98);  mul_44 = sum_98 = None
        sub_64 = torch.ops.aten.sub.Tensor(mul_192, sum_97);  mul_192 = sum_97 = None
        sub_65 = torch.ops.aten.sub.Tensor(sub_64, mul_194);  sub_64 = mul_194 = None
        div_13 = torch.ops.aten.div.Tensor(arg302_1, 512);  arg302_1 = None
        mul_195 = torch.ops.aten.mul.Tensor(div_13, sub_65);  div_13 = sub_65 = None
        permute_182 = torch.ops.aten.permute.default(mul_195, [0, 3, 1, 2]);  mul_195 = None
        convolution_backward_13 = torch.ops.aten.convolution_backward.default(permute_182, add_43, arg126_1, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, True]);  permute_182 = add_43 = arg126_1 = None
        getitem_39 = convolution_backward_13[0];  convolution_backward_13 = None
        add_80 = torch.ops.aten.add.Tensor(add_79, getitem_39);  add_79 = getitem_39 = None
        mul_198 = torch.ops.aten.mul.Tensor(add_80, view_22);  view_22 = None
        permute_183 = torch.ops.aten.permute.default(mul_198, [0, 2, 3, 1]);  mul_198 = None
        view_128 = torch.ops.aten.view.default(permute_183, [6272, 512]);  permute_183 = None
        mm_55 = torch.ops.aten.mm.default(view_128, arg425_1);  view_128 = arg425_1 = None
        view_129 = torch.ops.aten.view.default(mm_55, [32, 14, 14, 2048]);  mm_55 = None
        mul_199 = torch.ops.aten.mul.Tensor(view_129, arg426_1);  view_129 = arg426_1 = None
        view_131 = torch.ops.aten.view.default(mul_199, [6272, 2048]);  mul_199 = None
        mm_57 = torch.ops.aten.mm.default(view_131, arg427_1);  view_131 = arg427_1 = None
        view_132 = torch.ops.aten.view.default(mm_57, [32, 14, 14, 512]);  mm_57 = None
        mul_200 = torch.ops.aten.mul.Tensor(view_132, arg47_1);  view_132 = arg47_1 = None
        mul_201 = torch.ops.aten.mul.Tensor(mul_200, 512)
        sum_104 = torch.ops.aten.sum.dim_IntList(mul_200, [3], True)
        mul_202 = torch.ops.aten.mul.Tensor(mul_200, mul_42);  mul_200 = None
        sum_105 = torch.ops.aten.sum.dim_IntList(mul_202, [3], True);  mul_202 = None
        mul_203 = torch.ops.aten.mul.Tensor(mul_42, sum_105);  mul_42 = sum_105 = None
        sub_66 = torch.ops.aten.sub.Tensor(mul_201, sum_104);  mul_201 = sum_104 = None
        sub_67 = torch.ops.aten.sub.Tensor(sub_66, mul_203);  sub_66 = mul_203 = None
        div_14 = torch.ops.aten.div.Tensor(arg296_1, 512);  arg296_1 = None
        mul_204 = torch.ops.aten.mul.Tensor(div_14, sub_67);  div_14 = sub_67 = None
        permute_190 = torch.ops.aten.permute.default(mul_204, [0, 3, 1, 2]);  mul_204 = None
        convolution_backward_14 = torch.ops.aten.convolution_backward.default(permute_190, add_41, arg124_1, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, True]);  permute_190 = add_41 = arg124_1 = None
        getitem_42 = convolution_backward_14[0];  convolution_backward_14 = None
        add_81 = torch.ops.aten.add.Tensor(add_80, getitem_42);  add_80 = getitem_42 = None
        mul_207 = torch.ops.aten.mul.Tensor(add_81, view_21);  view_21 = None
        permute_191 = torch.ops.aten.permute.default(mul_207, [0, 2, 3, 1]);  mul_207 = None
        view_135 = torch.ops.aten.view.default(permute_191, [6272, 512]);  permute_191 = None
        mm_59 = torch.ops.aten.mm.default(view_135, arg428_1);  view_135 = arg428_1 = None
        view_136 = torch.ops.aten.view.default(mm_59, [32, 14, 14, 2048]);  mm_59 = None
        mul_208 = torch.ops.aten.mul.Tensor(view_136, arg429_1);  view_136 = arg429_1 = None
        view_138 = torch.ops.aten.view.default(mul_208, [6272, 2048]);  mul_208 = None
        mm_61 = torch.ops.aten.mm.default(view_138, arg430_1);  view_138 = arg430_1 = None
        view_139 = torch.ops.aten.view.default(mm_61, [32, 14, 14, 512]);  mm_61 = None
        mul_209 = torch.ops.aten.mul.Tensor(view_139, arg45_1);  view_139 = arg45_1 = None
        mul_210 = torch.ops.aten.mul.Tensor(mul_209, 512)
        sum_111 = torch.ops.aten.sum.dim_IntList(mul_209, [3], True)
        mul_211 = torch.ops.aten.mul.Tensor(mul_209, mul_40);  mul_209 = None
        sum_112 = torch.ops.aten.sum.dim_IntList(mul_211, [3], True);  mul_211 = None
        mul_212 = torch.ops.aten.mul.Tensor(mul_40, sum_112);  mul_40 = sum_112 = None
        sub_68 = torch.ops.aten.sub.Tensor(mul_210, sum_111);  mul_210 = sum_111 = None
        sub_69 = torch.ops.aten.sub.Tensor(sub_68, mul_212);  sub_68 = mul_212 = None
        div_15 = torch.ops.aten.div.Tensor(arg290_1, 512);  arg290_1 = None
        mul_213 = torch.ops.aten.mul.Tensor(div_15, sub_69);  div_15 = sub_69 = None
        permute_198 = torch.ops.aten.permute.default(mul_213, [0, 3, 1, 2]);  mul_213 = None
        convolution_backward_15 = torch.ops.aten.convolution_backward.default(permute_198, add_39, arg122_1, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, True]);  permute_198 = add_39 = arg122_1 = None
        getitem_45 = convolution_backward_15[0];  convolution_backward_15 = None
        add_82 = torch.ops.aten.add.Tensor(add_81, getitem_45);  add_81 = getitem_45 = None
        mul_216 = torch.ops.aten.mul.Tensor(add_82, view_20);  view_20 = None
        permute_199 = torch.ops.aten.permute.default(mul_216, [0, 2, 3, 1]);  mul_216 = None
        view_142 = torch.ops.aten.view.default(permute_199, [6272, 512]);  permute_199 = None
        mm_63 = torch.ops.aten.mm.default(view_142, arg431_1);  view_142 = arg431_1 = None
        view_143 = torch.ops.aten.view.default(mm_63, [32, 14, 14, 2048]);  mm_63 = None
        mul_217 = torch.ops.aten.mul.Tensor(view_143, arg432_1);  view_143 = arg432_1 = None
        view_145 = torch.ops.aten.view.default(mul_217, [6272, 2048]);  mul_217 = None
        mm_65 = torch.ops.aten.mm.default(view_145, arg433_1);  view_145 = arg433_1 = None
        view_146 = torch.ops.aten.view.default(mm_65, [32, 14, 14, 512]);  mm_65 = None
        mul_218 = torch.ops.aten.mul.Tensor(view_146, arg43_1);  view_146 = arg43_1 = None
        mul_219 = torch.ops.aten.mul.Tensor(mul_218, 512)
        sum_118 = torch.ops.aten.sum.dim_IntList(mul_218, [3], True)
        mul_220 = torch.ops.aten.mul.Tensor(mul_218, mul_38);  mul_218 = None
        sum_119 = torch.ops.aten.sum.dim_IntList(mul_220, [3], True);  mul_220 = None
        mul_221 = torch.ops.aten.mul.Tensor(mul_38, sum_119);  mul_38 = sum_119 = None
        sub_70 = torch.ops.aten.sub.Tensor(mul_219, sum_118);  mul_219 = sum_118 = None
        sub_71 = torch.ops.aten.sub.Tensor(sub_70, mul_221);  sub_70 = mul_221 = None
        div_16 = torch.ops.aten.div.Tensor(arg284_1, 512);  arg284_1 = None
        mul_222 = torch.ops.aten.mul.Tensor(div_16, sub_71);  div_16 = sub_71 = None
        permute_206 = torch.ops.aten.permute.default(mul_222, [0, 3, 1, 2]);  mul_222 = None
        convolution_backward_16 = torch.ops.aten.convolution_backward.default(permute_206, add_37, arg120_1, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, True]);  permute_206 = add_37 = arg120_1 = None
        getitem_48 = convolution_backward_16[0];  convolution_backward_16 = None
        add_83 = torch.ops.aten.add.Tensor(add_82, getitem_48);  add_82 = getitem_48 = None
        mul_225 = torch.ops.aten.mul.Tensor(add_83, view_19);  view_19 = None
        permute_207 = torch.ops.aten.permute.default(mul_225, [0, 2, 3, 1]);  mul_225 = None
        view_149 = torch.ops.aten.view.default(permute_207, [6272, 512]);  permute_207 = None
        mm_67 = torch.ops.aten.mm.default(view_149, arg434_1);  view_149 = arg434_1 = None
        view_150 = torch.ops.aten.view.default(mm_67, [32, 14, 14, 2048]);  mm_67 = None
        mul_226 = torch.ops.aten.mul.Tensor(view_150, arg435_1);  view_150 = arg435_1 = None
        view_152 = torch.ops.aten.view.default(mul_226, [6272, 2048]);  mul_226 = None
        mm_69 = torch.ops.aten.mm.default(view_152, arg436_1);  view_152 = arg436_1 = None
        view_153 = torch.ops.aten.view.default(mm_69, [32, 14, 14, 512]);  mm_69 = None
        mul_227 = torch.ops.aten.mul.Tensor(view_153, arg41_1);  view_153 = arg41_1 = None
        mul_228 = torch.ops.aten.mul.Tensor(mul_227, 512)
        sum_125 = torch.ops.aten.sum.dim_IntList(mul_227, [3], True)
        mul_229 = torch.ops.aten.mul.Tensor(mul_227, mul_36);  mul_227 = None
        sum_126 = torch.ops.aten.sum.dim_IntList(mul_229, [3], True);  mul_229 = None
        mul_230 = torch.ops.aten.mul.Tensor(mul_36, sum_126);  mul_36 = sum_126 = None
        sub_72 = torch.ops.aten.sub.Tensor(mul_228, sum_125);  mul_228 = sum_125 = None
        sub_73 = torch.ops.aten.sub.Tensor(sub_72, mul_230);  sub_72 = mul_230 = None
        div_17 = torch.ops.aten.div.Tensor(arg278_1, 512);  arg278_1 = None
        mul_231 = torch.ops.aten.mul.Tensor(div_17, sub_73);  div_17 = sub_73 = None
        permute_214 = torch.ops.aten.permute.default(mul_231, [0, 3, 1, 2]);  mul_231 = None
        convolution_backward_17 = torch.ops.aten.convolution_backward.default(permute_214, add_35, arg118_1, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, True]);  permute_214 = add_35 = arg118_1 = None
        getitem_51 = convolution_backward_17[0];  convolution_backward_17 = None
        add_84 = torch.ops.aten.add.Tensor(add_83, getitem_51);  add_83 = getitem_51 = None
        mul_234 = torch.ops.aten.mul.Tensor(add_84, view_18);  view_18 = None
        permute_215 = torch.ops.aten.permute.default(mul_234, [0, 2, 3, 1]);  mul_234 = None
        view_156 = torch.ops.aten.view.default(permute_215, [6272, 512]);  permute_215 = None
        mm_71 = torch.ops.aten.mm.default(view_156, arg437_1);  view_156 = arg437_1 = None
        view_157 = torch.ops.aten.view.default(mm_71, [32, 14, 14, 2048]);  mm_71 = None
        mul_235 = torch.ops.aten.mul.Tensor(view_157, arg438_1);  view_157 = arg438_1 = None
        view_159 = torch.ops.aten.view.default(mul_235, [6272, 2048]);  mul_235 = None
        mm_73 = torch.ops.aten.mm.default(view_159, arg439_1);  view_159 = arg439_1 = None
        view_160 = torch.ops.aten.view.default(mm_73, [32, 14, 14, 512]);  mm_73 = None
        mul_236 = torch.ops.aten.mul.Tensor(view_160, arg39_1);  view_160 = arg39_1 = None
        mul_237 = torch.ops.aten.mul.Tensor(mul_236, 512)
        sum_132 = torch.ops.aten.sum.dim_IntList(mul_236, [3], True)
        mul_238 = torch.ops.aten.mul.Tensor(mul_236, mul_34);  mul_236 = None
        sum_133 = torch.ops.aten.sum.dim_IntList(mul_238, [3], True);  mul_238 = None
        mul_239 = torch.ops.aten.mul.Tensor(mul_34, sum_133);  mul_34 = sum_133 = None
        sub_74 = torch.ops.aten.sub.Tensor(mul_237, sum_132);  mul_237 = sum_132 = None
        sub_75 = torch.ops.aten.sub.Tensor(sub_74, mul_239);  sub_74 = mul_239 = None
        div_18 = torch.ops.aten.div.Tensor(arg272_1, 512);  arg272_1 = None
        mul_240 = torch.ops.aten.mul.Tensor(div_18, sub_75);  div_18 = sub_75 = None
        permute_222 = torch.ops.aten.permute.default(mul_240, [0, 3, 1, 2]);  mul_240 = None
        convolution_backward_18 = torch.ops.aten.convolution_backward.default(permute_222, add_33, arg116_1, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, True]);  permute_222 = add_33 = arg116_1 = None
        getitem_54 = convolution_backward_18[0];  convolution_backward_18 = None
        add_85 = torch.ops.aten.add.Tensor(add_84, getitem_54);  add_84 = getitem_54 = None
        mul_243 = torch.ops.aten.mul.Tensor(add_85, view_17);  view_17 = None
        permute_223 = torch.ops.aten.permute.default(mul_243, [0, 2, 3, 1]);  mul_243 = None
        view_163 = torch.ops.aten.view.default(permute_223, [6272, 512]);  permute_223 = None
        mm_75 = torch.ops.aten.mm.default(view_163, arg440_1);  view_163 = arg440_1 = None
        view_164 = torch.ops.aten.view.default(mm_75, [32, 14, 14, 2048]);  mm_75 = None
        mul_244 = torch.ops.aten.mul.Tensor(view_164, arg441_1);  view_164 = arg441_1 = None
        view_166 = torch.ops.aten.view.default(mul_244, [6272, 2048]);  mul_244 = None
        mm_77 = torch.ops.aten.mm.default(view_166, arg442_1);  view_166 = arg442_1 = None
        view_167 = torch.ops.aten.view.default(mm_77, [32, 14, 14, 512]);  mm_77 = None
        mul_245 = torch.ops.aten.mul.Tensor(view_167, arg37_1);  view_167 = arg37_1 = None
        mul_246 = torch.ops.aten.mul.Tensor(mul_245, 512)
        sum_139 = torch.ops.aten.sum.dim_IntList(mul_245, [3], True)
        mul_247 = torch.ops.aten.mul.Tensor(mul_245, mul_32);  mul_245 = None
        sum_140 = torch.ops.aten.sum.dim_IntList(mul_247, [3], True);  mul_247 = None
        mul_248 = torch.ops.aten.mul.Tensor(mul_32, sum_140);  mul_32 = sum_140 = None
        sub_76 = torch.ops.aten.sub.Tensor(mul_246, sum_139);  mul_246 = sum_139 = None
        sub_77 = torch.ops.aten.sub.Tensor(sub_76, mul_248);  sub_76 = mul_248 = None
        div_19 = torch.ops.aten.div.Tensor(arg266_1, 512);  arg266_1 = None
        mul_249 = torch.ops.aten.mul.Tensor(div_19, sub_77);  div_19 = sub_77 = None
        permute_230 = torch.ops.aten.permute.default(mul_249, [0, 3, 1, 2]);  mul_249 = None
        convolution_backward_19 = torch.ops.aten.convolution_backward.default(permute_230, add_31, arg114_1, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, True]);  permute_230 = add_31 = arg114_1 = None
        getitem_57 = convolution_backward_19[0];  convolution_backward_19 = None
        add_86 = torch.ops.aten.add.Tensor(add_85, getitem_57);  add_85 = getitem_57 = None
        mul_252 = torch.ops.aten.mul.Tensor(add_86, view_16);  view_16 = None
        permute_231 = torch.ops.aten.permute.default(mul_252, [0, 2, 3, 1]);  mul_252 = None
        view_170 = torch.ops.aten.view.default(permute_231, [6272, 512]);  permute_231 = None
        mm_79 = torch.ops.aten.mm.default(view_170, arg443_1);  view_170 = arg443_1 = None
        view_171 = torch.ops.aten.view.default(mm_79, [32, 14, 14, 2048]);  mm_79 = None
        mul_253 = torch.ops.aten.mul.Tensor(view_171, arg444_1);  view_171 = arg444_1 = None
        view_173 = torch.ops.aten.view.default(mul_253, [6272, 2048]);  mul_253 = None
        mm_81 = torch.ops.aten.mm.default(view_173, arg445_1);  view_173 = arg445_1 = None
        view_174 = torch.ops.aten.view.default(mm_81, [32, 14, 14, 512]);  mm_81 = None
        mul_254 = torch.ops.aten.mul.Tensor(view_174, arg35_1);  view_174 = arg35_1 = None
        mul_255 = torch.ops.aten.mul.Tensor(mul_254, 512)
        sum_146 = torch.ops.aten.sum.dim_IntList(mul_254, [3], True)
        mul_256 = torch.ops.aten.mul.Tensor(mul_254, mul_30);  mul_254 = None
        sum_147 = torch.ops.aten.sum.dim_IntList(mul_256, [3], True);  mul_256 = None
        mul_257 = torch.ops.aten.mul.Tensor(mul_30, sum_147);  mul_30 = sum_147 = None
        sub_78 = torch.ops.aten.sub.Tensor(mul_255, sum_146);  mul_255 = sum_146 = None
        sub_79 = torch.ops.aten.sub.Tensor(sub_78, mul_257);  sub_78 = mul_257 = None
        div_20 = torch.ops.aten.div.Tensor(arg260_1, 512);  arg260_1 = None
        mul_258 = torch.ops.aten.mul.Tensor(div_20, sub_79);  div_20 = sub_79 = None
        permute_238 = torch.ops.aten.permute.default(mul_258, [0, 3, 1, 2]);  mul_258 = None
        convolution_backward_20 = torch.ops.aten.convolution_backward.default(permute_238, add_29, arg112_1, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, True]);  permute_238 = add_29 = arg112_1 = None
        getitem_60 = convolution_backward_20[0];  convolution_backward_20 = None
        add_87 = torch.ops.aten.add.Tensor(add_86, getitem_60);  add_86 = getitem_60 = None
        mul_261 = torch.ops.aten.mul.Tensor(add_87, view_15);  view_15 = None
        permute_239 = torch.ops.aten.permute.default(mul_261, [0, 2, 3, 1]);  mul_261 = None
        view_177 = torch.ops.aten.view.default(permute_239, [6272, 512]);  permute_239 = None
        mm_83 = torch.ops.aten.mm.default(view_177, arg446_1);  view_177 = arg446_1 = None
        view_178 = torch.ops.aten.view.default(mm_83, [32, 14, 14, 2048]);  mm_83 = None
        mul_262 = torch.ops.aten.mul.Tensor(view_178, arg447_1);  view_178 = arg447_1 = None
        view_180 = torch.ops.aten.view.default(mul_262, [6272, 2048]);  mul_262 = None
        mm_85 = torch.ops.aten.mm.default(view_180, arg448_1);  view_180 = arg448_1 = None
        view_181 = torch.ops.aten.view.default(mm_85, [32, 14, 14, 512]);  mm_85 = None
        mul_263 = torch.ops.aten.mul.Tensor(view_181, arg33_1);  view_181 = arg33_1 = None
        mul_264 = torch.ops.aten.mul.Tensor(mul_263, 512)
        sum_153 = torch.ops.aten.sum.dim_IntList(mul_263, [3], True)
        mul_265 = torch.ops.aten.mul.Tensor(mul_263, mul_28);  mul_263 = None
        sum_154 = torch.ops.aten.sum.dim_IntList(mul_265, [3], True);  mul_265 = None
        mul_266 = torch.ops.aten.mul.Tensor(mul_28, sum_154);  mul_28 = sum_154 = None
        sub_80 = torch.ops.aten.sub.Tensor(mul_264, sum_153);  mul_264 = sum_153 = None
        sub_81 = torch.ops.aten.sub.Tensor(sub_80, mul_266);  sub_80 = mul_266 = None
        div_21 = torch.ops.aten.div.Tensor(arg254_1, 512);  arg254_1 = None
        mul_267 = torch.ops.aten.mul.Tensor(div_21, sub_81);  div_21 = sub_81 = None
        permute_246 = torch.ops.aten.permute.default(mul_267, [0, 3, 1, 2]);  mul_267 = None
        convolution_backward_21 = torch.ops.aten.convolution_backward.default(permute_246, add_27, arg110_1, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, True]);  permute_246 = add_27 = arg110_1 = None
        getitem_63 = convolution_backward_21[0];  convolution_backward_21 = None
        add_88 = torch.ops.aten.add.Tensor(add_87, getitem_63);  add_87 = getitem_63 = None
        mul_270 = torch.ops.aten.mul.Tensor(add_88, view_14);  view_14 = None
        permute_247 = torch.ops.aten.permute.default(mul_270, [0, 2, 3, 1]);  mul_270 = None
        view_184 = torch.ops.aten.view.default(permute_247, [6272, 512]);  permute_247 = None
        mm_87 = torch.ops.aten.mm.default(view_184, arg449_1);  view_184 = arg449_1 = None
        view_185 = torch.ops.aten.view.default(mm_87, [32, 14, 14, 2048]);  mm_87 = None
        mul_271 = torch.ops.aten.mul.Tensor(view_185, arg450_1);  view_185 = arg450_1 = None
        view_187 = torch.ops.aten.view.default(mul_271, [6272, 2048]);  mul_271 = None
        mm_89 = torch.ops.aten.mm.default(view_187, arg451_1);  view_187 = arg451_1 = None
        view_188 = torch.ops.aten.view.default(mm_89, [32, 14, 14, 512]);  mm_89 = None
        mul_272 = torch.ops.aten.mul.Tensor(view_188, arg31_1);  view_188 = arg31_1 = None
        mul_273 = torch.ops.aten.mul.Tensor(mul_272, 512)
        sum_160 = torch.ops.aten.sum.dim_IntList(mul_272, [3], True)
        mul_274 = torch.ops.aten.mul.Tensor(mul_272, mul_26);  mul_272 = None
        sum_161 = torch.ops.aten.sum.dim_IntList(mul_274, [3], True);  mul_274 = None
        mul_275 = torch.ops.aten.mul.Tensor(mul_26, sum_161);  mul_26 = sum_161 = None
        sub_82 = torch.ops.aten.sub.Tensor(mul_273, sum_160);  mul_273 = sum_160 = None
        sub_83 = torch.ops.aten.sub.Tensor(sub_82, mul_275);  sub_82 = mul_275 = None
        div_22 = torch.ops.aten.div.Tensor(arg248_1, 512);  arg248_1 = None
        mul_276 = torch.ops.aten.mul.Tensor(div_22, sub_83);  div_22 = sub_83 = None
        permute_254 = torch.ops.aten.permute.default(mul_276, [0, 3, 1, 2]);  mul_276 = None
        convolution_backward_22 = torch.ops.aten.convolution_backward.default(permute_254, add_25, arg108_1, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, True]);  permute_254 = add_25 = arg108_1 = None
        getitem_66 = convolution_backward_22[0];  convolution_backward_22 = None
        add_89 = torch.ops.aten.add.Tensor(add_88, getitem_66);  add_88 = getitem_66 = None
        mul_279 = torch.ops.aten.mul.Tensor(add_89, view_13);  view_13 = None
        permute_255 = torch.ops.aten.permute.default(mul_279, [0, 2, 3, 1]);  mul_279 = None
        view_191 = torch.ops.aten.view.default(permute_255, [6272, 512]);  permute_255 = None
        mm_91 = torch.ops.aten.mm.default(view_191, arg452_1);  view_191 = arg452_1 = None
        view_192 = torch.ops.aten.view.default(mm_91, [32, 14, 14, 2048]);  mm_91 = None
        mul_280 = torch.ops.aten.mul.Tensor(view_192, arg453_1);  view_192 = arg453_1 = None
        view_194 = torch.ops.aten.view.default(mul_280, [6272, 2048]);  mul_280 = None
        mm_93 = torch.ops.aten.mm.default(view_194, arg454_1);  view_194 = arg454_1 = None
        view_195 = torch.ops.aten.view.default(mm_93, [32, 14, 14, 512]);  mm_93 = None
        mul_281 = torch.ops.aten.mul.Tensor(view_195, arg29_1);  view_195 = arg29_1 = None
        mul_282 = torch.ops.aten.mul.Tensor(mul_281, 512)
        sum_167 = torch.ops.aten.sum.dim_IntList(mul_281, [3], True)
        mul_283 = torch.ops.aten.mul.Tensor(mul_281, mul_24);  mul_281 = None
        sum_168 = torch.ops.aten.sum.dim_IntList(mul_283, [3], True);  mul_283 = None
        mul_284 = torch.ops.aten.mul.Tensor(mul_24, sum_168);  mul_24 = sum_168 = None
        sub_84 = torch.ops.aten.sub.Tensor(mul_282, sum_167);  mul_282 = sum_167 = None
        sub_85 = torch.ops.aten.sub.Tensor(sub_84, mul_284);  sub_84 = mul_284 = None
        div_23 = torch.ops.aten.div.Tensor(arg242_1, 512);  arg242_1 = None
        mul_285 = torch.ops.aten.mul.Tensor(div_23, sub_85);  div_23 = sub_85 = None
        permute_262 = torch.ops.aten.permute.default(mul_285, [0, 3, 1, 2]);  mul_285 = None
        convolution_backward_23 = torch.ops.aten.convolution_backward.default(permute_262, add_23, arg106_1, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, True]);  permute_262 = add_23 = arg106_1 = None
        getitem_69 = convolution_backward_23[0];  convolution_backward_23 = None
        add_90 = torch.ops.aten.add.Tensor(add_89, getitem_69);  add_89 = getitem_69 = None
        mul_288 = torch.ops.aten.mul.Tensor(add_90, view_12);  view_12 = None
        permute_263 = torch.ops.aten.permute.default(mul_288, [0, 2, 3, 1]);  mul_288 = None
        view_198 = torch.ops.aten.view.default(permute_263, [6272, 512]);  permute_263 = None
        mm_95 = torch.ops.aten.mm.default(view_198, arg455_1);  view_198 = arg455_1 = None
        view_199 = torch.ops.aten.view.default(mm_95, [32, 14, 14, 2048]);  mm_95 = None
        mul_289 = torch.ops.aten.mul.Tensor(view_199, arg456_1);  view_199 = arg456_1 = None
        view_201 = torch.ops.aten.view.default(mul_289, [6272, 2048]);  mul_289 = None
        mm_97 = torch.ops.aten.mm.default(view_201, arg457_1);  view_201 = arg457_1 = None
        view_202 = torch.ops.aten.view.default(mm_97, [32, 14, 14, 512]);  mm_97 = None
        mul_290 = torch.ops.aten.mul.Tensor(view_202, arg27_1);  view_202 = arg27_1 = None
        mul_291 = torch.ops.aten.mul.Tensor(mul_290, 512)
        sum_174 = torch.ops.aten.sum.dim_IntList(mul_290, [3], True)
        mul_292 = torch.ops.aten.mul.Tensor(mul_290, mul_22);  mul_290 = None
        sum_175 = torch.ops.aten.sum.dim_IntList(mul_292, [3], True);  mul_292 = None
        mul_293 = torch.ops.aten.mul.Tensor(mul_22, sum_175);  mul_22 = sum_175 = None
        sub_86 = torch.ops.aten.sub.Tensor(mul_291, sum_174);  mul_291 = sum_174 = None
        sub_87 = torch.ops.aten.sub.Tensor(sub_86, mul_293);  sub_86 = mul_293 = None
        div_24 = torch.ops.aten.div.Tensor(arg236_1, 512);  arg236_1 = None
        mul_294 = torch.ops.aten.mul.Tensor(div_24, sub_87);  div_24 = sub_87 = None
        permute_270 = torch.ops.aten.permute.default(mul_294, [0, 3, 1, 2]);  mul_294 = None
        convolution_backward_24 = torch.ops.aten.convolution_backward.default(permute_270, add_21, arg104_1, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, True]);  permute_270 = add_21 = arg104_1 = None
        getitem_72 = convolution_backward_24[0];  convolution_backward_24 = None
        add_91 = torch.ops.aten.add.Tensor(add_90, getitem_72);  add_90 = getitem_72 = None
        mul_297 = torch.ops.aten.mul.Tensor(add_91, view_11);  view_11 = None
        permute_271 = torch.ops.aten.permute.default(mul_297, [0, 2, 3, 1]);  mul_297 = None
        view_205 = torch.ops.aten.view.default(permute_271, [6272, 512]);  permute_271 = None
        mm_99 = torch.ops.aten.mm.default(view_205, arg458_1);  view_205 = arg458_1 = None
        view_206 = torch.ops.aten.view.default(mm_99, [32, 14, 14, 2048]);  mm_99 = None
        mul_298 = torch.ops.aten.mul.Tensor(view_206, arg459_1);  view_206 = arg459_1 = None
        view_208 = torch.ops.aten.view.default(mul_298, [6272, 2048]);  mul_298 = None
        mm_101 = torch.ops.aten.mm.default(view_208, arg460_1);  view_208 = arg460_1 = None
        view_209 = torch.ops.aten.view.default(mm_101, [32, 14, 14, 512]);  mm_101 = None
        mul_299 = torch.ops.aten.mul.Tensor(view_209, arg25_1);  view_209 = arg25_1 = None
        mul_300 = torch.ops.aten.mul.Tensor(mul_299, 512)
        sum_181 = torch.ops.aten.sum.dim_IntList(mul_299, [3], True)
        mul_301 = torch.ops.aten.mul.Tensor(mul_299, mul_20);  mul_299 = None
        sum_182 = torch.ops.aten.sum.dim_IntList(mul_301, [3], True);  mul_301 = None
        mul_302 = torch.ops.aten.mul.Tensor(mul_20, sum_182);  mul_20 = sum_182 = None
        sub_88 = torch.ops.aten.sub.Tensor(mul_300, sum_181);  mul_300 = sum_181 = None
        sub_89 = torch.ops.aten.sub.Tensor(sub_88, mul_302);  sub_88 = mul_302 = None
        div_25 = torch.ops.aten.div.Tensor(arg230_1, 512);  arg230_1 = None
        mul_303 = torch.ops.aten.mul.Tensor(div_25, sub_89);  div_25 = sub_89 = None
        permute_278 = torch.ops.aten.permute.default(mul_303, [0, 3, 1, 2]);  mul_303 = None
        convolution_backward_25 = torch.ops.aten.convolution_backward.default(permute_278, add_19, arg102_1, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, True]);  permute_278 = add_19 = arg102_1 = None
        getitem_75 = convolution_backward_25[0];  convolution_backward_25 = None
        add_92 = torch.ops.aten.add.Tensor(add_91, getitem_75);  add_91 = getitem_75 = None
        mul_306 = torch.ops.aten.mul.Tensor(add_92, view_10);  view_10 = None
        permute_279 = torch.ops.aten.permute.default(mul_306, [0, 2, 3, 1]);  mul_306 = None
        view_212 = torch.ops.aten.view.default(permute_279, [6272, 512]);  permute_279 = None
        mm_103 = torch.ops.aten.mm.default(view_212, arg461_1);  view_212 = arg461_1 = None
        view_213 = torch.ops.aten.view.default(mm_103, [32, 14, 14, 2048]);  mm_103 = None
        mul_307 = torch.ops.aten.mul.Tensor(view_213, arg462_1);  view_213 = arg462_1 = None
        view_215 = torch.ops.aten.view.default(mul_307, [6272, 2048]);  mul_307 = None
        mm_105 = torch.ops.aten.mm.default(view_215, arg463_1);  view_215 = arg463_1 = None
        view_216 = torch.ops.aten.view.default(mm_105, [32, 14, 14, 512]);  mm_105 = None
        mul_308 = torch.ops.aten.mul.Tensor(view_216, arg23_1);  view_216 = arg23_1 = None
        mul_309 = torch.ops.aten.mul.Tensor(mul_308, 512)
        sum_188 = torch.ops.aten.sum.dim_IntList(mul_308, [3], True)
        mul_310 = torch.ops.aten.mul.Tensor(mul_308, mul_18);  mul_308 = None
        sum_189 = torch.ops.aten.sum.dim_IntList(mul_310, [3], True);  mul_310 = None
        mul_311 = torch.ops.aten.mul.Tensor(mul_18, sum_189);  mul_18 = sum_189 = None
        sub_90 = torch.ops.aten.sub.Tensor(mul_309, sum_188);  mul_309 = sum_188 = None
        sub_91 = torch.ops.aten.sub.Tensor(sub_90, mul_311);  sub_90 = mul_311 = None
        div_26 = torch.ops.aten.div.Tensor(arg224_1, 512);  arg224_1 = None
        mul_312 = torch.ops.aten.mul.Tensor(div_26, sub_91);  div_26 = sub_91 = None
        permute_286 = torch.ops.aten.permute.default(mul_312, [0, 3, 1, 2]);  mul_312 = None
        convolution_backward_26 = torch.ops.aten.convolution_backward.default(permute_286, add_17, arg100_1, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, True]);  permute_286 = add_17 = arg100_1 = None
        getitem_78 = convolution_backward_26[0];  convolution_backward_26 = None
        add_93 = torch.ops.aten.add.Tensor(add_92, getitem_78);  add_92 = getitem_78 = None
        mul_315 = torch.ops.aten.mul.Tensor(add_93, view_9);  view_9 = None
        permute_287 = torch.ops.aten.permute.default(mul_315, [0, 2, 3, 1]);  mul_315 = None
        view_219 = torch.ops.aten.view.default(permute_287, [6272, 512]);  permute_287 = None
        mm_107 = torch.ops.aten.mm.default(view_219, arg464_1);  view_219 = arg464_1 = None
        view_220 = torch.ops.aten.view.default(mm_107, [32, 14, 14, 2048]);  mm_107 = None
        mul_316 = torch.ops.aten.mul.Tensor(view_220, arg465_1);  view_220 = arg465_1 = None
        view_222 = torch.ops.aten.view.default(mul_316, [6272, 2048]);  mul_316 = None
        mm_109 = torch.ops.aten.mm.default(view_222, arg466_1);  view_222 = arg466_1 = None
        view_223 = torch.ops.aten.view.default(mm_109, [32, 14, 14, 512]);  mm_109 = None
        mul_317 = torch.ops.aten.mul.Tensor(view_223, arg21_1);  view_223 = arg21_1 = None
        mul_318 = torch.ops.aten.mul.Tensor(mul_317, 512)
        sum_195 = torch.ops.aten.sum.dim_IntList(mul_317, [3], True)
        mul_319 = torch.ops.aten.mul.Tensor(mul_317, mul_16);  mul_317 = None
        sum_196 = torch.ops.aten.sum.dim_IntList(mul_319, [3], True);  mul_319 = None
        mul_320 = torch.ops.aten.mul.Tensor(mul_16, sum_196);  mul_16 = sum_196 = None
        sub_92 = torch.ops.aten.sub.Tensor(mul_318, sum_195);  mul_318 = sum_195 = None
        sub_93 = torch.ops.aten.sub.Tensor(sub_92, mul_320);  sub_92 = mul_320 = None
        div_27 = torch.ops.aten.div.Tensor(arg218_1, 512);  arg218_1 = None
        mul_321 = torch.ops.aten.mul.Tensor(div_27, sub_93);  div_27 = sub_93 = None
        permute_294 = torch.ops.aten.permute.default(mul_321, [0, 3, 1, 2]);  mul_321 = None
        convolution_backward_27 = torch.ops.aten.convolution_backward.default(permute_294, add_15, arg98_1, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, True]);  permute_294 = add_15 = arg98_1 = None
        getitem_81 = convolution_backward_27[0];  convolution_backward_27 = None
        add_94 = torch.ops.aten.add.Tensor(add_93, getitem_81);  add_93 = getitem_81 = None
        mul_324 = torch.ops.aten.mul.Tensor(add_94, view_8);  add_94 = view_8 = None
        permute_295 = torch.ops.aten.permute.default(mul_324, [0, 2, 3, 1]);  mul_324 = None
        view_226 = torch.ops.aten.view.default(permute_295, [6272, 512]);  permute_295 = None
        mm_111 = torch.ops.aten.mm.default(view_226, arg467_1);  view_226 = arg467_1 = None
        view_227 = torch.ops.aten.view.default(mm_111, [32, 14, 14, 2048]);  mm_111 = None
        mul_325 = torch.ops.aten.mul.Tensor(view_227, arg468_1);  view_227 = arg468_1 = None
        view_229 = torch.ops.aten.view.default(mul_325, [6272, 2048]);  mul_325 = None
        permute_299 = torch.ops.aten.permute.default(view_229, [1, 0]);  view_229 = None
        mm_112 = torch.ops.aten.mm.default(permute_299, arg213_1);  permute_299 = arg213_1 = None
        permute_300 = torch.ops.aten.permute.default(mm_112, [1, 0]);  mm_112 = None
        permute_301 = torch.ops.aten.permute.default(permute_300, [1, 0]);  permute_300 = None
        return (permute_301,)
        
args = [((128,), (1,), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((256,), (1,), torch.float32, 'cuda'), ((256,), (1,), torch.float32, 'cuda'), ((256,), (1,), torch.float32, 'cuda'), ((256,), (1,), torch.float32, 'cuda'), ((256,), (1,), torch.float32, 'cuda'), ((256,), (1,), torch.float32, 'cuda'), ((256,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((1024,), (1,), torch.float32, 'cuda'), ((1024,), (1,), torch.float32, 'cuda'), ((1024,), (1,), torch.float32, 'cuda'), ((1024,), (1,), torch.float32, 'cuda'), ((1024,), (1,), torch.float32, 'cuda'), ((1024,), (1,), torch.float32, 'cuda'), ((1024,), (1,), torch.float32, 'cuda'), ((128, 3, 4, 4), (48, 16, 4, 1), torch.float32, 'cuda'), ((128, 1, 7, 7), (49, 49, 7, 1), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((128, 1, 7, 7), (49, 49, 7, 1), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((128, 1, 7, 7), (49, 49, 7, 1), torch.float32, 'cuda'), ((128,), (1,), torch.float32, 'cuda'), ((256, 128, 2, 2), (512, 4, 2, 1), torch.float32, 'cuda'), ((256, 1, 7, 7), (49, 49, 7, 1), torch.float32, 'cuda'), ((256,), (1,), torch.float32, 'cuda'), ((256, 1, 7, 7), (49, 49, 7, 1), torch.float32, 'cuda'), ((256,), (1,), torch.float32, 'cuda'), ((256, 1, 7, 7), (49, 49, 7, 1), torch.float32, 'cuda'), ((256,), (1,), torch.float32, 'cuda'), ((512, 256, 2, 2), (1024, 4, 2, 1), torch.float32, 'cuda'), ((512, 1, 7, 7), (49, 49, 7, 1), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512, 1, 7, 7), (49, 49, 7, 1), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512, 1, 7, 7), (49, 49, 7, 1), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512, 1, 7, 7), (49, 49, 7, 1), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512, 1, 7, 7), (49, 49, 7, 1), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512, 1, 7, 7), (49, 49, 7, 1), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512, 1, 7, 7), (49, 49, 7, 1), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512, 1, 7, 7), (49, 49, 7, 1), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512, 1, 7, 7), (49, 49, 7, 1), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512, 1, 7, 7), (49, 49, 7, 1), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512, 1, 7, 7), (49, 49, 7, 1), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512, 1, 7, 7), (49, 49, 7, 1), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512, 1, 7, 7), (49, 49, 7, 1), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512, 1, 7, 7), (49, 49, 7, 1), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512, 1, 7, 7), (49, 49, 7, 1), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512, 1, 7, 7), (49, 49, 7, 1), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512, 1, 7, 7), (49, 49, 7, 1), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512, 1, 7, 7), (49, 49, 7, 1), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512, 1, 7, 7), (49, 49, 7, 1), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512, 1, 7, 7), (49, 49, 7, 1), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512, 1, 7, 7), (49, 49, 7, 1), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512, 1, 7, 7), (49, 49, 7, 1), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512, 1, 7, 7), (49, 49, 7, 1), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512, 1, 7, 7), (49, 49, 7, 1), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512, 1, 7, 7), (49, 49, 7, 1), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512, 1, 7, 7), (49, 49, 7, 1), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((512, 1, 7, 7), (49, 49, 7, 1), torch.float32, 'cuda'), ((512,), (1,), torch.float32, 'cuda'), ((1024, 512, 2, 2), (2048, 4, 2, 1), torch.float32, 'cuda'), ((1024, 1, 7, 7), (49, 49, 7, 1), torch.float32, 'cuda'), ((1024,), (1,), torch.float32, 'cuda'), ((1024, 1, 7, 7), (49, 49, 7, 1), torch.float32, 'cuda'), ((1024,), (1,), torch.float32, 'cuda'), ((1024, 1, 7, 7), (49, 49, 7, 1), torch.float32, 'cuda'), ((1024,), (1,), torch.float32, 'cuda'), ((32, 3, 224, 224), (150528, 50176, 224, 1), torch.float32, 'cuda'), ((32, 56, 56, 128), (401408, 56, 1, 3136), torch.float32, 'cuda'), ((32, 128, 56, 56), (401408, 1, 7168, 128), torch.float32, 'cuda'), ((32, 128, 56, 56), (401408, 3136, 56, 1), torch.float32, 'cuda'), ((32, 56, 56, 1), (3136, 56, 1, 100352), torch.float32, 'cuda'), ((32, 56, 56, 1), (3136, 56, 1, 100352), torch.float32, 'cuda'), ((100352, 128), (128, 1), torch.float32, 'cuda'), ((100352, 512), (512, 1), torch.float32, 'cuda'), ((100352, 128), (128, 1), torch.float32, 'cuda'), ((32, 128, 56, 56), (401408, 3136, 56, 1), torch.float32, 'cuda'), ((32, 56, 56, 1), (3136, 56, 1, 100352), torch.float32, 'cuda'), ((32, 56, 56, 1), (3136, 56, 1, 100352), torch.float32, 'cuda'), ((100352, 128), (128, 1), torch.float32, 'cuda'), ((100352, 512), (512, 1), torch.float32, 'cuda'), ((100352, 128), (128, 1), torch.float32, 'cuda'), ((32, 128, 56, 56), (401408, 3136, 56, 1), torch.float32, 'cuda'), ((32, 56, 56, 1), (3136, 56, 1, 100352), torch.float32, 'cuda'), ((32, 56, 56, 1), (3136, 56, 1, 100352), torch.float32, 'cuda'), ((100352, 128), (128, 1), torch.float32, 'cuda'), ((100352, 512), (512, 1), torch.float32, 'cuda'), ((100352, 128), (128, 1), torch.float32, 'cuda'), ((32, 56, 56, 128), (401408, 7168, 128, 1), torch.float32, 'cuda'), ((32, 128, 56, 56), (401408, 1, 7168, 128), torch.float32, 'cuda'), ((32, 256, 28, 28), (200704, 1, 7168, 256), torch.float32, 'cuda'), ((32, 256, 28, 28), (200704, 784, 28, 1), torch.float32, 'cuda'), ((32, 28, 28, 1), (784, 28, 1, 25088), torch.float32, 'cuda'), ((32, 28, 28, 1), (784, 28, 1, 25088), torch.float32, 'cuda'), ((25088, 256), (256, 1), torch.float32, 'cuda'), ((25088, 1024), (1024, 1), torch.float32, 'cuda'), ((25088, 256), (256, 1), torch.float32, 'cuda'), ((32, 256, 28, 28), (200704, 784, 28, 1), torch.float32, 'cuda'), ((32, 28, 28, 1), (784, 28, 1, 25088), torch.float32, 'cuda'), ((32, 28, 28, 1), (784, 28, 1, 25088), torch.float32, 'cuda'), ((25088, 256), (256, 1), torch.float32, 'cuda'), ((25088, 1024), (1024, 1), torch.float32, 'cuda'), ((25088, 256), (256, 1), torch.float32, 'cuda'), ((32, 256, 28, 28), (200704, 784, 28, 1), torch.float32, 'cuda'), ((32, 28, 28, 1), (784, 28, 1, 25088), torch.float32, 'cuda'), ((32, 28, 28, 1), (784, 28, 1, 25088), torch.float32, 'cuda'), ((25088, 256), (256, 1), torch.float32, 'cuda'), ((25088, 1024), (1024, 1), torch.float32, 'cuda'), ((25088, 256), (256, 1), torch.float32, 'cuda'), ((32, 28, 28, 256), (200704, 7168, 256, 1), torch.float32, 'cuda'), ((32, 256, 28, 28), (200704, 1, 7168, 256), torch.float32, 'cuda'), ((32, 512, 14, 14), (100352, 1, 7168, 512), torch.float32, 'cuda'), ((32, 512, 14, 14), (100352, 196, 14, 1), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((6272, 2048), (2048, 1), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((32, 512, 14, 14), (100352, 196, 14, 1), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((6272, 2048), (2048, 1), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((32, 512, 14, 14), (100352, 196, 14, 1), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((6272, 2048), (2048, 1), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((32, 512, 14, 14), (100352, 196, 14, 1), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((6272, 2048), (2048, 1), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((32, 512, 14, 14), (100352, 196, 14, 1), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((6272, 2048), (2048, 1), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((32, 512, 14, 14), (100352, 196, 14, 1), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((6272, 2048), (2048, 1), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((32, 512, 14, 14), (100352, 196, 14, 1), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((6272, 2048), (2048, 1), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((32, 512, 14, 14), (100352, 196, 14, 1), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((6272, 2048), (2048, 1), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((32, 512, 14, 14), (100352, 196, 14, 1), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((6272, 2048), (2048, 1), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((32, 512, 14, 14), (100352, 196, 14, 1), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((6272, 2048), (2048, 1), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((32, 512, 14, 14), (100352, 196, 14, 1), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((6272, 2048), (2048, 1), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((32, 512, 14, 14), (100352, 196, 14, 1), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((6272, 2048), (2048, 1), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((32, 512, 14, 14), (100352, 196, 14, 1), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((6272, 2048), (2048, 1), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((32, 512, 14, 14), (100352, 196, 14, 1), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((6272, 2048), (2048, 1), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((32, 512, 14, 14), (100352, 196, 14, 1), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((6272, 2048), (2048, 1), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((32, 512, 14, 14), (100352, 196, 14, 1), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((6272, 2048), (2048, 1), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((32, 512, 14, 14), (100352, 196, 14, 1), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((6272, 2048), (2048, 1), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((32, 512, 14, 14), (100352, 196, 14, 1), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((6272, 2048), (2048, 1), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((32, 512, 14, 14), (100352, 196, 14, 1), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((6272, 2048), (2048, 1), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((32, 512, 14, 14), (100352, 196, 14, 1), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((6272, 2048), (2048, 1), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((32, 512, 14, 14), (100352, 196, 14, 1), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((6272, 2048), (2048, 1), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((32, 512, 14, 14), (100352, 196, 14, 1), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((6272, 2048), (2048, 1), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((32, 512, 14, 14), (100352, 196, 14, 1), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((6272, 2048), (2048, 1), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((32, 512, 14, 14), (100352, 196, 14, 1), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((6272, 2048), (2048, 1), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((32, 512, 14, 14), (100352, 196, 14, 1), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((6272, 2048), (2048, 1), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((32, 512, 14, 14), (100352, 196, 14, 1), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((6272, 2048), (2048, 1), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((32, 512, 14, 14), (100352, 196, 14, 1), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((6272, 2048), (2048, 1), torch.float32, 'cuda'), ((6272, 512), (512, 1), torch.float32, 'cuda'), ((32, 14, 14, 512), (100352, 7168, 512, 1), torch.float32, 'cuda'), ((32, 512, 14, 14), (100352, 1, 7168, 512), torch.float32, 'cuda'), ((32, 1024, 7, 7), (50176, 1, 7168, 1024), torch.float32, 'cuda'), ((32, 1024, 7, 7), (50176, 49, 7, 1), torch.float32, 'cuda'), ((32, 7, 7, 1), (49, 7, 1, 1568), torch.float32, 'cuda'), ((32, 7, 7, 1), (49, 7, 1, 1568), torch.float32, 'cuda'), ((1568, 1024), (1024, 1), torch.float32, 'cuda'), ((1568, 4096), (4096, 1), torch.float32, 'cuda'), ((1568, 1024), (1024, 1), torch.float32, 'cuda'), ((32, 1024, 7, 7), (50176, 49, 7, 1), torch.float32, 'cuda'), ((32, 7, 7, 1), (49, 7, 1, 1568), torch.float32, 'cuda'), ((32, 7, 7, 1), (49, 7, 1, 1568), torch.float32, 'cuda'), ((1568, 1024), (1024, 1), torch.float32, 'cuda'), ((1568, 4096), (4096, 1), torch.float32, 'cuda'), ((1568, 1024), (1024, 1), torch.float32, 'cuda'), ((32, 1024, 7, 7), (50176, 49, 7, 1), torch.float32, 'cuda'), ((32, 7, 7, 1), (49, 7, 1, 1568), torch.float32, 'cuda'), ((32, 7, 7, 1), (49, 7, 1, 1568), torch.float32, 'cuda'), ((1568, 1024), (1024, 1), torch.float32, 'cuda'), ((1568, 4096), (4096, 1), torch.float32, 'cuda'), ((1568, 1024), (1024, 1), torch.float32, 'cuda'), ((32, 1, 1, 1024), (1024, 32768, 32768, 1), torch.float32, 'cuda'), ((32, 1024), (1024, 1), torch.float32, 'cuda'), ((1000, 1024), (1024, 1), torch.float32, 'cuda'), ((32, 1, 1, 1), (1, 32, 32, 32), torch.float32, 'cuda'), ((1024, 4096), (4096, 1), torch.float32, 'cuda'), ((32, 7, 7, 4096), (200704, 28672, 4096, 1), torch.float32, 'cuda'), ((4096, 1024), (1024, 1), torch.float32, 'cuda'), ((1024, 4096), (4096, 1), torch.float32, 'cuda'), ((32, 7, 7, 4096), (200704, 28672, 4096, 1), torch.float32, 'cuda'), ((4096, 1024), (1024, 1), torch.float32, 'cuda'), ((1024, 4096), (4096, 1), torch.float32, 'cuda'), ((32, 7, 7, 4096), (200704, 28672, 4096, 1), torch.float32, 'cuda'), ((4096, 1024), (1024, 1), torch.float32, 'cuda'), ((32, 14, 14, 1), (196, 14, 1, 6272), torch.float32, 'cuda'), ((512, 2048), (2048, 1), torch.float32, 'cuda'), ((32, 14, 14, 2048), (401408, 28672, 2048, 1), torch.float32, 'cuda'), ((2048, 512), (512, 1), torch.float32, 'cuda'), ((512, 2048), (2048, 1), torch.float32, 'cuda'), ((32, 14, 14, 2048), (401408, 28672, 2048, 1), torch.float32, 'cuda'), ((2048, 512), (512, 1), torch.float32, 'cuda'), ((512, 2048), (2048, 1), torch.float32, 'cuda'), ((32, 14, 14, 2048), (401408, 28672, 2048, 1), torch.float32, 'cuda'), ((2048, 512), (512, 1), torch.float32, 'cuda'), ((512, 2048), (2048, 1), torch.float32, 'cuda'), ((32, 14, 14, 2048), (401408, 28672, 2048, 1), torch.float32, 'cuda'), ((2048, 512), (512, 1), torch.float32, 'cuda'), ((512, 2048), (2048, 1), torch.float32, 'cuda'), ((32, 14, 14, 2048), (401408, 28672, 2048, 1), torch.float32, 'cuda'), ((2048, 512), (512, 1), torch.float32, 'cuda'), ((512, 2048), (2048, 1), torch.float32, 'cuda'), ((32, 14, 14, 2048), (401408, 28672, 2048, 1), torch.float32, 'cuda'), ((2048, 512), (512, 1), torch.float32, 'cuda'), ((512, 2048), (2048, 1), torch.float32, 'cuda'), ((32, 14, 14, 2048), (401408, 28672, 2048, 1), torch.float32, 'cuda'), ((2048, 512), (512, 1), torch.float32, 'cuda'), ((512, 2048), (2048, 1), torch.float32, 'cuda'), ((32, 14, 14, 2048), (401408, 28672, 2048, 1), torch.float32, 'cuda'), ((2048, 512), (512, 1), torch.float32, 'cuda'), ((512, 2048), (2048, 1), torch.float32, 'cuda'), ((32, 14, 14, 2048), (401408, 28672, 2048, 1), torch.float32, 'cuda'), ((2048, 512), (512, 1), torch.float32, 'cuda'), ((512, 2048), (2048, 1), torch.float32, 'cuda'), ((32, 14, 14, 2048), (401408, 28672, 2048, 1), torch.float32, 'cuda'), ((2048, 512), (512, 1), torch.float32, 'cuda'), ((512, 2048), (2048, 1), torch.float32, 'cuda'), ((32, 14, 14, 2048), (401408, 28672, 2048, 1), torch.float32, 'cuda'), ((2048, 512), (512, 1), torch.float32, 'cuda'), ((512, 2048), (2048, 1), torch.float32, 'cuda'), ((32, 14, 14, 2048), (401408, 28672, 2048, 1), torch.float32, 'cuda'), ((2048, 512), (512, 1), torch.float32, 'cuda'), ((512, 2048), (2048, 1), torch.float32, 'cuda'), ((32, 14, 14, 2048), (401408, 28672, 2048, 1), torch.float32, 'cuda'), ((2048, 512), (512, 1), torch.float32, 'cuda'), ((512, 2048), (2048, 1), torch.float32, 'cuda'), ((32, 14, 14, 2048), (401408, 28672, 2048, 1), torch.float32, 'cuda'), ((2048, 512), (512, 1), torch.float32, 'cuda'), ((512, 2048), (2048, 1), torch.float32, 'cuda'), ((32, 14, 14, 2048), (401408, 28672, 2048, 1), torch.float32, 'cuda'), ((2048, 512), (512, 1), torch.float32, 'cuda'), ((512, 2048), (2048, 1), torch.float32, 'cuda'), ((32, 14, 14, 2048), (401408, 28672, 2048, 1), torch.float32, 'cuda'), ((2048, 512), (512, 1), torch.float32, 'cuda'), ((512, 2048), (2048, 1), torch.float32, 'cuda'), ((32, 14, 14, 2048), (401408, 28672, 2048, 1), torch.float32, 'cuda'), ((2048, 512), (512, 1), torch.float32, 'cuda'), ((512, 2048), (2048, 1), torch.float32, 'cuda'), ((32, 14, 14, 2048), (401408, 28672, 2048, 1), torch.float32, 'cuda'), ((2048, 512), (512, 1), torch.float32, 'cuda'), ((512, 2048), (2048, 1), torch.float32, 'cuda'), ((32, 14, 14, 2048), (401408, 28672, 2048, 1), torch.float32, 'cuda'), ((2048, 512), (512, 1), torch.float32, 'cuda'), ((512, 2048), (2048, 1), torch.float32, 'cuda'), ((32, 14, 14, 2048), (401408, 28672, 2048, 1), torch.float32, 'cuda'), ((2048, 512), (512, 1), torch.float32, 'cuda'), ((512, 2048), (2048, 1), torch.float32, 'cuda'), ((32, 14, 14, 2048), (401408, 28672, 2048, 1), torch.float32, 'cuda'), ((2048, 512), (512, 1), torch.float32, 'cuda'), ((512, 2048), (2048, 1), torch.float32, 'cuda'), ((32, 14, 14, 2048), (401408, 28672, 2048, 1), torch.float32, 'cuda'), ((2048, 512), (512, 1), torch.float32, 'cuda'), ((512, 2048), (2048, 1), torch.float32, 'cuda'), ((32, 14, 14, 2048), (401408, 28672, 2048, 1), torch.float32, 'cuda'), ((2048, 512), (512, 1), torch.float32, 'cuda'), ((512, 2048), (2048, 1), torch.float32, 'cuda'), ((32, 14, 14, 2048), (401408, 28672, 2048, 1), torch.float32, 'cuda'), ((2048, 512), (512, 1), torch.float32, 'cuda'), ((512, 2048), (2048, 1), torch.float32, 'cuda'), ((32, 14, 14, 2048), (401408, 28672, 2048, 1), torch.float32, 'cuda'), ((2048, 512), (512, 1), torch.float32, 'cuda'), ((512, 2048), (2048, 1), torch.float32, 'cuda'), ((32, 14, 14, 2048), (401408, 28672, 2048, 1), torch.float32, 'cuda'), ((2048, 512), (512, 1), torch.float32, 'cuda'), ((512, 2048), (2048, 1), torch.float32, 'cuda'), ((32, 14, 14, 2048), (401408, 28672, 2048, 1), torch.float32, 'cuda'), ((2048, 512), (512, 1), torch.float32, 'cuda'), ((32, 28, 28, 1), (784, 28, 1, 25088), torch.float32, 'cuda'), ((256, 1024), (1024, 1), torch.float32, 'cuda'), ((32, 28, 28, 1024), (802816, 28672, 1024, 1), torch.float32, 'cuda'), ((1024, 256), (256, 1), torch.float32, 'cuda'), ((256, 1024), (1024, 1), torch.float32, 'cuda'), ((32, 28, 28, 1024), (802816, 28672, 1024, 1), torch.float32, 'cuda'), ((1024, 256), (256, 1), torch.float32, 'cuda'), ((256, 1024), (1024, 1), torch.float32, 'cuda'), ((32, 28, 28, 1024), (802816, 28672, 1024, 1), torch.float32, 'cuda'), ((1024, 256), (256, 1), torch.float32, 'cuda'), ((32, 56, 56, 1), (3136, 56, 1, 100352), torch.float32, 'cuda'), ((128, 512), (512, 1), torch.float32, 'cuda'), ((32, 56, 56, 512), (1605632, 28672, 512, 1), torch.float32, 'cuda'), ((512, 128), (128, 1), torch.float32, 'cuda'), ((128, 512), (512, 1), torch.float32, 'cuda'), ((32, 56, 56, 512), (1605632, 28672, 512, 1), torch.float32, 'cuda'), ((512, 128), (128, 1), torch.float32, 'cuda'), ((128, 512), (512, 1), torch.float32, 'cuda'), ((32, 56, 56, 512), (1605632, 28672, 512, 1), torch.float32, 'cuda'), ((512, 128), (128, 1), torch.float32, 'cuda'), ((32, 56, 56, 1), (3136, 56, 1, 100352), torch.float32, 'cuda'), ((32, 1000), (1000, 1), torch.float32, 'cuda')]
args = [rand_strided(sh, st, dt, dev) for (sh, st, dt, dev) in args]
mod = make_fx(Repro().to(device="cuda"))(*args)

from torch._dynamo.debug_utils import inductor_fails

if inductor_fails(mod, args):
    exit(1)
else:
    exit(0)
