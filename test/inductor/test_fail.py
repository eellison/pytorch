import torch
from torch import device
torch._inductor.config.force_disable_caches = True

def forward(self, arg0_1: "f8e8m0fnu[1703936][1]cuda:0", arg1_1: "f8e4m3fn[13312, 4096][4096, 1]cuda:0", arg2_1: "bf16[1024, 4096][4096, 1]cuda:0", arg3_1: "f8e8m0fnu[1703936][1]cuda:0", arg4_1: "f8e4m3fn[13312, 4096][4096, 1]cuda:0", arg5_1: "f8e8m0fnu[1703936][1]cuda:0", arg6_1: "f8e4m3fn[4096, 13312][13312, 1]cuda:0"):
        # File: /home/drisspg/meta/transformer_nuggets/transformer_nuggets/misc/mlp.py:47 in forward, code: return self.w2(F.silu(self.w1(x)) * self.w3(x))
    view: "bf16[131072, 32][32, 1]cuda:0" = torch.ops.aten.reshape.default(arg2_1, [-1, 32])
    abs_1: "bf16[131072, 32][32, 1]cuda:0" = torch.ops.aten.abs.default(view)
    amax: "bf16[131072][1]cuda:0" = torch.ops.aten.amax.default(abs_1, [1]);  abs_1 = None
    isnan: "b8[131072][1]cuda:0" = torch.ops.aten.isnan.default(amax)
    scalar_tensor: "u8[][]cuda:0" = torch.ops.aten.scalar_tensor.default(255, dtype = torch.uint8, layout = torch.strided, device = device(type='cuda', index=0))
    eq: "b8[131072][1]cuda:0" = torch.ops.aten.eq.Scalar(amax, 0)
    convert_element_type: "bf16[131072][1]cuda:0" = torch.ops.prims.convert_element_type.default(eq, torch.bfloat16);  eq = None
    mul: "bf16[131072][1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type, 1.1754943508222875e-38);  convert_element_type = None
    add: "bf16[131072][1]cuda:0" = torch.ops.aten.add.Tensor(amax, mul);  amax = mul = None
    view_1: "i16[131072][1]cuda:0" = torch.ops.aten.view.dtype(add, torch.int16);  add = None
    rshift: "i16[131072][1]cuda:0" = torch.ops.aten.__rshift__.Scalar(view_1, 7);  view_1 = None
    bitwise_and: "i16[131072][1]cuda:0" = torch.ops.aten.bitwise_and.Scalar(rshift, 255);  rshift = None
    sub: "i16[131072][1]cuda:0" = torch.ops.aten.sub.Tensor(bitwise_and, 127);  bitwise_and = None
    sub_1: "i16[131072][1]cuda:0" = torch.ops.aten.sub.Tensor(sub, 8);  sub = None
    clamp_min: "i16[131072][1]cuda:0" = torch.ops.aten.clamp_min.default(sub_1, -127);  sub_1 = None
    clamp_max: "i16[131072][1]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min, 128);  clamp_min = None
    add_1: "i16[131072][1]cuda:0" = torch.ops.aten.add.Tensor(clamp_max, 127);  clamp_max = None
    convert_element_type_1: "u8[131072][1]cuda:0" = torch.ops.prims.convert_element_type.default(add_1, torch.uint8);  add_1 = None
    where: "u8[131072][1]cuda:0" = torch.ops.aten.where.self(isnan, scalar_tensor, convert_element_type_1);  isnan = scalar_tensor = convert_element_type_1 = None
    convert_element_type_2: "i32[131072][1]cuda:0" = torch.ops.prims.convert_element_type.default(where, torch.int32)
    lshift: "i32[131072][1]cuda:0" = torch.ops.aten.__lshift__.Scalar(convert_element_type_2, 23);  convert_element_type_2 = None
    view_2: "f32[131072][1]cuda:0" = torch.ops.aten.view.dtype(lshift, torch.float32);  lshift = None
    clamp_min_1: "f32[131072][1]cuda:0" = torch.ops.aten.clamp_min.default(view_2, 1.1754943508222875e-38);  view_2 = None
    unsqueeze: "f32[131072, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clamp_min_1, 1);  clamp_min_1 = None
    div: "f32[131072, 32][32, 1]cuda:0" = torch.ops.aten.div.Tensor(view, unsqueeze);  view = unsqueeze = None
    clamp_min_2: "f32[131072, 32][32, 1]cuda:0" = torch.ops.aten.clamp_min.default(div, -448.0);  div = None
    clamp_max_1: "f32[131072, 32][32, 1]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_2, 448.0);  clamp_min_2 = None
    convert_element_type_3: "f8e4m3fn[131072, 32][32, 1]cuda:0" = torch.ops.prims.convert_element_type.default(clamp_max_1, torch.float8_e4m3fn);  clamp_max_1 = None
    view_3: "f8e4m3fn[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_3, [1024, 4096]);  convert_element_type_3 = None
    permute: "f8e4m3fn[4096, 13312][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
    view_4: "f8e8m0fnu[131072][1]cuda:0" = torch.ops.aten.view.dtype(where, torch.float8_e8m0fnu);  where = None
    view_5: "f8e8m0fnu[1024, 128][128, 1]cuda:0" = torch.ops.aten.reshape.default(view_4, [1024, 128]);  view_4 = None
    view_7: "f8e8m0fnu[8, 128, 32, 4][16384, 128, 4, 1]cuda:0" = torch.ops.aten.reshape.default(view_5, [8, 128, 32, 4]);  view_5 = None
    permute_2: "f8e8m0fnu[8, 32, 128, 4][16384, 4, 128, 1]cuda:0" = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None
    clone: "f8e8m0fnu[8, 32, 128, 4][16384, 512, 4, 1]cuda:0" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
    view_8: "f8e8m0fnu[256, 4, 32, 4][512, 128, 4, 1]cuda:0" = torch.ops.aten.reshape.default(clone, [256, 4, 32, 4]);  clone = None
    permute_3: "f8e8m0fnu[256, 32, 4, 4][512, 4, 128, 1]cuda:0" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
    clone_1: "f8e8m0fnu[256, 32, 4, 4][512, 16, 4, 1]cuda:0" = torch.ops.aten.clone.default(permute_3, memory_format = torch.contiguous_format);  permute_3 = None
    view_9: "f8e8m0fnu[256, 32, 16][512, 16, 1]cuda:0" = torch.ops.aten.reshape.default(clone_1, [256, 32, 16]);  clone_1 = None
    view_10: "f8e8m0fnu[131072][1]cuda:0" = torch.ops.aten.reshape.default(view_9, [131072]);  view_9 = None
    view_6: "f8e8m0fnu[13312, 128][128, 1]cuda:0" = torch.ops.aten.reshape.default(arg0_1, [13312, 128]);  arg0_1 = None
    view_11: "f8e8m0fnu[104, 128, 32, 4][16384, 128, 4, 1]cuda:0" = torch.ops.aten.reshape.default(view_6, [104, 128, 32, 4]);  view_6 = None
    permute_4: "f8e8m0fnu[104, 32, 128, 4][16384, 4, 128, 1]cuda:0" = torch.ops.aten.permute.default(view_11, [0, 2, 1, 3]);  view_11 = None
    clone_2: "f8e8m0fnu[104, 32, 128, 4][16384, 512, 4, 1]cuda:0" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
    view_12: "f8e8m0fnu[3328, 4, 32, 4][512, 128, 4, 1]cuda:0" = torch.ops.aten.reshape.default(clone_2, [3328, 4, 32, 4]);  clone_2 = None
    permute_5: "f8e8m0fnu[3328, 32, 4, 4][512, 4, 128, 1]cuda:0" = torch.ops.aten.permute.default(view_12, [0, 2, 1, 3]);  view_12 = None
    clone_3: "f8e8m0fnu[3328, 32, 4, 4][512, 16, 4, 1]cuda:0" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
    view_13: "f8e8m0fnu[3328, 32, 16][512, 16, 1]cuda:0" = torch.ops.aten.reshape.default(clone_3, [3328, 32, 16]);  clone_3 = None
    view_14: "f8e8m0fnu[1703936][1]cuda:0" = torch.ops.aten.reshape.default(view_13, [1703936]);  view_13 = None
    _scaled_mm: "bf16[1024, 13312][13312, 1]cuda:0" = torch.ops.aten._scaled_mm.default(view_3, permute, view_10, view_14, None, None, torch.bfloat16);  view_3 = permute = view_10 = view_14 = None
    convert_element_type_4: "f32[1024, 13312][13312, 1]cuda:0" = torch.ops.prims.convert_element_type.default(_scaled_mm, torch.float32);  _scaled_mm = None
    sigmoid: "f32[1024, 13312][13312, 1]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_4)
    mul_1: "f32[1024, 13312][13312, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_4, sigmoid);  convert_element_type_4 = sigmoid = None
    convert_element_type_5: "bf16[1024, 13312][13312, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1, torch.bfloat16);  mul_1 = None
    view_17: "bf16[131072, 32][32, 1]cuda:0" = torch.ops.aten.reshape.default(arg2_1, [-1, 32]);  arg2_1 = None
    abs_2: "bf16[131072, 32][32, 1]cuda:0" = torch.ops.aten.abs.default(view_17)
    amax_1: "bf16[131072][1]cuda:0" = torch.ops.aten.amax.default(abs_2, [1]);  abs_2 = None
    isnan_1: "b8[131072][1]cuda:0" = torch.ops.aten.isnan.default(amax_1)
    scalar_tensor_1: "u8[][]cuda:0" = torch.ops.aten.scalar_tensor.default(255, dtype = torch.uint8, layout = torch.strided, device = device(type='cuda', index=0))
    eq_1: "b8[131072][1]cuda:0" = torch.ops.aten.eq.Scalar(amax_1, 0)
    convert_element_type_6: "bf16[131072][1]cuda:0" = torch.ops.prims.convert_element_type.default(eq_1, torch.bfloat16);  eq_1 = None
    mul_2: "bf16[131072][1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_6, 1.1754943508222875e-38);  convert_element_type_6 = None
    add_2: "bf16[131072][1]cuda:0" = torch.ops.aten.add.Tensor(amax_1, mul_2);  amax_1 = mul_2 = None
    view_18: "i16[131072][1]cuda:0" = torch.ops.aten.view.dtype(add_2, torch.int16);  add_2 = None
    rshift_1: "i16[131072][1]cuda:0" = torch.ops.aten.__rshift__.Scalar(view_18, 7);  view_18 = None
    bitwise_and_1: "i16[131072][1]cuda:0" = torch.ops.aten.bitwise_and.Scalar(rshift_1, 255);  rshift_1 = None
    sub_2: "i16[131072][1]cuda:0" = torch.ops.aten.sub.Tensor(bitwise_and_1, 127);  bitwise_and_1 = None
    sub_3: "i16[131072][1]cuda:0" = torch.ops.aten.sub.Tensor(sub_2, 8);  sub_2 = None
    clamp_min_3: "i16[131072][1]cuda:0" = torch.ops.aten.clamp_min.default(sub_3, -127);  sub_3 = None
    clamp_max_2: "i16[131072][1]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_3, 128);  clamp_min_3 = None
    add_3: "i16[131072][1]cuda:0" = torch.ops.aten.add.Tensor(clamp_max_2, 127);  clamp_max_2 = None
    convert_element_type_7: "u8[131072][1]cuda:0" = torch.ops.prims.convert_element_type.default(add_3, torch.uint8);  add_3 = None
    where_1: "u8[131072][1]cuda:0" = torch.ops.aten.where.self(isnan_1, scalar_tensor_1, convert_element_type_7);  isnan_1 = scalar_tensor_1 = convert_element_type_7 = None
    convert_element_type_8: "i32[131072][1]cuda:0" = torch.ops.prims.convert_element_type.default(where_1, torch.int32)
    lshift_1: "i32[131072][1]cuda:0" = torch.ops.aten.__lshift__.Scalar(convert_element_type_8, 23);  convert_element_type_8 = None
    view_19: "f32[131072][1]cuda:0" = torch.ops.aten.view.dtype(lshift_1, torch.float32);  lshift_1 = None
    clamp_min_4: "f32[131072][1]cuda:0" = torch.ops.aten.clamp_min.default(view_19, 1.1754943508222875e-38);  view_19 = None
    unsqueeze_1: "f32[131072, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clamp_min_4, 1);  clamp_min_4 = None
    div_1: "f32[131072, 32][32, 1]cuda:0" = torch.ops.aten.div.Tensor(view_17, unsqueeze_1);  view_17 = unsqueeze_1 = None
    clamp_min_5: "f32[131072, 32][32, 1]cuda:0" = torch.ops.aten.clamp_min.default(div_1, -448.0);  div_1 = None
    clamp_max_3: "f32[131072, 32][32, 1]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_5, 448.0);  clamp_min_5 = None
    convert_element_type_9: "f8e4m3fn[131072, 32][32, 1]cuda:0" = torch.ops.prims.convert_element_type.default(clamp_max_3, torch.float8_e4m3fn);  clamp_max_3 = None
    view_20: "f8e4m3fn[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_9, [1024, 4096]);  convert_element_type_9 = None
    permute_6: "f8e4m3fn[4096, 13312][1, 4096]cuda:0" = torch.ops.aten.permute.default(arg4_1, [1, 0]);  arg4_1 = None
    view_21: "f8e8m0fnu[131072][1]cuda:0" = torch.ops.aten.view.dtype(where_1, torch.float8_e8m0fnu);  where_1 = None
    view_22: "f8e8m0fnu[1024, 128][128, 1]cuda:0" = torch.ops.aten.reshape.default(view_21, [1024, 128]);  view_21 = None
    view_24: "f8e8m0fnu[8, 128, 32, 4][16384, 128, 4, 1]cuda:0" = torch.ops.aten.reshape.default(view_22, [8, 128, 32, 4]);  view_22 = None
    permute_8: "f8e8m0fnu[8, 32, 128, 4][16384, 4, 128, 1]cuda:0" = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3]);  view_24 = None
    clone_4: "f8e8m0fnu[8, 32, 128, 4][16384, 512, 4, 1]cuda:0" = torch.ops.aten.clone.default(permute_8, memory_format = torch.contiguous_format);  permute_8 = None
    view_25: "f8e8m0fnu[256, 4, 32, 4][512, 128, 4, 1]cuda:0" = torch.ops.aten.reshape.default(clone_4, [256, 4, 32, 4]);  clone_4 = None
    permute_9: "f8e8m0fnu[256, 32, 4, 4][512, 4, 128, 1]cuda:0" = torch.ops.aten.permute.default(view_25, [0, 2, 1, 3]);  view_25 = None
    clone_5: "f8e8m0fnu[256, 32, 4, 4][512, 16, 4, 1]cuda:0" = torch.ops.aten.clone.default(permute_9, memory_format = torch.contiguous_format);  permute_9 = None
    view_26: "f8e8m0fnu[256, 32, 16][512, 16, 1]cuda:0" = torch.ops.aten.reshape.default(clone_5, [256, 32, 16]);  clone_5 = None
    view_27: "f8e8m0fnu[131072][1]cuda:0" = torch.ops.aten.reshape.default(view_26, [131072]);  view_26 = None
    view_23: "f8e8m0fnu[13312, 128][128, 1]cuda:0" = torch.ops.aten.reshape.default(arg3_1, [13312, 128]);  arg3_1 = None
    view_28: "f8e8m0fnu[104, 128, 32, 4][16384, 128, 4, 1]cuda:0" = torch.ops.aten.reshape.default(view_23, [104, 128, 32, 4]);  view_23 = None
    permute_10: "f8e8m0fnu[104, 32, 128, 4][16384, 4, 128, 1]cuda:0" = torch.ops.aten.permute.default(view_28, [0, 2, 1, 3]);  view_28 = None
    clone_6: "f8e8m0fnu[104, 32, 128, 4][16384, 512, 4, 1]cuda:0" = torch.ops.aten.clone.default(permute_10, memory_format = torch.contiguous_format);  permute_10 = None
    view_29: "f8e8m0fnu[3328, 4, 32, 4][512, 128, 4, 1]cuda:0" = torch.ops.aten.reshape.default(clone_6, [3328, 4, 32, 4]);  clone_6 = None
    permute_11: "f8e8m0fnu[3328, 32, 4, 4][512, 4, 128, 1]cuda:0" = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None
    clone_7: "f8e8m0fnu[3328, 32, 4, 4][512, 16, 4, 1]cuda:0" = torch.ops.aten.clone.default(permute_11, memory_format = torch.contiguous_format);  permute_11 = None
    view_30: "f8e8m0fnu[3328, 32, 16][512, 16, 1]cuda:0" = torch.ops.aten.reshape.default(clone_7, [3328, 32, 16]);  clone_7 = None
    view_31: "f8e8m0fnu[1703936][1]cuda:0" = torch.ops.aten.reshape.default(view_30, [1703936]);  view_30 = None
    _scaled_mm_1: "bf16[1024, 13312][13312, 1]cuda:0" = torch.ops.aten._scaled_mm.default(view_20, permute_6, view_27, view_31, None, None, torch.bfloat16);  view_20 = permute_6 = view_27 = view_31 = None
    mul_3: "bf16[1024, 13312][13312, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_5, _scaled_mm_1);  convert_element_type_5 = _scaled_mm_1 = None
    view_34: "bf16[425984, 32][32, 1]cuda:0" = torch.ops.aten.reshape.default(mul_3, [-1, 32]);  mul_3 = None
    abs_3: "bf16[425984, 32][32, 1]cuda:0" = torch.ops.aten.abs.default(view_34)
    amax_2: "bf16[425984][1]cuda:0" = torch.ops.aten.amax.default(abs_3, [1]);  abs_3 = None
    isnan_2: "b8[425984][1]cuda:0" = torch.ops.aten.isnan.default(amax_2)
    scalar_tensor_2: "u8[][]cuda:0" = torch.ops.aten.scalar_tensor.default(255, dtype = torch.uint8, layout = torch.strided, device = device(type='cuda', index=0))
    eq_2: "b8[425984][1]cuda:0" = torch.ops.aten.eq.Scalar(amax_2, 0)
    convert_element_type_10: "bf16[425984][1]cuda:0" = torch.ops.prims.convert_element_type.default(eq_2, torch.bfloat16);  eq_2 = None
    mul_4: "bf16[425984][1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_10, 1.1754943508222875e-38);  convert_element_type_10 = None
    add_4: "bf16[425984][1]cuda:0" = torch.ops.aten.add.Tensor(amax_2, mul_4);  amax_2 = mul_4 = None
    view_35: "i16[425984][1]cuda:0" = torch.ops.aten.view.dtype(add_4, torch.int16);  add_4 = None
    rshift_2: "i16[425984][1]cuda:0" = torch.ops.aten.__rshift__.Scalar(view_35, 7);  view_35 = None
    bitwise_and_2: "i16[425984][1]cuda:0" = torch.ops.aten.bitwise_and.Scalar(rshift_2, 255);  rshift_2 = None
    sub_4: "i16[425984][1]cuda:0" = torch.ops.aten.sub.Tensor(bitwise_and_2, 127);  bitwise_and_2 = None
    sub_5: "i16[425984][1]cuda:0" = torch.ops.aten.sub.Tensor(sub_4, 8);  sub_4 = None
    clamp_min_6: "i16[425984][1]cuda:0" = torch.ops.aten.clamp_min.default(sub_5, -127);  sub_5 = None
    clamp_max_4: "i16[425984][1]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_6, 128);  clamp_min_6 = None
    add_5: "i16[425984][1]cuda:0" = torch.ops.aten.add.Tensor(clamp_max_4, 127);  clamp_max_4 = None
    convert_element_type_11: "u8[425984][1]cuda:0" = torch.ops.prims.convert_element_type.default(add_5, torch.uint8);  add_5 = None
    where_2: "u8[425984][1]cuda:0" = torch.ops.aten.where.self(isnan_2, scalar_tensor_2, convert_element_type_11);  isnan_2 = scalar_tensor_2 = convert_element_type_11 = None
    convert_element_type_12: "i32[425984][1]cuda:0" = torch.ops.prims.convert_element_type.default(where_2, torch.int32)
    lshift_2: "i32[425984][1]cuda:0" = torch.ops.aten.__lshift__.Scalar(convert_element_type_12, 23);  convert_element_type_12 = None
    view_36: "f32[425984][1]cuda:0" = torch.ops.aten.view.dtype(lshift_2, torch.float32);  lshift_2 = None
    clamp_min_7: "f32[425984][1]cuda:0" = torch.ops.aten.clamp_min.default(view_36, 1.1754943508222875e-38);  view_36 = None
    unsqueeze_2: "f32[425984, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clamp_min_7, 1);  clamp_min_7 = None
    div_2: "f32[425984, 32][32, 1]cuda:0" = torch.ops.aten.div.Tensor(view_34, unsqueeze_2);  view_34 = unsqueeze_2 = None
    clamp_min_8: "f32[425984, 32][32, 1]cuda:0" = torch.ops.aten.clamp_min.default(div_2, -448.0);  div_2 = None
    clamp_max_5: "f32[425984, 32][32, 1]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_8, 448.0);  clamp_min_8 = None
    convert_element_type_13: "f8e4m3fn[425984, 32][32, 1]cuda:0" = torch.ops.prims.convert_element_type.default(clamp_max_5, torch.float8_e4m3fn);  clamp_max_5 = None
    view_37: "f8e4m3fn[1024, 13312][13312, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_13, [1024, 13312]);  convert_element_type_13 = None
    permute_12: "f8e4m3fn[13312, 4096][1, 13312]cuda:0" = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
    view_38: "f8e8m0fnu[425984][1]cuda:0" = torch.ops.aten.view.dtype(where_2, torch.float8_e8m0fnu);  where_2 = None
    view_39: "f8e8m0fnu[1024, 416][416, 1]cuda:0" = torch.ops.aten.reshape.default(view_38, [1024, 416]);  view_38 = None
    view_41: "f8e8m0fnu[8, 128, 104, 4][53248, 416, 4, 1]cuda:0" = torch.ops.aten.reshape.default(view_39, [8, 128, 104, 4]);  view_39 = None
    permute_14: "f8e8m0fnu[8, 104, 128, 4][53248, 4, 416, 1]cuda:0" = torch.ops.aten.permute.default(view_41, [0, 2, 1, 3]);  view_41 = None
    clone_8: "f8e8m0fnu[8, 104, 128, 4][53248, 512, 4, 1]cuda:0" = torch.ops.aten.clone.default(permute_14, memory_format = torch.contiguous_format);  permute_14 = None
    view_42: "f8e8m0fnu[832, 4, 32, 4][512, 128, 4, 1]cuda:0" = torch.ops.aten.reshape.default(clone_8, [832, 4, 32, 4]);  clone_8 = None
    permute_15: "f8e8m0fnu[832, 32, 4, 4][512, 4, 128, 1]cuda:0" = torch.ops.aten.permute.default(view_42, [0, 2, 1, 3]);  view_42 = None
    clone_9: "f8e8m0fnu[832, 32, 4, 4][512, 16, 4, 1]cuda:0" = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None
    view_43: "f8e8m0fnu[832, 32, 16][512, 16, 1]cuda:0" = torch.ops.aten.reshape.default(clone_9, [832, 32, 16]);  clone_9 = None
    view_44: "f8e8m0fnu[425984][1]cuda:0" = torch.ops.aten.reshape.default(view_43, [425984]);  view_43 = None
    view_40: "f8e8m0fnu[4096, 416][416, 1]cuda:0" = torch.ops.aten.reshape.default(arg5_1, [4096, 416]);  arg5_1 = None
    view_45: "f8e8m0fnu[32, 128, 104, 4][53248, 416, 4, 1]cuda:0" = torch.ops.aten.reshape.default(view_40, [32, 128, 104, 4]);  view_40 = None
    permute_16: "f8e8m0fnu[32, 104, 128, 4][53248, 4, 416, 1]cuda:0" = torch.ops.aten.permute.default(view_45, [0, 2, 1, 3]);  view_45 = None
    clone_10: "f8e8m0fnu[32, 104, 128, 4][53248, 512, 4, 1]cuda:0" = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format);  permute_16 = None
    view_46: "f8e8m0fnu[3328, 4, 32, 4][512, 128, 4, 1]cuda:0" = torch.ops.aten.reshape.default(clone_10, [3328, 4, 32, 4]);  clone_10 = None
    permute_17: "f8e8m0fnu[3328, 32, 4, 4][512, 4, 128, 1]cuda:0" = torch.ops.aten.permute.default(view_46, [0, 2, 1, 3]);  view_46 = None
    clone_11: "f8e8m0fnu[3328, 32, 4, 4][512, 16, 4, 1]cuda:0" = torch.ops.aten.clone.default(permute_17, memory_format = torch.contiguous_format);  permute_17 = None
    view_47: "f8e8m0fnu[3328, 32, 16][512, 16, 1]cuda:0" = torch.ops.aten.reshape.default(clone_11, [3328, 32, 16]);  clone_11 = None
    view_48: "f8e8m0fnu[1703936][1]cuda:0" = torch.ops.aten.reshape.default(view_47, [1703936]);  view_47 = None
    _scaled_mm_2: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten._scaled_mm.default(view_37, permute_12, view_44, view_48, None, None, torch.bfloat16);  view_37 = permute_12 = view_44 = view_48 = None
    return (_scaled_mm_2,)


from torch._dynamo.debug_utils import aot_graph_input_parser

torch.compile(forward)(**aot_graph_input_parser(forward))