class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "bf16[128, 4096]", arg1_1: "bf16[4096]"):
        # File: /data/users/eellison/pytorch/agent_space/nested_quant_comparison/pytorch_main/torch/nn/functional.py:3012 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type: "f32[128, 4096]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        pow_1: "f32[128, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type, 2)
        mean: "f32[128, 1]" = torch.ops.aten.mean.dim(pow_1, [1], True);  pow_1 = None
        add: "f32[128, 1]" = torch.ops.aten.add.Scalar(mean, 1.1920928955078125e-07);  mean = None
        rsqrt: "f32[128, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        mul: "f32[128, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type, rsqrt);  convert_element_type = rsqrt = None
        mul_1: "f32[128, 4096]" = torch.ops.aten.mul.Tensor(mul, arg1_1);  mul = arg1_1 = None
        convert_element_type_1: "bf16[128, 4096]" = torch.ops.prims.convert_element_type.default(mul_1, torch.bfloat16);  mul_1 = None

        # File: /data/users/eellison/pytorch/agent_space/nested_quant_comparison/bench_padded_quant_main.py:104 in rmsnorm_fp4_padded, code: normed = F.rms_norm(x, (hidden,), weight).view(rows, hidden // block, block)
        view: "bf16[128, 256, 16]" = torch.ops.aten.reshape.default(convert_element_type_1, [128, 256, 16]);  convert_element_type_1 = None

        # File: /data/users/eellison/pytorch/agent_space/nested_quant_comparison/bench_padded_quant_main.py:112 in rmsnorm_fp4_padded, code: pairs = normed.view(rows, hidden // block, block // 2, 2)
        view_1: "bf16[128, 256, 8, 2]" = torch.ops.aten.reshape.default(view, [128, 256, 8, 2])

        # File: /data/users/eellison/pytorch/agent_space/nested_quant_comparison/bench_padded_quant_main.py:114 in rmsnorm_fp4_padded, code: pairs[..., 0].float() * inv_scale.unsqueeze(-1),
        select: "bf16[128, 256, 8]" = torch.ops.aten.select.int(view_1, 3, 0)
        convert_element_type_6: "f32[128, 256, 8]" = torch.ops.prims.convert_element_type.default(select, torch.float32);  select = None

        # File: /data/users/eellison/pytorch/agent_space/nested_quant_comparison/bench_padded_quant_main.py:105 in rmsnorm_fp4_padded, code: amax = normed.abs().amax(dim=-1)
        abs_1: "bf16[128, 256, 16]" = torch.ops.aten.abs.default(view);  view = None
        amax: "bf16[128, 256]" = torch.ops.aten.amax.default(abs_1, [-1]);  abs_1 = None

        # File: /data/users/eellison/pytorch/agent_space/nested_quant_comparison/bench_padded_quant_main.py:107 in rmsnorm_fp4_padded, code: scale = (amax / FP4_MAX).clamp(min=1e-12, max=FP8_MAX).to(torch.float8_e4m3fn)
        div: "bf16[128, 256]" = torch.ops.aten.div.Tensor(amax, 6.0);  amax = None
        convert_element_type_2: "f32[128, 256]" = torch.ops.prims.convert_element_type.default(div, torch.float32);  div = None
        clamp_min: "f32[128, 256]" = torch.ops.aten.clamp_min.default(convert_element_type_2, 1e-12);  convert_element_type_2 = None
        clamp_max: "f32[128, 256]" = torch.ops.aten.clamp_max.default(clamp_min, 448.0);  clamp_min = None
        convert_element_type_3: "bf16[128, 256]" = torch.ops.prims.convert_element_type.default(clamp_max, torch.bfloat16);  clamp_max = None
        convert_element_type_4: "f8e4m3fn[128, 256]" = torch.ops.prims.convert_element_type.default(convert_element_type_3, torch.float8_e4m3fn);  convert_element_type_3 = None

        # File: /data/users/eellison/pytorch/agent_space/nested_quant_comparison/bench_padded_quant_main.py:108 in rmsnorm_fp4_padded, code: inv_scale = scale.float().reciprocal()
        convert_element_type_5: "f32[128, 256]" = torch.ops.prims.convert_element_type.default(convert_element_type_4, torch.float32)
        reciprocal: "f32[128, 256]" = torch.ops.aten.reciprocal.default(convert_element_type_5);  convert_element_type_5 = None

        # File: /data/users/eellison/pytorch/agent_space/nested_quant_comparison/bench_padded_quant_main.py:114 in rmsnorm_fp4_padded, code: pairs[..., 0].float() * inv_scale.unsqueeze(-1),
        unsqueeze: "f32[128, 256, 1]" = torch.ops.aten.unsqueeze.default(reciprocal, -1)
        mul_2: "f32[128, 256, 8]" = torch.ops.aten.mul.Tensor(convert_element_type_6, unsqueeze);  convert_element_type_6 = unsqueeze = None

        # File: /data/users/eellison/pytorch/agent_space/nested_quant_comparison/bench_padded_quant_main.py:115 in rmsnorm_fp4_padded, code: pairs[..., 1].float() * inv_scale.unsqueeze(-1),
        select_1: "bf16[128, 256, 8]" = torch.ops.aten.select.int(view_1, 3, 1);  view_1 = None
        convert_element_type_7: "f32[128, 256, 8]" = torch.ops.prims.convert_element_type.default(select_1, torch.float32);  select_1 = None
        unsqueeze_1: "f32[128, 256, 1]" = torch.ops.aten.unsqueeze.default(reciprocal, -1);  reciprocal = None
        mul_3: "f32[128, 256, 8]" = torch.ops.aten.mul.Tensor(convert_element_type_7, unsqueeze_1);  convert_element_type_7 = unsqueeze_1 = None

        # File: /data/users/eellison/pytorch/agent_space/nested_quant_comparison/bench_padded_quant_main.py:113 in rmsnorm_fp4_padded, code: packed = inline_asm_elementwise(
        inline_asm_elementwise: "i32[128, 256, 8]" = torch.ops.higher_order.inline_asm_elementwise(mul_2, mul_3, asm_str = '{.reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, $2, $1; cvt.u32.u8 $0, t;}', constraints = '=r,f,f', dtype = torch.int32, is_pure = True, pack = 1);  mul_2 = mul_3 = None

        # File: /data/users/eellison/pytorch/agent_space/nested_quant_comparison/bench_padded_quant_main.py:121 in rmsnorm_fp4_padded, code: ).to(torch.uint8).view(rows, hidden // 2)
        convert_element_type_8: "u8[128, 256, 8]" = torch.ops.prims.convert_element_type.default(inline_asm_elementwise, torch.uint8);  inline_asm_elementwise = None
        view_2: "u8[128, 2048]" = torch.ops.aten.reshape.default(convert_element_type_8, [128, 2048]);  convert_element_type_8 = None

        # File: /data/users/eellison/pytorch/agent_space/nested_quant_comparison/bench_padded_quant_main.py:33 in to_blocked_padded, code: blocks = scale.view(padded_rows // 128, 128, padded_cols // 4, 4)
        view_3: "f8e4m3fn[1, 128, 64, 4]" = torch.ops.aten.reshape.default(convert_element_type_4, [1, 128, 64, 4]);  convert_element_type_4 = None

        # File: /data/users/eellison/pytorch/agent_space/nested_quant_comparison/bench_padded_quant_main.py:34 in to_blocked_padded, code: return blocks.permute(0, 2, 1, 3).reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1)
        permute: "f8e4m3fn[1, 64, 128, 4]" = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None
        view_4: "f8e4m3fn[64, 4, 32, 4]" = torch.ops.aten.reshape.default(permute, [64, 4, 32, 4]);  permute = None
        permute_1: "f8e4m3fn[64, 32, 4, 4]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
        clone: "f8e4m3fn[64, 32, 4, 4]" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
        view_5: "f8e4m3fn[32768]" = torch.ops.aten.reshape.default(clone, [32768]);  clone = None
        return (view_2, view_5)
