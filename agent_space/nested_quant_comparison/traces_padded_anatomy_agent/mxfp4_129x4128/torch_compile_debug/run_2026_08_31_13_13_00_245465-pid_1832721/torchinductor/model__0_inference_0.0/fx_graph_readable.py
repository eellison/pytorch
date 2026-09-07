class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "bf16[129, 4128]", arg1_1: "bf16[4128]"):
        # File: /data/users/eellison/pytorch/agent_space/nested_quant_comparison/pytorch_main/torch/nn/functional.py:3012 in rms_norm, code: return torch.rms_norm(input, normalized_shape, weight, eps)
        convert_element_type: "f32[129, 4128]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        pow_1: "f32[129, 4128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type, 2)
        mean: "f32[129, 1]" = torch.ops.aten.mean.dim(pow_1, [1], True);  pow_1 = None
        add: "f32[129, 1]" = torch.ops.aten.add.Scalar(mean, 1.1920928955078125e-07);  mean = None
        rsqrt: "f32[129, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        mul: "f32[129, 4128]" = torch.ops.aten.mul.Tensor(convert_element_type, rsqrt);  convert_element_type = rsqrt = None
        mul_1: "f32[129, 4128]" = torch.ops.aten.mul.Tensor(mul, arg1_1);  mul = arg1_1 = None
        convert_element_type_1: "bf16[129, 4128]" = torch.ops.prims.convert_element_type.default(mul_1, torch.bfloat16);  mul_1 = None

        # File: /data/users/eellison/pytorch/agent_space/nested_quant_comparison/bench_padded_quant_main.py:104 in rmsnorm_fp4_padded, code: normed = F.rms_norm(x, (hidden,), weight).view(rows, hidden // block, block)
        view: "bf16[129, 129, 32]" = torch.ops.aten.view.default(convert_element_type_1, [129, 129, 32]);  convert_element_type_1 = None

        # File: /data/users/eellison/pytorch/agent_space/nested_quant_comparison/bench_padded_quant_main.py:105 in rmsnorm_fp4_padded, code: amax = normed.abs().amax(dim=-1)
        abs_1: "bf16[129, 129, 32]" = torch.ops.aten.abs.default(view)
        amax: "bf16[129, 129]" = torch.ops.aten.amax.default(abs_1, [-1]);  abs_1 = None

        # File: /data/users/eellison/pytorch/agent_space/nested_quant_comparison/bench_padded_quant_main.py:110 in rmsnorm_fp4_padded, code: scale = inductor_prims.cvt_e8m0_rceil((amax / FP4_MAX).clamp_min(1e-12))
        div: "bf16[129, 129]" = torch.ops.aten.div.Tensor(amax, 6.0);  amax = None
        clamp_min: "bf16[129, 129]" = torch.ops.aten.clamp_min.default(div, 1e-12);  div = None
        inductor_cvt_e8m0_rceil: "u8[129, 129]" = torch.ops.prims.inductor_cvt_e8m0_rceil.default(clamp_min);  clamp_min = None

        # File: /data/users/eellison/pytorch/agent_space/nested_quant_comparison/bench_main_cuda.py:301 in recip_ue8m0, code: return inline_asm_elementwise(scale.to(torch.int32), asm_str=RECIP_UE8M0_ASM, constraints="=f,r", dtype=torch.float32, is_pure=True, pack=1)
        convert_element_type_2: "i32[129, 129]" = torch.ops.prims.convert_element_type.default(inductor_cvt_e8m0_rceil, torch.int32)
        inline_asm_elementwise: "f32[129, 129]" = torch.ops.higher_order.inline_asm_elementwise(convert_element_type_2, asm_str = '{.reg .pred p_zero; .reg .s32 neg_exp; .reg .f32 neg_exp_f, result; setp.eq.u32 p_zero, $1, 0; sub.s32 neg_exp, 127, $1; cvt.rn.f32.s32 neg_exp_f, neg_exp; ex2.approx.f32 result, neg_exp_f; selp.f32 $0, 0f00000000, result, p_zero;}', constraints = '=f,r', dtype = torch.float32, is_pure = True, pack = 1);  convert_element_type_2 = None

        # File: /data/users/eellison/pytorch/agent_space/nested_quant_comparison/bench_padded_quant_main.py:112 in rmsnorm_fp4_padded, code: pairs = normed.view(rows, hidden // block, block // 2, 2)
        view_1: "bf16[129, 129, 16, 2]" = torch.ops.aten.view.default(view, [129, 129, 16, 2]);  view = None

        # File: /data/users/eellison/pytorch/agent_space/nested_quant_comparison/bench_padded_quant_main.py:114 in rmsnorm_fp4_padded, code: pairs[..., 0].float() * inv_scale.unsqueeze(-1),
        select: "bf16[129, 129, 16]" = torch.ops.aten.select.int(view_1, 3, 0)
        convert_element_type_3: "f32[129, 129, 16]" = torch.ops.prims.convert_element_type.default(select, torch.float32);  select = None
        unsqueeze: "f32[129, 129, 1]" = torch.ops.aten.unsqueeze.default(inline_asm_elementwise, -1)
        mul_2: "f32[129, 129, 16]" = torch.ops.aten.mul.Tensor(convert_element_type_3, unsqueeze);  convert_element_type_3 = unsqueeze = None

        # File: /data/users/eellison/pytorch/agent_space/nested_quant_comparison/bench_padded_quant_main.py:115 in rmsnorm_fp4_padded, code: pairs[..., 1].float() * inv_scale.unsqueeze(-1),
        select_1: "bf16[129, 129, 16]" = torch.ops.aten.select.int(view_1, 3, 1);  view_1 = None
        convert_element_type_4: "f32[129, 129, 16]" = torch.ops.prims.convert_element_type.default(select_1, torch.float32);  select_1 = None
        unsqueeze_1: "f32[129, 129, 1]" = torch.ops.aten.unsqueeze.default(inline_asm_elementwise, -1);  inline_asm_elementwise = None
        mul_3: "f32[129, 129, 16]" = torch.ops.aten.mul.Tensor(convert_element_type_4, unsqueeze_1);  convert_element_type_4 = unsqueeze_1 = None

        # File: /data/users/eellison/pytorch/agent_space/nested_quant_comparison/bench_padded_quant_main.py:113 in rmsnorm_fp4_padded, code: packed = inline_asm_elementwise(
        inline_asm_elementwise_1: "i32[129, 129, 16]" = torch.ops.higher_order.inline_asm_elementwise(mul_2, mul_3, asm_str = '{.reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, $2, $1; cvt.u32.u8 $0, t;}', constraints = '=r,f,f', dtype = torch.int32, is_pure = True, pack = 1);  mul_2 = mul_3 = None

        # File: /data/users/eellison/pytorch/agent_space/nested_quant_comparison/bench_padded_quant_main.py:121 in rmsnorm_fp4_padded, code: ).to(torch.uint8).view(rows, hidden // 2)
        convert_element_type_5: "u8[129, 129, 16]" = torch.ops.prims.convert_element_type.default(inline_asm_elementwise_1, torch.uint8);  inline_asm_elementwise_1 = None
        view_2: "u8[129, 2064]" = torch.ops.aten.view.default(convert_element_type_5, [129, 2064]);  convert_element_type_5 = None

        # File: /data/users/eellison/pytorch/agent_space/nested_quant_comparison/pytorch_main/torch/nn/functional.py:5861 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd: "u8[256, 132]" = torch.ops.aten.constant_pad_nd.default(inductor_cvt_e8m0_rceil, [0, 3, 0, 127], 0.0);  inductor_cvt_e8m0_rceil = None

        # File: /data/users/eellison/pytorch/agent_space/nested_quant_comparison/bench_padded_quant_main.py:33 in to_blocked_padded, code: blocks = scale.view(padded_rows // 128, 128, padded_cols // 4, 4)
        view_3: "u8[2, 128, 33, 4]" = torch.ops.aten.view.default(constant_pad_nd, [2, 128, 33, 4]);  constant_pad_nd = None

        # File: /data/users/eellison/pytorch/agent_space/nested_quant_comparison/bench_padded_quant_main.py:34 in to_blocked_padded, code: return blocks.permute(0, 2, 1, 3).reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1)
        permute: "u8[2, 33, 128, 4]" = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None
        clone: "u8[2, 33, 128, 4]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_4: "u8[66, 4, 32, 4]" = torch.ops.aten.view.default(clone, [66, 4, 32, 4]);  clone = None
        permute_1: "u8[66, 32, 4, 4]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
        clone_1: "u8[66, 32, 4, 4]" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
        view_5: "u8[33792]" = torch.ops.aten.view.default(clone_1, [33792]);  clone_1 = None
        return (view_2, view_5)
