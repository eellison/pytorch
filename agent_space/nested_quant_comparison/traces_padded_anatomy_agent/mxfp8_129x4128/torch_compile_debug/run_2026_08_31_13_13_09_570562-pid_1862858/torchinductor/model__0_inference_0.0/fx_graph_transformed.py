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

        # File: /data/users/eellison/pytorch/agent_space/nested_quant_comparison/bench_padded_quant_main.py:130 in rmsnorm_mxfp8_padded, code: groups = normed.view(rows, hidden // 32, 32)
        view: "bf16[129, 129, 32]" = torch.ops.aten.reshape.default(convert_element_type_1, [129, 129, 32]);  convert_element_type_1 = None

        # File: /data/users/eellison/pytorch/agent_space/nested_quant_comparison/bench_padded_quant_main.py:136 in rmsnorm_mxfp8_padded, code: (groups.float() / scale_f32.unsqueeze(-1))
        convert_element_type_4: "f32[129, 129, 32]" = torch.ops.prims.convert_element_type.default(view, torch.float32)

        # File: /data/users/eellison/pytorch/agent_space/nested_quant_comparison/bench_padded_quant_main.py:134 in rmsnorm_mxfp8_padded, code: scale_f32 = torch.ldexp(torch.ones_like(raw_scale), scale.to(torch.int32) - 127)
        full_default: "f32[129, 129]" = torch.ops.aten.full.default([129, 129], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /data/users/eellison/pytorch/agent_space/nested_quant_comparison/bench_padded_quant_main.py:131 in rmsnorm_mxfp8_padded, code: amax = groups.abs().float().amax(dim=-1)
        abs_1: "bf16[129, 129, 32]" = torch.ops.aten.abs.default(view);  view = None
        convert_element_type_2: "f32[129, 129, 32]" = torch.ops.prims.convert_element_type.default(abs_1, torch.float32);  abs_1 = None
        amax: "f32[129, 129]" = torch.ops.aten.amax.default(convert_element_type_2, [-1]);  convert_element_type_2 = None

        # File: /data/users/eellison/pytorch/agent_space/nested_quant_comparison/bench_padded_quant_main.py:132 in rmsnorm_mxfp8_padded, code: raw_scale = (amax / FP8_MAX).clamp_min(torch.finfo(torch.float32).tiny)
        div: "f32[129, 129]" = torch.ops.aten.div.Tensor(amax, 448.0);  amax = None
        clamp_min: "f32[129, 129]" = torch.ops.aten.clamp_min.default(div, 1.1754943508222875e-38);  div = None

        # File: /data/users/eellison/pytorch/agent_space/nested_quant_comparison/bench_padded_quant_main.py:133 in rmsnorm_mxfp8_padded, code: scale = inductor_prims.cvt_e8m0_rceil(raw_scale)
        inductor_cvt_e8m0_rceil: "u8[129, 129]" = torch.ops.prims.inductor_cvt_e8m0_rceil.default(clamp_min);  clamp_min = None

        # File: /data/users/eellison/pytorch/agent_space/nested_quant_comparison/bench_padded_quant_main.py:134 in rmsnorm_mxfp8_padded, code: scale_f32 = torch.ldexp(torch.ones_like(raw_scale), scale.to(torch.int32) - 127)
        convert_element_type_3: "i32[129, 129]" = torch.ops.prims.convert_element_type.default(inductor_cvt_e8m0_rceil, torch.int32)
        sub: "i32[129, 129]" = torch.ops.aten.sub.Tensor(convert_element_type_3, 127);  convert_element_type_3 = None
        ldexp: "f32[129, 129]" = torch.ops.aten.ldexp.Tensor(full_default, sub);  full_default = sub = None

        # File: /data/users/eellison/pytorch/agent_space/nested_quant_comparison/bench_padded_quant_main.py:136 in rmsnorm_mxfp8_padded, code: (groups.float() / scale_f32.unsqueeze(-1))
        unsqueeze: "f32[129, 129, 1]" = torch.ops.aten.unsqueeze.default(ldexp, -1);  ldexp = None
        div_1: "f32[129, 129, 32]" = torch.ops.aten.div.Tensor(convert_element_type_4, unsqueeze);  convert_element_type_4 = unsqueeze = None

        # File: /data/users/eellison/pytorch/agent_space/nested_quant_comparison/bench_padded_quant_main.py:137 in rmsnorm_mxfp8_padded, code: .clamp(-FP8_MAX, FP8_MAX)
        clamp_min_1: "f32[129, 129, 32]" = torch.ops.aten.clamp_min.default(div_1, -448.0);  div_1 = None
        clamp_max: "f32[129, 129, 32]" = torch.ops.aten.clamp_max.default(clamp_min_1, 448.0);  clamp_min_1 = None

        # File: /data/users/eellison/pytorch/agent_space/nested_quant_comparison/bench_padded_quant_main.py:138 in rmsnorm_mxfp8_padded, code: .to(torch.float8_e4m3fn)
        convert_element_type_5: "f8e4m3fn[129, 129, 32]" = torch.ops.prims.convert_element_type.default(clamp_max, torch.float8_e4m3fn);  clamp_max = None

        # File: /data/users/eellison/pytorch/agent_space/nested_quant_comparison/bench_padded_quant_main.py:139 in rmsnorm_mxfp8_padded, code: .view(rows, hidden)
        view_1: "f8e4m3fn[129, 4128]" = torch.ops.aten.reshape.default(convert_element_type_5, [129, 4128]);  convert_element_type_5 = None

        # File: /data/users/eellison/pytorch/agent_space/nested_quant_comparison/pytorch_main/torch/nn/functional.py:5861 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd: "u8[256, 132]" = torch.ops.aten.constant_pad_nd.default(inductor_cvt_e8m0_rceil, [0, 3, 0, 127], 0.0);  inductor_cvt_e8m0_rceil = None

        # File: /data/users/eellison/pytorch/agent_space/nested_quant_comparison/bench_padded_quant_main.py:33 in to_blocked_padded, code: blocks = scale.view(padded_rows // 128, 128, padded_cols // 4, 4)
        view_2: "u8[2, 128, 33, 4]" = torch.ops.aten.reshape.default(constant_pad_nd, [2, 128, 33, 4]);  constant_pad_nd = None

        # File: /data/users/eellison/pytorch/agent_space/nested_quant_comparison/bench_padded_quant_main.py:34 in to_blocked_padded, code: return blocks.permute(0, 2, 1, 3).reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1)
        permute: "u8[2, 33, 128, 4]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        clone: "u8[2, 33, 128, 4]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_3: "u8[66, 4, 32, 4]" = torch.ops.aten.reshape.default(clone, [66, 4, 32, 4]);  clone = None
        permute_1: "u8[66, 32, 4, 4]" = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None
        clone_1: "u8[66, 32, 4, 4]" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
        view_4: "u8[33792]" = torch.ops.aten.reshape.default(clone_1, [33792]);  clone_1 = None
        return (view_1, view_4)
