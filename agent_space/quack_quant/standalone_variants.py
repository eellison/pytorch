import torch
from torch._inductor import config, metrics
from torch._inductor.utils import fresh_inductor_cache, run_and_get_code
from quack.blockscaled import quantize as qz

M, K, BS = 2048, 3072, 32
x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)

def scale_and_hp(x):
    data_hp = x.reshape(M, K // BS, BS)
    max_abs = torch.amax(torch.abs(data_hp), -1).unsqueeze(-1).to(torch.float32)
    sb = qz._compute_e8m0_scale_floor(max_abs, qz.F4_E2M1_MAX_POW2)
    scale_fp32 = torch.clamp((sb.to(torch.int32) << qz.MBITS_F32).view(torch.float32), min=qz.F32_MIN_NORMAL)
    return data_hp.to(torch.float32) / scale_fp32, sb.view(torch.float8_e8m0fnu).squeeze(-1)

def v1_quack(x):
    return qz.to_mxfp4(x)

def v2_structured_pack(x):
    data_lp, scale = scale_and_hp(x)
    codes = qz._f32_to_floatx_unpacked(data_lp.reshape(M, K), 2, 1)
    c = codes.view(M, K // 2, 2)
    return c[..., 0] | (c[..., 1] << 4), scale

def v3_simple_codes_flat_pack(x):
    data_lp, scale = scale_and_hp(x)
    codes = data_lp.reshape(M, K).clamp(0, 15).to(torch.uint8)
    flat = codes.view(-1)
    return (flat[::2] | flat[1::2] << 4).view(M, K // 2), scale

def v4_simple_codes_structured_pack(x):
    data_lp, scale = scale_and_hp(x)
    codes = data_lp.reshape(M, K).clamp(0, 15).to(torch.uint8)
    c = codes.view(M, K // 2, 2)
    return c[..., 0] | (c[..., 1] << 4), scale

def v5_pairs_from_blocks(x):
    data_lp, scale = scale_and_hp(x)
    c = data_lp.view(M, K // BS, BS // 2, 2).clamp(0, 15).to(torch.uint8)
    return (c[..., 0] | (c[..., 1] << 4)).view(M, K // 2), scale

def v6_quack_no_pack(x):
    data_lp, scale = scale_and_hp(x)
    return qz._f32_to_floatx_unpacked(data_lp.reshape(M, K), 2, 1), scale

for fn in (v1_quack, v2_structured_pack, v3_simple_codes_flat_pack, v4_simple_codes_structured_pack, v5_pairs_from_blocks, v6_quack_no_pack):
    for nested in (True, False):
        torch._dynamo.reset(); metrics.reset()
        with fresh_inductor_cache(), config.patch({"triton.nested_reduction": nested, "triton.cudagraphs": False, "fx_graph_cache": False}):
            out, srcs = run_and_get_code(torch.compile(fn, fullgraph=True, dynamic=False), x)
        names = [l[4:l.index("(")] for s in srcs for l in s.splitlines() if l.startswith("def triton_")]
        print(f"{fn.__name__:32s} nested={'on ' if nested else 'off'} kernels={metrics.generated_kernel_count} nested_count={metrics.codegen_nested_reduction} {[n[:28] for n in names]}")
