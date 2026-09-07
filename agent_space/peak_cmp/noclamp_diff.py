import torch, importlib.util, os
spec = importlib.util.spec_from_file_location("hk", os.path.abspath("agent_space/peak_cmp/_hk_kernels.py")); hk = importlib.util.module_from_spec(spec); spec.loader.exec_module(hk)
M, K = 8192, 4096
torch.manual_seed(0)
x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16); w = torch.randn(K, device="cuda", dtype=torch.bfloat16)
x[0, :64] = 3.0e38; x[1, :32] = float("inf"); x[2, :8] = float("nan"); x[3] = 0.0
outs = {}
for name, kern in (("clamp", hk.k_fp32_resident), ("noclamp", hk.k_noclamp)):
    q = torch.empty(M, K, device="cuda", dtype=torch.float8_e4m3fn); sf = torch.empty(M * K // 32, device="cuda", dtype=torch.uint8)
    kern[(M,)](x, w, sf, q, K=K, XBLOCK=1, R0_BLOCK=K, num_warps=4); torch.cuda.synchronize(); outs[name] = q.clone()
a, b = outs["clamp"].view(torch.uint8), outs["noclamp"].view(torch.uint8)
diff = (a != b)
rows = diff.any(dim=1).nonzero().flatten().tolist()
print("rows with differences:", rows[:10], "count per row:", [int(diff[r].sum()) for r in rows[:10]])
for r in rows[:4]:
    cols = diff[r].nonzero().flatten()[:4].tolist()
    print(f"  row {r}: x={x[r, cols].float().tolist()} clamp={outs['clamp'][r, cols].float().tolist()} noclamp={outs['noclamp'][r, cols].float().tolist()} bytes clamp={a[r, cols].tolist()} noclamp={b[r, cols].tolist()}")
# clean input (no specials): exact?
torch.manual_seed(1); xc = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
res = {}
for name, kern in (("clamp", hk.k_fp32_resident), ("noclamp", hk.k_noclamp)):
    q = torch.empty(M, K, device="cuda", dtype=torch.float8_e4m3fn); sf = torch.empty(M * K // 32, device="cuda", dtype=torch.uint8)
    kern[(M,)](xc, w, sf, q, K=K, XBLOCK=1, R0_BLOCK=K, num_warps=4); torch.cuda.synchronize(); res[name] = (q.clone(), sf.clone())
print("clean randn input: q equal =", torch.equal(res["clamp"][0].view(torch.uint8), res["noclamp"][0].view(torch.uint8)), "sf equal =", torch.equal(res["clamp"][1], res["noclamp"][1]))
