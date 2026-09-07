import sys, torch, importlib.util, os
which = sys.argv[1]
spec = importlib.util.spec_from_file_location("hk", os.path.abspath("agent_space/peak_cmp/_hk_kernels.py")); hk = importlib.util.module_from_spec(spec); spec.loader.exec_module(hk)
M, K = 8192, 4096
torch.manual_seed(0)
x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16); w = torch.randn(K, device="cuda", dtype=torch.bfloat16)
q = torch.empty(M, K, device="cuda", dtype=torch.float8_e4m3fn); sf = torch.empty(M * K // 32, device="cuda", dtype=torch.uint8)
if which == "fused":
    run = lambda: hk.k_fp32_resident[(M,)](x, w, sf, q, K=K, XBLOCK=1, R0_BLOCK=K, num_warps=2)
else:
    sys.argv = ["x", "quant_mxfp8", "8192", "4096", "ours", "default"]
    exec(open("agent_space/peak_cmp/peak_cell.py").read().split("torch.manual_seed(0)")[0])
    fn = QUANT["mxfp8"]
    with config.patch({"triton.nested_reduction": True, "triton.cudagraphs": False, "fx_graph_cache": False}):
        compiled = torch.compile(fn, fullgraph=True, dynamic=False); compiled(x)
    run = lambda: compiled(x)
for _ in range(3): run()
torch.cuda.synchronize(); torch.cuda.nvtx.range_push("profile"); run(); torch.cuda.nvtx.range_pop(); torch.cuda.synchronize(); print("DONE")
