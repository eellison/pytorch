import sys, torch, importlib.util, os
src = open("agent_space/gluon_proto/rms_mxfp8_gluon.py").read()
mod_src = src.split("def graph_bench(")[0].replace('print("gl has:"', '_ = ("gl has:"')
path = os.path.abspath("agent_space/gluon_proto/_gluon_kernels.py"); open(path, "w").write(mod_src)
spec = importlib.util.spec_from_file_location("_gluon_kernels", path); gk = importlib.util.module_from_spec(spec); spec.loader.exec_module(gk)
M, K = 8192, 4096
torch.manual_seed(0)
x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16); w = torch.randn(K, device="cuda", dtype=torch.bfloat16)
q = torch.empty(M, K, device="cuda", dtype=torch.float8_e4m3fn); sf = torch.empty(M * K // 32, device="cuda", dtype=torch.uint8)
kern = {"group32": gk.rms_mxfp8_kernel, "convert": gk.rms_mxfp8_convert}[sys.argv[1]]
run = lambda: kern[(M,)](x, w, q, sf, K=K, GPT=1, NUM_WARPS=4, num_warps=4)
for _ in range(3): run()
torch.cuda.synchronize(); torch.cuda.nvtx.range_push("profile"); run(); torch.cuda.nvtx.range_pop(); torch.cuda.synchronize(); print("DONE")
