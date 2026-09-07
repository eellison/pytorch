import re, os, sys, glob, time, torch, contextlib
case = sys.argv[1]; T0 = time.time()
sys.argv = ["x", "rmsnorm_nvfp4", "8192", "4096", "ours", "default"]
exec(open("agent_space/peak_cmp/peak_cell.py").read().split("torch.manual_seed(0)")[0])
M, K = 8192, 4096
x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16); w = torch.randn(K, device="cuda", dtype=torch.bfloat16)
fmt = "nvfp4" if "nvfp4" in case else "mxfp8"
fn = (lambda x, w: QUANT[fmt](rms(x, w))) if case.startswith("rms") else QUANT[fmt]
args = (x, w) if case.startswith("rms") else (x,)
class _P(InductorChoices):
    @staticmethod
    def should_use_persistent_reduction(*a, **k): return True
ctx = V.set_choices_handler(_P()) if "persistent" in case else contextlib.nullcontext()
with config.patch({"triton.nested_reduction": True, "triton.cudagraphs": False, "fx_graph_cache": False, "triton.autotune_pointwise": False}), ctx:
    compiled = torch.compile(fn, fullgraph=True, dynamic=False); out, srcs = run_and_get_code(compiled, *args); torch.cuda.synchronize()
k = [s for s in srcs if "@triton_heuristics" in s][0]
body = k[k.index("@triton.jit"):]
print(f"===== {case}: {'persistent' if 'persistent_reduction' in k else 'looped'}")
if "--src" in sys.argv or True:
    for line in body.splitlines():
        l = line.strip()
        if any(t in l for t in ("tl.load(", "tl.store(", "reshape(", "tl.split(", "for r0_offset", "max2(", "tl.sum(", "broadcast_to(", "inline_asm", "def triton_", "XBLOCK", "R0_BLOCK")) and "constexpr" not in l:
            print("   ", l[:170])
found = 0
for p in sorted(glob.glob("/tmp/torchinductor_eellison/**/*.ttgir", recursive=True), key=os.path.getmtime):
    if os.path.getmtime(p) < T0: continue
    t = open(p).read()
    if "triton_" not in t[:3000]: continue
    found += 1
    layouts = re.findall(r"#(\w+) = #ttg\.blocked<\{sizePerThread = \[([\d, ]+)\], threadsPerWarp = \[([\d, ]+)\], warpsPerCTA = \[([\d, ]+)\], order = \[([\d, ]+)\]\}>", t)
    nw = re.search(r'"ttg.num-warps" = (\d+)', t)
    print(f"--- ttgir {os.path.basename(p)[:36]} num_warps={nw.group(1) if nw else '?'} convert_layout={t.count('ttg.convert_layout')} local_alloc={t.count('ttg.local_alloc')} local_store={t.count('ttg.local_store')} local_load={t.count('ttg.local_load')} reduce={t.count('tt.reduce')} reshape={t.count('tt.reshape')} split={t.count('tt.split')} join={t.count('tt.join')} shfl/warp_reduce={t.count('gpu.shuffle')}")
    for name, spt, tpw, wpc, order in layouts: print(f"      #{name}: sizePerThread=[{spt}] threadsPerWarp=[{tpw}] warpsPerCTA=[{wpc}] order=[{order}]")
    for m in re.finditer(r"(%\S+ = ttg\.convert_layout %\S+ : tensor<([^>]+)> -> tensor<([^>]+)>)", t):
        print("      ", m.group(1)[:160])
    if found >= 2: break
