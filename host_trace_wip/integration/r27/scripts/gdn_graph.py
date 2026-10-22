# gdn_graph.py: FlashInfer GDN prefill eager vs plain torch.cuda.CUDAGraph replay (no host trace, no vLLM), same process, same inputs:
# per shape, eager run twice and graph replay twice; bitwise agreement and NaN counts. A kernel whose result depends on launch timing
# shows eager vs graph differences.
import os
import torch
from flashinfer.gdn_prefill import chunk_gated_delta_rule

dev, HK, HV, D = "cuda", 16, 48, 128
for spec in os.environ.get("GDN_SHAPES", "1x64 1x512 1x2048 4x128").split():
    seqs, T = map(int, spec.split("x"))
    g = torch.Generator(device=dev).manual_seed(0)
    N = seqs * T
    q = torch.nn.functional.normalize(torch.randn(N, HK, D, device=dev, generator=g), dim=-1).to(torch.bfloat16)
    k = torch.nn.functional.normalize(torch.randn(N, HK, D, device=dev, generator=g), dim=-1).to(torch.bfloat16)
    v = torch.randn(N, HV, D, device=dev, generator=g).to(torch.bfloat16)
    gate = torch.exp(-torch.rand(N, HV, device=dev, generator=g))
    beta = torch.rand(N, HV, device=dev, generator=g)
    state0 = torch.zeros(seqs, HV, D, D, device=dev)
    cu = torch.arange(0, N + 1, T, dtype=torch.int64, device=dev)
    out = torch.empty(N, HV, D, dtype=torch.bfloat16, device=dev)
    ost = torch.empty(seqs, HV, D, D, dtype=torch.float32, device=dev)
    st = state0.clone()

    def call():
        st.copy_(state0)
        chunk_gated_delta_rule(q=q, k=k, v=v, g=gate, beta=beta, initial_state=st, output_final_state=True, cu_seqlens=cu, output=out, output_state=ost)

    res = {}
    for i in range(2):
        call(); torch.cuda.synchronize(); res[f"eager{i}"] = (out.clone(), ost.clone())
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        call()  # warm-up on the side stream
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for i in range(2):
        graph.replay(); torch.cuda.synchronize(); res[f"graph{i}"] = (out.clone(), ost.clone())
    eq = lambda a, b: torch.equal(a[0].nan_to_num(7.0), b[0].nan_to_num(7.0)) and torch.equal(a[1].nan_to_num(7.0), b[1].nan_to_num(7.0))
    nan = {k_: int(v_[0].isnan().sum()) for k_, v_ in res.items()}
    print(f"{spec}: eager-eager {eq(res['eager0'], res['eager1'])} graph-graph {eq(res['graph0'], res['graph1'])} "
          f"eager-graph {eq(res['eager0'], res['graph0'])} nan {nan}", flush=True)
