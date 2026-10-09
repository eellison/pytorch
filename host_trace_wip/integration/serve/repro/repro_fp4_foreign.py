# Repro for R1-V's 0 variants (decline "sNNNN is not read from the call's inputs" at lowering): vLLM's NVFP4 linear on
# Blackwell (FlashInferCuteDslNvFp4LinearKernel, auto-selected) = torch.ops._C.scaled_fp4_quant.out into tensors from
# create_fp4_output_tensors (torch.zeros scale buffer), then flashinfer.mm_fp4(backend="cute-dsl"). The GEMM reaches the
# tape as an eager call of a TVM-FFI function from outside CuTe DSL ("cute", _Foreign); one of its scalar arguments is
# 256*s<alloc.base/256>, the in-trace scale buffer's address, which the eager-call lowering cannot express.
# Random weights/scales (only the trace/lowering matters; replay vs eager is still compared bitwise). No vLLM engine.
# Run: CUDA_VISIBLE_DEVICES=1 [REPRO_OBSERVE=1] bash serve/python_vllm_cand3.sh serve/repro/repro_fp4_foreign.py
# REPRO_OBSERVE=1: torch.cuda._host_trace_cute.install() first, so the GEMM is compiled under observation (CuTe DSL route)
import os
from types import SimpleNamespace

import torch

if os.environ.get("REPRO_OBSERVE") == "1":  # observe cute.compile before FlashInfer compiles its GEMM (the CuTe DSL route)
    import cutlass  # noqa: F401
    import torch.cuda._host_trace_cute as _cute

    _cute.install()
import torch.cuda._host_trace_replay as R
from torch.cuda._host_trace_harvest import HarvestProvider

from vllm.model_executor.kernels.linear.nvfp4.flashinfer import FlashInferCuteDslNvFp4LinearKernel as K

torch.manual_seed(0)
N, KD = 512, 1024
layer = SimpleNamespace(weight=torch.randint(0, 255, (N, KD // 2), dtype=torch.uint8, device="cuda"),
                        weight_scale=(torch.rand(N, KD // 16, device="cuda") + 0.5).to(torch.float8_e4m3fn))
K.process_weights_after_loading(None, layer)
layer.alpha = torch.tensor(1.0, device="cuda")
layer.input_global_scale_inv = torch.tensor(1.0, device="cuda")
layer.output_size_per_partition = N


def fp4_linear(x, weight, weight_scale, alpha, gs):
    # every tensor the step reads is an argument (as arm V passes a model's parameters)
    lay = SimpleNamespace(weight=weight, weight_scale=weight_scale, alpha=alpha, input_global_scale_inv=gs,
                          output_size_per_partition=N, weights_padding_cols=getattr(layer, "weights_padding_cols", 0))
    return (K.apply_weights(None, lay, x),)


PARAMS = tuple(t.detach() for t in (layer.weight, layer.weight_scale, layer.alpha, layer.input_global_scale_inv))


ext = (torch.ops._C.scaled_fp4_quant.out,)
for name, prov in (("blas+extern(_C.scaled_fp4_quant)", HarvestProvider(("blas", "extern"), extern_ops=ext)), ("no provider", None)):
    entry = R.HostTraceReplay(fp4_linear, opaque=(prov,) if prov else (), memory="run_buffer")
    ok = []
    for m in (3, 17, 64, 17, 3):
        x = torch.randn(m, KD, device="cuda", dtype=torch.bfloat16)
        got, want = entry(x, *PARAMS), fp4_linear(x, *PARAMS)
        ok.append(all(torch.equal(g, h) for g, h in zip(got, want)))
    print(name, "traces", entry.traces, "replays", entry.replays, "eager", entry.eager, "variants", len(entry.variants), "bitwise", ok,
          "declines", sorted({str(d)[:160] for d in entry.declines})[:3], "retrace_causes", dict(list(getattr(entry, "retrace_causes", {}).items())[:3]), flush=True)
