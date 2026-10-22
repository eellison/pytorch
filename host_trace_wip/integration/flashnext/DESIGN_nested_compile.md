# Design note: nested torch.compile under a host trace's warm-up (Flash-Next Vc trace-call mismatch)

Status: for the user's review (coordinator asked for it 2026-10-14). No core change made. Lane: land/core/flashnext.

## Symptom
Qwen3.8-Flash-Next, arm Vc (host trace around vLLM's compiled forward), bitwise check of every call against a fresh
eager reference (fn_vc1 / fn_vc2 / fn_vc3, diagnosis config):
- every replay is bitwise (outputs and GDN state);
- every TRACE call is not (outputs max_abs 13-41; 73 state tensors differ);
- the decode variant has 48 eager steps, all vllm.moe_forward_shared, reason "torch.compile cannot JIT compile during
  CUDA graph capture".
The eager-forward arm (V) has neither problem.

## Cause
1. Under vLLM's compile mode most CustomOps are disabled. A disabled CustomOp with compile_native=True (SiluAndMul,
   QuantFP8, ...) runs `torch.compile(forward_native)` (CustomOp.maybe_compile). Inside the top-level compiled graph Dynamo
   inlines it. Inside an opaque custom op it is a nested compiled function. Flash-Next's case: moe_forward_shared ->
   SharedExperts -> Qwen2MoeMLP.act_fn = SiluAndMul (fn_vc3's stack).
2. HostTraceReplay's trace call returns its warm-up's result (_host_trace_replay.py: "the warm-up is the call"). The
   warm-up runs the function eagerly under the witness TorchDispatchMode.
3. Dynamo does not compile under a non-infra dispatch mode (torch/_dynamo/convert_frame.py: "non-infra torch dispatch mode
   present"). So during the warm-up every nested compiled function runs uncompiled: eager F.silu(x) * y with bf16 rounding
   between the ops, instead of Inductor's fused kernel.
4. At the trace itself the nested function tries to compile under the capture and declines. The opaque op becomes an eager
   step, and replays rerun it for real (no mode active), i.e. stock's compiled code. That is why replays match.
So the trace call runs other kernels than stock, and its output and state are what those kernels produce.

## Options
(a) Return a replay of the just-built variant as the trace call's result, not the warm-up's. This makes the trace call
    equal to later replays. It is not enough on its own. Today the nested region is an eager step, so the replay happens
    to run stock's code. But once that region is traced (option (c), or any nested compiled function inside a region the
    trace does record), the recorded kernels are the ones the witness pass ran: the uncompiled ones. Replays would then
    differ from stock too, consistently, and a bitwise check against an in-process stock reference would still catch it
    only if the numerics differ.
(b) RECOMMENDED correctness fix: the warm-up / witness mode must not change what nested torch.compile'd code does. Either
    the witness mode is marked as infra for Dynamo (the mode attribute Dynamo checks for "ignore compile internals"),
    so the nested function hits its compiled cache entry exactly as in stock, or the warm-up runs nested compiled
    functions outside the mode. Then the warm-up result is stock's, and a trace that records the region records stock's
    compiled kernels. Guards and recompiles must behave as in stock: a nested function compiled once at vLLM's warm-up
    should hit its cache, not recompile, during our warm-up.
(c) Follow-up, removes the 48 eager steps: trace through nested compiled functions as the Vc seam route does for the
    top-level compiled forward. At the trace, call the function's compiled artifact (Inductor wrapper) with the trace's
    values, instead of letting Dynamo try to compile under the capture. Needs (b) first, so what (c) records is stock's code.

## Test
tests/test_nested_compile_trace.py: a torch.library custom op whose body calls a torch.compile'd silu-and-mul (warmed up
eagerly first), under HostTraceReplay at sizes 4 / 8 / 64. On step331 + build_cpp14 (t_nc1, 03:4x):
test_the_probe_distinguishes_compiled_from_uncompiled and test_replays_run_stock_compiled_code pass;
test_trace_call_runs_stock_compiled_code fails (698 / 2048 elements differ) and should pass with (b). tests/repro_nested_compile.py prints, per
call, whether the output equals the compiled eager reference or the uncompiled one.

## Scope (read-only audit, vLLM 0.29)
CustomOps that maybe_compile under compile mode (compile_native=True): SiluAndMul, SituAndMul, SiluAndMulWithClamp,
ReLUSquaredActivation, XIELU, QuantFP8 (+ MLA's decode QuantFP8). Explicit @torch.compile on model paths: VocabParallel
embedding's get_masked_input_and_mask (TP > 1 only), fused_moe grouped_topk (non-monolithic routing), _swiglu_limit_torch,
gdn prepare_gdn_attention_core_inputs (ROCm aiter path only).
Per model (TP1, the arms' configs):
- Qwen3-8B: SiluAndMul in the MLP at top level (inlined); no compile_native op inside an opaque op. Not affected.
- Qwen3.8-27B NVFP4: SiluAndMul at top level (fused with the NVFP4 quant by vLLM's act_quant pass); QuantFP8 query quant
  in Attention.forward before the opaque attention op (top level); nothing compile_native inside the GDN core op. Not affected.
- MiMo-V2.6-Flash: QuantFP8 for the DeepGEMM linears runs at top level, before the opaque fp8_gemm_nt_op; MoE is
  FlashInfer's monolithic TRTLLM (routing in-kernel, no SiluAndMul); no shared experts. Not affected.
- Qwen3.8-Flash-Next: shared expert SiluAndMul inside moe_forward_shared. Affected (this note).
- Not in the current set but affected the same way: any MoE with shared experts on vLLM's generic MoE path
  (Qwen3.5-35B-A3B, Qwen3.5-397B-A17B).
The eager arms (V) are affected only by explicit @torch.compile functions, which compile even with enforce_eager; none
is on these four models' TP1 CUDA paths. Kernel identity in each lane's variants should still be checked against stock,
since an equal-numerics kernel swap would pass a bitwise check.

## Prototype of (b) (lane-local, no core edit; 2026-10-14)
scripts/witness_infra.py (FN_WITNESS_INFRA=1): `_host_trace_tape._Witness.ignore_compile_internals = classmethod(lambda cls:
True)`. By TorchDispatchMode's contract the mode is off while Dynamo looks up / compiles a frame and on for the compiled
artifact's runtime, so the nested function hits stock's cache entry during the warm-up; the witness still sees the
wrapper's operator calls. As a core change this is one classmethod on _Witness.
- tests/test_nested_compile_trace.py with it (t_nc2): 3/3 pass (the trace-call test failed without it, 698 / 2048).
  repro_nested_compile.py (r_nc2): the trace call now equals the compiled output; 5/5 replays as before.
- Flash-Next Vc decode, fn_vc2's config (fn_vc5): 20/20 checks bitwise incl. GDN state, all 4 trace calls included (fn_vc2:
  16/20, every trace call wrong). Variants unchanged (same kernel counts 1245 / 1245 / 1137 / 1340; 49-50 eager steps, 48 of
  them moe_forward_shared: (c) is still needed for those).
- Qwen3-8B Vc decode (bs 8 / 1 / 64 + fills, 2 rounds, same driver), without (q8_vc_base) and with (q8_vc_wi) the prototype:
  identical: 32/32 bitwise both; traces 3 / replays 27 / eager 2; variants 0 eager steps, 1 segment (397 / 505 kernels);
  aten::mm x28 only; the dumped variant tapes are equal record for record.
