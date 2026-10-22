# run_buffer NaN after a respec (vLLM open variant: trtllm fork + silu_and_mul port), candidate 2

Symptom: decode at bs 5 and 8 returns NaN (44/92 bitwise). Same run with ARMV_MEMORY=eager is 92/92. Only the variants
that respec built (the fork's other split regions in the bs 2..32 bucket; no trace) go bad; the traced variant is fine.

What the plan dump shows (out/diag_plan2_fork_silu.json, summary()["plan_dump"], plan_memory(lowered, "run_buffer")):
in the respec'd variants v1/v2 of decode/T1, layer 0's o_proj GEMM after the fmha kernel has launch seqs 35 (node 0)
and 34 (node 1), in that launch order, while the two allocations they use got seqs 37 and 38:
- allocation 22: planned interval (seq 37, last 34), inverted, used by launches 35 and 34;
- allocation 21: planned interval (seq 38, last 44), used by launch 35, before its seq.
The traced variant v0 has the same GEMM at seqs 37/38 with allocations 16/17 at (35, 43)/(36, 38): consistent.
So in the respec'd tape the re-bound opaque launches keep sequence numbers from before while the allocations they use
(the rebound mm's operands/scratch) are numbered after them. plan_memory then treats those allocations as dead during
the launches that use them, and run_buffer hands their bytes to other temporaries live at that time (with the silu port
in the trace there is such a temporary; with silu as an extern the layout happens not to collide).
Likely site: _host_trace_redispatch.respec (rebuilding the tape with the selectors' ops dispatched again and keyed
sites rebound) assigns new allocations seqs at the end of the tape, or re-inserts launches without renumbering.

Standalone check: probe/runbuffer_nan_repro.py (embedding small/large-index selector + two harvested mms + traced
pointwise; prints bitwise per call and, per variant, any use outside its allocation's planned interval and whether
launches are out of seq order). vLLM repro: see STATUS.md / the coordinator message (ARMV_TRACED_IMPLS_NAMES=silu_and_mul
with the fork).

## Candidate 3: fixed (2026-10-08 13:36)
Same vLLM repro (fork + silu port, decode bs 5/8/64, --check, ARMV_DUMP_PLAN=1) on candidate 3:
- int6 even top snap/step246 on build_cpp8 (out/c3nan_c3.json): 92/92 bitwise; traces 7, variants 7, eager 3; no respec
  (the attribute is gone); decode bs 2..32 bucket: 3 traces, retrace_causes {('dispatch',
  'flashinfer_ht.trtllm_paged_attention_decode.default', guard on the split count `min(max(19 // ..., 1), 8) == 4`,
  trtllm_trace.py:438, 'another launch topology in ...'): 2}; plan_dump: 0 uses before allocation in every variant.
- pinned lane (step242-based, out/c3nan_pinned.json): identical.
Why: respec was removed in candidate 3 (FZ08, from step 201), so a split-count flip in the fork's launcher now retraces
(its own dispatch is the recorded cause) instead of splicing a respec'd tape; and check_plan rejects inverted /
used-before-allocated intervals at build time. The standalone repro stays as is (it never triggered on candidate 2).
