# Integration: decisions to confer on (prepared 2026-10-06, while candidate 2 validates)

Base for everything below: candidate 2 (int6 step187 odd / step188 even) once validated.
Dry-run evidence (step177/178): SGLang F/Fc/P/Pc/mixed bitwise and at or below overlay trace counts;
vLLM drive 148/148, range 969/969 once the budget is gone (96).

Ask the user one at a time, in this order.

## 1. Scope: which engines and arms go in the paper
- SGLang F (untouched engine, adapter only): decode parity, prefill 1.02-3.48x (overlay numbers).
- SGLang P (adapter owns decode prep in one entry): ~0.5 ms/step faster synchronous, ~4% with the overlap scheduler.
  Open since 10-05 (user: "i cant make the call, yet"). Until decided, report both.
- vLLM V: prefill 2.2-4.9x, mixed 4.0-4.1x, decode 1.04-1.08x (overlay numbers).
- Question: both engines, and is P in or out of the headline?

## 2. Integration-side fixes (user: hashed-SymInt pins are fixed in integration, not core)
- C1/T2: SGLang `use_bf16_splitk_gemm`'s `(m, n, k) in SET` pins m. Options: rewrite as explicit comparisons in the adapter's
  copy, or keep the adapter's `_by_runs` learn. Recommendation: explicit comparisons (no learning, no extra traces).
- Drop `budget_unit="group"` from land/scratch/sglang/armF/prefill/{adapter,e2e}/adapter.py:163-164 (removed in 96; they raise
  TypeError until then) and retire vllm/bench/group_budget.py.
- Dead switches the adapters still set (int6 retired them): learn_in_learning_variants, redispatch_learners,
  cute_without_tvm_ffi (tgv adapter). Remove from the adapters.
- C4 (Triton alignment): not a core issue on int6 (per-op guard; probe in land/scratch/c4probe). The adapter's
  do_not_specialize copy (PREP_BS_META) can probably go; confirm counts without it.
- Question: OK to make these adapter changes, and the C1 rewrite as recommended?

## 3. Overlay-only behaviour not in int6
- INVENTORY.md: 7 missing on odd / 6 on even; the ones that matter are C1/T2, C3 (repro being written, next round) and C4 (above).
  None changed dry-run results.
- Question: anything there the user wants in core before measuring?

## 4. What we measure and how we report
- Workloads: one decode / prefill / mixed set per engine (feedback: no op long tail), open + closed variants.
- Counts first (feedback): traces and why, variants, boundaries, op re-runs vs the minimum, model forwards spent on capture
  (SGLang default: 126 buckets x full model; ours: 1 trace per class), startup seconds cold/warm, graph memory.
- Then wall-clock (speedup), coverage (bitwise vs eager, boundaries), memory last.
- Re-measure everything on candidate 2, GPU 0 (timing) / GPU 1 (correctness); the 10-03/04 numbers were overlays on GPU 2.
- Question: is that workload set right, and do we need any baseline beyond engine default (FULL graphs / piecewise)?

## 5. Remaining dry-run items (run after the above, on candidate 2)
- SGLang: odd mixed F (failed on 10-05 from foreign GPU memory), PREP_SEQ_LEN=250 page-crossing runs, Triton backend runs.
- vLLM: the same matrix on candidate 2 with no adapter budget option.
- Each core bug found gets a core regression test (feedback).
