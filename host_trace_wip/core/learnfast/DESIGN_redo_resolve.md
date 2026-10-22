# Design note (learnfast -> coordinator): op-local redispatch for library-body custom ops (27B GDN, mm_fp4)

Status: prototype as monkeypatches in learnfast/diag27.py (env DIAG27_*); nothing in core is changed. The model is the
user's op-owned guards: a failing op guard redispatches only that op (credit the user).

## Problem (measured; 27B Qwen3.8 NVFP4, c5h, r27 learn arm)
- d27a (sizes 1..400): the learn window took 596 s:
  - 25 whole-step traces, 553 s;
  - builds, 34 s;
  - redispatch attempts, 6 s (253, all refused);
  - cuBLAS harvests, 2.5 s (802 keys).
- Every retrace is class "dispatch": the graph guards hold and an op-owned guard fails, in
  vllm.qwen_gdn_attention_core_fused_norm_packed (188 refusals) or vllm.flashinfer_mm_fp4 (65 refusals).
- The redispatch is refused at _host_trace_redispatch.py:600, "dispatches otherwise". The redo recorded one op of kind
  "eager" where the variant's op is "traced", because the redo's dispatch declined.

## Rules, each found by fixing the one above it

| # | rule (prototype) | failure without it | soundness | failure mode with it |
|---|---|---|---|---|
| 1 | redo_resolves_arguments = True in the adapters (RD5, existing, user-approved) | conv1d weight from module state: "tensor the trace does not track" -> decline -> retrace | RD5's: a plain tensor with the same StorageImpl and view as exactly one call argument becomes its stand-in; the trace saw that argument there (the adapter swapped it in) | no match -> decline (retrace) |
| 2 | RD5's rule also in a TorchFunctionMode around op.redo | RD5 runs at TorchDispatchMode, below autograd: `weight.view()` -> ADInplaceOrView as_view(real non-inference base, inference stand-in) -> "Cannot set version_counter for inference tensor" | same rule, applied before autograd sees the tensor; tensors that match nothing stay plain | decline |
| 3 | _host_trace_cute._roots_of also searches tr.outer roots | mm_fp4 CuTe: "DataPointer address 256*s.. is not in one traced tensor's storage" (the pointer is into a variant allocation the redo built as an outer stand-in) | a DataPointer must still lie in exactly one root's base symbol; outer roots are variant roots like the trace's own allocations | none or several roots -> decline |
| 4a | RD5's rule at the Triton launch intercept and in record_launch (META values a grid callback sets) | conv state (`initial_states_ptr`) and causal_conv1d's `batch_ptr` reach the Triton launch from module/metadata state | same rule | decline |
| 4b | tie-break: several call arguments of one StorageImpl and view -> the one whose traced-call input IS the object the body reached (`x is t`), if exactly one | vLLM's hybrid KV shares one buffer across layer groups: the conv state view equals 3 call arguments (distinct roots p1812/p1814/p1816) -> RD5's "exactly one" fails | see below | no identity match, or several -> decline (never guess) |
| 5 | (OPSPAN, rejected as written) plan_memory keeps every allocation a guard-owning op's launches reach live through the op's last launch | RD2 section refused: "base 4 is not live over its op's launches" (the GDN op's input dies mid-chain, at seq 17 of 13..70) | lifetimes only lengthen | memory: +42% plan peak (T=8192: 1.29 -> 1.82 GB); see the alternative below |

### Why identity is exactly right for 4b, and what happens otherwise
- The adapter lifts state tensors as call arguments by object: the argument at position p is the tensor object found in
  module state. While the trace ran, it swapped stand-in(p) into that slot. So the trace's read of that slot is
  stand-in(p) for the p whose traced-call input is the object now in the slot.
- In the redo the body reads the same slot. If it finds the same object o, then "the argument whose traced-call input
  is o" names exactly the stand-in the trace used. Storage and view equality (RD5) is necessary; identity removes the
  ambiguity that aliasing across layer groups creates, and adds nothing else.
- If the body reaches a different Python object over the same storage (vLLM rebuilt the view, another layer's view, a
  view made in the body), identity finds no argument. With several storage/view candidates, 4b declines. With one, RD5
  applies as before. The op then runs eagerly and the call retraces. It never picks among candidates without identity.
- A per-call caveat RD5 already has: the redo resolves against the traced call's input objects. The adapter passes the
  same state objects every call (static prefix), so the position found is also the current call's.

## Measured with rules 1-4b (+5)
- d27l, sizes 1..40, all rules: 0 retraces in the learn range (d27a: 8, at 2/3/5/7/9/13/17/33). The prefill entry had 1
  trace, 5 redispatches and 47 replays. The window was 50.6 s for 40 sizes. The 5 redispatches took 39 s, each
  including a Triton JIT of a new GDN specialization (about 8-12 s; a persisted Triton cache removes it).
- d27m (1..1000) and c27a (--check, sizes 2..300): an ILLEGAL MEMORY ACCESS at the first sections past one chunk (T > 64;
  crash at 69/70 and at 100). Sizes <= 64 ran without error. Bitwise verdicts are pending (c27b: <= 64 with all rules;
  c27s: rules 1-4b without OPSPAN). Section entries for the GDN op beyond one chunk are therefore NOT safe as
  prototyped: this needs an RD2 investigation in core (chunk-dependent section temporaries, the op's memcpy, or a guard
  missing from the redo).
- mm_fp4's pow-2 flips need no section and redispatch with rules 1-4b. In d27l the traces at 3, 5 and 9 are gone. Over
  1..16384 that is about 14 bucket flips.

## Alternative to OPSPAN (rule 5) with no memory cost (not prototyped)
- After base b's last use, inside the op's chain, the only allocations that can take b's bytes before the chain ends are
  those made inside the chain.
- In a section these are either dropped (not made: the section's temporaries replace them) or kept (the op's outputs,
  keyed-site scratch).
- So check_section can accept b when no kept allocation made within (last use of b, end of chain] overlaps b's placed
  bytes in the plan (run_buffer offsets from place()).
- This is exact. It decides per section, at entry time, and changes no plan.

## Cold-start relevance
- 27B learn today is dominated by whole-step traces: d27a measured about 22 s per trace, about one per 27 sizes.
- With mm_fp4 redispatch only (rules 1-4b), traces drop by roughly 14 over the range.
- With working GDN sections, the remaining cost is redos: about 0.1-0.5 s each, plus about 10 s of Triton JIT per new
  specialization on a cold Triton cache. A persisted Triton cache removes the JIT.
- Numbers that include a cold Triton cache are labeled as such.
