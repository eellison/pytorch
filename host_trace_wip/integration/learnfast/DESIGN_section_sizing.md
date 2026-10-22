# Design note (learnfast -> coordinator): the RD2 "section sizing" crash is a redo baking per-step ints from forward context

Status: root cause identified from the prototype's dumps (9B q9e, 27B d27m/c27a, memcheck q9t). Nothing in core is
changed. Sections are refused at creation in the meantime (no_sections.py; the adapters' own audit refuses too).

## Evidence
- memcheck (q9t, Qwen3.5-9B): a section was made at T=60 and replayed at T=61. Result: 512 x "Invalid __global__ read of
  size 16 at chunk_fwd_kernel_o (FLA chunk_o.py:104)", up to a few hundred bytes past the end of a 2 MiB allocation.
  Without the sanitizer: IMA (Xid 31, MMU fault). 27B shows the same pattern past 64 tokens.
- The temporaries of one GDN section at creation (q9e, T=60; Temporary.sizes/strides as recorded, in the variant's
  context):

| base | dtype | sizes | strides |
|---|---|---|---|
| 1016, 1017 | bf16 | (s2348, 32) | (32, 1) |
| 1018 | bf16 | (8192, **60**) | (1, 8192) |
| 1019, 1020 | bf16 | (**60**, 16, 128) | (2048, 128, 1) |
| 1021 | bf16 | (**60**, s26, 128) | (128*s26, 128, 1) |
| 1022, 1023 | fp32 | (**60**, s26) | (s26, 1) |
| 1027 | fp32 | (1, **60**, s26) | (60*s26, s26, 1) |

  Every GDN layer's section shows the same pattern. The token count 60 is baked as a constant into the conv output,
  the q/k/v split, the gates and the beta buffers. Other temporaries of the same section are symbolic. At T=61 the
  section allocates 60-row temporaries, and the FLA kernels (whose T comes from cu_seqlens on the device) touch row 61.
- RD2 itself does what it says: each temporary's nbytes is a program row evaluated per call. The rows here are
  constants.

## Cause
- The op (vllm.qwen_gdn_attention_core_fused_norm_packed) reads per-step Python ints from vLLM's forward context
  (attention metadata: the number of actual tokens and lengths that size the conv and split buffers), not from its
  arguments.
- In the trace, the adapter swaps symbolic metadata into the forward context, so those ints are SymInts and the sizes
  are formulas.
- In a redo, the body runs outside that swap. It reads the current call's concrete ints, and the sizes become
  constants. Plain Python ints leave no guard, so nothing marks the entry as valid only at T=60.
- This is the int analogue of the module-state tensor problem (rules 1-4 / closure). Closure cannot cover it: these
  ints change every step.
- It is not limited to sections. A same-chain entry whose launch scalars or grid derive from those ints would also be
  baked. In our runs no such GDN entry was accepted, because GDN entries were sections and refused. mm_fp4's entries
  read only their arguments; that is why c27s / ccl were bitwise.

## Fix: (A) redo_context, (B) refusal without it

### (A) redo_context (opt-in; the vLLM adapters provide it)
- API: `HostTraceReplay(..., redo_context=None)`. A callable that takes the redo's stand-ins for every argument of the
  entry's call and returns a context manager.
- Building the stand-ins. Core builds them the way _redo builds an op's own call today:
  - a tensor argument: its root, at its input record's layout, in fresh symbols bound to the variant's (as RD5's
    resolver does for one tensor);
  - an int argument: a fresh SymInt bound to the variant's int-input symbol;
  - anything else: the traced value.
  _redo enters `redo_context(stand_ins)` around `op.redo(*call)`.
- The vLLM adapter's redo_context is its step function's own state swap, minus running the model:
  - the forward context rebuilt with `_md_rebuild` from the stand-in metadata tensors and ints (attn_metadata,
    slot_mapping);
  - the attribute swap (kv caches etc.);
  - `_reparametrize_module` with the stand-in parameters.
  The adapter already does exactly this when a trace runs the step. Under closure, the parameter and KV part becomes
  the declared constants and only the per-step part is rebuilt.
- Soundness: the redo's body then sees what a trace at the call sees for that op.
  - Per-step ints are SymInts of the variant. Their guards land in the entry's predicate, and sizes are formulas, so
    no baked constant remains.
  - State that the step function does not rebuild is fixed for the entry under the trace's existing contract (Python
    state is fixed at trace time). The redo assumes nothing beyond what a trace assumes.

### (B) Without redo_context: refuse
- Any redispatch of a non-ATen custom op or a dispatch unit is refused before the redo runs, and the call retraces.
  That is learnfast/no_lib_redo.py moved into core, default on.
- With redo_context declared, (B) does not apply to that entry's ops.
- ATen operators and Triton / CuTe DSL / jiterator launches redo from their recorded arguments and are unaffected.

### Why it matters now (measured)
- MiMo: about 33 s per new batch size, i.e. about 2.3 whole-forward retraces (10-12 s each) per size. All of them come
  from op-owned host choices that (B) refuses today: DeepGEMM's M <= 32 block config (13), its SM-wave count (2), FA4's
  tile count (3), the FlashInfer MoE tactic bucket (5). With (A) each becomes a local redispatch of one op.
- 27B: 77 whole-step retraces of about 22 s over 1..2050 (GDN classes plus mm_fp4 buckets).
- Estimates, not measured: a redo of one op is about 0.1-0.5 s (d27a's refused attempts cost 24 ms). A new kernel
  specialization adds its JIT (Triton about 10 s cold, cached afterwards; the same JIT a trace pays).
  - MiMo: a new batch size drops from about 33 s to about 1-2 s, plus any first-time JIT.
  - 27B learn 1..2050: about 1,700 s of traces drops to about 1-2 min of redos, plus cold-cache JIT.
  - Cold start: the per-size learn loop then needs no forward per size (the warm-up API), so the range warm-up
    becomes the redos plus the cuBLAS learn.

## Regression test (core; run under compute-sanitizer memcheck)
- A custom op whose Python body sizes a temporary from a per-step int held in module/context state (set by the step
  function around the call, like vLLM's forward context) and launches a kernel that touches every row.
- Trace at A. Force a redispatch at B (a guard flip). Replay at C > B.
- Without redo_context: the redo must refuse (B), and C traces again. Bitwise vs eager; memcheck clean.
- With redo_context: every temporary is sized at C. Bitwise; memcheck clean.
- The test fails today: it reproduces this crash.

## Interaction with OPSPAN / exact-overlap
- The q9t crash does not depend on OPSPAN's liveness change. OPSPAN only let the section pass check_section, and the
  baked sizes did the damage.
- Once (A)/(B) land, the exact-overlap check (DESIGN_redo_resolve.md) can replace OPSPAN without a memory cost.
