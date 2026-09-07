# Review: indexed forwarding (F1) + lazy domain projection (F2) + upper-stack replay

Date: 2026-08-22. Scope: the two uncommitted foundation layers
(`indexed_projection_work/`, `domain_projection_work/`) and the #190596/#191974
replay, per the READ_ME_FIRST review order items 4-5. #190594/#190595 were
reviewed previously and are not re-reviewed here. Method: coordinator code read
of the F2 core plus four adversarial tracks with live differential repros on
B200 (artifacts under `agent_space/rev_8bit/`, `rev_f1/`, `rev_replay/`,
`rev_evidence/`). Worktrees and canonical patches were not modified; the only
side effect is two fuzz-result JSONs regenerated with matching fresh runs
(original backed up in `rev_evidence/`).

## Verdict

- **F1 (indexed forwarding): correctness-clean under attack.** Guard
  semantics, must_forward reachability, exact-access reconstruction
  (zero misses), and aliasing/mutation all held. Two hardening items and one
  design boundary to document (P1.2, P1.3).
- **F2 (lazy projection): correctness-clean under attack, including the 8-bit
  frontier.** The frontier is placement-only: both gate outcomes materialize
  through the same broadcast emitter, so a mispredicted gate changes timing,
  never semantics. Its pow2-autotune arm rests on real, asserted invariants
  (every candidate R0_BLOCK capped at min(size_hint, TRITON_MAX_BLOCK); the CD
  tuner applies the same cap with pow2-only steps), verified in source and by
  20+ fresh boundary cases, all bitwise: fixed R0=16..8192 (incl. the
  oversized masked half-block), masked tails, divisible non-pow2 extents,
  autotuned pow2 deferral, CD on both arms, dynamic-R, partial x-tiles.
  FP8 bitcast handling is airtight in all three emitters. The implementation
  faithfully matches the Iteration 11 countersigned contract.
- **Upper-stack replay: one P0 miscompile introduced by conflict resolution**
  (below). Every other #190596 use-site migration was verified
  semantics-preserving, including the C1 looped-external reload, the
  pow2/strict-reduction gates, persistence gates, and contiguous lane proofs.
  CONTIGUOUS provably never enters lazy deferral (three independent gates).
- **Evidence base: genuine and unusually strong.** All patch hashes verify;
  the tested code is byte-identical to the recorded patches; suites and both
  fuzz matrices independently reproduce. Presentation flaws listed under P2.

## P0 -- replay miscompile (proven, fix proven)

`upper_projection_latest_replay_wt/torch/_inductor/codegen/simd.py:4303-4308`:
the replay computes `internal_source_names` as any `must_forward` source over
ALL layouts; the original #190596 restricted it to lane-projected
(INTERLEAVED/CONTIGUOUS) sources. Foundation records also include
IDENTITY/BROADCAST, and a parent-written reduction output consumed by the
epilogue is IDENTITY + must_forward=True, so the set is spuriously non-empty.
That skips the `codegen_body()` pass-boundary flush (simd.py:4376-4378), and
in LOOPED kernels the epilogue emits inside the parent reduction loop,
referencing the finalized accumulator defined only after the loop. Today:
Triton NameError; failure class: silent partial-reduction numerics.

- Repro: `rev_replay/repro_swiglu_looped.py` (fails on replay tree, passes
  bitwise on clean 3f62ae27baa). Full nested suite on the replay tree: 460
  ran, 6 errors -- four of #190596's own chunk tests plus two PRE-EXISTING
  interleaved tests (not contiguous-specific).
- Fix: restore `projection.layout in (INTERLEAVED, CONTIGUOUS)`; proven in
  `rev_replay/replay_fixed_tree/` (repro bitwise, all 6 tests pass).
- Why recorded validation was green: `UPPER_STACK_REPLAY.md`'s focused `-k`
  selection deselects all six failing tests, and its claim that the
  must_forward derivation "preserves #190596's existing legality and lifetime
  rules" is false for this site. The existing pinned pass-boundary tests
  structurally cannot catch this (their internal sources are lane-projected,
  where skipping the flush is correct).
- Required: apply the fix, regenerate `upper_stack_replay_190596_191974.patch`,
  rerun the FULL nested suite, fix the doc bullet, and add the repro as a
  regression test.

## P1 -- fix before these become PRs

1. **Fail-open `_PROJECTION_BARRIERS` (4th carry, now an unmet explicit
   requirement, with a concrete instance).** ARCH_REVIEW's operation contract
   required the fail-open deny-list not survive the PR (exhaustive
   classification in `ops_handler.py` + an OP_NAMES completeness test); later
   signoffs neither implemented nor retracted it. A mechanical audit against
   the full OpsHandler protocol found `store_reduction` missing: it falls to
   `_default`, is not a barrier, and is not overridden like `store`.
   Unreachable with deferred args today (reduction itself is a barrier), but
   it is the one gap where fail-open means a WRONG STORE (group-width value
   stored at lane indexing) rather than wrong perf. Add `store_reduction` to
   the set (or override it), and implement the completeness test or fail
   closed for unclassified ops. Subgraph `output`/`placeholder` are safe
   today only because barrier paths materialize their inputs.
2. **Standalone-path epilogue loads still ride name-keyed forwarding.** The
   standalone plan filters deps to full-numel accesses, so reduced-resolution
   parent outputs (e.g. the grouped amax scale) are unplanned and served by
   generic `CSE.store_cache` with zero index check (common.py:3008-3010,
   observed live). Protection is the generic fusion gate
   (`fusable_read_and_write` index equality) -- every shifted variant
   attempted declined fusion -- but the README's "keyed by exact normalized
   memory access" is only true for planned names, and the next layer must not
   assume all epilogue loads are index-checked. Either plan standalone reduced
   outputs as BROADCAST (the new `_sub_parent_broadcast_projections` is
   topology-agnostic) or document the asymmetry. Add the suggested debug
   tripwire: a planned NAME whose access misses the plan must never hit
   store_cache (guards future normalize drift, whose failure mode is silent).
3. **`_IndexedProjectedValueStore.record_store` records atomic-mode stores as
   forwardable** (no mode gate, unlike `CSEProxy._update_store_cache`).
   Forwarding an atomic store's register would forward the addend, not
   memory. No repro constructible through real lowerings today; add the
   one-line `mode is not None` skip to mirror the generic invariant.

## P2 -- before landing (cheap, mostly tests and docs)

1. Fold the new differential batteries into checked-in tests: fixed-config
   numerics (both gate arms; previously ZERO fuzz coverage -- including the
   oversized R0=8192 case, whose only prior evidence was form checks),
   CD-tuning cases, lane-width 8-bit stores of group values, group-domain
   where/two-scale/reload chains (`rev_8bit/gate_battery.py`,
   `trigger_battery.py`, `final_battery.py`), plus the coverage F1 genuinely
   lost: NVFP4-inline-asm + swizzled-fp8-scale combination, the
   `.to(tl.float8e4nv)`-exactly-once check, and the fixed-config looped
   variant.
2. Doc corrections: the persistent 4096x8192 "1.05x faster" row is
   contradicted by the newer uncited artifact (projection ~0.8% slower;
   should read "tied/noise"); the "after the 8-bit-frontier fix" looped table
   rows and register counts are pre-fix captures argued-unchanged -- label
   them or re-measure; the FlashInfer CD headline uses the best of 4 runs
   (reruns: 10.9-11.1%; conclusion stands); RELAYER_TEST_RESULTS' documented
   fuzz capture command is wrong (would compare projection to itself; actual
   captures verified genuine by re-capture).
3. Re-include the two `kernel_num_gb` tests: both F1 and evidence tracks
   independently proved the recorded failures are an artifact of the
   4-module preload runner (they pass under the full-package overlay at
   baseline, F1, and composed).
4. Delete the stale pre-frontier capture
   `domain_projection_work/oversized_group_gate_4096_rblock8192.py` (shows
   the deferred form; contradicts the current audit file).
5. Policy-text deviation (perf-only): 8-bit values produced by non-to_dtype
   ops (bool, pack-asm outputs, bitcasts) defer past the stated frontier.
   Align the README text or the trigger.

## P3 -- cleanups

- Dead `len(deferred) == 1` condition in the to_dtype trigger (to_dtype has
  exactly one tensor operand).
- `_view`/`_split` cache-miss fall-through re-emits against the source name
  instead of asserting liveness; confirmed dead code today (`cse.invalidate`
  has exactly one call site, inside `codegen_body`, and the derived stage
  flushes once after the whole output_groups loop) -- but if any future
  change flushes mid-stage, the fall-through re-emits a reshape of a
  Python-scoped variable holding the LAST loop iteration's tile: silent
  wrong data, the repro_looped_external_contiguous family. Raise on
  stale-cache, and assert liveness on every deferred consumption, mirroring
  `_materialize`.
- Perf-only frontier trigger gap: `trunc/ceil/floor/round_to_int` with an
  8-bit target narrow without hitting the to_dtype-only frontier (no current
  decomp emits that combination).
- `_try_get_sub_parent_source_projections`: lane proofs run
  `statically_known_equals` twice per consumer; `parent_index` recomputed
  per consumer; `next((...), None)` used for side effects.
- `projection_tail_source_audit`'s `has_r0_mask` flag is textual and matches
  the constant-true mask line; weak metric.

## Notable non-findings (attacked and held)

- must_forward compile error unreachable on legal schedules (four structural
  reasons: shared pending body when internal sources exist, producer
  ordering after the last DisableReduction flush, `disable_multi_kernel`
  hard-set, no codegen_body between output groups).
- Guard semantics: where-materialization carries the consumer's own fill,
  uncached per consumer; guarded->unguarded impossible by construction. Note
  low live exposure: admitted masked consumers today are fill=0 selects;
  differing-fill consumers break the lane proof and decline.
- Exact-access reconstruction: zero misses; both phases normalize through the
  shared `_RecordLoadStoreInner._normalize` on the same `_body` objects, so
  no cross-phase window exists.
- Bitwise-fuzz caveats for future work (both baseline-side hazards): (1)
  reference paths realize intermediates (bf16) where staged kernels keep
  fp32, so fuzz cases that shift fusion boundaries can see benign one-step
  fp8 divergence; (2) nested-off multi-kernel baselines recompute shared
  values (e.g. amax feeding frexp) on different precision paths across
  kernels and can disagree with themselves near pow2 boundaries -- the
  staged kernel was internally exact in that case. Neither is a projection
  bug; both can misfire a differential oracle.
- CD tuning empirics: candidate R0_BLOCKs are pow2-only and hard-capped at
  the extent for pow2 rnumel (435 candidates observed, none over); for
  non-pow2 rnumel CD DID benchmark R0_BLOCK > rnumel -- the masked-oversize
  case is real under CD, the gate correctly picks the frontier there, and it
  held bitwise. `triton.multi_kernel` has no staged interaction surface
  (staged kernels emit one def, no MultiKernelCall).
- frexp tuple wrapping verified live: both elements wrapped individually;
  the narrowed exponent hits the frontier, the mantissa defers to its
  lane-width consumer. Coverage nuance: the MXFP6 (4,3) spelling constructs
  no projection instance at all (INTERLEAVED-only, no BROADCAST consumer),
  so it does not exercise the view/split caches; the multi-output cache
  sharing evidence is the factor-4 fp8 battery cases.

## Bottom line

The "8 bit thing" is not a correctness hazard: it held under direct attack,
its gates rest on verified invariants, and misprediction is perf-only by
construction. The layers themselves are in strong shape -- the one proven
miscompile is in the replay's conflict resolution, has a one-line proven fix,
and was masked by a too-narrow recorded validation. Fix P0, close the three
P1 items, fold the new batteries in, and this is ready to become the F1/F2
PRs in the agreed order.

## Resolution (2026-08-22)

The current canonical patches supersede the implementation reviewed above.

- P0 is fixed. Upper-stack pass-boundary suppression now considers only
  `INTERLEAVED` and `CONTIGUOUS` projections. The original SwiGLU+GELU repro is
  bitwise correct, a checked-in looped regression covers the topology, and the
  full upper nested suite passes 447 tests with 13 skips and the two known
  `kernel_num_gb` preload cases deselected.
- P1.1 is fixed for the concrete gap: `store_reduction` is an explicit
  projection barrier with a behavior test. The scalar-by-default OpsHandler
  contract remains the classification policy rather than an exhaustive
  duplicated operation list.
- P1.3 is fixed. Non-`None` store modes are not recorded as forwardable exact
  values, with a focused atomic-store test.
- P1.2 is fixed. Standalone reduced side values are now planned as exact
  BROADCAST projections; shifted source/consumer accesses decline. Generated
  standalone factor-2 kernels remain byte-identical to the lower baseline.
- The 8-bit tail frontier was removed rather than expanded. Full blocks,
  oversized fixed blocks, static tails, and dynamic R keep group scalar work
  delayed. A CUDA-only staged sub-parent heuristic selects the measured
  Blackwell launch shapes; plain nested reductions and non-CUDA backends are
  unchanged.

Current hashes are recorded in `domain_projection_work/README.md` and
`domain_projection_work/UPPER_STACK_REPLAY.md`.
