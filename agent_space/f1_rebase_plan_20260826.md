# F1 indexed-forwarding rebase plan (2026-08-26)

Scope: rebase the "F1 indexed forwarding" design onto the current staged-reduction
split. This is a semantic conflict map plus a staged implementation plan; no code
was changed while producing it.

- TARGET STATE: `agent_space/pr191775_followups_wt` (uncommitted prerequisite +
  MXFP6 layers on `6b8ef64bd37`). All file:line references below are into this
  worktree unless marked OLD.
- OLD MATERIAL: `agent_space/f1f2_redesign/f1f2_redesigned_full.patch` (F1+F2
  combined, built on `931151d8c53`; phase2/3a/4 checkpoints are superseded by the
  full patch) and the applied reference tree `agent_space/f1f2_redesign_wt`.
- CONTRACT: `agent_space/sub_parent_indexed_forwarding_followup.md` (required
  properties 1-5, acceptance tests, deletion endpoint), `agent_space/FOLLOWUPS.md`
  item 1 and "Make staged plans authoritative for parent-to-grouped dependencies",
  `agent_space/pr191775_nested_equivalence_paths.md` (Path B),
  `agent_space/derived_domain_projection_proposals.md` (proposal E row of the
  status table; Iteration 8 R1-R5 constrain shape, not scope).

F2 (lazy derived-domain projection) is explicitly NOT planned here; its interface
points are marked in section 7.

---

## ROUND STATUS: COMPLETE (2026-08-27)

All four phases are landed on the fork (`agent_space/pr191775_followups_wt`,
uncommitted) and conformance-reviewed, with every acceptance battery green.
End state: 32 flagship kernels byte-identical through every phase; ONE fusion
prover (membership-only: strict dependency match or exact plan-pair
membership, `_fusable_read_after_index_equivalence` and the legacy branch
deleted, `_fusable_read_after_broadcast` relocated verbatim into planning);
per-read keyed source resolution in codegen (`_IndexedProjectedValueStore`,
per-key mask+fill guards, planner-recorded lanes, store-cache bypass for
planned misses); all name-based machinery deleted (three stage views,
latest-value-by-name tracking, family remapped_values/resolve_load,
masked-forward name sets); parent-to-grouped relations recorded as
PARENT_TO_GROUPED plan records spanning outer + grouped writers. Suites at
close: 150/441 plus the nested decline matrix, the 2b codegen-time and Phase-4
fusion-time mutation batteries, the P0 guard battery, shadow/agreement
sweeps, and differential fuzz (62 cases, max_abs_error 0.0). Realized
accounting: roughly +37 (P4) - 190 (P3) + 55 (2b) + 180 (2a) + 110 (P1) =
about +190 production LOC net-add against the pre-round tree, per the
section-5 ruling (mechanism-inventory reduction was the criterion; the number
differs from section 5's original +95 estimate mainly via the shadow-mode
commit and the Phase 4 builder).

Carried follow-ups (the round's accumulated list; none block shipping):

1. SHIP: organize the fork's uncommitted tree into the ruled shipping shape
   (phases 1-3 as one ghstack stack reviewed together; Phase 4 may ship
   independently). User-driven per AI policy: the user reads and submits.
2. Phase 4 optional hardening: replace the relocated flattened-normalize
   proof with an explicit parent/grouped frame proof (Path B bullet 2), as a
   separate reviewable commit.
3. F2 rebase preconditions (risk 7): clear `_materialized` alongside
   `cse.invalidate` if F2 moves materialization across a flush (risk 3
   standing requirement); redesign delta 4's loud-stale-cache raise belongs
   to F2's `_view`/`_split` caches; the two F2-only tests port with F2; keep
   the `resolve_source`/`materialize_source` seam.
4. Proof unification TODO (scheduler.py:1665-1669): make each access try
   every projection kind, removing the planner's name-partition proof
   selection.
5. Standing caveats, no action: reshape-family regime dependence (risk 8 --
   records load-bearing under loop_ordering_after_fusion=False; keep both
   pins); the value-collision nuance (fusion-accept + codegen-loud under
   artificial desync, Phase 4 status); the stricter dual-writer corner
   (a name written by producer and another staged node now declines);
   optional one-line comment at the `_materialize` memo site naming the
   flush-coupling requirement.
6. Doc sync when shipping: `agent_space/FOLLOWUPS.md` item 1 and the "Make
   staged plans authoritative" section are now satisfied; the contract doc's
   deletion endpoint (`sub_parent_indexed_forwarding_followup.md`) is fully
   met -- update those documents alongside the submission.

---

## 1. Already landed in the current split

The current split adopted most of old F1's planner side. Do not re-port these.

1. **Exact-dependency plan records.**
   `ProjectedSourceAccess` (scheduler.py:2100-2119, sources/consumers/layout with
   a one-buffer-name invariant), `MemoryDepMatch` (2123-2127), and
   `StagedReductionPlan.projected_access_pairs()` (2242-2248).
   Delta vs OLD: consumers are plain `MemoryDep`s; OLD wrapped them in
   `ProjectedConsumerAccess(access, lane)` and recorded the proven lane.

2. **Three-layout projection vocabulary and builders.**
   `SubParentSourceLayout.INTERLEAVED/BROADCAST/IDENTITY` (596-606).
   `_try_get_sub_parent_source_projections` (1118-1258; lane proof at 1229-1249,
   X/R-boundary handled at fusion time), `_sub_parent_internal_projections`
   (1285-1340, output-group-aware, cross-group trailing broadcast),
   `_sub_parent_broadcast_projections` (1390-1431 via
   `_sub_parent_consumer_is_trailing_broadcast` 1343-1387). Composed into the
   stage in `sub_parent_epilogue_plan` (728-777) and
   `_plan_nested_sub_parent_stage` (1645-1719).

3. **Plan-pair proving in fusion.**
   `_prove_staged_fusion_dependencies` (9419-9553): X/R-boundary validation for
   INTERLEAVED projections (9428-9482), mutation renames applied only at match
   time (9487-9496, contract property 1), multi-write decline (9527-9531),
   sub-parent reads require exact plan membership plus the
   `_memory_dep_supports_index_equivalence` safety gate (9541-9547).
   Staged scoring `_score_staged_fusion_memory_for_can_fuse` (10351-10388) and
   `_can_fuse_vertical_impl` (10015+) consume the `MemoryDepMatch` tuple.
   Four-path staged recognition in `_can_fuse` (9858-9905) with plan presence
   freezing generic loop rewrites (9915-9967).

4. **mask_vars_for_shape-style mask assignment.**
   `_DerivedIterationFamily.mask_vars_for_shape` (simd.py:1594-1618) and
   `set_value_masks` (1620-1633), applied on all three materialization paths of
   `materialize_value_at_sub_parent_resolution` (2092, 2105, 2128). This is the
   redesign's "masks assigned once, at materialization, by target shape" rule.

5. **Derived (not planned) `must_forward`** (OLD redesign delta 1): already
   landed as `must_forward = name in self._kernel.store_buffer_names`
   (simd.py:2404) with loud raises at 2411-2414 and 2425-2428.

6. **Internal-source deferral machinery.** `required_post_reduction_index`
   (StagedReductionPlan 2229-2230, validated 2237-2240), computed by
   `_order_sub_parent_parent_nodes` (785-874), consumed by
   `generate_node_schedule` (simd.py:2799-2806). Postdates the OLD patch (which
   had a `deferred_parent_start` plan field); looped internal sources scheduled
   into the final reduction pass are new structure.

7. **Plan rebuild after fusion + plan-loss guards.** Codegen replans from
   post-fusion nodes: `_find_sub_parent_epilogue_plan` (simd.py:2785-2797),
   `_sub_parent_epilogue_decision` (2689-2734), reduction-reduction plan-loss
   guard (2531-2539), FusedStagedReduction consumer guard (2650-2652).

Not landed (remaining scope, section 2): everything on the codegen resolution
side, per-key mask/fill, the three name views' deletion, and parent-to-grouped
plan records.

---

## 2. Remaining scope

### (a) Projection-aware load lookup keyed by full load identity

Replace name-keyed lookup with a key of (buffer name = current temporal/mutation
version, exact normalized access, mask, fill value), and resolve each consumer
read from its own planned relation.

- OLD hunks: simd.py `@@ -1514,6 +1517,78` (`_AccessGuard`, `_SourceAccessKey`,
  `_ProjectedLoad`, `_ResolvedProjectedSource`, `_access_guard`,
  `_logical_memory_access`); `@@ -2369,66 +2944,219` (`_IndexedProjectedValueStore`
  with `_record`, `store`, `store_reduction`, `record_store` (skips non-None
  modes), `_materialize` memoized per key with `cse.contains_value` liveness,
  `materialize_projections`, `get_projected_load`, `resolve_source` with
  unguarded-source fallback, `must_forward` loud raise, `materialize_source`
  with planner lane selection, `_apply_consumer_guard`).
- CURRENT targets: `_SubParentSourceLoadResolver` (simd.py:2360-2434) is the code
  it replaces; `_PointwiseRemapHandler.load/store` (2327-2357) is the consumer
  path it rewires; `_DerivedIterationFamily.remapped_values` TODO names this
  exact work (1542-1544, "#188180-style indexed projection cache").

### (b) Removal of latest-value-by-name tracking

- OLD: same store hunk; the name dict is gone, `cse.store_cache.get(name)`
  fallback is gone, planned-consumer misses go through
  `_load_without_store_forwarding` instead.
- CURRENT: `self._values: dict[str, TritonCSEVariable]` (simd.py:2386), the
  capture `self._values[name] = value` gated on the name sets and no load mask
  (2388-2394), and the name-keyed `store_cache` fallback (2406-2410).

### (c) Unified standalone/nested source resolution path

- OLD hunks: nested threading `@@ -3293,21 +4023,37` and `@@ -3333,32 +4079,38`
  (one `_IndexedProjectedValueStore` handed the full `source_projections`;
  eager `materialize_projections` of INTERLEAVED); standalone threading
  `@@ -3669,9 +4421,15` (internal_source_names derived from INTERLEAVED
  projection sources) and `@@ -3739,23 +4497,27` (same store class, projections
  passed whole, `materialize_projections` restricted to internal sources).
- CURRENT asymmetries to erase:
  - nested resolver gets `broadcast_source_names` (simd.py:3364-3372); the
    standalone resolver does not (3795-3802) -- standalone BROADCAST reads go
    through ordinary CSE/store-cache today;
  - nested `masked_forward_names = forwarded_store_names` only (3440);
    standalone `masked_forward_names = forwarded_store_names |
    internal_source_names` (3804);
  - standalone eager materialize loop by name (3809-3811) vs nested none.
  The warning in the contract doc stands: these distinctions are behavior, not
  accident. Unification must preserve them by construction (per-key resolution
  makes nested's reduced BROADCAST activation and standalone's ordinary
  broadcasting the same code answering different plans), not by dropping them.

### (d) Delete the three name-based compatibility properties

- CURRENT: `SubParentEpilogueStage.source_layouts` (scheduler.py:2168-2178),
  `broadcast_source_names` (2180-2188), `internal_dependency_names` (2190-2198).
  Consumers: simd.py 3369, 3371, 3422-3424 (nested); 3735, 3803 (standalone).
- OLD: the stage carried only `source_projections` (patch hunk
  `@@ -1891,16 +1986,28`); the properties never existed there (they are the
  compatibility layer the split added on top).

### (e) Parent-to-grouped relations as plan records; delete the legacy branch

- CURRENT: legacy branch in `_prove_staged_fusion_dependencies`
  (scheduler.py:9548-9552) guarded by the ownership classification
  (9522-9523, 9536-9547) with the TODO at 9538-9540; helpers
  `_fusable_read_after_index_equivalence` (10199-10208, only callers: this
  branch and tests) and `_fusable_read_after_broadcast` (10227-10282, only
  caller: that helper). Ownership pin:
  `test_nested_dependency_matches_scope_index_equivalence`
  (test_inductor_scheduler.py:746).
- OLD hunks: none. The OLD patch predates this machinery entirely; this scope
  item is specified only by FOLLOWUPS.md ("Make staged plans authoritative...")
  and pr191775_nested_equivalence_paths.md Path B. FOLLOWUPS (newer, reconciled
  2026-08-26) says to move the existing equivalence proof into planning
  verbatim; the Path B write-up asks for an explicit parent/grouped frame proof.
  Plan below follows FOLLOWUPS (verbatim move) and flags the frame-strictness
  upgrade as optional hardening, since tightening the proof while relocating it
  would conflate two behavior changes.
- REACHABILITY CORRECTION (2026-08-26, found during Phase 4 implementation):
  the legacy branch's acceptance set was not limited to parent-to-grouped --
  it also covered GROUPED-INTERNAL relations (earlier grouped-stage writes
  read by later grouped consumers), because append steps make the whole
  staged node the producer, so `producer.read_writes.writes` spans grouped
  writers. The relation records must span outer + grouped-stage writers; a
  parent-only write table regresses 8 suite tests. See the Phase 4 status
  block for the faithfulness argument.

### (f) Masked forwarding: name sets to per-key mask+fill

- OLD: guard captured at record time (`_access_guard`: `str(kernel._load_mask)`
  plus `kernel._load_other`), part of `_SourceAccessKey`; `resolve_source`
  accepts an unguarded source for a guarded consumer; `_apply_consumer_guard`
  re-applies the consumer predicate via `tl.full` + `where`, asserting on any
  other guard transition; masked loads are still never recorded (load() records
  only when `_load_mask is None`) -- the guard key exists for masked *stores*.
- CURRENT: `masked_forward_names` parameter and gate in
  `_PointwiseRemapHandler` (2317, 2325, 2329-2332 with the TODO "Preserve
  mask/fill in the indexed load cache instead"), populated at 3440 (nested) and
  3804 (standalone); resolver load gate at 2390.
- Pinning case: `test_standalone_sub_parent_mismatched_masked_source`
  (test_nested_reduction.py:2232) -- two pads of the same input with different
  fill values (0.0 vs 1.0). Under per-key forwarding these are distinct keys by
  fill; under name keying they would collide, which is why the current code
  refuses to capture masked loads at all.

---

## 3. Conflict map (old-patch region -> classification)

| OLD region (f1f2_redesigned_full.patch) | Classification | Notes / current target |
|---|---|---|
| scheduler.py: BROADCAST/IDENTITY enum members (`@@ -609`) | dead (landed) | scheduler.py:596-606 |
| scheduler.py: `broadcast_source_names` computation + `parent_source_names -=` (`@@ -703`, `@@ -719`) | dead (landed) | scheduler.py:700-727 |
| scheduler.py: `_try_get_sub_parent_source_projections` (`@@ -1040..-1154`) | landed, two deltas apply conceptually | current records RAW deps (1183, 1250) where OLD recorded `dep.normalize()`; current computes `lane_value` (1235-1242) but discards it. Phase 1 restores the lane on the record; normalization moves to codegen-store construction (see risks) |
| scheduler.py: `_sub_parent_internal_projections` (`@@ -1187`) | superseded | current version (1285-1340) is output-group-aware with cross-group trailing-broadcast consumers; OLD flat write==read version must not be restored |
| scheduler.py: `_sub_parent_broadcast_projections` (`@@ +1233` region) | superseded | current (1390-1431) uses the more general trailing-broadcast frame proof (1343-1387) instead of OLD's normalize-equality |
| scheduler.py: nested `_plan_nested_sub_parent_stage` changes (`@@ -1445..-1503`) | dead (landed) | scheduler.py:1603-1721; the name-partition TODO at 1661-1662 is phase 3 cleanup |
| scheduler.py: `ProjectedConsumerAccess` + stage field swap (`@@ -1891`) | applies conceptually | consumer lane wrapper is phase 1; stage field deletion is phase 3. Conflict: OLD `output_groups` were raw `(int, tuple)` pairs, current is `SubParentOutputGroup` (2140-2144) -- OLD tuple unpacking (`for output_lanes, stage_nodes in ...`) must be rewritten |
| scheduler.py: plan-source-names helper (`@@ -9015`) | dead (landed differently) | superseded by `projected_access_pairs` + per-site projection walks |
| simd.py: `_AccessGuard`/`_SourceAccessKey`/`_ProjectedLoad`/`_ResolvedProjectedSource`/`_logical_memory_access` (`@@ -1514`) | applies conceptually | phase 1 core; insert near `RemappedRangeValue` (simd.py:1516) |
| simd.py: delete `remapped_values` + `resolve_load` (`@@ -1538`, `@@ -1554`) | applies conceptually | phase 2/3; ALSO delete now-orphaned `lane_index_subs`/`lane_source_sizes` family fields (1548-1551) which the OLD patch left set-but-unread (f1f2_redesign_wt simd.py:1616-1619, no readers) |
| simd.py: name-free `materialize_value_at_sub_parent_resolution` + `broadcast_group_value_to_lanes` (`@@ -2060`, `@@ -2123`) | applies conceptually, hand-merge | current 2059-2131 gained `allow_reduced_broadcast`, bounds-carrying newvars (2124), and the rank-1/rank-2 reshape branches; keep those, adopt OLD's return-the-value form and layout dispatch (reference: f1f2_redesign_wt simd.py:2104-2161) |
| simd.py: `_DerivedValueDomain`/`_DeferredDerivedValue`/`_MaterializedProjectionCallback`/`_DerivedDomainProjection` (`@@ -2293`, +481 lines) | OUT OF SCOPE (F2) | do not port; section 7 marks the seams |
| simd.py: `_PointwiseRemapHandler` rewrite (`@@ -2309`) | applies conceptually minus F2 | keep: per-consumer resolve, IDENTITY fast-path, `materialize_source(required=must_forward)`, `_load_without_store_forwarding`, `record_store` on store. Strip: `_default` barrier hook, `_DeferredDerivedValue` returns, `defer_interleaved_projection` |
| simd.py: `_IndexedProjectedValueStore` (`@@ -2369`) | applies conceptually | phase 2 core; constructor consumes `stage.source_projections` directly (kills the three views); keep `can_defer_projection` (used by `_apply_consumer_guard`), it is also the F2 seam |
| simd.py: nested threading (`@@ -3272..-3372`) | applies conceptually, conflicts | current 3352-3441. OLD also wrapped the remapped-pointwise + grouped schedule in `set_ops_handler(projected_values)` so grouped-stage loads/stores record into the keyed store -- required, port it. Drop `staged_reduction` kwarg (F2) |
| simd.py: standalone threading (`@@ -3669..-3769`) | applies conceptually, conflicts | current 3719-3831. OLD `deferred_parent_start` logic is dead; integrate `materialize_projections` with the current `required_post_reduction_index` schedule and the flush choreography at 3805-3811 (see risk 4) |
| triton.py: `staged_reduction` kwarg + inductor_meta (`@@ -3296`, `@@ -7140`) | OUT OF SCOPE (F2 perf) | heuristics/reduction.py hunks likewise |
| triton.py: `emit_reshape_preserving_dtype` (`@@ -6222`) | OUT OF SCOPE (F2 view emission) | current `_emit_recursive_split`/`emit_split_via_reshape` (triton.py:6367-6418) already cover F1's factor-2/4 splits incl. fp8 bitcasting; no F1 change needed in triton.py |
| tests: `test_projected_access_*`, `test_guarded_projection_*`, `test_required_projected_access_rejects_exact_miss`, `test_projected_access_miss_bypasses_name_store_cache`, `..._does_not_forward_atomic_store`, `..._rejects_expired_value`, `test_projection_materializes_store_reduction`, `..._masked_callback_body`, `test_standalone_reduced_source_uses_exact_broadcast_projection` | applies conceptually | port with phases 1-3; rebase onto current test helpers (test_nested_reduction.py was deduplicated in `94367fffd77`) |
| tests: `test_projection_op_classification_is_complete`, float8 view, looped-deferral, `staged_reduction_inner_config`, NVFP4 swizzled-scale F2 assertions | OUT OF SCOPE (F2) | keep current kernel-form pins unchanged |
| redesign deltas (README): 1 derived must_forward | dead (landed, simd.py:2404) | |
| redesign delta 2 (no `_copy_masks`) | dead (landed as `set_value_masks` on all paths) | F2's remaining mask question does not exist in F1 |
| redesign delta 3 (fail-closed op partition) | OUT OF SCOPE (F2) | |
| redesign delta 4 (loud stale caches) | applies conceptually | phase 2 `_materialize` memoization must raise on a dead cached value rather than re-emitting (see risk 4) |

---

## 4. Staged plan

Ordering constraint honored: per-read resolution needs frame matching first, so
phase 1 is exactly that design plus its data model, landed with no behavior
change. Phase 4 is order-independent of 2-3 (fusion-side only) but sequenced
after phase 1 so its records use the final consumer type.

Shipping shape (review ruling 2026-08-26): phases 1-3 ship as a single ghstack
stack reviewed together -- phase 1 is ~110 lines with zero callers and would
draw dead-code objections as a standalone PR; phase 4 may ship independently.

### Prerequisites and inherited tripwires

A P0 guard-test battery is being written on the same fork and lands BEFORE
phase 1: extra-axis trailing-broadcast rejection, the TMP and sync-mode gates
in `_memory_dep_supports_index_equivalence`, a same-group transposed IDENTITY
read, a Path-4 append hostile read, plus positive controls on three reject
tests. Two obligations follow:

- Phase 1's `ProjectedConsumerAccess` wrapper must adapt any of those fixtures
  that construct `ProjectedSourceAccess` with bare `MemoryDep` consumers.
- The four guards join the mutation battery as inherited tripwires. They are a
  distinct class from the record-removal mutations: the guards pin the safety
  gates, the record-removal mutations pin the records. Both classes must stay
  green through every phase.

### Phase 1 -- Access identity: frame matching, guards, and consumer lanes (no behavior change)

**STATUS: DONE (2026-08-26, conformance-reviewed).** Approach A implemented
(`_logical_memory_access` simd.py:1560, data model 1518-1557, zero production
callers); `ProjectedConsumerAccess` at scheduler.py:2105 with lane recorded at
1250 and `.access` threaded at 2260/9490; agreement sweep
(test_nested_reduction.py:3147, test 3770: 4 forms x persistent/looped) ran
with zero drift and all legs nonzero, so Approach B is not needed; 32 captured
kernel files byte-identical (agent_space/p1_kernels_before|after); the one P0
fixture constructing `ProjectedSourceAccess` adapted
(test_inductor_scheduler.py:831).

The design problem (contract doc, "Main design problem"): planner records are
node-loop-frame `MemoryDep`s; codegen load() receives `(name, index)` already
substituted into kernel-tree frames by `sn._body(iter_vars)`. Two candidate
matching designs:

- **Approach A (OLD's, recommended): LoopBody-identity recovery.** At each
  load/store, rebuild the node-frame dep from the live interpretation state:
  `V.interpreter.current_node` gives the FX `call_method` node, whose index arg
  names `body.indexing_exprs[index_name]`; `V.kernel.current_node._body`
  supplies `var_ranges`; `.normalize()` canonicalizes. Match against plan
  records normalized the same way at store construction. Both sides read the
  same post-`merge_loops` body the rebuilt plan was derived from, so equality
  is exact-by-construction, not an equivalence proof. Proven in
  f1f2_redesign_wt (`test_projected_access_uses_loop_body_identity`).
  Cost: reaches through `V.interpreter` (fragile if emission ever leaves FX
  interpretation) and asserts on FX node shape; every emission site that should
  record (parent schedule body, grouped reduction body, remapped pointwise) runs
  under LoopBody interpretation with `set_current_node` today
  (simd.py:3374, 3615-3620, 3696-3697), so coverage holds.
- **Approach B (fallback): expected-index precomputation.** For each epilogue
  node, before `sn._body(iter_vars)` runs, substitute the plan consumer dep's
  vars with the same `iter_vars` produced by
  `_map_iteration_values_to_node_sizes` (simd.py:3678-3681) and match the
  incoming kernel-frame `index` by sympy equality. No interpreter
  reach-through; risks spurious mismatches from simplification differences and
  from `MemoryDep.var_names` canonical ordering vs body `var_ranges` ordering.
  A spurious mismatch on a must-forward value is a loud compile error (safe,
  but a robustness regression).
- Rejected variant: per-name load ordinals (k-th load of buffer in body maps to
  k-th read dep). Dead on arrival: `read_writes` dedups identical deps, so
  ordinals diverge from FX load order.

Deliverables:

1. `ProjectedConsumerAccess(access: MemoryDep, lane: int | None)` in
   scheduler.py; `_try_get_sub_parent_source_projections` records the
   `lane_value` it already computes (1235-1242); internal/broadcast builders
   record `lane=None`. `projected_access_pairs()` and
   `_prove_staged_fusion_dependencies` keep reading `.access` (raw deps,
   fusion contract untouched). Adapt any P0 guard-battery fixtures that
   construct `ProjectedSourceAccess` with bare `MemoryDep` consumers (see
   Prerequisites note above).
2. simd.py data model: `_AccessGuard`, `_SourceAccessKey`, `_ProjectedLoad`,
   `_ResolvedProjectedSource`, `_access_guard`, `_logical_memory_access`
   (Approach A). No production caller yet.
3. Ported unit tests: `test_projected_access_uses_loop_body_identity`,
   `test_projected_access_records_exact_index`,
   `test_projected_access_guard_includes_mask_and_fill`; plus a new
   plan-vs-body agreement sweep: for every consumer record produced while
   compiling the NVFP4/MXFP4/MXFP6/internal-source suites, assert
   `_logical_memory_access` at emission equals the normalized record
   (test-only instrumentation, keeps Approach A honest before phase 2 relies
   on it).

Acceptance: all suites green; generated kernels byte-identical (no production
call sites); the agreement sweep passes on standalone and nested forms, looped
and persistent.

### Phase 2 -- Keyed store replaces the name resolver (scope a, b, f; most of c)

Two LANDABLE commits (review ruling 2026-08-26). Load-side and store-side
forwarding still move together -- what is split is authority, not mechanism,
so no precedence ambiguity arises: in shadow mode only the name-based system
answers.

**Phase 2a (landable, shadow mode).**

**STATUS: DONE (2026-08-26, conformance-reviewed).** Zero disagreements under
the refined outcome contract across both suites (134/417, one pre-existing
AOTI env error), the fuzz matrix (p2a_fuzz_results.json, max_abs_error 0.0),
and the 7-leg x persistent/looped stats sweep (p2a_shadow_stats.py; e.g.
nvfp4_standalone persistent 2 records/2 forwards/2 fallthroughs vs looped
2/0/4 -- the loop-flush reload story -- and layernorm 0/0/0 because no
sub-parent stage means no store is constructed). 32 kernel files
byte-identical (p2a_kernels_before|after). scheduler.py untouched.

Build the full `_IndexedProjectedValueStore` and wire it into both threading
paths ALONGSIDE `_SubParentSourceLoadResolver` (as the resolver's inner, so
recording sees every parent-stage access exactly once); it records and
resolves, but the name path remains authoritative and is the only emitter.
The shadow contract compares at the ACTUAL OUTCOME, not the handler-level
decision (review ruling 2026-08-26, replacing the original "declines -> keyed
must miss" wording, which was wrong for standalone BROADCAST consumers: they
are name-forwarded by CSEProxy's store_cache BELOW the handler, so the
handler-level fallthrough returns a cached value without emitting):

- name path forwards at handler level -> keyed store must resolve the same
  source CSE value by object identity, the same layout as the name path's own
  classification, and a planner lane equal to the `interleaved_sub_parent_lane`
  the family computes from the kernel-frame index (validating phase 1 lanes);
- fallthrough returns the name-keyed store-cache value (standalone reduced
  broadcasts, exact-index reads of in-kernel writes) -> keyed store must
  resolve that SAME value;
- genuine reload (a load actually emitted) -> keyed store must miss, so any
  case where the 2b flip would change emitted code fails loudly pre-flip;
- unplanned reads are exempt (the risk-4 exact-index coupling, documented at
  the check site).

Agreement is at SOURCE-value level: 2a performs no materialization, so
projected-value equality is validated by 2b's byte-identity acceptance, not
by the shadow. Kernels are byte-identical by construction. Loud shadow asserts
are acceptable to land because the staged path is opt-in
(`config.triton.nested_reduction`).

**Phase 2b (landable, small authority flip).** The consumer path switches to
the keyed store; the loud must-forward raise activates with authority; shadow
asserts are removed; the name machinery stays in place but unread (deleted in
phase 3, keeping this commit small). Validated by every suite run accumulated
under shadow in 2a.

**STATUS: DONE (2026-08-26).** Implementation landed and conformance-reviewed;
the record-removal mutation battery closed the gate the same day: 6 permanent
tests, 12 cells (layout x persistent/looped x whole-record/consumer drops),
suites 144/431 green, production untouched, message-agnostic assertions, and
the core property held in every cell -- no silent name forwarding. Battery
outcome matrix: external INTERLEAVED drops degrade to real lane reloads
(persistent gains loads; looped already reloads), in-kernel INTERLEAVED drops
fail loudly in persistent mode (stage shape mismatch) and degrade to
barrier-guarded reloads in looped mode, in-kernel nested BROADCAST drops fail
compilation (the grouped scale exists only in registers), IDENTITY drops are
inert-and-correct by the exactness invariant. Three gate rulings:

- Gate ruling A (looped in-kernel drop is correct-by-machinery, VERIFIED
  CONCRETELY, not from byte-identity): re-derived the mutated looped MXFP6
  cell and inspected the kernel text (agent_space/verify_cell2_barrier.py,
  output agent_space/cell2_verify_out.txt): control 2 loads / 0 barriers,
  mutated 6 loads / 6 barriers, results bitwise equal, and `tl.debug_barrier()`
  immediately precedes each of the four lane re-reads of the kernel-written
  buffer (buf4, emitted as in_out_ptr1). Mechanism chain: the dropped record
  leaves `internal_source_names`, so the pre-epilogue flush runs;
  `cse.invalidate` moves the store_cache entry into `invalidated_stores`
  (common.py:2093-2097); the unplanned read misses store_cache and takes a
  real load; TritonKernel.load's read-after-own-store guard
  (triton.py:5062-5070, the "#1615 companion") emits the barrier; X-ownership
  (`xmask` on `x0` rows) means the program re-reads only rows it wrote. F1
  therefore DEGRADES TO THE STANDARD BARRIER-GUARDED RELOAD -- existing
  machinery, not luck.
- Gate ruling B (raise characterization): ACCEPTED as the designed backstop;
  do NOT build a "raise on unplanned access of a planned name" hardening --
  it would false-positive on legitimate exact-index reads that fusion admits
  without records (`fusable_read_and_write`, scheduler.py:9533-9534), which
  flow through the ordinary path by design (simd.py:2441-2445). The defense
  is two-layer and recorded in risk 4: width-MISMATCHED wrong forwards trip
  the block-shape invariant (shape_propagation.py:45-51) at compile time
  (in-kernel INTERLEAVED persistent, nested BROADCAST); width-COMPATIBLE
  forwards are benign by the exactness coupling (IDENTITY) or by genuine
  broadcast semantics (standalone reduced scale), never silently wrong.
- Gate ruling C (battery scope precision): mutations arm only at codegen
  rebuild (`_run_with_codegen_plan_mutation` patches
  `sub_parent_epilogue_plan`/`plan_from_topology` behind a post-fusion arm,
  test_nested_reduction.py:3390-3436), so every cell tests CODEGEN desync
  while fusion always saw the unmutated plan. IDENTITY records being
  redundant THERE is expected: their authoritative role is FUSION LEGALITY
  (exact plan-pair membership in `_prove_staged_fusion_dependencies`,
  scheduler.py:9541-9547), pinned by the P0/prove tests. "IDENTITY record is
  redundant at codegen" must never be read as "deletable."

Original gate note (retained for history): 32 kernel
files byte-identical (p2b_kernels_before|after); fuzz 62 cases, 104 exact /
260 numeric, max_abs_error 0.0 vs the pre-2b base; suites 144/419 (same
pre-existing AOTI env error); scheduler.py untouched; the risk-3 shared
helper `load_without_store_forwarding` (common.py:2779) is a
behavior-identical extraction of `CSEProxy.load` (verified case-by-case:
TMP ordering, invalidated-store must_keep bookkeeping including the
`V.kernel` quirk, store-cache semantics for invalidated-then-restored names).
Three review rulings:

1. Stale-memo raise removed: ACCEPTED. The applied OLD tree has no such raise
   (its memo hit returns directly); the 2a raise was this plan's
   over-extension of redesign delta 4, whose loud-stale-cache requirement
   belongs to F2's `_view`/`_split` caches. `cse.contains_value` checks only
   `_cache`/`store_cache`/`reduction_cache` values, so INTERLEAVED lane
   newvars are untrackable and the raise fired on every INTERLEAVED memo
   reuse -- it was never a valid guard. Memo-validity analysis per layout:
   standalone BROADCAST materialization is a passthrough (`parent_dim == "1"`),
   so the memo IS the source object and the live-source gate covers it
   exactly (this closes the accumulator case: source and memo are one value,
   live in reduction_cache/store_cache, emitted in the same post-loop suffix);
   nested BROADCAST memos are distinct cache-tracked vars, INTERLEAVED memos
   are newvar tuples -- for both, the load-bearing guards are the one-region
   invariant (all `_materialize` calls sit between two `codegen_body` flushes
   with none in between: standalone sweep simd.py:4167-4175 + groups
   4176-4193, nested groups 3772-3789) and `_record`'s memo pop on
   re-recording. Structurally unreachable today; documented at the site
   (simd.py:2646-2650) and pinned by the updated
   `test_projected_access_rejects_expired_value`
   (dead-source-invalidates-memo). FAIL-CLOSED REQUIREMENT recorded in risks
   and the F2 interface: any future flush point between stage emissions must
   clear `_materialized` alongside `cse.invalidate`.
2. Nested eager-INTERLEAVED sweep not added: ACCEPTED, lazy placement
   supersedes the OLD eager design. The OLD sweep mirrored its base's eager
   name path; the current name path materializes lazily at first consumer
   load, byte-identity is the acceptance bar, and the OLD README itself
   flagged full laziness as a desirable simplification. The standalone
   internal-source sweep sits at today's exact spot.
3. `test_projection_materializes_store_reduction` and
   `test_projection_materializes_masked_callback_body`: confirmed F2-ONLY
   (both subclass `_DerivedDomainProjection` and drive `_DeferredDerivedValue`
   through `projection.apply`; f1f2_redesign_wt test_nested_reduction.py:
   3793-3867). Moved out of the 2b list to the F2 interface; substitutes in
   place: store_reduction recording pinned by the 2a unit tests and exercised
   end-to-end by the nested BROADCAST kernel-form suites; masked-consumer
   semantics pinned by the guarded-projection matrix.

2b INHERITED FROM 2a (review ruling 2026-08-26: deferral confirmed within the
2a/2b split, not scope-shaving -- porting the emission leg in 2a would have
required calling the name-registering `materialize_value_at_sub_parent_resolution`,
mutating `family.remapped_values` and poisoning the authoritative name path).
Exhaustive list so nothing silently drops:

- `_materialize`'s emission leg: writing `_materialized[key]` (the memo-read
  branch and its expired-within-stage raise, redesign delta 4, already exist
  at simd.py:2716-2734 but are unreachable because nothing writes the memo).
- `materialize_source` with `_select_lane` planner-lane selection.
- `_apply_consumer_guard` (mask+fill reapplication, unsupported-transition
  assert) and `can_defer_projection` (used by the guard; also the F2 seam).
- `materialize_projections` and the standalone internal-source-restricted
  sweep, replacing the name loop at simd.py:4162-4164. (Amended by 2b ruling
  2: NO nested eager-INTERLEAVED sweep -- nested stays lazy at the first
  consumer load, matching the current name path and preserving byte-identity;
  the OLD eager sweep is superseded.)
- Store constructor gains the `layout`/`sub_parent_family` args it needs to
  materialize (deliberately absent from the 2a ctor at simd.py:2639-2646).
- The must-forward raise inside `resolve_source` (2a returns None at
  simd.py:2763; `must_forward` classification exists at 2765-2770).
- Name-free `materialize_value_at_sub_parent_resolution` (deliverable 3).
- `_load_without_store_forwarding` (deliverable 4).
- Removal of the shadow scaffolding: both `_shadow_check_*` methods, the
  shadow block in `_PointwiseRemapHandler.load` (simd.py:2422-2439 including
  the pre-load `cached` capture), and the shadow-mode docstring.

Deliverables:

1. (2a) `_IndexedProjectedValueStore`, full mechanism, built alongside
   `_SubParentSourceLoadResolver` (simd.py:2360-2434), which it replaces as
   authority only in 2b: constructor takes `stage.source_projections`, builds
   `_source_layouts: dict[MemoryDep, layout]` and
   `_consumer_projections: dict[MemoryDep, _ProjectedLoad]` from records
   normalized once here (loud on conflicting duplicates); `_record` keyed by
   `_SourceAccessKey(access, guard)`; `load` records only unmasked loads;
   `store`/`store_reduction`/`record_store` record writes (skip non-None store
   modes: the atomic gate); `resolve_source` with the unguarded-source
   fallback and the derived `must_forward` classification (raise deferred to
   2b); the shadow-check entry point consumed by `_PointwiseRemapHandler`
   during 2a.
2. (2a) Both threading paths construct the store: nested (simd.py:3358-3441)
   additionally wraps the remapped-pointwise and grouped-schedule emission in
   the store handler (OLD `@@ -3333` hunk) so recording sees every access;
   standalone (3793-3828) likewise.
3. (2b) `_materialize` memoized per key, raising on a CSE-dead cached value
   (redesign delta 4); `materialize_source` selecting the planner lane;
   `_apply_consumer_guard` re-applying mask+fill, asserting on unsupported
   guard transitions; name-free `materialize_value_at_sub_parent_resolution`
   returning `RemappedRangeValue | None` with layout dispatch (IDENTITY
   passthrough, BROADCAST via split-out `broadcast_group_value_to_lanes`,
   INTERLEAVED split), keeping current bounds propagation and
   `set_value_masks` on all paths (merge f1f2_redesign_wt simd.py:2104-2161
   into current 2059-2131); `materialize_projections` driving the
   internal-source-restricted eager splits in the standalone path.
4. (2b) `_PointwiseRemapHandler` authority flip: `get_projected_load` ->
   `resolve_source` -> IDENTITY fast-path / `materialize_source(required=
   must_forward)`; planned-consumer misses fall to
   `_load_without_store_forwarding` (bypass the name-keyed
   `cse.store_cache`, preserving `invalidated_stores`/`must_keep_buffers`/
   indirect/num_load/op-trace behavior -- see risk 3 for the implementation
   constraint); unplanned reads keep the ordinary `self._inner.load` path.
   `store` records via `record_store`. The name-based read path (the
   `masked_forward_names` gate and the family/resolver consult) stops being
   read here; the machinery and its parameters are deleted in phase 3.

Acceptance:

- 2a: kernels byte-identical by construction; shadow agreement holds across
  the full nested-reduction, scheduler, and heuristics suites plus the
  differential-fuzz matrix (any disagreement is a phase 1/2a bug surfaced
  before authority ever flips); recording/guard-capture unit tests green.
- 2a inheritance from phase 1 (2026-08-26): port the `_record` /
  materialized-invalidation half of OLD `test_projected_access_records_exact_index`
  (re-recording a key pops its `_materialized` entry). Phase 1's landed version
  pinned `_SourceAccessKey` keying only, because the recording store did not
  exist yet. DISCHARGED in 2a: test_inductor_scheduler.py:742-756. 2a also
  landed ahead of the 2b list: `test_projected_access_does_not_forward_atomic_store`,
  `test_projected_access_rejects_expired_value` (including the stale
  memo raise), and a new `test_projected_store_records_masked_store_guard`.
- 2b: kernel forms byte-identical for: nested broadcast, standalone reduced
  broadcast, looped internal source (test_nested_reduction.py:1880-1959),
  NVFP4 persistent/looped, MXFP4, MXFP6 (4,3) + preshuffle
  (`test_mxfp6_internal_source_kernel_form`, 3512), indirect-mask cases
  (981, 998). Any intentional diff is individually reviewed.
- 2b: ported tests green: `test_required_projected_access_rejects_exact_miss`,
  `test_projected_access_miss_bypasses_name_store_cache`
  (atomic-store and expired-value tests already landed in 2a),
  `test_guarded_projection_applies_guard`,
  `test_standalone_reduced_source_uses_exact_broadcast_projection`.
  (`test_projection_materializes_store_reduction` and
  `test_projection_materializes_masked_callback_body` moved to the F2
  interface by 2b ruling 3 -- their bodies are pure F2 plumbing; F1-level
  coverage substituted by the 2a store_reduction/masked-store-guard units,
  the nested BROADCAST kernel-form suites, and the guarded-projection matrix.)
- 2b: contract acceptance rows: unowned read declines; multiple same-name writes
  decline; mismatched-fill pad case (2232) stays unfused/exact; masked
  odd-tail gather keeps the target-family mask.
- 2b: mutation coverage (FOLLOWUPS requirement): for each layout family
  (INTERLEAVED, BROADCAST, IDENTITY), a focused one-kernel test that
  surgically drops the record from the rebuilt plan (monkeypatch the plan
  builder) and asserts a loud compile error (must-forward) or lost fusion --
  never silent name forwarding. This is the direct test of "an unplanned read
  must fail loudly rather than inheriting the name's treatment". The inherited
  P0 guard battery (Prerequisites note) stays green alongside it.
  DISCHARGED (2026-08-26): 6 permanent tests / 12 cells at
  test_nested_reduction.py (helpers 3334-3418, driver `_run_dropped_record_case`
  4053); refinement vs the original wording -- the accepted loud/degraded
  outcomes per cell are recorded in the 2b STATUS matrix above ("fail loudly
  or lose fusion" holds for the register-only cases; external and flushed
  in-kernel sources legitimately degrade to real reloads, and IDENTITY drops
  are inert by exactness -- neither is silent WRONG forwarding, which no cell
  produced).
- 2a and 2b: differential fuzz: re-run the 62-case adversarial matrix
  (f1f2_redesign/adv_fuzz_local.py pattern) with base captures from the
  unmodified followups worktree; expect bitwise-identical. Run through
  run_wt.py (the semantic_fuzz self-overlay caveat from the OLD README).

### Phase 3 -- Net-delete sweep and single resolution path (scope d; rest of c)

**STATUS: DONE (2026-08-26, conformance-reviewed).** Realized accounting:
about -190 net production lines (measured -166 simd.py + roughly -29
scheduler.py; implementer-reported -189) vs the -225 estimate -- the gap is
the shared builder plus its hook scaffolding, accepted. 32 kernel files
byte-identical (p3_kernels_before|after, verified diff -r clean); suites
146/431 with the battery green against the deleted views; common.py and the
risk-4 do-not-touch list untouched (phase-3 delta on common.py is zero).
Grep-clean with two deliberate retentions, both verified correct: the keyed
store's dep-keyed `_source_layouts` dict (not a name view) and the
planner-local `broadcast_source_names` variables (scheduler.py:701/1683) --
those are the planner's proof-selection classification (which sources get the
trailing-broadcast vs lane-projection proof), consumed only at plan
construction, never by codegen; they were never on the deletion list. The
resurrection tripwire constructs a REAL stage with real records
(`test_sub_parent_stage_has_no_name_views`, test_inductor_scheduler.py:969-992),
so its AttributeError assertions are meaningful. The standalone
`internal_source_names` re-derivation (simd.py:3946-3958) is genuinely
set-equivalent to the old view computation by the `ProjectedSourceAccess`
one-buffer-name invariant (`__post_init__`, scheduler.py:2126-2131):
`{sources[0].name}` equals `{s.name for s in sources}` per projection.

RULING (deleted `make_sub_parent_family` extent assert): deletion is CLEAN,
no replacement one-liner required. The old assert guarded live codegen
computation -- pre-F1 the lane was DERIVED at codegen from the kernel-frame
index through `lane_index_subs`, so an unsound extent substitution meant a
silently wrong lane. Post-F1 nothing derives values from extent
substitutions (their only consumer was the deleted `resolve_load` path), so
the assert revalidated an input to deleted code. The properties it checked
retain equivalent-or-stronger loud guards: (i) lane consistency is proven
statically per consumer at plan time (scheduler.py:1235-1249), re-proven at
the codegen plan rebuild whose failure path is loud ("sub-parent reduction
plan was lost before codegen", simd.py:3447), and revalidated at use --
`materialize_source` raises on a missing lane and `_select_lane` over the
factor-length tuple raises "invalid lane" for any lane outside [0, factor);
(ii) extent divisibility/structure is re-proven by the same planner paths at
rebuild (standalone `try_get_sub_parent_extent_subs`; nested grouped
topology), and any residual kernel-tree-vs-layout mismatch fails at Triton
compile time (reshape totals / block-shape invariant -- the gate-ruling-B
backstop class), never as silent numerics. Razor applied: revalidation earns
its keep when its absence converts a planner bug into silent wrongness; here
every residual failure mode is loud and the guarded computation is gone.

Accepted deviations recorded below in deliverable 4; the name-partition TODO
re-scope at scheduler.py:1665-1669 is APPROVED as accurate (the partition is
the planner's input classification selecting which proof each source gets,
projections are its output; unifying the proofs is the removal path).

Deliverables:

1. Delete `source_layouts`, `broadcast_source_names`,
   `internal_dependency_names` (scheduler.py:2168-2198); grep-clean.
2. Standalone `internal_source_names` from INTERLEAVED projection sources
   intersected with parent-written buffer names (OLD `@@ -3669` shape),
   replacing the view-based computation (simd.py:3735-3739).
3. Delete the name machinery orphaned by the 2b flip:
   `_SubParentSourceLoadResolver` itself, the family name registration in
   `_PointwiseRemapHandler.store`, the `forwarded_store_names`/
   `masked_forward_names` parameters and call-site arguments,
   `_DerivedIterationFamily.remapped_values`, `resolve_load`
   (simd.py:1558-1579), and the orphaned `lane_index_subs`/`lane_source_sizes`
   fields plus their `make_sub_parent_family` population (verify no other
   readers; the planner keeps its own `interleaved_sub_parent_lane`).
4. Extract one shared builder for the store + emission loop so standalone and
   nested call the same function. (As landed, amended by phase-3 review:
   `_codegen_sub_parent_stage(kernel, stage, layout, emit_parent_stages)`
   takes the STAGE, not the plan -- plan-level concerns (schedules,
   `required_post_reduction_index`, flush placement) stay path-specific inside
   the single `emit_parent_stages` callable hook, and the nested no-stage case
   is handled outside the builder so its contract stays non-optional. Accepted
   as a better factoring than the original parenthetical.) The name-partition
   TODO was re-scoped, not resolved: scheduler.py:1665-1669 now states the
   partition is planner-internal proof selection and unifying the proofs is
   the removal path.

Acceptance: byte-identical kernels; all suites green; the deletion endpoint
list from the contract doc is fully satisfied except the legacy fusion branch
(phase 4).

### Phase 4 -- Parent-to-grouped plan records; delete the legacy prover (scope e)

Order-independent of phases 2-3; requires phase 1 only for the consumer type.

**STATUS: DONE (2026-08-27).** The fusion-time mutation battery closed the
gate (suites 150/441 green): broadcast-family record drop declines 1->2
kernels with surviving-relation precision; single-pair granularity pinned on
a two-group workload; the REQUIRED plan-gate pin landed with three arms
pinning the 2054-2055 link; the collision-free sub-parent INTERLEAVED drop
declines 1->3; and the nvfp4_nested BROADCAST collision cell documents
fusion-accept + codegen-loud via assertRaises, exactly per the recorded
caveat. Two closing rulings:

- Closing ruling A (reshape-family structural inertness): VERIFIED and
  ACCEPTED. The reshape-equal family produced no e2e decline because both
  canonicalization mechanisms of the `loop_ordering_after_fusion=True` regime
  make those reads raw-exact before membership ever runs: pre-merge,
  `fusable_read_and_write` normalizes only under that config (the gated
  branch and re-check); post-merge, re-extraction yields var-merged raw deps.
  Under `loop_ordering_after_fusion=False` -- the fbcode default -- neither
  runs, and the reshape-family records are CONTRACT-REQUIRED at the prove
  step. Pinned both ways:
  `test_reshape_equal_grouped_relation_requires_membership`
  (test_inductor_scheduler.py:1302, unit, LOAF=False patch, parametrized
  recorded/unrecorded) and
  `test_dropped_grouped_relation_reshape_family_is_inert`
  (test_nested_reduction.py:4380, e2e, flags any future load-bearing shift).
  DO NOT simplify the reshape family away based on OSS-regime inertness: the
  records are load-bearing in the LOAF=False regime (the same
  regime-dependence theme the prerequisite review hit). Also recorded as
  risk 8.
- Closing ruling B (comparison methodology): ACCEPTED. Bitwise atol=0 between
  two INDEPENDENT compiles is unsound (autotune config selection changes
  float contraction; ~1 ulp flake reproduced ~1/6 in the control arm alone).
  Both batteries share `_assert_battery_results_match`
  (test_nested_reduction.py:4170-4182): floats >=16 bits at the file's 1e-2
  convention (garbage-catching -- observed lost-forwarding garbage is
  magnitude ~192), while integers and sub-16-bit floats (packed quant bytes,
  fp8/e8m0 scales) stay EXACT, so the quant-critical outputs remain bitwise.
  The 2b retrofit did not weaken any cell: every cell's discriminators are
  structural (kernel counts, load-count deltas, loud raises); numeric
  equality is the correctness backstop. Recorded in section 7. The AOTI
  suite error did not reproduce on the battery box (clean 431 before and
  after) -- environmental, not stack-related.

Original gate note (retained for history): implementation landed and
conformance-reviewed (2026-08-26). 32 kernels
byte-identical (p4_kernels_before|after); nested suite baseline-identical
including the decline matrix; net +37 measured (scheduler +98/-69 incl. the
relocated 57-line broadcast prover and the ~48-line builder; simd +8 for the
ctor guard) vs the ~-25 estimate -- the estimate wrongly assumed the builder
would be absorbed by the relocation; ACCEPTED, the mechanism win is intact
(one prover, membership-only fusion, `_fusable_read_after_index_equivalence`
and the legacy branch deleted, `_fusable_read_after_broadcast` relocated
verbatim into `NestedReduction`).

SPEC CORRECTION (gap found in implementation): section (e)'s "parent-written
buffers" was too narrow -- the deleted legacy branch also accepted
GROUPED-INTERNAL relations (an earlier grouped-stage write, e.g. a reduced
scale, read by a later grouped consumer), reachable because append steps make
the whole staged node the producer. The builder's write table therefore spans
outer + grouped-stage writers (scheduler.py:1453-1457); 8 suite tests
regressed under parent-only and pass with the union. Faithfulness argument
verified: the static table is the union over every fusion step's producer
write set restricted to staged nodes (epilogue writes excluded -- they are
unread by `_sub_parent_epilogue_outputs_unread`), and per-step scoping is
restored by the prove step's `producer_output_names` filter (reads of
non-producer buffers are skipped before membership), so extra records are
inert. One STRICTER corner, accepted: a name written by both the producer and
another staged node now yields no record (builder table sees two writes,
scheduler.py:1465-1468) where the old producer-scoped table saw one --
conservative, unobserved in the suites.

CRITICAL-QUESTION RESOLUTION (ownership precedence after the classification
sets' deletion): SAFE, no production change required, battery dispatched
as-is with two scope additions. The verified three-legged argument:

1. Builder exclusion exists BY CONSTRUCTION: the consumer walk iterates
   `grouped_stage_nodes` = grouped nodes MINUS `sub_parent_nodes`
   (scheduler.py:2058-2062), at the SOLE call site (2078) of the SOLE
   `NestedReductionStage` constructor (2072); epilogue nodes never enter the
   walk, and the docstring names the contract. No in-builder re-filter is
   required (it would need the sub-parent set threaded through the signature
   to guard an argument the only caller already filters).
2. Classification exhaustiveness: the nested stage planner's
   `parent_access_names` spans ALL non-epilogue staged nodes including
   grouped writers (parent_nodes = all_nodes minus epilogue,
   scheduler.py:1656-1690), so every epilogue read of any staged-accessed
   name lands in `parent_source_names` or `broadcast_source_names` and must
   pass a frame proof; "independent" epilogue inputs are exactly the names no
   staged node touches, which can never be producer outputs in staged fusion.
3. Hard plan gate: a frame-proof failure with SUB_PARENT-classified nodes
   present returns None from `plan_from_topology` (2054-2055) -- no plan, no
   records, hard fusion decline. Therefore in every plan-exists scenario an
   epilogue read either has its own frame-proved sub-parent pair (the grouped
   pair adds nothing) or the plan does not exist; the collapsed membership
   cannot accept what the old ordering rejected.

NUANCE the exclusion cannot close (recorded for the battery): dependency
normalization makes value-equal COLLISIONS real (an epilogue trailing-broadcast
read of the scale normalizes identically to a grouped consumer's read), so
under ARTIFICIAL record desync -- a fusion-armed mutation dropping the
sub-parent record where a value-equal PARENT_TO_GROUPED pair exists -- fusion
accepts via the grouped pair; the loud failure then comes from codegen (the
unmutated rebuild, or 2b's backstops under full desync), not from the prove
step. Battery consequences: (i) sub-parent-record drop cells must use
collision-free workloads, or expect fusion-acceptance with the codegen-loud
backstop noted; (ii) REQUIRED addition, the plan-gate pin: a unit test that
`plan_from_topology` returns None when SUB_PARENT-classified nodes exist and
stage planning fails (pins the load-bearing 2054-2055 link; the
frame-proof-rejects half is already pinned at
test_sub_parent_broadcast_projection_frame_contract).

Standard rulings: (2) PARENT_TO_GROUPED as a fourth `SubParentSourceLayout`
member SATISFIES "dedicated relation kind" -- the intent was that the codegen
store never dispatches on a grouped relation, enforced loudly by the positive
allowlist in the store ctor (simd.py:2440-2447, covering "any future
fusion-only record kinds") plus structural separation (`grouped_relations`
on `NestedReductionStage`; the store is built from `stage.source_projections`
only); a separate type is not required. (4) The `mode_requires_synchronization`
and `_memory_dep_supports_index_equivalence` staticmethod conversions are
body-identical (the latter's `self.` call became `Scheduler.`); all call
sites verified (instance access to a staticmethod is unchanged; the builder
uses class access from classmethod context). (5) The membership test's
3-arms-to-2 subsumption is VALID at the prove-step level -- ownership no
longer exists there, so unowned == unrecorded and the both-owned arm's
strictness now lives in the planner, which is exactly what the plan-gate pin
covers; the five migrated equivalence assertions
(test_inductor_scheduler.py:495-499) preserve every accept/decline
expectation through the builder. Battery patch point APPROVED:
`_parent_to_grouped_relations` is the single fusion-armed seam, family split
by re-running `deps_match_normalized` per pair distinguishes reshape-equal
from broadcast relations, and kernel count is the observable.

Deliverables:

1. Planner: during nested planning (`NestedReduction.plan`,
   `plan_from_topology` via `_plan_append`, and the codegen rebuild), walk
   non-epilogue grouped-stage reads of STAGED-WRITTEN buffers (outer AND
   grouped writers -- corrected from "parent-written" by the spec-gap fix
   above) that are not exact matches and record each as a planner-owned
   relation by moving the existing proof
   (`_memory_dep_supports_index_equivalence` safety +
   `deps_match_normalized` or `_fusable_read_after_broadcast`) into record
   construction (as landed: `_parent_to_grouped_relations`,
   scheduler.py:1440-1487). Store on `NestedReductionStage` as
   `grouped_relations: tuple[ProjectedSourceAccess, ...]` with the
   `PARENT_TO_GROUPED` relation kind, which must never reach the codegen
   store's layout dispatch -- enforced by the store ctor's positive allowlist
   (ruling 2 above). Rebuild with the plan; never cache across loop mutation
   (Path B rule 4).
2. Fusion: `_prove_staged_fusion_dependencies` requires exact membership for
   nested-stage reads (same shape as the sub-parent branch at 9541-9547);
   delete the legacy branch (9548-9552), then
   `_fusable_read_after_index_equivalence` (10199-10208) and
   `_fusable_read_after_broadcast` (10227-10282) once test callers are
   migrated. The ownership classification collapses to: strict match, or
   exact plan membership (either record family), or decline.
3. Optional hardening (separate, reviewable commit inside the phase): replace
   the relocated flattened-normalize proof with an explicit parent/grouped
   frame proof per Path B bullet 2. Do not bundle with the relocation.

Acceptance (from pr191775_nested_equivalence_paths.md "Required tests"):
grouped-axis X and R nested reductions remain one staged kernel;
parent-to-grouped reshape and broadcast remain accepted; shifted, transposed,
regrouped, indirect, multiwrite, synchronized decline; empty and nonempty match
sets preserve scoring and freeze loop rewrites (existing tests at
test_inductor_scheduler.py:714, 746 rewritten to the two-path contract, 806+);
NVFP4 kernel forms unchanged. Mutation coverage per relation family
(reshape-equal, consumer-broadcast): removing the planned record makes a
focused one-kernel test fail (kernel count or fusion metric), not silently
reduce fusion.

---

## 5. Net-delete accounting (production LOC, tests excluded; estimates)

| Phase | Adds | Deletes | Net | Notes |
|---|---|---|---|---|
| 1 | ~110 (dataclasses ~45, `_logical_memory_access` ~40, consumer-lane plumbing ~25) | ~0 | +110 | Pure scaffolding; every consumer arrives in phase 2. Ships inside the 1-3 stack, so no standalone dead-code window |
| 2a | ~180 (store ~150, shadow-assert harness ~30) | ~0 | +180 | Shadow mode: name path stays authoritative; kernels byte-identical by construction |
| 2b | ~85 (authority flip ~30, guard apply ~35, `_load_without_store_forwarding` ~15, materialize delta ~5) | ~30 (shadow asserts, name-read gates) | +55 | Small flip commit validated by every suite run accumulated under shadow |
| 3 | ~15 (shared builder) | ~240 (resolver 75, family resolve_load+dict+lane fields ~50, masked/forwarded plumbing ~40, three views 36, name calcs/partitions ~25, dead TODOs ~15) | -225 | The sweep absorbs the deletions the small 2b commit defers |
| 4 | ~70 (record builder ~55, membership branch ~10, stage field ~5) | ~76 gross (branch 10, helper 10, broadcast prover 56) of which ~55 relocates into planning | ~-25 true | One prover instead of two is the real win |
| End state | | | ~+95 | |

Honest bottom line: the end state is a modest net-ADD in raw production LOC
(~+95), not a net-delete. What net-deletes is the mechanism inventory the task
and contract enumerate: three name views, latest-value-by-name tracking, the
masked-forward name sets, the name-keyed store-cache fallback inside the staged
path, and the second fusion prover -- all out; one keyed cache in. The LOC add
is the price of exact identity (key dataclasses, guard application, the
store-cache bypass) and buys F2 a landing surface that re-adds none of the
deleted machinery.

RULING (2026-08-26, review): the ~+95 net-add end state is ACCEPTED -- the
criterion was mechanism-inventory reduction, not raw line count. The
previously documented lever for chasing raw net-delete (recording and
resolving unguarded keys only, at the cost of the contract's mask/fill
acceptance row) is REJECTED. Do not relitigate.

---

## 6. Risks

1. **Temporal writer identity across mutation renames.** Plan records keep
   node-local temporal names; `_prove_staged_fusion_dependencies` renames only
   at match time (9487-9496). The generic store cache propagates stored values
   to mutation aliases (`_update_store_cache`, common.py:3019-3024); the keyed
   store records under the store's own dep name only, so an aliased consumer
   read would miss. Today unreachable -- both planners decline
   `has_aliasing_or_mutation` epilogues (scheduler.py:674, 1624) -- but the
   store must fail closed (miss -> reload or loud must-forward error), never
   consult graph-final mutation maps at record time (contract property 1). Add
   a tripwire test that a mutation-bearing candidate still declines planning.
2. **Load-mask capture semantics.** Today: source loads under `_load_mask` are
   never captured (simd.py:2390); forwarding under a consumer mask happens only
   for `masked_forward_names` = planner-proved internal names (2329-2332,
   3440, 3804), and the consumer re-applies its own guard by construction
   (same body emits the guard). Per-key mapping: masked stores get guarded
   keys; guarded consumers may use unguarded sources with
   `_apply_consumer_guard` re-applying mask+fill; any other guard transition
   asserts. Two hazards: guard identity is the mask variable STRING -- stable
   for tree masks (xmask/r0_mask) but not for tmp-derived masks across loop
   bodies, so unmatched guards must decline (fail closed), and a decline on a
   must-forward value is a loud error -- the masked-store case is pinned by
   `test_projected_store_records_masked_store_guard` (landed in 2a; the
   masked-callback test originally cited here is F2-only per 2b ruling 3);
   and fill-value typing (`bool` vs numeric fill, OLD handles via
   `value.bounds.is_bool`).
3. **CSE invalidation interplay.** Liveness is `cse.contains_value`
   (common.py:2138-2141, scans store_cache values too); flushes move store_cache
   entries into `invalidated_stores` (2094-2097); `CSEProxy.load` consults
   `invalidated_stores` then store_cache by name (3002-3010).
   `_load_without_store_forwarding` duplicates CSEProxy.load minus the
   store_cache hit -- a drift hazard (this exact area just changed:
   `store_buffer_counts` accounting in the uncommitted diff). Constraint for
   phase 2: implement it as a shared helper next to `CSEProxy.load` (a
   skip-store-cache mode) rather than a copy in simd.py. [Resolved in 2b:
   `load_without_store_forwarding` at common.py:2779, behavior-identical
   extraction.] Also the memoized `_materialized` map assumes no
   CSE-invalidating flush inside one derived stage; all flushes sit BEFORE
   the sweep/epilogue region today, and the stale-cache raise was removed by
   2b ruling 1 (it false-fired on newvar lanes; the live-source gate plus the
   one-region structure plus `_record`'s memo pop are the real guards).
   STANDING FAIL-CLOSED REQUIREMENT: any future change that introduces a
   `codegen_body`/`cse.invalidate` point between stage emissions (F2 deferral
   is the likely candidate) must clear `_materialized` alongside
   `cse.invalidate`; nested BROADCAST memos are cache-tracked vars whose
   validity the source-liveness gate does NOT imply across a flush.
4. **Fall-through correctness invariant.** Unplanned reads of planned names
   still take the ordinary load path, which may hit the name-keyed store_cache.
   This is only sound because `_prove_staged_fusion_dependencies` admits
   non-plan reads solely when `fusable_read_and_write` holds (exact index), so
   name-forwarding coincides with exact forwarding. Document this coupling at
   both sites; the miss-bypass test plus the mutation battery guard it.
   BATTERY-VERIFIED (2026-08-26, gate rulings A/B): the fall-through has a
   layered backstop when the invariant is violated by a desynced plan --
   (i) width-mismatched wrong forwards (parent/group-width values in a
   lane-width expression) fail compilation at the block-shape invariant,
   shape_propagation.py:45-51; (ii) reads of a flushed in-kernel buffer miss
   store_cache (`invalidated_stores`) and become barrier-guarded real reloads
   via TritonKernel.load's read-after-own-store guard (triton.py:5062-5070),
   verified bitwise-correct in the mutated looped kernel text; (iii) the only
   silent forwards are width-compatible AND semantically correct (exact-index
   IDENTITY; broadcastable reduced values). Do NOT add a "raise on unplanned
   access of a planned name" hardening: it would false-positive on the
   legitimate exact reads this invariant exists to allow. The fall-through
   path is GENERIC CSEProxy machinery (store_cache, the shared reload helper,
   the triton barrier) -- phase 3's deletion sweep must not touch it.
5. **Recursive factor-4 / MXFP6 (4,3) shapes.** Lane tuples now select by the
   planner-recorded lane rather than recomputing from the kernel-frame index
   (`interleaved_sub_parent_lane` leaves codegen). Output groups with
   `output_lanes < factor` read a lane subset; verify per-consumer lanes cover
   exactly the consumed subset and that `emit_split_via_reshape`
   (triton.py:6407-6418) call sites are unchanged. MXFP6 kernel-form pins are
   the tripwire.
6. **Plan-vs-body dep drift (frame matching).** `extract_read_writes` may
   simplify indices differently from `body.indexing_exprs`; `.normalize()` on
   both sides absorbs known differences, and the phase 1 agreement sweep is the
   detection mechanism before any behavior depends on it. Approach B remains
   the documented fallback.
7. **F2 interaction points (interface only, not planned).**
   `resolve_source`/`materialize_source` split (F2 defers between them);
   `can_defer_projection`; `materialize_projections` as the eager-vs-lazy gate;
   a future `_default` hook on `_PointwiseRemapHandler`; `staged_reduction`
   kernel kwarg + heuristics; `emit_reshape_preserving_dtype`. Phase 2 must
   keep the resolve/materialize seam exactly so F2 rebases without reshaping
   F1. F2 REBASE PRECONDITIONS recorded by the 2b review: (i) if F2 moves
   materialization across a flush boundary, `_materialized` must be cleared
   with `cse.invalidate` (risk 3's standing requirement); (ii) redesign delta
   4's loud-stale-cache raise belongs to F2's `_view`/`_split` caches, not to
   F1's `_materialize` memo; (iii) the F2-only tests
   `test_projection_materializes_store_reduction` and
   `test_projection_materializes_masked_callback_body` (deferred-value
   plumbing through `projection.apply`) port with F2, not F1. Iteration 8's
   R3/R4 (record should live in generic codegen, shared with the ordinary
   store cache) is a further follow-up direction, not part of this rebase: F1
   keys the staged path only.
8. **Reshape-family regime dependence (Phase 4 closing ruling A).** The
   PARENT_TO_GROUPED reshape-equal records are structurally inert at the
   prove step under `loop_ordering_after_fusion=True` (both canonicalization
   mechanisms -- `fusable_read_and_write`'s config-gated normalize and
   post-merge re-extraction -- make those reads raw-exact before membership),
   but CONTRACT-REQUIRED under `loop_ordering_after_fusion=False`, the fbcode
   default. Do not delete or "simplify" the reshape family based on
   OSS-regime inertness. Pins:
   `test_reshape_equal_grouped_relation_requires_membership` (unit,
   LOAF=False) and `test_dropped_grouped_relation_reshape_family_is_inert`
   (e2e, flags any future load-bearing shift).

---

## 7. Test strategy

Existing suites that pin each phase (run per phase, full files not selectors):

- `test/inductor/test_nested_reduction.py` (~396 tests): kernel-form pins
  (NVFP4, MXFP4, MXFP6 4,3 + preshuffle), internal-source looped/persistent
  (1880-1959), masked/indirect (981, 998, 2232), source
  rejections (1178-1440, 2215-2377), standalone/nested numerics.
- `test/inductor/test_inductor_scheduler.py`: projection frame contracts (492,
  543), injective-producer and unsafe-write rejections (714, 774), ownership
  scoping (746; rewritten in phase 4), parent ordering (908, 943), loop-rewrite
  freezing (806+).
- `test/inductor/test_triton_heuristics.py`: unchanged by F1 (its OLD additions
  were F2); run as regression.

New tripwires by phase:

- All phases: the inherited P0 guard battery (extra-axis trailing-broadcast
  rejection, TMP/sync-mode gates, same-group transposed IDENTITY read, Path-4
  append hostile read, positive controls) stays green throughout. It pins the
  safety gates; the record-removal mutations below pin the records -- two
  distinct tripwire classes, both required.
- P1: plan-vs-body agreement sweep; lane recorded on every INTERLEAVED
  consumer; guard capture unit test; P0 fixtures adapted to
  `ProjectedConsumerAccess`.
- P2a: shadow-agreement sweep across all suites and the fuzz matrix (decision
  equality between keyed and name-based resolution, including planner-lane vs
  index-derived-lane equality).
- P2b: the four ported negative tests (exact miss, store-cache bypass, atomic,
  expired value); guarded projection parametrized matrix; mutation battery for
  INTERLEAVED/BROADCAST/IDENTITY record removal (loud failure or lost fusion in
  a one-kernel test); differential fuzz vs pre-phase captures (bitwise).
- P3: grep-based absence assertions are not tests -- rely on the phase 2
  battery still passing with the views gone; add one test constructing a stage
  and asserting the deleted attributes are gone (AttributeError) to prevent
  compat-layer resurrection.
- P4: mutation battery per grouped-relation family; the rewritten two-path
  ownership test; full nested-reduction file to catch silent lost fusion
  (kernel-count asserts in existing tests make lost fusion loud); negative
  decline matrix re-run.

Throughout: kernel sources byte-compared against the pre-phase tree for the
flagship forms; any diff is reviewed individually and recorded in the phase
commit message.

METHODOLOGY NOTE (2026-08-27, Phase 4 closing ruling B): bitwise equality
between two INDEPENDENT compiles is not a sound assertion -- autotune config
selection changes float contraction order, and ~1 ulp f32 flakes reproduced
at ~1/6 in a control arm comparing a compile against itself recompiled. Both
mutation batteries use the shared `_assert_battery_results_match`
(test_nested_reduction.py:4170): floats >=16 bits at the suite's 1e-2
convention (garbage-catching; lost-forwarding garbage observed at magnitude
~192), integers and sub-16-bit floats (packed quant bytes, fp8/e8m0 scales)
EXACT. Kernel-source byte-comparison remains valid (it compares code, not
runtime numerics). Future differential-fuzz claims must say "bitwise vs
recorded reference tensors of the same compile" or use the helper's
convention -- never assert bitwise across independent compiles. The recurring
AOTI suite error is environmental (did not reproduce on the battery box;
clean 431 before and after).
