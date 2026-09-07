# Stack review: sub-parent reduction epilogues (2026-08-18)

Stack reviewed, bottom to top, on base `41d7be3f48d`:

| # | commit | PR | subject |
|---|---|---|---|
| 1 | `86038d6e1cd` | #190594 | Fuse interleaved sub-parent reduction epilogues (standalone) |
| 2 | `1094cf01c2b` | #190595 | Fuse interleaved epilogues in nested reductions |
| 3 | `f8d435e3aa8` | #191775 | Fuse staged MXFP6 packing epilogues (4:3) |
| 4 | `282aa7405f9` | #190596 | Fuse contiguous sub-parent reduction epilogues |
| 5 | `d4d6f910e9f` | #191974 | Deduplicate nested reduction test helpers |

## Method and evidence standards

- One review agent per commit, each reviewing the isolated diff against its
  parent with the tree checked out at that exact commit
  (`/tmp/stack_review/wt1..wt4`), plus one cross-stack agent auditing
  inter-commit contracts ("Changed later" promises, invariant widening,
  feature interactions at the tip, bisectability of mid-stack test edits).
- Every finding below was independently re-verified against the actual trees
  before inclusion; agent claims that did not survive verification were
  dropped or explicitly marked refuted.
- Dynamic repros ran against the installed build, which serves Python from
  the working tree. NOTE: `wip_quant` is checked out at `94367fffd77`
  (Aug 14), a divergent older replay of this stack -- NOT the reviewed tip
  `d4d6f910e9f` (Aug 17); neither is an ancestor of the other. Wherever a
  repro ran, the relevant code was statically confirmed identical at the
  reviewed commits.
- Evidence labels used below:
  - [PROVEN] reproduced live, wrong numerics observed
  - [VERIFIED] mechanism confirmed by direct code read at the cited commit
  - [TRACED] agent-traced call path, spot-checked but not fully re-derived
  - [REFUTED] tested dynamically and did not hold as claimed

Repro scripts (kept in `agent_space/`):
- `repro_looped_external_contiguous.py` -- C1
- `repro_mxfp6_stack_dim2.py` -- refuted (4,3) lane-axis claim
- `repro_190595_claims.py` -- C3 tip behavior, topology-abort behavior,
  realized parent-full intermediate

Run with:

```
TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 \
LD_LIBRARY_PATH=/home/eellison/.conda/envs/pytorch-3.12/lib:$LD_LIBRARY_PATH \
PYTHONPATH=/data/users/eellison/pytorch \
/home/eellison/.conda/envs/pytorch-3.12/bin/python agent_space/<script>
```

## Summary

The core design held up under adversarial review: lane proofs, the
plan-rebuild-after-`merge_loops` contract, the #191775 dep-matching allowlist
(StarDep/WeakDep preservation), recursive split lane ordering, producer-first
ordering, and all "Changed later" promises between commits were verified
kept; no mid-stack test breakage or API bisect hazards. There is one proven
silent miscompile at the tip and one statically confirmed second correctness
hole, both in the CONTIGUOUS layout's "resident tile == logical extent"
assumption, plus several robustness and lost-fusion issues.

**Recommendation: request changes, scoped to #190596** (fix C1 and C2), plus
the one-word C3 fix in #190595. C4/C5 are cheap robustness fixes worth
folding in. Everything else is non-blocking but should land before
`nested_reduction` flips default-on.

---

## Correctness findings

### C1. [PROVEN] Looped kernel register-splits an external CONTIGUOUS source from a partial R0_BLOCK

Blame: #190596 x #191775 interaction. Found independently by the cross-stack
and #190596 agents; mechanics verified line-by-line at the tip; reproduced
live (staged kernel forms; output wrong with max-abs-err ~7.14 while
`nested_reduction=False` matches eager).

Mechanism (all cites at tip / `282aa7405f9`, which is torch/-identical to
`d4d6f910e9f`):

1. `_codegen_reduction_with_sub_parent_epilogue` skips `kernel.codegen_body()`
   whenever ANY internal source exists (`simd.py:3776-3777`), so all recorded
   parent-stage values -- including an external CONTIGUOUS source re-loaded in
   the deferred parent segment -- stay CSE-live. (The `DisableReduction`
   boundary before the deferred segment does flush earlier loop passes; the
   hole is specifically the never-flushed deferred segment, which is exactly
   the chain that recomputes the internal source and typically re-reads the
   same input the reduction read.)
2. `_SubParentSourceLoadResolver.resolve_load` lazily materializes any
   CSE-live planned source with no persistence or layout gating.
3. `materialize_value_at_sub_parent_resolution` (`simd.py:2083-2095`) has no
   `kernel.persistent_reduction` check on the CONTIGUOUS arm; it reshapes the
   resident tile to `(XBLOCK, factor, R0_BLOCK//factor)` and permutes -- i.e.
   it slices the BLOCK into chunks, which equals the logical lane only when
   the block covers the whole extent.
4. `_ContiguousSubParentRemappedValue` carries the full logical extent, so
   `sub_parent_contiguous_lane` returns a valid constant lane and
   `_select_lane` succeeds. No assertion fires; values are silently wrong for
   every loop iteration except the last.
5. The only guard, `_sub_parent_has_internal_contiguous_source`
   (`simd.py:2734-2746`), intersects CONTIGUOUS names with INTERNAL names, so
   an external contiguous source never forces persistence.

Trigger: internal source (any layout) + external CONTIGUOUS source + looped
kernel. The planner admits the mix (`_try_get_sub_parent_source_layouts`
chooses layouts per source name independently; mixed layouts are an
explicitly supported/tested feature, but only with all-external sources).

Repro shape (D static pow2, large; `split_reductions=False` to keep one
reduction node at very large D):

```python
h  = x + residual
v  = h.pow(2).mean(-1, keepdim=True)
hn = h * torch.rsqrt(v + 1e-6) * weight
inner = torch.ops._inductor_test.realize(F.silu(hn))  # internal, INTERLEAVED
even  = inner.view(B, D // 2, 2)[..., 0]
hi    = x.chunk(2, dim=-1)[1]                          # external, CONTIGUOUS lane 1
return even.float() + hi.float()
```

Autotuner caveat: at D=4096 and D=32768 the tuner happened to pick
R0_BLOCK == D (single-trip loop -- the one case where the split is
coincidentally correct), masking the bug; D=2^20 with `split_reductions=False`
reproduces it decisively. The emitted-kernel dump at D=4096 already shows the
wrong-form emission (`tl.split(tl.permute(tl.reshape(tmp7, [XBLOCK, 2,
R0_BLOCK//2])))` on an in-loop load), so the bug class exists at every size;
whether it fires depends on the tuned config.

Fix options (from the two agents, either suffices):
- Widen the persistence condition to
  `bool(contiguous_names) and bool(internal_source_names)` (any internal
  source suppresses the flush, so any external contiguous source then needs a
  resident full block), or
- Refuse the CONTIGUOUS projection in
  `materialize_value_at_sub_parent_resolution` when
  `not kernel.persistent_reduction` (falls back to the correct derived-index
  reload).

### C2. [VERIFIED] The contiguous gate proves pow2(parent_rnumel), not R0_BLOCK == parent_rnumel

Blame: #190596. The persistent split is only correct when the padded resident
block equals the logical extent. The gate checks that `parent_rnumel` is a
static power of two -- but `_get_persistent_reduction_block`
(`triton.py:7570-7578` at tip) accepts any strict-reduction rblock
`>= rnumel`, so under `TORCHINDUCTOR_NUMERICS=strict` (config.py:1065) a
persistent kernel can have `R0_BLOCK > parent_rnumel`; the contiguous split
then slices padding into lanes, silently.

The invariant is already recognized elsewhere:
- nested planning excludes strict reductions via `_is_enabled_for`
  (`scheduler.py:559-568`: `not outer_node.has_strict_reduction() and not
  grouped_node.has_strict_reduction()`);
- combo kernels: "Keep strict reductions standalone so their planned R0_BLOCK
  cannot change" (`scheduler.py:4477`).

The standalone `sub_parent_epilogue_plan` (`scheduler.py:649`) has no such
gate. INTERLEAVED is immune (trailing padding preserves parity and the child
mask covers it); this is CONTIGUOUS-specific.

Fix: add `has_strict_reduction()` exclusion to the standalone plan (mirroring
nested), and/or FileCheck `R0_BLOCK == rnumel` in the persistent chunk
kernel-form tests to pin every future source of block padding.

Not dynamically reproduced (strict mode not exercised); mechanism fully
verified statically.

### C3. [VERIFIED mechanism; tip impact contained] The SUB_PARENT/REDUCED tiebreak keys on used_buffer_names(), which includes the reduction's own output

Blame: #190595; persists at tip (`scheduler.py:1346, 1387-1393`).
`reduction_source_names = grouped_reduction.used_buffer_names()` is
reads|writes, so `reads_reduction_source` is true for essentially every
consumer of the grouped output, collapsing the G=2 tiebreak to
"sub_parent-compatible implies SUB_PARENT". The comment and the design intent
("reads the reduction's SOURCE") call for reads-only names.

- At the isolated commit 2, the agent traces a real lost-nested-fusion for a
  reduced-only G=2 consumer (`amax * 2`): the misclassified node produces an
  empty source set, `_plan_nested_sub_parent_stage` returns None, and the
  whole nested topology declines. [TRACED]
- At the full-stack level my dynamic check shows the same shape fuses into
  one staged kernel with correct numerics, so the practical tip impact
  appears contained -- but the classification is still wrong-by-construction
  and the shipped G=2 test cannot detect it (both of its consumers are
  genuinely sub-parent).

Fix: use the grouped reduction's `read_writes.reads` names. Add a
reduced-only G=2 consumer test.

Related [TRACED, commit-2-scoped]: a sub-parent stage that cannot be planned
aborts the entire nested topology at `plan()` time (`scheduler.py:1247-1261,
1476-1485` at wt2) rather than declining just the epilogue. Dynamically, at
the full-stack level, a broadcast-only epilogue declines gracefully (nested
pair still fuses, epilogue in kernel 2), so this appears fixed-or-masked by
the later rewrite; the append path always handled None gracefully. Worth a
tip-level test pinning "epilogue declines, topology survives".

### C4. [VERIFIED] Planner/codegen lane-recovery asymmetry (interleaved)

Blame: #190594; persists at tip. The planner calls
`interleaved_sub_parent_lane(child_index, factor, extent_subs,
source_sizes=(parent_numel, parent_rnumel))` (`scheduler.py:1160`); codegen's
`_DerivedIterationFamily.resolve_load` calls the same function with NO
`source_sizes` and with `lane_index_subs` derived only from the group extent
(`simd.py:1584`). A stride symbol provably equal to `numel*rnumel` can make
the planner's proof succeed while codegen's fails; by then the typed identity
is set and codegen cannot decline -- the compile dies with
`AssertionError("sub-parent load for ... has non-constant lane")`. Loud, not
silent; persistent-form only; medium reachability.

Fix: pass the same sizes through the family, or make an unresolved lane fall
back to an ordinary derived-index load instead of asserting.

### C5. [VERIFIED] sub_parent_contiguous_lane does not strip Identity

Blame: #190596. `interleaved_sub_parent_lane` begins with
`index.replace(Identity, ...)` (`scheduler.py:999`) and the planner strips
Identity from normalized dep indices; `sub_parent_contiguous_lane`
(`scheduler.py:1030-1046`) does not, and `resolve_load` hands it the raw body
index. An Identity-wrapped index (cat lowering) leaves `Mod` unfolded ->
`_select_lane` returns None -> planner-invariant AssertionError on an
approved graph. Narrow reachability (needs `masked_forward_names`); one-line
fix.

### C6. [TRACED] Persistence heuristic is re-consulted at codegen re-plan

Blame: #190596. The internal-contiguous plan gate calls
`V.choices.should_use_persistent_reduction` at fusion time AND at the codegen
re-plan; its inputs (`get_reduction_hint` over `read_writes`) are rewritten
by `merge_loops` (`refresh_dependencies(normalize=True)`). A flip across the
phase boundary is a hard AssertionError ("plan was lost before codegen" /
"looped sub-parent codegen cannot forward an internal contiguous source").
No concrete flip was constructed (loop merging moves deps toward contiguous,
the safe direction) -- an undefended invariant rather than a live bug. Unlike
the tiling decision, this is a perf heuristic riding the legality contract.
Fix: record the persistence requirement on the FusedStagedReduction identity
(decide once at fusion time).

### Refuted and downgraded claims

- **MXFP6 (4,3) lane-axis-not-innermost miscompile** (claimed for #191775):
  [REFUTED]. A `torch.stack((low, middle, high), dim=-2)` pack was tested
  dynamically: the planner declines it and emits a clean two-kernel fallback
  with correct bytes (the staged kernel keeps only amax + the single-lane
  intermediates). Admission-path code verified equivalent between the replay
  and the reviewed commits. The underlying point survives as hardening:
  nothing PROVES the lane-innermost invariant codegen relies on
  (`pair * output_lanes + output_lane`); add a cheap planner check (innermost
  extent == output_lanes) and a dim=-2 negative test pinning the fallback.
- **masked_forward_names discards mask and fill** (#191775): forwarding a
  store-cache value under an active `_load_mask` relies on the consumer
  re-guarding (true for `pointwise_cat`, which is the only current producer
  shape); the planner does not check this. No reachable counterexample was
  constructed. Leave a comment stating the actual safety argument.
- **rnumel <= factor domain coincidence** (three agents converged
  independently): when `parent_rnumel <= MAX_SUB_PARENT_FACTOR`, sub-parent
  and reduced domains coincide. #190596's check reorder
  (`scheduler.py:853-856`) resolves it toward REDUCED -- necessary for the
  widening to 16, but it silently drops genuine `rnumel == factor` sub-parent
  fusions; commits 1-3 resolve it the other way and are rescued only by the
  source-layout requirement. No failing shape exists (nothing tests
  `rnumel <= 4`). Document the reorder as load-bearing; decide the
  `rnumel == factor` policy explicitly.

---

## Compile-time / performance findings

### P1. [VERIFIED] Under config.triton.multi_kernel, staged kernels silently emit the looped form

Blame: #190594; persists at tip. `add_multi_kernel_choices` sorts persistent
last (`triton.py:8548`) and both staged paths take
`create_kernel_choices(...)[0]` (`simd.py:3299` nested, `simd.py:3745`
standalone), so whenever a persistent kernel was optional the looped twin is
chosen and the second kernel is constructed and discarded -- no MultiKernel,
no register split. Correct numerics, silent form change; every
`assert_single_kernel_form` golden would fail under that config. Only the
internal-contiguous case (which forces `override_persistent_reduction=True`)
is pinned under multi_kernel. Contradicts READ_ME_FIRST's claim of coverage
for "persistent selection under both single- and multi-kernel choice
generation".

### P2. [VERIFIED] Staged kernels' bandwidth/FLOP metrics omit the epilogue

Blame: #190594. `buf_accesses` (`simd_kernel_features.py:186`) and
`estimate_flops` (`simd.py:1322`) read `scheduler_nodes()` from the
parent-only schedule; the index-dtype split introduced
`indexing_node_schedule` but these two were not moved. Epilogue output
buffers are absent from `buf_accesses`, so `estimate_kernel_num_bytes`
appends 0 for them -- for NVFP4 that is both half-resolution output stores,
the majority of write traffic. Affects `kernel_num_gb` under
`benchmark_kernel` / `profile_bandwidth` / combo benchmarking -- i.e. exactly
the GB/s figures the stack's perf claims are read from.

### P3. [VERIFIED] Planning fan-out once the flag is on

- #191775 removed the `NestedReduction.is_candidate` early-return from
  `_nested_index_equivalent_dep_names` (verified wt2 vs wt3): every
  (reduction, pointwise) `_can_fuse` pair now runs a full
  `sub_parent_epilogue_plan` on the combined set, possibly a second plan on
  node1 alone, a speculative reindex (which invalidates dependency/tiling
  caches), and a third plan -- before `SIMDScheduling.can_fuse` runs its own
  plan over the same nodes.
- #190595's `_plan_append` calls `backend.fuse(self.node2, other)` per
  `can_fuse_with` QUERY, which runs the standalone planner including
  `_sub_parent_tiling_is_2d` -> `get_tiling_and_scores` (uncached), then
  `plan_from_topology` again; repeated in `fuse_with`. Side effect: the
  retained node2 is retyped FusedStagedReduction, routing later non-sub-parent
  appends through `_sub_parent_epilogue_decision` where a failed standalone
  plan can reject appends the nested planner would accept. [TRACED]
- #190594 runs `has_sub_parent_epilogue` for every accepted fusion graph-wide
  and `_sub_parent_tiling_is_2d` inside the plan builder (3 call sites).

All flag-gated (default off) and bounded, but the memoization TODO
(`scheduler.py:8949` area) should land before default-on.

### P4. [TRACED] Append-path ordering dependence for outer-stage reads

Blame: #190595. For sub-parent appends the producer switches to the whole
`FusedNestedReductions` (`producer_node=self`) but `index_equivalent_dep_names`
is built from node2's buffers only (`grouped_buf_names`), so a sub-parent
consumer reading an OUTER-stage output at derived resolution is always
rejected on the append path while the same read is accepted on the
`NestedReduction.plan()` path. Consequence: the flagship RMSNorm->NVFP4
fusion depends on the standalone amax+lanes fusion happening BEFORE the
nested pair; under the other ordering the epilogue silently stays in a second
kernel. The node2-only construction persists in the tip's append path.
Fix: union `_producer_output_names_read_by_consumer(self.node1, other)` into
the allowlist when `producer_node is self`. No test covers this append shape.

### Minor

- `_reindex_sub_parent_consumer` applies `apply_new_loop_order` /
  `apply_loop_reindexing` without consulting `loop_ordering_after_fusion` /
  `loop_reindexing_after_fusion`, unlike every other loop-mutation site --
  and the stack's own tests set `loop_ordering_after_fusion: False`.
  (#191775)
- `num_store` over-counts by `output_lanes - 1` when a multi-lane output is
  removed as kernel-local: `CSEProxy.store` increments per instruction,
  `remove_kernel_local_buffers` decrements once per buffer
  (`common.py:2535`). Skews `inductor_meta`. (#191775)
- The internal-source ordering gate (`_order_sub_parent_parent_nodes`
  rejection when the source chain feeds the reduction) is unconditional
  though the constraint is looped-only; persistent kernels lose legal
  fusions. Both rejection tests skipTest in the persistent class, so
  persistent behavior is unpinned in either direction. (#191775)
- `_prescan_host_tma_materializability` builds its blocklist from the
  parent-only `node_schedule`, so epilogue lane-strided reads are never
  scanned. Gated on `config.triton.use_tensor_descriptor` (default False).
  (#190594) [TRACED]
- Degenerate-plan guard: `_order_sub_parent_parent_nodes` can return an
  empty deferred chain (`deferred_start == len(parent_nodes)`) if the only
  internal-source writer is itself a reduction (scan); codegen then inserts a
  boundary that invalidates the value it hard-requires. Effectively
  unreachable today (Triton scans are persistent); cheap to validate
  `0 < deferred_parent_start < len(parent_nodes)` in
  `StagedReductionPlan.__post_init__`. (#191775) [TRACED]

---

## Cleanup

- **Twelve new `# noqa: S101` suppressions** across the stack, a direct
  CLAUDE.md violation ("Never silence S101 with a noqa" -- checks vanish
  under `python -O`). At tip: `simd.py` 1603, 2041, 2086, 2097, 2118, 2119,
  2122, 3379, 3380, 3756; `triton.py` 6262, 6301. Several guard invariants
  the guides call load-bearing (`len(kernel.range_trees) == 2`,
  `flat_index_derived_tree is not None`). Rewrite as
  `if not cond: raise AssertionError(...)`.
- **Guide/doc drift on nested rates**: READ_ME_FIRST's supported-forms table,
  the #190595 guide, and the #190596 guide all state nested append is
  factor-2 INTERLEAVED-only. At tip, `_nested_sub_parent_rate` admits any
  pow2 factor up to 4 AND the (4,3) multi-lane rate; factor-4 single-lane
  nested is tested, nested (4,3) is representable and emittable but untested.
  Tighten to `output_lanes == 1` or add the test; fix the guides either way.
- `test_producer_consumer_parent_full_intermediate`'s `source` never realizes
  (single FX user; inlined below `realize_reads_threshold`), so the
  "source written by pointwise in the fused group" feature
  (`include_writes=True` in `normalized_source_indices`) is untested -- the
  golden `tl.split` count (2) is the graph inputs x and weight. Dynamically
  confirmed: an explicitly realized source DECLINES fusion (2 kernels), so
  the test pins inlining, not the feature, and the guide overstates support.
  `test_producer_consumer_rejects_shifted_parent_full_intermediate` passes
  for the same wrong reason. (#190595)
- Leftover local `import torch.nn.functional as F` at
  `test_nested_reduction.py:1500` -- #191974 removed 38 of 39 local imports;
  this one contradicts the commit message's "once at module scope". Harmless.
- Dead `is_constexpr` flag in `_grouped_axis_named_constants`: group size is
  guaranteed `sympy.Integer` by `_get_grouped_reduction_and_size`, so the
  non-constexpr branch is unreachable; the plumbing for a symbolic group size
  does not exist. (#190595)
- `_sub_parent_tiling_is_2d` docstring is stale: says a 3D tiling "would
  surface as an assertion during codegen" and that codegen "calls the same
  helper" -- codegen force-creates the 2D tiling and never re-runs the
  heuristic. It is a profitability guard, not a legality guard. (#190594)
- `emit_split_via_reshape` re-implements the fp8-via-uint8 bitcast that
  `_bitcast_reshape_expr` (12 lines above) provides. (#190594)

---

## Test coverage gaps (concrete)

1. The C1 repro shape (internal source + external CONTIGUOUS source, looped)
   -- must-add with the fix.
2. A genuinely multi-trip looped contiguous kernel (`R0_BLOCK < rnumel`); the
   autotuner masked C1 at D=4096/32768 by picking whole-extent blocks.
3. `test_internal_contiguous_source` runs under a STUBBED persistence
   heuristic (class-level `_choices_context(force_persistent_outer_reduction)`),
   so the real gate at `simd.py:2727` is never evaluated and the codegen
   backstop can never fire. The multi_kernel half (override suppresses the
   looped twin) is genuine coverage; the gate's own decision is not.
4. The persistent fork of `assert_default_rmsnorm_chunk_kernel_form` is dead:
   `_InternalsBase` runs the real heuristic and D=8192 (INNER hint, threshold
   1024) always loops, so the `triton_per_fused` branch and all
   `persistent_*` golden args never execute -- the default heuristic never
   exercises a contiguous permute+split at all.
5. No multi_kernel coverage for plain staged kernels (P1).
6. `stack(dim=-2)` (4,3) negative test pinning the clean fallback.
7. Strict-reduction exclusion for the standalone plan + test (C2); or
   FileCheck `R0_BLOCK == rnumel` in persistent chunk kernel-form tests.
8. Reduced-only G=2 consumer (C3); `rnumel == factor` shapes.
9. Append of a sub-parent consumer reading an outer-stage output onto an
   existing FusedNestedReductions (P4).
10. The two hard invariants are unpinned: `"staged reduction reached generic
    SIMD codegen"` (`simd.py:3811`) and `"sub-parent reduction plan was lost
    before codegen"`.
11. A reindex that is applied, re-plans successfully (`keep_reindex=True`),
    and is then rejected by a LATER gate (`V.choices.can_fuse` /
    `backend.can_fuse_vertical`) -- the only rollback path not covered by the
    try/finally.
12. MXFP6 flagship input is a repeating 4-value pattern, weak against group
    off-by-one; the exact test's `arange(G) % 17 - 8` style would strengthen
    the non-exact tests (guide already lists this as a reservation).
13. No `benchmark_kernel` compile checks that `kernel_num_gb` accounts for
    epilogue outputs (P2).

---

## Cross-stack audit (verified clean)

- All "Changed later" promises kept completely: #190595 removes the
  combined-stage guard and routes rebuilds through
  `PointwiseDomainContext.create()` (single construction site, reached by
  both `plan()` and `plan_from_topology()`); #191775 widens INTERLEAVED to 4
  and adds the per-read normalized index proof; #190596 adds CONTIGUOUS and
  widens one-output factors to 16.
- No invariant drift: `INTERLEAVED_SUB_PARENT_FACTOR`,
  `NESTED_SUB_PARENT_FACTOR`, `PointwiseDomainContext.sub_parent_domain`,
  `allow_index_equivalence` have zero references at tip; surviving
  `factor == 2` in triton.py is the recursive-split base case; remaining 2D
  asserts are genuine tiling invariants.
- Dep-matching refactor consistent across all three fusion paths (nested
  pair, standalone, nested append) through the same
  `index_equivalent_dep_names` parameter; the append path passing no proven
  names defaults to the strictly safer meaning.
- Mid-stack test coherence: no commit edits an earlier commit's test in a way
  that leaves a mid-stack failure; helper signature changes
  (`check_kernel_meta`, `_try_get_sub_parent_source_layouts`,
  `emit_split_via_reshape`, `sub_parent_iteration_values`) all updated their
  call sites in the same commit.
- Feature interactions checked: CONTIGUOUS x multi-lane output groups is
  tested and store striding is source-layout independent; internal chains can
  read contiguous-split values safely; `allow_contiguous=False` threading is
  complete on the nested path (independently capped by
  `_nested_sub_parent_rate`); reindex-x-contiguous is guarded by the re-plan.
  The one uncovered cell was C1.
- `python -m py_compile` passes on all touched files at tip;
  `config.triton.nested_reduction` defaults OFF matching the README;
  `metrics.codegen_nested_reduction` and the `min_xblock`/`min_rblock`
  inductor_meta keys referenced by tests exist.
- #191974 verified byte-identical asm constant at all 6 sites, untouched
  golden FileChecks, unchanged MXFP6 producer (comment only), no hidden
  behavioral deltas.

## Per-commit verdicts

- **#190594** (standalone interleaved): core lane proof and phase contract
  sound; ships C4, P1, P2, and the S101 pattern. Fix C4 (or
  fallback-to-load), then non-blocking.
- **#190595** (nested interleaved): sound for the shapes it tests; C3
  tiebreak fix is one word; P3/P4 and the parent-full-intermediate test gap
  should be addressed before default-on.
- **#191775** (MXFP6 4:3): the most intricate commit and it held up --
  dep-matching allowlist, split lane ordering, deferral boundaries, rollback
  all verified. Remaining: P3 fan-out, lane-innermost hardening, minor
  accounting.
- **#190596** (contiguous): the shared lane formula and permute+split
  emission are correct, but the commit encodes "resident tile == logical
  extent" as two narrow proxies (pow2 extent; internal-only persistence) and
  both have reachable counterexamples (C1 proven, C2 verified). Request
  changes.
- **#191974** (test hygiene): purely mechanical; one leftover local import.

## Environment notes

- `wip_quant` is checked out at `94367fffd77` (Aug 14 replay), not the
  reviewed `d4d6f910e9f` (Aug 17). Re-sync before applying fixes.
- Review worktrees: `/tmp/stack_review/wt1..wt4` (commits 1-4; tip torch/ is
  identical to wt4). Safe to `git worktree remove` when done.
- During repro work, an incidental observation: on the replay, a
  nested-off compile followed by a nested-on compile hit the inductor cache
  and skipped staged codegen entirely -- consistent with the FX-graph-cache
  keying issue the resubmitted #190594 fixed; worth confirming the reviewed
  stack's fix covers `TORCHINDUCTOR_FORCE_DISABLE_CACHES=0` round-trips.

---

# Addendum: fix-bundle verification (2026-08-18, later)

Reviewed `agent_space/stack_review_fixes_20260818/` (four sequential patches
on `d4d6f910e9f`). Verified:

- All four patches apply cleanly to a clean detached worktree at the tip and
  recombine byte-identical to `combined.patch`.
- **C1 upgraded from replay-proven to tip-proven**: the repro run against the
  UNPATCHED `d4d6f910e9f` tree (module-overlay harness) reproduces the
  miscompile (staged=1, max-abs-err 7.138847 -- identical to the replay run).
- **C1 fix verified**: on the patched tree the staged fusion still forms
  (staged=1) and numerics match (max-err 1e-6). The fix gates the CONTIGUOUS
  projection on `kernel.persistent_reduction` in
  `materialize_value_at_sub_parent_resolution` -- the safe fallback (derived
  -index reload) rather than a lost fusion.
- C2: plan-time decline of CONTIGUOUS + strict-reduction parents, with a
  non-vacuous test (a pre-fusion pass asserts strictness was actually
  present). C3: reads-only tiebreak + reduced-only G=2 test. C4:
  `lane_source_sizes` threaded to `resolve_load` with the same values the
  planner uses. C5: Identity strip + sympy-level unit test. P1:
  `disable_multi_kernel` opt-out on both staged paths + kernel-form tests
  under multi_kernel. P2: buf_accesses/estimate_flops/TMA prescan moved to
  the indexing schedule + a kernel_num_gb floor test. P4: append-order union
  of node1's per-read-proven names + order-parametrized test. All S101
  suppressions owned by open PRs rewritten. num_store per-buffer counts.
  Deferred-boundary validation at both decline point and __post_init__.
  dim=-2 (4,3) fallback pinned by test.

**Correction to this review**: the `is_constexpr` "dead flag" cleanup item
(from the #190595 agent, listed under Cleanup above) was WRONG -- the flag is
live on the STANDALONE dynamic-R path, where `local_reduction_size` is the
symbolic `parent_rnumel`; the agent only traced the nested caller (static
`group_size`). The fixers demonstrated invalid Triton (`tl.constexpr = ks0`)
when applying it. Item withdrawn.

Remaining non-blocking items the bundle deliberately did not address:
planning-fan-out memoization (deferred until profiling, default-off), nested
(4,3) rate untested + guide "factor-2-only" drift, stubbed persistence
heuristic in `test_internal_contiguous_source`, dead persistent fork of the
default chunk kernel-form tests, persistent-only internal-source-chain lost
fusions, realized parent-full nested sources (tests/guides renamed to
"inlined" instead). One fragility note: `disable_multi_kernel` is popped only
in `TritonScheduling.create_kernel_choices`; a future non-Triton staged
backend would pass it to a kernel constructor unhandled.

Verdict: the bundle resolves every blocking finding (C1-C5, P1, P2, P4) with
minimal, correctly-scoped diffs and honest test renames. Ready to amend into
the respective PRs.

---

# Addendum 2: NVFP4 blocked-scale miscompile fix (2026-08-18, later still)

Reviewed `agent_space/nvfp4_scale_fix_20260818/` (single patch on the new tip
`5c77934b4c5`, which already contains the earlier review-fix bundle --
verified by grep for disable_multi_kernel / store_buffer_counts / the
reads-only tiebreak).

Independently verified:
- End-to-end repro on the clean tip reproduces the exact reported signature
  (131072/262144 packed-byte mismatches, 233248 nibbles, dequantized L2
  1.812); on the patched worktree it is bitwise exact with fusion retained
  (1 kernel both variants).
- The root cause is an UPSTREAM TRITON 3.6.0 MISCOMPILE, isolated in a
  hand-written kernel (repro_triton_fp8_broadcast.py, re-run and confirmed):
  fp32 -> float8e4nv of a reshape(broadcast_to(...)) tile returns 0x80 for
  half the elements; broadcast-derived fp32 arithmetic, bf16 casts, and
  convert-then-broadcast are all exact. Config-dependent (num_warps=8 +
  R0_BLOCK=1024 passes), which is why existing tests missed it.
- The fix (all in #190595 territory): _GroupInvariantBroadcast defers
  broadcasts -- reduced-domain sources forward at group resolution and widen
  only at the first op that meets lane data / a store / a non-elementwise op.
  The shared post-reduction chain (amax -> clamp -> E4M3) therefore re-emits
  at group resolution, CSE-dedupes against the reduced node's own entries,
  and exactly one conversion is emitted with only converted values crossing
  domains. This both avoids the Triton defect and implements the structural
  forwarding the brief demanded; it also strictly reduces emitted work
  (group-resolution reciprocal; 256 vs 2048 conversions in the inline-asm
  kernel form).
- Planning backstop _sub_parent_rebuilds_reduced_computation declines a stage
  when a NON_ELEMENTWISE op would consume all-group-invariant operands --
  deliberately NARROWER than the brief's literal wording (any duplicated
  post-reduction origin), because the deferral makes elementwise duplication
  correct and cheap. Honest caveat: no current lowering reaches it (probed
  with MXFP4 inline-asm, indirect-indexing lookup, bucketize); the rejection
  test stubs `mul` into the deny set to exercise the walk/decline/fallback.
- Suites (agent-run, spot-confirmed by my two independent repro runs):
  452 passed / 12 skipped nested-reduction, 88 scheduler, on the patched
  worktree; 8 affected test instances fail without the patch.

Approved, with flags for the author:
1. Visible golden change: test_nvfp4_inline_asm_kernel_form's uint8-bitcast
   forwarding checks are replaced by the broadcast-of-converted-value regex.
   Strict improvement, but the #190595 PR description should call it out.
2. The backstop narrowing vs the brief's wording needs author sign-off (the
   justification is sound: the literal predicate would reject the case the
   patch fixes).
3. FILE THE TRITON BUG UPSTREAM -- repro_triton_fp8_broadcast.py is a
   ready-made report. Also worth auditing whether any mainline (non-staged)
   Inductor codegen can emit an fp8 cast of a tl.broadcast_to-derived tile;
   the defect is not specific to this stack.
4. Maintenance hazard: NON_ELEMENTWISE_OPS is a deny set, so a future
   OpsHandler op defaults to "elementwise" (the unsafe direction for the
   backstop). A unit test asserting every listed op exists on OpsHandler,
   plus a note in OpsHandler itself, would catch drift.

Worktrees for this fix: /tmp/nvfp4_fix_wt (patched), /tmp/nvfp4_base_wt
(clean tip); remove with `git worktree remove --force` when done.
