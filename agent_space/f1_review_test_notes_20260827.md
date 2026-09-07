# F1 exact-indexed-forwarding test and evidence map

Date: 2026-08-27

Review snapshot:

- worktree: `/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt`
- F1 commit: `fc09fb38287d4639df6424c5c33cfb9749b000ba`
- current review state: the commit includes the simplification that removes the
  custom access-guard cache and leaves masked-load ownership with existing
  Triton codegen
- parent review snapshot: `d4167dba5b8`
- test diff at the current review state: `test_inductor_scheduler.py +235/-40`,
  `test_nested_reduction.py +18/-7`

This note separates tests that establish F1's new runtime contract from tests
whose diff is only the projection-to-access-relation vocabulary change. It is a
review aid, not a request to add one test for every branch.

## Short review order

For the highest signal, read these in order:

1. [`test_sub_parent_access_identity_and_source_cache`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_inductor_scheduler.py:434>)
2. [`test_sub_parent_resolver_uses_masked_load_ownership`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_inductor_scheduler.py:518>)
3. [`test_sub_parent_external_fallback_and_atomic_store`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_inductor_scheduler.py:552>)
4. [`test_group_width_equal_to_child_width_is_direct`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_inductor_scheduler.py:582>)
5. the new temporal-name row in
   [`test_nested_dependency_matches_require_injective_producer`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_inductor_scheduler.py:898>)
6. the new external-source arm in
   [`test_producer_consumer_inlined_parent_full_source`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_nested_reduction.py:1402>)

The renamed broadcast/internal relation tests are important background, but
their acceptance matrices already existed in the parent. Their F1 diff is
mostly mechanical API adaptation.

## 1. Exact consumer identity, source cache, and liveness

Test:
[`test_sub_parent_access_identity_and_source_cache`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_inductor_scheduler.py:434>)

Review weight: **highest**. This is the main unit test for the invariant that a
buffer name alone never authorizes forwarding.

### 1a. Reconstruct the logical access from the current FX operation

Lines 435-461 create two loads of the same buffer with different index
expressions:

```text
load 0: buf0[d0]
load 1: buf0[d0 + 1]
```

The test drives
[`_logical_memory_access`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:1527>)
through the real `LoopBody` indexing table and verifies that the first result is
the expected normalized `MemoryDep` and differs from the second. This pins the
planner/codegen identity boundary: codegen recovers the node-local logical
access rather than keying a value by `"buf0"`.

### 1b. Same name, wrong consumer index is a loud error

Lines 462-466 register a relation for `buf0[d0]`, move the interpreter to
`buf0[d0 + 1]`, and require
[`get_relation`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:2479>)
to raise `no planned relation`.

This is the core regression tripwire for the old name-keyed behavior. A
same-name read must neither receive the first read's value nor silently reload
as though it had no staged relation.

### 1c. Record only exact planned source accesses

Lines 470-481 call
[`_record`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:2407>)
first with `buf0[d0 + 1]` and then with the planned `buf0[d0]`.

It verifies:

- an unplanned same-name source is ignored;
- the exact unguarded source is cached directly by normalized `MemoryDep`; and
- recording a replacement invalidates the materialized value associated with
  the old source value.

That last assertion prevents a newly produced source from reusing a split or
broadcast derived from a stale CSE value.

Lines 483-487 also pin the ownership boundary: a consumer with a concrete
`_load_other` receives no forwarded source and proceeds through normal load
codegen. Section 2 reviews that rule directly.

### 1d. Required sources never fall back

Lines 488-503 exercise two distinct misses for a relation with
`requires_live_source=True`:

- there is no recorded source value; and
- a source and old materialization exist, but `cse.contains_value()` says the
  source is stale.

Both must raise an error containing the source set and consumer access. The
second arm is load-bearing: checking the materialization memo before source CSE
liveness would resurrect a value across `cse.invalidate()`.

### 1e. Equivalent source alternatives and eager short-circuiting

Lines 505-516 construct one relation with two valid source alternatives.

- `resolve_load()` sees the first source but cannot materialize it, continues,
  and returns the second source.
- `materialize_sources()` stops after the first usable alternative instead of
  eagerly emitting equivalent splits or broadcasts for every source.

This pins two bugs found during adversarial review: choosing the first live but
unusable source, and materializing duplicate alternatives before first use.

### Evidence boundary

This one method covers many contracts compactly, but most subcases use a
partially constructed resolver and mocked materializer. The end-to-end tests and
kernel hash corpus establish that the same rules compose through real planning
and emission.

## 2. Existing masked-load ownership

Test:
[`test_sub_parent_resolver_uses_masked_load_ownership`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_inductor_scheduler.py:518>)

Review weight: **highest**. The important design decision is ownership, not a
second mask/fill compatibility system inside the resolver.

Lines 518-550 pin three rules:

1. A source emitted while `kernel._load_mask` is active is not recorded by the
   sub-parent resolver.
2. An unguarded source may be forwarded when `_load_other is None`. This covers
   unmasked consumers and the existing `TritonKernelOverrides.masked()` path
   that emits an outer `where` around an in-kernel value.
3. A consumer with a concrete `_load_other` does not use forwarding. The normal
   physical load retains responsibility for applying its mask and fill.

The implementation is concentrated in
[`_record`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:2407>)
and
[`resolve_sources`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:2492>).
There is no `_AccessGuard`, guard-keyed cache, or resolver-owned fill synthesis.

This matches the pre-F1 source resolver and ordinary Triton masked-load
behavior. A concrete-fill external consumer falls through to `kernel.load`; a
required in-kernel miss still fails loudly.

The decision is backed by an instrumented full nested-reduction run:

- 562 planned-source records: 556 unguarded and 6 guarded;
- 943 successful forwards, all from unguarded sources;
- 234 masked successful consumers, all with `_load_other=None`;
- 12 concrete-fill consumers, all optional external relations that already
  fell back; and
- no required relation attempted resolution with a concrete `_load_other`.

Under the simplified ownership rule, all 399 nested-reduction tests pass and
all ten protected kernel hashes remain byte-identical. See
[`f1_access_guard_reachability_20260827.md`](</data/users/eellison/pytorch/agent_space/f1_access_guard_reachability_20260827.md>).

## 3. External fallback and store modes

Test:
[`test_sub_parent_external_fallback_and_atomic_store`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_inductor_scheduler.py:552>)

Review weight: **highest** for the load half, **medium** for the representative
store-mode loop.

### 3a. A planned external miss performs a physical remapped load

Lines 553-574 model a consumer for which a plan relation exists but no live CSE
source is available. The remapped consumer index is `7`.

It requires
[`_PointwiseRemapHandler.load`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:2321>)
to call `kernel.load("buf0", 7)` directly and never call `inner.load`.

The distinction matters because `inner.load` may consult the generic
name-keyed `store_cache`; a same-name value stored at a different logical index
would be unsafe. The direct kernel load is the valid fallback only for an
external relation (`requires_live_source=False`).

The test proves the bypass by call routing rather than by explicitly poisoning
`store_cache`. It also does not separately assert the load-count or operation-
trace bookkeeping performed by that direct path.

### 3b. Atomic/TMA stores are not source values

Lines 575-580 call the resolver's `store()` for `atomic_add`, `atomic_xchg`, and
`tma`, and require that `_record()` is never called.

For an atomic store, the operand passed to `store` is not the resulting memory
value, so forwarding it as though it were the post-store buffer contents would
be wrong. The implementation excludes every non-`None` mode; these three modes
are representative tripwires rather than an exhaustive mode enumeration.

## 4. Shape-derived materialization ambiguity

Test:
[`test_group_width_equal_to_child_width_is_direct`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_inductor_scheduler.py:582>)

Review weight: **high**. F1 removes the explicit source-layout enum, so this
ordered shape dispatch is now part of correctness.

The test forces `child_block` and `num_groups_str` to have the same string value.
That occurs for the degenerate `group_size == factor` case: one child element is
already one group.

It verifies that
[`materialize_value_at_sub_parent_resolution`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:2074>)
checks child width before group width:

- the original value is returned directly;
- target-family masks are assigned; and
- no group-to-child broadcast is emitted.

Reversing the two checks would perform a semantically unnecessary reshape and
would make shape strings act like the deleted layout enum in the ambiguous case.

## 5. Existing builder tests renamed to the new record API

These tests remain valuable context, but their F1 diff does not add new cases.
Read the contract once; skim the symbol-level changes in the commit.

### 5a. Reduced/group broadcast relations

Test:
[`test_sub_parent_broadcast_access_relation_frame_contract`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_inductor_scheduler.py:671>)

Review weight for existing behavior: **high**. Review weight for this commit's
test hunk: **low/mechanical**.

Existing cases retained:

- same prefix shape and address mapping is accepted;
- a retained extra consumer axis that affects the index is rejected;
- a regrouped raw frame is rejected;
- normalized versions of the same row-major address map are accepted;
- multiple same-name writers are rejected; and
- a `StarDep` consumer is rejected.

F1 changes only the test name, local `projection` variable, and called helper
from `_sub_parent_broadcast_projections` to
`_sub_parent_broadcast_access_relations`. The test still checks only whether a
relation set exists, not the new record fields.

### 5b. Values produced inside the sub-parent epilogue

Test:
[`test_sub_parent_internal_access_relation_preserves_emission_frame`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_inductor_scheduler.py:722>)

Review weight for existing behavior: **high**. Review weight for this commit's
test hunk: **low/mechanical**.

Existing cases retained:

- same output-group producer/consumer accesses may match by normalized row-major
  emission ordinal;
- a value written in an earlier group may be trailing-broadcast into a later
  group;
- a wrong cross-group frame is rejected;
- duplicate writers are rejected; and
- both cross-group and same-group consumer-before-writer cases are rejected.

F1 changes the helper name and adapts to flattened per-consumer relation records.
The extra parentheses around `consumer_before_writer` are formatting only.

### Important review boundary in both builder tests

Neither builder unit test directly asserts the new payload:

- exact `source_accesses`;
- exact `consumer_access`;
- `parent_lane`; or
- `requires_live_source`.

Those fields are exercised indirectly by the resolver tests and integration
suite. A reviewer should therefore inspect their construction in
[`_try_get_sub_parent_access_relations`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/scheduler.py:1094>)
and
[`_sub_parent_broadcast_access_relations`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/scheduler.py:1373>)
rather than assuming the renamed builder tests validate every field.

## 6. Temporal names and mutation aliases

Test:
[`test_nested_dependency_matches_require_injective_producer`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_inductor_scheduler.py:898>)

Review weight: **highest for the new `planned_name="other"` row**; the dense vs
alias writer rows existed already.

The test at lines 894-940 stores raw temporal names in the plan and gives the scheduler this
alias map:

```text
buf   -> renamed_buf
other -> renamed_buf
```

The three rows mean:

| Writer | Planned temporal name | Result |
| --- | --- | --- |
| dense/injective | `buf` | accept and return the mutation-renamed pair to ordinary vertical legality |
| aliased/non-injective | `buf` | reject despite exact plan membership |
| dense/injective | `other` | reject even though `other` and `buf` collapse to the same final mutation name |

This pins the phase boundary in
[`_prove_staged_fusion_dependencies`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/scheduler.py:9387>):

- exact plan membership is checked using raw node-local temporal dependencies;
- mutation renames are applied only to the pair returned to ordinary fusion
  legality.

If the plan relation set were renamed before membership testing, the `other`
row would be incorrectly accepted.

## 7. Fusion ownership and unsafe writes: adapted fixtures

These are behavior-bearing tests worth keeping open while reading the fusion
proof, but their F1 diffs are fixture renames rather than new coverage.

### 7a. Scoped legacy parent-to-grouped proof

Test:
[`test_nested_dependency_matches_scope_index_equivalence`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_inductor_scheduler.py:943>)

It retains this deliberate boundary:

- `nested` ownership may use the inherited parent-to-grouped index-equivalence
  proof;
- `both` ownership takes the stricter sub-parent exact-relation path and rejects
  when no relation is present; and
- `none` rejects.

F1 only renames `source_projections` to `access_relations` and
`projected_access_pairs()` to `sub_parent_access_pairs()`. The retained positive
`nested` row is the explicit P2G follow-up boundary, not indexed-codegen
authorization.

### 7b. Unsafe source writes

Test:
[`test_planned_dependency_matches_reject_unsafe_write`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_inductor_scheduler.py:973>)

It continues to reject:

- multiple same-name writes;
- TMP/indirect indexing; and
- atomic writes.

F1 only adapts the mock plan field and pair iterator names. The test remains the
defensive gate showing that exact relation membership is necessary but not
sufficient: the source write must also be safe for index-equivalent forwarding.

## 8. End-to-end resolver scoping and external source behavior

Test:
[`test_producer_consumer_inlined_parent_full_source`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_nested_reduction.py:1401>)

Review weight: **highest for `shared_external_source=True`**. This is the only
substantive change in `test_nested_reduction.py` for F1.

### Existing arm: in-kernel full-resolution source

With `shared_external_source=False`, the RMSNorm output and group scale produce
an in-kernel full-resolution `source`; the sub-parent pack consumes its two
lanes. The test requires correct numerics, one fused staged kernel, and exactly
two `tl.split` calls.

This is the required-live-source path: codegen must resolve the in-kernel value
from a live exact source and may not reload it from memory.

### New arm: one external input used in both parent and sub-parent stages

With `shared_external_source=True`, input `x` is used while building realized
parent-stage value `z` and is also consumed directly by the sub-parent pack.
The scale comes from `z`, while the packed lanes come from `x`.

This arm was added after an adversarial failure in which source resolution was
enabled too broadly: the parent-stage read of `x` was mistaken for the planned
sub-parent consumer read and compilation raised `no planned relation`.

The regression requires:

- exact numerics;
- one fused staged kernel;
- source recording may remain active across parent/grouped emission;
- consumer resolution is active only during sub-parent output replay; and
- exactly one split, rather than the two splits used by the in-kernel-source
  arm.

The split count is a useful generated-form pin: it makes accidental eager
materialization or resolving the wrong stage observable even when numerics stay
correct.

## 9. Unchanged tests that complete the F1 story

These tests are not part of the F1 diff, so they do not need line-by-line
rereview. They are the important integration backstop for the new mechanism.

- Target-family masks on indirect gathers, standalone and nested:
  [`test_standalone_sub_parent_preserves_indirect_index_mask`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_nested_reduction.py:981>) and
  [`test_nested_sub_parent_preserves_indirect_index_mask`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_nested_reduction.py:997>).
- Mutation after the forwarded read retains temporal value identity:
  [`test_producer_consumer_sub_parent_source_mutated_later`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_nested_reduction.py:1243>).
- Shifted/transposed same-name accesses decline rather than inherit a relation:
  [`test_producer_consumer_rejects_shifted_sub_parent_intermediate`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_nested_reduction.py:1301>),
  [`test_producer_consumer_rejects_transposed_sub_parent_frame`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_nested_reduction.py:1320>), and
  [`test_producer_consumer_rejects_shifted_reduced_source`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_nested_reduction.py:1341>).
- Internal epilogue source forwarding:
  [`test_producer_consumer_sub_parent_intermediate`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_nested_reduction.py:1363>).
- Outer reduction/group-width broadcast:
  [`test_producer_consumer_broadcasts_outer_reduction_output`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_nested_reduction.py:1381>).
- Independent external sub-parent input:
  [`test_producer_consumer_independent_sub_parent_source`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_nested_reduction.py:1433>) and
  [`test_independent_sub_parent_source`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_nested_reduction.py:1675>).
- Mismatched masked source and the conservative masked group-source fallback:
  [`test_standalone_sub_parent_mismatched_masked_source`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_nested_reduction.py:2243>) and
  [`test_standalone_sub_parent_masked_group_source_falls_back`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_nested_reduction.py:2263>).
- Ambiguous same-buffer source load and shifted reduction output decline:
  [`test_standalone_sub_parent_rejects_ambiguous_source_load`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_nested_reduction.py:2350>) and
  [`test_standalone_sub_parent_rejects_shifted_reduction_output`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_nested_reduction.py:2369>).
- Persistent/looped split counts and MXFP6 output grouping:
  [`test_mxfp6_four_to_three_pack_kernel_form`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_nested_reduction.py:3524>),
  [`test_mxfp6_internal_source_kernel_form`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_nested_reduction.py:3538>), and
  [`test_standalone_sub_parent_epilogue_kernel_form`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_nested_reduction.py:3552>).

## 10. Mechanical test hunks to skim

The following diff lines do not independently change behavior:

- import additions for the focused scheduler tests (`contextlib`,
  `PropertyMock`, and resolver helpers);
- `ProjectedSourceAccess` import replaced by `SubParentAccessRelation`;
- `_sub_parent_broadcast_projections` renamed to
  `_sub_parent_broadcast_access_relations` in the existing frame test;
- `_sub_parent_internal_projections` renamed to
  `_sub_parent_internal_access_relations` in the existing emission-frame test;
- `projection` local renamed to `relation` in those tests;
- mock `source_projections` fields renamed to `access_relations`;
- mock `projected_access_pairs()` renamed to `sub_parent_access_pairs()`;
- formatter-only parentheses around `consumer_before_writer`; and
- line wrapping around the parameterized nested test's `FileCheck` call.

Do not skim two changes that sit beside this mechanical churn:

- the `planned_name="other"` mutation-collision row at scheduler-test lines
  894-940; and
- the `shared_external_source=True` integration arm at nested-test lines
  1401-1431.

## 11. Evidence outside the changed tests

### Full-file suites

The pre-simplification F1 snapshot was run with the full-package worktree
overlay, caches disabled:

- scheduler suite: 130 passed, 6 skipped;
- nested-reduction suite: 399 passed, 8 skipped.

The final masked-load ownership simplification was then run through the current
verification set:

- scheduler suite: 118 passed, 6 skipped;
- focused `sub_parent` scheduler selection: 18 passed;
- additional focused ownership/integration selection: 4 passed; and
- nested-reduction suite: 399 tests ran successfully, 8 skipped. An earlier AOT
  checksum error was traced to selecting a broken system `openssl` under the
  Conda library path.

The replacement scheduler test now pins masked-load ownership directly instead
of maintaining the former custom guard-compatibility matrix.

Commands and results are recorded in
[`indexed_forwarding_rebase_20260826.md`](</data/users/eellison/pytorch/agent_space/indexed_forwarding_rebase_20260826.md:115>).
This matters because an earlier narrow overlay omitted `codegen/common.py`; the
final results do not rely on that runner.

### Protected generated-kernel corpus

The current F1 review state matches all ten published normalized kernel hashes
and emits one staged kernel in every case. The corpus covers:

- factor-2 persistent and looped;
- MXFP6 4:3 persistent and looped;
- internal-source persistent and looped;
- reduced broadcast;
- scale swizzle;
- aligned preshuffle; and
- DCN preshuffle.

See
[`f1_f2a_protected_kernel_reattest_20260827.md`](</data/users/eellison/pytorch/agent_space/f1_f2a_protected_kernel_reattest_20260827.md:33>).
For F1 this is a behavior-preservation gate: a normalized kernel-text change is
more suspicious than a benchmark result because F1 is intended to replace the
identity mechanism without changing generated forms.

### Adversarial findings already converted to permanent tests

The current changes directly retain tripwires for four bugs found while F1
was being reviewed:

- resolver active outside the sub-parent output replay: the new shared external
  source arm;
- forwarding taking ownership from ordinary masked-load codegen: the masked
  ownership test plus the cat and mismatched-pad integration cases;
- first live but unmaterializable source preventing use of a later alternative:
  the source-alternative arm; and
- eager materialization of every equivalent source: the one-call assertion in
  `materialize_sources()`.

The investigation is recorded in
[`indexed_codegen_adversarial_review_20260826.md`](</data/users/eellison/pytorch/agent_space/indexed_codegen_adversarial_review_20260826.md:67>).

## 12. Remaining evidence limits and follow-ups

These are review boundaries, not a recommendation to grow this already-large
test diff indiscriminately.

### 12a. P2G remains deliberately outside F1

The exact records are authoritative for sub-parent forwarding and codegen, but
the inherited parent-to-grouped fusion relation still uses the scoped legacy
equivalence proof. The `nested` row in
[`test_nested_dependency_matches_scope_index_equivalence`](</data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_inductor_scheduler.py:943>)
intentionally pins that exception.

When P2G is resumed, this test should change so nested-stage residuals require
source-anchored raw `MemoryDepMatch` records, and relation-removal/axis-swap
mutants should become its acceptance gate. That work is not an F1 codegen gap.

### 12b. New builder payloads are indirectly rather than locally asserted

The renamed broadcast/internal builder tests assert acceptance and rejection,
but not the exact relation fields. `parent_lane` and
`requires_live_source` are instead covered by resolver behavior and end-to-end
kernel forms. During review, inspect the constructors rather than treating the
renamed unit tests as a complete field-level oracle.

### 12c. A few defensive assertion branches lack direct permanent tests

No changed test directly constructs:

- two incompatible relations for one normalized consumer access, which should
  trip `_SubParentValueResolver.__init__`;
- a tuple materialization without a lane or with an invalid lane; or
- malformed FX operations for every `_logical_memory_access` validation arm.

These are fail-loud defensive branches. The current permanent suite covers the
main miscompile risks: same-name/different-index lookup, stale/absent required
sources, masked-load ownership, unsafe stores, and external fallback.

### 12d. Scratch mutant-to-permanent-test mapping is not yet recorded

The independent review's final remaining evidence request is to run the
existing scratch mutants with only checked-in tests as oracles and record which
test kills each mutant. See
[`f1_f2a_stack_review_20260827.md`](</data/users/eellison/pytorch/agent_space/f1_f2a_stack_review_20260827.md:284>).

This is a verification/documentation item, not a request to commit the mutant
framework. If a mutant survives all checked-in tests, that result would identify
a real targeted gap; otherwise the current compact permanent suite is enough.

## 13. Bottom line for the human review

The new tests are concentrated rather than sprawling:

- one exact-access/liveness/alternatives test;
- one masked-load ownership test;
- one physical-fallback/store-mode test;
- one degenerate shape-dispatch test; and
- one new end-to-end source-scope arm.

The only substantive modification to an existing scheduler proof test is its
temporal-name parameter extension: the new collision row plus nonempty mutation
renames on the existing dense/alias rows. Everything else in the test diff is
primarily the rename from layout-grouped projections to flattened per-read
access relations.

The highest-value question is not whether every helper branch has a test. It is
whether these tests jointly make the old unsafe behaviors impossible to restore:
name-only lookup, stale CSE reuse, stealing mask/fill ownership from ordinary
load codegen, fallback through store-cache, atomic operand forwarding, and
resolver use outside sub-parent replay. They do.
