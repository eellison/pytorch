# F1 scheduler review walkthrough

Date: 2026-08-27

Commit under review: `6d545ab0fb9b292539e4cae4f19c4eb7ea10d42f`
against parent `d4167dba5b8755cb8ee3ec41d1af7852ab5b87d1`.

Scope: `torch/_inductor/scheduler.py` only. The scheduler portion is 131 added
lines and 181 deleted lines, net -50. This document is AI-assisted scratch
material for human review; it does not change production code.

## One-sentence model

The old scheduler said, roughly, "buffer `x` uses the INTERLEAVED/BROADCAST/
IDENTITY policy." F1 instead records, for each exact consumer `MemoryDep`, the
exact source `MemoryDep` alternatives, the selected parent lane when applicable,
and whether that source must still be live in registers.

```text
old: buffer name -> layout category -> grouped consumer list
new: exact consumer access -> exact source alternatives + lane + liveness
```

That one record is used twice:

1. fusion checks that every nonstandard producer/consumer dependency is an
   exact relation proved by the plan; and
2. codegen resolves the actual load against that same relation.

## Execution map

The main flow, including unchanged entry points for context, is:

```text
Scheduler._can_fuse
  |-- standalone: NestedReduction.sub_parent_epilogue_plan
  |-- nested:     NestedReduction.plan -> _plan_nested_sub_parent_stage
  |
  |   planners build three relation families
  |     1. parent-width lane selections
  |     2. reduced/group-width broadcasts
  |     3. values written and reread inside the sub-parent epilogue
  |
  `-- _prove_staged_fusion_dependencies
        |-- validate raw X/R frame for lane relations
        |-- require exact raw plan membership for sub-parent residuals
        `-- return renamed MemoryDepMatch pairs to ordinary scoring/legality
```

The unchanged dispatch is visible at
[`Scheduler._can_fuse_impl`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/scheduler.py:9820):
standalone planning is called at line 9837 and the dependency proof at line
9870. The proof result is consumed by ordinary vertical legality at
[`_can_fuse_vertical_impl`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/scheduler.py:9980)
and staged reuse scoring at
[`_score_staged_fusion_memory_for_can_fuse`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/scheduler.py:10316).
Those entry/consumer paths predate F1; they explain where the changed scheduler
record goes.

## Recommended reading order

1. Read the new relation type at
   [`SubParentAccessRelation`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/scheduler.py:2095).
2. Read the parent-lane builder at
   [`_try_get_sub_parent_access_relations`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/scheduler.py:1094).
3. Read the fusion proof at
   [`_prove_staged_fusion_dependencies`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/scheduler.py:9387).
4. Then skim the internal and broadcast builders at lines 1266 and 1373.
5. Finally skim the standalone/nested assembly call sites at lines 636 and
   1599, plus the deleted compatibility surface described below.

Steps 1-3 contain the new correctness contract. Most other scheduler changes
are a representation rename or deletion.

## 1. Record contract

**Classification: REPLACED; high scrutiny.**

Read
[`SubParentAccessRelation`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/scheduler.py:2095).

Before F1:

```python
ProjectedSourceAccess(
    sources=(...),
    consumers=(...),
    layout=INTERLEAVED | BROADCAST | IDENTITY,
)
```

After F1:

```python
SubParentAccessRelation(
    source_accesses=(...),
    consumer_access=...,
    parent_lane=0 | 1 | ... | None,
    requires_live_source=True | False,
)
```

Field meanings:

| Field | Meaning and invariant |
| --- | --- |
| `source_accesses` | One or more equivalent raw source accesses. They all name the same buffer and give codegen alternatives when more than one exact access has a live CSE value. The tuple must be nonempty. |
| `consumer_access` | One exact raw `MemoryDep` for one epilogue load. F1 deliberately flattens the old many-consumer record into one record per consumer. |
| `parent_lane` | A statically proved lane for a parent-width value. For factor 2 it is normally `0` or `1`. `None` means no parent-lane selection; codegen distinguishes direct child-width use from group-width broadcast using the live value shape. |
| `requires_live_source` | `True` for a source written inside this kernel. Such a source cannot silently reload from memory; a missing live CSE value is a compiler error. `False` permits an external input to fall back to an ordinary load. |

`__post_init__` checks only the structural minimum: nonempty source alternatives
and one shared buffer name. The builders below own the stronger facts: lane
range, ordering, exact index relation, and liveness classification. Review those
builders together with this type rather than treating the dataclass validation
as the whole contract.

Concrete factor-2 example:

```text
parent source:   MemoryDep("x", 16*x + r,       ranges=[B, 16])
even consumer:  MemoryDep("x", 16*x + 2*child, ranges=[B, 8])
odd consumer:   MemoryDep("x", 16*x + 2*child + 1, ranges=[B, 8])

relations:
  (source -> even, parent_lane=0)
  (source -> odd,  parent_lane=1)
```

If `x` is an external argument, `requires_live_source=False`; if the same
access is produced by an earlier fused node, it is `True`.

## 2. Deleted layout and name API

**Classification: DELETED; important simplification.**

The old `NestedReduction.SubParentSourceLayout` enum and its explanatory note
were between `GroupedAxis` and `PointwiseDomainContext`. Their deletion is
visible at the current adjacency between
[`GroupedAxis`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/scheduler.py:580)
and
[`PointwiseDomainContext`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/scheduler.py:584).

The stage now stores `access_relations` at
[`SubParentEpilogueStage`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/scheduler.py:2157).
F1 deletes all three name-based compatibility views that used to follow
`epilogue_nodes`:

- `source_layouts`
- `broadcast_source_names`
- `internal_dependency_names`

This matters more than the enum rename. No scheduler/codegen boundary now says
"all loads of this name get one treatment." The boundary carries exact reads.
The current stage ends directly after `epilogue_nodes` and validation at lines
2158-2172.

The corresponding plan iterator is
[`StagedReductionPlan.sub_parent_access_pairs`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/scheduler.py:2211).
It is a rename and simplification of `projected_access_pairs`: each relation has
one consumer, so it only expands source alternatives against that consumer.

What to verify here:

- no old layout/name view is still required by scheduler callers;
- one consumer cannot inherit another consumer's relation merely because the
  buffer name matches; and
- `parent_lane=None` is intentionally not a replacement layout enum. The codegen
  resolver receives the exact relation and derives direct versus broadcast
  materialization from the live value shape at
  [`_SubParentValueResolver`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:2401).

## 3. Standalone plan assembly

**Classification: RENAMED/PLUMBING; low scrutiny.**

Read the changed block in
[`NestedReduction.sub_parent_epilogue_plan`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/scheduler.py:636),
especially lines 704-752.

The existing planner still:

1. identifies output groups;
2. partitions parent-width and reduced-width source names;
3. proves internal, parent-lane, and broadcast relations;
4. orders parent nodes needed after the reduction; and
5. returns one `StagedReductionPlan`.

F1 only changes the representation flowing through those steps:

```text
internal_projections  -> internal_relations
source_projections    -> source_relations
broadcast_projections -> broadcast_relations
stage.source_projections -> stage.access_relations
```

`planned_source_names` at lines 724-726 now reads the first
`source_accesses` entry. That is safe only because the relation dataclass
requires every source alternative to share one name. Candidate classification,
parent scheduling, and leaf-output checks are unchanged.

## 4. Parent-width lane relations

**Classification: RENAMED plus the main new planner semantics; high scrutiny.**

Read
[`_try_get_sub_parent_access_relations`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/scheduler.py:1094).

Most of this function is inherited. In particular, the local
`normalized_source_accesses` helper, common parent/child frames, divisibility
substitutions, unique normalized parent index, and
`parent_r = factor * child_r + lane` proof are not new.

The new work begins at
[`parent_writes`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/scheduler.py:1185):

1. Collect exact raw parent writes.
2. Preserve the raw source deps separately from their normalized indices at
   lines 1203-1207.
3. Set `requires_live_source` when any equivalent source access is one of those
   writes at line 1208.
4. For every child read, retain the already-proved integer lane and emit one
   `SubParentAccessRelation` at lines 1231-1237.

The old function emitted one `INTERLEAVED` projection per buffer name and
grouped all consumers inside it. The new function emits one exact relation per
consumer and deduplicates identical records with `OrderedSet`.

Why raw source identity is retained:

```text
source_deps:    original MemoryDep objects used by planning/codegen
parent_indices: their common normalized row-major expression used for proof
```

The normalized expression proves geometry; the raw dep identifies the actual
load/store that codegen must find. Substituting one for the other would recreate
the mismatch between fusion proof and codegen lookup that F1 is removing.

Review these failure boundaries:

- no parent access for a child-read name -> decline;
- multiple normalized parent indices for one name -> decline;
- lane cannot be proved as one static value in `[0, factor)` -> decline;
- substituted child index differs from the parent index -> decline.

## 5. Internal epilogue relations

**Classification: RENAMED plus flattened representation; medium scrutiny.**

Read
[`_sub_parent_internal_access_relations`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/scheduler.py:1266).

The proof itself is inherited:

- collect all epilogue writes by name;
- accept only one writer for an internally consumed name;
- require the writer to precede the consumer in `(group_index, node_index)`
  order;
- within one output group, require normalized access equality;
- across later output groups, require a trailing-axis broadcast.

Example of a later-group broadcast:

```text
group 1 write: MemoryDep("scale", g, ranges=[G])
group 3 read:  MemoryDep("scale", g, ranges=[G, 3])
```

The extra lane axis does not affect the address, so the later replay may reuse
the value. A read such as `3*g + lane` is rejected.

F1 changes the result from one `IDENTITY` record holding all reads to one
relation per read at lines 1315-1321. These are necessarily internal values, so
`requires_live_source=True`; they do not select a parent lane, so
`parent_lane=None`.

The rewrite at lines 1288-1301 from nested positive conditionals to early
`continue`/decline is mechanical clarification. Its acceptance set is intended
to be unchanged.

## 6. Reduced/group-width broadcast relations

**Classification: RENAMED plus flattened representation; medium scrutiny.**

Read
[`_sub_parent_broadcast_access_relations`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/scheduler.py:1373).

The accepted relation is:

```text
source:   MemoryDep("scale", G*x + g, ranges=[B, G])
consumer: MemoryDep("scale", G*x + g, ranges=[B, G, lanes])
```

The consumer may have trailing axes only when they do not affect its index.
The actual frame proof remains in
[`_sub_parent_consumer_is_trailing_broadcast`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/scheduler.py:1326)
and is inherited from the parent commit.

F1's substantive change is lines 1417-1425: instead of one `BROADCAST`
projection containing all consumers, emit one exact relation per consumer with
`parent_lane=None` and `requires_live_source=True`.

The unique-writer check and the adjacent raw/normalized stability checks at
lines 1394-1416 belong to the base correctness fix, not to F1. In this F1 review,
verify that flattening preserves their accepted consumer set; do not re-review
their frame mathematics as new F1 code.

## 7. Nested plan assembly

**Classification: RENAMED/PLUMBING; low scrutiny.**

Read
[`_plan_nested_sub_parent_stage`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/scheduler.py:1599),
specifically lines 1640 and 1656-1715.

This is the nested-topology twin of step 3. The domain classification and name
partition remain unchanged. Names still identify which proof family to try;
the resulting exact accesses, rather than names, cross the planner/codegen
boundary. The updated comment at lines 1656-1657 states that distinction.

The only output change is construction of
`SubParentEpilogueStage(access_relations=(...))` from the same three relation
families. The call into this builder from the larger nested plan is unchanged at
[`NestedReduction.plan`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/scheduler.py:1926).

## 8. Fusion authorization

**Classification: main correctness change; highest scrutiny.**

Read
[`_prove_staged_fusion_dependencies`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/scheduler.py:9387).

### 8a. Raw X/R boundary proof

Lines 9390-9443 select relations with `parent_lane is not None`, replacing the
old `layout is INTERLEAVED` filter. The same raw-frame proof then runs on every
source alternative and the relation's one consumer.

This is representation-equivalent, not a broader proof: a lane witness exists
only when the parent-lane builder already proved the interleaving equation.

### 8b. Keep raw and mutation-renamed writes

Lines 9445-9449 now store `(raw_write, renamed_write)` for each producer write.
The two identities serve different layers:

- raw deps are compared with the planner's temporal records;
- renamed deps are returned to ordinary fusion legality, which operates after
  mutation-name resolution.

### 8c. Build exact raw relation membership

Lines 9451-9454 build `relation_matches` directly from
`plan.sub_parent_access_pairs()`. The parent implementation renamed both sides
before building this set. F1 deliberately does not.

Concrete collision caught by this change:

```text
actual raw producer/read name: "buf"
incorrect planned relation name: "other"
mutation_renames:
    "buf"   -> "renamed_buf"
    "other" -> "renamed_buf"
```

Comparing only renamed pairs would make the incorrect relation look exact.
Comparing `(raw_write, raw_read)` with the raw plan rejects it.

### 8d. Prove each residual read

The residual loop is lines 9477-9509:

1. Ignore non-`MemoryDep` reads here; ordinary legality still handles their
   ordering semantics.
2. Ignore reads that are not producer outputs.
3. Require one producer write for the name; reaching-write analysis for
   multiple same-name writes is intentionally absent.
4. Let ordinary exact `fusable_read_and_write` matches pass first.
5. For a sub-parent-owned residual, require the exact raw relation at lines
   9494-9499 and independently require
   `_memory_dep_supports_index_equivalence` safety on the raw deps.
6. Only an inherited grouped-stage read may use the legacy normalized
   equivalence proof at lines 9502-9506.
7. Return the renamed `MemoryDepMatch` so the existing scoring and vertical
   legality consumers can remove that dependency.

The safety helper remains necessary even after exact plan membership. The plan
proves the source/consumer index relation; the helper separately excludes TMP
indices, synchronization-requiring writes, non-injective writers, and name
mismatches.

Ownership ordering is important: if a read is both sub-parent-owned and
nested-stage-owned, the sub-parent branch runs first and requires exact relation
membership. It cannot fall through to the legacy proof.

## 9. Nested append comment

**Classification: COMMENT ONLY; skip after confirming no code changed.**

At
[`FusedNestedReductions._plan_fusion_with`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/scheduler.py:4252),
the comment at lines 4263-4264 changes "projected dependencies" to "exact
sub-parent access relations." Control flow and values are unchanged.

## 10. Hunk inventory

| Current location | Change kind | Review weight |
| --- | --- | --- |
| lines 580-584 | delete `SubParentSourceLayout` and its note | High: confirms removal of layout policy |
| lines 704-752 | standalone variable/field plumbing | Low |
| lines 1094-1239 | rename builder; add raw sources, lane, liveness, per-read records | Highest |
| lines 1266-1323 | rename internal builder; flatten records | Medium |
| lines 1373-1426 | rename broadcast builder; flatten records | Medium |
| lines 1640, 1656-1715 | nested assembly plumbing/comment | Low |
| around line 1940 | remove one blank line | Mechanical |
| lines 2095-2114 | replace record type | Highest |
| lines 2151-2172 | rename stage field; delete three compatibility properties | High |
| lines 2205-2210 | rename/simplify pair iterator | Low |
| lines 4263-4264 | comment terminology | Mechanical |
| lines 9381-9509 | raw relation membership and renamed downstream matches | Highest |

## 11. Tests to use while reviewing scheduler semantics

- Exact access identity, source alternatives, and loud required-source misses:
  [`test_sub_parent_access_identity_and_cache_key`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_inductor_scheduler.py:438).
- External fallback and exclusion of atomic/TMA stores from forwarding:
  [`test_sub_parent_external_fallback_and_atomic_store`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_inductor_scheduler.py:602).
- Broadcast-frame acceptance, extra-axis/regroup rejection, unique writer, and
  non-`MemoryDep` rejection:
  [`test_sub_parent_broadcast_access_relation_frame_contract`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_inductor_scheduler.py:721).
- Internal writer ordering and same-group versus later-group access proofs:
  [`test_sub_parent_internal_access_relation_preserves_emission_frame`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_inductor_scheduler.py:772).
- Raw temporal identity versus mutation-renamed aliases, plus producer
  injectivity:
  [`test_nested_dependency_matches_require_injective_producer`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_inductor_scheduler.py:944).
- Ownership precedence for nested-only, dual-owned, and unowned reads:
  [`test_nested_dependency_matches_scope_index_equivalence`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_inductor_scheduler.py:992).
- Multiple writes, TMP indices, and atomic writes fail closed:
  [`test_planned_dependency_matches_reject_unsafe_write`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_inductor_scheduler.py:1022).
- End-to-end external versus in-kernel source behavior:
  [`test_producer_consumer_inlined_parent_full_source`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_nested_reduction.py:1401).

The guard/cache-key tests at lines 528-600 and the direct child-width test at
632-654 primarily review the codegen half of F1. They are useful contract tests,
but they need not be read deeply for the scheduler-only pass.

The builders' new `parent_lane` and `requires_live_source` payloads are checked
mainly through those resolver and end-to-end tests, rather than by direct field
assertions in each builder unit test. Review their construction at lines
1208-1237 explicitly.

## 12. What can be skimmed

These changes do not alter scheduler acceptance on their own:

- projection-to-relation local variable renames in both plan assembly paths;
- `source_projections` to `access_relations` field plumbing;
- the internal builder's early-continue rewrite;
- the single formatting change in the broadcast predicate;
- the blank-line deletion before `StagedReductionPlan` construction;
- the nested-append comment update; and
- test fixture renames from `source_projections`/`projected_access_pairs` to
  `access_relations`/`sub_parent_access_pairs` where assertions are unchanged.

Do not skim the flattened record creation merely because it appears next to
those renames: explicit `parent_lane`, exact `consumer_access`, and
`requires_live_source` are the functional payload of F1.

## 13. Final review questions

The scheduler review is complete when these questions have clear answers:

1. Does every lane-selecting consumer get its own exact raw relation and static
   lane witness?
2. Can any external source be incorrectly marked required, or any in-kernel
   write be incorrectly allowed to reload?
3. Can mutation renaming make an unplanned raw access appear planned?
4. Can a sub-parent-owned read escape into the legacy nested equivalence path?
5. Did removal of the three name views leave any scheduler or codegen caller
   authorizing by name alone?
6. Do standalone and nested plan assembly feed the same relation contract
   without changing their pre-F1 classification or ordering rules?

The highest-value review is questions 1-4. The remaining scheduler diff is
mostly evidence that the old layout/name compatibility layer is actually gone.
