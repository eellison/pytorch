# F1 Option C reviewer guide: exact scheduler proof, per-name replay

Date: 2026-08-27

This is an AI-assisted local review guide. It is not intended to be pasted into
GitHub without human review and the disclosure required by
[`AI_POLICY.md`](/data/users/eellison/pytorch/AI_POLICY.md:1).

This guide supersedes the exact-runtime-access description in
[`f1_detailed_review_guide_20260827.md`](/data/users/eellison/pytorch/agent_space/f1_detailed_review_guide_20260827.md:1).
That older guide describes `_logical_memory_access`, `get_relation`, and caches
keyed by `MemoryDep`. Option C deletes those runtime mechanisms. The scheduler
still proves exact access relations, but codegen validates and collapses their
consequences into a small contract keyed by buffer name.

## Snapshot and scope

Worktree:

`/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt`

Current amended F1 commit:

`859385382d515b81c551a60faac10a749491e377`

Prior exact-runtime Option A snapshot:

`fc09fb38287d4639df6424c5c33cfb9749b000ba`

Local review-base parent:

`d4167dba5b8755cb8ee3ec41d1af7852ab5b87d1`

The worktree is clean. Option C is now folded into `HEAD`. The rewrite from the
prior exact-runtime snapshot, `git diff fc09fb38287..HEAD`, has SHA-256
`87ae1da212b12ed79de50624a11000d464c7c9de4f9f08711d00a4f893aa4f6e` and
changes:

- [`torch/_inductor/codegen/simd.py`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:1519)
- [`torch/_inductor/scheduler.py`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/scheduler.py:2095), documentation only
- [`test/inductor/test_inductor_scheduler.py`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_inductor_scheduler.py:433)
- [`test/inductor/test_nested_reduction.py`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_nested_reduction.py:3386)

The complete amended F1 change, `git diff HEAD^ HEAD`, has SHA-256
`3fe900695c8c5eb0a21058a1e27614b161d644ce4b99fc04ed4d6473c8597d6a`.
It also includes the exact scheduler proof in
[`torch/_inductor/scheduler.py`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/scheduler.py:1094).
All line anchors in this guide were revalidated against `859385382d5`.

Use both views during review:

```bash
cd /data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt

# Option C versus the exact-runtime-access implementation
git diff fc09fb38287..HEAD -- torch/_inductor/scheduler.py
git diff fc09fb38287..HEAD -- torch/_inductor/codegen/simd.py
git diff fc09fb38287..HEAD -- test/inductor/test_inductor_scheduler.py
git diff fc09fb38287..HEAD -- test/inductor/test_nested_reduction.py

# Complete F1 versus its review base
git diff HEAD^ HEAD -- torch/_inductor/scheduler.py
git diff HEAD^ HEAD -- torch/_inductor/codegen/simd.py
git diff HEAD^ HEAD -- test/inductor/test_inductor_scheduler.py
git diff HEAD^ HEAD -- test/inductor/test_nested_reduction.py
```

The Option C rewrite totals `296 insertions, 248 deletions`: production is
`simd.py +123/-139` plus a documentation-only `scheduler.py +5/-4`; tests are
`test_inductor_scheduler.py +152/-96` and `test_nested_reduction.py +16/-9`.

The complete F1 commit totals `759 insertions, 447 deletions`: production is
`scheduler.py +132/-181` and `simd.py +302/-210`; tests are
`test_inductor_scheduler.py +291/-40` and `test_nested_reduction.py +34/-16`.

## The central review model

There are two deliberately different contracts:

| Layer | Representation | What it proves or does |
| --- | --- | --- |
| Scheduler | Exact `SubParentAccessRelation` per consumer `MemoryDep` | Authorizes every nonstandard source-to-consumer mapping, including frame preservation, lane, write identity, and liveness role. |
| Codegen | `_SubParentSourceContract` per buffer name | Captures the right kind of emitted value, derives the lane from the actual replay index, materializes the value, and either forwards, reloads, or fails loudly. |

The soundness claim is not "a buffer name identifies an access." It is:

```text
the planner exhaustively proves every tracked access
        +
codegen rejects inconsistent per-name consequences
        +
replay derives and validates the actual parent lane when a split is needed
        =
a name is sufficient as the runtime lookup key for this specialized stage
```

Exact access identity remains the fusion authorization boundary. The name-keyed
contract is only a codegen consequence of that proof. Review those layers
separately; treating the codegen name lookup as an independent alias analysis is
the wrong model.

## Concrete factor-4 example

Suppose the scheduler proves two child reads of `x`:

```text
parent source: x[4 * child_r + lane]
child read A:  x[4 * child_r + 0]  -> lane 0
child read B:  x[4 * child_r + 2]  -> lane 2
```

The scheduler retains two exact relations. Codegen validates that they have the
same source set and source role, then builds:

```text
x -> {source_is_internal: ..., parent_lanes: {0, 2}}
```

When replay executes a load, codegen computes `index % 4` after the required
symbolic extent substitutions. Lane 0 selects split part 0, lane 2 selects split
part 2, and lane 1 raises as unplanned. Codegen does not reconstruct the current
FX operation's `MemoryDep`.

For a scalar, singleton, child-width value, or group-width broadcast, the
materialized result is not a tuple and no lane selection is needed. Its safety
comes from the scheduler's exhaustive relation proof and the constructor's
per-name consistency checks.

## Recommended review order

1. Read the exact scheduler record at
   [`SubParentAccessRelation`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/scheduler.py:2095).
2. Read the three relation builders: parent-lane
   [`_try_get_sub_parent_access_relations`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/scheduler.py:1094), reduced/group
   [`_sub_parent_broadcast_access_relations`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/scheduler.py:1373), and in-epilogue
   [`_sub_parent_internal_access_relations`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/scheduler.py:1266).
3. Read the cross-domain fusion proof at
   [`_prove_staged_fusion_dependencies`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/scheduler.py:9388).
4. Read Option C's contract construction at
   [`_SubParentValueResolver.__init__`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:2338).
5. Read capture and last-writer behavior at
   [`_record`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:2395).
6. Read replay resolution at
   [`materialize_source`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:2483) and
   [`resolve_load`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:2520).
7. Read fallback and mask ownership at
   [`_PointwiseRemapHandler.load`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:2285) and
   [`resolve_sources`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:2471).
8. Read the focused contract tests beginning at
   [`test_sub_parent_resolver_rejects_inconsistent_name_contract`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_inductor_scheduler.py:433).

## 1. Scheduler: exact relations remain the proof

[`SubParentAccessRelation`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/scheduler.py:2095)
contains four facts:

| Field | Scheduler meaning | Codegen consequence |
| --- | --- | --- |
| `source_accesses` | Exact raw-frame source aliases that were proved equivalent for this relation. | All relations for one name must present the same source set. Values are captured by operation role, not by reconstructing these accesses. |
| `consumer_access` | The exact child read covered by the proof. | Its name groups the runtime contract. Its normalized identity is retained only to reject conflicting lane assignments. |
| `parent_lane` | The statically proved lane for a parent-width projection, or `None` for direct/broadcast use. | All non-`None` lanes become the allowed lane set. Direct and lane relations may not mix for one name. |
| `requires_live_source` | `True` when the value is produced inside the kernel and cannot safely reload; `False` for an external source. | Selects store capture versus load capture, and fail-loud versus physical fallback. |

The dataclass validates that the relation is nonempty and all accesses share a
name at
[`__post_init__`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/scheduler.py:2116).
Semantic validity is established by the builders and fusion proof.

### Parent-width lane relations

[`_try_get_sub_parent_access_relations`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/scheduler.py:1094)
is the main lane proof.

It first proves divisibility of the parent reduction extent by the sub-parent
factor at
[`try_get_sub_parent_extent_subs`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/scheduler.py:1013).
It then normalizes child accesses into `(x, child_r)` and parent accesses into
`(x, parent_r)` while retaining each original `MemoryDep` beside its normalized
index at
[`normalized_source_accesses`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/scheduler.py:1124).

For each tracked name, the builder requires all parent-side source accesses to
normalize to one parent index at
[`scheduler.py:1200`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/scheduler.py:1200).
For every child read, it then:

1. derives a static lane with
   [`interleaved_sub_parent_lane`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/scheduler.py:1030);
2. proves the lane is an integer in `[0, factor)` at
   [`scheduler.py:1216`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/scheduler.py:1216);
3. substitutes `parent_r = factor * child_r + lane` into the parent index; and
4. requires exact symbolic equality with the child index at
   [`scheduler.py:1226`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/scheduler.py:1226).

`requires_live_source` is derived from exact source membership in the parent
writes at
[`scheduler.py:1185`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/scheduler.py:1185).
Thus an external input relation is optional at runtime, while a parent-produced
intermediate is required.

### Reduced/group-width broadcast relations

[`_sub_parent_broadcast_access_relations`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/scheduler.py:1373)
requires one writer for the name, requires every consumer to be a `MemoryDep`,
and proves every read preserves the source prefix and address while adding only
trailing broadcast axes. The raw-frame proof is at
[`_sub_parent_consumer_is_trailing_broadcast`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/scheduler.py:1326).
When loop ordering after fusion is enabled, the same relation must survive the
normalized form at
[`scheduler.py:1403`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/scheduler.py:1403).

These relations have `parent_lane=None` and `requires_live_source=True`: the
writer is inside the kernel, and codegen determines from the live shape whether
the value is already child-width or must be broadcast from group width.

### Values produced inside the sub-parent epilogue

[`_sub_parent_internal_access_relations`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/scheduler.py:1266)
proves forwarding between epilogue nodes. It requires exactly one writer per
name, requires the writer to precede every reader in emission order, requires
same-group accesses to normalize identically, and permits later output groups
only through the trailing-broadcast proof. These relations are also direct and
required: `parent_lane=None`, `requires_live_source=True`.

### Exhaustive stage assembly

Standalone planning partitions names into parent-width and reduced/broadcast
sources, builds all three relation families, and stores them on one stage at
[`sub_parent_epilogue_plan`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/scheduler.py:636),
especially
[`scheduler.py:676`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/scheduler.py:676) through
[`scheduler.py:751`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/scheduler.py:751).

Nested planning performs the corresponding partition and builds the final stage
at
[`_plan_nested_sub_parent_stage`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/scheduler.py:1599),
especially
[`scheduler.py:1656`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/scheduler.py:1656) through
[`scheduler.py:1715`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/scheduler.py:1715).

This exhaustiveness is the premise Option C consumes. A planned name cannot
have an unexamined epilogue read and still be a valid stage.

### Cross-domain fusion authorization

[`StagedReductionPlan.sub_parent_access_pairs`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/scheduler.py:2212)
flattens every exact source/consumer pair. The fusion proof first checks that
lane relations preserve the raw X/R boundary at
[`scheduler.py:9394`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/scheduler.py:9394).
This closes the gap left by row-major normalization, which alone can erase axis
identity.

For each non-exact producer-output dependency, the proof then requires:

- one unambiguous producer write for the name;
- exact raw pair membership in the plan;
- `_memory_dep_supports_index_equivalence`, which rejects indirect indexes,
  synchronization-requiring modes, and non-injective producer writes; and
- ordinary vertical legality for every unrelated dependency.

The key checks are at
[`scheduler.py:9452`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/scheduler.py:9452) through
[`scheduler.py:9516`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/scheduler.py:9516),
with the safety predicate at
[`_memory_dep_supports_index_equivalence`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/scheduler.py:10176).
The resulting exact matches are passed into ordinary vertical fusion at
[`_can_fuse_vertical_impl`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/scheduler.py:9981).

The review question here is strict: does every relaxed dependency correspond to
one exact relation already proved by the stage? Codegen's later name lookup must
not be used to justify fusion.

## 2. Codegen: validate and collapse to one contract per name

The runtime record is intentionally small:

[`_SubParentSourceContract`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:1519)

```python
source_is_internal: bool
parent_lanes: frozenset[int] | None
```

[`_SubParentValueResolver.__init__`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:2338)
groups the final scheduler relations by consumer name and rejects any collapse
that would lose a semantically relevant distinction:

| Check | Failure it prevents |
| --- | --- |
| One normalized exact consumer cannot have two lanes | Contradictory scheduler facts hidden by taking only a lane set. |
| One source-access set per name | Two different logical sources cannot share one capture bucket. |
| One `requires_live_source` value per name | Loads and stores cannot compete as producers for the same contract. |
| No mixture of `parent_lane=None` and integer lanes | Direct/broadcast materialization cannot be confused with parent splitting. |

Those checks are concentrated at
[`simd.py:2353`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:2353) through
[`simd.py:2391`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:2391).
Different exact consumers of one name may have different lanes; their union is
the approved set. Source values and materializations are then cached by name and
CSE variable, not by `MemoryDep`, at
[`simd.py:2392`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:2392).

This is the main Option C review boundary. The constructor does not re-prove
index equivalence. It checks that the exact proof has one coherent set of
runtime consequences.

## 3. Capture roles

[`_record`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:2395)
uses `source_is_internal` to choose which emitted operation may populate a
contract:

| Contract role | Accepted operation | Runtime behavior |
| --- | --- | --- |
| Optional external (`False`) | `load` | Keep every distinct live CSE value. Equivalent parent/group/child-shaped loads may provide useful alternatives. Ignore stores. |
| Required internal (`True`) | ordinary `store` or `store_reduction` | Keep only the most recent value. Ignore loads. |

Unplanned names are ignored. Any operation emitted while `_load_mask` is active
is ignored so the resolver never assumes ownership of a masked source's fill
semantics. Ordinary stores are recorded only when `mode is None` at
[`simd.py:2421`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:2421).
Atomic and TMA-style modes expose an update operand or special operation, not a
plain post-store memory value, so they are not captured. Reduction stores are
captured explicitly at
[`simd.py:2434`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:2434).

The role split is stronger than merely preferring one candidate. A required
internal contract cannot accidentally capture an earlier external load of the
same name, and an optional external contract cannot capture a same-name store.

## 4. Replay lanes are derived from the emitted load index

Sub-parent bodies are replayed by output group and output lane at
[`_codegen_sub_parent_output_groups`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:3777).
[`sub_parent_iteration_values`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:1970)
constructs the concrete derived coordinates for that replay.

When a replayed load names a planned source,
[`_PointwiseRemapHandler.load`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:2285)
passes the actual emitted index to `resolve_load`. If materialization produced a
tuple of parent-width lane parts,
[`materialize_source`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:2483):

1. calls the same scheduler lane helper with the codegen family's extent
   substitutions and source sizes;
2. finds a statically equal member of the planner-approved lane set;
3. raises `unplanned lane` if none exists; and
4. selects exactly that tuple part.

The codegen family records the necessary dynamic/opaque extent substitutions at
[`make_sub_parent_family`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:1934).
The shared lane helper strips `Identity`, expands known size forms, applies
divisibility substitutions, and returns the simplified modulo at
[`interleaved_sub_parent_lane`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/scheduler.py:1030).

This is the runtime protection Option C retains from exact relations: it does
not identify the whole consumer `MemoryDep`, but it does not blindly select a
lane from the name either.

## 5. Shape-driven materialization

[`materialize_value_at_sub_parent_resolution`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:2038)
adapts a live CSE value according to its actual shape:

| Live shape on the parent axis | Materialization |
| --- | --- |
| Scalar `()` | Return directly. |
| Singleton `1` | Return directly; Triton broadcasting supplies the invariant axis. |
| Child width `parent_block / factor` | Return directly and attach derived-family masks. |
| Group width `num_groups` | Broadcast each group value across `group_size / factor` child elements, then attach masks. |
| Full parent width `parent_block` | Reshape to expose a trailing factor dimension, split into `factor` values, and attach masks to every part. |
| Unknown or incompatible | Return `None`. |

The child-width branch intentionally precedes the group-width branch at
[`simd.py:2059`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:2059).
When `group_size == factor`, those widths are equal and the value is already at
the correct resolution; broadcasting it would be wrong or redundant.

[`_materialize`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:2439)
first requires `kernel.cse.contains_value(value)`, then caches the result by CSE
variable. A Python reference in `_values` is therefore not enough to resurrect a
value after CSE invalidation. The liveness predicate includes ordinary CSE,
store cache, and reduction cache at
[`CSE.contains_value`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/common.py:2138).

[`resolve_load`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:2520)
tries each live source alternative until one materializes. This matters for
external values that may have been emitted in more than one equivalent shape.

### Early preservation boundaries

Some parent-width internal values must be split before a body flush discards the
parent-resolution CSE. [`materialize_sources`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:2455)
materializes once per name and raises only for an internal contract.

The nested pipeline offers all lane relations before sub-parent replay at
[`simd.py:3512`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:3512).
The standalone pipeline eagerly preserves only required lane relations at
[`simd.py:3840`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:3840) and
[`simd.py:3910`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:3910).
Optional external values may expire and reload. Direct and broadcast relations
remain lazy until their first consumer.

## 6. Optional fallback versus required failure

[`resolve_load`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:2520)
has two outcomes after it exhausts live/materializable values:

- Internal contract: raise `sub-parent stage lost required source`. Reloading is
  unsafe because the intermediate may be eliminated and cross-thread ordering
  is not established.
- External contract: return `None`, allowing a physical load in the derived
  iteration domain.

The external fallback goes through
[`_load_without_store_forwarding`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:2303),
not the ordinary wrapped `CSEProxy.load`. This is essential because generic
store forwarding is keyed only by name at
[`CSEProxy.load`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/common.py:3001).
A same-name cached store may represent another logical index, so a planned miss
must perform a real remapped load.

The local fallback preserves the other ordinary load behaviors: invalidated
stores add `must_keep_buffers`, TMP indexes use `indirect_load`, physical loads
update load counts, and operation tracing is recorded. This keeps the exception
localized without changing generic CSE behavior.

## 7. Mask ownership

Option C keeps mask and fill semantics in existing Triton codegen rather than
building a second guard-equivalence system.

There are three separate concerns:

| Concern | Owner | Rule |
| --- | --- | --- |
| Masked source capture | `_SubParentValueResolver._record` | Do not record while `_load_mask` is active. |
| Consumer predicate/fill | `TritonKernelOverrides.masked` | A physical load may receive `_load_other`; an in-kernel result uses an outer `where`. |
| Derived tile validity | `_DerivedIterationFamily.set_value_masks` | Attach the sub-parent family's shape-compatible masks to materialized values. |

[`TritonKernelOverrides.masked`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/triton.py:2532)
sets `_load_other` to a concrete fill only when the masked body can be represented
as a physical load. If the result may come from an in-kernel/store-cache value,
it sets `_load_other=None` and emits an outer `where` at
[`triton.py:2563`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/triton.py:2563).

Accordingly,
[`resolve_sources`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:2471)
declines forwarding whenever `_load_other` is concrete. An optional external
source then uses the physical fallback, which owns its predicate and fill. A
required internal source would fail loudly rather than silently lose the fill.
When `_load_other is None`, forwarding is allowed and the existing outer
`where`, if any, wraps the result.

Derived validity masks are separate. [`mask_vars_for_shape`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:1573)
aligns lower-rank shapes as Triton broadcasting does, retains only compatible
family masks, and lets the kernel remove statically unnecessary masks.
[`set_value_masks`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:1599)
applies those masks to direct child values, group broadcasts, and split parts.

## 8. Last-writer behavior during replay

The latest-writer rule at
[`simd.py:2404`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:2404)
is load-bearing, not an optimization.

[`_codegen_sub_parent_output_groups`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:3777)
may replay the same output-group nodes once per output lane. An internal buffer's
unique scheduler writer can therefore emit a new CSE value on every replay. The
consumer in that replay must see the newest value, not the first still-live value
recorded for the name.

For internal contracts, `_record` therefore:

1. removes materializations associated with every prior value;
2. replaces the name's candidate set with the new value; and
3. clears any materialization already associated with that value.

External contracts intentionally do the opposite: they retain an ordered set of
equivalent live values because parent-, group-, and child-shaped instances can
be alternative materialization sources.

The defect that established this invariant involved a `(4, 3)` MXFP6 packing
node replayed three times. Retaining all internal stores caused later replays to
select the first packed byte, producing a 66.7% mismatch. The corrected behavior
and validation record are summarized at
[`f1_option_c_results_20260827.md:36`](/data/users/eellison/pytorch/agent_space/f1_option_c_results_20260827.md:36).

## 9. Pipeline integration and containment

The resolver exists only when there is a sub-parent stage.

Codegen also rebuilds the plan from the final fused topology rather than reusing
an earlier scheduling-time object. Standalone codegen searches for a fresh plan
at
[`_find_sub_parent_epilogue_plan`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:2879),
while nested codegen calls `plan_from_topology`; both paths fail loudly if the
plan has been lost at
[`codegen_staged_reduction`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:3316).
This rebuild is part of Option C's trust argument: the per-name contract is made
from the relations for the nodes and loop state that will actually be emitted.

In nested reduction codegen, construction is guarded by
`sub_parent_stage is not None` at
[`simd.py:3452`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:3452).
It wraps the outer schedule and source-producing parent/grouped regions at
[`simd.py:3466`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:3466) and
[`simd.py:3492`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:3492).

In standalone staged codegen, construction occurs only inside
[`_codegen_reduction_with_sub_parent_epilogue`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:3823),
at
[`simd.py:3900`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:3900).

Option C adds no fields or behavior to generic CSE, `LoopBody`, dependency
extraction, scheduler APIs, or ordinary Triton load codegen. The only CSE
exception is the local physical-load helper used after a planned optional miss.
The neutral full-resolution nested test patches resolver construction to raise,
proving that ordinary nested codegen does not instantiate it at
[`test_fullres_kernel_form`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_nested_reduction.py:3386).

## 10. Focused tests

### Option C unit contract

Read these first:

| Test | Contract pinned |
| --- | --- |
| [`test_sub_parent_resolver_rejects_inconsistent_name_contract`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_inductor_scheduler.py:433) | Reject mixed direct/lane mode, conflicting lanes for one normalized consumer, mixed internal/external roles, and mixed source sets. |
| [`test_sub_parent_resolver_uses_planned_lane_set`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_inductor_scheduler.py:473) | Derive lanes 0 and 2 from replay indexes and reject unplanned lane 1. |
| [`test_sub_parent_source_capture_is_role_aware`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_inductor_scheduler.py:505) | Capture external loads, ignore wrong-role operations, retain external alternatives, replace internal writes, evict stale materialization, and continue after an unmaterializable alternative. |
| [`test_sub_parent_required_source_must_remain_live`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_inductor_scheduler.py:555) | Required sources fail both at lazy resolution and the eager preservation boundary. |
| [`test_sub_parent_resolver_uses_masked_load_ownership`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_inductor_scheduler.py:578) | Masked source loads are not captured; outer-where consumers can forward; concrete-fill consumers cannot. |
| [`test_sub_parent_external_fallback_and_atomic_store`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_inductor_scheduler.py:608) | Planned optional misses use the physical remapped load, bypass the inner/store cache, and ignore non-plain store modes. |
| [`test_group_width_equal_to_child_width_is_direct`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_inductor_scheduler.py:638) | Child-width dispatch wins over group-width broadcast when the symbolic widths coincide. |

### Scheduler proof and refusal cases

[`test_sub_parent_broadcast_access_relation_frame_contract`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_inductor_scheduler.py:727)
covers accepted trailing broadcasts and rejects an index-using extra axis,
regrouped raw frame, multiple writers, and `StarDep`.

[`test_sub_parent_internal_access_relation_preserves_emission_frame`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_inductor_scheduler.py:778)
covers same-group normalized identity, later-group broadcast, wrong frames,
duplicate writers, and consumers scheduled before their writer.

[`test_nested_dependency_matches_require_injective_producer`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_inductor_scheduler.py:954)
shows that temporal-name relations survive mutation renaming but still require
the exact planned raw pair and an injective producer. The unsafe multiwrite,
indirect, and atomic cases are rejected by
[`test_planned_dependency_matches_reject_unsafe_write`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_inductor_scheduler.py:1029).

### End-to-end behavior

High-signal coverage includes:

- Derived masks on indirect indexing in standalone and nested paths:
  [`test_standalone_sub_parent_preserves_indirect_index_mask`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_nested_reduction.py:982) and
  [`test_nested_sub_parent_preserves_indirect_index_mask`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_nested_reduction.py:999).
- Dynamic batch and reduction extents:
  [`test_dynamic_sub_parent_epilogue`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_nested_reduction.py:1034) and
  [`test_dynamic_standalone_sub_parent_epilogue`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_nested_reduction.py:1619).
- Required in-epilogue forwarding:
  [`test_producer_consumer_sub_parent_intermediate`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_nested_reduction.py:1364).
- External parent-width capture versus reload and split count:
  [`test_producer_consumer_inlined_parent_full_source`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_nested_reduction.py:1402).
- Independent inputs that must remain ordinary loads:
  [`test_producer_consumer_independent_sub_parent_source`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_nested_reduction.py:1434) and
  [`test_independent_sub_parent_source`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_nested_reduction.py:1676).
- Persistent and looped standalone execution:
  [`test_standalone_sub_parent_epilogue`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_nested_reduction.py:1603) and
  [`test_looped_standalone_sub_parent_large_group`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_nested_reduction.py:1778).
- Last-writer-sensitive MXFP6 replay:
  [`test_producer_consumer_mxfp6_preshuffled_four_to_three_pack`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_nested_reduction.py:2000), instantiated by both
  [`NestedReductionTest`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_nested_reduction.py:2608) and
  [`NestedReductionNonPersistentTest`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_nested_reduction.py:2613).
- Refusal of incompatible exact relations, including mismatched parent
  coordinates, offset accesses, ambiguous same-name access, shifted reduction
  output, and transposed frames:
  [`test_standalone_sub_parent_rejects_mismatched_parent_coordinates`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_nested_reduction.py:2212),
  [`test_standalone_sub_parent_rejects_offset_source`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_nested_reduction.py:2227),
  [`test_standalone_sub_parent_rejects_ambiguous_source_load`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_nested_reduction.py:2351),
  [`test_standalone_sub_parent_rejects_shifted_reduction_output`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_nested_reduction.py:2370), and
  [`test_standalone_sub_parent_rejects_transposed_sibling_frame`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_nested_reduction.py:2386).

### Kernel-form protection

The kernel-form suite checks load counts, split counts, stores, casts, and
broadcasts across persistent and looped variants. The most relevant entries are:

- MXFP6 four-to-three replay:
  [`test_mxfp6_four_to_three_pack_kernel_form`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_nested_reduction.py:3531).
- Internal-source replay:
  [`test_mxfp6_internal_source_kernel_form`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_nested_reduction.py:3545).
- Standalone parent-lane forwarding:
  [`test_standalone_sub_parent_epilogue_kernel_form`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_nested_reduction.py:3559).
- Pointwise parent producer and dynamic reduction width:
  [`test_pointwise_producer_sub_parent_kernel_form`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_nested_reduction.py:3607) and
  [`test_dynamic_r_sub_parent_kernel_form`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_nested_reduction.py:3624).

The same `_InternalsBase` methods run in persistent and nonpersistent classes at
[`test_nested_reduction.py:3662`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_nested_reduction.py:3662).

## 11. Recorded validation

The validation summary is at
[`f1_option_c_results_20260827.md`](/data/users/eellison/pytorch/agent_space/f1_option_c_results_20260827.md:47).
For this snapshot it records:

```text
test/inductor/test_inductor_scheduler.py: 124 tests, OK, 6 skipped
test/inductor/test_nested_reduction.py:   399 tests, OK, 8 skipped

62/62 differential fuzz cases
98 runtime invocations
70 exact tensors and 175 floating tensors compared
maximum absolute error: 0.0
staged kernels: 54 vs 54
generated kernels: 92 vs 92
no per-case conversion, split, or broadcast differences
```

All ten protected generated-kernel hashes matched the exact-runtime Option A
implementation. Coverage includes persistent and looped reductions, factors 2
and 4, bf16/fp16/fp32, FP8 and integer casts, tails, concrete-fill fallback, and
dynamic batch/reduction shapes. The detailed counts and hashes are at
[`f1_option_c_results_20260827.md:56`](/data/users/eellison/pytorch/agent_space/f1_option_c_results_20260827.md:56).

Complexity is slightly lower than Option A: aggregate production AST LOC
`16525 -> 16512`, cyclomatic complexity `3568 -> 3567`, and changed-function
AST LOC `229 -> 216`. See
[`f1_option_c_complexity_final4_20260827.json`](/data/users/eellison/pytorch/agent_space/f1_option_c_complexity_final4_20260827.json:8).

## 12. Reviewer checklist

The implementation is ready only if each statement below remains true:

- Fusion legality is justified exclusively by exact scheduler relations and
  ordinary dependency checks.
- Every access of a planned name is covered by one of the three exhaustive
  relation builders.
- Contract construction rejects mixed source sets, mixed roles, mixed
  direct/lane modes, and contradictory exact-consumer lanes.
- External values come only from loads and may retain multiple shape
  alternatives.
- Internal values come only from plain/reduction stores and use latest-writer
  semantics across replay.
- A parent-width tuple selects a lane derived from the actual replay index and
  accepted by the planned lane set.
- An internal source never falls back to memory.
- An optional planned fallback bypasses name-only store forwarding while
  preserving ordinary physical-load bookkeeping.
- Masked source values are not captured; consumer fills remain owned by
  `TritonKernelOverrides.masked`; derived shape masks remain owned by the
  iteration family.
- The resolver cannot be constructed on a nested reduction without a
  sub-parent stage.

The principal residual trust assumption is intentional and visible: codegen
does not independently re-identify each consumer `MemoryDep`. It relies on the
final rebuilt scheduler plan being exhaustive, then validates the precise facts
needed by replay. Any future planner that allows two semantic sources, two
capture roles, or direct and lane consumers to share one name must either make
the contract richer or decline that plan.
