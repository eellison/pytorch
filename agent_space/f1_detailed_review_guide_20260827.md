# F1 detailed review guide: exact sub-parent access forwarding

Date: 2026-08-27

This is an AI-assisted local review guide. It is not intended to be pasted into
GitHub without human review and the disclosure required by `AI_POLICY.md`.

## Snapshot and scope

Worktree:

`/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt`

Current F1 commit:

`fc09fb38287d4639df6424c5c33cfb9749b000ba`

Local review-base parent:

`d4167dba5b8755cb8ee3ec41d1af7852ab5b87d1`

The reviewed `_AccessGuard` removal is folded into this commit. The worktree is
clean, and nothing in this guide has been submitted to GitHub.

Use these commands to inspect only F1:

```bash
cd /data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt
git diff --stat HEAD^
git diff HEAD^ -- torch/_inductor/scheduler.py
git diff HEAD^ -- torch/_inductor/codegen/simd.py
git diff HEAD^ -- test/inductor/test_inductor_scheduler.py
git diff HEAD^ -- test/inductor/test_nested_reduction.py
```

Diff size:

| File | Added | Deleted | Net |
| --- | ---: | ---: | ---: |
| `torch/_inductor/scheduler.py` | 131 | 181 | -50 |
| `torch/_inductor/codegen/simd.py` | 316 | 208 | +108 |
| Production total | 447 | 389 | +58 |
| `test/inductor/test_inductor_scheduler.py` | 235 | 40 | +195 |
| `test/inductor/test_nested_reduction.py` | 18 | 7 | +11 |

The large raw diff overstates the conceptual change. F1 deletes a layout/name
compatibility layer and replaces it with one exact access record plus a runtime
resolver. The scheduler is net smaller; most new production code is exact
codegen-side identity, liveness, materialization, and fallback handling. Mask
and fill semantics remain owned by existing Triton masked-load codegen.

## What F1 changes

The #191775 base already knows how to run a pointwise epilogue at a smaller
iteration width than its parent reduction. Its forwarding interface is the
problem F1 addresses.

Before F1, planning classified an entire buffer name with a layout category:

```text
buffer name -> INTERLEAVED | BROADCAST | IDENTITY
```

Codegen then saw a load of that name and applied the category. The consumer's
actual index was no longer part of the authorization decision. Correctness
therefore depended on the planner proving that every relevant read of a name had
one compatible relationship.

After F1, planning records one relation per exact consumer read:

```text
exact consumer MemoryDep
    -> exact source MemoryDep alternatives
    -> optional parent lane
    -> whether a live in-kernel source is mandatory
```

Codegen reconstructs the current load's logical `MemoryDep`, finds that exact
relation, finds a live unguarded source value, and only then adapts its shape to
the sub-parent iteration family.

F1 therefore covers both halves:

1. **Scheduler:** construct and authorize exact access relations.
2. **Codegen:** record values by exact source access and resolve exact consumer
   loads.

F1 does not redesign the already-reviewed sub-parent topology, factor/rate
selection, output grouping, deferred parent scheduling, or recursive factor-4
split. It changes the interface through which those pieces forward values.

### Before/after symbol map

| Review-base symbol | F1 symbol or outcome | Meaning |
| --- | --- | --- |
| `ProjectedSourceAccess` | `SubParentAccessRelation` | A grouped name/layout record becomes one exact per-consumer access relation. |
| `SubParentSourceLayout` | deleted | Split, group broadcast, or direct use is inferred from the live CSE shape. |
| `SubParentEpilogueStage.source_projections` | `access_relations` | The stage exposes exact access facts directly. |
| `source_layouts` | deleted | Codegen no longer receives a name-to-layout adapter. |
| `broadcast_source_names` stage property | deleted | Reduced sources are exact relations, not a name set. |
| `internal_dependency_names` | deleted | Internal epilogue forwarding is exact per read. |
| `_SubParentSourceLoadResolver` | `_SubParentValueResolver` | The wrapper records unguarded loads, stores, and reduction stores by exact access. |
| `_DerivedIterationFamily.remapped_values` | deleted | Derived families retain only iteration/index/mask responsibilities. |
| `forwarded_store_names` / `masked_forward_names` | deleted | Exact relations replace the name sets; masked-load semantics remain in `TritonKernelOverrides.masked`. |
| `projected_access_pairs()` | `sub_parent_access_pairs()` | Fusion consumes the flattened exact relation pairs. |

The meaningful new types are limited to the planner record and one small
runtime record, `_ResolvedSubParentSource`. Runtime caches use normalized
`MemoryDep` objects directly.

## The review model

Keep this concrete factor-2 example in mind:

```text
parent source:
  MemoryDep("x", 16*b + r, ranges=[B, 16])

child reads:
  MemoryDep("x", 16*b + 2*c,     ranges=[B, 8])  lane 0
  MemoryDep("x", 16*b + 2*c + 1, ranges=[B, 8])  lane 1
```

The parent codegen has one value shaped like `[B, 16]`. The child epilogue runs
at `[B, 8]`. The planner proves the first child read is lane 0 and the second is
lane 1. Codegen reshapes the live parent value to `[B, 8, 2]`, splits it, and
returns only the lane recorded for the exact consumer read.

There are two other source shapes:

```text
group-width reduction value: [B, D / G]
    -> broadcast each group value to [B, D / factor]

already child-width value: [B, D / factor]
    -> use directly
```

The new record does not label those `BROADCAST` or `IDENTITY`. Codegen derives
the necessary materialization from the live value's actual shape. The exact
consumer access still determines whether that value is authorized for the load.

## End-to-end control flow

```text
Scheduler._can_fuse_impl
  |
  |-- recognize/build a staged plan
  |     |-- parent-width lane relations
  |     |-- reduced/group-width broadcast relations
  |     `-- internal epilogue relations
  |
  `-- _prove_staged_fusion_dependencies
        |-- preserve raw X/R frame for lane relations
        |-- require exact raw plan membership for sub-parent residuals
        `-- return renamed MemoryDepMatch pairs to ordinary fusion legality

SIMD codegen receives SubParentEpilogueStage.access_relations
  |
  `-- _SubParentValueResolver
        |-- normalize planned source and consumer accesses
        |-- wrap parent/grouped emission
        |-- reconstruct and record exact source loads/stores/reduction stores
        |-- materialize parent-lane values before a destructive CSE flush
        `-- during epilogue replay:
              reconstruct exact consumer access
              -> find exact relation
              -> find a live unguarded source
              -> direct use / group broadcast / parent split
              -> select planned lane
              -> let existing ops.masked code own any consumer predicate/fill

Missing source:
  in-kernel required source -> compiler assertion
  external optional source  -> real derived-domain reload, bypassing store cache
```

## Recommended review order

The highest-value path is steps 1 through 9. Steps 10 through 13 connect the
mechanism to existing emission and tests.

### Step 1: Review the planner/codegen record

Read
[`SubParentAccessRelation`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/scheduler.py:2095).

```python
SubParentAccessRelation(
    source_accesses=(...),
    consumer_access=...,
    parent_lane=...,
    requires_live_source=...,
)
```

Review each field as a separate contract:

| Field | Contract |
| --- | --- |
| `source_accesses` | Nonempty exact source alternatives. All alternatives name the same buffer. Multiple alternatives are allowed because the same logical value may be available through more than one equivalent source access. |
| `consumer_access` | One exact raw `MemoryDep`. F1 intentionally flattens the old record that grouped several consumers by buffer name. |
| `parent_lane` | A statically proved lane when a full parent-width value must be split. `None` means no lane selection; it does not itself mean broadcast or identity. |
| `requires_live_source` | `True` means some source is written inside this kernel and memory fallback is unsafe. `False` means the source is external and may be reloaded if no live CSE value remains. |

`__post_init__` validates only the structural minimum: a nonempty source tuple and
one shared name. The builders own the semantic proofs. That split is deliberate:
the record is the result of planning, not a second symbolic prover.

Questions to answer:

1. Is one record per exact consumer the right boundary between scheduler and
   codegen?
2. Is `parent_lane=None` unambiguous once codegen also sees the live value shape?
3. Is requiredness correctly a planner fact, rather than inferred later from a
   name or from whichever store happened to execute?

#### Why source accesses are plural

Only the parent-lane builder can produce more than one `source_accesses` entry.
It gathers every parent-side read and write of the temporal buffer name and
retains them when they all map to the same index in the explicit `(x, parent_r)`
frame. Target tests produce this for the same input represented in flat and
grouped parent frames, for example:

```text
flat:    x[1024*b + r]             ranges [32, 1024]
grouped: x[1024*b + 16*g + lane]   ranges [32, 64, 16]
child:   x[1024*b + 16*g + 2*c]    ranges [32, 64, 8]
```

The two parent accesses are raw-frame witnesses for the same logical elements,
not competing semantic values.

There is still only one `consumer_access` because the representation is
deliberately flattened per read. Two child reads, such as lane 0 and lane 1,
produce two relations that may share the same source tuple.

Instrumentation found these plural raw relations in the sub-parent and protected
NVFP4 tests. After loop merging and codegen replanning, the flat and grouped
forms normalize to one runtime source in the observed kernels. This means the
shared record currently serves two related roles: plural raw witnesses for
scheduler proof, and normally singular canonical identity for codegen.

Making the field singular mechanically would discard a real raw-frame witness.
A future cleanup has two plausible directions: infer the mapping from codegen
index expressions, following ordinary CSE but accepting its normalization
fragility; or have fused scheduler nodes retain the exact internal read/write
matches established during fusion and pass one canonical source to codegen.
The latter is not available today: fused nodes rebuild external dependency
summaries and discard satisfied internal edges, while staged append planning
rebuilds rather than retains its prior plan. The current resolver unit's
`d0`/`d0 + 1` alternatives are not a planner-realistic example and should be
replaced with a test of whichever phase boundary is chosen.

### Step 2: Review parent-width lane relation construction

Read
[`_try_get_sub_parent_access_relations`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/scheduler.py:1094),
with most attention on
[`parent_writes` and relation construction](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/scheduler.py:1185).

Most of this function predates F1. The inherited proof:

1. maps parent and child accesses into explicit common frames;
2. proves the parent extent is divisible by the sub-parent factor;
3. requires all parent accesses for a name to normalize to one parent index;
4. computes `parent_index(parent_r = factor * child_r + lane)`; and
5. requires the child index to equal that expression for one static lane.

F1 adds four pieces:

1. Retain the original source `MemoryDep` objects as `source_deps`, separately
   from the normalized expressions used for geometry.
2. Collect raw parent writes.
3. Derive `requires_live_source` from exact writer membership.
4. Emit one relation per child read, retaining the already-proved lane.

The raw/normalized distinction is important:

```text
normalized index: proves the geometric relation
raw MemoryDep:    identifies the actual planned/code-generated access
```

Review the fail-closed boundaries:

- no parent access for a child-read name;
- more than one normalized parent index;
- lane is not one static integer in the factor range;
- child index does not equal the substituted parent index.

The existing TODO at lines 1198-1199 is also a real limit: without a structural
load-index cache, several distinct parent indices for one buffer are declined.

### Step 3: Review the two non-lane relation builders

#### Internal epilogue values

Read
[`_sub_parent_internal_access_relations`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/scheduler.py:1266).

This handles a value written by one sub-parent output node and read by another.
It enforces:

- exactly one writer for the internally consumed name;
- writer precedes reader in `(output_group, node)` emission order;
- same-group reads normalize exactly like the write;
- later-group reads preserve the write through trailing broadcast axes.

For example:

```text
write: MemoryDep("scale", g, ranges=[G])
read:  MemoryDep("scale", g, ranges=[G, 3])       accepted
read:  MemoryDep("scale", 3*g + lane, [G, 3])    rejected
```

F1 does not invent this proof. It flattens the old `IDENTITY` projection into
one exact relation per read and marks it required/live because the writer is in
the epilogue itself.

#### Reduced/group-width values

Read
[`_sub_parent_broadcast_access_relations`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/scheduler.py:1373)
and its inherited frame predicate
[`_sub_parent_consumer_is_trailing_broadcast`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/scheduler.py:1326).

This handles a reduced scale-like source whose trailing consumer axes do not
change the address. F1's change is the result representation: one relation per
consumer, `parent_lane=None`, `requires_live_source=True`.

The adjacent unique-writer and raw-plus-normalized double proof are the lower
base fix for the masked-source fallback, not a new F1 design. While reviewing
F1, verify that flattening preserves its consumer set; there is no need to
re-review the raw/normalized workaround as if F1 introduced it.

### Step 4: Confirm the old layout/name API is actually gone

Read
[`SubParentEpilogueStage`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/scheduler.py:2157)
and
[`StagedReductionPlan.sub_parent_access_pairs`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/scheduler.py:2211).

F1 deletes:

- `SubParentSourceLayout`;
- `ProjectedSourceAccess`;
- `source_projections`;
- `source_layouts`;
- `broadcast_source_names` as a stage/codegen view;
- `internal_dependency_names`;
- the many-consumer Cartesian product in `projected_access_pairs`.

Local `broadcast_source_names` variables still exist inside planning. They only
partition candidate names so the correct proof builder can run. They do not
cross the planner/codegen boundary.

The desired endpoint of this step is simple: no runtime decision is authorized
only by a buffer name or an enum category.

### Step 5: Review fusion authorization

Read
[`_prove_staged_fusion_dependencies`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/scheduler.py:9387).

This is the highest-scrutiny scheduler hunk.

#### 5a. Raw X/R boundary

Relations with `parent_lane is not None` receive the inherited raw X/R boundary
check. This replaces `layout is INTERLEAVED`; it is not broader because only the
parent-lane builder can produce the lane witness.

The check runs before normalized `MemoryDep` equivalence can flatten away the
distinction between the outer X domain and the parent/child R domains.

#### 5b. Raw identity versus mutation-renamed identity

The producer write table now retains both forms:

```text
raw_write:     compared against the planner's temporal records
renamed_write: returned to ordinary fusion legality and scoring
```

This prevents mutation aliases from making an incorrect raw plan look exact.
For example, if raw names `buf` and `other` both rename to `renamed_buf`, an
incorrect relation for `other` must not authorize a real access to `buf`.

#### 5c. Residual ownership and exact membership

For each producer-output read that does not pass ordinary
`fusable_read_and_write`:

1. reject ambiguous multiple writes;
2. if the read belongs to a sub-parent epilogue, require the exact raw
   `MemoryDepMatch` from `plan.sub_parent_access_pairs()`;
3. independently require the existing safety predicate for name consistency,
   TMP/indirect indexing, synchronization-requiring store modes, writer
   injectivity, and contiguous normal or stride-order normalization;
4. only a grouped-stage-owned read may use the retained legacy nested
   equivalence proof;
5. return the renamed match to existing vertical legality and scoring.

Ownership order is load-bearing. A read owned by both the sub-parent and nested
stages takes the stricter exact-relation branch. It cannot fall through to the
legacy normalized proof.

The inline TODO marks the remaining boundary: parent-to-grouped nested legality
still uses `_fusable_read_after_index_equivalence`. F1 makes exact records the
sole authority for sub-parent forwarding; it does not yet delete that separate
fusion-only compatibility path.

### Step 6: Review the runtime value record and mask boundary

Read
[`_ResolvedSubParentSource`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:1519).

It keeps only the normalized source `MemoryDep`, live CSE value, and planned
parent lane. The resolver deliberately records only values emitted with no
active load mask, so a normalized access is the complete runtime cache key.

This is intentionally narrower than teaching the resolver a second mask/fill
algebra. Existing
[`TritonKernelOverrides.masked`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/triton.py:2532)
already owns that choice:

- a direct external load receives `_load_mask` plus a concrete `_load_other`;
- a body that may return an in-kernel/store-cache value receives
  `_load_other=None`, and the surrounding callback emits the final `where`.

The resolver therefore forwards only unguarded sources and only when
`_load_other is None`. A direct masked load with a concrete fill stays on the
ordinary physical-load path.

The deliberate capability loss is narrow: an optional external value already
loaded unguarded is reloaded when a later direct masked load needs a concrete
fill. The staged kernel remains valid; it may issue the extra load. No protected
target kernel uses that case.

### Step 7: Review logical access reconstruction

Read
[`_logical_memory_access`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:1527).

This is the bridge from an interpreted load/store back to the exact scheduler
record. It:

1. reads the current FX operation from the interpreter;
2. verifies the operation kind and temporal buffer name;
3. finds the named `get_index` expression in the current `SchedulerNode`'s
   `LoopBody`;
4. rebuilds a `MemoryDep` using that body's original variables and ranges;
5. checks the store mode; and
6. normalizes the result.

The critical choice is to use the node's logical loop-body index, not the
already-remapped Triton expression passed to the wrapper handler. The scheduler
record is in the node's logical frame; the handler may currently be executing
in a derived iteration family.

All inconsistencies assert. A best-effort fallback here would turn a planner /
codegen disagreement into potentially wrong register forwarding.

Review questions:

1. Do all wrapped load, store, and reduction-store paths have a current FX node
   from which the same logical access can be reconstructed?
2. Are temporal names preserved rather than mutation-renamed names?
3. Is normalization occurring at the same stage for planned and emitted
   accesses?

### Step 8: Review resolver construction and source recording

Read
[`_SubParentValueResolver`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:2365),
especially initialization through
[`store_reduction`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:2439).

Initialization builds two different kinds of state:

```text
planned/static:
  _source_accesses
  _relations_by_name[name][exact_consumer_access]

runtime:
  _values[normalized_source_access]
  _materialized[normalized_source_access]
```

The outer name dictionary is only a cheap filter. The inner exact
`MemoryDep` lookup is the authorization decision. Conflicting records for the
same exact consumer assert during construction.

The resolver wraps source-stage codegen and delegates first, then records the
emitted value only when the reconstructed source access is planned and the
source was emitted without an active load mask:

- `load`: records the returned load CSE;
- `store`: records the stored value for ordinary stores;
- `store_reduction`: records the completed reduction value.

Atomic and TMA stores are intentionally not recorded. Their operand is not
necessarily the value observable after the memory operation.

Guarded source executions are deliberately ignored. They continue through the
ordinary masked-load path rather than being widened by register forwarding.
Recording a newer unguarded value for the same access invalidates its cached
materialized form, preventing an old split/broadcast result from surviving a
new source value.

### Step 9: Review exact lookup, liveness, and failure policy

Read:

- [`get_relation`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:2479)
- [`resolve_sources`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:2492)
- [`materialize_source`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:2511)
- [`resolve_load`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:2531)

`get_relation` has three intentionally distinct outcomes:

| Situation | Result |
| --- | --- |
| Name has no planned relation | `None`; use an ordinary derived-domain load. |
| Exact normalized consumer access exists | Return its relation. |
| Name is planned but this exact access is absent | Raise `AssertionError`. |

The third case is the core loudness property. A same-name, different-index read
must not inherit another read's authorization or silently use name-keyed store
forwarding.

`resolve_sources` then filters relation alternatives by:

1. exact source access;
2. the existing masked-load ownership boundary; and
3. `kernel.cse.contains_value` liveness.

The mask/fill cases are:

```text
unguarded consumer                         -> forward an unguarded source
masked consumer, outer where owns fill     -> forward an unguarded source
masked consumer, physical load owns fill   -> do not forward; reload if optional
guarded source execution                    -> never enter the resolver cache
```

`_load_other is None` identifies the first two consumer cases. In the masked
case, `TritonKernelOverrides.masked` applies the predicate around the returned
value after the body finishes. The resolver does not recreate that `where`.

The Python object remaining in `_values` does not prove liveness. CSE flushes
drop values from the live caches, and `contains_value` is the authoritative
check.

If no source works:

- `requires_live_source=True`: assert with both source and consumer accesses;
- `requires_live_source=False`: return `None` so codegen reloads the external
  value.

An in-kernel source cannot safely take the reload path: the buffer may have been
removed, and a cross-thread global-memory read-after-write would not have the
required ordering.

### Step 10: Review shape materialization and masks

Read
[`materialize_value_at_sub_parent_resolution`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:2074),
[`mask_vars_for_shape`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:1609),
and
[`set_value_masks`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:1635).

Materialization is based on the live CSE shape:

| Live grouped-axis width | Action |
| --- | --- |
| scalar or singleton | Use directly. |
| child width `parent_block / factor` | Use directly and attach child-family masks. |
| group width `num_groups` | Reshape/broadcast each group across `G / factor` child positions. |
| full parent width `parent_block` | Reshape to `[..., child, factor]`, recursively split, and return all lanes. |
| anything else | Return `None`; another source, external reload, or required-source failure handles it. |

The child-width check intentionally precedes group-width. When
`num_groups == child_block`, one child element already represents one group and
must be used directly rather than broadcast again.

For a full-width factor-4 source:

```text
[B, R] -> [B, R/4, 4] -> four values [B, R/4]
```

The planner's `parent_lane` selects one result. Codegen no longer infers a lane
from the remapped emitted index.

For a group-width scale:

```text
[B, R/G] -> [B, R/G, G/4] -> [B, R/4]
```

Every split result, group-broadcast result, and direct child-width result
receives masks for its active family. Scalar and singleton-width values rely on
their invariant shape. Lower-rank shapes are left-padded with singleton
dimensions, matching Triton broadcasting, so an invariant axis does not inherit
a mask for an axis it does not vary over.

### Step 11: Review masked-load ownership and fallback loads

Read
[`resolve_sources`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:2492),
[`_PointwiseRemapHandler.load`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:2321),
and the existing
[`TritonKernelOverrides.masked`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/triton.py:2532).

There is no resolver-owned guard reconstruction. A concrete `_load_other`
means the predicate and fill belong on a physical `tl.load`, so resolution
declines. When `_load_other is None`, either the consumer is unmasked or the
surrounding `ops.masked` callback has chosen its existing outer-`where` path;
the forwarded value can be returned unchanged in both cases.

If an exact relation exists but an optional external source is no longer live,
the handler calls the kernel load directly through its remapped derived index.
It deliberately bypasses the ordinary wrapper/store cache. Otherwise a
same-name store at another index could be returned even though F1 correctly
decided to reload.

The direct kernel load still retains normal load accounting, indirect-index
handling, tracing, and the existing invalidated-store barrier behavior.

### Step 12: Review eager versus lazy materialization

Read
[`materialize_sources`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:2462).

Only relations with a `parent_lane` are candidates for early materialization. A
required full parent-width CSE must be split before a parent-body flush
invalidates it; once that in-kernel value is gone, the child lanes cannot be
reconstructed safely.

The two emission paths apply that rule slightly differently:

- nested codegen offers every lane relation for early materialization, but only
  a required relation must succeed;
- standalone codegen eagerly materializes required lane relations only, while
  an optional external lane source may be flushed and reloaded in the child
  domain.

Relations without a lane remain lazy until first use:

- already child-width values;
- group-width broadcasts;
- scalars/singletons;
- internal epilogue values.

This distinction is separate from `requires_live_source`:

```text
requires_live_source: may codegen fall back to memory?
parent_lane:          must codegen split before a possible CSE flush?
```

Conflating those two facts previously caused incorrect epilogue flush gating in
the upper replay experiments. Review that the current code keeps them separate.

### Step 13: Review nested and standalone integration

#### Nested path

Read the resolver construction and source-emission scopes beginning at
[`nested emission`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:3361),
then the common replay helper
[`_codegen_sub_parent_output_groups`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:3795).

The resolver wraps both source-producing regions:

1. outer reduction schedule;
2. parent-full pointwise work plus grouped reduction schedule.

Before output-group replay, every lane relation is offered for materialization
while its source CSE may still be live; required relations must succeed. Each
output group/lane then runs under the pointwise remap handler with exact
consumer resolution enabled.

#### Standalone path

Read
[`_codegen_reduction_with_sub_parent_epilogue`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:3841),
especially the parent emission and flush boundary near lines 3917-3939.

The parent schedule runs under the resolver. Then:

- if there is no required lane relation, codegen may flush the parent body;
- if a required lane relation exists, codegen materializes it before the flush
  and keeps the pending body available for the child replay.

Both nested and standalone paths use the same output-group replay helper. That
extraction is mostly a useful deduplication, but verify that resolver scope and
lane selection are identical in the two paths.

## Test-to-contract map

Keep these tests open beside the corresponding implementation rather than
reading the full test diff linearly.

F1 adds exactly four scheduler test methods, one substantive parameter extension
to an existing scheduler proof test, and one substantive arm to an existing
end-to-end test.
Several other test hunks only rename projection fixtures to access-relation
fixtures. They remain useful background, but they are not new F1 evidence.

### Exact identity, liveness, and alternatives

[`test_sub_parent_access_identity_and_source_cache`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_inductor_scheduler.py:434)

Pins:

- logical access reconstruction distinguishes two indices of the same name;
- a known name with an unplanned exact access fails loudly;
- only planned source accesses are recorded;
- a new source value invalidates its materialized cache entry;
- guarded source executions do not enter the unguarded source cache;
- absent and stale required values fail loudly;
- multiple source alternatives are tried rather than assuming the first works;
- eager source materialization stops after one source alternative succeeds.

This is the densest unit test and the best companion for steps 6 through 9.

### Existing masked-load ownership

[`test_sub_parent_resolver_uses_masked_load_ownership`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_inductor_scheduler.py:518)

Pins the conservative boundary: guarded sources are ignored, an unguarded source
may serve a masked consumer whose surrounding callback owns the `where`, and a
consumer with a concrete load fill declines forwarding.

### External fallback and unsafe stores

[`test_sub_parent_external_fallback_and_atomic_store`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_inductor_scheduler.py:552)

Pins that an optional miss emits a real remapped kernel load instead of calling
the inner name-keyed load, and that atomic/TMA stores never become source CSE
values.

### Shape ambiguity

[`test_group_width_equal_to_child_width_is_direct`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_inductor_scheduler.py:582)

Pins the ordering of the shape dispatch when `num_groups == child_block`:
direct child-width use wins over group-width broadcast.

### Broadcast and internal relation proofs

[`test_sub_parent_broadcast_access_relation_frame_contract`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_inductor_scheduler.py:671)

Pins accepted trailing broadcast, used extra axis rejection, regrouped raw-frame
rejection, normalized-frame behavior, unique writer, and non-`MemoryDep` read
rejection.

These cases already existed in the parent. F1 renames the helper and adapts the
fixture to the flattened relation API; it does not add a new acceptance case.

[`test_sub_parent_internal_access_relation_preserves_emission_frame`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_inductor_scheduler.py:722)

Pins writer ordering and the distinction between same-group exact access and
later-group trailing broadcast.

This acceptance matrix also predates F1. Its diff is primarily the helper rename
and flattened record shape.

### Fusion proof safety and ownership

[`test_nested_dependency_matches_require_injective_producer`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_inductor_scheduler.py:898)

Pins raw temporal relation matching through mutation renames and rejects a
non-injective producer even when a plan pair exists.

The new F1 evidence includes the `planned_name="other"` row: two raw names
collapse to the same mutation-renamed name, but the wrong temporal relation must
still be rejected. The existing dense/non-injective cases also now run with
nonempty mutation renames, pinning that raw membership is checked first and the
accepted pair returned to ordinary legality is renamed.

[`test_nested_dependency_matches_scope_index_equivalence`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_inductor_scheduler.py:943)

Pins the ownership split: nested-only reads may use the retained legacy proof;
sub-parent or dual-owned reads require exact relation membership; unowned reads
decline.

F1 only adapts this test's mocked field and iterator names. The ownership matrix
was already part of the reviewed base.

[`test_planned_dependency_matches_reject_unsafe_write`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_inductor_scheduler.py:973)

Pins multiple writes, TMP/indirect indexing, and atomic write rejection even
when an exact planned pair is present.

This test is likewise inherited behavior with mechanical mock-field changes in
F1.

### End-to-end temporal names and source policy

[`test_producer_consumer_sub_parent_source_mutated_later`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_nested_reduction.py:1243)

Pins that the relation follows the node's temporal read/write names rather than
a later mutation rename.

This end-to-end test is unchanged in F1; it is background coverage for the raw
temporal-name rule.

[`test_producer_consumer_inlined_parent_full_source`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/test/inductor/test_nested_reduction.py:1402)

The existing arm covers a required in-kernel full-resolution source. F1 adds the
`shared_external_source=True` arm, where one input is used in both parent and
sub-parent stages. It pins that source recording may span the earlier stages but
consumer resolution is enabled only during sub-parent replay. The generated
`tl.split` count also distinguishes the two materialization paths.

Existing end-to-end MXFP4, NVFP4, MXFP6, swizzle, preshuffle, and DCN tests are
also important as non-regression coverage, but their main purpose predates F1.
The protected generated-source corpus is a more efficient way to check that F1
did not perturb them.

The builder tests above do not directly assert every new relation field. In
particular, review `parent_lane` and `requires_live_source` at their constructors;
their behavior is primarily pinned through resolver and end-to-end tests.

Defensive assertions for conflicting normalized consumer records, malformed FX
operations, and invalid tuple lanes do not each have a permanent unit test. They
fail loudly and are lower-risk than the permanently covered miscompile paths:
same-name/wrong-index lookup, stale required sources, masked-load ownership, unsafe
stores, and store-cache fallback.

## What to scrutinize most

1. [`SubParentAccessRelation`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/scheduler.py:2095): does it carry every fact codegen needs without encoding policy by name?
2. [`_try_get_sub_parent_access_relations`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/scheduler.py:1094): raw source identity, static lane, and requiredness.
3. [`_prove_staged_fusion_dependencies`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/scheduler.py:9387): raw plan membership, ownership precedence, and safety checks.
4. [`_logical_memory_access`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:1527): exact reconstruction in the logical node frame.
5. [`_SubParentValueResolver`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:2365): exact lookup, masked-load ownership, liveness, alternatives, and failure policy.
6. [`materialize_value_at_sub_parent_resolution`](/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt/torch/_inductor/codegen/simd.py:2074): child/group/full-width dispatch and masks.
7. Standalone and nested flush boundaries: required parent-lane values must be split while live.

## What can be skimmed

- local variable renames from `projection` to `relation`;
- `source_projections` to `access_relations` field plumbing;
- `projected_access_pairs` to `sub_parent_access_pairs`;
- the internal builder's early-continue formatting rewrite;
- deletion of the old enum note after confirming no references remain;
- `RemappedRangeValue` to `MaterializedSubParentValue` naming;
- extraction of the duplicate output-group replay into
  `_codegen_sub_parent_output_groups`, after confirming both callers use the
  same resolver scope;
- test fixture field renames where assertions are otherwise unchanged.

Do not skim record construction merely because it is adjacent to renames. The
per-consumer `consumer_access`, `parent_lane`, and `requires_live_source` fields
are F1's functional payload.

## Known boundary and follow-up

F1 does not remove the inherited parent-to-grouped
`_fusable_read_after_index_equivalence` legality path. That path is scoped to
grouped-stage reads; it is not used by codegen forwarding.

The desired follow-up is not "allow more normalized matches." It is a
source-anchored parent-to-grouped relation that can be rebuilt after fusion and
then required by exact membership, with mutation tests proving that missing
records lose fusion rather than silently widening legality.

That follow-up should not obscure F1's endpoint: all sub-parent codegen
forwarding in this commit is exact-access based, and the old layout/name views
are gone.

F2a is also outside this review. It consumes F1's exact resolver API to perform
one narrow optimization:

```text
cast(broadcast(group_value)) -> broadcast(cast(group_value))
```

Review F1 first; F2a is substantially easier once the resolver contract is
understood.

## Complexity assessment

The current F1 production delta is net +58 lines. The scheduler file itself falls
from 11,214 to 11,187 LOC and from aggregate CC 2,641 to 2,629. The combined
production files move from aggregate CC 3,552 to 3,568. The AccessGuard removal
cuts the previous F1 codegen delta from +198 to +108 net lines and leaves the
changed-function aggregate at +16 CC.

The only new function above CC 6 besides the access reconstruction itself is:

| Function | Cyclomatic complexity | Why it is not split further |
| --- | ---: | --- |
| `_logical_memory_access` | 12 | Flat validation of operation kind, name, index node, ranges, and store mode; extracting one-use checks would hide the contract. |

`_SubParentValueResolver.materialize_sources` is now CC 6 rather than CC 11;
guard grouping and compatibility branches are gone. No new F1 function exceeds
CC 12. The final representation is already flattened: one relation per
consumer and normalized `MemoryDep` keys, not a grouped planner record plus a
second runtime identity system.

## Verification already completed

Current-worktree full-package runs with FX and Inductor caches disabled:

```bash
PYTORCH_WORKTREE="$PWD" TORCHINDUCTOR_FX_GRAPH_CACHE=0 \
TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 \
LD_LIBRARY_PATH=/home/eellison/.conda/envs/pytorch-3.12/lib \
conda run --no-capture-output -n pytorch-3.12 \
python ../run_wt.py test/inductor/test_inductor_scheduler.py -q
# 118 passed, 6 skipped

PYTORCH_WORKTREE="$PWD" TORCHINDUCTOR_FX_GRAPH_CACHE=0 \
TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 \
LD_LIBRARY_PATH=/home/eellison/.conda/envs/pytorch-3.12/lib \
conda run --no-capture-output -n pytorch-3.12 \
python ../run_wt.py test/inductor/test_nested_reduction.py -q
# 399 tests run, OK, 8 skipped
```

The focused cat/MXFP6 set passed 4 tests, and the scheduler `-k sub_parent`
selection passed 18 tests. `py_compile` and `git diff --check` also passed. An
earlier full run selected a broken system `openssl` under the active Conda
library path and failed before codegen; the AOT test and final full run pass
with the Conda executable first in `PATH`.

The conservative mask-ownership policy was exercised against the complete
nested-reduction suite: 399 tests passed and 8 were skipped. Instrumentation
observed 943 successful forwards, all from unguarded sources; the 234 masked
consumers all had `_load_other=None`, so the existing outer-`where` path owned
their predicate. The 12 concrete-fill attempts were optional external loads and
fell back normally. All ten published normalized source hashes remain
byte-identical, including factor-2 persistent/looped, MXFP6 4:3
persistent/looped, internal sources, standalone reduced broadcast, scale
swizzle, preshuffle, and DCN preshuffle. See
[`f1_f2a_protected_kernel_reattest_20260827.md`](/data/users/eellison/pytorch/agent_space/f1_f2a_protected_kernel_reattest_20260827.md).
The detailed guard reachability evidence is in
[`f1_access_guard_reachability_20260827.md`](/data/users/eellison/pytorch/agent_space/f1_access_guard_reachability_20260827.md).

## Final review checklist

The F1 review is complete when you can answer these questions:

1. For any forwarded child load, which exact `SubParentAccessRelation`
   authorizes it?
2. Does the planner retain raw access identity while using normalized indices
   only for symbolic geometry?
3. Can mutation renaming make an unplanned raw access appear planned?
4. Can a sub-parent-owned residual reach the legacy nested equivalence proof?
5. Can a same-name, different-index load receive a forwarded value?
6. Can a guarded source enter the resolver cache, or can a concrete-fill load
   be intercepted instead of following ordinary masked-load codegen?
7. Can a stale in-kernel value silently reload from memory?
8. Can an external value reload without accidentally consulting the
   name-keyed store cache?
9. Are required in-kernel full-width values split before any CSE flush that
   would destroy them?
10. Do child-width, group-width, and parent-width values take the intended
    materialization path, including the equal-width ambiguity?
11. Are derived masks attached to every nonscalar split, broadcast, and direct
    child-width result?
12. Is any old layout enum or name-based stage view still authorizing codegen?

If these answers are clear, the remaining diff is mainly deleted compatibility
surface, assembly plumbing, and tests.
