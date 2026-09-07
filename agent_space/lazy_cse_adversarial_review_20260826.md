# F2a lazy group-width CSE adversarial review

Date: 2026-08-26

Reviewed worktree: `agent_space/followup_lazy_projection_wt`

Snapshot:

- HEAD: `6b8ef64bd373b221c6ed1ba4f4b2f1edef2643c3`
- staged F1 diff: `195484cf6db6ebb7f4d3e9f649ce361ff4fc80d9e1de6027d3a5237649b658e1`
- unstaged F2 diff: `940d1ed26d8098d99ab6d06aa3c468e069e3c3be92cb8d2b0a74d20be0a8b80b`

## Verdict

No production correctness blocker remains in the reviewed F2a slice. The
production diff can freeze before the full scheduler/nested suites and
performance gate.

The original NVFP4 generated-form check did not distinguish F2a from F1. That
was a blocking test-coverage issue, but the current whole-kernel assertions now
reject frozen F1 and both reviewed mutations. No production fix resulted from
that finding.

## Findings

### Nonblocking: `store_reduction` is a barrier, but not a supported remapped node

`_PointwiseRemapHandler.store_reduction()` is inherited from `DefaultHandler`,
so it reaches `_default()`. Because `store_reduction` is not allowlisted, a
group-width value is materialized before `_SubParentValueResolver` records and
emits it. The new unit row correctly pins that dispatch property.

Unlike the explicit `store()` override, the inherited path does not remap the
store index. This is not reachable today: standalone candidate collection skips
reduction nodes at `scheduler.py:868-874`, nested pointwise classification skips
them at `scheduler.py:1529-1533`, and `_codegen_remapped_pointwise()` rejects
reduction variables. Do not add an override just for symmetry. If sub-parent
reduction nodes become legal later, their implementation must add an explicit
remapped `store_reduction()` boundary rather than treating this unit row as
end-to-end support.

### Nonblocking: test size is now proportionate

The first scheduler-test draft was about 170 added lines and executed every
allowlisted spelling even though most use the same generic shape-propagation
rule. The current test uses four representative operations while separately
pinning the exact allowlist, and adds the materially distinct mixed-shape,
unknown-shape, wrong-rank, unknown-op, pack-2, impure-asm, result-shape,
store, reduction-store, and masked-callback cases.

Current test delta is `+145` lines in the scheduler test and net `+11` in the
nested-reduction test. That is a reasonable cost for a `+125` net production
change with a positive policy and callback boundary. Further compression would
mostly hide setup or combine semantically different failure modes; it is not
required.

## Production audit

### Exact source selection, alternatives, and required misses

`_SubParentValueResolver.resolve_load()` keeps the existing ordered loop over
all live exact source alternatives. The optional predicate is evaluated only
after `resolve_sources()` has checked exact access/guard compatibility and CSE
liveness. A non-preservable or non-materializable source still falls through to
later alternatives. The existing required-source assertion remains after the
entire loop, and an optional miss still returns `None` to
`_load_without_store_forwarding()`; F2a does not reintroduce a name-keyed or
generic store-cache fallback.

### Shape proof and direct/group coincidence

`_GroupedReductionLayout.parent_dim(shape)` is now the shared rank classifier
for both eager F1 materialization and F2a recognition, avoiding duplicated
rank-1 singleton logic. `is_group_width_shape()` requires the grouped dimension
and explicitly excludes `child_block(factor)`, preserving direct-child
precedence when `num_groups == child_block`.

Before an allowlisted operation is emitted, the handler calls the canonical
`ShapePropagationOpsHandler`. Unknown shapes, wrong ranks, group-plus-child
broadcast failures, and unsupported operations materialize first. The caught
exceptions match the reachable shape handler failures for this allowlist:
`AssertionError`, `TypeError`, and `NotImplementedError`. After emission, the
actual result must still be one group-width `CSEVariable`, so backend/shape
contract drift fails closed.

The positive allowlist is narrow and sufficient for the inspected target
LoopBodies. NVFP4 uses `to_dtype` and ordinary scalar arithmetic; current MXFP4
uses pure pack-1 `inline_asm_elementwise`; MXFP6 first joins its scale with a
child-width operand at `truediv`. No target graph contains an `ops.reciprocal`,
so leaving it out is correct.

### Masks and guarded callbacks

The preserve predicate requires `parent_lane is None`, an exact group-width
shape, and `consumer_guard_is_deferred()`. An unguarded consumer is eligible. A
masked consumer is eligible only when its fill is `None`, which is the state
installed by `TritonOverrides.masked()` when it has selected an explicit outer
`where`. A scalar-fill direct-load path therefore remains on F1 eager
materialization and `_apply_consumer_guard()`; the fill cannot be dropped.

The explicit `masked()` override copies `body.graph`, leaves the same remap
handler installed while the callback executes, and materializes every returned
group-width leaf before the inner Triton handler applies its final `where`. It
also materializes group-width explicit mask/other operands. With no resolver it
passes through unchanged, so unrelated remapped pointwise paths are unaffected.
There is no callback mode or mutable state to leak on nesting or exceptions.

Every group-to-child widening reuses
`materialize_value_at_sub_parent_resolution()`, whose
`family.set_value_masks()` replaces source masks with masks derived from the
target child shape. No parent/reduced mask is copied or unioned onto the widened
value.

### Stores, inline assembly, factor 4, and lifetime

The explicit `store()` widens before index remapping and before delegation, so
the resolver records the concrete child value actually stored. The inherited
`store_reduction` path is a non-allowlisted barrier and likewise records a
materialized value under today's unreachable-for-replay invariant described
above.

`inline_asm_elementwise` remains narrow only for exact `pack == 1` and
`is_pure is True`. Packed and impure forms materialize before backend dispatch.
Shape propagation sees only the positional tensor operands, matching the
lowering contract; a child-shaped co-operand makes the preflight fail closed.

There is no F2a cache or wrapper. Preserved and derived values are ordinary
`CSEVariable`s, and source eligibility still uses `kernel.cse.contains_value()`.
Flush/invalidation therefore causes an ordinary resolver miss rather than a
stale deferred object. Current factor-4 MXFP6 graphs hit the mixed-width barrier
at their first `truediv`, so both persistent and looped forms remain unchanged.

## Complexity

Production delta in `torch/_inductor/codegen/simd.py`: `+130/-5`, net `+125`.
It adds no type and no cache; it adds seven small methods and changes three
existing methods plus the private `parent_dim` input contract.

Heuristic AST metrics (CC includes boolean terms; nesting counts structured
control flow):

| Method | LOC | CC | Max nesting |
| --- | ---: | ---: | ---: |
| `_PointwiseRemapHandler._is_group_width_value` | 7 | 3 | 0 |
| `_PointwiseRemapHandler._materialize_group_width` | 5 | 3 | 1 |
| `_PointwiseRemapHandler._keep_group_source` | 9 | 4 | 1 |
| `_PointwiseRemapHandler._default` | 33 | 10 | 2 |
| `_PointwiseRemapHandler.masked` | 14 | 2 | 1 |
| `_SubParentValueResolver.is_group_width_shape` | 7 | 2 | 0 |
| `_SubParentValueResolver.materialize_group_width` | 13 | 3 | 1 |
| `_SubParentValueResolver.resolve_load` | 19 | 6 | 2 |

The existing remapped `load()` remains CC 6/nesting 3. The largest new method
is `_default()` at CC 10/nesting 2. This is materially smaller than the
historical wrapper/domain/cache implementation and has no unnecessary
abstraction left. The two handler-local value helpers are justified by their
reuse in argument scans, stores, and callback result materialization.

## Validation

Reviewer-run checks on the current production snapshot:

- `python -m py_compile` for the changed production/test files: pass.
- `spin quicklint`: pass.
- `git diff --check`: pass.
- scheduler group-width focused tests: 6 passed.
- masked scalar-fill integration cases: 2 passed.
- indirect target-mask integration cases: 6 passed.
- dynamic sub-parent integration cases: 4 passed.
- MXFP4 kernel-form cases: 2 passed.
- current NVFP4 kernel-form cases: 4 passed.

Mutation checks on the NVFP4 generated-form test:

- frozen F1 production: fails both active persistent/nonpersistent cases;
- `_keep_group_source = False`: fails both active cases;
- remove `to_dtype` from the allowlist: fails both active cases.

Both mutations reproduce the frozen F1 NVFP4 source hash. Current F2a moves the
float8-to-float32 cast before the group-to-child broadcast:

| Form | Persistent SHA256 | Looped SHA256 |
| --- | --- | --- |
| F1 / either mutation | `4b368d4c...` | `721a7f98...` |
| F2a | `ea9ba825...` | `a32b060f...` |

Protected factor-4 forms are byte-identical to F1 after temporary-path
normalization:

| Form | Persistent SHA256 | Looped SHA256 |
| --- | --- | --- |
| MXFP6 4:3 | `74c9e1fa...` | `634c9c3b...` |
| MXFP6 internal source | `44db1170...` | `74620748...` |

Full scheduler/nested suites and the requested performance measurements remain
release gates, but they do not block freezing this production implementation
for those runs.
