# F2a lazy group-resolution CSE: minimal implementation design

Date: 2026-08-26

Reviewed worktree: `agent_space/followup_indexed_rebase_wt`

Snapshot:

- HEAD: `6b8ef64bd373b221c6ed1ba4f4b2f1edef2643c3`
- `torch/_inductor/codegen/simd.py`:
  `e52dbf70c9bb6545c6fcd6f8b38cfaf9b7582488a525432439ef052c0e6971bc`

## Recommendation

Implement F2a entirely in `torch/_inductor/codegen/simd.py`, on the existing
`_SubParentValueResolver` and `_PointwiseRemapHandler`. The delayed value stays
an ordinary `CSEVariable`; `CSEVariable.shape` is the only state that identifies
a group-width intermediate. Add no wrapper, domain enum, custom CSE cache, or
negative barrier inventory.

The design fits in about 100-130 net production lines. It needs one small
resolver hook, two geometry helpers, and handler-local operation interception.
There is no reason for a factor-2 gate and no reason to add `reciprocal` to the
initial allowlist.

The one correctness trap is `masked`: a local callback wrapper is necessary,
but it must preserve `body.graph`, and raw guarded sources are allowed only when
the existing F1 guard says the enclosing Triton `masked` chose its explicit
outer-`where` path. A scalar-fill direct-load path must stay eager.

## Existing F1 Contract To Preserve

The current flattened F1 already has the needed boundary:

- `_ResolvedSubParentSource` at `simd.py:1550` carries `key`, live `value`,
  `parent_lane`, and `consumer_guard`.
- `_SubParentValueResolver.resolve_sources()` at `simd.py:2538` returns every
  live exact alternative, in source/guard order.
- `materialize_source()` at `simd.py:2563` owns eager direct/broadcast/split and
  consumer guard application.
- `resolve_load()` at `simd.py:2585` tries all alternatives before applying the
  required-miss policy.
- `_codegen_sub_parent_output_groups()` at `simd.py:3883` is the only replay
  path that installs the resolver in `_PointwiseRemapHandler`.

Keep `_ResolvedSubParentSource.value`. F2a needs to inspect and return that live
value before eager materialization. Removing it would require reopening the
resolver's private value map or introducing a new deferred wrapper, both worse
interfaces. Do not add any other field or value record.

## Smallest Resolver Seam

Extend `resolve_load()` with one optional predicate:

```python
def resolve_load(
    self,
    relation: scheduler.SubParentAccessRelation,
    *,
    preserve_source: Callable[[_ResolvedSubParentSource], bool] | None = None,
) -> CSEVariable | None:
    for resolved in self.resolve_sources(relation):
        if preserve_source is not None and preserve_source(resolved):
            return resolved.value
        value = self.materialize_source(resolved)
        if value is not None:
            return value
    ...  # unchanged required miss / optional fallback result
```

This is preferable to restoring a singular `resolve_source()` API. A singular
selection previously lost valid later alternatives when the first live source
could not materialize. The predicate is evaluated only for an already-live,
exact source; the resolver retains ordering, liveness, and requiredness.

The handler predicate should accept a raw source only when all are true:

```text
resolved.parent_lane is None
resolver.consumer_guard_is_deferred(resolved)
resolved.value.shape is a non-direct group-width shape
```

`parent_lane != None` therefore remains the unchanged eager F1 split path.
Direct child-width values, scalars, unknown shapes, and unsupported ranks also
remain on F1's eager path.

## Shape And Materialization Helpers

Add two small methods to `_SubParentValueResolver`, which already owns the
layout, target family, and factor:

1. `is_group_width_shape(shape)` recognizes rank-2 values through
   `layout.parent_axis`, and rank-1 values only under the same singleton
   passthrough proof used by `layout.parent_dim()`.
2. `materialize_group_width(value)` leaves non-group values unchanged and calls
   `layout.materialize_value_at_sub_parent_resolution(...)` for a recognized
   group value. It asserts that the result is one `CSEVariable`, not `None` or a
   lane tuple.

The group predicate must preserve F1's direct-before-broadcast precedence:

```text
parent_dim == layout.num_groups_str
and parent_dim != layout.child_block(sub_parent_factor)
```

The second condition matters for `G == factor`, where group and child widths
can coincide. Such values are already concrete child values and must not acquire
an artificial broadcast.

`materialize_group_width()` must reuse the current layout materializer rather
than emit a second broadcast implementation. That path already calls
`family.set_value_masks(kernel, ...)`, replacing source masks with masks derived
from the child result shape. Do not copy or union `mask_vars`.

## Handler Policy

Add the following positive allowlist as a class constant on
`_PointwiseRemapHandler`:

```text
to_dtype
abs, neg
add, sub, mul, truediv
minimum, maximum
eq, ne, lt, le, gt, ge
where
```

Handle `inline_asm_elementwise` separately because it is allowed only when
`pack == 1` and `is_pure is True`.

Add these handler methods:

- `_is_group_width_value(value)`: `CSEVariable` plus the resolver's shape test.
- `_materialize_group_width(value)`: delegate to the resolver, or return the
  input unchanged when no resolver is installed.
- `_keep_group_source(resolved)`: the exact lane/guard/shape predicate above.
- `_default(name, args, kwargs)`: perform the positive-policy check and delegate.
- `masked(mask, body, other)`: the callback boundary described below.

Modify the existing methods only as follows:

- `load()` calls `resolve_load(..., preserve_source=self._keep_group_source)`.
- `store()` materializes a group-width value before index remapping and before
  delegating to the resolver. Consequently, exact store recording sees the
  child-width value that was actually stored.

Do not add a separate `store_reduction()` override. It is generated by
`DefaultHandler` and reaches `_PointwiseRemapHandler._default`; because it is
absent from the positive allowlist, its value is materialized before the
existing resolver records the reduction store. A direct test should pin this
dispatch property.

### Pre-operation shape proof

When `_default()` sees at least one group-width operand:

1. Reject the operation unless it is in the positive allowlist, or it is pure
   pack-1 inline asm.
2. For an otherwise allowed operation, call the same
   `ShapePropagationOpsHandler` operation that `CSEProxy` will use. Keep the
   operands narrow only if the predicted result is one supported group-width
   shape. Treat `None`, wrong rank/domain, multiple shapes, or an
   `AssertionError`/`TypeError`/`NotImplementedError` as a barrier.
3. On a barrier, `pytree.tree_map()` the resolver's group materializer over
   `args` and `kwargs`, then delegate normally.
4. On the narrow path, delegate normally and assert the actual result is one
   group-width `CSEVariable`. This turns any backend/shape-contract drift into a
   compile-time failure rather than a silently escaped lazy value.

Using the canonical shape handler is smaller and stricter than recreating the
historical `_is_lane_varying()` rules. In particular, a group plus child-shaped
operand currently fails broadcast-shape inference before materialization; after
the handler widens the group operand, normal child-shape propagation succeeds.

No F2 object is attached to the result. `CSEProxy._default()` records the
ordinary result shape, and the next handler operation rediscovers group width
from that shape.

## `masked` Callback Audit

`LoopBody.bind_masked_shim()` passes a `LoopBodyBlock` callback to
`V.ops.masked`. The group value can be created entirely inside that callback,
so scanning only the explicit `mask/body/other` arguments in `_default()` is
insufficient. Add an explicit handler override:

```python
def masked(self, mask, body, other):
    def materialized_body():
        result = body()
        return pytree.tree_map(self._materialize_group_width, result)

    materialized_body.graph = body.graph  # type: ignore[attr-defined]
    return self._inner.masked(
        self._materialize_group_width(mask),
        materialized_body,
        self._materialize_group_width(other),
    )
```

The `.graph` assignment is required: `TritonOverrides.masked()` inspects
`body.graph` before invoking the callback. A plain closure without it fails.
No callback class, disabled mode, or persistent callback state is needed. The
same remap handler remains installed while `body()` runs, so allowlisted scalar
work can stay group-width; every group-shaped return leaf is widened before the
inner masked implementation sees it.

The current guard representation already distinguishes the two masked paths:

| Triton masked path | Value installed in `kernel._load_other` | F2a action |
|---|---:|---|
| explicit final `where` | `None` | raw group source may be retained; callback widens it and the outer `where` reapplies the predicate/fill |
| direct-load fast path | scalar `other` | `consumer_guard_is_deferred()` is false; use F1 eager materialization plus `_apply_consumer_guard()` |

This follows directly from `triton.py:2541-2584`: Triton decides `need_where`
before invoking the body and enters `kernel.mask_loads(mask, value=None)` only
for the explicit-`where` path. Therefore F2a must not use `_load_mask is not
None` alone as permission to defer, and it does not need to duplicate Triton's
graph scan. The scalar-fill path would otherwise bypass the physical load and
silently drop the consumer fill.

Materialize a group-shaped explicit mask before entering the inner callback.
`other` is specified as a constant, but applying the same leaf conversion is a
cheap defensive no-op for conforming callers. Nested masked callbacks work by
ordinary closure nesting; each boundary widens its own returned group leaves.

## `inline_asm_elementwise(pack=1)` Audit

The OpsHandler signature has variadic positional tensor inputs and keyword-only
metadata. `lower_inline_asm_elementwise()` passes CSE inputs positionally and
passes `asm`, `constraints`, `dtype`, `is_pure`, `pack`, and `input_dtypes` in
kwargs. `ShapePropagationOpsHandler` has no special override, so it computes the
broadcast shape from the positional tensor inputs only.

The allow condition must be exact:

```python
name == "inline_asm_elementwise"
and kwargs.get("pack", 1) == 1
and kwargs.get("is_pure", True) is True
```

Do not mirror Triton's broader `pack <= 1` implementation. `pack=0`, `pack>1`,
an impure call, a child-shaped co-operand, unknown shape, wrong rank, or a
multiple-result shape is a barrier. The public higher-order op currently
rejects impure calls and enforces one output, but the lower OpsHandler boundary
should remain fail-closed independently.

Pack 1 is genuinely elementwise: the backend returns the direct inline-asm
expression and `CSEProxy` assigns the broadcasted positional-input shape. For
`pack > 1`, the backend emits `inline_asm_pack`/`inline_asm_unpack` and bases the
result on the first input shape, so it must see child-width operands.

## Scope Decisions

### Do not add `reciprocal`

A pre-fusion LoopBody dump of the current target graphs showed:

- NVFP4 scale consumption: `load -> to_dtype -> truediv`; there is no
  `reciprocal` op.
- MXFP4 scale reciprocal: `load -> to_dtype ->
  inline_asm_elementwise(pack=1, is_pure=True) -> store`, followed by a
  lane-shaped `mul` in the consuming body.
- MXFP6 4:3 quantization: the group scale meets the lane value at `truediv`,
  followed by `round` and bitwise packing.

Therefore `reciprocal` and `rsqrt` should remain barriers. Adding an unobserved
operator weakens the positive policy without helping these kernels. If a future
target graph emits `reciprocal`, add it only with its shape-contract row and a
kernel-form reason.

### Enable group-only deferral for every planned factor

Do not gate F2a on factor 2. The safety proof depends on exact `lane=None`
resolution, live shape, guard state, and operation shape, not on factor. The
existing layout emitter already handles the planned factor.

Current factor-4/MXFP6 graphs naturally remain eager-equivalent: their first
group-scale use is mixed with a lane-shaped value at `truediv`, so the
pre-operation shape check widens before the operation. `round`, shifts, and
bitwise packing are not allowlisted. Historical persistent and looped MXFP6
captures were byte-identical with group-only deferral enabled.

A factor-2-only gate would be policy without a correctness basis and would
retain duplicate work for a future factor-4 graph that does contain a genuine
group-only scalar chain. Keep the factor-2 restriction solely for optional F2b
parent-lane deferral. If measurement later finds a factor-specific regression,
use a measured profitability gate rather than changing F2a semantics.

## Masks, Stores, Fallback, And Lifetime

- Group-width scalar results retain the masks naturally propagated by ordinary
  `TritonCSEVariable.update_on_args()` while they remain group-width.
- Every widening calls the existing layout materializer, whose
  `family.set_value_masks()` assignment replaces masks using the target child
  shape. Never update/union source masks onto the widened result.
- `store()` widens before delegation, so `_SubParentValueResolver.store()`
  records the concrete child value. `store_reduction()` is an automatic
  `_default` barrier and records the same way.
- A resolver miss is unchanged: a required relation raises after every live
  alternative fails; an optional external relation falls through to
  `_load_without_store_forwarding()`. F2a must not consult generic
  `store_cache[name]` or reintroduce a name-keyed fallback.
- Do not change pre-epilogue flush suppression. It remains limited to required
  `parent_lane != None` relations. A `parent_lane is None` group source must not
  hold a reduction loop open.
- F2a owns no cache. A `codegen_body()`/CSE invalidation makes old source values
  unavailable through the existing `contains_value()` check. Derived group
  expressions and broadcasts are ordinary CSE expressions in the current pass;
  a CSE miss only duplicates work and cannot change correctness.

## Compact Permanent Test Plan

Keep the permanent addition to two new CPU test methods, then strengthen
existing GPU/kernel-form methods.

1. Add one parameterized/subtest method in
   `test/inductor/test_inductor_scheduler.py` for the group-width operation
   policy.
   - Run every allowlisted operation through the current
     `ShapePropagationOpsHandler`; group plus scalar-compatible inputs must
     produce exactly the group shape.
   - Assert the case table equals the production allowlist so additions require
     a test row.
   - Exercise group plus child, `shape=None`, wrong rank, `square` as a known but
     unapproved op, tuple-valued `frexp`, `store_reduction`, pure pack 2, and
     impure pack 1. Each must widen before the inner operation.
   - Exercise pure pack 1 and verify it remains group-width until a store/mixed
     operand, then widens once.
   - Include the `G == factor` direct-width coincidence and assert no broadcast.

2. Add one CPU method for `masked`.
   - The callback contains/returns a group-width scalar chain but there is no
     group CSE in the explicit masked arguments.
   - The local callback preserves the original `.graph`, executes with the same
     handler, and returns a child-width value after exactly one widening.
   - Cover `consumer_guard.fill is None` (raw source admitted) and scalar fill
     (F1 eager guard application; raw source rejected). Exercise nesting or an
     exception path if callback state is introduced; the recommended design has
     no such state.

3. Extend, rather than duplicate, existing integration coverage.
   - `test_mxfp4_inline_asm_kernel_form`: require one scale conversion and one
     reciprocal pack-1 asm, in order before the group-to-child broadcast.
   - `test_nvfp4_inline_asm_kernel_form`: require the one FP8 conversion before
     the group-to-child broadcast.
   - `test_mxfp6_four_to_three_pack_kernel_form` and the internal-source variant:
     retain current persistent/looped broadcast and split counts, proving the
     all-factor policy does not perturb factor 4.
   - Strengthen `_check_sub_parent_indirect_index_mask` to reject parent/reduced
     masks on the gather/assert lines, not merely require the lane-family mask.
   - Keep the existing mismatched masked-source numeric test as the scalar-fill
     regression and the existing dynamic/tail tests as the explicit-`where`
     path.

The full scheduler and nested-reduction suites remain required. Normalize the
temporary source-file path before byte/hash comparison of protected MXFP6 and
other unchanged kernels.

## Production Budget

Expected net additions:

- imports and one allowlist: 15-20 lines;
- resolver predicate hook plus two geometry/materialization helpers: 25-35;
- handler load/store changes, shape preflight, and masked callback override:
  55-70;
- assertions/comments: 10-15.

Total: approximately 105-140 net production lines, all in `simd.py`. If the
implementation grows beyond 150, the likely excess is a new wrapper/domain
abstraction, duplicated shape algebra, a broadcast cache, or per-operation
methods; none is needed for F2a.

## Stop Conditions

Do not land F2a if any of these is required to make it work:

- a tagged deferred value or domain enum;
- a second expression or broadcast cache;
- reopening source/consumer maps outside `_SubParentValueResolver`;
- name-based forwarding or mask exceptions;
- copying source masks to a child-shaped value;
- a negative barrier list;
- a factor-2-only group rule; or
- adding `reciprocal` without an actual target LoopBody op and shape-contract
  test.

Within those boundaries, no design blocker remains.
