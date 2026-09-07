# Narrow cast-before-broadcast review

Date: 2026-08-26

Reviewed prototype:
`/data/users/eellison/pytorch/agent_space/f2a_narrow_prototype_wt`

Final concrete snapshot reviewed:

- HEAD: `6b8ef64bd373b221c6ed1ba4f4b2f1edef2643c3`
- production delta: `torch/_inductor/codegen/simd.py`, `+71/-4`, net `+67`
- production diff SHA-256:
  `a165db8f8e54db06476b9a2828e0df4aed41afd661364e2fc4a2012b81dcc3ab`
- full unstaged diff SHA-256:
  `7ab016dc72657354c894e0252d74bc3d0bd8a9c5f4ab35423bf36785236e82b0`

## Verdict

The narrow design is sound and is preferable to the current net `+115`
general operation allowlist. It retains the measured NVFP4 code and performance
while deleting the unexercised arithmetic, comparison, inline-assembly, shape
preflight, and masked-callback policy.

I found no semantic blocker. The clarity change found during review has landed:
`to_dtype` is an explicit `_PointwiseRemapHandler` override rather than a string
case inside `_default`. The only allowed algebraic rewrite is now visible in the
method table and has the exact OpsHandler signature. Generic `_default` has one
rule: materialize every group-width operand before delegating.

## Contract reviewed

The proposed contract is:

1. Exact source lookup, guard compatibility, and CSE liveness still happen in
   `_SubParentValueResolver.resolve_sources()`.
2. `resolve_load()` may return the source at group width only when it has no
   parent-lane selection, the consumer is unguarded, and the shape is a true
   group-width shape rather than a direct child-width shape.
3. `to_dtype` converts that value at group width, verifies that the emitted CSE
   result still has group width, and immediately broadcasts it through the
   existing sub-parent materializer.
4. Every other operation and every store widens group-width operands before
   normal codegen.
5. Masked consumers remain on F1's eager resolution path. No masked callback
   wrapper or callback state is needed.

That is a closed rule rather than a general lazy-expression mechanism.

## Semantic equivalence

For the supported operation, the transformation is:

```text
cast(broadcast(x))  ->  broadcast(cast(x))
```

The broadcast only replicates elements, and `ops.to_dtype` converts each element
independently. Therefore the two expressions produce the same value and bits at
every valid child lane. This covers the relevant FP8-to-FP32 conversion and the
other deterministic dtype conversions implemented by Triton, including the
`src_dtype` and `use_compute_types` variants. There is no random state,
cross-lane reduction, or lane-dependent rounding in `to_dtype`.

The concrete NVFP4 result is exactly the intended rewrite:

```text
F1: FP8 -> uint8 bitcast -> broadcast -> FP8 bitcast -> FP32
narrow: FP8 -> FP32 -> broadcast
```

All 12 NVFP4 normalized source hashes in the narrow capture equal the current
general F2a hashes. Packed output bytes and scales also match exactly in the
four paired replay checks completed while this review was running.

## Exact sources and lifetime

The raw-value exit remains inside the existing ordered source-alternative loop.
It occurs only after exact `MemoryDep` lookup, guard matching, and
`cse.contains_value()` have succeeded. Consequently:

- no name-only forwarding is restored;
- a dead source is skipped;
- a noneligible source still falls through to eager materialization and then to
  later source alternatives;
- the required-source failure still runs only after all alternatives fail.

The cast result is widened immediately, so no new deferred object or lifetime
state survives an operation boundary. Both the source and derived value remain
ordinary `CSEVariable` objects governed by the existing CSE lifetime.

Returning the first eligible exact alternative is valid because all alternatives
belong to the same planned consumer relation. The cast preserves that equality.

## Lane and direct-width geometry

`parent_lane is None` is required. Any full-parent source that needs a selected
lane continues through `materialize_source()` and the existing split path.

`is_group_width_shape()` also excludes
`child_block(sub_parent_factor)`. This preserves F1's direct-before-broadcast
precedence when `group_size == factor`: a value already at child width is used
directly and is not mistaken for a group value merely because the extents
coincide.

The focused G-equals-factor test passes in both CPU and CUDA-instantiated test
classes. The helper shares `_GroupedReductionLayout.parent_dim()` with eager F1
materialization, so the narrow path does not introduce a second rank or
singleton-passthrough classifier.

## Guards, masks, and tails

The narrow raw-source check correctly uses:

```text
resolved.consumer_guard.mask is None
```

It must not use `consumer_guard_is_deferred()`, because that broader predicate
also accepts an `ops.masked` callback whose fill is owned by a later outer
`where`.

For a masked consumer, `resolve_load()` therefore uses the unchanged F1 path:

- explicit-`where` callbacks eagerly widen the value, then the outer `where`
  applies the predicate and fill;
- scalar-fill direct-load callbacks eagerly widen and use
  `_apply_consumer_guard()`;
- lane-selected sources remain eager independently of masking.

No custom `masked()` override is required. A load evaluated inside the callback
observes the active load mask and cannot take the raw-source exit. A captured
group value is also safe: the only narrow operation is a pure cast and its
result is widened immediately; every other callback operation goes through the
eager generic path.

Mask metadata is equivalent in both orders:

```text
eager:  source -> child broadcast assigns target masks -> cast inherits them
narrow: source -> cast inherits source masks -> child broadcast assigns target masks
```

The final value in both cases owns only the masks derived from the child family.
The planner's divisibility proof makes those target masks exact. Invalid tail
lanes may execute a pure cast before being discarded, but no memory access or
side effect is introduced.

## Store and operation boundaries

The explicit `store()` override widens before remapping and delegation, so the
resolver records the child-width value actually stored. `store_reduction` and
all other operations reach generic `_default`, which recursively materializes
group-width leaves in positional and keyword arguments before delegation.

This makes unknown operations fail closed by construction. There is no purity
allowlist, no packed-inline-assembly exception, and no shape-propagation
preflight to maintain. A future graph may widen earlier than optimal, but it
cannot accidentally carry a group value through an unreviewed operation.

## Dispatch audit

The first prototype implemented the exception as:

```python
if name == "to_dtype":
    ...
```

inside `_default`. That worked because `DefaultHandler`'s generated `to_dtype`
method forwards to `_default`, but it hid the only exceptional operation in a
string dispatcher.

The final prototype uses the clearer, more idiomatic explicit override:

```python
def to_dtype(
    self,
    value,
    dtype,
    src_dtype=None,
    use_compute_types=True,
):
    group_width = self._is_group_width_value(value)
    result = self._inner.to_dtype(
        value,
        dtype,
        src_dtype=src_dtype,
        use_compute_types=use_compute_types,
    )
    if not group_width:
        return result
    if not self._is_group_width_value(result):
        raise AssertionError("ops.to_dtype did not preserve group width")
    return self._materialize_group_width(result)
```

The implementation correctly calls `self._inner.to_dtype`, not
`super().to_dtype`: the inherited generated method would call this handler's
generic `_default` and materialize the input before casting.

`_default` now has only the ordinary eager rule. Its preliminary `tree_any`
scan was also removed: when a resolver is present, one recursive `tree_map`
applies the no-op-or-materialize helper to all arguments. This states the rule
directly, avoids two traversals when a group value is present, and lowers the
generic dispatcher to CC 2.

The unit test calls `handler.to_dtype(...)` with both optional arguments and
checks the exact delegated call. This verifies the real dispatch contract.

## Complexity comparison

Measured with `agent_space/complexity_audit_ast.py` against staged F1:

| Metric | General F2a | Narrow prototype |
| --- | ---: | ---: |
| Production diff | `+119/-4`, net `+115` | `+71/-4`, net `+67` |
| New methods | 6 | 6 |
| Aggregate CC delta | +26 | +19 |
| Maximum nesting delta | 0 | 0 |
| Largest new dispatcher | 32 LOC, CC 10 | `to_dtype`: 15 LOC, CC 3 |
| New policy set | 16 operations plus asm rule | none |
| Shape-preflight dependency | yes | no |
| Masked callback wrapper | yes | no |

The explicit override leaves the narrow version at roughly half the production
growth of the general version. Generic `_default` is 6 LOC at CC 2, compared
with 32 LOC at CC 10 in the general implementation.

The test delta also drops from `+163` scheduler lines to `+76`, while retaining
the generated NVFP4 form oracle and the direct-width geometry check.

## Evidence checked

- Prototype `git diff --check`: pass.
- Prototype changed-file `py_compile`: pass.
- `spin quicklint`: pass.
- Focused group-width/direct-width tests rerun by this reviewer: 4 passed
  across CPU and CUDA-instantiated classes.
- Implementer focused set: 6 passed.
- NVFP4/MXFP4 kernel-form set: 8 passed.
- Full scheduler file: 126 passed, 6 skipped.
- Full nested-reduction file: 389 passed, 8 skipped.
- Twelve post-cleanup NVFP4 normalized sources: byte-equivalent to general F2a.
- Ten protected factor-2/factor-4/swizzle/preshuffle sources: hashes equal the
  existing F1/general-F2 reference corpus, one kernel each.
- Representative paired replays completed so far preserve exact outputs and
  the general F2a resource/performance result:
  - `4096x4096` default: `20.50 -> 16.43 us`, registers `30 -> 28`;
  - `4096x4096` looped: `20.53 -> 18.47 us`, registers `128 -> 122`;
  - `4096x4608` persistent: `512.28 -> 173.70 us`, spills `1404 -> 724`;
  - `4096x8192` default: `40.95 -> 28.76 us`, registers `30 -> 28`.

## Recommendation

Adopt the narrow cast-before-broadcast implementation. The explicit `to_dtype`
cleanup is included in the reviewed snapshot. It preserves the demonstrated
optimization, keeps all unmeasured behavior on the established eager path, and
removes about half of F2a's production code plus its broadest maintenance
policy.

No semantic blocker remains. The post-cleanup source comparisons and full
scheduler/nested suites are complete.
