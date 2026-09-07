# Nested Reduction: Non-Local Optimum

## Goal

The local optimum for nested reduction is:

- use standard kernel load/store/indexing for reduced-output consumers
- use first-class derived range trees for the reduced-output and half-resolution paths
- keep the rest of the nested-reduction machinery specialized

The non-local optimum is larger:

- derived iteration spaces are a first-class scheduler and codegen concept
- pointwise consumers always run by activating a derived iteration family
- register-backed values and remapped memory loads use the same consumer path
- the same abstraction eventually supports reshape-derived spaces and affine masked subregions


## Core Model

A fused kernel has one or more **iteration families**.

Each family defines:

- active range trees
- index substitutions from logical body vars to active tree vars
- a validity domain for loads/stores in that family
- legality constraints on loads/stores in that family
- optional register-backed value sources keyed by buffer name

Examples:

- base family: the outer reduction tile
- reduced-output family: one value per group
- full-resolution lifted family: original tile resolution
- half-resolution family: parent axis divided by 2
- future affine subregion family: e.g. `big_col in [512, 576)` with `small_col = big_col - 512`


## Architecture

### 1. Scheduler owns derived spaces

Today, nested-reduction codegen reconstructs several derived spaces late.

The non-local optimum is:

- the scheduler computes and stores the derived families on the fused node
- legality is decided once, before codegen
- codegen only executes the precomputed families

Family construction is scheduler-owned.
Family activation remains a codegen concern:

```python
with family.activate(kernel):
    ...
```

That keeps the scheduler responsible for legality and profitability while
keeping the codegen API local and explicit.

This moves information such as:

- grouped reduction layout
- full-resolution legality
- half-resolution legality
- future affine-subregion legality

out of late codegen inference and into explicit node state.


### 2. One consumer path for all remapped pointwise

Today, nested reduction still has specialized paths for:

- reduced-resolution epilogues
- full-resolution epilogues
- half-resolution consumers

The non-local optimum is:

- activate family
- build the value environment for that family
- run pointwise bodies through one remap handler

The handler does only three things:

- return register-backed values when available
- remap indices through the active family
- fall back to normal kernel load/store

There is no separate half-resolution context or phase-specific load logic.
What remains specialized is only how a given family populates its value sources.


### 3. Derived families, not ad hoc phase objects

Today, `_ReducedOutputSpace` and `_HalfResolutionSpace` are close, but still separate.

The non-local optimum is one generic object, conceptually:

```python
DerivedIterationFamily(
    range_trees=...,
    index_subs=...,
    value_map=...,
    output_name=...,
)
```

Optional helpers can build common families:

- grouped output family
- lifted full-resolution family
- half-resolution family
- affine subregion family

But codegen consumes one abstraction.


### 4. Affine masked subregions are supported

This is the step that makes the abstraction useful beyond nested reduction.

Current nested reduction mostly needs factorization:

- `RBLOCK -> [RBLOCK // G, G]`
- `RBLOCK -> [RBLOCK // 2, 2]`

The non-local optimum also supports affine masked remaps such as:

- `small_col = big_col - 512`
- valid only when `512 <= big_col < 576`

That is the shape needed for future cat/scatter/MLA-style vertical fusion.

At that point, "derived iteration space" is not just a nested-reduction trick. It is a general fusion tool.


## Value Environment

Each family should have a single value environment:

- `buffer name -> remapped value source`

A source may be:

- direct CSE value
- split tuple of CSE values
- broadcast CSE value
- normal memory-backed buffer read

For the near-term design, the source type should stay explicit:

- `dict[str, CSEVariable | tuple[CSEVariable, ...]]`

Split-backed values are still index-aware. Lane selection belongs in the value
resolver, not in the family itself.

The consumer should not care which one it is. It asks for `load(name, index)`
and the active family plus value source resolve it.


## Legality

Legality should be attached to the family, not scattered across handlers.

Examples:

- grouped family: the grouped reduction is a valid reinterpretation of one axis
- full-resolution family: lifted values correspond to the original tile
- half-resolution family: split lane is invariant modulo the split factor
- affine family: remapped indices stay in bounds under the family mask

Affine families also need explicit validity semantics: a remapped value may only
be meaningful where the family mask holds.

This is also where future support for more general subregions belongs.


## Profitability

Profitability should also move up a level.

Today, nested reduction has custom heuristics for:

- persistent requirement
- xblock/rblock constraints
- half-resolution support

The non-local optimum is:

- each family contributes constraints and resource costs
- kernel selection considers the full fused family set

That is the only way to reason cleanly about later extensions where multiple families coexist.


## Codegen Shape

At the end, `codegen_nested_reduction()` should read like:

1. build base kernel
2. run outer reduction family
3. run grouped reduction family
4. run consumer families
5. finalize outputs

And each consumer family should be driven by the same mechanism:

```python
with family.activate(kernel):
    with V.set_ops_handler(FamilyPointwiseHandler(...)):
        body(iter_vars)
```

No phase-specific custom indexing logic beyond building the family itself.


## Migration Path

Reasonable staged path:

1. Land the current local optimum.
2. Unify reduced-output and half-resolution into one generic derived-family object.
3. Move legality and family construction onto `FusedNestedReductions`.
4. Teach the generic family model affine masked subregions.
5. Reuse it for future vertical pointwise fusion beyond nested reduction.


## Non-Goals For This PR

This PR should not try to:

- make derived iteration spaces fully generic across all backends
- solve MLA `Q`/`K` heterogeneous sibling fusion
- redesign the scheduler around arbitrary subregion fusion
- absorb combo/template scheduling cleanup into nested-reduction work

Those are follow-on projects.


## Summary

The non-local optimum is not "less code in nested reduction."

It is:

- a first-class derived-iteration-family abstraction
- owned by the scheduler
- consumed by one generic remapped pointwise path
- capable of both factorized and affine masked subspaces

That is the version that unifies nested reduction and future subregion fusion work under one model.
