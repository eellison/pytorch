# F2a lazy group-width CSE: final-delta review

Date: 2026-08-26

Worktree: `/data/users/eellison/pytorch/agent_space/followup_lazy_projection_wt`

Reviewed snapshot:

- HEAD: `6b8ef64bd373b221c6ed1ba4f4b2f1edef2643c3`
- staged F1 baseline: `195484cf6db6ebb7f4d3e9f649ce361ff4fc80d9e1de6027d3a5237649b658e1`
- unstaged F2a delta: `4ede6cc1a0bef8722e6ced466e433860c0a1ab160dfd1972b8e237b2adbd488a`

## Findings

No blocking production finding.

One test-only cleanup remains at
`test/inductor/test_inductor_scheduler.py:766-768`:
`resolver.consumer_guard_is_deferred.side_effect` is assigned on the handler
mock but never consulted. The resolver guard cases below use a different
`source_resolver`. Those three lines can be deleted; they do not justify a
production change or block freezing the implementation.

## Correctness review

`_SubParentValueResolver.resolve_load()` has one narrow raw-value exit. It is
reachable only after exact relation lookup and live-CSE selection, and then
requires all three facts: no parent-lane selection, no scalar fill that must be
applied locally, and an actual non-direct group-width shape. Every other source
continues through F1 materialization, later source alternatives, and the
unchanged required-source failure. This does not reopen name-only forwarding.

`_PointwiseRemapHandler._default()` is fail-closed. It first detects an actual
group-width operand, requires a positive operation policy, and requires the
canonical shape handler to predict a group-width result. Any unknown,
mixed-width, packed, impure, wrong-rank, or unsupported operation widens its
group operands before backend emission. A narrow emission is checked against
the actual returned `CSEVariable.shape`, so shape/backend drift fails at
compile time.

The `masked()` override closes the callback hole: group-width values created
inside the callback are widened before Triton's final `where`, while copying
`body.graph` preserves Triton's existing direct-load-versus-explicit-where
decision. Scalar-fill masked loads do not take the raw-source exit and retain
F1's explicit guard application. Every widening reuses
`materialize_value_at_sub_parent_resolution()`, so target-family masks replace
source masks rather than being copied or unioned.

Raw and derived values remain ordinary `CSEVariable`s. Source eligibility is
still guarded by `cse.contains_value()`, and eventual broadcasts use normal
Triton CSE keys, including the active load mask. No deferred wrapper can survive
a loop flush or carry stale liveness independently of CSE.

## Complexity and scaffolding

The F2a production delta is `+119/-4`, net `+115`, entirely in
`torch/_inductor/codegen/simd.py`. It adds six methods, no class/type, no cache,
no scheduler argument or policy, and no public API. The largest new method is
`_PointwiseRemapHandler._default()` at CC 10 and nesting depth 2; aggregate
cyclomatic complexity rises by 26 with no increase in file-wide maximum
nesting.

No production helper can be removed without either duplicating the
group/child geometry or hiding a fail-closed boundary. The two small handler
adapters are reused by argument traversal, stores, and masked callback results;
the resolver shape/materialization pair owns the factor and family state that
the handler should not duplicate.

No projection enum, projection record, codegen name view, remapped-value map,
or forwarding-name set has reappeared. The remaining
`broadcast_source_names` occurrences are planner-local construction variables
from staged F1, not F2 codegen state or API.

The test delta is `+187/-10`. Its main policy table is large but covers distinct
positive-policy, shape-barrier, asm, store, and callback contracts. Apart from
the dead mock assignment above, further compression would mostly hide the
policy rather than reduce implementation complexity.

## Verdict

**Freeze the F2a production implementation.** No code change is required from
this review. Overall acceptance still depends on the already-running full
suites, differential fuzzing, protected factor-4 source checks, and paired F1
versus F2 performance results. I did not start competing GPU-heavy tests.
