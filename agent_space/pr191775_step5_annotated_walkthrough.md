# PR #191775 Step 5: annotated final-pass scheduling walkthrough

Worktree:
`/data/users/eellison/pytorch/agent_space/pr191775_layered_split_wt`

## The problem

A looped reduction normally executes its full-resolution body once per
reduction block:

```text
loop 1: load x -> accumulate reduction
loop 2: reload x -> use completed reduction -> write output
```

The MXFP6 epilogue may consume a full-resolution value produced inside the
same fused kernel:

```text
source = realize(pointwise(x))
scale = reduce(x)
packed = pack(source, scale)
```

If `source` runs in loop 1, its CSE value contains only the current reduction
block. It is invalidated when that loop closes and cannot be used by `packed`.

We therefore move `source` and the necessary pointwise dependency closure into
the final reduction loop. The planner decides which nodes move. The existing
codegen scheduler remains responsible for emitting `DisableReduction` and
`EnableReduction` markers.

## 1. Find the final-loop node closure

Read:
[`NestedReduction._order_sub_parent_parent_nodes`](/data/users/eellison/pytorch/agent_space/pr191775_layered_split_wt/torch/_inductor/scheduler.py:769)

Inputs:

- `parent_nodes`: the current topologically ordered parent schedule;
- `internal_source_names`: buffers produced in this kernel and consumed by the
  sub-parent epilogue;
- `numel`, `rnumel`: the parent reduction domain.

Output:

```text
(reordered_parent_nodes, required_post_reduction_index)
```

Returning `None` means the requested placement cannot be proven safe.

### 1a. Map each temporal buffer name to its writer

[`scheduler.py:779`](/data/users/eellison/pytorch/agent_space/pr191775_layered_split_wt/torch/_inductor/scheduler.py:779)

The function builds a writer map from each node's own `read_writes`. Duplicate
writers reject the plan because there would be no unambiguous producer to
move.

### 1b. Seed the final-loop set

[`scheduler.py:787`](/data/users/eellison/pytorch/agent_space/pr191775_layered_split_wt/torch/_inductor/scheduler.py:787)

Every internal source required by the epilogue must have an in-kernel writer.
Those writers form the initial `final_nodes` set.

### 1c. Identify nodes that are legal to move

[`scheduler.py:793`](/data/users/eellison/pytorch/agent_space/pr191775_layered_split_wt/torch/_inductor/scheduler.py:793)

A movable node must:

- be pointwise rather than a reduction;
- use the complete parent `[X, R]` domain;
- not feed any reduction in the parent schedule.

The final condition matters because moving an input of the reduction after the
reduction would make loop 1 incomplete.

### 1d. Close over producers and consumers

[`scheduler.py:817`](/data/users/eellison/pytorch/agent_space/pr191775_layered_split_wt/torch/_inductor/scheduler.py:817)

The fixed-point loop expands `final_nodes` in both directions:

- Downstream: move full-resolution consumers of a moved value as well.
- Upstream: move full-resolution in-kernel producers needed by a moved value.

Upstream traversal uses direct reads so it can stop at an external input or a
completed reduced output. Downstream traversal uses `ancestors` because all
descendants of a moved value must remain after it.

Example:

```text
before: reduction, A, unrelated, B, source
                    A -> B -> source

after:  reduction, unrelated | A, B, source
                              ^ required post-reduction index
```

If `A` also feeds the reduction, it is not eligible and the plan is rejected.

### 1e. Preserve order and prove no backward dependency remains

[`scheduler.py:840`](/data/users/eellison/pytorch/agent_space/pr191775_layered_split_wt/torch/_inductor/scheduler.py:840)

The function performs a stable partition:

```text
leading nodes + final-loop nodes
```

It preserves the original order inside both partitions. It then rejects if a
leading node depends on a moved node. The returned index is the first node that
must execute after a reduction loop has completed.

## 2. Store the boundary in the plan

Read:
[`StagedReductionPlan.required_post_reduction_index`](/data/users/eellison/pytorch/agent_space/pr191775_layered_split_wt/torch/_inductor/scheduler.py:2186)

The field does not mean "always insert a new loop here." It means:

> The node at this index, and the pointwise suffix following it, must share a
> loop that begins after a reduction loop has completed.

An earlier dependency may already have caused ordinary scheduling to open that
final loop.

## 3. Let the existing scheduler realize the boundary

Read:
[`SIMDScheduling.generate_node_schedule`](/data/users/eellison/pytorch/agent_space/pr191775_layered_split_wt/torch/_inductor/codegen/simd.py:2797)

This is the existing routine that walks nodes and emits schedule entries,
including `DisableReduction` and `EnableReduction`. The new optional argument
adds a constraint without creating a second scheduling implementation.

### 3a. Track reduction-loop state

[`simd.py:2812`](/data/users/eellison/pytorch/agent_space/pr191775_layered_split_wt/torch/_inductor/codegen/simd.py:2812)

- `current_loop_has_reduction` says the active loop contains a reduction.
- `completed_reduction_loop` says codegen has already closed at least one
  reduction loop.

`end_current_reduction_loop` updates and resets these alongside its existing
CSE/readiness state.

### 3b. Enforce the required post-reduction point

[`simd.py:2886`](/data/users/eellison/pytorch/agent_space/pr191775_layered_split_wt/torch/_inductor/codegen/simd.py:2886)

At `required_post_reduction_index`, codegen verifies that the complete suffix
contains only main-body pointwise nodes. It then has three cases:

```text
current loop still contains a reduction -> close it once
an earlier dependency already closed it -> reuse the open final loop
no reduction loop has completed          -> assert planner/codegen mismatch
```

It also discards an uncommitted heuristic split at this point. Otherwise the
heuristic could introduce an extra loop between the internal source and its
epilogue.

## 4. Pass the planned boundary into codegen

Read:
[`_codegen_reduction_with_sub_parent_epilogue`](/data/users/eellison/pytorch/agent_space/pr191775_layered_split_wt/torch/_inductor/codegen/simd.py:3710)

[`simd.py:3731`](/data/users/eellison/pytorch/agent_space/pr191775_layered_split_wt/torch/_inductor/codegen/simd.py:3731) makes one call:

```python
parent_schedule = self.generate_node_schedule(
    parent_nodes,
    numel,
    rnumel,
    required_post_reduction_index=plan.required_post_reduction_index,
)
```

There is no separate leading/deferred schedule construction and no manual
marker stitching. The ordinary scheduler owns all loop-state transitions.

## 5. Resulting schedules

Normal case:

```text
reduction, source
```

becomes:

```text
reduction, DisableReduction, EnableReduction, source
```

If ordinary dependency scheduling already opened the final loop:

```text
reduction, post_reduction, source
```

becomes exactly:

```text
reduction, DisableReduction, EnableReduction, post_reduction, source
```

It does not create a third loop before `source`.

## 6. Tests to inspect

Marker-level scheduler tests:

- [required-boundary tests](/data/users/eellison/pytorch/agent_space/pr191775_layered_split_wt/test/inductor/test_inductor_scheduler.py:161)

End-to-end looped tests:

- [source computed in the second pass](/data/users/eellison/pytorch/agent_space/pr191775_layered_split_wt/test/inductor/test_nested_reduction.py:1873)
- [source chain may consume a completed reduction](/data/users/eellison/pytorch/agent_space/pr191775_layered_split_wt/test/inductor/test_nested_reduction.py:1898)
- [reuse an already-open final pass](/data/users/eellison/pytorch/agent_space/pr191775_layered_split_wt/test/inductor/test_nested_reduction.py:1925)
- [close over work needed by a later reduction pass](/data/users/eellison/pytorch/agent_space/pr191775_layered_split_wt/test/inductor/test_nested_reduction.py:1952)
- [reject a source chain also needed by a reduction](/data/users/eellison/pytorch/agent_space/pr191775_layered_split_wt/test/inductor/test_nested_reduction.py:2091)

The positive looped tests assert that generated code contains exactly two
`tl.range` loops.
