# Looped internal-source placement in staged reduction epilogues

## Problem

A looped reduction processes the reduction dimension in blocks. An ordinary
RMSNorm-like kernel naturally has two passes:

```text
pass 1: load x and accumulate the reduction
pass 2: reload x, combine it with the reduction result, and store the output
```

This works without special scheduling because `x` is an external buffer and can
be loaded again in the second pass.

An MXFP6 packing epilogue can instead consume a realized value produced inside
the fused kernel:

```text
source = realize(pointwise(x))
scale = reduction(x)
packed = pack(source, scale)
```

The realization is important. The three packed output bytes share the same
full-resolution conversion, so leaving the expression inline can duplicate that
conversion across output-lane passes.

After fusion, however, the realized buffer may be removed and represented only
by a CSE value. If `source` is emitted inside the first reduction loop, that CSE
value describes only the current reduction block. It cannot be reused after the
loop as though it represented the full reduction dimension.

Therefore looped codegen needs an explicit answer to this question:

> How does the sub-parent epilogue obtain an internally produced,
> full-resolution source after the reduction has completed?

## Existing general fallback

Inductor already supports a store followed later by a reload in one Triton
program:

```text
pass 1: compute source and store a full-resolution temporary
barrier
pass 2: reload the temporary and emit the epilogue
```

`CSEProxy.load()` marks an invalidated store as `must_keep_buffers`, preventing
kernel-local buffer removal. Triton load codegen emits `tl.debug_barrier()` before
the reload so writes performed by other warps in the same program are visible.

This mechanism is correctness-viable for a sub-parent epilogue. The projection
proof preserves ownership of the parent X coordinate, and one looped program
owns the complete parent R span. The derived load therefore redistributes values
only among threads in that same program, where the block barrier applies.

## Current approach

[`NestedReduction._order_sub_parent_parent_nodes`](/data/users/eellison/pytorch/agent_space/pr191775_layered_split_wt/torch/_inductor/scheduler.py:770)
identifies the pointwise chain producing each internal epilogue source. When the
chain is independent of every reduction, it moves that chain to the end of the
parent schedule.

The plan records `required_post_reduction_index`, the first parent node that
must run after a reduction loop has completed. Codegen passes that boundary to
the ordinary `generate_node_schedule` path, which either starts the final loop
there or reuses one that normal dependency scheduling already opened:

```text
pass 1: reduction and everything required by the reduction
pass 2: optional post-reduction work, compute the internal source, then
        immediately consume it in the sub-parent epilogue
```

The value is computed once and forwarded from registers. No full-resolution
temporary is allocated, stored, or reloaded.

The planner declines when the source chain also feeds a reduction. Moving such a
chain would make the first pass invalid. Supporting that case requires either
computing the chain in both passes or using the materialize/barrier/reload path.

## Why not always use materialize/barrier/reload?

The conventional path is simpler at the scheduler level, but it adds:

- one full-resolution temporary allocation;
- a full-resolution store during the first pass;
- a full-resolution load during the second pass;
- one or more block barriers;
- another reduction-dimension loop in the generated kernel in the MXFP6 cases.

For an `int32` MXFP6 source, it writes four bytes per original element and then
reads those four bytes back. The second-pass alternative would otherwise reload
only the original two-byte BF16/FP16 input and compute the source directly.

The focused B200 prototype measured:

| Case | Deferred computation | Materialize/reload | Change |
|---|---:|---:|---:|
| Simple factor-2 internal source | 5.59 us | 15.75 us | 2.82x slower |
| MXFP6 internal source | baseline | about 4-8% slower | regression |
| Large-group MXFP6 | 6.20 us | 8.02 us | 29% slower |

The generated materialization kernels also contain more loads, stores, barriers,
and code. Coalescing repeated barriers helps but does not remove the temporary
traffic.

## Decision

Keep second-pass placement for source chains that are independent of the
reduction. It is the explicit staged-codegen equivalent of the recomputation an
ordinary looped RMSNorm epilogue already performs.

Keep materialize/barrier/reload as a possible future fallback for sources that
must execute in the first pass because they also feed a reduction. It should not
replace second-pass placement as the default.

## Correctness conditions for deferral

The planner must reject unless all of the following hold:

1. Every moved node is pointwise and uses the full parent iteration domain.
2. No moved node feeds any reduction.
3. The complete producer and downstream dependency chain is moved together.
4. No node left in the leading schedule depends on a moved node.
5. The resulting boundary leaves nonempty leading and final-loop schedules.

These conditions are what `_order_sub_parent_parent_nodes` proves. Focused
tests also pin reuse of an already-open final loop without adding a third pass.
