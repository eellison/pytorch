# PR191775: materialized reload versus deferred register forwarding

## Conclusion

The ordinary store, CSE invalidation, barrier, and reload path can make looped
sub-parent epilogues correct. It is not a good replacement for
`_order_sub_parent_parent_nodes` in this PR if the performance target remains
important.

The replacement removes roughly 100 lines of production planning/state, but it
turns the internal source into a real global-memory temporary. For the MXFP6
cases tested here, that adds a third R pass, four child loads, one full-resolution
store, a temporary allocation, and at least one CTA barrier. The generated
kernel is also about 0.8-0.9 KB larger.

The useful role for this mechanism is a future fallback for cases that cannot be
legally deferred, not the default replacement for the current fast path.

## Why the generic path is correct

After a looped reduction pass, `TritonKernel.codegen_body()` invalidates
loop-local CSE values and moves their stored buffer names into
`cse.invalidated_stores`.

When the derived epilogue later reloads such a buffer:

1. `CSEProxy.load()` adds it to `kernel.must_keep_buffers`.
2. Kernel-local buffer removal therefore preserves its pointer, store, and
   wrapper allocation.
3. `TritonKernel.load()` emits `tl.debug_barrier()` before the reload.
4. The sub-parent projection proof already requires a unique, dense, injective
   write and preserves the parent X boundary. The parent store and child reload
   therefore belong to the same program instance; only threads/warps within that
   CTA exchange elements.
5. `tl.debug_barrier()` synchronizes all threads in the block, matching the
   existing reduction-loop readback mechanism and its regression test.

No grid barrier is required. A cross-program projection would be unsafe, but the
existing frame and write proofs reject that relation.

Tail safety is unchanged: the parent store uses the parent mask, while the child
reload uses the exact derived-family mask. Divisibility proves that every valid
child lane maps to a valid stored parent element.

## Required implementation plumbing

The minimal functional version needs:

1. Stop reordering the internal source chain and remove
   `deferred_parent_start`.
2. Flush the looped parent schedule before emitting the sub-parent epilogue, so
   reductions are finalized and loop-local store-cache values are invalidated.
3. Let a planned in-kernel source miss fall through to a real load after that
   flush. The current loud-on-miss rule must remain before the flush, so this
   needs explicit phase state rather than deleting the assertion globally.
4. Let the existing `CSEProxy.load()` path retain the buffer and insert the RAW
   barrier.

For acceptable generated code, more plumbing is needed:

- The stock path emits a barrier for every load attempt, including repeated CSE
  hits. Factor-4 packing produced six barrier statements per R-loop iteration.
  Clearing the invalidated state after its first barrier reduces this to one per
  buffer per loop, but changes general Triton CSE behavior.
- A specialized barrier can instead be emitted once between the two generated
  loops, followed by explicit `must_keep_buffers` marking and invalidation-state
  clearing. That is faster than repeated in-loop barriers, but is no longer the
  unmodified ordinary path.
- Fusion scoring and the looped memory estimator currently treat the
  intermediate as eliminated. A real fallback should account for the new store,
  reloads, allocation, and extra pass so fusion does not receive credit for
  traffic it still performs.

## Prototype

The scratch prototype is:

- `agent_space/prototype_deferred_vs_materialized.py`
- `agent_space/run_pr191775_overlay.py`

It overlays the current PR scheduler/SIMD/common files on the locally built
PyTorch, disables custom deferral, flushes before the derived epilogue, and
allows the planned internal source to reload. All three cases matched separately
compiled non-nested results.

The measurements were taken on an NVIDIA B200. Times are noisy microbenchmarks;
the generated-form differences are the stronger result.

| Case | Form | R loops | Loads | Stores | Splits | Barrier statements | Kernel chars | Event median (us) | Profiled device time (us) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| factor-2 internal source | deferred | 2 | 2 | 2 | 1 | 0 | 12048 | 22.73 | 5.59 |
| factor-2 internal source | materialized, one in-loop barrier | 2 | 3 | 3 | 0 | 1 | 12309 | 21.69 | 15.75 |
| MXFP6 internal source | deferred | 2 | 2 | 5 | 3 | 0 | 22674 | 24.59 | 2.38 |
| MXFP6 internal source | materialized, one in-loop barrier | 3 | 6 | 6 | 0 | 1 | 23468 | 26.60 | 2.47 |
| MXFP6 large group | deferred | 2 | 2 | 4 | 3 | 0 | 23450 | 25.98 | 6.20 |
| MXFP6 large group | materialized, one in-loop barrier | 3 | 6 | 5 | 0 | 1 | 24218 | 27.13 | 8.02 |

Without barrier coalescing, the two MXFP6 kernels contain six barrier statements
inside every R iteration. Their event medians were 30.56 and 30.77 us,
respectively.

## Code-size tradeoff

Removing the current mechanism deletes approximately:

- 68 lines from `_order_sub_parent_parent_nodes`;
- 9 lines from its planner call site;
- 10 lines of `deferred_parent_start` plan state and validation;
- 15 lines of split-schedule construction in codegen.

That is about 100 production lines. A minimal reload implementation adds roughly
10-20 lines, but preserving loud-on-miss, coalescing/hoisting barriers, and
accounting for memory traffic consumes part of that saving. The rejection tests
for source chains used by reductions would also either change into positive
materialization tests or require a remaining planner-side policy check.

## Recommendation

Keep deferred register forwarding in PR191775. It avoids a full-resolution
temporary and lets the derived epilogue share the final parent pass, which is the
main performance property of the internal-source work.

Record materialized reload as a possible future fallback for candidates that
cannot be deferred. Such a follow-up should benchmark against both the deferred
kernel and the unfused two-kernel form, use a single barrier between passes, and
teach fusion scoring about the retained temporary before enabling it broadly.
