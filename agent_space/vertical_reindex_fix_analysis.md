# Vertical reindex retry for reduction-to-pointwise fusion

## Question

Does the vertical retry from pytorch/pytorch#183521 fix the MXFP8
block-scale broadcast case in the right place, and why did the existing
reindexing not already handle it?

Short answer: yes, the retry is the right kind of fix. Existing reindexing
could produce a nonzero shared-data score, but it did not necessarily
rewrite the pointwise node before the vertical dependency legality check.
The new retry runs at the legality boundary: if normal vertical fusion fails,
it reindexes the pointwise to the reduction domain and then reruns the normal
vertical, choices, and backend checks.

## Concrete failing shape

From `/home/eellison/local/pytorch/work_dir/test_hi20.py`, standalone MXFP8
quantization has:

```text
block_amax reduction -> scale encode pointwise -> fp8 quant pointwise
```

After the block reduction and scale encode are fused, the remaining pair is:

```text
producer: reduction-shaped fused node
  group:  (8192, 32)
  writes: buf0[256*d0 + d1],        ranges {d0: 32, d1: 256}

consumer: elementwise fp8 quant
  group:  (262144, 1)
  reads:  buf0[256*c0 + c1 // 32],  ranges {c0: 32, c1: 8192}
```

These are the same logical scale values. The consumer is reading the
per-block scale after broadcasting it over the 32 element lanes. But the raw
`MemoryDep`s do not match:

```text
write: 256*d0 + d1
read:  256*c0 + c1 // 32
```

So unmodified vertical legality rejects the pair with "memory deps did not
match".

## What the old path did

The relevant scheduler sequence is:

1. Compute `shared_data_score`.
2. If `can_reorder` and the score is low, call
   `shared_data_after_reordering_loop()`.
3. If the final score is acceptable, `V.choices.can_fuse()` allows the pair.
4. Only then run vertical legality through `can_fuse_vertical()`.

`shared_data_after_reordering_loop()` is a scoring/profitability repair path.
It can mutate loops in some cases, but its contract is only "return a better
shared-data score", not "guarantee the final vertical deps are now legal".

For this MXFP8 pair, tracing showed:

```text
SHARED_REORDER op0_op2_op3 -> op1 res 524288
VERTICAL       op0_op2_op3 -> op1 res False
TRY_REINDEX    op0_op2_op3 -> op1 res True
VERTICAL       op0_op2_op3 -> op1 res True
```

The important part is the ordering:

```text
shared_data_after_reordering_loop() returned a score
normal can_fuse_vertical() still failed
the new vertical retry reindexed
can_fuse_vertical() then passed
```

So existing machinery recognized that there was shared data, but it did not
leave the consumer in the reduction iteration domain before vertical legality.

## Why existing reindexing did not fire

Inside `shared_data_after_reordering_loop()`, the first attempt is
`_try_reorder_loops_for_candidates()`. That helper can return a positive score
when deps match after normalization, including cases where FusedSchedulerNodes
have a different number of loop vars from a single SchedulerNode.

When that happens, `shared_data_after_reordering_loop()` returns immediately:

```text
score = _try_reorder_loops_for_candidates(...)
if score >= 0:
    return score
```

It never reaches `_try_reindex_pointwise_for_reduction()`.

For this case, the positive score came from a different shared buffer, not from
the problematic scale dependency. The fused reduction node and the pointwise
consumer both read the original input:

```text
common buffer: arg0_1

producer read: MemoryDep('arg0_1', 8192*d0 + 32*d1 + d2,
                         {d0: 32, d1: 256, d2: 32})
  normalize_with_stride_order -> MemoryDep('arg0_1', t0, {t0: 262144})

consumer read: MemoryDep('arg0_1', 8192*d0 + d1,
                         {d0: 32, d1: 8192})
  normalize_with_stride_order -> MemoryDep('arg0_1', t0, {t0: 262144})
```

That shared input read normalizes to the same dense access, so
`_try_reorder_loops_for_candidates()` returns a score of 524288. Since that is
non-negative, `shared_data_after_reordering_loop()` exits before trying
`_try_reindex_pointwise_for_reduction()`.

The scale dep still does not match:

```text
producer write: MemoryDep('buf0', 256*d0 + d1,        {d0: 32, d1: 256})
consumer read:  MemoryDep('buf0', 256*d0 + d1 // 32, {d0: 32, d1: 8192})
```

So the old path had enough score to reach `V.choices.can_fuse()`, but the
pointwise was not reindexed, and final vertical legality still rejected the
producer-output dependency.

## Why the vertical retry is the right place

The vertical retry is placed after normal vertical fusion fails:

```text
if normal can_fuse_vertical(...) and choices/backend checks:
    return True

if loop_reindexing_after_fusion and reduction -> pointwise:
    reindex pointwise to reduction domain
    rerun can_fuse_vertical(...)
    rerun choices/backend checks
```

That is the right semantic boundary because:

- The failure being repaired is a vertical dependency legality failure.
- The mutation is only attempted for reduction-to-pointwise pairs.
- The config gate is the same loop reindexing gate.
- It reruns normal legality after mutation instead of accepting by special
  equivalence.
- It reruns backend fusion checks, so codegen shape restrictions still apply.
- `_LoopMutationTracker` rolls the speculative mutation back if `can_fuse()`
  ultimately rejects.

This is better than widening `fusable_read_and_write()` for ordinary fusion,
because it turns the pointwise into the normal reduction domain and then uses
the existing exact-dependency path.

## Why this is not just nested-reduction logic

This case is not a nested reduction by itself. It is a regular reduction
followed by a pointwise consumer:

```text
block amax reduction -> fp8 quant pointwise
```

The consumer's use of `c1 // 32` comes from broadcasting a per-block scale
back to element granularity. Reindexing the consumer from `(262144, 1)` to
`(8192, 32)` makes the dependency compare in the same domain as the producer.

Nested reduction is needed for the larger RMSNorm + MXFP8 case:

```text
RMS reduction -> normalized pointwise -> block amax reduction -> quant pointwise
```

With `triton.nested_reduction=False`, the RMSNorm + MXFP8 reproducer still
uses two kernels. With nested reduction enabled, the current tree produces one
kernel for the RMSNorm cases too.

## Validation performed

Focused test with the current tree:

```text
python /home/eellison/local/pytorch/work_dir/test_hi20.py --device cuda
```

Observed:

```text
RMSNorm + MXFP8:       2 kernels with nested_reduction=False
standalone MXFP8:     1 kernel for M=1,2,4,8,32,64
```

Focused probe:

```text
mxfp8_quant_only, M=32
```

With the vertical retry temporarily disabled:

```text
kernels 2
```

With the vertical retry restored:

```text
kernels 1
```

So the retry is doing real work for this pattern.

## Remaining considerations

The fix should stay narrow. The useful constraints are:

- Only run it for vertical reduction-to-pointwise fusion.
- Keep it behind `config.loop_reindexing_after_fusion`.
- Do not use it for nested append paths with a staged producer node unless that
  path is explicitly modeled.
- Rerun normal vertical legality and backend legality after the mutation.

One possible cleanup is to recompute `shared_data_score` after the reindex
before the retry's `V.choices.can_fuse_vertical()` call. The current flow
already has a positive score before reaching the vertical branch, so this is
not required for the observed bug, but recomputing would make the local state
match the final reindexed dependency state more directly.

## Conclusion

This is a good fix. The old reindexing path was in the profitability/scoring
part of fusion and could return early with a positive score without making the
consumer legally match the reduction domain. The new retry is at the vertical
legality boundary, where the actual failure occurs, and it preserves the normal
legality checks after reindexing.

## Broader understanding

The scheduler is currently mixing three related but different concepts:

1. Shared-data scoring: do two nodes share enough memory to be worth fusing?
2. Loop mutation for scoring: can reordering/reindexing expose more shared
   memory score?
3. Vertical legality: can a producer write satisfy every consumer read that
   depends on it?

The bug appears because the old reindex path sits under the second concept,
while the failure is in the third concept.

In the MXFP8 example, score repair succeeds for an incidental shared input
read:

```text
arg0_1 producer read and arg0_1 consumer read both normalize to one dense
262144-element access.
```

That is a real cache/locality signal, so returning a positive shared-data score
is not wrong. But it says nothing about whether the producer output `buf0`
matches the consumer read. The blocking vertical dependency is:

```text
buf0 write: 256*d0 + d1
buf0 read:  256*d0 + d1 // 32
```

That dependency requires reindexing the pointwise to the reduction domain.
Since score repair already returned successfully, the old code never tried that
reindex.

The PR-style fix effectively says: scoring is allowed to be approximate, but
vertical legality is not. If the exact vertical check fails for a
reduction-to-pointwise pair, try the pointwise-to-reduction reindex and rerun
the exact checks.

## Alternative design: keep reindexing inside shared-data repair

One alternative is to make `shared_data_after_reordering_loop()` continue after
a positive score whenever the pair is vertical and producer-output deps still
do not pass vertical legality. The shape would be:

```text
score = try_loop_reorder()
if score >= 0:
    if not vertical producer/consumer:
        return score
    if can_fuse_vertical():
        return score
    # the score came from some other dep; try reduction reindex too

try_reindex_pointwise_for_reduction()
...
```

This would also fix the MXFP8 case. The downside is that
`shared_data_after_reordering_loop()` would no longer be just a score repair
helper. It would start making vertical dependency legality decisions and would
need to know which legality failures should trigger more loop mutation.

That is feasible, but it makes the helper harder to reason about:

- A positive score would no longer be enough to return.
- The helper would duplicate or pre-run `can_fuse_vertical()`.
- It would need careful handling for horizontal pairs, template pairs, nested
  staged producers, and rollback.
- It would couple profitability repair to producer-output dependency legality.

For the current bug, the vertical retry is simpler: wait until the vertical
check actually fails, then try the one mutation known to address that kind of
failure.

## Possible refinement: prioritize blocking write/read deps

There is still a reasonable future cleanup in the old score-time path.
`_try_reorder_loops_for_candidates()` currently prefers write/read deps over
shared reads, but it can still return based on a dep that is not the one that
blocks vertical legality. In the MXFP8 trace, it returns using the shared input
read `arg0_1`, while the producer-output `buf0` dep remains illegal.

For loop reindexing after fusion, a better heuristic would be:

1. If the pair is vertical, first identify producer writes read by the consumer.
2. Among those deps, prioritize ones that currently fail
   `fusable_read_and_write()`.
3. Try loop reindexing/reordering against those blocking write/read deps before
   accepting a score from unrelated shared reads.
4. Fall back to the broader shared-read scoring heuristic only when there is no
   blocking producer-output dep or the mutation cannot help.

That would make the score-time path more aligned with vertical legality and
could avoid needing a second retry in some cases. It is a broader heuristic
change, though. The vertical retry is still useful as a final guard because it
directly repairs the observed legality failure and preserves exact checks after
mutation.

I tried this locally by making `shared_data_after_reordering_loop()` fall
through to `_try_reindex_pointwise_for_reduction()` when the first loop-order
score succeeds but a producer write -> consumer read dep still fails
`fusable_read_and_write()`. With the later vertical retry temporarily disabled,
the focused MXFP8 case still fused:

```text
mxfp8_quant_only, M=32
vertical retry disabled
kernels 1
```

So this heuristic is sufficient for the observed case. It works because it
prevents the shared input read from masking the blocking scale output dep during
score repair.

## Practical recommendation

Keep the PR-style vertical retry. It is narrow, config-gated, and validates
itself by rerunning normal legality. If we want to improve the older shared-data
repair path, do it as a follow-up: make loop reindexing prefer blocking
producer-output write/read deps before using incidental shared reads for score.
