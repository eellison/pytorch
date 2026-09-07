# Nested Reduction Generic-Fusion Irregularities

This note records the places where the nested-reduction PR originally diverged
from generic scheduler fusion, plus the concrete fixes that moved the PR back
toward generic scoring/reindexing paths.

## Why This Is Not Only `can_fuse_vertical()`

The scheduler does not call vertical legality as the first or only fusion
decision. The current `Scheduler.can_fuse(node1, node2)` flow is:

1. Reject obvious global cases: same node, stream mismatch, incompatible fused
   node direction, grouped-node restrictions, extern/template-specific guards,
   explicit no-fuse buffers, and device mismatch.
2. Compute `shared_data_score = score_fusion_memory(node1, node2)`.
3. Optionally run generic loop-ordering / loop-reindexing / expand-dim repairs
   to improve the score.
4. Call `V.choices.can_fuse(...)`.
   This applies heuristic gates shared by horizontal and vertical fusion. The
   important one here is the score-0 "no shared data" rejection for reductions.
5. Only after that, if `node1.get_operation_names() & node2.ancestors`, enter
   the vertical branch:

   ```python
   can_fuse_vertical = nested_reduction_can_fuse or self.can_fuse_vertical(
       node1, node2
   )
   ```

6. Finally call `V.choices.can_fuse_vertical(...)` and backend
   `can_fuse_vertical(...)`.

So `can_fuse_vertical()` is not the entry point. It is a later legality check
after score-based shared-data heuristics have already accepted the pair.

This matters for nested reductions because the exact `MemoryDep` intersection
can be zero even when the fusion is semantically valid:

- the outer reduction and grouped reduction traverse the same logical elements,
  but with different `(numel, rnumel)` factorizations;
- the grouped output may be consumed at reduced resolution and then broadcast
  back to the parent tile;
- pointwise nodes around the grouped reduction may need reindexing before their
  deps look like ordinary vertical write/read pairs.

If we relied only on `can_fuse_vertical()`, these candidates could be rejected
earlier by `V.choices.can_fuse()` as "no shared data" and never reach vertical
legality.

Current nested policy:

- Legality does not use `MemoryDep` axis inference. Grouped-axis legality must
  come from explicit scheduler shape structure or from `LoopBody` iter/reduce
  variables. Ambiguous cases return `UNKNOWN` and do not fuse.
- `MemoryDep` is still used by the nested scoring bridge, but only to give the
  candidate enough shared-data score to pass the generic heuristic gate. A bad
  score can affect priority; it must not make an unsupported fusion legal.
- The actual legality check remains `NestedReduction.can_fuse()`, followed by
  the ordinary vertical/backend checks where applicable.

Possible generic cleanup:

- Split `Scheduler.can_fuse()` into an earlier structural/vertical dispatch
  stage and a later heuristic score stage, so vertical legality can classify a
  candidate before score-0 rejection.
- TODO: make dependent producer-consumer candidates flow more cleanly through
  `can_fuse_vertical()` before shared-data score heuristics can reject them.
  Nested reduction should not need to explain itself to the horizontal/common
  scoring path just to reach vertical legality.
- Or make `score_fusion_memory()` ask a generic vertical dependency-equivalence
  helper that understands reindexed producer/consumer deps without nested-only
  scoring.
- TODO: replace the nested-only X-axis scoring bridge with generic vertical dep
  equivalence across different factorizations such as `[B*K, D]` and
  `[B, D, K]`.
- TODO: split the transactional reindexing/rollback behavior into a separate
  generic scheduler PR. `_try_reindex_pointwise_for_reduction()` mutates loop
  state today; the rollback guarantee should be established independently from
  nested reduction, then nested can rely on it.
- TODO: clean up `can_reorder` plumbing. Threading `can_reorder` into
  `FusedNestedReductions.can_fuse_with()` is non-standard; it exists because
  generic loop reindex repair is controlled by the outer `Scheduler.can_fuse`
  call rather than by the vertical-fusion path itself. If reindex repair becomes
  a transactional vertical-legality step, fused nested follow-up fusion should
  not need custom `can_reorder` threading.

Either cleanup would be broader than nested reduction. Until then, the nested
score bridge is a narrow scheduler irregularity, while nested legality stays
explicit and conservative.

## Logged Vertical-Dep Mismatches

Command:

```bash
TORCHINDUCTOR_COMPILE_THREADS=1 \
  python agent_space/classify_nested_vertical_deps.py
```

Result:

```text
Ran 158 tests in 62.173s
OK
records=100

summary:
  count=82 key=(('MemoryDep',), (True,), False, True)
  count=8 key=(('MemoryDep',), (True,), True, True)
  count=6 key=((), (), True, False)
  count=4 key=(('MemoryDep', 'MemoryDep'), (True, True), False, True)
```

The logged records cover nested candidates where ordinary vertical dep matching
failed after the normal producer-write matching pass.

Observed producer-output mismatch:

- All unmatched producer-output deps are `MemoryDep`.
- All unmatched producer-output deps match a producer write after dropping read
  range variables that do not appear in the read index, then normalizing.
- No `WeakDep`, `StarDep`, indirect dep, mode mismatch, or unrelated
  transformation was required.

Representative R-axis case:

```text
producer write:
  MemoryDep('buf0', d0, {d0: 16})

consumer read:
  MemoryDep('buf0', d1, {d0: 1024, d1: 16})
```

`d0` is a broadcast-only range variable in the consumer read. Dropping it leaves
`MemoryDep('buf0', d1, {d1: 16})`, which normalizes to the producer write.

Representative X-axis case:

```text
producer write:
  MemoryDep('buf0', d0, {d0: 512})

consumer read:
  MemoryDep('buf0', 16*d0 + d2, {d0: 32, d1: 4096, d2: 16})
```

`d1` is a broadcast-only range variable in the consumer read. Dropping it leaves
the decomposed `[32, 16]` footprint, which normalizes to the flat `[512]`
producer write.

This supports keeping the nested vertical-dependency relaxation narrow:
`can_fuse_vertical()` should only opt into `normalize_without_broadcast()` for
nested reduction candidates, and only when the read can be proven to match one
of the producer's writes by this drop-unused-ranges + normalize rule.

## Full-Resolution Epilogue vs Generic Vertical Fusion

Experiment:

```python
return self.scheduler.can_fuse(
    self.node2,
    other,
    can_reorder=True,
)
```

in `FusedNestedReductions.can_fuse_with()`.

Test:

```bash
TORCHINDUCTOR_COMPILE_THREADS=1 \
  python test/inductor/test_nested_reduction.py \
  NestedReductionTest.test_producer_consumer_rmsnorm_fp8_quant
```

Result: the full-resolution FP8 epilogue split into a second pointwise kernel
(`generated_kernel_count == 2`). With temporary logging inside
`Scheduler.can_fuse_vertical()`, the generic path rejected with:

```text
NESTED_MISMATCH node1 writes:
  [MemoryDep('buf1', 32*d0 + d1, {d0: 128, d1: 32}),
   MemoryDep('buf2', 32*d0 + d1, {d0: 128, d1: 32})]

NESTED_MISMATCH node2 unmet:
  [MemoryDep('buf0', d0, {d0: 128}),
   MemoryDep('buf2', 32*d0 + ((d1//128)), {d0: 128, d1: 4096})]

NESTED_MISMATCH remaining: OrderedSet(['buf0', 'buf2'])
```

Interpretation:

- `buf2` is the real reduced-output/full-resolution mismatch. The grouped
  stage writes it at reduced resolution `[B, D / G]`:

  ```text
  32*d0 + d1, ranges {d0: 128, d1: 32}
  ```

  The full-resolution epilogue reads the same semantic value broadcast over
  the parent tile `[B, D]`:

  ```text
  32*d0 + (d1 // 128), ranges {d0: 128, d1: 4096}
  ```

  Generic `MemoryDep` matching cannot prove those are the same access, because
  it does not know that `d1 // 128` is selecting the grouped value for all
  lanes in that group. Nested codegen does know this, via
  `_GroupReductionLayout.resolve_full_resolution_load()`, and materializes the
  reduced-output register as a full parent-tile value.

- `buf0` remains because the attempted generic call used only `self.node2` as
  the producer. The full-resolution epilogue also reads a value produced by the
  outer `node1` part of the already-fused nested kernel. The enclosing
  `FusedNestedReductions` node knows `buf0` is already internal to the fused
  node; `scheduler.can_fuse(self.node2, other)` does not.

So this does not look like a wrong `rnumel` being passed around. It is a real
semantic gap: generic vertical fusion sees a reduced write and a broadcasted
full-resolution read as different memory dependencies.

Follow-up experiment: yes, generic reindexing can expose the fact that this is
the same dependency.

If loop-ordering is disabled, or if
`Scheduler._try_reindex_pointwise_for_reduction(self.node2, other)` is called
directly before generic `scheduler.can_fuse(self.node2, other)`, the full-res
epilogue is reindexed from `[B, D]` to the grouped reduction's flattened
domain. The relevant deps become:

```text
AFTER_DIRECT node1 writes:
  [MemoryDep('buf1', 32*d0 + d1, {d0: 128, d1: 32}),
   MemoryDep('buf2', 32*d0 + d1, {d0: 128, d1: 32})]

AFTER_DIRECT node2 unmet:
  [MemoryDep('buf0', (d0//32), {d0: 4096}),
   MemoryDep('buf2', d0, {d0: 4096})]
```

With `TORCHINDUCTOR_LOOP_ORDERING_AFTER_FUSION=0`, they canonicalize even more
directly:

```text
AFTER_REINDEX node1 writes:
  [MemoryDep('buf1', c0, {c0: 4096}),
   MemoryDep('buf2', c0, {c0: 4096})]

AFTER_REINDEX node2 unmet:
  [MemoryDep('buf0', (c0//32), {c0: 4096}),
   MemoryDep('buf2', c0, {c0: 4096})]
```

In that configuration the FP8 full-resolution epilogue fuses through the
generic path.

The reason the default generic path does not find this today is ordering of
generic repairs:

1. `scheduler.can_fuse(..., can_reorder=True)` only calls
   `shared_data_after_reordering_loop()` when the memory score is below
   `score_fusion_memory_threshold`.
2. This case already has nonzero shared data, so generic repair may not run.
3. When we force `shared_data_after_reordering_loop()` under default configs,
   loop-ordering returns early using the shared `buf0` read and never reaches
   `_try_reindex_pointwise_for_reduction()`.
4. The unresolved `buf2` write/read mismatch then reaches
   `can_fuse_vertical()` unchanged and is rejected.

So the deeper issue is not that generic reindexing is incapable. It is that
the generic scheduler currently chooses "enough shared data / loop-ordering
improved the score" before asking whether vertical dependency matching still
needs reindexing.

Fix:

Teach generic vertical fusion to try pointwise/reduction reindexing when
`can_fuse_vertical()` fails, then verify vertical legality again before keeping
the mutation:

```python
can_fuse_vertical = self.can_fuse_vertical(node1, node2)
if (
    not can_fuse_vertical
    and can_reorder
    and config.loop_reindexing_after_fusion
    and self._try_reindex_pointwise_for_reduction(
        node1,
        node2,
        require_vertical_fusion=True,
    )
):
    shared_data_score = self.score_fusion_memory(node1, node2)
    can_fuse_vertical = True
```

The `require_vertical_fusion` flag is important.
`_try_reindex_pointwise_for_reduction()` mutates scheduler-node loop state. If
reindexing increases shared-data score but still does not prove the vertical
write/read dependency legal, the helper rolls back instead of leaving a failed
fusion candidate mutated.

With this in place, `FusedNestedReductions.can_fuse_with()` no longer needs a
nested-specific reindex fallback. It performs the nested resolution/shape gate,
then delegates to `scheduler.can_fuse(self.node2, other, can_reorder=True)`.

Validation:

```bash
python -m py_compile torch/_inductor/scheduler.py

TORCHINDUCTOR_COMPILE_THREADS=1 python test/inductor/test_nested_reduction.py \
  NestedReductionTest.test_producer_consumer_rmsnorm_fp8_quant \
  NestedReductionTest.test_producer_consumer_rmsnorm_fp8_quant_B1 \
  NestedReductionTest.test_layernorm_block_amax_reduced_pointwise_epilogue \
  NestedReductionTest.test_fullres_prologue_small_dim_in_x_loop_order

TORCHINDUCTOR_COMPILE_THREADS=1 python test/inductor/test_loop_ordering.py \
  LoopOrderingTest.test_reshape_reindexing_transposed_input \
  LoopOrderingTest.test_reshape_reindexing_for_reduction \
  LoopOrderingTest.test_reshape_reindexing_without_loop_ordering \
  LoopOrderingTest.test_reindex_unfusable_write_read_dep \
  LoopOrderingTest.test_reindex_rollback_on_no_improvement \
  LoopOrderingTest.test_reshape_reindexing_fused_pointwise

TORCHINDUCTOR_COMPILE_THREADS=1 python test/inductor/test_nested_reduction.py
```

The focused nested tests and focused loop-ordering/reindexing tests passed.
The full nested suite passed.

## Why There Is a Nested Fused Node

`FusedNestedReductions` carries state that generic fusion does not have:

- `node1`: the outer reduction, which owns the parent physical tile.
- `node2`: the grouped reduction and reduced-resolution epilogues.
- `node2_reduction`: the single grouped reduction inside `node2`.
- `group_size`: the static local reduction width.
- `group_size_in_r`: whether the grouped axis splits the parent R tree or X
  tree.

Codegen uses this state to run:

1. the outer reduction over the parent tile,
2. the grouped reduction by remapping `node2` into that tile,
3. reduced-resolution epilogues over `[B, D / G]`,
4. optional full-resolution epilogues over `[B, D]` with grouped values
   broadcast back.

Generic fusion has no concept of these two resolutions inside one fused node.

## Why `NestedReduction.can_fuse()` Checks Ancestors

`NestedReduction.can_fuse()` is called in more places than a pure vertical
legality check:

- scoring (`Scheduler.score_fusion_memory()`),
- backend SIMD compatibility (`SIMDScheduling.can_fuse()`),
- fused-node construction (`Scheduler.fuse()`).

Those call sites can receive arbitrary scheduler pairs discovered by common
buffer grouping. Without checking that `node2` actually depends on `node1`,
same-input independent reductions can be misclassified as a sequential nested
reduction. But nested codegen assumes an ordered pipeline: `node1` first, then
`node2` remapped into `node1`'s tile.

Possible future cleanup:

- split the predicate into two layers:
  - a generic scheduler layer establishes "vertical producer-consumer pair",
  - a nested layer checks only shape/layout/reduction legality.

That would make the ancestry check feel less local to nested reduction, but it
is a larger scheduler API reshaping.

## Nested Scoring: Reusing the Shared Scoring Block

Review question:

Can `NestedReduction.get_fusion_score()` be deleted and replaced by the shared
scoring block already used for template/user-defined Triton epilogues?

Experiment:

- Delete the early nested scoring branch:

  ```python
  if NestedReduction.can_fuse(node1, node2):
      score = NestedReduction.get_fusion_score(node1, node2)
      return _construct_return_value(score, 0, False)
  ```

- Let nested pairs fall through to the shared scoring block.

Test:

```bash
TORCHINDUCTOR_COMPILE_THREADS=1 python test/inductor/test_nested_reduction.py \
  NestedReductionTest.test_layernorm_block_amax_reduced_pointwise_epilogue
```

Initial result:

```text
metrics.codegen_nested_reduction == 1
metrics.generated_kernel_count == 2  # expected 1
```

The nested reduction still happened, but it fused the wrong producer first.
LayerNorm has two sibling outer reductions before the grouped amax:

```text
op0: mean reduction
  reads  arg0_1, bytes=1048576
  writes buf0,  bytes=256

op2: variance reduction
  reads  arg0_1, bytes=1048576
  writes buf2,  bytes=256

op4: grouped block amax
  reads  arg0_1, bytes=1048576
  reads  buf0,  bytes=256
  reads  buf2,  bytes=256
  writes buf4,  bytes=65536

op5: reduced-output epilogue
  reads  buf4, scale, bias
```

With the current nested score, the first-round fusion candidates are:

```text
op0 -> op2   memory_score=1048576  # normal same-shape reduction fusion
op2 -> op4   memory_score=1048576  # nested
op0 -> op4   memory_score=1048576  # nested
op4 -> op5   memory_score=65536
```

The tie keeps the normal `op0 -> op2` layernorm fusion first. Then the fused
outer layernorm reduction can fuse with `op4`, and `op5` is absorbed into the
nested node:

```text
POST:
  FusedNestedReductions(op0_op2_op4_op5)
    node1 = op0_op2
    node2 = op4_op5
generated_kernel_count = 1
```

With unconditional name-based scoring, nested pairs also count the small
producer output read by `op4`:

```text
op2 -> op4:
  name-match arg0_1  score=1048576
  name-match buf2    score=256
  total              score=1048832

op0 -> op4:
  name-match arg0_1  score=1048576
  name-match buf0    score=256
  total              score=1048832

op0 -> op2:
  name-match arg0_1  score=1048576
```

That 256-byte bump is enough to make `op2 -> op4` win before `op0 -> op2`.
The final fused graph becomes:

```text
POST:
  SchedulerNode(op0)
  FusedNestedReductions(op2_op4_op5)
    node1 = op2
    node2 = op4_op5
generated_kernel_count = 2
```

Interpretation of the failure:

- Name-based scoring is correct for its original template/UDT use:
  if two nodes touch the same named buffer, count it.
- For nested reductions, counting both the large shared input and the small
  producer output over-prioritizes an incomplete nested pair.
- The current custom nested score intentionally avoids this:
  - when there are shared input reads, score only the shared input traffic;
  - only in pure producer-consumer cases, score node1's output read by node2.

So the old custom score was not only a nonzero-score workaround. It encoded an
ordering policy: finish sibling outer reductions first, then form the nested
pipeline. Without that policy, layernorm can split into "mean kernel" +
"variance/grouped-amax/epilogue nested kernel".

Possible fixes:

1. Keep the custom score. This is the smallest PR-local solution and matches
   the desired ordering.
2. Add a generic scheduler priority rule: if a consumer reduction depends on
   multiple sibling outer reductions, fuse those same-shape outer reductions
   before allowing a nested producer-consumer fusion. This would be broader
   and more semantic, but it is a scheduler policy change.
3. Add a nested legality rule requiring `node1` to include all same-shape
   reduction ancestors read by `node2` before nested fusion. This prevents the
   incomplete `op2 -> op4` nested fusion, but needs care: `node2` may also
   read legitimate non-nested side inputs that should not block fusion.

Focused experiment for (3):

Using a runtime monkey patch, nested fusion was rejected when `node2` read a
same-shape reduction ancestor not already included in `node1`. With name-based
score still enabled, this restored the desired order for the LayerNorm case:

```text
POST:
  FusedNestedReductions(op0_op2_op4_op5)
    node1 = op0_op2
    node2 = op4_op5
generated_kernel_count = 1
```

That suggests a real generic direction exists: nested fusion should probably
prefer an "outer reduction group is complete" invariant instead of relying only
on score tie-breaking. However, encoding that invariant is more semantic than
the current scoring helper:

- it needs to distinguish sibling outer reductions from unrelated side inputs,
- it needs to work after some ancestors are already inside a `FusedSchedulerNode`,
- it may interact with future patterns that intentionally read an already
  materialized reduction from outside the nested kernel.

So this is a plausible follow-up, not an obvious simplification for the first
landing.

Follow-up experiment:

Keep the shared scoring block, but rank nested fusion after otherwise
equivalent non-nested fusion in `FusionScore`. This preserves the ordering
policy without carrying nested-specific byte accounting:

```python
return FusionScore(
    template_score,
    not NestedReduction.can_fuse(node1, node2),
    type_score,
    memory_score,
    buffer_overlap_score,
    proximity_score,
)
```

With that change:

- `op0 -> op2` wins before either nested `op0/op2 -> op4` candidate, because
  it is a normal same-shape reduction fusion.
- The resulting fused outer reduction then fuses with `op4`.
- The reduced-output epilogue `op5` is absorbed after that.

Validation:

```bash
python -m py_compile torch/_inductor/choices.py \
  torch/_inductor/scheduler.py torch/_inductor/codegen/simd.py

TORCHINDUCTOR_COMPILE_THREADS=1 python test/inductor/test_nested_reduction.py \
  NestedReductionTest.test_layernorm_block_amax_reduced_pointwise_epilogue \
  NestedReductionTest.test_producer_consumer_rmsnorm_fp8_quant \
  NestedReductionTest.test_rmsnorm_weighted_sum_B_32_K_16 \
  NestedReductionInternalsPersistentTest.test_fullres_kernel_form

TORCHINDUCTOR_COMPILE_THREADS=1 python test/inductor/test_nested_reduction.py
```

The full nested suite passed (`152 tests`). A focused mix-order subset also
passed/skipped as expected:

```bash
TORCHINDUCTOR_COMPILE_THREADS=1 python test/inductor/test_mix_order_reduction.py \
  -k test_xmask -k test_XBLOCK_coordest_tuning -k test_fuse_non_contiguous_pointwise
```

Current recommendation for this PR:

Prefer the shared scorer plus a small nested-late priority in `FusionScore`.
For nested reductions, the scorer should prove shared traffic through normalized
`MemoryDep` equivalence rather than buffer-name overlap. This removes
`NestedReduction.get_fusion_score()` while keeping the important scheduler
behavior: ordinary reductions get a chance to complete before a nested
producer-consumer fusion consumes one member of the group.

## Nested Scoring: Why Exact Generic Scoring Still Fails

Follow-up experiment:

- Keep the current generic scheduler flow.
- Remove nested reduction from the name-based scoring block in
  `Scheduler.score_fusion_memory()`.
- Let nested candidates fall through to the ordinary exact `MemoryDep`
  intersection path.

Result:

```bash
TORCHINDUCTOR_COMPILE_THREADS=1 python test/inductor/test_nested_reduction.py
```

failed 52 nested tests. The common symptoms were:

- `metrics.codegen_nested_reduction == 0`, meaning the nested pair never
  fused.
- kernel-form tests found two kernels instead of one, e.g. outer RMSNorm in
  one reduction kernel and the weighted/grouped reduction in another.

Representative case:

```python
B, K, D = 32, 16, 4096

x_flat = x.reshape(B * K, D)
rms = torch.sqrt(torch.mean(x_flat * x_flat, dim=-1, keepdim=True) + 1e-6)
x_normed = (x_flat / rms).reshape(B, K, D)
out = (w[:, :, None] * x_normed).sum(dim=1)
```

With temporary logging inside `score_fusion_memory()` for the nested candidate
`op0 -> op1`:

```text
NESTED_PAIR ('op0', 'op1')
  exact_score 0
  exact_deps []
  normalized_score 8390656
  normalized_matches
    node1: MemoryDep('arg0_1', 4096*d0 + d1, {d0: 512, d1: 4096})
    node2: MemoryDep('arg0_1', 65536*d0 + d1 + 4096*d2,
                     {d0: 32, d1: 4096, d2: 16})

    node1: MemoryDep('buf0', d0, {d0: 512})
    node2: MemoryDep('buf0', 16*d0 + d2, {d0: 32, d1: 4096, d2: 16})
```

The normalized proof is not just buffer-name overlap:

```text
arg0_1:
  a.normalize_with_stride_order()
    MemoryDep('arg0_1', t0, {t0: 2097152})
  b.normalize_with_stride_order()
    MemoryDep('arg0_1', t0, {t0: 2097152})

buf0:
  a.normalize_without_broadcast()
    MemoryDep('buf0', c0, {c0: 512})
  b.normalize_without_broadcast()
    MemoryDep('buf0', c0, {c0: 512})
```

This is comparison-only. `normalize_with_stride_order()` returns a new
canonical `MemoryDep`; it does not mutate scheduler loop order. Real loop
mutation still happens only through the existing loop ordering/reindexing
helpers.

Interpretation:

- The two nodes do share real data:
  - both read the original input `arg0_1`,
  - node2 reads node1's output `buf0`.
- Exact `MemoryDep` intersection sees none of that, because the same buffers
  are expressed in different iteration spaces:
  - node1 has flattened `[B * K, D]`,
  - node2 has logical `[B, D, K]`,
  - node2's read of `buf0` broadcasts the per-row RMS value across the `D`
    dimension.

That zero score matters because scheduler fusion ordering is:

1. `Scheduler.can_fuse()` computes `shared_data_score`.
2. `V.choices.can_fuse()` rejects reduction-involved fusions when
   `shared_data_score == 0`.
3. Only after that does `Scheduler.can_fuse_vertical()` check producer-consumer
   legality.

So this failure happens before vertical dependency matching can say anything
about legality. The nested branch in `score_fusion_memory()` is therefore not a
correctness bypass; it is a scoring bridge that prevents an otherwise legal
vertical candidate from being rejected as "no shared data".

Potential generic fix:

The generic scheduler could support a vertical-only normalized-dependency score:

```python
is_vertical = bool(node1.get_operation_names() & node2.ancestors)
if score == 0 and is_vertical:
    score = score_normalized_memory_deps(node1, node2)
```

and still rely on `can_fuse_vertical()` for correctness. This would be broader
than nested reduction:

- it would not bless any fusion by itself,
- it would only keep the candidate alive long enough for vertical legality,
- it would naturally cover reshape/broadcast/reindex producer-consumer cases
  where exact `MemoryDep` equality is too strict.

Why not do that in this PR?

- `score_fusion_memory()` is used for ordering and peak-memory heuristics, not
  only for the boolean gate. Broadening normalized scoring for all vertical
  pairs could reorder unrelated fusions.
- Existing buffer-overlap scoring explicitly skips reductions today:

  ```python
  if node1.is_reduction() or node2.is_reduction():
      return False
  ```

  Lifting that restriction safely needs review outside nested reduction.
- The current PR already narrows the nested score path to vertical
  producer-consumer candidates only, so the irregularity is contained.

Good follow-up:

Add a generic vertical scoring helper that is used after exact scoring returns
zero but before `V.choices.can_fuse()` rejects. It should prove actual
`MemoryDep` equivalence using existing normalizations such as
`normalize_with_stride_order()` and `normalize_without_broadcast()`, not merely
match buffer names. Start with producer-consumer pairs only, keep horizontal
reduction pairs excluded, and validate on nested, template epilogues,
loop-reindexing, and existing buffer-overlap tests. If that works,
`NestedReduction.can_fuse()` can be removed from `score_fusion_memory()`
entirely.

## Why `FusedNestedReductions.can_fuse_with()` Still Has a Small Gate

After the initial nested pair is fused, later pointwise consumers of `node2`
can still be fused in. Some consumers run at reduced resolution; FP8 quant runs
at full resolution and reads grouped values via broadcast.

The generic scheduler now handles the dependency repair: if vertical
write/read matching fails, it can reindex the pointwise consumer into the
reduction's loop space and retry vertical legality. The nested fused node still
needs a small pre-gate for stage resolution:

- `other` must be downstream pointwise, not another reduction.
- It must consume `node2`, not an arbitrary earlier buffer.
- Its iteration space must be compatible with either reduced-output resolution
  or full resolution. Small-dim-in-X full-resolution epilogues remain guarded
  out because size-only compatibility can map `[B, D, K]` to `[B, K, D]`
  incorrectly.

After those nested-resolution checks, `can_fuse_with()` delegates to the
generic scheduler path.

## Possible Larger Refactor

A cleaner long-term shape would be to make consumer resolution more explicit:

```python
NestedConsumerResolution.REDUCED
NestedConsumerResolution.FULL
NestedConsumerResolution.HALF
```

Then `FusedNestedReductions.can_fuse_with()` could express the pre-gate as
resolution classification instead of a size/compatibility check. That is a good
fit for the later NVFP4 half-resolution consumer work.

## Profitability Does Not Reduce To ReductionHint

Status update: this section describes the earlier state when the PR still tried
to support `group_size_in_r == False` / XBLOCK grouped reductions. The current
simplification proposal is to defer that path, so `_is_profitable_group_size_in_x()`
can be removed from this PR instead of refined here.

Review question:

Could `_is_profitable_group_size_in_x()` be simplified to:

- parent reduction is inner/coalesced in R -> only allow group-size-in-R
- parent reduction is outer/coalesced in X -> allow group-size-in-X

Current answer: not safely in this PR.

For the supported weighted RMSNorm pattern:

```python
x_normed = rmsnorm(x.reshape(B * K, D)).reshape(B, K, D)
out = (w[:, :, None] * x_normed).sum(dim=1)
```

the grouped reduction is the small `K` reduction. It is a valid
`group_size_in_x` nested reduction and is covered by:

```bash
TORCHINDUCTOR_COMPILE_THREADS=1 python test/inductor/test_nested_reduction.py \
  NestedReductionTest.test_rmsnorm_weighted_sum_B_32_K_16
```

But the grouped reduction's IR reduction hint printed as:

```text
group_size_in_r = False
reduction_hint = ReductionHint.DEFAULT
ranges = ([32, 4096], [16])
```

So a rule that only distinguishes `ReductionHint.INNER` from
`ReductionHint.OUTER` would not classify this important supported case.
Treating `DEFAULT` as "inner" would reject weighted norm + reduce-K. Treating
`DEFAULT` as "outer" would be too permissive.

The current logic only invokes the coalescing/profitability check for the
ambiguous/sensitive case:

```python
group_divides_x and group_cannot_divide_r
```

That is the case where the group size is provably in X, so nested codegen must
force `min_xblock = group_size`. If the parent reduction's memory access is
still primarily R-coalesced, that is likely a bad fusion. The current check uses
`analyze_memory_coalescing()` directly to compare coalescing score of parent R
vars vs X vars.

Cleaner follow-up: expose a generic helper from tiling/coalescing analysis:

```python
is_coalesced_along(node, "x" | "r")
```

or return a simple enum:

```python
CoalescedAxis.X
CoalescedAxis.R
CoalescedAxis.UNKNOWN
```

Then nested reduction could express the reviewer's intended rule without
misclassifying `ReductionHint.DEFAULT` cases.

## Score-0 Fusion Gate

Status update: this section describes the earlier nested score bridge while the
PR still supported XBLOCK grouped reductions. With the current simplification,
the broadcast-stripped part of this bridge is dropped.

Review question:

Can nested reduction avoid a special case in `score_fusion_memory()` and instead
teach the generic fusion gate not to reject nested candidates solely because the
memory score is zero?

Current answer: yes. The nested-specific score-0 bypass in
`InductorChoices.can_fuse()` was removed.

The scheduler memory score is still exact-dependency-oriented. For nested
reduction, node1 and node2 may be semantically connected but have different
`MemoryDep` expressions because one side is reduced/reshaped:

```text
MemoryDep('buf2', 32*d0 + (d1//128), {d0: 128, d1: 4096})
```

That used to produce `shared_data_score == 0` before vertical legality had a
chance to recognize the nested producer-consumer relation. The current fix is a
nested-scoped normalized score:

- `score_fusion_memory()` first uses exact `MemoryDep` set intersection.
- If that score is zero and `NestedReduction.can_fuse(node1, node2)` is true,
  it scores same-buffer deps using existing normalizations:
  `normalize_with_stride_order()` and `normalize()`.
- `V.choices.can_fuse()` no longer has a nested-specific score-0 exception.
- `can_fuse_vertical()` and backend legality still own correctness.

Attempted broader fix:

Applying normalized scoring broadly to reduction->pointwise vertical pairs is
too broad. It can make
`LoopOrderingTest.test_reshape_reindexing_transposed_input` receive a nonzero
score early, which skips the loop-reindex repair that test needs. Separately,
the nested PR's first vertical-retry implementation could reindex a pointwise
twice: once through the existing loop-ordering path, then again when retrying
after vertical legality failed. The fix is to make
`_try_reindex_pointwise_for_reduction()` idempotent when the pointwise already
has the reduction's iteration sizes.

TODO:

If we want to broaden this later, split "score enough to avoid no-shared-data
rejection" from "score enough to skip loop reindexing." The current landable
scope is nested reduction only, which removes the `InductorChoices` bypass
without changing generic reduction or pointwise fusion ordering.

## Follow-up: XBLOCK Grouped Reductions and Broadcast Dep Equivalence

Update from the latest simplification pass:

We can avoid using broadcast-stripped memory-dep equivalence in nested-reduction
scoring if we stop claiming support for grouped reductions whose local group
splits the parent X tree. In other words, this PR can support the block-quant
shape we actually need:

```python
x_normed = layernorm_or_rmsnorm(x)              # [B, D]
groups = x_normed.reshape(B, D // G, G)
amax = groups.abs().amax(dim=-1)               # group is in parent R
```

and deliberately fall back for the less important XBLOCK-grouped pattern:

```python
x_normed = rmsnorm(x.reshape(B * K, D)).reshape(B, K, D)
out = (w[:, :, None] * x_normed).sum(dim=1)    # group is in parent X
```

Why this matters:

- The XBLOCK-grouped pattern needed nested scoring to treat broadcasted
  producer/consumer deps as equivalent so the pair was considered for fusion.
- That scoring equivalence is a generic scheduler concept. Landing it as part
  of nested reduction would make this PR responsible for a broader dependency
  change.
- The RBLOCK block-quant path does not need broadcast-aware nested scoring in
  the targeted validation:

  ```bash
  TORCHINDUCTOR_COMPILE_THREADS=1 python test/inductor/test_nested_reduction.py \
    NestedReductionTest.test_layernorm_block_amax_B_32_D_4096_G_16 \
    NestedReductionTest.test_layernorm_block_amax_B_4_D_384_G_128 \
    NestedReductionTest.test_producer_consumer_rmsnorm_fp8_quant \
    NestedReductionTest.test_fullres_epilogue_with_multiple_outputs \
    NestedReductionInternalsPersistentTest.test_fullres_kernel_form
  ```

Older simplified PR boundary considered while debugging X-axis support:

- Keep nested reduction legality restricted to `group_size_in_r == True`.
- Remove `MemoryDep.normalize_without_broadcast()` from this stack.
- Remove broadcast-aware scoring from
  `_score_fusion_memory_by_nested_reduction_deps()`.
- Remove the X-dim profitability/coalescing helper and X-axis codegen branch
  from `NestedReduction`.
- Make the generic pointwise/reduction reindex retry idempotent so removing the
  broadcast helper does not allow a second reindexing pass on old loop-ordering
  cases.
- Update X-dim tests to assert numerics and no nested fusion, rather than
  xfail. That verifies fallback behavior without preserving a special path.

We did not take this boundary. Instead, X-axis support remains in this stack,
and `normalize_without_broadcast()` is used only by nested reduction's explicit
vertical-dependency matching path.

Follow-up design if we want XBLOCK grouped reductions:

1. Add a generic scoring/dependency-equivalence helper for "same producer,
   broadcasted consumer" access patterns. This should be tested outside nested
   reduction.
2. Decide whether that helper belongs in `fusable_read_and_write()`,
   memory-scoring, or a separate vertical-dependency resolution step. The
   important constraint is that it must not skip loop-reindex repairs for normal
   pointwise/reduction fusion.
3. Re-enable `group_size_in_r == False` only after generic scoring and vertical
   legality agree on the same dep equivalence.
4. Restore XBLOCK kernel-form tests for weighted RMSNorm/reduce-K once the
   generic dep work is in place.
