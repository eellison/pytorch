# Vertical Broadcast Fusion Mechanism

Context: while splitting nested reduction, we considered making a first generic
prep commit for "broadcast dep equivalence":

- `MemoryDep.normalize_without_broadcast()`
- a fallback in `Scheduler.fusable_read_and_write()`
- vertical retry after loop reindexing

After probing real Inductor kernels, the broad generic broadcast-dep change is
not yet justified as a standalone prep commit. The obvious non-nested broadcast
cases already fuse through existing mechanisms.

## What Already Works

Example shape:

```python
def f(x):
    s = x.sum(dim=1)       # [B, D]
    return x + s[:, None]  # [B, K, D]
```

This is not a block-local nested reduction. It is an ordinary reduction with a
full-resolution broadcast epilogue.

Empirical result:

- Current code: one kernel.
- Old `fusable_read_and_write()` behavior, with the new
  `normalize_without_broadcast()` fallback monkey-patched out: still one kernel
  when compiled with a fresh Inductor cache.

The main FP8 nested shape also still fused with the old
`fusable_read_and_write()` behavior when compiled with a fresh cache:

```python
x = F.rms_norm(x, (D,), weight)
x_groups = x.view(B, D // G, G)
amax = x_groups.abs().amax(dim=-1)
scale = (amax / fp8_max).clamp(min=1e-12)
x_fp8 = (x_groups / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
```

That means the main nested FP8 path is not currently proving the generic
broadcast-dep fallback either.

## Existing Mechanisms Involved

### 1. `fusable_read_and_write()` already normalizes some broadcast cases

Before the proposed `normalize_without_broadcast()` fallback, vertical legality
already did this:

```python
if config.loop_ordering_after_fusion and read.num_vars != write.num_vars:
    read = read.normalize()
    write = write.normalize()
```

For many producer `[B, D]` / consumer `[B, K, D]` cases, the consumer read has an
extra loop variable that does not appear in the read index. Normalization can
merge/prune loop structure enough for the write/read match to succeed.

So "broadcasted read" does not automatically imply the new fallback is needed.

### 2. Fusion scoring can survive through other shared deps

The fusion candidate does not always rely on the broadcasted write/read dep for
its initial score. Common inputs can give a nonzero `score_fusion_memory()` and
let vertical legality run.

For example, `x.sum(dim=1)` and `x + s[:, None]` both read `x`, so the pair can
survive the initial "no shared data" filter even if the `s` write/read dep is
not an exact set-intersection match.

### 3. Loop reordering/reindexing repairs some non-identical layouts

`shared_data_after_reordering_loop()` already tries loop reordering and then
pointwise/reduction reindexing. This is why several reshape/reduction fusion
cases in `test_loop_ordering.py` work without nested reduction.

The retry added in the split commit is still plausibly useful, but it should be
justified by a kernel-level case where:

- initial vertical legality fails,
- reindexing repairs the pointwise/reduction coordinate relationship,
- the full vertical fusion decision then succeeds.

### 4. Nested score bridge is a separate issue

Nested reduction still needs a candidate-survival bridge because the initial
nested pair can have different `(numel, rnumel)` and exact `MemoryDep`
intersection may score zero before nested legality runs.

That is different from generic broadcast dep equivalence. The nested score
bridge is intentionally nested-scoped and should remain in the nested scheduler
commit unless/until generic scoring grows a separate "semantic vertical dep"
predicate.

## What The Proposed Generic Fallback Adds

The proposed fallback does this after exact/index-prefix matching fails:

```python
read = orig_read.normalize_without_broadcast()
write = orig_write.normalize_without_broadcast()
return read.index == write.index and read.size == write.size
```

This is stronger than the existing path because it can apply even when the
normal `read.num_vars != write.num_vars` normalization branch did not prove the
match.

Potentially useful cases:

- loop ordering disabled but broadcast-only variables need to be ignored,
- same number of loop variables but one side has a loop variable absent from
  the index,
- unusual fused-node loop structure where ordinary normalization does not fire.

Current problem:

We do not yet have a real compiled-kernel test where this fallback changes
fusion behavior. A raw `MemoryDep` unit test proves the helper, but not that
the scheduler needs it in practice.

## Recommendation For The Split

Do not lead the stack with a broad generic "vertical dep equivalence" commit
unless we find a real Inductor kernel-level failing case.

Better split options:

1. Drop the `normalize_without_broadcast()` / `fusable_read_and_write()` generic
   fallback from the first split for now.
2. Keep only the reindex retry/safety changes if they are needed by a real
   kernel-level test or by later nested tests.
3. Keep the nested score bridge nested-scoped in the nested scheduler commit.
4. Track generic broadcast dep equivalence as a follow-up, with the acceptance
   criterion: a non-nested compiled kernel that fuses only with the generic
   fallback.

Open question:

Can we construct a real non-nested kernel where ordinary normalization,
common-input scoring, and loop reindexing do not already cover the broadcasted
write/read dep? If yes, that is the right test for a generic prep PR. If no,
the generic fallback should probably not be part of the landing stack.
