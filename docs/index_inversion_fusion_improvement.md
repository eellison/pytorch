# Index Inversion Fusion Improvement for Multi-Dimensional Permutations

## Problem

When using `torch.compile` with MXFP8 quantization and swizzled scales (via `to_blocked`), the compiler was generating 3 separate kernels instead of fusing them:

1. **Quantization kernel**: Computes fp8 data and e8m0 scales
2. **First permutation kernel**: Applies first half of `to_blocked` permutation
3. **Second permutation kernel**: Applies second half of `to_blocked` permutation

This resulted in poor performance compared to the non-swizzled case:
- Non-swizzled: 0.040 ms (1 kernel)
- Swizzled: ~0.100 ms (3 kernels)

## Root Cause

The existing `shared_data_after_inverting_indexing` function in the scheduler could not handle the multi-dimensional indexing patterns used by the permutation kernels. Specifically:

1. **Multi-dimensional iteration**: The permutation kernels iterate with 3D variables `(p0, p1, p2)` but the function expected single-variable indexing

2. **Complex nested expressions**: The permutation involves nested `ModularIndexing` and `FloorDiv` patterns that weren't simplified

3. **Score calculation**: After successful inversion, `score_fusion_memory()` returned 0 because the modified MemoryDep objects didn't match, causing the "no shared data" heuristic to block fusion

4. **Peak memory heuristic**: `can_fusion_increase_peak_memory()` recalculated the score (getting 0) instead of using the pre-computed score

## Solution

### 1. Multi-dimensional to Linear Conversion

When the iteration space has multiple variables `(p0, p1, p2)`, convert them to a virtual linear index `_linear_idx`:

```python
# For iteration over (p0, p1, p2) with sizes (S0, S1, S2):
# p0 = _linear_idx // (S1 * S2)
# p1 = (_linear_idx // S2) % S1
# p2 = _linear_idx % S2
```

Substitute these into the read expression, then invert.

### 2. Nested ModularIndexing Simplification

Added simplification rules for nested patterns:

```python
# ModularIndexing(ModularIndexing(x, d1, m1), 1, m2) where m2 divides m1
# simplifies to ModularIndexing(x, d1, m2)
```

### 3. Proper Score Calculation

After successful inversion, return the buffer size directly instead of recalculating:

```python
score = self.dep_size_hint(node1_write)  # Size of shared buffer
```

### 4. Peak Memory Heuristic Fix

Modified `can_fusion_increase_peak_memory` to accept an optional pre-computed score:

```python
def can_fusion_increase_peak_memory(
    self, node1, node2, shared_data_score: int | None = None
) -> bool:
    if shared_data_score is not None:
        bw_saving = shared_data_score
    else:
        bw_saving = self.score_fusion_memory(node1, node2)
```

## Results

| Configuration | Before | After | Improvement |
|--------------|--------|-------|-------------|
| Kernel count | 3 | 2 | -1 kernel |
| Latency | ~0.100 ms | ~0.081 ms | ~19% faster |

All correctness tests pass.

## Files Modified

- `torch/_inductor/scheduler.py`: Extended `shared_data_after_inverting_indexing()` and `can_fusion_increase_peak_memory()`
- `torch/_inductor/choices.py`: Pass `shared_data_score` to peak memory check

## Future Work

To fully match the non-swizzled case (0.040 ms with 1 kernel), the quantization kernel would need to write scales directly in the swizzled format. This requires teaching inductor to fuse "compute + complex permute" patterns where the permutation is on the output side of the producer kernel.
