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

## Future Work: Final Fusion (Quantization + Permutation)

To fully match the non-swizzled case (0.040 ms with 1 kernel), the quantization kernel would need to write scales directly in the swizzled format.

### Why Final Fusion is Challenging

The remaining gap is between:
- **Current**: 2 kernels (~0.081 ms)
- **Goal**: 1 kernel (~0.040 ms)

The blocker is the `slice_scatter` operation in `to_blocked`:

```python
# In to_blocked:
padded = torch.zeros((padded_rows, padded_cols), ...)
padded[:rows, :cols] = input_matrix  # <-- Creates slice_scatter extern kernel
# ... permutation operations
```

The `slice_scatter` is an extern kernel that breaks the fusion chain between:
1. Quantization kernel (produces scales)
2. Permutation kernel (rearranges scales to swizzled format)

### Approaches Explored

1. **Inverse Swizzle Iteration**: Iterate over output positions and compute which input to process.
   - Works but is 4x slower (131072 vs 32768 iterations)
   - 75% of iterations compute redundant results for padding positions

2. **Replace slice_scatter with torch.cat**: Use concatenation instead of assignment.
   - May help fusion but needs further investigation

3. **index_put/scatter**: Use explicit indexing operations.
   - These are also extern kernels, don't help fusion

### Recommended Solutions

1. **Short-term: Custom Triton Kernel**

   Write a custom kernel that fuses quantization with swizzled output:
   ```python
   @triton.jit
   def fused_quant_swizzle_kernel(in_ptr, out_scale_ptr, out_data_ptr, ...):
       # x0 = program_id for each scale block
       # Read 32 input values, compute scale
       # Compute swizzled output index
       swizzled_idx = (x0 // 128) * 512 + (x0 % 32) * 16 + ((x0 % 128) // 32) * 4
       # Write scale at swizzled position
       tl.store(out_scale_ptr + swizzled_idx, scale)
   ```

2. **Long-term: Extend Inductor**

   Teach inductor to recognize "compute + pad + permute" patterns and generate fused kernels with transformed output indices. This requires:
   - Pattern matching for slice_scatter followed by permutation
   - Output index transformation during codegen
   - Handling of padding (write zeros to padding positions)

### Performance Summary

| Configuration | Kernels | Latency | vs Non-swizzled |
|--------------|---------|---------|-----------------|
| Original (3 kernels) | 3 | ~0.100 ms | 2.5x slower |
| After this PR (2 kernels) | 2 | ~0.081 ms | 2.0x slower |
| Goal (1 kernel) | 1 | ~0.040 ms | 1.0x |
| Non-swizzled baseline | 1 | ~0.040 ms | baseline |
