# Alternative: Range tree rebinding for nested reduction epilogue

## Current approach

The epilogue handler (`_EpilogueOpsHandler`) manually generates `tl.load` / `tl.store`
using `pass2_x`, `pass2_r`, `pass2_idx`, `pass2_mask` variables that are emitted to
`kernel.compute` during `store_reduction`. This bypasses the kernel's standard
`indexing()` / `load()` / `store()` infrastructure.

Works, is ~30 lines, and is self-contained. Downside: doesn't benefit from the
kernel's block pointer, TMA, or other load/store optimizations.

## Alternative: rebind range trees with new symbols

After pass 1 `codegen_body()`, construct new range trees for the post-reduction
iteration space and emit a second set of iteration variable definitions with
distinct symbol names (e.g., `x2index`, `x2offset`, `x2mask`).

### Sketch

```python
# After pass 1 codegen_body() and pass 2 body call...

# 1. Save original range trees
orig_range_trees = kernel.range_trees
orig_numels = kernel.numels

# 2. Construct new range trees for post-reduction space
#    Pattern 1: x_tree was B*TOPK, now B. r_tree unchanged.
#    Pattern 2: r_tree was DIM, now DIM/K. x_tree unchanged.
post_reduction_numels = {"x": groups, "r0_": other_tree.numel}
new_range_trees = kernel.construct_range_trees(
    numels=post_reduction_numels,
    prefix_suffix="2",  # <-- would need new param to get x2index, r2_0index
)

# 3. Swap in and emit headers
kernel.range_trees = new_range_trees
kernel.numels = post_reduction_numels
for tree in new_range_trees:
    if not tree.is_loop:
        kernel.iteration_ranges_codegen_header(tree, kernel.compute)

# 4. Codegen epilogue using standard infrastructure
#    kernel.load() / kernel.store() / kernel.indexing() now use x2index, x2mask, etc.
for ep_sn in sn2_epilogue:
    # standard codegen path, no custom handler needed
    ...

# 5. Restore
kernel.range_trees = orig_range_trees
kernel.numels = orig_numels
```

### What needs to change

- `IterationRangesRoot` / `construct_range_trees`: accept a symbol prefix suffix
  so symbols are `x2_0`, `x2_1`, ... instead of `x0`, `x1`, ...
  (or a full prefix override like `"x2"` instead of `"x"`).
- `codegen_range_tree` / `iteration_ranges_codegen_header`: support emitting to
  an arbitrary buffer (not just `self.body`), already partially supported.
- `indexing()`: uses `self.range_trees` to find matching trees for mask generation,
  so swapping `range_trees` should make it work with new symbols automatically.
- CSE cache: the pass 2 reduction output is in `kernel.cse.store_cache` keyed by
  buffer name. Epilogue `load()` would find it there via standard codegen if the
  buffer hasn't been removed yet. Need to defer `removed_buffers.add` (same as now).

### Benefits

- Epilogue uses standard `kernel.load()` which can emit block pointers / TMA.
- Shape propagation works automatically (no manual `output_shape` param).
- Standard masking via `indexing()` — no manual `pass2_mask`.
- Naturally handles broadcasting (standard kernel infrastructure).

### Drawbacks

- More invasive: touches range tree construction, symbol naming.
- Pass 2 body still needs `_Pass2OpsHandler` (reshape+reduce trick is custom).
- Only saves ~30 lines in the epilogue handler.
- Risk of interactions with other range tree consumers (cooperative reduction, etc).

### When to revisit

If epilogues become more complex (e.g., multi-output, indirect indexing, or if
we want block pointer / TMA optimizations on the epilogue stores).
