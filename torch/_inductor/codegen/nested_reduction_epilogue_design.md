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

---

# Follow-up: tiling_utils integration (node2 read/write remapping)

## Problem

`extract_normalized_read_writes` and `analyze_memory_coalescing` in `tiling_utils.py`
are never called for `FusedNestedReductions` because they bypass `codegen_node`.
If we want tiling heuristics to account for node2's memory accesses, we need to
express node2's read/write indices in terms of node1's `(numel, rnumel)` iteration
space.

## Current state

- `extract_normalized_read_writes` assumes all subnodes share the same
  `(pw_numel, red_numel)` from `node.group[1]`, decomposed via `get_pw_red_splits`.
- Node1's subnodes decompose `(B*K, DIM)` or `(B, DIM)` — works as-is.
- Node2's subnodes decompose a *different* total (e.g., `(B, K*DIM)` for pattern 1,
  `(B*DIM/G, G)` for pattern 2) — `get_pw_red_splits` cannot handle them in node1's
  iteration space.

## Approach: shared remapping utility

Factor out the variable remapping logic from `codegen_nested_reduction` into a
reusable helper. The same `topk_tree`/`other_tree` detection and `x_expr`/`r_expr`
computation used during codegen would produce a mapping from node2's body iter_vars
to expressions in node1's `(x, r)` space.

```python
def compute_nested_reduction_var_mapping(
    node1, node2, pw_numel, red_numel
) -> dict[sympy.Symbol, sympy.Expr]:
    """Map node2's body iter_vars to expressions in node1's (x, r) space."""
    # Reuse topk_tree/other_tree logic from codegen_nested_reduction
    # to compute pass2_x, pass2_r in terms of node1's x, r variables.
    ...
```

### Concrete remapping per pattern

Pattern 1 (topk in x), node1 space `(x, r)` with `x ∈ [0, B*K)`, `r ∈ [0, DIM)`:
- `pass2_x = x // K`          (range B)
- `pass2_r = (x % K)*DIM + r` (range K*DIM, node2 reduces over K)

Pattern 2 (topk in r), node1 space `(x, r)` with `x ∈ [0, B)`, `r ∈ [0, DIM)`:
- `pass2_x = x*(DIM/G) + r//G` (range B*DIM/G)
- `pass2_r = r % G`            (range G, node2 reduces over G)

### Substitution chain for node2 subnodes

1. Get read/write expressions in terms of node2's body iter_vars
2. Substitute `node2_body_var[i]` → `pass2_expr_i(x, r)` (nested reduction remap)
3. Substitute `x, r` → `norm_pw_vars, norm_red_vars` (standard normalization)

### Integration with extract_normalized_read_writes

```python
if isinstance(node, FusedNestedReductions):
    # Use node1's (numel, rnumel) as canonical space
    pointwise_numel, red_numel = node.node1.group[1]

    # Node1 subnodes: standard get_pw_red_splits path
    for n in node.node1.get_nodes():
        ...  # existing code

    # Node2 subnodes: bypass get_pw_red_splits, apply remap directly
    remap = compute_nested_reduction_var_mapping(...)
    for n in node.node2.get_nodes():
        body = n._body
        for inp in inputs:
            for expr in body.get_all_read_expr(inp):
                remapped = sympy_subs(expr, remap)  # now in (x, r) space
                normalized = sympy_subs(remapped, norm_map)  # now in (n0, n1, ...)
                reads[normalized].add(inp)
        # same for writes
```

### Why node1 dominates tiling decisions anyway

Node1 reads the full shared input (`B*K × DIM` elements). Node2 re-reads the same
input but also reads node1's reduction output (small, `B*K` elements) and writes
a smaller output (`B × DIM` or `B × DIM/G`). The shared input dominates memory
traffic, so node1's access patterns should drive XBLOCK/RBLOCK choices regardless.
Including node2's accesses would provide marginal improvement in edge cases where
node2 has additional large external loads.

### When to implement

When nested reductions are integrated into `codegen_node` rather than bypassing it,
or when tiling quality for nested reduction kernels needs improvement.
