# Consumer Reindex: Ours vs. PR #188637

This compares the relevant shape of the implementations. It is illustrative,
not a patch against the branch.

## 1. Ours: flatten, then retry the existing inversion path

Our preflight symbolically models the flat reindex without rebuilding the body:

```python
flat_size = sympy_product(producer_write.size)
flat_var = sympy.Dummy("reindex_flat", integer=True, nonnegative=True)
flattened_read = sympy_subs(
    read_expr,
    dict(zip(iter_vars, decompose_index(flat_var, iter_sizes))),
)
flattened_read = V.graph.sizevars.simplify_with_ranges(
    sympy.expand(flattened_read), {flat_var: flat_size}
)

inverse = generate_inverse_formula(flattened_read, flat_var, flat_size)
return inverse is not None
```

If the preflight succeeds, we mutate and recursively use the existing
one-dimensional inversion implementation:

```python
snapshot = _LoopStateSnapshot.create((consumer,))
consumer.apply_loop_reindexing([flat_size])
score = self.shared_data_after_inverting_indexing(producer, consumer)
if score < 0:
    snapshot.restore()
return score
```

Properties:

- No new generic reindex callback API.
- Non-invertible cases do not rebuild the `LoopBody`.
- The successful path analyzes the inverse twice.
- The successful path refreshes dependencies once for flattening and again for
  applying the inverse.
- The existing inversion path can handle a unary epilogue, not only a copy.

## 2. Theirs: directly map producer coordinates to old consumer coordinates

PR #188637 computes the inverse once and closes over it in an `indexer`:

```python
flat_var = sympy.Dummy("copy_flat", integer=True, nonnegative=True)
old_iter_idx = decompose_index(flat_var, old_iter_sizes)
flattened_read = sympy_subs(
    read_expr, dict(zip(iter_vars, old_iter_idx))
)
inverse_formula = generate_inverse_formula(flattened_read, flat_var)

def indexer(
    new_iter_idx: Sequence[sympy.Expr],
    old_iter_sizes: Sequence[sympy.Expr],
) -> Sequence[sympy.Expr]:
    producer_index = sympy_subs(
        producer_write.index,
        dict(zip(producer_write.var_names, new_iter_idx)),
    )
    old_flat = sympy_subs(inverse_formula, {flat_var: producer_index})
    return decompose_index(old_flat, old_iter_sizes)
```

The callback means: for each producer iteration, find its physical write
address, invert the consumer read permutation to recover the old consumer flat
iteration, then decompose that flat iteration into the old consumer coordinates.

It applies that mapping directly:

```python
consumer.apply_loop_reindexing(
    list(producer_write.size),
    indexer=indexer,
    refresh_dependencies=False,
)

canonical_read = sympy_subs(
    producer_write.index,
    dict(zip(producer_write.var_names, consumer._body.iter_vars)),
)
consumer._apply_indexing_expr_replacements(
    {load_entry.index_name: canonical_read},
    normalize=True,
    need_clear_tiling_cache=True,
)
```

Properties:

- One inverse analysis and one dependency extraction on success.
- The resulting consumer loop domain directly matches the producer domain.
- Requires optional `indexer` and `refresh_dependencies` arguments in the
  general `LoopBody` and `SchedulerNode` reindex APIs.
- Requires separately canonicalizing the load expression after proving it is
  equivalent.
- Currently restricted to a pure single-load/single-store memory copy.

## 3. Minimal hybrid: their indexer with our bounded analyzer

The smallest useful adoption would make our preflight return the callback rather
than a boolean. There is no cache or plan object:

```python
def _consumer_reindex_for_index_inversion(
    self,
    producer_write: MemoryDep,
    consumer_read: MemoryDep,
    consumer_write: MemoryDep,
    consumer: SchedulerNode,
    read_expr: sympy.Expr,
) -> tuple[
    list[sympy.Expr],
    Callable[
        [Sequence[sympy.Expr], Sequence[sympy.Expr]],
        Sequence[sympy.Expr],
    ],
] | None:
    # Keep our existing cheap structural and equal-numel checks here.
    flat_size = sympy_product(producer_write.size)
    body = consumer._body
    if body is None:
        return None

    iter_vars = body.vars[0]
    old_iter_sizes = body.sizes[0]
    flat_var = sympy.Dummy("reindex_flat", integer=True, nonnegative=True)
    flattened_read = sympy_subs(
        read_expr,
        dict(zip(iter_vars, decompose_index(flat_var, old_iter_sizes))),
    )
    flattened_read = V.graph.sizevars.simplify_with_ranges(
        sympy.expand(flattened_read), {flat_var: flat_size}
    )

    # Keep our finite-domain coverage checks and shared canonicalization.
    inverse = generate_inverse_formula(flattened_read, flat_var, flat_size)
    if inverse is None:
        return None

    producer_sizes = list(producer_write.size)

    def indexer(
        new_iter_idx: Sequence[sympy.Expr],
        old_iter_sizes: Sequence[sympy.Expr],
    ) -> Sequence[sympy.Expr]:
        producer_index = sympy_subs(
            producer_write.index,
            dict(zip(producer_write.var_names, new_iter_idx)),
        )
        old_flat = sympy_subs(inverse, {flat_var: producer_index})
        return decompose_index(old_flat, old_iter_sizes)

    return producer_sizes, indexer
```

The mutation site becomes:

```python
reindex = self._consumer_reindex_for_index_inversion(
    producer_write, consumer_read, consumer_write, consumer, read_expr
)
if reindex is not None:
    producer_sizes, indexer = reindex
    snapshot = _LoopStateSnapshot.create((consumer,))
    consumer.apply_loop_reindexing(
        producer_sizes,
        indexer=indexer,
        refresh_dependencies=False,
    )

    load_entry = consumer._body.memory_usage[MemoryUsageType.LOAD][0]
    canonical_read = sympy_subs(
        producer_write.index,
        dict(zip(producer_write.var_names, consumer._body.iter_vars)),
    )
    consumer._apply_indexing_expr_replacements(
        {load_entry.index_name: canonical_read},
        normalize=True,
        need_clear_tiling_cache=True,
    )

    score = self.score_fusion_memory(producer, consumer)
    if score < 0:
        snapshot.restore()
    return score
```

This still needs the three pieces of generic plumbing from their PR:

```python
LoopBody.reindex_iter_loops(new_sizes, indexer=None)
SchedulerNode.apply_loop_reindexing(
    new_sizes, indexer=None, *, refresh_dependencies=True
)
SchedulerNode._apply_indexing_expr_replacements(...)
```

The callback itself is simple. Most of the additional surface comes from
avoiding the second dependency extraction and from canonicalizing the proven
read. If we retain support for unary epilogues, this direct memory-copy path
would need to coexist with our current recursive fallback.
