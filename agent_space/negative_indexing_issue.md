## Negative index value semantics differ from indexing equivalents

### Summary

PyTorch eager has different behavior for negative integer index tensor values depending on the indexing API. This issue tests a valid negative index value (`-1`) and checks whether each operator matches:

- the closest ordinary PyTorch indexing expression, e.g. `x[idx]`, `x[:, idx]`, or `x[rows, idx]`
- the same ordinary NumPy indexing expression where there is a reasonable one

For the comparable cases in this script, the PyTorch and NumPy indexing expressions all wrap valid `-1`. The question is which PyTorch operators should also wrap and which are intentionally stricter.

### Reproduction

Standalone script:

```bash
python agent_space/negative_indexing_discrepancies.py
```

Optional dispatch trace:

```bash
python agent_space/negative_indexing_discrepancies.py --dispatch
```

The script checks a valid negative index value (`-1`) against each PyTorch operator call, the closest PyTorch indexing expression, and the parallel NumPy indexing expression when available. `wraps` means the expression accepts `-1` as indexing from the end. `rejects` means it errors. `consistent` means the PyTorch operator has the same wrap/reject behavior as the PyTorch indexing expression.

### Observed output

On my checkout:

```text
torch: 2.13.0a0+gite1067a5
numpy: 2.2.6

consistent:
torch `x[idx]` wraps; numpy `x[idx]` wraps
torch `out.index_put_((idx,), values)` wraps; torch indexing `out[idx] = values` wraps; numpy indexing `out[idx] = values` wraps
torch `torch.take(x, idx)` wraps; torch indexing `x.flatten()[idx]` wraps; numpy indexing `x.flatten()[idx]` wraps
torch `out.index_fill(0, idx, value)` wraps; torch indexing `out[idx] = value` wraps; numpy indexing `out[idx] = value` wraps

inconsistent:
torch `torch.gather(x, 1, idx)` rejects; torch indexing `x[rows, idx]` wraps; numpy indexing `x[rows, idx]` wraps
torch `out.scatter(1, idx, src)` rejects; torch indexing `out[rows, idx] = src` wraps; numpy indexing `out[rows, idx] = src` wraps
torch `F.embedding(idx, weight)` rejects; torch indexing `weight[idx]` wraps
torch `torch.index_select(x, 1, idx)` rejects; torch indexing `x[:, idx]` wraps; numpy indexing `x[:, idx]` wraps
torch `torch.index_add(out, 0, idx, src)` rejects; torch indexing `out[idx] += src` wraps; numpy indexing `out[idx] += src` wraps
torch `torch.index_copy(out, 0, idx, src)` rejects; torch indexing `out[idx] = src` wraps; numpy indexing `out[idx] = src` wraps
torch `torch.index_reduce(out, 0, idx, src, 'mean', include_self=True)` rejects; torch indexing `out[idx] = (out[idx] + src) / 2` wraps
```

### Notes

The surprising cases are:

- `gather`, `scatter`, `embedding`, `index_select`, `index_add`, `index_copy`, and `index_reduce` reject `-1` even though the closest PyTorch indexing expression wraps it.
- The NumPy indexing expressions also wrap `-1` for the comparable cases.
- `embedding` has no direct NumPy operator equivalent; the comparison shown is to the closest indexing expression, `weight[idx]`.
- The indexing expressions are comparison points for negative-index handling, not claims that the full operator semantics are identical for every case. For example, repeated indices may differ between `torch.index_add` and `out[idx] += src`.
- A `TorchDispatchMode` sanity check confirms these are different dispatched operators: indexing reads use `aten.index.Tensor`, indexing writes use `aten.index_put_.default`, while the rejecting APIs dispatch to dedicated ops such as `aten.gather.default`, `aten.scatter.src`, `aten.embedding.default`, `aten.index_select.default`, `aten.index_add.default`, `aten.index_copy.default`, and `aten.index_reduce.default`.
- `index_fill` behaves like indexing, while sibling-style ops such as `index_add`, `index_copy`, and `index_reduce` do not.

### Question

Which operator rows that reject `-1` are intentional PyTorch semantics, and which should be treated as inconsistencies to document or change?
