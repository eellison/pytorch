# Coalescing-analysis fuzz findings (2026-09-03)

Base: `5bd51f56358846780bd3d0f6ab2faa6543be474d`

Clean fix worktree: `/data/users/eellison/pytorch/agent_space/dcn_mxfp6_5bd51`

The worktree has two uncommitted source files:

- `torch/_inductor/tiling_utils.py`
- `test/inductor/test_loop_ordering.py`

## Confirmed bugs

### 1. `FloorDiv.is_constant()` compile crash

`solve_for_zero(FloorDiv(Mod(n1, 4), 2))` queried SymPy's generic
`is_constant()` implementation before rejecting `FloorDiv`. Random rational
substitution then reached an assertion in `FloorDiv.eval`.

Fix: reject top-level `FloorDiv` before calling `is_constant()`.

Standalone repro:
`agent_space/repro_coalesce_floordiv_is_constant.py`.

### 2. Unsound `ModularIndexing` zero solution

For `ModularIndexing(x - 4, 6, 9)`, the old solver returned `x = 13`, but the
expression evaluates to 1 there. It solved `base == modulus`, implicitly
treating the divisor as 1.

Fix: solve `base == divisor * modulus`, and substitute every proposed result
back into the original expression. `solve_for_tiling` also validates its final
candidate against the complete original expression.

Minimal escape through `solve_for_tiling`:

```python
x = sympy.Symbol("x", integer=True, nonnegative=True)
expr = 32 * ModularIndexing(x - 4, 6, 9) + FloorDiv(x, 13)
```

The old analysis returned 13 although `expr.subs(x, 13) == 33`.

### 3. Unsound direct-term coalescing shortcut

`find_coalesced_var` immediately returned a variable if it appeared as a
top-level additive term, without checking whether another term also depended
on it. For `x + FloorDiv(x, 2)`, it returned `x` although addresses at
`x = 0, 1, 2` are `0, 1, 3`.

Fix: remove the shortcut and use the existing consecutive-address check.

### 4. Incorrect collapsed-coordinate remapping

When a `[5, 6]` node was normalized to a `[30]` domain, `apply_var_mapping`
produced `{i: 6*n, j: n}`. A contiguous address `6*i + j` consequently became
`37*n` instead of `n`. An actual fused `[5, 6]` plus `[3, 10]` graph produced
normalized input addresses `37*n0` and `101*n0`.

Fix: unflatten each normalized coordinate with `FloorDiv` and
`ModularIndexing`. The same graph now produces `n0` for both inputs.

Standalone graph probe:
`agent_space/probe_collapsed_coalesce_graph.py`.

### 5. Buffer metadata lost when normalized expressions collided

Both the per-node remap and final range-aware simplification used dict
comprehensions. If distinct expressions normalized to the same address, the
last buffer set overwrote earlier sets, undercounting memory traffic.

Fix: union buffer-name sets at both normalization stages. The collapsed graph
now records `{n0: {arg0_1, arg1_1}}`.

## Fuzzing and validation

- Symbolic helper fuzzer: four seeds, 4,800 expressions after the fixes; no
  crashes or unsound returned zero/tiling/coalescing results.
- Coordinate-remapping fuzzer: four seeds, 40,000 factorizations; 31,738
  compatible mappings all preserved bounds and row-major linear offsets. The
  remaining 8,262 layouts failed closed with `CantSplit`.
- CUDA graph fuzzer: 60 static/dynamic graphs covering collapsed reshapes,
  reductions, transposes, broadcasts, and padding; all compiled and matched
  eager.
- `test/inductor/test_loop_ordering.py`: 127/127 passed.
- `spin quicklint torch/_inductor/tiling_utils.py test/inductor/test_loop_ordering.py`:
  passed.

## DCN checks

- Unpadded native MXFP6 pack: one nested kernel, four stores, 8.90 us (prior
  run 8.81 us).
- Padded 2048x3072 integration: compiled correctly, padding remained zero,
  23.96 us (prior 23.85 us).
- The known fused-arithmetic difference from eager is unchanged: 67 scale and
  16,359 packed-byte mismatches. Native and compiled-software paths remain
  bit-exact to each other.

Scratch fuzzers:

- `agent_space/fuzz_coalesce_symbolic.py`
- `agent_space/fuzz_coalesce_mapping.py`
- `agent_space/fuzz_coalesce_end_to_end.py`
