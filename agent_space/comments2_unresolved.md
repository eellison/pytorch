# Unresolved Review Items

## `max_xblock`

Status: unresolved.

Current behavior: nested reduction may set `max_xblock` as conservative tuner
metadata to cap XBLOCK when a required `min_rblock` or `min_xblock` would
otherwise allow too large a total tile.

Open decision: keep this metadata path as the conservative cap, or remove it
entirely and rely only on existing Triton block-size heuristics plus
`min_xblock`/`min_rblock`.

Relevant files:
- `torch/_inductor/codegen/simd.py`
- `torch/_inductor/runtime/triton_heuristics.py`
- `test/inductor/test_coordinate_descent_tuner.py`
- `test/inductor/test_nested_reduction.py`

Current test coverage:
- coordinate-descent metadata limits
- rounded non-power-of-two XBLOCK cap case

