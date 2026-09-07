# F2a cast-only group-width CSE: adversarial plan

Date: 2026-08-26

Target:
`/data/users/eellison/pytorch/agent_space/f2a_narrow_prototype_wt`

Reference:
`/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt`

The prototype and canonical F2 worktree have the same complete diff hash, so
prototype results transfer once that equality is maintained. All runners and
outputs remain under `agent_space/`.

## Contract under test

1. `resolve_load` returns a raw source only when it is lane-free, unguarded,
   live, and truly group-width.
2. `to_dtype` is the only operation allowed to consume that raw value.
3. The cast result is verified group-width and immediately materialized.
4. Every non-cast operation, masked or scalar-fill load, store, lane projection,
   unknown shape, and direct `G == factor` value follows the eager F1 path.

## CPU policy fuzz

Use real `CSEVariable`, guard, and relation objects with mocked emission. Cover:

- all six combinations at the resolver admission boundary;
- cast before broadcast and immediate post-cast materialization;
- loud cast shape drift;
- 500 non-cast recipes with nested pytrees and mixed widths;
- masked callback and scalar-fill guards;
- ordinary and reduction stores; and
- the `num_groups == child_block` direct-width ambiguity.

Required mutants:

- disable raw resolution;
- allow guarded raw resolution;
- skip post-cast materialization;
- materialize before the cast;
- allow non-cast raw use;
- classify `G == factor` as group-width; and
- skip store materialization.

## GPU differential

Capture F1 and compare F2 using two deterministic input sets. Require exact
integer, FP8, and packed outputs. Compare floating outputs bitwise first and
record any tolerance fallback explicitly.

The compact matrix covers persistent and looped NVFP4, cast followed by
non-cast arithmetic, a boolean non-cast barrier, the checked-in mismatched
masked-source graph, `G == factor`, factor-4/MXFP6 protected paths, plus looped
dynamic-R, tail, reciprocal, and indirect-index cases.

Generated-source checks require:

- NVFP4: one FP8 conversion, no uint8 bitcast round-trip, and cast before the
  group-to-child broadcast;
- the synthetic cast chain: broadcast before all later non-cast work;
- the boolean barrier: broadcast before `where`;
- one staged kernel and one total kernel for every case;
- one graph for all dynamic-R inputs; and
- byte-identical F1 source for masked/scalar-fill, `G == factor`, factor-4,
  MXFP6, and indirect-tail protected cases.

GPU mutations should detect the lost optimization, cast/broadcast reversal,
and raw non-cast use. Guard-only and immediate-materialization invariants may
remain CPU tripwires when no valid graph exposes the intermediate state.

## Result

Executed successfully. See:
`/data/users/eellison/pytorch/agent_space/lazy_cse_fuzz_report_20260826.md`.
