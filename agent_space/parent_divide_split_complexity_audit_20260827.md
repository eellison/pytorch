# F2b parent divide-before-split complexity audit

Date: 2026-08-27

Compared worktrees:

- F2a baseline: `agent_space/followup_lazy_projection_wt`
- F2b prototype: `agent_space/followup_parent_divide_split_wt`
- audited `simd.py` SHA256:
  `899c315c6e97b3f4891cd30b9f2164acf0eb9eea35531b410a274ca9ad11321e`

This audit is read-only. It measures the F2b delta over the final F2a API, not
the cumulative stack.

## Verdict

Reject this F2b prototype.

It fails both explicit complexity gates and the shipped-benefit gate:

- production delta: `+189/-12`, or `+177` net lines in `simd.py`;
- `_PointwiseRemapHandler._parent_width_binary`: McCabe CC 15, nesting depth 3;
- default-selected persistent `4096x1024`: `6.151 -> 6.144 us`, effectively a
  tie; and
- the default-selected configurations show no measurable benefit despite
  emitting the intended parent-width form.

The forced-persistent result is real and useful evidence, but not enough to
land this mechanism: `4096x4096` improves `28.667 -> 22.528 us` (`1.273x`),
with 255 registers, 104 spills, and 2048 bytes shared. The default scheduler
does not select that persistent configuration.

## Method

Cyclomatic complexity uses the installed `mccabe` package. Nesting comes from a
separate AST walk and ignores nested function bodies. As a stricter secondary
signal, an AST branch counter also counts each comprehension and boolean arm.

Across changed functions, aggregate McCabe complexity grows from 30 in F2a to
69 in F2b. Newly introduced functions account for 137 physical LOC and McCabe
CC 35. The stricter branch-count aggregate grows from 37 to 96, and reports 26
for `_parent_width_binary` alone.

## Per-method results

| Method | F2a LOC/McCabe/depth | F2b LOC/McCabe/depth | Assessment |
|---|---:|---:|---|
| `_PointwiseRemapHandler._parent_width_binary` | absent | 63 / 15 / 3 | Reject: exceeds both CC and nesting gates |
| `_PointwiseRemapHandler.to_dtype` | 15 / 3 / 1 | 29 / 7 / 2 | Duplicates group and pending-parent policy |
| `_PointwiseRemapHandler.reciprocal` | absent | 15 / 3 / 2 | Adds a second group-chain preservation path |
| `_PointwiseRemapHandler._materialize_value` | absent | 8 / 4 / 1 | Necessary only because two lazy value kinds coexist |
| `_PointwiseRemapHandler.masked` | absent | 11 / 2 / 0 | Required callback barrier once pending values exist |
| `_PointwiseRemapHandler.mul` | absent | 3 / 1 / 0 | Public interception surface for the helper |
| `_PointwiseRemapHandler.truediv` | absent | 3 / 1 / 0 | Public interception surface for the helper |
| `_SubParentValueResolver.materialize_parent_lane` | absent | 22 / 5 / 2 | Reasonable local split/cache operation |
| `_SubParentValueResolver.resolve_load` | 21 / 5 / 2 | 28 / 6 / 2 | Adds a second lazy result kind |
| `_SubParentValueResolver.materialize_sources` | 29 / 8 / 3 | 31 / 9 / 3 | Already complex; F2b adds another control exception |
| `_SubParentValueResolver.__init__` | 38 / 3 / 2 | 40 / 3 / 2 | Adds mode and cache state |
| shape/liveness helpers (four total) | absent | 12 / 4 total | Small individually, but expose a new domain policy API |

Unchanged-looking `_default`, `load`, and `store` also change semantically
because their accepted value type and materialization behavior now include the
pending-lane record.

## New state and API surface

F2b adds:

1. `_PendingSubParentLane(value, lane)`, a second non-CSE value flowing through
   the generic pointwise handler;
2. `defer_parent_lanes`, a mode bit derived from persistence and factor;
3. `_parent_lane_splits`, a stage-local all-lanes cache;
4. union return/input types on `load`, `store`, `to_dtype`, and `resolve_load`;
5. explicit `mul`, `truediv`, `reciprocal`, and `masked` interception;
6. parent-scalar, parent-width, group-to-parent, liveness, and parent-lane
   materialization APIs on the resolver; and
7. a new exception in eager pre-epilogue source materialization.

The dataclass and split cache are the minimum plausible state. The rest is a
small domain interpreter embedded in `_PointwiseRemapHandler`.

## Duplicated policy and proof

The same domain facts are re-derived in several places:

- `resolve_load` decides whether a relation may become a pending parent lane;
- `_parent_width_binary` reclassifies every operand as pending, group,
  parent-singleton, or unsupported;
- `reciprocal` separately recognizes group values;
- `to_dtype` separately recognizes both group and pending-parent values; and
- `_default`, `store`, and `masked` each define a materialization boundary.

This is the historical projection/layout machinery in smaller form: the enum is
gone, but operation-specific branches still carry an implicit domain state
machine. Extending the supported group scale chain would add more overrides and
more duplicated shape-preservation checks.

There is also a representation mismatch at the current group-to-parent join.
`is_group_width_shape` accepts rank-1 group values for singleton passthrough,
while `maybe_broadcast_value_to_parent_resolution` returns rank-1 values
unchanged. Supporting B=1 therefore needs either an explicit
group-to-parent-width helper or a fail-closed rank-1 rejection. The current API
does not make that contract structural.

## Why the current implementation is too broad

A clean F2a relation trace records the final FP8 scale directly:

```text
MemoryDep('buf2', c0, {c0: 8192})
    -> live tmp26 at group width, dtype FP8
    -> exact pack-consumer relation in both lane replays
```

The earlier conclusion that only raw amax was available came from a contaminated
capture whose local model also returned the widened float scale. The real graph
does not require a planner change or a lazy scale-expression interpreter.

This makes the current group-only branch in `_parent_width_binary` especially
hard to justify. A smaller implementation can keep only the final scale cast
and reciprocal at group width, join it with a pending parent expression, and
split once. The reciprocal can emit its underlying constant/divide directly;
it does not require `_parent_width_binary` to authorize arbitrary group-only
`mul`/`truediv` chains. Even that smaller implementation is not worth landing
today because fresh default-selected measurements are ties.

## Fresh benchmark provenance

The final default measurements were rerun from clean FX/Inductor caches using
the real benchmark graph, which returns the original FP8 scale. They did not use
the contaminated local prototype model. `f1_perf_baseline_20260826.py` calls
`torch._dynamo.reset()`, uses `fresh_inductor_cache()`, and disables the FX graph
cache for every compiled case.

Fresh F2a capture:

```bash
CUDA_VISIBLE_DEVICES=1 \
PYTORCH_WORKTREE=/data/users/eellison/pytorch/agent_space/followup_lazy_projection_wt \
conda run --no-capture-output -n pytorch-3.12 python \
  agent_space/domain_projection_work/run_worktree_script.py \
  agent_space/f1_perf_baseline_20260826.py \
  --formats nvfp4 --shapes 4096x512 4096x1024 --modes default \
  --label f2a_final --skip-live-timing \
  --output agent_space/f2b_final_audit_20260827/f2a_results.json \
  --generated-dir agent_space/f2b_final_audit_20260827/f2a_generated
```

Fresh F2b capture against audited SHA `899c315c...`:

```bash
CUDA_VISIBLE_DEVICES=1 \
PYTORCH_WORKTREE=/data/users/eellison/pytorch/agent_space/followup_parent_divide_split_wt \
conda run --no-capture-output -n pytorch-3.12 python \
  agent_space/domain_projection_work/run_worktree_script.py \
  agent_space/f1_perf_baseline_20260826.py \
  --formats nvfp4 --shapes 4096x512 4096x1024 --modes default \
  --label f2b_final --skip-live-timing \
  --output agent_space/f2b_final_audit_20260827/f2b_results.json \
  --generated-dir agent_space/f2b_final_audit_20260827/f2b_generated
```

Paired wrapper replay:

```bash
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n pytorch-3.12 \
  python agent_space/f2b_final_audit_20260827/replay_compare.py
```

Artifacts:

- `agent_space/f2b_final_audit_20260827/f2a_results.json`
- `agent_space/f2b_final_audit_20260827/f2b_results.json`
- `agent_space/f2b_final_audit_20260827/f2a_forced_results.json`
- `agent_space/f2b_final_audit_20260827/f2b_forced_results.json`
- `agent_space/f2b_final_audit_20260827/timings.json`
- both sets of generated wrappers under the adjacent `*_generated` directories

The F2b wrappers visibly contain the target order: final FP8 scale conversion,
group reciprocal, group-to-parent broadcast, one parent-width multiply, then
one `tl.split`. Results were bitwise equal to F2a.

The forced/default difference is therefore not a fusion or expression-shape
difference. It is launch pressure:

| Shape/config | F2a resources | F2b resources | Result |
|---|---|---|---|
| forced 4096x4096, XBLOCK=8, 4 warps | 255 regs, 184 spills, 4096 B shared | 255 regs, 104 spills, 2048 B shared | `1.273x` |
| default 4096x1024, XBLOCK=1, 1 warp | 72 regs, 0 spills, 512 B shared | 66 regs, 0 spills, 512 B shared | `1.001x` |
| default 4096x512, XBLOCK=2, 1 warp | 69 regs, 0 spills, 512 B shared | 60 regs, 0 spills, 512 B shared | `1.000x` |

Divide-before-split removes the expensive layout conversion and spill pressure
from the forced wide-X tile. Default selection already avoids that pressure
with XBLOCK 1 or 2, so reducing registers does not change occupancy or latency.
This is why the exact same F2b kernel-form improvement is valuable only in the
forced configuration measured here.

## Could a sub-100-line version exist?

Possibly, but it is not demonstrated. The exact final-scale relation removes
the main reason for the current generic group-only binary branch. A narrowly
scoped retry could contain only:

- the pending parent-lane record and one split cache;
- pending-aware `to_dtype`;
- one direct group-width reciprocal path;
- one linear operand-lift helper for the observed parent/group `mul` join;
- optional direct `truediv` support only if a measured target needs it; and
- the existing eager boundaries for unsupported operations, stores, and
  `masked` callbacks.

That is plausibly about 95-112 net lines. Splitting operand classification from
emission should keep each new method around McCabe CC 7-9 and nesting at 2. A
hard claim of `<=100` needs an actual implementation, especially because the
rank-1 B=1 broadcast and masked fallback cannot be omitted for correctness.

Do not build that retry now. Even a perfect 95-line implementation still fails
the agreed default-performance gate on the current launch policy.

## Deletable machinery

Given the failed default-performance gate, delete the entire F2b delta and keep
F2a. In particular, remove:

- `_PendingSubParentLane`;
- `_parent_width_binary`, `mul`, `truediv`, `reciprocal`, and `masked`;
- pending-parent handling in `to_dtype`, `load`, `store`, and `_default`;
- `defer_parent_lanes` and `_parent_lane_splits`;
- parent scalar/width/group-broadcast/liveness helpers;
- `materialize_parent_lane`; and
- the eager-materialization skip and pending return in the resolver.

If future default launch policy creates a real persistent workload that benefits,
restart from the narrow contract below rather than this prototype.

## Smallest future contract

A future retry can use the exact final-scale relation that F1 already provides.
Its codegen shape should be limited to:

```text
exact live parent source + proved lane
    -> pending parent lane
exact live final group scale
    -> existing group-to-parent broadcast
same original divide/multiply at flat parent width
    -> existing factor-2 materializer once
```

The hard gates remain:

- at most 130 net production lines, with a target below 100;
- no new method above CC 10 or nesting depth 2;
- no enum, operation allowlist, generic domain wrapper, view cache, name map,
  scheduler expression reconstruction, or Triton/common change;
- bitwise equality with F2a;
- one parent-width scale application followed by one split;
- looped and factor-4 source unchanged; and
- at least 5% improvement on a default-selected persistent workload.

The current prototype fails the first, second, and last gates.
