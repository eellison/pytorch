# F2b Parent-Width Divide-Before-Split Prototype

## Decision

Reject this F2b prototype. Do not move it into the canonical F2a worktree.

The narrow mechanism produces the desired kernel and a large win for the forced
persistent, no-FP-fusion benchmark, but it does not improve the default-selected
persistent kernels. Supporting those graphs would require preserving an entire
group-width scale expression chain rather than consuming one exact planned
source. That is outside the scoped design and is not justified by the shipped
performance result.

The canonical F2a worktree was not modified:

`/data/users/eellison/pytorch/agent_space/followup_lazy_projection_wt`

The rejected prototype remains available for inspection at:

`/data/users/eellison/pytorch/agent_space/followup_parent_divide_split_wt`

Stable `simd.py` SHA256 for the final audit candidate:

`899c315c6e97b3f4891cd30b9f2164acf0eb9eea35531b410a274ca9ad11321e`

## Intended Mechanism

The prototype applies only to persistent factor-2 sub-parent codegen and exact
planned relations with `parent_lane != None`.

1. Keep an exact parent-width source and its requested lane as one deferred
   `_PendingSubParentLane` value.
2. Keep an exact group-width scale at group resolution.
3. Broadcast that scale once to parent width when it meets the pending parent
   value.
4. Perform the original arithmetic once at parent width.
5. Reuse the existing factor-2 materialization path to split the result once and
   select the planned lane.

Unsupported operations materialize through the existing F2a paths. A CSE
liveness miss raises rather than falling back to memory for a required in-kernel
source.

## Kernel-Form Result

The forced persistent 4096x4096 no-FP-fusion kernel reaches the intended form:

```text
group scale reciprocal
group -> full-parent broadcast
one full-parent multiply
one factor-2 split
```

Artifact:

`agent_space/f2b_debug6_generated/f2b_debug6_nvfp4_4096x4096_persistent_0.py`

Resources:

```text
registers: 255
spills:    104
shared:    2048 B
```

## Performance Gates

Paired measurements supplied by the independent review session:

| Case | F2a | F2b | Speedup | Gate |
|---|---:|---:|---:|---|
| Forced persistent 4096x4096 | 28.674 us | 22.531 us | 1.273x | Pass |
| Default persistent 4096x1024 | 6.151 us | 6.145 us | 1.001x | Fail |

The keep criterion required meaningful gains in default-selected persistent
cases, not only a forced configuration. The 4096x1024 result is a tie and fails
that criterion.

## Why Default Codegen Does Not Benefit

The default 128x1024 relation audit shows exact lane relations for the parent
input and weight, but the only planned group-width scale-side source is `buf1`,
the raw `amax`:

```text
arg0_1 parent lane 0/1
arg1_1 parent lane 0/1
buf1   group-width amax, parent_lane=None
```

There is no exact relation for the final FP8 scale. Consequently the sub-parent
replay starts from `amax` and recomputes this chain at child width:

```text
scale multiply -> casts -> clamp -> FP8 conversion -> float conversion
```

The direct lane division therefore still occurs after `tl.split`. Generated
artifact:

`agent_space/f2b_relation_probe_128x1024.kernel.py`

Teaching the remap handler to preserve `mul`, `maximum`, `minimum`, casts, and
future scale expressions at group width would recreate a broad expression
interpreter. It is not an exact-source-only optimization and was intentionally
not pursued.

## Complexity

Relative to final F2a, the stable candidate adds:

```text
torch/_inductor/codegen/simd.py: +189 / -12, net +177 production LOC
test/inductor/test_inductor_scheduler.py: +3 fixture initializations
```

The central `_parent_width_binary` helper is 63 physical lines and has rough
branch-count complexity 24. This misses the additional review target of at most
about 130 net lines and no new method above roughly CC 10.

F2a itself is net +67 production lines, so the combined F2a+F2b prototype is
about net +244 production lines. That is below the original +300 ceiling but is
still disproportionate for an optimization that does not benefit the default
graphs.

## Additional Correctness Constraint

For batch size 1, a group value may be rank 1. The current classification can
recognize it as group width, while
`maybe_broadcast_value_to_parent_resolution` intentionally leaves rank-less-than-2
values unchanged. A retained implementation would need either an explicit
rank-1 group-to-parent broadcast or a fail-closed rank-1 decline. The prototype
does not claim B=1 coverage.

## Validation Performed

```bash
conda run --no-capture-output -n pytorch-3.12 \
  python -m py_compile torch/_inductor/codegen/simd.py
```

```bash
CUDA_VISIBLE_DEVICES=1 \
PYTORCH_WORKTREE=/data/users/eellison/pytorch/agent_space/followup_parent_divide_split_wt \
conda run --no-capture-output -n pytorch-3.12 python \
  agent_space/domain_projection_work/run_worktree_script.py \
  agent_space/f1_perf_baseline_20260826.py \
  --formats nvfp4 --shapes 4096x4096 --modes persistent \
  --label f2b_debug6 --skip-live-timing \
  --output agent_space/f2b_debug6_results.json \
  --generated-dir agent_space/f2b_debug6_generated
```

Earlier exact NVFP4 numerical checks passed for persistent and looped B=1/128
on an intermediate flat prototype. Broad tests, factor-4 regression tests, and
lint were deliberately not run after the default-performance and complexity
gates failed.

## Follow-Up Condition

Revisit divide-before-split only if graph/planner work exposes the final scale
as an exact planned source in the default graph. At that point F2b can remain a
narrow parent-lane plus group-scale join. Do not revive the broad group-expression
handling from this prototype.
