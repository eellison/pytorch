# MXFP6 native conversion and scale-swizzle validation

Date: 2026-08-26

This is a scratch-only investigation. No reviewed production worktree was
modified.

## Recommendation

Use `inline_asm_elementwise(..., pack=2)` for the full-resolution FP32-to-E2M3
conversion while preserving the logical `[rows, groups, 32]` result. Feed that
result to the existing `(4, 3)` staged packing epilogue.

This is the smallest robust form found:

- each HOP output depends only on the corresponding input element, so the
  unspecified physical pairing used by `pack=2` does not affect semantics;
- the value remains in the full-resolution source frame already supported by
  the staged planner;
- generated code contains one nested Triton kernel and no converted-value
  materialization;
- it is substantially faster than both software conversion and the extracted
  AITER kernel.

Do not use the apparently faster `pack=4` direct-triplet experiment. Each
output there depends on four logical inputs, but `inline_asm_elementwise` does
not promise which logical elements are grouped together. Its correctness was
an accidental lowering property rather than an API guarantee.

## Representative quantization matrix

GPU: NVIDIA B200. Coordinate descent disabled. The main comparison forces the
persistent outer-reduction choice; a default-choice rerun selected equivalent
kernels and produced the same timings. Times are CUDA Graph medians.

| Shape | AITER | Grouped software | Native pack=2 | Native / AITER |
|---|---:|---:|---:|---:|
| 8192x128 | 4.710 us | 3.749 us | 2.498 us | 0.530x |
| 32768x128 | 12.890 us | 10.792 us | 5.387 us | 0.418x |
| 1024x1024 | 4.363 us | 3.768 us | 2.498 us | 0.573x |
| 128x12288 | 5.899 us | 5.162 us | 3.031 us | 0.514x |

Every native result was bit-exact to AITER and compiled as one generated
kernel with `codegen_nested_reduction == 1`, one inline-asm call site, and four
stores.

Final exact-worktree rerun:

```bash
cd /data/users/eellison/pytorch/agent_space/pr191775_layered_split_wt
PYTHONPATH=/data/users/eellison/pytorch conda run --no-capture-output \
  -n pytorch-3.12 python \
  /data/users/eellison/pytorch/agent_space/bench_aiter_vs_inductor_grouped_native_subagent.py
```

The `8192x128` selected native kernel used `XBLOCK=32`, two warps, one stage,
32 registers per thread, zero spills, and 256 bytes of shared memory. The
software-conversion comparison selected `XBLOCK=8`, two warps, one stage, 30
registers, zero spills, and 8 bytes of shared memory. Repeated isolated timing
from batch sizes 1 through 500 converged to about 2.45-2.50 us, so the result is
not a CUDA Graph division artifact.

## Production-like DCN plus aligned scale preshuffle

Shape: `(2048, 3072)`. Scratch worktree:
`/data/users/eellison/pytorch/agent_space/mxfp6_43_append_guard_fix`.

For an archival rerun against the final reviewed worktree, set:

```bash
MXFP6_PROTOTYPE=/data/users/eellison/pytorch/agent_space/pr191775_layered_split_wt \
  PYTHONPATH=/data/users/eellison/pytorch \
  conda run --no-capture-output -n pytorch-3.12 python \
  /data/users/eellison/pytorch/agent_space/bench_dcn_scale_only_native_pack2.py
```

| Form | Time | Kernels | Nested | Asm sites |
|---|---:|---:|---:|---:|
| Fused software conversion | 16.381 us | 1 | 1 | 0 |
| Native pack=2 | 8.699 us | 1 | 1 | 1 |
| Native pack=2 plus zero canonicalization | 9.014 us | 1 | 1 | 1 |
| Native pack=2 plus scale reciprocal asm | 9.010 us | 1 | 1 | 2 |
| Corrected reference | 24.166 us | n/a | n/a | n/a |

All Inductor variants selected a 6,144-CTA launch with `XBLOCK=32`,
`R0_BLOCK=32`, two warps, and one stage. Software conversion used 47 registers
per thread; each native variant used 32. None spilled. The corrected reference
used grid 2,509, `GROUP_LOAD=64`, 79 groups per program, two loop iterations,
and four warps.

The native, canonicalized-native, and reciprocal outputs were bit-exact to the
compiled software graph. All variants had one `log2`/`ceil` scale computation;
native pack=2 had one `exp2`, while the reciprocal form removed it. The
reciprocal did not improve this production-like workload, so it should not be
part of the initial change.

The compiled software and native graphs have the same pre-existing differences
from eager/corrected-D DCN: 67 of 196,608 scale bytes and 16,359 of 4,718,592
packed bytes for the seeded input. The intrinsic introduced no additional
difference.

Native conversion preserves the sign of `-0.0`, matching AITER, while the old
Python software helper canonicalizes it. If compatibility with that helper is
required, an explicit `where(scaled == 0, 0, code)` is exact and costs about
0.3 us in this workload.

Generated source:
`/data/users/eellison/pytorch/agent_space/dumps/dcn_scale_only_native_pack2_0.py`.

## Odd and masked tails

At `rows=7`, `width=96` (three groups per row):

- native pack=2 is bit-exact;
- one kernel, one nested reduction, one inline-asm site;
- all four stores carry the appropriate X or derived-lane mask;
- runtime was 1.330 us in the combined scratch worktree.

The packed stores use `lane4_r0_index_mask & xmask`; the scale store uses
`xmask`. Thus the tail does not rely on unmasked extra elements reaching the
asm or stores.

## PR #193599 interaction

PR #193599 adds `PaddedScatter` plus an auxiliary padding-write loop. It can
fuse the dynamic padded XDL scale layout with the native `(4, 3)` pack, but it
does not solve the original full-resolution data-field permutation mismatch.

For the same `rows=7`, three-groups-per-row case, a scratch combination of
#193599, the staged stack, and the exact-type append guard produced:

- one generated kernel and one nested reduction;
- one native conversion asm site;
- five masked stores: scale, three packed-byte lanes, and auxiliary padding;
- bit-exact output;
- 1.536 us.

One integration detail is required: the staged planner's blanket
alias/mutation rejection must be scoped to epilogue nodes. `PaddedScatter` is a
parent-stage output mutation; rejecting all plan-node mutation prevents the
otherwise valid fusion. This was tested only in scratch at
`/data/users/eellison/pytorch/agent_space/mxfp6_native_tail_pr193599`.

For statically aligned scale preshuffle, #193599 is unnecessary. The ordinary
reshape/permute graph already lowers to the final scattered scale store inside
the same kernel. #193599 is useful for dynamic padding and auxiliary fill, not
for remapping the packed-data source across the parent X/R boundary.

## Rejected graph forms

- Pair conversion after reducing to child resolution: exact, but three kernels
  and no nested reduction. It needs a new two-parent-lane projection relation.
- Direct `pack=4` triplet HOP: fast and experimentally exact, but invalid as a
  production design because it depends on unspecified HOP grouping.
- Flattened AITER field order before conversion: the producer frame is
  `(rows * groups, 32)` while the pack reads `(rows, groups * 8)`. Current
  planning correctly declines because the X/R boundary is inside one raw axis.
  Supporting this requires a new split-frame proof plus post-reorder fusion
  reconsideration.
- Preserving `[rows, groups, 32]` after the field permutation avoids that new
  planner capability and is both simpler and faster.

## Artifacts

- `/data/users/eellison/pytorch/agent_space/bench_aiter_vs_inductor_grouped_native_subagent.py`
- `/data/users/eellison/pytorch/agent_space/validate_native_pack2_runtime.py`
- `/data/users/eellison/pytorch/agent_space/check_native_pack2_tail.py`
- `/data/users/eellison/pytorch/agent_space/bench_dcn_scale_only_native_pack2.py`
- `/data/users/eellison/pytorch/agent_space/probe_dcn_native_pack2_tail.py`
- `/data/users/eellison/pytorch/agent_space/mxfp6_aiter_field_order_diagnosis.md`
