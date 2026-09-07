# Sub-parent epilogue stack - start here

Locally rebuilt and verified on 2026-08-19, then resubmitted on 2026-08-21.
#190594 has landed and the four open upper PRs retain their ghstack identities.
Reviewed pre-amend patches are saved under
`agent_space/stack_review_fixes_20260818/`.

The #190595 review worktree is currently stopped at `359b11d5aa1` with
uncommitted review cleanups, including removal of `_GroupInvariantBroadcast`
and its swizzled-scale projection tests. Isolated #190595 now uses the simpler
eager group-to-lane broadcast path.

The removed optimization is being rebuilt as two uncommitted follow-ups after
#191775 and before #190596:

1. exact indexed projected-value forwarding:
   [README](/data/users/eellison/pytorch/agent_space/indexed_projection_work/README.md),
   [test results](/data/users/eellison/pytorch/agent_space/indexed_projection_work/TEST_RESULTS.md);
2. lazy derived-domain projection on top of that foundation:
   [README](/data/users/eellison/pytorch/agent_space/domain_projection_work/README.md),
   [test results](/data/users/eellison/pytorch/agent_space/domain_projection_work/TEST_RESULTS.md),
   [architecture review](/data/users/eellison/pytorch/agent_space/domain_projection_work/ARCH_REVIEW.md).

Neither follow-up has a commit or PR number yet. The second restores one scale
conversion in every supported factor-2 form and, for persistent factor 2, the
parent-divide-before-split form without retaining the name-keyed
`_GroupInvariantBroadcast` mechanism in #190595.

Canonical uncommitted artifacts:

- indexed forwarding:
  `domain_projection_work/foundation_after_no_group_invariant.patch`
  (`4586297d758bc6be27b0aa585bb6210457d712017953c3813d6ba43f352d5c90`);
- lazy projection:
  `domain_projection_work/projection_after_no_group_invariant.patch`
  (`69e9fe36c850d225997a2569e6173aa44116b9420699b9cc417f24e4accd6164`);
- cumulative replay through #190596/#191974:
  `domain_projection_work/upper_stack_replay_190596_191974.patch`
  (`683df48f1ddd5b90e648a5284a19d937a897128ed8e6bd0af6184acf93bbc49a`).

The intended order is therefore #190594, #190595, #191775, indexed forwarding,
lazy projection, #190596, then #191974. The current stack table below lists
only commits and PRs that already exist.

## Current stack

| # | commit | PR | full guide path |
|---|---|---|---|
| 1 | `86038d6e1cd` | [#190594 standalone interleaved](https://github.com/pytorch/pytorch/pull/190594) | [`/data/users/eellison/pytorch/agent_space/pr190594_interleaved_subparent.md`](/data/users/eellison/pytorch/agent_space/pr190594_interleaved_subparent.md) |
| 2 | `359b11d5aa1` + uncommitted review/removal changes | [#190595 nested interleaved](https://github.com/pytorch/pytorch/pull/190595) | [`/data/users/eellison/pytorch/agent_space/pr190595_nested_interleaved.md`](/data/users/eellison/pytorch/agent_space/pr190595_nested_interleaved.md) |
| 3 | `8dc79f8803b` | [#191775 MXFP6 4:3 packing](https://github.com/pytorch/pytorch/pull/191775) | [`/data/users/eellison/pytorch/agent_space/pr191775_mxfp6_staged_packing.md`](/data/users/eellison/pytorch/agent_space/pr191775_mxfp6_staged_packing.md) |
| 4 | `1e2b8890c98` | [#190596 contiguous](https://github.com/pytorch/pytorch/pull/190596) | [`/data/users/eellison/pytorch/agent_space/pr190596_contiguous_subparent.md`](/data/users/eellison/pytorch/agent_space/pr190596_contiguous_subparent.md) |
| 5 | `3f62ae27baa` | [#191974 test hygiene](https://github.com/pytorch/pytorch/pull/191974) | [`/data/users/eellison/pytorch/agent_space/pr191974_test_hygiene.md`](/data/users/eellison/pytorch/agent_space/pr191974_test_hygiene.md) |

The old `a38c9cac204` retained-plan commit is not in this stack. Its useful
pieces were folded into #190594 and #190595; its guide is historical.
#191696's mempool fix is folded into #190595, and #191975's cleanup is folded
into its owner commits; both superseded PRs are closed.

## Verification

The counts below describe the last submitted five-PR tip. The current
uncommitted #190595 review/removal diff and the two uncommitted follow-ups have
their own test records linked above.

Current isolated #190595 review-tree verification is 296 passed / 1 skipped in
`test_nested_reduction.py` and 60 passed / 6 skipped in
`test_inductor_scheduler.py`; `git diff --check`, `python -m py_compile`, and
`with-proxy spin quicklint` are clean.

For the two follow-ups, the isolated projection functional classes pass 334
tests with 6 skips; the full upper replay passes 447 tests with 13 skips and
the two known preload-only `kernel_num_gb` cases deselected. The heuristic
suite passes 15 tests with 2 skips, exact-access scheduler coverage passes 20
focused tests, and the final upper replay passes quicklint and diff checks.
On B200 without coordinate descent, swizzled NVFP4 is at parity or faster than
FlashInfer for `D=4096,6144,7168,8192`; `D=4352,4608` remain within 9.9% and
8.4%. Every measured Inductor form is one fused kernel with one FP8 scale
conversion.

At the final tip:

```text
python test/inductor/test_nested_reduction.py
  450 passed, 12 skipped

python test/inductor/test_inductor_scheduler.py
  88 passed, 6 skipped
```

The local tip passed the full nested-reduction and scheduler files,
`with-proxy lintrunner -a`, and `git diff --check`. #190595 and #191775 were
also tested independently after the replay. Adversarial reviews found and pinned a
multi-reduction pass-boundary bug in #191775 and a multi-kernel persistence
choice bug in #190596 before the final replay.

Additional tests reject shifted nested and standalone reads instead of
forwarding by name, retain ordinary indexed loads for external inputs, and
exercise mixed source layouts plus factor-8/16 CONTIGUOUS lanes. They also
cover a planned source mutated later in the graph, a rolled grouped-reduction
output, a three-pass looped internal-source schedule, and stable staged-kernel
selection under `triton.multi_kernel`. A symbolic CONTIGUOUS reduction extent
runs multiple real runtime sizes through one generic fallback graph. Kernel
bandwidth/FLOP metadata and TMA prescan now include every emitted stage.

## Architecture

`FusedStagedReduction` is the stable scheduler identity for a reduction group
that generic SIMD scheduling cannot emit. `FusedNestedReductions` is its nested
topology subclass. A `StagedReductionPlan` describes the final emission order:

```text
parent nodes
optional NestedReductionStage
zero or one SubParentEpilogueStage
```

`SubParentEpilogueStage.output_groups` groups nodes by the number of output
lanes emitted per input lane group. Factor-2 NVFP4/MXFP4 uses one output lane;
MXFP6 uses factor 4 with three output lanes.

Fusion builds an approval plan to decide legality and node identity. Codegen
builds the final plan after `merge_loops`, so it uses the actual post-fusion
ranges. This is deliberate recomputation, not a plan cache. Fusion planning
always enforces the complete stage legality contract; the feature is
default-off and no compile-time profile justifies caching that analysis.

Plans retain the temporal buffer names already recorded in each node's
`read_writes`; applying the scheduler-global final mutation map can collapse an
earlier source onto a later in-place version. Separately, a grouped-output name
never proves index equivalence: ordinary derived-domain reads must pass the
normalized reshape/broadcast matcher before dependency matching is relaxed.

## Supported forms

| form | layout/rate | dynamic R | looped | persistent |
|---|---|---:|---:|---:|
| standalone NVFP4/MXFP4 | interleaved 2:1 | yes, when divisibility is proved | yes | yes |
| nested RMSNorm -> NVFP4/MXFP4 | interleaved 2..4:1 | factor-2 covered | yes | yes |
| chunk/SwiGLU | contiguous 2..16:1 | fallback | external sources: yes; internal: rejected | yes |
| MXFP6, standalone and nested | interleaved 4:3 | batch dynamic | yes | yes |

Interleaved planning does not require a power-of-two logical reduction extent.
Tests cover `D=4608` and real runtime dynamic values. The factor still has to
divide the derived lane extent. Contiguous planning remains static and
power-of-two because persistent code splits the padded Triton block; strict
reductions are excluded because they may choose a larger padded block. Looped
external CONTIGUOUS sources reload instead of projecting a partial block.

## Combined dim0+dim1 quantization (successor work, not in this stack)

One kernel producing both rowwise (dim0) and columnwise (dim1) quantized
outputs from a single read of the input. Today this takes 3 kernels reading
x three times; with an RMSNorm producer it is 3 kernels at 267.2 us
(16384x7168, B200).

- Handwritten proof:
  [`nvfp4_combined_dim0_dim1_20260821.md`](/data/users/eellison/pytorch/agent_space/nvfp4_combined_dim0_dim1_20260821.md)
  with kernel `nvfp4_dual_dim_kernel.py`, bench `bench_nvfp4_dual_dim.py`,
  worktree `nvfp4_dual_dim_wt` (stack tip). One kernel, one load, four
  outputs, bitwise-equal to the compiled references and torchao's pure-torch
  reference; 1.8-1.9x vs the best cd-tuned status quo (95.1 vs 169.0 us).
  L1/compute-bound at 52% DRAM, so TMA/persistent pipelines are the wrong
  lever; ~80-90 us is the realistic ceiling for the register-level design.
- Codegen design proposal (draft for author review):
  [`blockwise2d_design_proposal.md`](/data/users/eellison/pytorch/agent_space/blockwise2d_design_proposal.md).
  Two primitives: grouped reductions and lane-split packed stores on a non-R
  axis via a transposed register view. The planner already classifies and
  emits X-grouped stages (`GroupedAxis.X`); dim1 packing is blocked only by
  the grouped-axis guard in `_nested_sub_parent_rate`, and the banded
  producer form is the nested topology with `XBLOCK` a multiple of the row
  group. PR ladder B1-B4; the producer case requires the indexed-forwarding
  foundation below.
- Probe evidence: `probe_mix_order_decline.py` and the writeup addenda.
  Mix-order reduction is structurally inapplicable to group-local dual quant
  (equal, never reversed, flattened groups); adding max/min to its
  reduction-type allowlist is an independent one-line 2.19x win on canonical
  row+column amax pairs and rides separately.

## Reading order

1. Read
   [`/data/users/eellison/pytorch/agent_space/pr190594_interleaved_subparent.md`](/data/users/eellison/pytorch/agent_space/pr190594_interleaved_subparent.md)
   for the fusion-to-codegen diagram and lane proof.
2. Read the guide for the feature commit under review.
3. Read
   [`/data/users/eellison/pytorch/agent_space/FOLLOWUPS.md`](/data/users/eellison/pytorch/agent_space/FOLLOWUPS.md)
   for remaining concerns.

The guides describe both their isolated commit and later changes. Historical
files under `agent_space/` may use old `PARENT_HALF` or retained-plan names and
must not be treated as current unless linked above.

## Submission state

All five local commits preserve their existing `ghstack-source-id` and
`Pull-Request` trailers. The former #191975 cleanup was folded into its owner
commits and is no longer a stack commit. The four open upper PRs have not yet
been updated to these 2026-08-19 local SHAs. The #190594
looped-kernel test now disables the orthogonal split-reduction
transform, and #191775 migrates inherited grouped-axis coverage to
`_nested_sub_parent_rate()`.
