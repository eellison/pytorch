# #191775 split review order

The original #191775 mixed two changes: replacing name-level fusion permission
with exact dependency records, and extending sub-parent codegen to MXFP6's 4:3
packing. Review the two uncommitted worktrees in this order.

Standalone guides:

- [Exact staged-dependency prerequisite](/data/users/eellison/pytorch/agent_space/pr191775_prereq_step_by_step.md)
- [MXFP6 4:3 layer](/data/users/eellison/pytorch/agent_space/pr191775_mxfp6_split_step_by_step.md)

## Part 1: exact staged dependencies (high scrutiny)

Worktree: `/data/users/eellison/pytorch/agent_space/pr191775_prereq_split_wt`

1. Read the projection record and stage compatibility views:
   [scheduler.py](/data/users/eellison/pytorch/agent_space/pr191775_prereq_split_wt/torch/_inductor/scheduler.py:1770).
   The plan retains raw producer and consumer `MemoryDep`s. Current codegen
   still consumes the derived name/layout views.
2. Review the existing interleaved proof now recording exact accesses:
   [scheduler.py](/data/users/eellison/pytorch/agent_space/pr191775_prereq_split_wt/torch/_inductor/scheduler.py:874).
   Fusion legality separately validates the raw `X | R` boundary for
   multi-axis accesses, while permitting one-active-axis broadcast reads.
3. **Highest scrutiny:** review the reduced-value broadcast proof:
   [scheduler.py](/data/users/eellison/pytorch/agent_space/pr191775_prereq_split_wt/torch/_inductor/scheduler.py:1061).
   It compares the retained source prefix and address mapping. Dependency
   extraction removes unused trailing broadcast axes; normalized dependencies
   may also merge equivalent row-major axes.
4. **Highest scrutiny:** review residual fusion coverage:
   [scheduler.py](/data/users/eellison/pytorch/agent_space/pr191775_prereq_split_wt/torch/_inductor/scheduler.py:9021).
   Strict matches pass normally. Non-exact sub-parent reads must be exact plan
   pairs. Only grouped-stage reads retain the inherited nested equivalence path.
5. Review branch-local staged recognition:
   [scheduler.py](/data/users/eellison/pytorch/agent_space/pr191775_prereq_split_wt/torch/_inductor/scheduler.py:9216).
   An unrelated pair follows ordinary fusion, a valid pair retains its
   `StagedReductionPlan`, and a recognizable but invalid candidate rejects
   before ordinary rewrites can run.
6. Continue through vertical legality and rewrite suppression in the same
   method. The staged branch calls explicitly named scoring and vertical
   helpers; ordinary APIs carry no staged-plan parameters. Plan presence, not
   match-tuple truthiness, freezes optional loop rewrites.
7. Review the scoring bridge:
   [scheduler.py](/data/users/eellison/pytorch/agent_space/pr191775_prereq_split_wt/torch/_inductor/scheduler.py:9970).
   It scores normalized or planned reuse once per producer write and avoids
   double-counting a raw exact dependency.
8. Review append replanning in `FusedNestedReductions._plan_fusion_with` and
   `fuse_with`; `_can_fuse` detects the nested node and obtains the plan locally:
   [scheduler.py](/data/users/eellison/pytorch/agent_space/pr191775_prereq_split_wt/torch/_inductor/scheduler.py:3892).
9. Review focused scheduler regressions:
   [test_inductor_scheduler.py](/data/users/eellison/pytorch/agent_space/pr191775_prereq_split_wt/test/inductor/test_inductor_scheduler.py:379).
10. Review the shifted/transposed end-to-end regressions:
    [test_nested_reduction.py](/data/users/eellison/pytorch/agent_space/pr191775_prereq_split_wt/test/inductor/test_nested_reduction.py:796).

## Part 2: MXFP6 4:3 extension

Worktree: `/data/users/eellison/pytorch/agent_space/pr191775_layered_split_wt`

1. Review supported rates and ordered output grouping:
   [scheduler.py](/data/users/eellison/pytorch/agent_space/pr191775_layered_split_wt/torch/_inductor/scheduler.py:578),
   [scheduler.py](/data/users/eellison/pytorch/agent_space/pr191775_layered_split_wt/torch/_inductor/scheduler.py:932).
2. Review nested rate validation against the grouped coordinate boundary:
   [scheduler.py](/data/users/eellison/pytorch/agent_space/pr191775_layered_split_wt/torch/_inductor/scheduler.py:987).
3. **Highest scrutiny:** review internal sub-parent forwarding:
   [scheduler.py](/data/users/eellison/pytorch/agent_space/pr191775_layered_split_wt/torch/_inductor/scheduler.py:1285).
   Same output group permits reshape-equivalent row-major accesses; a later
   output group must preserve the source frame and add only unused trailing axes.
4. Review parent-chain deferral for looped reductions:
   [scheduler.py](/data/users/eellison/pytorch/agent_space/pr191775_layered_split_wt/torch/_inductor/scheduler.py:785).
5. Review the stage representation and invariants:
   [scheduler.py](/data/users/eellison/pytorch/agent_space/pr191775_layered_split_wt/torch/_inductor/scheduler.py:2156).
6. Review current name-based codegen adaptation:
   [simd.py](/data/users/eellison/pytorch/agent_space/pr191775_layered_split_wt/torch/_inductor/codegen/simd.py:2358).
7. Review ordered output-group emission and deferred parent scheduling:
   [simd.py](/data/users/eellison/pytorch/agent_space/pr191775_layered_split_wt/torch/_inductor/codegen/simd.py:3418),
   [simd.py](/data/users/eellison/pytorch/agent_space/pr191775_layered_split_wt/torch/_inductor/codegen/simd.py:3717).
8. Review reshape/split emission:
   [triton.py](/data/users/eellison/pytorch/agent_space/pr191775_layered_split_wt/torch/_inductor/codegen/triton.py:6325).
9. **Highest scrutiny:** review the narrow exact parent-stage append rule and
   its exact-vs-shifted scale-swizzle regression:
   [scheduler.py](/data/users/eellison/pytorch/agent_space/pr191775_layered_split_wt/torch/_inductor/scheduler.py:9858),
   [test_nested_reduction.py](/data/users/eellison/pytorch/agent_space/pr191775_layered_split_wt/test/inductor/test_nested_reduction.py:2012).
10. Review MXFP6 functional cases, especially preshuffled and rejection paths:
    [test_nested_reduction.py](/data/users/eellison/pytorch/agent_space/pr191775_layered_split_wt/test/inductor/test_nested_reduction.py:1804).
11. Review kernel-form checks:
    [test_nested_reduction.py](/data/users/eellison/pytorch/agent_space/pr191775_layered_split_wt/test/inductor/test_nested_reduction.py:3498).

## Size comparison

- Current pushed #191775 scheduler: `+727/-262`.
- Prerequisite scheduler: `+594/-280`; full prerequisite diff: `+1029/-394`.
- MXFP6 layer relative to prerequisite: `+1517/-222` total, split into
  `+669/-205` production code and `+848/-17` tests.
- MXFP6 scheduler portion: `+487/-127`.
- Cumulative prerequisite plus MXFP6 worktree: `+2489/-559`.

The cumulative stack is larger because the correctness contract and its tests
are now explicit. The MXFP6 review itself is substantially smaller and no
longer introduces the generic fusion-legality mechanism at the same time.
