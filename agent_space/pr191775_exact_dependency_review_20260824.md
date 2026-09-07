# Review: #191775 exact dependency-plan redesign

> Historical monolithic review. The active implementation is split into the
> prerequisite and MXFP6 worktrees documented in
> [pr191775_split_step_by_step.md](/data/users/eellison/pytorch/agent_space/pr191775_split_step_by_step.md).

Date: 2026-08-24. Verdict: **approve**. Target verified: worktree
`agent_space/stack_rebase_20260824/wt` at `b225e3b078b`, uncommitted diff
byte-matches `pr191775_exact_dependency_plan.patch` (sha `82df89c0e65e...`,
2 files, +400/-226 net rework of scheduler.py + tests).

## Design fidelity

The patch implements the agreed contract exactly: strict matching first;
`_plan_fusion_dependency_matches` walks only `consumer.unmet_dependencies`;
every residual producer-output MemoryDep must pass the inherited normalized
equivalence proof or be an exact planner-proved `MemoryDepMatch`; the first
uncovered residual declines the candidate; Star/WeakDep stay on ordinary
vertical legality. The name-level escape hatch (`index_proven_names`,
`index_equivalent_dep_names`, `_producer_output_names_read_by_consumer`, the
name bypass inside `can_fuse_vertical`) is gone.

## What it resolves by construction (beyond the stated goal)

1. **Round-2 stale-context routing finding**: `FusedNestedReductions
   .can_fuse_with` now derives pointwise domains from `_plan_append`'s own
   plan instead of a separate stale classification -- the exact fix my
   earlier review suggested, plus admission/construction symmetry is
   preserved (fuse_with replans).
2. **Round-2 score-bridge TODO**: scoring and legality both consume the same
   exact `MemoryDepMatch` set; the score bridge can no longer diverge from
   what legality would accept.
3. **Loop-rewrite invalidation class**: `has_planned_matches` suppresses
   expand-dimension, reorder, and index-inversion rewrites when planned
   matches exist -- planned pairs describe current loop indexing, so
   rewriting after planning would silently invalidate them. This also
   removes the nested path's exposure to the rejected-probe reorder
   staleness class.

## Load-bearing guards preserved

The three guards my round-2 review verified as load-bearing (sync-mode
rejection, dense/injective producer writes, TMP-symbol exclusion, contiguous
normalized write) are factored into `_memory_dep_supports_index_equivalence`
and applied to BOTH proof paths (inherited equivalence and planned match).
Producer-write uniqueness is enforced (`len(writes) != 1` declines).

## Record unification

`ProjectedSourceAccess`/`ProjectedConsumerAccess` constructed at fusion time
are exactly the indexed-forwarding vocabulary (sources, consumers with
proved lane, layout), with `__post_init__` enforcing the single-buffer-name
invariant -- which also structurally validates the store_buffer_names-derived
must_forward rule in the forwarding follow-up. The compatibility
`source_layouts` property keeps current codegen unchanged; the scope
boundary (fusion permission now, codegen forwarding later) is clean. The
follow-up's one obligation, normalizing planned and emitted accesses, is
recorded in the redesign doc.

## Notes (non-blocking)

- Conservative narrowing: in the append path, a multi-write producer buffer
  now declines fusion outright where the old flow could still fuse via
  normal matching. Fail-safe and likely unreachable for approved plans;
  worth a one-line comment at the `len(writes) != 1` check.
- The generic-path fallback is unchanged and safe: a None match plan there
  degrades to strict-only matching, not a hard decline.

## Verification (mine, closing the focused-`-k` gap)

The doc's own verification ran focused subsets only -- the pattern that
masked the replay P0 -- so I ran the FULL suites differentially. The
full-package overlay (run_wt.py) is IMPOSSIBLE at this trunk distance: the
installed editable torch (git7c78410) predates `_c10d_functional
.wait_tensors`, `aten._philox_randint`, and the generated-kernel launch
interface; shimming the ops still leaves 377 launch-level failures on the
PRISTINE base. Instead: `agent_space/run_narrow_plus.py` (the narrow module
preload + `codegen/triton_utils`), sound for this patch because the diff
touches only overlaid modules and the base-vs-patched differential cancels
harness effects:

```text
nested suite   base: OK (7 skipped)     patched: 387 ran, OK (7 skipped)
scheduler      base: 2 errors (env)     patched: SAME 2 errors (flop_counter,
                                        identical on pristine base), +8 new
                                        tests incl. 26 dependency_matches
                                        passing
```

Zero regressions attributable to the patch.

## Operational flag for the author

Any future local verification of the resubmitted stack needs the installed
editable torch rebuilt onto (or past) the new base `e812f6fc74c`; until
then, full-package overlay testing is unavailable and only the narrow
differential protocol above is trustworthy. `run_wt_shimmed.py` (op shims)
is insufficient -- the skew is at the kernel-launch interface, not just op
registration.

## Checklist

`pr191775_step_by_step_checklist.md` is updated and usable: correct
worktree, both diff commands (local delta and full PR), line-pinned links
to `MemoryDepMatch` / `_plan_fusion_dependency_matches`, 25 steps.
