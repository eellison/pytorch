# #191775 exact dependency-plan redesign

## Review target

- PR: https://github.com/pytorch/pytorch/pull/191775
- Submitted orig commit: `b225e3b078b`
- Worktree: `/data/users/eellison/pytorch/agent_space/stack_rebase_20260824/wt`
- Local delta: [pr191775_exact_dependency_plan.patch](/data/users/eellison/pytorch/agent_space/pr191775_exact_dependency_plan.patch)
- Patch SHA256: `6e08c00459e55efdaf286fa64ea53311e960c32dc484aca7323d0942602922e9`

The delta is intentionally uncommitted. Review it with:

```bash
cd /data/users/eellison/pytorch/agent_space/stack_rebase_20260824/wt
git diff
```

## Problem

The submitted PR lets a staged reduction relax vertical fusion by buffer name.
The planner proves that particular consumer reads are valid lane projections,
but the scheduler reduces that proof to an allowlist of names. A different read
of the same buffer can therefore inherit permission that was proved for another
index.

## New contract

Normal strict `MemoryDep` matching remains unchanged and runs first. When it
leaves a consumer read of a producer output unmatched, the staged planner must
account for that exact read/write pair:

```text
producer write MemoryDep
    -> consumer read MemoryDep
    -> projected layout and lane
```

`_plan_fusion_dependency_matches` walks only `consumer.unmet_dependencies`.
Each residual producer-output `MemoryDep` must pass either:

1. the inherited normalized reshape/broadcast equivalence proof; or
2. an exact `ProjectedSourceAccess` relation built by the staged plan.

The first uncovered residual rejects the candidate. `StarDep` and `WeakDep`
remain on ordinary vertical legality. Producer writes must remain unique,
dense, injective, synchronization-free, and free of TMP symbols.

## Plan lifetime

Nothing is cached across mutable compiler phases:

1. `can_fuse` builds the staged plan and exact dependency matches.
2. Any planner-required consumer reindex happens before those matches are made.
3. Nonempty exact matches suppress later optional generic loop rewrites that
   would invalidate them.
4. Nested append replans after legality.
5. `fuse_with` replans again.
6. Codegen rebuilds the final plan after fusion and `merge_loops`.

This follows the same rebuild-and-assert lifecycle as #190594.

## Shared representation and scope boundary

The plan now owns `ProjectedSourceAccess` records containing source accesses,
consumer access plus lane, and the projected layout. Fusion derives exact
`MemoryDepMatch` edges from those records.

Current #191775 codegen still consumes the compatibility `(buffer_name,
layout)` view. The indexed-forwarding follow-up will normalize planned and
emitted accesses and consume `ProjectedSourceAccess` directly. That boundary is
intentional: this change removes the unsafe fusion permission without pulling
the separate codegen-forwarding mechanism into #191775.

## Historical compatibility

The redesign was checked against the original landed scheduler-equivalence
change, commit `546faa04929` / #183432. Its guarantees remain covered:

- quotient and pure consumer broadcasts are accepted;
- producer broadcasts and aliasing writes are rejected;
- strict matching remains the default;
- projected dependencies receive an admission score;
- multiple projected reads of one producer write score that write once.

## Highest-scrutiny locations

- [projection proof](/data/users/eellison/pytorch/agent_space/stack_rebase_20260824/wt/torch/_inductor/scheduler.py:1024)
- [projection record](/data/users/eellison/pytorch/agent_space/stack_rebase_20260824/wt/torch/_inductor/scheduler.py:1904)
- [residual dependency planner](/data/users/eellison/pytorch/agent_space/stack_rebase_20260824/wt/torch/_inductor/scheduler.py:9160)
- [nested candidate integration](/data/users/eellison/pytorch/agent_space/stack_rebase_20260824/wt/torch/_inductor/scheduler.py:9210)
- [append legality and replanning](/data/users/eellison/pytorch/agent_space/stack_rebase_20260824/wt/torch/_inductor/scheduler.py:9359)
- [vertical legality](/data/users/eellison/pytorch/agent_space/stack_rebase_20260824/wt/torch/_inductor/scheduler.py:9772)
- [score bridge](/data/users/eellison/pytorch/agent_space/stack_rebase_20260824/wt/torch/_inductor/scheduler.py:10084)

## Verification

Completed on B200 with the worktree preload harness:

```text
test_inductor_scheduler.py -k dependency_matches
  26 passed

test_inductor_scheduler.py -k fusable_read_and_write
  4 passed

test_inductor_scheduler.py -k nested_reduction_append_replans
  2 passed

test_nested_reduction.py -k mxfp6
  30 passed, 3 skipped

test_nested_reduction.py -k indirect_index_mask
  6 passed

test_nested_reduction.py -k shifted
  16 passed

test_nested_reduction.py -k parent_full_source
  4 passed

test_nested_reduction.py -k internal_source
  6 passed, 2 skipped
```

`py_compile` and `git diff --check` pass. `spin quicklint` is currently blocked
before lint execution by the historical checkout's type-stub regeneration using
the newer installed `torchgen` (`_philox_randint_` tag mismatch); it did not
modify tracked files.

An adversarial review found no remaining design or correctness blocker. Its one
forward-looking constraint is that the indexed-forwarding PR must normalize
both planned and emitted accesses when it replaces the compatibility name view.
