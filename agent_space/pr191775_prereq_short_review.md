# #191775 prerequisite: short review path

Worktree:
`/data/users/eellison/pytorch/agent_space/pr191775_prereq_split_wt`

This path assumes `ProjectedSourceAccess` and
`_try_get_sub_parent_source_projections` were already reviewed.

## 1. Broadcast projection contract

Read:

- [`_sub_parent_consumer_is_trailing_broadcast`](/data/users/eellison/pytorch/agent_space/pr191775_prereq_split_wt/torch/_inductor/scheduler.py:1014)
- [`_sub_parent_broadcast_projections`](/data/users/eellison/pytorch/agent_space/pr191775_prereq_split_wt/torch/_inductor/scheduler.py:1061)
- [Nested call-site source classification](/data/users/eellison/pytorch/agent_space/pr191775_prereq_split_wt/torch/_inductor/scheduler.py:1343)

Check that retained source axes form an equal prefix, retained extra axes do
not affect the address, and only `MemoryDep` reads become projection records.
Normalized dependencies intentionally use flat row-major address identity.

## 2. Fusion flow

Read:

- [`FusedNestedReductions._plan_fusion_with`](/data/users/eellison/pytorch/agent_space/pr191775_prereq_split_wt/torch/_inductor/scheduler.py:3892)
- [Staged recognition and proof in `_can_fuse`](/data/users/eellison/pytorch/agent_space/pr191775_prereq_split_wt/torch/_inductor/scheduler.py:9456)
- [Ordinary versus staged scoring](/data/users/eellison/pytorch/agent_space/pr191775_prereq_split_wt/torch/_inductor/scheduler.py:9498)

There is no plan argument on `_can_fuse`. Existing nested nodes derive their
prospective append plan locally. Initial nested and standalone sub-parent forms
are recognized later in the same method. A recognizable candidate whose plan
fails is rejected rather than falling through to ordinary fusion.

## 3. Proof consumption

Read:

- [`_can_fuse_vertical_impl`](/data/users/eellison/pytorch/agent_space/pr191775_prereq_split_wt/torch/_inductor/scheduler.py:9604)
- [`_score_staged_fusion_memory_for_can_fuse`](/data/users/eellison/pytorch/agent_space/pr191775_prereq_split_wt/torch/_inductor/scheduler.py:9970)

The ordinary `can_fuse_vertical` and scoring APIs are unchanged. The staged
branch alone passes the exact match tuple to private implementation code.
Vertical legality still checks every unmatched dependency, and scoring counts
each producer write at most once.

## Focused tests

- [Broadcast frame and normalization contract](/data/users/eellison/pytorch/agent_space/pr191775_prereq_split_wt/test/inductor/test_inductor_scheduler.py:379)
- [Empty staged plan suppresses loop rewrites](/data/users/eellison/pytorch/agent_space/pr191775_prereq_split_wt/test/inductor/test_inductor_scheduler.py:564)
- [Nested outer-reduction broadcast](/data/users/eellison/pytorch/agent_space/pr191775_prereq_split_wt/test/inductor/test_nested_reduction.py:1266)
- [Pointer-independent swizzle store check](/data/users/eellison/pytorch/agent_space/pr191775_prereq_split_wt/test/inductor/test_nested_reduction.py:2892)
