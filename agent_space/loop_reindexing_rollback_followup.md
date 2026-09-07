## Loop Reindexing Rollback Follow-up

During review of the rollback prep commit, we removed the new vertical fusion
retry path from `Scheduler.can_fuse()`.

Removed behavior:

- If `can_fuse_vertical(node1, node2)` failed, the branch tried
  `_try_reindex_pointwise_for_reduction(node1, node2)` when `can_reorder=True`
  and `config.loop_reindexing_after_fusion=True`.
- It then recomputed `shared_data_score` and retried vertical fusion.

Reason for removal:

- This is not just rollback infrastructure; it introduces a new fusion attempt.
- The prep commit should only make existing speculative loop reordering /
  reindexing safe to roll back when fusion later rejects.
- If a later nested-reduction commit needs recursive
  `can_fuse(..., can_reorder=True)` to perform this retry, add it there with
  tests that show the required fusion and rollback behavior.

Current prep-commit scope:

- `can_fuse()` installs a mutation tracker.
- Existing scoring/reordering paths may mutate loop state.
- If the final fusion decision returns false, the tracker restores the original
  loop state.
