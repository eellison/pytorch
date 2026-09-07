# Rollback Design Alternatives

## Problem

`_try_reindex_pointwise_for_reduction` mutates node loop state before the
caller knows whether vertical fusion will succeed. Today this is handled by a
`rollback_if_vertical_fusion_fails` flag threaded into the transform helper,
which makes the helper know about vertical fusion policy and only checks part
of the final decision.

## Option A: Return-snapshot (current dirty diff)

Change `_try_reindex_pointwise_for_reduction` from returning `bool` to
returning a snapshot-or-None. The caller owns commit/rollback:

```python
def _try_reindex_pointwise_for_reduction(self, node1, node2)
    -> snapshot | None:
    ...
    snapshot = self._snapshot_loop_state((pw_node,))
    # apply reindexing
    if not has_benefit:
        self._restore_loop_state(snapshot)
        return None
    return snapshot
```

Caller in `can_fuse`:

```python
snapshot = self._try_reindex_pointwise_for_reduction(node1, node2)
if snapshot is None:
    return False
if self.can_fuse_vertical(node1, node2) and ...:
    return True          # snapshot dropped = committed
self._restore_loop_state(snapshot)
return False
```

Extract `_snapshot_loop_state` / `_restore_loop_state` as static methods that
handle both child `SchedulerNode` state and `FusedSchedulerNode.group`.

**Pros**: Minimal change. No new abstraction. Caller sees exactly when
rollback happens.

**Cons**: Every call site that may need rollback must remember to call
`_restore_loop_state`. If `shared_data_after_reordering_loop` also needs
rollback in the future, each caller must add the same pattern. The snapshot
return type is a verbose tuple.

## Option B: Transaction context manager (proposal doc)

See `agent_space/loop_mutation_transaction_proposal.md` for full details.

Wrap the speculative section in a context manager that auto-restores unless
committed:

```python
with self.capture_loop_mutations(node1, node2) as tx:
    # existing reindexing / reordering code runs normally
    if full_fusion_decision_succeeds:
        tx.commit()
        return True
return False  # context restores automatically
```

Uses a mutation listener on `SchedulerNode` so snapshots are captured lazily
on first mutation. `FusedSchedulerNode.group` is captured eagerly.

**Pros**: Caller cannot forget rollback — it's automatic. One transaction
wraps both `_try_reindex_pointwise_for_reduction` and
`shared_data_after_reordering_loop`. Future loop transforms (loop reordering,
merge_loops) automatically participate via the mutation hook. No fusion-policy
flag in transform helpers.

**Cons**: More machinery (listener, dataclass, context manager). Slightly
harder to debug since rollback is implicit. Mutation listener adds a field to
every `SchedulerNode`.

## Recommendation

Option A is simpler and sufficient for the current single call site. Option B
becomes worthwhile if `shared_data_after_reordering_loop` also needs rollback,
or if more speculative transforms are added — at that point the pattern of
manually threading snapshots through every helper would get unwieldy.

For the landing stack: Option A. It removes the `rollback_if_vertical_fusion_fails`
flag cleanly with minimal new code. Option B can be a follow-up if the
pattern spreads.
