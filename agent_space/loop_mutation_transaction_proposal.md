# Loop Mutation Transaction Proposal

## Problem

Some fusion checks speculatively mutate scheduler loop state before the final
fusion decision is known. Today this can happen through:

- `SchedulerNode.apply_new_loop_order()` via loop reordering
- `SchedulerNode.apply_loop_reindexing()` via pointwise/reduction reindexing
- `FusedSchedulerNode.group` reassignment after child nodes are reindexed

The caller (`Scheduler.can_fuse`) owns the full fusion decision, but the callee
knows when mutation actually happens. Threading a flag like
`rollback_if_vertical_fusion_fails` into the transform helper is awkward because
it makes a local transform helper know about vertical fusion policy, and it only
checks part of the final decision.

## Proposed API

Add a small transaction context around speculative fusion transforms:

```python
with self.capture_loop_mutations(node1, node2) as tx:
    # Existing speculative loop-ordering / reindexing code runs normally.

    if full_fusion_decision_succeeds:
        tx.commit()
        return True

return False  # context restores if not committed
```

The context records loop mutations lazily and restores them automatically unless
the caller commits.

## SchedulerNode Hook

`SchedulerNode` owns the actual mutable loop state, so it should expose a hook
called immediately before loop state mutation:

```python
class SchedulerNode(BaseSchedulerNode):
    _loop_mutation_listener: Callable[[SchedulerNode], None] | None = None

    def _before_loop_state_mutation(self) -> None:
        if self._loop_mutation_listener is not None:
            self._loop_mutation_listener(self)

    def apply_new_loop_order(self, new_order):
        self._before_loop_state_mutation()
        ...

    def apply_loop_reindexing(self, new_iter_sizes):
        self._before_loop_state_mutation()
        ...
```

The listener snapshots the node only once, immediately before the first mutation.

## Transaction State

The transaction needs to track two layers:

- Child `SchedulerNode` state, captured lazily by the mutation listener.
- `FusedSchedulerNode.group`, captured eagerly for fused nodes participating in
  the fusion attempt.

The fused-node group is separate from child loop state. Reindexing children can
also do:

```python
pw_node.group = snodes[0].group
refresh_group_node_dependencies(pw_node)
```

A child-node listener does not see that direct assignment, so the transaction
must save fused groups separately.

Sketch:

```python
@dataclasses.dataclass
class _LoopMutationTransaction:
    scheduler_node_states: dict[SchedulerNode, tuple[Any, ...]]
    fused_node_groups: dict[FusedSchedulerNode, Any]
    committed: bool = False

    def track(self, sn: SchedulerNode) -> None:
        if sn not in self.scheduler_node_states:
            self.scheduler_node_states[sn] = sn.snapshot_loop_state()

    def commit(self) -> None:
        self.committed = True
```

Context manager:

```python
@contextlib.contextmanager
def capture_loop_mutations(self, *nodes: BaseSchedulerNode):
    tx = _LoopMutationTransaction(
        scheduler_node_states={},
        fused_node_groups={
            node: node.group
            for node in nodes
            if isinstance(node, FusedSchedulerNode)
        },
    )

    old_listeners = {}
    for node in nodes:
        for sn in node.get_nodes():
            if isinstance(sn, SchedulerNode):
                old_listeners[sn] = sn._loop_mutation_listener
                sn._loop_mutation_listener = tx.track

    try:
        yield tx
    finally:
        for sn, old in old_listeners.items():
            sn._loop_mutation_listener = old

        if not tx.committed:
            for sn, state in tx.scheduler_node_states.items():
                sn.restore_loop_state(state)
            for node, group in tx.fused_node_groups.items():
                node.group = group
                refresh_group_node_dependencies(node)
```

## Where To Use It

Use the transaction around the speculative portion of `Scheduler.can_fuse` that
may call:

- `shared_data_after_reordering_loop()`
- `_try_reindex_pointwise_for_reduction()`

Commit only after the full fusion decision passes, including:

- `V.choices.can_fuse(...)`
- `can_fuse_vertical(...)` or horizontal checks
- `V.choices.can_fuse_vertical(...)`
- backend `can_fuse_vertical(...)`

## Why This Is Better

- No fusion-policy flag threaded into a local transform helper.
- No return-token plumbing through every transform helper.
- Snapshot happens only if mutation actually occurs.
- Loop reordering and reindexing use the same rollback mechanism.
- The mutation hook is placed next to the mutating methods, so future loop
  transforms automatically participate.

## Open Questions

- Should the transaction cover all of `Scheduler.can_fuse`, or only the section
  after `shared_data_score` is below threshold and loop transforms are attempted?
- Should direct `FusedSchedulerNode.group` assignment eventually become a setter
  that also notifies the transaction, instead of eager group capture?
- Are there other loop-state mutators (`merge_loops`, `swap_pw_red_dimension`)
  that should opt into the same hook if used speculatively?
