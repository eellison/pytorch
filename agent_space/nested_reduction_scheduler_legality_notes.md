# Nested Reduction Scheduler Legality Notes

## Context

The scheduler commit adds legality for a nested reduction pair:

- `node1`: the outer reduction, for example RMSNorm/layernorm over `D`.
- `node2`: the grouped local reduction, for example `amax` over groups of `G`.
- `FusedNestedReductions`: a staged fused node that owns both reductions, but codegen still treats the outer reduction and grouped reduction as different phases.

That staged structure is the source of most of the awkwardness. Ordinary fusion treats a `FusedSchedulerNode` as one flat fused node. Nested reduction has to preserve stage identity because extra pointwise nodes may belong before the grouped reduction, after the grouped reduction, or at parent-full resolution.

## Pointwise Domains

Nested pointwise nodes can run in three domains:

- `REDUCED`: grouped reduction output, for example `[X, R / G]`.
- `LOCAL_REDUCTION_INPUT`: values before the grouped local reduction, for example `[X, R / G, G]`.
- `PARENT_FULL`: outer reduction parent tile after broadcast-back, for example `[X, R]`.

The current code first classifies pointwise nodes into one of these domains, then separately checks that each node's collapsed ranges match the domain. That separation is useful conceptually, but the plumbing is noisy because the same grouped context is passed through several functions.

I still think this local refactor is worthwhile:

```python
@dataclasses.dataclass(frozen=True)
class PointwiseDomainContext:
    grouped_reduction: SchedulerNode
    grouped_numel: sympy.Expr
    grouped_rnumel: sympy.Expr
    local_reduction_domain: tuple[sympy.Expr, ...]
    parent_full_domain: tuple[sympy.Expr, ...]
```

This is a readability cleanup, not a semantic change. It makes it clearer that classification and compatibility are operating against one shared geometry object.

## Why The Ancestor Issue Exists

`init_group_node()` builds fused-node ancestors as the union of all subnode ancestors:

```python
group_snode.ancestors = union(snode.ancestors for snode in snodes)
```

For ordinary fusion this usually works because later fusion legality is checked against the whole fused node. If `op0 -> op1` are fused and then `op2` is considered, generic fusion checks `can_fuse(fused(op0, op1), op2)`. Reads in `op2` from either `op0` or `op1` can match against the fused node's union of writes.

Nested append fusion is different. `FusedNestedReductions.can_fuse_with(other)` currently validates the append against `self.node2`, the grouped stage, because `fuse_with()` appends `other` into that stage:

```python
new_node2 = backend.fuse(self.node2, other)
return FusedNestedReductions(self.node1, new_node2)
```

That means a parent-full consumer that reads both:

- the grouped stage output, and
- an outer-stage value

is checked as `can_fuse(self.node2, other)`, not `can_fuse(self, other)`. The grouped-stage write can be matched, but the outer-stage read remains as an external dependency.

When the nested fused node still has internal ancestors, the generic vertical dependency check can see a false intermediate dependency:

```text
node1 = grouped stage: ops {op1, op2}
remaining dep = buf0 produced by outer op0
name_to_fused_node[op0].ancestors = {op0, op1}
node1_ops & fused_ancestors = {op1}
```

The scheduler rejects this as "intermediate nodes between node1 & node2", even though `op0` and `op1` are already inside the same nested fused node. This is why removing the nested ancestor cleanup makes `test_fullres_kernel_form` emit a separate pointwise kernel.

## Is This A Dependency Refresh Problem?

Mostly no. `refresh_group_node_dependencies()` recomputes `read_writes` and `unmet_dependencies` for the grouped node. The stale part is the meaning of `.ancestors`: the raw union includes edges that used to be external but became internal after fusion.

The ambiguity is that `.ancestors` is used as scheduler graph metadata, not just unmet dependency metadata. Once nested code asks generic legality about only an internal stage, ancestor metadata for the enclosing fused node can be too broad.

## Options Considered

### 1. Nested-local ancestor cleanup

Keep:

```python
self.ancestors -= self.get_operation_names()
```

inside `FusedNestedReductions.__init__`, with a comment explaining that internal producer-consumer edges should not be visible as ancestors of the fused container.

Pros:

- Smallest behavior change.
- Only affects the new staged nested node.
- Fixes the observed full-res epilogue fusion failure.

Cons:

- Atypical compared to ordinary `FusedSchedulerNode`.
- Easy for reviewers to ask why only nested needs it.

### 2. Normalize all fused/grouped ancestors in `init_group_node()`

Make the generic invariant:

```python
group_snode.ancestors -= group_snode.get_operation_names()
```

Pros:

- Conceptually clean: a fused node's ancestors are external predecessors only.
- Avoids nested-specific metadata fix.

Cons:

- Global scheduler semantic change.
- Could affect ordinary fusion, cycle checks, and foreach/grouped nodes.
- Needs broader review and tests than this nested scheduler commit.

I would not put this into the nested scheduler PR unless we intentionally split it into a separate scheduler cleanup.

### 3. Check append legality against the whole nested node

Have `can_fuse_with()` use generic fusion legality on `self` rather than `self.node2`, while `fuse_with()` still appends into `node2`.

Pros:

- More like ordinary fusion for dependency matching: outer-stage writes are visible.

Cons:

- It needs a way to avoid recursively calling `FusedNestedReductions.can_fuse_with()` from `Scheduler.can_fuse(self, other)`.
- A flag like `use_nested_reduction_hook=False` is ugly API surface.
- It mixes "legality is whole fused node" with "construction appends into one stage"; that needs careful proof to avoid accepting a node that generic fusion likes but staged codegen cannot place.

I do not like this as-is.

### 4. Custom nested vertical dependency legality

Write a nested-specific legality check that understands both stages and explicitly treats outer-stage reads as internal when appending grouped-stage consumers.

Pros:

- Most precise model of staged nested fusion.

Cons:

- Duplicates generic vertical dependency logic.
- Easy to miss scheduler invariants like weak deps, sync modes, mutation names, and intermediate-dependency checks.
- More code than the current problem warrants.

This should be a last resort.

## Append Consumers And The Stage Guard

`FusedNestedReductions.can_fuse_with(other)` has a guard:

```python
if not (self.node2.get_operation_names() & other.ancestors):
    return False
```

This is there because `fuse_with()` appends `other` into `node2`. If `other` only consumes the outer stage, appending it into the grouped stage is the wrong placement. Such nodes need to be fused before the nested pair is formed, or codegen needs a richer staged append API.

The comment should say this directly. The guard is not just an arbitrary filter; it protects the staged construction path.

## LOCAL_REDUCTION_INPUT vs PARENT_FULL In Append

`LOCAL_REDUCTION_INPUT` pointwise nodes are producers for the grouped reduction body. They must be inserted before the grouped reduction. The current append path can only append downstream consumers after the grouped reduction has already been staged, so it rejects them.

`PARENT_FULL` pointwise nodes are consumers after grouped reduction output is available, but they may run at parent resolution. In the conservative scheduler commit, these are rejected because they need relaxed dependency equivalence. In the follow-up index-equivalence commit, they can be allowed when the dependency matcher proves the broadcast/loop-order equivalence.

That split is fundamental to the commit staging:

- Scheduler legality commit: classify domains, reject parent-full append consumers.
- Dependency-equivalence commit: allow parent-full append consumers through the normal vertical path with explicit opt-in.

## Score Path Comment

The score fast path:

```python
if min(node1_dep_len, node2_dep_len) * 4 < max(node1_dep_len, node2_dep_len):
    ...
    if score:
        return ...
else:
    common_memory_deps = ...
```

exists to avoid building a potentially large full set intersection when one dep set is much smaller. The follow-up index-equivalence scoring needs the `if score: return` shape because if exact dep scoring is zero, it falls through to relaxed producer-consumer scoring and then buffer-overlap scoring.

For the scheduler-only commit, any unrelated comment/format churn around this block should be removed.

## Recommended Near-Term Shape

For the current PR stack, I would keep the changes narrowly scoped:

1. Keep `PointwiseDomainContext` as a local readability cleanup.
2. Do not change generic `init_group_node()` ancestor behavior in this commit.
3. Avoid a `use_nested_reduction_hook=False` style API.
4. Keep the nested-only ancestor handling only if the append path continues to validate against `self.node2`.
5. Add a clear comment explaining that nested staged append checks a substage, so internal ancestor edges must not be visible as external ancestors of the fused container.
6. Keep unrelated scoring/comment churn out of the scheduler legality commit.

Longer term, if we want this to feel fully native to the scheduler, the right abstraction is probably a staged fused node API: legality can reason about the whole fused node, while construction says which internal stage receives the appended consumer. That is bigger than this PR and should not be smuggled into the nested legality patch.

