# Overlap Scheduling Design Options

## Background

The overlap scheduler reorders FX graph nodes to maximize compute-collective overlap. The current approach uses a priority queue with scoring based on:

1. **Compute domination index**: Which compute node does this node block? (0, 1, 2, ... num_compute, or sys.maxsize for off-path)
2. **Local priority**: collective_start=-1, wait=1, other=0
3. **Original node index**: Tiebreaker for stability

```python
# Current scoring
return (
    self.compute_index_domination[node],
    compute_local_priority,
    self.node_idx[node],
)
```

## How Overlap is Achieved

Two mechanisms create overlap:

1. **Prefetching** (`_schedule_collectives_for_overlap`): During compute scheduling, we look for collectives that can be started early and hidden by the compute time. This pulls all_gather starts earlier.

2. **Wait deferral**: Exposed waits (not yet hidden) get priority=1, pushing them later in the schedule. This is especially important for reduce_scatter, where the start happens after compute and overlap comes from deferring the wait.

## Concerns with Current Approach

### 1. Inadvertent Off-Path Reordering
Off-path nodes (domination = sys.maxsize) sort to the very end of the graph. This can hurt cache locality.

Example: `reduce_scatter -> wait -> upcast -> use`

The upcast might drift far from the reduce_scatter wait, even though they should stay close together.

### 2. Unmodeled Behaviors
- Multi-stream usage not modeled
- Fragility of using compute nodes as the only "anchor"
- Other runtime behaviors we don't capture

### 3. Unpredictability
Hard to reason about what the scheduler will do. Small changes in the graph can cause large reorderings.

### 4. Forward Pass Memory Blowup
Pattern: `param -> bfloat16() -> all_gather`

If we defer an exposed wait too aggressively, we might schedule many dtype conversions early (because they lead to collectives), blowing up memory.

## What Domination Does Well

1. **Prevents premature upstream scheduling**: The `param -> bfloat16() -> all_gather` chain stays together, scheduled when actually needed.

2. **Forces waits at the right time**: When compute is blocked, the wait gets scheduled.

3. **On-path vs off-path distinction**: Useful signal for when nodes are "needed" vs "optional".

---

## Design Options

### Option A: Original Order + Bounded Wait Deferral

**Idea**: Schedule nodes primarily by original graph order. Only defer exposed waits, and bound how far they can be deferred using compute domination as a "distance" metric.

```python
def _compute_score(self, node):
    if is_exposed_wait(node) and should_defer(node):
        # Use domination to bound deferral distance
        wait_domination = self.compute_index_domination[self.wait_to_start[node]]
        compute_distance = wait_domination - self.current_compute_index

        if compute_distance <= MAX_DEFER_COMPUTE_DISTANCE:
            # Defer: add penalty to push later
            return (self.node_idx[node] + DEFER_PENALTY, self.node_idx[node])

    # Default: original order
    return (self.node_idx[node], self.node_idx[node])
```

**Pros**:
- Predictable: mostly original order
- Bounded deferral: uses domination for what it's good at
- Off-path nodes stay near original position

**Cons**:
- May lose some overlap opportunities
- DEFER_PENALTY and MAX_DEFER_COMPUTE_DISTANCE are tunable parameters

---

### Option B: Shadow Schedule for Off-Path Nodes

**Idea**: Track progress through the "main" (on-path) schedule. Schedule off-path nodes when we're "around" their original position, except defer waits until hidden.

```python
def _compute_score(self, node):
    if is_off_path(node):
        if is_exposed_wait(node):
            # Defer until hidden
            return (1, self.node_idx[node])
        else:
            # Schedule when we're near this node's original position
            distance = abs(self.node_idx[node] - self.current_main_schedule_idx)
            return (0, distance, self.node_idx[node])
    else:
        # On-path: use domination as before
        return (0, self.compute_index_domination[node], self.node_idx[node])
```

**Pros**:
- On-path behavior unchanged
- Off-path nodes anchored to original position
- Wait deferral still works

**Cons**:
- More complex tracking of "main schedule"
- Definition of "current_main_schedule_idx" needs care

---

### Option C: Keep Domination, Fix Off-Path Handling

**Idea**: Keep domination-based scoring for on-path nodes, but fix off-path nodes to stay near their original position instead of drifting to sys.maxsize.

```python
def _compute_score(self, node):
    domination = self.compute_index_domination[node]

    if domination == sys.maxsize:
        # Off-path: find nearest on-path ancestor/descendant
        # and inherit their domination (or use a bounded value)
        domination = self._get_anchored_domination(node)

    # ... rest of scoring
    return (domination, compute_local_priority, self.node_idx[node])
```

**Pros**:
- Minimal change to existing logic
- Off-path nodes stay "anchored" to nearby on-path nodes

**Cons**:
- `_get_anchored_domination` logic could be complex
- Still has some unpredictability from domination-based reordering

---

### Option D: Chain-Aware Scheduling

**Idea**: Identify "chains" of related nodes (e.g., `rs_wait -> upcast -> use`) and keep them together. When deferring a wait, the chain moves as a unit.

```python
# Precompute chains
self.node_to_chain = self._compute_chains()

def _compute_score(self, node):
    chain = self.node_to_chain[node]
    chain_head = chain[0]

    if is_exposed_wait(chain_head):
        # Defer whole chain
        return (1, self.node_idx[chain_head], self.node_idx[node])

    return (0, self.node_idx[chain_head], self.node_idx[node])
```

**Pros**:
- Chains stay together (good for cache locality)
- Conceptually clean

**Cons**:
- Chain detection logic needed
- What defines a "chain"? (unary ops? same domination?)

---

### Option E: No Priority Queue Reordering, Only Prefetching

**Idea**: Schedule strictly by original order. No wait deferral via priority queue. Only source of reordering is explicit prefetching during compute.

```python
def _compute_score(self, node):
    return (self.node_idx[node],)  # Just original order
```

**Pros**:
- Most predictable
- No surprising reorderings

**Cons**:
- Loses reduce_scatter overlap (which relies on wait deferral)
- May lose other overlap opportunities

---

### Option F: Quota-Based Wait Deferral

**Idea**: Allow at most K exposed waits to be deferred at any time. When we hit the quota, force the oldest deferred wait to be scheduled.

```python
def _compute_score(self, node):
    if is_exposed_wait(node) and self.deferred_wait_count < MAX_DEFERRED_WAITS:
        return (1, self.node_idx[node])  # Defer

    return (0, self.node_idx[node])  # Original order
```

**Pros**:
- Bounded memory from deferred waits
- Simple to implement

**Cons**:
- Doesn't address off-path drift
- MAX_DEFERRED_WAITS is arbitrary

---

### Option G: Hybrid - Keep On-Path Domination, Fix Off-Path with Separate Queue

**Idea**: Keep current domination-based scheduling for on-path nodes (which works well). Handle off-path nodes separately - either with a separate queue or by scheduling them when the main loop has passed their original index.

The key insight: on-path domination values (0, 1, 2, ..., num_compute) and off-path node indices (0, 1, ..., num_nodes) are **not comparable scales**. Mixing them in one priority queue causes problems:
- Off-path with sys.maxsize drifts to the end
- Off-path with node_idx would have arbitrary ordering relative to on-path domination

**Approach**: Two-queue or interleaved scheduling

```python
# Conceptually: two separate orderings
on_path_queue: sorted by (domination, compute_local_priority, node_idx)
off_path_queue: sorted by (node_idx)  # original order

# Main loop:
while nodes_to_schedule:
    # Schedule on-path nodes by domination
    on_path_node = on_path_queue.peek()

    # Check if any off-path nodes should be scheduled now
    # (their original idx <= current progress through graph)
    for off_path_node in ready_off_path:
        if off_path_node.node_idx <= current_scheduled_idx:
            if is_exposed_wait(off_path_node):
                # Defer until hidden
                continue
            schedule(off_path_node)

    # Schedule next on-path node
    schedule(on_path_node)
```

**Alternatively**: Single queue but with normalized scoring

```python
def _compute_score(self, node):
    domination = self.compute_index_domination[node]

    if domination == sys.maxsize:  # off-path
        if is_exposed_wait(node):
            # Defer: schedule after on-path nodes at similar position
            # Use a "virtual domination" based on when we'd naturally reach this node
            virtual_domination = self._get_virtual_domination(node)
            return (virtual_domination, 1, self.node_idx[node])
        else:
            # Schedule when main loop reaches this position
            virtual_domination = self._get_virtual_domination(node)
            return (virtual_domination, 0, self.node_idx[node])
    else:  # on-path - unchanged
        return (domination, compute_local_priority, self.node_idx[node])

def _get_virtual_domination(self, node):
    # Map node_idx to domination scale
    # e.g., find the domination of nearest on-path predecessor
    # or interpolate based on position in graph
    ...
```

**Pros**:
- On-path behavior unchanged (keeps the good properties)
- Off-path nodes stay near original position
- Clean separation of concerns
- Solves the "incomparable scales" problem

**Cons**:
- More complex implementation (two queues or virtual domination mapping)
- Need to define how to interleave on-path and off-path scheduling

---

## Recommendation

**Option G (Hybrid with Separate Off-Path Handling)** seems like the best approach:

1. **On-path unchanged**: Keeps the good properties of domination-based scheduling
2. **Off-path fixed**: Nodes stay near original position instead of drifting to end
3. **Wait deferral preserved**: Exposed waits still deferred for overlap
4. **Comparable scales**: Separate handling avoids mixing incomparable orderings

Implementation approaches for Option G:
- **Two queues**: Cleaner separation but more complex main loop
- **Virtual domination**: Map off-path node_idx to domination scale (e.g., via nearest on-path neighbor)

This could be combined with **Option F (Quota-Based)** for additional memory safety.

## Next Steps

1. Prototype Option G on a new branch
2. Test on llama1d_bwd to compare:
   - Overlap achieved (exposed vs hidden collectives)
   - Memory profile (peak, regression from baseline)
   - Node ordering predictability
3. Iterate based on results
