# ILP Overlap Scheduler

An Integer Linear Programming (ILP) based scheduler for optimizing compute-communication overlap in distributed training graphs.

## Overview

The ILP overlap scheduler (`ilp_overlap_scheduling.py`) finds an optimal ordering of operations that maximizes overlap between collective communications and compute operations while respecting data dependencies.

## Key Features

- **Optimal scheduling**: Uses ILP to find provably optimal operation ordering (within solver time limits)
- **Collective bucketing**: Groups independent collectives of the same type to reduce launch overhead
- **FIFO ordering**: Respects NCCL's FIFO requirement that waits appear in the same order as their corresponding collectives
- **Dependency preservation**: Maintains correctness by respecting all data dependencies

## Formal Problem Definition

### Sets

- **N**: Set of operations (nodes) to schedule, indexed by i ∈ {0, ..., n-1}
- **C**: Set of collective operations (subset of N)
- **W**: Set of wait operations (subset of N)
- **G**: Set of (process_group, collective_type) pairs
- **K_g**: Set of buckets for group g ∈ G, where |K_g| = |collectives in g|

### Parameters

- **d_i**: Execution time of operation i on main stream
- **c_i**: Communication time of collective i (for i ∈ C)
- **deps**: Set of dependency pairs (i, j) where i must execute before j
- **coll(w)**: The collective operation corresponding to wait w
- **M**: Large constant for big-M constraints

### Decision Variables

- **x_{i,t} ∈ {0,1}**: 1 if operation i is assigned to position t
- **b_{g,c,k} ∈ {0,1}**: 1 if collective c in group g is assigned to bucket k
- **f_{g,c,k} ∈ {0,1}**: 1 if collective c is the first in bucket k
- **T_t ∈ ℝ⁺**: Cumulative main-stream time at position t
- **S_t ∈ ℝ⁺**: Cumulative comm-stream time at position t

### Objective

Minimize total execution time:

```
minimize T_n
```

### Constraints

**1. Assignment**: Each operation assigned to exactly one position
```
∀i: Σ_t x_{i,t} = 1
∀t: Σ_i x_{i,t} = 1
```

**2. Dependencies**: If (i,j) ∈ deps, then i must be scheduled before j
```
∀(i,j) ∈ deps: Σ_t (t · x_{i,t}) + 1 ≤ Σ_t (t · x_{j,t})
```

**3. Bucket Assignment**: Each collective in exactly one bucket
```
∀g ∈ G, ∀c ∈ C_g: Σ_k b_{g,c,k} = 1
```

**4. FIFO Ordering**: Waits must appear in same relative order as their collectives
```
∀w1, w2 ∈ W where pos(coll(w1)) < pos(coll(w2)):
    pos(w1) < pos(w2)
```

**5. Main Stream Timing**: Cumulative time includes operation durations
```
T_0 = 0
∀t > 0: T_t ≥ T_{t-1} + Σ_i (d_i · x_{i,t})
```

**6. Communication Stream Timing**: Comm stream advances with bucketed collective costs
```
S_0 = 0
∀t > 0: S_t ≥ S_{t-1} + effective_cost(t)
```

where `effective_cost(t)` is the bucket cost when a collective is first-in-bucket at position t.

**7. Wait Synchronization**: Main stream blocks at wait until comm stream completes
```
∀w ∈ W, ∀t: T_t ≥ S_{comm_complete(w)} - M(1 - x_{w,t})
```

## API

```python
from torch._inductor.fx_passes.ilp_overlap_scheduling import schedule_overlap_ilp

schedule_overlap_ilp(
    gm,                           # GraphModule to optimize
    bucket_mode="custom_ops_multidtype",  # Bucketing strategy
    custom_runtime_estimation=None,       # Optional runtime estimator
)
```

## PuLP API Patterns

This implementation follows patterns from `torch/distributed/_tools/sac_ilp.py`:

- Uses `LpVariable` for decision variables with descriptive names
- Uses `lpDot` for weighted sums (e.g., bucket costs)
- Uses `.varValue` attribute to extract solution values
- Uses `PULP_CBC_CMD` solver with gap tolerance and time limits

## Tests

- **Unit tests**: `test/distributed/test_overlap_bucketing_unit.py::TestILPOverlapScheduler`
- **Multi-proc tests**: `test/distributed/test_aten_comm_compute_reordering.py::TestILPOverlapSchedulerMultiProc`
- **Comparison tests**: `test/distributed/test_aten_comm_compute_reordering.py::TestILPvsGreedyComparison`
