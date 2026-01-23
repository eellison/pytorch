"""
ILP-based overlap scheduling for distributed neural network execution.

This module provides an alternative to the greedy OverlapScheduler that uses
Integer Linear Programming (ILP) to jointly optimize:
1. Operation scheduling (ordering of all operations)
2. Collective bucketing (grouping collectives to amortize startup costs)
3. Compute-communication overlap (hiding collective latency behind compute)

The ILP formulation minimizes total execution time on the main stream by
modeling both main stream (compute) and side streams (per-process-group
collective execution).
"""

import logging
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import torch
import torch.fx as fx
from torch._dynamo.utils import counters
from torch._inductor.comm_analysis import estimate_fx_collective_memory_footprint
from torch._inductor.fx_passes.bucketing import (
    _schedulable_wait_node,
    bucket_key,
    BucketMode,
    is_all_gather_into_tensor,
    is_all_reduce_tensor,
    is_reduce_scatter_tensor,
)
from torch._inductor.fx_passes.fusion_regions import FusionRegion
from torch._inductor.fx_passes.memory_estimator import MemoryTracker
from torch._inductor.fx_passes.overlap_scheduling import (
    benchmark_node,
    CollectiveInfo,
    estimate_collective_time,
    estimate_mem_bound_runtime_ms,
    get_group_name,
    is_compute_node,
)
from torch._logging import trace_structured
from torch.utils._ordered_set import OrderedSet

from ..pattern_matcher import stable_topological_sort


log = logging.getLogger(__name__)


try:
    from pulp import (
        lpDot,
        LpBinary,
        LpContinuous,
        LpInteger,
        LpMinimize,
        LpProblem,
        LpStatus,
        lpSum,
        LpVariable,
        PULP_CBC_CMD,
    )

    HAS_PULP = True
except ImportError:
    HAS_PULP = False


@dataclass
class CollectiveType:
    ALL_GATHER = "AG"
    REDUCE_SCATTER = "RS"
    ALL_REDUCE = "AR"


def get_collective_type(node: fx.Node) -> str | None:
    if is_all_gather_into_tensor(node):
        return CollectiveType.ALL_GATHER
    elif is_reduce_scatter_tensor(node):
        return CollectiveType.REDUCE_SCATTER
    elif is_all_reduce_tensor(node):
        return CollectiveType.ALL_REDUCE
    return None


@dataclass
class ILPCollectiveInfo:
    """Information about a collective for ILP scheduling."""

    start_node: fx.Node
    wait_node: fx.Node
    size_bytes: int
    estimated_time_ms: float
    pg_name: str
    coll_type: str
    coll_idx: int  # Index within this (pg, type) group


@dataclass
class ILPSolution:
    """Solution from the ILP solver."""

    positions: dict[fx.Node, int]
    bucket_assignments: dict[fx.Node, int]  # collective -> bucket index
    total_time: float
    status: str


class ILPOverlapScheduler:
    """
    ILP-based scheduler that jointly optimizes scheduling and bucketing.

    Unlike the greedy OverlapScheduler, this uses Integer Linear Programming
    to find a globally optimal schedule that:
    1. Respects all DAG dependencies
    2. Maintains FIFO ordering within each process group
    3. Buckets collectives to minimize overhead
    4. Maximizes compute-communication overlap
    """

    def __init__(
        self,
        gm: torch.fx.GraphModule,
        bucket_mode: BucketMode = "custom_ops_multidtype",
        custom_runtime_estimation: Callable[[fx.Node, int | None], float | None]
        | None = None,
        solver_time_limit: int = 300,
        solver_gap: float = 0.05,
        insert_overlap_deps: bool = False,
        enable_fusion_regions: bool = False,
    ):
        if not HAS_PULP:
            raise ImportError(
                "PuLP is required for ILP overlap scheduling. "
                "Install with: pip install pulp"
            )

        self.gm = gm
        self.graph = gm.graph
        self.bucket_mode = bucket_mode
        self.custom_runtime_estimation = custom_runtime_estimation
        self.solver_time_limit = solver_time_limit
        self.solver_gap = solver_gap
        self.insert_overlap_deps = insert_overlap_deps

        # Build and collapse fusion regions FIRST so all subsequent operations
        # work on the collapsed graph where fused ops are atomic units
        self.region_of: dict[fx.Node, FusionRegion] = {}
        if enable_fusion_regions:
            from torch._inductor.fx_passes.fusion_regions import (
                build_fusion_regions,
                collapse_fusion_regions,
            )

            self.region_of = build_fusion_regions(self.gm)
            if self.region_of:
                self.region_of = collapse_fusion_regions(self.gm, self.region_of)
                # fuse_by_partitions replaces gm.graph, so we need to update our reference
                self.graph = gm.graph

        # Build graph structures
        stable_topological_sort(self.graph)
        self.nodes = list(self.graph.nodes)
        self.node_idx = {n: i for i, n in enumerate(self.nodes)}

        # Identify different node types
        self.compute_nodes: list[fx.Node] = []
        self.collective_info: dict[fx.Node, ILPCollectiveInfo] = {}
        self.wait_to_start: dict[fx.Node, fx.Node] = {}

        # Process group tracking
        self.all_pgs: OrderedSet[str] = OrderedSet()
        self.pg_type_collectives: dict[tuple[str, str], list[fx.Node]] = defaultdict(
            list
        )

        self._identify_nodes()
        self._collect_node_ancestors()

        # Memory tracking (only CUDA memory, primals not releasable)
        self.memory_tracker = MemoryTracker(self.graph)

        # Runtime estimations (populated lazily)
        self._runtime_cache: dict[fx.Node, float] = {}

    def _identify_nodes(self) -> None:
        """Identify compute nodes and collectives."""
        for node in self.nodes:
            if is_compute_node(node):
                self.compute_nodes.append(node)

            if _schedulable_wait_node(node):
                start = node.args[0]
                assert isinstance(start, fx.Node)

                pg_name = get_group_name(start)
                coll_type = get_collective_type(start)
                if coll_type is None:
                    continue

                self.all_pgs.add(pg_name)

                coll_idx = len(self.pg_type_collectives[(pg_name, coll_type)])
                self.pg_type_collectives[(pg_name, coll_type)].append(start)

                info = ILPCollectiveInfo(
                    start_node=start,
                    wait_node=node,
                    size_bytes=estimate_fx_collective_memory_footprint(start),
                    estimated_time_ms=estimate_collective_time(
                        start, custom_runtime_estimation=self.custom_runtime_estimation
                    ),
                    pg_name=pg_name,
                    coll_type=coll_type,
                    coll_idx=coll_idx,
                )
                self.collective_info[start] = info
                self.wait_to_start[node] = start

    def _collect_node_ancestors(self) -> None:
        """Collect ancestors for each node."""
        self.node_ancestors: dict[fx.Node, OrderedSet[fx.Node]] = defaultdict(
            OrderedSet
        )
        for node in self.nodes:
            for input_node in node.all_input_nodes:
                self.node_ancestors[node].add(input_node)
                self.node_ancestors[node] |= self.node_ancestors[input_node]

    def _get_runtime_estimate(self, node: fx.Node) -> float:
        """Get runtime estimate for a node in ms."""
        if node in self._runtime_cache:
            return self._runtime_cache[node]

        runtime = 0.0
        if is_compute_node(node):
            runtime = benchmark_node(node, self.custom_runtime_estimation)
        elif node in self.collective_info:
            runtime = self.collective_info[node].estimated_time_ms
        elif node in self.region_of:
            # Use precomputed cost for fusion region call_module nodes
            runtime = self.region_of[node].cost_ms
        elif node.op == "call_function":
            runtime = estimate_mem_bound_runtime_ms(node)

        self._runtime_cache[node] = runtime
        return runtime

    def _get_bucket_key(self, node: fx.Node) -> object | None:
        """Get bucket key for a collective."""
        return bucket_key(node, self.bucket_mode)

    def _build_ilp_problem(self) -> tuple[LpProblem, dict[str, Any]]:
        """
        Build the ILP problem.

        Returns the problem and a dictionary of all decision variables.
        """
        N = len(self.nodes)
        prob = LpProblem("OverlapScheduling", LpMinimize)

        # Big-M constants
        M_pos = N + 1
        M_time = 1e6  # Large upper bound on total time in ms

        vars_dict: dict[str, Any] = {}

        # =================================================================
        # Decision Variables
        # =================================================================

        # Position assignment: x[op, t] = 1 if op is at position t
        x = {}
        for i, node in enumerate(self.nodes):
            for t in range(N):
                x[i, t] = LpVariable(f"x_{i}_{t}", cat=LpBinary)
        vars_dict["x"] = x

        # Position variable: pi[op] = position of op
        pi = {}
        for i, node in enumerate(self.nodes):
            pi[i] = LpVariable(f"pi_{i}", lowBound=0, upBound=N - 1, cat=LpInteger)
        vars_dict["pi"] = pi

        # Precedence: p[i,j] = 1 if node i comes before node j
        # Only create for pairs that interact (same PG collectives, dependencies)
        p = {}
        pairs_needed = self._compute_precedence_pairs()
        for i, j in pairs_needed:
            p[i, j] = LpVariable(f"p_{i}_{j}", cat=LpBinary)
        vars_dict["p"] = p
        vars_dict["pairs_needed"] = pairs_needed

        # Bucket assignment: b[c, k] = 1 if collective c is in bucket k
        b = {}
        bucket_used = {}
        for (pg, coll_type), colls in self.pg_type_collectives.items():
            K = len(colls)
            for c_idx, coll in enumerate(colls):
                for k in range(K):
                    b[pg, coll_type, c_idx, k] = LpVariable(
                        f"b_{pg}_{coll_type}_{c_idx}_{k}", cat=LpBinary
                    )
            for k in range(K):
                bucket_used[pg, coll_type, k] = LpVariable(
                    f"bucket_used_{pg}_{coll_type}_{k}", cat=LpBinary
                )
        vars_dict["b"] = b
        vars_dict["bucket_used"] = bucket_used

        # Bucket-related variables for timing
        # is_first[c, k] = 1 if c is the first collective in bucket k
        is_first = {}
        for (pg, coll_type), colls in self.pg_type_collectives.items():
            K = len(colls)
            for c_idx in range(len(colls)):
                for k in range(K):
                    is_first[pg, coll_type, c_idx, k] = LpVariable(
                        f"is_first_{pg}_{coll_type}_{c_idx}_{k}", cat=LpBinary
                    )
        vars_dict["is_first"] = is_first

        # Timing variables
        T_main = {}
        for t in range(N + 1):
            T_main[t] = LpVariable(f"T_main_{t}", lowBound=0, cat=LpContinuous)
        vars_dict["T_main"] = T_main

        T_side = {}
        for pg in self.all_pgs:
            for t in range(N + 1):
                T_side[pg, t] = LpVariable(f"T_side_{pg}_{t}", lowBound=0, cat=LpContinuous)
        vars_dict["T_side"] = T_side

        # =================================================================
        # Constraints
        # =================================================================

        # 1. Position assignment constraints
        # Each op in exactly one position
        for i in range(N):
            prob += lpSum(x[i, t] for t in range(N)) == 1, f"op_{i}_one_position"

        # Each position has exactly one op
        for t in range(N):
            prob += lpSum(x[i, t] for i in range(N)) == 1, f"position_{t}_one_op"

        # Link pi to x
        for i in range(N):
            prob += pi[i] == lpSum(t * x[i, t] for t in range(N)), f"pi_link_{i}"

        # 2. Precedence linking
        for i, j in pairs_needed:
            # p[i,j] = 1 means pi[i] < pi[j]
            prob += pi[j] >= pi[i] + 1 - M_pos * (1 - p[i, j]), f"prec_fwd_{i}_{j}"
            prob += pi[i] >= pi[j] + 1 - M_pos * p[i, j], f"prec_bwd_{i}_{j}"

        # 3. DAG dependency constraints
        for j, node_j in enumerate(self.nodes):
            for input_node in node_j.all_input_nodes:
                i = self.node_idx[input_node]
                prob += pi[j] >= pi[i] + 1, f"dag_dep_{i}_{j}"

        # 4. Start before wait (per collective)
        for start, info in self.collective_info.items():
            start_idx = self.node_idx[start]
            wait_idx = self.node_idx[info.wait_node]
            prob += pi[wait_idx] >= pi[start_idx] + 1, f"start_before_wait_{start_idx}"

        # 5. FIFO ordering within process group
        # If start_i < start_j then wait_i < wait_j for same PG
        for pg in self.all_pgs:
            pg_colls = [
                c for c, info in self.collective_info.items() if info.pg_name == pg
            ]
            for i, coll_i in enumerate(pg_colls):
                for coll_j in pg_colls[i + 1 :]:
                    info_i = self.collective_info[coll_i]
                    info_j = self.collective_info[coll_j]
                    start_i = self.node_idx[coll_i]
                    start_j = self.node_idx[coll_j]
                    wait_i = self.node_idx[info_i.wait_node]
                    wait_j = self.node_idx[info_j.wait_node]

                    # Create precedence vars if not exists
                    if (start_i, start_j) in p:
                        p_starts = p[start_i, start_j]
                    else:
                        p_starts = LpVariable(f"p_fifo_start_{start_i}_{start_j}", cat=LpBinary)
                        p[start_i, start_j] = p_starts
                        prob += pi[start_j] >= pi[start_i] + 1 - M_pos * (1 - p_starts)
                        prob += pi[start_i] >= pi[start_j] + 1 - M_pos * p_starts

                    if (wait_i, wait_j) in p:
                        p_waits = p[wait_i, wait_j]
                    else:
                        p_waits = LpVariable(f"p_fifo_wait_{wait_i}_{wait_j}", cat=LpBinary)
                        p[wait_i, wait_j] = p_waits
                        prob += pi[wait_j] >= pi[wait_i] + 1 - M_pos * (1 - p_waits)
                        prob += pi[wait_i] >= pi[wait_j] + 1 - M_pos * p_waits

                    # FIFO: start order equals wait order
                    prob += p_starts == p_waits, f"fifo_{start_i}_{start_j}"

        # 6. Bucket assignment constraints
        for (pg, coll_type), colls in self.pg_type_collectives.items():
            K = len(colls)

            # Each collective in exactly one bucket
            for c_idx in range(len(colls)):
                prob += (
                    lpSum(b[pg, coll_type, c_idx, k] for k in range(K)) == 1,
                    f"one_bucket_{pg}_{coll_type}_{c_idx}",
                )

            # Bucket used indicator
            for k in range(K):
                num_in_bucket = lpSum(b[pg, coll_type, c_idx, k] for c_idx in range(len(colls)))
                prob += bucket_used[pg, coll_type, k] <= num_in_bucket
                prob += num_in_bucket <= len(colls) * bucket_used[pg, coll_type, k]

            # Symmetry breaking: lower-indexed buckets used first
            for k in range(1, K):
                prob += (
                    bucket_used[pg, coll_type, k] <= bucket_used[pg, coll_type, k - 1],
                    f"sym_break_{pg}_{coll_type}_{k}",
                )

            # Bucket adjacency: starts in same bucket must be consecutive
            # f_start[k] = first start position, l_start[k] = last start position
            # l_start[k] - f_start[k] + 1 = count of collectives in bucket k
            f_start = {}
            l_start = {}
            f_wait = {}
            l_wait = {}
            for k in range(K):
                f_start[k] = LpVariable(
                    f"f_start_{pg}_{coll_type}_{k}", lowBound=0, upBound=N - 1, cat=LpInteger
                )
                l_start[k] = LpVariable(
                    f"l_start_{pg}_{coll_type}_{k}", lowBound=0, upBound=N - 1, cat=LpInteger
                )
                f_wait[k] = LpVariable(
                    f"f_wait_{pg}_{coll_type}_{k}", lowBound=0, upBound=N - 1, cat=LpInteger
                )
                l_wait[k] = LpVariable(
                    f"l_wait_{pg}_{coll_type}_{k}", lowBound=0, upBound=N - 1, cat=LpInteger
                )

            # Link positions to bucket bounds
            for c_idx, coll in enumerate(colls):
                info = self.collective_info[coll]
                start_idx = self.node_idx[coll]
                wait_idx = self.node_idx[info.wait_node]
                for k in range(K):
                    # If c is in bucket k, its start is within [f_start[k], l_start[k]]
                    prob += (
                        pi[start_idx] >= f_start[k] - M_pos * (1 - b[pg, coll_type, c_idx, k]),
                        f"start_ge_f_{pg}_{coll_type}_{c_idx}_{k}",
                    )
                    prob += (
                        pi[start_idx] <= l_start[k] + M_pos * (1 - b[pg, coll_type, c_idx, k]),
                        f"start_le_l_{pg}_{coll_type}_{c_idx}_{k}",
                    )
                    # Same for waits
                    prob += (
                        pi[wait_idx] >= f_wait[k] - M_pos * (1 - b[pg, coll_type, c_idx, k]),
                        f"wait_ge_f_{pg}_{coll_type}_{c_idx}_{k}",
                    )
                    prob += (
                        pi[wait_idx] <= l_wait[k] + M_pos * (1 - b[pg, coll_type, c_idx, k]),
                        f"wait_le_l_{pg}_{coll_type}_{c_idx}_{k}",
                    )

            # Span equals count (ensures no gaps) - only for used buckets
            for k in range(K):
                bucket_count = lpSum(b[pg, coll_type, c_idx, k] for c_idx in range(len(colls)))
                # l_start - f_start + 1 = count (if bucket used)
                prob += (
                    l_start[k] - f_start[k] + 1 >= bucket_count - M_pos * (1 - bucket_used[pg, coll_type, k]),
                    f"start_span_ge_{pg}_{coll_type}_{k}",
                )
                prob += (
                    l_start[k] - f_start[k] + 1 <= bucket_count + M_pos * (1 - bucket_used[pg, coll_type, k]),
                    f"start_span_le_{pg}_{coll_type}_{k}",
                )
                # Same for waits
                prob += (
                    l_wait[k] - f_wait[k] + 1 >= bucket_count - M_pos * (1 - bucket_used[pg, coll_type, k]),
                    f"wait_span_ge_{pg}_{coll_type}_{k}",
                )
                prob += (
                    l_wait[k] - f_wait[k] + 1 <= bucket_count + M_pos * (1 - bucket_used[pg, coll_type, k]),
                    f"wait_span_le_{pg}_{coll_type}_{k}",
                )

            # Only collectives with same bucket key can be bucketed together
            # Group by bucket key
            key_groups: dict[object, list[int]] = defaultdict(list)
            for c_idx, coll in enumerate(colls):
                key = self._get_bucket_key(coll)
                if key is not None:
                    key_groups[key].append(c_idx)

            # Different keys cannot be in same bucket
            all_keys = list(key_groups.keys())
            for i, key1 in enumerate(all_keys):
                for key2 in all_keys[i + 1 :]:
                    for c1_idx in key_groups[key1]:
                        for c2_idx in key_groups[key2]:
                            for k in range(K):
                                prob += (
                                    b[pg, coll_type, c1_idx, k]
                                    + b[pg, coll_type, c2_idx, k]
                                    <= 1,
                                    f"diff_key_{pg}_{coll_type}_{c1_idx}_{c2_idx}_{k}",
                                )

            # is_first constraints: is_first[c,k] = 1 iff c is in bucket k
            # AND no other collective in bucket k has its start before c's start
            for c_idx, coll in enumerate(colls):
                start_idx = self.node_idx[coll]
                for k in range(K):
                    # is_first[c,k] <= b[c,k] (must be in bucket to be first)
                    prob += (
                        is_first[pg, coll_type, c_idx, k] <= b[pg, coll_type, c_idx, k],
                        f"is_first_in_bucket_{pg}_{coll_type}_{c_idx}_{k}",
                    )

                    # For each other collective j, if j is also in bucket k and
                    # j's start comes before c's start, then c cannot be first
                    for j_idx, other_coll in enumerate(colls):
                        if j_idx == c_idx:
                            continue
                        other_start_idx = self.node_idx[other_coll]

                        # We need: if b[j,k] = 1 AND p[other_start, start] = 1,
                        # then is_first[c,k] = 0
                        # Equivalently: is_first[c,k] <= 1 - (b[j,k] + p[j_before_c] - 1)
                        #             = 2 - b[j,k] - p[j_before_c]

                        # Get or create precedence variable for other_start before start
                        if (other_start_idx, start_idx) in p:
                            p_j_before_c = p[other_start_idx, start_idx]
                        elif (start_idx, other_start_idx) in p:
                            # p[c, j] exists, so p[j, c] = 1 - p[c, j]
                            # We need a new variable for clean constraint
                            p_j_before_c = LpVariable(
                                f"p_is_first_{other_start_idx}_{start_idx}",
                                cat=LpBinary,
                            )
                            p[other_start_idx, start_idx] = p_j_before_c
                            # Link: p[j,c] = 1 - p[c,j]
                            prob += p_j_before_c + p[start_idx, other_start_idx] == 1
                            # Also need position constraints
                            prob += pi[start_idx] >= pi[other_start_idx] + 1 - M_pos * (1 - p_j_before_c)
                            prob += pi[other_start_idx] >= pi[start_idx] + 1 - M_pos * p_j_before_c
                        else:
                            # Create new precedence variable
                            p_j_before_c = LpVariable(
                                f"p_is_first_{other_start_idx}_{start_idx}",
                                cat=LpBinary,
                            )
                            p[other_start_idx, start_idx] = p_j_before_c
                            prob += pi[start_idx] >= pi[other_start_idx] + 1 - M_pos * (1 - p_j_before_c)
                            prob += pi[other_start_idx] >= pi[start_idx] + 1 - M_pos * p_j_before_c

                        # Constraint: is_first[c,k] <= 2 - b[j,k] - p[j_before_c]
                        prob += (
                            is_first[pg, coll_type, c_idx, k]
                            <= 2 - b[pg, coll_type, j_idx, k] - p_j_before_c,
                            f"is_first_no_before_{pg}_{coll_type}_{c_idx}_{j_idx}_{k}",
                        )

                    # Lower bound: if c is in bucket k and no one before c in bucket k,
                    # then is_first[c,k] >= 1
                    # This is: is_first[c,k] >= b[c,k] - sum_j (b[j,k] * p[j_before_c])
                    # But linearizing products is complex. Since we minimize,
                    # we can rely on the optimizer to set is_first=1 when possible
                    # for cost reduction. Add a simple lower bound:
                    # is_first[c,k] >= b[c,k] - sum_{j!=c} b[j,k]
                    # This ensures if c is the only one in bucket, is_first=1
                    prob += (
                        is_first[pg, coll_type, c_idx, k]
                        >= b[pg, coll_type, c_idx, k]
                        - lpSum(b[pg, coll_type, j_idx, k] for j_idx in range(len(colls)) if j_idx != c_idx),
                        f"is_first_lb_{pg}_{coll_type}_{c_idx}_{k}",
                    )

        # Bucket cost: total cost of all collectives in bucket k
        # (using lpDot for weighted sum, consistent with sac_ilp.py)
        bucket_cost = {}
        for (pg, coll_type), colls in self.pg_type_collectives.items():
            K = len(colls)
            costs = [self.collective_info[coll].estimated_time_ms for coll in colls]
            for k in range(K):
                bucket_cost[pg, coll_type, k] = LpVariable(
                    f"bucket_cost_{pg}_{coll_type}_{k}", lowBound=0, cat=LpContinuous
                )
                # Bucket cost = sum of costs of collectives assigned to this bucket
                b_vars = [b[pg, coll_type, c_idx, k] for c_idx in range(len(colls))]
                prob += (
                    bucket_cost[pg, coll_type, k] == lpDot(costs, b_vars),
                    f"bucket_cost_def_{pg}_{coll_type}_{k}",
                )
        vars_dict["bucket_cost"] = bucket_cost

        # Effective cost for each collective (cost paid when it's first in its bucket)
        eff_cost = {}
        for (pg, coll_type), colls in self.pg_type_collectives.items():
            K = len(colls)
            for c_idx in range(len(colls)):
                eff_cost[pg, coll_type, c_idx] = LpVariable(
                    f"eff_cost_{pg}_{coll_type}_{c_idx}",
                    lowBound=0,
                    cat=LpContinuous,
                )

                # Linearization: eff_cost = sum_k is_first[c,k] * bucket_cost[k]
                # Lower bounds: eff_cost >= bucket_cost[k] - M*(1 - is_first[c,k])
                for k in range(K):
                    prob += (
                        eff_cost[pg, coll_type, c_idx]
                        >= bucket_cost[pg, coll_type, k]
                        - M_time * (1 - is_first[pg, coll_type, c_idx, k]),
                        f"eff_cost_lb_{pg}_{coll_type}_{c_idx}_{k}",
                    )

                # Upper bound: eff_cost <= M * sum_k is_first[c,k]
                is_first_any = lpSum(is_first[pg, coll_type, c_idx, k] for k in range(K))
                prob += (
                    eff_cost[pg, coll_type, c_idx] <= M_time * is_first_any,
                    f"eff_cost_ub_{pg}_{coll_type}_{c_idx}",
                )
        vars_dict["eff_cost"] = eff_cost

        # 7. Initial timing conditions
        prob += T_main[0] == 0, "T_main_init"
        for pg in self.all_pgs:
            prob += T_side[pg, 0] == 0, f"T_side_init_{pg}"

        # 8. Timing transitions
        # This is complex because we need to know what operation is at each position
        # We use big-M constraints to model the transitions

        for t in range(N):
            # For each operation, model its effect when at position t
            for i, node in enumerate(self.nodes):
                runtime = self._get_runtime_estimate(node)

                if is_compute_node(node) or (
                    node.op == "call_function" and node not in self.collective_info
                    and node not in self.wait_to_start
                ):
                    # Compute/other: T_main advances, T_side unchanged
                    prob += (
                        T_main[t + 1] >= T_main[t] + runtime - M_time * (1 - x[i, t]),
                        f"T_main_compute_{i}_{t}",
                    )

                elif node in self.collective_info:
                    # Collective start: nearly free on main, adds to side stream
                    info = self.collective_info[node]
                    pg = info.pg_name
                    coll_type = info.coll_type
                    c_idx = info.coll_idx

                    # Main stream: no cost for async launch
                    prob += (
                        T_main[t + 1] >= T_main[t] - M_time * (1 - x[i, t]),
                        f"T_main_start_{i}_{t}",
                    )

                    # Side stream: add effective cost (bucket cost if first, 0 otherwise)
                    prob += (
                        T_side[pg, t + 1]
                        >= T_side[pg, t] + eff_cost[pg, coll_type, c_idx] - M_time * (1 - x[i, t]),
                        f"T_side_start_{i}_{t}",
                    )

                elif node in self.wait_to_start:
                    # Wait: synchronize main and side streams
                    start = self.wait_to_start[node]
                    info = self.collective_info[start]
                    pg = info.pg_name

                    # Main stream: max of main and side
                    prob += (
                        T_main[t + 1] >= T_main[t] - M_time * (1 - x[i, t]),
                        f"T_main_wait_main_{i}_{t}",
                    )
                    prob += (
                        T_main[t + 1] >= T_side[pg, t] - M_time * (1 - x[i, t]),
                        f"T_main_wait_side_{i}_{t}",
                    )

                    # Side stream syncs to main
                    prob += (
                        T_side[pg, t + 1] >= T_main[t + 1] - M_time * (1 - x[i, t]),
                        f"T_side_wait_{i}_{t}",
                    )

                else:
                    # Other nodes (placeholder, output, etc.)
                    prob += (
                        T_main[t + 1] >= T_main[t] - M_time * (1 - x[i, t]),
                        f"T_main_other_{i}_{t}",
                    )

            # Also need: if nothing uses this slot, time is unchanged
            # This is implicit since we have >= constraints and minimize

        # Side streams: propagate time when no collective on that PG
        for pg in self.all_pgs:
            for t in range(N):
                prob += T_side[pg, t + 1] >= T_side[pg, t], f"T_side_monotone_{pg}_{t}"

        # Main stream monotonicity
        for t in range(N):
            prob += T_main[t + 1] >= T_main[t], f"T_main_monotone_{t}"

        # =================================================================
        # Objective: Minimize total main stream time
        # =================================================================
        prob += T_main[N], "objective"

        return prob, vars_dict

    def _compute_precedence_pairs(self) -> set[tuple[int, int]]:
        """Compute which pairs need precedence variables."""
        pairs = set()

        # DAG edges
        for j, node_j in enumerate(self.nodes):
            for input_node in node_j.all_input_nodes:
                i = self.node_idx[input_node]
                pairs.add((i, j))

        # Same PG collectives (for FIFO)
        for pg in self.all_pgs:
            pg_colls = [
                c for c, info in self.collective_info.items() if info.pg_name == pg
            ]
            for i, coll_i in enumerate(pg_colls):
                for coll_j in pg_colls[i + 1 :]:
                    idx_i = self.node_idx[coll_i]
                    idx_j = self.node_idx[coll_j]
                    pairs.add((min(idx_i, idx_j), max(idx_i, idx_j)))

                    # Also waits
                    wait_i = self.collective_info[coll_i].wait_node
                    wait_j = self.collective_info[coll_j].wait_node
                    widx_i = self.node_idx[wait_i]
                    widx_j = self.node_idx[wait_j]
                    pairs.add((min(widx_i, widx_j), max(widx_i, widx_j)))

        return pairs

    def _extract_solution(
        self, prob: LpProblem, vars_dict: dict[str, Any]
    ) -> ILPSolution:
        """Extract solution from solved ILP problem."""
        x = vars_dict["x"]
        b = vars_dict["b"]
        T_main = vars_dict["T_main"]
        N = len(self.nodes)

        # Extract positions (using .varValue for consistency with sac_ilp.py)
        positions: dict[fx.Node, int] = {}
        for i, node in enumerate(self.nodes):
            for t in range(N):
                val = x[i, t].varValue
                if val is not None and val > 0.5:
                    positions[node] = t
                    break

        # Extract bucket assignments
        bucket_assignments: dict[fx.Node, int] = {}
        for (pg, coll_type), colls in self.pg_type_collectives.items():
            K = len(colls)
            for c_idx, coll in enumerate(colls):
                for k in range(K):
                    val = b[pg, coll_type, c_idx, k].varValue
                    if val is not None and val > 0.5:
                        bucket_assignments[coll] = k
                        break

        total_time = T_main[N].varValue if T_main[N].varValue is not None else 0.0

        return ILPSolution(
            positions=positions,
            bucket_assignments=bucket_assignments,
            total_time=total_time,
            status=LpStatus[prob.status],
        )

    def _apply_schedule(self, solution: ILPSolution) -> None:
        """Apply the ILP solution to reorder the graph."""
        # Sort nodes by their assigned positions
        sorted_nodes = sorted(
            solution.positions.keys(), key=lambda n: solution.positions[n]
        )

        # Track memory for the new schedule order
        for node in sorted_nodes:
            if node.op not in ("placeholder", "output"):
                self.memory_tracker.schedule_node(node)

        log.info(
            "ILP schedule peak memory: %d MB",
            self.memory_tracker.peak_memory // (1024 * 1024),
        )

        # Reorder graph
        output_node = self.graph.output_node()
        for node in sorted_nodes:
            if node.op == "placeholder":
                continue
            output_node.prepend(node)

        self.graph.lint()

    def _apply_bucketing(self, solution: ILPSolution) -> None:
        """Apply bucketing based on ILP solution."""
        from torch._inductor.fx_passes.bucketing import (
            merge_all_gather_bucket,
            merge_all_reduce_bucket,
            merge_reduce_scatter_bucket,
        )

        # Group collectives by bucket
        buckets: dict[tuple[str, str, int], list[fx.Node]] = defaultdict(list)
        for coll, bucket_idx in solution.bucket_assignments.items():
            info = self.collective_info[coll]
            key = (info.pg_name, info.coll_type, bucket_idx)
            buckets[key].append(coll)

        # Sort each bucket by position and apply
        for (pg, coll_type, bucket_idx), colls in buckets.items():
            if len(colls) <= 1:
                continue

            # Sort by position in schedule
            colls.sort(key=lambda c: solution.positions[c])

            log.info(
                "ILP bucketing %d collectives of type %s on pg %s into bucket %d",
                len(colls),
                coll_type,
                pg,
                bucket_idx,
            )

            counters["inductor"]["ilp_collective_buckets"] += 1

            if coll_type == CollectiveType.ALL_GATHER:
                merge_all_gather_bucket(self.graph, colls, mode="custom_ops")
            elif coll_type == CollectiveType.REDUCE_SCATTER:
                merge_reduce_scatter_bucket(self.graph, colls, mode="custom_ops")
            elif coll_type == CollectiveType.ALL_REDUCE:
                merge_all_reduce_bucket(self.graph, colls, mode="custom_ops")

    def run(self) -> torch.fx.GraphModule:
        """Run the ILP scheduler."""
        log.info(
            "ILP Overlap Scheduling: %d nodes, %d compute, %d collectives, %d PGs",
            len(self.nodes),
            len(self.compute_nodes),
            len(self.collective_info),
            len(self.all_pgs),
        )

        if not self.collective_info:
            log.info("No collectives found, skipping ILP scheduling")
            return self.gm

        # Build and solve ILP
        prob, vars_dict = self._build_ilp_problem()

        solver = PULP_CBC_CMD(
            gapRel=self.solver_gap,
            timeLimit=self.solver_time_limit,
            msg=1 if log.isEnabledFor(logging.DEBUG) else 0,
        )

        log.info("Solving ILP problem...")
        status = prob.solve(solver)

        if status != 1:  # Not optimal
            log.warning(
                "ILP solver did not find optimal solution: %s. "
                "Falling back to original order.",
                LpStatus[status],
            )
            counters["inductor"]["ilp_scheduling_failed"] += 1
            return self.gm

        # Extract and apply solution
        solution = self._extract_solution(prob, vars_dict)

        log.info(
            "ILP solution: status=%s, total_time=%.2f ms",
            solution.status,
            solution.total_time,
        )

        # Apply schedule
        self._apply_schedule(solution)

        # Apply bucketing
        self._apply_bucketing(solution)

        counters["inductor"]["ilp_scheduling_success"] += 1
        counters["inductor"]["ilp_scheduled_time_ms"] = int(solution.total_time)
        counters["inductor"]["ilp_scheduled_mem"] = self.memory_tracker.peak_memory

        return self.gm


def schedule_overlap_ilp(
    gm: torch.fx.GraphModule,
    bucket_mode: BucketMode = "custom_ops_multidtype",
    custom_runtime_estimation: Callable[[fx.Node, int | None], float | None]
    | None = None,
    solver_time_limit: int = 300,
    solver_gap: float = 0.05,
    insert_overlap_deps: bool = False,
    enable_fusion_regions: bool = False,
) -> torch.fx.GraphModule:
    """
    Schedule nodes to maximize compute-collective overlap using ILP.

    This is an alternative to schedule_overlap_bucketing that uses Integer
    Linear Programming to find a globally optimal schedule.

    Args:
        gm: Input graph module to optimize.
        bucket_mode: Mode for bucket key computation.
        custom_runtime_estimation: Custom runtime estimation function.
        solver_time_limit: Maximum time for ILP solver in seconds.
        solver_gap: Acceptable optimality gap for solver.
        insert_overlap_deps: Whether to insert overlap dependencies.
        enable_fusion_regions: Build and collapse fusion regions before scheduling.

    Returns:
        Optimized graph module.
    """
    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": "ilp_overlap_scheduling_graph_before",
            "encoding": "string",
        },
        payload_fn=lambda: gm.print_readable(False),
    )

    scheduler = ILPOverlapScheduler(
        gm,
        bucket_mode=bucket_mode,
        custom_runtime_estimation=custom_runtime_estimation,
        solver_time_limit=solver_time_limit,
        solver_gap=solver_gap,
        insert_overlap_deps=insert_overlap_deps,
        enable_fusion_regions=enable_fusion_regions,
    )
    result = scheduler.run()

    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": "ilp_overlap_scheduling_graph_after",
            "encoding": "string",
        },
        payload_fn=lambda: result.print_readable(False),
    )

    return result
