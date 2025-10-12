import functools
import heapq
import itertools
import logging
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import torch
import torch.fx as fx
from torch._dynamo.utils import counters, dynamo_timed
from torch._inductor.fx_passes.bucketing import is_wait_tensor
from torch._inductor.fx_passes.memory_estimator import MemoryTracker
from torch._inductor.fx_passes.subgraph_analyzer import SubgraphAnalyzer, SchedulableSubgraph
from torch.utils._mode_utils import no_dispatch
from torch.utils._ordered_set import OrderedSet


log = logging.getLogger(__name__)

from ..pattern_matcher import stable_topological_sort


def get_custom_estimation(n: fx.Node) -> Optional[float]:
    runtime_estimation = torch._inductor.config.test_configs.estimate_aten_runtime
    if runtime_estimation == "default":
        return None

    assert callable(runtime_estimation)
    return runtime_estimation(n)


def estimate_collective_time(n: fx.Node) -> float:
    if (est := get_custom_estimation(n)) is not None:
        return est

    return torch._inductor.comm_analysis.estimate_nccl_collective_runtime_from_fx_node(
        n
    )


def estimate_fx_collective_size(fx_node: torch.fx.Node) -> int:
    size = 0
    for node in fx_node.all_input_nodes:
        if (t := node.meta.get("val")) is not None:
            # todo - symbolic
            size += t.numel() * t.element_size()

    return size


def is_compute_node(n: fx.Node) -> bool:
    return (
        getattr(n.target, "overloadpacket", None)
        in torch.utils.flop_counter.flop_registry
    )


def get_hint(x: Union[int, torch.SymInt]) -> Optional[int]:
    if isinstance(x, int):
        return x
    assert isinstance(x, torch.SymInt)
    if not x.node.has_hint():
        return None
    return x.node.hint


def get_collective_do_bench() -> Callable[[Callable[[], Any]], float]:
    with dynamo_timed("collective_compute_do_bench"):
        return functools.partial(
            torch._inductor.runtime.benchmarking.benchmarker.benchmark_gpu,
            warmup=5,
        )


def benchmark_node_with_cache_key(n: fx.Node) -> tuple[float, Optional[str]]:
    assert is_compute_node(n)

    from torch._dynamo.testing import rand_strided

    # todo - skip unbacked, symbolic
    success, args, kwargs = torch._inductor.fx_utils.get_fake_args_kwargs(n)

    if not success:
        return 0, None

    unbacked_tensor = False

    key = f"{str(n.target)}: "

    def to_real(t: torch.Tensor) -> Optional[torch.Tensor]:
        shape = [get_hint(dim) for dim in t.shape]
        stride = [get_hint(s) for s in t.stride()]

        if any(s is None for s in itertools.chain(shape, stride)):
            nonlocal unbacked_tensor
            unbacked_tensor = True
            return None

        nonlocal key
        key += f"T: {shape, stride, t.dtype} "
        return rand_strided(shape, stride, device=t.device, dtype=t.dtype)  # type: ignore[arg-type]

    with no_dispatch():
        args, kwargs = torch.utils._pytree.tree_map_only(
            torch.Tensor,
            lambda t: to_real(t),
            (args, kwargs),
        )

        if val := get_cached_node_time(key):
            return val, key

        if unbacked_tensor:
            return 0, key

        if (est := get_custom_estimation(n)) is not None:
            set_cached_node_time(key, est)
            return est, key

        bench = get_collective_do_bench()
        out = bench(lambda: n.target(*args, **kwargs))  # type: ignore[operator]
        set_cached_node_time(key, out)
        return out, key


def benchmark_node(n: fx.Node) -> float:
    return benchmark_node_with_cache_key(n)[0]


@functools.cache
def get_benchmark_cache() -> torch._inductor.codecache.LocalCache:
    return torch._inductor.codecache.LocalCache()


def get_cached_node_time(key: str) -> float:
    return get_benchmark_cache().lookup(key)  # type: ignore[return-value]


def set_cached_node_time(key: str, value: float) -> None:
    return get_benchmark_cache().set_value(key, value=value)


@dataclass
class CollectiveInfo:
    """Track info about a collective operation"""

    start_node: fx.Node
    wait_node: fx.Node
    size_bytes: int
    estimated_time_ms: float
    exposed_time_ms: float  # How much of this collective is still exposed
    hiding_node: Optional[fx.Node] = None  # Node that hides this collective

    @property
    def is_exposed(self) -> bool:
        return self.exposed_time_ms != 0


@dataclass
class CollBucket:
    """Track information about a bucket of collectives."""

    collectives: list[fx.Node]  # Original collective starts
    bucketed_start: Optional[fx.Node] = None  # After bucketing
    bucketed_wait: Optional[fx.Node] = None  # After bucketing
    total_bytes: int = 0


class OverlapScheduler:
    """
    Scheduler that reorders operations to maximize compute-collective overlap.

    The reordering is done as a scheduling pass. We maintain a priority queue of
    schedulable nodes. The nodes are ranked by:

    1) the compute node depth they dominate. this allows reordering locally, such as with
    parallel mms, and also allows overlapping reduce scatter nodes outputs in the backward
    with compute by deferring their waits.

    2) whether the current node is a collective or wait that is currently exposed but has a compute
    node which it could be overlapped with.

    3) original order in the graph for stability.

    When we schedule compute nodes, we first overlap exposed in-flight collectives, then look for unscheduled
    collectives that can be scheduled concurrently.

    TODO:
        - experiment with other priority scores / allow other mechanisms of reorder / more strict adherence to original graph
        - memory limit for deferred scheduling of reduce_scatter nodes.
    """

    def __init__(
        self,
        gm: torch.fx.GraphModule,
        max_in_flight_gb: float = 2.0,
        compute_overlap_multipler: float = 2.0,
        max_coll_distance: int = 1000,
    ):
        self.gm = gm
        self.graph = gm.graph
        self.compute_overlap_multipler = compute_overlap_multipler
        self.max_node_distance = max_coll_distance
        self.max_in_flight_bytes: int = int(max_in_flight_gb * 1024 * 1024 * 1024)

        # Build structures
        stable_topological_sort(self.graph)
        self.nodes = list(self.graph.nodes)
        self.node_idx = {n: i for i, n in enumerate(self.nodes)}
        self.node_ancestors: dict[fx.Node, OrderedSet[fx.Node]] = (
            self._collect_node_ancestors()
        )

        # Identify collectives and compute nodes
        self.collective_info: dict[fx.Node, CollectiveInfo] = {}
        self.unscheduled_collectives: OrderedSet[fx.Node] = OrderedSet()

        self.wait_to_start: dict[fx.Node, fx.Node] = {}
        self._identify_collectives()

        self.compute_depth = self._calculate_compute_node_depth()
        self.compute_nodes = [n for n in self.nodes if is_compute_node(n)]

        # Memory tracking using abstracted MemoryTracker
        def device_filter(device):
            return device.type != "cpu"

        self.memory_tracker = MemoryTracker(self.graph, device_filter=device_filter)

        # Subgraph analyzer for finding memory-reducing subgraphs
        self.subgraph_analyzer = SubgraphAnalyzer(self.memory_tracker)

        # Scheduling state
        self.potentially_hidden_collectives = (
            self.compute_potential_hidden_collectives()
        )
        self.potentially_hidden_waits = self.compute_potential_hidden_waits()
        self.in_degree = Counter(user for node in self.nodes for user in node.users)
        self.ready: list[tuple[object, fx.Node]] = []

        for node in self.nodes:
            if self.in_degree[node] == 0:
                heapq.heappush(self.ready, (self._compute_score(node), node))

        self.in_flight: dict[fx.Node, CollectiveInfo] = {}  # start -> info
        self.in_flight_bytes = 0
        self.scheduled: OrderedSet[fx.Node] = OrderedSet()

        # Partial subgraph tracking
        self.partial_subgraphs: list[SchedulableSubgraph] = []  # Chains with remaining nodes to schedule

    def _collect_node_ancestors(self) -> dict[fx.Node, OrderedSet[fx.Node]]:
        """Collect all ancestors for each node."""
        ancestors: dict[fx.Node, OrderedSet[fx.Node]] = defaultdict(OrderedSet)
        for node in self.nodes:
            for input_node in node.all_input_nodes:
                ancestors[node].add(input_node)
                ancestors[node] |= ancestors[input_node]

        return ancestors

    def _identify_collectives(self) -> None:
        """Identify all collective operations."""
        for node in self.nodes:
            if is_wait_tensor(node):
                start = node.args[0]
                coll_time_ms = estimate_collective_time(start)

                info = CollectiveInfo(
                    start_node=start,
                    wait_node=node,
                    size_bytes=estimate_fx_collective_size(start),
                    estimated_time_ms=coll_time_ms,
                    exposed_time_ms=coll_time_ms,  # Initially fully exposed
                )
                self.collective_info[start] = info
                self.wait_to_start[node] = start
                self.unscheduled_collectives.add(start)

    def _calculate_compute_node_depth(self) -> dict[fx.Node, int]:
        """Compute forward depth and minimum dominance depth (infinity if blocks no compute)."""

        # First pass: forward compute depth
        in_degree: dict[fx.Node, int] = {}
        compute_depth: dict[fx.Node, int] = {}
        queue: list[fx.Node] = []

        for node in self.graph.nodes:
            num_inputs = len(node.all_input_nodes)
            if num_inputs == 0:
                queue.append(node)
            else:
                in_degree[node] = num_inputs

        while queue:
            node = queue.pop()

            max_input_depth = max(
                (compute_depth[inp] for inp in node.all_input_nodes), default=0
            )
            compute_depth[node] = max_input_depth + is_compute_node(node)

            for use in node.users:
                in_degree[use] -= 1
                if in_degree[use] == 0:
                    queue.append(use)

        # Second pass: minimum dominance (what's the earliest compute this blocks)
        compute_depth_dominance: dict[fx.Node, int] = {}

        for node in reversed(self.graph.nodes):
            if is_compute_node(node):
                # consider compute nodes to be at their own depth
                dominance = compute_depth[node]
            else:
                # For non-compute nodes, find minimum compute they block
                dominance = min(
                    (compute_depth_dominance[succ] for succ in node.users),
                    default=sys.maxsize,
                )

            compute_depth_dominance[node] = dominance

        return compute_depth_dominance

    def _align_compute_nodes_runtime_estimations_across_all_distributed_ranks(
        self,
    ) -> None:
        import torch.distributed as dist

        # Skip alignment if using fake backend (for testing)
        if hasattr(dist, 'get_backend') and dist.get_backend() == 'fake':
            log.info("Overlap scheduling: Skipping runtime estimation alignment for fake backend")
            # Still need to benchmark nodes for individual estimation
            for n in self.compute_nodes:
                benchmark_node_with_cache_key(n)
            return

        log.info(
            "Overlap scheduling: Aligning runtime estimations across all distributed ranks"
        )
        runtime_estimations_keys: list[Optional[str]] = []
        runtime_estimations: list[float] = []
        for n in self.compute_nodes:
            val, key = benchmark_node_with_cache_key(n)
            runtime_estimations.append(val)
            runtime_estimations_keys.append(key)

        from torch.distributed.distributed_c10d import _get_default_group

        world_size = dist.get_world_size()
        pg = _get_default_group()
        with no_dispatch():
            gathered_runtime_estimations: list[list[float]] = [
                [] for _ in range(world_size)
            ]
            dist.all_gather_object(
                gathered_runtime_estimations, runtime_estimations, pg
            )
            median_runtime_estimations = torch.median(
                torch.tensor(gathered_runtime_estimations), dim=0
            ).values.tolist()
        for key, median_runtime_estimation in zip(
            runtime_estimations_keys, median_runtime_estimations
        ):
            if key is None:
                continue
            set_cached_node_time(key, median_runtime_estimation)
        log.info(
            "Overlap scheduling: Runtime estimations across all distributed ranks were aligned"
        )

    def run(self) -> torch.fx.GraphModule:
        """Run the scheduling algorithm."""
        # All ranks must make identical decisions on overlap reordering,
        # Thus we must have identical runtime estimations across ranks.
        # For now we do benchmarking only for compute nodes.
        self._align_compute_nodes_runtime_estimations_across_all_distributed_ranks()

        while self.ready:
            if self._should_force_wait_for_memory():
                # Priority 1: Force hidden waits first (no overlap cost)
                if self._force_hidden_waits():
                    continue

                # Priority 2: No additional memory-reducing logic needed here
                # (subgraphs are built naturally in main loop from non-critical nodes)

                # Fallback: Force any remaining wait
                if self.in_flight:
                    self._force_oldest_wait()
                continue

            _, node = heapq.heappop(self.ready)

            # we don't always remove nodes from the heap when we schedule them
            if node in self.scheduled:
                continue

            # Try to build memory-reducing subgraph from this node if it's non-critical
            if self._try_schedule_memory_reducing_subgraph_from(node):
                continue

            if is_compute_node(node):
                self._handle_compute(node)
            elif node in self.collective_info:
                self._handle_collective_start(node)
            elif is_wait_tensor(node):
                self._handle_wait(node)
            else:
                self._handle_other(node)

        self._reorder_graph()

        if torch._inductor.config.test_configs.aten_fx_overlap_preserving_bucketing:
            self._bucket_collectives()
        elif torch._inductor.config.test_configs.aten_fx_overlap_insert_overlap_deps:
            # If not bucketing, add effect tokens to preserve hiding dependencies
            self._add_effect_tokens_for_overlap()

        return self.gm

    def _add_effect_tokens_for_overlap(self) -> None:
        """
        Add effect tokens to preserve hiding dependency relationships when not bucketing.

        This ensures that communication-compute overlap is preserved through effect tokens
        when overlap preserving bucketing is not enabled.
        """
        from torch._inductor.fx_passes.control_dependencies import (
            preserve_node_ordering,
        )

        # Collect hiding dependencies: hiding_node -> collective_start, wait -> hiding_node
        additional_deps: dict[fx.Node, OrderedSet[fx.Node]] = defaultdict(OrderedSet)

        for start_node, info in self.collective_info.items():
            if info.hiding_node and not info.is_exposed:
                # Compute depends on collective start (compute must wait for collective to start)
                additional_deps[info.hiding_node].add(start_node)
                # Wait depends on compute (wait must wait for compute to finish)
                additional_deps[info.wait_node].add(info.hiding_node)

        # Apply effect tokens to preserve these dependencies
        if additional_deps:
            preserve_node_ordering(self.graph, additional_deps)

    def _handle_other(self, node: fx.Node) -> None:
        self._schedule(node)

    def _schedule(self, node: fx.Node) -> None:
        """Schedule a node."""
        assert node not in self.scheduled
        assert all(n in self.scheduled for n in node.all_input_nodes)

        # Update memory tracking and mark as scheduled
        self.memory_tracker.schedule_node(node)
        self.scheduled.add(node)

        # Notify subgraph analyzer for cache invalidation
        self.subgraph_analyzer.notify_node_scheduled(node)

        # Update partial subgraphs - mark this node as consumed
        self._update_partial_subgraphs(node)

        # Check for newly unblocked subgraphs and merge them optimally
        self._handle_unblocked_subgraphs()

        for user in node.users:
            self.in_degree[user] -= 1
            if self.in_degree[user] == 0:
                heapq.heappush(self.ready, (self._compute_score(user), user))

    def _compute_score(self, node: fx.Node) -> object:
        """Compute priority score for a node"""

        if is_wait_tensor(node):
            info = self.collective_info[self.wait_to_start[node]]
            # TODO: we could consider even deferring waits that are not potentially hidden
            # so as to overlap comm with itself. although exposed comms should bucketed with each other.
            overlappable = info.is_exposed and node in self.potentially_hidden_waits
        else:
            overlappable = self.in_overlappable_collective_unary_subgraph(node)

        return (
            self.compute_depth[node],  # what depth compute it blocks
            overlappable,  # Defer hideable collective ops
            self.node_idx[node],  # Original order for stability
        )

    @staticmethod
    def is_cheap_fn(node: fx.Node) -> bool:
        return getattr(node.target, "is_view", False) or torch.Tag.pointwise in getattr(
            node.target, "tags", ()
        )

    def in_overlappable_collective_unary_subgraph(self, curr: fx.Node) -> bool:
        while True:
            if len(curr.users) != 1:
                return False

            user = next(iter(curr.users))
            if len(user.all_input_nodes) != 1:
                return False

            if user in self.unscheduled_collectives:
                return user in self.potentially_hidden_collectives

            if not self.is_cheap_fn(user):
                return False

            curr = user

        return False

    def _should_force_wait_for_memory(self) -> bool:
        """Check if we need to force a wait due to memory pressure"""
        # Only force wait if we have in-flight operations to wait for
        if not self.in_flight:
            return False

        # Check both in-flight collective bytes and total tracked memory
        collective_pressure = self.in_flight_bytes >= self.max_in_flight_bytes

        # More aggressive memory pressure detection (lower threshold)
        current_memory_gb = self.memory_tracker.get_current_memory_bytes() / (1024**3)
        memory_limit_gb = self.max_in_flight_bytes / (1024**3)
        memory_pressure = current_memory_gb >= (memory_limit_gb * 0.75)  # 75% threshold instead of 100%

        return collective_pressure or memory_pressure

    def _force_oldest_wait(self) -> None:
        """Schedule the oldest in flight wait"""
        self._handle_wait(self._get_oldest_wait())

    def _force_hidden_waits(self) -> bool:
        """
        Force waits on hidden collectives (no overlap cost).

        Returns True if any hidden waits were scheduled, False otherwise.
        """
        forced_any = False

        # Look for hidden waits we can force without losing overlap
        for start_node, info in list(self.in_flight.items()):
            if not info.is_exposed and info.hiding_node:
                # This collective is already hidden, force its wait
                self._handle_wait(info.wait_node)
                forced_any = True

        return forced_any


    def _try_schedule_memory_reducing_subgraph_from(self, node: fx.Node) -> bool:
        """
        Try to find and schedule a memory-reducing subgraph starting from a specific node.

        Returns True if we scheduled a subgraph, False if we should schedule node normally.
        """
        # Try building subgraphs from non-critical path nodes, but also be more aggressive
        # when memory pressure is high
        current_memory_gb = self.memory_tracker.get_current_memory_bytes() / (1024**3)
        memory_limit_gb = self.max_in_flight_bytes / (1024**3)
        memory_usage_ratio = current_memory_gb / memory_limit_gb

        # Only build subgraphs from non-critical path nodes to avoid scheduling violations
        if self.compute_depth[node] != sys.maxsize:
            return False

        # More generous memory headroom calculation - allow temporary spikes
        available_memory_bytes = max(
            int(memory_limit_gb * 0.2 * 1024**3),  # At least 20% of limit
            int((memory_limit_gb - current_memory_gb + memory_limit_gb * 0.1) * 1024**3)  # Current + 10% buffer
        )

        # Filter function: avoid nodes that would block compute
        def is_schedulable(n: fx.Node) -> bool:
            # Allow wait tensors (they free memory)
            if is_wait_tensor(n):
                return True
            # Allow non-blocking nodes
            if self.compute_depth[n] == sys.maxsize:
                return True
            # Don't include compute nodes or collective starts in subgraphs
            return False

        subgraph = self.subgraph_analyzer.find_subgraph_from_node(
            start_node=node,
            scheduled=self.scheduled,
            available_memory=available_memory_bytes,
            is_schedulable_fn=is_schedulable,
        )

        if subgraph:
            log.debug(f"Scheduling memory-reducing subgraph: {len(subgraph.nodes)} nodes, "
                     f"peak +{subgraph.peak_increase // (1024*1024)}MB, "
                     f"net {subgraph.net_change // (1024*1024)}MB")

            for subgraph_node in subgraph.nodes:
                self._schedule(subgraph_node)
            return True

        return False

    def _handle_collective_start(self, node: fx.Node) -> None:
        """Handle scheduling a collective start."""
        info = self.collective_info[node]
        self.in_flight[node] = info
        self.in_flight_bytes += info.size_bytes
        self.unscheduled_collectives.discard(node)
        self._schedule(node)

    def _handle_wait(self, node: fx.Node) -> None:
        """Handle scheduling a wait."""
        assert node in self.wait_to_start
        coll_start = self.wait_to_start[node]

        assert coll_start in self.in_flight
        self.in_flight_bytes -= self.in_flight[coll_start].size_bytes
        del self.in_flight[coll_start]
        self._schedule(node)

    def _handle_compute(self, node: fx.Node) -> None:
        """Handle scheduling compute and finding overlaps."""

        compute_time = benchmark_node(node)
        available_compute = compute_time * self.compute_overlap_multipler

        # First reduce exposed time of in-flight collectives
        for info in self.in_flight.values():
            if info.exposed_time_ms == 0:
                continue
            overlap_amount = min(info.exposed_time_ms, available_compute)
            info.exposed_time_ms -= overlap_amount
            available_compute -= overlap_amount
            if info.exposed_time_ms == 0:
                info.hiding_node = node
            elif available_compute == 0:
                break

        # Then, look for unscheduled collectives we can overlap
        if available_compute:
            self._schedule_collectives_for_overlap(node, available_compute)

        self._schedule(node)

    def _schedule_collectives_for_overlap(
        self, compute_node: fx.Node, available_compute_time: float
    ) -> None:
        """Opportunistically schedule collectives that can be hidden by compute."""
        compute_ancestors = self.node_ancestors[compute_node]

        # copy unscheduled_collectives to local because we modify it during iteration
        possible_collectives = []
        for collective in self.unscheduled_collectives:
            distance = abs(self.node_idx[compute_node] - self.node_idx[collective])
            if distance > self.max_node_distance:
                break

            possible_collectives.append(collective)

        for collective in possible_collectives:
            if available_compute_time == 0:
                break

            info = self.collective_info[collective]

            # Skip if compute depends on collective or vice versa
            if (
                collective in compute_ancestors
                or compute_node in self.node_ancestors[collective]
            ):
                continue

            # Check memory pressure before scheduling more collectives
            current_memory_gb = self.memory_tracker.get_current_memory_bytes() / (1024**3)
            memory_limit_gb = self.max_in_flight_bytes / (1024**3)
            if current_memory_gb > memory_limit_gb * 0.8:  # 80% memory threshold
                log.debug(f"Skipping collective scheduling due to memory pressure: {current_memory_gb:.2f}GB")
                break

            while (
                self.in_flight
                and (self.max_in_flight_bytes - self.in_flight_bytes) < info.size_bytes
                and self._wait_is_hidden(self._get_oldest_wait(), compute_node)
            ):
                self._force_oldest_wait()

            if (self.max_in_flight_bytes - self.in_flight_bytes) < info.size_bytes:
                continue

            # Check if we can reach this collective without scheduling compute, other collectives, or waits
            path = self._find_schedulable_path(collective, compute_node)
            if path is None:
                continue

            # Schedule path to this collective
            self._schedule_path_to_collective(path, compute_node)
            # Update the exposed time for this newly scheduled collective
            overlap_amount = min(info.estimated_time_ms, available_compute_time)
            info.exposed_time_ms -= overlap_amount
            if info.exposed_time_ms == 0:
                info.hiding_node = compute_node
            available_compute_time -= overlap_amount
            self._handle_collective_start(collective)

    def _find_schedulable_path(
        self, target: fx.Node, curr_compute_node: Optional[fx.Node]
    ) -> Optional[OrderedSet[fx.Node]]:
        """Find path to target by collecting unscheduled dependencies."""

        # TODO - following path faster than doing set difference here
        unscheduled_ancestors = self.node_ancestors[target] - self.scheduled

        # only schedule non distributed, non compute nodes
        for node in unscheduled_ancestors:
            if is_compute_node(node):
                return None

            if node in self.unscheduled_collectives:
                return None

            # if we schedule a wait tensor whose start collective is hidden by the
            # current compute node we are scheduling, then we are effectively exposing it.
            # similarly, dont schedule a wait of a collective that could be otherwise hidden,
            # thus forcing it to be exposed.
            # however, if it is already hidden or it cannot be possible hidden,
            # it's fine to schedule it
            if is_wait_tensor(node):
                info = self.collective_info[self.wait_to_start[node]]
                if info.hiding_node and info.hiding_node != curr_compute_node:
                    continue
                elif node not in self.potentially_hidden_waits:
                    continue

                return None

        return unscheduled_ancestors

    def _get_oldest_wait(self) -> fx.Node:
        oldest_start = next(iter(self.in_flight))
        return self.collective_info[oldest_start].wait_node

    def _wait_is_hidden(
        self, wait_node: fx.Node, compute_node: Optional[fx.Node] = None
    ) -> bool:
        assert is_wait_tensor(wait_node)
        info = self.collective_info[self.wait_to_start[wait_node]]
        return not info.is_exposed and info.hiding_node != compute_node

    def _schedule_path_to_collective(
        self, path: OrderedSet[fx.Node], curr_compute_node: fx.Node
    ) -> None:
        """Schedule all nodes needed to reach a collective."""
        for node in sorted(path, key=lambda n: self.node_idx[n]):
            assert not (is_compute_node(node) or node in self.unscheduled_collectives)

            if is_wait_tensor(node):
                info = self.collective_info[self.wait_to_start[node]]
                assert info.hiding_node != curr_compute_node
                self._handle_wait(node)
                continue

            self._schedule(node)

    def reorder_graph(self) -> None:
        output_node = self.graph.output_node()
        for node in self.scheduled:
            if node.op == "placeholder":
                continue
            output_node.prepend(node)
        self.graph.lint()

    def _reorder_graph(self) -> None:
        """Reorder graph based on schedule."""
        exposed = [
            c
            for c in self.collective_info.values()
            if c.exposed_time_ms == c.estimated_time_ms
        ]

        potentially_hidden_collectives = self.compute_potential_hidden_collectives(
            limit_coll_per_compute=True
        )
        bad_exposed = [
            c for c in exposed if c.start_node in potentially_hidden_collectives
        ]

        counters["inductor"]["overlap_scheduling_exposed"] += len(exposed)
        counters["inductor"]["overlap_scheduling_bad_exposed"] += len(bad_exposed)
        counters["inductor"]["overlap_scheduling_potentially_hidden"] += len(
            potentially_hidden_collectives
        )

        log.info(
            "Overlap scheduling: total exposed %s, total bad exposed %s, total potentially hidden %s",
            len(exposed),
            len(bad_exposed),
            len(potentially_hidden_collectives),
        )

        self.reorder_graph()

    def _bucket_collectives(self) -> None:
        from torch._inductor.fx_passes.overlap_preserving_bucketer import (
            OverlapPreservingBucketer,
        )

        bucketer = OverlapPreservingBucketer(
            graph=self.graph,
            collective_info=self.collective_info,
            node_ancestors=self.node_ancestors,
            scheduled=self.scheduled,
            max_bucket_memory_gb=1.0,  # Could make this configurable
        )
        bucketer.bucket_collectives()

    def compute_potential_hidden_nodes(
        self, nodes_to_check: Iterable[fx.Node], limit_coll_per_compute: bool = False
    ) -> dict[fx.Node, fx.Node]:
        """
        Returns a dict containing a mapping of nodes which could potentially be hidden to their hiding node
        """

        used_compute_nodes: OrderedSet[fx.Node] = OrderedSet()

        def could_be_hidden(start: fx.Node) -> Optional[fx.Node]:
            for compute_node in self.compute_nodes:
                if limit_coll_per_compute and compute_node in used_compute_nodes:
                    continue
                if (
                    start not in self.node_ancestors[compute_node]
                    and compute_node not in self.node_ancestors[start]
                ):
                    if limit_coll_per_compute:
                        used_compute_nodes.add(compute_node)
                    return compute_node

            return None

        # TODO: We could potentially limit compute nodes per overlap time,
        # today, this is optimistic, and just serves to avoid deferring
        # collectives/waits that have no possible overlap as well as for analysis of how
        # successfully we hid compute
        potentially_hidden = {}
        for node in nodes_to_check:
            if mm := could_be_hidden(node):
                potentially_hidden[node] = mm

        return potentially_hidden

    def compute_potential_hidden_collectives(
        self, limit_coll_per_compute: bool = False
    ) -> dict[fx.Node, fx.Node]:
        """Compute which collective operations could be hidden by compute."""
        return self.compute_potential_hidden_nodes(
            self.collective_info.keys(), limit_coll_per_compute
        )

    def compute_potential_hidden_waits(
        self, limit_coll_per_compute: bool = False
    ) -> dict[fx.Node, fx.Node]:
        """Compute which wait operations could be hidden by compte."""
        wait_nodes = [info.wait_node for info in self.collective_info.values()]
        return self.compute_potential_hidden_nodes(wait_nodes, limit_coll_per_compute)

    def _update_partial_subgraphs(self, scheduled_node: fx.Node) -> None:
        """Update partial subgraphs when a node is scheduled."""
        for subgraph in self.partial_subgraphs:
            if scheduled_node in subgraph.nodes and scheduled_node not in subgraph.consumed_nodes:
                subgraph.consumed_nodes.add(scheduled_node)

                # Recalculate remaining net change (simple difference)
                scheduled_portion_net = self._calculate_net_memory_for_nodes([scheduled_node])
                subgraph.remaining_net_change -= scheduled_portion_net

    def _handle_unblocked_subgraphs(self) -> None:
        """Check for newly unblocked cached subgraphs and merge them optimally."""
        unblocked_subgraphs = self.subgraph_analyzer.get_unblocked_subgraphs(self.scheduled)

        if not unblocked_subgraphs:
            return

        # Merge unblocked subgraphs optimally based on memory characteristics
        if len(unblocked_subgraphs) > 1:
            merged_subgraph = self.subgraph_analyzer.merge_subgraphs_optimally(unblocked_subgraphs)
            self.partial_subgraphs.append(merged_subgraph)
        else:
            self.partial_subgraphs.extend(unblocked_subgraphs)

    def _schedule_subgraph_up_to_collective(self, subgraph: SchedulableSubgraph) -> bool:
        """
        Schedule nodes from a subgraph up to the first collective operation.

        Returns True if any nodes were scheduled, False otherwise.
        """
        nodes_to_schedule = subgraph.get_nodes_up_to_collective()

        if not nodes_to_schedule:
            return False

        # Check if all nodes are ready to be scheduled
        for node in nodes_to_schedule:
            if not all(inp in self.scheduled for inp in node.all_input_nodes):
                return False

        # Schedule the nodes
        scheduled_any = False
        for node in nodes_to_schedule:
            if node not in self.scheduled:
                if node in self.collective_info:
                    self._handle_collective_start(node)
                else:
                    self._schedule(node)
                scheduled_any = True

        return scheduled_any

    def _schedule_remaining_subgraph_nodes(self, subgraph: SchedulableSubgraph) -> bool:
        """
        Schedule remaining nodes from a subgraph (typically after collectives are hidden).

        Returns True if any nodes were scheduled, False otherwise.
        """
        remaining_nodes = subgraph.remaining_nodes

        if not remaining_nodes:
            return False

        # Check if all remaining nodes are ready
        for node in remaining_nodes:
            if not all(inp in self.scheduled for inp in node.all_input_nodes):
                return False

        # Schedule remaining nodes
        scheduled_any = False
        for node in remaining_nodes:
            if node not in self.scheduled:
                self._schedule(node)
                scheduled_any = True

        return scheduled_any

    def _calculate_net_memory_for_nodes(self, nodes: list[fx.Node]) -> int:
        """Calculate net memory change for a list of nodes."""
        # Simple estimation - could be enhanced with actual memory simulation
        total_allocated = 0
        total_freed = 0

        for node in nodes:
            # Get fresh allocations
            fresh_allocations = self.memory_tracker.alias_tracker.get_fresh_allocations(node)
            allocated = sum(
                self.memory_tracker._get_storage_size(sk) for sk in fresh_allocations
                if self.memory_tracker.device_filter(sk.device)
            )

            # Estimate freed memory (simplified)
            input_storages = self.memory_tracker.alias_tracker.get_storage_uses(node)
            freed = 0
            for storage_key in input_storages:
                all_uses = self.memory_tracker.alias_tracker.storage_to_uses[storage_key]
                unscheduled_uses = all_uses - self.scheduled

                # If this is the last unscheduled use, count as freed
                if len(unscheduled_uses) == 1 and node in unscheduled_uses:
                    freed += self.memory_tracker._get_storage_size(storage_key)

            total_allocated += allocated
            total_freed += freed

        return total_allocated - total_freed  # Positive = net allocation, negative = net release


def schedule_overlap_bucketing(
    gm: torch.fx.GraphModule,
    max_in_flight_gb: float = 2.0,
    compute_overlap_multipler: float = 1.0,
    max_coll_distance: int = 1000,
) -> torch.fx.GraphModule:
    """Schedule nodes to maximize compute-collective overlap.

    Args:
        gm: Input graph module to optimize.
        max_in_flight_gb: Maximum GB of concurrent collective data.
        compute_overlap_multipler: Scale factor for compute time used to hide collectives.
        max_coll_distance: Maximum node distance for overlap consideration.
    """
    return OverlapScheduler(
        gm,
        compute_overlap_multipler=compute_overlap_multipler,
        max_in_flight_gb=max_in_flight_gb,
        max_coll_distance=max_coll_distance,
    ).run()