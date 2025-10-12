"""
Subgraph analysis for memory-aware scheduling.

Finds independent schedulable subgraphs that reduce memory usage.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional
from collections import deque

import torch.fx as fx
from torch._inductor.fx_passes.bucketing import is_wait_tensor
from torch._inductor.fx_passes.memory_estimator import MemoryTracker
from torch.utils._ordered_set import OrderedSet


def is_collective_start(node: fx.Node) -> bool:
    return len(node.users) == 1 and is_wait_tensor(next(iter(node.users)))


@dataclass
class SchedulableSubgraph:
    """A schedulable subgraph of nodes with memory characteristics."""

    nodes: list[fx.Node]
    peak_increase: int  # Max memory increase during execution (bytes)
    net_change: int    # Net memory change at end (bytes, negative = memory released)
    collective_starts: list[fx.Node]  # Collective operations scheduled in this subgraph

    def schedule_partial(self, nodes_to_schedule: list[fx.Node], memory_tracker: MemoryTracker):
        """Schedule part of this subgraph and update to remaining slice."""
        # Remove scheduled nodes from our lists
        remaining_nodes = [n for n in self.nodes if n not in nodes_to_schedule]
        remaining_collectives = [c for c in self.collective_starts if c not in nodes_to_schedule]

        # Update our state to the remaining slice
        self.nodes = remaining_nodes
        self.collective_starts = remaining_collectives

        # Recalculate metrics for remaining nodes
        if remaining_nodes:
            self.peak_increase, self.net_change = memory_tracker.simulate_subgraph_memory(remaining_nodes)
        else:
            self.peak_increase, self.net_change = 0, 0

    def get_nodes_up_to_collective(self) -> list[fx.Node]:
        """Get nodes from start up to first collective (inclusive)."""
        collective_indices = [i for i, node in enumerate(self.nodes) if node in self.collective_starts]

        if collective_indices:
            return self.nodes[:collective_indices[0] + 1]
        else:
            return self.nodes



class SubgraphAnalyzer:
    """Finds memory-reducing subgraphs of schedulable nodes."""

    def __init__(self, memory_tracker: MemoryTracker):
        self.memory_tracker = memory_tracker
        self.cached_subgraphs: dict[fx.Node, list[SchedulableSubgraph]] = {}

    def find_subgraph_from_node(
        self,
        start_node: fx.Node,
        scheduled: OrderedSet[fx.Node],
        available_memory: int,
        is_schedulable_fn: Callable[[fx.Node], bool],
    ) -> Optional[SchedulableSubgraph]:
        """
        Find memory-reducing subgraph starting from a specific node.
        """
        if start_node in scheduled or not is_schedulable_fn(start_node):
            return None

        # First try to extend from cached exploration results
        subgraph = self._try_extend_from_cache(start_node, scheduled, is_schedulable_fn)

        # If no cached extension possible, build from scratch
        if not subgraph:
            subgraph = self._build_subgraph_from(start_node, scheduled, is_schedulable_fn)

        if not subgraph:
            return None

        # Only return subgraphs that fit in available memory and reduce net memory
        if subgraph.peak_increase <= available_memory and subgraph.net_change < 0:
            return subgraph

        return None

    def notify_node_scheduled(self, node: fx.Node):
        """Notify the analyzer that a node has been scheduled (for cache invalidation)."""
        if node in self.cached_subgraphs:
            del self.cached_subgraphs[node]

    def get_unblocked_subgraphs(self, scheduled: OrderedSet[fx.Node]) -> list[SchedulableSubgraph]:
        """
        Get all cached subgraphs that are now unblocked and can be extended.
        """
        unblocked_subgraphs = []

        for node in list(self.cached_subgraphs.keys()):
            if node in scheduled:  # Can extend if the node is now scheduled
                cached_subgraphs = self.cached_subgraphs.get(node, [])
                unblocked_subgraphs.extend(cached_subgraphs)

        return unblocked_subgraphs

    def merge_subgraphs_pairwise(self, subgraphs: list[SchedulableSubgraph]) -> list[SchedulableSubgraph]:
        """
        Sort subgraphs by memory benefit and return them (greedy scheduling).
        """
        if len(subgraphs) <= 1:
            return subgraphs

        # Sort by net_change (most negative first = greatest memory reduction)
        return sorted(subgraphs, key=lambda sg: sg.net_change)

    def _try_extend_from_cache(
        self,
        start_node: fx.Node,
        scheduled: OrderedSet[fx.Node],
        is_schedulable_fn: Callable[[fx.Node], bool]
    ) -> Optional[SchedulableSubgraph]:
        """Try to extend subgraph from cached exploration results."""
        # Check if we can extend from any cached subgraphs
        cached_subgraphs = self.cached_subgraphs.get(start_node, [])
        if not cached_subgraphs:
            return None

        if len(cached_subgraphs) == 1:
            return cached_subgraphs[0]

        # Multiple cached subgraphs - merge them simply
        all_nodes = []
        all_collectives = []

        for subgraph in cached_subgraphs:
            all_nodes.extend(subgraph.nodes)
            all_collectives.extend(subgraph.collective_starts)

        # Remove duplicates while preserving order
        combined_nodes = list(OrderedSet(all_nodes))
        combined_collectives = list(OrderedSet(all_collectives))

        # Recalculate memory for merged subgraph
        peak_increase, net_change = self.memory_tracker.simulate_subgraph_memory(combined_nodes)

        return SchedulableSubgraph(
            nodes=combined_nodes,
            peak_increase=peak_increase,
            net_change=net_change,
            collective_starts=combined_collectives
        )

    def _build_subgraph_from(
        self,
        start_node: fx.Node,
        scheduled: OrderedSet[fx.Node],
        is_schedulable_fn: Callable[[fx.Node], bool]
    ) -> Optional[SchedulableSubgraph]:
        """Build a schedulable subgraph using cache merging instead of recursive dependencies."""
        MAX_SUBGRAPH_LENGTH = 50

        subgraph_nodes = []
        collective_starts = []

        # Queue of ready nodes to explore
        ready_queue = deque([start_node])

        while ready_queue and len(subgraph_nodes) < MAX_SUBGRAPH_LENGTH:
            node = ready_queue.popleft()

            if node in scheduled or node in subgraph_nodes:
                continue

            if not is_schedulable_fn(node):
                continue

            # Check if scheduling this node unblocks any cached subgraphs
            cached_subgraphs = self.cached_subgraphs.get(node, [])
            if cached_subgraphs:
                # Merge cached subgraphs with our current progress
                for cached_subgraph in cached_subgraphs:
                    subgraph_nodes.extend(cached_subgraph.nodes)
                    collective_starts.extend(cached_subgraph.collective_starts)

                # Remove from cache since we're using it
                del self.cached_subgraphs[node]

            # Add the node itself
            subgraph_nodes.append(node)

            # Track if this node is a collective
            if is_collective_start(node):
                collective_starts.append(node)

            # Add users to ready queue for exploration
            for user in node.users:
                if user not in scheduled and user not in subgraph_nodes:
                    ready_queue.append(user)

        if not subgraph_nodes:
            return None

        # Remove duplicates while preserving order
        unique_nodes = list(OrderedSet(subgraph_nodes))
        unique_collectives = list(OrderedSet(collective_starts))

        # Calculate memory profile for this subgraph
        peak_increase, net_change = self.memory_tracker.simulate_subgraph_memory(unique_nodes)

        # TODO: Consider cross-subgraph memory optimization
        # The current memory calculation is conservative - it only accounts for storage
        # freed when ALL uses are within this subgraph. We could improve net_change
        # calculation by considering:
        # 1. Storage where this subgraph has N-1 uses and another subgraph has 1 use
        # 2. Compute-blocking nodes that will be scheduled later completing storage lifetime
        # 3. Cooperative scheduling between subgraphs to maximize memory reuse

        return SchedulableSubgraph(
            nodes=unique_nodes,
            peak_increase=peak_increase,
            net_change=net_change,
            collective_starts=unique_collectives
        )




