"""
Utility functions for subgraph analysis and memory-aware scheduling.

Factored out common operations to reduce duplication and improve maintainability.
"""

from typing import Callable
import torch.fx as fx
from torch._inductor.fx_passes.bucketing import is_wait_tensor
from torch.utils._ordered_set import OrderedSet


# =============================================================================
# Collective Operation Utilities
# =============================================================================

def is_collective_start(node: fx.Node) -> bool:
    """Check if a node is a collective operation start (not wait)."""
    return (
        node.op == "call_function"
        and hasattr(node.target, "_schema")
        and "c10d_functional" in str(node.target)
        and "wait_tensor" not in str(node.target)
    )


def find_collective_indices(nodes: list[fx.Node], collective_starts: list[fx.Node]) -> list[int]:
    """Find indices of collective operations in a list of nodes."""
    return [
        i for i, node in enumerate(nodes)
        if node in collective_starts
    ]


# =============================================================================
# Dependency Resolution Utilities
# =============================================================================

def can_schedule_with_dependencies(
    node: fx.Node,
    scheduled: OrderedSet[fx.Node],
    current_subgraph: OrderedSet[fx.Node],
    is_schedulable_fn: Callable[[fx.Node], bool]
) -> bool:
    """
    Check if a node can be scheduled by recursively satisfying its dependencies.

    Returns True if all input dependencies are either:
    - Already scheduled
    - Already in the current subgraph
    - Can be recursively scheduled (all their deps are ready)
    """
    for inp in node.all_input_nodes:
        if inp in scheduled or inp in current_subgraph:
            continue

        if not is_schedulable_fn(inp):
            return False

        if not can_schedule_with_dependencies(inp, scheduled, current_subgraph, is_schedulable_fn):
            return False

    return True


def collect_dependencies_topologically(
    node: fx.Node,
    scheduled: OrderedSet[fx.Node],
    current_subgraph: OrderedSet[fx.Node],
    is_schedulable_fn: Callable[[fx.Node], bool]
) -> tuple[OrderedSet[fx.Node], list[fx.Node]]:
    """
    Recursively collect required dependencies in topological order.

    Returns:
        (dependencies_to_add, collective_operations_found)
    """
    deps_to_add = OrderedSet()
    collective_deps = []

    def collect_deps(n: fx.Node) -> None:
        for inp in n.all_input_nodes:
            if inp in scheduled or inp in current_subgraph or inp in deps_to_add:
                continue

            # Don't traverse through wait tensors in dependencies
            if is_wait_tensor(inp):
                continue

            if is_schedulable_fn(inp):
                collect_deps(inp)  # Recurse first for topological order
                deps_to_add.add(inp)

                # Track collective operations in dependencies
                if is_collective_start(inp):
                    collective_deps.append(inp)

    collect_deps(node)
    return deps_to_add, collective_deps


# =============================================================================
# Memory Calculation Utilities
# =============================================================================

def calculate_combined_peak_memory(subgraph1_peak: int, subgraph1_net: int, subgraph2_peak: int) -> int:
    """
    Calculate peak memory when executing subgraph1 followed by subgraph2.

    Core algorithm: max(subgraph1.peak, subgraph1.net + subgraph2.peak)
    """
    return max(subgraph1_peak, subgraph1_net + subgraph2_peak)


def find_optimal_subgraph_order(
    subgraph1_peak: int, subgraph1_net: int,
    subgraph2_peak: int, subgraph2_net: int
) -> tuple[bool, int]:
    """
    Find optimal ordering of two subgraphs based on memory usage.

    Returns:
        (order_1_then_2_is_better, optimal_peak_memory)
    """
    option1_peak = calculate_combined_peak_memory(subgraph1_peak, subgraph1_net, subgraph2_peak)
    option2_peak = calculate_combined_peak_memory(subgraph2_peak, subgraph2_net, subgraph1_peak)

    if option1_peak <= option2_peak:
        return True, option1_peak
    else:
        return False, option2_peak


def combine_node_lists_preserving_order(first_nodes: list[fx.Node], second_nodes: list[fx.Node]) -> list[fx.Node]:
    """Combine two node lists, avoiding duplicates while preserving order."""
    combined_set = OrderedSet(first_nodes)
    combined_set.update(second_nodes)
    return list(combined_set)


def combine_collective_lists(first_collectives: list[fx.Node], second_collectives: list[fx.Node]) -> list[fx.Node]:
    """Combine collective operation lists, avoiding duplicates."""
    return first_collectives + [c for c in second_collectives if c not in first_collectives]


# =============================================================================
# Subgraph Construction Utilities
# =============================================================================

def add_dependencies_to_subgraph(
    node: fx.Node,
    scheduled: OrderedSet[fx.Node],
    current_subgraph: OrderedSet[fx.Node],
    is_schedulable_fn: Callable[[fx.Node], bool]
) -> list[fx.Node]:
    """
    Add required dependencies to the subgraph before scheduling node.

    Dependencies are added in topological order to ensure correct scheduling.
    Returns list of collective starts that were added as dependencies.
    """
    deps_to_add, collective_deps = collect_dependencies_topologically(
        node, scheduled, current_subgraph, is_schedulable_fn
    )

    # Add dependencies to subgraph in topological order
    for dep in deps_to_add:
        current_subgraph.add(dep)

    return collective_deps


def validate_subgraph_constraints(
    subgraph_nodes: list[fx.Node],
    peak_increase: int,
    net_change: int,
    available_memory: int
) -> bool:
    """
    Validate that a subgraph meets scheduling constraints.

    Returns True if subgraph fits in available memory and reduces net memory.
    """
    return peak_increase <= available_memory and net_change < 0


# =============================================================================
# Partial Scheduling Utilities
# =============================================================================

def get_remaining_nodes(all_nodes: list[fx.Node], consumed_nodes: OrderedSet[fx.Node]) -> list[fx.Node]:
    """Get nodes not yet scheduled from a subgraph."""
    return [n for n in all_nodes if n not in consumed_nodes]


def get_nodes_up_to_collective(remaining_nodes: list[fx.Node], collective_starts: list[fx.Node]) -> list[fx.Node]:
    """Get nodes from start of remaining subgraph up to first collective."""
    collective_indices = find_collective_indices(remaining_nodes, collective_starts)

    if collective_indices:
        # Include the collective start but not beyond
        return remaining_nodes[:collective_indices[0] + 1]
    else:
        # No collectives remaining, return all remaining nodes
        return remaining_nodes


def get_nodes_after_collectives(remaining_nodes: list[fx.Node], collective_starts: list[fx.Node]) -> list[fx.Node]:
    """Get nodes that come after collective operations."""
    collective_indices = find_collective_indices(remaining_nodes, collective_starts)

    if collective_indices:
        # Return nodes after the last collective
        return remaining_nodes[collective_indices[-1] + 1:]
    else:
        # No collectives, return empty
        return []


# =============================================================================
# Best Merge Finding Utilities
# =============================================================================

def find_best_merge_indices(
    subgraphs: list,
    calculate_peak_fn: Callable[[int, int], int]
) -> tuple[int, int, int, object]:
    """
    Find the best pair of subgraphs to merge based on peak memory.

    Returns:
        (index1, index2, best_peak, best_merged_object)
    """
    best_merge = None
    best_peak = float('inf')
    best_indices = None

    # Try all pairs to find optimal merge
    for i in range(len(subgraphs)):
        for j in range(i + 1, len(subgraphs)):
            subgraph1, subgraph2 = subgraphs[i], subgraphs[j]

            # Try both orderings and pick better one
            peak1 = calculate_peak_fn(subgraph1, subgraph2)
            peak2 = calculate_peak_fn(subgraph2, subgraph1)

            if min(peak1, peak2) < best_peak:
                best_peak = min(peak1, peak2)
                # The merge object would be created by caller
                best_indices = (i, j)

    return best_indices[0] if best_indices else -1, best_indices[1] if best_indices else -1, best_peak, best_merge