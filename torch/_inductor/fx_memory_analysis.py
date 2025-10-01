"""
Memory analysis utility for FX graphs with fake tensors.

This module provides utilities to analyze memory usage patterns in FX graphs
based on fake tensor metadata, computing peak memory usage and tracking
storage lifetimes.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
import torch.fx as fx

log = logging.getLogger(__name__)


@dataclass
class StorageInfo:
    """Information about a tensor storage."""
    storage_id: int
    size_bytes: int
    first_use_node: fx.Node
    last_use_node: fx.Node
    is_input: bool = False
    is_output: bool = False
    deallocates_on_backward: bool = False


@dataclass
class NodeMemoryInfo:
    """Memory information for a single node."""
    node: fx.Node
    allocations: int = 0  # Memory allocated by this node
    deallocations: int = 0  # Memory deallocated by this node
    net_change: int = 0  # Net memory change (allocations - deallocations)
    peak_during_execution: int = 0  # Peak memory during node execution


@dataclass
class ChainMemoryInfo:
    """Memory information for a chain of nodes."""
    nodes: List[fx.Node]
    final_memory_change: int  # Net change after executing entire chain
    peak_memory_increase: int  # Maximum temporary increase during chain execution
    reduces_memory: bool  # Whether chain ultimately reduces memory


@dataclass
class GraphMemoryAnalysis:
    """Complete memory analysis of an FX graph."""
    storage_info: Dict[int, StorageInfo]
    node_memory_info: Dict[fx.Node, NodeMemoryInfo]
    peak_memory_bytes: int
    total_input_memory: int
    total_output_memory: int
    memory_timeline: List[Tuple[fx.Node, int]]  # (node, memory_after_node)


class FXMemoryAnalyzer:
    """Analyzer for memory usage patterns in FX graphs with fake tensors."""

    def __init__(self, graph: fx.Graph):
        self.graph = graph
        self.nodes = list(graph.nodes)

    def _get_fake_tensor_storage_size(self, fake_tensor: torch.Tensor) -> int:
        """Get the size of a fake tensor's storage in bytes."""
        if not hasattr(fake_tensor, 'untyped_storage'):
            return getattr(fake_tensor, 'numel', lambda: 0)() * fake_tensor.element_size()

        storage = fake_tensor.untyped_storage()
        if hasattr(storage, 'size'):
            return storage.size()

        # Fallback: estimate from tensor shape and dtype
        return fake_tensor.numel() * fake_tensor.element_size()

    def _get_storage_id(self, fake_tensor: torch.Tensor) -> int:
        """Get a unique identifier for the fake tensor's storage."""
        if hasattr(fake_tensor, 'untyped_storage'):
            storage = fake_tensor.untyped_storage()
            return id(storage)
        # Fallback to tensor id if no storage available
        return id(fake_tensor)

    def _is_tangent_input(self, node: fx.Node) -> bool:
        """Check if a node represents a tangent (backward pass gradient) input using PyTorch patterns."""
        if node.op != 'placeholder':
            return False

        # Use the same pattern as torch/_inductor/utils.py:is_nonfreeable_buffers
        # Tangents start with "tangents" and are freeable after use
        node_name = node.name

        # Handle subgraph prefixes (similar to is_nonfreeable_buffers)
        # Note: We don't have access to V.graph.name here, but we can still check patterns
        return node_name.startswith("tangents")

    def _is_primal_input(self, node: fx.Node) -> bool:
        """Check if a node represents a primal (forward pass) input."""
        if node.op != 'placeholder':
            return False

        node_name = node.name
        return node_name.startswith(("primals_", "arg"))

    def _is_freeable_input(self, node: fx.Node) -> bool:
        """Check if an input can be freed after use (tangents are freeable)."""
        return self._is_tangent_input(node)

    def _extract_tensors_from_value(self, value) -> List[torch.Tensor]:
        """Extract all tensors from a potentially nested value."""
        tensors = []
        if isinstance(value, torch.Tensor):
            tensors.append(value)
        elif isinstance(value, (list, tuple)):
            for item in value:
                tensors.extend(self._extract_tensors_from_value(item))
        elif isinstance(value, dict):
            for item in value.values():
                tensors.extend(self._extract_tensors_from_value(item))
        return tensors

    def analyze(self) -> GraphMemoryAnalysis:
        """Perform complete memory analysis of the graph."""
        log.debug(f"Analyzing memory for graph with {len(self.nodes)} nodes")

        # First pass: identify all storages and their lifetimes
        storage_info = self._analyze_storage_lifetimes()

        # Second pass: compute memory changes per node
        node_memory_info = self._compute_node_memory_info(storage_info)

        # Third pass: simulate execution to get peak memory and timeline
        peak_memory, timeline = self._simulate_memory_timeline(node_memory_info)

        # Compute input/output memory
        input_memory = sum(
            info.size_bytes for info in storage_info.values() if info.is_input
        )
        output_memory = sum(
            info.size_bytes for info in storage_info.values() if info.is_output
        )

        return GraphMemoryAnalysis(
            storage_info=storage_info,
            node_memory_info=node_memory_info,
            peak_memory_bytes=peak_memory,
            total_input_memory=input_memory,
            total_output_memory=output_memory,
            memory_timeline=timeline,
        )

    def _analyze_storage_lifetimes(self) -> Dict[int, StorageInfo]:
        """First pass: analyze storage lifetimes."""
        storage_to_first_use: Dict[int, fx.Node] = {}
        storage_to_last_use: Dict[int, fx.Node] = {}
        storage_to_size: Dict[int, int] = {}

        input_nodes = set()
        output_nodes = set()

        for node in self.nodes:
            if node.op == 'placeholder':
                input_nodes.add(node)
            elif node.op == 'output':
                output_nodes.add(node)

            # Process node outputs (what this node creates)
            if hasattr(node, 'meta') and 'val' in node.meta:
                tensors = self._extract_tensors_from_value(node.meta['val'])
                for tensor in tensors:
                    storage_id = self._get_storage_id(tensor)
                    size = self._get_fake_tensor_storage_size(tensor)

                    if storage_id not in storage_to_first_use:
                        storage_to_first_use[storage_id] = node
                    storage_to_last_use[storage_id] = node
                    storage_to_size[storage_id] = size

            # Process node inputs (what this node consumes)
            for input_node in node.all_input_nodes:
                if hasattr(input_node, 'meta') and 'val' in input_node.meta:
                    tensors = self._extract_tensors_from_value(input_node.meta['val'])
                    for tensor in tensors:
                        storage_id = self._get_storage_id(tensor)
                        storage_to_last_use[storage_id] = node  # Update last use

        # Build storage info
        storage_info: Dict[int, StorageInfo] = {}
        for storage_id, size in storage_to_size.items():
            first_use = storage_to_first_use.get(storage_id)
            last_use = storage_to_last_use.get(storage_id)

            if first_use and last_use:
                storage_info[storage_id] = StorageInfo(
                    storage_id=storage_id,
                    size_bytes=size,
                    first_use_node=first_use,
                    last_use_node=last_use,
                    is_input=first_use in input_nodes,
                    is_output=last_use in output_nodes,
                    deallocates_on_backward=self._is_tangent_input(first_use)
                )

        return storage_info

    def _compute_node_memory_info(self, storage_info: Dict[int, StorageInfo]) -> Dict[fx.Node, NodeMemoryInfo]:
        """Second pass: compute memory changes per node."""
        node_memory_info: Dict[fx.Node, NodeMemoryInfo] = {}

        # Initialize all nodes
        for node in self.nodes:
            node_memory_info[node] = NodeMemoryInfo(node=node)

        # Compute allocations and deallocations
        for storage in storage_info.values():
            # Allocation happens at first use
            alloc_node = storage.first_use_node
            node_memory_info[alloc_node].allocations += storage.size_bytes

            # Deallocation happens at last use (or immediately if backward input)
            if storage.deallocates_on_backward:
                dealloc_node = storage.first_use_node
            else:
                dealloc_node = storage.last_use_node

            node_memory_info[dealloc_node].deallocations += storage.size_bytes

        # Compute net changes
        for info in node_memory_info.values():
            info.net_change = info.allocations - info.deallocations

        return node_memory_info

    def _simulate_memory_timeline(self, node_memory_info: Dict[fx.Node, NodeMemoryInfo]) -> Tuple[int, List[Tuple[fx.Node, int]]]:
        """Third pass: simulate execution to get peak memory."""
        current_memory = 0
        peak_memory = 0
        timeline = []

        for node in self.nodes:
            info = node_memory_info[node]

            # First allocate memory for outputs
            current_memory += info.allocations
            peak_during_execution = current_memory

            # Then deallocate inputs that are last used
            current_memory -= info.deallocations

            # Update peak tracking
            peak_memory = max(peak_memory, peak_during_execution)
            info.peak_during_execution = peak_during_execution

            timeline.append((node, current_memory))

        return peak_memory, timeline

    def analyze_chain_memory(self, chain: List[fx.Node], current_memory: int) -> ChainMemoryInfo:
        """Analyze memory impact of executing a chain of nodes."""
        if not chain:
            return ChainMemoryInfo(
                nodes=chain,
                final_memory_change=0,
                peak_memory_increase=0,
                reduces_memory=False
            )

        # Analyze the chain assuming it executes in order
        simulated_memory = current_memory
        peak_increase = 0

        # We need node memory info for this analysis
        analysis = self.analyze()

        for node in chain:
            node_info = analysis.node_memory_info.get(node)
            if node_info:
                # Allocate first
                simulated_memory += node_info.allocations
                temp_increase = simulated_memory - current_memory
                peak_increase = max(peak_increase, temp_increase)

                # Then deallocate
                simulated_memory -= node_info.deallocations

        final_change = simulated_memory - current_memory

        return ChainMemoryInfo(
            nodes=chain,
            final_memory_change=final_change,
            peak_memory_increase=peak_increase,
            reduces_memory=final_change < 0
        )


def analyze_graph_memory(graph: fx.Graph) -> GraphMemoryAnalysis:
    """Convenience function to analyze memory usage of an FX graph."""
    analyzer = FXMemoryAnalyzer(graph)
    return analyzer.analyze()


def get_peak_memory_bytes(graph: fx.Graph) -> int:
    """Get the peak memory usage of an FX graph in bytes."""
    analysis = analyze_graph_memory(graph)
    return analysis.peak_memory_bytes


def get_peak_memory_gb(graph: fx.Graph) -> float:
    """Get the peak memory usage of an FX graph in GB."""
    peak_bytes = get_peak_memory_bytes(graph)
    return peak_bytes / (1024 * 1024 * 1024)


def suggest_memory_limit_gb(graph: fx.Graph, safety_factor: float = 1.2) -> float:
    """Suggest an appropriate memory limit based on graph analysis.

    Args:
        graph: FX graph to analyze
        safety_factor: Multiplier for peak memory to account for scheduling overhead

    Returns:
        Suggested memory limit in GB
    """
    analysis = analyze_graph_memory(graph)
    peak_gb = analysis.peak_memory_bytes / (1024 * 1024 * 1024)

    # Add safety factor for scheduling overhead and temporary increases
    suggested_limit = peak_gb * safety_factor

    # Minimum reasonable limit (100MB)
    min_limit_gb = 0.1

    return max(suggested_limit, min_limit_gb)