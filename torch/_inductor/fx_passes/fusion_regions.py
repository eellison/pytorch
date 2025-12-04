"""Detect fusion regions for overlap scheduling using CapabilityBasedPartitioner."""

from dataclasses import dataclass, field

import torch
import torch.fx as fx
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner, Partition
from torch.fx.passes.operator_support import OperatorSupportBase
from torch.utils._ordered_set import OrderedSet


@dataclass
class FusionRegion:
    """Represents a connected set of fusible operations that will fuse together."""

    nodes: OrderedSet[fx.Node]  # All nodes in topo order
    cost_ms: float = field(default=0.0, init=False)  # Estimated cost in milliseconds
    external_inputs: OrderedSet[fx.Node] = field(default=None, init=False)
    external_outputs: OrderedSet[fx.Node] = field(default=None, init=False)
    external_users: OrderedSet[fx.Node] = field(default=None, init=False)
    subgraph_node: fx.Node | None = field(default=None, init=False)

    def __post_init__(self):
        """Compute cost and external inputs/outputs."""
        from torch._inductor.utils import get_gpu_dram_gbps

        region_set = set(self.nodes)
        self.external_inputs = OrderedSet()
        self.external_outputs = OrderedSet()
        self.external_users = OrderedSet()

        for node in self.nodes:
            # Collect all external inputs (not just tensors)
            for inp in node.all_input_nodes:
                if inp not in region_set:
                    self.external_inputs.add(inp)

            # Collect all external outputs (not just tensors)
            if any(u not in region_set for u in node.users) or len(node.users) == 0:
                self.external_outputs.add(node)

            # Collect all external users
            for user in node.users:
                if user not in region_set:
                    self.external_users.add(user)

        # Calculate cost from tensor metadata of external IO (bandwidth-bound)
        total_bytes = 0
        for node in self.external_inputs | self.external_outputs:
            val = node.meta.get("val")
            if not isinstance(val, torch.Tensor):
                continue

            total_bytes += val.numel() * val.element_size()

        if total_bytes > 0:
            fusion_bw_gbps = get_gpu_dram_gbps()
            fusion_bw_bytes_per_s = fusion_bw_gbps * 1e9
            self.cost_ms = (total_bytes / fusion_bw_bytes_per_s) * 1000

    @property
    def start(self) -> fx.Node:
        """First node in the region."""
        return next(iter(self.nodes))

    @property
    def end(self) -> fx.Node:
        """Last node (anchor) in the region."""
        return list(self.nodes)[-1]


def is_fusible_node(n: fx.Node) -> bool:
    """Check if a node is fusible (pointwise, reduction, views, indexing ops).

    Excludes: mm/conv, collectives, waits, placeholders, outputs.
    """
    if n.op != "call_function":
        return False

    # Include pointwise, reduction, views
    tags = getattr(n.target, "tags", ())
    if torch.Tag.pointwise in tags or torch.Tag.reduction in tags:
        return True

    if getattr(n.target, "is_view", False):
        return True

    # Include specific indexing ops
    aten = torch.ops.aten
    if n.target in (aten.slice.Tensor, aten.gather.default, aten.embedding.default):
        return True

    return False


class FusibleOperatorSupport(OperatorSupportBase):
    """Operator support for fusible operations (pointwise, reduction, view)."""

    def is_node_supported(
        self, submodules: dict[str, torch.nn.Module], node: fx.Node
    ) -> bool:
        return is_fusible_node(node)


def build_fusion_regions(
    gm: fx.GraphModule,
) -> dict[fx.Node, FusionRegion]:
    """Build fusion regions using CapabilityBasedPartitioner.

    Returns a dict mapping each node to its containing region (if any).

    Uses CapabilityBasedPartitioner to:
    1. Identify fusible nodes (pointwise, reduction, view)
    2. Group connected nodes into partitions
    3. Convert partitions to FusionRegion objects
    """
    operator_support = FusibleOperatorSupport()
    partitioner = CapabilityBasedPartitioner(
        gm,
        operator_support,
        allows_single_node_partition=False,
    )

    partitions = partitioner.propose_partitions()

    # Convert partitions to FusionRegion objects
    region_of: dict[fx.Node, FusionRegion] = {}

    for partition in partitions:
        nodes_list = list(partition.nodes.keys())
        if len(nodes_list) < 2:
            continue

        # Sort nodes topologically
        nodes_sorted = _topological_sort_nodes(nodes_list)
        region = FusionRegion(nodes=OrderedSet(nodes_sorted))

        # Only include regions with positive cost (have tensor I/O)
        if region.cost_ms > 0:
            for node in nodes_sorted:
                region_of[node] = region

    return region_of


def _topological_sort_nodes(nodes: list[fx.Node]) -> list[fx.Node]:
    """Topologically sort nodes based on dependencies."""
    if len(nodes) <= 1:
        return nodes

    node_set = set(nodes)

    # Calculate in-degrees
    in_degree = {n: 0 for n in nodes}
    for node in nodes:
        for inp in node.all_input_nodes:
            if inp in node_set:
                in_degree[node] += 1

    # Kahn's algorithm
    queue = [n for n in nodes if in_degree[n] == 0]
    result = []

    while queue:
        queue.sort(key=lambda n: n.name)
        node = queue.pop(0)
        result.append(node)

        for user in node.users:
            if user in node_set:
                in_degree[user] -= 1
                if in_degree[user] == 0:
                    queue.append(user)

    return result if len(result) == len(nodes) else nodes


def collapse_fusion_regions(
    gm: fx.GraphModule,
    region_of: dict[fx.Node, FusionRegion],
) -> tuple[dict[fx.Node, FusionRegion], dict[fx.Node, fx.Node]]:
    """
    Collapse fusion regions into call_module nodes using fuser_utils.

    Each fusion region is replaced with a single call_module node.
    The original nodes are erased.

    Args:
        gm: The GraphModule to modify
        region_of: Mapping of nodes to their fusion regions

    Returns:
        (new_region_of, replaced) where:
        - new_region_of: Mapping from module nodes to their regions
        - replaced: Mapping from original nodes to module node
    """
    from torch.fx.passes.utils.fuser_utils import (
        erase_nodes,
        fuse_as_graphmodule,
        insert_subgm,
        topo_sort,
    )

    replaced: dict[fx.Node, fx.Node] = {}

    if not region_of:
        return region_of, replaced

    # Get unique regions
    unique_regions: list[FusionRegion] = []
    seen_region_ids: set[int] = set()
    for region in region_of.values():
        region_id = id(region)
        if region_id not in seen_region_ids:
            seen_region_ids.add(region_id)
            unique_regions.append(region)

    new_region_of: dict[fx.Node, FusionRegion] = {}

    for region_idx, region in enumerate(unique_regions):
        nodes_list = list(region.nodes)
        if len(nodes_list) < 2:
            if nodes_list:
                new_region_of[nodes_list[0]] = region
            continue

        # Sort nodes topologically
        sorted_nodes = topo_sort(nodes_list)

        # Create subgraph using fuser_utils
        subgraph_name = f"_fusion_region_{region_idx}"
        try:
            sub_gm, orig_inputs, orig_outputs = fuse_as_graphmodule(
                gm,
                sorted_nodes,
                subgraph_name,
            )
        except AssertionError:
            # Invalid partition (cycle or other issue), skip this region
            continue

        # Insert the subgraph module into the main graph
        insert_subgm(gm, sub_gm, orig_inputs, orig_outputs)

        # Find the call_module node that was just inserted
        module_node = None
        for node in gm.graph.nodes:
            if node.op == "call_module" and node.target == subgraph_name:
                module_node = node
                break

        if module_node is None:
            continue

        # Map original nodes to module node
        for node in sorted_nodes:
            replaced[node] = module_node

        # Erase original nodes
        erase_nodes(gm, sorted_nodes)

        # Store module info in region
        region.subgraph_node = module_node
        new_region_of[module_node] = region

    # Fix graph ordering after insertions
    from torch._dynamo.graph_deduplication import _stable_topological_sort

    _stable_topological_sort(gm.graph, {})

    return new_region_of, replaced


def expand_fusion_regions(
    gm: fx.GraphModule,
    region_of: dict[fx.Node, FusionRegion],
    replaced: dict[fx.Node, fx.Node],
) -> dict[fx.Node, fx.Node]:
    """
    Expand call_module nodes back to their original nodes.

    Args:
        gm: The GraphModule
        region_of: Mapping from module nodes to their fusion regions
        replaced: Mapping from original nodes to module nodes (will be updated)

    Returns:
        Updated replaced mapping (original_node -> new_node)
    """
    if not region_of:
        return replaced

    graph = gm.graph

    for module_node, region in list(region_of.items()):
        if module_node not in graph.nodes:
            continue

        if module_node.op != "call_module":
            continue

        nodes_list = list(region.nodes)
        if len(nodes_list) < 2:
            continue

        # Get the subgraph module name and inputs
        subgraph_name = module_node.target
        subgraph_inputs = module_node.args

        # Get the subgraph module
        subgraph_module = getattr(gm, subgraph_name, None)
        if subgraph_module is None:
            continue

        subgraph_graph = subgraph_module.graph

        # Map from subgraph nodes to main graph nodes
        node_map: dict[fx.Node, fx.Node] = {}

        # Map subgraph placeholders to actual inputs
        placeholder_idx = 0
        for sg_node in subgraph_graph.nodes:
            if sg_node.op == "placeholder":
                if placeholder_idx < len(subgraph_inputs):
                    node_map[sg_node] = subgraph_inputs[placeholder_idx]
                placeholder_idx += 1

        # Inline subgraph nodes into main graph
        last_inlined_node = None
        with graph.inserting_before(module_node):
            for sg_node in subgraph_graph.nodes:
                if sg_node.op == "placeholder":
                    continue
                if sg_node.op == "output":
                    continue

                # Map args through node_map
                def map_arg(arg, nm=node_map):
                    if isinstance(arg, fx.Node):
                        return nm.get(arg, arg)
                    elif isinstance(arg, (list, tuple)):
                        return type(arg)(map_arg(a, nm) for a in arg)
                    elif isinstance(arg, dict):
                        return {k: map_arg(v, nm) for k, v in arg.items()}
                    return arg

                new_args = tuple(map_arg(a) for a in sg_node.args)
                new_kwargs = {k: map_arg(v) for k, v in sg_node.kwargs.items()}

                new_node = graph.create_node(
                    op=sg_node.op,
                    target=sg_node.target,
                    args=new_args,
                    kwargs=new_kwargs,
                )
                new_node.meta.update(sg_node.meta)
                node_map[sg_node] = new_node
                last_inlined_node = new_node

        # Update replaced mapping: map original region nodes to new inlined nodes
        sg_call_nodes = [n for n in subgraph_graph.nodes if n.op == "call_function"]
        for i, old_node in enumerate(nodes_list):
            if i < len(sg_call_nodes):
                sg_node = sg_call_nodes[i]
                if sg_node in node_map:
                    replaced[old_node] = node_map[sg_node]

        # Replace uses of module node with the last inlined node
        if last_inlined_node is not None:
            module_node.replace_all_uses_with(last_inlined_node)
            replaced[module_node] = last_inlined_node

        # Erase the module node
        graph.erase_node(module_node)

        # Remove the submodule
        if hasattr(gm, subgraph_name):
            delattr(gm, subgraph_name)

    return replaced


def resolve_replacement_chain(
    node: fx.Node,
    replaced: dict[fx.Node, fx.Node],
) -> fx.Node:
    """Follow replacement chain to get the final node."""
    visited = set()
    while node in replaced and node not in visited:
        visited.add(node)
        node = replaced[node]
    return node


def transfer_deps_to_region_outputs(
    region_of: dict[fx.Node, FusionRegion],
    deps: dict[fx.Node, OrderedSet[fx.Node]],
) -> dict[fx.Node, OrderedSet[fx.Node]]:
    """
    Transfer dependencies from internal region nodes to region outputs.

    For any dependency on an internal node, redirect it to the region's
    last output node (the anchor).

    Args:
        region_of: Mapping from nodes to their fusion regions
        deps: Dependencies to transfer

    Returns:
        New dependencies with internal nodes redirected to region outputs
    """
    if not region_of:
        return deps

    # Build map from internal nodes to their region's output
    node_to_output: dict[fx.Node, fx.Node] = {}
    for node, region in region_of.items():
        # Use the last node (end) as the output/anchor
        node_to_output[node] = region.end

    # Transfer dependencies
    new_deps: dict[fx.Node, OrderedSet[fx.Node]] = {}

    for target, target_deps in deps.items():
        # Redirect target if it's internal to a region
        new_target = node_to_output.get(target, target)

        # Redirect each dependency
        new_target_deps = OrderedSet()
        for dep in target_deps:
            new_dep = node_to_output.get(dep, dep)
            # Avoid self-dependencies
            if new_dep != new_target:
                new_target_deps.add(new_dep)

        if new_target_deps:
            if new_target in new_deps:
                new_deps[new_target].update(new_target_deps)
            else:
                new_deps[new_target] = new_target_deps

    return new_deps
