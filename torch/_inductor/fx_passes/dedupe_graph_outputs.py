# mypy: allow-untyped-defs
"""
Share computation for identical graph outputs while preserving output storage.

When multiple graph outputs compute the exact same value (structurally
identical computation DAGs), this pass computes the value once and clones it
for each duplicate output. This avoids scheduling redundant expensive
computations without changing the graph's output aliasing contract.

The pass assigns structural equivalence classes in topological order: two nodes
are equivalent if they have the same op, target, and equivalent arguments.
"""

import struct

import torch
from torch import fx
from torch._dynamo.device_interface import get_interface_for_device
from torch._dynamo.utils import counters
from torch._subclasses.fake_tensor import FakeTensor
from torch.utils._ordered_set import OrderedSet

from .. import config


_MIN_IDENTICAL_GRAPH_OUTPUTS = 8
_MAX_IDENTICAL_GRAPH_OUTPUTS = 32
_SAFE_OP_NAMESPACES = OrderedSet(["aten", "prims"])
_REJECTED_OP_TAGS = OrderedSet(
    [
        torch.Tag.cudagraph_unsafe,
        torch.Tag.data_dependent_output,
        torch.Tag.dynamic_output_shape,
        torch.Tag.inplace,
        torch.Tag.maybe_aliasing_or_mutating,
        torch.Tag.nondeterministic_bitwise,
        torch.Tag.nondeterministic_seeded,
        torch.Tag.out,
        torch.Tag.out_variant,
    ]
)
_REJECTED_OP_PACKETS = OrderedSet(
    [
        torch.ops.aten._conj,
        torch.ops.aten._empty_affine_quantized,
        torch.ops.aten._empty_per_channel_affine_quantized,
        torch.ops.aten._neg_view,
        torch.ops.aten.empty,
        torch.ops.aten.empty_like,
        torch.ops.aten.empty_permuted,
        torch.ops.aten.empty_quantized,
        torch.ops.aten.empty_strided,
        torch.ops.aten.new_empty,
        torch.ops.aten.new_empty_strided,
        torch.ops.prims.empty,
        torch.ops.prims.empty_permuted,
        torch.ops.prims.empty_strided,
    ]
)
_StorageKey = tuple[int, torch.device]


def is_output_computation_sharing_supported(gm: fx.GraphModule) -> bool:
    """
    Limit default-on use to the configuration covered by the performance audit.

    Use the graph's output device instead of the process's current device: a
    compile worker may be compiling for a different local device.
    """
    if torch.version.hip is not None or config.cuda_backend != "triton":
        return False

    output_node = next((node for node in gm.graph.nodes if node.op == "output"), None)
    if output_node is None:
        return False

    devices = OrderedSet[torch.device]()
    for arg in torch.utils._pytree.arg_tree_leaves(
        *output_node.args, **output_node.kwargs
    ):
        if isinstance(arg, fx.Node):
            devices.update(
                leaf.device
                for leaf in torch.utils._pytree.tree_leaves(arg.meta.get("val"))
                if isinstance(leaf, torch.Tensor)
            )
    if len(devices) != 1:
        return False
    device = next(iter(devices))
    if device.type != "cuda":
        return False

    device_interface = get_interface_for_device(device.type)
    properties = device_interface.Worker.get_device_properties(device)
    return properties.major == 10


def _constant_key(value):
    """Return a type-sensitive key, or None for constants we cannot compare."""
    value_type = type(value)
    if value is None:
        return (value_type,)
    if value_type in (bool, int, str, bytes):
        return (value_type, value)
    if value_type is float:
        return (value_type, struct.pack("!d", value))
    if value_type is complex:
        return (
            value_type,
            struct.pack("!d", value.real),
            struct.pack("!d", value.imag),
        )
    if value_type in (torch.dtype, torch.device, torch.layout, torch.memory_format):
        return (value_type, value)
    return None


def _flatten_node_arguments(node: fx.Node):
    return torch.utils._pytree.tree_flatten(
        (node.args, tuple(sorted(node.kwargs.items())))
    )


def _is_shareable_node(node: fx.Node) -> bool:
    if (
        node.op != "call_function"
        or not isinstance(node.target, torch._ops.OpOverload)
        or node.target.namespace not in _SAFE_OP_NAMESPACES
        or node.target.overloadpacket in _REJECTED_OP_PACKETS
    ):
        return False

    if _REJECTED_OP_TAGS.intersection(node.target.tags):
        return False
    if node.is_impure():
        return False
    return not any(
        arg.alias_info is not None and arg.alias_info.is_write
        for arg in node.target._schema.arguments
    )


def _is_pure_functional_graph(graph: fx.Graph) -> bool:
    """
    Reject graphs whose effects are not represented in data dependencies.

    Structural equivalence alone cannot distinguish reads of a tensor before
    and after an in-graph mutation. The audited inference case is entirely
    functional, so fail closed instead of attempting mutation versioning here.
    """
    for node in graph.nodes:
        if node.op in ("placeholder", "get_attr", "output"):
            continue
        if (
            node.op != "call_function"
            or not isinstance(node.target, torch._ops.OpOverload)
            or node.target.namespace not in _SAFE_OP_NAMESPACES
            or node.is_impure()
        ):
            return False
    return True


def _storage_key(value) -> _StorageKey | None:
    if (
        type(value) is not FakeTensor
        or value.layout is not torch.strided
        or not torch._C._has_storage(value)
    ):
        return None
    try:
        storage = value.untyped_storage()
    except RuntimeError:
        return None
    return (storage._cdata, value.device)


def _metadata_storage_keys(value) -> OrderedSet[_StorageKey]:
    keys: OrderedSet[_StorageKey] = OrderedSet()
    for leaf in torch.utils._pytree.tree_leaves(value):
        if (key := _storage_key(leaf)) is not None:
            keys.add(key)
    return keys


def _has_clone_compatible_output_storage(
    node: fx.Node,
    input_storage_keys: OrderedSet[_StorageKey],
    output_storage_users: dict[_StorageKey, OrderedSet[fx.Node]],
) -> bool:
    value = node.meta.get("val")
    key = _storage_key(value)
    if (
        key is None
        or value.layout is not torch.strided
        or not all(isinstance(size, int) for size in value.size())
        or not all(isinstance(stride, int) for stride in value.stride())
        or not isinstance(value.storage_offset(), int)
        or value.storage_offset() != 0
        or not torch._prims_common.is_non_overlapping_and_dense_or_false(value)
    ):
        return False

    try:
        storage_nbytes = value.untyped_storage().nbytes()
    except RuntimeError:
        return False

    return (
        key not in input_storage_keys
        and len(output_storage_users.get(key, ())) == 1
        and storage_nbytes == value.numel() * value.element_size()
    )


def _compute_structural_classes(graph: fx.Graph) -> dict[fx.Node, int]:
    """
    Assign identical pure computation DAGs the same collision-free class.

    Post-grad graphs are topologically sorted, so every input node already has
    a class. Unsafe or unsupported nodes receive unique classes. Descendants
    may still share computation when they consume the exact same unsafe node;
    the unsafe operation itself is never removed.
    """
    classes: dict[fx.Node, int] = {}
    class_by_key: dict[tuple[object, ...], int] = {}
    next_class = 0

    for node in graph.nodes:
        key = None
        if _is_shareable_node(node):
            leaves, spec = _flatten_node_arguments(node)
            leaf_keys = []
            for leaf in leaves:
                if isinstance(leaf, fx.Node):
                    if leaf not in classes:
                        break
                    leaf_keys.append(("node", classes[leaf]))
                elif (constant_key := _constant_key(leaf)) is not None:
                    leaf_keys.append(("constant", constant_key))
                else:
                    break
            else:
                key = (node.target, spec, *leaf_keys)

        if key is not None and key in class_by_key:
            classes[node] = class_by_key[key]
        else:
            classes[node] = next_class
            if key is not None:
                class_by_key[key] = next_class
            next_class += 1

    return classes


def dedupe_graph_outputs_pass(graph: fx.Graph) -> None:
    """
    Share identical graph-output computation while preserving distinct storage.

    Walks the graph output tuple, groups outputs by structural hash, confirms
    equivalence, and replaces each duplicate with a clone of the canonical
    value. Then runs dead code elimination to remove the now-unused duplicate
    subgraphs.
    """
    import torch.utils._pytree as pytree

    output_node = next((node for node in graph.nodes if node.op == "output"), None)
    if output_node is None:
        return

    output_args = pytree.arg_tree_leaves(*output_node.args, **output_node.kwargs)
    output_nodes = [node for node in output_args if isinstance(node, fx.Node)]
    if len(output_nodes) <= 1 or not _is_pure_functional_graph(graph):
        return

    structural_classes = _compute_structural_classes(graph)
    equivalent_groups: dict[int, OrderedSet[fx.Node]] = {}
    for node in output_nodes:
        equivalent_groups.setdefault(structural_classes[node], OrderedSet()).add(node)

    if not any(
        _MIN_IDENTICAL_GRAPH_OUTPUTS <= len(group) <= _MAX_IDENTICAL_GRAPH_OUTPUTS
        for group in equivalent_groups.values()
    ):
        return

    input_storage_keys: OrderedSet[_StorageKey] = OrderedSet()
    for node in graph.nodes:
        if node.op in ("placeholder", "get_attr"):
            input_storage_keys.update(_metadata_storage_keys(node.meta.get("val")))

    output_storage_users: dict[_StorageKey, OrderedSet[fx.Node]] = {}
    for output in output_nodes:
        if (key := _storage_key(output.meta.get("val"))) is not None:
            output_storage_users.setdefault(key, OrderedSet()).add(output)

    # The canonical remains in place, including any non-output users. Only
    # output-only duplicate branches are replaced.
    replacements_made = 0
    for group in equivalent_groups.values():
        if not (
            _MIN_IDENTICAL_GRAPH_OUTPUTS <= len(group) <= _MAX_IDENTICAL_GRAPH_OUTPUTS
        ):
            continue

        eligible_nodes = [
            node
            for node in group
            if _has_clone_compatible_output_storage(
                node, input_storage_keys, output_storage_users
            )
        ]
        if len(eligible_nodes) < _MIN_IDENTICAL_GRAPH_OUTPUTS:
            continue

        def is_output_only(node: fx.Node) -> bool:
            return len(node.users) == 1 and output_node in node.users

        # Prefer a node that must remain for an internal user. Otherwise output
        # order can leave that computation live after sharing the output-only
        # branches.
        canonical_node = next(
            (node for node in eligible_nodes if not is_output_only(node)),
            eligible_nodes[0],
        )
        duplicate_nodes = [
            node
            for node in eligible_nodes
            if node is not canonical_node and is_output_only(node)
        ]
        if len(duplicate_nodes) + 1 < _MIN_IDENTICAL_GRAPH_OUTPUTS:
            continue

        for duplicate_node in duplicate_nodes:
            with graph.inserting_before(output_node):
                output_copy = graph.call_function(
                    torch.ops.aten.clone.default, (canonical_node,)
                )
            duplicate_node.replace_all_uses_with(output_copy)
            replacements_made += 1

    if replacements_made > 0:
        graph.eliminate_dead_code()
        graph.lint()
        counters["inductor"]["dedupe_graph_outputs"] += replacements_made
