# mypy: allow-untyped-defs
"""
Deduplicate identical graph outputs.

When multiple graph outputs compute the exact same value (structurally identical
computation DAGs), this pass replaces duplicates with a copy of the canonical
(first) node.  This avoids scheduling redundant kernels for identical
computations — a common pattern in causal mask generation where e.g. 32
attention heads each produce the same mask tensor.

The pass uses structural hashing: two nodes are equivalent if they have the same
op, target, and their arguments are pairwise equivalent (recursively for Node
args, value-equal for constants/scalars).
"""

import torch
from torch import fx
from torch._dynamo.utils import counters


def _structural_hash(node: fx.Node, cache: dict[fx.Node, int]) -> int:
    """
    Compute a structural hash for an FX node based on its computation DAG.

    Two nodes with the same structural hash are candidates for being identical
    (same op, target, and recursively equivalent arguments).
    """
    if node in cache:
        return cache[node]

    if node.op in ("placeholder", "get_attr"):
        # Placeholders and get_attr are unique by identity
        h = hash(id(node))
    elif node.op == "call_function":
        # Hash based on target + structural hashes of args
        arg_hashes = []
        for arg in node.args:
            if isinstance(arg, fx.Node):
                arg_hashes.append(_structural_hash(arg, cache))
            else:
                # For scalar constants, lists, tuples, etc., hash the value
                try:
                    arg_hashes.append(hash(arg))
                except TypeError:
                    # Unhashable args (e.g., lists) — convert to tuple
                    arg_hashes.append(hash(str(arg)))

        kwarg_hashes = []
        for k in sorted(node.kwargs.keys()):
            v = node.kwargs[k]
            if isinstance(v, fx.Node):
                kwarg_hashes.append((k, _structural_hash(v, cache)))
            else:
                try:
                    kwarg_hashes.append((k, hash(v)))
                except TypeError:
                    kwarg_hashes.append((k, hash(str(v))))

        h = hash((node.target, tuple(arg_hashes), tuple(kwarg_hashes)))
    else:
        # output, etc. — use identity
        h = hash(id(node))

    cache[node] = h
    return h


def _structurally_equal(a: fx.Node, b: fx.Node, memo: dict[tuple[fx.Node, fx.Node], bool]) -> bool:
    """
    Check if two FX nodes compute the same value by structural comparison.

    This recursively checks that:
    - Both have the same op and target
    - All arguments are pairwise equivalent (same Node for shared inputs,
      or structurally equal for independently computed inputs; same value for constants)
    """
    if a is b:
        return True

    key = (a, b)
    if key in memo:
        return memo[key]

    # Prevent infinite recursion for cycles (shouldn't happen in DAGs but be safe)
    memo[key] = False

    if a.op != b.op:
        memo[key] = False
        return False

    if a.op in ("placeholder", "get_attr"):
        # These are only equal if they are the same node
        result = a is b
        memo[key] = result
        return result

    if a.op != "call_function":
        memo[key] = False
        return False

    if a.target != b.target:
        memo[key] = False
        return False

    # Check args
    if len(a.args) != len(b.args):
        memo[key] = False
        return False

    for arg_a, arg_b in zip(a.args, b.args):
        if isinstance(arg_a, fx.Node) and isinstance(arg_b, fx.Node):
            if not _structurally_equal(arg_a, arg_b, memo):
                memo[key] = False
                return False
        elif isinstance(arg_a, fx.Node) or isinstance(arg_b, fx.Node):
            memo[key] = False
            return False
        else:
            # Both are constants
            if arg_a != arg_b:
                memo[key] = False
                return False

    # Check kwargs
    if set(a.kwargs.keys()) != set(b.kwargs.keys()):
        memo[key] = False
        return False

    for k in a.kwargs:
        va = a.kwargs[k]
        vb = b.kwargs[k]
        if isinstance(va, fx.Node) and isinstance(vb, fx.Node):
            if not _structurally_equal(va, vb, memo):
                memo[key] = False
                return False
        elif isinstance(va, fx.Node) or isinstance(vb, fx.Node):
            memo[key] = False
            return False
        else:
            if va != vb:
                memo[key] = False
                return False

    memo[key] = True
    return True


def dedupe_graph_outputs_pass(graph: fx.Graph) -> None:
    """
    Deduplicate identical graph outputs.

    Walks the graph output tuple, groups outputs by structural hash, confirms
    equivalence, and replaces duplicates with the canonical (first) node.
    Then runs dead code elimination to remove the now-unused duplicate subgraphs.
    """
    import torch.utils._pytree as pytree

    # Find the output node
    output_node = None
    for n in graph.nodes:
        if n.op == "output":
            output_node = n
            break

    if output_node is None:
        return

    # Get the flat list of output nodes
    output_args = pytree.arg_tree_leaves(*output_node.args, **output_node.kwargs)
    output_nodes = [n for n in output_args if isinstance(n, fx.Node)]

    if len(output_nodes) <= 1:
        return

    # Compute structural hashes for all output nodes
    hash_cache: dict[fx.Node, int] = {}
    output_hashes = [(i, n, _structural_hash(n, hash_cache)) for i, n in enumerate(output_nodes)]

    # Group by hash
    hash_groups: dict[int, list[tuple[int, fx.Node]]] = {}
    for i, n, h in output_hashes:
        if h not in hash_groups:
            hash_groups[h] = []
        hash_groups[h].append((i, n))

    # For each group with more than one member, confirm structural equality
    # and replace duplicates with a canonical node.
    #
    # The canonical node must be the EARLIEST (in graph order) member of each
    # equivalence class, not simply the first in the output tuple: duplicates
    # can have non-output users (e.g. AOT backward graphs where a BN bias-grad
    # sum also feeds the input-grad computation).  The graph is topologically
    # sorted when this pass runs (post_grad runs stable_topological_sort just
    # before it), so every user of a duplicate appears after the duplicate —
    # and therefore after the earlier canonical node.  Replacing a duplicate
    # with a LATER node instead would make earlier users reference a
    # not-yet-defined value ("used before it has been defined" lint failure).
    replacements_made = 0
    memo: dict[tuple[fx.Node, fx.Node], bool] = {}
    graph_order = {n: i for i, n in enumerate(graph.nodes)}

    for h, group in hash_groups.items():
        if len(group) <= 1:
            continue

        canonical_idx, canonical_node = group[0]

        for dup_idx, dup_node in group[1:]:
            if dup_node is canonical_node:
                # Already the same node
                continue
            if _structurally_equal(canonical_node, dup_node, memo):
                # Keep whichever node is defined earlier in graph order;
                # replace the later one with it.
                if graph_order[dup_node] < graph_order[canonical_node]:
                    canonical_node, dup_node = dup_node, canonical_node
                dup_node.replace_all_uses_with(canonical_node)
                replacements_made += 1

    if replacements_made > 0:
        graph.eliminate_dead_code()
        graph.lint()
        counters["inductor"]["dedupe_graph_outputs"] += replacements_made
