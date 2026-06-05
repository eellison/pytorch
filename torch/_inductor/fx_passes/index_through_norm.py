# mypy: allow-untyped-defs
"""
Index-through-norm pass: pushes select/index backward through LayerNorm chains.

When the selected dimension is NOT a normalization dimension, the select/index
can be pushed before the normalization, narrowing the iteration domain and
reducing memory traffic.

Identity:
    LayerNorm(x, norm_dims=[2])[select on dim=1] == LayerNorm(x[select on dim=1], norm_dims=[2])

The decomposed LayerNorm chain in post_grad looks like:
    var_mean = aten.var_mean.correction(x, [norm_dim], correction=0)
    var = getitem(var_mean, 0)
    mean = getitem(var_mean, 1)
    sub = aten.sub(x, mean)
    add_eps = aten.add(var, eps)
    rsqrt = aten.rsqrt(add_eps)
    normalized = aten.mul(sub, rsqrt)
    scaled = aten.mul(normalized, weight)
    output = aten.add(scaled, bias)
    # Then: select(output, dim, idx) or index(output, [None, idx_tensor])
"""

import copy
import logging
import operator
from collections import defaultdict
from typing import Optional

import torch
import torch.fx as fx
from torch._dynamo.utils import counters


log = logging.getLogger(__name__)
aten = torch.ops.aten


def _get_val(node: fx.Node):
    """Get the fake tensor value from node metadata."""
    return node.meta.get("val")


def _get_shape(node: fx.Node) -> Optional[list]:
    """Get shape from node metadata."""
    val = _get_val(node)
    if val is not None and hasattr(val, "shape"):
        return list(val.shape)
    return None


def _copy_meta(src_node: fx.Node, dst_node: fx.Node, new_val):
    """Copy metadata from src to dst, overriding the val."""
    dst_node.meta = copy.copy(src_node.meta)
    dst_node.meta["val"] = new_val
    if "tensor_meta" in src_node.meta and hasattr(new_val, "shape"):
        tm = copy.copy(src_node.meta["tensor_meta"])
        dst_node.meta["tensor_meta"] = tm._replace(
            shape=torch.Size(new_val.shape),
            stride=new_val.stride() if hasattr(new_val, "stride") else tm.stride,
        )


def _find_layernorm_chain(select_node: fx.Node):
    """
    Given a select/index node whose input is the output of a LayerNorm chain,
    trace backward to find the chain components.

    Returns a dict with chain nodes or None if pattern doesn't match.
    """
    # The select/index operates on the output of the chain.
    # The last op in the chain is typically add(scaled, bias) or mul(normalized, weight)
    # Walk backward through the chain.

    # Start from the input to the select/index
    if select_node.target == aten.select.int:
        chain_output = select_node.args[0]
    elif select_node.target == aten.index.Tensor:
        chain_output = select_node.args[0]
    else:
        return None

    if not isinstance(chain_output, fx.Node):
        return None

    # Try to match the full pattern working backwards from chain_output
    # Pattern: output = add(mul(mul(sub(x, mean), rsqrt(add(var, eps))), weight), bias)
    # OR:      output = mul(mul(sub(x, mean), rsqrt(add(var, eps))), weight)  (no bias)
    # OR:      output = mul(sub(x, mean), rsqrt(add(var, eps)))  (no weight/bias)

    chain = {}

    current = chain_output
    # Check if current is add(scaled, bias) - with bias
    if (current.op == "call_function" and current.target in (aten.add.Tensor, aten.add.default)):
        chain["output_add"] = current
        chain["bias_node"] = current.args[1]  # bias
        current = current.args[0]  # scaled
        if not isinstance(current, fx.Node):
            return None
    else:
        chain["output_add"] = None
        chain["bias_node"] = None

    # Check if current is mul(normalized, weight) - with weight
    if (isinstance(current, fx.Node) and current.op == "call_function"
            and current.target in (aten.mul.Tensor, aten.mul.default)):
        # One arg should be the normalized tensor, the other the weight
        # Weight is typically a 1D tensor matching hidden dim
        chain["scale_mul"] = current
        chain["weight_node"] = current.args[1]  # weight
        current = current.args[0]  # normalized = mul(sub, rsqrt)
        if not isinstance(current, fx.Node):
            return None
    else:
        chain["scale_mul"] = None
        chain["weight_node"] = None

    # current should be mul(sub, rsqrt) - the normalization multiply
    if not (isinstance(current, fx.Node) and current.op == "call_function"
            and current.target in (aten.mul.Tensor, aten.mul.default)):
        return None

    chain["norm_mul"] = current
    # args: (sub_result, rsqrt_result) — but order might vary
    arg0, arg1 = current.args[0], current.args[1]

    # Find which is sub and which is rsqrt
    sub_node = rsqrt_node = None
    for a in (arg0, arg1):
        if isinstance(a, fx.Node) and a.op == "call_function":
            if a.target in (aten.sub.Tensor, aten.sub.default):
                sub_node = a
            elif a.target == aten.rsqrt.default:
                rsqrt_node = a

    if sub_node is None or rsqrt_node is None:
        return None

    chain["sub"] = sub_node
    chain["rsqrt"] = rsqrt_node

    # rsqrt(add(var, eps))
    add_eps_node = rsqrt_node.args[0]
    if not (isinstance(add_eps_node, fx.Node) and add_eps_node.op == "call_function"
            and add_eps_node.target in (aten.add.Tensor, aten.add.default)):
        return None
    chain["add_eps"] = add_eps_node
    chain["eps"] = add_eps_node.args[1]  # epsilon value

    # add_eps.args[0] is var (from var_mean getitem 0)
    var_node = add_eps_node.args[0]
    if not isinstance(var_node, fx.Node):
        return None
    chain["var"] = var_node

    # var should be getitem(var_mean, 0)
    if not (var_node.op == "call_function" and var_node.target == operator.getitem
            and var_node.args[1] == 0):
        return None

    var_mean_node = var_node.args[0]
    if not isinstance(var_mean_node, fx.Node):
        return None

    # sub(x, mean) where mean is getitem(var_mean, 1)
    # sub.args = (x, mean)
    x_node = sub_node.args[0]
    mean_node = sub_node.args[1]
    if not isinstance(mean_node, fx.Node):
        return None

    # mean should be getitem(var_mean, 1)
    if not (mean_node.op == "call_function" and mean_node.target == operator.getitem
            and mean_node.args[1] == 1):
        return None

    mean_var_mean_node = mean_node.args[0]
    if mean_var_mean_node is not var_mean_node:
        return None

    chain["var_mean"] = var_mean_node
    chain["mean"] = mean_node
    chain["x"] = x_node

    # Verify var_mean is aten.var_mean.correction
    if not (var_mean_node.op == "call_function"
            and var_mean_node.target == aten.var_mean.correction):
        return None

    # Get the reduction dims from var_mean
    chain["norm_dims"] = var_mean_node.args[1]  # list of dims
    # Verify input to var_mean is x
    if var_mean_node.args[0] is not x_node:
        return None

    chain["chain_output"] = chain_output
    return chain


def _get_select_dim_and_index(node: fx.Node):
    """
    For a select.int or index.Tensor node, return (dim, index_node_or_value, is_index_tensor).

    For select.int: dim is an int, index is an int
    For index.Tensor: we handle [None, ..., idx_tensor, None, ...] patterns
        where exactly one entry is a tensor and the rest are None.

    Returns (dim, index, is_index_tensor) or None if not a simple pattern.
    """
    if node.target == aten.select.int:
        dim = node.args[1]
        index = node.args[2]
        return (dim, index, False)

    if node.target == aten.index.Tensor:
        indices = node.args[1]  # list of index tensors / Nones
        # Find the single non-None index
        tensor_dim = None
        tensor_idx = None
        for i, idx in enumerate(indices):
            if idx is not None:
                if tensor_dim is not None:
                    # Multiple indexed dims — not a simple select pattern
                    return None
                tensor_dim = i
                tensor_idx = idx
        if tensor_dim is None:
            return None
        return (tensor_dim, tensor_idx, True)

    return None


def _all_users_are_selects_on_same_dim(chain_output: fx.Node, select_dim: int):
    """Check that all users of the chain output are selects/indexes on the same dim."""
    for user in chain_output.users:
        info = _get_select_dim_and_index(user)
        if info is None:
            return False
        if info[0] != select_dim:
            return False
    return True


def _adjust_dim(dim: int, select_dim: int, ndim: int) -> int:
    """
    Adjust a normalization dim after a select on select_dim removes that dimension.

    If select_dim < norm_dim, norm_dim decreases by 1.
    If select_dim > norm_dim, norm_dim stays the same.
    select_dim == norm_dim is not allowed (checked before calling this).
    """
    # Normalize dims to positive
    if dim < 0:
        dim = dim + ndim
    if select_dim < 0:
        select_dim = select_dim + ndim

    if select_dim < dim:
        return dim - 1
    return dim


def index_through_norm_pass(graph: fx.Graph) -> None:
    """
    Push select/index backward through LayerNorm chains when the selected
    dimension is not a normalization dimension.
    """
    changed = False

    # Collect candidate select/index nodes
    select_nodes = []
    for node in graph.nodes:
        if node.op != "call_function":
            continue
        if node.target in (aten.select.int, aten.index.Tensor):
            select_nodes.append(node)

    # Group selects by their input (chain output)
    selects_by_input = defaultdict(list)
    for sn in select_nodes:
        inp = sn.args[0]
        if isinstance(inp, fx.Node):
            selects_by_input[inp].append(sn)

    # Process each chain output that has select users
    for chain_output, selects in selects_by_input.items():
        if not selects:
            continue

        # Get select info from the first select
        first_info = _get_select_dim_and_index(selects[0])
        if first_info is None:
            continue
        select_dim, _, _ = first_info

        # Verify all users of chain_output are selects on the same dim
        if not _all_users_are_selects_on_same_dim(chain_output, select_dim):
            continue

        # Try to match layernorm chain
        # Use the first select to find the chain
        chain = _find_layernorm_chain(selects[0])
        if chain is None:
            continue

        # Verify that select_dim is NOT in norm_dims
        norm_dims = chain["norm_dims"]
        x_node = chain["x"]
        x_shape = _get_shape(x_node)
        if x_shape is None:
            continue
        ndim = len(x_shape)

        # Normalize select_dim
        sel_dim = select_dim if select_dim >= 0 else select_dim + ndim
        # Normalize norm_dims
        normalized_norm_dims = [(d if d >= 0 else d + ndim) for d in norm_dims]

        if sel_dim in normalized_norm_dims:
            continue  # Cannot push select through norm on a reduction dim

        # Check that intermediate chain nodes are only used by other chain nodes.
        # var_mean is special: it has 2 users (getitem for var and mean), both in-chain.
        # The chain_output itself can have multiple select users.
        chain_node_set = set()
        for key in ("var_mean", "var", "mean", "sub", "add_eps", "rsqrt",
                    "norm_mul", "scale_mul", "output_add"):
            if chain.get(key) is not None:
                chain_node_set.add(chain[key])
        # chain_output is used by selects — that's fine
        chain_node_set.add(chain_output)

        # For single-user check, exclude var_mean (it legitimately has 2 in-chain users)
        # and exclude chain_output (its users are the selects we're rewriting).
        chain_nodes_to_check = [
            chain["var"], chain["mean"],
            chain["sub"], chain["add_eps"], chain["rsqrt"], chain["norm_mul"],
        ]
        if chain["scale_mul"] is not None and chain["scale_mul"] is not chain_output:
            chain_nodes_to_check.append(chain["scale_mul"])
        if chain["output_add"] is not None and chain["output_add"] is not chain_output:
            chain_nodes_to_check.append(chain["output_add"])

        skip = False
        for cn in chain_nodes_to_check:
            if cn is None:
                continue
            # All users of this node must be within the chain
            for user in cn.users:
                if user not in chain_node_set:
                    skip = True
                    break
            if skip:
                break

        # Also check var_mean: its only users should be the var and mean getitems
        if not skip:
            vm_node = chain["var_mean"]
            for user in vm_node.users:
                if user not in chain_node_set:
                    skip = True
                    break

        if skip:
            continue

        log.debug("index_through_norm: found pattern, select_dim=%d, norm_dims=%s",
                  select_dim, norm_dims)

        # For each select user, create the narrowed chain
        for select in list(chain_output.users.keys()):
            info = _get_select_dim_and_index(select)
            if info is None:
                continue
            _, index_val, is_index_tensor = info

            # Compute the narrowed x shape to determine new fake tensor vals
            x_val = _get_val(x_node)
            if x_val is None:
                continue

            # Insert new nodes before the original select
            with graph.inserting_before(select):
                # 1. Narrow x: select/index on x at select_dim
                if is_index_tensor:
                    # For index.Tensor: build the indices list for x
                    indices_for_x = [None] * ndim
                    indices_for_x[sel_dim] = index_val
                    new_x_select = graph.call_function(
                        aten.index.Tensor, (x_node, indices_for_x)
                    )
                    # Compute narrowed val
                    try:
                        narrowed_x_val = aten.index.Tensor(x_val, [
                            None if i != sel_dim else (
                                _get_val(index_val) if isinstance(index_val, fx.Node) else index_val
                            )
                            for i in range(ndim)
                        ])
                    except Exception:
                        continue
                else:
                    # For select.int
                    new_x_select = graph.call_function(
                        aten.select.int, (x_node, select_dim, index_val)
                    )
                    try:
                        narrowed_x_val = aten.select.int(x_val, select_dim, index_val)
                    except Exception:
                        continue

                _copy_meta(x_node, new_x_select, narrowed_x_val)

                # 2. Compute new norm dims (adjusted for removed dimension)
                if is_index_tensor:
                    # index.Tensor doesn't remove the dim, it narrows it
                    new_norm_dims = list(norm_dims)
                else:
                    # select.int removes the dim
                    new_norm_dims = [
                        _adjust_dim(d, sel_dim, ndim) for d in norm_dims
                    ]

                # 3. var_mean on narrowed x
                # Copy kwargs from original var_mean
                orig_vm = chain["var_mean"]
                vm_kwargs = {}
                if len(orig_vm.args) > 2:
                    vm_kwargs["correction"] = orig_vm.args[2]
                elif "correction" in (orig_vm.kwargs or {}):
                    vm_kwargs["correction"] = orig_vm.kwargs["correction"]
                else:
                    vm_kwargs["correction"] = 0
                if "keepdim" in (orig_vm.kwargs or {}):
                    vm_kwargs["keepdim"] = orig_vm.kwargs["keepdim"]
                else:
                    # Check positional args
                    if len(orig_vm.args) > 3:
                        vm_kwargs["keepdim"] = orig_vm.args[3]
                    else:
                        vm_kwargs["keepdim"] = True

                new_var_mean = graph.call_function(
                    aten.var_mean.correction,
                    (new_x_select, new_norm_dims),
                    vm_kwargs,
                )
                try:
                    narrowed_vm_val = aten.var_mean.correction(
                        narrowed_x_val, new_norm_dims, **vm_kwargs
                    )
                except Exception:
                    continue
                new_var_mean.meta = copy.copy(orig_vm.meta)
                new_var_mean.meta["val"] = narrowed_vm_val

                # 4. getitem for var (index 0) and mean (index 1)
                new_var = graph.call_function(operator.getitem, (new_var_mean, 0))
                new_var.meta = copy.copy(chain["var"].meta)
                new_var.meta["val"] = narrowed_vm_val[0]

                new_mean = graph.call_function(operator.getitem, (new_var_mean, 1))
                new_mean.meta = copy.copy(chain["mean"].meta)
                new_mean.meta["val"] = narrowed_vm_val[1]

                # 5. sub(x_narrow, mean_narrow)
                new_sub = graph.call_function(aten.sub.Tensor, (new_x_select, new_mean))
                try:
                    narrowed_sub_val = aten.sub.Tensor(narrowed_x_val, narrowed_vm_val[1])
                except Exception:
                    continue
                _copy_meta(chain["sub"], new_sub, narrowed_sub_val)

                # 6. add(var, eps)
                eps = chain["eps"]
                new_add_eps = graph.call_function(aten.add.Tensor, (new_var, eps))
                try:
                    if isinstance(eps, fx.Node):
                        eps_v = _get_val(eps)
                    else:
                        eps_v = eps
                    narrowed_add_eps_val = aten.add.Tensor(narrowed_vm_val[0], eps_v)
                except Exception:
                    continue
                _copy_meta(chain["add_eps"], new_add_eps, narrowed_add_eps_val)

                # 7. rsqrt(add_eps)
                new_rsqrt = graph.call_function(aten.rsqrt.default, (new_add_eps,))
                try:
                    narrowed_rsqrt_val = aten.rsqrt.default(narrowed_add_eps_val)
                except Exception:
                    continue
                _copy_meta(chain["rsqrt"], new_rsqrt, narrowed_rsqrt_val)

                # 8. mul(sub, rsqrt) — normalized
                new_norm_mul = graph.call_function(aten.mul.Tensor, (new_sub, new_rsqrt))
                try:
                    narrowed_norm_val = aten.mul.Tensor(narrowed_sub_val, narrowed_rsqrt_val)
                except Exception:
                    continue
                _copy_meta(chain["norm_mul"], new_norm_mul, narrowed_norm_val)

                # 9. mul(normalized, weight) if weight exists
                current_output = new_norm_mul
                current_val = narrowed_norm_val
                if chain["scale_mul"] is not None:
                    weight_node = chain["weight_node"]
                    new_scale_mul = graph.call_function(
                        aten.mul.Tensor, (current_output, weight_node)
                    )
                    try:
                        weight_val = _get_val(weight_node) if isinstance(weight_node, fx.Node) else weight_node
                        narrowed_scale_val = aten.mul.Tensor(current_val, weight_val)
                    except Exception:
                        continue
                    _copy_meta(chain["scale_mul"], new_scale_mul, narrowed_scale_val)
                    current_output = new_scale_mul
                    current_val = narrowed_scale_val

                # 10. add(scaled, bias) if bias exists
                if chain["output_add"] is not None:
                    bias_node = chain["bias_node"]
                    new_output_add = graph.call_function(
                        aten.add.Tensor, (current_output, bias_node)
                    )
                    try:
                        bias_val = _get_val(bias_node) if isinstance(bias_node, fx.Node) else bias_node
                        narrowed_output_val = aten.add.Tensor(current_val, bias_val)
                    except Exception:
                        continue
                    _copy_meta(chain["output_add"], new_output_add, narrowed_output_val)
                    current_output = new_output_add
                    current_val = narrowed_output_val

                # Replace the original select with the narrowed output
                select.replace_all_uses_with(current_output)

            changed = True

    if changed:
        # DCE the dead full-size chain
        graph.eliminate_dead_code()
        counters["inductor"]["index_through_norm"] += 1
