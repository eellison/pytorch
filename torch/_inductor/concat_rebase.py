"""Remove copies into concat storage after layouts are finalized."""

from __future__ import annotations

import dataclasses
import math
import operator
from typing import Any, TYPE_CHECKING

import sympy
from sympy import Expr

from torch.utils._ordered_set import OrderedSet

from . import config
from .ir import (
    as_storage_and_layout,
    BaseView,
    Buffer,
    ComputedBuffer,
    ConcatKernel,
    FixedLayout,
    InputsKernel,
    IRNode,
    is_storage_and_layout,
    NonOwningLayout,
    Operation,
    Pointwise,
    ReinterpretView,
    SliceView,
    StorageBox,
    TensorBox,
)
from .virtualized import V


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from .graph import GraphLowering
    from .virtualized import OpsValue


@dataclasses.dataclass(frozen=True, eq=False)
class _ConcatCopy:
    original: Callable[[Sequence[Expr]], OpsValue]
    inputs: tuple[IRNode, ...]
    dim: int

    def __call__(self, index: Sequence[Expr]) -> OpsValue:
        return self.original(index)


def _static_positive(values: Sequence[Any]) -> bool:
    return all(isinstance(v, (int, sympy.Integer)) and v > 0 for v in values)


def _known_positive(values: Sequence[Any]) -> bool:
    return all(V.graph.sizevars.statically_known_gt(v, 0) for v in values)


def _reshape_stride(
    size: Sequence[Any], stride: Sequence[Any], shape: Sequence[Any]
) -> tuple[int, ...] | None:
    from torch._subclasses.fake_impls import _compute_stride

    prefix: tuple[int, ...] = ()
    if not _static_positive((*size, *stride, *shape)):
        # Preserve the batch axis and its pitch; reshape only the static row.
        if not (
            size
            and shape
            and _static_positive((*size[1:], *stride, *shape[1:]))
            and math.prod(size[1:]) == math.prod(shape[1:])
            and V.graph.sizevars.statically_known_gt(size[0], 1)
            and V.graph.sizevars.statically_known_equals(size[0], shape[0])
        ):
            return None
        prefix = (int(stride[0]),)
        size, stride, shape = size[1:], stride[1:], shape[1:]
    result = _compute_stride(
        tuple(map(int, size)), tuple(map(int, stride)), tuple(map(int, shape))
    )
    return (*prefix, *result) if result is not None else None


def _storage_buffer(value: Any) -> Any:
    while isinstance(value, (TensorBox, StorageBox, BaseView)):
        value = value.data
    return value


def _pure_pointwise_leaf(buffer: ComputedBuffer) -> bool:
    from .ops_handler import registered_pointwise_ops

    if type(buffer.data) is not Pointwise:
        return False
    _, body, _ = buffer.get_default_sizes_body()
    pure_ops = registered_pointwise_ops | OrderedSet(
        [
            "and_",
            "load",
            "constant",
            "index_expr",
            "value_expr",
            "to_dtype_bitcast",
        ]
    )
    stores = 0
    for node in body.get_nodes():
        if node.op == "call_method":
            if node.target == "inline_asm_elementwise":
                if node.kwargs.get("is_pure", True) is not True:
                    return False
            elif node.target == "store":
                stores += 1
                mode = node.args[4] if len(node.args) > 4 else node.kwargs.get("mode")
                if node.args[1] != buffer.get_name() or mode is not None:
                    return False
            elif node.target not in pure_ops:
                return False
        elif node.op == "call_module":
            if node.target != "get_index" and node.target not in body.subblocks:
                return False
        elif node.op == "call_function":
            if node.target is not operator.getitem:
                return False
        elif node.op not in ("placeholder", "output"):
            return False
    return stores == 1


def _literal_fill(value: IRNode) -> bool:
    return (
        type(value) is Pointwise
        and not value.get_read_names()
        and value.inner_fn_opcount().used_ops <= OrderedSet(["constant", "index_expr"])
    )


def _rebase_source(value: IRNode) -> bool:
    return isinstance(value, ConcatKernel) or (
        isinstance(value, ComputedBuffer) and type(value.data) is Pointwise
    )


def _snapshot_input(value: IRNode) -> IRNode | None:
    root = _storage_buffer(value)
    if (
        _rebase_source(root)
        and type(root.layout) is FixedLayout
        and is_storage_and_layout(value)
    ):
        base, layout = as_storage_and_layout(value, freeze=False)
        if base.data is root and type(layout) is FixedLayout:
            frozen = FixedLayout(
                layout.device,
                layout.dtype,
                list(layout.size),
                list(layout.stride),
                layout.offset,
            )
            return ReinterpretView(data=StorageBox(root), layout=frozen)
    if (
        isinstance(value, TensorBox)
        and isinstance(value.data, StorageBox)
        and _literal_fill(value.data.data)
    ):
        return TensorBox(StorageBox(value.data.data))
    return None


def get_partitions(
    node: ConcatKernel, graph: GraphLowering
) -> list[tuple[Buffer, Expr, Expr]] | None:
    """Resolve recorded slices and validate their current storage ownership.

    Callers validate the iteration shapes and strides they require.
    """
    size, stride, dim = node.get_size(), node.get_stride(), node.dim
    if (
        not node.inputs
        or not 0 <= dim < len(size)
        or len(node.inputs) != len(node.slices)
    ):
        return None
    equal = graph.sizevars.statically_known_equals
    previous, result = 0, []
    for child, (start, end) in zip(node.inputs, node.slices):
        if not equal(start, previous) or not graph.sizevars.statically_known_lt(
            start, end
        ):
            return None
        if not isinstance(child, Buffer):
            return None
        child = graph.name_to_buffer.get(child.get_name())
        if child is None:
            return None
        layout = child.get_layout()
        if (
            not isinstance(layout, NonOwningLayout)
            or _storage_buffer(layout.view) is not node
        ):
            return None
        view = layout.view.get_layout()
        if (
            view.dtype != node.get_dtype()
            or view.device != node.get_device()
            or not equal(view.offset, node.get_layout().offset + start * stride[dim])
        ):
            return None
        result.append((child, start, end))
        previous = end
    return result if equal(previous, size[dim]) else None


def _rebase_plan(
    src_view: IRNode,
    dst_slice: IRNode,
    allowed_copy_ops: Sequence[Operation] = (),
    *,
    graph: GraphLowering,
    readers: dict[str, OrderedSet[str]],
) -> list[tuple[Buffer, NonOwningLayout]] | None:
    """Plan a private producer's move using complete, finalized IR.

    Exact slice geometry and a closed alias set permit changing layouts;
    only the replaced copy and virtual concat dependencies may read them.
    """
    root = _storage_buffer(src_view)
    if (
        not _rebase_source(root)
        or type(root.layout) is not FixedLayout
        or root.layout.offset != 0
        or not isinstance(dst_slice, ReinterpretView)
        or not isinstance(dst_slice.data, StorageBox)
        or not isinstance(dst_slice.data.data, ConcatKernel)
        or not is_storage_and_layout(src_view)
        or src_view.get_layout().offset != root.layout.offset
        or src_view.get_dtype() != root.get_dtype()
        or src_view.get_device() != root.get_device()
        or _reshape_stride(root.get_size(), root.get_stride(), src_view.get_size())
        != tuple(src_view.get_stride())
    ):
        return None
    plan = {}

    def subtree(node, target):
        if (
            id(node) in plan
            or graph.name_to_buffer.get(node.get_name()) is not node
            or graph.name_to_op.get(node.get_operation_name()) is not node
            or node.get_dtype() != target.get_dtype()
            or node.get_device() != target.get_device()
        ):
            return False
        strides = _reshape_stride(
            target.get_size(), target.get_stride(), node.get_size()
        )
        if strides is None:
            return False
        view = ReinterpretView(
            data=target.data,
            layout=FixedLayout(
                node.get_device(),
                node.get_dtype(),
                node.get_size(),
                list(strides),
                target.get_layout().offset,
            ),
        )
        layout = NonOwningLayout(view)
        plan[id(node)] = node, layout
        if isinstance(node, ComputedBuffer):
            return _pure_pointwise_leaf(node)
        if (
            not isinstance(node, ConcatKernel)
            or not _known_positive((*node.get_size(), *node.get_stride()))
            or (slices := get_partitions(node, graph)) is None
        ):
            return False
        size, dim = node.get_size(), node.dim
        if any(
            _reshape_stride(
                [*size[:dim], end - start, *size[dim + 1 :]],
                node.get_stride(),
                child.get_size(),
            )
            != tuple(child.get_layout().view.get_stride())
            for child, start, end in slices
        ):
            return False
        relative = ReinterpretView(data=StorageBox(node), layout=layout)
        return all(
            subtree(
                child, SliceView.create(relative, node.dim, start, end, clamp=False)
            )
            for child, start, end in slices
        )

    if not subtree(root, dst_slice):
        return None
    names = OrderedSet([node.get_name() for node, _ in plan.values()])
    if names.intersection(
        (*graph.get_output_names(), *graph.graph_inputs, *graph.mutated_buffers)
    ):
        return None
    for buffer in graph.buffers:
        if names.intersection(buffer.get_mutation_names()) or any(
            (buffer.get_name() in names) != (alias in names)
            for alias in buffer.get_inputs_that_alias_output()
        ):
            return None
    if any(names.intersection(op.get_mutation_names()) for op in graph.operations):
        return None
    allowed = OrderedSet([op.get_operation_name() for op in allowed_copy_ops])
    allowed.update(
        node.get_operation_name()
        for node, _ in plan.values()
        if isinstance(node, ConcatKernel)
    )
    if any(readers.get(name, OrderedSet()) - allowed for name in names):
        return None
    return list(plan.values())


def copy_loader(
    original: Callable[[Sequence[Expr]], OpsValue],
    inputs: Sequence[IRNode],
    dim: int,
) -> Callable[[Sequence[Expr]], OpsValue]:
    if not config.rebase_concat_copies:
        return original
    # realize_into's single-input pointwise copies already belong to a concat.
    # Preserve that owner so sibling stores can still be coalesced.
    if len(inputs) == 1 and not isinstance(_storage_buffer(inputs[0]), ConcatKernel):
        return original
    if isinstance(original, _ConcatCopy) or not any(
        _rebase_source(_storage_buffer(value)) for value in inputs
    ):
        return original
    sources = tuple(_snapshot_input(value) for value in inputs)
    if any(value is None for value in sources):
        return original
    return _ConcatCopy(original, sources, dim)


def rebase_copies(graph: GraphLowering) -> None:
    """Place private concat inputs in finalized output storage and remove copies."""
    readers: dict[str, OrderedSet[str]] | None = None
    for old in list(graph.buffers):
        if not (
            isinstance(old, ComputedBuffer)
            and type(old.data) is Pointwise
            and isinstance(old.data.inner_fn, _ConcatCopy)
            and graph.name_to_buffer.get(old.get_name()) is old
            and graph.name_to_op.get(old.get_operation_name()) is old
            and type(old.layout) in (FixedLayout, NonOwningLayout)
            and old.layout.offset == 0
            and _known_positive((*old.get_size(), *old.get_stride()))
            and old.get_name() not in graph.mutated_buffers
            and not old.get_mutation_names()
        ):
            continue
        copy = old.data.inner_fn
        shape = list(copy.inputs[0].get_size())
        if not shape or not 0 <= copy.dim < len(shape):
            continue
        shape[copy.dim] = sum(value.get_size()[copy.dim] for value in copy.inputs)
        source_names = OrderedSet(
            [
                _storage_buffer(value).get_name()
                for value in copy.inputs
                if _rebase_source(_storage_buffer(value))
            ]
        )
        if (
            list(old.data.get_size()) != shape
            or list(old.get_size()) != shape
            or old.get_dtype() != copy.inputs[0].get_dtype()
            or old.get_device() != copy.inputs[0].get_device()
            or OrderedSet([dep.name for dep in old.get_read_writes().reads])
            != source_names
        ):
            continue
        if readers is None:
            readers = {}
            for operation in graph.operations:
                for dep in operation.get_read_writes().reads:
                    readers.setdefault(dep.name, OrderedSet()).add(
                        operation.get_operation_name()
                    )
        destination = ConcatKernel(
            name=old.get_name(), layout=old.layout, inputs=[], dim=copy.dim, slices=()
        )
        destination.origins = old.origins
        destination.origin_node = old.origin_node
        destination.traceback = old.traceback
        destination.stream_idx = old.stream_idx
        destination.mempool = old.mempool
        destination._config_patches = old._config_patches
        target_box = StorageBox(destination)
        plan, writers, fills, slices = {}, [], [], []
        offset = 0
        for value in copy.inputs:
            if not _known_positive(value.get_size()):
                break
            end = offset + value.get_size()[copy.dim]
            target = SliceView.create(target_box, copy.dim, offset, end, clamp=False)
            root = _storage_buffer(value)
            if _rebase_source(root):
                part = _rebase_plan(
                    value,
                    target,
                    allowed_copy_ops=(old,),
                    graph=graph,
                    readers=readers,
                )
                if not part or any(id(buffer) in plan for buffer, _ in part):
                    break
                plan.update((id(buffer), (buffer, layout)) for buffer, layout in part)
                writers.append(root)
            elif _literal_fill(root):
                fills.append((len(writers), value, target))
                writers.append(None)
            else:
                break
            slices.append((offset, end))
            offset = end
        else:
            if not plan or offset != shape[copy.dim]:
                continue
            for buffer, layout in plan.values():
                buffer.layout = layout
                buffer.get_free_symbol_uses.clear_cache(buffer)
                if isinstance(buffer, ComputedBuffer):
                    buffer.get_default_sizes_body.clear_cache(buffer)
                else:
                    # ConcatKernel delegates to this separately cached method.
                    InputsKernel.get_free_symbol_uses.clear_cache(buffer)
            for index, value, target in fills:
                writers[index] = ConcatKernel.realize_into(value, target)
            destination.inputs = writers
            destination.slices = tuple(slices)
            destination.name = graph.register_buffer(destination)
            graph.register_operation(destination)
            graph.replace_operation_buffer(old, destination)
            # Retained computations only change write layouts. The replaced
            # copy changes read names; the new literal fillers read nothing.
            name = destination.get_operation_name()
            for source_name in source_names:
                readers[source_name].remove(name)
            for writer in writers:
                readers.setdefault(writer.get_name(), OrderedSet()).add(name)
