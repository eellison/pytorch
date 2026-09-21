from __future__ import annotations
from itertools import pairwise
from typing import Any
from torch._inductor.runtime.cutedsl_parameter_analysis import _raw_value, _walk_block


def _owner(value: Any, name: str) -> Any:
    from cutlass._mlir import ir

    op = _raw_value(value).owner
    if isinstance(op, ir.OpView):
        op = op.operation
    if not isinstance(op, ir.Operation) or op.name != name:
        raise ValueError(f"Expected {name} for ABI provenance")
    return op


def _integer(value: Any) -> int:
    from cutlass._mlir import ir

    op = _owner(value, "llvm.mlir.constant")
    attr = op.attributes.get("value")
    if op.operands or not isinstance(attr, ir.IntegerAttr):
        raise ValueError("Expected an exact lowered integer constant")
    return attr.value


def _uses(value: Any) -> list[tuple[Any, int]]:
    return [(use.owner.operation, use.operand_number) for use in _raw_value(value).uses]


def _exact_uses(value: Any, expected: list[tuple[Any, int]]) -> None:
    actual = _uses(value)
    if len(actual) != len(expected) or any(item not in expected for item in actual):
        raise ValueError("Unexpected alias, escape or additional use of ABI storage")


def _ordered(*ops: Any) -> None:
    if not ops or any(op.block != ops[0].block for op in ops):
        raise ValueError("ABI allocation, initialization and use must share a block")
    body = [view.operation for view in ops[0].block.operations]
    positions = [body.index(op) for op in ops]
    if any(left >= right for left, right in pairwise(positions)):
        raise ValueError("ABI storage initialization must precede its use")


def _gep_indices(op: Any) -> tuple[int, ...]:
    from cutlass._mlir import ir

    raw = op.attributes.get("rawConstantIndices")
    if op.name != "llvm.getelementptr" or not isinstance(raw, ir.DenseI32ArrayAttr):
        raise ValueError("Expected typed constant ABI pointer indexing")
    dynamic = iter(op.operands[1:])
    indices = []
    for index in raw:
        if index == -(2**31):
            value = next(dynamic, None)
            if value is None:
                raise ValueError("Missing ABI pointer index")
            index = _integer(value)
        indices.append(index)
    if next(dynamic, None) is not None:
        raise ValueError("Unexpected ABI pointer index operands")
    return tuple(indices)


def _allocation(value: Any, count: int, element: Any) -> Any:
    op = _owner(value, "llvm.alloca")
    if (
        len(op.operands) != 1 or _integer(op.operands[0]) != count
        or op.attributes["elem_type"].value != element
    ):
        raise ValueError("Unexpected ABI allocation element type or count")
    return op


def _function(module: Any, name: str) -> Any:
    found = [view.operation for view in module.body.operations
             if view.operation.name == "llvm.func"
             and view.operation.attributes["sym_name"].value == name]
    if len(found) != 1 or len(found[0].regions) != 1 or not found[0].regions[0].blocks:
        raise ValueError(f"Expected the original lowered function {name}")
    return found[0]


def _calls(function: Any, callee: str) -> list[Any]:
    return [op for block in function.regions[0].blocks for op in _walk_block(block)
            if op.name == "llvm.call" and op.attributes.get("callee") is not None
            and op.attributes["callee"].value == callee]
