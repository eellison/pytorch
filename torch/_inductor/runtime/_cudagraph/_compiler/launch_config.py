from __future__ import annotations
from typing import Any


_DEFAULT_ATTRIBUTES = {
    "cuda.launch_cfg.cooperative", "cuda.launch_cfg.programmatic_stream_serialization_allowed",
}


def _owner(value: Any) -> Any:
    from cutlass._mlir import ir

    owner = value.owner
    return owner.operation if isinstance(owner, ir.OpView) else owner


def _constant(value: Any, width: int) -> int:
    from cutlass._mlir import ir

    owner = _owner(value)
    if (not isinstance(owner, ir.Operation) or owner.name != "arith.constant" or owner.operands
            or len(owner.results) != 1 or set(owner.attributes) != {"value"}
            or value.type != ir.IntegerType.get_signless(width)):
        raise ValueError("Expected a typed integer constant in the launch protocol")
    attribute = owner.attributes["value"]
    if not isinstance(attribute, ir.IntegerAttr) or attribute.type != value.type:
        raise ValueError("Launch constant has an incompatible attribute type")
    return attribute.value
