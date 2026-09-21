from __future__ import annotations

import re
from typing import Any


def integer_array_parts(typ: Any) -> tuple[int, Any] | None:
    """Read supported array syntax and verify it with the owning MLIR context."""
    from cutlass._mlir import ir

    if not isinstance(typ, ir.Type):
        raise TypeError("Expected an actual MLIR type")
    spelling = str(typ)
    if not spelling.startswith("!llvm.array<"):
        return None
    match = re.fullmatch(r"!llvm.array<([0-9]+) x (i(?:1|8|16|32|64))>", spelling)
    if match is None:
        raise ValueError("Unsupported LLVM array element type")
    count = int(match[1])
    with typ.context:
        element = ir.Type.parse(match[2])
        reconstructed = ir.Type.parse(f"!llvm.array<{count} x {element}>")
    if reconstructed != typ or not isinstance(element, ir.IntegerType) or not element.is_signless:
        raise ValueError("LLVM array syntax did not preserve its exact type")
    return count, element
