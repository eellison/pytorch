from __future__ import annotations

from typing import Any

from torch._inductor.runtime._cudagraph._compiler.layout_ir import LayoutIR, _queries


def _type(typ: Any) -> str:
    from torch._inductor.runtime._cudagraph._compiler.aggregate_plan import _integer_vector_parts
    from cutlass._mlir import ir
    from cutlass._mlir.dialects import llvm
    from torch._inductor.runtime._cudagraph._compiler.llvm_types import integer_array_parts

    if isinstance(typ, ir.IntegerType) and typ.is_signless:
        return "i" + str(typ.width)
    if isinstance(typ, llvm.PointerType):
        return "ptr" if typ.address_space == 0 else f"ptr addrspace({typ.address_space})"
    vector = _integer_vector_parts(typ)
    if vector is not None:
        count, element = vector
        return f"<{count} x {_type(element)}>"
    array = integer_array_parts(typ)
    if array is not None:
        count, element = array
        return f"[{count} x {_type(element)}]"
    if isinstance(typ, llvm.StructType) and not typ.opaque:
        body = "{ " + ", ".join(_type(child) for child in typ.body) + " }"
        return "<" + body + ">" if typ.packed else body
    raise ValueError(f"Unsupported LLVM query type: {typ}")


def emit_llvm_query(query: LayoutIR) -> str:
    """Keep target-dependent GEPs until LLVM object generation assigns its DataLayout."""
    from cutlass._mlir import ir

    query.check()
    with query.context, ir.Location.unknown(), ir.raw_values():
        rows, expressions = _queries(query.plan, query.aggregate_types)
        if rows != query.rows:
            raise RuntimeError("LLVM layout questions differ from the original typed requests")
        values = []
        for typ, indices in expressions:
            path = ", ".join(f"i{64 if index == 0 else 32} {value}" for index, value in enumerate(indices))
            values.append(f"  i64 ptrtoint (ptr getelementptr ({_type(typ)}, ptr null, {path}) to i64)")
    return (f'target triple = "{query.host_target}"\n'
            f"@{query.table_symbol} = constant [{len(values)} x i64] [\n" + ",\n".join(values) + "\n]\n")
