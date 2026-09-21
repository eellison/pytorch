from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from torch._inductor.runtime._cudagraph._compiler.aggregate_plan import AggregatePlan


LAYOUT_IR_VERSION = 1


@dataclass(frozen=True)
class LayoutRow:
    index: int
    source_arg_index: int
    llvm_arg_index: int
    aggregate_type: str
    kind: str
    path: tuple[int, ...]
    queried_type: str


def _snapshot(module: Any) -> tuple[str, bytes]:
    buffer = io.BytesIO()
    module.operation.write_bytecode(buffer)
    return str(module), buffer.getvalue()


def _aggregate_types(plan: AggregatePlan) -> tuple[Any, ...]:
    from torch._inductor.runtime._cudagraph._compiler.accessors import _function, _tagged_arguments

    program = plan.mapping.program
    host = _function(program.module, program.function_name, "llvm.func")
    arguments = _tagged_arguments(host)
    ordered = tuple(arguments)
    result = []
    for formal in plan.formals:
        if (formal.llvm_arg_index >= len(ordered) or ordered[formal.llvm_arg_index] != formal.source_arg_index
                or str(arguments[formal.source_arg_index].type) != formal.llvm_type):
            raise RuntimeError("Layout request differs from the original aggregate formal")
        result.append(arguments[formal.source_arg_index].type)
    return tuple(result)


def _queries(plan: AggregatePlan, types: tuple[Any, ...]) -> tuple[tuple[LayoutRow, ...], tuple[tuple[Any, tuple[int, ...]], ...]]:
    from cutlass._mlir import ir
    from cutlass._mlir.dialects import llvm
    from torch._inductor.runtime._cudagraph._compiler.llvm_types import integer_array_parts

    rows, expressions = [], []
    i8 = ir.IntegerType.get_signless(8)
    for formal, typ in zip(plan.formals, types):
        requests = [("allocation_size", (), typ), ("abi_alignment", (), typ)]
        for leaf in formal.leaves:
            leaf_type = typ
            for index in leaf.path:
                array = integer_array_parts(leaf_type)
                if array is not None and 0 <= index < array[0]:
                    leaf_type = array[1]
                elif (isinstance(leaf_type, llvm.StructType) and not leaf_type.opaque
                      and 0 <= index < len(leaf_type.body)):
                    leaf_type = leaf_type.body[index]
                else:
                    raise ValueError("Layout request has an invalid typed component path")
            if str(leaf_type) != leaf.llvm_type:
                raise RuntimeError("Aggregate leaf type changed before layout emission")
            requests.extend((("offset", leaf.path, leaf_type),
                             ("leaf_allocation_size", leaf.path, leaf_type),
                             ("leaf_abi_alignment", leaf.path, leaf_type)))
        for kind, path, queried_type in requests:
            if kind == "offset":
                gep_type, indices = typ, (0, *path)
            elif kind in ("abi_alignment", "leaf_abi_alignment"):
                gep_type, indices = llvm.StructType.get_literal([i8, queried_type], packed=False), (0, 1)
            else:
                gep_type, indices = queried_type, (1,)
            rows.append(LayoutRow(len(rows), formal.source_arg_index, formal.llvm_arg_index,
                                  str(typ), kind, path, str(queried_type)))
            expressions.append((gep_type, indices))
    return tuple(rows), tuple(expressions)


@dataclass(frozen=True)
class LayoutIR:
    plan: AggregatePlan = field(repr=False, compare=False)
    module: Any = field(repr=False, compare=False)
    context: Any = field(repr=False, compare=False)
    aggregate_types: tuple[Any, ...] = field(repr=False, compare=False)
    host_target: str
    table_symbol: str
    rows: tuple[LayoutRow, ...]
    text: str = field(repr=False)
    bytecode: bytes = field(repr=False)
    sha256: str
    _owners: tuple[Any, ...] = field(repr=False, compare=False)
    _fingerprint: str = field(repr=False)

    def _digest(self) -> str:
        return sha256(repr((LAYOUT_IR_VERSION, self.host_target, self.table_symbol, self.rows)).encode()).hexdigest()

    def check(self) -> None:
        from cutlass._mlir import ir

        owned = (self.plan, self.module, self.context, self.aggregate_types, self.rows)
        if (any(value is not owner for value, owner in zip(owned, self._owners))
                or self.context is not self.plan.mapping.program.context or self.module.context != self.context
                or self._digest() != self._fingerprint or sha256(self.bytecode).hexdigest() != self.sha256):
            raise RuntimeError("Layout table source ownership or schema changed")
        self.plan.check()
        with self.context, ir.Location.unknown(), ir.raw_values():
            if (_aggregate_types(self.plan) != self.aggregate_types
                    or _queries(self.plan, self.aggregate_types)[0] != self.rows
                    or _snapshot(self.module) != (self.text, self.bytecode)):
                raise RuntimeError("Layout table differs from its original typed requests")


def emit_layout_ir(plan: AggregatePlan, host_target: str, *, table_symbol: str = "cudagraph_layout") -> LayoutIR:
    """Emit compiler layout questions, without determining offsets or executing code."""
    from torch._inductor.runtime._cudagraph._compiler.aggregate_plan import AggregatePlan
    from cutlass._mlir import ir
    from cutlass._mlir.dialects import llvm

    if type(plan) is not AggregatePlan:
        raise TypeError("Expected an accepted compiler-owned AggregatePlan")
    if type(host_target) is not str or not host_target or not host_target.isascii() or any(c.isspace() for c in host_target):
        raise ValueError("Expected the explicit compiler host target")
    if type(table_symbol) is not str or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table_symbol) is None:
        raise ValueError("Expected a named external layout table")
    plan.check()
    context = plan.mapping.program.context
    with context, ir.Location.unknown(), ir.raw_values():
        types = _aggregate_types(plan)
        rows, expressions = _queries(plan, types)
        if not rows:
            raise ValueError("Layout table requires at least one tensor aggregate")
        module = ir.Module.create()
        module.operation.attributes["llvm.target_triple"] = ir.StringAttr.get(host_target)
        i64, pointer_type = ir.IntegerType.get_signless(64), llvm.PointerType.get()
        table_type = ir.Type.parse(f"!llvm.array<{len(rows)} x i64>")
        with ir.InsertionPoint(module.body):
            table = llvm.GlobalOp(table_type, table_symbol, ir.Attribute.parse("#llvm.linkage<external>"), constant=True)
        table.initializer.blocks.append()
        with ir.InsertionPoint(table.initializer.blocks[0]):
            null = llvm.ZeroOp(pointer_type).result
            values = llvm.UndefOp(table_type).result
            for row, (gep_type, indices) in zip(rows, expressions):
                address = llvm.GEPOp(pointer_type, null, [], ir.DenseI32ArrayAttr.get(indices), gep_type, []).result
                value = llvm.PtrToIntOp(i64, address).result
                values = llvm.InsertValueOp(values, value, [row.index]).result
            llvm.ReturnOp(arg=values)
        if not module.operation.verify():
            raise RuntimeError("Generated layout table failed MLIR verification")
        text, bytecode = _snapshot(module)
    plan.check()
    fingerprint = sha256(repr((LAYOUT_IR_VERSION, host_target, table_symbol, rows)).encode()).hexdigest()
    return LayoutIR(plan, module, context, types, host_target, table_symbol, rows, text, bytecode,
                    sha256(bytecode).hexdigest(), (plan, module, context, types, rows), fingerprint)
